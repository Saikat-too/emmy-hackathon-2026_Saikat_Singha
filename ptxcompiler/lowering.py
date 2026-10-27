"""Phase 1: walk the Kernel IR and append Uops to a flat list."""

from pickle import BINBYTES
import sys

from .uops import INVERSE_COMPARE, LOG2E, REG_CLASSES, ALU,BINARY_FLOAT_OPS, SIMPLE_UNARY_OPS,UNARY_FLOAT_OPS, Uop


class Lowerer:
    def __init__(self, kernel, arg_order, shapes):
        self.kernel, self.arg_order = kernel, arg_order
        self.shapes = shapes
        self.uops = []
        self.counts = {cls: 0 for cls in REG_CLASSES}   # one counter per register class
        self.ptr = {}      # buffer name -> register holding its address
        self.env = {}      # IR name -> register
        self.smem = {}     # shared array name -> size in bytes
        self.smem_base = {} # shared array bname -> register holding its base address
        self.n_labels = 0

    # ---- helpers used everywhere ----

    def reg(self, cls):
        """A fresh register of class cls: %f1, %f2, ..."""
        self.counts[cls] += 1
        return f"%{cls}{self.counts[cls]}"

    def new_label(self, prefix):
        self.n_labels += 1
        return f"{prefix}_{self.n_labels}"

    def add(self, kind, dst=None, srcs=(), arg=None):
        """Append one Uop; return its destination register."""
        self.uops.append(Uop(kind, dst, tuple(srcs), arg))
        return dst

    def alu(self, op, cls, *srcs, into=None):
        """Append an ALU Uop writing a fresh register; return that register.
          By default the result goes to a fresh register. With into=reg it is written to an existing register instead
        """
        if op not in ALU:
            raise KeyError(f"unknown ALU op {op!r}")
        return self.add("ALU", into or self.reg(cls), srcs, op)

    # ---- driver ----

    def lower(self):
        tile = self.kernel.body[0]
        self.prologue(tile)
        self.lower_body(tile.body)
        self.add("LABEL", arg="DONE")
        return self.uops

    def lower_body(self, stmts):
        for stmt in stmts:
            # one method per IR node type
            kind = type(stmt).__name__
            handler = getattr(self, "stmt_" + kind, None)
            if handler is None:
                sys.exit(f"Compile Error: statment {kind} not supported yet")
            handler(stmt)

    def prologue(self, tile):
        for name in self.arg_order:
            self.ptr[name] = self.add("PARAM", self.reg("rd"), arg=name)
        tid = self.add("SREG", self.reg("r"), arg="%tid.x")
        ntid = self.add("SREG", self.reg("r"), arg="%ntid.x")
        cta = self.add("SREG", self.reg("r"), arg="%ctaid.x")
        gid = self.alu("mad", "r", cta, ntid, tid)      # my thread number
        self.env["lane"] = self.alu("and", "r", tid , 31) # position inside my warp
        self.env["warp"] = self.alu("shr", "r", tid, 5) # which warp of the block
        extents = [int(ax.extent.value) for ax in tile.axes]
        total = 1
        for e in extents:
            total *= e
        guard = self.alu("ige", "p", gid, total)
        self.add("BRANCH",  None, (guard,), "DONE")
        rest = gid
        for ax , ext in reversed(list(zip(tile.axes, extents))[1:]):
            self.env[ax.name] = self.alu("irem", "r", rest, ext)
            rest = self.alu("idiv", "r", rest, ext)
        self.env[tile.axes[0].name] = rest

    # ---- index expressions: return an int or a register ----

    def expr(self, e):
        kind = type(e).__name__
        if kind == "Literal":
            return int(e.value)
        if kind == "Var":
            return self.env[e.name]
        if kind == "BinaryExpr":
            a, b = self.expr(e.left), self.expr(e.right)
            if e.op == "+":
                if a == 0: return b
                if b == 0: return a
                return self.alu("iadd", "r", a, b)
            if e.op == "*":
                return self.alu("imul", "r", a, b)
        sys.exit(f"compile error: index expression {kind} not supported")

    def float_index(self, buffer, index):
        """ Row major element number of buffer[index]; an empty index means element 0."""
        if not index:
            return 0
        shape = self.shapes[buffer]
        if len(index) != len(shape):
            sys.exit(f"Compile Error: {buffer} has {len(shape)} dims but the index has {len(index)}")
        flat = self.expr(index[0])
        for e, dim in zip(index[1:], shape[1:]):
            coord = self.expr(e)
            if isinstance(flat, int) and isinstance(coord, int):
                flat = flat * dim + coord
            elif isinstance(flat, int):
                flat = self.alu("iadd", "r", coord, flat * dim) if flat else coord
            else:
                flat = self.alu("mad", "r", flat, dim, coord) # flat * dim + coord
        return flat

    def address(self, buffer, index):
        i = self.float_index(buffer, index)
        if isinstance(i, int):  # constant index, e.g. mul_c1[0]
            if i == 0:
                return self.ptr[buffer]
            return self.alu("ptr_add", "rd", self.ptr[buffer], i * 4)
        off = self.alu("wide_imul", "rd", i, 4)   # element -> bytes
        return self.alu("ptr_add", "rd", self.ptr[buffer], off)

    def smem_address(self, buffer, index):
        """ Registers holding the shared-memory address of buffer[index] (index: int or register)"""
        base = self.smem_base[buffer]
        if isinstance(index, int):
            return base if index == 0 else self.alu("iadd", "r", base, index*4)
        return self.alu("mad", "r", index, 4, base)

    # ---- statements ----

    def stmt_Load(self, s):
        dst = self.add("LOAD", self.reg("f"), (self.address(s.input, s.index),))
        self.env[s.names[0]] = dst

    def stmt_Assign(self,s):
        self.assign(s)

    def assign(self, s, into=None):
        name = s.op.name
        args = [self.env[x] for x in s.args]
        if name in BINARY_FLOAT_OPS:
            dst = self.alu(BINARY_FLOAT_OPS[name], "f", *args, into=into)
        elif name in UNARY_FLOAT_OPS:
            op, const = UNARY_FLOAT_OPS[name]
            dst = self.alu(op, "f", args[0], const, into=into)
        elif name == "exp":
            scaled = self.alu("fmul", "f", args[0], LOG2E)
            dst = self.alu("fex2", "f", scaled, into=into)
        elif name == "copy":
            dst = self.add("MOV", into or self.reg("f"), (args[0],), "f32")
        elif name in SIMPLE_UNARY_OPS:
            dst = self.alu(SIMPLE_UNARY_OPS[name], "f", args[0], into=into)
        else:
            sys.exit(f"Compile Error: op {name} not supported")
        self.env[s.name] = dst

    def stmt_Write(self, s):
        if s.atomic:
            sys.exit("compile error: atomic Write not supported yet")
        if s.output in self.smem:                       # write into a shared array
            if len(s.values) != 1:
                sys.exit("compile error: grouped shared-memory Write not supported yet")
            index = self.expr(s.index[0]) if s.index else 0
            self.add("ST_SHARED", None, (self.smem_address(s.output, index), self.env[s.values[0]]))
            return
        addr = self.address(s.output, s.index)
        self.add("STORE", None, (addr, *(self.env[v] for v in s.values)))

    def stmt_Let(self, s):
        """ Name = Literal Constant."""
        if type(s.value).__name__ != "Literal":
            sys.exit("Compile Error: Let with a non Literal Value Not supported Yet !")
        self.env[s.name] = self.add("MOV", self.reg("f"), (float(s.value.value),), "f32")

    def stmt_StridedLoop(self, s):
        """for a0 in start..end:step - the loop counter is one register, updated in place."""
        for inner in s.body:
            if type(inner).__name__ == "Accum":
                self.env[inner.name] = self.add("MOV", self.reg("f"), (float(inner.init.value),), "f32")
        i = self.add("MOV", self.reg("r"), (self.expr(s.start),), "u32")
        bound = int(s.axis.extent.value) if s.end is None else self.expr(s.end)
        step = self.expr(s.step)
        top , end = self.new_label("LOOP"), self.new_label("LOOP_END")
        self.add("LABEL", arg=top)
        done = self.alu("ige", "p", i, bound)
        self.add("BRANCH", None, (done,), end)
        self.env[s.axis.name] = i
        self.lower_body(s.body)
        self.alu("iadd", "r", i, step, into=i)
        self.add("JUMP", arg=top)
        self.add("LABEL", arg=end)

    def stmt_Accum(self, s):
        """ acc <- op(acc, value), written back into the accumulator's own registe"""
        op = BINARY_FLOAT_OPS.get(s.op.name)
        if op is None:
            sys.exit(f"Compile error: accumulate op {s.op.name} is not supported")
        acc = self.env[s.name]
        left = self.env[s.base] if s.base else acc
        self.alu(op, "f",left , self.env[s.value], into=acc)

    def stmt_WarpShuffle(self, s):
        """ Butterfly across the warp: offsets length/2,.., 2, 1; combine after each round"""
        offset = s.length // 2
        while offset >= 1:
            for name, name_b in zip(s.state, s.state_b):
                bits = self.add("MOV", self.reg("r"), (self.env[name],), "b32")
                got = self.add("SHFL", self.reg("r"), (bits,), offset)
                self.env[name_b] = self.add("MOV", self.reg("f"), (got,), "b32")
            for c in s.combine_states:
                self.assign(c, into=self.env[c.name] if c.name in s.state else None)
            offset //=2

    def stmt_Smem(self, s):
        if s.dtype != "float":
            sys.exit(f"Compile error: share array type {s.dtype} not supported yet")
        size = 4
        for e in s.extents:
            size *= int(getattr(e, "value", e))
        self.smem[s.name] = size
        self.smem_base[s.name] = self.add("SMEM_BASE", self.reg("r"), arg=s.name)

    def stmt_Cond(self, s):
        """ if cond: body - jump past the body when the opposite comparison holds"""
        c = s.cond
        if s.else_body or type(c).__name__ != "BinaryExpr" or c.op not in INVERSE_COMPARE:
            sys.exit("Compile Error: this kind of if is not supported yet")
        skip = self.alu(INVERSE_COMPARE[c.op], "p", self.expr(c.left), self.expr(c.right))
        end = self.new_label("ENDIF")
        self.add("BRANCH", None, (skip,), end)
        self.lower_body(s.body)
        self.add("LABEL", arg=end)

    def stmt_Sync(self, s):
        if s.count is  not None or s.warp:
            sys.exit("Compile Error: THis kind of sync is not supported yet")
        self.add("BAR", arg=s.barrier_id)

    def stmt_TreeHalve(self, s):
        """Fold shared-memory partials by halving; length is static, so the loop is unrolled here."""
        if s.inner is not None:
            sys.exit("compile error: TreeHalve with inner not supported yet")
        tid = self.env[s.tid_var]
        own = {name: self.env[name] for name in s.state}    # each thread's own accumulator registers
        step = s.length // 2
        while step >= 1:
            skip = self.alu("ige", "p", tid, step)          # only tid < step does work
            end = self.new_label("HALVE")
            self.add("BRANCH", None, (skip,), end)
            other = self.alu("iadd", "r", tid, step)
            for buf, name, name_b in zip(s.bufs, s.state, s.state_b):
                self.env[name] = self.add("LD_SHARED", self.reg("f"), (self.smem_address(buf, tid),))
                self.env[name_b] = self.add("LD_SHARED", self.reg("f"), (self.smem_address(buf, other),))
            for c in s.combine_states:
                self.assign(c, into=self.env[c.name] if c.name in s.state else None)
            for buf, name in zip(s.bufs, s.state):
                self.add("ST_SHARED", None, (self.smem_address(buf, tid), self.env[name]))
            self.add("LABEL", arg=end)
            self.add("BAR", arg=s.barrier_id)               # every thread reaches the barrier
            step //= 2
        for buf, name in zip(s.bufs, s.state):              # everyone reads the final total
            self.add("LD_SHARED", own[name], (self.smem_address(buf, 0),))
            self.env[name] = own[name]
