"""Phase 1: walk the Kernel IR and append Uops to a flat list."""

import sys
from math import prod

from .uops import ALU, REG_CLASSES,UNARY_FLOAT_OPS,SIMPLE_UNARY_OPS,BINARY_FLOAT_OPS,INT_BINARY,INT_COMPARE,INVERSE_COMPARE,SMEM_STATIC_LIMIT,LOG2E, SmemLayout, Uop


class Lowerer:
    def __init__(self, kernel, arg_order, shapes):
        self.kernel, self.arg_order = kernel, arg_order
        self.shapes = shapes   # buffer name -> list of dimension sizes
        self.uops = []
        self.counts = {cls: 0 for cls in REG_CLASSES}   # one counter per register class
        self.ptr = {}          # buffer name -> register holding its global address
        self.env = {}          # IR name -> register
        self.smem = {}         # shared array name -> size in bytes
        self.smem_layout = SmemLayout({}, False, 4)
        self.smem_base = {}    # shared array name -> register holding its base address
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
        """Append an ALU Uop and return the register it writes.

        By default the result goes to a fresh register. With into=reg it is
        written to an existing register instead (a variable updated in place).
        """
        if op not in ALU:
            raise KeyError(f"unknown ALU op {op!r}")
        return self.add("ALU", into or self.reg(cls), srcs, op)

    # ---- driver ----

    def lower(self):
        tile = self.kernel.body[0]
        self.plan_smem(tile.body)
        self.prologue(tile)
        self.lower_body(tile.body)
        self.add("LABEL", arg="DONE")
        return self.uops

    def find(self, stmts, kind):
        """Yield every statement of type `kind`, looking inside nested bodies."""
        for s in stmts:
            if type(s).__name__ == kind:
                yield s
            for attr in ("body", "else_body"):
                inner = getattr(s, attr, None)
                if inner and not isinstance(inner, str):
                    yield from self.find(inner, kind)

    def plan_smem(self, stmts):
        """Lay the shared arrays out one after another; above 48 KB they must share one dynamic pool."""
        arrays, top, pool_align = {}, 0, 4
        for s in self.find(stmts, "Smem"):
            if s.dtype != "float":
                sys.exit(f"compile error: shared array dtype {s.dtype} not supported yet")
            size = 4 * prod(int(getattr(e, "value", e)) for e in s.extents)
            align = max(int(s.align or 0), 4)
            top = -(-top // align) * align              # round the offset up to the alignment
            arrays[s.name] = (top, size, align)
            top += size
            pool_align = max(pool_align, align)
        self.smem_layout = SmemLayout(arrays, top > SMEM_STATIC_LIMIT, pool_align)

    def lower_body(self, stmts):
        """Lower a list of statements, one stmt_* method per node type."""
        for stmt in stmts:
            kind = type(stmt).__name__
            handler = getattr(self, "stmt_" + kind, None)
            if handler is None:
                sys.exit(f"compile error: statement {kind} not supported yet")
            handler(stmt)

    def prologue(self, tile):
        for name in self.arg_order:
            self.ptr[name] = self.add("PARAM", self.reg("rd"), arg=name)
        tid = self.add("SREG", self.reg("r"), arg="%tid.x")
        ntid = self.add("SREG", self.reg("r"), arg="%ntid.x")
        cta = self.add("SREG", self.reg("r"), arg="%ctaid.x")
        gid = self.alu("mad", "r", cta, ntid, tid)      # my thread number
        self.env["lane"] = self.alu("and", "r", tid, 31)    # position inside my warp
        self.env["warp"] = self.alu("shr", "r", tid, 5)     # which warp of the block
        extents = [int(ax.extent.value) for ax in tile.axes]
        total = 1
        for e in extents:
            total *= e
        guard = self.alu("ige", "p", gid, total)
        self.add("BRANCH", None, (guard,), "DONE")      # thread >= N cells: leave
        rest = gid                                      # split gid into one coordinate per axis;
        for ax, ext in reversed(list(zip(tile.axes, extents))[1:]):   # the last axis varies fastest
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
            return self.int_op(e.op, self.expr(e.left), self.expr(e.right))
        if kind == "TernaryExpr":
            cond, a, b = self.expr(e.cond), self.expr(e.if_true), self.expr(e.if_false)
            if isinstance(cond, int):                   # constant condition: pick a side now
                return a if cond else b
            return self.alu("isel", "r", a, b, cond)    # cond ? a : b
        sys.exit(f"compile error: index expression {kind} not supported")

    def int_op(self, op, a, b):
        """Integer arithmetic or comparison on two operands (int constants or registers)."""
        if isinstance(a, int) and isinstance(b, int):   # both constants: compute now, with C semantics
            if op in ("/", "%") and b == 0:
                sys.exit("compile error: division by zero in an index")
            q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1) if op in ("/", "%") else 0
            return {"+": a + b, "-": a - b, "*": a * b, "/": q, "%": a - b * q if b else 0,
                    "<": int(a < b), "<=": int(a <= b), ">": int(a > b), ">=": int(a >= b),
                    "==": int(a == b), "!=": int(a != b)}[op]
        if op == "+" and a == 0: return b
        if op in ("+", "-") and b == 0: return a
        if op == "*" and a == 1: return b
        if op in ("*", "/") and b == 1: return a
        if op in INT_BINARY:
            return self.alu(INT_BINARY[op], "r", a, b)
        if op in INT_COMPARE:
            return self.alu(INT_COMPARE[op], "p", a, b)
        sys.exit(f"compile error: index operator {op} not supported")

    def flat_index(self, buffer, index):
        """Row-major element number of buffer[index]; an empty index means element 0."""
        return self.flatten(buffer, [self.expr(e) for e in index])

    def flatten(self, buffer, coords):
        """Combine already-evaluated coordinates (ints or registers) with the buffer's shape."""
        if not coords:
            return 0
        shape = self.shapes[buffer]
        if len(coords) != len(shape):
            sys.exit(f"compile error: {buffer} has {len(shape)} dims but the index has {len(coords)}")
        flat = coords[0]
        for coord, dim in zip(coords[1:], shape[1:]):
            if isinstance(flat, int) and isinstance(coord, int):
                flat = flat * dim + coord
            elif isinstance(flat, int):
                flat = self.alu("iadd", "r", coord, flat * dim) if flat else coord
            else:
                flat = self.alu("mad", "r", flat, dim, coord)     # flat * dim + coord
        return flat

    def address(self, buffer, index):
        """Register holding the global address of buffer[index]; an empty index means element 0."""
        i = self.flat_index(buffer, index)
        if isinstance(i, int):                          # constant index, e.g. mul_c1[0]
            if i == 0:
                return self.ptr[buffer]                 # element 0 is the base address
            return self.alu("ptr_add", "rd", self.ptr[buffer], i * 4)
        off = self.alu("wide_imul", "rd", i, 4)         # element -> bytes
        return self.alu("ptr_add", "rd", self.ptr[buffer], off)

    def smem_address(self, buffer, index):
        """Register holding the shared-memory address of buffer[index] (index: int or register)."""
        base = self.smem_base[buffer]
        if isinstance(index, int):
            return base if index == 0 else self.alu("iadd", "r", base, index * 4)
        return self.alu("mad", "r", index, 4, base)     # base + index * 4

    # ---- statements ----

    def stmt_Load(self, s):
        """ names = load input[index]. Several names = consecutive elements = one vector load."""
        n = len(s.names)
        if n not in (1,2,4):
            sys.exit(f"Compile Error: a load of {n} values is not supported (1,2 or 4)")
        shared = s.input in self.smem_base
        if shared:                   # read from a shared array
            addr = self.smem_address(s.input, self.flat_index(s.input, s.index))
        else:
            addr = self.address(s.input, s.index)
        kind = "LD_SHARED" if shared else "LOAD"
        if n == 1:
            self.env[s.names[0]] = self.add(kind, self.reg("f"), (addr,))
            return
        regs = tuple(self.reg("f") for _ in range(n))
        self.add (kind, regs, (addr,))
        for name , r in zip(s.names, regs):
            self.env[name] = r

    def stmt_Assign(self, s):
        self.assign(s)

    def assign(self, s, into=None):
        """Lower name = op(args). With into=reg the result overwrites that register."""
        name = s.op.name
        args = [self.env[x] for x in s.args]
        if name in BINARY_FLOAT_OPS:
            dst = self.alu(BINARY_FLOAT_OPS[name], "f", *args, into=into)
        elif name in UNARY_FLOAT_OPS:
            op, const = UNARY_FLOAT_OPS[name]
            dst = self.alu(op, "f", args[0], const, into=into)   # const stays a Python float
        elif name == "exp":
            scaled = self.alu("fmul", "f", args[0], LOG2E)
            dst = self.alu("fex2", "f", scaled, into=into)
        elif name == "copy":
            dst = self.add("MOV", into or self.reg("f"), (args[0],), "f32")
        elif name in SIMPLE_UNARY_OPS:
            dst = self.alu(SIMPLE_UNARY_OPS[name], "f", args[0], into=into)
        else:
            sys.exit(f"compile error: op {name} not supported")
        self.env[s.name] = dst

    def stmt_Let(self, s):
        """name = literal constant."""
        if type(s.value).__name__ != "Literal":
            sys.exit("compile error: Let with a non-literal value not supported yet")
        self.env[s.name] = self.add("MOV", self.reg("f"), (float(s.value.value),), "f32")

    def stmt_Write(self, s):
        if s.atomic:                                    # out[index] += value, one red per value
            if s.output in self.smem_base:
                sys.exit("compile error: atomic Write to shared memory not supported yet")
            addr = self.address(s.output, s.index)
            for k, v in enumerate(s.values):
                self.add("ATOMIC", None, (addr, self.env[v]), 4 * k)
            return
        if s.output in self.smem:                       # write into a shared array
            if len(s.values) != 1:
                sys.exit("compile error: grouped shared-memory Write not supported yet")
            index = self.expr(s.index[0]) if s.index else 0
            self.add("ST_SHARED", None, (self.smem_address(s.output, index), self.env[s.values[0]]))
            return
        addr = self.address(s.output, s.index)
        self.add("STORE", None, (addr, *(self.env[v] for v in s.values)))

    def stmt_StridedLoop(self, s):
        """for a0 in start..end:step"""
        bound = int(s.axis.extent.value) if s.end is None else self.expr(s.end)
        self.loop(s, self.expr(s.start), bound, self.expr(s.step))

    def stmt_Loop(self, s):
        """for a0 in 0..extent"""
        self.loop(s, 0, int(s.axis.extent.value), 1)

    def loop(self, s, start, bound, step):
        """A counted loop: the counter is one register, updated in place."""
        for inner in s.body:                            # seed accumulators that no Init declared
            if type(inner).__name__ == "Accum" and inner.name not in self.env:
                self.env[inner.name] = self.add("MOV", self.reg("f"), (float(inner.init.value),), "f32")
        i = self.add("MOV", self.reg("r"), (start,), "u32")
        top, end = self.new_label("LOOP"), self.new_label("LOOP_END")
        self.add("LABEL", arg=top)
        done = self.alu("ige", "p", i, bound)
        self.add("BRANCH", None, (done,), end)          # counter >= bound: leave the loop
        self.env[s.axis.name] = i
        self.lower_body(s.body)
        self.alu("iadd", "r", i, step, into=i)          # counter += step
        self.add("JUMP", arg=top)
        self.add("LABEL", arg=end)

    def stmt_Init(self, s):
        """Carried state seeded with a literal; later Accums update it in place."""
        self.env[s.name] = self.add("MOV", self.reg("f"), (float(s.identity),), "f32")

    def stmt_Accum(self, s):
        """acc <- op(acc, value), written back into the accumulator's own register."""
        op = BINARY_FLOAT_OPS.get(s.op.name)
        if op is None:
            sys.exit(f"compile error: accumulate op {s.op.name} not supported")
        acc = self.env[s.name]
        left = self.env[s.base] if s.base else acc      # base: acc = op(base, value)
        self.alu(op, "f", left, self.env[s.value], into=acc)

    def stmt_WarpShuffle(self, s):
        """Butterfly across the warp: offsets length/2, ..., 2, 1; combine after each round."""
        offset = s.length // 2
        while offset >= 1:
            for name, name_b in zip(s.state, s.state_b):
                bits = self.add("MOV", self.reg("r"), (self.env[name],), "b32")   # shfl works on b32
                got = self.add("SHFL", self.reg("r"), (bits,), offset)
                self.env[name_b] = self.add("MOV", self.reg("f"), (got,), "b32")
            for c in s.combine_states:                  # state variables update in place,
                self.assign(c, into=self.env[c.name] if c.name in s.state else None)   # temporaries do not
            offset //= 2

    def stmt_Smem(self, s):
        offset, size, _ = self.smem_layout.arrays[s.name]
        self.smem[s.name] = size
        self.shapes[s.name] = [int(getattr(e, "value", e)) for e in s.extents]
        symbol = "_smem_pool" if self.smem_layout.pooled else s.name
        base = offset if self.smem_layout.pooled else 0
        self.smem_base[s.name] = self.add("SMEM_BASE", self.reg("r"), arg=(symbol, base))

    def stmt_CpAsyncCopy(self, s):
        """Copy nbytes from global memory straight into shared memory, no register in between."""
        if s.swizzle != "NONE":
            sys.exit("compile error: swizzled cp.async not supported yet")
        coords = [self.expr(e) for e in s.smem_index]
        if s.lane_index:                                # the thread's own slot is added to the base
            coords = [self.int_op("+", c, self.expr(l)) for c, l in zip(coords, s.lane_index)]
        dst = self.smem_address(s.smem, self.flatten(s.smem, coords))
        src = self.address(s.src, s.src_index)
        self.add("CP_ASYNC", None, (dst, src), s.nbytes)

    def stmt_CpAsyncCommit(self, s):
        self.add("CP_COMMIT")

    def stmt_CpAsyncWait(self, s):
        self.add("CP_WAIT", arg=s.group)

    def stmt_Cond(self, s):
        """if cond: body - jump past the body when the OPPOSITE comparison holds."""
        c = s.cond
        if s.else_body or type(c).__name__ != "BinaryExpr" or c.op not in INVERSE_COMPARE:
            sys.exit("compile error: this kind of if is not supported yet")
        skip = self.alu(INVERSE_COMPARE[c.op], "p", self.expr(c.left), self.expr(c.right))
        end = self.new_label("ENDIF")
        self.add("BRANCH", None, (skip,), end)
        self.lower_body(s.body)
        self.add("LABEL", arg=end)

    def stmt_Sync(self, s):
        if s.count is not None or s.warp:
            sys.exit("compile error: this kind of Sync is not supported yet")
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
