"""Phase 1: walk the Kernel IR and append Uops to a flat list."""

from pickle import BINBYTES
import sys

from .uops import REG_CLASSES, ALU,BINARY_FLOAT_OPS,UNARY_FLOAT_OPS, Uop


class Lowerer:
    def __init__(self, kernel, arg_order):
        self.kernel, self.arg_order = kernel, arg_order
        self.uops = []
        self.counts = {cls: 0 for cls in REG_CLASSES}   # one counter per register class
        self.ptr = {}      # buffer name -> register holding its address
        self.env = {}      # IR name -> register

    # ---- helpers used everywhere ----

    def reg(self, cls):
        """A fresh register of class cls: %f1, %f2, ..."""
        self.counts[cls] += 1
        return f"%{cls}{self.counts[cls]}"

    def add(self, kind, dst=None, srcs=(), arg=None):
        """Append one Uop; return its destination register."""
        self.uops.append(Uop(kind, dst, tuple(srcs), arg))
        return dst

    def alu(self, op, cls, *srcs):
        """Append an ALU Uop writing a fresh register; return that register."""
        if op not in ALU:
            raise KeyError(f"unknown ALU op {op!r}")
        return self.add("ALU", self.reg(cls), srcs, op)

    # ---- driver ----

    def lower(self):
        tile = self.kernel.body[0]
        self.prologue(tile)
        for stmt in tile.body:                          # one method per IR node type
            getattr(self, "stmt_" + type(stmt).__name__)(stmt)
        self.add("LABEL", arg="DONE")
        return self.uops

    def prologue(self, tile):
        for name in self.arg_order:
            self.ptr[name] = self.add("PARAM", self.reg("rd"), arg=name)
        tid = self.add("SREG", self.reg("r"), arg="%tid.x")
        ntid = self.add("SREG", self.reg("r"), arg="%ntid.x")
        cta = self.add("SREG", self.reg("r"), arg="%ctaid.x")
        gid = self.alu("mad", "r", cta, ntid, tid)      # my thread number
        guard = self.alu("ige", "p", gid, int(tile.axes[0].extent.value))
        self.add("BRANCH", None, (guard,), "DONE")      # thread >= N: leave
        self.env[tile.axes[0].name] = gid

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

    def address(self, buffer, index):
        i = self.expr(index[0])
        if isinstance(i, int):  # constant index, e.g. mul_c1[0]
            if i == 0:
                return self.ptr[buffer]
            return self.alu("ptr_add", "rd", self.ptr[buffer], i * 4)
        off = self.alu("wide_imul", "rd", self.expr(index[0]), 4)   # element -> bytes
        return self.alu("ptr_add", "rd", self.ptr[buffer], off)

    # ---- statements ----

    def stmt_Load(self, s):
        dst = self.add("LOAD", self.reg("f"), (self.address(s.input, s.index),))
        self.env[s.names[0]] = dst

    def stmt_Assign(self, s):
        name = s.op.name
        args = [self.env[x] for x in s.args]
        if name in BINARY_FLOAT_OPS:
            self.env[s.name] = self.alu(BINARY_FLOAT_OPS[name], "f", *args)
        elif name in UNARY_FLOAT_OPS:
            op, const = UNARY_FLOAT_OPS[name]
            self.env[s.name] = self.alu(op, "f", args[0], const)
        else:
            sys.exit(f"Compile Error: op {name} not supported")

    def stmt_Write(self, s):
        addr = self.address(s.output, s.index)
        self.add("STORE", None, (addr, *(self.env[v] for v in s.values)))
