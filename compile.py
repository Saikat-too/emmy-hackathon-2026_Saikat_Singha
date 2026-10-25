"""compile.py - Kernel IR -> PTX (p00 scope).

    python compile.py path/to/kernel.ir.json > out.ptx

Design in two phases (idea borrowed from tinygrad's PTX renderer; the code is ours):
    1. Lowerer  walks the IR and appends simple records ("uops") to a flat list.
    2. render() turns each record into PTX text, using small lookup tables.
Registers come in typed classes with one counter each; their .reg declarations
are written last, once the counts are known.
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple

from emmy.compiler.graph import Graph

# register class -> (name prefix, PTX type).  Declared as  .reg .f32 %f<N>;
REG_CLASSES = {"p": "pred", "r": "b32", "rd": "b64", "f": "f32"}

# abstract ALU op -> PTX mnemonic (types and rounding live here, in one place)
ALU = {
    "iadd": "add.s32",
    "imul": "mul.lo.s32",
    "mad": "mad.lo.s32",
    "wide_imul": "mul.wide.s32",
    "ptr_add": "add.s64",
    "ige": "setp.ge.s32",
    "fadd": "add.rn.f32",
}


class Uop(NamedTuple):
    kind: str            # PARAM, SREG, ALU, LOAD, STORE, BRANCH, LABEL
    dst: str | None      # register written, if any
    srcs: tuple          # registers or integer constants read
    arg: object = None   # extra info: param name, ALU op key, label, ...


# ---------------------------------------------------------------- phase 1: IR -> uops

class Lowerer:
    def __init__(self, kernel, arg_order):
        self.kernel, self.arg_order = kernel, arg_order
        self.uops = []
        self.counts = {cls: 0 for cls in REG_CLASSES}
        self.ptr = {}      # buffer name -> register holding its address
        self.env = {}      # IR name -> register

    def reg(self, cls):
        self.counts[cls] += 1
        return f"%{cls}{self.counts[cls]}"

    def add(self, kind, dst=None, srcs=(), arg=None):
        self.uops.append(Uop(kind, dst, tuple(srcs), arg))
        return dst

    def alu(self, op, cls, *srcs):
        return self.add("ALU", self.reg(cls), srcs, op)

    def lower(self):
        tile = self.kernel.body[0]
        self.prologue(tile)
        for stmt in tile.body:
            getattr(self, "stmt_" + type(stmt).__name__)(stmt)
        self.add("LABEL", arg="DONE")
        return self.uops

    def prologue(self, tile):
        for name in self.arg_order:
            self.ptr[name] = self.add("PARAM", self.reg("rd"), arg=name)
        tid = self.add("SREG", self.reg("r"), arg="%tid.x")
        ntid = self.add("SREG", self.reg("r"), arg="%ntid.x")
        cta = self.add("SREG", self.reg("r"), arg="%ctaid.x")
        gid = self.alu("mad", "r", cta, ntid, tid)                  # my thread number
        guard = self.alu("ige", "p", gid, int(tile.axes[0].extent.value))
        self.add("BRANCH", None, (guard,), "DONE")                  # thread >= N: leave
        self.env[tile.axes[0].name] = gid

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
        off = self.alu("wide_imul", "rd", self.expr(index[0]), 4)    # element -> bytes
        return self.alu("ptr_add", "rd", self.ptr[buffer], off)

    def stmt_Load(self, s):
        dst = self.add("LOAD", self.reg("f"), (self.address(s.input, s.index),))
        self.env[s.names[0]] = dst

    def stmt_Assign(self, s):
        if s.op.name != "add":
            sys.exit(f"compile error: op {s.op.name} not supported")
        a, b = (self.env[x] for x in s.args)
        self.env[s.name] = self.alu("fadd", "f", a, b)

    def stmt_Write(self, s):
        addr = self.address(s.output, s.index)
        self.add("STORE", None, (addr, *(self.env[v] for v in s.values)))


# ---------------------------------------------------------------- phase 2: uops -> text

def r_param(u):  return [f"ld.param.u64 {u.dst}, [param_{u.arg}];", f"cvta.to.global.u64 {u.dst}, {u.dst};"]
def r_sreg(u):   return [f"mov.u32 {u.dst}, {u.arg};"]
def r_alu(u):    return [f"{ALU[u.arg]} {u.dst}, {', '.join(map(str, u.srcs))};"]
def r_load(u):   return [f"ld.global.f32 {u.dst}, [{u.srcs[0]}];"]
def r_branch(u): return [f"@{u.srcs[0]} bra {u.arg};"]
def r_label(u):  return [f"{u.arg}:"]

def r_store(u):
    addr, *vals = u.srcs
    vec = f".v{len(vals)}" if len(vals) > 1 else ""
    return [f"st.global{vec}.f32 [{addr}], {{{', '.join(vals)}}};" if vec
            else f"st.global.f32 [{addr}], {vals[0]};"]

RENDER = {"PARAM": r_param, "SREG": r_sreg, "ALU": r_alu, "LOAD": r_load,
          "STORE": r_store, "BRANCH": r_branch, "LABEL": r_label}


def render(name, arg_order, uops, counts):
    pad = "        "
    body = []
    for u in uops:
        for line in RENDER[u.kind](u):
            body.append(line if u.kind == "LABEL" else pad + line)
    params = ",\n".join(f"{pad}.param .u64 param_{n}" for n in arg_order)
    regs = [f"{pad}.reg .{ty} %{cls}<{counts[cls] + 1}>;" for cls, ty in REG_CLASSES.items()]
    head = [".version 8.0", ".target sm_89", ".address_size 64", "",
            f".visible .entry {name}(", params, ")", "{"]
    return "\n".join(head + regs + body + [pad + "ret;", "}", ""])


# ---------------------------------------------------------------- glue

def main():
    ir_path = Path(sys.argv[1])
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    meta = json.loads((ir_path.parent / "meta.json").read_text())
    arg_order = next(k["arg_order"] for k in meta["kernels"] if k["entry"] == node.op.name)

    lowerer = Lowerer(node.op, arg_order)
    uops = lowerer.lower()
    sys.stdout.write(render(node.op.name, arg_order, uops, lowerer.counts))


if __name__ == "__main__":
    main()
