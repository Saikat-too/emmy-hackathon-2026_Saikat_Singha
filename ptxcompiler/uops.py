"""Shared vocabulary of the two phases: the Uop record and the lookup tables.

The Lowerer (phase 1) writes Uops, using the keys of ALU as op names.
render() (phase 2) reads Uops, using the values of ALU as PTX mnemonics.
"""

from typing import NamedTuple

# register class -> PTX type.  Declared as  .reg .f32 %f<N>;
REG_CLASSES = {"p": "pred", "r": "b32", "rd": "b64", "f": "f32"}

# abstract ALU op -> PTX mnemonic (types and rounding live here, in one place)
ALU = {
    "iadd": "add.s32",
    "imul": "mul.lo.s32",
    "irem": "rem.s32",
    "idiv": "div.s32",
    "mad": "mad.lo.s32",
    "wide_imul": "mul.wide.s32",
    "ptr_add": "add.s64",
    "and": "and.b32",
    "shr": "shr.u32",
    "ilt": "setp.lt.s32",
    "ige": "setp.ge.s32",
    "ieq": "setp.eq.s32",
    "ine": "setp.ne.s32",
    "fadd": "add.rn.f32",
    "fmul": "mul.rn.f32",
    "fmax": "max.f32",
    "fdiv": "div.rn.f32",
}

# IR elementwise op name -> ALU op key
BINARY_FLOAT_OPS = {"add": "fadd", "multiply": "fmul", "divide": "fdiv"}

# relu(x) = max(x, 0.0)
UNARY_FLOAT_OPS = {"relu": ("fmax", 0.0)}

# jump past as an if body if (lane == 0) -> setp.ne.s32 -> jump past the body when lane not equal to 0
INVERSE_COMPARE = {"==": "ine", "!=":"ieq", "<": "ige", ">=": "ilt"}

class Uop(NamedTuple):
    kind: str            # PARAM, SREG, ALU, LOAD, STORE, BRANCH, LABEL
    dst: str | None      # register written, if any
    srcs: tuple          # registers or integer constants read
    arg: object = None   # extra info: param name, ALU op key, label, ...
