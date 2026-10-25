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
