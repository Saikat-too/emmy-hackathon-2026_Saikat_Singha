"""Phase 2: turn the flat Uop list into PTX text."""

import struct

from .uops import ALU, REG_CLASSES

PAD = "        "

def operand(x):
    if isinstance(x, float):
        return "0f%08X" % struct.unpack(">I", struct.pack(">f", x))[0]
    return str(x);

def r_param(u):  return [f"ld.param.u64 {u.dst}, [param_{u.arg}];", f"cvta.to.global.u64 {u.dst}, {u.dst};"]
def r_sreg(u):   return [f"mov.u32 {u.dst}, {u.arg};"]
def r_alu(u):    return [f"{ALU[u.arg]} {u.dst}, {', '.join(map(operand, u.srcs))};"]
def r_load(u):   return [f"ld.global.f32 {u.dst}, [{u.srcs[0]}];"]
def r_branch(u): return [f"@{u.srcs[0]} bra {u.arg};"]
def r_label(u):  return [f"{u.arg}:"]


def r_store(u):
    addr, *vals = u.srcs
    if len(vals) == 1:
        return [f"st.global.f32 [{addr}], {vals[0]};"]
    return [f"st.global.v{len(vals)}.f32 [{addr}], {{{', '.join(vals)}}};"]   # grouped write


RENDER = {"PARAM": r_param, "SREG": r_sreg, "ALU": r_alu, "LOAD": r_load,
          "STORE": r_store, "BRANCH": r_branch, "LABEL": r_label}


def render(name, arg_order, uops, counts):
    body = []
    for u in uops:
        for line in RENDER[u.kind](u):
            body.append(line if u.kind == "LABEL" else PAD + line)
    params = ",\n".join(f"{PAD}.param .u64 param_{n}" for n in arg_order)
    regs = [f"{PAD}.reg .{ty} %{cls}<{counts[cls] + 1}>;" for cls, ty in REG_CLASSES.items()]
    head = [".version 8.0", ".target sm_89", ".address_size 64", "",
            f".visible .entry {name}(", params, ")", "{"]
    return "\n".join(head + regs + body + [PAD + "ret;", "}", ""])
