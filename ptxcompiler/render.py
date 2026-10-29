"""Phase 2: turn the flat Uop list into PTX text."""

import struct

from .uops import ALU, REG_CLASSES

PAD = "        "


def operand(x):
    """Registers are strings, ints print as-is, floats become PTX hex literals (0f + 8 hex digits)."""
    if isinstance(x, float):
        return "0f%08X" % struct.unpack(">I", struct.pack(">f", x))[0]
    return str(x)

def vec(dst):
    """ ('%f1', '%f2) -> ('.v2', '{%f1, %f2}'; a single register stays scalar."""
    if isinstance(dst, tuple):
       return f".v{len(dst)}","{" + ", ".join(dst) + "}"
    return " ", dst

def r_param(u):  return [f"ld.param.u64 {u.dst}, [param_{u.arg}];", f"cvta.to.global.u64 {u.dst}, {u.dst};"]
def r_sreg(u):   return [f"mov.u32 {u.dst}, {u.arg};"]
def r_alu(u):    return [f"{ALU[u.arg]} {u.dst}, {', '.join(map(operand, u.srcs))};"]
def r_mov(u):    return [f"mov.{u.arg} {u.dst}, {operand(u.srcs[0])};"]
def r_load(u):
    v , d = vec(u.dst)
    return [f"ld.global{v}.f32 {d}, [{u.srcs[0]}];"]
def r_branch(u): return [f"@{u.srcs[0]} bra {u.arg};"]
def r_jump(u):   return [f"bra {u.arg};"]
def r_label(u):  return [f"{u.arg}:"]
def r_ld_shared(u):
    v , d = vec(u.dst)
    return [f"ld.shared{v}.f32 {d}, [{u.srcs[0]}];"]
def r_st_shared(u):   return [f"st.shared.f32 [{u.srcs[0]}], {u.srcs[1]};"]


def r_smem_base(u):
    symbol, offset = u.arg
    lines = [f"mov.u32 {u.dst}, {symbol};"]
    return lines + ([f"add.u32 {u.dst}, {u.dst}, {offset};"] if offset else [])


def r_cp_async(u):
    kind = "cg" if u.arg == 16 else "ca"                # .cg only exists for 16-byte copies
    return [f"cp.async.{kind}.shared.global [{u.srcs[0]}], [{u.srcs[1]}], {u.arg};"]


def r_cp_commit(u):  return ["cp.async.commit_group;"]
def r_cp_wait(u):    return [f"cp.async.wait_group {u.arg};"]


def r_atomic(u):
    plus = f"+{u.arg}" if u.arg else ""
    return [f"red.global.add.f32 [{u.srcs[0]}{plus}], {u.srcs[1]};"]


def r_shfl(u):   return [f"shfl.sync.bfly.b32 {u.dst}, {u.srcs[0]}, {u.arg}, 0x1f, 0xffffffff;"]
def r_bar(u):    return [f"bar.sync {u.arg};"]


def r_store(u):
    addr, *vals = u.srcs
    if len(vals) == 1:
        return [f"st.global.f32 [{addr}], {vals[0]};"]
    return [f"st.global.v{len(vals)}.f32 [{addr}], {{{', '.join(vals)}}};"]   # grouped write


RENDER = {"PARAM": r_param, "SREG": r_sreg, "ALU": r_alu, "MOV": r_mov, "LOAD": r_load,
          "STORE": r_store, "LD_SHARED": r_ld_shared, "ST_SHARED": r_st_shared,
          "SMEM_BASE": r_smem_base, "SHFL": r_shfl, "BAR": r_bar,
          "BRANCH": r_branch, "JUMP": r_jump, "LABEL": r_label,
          "CP_ASYNC": r_cp_async, "CP_COMMIT": r_cp_commit, "CP_WAIT": r_cp_wait, "ATOMIC": r_atomic}


def render(name, arg_order, uops, counts, smem):
    body = []
    for u in uops:
        for line in RENDER[u.kind](u):
            body.append(line if u.kind == "LABEL" else PAD + line)
    params = ",\n".join(f"{PAD}.param .u64 param_{n}" for n in arg_order)
    if smem.pooled:                                     # one dynamic pool shared by every array
        shared = [f".extern .shared .align {smem.align} .b8 _smem_pool[];"]
    else:
        shared = [f".shared .align {a} .b8 {n}[{size}];" for n, (off, size, a) in smem.arrays.items()]
    regs = [f"{PAD}.reg .{ty} %{cls}<{counts[cls] + 1}>;" for cls, ty in REG_CLASSES.items()]
    head = [".version 8.0", ".target sm_89", ".address_size 64", ""] + shared + \
           ([""] if shared else []) + [f".visible .entry {name}(", params, ")", "{"]
    return "\n".join(head + regs + body + [PAD + "ret;", "}", ""])
