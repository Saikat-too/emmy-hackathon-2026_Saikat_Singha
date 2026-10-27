"""compile.py - Kernel IR -> PTX.

    python compile.py path/to/kernel.ir.json > out.ptx

    ptxc/uops.py      shared Uop record and lookup tables
    ptxc/lowering.py  phase 1: IR -> flat list of Uops
    ptxc/render.py    phase 2: Uops -> PTX text

Design idea (flat list + one renderer);
"""

import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph

from ptxcompiler.lowering import Lowerer
from ptxcompiler.render import render


def load_kernel(ir_path):
    """Return (KernelOp, arg_order) for the IR file; arg_order comes from meta.json."""
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    meta = json.loads((ir_path.parent / "meta.json").read_text())
    arg_order = next(k["arg_order"] for k in meta["kernels"] if k["entry"] == node.op.name)
    shapes = {n: [int(d.value) for d in graph.nodes[n].output.shape] for n in arg_order}
    return node.op, arg_order, shapes


def main():
    kernel, arg_order, shapes = load_kernel(Path(sys.argv[1]))
    lowerer = Lowerer(kernel, arg_order, shapes)
    uops = lowerer.lower()
    sys.stdout.write(render(kernel.name, arg_order, uops, lowerer.counts, lowerer.smem))


if __name__ == "__main__":
    main()
