# Kernel IR to PTX Compiler

A compiler for the **Emmy Hackathon 2026** task: it takes Emmy's *Kernel IR* (a JSON tree describing what one GPU thread does) and emits **PTX** directly, skipping the CUDA C++ and `nvcc` step.

```
python compile.py path/to/kernel.ir.json > out.ptx
```

The translation from IR to PTX is written from scratch in this repository. The `emmy-ml` package is used only to *load* the IR (`Graph.from_dict`); none of Emmy's own code generation is used.

## Contents

1. [Quick start](#1-quick-start)
2. [Project structure](#2-project-structure)
3. [How it works](#3-how-it-works)
4. [Problems and categories covered](#4-problems-and-categories-covered)
5. [Supported IR statements and operations](#5-supported-ir-statements-and-operations)
6. [What is not covered](#6-what-is-not-covered)
7. [Extending the compiler](#7-extending-the-compiler)
8. [Notes](#8-notes)

---

## 1. Quick start

Requires **Python 3.12 or newer**. No GPU and no CUDA toolkit are needed to compile.

```bash
python -m venv .venv && source .venv/bin/activate
pip install emmy-ml==0.3.11 numpy requests

# compile one kernel and look at the PTX
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json

# compile it, run it on the hosted GPU runner and check the result
python submit.py p00_scalar_add --compiler compile.py
```

Run every finished problem in one go:

```bash
for p in p00_scalar_add p01_vector_add p02_scale_shift p03_relu_mul \
         r01_array_sum r02_big_sum r03_mean r04_row_sum \
         m01_matmul_512 m02_matmul_rect \
         f01_rmsnorm f02_row_softmax f03_l2norm \
         a01_attention a02_attention_128; do
  python submit.py $p --compiler compile.py
done
```
---

## 2. Project structure

```
.
├── compile.py              entry point: loads the IR, runs both phases, prints PTX
├── ptxcompiler/            the compiler package
│   ├── __init__.py
│   ├── uops.py             shared vocabulary: Uop record, ALU table, register classes, shared-memory layout
│   ├── lowering.py         phase 1: Kernel IR  ->  flat list of Uop records
│   └── render.py           phase 2: Uop records  ->  PTX text
├── problems/               the open problem set (see section 4)
│   └── <problem>/          *.ir.json (input), *.ir.txt (readable), *.ref.cu (reference CUDA),
│                           meta.json (launch geometry, argument order), expected.npz (expected outputs)
├── examples/               hand-written reference PTX from the task authors
├── docs/API.md             API of the hosted runner
├── tools/                  how the problems were generated
├── submit.py               compile + run on the runner + compare with expected outputs
├── TASK.md                 the task statement and IR reference
├── MVP.md                  the minimum bar for a valid submission
└── README.md               this file
```

| File | Responsibility |
|---|---|
| `compile.py` | Reads the IR with Emmy's `Graph`, reads `arg_order` from the `meta.json` next to the IR, collects buffer shapes, then calls `Lowerer` and `render`. |
| `ptxcompiler/uops.py` | Defines `Uop` (one abstract instruction), `REG_CLASSES` (register types), `ALU` (abstract op name to PTX mnemonic) and `SmemLayout` (shared-memory plan). |
| `ptxcompiler/lowering.py` | The `Lowerer` class: one `stmt_<Name>` method per IR statement type, plus index-expression compilation. Produces Uops, never text. |
| `ptxcompiler/render.py` | One small `r_<kind>` function per Uop kind, a `RENDER` dispatch table, and the code that writes the PTX header, parameter list, shared-memory declarations and register declarations. |

---

## 3. How it works

### The pipeline

```
kernel.ir.json  +  meta.json
          |
          v
   compile.py        load the IR (emmy Graph.from_dict), read argument order and buffer shapes
          |
          v
   Lowerer           PHASE 1  walk the IR statements, append Uop records to a flat list
          |
          v
   render()          PHASE 2  print each Uop as PTX; write header and .reg declarations last
          |
          v
      out.ptx   ->   ptxas   ->   GPU binary
```

The two phases only share the `Uop` record, so a bug is easy to place: wrong records mean phase 1, right records but wrong text mean phase 2. You can print the records with `print(*uops, sep="\n", file=sys.stderr)` after `lowerer.lower()`.

A `Uop` is a small note such as `Uop(kind="ALU", dst="%f2", srcs=("%f1", 0.0), arg="fmax")`. It holds no PTX text. `render()` turns it into `max.f32 %f2, %f1, 0f00000000;`.

### Main ideas

- **One thread per Tile cell.** The IR describes the work of one thread. The thread number is split row-major into one coordinate per Tile axis (the last axis varies fastest). A guard `if gid >= N: exit` retires extra threads.
- **A name-to-register map.** `Lowerer.env` maps IR value names to registers. Each statement looks up its inputs there and records its result. `render()` never sees IR names.
- **In-place updates.** Accumulators and loop counters are updated in the same register on each iteration (`alu(..., into=reg)`), unlike ordinary values, which get fresh registers.
- **Index expressions.** Compiled recursively with constant folding, including `+ - * / %`, comparisons and the ternary operator. Multi-dimensional buffers are flattened row-major from their shapes (`flat_index`).
- **Control flow.** Loops become labels plus conditional and unconditional branches. An `if` jumps over its body using the *opposite* comparison.
- **Reductions.** Each thread accumulates over a strided loop, a warp butterfly (`shfl.sync.bfly`) combines the 32 lanes, per-warp partial results go to shared memory, and a tree halving with barriers combines the warps.
- **Shared memory planning.** All shared arrays are laid out up front at aligned offsets. Up to 48 KiB they are static `.shared` arrays. Above that (`m01_matmul_512` uses 73,728 bytes) they share one dynamic `.extern .shared` pool, which relies on the launch passing `smem_bytes` from `meta.json`.
- **Vector loads.** A `Load` that binds 2 or 4 names (consecutive elements) becomes one `ld.global.v2/v4.f32` (or `ld.shared`) with one register per name. Other widths fall back to one scalar load per element. The vector address is assumed aligned, as in the reference CUDA.
- **Asynchronous copies.** `CpAsyncCopy` becomes `cp.async.cg.shared.global`, with `commit_group` and `wait_group` emitted in IR order.
- **Rounding that matches numpy.** Add, subtract and multiply use the `.rn` forms so `ptxas` never fuses a multiply and add into an `fma`. Pointwise results are bit-identical to the reference.
- **Registers.** The compiler uses unlimited virtual registers with one counter per register class; `ptxas` does the real allocation. The `.reg` declarations are written last because the counts are only known at the end.
- **No hardcoding.** Nothing depends on a problem's name or values. Launch geometry, argument order, shapes and extents all come from the IR and `meta.json`.

---

## 4. Problems and categories covered

The open set has 15 problems in 5 categories. "Verified on the hosted runner" means `submit.py` reported PASS on the real GPU.

| # | Category | Problem | What it computes | Status | Max abs error |
|---|---|---|---|---|---|
| 1 | Pointwise | `p00_scalar_add` | elementwise add | verified on runner | 0 |
| 1 | Pointwise | `p01_vector_add` | elementwise add, wider stores | verified on runner | 0 |
| 1 | Pointwise | `p02_scale_shift` | `c1 + c2 * x`, constants as 1-element buffers | verified on runner | 0 |
| 1 | Pointwise | `p03_relu_mul` | `y * relu(x)` | verified on runner | 0 |
| 2 | Reduction | `r01_array_sum` | sum of an array | verified on runner | 3.8e-06 |
| 2 | Reduction | `r02_big_sum` | sum over a larger array | verified on runner | 1.5e-05 |
| 2 | Reduction | `r03_mean` | mean (sum divided by a count) | verified on runner | 9.3e-10 |
| 2 | Reduction | `r04_row_sum` | sum of each row of a 2-D array | verified on runner | 5.7e-06 |
| 3 | Matmul | `m01_matmul_512` | 512x512 matmul, tiled, `cp.async` pipeline, 72 KiB shared memory | verified on runner | 8.8e-05 |
| 3 | Matmul | `m02_matmul_rect` | rectangular matmul, split-K with atomic adds | verified on runner | 1.335e-05 |
| 4 | Fused | `f01_rmsnorm` | RMS normalisation (reduction plus elementwise pass) | verified on runner | 1.9e-06 |
| 4 | Fused | `f02_row_softmax` | row softmax with online max and sum | verified on runner | 7.5e-09 |
| 4 | Fused | `f03_l2norm` | L2 normalisation | verified on runner | 1.5e-08 |
| 5 | Attention | `a01_attention` | fused matmul, softmax and reduce; vector loads, 128 registers | verified on runner | 6.6e-07 |
| 5 | Attention | `a02_attention_128` | attention-like kernel, larger head size | verified on runner | 4.8e-07 |

All 15 open problems pass on the hosted runner. The task's minimum bar (MVP) is categories 1 and 2, which are complete, and categories 3 to 5 are complete as well. The evaluation uses a hidden set of kernels of the same kinds with different shapes and operations, so the compiler is written to be general rather than tuned to the open problems. See section 6 for the known gaps that a hidden kernel could hit.

---

## 5. Supported IR statements and operations

### Statements

| IR node | Translation |
|---|---|
| `Tile` (any number of axes) | thread id, decoded into axis coordinates, with a bounds guard |
| `Load` | `ld.global.f32` / `ld.shared.f32` (one name); `ld.global.v2/v4.f32` / `ld.shared.v2/v4.f32` (2 or 4 names); scalar loads at consecutive addresses for any other count |
| `Assign` | float arithmetic, see below |
| `Let` | constant register (literal values only) |
| `Write` | `st.global.f32`, or `st.global.v2/v4.f32` for grouped values; `st.shared.f32`; atomic `red.global.add.f32` |
| `StridedLoop`, `Loop` | counted loop with labels and branches |
| `Init`, `Accum` | accumulator seeded with a literal, updated in place (`acc = op(acc, value)` or `op(base, value)`) |
| `WarpShuffle` | butterfly with `shfl.sync.bfly.b32`, offsets 16, 8, 4, 2, 1 |
| `Smem` | static array or slice of the dynamic pool |
| `Cond` | `if` with a comparison on `==`, `!=`, `<` or `>=` (no `else`) |
| `Sync` | `bar.sync` |
| `TreeHalve` | halving combine through shared memory with barriers |
| `CpAsyncCopy`, `CpAsyncCommit`, `CpAsyncWait` | `cp.async.cg.shared.global`, `commit_group`, `wait_group` |

### Float operations (`f32`)

`add`, `subtract`, `multiply`, `divide`, `maximum`, `relu`, `sqrt`, `reciprocal`, `exp`, `copy`.

`exp(x)` is computed as `ex2.approx(x * log2(e))`. Division, square root and reciprocal use the correctly rounded `.rn` forms.

### Integer index expressions

`+`, `-`, `*`, `/`, `%`, the comparisons `< <= > >= == !=`, and the ternary `cond ? a : b`.

---

## 6. What is not covered

These are known gaps. A hidden-set kernel that needs any of them will fail with a clear message on stderr.

**Data types**
- Only `f32` is supported. There is no `f16`, `bf16`, `fp8`, `f64` or integer data (as data, not as indices).

**Tensor cores and Hopper/Blackwell features**
- No `mma.sync`, `wgmma` or TMA instructions. The task marks these as not required, and no open problem uses them. Matmul is done with ordinary multiply-adds.

**Memory access**
- Vector loads (2 or 4 values) assume the address is aligned to the vector size (8 or 16 bytes). The compiler does not check this; the IR is expected to guarantee it, as the reference CUDA does.
- Stores wider than 4 values are not split into several vector stores.
- Swizzled `cp.async` layouts are not supported, and shared-memory writes with several values are not supported.
- Atomic writes work for global memory only.

**Operations**
- Missing elementwise ops include `negative`, `absolute`, `minimum`, `square`, `rsqrt`, `log`, `sigmoid`, `silu`, `tanh` and `where`. Each is a one-line addition (section 7).
- Accumulations use the supported float ops, so a min-reduction is not available until `minimum` exists.

**Control flow and structure**
- `Cond` has no `else` branch and only the four comparisons listed above.
- `Let` accepts literal values only.
- `Sync` supports only the whole-block barrier (no counts, no warp-level variant), and `TreeHalve` with an `inner` loop is not supported.
- `unroll` flags are ignored; loops are emitted as real loops.
- Warp shuffles use the full mask `0xffffffff`, so every lane of a warp must reach them. This holds for the open problems (cell counts that are multiples of 32).
- Only 1-D launches are handled (`%tid.x`, `%ctaid.x`), and the compiler assumes one kernel per IR file.
- `meta.json` must be in the same folder as the IR; there is no fallback if it is missing.

**Performance**
- No optimisation is done. Common subexpressions (such as `a0 * 2` recomputed for every address) are recomputed, loops are not unrolled, and instructions are not scheduled. The task treats speed only as a final tiebreak, and `ptxas` removes much of the redundancy itself.

**Target**
- The PTX targets `sm_89`, which the task says runs unchanged on the runner's `sm_120` GPU.

---

## 7. Extending the compiler

**Add a float operation.** Add an entry to the `ALU` table in `ptxcompiler/uops.py` if it needs a new instruction (for example `"fmin": "min.f32"`), then map the IR op name to it in the op tables in `ptxcompiler/uops.py` (for example `"minimum": "fmin"` in `BINARY_FLOAT_OPS`). No change to `render.py` is needed, because `r_alu` prints any ALU record.

**Add an IR statement type.**
1. Add a method `stmt_<NodeName>` to `Lowerer` in `lowering.py`. The method name must match the IR class name exactly, because statements are dispatched by name.
2. If it needs PTX that no existing record can express, add a new `Uop` kind (list it in the comment in `uops.py`), a matching `r_<kind>` function in `render.py`, and an entry in the `RENDER` table.
3. Remember that a statement that defines a value must store it in `self.env`.

---

## 8. Notes

- **Rules.** The task requires the IR-to-PTX lowering to be original code. This project uses Emmy only to load the IR.
- **Design credit.** The two-phase structure (a flat list of simple operations plus a separate renderer) is inspired by the PTX renderer in [tinygrad](https://github.com/tinygrad/tinygrad). No code was copied from it.
- **Tooling.** AI coding assistance was used during development, as the hackathon allows.
- **Task documents.** `TASK.md` and `MVP.md` contain the original task statement, the IR reference and the minimum bar.
