# TK-3 — Front-end convergence: Clang-derived semantics as the authority

Parent step: TK-3 in [the master](../PLAN.md#12-authoritative-execution-order).
State: PENDING; starts after TK-2 or an explicit Committee decision, under a
scoped handover. Machines: the M1 MacBook Pro, with the M4 as a second check.

## Objective and entry

Bring the toolkit into conformance with the accepted architecture (master
§3; M2E): declarations, types, address spaces, expressions, control flow and
builtins come from Clang (AST or LLVM IR), pass through an owned typed IR with
source provenance, and are lowered to MSL by the rules of `toolkit/DESIGN.md`
§2. The regex translator stops being the semantic authority; it may survive as
a lowering back-end fed by the IR, or be replaced. This is the capability gate
that M2B requires.

Entry: TK-2 report accepted or a decision to proceed without it; the Codex
importer (`tools/m2a/clang_ast.py`, `tools/m2a/native_aot.py`, 26 test
functions in `tests/m2a/test_native_aot.py`, 69 cases at `de69952`) and the
driver's existing Clang invocations (`nvcc_main.cpp:177-216`) as inputs.

## Execution order

| Child ID | Depends on | Concrete task | Acceptance |
|---|---|---|---|
| TK-3-0 | Approved handover | Decide the semantic source for v1 — Clang AST JSON (the M2A-R1 contract) or Clang LLVM IR from the driver's device pass — with a written comparison on the nine oracles and Celeritas rung 1; freeze the IR schema version. | Decision recorded in `toolkit/DESIGN.md` as a dated §3; one path chosen, the other retained as history. |
| TK-3-1 | -0 | Implement the chosen import for the kernel subset the nine oracles need; keep M2A-R1's fail-closed behaviour for unknown nodes, types and address spaces. | Portable tests pass on Linux and macOS; unknown constructs diagnose with source locations. |
| TK-3-2 | -1 | Lower the IR to MSL with the `DESIGN.md` §2 rules (ticket remap, pointer model, soft64, barriers, SIMD intrinsics, atomics, spill); emit the registration source the runtime expects. | Generated MSL for each oracle is reproducible from the IR; a diff against the regex translator's output is kept. |
| TK-3-3 | -2 | Run TK-0's harness on the M1 Pro and the M4 with emulation off. | Nine oracles classified per kernel; every pass is a Metal dispatch from the Clang-semantic path. |
| TK-3-4 | -3 | Celeritas rungs 1–2 (`corecel` collections, range, algorithms, numeric limits, stack allocator, RNG) through the new path. | Metal-dispatched and correct, or refused with a reason. |
| TK-3-5 | -4 | Retire the regex translator as authority in code: it cannot be reached without the IR or is removed; update `toolkit/DESIGN.md`. | No source-text pattern decides a kernel's semantics. |
| TK-3-6 | -5 | Report with the capability matrix and the M2B readiness statement. | Committee decides M2B. |

## Boundaries

No new CUDA language surface beyond what the nine oracles and Celeritas rungs
1–2 need; FP64 stays held (soft64 lowering exists but the double build is
M6-7); no library work. Any `asm` block remains a refusal unless a later step
designs its treatment (M6-3, R-NVIDIA).

## Acceptance

The nine oracles and Celeritas rungs 1–2 pass on Metal without emulation from
the Clang-semantic path on both machines, with per-kernel records; refusals
are diagnostics with locations. This closes the intent of M2A-R2 and opens
M2B.
