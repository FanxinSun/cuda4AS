# TK-1 — Emulation containment, diagnostics and run-time parameterisation

Parent step: TK-1 in [the master](../PLAN.md#12-authoritative-execution-order).
State: PENDING; starts after the Committee accepts the TK-0 report and issues
a scoped handover. Machine: the M1 MacBook Pro, then the M4 as a second check
when available.

## Objective and entry

Turn the merged toolkit from a seed that can silently succeed on the CPU into
a runtime and driver whose every result is attributable. Repair the baked-in
machine constants so the same binary behaves correctly on a 14-core M1 Pro, a
10-core M4 and chips not yet built (project rule 1). Re-run the nine oracles
under the contained runtime and bind each result to a Metal dispatch or an
explicit refusal.

Entry: accepted TK-0 report; the TK-0 instrumentation patch as the starting
point; no change to the accepted lowering rules of `toolkit/DESIGN.md` §2.

## Execution order

| Child ID | Depends on | Concrete task | Acceptance |
|---|---|---|---|
| TK-1-0 | Approved handover | Freeze the behaviour contract: default no emulation; `CUDA4AS_HOST_EMULATION=1` enables it with a per-launch stderr line naming the kernel; a summary at exit counts Metal launches, emulated launches and refusals. | Contract written into `toolkit/DESIGN.md` as a dated addition (§2.10) before code changes. |
| TK-1-1 | -0 | Runtime: implement the contract in `cudart.mm`; surface MSL compile errors and pipeline-creation errors with the kernel name; make `cudaLaunch` without a pipeline an error with a diagnostic. | No path returns `cudaSuccess` without a Metal dispatch unless emulation is on and announced. |
| TK-1-2 | -1 | Device constants from the device: `multiProcessorCount` from IOKit `gpu-core-count` or `MTLDevice` facts; an explicit, documented mapping for `cudaDevAttrMultiProcessorCount` (today a constant 84) and `clockRate`; `sharedMemPerBlock` from `maxThreadgroupMemoryLength`; `totalGlobalMem` from `recommendedMaxWorkingSetSize`; the spill buffer sized from the kernel's declared shared bytes, not 64 KiB per block on every launch; MSL language version chosen from the SDK with an override. | Each constant has a source or a named parameter with its derivation; the M4 values appear nowhere as literals. |
| TK-1-3 | -1 | Driver: remove the author's home path and `/usr/bin/python3` literals (resolve the translator relative to the install tree, python through `PATH` or a configured interpreter); make the `sm_80` / `__CUDA_ARCH__=800` and "CUDA 12.0" identities explicit configuration with their rationale; propagate translator refusals as compile errors with the source location. | `nvcc -v` shows every resolved tool; a refusal reads as a compiler diagnostic. |
| TK-1-4 | -1 | Translator: every regex miss or unsupported construct becomes a diagnostic with file and line instead of a silent fall-through to the stitcher. | No kernel reaches emulation without a recorded reason. |
| TK-1-5 | -2, -3, -4 | Re-run TK-0's harness on the M1 Pro; the nine results each bound to `metal` or `refused`; the emulation pass recorded separately. | Per-kernel table with no `host` path in the GPU-only run. |
| TK-1-6 | -5 | Report with diffs, the per-kernel table, the launch summary and the parameter table. | Committee decides TK-2. |

## Boundaries

No new language features, no FP64 work, no Celeritas. Emulation is a
containment question, not a feature: whether it survives at all is a user
decision (master §16). The M4's `RESULTS/g-c1/*.json` are not edited; new
records go under `RESULTS/g-c1/<hostname>/`.

## Acceptance

TK-1 closes only when no launch can succeed silently on the CPU, when the
device-dependent constants come from the device or a documented parameter, and
when the nine-oracle re-run on the M1 Pro is attributable per kernel. A kernel
that only works under emulation stays SKIP_UNSUPPORTED or FAIL on the GPU-only
table.
