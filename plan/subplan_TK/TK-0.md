# TK-0 — M1 MacBook Pro bring-up and tracked re-baseline

Parent step: TK-0 in [the master](../PLAN.md#12-authoritative-execution-order).
State: NEXT_ELIGIBLE. Handover: `HANDOFF.md` Part A. Executor: the Claude Code
session the user opens on the M1 MacBook Pro. Starts on the user's word in that
session; this document is the step's contract, not a go.

## Objective and entry

Measure what the merged toolkit actually does on a second Apple chip, with
evidence that can be trusted: build it from source on `MacBookPro18,3` (Apple
M1 Pro, 14 GPU cores, 16 GiB), re-run the hardware probes there, and run the
nine unchanged oracle programs through a tracked harness that records, per
kernel, the compiler commands, the dispatch path, the raw output bytes and the
classification. Host emulation is disabled for the measurement.

Entry requires the user's kickoff on that machine. The machine's last recorded
inventory is in master §10; TK-0-0 re-checks it, and a difference is a finding,
not a blocker, unless it removes a prerequisite (Xcode, LLVM, CMake, Ninja).

## Scope and boundaries

In scope: the repository clone, `~/system-env` layout and `env.sh`, a minimal
tracked instrumentation patch (TK-0-2), the toolkit build and install, the
probes, the oracle harness and its nine runs, the report. Out of scope:
Celeritas and CCCL (TK-2), any download beyond the repository clone, Geant4,
edits to `oracle/src/` or `oracle/ref/`, edits to `plan/` (report instead),
commits and pushes unless the user asks for them in that session.

Resource and stop contract: at most 5 GiB added disk, at most 30 minutes per
oracle run, no network beyond the clone, no installation into anything that
exists, no Xcode or SDK change. Stop and report when a prerequisite is missing,
when the Homebrew compiler is not the one the driver picks up, when
`MTLCopyAllDevices()` returns no device, or when a step would need a download.

## Execution order

| Child ID | Depends on | Concrete task | Deliverable / exit check |
|---|---|---|---|
| TK-0-0 | User kickoff | Inventory re-check: `sw_vers`, `xcode-select -p`, `xcrun metal --version`, `$CUDA4AS_LLVM/bin/clang++ --version`, `cmake --version`, `ninja --version`, `python3 --version`, free disk, and the Metal device list through the tracked probe `p0_device`. Record under `RESULTS/tk0/<hostname>/env.txt`. | Every value present; differences from master §10 listed. |
| TK-0-1 | -0 | Clone `cuda4AS` directly (`--depth 1`, about 10 MB, stated to the user before fetching) at `382a66f` or later; create `~/system-env/{src,build,install,logs,cuda4AS}` if absent; write `~/system-env/cuda4AS/env.sh` with `CUDA4AS_LLVM=/opt/homebrew/opt/llvm` and the install prefix below. | `git rev-parse HEAD` recorded; no file written outside the clone and `~/system-env`. |
| TK-0-2 | -1 | Instrumentation patch, tracked and minimal, before building: (a) the host path in `Cuda4ASContext::launchKernel` runs only when `CUDA4AS_HOST_EMULATION=1`; otherwise it prints one line `cuda4as: <kernel>: no Metal pipeline; host emulation disabled` and returns an error; (b) MSL compile errors from `newLibraryWithSource` are printed to stderr; (c) with `CUDA4AS_TRACE=1` every launch prints `cuda4as: launch <kernel> path=metal|host grid=<x,y,z> block=<x,y,z>`; (d) `MTLCompileOptions.languageVersion` comes from `CUDA4AS_MSL_VERSION` when set, default unchanged (3.0). Nothing else changes. | Diff kept with the report; behaviour without the variables is unchanged except that silent emulation is gone. |
| TK-0-3 | -2 | Configure, build and install the toolkit: `cmake -B ~/system-env/build/cuda4AS-toolkit -S toolkit -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/system-env/install/cuda4AS-toolkit-0.1.0`; `ninja install`; `nvcc --version`. Logs under `~/system-env/logs/cuda4AS-toolkit/`. | Install tree present; the compilers the driver resolves (`$CUDA4AS_LLVM/bin/clang++`, host `clang++`) recorded with versions. |
| TK-0-4 | -1 | Probes: `drop/as-phase0/run.sh` on this machine (a second row for it; the 2026-09-05 row is `RESULTS/as-phase0/Studio/`), then `p9_lookback` and `p10_fp64` built from `probe/` as on the M4. Results under `RESULTS/as-phase0/<hostname>/` and `RESULTS/s1/<hostname>/`. | Every probe has a JSON with the machine block or a verbatim refusal; residency, 64-bit single-copy and acquire/release refusal re-measured on the M1 Pro. |
| TK-0-5 | -3 | Write the tracked harness `oracle/run_cuda4as_oracle.py`, modelled on `oracle/run_oracle.py` lines 96–183: per kernel, compile with the installed `nvcc` (record argv, exit, stdout, stderr), run with `CUDA4AS_TRACE=1` and without `CUDA4AS_HOST_EMULATION`, keep the raw output bytes, hash them, compare with `sha256` in `oracle/ref/<kernel>.json`, parse the dispatch path from the trace, classify per master §9, and write `RESULTS/g-c1/<hostname>/<kernel>.json` plus `<kernel>.build.log` and `<kernel>.run.log` with the machine block. | The harness reads references only from tracked JSON and never regenerates them; the M4 records in `RESULTS/g-c1/*.json` stay untouched. |
| TK-0-6 | -5 | Run the nine. Then run the nine once more with `CUDA4AS_HOST_EMULATION=1` into `RESULTS/g-c1/<hostname>/emulation/` to learn which kernels only complete on the CPU. | Two tables: GPU-only and emulation-permitted; the second never yields PASS_GPU. |
| TK-0-7 | -6 | Report `RESULTS/tk0/<hostname>/REPORT.md`: inventory differences, probe summary, per-kernel table with path and classification, differences from the M4 records, the instrumentation diff, the exact git state. Leave everything uncommitted unless the user asks otherwise; tell the user it is ready for the Committee. | The Committee can classify each of the nine for the M1 Pro from the report and the raw records alone. |

## Harness and evidence contract (inherited from master §13)

Required per kernel: the unchanged source identity (`oracle/src/<kernel>.cu`
hash), the compiler executable and version, argv and cwd of every command,
exit codes, stdout and stderr, generated MSL and the registration source when
produced, the MSL compiler's diagnostics verbatim, the dispatch path, the Metal
device identity (`MTLDevice.name`, registry ID), the raw output bytes and
their hash, the reference hash and byte count from `oracle/ref/<kernel>.json`,
wall time, and the classification with the first failing stage and dependent
NOT_RUN stages.

Adversarial checks the harness runs once: a kernel whose MSL fails to compile
must show the compiler's text and a `refused` path; a run with emulation
disabled and no pipeline must fail, not pass; a tampered reference hash must
produce a mismatch; a missing `nvcc` must produce NOT_RUN, never a synthetic
pass. A refusal is a result and is recorded with the toolkit's exact words.

## Classification rules for this step

- PASS_GPU: path `metal`, output hash equals the reference, no emulation.
- FAIL: path `metal`, output differs, or a stage failed after a Metal dispatch
  was attempted; first failing stage named.
- SKIP_UNSUPPORTED: the translator or the MSL compiler refused with a
  diagnostic (for example `mma_inline_ptx`).
- NOT_RUN: a prerequisite stage did not run; dependent stages inherit it.
- A kernel that completes only under `CUDA4AS_HOST_EMULATION=1` is recorded as
  `host-emulation` beside its GPU-only classification; it earns no pass.

## Acceptance and downstream decision

TK-0 closes with any per-kernel outcome. Acceptance requires: all nine have a
classification with a path; no silent success; references from tracked JSON
only; raw outputs and logs kept; refusal texts verbatim; the machine block in
every record; no download beyond the clone without the user's approval; no
change to `oracle/src`, `oracle/ref` or `plan/`. The Committee decides TK-1's
scope from the report; a poor result on the M1 Pro is information, not a
reason to reopen M2A-R2.
