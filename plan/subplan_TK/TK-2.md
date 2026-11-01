# TK-2 — Celeritas 0.6.3 on the M1 Pro, with evidence under RESULTS/

Parent step: TK-2 in [the master](../PLAN.md#12-authoritative-execution-order).
State: PENDING; starts after TK-1 and a scoped handover. Machine: the M1
MacBook Pro (16 GiB).

## Objective and entry

Re-establish the M4 line's Celeritas stages S0–S3 on the development machine
with evidence the plan can accept: CPU oracles (double and float), the float
GPU build through the contained toolkit, and the device unit-test ladder with
per-launch path accounting, all recorded in the repository. Celeritas and
caloml are not edited; configure options are allowed (caloml record, "Rules
for this sample").

Entry: TK-1 accepted; the user's approval of the downloads on the metered
link — Celeritas 0.6.3 release tarball plus CLI11 2.5.0, GoogleTest 1.17.0 and
nlohmann_json 3.12.0 (3.7 MB in all, SHA-256 pinned by Celeritas), and CCCL
`v3.2.0` (`ca44ee7`, about 42 MB as a shallow clone) — or a copy of the M4's
`~/system-env/src` instead of a fetch. Geant4 is not part of this step.

## Execution order

| Child ID | Depends on | Concrete task | Acceptance |
|---|---|---|---|
| TK-2-0 | Approved handover and downloads | Place sources under `~/system-env/src/<pkg>-<ver>` with tarball hashes appended to `~/system-env/logs/source-sha256.txt`; FetchContent disconnected. | Hashes equal the pinned values; nothing else fetched. |
| TK-2-1 | -0 | S0 on the M1 Pro: native CPU Celeritas, double, same configuration as the M4's S0 (ORANGE, xorwow, CGS, Release, every optional package off). Record build and `ctest` counts under `RESULTS/celeritas/<hostname>/s0/`. | Counts recorded; the M4's 239/0/36 is the comparison, not the expectation. |
| TK-2-2 | -1 | S1 on the M1 Pro: the float CPU oracle; record the `UniformLogGridCalculatorTest.spline` tolerance failure if it reproduces. | Under `RESULTS/celeritas/<hostname>/s1/`. |
| TK-2-3 | -0 | Install CCCL 3.2.0 into the toolkit's `include/`; configure the float GPU build with `CMAKE_CUDA_COMPILER` = the toolkit's `nvcc`, `CELERITAS_USE_CUDA=ON`, `CELERITAS_REAL_TYPE=float`, `CELERITAS_BUILD_TESTS=ON`; build. | Configure and build logs under `RESULTS/celeritas/<hostname>/s2/`; translator refusals and emulation-eligible kernels listed from the driver's diagnostics. |
| TK-2-4 | -3 | S3 ladder with `CUDA4AS_TRACE=1` and emulation off: run `ctest -L gpu` and the full suite; from the traces count, per test, Metal launches, refused launches and failures. Repeat once with emulation on, recorded separately. | Under `RESULTS/celeritas/<hostname>/s3/`: per-test outcome, launch counts by path, verbatim refusals. |
| TK-2-5 | -4 | Report: what Celeritas actually runs on the Metal GPU of the M1 Pro, what is refused and why, the float-build memory use on 16 GiB, and the differences from the M4's reported 214/215 and 39/39. | The Committee can state Celeritas' TRUST stage per master §9 for this machine. |

## Boundaries and resource contract

At most 20 GiB added disk, builds with at most 6 jobs on the 8-core M1 Pro,
`ctest` runs bounded by Celeritas' own timeouts. No Geant4, ROOT, VecGeom,
HepMC3 or covfie. No patch to any source. If the float build fails to link
because of a toolkit defect, the defect is reported with the first error and
TK-2 closes negative for that stage; the fix is a new bounded decision.

## Acceptance

TK-2 closes with the dispatch-path accounting of Celeritas' device tests,
whatever the count. A test that passes only under emulation is a CPU result.
Only Metal-dispatched tests with correct results count toward M3, M5 (CCCL)
and M6-7.
