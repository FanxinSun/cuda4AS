# cuda4AS — a native CUDA toolchain for Apple silicon

This directory is the code half of `docs/cuda-on-apple-silicon.md`. That
document is the blueprint — envelopes, modules, contracts, gates, forks,
effort. This is where the gates get answered.

**The product is a native toolchain**: `.cu` sources and the build systems
around them go in unchanged, and the program runs on the Apple GPU — the
newest chips first — with the CUDA libraries it calls running there too. Not
a virtual machine, not container pass-through, not CUDA over a boundary. That
scope was fixed by the user on 2026-09-03 and §2.9 of the blueprint records
why, so it is not reopened by accident.

## Layout

```
README.md          this file
HANDOVER.md        who owns the project now, its rules, open items, residuals
                   and next step (from 2026-10-26: the agy session "cuda4AS")
HANDOFF.md         handoff guide: setting up and verifying cuda4AS on another
                   Apple Silicon machine
toolkit/           the cuda4AS compiler driver (nvcc), translator, clean-room
                   CUDA headers, and Metal/unified-memory runtime (libcudart)
probe/             the Swift + MSL probes -- source of truth
                   SELF_REVIEW.md says what each one leans on and how far it
                   was verified across the M1 Pro drop and the M4 Mac mini
oracle/            the phase-0 CUDA kernel corpus, its nvcc harness, the Colab
                   correctness harness (oracle/colab/), and reference outputs
drop/as-phase0/    the self-contained directory that goes to an Apple silicon
                   Mac; the user runs ./run.sh and brings back a tarball
RESULTS/           where probe, oracle (RESULTS/g-c1/), and S1 (RESULTS/s1/)
                   results are stored
samples/           real applications as end-to-end tests, each with its plan;
                   samples/caloml/ is Celeritas GPU transport (PLAN.md)
tools/             our own preparation scripts (drop data generation)
```

## Where the work happens

Since 2026-10-14 development is on an **Apple M4 Mac mini**: it builds Swift,
C++/Objective-C++, and Metal, runs the probes, translator, and `nvcc` driver,
and checks results against the oracle without leaving the machine. The Linux box
with the RTX 5080 (and the Google Colab GPU harness in `oracle/colab/` for
third-party corpus benches in `cuda-projects/`) serves as the **`nvcc`
correctness oracle only**: it produces reference outputs (`oracle/ref/` and
`RESULTS/cuda-projects/`) and nothing else.

The drop/results loop below is for the other direction: running the probes on
Apple silicon Macs we do not develop on, to collect more samples of the
hardware. Each Mac that runs a drop is one more row in `RESULTS/`.

## The drop/results loop (other Macs)

Each round trip is expensive, so the drop is built to bring back everything in
one pass:

1. Everything is written and reviewed in `probe/`, `oracle/` and `tools/`.
2. `drop/as-phase0/` is assembled by `tools/make_drop.sh`: self-contained,
   ~23 MB, no path in it points outside itself. `--package` tars it and
   `--publish` attaches it to a GitHub release.
3. It is published as a **release asset** — `gh release download as-phase0` —
   and the user unpacks it on a Mac and runs `./run.sh`. It builds and runs
   every probe, **never stops on a failure**, and tars up one results directory
   holding a JSON file per probe, every compiler message, and every run log.

   The drop travels as a release asset rather than through a clone because a
   `git clone` sends its whole pack in one unresumable shot, and 30 MB of it
   failed repeatedly on the first Mac that tried. Generated binaries — the
   drop's stream data and the oracle's reference outputs — are therefore kept
   out of git and attached to the release instead, so `git clone --depth 1` of
   this repository is a few hundred KB and works on anything.
4. The tarball comes back and is unpacked under
   `RESULTS/as-phase0/<hostname>/`.
5. The results are read, the gate verdict is given, and the next drop is
   written.

Step 3 is why the drop is built the way it is. One round trip has to bring
back *all* the errors, so every build has two routes, every runtime shader
compile is caught and its message recorded, the riskiest probe is split into
three executables, and the single most important question in the gate is asked
twice in two spellings.

## The four standing rules, in two lines each

**1. The instrument is the hardware, not the machine on the desk.** The M4 Mac
mini, the M1 Pro that ran G-C0 and any Mac a drop reaches are each one sample
of Apple silicon, exactly as the RTX 5080 is one sample of a CUDA host; the
design has to hold from an M1 with 32 KB of threadgroup memory
and no accelerators through an M5 Ultra and on chips not yet built, so every
number carries its machine and every absent capability is recorded as a
result rather than a failure.

**2. Nothing is installed into anything that exists.** The conda
environments, system Python, `/usr/local/cuda-13.2` and apt are used
read-only; `nvcc` and the RTX 5080 are the correctness oracle and nothing
else. Anything new goes somewhere new.

**3. The network may be metered.** Phase 0a downloaded nothing. Anything
larger than a few tens of MB is sized and put to the user before it is
fetched.

**4. `lmz/` is not ours.** It is read and run, and files are copied *out* of
it with provenance recorded (`drop/as-phase0/lmz/PROVENANCE.md`); it is never
edited, never committed inside, and its gitlink is never staged.

## Priority — compatibility first, speed second

Set by the user on 2026-09-05: **"Our main purpose is to make sure all CUDA
scripts/programs can smoothly run on AS, the speed is the second issue."**

That sequences the gates. *Does it build and run, unmodified* comes before
*how fast it goes*, so **G-C1** and **G-C3** lead and **G-C2** follows. The
performance work still has to happen — §2.6 defines smoothness as a table of
measurable things and "runs at all" is its first row, not its only one — but it
is not what the next months are spent on.

It also changes how a result is read. A construct that is **refused or absent**
is a correctness blocker and outranks any throughput number from the same run.

| gate | asks | status |
|---|---|---|
| **G-C0** probes | the alias, the heap ceiling, co-residency, BF16 on the accelerators | **PASSED** on an M1 Pro (2026-09-05, `RESULTS/as-phase0/Studio/`) and on the M4 Mac mini (2026-10-26, `RESULTS/as-phase0/masshjs-Mac-mini/`). Two forks taken, the BF16 question deferred to M5-class hardware |
| **G-C1** compiler | nine-kernel oracle corpus, cuda-samples, CUTLASS Ampere examples and llama.cpp `GGML_CUDA=ON` build with **zero edits** and run correctly | **in progress (top gate).** Nine-kernel oracle corpus (`RESULTS/g-c1/`, 9/9) and Celeritas 0.6.3 float build + device test suite (S2/S3, 214/215 tests, 39/39 GPU tests) **PASSED** on M4 |
| **G-C3** smoothness | drop-in `nvcc` for CMake/setuptools/PyTorch; PyTorch from source; `pip install flash-attn` | second. CMake `enable_language(CUDA)` + `find_package(CUDAToolkit)` working for Celeritas 0.6.3 |
| **G-C6** coded tier | lmz's Metal decoder verifies byte-identical | **PASSED** on the M4 Mac mini (2026-10-26, `RESULTS/as-phase0/masshjs-Mac-mini/run-p8_lmz_decoder.log`, both kernels byte-identical across 512 streams / 16.8 MB) |
| **G-C2** matrix units | a cuBLAS-ABI GEMM at ≥ 80 % of MLX on the same chip | after the compatibility gates. Needs M5-class hardware to be worth measuring |

### What G-C0 and S1 found, under this priority

Six things that **already work**, and cost the translator nothing:

- warp ↔ SIMD-group is **1:1 at 32 lanes**, so the whole `__shfl_*` /
  `__ballot_sync` / `__any` / `__all` family maps directly
- `simd_ballot` + `popcount` are exact
- `simdgroup_barrier` fences memory correctly inside a divergent branch
- `makeBuffer(bytesNoCopy:)` is coherent, so host allocations are kernel-bindable
- `atomic_float` add is native — `atomicAdd(float*)` needs no emulation
- a 1024-thread block maps 1:1 to a threadgroup

Seven that are **correctness blockers**, every one of which must be built
before a real CUDA program runs unmodified:

| found | consequence |
|---|---|
| threadgroup memory is exactly 32,768 B | a 48 KB static `__shared__` array cannot launch; §2.3's spill transform is **mandatory** |
| `atomic_ulong` has no operations at all | 64-bit atomics are **absent**, not partial; the lock-backed fallback is required |
| `double` refused: *"'double' is not supported in Metal"* | the FP64 software path or the CPU device is required |
| no independent thread scheduling — 3 lanes of 256 acquired | CUDA intra-warp spin locks **deadlock**; the translator must detect and reject them with a diagnostic |
| `vm_remap` and `mach_vm_remap` both `KERN_NO_SPACE` | `cudaMallocManaged` pointer identity must be reconstructed by compiler rebasing |
| `maxBufferLength` = 8.00 GiB = 50 % of RAM | `cudaMalloc` past that fails until the §2.7 tiers exist |
| `MTLCreateSystemDefaultDevice()` returns nil in a normal Aqua session | the runtime must never acquire its device through the system default alone |

**Nothing found is impossible.** Every blocker has a known mechanism in the
blueprint; they are work, not walls.

### What S1 (`p9_lookback` and `p10_fp64`) added on the M4 (`RESULTS/s1/`)

- **Decoupled look-back (`p9_lookback`):** M4 co-residency is 160 threadgroups (at 256 threads/group), and threadgroups above co-residency are scheduled out of `threadgroup_position_in_grid` order without preemption. Assigning `blockIdx` from a per-launch atomic arrival ticket (`atomic_fetch_add`) guarantees `ticket(A) < ticket(B)` implies `A` is already resident when `B` spins, yielding 0 timeouts at 10,240 threadgroups. `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)` provides 0 stale reads across 51.2M flag+value checks (`acquire`/`release` are rejected by MSL 3.0–3.2).
- **Double-precision emulation (`p10_fp64`):** Double-float (`df64`) reaches 191.9–226.4 GFLOP/s on M4 but suffers up to `8.72e7` ULP error on multiplication near underflow/overflow boundaries (`[1e-30, 1e30]`) and underflows to zero on Celeritas physical constants (`h^2 = 4.39e-53`). Bit-exact IEEE binary64 software emulation (`soft64`) achieves 0 ULP across all ranges (75.9 GFLOP/s add, 43.3 GFLOP/s mul, 6.2–6.3 GFLOP/s div/sqrt) and is the default `double` representation (D1). Lock-backed 64-bit `atomicAdd(double*)` with intra-SIMD-group lane serialization completes without deadlock across 65,536 threads.

`oracle/ref/` holds the bit-exact reference outputs for the nine-kernel corpus
G-C1's translator will be checked against — produced on the RTX 5080 in this
box, which is a **correctness oracle only**.

## Buy me a coffee

cuda4AS is free and open source. If the work is useful to you —

### [☕ **Buy me a coffee**](https://buymeacoffee.com/fanxinsun)
