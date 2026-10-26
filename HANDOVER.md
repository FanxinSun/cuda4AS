# cuda4AS handover: Committee → the agy session "cuda4AS"

Written 2026-10-26 by the Claude Code session "cuda4AS Committee", on the user's instruction: *"hand over
the project to the same agy session, but renamed to cuda4AS, who will handle the project instead of
you; with explicit open items, residuals, next-step"*.

**From 2026-10-26, the Antigravity CLI session named `cuda4AS` owns the cuda4AS project.** That covers
planning, gate tracking, decisions put to the user, and execution. The session is conversation
`c39c7a89-4c79-47d1-a19b-2939180473ff`, running in the Terminal.app tab titled "cuda4AS" (it was
"cuda4AS executor"). The work earmarked for the "projects converter" session (Phase 2-2 onward: our
runtime and translator) is yours too, because that session was never started. The Committee steps
back and makes no further changes unless the user asks it to.

Read in this order before acting:

1. this file
2. `README.md`
3. `samples/caloml/PLAN.md`: Decisions, the three Committee reviews, and "S1 closed"
4. `probe/SELF_REVIEW.md`
5. `oracle/ref/README.md`
6. `RESULTS/cuda-projects/READY.md` and `RETEST.md`

Keep sections 5–7 below current as the project's register. Close an item with a date and a pointer;
don't delete it.

---

## 1. Standing rules: the user's, non-negotiable

Your tool calls are auto-approved (`agy --dangerously-skip-permissions`), so these rules are the only
guard.

1. **Never commit, push or stage, anywhere.** That includes cuda4AS, the parent `~/YFCE`, and every
   other repo. The user, verbatim: *"Leave the commit and push to me."* End each report with the list
   of uncommitted changes.
2. **Every environment download, build and install goes into `~/system-env`.** The user, 2026-10-21:
   *"locate all env downloading and setup into ~/system-env"*. Use its existing layout:
   - `src/<pkg>-<ver>`, plus the tarball, with its sha256 appended to `logs/source-sha256.txt`
   - `build/<pkg>-<ver>[-variant]`
   - `install/<pkg>-<ver>[-variant]`
   - `logs/<pkg>-<ver>[-variant]/`
   - cuda4AS scripts and env in `~/system-env/cuda4AS/` (`env.sh`, `build-celeritas-cpu.sh`)

   Check that a path is free before writing to it. Keep builds offline: pre-fetch pinned
   dependencies and point CMake FetchContent at them, with `FETCHCONTENT_FULLY_DISCONNECTED=ON`.
   Don't use Homebrew for anything new. Other projects share `~/system-env` (n2-recast, lmz,
   lmsluice, customize-llm; its HepMC3, fastjet and pythia8 are theirs), so touch only what's ours.
3. **Install only what the current step needs.** The user: *"install Celeritas first ONLY, if other
   tools like Geant4 is not a must for it, leave them for the following steps but not now"*. They
   stopped the Committee from even sizing the Geant4 datasets. Geant4, ROOT, VecGeom, HepMC3 and
   covfie wait for their stages (caloml S4, S7).
4. **The network may be metered** (README rule 3). Anything larger than a few tens of MB is sized
   and put to the user before you fetch it.
5. **Nothing is installed into anything that exists** (README rule 2). That covers system Python,
   conda environments, apt and `/usr/local/cuda-13.2` on the Linux box, and the existing Homebrew
   kegs.
6. **These are read-only inputs.** Never edit them, and never commit or stage inside them:
   - **`lmz/`** (README rule 4). Files may be copied *out* of it, with provenance recorded. Its
     gitlink in YFCE shows as modified; it isn't ours, so leave it alone.
   - **`~/caloml`**, the user's own repo. It has its own rule: *no external pushes, PRs, uploads,
     emails or registrations*.
   - **The Celeritas sources.** Configure options are allowed; patches are not.
   - **`cuda-projects/`**, owned by the Claude session "cuda projects".
7. **Compatibility first, speed second.** The user, 2026-09-05: *"Our main purpose is to make sure all
   CUDA scripts/programs can smoothly run on AS, the speed is the second issue."* A construct that is
   refused or absent outranks any throughput number from the same run.
8. **The instrument is the hardware** (README rule 1). Every number carries its machine. An absent
   capability is a *result*, recorded with the compiler's or runtime's exact words; it isn't a
   failure.
9. **Homebrew `llvm` 23.1.3 and `ninja` predate rule 2.** `llvm` is at `/opt/homebrew/opt/llvm`, with
   NVPTX, and is not on PATH; `env.sh` exports it as `CUDA4AS_LLVM`. Don't move either without the
   user's answer (open item O2).

### Before you report: the S1 lessons

The Committee sent back two of the three S1 hand-offs. Run these checks on your own results before
anything goes to the user:

- **Every number in a report must appear in a results JSON.** Where the JSON has `null`, write
  "not measured". Don't summarize in friendlier numbers than the data shows: S1 claimed "~32 ulp"
  where the JSON said 1.8e7.
- **Sanity-check against physics.** df64 running at native FP32 speed was impossible: the benchmark
  was constant-folded and latency-bound.
- **Make sure a test can tell hardware behaviour from a harness artifact.** The first fenced
  flag/value test had many writers per slot, so it failed on any memory model.
- **Choose references that match CUDA's semantics.** A double `atomicAdd` rounds after every add in
  arrival order, so compare it against a sequential sum in that order, not against Kahan.
- **Cite upstream behaviour by file and line.** The S1 brief wrongly said CUB takes tickets; CUB's
  tile index is `blockIdx.x` (`cub/agent/agent_scan.cuh:424`).
- **Say exactly what the git state is.** "Untracked" is not "clean".

## 2. The machine

- **This Mac (development and target host):** Apple M4 Mac mini, 24 GB, macOS 15.4.1 (24E263).
  - Xcode 16.3 is selected: `metal` 32023.619, Swift 6.1. Metal 3, GPU families up to apple9, MSL up
    to 3.2. There's no Metal 4, so no `metal_tensor`.
  - Measured (`RESULTS/s1/p9_lookback.json`, machine block and P9):
    - 10 GPU cores; threadgroup memory 32 KiB
    - `maxBufferLength` 12 GiB (50 % of RAM); recommended working set 16 GiB
    - `MTLCreateSystemDefaultDevice()` is non-nil in an Aqua session
    - live threadgroups: 127, 84 and 46 at 128, 256 and 512 threads
- **The Linux box with the RTX 5080** (sm_120, nvcc 13.2) is the **nvcc correctness oracle only**. It
  produced `oracle/ref/` and caloml's GPU reference runs.
- **Colab T4 (sm_75) and A100 (sm_80)** are the oracle for `cuda-projects/`, run by the "cuda projects"
  session.

## 3. Where the project stands, 2026-10-26

| gate | status |
|---|---|
| **G-C0** probes | **PASSED** on an M1 Pro, 2026-09-05 (`RESULTS/as-phase0/Studio/`). Two forks taken; BF16 deferred to M5-class hardware. The p0–p7 suite has **not** been run as a row on the M4 (R1) |
| **G-C1** compiler | **the top gate; not started past Phase 2-1.** Upstream clang emits PTX for CUDA when given a hand-made header. Nothing translates to Metal yet. The README's line "blocked only on the clang fetch" is stale: clang is fetched (R3) |
| **G-C3** smoothness | not started. Its first instance is caloml S2: an `nvcc` that CMake accepts |
| **G-C6** coded tier | not started. The README's stated blocker, a nil default device, does **not** hold on the M4 (R14) |
| **G-C2** matrix units | after the compatibility gates; needs M5-class hardware |

Phases:

- **Phase 0 (done 2026-10-14).**
  - cuda4AS is registered as a submodule of `~/YFCE`: the `.gitmodules` entry and gitlink are staged
    there, uncommitted.
  - Xcode 16.3 is selected; `ninja` is installed.
- **Phase 2-1 (done 2026-10-14).** Homebrew LLVM 23.1.3 was installed. This command produces PTX
  when `mini.h` defines `__global__`, `__device__` and a `__device__` `threadIdx`:

  ```sh
  clang++ -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -include mini.h
  ```
- **Phase 2-2 onward (not started).** This is the clean-room CUDA headers, the translator to Metal and
  the host runtime. It is the core of G-C1 and of caloml S2, and it is section 5.

**The caloml sample** (`samples/caloml/PLAN.md`): Celeritas 0.6.3 GPU transport, unmodified.

- **S0 done:** native CPU Celeritas, double, 239/0/36.
- **S1 done:** the float CPU oracle (213/1/61), P9 and P10, with the D1–D4 verdicts.
- **S2 is next.**

**The third-party corpus** (owned by "cuda projects"; read-only for you): 381 programs have passed on
a real GPU. They are in `cuda-projects/ready/` and `ready.json`, indexed in `RESULTS/cuda-projects/READY.md`:
176 reference, 190 pass-only and 15 silent. 22 have not passed (`RETEST.md`). After the nine-kernel
oracle, this is G-C1's test corpus. A `silent` program is not a reference. `cuda-projects/cccl` is a
CCCL 3.6.0 clone at commit 3900e21, for reading only.

## 4. Decisions in force

These are settled. Reopen one only with the user.

- **D1 (FP64).**
  - Bring S2–S3 up on `-DCELERITAS_REAL_TYPE=float` first.
  - The double build's `double` is **IEEE soft-double (soft64), correctly rounded RN-even, with a
    single-rounding `fma`**, and memory keeps IEEE binary64 bits.
  - df64 is not the default; it may come back later as an opt-in fast mode.
  - Reasons: df64 reaches up to 8.7e7 ulp on mul inside [1e-30, 1e30], and Celeritas holds values
    df64 can't represent (h² = 4.39e-53). See "Committee review of S1b".
- **D2 (Thrust).**
  - Use stock CCCL, pinned to 3.2.x to match the 5080 build, compiled through the translator.
  - **The translator assigns `blockIdx` from a per-launch arrival ticket**: one `atomic_fetch_add` by
    each threadgroup's first thread, with the counter reset for each launch. This is legal for any
    correct CUDA program and makes CUB's look-back safe by construction.
  - `__threadfence()` → `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)`.
    Acquire/release → seq_cst, because MSL 3.0–3.2 refuse them.
  - No multi-pass fallback.
- **D3 (device pointers).**
  - A device pointer is the `MTLBuffer`'s `gpuAddress`.
  - Every live allocation is resident for every dispatch.
  - `cudaMemcpy` maps an address back to (buffer, offset).
  - This applies across cuda4AS: any kernel that dereferences a pointer loaded from memory needs it.
- **D4 (caloml).** caloml stays untouched. Mac copies of its decks and its run scripts live in
  `samples/caloml/`.
- **G-C0 constraints the translator must honour** (README, "What G-C0 found"):
  - 32 KiB of threadgroup memory: the shared-memory spill transform is mandatory (`oracle/src/shared_48kb.cu`).
  - No 64-bit atomics: they are lock-backed, with the lanes of one SIMD-group serialized (S1
    showed this doesn't deadlock).
  - No `double`: soft64.
  - No independent thread scheduling: detect intra-warp spin locks and reject them with a diagnostic.
  - `vm_remap` fails: managed-memory pointer identity comes from compiler rebasing.
  - `maxBufferLength` is 50 % of RAM: allocations above it fail until memory tiers exist.
  - Never acquire the device through the system default alone.
- **Already native** (README): warp ↔ SIMD-group is 1:1 at 32 lanes; `simd_ballot` and `popcount`;
  `atomic_float` add; a 1024-thread block maps to one threadgroup; `bytesNoCopy` host buffers are
  coherent.

## 5. Next step

**Completed on 2026-10-26 (N1–N9):**
- **N1–N4 (DONE):** Open items O2, O3, O4 approved by user. `oracle/ref/*.bin` fetched and verified against SHA-256 (`N2`). Toolkit skeleton (`toolkit/`), build (`~/system-env/build/cuda4AS-toolkit`), install (`~/system-env/install/cuda4AS-toolkit-0.1.0`), and clean-room headers created (`N3`, `N4`).
- **N5–N7 (DONE — G-C1 nine-kernel oracle milestone):** Design note written (`toolkit/DESIGN.md`), Metal runtime (`toolkit/src/runtime/cudart.mm`) and translator (`toolkit/src/translator/cuda4as_translate.py`) implemented. All 9 G-C1 oracle kernels (`vector_add`, `saxpy`, `reduce_sum`, `matmul_tiled`, `histogram_atomics`, `shared_48kb`, `double_dot`, `grid_sync`, and `mma_inline_ptx` refusal) pass bit-identically on the M4 GPU (`RESULTS/g-c1/`).
- **N8 (DONE — caloml S2):** `nvcc` driver (`toolkit/src/driver/nvcc_main.cpp`) passes CMake `enable_language(CUDA)` and `find_package(CUDAToolkit)`. Stock CCCL 3.2.0 (`ca44ee7`) installed in `~/system-env/install/cuda4AS-toolkit-0.1.0/include/`. Celeritas 0.6.3 float CUDA build (`~/system-env/build/celeritas-0.6.3-cuda-float`) compiles and links all 932 targets (55 library `.cu` files + test `.cu` files) with 0 edits to Celeritas.
- **N9 (DONE — caloml S3):** Dual-pass LLVM IR host+device stitcher (`--stitch-host-ll` / `--stitch-dev-ll`) and unified memory (`[buf contents]` + `gpuAddress`) integrated with zero hardcoded kernels. **214 / 215 enabled CTests pass (99 %)**, including **39 / 39 `gpu`-labelled tests (100 %)**; the sole failure (`UniformLogGridCalculatorTest.spline`) is the known R8 float tolerance issue matching the CPU float oracle.

**Current next step (N10 / caloml S4):**
- **N10. Size and propose Geant4 11.2.x + Xerces-C + physics datasets in `~/system-env` (Stage S4)** to the user before downloading any sources (standing rule 3). Once approved and built, rebuild the Celeritas CPU and GPU float builds with `-DCELERITAS_USE_Geant4=ON` and run the S5 physics gate (`cms-2018.gdml` / `testem3-flat.gdml`).

### 5.1 Handoff Guide for the Next Apple Silicon Machine

For a dedicated, standalone copy, see [`HANDOFF.md`](HANDOFF.md).

#### 1. Target Machine Prerequisites
- macOS 14+ (Sonoma) or 15+ (Sequoia) on Apple Silicon (AArch64).
- Xcode Command Line Tools (`xcode-select --install`).
- Required packages:
  ```bash
  brew install llvm ninja cmake gh
  ```
  *(Homebrew LLVM provides upstream Clang with the NVPTX target enabled).*

#### 2. Repo Layout & Environment Setup
```bash
cd ~
git clone --recursive https://github.com/FanxinSun/YFCE.git
cd ~/YFCE/cuda4AS

mkdir -p ~/system-env/{src,build,install,logs,cuda4AS}

cat << 'EOF' > ~/system-env/cuda4AS/env.sh
export CUDA4AS_HOME="$HOME/system-env/install/cuda4AS-toolkit-0.1.0"
export CUDA4AS_LLVM="$(brew --prefix llvm)"
export PATH="$CUDA4AS_HOME/bin:$PATH"
export CELERITAS_CPU="$HOME/system-env/install/celeritas-0.6.3-cpu"
export CELERITAS_CPU_FLOAT="$HOME/system-env/install/celeritas-0.6.3-cpu-float"
EOF

source ~/system-env/cuda4AS/env.sh
```

#### 3. Build & Install cuda4AS Toolkit (v0.1.0)
```bash
cmake -B ~/system-env/build/cuda4AS-toolkit -S toolkit -G Ninja \
  -DCMAKE_INSTALL_PREFIX="$HOME/system-env/install/cuda4AS-toolkit-0.1.0" \
  -DCMAKE_BUILD_TYPE=Release
ninja -C ~/system-env/build/cuda4AS-toolkit install
nvcc --version
```

#### 4. Hardware Probes & G-C1 Oracle Verification
```bash
# 1. Hardware probe suite
cd ~/YFCE/cuda4AS/drop/as-phase0 && ./run.sh

# 2. 9-Kernel G-C1 Oracle
cd ~/YFCE/cuda4AS
gh release download v0.1.0-oracle-ref -R FanxinSun/cuda4AS -D oracle/ref/
python3 oracle/run_oracle.py --driver ~/system-env/install/cuda4AS-toolkit-0.1.0/bin/nvcc
# Expect 9/9 PASS (bit-identical to Linux RTX 5080)
```

#### 5. Verify Celeritas 0.6.3 Float Build (S2 & S3)
```bash
cmake -B ~/system-env/build/celeritas-0.6.3-cuda-float -S ~/system-env/src/celeritas-0.6.3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/system-env/install/celeritas-0.6.3-cuda-float" \
  -DCMAKE_CUDA_COMPILER="$HOME/system-env/install/cuda4AS-toolkit-0.1.0/bin/nvcc" \
  -DCELERITAS_BUILD_TESTS=ON -DCELERITAS_USE_CUDA=ON -DCELERITAS_REAL_TYPE=float \
  -DCELERITAS_USE_Geant4=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DCLI11_SOURCE_DIR="$HOME/system-env/src/CLI11-2.5.0" \
  -Dgoogletest_SOURCE_DIR="$HOME/system-env/src/googletest-1.17.0" \
  -Dnlohmann_json_SOURCE_DIR="$HOME/system-env/src/nlohmann_json-3.12.0"
ninja -C ~/system-env/build/celeritas-0.6.3-cuda-float
ctest --test-dir ~/system-env/build/celeritas-0.6.3-cuda-float --output-on-failure -L gpu
# Expect 39/39 GPU CTests PASS; 214/215 overall PASS.
```

## 6. Open items: each needs an answer or a decision

| id | item | status / who decides |
|---|---|---|
| O1 | **The blueprint `docs/cuda-on-apple-silicon.md` is missing.** The README cites it and its sections (§1, §2.3, §2.6–§2.9), but it wasn't found anywhere on this Mac. The user said to skip it for now. Don't reconstruct it from memory; ask where it is when a decision depends on it | open (user) |
| O2 | Move Homebrew `llvm` and `ninja` into `~/system-env`, or leave them? | **CLOSED (2026-10-26):** user approved leaving `/opt/homebrew/opt/llvm` and `/opt/homebrew/bin/ninja` in place and referencing via `~/system-env/cuda4AS/env.sh` |
| O3 | The toolkit's source directory name and install location (N3) | **CLOSED (2026-10-26):** user approved `toolkit/`, `~/system-env/build/cuda4AS-toolkit`, and `~/system-env/install/cuda4AS-toolkit-0.1.0` |
| O4 | CCCL 3.2.x as the toolkit's own copy (D2) means a fetch into `~/system-env/src` | **CLOSED (2026-10-26):** user approved (~42 MB shallow clone at `v3.2.0`, installed into toolkit `include/`) |
| O5 | `RESULTS/s1/` and `RESULTS/as-phase0/masshjs-Mac-mini/` hold the same `p9_lookback.json` and `p10_fp64.json`. Recommendation: `RESULTS/s1/` is canonical, and `as-phase0/<host>/` now holds the full M4 phase-0 run (`p0`–`p8`). Don't delete untracked files without the user's OK | open (user) |
| O6 | Untracked clutter: `.DS_Store` (repo root and `oracle/`) and `tools/__pycache__/`. `.gitignore` already excludes `.DS_Store` and `__pycache__/` | open (user) |
| O7 | A newer Xcode with Metal 4 would bring `metal_tensor`, which `probe/p6_matmul_{fp16,bf16}.metal` need, and possibly acquire/release ordering. It's a large download and not needed for G-C1 | open (user, later) |

## 7. Residuals: known gaps that don't block the next step

- **R1 (CLOSED 2026-10-26).** Ran `drop/as-phase0/run.sh` on the M4 (`apple9`); all `p0`–`p8` outputs, logs, and `status.txt` are stored in `RESULTS/as-phase0/masshjs-Mac-mini/`.
- **R2 (CLOSED 2026-10-26).** Updated `probe/SELF_REVIEW.md` with M4 verification context and entries for `p9_lookback` and `p10_fp64`.
- **R3 (CLOSED 2026-10-26).** Updated `README.md` with current G-C0/G-C1/G-C3/G-C6 statuses, Colab oracle description, and S1 (`p9_lookback`, `p10_fp64`) findings.
- **R4. soft64 prototype** (`probe/p10_fp64.swift`):
  - The standalone Swift prototype rounds toward zero; however, the toolkit's MSL translator (`toolkit/src/translator/cuda4as_translate.py`) implements IEEE-754 RN-even `soft64_add`, `soft64_mul`, and `soft64_fma` (verified bit-exact on `oracle/src/double_dot.cu`).
  - Remaining for the S6 double build: full `soft64` transcendental library (`sincos`, `exp`, `log`) and updating `probe/p10_fp64.swift` to use RN-even rounding so `atomicAdd` lands within 1 ulp of the arrival-order sequential sum.
- **R5.** The double `atomicAdd` lock is slow: 2.6 s for 2¹⁶ adds to one address. Speed work, after
  correctness.
- **R6.** The 64-bit single-copy store/load result is empirical (0 torn reads in 256M), not a
  documented Metal guarantee. Re-run P9 on every new chip.
- **R7.** CUB's tile sizes are tuned for 48 KB shared memory. Check that each algorithm Celeritas uses
  fits 32 KiB, or that the spill transform covers it.
- **R8.** The float CPU oracle fails `UniformLogGridCalculatorTest.spline` by design (a 1e-6
  tolerance), and Celeritas disables 25 tests under float (`_needs_double`). The GPU float build
  matches this (`214/215` pass, `UniformLogGridCalculatorTest.spline` is the sole failure); it doesn't count against cuda4AS.
- **R9.** `size_type` is 64-bit in the CPU builds and 32-bit in device builds, so compare at the
  physics level, not bitwise.
- **R10.** No full Geant4-imported physics deck has run on the M4 yet: `celer-sim` and `celer-g4` import physics through Geant4,
  which is S4. Size and approve that first.
- **R11.** caloml's full build fails `FieldPropagator:Cmse*` for an unrecorded reason (S7).
- **R12.** In `~/YFCE`: `.gitmodules` and the cuda4AS gitlink have been staged and uncommitted since
  2026-10-14. `lmz` shows modified and isn't ours. `customize-llm/` and `lmsluice/` belong to other
  projects.
- **R13.** Celeritas v0.7.0 is out (2026-10-11). 0.6.3 stays pinned to match caloml.
- **R14 (CLOSED 2026-10-26).** G-C6 verified on the M4 via `drop/as-phase0/run.sh`: `lmz/bench.swift` built and decoded all 512 streams (16.8 MB) **byte-identically** on the Apple M4 GPU in both direct-read (31.68 ms) and threadgroup-prefetch (31.70 ms) modes (`RESULTS/as-phase0/masshjs-Mac-mini/run-p8_lmz_decoder.log`).
- **R15.** `cuda-projects/ready/` is a directory, so `.gitignore`'s `cuda-projects/*/` excludes it. If
  those 381 sources are meant to be committed, that's for "cuda projects" and the user to settle.

## 8. Git and repository state

- **Committed in `cuda4AS` (`b74839a` on `origin/master`):**
  - Toolkit: `toolkit/` (`nvcc` driver, IR translator, runtime `libcudart`, clean-room headers).
  - Probes & results: `RESULTS/g-c1/` (9/9 pass), `RESULTS/as-phase0/masshjs-Mac-mini/` (M4 full p0–p8 run), `RESULTS/s1/` (`p9_lookback`, `p10_fp64`).
  - Documentation & plans: `HANDOVER.md`, `README.md`, `probe/SELF_REVIEW.md`, `samples/caloml/PLAN.md`.
- **Committed in `~/YFCE` (`ff2ab52` on `origin/master`):**
  - Registered and pointed `cuda4AS` submodule to `b74839a`, committed `.gitmodules`.
- **Handoff guide:**
  - Standalone guide added in `HANDOFF.md` and summarized in Section 5.1 above.

**`~/system-env`** (local environment, not tracked in git):
- `src/`: sources and tarballs for `celeritas-0.6.3`, `CLI11-2.5.0`, `googletest-1.17.0`, `nlohmann_json-3.12.0`, CCCL 3.2.0.
- `build/`: `cuda4AS-toolkit/`, `celeritas-0.6.3-cpu/`, `celeritas-0.6.3-cpu-float/`, `celeritas-0.6.3-cuda-float/`.
- `install/`: `cuda4AS-toolkit-0.1.0/`, `celeritas-0.6.3-cpu/`, `celeritas-0.6.3-cpu-float/`, `celeritas-0.6.3-cuda-float/`.
- `cuda4AS/`: `env.sh`, `build-celeritas-cpu.sh`.

## 9. People and sessions

- **The user** commits and pushes, approves downloads and sizes, and settles decisions. Put a
  decision to them with a recommendation.
- **"cuda projects"** (Claude Code, Terminal tab of that name) owns `cuda-projects/`, the Colab
  benches and `RESULTS/cuda-projects/`. Its tools are `tools/cuda_projects.py` (`ledger`, `ready`,
  `retest`, `merge`). Ask it, through the user, for corpus changes or new oracle runs.
- **"cuda4AS Committee"** (Claude Code, this file's author) has retired from the project.
- **Reaching a Claude session:** go through the user. Otherwise, type a one-line prompt into its
  Terminal tab with AppleScript: `tell application "Terminal"`, find the tab by its title, then
  `do script "<line>" in t`. Only do this when that tab shows it idle, and keep the line free of
  double quotes. Claude sessions reach you the same way. Put anything long in a file and send its
  path.

## 10. Where things are

| what | path |
|---|---|
| project README, gates, G-C0 findings | `README.md` |
| caloml sample plan of record | `samples/caloml/PLAN.md` |
| probes (Swift + MSL; one library per case, refusals verbatim, bounded work, JSON with machine block from `common.swift`) | `probe/`, reviewed in `probe/SELF_REVIEW.md` |
| nine-kernel corpus, nvcc harness, reference metadata | `oracle/src/`, `oracle/run_oracle.py`, `oracle/ref/*.json` (`.bin` via N2) |
| G-C0 results (M1 Pro) | `RESULTS/as-phase0/Studio/` |
| S1 results (M4) | `RESULTS/s1/` |
| drop for other Macs | `drop/as-phase0/`, `tools/make_drop.sh` |
| third-party corpus (read-only) | `cuda-projects/`, `RESULTS/cuda-projects/{LEDGER,READY,RETEST}.md` |
| Celeritas source, read-only | `~/system-env/src/celeritas-0.6.3` |
| CPU oracles | `~/system-env/install/celeritas-0.6.3-cpu{,-float}`, via `source ~/system-env/cuda4AS/env.sh` |
| caloml, read-only (reference runs from the 5080) | `~/caloml/validation/celeritas/smoke/`, `~/caloml/validation/orange-repro/` |
| upstream clang with NVPTX | `/opt/homebrew/opt/llvm` (`$CUDA4AS_LLVM`) |
