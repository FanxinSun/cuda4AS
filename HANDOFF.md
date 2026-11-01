# cuda4AS — handoff guide

> **Two parts.** Part A (2026-11-01) is the current handover: the next
> development happens on the M1 MacBook Pro, and Part A is the brief for the
> session that does it. Part B is the 2026-10-26 guide written on the M4 Mac
> mini, retained as history with dated corrections. Read Part A first. The
> project's one plan is `plan/PLAN.md`; nothing in this file is an execution
> order of its own.

# Part A — Handover to the M1 MacBook Pro (2026-11-01)

| | |
|---|---|
| **From (manager)** | session **"cuda4AS Committee"** (Claude Code, the Linux box, `/home/rog/business/YFCE/cuda4AS`) |
| **To (executor)** | the first Claude Code session the user opens on the **M1 MacBook Pro** (`MacBookPro18,3`); suggested name **"cuda4AS M1 executor"** |
| **Plan step** | **TK-0** — `plan/PLAN.md` §12 row 11; contract in `plan/subplan_TK/TK-0.md` |
| **Starts when** | the user says so in that session. This file alone is not a go. |
| **Report to** | `RESULTS/tk0/<hostname>/REPORT.md` in the repository, then tell the user. The Committee reads it once the user has committed and pushed it, or copied it. |
| **Date** | 2026-11-01 (Asia/Shanghai) |

## A1. What you are taking over

cuda4AS is a native ahead-of-time CUDA toolchain for Apple silicon: unchanged
`.cu` sources and their build systems go in, and the program runs on the Apple
GPU, with no silent CPU fallback and no runtime compilation. Two lines of work
were merged on 2026-11-01 (`master` = `382a66f`):

- the **Codex lane** from the Linux box (2026-09): the milestone record M0 to
  M2A-R1, a Clang-semantic importer in `tools/m2a/`, and the canonical plan
  tree `plan/`;
- the **M4 Mac mini line** (2026-10): `toolkit/` — an `nvcc` driver that CMake
  accepts, a translator, clean-room CUDA headers and a Metal runtime — plus
  probe results, a nine-kernel oracle record and a Celeritas 0.6.3 float
  build.

The Committee read the toolkit against the plan on 2026-11-01 and found that
its runtime **silently runs any kernel it cannot dispatch to Metal on the CPU
and reports success** (`toolkit/src/runtime/cudart.mm:202-262`), that its
translator decides semantics with regular expressions, and that the nine
oracle records in `RESULTS/g-c1/` have no tracked harness behind them
(`plan/PLAN.md` §3, findings F1–F5). So the first job on this machine is not
to build features. It is to **measure what the toolkit really does on a
second chip, with evidence that can be trusted**: build it here, run the
hardware probes here, and run the nine unchanged oracle programs through a
tracked harness with emulation off and the dispatch path recorded. That is
TK-0. Read, in this order: this part; `plan/PLAN.md` §1–§3, §9, §10, §12;
`plan/subplan_TK/TK-0.md`; `README.md`; `HANDOVER.md` §1 (the user's rules on
a Mac) and §4; `toolkit/DESIGN.md`.

## A2. The machine, as last recorded

| fact | recorded value | recorded when / where |
|---|---|---|
| model, chip | `MacBookPro18,3`, Apple M1 Pro, **14 GPU cores**, 8 CPU (6P + 2E) | 2026-09-05, `RESULTS/as-phase0/Studio/README.md` |
| memory | 16 GiB unified; `maxBufferLength` 8 GiB; recommended working set 11.45 GB | 2026-09-05/06 |
| OS, Xcode, SDK | macOS 14.4 (23E214); Xcode 15.2 (15C500b); SDK 14.2; Metal compiler 32023.101; Swift 5.9.2 | 2026-09-06, `docs/m1/dependencies-mac.json` |
| GPU families | apple1–apple7, mac2, common1–3, metal3; **no apple8+, no Metal 4**; MSL up to 3.1 | 2026-09-05 |
| Metal device | `MTLCreateSystemDefaultDevice()` returns **nil** in a normal Aqua session; `MTLCopyAllDevices()` returns the M1 Pro. The merged runtime already enumerates (`cudart.mm:27`) | 2026-09-05 |
| tools present | Homebrew 5.1.15 at `/opt/homebrew`; **LLVM 23.1.0 at `/opt/homebrew/opt/llvm`** (not on `PATH`); CMake 4.3.3; Ninja 1.12.1; `/usr/bin/python3` | 2026-09-10 |
| disk | about 80 GB free | 2026-09-06 |
| hostname seen | `fanxins-macbook-pro.local` (the 2026-09-05 results directory was named `Studio`) | 2026-09-05 |

Step TK-0-0 re-checks every row. A difference is a finding to record, not a
reason to install anything.

## A3. The user's standing rules on this machine

These are the user's, not the Committee's. They hold in every step.

1. **This PC is the instrument, not the target.** The M1 Pro is one sample of
   Apple silicon, as the M4 and the RTX 5080 are samples of their kinds. Every
   number you write carries its machine and conditions. A refused or absent
   capability is a *result*, recorded with the compiler's or runtime's exact
   words, never a failure to hide. Constants you calibrate here are parameters
   with their derivation, never literals. The design has to hold from an M1
   with 32 KB of threadgroup memory to chips not yet built.
2. **Nothing is installed into anything that exists.** System Python, the
   Xcode toolchain and the existing Homebrew kegs are read-only. Everything
   new goes into `~/system-env` (layout in A5). No `brew install`, no `pip
   install`, no Xcode or SDK change, no `sudo`.
3. **The network may be metered.** Say what you will fetch and how large it
   is before fetching. This handover fetches one thing: the repository clone
   (about 10 MB, A4). Anything else needs the user's yes.
4. **Install only what the current step needs.** TK-0 needs no Celeritas, no
   CCCL, no Geant4. They belong to later steps with their own approvals.
5. **Read-only inputs:** `lmz/` (and its copies under `drop/as-phase0/lmz/`),
   `cuda-projects/`, `oracle/src/` and `oracle/ref/`, `~/caloml` if present.
   Never edit, stage or commit inside them. Do not edit `plan/` either: report
   what should change.
6. **Commits and pushes are the user's**, unless the user tells you otherwise
   in your session. End every report with the exact git state (`git status
   --porcelain`); "untracked" is not "clean".
7. **Compatibility first, speed second.** A construct that is refused outranks
   any throughput number from the same run.
8. **No silent emulation.** You measure with host emulation disabled (A5,
   TK-0-2). A kernel that only completes on the CPU is recorded as such and
   earns no pass.
9. **Every number in a report appears in a results file.** Where a value was
   not measured, write "not measured". Do not round into friendlier numbers.
   (This is the S1 lesson of `HANDOVER.md` §1.)
10. **Decisions are the user's and the Committee's.** You execute TK-0 and
    report facts. If a step cannot be done as written, stop at that step,
    write what you have, and tell the user.

## A4. Network budget

- `git clone --depth 1 https://github.com/FanxinSun/cuda4AS.git` transfers
  about **10 MB** (a `tar.gz` of the tree is 9.6 MB; 8.2 MB of it is one
  tracked archive, `tools/m1/artifacts/cuda4as-m1-native-feasibility-v1.tgz`).
  The README's "a few hundred KB" predates the M4 line. State the size to the
  user before cloning. If the user prefers no fetch, a `git bundle` made on
  the Linux box or the M4 and copied over works the same.
- Nothing else. The oracle's reference `.bin` files are **not** needed: the
  harness compares SHA-256 values that are tracked in `oracle/ref/*.json`.
  The drop's `data/*.bin` are regenerated locally by
  `tools/prep_synth_shared.py` (seconds, no download).

## A5. Steps, in order (TK-0-0 to TK-0-7)

All paths below assume the clone at `~/YFCE/cuda4AS`; any location the user
prefers is fine, recorded in the report. `<host>` is `$(hostname -s)`.

**TK-0-0 — inventory re-check.** Record into `RESULTS/tk0/<host>/env.txt`:

```sh
sw_vers; uname -a; xcode-select -p; xcrun --show-sdk-version
xcrun metal --version
/opt/homebrew/opt/llvm/bin/clang++ --version; clang++ --version
cmake --version; ninja --version; python3 --version
df -h ~ ; sysctl -n hw.memsize; system_profiler SPDisplaysDataType | grep -i -E 'chip|cores|metal'
```

Compare with A2. Then build and run `probe/p0_device.swift` with the drop's
route (TK-0-4 does the full set); its JSON machine block is the authoritative
device identity for every later record.

**TK-0-1 — clone and layout.**

```sh
cd ~ && mkdir -p YFCE && cd YFCE
git clone --depth 1 https://github.com/FanxinSun/cuda4AS.git     # ~10 MB, after the user's yes
cd cuda4AS && git rev-parse HEAD                                 # 382a66f… or later; write it down
mkdir -p ~/system-env/{src,build,install,logs,cuda4AS}
cat > ~/system-env/cuda4AS/env.sh <<'ENV'
export CUDA4AS_LLVM="/opt/homebrew/opt/llvm"
export CUDA4AS_HOME="$HOME/system-env/install/cuda4AS-toolkit-0.1.0"
export PATH="$CUDA4AS_HOME/bin:$PATH"
ENV
source ~/system-env/cuda4AS/env.sh
```

Check that each `~/system-env` path you create was free before you wrote it;
other projects share that directory on the user's Macs.

**TK-0-2 — the instrumentation patch, before building.** Four changes to
`toolkit/src/runtime/cudart.mm`, nothing else, kept as a diff in the report:

1. In `Cuda4ASContext::launchKernel`, the host path (the `if (hostEntry)`
   block) runs only when `getenv("CUDA4AS_HOST_EMULATION")` is `"1"`.
   Otherwise print one line to stderr, `cuda4as: <kernel>: no Metal pipeline;
   host emulation disabled`, and return `cudaErrorLaunchFailure`.
2. In `getPipelineState`, when `newLibraryWithSource` returns nil, print
   `cuda4as: MSL compile failed: <err.localizedDescription>` to stderr; when
   `newComputePipelineStateWithFunction` fails, print the kernel name and the
   error.
3. When `getenv("CUDA4AS_TRACE")` is `"1"`, print on every launch
   `cuda4as: launch <kernel> path=metal|host grid=<x>,<y>,<z> block=<x>,<y>,<z>`.
4. `opts.languageVersion` comes from `CUDA4AS_MSL_VERSION` when set
   (`3.0`, `3.1`), default unchanged.

Without the variables the only behavioural change is that silent emulation is
gone. Do not touch the translator, the driver, `oracle/` or `plan/`.

**TK-0-3 — build and install the toolkit.**

```sh
mkdir -p ~/system-env/logs/cuda4AS-toolkit
cmake -B ~/system-env/build/cuda4AS-toolkit -S toolkit -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/system-env/install/cuda4AS-toolkit-0.1.0" \
  2>&1 | tee ~/system-env/logs/cuda4AS-toolkit/configure.log
ninja -C ~/system-env/build/cuda4AS-toolkit install 2>&1 | tee ~/system-env/logs/cuda4AS-toolkit/build.log
nvcc --version
nvcc -v -c oracle/src/vector_add.cu -o /tmp/va.o      # record every tool the driver resolves
```

The driver picks `$CUDA4AS_LLVM/bin/clang++` for host and device passes and
falls back to `/usr/bin/clang++`; record which it used. Known trap: Homebrew
clang 23 headers against the macOS 14.4 libc++ can produce link or load errors
on newer C++ symbols. If that happens, record the exact text and stop; do not
install anything.

**TK-0-4 — the probes.**

```sh
python3 tools/prep_synth_shared.py          # regenerates drop/as-phase0/data/*.bin locally
cd drop/as-phase0 && ./run.sh && cd ../..   # p0–p8; results under drop/as-phase0/results/<host>/
mkdir -p RESULTS/as-phase0/<host> && cp -R drop/as-phase0/results/<host>/. RESULTS/as-phase0/<host>/
```

Then build and run `probe/p9_lookback.swift` and `probe/p10_fp64.swift` the
way the M4 did (one `swiftc -O` each with `probe/common.swift`; see
`probe/SELF_REVIEW.md`), writing to `RESULTS/s1/<host>/`. Expected on this
machine, to be confirmed rather than assumed: `double` refused, `atomic_ulong`
without operations, residency and 64-bit single-copy figures that differ from
the M4's, `memory_order_acquire`/`release` refused at MSL 3.0 and 3.1. Each is
a result with the compiler's text. The 2026-09-05 row for this machine
(`RESULTS/as-phase0/Studio/`) is the comparison.

**TK-0-5 — the tracked harness.** Write `oracle/run_cuda4as_oracle.py`,
modelled on `oracle/run_oracle.py` lines 96–183 (its kernel list, metadata
reading and summary JSON), that for each of the nine `oracle/src/*.cu`:

1. compiles with the installed `nvcc` (`-o <build>/<kernel>`), keeping argv,
   cwd, exit code, stdout and stderr in `<kernel>.build.log`;
2. runs the binary with `CUDA4AS_TRACE=1` set and `CUDA4AS_HOST_EMULATION`
   unset, with a 30-minute limit, keeping stdout, stderr and the exit code in
   `<kernel>.run.log`;
3. keeps the raw output bytes the program writes (as `oracle/run_oracle.py`
   expects them) and their SHA-256;
4. compares with `sha256` and `bytes` from `oracle/ref/<kernel>.json` — the
   only reference source; never regenerates a reference;
5. reads the dispatch path from the trace lines (`metal`, `host`, or `refused`
   when the compile or pipeline failed);
6. classifies: PASS_GPU (path `metal`, hash equal), FAIL (path `metal`, hash
   different, or a failed stage after a dispatch), SKIP_UNSUPPORTED (a
   translator or MSL refusal, text kept verbatim), NOT_RUN (a prerequisite
   stage did not run);
7. writes `RESULTS/g-c1/<host>/<kernel>.json` with: source hash, compiler
   executables and versions, every command, exit codes, path, device name and
   registry ID, output hash and byte count, reference hash and byte count,
   classification, first failing stage, wall time, and the machine block from
   TK-0-0's `p0_device.json`.

The M4's records in `RESULTS/g-c1/*.json` stay untouched. Run the harness's
own four checks once and keep their output: a kernel whose MSL fails must show
the compiler text and `refused`; emulation off and no pipeline must fail, not
pass; a tampered reference hash must mismatch; a missing `nvcc` must give
NOT_RUN.

**TK-0-6 — the runs.** Run the nine with emulation off. Then run the nine
again with `CUDA4AS_HOST_EMULATION=1` into `RESULTS/g-c1/<host>/emulation/`,
so the report can say which kernels only complete on the CPU. The second run
never produces PASS_GPU. For `mma_inline_ptx` the tracked translator refuses
(`cuda4as_translate.py:166`); the expected classification is
SKIP_UNSUPPORTED with that text — if you observe anything else, that is the
most important line of your report.

**TK-0-7 — the report.** `RESULTS/tk0/<host>/REPORT.md`:

1. Outcome: the per-kernel table (kernel, path, classification, hash equal or
   not, first failing stage, wall time), GPU-only and emulation-permitted.
2. Inventory: A2 rows with today's values; differences listed.
3. Probes: one line per probe with the headline figure and its machine label;
   refusals verbatim; differences from `RESULTS/as-phase0/Studio/` and from
   the M4.
4. The instrumentation diff and the exact toolkit install tree.
5. Differences from the M4's `RESULTS/g-c1/*.json`, record by record.
6. What you could not do, with the exact error text.
7. The git state, verbatim.

Then tell the user it is ready for the Committee.

## A6. What done looks like (acceptance)

- All nine kernels have a classification with a dispatch path; no silent
  success anywhere; references came from tracked JSON only; raw outputs and
  logs kept; refusals verbatim; the machine block in every record.
- Probe rows for this machine exist under `RESULTS/as-phase0/<host>/` and
  `RESULTS/s1/<host>/`.
- No download beyond the clone without the user's yes; nothing installed into
  anything that existed; no change to `oracle/src`, `oracle/ref`, `plan/`,
  `cuda-projects/`, `lmz/`.
- Any outcome closes TK-0. A poor result on the M1 Pro is information. The
  Committee decides TK-1's scope from the report.

## A7. Not in this handover

Celeritas, CCCL 3.2.0 (about 42 MB) and the S2/S3 ladder are **TK-2** and need
the user's approval of their downloads. Geant4 and Xerces-C (1.5–2 GB) are
gated by the user for M6-7 stage S4. Containing the emulation properly,
parameterising the M4 constants and the translator's diagnostics are **TK-1**.
Replacing the regex front-end with Clang-derived semantics is **TK-3**. Do not
start them, and do not "fix" the toolkit beyond TK-0-2; write down what you
would fix.

## A8. Known traps on this machine

- The nil default device (A2): anything that calls
  `MTLCreateSystemDefaultDevice()` alone finds no GPU here. The probes and the
  merged runtime enumerate; `lmz/bench.swift` (p8) does not, and failed here
  on 2026-09-05 for that reason — a result, not a defect to patch.
- Xcode 15.2: Metal compiler 32023.101 and MSL ≤ 3.1, against the M4's
  32023.619 and MSL 3.2. Refusal texts may differ; record them.
- The runtime hard-codes the M4 (`multiProcessorCount = 10`, SM count 84,
  1.45 GHz, a 64 KiB spill buffer per block) and `/usr/bin/python3`; the
  driver carries the M4 author's home path as a fallback. On this machine
  these are wrong or irrelevant; record their effects, do not fix them (TK-1).
- 16 GiB and an 8 GiB `maxBufferLength`: none of the nine oracles comes close;
  `p4_bandwidth` sizes itself.
- Thermal and other load: run the oracles with nothing else heavy running, and
  say in the report what else was running.

# Part B — The 2026-10-26 M4 Mac mini guide (retained; corrections in dated notes)

> **Status (2026-11-01):** retained as history. Its commands worked on the M4
> Mac mini for its author. Where a line is wrong for the M1 MacBook Pro, or
> states something the plan no longer accepts, a dated "M1 note" says so
> beside it. Nothing below is an execution order: see `plan/PLAN.md` §12.

# cuda4AS — Apple Silicon Machine Handoff Guide

This guide provides step-by-step instructions for bootstrapping `cuda4AS` on a new Apple Silicon machine (e.g. M1/M2/M3/M4/M5 series) and resuming development seamlessly.

---

## 1. Prerequisites on the New Apple Silicon Machine

Ensure the target machine satisfies the following environment requirements:

- **Operating System**: macOS 14.x (Sonoma) or macOS 15.x (Sequoia) running on Apple Silicon (AArch64).
- **Apple Developer Tools**: Xcode Command Line Tools or Xcode 15/16 (`clang`, `swift`, `metal`).
  ```bash
  xcode-select --install
  ```
- **Package Manager & Upstream Tools**:
  ```bash
  brew install llvm ninja cmake gh
  ```
  > [!NOTE]
  > Homebrew `llvm` contains the upstream Clang with the NVPTX target enabled (`-target nvptx64-nvidia-cuda`), which `cuda4AS` requires to compile device `.cu` code into NVPTX64 IR.

> **M1 note (2026-11-01):** on `MacBookPro18,3` Homebrew LLVM 23.1.0, CMake
> 4.3.3 and Ninja 1.12.1 were present on 2026-09-10
> (`docs/m1/dependencies-mac.json`); install nothing unless TK-0-0 finds a
> prerequisite missing, and size it first. `gh` is not needed for TK-0. The
> NVPTX target is not what the driver uses: it compiles the device pass with
> `--cuda-host-only -Xclang -fcuda-is-device` for the host triple
> (`toolkit/src/driver/nvcc_main.cpp:202-216`); what it needs is a clang that
> accepts CUDA language mode.

---

## 2. Repository Layout & Cloning

Clone the parent repository recursively so all submodules (including `cuda4AS`) are checked out:

```bash
cd ~
git clone --recursive https://github.com/FanxinSun/YFCE.git
cd ~/YFCE/cuda4AS
```

Create the standardized `~/system-env` directory structure (standing rule 2: all builds, tools, and environments live in `~/system-env`):

```bash
mkdir -p ~/system-env/{src,build,install,logs,cuda4AS}
```

Set up the environment configuration script in `~/system-env/cuda4AS/env.sh`:

```bash
cat << 'EOF' > ~/system-env/cuda4AS/env.sh
export CUDA4AS_HOME="$HOME/system-env/install/cuda4AS-toolkit-0.1.0"
export CUDA4AS_LLVM="$(brew --prefix llvm)"
export PATH="$CUDA4AS_HOME/bin:$PATH"
export CELERITAS_CPU="$HOME/system-env/install/celeritas-0.6.3-cpu"
export CELERITAS_CPU_FLOAT="$HOME/system-env/install/celeritas-0.6.3-cpu-float"
EOF

source ~/system-env/cuda4AS/env.sh
```

> **M1 note (2026-11-01):** clone `cuda4AS` directly, not the parent `YFCE`
> recursively: the parent's local and remote histories are not yet reconciled
> (`plan/PLAN.md` §16). A shallow clone is about 10 MB (Part A, A4). The
> `CELERITAS_CPU*` variables belong to TK-2.

---

## 3. Build & Install the cuda4AS Toolkit (v0.1.0)

Build and install the `cuda4AS` toolkit (`nvcc` driver, LLVM IR translator, clean-room CUDA headers, CCCL 3.2.0, and Metal runtime `libcudart`):

```bash
cmake -B ~/system-env/build/cuda4AS-toolkit -S toolkit -G Ninja \
  -DCMAKE_INSTALL_PREFIX="$HOME/system-env/install/cuda4AS-toolkit-0.1.0" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C ~/system-env/build/cuda4AS-toolkit install
```

Verify that `nvcc` is functional:

```bash
nvcc --version
# Expected output:
# cuda4AS nvcc driver version 0.1.0 (Apple Silicon Metal backend)
```

> **M1 note (2026-11-01):** apply the TK-0-2 instrumentation patch before
> this build (Part A, A5). CCCL is not part of the toolkit's source tree; it
> was fetched separately on the M4 and belongs to TK-2.

---

## 4. Hardware Verification & G-C1 Oracle Check

### 4.1 Run Phase 0 Probes
Run the hardware validation probe suite on the new Apple Silicon chip:

```bash
cd ~/YFCE/cuda4AS/drop/as-phase0
./run.sh
```
Results, logs, and `status.txt` will be generated in `results/$(hostname -s)/`.

### 4.2 Run G-C1 Nine-Kernel Correctness Oracle
Download the ground-truth binary reference outputs produced by an NVIDIA RTX 5080 (sm_120) and verify bit-identical execution:

```bash
cd ~/YFCE/cuda4AS

# Download reference binaries from GitHub release if not already cached
gh release download v0.1.0-oracle-ref -R FanxinSun/cuda4AS -D oracle/ref/

# Run the 9-kernel oracle verification suite
python3 oracle/run_oracle.py --driver ~/system-env/install/cuda4AS-toolkit-0.1.0/bin/nvcc
```

**Expected Result**:
- `vector_add`, `saxpy`, `reduce_sum`, `matmul_tiled`, `histogram_atomics`, `shared_48kb`, `double_dot`, `grid_sync`: **PASS** (100 % bit-identical to Linux RTX 5080 ground truth).
- `mma_inline_ptx`: **PASS** (diagnostic refusal on Apple Silicon GPU without matrix units).
- **Total: 9 / 9 PASS (100 %)**.

> **M1 note (2026-11-01):** `oracle/run_oracle.py` has no `--driver` option
> and needs `nvcc` and `nvidia-smi`; it cannot run on a Mac, and no tracked
> file produced the `RESULTS/g-c1/*.json` records. TK-0-5 writes the tracked
> harness. "9 / 9 PASS" is not an expectation; a refusal of `mma_inline_ptx`
> is (`plan/PLAN.md` §3, F3). The reference `.bin` download is unnecessary:
> the harness compares SHA-256 values from `oracle/ref/*.json`. The drop's
> `data/*.bin` are regenerated by `tools/prep_synth_shared.py`.

---

## 5. Celeritas 0.6.3 Verification (Stage S2 & S3)

To verify real-world application compatibility with Celeritas 0.6.3 without edits:

### 5.1 Pre-fetch Dependencies into `~/system-env/src`
Ensure the pinned dependency archives are extracted into `~/system-env/src`:
- `celeritas-0.6.3`
- `CLI11-2.5.0`
- `googletest-1.17.0`
- `nlohmann_json-3.12.0`

### 5.2 Configure & Build Celeritas Float Build
```bash
cmake -B ~/system-env/build/celeritas-0.6.3-cuda-float -S ~/system-env/src/celeritas-0.6.3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/system-env/install/celeritas-0.6.3-cuda-float" \
  -DCMAKE_CUDA_COMPILER="$HOME/system-env/install/cuda4AS-toolkit-0.1.0/bin/nvcc" \
  -DCELERITAS_BUILD_TESTS=ON \
  -DCELERITAS_USE_CUDA=ON \
  -DCELERITAS_REAL_TYPE=float \
  -DCELERITAS_USE_Geant4=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DCLI11_SOURCE_DIR="$HOME/system-env/src/CLI11-2.5.0" \
  -Dgoogletest_SOURCE_DIR="$HOME/system-env/src/googletest-1.17.0" \
  -Dnlohmann_json_SOURCE_DIR="$HOME/system-env/src/nlohmann_json-3.12.0"

ninja -C ~/system-env/build/celeritas-0.6.3-cuda-float
```

### 5.3 Run CTests
```bash
ctest --test-dir ~/system-env/build/celeritas-0.6.3-cuda-float --output-on-failure -L gpu
```
**Expected Result**:
- **39 / 39 `gpu`-labelled tests PASS (100 %)**.
- Across all 215 tests: **214 / 215 PASS (99 %)**, matching the CPU float oracle (the single failure is `UniformLogGridCalculatorTest.spline`, which is due to a 1e-6 float tolerance in Celeritas test code, known residual R8).

> **M1 note (2026-11-01):** Celeritas is TK-2, not part of the M1 handover;
> its downloads (3.7 MB of pinned sources plus CCCL 3.2.0, about 42 MB) need
> the user's approval first, or a copy from the M4's `~/system-env/src`. The
> "39 / 39" and "214 / 215" figures are reported by the M4 session; whether
> each test's kernels ran on Metal or in the runtime's CPU emulation is
> unverified (`plan/PLAN.md` §3, F1), which is what TK-2 measures.

---

## 6. Immediate Next Goal: Stage S4 (Geant4 + Xerces-C)

> **Retired on 2026-11-01 as an execution order.** S4 is not the next step;
> the order is `plan/PLAN.md` §12 (TK-0 next, on the M1 MacBook Pro). Geant4
> stays gated by the user (`plan/PLAN.md` §16). The version differs between
> this file (11.2.x) and the caloml record (11.4.2); it is fixed before
> sizing. The text below is retained unchanged.

The next active milestone on the new machine is **Stage S4**:
1. **Size and Propose**: Size Geant4 11.2.x source + datasets (~1.5–2 GB) and Xerces-C 3.2.x (~10 MB) to the user for approval before downloading (per standing rule 3 & 4).
2. **Build Libraries**: Build Xerces-C and Geant4 in `~/system-env/build/` and install to `~/system-env/install/`.
3. **Rebuild Celeritas**: Reconfigure Celeritas float build with `-DCELERITAS_USE_Geant4=ON`.
4. **Physics Import Gate (Stage S5)**: Run `celer-sim` and `celer-g4` with GDML geometries (`cms-2018.gdml` / `testem3-flat.gdml`).
