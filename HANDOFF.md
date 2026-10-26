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

---

## 6. Immediate Next Goal: Stage S4 (Geant4 + Xerces-C)

The next active milestone on the new machine is **Stage S4**:
1. **Size and Propose**: Size Geant4 11.2.x source + datasets (~1.5–2 GB) and Xerces-C 3.2.x (~10 MB) to the user for approval before downloading (per standing rule 3 & 4).
2. **Build Libraries**: Build Xerces-C and Geant4 in `~/system-env/build/` and install to `~/system-env/install/`.
3. **Rebuild Celeritas**: Reconfigure Celeritas float build with `-DCELERITAS_USE_Geant4=ON`.
4. **Physics Import Gate (Stage S5)**: Run `celer-sim` and `celer-g4` with GDML geometries (`cms-2018.gdml` / `testem3-flat.gdml`).
