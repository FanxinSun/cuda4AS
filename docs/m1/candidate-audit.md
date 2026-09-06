# M1 pinned candidate audit

Status: source and dependency audit complete; every native cuda4AS case remains
`NOT_RUN` pending the user's Mac return and native run.

The machine-readable evidence and all selected raw-file hashes are recorded in
[`upstream-evidence.json`](upstream-evidence.json). Statements below describe
the pinned revision only. Upstream reports are labeled as reports and are not
cuda4AS validation.

## Pin and source integrity

- Candidate: [Lulzx/cuda-metal](https://github.com/Lulzx/cuda-metal) at
  `f486e5ebcfd381d06e3297afd65dbcbd5006a902` (CuMetal 0.5.0 in its root
  CMake project).
- Commit timestamp: `2026-09-05T14:20:59Z`.
- Pinned source archive: 6,694,104 bytes, SHA-256
  `57358b123daece57e472a8bf2805a0919e6879e7a1a781d8d20866b12ffaafbd`.
- The complete recursive tree contains 923 blobs and 15,536,441 expanded blob
  bytes. Every archive file was checked against its Git blob ID and size; no
  blob was missing, added, or different.
- The sole submodule is `third_party/VF64-metal`, pinned at
  `729021777455da72db8809d9ef1269c677d88b3f`. Its separate archive is 134,892
  bytes with SHA-256
  `c9e0308a54a3beec0dba15a12b81a68cda9ad502a919a6dd1cfe193a4bd6e5a5`.
  Its 142 blobs and 574,689 expanded bytes also match the recursive Git tree.
- The combined expanded source payload is 16,111,130 file bytes. The main
  codeload archive does not contain submodule data, so both pins are required
  for a reproducible build.

## Build surface selected for M1

CuMetal requires CMake 3.28, C++20 and Objective-C++20. Its build looks for
LLVM through a CMake config and enables genuine NVVM import only with LLVM 18
or newer. LZ4 and Zstd headers and libraries are required. The runtime links
Apple Accelerate, Foundation, Metal, Metal Performance Shaders, and QuartzCore.
The upstream build guide requires Apple Silicon and macOS 14 or newer, and
identifies full Xcode as necessary for its reference-metallib validation path.
The user's inventory will establish which of these are already present.

The M1 candidate build is pinned to:

```text
CMAKE_BUILD_TYPE=Release
CUMETAL_BUILD_TESTS=OFF
CUMETAL_ENABLE_CUDA_REGISTRATION=ON
CUMETAL_ENABLE_BINARY_SHIM=OFF
CUMETAL_CUDA_ARCH=sm_86
CUMETAL_FP64_MODE=ieee64
```

`sm_86` and its resulting `__CUDA_ARCH__=860` behavior are an Ampere source
dispatch hypothesis, not a description of Apple hardware. The native runner
must record the exact value used. The explicit `ieee64` setting prevents the
candidate's reduced-precision `fast48` runtime default from silently becoming
the cuda4AS default. None of the three initial M1 fixtures exercises FP64, so
this flag does not validate the candidate's software binary64 claims.

The build uses its task-local build tree and runtime cache. No install prefix,
shell startup edit, global loader variable, or system path change is needed for
the feasibility run.

## Compilation routes

| M1 case | Pinned route | Source-audit finding |
|---|---|---|
| `oracle.vector_add` | Direct `cumetalc`, explicit typed `cumetal-ir`, native-AOT executable, `--cuda-arch sm_86`, `--fp64=ieee64` | Direct `.cu` input defaults to the typed route and accepts one input file. The pinned compiler labels this native typed path `provenance=generic_nvvm_lowering`, rendered by the runtime as `source=generic_nvvm`; M1 requires that exact source plus `device=apple_gpu`, `launch_success=true`, a completed duration, no PTX-JIT cache, and the enrolled 4,194,304-byte golden. |
| `integration.minimal_cmake_cuda` | Ordinary unchanged CMake CUDA project using a task-local M1 adapter derived from the pinned upstream source-toolkit generator | The tree does contain a CMake-facing `nvcc` shim, but it is generated inside `scripts/build_llama_cpp_cumetal.sh`, not installed as a general CMake compiler package. It handles CMake compiler probes and delegates source compilation to CUDA-capable Clang. The M1 adapter remains outside the application and will be hashed and disclosed. |
| `integration.multi_tu_device_link` | Same CMake adapter, three unchanged CUDA translation units, separable compilation, required device-link step | The upstream shim answers `-dlink` by emitting an empty host object because it expects each translation unit to register its own kernel image. That is not a device-image link and does not establish resolution of an external `__device__` call across translation units. This fixture deliberately crosses that boundary; only its returned configure/compile/device-link/launch logs determine `PASS_GPU`, `FAIL`, or `UNSUPPORTED`. |

The source-toolkit generator hardcodes a CUDA 12.2 discovery surface and also
creates a task-local `libcuda.dylib` alias. The M1 adapter will remove that
binary alias and the unused `CUDA::cuda_driver` mapping before execution. This
keeps the experiment on source compilation with
`CUMETAL_ENABLE_BINARY_SHIM=OFF`; no SASS or prebuilt CUDA binary route is part
of M1. The adapter's own diff and hash will be returned. Application sources
and their CMake files remain byte-for-byte pinned.

Native-AOT, registration/PTX, and any control result must remain separate.
The first oracle route is native-AOT. The CMake shim produces source-built
objects that register embedded device material with the runtime, so those
results are labeled registration/PTX even if they launch successfully. A
control cannot substitute for any required case.

## GPU and numerical proof boundary

CuMetal documents `CUMETAL_TRACE_GPU=1` provenance containing device, lowering
source, semantic quality, and launch completion. M1 accepts `PASS_GPU` only
when the returned record names an inventoried Metal device, reports an actual
Apple-GPU launch and completion, rejects fallback/stub/approximate provenance,
and matches the fixture's complete expected output. Process success, a printed
`PASS`, or the presence of a metallib is insufficient.

The upstream repository reports extensive M4 Pro results, including unmodified
CUDA sample and framework workloads. Those reports justified testing this
candidate but contribute zero cuda4AS compatibility passes. This repository
has no independent native result yet.

## License and provenance record

- The CuMetal repository declares Apache-2.0 and contains the Apache 2.0 text.
- Vendored VkFFT declares version 1.3.4, an MIT notice, and local modifications
  in `third_party/VkFFT/README.Gromacs`.
- Vendored metal-cpp contains an Apache-2.0 license with Apple Inc. 2024
  copyright.
- `VF64-metal` has no license-named file in the pinned recursive tree and its
  GitHub repository metadata has no license declaration. The parent CuMetal
  license is not assumed to license a separately pinned submodule. This is a
  material provenance gap for adoption or redistribution and must remain open
  for manager judgment. Internal feasibility testing does not resolve it.
- CuMetal's legal notice describes a clean-room/source-recompilation policy and
  distinguishes the optional `libcuda.dylib` alias. This record reports that
  upstream position and does not make a legal determination.

## Current recommendation boundary

The source audit supports continuing the bounded Mac feasibility run: the
candidate has a real typed source route, positive GPU provenance fields, and a
CMake-oriented shim worth testing. Adoption is not yet supported. The
unverified VF64 license, task-specific CMake adapter, empty device-link action,
reduced-precision default, and absence of independent Apple-GPU evidence are
all open gates. The adopt, selective-reuse, or reject recommendation will be
updated from returned native evidence and left for manager acceptance.
