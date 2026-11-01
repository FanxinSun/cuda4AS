# M1 PyTorch and CUDA-extension platform audit

Status: pinned source audit complete; no PyTorch checkout, configure, build, or
extension execution was attempted.

## Audit pin

The audit uses the latest official GitHub release visible on 2026-09-06:
[PyTorch v2.14.0](https://github.com/pytorch/pytorch/releases/tag/v2.14.0),
published `2026-09-02T17:40:10Z`. Its lightweight tag resolves to commit
`2b3ec34829036a65cd9d1398ea72a0167dc37470`. Exact hashes for the five inspected
files are in [`upstream-evidence.json`](upstream-evidence.json).

The inspected primary sources are the pinned root `CMakeLists.txt`,
`cmake/Dependencies.cmake`, `cmake/public/cuda.cmake`, `setup.py`, and
`torch/utils/cpp_extension.py`. The full source archive was not downloaded and
a full framework build is outside M1.

## Full framework build surface

At this pin, `USE_CUDA` enters the legacy CMake `find_package(CUDA)` path,
enables CMake's CUDA language, and then requires `find_package(CUDAToolkit)`.
It requires the CUDA compiler and toolkit versions to agree and rejects CUDA
versions below 12.6. It also compiles and runs a header-version probe against
the discovered CUDA libraries.

The CUDA dependency graph creates or consumes normal CUDA imported targets,
including driver/runtime, cuBLAS/cuBLASLt, cuFFT, cuRAND, NVRTC, and optional
cuDNN and cuSPARSELt surfaces. Other options bring additional platform and
distributed dependencies. Disabling individual optional libraries does not
remove the core compiler, toolkit, runtime, and PyTorch CUDA-backend
requirements.

The pinned CuMetal source-toolkit generator advertises CUDA 12.2.0. It therefore
fails PyTorch 2.14's 12.6 minimum before kernel compatibility is tested. Merely
changing the advertised string would not prove the matching header, compiler,
runtime-library, generated-code, or operator contracts that PyTorch checks.

## Out-of-tree CUDA extension surface

`torch.utils.cpp_extension` first derives `CUDA_HOME` from `CUDA_HOME`,
`CUDA_PATH`, `nvcc` on `PATH`, or a conventional CUDA directory. The module only
initializes its global `CUDA_HOME` when the installed PyTorch itself reports a
compiled CUDA backend and a non-null `torch.version.cuda`.

`CUDAExtension` then:

- adds CUDA include and library paths;
- links `cudart`, `c10_cuda`, and `torch_cuda` in addition to the ordinary
  PyTorch libraries;
- checks `$CUDA_HOME/bin/nvcc --version` against `torch.version.cuda`;
- sends `.cu` files to that `nvcc` command;
- sends relocatable-device-code extensions through an explicit `nvcc -dlink`
  stage; and
- requires Ninja when `dlink=True`.

A normal macOS CPU/MPS PyTorch installation does not become a CUDA-extension
host solely because a compiler shim is placed on `PATH`: its own `torch_cuda`
and `c10_cuda` libraries and CUDA build metadata are also required. No such
candidate-built PyTorch distribution exists in this repository.

## Blockers by owner

| Owner | Concrete blocker at the audited pins | Evidence needed to clear it |
|---|---|---|
| Candidate compiler/toolkit integration | CuMetal's generated CMake toolkit reports CUDA 12.2, while PyTorch requires 12.6 or newer and checks compiler/toolkit/header agreement. | A maintained candidate integration that truthfully satisfies the 12.6+ discovery and compile probes without version spoofing. |
| Candidate runtime/library surface | PyTorch expects broad CUDA driver, runtime, math-library, and NVRTC targets and behavior. CuMetal exposes several names, but cuda4AS has reproduced none of the PyTorch-required API or operator surface. | Leaf API inventories plus source-built, Apple-GPU, numerical tests for every enabled library path. |
| Upstream build logic | The full build assumes standard CUDA CMake discovery and generated build behavior. Candidate-specific fake-toolkit wiring is not an upstream PyTorch platform backend. | A bounded, reviewable build integration whose changes and unsupported options are explicit. |
| Installed PyTorch package | `cpp_extension` requires PyTorch's own CUDA build metadata and links `c10_cuda` and `torch_cuda`; an ordinary macOS CPU/MPS package does not supply this contract. | A pinned PyTorch build produced against the candidate, followed by import, dispatch, loader, and extension ABI tests. |
| Device linking | PyTorch extensions can require real RDC/device linking. CuMetal's audited CMake shim emits an empty object for `-dlink`; it has not shown cross-translation-unit external device-symbol resolution. | The M1 multi-TU fixture, followed by PyTorch's actual extension dlink flow if that foundational case succeeds. |
| Platform code | PyTorch has ordinary Apple host/rpath handling and disables some CUDA ecosystem options on Apple, but it has no audited CuMetal/Apple-CUDA backend contract at this pin. | Platform-source audit and small, owned integration cases before any framework-scale build. |
| Resource/environment | A full PyTorch checkout/build is large and is outside the approved milestone. The user's Mac inventory and candidate native proof are still outstanding. | Separate future scope and explicit resource approval after the foundational M1 gates. |

## M1 conclusion

PyTorch 2.14 and its CUDA extensions are not a viable first native proof for
this milestone. The blockers arise before application kernel semantics and
span both build discovery and missing binary/runtime contracts. M1 should first
resolve the direct source, unchanged minimal CMake, and real multi-TU
device-link cases. Even if all three succeed, this audit supports only a later,
separately scoped PyTorch adapter investigation; it does not support claiming
framework compatibility.
