# Preliminary dependency inventory

Snapshot date: 2026-09-06 (Asia/Shanghai). This inventory describes the
existing host and verifiable candidate requirements for later work. M0 did not
install, upgrade, fetch, clone, or build a compiler or runtime. Package sizes
below are package-manager estimates, not approval to consume them.

## Current WSL host

| Item | Observed value | M0 use or limitation |
| --- | --- | --- |
| Distribution | Ubuntu 26.04 under WSL2, x86-64 | evidence inspection and possible portable compiler work only |
| Kernel | `6.18.33.2-microsoft-standard-WSL2` | not a native macOS deployment environment |
| Memory | 25,199,013,888 bytes reported by the host | point-in-time host fact |
| Workspace availability | about 570.6 GB during the scan | point-in-time filesystem fact, not a reserved budget |
| NVIDIA device | GeForce RTX 5080; driver 616.64 during M0 | may support future independent oracle candidates; historical goldens were not rerun |
| Shell/tool path | normal system environment; no project environment created | installed environments remained read-only |

Installed command-line tools relevant to dependency resolution were:

| Tool | Observed version |
| --- | --- |
| Git | 2.53.0 |
| CMake | 4.2.3 |
| Ninja | 1.13.2 |
| GNU Make | 4.4.1 |
| GCC / G++ | 15.2.0 |
| Python | 3.14.4 |
| pkg-config | 2.5.1 |
| ccache | 4.12.3 |
| jq | 1.8.1 |
| GNU tar | 1.35 |
| GitHub CLI | 2.97.0 |
| curl | 8.18.0 |

The default `/usr/bin/nvcc` reports CUDA 12.4. The recorded oracle was produced
with the separately installed `/usr/local/cuda-13/bin/nvcc`, CUDA 13.2 build
`cuda_13.2.r13.2/compiler.37953736_0`, and that compiler remains present. The
recorded producer used driver 610.88 and `sm_120`; the current driver difference
is retained rather than folded into the historical record.

Development packages already installed include:

| Package/component | Observed version or file | State |
| --- | --- | --- |
| LLVM shared library | `/usr/lib/llvm-21/lib/libLLVM.so.21.1`, about 133 MB | installed runtime/shared library |
| Clang C++ shared library | `/usr/lib/llvm-21/lib/libclang-cpp.so.21.1`, about 58 MB | installed package `libclang-cpp21` 21.1.8-6ubuntu1 |
| Clang shared library | `/usr/lib/llvm-21/lib/libclang-21.so.21`, about 32 MB | installed package `libclang1-21` 21.1.8-6ubuntu1 |
| LZ4 headers/library | `liblz4-dev` 1.10.0-8 | installed |
| Zstandard headers/library | `libzstd-dev` 1.5.7 | installed; the `zstd` CLI is absent |
| zlib headers/library | 1.3.1 development package | installed |

## Missing Linux build requirements and package estimates

`clang`, `clang++`, and `llvm-config` were not on `PATH`. The installed shared
libraries therefore do not constitute a usable LLVM development toolchain.
Ubuntu package metadata offers LLVM/Clang majors 18 through 21; the installed
libraries make 21.1.8-6ubuntu1 the least surprising candidate to evaluate, but
M0 does not select or install it.

Read-only `apt-get -s --no-install-recommends` simulations produced these
current estimates:

| Simulated request | New packages | Package download fields | Installed-size fields |
| --- | ---: | ---: | ---: |
| `clang-21 llvm-21-dev` | 12 | 75,810,632 bytes | 503,699,456 bytes |
| `clang-21 llvm-21-dev libclang-21-dev` | 13 | 106,013,156 bytes | 803,062,784 bytes |

These sums were computed from the packages selected by the simulation. They do
not include filesystem overhead, later repository changes, optional tools, a
source checkout, or build products. They must be refreshed before an approved
installation. The apt cache contained a 13,724,698-byte
`libclang-cpp21_21.1.8-6ubuntu1_amd64.deb`, but that package is already
installed; none of the newly selected closure was established as cached. The
entire apt cache was about 3.48 GB and is not a meaningful project dependency
estimate. The historical 117.8 MB compiler figure is therefore neither a
verified current download nor a complete disk requirement.

Linux remains insufficient for an end-to-end product pass. LLVM's official
[CUDA compilation documentation](https://llvm.org/docs/CompileCudaWithLLVM.html)
states that its normal CUDA compilation path is no longer supported on macOS;
the candidate must provide and validate its own native host/device integration.

## First reuse candidate: CuMetal

M0 resolved upstream CuMetal (`Lulzx/cuda-metal`) `main` and `HEAD` to
[`f486e5ebcfd381d06e3297afd65dbcbd5006a902`](https://github.com/Lulzx/cuda-metal/commit/f486e5ebcfd381d06e3297afd65dbcbd5006a902),
committed 2026-09-05T14:20:59Z. The repository API reported Apache-2.0 metadata,
a 9,882 KiB repository-size field, and a last push at
2026-09-05T14:21:05Z. That size field is hosting metadata, not a measured clone,
expanded source, dependency, or build size. `main` is moving; only an explicit
revision can enter a reproducible M1 manifest.

Inspection of files at that pinned revision identifies these candidate
requirements:

| Requirement | Pinned upstream fact | Local/Mac state |
| --- | --- | --- |
| Build generator | CMake minimum 3.28 | WSL CMake is new enough; native Mac value unrecorded |
| Languages | C++20 and Objective-C++20 | WSL GCC is not evidence for the Apple host path |
| LLVM | CONFIG package, major 18 or later, for the genuine NVVM importer | usable headers, tools, CMake config, and selected revision are missing locally |
| Compression | LZ4 and Zstandard headers/libraries | development packages exist in WSL; native Mac state unrecorded |
| Apple frameworks | Accelerate, Foundation, Metal, MetalPerformanceShaders, QuartzCore | only available and testable on native macOS |
| Metal build | full Xcode for reference metallib generation in the documented path | returned run reports Xcode 15, but no current build-machine inventory exists |
| Target runtime | supported arm64 macOS plus an Apple GPU | one historical M1 Pro/macOS 14.4 run exists; no cuda4AS run exists |

CuMetal's pinned
[verified-results document](https://github.com/Lulzx/cuda-metal/blob/f486e5ebcfd381d06e3297afd65dbcbd5006a902/docs/verified-results.md)
reports unchanged-source cuda-samples and llama.cpp slices. Those are upstream
claims to reproduce. M0 made no adoption, fork, license-distribution, or
selective-reuse decision and did not clone the repository.

## Native macOS information still required

The only returned machine record is MacBookPro18,3 with an M1 Pro, 14 GPU
cores, 16 GiB memory, macOS 14.4 build 23E214, Xcode 15, Swift 5.9.2, and Metal
compiler 32023.101. Its SDK query failed and the result records
`sdk_version: null`. That evidence does not answer the current versions or
availability of CMake, Ninja, LLVM/Clang development files, Homebrew/MacPorts,
Xcode command-line selection, signing identity, deployment target, free space,
or a newer OS/SDK configuration.

Before M1 consumes dependencies, its approved manifest needs:

1. an exact CuMetal or alternate candidate revision and recorded license files;
2. separate Linux and Mac dependency closures with source, version, checksum,
   download estimate, installed/build-space estimate, and cache status;
3. an approved native Mac target and OS/SDK/Xcode matrix;
4. configure/build/install prefixes, environment isolation, rollback, and
   artifact-retention rules;
5. the smallest unchanged-source test and execution proof that will decide
   whether to adopt, selectively reuse, or build the component.

## M0 resource-change record

No package installation, package download, source clone, compiler build,
environment creation, hardware purchase, paid service, or OS upgrade occurred.
Network use was limited to small primary-source documentation, repository
metadata, pinned text files, and a `git ls-remote` revision check; no upstream
source tree or release archive was retained. All package-manager operations were
queries or simulations.
