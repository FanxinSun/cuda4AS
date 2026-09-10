# Proposed M2 handoff — Native AOT Core v1 first vertical slice

**State:** Tracked implementation handoff for the separately authorized M2A
slice. The first vector-add path is implemented under `tools/m2a`; M2B/M3 remain
outside scope.

**Prerequisite:** Manager acceptance of
[`architecture-decision.md`](architecture-decision.md) and a new scoped user
authorization for implementation.

## First vertical slice

Build one complete, inspectable path for unchanged `oracle.vector_add`:

1. Discover the pinned arm64 Clang/Xcode tools and the minimal declared CUDA
   header/runtime surface.
2. Parse the CUDA source into cuda4AS typed SSA device IR while separately
   producing native arm64 host objects.
3. Validate and serialize the IR with address-space, call-graph, source,
   kernel-metadata, numerical-mode, and capability records.
4. Lower the supported FP32 vector-add subset to MSL, invoke `metal` and
   `metallib`, and create a versioned AOT package.
5. Load the package through the Objective-C++ runtime, allocate buffers, copy
   inputs, launch one kernel, synchronize, report errors, and validate the
   immutable exact output.
6. Run the same package through the real device-link pipeline even when the
   first single-source case has no external device call; preserve the link
   record for the subsequent multi-TU gate.

The slice ends at one exact, inventoried Apple-GPU pass or a documented gate
failure. It does not broaden the supported CUDA surface to make the case pass.

## Proposed ownership paths and interfaces

These are proposed ownership locations for a future implementation; creating
them is outside this entry-gate task.

| Path/component | Responsibility and first interface |
| --- | --- |
| `compiler/driver/` | CUDA input/profile discovery, response files, host/device split, pinned header checks, capability diagnostics, and build provenance. |
| `compiler/device_ir/include/cuda4as/device_ir/` | Versioned typed SSA types for values, pointers/address spaces, calls, source locations, kernel metadata, numerical modes, and unsupported capabilities. |
| `compiler/device_ir/src/` | Parser/importer, verifier, serializer, and deterministic IR dump. |
| `compiler/passes/` | Address-space, pointer, intrinsic, synchronization, and capability passes with explicit rejection diagnostics. |
| `compiler/device_link/` | Module symbol tables, canonical names, whole-program resolution, duplicate/unresolved checks, linked metadata, and output identity. |
| `compiler/msl_backend/` | Supported IR-to-MSL lowering, MSL diagnostics, `metal`/`metallib` invocation, and package inputs. |
| `compiler/package/` | AOT package manifest, kernel entry ABI, argument/pointer descriptors, capabilities, numerical mode, and provenance. |
| `runtime/metal_aot/` | Objective-C++ device/buffer/copy/registration/launch/stream/event/error/lifetime implementation and AOT loader. |
| `cmake/Cuda4ASNativeAOT.cmake` | Native host/device build graph, observable device compile and real device-link stages, SDK/tool discovery, and dependency files. |
| `tests/m2/native_aot/` | Portable IR/link/package tests and Mac-run harnesses; fixture inputs stay under the existing immutable M0/M1 paths. |
| `tools/m2/` | User-operated Mac drop, safe return archive, inventory binding, analyzer, and exact-output validator. |
| `docs/m2/` | Versioned contracts, architecture decisions, handoff records, limits, and milestone reports. |

The frontend owns source interpretation, the IR owns semantics, the backend
owns MSL legality, the linker owns whole-program device symbols, and the
runtime owns Metal lifecycle and submission. Cross-boundary data structures
must be versioned and tested; no component may silently replace a failed
device operation with host execution.

## Reference cases and dependency order

The five initial M2 oracle programs from the planning corpus are
`oracle.vector_add`, `oracle.saxpy`, `oracle.reduce_sum`,
`oracle.matmul_tiled`, and `oracle.histogram_atomics`. Keep these existing
references and source hashes immutable. Retain the three M1 entry cases as
independent integration gates:

- `oracle.vector_add` exercises direct AOT and the host/device/runtime path.
- `integration.minimal_cmake_cuda` exercises unchanged CMake discovery and a
  normal CUDA-language project.
- `integration.multi_tu_device_link` exercises relocatable compilation and a
  real external `__device__` call across translation units.

The proposed execution order is:

1. vector-add IR/ABI/backend/runtime vertical slice;
2. SAXPY pointer/scalar/stride ABI;
3. reduction shared storage and barriers;
4. tiled matmul resource and synchronization behavior;
5. histogram atomics and visibility;
6. unchanged minimal CMake vector-add;
7. unchanged multi-TU device link;
8. only then an anchor-application configure/build probe.

Each case receives an independent stage record, exact output comparison,
Apple-GPU provenance, unsupported-feature status, and resource record. A
single pass cannot erase a later device-link or application failure.

## User-operated Mac cadence

The executor prepares one small deterministic drop per accepted gate. Each drop
must include its package size/SHA-256, source/build identities, exact commands,
expected duration, disk/memory bound, stop conditions, return archive name,
and WSL destination. The user runs the Mac command in the visible session and
returns exactly one archive unchanged. The repository analyzer then validates
all positive, negative, failed, and inconclusive results before the next drop.

No remote control, hidden process, automatic installation, `sudo`, or large
download is part of this cadence. The current M1 Pro inventory is the first
target. A changed OS, SDK, compiler, or GPU requires a new inventory record and
an explicit binding decision.

## Required resources

- Existing Mac: arm64 MacBookPro18,3 / Apple M1 Pro, 16 GiB, macOS 14.4
  build 23E214, Xcode 15.2 build 15C500b, SDK 14.2.
- Existing tools: CMake 4.3.3, Ninja 1.12.1, Homebrew LLVM 23.1.0, LZ4
  1.10.0, and Zstd 1.5.7.
- Native run bound: at least 5 GiB free before starting, at most four jobs,
  and no more than 20 GiB task-local use; stop at 30 minutes unless a new
  decision changes the bound.
- Portable validation may run in WSL, but Apple frameworks, `metal`,
  `metallib`, native deployment, and GPU execution remain Mac-only evidence.
- No new package, toolkit, paid service, or download is assumed. Ask the user
  directly if a concrete dependency exceeds the approved resource boundary.

## Stop/review gates

The manager reviews the complete returned record before the next gate. Stop and
return the evidence if any of these occur:

- the frontend cannot produce typed IR without changing application inputs;
- address spaces, pointer provenance, source locations, calls, or kernel
  metadata are dropped or guessed;
- MSL output requires runtime compilation, hidden CPU work, or approximate
  arithmetic;
- device-link is an empty object, leaves an external symbol unresolved, or
  lacks an inspectable linked image;
- the runtime cannot show an inventoried Apple-GPU launch and exact output;
- FP64/VF64, multi-GPU, remote, SASS, CUDA-binary, or other deferred feature
  becomes a hidden dependency;
- a build requires a second candidate patch, changed fixture/profile, install,
  large download, or resource increase;
- a source or mechanism crosses the CuMetal/VF64 license boundary without a
  separately reviewed provenance decision.

The first vertical-slice review asks whether the IR/ABI and backend boundaries
are stable enough to continue. The device-link review asks whether ordinary
multi-file CUDA can be represented honestly. The five-oracle review asks
whether FP32/int32 correctness and resource costs justify an anchor probe.

## Validation and performance deliverables

Every gate returns:

- immutable fixture-input hashes and unchanged-state assertions;
- source, patch, toolchain, SDK, and machine identities;
- separate configure, host compile, device compile, device-link, native-link,
  launch, and validation records;
- AOT package and kernel ABI hashes;
- Apple-GPU registry identity and backend provenance;
- exact expected/actual bytes and numerical assertions;
- resource usage, cold/warm timings, and all failures or inconclusive stages.

Timing is diagnostic until at least one exact Apple-GPU pass exists. After that,
compare matched native Metal/MLX work on the same Mac, retaining algorithm,
shape, precision, warm/cold state, memory, and power/thermal conditions. No
performance target becomes a release gate until the comparator and workload
are frozen.

## Estimate and next authorization boundary

The replacement path first needs the high-uncertainty architecture-bootstrap
slice estimated at roughly 3–8 weeks, then a re-estimate. A provisional 4–8
weeks after bootstrap covers the first five-oracle vertical slice, subject to
the stop gates above. These ranges are planning information only.

The next authorized handoff would need to name the implementation files,
first gate, exact Mac drop, resource budget, and whether a small prototype or
tracked code is allowed. This proposal itself starts no implementation and
does not authorize M3, an anchor application build, installations, downloads,
commits, pushes, merges, tags, releases, or Git-history changes.
