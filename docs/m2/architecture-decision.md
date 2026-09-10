# ADR: cuda4AS Native AOT Core v1

**Status:** Selected replacement architecture; the bounded M2A vector-add slice
is implemented under a new scoped handover. Broader M2 work remains outside this
record.

**Date:** 2026-09-10 (Asia/Shanghai)

**Decision owner:** cuda4AS Planning (Manager+Executor)

**Scope:** This record closes the bounded M2 entry gate. It is an architecture
decision and implementation handoff, not an Apple-GPU pass, a compatibility
claim, or authorization to implement the M2 slice.

## Decision

Adopt **cuda4AS Native AOT Core v1** as the replacement base for the next,
separately authorized M2 implementation slice. Keep CUDA source and ordinary
build inputs at the product boundary. Compile native arm64 host code and a
device-only typed input with the pinned Homebrew Clang CUDA frontend, lower the
device program through a cuda4AS-owned typed SSA device IR, emit initial MSL,
compile it with the official Xcode `metal`/`metallib` tools, and execute a
versioned native AOT package through an Objective-C++ Metal runtime. The device
link phase must be a real whole-program link that resolves external
`__device__` calls before backend packaging.

The first implementation profile is one inventoried Apple GPU with FP32 and
int32. FP64, including VF64, is explicitly unsupported until a separately
accepted numerical and provenance design exists. The design does not include
SASS, CUDA-binary compatibility, runtime compilation, CPU fallback,
multi-GPU, or remote execution.

CuMetal may inform an interface or provide an individually reviewed
Apache-2.0 mechanism, but it is not the replacement's dependency or source
base. The VF64 source is excluded from the replacement and remains under its
existing distribution hold.

## Evidence and rejected immediate base

The inventory-bound M2 retry used the same Mac and options as M1:

- Candidate `Lulzx/cuda-metal`, revision
  `f486e5ebcfd381d06e3297afd65dbcbd5006a902`.
- VF64 reference revision `729021777455da72db8809d9ef1269c677d88b3f`,
  extracted only by the existing M1 package and not adopted here.
- Exactly one candidate-owned patch, `cumetal-lower_to_llvm-direct-array-include-v1`,
  adding a direct `#include <array>` to
  `compiler/ptx/src/lower_to_llvm.cpp`.
- Original target SHA-256
  `df01bfcd8e0774d166bc9b56548589c4dc1840cdcd8123f2ca9662fd3581fcb2`;
  patched target SHA-256
  `4db44ce15f96d9066277d646c113676941685dba32f3c899d5b0b43ddca6678a`.
- Clean tree: 1,065 files, 16,111,130 bytes,
  manifest `b5ee0a6c8c695b4dc9e0c6527566340b18b338a223a2cd81392b81112b96a37c`.
- Patched tree: 1,065 files, 16,111,147 bytes,
  manifest `504067b6724d218a2a77182f004045b0f10069674b67e4dd02a5f1c85d8a8de0`.
- Corrected retry delta: 137,793 bytes,
  SHA-256 `1ffb31b06c0800b11225fa730908404604923e89cf5847e82e883e7ecb26525e`.
- Returned Mac evidence: 169,070 bytes,
  SHA-256 `5a00d53ab7e4e5e4075c025825b031769d08ed84eaa03a2a6a1b0bb9967432cc`,
  run `m2-entry-20260910T151705Z`, runner exit `1`.

The corrected return passed package integrity, inventory binding, the exact
single-patch binding, clean and patched tree identities, fixture-before/after
checks, preflight, and CMake configure. The full 117-target candidate build
then failed at `runtime/metal_backend/metal_backend.mm` before any enrolled
case ran. The compiler emitted four independent API errors under the bound
Xcode 15.2/SDK 14.2 headers:

1. `MTLCompileOptions` has no `mathMode` property at line 1088.
2. `MTLMathModeSafe` is undeclared; the available declaration is
   `MTL::MathModeSafe`.
3. `MTLMathModeFast` is undeclared; the available declaration is
   `MTL::MathModeFast`.
4. `MTLPipelineOptionBindingInfo` is undeclared; the available declaration is
   `MTL::PipelineOptionBindingInfo`.

The exact build command and all diagnostics remain in the unchanged returned
archive and its repository-side analysis at
`RESULTS/m2/native-entry-20260910T151705Z.analysis.json`. The three required
cases are independently recorded as `NOT_RUN`: `oracle.vector_add`,
`integration.minimal_cmake_cuda`, and `integration.multi_tu_device_link`.
There is no Apple-GPU launch, output comparison, device-link result, or
`PASS_GPU`.

An earlier return from the first published package stopped at an order-only
file-list comparison before compilation. That harness defect was repaired by
sorting both sides of the file-set comparison; the earlier archive remains
unchanged and is not used as build evidence. The corrected return is the
decision record. Rejecting the exact CuMetal pin as the immediate M2 base does
not prove that every CuMetal mechanism is infeasible; it records that another
candidate-source repair would exceed the one-patch entry gate and that no
validated pass was obtained.

## Compatibility boundary and ownership

| Boundary | Native AOT Core v1 contract | Initial owner |
| --- | --- | --- |
| CUDA product input | Unchanged `.cu`, headers, ordinary CMake/build inputs, and declared CUDA options enter the driver. Application fixtures are never rewritten to fit the backend. | Driver/build integration |
| Frontend | Pinned Homebrew LLVM/Clang provides native arm64 host compilation and device-only typed LLVM/NVVM input. A minimal pinned CUDA header/runtime discovery surface is explicit; a host compiler version alone is not the contract. | `compiler/driver` and frontend adapter |
| Device IR | cuda4AS-owned typed SSA IR records CUDA address spaces, pointer provenance, types, call graph, source locations, kernel metadata, numerical mode, and explicit unsupported capabilities. It has a versioned serialization/ABI contract. | `compiler/device_ir` |
| Legalization | Target-independent passes lower CUDA semantics to the IR while preserving diagnostics and rejecting unsupported operations. No silent CPU or approximate route is allowed. | `compiler/passes` |
| Metal backend | Initial backend emits reviewable MSL source and invokes Xcode `metal` then `metallib`. Direct AIR generation is deferred until evidence justifies it. | `compiler/msl_backend` |
| Kernel package/ABI | A versioned package carries entry points, argument layout, pointer/address-space descriptors, globals, shared/scratch requirements, capabilities, numerical mode, and source/build identity. | `compiler/package` |
| Device link | A whole-program device linker merges modules, canonicalizes symbols, resolves external `__device__` calls, checks address-space/type compatibility, rejects unresolved/duplicate/unsupported symbols, and produces a non-empty device image. | `compiler/device_link` |
| Host/runtime boundary | Objective-C++ owns device enumeration, buffers, copies, kernel registration, launch, stream/event ordering, error propagation, and lifetime. It loads only the declared AOT package. | `runtime/metal` |
| Build/deployment | CMake/Ninja produce host arm64 code, device objects, a real device-linked package, `metallib`, and a native executable/library with inspectable provenance. | `cmake` and `tools/m2` |
| Validation | Independent validators check unchanged inputs, stage evidence, Apple-GPU identity, exact outputs, unsupported-route absence, resource bounds, and aggregate classifications. | `tests/m2` and result analyzers |

The driver owns source/profile selection and rejects unsupported capabilities
before code generation. The IR owns semantic identity rather than adopting a
backend-specific AST. The backend owns MSL syntax/legalization, while the
runtime owns Metal object lifetimes and submission. No layer may quietly take
over another layer's semantics or turn an unsupported operation into a CPU
call.

## Toolchain and deployment flow

The proposed flow is:

1. Discover and record the pinned Clang, minimal CUDA headers/runtime surface,
   Xcode SDK, `metal`, `metallib`, and arm64 target. Refuse an unbound toolchain
   change.
2. Parse CUDA source through the supported Clang frontend. Produce native host
   objects and a device-only typed input; do not consume SASS or a CUDA binary.
3. Import device operations into the cuda4AS typed SSA IR. Preserve source
   locations, address spaces, calls, kernel metadata, and numerical mode.
4. Run semantic legalization and explicit capability checks. Unsupported FP64,
   VF64, runtime compilation, and other deferred features produce a visible
   diagnostic.
5. Link all device modules with the real device-link phase, then lower the
   linked program to MSL and invoke the Xcode tools to create `metallib`.
6. Package the `metallib`, ABI metadata, entry-point table, capability profile,
   and provenance under a versioned AOT package ID.
7. Compile/link native arm64 host code with the Objective-C++ runtime. At load,
   enumerate exactly one declared Apple GPU, load the AOT library, create
   pipelines, bind buffers, submit work, and report streams/events/errors.
8. Validate the exact output and all route/provenance/resource assertions in a
   separate result record. A successful process without Apple-GPU evidence is
   not a pass.

There is no runtime PTX or source compiler, no CPU fallback, and no implicit
download or installation in this flow. A cache, if introduced later, is keyed
by compiler/runtime ABI, device capability, numerical mode, and relevant SDK
identity; a stale cache is a validation failure.

## Device-link design

Each device module emits a symbol table and typed call graph containing the
canonical name, definition/declaration kind, parameter and return types,
address spaces, source location, kernel metadata, and numerical mode. The
linker then:

1. loads all modules named by the build graph;
2. canonicalizes names and ABI records without discarding address-space
   qualifiers;
3. merges globals and kernel metadata with duplicate-definition checks;
4. resolves every external `__device__` reference to one compatible definition;
5. rejects unresolved, ambiguous, unsupported, or incompatible references with
   source-linked diagnostics;
6. lowers the resolved whole program to MSL and records the input module set,
   resolved symbols, and output hash.

The linker must produce an actual device image or a visible failure. An empty
host object or a successful no-op `-dlink` command cannot satisfy this contract.
`integration.multi_tu_device_link` is the first mandatory oracle for this
boundary.

## First five oracle dependencies

The first implementation slice follows the planning corpus and keeps the
existing exact references immutable:

| Oracle | Dependency question answered | Required initial profile |
| --- | --- | --- |
| `oracle.vector_add` | End-to-end frontend, typed IR, buffer allocation/copy, one kernel entry, launch, event/error handling, and exact output. | FP32, one device, direct AOT |
| `oracle.saxpy` | Pointer/stride argument ABI, scalar parameters, elementwise arithmetic, and repeatable launch ordering. | FP32, one device, direct AOT |
| `oracle.reduce_sum` | Shared/local storage, reduction control flow, barriers, accumulation, and exact numerical policy. | FP32, one device, direct AOT |
| `oracle.matmul_tiled` | Tiled indexing, shared storage, synchronization, kernel metadata, and larger argument/resource descriptions. | FP32, one device, direct AOT |
| `oracle.histogram_atomics` | Atomic operation lowering, contention, visibility, error handling, and output determinism. | int32/FP32 as declared by the immutable oracle |

These five depend on the driver, typed IR, MSL backend, AOT package/ABI,
Objective-C++ runtime, and validator. The M1 CMake vector-add case remains a
separate build-integration oracle, and the M1 multi-TU case remains the device
link gate; neither is removed when the five direct cases are added.

## Validation and performance plan

Every slice keeps positive, boundary, and rejection coverage. The initial
correctness ladder is:

- IR unit cases for address spaces, pointer merges, call graphs, source
  locations, kernel metadata, FP32/int32 modes, and unsupported-capability
  diagnostics;
- frontend/build cases for unchanged CUDA input, dependency discovery,
  response files, native host compilation, and separately observable device
  compilation/link stages;
- a real multi-module device-link fixture with an external `__device__` call;
- direct AOT runs for the five oracles plus the unchanged M1 fixtures;
- exact byte comparisons, Apple-GPU provenance, route checks, and failure
  records for every case.

Performance is diagnostic until correctness passes exist. Record cold and warm
compile/package time, process startup, buffer copies, first launch, subsequent
launches, synchronization, peak task-local space, and relevant memory sizes.
Use a matched native Metal or MLX comparator on the same Mac only after a
correct Apple-GPU run. Do not turn a compile failure, a reference-environment
run, or a partial pass into a speed or broad compatibility claim.

## Provenance and license boundary

The M1 candidate and VF64 archives remain historical evidence and are not
copied into the replacement implementation. The separate VF64 repository had
no verified license-named file or GitHub license declaration in the M1 audit;
this record makes no legal conclusion and keeps its distribution hold. Any
CuMetal mechanism considered for reuse must be identified at file/function
granularity, independently reviewed for its Apache-2.0 terms, and reimplemented
or imported only under a separately recorded decision. A source resemblance or
an upstream claim is not a license clearance.

## Risks, resources, and estimate

The principal risks are CUDA header/frontend drift, incomplete pointer and
address-space semantics, MSL type and synchronization differences, a weak
device-link ABI, Objective-C++ Metal SDK variation, numerical behavior,
resource limits on a 16 GiB Mac, and accidental dependence on unreviewed
upstream/VF64 code. The M2 design must stop and revisit the architecture if a
shared mechanism requires a hidden fallback, cannot preserve provenance, or
cannot produce a real device-linked image.

The recorded Mac is an arm64 MacBookPro18,3 / Apple M1 Pro with macOS 14.4
build 23E214, Xcode 15.2 build 15C500b, SDK 14.2, CMake 4.3.3, Ninja 1.12.1,
Homebrew LLVM 23.1.0, LZ4 1.10.0, and Zstd 1.5.7. Keep the existing user-run
limits: at least 5 GiB free before a native run, at most four jobs, no new
installation/download without a new decision, and no more than 20 GiB of
task-local use. The current corrected run used no network, installation, or
`sudo`; its recorded free space was 79,476,252 KiB and task tree was 36,908
KiB.

The planning estimate for a replacement path is a high-uncertainty
**3–8-week architecture-bootstrap slice**, followed by a re-estimate of the
first useful M2 implementation. A provisional **4–8 weeks after bootstrap**
for the first five-oracle vertical slice is a planning range only; it is not a
commitment or authorization to implement it in this entry-gate task.

## Stop and review gates

The follow-on work must stop for manager review at each gate:

1. typed IR can represent and round-trip the vector-add kernel with source and
   capability metadata;
2. one FP32 kernel emits MSL/metallib and launches through the AOT runtime with
   exact output and Apple-GPU evidence;
3. the real device linker resolves the unchanged multi-TU external device call;
4. all five direct oracles pass with unchanged inputs and explicit resource
   records;
5. matched cold/warm timings and an anchor-application probe justify the next
   scope.

Stop immediately on a second unscoped source patch, a changed fixture or
profile, absent GPU provenance, CPU/fallback execution, approximate output,
fake/empty device linking, a new install/download requirement, an unreviewed
license dependency, or resource use beyond the declared bound.

## Explicit non-goals

This ADR does not implement the compiler, IR, linker, runtime, backend, CMake
integration, five kernels, anchor application, performance optimization,
PyTorch support, M3 semantics, residency, FP64/VF64, multi-GPU, remote
execution, runtime compilation, CUDA binary compatibility, SASS support,
release packaging, or distribution licensing. It is not an Apple-GPU pass and
does not claim implemented CUDA compatibility.

The concrete work sequence, proposed ownership paths, Mac cadence, and review
artifacts are in [`m2-handoff-proposal.md`](m2-handoff-proposal.md).
