> **Current implementation history, not an execution plan.** The Native AOT
> design remains accepted. Portable M2A-R1 corrections through commit
> de69952dc986173d0b30cca77f55c07b4c14b0ed are accepted only as frontend
> contract evidence; the historical native package failed at device import and
> live compiler/native/GPU validation is NOT_RUN. Current gates and status are
> in [plan/PLAN.md](../../plan/PLAN.md).

# M2A Native AOT Core v1 implementation record

This record applies the selected Route B architecture to the single approved
`oracle/src/vector_add.cu` slice. The source and `oracle/src/oracle.h` remain
byte unchanged. The implementation is deliberately small and source-driven;
it is not a general CUDA compatibility layer.

## Owned boundaries

`tools/m2a/native_aot.py` discovers the pinned Clang frontend and passes the
unchanged CUDA translation unit through
`tools/m2a/clang_cuda_facade/cuda_runtime.h`. The AST importer in
`tools/m2a/clang_ast.py` traverses typed compiler records for kernel identity,
parameters, address spaces, expressions, control flow, built-ins, and source
locations; it then builds versioned typed SSA values and explicit control-flow
operations, verifies capabilities, and serializes deterministic IR/ABI/MSL/
device-link records. The source tokenizer is limited to host launch discovery
and an explicitly `NOT_RUN` packaging path. `tools/m2a/runtime.mm` owns the Objective-C++
Metal device, buffer, command queue, AOT library, pipeline, launch, completion,
copy, exact output validation, and error/lifetime path. It has no CPU kernel
fallback. `tools/m2a/run-m2a-native.sh` is the bounded user-operated Mac drop;
it invokes `metal` and `metallib`, compiles the native host, and archives all
stage evidence. The return analyzer validates archive safety, inventory
identity, source hashes, stage records, output bytes/hash, and GPU provenance.

The current IR schema is `cuda4as-device-ssa-ir-v1`; ABI is
`cuda4as-native-aot-abi-v1`; link records are `cuda4as-device-link-v1`.
The device link emits a non-empty inspectable JSON linked image whose identity
is tied to the IR hash and symbol table. It is a real single-module link record
and does not claim the deferred multi-translation-unit gate.

The facade is versioned `1.0.0`, maps CUDA annotations to Clang attributes,
declares typed `dim3` execution built-ins with `device_builtin`, and exposes
only the host runtime declarations needed by the unchanged oracle. Clang
accepts the `device_builtin` spelling for CUDA frontend semantics, but LLVM
defines that attribute as ignored (`ASTNode=0`), so it does not appear as an
AST JSON child. Builtin identity therefore comes from the exact referenced
declaration ID, a direct emitted `CUDADeviceAttr`, the pinned `const dim3`
type, and the manifest's immutable facade header hash plus line/source-anchor
contract. Same-name user declarations, missing or conflicting attributes,
wrong types, and locations outside the verified facade fail closed. Its
`manifest.json` binds the header SHA-256, versioned builtin declaration
contract, LLVM Clang support range (18 through 23), intended Homebrew LLVM
23.1.0 identity, and its `NOT_RUN` validation status, `sm_86` target, and
exact frontend argument template. The live frontend uses
`-nocudainc -nocudalib -I tools/m2a/clang_cuda_facade` and
`-Xclang -ast-dump=json`. A compiler-output fixture is available for portable
importer tests. The fixture is explicitly synthetic schema-conformance data,
with a sidecar recording its inputs and non-capture status; it is not native
compiler validation. It models declaration `loc`, statement/expression
`range`, `referencedDecl` identity, direct `inner` `*Attr` records under the
emitted C linkage wrapper, and Clang's stateful file/line elision. The importer
normalizes that serializer state in emitted order and labels recovered
coordinates `exact_deelided` (or `exact_offset_deelided`); coordinates supplied
only by semantic parent fallback are labelled `inherited`. Missing first state,
wrong capture working directories, unknown source files, and offset/line
conflicts fail closed.

Clang `createQualType` supplies `qualType` (and optional desugaring/alias IDs),
not a fabricated `addressSpace` member. Plain pointer parameters therefore
retain the compiler fact `address_space=unknown`; explicit textual CUDA address
spaces are preserved when present. The backend applies the separate, pinned
`cuda4as-vector-add-kernel-abi-v1` lowering record: only the four-argument
`vector_add` kernel's `unknown` or `global` pointer facts lower to global CUDA
ABI/device Metal buffers. Generated ABI records carry the semantic fact,
lowered address space, rule version, and `EXPLICIT_PORTABLE_RULE` provenance;
generic, constant, shared, and other unsupported pointer spaces are rejected.
Fixture emission records `validation_status=FIXTURE` and
`compiler_validation_status=NOT_RUN`.
The generated `aot-manifest.json` repeats this separation under
`abi_address_space_contract`: `compiler_facts` are imported Clang facts and
`arguments`/`lowering` are the explicit backend transformation record.
`--skip-clang` emits a visible `NOT_RUN` record and returns 77, so it cannot
satisfy the frontend or M2A validation gate.

## Supported profile and explicit stops

The first profile contains FP32 global loads/add/store, int32 index/bounds
arithmetic, `blockIdx.x`, `blockDim.x`, `threadIdx.x`, four arguments, and the
recorded one-dimensional launch. FP64/VF64, atomics, barriers, shared memory,
multi-TU device calls, runtime compilation, and CPU fallback are explicit
capability rejections. Changes to source operands or launch block size alter
the parsed IR, ABI, MSL, and source identity; no fixture filename/hash selects
generated code.

## Mac contract

The package binds the validated M1 inventory (arm64 MacBookPro18,3, Apple M1
Pro, macOS 14.4/23E214, Xcode 15.2/SDK 14.2, Homebrew LLVM 23.1.0). The runner
uses at most four jobs, requires at least 5 GiB free, stays below 20 GiB of
task-local space, and stops at 30 minutes. A device name/registry mismatch is a
failure or environment gate; it cannot become a pass. Timings are diagnostic
only. The package performs no network, install, update, sudo, or remote action.
