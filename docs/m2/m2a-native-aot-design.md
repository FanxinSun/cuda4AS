# M2A Native AOT Core v1 implementation record

This record applies the selected Route B architecture to the single approved
`oracle/src/vector_add.cu` slice. The source and `oracle/src/oracle.h` remain
byte unchanged. The implementation is deliberately small and source-driven;
it is not a general CUDA compatibility layer.

## Owned boundaries

`tools/m2a/native_aot.py` tokenizes and parses the supported CUDA kernel,
retains source locations and CUDA address spaces, builds versioned typed SSA
values and explicit control-flow operations, verifies capabilities, imports a
real Clang CUDA AST JSON device translation unit, and serializes deterministic
IR/ABI/MSL/device-link records. `tools/m2a/runtime.mm` owns the Objective-C++
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
