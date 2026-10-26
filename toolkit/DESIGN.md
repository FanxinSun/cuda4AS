# cuda4AS Device Lowering Design Note (N5)

**Author:** cuda4AS (Antigravity session `c39c7a89-4c79-47d1-a19b-2939180473ff`)  
**Date:** 2026-10-26  
**Status:** Plan of Record for G-C1 Milestone 1 (Oracle Kernels N5–N7)

---

## 1. Lowering Target Choice: MSL Source vs. AIR Bitcode

### Options
1. **AIR (Apple Intermediate Representation / LLVM Bitcode)**:
   - *Pros:* Skips textual MSL parsing during shader compilation.
   - *Cons:* AIR is an undocumented, proprietary internal dialect of LLVM bitcode that changes silently across macOS and Xcode versions. The container format (`.metallib`), metadata nodes, and intrinsic naming conventions are unversioned and reverse-engineered. Targeting AIR directly introduces brittle failure modes on every Xcode update.
2. **MSL Source Emission (Selected)**:
   - *Pros:*
     - Official, fully documented, and guaranteed stable interface across Apple Silicon generations.
     - Human-readable, transparent, and debuggable.
     - Direct runtime diagnostic reporting: compiler refusals and warnings are captured verbatim per the standing hardware-as-instrument rule.
     - Native integration with `MTLDevice.makeLibrary(source:options:)` and offline compilation via `xcrun -sdk macosx metal`.
     - Zero risk of breakage from internal LLVM bitcode metadata changes.

### Decision
**We select MSL Source Emission as the lowering target.**
The translator pipeline is:
$$\text{CUDA Source } (.cu) \xrightarrow[\text{AST / LLVM IR}]{\text{Clang Frontend}} \text{cuda4AS Lowering Pass} \xrightarrow{} \text{Metal Shading Language (MSL 3.0+)} \xrightarrow{\text{Metal API / metal}} \text{Metal Pipeline State}$$

---

## 2. Core Architecture & Mapping Rules

### 2.1 Kernel Arguments & Entry Point ABI
Each `__global__` function is lowered to a Metal `kernel` function.
- Kernel parameters of pointer type (`const T*`, `T*`) map to `device const T* [[buffer(i)]]` or `device T* [[buffer(i)]]`.
- Scalar parameters map to `constant T& [[buffer(i)]]`.
- An implicit buffer argument (`[[buffer(N)]]`) is appended for the **D2 arrival-ticket counter**.
- Metal builtin attributes supply hardware grid and threadgroup coordinates:
  - `uint3 ctaid_grid [[threadgroup_position_in_grid]]`
  - `uint3 tid [[thread_position_in_threadgroup]]`
  - `uint3 ntid [[threads_per_threadgroup]]`
  - `uint3 nctaid [[threadgroups_per_grid]]`
  - `uint simd_lane [[thread_index_in_simdgroup]]`
  - `uint simd_group [[simdgroup_index_in_threadgroup]]`

### 2.2 Decision D2 Mechanism: Arrival-Ticket `blockIdx` Remap
Metal threadgroups dispatch out of grid order (measured at 69.4%–100% out-of-order in S1c). To eliminate deadlock in CUB decoupled look-back and preserve forward progress by construction:
```metal
threadgroup uint cuda4as_block_ticket;
if (tid.x == 0 && tid.y == 0 && tid.z == 0) {
    cuda4as_block_ticket = atomic_fetch_add_explicit(cuda4as_arrival_counter, 1u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint3 blockIdx;
blockIdx.x = cuda4as_block_ticket % nctaid.x;
blockIdx.y = (cuda4as_block_ticket / nctaid.x) % nctaid.y;
blockIdx.z = cuda4as_block_ticket / (nctaid.x * nctaid.y);
```
All kernel code referencing `blockIdx` uses this remapped `blockIdx`.

### 2.3 Decision D3 Mechanism: Device Pointer Model
- Device pointers are 64-bit addresses (`MTLBuffer.gpuAddress`).
- Metal 3 supports raw GPU virtual addressing.
- Host runtime marks all allocated buffers resident before command buffer commitment:
  `[encoder useResource:buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite]`.
- Pointer dereferences loaded from memory in device code are directly valid `device T*` pointers.

### 2.4 Decision D1: FP64 `soft64` Arithmetic
- `double` is mapped to IEEE soft-double (`soft64`) representation using two 32-bit unsigned integers (`uint2 { lo, hi }`).
- Memory storage is bit-exact IEEE binary64.
- Arithmetic (`+`, `-`, `*`, `/`, `fma`, `sqrt`) operates with round-to-nearest-even (RN).

### 2.5 Barriers and Synchronization
- `__syncthreads()` maps to:
  `threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);`
- `__threadfence()` and acquire/release semantics map to:
  `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);`
  *(MSL 3.0–3.2 refuse `memory_order_acquire` and `memory_order_release` verbatim).*

### 2.6 Warp Intrinsics (1:1 Warp $\leftrightarrow$ SIMD-Group at 32 Lanes)
- `__shfl_down_sync(mask, v, delta)` $\to$ `simd_shuffle_down(v, delta)`
- `__shfl_sync(mask, v, src)` $\to$ `simd_shuffle(v, src)`
- `__ballot_sync(mask, pred)` $\to$ `as_type<uint4>(simd_ballot(pred)).x`
- `__any_sync(mask, pred)` $\to$ `simd_any(pred)`
- `__all_sync(mask, pred)` $\to$ `simd_all(pred)`

### 2.7 Atomics
- **32-bit Integer Atomics**: `atomicAdd(int*)` and `atomicAdd(uint*)` map to:
  `atomic_fetch_add_explicit((device atomic_int*)addr, val, memory_order_relaxed);`
- **32-bit Float Atomics**: `atomicAdd(float*)` maps to native Metal 3:
  `atomic_fetch_add_explicit((device atomic_float*)addr, val, memory_order_relaxed);`
- **64-bit Atomics & `atomicAdd(double*)`**:
  Lowered to a lock-backed critical section with SIMD-lane serialization to prevent intra-warp spin-lock deadlocks (G-C0 / S1b validated).

### 2.8 Shared Memory & $>32\text{ KiB}$ Spill Transform
- Static/dynamic `__shared__` totaling $\le 32,768$ bytes maps to:
  `threadgroup T shared_var[N];`
- Static/dynamic `__shared__` exceeding $32\text{ KiB}$ (e.g. `shared_48kb.cu`):
  Lowered to a device memory spill buffer allocated by the runtime of size $(\text{grid\_size} \times \text{shared\_size})$, indexed by:
  `&spill_buffer[cuda4as_block_ticket * SHARED_BYTES + offset]`.

### 2.9 Intra-Warp Spin Lock Detection
Apple Silicon GPUs do not possess independent thread scheduling (SIMT lockstep per SIMD-group).
The translator inspects loops containing memory reads and barrier-less back-edges:
If an intra-warp spin-lock condition is identified, the compiler issues a diagnostic refusal:
`"error: intra-warp spin lock detected; Apple GPU SIMD-groups execute in lockstep without independent thread scheduling"`.
