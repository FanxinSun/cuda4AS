# SELF_REVIEW — the probes, and how far they were verified

Probes `p0`–`p8` were originally written on a Linux/WSL2 host before an Apple
silicon Mac was on the desk, verified first on an M1 Pro (`RESULTS/as-phase0/Studio/`)
and subsequently executed natively on the Apple M4 Mac mini (`RESULTS/as-phase0/masshjs-Mac-mini/`).
Probes `p9_lookback` and `p10_fp64` were written, compiled, and executed directly
on the M4 Mac mini (`RESULTS/s1/`) for Stage S1 of the `caloml` sample plan.

**Be honest about the basis.** Items below record how each probe was originally
constructed for the one-round-trip drop (`p0`–`p8`) as well as the direct M4
verification of `p9_lookback` and `p10_fp64`.

| verified how | items |
|---|---|
| fetched this session | Metal 4 tensor-op MSL spelling — `mpp::tensor_ops::matmul2d`, `matmul2d_descriptor(M, N, K, tl, tr, rp, mode)`, `tensor<device T, dextents<int32_t,2>, tensor_inline>`, `execution_simdgroups<N>`, static `.slice<W,H>()`, and the `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` include — from `github.com/liuliu/example_matmul_metal4`, `Sources/matmul/shader.metal`, a working example. `MTLGPUFamily.metal4 == 5002` from an enum listing. |
| read in this tree | `simd_ballot` / `popcount` / `simd_shuffle` / `[[thread_index_in_simdgroup]]` idiom and `simdgroup_barrier(mem_flags::mem_threadgroup)`, copied from `lmz/scratchpad/gpu/metal/lmz_rans.metal` — code written by someone else for this exact platform. `MTLDevice`/`MTLBuffer`/`MTLComputeCommandEncoder` usage patterns and `.storageModeShared` from `lmz/scratchpad/gpu/metal/bench.swift`. |
| API knowledge, not re-checked | everything else listed below. |

---

## Design decisions taken because there is only one round trip

1. **Two build routes in `run.sh`.** Route 1 compiles `common.swift` plus the
   probe, letting the probe's `@main` supply the entry point. Route 2
   concatenates the two into one script-mode file, defines `PROBE_SCRIPT_MODE`
   so the `@main` struct compiles out, and appends a bare `probeMain()` call.
   Different mechanisms, so a disagreement about Swift entry points costs one
   route, not the drop.

2. **`p1` split into three executables.** `p1_heap` (buffer bisect,
   `gpuAddress`, `bytesNoCopy`, address coincidence) has no mach VM call in it
   at all. The alias experiment — the highest-risk code here and the single
   most important G-C0 question — lives in `p1b_vmremap` and `p1c_machvmremap`,
   which ask it through `vm_remap` and `mach_vm_remap` respectively. If one
   spelling will not build against this SDK the other may, and either way
   `p1_heap`'s four answers survive.

3. **`p1b`/`p1c` write their JSON twice.** Once with the `kern_return_t` alone,
   before dereferencing an address the kernel has just handed back, and once
   with the readback. A fault in the second half cannot destroy the first
   half's answer.

4. **Raw values, not case names.** `MTLGPUFamily(rawValue:)` and
   `MTLLanguageVersion(rawValue:)` are constructed from integers, never written
   as `.apple9` or `.version4_0`. Naming a case an older SDK has never heard of
   is a compile error that would cost the round trip; a raw value an SDK does
   not know simply returns `nil` and is reported as unknown.

5. **Every runtime MSL compile is failure-isolated.** `compileMSL` returns the
   compiler's message instead of throwing, and probes that ask several
   independent questions (`p5`, `p7`) compile a separate library per question,
   so one refusal costs one row. A refusal is written into the JSON verbatim,
   because for `p6` and `p7` the refusal *is* the result.

6. **`Darwin` is imported by three probes.** The acceptance note asked for
   `Foundation` + `Metal` only; `p1_heap` needs `mmap`/`munmap`/`getpagesize`
   and `p1b`/`p1c` need `vm_remap`/`mach_vm_remap`/`mach_task_self_`, which are
   the mechanisms those probes exist to measure. No other probe imports it.

---

## Per probe

### `common.swift` — shared
`MTLCreateSystemDefaultDevice`, `MTLDevice.name / registryID / hasUnifiedMemory
/ recommendedMaxWorkingSetSize / maxBufferLength / maxThreadgroupMemoryLength /
maxThreadsPerThreadgroup / supportsFamily(_:)`, `MTLCompileOptions.languageVersion`,
`MTLDevice.makeLibrary(source:options:)`, `makeComputePipelineState(function:)`,
`MTLCommandBuffer.gpuStartTime/gpuEndTime`; Foundation's `Process`/`Pipe`,
`ProcessInfo.physicalMemory / hostName / operatingSystemVersionString /
systemUptime`, `ISO8601DateFormatter`, `FileManager`.
JSON is hand-rolled so key order is deterministic and no `Codable` synthesis is
involved. `physicalMemory` is used as the primary RAM figure with `sysctl -n
hw.memsize` recorded beside it, rather than calling `sysctlbyname` and taking a
dependency on the Darwin overlay from the shared file.
*Risk:* low. **Watch:** `MTLArgumentBuffersTier.rawValue` is `UInt` and is
wrapped in `Int(...)` at the one place it is used.

### `p0_device`
Adds `MTLCopyAllDevices`, `isLowPower / isRemovable / isHeadless`,
`supports32BitFloatFiltering`, `supportsShaderBarycentricCoordinates`,
`supportsDynamicLibraries`, `supportsFunctionPointers`, `supportsRaytracing`,
`maxArgumentBufferSamplerCount`, `currentAllocatedSize`,
`MTLComputePipelineState.threadExecutionWidth / maxTotalThreadsPerThreadgroup /
staticThreadgroupMemoryLength`.
*Risk:* low, and the probe is the one most likely to survive if others do not,
so it is first in `run.sh`.

### `p1_heap`
`makeBuffer(length:options:)` bisected down from `maxBufferLength` (never
touched, so pages stay unbacked and the machine is not pushed into swap),
`MTLBuffer.gpuAddress`, `contents()`,
`makeBuffer(bytesNoCopy:length:options:deallocator:)` over an `mmap`ed
page-aligned region, `dispatchThreads`.
*Risk:* medium. `mmap`'s result is compared by bit pattern against −1 rather
than against `MAP_FAILED`, whose Swift typing has moved between SDKs.
`dispatchThreads` needs non-uniform threadgroup support, which every Apple
silicon GPU has.

### `p1b_vmremap` / `p1c_machvmremap`
`vm_remap` / `mach_vm_remap` with `mach_task_self_`; flags `0` written as a
literal for `VM_FLAGS_FIXED` and inheritance `vm_inherit_t(0)` for
`VM_INHERIT_SHARE`, so the probe does not depend on those macros being imported
into Swift.
*Risk:* **highest in the drop, and unverified.** The argument types
(`vm_address_t`/`vm_size_t` versus `mach_vm_address_t`/`mach_vm_size_t`,
`boolean_t`, `vm_prot_t`, `vm_inherit_t`) are the reason there are two of these.
If both fail to build, the compiler text in `build-p1b_vmremap.log` and
`build-p1c_machvmremap.log` is what the next iteration is written from, and
`p1_heap`'s `gpu_address_equals_cpu_address` still says whether the alias is
even needed.

### `p2_coresident`
`setThreadgroupMemoryLength(_:index:)`, `dispatchThreadgroups`,
`atomic_fetch_add_explicit` / `atomic_load_explicit` on `device atomic_uint`,
`threadgroup_barrier`.
Bounded: `SPIN_CAP_MAX = 2^20` per thread, stated in the source and in the JSON,
reduced automatically (never raised) if a dispatch exceeds 1.5 s, with the cap
actually used recorded per row. The search doubles until failure then bisects,
so no dispatch ever exceeds twice the answer.
*Risk:* medium — not of compiling, but of a long-running compute command buffer
being cut short by the OS. That would arrive as `cb.error`, which is captured
and reported per row rather than ending the probe.
**Note:** `setThreadgroupMemoryLength` takes a positive multiple of 16, so the
"0 KB" row of the requested sweep is 16 B and is labelled as such.

### `p3_lockstep`
`simd_ballot` / `simd_vote::vote_t` / `popcount` / `simdgroup_barrier` /
`[[thread_index_in_simdgroup]]`, all copied verbatim in form from lmz's shader;
`atomic_compare_exchange_weak_explicit` on `device atomic_uint`.
Bounded: `LOCK_SPIN_CAP = 2^16`. Test B terminates under lockstep *and* under
independent thread scheduling — the discriminator is how many lanes acquired
the lock, not whether the kernel finishes — so nothing here can hang.
*Risk:* low.

### `p4_bandwidth`
`uint4` loads, `dispatchThreads`, `.storageModePrivate` for the working set.
Working-set size is `min(1 GiB, maxBufferLength/4, recommendedMaxWorkingSetSize/8)`
so an 8 GB M1 and a 512 GB Studio both run it, and the size used is in the JSON.
*Risk:* low.

### `p5_simd`
Four separately compiled libraries. `bfloat` is Metal 3.1+; a refusal is
recorded as a result. Multiply-and-add is written `a*b+d` rather than `fma(...)`
deliberately: a missing `fma` overload for `bfloat` would otherwise be misread
as "this chip has no bfloat", and the FLOP count is 2 either way.
`simdgroup_half8x8`, `simdgroup_load`, `simdgroup_multiply_accumulate`,
`simdgroup_store`; the accumulators are zeroed by loading a zeroed threadgroup
tile rather than by a scalar constructor whose spelling varies.
*Risk:* medium for the simdgroup-matrix library, low for the three FMA ones, and
they cannot take each other down.

### `p6_tensorops`
The Metal 4 path, in two routes (runtime compile, and a `.metallib` built by
`run.sh` with `xcrun -sdk macosx metal`, trying `-std=metal4.1`, `metal4.0`,
`macos-metal4.0`, `metal3.2` and no flag in turn). MSL spelling from the
fetched working example above.
*Risk:* **high, deliberately.** The example's author notes the
MetalPerformancePrimitives header may not be reachable from the runtime
compiler, and a reverse-engineering paper on Metal 4.1 tensor ops reports that
language version 4.1 needs macOS 27 / Xcode 27 and refuses to load on macOS
26.5. Every one of those outcomes is a finding this probe is built to bring
back with the compiler's own words attached, and the FP16 and BF16 halves are
independent.

### `p7_fp64_atomics`
Six independently compiled cases: `double` arithmetic (expected to be refused —
the message is the deliverable), `atomic_uint` add as the control,
`atomic_ulong` add / min / compare-exchange, and `atomic_float` add. The
`atomic_ulong` add uses 2³³ per thread so a silently narrowed 32-bit atomic
cannot produce the right answer.
*Risk:* low; every case is expected to survive its own refusal.

### `p8` — lmz's decoder
Not ours and not modified. See `../drop/as-phase0/lmz/PROVENANCE.md`. It
compiles its shader at run time, so it needs no Metal toolchain;
`run.sh` builds it with `swiftc -O` and runs it from its own directory because
it looks for `lmz_rans.metal` next to itself first.

### `p9_lookback` — decoupled look-back & memory ordering (S1)
Compiled and verified directly on the Apple M4 (`RESULTS/s1/p9_lookback.json`).
Tests six properties across separate runtime-compiled MSL libraries:
1. Co-resident threadgroup measurement (`p9_measure_residency`, bounded by `spinDelay`).
2. Stock CUB tile assignment (`tile = tgid.x`) with arrival-order tracking alongside,
   demonstrating out-of-order threadgroup scheduling on M4 above co-residency.
3. Arrival-ticket remap (`tile = atomic_fetch_add(ticketCounter, 1)`), proving 0 timeouts
   at 10,240 threadgroups with `SPIN_CAP = 2^20`.
4. Adversarial reverse ticket (`tile = (N - 1) - ticket`), proving non-preemptive
   starvation once grid size exceeds co-resident capacity (160 threadgroups on M4).
5. Single-word 64-bit aligned `ulong` store/load tear test (256M reads, 0 torn reads)
   and 32-bit packed descriptor control.
6. Two-word flag+value test with `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)`
   (51.2M reads, 0 stale reads) and separate MSL compiler probes confirming that
   `memory_order_acquire` and `memory_order_release` are rejected by MSL 3.0–3.2.
*Risk:* low (bounded by `SPIN_CAP = 2^20`; verified on M4).

### `p10_fp64` — double-precision emulation (`df64` vs `soft64`, S1)
Compiled and verified directly on the Apple M4 (`RESULTS/s1/p10_fp64.json`).
Measures accuracy (`10^6` samples across `[1, 2)`, `[1e-30, 1e30]`, and subnormals/extremes),
ILP=8 throughput against an FP32 baseline (`ASSUMED_M4_FP32_PEAK_TFLOPS = 3.712`),
Celeritas physical constant representability (`h^2 = 4.39e-53`), and lock-backed
64-bit `atomicAdd(double*)` with intra-SIMD-group lane serialization over 65,536 threads.
*Risk:* low (verified on M4; motivates decision D1 to use IEEE `soft64` rather than `df64`).

---

## Lines per probe

All phase-0a probes (`p0`–`p7`) are inside the ~200-line guidance except `p2` and `p3`,
whose bodies are mostly embedded MSL and result-interpretation strings. The S1 probes
(`p9_lookback` and `p10_fp64`) are standalone multi-experiment suites developed directly on the M4.
