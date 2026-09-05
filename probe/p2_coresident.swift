// p2_coresident -- how many threadgroups are resident at once?
//
// This is the number behind `grid.sync()` (blueprint §2.8).  CUDA cooperative
// launches promise every block in the grid is running simultaneously; Metal
// promises nothing grid-wide.  A persistent-threads barrier is correct exactly
// when every threadgroup of the dispatch is co-resident, so the runtime must
// KNOW that number rather than assume it.  cuda-metal caps cooperative grids at
// one block per core because it will not probe the hardware; §2.8 says probe.
//
// Method.  N threadgroups each atomically increment a device counter, then spin
// reading it until it reaches N.  A group that sees N knows every other group
// was running at the same time.  Largest N where EVERY group saw N is the
// co-resident capacity for that (threadgroup size, threadgroup memory) pair --
// and it depends on both, which is why this sweeps a grid rather than reporting
// one number.
//
// Search: double N from 1 until a dispatch fails, then bisect.  The expensive
// dispatches are the failing ones (non-resident groups spin the full cap before
// retiring and the next wave starts), and doubling-then-bisect never dispatches
// more than twice the answer.
//
// SAFETY.  Every spin is bounded by SPIN_CAP iterations per thread and the cap
// is reported in the results; nothing here relies on the GPU watchdog to stop
// it.  If a dispatch nevertheless takes longer than SLOW_SECONDS the cap is
// reduced for later dispatches and the cap actually used is recorded per row.
//
// THE MACHINE RULE: co-residency is a property of THIS chip's core count,
// register file and threadgroup memory, and of this OS's scheduler.  It is the
// canonical example of a number that must never be a constant in source: an M1
// and an M5 Ultra differ by more than an order of magnitude, and the whole
// point of §2.8 is that the runtime measures it on the machine it wakes up on.

import Foundation
import Metal

let SPIN_CAP_MAX: UInt32 = 1 << 20      // hard ceiling, per the handover
let SPIN_CAP_MIN: UInt32 = 1 << 14
let SLOW_SECONDS = 1.5
let FALLBACK_GROUP_LIMIT = 4096          // when the GPU core count is unknown

let coresidentSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void coresident(device atomic_uint *arrived [[buffer(0)]],
                       device uint *maxSeen        [[buffer(1)]],
                       constant uint &N            [[buffer(2)]],
                       constant uint &cap          [[buffer(3)]],
                       constant uint &scratchBytes [[buffer(4)]],
                       threadgroup uchar *scratch  [[threadgroup(0)]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint tid  [[thread_position_in_threadgroup]],
                       uint tgsz [[threads_per_threadgroup]])
{
    // Occupy the threadgroup memory for real, so the allocation is not
    // something the compiler can reason away and occupancy reflects it.
    for (uint i = tid; i < scratchBytes; i += tgsz) scratch[i] = uchar(i);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint sBest;
    if (tid == 0) {
        atomic_fetch_add_explicit(arrived, 1u, memory_order_relaxed);
        uint best = 0;
        // BOUNDED: at most `cap` iterations, cap <= 2^20 (SPIN_CAP_MAX).
        for (uint i = 0; i < cap; i++) {
            uint seen = atomic_load_explicit(arrived, memory_order_relaxed);
            if (seen > best) best = seen;
            if (best >= N) break;
        }
        sBest = best;
    }
    // Hold the whole group alive until the spin ends: a group whose other
    // threads had already retired would still occupy its slot, but waiting
    // here is what a real persistent-threads barrier does.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) maxSeen[tgid] = sBest;
}
"""

struct Attempt {
    var allSawN = false
    var minSeen = 0
    var maxSeen = 0
    var seconds = 0.0
    var capUsed: UInt32 = 0
    var error: String? = nil
}

func probeMain() {
    var body: [(String, J)] = []
    let dev = acquireDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("spin_cap_max", .u(UInt64(SPIN_CAP_MAX))))
    body.append(("spin_cap_min", .u(UInt64(SPIN_CAP_MIN))))
    body.append(("method", .s("global-atomic arrival barrier; largest N where every threadgroup observed N")))

    guard let dev = dev, let queue = dev.makeCommandQueue() else {
        body.append(("status", .s("no Metal device or command queue")))
        writeResult("p2_coresident", body)
        return
    }
    let (lib, cerr) = compileMSL(dev, coresidentSrc)
    guard let lib = lib else {
        body.append(("compile_error", jOptS(cerr)))
        body.append(("status", .s("the probe kernel did not compile -- that message is the result")))
        writeResult("p2_coresident", body)
        return
    }
    let (psoOpt, perr) = pipeline(dev, lib, "coresident")
    guard let pso = psoOpt else {
        body.append(("pipeline_error", jOptS(perr)))
        body.append(("status", .s("pipeline creation failed")))
        writeResult("p2_coresident", body)
        return
    }

    let cores = gpuCoreCount()
    let groupLimit = cores != nil ? 64 * cores! : FALLBACK_GROUP_LIMIT
    body.append(("gpu_core_count", jOptI(cores)))
    body.append(("group_limit", .i(groupLimit)))
    body.append(("max_total_threads_per_threadgroup", .i(pso.maxTotalThreadsPerThreadgroup)))
    body.append(("thread_execution_width", .i(pso.threadExecutionWidth)))

    let arrived = dev.makeBuffer(length: 4, options: .storageModeShared)!
    let seen = dev.makeBuffer(length: 4 * groupLimit, options: .storageModeShared)!
    var spinCap = SPIN_CAP_MAX

    // One dispatch of N groups.  Returns what every group managed to see.
    func attempt(_ n: Int, _ tgSize: Int, _ tgMem: Int, _ touch: Int) -> Attempt {
        var a = Attempt()
        a.capUsed = spinCap
        memset(arrived.contents(), 0, 4)
        memset(seen.contents(), 0, 4 * groupLimit)
        var nvar = UInt32(n), capvar = spinCap, touchvar = UInt32(touch)
        let t0 = now()
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            a.error = "could not make a command encoder"
            return a
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(arrived, offset: 0, index: 0)
        enc.setBuffer(seen, offset: 0, index: 1)
        enc.setBytes(&nvar, length: 4, index: 2)
        enc.setBytes(&capvar, length: 4, index: 3)
        enc.setBytes(&touchvar, length: 4, index: 4)
        enc.setThreadgroupMemoryLength(tgMem, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        a.seconds = now() - t0
        if let e = cb.error { a.error = "\(e)"; return a }
        let p = seen.contents().bindMemory(to: UInt32.self, capacity: groupLimit)
        var lo = Int.max, hi = 0
        for i in 0..<n {
            let v = Int(p[i])
            if v < lo { lo = v }
            if v > hi { hi = v }
        }
        a.minSeen = lo == Int.max ? 0 : lo
        a.maxSeen = hi
        a.allSawN = a.minSeen >= n
        // Keep the sweep bounded in wall clock without ever letting a spin run
        // unbounded: shorten the cap, and say so in the row.
        if a.seconds > SLOW_SECONDS && spinCap > SPIN_CAP_MIN {
            spinCap = max(SPIN_CAP_MIN, spinCap / 4)
        }
        return a
    }

    var rows: [J] = []
    // 16 B stands in for "no threadgroup memory": setThreadgroupMemoryLength
    // takes a positive multiple of 16, so 16 is the smallest bindable and the
    // row is labelled with what was actually asked for.
    let memConfigs = [(16, "16 B (minimum bindable, stands for 0)"),
                      (16 * 1024, "16 KiB"),
                      (32 * 1024, "32 KiB")]
    for tgSize in [64, 256, 1024] {
        for (tgMem, memLabel) in memConfigs {
            var row: [(String, J)] = [
                ("threadgroup_size", .i(tgSize)),
                ("threadgroup_memory_bytes", .i(tgMem)),
                ("threadgroup_memory_label", .s(memLabel)),
            ]
            if tgSize > pso.maxTotalThreadsPerThreadgroup {
                row.append(("skipped", .s("threadgroup size exceeds this pipeline's maxTotalThreadsPerThreadgroup (\(pso.maxTotalThreadsPerThreadgroup)) -- a fact about this chip, not a failure")))
                rows.append(.o(row)); continue
            }
            if tgMem > dev.maxThreadgroupMemoryLength {
                row.append(("skipped", .s("threadgroup memory exceeds maxThreadgroupMemoryLength (\(dev.maxThreadgroupMemoryLength)) -- absent on this chip")))
                rows.append(.o(row)); continue
            }
            let touch = min(tgMem, 4096)
            var lastGood = 0
            var firstBad = 0
            var dispatches = 0
            var trail: [J] = []
            var hardError: String? = nil

            var n = 1
            while n <= groupLimit {
                let a = attempt(n, tgSize, tgMem, touch)
                dispatches += 1
                trail.append(.o([("n", .i(n)), ("all_saw_n", .b(a.allSawN)),
                                 ("min_seen", .i(a.minSeen)), ("max_seen", .i(a.maxSeen)),
                                 ("seconds", .d(a.seconds)), ("spin_cap", .u(UInt64(a.capUsed))),
                                 ("error", jOptS(a.error))]))
                if let e = a.error { hardError = e; break }
                if a.allSawN { lastGood = n } else { firstBad = n; break }
                if n == groupLimit { break }
                n = min(n * 2, groupLimit)
            }
            // Bisect the gap the doubling left behind.
            if hardError == nil && firstBad > lastGood + 1 {
                var lo = lastGood, hi = firstBad
                while lo + 1 < hi {
                    let mid = lo + (hi - lo) / 2
                    let a = attempt(mid, tgSize, tgMem, touch)
                    dispatches += 1
                    trail.append(.o([("n", .i(mid)), ("all_saw_n", .b(a.allSawN)),
                                     ("min_seen", .i(a.minSeen)), ("max_seen", .i(a.maxSeen)),
                                     ("seconds", .d(a.seconds)), ("spin_cap", .u(UInt64(a.capUsed))),
                                     ("error", jOptS(a.error))]))
                    if a.error != nil { hardError = a.error; break }
                    if a.allSawN { lo = mid; lastGood = mid } else { hi = mid }
                }
            }
            row.append(("coresident_threadgroups", .i(lastGood)))
            row.append(("reached_search_ceiling", .b(lastGood >= groupLimit)))
            if let c = cores, c > 0 {
                row.append(("groups_per_gpu_core", .d(Double(lastGood) / Double(c))))
            } else {
                row.append(("groups_per_gpu_core", .null))
            }
            row.append(("threads_coresident", .i(lastGood * tgSize)))
            row.append(("dispatches", .i(dispatches)))
            row.append(("command_buffer_error", jOptS(hardError)))
            row.append(("search_trail", .a(trail)))
            rows.append(.o(row))
            note("tg=\(tgSize) mem=\(tgMem)B -> \(lastGood) co-resident groups"
                 + (hardError != nil ? "  [\(hardError!)]" : ""))
        }
    }
    body.append(("sweep", .a(rows)))
    body.append(("final_spin_cap", .u(UInt64(spinCap))))
    body.append(("status", .s("ok")))
    writeResult("p2_coresident", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
