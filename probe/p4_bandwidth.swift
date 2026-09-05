// p4_bandwidth -- the bandwidth row of the machine profile.
//
// Blueprint §3 turns every decode claim into `bandwidth / bytes per token`, and
// the FLOP-per-byte column of the §3 table is `matrix rate / bandwidth`.  Both
// need a measured number, not a spec sheet: Apple's published figure is the
// LPDDR peak and the M5 measurement in §3 already shows 121.8 of 153.6 GB/s
// reached, which is 79 %.  A design that budgets against peak budgets wrong.
//
// Two kernels, because read+write and read-only are different machines:
//   copy    16 B in and 16 B out per thread -- the number a memcpy-shaped
//           residency transfer gets
//   read    16 B in per thread, result consumed by a comparison the compiler
//           cannot fold away -- the number a weight-streaming decode gets
//
// Both report GB/s with the traffic they counted stated, because "bandwidth"
// without that is two different numbers a factor of two apart.
//
// THE MACHINE RULE.  Working-set size is chosen from THIS machine's limits, not
// fixed: min(1 GiB, quarter of maxBufferLength, eighth of the working-set
// limit), so an 8 GB M1 and a 512 GB Studio both run it without swapping, and
// the size actually used is reported beside the number.  A GB/s figure without
// its chip, its buffer size and its traffic definition is not a measurement.

import Foundation
import Metal

let REPEATS = 5           // best of 5, per the handover
let WARMUPS = 2

let bwSrc = """
#include <metal_stdlib>
using namespace metal;

// 16 B read + 16 B written per thread.
kernel void bw_copy(device const uint4 *src [[buffer(0)]],
                    device uint4 *dst       [[buffer(1)]],
                    constant uint &n        [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = src[gid];
}

// 16 B read per thread and nothing written.  The comparison keeps the load
// live without adding a store: the constant is never produced by the data.
kernel void bw_read(device const uint4 *src [[buffer(0)]],
                    device uint *sink       [[buffer(1)]],
                    constant uint &n        [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint4 v = src[gid];
    uint r = v.x ^ v.y ^ v.z ^ v.w;
    if (r == 0xDEADBEEFu) sink[0] = r;
}
"""

func probeMain() {
    var body: [(String, J)] = []
    let dev = acquireDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("repeats", .i(REPEATS)))
    body.append(("warmups", .i(WARMUPS)))
    body.append(("timing_method", .s("MTLCommandBuffer gpuEndTime - gpuStartTime, best of \(REPEATS) after \(WARMUPS) warmups; wall clock reported alongside")))

    guard let dev = dev, let queue = dev.makeCommandQueue() else {
        body.append(("status", .s("no Metal device"))); writeResult("p4_bandwidth", body); return
    }
    let (libOpt, cerr) = compileMSL(dev, bwSrc)
    guard let lib = libOpt else {
        body.append(("compile_error", jOptS(cerr)))
        body.append(("status", .s("bandwidth kernels did not compile")))
        writeResult("p4_bandwidth", body); return
    }

    // Working set, chosen from this machine rather than fixed.
    let oneGiB = 1024 * 1024 * 1024
    let byBuffer = dev.maxBufferLength / 4
    let byWorkingSet = Int(dev.recommendedMaxWorkingSetSize / 8)
    var bytes = min(oneGiB, min(byBuffer, byWorkingSet))
    bytes = (bytes / 16384) * 16384                    // whole uint4 and pages
    if bytes < 16 * 1024 * 1024 { bytes = min(16 * 1024 * 1024, byBuffer) }
    let vecs = bytes / 16
    body.append(("working_set_bytes", .i(bytes)))
    body.append(("working_set_mib", .d(Double(bytes) / 1048576.0)))
    body.append(("working_set_choice", .s("min(1 GiB, maxBufferLength/4, recommendedMaxWorkingSetSize/8)")))
    body.append(("elements_uint4", .i(vecs)))
    note("working set: \(bytes / 1048576) MiB (\(vecs) uint4)")

    guard let src = dev.makeBuffer(length: bytes, options: .storageModePrivate),
          let dst = dev.makeBuffer(length: bytes, options: .storageModePrivate),
          let sink = dev.makeBuffer(length: 16, options: .storageModeShared) else {
        body.append(("status", .s("could not allocate \(bytes) B x2 -- report this as the machine's limit, not a probe failure")))
        writeResult("p4_bandwidth", body); return
    }

    func measure(_ fn: String, trafficPerThread: Int, writes: Bool) -> J {
        let (psoOpt, perr) = pipeline(dev, lib, fn)
        guard let pso = psoOpt else { return .o([("error", jOptS(perr))]) }
        var n = UInt32(vecs)
        var best = Double.infinity
        var bestWall = Double.infinity
        var err: String? = nil
        let tgw = pso.maxTotalThreadsPerThreadgroup
        for i in 0..<(WARMUPS + REPEATS) {
            let w0 = now()
            guard let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { err = "no encoder"; break }
            enc.setComputePipelineState(pso)
            enc.setBuffer(src, offset: 0, index: 0)
            enc.setBuffer(writes ? dst : sink, offset: 0, index: 1)
            enc.setBytes(&n, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: vecs, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgw, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let wall = now() - w0
            if let e = cb.error { err = "\(e)"; break }
            if i < WARMUPS { continue }
            let t = gpuSeconds(cb)
            if t.isFinite && t < best { best = t }
            if wall < bestWall { bestWall = wall }
        }
        if let e = err { return .o([("error", .s(e))]) }
        let traffic = Double(vecs) * Double(trafficPerThread)
        return .o([
            ("threads", .i(vecs)),
            ("threads_per_threadgroup", .i(tgw)),
            ("bytes_per_thread", .i(trafficPerThread)),
            ("traffic_bytes", .d(traffic)),
            ("traffic_definition", .s(writes ? "16 B read + 16 B written" : "16 B read, nothing written")),
            ("best_gpu_seconds", .d(best)),
            ("best_wall_seconds", .d(bestWall)),
            ("gb_per_s", .d(best.isFinite ? traffic / best / 1e9 : Double.nan)),
            ("gb_per_s_wall", .d(bestWall.isFinite ? traffic / bestWall / 1e9 : Double.nan)),
        ])
    }

    let copyRes = measure("bw_copy", trafficPerThread: 32, writes: true)
    let readRes = measure("bw_read", trafficPerThread: 16, writes: false)
    body.append(("copy_read_plus_write", copyRes))
    body.append(("read_only", readRes))

    // The two ratios the design actually consumes.
    if case .o(let r) = readRes {
        for (k, v) in r where k == "gb_per_s" {
            if case .d(let gbps) = v, gbps.isFinite {
                note("read-only: \(String(format: "%.1f", gbps)) GB/s")
                body.append(("read_gb_per_s", .d(gbps)))
                // §3's decode claim, restated for this machine: batch-1 token
                // generation is bandwidth divided by bytes per token.
                body.append(("bf16_8b_tokens_per_s_if_bandwidth_bound",
                             .d(gbps * 1e9 / (8.0e9 * 2))))
                body.append(("bf16_8b_tokens_note",
                             .s("read GB/s / (8e9 params x 2 B) -- the §3 decode roofline for an 8B BF16 model on THIS machine, weights uncoded")))
            }
        }
    }

    body.append(("status", .s("ok")))
    writeResult("p4_bandwidth", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
