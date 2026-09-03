// p5_simd -- the SIMD and simdgroup-matrix rows of the machine profile.
//
// These are the rates the FP32 SIMD and FP16 matrix columns of blueprint §3 are
// projected from, and the M1–M4 half of §2.4's "one ABI, two back-ends": on a
// chip without Neural Accelerators, `simdgroup_matrix` 8x8 IS the matrix unit,
// and cuBLAS on that chip runs at whatever this probe measures.
//
// Four separately compiled libraries, so one refusal costs one number:
//   f32     FP32 FMA chains          -- the SIMD ALU rate
//   f16     FP16 FMA chains          -- half rate, or 2x if the ALU is packed
//   bf16    BF16 FMA chains          -- §1 says MSL has `bfloat` for ALU work
//                                       since Metal 3.1; this is the ALU
//                                       question, NOT the accelerator question
//                                       (that is p6, and §3 turns on it)
//   sgmm    simdgroup_matrix<half,8,8> multiply-accumulate
//
// Method.  Eight independent dependency chains per thread (four accumulators
// for the matrix case) so the measurement is throughput and not FMA latency;
// iteration count calibrated at run time to land near TARGET_SECONDS on
// whatever this chip is, then reported.  The result is consumed by a
// comparison against a constant the data cannot produce, so nothing is folded
// away and nothing is stored.
//
// THE MACHINE RULE.  A TFLOPS number is meaningless without its chip, its clock
// state and its occupancy, all of which vary with thermals; this runs a few
// milliseconds and reports the best of REPEATS, which measures the boost clock,
// not the sustained one.  Say so wherever the number is used.  A phone-class
// SoC and an M5 Ultra differ by ~40x here and both are targets.

import Foundation
import Metal

let CHAINS = 8
let ACCUMULATORS = 4
let THREADS = 1 << 20
let CALIB_ITERS: UInt32 = 64
let TARGET_SECONDS = 0.03
let REPEATS = 5

func fmaSource(_ type: String) -> String {
    return """
#include <metal_stdlib>
using namespace metal;
kernel void fma_chain(device \(type) *out [[buffer(0)]],
                      constant uint &iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    \(type) a[\(CHAINS)];
    for (uint c = 0; c < \(CHAINS)u; c++) a[c] = \(type)(float(gid + c) * 1e-6f);
    \(type) b = \(type)(1.0000001f);
    \(type) d = \(type)(1e-7f);
    for (uint i = 0; i < iters; i++) {
        // mul+add rather than fma(): a missing fma() overload for `bfloat`
        // would be misread as "this chip has no bfloat".  Metal contracts this
        // into an FMA by default, and the FLOP count is 2 either way.
        for (uint c = 0; c < \(CHAINS)u; c++) a[c] = a[c] * b + d;
    }
    \(type) s = \(type)(0);
    for (uint c = 0; c < \(CHAINS)u; c++) s += a[c];
    if (s == \(type)(1234567.0f)) out[gid] = s;   // never true; keeps it live
}
"""
}

let sgmmSrc = """
#include <metal_stdlib>
using namespace metal;
// simdgroup_matrix<half,8,8>: one multiply-accumulate is 2*8*8*8 = 1024 FLOP
// for the whole SIMD-group.  Four independent accumulators so the chain is a
// throughput measurement rather than a latency one.
kernel void sgmm(device half *out       [[buffer(0)]],
                 constant uint &iters   [[buffer(1)]],
                 uint gid [[thread_position_in_grid]],
                 uint tid [[thread_position_in_threadgroup]],
                 uint tgsz [[threads_per_threadgroup]]) {
    threadgroup half scratch[64];
    threadgroup half zeros[64];
    for (uint i = tid; i < 64u; i += tgsz) {
        scratch[i] = half(float(i % 7u) * 0.125f);
        zeros[i] = half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 A, B;
    simdgroup_load(A, scratch, 8);
    simdgroup_load(B, scratch, 8);
    // Zeroed by load rather than by a scalar constructor: simdgroup_load is
    // already required here, so this adds no new API surface to get wrong.
    simdgroup_half8x8 C0, C1, C2, C3;
    simdgroup_load(C0, zeros, 8);
    simdgroup_load(C1, zeros, 8);
    simdgroup_load(C2, zeros, 8);
    simdgroup_load(C3, zeros, 8);
    for (uint i = 0; i < iters; i++) {
        simdgroup_multiply_accumulate(C0, A, B, C0);
        simdgroup_multiply_accumulate(C1, A, B, C1);
        simdgroup_multiply_accumulate(C2, A, B, C2);
        simdgroup_multiply_accumulate(C3, A, B, C3);
    }
    threadgroup half res[64];
    simdgroup_store(C0, res, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (res[tid % 64u] == half(1234567.0f)) out[gid] = res[0];
}
"""

func probeMain() {
    var body: [(String, J)] = []
    let dev = MTLCreateSystemDefaultDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("threads", .i(THREADS)))
    body.append(("chains_per_thread", .i(CHAINS)))
    body.append(("matrix_accumulators", .i(ACCUMULATORS)))
    body.append(("repeats", .i(REPEATS)))
    body.append(("method", .s("iteration count calibrated at run time to ~\(TARGET_SECONDS) s on this chip; best of \(REPEATS) GPU times; boost clock, not sustained")))

    guard let dev = dev, let queue = dev.makeCommandQueue() else {
        body.append(("status", .s("no Metal device"))); writeResult("p5_simd", body); return
    }
    let out = dev.makeBuffer(length: 4 * THREADS, options: .storageModePrivate)!

    /// Time one dispatch; returns GPU seconds or nan.
    func timeOnce(_ pso: MTLComputePipelineState, _ iters: UInt32) -> (Double, String?) {
        var it = iters
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return (Double.nan, "no encoder") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBytes(&it, length: 4, index: 1)
        let w = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: THREADS, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { return (Double.nan, "\(e)") }
        return (gpuSeconds(cb), nil)
    }

    /// Calibrate the iteration count, then take the best of REPEATS.
    func run(_ label: String, _ src: String, _ fn: String,
             _ flopsFor: (UInt32) -> Double) -> J {
        let (libOpt, cerr) = compileMSL(dev, src)
        guard let lib = libOpt else {
            note("\(label): compile refused")
            return .o([("compiled", .b(false)), ("compile_error", jOptS(cerr)),
                       ("reading", .s("refused by this Metal compiler -- an absent capability on this chip/OS, recorded as a result"))])
        }
        let (psoOpt, perr) = pipeline(dev, lib, fn)
        guard let pso = psoOpt else {
            return .o([("compiled", .b(true)), ("pipeline_error", jOptS(perr))])
        }
        let (t0, e0) = timeOnce(pso, CALIB_ITERS)
        if let e = e0 { return .o([("compiled", .b(true)), ("error", .s(e))]) }
        var iters = CALIB_ITERS
        if t0.isFinite && t0 > 0 {
            let scaled = Double(CALIB_ITERS) * TARGET_SECONDS / t0
            iters = UInt32(max(Double(CALIB_ITERS), min(65536.0, scaled)))
        }
        var best = Double.infinity
        var err: String? = nil
        for _ in 0..<REPEATS {
            let (t, e) = timeOnce(pso, iters)
            if let e = e { err = e; break }
            if t.isFinite && t < best { best = t }
        }
        if let e = err { return .o([("compiled", .b(true)), ("error", .s(e))]) }
        let flops = flopsFor(iters)
        let tflops = best.isFinite ? flops / best / 1e12 : Double.nan
        note("\(label): \(String(format: "%.2f", tflops)) TFLOPS (iters=\(iters))")
        return .o([
            ("compiled", .b(true)),
            ("iterations", .u(UInt64(iters))),
            ("calibration_seconds", .d(t0)),
            ("best_gpu_seconds", .d(best)),
            ("flop", .d(flops)),
            ("tflops", .d(tflops)),
            ("max_total_threads_per_threadgroup", .i(pso.maxTotalThreadsPerThreadgroup)),
            ("thread_execution_width", .i(pso.threadExecutionWidth)),
        ])
    }

    let chainFlops: (UInt32) -> Double = { it in
        Double(THREADS) * Double(CHAINS) * Double(it) * 2.0
    }
    body.append(("fp32_fma", run("fp32 FMA", fmaSource("float"), "fma_chain", chainFlops)))
    body.append(("fp16_fma", run("fp16 FMA", fmaSource("half"), "fma_chain", chainFlops)))
    body.append(("bf16_fma", run("bf16 FMA", fmaSource("bfloat"), "fma_chain", chainFlops)))
    body.append(("bf16_note", .s("this is the ALU's bfloat, which MSL has had since Metal 3.1 -- NOT the Neural Accelerators' BF16 support, which p6 asks about and which §3's block-scaled FP16 strategy turns on")))

    let sgmmFlops: (UInt32) -> Double = { it in
        // Per SIMD-group, per iteration: ACCUMULATORS multiply-accumulates of
        // 8x8x8, each 2*8*8*8 = 1024 FLOP.
        let simdgroups = Double(THREADS) / 32.0
        return simdgroups * Double(ACCUMULATORS) * Double(it) * 1024.0
    }
    body.append(("simdgroup_matrix_half8x8", run("simdgroup_matrix half", sgmmSrc, "sgmm", sgmmFlops)))
    body.append(("simdgroup_matrix_note", .s("FLOP counted as simdgroups x accumulators x iterations x 2*8*8*8; the SIMD-group count assumes 32 lanes, which p0 and p3 verify")))

    body.append(("status", .s("ok")))
    writeResult("p5_simd", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
