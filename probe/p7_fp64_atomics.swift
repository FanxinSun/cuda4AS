// p7_fp64_atomics -- the two rows of blueprint §1 that say "software" and
// "partial", asked of the compiler in front of us.
//
//   FP64            | none; Apple's Metal compiler rejects `double` arithmetic
//                   | software: double-float (~48-bit mantissa) or IEEE
//                     soft-double, or the CPU device
//   64-bit atomics  | partial | lock-backed fallback
//
// Both rows were written from the public record.  This probe replaces them with
// this compiler's own words.  The `double` rejection message matters as much as
// the rejection: §2.6's diagnostics contract promises every refusal names the
// .cu line and says what to do, which means the translator has to recognise
// this message and translate it, not pass it through.
//
// 32-bit atomics are included as the control: §1 calls them "native", and a
// probe whose baseline fails is measuring something other than what it thinks.
// `atomic_float` add is included because CUDA's atomicAdd(float*) is in almost
// every reduction anyone ships, and whether it is native here decides whether
// the runtime emulates it with a CAS loop.
//
// Every kernel is compiled in its OWN library, so one refusal costs one row.
// A refusal is a RESULT: the compiler's text is copied into the JSON verbatim.
//
// THE MACHINE RULE.  Atomic support is per GPU family and per Metal version,
// not per vendor: `atomic_ulong` may be present on an M3 and absent on an M1,
// and the runtime picks its fallback from the profile (§2.8) rather than from a
// compile-time constant.  Report which chip and which OS answered.
//
// Bounded work: one dispatch of THREADS per case; the CAS loop is capped.

import Foundation
import Metal

let THREADS = 1 << 16
let CAS_CAP: UInt32 = 1 << 12

func head() -> String { return "#include <metal_stdlib>\nusing namespace metal;\n" }

// Each case: a name, its MSL, the kernel entry point, the output element size,
// and what the answer must be if the hardware did what was asked.
let cases: [(String, String, String, String)] = [

  ("fp64_arithmetic", head() + """
  // §1: "Apple's Metal compiler rejects `double` arithmetic".  Verbatim text
  // of the refusal is the deliverable.
  kernel void k(device double *out [[buffer(0)]],
                constant uint &n [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
      if (gid >= n) return;
      double a = double(gid) * 1.0000000001;
      double b = a * a + 1.0;
      out[gid] = b / (a + 1.0);
  }
  """, "k", "expected to be REFUSED; the message is the finding"),

  ("atomic_uint_add", head() + """
  kernel void k(device atomic_uint *acc [[buffer(0)]],
                constant uint &n [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
      if (gid < n) atomic_fetch_add_explicit(acc, 1u, memory_order_relaxed);
  }
  """, "k", "control: acc == THREADS"),

  ("atomic_ulong_add", head() + """
  // 2^33 per thread, so a 32-bit accumulator could not hold the answer even
  // once: this distinguishes a real 64-bit atomic from a silently narrowed one.
  kernel void k(device atomic_ulong *acc [[buffer(0)]],
                constant uint &n [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
      if (gid < n) atomic_fetch_add_explicit(acc, 8589934592ul,
                                             memory_order_relaxed);
  }
  """, "k", "acc == THREADS * 2^33"),

  ("atomic_ulong_min", head() + """
  kernel void k(device atomic_ulong *acc [[buffer(0)]],
                constant uint &n [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
      if (gid < n) atomic_fetch_min_explicit(acc, ulong(gid) + 1000ul,
                                             memory_order_relaxed);
  }
  """, "k", "acc == 1000 (the smallest value offered)"),

  ("atomic_ulong_compare_exchange", head() + """
  // A lock-free maximum by compare-exchange -- the shape every 64-bit atomic
  // fallback in the runtime would take.  BOUNDED by cap iterations.
  kernel void k(device atomic_ulong *acc [[buffer(0)]],
                constant uint &n [[buffer(1)]],
                constant uint &cap [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
      if (gid >= n) return;
      ulong mine = ulong(gid) + 1000ul;
      ulong seen = atomic_load_explicit(acc, memory_order_relaxed);
      for (uint i = 0; i < cap; i++) {
          if (seen >= mine) break;
          if (atomic_compare_exchange_weak_explicit(acc, &seen, mine,
                  memory_order_relaxed, memory_order_relaxed)) break;
      }
  }
  """, "k", "acc == 1000 + THREADS - 1 (the largest value offered)"),

  ("atomic_float_add", head() + """
  // CUDA's atomicAdd(float*) is in nearly every reduction that ships.  1.0 per
  // thread stays exact in FP32 up to 2^24, well above THREADS.
  kernel void k(device atomic_float *acc [[buffer(0)]],
                constant uint &n [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
      if (gid < n) atomic_fetch_add_explicit(acc, 1.0f, memory_order_relaxed);
  }
  """, "k", "acc == THREADS as a float"),
]

func probeMain() {
    var body: [(String, J)] = []
    let dev = acquireDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("threads", .i(THREADS)))
    body.append(("cas_cap", .u(UInt64(CAS_CAP))))

    guard let dev = dev, let queue = dev.makeCommandQueue() else {
        body.append(("status", .s("no Metal device"))); writeResult("p7_fp64_atomics", body); return
    }
    let acc = dev.makeBuffer(length: 4096, options: .storageModeShared)!
    let out = dev.makeBuffer(length: 8 * THREADS, options: .storageModePrivate)!

    var rows: [J] = []
    for (name, src, fn, expectation) in cases {
        var r: [(String, J)] = [("case", .s(name)), ("expectation", .s(expectation))]
        let (libOpt, cerr) = compileMSL(dev, src)
        r.append(("compiled", .b(libOpt != nil)))
        r.append(("compiler_message", jOptS(cerr)))
        guard let lib = libOpt else {
            r.append(("reading", .s("refused by this Metal compiler; the message above is the result, and §2.6 requires the translator to recognise and rewrite it rather than pass it through")))
            note("\(name): REFUSED")
            rows.append(.o(r)); continue
        }
        let (psoOpt, perr) = pipeline(dev, lib, fn)
        guard let pso = psoOpt else {
            r.append(("pipeline_error", jOptS(perr)))
            rows.append(.o(r)); continue
        }
        memset(acc.contents(), 0, 4096)
        if name == "atomic_ulong_min" {
            // fetch_min needs a maximum starting point to be meaningful.
            acc.contents().bindMemory(to: UInt64.self, capacity: 1)[0] = UInt64.max
        }
        var n = UInt32(THREADS), cap = CAS_CAP
        let isDouble = name == "fp64_arithmetic"
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            r.append(("error", .s("no encoder"))); rows.append(.o(r)); continue
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(isDouble ? out : acc, offset: 0, index: 0)
        enc.setBytes(&n, length: 4, index: 1)
        if name == "atomic_ulong_compare_exchange" { enc.setBytes(&cap, length: 4, index: 2) }
        let w = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: THREADS, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error {
            r.append(("dispatch_error", .s("\(e)"))); rows.append(.o(r)); continue
        }
        r.append(("ran", .b(true)))
        r.append(("gpu_seconds", .d(gpuSeconds(cb))))

        let u32 = acc.contents().bindMemory(to: UInt32.self, capacity: 1)
        let u64 = acc.contents().bindMemory(to: UInt64.self, capacity: 1)
        let f32 = acc.contents().bindMemory(to: Float.self, capacity: 1)
        switch name {
        case "fp64_arithmetic":
            r.append(("note", .s("this compiler ACCEPTED double arithmetic -- §1's FP64 row is wrong on this machine and OS, which changes §2.5's routing rule")))
        case "atomic_uint_add":
            r.append(("observed", .u(UInt64(u32[0]))))
            r.append(("correct", .b(u32[0] == UInt32(THREADS))))
        case "atomic_ulong_add":
            let want = UInt64(THREADS) &* 8589934592
            r.append(("observed", .u(u64[0]))); r.append(("expected", .u(want)))
            r.append(("correct", .b(u64[0] == want)))
            r.append(("exceeds_32_bits", .b(want > 0xFFFFFFFF)))
        case "atomic_ulong_min":
            r.append(("observed", .u(u64[0]))); r.append(("expected", .u(1000)))
            r.append(("correct", .b(u64[0] == 1000)))
        case "atomic_ulong_compare_exchange":
            let want = UInt64(1000 + THREADS - 1)
            r.append(("observed", .u(u64[0]))); r.append(("expected", .u(want)))
            r.append(("correct", .b(u64[0] == want)))
        case "atomic_float_add":
            r.append(("observed", .d(Double(f32[0]))))
            r.append(("expected", .d(Double(THREADS))))
            r.append(("correct", .b(f32[0] == Float(THREADS))))
        default: break
        }
        note("\(name): compiled and ran")
        rows.append(.o(r))
    }
    body.append(("cases", .a(rows)))
    body.append(("status", .s("ok")))
    writeResult("p7_fp64_atomics", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
