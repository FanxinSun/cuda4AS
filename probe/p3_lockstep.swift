// p3_lockstep -- does a SIMD-group run in lockstep, and is simdgroup_barrier a
// fence or an execution barrier?
//
// Blueprint §1: "independent thread scheduling (Volta+): intra-warp spin locks
// legal | SIMD-group runs in lockstep with a divergence mask, AS FAR AS THE
// PUBLIC RECORD GOES; simdgroup_barrier is a fence".  That row is an assumption
// with a design hanging off it in two directions:
//   * lmz's Metal decoder (`lmz/scratchpad/gpu/metal/README.md`, item 1) is
//     correct ONLY if simdgroup_barrier fences memory for the lanes executing
//     it while other lanes of the same SIMD-group are elsewhere.  Test A1.
//   * CUDA code written for Volta's independent thread scheduling -- intra-warp
//     spin locks, producer/consumer between lanes -- deadlocks without it.  The
//     translator must detect and reject that, so it has to know.  Test B.
//
// Four experiments, all bounded, none relying on a watchdog:
//   A1  divergent simdgroup_barrier used as lmz uses it (within one branch)
//   A2  cross-branch visibility -- distinguishes "memory fence" from
//       "execution barrier that reconverges"
//   B   a lane-level lock whose holder releases only AFTER the spin loop.
//       Under lockstep exactly one lane per SIMD-group can ever acquire it;
//       under independent thread scheduling every lane does.  Terminates
//       either way because the loop is capped.
//   C   simd_ballot + popcount prefix sum -- the one SIMD instruction lmz's
//       decoder replaces a serial cursor walk with.  Correctness, not speed.
//
// THE MACHINE RULE: this measures THIS chip and THIS Metal runtime.  Apple has
// never promised lockstep in writing, and a future chip may schedule lanes
// independently.  The design must therefore treat the answer as a profile entry
// (§2.8), not as a fact about Apple silicon -- which is exactly why it is
// probed rather than assumed.

import Foundation
import Metal

let LOCK_SPIN_CAP: UInt32 = 1 << 16     // BOUNDED: iterations per lane in test B
let TG = 32                             // one SIMD-group per threadgroup
let NGROUPS = 8

let lockstepSrc = """
#include <metal_stdlib>
using namespace metal;

// A1: lmz's pattern.  Two halves of a SIMD-group take different branches; the
// lanes inside each branch exchange values through threadgroup memory across a
// simdgroup_barrier.  Correct iff the barrier fences memory for the lanes that
// execute it.
kernel void a1_divergent_fence(device uint *out [[buffer(0)]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint gid  [[thread_position_in_grid]]) {
    threadgroup uint buf[256];
    buf[tid] = 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = tid - lane;
    if (lane < 16u) {
        buf[tid] = lane * 3u + 1u;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint partner = (lane + 8u) % 16u;                 // stays in this half
        out[gid] = buf[base + partner];
    } else {
        buf[tid] = lane * 7u + 2u;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint partner = 16u + ((lane - 16u + 8u) % 16u);   // stays in this half
        out[gid] = buf[base + partner];
    }
}

// A2: each half reads what the OTHER half wrote.  If the branches are
// serialised behind a divergence mask, the branch that runs first cannot see
// the other's writes and the failure is one-sided -- that asymmetry is the
// signature of a fence, as against an execution barrier that reconverges.
kernel void a2_cross_branch(device uint *out [[buffer(0)]],
                            uint tid  [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint gid  [[thread_position_in_grid]]) {
    threadgroup uint buf[256];
    buf[tid] = 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = tid - lane;
    if (lane < 16u) {
        buf[tid] = 0xA0000u + lane;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        out[gid] = buf[base + 16u + lane];
    } else {
        buf[tid] = 0xB0000u + (lane - 16u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        out[gid] = buf[base + (lane - 16u)];
    }
}

// B: the independent-thread-scheduling test.  The winner does NOT release
// inside the loop, so under lockstep the loop cannot end for it until every
// lane's predicate is false -- i.e. until the cap -- and no second lane ever
// acquires.  BOUNDED by cap.
kernel void b_lane_lock(device atomic_uint *lock [[buffer(0)]],
                        device uint *out         [[buffer(1)]],
                        constant uint &cap       [[buffer(2)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint gid [[thread_position_in_grid]]) {
    bool got = false;
    for (uint i = 0; i < cap && !got; i++) {
        uint expected = 0u;
        got = atomic_compare_exchange_weak_explicit(lock, &expected, 1u,
                memory_order_relaxed, memory_order_relaxed);
    }
    if (got) atomic_store_explicit(lock, 0u, memory_order_relaxed);
    out[gid] = got ? 1u : 0u;
}

// C: simd_ballot + popcount exclusive prefix sum, the idiom lmz's decoder is
// built on (lmz_rans.metal line 118).  Every lane's answer is checkable.
kernel void c_ballot_prefix(device uint *ballotOut [[buffer(0)]],
                            device uint *prefixOut [[buffer(1)]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint gid  [[thread_position_in_grid]]) {
    bool need = ((lane * 7u + 3u) % 5u) < 2u;             // a mixed pattern
    uint ball = (uint)((simd_vote::vote_t)simd_ballot(need));
    ballotOut[gid] = ball;
    prefixOut[gid] = popcount(ball & ((1u << lane) - 1u));
}
"""

func probeMain() {
    var body: [(String, J)] = []
    let dev = MTLCreateSystemDefaultDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("lock_spin_cap", .u(UInt64(LOCK_SPIN_CAP))))
    body.append(("threadgroup_size", .i(TG)))
    body.append(("threadgroups", .i(NGROUPS)))

    guard let dev = dev, let queue = dev.makeCommandQueue() else {
        body.append(("status", .s("no Metal device"))); writeResult("p3_lockstep", body); return
    }
    let (libOpt, cerr) = compileMSL(dev, lockstepSrc)
    guard let lib = libOpt else {
        body.append(("compile_error", jOptS(cerr)))
        body.append(("status", .s("the probe kernels did not compile -- that message is the result")))
        writeResult("p3_lockstep", body); return
    }
    let n = TG * NGROUPS

    func run(_ fn: String, _ bind: (MTLComputeCommandEncoder) -> Void) -> String? {
        let (psoOpt, perr) = pipeline(dev, lib, fn)
        guard let pso = psoOpt else { return perr ?? "no pipeline" }
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return "no encoder" }
        enc.setComputePipelineState(pso)
        bind(enc)
        enc.dispatchThreadgroups(MTLSize(width: NGROUPS, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: TG, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { return "\(e)" }
        return nil
    }

    let out = dev.makeBuffer(length: 4 * n, options: .storageModeShared)!
    let out2 = dev.makeBuffer(length: 4 * n, options: .storageModeShared)!
    let lock = dev.makeBuffer(length: 4, options: .storageModeShared)!
    func words(_ b: MTLBuffer) -> UnsafeMutablePointer<UInt32> {
        return b.contents().bindMemory(to: UInt32.self, capacity: n)
    }

    var width: Int? = nil
    if let fn = lib.makeFunction(name: "a1_divergent_fence"),
       let pso = try? dev.makeComputePipelineState(function: fn) {
        width = pso.threadExecutionWidth
    }
    body.append(("thread_execution_width", jOptI(width)))
    body.append(("simdgroup_is_32_lanes", .b(width == 32)))

    // ---- A1 ------------------------------------------------------------
    memset(out.contents(), 0, 4 * n)
    var a1: [(String, J)] = []
    if let e = run("a1_divergent_fence", { $0.setBuffer(out, offset: 0, index: 0) }) {
        a1.append(("error", .s(e)))
    } else {
        let p = words(out)
        var bad = 0
        for i in 0..<n {
            let lane = UInt32(i % TG)
            let want: UInt32 = lane < 16
                ? ((lane + 8) % 16) * 3 + 1
                : (16 + ((lane - 16 + 8) % 16)) * 7 + 2
            if p[i] != want { bad += 1 }
        }
        a1.append(("mismatches", .i(bad)))
        a1.append(("correct", .b(bad == 0)))
        a1.append(("reading", .s(bad == 0
            ? "simdgroup_barrier fences memory for the lanes executing it inside a divergent branch -- lmz's Metal decoder rests on exactly this"
            : "the divergent fence did NOT hold; lmz_decode_plane_prefetch is unsafe on this chip and the direct kernel is the usable answer")))
    }
    body.append(("a1_divergent_fence", .o(a1)))

    // ---- A2 ------------------------------------------------------------
    memset(out.contents(), 0, 4 * n)
    var a2: [(String, J)] = []
    if let e = run("a2_cross_branch", { $0.setBuffer(out, offset: 0, index: 0) }) {
        a2.append(("error", .s(e)))
    } else {
        let p = words(out)
        var lowHalfSaw = 0, highHalfSaw = 0
        for i in 0..<n {
            let lane = UInt32(i % TG)
            if lane < 16 { if p[i] == 0xB0000 + lane { lowHalfSaw += 1 } }
            else { if p[i] == 0xA0000 + (lane - 16) { highHalfSaw += 1 } }
        }
        a2.append(("low_half_saw_high_writes", .i(lowHalfSaw)))
        a2.append(("high_half_saw_low_writes", .i(highHalfSaw)))
        a2.append(("lanes_per_half", .i(16 * NGROUPS)))
        let both = lowHalfSaw == 16 * NGROUPS && highHalfSaw == 16 * NGROUPS
        let neither = lowHalfSaw == 0 && highHalfSaw == 0
        a2.append(("reading", .s(both
            ? "both halves saw each other: simdgroup_barrier reconverges, i.e. behaves as an execution barrier here"
            : (neither
               ? "neither half saw the other: the barrier is a memory fence only and the branches did not reconverge at it"
               : "one-sided -- the branches were serialised behind a divergence mask and the barrier is a fence, not a reconvergence point"))))
    }
    body.append(("a2_cross_branch", .o(a2)))

    // ---- B ---------------------------------------------------------------
    memset(out.contents(), 0, 4 * n)
    memset(lock.contents(), 0, 4)
    var b: [(String, J)] = []
    var cap = LOCK_SPIN_CAP
    let t0 = now()
    if let e = run("b_lane_lock", { enc in
        enc.setBuffer(lock, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&cap, length: 4, index: 2)
    }) {
        b.append(("error", .s(e)))
    } else {
        let p = words(out)
        var acquired = 0
        for i in 0..<n where p[i] == 1 { acquired += 1 }
        b.append(("lanes_total", .i(n)))
        b.append(("lanes_acquired", .i(acquired)))
        b.append(("acquired_per_simdgroup", .d(Double(acquired) / Double(NGROUPS))))
        b.append(("seconds", .d(now() - t0)))
        b.append(("independent_thread_scheduling", .b(acquired > NGROUPS)))
        b.append(("reading", .s(acquired <= NGROUPS
            ? "at most one lane per SIMD-group ever acquired: no independent thread scheduling, so CUDA intra-warp spin locks deadlock and the translator must reject them"
            : "more than one lane per SIMD-group acquired: lanes make independent forward progress on this chip, which would make Volta-style intra-warp locks legal")))
    }
    body.append(("b_lane_lock", .o(b)))

    // ---- C ---------------------------------------------------------------
    memset(out.contents(), 0, 4 * n)
    memset(out2.contents(), 0, 4 * n)
    var c: [(String, J)] = []
    if let e = run("c_ballot_prefix", { enc in
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBuffer(out2, offset: 0, index: 1)
    }) {
        c.append(("error", .s(e)))
    } else {
        // Same predicate the kernel used, evaluated here.
        var wantMask: UInt32 = 0
        for lane in 0..<32 where ((UInt32(lane) &* 7 &+ 3) % 5) < 2 {
            wantMask |= (UInt32(1) << UInt32(lane))
        }
        let bp = words(out), pp = words(out2)
        var badBallot = 0, badPrefix = 0
        for i in 0..<n {
            let lane = UInt32(i % TG)
            if bp[i] != wantMask { badBallot += 1 }
            let below: UInt32 = (UInt32(1) << lane) &- 1
            let wantPrefix = UInt32((wantMask & below).nonzeroBitCount)
            if pp[i] != wantPrefix { badPrefix += 1 }
        }
        c.append(("expected_ballot_mask", .s(String(format: "0x%08x", wantMask))))
        c.append(("ballot_mismatches", .i(badBallot)))
        c.append(("prefix_mismatches", .i(badPrefix)))
        c.append(("correct", .b(badBallot == 0 && badPrefix == 0)))
        c.append(("reading", .s("simd_ballot + popcount is the instruction lmz's decoder replaces a serial cursor walk with; CUDA's __ballot_sync/__popc map here 1:1")))
    }
    body.append(("c_ballot_prefix", .o(c)))

    body.append(("status", .s("ok")))
    writeResult("p3_lockstep", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
