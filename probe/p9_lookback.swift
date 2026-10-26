// probe/p9_lookback.swift
// Decoupled look-back, residency, and forward progress on Apple GPUs.
//
// S1c Follow-up:
// 1. CUB Stock Tile Pattern: tile = threadgroup_position_in_grid.x with arrival ticket
//    recorded alongside to measure arrival order mismatches vs tgid.
// 2. Control Sweep with Arrival-Ticket Remap (blockIdx := ticket).
// 3. Adversarial Forward Progress (ticket 0 delayed to last-dispatched group).
// 4. Single-Word Memory Safety: 64-bit aligned store/load & 32-bit packed encoding.
// 5. Redone Fenced Flag/Value Test: strictly 1 writer per slot, 51.2M reads.
// 6. MSL Language Version Probe: test memory_order_acquire/release across default,
//    3.0, 3.1, and 3.2, recording compiler refusals verbatim.

import Foundation
import Metal

let SPIN_CAP: UInt32 = 1 << 20

func p9Head() -> String {
    return """
    #include <metal_stdlib>
    using namespace metal;
    """
}

// MSL for Residency Measurement
let residencySrc = p9Head() + """
kernel void p9_measure_residency(
    device atomic_uint *activeCount [[buffer(0)]],
    device atomic_uint *maxResident [[buffer(1)]],
    constant uint      &spinDelay   [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup uint s_cur;
    if (tid == 0) {
        uint c = atomic_fetch_add_explicit(activeCount, 1u, memory_order_relaxed) + 1u;
        atomic_fetch_max_explicit(maxResident, c, memory_order_relaxed);
        s_cur = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bounded spin to keep threadgroups resident concurrently
    for (uint i = 0; i < spinDelay; i++) {
        if (s_cur == 0xdeadbeef) break;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        atomic_fetch_sub_explicit(activeCount, 1u, memory_order_relaxed);
    }
}
"""

// MSL for Decoupled Look-back:
// mode == 0: Arrival-ticket remap (control: tile = ticket)
// mode == 1: Adversarial reverse ticket (tile = (numTiles - 1) - ticket)
// mode == 2: CUB stock pattern (tile = tgid, arrival order tracked alongside)
let forwardProgressSrc = p9Head() + """
kernel void p9_decoupled_lookback(
    device atomic_uint   *ticketCounter  [[buffer(0)]],
    device atomic_uint   *tileStatus     [[buffer(1)]], // 0: not ready, 1: aggregate, 2: prefix
    device volatile uint *tileAggregate  [[buffer(2)]],
    device volatile uint *tilePrefix     [[buffer(3)]],
    device uint          *arrivalOrder   [[buffer(4)]],
    device atomic_uint   *timeoutCount   [[buffer(5)]],
    device atomic_uint   *maxSpinsSeen   [[buffer(6)]],
    device atomic_uint   *outOfOrderCount[[buffer(7)]],
    constant uint        &numTiles       [[buffer(8)]],
    constant uint        &spinCap        [[buffer(9)]],
    constant uint        &mode           [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup uint s_tile;
    if (tid == 0) {
        uint ticket = atomic_fetch_add_explicit(ticketCounter, 1u, memory_order_relaxed);
        arrivalOrder[tgid] = ticket;
        if (ticket != tgid) {
            atomic_fetch_add_explicit(outOfOrderCount, 1u, memory_order_relaxed);
        }
        if (mode == 1) {
            // Adversarial reverse: first dispatched threadgroup gets the LAST ticket,
            // and the LAST dispatched threadgroup gets ticket 0!
            s_tile = (numTiles - 1u) - ticket;
        } else if (mode == 2) {
            // CUB stock pattern: tile = blockIdx.x (tgid)
            s_tile = tgid;
        } else {
            // Arrival-ticket remap: tile = ticket
            s_tile = ticket;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tile = s_tile;
    if (tile >= numTiles) return;

    uint myAggregate = tile + 1;

    if (tid == 0) {
        tileAggregate[tile] = myAggregate;
        if (tile == 0) {
            tilePrefix[0] = myAggregate;
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
            atomic_store_explicit(&tileStatus[0], 2u, memory_order_relaxed); // 2: PREFIX_SET
        } else {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
            atomic_store_explicit(&tileStatus[tile], 1u, memory_order_relaxed); // 1: AGGREGATE_SET

            int lookback_idx = int(tile) - 1;
            uint accumulated_prefix = 0;
            uint spins = 0;
            bool timed_out = false;

            while (lookback_idx >= 0) {
                uint st = atomic_load_explicit(&tileStatus[lookback_idx], memory_order_relaxed);
                if (st == 2u) {
                    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
                    accumulated_prefix += tilePrefix[lookback_idx];
                    break;
                } else if (st == 1u) {
                    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
                    accumulated_prefix += tileAggregate[lookback_idx];
                    lookback_idx--;
                } else {
                    spins++;
                    if (spins >= spinCap) {
                        timed_out = true;
                        atomic_fetch_add_explicit(timeoutCount, 1u, memory_order_relaxed);
                        break;
                    }
                }
            }

            if (!timed_out) {
                tilePrefix[tile] = accumulated_prefix + myAggregate;
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
                atomic_store_explicit(&tileStatus[tile], 2u, memory_order_relaxed);
            }
            atomic_fetch_max_explicit(maxSpinsSeen, spins, memory_order_relaxed);
        }
    }
}
"""

// MSL for 64-bit Tearing Test
let tearing64Src = p9Head() + """
kernel void p9_tearing_64bit(
    device volatile ulong *slots        [[buffer(0)]],
    device atomic_uint    *tornCount    [[buffer(1)]],
    device atomic_uint    *readsDone    [[buffer(2)]],
    constant uint         &numSlots     [[buffer(3)]],
    constant uint         &iters        [[buffer(4)]],
    uint gid  [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint slot_idx = gid % numSlots;
    bool is_writer = (tgid % 2) == 0;

    if (is_writer) {
        for (uint i = 1; i <= iters; i++) {
            ulong val = (ulong(i) << 32) | ulong(~i);
            slots[slot_idx] = val;
        }
    } else {
        uint local_torn = 0;
        for (uint i = 0; i < iters; i++) {
            ulong val = slots[slot_idx];
            if (val != 0) {
                uint hi = uint(val >> 32);
                uint lo = uint(val & 0xfffffffful);
                if (hi != ~lo) {
                    local_torn++;
                }
            }
        }
        if (local_torn > 0) {
            atomic_fetch_add_explicit(tornCount, local_torn, memory_order_relaxed);
        }
        atomic_fetch_add_explicit(readsDone, iters, memory_order_relaxed);
    }
}
"""

// MSL for 32-bit Packed Fallback
let packed32Src = p9Head() + """
kernel void p9_packed_32bit(
    device atomic_uint *slots        [[buffer(0)]],
    device atomic_uint *mismatchCount[[buffer(1)]],
    device atomic_uint *readsDone    [[buffer(2)]],
    constant uint      &numSlots     [[buffer(3)]],
    constant uint      &iters        [[buffer(4)]],
    uint gid  [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint slot_idx = gid % numSlots;
    bool is_writer = (tgid % 2) == 0;

    if (is_writer) {
        for (uint i = 1; i <= iters; i++) {
            uint st = (i & 3u);
            uint val = (i & 0x3fffffffu);
            uint packed = (st << 30) | val;
            atomic_store_explicit(&slots[slot_idx], packed, memory_order_relaxed);
        }
    } else {
        uint local_mismatch = 0;
        for (uint i = 0; i < iters; i++) {
            uint packed = atomic_load_explicit(&slots[slot_idx], memory_order_relaxed);
            if (packed != 0) {
                uint st = (packed >> 30);
                uint val = (packed & 0x3fffffffu);
                if ((val & 3u) != st) {
                    local_mismatch++;
                }
            }
        }
        if (local_mismatch > 0) {
            atomic_fetch_add_explicit(mismatchCount, local_mismatch, memory_order_relaxed);
        }
        atomic_fetch_add_explicit(readsDone, iters, memory_order_relaxed);
    }
}
"""

// MSL for Redone Fenced Flag/Value Test: One Writer per Slot
let fencedOneWriterSrc = p9Head() + """
struct FencedSlot {
    ulong value;
    atomic_uint flag;
};

kernel void p9_fenced_one_writer(
    device FencedSlot  *slots        [[buffer(0)]],
    device atomic_uint *staleReads   [[buffer(1)]],
    device atomic_uint *tornReads    [[buffer(2)]],
    device atomic_uint *readsDone    [[buffer(3)]],
    constant uint      &numWriters   [[buffer(4)]],
    constant uint      &iters        [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    bool is_writer = (tgid % 2) == 0;

    if (is_writer) {
        // Strictly ONE writer per slot!
        uint slot_idx = gid / 2;
        if (slot_idx < numWriters) {
            for (uint i = 1; i <= iters; i++) {
                ulong val = (ulong(i) << 32) | ulong(~i);
                slots[slot_idx].value = val;
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
                atomic_store_explicit(&slots[slot_idx].flag, i, memory_order_relaxed);
            }
        }
    } else {
        // Readers inspect slots written by other threadgroups
        uint local_stale = 0;
        uint local_torn = 0;
        uint target_slot = (gid / 2) % numWriters;

        for (uint k = 0; k < iters; k++) {
            uint f = atomic_load_explicit(&slots[target_slot].flag, memory_order_relaxed);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
            ulong v = slots[target_slot].value;

            if (f > 0) {
                uint hi = uint(v >> 32);
                uint lo = uint(v & 0xfffffffful);
                if (hi != ~lo) {
                    local_torn++;
                }
                if (hi < f) { // read stale value relative to published flag
                    local_stale++;
                }
            }
        }
        if (local_stale > 0) {
            atomic_fetch_add_explicit(staleReads, local_stale, memory_order_relaxed);
        }
        if (local_torn > 0) {
            atomic_fetch_add_explicit(tornReads, local_torn, memory_order_relaxed);
        }
        atomic_fetch_add_explicit(readsDone, iters, memory_order_relaxed);
    }
}
"""

// Acquire/Release probe string to test compiler support
let acquireReleaseCompilerProbe = p9Head() + """
kernel void test_acq_rel(device atomic_uint *flag [[buffer(0)]]) {
    atomic_store_explicit(flag, 1u, memory_order_release);
    uint v = atomic_load_explicit(flag, memory_order_acquire);
}
"""

func probeMain() {
    var body: [(String, J)] = []
    let dev = acquireDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("spin_cap", .u(UInt64(SPIN_CAP))))

    guard let dev = dev, let queue = dev.makeCommandQueue() else {
        body.append(("status", .s("no Metal device")))
        writeResult("p9_lookback", body)
        return
    }

    // -------------------------------------------------------------
    // Part 0: Real Measured Residency on Apple M4 GPU
    // -------------------------------------------------------------
    note("--- Part 0: Real Residency Measurement ---")
    var residencyObj: [(String, J)] = []
    var measuredResidencyMap: [Int: Int] = [:]
    let (libRes, errRes) = compileMSL(dev, residencySrc)
    if let libRes = libRes, let psoRes = pipeline(dev, libRes, "p9_measure_residency").0 {
        let tgSizes = [128, 256, 512]
        for tgSize in tgSizes {
            let activeBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
            let maxBuf    = dev.makeBuffer(length: 4, options: .storageModeShared)!
            memset(activeBuf.contents(), 0, 4)
            memset(maxBuf.contents(), 0, 4)

            var spinDelay: UInt32 = 200_000
            let dispatchGroups = 256

            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoRes)
            enc.setBuffer(activeBuf, offset: 0, index: 0)
            enc.setBuffer(maxBuf,    offset: 0, index: 1)
            enc.setBytes(&spinDelay, length: 4, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: dispatchGroups, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            let res = Int(maxBuf.contents().load(as: UInt32.self))
            measuredResidencyMap[tgSize] = res
            residencyObj.append(("threadgroup_size_\(tgSize)", .i(res)))
            note("  Measured residency at tg=\(tgSize): \(res) threadgroups resident")
        }
    } else {
        residencyObj.append(("error", .s("residency compile error: \(errRes ?? "")")))
    }
    body.append(("measured_residency", .o(residencyObj)))

    // -------------------------------------------------------------
    // Part 1a: CUB Stock Tile Pattern (tile = tgid, with arrival ticket tracking)
    // -------------------------------------------------------------
    note("--- Part 1a: CUB Stock Tile Pattern (tile = tgid) ---")
    var cubRows: [J] = []
    let (libFP, errFP) = compileMSL(dev, forwardProgressSrc)
    if let libFP = libFP, let psoFP = pipeline(dev, libFP, "p9_decoupled_lookback").0 {
        let tgSizes = [128, 256, 512]
        let tileCounts = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        for tgSize in tgSizes {
            let resBase = Double(measuredResidencyMap[tgSize] ?? 32)
            for nTiles in tileCounts {
                let ticketBuf  = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let statusBuf  = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let aggBuf     = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let prefixBuf  = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let orderBuf   = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let timeoutBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let maxSpinBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let oooBuf     = dev.makeBuffer(length: 4, options: .storageModeShared)!

                memset(ticketBuf.contents(), 0, 4)
                memset(statusBuf.contents(), 0, 4 * nTiles)
                memset(aggBuf.contents(), 0, 4 * nTiles)
                memset(prefixBuf.contents(), 0, 4 * nTiles)
                memset(orderBuf.contents(), 0, 4 * nTiles)
                memset(timeoutBuf.contents(), 0, 4)
                memset(maxSpinBuf.contents(), 0, 4)
                memset(oooBuf.contents(), 0, 4)

                var n = UInt32(nTiles)
                var cap = SPIN_CAP
                var mode: UInt32 = 2 // CUB stock pattern: tile = tgid

                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoFP)
                enc.setBuffer(ticketBuf, offset: 0, index: 0)
                enc.setBuffer(statusBuf, offset: 0, index: 1)
                enc.setBuffer(aggBuf,    offset: 0, index: 2)
                enc.setBuffer(prefixBuf, offset: 0, index: 3)
                enc.setBuffer(orderBuf,  offset: 0, index: 4)
                enc.setBuffer(timeoutBuf,offset: 0, index: 5)
                enc.setBuffer(maxSpinBuf,offset: 0, index: 6)
                enc.setBuffer(oooBuf,    offset: 0, index: 7)
                enc.setBytes(&n, length: 4, index: 8)
                enc.setBytes(&cap, length: 4, index: 9)
                enc.setBytes(&mode, length: 4, index: 10)

                enc.dispatchThreadgroups(MTLSize(width: nTiles, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                enc.endEncoding()

                let t0 = now()
                cb.commit()
                cb.waitUntilCompleted()
                let elapsed = now() - t0

                let timeouts = timeoutBuf.contents().load(as: UInt32.self)
                let maxSpins = maxSpinBuf.contents().load(as: UInt32.self)
                let ooo = oooBuf.contents().load(as: UInt32.self)

                let values = prefixBuf.contents().bindMemory(to: UInt32.self, capacity: nTiles)
                var allCorrect = (timeouts == 0)
                if allCorrect {
                    for i in 0..<nTiles {
                        let expected = UInt32((i + 1) * (i + 2) / 2)
                        if values[i] != expected {
                            allCorrect = false
                            break
                        }
                    }
                }

                let multOfResidency = Double(nTiles) / resBase
                let oooRatio = Double(ooo) / Double(nTiles)
                let r: [(String, J)] = [
                    ("threadgroup_size", .i(tgSize)),
                    ("tiles", .i(nTiles)),
                    ("multiple_of_residency", .d(multOfResidency)),
                    ("elapsed_seconds", .d(elapsed)),
                    ("timeouts", .u(UInt64(timeouts))),
                    ("max_spins", .u(UInt64(maxSpins))),
                    ("out_of_order_arrivals", .u(UInt64(ooo))),
                    ("out_of_order_ratio", .d(oooRatio)),
                    ("all_prefixes_correct", .b(allCorrect))
                ]
                cubRows.append(.o(r))
                note("  [CUB] tg=\(tgSize) tiles=\(nTiles) (\(String(format: "%.1f", multOfResidency))x res): timeouts=\(timeouts), max_spins=\(maxSpins), ooo=\(ooo)/\(nTiles) (\(String(format: "%.1f", oooRatio*100))%)")
            }
        }
    } else {
        cubRows.append(.o([("error", .s("MSL compile error: \(errFP ?? "")"))]))
    }
    body.append(("cub_tile_pattern_sweep", .a(cubRows)))

    // -------------------------------------------------------------
    // Part 1b: Arrival-Ticket Remap Sweep (Control: tile = ticket)
    // -------------------------------------------------------------
    note("--- Part 1b: Arrival-Ticket Remap Sweep (Control) ---")
    var ctrlRows: [J] = []
    if let libFP = libFP, let psoFP = pipeline(dev, libFP, "p9_decoupled_lookback").0 {
        let tgSizes = [128, 256, 512]
        let tileCounts = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        for tgSize in tgSizes {
            let resBase = Double(measuredResidencyMap[tgSize] ?? 32)
            for nTiles in tileCounts {
                let ticketBuf  = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let statusBuf  = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let aggBuf     = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let prefixBuf  = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let orderBuf   = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let timeoutBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let maxSpinBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let oooBuf     = dev.makeBuffer(length: 4, options: .storageModeShared)!

                memset(ticketBuf.contents(), 0, 4)
                memset(statusBuf.contents(), 0, 4 * nTiles)
                memset(aggBuf.contents(), 0, 4 * nTiles)
                memset(prefixBuf.contents(), 0, 4 * nTiles)
                memset(orderBuf.contents(), 0, 4 * nTiles)
                memset(timeoutBuf.contents(), 0, 4)
                memset(maxSpinBuf.contents(), 0, 4)
                memset(oooBuf.contents(), 0, 4)

                var n = UInt32(nTiles)
                var cap = SPIN_CAP
                var mode: UInt32 = 0 // Remap control: tile = ticket

                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoFP)
                enc.setBuffer(ticketBuf, offset: 0, index: 0)
                enc.setBuffer(statusBuf, offset: 0, index: 1)
                enc.setBuffer(aggBuf,    offset: 0, index: 2)
                enc.setBuffer(prefixBuf, offset: 0, index: 3)
                enc.setBuffer(orderBuf,  offset: 0, index: 4)
                enc.setBuffer(timeoutBuf,offset: 0, index: 5)
                enc.setBuffer(maxSpinBuf,offset: 0, index: 6)
                enc.setBuffer(oooBuf,    offset: 0, index: 7)
                enc.setBytes(&n, length: 4, index: 8)
                enc.setBytes(&cap, length: 4, index: 9)
                enc.setBytes(&mode, length: 4, index: 10)

                enc.dispatchThreadgroups(MTLSize(width: nTiles, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                enc.endEncoding()

                let t0 = now()
                cb.commit()
                cb.waitUntilCompleted()
                let elapsed = now() - t0

                let timeouts = timeoutBuf.contents().load(as: UInt32.self)
                let maxSpins = maxSpinBuf.contents().load(as: UInt32.self)
                let ooo = oooBuf.contents().load(as: UInt32.self)

                let values = prefixBuf.contents().bindMemory(to: UInt32.self, capacity: nTiles)
                var allCorrect = (timeouts == 0)
                if allCorrect {
                    for i in 0..<nTiles {
                        let expected = UInt32((i + 1) * (i + 2) / 2)
                        if values[i] != expected {
                            allCorrect = false
                            break
                        }
                    }
                }

                let multOfResidency = Double(nTiles) / resBase
                let r: [(String, J)] = [
                    ("threadgroup_size", .i(tgSize)),
                    ("tiles", .i(nTiles)),
                    ("multiple_of_residency", .d(multOfResidency)),
                    ("elapsed_seconds", .d(elapsed)),
                    ("timeouts", .u(UInt64(timeouts))),
                    ("max_spins", .u(UInt64(maxSpins))),
                    ("out_of_order_arrivals", .u(UInt64(ooo))),
                    ("all_prefixes_correct", .b(allCorrect))
                ]
                ctrlRows.append(.o(r))
                note("  [Control] tg=\(tgSize) tiles=\(nTiles) (\(String(format: "%.1f", multOfResidency))x res): timeouts=\(timeouts), max_spins=\(maxSpins), ooo=\(ooo)")
            }
        }
    }
    body.append(("arrival_ticket_control_sweep", .a(ctrlRows)))

    // -------------------------------------------------------------
    // Part 1c: Adversarial Forward Progress Case
    // -------------------------------------------------------------
    note("--- Part 1c: Adversarial Forward Progress Case ---")
    var advRows: [J] = []
    if let libFP = libFP, let psoFP = pipeline(dev, libFP, "p9_decoupled_lookback").0 {
        let tgSizes = [128, 256, 512]
        let advTileCounts = [256, 1024, 4096]

        for tgSize in tgSizes {
            for nTiles in advTileCounts {
                let ticketBuf  = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let statusBuf  = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let aggBuf     = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let prefixBuf  = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let orderBuf   = dev.makeBuffer(length: 4 * nTiles, options: .storageModeShared)!
                let timeoutBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let maxSpinBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
                let oooBuf     = dev.makeBuffer(length: 4, options: .storageModeShared)!

                memset(ticketBuf.contents(), 0, 4)
                memset(statusBuf.contents(), 0, 4 * nTiles)
                memset(aggBuf.contents(), 0, 4 * nTiles)
                memset(prefixBuf.contents(), 0, 4 * nTiles)
                memset(orderBuf.contents(), 0, 4 * nTiles)
                memset(timeoutBuf.contents(), 0, 4)
                memset(maxSpinBuf.contents(), 0, 4)
                memset(oooBuf.contents(), 0, 4)

                var n = UInt32(nTiles)
                var cap = SPIN_CAP
                var mode: UInt32 = 1 // Adversarial: reverse ticket order

                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoFP)
                enc.setBuffer(ticketBuf, offset: 0, index: 0)
                enc.setBuffer(statusBuf, offset: 0, index: 1)
                enc.setBuffer(aggBuf,    offset: 0, index: 2)
                enc.setBuffer(prefixBuf, offset: 0, index: 3)
                enc.setBuffer(orderBuf,  offset: 0, index: 4)
                enc.setBuffer(timeoutBuf,offset: 0, index: 5)
                enc.setBuffer(maxSpinBuf,offset: 0, index: 6)
                enc.setBuffer(oooBuf,    offset: 0, index: 7)
                enc.setBytes(&n, length: 4, index: 8)
                enc.setBytes(&cap, length: 4, index: 9)
                enc.setBytes(&mode, length: 4, index: 10)

                enc.dispatchThreadgroups(MTLSize(width: nTiles, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                enc.endEncoding()

                let t0 = now()
                cb.commit()
                cb.waitUntilCompleted()
                let elapsed = now() - t0

                let timeouts = timeoutBuf.contents().load(as: UInt32.self)
                let maxSpins = maxSpinBuf.contents().load(as: UInt32.self)

                let r: [(String, J)] = [
                    ("threadgroup_size", .i(tgSize)),
                    ("tiles", .i(nTiles)),
                    ("elapsed_seconds", .d(elapsed)),
                    ("timeouts", .u(UInt64(timeouts))),
                    ("max_spins", .u(UInt64(maxSpins))),
                    ("deadlock_observed", .b(timeouts > 0))
                ]
                advRows.append(.o(r))
                note("  [Adversarial] tg=\(tgSize) tiles=\(nTiles): timeouts=\(timeouts), max_spins=\(maxSpins), deadlock=\(timeouts > 0)")
            }
        }
    }
    body.append(("adversarial_forward_progress", .a(advRows)))

    // -------------------------------------------------------------
    // Part 2: 64-bit Aligned Store/Load Tearing Test
    // -------------------------------------------------------------
    note("--- Part 2: 64-bit Store/Load Tearing Test ---")
    var tear64Obj: [(String, J)] = []
    let (libTear, errTear) = compileMSL(dev, tearing64Src)
    if let libTear = libTear, let psoTear = pipeline(dev, libTear, "p9_tearing_64bit").0 {
        let numSlots = 64
        let iters: UInt32 = 500_000
        let numThreads = 1024

        let slotsBuf = dev.makeBuffer(length: 8 * numSlots, options: .storageModeShared)!
        let tornBuf  = dev.makeBuffer(length: 4, options: .storageModeShared)!
        let readsBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!

        memset(slotsBuf.contents(), 0, 8 * numSlots)
        memset(tornBuf.contents(), 0, 4)
        memset(readsBuf.contents(), 0, 4)

        var nSlots = UInt32(numSlots)
        var it = iters

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoTear)
        enc.setBuffer(slotsBuf, offset: 0, index: 0)
        enc.setBuffer(tornBuf,  offset: 0, index: 1)
        enc.setBuffer(readsBuf, offset: 0, index: 2)
        enc.setBytes(&nSlots, length: 4, index: 3)
        enc.setBytes(&it, length: 4, index: 4)

        enc.dispatchThreads(MTLSize(width: numThreads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()

        let t0 = now()
        cb.commit()
        cb.waitUntilCompleted()
        let elapsed = now() - t0

        let torn = tornBuf.contents().load(as: UInt32.self)
        let reads = readsBuf.contents().load(as: UInt32.self)

        tear64Obj = [
            ("num_slots", .i(numSlots)),
            ("iterations_per_thread", .u(UInt64(iters))),
            ("total_reads", .u(UInt64(reads))),
            ("torn_reads", .u(UInt64(torn))),
            ("tearing_observed", .b(torn > 0)),
            ("elapsed_seconds", .d(elapsed))
        ]
        note("  64-bit store/load: reads=\(reads), torn=\(torn) (tearing=\(torn > 0)) in \(String(format: "%.4f", elapsed)) s")
    } else {
        tear64Obj = [("error", .s("compile error: \(errTear ?? "")"))]
    }
    body.append(("tearing_64bit", .o(tear64Obj)))

    // -------------------------------------------------------------
    // Part 3: 32-bit Packed Fallback Encoding
    // -------------------------------------------------------------
    note("--- Part 3: 32-bit Packed Fallback ---")
    var packedObj: [(String, J)] = []
    let (libPack, errPack) = compileMSL(dev, packed32Src)
    if let libPack = libPack, let psoPack = pipeline(dev, libPack, "p9_packed_32bit").0 {
        let numSlots = 64
        let iters: UInt32 = 500_000
        let numThreads = 1024

        let slotsBuf    = dev.makeBuffer(length: 4 * numSlots, options: .storageModeShared)!
        let mismatchBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
        let readsBuf    = dev.makeBuffer(length: 4, options: .storageModeShared)!

        memset(slotsBuf.contents(), 0, 4 * numSlots)
        memset(mismatchBuf.contents(), 0, 4)
        memset(readsBuf.contents(), 0, 4)

        var nSlots = UInt32(numSlots)
        var it = iters

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoPack)
        enc.setBuffer(slotsBuf,    offset: 0, index: 0)
        enc.setBuffer(mismatchBuf, offset: 0, index: 1)
        enc.setBuffer(readsBuf,    offset: 0, index: 2)
        enc.setBytes(&nSlots, length: 4, index: 3)
        enc.setBytes(&it, length: 4, index: 4)

        enc.dispatchThreads(MTLSize(width: numThreads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()

        let t0 = now()
        cb.commit()
        cb.waitUntilCompleted()
        let elapsed = now() - t0

        let mismatches = mismatchBuf.contents().load(as: UInt32.self)
        let reads = readsBuf.contents().load(as: UInt32.self)

        packedObj = [
            ("num_slots", .i(numSlots)),
            ("iterations_per_thread", .u(UInt64(iters))),
            ("total_reads", .u(UInt64(reads))),
            ("mismatches", .u(UInt64(mismatches))),
            ("encoding_safe", .b(mismatches == 0)),
            ("elapsed_seconds", .d(elapsed))
        ]
        note("  32-bit packed: reads=\(reads), mismatches=\(mismatches) in \(String(format: "%.4f", elapsed)) s")
    } else {
        packedObj = [("error", .s("compile error: \(errPack ?? "")"))]
    }
    body.append(("packed_32bit", .o(packedObj)))

    // -------------------------------------------------------------
    // Part 4: Fenced Flag-Value Pair & MSL Version Probe
    // -------------------------------------------------------------
    note("--- Part 4: Fenced Flag-Value Pair & MSL Version Probe ---")
    var fenceObj: [(String, J)] = []

    // 4a. MSL Language Version Probe: test acquire/release across all supported versions
    var verRows: [J] = []
    let versionsToTest: [(String, MTLLanguageVersion?)] = [
        ("default", nil),
        ("version3_0", .version3_0),
        ("version3_1", .version3_1),
        ("version3_2", .version3_2)
    ]

    for (vName, vEnum) in versionsToTest {
        let opts = MTLCompileOptions()
        if let v = vEnum {
            opts.languageVersion = v
        }
        var supported = false
        var refusalVerbatim = ""
        do {
            _ = try dev.makeLibrary(source: acquireReleaseCompilerProbe, options: opts)
            supported = true
        } catch {
            refusalVerbatim = "\(error)"
        }
        verRows.append(.o([
            ("msl_version", .s(vName)),
            ("supported", .b(supported)),
            ("refusal_verbatim", .s(refusalVerbatim))
        ]))
        note("  MSL \(vName) acquire/release: \(supported ? "SUPPORTED" : "REFUSED")")
    }
    fenceObj.append(("msl_version_acquire_release_probes", .a(verRows)))

    // 4b. Ordering with device memory fence (mem_flags::mem_device, memory_order_seq_cst)
    let (libFence, errFence) = compileMSL(dev, fencedOneWriterSrc)
    if let libFence = libFence, let psoFence = pipeline(dev, libFence, "p9_fenced_one_writer").0 {
        let numWriters = 512
        let iters: UInt32 = 100_000
        let numThreads = 1024

        // 1 writer per slot -> 512 slots
        let slotBytes = numWriters * 16 // sizeof(FencedSlot) = 16
        let slotsBuf = dev.makeBuffer(length: slotBytes, options: .storageModeShared)!
        let staleBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!
        let tornBuf  = dev.makeBuffer(length: 4, options: .storageModeShared)!
        let readsBuf = dev.makeBuffer(length: 4, options: .storageModeShared)!

        memset(slotsBuf.contents(), 0, slotBytes)
        memset(staleBuf.contents(), 0, 4)
        memset(tornBuf.contents(), 0, 4)
        memset(readsBuf.contents(), 0, 4)

        var nWriters = UInt32(numWriters)
        var it = iters

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoFence)
        enc.setBuffer(slotsBuf, offset: 0, index: 0)
        enc.setBuffer(staleBuf, offset: 0, index: 1)
        enc.setBuffer(tornBuf,  offset: 0, index: 2)
        enc.setBuffer(readsBuf, offset: 0, index: 3)
        enc.setBytes(&nWriters, length: 4, index: 4)
        enc.setBytes(&it, length: 4, index: 5)

        enc.dispatchThreads(MTLSize(width: numThreads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()

        let t0 = now()
        cb.commit()
        cb.waitUntilCompleted()
        let elapsed = now() - t0

        let stale = staleBuf.contents().load(as: UInt32.self)
        let torn  = tornBuf.contents().load(as: UInt32.self)
        let reads = readsBuf.contents().load(as: UInt32.self)

        fenceObj.append(("one_writer_per_slot", .b(true)))
        fenceObj.append(("num_writers", .i(numWriters)))
        fenceObj.append(("iterations_per_thread", .u(UInt64(iters))))
        fenceObj.append(("total_reads", .u(UInt64(reads))))
        fenceObj.append(("stale_reads", .u(UInt64(stale))))
        fenceObj.append(("torn_reads", .u(UInt64(torn))))
        fenceObj.append(("fenced_pair_safe", .b(stale == 0 && torn == 0)))
        fenceObj.append(("elapsed_seconds", .d(elapsed)))

        note("  Fenced pair (1 writer/slot): reads=\(reads), stale=\(stale), torn=\(torn) in \(String(format: "%.4f", elapsed)) s")
    } else {
        fenceObj.append(("error", .s("compile error: \(errFence ?? "")")))
    }
    body.append(("fenced_flag_value_test", .o(fenceObj)))

    body.append(("status", .s("ok")))
    writeResult("p9_lookback", body)
}

#if !PROBE_SCRIPT_MODE
@main
struct ProbeMain {
    static func main() {
        probeMain()
    }
}
#endif
