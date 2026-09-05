// p1_heap -- the device heap of blueprint §2.3, minus the vm_remap alias.
//
// Four questions, in the order the design depends on them:
//   1. How large can ONE MTLBuffer be?  §2.3's heap is one buffer per device,
//      sub-allocated at 64 KiB, so this is the ceiling on cudaMalloc before the
//      tiers take over.  G-C0 asks whether it reaches 75 % of RAM.
//   2. Is `gpuAddress` readable?  A CUDA device pointer is that address plus an
//      offset; without it there is no pointer arithmetic in kernels.
//   3. Does `makeBuffer(bytesNoCopy:)` accept pages this process already owns?
//      That is the mechanism for making host allocations kernel-bindable.
//   4. Do the CPU address and the GPU address of those same pages COINCIDE?
//      If they do, cudaMallocManaged is free; if they do not, §2.3's second
//      mechanism (the vm_remap alias, p1b/p1c) or compiler rebasing is needed.
//
// THE MACHINE RULE: one sample.  maxBufferLength and the working-set limit are
// per-chip and per-OS; a 8 GB M1 and a 512 GB M5 Ultra sit at opposite ends and
// the design must hold at both.  Report, do not generalise.
//
// Bounded work: the bisect runs at most 40 probe allocations (BISECT_STEPS);
// no kernel here loops.

import Foundation
import Metal
import Darwin   // mmap / munmap / getpagesize: the pages under question 3

let BISECT_STEPS = 40
let GRAIN = 64 * 1024            // lmz's block, and vram/'s residency unit

let touchSrc = """
#include <metal_stdlib>
using namespace metal;
// Writes a checkable pattern so the CPU can tell whether it is looking at the
// same physical pages the GPU wrote.
kernel void stamp(device uint *p [[buffer(0)]],
                  constant uint &n [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid < n) p[gid] = gid * 2654435761u + 12345u;
}
"""

func expectedStamp(_ i: UInt32) -> UInt32 {
    return i &* 2654435761 &+ 12345
}

/// Largest single buffer this device will hand out, by bisection from
/// maxBufferLength.  Never touched, so the pages stay unbacked and the probe
/// does not push the machine into swap.
func largestBuffer(_ dev: MTLDevice, _ opts: MTLResourceOptions) -> (Int, Int) {
    var lo = 0
    var hi = dev.maxBufferLength
    var tries = 0
    // Fast path: the documented ceiling usually works.
    var ok = false
    autoreleasepool {
        var b: MTLBuffer? = dev.makeBuffer(length: hi, options: opts)
        ok = (b != nil)
        b = nil
    }
    tries += 1
    if ok { return (hi, tries) }
    while lo + GRAIN < hi && tries < BISECT_STEPS {
        let mid = lo + ((hi - lo) / 2 / GRAIN) * GRAIN
        if mid <= lo { break }
        var got = false
        autoreleasepool {
            var b: MTLBuffer? = dev.makeBuffer(length: mid, options: opts)
            got = (b != nil)
            b = nil
        }
        tries += 1
        if got { lo = mid } else { hi = mid }
    }
    return (lo, tries)
}

func probeMain() {
    var body: [(String, J)] = []
    let dev = acquireDevice()
    body.append(("machine", machineBlock(dev)))
    guard let dev = dev else {
        body.append(("status", .s("no Metal device")))
        writeResult("p1_heap", body)
        return
    }

    let ram = Double(ProcessInfo.processInfo.physicalMemory)
    body.append(("bisect_steps_cap", .i(BISECT_STEPS)))
    body.append(("suballocation_grain_bytes", .i(GRAIN)))

    // ---- 1. largest single buffer -------------------------------------
    var rows: [J] = []
    for (label, opts) in [("shared", MTLResourceOptions.storageModeShared),
                          ("private", MTLResourceOptions.storageModePrivate)] {
        let t0 = now()
        let (best, tries) = largestBuffer(dev, opts)
        rows.append(.o([
            ("storage_mode", .s(label)),
            ("largest_bytes", .i(best)),
            ("largest_gib", .d(Double(best) / 1073741824.0)),
            ("fraction_of_ram", .d(Double(best) / ram)),
            ("reached_max_buffer_length", .b(best == dev.maxBufferLength)),
            ("probe_allocations", .i(tries)),
            ("seconds", .d(now() - t0)),
        ]))
        note("largest \(label) buffer: \(best) B (\(String(format: "%.1f", Double(best) / 1073741824.0)) GiB)")
    }
    body.append(("largest_single_buffer", .a(rows)))

    // ---- 2. gpuAddress on an ordinary buffer ---------------------------
    let probeLen = 64 * 1024 * 1024
    var gpuAddrOK = false
    var normalGPUAddr: UInt64 = 0
    var normalCPUAddr: UInt64 = 0
    if let b = dev.makeBuffer(length: probeLen, options: .storageModeShared) {
        normalGPUAddr = b.gpuAddress
        normalCPUAddr = UInt64(UInt(bitPattern: b.contents()))
        gpuAddrOK = normalGPUAddr != 0
        body.append(("ordinary_buffer", .o([
            ("length", .i(probeLen)),
            ("gpu_address", .u(normalGPUAddr)),
            ("cpu_address", .u(normalCPUAddr)),
            ("gpu_address_nonzero", .b(gpuAddrOK)),
            ("addresses_coincide", .b(normalGPUAddr == normalCPUAddr)),
            ("gpu_minus_cpu", .s(String(format: "0x%llx",
                normalGPUAddr &- normalCPUAddr))),
        ])))
        note("ordinary buffer: cpu=0x\(String(normalCPUAddr, radix: 16)) gpu=0x\(String(normalGPUAddr, radix: 16))")
    } else {
        body.append(("ordinary_buffer", .o([("error", .s("allocation of \(probeLen) B failed"))])))
    }

    // ---- 3 + 4. bytesNoCopy over pages this process owns ---------------
    // mmap gives page-aligned anonymous pages; Metal requires page alignment
    // for bytesNoCopy and this is the documented way to get it.  The
    // deallocator is nil: this probe unmaps by hand after the buffer is gone.
    let pageSize = Int(getpagesize())
    let mapLen = ((16 * 1024 * 1024) / pageSize) * pageSize
    let rawOpt: UnsafeMutableRawPointer? = mmap(nil, mapLen,
                   PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
    // -1 is MAP_FAILED.  Compared by bit pattern rather than against the
    // MAP_FAILED constant, whose Swift type has moved between SDK versions and
    // which this drop cannot afford to get wrong.
    if rawOpt == nil || Int(bitPattern: rawOpt) == -1 {
        body.append(("bytes_no_copy", .o([
            ("supported", .b(false)),
            ("error", .s("mmap of \(mapLen) B failed, errno \(errno)")),
        ])))
    } else {
        let ptr = rawOpt!
        memset(ptr, 0, mapLen)
        var nc: [(String, J)] = []
        nc.append(("map_length", .i(mapLen)))
        nc.append(("page_size", .i(pageSize)))
        nc.append(("cpu_address", .u(UInt64(UInt(bitPattern: ptr)))))

        let buf = dev.makeBuffer(bytesNoCopy: ptr, length: mapLen,
                                 options: .storageModeShared, deallocator: nil)
        nc.append(("supported", .b(buf != nil)))
        if let buf = buf {
            let ga = buf.gpuAddress
            let ca = UInt64(UInt(bitPattern: buf.contents()))
            nc.append(("gpu_address", .u(ga)))
            nc.append(("buffer_contents_address", .u(ca)))
            nc.append(("contents_equals_mapped_pointer",
                       .b(ca == UInt64(UInt(bitPattern: ptr)))))
            // THE §2.3 QUESTION: is the GPU virtual address the same number as
            // the CPU one?  If yes, cudaMallocManaged needs no machinery at all.
            nc.append(("gpu_address_equals_cpu_address", .b(ga == ca)))
            nc.append(("gpu_minus_cpu", .s(String(format: "0x%llx", ga &- ca))))

            // Coherence: the GPU stamps a pattern, the CPU reads it back
            // through the ORIGINAL mmap pointer, not through buf.contents().
            let (lib, cerr) = compileMSL(dev, touchSrc)
            if let lib = lib, let queue = dev.makeCommandQueue() {
                let (pso, perr) = pipeline(dev, lib, "stamp")
                if let pso = pso {
                    let n = UInt32(mapLen / 4)
                    var nvar = n
                    if let cb = queue.makeCommandBuffer(),
                       let enc = cb.makeComputeCommandEncoder() {
                        enc.setComputePipelineState(pso)
                        enc.setBuffer(buf, offset: 0, index: 0)
                        enc.setBytes(&nvar, length: 4, index: 1)
                        let w = pso.maxTotalThreadsPerThreadgroup
                        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
                        enc.endEncoding()
                        cb.commit()
                        cb.waitUntilCompleted()
                        if let e = cb.error {
                            nc.append(("kernel_error", .s("\(e)")))
                        } else {
                            let words = ptr.bindMemory(to: UInt32.self, capacity: Int(n))
                            var bad = 0
                            var i: UInt32 = 0
                            while i < n {
                                if words[Int(i)] != expectedStamp(i) { bad += 1 }
                                i += 1
                            }
                            nc.append(("gpu_write_visible_to_cpu", .b(bad == 0)))
                            nc.append(("mismatched_words", .i(bad)))
                            nc.append(("words_checked", .i(Int(n))))
                            note("bytesNoCopy coherence: \(bad == 0 ? "GPU writes visible through the mmap pointer" : "\(bad) mismatched words")")
                        }
                    }
                } else {
                    nc.append(("pipeline_error", jOptS(perr)))
                }
            } else {
                nc.append(("compile_error", jOptS(cerr)))
            }
        } else {
            nc.append(("error", .s("makeBuffer(bytesNoCopy:) returned nil")))
        }
        body.append(("bytes_no_copy", .o(nc)))
        munmap(ptr, mapLen)
    }

    body.append(("status", .s("ok")))
    writeResult("p1_heap", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
