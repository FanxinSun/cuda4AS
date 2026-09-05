// p1c_machvmremap -- the same §2.3 alias question as p1b, asked through
// mach_vm_remap instead of vm_remap.
//
// Not redundancy for its own sake.  The two calls differ in their address and
// size types (mach_vm_address_t/mach_vm_size_t are 64-bit everywhere; vm_
// address_t/vm_size_t follow the pointer width), and which of the two the
// Swift Darwin overlay exposes cleanly has moved between SDK versions.  This
// drop cannot be compiled on the machine that wrote it and gets ONE round
// trip, so the most important question in G-C0 is asked twice, in two
// spellings, and whichever builds answers it.
//
// If both build and both run, the two JSON files should agree; if they do not,
// that disagreement is itself the finding.
//
// THE MACHINE RULE: as p1b -- a free or occupied GPU address range is a fact
// about this OS build and this process, not about Apple silicon.
//
// Bounded work: no loops beyond a fixed word-by-word comparison of 16 MiB.

import Foundation
import Metal
import Darwin   // mach_vm_remap, mach_task_self_

let ALIAS_LEN = 16 * 1024 * 1024

let stampSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void stamp(device uint *p [[buffer(0)]],
                  constant uint &n [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid < n) p[gid] = gid * 2246822519u + 7u;
}
"""

func expectedStamp(_ i: UInt32) -> UInt32 { return i &* 2246822519 &+ 7 }

func krName(_ kr: kern_return_t) -> String {
    switch kr {
    case 0: return "KERN_SUCCESS"
    case 1: return "KERN_INVALID_ADDRESS"
    case 2: return "KERN_PROTECTION_FAILURE"
    case 3: return "KERN_NO_SPACE"
    case 4: return "KERN_INVALID_ARGUMENT"
    case 5: return "KERN_FAILURE"
    case 8: return "KERN_NO_ACCESS"
    case 9: return "KERN_MEMORY_FAILURE"
    default: return "kern_return_t \(kr)"
    }
}

func probeMain() {
    var body: [(String, J)] = []
    let dev = acquireDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("api", .s("mach_vm_remap")))
    body.append(("alias_length", .i(ALIAS_LEN)))

    guard let dev = dev else {
        body.append(("status", .s("no Metal device")))
        writeResult("p1c_machvmremap", body)
        return
    }
    guard let buf = dev.makeBuffer(length: ALIAS_LEN, options: .storageModeShared) else {
        body.append(("status", .s("could not allocate the \(ALIAS_LEN) B buffer")))
        writeResult("p1c_machvmremap", body)
        return
    }

    let cpuAddr = UInt(bitPattern: buf.contents())
    let gpuAddr = buf.gpuAddress
    body.append(("cpu_address", .u(UInt64(cpuAddr))))
    body.append(("gpu_address", .u(gpuAddr)))
    body.append(("already_equal", .b(UInt64(cpuAddr) == gpuAddr)))

    var target = mach_vm_address_t(gpuAddr)
    var cur = vm_prot_t(0)
    var maxp = vm_prot_t(0)
    let kr: kern_return_t = mach_vm_remap(
        mach_task_self_,
        &target,
        mach_vm_size_t(ALIAS_LEN),
        mach_vm_offset_t(0),      // mask: no alignment constraint
        0,                        // VM_FLAGS_FIXED
        mach_task_self_,
        mach_vm_address_t(cpuAddr),
        0,                        // copy = false: share the physical pages
        &cur, &maxp,
        vm_inherit_t(0))          // VM_INHERIT_SHARE

    body.append(("mach_vm_remap_kr", .i(Int(kr))))
    body.append(("mach_vm_remap_kr_name", .s(krName(kr))))
    body.append(("alias_succeeded", .b(kr == 0)))
    body.append(("target_address_returned", .u(UInt64(target))))
    body.append(("target_is_gpu_address", .b(kr == 0 && UInt64(target) == gpuAddr)))
    body.append(("cur_protection", .i(Int(cur))))
    body.append(("max_protection", .i(Int(maxp))))
    note("mach_vm_remap -> \(krName(kr)) (target 0x\(String(target, radix: 16)))")

    // Checkpoint before dereferencing, exactly as p1b does and for the same
    // reason: the remap verdict must survive a fault in the readback.
    body.append(("phase", .s("remap_result_only")))
    writeResult("p1c_machvmremap", body)
    body.removeLast()

    if kr == 0 && UInt64(target) == gpuAddr {
        var readback: [(String, J)] = []
        let (lib, cerr) = compileMSL(dev, stampSrc)
        if let lib = lib, let queue = dev.makeCommandQueue() {
            let (pso, perr) = pipeline(dev, lib, "stamp")
            if let pso = pso, let cb = queue.makeCommandBuffer(),
               let enc = cb.makeComputeCommandEncoder() {
                let n = UInt32(ALIAS_LEN / 4)
                var nvar = n
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
                    readback.append(("kernel_error", .s("\(e)")))
                } else if let aliasPtr = UnsafeMutableRawPointer(bitPattern: UInt(target)) {
                    let words = aliasPtr.bindMemory(to: UInt32.self, capacity: Int(n))
                    var bad = 0
                    var i: UInt32 = 0
                    while i < n {
                        if words[Int(i)] != expectedStamp(i) { bad += 1 }
                        i += 1
                    }
                    readback.append(("words_checked", .i(Int(n))))
                    readback.append(("mismatched_words", .i(bad)))
                    readback.append(("alias_reads_gpu_writes", .b(bad == 0)))
                } else {
                    readback.append(("error", .s("target address is not a usable pointer")))
                }
            } else {
                readback.append(("pipeline_error", jOptS(perr)))
            }
        } else {
            readback.append(("compile_error", jOptS(cerr)))
        }
        body.append(("alias_readback", .o(readback)))
    } else {
        body.append(("alias_readback", .o([
            ("skipped", .b(true)),
            ("reason", .s("the alias was not established")),
        ])))
    }

    body.append(("phase", .s("complete")))
    body.append(("status", .s("ok")))
    writeResult("p1c_machvmremap", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
