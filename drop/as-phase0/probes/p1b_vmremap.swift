// p1b_vmremap -- can one heap be aliased at ONE address?  (vm_remap variant.)
//
// This is the single question blueprint §2.3 hangs cudaMallocManaged on, and
// G-C0 names it first: a CUDA managed pointer is valid on host and device, but
// a Metal shared buffer has a CPU address (`contents`) and a GPU address
// (`gpuAddress`) that differ.  §2.3's first mechanism is to map the buffer's
// own pages a second time, at the GPU virtual address, inside this process --
// after which one number dereferences correctly on both sides and managed
// memory costs nothing.  The fallback if this fails is compiler-inserted
// rebasing of every pointer load in kernels that touch managed allocations,
// at a cost that has to be measured separately.
//
// Split out of p1_heap deliberately.  The mach VM call is the least portable
// thing in this drop and the one most likely to disagree with a given SDK; if
// it fails to compile, the failure must not cost p1_heap's four other answers.
// p1c_machvmremap asks the same question through mach_vm_remap, so one round
// trip gets an answer even if one of the two spellings will not build.
//
// THE MACHINE RULE: whether the GPU address range is free in this process is a
// property of this OS build and this process's layout, not of Apple silicon.
// A failure here is evidence, not a verdict, and must be repeated on another
// macOS version before §2.3 picks the rebasing fallback.
//
// Bounded work: no loops beyond a fixed word-by-word comparison of 16 MiB.

import Foundation
import Metal
import Darwin   // vm_remap, mach_task_self_: the mechanism under test

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

/// The handful of mach codes this probe can actually provoke, named so the
/// report does not need a lookup table.
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
    let dev = MTLCreateSystemDefaultDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("api", .s("vm_remap")))
    body.append(("alias_length", .i(ALIAS_LEN)))

    guard let dev = dev else {
        body.append(("status", .s("no Metal device")))
        writeResult("p1b_vmremap", body)
        return
    }
    guard let buf = dev.makeBuffer(length: ALIAS_LEN, options: .storageModeShared) else {
        body.append(("status", .s("could not allocate the \(ALIAS_LEN) B buffer")))
        writeResult("p1b_vmremap", body)
        return
    }

    let cpu = buf.contents()
    let cpuAddr = UInt(bitPattern: cpu)
    let gpuAddr = buf.gpuAddress
    body.append(("cpu_address", .u(UInt64(cpuAddr))))
    body.append(("gpu_address", .u(gpuAddr)))
    body.append(("already_equal", .b(UInt64(cpuAddr) == gpuAddr)))
    note("cpu=0x\(String(cpuAddr, radix: 16))  gpu=0x\(String(gpuAddr, radix: 16))")

    // Ask for the buffer's own pages a second time, FIXED at the GPU address.
    // flags 0 is VM_FLAGS_FIXED; inheritance 0 is VM_INHERIT_SHARE; copy 0
    // means share the pages rather than copy-on-write them, which is the whole
    // point -- a copy would alias the address without aliasing the memory.
    var target = vm_address_t(gpuAddr)
    var cur = vm_prot_t(0)
    var maxp = vm_prot_t(0)
    let kr: kern_return_t = vm_remap(
        mach_task_self_,
        &target,
        vm_size_t(ALIAS_LEN),
        vm_address_t(0),          // mask: no alignment constraint
        0,                        // VM_FLAGS_FIXED
        mach_task_self_,
        vm_address_t(cpuAddr),
        0,                        // copy = false: share the physical pages
        &cur, &maxp,
        vm_inherit_t(0))          // VM_INHERIT_SHARE

    body.append(("vm_remap_kr", .i(Int(kr))))
    body.append(("vm_remap_kr_name", .s(krName(kr))))
    body.append(("alias_succeeded", .b(kr == 0)))
    body.append(("target_address_returned", .u(UInt64(target))))
    body.append(("target_is_gpu_address", .b(kr == 0 && UInt64(target) == gpuAddr)))
    body.append(("cur_protection", .i(Int(cur))))
    body.append(("max_protection", .i(Int(maxp))))
    note("vm_remap -> \(krName(kr)) (target 0x\(String(target, radix: 16)))")

    // Checkpoint.  Everything after this dereferences an address the kernel
    // just told us about; if that faults, the answer above must still survive
    // the trip back, because this drop gets one round trip.
    body.append(("phase", .s("remap_result_only")))
    writeResult("p1b_vmremap", body)
    body.removeLast()

    if kr == 0 && UInt64(target) == gpuAddr {
        // The GPU writes through the buffer; the CPU reads through the alias
        // at the GPU address.  Same pages or not is now a fact, not a claim.
        var readback: [(String, J)] = []
        let (lib, cerr) = compileMSL(dev, stampSrc)
        if let lib = lib, let queue = dev.makeCommandQueue() {
            let (pso, perr) = pipeline(dev, lib, "stamp")
            if let pso = pso {
                let n = UInt32(ALIAS_LEN / 4)
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
                        note("alias readback: \(bad == 0 ? "identical" : "\(bad) mismatched words")")
                    } else {
                        readback.append(("error", .s("target address is not a usable pointer")))
                    }
                }
            } else {
                readback.append(("pipeline_error", jOptS(perr)))
            }
        } else {
            readback.append(("compile_error", jOptS(cerr)))
        }
        body.append(("alias_readback", .o(readback)))
        // Leave the alias mapped: the process is about to exit and unmapping a
        // range Metal also owns is a good way to turn a result into a crash.
    } else {
        body.append(("alias_readback", .o([
            ("skipped", .b(true)),
            ("reason", .s("the alias was not established, so there is nothing to read through")),
        ])))
    }

    body.append(("phase", .s("complete")))
    body.append(("status", .s("ok")))
    writeResult("p1b_vmremap", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
