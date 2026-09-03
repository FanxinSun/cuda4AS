// p0_device -- what this Apple GPU says about itself.
//
// Answers the "machine profile" half of blueprint §2.8 and the reported-device-
// properties half of §2.3: a CUDA program asks for multiprocessor count, shared
// memory per block and total device memory, and Metal answers none of those
// directly.  This probe records everything Metal WILL answer, so the profile is
// measured rather than constant.
//
// THE MACHINE RULE: this is one sample of Apple silicon.  Nothing here is a
// constant of the design; the numbers are inputs to a profile that must also
// hold on an M1 (32 KB threadgroup memory, no accelerators) and on chips that
// do not exist yet.  A family reported false is a fact about this chip.
//
// Bounded work: none -- this probe launches one trivial pipeline and no loops.

import Foundation
import Metal

// The trivial kernel exists only so that threadExecutionWidth and
// maxTotalThreadsPerThreadgroup can be read off a real pipeline: they are
// pipeline properties, not device properties, and the CUDA notion of "warp
// size" lives here rather than on MTLDevice.
let trivialSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void nop(device uint *out [[buffer(0)]],
                uint gid [[thread_position_in_grid]]) {
    out[gid] = gid;
}
"""

func probeMain() {
    var body: [(String, J)] = []

    let dev = MTLCreateSystemDefaultDevice()
    body.append(("machine", machineBlock(dev)))

    guard let dev = dev else {
        body.append(("status", .s("no Metal device -- nothing else can be measured")))
        writeResult("p0_device", body)
        return
    }
    note("device: \(dev.name)")

    // Every Metal device the system has, not just the default one.  A Mac with
    // an eGPU or more than one GPU is a different residency problem, and the
    // profile has to know.
    var devs: [J] = []
    for d in MTLCopyAllDevices() {
        devs.append(.o([
            ("name", .s(d.name)),
            ("registry_id", .u(d.registryID)),
            ("has_unified_memory", .b(d.hasUnifiedMemory)),
            ("low_power", .b(d.isLowPower)),
            ("removable", .b(d.isRemovable)),
            ("headless", .b(d.isHeadless)),
            ("max_buffer_length", .i(d.maxBufferLength)),
            ("recommended_max_working_set_size", .u(d.recommendedMaxWorkingSetSize)),
        ]))
    }
    body.append(("all_devices", .a(devs)))

    // The two ratios G-C0 asks for directly.
    let ram = Double(ProcessInfo.processInfo.physicalMemory)
    body.append(("working_set_fraction_of_ram",
                 .d(Double(dev.recommendedMaxWorkingSetSize) / ram)))
    body.append(("max_buffer_fraction_of_ram",
                 .d(Double(dev.maxBufferLength) / ram)))

    // Argument-buffer and resource limits the runtime's argument tables need.
    body.append(("max_threadgroup_memory_length", .i(dev.maxThreadgroupMemoryLength)))
    body.append(("cuda_static_shared_request", .i(48 * 1024)))
    body.append(("static_48kb_fits_natively",
                 .b(dev.maxThreadgroupMemoryLength >= 48 * 1024)))
    body.append(("argument_buffers_support", .i(Int(dev.argumentBuffersSupport.rawValue))))
    body.append(("supports_32bit_float_filtering", .b(dev.supports32BitFloatFiltering)))
    body.append(("supports_shader_barycentric_coordinates",
                 .b(dev.supportsShaderBarycentricCoordinates)))
    body.append(("supports_dynamic_libraries", .b(dev.supportsDynamicLibraries)))
    body.append(("supports_function_pointers", .b(dev.supportsFunctionPointers)))
    body.append(("supports_raytracing", .b(dev.supportsRaytracing)))
    body.append(("max_argument_buffer_sampler_count",
                 .i(dev.maxArgumentBufferSamplerCount)))
    body.append(("current_allocated_size", .i(dev.currentAllocatedSize)))

    // Which MSL language versions this toolchain will accept, asked by raw
    // value so a version newer than this file still gets asked about.  This is
    // the gate on §2.1's "emit MSL and compile it": if the Metal 4 language
    // version is refused, the tensor-op path of p6 cannot exist on this OS
    // whatever the hardware can do.
    var versions: [J] = []
    for (major, minor) in [(2, 4), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)] {
        let raw = UInt(major << 16 | minor)
        let (lib, err) = compileMSL(dev, trivialSrc, version: raw)
        versions.append(.o([
            ("version", .s("\(major).\(minor)")),
            ("raw", .u(UInt64(raw))),
            ("accepted", .b(lib != nil)),
            ("error", jOptS(err)),
        ]))
        note("  MSL \(major).\(minor): \(lib != nil ? "accepted" : "refused")")
    }
    body.append(("msl_language_versions", .a(versions)))

    // threadExecutionWidth: CUDA's warp is 32 lanes and the whole translation
    // in §1 rests on this being 32 here too.  Measured, not assumed.
    let (lib, cerr) = compileMSL(dev, trivialSrc)
    if let lib = lib {
        let (pso, perr) = pipeline(dev, lib, "nop")
        if let pso = pso {
            body.append(("thread_execution_width", .i(pso.threadExecutionWidth)))
            body.append(("warp_maps_1to1_to_simdgroup",
                         .b(pso.threadExecutionWidth == 32)))
            body.append(("max_total_threads_per_threadgroup",
                         .i(pso.maxTotalThreadsPerThreadgroup)))
            body.append(("static_threadgroup_memory_length",
                         .i(pso.staticThreadgroupMemoryLength)))
            note("  threadExecutionWidth: \(pso.threadExecutionWidth)")
        } else {
            body.append(("pipeline_error", jOptS(perr)))
        }
    } else {
        body.append(("default_compile_error", jOptS(cerr)))
    }

    body.append(("status", .s("ok")))
    writeResult("p0_device", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
