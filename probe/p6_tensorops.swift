// p6_tensorops -- Metal 4 tensor ops on the Neural Accelerators.
//
// The single most valuable number in this drop.  §2.4 puts cuBLAS, cuBLASLt,
// cuDNN and attention on this path for M5 and later; §3's whole roofline table
// is projected from one A19 microbenchmark through it; and §3's BF16 strategy
// (block-scaled FP16 with a per-tile exponent) exists ONLY because that
// measurement reported no BF16 path.  G-C0's last question is "do the
// accelerators take BF16 through the SDK?".  This probe asks it.
//
// TWO COMPILATION PATHS, reported separately, because they are two different
// products in §2.1:
//   * runtime `makeLibrary(source:)` -- our nvrtc / cuModuleLoadData path.  The
//     author of the reference example notes the MetalPerformancePrimitives
//     header may not be available to the runtime compiler; if so, a JIT refusal
//     here is a REAL constraint on the toolchain's JIT half and not a probe bug.
//   * a .metallib built ahead of time by `xcrun metal` -- our AOT path, the one
//     §2.1 actually chose ("ahead of time, not JIT, for source").  run.sh builds
//     it if the Metal toolchain is installed; if it is not, that too is a result.
//
// Self-skips with a clear message when the OS or SDK predates Metal 4.  An
// absent capability is recorded as absent; nothing here is a failure.
//
// THE MACHINE RULE.  Tensor ops are an M5-and-later story; on an M1–M4 this
// probe is expected to report the path missing, and that is a supported machine,
// not a broken one.  §2.4's "one ABI, two back-ends" is precisely the design
// that makes both outcomes shippable, so record the outcome and move on.
//
// Bounded work: fixed 1024x1024x1024, no loops beyond REPEATS dispatches.

import Foundation
import Metal

let MDIM = 1024, NDIM = 1024, KDIM = 1024
let TILE_M = 64, TILE_N = 32
let SIMDGROUPS = 4
let REPEATS = 5
let SAMPLES = 64

func f16ToFloat(_ h: UInt16) -> Float {
    let sign = UInt32(h & 0x8000) << 16
    let exp = Int((h >> 10) & 0x1F)
    let man = UInt32(h & 0x3FF)
    if exp == 0 {
        if man == 0 { return Float(bitPattern: sign) }
        // subnormal: normalise
        var e = -1, m = man
        repeat { e += 1; m <<= 1 } while (m & 0x400) == 0
        let bits = sign | UInt32(127 - 15 - e) << 23 | ((m & 0x3FF) << 13)
        return Float(bitPattern: bits)
    }
    if exp == 31 {
        return Float(bitPattern: sign | 0x7F800000 | (man << 13))
    }
    return Float(bitPattern: sign | UInt32(exp - 15 + 127) << 23 | (man << 13))
}

func bf16ToFloat(_ b: UInt16) -> Float {
    return Float(bitPattern: UInt32(b) << 16)
}

/// C[m][n] for the 0/1 pattern the shaders fill: A[m][k] = ((m+k)%5==0),
/// B[k][n] = ((k+n)%3==0).  Exact in both FP16 and BF16 at this magnitude.
func expectedC(_ m: Int, _ n: Int) -> Float {
    var acc = 0
    for k in 0..<KDIM where (m + k) % 5 == 0 && (k + n) % 3 == 0 { acc += 1 }
    return Float(acc)
}

func probeMain() {
    var body: [(String, J)] = []
    let dev = MTLCreateSystemDefaultDevice()
    body.append(("machine", machineBlock(dev)))
    body.append(("shape", .o([("m", .i(MDIM)), ("n", .i(NDIM)), ("k", .i(KDIM))])))
    body.append(("tile", .o([("m", .i(TILE_M)), ("n", .i(TILE_N)),
                             ("simdgroups", .i(SIMDGROUPS))])))
    body.append(("flop_per_matmul", .d(2.0 * Double(MDIM) * Double(NDIM) * Double(KDIM))))
    body.append(("reference_spelling", .s("github.com/liuliu/example_matmul_metal4 Sources/matmul/shader.metal, checked 2026-09-03; mpp::tensor_ops::matmul2d over tensor_inline views, static slices")))

    guard let dev = dev, let queue = dev.makeCommandQueue() else {
        body.append(("status", .s("no Metal device"))); writeResult("p6_tensorops", body); return
    }

    // Self-skip signals, gathered before anything is attempted.
    let metal4Family = MTLGPUFamily(rawValue: 5002).map { dev.supportsFamily($0) }
    body.append(("gpu_family_metal4_supported", metal4Family == nil ? .null : .b(metal4Family!)))
    var msl: [(String, J)] = []
    let trivial = "#include <metal_stdlib>\nusing namespace metal;\nkernel void nop() {}"
    for (maj, min) in [(4, 0), (4, 1)] {
        let (l, e) = compileMSL(dev, trivial, version: UInt(maj << 16 | min))
        msl.append(("msl_\(maj)_\(min)", .o([("accepted", .b(l != nil)), ("error", jOptS(e))])))
    }
    body.append(("language_versions", .o(msl)))

    let srcDir = ProcessInfo.processInfo.environment["P6_SRC_DIR"] ?? "./probes"
    let libDir = ProcessInfo.processInfo.environment["P6_LIB_DIR"] ?? "./build"

    func measure(_ tag: String, _ elemBytes: Int, _ decode: @escaping (UInt16) -> Float) -> J {
        var r: [(String, J)] = []
        let srcPath = "\(srcDir)/p6_matmul_\(tag).metal"
        let libPath = "\(libDir)/p6_matmul_\(tag).metallib"
        r.append(("source_path", .s(srcPath)))
        r.append(("metallib_path", .s(libPath)))

        // --- route 1: the runtime compiler (our JIT / nvrtc path) -----------
        var jit: [(String, J)] = []
        var lib: MTLLibrary? = nil
        if let text = try? String(contentsOfFile: srcPath, encoding: .utf8) {
            jit.append(("source_found", .b(true)))
            var attempts: [J] = []
            // Default language version first, then Metal 4.0 and 4.1 explicitly.
            let versions: [UInt?] = [nil, UInt(4 << 16 | 0), UInt(4 << 16 | 1)]
            for v in versions {
                let (l, e) = compileMSL(dev, text, version: v)
                attempts.append(.o([
                    ("language_version", v == nil ? .s("default") : .s("\((v! >> 16)).\(v! & 0xFFFF)")),
                    ("compiled", .b(l != nil)),
                    ("error", jOptS(e)),
                ]))
                if l != nil && lib == nil { lib = l }
            }
            jit.append(("attempts", .a(attempts)))
            jit.append(("any_succeeded", .b(lib != nil)))
        } else {
            jit.append(("source_found", .b(false)))
            jit.append(("note", .s("the .metal source was not next to the probe; set P6_SRC_DIR")))
        }
        r.append(("runtime_compile", .o(jit)))

        // --- route 2: the ahead-of-time metallib (our AOT path) -------------
        var aot: [(String, J)] = []
        var usedRoute = lib != nil ? "runtime_compile" : "none"
        if FileManager.default.fileExists(atPath: libPath) {
            aot.append(("metallib_present", .b(true)))
            do {
                let l = try dev.makeLibrary(URL: URL(fileURLWithPath: libPath))
                aot.append(("loaded", .b(true)))
                // Prefer the AOT library: it is the path §2.1 chose.
                lib = l
                usedRoute = "ahead_of_time_metallib"
            } catch {
                aot.append(("loaded", .b(false)))
                aot.append(("error", .s("\(error)")))
            }
        } else {
            aot.append(("metallib_present", .b(false)))
            aot.append(("note", .s("run.sh could not build it -- either the Metal toolchain is not installed (xcodebuild -downloadComponent MetalToolchain) or the compile failed; see the build log")))
        }
        r.append(("ahead_of_time", .o(aot)))
        r.append(("library_route_used", .s(usedRoute)))

        guard let lib = lib else {
            r.append(("ran", .b(false)))
            r.append(("reading", .s("no library for \(tag) by either route on this machine and OS -- recorded as absent")))
            return .o(r)
        }

        // --- run it -------------------------------------------------------
        let (fillPSO, fillErr) = pipeline(dev, lib, "fill_ab")
        let (zeroPSO, zeroErr) = pipeline(dev, lib, "zero_c")
        let (mmPSO, mmErr) = pipeline(dev, lib, "matmul_tensorops")
        guard let fillPSO = fillPSO, let zeroPSO = zeroPSO, let mmPSO = mmPSO else {
            r.append(("ran", .b(false)))
            r.append(("pipeline_errors", .o([("fill", jOptS(fillErr)),
                                             ("zero", jOptS(zeroErr)),
                                             ("matmul", jOptS(mmErr))])))
            return .o(r)
        }
        let aBuf = dev.makeBuffer(length: MDIM * KDIM * elemBytes, options: .storageModeShared)!
        let bBuf = dev.makeBuffer(length: KDIM * NDIM * elemBytes, options: .storageModeShared)!
        let cBuf = dev.makeBuffer(length: MDIM * NDIM * elemBytes, options: .storageModeShared)!

        func encode(_ pso: MTLComputePipelineState,
                    _ bind: (MTLComputeCommandEncoder) -> Void,
                    _ grid: MTLSize, _ tg: MTLSize, byThreadgroups: Bool) -> (Double, String?) {
            guard let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { return (Double.nan, "no encoder") }
            enc.setComputePipelineState(pso)
            bind(enc)
            if byThreadgroups {
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            } else {
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { return (Double.nan, "\(e)") }
            return (gpuSeconds(cb), nil)
        }

        let (_, fe) = encode(fillPSO, { e in
            e.setBuffer(aBuf, offset: 0, index: 0); e.setBuffer(bBuf, offset: 0, index: 1)
        }, MTLSize(width: max(KDIM, NDIM), height: max(MDIM, KDIM), depth: 1),
           MTLSize(width: 16, height: 16, depth: 1), byThreadgroups: false)
        if let e = fe { r.append(("ran", .b(false))); r.append(("fill_error", .s(e))); return .o(r) }

        let mmGrid = MTLSize(width: NDIM / TILE_N, height: MDIM / TILE_M, depth: 1)
        let mmTG = MTLSize(width: SIMDGROUPS * 32, height: 1, depth: 1)
        var best = Double.infinity
        var runErr: String? = nil
        for _ in 0..<REPEATS {
            let (_, ze) = encode(zeroPSO, { $0.setBuffer(cBuf, offset: 0, index: 0) },
                MTLSize(width: MDIM * NDIM, height: 1, depth: 1),
                MTLSize(width: 256, height: 1, depth: 1), byThreadgroups: false)
            if let e = ze { runErr = "zero_c: \(e)"; break }
            let (t, me) = encode(mmPSO, { e in
                e.setBuffer(aBuf, offset: 0, index: 0)
                e.setBuffer(bBuf, offset: 0, index: 1)
                e.setBuffer(cBuf, offset: 0, index: 2)
            }, mmGrid, mmTG, byThreadgroups: true)
            if let e = me { runErr = "matmul: \(e)"; break }
            if t.isFinite && t < best { best = t }
        }
        if let e = runErr {
            r.append(("ran", .b(false))); r.append(("dispatch_error", .s(e))); return .o(r)
        }

        // Correctness on sampled elements: the full reference would be 1 G MAC.
        let c = cBuf.contents().bindMemory(to: UInt16.self, capacity: MDIM * NDIM)
        var bad = 0
        var firstBad: J = .null
        for s in 0..<SAMPLES {
            let m = (s * 37) % MDIM
            let n = (s * 91 + 5) % NDIM
            let got = decode(c[m * NDIM + n])
            let want = expectedC(m, n)
            if got != want {
                bad += 1
                if case .null = firstBad {
                    firstBad = .o([("m", .i(m)), ("n", .i(n)),
                                   ("got", .d(Double(got))), ("want", .d(Double(want)))])
                }
            }
        }
        let flop = 2.0 * Double(MDIM) * Double(NDIM) * Double(KDIM)
        let tflops = best.isFinite ? flop / best / 1e12 : Double.nan
        r.append(("ran", .b(true)))
        r.append(("samples_checked", .i(SAMPLES)))
        r.append(("sample_mismatches", .i(bad)))
        r.append(("correct", .b(bad == 0)))
        r.append(("first_mismatch", firstBad))
        r.append(("best_gpu_seconds", .d(best)))
        r.append(("tflops", .d(tflops)))
        r.append(("accumulate_precision", .s("whatever Metal Performance Primitives selected for this descriptor; the output tensor has the same element type as the operands, and FP32-accumulate is a phase-0b question")))
        note("\(tag) tensor ops: \(bad == 0 ? "correct" : "\(bad)/\(SAMPLES) wrong"), \(String(format: "%.2f", tflops)) TFLOPS via \(usedRoute)")
        return .o(r)
    }

    body.append(("fp16", measure("fp16", 2, f16ToFloat)))
    body.append(("bf16", measure("bf16", 2, bf16ToFloat)))
    body.append(("bf16_verdict_note", .s("if bf16 neither compiles nor runs here, §3's block-scaled FP16 GEMM is mandatory rather than optional, and that is the finding G-C0 needs")))
    body.append(("status", .s("ok")))
    writeResult("p6_tensorops", body)
}

#if !PROBE_SCRIPT_MODE
@main struct ProbeEntry { static func main() { probeMain() } }
#endif
