// Shared scaffolding for the phase-0a Apple silicon probes.
//
// Compiled into every probe executable:
//     swiftc -O common.swift pN_name.swift -o bin/pN_name
// so `@main` supplies the entry point (no file is named main.swift).  run.sh
// keeps a second route -- concatenate the two and compile in script mode with
// -D PROBE_SCRIPT_MODE -- because nothing in this drop can be compiled on the
// machine it was written on, and one round trip has to survive an entry-point
// disagreement between Swift versions.
//
// Conservative Swift on purpose: no macros, no result builders, no Codable, no
// property wrappers, no async.  Foundation and Metal only; Darwin is imported
// by p1b/p1c alone, where mmap and vm_remap are the thing being measured.
//
// THE MACHINE RULE.  The Mac this runs on is ONE SAMPLE of Apple silicon, not
// the target.  Every JSON file therefore carries a `machine` block, and every
// number in it is meaningless without that block.  A feature this machine does
// not have is a RESULT ("absent on this chip"), never a failure of the probe.

import Foundation
import Metal

// MARK: - JSON, hand-rolled so key order is deterministic and diffable

indirect enum J {
    case s(String)
    case i(Int)
    case u(UInt64)
    case d(Double)
    case b(Bool)
    case null
    case a([J])
    case o([(String, J)])
}

func jsonEscape(_ s: String) -> String {
    var out = "\""
    for ch in s.unicodeScalars {
        switch ch {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if ch.value < 0x20 {
                out += String(format: "\\u%04x", ch.value)
            } else {
                out.unicodeScalars.append(ch)
            }
        }
    }
    return out + "\""
}

func jsonNumber(_ v: Double) -> String {
    // NaN and infinity are not JSON.  A probe that could not measure something
    // says null rather than emitting a token no parser will read back.
    if v.isNaN || v.isInfinite { return "null" }
    if v == v.rounded() && abs(v) < 1e15 { return String(format: "%.1f", v) }
    return String(format: "%.6g", v)
}

func jsonRender(_ v: J, _ indent: Int) -> String {
    let pad = String(repeating: "  ", count: indent)
    let pad1 = String(repeating: "  ", count: indent + 1)
    switch v {
    case .s(let x): return jsonEscape(x)
    case .i(let x): return String(x)
    case .u(let x): return String(x)
    case .d(let x): return jsonNumber(x)
    case .b(let x): return x ? "true" : "false"
    case .null: return "null"
    case .a(let xs):
        if xs.isEmpty { return "[]" }
        let body = xs.map { pad1 + jsonRender($0, indent + 1) }
        return "[\n" + body.joined(separator: ",\n") + "\n" + pad + "]"
    case .o(let kvs):
        if kvs.isEmpty { return "{}" }
        let body = kvs.map { pad1 + jsonEscape($0.0) + ": " + jsonRender($0.1, indent + 1) }
        return "{\n" + body.joined(separator: ",\n") + "\n" + pad + "}"
    }
}

func jOptS(_ s: String?) -> J { return s == nil ? .null : .s(s!) }
func jOptI(_ i: Int?) -> J { return i == nil ? .null : .i(i!) }
func jOptD(_ d: Double?) -> J { return d == nil ? .null : .d(d!) }

// MARK: - shelling out, for the facts Metal does not carry

/// Run a command and return trimmed stdout, or nil if it could not be run.
/// Everything here is a read-only system query; nothing is installed.
func sh(_ path: String, _ args: [String]) -> String? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
    let out = Pipe(), err = Pipe()
    p.standardOutput = out
    p.standardError = err
    do { try p.run() } catch { return nil }
    let data = out.fileHandleForReading.readDataToEndOfFile()
    _ = err.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    guard let s = String(data: data, encoding: .utf8) else { return nil }
    let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
    return t.isEmpty ? nil : t
}

func sysctlStr(_ key: String) -> String? { return sh("/usr/sbin/sysctl", ["-n", key]) }
func sysctlInt(_ key: String) -> Int? {
    guard let s = sysctlStr(key) else { return nil }
    return Int(s)
}

/// GPU core count, from the accelerator's IORegistry node.  Absent on machines
/// whose driver does not publish it -- reported as null, which is a result.
func gpuCoreCount() -> Int? {
    for args in [["-rd1", "-c", "AGXAccelerator"], ["-l", "-w", "0"]] {
        guard let text = sh("/usr/sbin/ioreg", args) else { continue }
        guard let r = text.range(of: "gpu-core-count") else { continue }
        var digits = ""
        var seen = false
        for ch in text[r.upperBound...] {
            if ch.isNumber { digits.append(ch); seen = true }
            else if seen { break }
            else if ch == "\n" { break }
        }
        if let n = Int(digits) { return n }
    }
    return nil
}

// MARK: - getting a device at all

/// Which call produced the device.  Recorded because on 2026-09-05 an M1 Pro
/// returned nil from MTLCreateSystemDefaultDevice() for every probe in the
/// drop, and a drop that cannot tell "no GPU on this chip" from "no GPU in
/// this session" has wasted its round trip.
var deviceRoute = "not attempted"

/// Try the system default, then fall back to enumerating.  They fail
/// independently: the system default needs a window-server session, while
/// enumeration needs IOKit access to the GPU driver class, and a sandbox or a
/// headless login can remove either one.  Whichever works is recorded.
func acquireDevice() -> MTLDevice? {
    if let d = MTLCreateSystemDefaultDevice() {
        deviceRoute = "MTLCreateSystemDefaultDevice()"
        return d
    }
    let all = MTLCopyAllDevices()
    if let d = all.first {
        deviceRoute = "MTLCopyAllDevices().first -- the system default was nil, "
            + "which means this session has no window server but IOKit enumeration works"
        return d
    }
    deviceRoute = "NONE: MTLCreateSystemDefaultDevice() was nil and "
        + "MTLCopyAllDevices() was empty"
    return nil
}

/// What kind of session this is.  The difference between an Aqua login and an
/// SSH one decides whether a nil device is a fact about the chip or a fact
/// about how the drop was launched, and only the second is fixable by rerunning.
func sessionBlock() -> J {
    let e = ProcessInfo.processInfo.environment
    let manager = sh("/bin/launchctl", ["managername"])
    let ssh = e["SSH_CONNECTION"] != nil || e["SSH_TTY"] != nil || e["SSH_CLIENT"] != nil
    return .o([
        ("launchctl_managername", jOptS(manager)),
        ("is_aqua_session", manager == nil ? .null : .b(manager! == "Aqua")),
        ("ssh_env_present", .b(ssh)),
        ("ssh_connection", jOptS(e["SSH_CONNECTION"])),
        ("ssh_tty", jOptS(e["SSH_TTY"])),
        ("term_program", jOptS(e["TERM_PROGRAM"])),
        ("metal_device_count", .i(MTLCopyAllDevices().count)),
        ("reading", .s(ssh || (manager != nil && manager! != "Aqua")
            ? "NOT a window-server (Aqua) session -- Metal device creation is expected to fail here, and a nil device says nothing about the hardware"
            : "a window-server session; a nil device here would be a real finding")),
    ])
}

// MARK: - the machine block

/// Every MTLGPUFamily this build knows to ask about, by RAW VALUE rather than
/// by case name.  Naming `.apple9` or `.metal4` in source would fail to
/// compile against an older SDK and cost the whole round trip; raw values ask
/// the question on any SDK, and a family the runtime does not recognise simply
/// answers false.  Unknown-but-supported values are still reported, so a chip
/// newer than this file shows up rather than disappearing.
let knownFamilies: [(Int, String)] = [
    (1001, "apple1"), (1002, "apple2"), (1003, "apple3"), (1004, "apple4"),
    (1005, "apple5"), (1006, "apple6"), (1007, "apple7"), (1008, "apple8"),
    (1009, "apple9"), (1010, "apple10"), (1011, "apple11"),
    (2001, "mac1"), (2002, "mac2"),
    (3001, "common1"), (3002, "common2"), (3003, "common3"),
    (4001, "macCatalyst1"), (4002, "macCatalyst2"),
    (5001, "metal3"), (5002, "metal4"), (5003, "metal5"),
]

func familyBlock(_ dev: MTLDevice) -> J {
    var rows: [J] = []
    for (raw, name) in knownFamilies {
        guard let fam = MTLGPUFamily(rawValue: raw) else {
            rows.append(.o([("raw", .i(raw)), ("name", .s(name)),
                            ("known_to_sdk", .b(false)), ("supported", .null)]))
            continue
        }
        rows.append(.o([("raw", .i(raw)), ("name", .s(name)),
                        ("known_to_sdk", .b(true)),
                        ("supported", .b(dev.supportsFamily(fam)))]))
    }
    return .a(rows)
}

/// The block every probe repeats.  Repeated on purpose: a results file that
/// travels on its own must still say which machine produced it.
func machineBlock(_ dev: MTLDevice?) -> J {
    var kv: [(String, J)] = []
    kv.append(("collected_utc", .s(ISO8601DateFormatter().string(from: Date()))))
    kv.append(("hostname", .s(ProcessInfo.processInfo.hostName)))
    kv.append(("os_product", jOptS(sh("/usr/bin/sw_vers", ["-productName"]))))
    kv.append(("os_version", jOptS(sh("/usr/bin/sw_vers", ["-productVersion"]))))
    kv.append(("os_build", jOptS(sh("/usr/bin/sw_vers", ["-buildVersion"]))))
    kv.append(("os_version_string", .s(ProcessInfo.processInfo.operatingSystemVersionString)))
    kv.append(("arch", jOptS(sh("/usr/bin/uname", ["-m"]))))
    kv.append(("cpu_brand", jOptS(sysctlStr("machdep.cpu.brand_string"))))
    kv.append(("cpu_logical", jOptI(sysctlInt("hw.logicalcpu"))))
    kv.append(("cpu_perf_cores", jOptI(sysctlInt("hw.perflevel0.logicalcpu"))))
    kv.append(("cpu_eff_cores", jOptI(sysctlInt("hw.perflevel1.logicalcpu"))))
    kv.append(("ram_bytes", .u(ProcessInfo.processInfo.physicalMemory)))
    kv.append(("ram_bytes_sysctl", jOptI(sysctlInt("hw.memsize"))))
    kv.append(("page_size", jOptI(sysctlInt("hw.pagesize"))))
    kv.append(("developer_dir", jOptS(sh("/usr/bin/xcode-select", ["-p"]))))
    kv.append(("swiftc_version", jOptS(sh("/usr/bin/xcrun", ["swiftc", "--version"]))))
    kv.append(("sdk_version", jOptS(sh("/usr/bin/xcrun", ["--show-sdk-version"]))))
    kv.append(("metal_compiler", jOptS(sh("/usr/bin/xcrun", ["-sdk", "macosx", "metal", "--version"]))))
    kv.append(("gpu_core_count", jOptI(gpuCoreCount())))
    kv.append(("device_acquisition_route", .s(deviceRoute)))
    kv.append(("session", sessionBlock()))

    if let d = dev {
        kv.append(("gpu_name", .s(d.name)))
        kv.append(("gpu_registry_id", .u(d.registryID)))
        kv.append(("has_unified_memory", .b(d.hasUnifiedMemory)))
        kv.append(("recommended_max_working_set_size", .u(d.recommendedMaxWorkingSetSize)))
        kv.append(("max_buffer_length", .i(d.maxBufferLength)))
        kv.append(("max_threadgroup_memory_length", .i(d.maxThreadgroupMemoryLength)))
        let mt = d.maxThreadsPerThreadgroup
        kv.append(("max_threads_per_threadgroup",
                   .o([("width", .i(mt.width)), ("height", .i(mt.height)),
                       ("depth", .i(mt.depth))])))
        kv.append(("gpu_families", familyBlock(d)))
    } else {
        kv.append(("gpu_name", .null))
        kv.append(("no_metal_device", .b(true)))
        kv.append(("no_metal_device_reading", .s("NOTHING IN THIS FILE IS A MEASUREMENT. "
            + "Metal handed out no device, so every probe below reports absence of a "
            + "session, not absence of a capability. Check the `session` block: outside "
            + "an Aqua login (over SSH, for instance) this is expected and the drop must "
            + "be rerun from the Mac's own screen.")))
    }
    return .o(kv)
}

// MARK: - result files

/// Where results land.  run.sh sets PROBE_RESULTS; a probe run by hand writes
/// into ./results so a bare `./p0_device` still leaves something behind.
func resultsDir() -> String {
    if let d = ProcessInfo.processInfo.environment["PROBE_RESULTS"], !d.isEmpty {
        return d
    }
    return "./results"
}

func writeResult(_ probe: String, _ body: [(String, J)]) {
    var kv: [(String, J)] = [("probe", .s(probe)), ("schema", .i(1))]
    kv.append(contentsOf: body)
    let text = jsonRender(.o(kv), 0) + "\n"
    let dir = resultsDir()
    try? FileManager.default.createDirectory(atPath: dir,
                                             withIntermediateDirectories: true)
    let path = dir + "/" + probe + ".json"
    do {
        try text.write(toFile: path, atomically: true, encoding: .utf8)
        print("wrote \(path)")
    } catch {
        // Still print the payload: a results directory that cannot be written
        // must not cost the measurement, because the run cannot be repeated
        // from the machine that built this drop.
        FileHandle.standardError.write(
            ("cannot write \(path): \(error)\n").data(using: .utf8)!)
        print(text)
    }
}

// MARK: - Metal helpers

/// Compile MSL at run time.  A compile failure is a RESULT -- the compiler's
/// own text is the finding -- so this returns the message instead of dying.
/// `version` is an MTLLanguageVersion raw value ((major << 16) | minor);
/// nil leaves the default.  Raw values again, so naming a version this SDK
/// has never heard of cannot break the build.
func compileMSL(_ dev: MTLDevice, _ src: String, version: UInt? = nil)
        -> (MTLLibrary?, String?) {
    let opts = MTLCompileOptions()
    if let v = version {
        guard let lv = MTLLanguageVersion(rawValue: v) else {
            return (nil, "MTLLanguageVersion raw value \(v) unknown to this SDK")
        }
        opts.languageVersion = lv
    }
    do {
        return (try dev.makeLibrary(source: src, options: opts), nil)
    } catch {
        return (nil, "\(error)")
    }
}

func pipeline(_ dev: MTLDevice, _ lib: MTLLibrary, _ name: String)
        -> (MTLComputePipelineState?, String?) {
    guard let fn = lib.makeFunction(name: name) else {
        return (nil, "no function named \(name) in library")
    }
    do {
        return (try dev.makeComputePipelineState(function: fn), nil)
    } catch {
        return (nil, "\(error)")
    }
}

/// GPU time for one command buffer, in seconds, from Metal's own timestamps.
/// Wall clock is reported alongside wherever it matters; a number without its
/// method is not a measurement.
func gpuSeconds(_ cb: MTLCommandBuffer) -> Double {
    let t = cb.gpuEndTime - cb.gpuStartTime
    return t > 0 ? t : Double.nan
}

func now() -> Double { return ProcessInfo.processInfo.systemUptime }

func note(_ s: String) {
    print(s)
    fflush(stdout)
}
