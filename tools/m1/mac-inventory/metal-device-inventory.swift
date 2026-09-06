// Read-only Metal device enumeration for the cuda4AS M1 inventory.
// This helper does not compile a Metal kernel, allocate a buffer, or submit GPU work.

import Foundation
import Metal

func deviceRecord(_ device: MTLDevice) -> [String: Any] {
    var record: [String: Any] = [
        "name": device.name,
        "registry_id": String(device.registryID),
        "is_low_power": device.isLowPower,
        "is_removable": device.isRemovable,
        "is_headless": device.isHeadless,
        "max_buffer_length": String(device.maxBufferLength),
        "recommended_max_working_set_size": String(device.recommendedMaxWorkingSetSize),
        "max_threads_per_threadgroup": [
            device.maxThreadsPerThreadgroup.width,
            device.maxThreadsPerThreadgroup.height,
            device.maxThreadsPerThreadgroup.depth
        ]
    ]

    if #available(macOS 10.15, *) {
        record["has_unified_memory"] = device.hasUnifiedMemory
    }
    return record
}

let devices = MTLCopyAllDevices()
let defaultDevice = MTLCreateSystemDefaultDevice()
let defaultDeviceRegistryID: Any
let defaultDeviceName: Any
if let selected = defaultDevice {
    defaultDeviceRegistryID = String(selected.registryID)
    defaultDeviceName = selected.name
} else {
    defaultDeviceRegistryID = NSNull()
    defaultDeviceName = NSNull()
}
let document: [String: Any] = [
    "schema": "cuda4as-m1-metal-device-inventory-v1",
    "purpose": "device enumeration and default-device selection only; no GPU work submitted",
    "device_count": devices.count,
    "default_device_registry_id": defaultDeviceRegistryID,
    "default_device_name": defaultDeviceName,
    "devices": devices.map(deviceRecord)
]

do {
    let data = try JSONSerialization.data(
        withJSONObject: document,
        options: [.prettyPrinted, .sortedKeys]
    )
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
} catch {
    FileHandle.standardError.write(Data("JSON serialization failed: \(error)\n".utf8))
    exit(2)
}
