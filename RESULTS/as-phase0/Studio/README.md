# Studio — MacBook Pro (16-inch, 2021), Apple M1 Pro

    hw.model        MacBookPro18,3
    chip            Apple M1 Pro, 14 GPU cores, 8 CPU (6P + 2E)
    memory          16 GiB unified
    OS              macOS 14.4 (23E214)
    toolchain       Xcode 15, Swift 5.9.2, Metal compiler 32023.101
    GPU families    apple1–apple7, mac2, common1–3, metal3   (apple8+, metal4: no)

Two runs. **Read the second one.**

| run | outcome |
|---|---|
| `run1-20260905-130047` | measured **nothing**. Every probe got a nil Metal device and the status table said "ok" ten times — a bug in `run.sh`, kept on the record because a drop that reports success while measuring nothing is the failure most worth remembering |
| `run2-20260905-134706` | the real data, after the probes gained an `MTLCopyAllDevices()` fallback |

## Why run 1 failed, which turned out to be a finding

`MTLCreateSystemDefaultDevice()` returns **nil** on this machine — in a normal
Aqua login, in Terminal.app, with no SSH anywhere, and with
`MTLCopyAllDevices()` returning the M1 Pro perfectly well. The session block in
run 2 records all of it: `launchctl managername = Aqua`,
`ssh_env_present = false`, `term_program = Apple_Terminal`,
`metal_device_count = 1`.

That is not an environment mistake, it is a fact about this OS and this kind of
process, and it has a design consequence: **the runtime must never acquire its
device through `MTLCreateSystemDefaultDevice()` alone.** It is also why `p8`
still fails here — lmz's `bench.swift` uses exactly that call, unmodified, with
no fallback.
