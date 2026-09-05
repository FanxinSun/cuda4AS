# cuda4AS — phase 0a probe drop for Apple silicon

Get this onto an Apple silicon Mac, run one command, send back one tarball.

```sh
gh release download as-phase0 --repo FanxinSun/cuda4AS --pattern 'as-phase0.tgz*'
shasum -a 256 -c as-phase0.tgz.sha256
tar xzf as-phase0.tgz
cd as-phase0
./run.sh
```

**Download the release rather than cloning the repository.** The drop carries
22.6 MB of generated lmz stream data; a `git clone` transfers its whole pack in
one shot with no way to resume, and that failed repeatedly on the first Mac
that tried it. A release asset is CDN-served and resumes. (If you want the
repository too — for the results loop below — `git clone --depth 1` is a few
hundred KB, because the data is not in git.)

It takes a few minutes, installs nothing, writes nothing outside this
directory, and never stops on a failure. When it finishes it prints a status
table and names a file:

```
Send back:  .../as-phase0/results-<hostname>-<date>.tgz
```

That tarball is the deliverable. **Send it back even if half the table says
FAILED** — the failures are the point as much as the numbers are. Nothing in
this drop has ever been compiled by a Swift compiler or seen a Metal driver:
it was written on a Linux box with an NVIDIA card and no Apple hardware at
all, so this run is the first contact, and one round trip has to bring back
every error at once.

---

## What you need

**Only the Xcode Command Line Tools.** If `swiftc` is missing, run:

```sh
xcode-select --install
```

macOS 14 or newer. Every probe compiles its Metal shaders at run time through
`MTLDevice.makeLibrary(source:)`, so the separately-downloaded Metal toolchain
is **not required**.

One probe — `p6_tensorops`, the Metal 4 tensor-op probe — has a second,
optional route that needs that toolchain (`xcodebuild -downloadComponent
MetalToolchain`, roughly a 2 GB download on a metered link, so **only if you
want it**). Without it p6 still runs and reports whether the runtime compiler
can reach the tensor-op headers, which is itself one of the things being
asked. Do not download it on a cellular connection.

If this Mac predates Metal 4, p6 self-skips with a message saying so. That is
a supported machine, not a broken one.

---

## What it measures, and why each one matters

Everything here is a question the design in `docs/cuda-on-apple-silicon.md`
currently answers from the public record rather than from hardware.

| probe | the question | what turns on it |
|---|---|---|
| `p0_device` | what this GPU says about itself: families, threadgroup memory, buffer and working-set limits, unified memory, SIMD width, which MSL language versions the compiler accepts | the machine profile of §2.8, and the device properties §2.3 has to report to CUDA programs |
| `p1_heap` | how large one `MTLBuffer` can be; is `gpuAddress` readable; does `makeBuffer(bytesNoCopy:)` take pages this process owns; do the CPU and GPU addresses of those pages coincide | the one-buffer device heap of §2.3 and where the tiers have to start |
| `p1b_vmremap`, `p1c_machvmremap` | can the heap's pages be mapped a second time **at the GPU virtual address**, so one number dereferences on both sides | `cudaMallocManaged`. If yes it is free; if no, every pointer load in a managed kernel is rebased by the compiler. Asked twice in two spellings on purpose |
| `p2_coresident` | how many threadgroups actually run at once, swept over threadgroup size and threadgroup memory | `grid.sync()`. cuda-metal caps cooperative grids at one block per core because it will not measure this |
| `p3_lockstep` | is `simdgroup_barrier` a memory fence or an execution barrier; do lanes make independent forward progress | whether lmz's prefetch decoder is safe, and whether CUDA code written for Volta's independent thread scheduling can be translated at all or must be rejected |
| `p4_bandwidth` | GB/s, read-only and read+write, with the traffic definition stated | every decode and residency claim in §3 is bandwidth ÷ bytes |
| `p5_simd` | FP32 / FP16 / BF16 FMA rates and `simdgroup_matrix<half,8,8>` rate | the M1–M4 half of §2.4's "one ABI, two back-ends" — on a chip with no Neural Accelerators this **is** the matrix unit |
| `p6_tensorops` | do Metal 4 tensor ops compile and run here, in FP16 and in **BF16**, and at what rate | the most valuable number in the drop. §3's whole strategy exists because one A19 measurement reported no BF16 path; PyTorch trains in BF16 |
| `p7_fp64_atomics` | does this compiler reject `double`; are 64-bit and float atomics native | §1's "software" and "partial" rows, replaced by the compiler's own words |
| `p8_lmz_decoder` | lmz's rANS decoder, unmodified, on real Apple hardware for the first time | gate G-C6: the coded weight tier. It has never run on an Apple GPU |

A capability this Mac does not have is recorded as **absent**, never as a
failure. The Mac you run this on is one sample of Apple silicon; the design
has to hold from an M1 with 32 KB of threadgroup memory and no accelerators
through an M5 Ultra, and on chips that do not exist yet. Every result file
carries a `machine` block for exactly that reason.

---

## What comes back

```
results/
  status.txt                 the table run.sh printed
  env.txt                    OS, CPU, RAM, Xcode, SDK, Metal, GPU core count
  p0_device.json … p7_fp64_atomics.json    one per probe, with a machine block
  build-<probe>.log          the compiler's output, success or failure
  run-<probe>.log            everything the probe printed
```

Roughly 22 MB goes to the Mac; the tarball coming back is a few tens of KB.

To send the results home, the neatest route is to commit them into the
cuda4AS repository — `git clone --depth 1` it, and its `RESULTS/README.md` has
the four commands. Otherwise just send the tarball itself.

## What is in here

```
run.sh                     builds and runs everything, captures, tars
probes/                    the Swift probes and p6's two .metal shaders
lmz/                       lmz's decoder harness + shader, UNMODIFIED,
                           with PROVENANCE.md naming the commit
data/                      22.6 MB of synthetic lmz streams (512 x 32 KiB
                           planes, ~2.8 bits/symbol, coded by lmz's own
                           encoder and verified through lmz's own decoder
                           before shipping).  In the release tarball but not
                           in git -- data/README.md says why.  Only p8 needs
                           it; if it is missing, p8 says so and the run
                           carries on.
```

Nothing in this directory refers to a path outside it.
