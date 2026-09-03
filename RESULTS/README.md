# RESULTS — where the Macs' answers land

Empty until a drop comes back.

## Putting a tarball here

`run.sh` on the Mac leaves a file called
`results-<hostname>-<UTC date>.tgz` next to itself. Unpack it under a
directory named for the machine that produced it:

```sh
mkdir -p cuda4AS/RESULTS/as-phase0/<hostname>
tar xzf results-<hostname>-<date>.tgz -C cuda4AS/RESULTS/as-phase0/<hostname> --strip-components=1
```

so the tree reads:

```
RESULTS/as-phase0/<hostname>/
  status.txt                 the table run.sh printed
  env.txt                    OS, CPU, RAM, Xcode, SDK, Metal, GPU core count
  p0_device.json … p7_fp64_atomics.json
  build-<probe>.log          the compiler's output, success or failure
  run-<probe>.log            everything the probe printed
```

Keep the original tarball beside the unpacked directory. Never edit a
returned file: if something in it needs correcting, correct the probe and run
another drop.

## One machine is one sample

Name the directory after the machine, not after the drop, because **the point
of collecting several is that they disagree**. An M1 has no Neural
Accelerators and 32 KB of threadgroup memory; an M5 Ultra has 80 cores and
1.2 TB/s; a phone-class SoC sits an order of magnitude below both. Every
result file carries its own `machine` block for this reason, and a number
quoted anywhere else in this project has to carry the chip, the OS and the
conditions with it.

A conclusion drawn from one directory here is a conclusion about one Mac. If
it is to become a design decision, say which machines it was checked on and
where in the range it would flip — that is `CLAUDE.md` rule 1, and it is the
rule this project has broken most often.

## Reading a drop

1. `status.txt` first: which probes built, which ran, which failed.
2. Every `build-*.log` with a failure in it — those are the compiler's own
   words and they are what the next drop is written from.
3. The JSON files, against the questions in `../drop/as-phase0/README.md`.
4. Then the gate verdict, which is recorded in
   `~/.claude/handover/2026-09-03-cuda-as-phase0.REPORT.md` and in
   `docs/cuda-on-apple-silicon.md` §7.

G-C0 asks four things and each has a fork behind it:

| if | then |
|---|---|
| the heap cannot be aliased at one address (`p1b`/`p1c`) | managed memory is compiler-rebased only, at a cost that must then be measured |
| the working-set limit or `maxBufferLength` is well below 75 % of RAM (`p1_heap`) | the tiers of §2.7 start earlier than planned |
| co-residency cannot be probed (`p2`) | `grid.sync()` falls back to one threadgroup per core, which is where cuda-metal already is |
| the accelerators do not take BF16 (`p6`) | §3's block-scaled FP16 GEMM stops being an optimisation and becomes mandatory |
