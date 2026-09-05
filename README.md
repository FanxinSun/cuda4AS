# cuda4AS — a native CUDA toolchain for Apple silicon

This directory is the code half of `docs/cuda-on-apple-silicon.md`. That
document is the blueprint — envelopes, modules, contracts, gates, forks,
effort. This is where the gates get answered.

**The product is a native toolchain**: `.cu` sources and the build systems
around them go in unchanged, and the program runs on the Apple GPU — the
newest chips first — with the CUDA libraries it calls running there too. Not
a virtual machine, not container pass-through, not CUDA over a boundary. That
scope was fixed by the user on 2026-09-03 and §2.9 of the blueprint records
why, so it is not reopened by accident.

## Layout

```
README.md          this file
probe/             the Swift + MSL probes -- source of truth
                   SELF_REVIEW.md says what each one leans on and how far it
                   was verified, which matters because none of it can be
                   compiled on this box
oracle/            the phase-0 CUDA kernel corpus, its nvcc harness, and the
                   reference outputs produced here on the RTX 5080
drop/as-phase0/    the self-contained directory that goes to an Apple silicon
                   Mac; the user runs ./run.sh and brings back a tarball
RESULTS/           where returned tarballs are unpacked and read
tools/             our own preparation scripts (drop data generation)
```

## The drop/results loop

This box has no Apple GPU and cannot compile a line of Swift or Metal. So the
work moves in round trips, and each one is expensive:

1. Everything is written and reviewed here, in `probe/`, `oracle/` and
   `tools/`.
2. `drop/as-phase0/` is assembled by `tools/make_drop.sh`: self-contained,
   ~23 MB, no path in it points outside itself. `--package` tars it and
   `--publish` attaches it to a GitHub release.
3. It is published as a **release asset** — `gh release download as-phase0` —
   and the user unpacks it on a Mac and runs `./run.sh`. It builds and runs
   every probe, **never stops on a failure**, and tars up one results directory
   holding a JSON file per probe, every compiler message, and every run log.

   The drop travels as a release asset rather than through a clone because a
   `git clone` sends its whole pack in one unresumable shot, and 30 MB of it
   failed repeatedly on the first Mac that tried. Generated binaries — the
   drop's stream data and the oracle's reference outputs — are therefore kept
   out of git and attached to the release instead, so `git clone --depth 1` of
   this repository is a few hundred KB and works on anything.
4. The tarball comes back and is unpacked under
   `RESULTS/as-phase0/<hostname>/`.
5. The results are read, the gate verdict is given, and the next drop is
   written.

Step 3 is why the drop is built the way it is. One round trip has to bring
back *all* the errors, so every build has two routes, every runtime shader
compile is caught and its message recorded, the riskiest probe is split into
three executables, and the single most important question in the gate is asked
twice in two spellings.

## The four standing rules, in two lines each

**1. This PC is the instrument, not the target.** The Mac a drop runs on is
one sample of Apple silicon, exactly as this Linux box is one sample of a
CUDA host; the design has to hold from an M1 with 32 KB of threadgroup memory
and no accelerators through an M5 Ultra and on chips not yet built, so every
number carries its machine and every absent capability is recorded as a
result rather than a failure.

**2. Nothing is installed into anything that exists.** The conda
environments, system Python, `/usr/local/cuda-13.2` and apt are used
read-only; `nvcc` and the RTX 5080 are the correctness oracle and nothing
else. Anything new goes somewhere new.

**3. The network may be metered.** Phase 0a downloaded nothing. Anything
larger than a few tens of MB is sized and put to the user before it is
fetched.

**4. `lmz/` is not ours.** It is read and run, and files are copied *out* of
it with provenance recorded (`drop/as-phase0/lmz/PROVENANCE.md`); it is never
edited, never committed inside, and its gitlink is never staged.

## Where the blueprint's gates stand

| gate | asks | phase 0a status |
|---|---|---|
| **G-C0** probes | the alias, the heap ceiling, co-residency, BF16 on the accelerators | the drop is built and asks all four; unanswered until a Mac runs it |
| **G-C1** compiler | cuda-samples, CUTLASS, llama.cpp build unchanged through a reused front-end | phase 0b; gated on the clang fetch |
| **G-C2** matrix units | a cuBLAS-ABI GEMM at ≥ 80 % of MLX on the same chip | after G-C0's tensor-op answer |
| **G-C6** coded tier | lmz's Metal decoder verifies byte-identical | in the drop as `p8`; first Apple-GPU run of it, ever |

`oracle/ref/` holds the bit-exact reference outputs for the nine-kernel
corpus that G-C1's translator will be checked against. They were produced on
the RTX 5080 in this box, which is a **correctness oracle only** — no timing
was recorded and no performance target is set from that card.
