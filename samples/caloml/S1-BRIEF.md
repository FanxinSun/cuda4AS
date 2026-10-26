# S1 brief — for the cuda4AS executor session

Written 2026-10-26 by the Committee session. The plan of record is `samples/caloml/PLAN.md`. Read it
first, especially "What Celeritas asks of cuda4AS", "Against the G-C0 findings" and "Decisions". The
user accepted D1–D4 as recommended on 2026-10-26.

S1 has three tasks. None of them needs Geant4, and none needs a download of more than a few MB.

## Rules (these override anything that conflicts with them)

- **Never commit or push.** The user does both. Do not stage files either.
- **Every download, build and install goes into `~/system-env`**, using its layout:
  - `src/<pkg>-<ver>`, plus the tarball, with its sha256 appended to `logs/source-sha256.txt`
  - `build/<pkg>-<ver>-<variant>`
  - `install/<pkg>-<ver>-<variant>`
  - `logs/<pkg>-<ver>-<variant>/`
  - scripts and env in `~/system-env/cuda4AS/`

  Don't use Homebrew for anything new. Before writing to a path, check that it is free.
- **Install only what the task needs.** Geant4, ROOT, VecGeom and HepMC3 wait for S4/S7: don't size
  or fetch them.
- **Anything over a few tens of MB** must be sized and put to the user before you fetch it (README rule 3).
- **Do not edit Celeritas, caloml (`~/caloml`), `lmz/` or `cuda-projects/`.** They are read-only input.
- **A refused or absent construct is a result, not a failure.** Record the compiler's or runtime's
  exact text in the probe JSON (the house style is in `probe/SELF_REVIEW.md` and `probe/p7_fp64_atomics.swift`).
  Every number carries its machine: Apple M4, macOS 15.4.1, Xcode 16.3, Metal 3, apple9.

## Task 1 — the float CPU oracle (D1)

The GPU build comes up first as `-DCELERITAS_REAL_TYPE=float`, so it needs a float CPU oracle to
compare against on the same machine.

- Add a `float` variant to `~/system-env/cuda4AS/build-celeritas-cpu.sh`. Do it with a variable,
  for example `REAL=float` → `NAME=celeritas-0.6.3-cpu-float`, and leave the double default unchanged.
- Build it into `build/` and `install/celeritas-0.6.3-cpu-float`. The sources are already in
  `~/system-env/src/`, so nothing needs downloading.
- Run ctest and record pass/fail/disabled. The double build's result is 239/0/36. Any float-only
  failure is a finding: name the test and quote its tolerance message.
- Optionally add `CELERITAS_CPU_FLOAT` to `~/system-env/cuda4AS/env.sh`.
- Add an "S1 — float CPU oracle" results table to PLAN.md, in the same style as the S0 one.

## Task 2 — P9: decoupled look-back on Apple GPUs (decides D2's fallback)

CUB's single-pass scan, select and partition have each threadgroup spin on its predecessors' tile
status. The status word packs status and value into one word twice the value's size, and that word
must be stored and loaded as a single copy. Celeritas runs `exclusive_scan`, `copy_if` and
`remove_if` every step. P9 asks two questions:

1. **Forward progress.** With a grid larger than what the device can hold resident at once, does
   tile *i* always eventually see tile *i−1* publish? Or can a waiting threadgroup starve one that
   hasn't been scheduled yet?
   - Measure with a bounded spin (a cap plus a timeout flag; never an unbounded loop).
   - Vary the grid from 1× to at least 64× resident capacity and the threadgroup size over 128/256/512.
   - Report: hangs, cap hits, and the order in which tiles start (from an atomic ticket).
2. **64-bit single-copy store/load.** `atomic_ulong` has no operations on this machine (G-C0). Test
   whether a plain 64-bit `device ulong` store paired with a 64-bit load is ever observed torn
   (high half from one write, low half from another), under concurrent writers and readers. Also
   test the 32-bit fallback encodings: a status+value packed into 32 bits, and a flag-then-value
   pair separated by `atomic_thread_fence`.

Both questions must use CUB's own pattern: tickets taken from an atomic counter, not
`threadgroup_position_in_grid`. That's how CUB stays safe without assuming dispatch order.

Write `probe/p9_lookback.swift` in the style of the existing probes: one library per case, refusals
captured, bounded work, and JSON output. Results go to `RESULTS/as-phase0/<hostname>/` (or a new
`RESULTS/s1/` if that fits better; say which). In PLAN.md under D2, state the verdict: **look-back
safe as-is**, **safe with encoding X**, or **unsafe → toolkit-side multi-pass fallback**.

## Task 3 — P10: the FP64 software path (D1 acceptance)

Choose the representation from measurements, not from the blueprint's guess:

- Candidates: double-float (float-float, about 48-bit mantissa) and IEEE soft-double (bit-exact,
  slower). Implement each one in MSL for `+ − × ÷`, `fma`, `sqrt` and comparisons, plus the three
  libm functions Celeritas leans on most: `exp`, `log` and `sincos`.
- **Accuracy:** test against the host's IEEE double on at least 10⁶ inputs. Include the edge cases:
  denormals, inf, NaN, signed zero, and the overflow/underflow edges. Report max ulp and the
  distribution.
- **Throughput:** ops/s on the M4 GPU for each candidate, against native float.
- **`atomicAdd(double*)`** (Celeritas `SimpleCalo`): `atomic_ulong` is absent, so test a
  lock-backed version that serializes the lanes of one SIMD-group (G-C0: no independent thread
  scheduling, so a naive spin lock deadlocks). Prove it with a 2¹⁶-thread sum where the exact
  answer is known.
- Write `probe/p10_fp64.swift`. In PLAN.md under D1, state which representation the double build
  will use and why. Remember that physics equivalence is statistical: soft-double is required
  only where results must match bit for bit.

## Hand back

When S1 is done:

- Write a short report at the bottom of PLAN.md ("S1 results"). Include each verdict, the numbers,
  and anything that changes S2's plan.
- Leave every change uncommitted.
- Tell the user it's ready for the Committee. The next stage, S2 (the toolkit surface), belongs to
  the projects converter session.
