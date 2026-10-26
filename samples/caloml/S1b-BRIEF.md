# S1b brief — rework of P9 part 4 and P10, for the cuda4AS executor

Written 2026-10-26 by the Committee session. First read "Committee review of S1" at the bottom of
`samples/caloml/PLAN.md`. It explains what was not accepted and why. Every rule in `S1-BRIEF.md`
still applies: install into `~/system-env`, no commits or pushes, nothing large without asking the
user first, and no Geant4.

**House rule for this round:** the hand-off text must match the JSON. Every number quoted in
PLAN.md must appear in `RESULTS/s1/*.json`. Where the JSON contains `null`, say "not measured".
Don't add any number of your own.

## A. P9 part 4 — redo the fenced flag/value test

- Use **one writer per slot**. Give each writer thread its own slot (`slot = writer index`), and point
  readers at slots owned by writers in other threadgroups.
- Make `value` 64 bits wide, with a self-checking pattern such as `(i << 32) | ~i`. Its `flag` is the
  sequence number `i`.
- Test both orderings and report them separately:
  1. a fence with `mem_flags::mem_device` and `memory_order_seq_cst` (as now);
  2. `atomic_store_explicit(flag, i, memory_order_release)` paired with a `memory_order_acquire`
     load, if the compiler accepts them. If it refuses, record the refusal verbatim.
- A mismatch means: the reader saw flag `f`, then read a value with sequence number `< f`.
- Also measure **real residency**. Use a kernel that atomically counts how many threadgroups are live
  at once, at 128, 256 and 512 threads. Restate the forward-progress sweep as a multiple of that
  measured residency.
- Add one **adversarial case**. Issue tickets so that the threadgroup holding ticket 0 is the
  *last* one dispatched: for example, have every threadgroup spin-wait briefly before taking its
  ticket, with a cap. Report whether look-back still completes.

## B. P10 — redo the measurements and the verdict

1. **Fix the harness before you trust any number.**
   - Compute ulp distance against the host's IEEE double result, using the binary64 ordering
     (bit pattern as a signed-magnitude integer). No op may report `null`.
   - Convert df64 to double as `Double(hi) + Double(lo)`.
   - Test the conversion on its own first: IEEE double → df64 → IEEE double, on 10⁶ values.
2. Use **≥ 10⁶ samples per op**, drawn in three ranges, each reported separately:
   - (a) |x| in [1e-30, 1e30];
   - (b) the full binary64 range;
   - (c) the edge cases: ±0, denormals, ±inf, NaN, and values near overflow/underflow.

   Report the max ulp, the p50/p99 ulp, the exact ratio, and the number of non-finite results that
   should have been finite.
3. **Fix or drop the broken soft64 ops.** The JSON shows max ulp 9e15 on div, 2e23 on sqrt, and 2e19
   on sincos. Soft64's job is bit-exactness, so either reach ≤ 1 ulp (≤ 0.5 ulp on the basic ops)
   or record the op as "not implemented". Do the same for df64's div, sqrt and sincos.
4. **Throughput.**
   - Run several independent chains per thread (for example 8), so the kernel is
     throughput-bound and not latency-bound.
   - Load the operands from a buffer, with non-zero lo parts, so nothing can constant-fold.
   - Check that native FP32 reaches a plausible fraction of the M4's FP32 peak, and report both the
     fraction and the peak you assumed.
   - Count df64 and soft64 as ops of *their own type*: one df64 fma = 1 op.
5. **Conversion cost.** Measure an IEEE-double-in-memory load → df64 → store-back loop, because
   that's what every Celeritas `double` access will cost.
6. **`atomicAdd(double*)`.**
   - Hold the value as IEEE double bits in a `device ulong`, through two 32-bit atomics or a lock.
     Do the add in the chosen representation, and round it back to IEEE.
   - The lock must use acquire and release ordering, or record the compiler's refusal.
   - Test with values that float can't sum exactly: for example 2¹⁶ threads each adding
     `1.0 + k·2⁻⁴⁰`. Compare against the host's exact (sorted, Kahan or long-double) sum, and
     report the error in ulp. Also test on a slot under heavy contention.
7. **Range check for Celeritas.** Without editing Celeritas, find the extreme magnitudes of the
   `real_type` values its device data holds. Two sources:
   - run the S0 double CPU build's unit tests and apps, and scan their output;
   - read `units`/`constants` in the headers.

   List any values outside [1e-30, 1e30]. That decides whether df64's range is acceptable.

## C. The verdict

In PLAN.md, replace the D1 and D2 verdicts. State the representation for the double build, with:

- the accuracy table;
- the corrected throughput;
- the conversion cost;
- the range finding.

Don't quote a number that isn't in the JSON. If neither df64 nor soft64 is good enough as it
stands, say so. A per-op hybrid is also a valid answer: for example df64 arithmetic, an IEEE
memory format, and soft64 for any op whose result must be exact.

Hand back as before: a short "S1b results" section in PLAN.md, everything uncommitted.
