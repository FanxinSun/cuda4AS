# M1 result schema v1

Schema ID: `cuda4as-m1-result-v1`. The structural schema is
[`result-schema-v1.schema.json`](result-schema-v1.schema.json), and the
normative cross-field validator is
[`../../tools/m1/validate_results.py`](../../tools/m1/validate_results.py).
The validator uses only the Python standard library.

## Record boundary

One document describes one reproducible machine/candidate run and contains one
or more uniquely identified case results. It records the contract and corpus
versions, candidate revision/options, native machine and toolchain, immutable
source hashes and diff state, individual build stages, execution route and GPU
provenance, numerical assertions, diagnostics, environment gaps, and aggregate
counts.

All six M0 classifications remain unchanged: `PASS_GPU`,
`PASS_CPU_EXPLICIT`, `FAIL`, `UNSUPPORTED`, `SKIP_ENVIRONMENT`, and `NOT_RUN`.
The top-level `record_status` is bookkeeping, not a seventh compatibility
classification:

- `FAIL` means at least one required case is `FAIL`;
- `INCOMPLETE` means no required failure exists but at least one required case
  is `NOT_RUN`;
- `COMPLETE` means every required case has a qualifying terminal
  classification, which may still include visible unsupported or environment
  exclusions.

## Pass invariants

The validator rejects `PASS_GPU` unless all of these are present together:

1. application source and build inputs are recorded unchanged;
2. every declared required stage passed, including device compilation, native
   link, launch, and validation;
3. attempted execution is explicitly `apple_gpu` with exit code zero;
4. backend evidence names an inventoried Metal device and has a completed,
   successful launch with a non-fallback, non-stub, non-approximate source;
5. validation ran, all assertions passed, and an exact case has identical
   expected/actual byte counts and SHA-256 values.

`PASS_CPU_EXPLICIT` instead requires an explicit CPU selection and contains no
GPU provenance. An `UNSUPPORTED` result requires passing, source-located
diagnostic assertions and claims no execution. `SKIP_ENVIRONMENT` names its
missing external conditions and leaves every required case stage unrun.
`NOT_RUN` likewise leaves every required stage unrun and contains no execution,
provenance, or numerical-validation claim. `FAIL` carries an actual failed
stage, nonzero attempted execution, failed validation, or false assertion.

These cross-field rules directly prevent the historical “no device, process
returned zero, status ok” shape from becoming a pass. They also prevent compile
failure, absent GPU provenance, bad exact output, and a required failure hidden
by aggregate counts.

## Native-return normalization

`tools/m1/analyze_native_return.py` is the prescribed bridge from the
user-returned native archive to this schema. It does not extract the archive.
It rejects unsafe members, verifies the exact inner manifest and the checked
package manifest, requires a binding to the separately validated Mac inventory,
recomputes output sizes and SHA-256 values, and parses each
`CUMETAL_PROVENANCE` record. The analyzer derives `completed=true` only for a
successful launch record emitted after completion with a nonnegative GPU
duration. A device name must appear in the bound inventory. Any execution that
does not satisfy all GPU-pass invariants is retained as a visible failure; the
analyzer never infers `UNSUPPORTED` without a later human-reviewed,
source-located diagnostic record.
