# M2A-R2 — Live Clang and native vector-add closure

Parent step: M2A-R2 in [the master](../PLAN.md).
State: NEXT_ELIGIBLE_NOT_AUTHORIZED. Planning detail dated 2026-09-27.
This is a proposed bounded attempt, not an instruction to start or run the Mac.

## Objective and entry gate

Establish whether the corrected frontend can carry the unchanged enrolled
vector_add through real Clang JSON, owned IR, native AOT build and Apple-GPU
output validation. An accepted negative report closes this attempt but does
not satisfy the positive prerequisite for M2B.

Entry requires a separately approved handover naming the executor, accepted
R1 source revision, unchanged oracle/source identities, and a current inventory
binding. Reconcile the recorded Mac rather than assuming its environment is
unchanged. Existing reference: M1 Pro/MacBookPro18,3, 16 GiB, macOS14.4 23E214,
Xcode15.2 15C500b, SDK14.2, Homebrew LLVM23.1.0. Compiler version range in the
facade is an intended envelope, not validation of every version.

## Scope and implementation boundary

Reuse tools/m2a and tests/m2a. Keep the unchanged oracle input; do not reconstruct
device source or rewrite the application to fit a parser. Only the vector_add
ABI profile is eligible. This attempt does not generalize the frontend, add
libraries, repair arbitrary compiler nodes or install another toolchain.

The historical tracked package is 22535 bytes with SHA
1184651ff56d76fbea08fce7c68ca857615d1f27fe78934470d218313fbbe1c0.
It predates R1 and is not the proposed retry payload. Preserve it as history;
a newly identified current-source package must be prepared and validated.

## Execution order

| Child ID | Depends on | Action and concrete deliverable | Exit check |
|---|---|---|---|
| M2A-R2-0 | Approved handover | Freeze source/fixture/facade/compiler contract, inventory and attempt/resource IDs. | No unrecorded changed input; stale inventory produces a hold, not a silent rebind. |
| M2A-R2-1 | -0; live portion gated by -2 | [Capture/import contract](subplan_M2A-R2/M2A-R2-1.md): prepare checks before packaging; capture real JSON during -3 and verify compiler semantics. | Real capture, provenance and verified typed IR; synthetic fixtures never substitute. |
| M2A-R2-2 | -0 and prepared -1 contract | [Current-source delivery](subplan_M2A-R2/M2A-R2-2.md): package the corrected code and exact inputs, clean-checkout validate, then publish only under the future approved probe scope. | Exact remote tree contains the no-argument script and all required inputs. |
| M2A-R2-3 | -2; consumes -1 checks | [Bounded native attempt](subplan_M2A-R2/M2A-R2-3.md): execute the complete staged AOT pipeline once on the user's Mac. | Every stage has command, exit, log and artifact identity or explicit NOT_RUN. |
| M2A-R2-4 | -3 return | [Independent evidence validation](subplan_M2A-R2/M2A-R2-4.md): safely analyze unchanged return and compare output. | Classification follows evidence, not the runner's self-report. |
| M2A-R2-5 | -4 | Submit complete report and manager acceptance decision; update canonical status additively. | Positive gate or precise closed-negative attempt; no automatic follow-up dispatch. |

The -1 row spans contract preparation and its single live validation inside -3;
it is not a second Mac run or a dependency cycle. A failed contract stops -3's
dependent stages. The child plans specify this shared stage ownership.

## Evidence and deliverables

Use new attempt-qualified paths under RESULTS/m2a; raw logs/archives remain
ignored as appropriate, while redacted analysis and approved provenance records
can be tracked under the future scope. Preserve the earlier return unchanged.

Required artifacts: inventory and resource record; package member manifest;
original source/golden identities; facade contract; Clang executable/version,
argv and cwd; raw AST/hash; semantic IR and verifier report; linked-device record
with honest single-TU scope; MSL/metallib; AOT manifest; host and native-link
records; runtime device/launch record; output bytes/hash; independent comparison;
stage durations, transfer/launch timing when reached; exit and classification.

## Validation matrix

| Area | Required positive case | Required negative/boundary evidence |
|---|---|---|
| Compiler | Installed intended Clang accepts the pinned facade and unchanged source. | Wrong compiler signature/range, corrupted facade, malformed AST, wrong source/cwd/provenance fail closed. |
| IR and ABI | Exact four-parameter vector_add profile, facts distinct from Metal lowering. | Unknown nodes, conflicting types, generic/shared/unsupported address-space forms rejected. |
| Packaging | Clean tracked export resolves inputs relative to script. | Missing/stale/tampered member and inventory mismatch cannot run. |
| Native execution | Recorded Apple GPU executes the built metallib. | Missing device/tool or resource limit is explicit; no CPU fallback or JIT. |
| Comparison | Exact enrolled shape/input and frozen comparator. | Truncated output, wrong hashes, missing launch identity or mismatch cannot PASS_GPU. |
| Analyzer | Independently verified complete stage chain. | Unsafe archive, duplicate member, altered manifest and false positive runner label rejected. |

Portable regression suite includes every existing M1/M2/M2A test plus new
adversarial cases. Baseline was 69 tests, not a permanent target count. Re-run
syntax, JSON/schema checks and a clean-export no-Clang path; skip remains
NOT_RUN/exit77 and cannot establish native success.

## Resource and stop contract

Retain the bounded envelope: at least 5 GiB free at preflight, at most four
parallel build jobs, at most 20 GiB added disk and 30 minutes for the attempt.
Check effective free space and process completion; preserve partial evidence
on time/resource exhaustion. No installs, downloads, network dependencies,
sudo, OS/SDK changes, external GPU work, runtime compilation or CPU fallback.

Stop at the first dependent-stage failure. A new compiler/importer defect is
reported with raw evidence; fixing it and rerunning requires a new bounded
decision rather than editing around it inside this attempt. No M2B/M3 starts.

## Acceptance and downstream decision

PASS_GPU requires every prerequisite through real launch and correct output,
plus independent validation. A build-only success is not enough. A bounded
failure must still return complete findings and dependent NOT_RUN states.

A positive vector_add result proves one route/profile/target/case, not general
CUDA compatibility, performance parity or real multi-TU linking. The manager
assesses the complete return before deciding M2B eligibility. Dispatch also
requires original-task completion and current executor readiness.

