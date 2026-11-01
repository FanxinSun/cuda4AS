# INT-1 — Planning reconciliation after the integration (CUDA-INTEGRATE-01)

Parent step: INT-1 in [the shared master](../PLAN.md#12-authoritative-execution-order).
State: CLOSED_PLANNING. Recorded 2026-11-01 by the Linux session "cuda4AS Committee".

## Objective and scope

Absorb the M4 line into the one plan: classify the merged toolkit against the
accepted architecture, decide the evidence state of the M4 results, retire the
competing "next step" statements, and define the next steps with the M1
MacBook Pro as the development machine. Documentation and decisions only; no
run, download, install, merge or release.

This is the unique retrospective subplan for the named closed step. No
executor assignment for implementation follows from this document; the
mechanical closure of the revision is a separate Linux handover (INT-1-7).

## Inputs and evidence

Every planning document was read in full before editing (master §15):
`plan/PLAN.md` and all 25 subplans; `samples/caloml/PLAN.md` and the three S1
briefs; `HANDOVER.md`, `HANDOFF.md`, `toolkit/DESIGN.md`, `README.md`,
`probe/SELF_REVIEW.md`, `cuda-projects/README.md`,
`RESULTS/cuda-projects/LEDGER.md` and `RETEST.md`; the three `handover/`
packets. Code read for the findings in master §3: `toolkit/src/runtime/cudart.mm`,
`toolkit/src/driver/nvcc_main.cpp`, `toolkit/src/translator/cuda4as_translate.py`,
`toolkit/include/*.h`, `toolkit/CMakeLists.txt`, `oracle/run_oracle.py`,
`RESULTS/g-c1/*.json` against `oracle/ref/*.json`, `docs/m1/dependencies-mac.json`,
`RESULTS/m1/*.analysis.json`, `RESULTS/as-phase0/Studio/README.md` and the
M1 Pro `p0_device.json`. The review ledger with paths and hashes:
`docs/ledgers/2026-11-01-cuda4as-planning-review-ledger.md`.

## Execution order — historical, non-executable

| Child ID | Completed scope | Evidence or decision |
|---|---|---|
| INT-1-1 | Read every line of every planning document and the toolkit sources named above. | Findings F1–F5 in master §3, each with its file and line range. |
| INT-1-2 | Decide the evidence state of the M4 results. | `RESULTS/g-c1`: REPORTED_NOT_REPRODUCED, `mma_inline_ptx` INCONSISTENT; Celeritas S2: TRUST-BUILT reported; Celeritas S3: dispatch path unknown; S1 probes: accepted for the M4 under the M4 Committee's reviews (master §3 table). |
| INT-1-3 | Classify the toolkit components: adopt, seed, contain, retire. | master §3 table. |
| INT-1-4 | Retire the competing orders and labels; register the M4 documents. | master §2 (added bullets and retired statements), §12 (2026-09-27 table retired in place; current table), §13 note, §17 note. |
| INT-1-5 | Define the INT and TK steps and the caloml route M6-7; name the M1 MacBook Pro. | master §10, §12 rows 9–14 and 19; `subplan_INT/`, `subplan_TK/`, `subplan_M/subplan_M6/M6-7.md`. |
| INT-1-6 | Write the TK-0 handover (`HANDOFF.md` Part A) and the register entry (`HANDOVER.md` §11). | In the repository. |
| INT-1-7 | Hand the mechanical closure to the Linux executor: rename `samples/caloml/PLAN.md` to `CALOML-RECORD.md` with notes, README pointers, the review ledger, link check, commits, push. | `~/.claude/handover/2026-11-01-cuda4as-plan-reconciliation.md` and its report. |

## Acceptance and classification

Planning acceptance only. Nothing here asserts an Apple-GPU result, a
Celeritas result or a toolkit capability; it states what the records show and
what remains to be measured, and where.

## Preserved limits and downstream use

Increase-only: every original sentence of every plan file is retained;
retirements are dated notes beside the text they retire. Future revisions keep
the same discipline and add a new ledger file under `docs/ledgers/`.

Current execution eligibility comes only from the master and an authorized
handover. TK-0's handover is `HANDOFF.md` Part A; it binds the session the
user opens on the M1 MacBook Pro once the user tells it to start.
