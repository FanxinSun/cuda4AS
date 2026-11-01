# M1R — One-patch reuse retry and route retirement

Parent step: M1R in [the shared master](../PLAN.md#12-authoritative-execution-order).
State: CLOSED_NEGATIVE_RETIRED_ROUTE. Detail recorded 2026-09-27.

## Objective and scope

Test only the disclosed direct <array> include correction. Stop on an independent candidate defect; this was not an open-ended upstream repair campaign.

This is the unique retrospective subplan for the named closed step. Its child
IDs identify the historical work breakdown; they do not claim new executions
or reopen the step. No executor assignment follows from this document.

## Inputs and evidence

[Architecture decision](../../docs/m2/architecture-decision.md), [entry evidence contract](../../docs/m2/m2-evidence-contract.md). Final report: /home/rog/.codex/handover/2026-09-10-cuda4as-m2-entry-gate.REPORT.md.

## Execution order — historical, non-executable

| Child ID | Completed scope | Evidence or decision |
|---|---|---|
| M1R-1 | Bind source, patch and package to the validated inventory. | Patch cumetal-lower_to_llvm-direct-array-include-v1; SHA a192690749e55a95e9c826915c0b116fb72e604110194ca852378858c37fd9e2. |
| M1R-2 | Verify clean tree, apply exactly one patch, verify patched tree and unchanged fixtures. | All integrity gates passed; no fixture edits or second patch. |
| M1R-3 | Configure and build on the recorded Mac. | Configure passed; Ninja stopped at 113/117 in runtime/metal_backend/metal_backend.mm. |
| M1R-4 | Classify the second independent failure and preserve dependent NOT_RUN states. | MTLCompileOptions.mathMode and Metal enum availability/namespace errors; no launch, comparison or device-link result. |
| M1R-5 | Retire Route A under the agreed stop condition. | Corrected return SHA 5a00d53ab7e4e5e4075c025825b031769d08ed84eaa03a2a6a1b0bb9967432cc; Route B selected. |

## Acceptance and classification

The patch crossed its intended include boundary but did not establish a buildable candidate. All three fixture cases remained NOT_RUN. The route was retired for this configuration; this is not a universal claim that every upstream revision or Apple target must fail.

## Preserved limits and downstream use

No further upstream patch, SDK migration or build-tool replacement is implied. The completed retry justifies M2E's replacement selection, not a GPU pass. Historical retry instructions are retained as evidence and are not current launch instructions.

Current execution eligibility comes only from the master and an authorized
handover. New experiments append new immutable evidence; they never replace
this step's negative, incomplete or NOT_RUN findings.

