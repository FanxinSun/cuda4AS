# M2E — Native AOT Core v1 architecture decision

Parent step: M2E in [the shared master](../PLAN.md#12-authoritative-execution-order).
State: CLOSED_ARCHITECTURE_SELECTED. Detail recorded 2026-09-27.

## Objective and scope

Select a replacement architecture with explicit compiler, linker, runtime and evidence boundaries. An architecture selection can close this decision without pretending implementation has succeeded.

This is the unique retrospective subplan for the named closed step. Its child
IDs identify the historical work breakdown; they do not claim new executions
or reopen the step. No executor assignment follows from this document.

## Inputs and evidence

[ADR](../../docs/m2/architecture-decision.md), [original proposal](../../docs/m2/m2-handoff-proposal.md), [Native AOT design](../../docs/m2/m2a-native-aot-design.md). Later canonical steps refine these historical documents.

## Execution order — historical, non-executable

| Child ID | Completed scope | Evidence or decision |
|---|---|---|
| M2E-1 | Assess the failed exact candidate and bounded retry. | Two independent native-build boundaries retained, with no Apple-GPU pass. |
| M2E-2 | Fix the user-visible contract. | Unchanged enrolled CUDA inputs; source-based AOT, no SASS or drop-in binary CUDA promise. |
| M2E-3 | Define compiler and device-link boundaries. | Pinned Clang/header surface → owned typed SSA IR → genuine whole-program device resolution → MSL/metallib. |
| M2E-4 | Define host/runtime boundaries. | Objective-C++ Metal runtime; manifest-bound kernel ABI; no silent CPU fallback or runtime compilation. |
| M2E-5 | Choose the smallest native validation slice and defer unsupported breadth. | FP32/int32 vector_add first; real multi-TU next; FP64/VF64, broad libraries/frameworks separately gated. |

## Acceptance and classification

The decision selected Native AOT Core v1, not a mature CUDA implementation. M2A-R1 now supplies the corrected compiler-semantic contract. Proposed directories and interface names in the ADR are design intentions unless supported by checked-in implementation and tests.

## Preserved limits and downstream use

Raw-source pattern matching must never become the semantic authority. Application-specific ABI lowering must remain distinct from compiler facts. The old reuse-first wording is retired by this choice; only an explicit new decision can reopen it.

Current execution eligibility comes only from the master and an authorized
handover. New experiments append new immutable evidence; they never replace
this step's negative, incomplete or NOT_RUN findings.

