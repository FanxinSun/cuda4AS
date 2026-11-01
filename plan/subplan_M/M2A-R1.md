# M2A-R1 — Pinned Clang facade and compiler-semantic correction

Parent step: M2A-R1 in [the shared master](../PLAN.md#12-authoritative-execution-order).
State: CLOSED_PORTABLE_ACCEPTED. Detail recorded 2026-09-27.

## Objective and scope

Record the accepted frontend repair while keeping synthetic schema evidence and real compiler validation separate. This correction did not authorize a Mac retry.

This is the unique retrospective subplan for the named closed step. Its child
IDs identify the historical work breakdown; they do not claim new executions
or reopen the step. No executor assignment follows from this document.

## Inputs and evidence

[Facade manifest](../../tools/m2a/clang_cuda_facade/manifest.json), [AST importer](../../tools/m2a/clang_ast.py), [AOT lowering](../../tools/m2a/native_aot.py), [tests](../../tests/m2a/test_native_aot.py), [design](../../docs/m2/m2a-native-aot-design.md).

## Execution order — historical, non-executable

| Child ID | Completed scope | Evidence or decision |
|---|---|---|
| M2A-R1-1 | Pin CUDA annotation/builtin facade and compiler arguments. | Facade v1.0.0; declared LLVM Clang >=18,<24; intended Homebrew LLVM23.1.0/sm_86, not a claim of validation across that range. |
| M2A-R1-2 | Use original CUDA source and typed Clang JSON as semantic authority. | Arguments, types, declaration IDs, expressions, control flow, loads/stores and builtins derive from compiler output; source signature mismatches fail closed. |
| M2A-R1-3 | Correct real JSON attribute and builtin assumptions. | Direct declaration Attr children; ignored device_builtin is not a required AST node; exact facade provenance and declaration binding retained. |
| M2A-R1-4 | Normalize serializer-elided locations and separate ABI policy from compiler facts. | Stateful file/line recovery; explicit capture cwd; qualType address-space facts; versioned vector_add ABI lowering. |
| M2A-R1-5 | Validate positive/negative portable contracts and package construction. | 69 tests passed at de69952dc986173d0b30cca77f55c07b4c14b0ed; temporary corrected package verified; no live compiler available. |

## Acceptance and classification

Latest complete report: /home/rog/.codex/handover/2026-09-11-cuda4as-m2a-r1-real-json-location-address-space.REPORT.md. Fixture status remains SYNTHETIC_SCHEMA_CONFORMANCE/NOT_CAPTURED. Compiler/native/GPU validation remains NOT_RUN. Earlier local commits 8eb7173, e505213 and 92ad9c0 are preserved corrections leading to de69952.

## Preserved limits and downstream use

Accepted portable correctness is not proof the installed Clang emits the assumed supported slice. M2A-R2 must capture actual JSON and preserve unknown semantic facts; it may not broaden unsupported nodes or address spaces silently. The tracked historical package was not refreshed by this correction.

Current execution eligibility comes only from the master and an authorized
handover. New experiments append new immutable evidence; they never replace
this step's negative, incomplete or NOT_RUN findings.

