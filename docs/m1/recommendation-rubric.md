# M1 recommendation rubric

Status: predeclared before native results. The originating manager retains the
adoption decision and M1 acceptance verdict.

## Evidence boundary

Only a validated `cuda4as-m1-result-v1` record from the inventory-bound native
drop can establish a cuda4AS capability. Upstream reports, the WSL NVIDIA
references, package integrity checks, and synthetic analyzer tests establish
provenance or harness behavior only. They contribute no `PASS_GPU` case.

An absent user return keeps the recommendation pending. A preflight-only
`SKIP_ENVIRONMENT` identifies an environment or resource decision and does not
support an adverse candidate verdict. A candidate build failure after a ready
inventory is evidence against this exact pin and configuration, while the three
application cases remain `NOT_RUN` if none reached its first stage.

## Outcome-to-proposal mapping

| Validated native outcome | Bounded executor proposal for manager judgment |
|---|---|
| Direct source, minimal CMake, and multi-TU cases all `PASS_GPU` | Treat the pin as an adoption candidate for the tested vector-add boundary only. Require license resolution, wider semantic corpus work, and later performance/residency evidence before product adoption. |
| Direct source and minimal CMake pass; multi-TU device-link fails | Selective reuse candidate. Record external cross-TU device resolution as a concrete gap and exclude general separable-compilation claims. |
| Direct source passes; minimal CMake integration fails | Selective compiler/runtime study only. Do not describe the pin as a compatible CMake CUDA toolchain. |
| Direct source fails after candidate build succeeds | Reject the pin for the M1 unchanged-source objective unless the returned evidence isolates a harness defect that can be corrected within M1 without changing the application. |
| Candidate configure/build fails with a ready, matching inventory | Reject the exact pinned build route or request a narrowly evidenced M1 harness correction; do not relabel application cases as executed. |
| Any case has correct output without complete inventoried Apple-GPU provenance | `FAIL` for that case. No compatibility credit is assigned to a zero exit code, CPU path, stub, approximate semantics, or an unverified device name. |
| Native preflight reports a missing prerequisite | Keep the affected cases `SKIP_ENVIRONMENT`; request only the concrete user-owned dependency action, if worthwhile, with current package size and target path. |

The PyTorch result is evaluated separately from these three native cases. The
audited upstream build and extension paths require a CUDA-enabled PyTorch build,
CUDA 12.6-or-newer discovery, standard CUDA libraries/targets, and other
platform assumptions that CuMetal's advertised CUDA 12.2 toolkit shim does not
satisfy. A successful vector-add run therefore does not imply PyTorch extension
compatibility.

## Standing blockers on adoption

- The pinned VF64-metal submodule has no license-named file or repository
  license declaration in the complete source audit. Redistribution or adoption
  remains blocked pending provenance/license resolution even if all three
  technical cases pass.
- M1 covers two vector-add programs and one deliberately small external-device
  linkage boundary. It does not establish broad CUDA semantics, performance,
  long-running stability, or framework compatibility.
- The `sm_86` source profile and explicit `ieee64` policy are pinned hypotheses.
  Current fixtures do not exercise FP64 behavior.
- M2 implementation and M3 synchronization work remain outside this rubric and
  outside the current authorization.

## Current proposal

`PENDING_USER_MAC_INVENTORY`. All three concrete M1 cases remain `NOT_RUN`, so
no adopt, selective-reuse, or reject proposal is yet supported.
