# cuda4AS owner handover: framework, transport and Metal support boundaries

Date: 2026-09-27, Asia/Shanghai.

Originating manager: **YFCE Supreme Committee**,
thread **01a08bd8-b222-7c21-9509-56020e4a5515**.

Intended recipient: **cuda4AS Planning (Manager+Executor)**,
thread **01a075a7-e2b2-7792-87c6-34f6c0f036aa**.

Packet: **CUDA-BACKEND-BOUNDARY-01**. This is a handover identifier, not a new
project step or execution order.

Project-local delivery:
`/home/rog/business/YFCE/cuda4AS/handover/2026-09-27-backend-boundaries.md`.

Identical originating brief:
`/home/rog/.codex/handover/2026-09-27-cuda4as-backend-boundaries.md`.

## Authorization and scope

The user explicitly instructed YFCE Supreme Committee:

> hand over this conclusion for cuda4AS project by adding docs in their folders

This follows the user's prior explicit designation of cuda4AS Planning as the
recipient. The approved scope is the local documentation handover of the
conclusion below and its documentation-only intake by that owner. No second
approval is needed for this scope.

Use the existing [canonical plan](/home/rog/business/YFCE/cuda4AS/plan/PLAN.md),
especially TRUST-0 and its R-KERNEL, R-LIB, R-APP and R-NVIDIA definitions.
This handover adds explanatory evidence; it creates no competing plan,
implementation assignment, new milestone, or change in execution order.

Implementation, native/GPU runs, downloads, installs, sibling edits, benchmark
campaigns, commits, pushes, merges and releases are outside this packet.
Existing independently authorized work in the owner's conversation keeps its
scope and priority. Finish prior accepted work and its reporting before intake.

## Settled conclusion

PyTorch already provides CUDA and Apple MPS execution backends, but their
existence does not establish equal operator support, performance, numerical
behavior or application compatibility. lmz and lmsluice facilitate artifact
decoding and delivery; neither converts arbitrary CUDA applications to Metal.

The three separate questions are:

- Can a named application use PyTorch's existing MPS backend?
- Can cuda4AS preserve an enrolled CUDA workload's contract on an Apple GPU?
- Can lmz/lmsluice decode and place the required bytes through a validated
  backend on that target?

A positive answer to one question does not establish the others.

## Current capabilities and evidence limits

| Component | Responsibility | NVIDIA/CUDA evidence boundary | Apple Silicon/Metal evidence boundary |
|---|---|---|---|
| PyTorch | Tensor operations and model execution through a selected backend | CUDA backend is a framework facility; availability does not qualify every application or extension. | MPS is an existing backend. Qualify the specific operators, dtypes, framework build, macOS and hardware. |
| lmz | Lossless codecs, archive structure and decoder semantics | Optional CUDA decoder exists. Its supported codec/profile must be checked; archive decode is not automatically GPU accelerated. | Host/native CPU paths are available subject to build and codec prerequisites. The documented Metal scratchpad is written but unrun; no validated native Metal decoder is established here. |
| lmsluice | Sources, routing, delivery, placement and optional consumer adapters | CPU and CUDA tensor placement are implemented in its public torch adapter. | Its core/plain host route is distinct from native Apple GPU placement. The inspected torch adapter explicitly rejects device types other than CPU/CUDA, including MPS. |
| cuda4AS | Compatibility toolchain for enrolled unchanged CUDA inputs, with a native Apple GPU execution goal | CUDA source/runtime/library contracts define the compatibility surface and comparison inputs. | The current plan records M2A-0 failing at device_import and M2A-R1 accepted only as a portable frontend contract. Corrected live compiler/native/GPU validation remains NOT_RUN. |

This matrix records source, documentation and prior-report evidence as observed
on 2026-09-27. No fresh GPU execution or complete application qualification was
performed for this handover. It is not a promise for all models or all devices.

The earlier conversational shorthand "same workflow on both machines via CPU:
yes" must be read as availability of a host route under its dependencies, not
proof that every application has passed on a Mac. In particular, lmz documents
that decoding existing rANS archives requires its native kernel; pure Python
availability alone is insufficient.

PyTorch documents explicit tensor/model placement on the MPS device.
Unsupported MPS operations may use CPU fallback when enabled. MPS does not
support float64/complex128 tensors; that dtype limitation is not cured by the
operator fallback option. Sources:
[PyTorch MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html),
[MPS dtype limits](https://docs.pytorch.org/docs/main/notes/mps.html#double-precision-types),
[MPS environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html).

PyTorch also states that floating-point results are not guaranteed to be
bitwise identical across platforms or releases. Equal input bytes do not
guarantee equal inference/training outputs.
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

## Composition and ownership

lmsluice can orchestrate a plain source or use lmz as an optional decoder
provider, then deliver bytes to a consumer. PyTorch is one possible consumer.
Neither PyTorch nor lmz is mandatory for lmsluice's plain core route.

For an application whose operators and dtypes support MPS, a possible route is
host loading followed by an explicit application-owned transfer to MPS.
That transfer and subsequent compute belong to the application's framework
path. They do not establish direct MPS placement in lmsluice, zero-copy
interoperability, or cuda4AS translation. Account for transfer, initialization,
synchronization and actual compute placement when qualifying such a route.

Changing device selection alone is not evidence that CUDA-specific extensions,
library calls or custom kernels have gained an Apple implementation; each must
be inventoried and qualified.

Decoder semantics, including any future Metal decoder, remain lmz's ownership.
Placement, device buffers, lifetime, synchronization and optional tensor
adapters remain lmsluice's ownership. A possible future lmz Metal decoder and
lmsluice MPS/Metal adapter are separate owner-scoped work; this packet assigns
neither. A decoder kernel translated by cuda4AS would still need the surrounding
host/runtime integration before it could establish a usable sibling backend.

## Mapping to existing cuda4AS decisions

These are references to the current shared plan, not an additional sequence.

| Existing plan reference | Consequence of this clarification |
|---|---|
| TRUST-0; sections 4, 7 and 9 | Preserve independent route claims, frozen result contracts and evidence stages. Support in another project or framework grants no cuda4AS pass. |
| M2A-R2 | Retain the existing separately authorized live closure gate. This document neither dispatches it nor changes its NOT_AUTHORIZED status. |
| M3 and M4; WP1 | lmz/lmsluice source may supply candidate byte/integer/runtime cases after enrollment. A kernel pass does not qualify the full sibling application. |
| M5; R-LIB | Qualify each required CUDA API/library surface independently. Framework backend availability does not implement that surface. |
| M8; WP4; R-APP | PyTorch MPS execution is adapted-application evidence with backend, version, changes and any CPU work disclosed. |
| M9 | Public support claims must state the component, route, target, supported scope and evidence, including untested and unsupported boundaries. |

Use the existing route taxonomy. A framework-MPS result maps to R-APP; a
cuda4AS native kernel result maps to R-KERNEL; library compatibility maps to
R-LIB when its own contract passes. CPU/plain execution is a placement or
reference fact, not a fifth compatibility route or a Native AOT PASS_GPU.

Keep byte integrity and compute validity separate. RC-EXACT-1 governs lossless
decode and transport; RC-FLOAT-1/RC-ML-1 and applicable domain contracts govern
compute. Numerical tolerances, accumulation/precision, seeds, error behavior,
fallback policy and references must be frozen before evaluation. NVIDIA is
one comparison implementation alongside specification/independent references.
Performance comparisons follow correctness and identify their timing boundary.

## Owner intake and acceptance criteria

The originating Committee has reviewed the complete CUDA-TRUST-INTAKE-01 return
and current canonical plan and accepts that bounded planning reconciliation.
The retained native failure and NOT_RUN boundaries still apply. The separate
[Committee review record](/home/rog/.codex/handover/2026-09-27-cuda4as-backend-boundaries.RECORD.md)
records this assessment and its checks. Acceptance does not authorize M2A-R2.

When the existing named conversation is available after its prior task, assess
this conclusion against the current shared plan. Retain existing correct
statements and reference this handover. If a documentation gap needs an
additive clarification, follow the full-line planning review and increase-only
rules, preserve all historical text, and use current stable IDs. A documented
NO_PLAN_CHANGE_REQUIRED verdict is valid when existing text already covers the
conclusion. No implementation follows automatically from intake.

Acceptance requires:

- Explicit acknowledgment of the distinct framework, decoder, transport and
  toolchain responsibilities and their current evidence limits.
- No blanket CUDA/MPS parity claim or claim of a validated lmz/lmsluice Metal
  backend based on the sources above.
- Mapping to existing canonical steps and route definitions, retaining one
  authoritative execution order.
- Truthful accounting of CPU transfer/fallback and separation of exact byte
  delivery from numerical/task correctness.
- A documentation-only result with evidence references, changed paths (if any),
  verification, unresolved questions and all failed/NOT_RUN/inconclusive facts.
  Any contradiction in newer evidence must be identified before changing claims.

Owner's complete final report path:
`/home/rog/.codex/handover/2026-09-27-cuda4as-backend-boundaries.REPORT.md`.

Return the complete outcome and report to YFCE Supreme Committee, thread
`01a08bd8-b222-7c21-9509-56020e4a5515`, after the entire intake is complete.
Final-only reporting; retain a pending-return record if delivery is unavailable.
Decisions outside this bounded scope remain with the owner's user.

## Source anchors and observed revisions

- [cuda4AS canonical plan](/home/rog/business/YFCE/cuda4AS/plan/PLAN.md):
  read completely, 559 lines; SHA-256
  `1b56cd9e3bafb928296c82e935e4768244b5d2e0216ed0b38080e9e1e90c7732`.
- [Prior complete planning return](/home/rog/.codex/handover/2026-09-25-cuda4as-workload-trust.REPORT.md):
  read completely, 431 lines; SHA-256
  `7657b4a3a4adc13916fb075e66799bd3a03404d0d9617c851470d3c53e0748a7`.
- [lmz documented backend limits](/home/rog/business/YFCE/lmz/docs/limitations.md):
  native-kernel dependency near line 18; CUDA scope near line 57;
  unrun Metal port at line 87. File SHA-256
  `f06f38bc72566e46f2a95faac0a61395cd4afe4fc1d5bf4f56207f9a5c4cd19f`.
- [lmsluice tensor adapter](/home/rog/business/YFCE/lmsluice/lmsluice/torchadapter.py):
  load_file at line 98; supported device branches and explicit rejection at
  lines 114-129. File SHA-256
  `920e68148ab9268da78eac4aadacbebd901ea2f56c9458af2952654794d08be9`.
- [lmz/lmsluice ownership boundary](/home/rog/business/YFCE/lmsluice/docs/boundary.md):
  decoder and device-placement ownership; file SHA-256
  `f7c2f3e276dfe3ffcdb131c2783d992d5186e8d2246380910f278cafbe063409`.

Observed HEADs: cuda4AS `de69952dc986173d0b30cca77f55c07b4c14b0ed`;
lmz `2d7517e682a8b0a37fb30fce45d81e4effc210d3`;
lmsluice `4d716948941496c2d55d4f3d5056c192ce3c6705`.
Working-tree file hashes above identify inspected content where dirty docs
exist. These are dated observations, not instructions to restore old state.
External sources were checked on 2026-09-27; they are not evidence of an
installed version or an executed workload.

## Delivery state at preparation

The user's requested project-folder handover is saved locally. A single bounded
status check at 2026-09-27T03:14:21.462901+00:00 identified the intended Planning
thread as offline, with idle unknown. Its previous packet has an explicit final
completion return, but this does not establish present availability.

This new packet has not been queued to an offline executor. Document availability
does not establish recipient receipt, review, acceptance or execution. Keep
session delivery pending until the same named owner is live and ready. No
replacement session was created.
