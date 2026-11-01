# Project-local owner handover — cuda4AS Planning (Manager+Executor)

Saved in this project on 2026-09-25 (Asia/Shanghai) by **YFCE Supreme Committee**,
thread **01a08bd8-b222-7c21-9509-56020e4a5515**, for **cuda4AS Planning (Manager+Executor)**.

The user explicitly instructed: **"no giving me. you save handover docs in their
folders"**. This file is that local delivery. Its retained originating brief is
`/home/rog/.codex/handover/2026-09-25-cuda4as-workload-trust.md`; the complete brief follows below. This is the same approved packet,
not a new master plan or an additional execution order. Read current shared
project plans before acting; recorded old commits and availability are
historical context, not commands to restore that state.

The current intake rules apply to all text below: preserve every original
planning term/step/decision, annotate superseded content **retired**, add current
replacements, and keep one authoritative current order per canonical document.
Every planning update requires a full line review of all project planning
documents, with reviewed paths/hashes in a separate agent ledger. Follow the
current single-master/shared-plan rules and existing scope/prerequisite gates.
Codex-to-Codex receipt does not trigger a new non-Codex takeover snapshot.

Saving this file does not establish a session's acknowledgment, acceptance or
execution. The named owner must finish any prior accepted task and reporting
before starting this packet. Already-granted scope approval need not be repeated.
The full report path and return obligations in the retained brief still apply.

## Complete originating brief

# cuda4AS workload portability and trust — approved planning handover

Date: 2026-09-25, Asia/Shanghai.

Originating manager: **YFCE Supreme Committee**, thread
**01a08bd8-b222-7c21-9509-56020e4a5515**.

Intended recipient and decision owner: **cuda4AS Planning (Manager+Executor)**,
thread **01a075a7-e2b2-7792-87c6-34f6c0f036aa**.

Packet ID: **CUDA-TRUST-INTAKE-01**. This identifies this bounded handover;
project execution remains governed by the shared canonical plan and its IDs.

Status when prepared: **approved; delivery pending**. The named conversation
was offline at 2026-09-24T21:44:15.850392+00:00. No prior-task completion or
current idle state is inferred from that fact. Hold dispatch until the existing
conversation is available and its prior accepted task/report obligations are
complete. Do not start a replacement session.

Accessible WSL file:
`/home/rog/.codex/handover/2026-09-25-cuda4as-workload-trust.md`.

## Authorization and requested outcome

The user asked whether general GPU workloads found throughout `/home/rog` can
run on Metal with results as trustworthy as those on NVIDIA, requested a
home-wide GPU-usage survey, then explicitly instructed this Committee:

> Sounds good for cuda4AS, update the new concepts to cuda4AS Planning session

This approves this named recipient's **planning reconciliation and strategy
update**. Incorporate the accepted concepts below into cuda4AS's one shared
canonical plan, map them to current milestones, and return the complete outcome.
Do not request a second approval for this bounded documentation work.

Implementation, new GPU/native runs, training, downloads, installations,
framework or library ports, sibling changes, merges, tags, releases and public
publication are not assigned by this packet. Existing authorizations in the
recipient's own conversation continue to govern its original work; this packet
does not cancel or silently broaden them.

## Current local evidence and reconciliation inputs

Repository: `/home/rog/business/YFCE/cuda4AS`.

Read-only observation on 2026-09-25:

- Branch: `codex/m2a-r1-clang-semantic-import`.
- HEAD: `de69952dc986173d0b30cca77f55c07b4c14b0ed`.
- Existing dirty work: modified `.gitignore`; untracked `RESULTS/m2a/`.
- Legacy master: `planning-codex/cuda4AS-project-plan-codex.md`.
- `plan/PLAN.md` was absent at this observation; recheck before editing.
- `docs/m2/architecture-decision.md` records selection of **Native AOT Core v1**
  after the bounded CuMetal retry failed.
- `docs/m2/m2a-native-aot-design.md` records the bounded vector-add slice and
  typed Clang importer, native runtime, evidence contract and explicit stops.
- The legacy master still describes M1R as pending. That stale text must not
  reset the project to M1R or erase the later M2/M2A decisions. Reconcile from
  the owner's current accepted reports and current documents.

The Committee has not assessed all current M2A returns. This handover does not
declare M2A accepted, failed, complete or ready for a new native run. Historical
M1 Pro results and the current small supported profile do not establish general
Metal compatibility.

The earlier optional PaP packet remains an input:
`/home/rog/.codex/handover/2026-09-23-pap-ai-cuda4as.md`, step `CUDA-PAP-0`.
Its PaP boundary remains valid: cuda4AS is optional, no PaP engine is selected,
and initial CPU/replay/commodity-engine PaP work must not depend on cuda4AS.
This intake consolidates the pending planning reconciliation with the new
strategy decision; do not perform two competing master-plan migrations. Preserve
the old packet as provenance and report both mappings in the new final report.

## Accepted strategy decisions to incorporate

### Product boundary and routes

Expand the success criterion to **qualified GPU workload coverage**: a workload
runs on a named Apple target and meets a declared semantic and result contract.
Keep the existing unchanged CUDA-source/build-input native-toolchain objective.
Growing the CUDA corpus remains useful for discovery, stress testing and
regression; raw program count is not the main support metric.

Use four explicit porting routes in the planning and support matrix:

| Route | Responsibility and claim boundary |
| --- | --- |
| CUDA kernel translation | Preserve supported source/build semantics through the native toolchain and Metal runtime. |
| CUDA library compatibility | Specify contracts for called APIs such as cuBLAS, cuDNN, CUB and Thrust; substitute implementations only under a validated API/numerical/resource contract. |
| Framework application port | Assess named PyTorch, TensorFlow, JAX, ONNX Runtime or other application paths using available Apple backends. Record backend/version/operator limits and required application changes. |
| NVIDIA-specific replacement or unsupported | OptiX, architecture-specific PTX, matrix instructions and unsupported synchronization need separate designs or explicit rejection. |

An application adapted to MPS/MLX/Core ML or another framework backend is an
**application-port result**, not evidence that its unchanged CUDA path passes
cuda4AS. Keep those denominators and support labels separate. Native Metal
execution, CPU execution and remote execution must be identifiable; the
selected Native AOT Core's no-CPU-fallback boundary remains in force.

Do not assume every framework or library has a usable Apple replacement.
Availability, supported operators, license, target and numerical behavior are
facts to qualify for each future named candidate. A future redesign route does
not expand the currently supported compiler profile.

### Workload and semantic coverage

Index coverage by workload family, semantic feature and target profile:

- Workloads: exact transforms/codecs; dense linear algebra; reductions/scans;
  transfer/overlap; irregular arrays/graphs; geometry/transport; stochastic
  simulation; learning/inference; rendering and complete applications.
- Semantics: address spaces/pointer identity; lifetimes; launch/indexing;
  shared memory; barriers/subgroups; atomics/order/progress; streams/events;
  graph/device-link/runtime dependencies; numeric formats/rounding/exceptional
  values; error propagation, cancellation and resource ceilings.
- Targets: Apple GPU family/device, macOS/SDK/Metal/compiler/runtime versions,
  numerical profile, and the independently recorded CPU/NVIDIA reference setup.

Track duplicate checkouts, vendored/reference code, installed environments and
generated artifacts separately from distinct project workloads. A semantic
feature must not be counted as covered merely because its name appears in code.

### Result contracts and independent references

| Workload | Required future contract |
| --- | --- |
| Integer/byte transforms, indexing and lossless decoding | Exact outputs against an independently defined CPU/specification reference and NVIDIA where available. Check bounds and invalid input behavior. |
| Floating-point/GEMM/reductions | Predeclared absolute/relative/ULP or problem-conditioned bounds, reference precision, accumulation policy, invariants, NaN/Inf/signed-zero/subnormal rules. |
| Concurrent/atomic algorithms | Allowed outcomes, memory ordering, safety and progress across adversarial schedules; matching one NVIDIA schedule is insufficient. |
| RNG and stochastic simulation | RNG algorithm/state/seed policy; exact stream comparison when promised, otherwise preregistered distribution tests with adequate sample size and uncertainty. |
| Physics/geometry/scientific workloads | Domain invariants, boundary cases, precision requirements and approved scientific observables; distribution agreement alone does not prove the underlying model valid. |
| Learning/inference | Identical model/input identity, operator/tensor error criteria and task-quality metrics; training uses its own convergence/statistical contract. |
| Rendering/robotics | Image and rendering metrics separate from physical state, trajectory, integration and safety invariants. |

NVIDIA is a comparison implementation, not an infallible oracle. Triangulate
with independently implemented CPU, higher-precision or analytic references
where applicable. Classify undefined-source behavior and inconsistent reference
outputs explicitly; do not tune tolerances after seeing a failing result.
Same-seed cross-backend equality is not promised unless the RNG/state contract
supports it. Stress testing or clean sanitizers do not prove every execution safe.

Scientific FP64 requirements need an explicit capability decision. Do not
silently lower them to FP32 or relax tolerances to obtain a pass. The current
M2A profile explicitly rejects FP64/VF64 and retains the existing VF64 hold.

### Trust status, math profiles and release gates

Retain evidence distinctions for translation, build, native GPU launch, fixture
result, broader differential/property checks, concurrency/error validation,
application qualification and cross-target/version regression. A build or one
vector-add pass does not imply application support. `NOT_RUN`, unavailable,
unsupported, failed and inconclusive remain separate from passes; disclose
expected negative tests without inflating positive application counts.

Define versioned strict, compatible and fast numerical profiles in the plan.
Specify actual compiler flags/APIs available on each pinned SDK, source CUDA
math settings and workload-specific tolerances. Profile names alone guarantee
nothing. In particular, do not assume newer `MTLMathMode` APIs exist on the
currently pinned Xcode 15.2/SDK 14.2; the M2 entry record already documents that
compatibility failure. Fast-profile results never count as strict results.

Each future qualification record should bind source/input/model identities,
compiler/runtime/library/backend versions, transformation route, target identity,
precision/math settings, reference method, tolerances, outputs, seeds/repetitions,
stage failures, resource limits and observed memory/time. Keep missing measurements
null/unknown. Source inspection and plans do not establish GPU execution.

Prioritize correctness and observable unsupported behavior before throughput.
Use native implementations on the same Apple chip for optimization comparisons
where available; label comparisons to different NVIDIA hardware as contextual.

## Home-wide screening evidence and its limits

The Committee screened `/home/rog` on 2026-09-25 using Git-root discovery,
tracked source/build/dependency searches, GPU-source suffixes and accelerator
API/framework references, plus selected non-Git/top-level searches. The raw
discovery counted **133 Git metadata locations**, including nested dependencies,
archives, copies and tooling. This is not 133 independent active projects and
not 133 demonstrated GPU applications.

Representative source anchors for planning:

| Family | Accessible local evidence | Classification |
| --- | --- | --- |
| Exact transform/codec | `/home/rog/business/YFCE/lmsluice/lmsluice/merge.cu`; `/home/rog/business/YFCE/lmz/lmz/gpu/lmzgpu.cu` | First-party CUDA source; siblings remain read-only. |
| Dense math/overlap/PTX | `/home/rog/business/YFCE/vram/probe/gemm.cu`; `probe/overlap.cu`; `fused/fused_gemm.cu` under the same vram root | First-party CUDA/cuBLAS source. |
| Inference/concurrency | `/home/rog/business/YFCE/experiments/e4_perception/bench.py`; `/home/rog/business/YFCE/organism/ladder/run_ladder.py` | PyTorch CUDA paths. |
| LLM adaptation | `/home/rog/business/customize-llm/bench/qlora_bench.py` | Framework-level CUDA workload; no handwritten CUDA required. |
| Calorimeter ML | `/home/rog/Papers/caloml/geant4/Par04-bench/training_vae/core/model.py`; `/home/rog/Papers/caloml/repos/CaloDiffusionPaper/scripts/train_diffu.py`; `/home/rog/Papers/caloml/repos/calo_dreamer/src/main.py` | TensorFlow/PyTorch paths plus reference/reproduction repositories. |
| Particle transport | `/home/rog/Papers/caloml/repos/celeritas/src/celeritas/em/model/BetheBlochModel.cu` | Research dependency/reference CUDA source; another celeritas-develop tree is not an independent workload by default. |
| Tracking/geometry | `/home/rog/oss/acts/Plugins/Gnn/src/CudaTrackBuilding.cu`; `/home/rog/sw/src/vecgeom-1.2.11/source/CudaManager.cu` | OSS/reference CUDA paths; ACTS tree also contains Traccc/Detray. |
| Jagged analytics | `/home/rog/oss/awkward/src/awkward/_connect/cuda/cuda_kernels/awkward_ByteMaskedArray_to_IndexedOptionArray.cu`; `/home/rog/oss/uproot5/src/uproot/models/RNTuple.py`; `/home/rog/oss/coffea/src/coffea/ml_tools/torch_wrapper.py` | CUDA/CuPy/framework library capabilities, not evidence of first-party GPU runs. |
| Fitting/deep learning | `/home/rog/oss/root/roofit/batchcompute/src/ComputeFunctions.cu`; `/home/rog/oss/root/tmva/tmva/src/DNN/Architectures/Cuda.cu`; `/home/rog/oss/zfit`; `/home/rog/oss/pyhf` | CUDA and framework-enabled library populations; GPU selection/availability needs per-case verification. |
| Photon transport | `/home/rog/business/opticks-deploy/gate0/optix/optix_triangle.cu`; `/home/rog/business/opticks-deploy/opticks/CSGOptiX/CSGOptiX7.cu` | CUDA/OptiX source and local integration harnesses. |
| Robotics/simulation | `/home/rog/MuJoCo/reference/src/mujoco/mjx/mujoco/mjx/_src/io.py`; `/home/rog/MuJoCo/projects/mujoco-physics-checks/checks/render_frames.py` | Reference MJX/JAX/Warp paths; first-party rendering evidence is separate from GPU physics. |

Scope limits: broad keyword searches had false positives such as `cupy` within
`occupy`, vendored Eigen SYCL headers and GPU mentions in documentation. Those
were filtered in the interpretation. General framework imports do not prove GPU
placement. Some outputs were sampled/truncated; ignored content and installed
environments were not exhaustively audited. Negative results mean **no direct
GPU implementation found in screened first-party files**, not proof of no GPU
use. The survey did not execute any workload, inspect active process GPU use,
validate current scientific results or establish Metal support. Do not inherit
the earlier conversational phrase “full audit” as an exhaustive validation claim.

## Approved priorities to map into the existing canonical order

These are priority categories for the owner to integrate into existing stable
steps, not a second executable project plan or permission to run them now:

| Category | Intended emphasis |
| --- | --- |
| P0 | Reconcile current Native AOT/M2A status; design workload identity, independent references, numerical profiles and evidence/qualification gates. |
| P1 | Small exact byte/integer/indexing examples drawn from lmz/lmsluice and selected Awkward algorithms, with licensing/provenance and compiler prerequisite checks. |
| P2 | Dense numerical kernels, reductions, transfer/stream behavior and selected ROOT/TMVA primitives, gated by actual compiler/runtime/library capabilities. |
| P3 | Irregular arrays, geometry, transport and tracking; small extracted representative cases before full Celeritas/ACTS/VecGeom applications. |
| P4 | Framework applications from customize-llm, CaloML and MuJoCo; qualify candidate backend viability and end-to-end metrics separately from unchanged CUDA compatibility. |
| P5 | NVIDIA-specific frontier such as OptiX/PTX/matrix/cooperative-grid cases, with explicit redesign or unsupported outcomes. |

Choose a bounded representative workload per missing semantic family before
expanding many near-duplicate programs. Priorities are subject to prerequisites;
they are not claims about porting cost or feasibility. Do not reopen closed
M1/M2 architecture choices or interrupt the current packet merely to fit this
taxonomy. Record the earliest eligible planning/implementation gate after the
existing task completes.

## Concrete documentation scope and acceptance

Read every line of all current cuda4AS planning documents, including the legacy
master/history, relevant parent blueprint, M-stage plans/ADRs and recursive
subplans, before planning edits. Inventory reviewed paths and hashes in a
separate agent ledger. Check the parent's blueprint as an input without changing
the parent repository. Repository copies/build exports are historical inputs,
not new plan authorities. Do not substitute this packet for that review.

Reuse the existing canonical master if one now exists; otherwise reconcile to
`/home/rog/business/YFCE/cuda4AS/plan/PLAN.md` under current policy. Preserve every
original term, step and explanation in the canonical hierarchy. Mark superseded
orders/statuses explicitly **retired**, add current replacements and mappings,
keep stable IDs, and retain exactly one current execution order per document.
Use uniquely linked step subplans only when an expanded step needs detail.
Preserve reference paths with explanatory pointers as needed. Agent bookkeeping
belongs in a separate ledger. This Codex-to-Codex update does not trigger a new
non-Codex takeover snapshot or private-repository creation.

Accept the packet only when:

- The canonical plan incorporates all four routes, coverage dimensions, result
  contracts, trust stages, profile distinctions and prerequisite-based priorities.
- Unchanged-source compatibility and adapted-application qualification have
  separate claims, denominators and acceptance evidence.
- Current M2/M2A facts are reconciled against accepted evidence; obsolete M1R
  text is retained but retired. Unknown native results stay unknown.
- `CUDA-PAP-0` remains optional/not selected without a new PaP engine dependency;
  independent affordability and specialist assessments remain independent.
- The smallest future implementation unit is specified with entry/exit gates,
  references, adversarial cases and resource limits, but is not dispatched here.
- Full planning-review coverage, one master/current order, unique linked
  subplans, local links and the scoped documentation diff are validated.
- Dirty baseline hashes and Git state are recorded, pre-existing work and
  `RESULTS/m2a/` are preserved, and no sibling/code/environment change is mixed in.

## Report and return

Required complete report:
`/home/rog/.codex/handover/2026-09-25-cuda4as-workload-trust.REPORT.md`.

After the entire bounded packet and validation finish, return that report to
**YFCE Supreme Committee**, thread **01a08bd8-b222-7c21-9509-56020e4a5515**.
Include every reviewed planning path/revision, baseline, changed documents,
step mappings, accepted/rejected/deferred concepts with reasons, validation,
positive/negative/failed/skipped/NOT_RUN/inconclusive findings, preserved files,
evidence hashes, delivery state and unresolved decisions. The return is required
as part of the prior user-authorized cross-project reporting workflow.

Final-only reporting to the originating Committee; no routine interim reports.
A strong evidence-backed contradiction to the brief may be escalated early.
User decisions outside this packet belong in the recipient's own visible
conversation. When return transport is unavailable, retain the complete report
and record delivery pending under the standing rules. A saved report is not
evidence of this Committee's review or acceptance.
