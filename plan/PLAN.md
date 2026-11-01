# cuda4AS canonical project plan

Status: active shared master plan

Current planning revision: CUDA-TRUST-INTAKE-01, 2026-09-25 — superseded as
the current revision by CUDA-INTEGRATE-01 below; retained as history.

Additive clarification: CUDA-PORTFOLIO-ROUTE-01, 2026-09-27

Detailed planning expansion: 2026-09-27, requested directly by the user.
The [milestone subplan index](#17-detailed-subplan-index) links the complete
refinements. Section 12 remains the sole project execution order.

Current planning revision: **CUDA-INTEGRATE-01, 2026-11-01**, written by the
Linux session "cuda4AS Committee" on the user's instruction *"help to update
the whole future plannings, then draft a handover guide for next development
on a M1 macbook pro"*. It reconciles this plan with the M4 Mac mini line that
was merged into `master` on 2026-11-01 (merge `382a66f`, parents `1f75183`
and `be172c0`; tag `pre-mac-integration-2026-11-01`), classifies the merged
toolkit against the accepted architecture, retires M2A-R2 as the next unit,
adds the INT and TK steps, and names the M1 MacBook Pro as the next
development machine. Every earlier line of this file is retained; changes are
additions and dated retirement notes. Section 12 keeps one execution order:
its 2026-09-27 table is retained there as retired history and the current
table follows it.

This is the single shared planning authority for cuda4AS. Every AI agent and
tool working on the project uses this file. Agent-specific watermarks,
checkpoints, and review ledgers are references only and must not become
alternative execution plans.

## 1. Authority, scope, and evidence boundary

cuda4AS is a native ahead-of-time CUDA compatibility toolchain for Apple
silicon. The product goal remains unchanged CUDA source and familiar CUDA build
boundaries wherever the declared compatibility profile supports them, with
execution on the Apple GPU and no silent CPU fallback or runtime compilation.
The measurable success criterion is qualified GPU workload coverage: a named,
immutable workload executes on a named Apple target and satisfies its declared
semantic, numerical, resource, and result contracts.

The project is project-led and architecture-aware:

- representative real workloads determine feature priority and expose missing
  semantics;
- compiler, library, framework, and NVIDIA-specific boundaries are separated
  so that success in one route is not counted in another;
- existing products and specifications inform design and competitive targets,
  but upstream claims are not cuda4AS evidence;
- correctness and inspectability gate performance work;
- every compatibility or performance claim is tied to a frozen input,
  environment, profile, route, comparator, and evidence record.

This planning revision is documentation and decision reconciliation only. It
does not authorize compiler or library implementation, downloads, installation,
GPU execution, sibling-project changes, a merge, a release, or publication.
Future work begins only through a separately approved scoped handover.

Revision CUDA-INTEGRATE-01 (2026-11-01) is again documentation and decision
reconciliation. It authorizes no run by itself. The scoped handover for the
first new step, TK-0, is `HANDOFF.md` Part A (the M1 MacBook Pro handover);
TK-0 begins when the user starts the named session on that machine from it.

## 2. Preserved planning history and retired authority

All historical planning text remains preserved in its original document.
Retired means historically retained but non-executable. It does not mean
deleted, disproved in every context, or silently replaced.

- planning-codex/cuda4AS-project-plan-codex.md is the complete 2026-09-10
  strategy record. Its role as an executable master and its embedded immediate
  execution order are retired in favor of this file.
- planning-codex/history/cuda4AS-project-plan-codex-pre-M0-2026-09-06.md is the
  pre-M0 proposal and takeover record. Its proposed execution order is retired.
- docs/m0 records retain the sealed baseline, corpus, assumptions, dependency
  evidence, and compatibility vocabulary.
- docs/m1 records retain the candidate audit, result schema, rubric, inventory
  workflow, PyTorch audit, and synchronized grid.
- docs/m2 records retain the Route B architecture decision, entry-gate evidence,
  implementation handoff, and M2A design history.
- README.md remains the product and repository introduction, but this file is
  the planning authority.
- the parent YFCE blueprints remain contextual design and portfolio inputs, not
  a second cuda4AS execution plan.
- generated candidate roadmaps inside dist are source-package evidence only.
  They are not cuda4AS planning authority.

Added 2026-11-01 — documents that arrived with the M4 Mac mini line (merge
`382a66f`). All are preserved unchanged except for dated notes; none is a
plan:

- `HANDOVER.md` (2026-10-26, the M4 "cuda4AS Committee" session to the
  Antigravity session "cuda4AS"): its §4 decisions D1–D4 are adopted into
  section 3; its §6 open items and §7 residuals are tracked in sections 3
  and 16; its §5 "Next step" (N1–N10) and §3 status table are retired as an
  execution order and status authority. Its register role (sections 5–7)
  continues for the M4 machine only.
- `HANDOFF.md` (2026-10-26, "Apple Silicon Machine Handoff Guide"): its §6
  "Immediate Next Goal: Stage S4" is retired as an execution order. The file
  now carries, as Part A, the 2026-11-01 handover to the M1 MacBook Pro.
- `samples/caloml/PLAN.md` (2026-10-21, the M4 Committee): the caloml sample
  record. Its stages S0–S3 are closed history; S4–S7 and decisions D1–D4 are
  migrated into [M6-7](subplan_M/subplan_M6/M6-7.md) and section 3. Its
  "plan of record" statements are retired. Because this project has one plan
  file, it is renamed `samples/caloml/CALOML-RECORD.md` with its content
  intact; references in the M4 documents are annotated, not rewritten.
- `toolkit/DESIGN.md` (2026-10-26, Antigravity): the device lowering design
  note. Its §2 mapping rules are adopted (section 3); its "Plan of Record"
  status label is retired.
- `README.md` gate table and "Where the work happens" (2026-10-26): product
  introduction; its status cells are reported statements, not accepted
  evidence (section 3, F3).
- `samples/caloml/S1-BRIEF.md`, `S1b-BRIEF.md`, `S1c-BRIEF.md`: the M4
  Committee's executor briefs, closed.
- `RESULTS/g-c1/*.json`, `RESULTS/s1/*.json`, `RESULTS/as-phase0/
  masshjs-Mac-mini/`, `RESULTS/cuda-projects/`: evidence records; their
  evidence state is given in sections 3 and 9.
- `handover/2026-09-25-workload-trust.md`, `handover/2026-09-27-*.md`: the
  Codex-era packets CUDA-TRUST-INTAKE-01, CUDA-BACKEND-BOUNDARY-01 and
  CUDA-PORTFOLIO-ROUTE-01, now tracked; all three were absorbed by TRUST-0.

The following prior active statements are explicitly retired:

- M1R as a pending one-patch retry: the retry completed negatively after a
  second independent CuMetal build defect.
- CuMetal as the M2 implementation base: Route A stopped and Route B was
  selected.
- llama.cpp as the automatic first application anchor: it is now one candidate
  subject to enrollment and evidence readiness.
- raw project or API counts as a support metric: support is measured by
  qualified behavior and coverage, not name counts.
- NVIDIA output as a sole infallible oracle: NVIDIA is one triangulation leg
  and is subject to the same input, profile, and evidence discipline.
- hidden CPU fallback or approximate FP64 as an acceptable Native AOT result:
  unsupported precision is explicit and cannot be relabeled as GPU success.
- broad Phase 0 capability language as a compatibility claim: discovery and
  static analysis establish priorities, not executable support.

Retired on 2026-11-01 (CUDA-INTEGRATE-01):

- M2A-R2 as the next eligible unit and the package-to-Mac delivery route for
  the first live closure: superseded by TK-0 and TK-3 (section 12).
- "S2 is next" (`HANDOVER.md` §3), "Next: S4" (caloml record), "Current next
  step (N10 / caloml S4)" (`HANDOVER.md` §5) and "Immediate Next Goal: Stage
  S4" (`HANDOFF.md` §6) as execution orders: section 12 is the only order.
- "plan of record" as a label for any file other than this one.
- "9/9 PASS, bit-identical" for the nine oracles and "39/39 gpu-labelled
  tests passed" for Celeritas as accepted Apple-GPU evidence: they are
  REPORTED_NOT_REPRODUCED (section 3, F1 and F3) until TK-0 and TK-2.
- "Since 2026-10-14 development is on an Apple M4 Mac mini" as the project's
  target statement: the next development machine is the M1 MacBook Pro
  (section 10); the M4 is a second sample.
- "Phase 2-2 onward (not started)" (`HANDOVER.md` §3): the M4 line started it;
  its state is given by section 3.
- The Antigravity session's standing as the sole owner of planning and
  decisions (`HANDOVER.md` line 7): planning authority is this file; the user
  decides who executes on which machine.

## 3. Accepted current architecture

The selected architecture is Native AOT Core v1:

- an unchanged CUDA-facing boundary for enrolled inputs;
- a pinned Clang CUDA annotation and builtin facade;
- Clang AST or LLVM-derived compiler semantics as the source for declarations,
  types, address spaces, expressions, control flow, and builtins;
- an owned typed semantic IR with explicit source provenance;
- a real whole-program device-link boundary;
- ahead-of-time MSL generation and metallib construction;
- an Objective-C++ Metal runtime with explicit resource and launch contracts;
- no network access, installation, runtime compilation, or CPU fallback in a
  qualification run;
- FP32 and int32 as the first active profile; FP64 and VF64 remain explicitly
  held until separately specified and validated.

The accepted M2A-R1 portable correction at commit
de69952dc986173d0b30cca77f55c07b4c14b0ed establishes a portable frontend
contract only. It does not establish live Clang conformance, native Apple
buildability, GPU launch, numerical correctness, performance, or general CUDA
compatibility. The earlier native M2A package failed at device import; dependent
stages did not run.

### Toolkit v0.1 (M4 line, 2026-10-26) measured against this architecture — added 2026-11-01

The merge of 2026-11-01 brought `toolkit/` into `master`: an `nvcc` driver
(`toolkit/src/driver/nvcc_main.cpp`), a translator
(`toolkit/src/translator/cuda4as_translate.py`), clean-room CUDA headers
(`toolkit/include/`), stock CCCL 3.2.0 installed beside them on the M4, and an
Objective-C++ Metal runtime (`toolkit/src/runtime/cudart.mm`). The Committee
read it against the accepted architecture on 2026-11-01. Findings, with the
lines they rest on, and the resulting classification:

- **F1 — silent host emulation.** `Cuda4ASContext::launchKernel`
  (`cudart.mm:202-262`): when no Metal pipeline exists for a kernel and the
  stitched host entry exists (the driver produces one whenever the device IR
  compiles, `nvcc_main.cpp:227-245`), the kernel body runs on the CPU, one
  thread at a time in nested loops, with `__syncthreads` reduced to a fence,
  and the call returns `cudaSuccess`. Nothing is logged, and MSL compile
  failures are swallowed (`cudart.mm:119-128`). This is the fallback that
  sections 1 and 15 forbid. Consequence: every result produced through this
  runtime is TRUST-BUILT at most until the dispatch path of each launch is
  recorded. That covers the Celeritas S3 result (214/215 tests, 39/39
  gpu-labelled) and every oracle record whose mechanism is not a Metal
  dispatch proved by a trace.
- **F2 — regex front-end.** The translator's semantic authority is regular
  expressions over source text (`cuda4as_translate.py:156-215`), not Clang
  facts. M2E's preserved limit, "raw-source pattern matching must never become
  the semantic authority", applies. The Clang-semantic importer of M2A-R1
  (`tools/m2a/clang_ast.py`, `tools/m2a/native_aot.py`) remains the accepted
  direction; the two are converged in TK-3.
- **F3 — unreproducible records.** `RESULTS/g-c1/*.json` were not written by
  any tracked harness: `oracle/run_oracle.py` needs `nvcc` and `nvidia-smi`
  and has no `--driver` option, and no tracked file writes
  `mechanisms_exercised`. Their reference hashes equal those in
  `oracle/ref/*.json`. `mma_inline_ptx.json` claims a bitwise match although
  the tracked translator refuses `mma`/`wmma` inline PTX
  (`cuda4as_translate.py:166`) and the driver then fails the compile
  (`nvcc_main.cpp:164-170`); no tracked path can have produced that output.
  Classification of all nine: REPORTED_NOT_REPRODUCED; `mma_inline_ptx`
  additionally INCONSISTENT. TK-0 reproduces them with a tracked harness.
- **F4 — machine constants baked in.** `multiProcessorCount = 10`,
  `clockRate = 1450000`, `cudaDevAttrMultiProcessorCount = 84` ("to match
  reference occupancy"), a 64 KiB per-block spill buffer on every launch, MSL
  fixed at 3.0, `/usr/bin/python3`, and the M4 author's home directory as the
  translator fallback path (`nvcc_main.cpp:152`). Project rule 1 applies
  (this PC is the instrument; parameterise); TK-1 owns the repair.
- **F5 — a driver surface CMake accepts.** `enable_language(CUDA)` and
  `find_package(CUDAToolkit)` succeed; `-c`, `-o`, `-M*`, `-Xcompiler`,
  `-arch` and `-gencode` are handled or ignored deliberately; the clean-room
  headers compile Celeritas 0.6.3 and stock CCCL 3.2.0. This is real
  M2B/M3-surface work and is adopted as the seed. `nvlink` and `fatbinary`
  are 20-line stubs that write empty files; M2B's no-op-dlink clause applies
  to them.

Classification, binding until a later revision changes it:

| Component | Decision | Continues in |
|---|---|---|
| Lowering rules of `toolkit/DESIGN.md` §2: kernel ABI, D2 arrival-ticket `blockIdx`, D3 `gpuAddress` pointers with full residency, D1 soft64 RN-even, barrier and fence mapping, SIMD-group intrinsics, atomics, >32 KiB spill, intra-warp spin-lock refusal | ADOPTED as the Metal lowering contract of Native AOT Core v1 | TK-1, TK-3, M4 |
| Decisions D1–D4 of the caloml record, accepted by the user on 2026-10-26 | ADOPTED as architecture decisions. D1 defines the held FP64 profile (soft64, IEEE binary64 memory format, round-to-nearest-even, single-rounding `fma`; df64 opt-in only) without lifting the FP64 hold of section 3 | §8, M6-7 |
| `nvcc` driver, install layout, version files, clean-room headers, CCCL 3.2.0 integration | ADOPTED as seed; hard-coded paths and the emulated `sm_80` / "CUDA 12.0" identity to be made explicit and parameterised | M2B, TK-1 |
| Metal runtime `cudart.mm` | ADOPTED as seed and CONTAINED: host emulation becomes opt-in and loud; every launch records its path | TK-1, M3 |
| Regex translator | RETAINED as a lowering back-end only; its role as semantic authority is RETIRED | TK-3 |
| `RESULTS/g-c1/*.json` (M4) | REPORTED_NOT_REPRODUCED; `mma_inline_ptx` INCONSISTENT | TK-0 |
| Celeritas S2 (float build, 932 targets, 0 edits) | TRUST-BUILT evidence of the driver and headers on the M4, reported; lives in `~/system-env`, no record under `RESULTS/` | TK-2 |
| Celeritas S3 (device test ladder) | dispatch path unknown; not GPU evidence until TK-2 | TK-2, M3 |
| `tools/m2a` importer, packages, analyzers and tests (Codex lane) | RETAINED as the Clang-semantic seed and regression suite; the package-to-Mac delivery route is RETIRED, the toolkit builds on the Mac directly | TK-3 |
| `probe/p9_lookback.swift`, `probe/p10_fp64.swift` and `RESULTS/s1/` (M4) | ACCEPTED as probe evidence for the M4 under the M4 Committee's reviews in the caloml record; to be re-run per chip (residency and 64-bit single-copy results are per-chip facts) | TK-0 |

## 4. Compatibility routes and claim denominators

Every enrolled case belongs to exactly one primary route for its headline claim.
Supporting evidence may cross routes, but denominators may not be merged.

### R-KERNEL — unchanged CUDA language and kernel route

Measures whether an unchanged enrolled CUDA translation unit passes the declared
frontend, semantic, device-link, AOT, launch, and result contracts. This is the
only route that may establish an unchanged-CUDA kernel compatibility claim.

### R-LIB — CUDA library compatibility route

Measures source and behavioral compatibility for declared library surfaces such
as runtime, BLAS, FFT, sparse, random, image, or collective operations.
Translated primitives, vendored replacements, and native framework calls must
be identified individually. Kernel-route success does not imply library support.
Future named surfaces may include cuBLAS, cuDNN, CUB, Thrust, or other CUDA
libraries, but each is independently qualified for API, numerical, algorithm,
workspace, resource, target, and license behavior.

### R-APP — adapted framework or application route

Measures a documented application or framework adaptation on Apple GPU. Adapter
patches, backend selection, graph rewrites, custom operators, and excluded
features are part of the claim. R-APP success must never be reported as
unchanged-CUDA compatibility.
MPS, MLX, Core ML, JAX/Metal, ONNX Runtime, or another Apple backend may be an
R-APP route only after its actual availability, operator coverage, version,
license, target, and numerical behavior are recorded.

### R-NVIDIA — NVIDIA-specific replacement or unsupported route

Classifies features tied to NVIDIA hardware or proprietary behavior. Each item
is either replaced by a disclosed Apple-native contract, emulated under a
bounded profile, or explicitly unsupported. Examples include warp-specialized
assumptions, tensor-core-specific instructions, CUDA graphs, dynamic
parallelism, cooperative launch details, and vendor-library-only algorithms.

## 5. Workload qualification taxonomy

The project tracks semantic difficulty and target relevance, not just repository
popularity. An enrolled workload has an immutable source identity, route,
profile, fixtures, comparator, resource envelope, and result contract.

- W-EXACT: integer, indexing, copy, transform, codec, lossless decode, and other
  byte-exact kernels.
- W-DENSE: dense vector, matrix, convolution, and tensor arithmetic.
- W-REDUCE: reductions, scans, atomics, histograms, and order-sensitive math.
- W-TRANSFER: memory movement, layout conversion, host/device transfer, and
  synchronization.
- W-IRREGULAR: sparse, graph, pointer-rich, divergent, and dynamic workloads.
- W-SCIENCE: simulation, stencil, solver, spectral, statistical, and
  domain-science workloads.
- W-ML: model operators, training or inference components, custom kernels, and
  framework integrations.
- W-RENDER: raster, ray, image, geometry, and visualization workloads.
- W-APP: end-to-end applications with observable output and resource contracts.

Static survey candidates remain DISCOVERED_NOT_ENROLLED until their license,
source identity, build path, fixture, expected behavior, route, and resource
envelope are recorded. Discovery is not execution and contributes no pass.
Duplicate checkouts, vendored or reference code, installed environments, and
generated artifacts are counted separately from distinct project workloads.

## 6. Semantic and target coverage dimensions

Coverage records must expose at least the following dimensions when applicable:

- declarations, overloads, templates, namespaces, linkage, and translation-unit
  boundaries;
- scalar, vector, aggregate, pointer, qualifier, alignment, layout, and address
  space facts;
- indexing, builtin variables, calls, casts, control flow, loads, stores, and
  expression semantics;
- global, constant, threadgroup/shared, private, and generic memory behavior;
- barriers, atomics, memory order, scopes, synchronization, and host/device
  ordering;
- separate compilation, device symbols, relocations, registration, and real
  whole-program device link;
- runtime allocation, copy, stream, event, launch, error, and lifetime behavior;
- library entry points, data types, algorithms, determinism, workspace, and
  fallback behavior;
- architecture or vendor assumptions and the disclosed replacement boundary;
- target facts for Apple GPU family, OS, Xcode, SDK, Metal language version,
  compiler, and selected numerical profile.

Coverage status is evidence-derived. Unseen or unexecuted dimensions remain
NOT_RUN or unsupported; they are never inferred from a nearby passing case.

### Versioned compatibility surfaces

Backward compatibility is declared, not assumed. Each release matrix identifies
the accepted CUDA language subset, toolkit/header identity, runtime and library
API subset, host compiler range, CMake or build-system behavior, Apple target
range, and numerical profile. A newer frontend or API implementation does not
automatically inherit an older or newer CUDA version claim. Removed, changed,
emulated, and unsupported behavior remains visible with migration guidance and
negative tests. Build files, registration boundaries, symbols, diagnostics, and
error behavior are compatibility surfaces alongside source syntax and results.

## 7. Result contracts

Each enrolled case selects one versioned result contract. A project may add
domain-specific clauses, but may not weaken the selected base contract without
creating a new profile.

### RC-EXACT-1

Byte-for-byte or element-for-element identity, including shape, type, length,
and stable output hashing. Used for integer and exactly specified behavior.

### RC-FLOAT-1

Finite/non-finite classification, absolute and relative tolerance, ULP policy
where relevant, signed-zero and subnormal treatment, accumulation sensitivity,
and a justified comparator. Aggregate statistics cannot conceal per-element
violations unless the contract explicitly permits them.

### RC-CONCUR-1

Correctness across declared launch sizes and schedules, repeated-run stability,
race-sensitive stress, synchronization and atomic invariants, and explicit
determinism expectations. A single successful schedule is insufficient.

### RC-RNG-1

Generator identity and seed semantics, reproducibility class, distribution
tests, independence assumptions, sample size, and false-positive controls.
Statistical similarity is not bitwise equivalence.

### RC-SCI-1

Domain invariants, conservation or residual bounds, convergence behavior,
reference precision, timestep or iteration stability, and acceptance thresholds
defined before execution.

### RC-ML-1

Model and dataset identity, preprocessing, checkpoint, operator route, task
metric, per-layer or output tolerances, repeatability, and baseline comparison.
Framework adaptation and unchanged-CUDA compatibility remain separate fields.

### RC-RENDER-1

Dimensions and format, exact metadata, pixel or image metric, alpha and color
space rules, geometry or scene identity, artifact inspection thresholds, and
deterministic versus stochastic sampling policy.

### RC-ROBOT-1

Physical state, trajectory, integration, collision, constraint, stability, and
safety invariants are qualified independently from rendered images. A visually
similar frame cannot establish equivalent simulated or robotic behavior.

## 8. Numerical profiles

The selected profile is part of every claim and artifact.

### NP-STRICT-1

Prioritizes semantic and numerical fidelity: conservative math, no unapproved
contraction or reassociation, explicit denormal and rounding treatment, strict
error bounds, and deterministic behavior where the source contract requires it.

### NP-COMPAT-1

Allows documented implementation differences that remain within a frozen
application-relevant contract. Each relaxation is explicit, justified, and
covered by differential tests.

### NP-FAST-1

Allows disclosed fast-math and performance transformations. It has independent
accuracy and stability gates and cannot inherit a strict-profile pass.

## 9. Classification and evidence model

The established case classifications remain:

- PASS_GPU
- FAIL
- SKIP_ENVIRONMENT
- SKIP_UNSUPPORTED
- NOT_RUN

Administrative analysis may additionally report unavailable or inconclusive
evidence, but those labels do not replace the normative case classification.

Trust is cumulative and stage-specific. These are evidence maturity states, not
a second project execution order:

- TRUST-SOURCE: sources, inputs, patches, licenses, and environment are pinned.
- TRUST-TRANSLATED: compiler-semantic import and owned IR verify.
- TRUST-BUILT: declared host, device, library, and link stages complete.
- TRUST-LAUNCHED: a named Apple GPU launches with no hidden CPU or remote route.
- TRUST-FIXTURE: the frozen result contract passes for the enrolled fixture.
- TRUST-DIFFERENTIAL: broader independent reference, property, and boundary
  checks pass or disclose their differences.
- TRUST-CONCURRENCY: adversarial schedule, memory-order, progress, error, and
  repeatability contracts pass where applicable.
- TRUST-APPLICATION: an end-to-end application contract passes with every
  adaptation and unsupported feature disclosed.
- TRUST-CROSS-TARGET: the declared target/version matrix and regression set pass.

A record may claim only its highest completed stage and must retain the first
failure plus every dependent NOT_RUN stage. A lower stage never implies a higher
one.

For every claim, record the applicable evidence dimensions:

- E-SOURCE: immutable sources, patches, licenses, manifests, and provenance.
- E-TRANSLATE: compiler-semantic import and lowered representation.
- E-BUILD: host, device, Metal, library, and final-link construction.
- E-DLINK: multiple translation units and device-link semantics.
- E-LAUNCH: real Apple-GPU identity, launch, and no hidden fallback.
- E-FIXTURE: input, expected result, comparator, and immutable output evidence.
- E-DIFFERENTIAL: CPU/specification/NVIDIA triangulation and discrepancy notes.
- E-CONCURRENCY: schedule, synchronization, race, and repeatability coverage.
- E-APPLICATION: end-to-end behavior and disclosed adaptation.
- E-CROSS-TARGET: declared Apple targets and negative boundary cases.

CPU, specifications, and NVIDIA form a triangulation rather than a hierarchy:

- the specification defines required semantics and allowed latitude;
- a CPU or independent implementation provides an inspectable reference where
  meaningful;
- NVIDIA supplies ecosystem and CUDA implementation behavior under a matched
  environment;
- cuda4AS evidence records disagreements instead of automatically treating one
  leg as ground truth.

No claim is complete without immutable source and environment identities, route,
profile, target, stage outcomes, output evidence, resource use, and negative
findings.

Every qualification record binds source, input, model, fixture, compiler,
runtime, library and backend identities; transformation route; Apple and
reference targets; source CUDA math settings; actual supported compiler flags
and APIs for the pinned SDK; numerical profile and tolerances; outputs; seeds,
repetitions and timing method; stage failures; resource ceilings; and observed
memory and time. Missing measurements remain null or unknown. Profile names and
planned flags alone establish nothing.

### Execution-path evidence — added 2026-11-01

Every launch record names its dispatch path: `metal` (a Metal compute pipeline
state was created from generated MSL and dispatched), `host-emulation` (the
device body ran on the CPU), or `refused` (a translator or runtime diagnostic
stopped it). A record without the path is TRUST-BUILT at most. Host emulation
never yields PASS_GPU; it is a CPU placement fact and, where the user opts into
it, a disclosed adaptation. A result reported by a session without a tracked
harness, raw outputs and logs is REPORTED_NOT_REPRODUCED until a tracked run
reproduces it. These two labels are planning states beside the normative case
classifications above, not new classifications.

## 10. Performance and competitive specifications

Correctness gates performance. Measurements begin only after a case passes its
selected result contract on real Apple GPU hardware.

Comparators are role-specific:

- same-chip native Metal or another Apple-native implementation measures backend
  quality and achievable hardware use;
- NVIDIA CUDA provides a contextual ecosystem reference when inputs and
  numerical profiles can be aligned;
- existing CUDA-on-Apple products are compared by supported route, semantic
  coverage, build integration, transparency, maintenance, and performance;
- CPU baselines help identify acceleration but are not the primary GPU
  competitiveness target.

Every benchmark fixes warmup, repetitions, synchronization, timing boundaries,
input size, output validation, power mode, thermal notes, compiler flags,
hardware and software identity, and reports distributions rather than a single
best run.

The former provisional goals of approximately 1.2 times native Metal latency
and 80 percent of native throughput remain research targets only. They become
acceptance thresholds only after a representative workload, comparator, and
method are frozen. A lower speed, coverage, or resource result is recorded
truthfully and drives prioritization; it is not hidden by reducing the
denominator.

### Recurring-workload portfolio utility gate

cuda4AS remains relevant to the user's personal infrastructure when a recurring
workload can use the recorded `MacBookPro18,3` Apple M1 Pro target and thereby
release useful RTX 5080 capacity. This is a workload-value question, not a new
compatibility route, result classification, execution step, or hardware purchase
assumption. No larger Mac is presumed.

Before a workload is described as useful for that offload, its qualification
record must fix its identity and revision; repeat frequency; current RTX 5080
hours per week; deadline, latency, and acceptable wall-time requirements; peak
memory and storage; precision requirements including FP64; PyTorch or other
framework dependencies; custom CUDA and CUDA-library dependencies; data-transfer,
setup, supervision, and recovery costs; and whether an existing Apple-native
framework or MPS route already satisfies the job.

A workload may be useful on the M1 Pro even when it is slower than the RTX 5080
if it meets its deadline and releases scarce RTX time after operator and transfer
costs. Peak-device ratios alone neither prove nor disprove that value. The
utility record remains separate from compatibility evidence: an R-APP
PyTorch/MPS success may solve the user's job but earns no R-KERNEL or R-LIB
credit, while an R-KERNEL pass alone does not establish a complete framework,
model, or recurring workflow. Missing measurements remain unknown and no
portfolio value claim is made until the record is complete.

Added 2026-11-01: the recorded `MacBookPro18,3` is the next development
machine as well as the utility target. As inventoried on 2026-09-06 and
2026-09-10 (`docs/m1/dependencies-mac.json`, `RESULTS/m1/`) and measured on
2026-09-05 (`RESULTS/as-phase0/Studio/`): Apple M1 Pro, 14 GPU cores, 16 GiB,
macOS 14.4 (23E214), Xcode 15.2 (15C500b), SDK 14.2, Metal compiler
32023.101, GPU families up to apple7 and metal3, Homebrew LLVM 23.1.0 at
`/opt/homebrew/opt/llvm`, CMake 4.3.3, Ninja 1.12.1, about 80 GB free,
`maxBufferLength` 8 GiB, recommended working set 11.45 GB, and
`MTLCreateSystemDefaultDevice()` nil in an Aqua session while
`MTLCopyAllDevices()` enumerates the GPU. The M4 Mac mini
(`RESULTS/as-phase0/masshjs-Mac-mini/`: 10 GPU cores, 24 GB, macOS 15.4.1,
Xcode 16.3, SDK 15.4, Metal 3.2) is a second sample. Each number carries its
machine; neither machine stands for Apple silicon, and no larger Mac is
presumed.

## 11. Representative priorities

The home-directory survey supplies a static candidate population and priority
signal, not runtime evidence. It found 133 Git metadata locations, including
nested dependencies, archives, copies, and tooling; that count is neither 133
independent projects nor 133 GPU workloads. Broad searches contained false
positives and sampled or truncated output, and ignored content and installed
environments were not exhaustively audited. No surveyed workload was executed
or qualified for Metal.

The following draft semantic bands were authored during TRUST-0 and are retained
for history, but are **retired before acceptance** because their P0–P5 labels do
not match the approved intake taxonomy. They are non-executable:

- P0: exact vector and indexing kernels that close the native AOT path.
- P1: multi-translation-unit/device-link, shared memory, barriers, reductions,
  and atomics.
- P2: dense arithmetic and common library-shaped operations with clear CPU and
  native-Metal comparators.
- P3: scientific and irregular kernels that stress memory, precision,
  synchronization, and control flow.
- P4: ML custom kernels and adapted framework components with explicit R-APP
  accounting.
- P5: rendering and complete applications with domain-specific output contracts.

The active prerequisite-based priority mapping uses new, non-recycled IDs:

- WP0: reconcile Native AOT/M2A facts and define workload identity, independent
  references, numerical profiles, trust stages, and evidence gates. TRUST-0
  closes the planning portion; M2A-R2 remains the earliest live closure gate.
- WP1: enroll small exact byte, integer, indexing, transform, or codec cases from
  lmz, lmsluice, or selected Awkward algorithms only after provenance, license,
  fixture, and compiler prerequisites pass.
- WP2: enroll dense numerical kernels, reductions, transfer/stream behavior, and
  selected ROOT/TMVA primitives only after the actual compiler, runtime, and
  required library surfaces exist.
- WP3: enroll bounded irregular-array, geometry, transport, and tracking extracts
  before attempting complete Celeritas, ACTS, or VecGeom applications.
- WP4: qualify framework applications from customize-llm, CaloML, or MuJoCo
  under R-APP, including backend viability and end-to-end metrics separate from
  unchanged CUDA compatibility.
- WP5: classify the NVIDIA-specific frontier, including OptiX, architecture PTX,
  matrix instructions, and cooperative-grid behavior, as a disclosed redesign,
  bounded emulation, or unsupported result.

Representative, read-only candidate anchors from the static survey include:

| Active priority | Candidate evidence | Current status |
|---|---|---|
| WP1 | /home/rog/business/YFCE/lmsluice/lmsluice/merge.cu; /home/rog/business/YFCE/lmz/lmz/gpu/lmzgpu.cu; selected Awkward CUDA algorithms | DISCOVERED_NOT_ENROLLED |
| WP2 | /home/rog/business/YFCE/vram/probe/gemm.cu; overlap.cu and fused_gemm.cu in that project; selected ROOT/TMVA CUDA primitives | DISCOVERED_NOT_ENROLLED |
| WP3 | Celeritas BetheBlochModel.cu; ACTS CudaTrackBuilding.cu; VecGeom CudaManager.cu | DISCOVERED_NOT_ENROLLED |
| WP4 | customize-llm qlora_bench.py; CaloML TensorFlow/PyTorch paths; MuJoCo MJX/render checks | DISCOVERED_NOT_ENROLLED |
| WP5 | opticks-deploy OptiX CUDA sources and future PTX/matrix/cooperative-grid cases | DISCOVERED_NOT_ENROLLED |
| WP3 (row added 2026-11-01) | Celeritas 0.6.3, unmodified, through CMake and stock CCCL 3.2.0: CPU oracles S0/S1 and the float GPU build S2/S3 on the M4 (caloml record; `HANDOVER.md` §5) | REPORTED_BY_M4_SESSION; dispatch path unverified; not enrolled; continues in TK-2 and M6-7 |
| WP1–WP3 (row added 2026-11-01) | `cuda-projects/ready/` — 381 third-party CUDA programs that ran clean on an NVIDIA T4 or A100 with recorded stdout hashes (`cuda-projects/ready.json`, `RESULTS/cuda-projects/READY.md`: 176 reference, 190 pass-only, 15 silent); 22 unresolved in `RETEST.md` | DISCOVERED_WITH_NVIDIA_REFERENCE_NOT_ENROLLED; enters through M4-8 as a regression source |

These sibling paths remain read-only. Candidate selection chooses one bounded
representative per missing semantic family and does not treat duplicate or
vendored copies as independent coverage.

Candidate anchors such as llama.cpp, PyTorch surfaces, CUDA samples, scientific
codes, and local home-directory projects remain candidates until enrolled.
Selection must maximize new semantic information per bounded engineering and
Mac-validation cost, not merely favor the easiest demo.

The CUDA-PAP-0 intake is closed as a non-gating no-op: cuda4AS may be used by
the PaP project only if a future PaP design explicitly selects it and satisfies
cuda4AS evidence gates. No current cuda4AS step depends on PaP, and no sibling
change is authorized.

## 12. Authoritative execution order

This table is the only authoritative current project execution order. Status
records what has happened; it does not itself authorize the next row.

> **Retired on 2026-11-01:** the table immediately below is the 2026-09-27
> order, retained unchanged as history. It is no longer the current order
> because it names M2A-R2 as the next unit and predates the M4 line. The
> current order is the second table, "Current execution order
> (CUDA-INTEGRATE-01)", further down in this section. There is one order.

| Order | Step ID | State | Current decision and exit |
|---:|---|---|---|
| 0 | M0 | CLOSED_ACCEPTED | Evidence contract and corpus baseline preserved; 25 corpus cases remained NOT_RUN. |
| 1 | M1 | CLOSED_NEGATIVE | Reuse feasibility completed; exact unmodified CuMetal candidate configured but did not build; no fixture ran. |
| 2 | M1R | CLOSED_NEGATIVE_RETIRED_ROUTE | One disclosed include patch passed its intended boundary, then a second independent CuMetal Metal-API build defect stopped Route A. |
| 3 | M2E | CLOSED_ARCHITECTURE_SELECTED | Route B, Native AOT Core v1, selected. No Apple-GPU pass was claimed. |
| 4 | M2A-0 | CLOSED_FAIL | First packaged native vector-add attempt failed at device_import; later stages were NOT_RUN. |
| 5 | M2A-R1 | CLOSED_PORTABLE_ACCEPTED | Pinned facade and Clang semantic importer corrections accepted as portable contract evidence through de69952; live Clang/native/GPU proof remains NOT_RUN. |
| 6 | CUDA-PAP-0 | CLOSED_NON_GATING_NO_OP | PaP relationship recorded as optional; no cuda4AS or sibling implementation selected. |
| 7 | TRUST-0 | CLOSED_PLANNING | Workload routes, semantic coverage, contracts, profiles, triangulation, evidence gates, and priorities reconciled into this master. |
| 8 | M2A-R2 | NEXT_ELIGIBLE_NOT_AUTHORIZED | Perform one bounded live compiler/native vector-add closure using the corrected frontend contract and current package. Exit requires either a validated Apple-GPU result or a complete stage-specific negative result. |
| 9 | M2B | PENDING | Add real multi-translation-unit device link and one cross-TU device-symbol case after M2A-R2 evidence is accepted. |
| 10 | M3 | PENDING | Establish runtime core behavior: allocation, copies, launch, errors, streams/events subset, lifetime, and no-fallback evidence. |
| 11 | M4 | PENDING | Expand language semantics using qualified P1 and P2 cases, including shared memory, barriers, reductions, atomics, types, and control flow. The retained phrase “P1 and P2” is retired; its current mapping is WP1 and WP2. |
| 12 | M5 | PENDING | Deliver the first declared R-LIB surfaces with independent library result contracts and negative boundaries. |
| 13 | M6 | PENDING | Qualify scientific, irregular, numerical, and concurrency workloads across selected Apple targets and profiles. |
| 14 | M7 | PENDING | Optimize only accepted-correct routes; benchmark against frozen native-Metal and contextual CUDA comparators. |
| 15 | M8 | PENDING | Qualify adapted ML/framework and representative application routes without converting R-APP results into unchanged-CUDA claims. |
| 16 | M9 | PENDING | Release-readiness audit: declared compatibility matrix, reproducible packaging, provenance, security, support boundaries, and evidence-backed public claims. |

### Current execution order (CUDA-INTEGRATE-01, 2026-11-01)

Rows 0–7 are unchanged closed steps. Row 8 is retired in place. Rows 9–14 are
new stable IDs. Rows 15–22 are the original pending steps with their original
text kept and dated additions appended. Status records what has happened; it
does not itself authorize the next row.

| Order | Step ID | State | Current decision and exit |
|---:|---|---|---|
| 0 | M0 | CLOSED_ACCEPTED | Evidence contract and corpus baseline preserved; 25 corpus cases remained NOT_RUN. |
| 1 | M1 | CLOSED_NEGATIVE | Reuse feasibility completed; exact unmodified CuMetal candidate configured but did not build; no fixture ran. |
| 2 | M1R | CLOSED_NEGATIVE_RETIRED_ROUTE | One disclosed include patch passed its intended boundary, then a second independent CuMetal Metal-API build defect stopped Route A. |
| 3 | M2E | CLOSED_ARCHITECTURE_SELECTED | Route B, Native AOT Core v1, selected. No Apple-GPU pass was claimed. |
| 4 | M2A-0 | CLOSED_FAIL | First packaged native vector-add attempt failed at device_import; later stages were NOT_RUN. |
| 5 | M2A-R1 | CLOSED_PORTABLE_ACCEPTED | Pinned facade and Clang semantic importer corrections accepted as portable contract evidence through de69952; live Clang/native/GPU proof remains NOT_RUN. |
| 6 | CUDA-PAP-0 | CLOSED_NON_GATING_NO_OP | PaP relationship recorded as optional; no cuda4AS or sibling implementation selected. |
| 7 | TRUST-0 | CLOSED_PLANNING | Workload routes, semantic coverage, contracts, profiles, triangulation, evidence gates, and priorities reconciled into this master. |
| 8 | M2A-R2 | RETIRED_SUPERSEDED (2026-11-01) | Never dispatched. Its purpose — a live compiler and Apple-GPU closure of the unchanged vector_add — is carried by TK-0 (tracked re-run of the nine oracles through the merged toolkit on the M1 Pro, with dispatch-path evidence) and TK-3 (Clang-semantic front-end). Its evidence list in section 13 is inherited by TK-0. |
| 9 | INT-0 | CLOSED_INTEGRATED (2026-11-01) | Codex lane and M4 line merged on `master` (`382a66f`); snapshot of the Codex lane tagged `pre-mac-integration-2026-11-01` (`be172c0`); only `.gitignore` conflicted; no content of either line changed. Record: [INT-0](subplan_INT/INT-0.md). |
| 10 | INT-1 | CLOSED_PLANNING (2026-11-01) | This revision: M4 documents absorbed (section 2), toolkit v0.1 classified (section 3), M2A-R2 retired, TK-0…TK-3 and M6-7 added, caloml plan retired as authority and renamed, `HANDOFF.md` Part A written. Record: [INT-1](subplan_INT/INT-1.md). |
| 11 | TK-0 | NEXT_ELIGIBLE — handover drafted; starts on the user's kickoff on the M1 MacBook Pro | Bring the merged toolkit up on `MacBookPro18,3` inside `~/system-env`; re-run probes p0–p10 there; run the nine oracles through a tracked harness with host emulation disabled and the dispatch path recorded per kernel. Exit: a per-kernel table for the M1 Pro (PASS_GPU / FAIL / SKIP_UNSUPPORTED / NOT_RUN, each with its path), raw records under `RESULTS/g-c1/<host>/`, and the toolkit's actual behaviour on a second chip. Any classification closes it; it is a measurement. |
| 12 | TK-1 | PENDING — after the TK-0 report is accepted | Contain and instrument: host emulation off by default and opt-in with a per-launch diagnostic; MSL compile errors surfaced; device constants discovered at run time (core count, SM mapping, clocks, spill size, MSL version); translator refusals and errors as diagnostics with source locations; hard-coded paths removed. Exit: the nine oracles re-run on the M1 Pro under the contained runtime, every result bound to a Metal dispatch or to an explicit refusal. |
| 13 | TK-2 | PENDING — after TK-1 | Celeritas 0.6.3 unmodified on the M1 Pro: CPU oracles (S0 double, S1 float), float GPU build (S2), device ladder (S3) with per-launch path accounting; everything recorded under `RESULTS/celeritas/<host>/`. Exit: how many of Celeritas' device tests dispatch to Metal, which are refused, and the float-build result on 16 GiB. Downloads (Celeritas plus three dependencies, 3.7 MB; CCCL 3.2.0, about 42 MB) are sized and approved first or copied from the M4. |
| 14 | TK-3 | PENDING — after TK-2 or an explicit decision | Front-end convergence with M2E: the semantic authority becomes Clang-derived (AST or LLVM IR) with an owned typed IR; the regex translator retires to a lowering back-end or is replaced; the lowering rules of `toolkit/DESIGN.md` §2 are kept. Exit: the nine oracles and Celeritas rungs 1–2 pass on Metal without emulation from the Clang-semantic path, on the M1 Pro and the M4. This is the capability gate M2B requires. |
| 15 | M2B | PENDING | Add real multi-translation-unit device link and one cross-TU device-symbol case after M2A-R2 evidence is accepted. *Added 2026-11-01:* the prerequisite is now accepted TK-3 evidence; the merged `nvlink` and `fatbinary` are 20-line stubs that write empty files, to which M2B's no-op-dlink clause applies; the merged driver's CMake acceptance is the seed for M2B-4. |
| 16 | M3 | PENDING | Establish runtime core behavior: allocation, copies, launch, errors, streams/events subset, lifetime, and no-fallback evidence. *Added 2026-11-01:* the seed is `toolkit/src/runtime/cudart.mm` after TK-1's containment; the Celeritas S3 record is runtime evidence of unknown dispatch path until TK-2. |
| 17 | M4 | PENDING | Expand language semantics using qualified P1 and P2 cases, including shared memory, barriers, reductions, atomics, types, and control flow. The retained phrase “P1 and P2” is retired; its current mapping is WP1 and WP2. *Added 2026-11-01:* the nine-oracle status from TK-0 and TK-1 feeds M4-4; `cuda-projects/ready/` becomes the regression source through M4-8. |
| 18 | M5 | PENDING | Deliver the first declared R-LIB surfaces with independent library result contracts and negative boundaries. *Added 2026-11-01:* stock CCCL 3.2.0 compiled through the toolkit (decision D2) is the first candidate CUB/Thrust surface; cuBLAS-class surfaces are unchanged. |
| 19 | M6 | PENDING | Qualify scientific, irregular, numerical, and concurrency workloads across selected Apple targets and profiles. *Added 2026-11-01:* the Celeritas/caloml route S4–S7 is [M6-7](subplan_M/subplan_M6/M6-7.md); the soft64 double build (D1) is the FP64 decision candidate; Geant4 is gated by the user. |
| 20 | M7 | PENDING | Optimize only accepted-correct routes; benchmark against frozen native-Metal and contextual CUDA comparators. |
| 21 | M8 | PENDING | Qualify adapted ML/framework and representative application routes without converting R-APP results into unchanged-CUDA claims. *Added 2026-11-01:* caloml's `Par04-bench` (S6) is an R-APP candidate. |
| 22 | M9 | PENDING | Release-readiness audit: declared compatibility matrix, reproducible packaging, provenance, security, support boundaries, and evidence-backed public claims. *Added 2026-11-01:* any release discloses emulation, refusals and the toolkit's "CUDA 12.0" / `sm_80` compatibility labels as declared surfaces. |

## 13. M2A-R2 bounded gate contract

> **Retired on 2026-11-01 as the next unit.** M2A-R2 was never dispatched;
> the M4 line built a toolkit that compiles and launches the unchanged oracles
> on the Mac directly, so the package-to-Mac route this section describes is
> no longer the smallest eligible unit. The section is retained unchanged
> below. Its required-evidence list and adversarial cases are inherited by
> TK-0's harness contract ([TK-0](subplan_TK/TK-0.md)); its live Clang
> capture and semantic-import requirement is inherited by TK-3.

M2A-R2 is the smallest eligible next unit; this section defines its gate but is
not authorization or a second execution order.

Entry conditions:

- manager acceptance of this planning revision;
- a separately approved executor handover;
- a clean, inventory-bound package derived from the current corrected source;
- pinned supported Clang identity and captured real compiler output;
- unchanged enrolled vector_add source and oracle identity;
- no network, installation, update, sudo, runtime compilation, or CPU fallback.

Required evidence:

- facade header and manifest integrity;
- live Clang AST JSON capture with compiler command, version, working directory,
  source hashes, and raw capture hash;
- semantic import and IR verification;
- device-link stage record even for the single-TU profile, with its exact
  boundary made explicit;
- MSL, Metal compile, metallib, host compile, and final native link records;
- real Apple-GPU device identity and launch evidence;
- immutable input and output bytes, hashes, comparator result, and timings;
- complete PASS, FAIL, SKIP, and NOT_RUN stage accounting.

Adversarial cases must include corrupted facade identity, unsupported compiler
identity, malformed or inconsistent AST, provenance mismatch, unsupported
address space, and a no-Clang path that remains NOT_RUN rather than synthetic
success.

Exit is one of:

- PASS_GPU for the enrolled vector-add contract with all dependent evidence;
- FAIL with the first failing stage and dependent stages NOT_RUN;
- SKIP_ENVIRONMENT or SKIP_UNSUPPORTED with a complete reason and preserved
  evidence.

The portable resource envelope remains at least 5 GiB free space, no more than
four build jobs, no more than 20 GiB of added disk, and a 30-minute bounded run
unless a future approved handover changes it.

## 14. Later milestone gates

M2B must prove real cross-translation-unit device semantics rather than
concatenation or host-only linking. M3 must demonstrate runtime behavior and
explicit errors without hidden fallbacks. M4 through M6 expand only with
enrolled cases and versioned result contracts. M7 cannot optimize a case that
has not passed correctness. M8 must report every adaptation and maintain route
separation. M9 publishes only the intersection of accepted evidence; unknown,
unavailable, and unsupported surfaces remain visible.

Regression retains every accepted case and every meaningful negative fixture.
Previously accepted behavior may be reclassified only with new evidence,
preserved history, and a disclosed reason.

Added 2026-11-01: TK-0 may close with any per-kernel classification; it is a
measurement, not a capability gate. TK-1 closes only when no launch can
succeed silently on the CPU. TK-2 closes with the dispatch-path accounting of
Celeritas' device tests, whatever the count. TK-3 is the capability gate that
M2B requires. No TK step lifts the FP64 hold; the double build is M6-7.

## 15. Provenance, safety, and maintenance

- Pin sources, submodules, patches, toolchains, licenses, fixtures, and expected
  outputs before qualification.
- Preserve raw returns unchanged; track redacted analyses when raw artifacts are
  intentionally ignored.
- Never count configure, compile, or package verification as GPU execution.
- Never infer a later-stage result when an earlier stage failed.
- Never hide unsupported features behind fallback, approximate precision, or
  framework adaptation.
- Keep resource and environment changes opt-in and scoped.
- Update this plan additively. Retire obsolete text explicitly and add new stable
  step IDs rather than recycling existing ones.
- Before every planning update, review every line of this master and all project
  planning documents, then record the reviewed revisions in a separate agent
  ledger.
- Add a unique recursive subplan only when a step actually needs detailed
  expansion. No subplan is required by this revision.

> **Retired on 2026-09-27:** the sentence “No subplan is required by this
> revision” described the earlier outline-only delivery. The user requested
> detailed subplans. Section 17 and its linked hierarchy now supply them; the
> original sentence is retained as history, not current planning guidance.

Added 2026-11-01:

- Record the dispatch path of every launch. A runtime that can run device
  code on the CPU does so only on explicit opt-in and says so on every launch.
- Constants calibrated on one machine — core counts, SM mapping, clocks,
  spill sizes, MSL version, tool paths — are discovered at run time or carried
  as named parameters with their derivation written down (project rule 1).
- A result record names the tracked harness, command lines, logs and raw
  outputs that produced it, or it is REPORTED_NOT_REPRODUCED.
- This project has one plan file. `samples/caloml/PLAN.md` is renamed
  `samples/caloml/CALOML-RECORD.md`; its planning content lives in M6-7.
- Agent ledgers for planning reviews live under `docs/ledgers/` and are never
  plan files. The ledger of this revision is
  `docs/ledgers/2026-11-01-cuda4as-planning-review-ledger.md`.
- On every machine the toolkit is built from source into that machine's
  `~/system-env` (the M4 convention, adopted); nothing is installed into
  anything that exists, and downloads above a few tens of MB are sized and
  approved first.

## 16. Current unresolved decisions

No unresolved decision blocks this planning reconciliation. Future user or
manager decisions are required before:

- authorizing and assigning M2A-R2;
- freezing the first post-vector-add representative workloads;
- selecting the first R-LIB surface;
- choosing Apple target breadth and numerical-profile thresholds;
- promoting provisional performance goals into acceptance thresholds;
- freezing the first recurring-workload utility record, including its acceptable
  M1 Pro wall time and transfer, supervision, and recovery budget;
- selecting any PaP, ML-framework, or end-to-end application integration.

Until those decisions and their scoped handovers exist, the associated steps
remain pending and make no compatibility or performance claim.

Added 2026-11-01 — decisions still open after CUDA-INTEGRATE-01 (the first
bullet above, "authorizing and assigning M2A-R2", is retired with M2A-R2):

- the user's kickoff of TK-0 on the M1 MacBook Pro from `HANDOFF.md` Part A;
- approval of TK-2's downloads on the metered link (Celeritas 0.6.3 plus
  CLI11, GoogleTest and nlohmann_json, 3.7 MB; CCCL 3.2.0 shallow clone, about
  42 MB), or a copy of `~/system-env/src` from the M4;
- Geant4 and Xerces-C for M6-7 stage S4 (1.5–2 GB): the version is stated as
  11.2.x in `HANDOVER.md` §5 and `HANDOFF.md` §6 and as 11.4.2 in the caloml
  record's stage table; it is fixed before sizing;
- whether host emulation remains available after TK-1 (opt-in, loud) or is
  removed from the runtime;
- the standing of the M4 Mac mini and its sessions: the M4 remains a target
  and a second sample; who, if anyone, continues to execute there is the
  user's call;
- the parent YFCE repository: its `cuda4AS` gitlink is `ad30bbd`, its local
  `master` holds nine unpushed commits and its remote `master` has moved; a
  separate handover integrates it and points the gitlink at the merged commit;
- `HANDOVER.md` §6 items O1 (missing blueprint), O5 (duplicate S1 records),
  O6 (untracked clutter) and O7 (newer Xcode) remain the user's.

Decided by this revision, recorded for visibility: M2A-R2 retired; the regex
translator is not the semantic authority; the `RESULTS/g-c1` records are
REPORTED_NOT_REPRODUCED; `samples/caloml/PLAN.md` renamed; the M1 MacBook Pro
is the next development machine; the M4 line's D1–D4 and lowering rules are
adopted.

## 17. Detailed subplan index

This is a navigation index refining the existing steps in section 12, not
another project execution sequence. Every step has one linked detail document.
Closed-step documents retain the scope, completed evidence and closure limits;
their historical order does not reopen execution. Open-step documents contain
concrete ordered tasks, inputs, deliverables, validation and stop conditions.
Complex child steps link to their unique recursive subplan. Unexpanded children
are bounded leaves; no empty placeholder documents are required.

| Parent step in section 12 | Unique detailed subplan | Coverage |
|---|---|---|
| M0 | [M0](subplan_M/M0.md) | Baseline, immutable corpus, assumptions, result contract and closeout |
| M1 | [M1](subplan_M/M1.md) | Candidate provenance, Mac inventory, fixtures, build failure and reuse verdict |
| M1R | [M1R](subplan_M/M1R.md) | One-patch retry, second build defect and closed candidate route |
| M2E | [M2E](subplan_M/M2E.md) | Native AOT architecture selection and component contracts |
| M2A-0 | [M2A-0](subplan_M/M2A-0.md) | First native package and retained device-import failure |
| M2A-R1 | [M2A-R1](subplan_M/M2A-R1.md) | Facade, compiler semantic import, JSON conformance and portable-only closure |
| CUDA-PAP-0 | [CUDA-PAP-0](subplan_CUDA/CUDA-PAP-0.md) | Optional PaP capability decision and non-gating boundary |
| TRUST-0 | [TRUST-0](subplan_TRUST/TRUST-0.md) | Qualification contracts, routes, profiles, references and portfolio intake |
| M2A-R2 | [M2A-R2](subplan_M/M2A-R2.md) | Live compiler capture, current package, native AOT run and independent return review |
| M2B | [M2B](subplan_M/M2B.md) | Device-module ABI, real linker, unchanged CMake and multi-TU regression |
| M3 | [M3](subplan_M/M3.md) | Runtime API, buffers, launch, ordering, lifetime, errors and resource limits |
| M4 | [M4](subplan_M/M4.md) | Language, pointers, shared memory, barriers, atomics, first five oracles and WP1/WP2 |
| M5 | [M5](subplan_M/M5.md) | Workload-driven library surfaces, ABI, precision, workspaces and negative cases |
| M6 | [M6](subplan_M/M6.md) | Scientific/irregular cases, concurrency, advanced feature decisions and target coverage |
| M7 | [M7](subplan_M/M7.md) | Matched benchmarking, bottleneck repair, resource efficiency and regression budgets |
| M8 | [M8](subplan_M/M8.md) | Framework routes, recurring-workload value, training/inference and optional integrations |
| M9 | [M9](subplan_M/M9.md) | Claims, deployment, provenance, reproducibility, maintenance and release readiness |
| INT-0 (added 2026-11-01) | [INT-0](subplan_INT/INT-0.md) | Git integration record of the Codex lane and the M4 line |
| INT-1 (added 2026-11-01) | [INT-1](subplan_INT/INT-1.md) | This planning reconciliation, its findings and its review ledger |
| TK-0 (added 2026-11-01) | [TK-0](subplan_TK/TK-0.md) | M1 MacBook Pro bring-up, probes p0–p10, tracked nine-oracle re-run with dispatch-path evidence |
| TK-1 (added 2026-11-01) | [TK-1](subplan_TK/TK-1.md) | Emulation containment, diagnostics, run-time parameterisation, re-run |
| TK-2 (added 2026-11-01) | [TK-2](subplan_TK/TK-2.md) | Celeritas S0–S3 on the M1 Pro with evidence under `RESULTS/` |
| TK-3 (added 2026-11-01) | [TK-3](subplan_TK/TK-3.md) | Clang-semantic front-end convergence; the gate M2B requires |
| M6-7 (added 2026-11-01) | [M6-7](subplan_M/subplan_M6/M6-7.md) | Celeritas/caloml route S4–S7, soft64 double build, Geant4 gate |

Planning is authorized by the user's request; implementation eligibility is
unchanged. M2A-0 is CLOSED_FAIL, M2A-R1 is CLOSED_PORTABLE_ACCEPTED, and M2A-R2
is NEXT_ELIGIBLE_NOT_AUTHORIZED. Creating these documents assigns no executor
job or Mac action. Future scoped handovers cite these paths and child IDs.

> **Retired on 2026-11-01:** "M2A-R2 is NEXT_ELIGIBLE_NOT_AUTHORIZED" in the
> paragraph above. M2A-R2 is RETIRED_SUPERSEDED and TK-0 is the next eligible
> step; its scoped handover is `HANDOFF.md` Part A, which assigns the job to
> the session the user opens on the M1 MacBook Pro and starts on the user's
> word there.

### Interpretation shared by the subplans

- A finished bounded attempt can return a valid negative record without passing
  the capability gate. In particular a failed M2A-R2 does not unlock M2B merely
  because its report is complete. Progress requires an accepted positive
  prerequisite or an explicit new architecture/scope decision.
- The legacy M0/M1 schemas retain `UNSUPPORTED` and `PASS_CPU_EXPLICIT` in
  their immutable records. Section 9's `SKIP_UNSUPPORTED` is the current planning
  vocabulary, not a silent schema edit. Any new analyzer must preserve raw
  labels, pin its schema, and provide an explicit mapping; CPU results carry
  no Native AOT GPU credit. No completed classification is rewritten here.
- Historical phase names in the retired plans are not reused with their old
  meanings. Current M3 owns the runtime, M4 language semantics, M5 libraries,
  M6 scientific qualification, and M8 adapted applications. The earlier
  residency/coded-tier ambition remains a held optional M8 integration decision;
  it is neither erased nor added to the default critical path.
- The first five unchanged oracles close across M2A-R2, M3 and M4. CMake and
  genuine device-link qualification belong to M2B. Initial matched timings can
  be collected after each accepted-correct native path; M7 owns broad tuning.
- Proposed future paths in subplans describe deliverables, not existing files
  or completed work. Later dependency versions, numeric tolerances and resource
  ceilings are frozen at enrollment; a proposed profile is not evidence.
- Every human-run validation delivery follows the standing published-file
  workflow: an actual tracked shell script and necessary inputs, verified at
  the remote revision, acquired by Git pull and launched with one no-argument
  `/bin/bash` command. Preparation, checks, logging and collection belong in
  that file. These plans themselves are not a human probe delivery.

- Added 2026-11-01: a launch is evidence only with its dispatch path (section
  9). The merged toolkit is a seed, classified component by component in
  section 3; a subplan that names a `toolkit/` file as its starting point
  inherits that classification, including the containment of host emulation.
- Added 2026-11-01: development now happens on the Mac itself (M1 MacBook Pro
  next, M4 Mac mini as a second sample), so the "human-run validation
  delivery" bullet above describes the drop route to Macs we do not develop
  on; on a development Mac the executor session builds and runs directly,
  inside `~/system-env`, and the user still owns commits and pushes there
  unless the user says otherwise in that session.

The former dates and effort ranges in retired plans remain historical estimates.
The next Mac return is the first useful re-estimation point. Subsequent estimates
must name their fixture, missing mechanisms and Mac turnaround assumptions;
this hierarchy does not promise a full-framework schedule from portable tests.

Added 2026-11-01: the first re-estimation point is the TK-0 report from the M1
MacBook Pro. The M4 line's "N1–N9 done in one day" is not an estimate for any
step here, because its evidence state is unresolved (section 3).
