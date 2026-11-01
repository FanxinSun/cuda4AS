# M2B — Genuine device linking and unchanged build integration

Parent step: M2B in [the master](../PLAN.md).
State: PENDING; implementation is not authorized by this plan.

## Objective and prerequisites

Turn the single-TU proof into a real separable-compilation contract. One CUDA
translation unit calls a device function defined in another; symbol resolution,
type checking and executable code must be genuine. A no-op dlink or an empty
object is explicitly unacceptable.

Require an accepted positive M2A-R2 frontend/native prerequisite and a scoped
handover. A complete negative R2 report alone is insufficient. Freeze the exact
Clang/facade/IR/ABI versions already proved, the CUDA-language subset and SDK
target. Do not mix a new compiler migration with first linker qualification.

## Inputs and owned surfaces

Existing fixtures:
tests/m1/fixtures/cmake-vector-add/CMakeLists.txt and main.cu;
tests/m1/fixtures/cmake-device-link/CMakeLists.txt, device-api.cuh,
device-function.cu, kernel.cu and main.cu.
Use docs/m1/fixtures.json for immutable enrollment. Keep both fixture trees
unchanged; a modified fixture gets a new identity and no unchanged-build credit.

Extend tools/m2a only where reuse is sound. A reusable driver/module-linker and
tests/m2b are proposed deliverables, not existing completed components.
Before implementation, freeze a versioned serialized device-module schema and
the link manifest interface consumed by aot_manifest.py/runtime.mm.

## Execution order

| Child ID | Depends on | Action and deliverable | Acceptance |
|---|---|---|---|
| M2B-0 | Positive R2; approval | Freeze fixture hashes, compile/link driver contract and symbol/ABI schema. | Exact build inputs and supported options declared before execution. |
| M2B-1 | -0 | Emit one device module per TU with definitions, declarations, source provenance, signatures and linkage. | Independent modules exist; compiler semantics preserved, no concatenated-source substitute. |
| M2B-2 | -1 | Resolve symbols, reject incompatible declarations and construct a whole-program call graph. | Cross-TU device call binds exactly once; unresolved/duplicate/ABI conflicts fail deterministically. |
| M2B-3 | -2 | Lower the resolved program to MSL/metallib and record reachable entry/device code. | Nonempty bound image contains the callee's behavior; output changes in controlled new negative fixtures demonstrate resolution. |
| M2B-4 | -3 | Implement the declared CUDA compiler-driver/CMake surface, including real device-link invocation. | Unchanged minimal and multi-TU CMake projects configure, build and execute. |
| M2B-5 | -4 | Qualify incremental builds, paths with spaces, response files, dependency files and declared archive behavior. | Changes rebuild required units; clean/incremental outputs agree; unsupported options are errors. |
| M2B-6 | -5 | Freeze compatibility matrix and return positive/negative evidence. | integration.minimal_cmake_cuda and integration.multi_tu_device_link classified independently. |

## Linker semantic contract

Module identity includes source hash, compiler argv/identity, target profile and
schema version. Symbols retain canonical types, parameter order, address-space
facts, calling convention and declaration/definition provenance. Resolve only
supported linkage forms; do not flatten namespaces or overloads by spelling.
Extern declarations are not executable definitions. Unsupported indirect calls,
recursion, dynamic parallelism, device globals or weak/COMDAT forms need explicit
diagnostics until their semantics are designed and enrolled.

A link map records every resolved and unresolved symbol, its defining module,
discard/reachability decisions and final entrypoint mapping. Duplicate strong
definitions, absent callees and conflicting ABI must fail before GPU launch.
If static archives are supported, define member extraction and ordering; if not,
reject them honestly rather than producing an apparently successful empty link.

## Build integration contract

Separate host compile, device compile, device link and native host link records.
Preserve the host-facing CUDA launch/API boundary within the declared subset.
CMake compiler identification success alone is not integration success.
No CMake shim may return success while skipping a required stage.

Declare behavior for -c, output selection, includes/defines, dependency emission,
response files, separable compilation and unsupported architecture flags.
Record actual invocations for clean and incremental builds. Fix the toolchain
driver rather than rewriting the enrolled CMakeLists.txt to custom MSL commands.
Do not install a fake nvcc globally or replace unrelated compiler configuration.

## Validation and artifacts

Positive cases: single-TU regression; cross-TU call; separate object production;
repeatable link output; unchanged CMake entrypoints; correct output bytes.
Negative cases: unresolved symbol, duplicate definition, parameter type/order
mismatch, changed module/facade hash, mixed profile/version, unsupported linkage,
missing archive member, stale dependency and unsupported flag.
Create negative inputs as newly enrolled fixtures; never mutate frozen originals.

Deliver module schema, implementation/tests, command traces, link map, compiled
artifacts and hashes, clean-build/incremental comparison, output analysis and
route/target/profile matrix. Record build time as diagnostic only; do not trade
semantic checks for speed.

## Resource, stop and closure

Default to the inherited recorded Mac and small fixtures; any resource increase
or dependency installation is a separate decision. Human-run delivery obeys the
master's published-file contract. Stop if correctness needs a new language or
ABI surface outside the handover. Do not hide a link error as environment skip.

Accept only a true cross-TU Apple-GPU pass plus the declared negative diagnostics.
A useful bounded failure is reportable but does not unlock runtime breadth by
status alone. Broader project/CMake compatibility, CUDA binary compatibility and
full nvcc command-line emulation remain unclaimed.

