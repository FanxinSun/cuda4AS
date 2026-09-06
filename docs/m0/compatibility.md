# Initial compatibility contract

Contract ID: `cuda4as-m0-initial-ampere-source-profile`, revision 0.1,
2026-09-06. This is the initial test contract, not a claim that the described
capabilities exist. M0 has run no cuda4AS compiler or runtime test.

## Product objective and declared profile

cuda4AS aims to let existing CUDA source projects build as native arm64 macOS
applications and execute their declared CUDA work on Apple GPUs with minimal
user intervention. Correct execution and useful failure behavior take priority
over performance. Virtual machines, remote CUDA execution, Linux binary
execution, and SASS execution are outside this product boundary.

The provisional first profile is:

| Dimension | Initial contract |
| --- | --- |
| Host | native arm64 macOS application and toolchain |
| Device | one Apple GPU, one process, selected by observed device capability |
| OS | macOS 14 or later only on configurations actually validated; the version number alone is not a capability claim |
| CUDA source profile | Ampere-oriented language and API slice enrolled by the corpus |
| Architecture macro | provisional `__CUDA_ARCH__=860`, subject to source-dispatch tests; it does not promise all compute capability 8.6 behavior |
| Compilation | ahead-of-time host/device build is the primary source path; runtime compilation and PTX module loading are separate capabilities |
| Default execution policy | execute with contract-preserving Apple-GPU semantics or return a clear unsupported/error result |
| CPU mode | allowed only when explicitly selected and reported as `PASS_CPU_EXPLICIT` |
| Numerical policy | faithful declared CUDA behavior by default; reduced precision is a named opt-in mode with separate results |

Architecture flags must be parsed and diagnosed coherently. A flag may select
the provisional profile when its program uses only enrolled behavior. A flag
or source path requiring an unavailable mandatory feature must produce a
source-located diagnostic. Reporting a familiar compute capability never
creates hardware or runtime behavior that has not passed conformance tests.

## Four compatibility boundaries

### Source compatibility

The pinned application `.cu`, C/C++, Python, generated-source inputs, and
headers owned by that application remain byte-for-byte unchanged. Each result
records the upstream revision, enrolled file hashes, dirty state, and source
diff. Toolchain-owned headers, shims, lowering passes, runtime libraries, and
packaging are permitted when disclosed. The toolchain must not select a hidden
replacement by source hash, kernel name, or benchmark identity.

A maintained application patch can be a useful experiment, but it does not
satisfy an unchanged-source case. The result must attach the patch and leave
the unchanged case nonpassing.

### Build compatibility

The pinned project's CMake files, setup scripts, package sources, and normal
target structure remain unchanged. Deliberately selecting cuda4AS is allowed
through documented interfaces such as a compiler path, toolchain file,
installation prefix, environment variable, or an existing upstream build
option. Every nondefault option is recorded.

Passing build compatibility requires configuration, compilation, device
compilation, device link where used, native host link, dependency generation,
and loader discovery required by the case. A translated `.metal` file alone is
not a build pass. Special application build-file edits, disabled features, or
hand-linked replacement binaries cannot satisfy this boundary.

### Execution compatibility

Observable API results, device selection, allocation and lifetime, launch
arguments, ordering, synchronization, numerical output, errors, and resource
ownership must match the enrolled CUDA contract. Build success, command
submission, numerical success, and execution backend are distinct checks.

`PASS_GPU` requires positive evidence that the enrolled work was submitted to
and completed on the Apple GPU. CPU work hidden behind a CUDA-looking success
cannot receive that status. An explicitly selected CPU implementation may
receive `PASS_CPU_EXPLICIT` after its own semantic and numerical checks.

### Deployment compatibility

The installed compiler, runtime, libraries, headers, metadata, and generated
application must work from the documented prefix on the declared Mac/OS/SDK
matrix. Build-machine requirements and end-user runtime requirements are listed
separately. A release claim requires clean configure/build/run checks,
relocatable discovery where promised, dependency and license provenance,
upgrade/uninstall behavior, and useful missing-prerequisite diagnostics.

Linux may validate portable compiler components. It cannot produce an
end-to-end native macOS compatibility pass because native arm64 linking, Apple
frameworks, Metal compilation, packaging, and GPU execution are separate
requirements.

## Result classifications

Every enrolled test has exactly one of these classifications:

| Classification | Meaning |
| --- | --- |
| `PASS_GPU` | All required source, build, execution, numerical, and Apple-GPU provenance assertions for this case passed on the recorded configuration. |
| `PASS_CPU_EXPLICIT` | The user or test explicitly selected the declared CPU mode and all CPU semantic/numerical assertions passed. It is excluded from GPU pass counts. |
| `FAIL` | The case is expected to work in the declared profile, but a required stage or assertion failed, crashed, timed out, returned the wrong result, or used an undisclosed backend. |
| `UNSUPPORTED` | The requested feature is outside the implemented contract or capability set and the toolchain reports it as such. It is excluded from pass counts. |
| `SKIP_ENVIRONMENT` | A required external condition is unavailable on this run, such as the declared device, SDK, hardware feature, or optional dependency. It is excluded from pass counts and remains enrolled. |
| `NOT_RUN` | No qualifying attempt has been made for this case and configuration. |

A negative diagnostic test can pass its own assertion while the requested
feature remains `UNSUPPORTED`. The record therefore carries assertion results
separately from the classification; an expected rejection never inflates GPU
compatibility counts. Likewise, a probe's historical top-level string `ok`
does not map automatically to any classification above.

Reports publish counts for all six classifications, the complete enrolled test
count, and exclusions with reasons. Percentages use the full declared corpus or
state their denominator explicitly. Skipped and unsupported cases are never
removed solely to improve a rate.

## Required result record

Every future result must contain enough information to reproduce and classify
it without relying on a private session:

- contract and corpus schema versions;
- test ID, source/dependency revisions, source hashes, dirty state, and diff;
- compiler, runtime, library, SDK, OS, machine, GPU, and capability versions;
- complete configure/build commands, documented toolchain-selection options,
  and relevant environment values;
- separate configure, host compile, device compile, device link, native link,
  launch, and validation outcomes when those stages apply;
- execution route (`apple_gpu`, `cpu_explicit`, or no execution), GPU submission
  evidence, input dimensions/seeds, and resource limits;
- expected and actual outcome, comparison rule, assertion results, logs, and
  the final classification;
- timing conditions only when timing is reported: first-use versus warm,
  repetitions, median/tail, power/thermal state, and measured bytes/work.

The runner must validate required fields and assertions before emitting a pass.
It must continue through independent failures and return an aggregate failure
when any required case is `FAIL`. Timeouts remain bounded; killing a host
process is not evidence that previously submitted GPU work was cancelled.

## Numerical contract

The nine inherited reference cases use exact byte comparison. Their inputs were
constructed so the recorded operations are exactly representable for those
specific shapes. That property does not establish general floating-point
conformance.

For future cases:

- exact integer and exact-representable floating-point fixtures compare every
  output byte;
- tolerance-based tests state datatype, reference implementation, absolute and
  relative thresholds, ULP treatment, reduction/order allowance, NaN and
  infinity handling, signed-zero behavior, and the reason for each tolerance;
- default FP64 aims at faithful binary64 semantics for the declared operation
  set, including exceptional values and rounding behavior. A `fast48`,
  double-float, or other reduced representation is a named opt-in numerical
  mode and cannot silently satisfy default FP64;
- fast math is selected explicitly and tested against its declared CUDA
  behavior; tolerances cannot be weakened after observing a failure without a
  versioned contract change;
- CPU results, reduced-precision results, semantic specializations, and native
  execution of the translated kernel remain distinguishable.

Golden outputs are immutable inputs. A new producer run writes a separate
candidate record and must explain any disagreement; it never overwrites an
accepted reference as part of a test run.

## Exact future pass criteria

An existing oracle case may become `PASS_GPU` only when all of the following
are true on a recorded supported Mac configuration:

1. the `.cu` source hash equals [`corpus.json`](corpus.json), with no source or
   build-file modification;
2. the documented cuda4AS toolchain configures, compiles, and links a native
   arm64 executable through the intended source path;
3. the executable submits the named kernel work to the Apple GPU, and the
   result contains backend evidence tied to that launch;
4. execution completes without an unexpected diagnostic, crash, timeout, or
   fallback;
5. the entire output length and bytes equal the enrolled reference, and its
   SHA-256 equals the manifest;
6. required logs, machine conditions, revisions, commands, assertions, and
   result classification validate against the result schema.

An integration case passes only when its own source/build tree is pinned and
clean, its stated configure/build/link/install behavior completes without
application edits, and every execution and output assertion in the manifest
passes. A semantic negative case passes its diagnostic assertion only when the
diagnostic names the original source location and construct and offers the
declared action; the requested feature still remains `UNSUPPORTED`.

No case met these cuda4AS criteria during M0. The inherited NVIDIA outputs and
Mac probes are evidence inputs, not compatibility passes.
