# cuda4AS M1 native feasibility drop

This package performs the bounded M1 source/build feasibility run on an Apple
Silicon Mac after the inventory gate has confirmed its existing tools. It does
not use the network, install or update software, invoke `sudo`, edit shell
startup files, or change the selected Xcode/SDK.

The run verifies the complete package and pinned candidate source, builds
CuMetal in a new task-local Release tree with CUDA registration enabled and its
binary shim disabled, then attempts three separately logged cases:

1. the existing `oracle.vector_add` through the typed direct native-AOT route;
2. an unchanged minimal CMake CUDA project through a disclosed source-only
   CMake adapter; and
3. an unchanged three-translation-unit CMake project with a required device
link and an external device function.

A provisional package whose `target-inventory-binding.json` still says
`PENDING_USER_INVENTORY` stops before machine preflight or any build. Only the
inventory-bound package announced by the executor is actionable.

Every launch requests CuMetal GPU provenance. Full output bytes, SHA-256,
candidate/adapter details, stage events, commands, and logs are returned. A
zero runner exit means all raw gates passed; `1` means at least one build,
launch, provenance, or exact-output gate failed; `77` means an existing
environment prerequisite was missing. The exit code is not itself a cuda4AS
classification. The returned archive must be validated and normalized against
`cuda4as-m1-result-v1` off the Mac.

Every return includes the exact target-inventory binding and static package
manifest that the runner verified before doing any work. The repository-side
`tools/m1/analyze_native_return.py` reads the return without extracting it,
checks every member against the inner manifest, confirms that binding against
the validated inventory, independently hashes outputs, and parses the raw
CuMetal provenance. It emits no `PASS_GPU` result unless all of those gates and
the normative result validator pass.

Run only with the exact commands in `docs/m1/MAC-ACTION-REQUIRED.md`. The
script creates work, results, and return archives only beneath the extracted
package directory. It prints the return archive path, byte count, and SHA-256
even when it records a controlled failure or environment gap.
