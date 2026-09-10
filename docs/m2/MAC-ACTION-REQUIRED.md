# MAC ACTION REQUIRED — CUDA4AS M2 ENTRY RETRY IS READY

The executor prepared a 137,793-byte corrected delta package. It reuses the existing
M1 package and contains no candidate or VF64 source archive. Verify:

```
cuda4as-m2-entry-retry-v1.tgz
1ffb31b06c0800b11225fa730908404604923e89cf5847e82e883e7ecb26525e
```

The existing base package must also remain exactly 8,549,223 bytes with SHA-256
`dbb390b4f470b8f286ccb65a2b8565a235e749d9122c16e8e92ade96e0099bc7`.

After pulling `codex/m2-entry-gate` on the Mac, the all-in-one command is:

```
./tools/m2/run-m2-entry-retry.sh
```

It uses the tracked M1 base and M2 delta automatically. The equivalent
explicit commands, useful when the artifacts were copied to Downloads, are:

```
DELTA_ARTIFACT="$HOME/Downloads/cuda4as-m2-entry-retry-v1.tgz"
BASE_ARTIFACT="$HOME/Downloads/cuda4as-m1-native-feasibility-v1.tgz"
shasum -a 256 "$DELTA_ARTIFACT"
mkdir -p "$HOME/cuda4as-m2/drop-v1"
tar -xzf "$DELTA_ARTIFACT" -C "$HOME/cuda4as-m2/drop-v1"
cd "$HOME/cuda4as-m2/drop-v1/cuda4as-m2-entry-retry-v1"
CUDA4AS_M1_ARTIFACT="$BASE_ARTIFACT" \
CUDA4AS_M2_TASK_ROOT="$HOME/cuda4as-m2/entry-v1" \
./run-m2-entry-retry.sh | tee native-run.console.txt
```

The previously recorded inventory is the expected environment: arm64 Apple M1
Pro / MacBookPro18,3, macOS 14.4 build 23E214, Xcode 15.2 build 15C500b,
SDK 14.2, CMake 4.3.3, Ninja 1.12.1, Homebrew LLVM 23.1.0, LZ4 1.10.0,
and Zstd 1.5.7. Do not install, update, download, invoke `sudo`, change the
fixtures or options, or apply a second patch. The run requires at least 5 GiB
free, uses at most four jobs, and should stop after 30 minutes or 20 GiB of
task-local use.

Return exactly the printed `cuda4as-m2-native-return-<UTC>.tgz` unchanged to:

```
/home/rog/business/YFCE/cuda4AS/RESULTS/m2/incoming/
```

The repository-side analyzer will validate the package, inventory, patch,
fixtures, outputs, provenance, and all three case classifications. One
validated Apple-GPU pass is sufficient for the entry gate; otherwise the
preselected Native AOT Core v1 architecture route is recorded.

## M2A — MAC ACTION REQUIRED after portable gates

The M2A Native AOT Core v1 vector-add package is the small, separate artifact:

```
tools/m2a/artifacts/cuda4as-m2a-native-aot-vector-add-v1.tgz
```

Current package identity: 22,535 bytes, SHA-256
`1184651ff56d76fbea08fce7c68ca857615d1f27fe78934470d218313fbbe1c0`.

Pull the published `codex/m2a-native-aot-vector-add` branch on the Mac. The
tracked package is resolved relative to the tracked wrapper, so no separate
package copy, extraction, or package argument is needed:

```bash
git pull --ff-only
/bin/bash tools/m2a/run-m2a-native-aot-vector-add-all-in-one.sh
```

The wrapper verifies the tracked package identity, creates a fresh task
directory, performs safe extraction and package validation, runs the native
probe, logs the console, and copies the unchanged return archive to `~/Downloads`.

The runner is bound to arm64 MacBookPro18,3 / Apple M1 Pro, macOS 14.4 build
23E214, Xcode 15.2 / SDK 14.2, and Homebrew LLVM 23.1.0. It requires at least
5 GiB free, uses no more than four build jobs and 20 GiB task-local space, and
stops at 30 minutes. Stop if the package reports an inventory mismatch,
missing tool, compiler/AOT error, or resource limit. Return exactly the printed
`cuda4as-m2a-native-return-<UTC>.tgz` unchanged to
`RESULTS/m2a/incoming/` for repository-side analysis.

### Broken-checkout bootstrap

If the checkout was created from `codex/m2-entry-gate` and does not yet contain
the wrapper, stay in the existing checkout and acquire the tracked bootstrap,
wrapper, and package through Git. Pull does not switch branches; this sequence
uses the explicit task branch so it also works when the local target branch has
the old `codex/m2-entry-gate` upstream:

```bash
git pull --ff-only origin codex/m2a-native-aot-vector-add
/bin/bash tools/m2a/bootstrap-m2a-native-aot-probe.sh
```

The standalone bootstrap explicitly fetches the narrow branch ref, repairs an
existing wrong upstream, fast-forwards only, preserves local divergence, and
launches the tracked wrapper after verifying the package. It creates a fresh
bootstrap log directory outside the checkout.
