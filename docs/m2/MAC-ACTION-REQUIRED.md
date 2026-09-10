# MAC ACTION REQUIRED — CUDA4AS M2 ENTRY RETRY IS READY

The executor prepared a 137,652-byte delta package. It reuses the existing
M1 package and contains no candidate or VF64 source archive. Verify:

```
cuda4as-m2-entry-retry-v1.tgz
526d5d36e418e0d29fc321fbefad57b754608aa3f32cedfd3f52c8200091d31f
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
