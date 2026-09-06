# MAC ACTION REQUIRED — CUDA4AS M1 IS READY FOR YOUR MAC RUN

> **Inventory checkpoint, 2026-09-06:** the returned inventory archive was
> internally valid and enumerated one Apple M1 Pro Metal device. The only native
> prerequisite gap is LLVM; no native build/GPU action is issued until the user
> decides on the separately described Homebrew LLVM installation. The absent
> system-default Metal device is retained as an observation, not replaced with
> the enumerated device.

## Action 1: read-only Mac inventory

This is the first M1 inventory run. It records the current Apple chip and Metal
devices, OS build, selected developer directory, full-Xcode versus
command-line-tools selection, SDK and Metal tools, CMake/Ninja/LLVM and
compression-library availability, and free disk space. It does not build
CuMetal or cuda4AS, submit GPU work, use the network, install anything, invoke
`sudo`, or change the OS/SDK.

### Artifact

- WSL path:
  `/home/rog/business/YFCE/cuda4AS/dist/m1/cuda4as-m1-mac-inventory-v1.tgz`
- Windows source:
  `\\wsl.localhost\Ubuntu\home\rog\business\YFCE\cuda4AS\dist\m1\cuda4as-m1-mac-inventory-v1.tgz`
- Size: 4,230 bytes.
- SHA-256:
  `febe1d8d7d22b591236f703b4a84555b8884bf329f1ead0612eac941790a8213`.
- Manifest: [`mac-inventory-artifact.json`](mac-inventory-artifact.json).

If it helps with transfer, copy it from WSL to Windows Downloads in PowerShell:

```powershell
Copy-Item "\\wsl.localhost\Ubuntu\home\rog\business\YFCE\cuda4AS\dist\m1\cuda4as-m1-mac-inventory-v1.tgz" "$HOME\Downloads\cuda4as-m1-mac-inventory-v1.tgz"
```

Transfer that one file to the Mac and place it at
`~/Downloads/cuda4as-m1-mac-inventory-v1.tgz`. AirDrop, a shared folder, USB,
or another user-controlled method is fine; the Mac has no direct dependency on
the WSL path.

### Git checkout route

If the Mac will pull the M1 branch directly, the checked-out inventory source
can be run without transferring the generated archive. Copy it to a separate
task directory first so the returned archive and temporary Swift output do not
alter the Git checkout:

```bash
set -euo pipefail

REPO="/absolute/path/to/cuda4AS"
TASK_ROOT="$HOME/cuda4as-m1"
RUN_DIR="$TASK_ROOT/mac-inventory-git-$(date -u +%Y%m%dT%H%M%SZ)"

test -x "$REPO/tools/m1/mac-inventory/run-inventory.sh"
mkdir -p "$RUN_DIR"
cp -R "$REPO/tools/m1/mac-inventory/." "$RUN_DIR/"
cd "$RUN_DIR"
./run-inventory.sh | tee inventory-run.console.txt

RETURN_ARCHIVE="$(find "$PWD/returns" -maxdepth 1 -type f -name 'cuda4as-m1-mac-inventory-return-*.tgz' -print | sort | tail -n 1)"
test -n "$RETURN_ARCHIVE"
shasum -a 256 "$RETURN_ARCHIVE"
printf 'RETURN THIS FILE: %s\n' "$RETURN_ARCHIVE"
```

Set `REPO` to the pulled checkout. This route runs the same three versioned
inventory files as the archive route; it does not require Python, a package
manager, or an installation.

### Exact Mac commands

Open Terminal on the Mac and run this block exactly:

```bash
set -euo pipefail

ARTIFACT="$HOME/Downloads/cuda4as-m1-mac-inventory-v1.tgz"
EXPECTED="febe1d8d7d22b591236f703b4a84555b8884bf329f1ead0612eac941790a8213"
TASK_ROOT="$HOME/cuda4as-m1"

printf '%s  %s\n' "$EXPECTED" "$ARTIFACT" | shasum -a 256 -c -
mkdir -p "$TASK_ROOT"
tar -xzf "$ARTIFACT" -C "$TASK_ROOT"
cd "$TASK_ROOT/cuda4as-m1-mac-inventory-v1"
./run-inventory.sh | tee inventory-run.console.txt

RETURN_ARCHIVE="$(find "$PWD/returns" -maxdepth 1 -type f -name 'cuda4as-m1-mac-inventory-return-*.tgz' -print | sort | tail -n 1)"
test -n "$RETURN_ARCHIVE"
shasum -a 256 "$RETURN_ARCHIVE"
printf 'RETURN THIS FILE: %s\n' "$RETURN_ARCHIVE"
```

The first checksum must print `OK`. The script then prints
`CUDA4AS M1 MAC INVENTORY COMPLETE`, the return archive path, its exact byte
count, and its SHA-256.

### Time and resource limits

- Expected duration: 1–3 minutes; this has not yet been timed on the target Mac.
- Network use: none.
- GPU workload: none. Device enumeration only.
- Disk: 4,230-byte input; normally well below 250 MB of transient output,
  including the optional Swift module cache; the returned archive should be
  much smaller. If the command runs for more than 10 minutes or the task
  directory exceeds 300 MB, interrupt it with Control-C and report that fact.
- Existing prerequisites only: standard macOS shell tools. If `xcrun`, Xcode,
  CMake, Ninja, LLVM, Homebrew, MacPorts, or compression libraries are missing,
  the script records `missing`; do not install them for this inventory.

### Return path

Transfer the one generated `.tgz` back to Windows without extracting or editing
it. Then use PowerShell to place it in the prepared WSL incoming directory,
substituting the actual UTC filename:

```powershell
$Return = "$HOME\Downloads\cuda4as-m1-mac-inventory-return-<UTC>.tgz"
Copy-Item $Return "\\wsl.localhost\Ubuntu\home\rog\business\YFCE\cuda4AS\RESULTS\m1\incoming\"
```

The intended WSL destination is:

```text
/home/rog/business/YFCE/cuda4AS/RESULTS/m1/incoming/cuda4as-m1-mac-inventory-return-<UTC>.tgz
```

Reply with the returned filename and SHA-256 after copying it. The executor
will validate the archive and its internal manifest before using any facts.

### Result handling

- A successful inventory run creates the archive even when optional tools are
  missing. Each check is labeled `ok`, `missing`, `exit_<code>`, or `not_run`.
- Missing prerequisites stay recorded as missing and will inform a concrete
  dependency proposal; they are not installed and do not become compatibility
  failures.
- If the script exits before producing an archive, keep the extracted task
  directory and report the command and terminal error. Do not retry by
  installing or upgrading anything.
- This return is required before selecting the native dependency route and
  issuing the build/GPU feasibility drop. Independent candidate inspection,
  fixtures, result-schema work, and PyTorch platform audit continue locally.
