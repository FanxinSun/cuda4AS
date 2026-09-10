#!/bin/bash
# CUDA4AS M2A Native AOT vector_add: complete user-operated Mac workflow.
# Run with: /bin/bash /actual/path/run-m2a-native-aot-vector-add-all-in-one.sh
# Optional input override: /bin/bash /actual/path/run-m2a-native-aot-vector-add-all-in-one.sh /path/package.tgz
set -u

PACKAGE_NAME="cuda4as-m2a-native-aot-vector-add-v1.tgz"
PACKAGE_SHA="1184651ff56d76fbea08fce7c68ca857615d1f27fe78934470d218313fbbe1c0"
PACKAGE_BYTES="22535"
DEFAULT_ARCHIVE="$HOME/Downloads/$PACKAGE_NAME"
ARCHIVE="${1:-$DEFAULT_ARCHIVE}"
RUNS_ROOT="$HOME/cuda4as-m2a-native-aot-runs"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

printf 'CUDA4AS M2A Native AOT vector_add\n'
printf 'Input package: %s\n' "$ARCHIVE"

if [ "${CUDA4AS_M2A_DRY_RUN:-0}" != "1" ]; then
  OS_NAME="$(uname -s 2>/dev/null || true)"
  CPU_ARCH="$(uname -m 2>/dev/null || true)"
  [ "$OS_NAME" = "Darwin" ] || fail "target OS is $OS_NAME; run this on the Mac"
  [ "$CPU_ARCH" = "arm64" ] || fail "target architecture is $CPU_ARCH; expected arm64"
else
  printf 'DRY_RUN=1: target OS/architecture check bypassed for wrapper validation only\n'
fi

for REQUIRED_TOOL in bash python3 shasum tar wc tr awk tee cp mkdir mktemp; do
  command -v "$REQUIRED_TOOL" >/dev/null 2>&1 ||
    fail "required command is unavailable: $REQUIRED_TOOL"
done

[ -f "$ARCHIVE" ] || fail "input package does not exist: $ARCHIVE"
[ -r "$ARCHIVE" ] || fail "input package is not readable: $ARCHIVE"
[ ! -L "$ARCHIVE" ] || fail "input package must be a regular file, not a symlink"

ACTUAL_BYTES="$(wc -c <"$ARCHIVE" | tr -d '[:space:]')"
ACTUAL_SHA="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
printf 'Package bytes: %s\nPackage SHA-256: %s\n' "$ACTUAL_BYTES" "$ACTUAL_SHA"
[ "$ACTUAL_BYTES" = "$PACKAGE_BYTES" ] ||
  fail "package byte count mismatch: expected $PACKAGE_BYTES"
[ "$ACTUAL_SHA" = "$PACKAGE_SHA" ] ||
  fail "package SHA-256 mismatch: expected $PACKAGE_SHA"

mkdir -p "$RUNS_ROOT" || fail "cannot create run root: $RUNS_ROOT"
TASK_ROOT="$(mktemp -d "$RUNS_ROOT/run.XXXXXX")" ||
  fail "cannot create a fresh task directory"
EXTRACT_ROOT="$TASK_ROOT/extracted"
CONSOLE_LOG="$TASK_ROOT/runner.console.txt"
mkdir -p "$EXTRACT_ROOT" || fail "cannot create extraction directory"

printf 'Task root: %s\n' "$TASK_ROOT"

python3 - "$ARCHIVE" "$EXTRACT_ROOT" <<'PY'
from pathlib import Path, PurePosixPath
import os
import sys
import tarfile

archive = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)

with tarfile.open(archive, "r:gz") as tar:
    members = tar.getmembers()

    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise SystemExit("unsafe archive member: " + member.name)
        if member.issym() or member.islnk():
            raise SystemExit("links are not accepted: " + member.name)
        if not (member.isdir() or member.isfile()):
            raise SystemExit("unsupported archive member: " + member.name)

        destination = (root.joinpath(*name.parts)).resolve()
        if destination != root and root not in destination.parents:
            raise SystemExit("archive path escapes extraction root")

    for member in members:
        name = PurePosixPath(member.name)
        destination = root.joinpath(*name.parts)

        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        source = tar.extractfile(member)
        if source is None:
            raise SystemExit("cannot read archive member: " + member.name)

        with destination.open("wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        os.chmod(destination, member.mode & 0o7777)
PY
EXTRACT_EXIT=$?
[ "$EXTRACT_EXIT" -eq 0 ] ||
  fail "safe package extraction failed; evidence directory: $TASK_ROOT"

PACKAGE_ROOT="$EXTRACT_ROOT/cuda4as-m2a-native-aot-vector-add-v1"
[ -d "$PACKAGE_ROOT" ] || fail "expected package root is missing"
[ -f "$PACKAGE_ROOT/run-m2a-native.sh" ] || fail "package runner is missing"
[ -f "$PACKAGE_ROOT/PACKAGE-MANIFEST.sha256" ] || fail "package manifest is missing"

if [ "${CUDA4AS_M2A_DRY_RUN:-0}" = "1" ]; then
  printf 'DRY_RUN_COMPLETE=1\nExtracted package: %s\n' "$PACKAGE_ROOT"
  printf 'No Mac/GPU command was run and no PASS_GPU claim is made.\n'
  exit 77
fi

printf 'Starting native runner; full console is saved at %s\n' "$CONSOLE_LOG"
set +e
bash "$PACKAGE_ROOT/run-m2a-native.sh" 2>&1 | tee "$CONSOLE_LOG"
PIPE_STATUS=("${PIPESTATUS[@]}")
RUN_EXIT="${PIPE_STATUS[0]}"
TEE_EXIT="${PIPE_STATUS[1]}"
set -u

[ "$TEE_EXIT" -eq 0 ] ||
  printf 'WARNING: console log tee failed with exit %s\n' "$TEE_EXIT" >&2

RETURN_ARCHIVE=""
for CANDIDATE in "$PACKAGE_ROOT"/returns/cuda4as-m2a-native-return-*.tgz; do
  if [ -f "$CANDIDATE" ]; then
    RETURN_ARCHIVE="$CANDIDATE"
  fi
done

if [ -z "$RETURN_ARCHIVE" ]; then
  printf 'RUN_EXIT=%s\n' "$RUN_EXIT"
  printf 'RETURN_ARCHIVE=missing\n'
  printf 'EVIDENCE_DIR=%s\n' "$TASK_ROOT"
  exit "$RUN_EXIT"
fi

RETURN_NAME="${RETURN_ARCHIVE##*/}"
COLLECTED_ARCHIVE="$TASK_ROOT/$RETURN_NAME"
cp "$RETURN_ARCHIVE" "$COLLECTED_ARCHIVE" ||
  fail "could not collect return archive"

RETURN_BYTES="$(wc -c <"$RETURN_ARCHIVE" | tr -d '[:space:]')"
RETURN_SHA="$(shasum -a 256 "$RETURN_ARCHIVE" | awk '{print $1}')"
COLLECTED_SHA="$(shasum -a 256 "$COLLECTED_ARCHIVE" | awk '{print $1}')"
[ "$RETURN_SHA" = "$COLLECTED_SHA" ] ||
  fail "collected archive hash mismatch"

DOWNLOAD_COPY="$HOME/Downloads/$RETURN_NAME"
if [ "$RETURN_ARCHIVE" != "$DOWNLOAD_COPY" ]; then
  cp "$RETURN_ARCHIVE" "$DOWNLOAD_COPY"
fi

DOWNLOAD_STATUS="missing"
if [ -f "$DOWNLOAD_COPY" ]; then
  DOWNLOAD_SHA="$(shasum -a 256 "$DOWNLOAD_COPY" | awk '{print $1}')"
  if [ "$DOWNLOAD_SHA" = "$RETURN_SHA" ]; then
    DOWNLOAD_STATUS="verified"
  else
    DOWNLOAD_STATUS="hash-mismatch"
  fi
fi

printf '\nCUDA4AS M2A NATIVE AOT VECTOR_ADD COMPLETE\n'
printf 'RUN_EXIT=%s\n' "$RUN_EXIT"
printf 'RETURN_ARCHIVE=%s\n' "$RETURN_ARCHIVE"
printf 'COLLECTED_ARCHIVE=%s\n' "$COLLECTED_ARCHIVE"
printf 'DOWNLOAD_COPY=%s\n' "$DOWNLOAD_COPY"
printf 'DOWNLOAD_STATUS=%s\n' "$DOWNLOAD_STATUS"
printf 'RETURN_BYTES=%s\n' "$RETURN_BYTES"
printf 'RETURN_SHA256=%s\n' "$RETURN_SHA"
printf 'CONSOLE_LOG=%s\n' "$CONSOLE_LOG"
printf 'Return the archive at RETURN_ARCHIVE unchanged.\n'

# Preserve target runner result: 0=PASS_GPU, 1=failure, 77=environment gap.
exit "$RUN_EXIT"
