#!/usr/bin/env bash
# Run the bounded cuda4AS M1 native feasibility probe from a Git checkout.
# The tracked archive is extracted into a separate task directory; the
# checkout itself is never used as a build or results directory.

set -euo pipefail
umask 077

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
ARTIFACT="${REPO_ROOT}/tools/m1/artifacts/cuda4as-m1-native-feasibility-v1.tgz"
EXPECTED="dbb390b4f470b8f286ccb65a2b8565a235e749d9122c16e8e92ade96e0099bc7"
PACKAGE_NAME="cuda4as-m1-native-feasibility-v1"
TASK_ROOT="${CUDA4AS_M1_TASK_ROOT:-${HOME}/cuda4as-m1/native-v1}"
PACKAGE_DIR="${TASK_ROOT}/${PACKAGE_NAME}"

if [[ ! -f "${ARTIFACT}" ]]; then
  printf 'error: tracked native-feasibility archive is missing: %s\n' "${ARTIFACT}" >&2
  exit 2
fi

printf '%s  %s\n' "${EXPECTED}" "${ARTIFACT}" | shasum -a 256 -c -

if [[ -e "${PACKAGE_DIR}" ]]; then
  printf 'error: task directory already exists: %s\n' "${PACKAGE_DIR}" >&2
  printf 'choose a fresh directory, for example:\n' >&2
  printf '  CUDA4AS_M1_TASK_ROOT="$HOME/cuda4as-m1/native-v2" %s\n' \
    "${REPO_ROOT}/tools/m1/run-native-probe.sh" >&2
  exit 2
fi

mkdir -p "${TASK_ROOT}"
tar -xzf "${ARTIFACT}" -C "${TASK_ROOT}"
cd "${PACKAGE_DIR}"
test -x ./run-native-feasibility.sh

set +e
./run-native-feasibility.sh | tee native-run.console.txt
pipeline_status=("${PIPESTATUS[@]}")
set -e

run_exit="${pipeline_status[0]:-1}"
tee_exit="${pipeline_status[1]:-1}"
if [[ "${tee_exit}" -ne 0 ]]; then
  printf 'error: could not write native-run.console.txt (tee exit %s)\n' \
    "${tee_exit}" >&2
  [[ "${run_exit}" -eq 0 ]] && run_exit=1
fi

RETURN_ARCHIVE="$(find "${PACKAGE_DIR}/returns" -maxdepth 1 -type f \
  -name 'cuda4as-m1-native-return-*.tgz' -print | sort | tail -n 1)"
if [[ -z "${RETURN_ARCHIVE}" ]]; then
  printf 'error: the probe did not produce a native return archive\n' >&2
  exit "${run_exit}"
fi

RETURN_HASH="$(shasum -a 256 "${RETURN_ARCHIVE}" | awk '{print $1}')"
RETURN_BYTES="$(wc -c <"${RETURN_ARCHIVE}" | tr -d '[:space:]')"
printf '\nRUN_EXIT=%s\nRETURN_ARCHIVE=%s\nRETURN_BYTES=%s\nRETURN_SHA256=%s\n' \
  "${run_exit}" "${RETURN_ARCHIVE}" "${RETURN_BYTES}" "${RETURN_HASH}"
printf 'Return this one archive unchanged for repository-side validation.\n'
exit "${run_exit}"
