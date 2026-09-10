#!/bin/bash
# One-command user-operated M2 entry retry from a cuda4AS checkout.
# No network, installation, sudo, host modification, or second patch.

set -u
umask 077

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
DELTA_ARTIFACT="${CUDA4AS_M2_ARTIFACT:-${REPO_ROOT}/tools/m2/artifacts/cuda4as-m2-entry-retry-v1.tgz}"
BASE_ARTIFACT="${CUDA4AS_M1_ARTIFACT:-${REPO_ROOT}/tools/m1/artifacts/cuda4as-m1-native-feasibility-v1.tgz}"
TASK_ROOT="${CUDA4AS_M2_TASK_ROOT:-${HOME}/cuda4as-m2/entry-v1}"
START_UTC="$(/bin/date -u '+%Y%m%dT%H%M%SZ')"
EXPECTED_DELTA_SHA256="526d5d36e418e0d29fc321fbefad57b754608aa3f32cedfd3f52c8200091d31f"
EXPECTED_DELTA_BYTES=137652

if [ ! -f "${DELTA_ARTIFACT}" ] || [ ! -f "${BASE_ARTIFACT}" ]; then
  printf 'error: expected M1 base and M2 delta artifacts are missing\n' >&2
  printf '  base:  %s\n  delta: %s\n' "${BASE_ARTIFACT}" "${DELTA_ARTIFACT}" >&2
  exit 2
fi
delta_bytes="$(/usr/bin/wc -c <"${DELTA_ARTIFACT}" | /usr/bin/tr -d '[:space:]')"
delta_sha256="$(/usr/bin/shasum -a 256 "${DELTA_ARTIFACT}" | /usr/bin/awk '{print $1}')"
if [ "${delta_bytes}" != "${EXPECTED_DELTA_BYTES}" ] || [ "${delta_sha256}" != "${EXPECTED_DELTA_SHA256}" ]; then
  printf 'error: M2 delta identity mismatch\n' >&2
  printf 'expected bytes=%s sha256=%s\n' "${EXPECTED_DELTA_BYTES}" "${EXPECTED_DELTA_SHA256}" >&2
  printf 'observed bytes=%s sha256=%s\n' "${delta_bytes}" "${delta_sha256}" >&2
  exit 3
fi

DROP_ROOT="${TASK_ROOT}/drop/${START_UTC}"
mkdir -p "${DROP_ROOT}"
if ! /usr/bin/tar -xzf "${DELTA_ARTIFACT}" -C "${DROP_ROOT}"; then
  printf 'error: M2 delta extraction failed\n' >&2
  exit 4
fi
PACKAGE_ROOT="${DROP_ROOT}/cuda4as-m2-entry-retry-v1"
if [ ! -x "${PACKAGE_ROOT}/run-m2-entry-retry.sh" ]; then
  printf 'error: extracted M2 runner is missing\n' >&2
  exit 5
fi

printf '%s\n' "M2 delta verified: ${delta_bytes} bytes ${delta_sha256}"
printf '%s\n' "M1 base: ${BASE_ARTIFACT}"
cd "${PACKAGE_ROOT}" || exit 6
exec env \
  CUDA4AS_M1_ARTIFACT="${BASE_ARTIFACT}" \
  CUDA4AS_M2_TASK_ROOT="${TASK_ROOT}" \
  ./run-m2-entry-retry.sh
