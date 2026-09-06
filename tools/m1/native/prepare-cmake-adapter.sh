#!/bin/bash
# Prepare CuMetal's task-local CMake CUDA compiler adapter without exposing its
# optional libcuda binary alias. This changes no candidate or application file.

set -eu
umask 077

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <candidate-source> <candidate-build> <adapter-work> <clang++>" >&2
  exit 2
fi

CANDIDATE_SOURCE="$(CDPATH= cd -- "$1" && pwd -P)"
CANDIDATE_BUILD="$(CDPATH= cd -- "$2" && pwd -P)"
mkdir -p "$3"
ADAPTER_WORK="$(CDPATH= cd -- "$3" && pwd -P)"
CLANGXX="$4"
UPSTREAM_SCRIPT="${CANDIDATE_SOURCE}/scripts/build_llama_cpp_cumetal.sh"
VIEW_ROOT="${ADAPTER_WORK}/source-view"
VIEW_SCRIPTS="${VIEW_ROOT}/scripts"
PATCHED_SCRIPT="${VIEW_SCRIPTS}/build_llama_cpp_cumetal.sh"

test -f "${UPSTREAM_SCRIPT}"
test -f "${CANDIDATE_SOURCE}/scripts/cumetal_cuda_flags.sh"
test -d "${CANDIDATE_SOURCE}/scripts/cuda_toolchain"
test -d "${CANDIDATE_SOURCE}/runtime"
test -f "${CANDIDATE_BUILD}/libcumetal.dylib"
test -x "${CLANGXX}"
test ! -e "${VIEW_ROOT}"

mkdir -p "${VIEW_SCRIPTS}"
ln -s "${CANDIDATE_SOURCE}/runtime" "${VIEW_ROOT}/runtime"
ln -s "${CANDIDATE_SOURCE}/scripts/cumetal_cuda_flags.sh" \
  "${VIEW_SCRIPTS}/cumetal_cuda_flags.sh"
ln -s "${CANDIDATE_SOURCE}/scripts/cuda_toolchain" \
  "${VIEW_SCRIPTS}/cuda_toolchain"

# Keep the upstream source-compiler adapter byte-for-byte except for the two
# binary-driver-alias provisions. The exact diff and both hashes are retained.
awk '
  /FAKE_CUDA.*libcuda[.]dylib/ { next }
  /string\(REPLACE "cuda_driver"/ { next }
  {
    sub(/cudart cudart_static cuda_driver /, "cudart cudart_static ")
    print
  }
' "${UPSTREAM_SCRIPT}" >"${PATCHED_SCRIPT}"
chmod 0700 "${PATCHED_SCRIPT}"

if grep -E 'ln .*libcuda[.]dylib|foreach\([^)]*cuda_driver|string\(REPLACE "cuda_driver"' \
    "${PATCHED_SCRIPT}" >/dev/null 2>&1; then
  echo "adapter sanitization failed: active libcuda/driver mapping remains" >&2
  exit 3
fi

shasum -a 256 "${UPSTREAM_SCRIPT}" "${PATCHED_SCRIPT}" \
  >"${ADAPTER_WORK}/script-hashes.sha256"
diff_status=0
diff -u "${UPSTREAM_SCRIPT}" "${PATCHED_SCRIPT}" \
  >"${ADAPTER_WORK}/source-toolkit-adapter.diff" || diff_status=$?
if [ "${diff_status}" -ne 1 ]; then
  echo "unexpected diff exit ${diff_status}" >&2
  exit 4
fi

CUMETAL_BUILD_DIR="${CANDIDATE_BUILD}" \
CUMETAL_CLANG="${CLANGXX}" \
CUMETAL_CUDA_ARCH="sm_86" \
  /bin/bash "${PATCHED_SCRIPT}" --toolkit-only

TOOLKIT="${CANDIDATE_BUILD}/cumetal-cuda-toolkit"
test -x "${TOOLKIT}/bin/nvcc"
test -e "${TOOLKIT}/lib/libcudart.dylib"
if find "${TOOLKIT}" -name 'libcuda.dylib' -print | grep . >/dev/null 2>&1; then
  echo "adapter produced forbidden libcuda.dylib alias" >&2
  exit 5
fi

find "${TOOLKIT}" -type f -o -type l | LC_ALL=C sort \
  >"${ADAPTER_WORK}/toolkit-files.txt"
echo "M1 source-only CMake adapter ready: ${TOOLKIT}"
