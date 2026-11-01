#!/bin/bash
# cuda4AS M1 user-operated native Mac feasibility run.
# This script performs no download, installation, sudo, or OS/SDK change.

set -u
umask 077

SCHEMA="cuda4as-m1-native-return-v1"
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
START_UTC="$(/bin/date -u '+%Y%m%dT%H%M%SZ')"
RUN_DIR="${ROOT_DIR}/results/${START_UTC}"
WORK_DIR="${ROOT_DIR}/work/${START_UTC}"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${RUN_DIR}/outputs"
HASH_DIR="${RUN_DIR}/hashes"
ADAPTER_RESULT_DIR="${RUN_DIR}/adapter"
RETURN_DIR="${ROOT_DIR}/returns"
FACTS="${RUN_DIR}/facts.tsv"
EVENTS="${RUN_DIR}/case-stage-events.tsv"
ASSERTIONS="${RUN_DIR}/assertions.tsv"
COMMANDS="${RUN_DIR}/commands.txt"
GAPS="${RUN_DIR}/environment-gaps.txt"
SEQUENCE=0

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${HASH_DIR}" \
  "${ADAPTER_RESULT_DIR}" "${RETURN_DIR}" "${WORK_DIR}"
: >"${COMMANDS}"
: >"${GAPS}"
printf 'key\tvalue\n' >"${FACTS}"
printf 'sequence\tcase_id\tstage\tstatus\texit_code\tlog\tnote\n' >"${EVENTS}"
printf 'case_id\tassertion_id\tpassed\tevidence\n' >"${ASSERTIONS}"

one_line() {
  printf '%s' "$1" | /usr/bin/tr '\t\r\n' '   '
}

write_fact() {
  printf '%s\t%s\n' "$(one_line "$1")" "$(one_line "$2")" >>"${FACTS}"
}

write_event() {
  SEQUENCE=$((SEQUENCE + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${SEQUENCE}" "$(one_line "$1")" "$(one_line "$2")" \
    "$(one_line "$3")" "$(one_line "$4")" "$(one_line "$5")" \
    "$(one_line "$6")" >>"${EVENTS}"
}

write_assertion() {
  printf '%s\t%s\t%s\t%s\n' "$(one_line "$1")" "$(one_line "$2")" \
    "$(one_line "$3")" "$(one_line "$4")" >>"${ASSERTIONS}"
}

record_command() {
  command_id="$1"
  shift
  printf '%s' "${command_id}:" >>"${COMMANDS}"
  for command_arg in "$@"; do
    printf ' %q' "${command_arg}" >>"${COMMANDS}"
  done
  printf '\n' >>"${COMMANDS}"
}

run_logged() {
  command_id="$1"
  shift
  stdout_path="${LOG_DIR}/${command_id}.stdout.txt"
  stderr_path="${LOG_DIR}/${command_id}.stderr.txt"
  record_command "${command_id}" "$@"
  "$@" >"${stdout_path}" 2>"${stderr_path}"
  command_status=$?
  printf '%s' "${command_status}"
}

version_at_least() {
  /usr/bin/awk -v actual="$1" -v required="$2" 'BEGIN {
    split(actual, a, "."); split(required, r, ".");
    for (i = 1; i <= 4; ++i) {
      av = (a[i] == "" ? 0 : a[i] + 0); rv = (r[i] == "" ? 0 : r[i] + 0);
      if (av > rv) exit 0; if (av < rv) exit 1;
    }
    exit 0;
  }'
}

mark_initial_not_run() {
  case_id="$1"
  shift
  for stage_name in "$@"; do
    write_event "${case_id}" "${stage_name}" "NOT_RUN" "-" "-" \
      "initial state; later event for the same case/stage supersedes this one"
  done
}

finish_and_exit() {
  final_status="$1"
  write_fact "ended_utc" "$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if [ -d "${WORK_DIR}" ]; then
    work_kib="$(/usr/bin/du -sk "${WORK_DIR}" 2>/dev/null | /usr/bin/awk '{print $1}')"
    write_fact "work_directory_kib" "${work_kib:-unknown}"
  fi

  if [ -f "${ROOT_DIR}/fixtures/INPUTS.sha256" ]; then
    (
      cd "${ROOT_DIR}" || exit 1
      /usr/bin/shasum -a 256 -c fixtures/INPUTS.sha256
    ) >"${LOG_DIR}/fixture-inputs-after.stdout.txt" \
      2>"${LOG_DIR}/fixture-inputs-after.stderr.txt"
    inputs_after_status=$?
    write_assertion "_package" "fixture_inputs_unchanged_after_run" \
      "$([ "${inputs_after_status}" -eq 0 ] && echo true || echo false)" \
      "logs/fixture-inputs-after.stdout.txt; exit=${inputs_after_status}"
    if [ "${inputs_after_status}" -ne 0 ]; then
      final_status=1
    fi
  fi
  write_fact "runner_exit_code" "${final_status}"

  (
    cd "${RUN_DIR}" || exit 1
    /usr/bin/find . -type f ! -name MANIFEST.sha256 -print \
      | LC_ALL=C /usr/bin/sort \
      | /usr/bin/sed 's#^\./##' \
      | while IFS= read -r returned_file; do
          /usr/bin/shasum -a 256 "${returned_file}"
        done
  ) >"${RUN_DIR}/MANIFEST.sha256"
  manifest_status=$?
  if [ "${manifest_status}" -ne 0 ]; then
    echo "fatal: could not create return manifest" >&2
    exit 4
  fi

  return_name="cuda4as-m1-native-return-${START_UTC}.tgz"
  return_path="${RETURN_DIR}/${return_name}"
  /usr/bin/tar -czf "${return_path}" -C "${RUN_DIR}" .
  tar_status=$?
  if [ "${tar_status}" -ne 0 ]; then
    echo "fatal: could not create return archive" >&2
    exit 5
  fi
  return_hash="$(/usr/bin/shasum -a 256 "${return_path}" | /usr/bin/awk '{print $1}')"
  return_bytes="$(/usr/bin/wc -c <"${return_path}" | /usr/bin/tr -d '[:space:]')"

  echo
  echo "CUDA4AS M1 NATIVE EVIDENCE COLLECTION COMPLETE"
  echo "Runner exit: ${final_status} (0=all raw gates passed, 1=failure, 77=environment gap)"
  echo "Return archive: ${return_path}"
  echo "Bytes: ${return_bytes}"
  echo "SHA-256: ${return_hash}"
  echo "Return this one archive unchanged. A cuda4AS classification is assigned only after validation."
  exit "${final_status}"
}

mark_initial_not_run "oracle.vector_add" \
  host_compile device_compile native_link launch validation
mark_initial_not_run "integration.minimal_cmake_cuda" \
  configure host_compile device_compile native_link launch validation
mark_initial_not_run "integration.multi_tu_device_link" \
  configure host_compile device_compile device_link native_link launch validation

write_fact "schema" "${SCHEMA}"
write_fact "started_utc" "$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')"
write_fact "package_root" "${ROOT_DIR}"
write_fact "work_directory" "${WORK_DIR}"
write_fact "network_operations" "none"
write_fact "install_update_sudo_operations" "none"
write_fact "candidate_revision" "f486e5ebcfd381d06e3297afd65dbcbd5006a902"
write_fact "vf64_revision" "729021777455da72db8809d9ef1269c677d88b3f"
write_fact "build_type" "Release"
write_fact "CUMETAL_BUILD_TESTS" "OFF"
write_fact "CUMETAL_ENABLE_CUDA_REGISTRATION" "ON"
write_fact "CUMETAL_ENABLE_BINARY_SHIM" "OFF"
write_fact "CUMETAL_CUDA_ARCH" "sm_86"
write_fact "CUMETAL_FP64_MODE" "ieee64"

package_check_status="$(run_logged package_manifest \
  /bin/bash -c 'cd "$1" && /usr/bin/shasum -a 256 -c PACKAGE-MANIFEST.sha256' \
  _ "${ROOT_DIR}")"
write_event "_package" "integrity" \
  "$([ "${package_check_status}" -eq 0 ] && echo PASS || echo FAIL)" \
  "${package_check_status}" "logs/package_manifest.stdout.txt" \
  "complete static package manifest"
if [ "${package_check_status}" -ne 0 ]; then
  finish_and_exit 1
fi
mkdir -p "${RUN_DIR}/package"
/bin/cp "${ROOT_DIR}/target-inventory-binding.json" \
  "${RUN_DIR}/package/target-inventory-binding.json"
/bin/cp "${ROOT_DIR}/PACKAGE-MANIFEST.sha256" \
  "${RUN_DIR}/package/PACKAGE-MANIFEST.sha256"
if ! /usr/bin/grep -q '"status": "BOUND_TO_RETURNED_INVENTORY"' \
    "${ROOT_DIR}/target-inventory-binding.json"; then
  write_event "_package" "inventory_binding" "FAIL" "1" \
    "package/target-inventory-binding.json" \
    "provisional package is not authorized for native build/run"
  finish_and_exit 1
fi
write_event "_package" "inventory_binding" "PASS" "0" \
  "package/target-inventory-binding.json" "bound to a validated inventory return"

fixture_check_status="$(run_logged fixture_inputs_before \
  /bin/bash -c 'cd "$1" && /usr/bin/shasum -a 256 -c fixtures/INPUTS.sha256' \
  _ "${ROOT_DIR}")"
write_assertion "_package" "fixture_inputs_match_before_run" \
  "$([ "${fixture_check_status}" -eq 0 ] && echo true || echo false)" \
  "logs/fixture_inputs_before.stdout.txt; exit=${fixture_check_status}"
if [ "${fixture_check_status}" -ne 0 ]; then
  finish_and_exit 1
fi

# Read-only native machine snapshot.
uname_system="$(/usr/bin/uname -s 2>/dev/null || true)"
uname_machine="$(/usr/bin/uname -m 2>/dev/null || true)"
product_version="$(/usr/bin/sw_vers -productVersion 2>/dev/null || true)"
product_build="$(/usr/bin/sw_vers -buildVersion 2>/dev/null || true)"
developer_dir="$(/usr/bin/xcode-select -p 2>/dev/null || true)"
machine_model="$(/usr/sbin/sysctl -n hw.model 2>/dev/null || true)"
xcode_version="$(/usr/bin/xcodebuild -version 2>/dev/null \
  | /usr/bin/tr '\n' ';' || true)"
write_fact "uname_system" "${uname_system:-missing}"
write_fact "uname_machine" "${uname_machine:-missing}"
write_fact "macos_product_version" "${product_version:-missing}"
write_fact "macos_build" "${product_build:-missing}"
write_fact "developer_directory" "${developer_dir:-missing}"
write_fact "machine_model" "${machine_model:-missing}"
write_fact "xcode_version" "${xcode_version:-missing}"

run_logged snapshot_sw_vers /usr/bin/sw_vers >/dev/null
run_logged snapshot_uname /usr/bin/uname -a >/dev/null
run_logged snapshot_displays /usr/sbin/system_profiler SPDisplaysDataType -json -detailLevel mini >/dev/null
run_logged snapshot_xcode /usr/bin/xcodebuild -version >/dev/null
run_logged snapshot_sdks /usr/bin/xcodebuild -showsdks >/dev/null

PREFLIGHT_OK=1
if [ "${uname_system}" != "Darwin" ]; then
  echo "requires Darwin; observed ${uname_system:-missing}" >>"${GAPS}"
  PREFLIGHT_OK=0
fi
if [ "${uname_machine}" != "arm64" ]; then
  echo "requires arm64 Apple Silicon; observed ${uname_machine:-missing}" >>"${GAPS}"
  PREFLIGHT_OK=0
fi
if [ -z "${product_version}" ] || ! version_at_least "${product_version}" "14.0"; then
  echo "requires macOS 14 or newer; observed ${product_version:-missing}" >>"${GAPS}"
  PREFLIGHT_OK=0
fi

CMAKE_BIN="$(command -v cmake 2>/dev/null || true)"
NINJA_BIN="$(command -v ninja 2>/dev/null || true)"
if [ -z "${CMAKE_BIN}" ]; then
  echo "cmake missing (requires >=3.28)" >>"${GAPS}"
  PREFLIGHT_OK=0
else
  cmake_version="$(${CMAKE_BIN} --version 2>/dev/null | /usr/bin/awk 'NR==1 {print $3}')"
  write_fact "cmake_path" "${CMAKE_BIN}"
  write_fact "cmake_version" "${cmake_version:-unknown}"
  if [ -z "${cmake_version}" ] || ! version_at_least "${cmake_version}" "3.28"; then
    echo "cmake >=3.28 required; observed ${cmake_version:-unknown}" >>"${GAPS}"
    PREFLIGHT_OK=0
  fi
fi
if [ -z "${NINJA_BIN}" ]; then
  echo "ninja missing (required for explicit object/device-link stages)" >>"${GAPS}"
  PREFLIGHT_OK=0
else
  write_fact "ninja_path" "${NINJA_BIN}"
  write_fact "ninja_version" "$(${NINJA_BIN} --version 2>/dev/null || echo unknown)"
fi

LLVM_CONFIG=""
for llvm_candidate in \
  /opt/homebrew/opt/llvm/bin/llvm-config \
  /usr/local/opt/llvm/bin/llvm-config \
  "$(command -v llvm-config 2>/dev/null || true)"; do
  if [ -n "${llvm_candidate}" ] && [ -x "${llvm_candidate}" ]; then
    LLVM_CONFIG="${llvm_candidate}"
    break
  fi
done
if [ -z "${LLVM_CONFIG}" ]; then
  echo "llvm-config missing (requires LLVM >=18)" >>"${GAPS}"
  PREFLIGHT_OK=0
  LLVM_VERSION=""
  LLVM_PREFIX=""
  LLVM_DIR=""
  CLANGXX=""
  CLANG=""
else
  LLVM_VERSION="$(${LLVM_CONFIG} --version 2>/dev/null || true)"
  LLVM_PREFIX="$(${LLVM_CONFIG} --prefix 2>/dev/null || true)"
  LLVM_DIR="$(${LLVM_CONFIG} --cmakedir 2>/dev/null || true)"
  LLVM_BINDIR="$(${LLVM_CONFIG} --bindir 2>/dev/null || true)"
  CLANGXX="${LLVM_BINDIR}/clang++"
  CLANG="${LLVM_BINDIR}/clang"
  write_fact "llvm_config" "${LLVM_CONFIG}"
  write_fact "llvm_version" "${LLVM_VERSION:-unknown}"
  write_fact "llvm_prefix" "${LLVM_PREFIX:-unknown}"
  write_fact "llvm_cmakedir" "${LLVM_DIR:-unknown}"
  write_fact "clangxx" "${CLANGXX}"
  if [ -z "${LLVM_VERSION}" ] || ! version_at_least "${LLVM_VERSION}" "18.0"; then
    echo "LLVM >=18 required; observed ${LLVM_VERSION:-unknown}" >>"${GAPS}"
    PREFLIGHT_OK=0
  fi
  if [ ! -x "${CLANGXX}" ] || [ ! -x "${CLANG}" ]; then
    echo "CUDA-capable clang/clang++ missing beside llvm-config" >>"${GAPS}"
    PREFLIGHT_OK=0
  fi
fi

METAL_BIN="$(/usr/bin/xcrun --sdk macosx --find metal 2>/dev/null || true)"
METALLIB_BIN="$(/usr/bin/xcrun --sdk macosx --find metallib 2>/dev/null || true)"
SDK_PATH="$(/usr/bin/xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
SDK_VERSION="$(/usr/bin/xcrun --sdk macosx --show-sdk-version 2>/dev/null || true)"
write_fact "metal_compiler" "${METAL_BIN:-missing}"
write_fact "metallib_tool" "${METALLIB_BIN:-missing}"
write_fact "sdk_path" "${SDK_PATH:-missing}"
write_fact "sdk_version" "${SDK_VERSION:-missing}"
if [ -z "${METAL_BIN}" ] || [ -z "${METALLIB_BIN}" ] || [ -z "${SDK_PATH}" ]; then
  echo "xcrun macOS SDK Metal compiler/metallib tools missing" >>"${GAPS}"
  PREFLIGHT_OK=0
fi

BREW_BIN="$(command -v brew 2>/dev/null || true)"
PKG_CONFIG_BIN="$(command -v pkg-config 2>/dev/null || true)"
component_prefix() {
  component="$1"
  pkg_name="$2"
  found_prefix=""
  if [ -n "${BREW_BIN}" ]; then
    found_prefix="$(${BREW_BIN} --prefix "${component}" 2>/dev/null || true)"
  fi
  if [ -z "${found_prefix}" ] && [ -n "${PKG_CONFIG_BIN}" ]; then
    found_prefix="$(${PKG_CONFIG_BIN} --variable=prefix "${pkg_name}" 2>/dev/null || true)"
  fi
  printf '%s' "${found_prefix}"
}
LZ4_PREFIX="$(component_prefix lz4 liblz4)"
ZSTD_PREFIX="$(component_prefix zstd libzstd)"
write_fact "lz4_prefix" "${LZ4_PREFIX:-missing}"
write_fact "zstd_prefix" "${ZSTD_PREFIX:-missing}"

LZ4_INCLUDE="${LZ4_PREFIX}/include"
ZSTD_INCLUDE="${ZSTD_PREFIX}/include"
LZ4_LIBRARY="${LZ4_PREFIX}/lib/liblz4.dylib"
ZSTD_LIBRARY="${ZSTD_PREFIX}/lib/libzstd.dylib"
if [ -z "${LZ4_PREFIX}" ] || [ ! -f "${LZ4_INCLUDE}/lz4.h" ] || [ ! -e "${LZ4_LIBRARY}" ]; then
  echo "LZ4 headers/library missing" >>"${GAPS}"
  PREFLIGHT_OK=0
fi
if [ -z "${ZSTD_PREFIX}" ] || [ ! -f "${ZSTD_INCLUDE}/zstd.h" ] || [ ! -e "${ZSTD_LIBRARY}" ]; then
  echo "Zstd headers/library missing" >>"${GAPS}"
  PREFLIGHT_OK=0
fi

free_kib="$(/bin/df -Pk "${ROOT_DIR}" 2>/dev/null | /usr/bin/awk 'NR==2 {print $4}')"
write_fact "task_filesystem_free_kib" "${free_kib:-unknown}"
if [ -z "${free_kib}" ] || [ "${free_kib}" -lt 5242880 ] 2>/dev/null; then
  echo "at least 5 GiB free space required for the bounded build" >>"${GAPS}"
  PREFLIGHT_OK=0
fi

if [ "${PREFLIGHT_OK}" -ne 1 ]; then
  write_event "_environment" "preflight" "FAIL" "77" \
    "environment-gaps.txt" "no install or update attempted"
  finish_and_exit 77
fi
write_event "_environment" "preflight" "PASS" "0" "facts.tsv" \
  "all required existing native tools present"

hardware_cpus="$(/usr/sbin/sysctl -n hw.ncpu 2>/dev/null || echo 2)"
case "${hardware_cpus}" in
  ''|*[!0-9]*) JOBS=2 ;;
  *) if [ "${hardware_cpus}" -gt 4 ]; then JOBS=4; else JOBS="${hardware_cpus}"; fi ;;
esac
if [ "${JOBS}" -lt 1 ]; then JOBS=1; fi
write_fact "parallel_jobs" "${JOBS}"

# Reconstruct and verify the exact pinned candidate source without network use.
SOURCE_PARENT="${WORK_DIR}/src"
mkdir -p "${SOURCE_PARENT}"
extract_main_status="$(run_logged extract_cumetal \
  /usr/bin/tar -xzf "${ROOT_DIR}/candidate/cuda-metal-f486e5eb.tar.gz" \
  -C "${SOURCE_PARENT}")"
extract_vf_status="$(run_logged extract_vf64 \
  /usr/bin/tar -xzf "${ROOT_DIR}/candidate/vf64-metal-72902177.tar.gz" \
  -C "${SOURCE_PARENT}")"
CANDIDATE_SOURCE="${SOURCE_PARENT}/cuda-metal-f486e5ebcfd381d06e3297afd65dbcbd5006a902"
VF64_SOURCE="${SOURCE_PARENT}/VF64-metal-729021777455da72db8809d9ef1269c677d88b3f"
if [ "${extract_main_status}" -ne 0 ] || [ "${extract_vf_status}" -ne 0 ] || \
   [ ! -d "${CANDIDATE_SOURCE}" ] || [ ! -d "${VF64_SOURCE}" ]; then
  write_event "_candidate" "source_extract" "FAIL" "1" \
    "logs/extract_cumetal.stderr.txt" "pinned archive extraction failed"
  finish_and_exit 1
fi
mkdir -p "${CANDIDATE_SOURCE}/third_party/VF64-metal"
/bin/cp -R "${VF64_SOURCE}/." "${CANDIDATE_SOURCE}/third_party/VF64-metal/"
source_check_status="$(run_logged candidate_tree \
  /bin/bash -c 'cd "$1" && /usr/bin/shasum -a 256 -c "$2"' _ \
  "${CANDIDATE_SOURCE}" "${ROOT_DIR}/candidate/combined-tree.sha256")"
if [ "${source_check_status}" -ne 0 ]; then
  write_event "_candidate" "source_integrity" "FAIL" "${source_check_status}" \
    "logs/candidate_tree.stdout.txt" "combined main/submodule tree mismatch"
  finish_and_exit 1
fi
write_event "_candidate" "source_integrity" "PASS" "0" \
  "logs/candidate_tree.stdout.txt" "1,065 pinned files"

CANDIDATE_BUILD="${WORK_DIR}/build/cumetal"
mkdir -p "${CANDIDATE_BUILD}"
CMAKE_PREFIX_PATH_VALUE="${LLVM_PREFIX};${LZ4_PREFIX};${ZSTD_PREFIX}"
candidate_configure_status="$(run_logged candidate_configure \
  /usr/bin/env CUMETAL_CUDA_ARCH=sm_86 CUMETAL_FP64_MODE=ieee64 \
  "${CMAKE_BIN}" -S "${CANDIDATE_SOURCE}" -B "${CANDIDATE_BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CLANG}" \
  -DCMAKE_CXX_COMPILER="${CLANGXX}" \
  -DCMAKE_OBJCXX_COMPILER="${CLANGXX}" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH_VALUE}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DCUMETAL_LZ4_INCLUDE_DIR="${LZ4_INCLUDE}" \
  -DCUMETAL_LZ4_LIBRARY="${LZ4_LIBRARY}" \
  -DCUMETAL_ZSTD_INCLUDE_DIR="${ZSTD_INCLUDE}" \
  -DCUMETAL_ZSTD_LIBRARY="${ZSTD_LIBRARY}" \
  -DCUMETAL_BUILD_TESTS=OFF \
  -DCUMETAL_ENABLE_CUDA_REGISTRATION=ON \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF)"
write_event "_candidate" "configure" \
  "$([ "${candidate_configure_status}" -eq 0 ] && echo PASS || echo FAIL)" \
  "${candidate_configure_status}" "logs/candidate_configure.stdout.txt" \
  "Release source build; registration ON; binary shim OFF"
if [ "${candidate_configure_status}" -ne 0 ]; then
  finish_and_exit 1
fi

candidate_build_status="$(run_logged candidate_build \
  "${NINJA_BIN}" -C "${CANDIDATE_BUILD}" -j "${JOBS}" \
  cumetal_runtime cumetalc cumetal_ptxas_shim cumetal_fatbinary_shim)"
write_event "_candidate" "build" \
  "$([ "${candidate_build_status}" -eq 0 ] && echo PASS || echo FAIL)" \
  "${candidate_build_status}" "logs/candidate_build.stdout.txt" \
  "minimum source/compiler/runtime targets"
if [ "${candidate_build_status}" -eq 0 ] && \
   { [ ! -x "${CANDIDATE_BUILD}/cumetalc" ] || \
     [ ! -f "${CANDIDATE_BUILD}/libcumetal.dylib" ]; }; then
  write_event "_candidate" "build_artifacts" "FAIL" "1" \
    "logs/candidate_build.stdout.txt" "expected compiler/runtime artifact missing"
  finish_and_exit 1
fi
if [ "${candidate_build_status}" -ne 0 ]; then
  finish_and_exit 1
fi

if /usr/bin/find "${CANDIDATE_BUILD}" -name 'libcuda.dylib' -print | /usr/bin/grep . \
    >"${LOG_DIR}/candidate_libcuda_aliases.txt"; then
  write_assertion "_candidate" "binary_shim_absent" "false" \
    "logs/candidate_libcuda_aliases.txt"
  finish_and_exit 1
else
  write_assertion "_candidate" "binary_shim_absent" "true" \
    "no libcuda.dylib in candidate build"
fi
run_logged candidate_cumetalc_version "${CANDIDATE_BUILD}/cumetalc" --version >/dev/null
run_logged candidate_runtime_links /usr/bin/otool -L "${CANDIDATE_BUILD}/libcumetal.dylib" >/dev/null

validate_output() {
  case_id="$1"
  actual_path="$2"
  expected_path="$3"
  expected_bytes="$4"
  expected_hash="$5"
  if [ ! -f "${actual_path}" ]; then
    write_assertion "${case_id}" "output_present" "false" "${actual_path}"
    return 1
  fi
  actual_bytes="$(/usr/bin/wc -c <"${actual_path}" | /usr/bin/tr -d '[:space:]')"
  actual_hash="$(/usr/bin/shasum -a 256 "${actual_path}" | /usr/bin/awk '{print $1}')"
  printf '%s  %s\n' "${actual_hash}" "${actual_path}" >>"${HASH_DIR}/actual-outputs.sha256"
  write_assertion "${case_id}" "output_bytes" \
    "$([ "${actual_bytes}" = "${expected_bytes}" ] && echo true || echo false)" \
    "actual=${actual_bytes}; expected=${expected_bytes}"
  write_assertion "${case_id}" "output_sha256" \
    "$([ "${actual_hash}" = "${expected_hash}" ] && echo true || echo false)" \
    "actual=${actual_hash}; expected=${expected_hash}"
  /usr/bin/cmp "${actual_path}" "${expected_path}" \
    >"${LOG_DIR}/$(echo "${case_id}" | /usr/bin/tr '.' '_')_cmp.stdout.txt" \
    2>"${LOG_DIR}/$(echo "${case_id}" | /usr/bin/tr '.' '_')_cmp.stderr.txt"
  compare_status=$?
  write_assertion "${case_id}" "full_output_comparison" \
    "$([ "${compare_status}" -eq 0 ] && echo true || echo false)" \
    "cmp exit=${compare_status}"
  if [ "${actual_bytes}" = "${expected_bytes}" ] && \
     [ "${actual_hash}" = "${expected_hash}" ] && [ "${compare_status}" -eq 0 ]; then
    return 0
  fi
  return 1
}

validate_provenance() {
  case_id="$1"
  combined_log="$2"
  expected_source="$3"
  provenance_lines="${LOG_DIR}/$(echo "${case_id}" | /usr/bin/tr '.' '_').provenance.txt"
  grep 'CUMETAL_PROVENANCE' "${combined_log}" >"${provenance_lines}" 2>/dev/null || true
  gpu_ok=false
  quality_ok=false
  device_name_ok=false
  duration_ok=false
  source_ok=false
  fallback_absent=true
  if grep -E 'device=apple_gpu .*launch_success=true' "${provenance_lines}" >/dev/null 2>&1; then gpu_ok=true; fi
  if grep -E 'semantic_quality=exact' "${provenance_lines}" >/dev/null 2>&1; then quality_ok=true; fi
  if grep -E 'device_name="[^"]+"' "${provenance_lines}" >/dev/null 2>&1; then device_name_ok=true; fi
  if grep -E 'duration_ns=[0-9]+' "${provenance_lines}" >/dev/null 2>&1; then duration_ok=true; fi
  if grep -E "source=${expected_source}" "${provenance_lines}" >/dev/null 2>&1; then
    source_ok=true
  fi
  if grep -E 'source=(cpu_fallback|stub)|semantic_quality=(approximate|reduced_precision_fp64)' \
      "${provenance_lines}" >/dev/null 2>&1; then fallback_absent=false; fi
  write_assertion "${case_id}" "apple_gpu_launch" "${gpu_ok}" "${provenance_lines#${RUN_DIR}/}"
  write_assertion "${case_id}" "semantic_quality_exact" "${quality_ok}" "${provenance_lines#${RUN_DIR}/}"
  write_assertion "${case_id}" "inventoriable_device_name" "${device_name_ok}" "${provenance_lines#${RUN_DIR}/}"
  write_assertion "${case_id}" "completed_duration" "${duration_ok}" "${provenance_lines#${RUN_DIR}/}"
  write_assertion "${case_id}" "expected_lowering_source" "${source_ok}" "expected=${expected_source}"
  write_assertion "${case_id}" "fallback_stub_approximate_absent" "${fallback_absent}" "${provenance_lines#${RUN_DIR}/}"
  [ "${gpu_ok}" = true ] && [ "${quality_ok}" = true ] && \
    [ "${device_name_ok}" = true ] && [ "${duration_ok}" = true ] && \
    [ "${source_ok}" = true ] && [ "${fallback_absent}" = true ]
}

OVERALL=0

# Case 1: direct typed native-AOT compilation of the existing oracle source.
DIRECT_WORK="${WORK_DIR}/cases/oracle-vector-add"
DIRECT_CACHE="${DIRECT_WORK}/runtime-cache"
mkdir -p "${DIRECT_WORK}" "${DIRECT_CACHE}"
direct_compile_status="$(run_logged oracle_vector_add_compile \
  /usr/bin/env CUMETAL_CUDA_CLANG="${CLANGXX}" CUMETAL_FP64_MODE=ieee64 \
  "${CANDIDATE_BUILD}/cumetalc" \
  "${ROOT_DIR}/fixtures/oracle/vector_add.cu" \
  --backend=cumetal-ir --cuda-arch sm_86 --fp64=ieee64 \
  -o "${DIRECT_WORK}/vector_add")"
/bin/cat "${LOG_DIR}/oracle_vector_add_compile.stdout.txt" \
    "${LOG_DIR}/oracle_vector_add_compile.stderr.txt" \
    >"${LOG_DIR}/oracle_vector_add_compile.combined.txt"
if [ "${direct_compile_status}" -eq 0 ] && [ -x "${DIRECT_WORK}/vector_add" ]; then
  write_event "oracle.vector_add" "host_compile" "PASS" "0" \
    "logs/oracle_vector_add_compile.combined.txt" "integrated direct compiler"
  write_event "oracle.vector_add" "device_compile" "PASS" "0" \
    "logs/oracle_vector_add_compile.combined.txt" "typed cumetal-ir"
  write_event "oracle.vector_add" "native_link" "PASS" "0" \
    "logs/oracle_vector_add_compile.combined.txt" "native-AOT executable"
  direct_run_status="$(run_logged oracle_vector_add_run \
    /usr/bin/env CUMETAL_TRACE_GPU=1 CUMETAL_FP64_MODE=ieee64 \
    CUMETAL_CACHE_DIR="${DIRECT_CACHE}" \
    "${DIRECT_WORK}/vector_add" "${OUTPUT_DIR}/oracle-vector-add.bin")"
  cat "${LOG_DIR}/oracle_vector_add_run.stdout.txt" \
      "${LOG_DIR}/oracle_vector_add_run.stderr.txt" \
      >"${LOG_DIR}/oracle_vector_add_run.combined.txt"
  if [ "${direct_run_status}" -eq 0 ] && \
     validate_provenance "oracle.vector_add" \
       "${LOG_DIR}/oracle_vector_add_run.combined.txt" generic_nvvm; then
    write_event "oracle.vector_add" "launch" "PASS" "0" \
      "logs/oracle_vector_add_run.combined.txt" "positive completed Apple-GPU provenance"
  else
    write_event "oracle.vector_add" "launch" "FAIL" "${direct_run_status}" \
      "logs/oracle_vector_add_run.combined.txt" "execution or provenance gate failed"
    OVERALL=1
  fi
  if validate_output "oracle.vector_add" "${OUTPUT_DIR}/oracle-vector-add.bin" \
      "${ROOT_DIR}/fixtures/expected/oracle-vector-add.bin" 4194304 \
      ed551637cf393112d0093037a0b41b9d1e9bd213c6037e8efcde30cf480f0332; then
    write_event "oracle.vector_add" "validation" "PASS" "0" \
      "hashes/actual-outputs.sha256" "full exact comparison"
  else
    write_event "oracle.vector_add" "validation" "FAIL" "1" \
      "hashes/actual-outputs.sha256" "output mismatch or absent"
    OVERALL=1
  fi
  if [ -d "${DIRECT_CACHE}/registration-jit" ]; then
    write_assertion "oracle.vector_add" "registration_jit_absent" "false" \
      "${DIRECT_CACHE}/registration-jit"
    OVERALL=1
  else
    write_assertion "oracle.vector_add" "registration_jit_absent" "true" \
      "native-AOT route did not create registration-jit"
  fi
  if [ -d "${DIRECT_CACHE}/native-aot" ]; then
    write_assertion "oracle.vector_add" "native_aot_cache_present" "true" \
      "${DIRECT_CACHE}/native-aot"
  else
    write_assertion "oracle.vector_add" "native_aot_cache_present" "false" \
      "${DIRECT_CACHE}/native-aot"
    OVERALL=1
  fi
else
  write_event "oracle.vector_add" "device_compile" "FAIL" \
    "${direct_compile_status}" "logs/oracle_vector_add_compile.combined.txt" \
    "integrated direct compile/link failed"
  OVERALL=1
fi

# Prepare the source-only CMake compiler adapter. It is outside candidate and
# fixture trees and deliberately omits the upstream task-local libcuda alias.
ADAPTER_WORK="${WORK_DIR}/adapter"
adapter_status="$(run_logged prepare_cmake_adapter /bin/bash \
  "${ROOT_DIR}/tools/prepare-cmake-adapter.sh" \
  "${CANDIDATE_SOURCE}" "${CANDIDATE_BUILD}" "${ADAPTER_WORK}" "${CLANGXX}")"
if [ -f "${ADAPTER_WORK}/source-toolkit-adapter.diff" ]; then
  /bin/cp "${ADAPTER_WORK}/source-toolkit-adapter.diff" "${ADAPTER_RESULT_DIR}/"
fi
if [ -f "${ADAPTER_WORK}/script-hashes.sha256" ]; then
  /bin/cp "${ADAPTER_WORK}/script-hashes.sha256" "${ADAPTER_RESULT_DIR}/"
fi
if [ -f "${ADAPTER_WORK}/toolkit-files.txt" ]; then
  /bin/cp "${ADAPTER_WORK}/toolkit-files.txt" "${ADAPTER_RESULT_DIR}/"
fi
TOOLKIT="${CANDIDATE_BUILD}/cumetal-cuda-toolkit"
if [ "${adapter_status}" -ne 0 ] || [ ! -x "${TOOLKIT}/bin/nvcc" ]; then
  write_event "_adapter" "prepare" "FAIL" "${adapter_status}" \
    "logs/prepare_cmake_adapter.stderr.txt" "CMake cases remain NOT_RUN"
  finish_and_exit 1
fi
write_event "_adapter" "prepare" "PASS" "0" \
  "adapter/source-toolkit-adapter.diff" "source-only adapter; no libcuda alias"

run_cmake_case() {
  case_id="$1"
  fixture_name="$2"
  target_name="$3"
  expected_name="$4"
  expected_bytes="$5"
  expected_hash="$6"
  has_device_link="$7"
  case_slug="$(echo "${case_id}" | /usr/bin/tr '.' '_')"
  case_build="${WORK_DIR}/cases/${fixture_name}-build"
  case_cache="${WORK_DIR}/cases/${fixture_name}-cache"
  mkdir -p "${case_build}" "${case_cache}"

  configure_status="$(run_logged "${case_slug}_configure" \
    /usr/bin/env PATH="${TOOLKIT}/bin:${CANDIDATE_SOURCE}/scripts/cuda_toolchain:${PATH}" \
    CUMETAL_CUDA_ARCH=sm_86 CUMETAL_FP64_MODE=ieee64 \
    "${CMAKE_BIN}" -S "${ROOT_DIR}/fixtures/${fixture_name}" -B "${case_build}" \
    -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${CLANGXX}" \
    -DCMAKE_CUDA_HOST_COMPILER="${CLANGXX}" \
    -DCMAKE_CUDA_COMPILER="${TOOLKIT}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER_LIBRARY_ROOT="${TOOLKIT}" \
    -DCUDAToolkit_ROOT="${TOOLKIT}" \
    -DCMAKE_BUILD_RPATH="${CANDIDATE_BUILD}")"
  /bin/cat "${LOG_DIR}/${case_slug}_configure.stdout.txt" \
      "${LOG_DIR}/${case_slug}_configure.stderr.txt" \
      >"${LOG_DIR}/${case_slug}_configure.combined.txt"
  if [ "${configure_status}" -ne 0 ]; then
    write_event "${case_id}" "configure" "FAIL" "${configure_status}" \
      "logs/${case_slug}_configure.combined.txt" "unchanged CMake configure failed"
    OVERALL=1
    return
  fi
  write_event "${case_id}" "configure" "PASS" "0" \
    "logs/${case_slug}_configure.combined.txt" "unchanged CMake project"

  if [ "${has_device_link}" = true ]; then
    object_targets="CMakeFiles/${target_name}.dir/main.cu.o CMakeFiles/${target_name}.dir/kernel.cu.o CMakeFiles/${target_name}.dir/device-function.cu.o"
  else
    object_targets="CMakeFiles/${target_name}.dir/main.cu.o"
  fi
  # Intentional word splitting: Ninja receives the pinned list of target paths.
  # shellcheck disable=SC2086
  object_status="$(run_logged "${case_slug}_objects" \
    "${NINJA_BIN}" -C "${case_build}" -v ${object_targets})"
  /bin/cat "${LOG_DIR}/${case_slug}_objects.stdout.txt" \
      "${LOG_DIR}/${case_slug}_objects.stderr.txt" \
      >"${LOG_DIR}/${case_slug}_objects.combined.txt"
  if [ "${object_status}" -ne 0 ]; then
    write_event "${case_id}" "device_compile" "FAIL" "${object_status}" \
      "logs/${case_slug}_objects.combined.txt" "CUDA object build failed"
    OVERALL=1
    return
  fi
  write_event "${case_id}" "host_compile" "PASS" "0" \
    "logs/${case_slug}_objects.combined.txt" "combined Clang CUDA object build"
  write_event "${case_id}" "device_compile" "PASS" "0" \
    "logs/${case_slug}_objects.combined.txt" "combined Clang CUDA object build"

  if [ "${has_device_link}" = true ]; then
    dlink_target="CMakeFiles/${target_name}.dir/cmake_device_link.o"
    dlink_status="$(run_logged "${case_slug}_device_link" \
      "${NINJA_BIN}" -C "${case_build}" -v "${dlink_target}")"
    /bin/cat "${LOG_DIR}/${case_slug}_device_link.stdout.txt" \
        "${LOG_DIR}/${case_slug}_device_link.stderr.txt" \
        >"${LOG_DIR}/${case_slug}_device_link.combined.txt"
    if [ "${dlink_status}" -ne 0 ]; then
      write_event "${case_id}" "device_link" "FAIL" "${dlink_status}" \
        "logs/${case_slug}_device_link.combined.txt" "required device-link command failed"
      OVERALL=1
      return
    fi
    write_event "${case_id}" "device_link" "PASS" "0" \
      "logs/${case_slug}_device_link.combined.txt" \
      "upstream shim emits an empty host object; runtime outcome still required"
  fi

  link_status="$(run_logged "${case_slug}_native_link" \
    "${NINJA_BIN}" -C "${case_build}" -v "${target_name}")"
  /bin/cat "${LOG_DIR}/${case_slug}_native_link.stdout.txt" \
      "${LOG_DIR}/${case_slug}_native_link.stderr.txt" \
      >"${LOG_DIR}/${case_slug}_native_link.combined.txt"
  if [ "${link_status}" -ne 0 ] || [ ! -x "${case_build}/${target_name}" ]; then
    write_event "${case_id}" "native_link" "FAIL" "${link_status}" \
      "logs/${case_slug}_native_link.combined.txt" "native executable link failed"
    OVERALL=1
    return
  fi
  write_event "${case_id}" "native_link" "PASS" "0" \
    "logs/${case_slug}_native_link.combined.txt" "native arm64 executable"

  run_status="$(run_logged "${case_slug}_run" \
    /usr/bin/env CUMETAL_TRACE_GPU=1 CUMETAL_FP64_MODE=ieee64 \
    CUMETAL_CACHE_DIR="${case_cache}" \
    "${case_build}/${target_name}" "${OUTPUT_DIR}/${expected_name}")"
  cat "${LOG_DIR}/${case_slug}_run.stdout.txt" \
      "${LOG_DIR}/${case_slug}_run.stderr.txt" \
      >"${LOG_DIR}/${case_slug}_run.combined.txt"
  if [ "${run_status}" -eq 0 ] && \
     validate_provenance "${case_id}" "${LOG_DIR}/${case_slug}_run.combined.txt" generic_ptx; then
    write_event "${case_id}" "launch" "PASS" "0" \
      "logs/${case_slug}_run.combined.txt" "positive completed Apple-GPU registration/PTX provenance"
  else
    write_event "${case_id}" "launch" "FAIL" "${run_status}" \
      "logs/${case_slug}_run.combined.txt" "execution or provenance gate failed"
    OVERALL=1
  fi
  if validate_output "${case_id}" "${OUTPUT_DIR}/${expected_name}" \
      "${ROOT_DIR}/fixtures/expected/${expected_name}" \
      "${expected_bytes}" "${expected_hash}"; then
    write_event "${case_id}" "validation" "PASS" "0" \
      "hashes/actual-outputs.sha256" "full exact comparison"
  else
    write_event "${case_id}" "validation" "FAIL" "1" \
      "hashes/actual-outputs.sha256" "output mismatch or absent"
    OVERALL=1
  fi
}

run_cmake_case "integration.minimal_cmake_cuda" "cmake-vector-add" \
  "cuda4as_m1_cmake_vector_add" "cmake-vector-add.bin" 16384 \
  8fe6be55ab663bfedbdd6435befcae6ffc5ab6f78059f2d20cd6ad50a417f4ef false
run_cmake_case "integration.multi_tu_device_link" "cmake-device-link" \
  "cuda4as_m1_cmake_device_link" "cmake-device-link.bin" 8192 \
  d173cf4cf4849f14508f67d6f7ec3f3f7080eac5305785b9f071e9feeb6269a1 true

finish_and_exit "${OVERALL}"
