#!/bin/bash
# cuda4AS M1 native-Mac inventory. This script only reads machine/tool state
# and writes beneath the directory containing this script.

set -u
umask 077

SCHEMA="cuda4as-m1-mac-inventory-v1"
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
START_UTC="$(/bin/date -u '+%Y%m%dT%H%M%SZ')"
RUN_DIR="${ROOT_DIR}/results/${START_UTC}"
LOG_DIR="${RUN_DIR}/logs"
TMP_DIR="${RUN_DIR}/tmp"
RETURN_DIR="${ROOT_DIR}/returns"
FACTS="${RUN_DIR}/facts.tsv"
STATUS="${RUN_DIR}/status.tsv"

if ! /bin/mkdir -p "${LOG_DIR}" "${TMP_DIR}" "${RETURN_DIR}"; then
  printf '%s\n' "fatal: cannot create inventory output under ${ROOT_DIR}" >&2
  exit 2
fi

printf 'key\tvalue\n' >"${FACTS}"
printf 'check_id\tstatus\texit_code\tcommand\n' >"${STATUS}"

one_line() {
  printf '%s' "$1" | /usr/bin/tr '\t\r\n' '   '
}

write_fact() {
  key="$(one_line "$1")"
  value="$(one_line "$2")"
  printf '%s\t%s\n' "${key}" "${value}" >>"${FACTS}"
}

write_status() {
  check_id="$(one_line "$1")"
  state="$(one_line "$2")"
  exit_code="$(one_line "$3")"
  command_text="$(one_line "$4")"
  printf '%s\t%s\t%s\t%s\n' \
    "${check_id}" "${state}" "${exit_code}" "${command_text}" >>"${STATUS}"
}

run_command() {
  check_id="$1"
  shift
  executable="$1"
  stdout_path="${LOG_DIR}/${check_id}.stdout.txt"
  stderr_path="${LOG_DIR}/${check_id}.stderr.txt"
  command_text="$*"

  if ! command -v "${executable}" >/dev/null 2>&1; then
    : >"${stdout_path}"
    printf 'missing executable: %s\n' "${executable}" >"${stderr_path}"
    write_status "${check_id}" "missing" "127" "${command_text}"
    return 0
  fi

  "$@" >"${stdout_path}" 2>"${stderr_path}"
  rc=$?
  if [ "${rc}" -eq 0 ]; then
    write_status "${check_id}" "ok" "0" "${command_text}"
  else
    write_status "${check_id}" "exit_${rc}" "${rc}" "${command_text}"
  fi
  return 0
}

write_fact "schema" "${SCHEMA}"
write_fact "started_utc" "${START_UTC}"
write_fact "inventory_root" "${ROOT_DIR}"
write_fact "run_directory" "${RUN_DIR}"
write_fact "mutating_operations" "none_requested; no sudo, install, update, or OS change"
write_fact "device_probe_purpose" "enumerate Metal devices and default selection; no kernel is compiled or launched"

run_command date_utc /bin/date -u '+%Y-%m-%dT%H:%M:%SZ'
run_command sw_vers /usr/bin/sw_vers
run_command uname /usr/bin/uname -a
run_command architecture /usr/bin/arch
run_command model /usr/sbin/sysctl -n hw.model
run_command cpu_brand /usr/sbin/sysctl -n machdep.cpu.brand_string
run_command memory_bytes /usr/sbin/sysctl -n hw.memsize
run_command physical_cpu /usr/sbin/sysctl -n hw.physicalcpu
run_command logical_cpu /usr/sbin/sysctl -n hw.logicalcpu
run_command displays /usr/sbin/system_profiler SPDisplaysDataType -json -detailLevel mini
run_command task_disk_free /bin/df -Pk "${ROOT_DIR}"
run_command root_disk_free /bin/df -Pk /

run_command xcode_select /usr/bin/xcode-select -p
selected_developer_dir="$(/usr/bin/xcode-select -p 2>/dev/null || true)"
write_fact "selected_developer_directory" "${selected_developer_dir:-missing}"
case "${selected_developer_dir}" in
  *.app/Contents/Developer) developer_kind="full_xcode" ;;
  */CommandLineTools) developer_kind="command_line_tools" ;;
  "") developer_kind="missing" ;;
  *) developer_kind="other" ;;
esac
write_fact "selected_developer_kind" "${developer_kind}"
if [ -d /Applications/Xcode.app/Contents/Developer ]; then
  write_fact "applications_xcode_present" "true"
else
  write_fact "applications_xcode_present" "false"
fi

run_command xcodebuild_version /usr/bin/xcodebuild -version
run_command xcodebuild_sdks /usr/bin/xcodebuild -showsdks
run_command sdk_path /usr/bin/xcrun --sdk macosx --show-sdk-path
run_command sdk_version /usr/bin/xcrun --sdk macosx --show-sdk-version
run_command sdk_build_version /usr/bin/xcrun --sdk macosx --show-sdk-build-version
run_command find_metal /usr/bin/xcrun --sdk macosx --find metal
run_command metal_version /usr/bin/xcrun --sdk macosx metal --version
run_command find_metallib /usr/bin/xcrun --sdk macosx --find metallib
run_command find_metal_ar /usr/bin/xcrun --sdk macosx --find metal-ar
run_command find_swiftc /usr/bin/xcrun --find swiftc
run_command swiftc_version /usr/bin/xcrun swiftc --version
run_command find_clang /usr/bin/xcrun --find clang
run_command clang_version /usr/bin/xcrun clang --version

run_command git_version git --version
run_command cmake_path /usr/bin/which cmake
run_command cmake_version cmake --version
run_command ninja_path /usr/bin/which ninja
run_command ninja_version ninja --version
run_command llvm_config_path /usr/bin/which llvm-config
run_command llvm_config_version llvm-config --version
run_command llvm_config_prefix llvm-config --prefix
run_command llvm_config_cmakedir llvm-config --cmakedir
run_command homebrew_llvm_version /opt/homebrew/opt/llvm/bin/llvm-config --version
run_command intel_homebrew_llvm_version /usr/local/opt/llvm/bin/llvm-config --version
run_command pkg_config_version pkg-config --version
run_command pkg_config_lz4 pkg-config --modversion liblz4
run_command pkg_config_zstd pkg-config --modversion libzstd
run_command brew_version brew --version
run_command brew_prefix brew --prefix
run_command brew_components brew list --versions cmake ninja llvm lz4 zstd
run_command macports_version port version
run_command macports_components port installed cmake ninja llvm lz4 zstd

device_compile_out="${LOG_DIR}/metal_device_compile.stdout.txt"
device_compile_err="${LOG_DIR}/metal_device_compile.stderr.txt"
device_run_out="${LOG_DIR}/metal_device_inventory.stdout.json"
device_run_err="${LOG_DIR}/metal_device_inventory.stderr.txt"
swiftc_path="$(/usr/bin/xcrun --find swiftc 2>/dev/null || true)"
if [ -n "${swiftc_path}" ] && [ -x "${swiftc_path}" ]; then
  /bin/mkdir -p "${TMP_DIR}/module-cache"
  TMPDIR="${TMP_DIR}" "${swiftc_path}" \
    -module-cache-path "${TMP_DIR}/module-cache" \
    "${ROOT_DIR}/metal-device-inventory.swift" \
    -framework Foundation -framework Metal \
    -o "${TMP_DIR}/metal-device-inventory" \
    >"${device_compile_out}" 2>"${device_compile_err}"
  compile_rc=$?
  if [ "${compile_rc}" -eq 0 ]; then
    write_status "metal_device_compile" "ok" "0" \
      "xcrun swiftc metal-device-inventory.swift -framework Foundation -framework Metal"
    TMPDIR="${TMP_DIR}" "${TMP_DIR}/metal-device-inventory" \
      >"${device_run_out}" 2>"${device_run_err}"
    run_rc=$?
    if [ "${run_rc}" -eq 0 ] && [ -s "${device_run_out}" ]; then
      write_status "metal_device_inventory" "ok" "0" \
        "compiled Metal device enumeration helper"
    else
      write_status "metal_device_inventory" "exit_${run_rc}" "${run_rc}" \
        "compiled Metal device enumeration helper"
    fi
  else
    : >"${device_run_out}"
    : >"${device_run_err}"
    write_status "metal_device_compile" "exit_${compile_rc}" "${compile_rc}" \
      "xcrun swiftc metal-device-inventory.swift -framework Foundation -framework Metal"
    write_status "metal_device_inventory" "not_run" "-" \
      "compile prerequisite failed"
  fi
else
  : >"${device_compile_out}"
  printf '%s\n' "swiftc unavailable through xcrun" >"${device_compile_err}"
  : >"${device_run_out}"
  : >"${device_run_err}"
  write_status "metal_device_compile" "missing" "127" "xcrun --find swiftc"
  write_status "metal_device_inventory" "not_run" "-" "swiftc unavailable"
fi

INVENTORY_TEXT="${RUN_DIR}/inventory.txt"
{
  printf 'cuda4AS M1 Mac inventory\n'
  printf 'schema: %s\n' "${SCHEMA}"
  printf 'started UTC: %s\n' "${START_UTC}"
  printf 'This run made no requested installation, update, sudo, or GPU workload.\n\n'
  printf '== facts.tsv ==\n'
  /bin/cat "${FACTS}"
  printf '\n== status.tsv ==\n'
  /bin/cat "${STATUS}"
  printf '\n== command stdout ==\n'
  for output_file in "${LOG_DIR}"/*.stdout.txt "${LOG_DIR}"/*.stdout.json; do
    [ -e "${output_file}" ] || continue
    printf '\n--- %s ---\n' "$(basename "${output_file}")"
    /bin/cat "${output_file}"
  done
  printf '\n== nonempty command stderr ==\n'
  for error_file in "${LOG_DIR}"/*.stderr.txt; do
    [ -s "${error_file}" ] || continue
    printf '\n--- %s ---\n' "$(basename "${error_file}")"
    /bin/cat "${error_file}"
  done
} >"${INVENTORY_TEXT}"

(
  cd "${RUN_DIR}" || exit 1
  /usr/bin/find facts.tsv status.tsv inventory.txt logs -type f -print \
    | LC_ALL=C /usr/bin/sort \
    | while IFS= read -r item; do
        /usr/bin/shasum -a 256 "${item}"
      done
) >"${RUN_DIR}/MANIFEST.sha256"
manifest_rc=$?
if [ "${manifest_rc}" -ne 0 ]; then
  printf '%s\n' "fatal: could not create returned-file manifest" >&2
  exit 3
fi

RETURN_NAME="cuda4as-m1-mac-inventory-return-${START_UTC}.tgz"
RETURN_PATH="${RETURN_DIR}/${RETURN_NAME}"
if ! /usr/bin/tar -czf "${RETURN_PATH}" -C "${RUN_DIR}" \
    facts.tsv status.tsv inventory.txt MANIFEST.sha256 logs; then
  printf '%s\n' "fatal: could not create return archive" >&2
  exit 4
fi

RETURN_HASH="$(/usr/bin/shasum -a 256 "${RETURN_PATH}" | /usr/bin/awk '{print $1}')"
RETURN_BYTES="$(/usr/bin/wc -c <"${RETURN_PATH}" | /usr/bin/tr -d '[:space:]')"

printf '\n%s\n' "CUDA4AS M1 MAC INVENTORY COMPLETE"
printf 'Return archive: %s\n' "${RETURN_PATH}"
printf 'Bytes: %s\n' "${RETURN_BYTES}"
printf 'SHA-256: %s\n' "${RETURN_HASH}"
printf '%s\n' "Copy this one .tgz file back to the requested RESULTS/m1/incoming path."
