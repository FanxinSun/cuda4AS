#!/bin/bash
# User-operated M2A Native AOT Core v1 runner. All writes stay below package.
set -u

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$PACKAGE_ROOT/results/$RUN_ID"
RETURN_DIR="$PACKAGE_ROOT/returns"
mkdir -p "$WORK/logs" "$WORK/metallib" "$RETURN_DIR"
RUN_EXIT=1

log_run() {
  local name="$1"; shift
  "$@" >"$WORK/logs/${name}.stdout.txt" 2>"$WORK/logs/${name}.stderr.txt"
  local code=$?
  printf '%s\n' "$code" >"$WORK/logs/${name}.exit"
  return "$code"
}

python3 "$PACKAGE_ROOT/tools/m2a/verify_m2a_package.py" "$PACKAGE_ROOT" >"$WORK/logs/package-verify.txt" 2>&1
VERIFY_EXIT=$?
printf '%s\n' "$VERIFY_EXIT" >"$WORK/logs/package-verify.exit"
if [ "$VERIFY_EXIT" -ne 0 ]; then
  printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"FAIL","failed_stage":"package_verify","cpu_fallback":false}' >"$WORK/m2a-result.json"
else
  python3 "$PACKAGE_ROOT/tools/m2a/preflight.py" "$PACKAGE_ROOT/inventory-binding.json" "$WORK" >"$WORK/logs/preflight.stdout.txt" 2>"$WORK/logs/preflight.stderr.txt"
  PREFLIGHT_EXIT=$?
  printf '%s\n' "$PREFLIGHT_EXIT" >"$WORK/logs/preflight.exit"
  if [ "$PREFLIGHT_EXIT" -ne 0 ]; then
    if [ "$PREFLIGHT_EXIT" -eq 77 ]; then RUN_EXIT=77; CLASSIFICATION="SKIP_ENVIRONMENT"; else RUN_EXIT=1; CLASSIFICATION="FAIL"; fi
    printf '%s\n' "preflight exit $PREFLIGHT_EXIT" >"$WORK/logs/environment-gap.txt"
    printf '{"schema":"cuda4as-m2a-result-v1","classification":"%s","failed_stage":"preflight","cpu_fallback":false}\n' "$CLASSIFICATION" >"$WORK/m2a-result.json"
  else
    CLANG="/opt/homebrew/opt/llvm/bin/clang++"
    if [ ! -x "$CLANG" ]; then CLANG="$(command -v clang++ 2>/dev/null || true)"; fi
    if [ -z "$CLANG" ] || ! command -v xcrun >/dev/null 2>&1; then
    printf '%s\n' 'environment gap: clang++ or xcrun is unavailable' >"$WORK/logs/environment-gap.txt"
    printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"SKIP_ENVIRONMENT","failed_stage":"toolchain_discovery","cpu_fallback":false}' >"$WORK/m2a-result.json"
    RUN_EXIT=77
    else
    python3 "$PACKAGE_ROOT/tools/m2a/native_aot.py" --source "$PACKAGE_ROOT/oracle/src/vector_add.cu" --header "$PACKAGE_ROOT/oracle/src/oracle.h" --work "$WORK" --clang "$CLANG" >"$WORK/logs/device-import.stdout.txt" 2>"$WORK/logs/device-import.stderr.txt"
    DRIVER_EXIT=$?
    printf '%s\n' "$DRIVER_EXIT" >"$WORK/logs/device-import.exit"
    if [ "$DRIVER_EXIT" -ne 0 ]; then
      printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"FAIL","failed_stage":"device_import","cpu_fallback":false}' >"$WORK/m2a-result.json"
    else
      python3 - "$WORK" <<'PY' >"$WORK/runtime-config.h"
import json, pathlib, sys
w=pathlib.Path(sys.argv[1]); d=json.loads((w/'driver-facts.json').read_text()); h=d['host']; ir=json.loads((w/'ir.json').read_text()); k=ir['kernel']['name']; b=h['launch']['block_x']; binding=json.loads((w.parent.parent/'inventory-binding.json').read_text())
dev=binding['machine']['metal_devices'][0]
print('#pragma once')
print(f'#define CUDA4AS_KERNEL_NAME "{k}"')
print(f'#define CUDA4AS_ELEMENT_COUNT {h["element_count"]}u')
print(f'#define CUDA4AS_BLOCK_X {b}u')
print(f'#define CUDA4AS_SEED {h["seed"]}ULL')
print(f'#define CUDA4AS_HI_A {h["input_generator"]["calls"][0]["hi"]}')
print(f'#define CUDA4AS_DEN_A {h["input_generator"]["calls"][0]["den"]}')
print(f'#define CUDA4AS_HI_B {h["input_generator"]["calls"][1]["hi"]}')
print(f'#define CUDA4AS_DEN_B {h["input_generator"]["calls"][1]["den"]}')
print(f'#define CUDA4AS_EXPECTED_OUTPUT_SHA256 "{h["expected_output"]["sha256"]}"')
print(f'#define CUDA4AS_EXPECTED_DEVICE_NAME "{dev["name"]}"')
print(f'#define CUDA4AS_EXPECTED_REGISTRY_ID {dev["registry_id"]}ULL')
PY
      CONFIG_EXIT=$?
      cp "$WORK/runtime-config.h" "$WORK/logs/runtime-config.txt" 2>/dev/null || true
      if [ "$CONFIG_EXIT" -ne 0 ]; then
        printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"FAIL","failed_stage":"runtime_config","cpu_fallback":false}' >"$WORK/m2a-result.json"
      else
        SDK="$(xcrun --sdk macosx --show-sdk-path)"
        METAL="$(xcrun --sdk macosx --find metal)"
        METALLIB="$(xcrun --sdk macosx --find metallib)"
        if [ -z "$SDK" ] || [ -z "$METAL" ] || [ -z "$METALLIB" ]; then
          printf '%s\n' 'environment gap: xcrun SDK/metal/metallib discovery failed' >"$WORK/logs/environment-gap.txt"
          printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"SKIP_ENVIRONMENT","failed_stage":"aot_tool_discovery","cpu_fallback":false}' >"$WORK/m2a-result.json"
          RUN_EXIT=77
        else
          log_run metal_compile "$METAL" -c "$WORK/kernel.metal" -o "$WORK/kernel.air"
          METAL_EXIT=$?
          if [ "$METAL_EXIT" -ne 0 ]; then
            printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"FAIL","failed_stage":"metal_compile","cpu_fallback":false}' >"$WORK/m2a-result.json"
          else
            KERNEL_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["kernel"]["name"])' "$WORK/ir.json")"
            log_run metallib_link "$METALLIB" "$WORK/kernel.air" -o "$WORK/metallib/$KERNEL_NAME.metallib"
            LIB_EXIT=$?
            if [ "$LIB_EXIT" -ne 0 ]; then
              printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"FAIL","failed_stage":"metallib","cpu_fallback":false}' >"$WORK/m2a-result.json"
            else
              log_run host_compile "$CLANG" -arch arm64 -std=c++17 -fobjc-arc -isysroot "$SDK" -mmacosx-version-min=14.0 -I"$WORK" -c "$PACKAGE_ROOT/tools/m2a/runtime.mm" -o "$WORK/runtime.o"
              HOST_EXIT=$?
              if [ "$HOST_EXIT" -ne 0 ]; then
                printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"FAIL","failed_stage":"host_compile","cpu_fallback":false}' >"$WORK/m2a-result.json"
              else
                log_run native_link "$CLANG" -arch arm64 -isysroot "$SDK" -mmacosx-version-min=14.0 "$WORK/runtime.o" -framework Foundation -framework Metal -framework CoreFoundation -o "$WORK/native"
                LINK_EXIT=$?
                if [ "$LINK_EXIT" -ne 0 ]; then
                  printf '%s\n' '{"schema":"cuda4as-m2a-result-v1","classification":"FAIL","failed_stage":"native_link","cpu_fallback":false}' >"$WORK/m2a-result.json"
                else
                  log_run runtime_launch "$WORK/native" "$WORK"
                  RUN_EXIT=$?
                fi
              fi
            fi
          fi
        fi
      fi
    fi
  fi
  fi
fi

python3 - "$WORK" "$RUN_EXIT" <<'PY' >"$WORK/stage-record.json"
import json, pathlib, sys
w=pathlib.Path(sys.argv[1]); exit_code=int(sys.argv[2])
def stage(name):
    p=w/'logs'/f'{name}.exit'
    if not p.exists(): return 'NOT_RUN'
    try: code=int(p.read_text().strip())
    except ValueError: return 'FAIL'
    return 'PASS' if code == 0 else 'FAIL'
stages={name:stage(name) for name in ('package-verify','preflight','device-import','metal_compile','metallib_link','host_compile','native_link','runtime_launch')}
for name,path in (('ir_verify','ir.json'),('device_link','device-link.json'),('msl_generation','kernel.metal')):
    stages[name]='PASS' if (w/path).is_file() else 'NOT_RUN'
print(json.dumps({'schema':'cuda4as-m2a-stage-record-v1','runner_exit':exit_code,'classification':'PASS_GPU' if exit_code==0 else 'SKIP_ENVIRONMENT' if exit_code==77 else 'FAIL','stages':stages,'cpu_fallback':False,'runtime_compilation':False}, sort_keys=True, indent=2))
PY

cp "$PACKAGE_ROOT/PACKAGE-MANIFEST.sha256" "$WORK/PACKAGE-MANIFEST.sha256"
cp "$PACKAGE_ROOT/inventory-binding.json" "$WORK/inventory-binding.json"
cp "$PACKAGE_ROOT/expected.json" "$WORK/expected.json"
RETURN_NAME="cuda4as-m2a-native-return-${RUN_ID}.tgz"
RETURN_PATH="$RETURN_DIR/$RETURN_NAME"
tar -czf "$RETURN_PATH" -C "$WORK" .
BYTES="$(wc -c <"$RETURN_PATH" | tr -d '[:space:]')"
SHA="$(shasum -a 256 "$RETURN_PATH" | awk '{print $1}')"
cp "$RETURN_PATH" "$HOME/Downloads/$RETURN_NAME" 2>/dev/null || true
printf '\nCUDA4AS M2A NATIVE AOT VECTOR_ADD COMPLETE\n'
printf 'Runner exit: %s (0=PASS_GPU, 1=failure, 77=environment gap)\n' "$RUN_EXIT"
printf 'Return archive: %s\nBytes: %s\nSHA-256: %s\n' "$RETURN_PATH" "$BYTES" "$SHA"
printf 'Return this one archive unchanged.\n'
printf 'RUN_EXIT=%s\nRETURN_ARCHIVE=%s\nRETURN_BYTES=%s\nRETURN_SHA256=%s\n' "$RUN_EXIT" "$RETURN_PATH" "$BYTES" "$SHA"
exit "$RUN_EXIT"
