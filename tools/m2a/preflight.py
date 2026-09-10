#!/usr/bin/env python3
"""Check the exact M1 inventory/toolchain binding before native AOT work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(argv: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        return 127, str(exc)
    return p.returncode, (p.stdout or p.stderr).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("binding", type=Path)
    p.add_argument("work", type=Path)
    args = p.parse_args()
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    machine = binding["machine"]
    expected = {
        "arch": machine["arch"], "model": machine["model"], "os_version": machine["os_version"],
        "os_build": machine["os_build"], "sdk_version": machine["sdk_version"], "llvm_version": binding["llvm"]["version"],
    }
    commands = {
        "arch": ["/usr/bin/uname", "-m"],
        "os_version": ["/usr/bin/sw_vers", "-productVersion"],
        "os_build": ["/usr/bin/sw_vers", "-buildVersion"],
        "model": ["/usr/sbin/sysctl", "-n", "hw.model"],
        "cpu_brand": ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        "sdk_version": ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version"],
        "sdk_path": ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
        "metal_compiler": ["/usr/bin/xcrun", "--sdk", "macosx", "--find", "metal"],
        "metallib_tool": ["/usr/bin/xcrun", "--sdk", "macosx", "--find", "metallib"],
    }
    observed: dict[str, str] = {}
    gaps: list[str] = []
    for key, argv in commands.items():
        code, value = run(argv)
        observed[key] = value if code == 0 else "missing"
        if code != 0:
            gaps.append(f"{key}: command failed ({code})")
    clang = "/opt/homebrew/opt/llvm/bin/clang++"
    if not Path(clang).is_file():
        gaps.append("Homebrew LLVM clang++ is missing: " + clang)
        observed["llvm_version"] = "missing"
    else:
        code, value = run([clang, "--version"])
        observed["llvm_version"] = value.splitlines()[0] if code == 0 and value else "missing"
        if code != 0:
            gaps.append("Homebrew LLVM clang++ --version failed")
    usage = shutil.disk_usage(args.work)
    observed["free_bytes"] = str(usage.free)
    checks = {
        "arch": observed.get("arch") == expected["arch"],
        "model": observed.get("model") == expected["model"],
        "os_version": observed.get("os_version") == expected["os_version"],
        "os_build": observed.get("os_build") == expected["os_build"],
        "sdk_version": observed.get("sdk_version") == expected["sdk_version"],
        "llvm_version": expected["llvm_version"] in observed.get("llvm_version", ""),
        "free_space": usage.free >= 5 * 1024**3,
        "metal_tools": observed.get("metal_compiler") == machine["metal_compiler"] and bool(observed.get("metallib_tool")) and Path(observed.get("metal_compiler", "missing")).is_file() and Path(observed.get("metallib_tool", "missing")).is_file(),
    }
    mismatches = [key for key, ok in checks.items() if not ok]
    doc = {"schema": "cuda4as-m2a-preflight-v1", "status": "PASS" if not gaps and not mismatches else "ENVIRONMENT_GAP" if gaps else "FAIL", "expected": expected, "observed": observed, "checks": checks, "gaps": gaps, "mismatches": mismatches, "network_operations": "none", "install_update_sudo_operations": "none"}
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "inventory-facts.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if gaps:
        print("M2A_PREFLIGHT_ENVIRONMENT_GAP", file=sys.stderr)
        return 77
    if mismatches:
        print("M2A_PREFLIGHT_MISMATCH: " + ",".join(mismatches), file=sys.stderr)
        return 1
    print("M2A_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
