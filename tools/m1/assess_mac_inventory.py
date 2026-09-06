#!/usr/bin/env python3
"""Assess a normalized M1 Mac inventory against the native-run prerequisites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


SCHEMA = "cuda4as-m1-mac-readiness-v1"
NORMALIZED_SCHEMA = "cuda4as-m1-mac-inventory-normalized-v1"
MIN_FREE_KIB = 5 * 1024 * 1024


def version_tuple(text: Any) -> tuple[int, ...] | None:
    match = re.search(r"(?<![0-9])([0-9]+(?:\.[0-9]+)+)", str(text or ""))
    if not match:
        return None
    return tuple(int(item) for item in match.group(1).split("."))


def version_at_least(text: Any, required: tuple[int, ...]) -> bool:
    actual = version_tuple(text)
    if actual is None:
        return False
    width = max(len(actual), len(required))
    return actual + (0,) * (width - len(actual)) >= required + (0,) * (
        width - len(required)
    )


def check_ok(tools: dict[str, Any], check_id: str) -> bool:
    value = tools.get(check_id)
    return isinstance(value, dict) and value.get("status") == "ok"


def stdout(tools: dict[str, Any], check_id: str) -> str:
    value = tools.get(check_id)
    if not isinstance(value, dict):
        return ""
    return str(value.get("stdout") or "")


def installed_component(tools: dict[str, Any], name: str) -> tuple[bool, str]:
    pkg_id = "pkg_config_lz4" if name == "lz4" else "pkg_config_zstd"
    if check_ok(tools, pkg_id):
        return True, f"{pkg_id}: {stdout(tools, pkg_id)}"
    brew_text = stdout(tools, "brew_components")
    if re.search(rf"(?m)^{re.escape(name)}\s+", brew_text):
        return True, f"Homebrew: {next(line for line in brew_text.splitlines() if line.startswith(name + ' '))}"
    port_text = stdout(tools, "macports_components")
    if re.search(rf"(?mi)^\s*{re.escape(name)}\s+@.*\(active\)", port_text):
        return False, (
            f"MacPorts reports active {name}, but the current runner has no verified "
            "MacPorts prefix/library binding; review before packaging"
        )
    return False, "not found through pkg-config, Homebrew inventory, or active MacPorts listing"


def assess(inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema") != NORMALIZED_SCHEMA:
        raise ValueError(f"unexpected inventory schema: {inventory.get('schema')!r}")
    machine = inventory.get("machine")
    tools = inventory.get("tools")
    if not isinstance(machine, dict) or not isinstance(tools, dict):
        raise ValueError("normalized inventory lacks machine/tools objects")

    requirements: list[dict[str, Any]] = []

    def record(requirement_id: str, ready: bool, observed: Any, constraint: str) -> None:
        requirements.append(
            {
                "id": requirement_id,
                "state": "READY" if ready else "MISSING_OR_INCOMPATIBLE",
                "constraint": constraint,
                "observed": observed,
            }
        )

    record(
        "apple_silicon",
        machine.get("arch") == "arm64",
        machine.get("arch"),
        "arm64 Apple Silicon",
    )
    record(
        "macos",
        machine.get("os_name") == "macOS"
        and version_at_least(machine.get("os_version"), (14, 0)),
        {"name": machine.get("os_name"), "version": machine.get("os_version")},
        "macOS >=14.0",
    )
    devices = machine.get("metal_devices")
    record(
        "metal_device",
        inventory.get("inventory_status") == "COMPLETE"
        and isinstance(devices, list)
        and bool(devices),
        devices,
        "successful enumeration with at least one named Metal device",
    )
    sdk_checks = ("sdk_path", "sdk_version", "find_metal", "find_metallib")
    record(
        "xcode_sdk_metal_tools",
        all(check_ok(tools, item) for item in sdk_checks),
        {item: tools.get(item) for item in sdk_checks},
        "selected macOS SDK plus xcrun metal and metallib",
    )
    cmake_text = stdout(tools, "cmake_version")
    record(
        "cmake",
        check_ok(tools, "cmake_path")
        and check_ok(tools, "cmake_version")
        and version_at_least(cmake_text, (3, 28)),
        cmake_text or tools.get("cmake_path"),
        "CMake >=3.28",
    )
    record(
        "ninja",
        check_ok(tools, "ninja_path") and check_ok(tools, "ninja_version"),
        stdout(tools, "ninja_version") or tools.get("ninja_path"),
        "Ninja available for explicit object/device-link/native-link targets",
    )

    llvm_candidates = []
    for check_id, origin in (
        ("homebrew_llvm_version", "/opt/homebrew/opt/llvm"),
        ("intel_homebrew_llvm_version", "/usr/local/opt/llvm"),
        ("llvm_config_version", "llvm-config on PATH"),
    ):
        value = stdout(tools, check_id)
        llvm_candidates.append(
            {
                "check": check_id,
                "origin": origin,
                "status": tools.get(check_id, {}).get("status")
                if isinstance(tools.get(check_id), dict)
                else "absent",
                "version": value or None,
                "eligible": check_ok(tools, check_id) and version_at_least(value, (18, 0)),
            }
        )
    selected_llvm = next(
        (item for item in llvm_candidates if item["status"] == "ok"), None
    )
    generic_config_ready = (
        selected_llvm is None
        or selected_llvm["check"] != "llvm_config_version"
        or (
            check_ok(tools, "llvm_config_path")
            and check_ok(tools, "llvm_config_prefix")
            and check_ok(tools, "llvm_config_cmakedir")
        )
    )
    record(
        "llvm",
        bool(selected_llvm and selected_llvm["eligible"] and generic_config_ready),
        {"selection_order": llvm_candidates, "selected": selected_llvm},
        "LLVM >=18 with llvm-config, CMake config, and sibling CUDA-capable clang/clang++",
    )

    lz4_ready, lz4_observed = installed_component(tools, "lz4")
    zstd_ready, zstd_observed = installed_component(tools, "zstd")
    record("lz4", lz4_ready, lz4_observed, "LZ4 headers and linkable library")
    record("zstd", zstd_ready, zstd_observed, "Zstd headers and linkable library")
    free_kib = machine.get("task_filesystem_free_kib")
    record(
        "task_disk_space",
        isinstance(free_kib, int) and free_kib >= MIN_FREE_KIB,
        {"available_kib": free_kib},
        f"at least {MIN_FREE_KIB} KiB free on the task filesystem",
    )

    gaps = [item["id"] for item in requirements if item["state"] != "READY"]
    return {
        "schema": SCHEMA,
        "inventory_archive": inventory.get("source_archive"),
        "ready_for_native_drop": not gaps,
        "requirements": requirements,
        "gaps": gaps,
        "policy": {
            "assessment_is_install_authorization": False,
            "missing_dependency_action": (
                "Ask the user directly with exact package, version, size, and target; do not install automatically."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        result = assess(inventory)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"INVALID {args.inventory}: {exc}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    state = "READY" if result["ready_for_native_drop"] else "NOT_READY"
    print(f"{state} {args.inventory} -> {args.output}; gaps={result['gaps']}")
    return 0 if result["ready_for_native_drop"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
