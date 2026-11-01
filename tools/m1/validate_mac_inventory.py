#!/usr/bin/env python3
"""Validate and normalize a user-returned M1 Mac inventory archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any


MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXPANDED_BYTES = 200 * 1024 * 1024
MAX_MEMBERS = 2_000
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
EXPECTED_CHECK_IDS = {
    "date_utc",
    "sw_vers",
    "uname",
    "architecture",
    "model",
    "cpu_brand",
    "memory_bytes",
    "physical_cpu",
    "logical_cpu",
    "displays",
    "task_disk_free",
    "root_disk_free",
    "xcode_select",
    "xcodebuild_version",
    "xcodebuild_sdks",
    "sdk_path",
    "sdk_version",
    "sdk_build_version",
    "find_metal",
    "metal_version",
    "find_metallib",
    "find_metal_ar",
    "find_swiftc",
    "swiftc_version",
    "find_clang",
    "clang_version",
    "git_version",
    "cmake_path",
    "cmake_version",
    "ninja_path",
    "ninja_version",
    "llvm_config_path",
    "llvm_config_version",
    "llvm_config_prefix",
    "llvm_config_cmakedir",
    "homebrew_llvm_version",
    "intel_homebrew_llvm_version",
    "pkg_config_version",
    "pkg_config_lz4",
    "pkg_config_zstd",
    "brew_version",
    "brew_prefix",
    "brew_components",
    "macports_version",
    "macports_components",
    "metal_device_compile",
    "metal_device_inventory",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name


def safe_name(raw_name: str) -> str:
    name = normalized_name(raw_name)
    parts = PurePosixPath(name).parts
    if (
        not name
        or raw_name.startswith("/")
        or ".." in parts
        or "\\" in name
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError(f"unsafe archive path: {raw_name!r}")
    return name


def load_verified_files(archive: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("archive exceeds the 100 MiB inventory limit")
    files: dict[str, bytes] = {}
    expanded = 0
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        if len(members) > MAX_MEMBERS:
            raise ValueError("archive contains too many members")
        for member in members:
            if member.isdir():
                if normalized_name(member.name):
                    safe_name(member.name)
                continue
            name = safe_name(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported archive member type: {member.name!r}")
            if not member.isfile():
                raise ValueError(f"unsupported archive member: {member.name!r}")
            if name in files:
                raise ValueError(f"duplicate archive file: {name}")
            handle = tf.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read archive file: {name}")
            data = handle.read()
            expanded += len(data)
            if expanded > MAX_EXPANDED_BYTES:
                raise ValueError("expanded archive exceeds the 200 MiB limit")
            files[name] = data

    required = {"facts.tsv", "status.tsv", "inventory.txt", "MANIFEST.sha256"}
    missing = sorted(required - files.keys())
    if missing:
        raise ValueError(f"missing required archive files: {missing}")

    expected: dict[str, str] = {}
    for line_number, line in enumerate(
        files["MANIFEST.sha256"].decode("utf-8").splitlines(), 1
    ):
        match = SHA_LINE.fullmatch(line)
        if not match:
            raise ValueError(f"malformed manifest line {line_number}")
        digest, name = match.groups()
        name = safe_name(name)
        if name in expected:
            raise ValueError(f"duplicate manifest path: {name}")
        expected[name] = digest
    actual_names = set(files) - {"MANIFEST.sha256"}
    if set(expected) != actual_names:
        raise ValueError(
            f"manifest member mismatch; missing={sorted(set(expected)-actual_names)}, "
            f"extra={sorted(actual_names-set(expected))}"
        )
    for name, digest in expected.items():
        if sha256(files[name]) != digest:
            raise ValueError(f"manifest hash mismatch: {name}")
    return files, {
        "archive_members": len(files),
        "expanded_file_bytes": expanded,
        "manifest_entries": len(expected),
        "internal_manifest_sha256": sha256(files["MANIFEST.sha256"]),
    }


def parse_tsv(
    data: bytes, expected_columns: tuple[str, ...], label: str
) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != expected_columns:
        raise ValueError(
            f"{label} columns must be {expected_columns}, "
            f"observed {tuple(reader.fieldnames or ())}"
        )
    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"malformed {label} row {line_number}")
        rows.append({key: str(value) for key, value in row.items()})
    return rows


def first_line(files: dict[str, bytes], path: str) -> str | None:
    data = files.get(path, b"").decode("utf-8", errors="replace").strip()
    return data.splitlines()[0] if data else None


def command_text(files: dict[str, bytes], check_id: str) -> str | None:
    data = files.get(f"logs/{check_id}.stdout.txt")
    if data is None:
        data = files.get(f"logs/{check_id}.stdout.json", b"")
    return data.decode("utf-8", errors="replace").strip() or None


def df_available_kib(text: str | None) -> int | None:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    fields = lines[-1].split()
    if len(fields) < 4:
        return None
    try:
        return int(fields[3])
    except ValueError:
        return None


def parse_sw_vers(text: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def normalize(archive: Path) -> dict[str, Any]:
    archive_data = archive.read_bytes()
    files, validation = load_verified_files(archive)
    facts_rows = parse_tsv(files["facts.tsv"], ("key", "value"), "facts.tsv")
    status_rows = parse_tsv(
        files["status.tsv"],
        ("check_id", "status", "exit_code", "command"),
        "status.tsv",
    )
    facts: dict[str, str] = {}
    for row in facts_rows:
        if not row["key"] or row["key"] in facts:
            raise ValueError(f"empty or duplicate facts key: {row['key']!r}")
        facts[row["key"]] = row["value"]
    checks: dict[str, dict[str, str]] = {}
    for row in status_rows:
        check_id = row["check_id"]
        if not check_id or check_id in checks:
            raise ValueError(f"empty or duplicate check id: {check_id!r}")
        status = row["status"]
        if status not in {"ok", "missing", "not_run"} and not re.fullmatch(
            r"exit_[0-9]+", status
        ):
            raise ValueError(f"invalid status for {check_id}: {status!r}")
        exit_code = row["exit_code"]
        if status == "ok" and exit_code != "0":
            raise ValueError(f"ok check has nonzero exit for {check_id}")
        if status == "missing" and exit_code != "127":
            raise ValueError(f"missing check has unexpected exit for {check_id}")
        if status == "not_run" and exit_code != "-":
            raise ValueError(f"not_run check has an exit code for {check_id}")
        if status.startswith("exit_") and exit_code != status.removeprefix("exit_"):
            raise ValueError(f"exit status/code disagree for {check_id}")
        checks[check_id] = row
    missing_checks = sorted(EXPECTED_CHECK_IDS - checks.keys())
    extra_checks = sorted(checks.keys() - EXPECTED_CHECK_IDS)
    if missing_checks or extra_checks:
        raise ValueError(
            f"inventory check set mismatch; missing={missing_checks}, extra={extra_checks}"
        )
    if facts.get("schema") != "cuda4as-m1-mac-inventory-v1":
        raise ValueError(f"unexpected inventory schema: {facts.get('schema')!r}")

    sw_vers = parse_sw_vers(command_text(files, "sw_vers"))
    devices: list[dict[str, Any]] = []
    device_document: dict[str, Any] | None = None
    default_device_status = "NOT_REPORTED"
    device_bytes = files.get("logs/metal_device_inventory.stdout.json", b"").strip()
    device_status = checks["metal_device_inventory"]["status"]
    if device_status == "ok" and not device_bytes:
        raise ValueError("successful Metal device inventory has no JSON output")
    if device_status != "ok" and device_bytes:
        raise ValueError("non-successful Metal device inventory has JSON output")
    if device_bytes:
        value = json.loads(device_bytes)
        if not isinstance(value, dict) or value.get("schema") != (
            "cuda4as-m1-metal-device-inventory-v1"
        ):
            raise ValueError("unexpected Metal device inventory schema")
        raw_devices = value.get("devices")
        if not isinstance(raw_devices, list):
            raise ValueError("Metal devices must be an array")
        if value.get("device_count") != len(raw_devices):
            raise ValueError("Metal device_count does not match devices array")
        seen_registry_ids: set[str] = set()
        for index, device in enumerate(raw_devices):
            if not isinstance(device, dict):
                raise ValueError(f"Metal device {index} must be an object")
            name = device.get("name")
            registry_id = device.get("registry_id")
            if not isinstance(name, str) or not name or not isinstance(
                registry_id, str
            ) or not registry_id:
                raise ValueError(f"Metal device {index} lacks name/registry_id")
            if registry_id in seen_registry_ids:
                raise ValueError(f"duplicate Metal registry_id: {registry_id}")
            seen_registry_ids.add(registry_id)
            devices.append(device)
        device_document = value
        default_name = value.get("default_device_name")
        default_registry_id = value.get("default_device_registry_id")
        if default_name is None and default_registry_id is None:
            default_device_status = "UNAVAILABLE"
        elif (
            isinstance(default_name, str)
            and default_name
            and isinstance(default_registry_id, str)
            and default_registry_id
        ):
            if not any(
                device["name"] == default_name
                and device["registry_id"] == default_registry_id
                for device in devices
            ):
                raise ValueError("default Metal device is absent from enumerated devices")
            default_device_status = "MATCHED"
        else:
            raise ValueError(
                "default Metal device identity must be both null or a nonempty name/registry_id"
            )
    if device_status == "ok" and checks["metal_device_compile"]["status"] != "ok":
        raise ValueError("Metal inventory ran without a successful helper compile")

    non_ok = [row for row in status_rows if row.get("status") != "ok"]
    normalized = {
        "schema": "cuda4as-m1-mac-inventory-normalized-v1",
        "source_archive": {
            "filename": archive.name,
            "bytes": len(archive_data),
            "sha256": sha256(archive_data),
        },
        "validation": {
            **validation,
            "archive_members_safe": True,
            "internal_manifest_valid": True,
        },
        "inventory_status": (
            "COMPLETE"
            if device_status == "ok"
            else "INCOMPLETE"
        ),
        "machine": {
            "os_name": sw_vers.get("ProductName", "unknown"),
            "os_version": sw_vers.get("ProductVersion", "unknown"),
            "os_build": sw_vers.get("BuildVersion", "unknown"),
            "arch": first_line(files, "logs/architecture.stdout.txt") or "unknown",
            "model": first_line(files, "logs/model.stdout.txt") or "unknown",
            "cpu_brand": first_line(files, "logs/cpu_brand.stdout.txt") or "unknown",
            "memory_bytes": first_line(files, "logs/memory_bytes.stdout.txt"),
            "task_filesystem_free_kib": df_available_kib(
                command_text(files, "task_disk_free")
            ),
            "root_filesystem_free_kib": df_available_kib(
                command_text(files, "root_disk_free")
            ),
            "developer_directory": facts.get(
                "selected_developer_directory", "unknown"
            ),
            "developer_kind": facts.get("selected_developer_kind", "unknown"),
            "applications_xcode_present": facts.get(
                "applications_xcode_present", "unknown"
            ),
            "xcode_version": command_text(files, "xcodebuild_version") or "unknown",
            "sdk_version": first_line(files, "logs/sdk_version.stdout.txt")
            or "unknown",
            "sdk_build_version": first_line(
                files, "logs/sdk_build_version.stdout.txt"
            ),
            "sdk_path": first_line(files, "logs/sdk_path.stdout.txt"),
            "metal_compiler": first_line(files, "logs/find_metal.stdout.txt")
            or "missing",
            "metallib_tool": first_line(files, "logs/find_metallib.stdout.txt"),
            "metal_devices": devices,
            "default_metal_device_name": (
                device_document or {}
            ).get("default_device_name"),
            "default_metal_device_registry_id": (
                device_document or {}
            ).get("default_device_registry_id"),
            "default_metal_device_status": default_device_status,
        },
        "tools": {
            check_id: {
                "status": row["status"],
                "exit_code": row["exit_code"],
                "command": row["command"],
                "stdout": command_text(files, check_id),
            }
            for check_id, row in checks.items()
        },
        "checks": status_rows,
        "non_ok_checks": non_ok,
        "facts": facts,
    }
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"archive does not exist: {args.archive}")
    try:
        result = normalize(args.archive)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        print(f"INVALID {args.archive}: {exc}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"VALID {args.archive} -> {args.output} "
        f"({len(result['machine']['metal_devices'])} Metal devices, "
        f"{result['inventory_status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
