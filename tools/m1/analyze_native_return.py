#!/usr/bin/env python3
"""Validate an M1 native return and emit a normative result record.

The analyzer never extracts the returned archive. It verifies the archive's
complete internal manifest, binds it to a previously normalized Mac inventory,
recomputes output hashes, parses CuMetal provenance, and then applies the M1
cross-field validator before writing a result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile
from typing import Any

try:
    from .validate_results import CLASSIFICATIONS, validate_document
except ImportError:  # Direct invocation: python3 tools/m1/analyze_native_return.py
    from validate_results import CLASSIFICATIONS, validate_document


ROOT = Path(__file__).resolve().parents[2]
FIXTURES_PATH = ROOT / "docs/m1/fixtures.json"
CANDIDATE_REVISION = "f486e5ebcfd381d06e3297afd65dbcbd5006a902"
VF64_REVISION = "729021777455da72db8809d9ef1269c677d88b3f"
CANDIDATE_ARCHIVE_SHA256 = "57358b123daece57e472a8bf2805a0919e6879e7a1a781d8d20866b12ffaafbd"
VF64_ARCHIVE_SHA256 = "c9e0308a54a3beec0dba15a12b81a68cda9ad502a919a6dd1cfe193a4bd6e5a5"
RESULT_SCHEMA = "cuda4as-m1-result-v1"
RETURN_SCHEMA = "cuda4as-m1-native-return-v1"
INVENTORY_SCHEMA = "cuda4as-m1-mac-inventory-normalized-v1"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_BYTES = 1024 * 1024 * 1024
MAX_MEMBERS = 20_000
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

CASE_OUTPUTS = {
    "oracle.vector_add": "outputs/oracle-vector-add.bin",
    "integration.minimal_cmake_cuda": "outputs/cmake-vector-add.bin",
    "integration.multi_tu_device_link": "outputs/cmake-device-link.bin",
}
EXPECTED_KERNELS = {
    "oracle.vector_add": "vector_add",
    "integration.minimal_cmake_cuda": "vector_add",
    "integration.multi_tu_device_link": "transform_kernel",
}
EXPECTED_LOWERING_SOURCES = {
    "oracle.vector_add": {"generic_nvvm"},
    "integration.minimal_cmake_cuda": {"generic_ptx"},
    "integration.multi_tu_device_link": {"generic_ptx"},
}
CASE_TYPES = {
    "oracle.vector_add": "existing_oracle",
    "integration.minimal_cmake_cuda": "unchanged_cmake_project",
    "integration.multi_tu_device_link": "unchanged_cmake_project",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name


def _safe_name(raw_name: str) -> str:
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


def parse_checksum_manifest(data: bytes, label: str) -> dict[str, str]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    records: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        match = SHA_LINE.fullmatch(line)
        if not match:
            raise ValueError(f"malformed {label} line {line_number}")
        digest, raw_name = match.groups()
        name = _safe_name(raw_name)
        if name in records:
            raise ValueError(f"duplicate {label} path: {name}")
        records[name] = digest
    if not records:
        raise ValueError(f"{label} is empty")
    return records


def load_verified_files(archive: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Read a return without extraction and verify its exact inner manifest."""

    archive_bytes = archive.stat().st_size
    if archive_bytes > MAX_ARCHIVE_BYTES:
        raise ValueError("archive exceeds the 512 MiB return limit")
    files: dict[str, bytes] = {}
    expanded = 0
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        if len(members) > MAX_MEMBERS:
            raise ValueError("archive contains too many members")
        declared_expanded = sum(member.size for member in members if member.isfile())
        if declared_expanded > MAX_EXPANDED_BYTES:
            raise ValueError("expanded archive exceeds the 1 GiB limit")
        for member in members:
            if member.isdir():
                if normalized_name(member.name):
                    _safe_name(member.name)
                continue
            name = _safe_name(member.name)
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
            if len(data) != member.size:
                raise ValueError(f"truncated archive file: {name}")
            expanded += len(data)
            if expanded > MAX_EXPANDED_BYTES:
                raise ValueError("expanded archive exceeds the 1 GiB limit")
            files[name] = data

    required = {
        "facts.tsv",
        "case-stage-events.tsv",
        "assertions.tsv",
        "environment-gaps.txt",
        "MANIFEST.sha256",
        "package/PACKAGE-MANIFEST.sha256",
        "package/target-inventory-binding.json",
    }
    missing = sorted(required - files.keys())
    if missing:
        raise ValueError(f"missing required archive files: {missing}")
    expected = parse_checksum_manifest(files["MANIFEST.sha256"], "return manifest")
    actual_names = set(files) - {"MANIFEST.sha256"}
    if set(expected) != actual_names:
        raise ValueError(
            "return manifest member mismatch; "
            f"missing={sorted(set(expected) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected))}"
        )
    for name, digest in expected.items():
        if sha256(files[name]) != digest:
            raise ValueError(f"return manifest hash mismatch: {name}")
    return files, {
        "archive_bytes": archive_bytes,
        "archive_sha256": sha256(archive.read_bytes()),
        "archive_members": len(files),
        "expanded_file_bytes": expanded,
        "internal_manifest_sha256": sha256(files["MANIFEST.sha256"]),
    }


def parse_tsv(data: bytes, columns: tuple[str, ...], label: str) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != columns:
        raise ValueError(
            f"{label} columns must be {columns}, observed {tuple(reader.fieldnames or ())}"
        )
    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"malformed {label} row {line_number}")
        rows.append({key: str(value) for key, value in row.items()})
    return rows


def unique_map(rows: list[dict[str, str]], key: str, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        item = row[key]
        if not item:
            raise ValueError(f"empty {label} key")
        if item in result:
            raise ValueError(f"duplicate {label} key: {item}")
        result[item] = row["value"]
    return result


def return_ref(archive: Path, member: str | None) -> str | None:
    return None if member is None else f"return:{archive.name}#{member}"


def parse_exit_code(value: str, label: str) -> int | None:
    if value == "-":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid exit code for {label}: {value!r}") from exc


def parse_events(
    files: dict[str, bytes], fixture_cases: dict[str, dict[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, str]]]:
    rows = parse_tsv(
        files["case-stage-events.tsv"],
        ("sequence", "case_id", "stage", "status", "exit_code", "log", "note"),
        "case-stage-events.tsv",
    )
    previous = 0
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    allowed_status = {"PASS", "FAIL", "NOT_RUN", "NOT_APPLICABLE"}
    for row in rows:
        try:
            sequence = int(row["sequence"])
        except ValueError as exc:
            raise ValueError(f"invalid event sequence: {row['sequence']!r}") from exc
        if sequence <= previous:
            raise ValueError("event sequence must be strictly increasing")
        previous = sequence
        case_id = row["case_id"]
        stage = row["stage"]
        status = row["status"]
        if status not in allowed_status:
            raise ValueError(f"invalid event status: {status!r}")
        if case_id in fixture_cases and stage not in fixture_cases[case_id]["future_required_stages"]:
            raise ValueError(f"unknown stage {stage!r} for {case_id}")
        if case_id not in fixture_cases and not case_id.startswith("_"):
            raise ValueError(f"unknown event case id: {case_id}")
        exit_code = parse_exit_code(row["exit_code"], f"{case_id}/{stage}")
        log = None if row["log"] == "-" else _safe_name(row["log"])
        if status == "NOT_RUN":
            if exit_code is not None or log is not None:
                raise ValueError(f"NOT_RUN event carries execution evidence: {case_id}/{stage}")
        else:
            if exit_code is None or log is None:
                raise ValueError(f"completed event lacks exit/log: {case_id}/{stage}")
            if log not in files:
                raise ValueError(f"event references missing log: {log}")
            if status == "PASS" and exit_code != 0:
                raise ValueError(f"PASS event has nonzero exit: {case_id}/{stage}")
        latest[(case_id, stage)] = {
            "status": status,
            "exit_code": exit_code,
            "log": log,
            "note": row["note"],
            "sequence": sequence,
        }
    for case_id, fixture in fixture_cases.items():
        for stage in fixture["future_required_stages"]:
            if (case_id, stage) not in latest:
                raise ValueError(f"missing stage event: {case_id}/{stage}")
    return latest, rows


def parse_assertions(files: dict[str, bytes]) -> dict[str, list[dict[str, Any]]]:
    rows = parse_tsv(
        files["assertions.tsv"],
        ("case_id", "assertion_id", "passed", "evidence"),
        "assertions.tsv",
    )
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["case_id"], row["assertion_id"])
        if not all(key) or key in seen:
            raise ValueError(f"empty or duplicate assertion: {key}")
        seen.add(key)
        if row["passed"] not in {"true", "false"}:
            raise ValueError(f"invalid assertion boolean for {key}: {row['passed']!r}")
        result.setdefault(row["case_id"], []).append(
            {"id": row["assertion_id"], "passed": row["passed"] == "true"}
        )
    return result


def parse_provenance_line(line: str) -> dict[str, Any]:
    marker = "CUMETAL_PROVENANCE"
    if marker not in line:
        raise ValueError("line does not contain CuMetal provenance")
    try:
        tokens = shlex.split(line.split(marker, 1)[1].strip(), posix=True)
    except ValueError as exc:
        raise ValueError(f"malformed CuMetal provenance quoting: {line!r}") from exc
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate provenance field {key!r}")
        fields[key] = value
    required = {
        "event",
        "kernel",
        "source",
        "semantic_quality",
        "device",
        "device_name",
        "launch_success",
        "duration_ns",
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise ValueError(f"provenance line lacks fields: {missing}")
    if fields["event"] != "kernel_launch":
        raise ValueError(f"unexpected provenance event: {fields['event']!r}")
    if fields["launch_success"] not in {"true", "false"}:
        raise ValueError("provenance launch_success is not boolean")
    try:
        duration_ns = int(fields["duration_ns"])
    except ValueError as exc:
        raise ValueError("provenance duration_ns is not an integer") from exc
    for key in ("kernel", "source", "semantic_quality", "device", "device_name"):
        if not fields[key]:
            raise ValueError(f"empty provenance field: {key}")
    launch_success = fields["launch_success"] == "true"
    # CuMetal emits this record from its completion handler or immediately
    # after waitUntilCompleted. Requiring a nonnegative GPU duration gives the
    # result schema an independently checkable completed=true condition.
    return {
        "kernel": fields["kernel"],
        "device": fields["device"],
        "device_name": fields["device_name"],
        "source": fields["source"],
        "semantic_quality": fields["semantic_quality"],
        "launch_success": launch_success,
        "completed": launch_success and duration_ns >= 0,
    }


def provenance_from_log(data: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    text = data.decode("utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if "CUMETAL_PROVENANCE" not in line:
            continue
        try:
            records.append(parse_provenance_line(line))
        except ValueError as exc:
            errors.append(f"line {line_number}: {exc}")
    return records, errors


def load_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def canonical_version(value: Any) -> str:
    return ";".join(
        part.strip() for part in re.split(r"[;\r\n]+", str(value)) if part.strip()
    )


def observations_match(left: Any, right: Any) -> bool:
    missing_values = {"", "missing", "unknown", "none"}
    left_text = str(left).strip().lower() if left is not None else "none"
    right_text = str(right).strip().lower() if right is not None else "none"
    if left_text in missing_values and right_text in missing_values:
        return True
    return str(left) == str(right)


def validate_inventory_binding(
    files: dict[str, bytes], inventory: dict[str, Any], fixtures: dict[str, Any]
) -> None:
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"unexpected normalized inventory schema: {inventory.get('schema')!r}")
    validation = inventory.get("validation")
    if not isinstance(validation, dict) or validation.get("archive_members_safe") is not True \
            or validation.get("internal_manifest_valid") is not True:
        raise ValueError("normalized inventory lacks successful archive validation")
    binding = load_json(
        files["package/target-inventory-binding.json"], "inventory binding"
    )
    if binding.get("schema") != "cuda4as-m1-inventory-binding-v1" or binding.get(
        "status"
    ) != "BOUND_TO_RETURNED_INVENTORY":
        raise ValueError("native package was not bound to a returned inventory")
    bound = binding.get("inventory_archive")
    source = inventory.get("source_archive")
    if not isinstance(bound, dict) or not isinstance(source, dict) or bound != source:
        raise ValueError("native package inventory binding does not match normalized inventory")

    package_manifest = parse_checksum_manifest(
        files["package/PACKAGE-MANIFEST.sha256"], "package manifest"
    )
    expected_binding_hash = package_manifest.get("target-inventory-binding.json")
    if expected_binding_hash != sha256(files["package/target-inventory-binding.json"]):
        raise ValueError("returned inventory binding does not match the checked package manifest")
    for packaged_path, local_path in (
        ("README.md", ROOT / "tools/m1/native/README.md"),
        ("run-native-feasibility.sh", ROOT / "tools/m1/native/run-native-feasibility.sh"),
        ("tools/prepare-cmake-adapter.sh", ROOT / "tools/m1/native/prepare-cmake-adapter.sh"),
        ("fixtures/fixtures.json", FIXTURES_PATH),
        ("contracts/result-schema-v1.md", ROOT / "docs/m1/result-schema-v1.md"),
        (
            "contracts/result-schema-v1.schema.json",
            ROOT / "docs/m1/result-schema-v1.schema.json",
        ),
    ):
        if package_manifest.get(packaged_path) != sha256(local_path.read_bytes()):
            raise ValueError(f"package evidence does not match local M1 pin: {packaged_path}")
    expected_package_hashes = {
        "candidate/cuda-metal-f486e5eb.tar.gz": CANDIDATE_ARCHIVE_SHA256,
        "candidate/vf64-metal-72902177.tar.gz": VF64_ARCHIVE_SHA256,
    }
    output_paths = {
        "oracle.vector_add": "fixtures/expected/oracle-vector-add.bin",
        "integration.minimal_cmake_cuda": "fixtures/expected/cmake-vector-add.bin",
        "integration.multi_tu_device_link": "fixtures/expected/cmake-device-link.bin",
    }
    for case in fixtures["cases"]:
        for record in case["source_files"] + case["build_files"]:
            source_path = record["path"]
            if source_path.startswith("oracle/src/"):
                packaged_path = "fixtures/oracle/" + source_path.removeprefix("oracle/src/")
            elif source_path.startswith("tests/m1/fixtures/"):
                packaged_path = "fixtures/" + source_path.removeprefix("tests/m1/fixtures/")
            else:
                raise ValueError(f"unmapped fixture source path: {source_path}")
            expected_package_hashes[packaged_path] = record["sha256"]
        expected_package_hashes[output_paths[case["id"]]] = case["expected_output"][
            "sha256"
        ]
    for packaged_path, expected_hash in expected_package_hashes.items():
        if package_manifest.get(packaged_path) != expected_hash:
            raise ValueError(f"package manifest pin mismatch: {packaged_path}")


def validate_run_identity(facts: dict[str, str], inventory: dict[str, Any]) -> int:
    expected_facts = {
        "schema": RETURN_SCHEMA,
        "candidate_revision": CANDIDATE_REVISION,
        "vf64_revision": VF64_REVISION,
        "build_type": "Release",
        "CUMETAL_BUILD_TESTS": "OFF",
        "CUMETAL_ENABLE_CUDA_REGISTRATION": "ON",
        "CUMETAL_ENABLE_BINARY_SHIM": "OFF",
        "CUMETAL_CUDA_ARCH": "sm_86",
        "CUMETAL_FP64_MODE": "ieee64",
        "network_operations": "none",
        "install_update_sudo_operations": "none",
    }
    for key, expected in expected_facts.items():
        if facts.get(key) != expected:
            raise ValueError(f"unexpected native fact {key}: {facts.get(key)!r}")
    for key in ("started_utc", "ended_utc"):
        if not UTC.fullmatch(facts.get(key, "")):
            raise ValueError(f"invalid native UTC fact {key}: {facts.get(key)!r}")
    try:
        runner_exit = int(facts["runner_exit_code"])
    except (KeyError, ValueError) as exc:
        raise ValueError("missing or invalid runner_exit_code") from exc
    if runner_exit not in {0, 1, 77}:
        raise ValueError(f"unexpected runner exit: {runner_exit}")

    machine = inventory.get("machine")
    if not isinstance(machine, dict):
        raise ValueError("normalized inventory lacks machine facts")
    comparisons = {
        "uname_machine": machine.get("arch"),
        "macos_product_version": machine.get("os_version"),
        "macos_build": machine.get("os_build"),
        "developer_directory": machine.get("developer_directory"),
        "machine_model": machine.get("model"),
        "sdk_version": machine.get("sdk_version"),
        "metal_compiler": machine.get("metal_compiler"),
    }
    for fact_key, inventory_value in comparisons.items():
        if not observations_match(facts.get(fact_key), inventory_value):
            raise ValueError(
                f"native run does not match bound inventory for {fact_key}: "
                f"run={facts.get(fact_key)!r}, inventory={inventory_value!r}"
            )
    if facts.get("uname_system") != "Darwin":
        raise ValueError(f"bound native run is not Darwin: {facts.get('uname_system')!r}")
    run_xcode = canonical_version(facts.get("xcode_version"))
    inventory_xcode = canonical_version(machine.get("xcode_version"))
    if not observations_match(run_xcode, inventory_xcode):
        raise ValueError("native Xcode version does not match bound inventory")
    return runner_exit


def stage_record(archive: Path, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": event["status"],
        "exit_code": event["exit_code"],
        "log": return_ref(archive, event["log"]),
    }


def _inventory_machine(inventory: dict[str, Any]) -> dict[str, Any]:
    machine = inventory["machine"]
    raw_devices = machine.get("metal_devices")
    if not isinstance(raw_devices, list):
        raise ValueError("normalized inventory Metal devices must be an array")
    devices = []
    for index, device in enumerate(raw_devices):
        if not isinstance(device, dict) or not isinstance(device.get("name"), str) \
                or not isinstance(device.get("registry_id"), str) \
                or not device["name"] or not device["registry_id"]:
            raise ValueError(f"invalid normalized Metal device {index}")
        devices.append({"name": device["name"], "registry_id": device["registry_id"]})
    return {
        "os_name": machine.get("os_name"),
        "os_version": machine.get("os_version"),
        "os_build": machine.get("os_build"),
        "arch": machine.get("arch"),
        "model": machine.get("model"),
        "developer_directory": machine.get("developer_directory"),
        "xcode_version": machine.get("xcode_version"),
        "sdk_version": machine.get("sdk_version"),
        "metal_compiler": machine.get("metal_compiler"),
        "metal_devices": devices,
    }


def _assertion_value(
    assertions: dict[str, list[dict[str, Any]]], case_id: str, assertion_id: str
) -> bool | None:
    for record in assertions.get(case_id, []):
        if record["id"] == assertion_id:
            return bool(record["passed"])
    return None


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(case["classification"] for case in cases)
    required_failures = sum(
        case["required"] and case["classification"] == "FAIL" for case in cases
    )
    required_not_run = sum(
        case["required"] and case["classification"] == "NOT_RUN" for case in cases
    )
    return {
        "case_count": len(cases),
        "counts": {name: counts[name] for name in CLASSIFICATIONS},
        "required_failures": required_failures,
        "all_required_attempted": required_not_run == 0,
        "record_status": (
            "FAIL" if required_failures else "INCOMPLETE" if required_not_run else "COMPLETE"
        ),
    }


def analyze(
    archive: Path, inventory: dict[str, Any], fixtures: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    files, archive_validation = load_verified_files(archive)
    if fixtures.get("schema") != "cuda4as-m1-fixture-manifest-v1":
        raise ValueError("unexpected fixture manifest schema")
    if fixtures.get("candidate", {}).get("revision") != CANDIDATE_REVISION:
        raise ValueError("fixture manifest candidate revision mismatch")
    fixture_list = fixtures.get("cases")
    if not isinstance(fixture_list, list):
        raise ValueError("fixture cases must be an array")
    fixture_cases = {
        case["id"]: case for case in fixture_list if isinstance(case, dict) and "id" in case
    }
    if set(fixture_cases) != set(CASE_OUTPUTS) or len(fixture_cases) != len(fixture_list):
        raise ValueError("fixture case IDs do not match the native runner")

    validate_inventory_binding(files, inventory, fixtures)
    facts_rows = parse_tsv(files["facts.tsv"], ("key", "value"), "facts.tsv")
    facts = unique_map(facts_rows, "key", "facts.tsv")
    runner_exit = validate_run_identity(facts, inventory)
    events, _ = parse_events(files, fixture_cases)
    assertions = parse_assertions(files)
    for required_assertion in (
        "fixture_inputs_match_before_run",
        "fixture_inputs_unchanged_after_run",
    ):
        if _assertion_value(assertions, "_package", required_assertion) is None:
            raise ValueError(f"missing package assertion: {required_assertion}")
    inputs_unchanged = all(
        _assertion_value(assertions, "_package", item) is True
        for item in (
            "fixture_inputs_match_before_run",
            "fixture_inputs_unchanged_after_run",
        )
    )
    source_event = events.get(("_candidate", "source_integrity"))
    source_state = (
        "clean"
        if source_event and source_event["status"] == "PASS"
        else "dirty"
        if source_event and source_event["status"] == "FAIL"
        else "not_present"
    )
    gaps = [
        line.strip()
        for line in files["environment-gaps.txt"].decode("utf-8").splitlines()
        if line.strip()
    ]

    inventory_devices = {
        item["name"] for item in _inventory_machine(inventory)["metal_devices"]
    }
    cases: list[dict[str, Any]] = []
    for case_id in CASE_OUTPUTS:
        fixture = fixture_cases[case_id]
        required_stages = list(fixture["future_required_stages"])
        stages = {
            stage: stage_record(archive, events[(case_id, stage)])
            for stage in required_stages
        }
        launch_event = events[(case_id, "launch")]
        execution_attempted = launch_event["status"] != "NOT_RUN"
        backend_member = launch_event["log"] if execution_attempted else None
        provenance, provenance_errors = (
            provenance_from_log(files[backend_member])
            if backend_member is not None
            else ([], [])
        )
        expected_kernel = EXPECTED_KERNELS[case_id]
        allowed_sources = EXPECTED_LOWERING_SOURCES[case_id]
        provenance_ok = not provenance_errors and bool(provenance) and all(
            record["kernel"] == expected_kernel
            and record["device"] == "apple_gpu"
            and record["device_name"] in inventory_devices
            and record["source"] in allowed_sources
            and record["semantic_quality"] == "exact"
            and record["launch_success"] is True
            and record["completed"] is True
            for record in provenance
        )
        if execution_attempted and not provenance_ok and stages["launch"]["status"] == "PASS":
            stages["launch"]["status"] = "FAIL"

        validation_event = events[(case_id, "validation")]
        validation_attempted = validation_event["status"] != "NOT_RUN"
        expected = fixture["expected_output"]
        output_member = CASE_OUTPUTS[case_id]
        output_data = files.get(output_member)
        actual = {
            "bytes": len(output_data) if output_data is not None else None,
            "sha256": sha256(output_data) if output_data is not None else None,
        }
        expected_side = {"bytes": expected["bytes"], "sha256": expected["sha256"]}
        exact_output = actual == expected_side

        case_assertions = list(assertions.get(case_id, []))
        if execution_attempted:
            case_assertions.append(
                {"id": "analyzer.gpu_provenance_parse", "passed": not provenance_errors}
            )
            case_assertions.append(
                {"id": "analyzer.gpu_provenance_inventory_match", "passed": provenance_ok}
            )
        if validation_attempted:
            case_assertions.append(
                {"id": "analyzer.returned_output_exact", "passed": exact_output}
            )
        if any(record["status"] != "NOT_RUN" for record in stages.values()):
            case_assertions.append(
                {"id": "analyzer.application_inputs_unchanged", "passed": inputs_unchanged}
            )
        returned_assertions_ok = bool(case_assertions) and all(
            item["passed"] is True for item in case_assertions
        )
        if validation_attempted and (
            validation_event["status"] != "PASS" or not exact_output or not returned_assertions_ok
        ):
            stages["validation"]["status"] = "FAIL"
        validation_status = (
            "NOT_RUN"
            if not validation_attempted
            else "PASS"
            if stages["validation"]["status"] == "PASS"
            else "FAIL"
        )
        required_pass = all(stages[stage]["status"] == "PASS" for stage in required_stages)
        pass_gpu = (
            inputs_unchanged
            and required_pass
            and execution_attempted
            and launch_event["exit_code"] == 0
            and provenance_ok
            and validation_attempted
            and validation_status == "PASS"
            and returned_assertions_ok
            and exact_output
        )
        failure_evidence = any(
            record["status"] == "FAIL" for record in stages.values()
        ) or any(item["passed"] is False for item in case_assertions)
        if execution_attempted and not pass_gpu:
            failure_evidence = True

        if pass_gpu:
            classification = "PASS_GPU"
            reason = (
                "All required stages, exact output, and inventoried Apple-GPU "
                "provenance passed."
            )
        elif failure_evidence:
            classification = "FAIL"
            failed_stages = [
                stage for stage, record in stages.items() if record["status"] == "FAIL"
            ]
            failed_assertions = [
                item["id"] for item in case_assertions if item["passed"] is False
            ]
            reason = (
                "Native feasibility evidence failed; "
                f"failed_stages={failed_stages}, failed_assertions={failed_assertions}, "
                f"provenance_parse_errors={provenance_errors[:3]}."
            )
        elif runner_exit == 77:
            if execution_attempted:
                raise ValueError("environment skip contains case execution")
            if not gaps:
                raise ValueError("environment skip has no recorded gaps")
            classification = "SKIP_ENVIRONMENT"
            reason = "Native run stopped at preflight because existing prerequisites were missing."
        else:
            classification = "NOT_RUN"
            reason = (
                "The case was not executed; inspect candidate, adapter, and stage logs "
                "in the returned archive for the preceding gate."
            )

        source_files = [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in fixture["source_files"] + fixture["build_files"]
        ]
        cases.append(
            {
                "id": case_id,
                "case_type": CASE_TYPES[case_id],
                "required": True,
                "classification": classification,
                "source": {
                    "files": source_files,
                    "application_source_unchanged": (
                        inputs_unchanged
                        if any(record["status"] != "NOT_RUN" for record in stages.values())
                        else None
                    ),
                    "application_build_unchanged": (
                        inputs_unchanged
                        if any(record["status"] != "NOT_RUN" for record in stages.values())
                        else None
                    ),
                    "diff_path": None,
                },
                "required_stages": required_stages,
                "stages": stages,
                "execution": {
                    "attempted": execution_attempted,
                    "route": "apple_gpu"
                    if any(item["device"] == "apple_gpu" for item in provenance)
                    else "none",
                    "explicit_cpu_selection": False,
                    "exit_code": launch_event["exit_code"] if execution_attempted else None,
                    "backend_evidence_log": return_ref(archive, backend_member),
                    "gpu_provenance": provenance,
                },
                "validation": {
                    "attempted": validation_attempted,
                    "status": validation_status,
                    "comparison": "exact",
                    "expected": expected_side,
                    "actual": actual,
                    "assertions": case_assertions,
                },
                "diagnostic_assertions": [],
                "environment_gaps": gaps if classification == "SKIP_ENVIRONMENT" else [],
                "reason": reason,
            }
        )

    created_utc = facts["ended_utc"]
    document = {
        "schema": RESULT_SCHEMA,
        "run_id": "m1-native-" + facts["started_utc"].replace(":", "").replace("-", ""),
        "created_utc": created_utc,
        "contract": {
            "id": "cuda4as-m0-initial-ampere-source-profile",
            "revision": "0.1",
            "corpus_schema_version": 1,
        },
        "candidate": {
            "repository": "https://github.com/Lulzx/cuda-metal.git",
            "revision": CANDIDATE_REVISION,
            "source_state": source_state,
            "build_type": "Release",
            "options": {
                "CUMETAL_BUILD_TESTS": "OFF",
                "CUMETAL_ENABLE_CUDA_REGISTRATION": "ON",
                "CUMETAL_ENABLE_BINARY_SHIM": "OFF",
                "CUMETAL_CUDA_ARCH": "sm_86",
                "CUMETAL_FP64_MODE": "ieee64",
            },
        },
        "machine": _inventory_machine(inventory),
        "cases": cases,
        "summary": _summary(cases),
    }
    errors = validate_document(document)
    if errors:
        raise ValueError("normalized result failed its normative validator: " + "; ".join(errors))
    return document, archive_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="untouched native return .tgz")
    parser.add_argument(
        "--inventory", type=Path, required=True, help="validated normalized inventory JSON"
    )
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"archive does not exist: {args.archive}")
    try:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
        document, validation = analyze(args.archive, inventory, fixtures)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        print(f"INVALID {args.archive}: {exc}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    counts = document["summary"]["counts"]
    print(
        f"VALID {args.archive} -> {args.output}; "
        f"PASS_GPU={counts['PASS_GPU']} FAIL={counts['FAIL']} "
        f"SKIP_ENVIRONMENT={counts['SKIP_ENVIRONMENT']} "
        f"NOT_RUN={counts['NOT_RUN']}; archive_sha256={validation['archive_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
