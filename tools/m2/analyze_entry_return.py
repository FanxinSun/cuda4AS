#!/usr/bin/env python3
"""Validate the bounded cuda4AS M2 entry-retry return without extraction."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

try:
    from tools.m1.analyze_native_return import (
        CASE_OUTPUTS,
        CASE_TYPES,
        EXPECTED_KERNELS,
        EXPECTED_LOWERING_SOURCES,
        _inventory_machine,
        _safe_name,
        canonical_version,
        load_json,
        load_verified_files,
        observations_match,
        parse_assertions,
        parse_checksum_manifest,
        parse_events,
        provenance_from_log,
        return_ref,
        sha256,
        unique_map,
        parse_tsv,
    )
except ModuleNotFoundError:  # Direct invocation from the repository root.
    import sys

    sys.path.insert(0, str(ROOT))
    from tools.m1.analyze_native_return import (
        CASE_OUTPUTS,
        CASE_TYPES,
        EXPECTED_KERNELS,
        EXPECTED_LOWERING_SOURCES,
        _inventory_machine,
        _safe_name,
        canonical_version,
        load_json,
        load_verified_files,
        observations_match,
        parse_assertions,
        parse_checksum_manifest,
        parse_events,
        provenance_from_log,
        return_ref,
        sha256,
        unique_map,
        parse_tsv,
    )
FIXTURES_PATH = ROOT / "docs/m1/fixtures.json"
ARTIFACT_MANIFEST_PATH = ROOT / "docs/m1/native-feasibility-artifact.json"
CANDIDATE_REVISION = "f486e5ebcfd381d06e3297afd65dbcbd5006a902"
VF64_REVISION = "729021777455da72db8809d9ef1269c677d88b3f"
RETURN_SCHEMA = "cuda4as-m2-native-return-v1"
RESULT_SCHEMA = "cuda4as-m2-entry-result-v1"
INVENTORY_SCHEMA = "cuda4as-m1-mac-inventory-normalized-v1"
BASE_ARTIFACT_SHA256 = "dbb390b4f470b8f286ccb65a2b8565a235e749d9122c16e8e92ade96e0099bc7"
BASE_ARTIFACT_BYTES = 8549223
PATCH_ID = "cumetal-lower_to_llvm-direct-array-include-v1"
PATCH_SHA256 = "a192690749e55a95e9c826915c0b116fb72e604110194ca852378858c37fd9e2"
PATCH_PATH = "candidate/lower_to_llvm-array.patch"
PATCH_TARGET = "compiler/ptx/src/lower_to_llvm.cpp"
PATCH_ORIGINAL_SHA256 = "df01bfcd8e0774d166bc9b56548589c4dc1840cdcd8123f2ca9662fd3581fcb2"
PATCH_PATCHED_SHA256 = "4db44ce15f96d9066277d646c113676941685dba32f3c899d5b0b43ddca6678a"
CLEAN_TREE_SHA256 = "b5ee0a6c8c695b4dc9e0c6527566340b18b338a223a2cd81392b81112b96a37c"
PATCHED_TREE_SHA256 = "504067b6724d218a2a77182f004045b0f10069674b67e4dd02a5f1c85d8a8de0"
PATCH_BINDING_SCHEMA = "cuda4as-m2-candidate-patch-binding-v1"
UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _base_manifest() -> dict[str, str]:
    artifact = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if artifact.get("schema") != "cuda4as-m1-artifact-manifest-v1" or artifact.get(
        "artifact_id"
    ) != "cuda4as-m1-native-feasibility-v1":
        raise ValueError("unexpected bound M1 artifact manifest")
    records = artifact.get("files")
    if not isinstance(records, list):
        raise ValueError("M1 artifact manifest has no files")
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("malformed M1 artifact manifest record")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("malformed M1 artifact manifest path/hash")
        path = _safe_name(path)
        if path in result:
            raise ValueError(f"duplicate M1 artifact member: {path}")
        result[path] = digest
    result.pop("PACKAGE-MANIFEST.sha256", None)
    return result


def _validate_manifest(files: dict[str, bytes], manifest_name: str, label: str) -> None:
    expected = parse_checksum_manifest(files[manifest_name], label)
    actual = set(files) - {manifest_name}
    if set(expected) != actual:
        raise ValueError(
            f"{label} member mismatch: missing={sorted(set(expected)-actual)}, "
            f"extra={sorted(actual-set(expected))}"
        )
    for name, digest in expected.items():
        if sha256(files[name]) != digest:
            raise ValueError(f"{label} hash mismatch: {name}")


def _delta_files(files: dict[str, bytes]) -> dict[str, bytes]:
    prefix = "package/m2-delta/"
    result = {name[len(prefix) :]: data for name, data in files.items() if name.startswith(prefix)}
    if "PACKAGE-MANIFEST.sha256" not in result:
        raise ValueError("returned archive lacks package/m2-delta manifest")
    _validate_manifest(result, "PACKAGE-MANIFEST.sha256", "M2 delta manifest")
    return result


def validate_package_and_patch(files: dict[str, bytes], inventory: dict[str, Any]) -> dict[str, Any]:
    package_manifest = parse_checksum_manifest(
        files["package/PACKAGE-MANIFEST.sha256"], "base package manifest"
    )
    if package_manifest != _base_manifest():
        missing = sorted(set(_base_manifest()) - set(package_manifest))
        extra = sorted(set(package_manifest) - set(_base_manifest()))
        changed = sorted(
            path
            for path in set(package_manifest) & set(_base_manifest())
            if package_manifest[path] != _base_manifest()[path]
        )
        raise ValueError(f"base M1 package manifest mismatch: missing={missing}, extra={extra}, changed={changed}")

    binding = load_json(files["package/target-inventory-binding.json"], "inventory binding")
    if binding.get("schema") != "cuda4as-m1-inventory-binding-v1" or binding.get(
        "status"
    ) != "BOUND_TO_RETURNED_INVENTORY":
        raise ValueError("returned base package is not inventory-bound")
    source_archive = inventory.get("source_archive")
    if binding.get("inventory_archive") != source_archive:
        raise ValueError("returned base package inventory binding mismatch")

    delta = _delta_files(files)
    patch_binding = load_json(delta["candidate/patch-binding.json"], "patch binding")
    if patch_binding.get("schema") != PATCH_BINDING_SCHEMA or patch_binding.get(
        "status"
    ) != "BOUND_TO_SINGLE_DISCLOSED_PATCH":
        raise ValueError("unexpected M2 patch binding")
    if patch_binding.get("patch_id") != PATCH_ID:
        raise ValueError("unexpected M2 patch ID")
    base = patch_binding.get("base_artifact")
    if not isinstance(base, dict) or base.get("bytes") != BASE_ARTIFACT_BYTES or base.get(
        "sha256"
    ) != BASE_ARTIFACT_SHA256:
        raise ValueError("M2 patch binding does not bind the M1 artifact")
    candidate = patch_binding.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("revision") != CANDIDATE_REVISION:
        raise ValueError("M2 patch binding candidate revision mismatch")
    patch = patch_binding.get("patch")
    if not isinstance(patch, dict):
        raise ValueError("M2 patch binding has no patch record")
    expected_patch = {
        "path": PATCH_PATH,
        "bytes": 232,
        "sha256": PATCH_SHA256,
        "hunks": 1,
        "changed_files": [PATCH_TARGET],
        "source_path": PATCH_TARGET,
        "original_sha256": PATCH_ORIGINAL_SHA256,
        "patched_sha256": PATCH_PATCHED_SHA256,
    }
    for key, expected in expected_patch.items():
        if patch.get(key) != expected:
            raise ValueError(f"M2 patch binding mismatch for {key}")
    patch_data = delta[PATCH_PATH]
    if sha256(patch_data) != PATCH_SHA256 or len(patch_data) != 232:
        raise ValueError("M2 patch bytes/hash mismatch")
    patch_text = patch_data.decode("utf-8")
    if (
        patch_text.count("@@") != 2
        or patch_text.count("+#include <array>") != 1
        or patch_text.count("--- a/" + PATCH_TARGET) != 1
        or patch_text.count("+++ b/" + PATCH_TARGET) != 1
        or any(line.startswith("-") and not line.startswith("---") for line in patch_text.splitlines())
    ):
        raise ValueError("M2 patch is not the one disclosed include-only hunk")
    trees = patch_binding.get("tree_identity")
    if not isinstance(trees, dict) or trees.get("clean_manifest_sha256") != CLEAN_TREE_SHA256 \
        or trees.get("patched_manifest_sha256") != PATCHED_TREE_SHA256 \
        or trees.get("clean_files") != 1065 or trees.get("patched_files") != 1065 \
        or trees.get("clean_bytes") != 16111130 or trees.get("patched_bytes") != 16111147:
        raise ValueError("M2 clean/patched tree identity mismatch")
    for name, expected in (
        ("candidate/combined-tree-clean.sha256", CLEAN_TREE_SHA256),
        ("candidate/combined-tree-patched.sha256", PATCHED_TREE_SHA256),
        ("candidate/combined-tree-clean-files.txt", "aebda5a6bb0db8542aa46a48090bc710e7e79df2b97837e62cc5e32e7377300b"),
        ("candidate/combined-tree-patched-files.txt", "aebda5a6bb0db8542aa46a48090bc710e7e79df2b97837e62cc5e32e7377300b"),
    ):
        if sha256(delta[name]) != expected:
            raise ValueError(f"M2 tree identity file mismatch: {name}")
    return {
        "patch_binding": patch_binding,
        "delta_manifest_sha256": sha256(delta["PACKAGE-MANIFEST.sha256"]),
        "patch_binding_sha256": sha256(delta["candidate/patch-binding.json"]),
    }


def validate_identity(facts: dict[str, str], inventory: dict[str, Any]) -> int:
    expected = {
        "schema": RETURN_SCHEMA,
        "m2_entry_gate": "single_candidate_patch_retry",
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
        "patch_id": PATCH_ID,
        "patch_path": PATCH_PATH,
        "patch_sha256": PATCH_SHA256,
        "patch_changed_files": PATCH_TARGET,
        "patch_original_sha256": PATCH_ORIGINAL_SHA256,
        "patch_patched_sha256": PATCH_PATCHED_SHA256,
        "clean_tree_manifest_sha256": CLEAN_TREE_SHA256,
        "patched_tree_manifest_sha256": PATCHED_TREE_SHA256,
        "base_artifact_sha256": BASE_ARTIFACT_SHA256,
        "base_artifact_bytes": str(BASE_ARTIFACT_BYTES),
    }
    for key, value in expected.items():
        if facts.get(key) != value:
            raise ValueError(f"unexpected native fact {key}: {facts.get(key)!r}")
    for key in ("started_utc", "ended_utc"):
        if not UTC.fullmatch(facts.get(key, "")):
            raise ValueError(f"invalid native UTC fact {key}")
    try:
        runner_exit = int(facts["runner_exit_code"])
    except (KeyError, ValueError) as exc:
        raise ValueError("missing or invalid runner_exit_code") from exc
    if runner_exit not in {0, 1, 77}:
        raise ValueError(f"unexpected runner exit: {runner_exit}")
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("unexpected normalized inventory schema")
    validation = inventory.get("validation")
    if not isinstance(validation, dict) or validation.get("archive_members_safe") is not True \
        or validation.get("internal_manifest_valid") is not True:
        raise ValueError("normalized inventory lacks successful archive validation")
    machine = inventory.get("machine")
    if not isinstance(machine, dict):
        raise ValueError("inventory lacks machine facts")
    if machine.get("arch") != "arm64" or not isinstance(machine.get("metal_devices"), list) \
        or not machine.get("metal_devices"):
        raise ValueError("bound inventory lacks an Apple-Silicon Metal device")
    comparisons = {
        "uname_machine": machine.get("arch"),
        "macos_product_version": machine.get("os_version"),
        "macos_build": machine.get("os_build"),
        "developer_directory": machine.get("developer_directory"),
        "machine_model": machine.get("model"),
        "sdk_version": machine.get("sdk_version"),
        "metal_compiler": machine.get("metal_compiler"),
    }
    for key, value in comparisons.items():
        if not observations_match(facts.get(key), value):
            raise ValueError(f"native run does not match inventory for {key}")
    if facts.get("uname_system") != "Darwin":
        raise ValueError("native run is not Darwin")
    if not observations_match(canonical_version(facts.get("xcode_version")), canonical_version(machine.get("xcode_version"))):
        raise ValueError("native Xcode version does not match inventory")
    return runner_exit


def _stage(event: dict[str, Any], archive: Path) -> dict[str, Any]:
    return {"status": event["status"], "exit_code": event["exit_code"], "log": return_ref(archive, event["log"])}


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(case["classification"] for case in cases)
    required_not_run = sum(case["classification"] == "NOT_RUN" for case in cases)
    required_fail = sum(case["classification"] == "FAIL" for case in cases)
    return {
        "case_count": len(cases),
        "counts": {name: counts[name] for name in ("PASS_GPU", "FAIL", "SKIP_ENVIRONMENT", "NOT_RUN")},
        "required_failures": required_fail,
        "all_required_attempted": required_not_run == 0,
        "record_status": "FAIL" if required_fail else "INCOMPLETE" if required_not_run else "COMPLETE",
    }


def _parse_m2_events(
    files: dict[str, bytes], fixture_cases: dict[str, dict[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], bool]:
    """Parse the published runner's one known package-log alias safely.

    The first published M2 runner recorded the patch-binding event with the
    delta-relative path, while the returned archive stores that file beneath
    ``package/m2-delta``.  Accept that exact, unambiguous alias so an already
    completed Mac run remains usable; the runner is repaired separately.
    """

    try:
        events, _ = parse_events(files, fixture_cases)
        return events, False
    except ValueError as exc:
        marker = b"\tcandidate/patch-binding.json\t"
        replacement = b"\tpackage/m2-delta/candidate/patch-binding.json\t"
        raw = files["case-stage-events.tsv"]
        if "event references missing log: candidate/patch-binding.json" not in str(exc) or raw.count(marker) != 1:
            raise
        repaired = dict(files)
        repaired["case-stage-events.tsv"] = raw.replace(marker, replacement)
        events, _ = parse_events(repaired, fixture_cases)
        return events, True


def analyze(archive: Path, inventory: dict[str, Any], fixtures: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    files, archive_validation = load_verified_files(archive)
    if fixtures.get("schema") != "cuda4as-m1-fixture-manifest-v1" or fixtures.get("candidate", {}).get("revision") != CANDIDATE_REVISION:
        raise ValueError("unexpected immutable fixture manifest")
    fixture_list = fixtures.get("cases")
    if not isinstance(fixture_list, list):
        raise ValueError("fixture cases are not a list")
    fixture_cases = {case["id"]: case for case in fixture_list if isinstance(case, dict) and "id" in case}
    if set(fixture_cases) != set(CASE_OUTPUTS) or len(fixture_cases) != len(fixture_list):
        raise ValueError("fixture case IDs do not match the enrolled cases")
    package = validate_package_and_patch(files, inventory)
    facts = unique_map(parse_tsv(files["facts.tsv"], ("key", "value"), "facts.tsv"), "key", "facts.tsv")
    runner_exit = validate_identity(facts, inventory)
    events, event_log_alias_repaired = _parse_m2_events(files, fixture_cases)
    assertions = parse_assertions(files)
    package_assertions = {item["id"]: item["passed"] for item in assertions.get("_package", [])}
    for key in ("fixture_inputs_match_before_run", "fixture_inputs_unchanged_after_run"):
        if key not in package_assertions:
            raise ValueError(f"missing package assertion: {key}")
    inputs_unchanged = all(package_assertions[key] for key in ("fixture_inputs_match_before_run", "fixture_inputs_unchanged_after_run"))

    candidate_keys = [
        ("_package", "m2_delta_integrity"),
        ("_package", "patch_binding"),
        ("_candidate", "source_integrity_clean"),
        ("_candidate", "candidate_patch"),
        ("_candidate", "source_integrity_patched"),
    ]
    candidate_gate = {
        f"{case}/{stage}": {"status": events[(case, stage)]["status"], "exit_code": events[(case, stage)]["exit_code"], "log": return_ref(archive, events[(case, stage)]["log"])}
        for case, stage in candidate_keys
        if (case, stage) in events
    }
    candidate_gate_complete = len(candidate_gate) == len(candidate_keys) and all(item["status"] == "PASS" for item in candidate_gate.values())
    candidate_attempted = ("_candidate", "source_integrity_clean") in events
    gaps = [line.strip() for line in files["environment-gaps.txt"].decode("utf-8").splitlines() if line.strip()]
    inventory_devices = {item["name"] for item in _inventory_machine(inventory)["metal_devices"]}
    cases: list[dict[str, Any]] = []
    for case_id in CASE_OUTPUTS:
        fixture = fixture_cases[case_id]
        stages = {stage: _stage(events[(case_id, stage)], archive) for stage in fixture["future_required_stages"]}
        launch = events[(case_id, "launch")]
        execution_attempted = launch["status"] != "NOT_RUN"
        backend_member = launch["log"] if execution_attempted else None
        provenance, provenance_errors = provenance_from_log(files[backend_member]) if backend_member else ([], [])
        expected_kernel = EXPECTED_KERNELS[case_id]
        allowed_sources = EXPECTED_LOWERING_SOURCES[case_id]
        provenance_ok = not provenance_errors and bool(provenance) and all(
            item["kernel"] == expected_kernel and item["device"] == "apple_gpu" and item["device_name"] in inventory_devices
            and item["source"] in allowed_sources and item["semantic_quality"] == "exact" and item["launch_success"] is True and item["completed"] is True
            for item in provenance
        )
        if execution_attempted and not provenance_ok and stages["launch"]["status"] == "PASS":
            stages["launch"]["status"] = "FAIL"
        validation_event = events[(case_id, "validation")]
        validation_attempted = validation_event["status"] != "NOT_RUN"
        output_data = files.get(CASE_OUTPUTS[case_id])
        expected = fixture["expected_output"]
        actual = {"bytes": len(output_data) if output_data is not None else None, "sha256": sha256(output_data) if output_data is not None else None}
        exact_output = actual == {"bytes": expected["bytes"], "sha256": expected["sha256"]}
        case_assertions = list(assertions.get(case_id, []))
        if execution_attempted:
            case_assertions.extend([
                {"id": "analyzer.gpu_provenance_parse", "passed": not provenance_errors},
                {"id": "analyzer.gpu_provenance_inventory_match", "passed": provenance_ok},
            ])
        if validation_attempted:
            case_assertions.append({"id": "analyzer.returned_output_exact", "passed": exact_output})
        if any(item["status"] != "NOT_RUN" for item in stages.values()):
            case_assertions.append({"id": "analyzer.application_inputs_unchanged", "passed": inputs_unchanged})
        returned_assertions_ok = bool(case_assertions) and all(item["passed"] is True for item in case_assertions)
        required_pass = all(item["status"] == "PASS" for item in stages.values())
        pass_gpu = candidate_gate_complete and inputs_unchanged and required_pass and execution_attempted and launch["exit_code"] == 0 and provenance_ok and validation_attempted and validation_event["status"] == "PASS" and returned_assertions_ok and exact_output
        failure_evidence = any(item["status"] == "FAIL" for item in stages.values()) or any(not item["passed"] for item in case_assertions)
        if execution_attempted and not pass_gpu:
            failure_evidence = True
        if pass_gpu:
            classification, reason = "PASS_GPU", "All case stages, exact output, inventory-bound Apple-GPU provenance, and the single patch gate passed."
        elif failure_evidence:
            classification, reason = "FAIL", "Native case evidence failed; inspect returned stage, assertion, and provenance records."
        elif runner_exit == 77:
            if execution_attempted or not gaps:
                raise ValueError("invalid environment-gap return")
            classification, reason = "SKIP_ENVIRONMENT", "Native run stopped at preflight because an existing prerequisite was unavailable."
        else:
            classification, reason = "NOT_RUN", "The case was not executed; the candidate or adapter gate stopped first."
        cases.append({
            "id": case_id, "case_type": CASE_TYPES[case_id], "required": True, "classification": classification,
            "source": {"files": [{"path": item["path"], "sha256": item["sha256"]} for item in fixture["source_files"] + fixture["build_files"]], "application_source_unchanged": inputs_unchanged if any(item["status"] != "NOT_RUN" for item in stages.values()) else None, "application_build_unchanged": inputs_unchanged if any(item["status"] != "NOT_RUN" for item in stages.values()) else None, "diff_path": None},
            "required_stages": list(fixture["future_required_stages"]), "stages": stages,
            "execution": {"attempted": execution_attempted, "route": "apple_gpu" if any(item["device"] == "apple_gpu" for item in provenance) else "none", "explicit_cpu_selection": False, "exit_code": launch["exit_code"] if execution_attempted else None, "backend_evidence_log": return_ref(archive, backend_member), "gpu_provenance": provenance},
            "validation": {"attempted": validation_attempted, "status": "PASS" if validation_attempted and validation_event["status"] == "PASS" and exact_output and returned_assertions_ok else "FAIL" if validation_attempted else "NOT_RUN", "comparison": "exact", "expected": {"bytes": expected["bytes"], "sha256": expected["sha256"]}, "actual": actual, "assertions": case_assertions},
            "diagnostic_assertions": [], "environment_gaps": gaps if classification == "SKIP_ENVIRONMENT" else [], "reason": reason,
        })

    document = {
        "schema": RESULT_SCHEMA,
        "run_id": "m2-entry-" + facts["started_utc"].replace(":", "").replace("-", ""),
        "created_utc": facts["ended_utc"],
        "contract": {"id": "cuda4as-m2-entry-gate", "revision": "1.0", "base_contract": "cuda4as-m1-result-v1"},
        "candidate": {"repository": "https://github.com/Lulzx/cuda-metal.git", "revision": CANDIDATE_REVISION, "vf64_revision": VF64_REVISION, "source_state": "patched" if candidate_gate_complete else "patch_not_validated", "patch_id": PATCH_ID, "patch_sha256": PATCH_SHA256, "patch_binding_sha256": package["patch_binding_sha256"], "clean_tree_manifest_sha256": CLEAN_TREE_SHA256, "patched_tree_manifest_sha256": PATCHED_TREE_SHA256, "build_type": "Release", "options": {"CUMETAL_BUILD_TESTS": "OFF", "CUMETAL_ENABLE_CUDA_REGISTRATION": "ON", "CUMETAL_ENABLE_BINARY_SHIM": "OFF", "CUMETAL_CUDA_ARCH": "sm_86", "CUMETAL_FP64_MODE": "ieee64"}},
        "candidate_gate": {"attempted": candidate_attempted, "complete_pass": candidate_gate_complete, "stages": candidate_gate},
        "contract_repairs": ["mapped published patch-binding event log to package/m2-delta/candidate/patch-binding.json"] if event_log_alias_repaired else [],
        "machine": _inventory_machine(inventory), "cases": cases, "summary": _summary(cases),
    }
    return document, archive_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
        document, validation = analyze(args.archive, inventory, fixtures)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"INVALID {args.archive}: {exc}")
        return 1
    counts = document["summary"]["counts"]
    print(f"VALID {args.archive} -> {args.output}; PASS_GPU={counts['PASS_GPU']} FAIL={counts['FAIL']} SKIP_ENVIRONMENT={counts['SKIP_ENVIRONMENT']} NOT_RUN={counts['NOT_RUN']}; archive_sha256={validation['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
