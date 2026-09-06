#!/usr/bin/env python3
"""Validate cuda4AS M1 result records without third-party dependencies."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any


SCHEMA_ID = "cuda4as-m1-result-v1"
CLASSIFICATIONS = (
    "PASS_GPU",
    "PASS_CPU_EXPLICIT",
    "FAIL",
    "UNSUPPORTED",
    "SKIP_ENVIRONMENT",
    "NOT_RUN",
)
STAGE_NAMES = (
    "configure",
    "host_compile",
    "device_compile",
    "device_link",
    "native_link",
    "install",
    "loader_discovery",
    "launch",
    "validation",
)
STAGE_STATUSES = ("PASS", "FAIL", "NOT_RUN", "NOT_APPLICABLE")
EXECUTION_ROUTES = ("apple_gpu", "cpu_explicit", "none")
VALIDATION_STATUSES = ("PASS", "FAIL", "NOT_RUN")
RECORD_STATUSES = ("COMPLETE", "INCOMPLETE", "FAIL")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_document(document: Any) -> list[str]:
    """Return all structural and cross-field validation errors."""

    errors: list[str] = []

    def error(path: str, message: str) -> None:
        errors.append(f"{path}: {message}")

    def require_keys(path: str, value: Any, keys: tuple[str, ...]) -> bool:
        if not isinstance(value, dict):
            error(path, "must be an object")
            return False
        for key in keys:
            if key not in value:
                error(f"{path}.{key}", "is required")
        return True

    if not require_keys(
        "$",
        document,
        (
            "schema",
            "run_id",
            "created_utc",
            "contract",
            "candidate",
            "machine",
            "cases",
            "summary",
        ),
    ):
        return errors

    if document.get("schema") != SCHEMA_ID:
        error("$.schema", f"must equal {SCHEMA_ID!r}")
    if not _nonempty_string(document.get("run_id")):
        error("$.run_id", "must be a nonempty string")
    if not isinstance(document.get("created_utc"), str) or not UTC_RE.fullmatch(
        document["created_utc"]
    ):
        error("$.created_utc", "must be UTC in YYYY-MM-DDTHH:MM:SSZ form")

    contract = document.get("contract")
    if require_keys("$.contract", contract, ("id", "revision", "corpus_schema_version")):
        if not _nonempty_string(contract.get("id")):
            error("$.contract.id", "must be a nonempty string")
        if not _nonempty_string(contract.get("revision")):
            error("$.contract.revision", "must be a nonempty string")
        if not isinstance(contract.get("corpus_schema_version"), int):
            error("$.contract.corpus_schema_version", "must be an integer")

    candidate = document.get("candidate")
    if require_keys(
        "$.candidate",
        candidate,
        ("repository", "revision", "source_state", "build_type", "options"),
    ):
        if not _nonempty_string(candidate.get("repository")):
            error("$.candidate.repository", "must be a nonempty string")
        if not isinstance(candidate.get("revision"), str) or not REVISION_RE.fullmatch(
            candidate["revision"]
        ):
            error("$.candidate.revision", "must be a lowercase 40-hex revision")
        if candidate.get("source_state") not in ("clean", "dirty", "not_present"):
            error("$.candidate.source_state", "must be clean, dirty, or not_present")
        if not _nonempty_string(candidate.get("build_type")):
            error("$.candidate.build_type", "must be a nonempty string")
        if not isinstance(candidate.get("options"), dict):
            error("$.candidate.options", "must be an object")

    machine = document.get("machine")
    machine_ok = require_keys(
        "$.machine",
        machine,
        (
            "os_name",
            "os_version",
            "os_build",
            "arch",
            "model",
            "developer_directory",
            "xcode_version",
            "sdk_version",
            "metal_compiler",
            "metal_devices",
        ),
    )
    if machine_ok and not isinstance(machine.get("metal_devices"), list):
        error("$.machine.metal_devices", "must be an array")
        machine_ok = False
    if not isinstance(machine, dict):
        machine = {}
    if machine_ok:
        for index, device in enumerate(machine["metal_devices"]):
            path = f"$.machine.metal_devices[{index}]"
            if not require_keys(path, device, ("name", "registry_id")):
                continue
            if not _nonempty_string(device.get("name")):
                error(f"{path}.name", "must be a nonempty string")
            if not _nonempty_string(device.get("registry_id")):
                error(f"{path}.registry_id", "must be a nonempty string")

    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        error("$.cases", "must be a nonempty array")
        cases = []

    case_ids: list[str] = []
    classifications: list[str] = []
    required_failures = 0
    required_not_run = 0

    for index, case in enumerate(cases):
        path = f"$.cases[{index}]"
        if not require_keys(
            path,
            case,
            (
                "id",
                "case_type",
                "required",
                "classification",
                "source",
                "required_stages",
                "stages",
                "execution",
                "validation",
                "diagnostic_assertions",
                "environment_gaps",
                "reason",
            ),
        ):
            continue

        case_id = case.get("id")
        if not _nonempty_string(case_id):
            error(f"{path}.id", "must be a nonempty string")
        else:
            case_ids.append(case_id)
        if not _nonempty_string(case.get("case_type")):
            error(f"{path}.case_type", "must be a nonempty string")
        if not isinstance(case.get("required"), bool):
            error(f"{path}.required", "must be boolean")

        classification = case.get("classification")
        if classification not in CLASSIFICATIONS:
            error(f"{path}.classification", f"must be one of {CLASSIFICATIONS}")
        else:
            classifications.append(classification)
            if case.get("required") and classification == "FAIL":
                required_failures += 1
            if case.get("required") and classification == "NOT_RUN":
                required_not_run += 1

        source = case.get("source")
        source_ok = require_keys(
            f"{path}.source",
            source,
            (
                "files",
                "application_source_unchanged",
                "application_build_unchanged",
                "diff_path",
            ),
        )
        if not isinstance(source, dict):
            source = {}
        if source_ok:
            files = source.get("files")
            if not isinstance(files, list) or not files:
                error(f"{path}.source.files", "must be a nonempty array")
            else:
                for file_index, file_record in enumerate(files):
                    file_path = f"{path}.source.files[{file_index}]"
                    if not require_keys(file_path, file_record, ("path", "sha256")):
                        continue
                    if not _nonempty_string(file_record.get("path")):
                        error(f"{file_path}.path", "must be nonempty")
                    if not isinstance(file_record.get("sha256"), str) or not SHA256_RE.fullmatch(
                        file_record["sha256"]
                    ):
                        error(f"{file_path}.sha256", "must be lowercase 64-hex")
            for key in ("application_source_unchanged", "application_build_unchanged"):
                if source.get(key) not in (True, False, None):
                    error(f"{path}.source.{key}", "must be boolean or null")
            if source.get("diff_path") is not None and not _nonempty_string(
                source.get("diff_path")
            ):
                error(f"{path}.source.diff_path", "must be null or nonempty")

        required_stages = case.get("required_stages")
        if not isinstance(required_stages, list):
            error(f"{path}.required_stages", "must be an array")
            required_stages = []
        elif len(required_stages) != len(
            {json.dumps(item, sort_keys=True, default=str) for item in required_stages}
        ):
            error(f"{path}.required_stages", "must not contain duplicates")
        valid_required_stages: list[str] = []
        for stage in required_stages:
            if not isinstance(stage, str) or stage not in STAGE_NAMES:
                error(f"{path}.required_stages", f"unknown stage {stage!r}")
            else:
                valid_required_stages.append(stage)

        stages = case.get("stages")
        if not isinstance(stages, dict):
            error(f"{path}.stages", "must be an object")
            stages = {}
        for stage in valid_required_stages:
            if stage not in stages:
                error(f"{path}.stages.{stage}", "required stage record is missing")
        for stage, record in stages.items():
            stage_path = f"{path}.stages.{stage}"
            if stage not in STAGE_NAMES:
                error(stage_path, "unknown stage")
                continue
            if not require_keys(stage_path, record, ("status", "exit_code", "log")):
                continue
            if record.get("status") not in STAGE_STATUSES:
                error(f"{stage_path}.status", f"must be one of {STAGE_STATUSES}")
            if record.get("exit_code") is not None and not isinstance(
                record.get("exit_code"), int
            ):
                error(f"{stage_path}.exit_code", "must be integer or null")
            if record.get("log") is not None and not _nonempty_string(record.get("log")):
                error(f"{stage_path}.log", "must be null or nonempty")

        execution = case.get("execution")
        execution_ok = require_keys(
            f"{path}.execution",
            execution,
            (
                "attempted",
                "route",
                "explicit_cpu_selection",
                "exit_code",
                "backend_evidence_log",
                "gpu_provenance",
            ),
        )
        if not isinstance(execution, dict):
            execution = {}
        if execution_ok:
            if not isinstance(execution.get("attempted"), bool):
                error(f"{path}.execution.attempted", "must be boolean")
            if execution.get("route") not in EXECUTION_ROUTES:
                error(f"{path}.execution.route", f"must be one of {EXECUTION_ROUTES}")
            if not isinstance(execution.get("explicit_cpu_selection"), bool):
                error(f"{path}.execution.explicit_cpu_selection", "must be boolean")
            if execution.get("exit_code") is not None and not isinstance(
                execution.get("exit_code"), int
            ):
                error(f"{path}.execution.exit_code", "must be integer or null")
            if execution.get("backend_evidence_log") is not None and not _nonempty_string(
                execution.get("backend_evidence_log")
            ):
                error(f"{path}.execution.backend_evidence_log", "must be null or nonempty")
            provenance = execution.get("gpu_provenance")
            if not isinstance(provenance, list):
                error(f"{path}.execution.gpu_provenance", "must be an array")
                provenance = []
            for prov_index, record in enumerate(provenance):
                prov_path = f"{path}.execution.gpu_provenance[{prov_index}]"
                if not require_keys(
                    prov_path,
                    record,
                    (
                        "kernel",
                        "device",
                        "device_name",
                        "source",
                        "semantic_quality",
                        "launch_success",
                        "completed",
                    ),
                ):
                    continue
                for key in ("kernel", "device", "device_name", "source", "semantic_quality"):
                    if not _nonempty_string(record.get(key)):
                        error(f"{prov_path}.{key}", "must be a nonempty string")
                for key in ("launch_success", "completed"):
                    if not isinstance(record.get(key), bool):
                        error(f"{prov_path}.{key}", "must be boolean")
        else:
            provenance = []

        validation = case.get("validation")
        validation_ok = require_keys(
            f"{path}.validation",
            validation,
            ("attempted", "status", "comparison", "expected", "actual", "assertions"),
        )
        if not isinstance(validation, dict):
            validation = {}
        assertions: list[Any] = []
        if validation_ok:
            if not isinstance(validation.get("attempted"), bool):
                error(f"{path}.validation.attempted", "must be boolean")
            if validation.get("status") not in VALIDATION_STATUSES:
                error(f"{path}.validation.status", f"must be one of {VALIDATION_STATUSES}")
            if validation.get("comparison") not in ("exact", "declared", "none"):
                error(f"{path}.validation.comparison", "must be exact, declared, or none")
            for side in ("expected", "actual"):
                record = validation.get(side)
                side_path = f"{path}.validation.{side}"
                if not require_keys(side_path, record, ("bytes", "sha256")):
                    continue
                if record.get("bytes") is not None and (
                    not isinstance(record.get("bytes"), int) or record["bytes"] < 0
                ):
                    error(f"{side_path}.bytes", "must be a nonnegative integer or null")
                if record.get("sha256") is not None and (
                    not isinstance(record.get("sha256"), str)
                    or not SHA256_RE.fullmatch(record["sha256"])
                ):
                    error(f"{side_path}.sha256", "must be lowercase 64-hex or null")
            assertions = validation.get("assertions")
            if not isinstance(assertions, list):
                error(f"{path}.validation.assertions", "must be an array")
                assertions = []
            for assertion_index, assertion in enumerate(assertions):
                assertion_path = f"{path}.validation.assertions[{assertion_index}]"
                if not require_keys(assertion_path, assertion, ("id", "passed")):
                    continue
                if not _nonempty_string(assertion.get("id")):
                    error(f"{assertion_path}.id", "must be nonempty")
                if not isinstance(assertion.get("passed"), bool):
                    error(f"{assertion_path}.passed", "must be boolean")

        diagnostics = case.get("diagnostic_assertions")
        if not isinstance(diagnostics, list):
            error(f"{path}.diagnostic_assertions", "must be an array")
            diagnostics = []
        for diag_index, assertion in enumerate(diagnostics):
            diag_path = f"{path}.diagnostic_assertions[{diag_index}]"
            if not require_keys(diag_path, assertion, ("id", "passed", "source_location")):
                continue
            if not _nonempty_string(assertion.get("id")):
                error(f"{diag_path}.id", "must be nonempty")
            if not isinstance(assertion.get("passed"), bool):
                error(f"{diag_path}.passed", "must be boolean")
            if assertion.get("source_location") is not None and not _nonempty_string(
                assertion.get("source_location")
            ):
                error(f"{diag_path}.source_location", "must be null or nonempty")

        environment_gaps = case.get("environment_gaps")
        if not isinstance(environment_gaps, list) or any(
            not _nonempty_string(item) for item in environment_gaps
        ):
            error(f"{path}.environment_gaps", "must be an array of nonempty strings")
            environment_gaps = []

        reason = case.get("reason")
        if not _nonempty_string(reason):
            error(f"{path}.reason", "must be a nonempty string")

        # Cross-field rules prevent a completed process or output marker from
        # being promoted into a compatibility pass without all required evidence.
        if classification in ("PASS_GPU", "PASS_CPU_EXPLICIT"):
            if candidate.get("source_state") != "clean":
                error("$.candidate.source_state", "must be clean when any case passes")
            if source.get("application_source_unchanged") is not True:
                error(f"{path}.source.application_source_unchanged", "must be true for a pass")
            if source.get("application_build_unchanged") is not True:
                error(f"{path}.source.application_build_unchanged", "must be true for a pass")
            for stage in valid_required_stages:
                stage_record = stages.get(stage)
                if not isinstance(stage_record, dict) or stage_record.get("status") != "PASS":
                    error(f"{path}.stages.{stage}.status", "must be PASS for a pass classification")
            if not validation.get("attempted") or validation.get("status") != "PASS":
                error(f"{path}.validation", "must be attempted with PASS status")
            if not assertions or any(
                not isinstance(item, dict) or item.get("passed") is not True
                for item in assertions
            ):
                error(f"{path}.validation.assertions", "must be nonempty and all pass")

        if classification == "PASS_GPU":
            minimum = {"device_compile", "native_link", "launch", "validation"}
            if not minimum.issubset(set(valid_required_stages)):
                error(f"{path}.required_stages", f"PASS_GPU requires at least {sorted(minimum)}")
            if not execution.get("attempted") or execution.get("route") != "apple_gpu":
                error(f"{path}.execution", "PASS_GPU requires attempted apple_gpu execution")
            if execution.get("explicit_cpu_selection") is not False:
                error(f"{path}.execution.explicit_cpu_selection", "must be false for PASS_GPU")
            if execution.get("exit_code") != 0:
                error(f"{path}.execution.exit_code", "must be zero for PASS_GPU")
            if not _nonempty_string(execution.get("backend_evidence_log")):
                error(f"{path}.execution.backend_evidence_log", "is required for PASS_GPU")
            successful_gpu = [
                item
                for item in provenance
                if isinstance(item, dict)
                and item.get("device") == "apple_gpu"
                and item.get("launch_success") is True
                and item.get("completed") is True
                and item.get("source") not in ("cpu_fallback", "stub", "approximate_stub")
                and item.get("semantic_quality") in ("exact", "declared")
            ]
            if not successful_gpu:
                error(f"{path}.execution.gpu_provenance", "needs a completed exact/declared Apple-GPU launch")
            if len(successful_gpu) != len(provenance):
                error(
                    f"{path}.execution.gpu_provenance",
                    "every PASS_GPU provenance record must be a completed exact/declared Apple-GPU launch",
                )
            device_names = {
                item.get("name") for item in machine.get("metal_devices", []) if isinstance(item, dict)
            }
            if not device_names:
                error("$.machine.metal_devices", "cannot be empty when a case is PASS_GPU")
            for record in successful_gpu:
                if record.get("device_name") not in device_names:
                    error(
                        f"{path}.execution.gpu_provenance",
                        "device_name must match an inventoried Metal device",
                    )
            if validation.get("comparison") == "exact":
                expected = validation.get("expected", {})
                actual = validation.get("actual", {})
                if expected.get("bytes") is None or expected.get("sha256") is None:
                    error(f"{path}.validation.expected", "exact PASS_GPU requires bytes and sha256")
                if expected != actual:
                    error(f"{path}.validation.actual", "must exactly equal expected bytes and sha256")

        elif classification == "PASS_CPU_EXPLICIT":
            if not execution.get("attempted") or execution.get("route") != "cpu_explicit":
                error(f"{path}.execution", "PASS_CPU_EXPLICIT requires attempted cpu_explicit execution")
            if execution.get("explicit_cpu_selection") is not True:
                error(f"{path}.execution.explicit_cpu_selection", "must be true")
            if execution.get("exit_code") != 0:
                error(f"{path}.execution.exit_code", "must be zero")
            if provenance:
                error(f"{path}.execution.gpu_provenance", "must be empty for an explicit CPU pass")

        elif classification == "FAIL":
            evidence_of_failure = any(
                record.get("status") == "FAIL" for record in stages.values() if isinstance(record, dict)
            ) or (execution.get("attempted") and execution.get("exit_code") not in (None, 0))
            evidence_of_failure = evidence_of_failure or validation.get("status") == "FAIL"
            evidence_of_failure = evidence_of_failure or any(
                item.get("passed") is False for item in assertions if isinstance(item, dict)
            )
            if not evidence_of_failure:
                error(path, "FAIL needs a failed stage, execution, validation, or assertion")

        elif classification == "UNSUPPORTED":
            if execution.get("attempted") or execution.get("route") != "none":
                error(f"{path}.execution", "UNSUPPORTED must not claim execution")
            if execution.get("exit_code") is not None or execution.get("backend_evidence_log") is not None:
                error(f"{path}.execution", "UNSUPPORTED must not carry execution result/log fields")
            if provenance:
                error(f"{path}.execution.gpu_provenance", "must be empty for UNSUPPORTED")
            if validation.get("attempted") or validation.get("status") != "NOT_RUN":
                error(f"{path}.validation", "UNSUPPORTED must not claim numerical validation")
            if not diagnostics or any(
                not isinstance(item, dict) or item.get("passed") is not True
                for item in diagnostics
            ):
                error(f"{path}.diagnostic_assertions", "must be nonempty and all pass")
            if any(
                not isinstance(item, dict)
                or not _nonempty_string(item.get("source_location"))
                for item in diagnostics
            ):
                error(f"{path}.diagnostic_assertions", "each unsupported diagnostic needs a source location")

        elif classification == "SKIP_ENVIRONMENT":
            if execution.get("attempted") or execution.get("route") != "none":
                error(f"{path}.execution", "SKIP_ENVIRONMENT must not claim execution")
            if execution.get("exit_code") is not None or execution.get("backend_evidence_log") is not None:
                error(f"{path}.execution", "SKIP_ENVIRONMENT must not carry execution result/log fields")
            if provenance:
                error(f"{path}.execution.gpu_provenance", "must be empty for SKIP_ENVIRONMENT")
            if validation.get("attempted") or validation.get("status") != "NOT_RUN":
                error(f"{path}.validation", "SKIP_ENVIRONMENT must not claim numerical validation")
            if any(
                not isinstance(stages.get(stage), dict)
                or stages[stage].get("status") != "NOT_RUN"
                for stage in valid_required_stages
            ):
                error(f"{path}.stages", "required stages must all be NOT_RUN for SKIP_ENVIRONMENT")
            if not environment_gaps:
                error(f"{path}.environment_gaps", "must identify at least one missing condition")

        elif classification == "NOT_RUN":
            if execution.get("attempted") or execution.get("route") != "none":
                error(f"{path}.execution", "NOT_RUN must not claim execution")
            if execution.get("exit_code") is not None or execution.get("backend_evidence_log") is not None:
                error(f"{path}.execution", "NOT_RUN must not carry execution result/log fields")
            if provenance:
                error(f"{path}.execution.gpu_provenance", "must be empty for NOT_RUN")
            if validation.get("attempted") or validation.get("status") != "NOT_RUN":
                error(f"{path}.validation", "NOT_RUN must not claim numerical validation")
            if any(
                not isinstance(stages.get(stage), dict)
                or stages[stage].get("status") != "NOT_RUN"
                for stage in valid_required_stages
            ):
                error(f"{path}.stages", "required stages must all be NOT_RUN for NOT_RUN")

    duplicates = sorted(item for item, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        error("$.cases", f"duplicate case ids: {duplicates}")

    summary = document.get("summary")
    if require_keys(
        "$.summary",
        summary,
        ("case_count", "counts", "required_failures", "all_required_attempted", "record_status"),
    ):
        expected_counts = {name: classifications.count(name) for name in CLASSIFICATIONS}
        if summary.get("case_count") != len(cases):
            error("$.summary.case_count", f"must equal {len(cases)}")
        if summary.get("counts") != expected_counts:
            error("$.summary.counts", f"must equal {expected_counts}")
        if summary.get("required_failures") != required_failures:
            error("$.summary.required_failures", f"must equal {required_failures}")
        all_required_attempted = required_not_run == 0
        if summary.get("all_required_attempted") is not all_required_attempted:
            error("$.summary.all_required_attempted", f"must be {all_required_attempted}")
        expected_record_status = (
            "FAIL" if required_failures else "INCOMPLETE" if required_not_run else "COMPLETE"
        )
        if summary.get("record_status") not in RECORD_STATUSES:
            error("$.summary.record_status", f"must be one of {RECORD_STATUSES}")
        elif summary.get("record_status") != expected_record_status:
            error("$.summary.record_status", f"must equal {expected_record_status}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    args = parser.parse_args(argv)
    invalid = 0
    for path in args.records:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"INVALID {path}: {exc}")
            invalid += 1
            continue
        errors = validate_document(document)
        if errors:
            print(f"INVALID {path} ({len(errors)} errors)")
            for item in errors:
                print(f"  {item}")
            invalid += 1
        else:
            print(f"VALID {path} ({len(document['cases'])} cases)")
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
