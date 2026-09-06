from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest

from tools.m1.validate_results import CLASSIFICATIONS, validate_document


HASH_A = "a" * 64
HASH_B = "b" * 64


def pass_stage() -> dict:
    return {"status": "PASS", "exit_code": 0, "log": "logs/stage.log"}


def base_gpu_case() -> dict:
    required_stages = [
        "configure",
        "host_compile",
        "device_compile",
        "native_link",
        "launch",
        "validation",
    ]
    return {
        "id": "synthetic.validator.gpu",
        "case_type": "validator_fixture",
        "required": True,
        "classification": "PASS_GPU",
        "source": {
            "files": [{"path": "fixture.cu", "sha256": HASH_A}],
            "application_source_unchanged": True,
            "application_build_unchanged": True,
            "diff_path": None,
        },
        "required_stages": required_stages,
        "stages": {stage: pass_stage() for stage in required_stages},
        "execution": {
            "attempted": True,
            "route": "apple_gpu",
            "explicit_cpu_selection": False,
            "exit_code": 0,
            "backend_evidence_log": "logs/gpu.log",
            "gpu_provenance": [
                {
                    "kernel": "vector_add",
                    "device": "apple_gpu",
                    "device_name": "Synthetic Apple GPU",
                    "source": "native_aot",
                    "semantic_quality": "exact",
                    "launch_success": True,
                    "completed": True,
                }
            ],
        },
        "validation": {
            "attempted": True,
            "status": "PASS",
            "comparison": "exact",
            "expected": {"bytes": 16, "sha256": HASH_B},
            "actual": {"bytes": 16, "sha256": HASH_B},
            "assertions": [{"id": "all_bytes_equal", "passed": True}],
        },
        "diagnostic_assertions": [],
        "environment_gaps": [],
        "reason": "Synthetic positive fixture for validator tests only.",
    }


def document_for(*cases: dict) -> dict:
    document = {
        "schema": "cuda4as-m1-result-v1",
        "run_id": "synthetic-validator-test",
        "created_utc": "2026-09-06T00:00:00Z",
        "contract": {
            "id": "cuda4as-m0-initial-ampere-source-profile",
            "revision": "0.1",
            "corpus_schema_version": 1,
        },
        "candidate": {
            "repository": "https://github.com/Lulzx/cuda-metal.git",
            "revision": "f486e5ebcfd381d06e3297afd65dbcbd5006a902",
            "source_state": "clean",
            "build_type": "Synthetic",
            "options": {"validator_fixture": True},
        },
        "machine": {
            "os_name": "macOS",
            "os_version": "synthetic",
            "os_build": "synthetic",
            "arch": "arm64",
            "model": "synthetic",
            "developer_directory": "/Synthetic/Xcode.app/Contents/Developer",
            "xcode_version": "synthetic",
            "sdk_version": "synthetic",
            "metal_compiler": "synthetic",
            "metal_devices": [
                {"name": "Synthetic Apple GPU", "registry_id": "synthetic-1"}
            ],
        },
        "cases": list(cases),
    }
    refresh_summary(document)
    return document


def refresh_summary(document: dict) -> None:
    counts = Counter(case["classification"] for case in document["cases"])
    required_failures = sum(
        case["required"] and case["classification"] == "FAIL"
        for case in document["cases"]
    )
    required_not_run = sum(
        case["required"] and case["classification"] == "NOT_RUN"
        for case in document["cases"]
    )
    document["summary"] = {
        "case_count": len(document["cases"]),
        "counts": {name: counts[name] for name in CLASSIFICATIONS},
        "required_failures": required_failures,
        "all_required_attempted": required_not_run == 0,
        "record_status": (
            "FAIL" if required_failures else "INCOMPLETE" if required_not_run else "COMPLETE"
        ),
    }


class ResultValidatorTests(unittest.TestCase):
    def assert_invalid_with(self, document: dict, fragment: str) -> None:
        errors = validate_document(document)
        self.assertTrue(errors, "document unexpectedly validated")
        self.assertTrue(
            any(fragment in item for item in errors),
            f"no error contained {fragment!r}: {errors}",
        )

    def test_valid_gpu_record(self) -> None:
        self.assertEqual(validate_document(document_for(base_gpu_case())), [])

    def test_valid_explicit_cpu_record(self) -> None:
        case = base_gpu_case()
        case["id"] = "synthetic.validator.cpu"
        case["classification"] = "PASS_CPU_EXPLICIT"
        case["execution"].update(
            {
                "route": "cpu_explicit",
                "explicit_cpu_selection": True,
                "backend_evidence_log": "logs/cpu.log",
                "gpu_provenance": [],
            }
        )
        self.assertEqual(validate_document(document_for(case)), [])

    def test_valid_unsupported_diagnostic(self) -> None:
        case = base_gpu_case()
        case["id"] = "synthetic.validator.unsupported"
        case["classification"] = "UNSUPPORTED"
        case["required_stages"] = ["device_compile"]
        case["stages"] = {
            "device_compile": {"status": "FAIL", "exit_code": 1, "log": "logs/compile.log"}
        }
        case["execution"] = {
            "attempted": False,
            "route": "none",
            "explicit_cpu_selection": False,
            "exit_code": None,
            "backend_evidence_log": None,
            "gpu_provenance": [],
        }
        case["validation"] = {
            "attempted": False,
            "status": "NOT_RUN",
            "comparison": "none",
            "expected": {"bytes": None, "sha256": None},
            "actual": {"bytes": None, "sha256": None},
            "assertions": [],
        }
        case["diagnostic_assertions"] = [
            {
                "id": "source_located_rejection",
                "passed": True,
                "source_location": "fixture.cu:7",
            }
        ]
        self.assertEqual(validate_document(document_for(case)), [])

    def test_valid_required_not_run_is_incomplete(self) -> None:
        case = base_gpu_case()
        case["id"] = "synthetic.validator.not-run"
        case["classification"] = "NOT_RUN"
        case["stages"] = {
            stage: {"status": "NOT_RUN", "exit_code": None, "log": None}
            for stage in case["required_stages"]
        }
        case["execution"] = {
            "attempted": False,
            "route": "none",
            "explicit_cpu_selection": False,
            "exit_code": None,
            "backend_evidence_log": None,
            "gpu_provenance": [],
        }
        case["validation"] = {
            "attempted": False,
            "status": "NOT_RUN",
            "comparison": "exact",
            "expected": {"bytes": 16, "sha256": HASH_B},
            "actual": {"bytes": None, "sha256": None},
            "assertions": [],
        }
        document = document_for(case)
        self.assertEqual(document["summary"]["record_status"], "INCOMPLETE")
        self.assertEqual(validate_document(document), [])

    def test_rejects_failed_stage_hidden_as_not_run(self) -> None:
        case = base_gpu_case()
        case["classification"] = "NOT_RUN"
        case["stages"] = {
            stage: {"status": "NOT_RUN", "exit_code": None, "log": None}
            for stage in case["required_stages"]
        }
        case["stages"]["device_compile"] = {
            "status": "FAIL",
            "exit_code": 1,
            "log": "logs/device-compile.log",
        }
        case["execution"] = {
            "attempted": False,
            "route": "none",
            "explicit_cpu_selection": False,
            "exit_code": None,
            "backend_evidence_log": None,
            "gpu_provenance": [],
        }
        case["validation"] = {
            "attempted": False,
            "status": "NOT_RUN",
            "comparison": "exact",
            "expected": {"bytes": 16, "sha256": HASH_B},
            "actual": {"bytes": None, "sha256": None},
            "assertions": [],
        }
        document = document_for(case)
        self.assert_invalid_with(document, "required stages must all be NOT_RUN for NOT_RUN")

    def test_rejects_m0_no_device_but_ok_shape(self) -> None:
        document = document_for(base_gpu_case())
        document["machine"]["metal_devices"] = []
        case = document["cases"][0]
        case["execution"].update(
            {
                "attempted": False,
                "route": "none",
                "exit_code": 0,
                "backend_evidence_log": None,
                "gpu_provenance": [],
            }
        )
        case["reason"] = "Synthetic historical pattern: no Metal device but process said ok."
        self.assert_invalid_with(document, "cannot be empty when a case is PASS_GPU")
        self.assert_invalid_with(document, "requires attempted apple_gpu execution")

    def test_rejects_failed_compilation_hidden_by_pass(self) -> None:
        document = document_for(base_gpu_case())
        document["cases"][0]["stages"]["device_compile"] = {
            "status": "FAIL",
            "exit_code": 1,
            "log": "logs/device-compile.log",
        }
        self.assert_invalid_with(document, "must be PASS for a pass classification")

    def test_rejects_pass_from_unverified_candidate_source(self) -> None:
        document = document_for(base_gpu_case())
        document["candidate"]["source_state"] = "not_present"
        self.assert_invalid_with(document, "must be clean when any case passes")

    def test_rejects_missing_gpu_provenance(self) -> None:
        document = document_for(base_gpu_case())
        document["cases"][0]["execution"]["gpu_provenance"] = []
        self.assert_invalid_with(document, "needs a completed exact/declared Apple-GPU launch")

    def test_rejects_mixed_gpu_and_cpu_fallback_provenance(self) -> None:
        document = document_for(base_gpu_case())
        document["cases"][0]["execution"]["gpu_provenance"].append(
            {
                "kernel": "vector_add",
                "device": "cpu",
                "device_name": "Synthetic CPU",
                "source": "cpu_fallback",
                "semantic_quality": "exact",
                "launch_success": True,
                "completed": True,
            }
        )
        self.assert_invalid_with(document, "every PASS_GPU provenance record")

    def test_rejects_reduced_precision_gpu_provenance(self) -> None:
        document = document_for(base_gpu_case())
        document["cases"][0]["execution"]["gpu_provenance"][0][
            "semantic_quality"
        ] = "reduced_precision_fp64"
        self.assert_invalid_with(document, "needs a completed exact/declared Apple-GPU launch")

    def test_rejects_bad_exact_output(self) -> None:
        document = document_for(base_gpu_case())
        document["cases"][0]["validation"]["actual"]["sha256"] = HASH_A
        self.assert_invalid_with(document, "must exactly equal expected bytes and sha256")

    def test_rejects_required_failure_hidden_by_aggregate(self) -> None:
        case = base_gpu_case()
        case["classification"] = "FAIL"
        case["validation"]["status"] = "FAIL"
        case["validation"]["assertions"][0]["passed"] = False
        document = document_for(case)
        document["summary"]["required_failures"] = 0
        document["summary"]["record_status"] = "COMPLETE"
        self.assert_invalid_with(document, "$.summary.required_failures")
        self.assert_invalid_with(document, "$.summary.record_status")

    def test_malformed_record_returns_errors_instead_of_crashing(self) -> None:
        document = document_for(base_gpu_case())
        case = document["cases"][0]
        case["source"] = None
        case["execution"] = None
        case["validation"] = None
        case["required_stages"] = [{"not": "a stage"}]
        errors = validate_document(document)
        self.assertTrue(errors)
        self.assertTrue(any("must be an object" in item for item in errors))

    def test_schema_document_is_json(self) -> None:
        schema = Path("docs/m1/result-schema-v1.schema.json")
        document = json.loads(schema.read_text())
        self.assertEqual(document["properties"]["schema"]["const"], "cuda4as-m1-result-v1")

    def test_repository_pending_record_is_valid_and_incomplete(self) -> None:
        document = json.loads(Path("docs/m1/m1-result-pending.json").read_text())
        self.assertEqual(document["summary"]["counts"]["NOT_RUN"], 3)
        self.assertEqual(document["summary"]["record_status"], "INCOMPLETE")
        self.assertEqual(validate_document(document), [])


if __name__ == "__main__":
    unittest.main()
