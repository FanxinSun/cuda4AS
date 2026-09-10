from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest

from tools.m1.analyze_native_return import analyze, load_verified_files
from tools.m1.validate_results import validate_document


REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURES = json.loads((REPOSITORY / "docs/m1/fixtures.json").read_text())
ARTIFACT = json.loads((REPOSITORY / "docs/m1/native-feasibility-artifact.json").read_text())
ARTIFACT_PACKAGE_ENTRIES = {
    record["path"]: record["sha256"]
    for record in ARTIFACT["files"]
    if record["path"] != "PACKAGE-MANIFEST.sha256"
}
CANDIDATE = "f486e5ebcfd381d06e3297afd65dbcbd5006a902"
VF64 = "729021777455da72db8809d9ef1269c677d88b3f"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def u32(value: int) -> int:
    return value & 0xFFFFFFFF


def expected_outputs() -> dict[str, bytes]:
    vector_values = []
    for index in range(4096):
        a = u32(index * 2654435761 + 17)
        b = u32((index ^ 0xA5A5A5A5) * 2246822519)
        vector_values.append(u32(a + b))
    link_values = []
    for index in range(2048):
        value = u32(index * 747796405 + 2891336453)
        value = u32(value ^ u32(index * 0x9E3779B9))
        value = u32((value << 7) | (value >> 25))
        link_values.append(u32(value * 2246822519 + 3266489917))
    return {
        "outputs/oracle-vector-add.bin": (REPOSITORY / "oracle/ref/vector_add.bin").read_bytes(),
        "outputs/cmake-vector-add.bin": b"".join(
            struct.pack("<I", value) for value in vector_values
        ),
        "outputs/cmake-device-link.bin": b"".join(
            struct.pack("<I", value) for value in link_values
        ),
    }


def normalized_inventory() -> dict:
    return {
        "schema": "cuda4as-m1-mac-inventory-normalized-v1",
        "source_archive": deepcopy(ARTIFACT["inventory_binding"]["inventory_archive"]),
        "validation": {
            "archive_members_safe": True,
            "internal_manifest_valid": True,
        },
        "inventory_status": "COMPLETE",
        "machine": {
            "os_name": "macOS",
            "os_version": "15.6",
            "os_build": "24G84",
            "arch": "arm64",
            "model": "Mac16,1",
            "developer_directory": "/Applications/Xcode.app/Contents/Developer",
            "xcode_version": "Xcode 16.4\nBuild version 16F6",
            "sdk_version": "15.5",
            "metal_compiler": "/Applications/Xcode.app/metal",
            "metal_devices": [
                {"name": "Apple M4", "registry_id": "1234", "ignored_extra": True}
            ],
        },
    }


def tsv(header: tuple[str, ...], rows: list[tuple[object, ...]]) -> bytes:
    lines = ["\t".join(header)]
    lines.extend("\t".join(str(item) for item in row) for row in rows)
    return ("\n".join(lines) + "\n").encode()


def add_manifest(files: dict[str, bytes]) -> None:
    files["MANIFEST.sha256"] = "".join(
        f"{digest(data)}  {name}\n" for name, data in sorted(files.items())
    ).encode()


def write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(data)
            info.mode = 0o600
            tf.addfile(info, io.BytesIO(data))


def provenance(kernel: str, source: str, device_name: str = "Apple M4") -> bytes:
    return (
        "CUMETAL_PROVENANCE event=kernel_launch "
        f'kernel="{kernel}" source={source} provenance=test semantic_quality=exact '
        f'device=apple_gpu device_name="{device_name}" math_mode=ieee64 '
        "compile_cache_hit=false launch_success=true duration_ns=42 "
        'grid=(1,1,1) block=(1,1,1) unsupported_reason=""\n'
    ).encode()


def build_return_files(inventory: dict, mode: str = "pass") -> dict[str, bytes]:
    binding = {
        "schema": "cuda4as-m1-inventory-binding-v1",
        "status": "BOUND_TO_RETURNED_INVENTORY",
        "inventory_archive": inventory["source_archive"],
    }
    binding_bytes = (json.dumps(binding, indent=2) + "\n").encode()
    package_entries = dict(ARTIFACT_PACKAGE_ENTRIES)
    if digest(binding_bytes) != package_entries["target-inventory-binding.json"]:
        raise AssertionError("synthetic inventory binding does not match bound artifact")
    package_manifest = "".join(
        f"{value}  {key}\n" for key, value in sorted(package_entries.items())
    ).encode()
    facts = [
        ("schema", "cuda4as-m1-native-return-v1"),
        ("started_utc", "2026-09-06T12:00:00Z"),
        ("package_root", "/tmp/native-v1"),
        ("work_directory", "/tmp/native-v1/work/run"),
        ("network_operations", "none"),
        ("install_update_sudo_operations", "none"),
        ("candidate_revision", CANDIDATE),
        ("vf64_revision", VF64),
        ("build_type", "Release"),
        ("CUMETAL_BUILD_TESTS", "OFF"),
        ("CUMETAL_ENABLE_CUDA_REGISTRATION", "ON"),
        ("CUMETAL_ENABLE_BINARY_SHIM", "OFF"),
        ("CUMETAL_CUDA_ARCH", "sm_86"),
        ("CUMETAL_FP64_MODE", "ieee64"),
        ("uname_system", "Darwin"),
        ("uname_machine", "arm64"),
        ("macos_product_version", "15.6"),
        ("macos_build", "24G84"),
        ("developer_directory", "/Applications/Xcode.app/Contents/Developer"),
        ("machine_model", "Mac16,1"),
        ("xcode_version", "Xcode 16.4;Build version 16F6;"),
        ("metal_compiler", "/Applications/Xcode.app/metal"),
        ("sdk_version", "15.5"),
        ("ended_utc", "2026-09-06T12:01:00Z"),
        ("runner_exit_code", "0" if mode == "pass" else "77"),
    ]
    files: dict[str, bytes] = {
        "facts.tsv": tsv(("key", "value"), facts),
        "package/target-inventory-binding.json": binding_bytes,
        "package/PACKAGE-MANIFEST.sha256": package_manifest,
        "logs/package.txt": b"package ok\n",
        "logs/environment.txt": b"environment checked\n",
        "commands.txt": b"synthetic analyzer fixture\n",
    }
    cases = {case["id"]: case for case in FIXTURES["cases"]}
    sequence = 0
    event_rows: list[tuple[object, ...]] = []
    for case_id, case in cases.items():
        for stage in case["future_required_stages"]:
            sequence += 1
            event_rows.append((sequence, case_id, stage, "NOT_RUN", "-", "-", "initial"))
    sequence += 1
    event_rows.append((sequence, "_package", "integrity", "PASS", 0, "logs/package.txt", "ok"))
    assertion_rows: list[tuple[object, ...]] = [
        ("_package", "fixture_inputs_match_before_run", "true", "package"),
        ("_package", "fixture_inputs_unchanged_after_run", "true", "package"),
    ]
    if mode == "skip":
        sequence += 1
        event_rows.append(
            (sequence, "_environment", "preflight", "FAIL", 77, "environment-gaps.txt", "missing")
        )
        files["environment-gaps.txt"] = b"synthetic missing prerequisite\n"
    else:
        sequence += 1
        event_rows.append(
            (sequence, "_environment", "preflight", "PASS", 0, "logs/environment.txt", "ok")
        )
        sequence += 1
        event_rows.append(
            (sequence, "_candidate", "source_integrity", "PASS", 0, "logs/package.txt", "ok")
        )
        files["environment-gaps.txt"] = b""
        files.update(expected_outputs())
        for case_id, case in cases.items():
            slug = case_id.replace(".", "_")
            kernel = "transform_kernel" if "multi_tu" in case_id else "vector_add"
            source = "generic_nvvm" if case_id == "oracle.vector_add" else "generic_ptx"
            run_log = f"logs/{slug}_run.txt"
            files[run_log] = provenance(kernel, source)
            for stage in case["future_required_stages"]:
                sequence += 1
                log = run_log if stage == "launch" else "logs/package.txt"
                event_rows.append((sequence, case_id, stage, "PASS", 0, log, "passed"))
            assertion_rows.extend(
                [
                    (case_id, "output_bytes", "true", "synthetic"),
                    (case_id, "output_sha256", "true", "synthetic"),
                    (case_id, "full_output_comparison", "true", "synthetic"),
                    (case_id, "apple_gpu_launch", "true", run_log),
                    (case_id, "semantic_quality_exact", "true", run_log),
                ]
            )
    files["case-stage-events.tsv"] = tsv(
        ("sequence", "case_id", "stage", "status", "exit_code", "log", "note"),
        event_rows,
    )
    files["assertions.tsv"] = tsv(
        ("case_id", "assertion_id", "passed", "evidence"), assertion_rows
    )
    add_manifest(files)
    return files


class NativeReturnAnalyzerTests(unittest.TestCase):
    def make_archive(self, files: dict[str, bytes]) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "native-return.tgz"
        write_tar(path, files)
        return temp, path

    def test_all_three_cases_require_and_receive_gpu_evidence(self) -> None:
        inventory = normalized_inventory()
        temp, path = self.make_archive(build_return_files(inventory))
        self.addCleanup(temp.cleanup)
        result, validation = analyze(path, inventory, FIXTURES)
        self.assertEqual(result["summary"]["counts"]["PASS_GPU"], 3)
        self.assertEqual(result["summary"]["record_status"], "COMPLETE")
        self.assertEqual(validate_document(result), [])
        self.assertEqual(validation["archive_sha256"], digest(path.read_bytes()))

    def test_recomputed_bad_output_cannot_inherit_returned_pass_assertions(self) -> None:
        inventory = normalized_inventory()
        files = build_return_files(inventory)
        files["outputs/cmake-vector-add.bin"] += b"bad"
        files.pop("MANIFEST.sha256")
        add_manifest(files)
        temp, path = self.make_archive(files)
        self.addCleanup(temp.cleanup)
        result, _ = analyze(path, inventory, FIXTURES)
        cases = {case["id"]: case for case in result["cases"]}
        case = cases["integration.minimal_cmake_cuda"]
        self.assertEqual(case["classification"], "FAIL")
        self.assertEqual(case["validation"]["status"], "FAIL")
        self.assertIn(
            {"id": "analyzer.returned_output_exact", "passed": False},
            case["validation"]["assertions"],
        )
        self.assertEqual(validate_document(result), [])

    def test_uninventoried_device_is_failure(self) -> None:
        inventory = normalized_inventory()
        files = build_return_files(inventory)
        for name in list(files):
            if name.endswith("_run.txt"):
                files[name] = files[name].replace(b"Apple M4", b"Unlisted GPU")
        files.pop("MANIFEST.sha256")
        add_manifest(files)
        temp, path = self.make_archive(files)
        self.addCleanup(temp.cleanup)
        result, _ = analyze(path, inventory, FIXTURES)
        self.assertEqual(result["summary"]["counts"]["PASS_GPU"], 0)
        self.assertEqual(result["summary"]["counts"]["FAIL"], 3)
        self.assertEqual(validate_document(result), [])

    def test_malformed_candidate_provenance_becomes_visible_failure(self) -> None:
        inventory = normalized_inventory()
        files = build_return_files(inventory)
        files[
            "logs/integration_minimal_cmake_cuda_run.txt"
        ] = b"CUMETAL_PROVENANCE event=kernel_launch malformed\n"
        files.pop("MANIFEST.sha256")
        add_manifest(files)
        temp, path = self.make_archive(files)
        self.addCleanup(temp.cleanup)
        result, _ = analyze(path, inventory, FIXTURES)
        cases = {case["id"]: case for case in result["cases"]}
        case = cases["integration.minimal_cmake_cuda"]
        self.assertEqual(case["classification"], "FAIL")
        self.assertIn(
            {"id": "analyzer.gpu_provenance_parse", "passed": False},
            case["validation"]["assertions"],
        )
        self.assertIn("provenance_parse_errors", case["reason"])
        self.assertEqual(validate_document(result), [])

    def test_preflight_gap_is_skip_environment_without_execution(self) -> None:
        inventory = normalized_inventory()
        temp, path = self.make_archive(build_return_files(inventory, mode="skip"))
        self.addCleanup(temp.cleanup)
        result, _ = analyze(path, inventory, FIXTURES)
        self.assertEqual(result["summary"]["counts"]["SKIP_ENVIRONMENT"], 3)
        self.assertTrue(result["summary"]["all_required_attempted"])
        self.assertEqual(validate_document(result), [])

    def test_inventory_binding_mismatch_is_rejected(self) -> None:
        inventory = normalized_inventory()
        files = build_return_files(inventory)
        changed = deepcopy(inventory)
        changed["source_archive"]["sha256"] = "e" * 64
        temp, path = self.make_archive(files)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "binding does not match"):
            analyze(path, changed, FIXTURES)

    def test_package_manifest_mismatch_is_rejected(self) -> None:
        inventory = normalized_inventory()
        files = build_return_files(inventory)
        expected = ARTIFACT_PACKAGE_ENTRIES["README.md"]
        files["package/PACKAGE-MANIFEST.sha256"] = files[
            "package/PACKAGE-MANIFEST.sha256"
        ].replace(
            f"{expected}  README.md\n".encode(),
            f"{'0' * 64}  README.md\n".encode(),
        )
        files.pop("MANIFEST.sha256")
        add_manifest(files)
        temp, path = self.make_archive(files)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "package manifest does not match"):
            analyze(path, inventory, FIXTURES)

    def test_archive_link_member_is_rejected_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "link.tgz"
            with tarfile.open(path, "w:gz") as tf:
                info = tarfile.TarInfo("./facts.tsv")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                tf.addfile(info)
            with self.assertRaisesRegex(ValueError, "unsupported archive member type"):
                load_verified_files(path)

    def test_archive_traversal_member_is_rejected_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traversal.tgz"
            with tarfile.open(path, "w:gz") as tf:
                data = b"bad"
                info = tarfile.TarInfo("../outside")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                load_verified_files(path)


if __name__ == "__main__":
    unittest.main()
