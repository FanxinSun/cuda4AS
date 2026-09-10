from __future__ import annotations

import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from tools.m2a.analyze_m2a_return import analyze
from tools.m2a.native_aot import AOTError, KernelParser, SSABuilder, verify_abi, verify_device_link


ROOT = Path(__file__).resolve().parents[2]
CU = ROOT / "oracle/src/vector_add.cu"
HDR = ROOT / "oracle/src/oracle.h"
SRC_HASH = "b3f205cabc42a697276244d5810ce62ff40a5c9cc6a8e1078420ccbefa88d0e2"
HDR_HASH = "6e7cc4c2e59e10db5b68df4116ed67ec3fe1d88d559c41e0a9feb2da3dd643bc"
OUT_HASH = "ed551637cf393112d0093037a0b41b9d1e9bd213c6037e8efcde30cf480f0332"


class NativeAOTCompilerTests(unittest.TestCase):
    def run_emit(self, root: Path, source: Path = CU) -> Path:
        import subprocess

        work = root / "work"
        proc = subprocess.run(
            ["python3", "tools/m2a/native_aot.py", "--source", str(source), "--header", str(HDR), "--work", str(work), "--skip-clang"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return work

    def test_ir_msl_abi_link_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a, b = self.run_emit(d / "a"), self.run_emit(d / "b")
            for name in ("ir.json", "abi.json", "device-link.json", "device-link-image.json", "kernel.metal", "device-split.cu"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), name)
            self.assertEqual(json.loads((a / "device-link.json").read_text())["empty"], False)

    def test_supported_operand_change_flows_into_ir_and_msl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            changed = d / "changed.cu"
            changed.write_text(CU.read_text().replace("a[i] + b[i]", "b[i] + a[i]"), encoding="utf-8")
            original_work, changed_work = self.run_emit(d / "original"), self.run_emit(d / "changed", changed)
            self.assertNotEqual((original_work / "kernel.metal").read_bytes(), (changed_work / "kernel.metal").read_bytes())
            self.assertEqual(json.loads((original_work / "ir.json").read_text())["source"]["sha256"], SRC_HASH)
            self.assertNotEqual(json.loads((original_work / "ir.json").read_text())["source"]["sha256"], json.loads((changed_work / "ir.json").read_text())["source"]["sha256"])

    def test_launch_size_flows_into_backend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            changed = d / "launch.cu"
            changed.write_text(CU.read_text().replace("vector_add<<<(N + 255) / 256, 256>>>", "vector_add<<<(N + 127) / 128, 128>>>"), encoding="utf-8")
            work = self.run_emit(d / "changed", changed)
            host = json.loads((work / "driver-facts.json").read_text())["host"]
            self.assertEqual(host["launch"]["block_x"], 128)
            self.assertIn("128u", (work / "kernel.metal").read_text())

    def test_unsupported_operation_and_address_space_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            bad_op = d / "bad-op.cu"
            bad_op.write_text(CU.read_text().replace("a[i] + b[i]", "a[i] && b[i]"), encoding="utf-8")
            bad_addr = d / "bad-address.cu"
            bad_addr.write_text(CU.read_text().replace("const float *a", "__constant__ float *a"), encoding="utf-8")
            bad_kernel, _ = KernelParser(bad_op.read_text()).find_kernel()
            self.assertRaises(AOTError, SSABuilder(bad_kernel).build)
            self.assertRaises(AOTError, KernelParser(bad_addr.read_text()).find_kernel)

    def test_malformed_abi_unresolved_link_and_fallback_fail(self) -> None:
        good = {"schema": "cuda4as-native-aot-abi-v1", "version": 1, "arguments": [{"index": i, "kind": "buffer" if i < 3 else "scalar", "address_space": "global" if i < 3 else "constant"} for i in range(4)], "thread_position": "1d_grid", "cpu_fallback": False}
        bad = dict(good); bad["arguments"] = good["arguments"][:3]
        self.assertRaises(AOTError, verify_abi, bad)
        bad_fallback = dict(good); bad_fallback["cpu_fallback"] = True
        self.assertRaises(AOTError, verify_abi, bad_fallback)
        link = {"schema": "cuda4as-device-link-v1", "version": 1, "status": "LINKED", "empty": False, "module_count": 1, "modules": [{"name": "k"}], "resolved_symbols": ["k"], "unresolved_symbols": ["missing"], "duplicate_symbols": []}
        self.assertRaises(AOTError, verify_device_link, link)


def make_return(path: Path, result: dict, *, unsafe: bool = False) -> None:
    with tarfile.open(path, "w:gz") as tar:
        members = {
            "m2a-result.json": json.dumps(result).encode() + b"\n",
            "expected.json": json.dumps({"cpu_fallback": False, "runtime_compilation": False}).encode() + b"\n",
            "inventory-binding.json": json.dumps({"schema": "cuda4as-m2a-inventory-binding-v1", "status": "BOUND_TO_VALIDATED_M1_INVENTORY", "machine": {"arch": "arm64", "model": "MacBookPro18,3", "metal_devices": [{"name": "Apple M1 Pro", "registry_id": 4294969587}]}}).encode() + b"\n",
            "PACKAGE-MANIFEST.sha256": f"{SRC_HASH}  oracle/src/vector_add.cu\n{HDR_HASH}  oracle/src/oracle.h\n".encode(),
            "stage-record.json": json.dumps({"schema": "cuda4as-m2a-stage-record-v1", "runner_exit": 0, "classification": "PASS_GPU", "cpu_fallback": False, "runtime_compilation": False, "stages": {k: "PASS" for k in ("package-verify", "preflight", "device-import", "ir_verify", "device_link", "msl_generation", "metal_compile", "metallib_link", "aot_manifest", "host_compile", "native_link", "runtime_launch")}}).encode() + b"\n",
            "aot-manifest.json": json.dumps({"schema": "cuda4as-m2a-aot-manifest-v1", "cpu_fallback": False, "runtime_compilation": False}).encode() + b"\n",
        }
        if result.get("classification") == "PASS_GPU":
            members["metallib/vector_add.metallib"] = b"AIR-LINKED-METALLIB"
            members["device-link-image.json"] = b"{\"status\":\"linked\"}\n"
        if unsafe:
            members["../outside"] = b"x"
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, __import__("io").BytesIO(data))


class ReturnAnalyzerTests(unittest.TestCase):
    def base_result(self) -> dict:
        return {"schema": "cuda4as-m2a-result-v1", "classification": "PASS_GPU", "device": {"name": "Apple M1 Pro", "registry_id": 4294969587, "route": "apple_gpu"}, "output": {"bytes": 4194304, "sha256": OUT_HASH, "expected_sha256": OUT_HASH, "mismatches": 0}, "stages": {k: "PASS" for k in ("allocation", "h2d", "aot_load", "launch", "last_error", "synchronize", "d2h", "validation", "cleanup")}, "cpu_fallback": False}

    def test_success_compile_failure_launch_failure_and_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            good = d / "good.tgz"; make_return(good, self.base_result()); self.assertEqual(analyze(good)["classification"], "PASS_GPU")
            for stage in ("metal_compile", "launch"):
                failed = self.base_result(); failed["classification"] = "FAIL"; failed["failed_stage"] = stage
                p = d / f"{stage}.tgz"; make_return(p, failed); self.assertEqual(analyze(p)["classification"], "FAIL")
            nr = d / "not-run.tgz"; make_return(nr, {"schema": "cuda4as-m2a-result-v1", "classification": "NOT_RUN", "cpu_fallback": False}); self.assertEqual(analyze(nr)["classification"], "NOT_RUN")

    def test_provenance_and_output_mismatch_demote_reported_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            wrong_device = self.base_result(); wrong_device["device"]["name"] = "Apple M2"; p = d / "device.tgz"; make_return(p, wrong_device); self.assertEqual(analyze(p)["classification"], "FAIL")
            wrong_output = self.base_result(); wrong_output["output"]["sha256"] = "0" * 64; p = d / "output.tgz"; make_return(p, wrong_output); self.assertEqual(analyze(p)["classification"], "FAIL")

    def test_archive_path_safety(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "unsafe.tgz"; make_return(p, {"schema": "cuda4as-m2a-result-v1", "classification": "NOT_RUN", "cpu_fallback": False}, unsafe=True)
            with self.assertRaises(ValueError): analyze(p)


if __name__ == "__main__":
    unittest.main()
