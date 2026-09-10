from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import tarfile
import tempfile
import unittest

from tools.m2a.analyze_m2a_return import analyze
from tools.m2a.clang_ast import ClangASTImportError, import_ast, import_ast_file
from tools.m2a.native_aot import (
    AOTError,
    KernelParser,
    SSABuilder,
    discover_toolchain,
    load_facade_contract,
    materialize_facade_arguments,
    run_clang_device,
    verify_abi,
    verify_device_link,
    verify_ir,
    verify_source_compiler_signature,
)


ROOT = Path(__file__).resolve().parents[2]
CU = ROOT / "oracle/src/vector_add.cu"
HDR = ROOT / "oracle/src/oracle.h"
SRC_HASH = "b3f205cabc42a697276244d5810ce62ff40a5c9cc6a8e1078420ccbefa88d0e2"
HDR_HASH = "6e7cc4c2e59e10db5b68df4116ed67ec3fe1d88d559c41e0a9feb2da3dd643bc"
OUT_HASH = "ed551637cf393112d0093037a0b41b9d1e9bd213c6037e8efcde30cf480f0332"
AST_FIXTURE = ROOT / "tests/m2a/fixtures/vector_add.clang-ast.json"
FACADE = ROOT / "tools/m2a/clang_cuda_facade/cuda_runtime.h"


def _walk_dicts(node: object):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_dicts(value)


class NativeAOTCompilerTests(unittest.TestCase):
    def run_emit(self, root: Path, source: Path = CU, ast_fixture: Path = AST_FIXTURE) -> Path:
        work = root / "work"
        proc = subprocess.run(
            ["python3", "tools/m2a/native_aot.py", "--source", str(source), "--header", str(HDR), "--work", str(work), "--ast-fixture", str(ast_fixture)],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return work

    def test_ir_msl_abi_link_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a, b = self.run_emit(d / "a"), self.run_emit(d / "b")
            for name in ("ir.json", "abi.json", "device-link.json", "device-link-image.json", "kernel.metal", "device-source.cu"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), name)
            self.assertEqual(json.loads((a / "device-link.json").read_text())["empty"], False)

    def test_supported_operand_change_flows_into_ir_and_msl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            changed = d / "changed.cu"
            changed.write_text(CU.read_text().replace("a[i] + b[i]", "b[i] + a[i]"), encoding="utf-8")
            changed_ast = d / "changed-ast.json"
            ast = json.loads(AST_FIXTURE.read_text())
            def mutate(node: object) -> bool:
                if isinstance(node, dict):
                    if node.get("kind") == "BinaryOperator" and node.get("opcode") == "+" and node.get("type", {}).get("qualType") == "float":
                        node["opcode"] = "-"
                        return True
                    return any(mutate(v) for v in node.values())
                if isinstance(node, list):
                    return any(mutate(v) for v in node)
                return False
            self.assertTrue(mutate(ast))
            changed_ast.write_text(json.dumps(ast), encoding="utf-8")
            original_work, changed_work = self.run_emit(d / "original"), self.run_emit(d / "changed", changed, changed_ast)
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

    def test_compiler_ast_is_semantic_authority_for_types_builtins_and_control_flow(self) -> None:
        kernel, metadata = import_ast_file(AST_FIXTURE, "vector_add")
        self.assertTrue(metadata["authoritative"])
        self.assertEqual(metadata["kind"], "clang_cuda_ast_json")
        self.assertEqual([p["address_space"] for p in kernel["params"]], ["global", "global", "global", "constant"])
        self.assertEqual([p["qual_type"] for p in kernel["params"]], ["const float *", "const float *", "float *", "int"])
        init = kernel["body"][0]["init"]
        self.assertEqual(init["type"], "u32")
        self.assertEqual(init["left"]["operator"], "*")
        self.assertEqual(init["left"]["left"]["base"]["builtin"], "blockIdx")
        self.assertEqual(kernel["body"][1]["kind"], "if")
        self.assertEqual(kernel["body"][1]["condition"]["operator"], "<")
        self.assertEqual(kernel["body"][1]["then"][0]["lhs"]["address_space"], "global")

    def test_compiler_ast_mutations_change_or_fail_import(self) -> None:
        root = json.loads(AST_FIXTURE.read_text())
        changed = json.loads(AST_FIXTURE.read_text())
        def find_add(node: object) -> dict[str, object] | None:
            if isinstance(node, dict):
                if node.get("kind") == "BinaryOperator" and node.get("opcode") == "+" and node.get("type", {}).get("qualType") == "float":
                    return node
                for value in node.values():
                    hit = find_add(value)
                    if hit is not None:
                        return hit
            elif isinstance(node, list):
                for value in node:
                    hit = find_add(value)
                    if hit is not None:
                        return hit
            return None
        add = find_add(changed)
        self.assertIsNotNone(add)
        add["opcode"] = "-"
        original_kernel, _ = import_ast(root, "vector_add")
        changed_kernel, _ = import_ast(changed, "vector_add")
        self.assertEqual(original_kernel["body"][1]["then"][0]["rhs"]["operator"], "+")
        self.assertEqual(changed_kernel["body"][1]["then"][0]["rhs"]["operator"], "-")
        malformed = json.loads(AST_FIXTURE.read_text())
        builtin = next(node for node in _walk_dicts(malformed) if node.get("kind") == "VarDecl" and node.get("name") == "blockIdx")
        builtin["attrs"] = []
        for node in _walk_dicts(malformed):
            if node.get("kind") == "DeclRefExpr" and node.get("name") == "blockIdx":
                node["referencedDecl"]["attrs"] = []
        with self.assertRaises(ClangASTImportError):
            import_ast(malformed, "vector_add")

        missing_body = json.loads(AST_FIXTURE.read_text())
        function = next(node for node in _walk_dicts(missing_body) if node.get("kind") == "FunctionDecl" and node.get("name") == "vector_add")
        function["inner"] = [node for node in function["inner"] if node.get("kind") != "CompoundStmt"]
        with self.assertRaises(ClangASTImportError):
            import_ast(missing_body, "vector_add")

        mismatched_kernel = json.loads(AST_FIXTURE.read_text())
        mismatched_function = next(node for node in _walk_dicts(mismatched_kernel) if node.get("kind") == "FunctionDecl" and node.get("name") == "vector_add")
        mismatched_function["name"] = "other_kernel"
        with self.assertRaises(ClangASTImportError):
            import_ast(mismatched_kernel, "vector_add")

        missing_location = json.loads(AST_FIXTURE.read_text())
        location_function = next(node for node in _walk_dicts(missing_location) if node.get("kind") == "FunctionDecl" and node.get("name") == "vector_add")
        location_function.pop("loc", None)
        with self.assertRaises(ClangASTImportError):
            import_ast(missing_location, "vector_add")

    def test_source_and_compiler_parameter_contract_mismatch_fails_closed(self) -> None:
        source_kernel, _ = KernelParser(CU.read_text()).find_kernel()
        compiler_kernel, _ = import_ast_file(AST_FIXTURE, "vector_add")
        compiler_kernel["params"][3]["type"] = "float"
        with self.assertRaises(AOTError):
            verify_source_compiler_signature(source_kernel, compiler_kernel)

    def test_facade_is_pinned_and_annotation_surface_is_explicit(self) -> None:
        text = FACADE.read_text()
        self.assertIn('CUDA4AS_CLANG_CUDA_FACADE_VERSION "1.0.0"', text)
        for attr in ("host", "device", "global", "constant", "shared", "device_builtin"):
            self.assertIn(f"__attribute__(({attr}))", text)
        self.assertIn("extern \"C\" __device__ __device_builtin__ const dim3 blockIdx", text)
        self.assertIn("cudaMalloc(T **device_pointer, size_t bytes)", text)

    def test_facade_manifest_binds_hash_target_arguments_and_locations(self) -> None:
        contract = load_facade_contract(FACADE.parent)
        self.assertEqual(contract["version"], "1.0.0")
        self.assertEqual(contract["supported_clang"]["validated_major"], 23)
        self.assertEqual(contract["target"]["cuda_gpu_arch"], "sm_86")
        self.assertEqual(contract["header_sha256"], "e9fabf30365d3b5e3ecaab8e3119d80723eb5c8a20ee32a95ebbb95497d8208b")
        args = materialize_facade_arguments(CU, FACADE.parent)
        self.assertEqual(args[args.index("-I") + 1], str(FACADE.parent))
        self.assertEqual(args[-1], str(CU))
        kernel, _ = import_ast_file(AST_FIXTURE, "vector_add")
        for expression in _walk_dicts(kernel):
            if expression.get("kind") in {"name", "member", "index", "binary", "literal", "cast", "unary"}:
                location = expression.get("source_location", {})
                self.assertGreater(location.get("line", 0), 0)
                self.assertGreater(location.get("column", 0), 0)

    def test_toolchain_identity_is_pinned_to_supported_clang_range(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            good = d / "clang-good"
            good.write_text("#!/bin/sh\nprintf '%s\\n' 'Homebrew clang version 23.1.0'\n")
            good.chmod(good.stat().st_mode | stat.S_IXUSR)
            contract = load_facade_contract(FACADE.parent)
            found = discover_toolchain(str(good), facade_contract=contract)
            self.assertEqual(found["major"], 23)
            self.assertEqual(found["supported_range"], ">=18,<24")
            old = d / "clang-old"
            old.write_text("#!/bin/sh\nprintf '%s\\n' 'Ubuntu clang version 17.0.6'\n")
            old.chmod(old.stat().st_mode | stat.S_IXUSR)
            with self.assertRaises(AOTError):
                discover_toolchain(str(old), facade_contract=contract)

    def test_live_frontend_command_uses_original_source_and_facade(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            fake = d / "fake-clang"
            args_file = d / "args.txt"
            source_file = d / "source.captured"
            fake.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$@\" > \"" + str(args_file) + "\"\n"
                "for arg do last=\"$arg\"; done\n"
                "cat \"$last\" > \"" + str(source_file) + "\"\n"
                "cat \"" + str(AST_FIXTURE) + "\"\n"
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            ast = d / "ast.json"
            log = d / "clang.log"
            result = run_clang_device(str(fake), CU, ast, log, FACADE.parent)
            self.assertEqual(result["exit_code"], 0)
            args = args_file.read_text().splitlines()
            self.assertIn(str(CU), args)
            self.assertIn(str(FACADE.parent), args)
            self.assertNotIn("device-split.cu", " ".join(args))
            self.assertEqual(source_file.read_bytes(), CU.read_bytes())

    def test_skip_clang_is_explicitly_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "work"
            proc = subprocess.run(
                ["python3", "tools/m2a/native_aot.py", "--source", str(CU), "--header", str(HDR), "--work", str(work), "--skip-clang"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(proc.returncode, 77, proc.stderr)
            ir = json.loads((work / "ir.json").read_text())
            self.assertEqual(ir["validation_status"], "NOT_RUN")
            self.assertFalse(ir["semantic_source"]["authoritative"])
            self.assertEqual(ir["frontend"]["status"], "NOT_RUN")
            with self.assertRaises(AOTError):
                verify_ir(ir)


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
