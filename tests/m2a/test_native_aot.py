from __future__ import annotations

import json
import hashlib
from pathlib import Path
import stat
import subprocess
import tarfile
import tempfile
import unittest

from tools.m2a.analyze_m2a_return import analyze
from tools.m2a.clang_ast import ClangASTImportError, _location, import_ast, import_ast_file
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
AST_FIXTURE = ROOT / "tests/m2a/fixtures/vector_add.clang-json-conformance.json"
AST_FIXTURE_META = ROOT / "tests/m2a/fixtures/vector_add.clang-json-conformance.provenance.json"
FACADE = ROOT / "tools/m2a/clang_cuda_facade/cuda_runtime.h"
FACADE_CONTRACT = load_facade_contract(FACADE.parent)


def _walk_dicts(node: object):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_dicts(value)


def import_fixture(root: dict, *, source_bytes: bytes | None = None, facade_contract: dict | None = FACADE_CONTRACT):
    return import_ast(
        root,
        "vector_add",
        source_bytes=source_bytes,
        source_path=CU,
        facade_contract=facade_contract,
    )


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
            abi = json.loads((a / "abi.json").read_text())
            self.assertEqual(abi["address_space_mapping"]["schema"], "cuda4as-address-space-lowering-v1")
            self.assertEqual(abi["arguments"][0]["semantic_address_space"], "global")
            self.assertEqual(abi["arguments"][0]["metal_address_space"], "device")

    def test_schema_fixture_is_not_live_compiler_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work = self.run_emit(Path(td) / "fixture")
            ir = json.loads((work / "ir.json").read_text())
            driver = json.loads((work / "driver-facts.json").read_text())
            self.assertEqual(ir["validation_status"], "FIXTURE")
            self.assertEqual(ir["compiler_validation_status"], "NOT_RUN")
            self.assertEqual(driver["compiler_validation_status"], "NOT_RUN")
            self.assertEqual(ir["semantic_source"]["status"], "FIXTURE")
            self.assertEqual(ir["semantic_source"]["validation_status"], "NOT_RUN")

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
        good = {
            "schema": "cuda4as-native-aot-abi-v1", "version": 1,
            "arguments": [{"index": i, "kind": "buffer" if i < 3 else "scalar", "address_space": "global" if i < 3 else "constant", "metal_address_space": "device" if i < 3 else "constant", "semantic_address_space": "global" if i < 3 else "unknown"} for i in range(4)],
            "address_space_mapping": {"schema": "cuda4as-address-space-lowering-v1", "status": "EXPLICIT_PORTABLE_RULE", "rules": [{"compiler_kind": "pointer", "compiler_address_space": "global", "metal_address_space": "device"}, {"compiler_kind": "scalar", "compiler_address_space": "any", "metal_address_space": "constant"}]},
            "thread_position": "1d_grid", "cpu_fallback": False,
        }
        bad = dict(good); bad["arguments"] = good["arguments"][:3]
        self.assertRaises(AOTError, verify_abi, bad)
        bad_fallback = dict(good); bad_fallback["cpu_fallback"] = True
        self.assertRaises(AOTError, verify_abi, bad_fallback)
        link = {"schema": "cuda4as-device-link-v1", "version": 1, "status": "LINKED", "empty": False, "module_count": 1, "modules": [{"name": "k"}], "resolved_symbols": ["k"], "unresolved_symbols": ["missing"], "duplicate_symbols": []}
        self.assertRaises(AOTError, verify_device_link, link)

    def test_compiler_ast_is_semantic_authority_for_types_builtins_and_control_flow(self) -> None:
        kernel, metadata = import_ast_file(AST_FIXTURE, "vector_add", source=CU, facade_contract=FACADE_CONTRACT)
        self.assertTrue(metadata["authoritative"])
        self.assertEqual(metadata["kind"], "clang_cuda_ast_json")
        self.assertEqual([p["address_space"] for p in kernel["params"]], ["global", "global", "global", "unknown"])
        self.assertEqual([p["qual_type"] for p in kernel["params"]], ["const float *", "const float *", "float *", "int"])
        init = kernel["body"][0]["init"]
        self.assertEqual(init["type"], "u32")
        self.assertEqual(init["left"]["operator"], "*")
        self.assertEqual(init["left"]["left"]["base"]["builtin"], "blockIdx")
        self.assertEqual(init["source_location"]["provenance"], "exact")
        self.assertEqual(init["left"]["source_location"]["provenance"], "inherited")
        self.assertEqual(kernel["body"][1]["kind"], "if")
        self.assertEqual(kernel["body"][1]["condition"]["operator"], "<")
        self.assertEqual(kernel["body"][1]["then"][0]["lhs"]["address_space"], "global")
        self.assertEqual(metadata["kernel_attributes"], ["CUDAGlobalAttr"])

    def test_fixture_is_explicit_schema_conformance_data(self) -> None:
        provenance = json.loads(AST_FIXTURE_META.read_text())
        self.assertEqual(provenance["status"], "SYNTHETIC_SCHEMA_CONFORMANCE")
        self.assertFalse(provenance["capture"]["native_clang_capture"])
        self.assertEqual(provenance["fixture_sha256"], hashlib.sha256(AST_FIXTURE.read_bytes()).hexdigest())
        self.assertEqual(provenance["compiler_identity"]["validation_status"], "NOT_RUN")
        self.assertEqual(provenance["capture"]["status"], "NOT_CAPTURED")
        self.assertEqual(provenance["producer"], "cuda4AS hand-authored Clang JSON schema conformance fixture")
        builtin_kinds = {
            child.get("kind")
            for node in _walk_dicts(json.loads(AST_FIXTURE.read_text()))
            if node.get("kind") == "VarDecl" and node.get("name") in {"blockIdx", "blockDim", "threadIdx"}
            for child in node.get("inner", [])
        }
        self.assertEqual(builtin_kinds, {"CUDADeviceAttr"})

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
        original_kernel, _ = import_fixture(root)
        changed_kernel, _ = import_fixture(changed)
        self.assertEqual(original_kernel["body"][1]["then"][0]["rhs"]["operator"], "+")
        self.assertEqual(changed_kernel["body"][1]["then"][0]["rhs"]["operator"], "-")
        malformed = json.loads(AST_FIXTURE.read_text())
        for node in _walk_dicts(malformed):
            if node.get("kind") == "DeclRefExpr" and node.get("name") == "blockIdx":
                node["referencedDecl"]["id"] = "missing-blockIdx"
        with self.assertRaises(ClangASTImportError):
            import_fixture(malformed)

        missing_device = json.loads(AST_FIXTURE.read_text())
        builtin = next(node for node in _walk_dicts(missing_device) if node.get("kind") == "VarDecl" and node.get("name") == "blockIdx")
        builtin["inner"] = [node for node in builtin["inner"] if node.get("kind") != "CUDADeviceAttr"]
        with self.assertRaises(ClangASTImportError):
            import_fixture(missing_device)

        conflicting_builtin = json.loads(AST_FIXTURE.read_text())
        builtin = next(node for node in _walk_dicts(conflicting_builtin) if node.get("kind") == "VarDecl" and node.get("name") == "blockIdx")
        builtin["inner"].append({"kind": "CUDAHostAttr"})
        with self.assertRaises(ClangASTImportError):
            import_fixture(conflicting_builtin)

        old_layout = json.loads(AST_FIXTURE.read_text())
        old_function = next(node for node in _walk_dicts(old_layout) if node.get("kind") == "FunctionDecl" and node.get("name") == "vector_add")
        old_function["isCUDAKernel"] = True
        old_function["attrs"] = [{"kind": "CUDAKernelAttr"}]
        with self.assertRaises(ClangASTImportError):
            import_fixture(old_layout)

        missing_body = json.loads(AST_FIXTURE.read_text())
        function = next(node for node in _walk_dicts(missing_body) if node.get("kind") == "FunctionDecl" and node.get("name") == "vector_add")
        function["inner"] = [node for node in function["inner"] if node.get("kind") != "CompoundStmt"]
        with self.assertRaises(ClangASTImportError):
            import_fixture(missing_body)

        mismatched_kernel = json.loads(AST_FIXTURE.read_text())
        mismatched_function = next(node for node in _walk_dicts(mismatched_kernel) if node.get("kind") == "FunctionDecl" and node.get("name") == "vector_add")
        mismatched_function["name"] = "other_kernel"
        with self.assertRaises(ClangASTImportError):
            import_fixture(mismatched_kernel)

        missing_location = json.loads(AST_FIXTURE.read_text())
        location_function = next(node for node in _walk_dicts(missing_location) if node.get("kind") == "FunctionDecl" and node.get("name") == "vector_add")
        location_function.pop("loc", None)
        with self.assertRaises(ClangASTImportError):
            import_fixture(missing_location)

    def test_source_and_compiler_parameter_contract_mismatch_fails_closed(self) -> None:
        source_kernel, _ = KernelParser(CU.read_text()).find_kernel()
        compiler_kernel, _ = import_ast_file(AST_FIXTURE, "vector_add", source=CU, facade_contract=FACADE_CONTRACT)
        compiler_kernel["params"][3]["type"] = "float"
        with self.assertRaises(AOTError):
            verify_source_compiler_signature(source_kernel, compiler_kernel)

    def test_address_space_is_preserved_and_backend_rejects_unknown(self) -> None:
        generic_ast = json.loads(AST_FIXTURE.read_text())
        generic_param = next(node for node in _walk_dicts(generic_ast) if node.get("kind") == "ParmVarDecl" and node.get("name") == "a")
        generic_param["type"]["addressSpace"] = "generic"
        generic_kernel, _ = import_fixture(generic_ast)
        self.assertEqual(generic_kernel["params"][0]["address_space"], "generic")
        with self.assertRaises(AOTError):
            SSABuilder(generic_kernel).build()

        unknown_ast = json.loads(AST_FIXTURE.read_text())
        unknown_param = next(node for node in _walk_dicts(unknown_ast) if node.get("kind") == "ParmVarDecl" and node.get("name") == "a")
        unknown_param["type"].pop("addressSpace")
        unknown_kernel, _ = import_fixture(unknown_ast)
        self.assertEqual(unknown_kernel["params"][0]["address_space"], "unknown")
        with self.assertRaises(AOTError):
            SSABuilder(unknown_kernel).build()

        constant_ast = json.loads(AST_FIXTURE.read_text())
        constant_param = next(node for node in _walk_dicts(constant_ast) if node.get("kind") == "ParmVarDecl" and node.get("name") == "a")
        constant_param["type"]["addressSpace"] = "constant"
        constant_kernel, _ = import_fixture(constant_ast)
        self.assertEqual(constant_kernel["params"][0]["address_space"], "constant")
        with self.assertRaises(AOTError):
            SSABuilder(constant_kernel).build()

    def test_builtin_binding_uses_compiler_identity_not_same_name(self) -> None:
        extra = json.loads(AST_FIXTURE.read_text())
        extra["inner"].insert(1, {"kind": "VarDecl", "id": "user-blockIdx", "name": "blockIdx", "type": {"qualType": "const dim3"}, "loc": {"file": "oracle/src/vector_add.cu", "line": 50, "col": 1}, "inner": [{"kind": "CUDADeviceAttr"}]})
        kernel, _ = import_fixture(extra)
        self.assertEqual(kernel["body"][0]["init"]["left"]["left"]["base"]["builtin"], "blockIdx")
        wrong_ref = json.loads(AST_FIXTURE.read_text())
        wrong_ref["inner"].insert(1, {"kind": "VarDecl", "id": "user-blockIdx", "name": "blockIdx", "type": {"qualType": "const dim3"}, "loc": {"file": "oracle/src/vector_add.cu", "line": 50, "col": 1}, "inner": [{"kind": "CUDADeviceAttr"}]})
        first_ref = next(node for node in _walk_dicts(wrong_ref) if node.get("kind") == "DeclRefExpr" and node.get("name") == "blockIdx")
        first_ref["referencedDecl"]["id"] = "user-blockIdx"
        with self.assertRaises(ClangASTImportError):
            import_fixture(wrong_ref)

    def test_builtin_facade_provenance_and_contract_fail_closed(self) -> None:
        no_contract = json.loads(AST_FIXTURE.read_text())
        with self.assertRaises(ClangASTImportError):
            import_fixture(no_contract, facade_contract=None)

        wrong_file = json.loads(AST_FIXTURE.read_text())
        builtin = next(node for node in _walk_dicts(wrong_file) if node.get("kind") == "VarDecl" and node.get("name") == "blockIdx")
        builtin["loc"]["file"] = "oracle/src/vector_add.cu"
        with self.assertRaises(ClangASTImportError):
            import_fixture(wrong_file)

        wrong_line = json.loads(AST_FIXTURE.read_text())
        builtin = next(node for node in _walk_dicts(wrong_line) if node.get("kind") == "VarDecl" and node.get("name") == "blockIdx")
        builtin["loc"]["line"] += 1
        with self.assertRaises(ClangASTImportError):
            import_fixture(wrong_line)

        wrong_type = json.loads(AST_FIXTURE.read_text())
        builtin = next(node for node in _walk_dicts(wrong_type) if node.get("kind") == "VarDecl" and node.get("name") == "blockIdx")
        builtin["type"]["qualType"] = "dim3"
        with self.assertRaises(ClangASTImportError):
            import_fixture(wrong_type)

        wrong_name = json.loads(AST_FIXTURE.read_text())
        builtin = next(node for node in _walk_dicts(wrong_name) if node.get("kind") == "VarDecl" and node.get("name") == "blockIdx")
        builtin["name"] = "not_blockIdx"
        with self.assertRaises(ClangASTImportError):
            import_fixture(wrong_name)

        broken_contract = dict(FACADE_CONTRACT)
        broken_contract["builtin_contract"] = dict(FACADE_CONTRACT["builtin_contract"])
        broken_contract["builtin_contract"]["header_sha256"] = "0" * 64
        with self.assertRaises(ClangASTImportError):
            import_fixture(json.loads(AST_FIXTURE.read_text()), facade_contract=broken_contract)

    def test_offset_location_is_exact_and_parent_fallback_is_inherited(self) -> None:
        source = CU.read_bytes()
        exact = _location({"loc": {"offset": 4}}, source_bytes=source)
        self.assertEqual(exact["provenance"], "exact_offset")
        self.assertEqual((exact["line"], exact["column"]), (1, 5))
        inherited = _location({}, {"line": 9, "column": 7, "provenance": "exact"}, source_bytes=source)
        self.assertEqual(inherited["provenance"], "inherited")
        self.assertEqual((inherited["line"], inherited["column"]), (9, 7))
        offset_ast = json.loads(AST_FIXTURE.read_text())
        function = next(node for node in _walk_dicts(offset_ast) if node.get("kind") == "FunctionDecl" and node.get("name") == "vector_add")
        function["loc"] = {"offset": source.index(b"__global__")}
        kernel, _ = import_fixture(offset_ast, source_bytes=source)
        self.assertEqual(kernel["source_location"]["provenance"], "exact_offset")
        self.assertEqual((kernel["source_location"]["line"], kernel["source_location"]["column"]), (8, 1))

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
        self.assertEqual(contract["supported_clang"]["intended_major"], 23)
        self.assertEqual(contract["supported_clang"]["validation_status"], "NOT_RUN")
        self.assertNotIn("validated_identity", contract["supported_clang"])
        self.assertNotIn("validated_major", contract["supported_clang"])
        self.assertEqual(contract["target"]["cuda_gpu_arch"], "sm_86")
        self.assertEqual(contract["header_sha256"], "e4fe928dcbe5e80cee2050f9527ca8a0fc22349192033057b56fa5ee0df46cae")
        self.assertEqual(contract["builtin_contract"]["schema"], "cuda4as-clang-cuda-facade-builtins-v1")
        self.assertEqual(set(contract["builtin_contract"]["declarations"]), {"blockIdx", "blockDim", "threadIdx"})
        self.assertEqual(contract["builtin_contract"]["device_builtin_semantics"]["status"], "IGNORED_NON_AST")
        self.assertEqual(contract["builtin_contract"]["device_builtin_semantics"]["ast_node"], 0)
        self.assertFalse(contract["builtin_contract"]["device_builtin_semantics"]["ast_json_child"])
        args = materialize_facade_arguments(CU, FACADE.parent)
        self.assertEqual(args[args.index("-I") + 1], str(FACADE.parent))
        self.assertEqual(args[-1], str(CU))
        kernel, _ = import_ast_file(AST_FIXTURE, "vector_add", source=CU, facade_contract=FACADE_CONTRACT)
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
