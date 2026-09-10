#!/usr/bin/env python3
"""Bind AOT outputs, ABI/IR identities, and native tool provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def tool_identity(path: str) -> dict[str, object]:
    try:
        p = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
        return {"path": path, "exit_code": p.returncode, "version": (p.stdout or p.stderr).strip()[:1000]}
    except OSError as exc:
        return {"path": path, "exit_code": 127, "version": str(exc)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("work", type=Path)
    p.add_argument("kernel")
    p.add_argument("metal")
    p.add_argument("metallib")
    p.add_argument("clang")
    p.add_argument("sdk")
    args = p.parse_args()
    w = args.work
    ir = json.loads((w / "ir.json").read_text(encoding="utf-8"))
    abi = json.loads((w / "abi.json").read_text(encoding="utf-8"))
    link = json.loads((w / "device-link.json").read_text(encoding="utf-8"))
    lib = w / "metallib" / f"{args.kernel}.metallib"
    files = {name: digest(w / name) for name in ("ir.json", "abi.json", "device-link.json", "device-link-image.json", "kernel.metal", "clang-device-ast.json", "device-source.cu")}
    facade_dir = Path(__file__).resolve().parent / "clang_cuda_facade"
    facade = facade_dir / "cuda_runtime.h"
    facade_manifest_path = facade_dir / "manifest.json"
    facade_manifest = json.loads(facade_manifest_path.read_text(encoding="utf-8"))
    if facade_manifest.get("schema") != "cuda4as-clang-cuda-facade-manifest-v1":
        raise ValueError("unsupported Clang CUDA facade manifest schema")
    if digest(facade) != facade_manifest.get("header", {}).get("sha256"):
        raise ValueError("Clang CUDA facade header does not match its manifest")
    builtin_contract = facade_manifest.get("builtin_contract")
    if not isinstance(builtin_contract, dict) or builtin_contract.get("schema") != "cuda4as-clang-cuda-facade-builtins-v1":
        raise ValueError("Clang CUDA facade manifest has no builtin declaration contract")
    if (
        builtin_contract.get("header_path") != facade_manifest.get("header", {}).get("path")
        or builtin_contract.get("header_sha256") != facade_manifest.get("header", {}).get("sha256")
    ):
        raise ValueError("Clang CUDA facade builtin contract is not bound to its header")
    declarations = builtin_contract.get("declarations")
    if not isinstance(declarations, dict) or set(declarations) != {"blockIdx", "blockDim", "threadIdx"}:
        raise ValueError("Clang CUDA facade builtin contract has the wrong declaration set")
    facade_lines = facade.read_text(encoding="utf-8").splitlines()
    for name, declaration in declarations.items():
        if (
            not isinstance(declaration, dict)
            or declaration.get("qual_type") != "const dim3"
            or not isinstance(declaration.get("line"), int)
            or not isinstance(declaration.get("column"), int)
            or not isinstance(declaration.get("source_anchor"), str)
            or declaration["line"] <= 0
            or declaration["column"] <= 0
            or declaration["line"] > len(facade_lines)
            or facade_lines[declaration["line"] - 1].strip() != declaration["source_anchor"]
        ):
            raise ValueError(f"Clang CUDA facade builtin contract anchor is invalid for {name}")
    supported_clang = facade_manifest.get("supported_clang")
    if (
        not isinstance(supported_clang, dict)
        or not isinstance(supported_clang.get("intended_identity"), str)
        or not isinstance(supported_clang.get("intended_major"), int)
        or supported_clang.get("validation_status") not in {"NOT_RUN", "VALIDATED"}
        or "validated_identity" in supported_clang
        or "validated_major" in supported_clang
    ):
        raise ValueError("Clang CUDA facade manifest has an invalid intended/validation identity")
    files["tools/m2a/clang_cuda_facade/cuda_runtime.h"] = digest(facade)
    files["tools/m2a/clang_cuda_facade/manifest.json"] = digest(facade_manifest_path)
    files["metallib"] = digest(lib)
    doc = {
        "schema": "cuda4as-m2a-aot-manifest-v1", "version": 1, "kernel": args.kernel,
        "numerical_mode": ir["module"]["numerical_mode"], "capabilities": sorted(ir["capabilities"]),
        "source": ir["source"], "files": files, "abi_schema": abi["schema"], "device_link_schema": link["schema"],
        "facade": {
            "version": facade_manifest["version"],
            "manifest": "tools/m2a/clang_cuda_facade/manifest.json",
            "manifest_sha256": files["tools/m2a/clang_cuda_facade/manifest.json"],
            "header": facade_manifest["header"]["path"],
            "header_sha256": facade_manifest["header"]["sha256"],
            "builtin_contract": builtin_contract,
            "supported_clang": supported_clang,
            "target": facade_manifest["target"],
            "compiler_arguments": facade_manifest["compiler_arguments"],
        },
        "tools": {"clang": tool_identity(args.clang), "metal": tool_identity(args.metal), "metallib": tool_identity(args.metallib), "sdk_path": args.sdk},
        "semantic_validation_status": ir.get("validation_status"),
        "compiler_validation_status": ir.get("compiler_validation_status"),
        "cpu_fallback": False, "runtime_compilation": False, "device_link_empty": link["empty"],
    }
    (w / "aot-manifest.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
