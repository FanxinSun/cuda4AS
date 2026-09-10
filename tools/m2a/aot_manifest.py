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
    files = {name: digest(w / name) for name in ("ir.json", "abi.json", "device-link.json", "device-link-image.json", "kernel.metal", "clang-device-ast.json", "device-split.cu")}
    files["metallib"] = digest(lib)
    doc = {
        "schema": "cuda4as-m2a-aot-manifest-v1", "version": 1, "kernel": args.kernel,
        "numerical_mode": ir["module"]["numerical_mode"], "capabilities": sorted(ir["capabilities"]),
        "source": ir["source"], "files": files, "abi_schema": abi["schema"], "device_link_schema": link["schema"],
        "tools": {"clang": tool_identity(args.clang), "metal": tool_identity(args.metal), "metallib": tool_identity(args.metallib), "sdk_path": args.sdk},
        "cpu_fallback": False, "runtime_compilation": False, "device_link_empty": link["empty"],
    }
    (w / "aot-manifest.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
