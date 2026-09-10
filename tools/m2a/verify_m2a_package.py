#!/usr/bin/env python3
"""Verify an extracted M2A package without following links or paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "PACKAGE-MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "cuda4as-m2a-package-manifest-v1":
        raise SystemExit("invalid package manifest schema")
    expected = {}
    for item in manifest.get("files", []):
        rel = item.get("path")
        if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in Path(rel).parts or rel in expected:
            raise SystemExit(f"unsafe or duplicate member: {rel!r}")
        expected[rel] = item
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and p.name not in {"PACKAGE-MANIFEST.json", "PACKAGE-MANIFEST.sha256"}
        and p.relative_to(root).parts[:1] not in (("results",), ("returns",))
    }
    if actual != set(expected):
        raise SystemExit(f"package member mismatch: missing={sorted(set(expected)-actual)} extra={sorted(actual-set(expected))}")
    lines = (root / "PACKAGE-MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    recorded = {}
    for line in lines:
        digest, rel = line.split("  ", 1)
        recorded[rel] = digest
    if set(recorded) != set(expected):
        raise SystemExit("checksum member mismatch")
    for rel, item in expected.items():
        path = root / rel
        if path.is_symlink() or not path.is_file() or sha256(path) != item["sha256"] or sha256(path) != recorded[rel] or path.stat().st_size != item["bytes"]:
            raise SystemExit(f"package file verification failed: {rel}")
    binding = json.loads((root / "inventory-binding.json").read_text(encoding="utf-8"))
    if binding.get("schema") != "cuda4as-m2a-inventory-binding-v1" or binding.get("status") != "BOUND_TO_VALIDATED_M1_INVENTORY":
        raise SystemExit("inventory binding is not validated")
    expected_doc = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    if expected_doc.get("cpu_fallback") is not False or expected_doc.get("runtime_compilation") is not False:
        raise SystemExit("prohibited fallback/runtime compilation flag")
    print("M2A_PACKAGE_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
