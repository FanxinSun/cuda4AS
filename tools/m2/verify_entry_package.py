#!/usr/bin/env python3
"""Verify the deterministic M2 entry-retry delta before user transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
try:
    from tools.m1.analyze_native_return import _safe_name, parse_checksum_manifest, sha256
except ModuleNotFoundError:  # Direct invocation from the repository root.
    import sys

    sys.path.insert(0, str(ROOT))
    from tools.m1.analyze_native_return import _safe_name, parse_checksum_manifest, sha256


BASE_SHA256 = "dbb390b4f470b8f286ccb65a2b8565a235e749d9122c16e8e92ade96e0099bc7"
BASE_BYTES = 8549223
PATCH_SHA256 = "a192690749e55a95e9c826915c0b116fb72e604110194ca852378858c37fd9e2"
PATCH_PATH = "candidate/lower_to_llvm-array.patch"
PATCH_TARGET = "compiler/ptx/src/lower_to_llvm.cpp"


def read_archive(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            name = _safe_name(member.name)
            if not name.startswith("cuda4as-m2-entry-retry-v1/"):
                raise ValueError(f"unexpected archive member: {member.name}")
            rel = name.split("/", 1)[1]
            if not rel or member.isdir():
                continue
            if not member.isfile() or rel in files:
                raise ValueError(f"unsupported or duplicate archive member: {member.name}")
            handle = tf.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            files[rel] = handle.read()
    return files


def verify(path: Path) -> dict[str, object]:
    files = read_archive(path)
    if "PACKAGE-MANIFEST.sha256" not in files:
        raise ValueError("missing outer package manifest")
    outer = parse_checksum_manifest(files["PACKAGE-MANIFEST.sha256"], "outer package manifest")
    if set(outer) != set(files) - {"PACKAGE-MANIFEST.sha256"}:
        raise ValueError("outer package manifest member mismatch")
    for name, digest in outer.items():
        if sha256(files[name]) != digest:
            raise ValueError(f"outer package hash mismatch: {name}")

    prefix = "m2-delta/"
    delta = {name[len(prefix) :]: data for name, data in files.items() if name.startswith(prefix)}
    inner = parse_checksum_manifest(delta["PACKAGE-MANIFEST.sha256"], "inner package manifest")
    if set(inner) != set(delta) - {"PACKAGE-MANIFEST.sha256"}:
        raise ValueError("inner package manifest member mismatch")
    for name, digest in inner.items():
        if sha256(delta[name]) != digest:
            raise ValueError(f"inner package hash mismatch: {name}")

    binding = json.loads(delta["candidate/patch-binding.json"])
    if binding.get("base_artifact", {}).get("bytes") != BASE_BYTES or binding.get("base_artifact", {}).get("sha256") != BASE_SHA256:
        raise ValueError("base artifact binding mismatch")
    patch = binding.get("patch", {})
    if patch.get("path") != "candidate/lower_to_llvm-array.patch" or patch.get("changed_files") != [PATCH_TARGET] or patch.get("hunks") != 1:
        raise ValueError("patch binding is not exactly one approved target")
    if sha256(delta[PATCH_PATH]) != PATCH_SHA256 or patch.get("sha256") != PATCH_SHA256:
        raise ValueError("patch hash mismatch")
    text = delta[PATCH_PATH].decode("utf-8")
    if text.count("+#include <array>") != 1 or text.count("@@") != 2:
        raise ValueError("patch contents are not the disclosed include hunk")
    return {"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()), "members": len(files), "outer_manifest_sha256": sha256(files["PACKAGE-MANIFEST.sha256"]), "inner_manifest_sha256": sha256(delta["PACKAGE-MANIFEST.sha256"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.artifact)
    except (OSError, ValueError, tarfile.TarError, json.JSONDecodeError) as exc:
        print(f"INVALID {args.artifact}: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
