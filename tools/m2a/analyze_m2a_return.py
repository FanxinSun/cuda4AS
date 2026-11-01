#!/usr/bin/env python3
"""Safely validate a returned M2A Native AOT evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import tarfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CU = "b3f205cabc42a697276244d5810ce62ff40a5c9cc6a8e1078420ccbefa88d0e2"
EXPECTED_H = "6e7cc4c2e59e10db5b68df4116ed67ec3fe1d88d559c41e0a9feb2da3dd643bc"
EXPECTED_OUT = "ed551637cf393112d0093037a0b41b9d1e9bd213c6037e8efcde30cf480f0332"
EXPECTED_BYTES = 4_194_304
EXPECTED_DEVICE = "Apple M1 Pro"
EXPECTED_REGISTRY = 4_294_969_587


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_member(name: str) -> str:
    if not name or name.startswith("/") or "\\" in name:
        raise ValueError(f"unsafe return member: {name!r}")
    clean = posixpath.normpath(name)
    if clean in {"", "."} or clean == ".." or clean.startswith("../"):
        raise ValueError(f"path traversal in return member: {name!r}")
    return clean


def load_archive(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                # ``tar -C work .`` emits a harmless root/directory record;
                # validate its path but do not treat it as evidence content.
                if member.name not in {".", "./"}:
                    safe_member(member.name.rstrip("/"))
                continue
            name = safe_member(member.name)
            if member.issym() or member.islnk() or not member.isfile():
                raise ValueError(f"return contains non-regular member: {member.name}")
            if name in files:
                raise ValueError(f"duplicate return member: {name}")
            f = tar.extractfile(member)
            if f is None:
                raise ValueError(f"cannot read return member: {name}")
            files[name] = f.read()
    return files


def _find(files: dict[str, bytes], suffix: str) -> bytes | None:
    hits = [v for k, v in files.items() if k == suffix or k.endswith("/" + suffix)]
    if len(hits) > 1:
        raise ValueError(f"duplicate required return suffix: {suffix}")
    return hits[0] if hits else None


def analyze(path: Path) -> dict[str, Any]:
    files = load_archive(path)
    result_raw = _find(files, "m2a-result.json")
    if result_raw is None:
        raise ValueError("return has no m2a-result.json")
    result = json.loads(result_raw.decode("utf-8"))
    if result.get("schema") != "cuda4as-m2a-result-v1":
        raise ValueError("unexpected M2A result schema")
    classification = result.get("classification")
    if classification not in {"PASS_GPU", "FAIL", "SKIP_ENVIRONMENT", "NOT_RUN"}:
        raise ValueError("invalid M2A classification")
    expected = json.loads((_find(files, "expected.json") or b"{}").decode("utf-8"))
    if expected.get("cpu_fallback") is not False or expected.get("runtime_compilation") is not False:
        raise ValueError("package permits a prohibited fallback or runtime compilation")
    binding = json.loads((_find(files, "inventory-binding.json") or b"{}").decode("utf-8"))
    machine = binding.get("machine", {})
    devices = machine.get("metal_devices", [])
    if binding.get("schema") != "cuda4as-m2a-inventory-binding-v1" or binding.get("status") != "BOUND_TO_VALIDATED_M1_INVENTORY":
        raise ValueError("invalid inventory binding")
    source_manifest = _find(files, "PACKAGE-MANIFEST.sha256")
    stage_raw = _find(files, "stage-record.json")
    stage_record = json.loads(stage_raw.decode("utf-8")) if stage_raw else None
    aot_raw = _find(files, "aot-manifest.json")
    aot_manifest = json.loads(aot_raw.decode("utf-8")) if aot_raw else None
    source_hashes = {"cuda": EXPECTED_CU, "header": EXPECTED_H}
    if source_manifest is not None:
        text = source_manifest.decode("utf-8")
        if "oracle/src/vector_add.cu" in text and EXPECTED_CU not in text:
            raise ValueError("oracle source hash is not bound")
        if "oracle/src/oracle.h" in text and EXPECTED_H not in text:
            raise ValueError("oracle header hash is not bound")
    checks: dict[str, bool] = {
        "safe_archive_members": True,
        "immutable_source_hashes_bound": EXPECTED_CU in (source_manifest or b"").decode("utf-8") and EXPECTED_H in (source_manifest or b"").decode("utf-8"),
        "inventory_binding": bool(devices) and machine.get("arch") == "arm64" and machine.get("model") == "MacBookPro18,3",
        "fallback_prohibited": expected.get("cpu_fallback") is False,
    }
    if classification == "PASS_GPU":
        device = result.get("device", {})
        output = result.get("output", {})
        stages = result.get("stages", {})
        checks.update({
            "device_name_exact": device.get("name") == EXPECTED_DEVICE,
            "device_registry_exact": int(device.get("registry_id", -1)) == EXPECTED_REGISTRY,
            "apple_gpu_route": device.get("route") == "apple_gpu",
            "output_bytes_exact": output.get("bytes") == EXPECTED_BYTES,
            "output_sha256_exact": output.get("sha256") == EXPECTED_OUT and output.get("expected_sha256") == EXPECTED_OUT,
            "zero_mismatches": output.get("mismatches") == 0,
            "all_runtime_stages_pass": all(stages.get(k) == "PASS" for k in ("allocation", "h2d", "aot_load", "launch", "last_error", "synchronize", "d2h", "validation", "cleanup")),
            "no_cpu_fallback": result.get("cpu_fallback") is False,
            "metallib_evidence": any(k.endswith(".metallib") for k in files),
            "linked_image_evidence": any(k.endswith("device-link-image.json") for k in files),
            "aot_manifest_complete": isinstance(aot_manifest, dict) and aot_manifest.get("schema") == "cuda4as-m2a-aot-manifest-v1" and aot_manifest.get("cpu_fallback") is False and aot_manifest.get("runtime_compilation") is False,
            "stage_record_complete": isinstance(stage_record, dict) and stage_record.get("schema") == "cuda4as-m2a-stage-record-v1",
        })
        if isinstance(stage_record, dict):
            checks["all_build_stages_pass"] = all(stage_record.get("stages", {}).get(k) == "PASS" for k in ("package-verify", "preflight", "device-import", "ir_verify", "device_link", "msl_generation", "metal_compile", "metallib_link", "aot_manifest", "host_compile", "native_link", "runtime_launch"))
    valid = all(checks.values()) if classification == "PASS_GPU" else True
    if classification == "PASS_GPU" and not valid:
        classification = "FAIL"
    return {
        "schema": "cuda4as-m2a-analysis-v1",
        "archive": {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path.read_bytes()), "members": len(files)},
        "classification": classification,
        "reported_classification": result.get("classification"),
        "failed_stage": result.get("failed_stage"),
        "checks": checks,
        "source_hashes": source_hashes,
        "device": result.get("device"),
        "output": result.get("output"),
        "stages": result.get("stages"),
        "stage_record": stage_record,
        "aot_manifest": aot_manifest,
        "diagnostics": {"message": result.get("message"), "environment_gap": _find(files, "environment-gap.txt") is not None},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        doc = analyze(args.archive)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(doc["classification"])
        return 0 if doc["classification"] in {"PASS_GPU", "FAIL", "SKIP_ENVIRONMENT", "NOT_RUN"} else 1
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        print(f"INVALID: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
