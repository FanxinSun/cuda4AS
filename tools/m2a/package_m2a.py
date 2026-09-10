#!/usr/bin/env python3
"""Build the deterministic, inventory-bound M2A Mac drop."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ID = "cuda4as-m2a-native-aot-vector-add-v1"
INVENTORY = ROOT / ".m1-work/validated/20260907T005814Z/inventory.normalized.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()


def make_tar(src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                members = sorted((p for p in src.rglob("*") if p.is_file()), key=lambda p: p.relative_to(src).as_posix())
                for path in members:
                    rel = path.relative_to(src).as_posix()
                    info = tar.gettarinfo(str(path), arcname=f"{PACKAGE_ID}/{rel}")
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.pax_headers = {}
                    with path.open("rb") as fh:
                        tar.addfile(info, fh)


def build(output: Path) -> dict[str, object]:
    output = output.resolve()
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    machine = inventory["machine"]
    binding = {
        "schema": "cuda4as-m2a-inventory-binding-v1",
        "status": "BOUND_TO_VALIDATED_M1_INVENTORY",
        "inventory_archive": inventory["source_archive"],
        "machine": {
            "arch": machine["arch"], "model": machine["model"], "cpu_brand": machine["cpu_brand"],
            "os_version": machine["os_version"], "os_build": machine["os_build"], "sdk_version": machine["sdk_version"],
            "xcode_version": machine["xcode_version"], "metal_compiler": machine["metal_compiler"],
            "metal_devices": machine["metal_devices"],
        },
        "llvm": {"origin": "/opt/homebrew/opt/llvm", "version": "23.1.0", "minimum_major": 18},
        "limits": {"max_jobs": 4, "min_free_gib": 5, "max_task_disk_gib": 20, "max_minutes": 30},
    }
    expected = {
        "schema": "cuda4as-m2a-package-v1", "package_id": PACKAGE_ID,
        "source": {"cuda": "oracle/src/vector_add.cu", "header": "oracle/src/oracle.h"},
        "output": {"bytes": 4194304, "sha256": "ed551637cf393112d0093037a0b41b9d1e9bd213c6037e8efcde30cf480f0332", "comparison": "exact"},
        "route": "apple_gpu", "cpu_fallback": False, "runtime_compilation": False,
        "aot_tools": ["metal", "metallib"], "schemas": ["cuda4as-device-ssa-ir-v1", "cuda4as-native-aot-abi-v1", "cuda4as-device-link-v1"],
    }
    with tempfile.TemporaryDirectory(prefix="cuda4as-m2a-package-") as td:
        stage = Path(td)
        files: dict[str, bytes] = {}
        files["oracle/src/vector_add.cu"] = (ROOT / "oracle/src/vector_add.cu").read_bytes()
        files["oracle/src/oracle.h"] = (ROOT / "oracle/src/oracle.h").read_bytes()
        files["tools/m2a/native_aot.py"] = (ROOT / "tools/m2a/native_aot.py").read_bytes()
        files["tools/m2a/preflight.py"] = (ROOT / "tools/m2a/preflight.py").read_bytes()
        files["tools/m2a/aot_manifest.py"] = (ROOT / "tools/m2a/aot_manifest.py").read_bytes()
        files["tools/m2a/runtime.mm"] = (ROOT / "tools/m2a/runtime.mm").read_bytes()
        files["tools/m2a/verify_m2a_package.py"] = (ROOT / "tools/m2a/verify_m2a_package.py").read_bytes()
        files["inventory-binding.json"] = canonical(binding)
        files["expected.json"] = canonical(expected)
        readme = f"""# {PACKAGE_ID}\n\nThis is the user-operated M2A Native AOT Core v1 drop. It performs no network, install, update, sudo, or CPU fallback. The runner writes all evidence beneath this extracted directory and returns one archive.\n\nBound machine: {machine['model']} / {machine['cpu_brand']} / {machine['os_version']} ({machine['os_build']}); Xcode {machine['xcode_version'].replace(chr(10), ' ')}; SDK {machine['sdk_version']}; LLVM 23.1.0.\n\nThe package contains the unchanged oracle source and header. Do not edit them.\n""".encode()
        files["README.md"] = readme
        for rel, data in files.items():
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        runner = (ROOT / "tools/m2a/run-m2a-native.sh").read_bytes()
        (stage / "run-m2a-native.sh").write_bytes(runner)
        (stage / "run-m2a-native.sh").chmod(0o755)
        entries = {rel: {"bytes": len(data), "sha256": digest(data), "mode": "0755" if rel == "run-m2a-native.sh" else "0644"} for rel, data in files.items()}
        entries["run-m2a-native.sh"] = {"bytes": len(runner), "sha256": digest(runner), "mode": "0755"}
        manifest = {"schema": "cuda4as-m2a-package-manifest-v1", "package_id": PACKAGE_ID, "files": [{"path": p, **entries[p]} for p in sorted(entries)]}
        (stage / "PACKAGE-MANIFEST.json").write_bytes(canonical(manifest))
        manifest_sha = "\n".join(f"{entries[p]['sha256']}  {p}" for p in sorted(entries)) + "\n"
        (stage / "PACKAGE-MANIFEST.sha256").write_text(manifest_sha, encoding="utf-8")
        make_tar(stage, output)
    try:
        generated_path = str(output.relative_to(ROOT))
    except ValueError:
        generated_path = str(output)
    metadata = {"schema": "cuda4as-m2a-artifact-manifest-v1", "artifact_id": PACKAGE_ID, "generated_path": generated_path, "bytes": output.stat().st_size, "sha256": digest(output.read_bytes()), "archive_prefix": PACKAGE_ID, "inventory_binding": binding, "files": entries}
    output.with_suffix(".manifest.json").write_bytes(canonical(metadata))
    return metadata


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "tools/m2a/artifacts" / f"{PACKAGE_ID}.tgz")
    args = p.parse_args()
    metadata = build(args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
