#!/usr/bin/env python3
"""Build a deterministic, self-contained M1 native-Mac feasibility drop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
from typing import Iterable

try:
    from .validate_mac_inventory import normalize as normalize_inventory
    from .assess_mac_inventory import assess as assess_inventory
except ImportError:  # Direct invocation from the repository root.
    from validate_mac_inventory import normalize as normalize_inventory
    from assess_mac_inventory import assess as assess_inventory


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PREFIX = "cuda4as-m1-native-feasibility-v1"
OUTPUT = ROOT / "dist/m1/cuda4as-m1-native-feasibility-v1.tgz"
ARTIFACT_MANIFEST = ROOT / "docs/m1/native-feasibility-artifact.json"
CANDIDATE_REVISION = "f486e5ebcfd381d06e3297afd65dbcbd5006a902"
VF64_REVISION = "729021777455da72db8809d9ef1269c677d88b3f"


@dataclass(frozen=True)
class Member:
    path: str
    data: bytes
    mode: int = 0o644


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_member(path: str, source: Path, mode: int = 0o644) -> Member:
    return Member(path, source.read_bytes(), mode)


def checksum_lines(members: Iterable[Member], prefix: str = "") -> bytes:
    lines = [f"{digest(member.data)}  {prefix}{member.path}\n" for member in members]
    return "".join(lines).encode()


def candidate_tree_members() -> tuple[bytes, bytes, int, int]:
    source = (
        ROOT
        / ".m1-work/source-v1"
        / f"cuda-metal-{CANDIDATE_REVISION}"
    )
    if not source.is_dir():
        raise SystemExit(f"missing verified combined source tree: {source}")
    records: list[tuple[str, bytes]] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        records.append((relative, path.read_bytes()))
    checksums = "".join(f"{digest(data)}  {path}\n" for path, data in records).encode()
    paths = "".join(f"{path}\n" for path, _ in records).encode()
    return checksums, paths, len(records), sum(len(data) for _, data in records)


def inventory_binding(path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "schema": "cuda4as-m1-inventory-binding-v1",
            "status": "PENDING_USER_INVENTORY",
            "inventory_archive": None,
        }
    normalized = normalize_inventory(path)
    if normalized.get("inventory_status") != "COMPLETE":
        raise SystemExit("inventory return is valid but Metal device inventory is incomplete")
    machine = normalized.get("machine", {})
    if machine.get("arch") != "arm64" or not machine.get("metal_devices"):
        raise SystemExit("inventory return does not identify an Apple-Silicon Metal device")
    readiness = assess_inventory(normalized)
    if not readiness["ready_for_native_drop"]:
        raise SystemExit(
            "inventory does not satisfy native-run prerequisites: "
            + ", ".join(readiness["gaps"])
        )
    data = path.read_bytes()
    return {
        "schema": "cuda4as-m1-inventory-binding-v1",
        "status": "BOUND_TO_RETURNED_INVENTORY",
        "inventory_archive": {
            "filename": path.name,
            "bytes": len(data),
            "sha256": digest(data),
        },
    }


def build_members(binding: dict[str, object]) -> list[Member]:
    source_archives = ROOT / ".m1-work/downloads"
    expected_outputs = {
        "fixtures/expected/oracle-vector-add.bin": ROOT / "oracle/ref/vector_add.bin",
        "fixtures/expected/cmake-vector-add.bin": (
            ROOT / ".m1-work/nvidia-cmake-vector/output.bin"
        ),
        "fixtures/expected/cmake-device-link.bin": (
            ROOT / ".m1-work/nvidia-cmake-device-link/output.bin"
        ),
    }
    application_sources = {
        "fixtures/oracle/vector_add.cu": ROOT / "oracle/src/vector_add.cu",
        "fixtures/oracle/oracle.h": ROOT / "oracle/src/oracle.h",
        "fixtures/cmake-vector-add/CMakeLists.txt": (
            ROOT / "tests/m1/fixtures/cmake-vector-add/CMakeLists.txt"
        ),
        "fixtures/cmake-vector-add/main.cu": (
            ROOT / "tests/m1/fixtures/cmake-vector-add/main.cu"
        ),
        "fixtures/cmake-device-link/CMakeLists.txt": (
            ROOT / "tests/m1/fixtures/cmake-device-link/CMakeLists.txt"
        ),
        "fixtures/cmake-device-link/device-api.cuh": (
            ROOT / "tests/m1/fixtures/cmake-device-link/device-api.cuh"
        ),
        "fixtures/cmake-device-link/device-function.cu": (
            ROOT / "tests/m1/fixtures/cmake-device-link/device-function.cu"
        ),
        "fixtures/cmake-device-link/kernel.cu": (
            ROOT / "tests/m1/fixtures/cmake-device-link/kernel.cu"
        ),
        "fixtures/cmake-device-link/main.cu": (
            ROOT / "tests/m1/fixtures/cmake-device-link/main.cu"
        ),
    }
    members = [
        read_member("README.md", ROOT / "tools/m1/native/README.md"),
        read_member(
            "run-native-feasibility.sh",
            ROOT / "tools/m1/native/run-native-feasibility.sh",
            0o755,
        ),
        read_member(
            "tools/prepare-cmake-adapter.sh",
            ROOT / "tools/m1/native/prepare-cmake-adapter.sh",
            0o755,
        ),
        read_member(
            "candidate/cuda-metal-f486e5eb.tar.gz",
            source_archives / "cuda-metal-f486e5eb.tar.gz",
        ),
        read_member(
            "candidate/vf64-metal-72902177.tar.gz",
            source_archives / "vf64-metal-72902177.tar.gz",
        ),
        read_member("fixtures/fixtures.json", ROOT / "docs/m1/fixtures.json"),
        read_member(
            "contracts/result-schema-v1.md", ROOT / "docs/m1/result-schema-v1.md"
        ),
        read_member(
            "contracts/result-schema-v1.schema.json",
            ROOT / "docs/m1/result-schema-v1.schema.json",
        ),
        Member(
            "target-inventory-binding.json",
            (json.dumps(binding, indent=2) + "\n").encode(),
        ),
    ]
    for path, source in sorted(application_sources.items()):
        members.append(read_member(path, source))
    for path, source in sorted(expected_outputs.items()):
        members.append(read_member(path, source))

    tree_hashes, tree_paths, tree_count, tree_bytes = candidate_tree_members()
    members.extend(
        [
            Member("candidate/combined-tree.sha256", tree_hashes),
            Member("candidate/combined-tree-files.txt", tree_paths),
            Member(
                "candidate/combined-tree-summary.json",
                (
                    json.dumps(
                        {
                            "schema": "cuda4as-m1-combined-candidate-tree-v1",
                            "candidate_revision": CANDIDATE_REVISION,
                            "vf64_revision": VF64_REVISION,
                            "files": tree_count,
                            "bytes": tree_bytes,
                        },
                        indent=2,
                    )
                    + "\n"
                ).encode(),
            ),
        ]
    )

    input_members = [member for member in members if member.path.startswith("fixtures/")]
    members.append(Member("fixtures/INPUTS.sha256", checksum_lines(input_members)))
    return sorted(members, key=lambda member: member.path)


def write_archive(members: list[Member], output: Path) -> bytes:
    manifest = Member("PACKAGE-MANIFEST.sha256", checksum_lines(members))
    archive_members = sorted([*members, manifest], key=lambda member: member.path)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for member in archive_members:
                info = tarfile.TarInfo(f"{ARCHIVE_PREFIX}/{member.path}")
                info.size = len(member.data)
                info.mode = member.mode
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                tf.addfile(info, io.BytesIO(member.data))
    data = buffer.getvalue()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory-archive",
        type=Path,
        help="validated user-returned inventory archive to bind into the drop",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=ARTIFACT_MANIFEST)
    args = parser.parse_args()

    if args.inventory_archive is not None and not args.inventory_archive.is_file():
        parser.error(f"inventory archive does not exist: {args.inventory_archive}")
    binding = inventory_binding(args.inventory_archive)
    members = build_members(binding)
    first = write_archive(members, args.output)
    second = write_archive(members, args.output.with_suffix(args.output.suffix + ".repeat"))
    deterministic = first == second
    args.output.with_suffix(args.output.suffix + ".repeat").unlink()
    if not deterministic:
        raise SystemExit("nondeterministic archive output")

    archive_manifest = {
        "schema": "cuda4as-m1-artifact-manifest-v1",
        "artifact_id": ARCHIVE_PREFIX,
        "status": binding["status"],
        "m1_baseline": "ce4a1c2e60688e9ea7641202cac3d2036529ffe6",
        "candidate_revision": CANDIDATE_REVISION,
        "vf64_revision": VF64_REVISION,
        "generated_path": args.output.relative_to(ROOT).as_posix(),
        "bytes": len(first),
        "sha256": digest(first),
        "archive_prefix": ARCHIVE_PREFIX,
        "files": [
            {
                "path": member.path,
                "mode": f"{member.mode:04o}",
                "bytes": len(member.data),
                "sha256": digest(member.data),
            }
            for member in sorted(
                [*members, Member("PACKAGE-MANIFEST.sha256", checksum_lines(members))],
                key=lambda item: item.path,
            )
        ],
        "inventory_binding": binding,
        "behavior": {
            "network": "none",
            "sudo": False,
            "installs_or_updates": False,
            "writes": "only beneath the extracted artifact directory",
            "candidate_configuration": {
                "build_type": "Release",
                "registration": "ON",
                "binary_shim": "OFF",
                "cuda_arch": "sm_86",
                "fp64_policy": "ieee64",
            },
        },
        "local_validation": {
            "deterministic_packaging": deterministic,
            "native_macos_run": "NOT_RUN",
        },
    }
    args.manifest.write_text(json.dumps(archive_manifest, indent=2) + "\n")
    print(f"{args.output} {len(first)} bytes {digest(first)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
