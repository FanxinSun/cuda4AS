#!/usr/bin/env python3
"""Build the deterministic, ignored cuda4AS M1 Mac-inventory archive."""

from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import tarfile


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "tools" / "m1" / "mac-inventory"
OUTPUT = REPO / "dist" / "m1" / "cuda4as-m1-mac-inventory-v1.tgz"
PREFIX = "cuda4as-m1-mac-inventory-v1"
FILES = (
    "README.md",
    "run-inventory.sh",
    "metal-device-inventory.swift",
)


def add_directory(archive: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name=name.rstrip("/") + "/")
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    archive.addfile(info)


def add_file(archive: tarfile.TarFile, relative: str) -> None:
    data = (SOURCE / relative).read_bytes()
    info = tarfile.TarInfo(name=f"{PREFIX}/{relative}")
    info.size = len(data)
    info.mode = 0o755 if relative == "run-inventory.sh" else 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def main() -> None:
    missing = [str(SOURCE / name) for name in FILES if not (SOURCE / name).is_file()]
    if missing:
        raise SystemExit("missing inventory source: " + ", ".join(missing))

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        add_directory(archive, PREFIX)
        for name in sorted(FILES):
            add_file(archive, name)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(tar_buffer.getvalue())

    data = OUTPUT.read_bytes()
    print(f"path={OUTPUT}")
    print(f"bytes={len(data)}")
    print(f"sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
