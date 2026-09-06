from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.m1.validate_mac_inventory import EXPECTED_CHECK_IDS, normalize


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_manifest(files: dict[str, bytes]) -> None:
    files["MANIFEST.sha256"] = "".join(
        f"{digest(data)}  {name}\n" for name, data in sorted(files.items())
    ).encode()


def write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def inventory_files() -> dict[str, bytes]:
    facts = (
        "key\tvalue\n"
        "schema\tcuda4as-m1-mac-inventory-v1\n"
        "selected_developer_directory\t/Applications/Xcode.app/Contents/Developer\n"
        "selected_developer_kind\tfull_xcode\n"
    ).encode()
    statuses = []
    for check_id in sorted(EXPECTED_CHECK_IDS):
        if check_id in {"metal_device_compile", "metal_device_inventory"}:
            statuses.append((check_id, "ok", "0", "synthetic"))
        else:
            statuses.append((check_id, "missing", "127", "synthetic"))
    status = (
        "check_id\tstatus\texit_code\tcommand\n"
        + "".join("\t".join(row) + "\n" for row in statuses)
    ).encode()
    device = {
        "schema": "cuda4as-m1-metal-device-inventory-v1",
        "purpose": "synthetic validator fixture",
        "device_count": 1,
        "default_device_registry_id": "1234",
        "default_device_name": "Apple M Test",
        "devices": [
            {
                "name": "Apple M Test",
                "registry_id": "1234",
                "is_low_power": False,
            }
        ],
    }
    files = {
        "facts.tsv": facts,
        "status.tsv": status,
        "inventory.txt": b"synthetic complete inventory\n",
        "logs/metal_device_inventory.stdout.json": (
            json.dumps(device, indent=2) + "\n"
        ).encode(),
    }
    add_manifest(files)
    return files


class MacInventoryValidatorTests(unittest.TestCase):
    def archive(self, files: dict[str, bytes]) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "inventory-return.tgz"
        write_tar(path, files)
        return temp, path

    def test_complete_inventory_is_normalized_with_device_identity(self) -> None:
        temp, path = self.archive(inventory_files())
        self.addCleanup(temp.cleanup)
        result = normalize(path)
        self.assertEqual(result["inventory_status"], "COMPLETE")
        self.assertEqual(result["machine"]["default_metal_device_name"], "Apple M Test")
        self.assertEqual(result["machine"]["default_metal_device_status"], "MATCHED")
        self.assertEqual(result["machine"]["metal_devices"][0]["registry_id"], "1234")

    def test_absent_system_default_is_preserved_with_enumerated_device(self) -> None:
        files = inventory_files()
        device = json.loads(files["logs/metal_device_inventory.stdout.json"])
        device["default_device_name"] = None
        device["default_device_registry_id"] = None
        files["logs/metal_device_inventory.stdout.json"] = (
            json.dumps(device, indent=2) + "\n"
        ).encode()
        files.pop("MANIFEST.sha256")
        add_manifest(files)
        temp, path = self.archive(files)
        self.addCleanup(temp.cleanup)
        result = normalize(path)
        self.assertEqual(result["inventory_status"], "COMPLETE")
        self.assertEqual(result["machine"]["default_metal_device_status"], "UNAVAILABLE")
        self.assertEqual(len(result["machine"]["metal_devices"]), 1)

    def test_partial_default_identity_is_rejected(self) -> None:
        files = inventory_files()
        device = json.loads(files["logs/metal_device_inventory.stdout.json"])
        device["default_device_name"] = None
        files["logs/metal_device_inventory.stdout.json"] = (
            json.dumps(device, indent=2) + "\n"
        ).encode()
        files.pop("MANIFEST.sha256")
        add_manifest(files)
        temp, path = self.archive(files)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "both null or a nonempty"):
            normalize(path)

    def test_missing_check_row_is_rejected(self) -> None:
        files = inventory_files()
        lines = files["status.tsv"].decode().splitlines()
        files["status.tsv"] = ("\n".join(line for line in lines if not line.startswith("cmake_path\t")) + "\n").encode()
        files.pop("MANIFEST.sha256")
        add_manifest(files)
        temp, path = self.archive(files)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "inventory check set mismatch"):
            normalize(path)

    def test_duplicate_fact_is_rejected(self) -> None:
        files = inventory_files()
        files["facts.tsv"] += b"schema\tcuda4as-m1-mac-inventory-v1\n"
        files.pop("MANIFEST.sha256")
        add_manifest(files)
        temp, path = self.archive(files)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "duplicate facts key"):
            normalize(path)


if __name__ == "__main__":
    unittest.main()
