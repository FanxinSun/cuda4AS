from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from tools.m1.analyze_native_return import load_verified_files
from tools.m2.analyze_entry_return import validate_package_and_patch


ROOT = Path(__file__).resolve().parents[2]
RETURN_GLOB = ROOT / "dist/m2/test-run"
INVENTORY = ROOT / ".m1-work/validated/20260907T005814Z/inventory.normalized.json"


class M2EntryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        returns = sorted(RETURN_GLOB.rglob("cuda4as-m2-native-return-*.tgz")) if RETURN_GLOB.exists() else []
        if not returns or not INVENTORY.is_file():
            raise unittest.SkipTest("local M2 controlled return or inventory is unavailable")
        cls.archive = returns[-1]
        cls.inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
        cls.files, _ = load_verified_files(cls.archive)

    @staticmethod
    def _refresh_delta_hash(files: dict[str, bytes], member: str) -> None:
        manifest = "package/m2-delta/PACKAGE-MANIFEST.sha256"
        digest = hashlib.sha256(files[member].replace(b"#include <array>", b"#include <vector>")).hexdigest() if member.endswith("lower_to_llvm-array.patch") else hashlib.sha256(files[member]).hexdigest()
        lines = []
        for line in files[manifest].decode().splitlines():
            old_digest, name = line.split("  ", 1)
            lines.append(f"{digest if name == member.removeprefix('package/m2-delta/') else old_digest}  {name}")
        files[manifest] = ("\n".join(lines) + "\n").encode()

    def test_bound_delta_is_accepted(self) -> None:
        details = validate_package_and_patch(self.files, self.inventory)
        self.assertEqual(details["patch_binding_sha256"], "d0866ca748e6a8e4596f75626ae4571da9b2407a7a8613774293013c9bb6e4e7")

    def test_different_patch_bytes_are_rejected(self) -> None:
        files = dict(self.files)
        path = "package/m2-delta/candidate/lower_to_llvm-array.patch"
        files[path] = files[path].replace(b"#include <array>", b"#include <vector>")
        self._refresh_delta_hash(files, path)
        with self.assertRaisesRegex(ValueError, "patch bytes/hash mismatch"):
            validate_package_and_patch(files, self.inventory)

    def test_extra_changed_file_in_binding_is_rejected(self) -> None:
        files = dict(self.files)
        path = "package/m2-delta/candidate/patch-binding.json"
        binding = json.loads(files[path])
        binding["patch"]["changed_files"] = [
            "compiler/ptx/src/lower_to_llvm.cpp",
            "compiler/ptx/src/parser.cpp",
        ]
        files[path] = (json.dumps(binding, indent=2) + "\n").encode()
        self._refresh_delta_hash(files, path)
        with self.assertRaisesRegex(ValueError, "M2 patch binding mismatch for changed_files"):
            validate_package_and_patch(files, self.inventory)

    def test_clean_tree_identity_is_bound(self) -> None:
        files = dict(self.files)
        path = "package/m2-delta/candidate/patch-binding.json"
        binding = json.loads(files[path])
        binding["tree_identity"]["clean_manifest_sha256"] = "0" * 64
        files[path] = (json.dumps(binding, indent=2) + "\n").encode()
        self._refresh_delta_hash(files, path)
        with self.assertRaisesRegex(ValueError, "clean/patched tree identity mismatch"):
            validate_package_and_patch(files, self.inventory)


if __name__ == "__main__":
    unittest.main()
