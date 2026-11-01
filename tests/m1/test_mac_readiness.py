from __future__ import annotations

from copy import deepcopy
import unittest

from tools.m1.assess_mac_inventory import assess, version_at_least


def tool(stdout: str = "ok") -> dict:
    return {"status": "ok", "exit_code": "0", "command": "synthetic", "stdout": stdout}


def ready_inventory() -> dict:
    return {
        "schema": "cuda4as-m1-mac-inventory-normalized-v1",
        "source_archive": {
            "filename": "inventory.tgz",
            "bytes": 1,
            "sha256": "a" * 64,
        },
        "inventory_status": "COMPLETE",
        "machine": {
            "os_name": "macOS",
            "os_version": "15.6",
            "arch": "arm64",
            "metal_devices": [{"name": "Apple M4", "registry_id": "1"}],
            "task_filesystem_free_kib": 6 * 1024 * 1024,
        },
        "tools": {
            "sdk_path": tool("/SDK"),
            "sdk_version": tool("15.5"),
            "find_metal": tool("/SDK/usr/bin/metal"),
            "find_metallib": tool("/SDK/usr/bin/metallib"),
            "cmake_path": tool("/opt/homebrew/bin/cmake"),
            "cmake_version": tool("cmake version 3.31.8"),
            "ninja_path": tool("/opt/homebrew/bin/ninja"),
            "ninja_version": tool("1.12.1"),
            "homebrew_llvm_version": tool("19.1.7"),
            "intel_homebrew_llvm_version": {
                "status": "missing",
                "exit_code": "127",
                "command": "synthetic",
                "stdout": None,
            },
            "llvm_config_version": tool("19.1.7"),
            "pkg_config_lz4": tool("1.10.0"),
            "pkg_config_zstd": tool("1.5.7"),
            "brew_components": tool("cmake 3.31.8\nninja 1.12.1\nllvm 19.1.7\nlz4 1.10.0\nzstd 1.5.7"),
            "macports_components": {
                "status": "missing",
                "exit_code": "127",
                "command": "synthetic",
                "stdout": None,
            },
        },
    }


class MacReadinessTests(unittest.TestCase):
    def test_ready_inventory_satisfies_every_bounded_runner_requirement(self) -> None:
        result = assess(ready_inventory())
        self.assertTrue(result["ready_for_native_drop"])
        self.assertEqual(result["gaps"], [])
        self.assertTrue(all(item["state"] == "READY" for item in result["requirements"]))

    def test_old_cmake_and_missing_ninja_are_explicit_gaps(self) -> None:
        inventory = deepcopy(ready_inventory())
        inventory["tools"]["cmake_version"]["stdout"] = "cmake version 3.27.9"
        inventory["tools"]["ninja_path"]["status"] = "missing"
        result = assess(inventory)
        self.assertFalse(result["ready_for_native_drop"])
        self.assertEqual(result["gaps"], ["cmake", "ninja"])

    def test_version_comparison_handles_major_and_patch_boundaries(self) -> None:
        self.assertTrue(version_at_least("Homebrew LLVM version 18.0.0", (18, 0)))
        self.assertTrue(version_at_least("cmake version 4.2.3", (3, 28)))
        self.assertFalse(version_at_least("17.0.6", (18, 0)))
        self.assertFalse(version_at_least("missing", (18, 0)))

    def test_runner_selection_does_not_skip_an_old_first_llvm(self) -> None:
        inventory = ready_inventory()
        inventory["tools"]["homebrew_llvm_version"]["stdout"] = "17.0.6"
        inventory["tools"]["llvm_config_version"]["stdout"] = "19.1.7"
        result = assess(inventory)
        self.assertIn("llvm", result["gaps"])


if __name__ == "__main__":
    unittest.main()
