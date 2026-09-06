from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY / "docs/m1/fixtures.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def u32(value: int) -> int:
    return value & 0xFFFFFFFF


class FixtureManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text())
        cls.cases = {case["id"]: case for case in cls.manifest["cases"]}

    def test_all_pinned_inputs_exist_and_match(self) -> None:
        self.assertEqual(
            len(self.cases),
            len(self.manifest["cases"]),
            "fixture IDs must be unique",
        )
        for case in self.manifest["cases"]:
            for record in case["source_files"] + case["build_files"]:
                path = REPOSITORY / record["path"]
                self.assertTrue(path.is_file(), record["path"])
                self.assertEqual(sha256(path.read_bytes()), record["sha256"])

    def test_existing_oracle_bytes_match(self) -> None:
        case = self.cases["oracle.vector_add"]
        expected = case["expected_output"]
        data = (REPOSITORY / "oracle/ref/vector_add.bin").read_bytes()
        self.assertEqual(len(data), expected["bytes"])
        self.assertEqual(sha256(data), expected["sha256"])

    def test_cmake_vector_add_expectation_is_independent(self) -> None:
        values = []
        for index in range(4096):
            a = u32(index * 2654435761 + 17)
            b = u32((index ^ 0xA5A5A5A5) * 2246822519)
            values.append(u32(a + b))
        data = b"".join(struct.pack("<I", value) for value in values)
        expected = self.cases["integration.minimal_cmake_cuda"]["expected_output"]
        self.assertEqual(expected["byte_order"], "little-endian")
        self.assertEqual(len(data), expected["bytes"])
        self.assertEqual(sha256(data), expected["sha256"])

    def test_device_link_expectation_is_independent(self) -> None:
        values = []
        for index in range(2048):
            value = u32(index * 747796405 + 2891336453)
            value = u32(value ^ u32(index * 0x9E3779B9))
            value = u32((value << 7) | (value >> 25))
            values.append(u32(value * 2246822519 + 3266489917))
        data = b"".join(struct.pack("<I", value) for value in values)
        expected = self.cases["integration.multi_tu_device_link"]["expected_output"]
        self.assertEqual(expected["byte_order"], "little-endian")
        self.assertEqual(len(data), expected["bytes"])
        self.assertEqual(sha256(data), expected["sha256"])


if __name__ == "__main__":
    unittest.main()
