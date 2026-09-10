from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class EvidenceContractTests(unittest.TestCase):
    def test_historical_m2_analysis_distinguishes_integrity_and_build(self) -> None:
        path = ROOT / "RESULTS/m2/native-entry-20260910T151705Z.analysis.json"
        if not path.is_file():
            self.skipTest("historical M2 analysis is unavailable")
        gate = json.loads(path.read_text(encoding="utf-8"))["candidate_gate"]
        self.assertTrue(gate["integrity_pass"])
        self.assertTrue(gate["complete_pass"])  # compatibility alias only
        self.assertEqual(gate["build_outcome"], "FAIL")
        self.assertFalse(gate["build_pass"])


if __name__ == "__main__":
    unittest.main()
