"""Phase 0 fixture checks. No GPU required."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from mag.gate import check_log

ROOT = Path(__file__).resolve().parents[1]


class GateTests(unittest.TestCase):
    def test_log_patterns(self):
        self.assertEqual(check_log((ROOT / "fixtures/gate_logs/pass.log").read_text(), 64), [])
        self.assertTrue(check_log((ROOT / "fixtures/gate_logs/no_offload.log").read_text(), 64))
        self.assertTrue(check_log((ROOT / "fixtures/gate_logs/cpu_kv.log").read_text(), 64))
        self.assertTrue(check_log((ROOT / "fixtures/gate_logs/no_projector.log").read_text(), 64))

    def test_shell_negatives_exit_1(self):
        proc = subprocess.run(["bash", str(ROOT / "scripts/negative_gate.sh")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("negative tests exited 1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
