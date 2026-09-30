"""Phase 0 fixture checks. No GPU required."""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

from mag.gate import check_log
from mag.gatetest import must_pass_cases, negative_cases, real_log_is_accepted

ROOT = Path(__file__).resolve().parents[1]


class GateTests(unittest.TestCase):
    def test_log_patterns(self):
        """Against the real llama-server strings, not invented ones."""
        from mag.gatetest import CPU_ONLY_LOG, PASS_LOG
        from mag.gate import HOST_BUFFER_MAX_MIB

        self.assertEqual(check_log(PASS_LOG, HOST_BUFFER_MAX_MIB), [])
        self.assertTrue(check_log(CPU_ONLY_LOG), "a CPU-only run must be rejected")

    def test_real_capture_is_accepted(self):
        name, failures = real_log_is_accepted()
        self.assertEqual(failures, [], f"{name} was wrongly rejected: {failures}")

    def test_python_negatives(self):
        """Every negative case, in Python. Runs on Windows without bash."""
        cases = negative_cases()
        self.assertGreaterEqual(len(cases), 8, cases)
        for name, rejected in cases:
            with self.subTest(case=name):
                self.assertTrue(rejected, f"{name} was not rejected")

    def test_clean_cases_are_accepted(self):
        """A gate that rejects a correct launch is as broken as one that passes a bad one."""
        for name, accepted in must_pass_cases():
            with self.subTest(case=name):
                self.assertTrue(accepted, f"{name} was wrongly rejected")

    @unittest.skipUnless(shutil.which("bash"), "POSIX shell not on PATH")
    def test_shell_negatives_exit_1(self):
        proc = subprocess.run(["bash", str(ROOT / "scripts/negative_gate.sh")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("negative tests exited 1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
