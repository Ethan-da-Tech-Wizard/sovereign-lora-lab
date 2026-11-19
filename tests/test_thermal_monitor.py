"""Regression tests for core.thermal_monitor fail-closed behaviour."""

import tempfile
import unittest
from pathlib import Path

from core.thermal_monitor import ThermalSafetyGuard


class TestPreflightFailClosed(unittest.TestCase):
    """check_preflight is documented as fail-closed, so missing readings must block."""

    def setUp(self):
        """Point the STOP file at a temp path that does not exist (so it never blocks)."""
        self.tmp = tempfile.TemporaryDirectory()
        self.guard = ThermalSafetyGuard(stop_file=Path(self.tmp.name) / "STOP")

    def tearDown(self):
        self.tmp.cleanup()

    def test_healthy_readings_pass(self):
        """Known, cool readings and enough RAM pass with no problems."""
        ok, problems = self.guard.check_preflight(
            gpu_fn=lambda: 50.0, cpu_fn=lambda: 50.0, ram_fn=lambda: 32.0)
        self.assertTrue(ok, problems)

    def test_unknown_gpu_blocks(self):
        """A None GPU reading must block the run instead of being skipped."""
        ok, problems = self.guard.check_preflight(
            gpu_fn=lambda: None, cpu_fn=lambda: 50.0, ram_fn=lambda: 32.0)
        self.assertFalse(ok)
        self.assertTrue(any("GPU" in p and "unavailable" in p for p in problems))

    def test_unknown_cpu_and_nan_block(self):
        """None and NaN CPU readings both count as unknown."""
        for bad in (None, float("nan")):
            ok, problems = self.guard.check_preflight(
                gpu_fn=lambda: 50.0, cpu_fn=lambda bad=bad: bad, ram_fn=lambda: 32.0)
            self.assertFalse(ok)
            self.assertTrue(any("CPU" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
