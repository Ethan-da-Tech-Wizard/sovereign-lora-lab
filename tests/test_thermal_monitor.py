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

    # Removes the temp directory the monitor's log/state files were written into.
    def tearDown(self):
        self.tmp.cleanup()

    def test_healthy_readings_pass(self):
        """Known, cool readings and enough RAM pass with no problems."""
        # Lambdas stand in for the real sensor functions (injectable parameters), so the test needs no GPU.
        ok, problems = self.guard.check_preflight(
            gpu_fn=lambda: 50.0, cpu_fn=lambda: 50.0, ram_fn=lambda: 32.0)
        self.assertTrue(ok, problems)

    def test_unknown_gpu_blocks(self):
        """A None GPU reading must block the run instead of being skipped."""
        # gpu_fn returns None: the sensor is unreadable. Before the fix this was silently skipped and the run was allowed.
        ok, problems = self.guard.check_preflight(
            gpu_fn=lambda: None, cpu_fn=lambda: 50.0, ram_fn=lambda: 32.0)
        self.assertFalse(ok)
        self.assertTrue(any("GPU" in p and "unavailable" in p for p in problems))

    def test_unknown_cpu_and_nan_block(self):
        """None and NaN CPU readings both count as unknown."""
        # NaN is what a broken sensor parse can produce; known() must treat it exactly like None.
        for bad in (None, float("nan")):
            ok, problems = self.guard.check_preflight(
                gpu_fn=lambda: 50.0, cpu_fn=lambda bad=bad: bad, ram_fn=lambda: 32.0)
            self.assertFalse(ok)
            self.assertTrue(any("CPU" in p for p in problems))


class TestGpuTempHonesty(unittest.TestCase):
    """gpu_temp must not fabricate a temperature when no real sensor is available."""

    def test_no_sensor_returns_none_not_synthetic_value(self):
        """With no nvidia-smi and no env override, the answer is None on every platform."""
        import os
        from unittest import mock
        from core import thermal_monitor as tm

        # Strip any real override from the environment so the result depends only on the (mocked) absence of sensors.
        env = {k: v for k, v in os.environ.items() if k != "SOVEREIGN_MOCK_GPU_TEMP"}
        with mock.patch.object(tm.shutil, "which", return_value=None), mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(tm.gpu_temp())

    def test_env_override_still_works(self):
        """The documented test hook keeps working for hardware-less hosts."""
        import os
        from unittest import mock
        from core import thermal_monitor as tm

        with mock.patch.object(tm.shutil, "which", return_value=None), \
                mock.patch.dict(os.environ, {"SOVEREIGN_MOCK_GPU_TEMP": "61.5"}):
            self.assertEqual(tm.gpu_temp(), 61.5)


if __name__ == "__main__":
    unittest.main()
