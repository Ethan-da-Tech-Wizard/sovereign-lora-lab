# ==============================================================================
# Program: test_autonomous_dsce_trainer.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Unit tests for DSCE Oracle verification and DPO harvesting.
#          Verifies query_dsce_oracle lookup logic, IntentCommandGuardian safety
#          filtering, task execution and contrastive pair formation, and
#          automated benchmark reporting.
# ==============================================================================

import json
import tempfile
import unittest
from pathlib import Path

from core.autonomous_dsce_trainer import (
    AutonomousDSCETrainer,
    IntentCommandGuardian,
    STANDARD_BENCHMARK_TASKS,
    query_dsce_oracle,
)


class TestAutonomousDSCETrainer(unittest.TestCase):
    """Test suite for DSCE knowledge vials, command guardian, and DPO generation."""

    def setUp(self):
        """Create temporary directory and mock trainer instance."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dpo_out = Path(self.temp_dir.name) / "test_dpo.jsonl"
        # Output path points into a temp dir; no model endpoint is contacted because tests pass candidate_response.
        self.trainer = AutonomousDSCETrainer(dpo_output_path=self.dpo_out)
        # Hermetic: never talk to a real llama.cpp server that may be listening on the default port
        # (a live server made the offline-skip tests flaky). Tests that need a model response pass
        # candidate_response explicitly.
        self.trainer.query_local_model = lambda prompt, timeout=3.0: None

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_query_dsce_oracle_valid(self):
        """Verify successful query retrieval from DSCE knowledge vials."""
        res = query_dsce_oracle("compositor", "kill_active_window")
        self.assertTrue(res["success"])
        self.assertEqual(res["answer"], "hyprctl dispatch killactive")
        self.assertIn("Proof", res["proof_tree"])

    def test_query_dsce_oracle_unknown(self):
        """Verify graceful failure on queries not present in knowledge vials."""
        # Unknown keys must fail soft (success=False, answer None), never guess.
        res = query_dsce_oracle("unknown_sub", "unknown_pred")
        self.assertFalse(res["success"])
        self.assertIsNone(res["answer"])

    def test_intent_command_guardian(self):
        """Verify guardian passes safe commands and flags destructive ones."""
        guardian = IntentCommandGuardian()
        # Safe command check
        # A benign compositor command is expected to pass the guardian.
        check_safe = guardian.verify("hyprctl dispatch killactive")
        self.assertEqual(check_safe["verdict"], "PASS")

        # Dangerous command check
        check_unsafe = guardian.verify("rm -rf /")
        self.assertEqual(check_unsafe["verdict"], "FLAGGED")

    def test_execute_and_verify_task_passing(self):
        """Verify task execution records passed status and writes DPO pair."""
        task = STANDARD_BENCHMARK_TASKS[0]  # hyprctl dispatch killactive
        # Passing an explicit candidate skips the network call entirely.
        res = self.trainer.execute_and_verify_task(task, candidate_response="hyprctl dispatch killactive")
        self.assertEqual(res["status"], "PASSED_DSCE_GATE")
        self.assertEqual(res["proven_truth"], "hyprctl dispatch killactive")

        # Verify DPO output file has the written pair
        lines = self.dpo_out.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertTrue(record["passed"])
        self.assertEqual(record["task_id"], task["id"])

    def test_execute_and_verify_task_rejection(self):
        """Verify task execution records rejected status when model guesses wrong."""
        task = STANDARD_BENCHMARK_TASKS[0]
        # A wrong (and unsafe-looking) answer must be rejected and still harvested as a DPO "rejected" sample.
        res = self.trainer.execute_and_verify_task(task, candidate_response="kill -9 9999")
        self.assertEqual(res["status"], "REJECTED_BY_DSCE_GATE")

        lines = self.dpo_out.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertFalse(record["passed"])

    def test_run_benchmark(self):
        """Verify end-to-end benchmark execution across task slices (offline = honest skip)."""
        # No model server is reachable in tests, so every task must be SKIPPED, not silently passed.
        report = self.trainer.run_benchmark(STANDARD_BENCHMARK_TASKS[:3])
        self.assertEqual(report["total_tasks"], 3)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["skipped"], 3)
        self.assertEqual(report["pass_rate"], 0.0)
        # Skipped tasks must not write fabricated DPO pairs.
        self.assertFalse(self.dpo_out.exists())

    def test_offline_model_never_echoes_expected_answer(self):
        """Regression: with no model response the task is skipped, never auto-passed."""
        task = STANDARD_BENCHMARK_TASKS[0]
        self.trainer.query_local_model = lambda prompt, timeout=10.0: None
        res = self.trainer.execute_and_verify_task(task)
        self.assertEqual(res["status"], "SKIPPED_MODEL_UNAVAILABLE")
        self.assertIsNone(res["model_response"])


if __name__ == "__main__":
    unittest.main()
