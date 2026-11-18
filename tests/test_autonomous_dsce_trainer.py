# Program: sovereign-lora-lab - test_autonomous_dsce_trainer.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Unit tests for the DSCE oracle lookup, command safety guard, task execution, DPO pair creation and benchmark reporting.

"""Unit tests for the DSCE oracle lookup, command safety guard, task execution, DPO pair creation and benchmark reporting."""

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
        self.trainer = AutonomousDSCETrainer(dpo_output_path=self.dpo_out)

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
        res = query_dsce_oracle("unknown_sub", "unknown_pred")
        self.assertFalse(res["success"])
        self.assertIsNone(res["answer"])

    def test_intent_command_guardian(self):
        """Verify guardian passes safe commands and flags destructive ones."""
        guardian = IntentCommandGuardian()
        # Safe command check
        check_safe = guardian.verify("hyprctl dispatch killactive")
        self.assertEqual(check_safe["verdict"], "PASS")

        # Dangerous command check
        check_unsafe = guardian.verify("rm -rf /")
        self.assertEqual(check_unsafe["verdict"], "FLAGGED")

    def test_execute_and_verify_task_passing(self):
        """Verify task execution records passed status and writes DPO pair."""
        task = STANDARD_BENCHMARK_TASKS[0]  # hyprctl dispatch killactive
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
        res = self.trainer.execute_and_verify_task(task, candidate_response="kill -9 9999")
        self.assertEqual(res["status"], "REJECTED_BY_DSCE_GATE")

        lines = self.dpo_out.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertFalse(record["passed"])

    def test_run_benchmark(self):
        """Verify end-to-end benchmark execution across task slices."""
        report = self.trainer.run_benchmark(STANDARD_BENCHMARK_TASKS[:3])
        self.assertEqual(report["total_tasks"], 3)
        self.assertIn("pass_rate", report)
        self.assertTrue(self.dpo_out.exists())


if __name__ == "__main__":
    unittest.main()
