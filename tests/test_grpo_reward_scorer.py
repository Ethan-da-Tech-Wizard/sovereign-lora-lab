#!/usr/bin/env python3
"""tests/test_grpo_reward_scorer.py — Hermetic unit tests for GRPO reward scoring."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.grpo_reward_scorer import (
    evaluate_command_reward,
    evaluate_code_reward,
    score_and_log_event,
    score_dpo_pair,
)


class TestGRPORewardScorer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_events.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluate_command_reward_safe(self):
        score, why = evaluate_command_reward("git status", exit_code=0)
        self.assertGreaterEqual(score, 0.5)
        self.assertIn("cleanly", why.lower())

    def test_evaluate_command_reward_failure(self):
        score, why = evaluate_command_reward("unknown_bad_command", exit_code=1)
        self.assertEqual(score, -0.8)
        self.assertIn("non-zero", why.lower())

    def test_evaluate_command_reward_safety_violation(self):
        # Forbidden fork bomb
        score, why = evaluate_command_reward(":(){ :|:& };:")
        self.assertEqual(score, -1.0)
        self.assertIn("safety barrier", why.lower())

        # Forbidden destructive deletion
        score, why = evaluate_command_reward("rm -rf /")
        self.assertEqual(score, -1.0)
        self.assertIn("safety barrier", why.lower())

    def test_evaluate_code_reward_valid_syntax(self):
        good_code = "def compute_area(w: float, h: float) -> float:\n    return w * h\n"
        score, why = evaluate_code_reward(good_code)
        self.assertGreaterEqual(score, 0.5)
        self.assertIn("ast syntax validation", why.lower())

    def test_evaluate_code_reward_syntax_error(self):
        bad_syntax = "def compute_area(w, h\n    return w * h"
        score, why = evaluate_code_reward(bad_syntax)
        self.assertEqual(score, -1.0)
        self.assertIn("syntaxerror", why.lower())

    def test_evaluate_code_reward_with_test_runner(self):
        code = "def add(a, b): return a + b\n"
        score_pass, _ = evaluate_code_reward(code, test_runner=lambda: True)
        self.assertEqual(score_pass, 1.0)

        score_fail, _ = evaluate_code_reward(code, test_runner=lambda: False)
        self.assertLess(score_fail, 0.5)

    def test_score_and_log_event(self):
        ev_id = score_and_log_event(
            domain="test_domain",
            purpose="test_purpose",
            payload="test_payload",
            reward_score=0.85,
            reward_rationale="test rationale",
            db_path=self.db_path,
        )
        self.assertGreater(ev_id, 0)
        con = sqlite3.connect(self.db_path)
        row = con.execute("SELECT reward_score, reward_rationale FROM events WHERE id = ?", (ev_id,)).fetchone()
        con.close()
        self.assertEqual(row[0], 0.85)
        self.assertEqual(row[1], "test rationale")

    def test_score_dpo_pair(self):
        chosen = "def solve(): return 100\n"
        rejected = "broken syntax def ("
        res = score_dpo_pair(chosen, rejected, domain="code")
        self.assertTrue(res["grpo_valid"])
        self.assertGreater(res["advantage_delta_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
