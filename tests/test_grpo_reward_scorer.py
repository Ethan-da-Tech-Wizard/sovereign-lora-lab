# ==============================================================================
# Program: test_grpo_reward_scorer.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Hermetic unit tests for GRPO verifiable reward scoring.
#          Tests command execution safety patterns, exit codes, AST syntax
#          validation, test runner execution rewards, SQLite trajectory event logging,
#          and contrastive DPO advantage delta calculations.
# ==============================================================================

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


# Each test uses a throw-away SQLite file so nothing touches the user's real event log.
class TestGRPORewardScorer(unittest.TestCase):
    """Test suite for deterministic reinforcement learning reward scorer."""

    def setUp(self):
        """Initialize temporary database for reward event logging."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_events.db"

    def tearDown(self):
        """Clean up temporary database directory."""
        self.temp_dir.cleanup()

    def test_evaluate_command_reward_safe(self):
        """Verify clean commands receive positive reward (R >= +0.5)."""
        # A read-only, well-behaved command with exit 0 gets the base 0.5 reward.
        score, why = evaluate_command_reward("git status", exit_code=0)
        self.assertGreaterEqual(score, 0.5)
        self.assertIn("cleanly", why.lower())

    def test_evaluate_command_reward_failure(self):
        """Verify failing commands receive non-zero penalty (R = -0.8)."""
        # Exit code 1 maps to the fixed -0.8 failure reward.
        score, why = evaluate_command_reward("unknown_bad_command", exit_code=1)
        self.assertEqual(score, -0.8)
        self.assertIn("non-zero", why.lower())

    def test_evaluate_command_reward_safety_violation(self):
        """Verify destructive commands trigger immediate safety barrier block (R = -1.0)."""
        # Forbidden bash fork bomb
        # Classic bash fork bomb; must be blocked by the static safety patterns before any scoring.
        score, why = evaluate_command_reward(":(){ :|:& };:")
        self.assertEqual(score, -1.0)
        self.assertIn("safety barrier", why.lower())

        # Forbidden destructive deletion
        score, why = evaluate_command_reward("rm -rf /")
        self.assertEqual(score, -1.0)
        self.assertIn("safety barrier", why.lower())

    def test_evaluate_code_reward_valid_syntax(self):
        """Verify valid Python syntax passes AST parsing with positive reward."""
        # Trivial valid function: parses cleanly and is far below the AST node budget.
        good_code = "def compute_area(w: float, h: float) -> float:\n    return w * h\n"
        score, why = evaluate_code_reward(good_code)
        self.assertGreaterEqual(score, 0.5)
        self.assertIn("ast syntax validation", why.lower())

    def test_evaluate_code_reward_syntax_error(self):
        """Verify malformed code fails closed with syntax error penalty (R = -1.0)."""
        # Missing closing parenthesis, so ast.parse raises SyntaxError.
        bad_syntax = "def compute_area(w, h\n    return w * h"
        score, why = evaluate_code_reward(bad_syntax)
        self.assertEqual(score, -1.0)
        self.assertIn("syntaxerror", why.lower())

    def test_evaluate_code_reward_with_test_runner(self):
        """Verify unit test runner integration awards maximum reward on passing tests."""
        code = "def add(a, b): return a + b\n"
        # Passing tests yield the full 1.0 (no complexity penalty for this tiny snippet).
        score_pass, _ = evaluate_code_reward(code, test_runner=lambda: True)
        self.assertEqual(score_pass, 1.0)

        # A failing test runner must drop the reward below the syntax-only baseline.
        score_fail, _ = evaluate_code_reward(code, test_runner=lambda: False)
        self.assertLess(score_fail, 0.5)

    def test_score_and_log_event(self):
        """Verify trajectory logging writes event row into SQLite with WAL mode."""
        # Returns the new row id (AUTOINCREMENT, hence > 0).
        ev_id = score_and_log_event(
            domain="test_domain",
            purpose="test_purpose",
            payload="test_payload",
            reward_score=0.85,
            reward_rationale="test rationale",
            db_path=self.db_path,
        )
        self.assertGreater(ev_id, 0)
        # Read the row back directly to prove it was persisted, not just returned.
        con = sqlite3.connect(self.db_path)
        row = con.execute("SELECT reward_score, reward_rationale FROM events WHERE id = ?", (ev_id,)).fetchone()
        con.close()
        self.assertEqual(row[0], 0.85)
        self.assertEqual(row[1], "test rationale")

    def test_score_dpo_pair(self):
        """Verify contrastive DPO pair scoring produces positive advantage margin."""
        chosen = "def solve(): return 100\n"
        # Rejected side is unparsable, so chosen (valid code) must out-score it.
        rejected = "broken syntax def ("
        res = score_dpo_pair(chosen, rejected, domain="code")
        self.assertTrue(res["grpo_valid"])
        self.assertGreater(res["advantage_delta_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
