#!/usr/bin/env python3
"""core/grpo_reward_scorer.py — Deterministic GRPO Verifiable Reward Scorer.

Computes mathematically verifiable reinforcement learning rewards R in [-1.0, +1.0]:
  - R = -1.0 : Blocked by safety barrier (dangerous commands, AST syntax errors)
  - R = -0.8 : Runtime command failure (non-zero exit code)
  - R = +0.5 : Clean execution (valid syntax, exit code 0)
  - R = +1.0 : Full verification (passes hermetic tests & AST token budget)

Provides SQLite event trajectory logging and DPO advantage delta calculation:
  Delta R = R_chosen - R_rejected
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Configurable paths
DEFAULT_EVENTS_DB = Path.home() / ".sovereign_lora" / "events.db"

# Safety regex patterns for command gating
FORBIDDEN_COMMAND_PATTERNS = [
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),  # Fork bomb
    re.compile(r"\brm\s+-[rR][fF]?\s+(?:/[*\s]*|~[*\s]*|\.\.[*\s]*)"),  # Root or home recursive delete
    re.compile(r"\bmkfs(?:\.\w+)?\b"),  # Filesystem creation
    re.compile(r"\bdd\s+if=.*of=/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z]|disk\d+)"),  # Raw device overwrite
    re.compile(r"\bcurl\b.*\|\s*(?:ba|z|t?c)?sh\b"),  # Pipe untrusted curl to shell
    re.compile(r"\bwget\b.*\|\s*(?:ba|z|t?c)?sh\b"),  # Pipe untrusted wget to shell
    re.compile(r"\bchmod\s+(?:-R\s+)?777\s+/"),  # Dangerous global permissions on root
    re.compile(r"\b(?:sudo\s+pacman|sudo\s+apt|sudo\s+dnf|sudo\s+zypper)\b.*"),  # Unsafe root package manager invocations
]


def init_db_schema(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite trajectory and reward event logging database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            host TEXT,
            domain TEXT,
            purpose TEXT,
            payload TEXT,
            reward_score REAL DEFAULT 0.0,
            reward_rationale TEXT DEFAULT ''
        )
        """
    )
    # Ensure column presence if existing table is older
    cols = [col[1] for col in con.execute("PRAGMA table_info(events)").fetchall()]
    if "reward_score" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN reward_score REAL DEFAULT 0.0")
    if "reward_rationale" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN reward_rationale TEXT DEFAULT ''")
    con.commit()
    return con


def evaluate_command_reward(
    command: str,
    exit_code: int = 0,
    stdout: str = "",
    expected_output_pattern: Optional[str] = None
) -> Tuple[float, str]:
    """Deterministically score a single command action using safety patterns and execution status."""
    # 1. Check safety barriers
    for pat in FORBIDDEN_COMMAND_PATTERNS:
        if pat.search(command):
            return -1.0, f"Blocked by safety barrier (runtime_math_gate: forbidden pattern '{pat.pattern}')"

    # 2. Check exit code
    if exit_code != 0:
        return -0.8, f"Command failed with non-zero exit code: {exit_code}"

    # 3. Clean execution verification
    base_reward = 0.5
    if expected_output_pattern:
        if re.search(expected_output_pattern, stdout):
            return 1.0, "Command executed cleanly (exit code 0) and matched expected output specification."
        return 0.3, "Command exited cleanly (exit code 0) but did not match expected output pattern."

    return base_reward, "Command executed cleanly (exit code 0)."


def evaluate_code_reward(
    code_str: str,
    test_runner: Optional[Callable[[], bool]] = None,
    max_ast_tokens: int = 500
) -> Tuple[float, str]:
    """Score code generation or repair via deterministic AST validation and test execution."""
    # 1. Parse AST
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return -1.0, f"SyntaxError during AST parse: {e}"

    # 2. Token / AST node budget penalty
    num_nodes = len(list(ast.walk(tree)))
    penalty = 0.0
    if num_nodes > max_ast_tokens:
        penalty = min(0.3, (num_nodes - max_ast_tokens) / max_ast_tokens * 0.3)

    base_reward = 0.5 - penalty

    # 3. Hermetic test runner execution if provided
    if test_runner is not None:
        try:
            passed = test_runner()
            if passed:
                return round(1.0 - penalty, 4), f"Code passed AST parse ({num_nodes} nodes) and verified unit tests."
            else:
                return round(0.1 - penalty, 4), "Code passed AST parse but failed unit tests."
        except Exception as e:
            return -0.5, f"Test runner raised exception: {e}"

    return round(base_reward, 4), f"Code passed AST syntax validation ({num_nodes} nodes)."


def score_dpo_pair(
    chosen: str,
    rejected: str,
    domain: str = "code",
    expected_output_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate contrastive advantage delta between chosen and rejected candidates."""
    if domain == "command":
        r_c, why_c = evaluate_command_reward(chosen, exit_code=0, expected_output_pattern=expected_output_pattern)
        r_r, why_r = evaluate_command_reward(rejected, exit_code=1)
    else:  # code
        r_c, why_c = evaluate_code_reward(chosen)
        r_r, why_r = evaluate_code_reward(rejected)

    delta_r = round(r_c - r_r, 4)
    return {
        "grpo_valid": r_c > r_r,
        "r_chosen": r_c,
        "r_rejected": r_r,
        "advantage_delta_r": delta_r,
        "rationale_chosen": why_c,
        "rationale_rejected": why_r,
    }


def score_and_log_event(
    domain: str,
    purpose: str,
    payload: str,
    reward_score: float,
    reward_rationale: str,
    db_path: Optional[Path] = None,
) -> int:
    """Record an evaluation event and reward receipt into SQLite."""
    target_db = db_path or DEFAULT_EVENTS_DB
    con = init_db_schema(target_db)
    now = datetime.now(timezone.utc).isoformat()
    host = os.uname().nodename.split(".")[0]

    cur = con.execute(
        """
        INSERT INTO events (ts, host, domain, purpose, payload, reward_score, reward_rationale)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (now, host, domain, purpose, payload, float(reward_score), str(reward_rationale)),
    )
    con.commit()
    ev_id = cur.lastrowid
    con.close()
    return ev_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifiable GRPO Reward Scorer")
    parser.add_argument("--domain", choices=["code", "command"], default="code")
    parser.add_argument("--eval-code", type=str, help="Python code string to evaluate")
    parser.add_argument("--eval-command", type=str, help="Shell command string to evaluate")
    parser.add_argument("--exit-code", type=int, default=0, help="Command return code")
    args = parser.parse_args()

    if args.domain == "command" and args.eval_command:
        score, why = evaluate_command_reward(args.eval_command, exit_code=args.exit_code)
        print(json.dumps({"reward_score": score, "rationale": why}, indent=2))
    elif args.eval_code:
        score, why = evaluate_code_reward(args.eval_code)
        print(json.dumps({"reward_score": score, "rationale": why}, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
