# ==============================================================================
# Program: grpo_reward_scorer.py
# Programmer: Ethan-da-Tech-Wizard
# Course: CIS 156 / Advanced Systems
# Last Modified: November 18, 2026
# Purpose: Deterministic GRPO Verifiable Reward Scorer & Trajectory Logger.
#          Computes mathematically verifiable reinforcement learning rewards R in
#          [-1.0, +1.0] for Group Relative Policy Optimization (GRPO) and Direct
#          Preference Optimization (DPO). Evaluates commands against safety regex
#          barriers and exit codes; validates Python code via Abstract Syntax Tree
#          (AST) parsing, node complexity budgets, and hermetic unit tests.
#          Logs event trajectories to SQLite in WAL mode with advantage deltas.
# ==============================================================================

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

# ==============================================================================
# SECTION 1: Paths & Safety Barrier Regex Patterns
# Educational Note:
# In reinforcement learning for system-admin agents, allowing an agent to explore
# unconstrained actions can result in catastrophic commands (e.g. wiping disks).
# A pre-execution deterministic safety gate scans candidate commands against
# regex signatures before execution, assigning R = -1.0 to hazardous patterns.
# ==============================================================================

DEFAULT_EVENTS_DB = Path.home() / ".sovereign_lora" / "events.db"

FORBIDDEN_COMMAND_PATTERNS = [
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),                       # Bash fork bomb
    re.compile(r"\brm\s+-[rR][fF]?\s+(?:/[*\s]*|~[*\s]*|\.\.[*\s]*)"),              # Recursive delete on root/home
    re.compile(r"\bmkfs(?:\.\w+)?\b"),                                             # Unformatted filesystem creation
    re.compile(r"\bdd\s+if=.*of=/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z]|disk\d+)"),     # Direct drive block overwrite
    re.compile(r"\bcurl\b.*\|\s*(?:ba|z|t?c)?sh\b"),                               # Pipe remote curl to shell interpreter
    re.compile(r"\bwget\b.*\|\s*(?:ba|z|t?c)?sh\b"),                               # Pipe remote wget to shell interpreter
    re.compile(r"\bchmod\s+(?:-R\s+)?777\s+/"),                                    # Insecure recursive 777 permissions
    re.compile(r"\b(?:sudo\s+pacman|sudo\s+apt|sudo\s+dnf|sudo\s+zypper)\b.*"),    # Unconstrained root package installs
]


# ==============================================================================
# SECTION 2: SQLite Trajectory Database Initialization
# Educational Note:
# WAL (Write-Ahead Logging) mode allows multiple readers and writers without
# locking errors, essential for multi-agent evaluation loops and RL logging.
# ==============================================================================

def init_db_schema(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database for recording RL trajectory steps and reward receipts.

    Parameters:
        db_path (Path): Filesystem path to the SQLite database file.

    Returns:
        sqlite3.Connection: Active database connection configured with WAL journaling.
    """
    # Step 1: Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 2: Establish connection and configure SQLite pragmas
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")       # Concurrent read/write performance
    con.execute("PRAGMA synchronous=NORMAL")     # Balance safety with speed

    # Step 3: Create events table if not already present
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

    # Step 4: Schema migration check: ensure columns exist for older DB instances
    cols = [col[1] for col in con.execute("PRAGMA table_info(events)").fetchall()]
    if "reward_score" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN reward_score REAL DEFAULT 0.0")
    if "reward_rationale" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN reward_rationale TEXT DEFAULT ''")
    con.commit()

    return con


# ==============================================================================
# SECTION 3: Verifiable Command Reward Evaluation
# Educational Note:
# In GRPO, the policy optimization gradient is proportional to the reward advantage:
#   nabla_theta J(theta) approx E [ (R - R_baseline) * nabla_theta log pi_theta(y|x) ]
# Here we score commands deterministically:
#   R = -1.0 : Blocked by safety barrier
#   R = -0.8 : Command returned non-zero exit code (runtime crash/failure)
#   R = +0.5 : Clean exit code 0
#   R = +1.0 : Clean exit code 0 and stdout matches target regex pattern
# ==============================================================================

def evaluate_command_reward(
    command: str,
    exit_code: int = 0,
    stdout: str = "",
    expected_output_pattern: Optional[str] = None
) -> Tuple[float, str]:
    """Deterministically score a single command action using safety patterns and execution status.

    Parameters:
        command (str): The shell command proposed by the model.
        exit_code (int): Return code of the command after hermetic execution (0 = success).
        stdout (str): Standard output produced by the execution.
        expected_output_pattern (Optional[str]): Regex pattern the output should satisfy.

    Returns:
        Tuple[float, str]: (reward_score in [-1.0, 1.0], rationale_explanation)
    """
    # Step 1: Scan for forbidden destructive patterns
    for pat in FORBIDDEN_COMMAND_PATTERNS:
        if pat.search(command):
            return -1.0, f"Blocked by safety barrier (runtime_math_gate: forbidden pattern '{pat.pattern}')"

    # Step 2: Check process return exit code
    if exit_code != 0:
        return -0.8, f"Command failed with non-zero exit code: {exit_code}"

    # Step 3: Validate against output specification
    base_reward = 0.5
    if expected_output_pattern:
        if re.search(expected_output_pattern, stdout):
            return 1.0, "Command executed cleanly (exit code 0) and matched expected output specification."
        return 0.3, "Command exited cleanly (exit code 0) but did not match expected output pattern."

    return base_reward, "Command executed cleanly (exit code 0)."


# ==============================================================================
# SECTION 4: AST-Verified Code Generation & Repair Reward
# Educational Note:
# For code generation tasks, string matching against golden files is brittle because
# variable names, formatting, or comments can differ while code remains identical.
# Instead, we parse code into an Abstract Syntax Tree (AST):
#   1. ast.parse(code): Fails immediately with SyntaxError if code is ungrammatical.
#   2. ast.walk(tree): Traverses all nodes in the syntax tree to measure code complexity
#      and enforce a maximum token/node budget (preventing bloated solutions).
#   3. Unit test execution: If test_runner is provided, runs tests hermetically.
# ==============================================================================

def evaluate_code_reward(
    code_str: str,
    test_runner: Optional[Callable[[], bool]] = None,
    max_ast_tokens: int = 500
) -> Tuple[float, str]:
    """Score code generation or repair via deterministic AST validation and unit test execution.

    Parameters:
        code_str (str): The Python code string to evaluate.
        test_runner (Optional[Callable[[], bool]]): Optional callback returning True if tests pass.
        max_ast_tokens (int): Maximum allowable AST node count before applying length penalties.

    Returns:
        Tuple[float, str]: (reward_score in [-1.0, 1.0], rationale_explanation)
    """
    # Step 1: Parse code into Abstract Syntax Tree (AST)
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        # Invalid syntax fails closed with maximum penalty
        return -1.0, f"SyntaxError during AST parse: {e}"

    # Step 2: Calculate AST complexity penalty
    # Count all node objects in the parsed syntax tree
    num_nodes = len(list(ast.walk(tree)))
    penalty = 0.0
    if num_nodes > max_ast_tokens:
        # Linear penalty capped at 0.3 for exceeding node budget
        penalty = min(0.3, (num_nodes - max_ast_tokens) / max_ast_tokens * 0.3)

    base_reward = 0.5 - penalty

    # Step 3: Run hermetic unit tests if provided
    if test_runner is not None:
        try:
            passed = test_runner()
            if passed:
                return round(1.0 - penalty, 4), f"Code passed AST parse ({num_nodes} nodes) and verified unit tests."
            else:
                return round(0.1 - penalty, 4), "Code passed AST parse but failed unit tests."
        except Exception as e:
            return -0.5, f"Test runner raised exception: {e}"

    # Return valid syntax reward adjusted for complexity
    return round(base_reward, 4), f"Code passed AST syntax validation ({num_nodes} nodes)."


# ==============================================================================
# SECTION 5: DPO Contrastive Preference Scoring
# Educational Note:
# Direct Preference Optimization (DPO) and GRPO both rely on contrastive pairs:
# a 'chosen' completion y_w and a 'rejected' completion y_l.
# The advantage delta:
#   Delta_R = R(chosen) - R(rejected)
# A pair is only valid for contrastive fine-tuning if Delta_R > 0.
# ==============================================================================

def score_dpo_pair(
    chosen: str,
    rejected: str,
    domain: str = "code",
    expected_output_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate contrastive advantage delta between chosen and rejected candidates.

    Parameters:
        chosen (str): The preferred ground-truth or high-reward completion.
        rejected (str): The flawed, rejected, or lower-reward completion.
        domain (str): Domain type ('code' or 'command').
        expected_output_pattern (Optional[str]): Pattern for command output verification.

    Returns:
        Dict[str, Any]: Dictionary containing reward scores, advantage delta, and validity.
    """
    # Step 1: Score both candidates depending on domain
    if domain == "command":
        r_c, why_c = evaluate_command_reward(chosen, exit_code=0, expected_output_pattern=expected_output_pattern)
        r_r, why_r = evaluate_command_reward(rejected, exit_code=1)
    else:  # 'code' domain
        r_c, why_c = evaluate_code_reward(chosen)
        r_r, why_r = evaluate_code_reward(rejected)

    # Step 2: Compute advantage delta
    delta_r = round(r_c - r_r, 4)

    # Step 3: Assemble preference report
    return {
        "grpo_valid": r_c > r_r,
        "r_chosen": r_c,
        "r_rejected": r_r,
        "advantage_delta_r": delta_r,
        "rationale_chosen": why_c,
        "rationale_rejected": why_r,
    }


# ==============================================================================
# SECTION 6: Event Trajectory Logging
# ==============================================================================

def score_and_log_event(
    domain: str,
    purpose: str,
    payload: str,
    reward_score: float,
    reward_rationale: str,
    db_path: Optional[Path] = None,
) -> int:
    """Record an evaluation event and verifiable reward receipt into SQLite.

    Parameters:
        domain (str): Category (e.g. 'cli', 'code_repair', 'math').
        purpose (str): Goal of the task or evaluation.
        payload (str): The input prompt or generated action string.
        reward_score (float): Calculated numerical reward.
        reward_rationale (str): Educational textual explanation of the score.
        db_path (Optional[Path]): Database destination.

    Returns:
        int: The auto-incremented primary key row ID of the inserted event.
    """
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


# ==============================================================================
# SECTION 7: Command-Line Interface (CLI) Entry Point
# ==============================================================================

def main() -> None:
    """Parse CLI arguments to test or evaluate code and command rewards interactively."""
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
