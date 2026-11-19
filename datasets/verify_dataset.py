# ==============================================================================
# Program: verify_dataset.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Dataset Validation, Deduplication & Quality Gate Utility.
#          Audits JSONL fine-tuning and alignment datasets:
#            1. Structural JSON syntax & schema conformance validation
#            2. Abstract Syntax Tree (AST) syntax validation for Python completions
#            3. Exact duplicate prompt/completion removal
#            4. Statistical character and token length distribution analysis
#            5. Family-held-out disjoint train/dev split partitioning
# ==============================================================================

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ==============================================================================
# SECTION 1: Dataset Verification & Audit Function
# Educational Note:
# Feeding corrupted JSON or syntactically invalid Python code into an LLM fine-tuning
# run wastes expensive GPU hours. This function acts as a pre-training quality gate,
# catching bad lines, invalid ASTs, and duplicate prompts before training begins.
# ==============================================================================

def verify_file(
    file_path: Union[str, Path],
    mode: str = "auto",
    check_ast: bool = False,
) -> Dict[str, Any]:
    """Audit a JSONL dataset file for validity, duplicates, and token lengths.

    Step-by-Step Educational Pipeline:
        1. Open the file and iterate line-by-line.
        2. Attempt json.loads() on each line; count malformed lines.
        3. Verify required keys ('prompt' and 'chosen') exist.
        4. Check for duplicate prompt/completion pairs using a hash set.
        5. (Optional) Parse code completions into Python AST to verify syntax.
        6. Calculate character length summary statistics (min, max, mean, p50).

    Parameters:
        file_path (Union[str, Path]): Filesystem path to the JSONL dataset.
        mode (str): Reserved. Accepted for API/CLI compatibility but currently IGNORED: the same
            checks run for every value ('auto', 'sft', 'dpo' do not change behaviour).
        check_ast (bool): If True, parses string completions with ast.parse().

    Returns:
        Dict[str, Any]: Comprehensive audit report dictionary.
    """
    # `mode` is intentionally unused for now (documented above); do not rely on it to select checks.
    p = Path(file_path)
    if not p.exists():
        return {"status": "ERROR", "reason": f"File not found: {file_path}"}

    lines = p.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    valid_rows: List[Dict[str, Any]] = []
    malformed = 0
    ast_errors = 0
    # Duplicate key = (prompt, canonical completion text).
    seen_keys: Set[Tuple[str, str]] = set()
    exact_duplicates = 0

    # Step 1: Scan line-by-line
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            malformed += 1
            continue

        # Step 2: Validate presence of essential fields
        if not isinstance(row, dict) or "prompt" not in row or "chosen" not in row:
            malformed += 1
            continue

        prompt = str(row["prompt"])
        # Structured completions (dict/list) are JSON-serialized with sorted keys so equal objects compare equal.
        chosen_str = row["chosen"] if isinstance(row["chosen"], str) else json.dumps(row["chosen"], sort_keys=True)
        key = (prompt, chosen_str)

        # Step 3: Deduplicate exact identical pairs
        if key in seen_keys:
            exact_duplicates += 1
            continue
        seen_keys.add(key)

        # Step 4: Optional AST verification for Python code completions
        if check_ast and isinstance(row["chosen"], str):
            try:
                ast.parse(row["chosen"])
            except SyntaxError:
                ast_errors += 1

        valid_rows.append(row)

    # Step 5: Compute length distributions
    prompt_lens = [len(r["prompt"]) for r in valid_rows]
    completion_lens = [len(r["chosen"] if isinstance(r["chosen"], str) else json.dumps(r["chosen"])) for r in valid_rows]

    def stats(lens: List[int]) -> Dict[str, Any]:
        """Compute min, max, mean, and median (p50) from integer list."""
        if not lens:
            return {"min": 0, "max": 0, "mean": 0, "p50": 0}
        sl = sorted(lens)
        return {
            "min": sl[0],
            "max": sl[-1],
            "mean": round(sum(sl) / len(sl), 1),
            "p50": sl[len(sl) // 2],
        }

    return {
        "status": "PASSED" if malformed == 0 else "WARNED",
        "file": str(p),
        "total_lines": total,
        "valid_records": len(valid_rows),
        "malformed_lines": malformed,
        "exact_duplicates_removed": exact_duplicates,
        "ast_syntax_errors": ast_errors,
        "prompt_char_stats": stats(prompt_lens),
        "completion_char_stats": stats(completion_lens),
    }


# ==============================================================================
# SECTION 2: Family-Held-Out Disjoint Train / Dev Splitter
# Educational Note:
# Datasets often have clusters of examples belonging to the same problem family
# (e.g. variations of binary search). If members of the same family are mixed across
# train and dev, data leakage occurs. Family-held-out splitting isolates entire
# families into either train or dev, preventing prompt leakage.
# ==============================================================================

def split_family_held_out(
    rows: List[Dict[str, Any]],
    dev_fraction: float = 0.2,
    seed: int = 1101,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows so that every family lands entirely in train or entirely in dev.

    Whole families are moved to dev (in a seeded random order) until dev holds at least
    ``dev_fraction`` of the rows; at least one family always stays in train. Because the
    unit of assignment is the family, the dev fraction is approximate (it overshoots by at
    most one family). A final pass drops train rows whose prompt text also appears in dev
    (duplicate prompts filed under different families), so prompts are disjoint too; in
    that rare case train + dev can be smaller than the input.

    Parameters:
        rows (List[Dict[str, Any]]): Valid input dataset rows. Rows without a "family"
            key are treated as one shared family called "default".
        dev_fraction (float): Target fraction of rows for the dev split (0.2 = 20%).
        seed (int): Seed for deterministic random number generation.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (train_rows, dev_rows)

    Raises:
        ValueError: If the rows contain fewer than two families, because a family-held-out
            split is impossible (it would leave train or dev empty).
    """
    # Group all rows by their family tag
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        fam = r.get("family", "default")
        by_family.setdefault(fam, []).append(r)

    if len(by_family) < 2:
        raise ValueError("family-held-out split needs at least 2 distinct families")

    # Sorted family names first so the shuffle depends only on the seed, not on row order.
    rng = random.Random(seed)
    families = sorted(by_family)
    rng.shuffle(families)

    train: List[Dict[str, Any]] = []
    dev: List[Dict[str, Any]] = []
    target_dev = dev_fraction * len(rows)

    # Move whole families to dev until the target is reached, always leaving the last family
    # for train so neither side can end up empty.
    for i, fam in enumerate(families):
        remaining_after = len(families) - i - 1
        if len(dev) < target_dev and (remaining_after > 0 or train):
            dev.extend(by_family[fam])
        else:
            train.extend(by_family[fam])

    # Strictly enforce that no prompt in dev appears in train
    dev_prompts = {r["prompt"] for r in dev}
    # Drop train rows whose prompt text also appears in dev (e.g. duplicate prompts across families).
    train = [r for r in train if r["prompt"] not in dev_prompts]
    return train, dev


# ==============================================================================
# SECTION 3: CLI Entry Point
# ==============================================================================

def main() -> None:
    """Parse CLI arguments to verify dataset or generate train/dev split files."""
    parser = argparse.ArgumentParser(description="Dataset Quality Gate & Verifier")
    parser.add_argument("file", help="Path to JSONL dataset")
    parser.add_argument("--check-ast", action="store_true", help="Verify AST syntax on completions")
    parser.add_argument("--split", action="store_true", help="Generate train and dev split files")
    parser.add_argument("--dev-fraction", type=float, default=0.2, help="Fraction for dev split")
    args = parser.parse_args()

    report = verify_file(args.file, check_ast=args.check_ast)
    print(json.dumps(report, indent=2))

    if args.split:
        p = Path(args.file)
        # Re-read the raw file (including duplicates) for splitting; verify_file only reports statistics.
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        try:
            train, dev = split_family_held_out(rows, dev_fraction=args.dev_fraction)
        except ValueError as exc:
            # Too few families to hold any out: report it instead of writing a leaky split.
            raise SystemExit(f"Cannot split: {exc}")
        # Outputs sit beside the input as <name>_train.jsonl / <name>_dev.jsonl.
        train_path = p.with_name(p.stem + "_train.jsonl")
        dev_path = p.with_name(p.stem + "_dev.jsonl")
        train_path.write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
        dev_path.write_text("\n".join(json.dumps(r) for r in dev) + "\n", encoding="utf-8")
        print(f"Split completed: train={len(train)} -> {train_path}, dev={len(dev)} -> {dev_path}")


if __name__ == "__main__":
    main()
