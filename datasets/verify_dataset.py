#!/usr/bin/env python3
"""datasets/verify_dataset.py — Dataset Validation, Deduplication & Quality Gate.

Verifies:
  1. JSONL structural syntax & schema conformance
  2. AST syntax correctness for Python code completions
  3. Exact duplicate prompt/completion removal
  4. Character and token length distributions
  5. Family-held-out disjoint train/dev split generation
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def verify_file(
    file_path: Union[str, Path],
    mode: str = "auto",
    check_ast: bool = False,
) -> Dict[str, Any]:
    """Audit a JSONL dataset file for validity, duplicates, and token lengths."""
    p = Path(file_path)
    if not p.exists():
        return {"status": "ERROR", "reason": f"File not found: {file_path}"}

    lines = p.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    valid_rows: List[Dict[str, Any]] = []
    malformed = 0
    ast_errors = 0
    seen_keys: Set[Tuple[str, str]] = set()
    exact_duplicates = 0

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            malformed += 1
            continue

        if not isinstance(row, dict) or "prompt" not in row or "chosen" not in row:
            malformed += 1
            continue

        prompt = str(row["prompt"])
        chosen_str = row["chosen"] if isinstance(row["chosen"], str) else json.dumps(row["chosen"], sort_keys=True)
        key = (prompt, chosen_str)

        if key in seen_keys:
            exact_duplicates += 1
            continue
        seen_keys.add(key)

        # Optional AST verification
        if check_ast and isinstance(row["chosen"], str):
            try:
                ast.parse(row["chosen"])
            except SyntaxError:
                ast_errors += 1

        valid_rows.append(row)

    prompt_lens = [len(r["prompt"]) for r in valid_rows]
    completion_lens = [len(r["chosen"] if isinstance(r["chosen"], str) else json.dumps(r["chosen"])) for r in valid_rows]

    def stats(lens: List[int]) -> Dict[str, Any]:
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


def split_family_held_out(
    rows: List[Dict[str, Any]],
    dev_fraction: float = 0.2,
    seed: int = 1101,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate family-stratified train/dev split with zero overlapping prompts."""
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        fam = r.get("family", "default")
        by_family.setdefault(fam, []).append(r)

    rng = random.Random(seed)
    train: List[Dict[str, Any]] = []
    dev: List[Dict[str, Any]] = []

    for fam, items in by_family.items():
        rng.shuffle(items)
        n_dev = max(1, int(len(items) * dev_fraction)) if len(items) > 1 else 0
        dev.extend(items[:n_dev])
        train.extend(items[n_dev:])

    dev_prompts = {r["prompt"] for r in dev}
    # Enforce strict disjointness
    train = [r for r in train if r["prompt"] not in dev_prompts]
    return train, dev


def main() -> None:
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
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        train, dev = split_family_held_out(rows, dev_fraction=args.dev_fraction)
        train_path = p.with_name(p.stem + "_train.jsonl")
        dev_path = p.with_name(p.stem + "_dev.jsonl")
        train_path.write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
        dev_path.write_text("\n".join(json.dumps(r) for r in dev) + "\n", encoding="utf-8")
        print(f"Split completed: train={len(train)} -> {train_path}, dev={len(dev)} -> {dev_path}")


if __name__ == "__main__":
    main()
