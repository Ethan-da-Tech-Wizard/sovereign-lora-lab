# ==============================================================================
# Program: test_dataset_verifier.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Unit tests for dataset validation, schema checking, and train/dev splitting.
#          Tests JSON structure verification, duplicate elimination, AST syntax
#          error detection on Python code completions, and family-held-out disjointness.
# ==============================================================================

import json
import tempfile
import unittest
from pathlib import Path

from datasets.verify_dataset import split_family_held_out, verify_file


class TestDatasetVerifier(unittest.TestCase):
    """Test suite for dataset audit gate and train/dev partitioner."""

    def setUp(self):
        """Create temporary workspace directory for test datasets."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.d = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_verify_valid_sample_files(self):
        """Verify bundled sample dataset files pass audit without errors."""
        # Sample data ships in datasets/; resolve relative to this file so the test works from any cwd.
        tooluse_path = Path(__file__).resolve().parent.parent / "datasets" / "sample_tooluse_pairs.jsonl"
        rep = verify_file(tooluse_path)
        self.assertEqual(rep["status"], "PASSED")
        self.assertGreater(rep["valid_records"], 0)
        self.assertEqual(rep["malformed_lines"], 0)

        repair_path = Path(__file__).resolve().parent.parent / "datasets" / "sample_code_repair_pairs.jsonl"
        # check_ast=True additionally parses string completions as Python.
        rep_repair = verify_file(repair_path, check_ast=True)
        self.assertEqual(rep_repair["status"], "PASSED")
        self.assertGreater(rep_repair["valid_records"], 0)
        self.assertEqual(rep_repair["ast_syntax_errors"], 0)

    def test_verify_malformed_and_duplicate_detection(self):
        """Verify detector catches malformed JSON lines and exact duplicates."""
        test_file = self.d / "corrupt.jsonl"
        # Two malformed lines (bad JSON, wrong schema) and one exact duplicate among five lines.
        lines = [
            json.dumps({"prompt": "p1", "chosen": "c1"}),
            "not a json line",
            json.dumps({"bad_schema": True}),
            json.dumps({"prompt": "p1", "chosen": "c1"}),  # duplicate
            json.dumps({"prompt": "p2", "chosen": "c2"}),
        ]
        test_file.write_text("\n".join(lines))
        rep = verify_file(test_file)
        # Malformed = invalid JSON plus a row missing the required keys.
        self.assertEqual(rep["malformed_lines"], 2)
        self.assertEqual(rep["exact_duplicates_removed"], 1)
        self.assertEqual(rep["valid_records"], 2)

    def test_split_family_held_out_disjointness(self):
        """Verify train and dev partitions share zero overlapping prompts."""
        # 30 synthetic rows across 3 families.
        rows = [
            {"prompt": f"prompt_{i}", "chosen": f"chosen_{i}", "family": f"fam_{i % 3}"}
            for i in range(30)
        ]
        # Fixed seed keeps the assertions stable across runs.
        train, dev = split_family_held_out(rows, dev_fraction=0.25, seed=42)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(dev), 0)
        self.assertEqual(len(train) + len(dev), 30)

        # Enforce zero prompt overlap (strict disjointness)
        train_prompts = {r["prompt"] for r in train}
        dev_prompts = {r["prompt"] for r in dev}
        self.assertEqual(len(train_prompts & dev_prompts), 0)


if __name__ == "__main__":
    unittest.main()
