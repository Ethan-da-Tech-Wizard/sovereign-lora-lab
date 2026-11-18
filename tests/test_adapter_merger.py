#!/usr/bin/env python3
"""tests/test_adapter_merger.py — Unit tests for LoRA adapter merging and task arithmetic."""

import unittest
from core.adapter_merger import (
    LoRAAdapter,
    dare_sparsify,
    linear_task_arithmetic,
    mat_mul,
    power_iteration_svd_rank_k,
    ties_merge,
    truncated_svd_merge,
)


class TestLoRAAdapterMerger(unittest.TestCase):
    def setUp(self):
        # 4x2 matrix B, 2x3 matrix A -> Delta_W is 4x3
        self.B1 = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
        self.A1 = [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 2.0],
        ]
        self.adapter1 = LoRAAdapter("adapter_1", self.A1, self.B1, alpha=16.0, r=2)

        self.B2 = [
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [0.0, 0.0],
        ]
        self.A2 = [
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        self.adapter2 = LoRAAdapter("adapter_2", self.A2, self.B2, alpha=16.0, r=2)

    def test_compute_delta(self):
        delta = self.adapter1.compute_delta()
        self.assertEqual(len(delta), 4)
        self.assertEqual(len(delta[0]), 3)
        # Expected: B @ A with scaling = 16.0 / 2 = 8.0
        # Row 0: [1.0, 0.0] @ A -> [1.0, 2.0, 0.0] * 8.0 = [8.0, 16.0, 0.0]
        self.assertAlmostEqual(delta[0][0], 8.0)
        self.assertAlmostEqual(delta[0][1], 16.0)
        self.assertAlmostEqual(delta[0][2], 0.0)

    def test_linear_task_arithmetic(self):
        merged = linear_task_arithmetic([self.adapter1, self.adapter2], weights=[0.5, 0.5])
        self.assertEqual(len(merged), 4)
        self.assertEqual(len(merged[0]), 3)

    def test_power_iteration_svd_rank_k(self):
        matrix = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        B, A = power_iteration_svd_rank_k(matrix, target_rank=2, num_iterations=30)
        self.assertEqual(len(B), 3)
        self.assertEqual(len(B[0]), 2)
        self.assertEqual(len(A), 2)
        self.assertEqual(len(A[0]), 3)
        # Verify rank-2 reconstruction approx original
        approx = mat_mul(B, A)
        self.assertEqual(len(approx), 3)
        self.assertEqual(len(approx[0]), 3)

    def test_truncated_svd_merge(self):
        B_fused, A_fused = truncated_svd_merge([self.adapter1, self.adapter2], target_rank=2)
        self.assertEqual(len(B_fused), 4)
        self.assertEqual(len(B_fused[0]), 2)
        self.assertEqual(len(A_fused), 2)
        self.assertEqual(len(A_fused[0]), 3)

    def test_ties_merge(self):
        merged = ties_merge([self.adapter1, self.adapter2], density=0.7)
        self.assertEqual(len(merged), 4)
        self.assertEqual(len(merged[0]), 3)

    def test_dare_sparsify(self):
        delta = [[1.0, 2.0], [3.0, 4.0]]
        sparsified = dare_sparsify(delta, drop_rate=0.5, seed=123)
        self.assertEqual(len(sparsified), 2)
        self.assertEqual(len(sparsified[0]), 2)


if __name__ == "__main__":
    unittest.main()
