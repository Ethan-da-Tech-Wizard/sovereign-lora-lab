# ==============================================================================
# Program: test_adapter_merger.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Unit tests for LoRA adapter merging and task arithmetic.
#          Tests Delta_W calculation, Linear Task Arithmetic, Power Iteration
#          Truncated SVD rank compression, TIES-Merging, and DARE sparsification.
# ==============================================================================

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
    """Test suite validating matrix operations and adapter fusion algorithms.

    All matrices are tiny pure-Python lists (core/adapter_merger.py has no NumPy dependency),
    so every expected value can be checked by hand. Several tests below assert only output
    SHAPES, not numeric values: they catch indexing and dimension bugs but would not catch a
    wrong merge formula. test_compute_delta is the one test that pins exact numbers.
    """

    def setUp(self):
        """Set up synthetic low-rank adapter matrices for testing."""
        # 4x2 matrix B, 2x3 matrix A -> Delta_W is 4x3
        # LoRA factors: B is (out x r) and A is (r x in); the weight delta is (alpha/r) * B @ A.
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
        """Verify Delta_W = (alpha / r) * (B @ A) computation.

        Scenario: adapter_1 (alpha=16, r=2). Why it matters: every merge method starts from
        this per-adapter weight delta, so an error here corrupts all of them.
        """
        delta = self.adapter1.compute_delta()
        self.assertEqual(len(delta), 4)
        self.assertEqual(len(delta[0]), 3)
        # Expected: B @ A with scaling = 16.0 / 2 = 8.0
        # Row 0: [1.0, 0.0] @ A -> [1.0, 2.0, 0.0] * 8.0 = [8.0, 16.0, 0.0]
        # Hand-computed: scale = alpha/r = 16/2 = 8, row 0 of B@A is [1, 2, 0].
        self.assertAlmostEqual(delta[0][0], 8.0)
        self.assertAlmostEqual(delta[0][1], 16.0)
        self.assertAlmostEqual(delta[0][2], 0.0)

    def test_linear_task_arithmetic(self):
        """Verify weighted linear task arithmetic across multiple adapters.

        Scenario: two adapters averaged with weights 0.5/0.5. Limit: shape-only assertions.
        """
        # Equal 0.5/0.5 weights average the two adapters' deltas; only the shape is asserted here.
        merged = linear_task_arithmetic([self.adapter1, self.adapter2], weights=[0.5, 0.5])
        self.assertEqual(len(merged), 4)
        self.assertEqual(len(merged[0]), 3)

    def test_power_iteration_svd_rank_k(self):
        """Verify power iteration SVD factorizes matrix into rank-k factors.

        Scenario: the classic 3x3 matrix 1..9, which has rank 2. Why it matters: truncated SVD
        merging relies on this routine to compress a fused delta back into LoRA form.
        Limit: the reconstruction is checked for shape only, not for closeness to the input.
        """
        # 3x3 matrix of rank 2 (rows are linearly dependent), so a rank-2 factorization can represent it.
        matrix = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        # 30 iterations is ample for this tiny matrix; factors must have shapes (3x2) and (2x3).
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
        """Verify truncated SVD compresses multi-adapter sum into target rank.

        Scenario: sum two rank-2 adapters, re-factor at rank 2. The result must again be a
        valid (out x r, r x in) LoRA pair so it can be loaded as an adapter.
        """
        # Fusing two rank-2 adapters and re-compressing to rank 2 must still give (out x r) and (r x in) factors.
        B_fused, A_fused = truncated_svd_merge([self.adapter1, self.adapter2], target_rank=2)
        self.assertEqual(len(B_fused), 4)
        self.assertEqual(len(B_fused[0]), 2)
        self.assertEqual(len(A_fused), 2)
        self.assertEqual(len(A_fused[0]), 3)

    def test_ties_merge(self):
        """Verify TIES merging trims parameters and resolves sign collisions.

        Scenario: keep the top 70% of entries by magnitude from each adapter, then resolve sign
        conflicts. Limit: shape-only assertions; trimming and sign election are not checked.
        """
        # density=0.7 keeps the top 70% of entries by magnitude before resolving sign conflicts.
        merged = ties_merge([self.adapter1, self.adapter2], density=0.7)
        self.assertEqual(len(merged), 4)
        self.assertEqual(len(merged[0]), 3)

    def test_dare_sparsify(self):
        """Verify DARE sparsification rescales un-dropped parameters.

        Scenario: drop half of a 2x2 delta with a fixed seed. Limit: only the output shape is
        asserted, not which entries were dropped or the 1/(1-drop_rate) rescaling.
        """
        # Seeded so the random drop mask is reproducible; only output shape is asserted.
        delta = [[1.0, 2.0], [3.0, 4.0]]
        sparsified = dare_sparsify(delta, drop_rate=0.5, seed=123)
        self.assertEqual(len(sparsified), 2)
        self.assertEqual(len(sparsified[0]), 2)


if __name__ == "__main__":
    unittest.main()
