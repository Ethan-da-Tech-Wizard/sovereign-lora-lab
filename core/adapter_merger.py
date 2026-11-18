#!/usr/bin/env python3
"""core/adapter_merger.py — Offline LoRA Adapter Fusion & Task Arithmetic Engine.

Combines multiple trained low-rank adapters (LoRAs) into a unified master adapter
or directly merges them into base weights with zero cloud dependencies.

Supports:
  1. Linear Task Arithmetic (weighted sum)
  2. Truncated SVD / Power Iteration Rank Compression (maintains fixed rank r)
  3. TIES Merging (Trimming, Electing Sign, Disjoint Merging)
  4. DARE Sparsification (Drop And REscale)
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Pure Python Standard Library Matrix Utilities
# ============================================================================

def mat_zeros(rows: int, cols: int) -> list[list[float]]:
    """Create a rows x cols matrix initialized to zero."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply matrix A (r x c1) with B (c1 x c2)."""
    r, c1 = len(A), len(A[0])
    c2 = len(B[0])
    C = mat_zeros(r, c2)
    for i in range(r):
        for k in range(c1):
            aik = A[i][k]
            if aik != 0.0:
                for j in range(c2):
                    C[i][j] += aik * B[k][j]
    return C


def mat_scale(A: list[list[float]], factor: float) -> list[list[float]]:
    """Multiply matrix by scalar factor."""
    return [[x * factor for x in row] for row in A]


def mat_add(A: list[list[float]], B: list[list[float]], w_a: float = 1.0, w_b: float = 1.0) -> list[list[float]]:
    """Weighted sum of two identically shaped matrices."""
    r, c = len(A), len(A[0])
    return [[w_a * A[i][j] + w_b * B[i][j] for j in range(c)] for i in range(r)]


class LoRAAdapter:
    """Represents a low-rank adapter delta matrix for a layer: Delta_W = (alpha / r) * (B @ A)."""
    __slots__ = ("name", "A", "B", "alpha", "r")

    def __init__(self, name: str, A: list[list[float]], B: list[list[float]], alpha: float = 16.0, r: int = 16):
        self.name = name
        self.A = A  # Shape: (r, in_features)
        self.B = B  # Shape: (out_features, r)
        self.alpha = float(alpha)
        self.r = int(r)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def compute_delta(self) -> list[list[float]]:
        """Computes the full explicit delta weight matrix Delta_W."""
        product = mat_mul(self.B, self.A)
        return mat_scale(product, self.scaling)


def linear_task_arithmetic(
    adapters: List[LoRAAdapter],
    weights: Optional[List[float]] = None
) -> list[list[float]]:
    """Computes a linear weighted combination of adapter task vectors: sum(w_i * Delta_W_i)."""
    if not adapters:
        raise ValueError("Must provide at least one adapter.")
    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters)
    assert len(adapters) == len(weights), "Adapters and weights length must match."

    first_delta = adapters[0].compute_delta()
    rows, cols = len(first_delta), len(first_delta[0])
    total_delta = mat_zeros(rows, cols)

    for adapter, weight in zip(adapters, weights):
        delta = adapter.compute_delta()
        for i in range(rows):
            for j in range(cols):
                total_delta[i][j] += weight * delta[i][j]

    return total_delta


def power_iteration_svd_rank_k(
    matrix: list[list[float]],
    target_rank: int = 8,
    num_iterations: int = 20
) -> Tuple[list[list[float]], list[list[float]]]:
    """Computes top-k singular components using Power Iteration & Gram-Schmidt.
    Returns (B_compressed, A_compressed) such that B @ A approx matrix.
    Pure standard library implementation with zero external dependencies.
    """
    m, n = len(matrix), len(matrix[0])
    k = min(target_rank, m, n)

    A_work = [row[:] for row in matrix]
    B_cols: list[list[float]] = []
    A_rows: list[list[float]] = []

    for comp in range(k):
        rng = random.Random(42 + comp)
        v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        v_norm = math.sqrt(sum(x * x for x in v))
        v = [x / (v_norm + 1e-12) for x in v]

        for _ in range(num_iterations):
            # u = A_work @ v
            u = [sum(A_work[i][j] * v[j] for j in range(n)) for i in range(m)]
            u_norm = math.sqrt(sum(x * x for x in u))
            if u_norm < 1e-12:
                break
            u = [x / u_norm for x in u]

            # v = A_work.T @ u
            v = [sum(A_work[i][j] * u[i] for i in range(m)) for j in range(n)]
            v_norm = math.sqrt(sum(x * x for x in v))
            if v_norm < 1e-12:
                break
            v = [x / v_norm for x in v]

        # Singular value sigma = u.T @ A_work @ v
        Av = [sum(A_work[i][j] * v[j] for j in range(n)) for i in range(m)]
        sigma = sum(u[i] * Av[i] for i in range(m))

        if sigma <= 0.0:
            break

        sqrt_s = math.sqrt(sigma)
        u_scaled = [x * sqrt_s for x in u]
        v_scaled = [x * sqrt_s for x in v]

        B_cols.append(u_scaled)
        A_rows.append(v_scaled)

        # Deflate: A_work = A_work - sigma * (u @ v.T)
        for i in range(m):
            ui = u[i]
            for j in range(n):
                A_work[i][j] -= sigma * ui * v[j]

    actual_k = len(B_cols)
    B_out = mat_zeros(m, actual_k)
    for j in range(actual_k):
        for i in range(m):
            B_out[i][j] = B_cols[j][i]

    A_out = A_rows
    return B_out, A_out


def truncated_svd_merge(
    adapters: List[LoRAAdapter],
    target_rank: int = 8,
    weights: Optional[List[float]] = None
) -> Tuple[list[list[float]], list[list[float]]]:
    """Merges adapters and compresses the resulting delta matrix into
    a compact rank-(target_rank) pair (B_merged, A_merged).
    """
    merged_delta = linear_task_arithmetic(adapters, weights)
    return power_iteration_svd_rank_k(merged_delta, target_rank=target_rank)


def ties_merge(
    adapters: List[LoRAAdapter],
    density: float = 0.5,
    weights: Optional[List[float]] = None
) -> list[list[float]]:
    """TIES-Merging (Trimming, Electing Sign, Disjoint Merging)."""
    if not adapters:
        raise ValueError("Must provide at least one adapter.")
    deltas = [a.compute_delta() for a in adapters]
    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters)

    rows, cols = len(deltas[0]), len(deltas[0][0])
    total_elements = rows * cols
    k_retain = max(1, int(total_elements * density))

    # 1. Trimming
    trimmed_deltas = []
    for d in deltas:
        flat_abs = [abs(d[i][j]) for i in range(rows) for j in range(cols)]
        flat_abs.sort(reverse=True)
        thresh = flat_abs[min(k_retain - 1, len(flat_abs) - 1)]

        trimmed = [[d[i][j] if abs(d[i][j]) >= thresh else 0.0 for j in range(cols)] for i in range(rows)]
        trimmed_deltas.append(trimmed)

    # 2. Elect Sign & Disjoint Accumulate
    merged = mat_zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            vote = sum(1.0 if td[i][j] > 0 else (-1.0 if td[i][j] < 0 else 0.0) for td in trimmed_deltas)
            elected_sign = 1.0 if vote >= 0 else -1.0

            val_accum = 0.0
            weight_accum = 0.0
            for td, w in zip(trimmed_deltas, weights):
                v = td[i][j]
                if v != 0.0 and ((v > 0 and elected_sign > 0) or (v < 0 and elected_sign < 0)):
                    val_accum += w * v
                    weight_accum += w

            if weight_accum > 0.0:
                merged[i][j] = val_accum / weight_accum

    return merged


def dare_sparsify(delta: list[list[float]], drop_rate: float = 0.5, seed: int = 42) -> list[list[float]]:
    """DARE (Drop And REscale):
    Randomly drops `drop_rate` fraction of weights and rescales remaining by 1 / (1 - drop_rate).
    """
    rng = random.Random(seed)
    scale = 1.0 / max(1e-6, 1.0 - drop_rate)
    return [
        [0.0 if rng.random() < drop_rate else val * scale for val in row]
        for row in delta
    ]
