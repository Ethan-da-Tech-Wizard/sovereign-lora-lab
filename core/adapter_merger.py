# ==============================================================================
# Program: adapter_merger.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Offline Multi-LoRA Adapter Fusion & Task Arithmetic Engine.
#          Combines multiple Low-Rank Adapters (LoRAs) into a unified master
#          adapter or directly merges them into base model weights with zero
#          cloud dependencies. Implements Linear Task Arithmetic, Power Iteration
#          Truncated SVD rank compression (preserving rank r without parameter explosion),
#          TIES-Merging (Trimming, Electing Signs, Disjoint Merging) to resolve
#          parameter interference, and DARE (Drop And REscale) sparsification.
# ==============================================================================

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

# Try importing numpy for acceleration if available; fallback to pure Python
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ==============================================================================
# SECTION 1: Pure Python Standard Library Matrix Utilities
# Educational Note: These basic linear algebra functions allow the merger to run
# anywhere—even on minimal Python environments without NumPy or PyTorch installed.
# ==============================================================================

def mat_zeros(rows: int, cols: int) -> list[list[float]]:
    """Create a 2D matrix of dimensions rows x cols initialized to 0.0.

    Parameters:
        rows (int): Number of horizontal rows.
        cols (int): Number of vertical columns.

    Returns:
        list[list[float]]: A 2D list where every element is initialized to float 0.0.
    """
    # Use nested list comprehension for clean, independent row allocation
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two 2D matrices A and B using the standard dot product algorithm.

    Mathematical Concept:
        Matrix multiplication C = A @ B requires that the number of columns in A
        matches the number of rows in B. If A is (m x k) and B is (k x n), the
        resulting matrix C has dimensions (m x n). Each cell C[i][j] is computed
        as the sum of A[i][k] * B[k][j] for all k from 0 to c1-1.

    Parameters:
        A (list[list[float]]): Left matrix of shape (r, c1).
        B (list[list[float]]): Right matrix of shape (c1, c2).

    Returns:
        list[list[float]]: Result matrix C of shape (r, c2).
    """
    # Step 1: Extract matrix dimensions
    r, c1 = len(A), len(A[0])
    c2 = len(B[0])

    # Step 2: Initialize result matrix C with all zeros
    C = mat_zeros(r, c2)

    # Step 3: Compute dot products using a cache-friendly loop order (i, k, j)
    # Loop over each row i of matrix A
    for i in range(r):
        # Loop over each shared inner dimension k
        for k in range(c1):
            aik = A[i][k]
            # Optimization: Skip multiplication if scalar is zero
            if aik != 0.0:
                # Accumulate into each column j of matrix B
                for j in range(c2):
                    C[i][j] += aik * B[k][j]

    # Step 4: Return completed product matrix
    return C


def mat_scale(A: list[list[float]], factor: float) -> list[list[float]]:
    """Multiply every element in matrix A by a scalar float factor.

    Mathematical Concept:
        Scalar multiplication scales the magnitude of a matrix uniformly:
        (c * A)[i][j] = c * A[i][j]. Used to apply LoRA scaling (alpha / r).

    Parameters:
        A (list[list[float]]): The input matrix.
        factor (float): The scalar scaling multiplier.

    Returns:
        list[list[float]]: A new matrix with all elements scaled.
    """
    return [[x * factor for x in row] for row in A]


def mat_add(A: list[list[float]], B: list[list[float]], w_a: float = 1.0, w_b: float = 1.0) -> list[list[float]]:
    """Compute a weighted element-wise sum of two identically shaped matrices.

    Mathematical Concept:
        Result[i][j] = (w_a * A[i][j]) + (w_b * B[i][j])

    Parameters:
        A (list[list[float]]): First matrix of shape (r, c).
        B (list[list[float]]): Second matrix of shape (r, c).
        w_a (float): Weight scalar for matrix A (default: 1.0).
        w_b (float): Weight scalar for matrix B (default: 1.0).

    Returns:
        list[list[float]]: Weighted sum matrix of shape (r, c).
    """
    r, c = len(A), len(A[0])
    return [[w_a * A[i][j] + w_b * B[i][j] for j in range(c)] for i in range(r)]


# ==============================================================================
# SECTION 2: LoRA Adapter Abstraction
# Educational Note: Low-Rank Adaptation (LoRA) freezes the pre-trained weights W_0
# and decomposes the weight update into two low-rank matrices B and A:
#   Delta_W = (alpha / r) * (B @ A)
# Where:
#   - A has shape (r, in_features) and is typically initialized from Gaussian noise
#   - B has shape (out_features, r) and is initialized to zero so Delta_W starts at 0
#   - r is the rank (e.g., 8, 16, 32), which is much smaller than in_features / out_features
#   - alpha is a constant scaling hyperparameter (typically 2 * r or 1 * r)
# ==============================================================================

class LoRAAdapter:
    """Represents a low-rank adapter delta matrix for a neural network layer.

    Attributes:
        name (str): Identifier for the adapter (e.g. 'coding_expert', 'tool_use').
        A (list[list[float]]): Down-projection matrix of shape (r, in_features).
        B (list[list[float]]): Up-projection matrix of shape (out_features, r).
        alpha (float): LoRA scaling factor (typically 16.0 or 32.0).
        r (int): Intrinsic rank dimension of the adapter.
    """
    __slots__ = ("name", "A", "B", "alpha", "r")

    def __init__(self, name: str, A: list[list[float]], B: list[list[float]], alpha: float = 16.0, r: int = 16):
        """Initialize the LoRA adapter with its name, factor matrices, and scaling."""
        self.name = name
        self.A = A          # Shape: (r, in_features)
        self.B = B          # Shape: (out_features, r)
        self.alpha = float(alpha)
        self.r = int(r)

    @property
    def scaling(self) -> float:
        """Compute the constant scaling factor: alpha / r.

        Educational Note: This scaling factor stabilizes training when changing rank r,
        ensuring the magnitude of gradient updates remains consistent.
        """
        return self.alpha / self.r

    def compute_delta(self) -> list[list[float]]:
        """Compute the full explicit delta weight matrix Delta_W = (alpha / r) * (B @ A).

        Educational Note:
            While during inference or training LoRA computes (x @ A.T @ B.T) to save memory,
            when merging adapters or fusing them into base model weights, we compute the
            full explicit Delta_W = B @ A scaled by (alpha / r).

        Returns:
            list[list[float]]: The 2D matrix of shape (out_features, in_features).
        """
        # Step 1: Multiply up-projection matrix B by down-projection matrix A
        product = mat_mul(self.B, self.A)

        # Step 2: Scale the resulting product matrix by (alpha / r)
        return mat_scale(product, self.scaling)


# ==============================================================================
# SECTION 3: Task Arithmetic & Linear Merging
# Educational Note: Task Arithmetic treats fine-tuned weights as directional vectors
# in parameter space. By taking linear combinations of task vectors, a model can
# acquire multiple capabilities simultaneously (e.g., coding + tool use).
# ==============================================================================

def linear_task_arithmetic(
    adapters: List[LoRAAdapter],
    weights: Optional[List[float]] = None
) -> list[list[float]]:
    """Compute a linear weighted combination of adapter task vectors: sum(w_i * Delta_W_i).

    Mathematical Formulation:
        Delta_W_merged = w_1 * Delta_W_1 + w_2 * Delta_W_2 + ... + w_k * Delta_W_k

    Parameters:
        adapters (List[LoRAAdapter]): List of LoRA adapters to combine.
        weights (Optional[List[float]]): Importance weight for each adapter. If None,
                                         weights default to equal uniform distribution (1/k).

    Returns:
        list[list[float]]: The fused full delta weight matrix Delta_W_merged.

    Raises:
        ValueError: If adapters list is empty or weights length does not match adapters.
    """
    # Step 1: Validate input adapters
    if not adapters:
        raise ValueError("Must provide at least one adapter.")

    # Step 2: Default to uniform weighting if no weights were provided
    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters)
    assert len(adapters) == len(weights), "Adapters and weights length must match."

    # Step 3: Determine dimensions from the first adapter's delta
    first_delta = adapters[0].compute_delta()
    rows, cols = len(first_delta), len(first_delta[0])

    # Step 4: Initialize the accumulator matrix with zeros
    total_delta = mat_zeros(rows, cols)

    # Step 5: Accumulate each adapter's scaled delta matrix into total_delta
    for adapter, weight in zip(adapters, weights):
        delta = adapter.compute_delta()
        for i in range(rows):
            for j in range(cols):
                total_delta[i][j] += weight * delta[i][j]

    # Step 6: Return the linearly combined delta matrix
    return total_delta


# ==============================================================================
# SECTION 4: Truncated SVD via Power Iteration & Deflation
# Educational Note:
# When fusing multiple adapters of rank r (say k adapters), the rank of the sum can
# reach k * r. If we want to serve this master adapter using a single LoRA module of
# fixed rank r without exploding VRAM, we use Singular Value Decomposition (SVD):
#   Delta_W = U @ Sigma @ V.T
# By keeping only the top-k singular values (truncated SVD), we get the mathematically
# optimal rank-k approximation (Eckart-Young-Mirsky Theorem).
#
# Here we implement Power Iteration with Hotelling Deflation in pure Python:
#   1. Start with a random unit vector v.
#   2. Repeatedly compute u = A @ v, normalize u; then v = A.T @ u, normalize v.
#   3. Compute singular value sigma = u.T @ A @ v.
#   4. Store factors and "deflate" (subtract sigma * u @ v.T from matrix) to find the next component.
# ==============================================================================

def power_iteration_svd_rank_k(
    matrix: list[list[float]],
    target_rank: int = 8,
    num_iterations: int = 20
) -> Tuple[list[list[float]], list[list[float]]]:
    """Compute top-k singular components using Power Iteration & Deflation.

    Mathematical Concept:
        Any real matrix M can be decomposed as:
            M = sum_{i=1}^k sigma_i * (u_i @ v_i^T)
        where:
            sigma_i is the i-th singular value (strength of feature i)
            u_i is the left singular vector (output direction)
            v_i is the right singular vector (input direction)

        To factorize back into LoRA form (B @ A):
            B[:, i] = sqrt(sigma_i) * u_i
            A[i, :] = sqrt(sigma_i) * v_i
        So that:
            (B @ A) = sum_i sqrt(sigma_i)*u_i * sqrt(sigma_i)*v_i^T = sum_i sigma_i * (u_i @ v_i^T) approx M

    Parameters:
        matrix (list[list[float]]): Target matrix to compress of shape (m, n).
        target_rank (int): Number of top singular components to extract (k).
        num_iterations (int): Power iteration convergence loops per component (default: 20).

    Returns:
        Tuple[list[list[float]], list[list[float]]]:
            (B_compressed, A_compressed) where B has shape (m, k) and A has shape (k, n).
    """
    m, n = len(matrix), len(matrix[0])
    # The maximum possible rank cannot exceed the smaller matrix dimension
    k = min(target_rank, m, n)

    # Step 1: Create a working copy of the matrix so we do not mutate the input
    A_work = [row[:] for row in matrix]
    B_cols: list[list[float]] = []  # Holds column vectors of B
    A_rows: list[list[float]] = []  # Holds row vectors of A

    # Step 2: Iterate to extract each of the k singular components sequentially
    for comp in range(k):
        # Step 2a: Initialize a deterministic pseudo-random unit vector v
        rng = random.Random(42 + comp)
        v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        v_norm = math.sqrt(sum(x * x for x in v))
        v = [x / (v_norm + 1e-12) for x in v]

        # Step 2b: Power iteration loop to converge v and u to the principal directions
        for _ in range(num_iterations):
            # Forward projection: u = A_work @ v (shape: m)
            u = [sum(A_work[i][j] * v[j] for j in range(n)) for i in range(m)]
            u_norm = math.sqrt(sum(x * x for x in u))
            if u_norm < 1e-12:
                # Vanishing vector: no more non-trivial energy in the matrix
                break
            # Normalize u to unit length
            u = [x / u_norm for x in u]

            # Backward projection: v = A_work.T @ u (shape: n)
            v = [sum(A_work[i][j] * u[i] for i in range(m)) for j in range(n)]
            v_norm = math.sqrt(sum(x * x for x in v))
            if v_norm < 1e-12:
                break
            # Normalize v to unit length
            v = [x / v_norm for x in v]

        # Step 2c: Compute singular value sigma = u.T @ A_work @ v (Rayleigh quotient)
        Av = [sum(A_work[i][j] * v[j] for j in range(n)) for i in range(m)]
        sigma = sum(u[i] * Av[i] for i in range(m))

        # If sigma is non-positive or negligible, no further meaningful ranks exist
        if sigma <= 0.0:
            break

        # Step 2d: Distribute sqrt(sigma) equally across u and v
        sqrt_s = math.sqrt(sigma)
        u_scaled = [x * sqrt_s for x in u]
        v_scaled = [x * sqrt_s for x in v]

        B_cols.append(u_scaled)
        A_rows.append(v_scaled)

        # Step 2e: Hotelling Deflation: Subtract the found component from A_work
        # A_work = A_work - sigma * (u @ v.T)
        for i in range(m):
            ui = u[i]
            for j in range(n):
                A_work[i][j] -= sigma * ui * v[j]

    # Step 3: Reconstruct B matrix from column vectors B_cols
    actual_k = len(B_cols)
    B_out = mat_zeros(m, actual_k)
    for j in range(actual_k):
        for i in range(m):
            B_out[i][j] = B_cols[j][i]

    # Step 4: A matrix is simply the list of row vectors A_rows
    A_out = A_rows
    return B_out, A_out


def truncated_svd_merge(
    adapters: List[LoRAAdapter],
    target_rank: int = 8,
    weights: Optional[List[float]] = None
) -> Tuple[list[list[float]], list[list[float]]]:
    """Merge adapters and compress the resulting delta matrix into a rank-k pair (B_merged, A_merged).

    Algorithm Steps:
        1. Compute the linearly combined Delta_W from all adapters.
        2. Perform power iteration SVD on Delta_W to extract top `target_rank` singular values.
        3. Return decomposed factor matrices (B, A) ready for serving.

    Parameters:
        adapters (List[LoRAAdapter]): Adapters to fuse.
        target_rank (int): Desired final rank of the fused adapter.
        weights (Optional[List[float]]): Importance weight per adapter.

    Returns:
        Tuple[list[list[float]], list[list[float]]]: (B_merged, A_merged).
    """
    # Step 1: Perform task arithmetic to form the uncompressed fused delta matrix
    merged_delta = linear_task_arithmetic(adapters, weights)

    # Step 2: Compress into target_rank using Power Iteration SVD
    return power_iteration_svd_rank_k(merged_delta, target_rank=target_rank)


# ==============================================================================
# SECTION 5: TIES-Merging (Trimming, Electing Sign, Disjoint Merging)
# Educational Note:
# When combining multiple fine-tuned models, naive averaging causes parameter
# interference: positive and negative updates from different tasks cancel each other
# out, degrading performance. TIES solves this in three steps:
#   1. Trimming: Keep only the top-density fraction of parameters by absolute value
#      (throwing away noise).
#   2. Electing Sign: For each parameter position, compute the majority sign across
#      models (+1 or -1).
#   3. Disjoint Merging: Only average parameters from models whose sign agrees with
#      the elected sign, preventing cancellation.
# ==============================================================================

def ties_merge(
    adapters: List[LoRAAdapter],
    density: float = 0.5,
    weights: Optional[List[float]] = None
) -> list[list[float]]:
    """Perform TIES-Merging across multiple LoRA adapters.

    Parameters:
        adapters (List[LoRAAdapter]): List of adapters to merge.
        density (float): Fraction of top-magnitude weights to retain (0.0 to 1.0).
                         e.g. 0.5 retains the top 50% largest magnitude values.
        weights (Optional[List[float]]): Optional weights per adapter.

    Returns:
        list[list[float]]: Fused delta weight matrix resolving sign collisions.
    """
    # Step 1: Validation
    if not adapters:
        raise ValueError("Must provide at least one adapter.")
    deltas = [a.compute_delta() for a in adapters]
    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters)

    rows, cols = len(deltas[0]), len(deltas[0][0])
    total_elements = rows * cols
    k_retain = max(1, int(total_elements * density))

    # -------------------------------------------------------------------------
    # STEP 1: Trimming (Keep only top-k highest absolute magnitude parameters)
    # -------------------------------------------------------------------------
    trimmed_deltas = []
    for d in deltas:
        # Collect absolute values of all weights in this adapter
        flat_abs = [abs(d[i][j]) for i in range(rows) for j in range(cols)]
        flat_abs.sort(reverse=True)
        # Find threshold value at position k_retain - 1
        thresh = flat_abs[min(k_retain - 1, len(flat_abs) - 1)]

        # Zero out any weight smaller than the threshold
        trimmed = [[d[i][j] if abs(d[i][j]) >= thresh else 0.0 for j in range(cols)] for i in range(rows)]
        trimmed_deltas.append(trimmed)

    # -------------------------------------------------------------------------
    # STEP 2 & 3: Elect Sign & Disjoint Accumulation
    # -------------------------------------------------------------------------
    merged = mat_zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            # Step 2: Elect sign by voting (+1 for positive, -1 for negative, 0 for zero)
            vote = sum(1.0 if td[i][j] > 0 else (-1.0 if td[i][j] < 0 else 0.0) for td in trimmed_deltas)
            elected_sign = 1.0 if vote >= 0 else -1.0

            # Step 3: Accumulate only values that agree with the elected sign
            val_accum = 0.0
            weight_accum = 0.0
            for td, w in zip(trimmed_deltas, weights):
                v = td[i][j]
                if v != 0.0 and ((v > 0 and elected_sign > 0) or (v < 0 and elected_sign < 0)):
                    val_accum += w * v
                    weight_accum += w

            # Compute weighted average of agreeing values
            if weight_accum > 0.0:
                merged[i][j] = val_accum / weight_accum

    return merged


# ==============================================================================
# SECTION 6: DARE (Drop And REscale) Sparsification
# Educational Note:
# DARE randomly zeroes out a fraction p of weight updates and rescales the remaining
# updates by 1 / (1 - p). By doing so, the mathematical expected value of each weight
# is perfectly preserved:
#   E[W_dare] = (1 - p) * (W / (1 - p)) + p * 0 = W
# This drastically reduces adapter density and interference without sacrificing accuracy.
# ==============================================================================

def dare_sparsify(delta: list[list[float]], drop_rate: float = 0.5, seed: int = 42) -> list[list[float]]:
    """DARE (Drop And REscale): Randomly drop `drop_rate` fraction of weights and rescale remaining.

    Mathematical Concept:
        For each weight w:
            w_new = 0                       with probability `drop_rate`
            w_new = w / (1 - drop_rate)     with probability `1 - drop_rate`

    Parameters:
        delta (list[list[float]]): Input delta matrix.
        drop_rate (float): Fraction of weights to drop randomly (0.0 <= drop_rate < 1.0).
        seed (int): Random seed for reproducibility.

    Returns:
        list[list[float]]: Sparsified and rescaled matrix.
    """
    # Step 1: Initialize random generator with seed
    rng = random.Random(seed)

    # Step 2: Compute rescaling factor: 1 / (1 - drop_rate)
    scale = 1.0 / max(1e-6, 1.0 - drop_rate)

    # Step 3: Apply drop and rescale to each element
    return [
        [0.0 if rng.random() < drop_rate else val * scale for val in row]
        for row in delta
    ]
