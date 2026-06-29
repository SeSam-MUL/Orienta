"""Weighted quaternion averaging for post-FAISS refinement.

Refines discrete nearest-neighbor orientations into sub-grid accuracy
using distance-weighted quaternion averaging (Markley et al., 2007).
"""
from __future__ import annotations

import numpy as np


def weighted_quaternion_average(
    quaternions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute weighted average of quaternions.

    Uses the eigenvector method (Markley et al., 2007):
    the average quaternion is the eigenvector corresponding to
    the largest eigenvalue of the weighted outer product sum.

    Parameters
    ----------
    quaternions : np.ndarray
        (K, 4) unit quaternions.
    weights : np.ndarray
        (K,) non-negative weights (need not sum to 1).

    Returns
    -------
    np.ndarray
        (4,) unit quaternion average.
    """
    weights = weights / weights.sum()

    # Build 4x4 weighted outer product matrix
    M = np.zeros((4, 4), dtype=np.float64)
    for i in range(len(quaternions)):
        q = quaternions[i].astype(np.float64)
        M += weights[i] * np.outer(q, q)

    # Average quaternion is eigenvector of largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg = eigenvectors[:, -1]  # largest eigenvalue is last

    # Ensure unit norm
    avg = avg / np.linalg.norm(avg)
    return avg.astype(np.float32)


def refine_orientations(
    top_k_orientations: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Refine orientations using distance-weighted quaternion averaging.

    Parameters
    ----------
    top_k_orientations : np.ndarray
        (N, K, 4) top-K quaternions per query.
    distances : np.ndarray
        (N, K) inner product distances (higher = closer).

    Returns
    -------
    np.ndarray
        (N, 4) refined unit quaternions.
    """
    n = top_k_orientations.shape[0]
    refined = np.empty((n, 4), dtype=np.float32)

    for i in range(n):
        # Convert inner product distances to weights via softmax
        d = distances[i].astype(np.float64)
        d = d - d.max()  # numerical stability
        weights = np.exp(d * 10.0)  # temperature scaling
        weights = weights / weights.sum()

        refined[i] = weighted_quaternion_average(
            top_k_orientations[i], weights.astype(np.float32),
        )

    return refined
