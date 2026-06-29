"""Benchmarking framework for EBSD embedding indexing.

Provides metrics computation, throughput measurement, and comparison
tables for evaluating embedding-based indexing against Dictionary,
Hough, and Spherical methods.
"""
from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np


def compute_metrics(angular_errors: np.ndarray) -> dict[str, float]:
    """Compute standard indexing quality metrics.

    Parameters
    ----------
    angular_errors : np.ndarray
        (N,) angular errors in degrees.

    Returns
    -------
    dict
        Keys: mae, median, p95, max, std.
    """
    return {
        "mae": float(np.mean(angular_errors)),
        "median": float(np.median(angular_errors)),
        "p95": float(np.percentile(angular_errors, 95)),
        "max": float(np.max(angular_errors)),
        "std": float(np.std(angular_errors)),
    }


def measure_throughput(
    fn: Callable,
    patterns: np.ndarray,
    n_warmup: int = 1,
) -> float:
    """Measure indexing throughput in patterns/second.

    Parameters
    ----------
    fn : callable
        Function that takes patterns array and returns results.
    patterns : np.ndarray
        (N, H, W) test patterns.
    n_warmup : int
        Number of warmup runs.

    Returns
    -------
    float
        Throughput in patterns/second.
    """
    # Warmup
    for _ in range(n_warmup):
        fn(patterns[:1])

    start = time.perf_counter()
    fn(patterns)
    elapsed = time.perf_counter() - start

    return len(patterns) / max(elapsed, 1e-9)


class ComparisonTable:
    """Table for comparing indexing methods."""

    def __init__(self) -> None:
        self._results: Dict[str, dict] = {}

    def add_result(self, method: str, metrics: dict) -> None:
        """Add metrics for a method."""
        self._results[method] = metrics

    def to_dict(self) -> Dict[str, dict]:
        """Get all results as dict."""
        return dict(self._results)

    def to_markdown(self) -> str:
        """Render comparison as markdown table."""
        if not self._results:
            return ""

        methods = list(self._results.keys())
        all_keys = set()
        for m in self._results.values():
            all_keys.update(m.keys())
        keys = sorted(all_keys)

        header = "| Method | " + " | ".join(keys) + " |"
        sep = "|---" * (len(keys) + 1) + "|"
        rows = []
        for method in methods:
            vals = [f"{self._results[method].get(k, '-')}" for k in keys]
            rows.append(f"| {method} | " + " | ".join(vals) + " |")

        return "\n".join([header, sep] + rows)


class BenchmarkRunner:
    """Orchestrates benchmarking runs.

    Generates test datasets and runs indexing methods for comparison.
    """

    def __init__(self) -> None:
        self.table = ComparisonTable()

    def generate_test_dataset(
        self,
        n: int = 100,
        h: int = 128,
        w: int = 128,
        noise_level: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Generate a synthetic test dataset.

        Parameters
        ----------
        n : int
            Number of patterns.
        h, w : int
            Pattern dimensions.
        noise_level : float
            Gaussian noise sigma.

        Returns
        -------
        dict
            Keys: patterns, orientations, phase_ids.
        """
        patterns = np.random.rand(n, h, w).astype(np.float32)
        if noise_level > 0:
            patterns += np.random.randn(n, h, w).astype(np.float32) * noise_level

        orientations = np.random.randn(n, 4).astype(np.float32)
        orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
        phase_ids = np.zeros(n, dtype=np.int64)

        return {
            "patterns": patterns,
            "orientations": orientations,
            "phase_ids": phase_ids,
        }
