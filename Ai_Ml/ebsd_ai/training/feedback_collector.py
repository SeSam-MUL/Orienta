"""Feedback loop collector for continuous ML improvement.

Collects indexing results from Hough, Dictionary, and Spherical methods
to build training data for embedding retraining. Tracks sample counts
and notifies when retraining threshold is reached.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects indexing results for ML retraining.

    Parameters
    ----------
    retrain_threshold : int
        Number of samples needed before suggesting retraining.
    """

    def __init__(self, retrain_threshold: int = 1000) -> None:
        self.retrain_threshold = retrain_threshold
        self._samples: List[dict] = []
        self._method_counts: Dict[str, int] = defaultdict(int)

    def add_indexing_result(
        self,
        patterns: np.ndarray,
        orientations: np.ndarray,
        confidence: np.ndarray,
        method: str,
        min_confidence: float = 0.0,
    ) -> int:
        """Add indexing results to the feedback pool.

        Parameters
        ----------
        patterns : np.ndarray
            (N, H, W) indexed patterns.
        orientations : np.ndarray
            (N, 4) resulting quaternions.
        confidence : np.ndarray
            (N,) confidence scores.
        method : str
            Indexing method name (hough, dictionary, spherical).
        min_confidence : float
            Minimum confidence to include a sample.

        Returns
        -------
        int
            Number of samples added.
        """
        mask = confidence >= min_confidence
        n_added = int(mask.sum())

        if n_added == 0:
            return 0

        self._samples.append({
            "patterns": patterns[mask],
            "orientations": orientations[mask],
            "confidence": confidence[mask],
            "method": method,
        })
        self._method_counts[method] += n_added

        logger.info(
            "Added %d samples from %s (total: %d)",
            n_added, method, self.count(),
        )

        return n_added

    def count(self) -> int:
        """Total number of collected samples."""
        return sum(self._method_counts.values())

    def count_by_method(self) -> Dict[str, int]:
        """Sample counts per indexing method."""
        return dict(self._method_counts)

    def should_retrain(self) -> bool:
        """Check if retraining threshold is reached."""
        return self.count() >= self.retrain_threshold

    def get_training_data(self) -> dict[str, np.ndarray]:
        """Combine all collected samples into training arrays.

        Returns
        -------
        dict
            Keys: patterns (N,H,W), orientations (N,4), confidence (N,).
        """
        if not self._samples:
            return {
                "patterns": np.empty((0, 128, 128), dtype=np.float32),
                "orientations": np.empty((0, 4), dtype=np.float32),
                "confidence": np.empty((0,), dtype=np.float32),
            }

        return {
            "patterns": np.concatenate([s["patterns"] for s in self._samples]),
            "orientations": np.concatenate([s["orientations"] for s in self._samples]),
            "confidence": np.concatenate([s["confidence"] for s in self._samples]),
        }

    def clear(self) -> None:
        """Clear all collected samples."""
        self._samples.clear()
        self._method_counts.clear()
