"""Quality scoring and grain boundary detection for indexed EBSD maps.

Provides pixel-level quality flags based on indexing confidence,
FAISS distances, and anomaly detection for unknown phases.
"""
from __future__ import annotations

import numpy as np


class QualityScorer:
    """Quality scorer for embedding-indexed EBSD maps.

    Parameters
    ----------
    gb_confidence_threshold : float
        Confidence below this is flagged as grain boundary.
    anomaly_distance_threshold : float
        Inner-product distance below this is flagged as anomaly.
    """

    def __init__(
        self,
        gb_confidence_threshold: float = 0.2,
        anomaly_distance_threshold: float = 0.5,
    ) -> None:
        self.gb_confidence_threshold = gb_confidence_threshold
        self.anomaly_distance_threshold = anomaly_distance_threshold

    def detect_grain_boundaries(
        self,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """Detect grain boundary pixels via low confidence.

        Parameters
        ----------
        confidence : np.ndarray
            (N,) confidence values.

        Returns
        -------
        np.ndarray
            (N,) boolean mask, True = grain boundary.
        """
        return confidence < self.gb_confidence_threshold

    def detect_anomalies(
        self,
        distances: np.ndarray,
    ) -> np.ndarray:
        """Detect anomalous pixels (unknown phases, bad patterns).

        Parameters
        ----------
        distances : np.ndarray
            (N,) best-match inner product distances.

        Returns
        -------
        np.ndarray
            (N,) boolean mask, True = anomaly.
        """
        return distances < self.anomaly_distance_threshold

    def score(
        self,
        confidence: np.ndarray,
        distances: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run full quality scoring pipeline.

        Parameters
        ----------
        confidence : np.ndarray
            (N,) confidence values.
        distances : np.ndarray
            (N,) best-match distances.

        Returns
        -------
        dict
            Keys: grain_boundary_mask, anomaly_mask, quality_flags.
            quality_flags: 0=good, 1=grain_boundary, 2=anomaly, 3=both.
        """
        gb_mask = self.detect_grain_boundaries(confidence)
        anomaly_mask = self.detect_anomalies(distances)

        flags = np.zeros(len(confidence), dtype=np.int32)
        flags[gb_mask] |= 1
        flags[anomaly_mask] |= 2

        return {
            "grain_boundary_mask": gb_mask,
            "anomaly_mask": anomaly_mask,
            "quality_flags": flags,
        }
