"""EDS confirmation module for resolving ambiguous phase identifications.

Uses EDS composition data to confirm, correct, or flag uncertain
ML-predicted phase IDs, especially for structurally similar phases
(e.g., BCC Fe vs BCC Cr).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EDSConfirmation:
    """EDS-based phase confirmation.

    Parameters
    ----------
    confidence_threshold : float
        Below this ML confidence, EDS is consulted.
    eds_match_threshold : float
        Below this EDS match score, result is uncertain.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        eds_match_threshold: float = 0.7,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.eds_match_threshold = eds_match_threshold

    @staticmethod
    def _match_score(
        composition: dict[str, float],
        reference: dict[str, float],
    ) -> float:
        """Compute match score between EDS composition and reference.

        Uses cosine similarity on composition vectors.

        Parameters
        ----------
        composition : dict
            Measured element fractions (e.g., {"Fe": 0.95, "Cr": 0.03}).
        reference : dict
            Expected element fractions for a phase.

        Returns
        -------
        float
            Match score in [0, 1].
        """
        all_elements = set(composition.keys()) | set(reference.keys())
        if not all_elements:
            return 0.0

        vec_c = np.array([composition.get(e, 0.0) for e in all_elements])
        vec_r = np.array([reference.get(e, 0.0) for e in all_elements])

        norm_c = np.linalg.norm(vec_c)
        norm_r = np.linalg.norm(vec_r)

        if norm_c < 1e-8 or norm_r < 1e-8:
            return 0.0

        return float(np.dot(vec_c, vec_r) / (norm_c * norm_r))

    def confirm(
        self,
        predicted_phase: str,
        eds_composition: dict[str, float],
        phase_library: dict[str, dict[str, float]],
        ml_confidence: float = 1.0,
    ) -> dict:
        """Confirm or correct a phase prediction using EDS data.

        Parameters
        ----------
        predicted_phase : str
            ML-predicted phase name.
        eds_composition : dict
            Measured element fractions.
        phase_library : dict
            Mapping of phase_name -> expected element fractions.
        ml_confidence : float
            ML prediction confidence.

        Returns
        -------
        dict
            Keys: outcome ("confirmed"/"corrected"/"uncertain"),
            phase (final phase name), scores (dict of match scores).
        """
        # Compute match scores for all phases
        scores = {}
        for phase_name, reference in phase_library.items():
            scores[phase_name] = self._match_score(eds_composition, reference)

        predicted_score = scores.get(predicted_phase, 0.0)

        # Find best EDS match
        best_phase = max(scores, key=scores.get)
        best_score = scores[best_phase]

        # Decision logic
        if ml_confidence >= self.confidence_threshold and predicted_score >= self.eds_match_threshold:
            return {
                "outcome": "confirmed",
                "phase": predicted_phase,
                "scores": scores,
            }

        if best_score >= self.eds_match_threshold and best_phase != predicted_phase:
            return {
                "outcome": "corrected",
                "phase": best_phase,
                "scores": scores,
            }

        return {
            "outcome": "uncertain",
            "phase": predicted_phase,
            "scores": scores,
        }
