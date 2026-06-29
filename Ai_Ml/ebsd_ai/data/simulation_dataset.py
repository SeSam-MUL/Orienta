"""PyTorch Dataset for simulated EBSD patterns with orientations.

Wraps pre-computed or on-the-fly simulated patterns with their
corresponding crystal orientations and phase IDs.  Supports optional
physics-based augmentation for contrastive embedding training.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    """Dataset of simulated EBSD patterns with orientations.

    Parameters
    ----------
    patterns : np.ndarray
        (N, H, W) float32 simulated patterns.
    orientations : np.ndarray
        (N, 4) unit quaternions (scalar-first).
    phase_ids : np.ndarray
        (N,) integer phase identifiers.
    augmentation : callable or None
        Optional augmentation function applied to each pattern.
        Expected signature: ``aug(pattern_2d) -> pattern_2d``.
    """

    def __init__(
        self,
        patterns: np.ndarray,
        orientations: np.ndarray,
        phase_ids: np.ndarray,
        augmentation: Any | None = None,
    ) -> None:
        assert len(patterns) == len(orientations) == len(phase_ids)
        self.patterns = patterns.astype(np.float32)
        self.orientations = orientations.astype(np.float32)
        self.phase_ids = phase_ids.astype(np.int64)
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.patterns)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pattern = self.patterns[idx].copy()

        if self.augmentation is not None:
            pattern = self.augmentation(pattern)

        # Add channel dimension: (H, W) -> (1, H, W)
        pattern_t = torch.from_numpy(pattern).unsqueeze(0).float()
        orientation_t = torch.from_numpy(self.orientations[idx]).float()
        phase_id_t = torch.tensor(self.phase_ids[idx], dtype=torch.long)

        return {
            "pattern": pattern_t,
            "orientation": orientation_t,
            "phase_id": phase_id_t,
        }
