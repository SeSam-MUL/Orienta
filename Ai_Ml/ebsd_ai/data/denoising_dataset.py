"""PyTorch Dataset for denoising training pairs.

Filters a TrainingStore for ``pair_type == "denoising"`` samples and
returns (input, target, detector_encoding) tensors suitable for the
PatternEnhancer U-Net.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ebsd_ai.data.training_store import TrainingStore


class DenoisingDataset(Dataset):  # type: ignore[type-arg]
    """Dataset that yields denoising pairs from a TrainingStore.

    Only samples with ``pair_type == "denoising"`` are included.

    Parameters
    ----------
    store : TrainingStore
        The backing HDF5 store.
    augment : bool
        Whether to apply random flips (applied identically to input and
        target).  Brightness/noise augmentation is applied to input only.
    """

    def __init__(
        self,
        store: TrainingStore,
        augment: bool = False,
    ) -> None:
        self.store = store
        self.augment = augment
        # Pre-scan to find denoising pair indices
        self._indices: list[int] = []
        for i, meta in enumerate(store.iter_metadata_v2()):
            if meta.get("pair_type", "classification") == "denoising":
                self._indices.append(i)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.store.get_sample(self._indices[idx])

        inp = sample["pattern_normalized"].astype(np.float32)
        target = sample["pattern_target"].astype(np.float32)

        # Augmentation: random flips applied to both input and target
        if self.augment:
            if np.random.random() > 0.5:
                inp = np.flip(inp, axis=0).copy()
                target = np.flip(target, axis=0).copy()
            if np.random.random() > 0.5:
                inp = np.flip(inp, axis=1).copy()
                target = np.flip(target, axis=1).copy()

        # Detector encoding
        det_enc = sample["detector_info"].encode()

        return {
            "input": torch.from_numpy(inp).unsqueeze(0),
            "target": torch.from_numpy(target).unsqueeze(0),
            "detector_encoding": torch.from_numpy(det_enc),
        }
