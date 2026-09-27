"""PyTorch Dataset class for EBSD phase classification.

Provides ``EBSDPhaseDataset`` — a map-style Dataset that reads samples from
a :class:`TrainingStore`, applies augmentation and EDS dropout, and returns
tensors ready for the ``PhaseClassifier`` forward pass.

Key features:
- Lazy loading from HDF5 via :class:`TrainingStore`
- EDS dropout augmentation (randomly zeroes EDS for a fraction of samples
  even when EDS is available, forcing the model to work without EDS)
- Optional pattern / EDS augmentation via :class:`AugmentationConfig`
- Phase-balanced sampling weights for ``WeightedRandomSampler``
- Stratified train / val / test split preserving phase AND has_eds distribution
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ebsd_ai.config import DetectorInfo
from ebsd_ai.data.augmentation import (
    AugmentationConfig,
    augment_eds,
    augment_pattern,
)
from ebsd_ai.data.training_store import TrainingStore

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EBSDPhaseDataset(Dataset):
    """PyTorch Dataset wrapping a :class:`TrainingStore`.

    Each sample returns:

    - ``pattern`` : ``(1, H, H)`` float32 tensor (H = ``pattern_size``)
    - ``eds_input`` : ``(108,)`` float32 tensor (92 EDS + 15 detector + 1 has_eds flag)
    - ``phase_label`` : ``int64`` scalar

    Parameters
    ----------
    store : TrainingStore
        Training data backend.
    phase_names : list[str], optional
        Ordered phase names that define the label mapping.  If *None*,
        ``store.get_phase_names()`` is used (sorted alphabetically).
    augmentation : AugmentationConfig, optional
        Augmentation settings.  Pass ``None`` or set ``enabled=False`` to
        disable.
    eds_dropout_rate : float
        Fraction of samples where EDS is randomly zeroed even when present.
        Set to 0.0 to disable.
    indices : list[int], optional
        Subset of store indices to expose.  Used internally by split helpers.
    seed : int
        Random seed for augmentation and EDS dropout.
    """

    def __init__(
        self,
        store: TrainingStore,
        phase_names: list[str] | None = None,
        augmentation: AugmentationConfig | None = None,
        eds_dropout_rate: float = 0.3,
        indices: list[int] | None = None,
        seed: int = 42,
    ) -> None:
        self.store = store
        self.augmentation = augmentation
        self.eds_dropout_rate = eds_dropout_rate
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Build phase -> label mapping
        if phase_names is None:
            phase_names = store.get_phase_names()
        self.phase_names: list[str] = list(phase_names)
        self._phase_to_label: dict[str, int] = {
            name: i for i, name in enumerate(self.phase_names)
        }

        # Subset indices (for train/val/test splits)
        if indices is not None:
            self._indices = list(indices)
        else:
            self._indices = list(range(len(store)))

        # Pre-load lightweight metadata for sampling weights.
        # Uses get_metadata() which reads only HDF5 attributes (no
        # pattern/EDS arrays), avoiding expensive full-sample reads.
        self._labels: list[int] = []
        self._has_eds_flags: list[bool] = []
        for idx in self._indices:
            meta = store.get_metadata(idx)
            phase = meta["confirmed_phase"]
            label = self._phase_to_label.get(phase, -1)
            self._labels.append(label)
            self._has_eds_flags.append(meta["has_eds"])

    # -- PyTorch Dataset interface ------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a single sample as a dict of tensors.

        Returns
        -------
        dict
            ``pattern`` : ``(1, H, H)`` float32
            ``eds_input`` : ``(108,)`` float32
            ``phase_label`` : ``()`` int64
        """
        global_idx = self._indices[index]
        sample = self.store.get_sample(global_idx)

        pattern = sample["pattern_normalized"]  # (H, H) float32
        eds_vec = sample["eds_atomic_pct"]  # (92,) float32
        sample_has_eds = sample["has_eds"]
        detector_info: DetectorInfo = sample["detector_info"]
        phase = sample["confirmed_phase"]

        # Per-sample RNG derived from seed + index for reproducibility
        item_rng = np.random.default_rng(self.seed + index)

        # --- Augmentation (pattern + EDS) ---
        if self.augmentation is not None and self.augmentation.enabled:
            pattern = augment_pattern(pattern, self.augmentation, item_rng)
            eds_vec = augment_eds(eds_vec, self.augmentation, item_rng)

        # --- EDS dropout ---
        if sample_has_eds and self.eds_dropout_rate > 0:
            if item_rng.random() < self.eds_dropout_rate:
                eds_vec = np.zeros_like(eds_vec)
                sample_has_eds = False

        # --- Build tensors (sanitize non-finite values) ---
        pattern_f32 = pattern.astype(np.float32)
        np.nan_to_num(pattern_f32, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        pattern_tensor = torch.from_numpy(pattern_f32).unsqueeze(0)  # (1, H, H)

        # EDS input: [eds(92) + detector(15) + has_eds(1)] = 108
        det_encoded = detector_info.encode()  # (15,)
        has_eds_flag = np.array([1.0 if sample_has_eds else 0.0], dtype=np.float32)
        eds_input = np.concatenate([eds_vec, det_encoded, has_eds_flag])
        np.nan_to_num(eds_input, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        eds_input_tensor = torch.from_numpy(eds_input)

        # Phase label
        label = self._phase_to_label.get(phase, 0)
        label_tensor = torch.tensor(label, dtype=torch.int64)

        return {
            "pattern": pattern_tensor,
            "eds_input": eds_input_tensor,
            "phase_label": label_tensor,
        }

    # -- Sampling helpers ---------------------------------------------------

    @property
    def num_phases(self) -> int:
        """Number of distinct phases."""
        return len(self.phase_names)

    def get_phase_weights(self) -> torch.Tensor:
        """Compute inverse-frequency weights per sample for balanced sampling.

        Returns
        -------
        torch.Tensor
            Float64 tensor of length ``len(self)`` with per-sample weights.
        """
        label_arr = np.array(self._labels)
        n_classes = self.num_phases
        counts = np.zeros(n_classes, dtype=np.float64)
        for lbl in label_arr:
            if 0 <= lbl < n_classes:
                counts[lbl] += 1.0

        # Inverse frequency; avoid division by zero
        class_weights = np.where(counts > 0, 1.0 / counts, 0.0)
        sample_weights = np.array(
            [class_weights[lbl] if 0 <= lbl < n_classes else 0.0 for lbl in label_arr],
            dtype=np.float64,
        )
        return torch.from_numpy(sample_weights)

    def make_sampler(self, num_samples: int | None = None) -> WeightedRandomSampler:
        """Create a ``WeightedRandomSampler`` for phase-balanced training.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples per epoch.  Defaults to ``len(self)``.

        Returns
        -------
        WeightedRandomSampler
        """
        weights = self.get_phase_weights()
        if num_samples is None:
            num_samples = len(self)
        return WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=num_samples,
            replacement=True,
        )

    # -- Split helpers ------------------------------------------------------


def train_val_split(
    dataset: EBSDPhaseDataset,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[EBSDPhaseDataset, EBSDPhaseDataset]:
    """Split a dataset into train and validation subsets.

    The split is stratified by *both* phase label and has_eds flag so that
    every combination is represented in both subsets (when possible).

    Parameters
    ----------
    dataset : EBSDPhaseDataset
        The full dataset.
    val_fraction : float
        Fraction of data reserved for validation.
    seed : int
        Random seed.

    Returns
    -------
    tuple[EBSDPhaseDataset, EBSDPhaseDataset]
        ``(train_dataset, val_dataset)``
    """
    rng = np.random.default_rng(seed)

    # Group indices by (phase_label, has_eds)
    groups: dict[tuple[int, bool], list[int]] = {}
    for i, (lbl, has) in enumerate(zip(dataset._labels, dataset._has_eds_flags)):
        key = (lbl, has)
        groups.setdefault(key, []).append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for _key, idxs in sorted(groups.items()):
        rng.shuffle(idxs)  # type: ignore[arg-type]
        n_val = max(1, int(len(idxs) * val_fraction))
        # If only 1 sample, put in train (can't split further)
        if len(idxs) == 1:
            train_indices.extend(idxs)
        else:
            val_indices.extend(idxs[:n_val])
            train_indices.extend(idxs[n_val:])

    # Map local indices back to global store indices
    train_global = [dataset._indices[i] for i in train_indices]
    val_global = [dataset._indices[i] for i in val_indices]

    train_ds = EBSDPhaseDataset(
        store=dataset.store,
        phase_names=dataset.phase_names,
        augmentation=dataset.augmentation,
        eds_dropout_rate=dataset.eds_dropout_rate,
        indices=train_global,
        seed=dataset.seed,
    )

    val_ds = EBSDPhaseDataset(
        store=dataset.store,
        phase_names=dataset.phase_names,
        augmentation=None,  # No augmentation for validation
        eds_dropout_rate=0.0,  # No EDS dropout for validation
        indices=val_global,
        seed=dataset.seed,
    )

    return train_ds, val_ds


def train_val_test_split(
    dataset: EBSDPhaseDataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[EBSDPhaseDataset, EBSDPhaseDataset, EBSDPhaseDataset]:
    """Split a dataset into train, validation, and test subsets.

    The split is stratified by *both* phase label and has_eds flag.

    Parameters
    ----------
    dataset : EBSDPhaseDataset
        The full dataset.
    val_fraction : float
        Fraction of data reserved for validation.
    test_fraction : float
        Fraction of data reserved for testing.
    seed : int
        Random seed.

    Returns
    -------
    tuple[EBSDPhaseDataset, EBSDPhaseDataset, EBSDPhaseDataset]
        ``(train_dataset, val_dataset, test_dataset)``
    """
    rng = np.random.default_rng(seed)

    # Group indices by (phase_label, has_eds)
    groups: dict[tuple[int, bool], list[int]] = {}
    for i, (lbl, has) in enumerate(zip(dataset._labels, dataset._has_eds_flags)):
        key = (lbl, has)
        groups.setdefault(key, []).append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for _key, idxs in sorted(groups.items()):
        rng.shuffle(idxs)  # type: ignore[arg-type]
        n = len(idxs)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))

        if n == 1:
            # Only one sample: put in train
            train_indices.extend(idxs)
        elif n == 2:
            # Two samples: train + val
            train_indices.append(idxs[0])
            val_indices.append(idxs[1])
        else:
            test_indices.extend(idxs[:n_test])
            val_indices.extend(idxs[n_test : n_test + n_val])
            train_indices.extend(idxs[n_test + n_val :])

    train_global = [dataset._indices[i] for i in train_indices]
    val_global = [dataset._indices[i] for i in val_indices]
    test_global = [dataset._indices[i] for i in test_indices]

    train_ds = EBSDPhaseDataset(
        store=dataset.store,
        phase_names=dataset.phase_names,
        augmentation=dataset.augmentation,
        eds_dropout_rate=dataset.eds_dropout_rate,
        indices=train_global,
        seed=dataset.seed,
    )

    val_ds = EBSDPhaseDataset(
        store=dataset.store,
        phase_names=dataset.phase_names,
        augmentation=None,
        eds_dropout_rate=0.0,
        indices=val_global,
        seed=dataset.seed,
    )

    test_ds = EBSDPhaseDataset(
        store=dataset.store,
        phase_names=dataset.phase_names,
        augmentation=None,
        eds_dropout_rate=0.0,
        indices=test_global,
        seed=dataset.seed,
    )

    return train_ds, val_ds, test_ds
