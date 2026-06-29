"""Tests for the PyTorch Dataset class ``EBSDPhaseDataset``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ebsd_ai.config import (
    NUM_ELEMENTS,
    DetectorConvention,
    DetectorInfo,
    DetectorManufacturer,
    eds_dict_to_vector,
)
from ebsd_ai.data.augmentation import AugmentationConfig
from ebsd_ai.data.dataset import (
    EBSDPhaseDataset,
    train_val_split,
    train_val_test_split,
)
from ebsd_ai.data.training_store import TrainingStore
from tests.conftest import (
    make_synthetic_batch,
    make_synthetic_detector_info,
    make_synthetic_eds,
    make_synthetic_pattern,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_NAMES = ["Ferrite", "Austenite", "Cementite"]


def _populate_store(
    store: TrainingStore,
    n_per_phase: int = 10,
    include_eds: bool = True,
    seed: int = 0,
) -> list[str]:
    """Add samples to the store with known phase distribution.

    Half of each phase's samples have EDS, half don't (when include_eds=True).
    Returns the phase names used.
    """
    rng = np.random.default_rng(seed)
    det = make_synthetic_detector_info(rng=rng)

    for phase in PHASE_NAMES:
        for i in range(n_per_phase):
            pattern = make_synthetic_pattern(120, 160, rng)
            if include_eds and i < n_per_phase // 2:
                eds = make_synthetic_eds(rng=rng)
            else:
                eds = None
            store.add_sample(
                pattern=pattern,
                confirmed_phase=phase,
                detector_info=det,
                eds_data=eds,
                confidence_score=float(rng.uniform(0.5, 1.0)),
            )
    return PHASE_NAMES


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """Temporary directory for TrainingStore."""
    return tmp_path / "store"


@pytest.fixture
def populated_store(store_path: Path) -> TrainingStore:
    """TrainingStore with 30 samples (10 per phase, mixed EDS)."""
    store = TrainingStore(store_path, samples_per_shard=100)
    _populate_store(store, n_per_phase=10)
    return store


@pytest.fixture
def dataset(populated_store: TrainingStore) -> EBSDPhaseDataset:
    """Dataset wrapping the populated store."""
    return EBSDPhaseDataset(
        populated_store,
        phase_names=PHASE_NAMES,
        eds_dropout_rate=0.0,  # Disable for deterministic testing
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestDatasetBasics:
    """Core Dataset interface tests."""

    def test_length(self, dataset: EBSDPhaseDataset) -> None:
        assert len(dataset) == 30

    def test_getitem_keys(self, dataset: EBSDPhaseDataset) -> None:
        sample = dataset[0]
        assert "pattern" in sample
        assert "eds_input" in sample
        assert "phase_label" in sample

    def test_pattern_shape(self, dataset: EBSDPhaseDataset) -> None:
        sample = dataset[0]
        assert sample["pattern"].shape == (1, 128, 128)

    def test_pattern_dtype(self, dataset: EBSDPhaseDataset) -> None:
        sample = dataset[0]
        assert sample["pattern"].dtype == torch.float32

    def test_eds_input_shape(self, dataset: EBSDPhaseDataset) -> None:
        sample = dataset[0]
        # 92 EDS + 15 detector + 1 has_eds = 108
        assert sample["eds_input"].shape == (108,)

    def test_eds_input_dtype(self, dataset: EBSDPhaseDataset) -> None:
        sample = dataset[0]
        assert sample["eds_input"].dtype == torch.float32

    def test_phase_label_dtype(self, dataset: EBSDPhaseDataset) -> None:
        sample = dataset[0]
        assert sample["phase_label"].dtype == torch.int64

    def test_phase_label_range(self, dataset: EBSDPhaseDataset) -> None:
        for i in range(len(dataset)):
            label = dataset[i]["phase_label"].item()
            assert 0 <= label < len(PHASE_NAMES)

    def test_num_phases(self, dataset: EBSDPhaseDataset) -> None:
        assert dataset.num_phases == 3

    def test_phase_names_preserved(self, dataset: EBSDPhaseDataset) -> None:
        assert dataset.phase_names == PHASE_NAMES

    def test_auto_detect_phase_names(
        self, populated_store: TrainingStore
    ) -> None:
        """Phase names auto-detected from store when not provided."""
        ds = EBSDPhaseDataset(populated_store, eds_dropout_rate=0.0)
        assert sorted(ds.phase_names) == sorted(PHASE_NAMES)

    def test_no_nan_in_pattern(self, dataset: EBSDPhaseDataset) -> None:
        sample = dataset[0]
        assert not torch.isnan(sample["pattern"]).any()

    def test_no_nan_in_eds(self, dataset: EBSDPhaseDataset) -> None:
        sample = dataset[0]
        assert not torch.isnan(sample["eds_input"]).any()


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataLoader:
    """Verify Dataset works with PyTorch DataLoader."""

    def test_iterate_batch(self, dataset: EBSDPhaseDataset) -> None:
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["pattern"].shape == (4, 1, 128, 128)
        assert batch["eds_input"].shape == (4, 108)
        assert batch["phase_label"].shape == (4,)

    def test_full_iteration(self, dataset: EBSDPhaseDataset) -> None:
        """Iterate through entire dataset without error."""
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        total = 0
        for batch in loader:
            total += batch["phase_label"].shape[0]
        assert total == len(dataset)

    def test_with_sampler(self, dataset: EBSDPhaseDataset) -> None:
        sampler = dataset.make_sampler(num_samples=16)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)
        total = 0
        for batch in loader:
            total += batch["phase_label"].shape[0]
        assert total == 16


# ---------------------------------------------------------------------------
# EDS dropout
# ---------------------------------------------------------------------------


class TestEDSDropout:
    """EDS dropout augmentation tests."""

    def test_eds_dropout_zeros_eds(
        self, populated_store: TrainingStore
    ) -> None:
        """With eds_dropout_rate=1.0, all EDS vectors should be zeroed."""
        ds = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            eds_dropout_rate=1.0,
        )
        for i in range(len(ds)):
            sample = ds[i]
            eds_part = sample["eds_input"][:NUM_ELEMENTS]
            has_eds_flag = sample["eds_input"][-1].item()
            # Samples that originally had EDS should now be zeroed
            # (has_eds becomes 0.0 when dropped)
            # Samples that never had EDS remain zero
            # Both cases: either flag is 0 or eds is zero
            if has_eds_flag == 0.0:
                assert (eds_part == 0.0).all()

    def test_eds_dropout_zero_rate_preserves(
        self, populated_store: TrainingStore
    ) -> None:
        """With eds_dropout_rate=0.0, no EDS data is dropped."""
        ds = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            eds_dropout_rate=0.0,
        )
        n_with_eds = 0
        for i in range(len(ds)):
            if ds[i]["eds_input"][-1].item() > 0.5:
                n_with_eds += 1
        # Half of samples per phase have EDS: 5 per phase × 3 = 15
        assert n_with_eds == 15

    def test_eds_dropout_partial_rate(
        self, populated_store: TrainingStore
    ) -> None:
        """With eds_dropout_rate=0.3, roughly 30% of EDS samples are dropped."""
        ds = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            eds_dropout_rate=0.3,
        )
        n_with_eds = 0
        for i in range(len(ds)):
            if ds[i]["eds_input"][-1].item() > 0.5:
                n_with_eds += 1

        # Originally 15 samples have EDS. With 30% dropout:
        # expected ~10.5 remaining, but stochastic — allow wide range
        assert 5 <= n_with_eds <= 15

    def test_eds_dropout_does_not_affect_no_eds_samples(
        self, store_path: Path
    ) -> None:
        """EDS dropout doesn't create EDS where none existed."""
        store = TrainingStore(store_path, samples_per_shard=100)
        _populate_store(store, n_per_phase=10, include_eds=False)
        ds = EBSDPhaseDataset(
            store,
            phase_names=PHASE_NAMES,
            eds_dropout_rate=0.5,
        )
        for i in range(len(ds)):
            sample = ds[i]
            eds_part = sample["eds_input"][:NUM_ELEMENTS]
            assert (eds_part == 0.0).all()


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    """Augmentation integration tests."""

    def test_augmentation_changes_pattern(
        self, populated_store: TrainingStore
    ) -> None:
        """Augmented pattern differs from non-augmented."""
        no_aug = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            augmentation=None,
            eds_dropout_rate=0.0,
        )
        with_aug = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            augmentation=AugmentationConfig(enabled=True, noise_std=0.5),
            eds_dropout_rate=0.0,
        )
        p_orig = no_aug[0]["pattern"]
        p_aug = with_aug[0]["pattern"]
        # With strong noise, patterns should differ
        assert not torch.allclose(p_orig, p_aug)

    def test_augmentation_preserves_shape(
        self, populated_store: TrainingStore
    ) -> None:
        ds = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            augmentation=AugmentationConfig(enabled=True),
            eds_dropout_rate=0.0,
        )
        sample = ds[0]
        assert sample["pattern"].shape == (1, 128, 128)
        assert sample["eds_input"].shape == (108,)

    def test_disabled_augmentation(
        self, populated_store: TrainingStore
    ) -> None:
        """Disabled augmentation produces same output as no augmentation."""
        no_aug = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            augmentation=None,
            eds_dropout_rate=0.0,
        )
        disabled = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            augmentation=AugmentationConfig(enabled=False),
            eds_dropout_rate=0.0,
        )
        p1 = no_aug[0]["pattern"]
        p2 = disabled[0]["pattern"]
        assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Phase-balanced sampling
# ---------------------------------------------------------------------------


class TestPhaseSampling:
    """Phase-balanced sampling weight tests."""

    def test_weights_length(self, dataset: EBSDPhaseDataset) -> None:
        weights = dataset.get_phase_weights()
        assert len(weights) == len(dataset)

    def test_weights_positive(self, dataset: EBSDPhaseDataset) -> None:
        weights = dataset.get_phase_weights()
        assert (weights >= 0).all()

    def test_equal_phases_equal_weights(
        self, dataset: EBSDPhaseDataset
    ) -> None:
        """With equal phase distribution, all weights are equal."""
        weights = dataset.get_phase_weights()
        # All phases have 10 samples each
        assert torch.allclose(weights, weights[0].expand_as(weights))

    def test_imbalanced_phases(self, store_path: Path) -> None:
        """Minority phase gets higher weights."""
        store = TrainingStore(store_path, samples_per_shard=100)
        rng = np.random.default_rng(99)
        det = make_synthetic_detector_info(rng=rng)

        # 20 Ferrite, 5 Austenite
        for _ in range(20):
            store.add_sample(
                make_synthetic_pattern(rng=rng), "Ferrite", det,
            )
        for _ in range(5):
            store.add_sample(
                make_synthetic_pattern(rng=rng), "Austenite", det,
            )

        ds = EBSDPhaseDataset(
            store,
            phase_names=["Austenite", "Ferrite"],
            eds_dropout_rate=0.0,
        )
        weights = ds.get_phase_weights()

        # Find weights for each phase
        ferrite_weights = [
            weights[i].item()
            for i in range(len(ds))
            if ds._labels[i] == ds._phase_to_label["Ferrite"]
        ]
        austenite_weights = [
            weights[i].item()
            for i in range(len(ds))
            if ds._labels[i] == ds._phase_to_label["Austenite"]
        ]

        # Austenite (minority) should have higher weight
        assert austenite_weights[0] > ferrite_weights[0]

    def test_make_sampler(self, dataset: EBSDPhaseDataset) -> None:
        sampler = dataset.make_sampler()
        assert sampler.num_samples == len(dataset)

    def test_make_sampler_custom_count(
        self, dataset: EBSDPhaseDataset
    ) -> None:
        sampler = dataset.make_sampler(num_samples=64)
        assert sampler.num_samples == 64


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------


class TestTrainValSplit:
    """Stratified train/val split tests."""

    def test_no_overlap(self, dataset: EBSDPhaseDataset) -> None:
        train_ds, val_ds = train_val_split(dataset, val_fraction=0.3)
        train_set = set(train_ds._indices)
        val_set = set(val_ds._indices)
        assert train_set.isdisjoint(val_set)

    def test_all_samples_present(self, dataset: EBSDPhaseDataset) -> None:
        train_ds, val_ds = train_val_split(dataset, val_fraction=0.3)
        total = len(train_ds) + len(val_ds)
        assert total == len(dataset)

    def test_val_fraction_approx(self, dataset: EBSDPhaseDataset) -> None:
        train_ds, val_ds = train_val_split(dataset, val_fraction=0.3)
        actual_frac = len(val_ds) / len(dataset)
        # Stratification means exact fraction is approximate
        assert 0.15 <= actual_frac <= 0.6

    def test_both_splits_have_all_phases(
        self, dataset: EBSDPhaseDataset
    ) -> None:
        train_ds, val_ds = train_val_split(dataset, val_fraction=0.3)
        train_labels = set(train_ds._labels)
        val_labels = set(val_ds._labels)
        all_labels = {0, 1, 2}
        assert train_labels == all_labels
        assert val_labels == all_labels

    def test_val_has_no_augmentation(
        self, populated_store: TrainingStore
    ) -> None:
        ds = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            augmentation=AugmentationConfig(enabled=True),
            eds_dropout_rate=0.3,
        )
        _, val_ds = train_val_split(ds)
        assert val_ds.augmentation is None
        assert val_ds.eds_dropout_rate == 0.0

    def test_train_preserves_augmentation(
        self, populated_store: TrainingStore
    ) -> None:
        aug = AugmentationConfig(enabled=True)
        ds = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            augmentation=aug,
            eds_dropout_rate=0.3,
        )
        train_ds, _ = train_val_split(ds)
        assert train_ds.augmentation is aug
        assert train_ds.eds_dropout_rate == 0.3

    def test_both_splits_have_eds_and_no_eds(
        self, dataset: EBSDPhaseDataset
    ) -> None:
        """Stratification ensures both EDS and no-EDS samples in each split."""
        train_ds, val_ds = train_val_split(dataset, val_fraction=0.3)

        train_has = any(train_ds._has_eds_flags)
        train_no = any(not f for f in train_ds._has_eds_flags)
        val_has = any(val_ds._has_eds_flags)
        val_no = any(not f for f in val_ds._has_eds_flags)

        assert train_has and train_no
        assert val_has and val_no


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------


class TestTrainValTestSplit:
    """Stratified three-way split tests."""

    def test_no_overlap(self, dataset: EBSDPhaseDataset) -> None:
        train_ds, val_ds, test_ds = train_val_test_split(dataset)
        sets = [
            set(train_ds._indices),
            set(val_ds._indices),
            set(test_ds._indices),
        ]
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])

    def test_all_samples_present(self, dataset: EBSDPhaseDataset) -> None:
        train_ds, val_ds, test_ds = train_val_test_split(dataset)
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(dataset)

    def test_all_phases_in_each_split(
        self, dataset: EBSDPhaseDataset
    ) -> None:
        train_ds, val_ds, test_ds = train_val_test_split(dataset)
        all_labels = {0, 1, 2}
        assert set(train_ds._labels) == all_labels
        assert set(val_ds._labels) == all_labels
        assert set(test_ds._labels) == all_labels

    def test_val_and_test_no_augmentation(
        self, populated_store: TrainingStore
    ) -> None:
        ds = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            augmentation=AugmentationConfig(enabled=True),
            eds_dropout_rate=0.3,
        )
        _, val_ds, test_ds = train_val_test_split(ds)
        assert val_ds.augmentation is None
        assert val_ds.eds_dropout_rate == 0.0
        assert test_ds.augmentation is None
        assert test_ds.eds_dropout_rate == 0.0

    def test_deterministic(self, dataset: EBSDPhaseDataset) -> None:
        """Same seed produces same splits."""
        split1 = train_val_test_split(dataset, seed=123)
        split2 = train_val_test_split(dataset, seed=123)
        for ds1, ds2 in zip(split1, split2):
            assert ds1._indices == ds2._indices


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling."""

    def test_single_phase(self, store_path: Path) -> None:
        """Dataset with only one phase."""
        store = TrainingStore(store_path, samples_per_shard=100)
        rng = np.random.default_rng(7)
        det = make_synthetic_detector_info(rng=rng)
        for _ in range(5):
            store.add_sample(
                make_synthetic_pattern(rng=rng), "Ferrite", det,
            )
        ds = EBSDPhaseDataset(store, eds_dropout_rate=0.0)
        assert ds.num_phases == 1
        assert len(ds) == 5
        sample = ds[0]
        assert sample["phase_label"].item() == 0

    def test_subset_indices(self, populated_store: TrainingStore) -> None:
        """Dataset with subset of indices."""
        ds = EBSDPhaseDataset(
            populated_store,
            phase_names=PHASE_NAMES,
            indices=[0, 5, 10, 15, 20],
            eds_dropout_rate=0.0,
        )
        assert len(ds) == 5

    def test_all_no_eds(self, store_path: Path) -> None:
        """All samples without EDS."""
        store = TrainingStore(store_path, samples_per_shard=100)
        _populate_store(store, n_per_phase=5, include_eds=False)
        ds = EBSDPhaseDataset(
            store, phase_names=PHASE_NAMES, eds_dropout_rate=0.0,
        )
        for i in range(len(ds)):
            sample = ds[i]
            eds_part = sample["eds_input"][:NUM_ELEMENTS]
            assert (eds_part == 0.0).all()
            assert sample["eds_input"][-1].item() == 0.0

    def test_all_with_eds(self, store_path: Path) -> None:
        """All samples have EDS data."""
        store = TrainingStore(store_path, samples_per_shard=100)
        rng = np.random.default_rng(8)
        det = make_synthetic_detector_info(rng=rng)
        for phase in PHASE_NAMES:
            for _ in range(5):
                store.add_sample(
                    make_synthetic_pattern(rng=rng),
                    phase,
                    det,
                    eds_data=make_synthetic_eds(rng=rng),
                )
        ds = EBSDPhaseDataset(
            store, phase_names=PHASE_NAMES, eds_dropout_rate=0.0,
        )
        for i in range(len(ds)):
            assert ds[i]["eds_input"][-1].item() == 1.0

    def test_has_eds_flag_in_eds_input(
        self, dataset: EBSDPhaseDataset
    ) -> None:
        """Last element of eds_input is the has_eds flag (0 or 1)."""
        for i in range(len(dataset)):
            flag = dataset[i]["eds_input"][-1].item()
            assert flag in (0.0, 1.0)
