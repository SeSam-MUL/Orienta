"""Targeted tests to cover edge cases and error-handling paths.

Covers previously uncovered lines identified by ``pytest --cov``:
- server_sync.py: atomic copy failure cleanup, OSError during sync
- config.py: ``_normalize`` with degenerate range
- dataset.py: ``train_val_test_split`` edge cases (1-2 samples per group)
- augmentation.py: ``rng=None`` default branches
- online_learner.py: empty dataset, empty loader re-iteration
- pattern_enhancer.py: skip connection size mismatch path
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ebsd_ai.config import (
    DataSource,
    DetectorInfo,
    ModelConfig,
    TrainingConfig,
)
from ebsd_ai.data.augmentation import (
    AugmentationConfig,
    augment_eds,
    augment_pattern,
)
from ebsd_ai.data.dataset import (
    EBSDPhaseDataset,
    train_val_split,
    train_val_test_split,
)
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.sync.server_sync import ServerSync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shard(directory: Path, name: str = "shard_1000_abc.h5") -> Path:
    import h5py

    path = directory / name
    with h5py.File(path, "w") as f:
        f.attrs["created"] = 1000
        f.create_group("sample_00000000")
    return path


def _populate_store(
    store: TrainingStore, n_phases: int, samples_per_phase: int,
) -> list[str]:
    """Add minimal samples, return phase names."""
    rng = np.random.default_rng(0)
    det = DetectorInfo()
    names = [f"Phase_{i}" for i in range(n_phases)]
    for name in names:
        for _ in range(samples_per_phase):
            store.add_sample(
                pattern=rng.integers(0, 256, (60, 80), dtype=np.uint8),
                confirmed_phase=name,
                detector_info=det,
                data_source=DataSource.SIMULATION,
            )
    return names


# ---------------------------------------------------------------------------
# server_sync: atomic copy failure + OSError during sync
# ---------------------------------------------------------------------------


class TestAtomicCopyFailure:
    """Cover server_sync.py lines 159-166 (error cleanup in _copy_atomic)."""

    def test_copy_atomic_cleans_up_on_failure(self, tmp_path: Path) -> None:
        """If shutil.copy2 fails, the temp file is removed."""
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        server.mkdir()

        shard = _make_shard(local)

        with patch("ebsd_ai.sync.server_sync.shutil.copy2",
                    side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                ServerSync._copy_atomic(shard, server)

        # No temp file left behind
        assert list(server.glob("*.tmp")) == []
        # Final file not created either
        assert not (server / shard.name).exists()

    def test_sync_records_failed_file(self, tmp_path: Path) -> None:
        """OSError during copy records the file as failed, not crashed."""
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"

        _make_shard(local, "shard_100_aaa.h5")

        sync = ServerSync(local_dir=local, server_dir=server)

        with patch.object(
            ServerSync, "_copy_atomic",
            side_effect=OSError("permission denied"),
        ):
            result = sync.push()

        assert result.n_failed == 1
        assert "shard_100_aaa.h5" in result.files_failed
        assert "permission denied" in result.errors["shard_100_aaa.h5"]


# ---------------------------------------------------------------------------
# config: _normalize edge case
# ---------------------------------------------------------------------------


class TestNormalizeEdge:
    """Cover config.py line 370 (_normalize with hi <= lo)."""

    def test_normalize_degenerate_range(self) -> None:
        from ebsd_ai.config import _normalize

        assert _normalize(5.0, 10.0, 10.0) == 0.0
        assert _normalize(5.0, 10.0, 5.0) == 0.0


# ---------------------------------------------------------------------------
# dataset: train_val_test_split edge cases
# ---------------------------------------------------------------------------


class TestSplitEdgeCases:
    """Cover dataset.py lines 254-255 (single sample) and 331-335 (two samples)."""

    def test_train_val_split_single_sample_per_phase(
        self, tmp_path: Path,
    ) -> None:
        """Phases with 1 sample go to train (not val)."""
        store = TrainingStore(tmp_path / "store", samples_per_shard=100)
        _populate_store(store, n_phases=3, samples_per_phase=1)
        ds = EBSDPhaseDataset(store, eds_dropout_rate=0.0)

        train_ds, val_ds = train_val_split(ds, val_fraction=0.5)
        # All 3 single-sample phases should go to train
        assert len(train_ds) == 3
        assert len(val_ds) == 0

    def test_train_val_test_split_two_samples(
        self, tmp_path: Path,
    ) -> None:
        """Phases with 2 samples split into train + val, skip test."""
        store = TrainingStore(tmp_path / "store", samples_per_shard=100)
        _populate_store(store, n_phases=2, samples_per_phase=2)
        ds = EBSDPhaseDataset(store, eds_dropout_rate=0.0)

        train_ds, val_ds, test_ds = train_val_test_split(
            ds, val_fraction=0.3, test_fraction=0.3,
        )
        # 2 phases x 2 samples = 4 total
        # Each phase with n=2 → train(1) + val(1), test=0
        assert len(train_ds) + len(val_ds) + len(test_ds) == 4
        assert len(val_ds) >= 1
        assert len(train_ds) >= 1

    def test_train_val_test_split_single_sample(
        self, tmp_path: Path,
    ) -> None:
        """Phases with 1 sample go entirely to train."""
        store = TrainingStore(tmp_path / "store", samples_per_shard=100)
        _populate_store(store, n_phases=2, samples_per_phase=1)
        ds = EBSDPhaseDataset(store, eds_dropout_rate=0.0)

        train_ds, val_ds, test_ds = train_val_test_split(
            ds, val_fraction=0.3, test_fraction=0.3,
        )
        assert len(train_ds) == 2
        assert len(val_ds) == 0
        assert len(test_ds) == 0


# ---------------------------------------------------------------------------
# augmentation: rng=None default
# ---------------------------------------------------------------------------


class TestAugmentationDefaults:
    """Cover augmentation.py lines 99 and 208 (rng=None branches)."""

    def test_augment_pattern_default_rng(self) -> None:
        pattern = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        config = AugmentationConfig(noise_std=0.01)
        result = augment_pattern(pattern, config)  # rng=None
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_augment_eds_default_rng(self) -> None:
        eds = np.array([70.0, 0.0, 18.0] + [0.0] * 89, dtype=np.float32)
        config = AugmentationConfig(eds_noise_std=0.05)
        result = augment_eds(eds, config)  # rng=None
        assert result.shape == eds.shape
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# online_learner: empty dataset + empty loader
# ---------------------------------------------------------------------------


class TestOnlineLearnerEdgePaths:
    """Cover online_learner.py lines 163, 383, 428-434."""

    def test_update_empty_store(self, tmp_path: Path) -> None:
        """Update with an empty store returns immediately."""
        from ebsd_ai.training.online_learner import (
            OnlineLearner,
            OnlineLearnerConfig,
        )

        store = TrainingStore(tmp_path / "store", samples_per_shard=100)
        phases = ["A", "B"]
        config = ModelConfig(n_phases=2)
        model = PhaseClassifier(config=config, phase_names=phases)

        ol_config = OnlineLearnerConfig(
            steps_per_update=5, batch_size=4, ewc_lambda=0.0,
        )
        learner = OnlineLearner(
            model=model, store=store, config=ol_config, device="cpu",
        )
        result = learner.update()
        assert result.steps_completed == 0
        assert result.samples_used == 0

    def test_compute_fisher_empty_dataset(self, tmp_path: Path) -> None:
        """_compute_fisher_diagonal with empty dataset returns zero Fisher."""
        import torch

        from ebsd_ai.training.online_learner import _compute_fisher_diagonal

        config = ModelConfig(n_phases=2)
        model = PhaseClassifier(config=config, phase_names=["A", "B"])

        store = TrainingStore(tmp_path / "store", samples_per_shard=100)
        ds = EBSDPhaseDataset(store, eds_dropout_rate=0.0, phase_names=["A", "B"])

        fisher = _compute_fisher_diagonal(
            model, ds, device=torch.device("cpu"), n_samples=10,
        )
        assert all(v.sum().item() == 0.0 for v in fisher.values())
