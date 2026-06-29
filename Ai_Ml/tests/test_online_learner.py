"""Tests for the OnlineLearner continual learning module.

Tests cover:
- OnlineLearnerConfig serialization
- OnlineUpdateResult defaults
- OnlineLearner initialization and device handling
- Basic update on synthetic data (loss decreases or stays finite)
- Phase expansion when new phases appear in the store
- Experience replay (mixed old/new sampling)
- EWC penalty computation and integration
- Checkpoint saving after update
- Multiple sequential updates
- Progress callback invocation
- Edge cases: empty store, single sample, no new indices
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ebsd_ai.config import DataSource, DetectorInfo, ModelConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.training.online_learner import (
    OnlineLearner,
    OnlineLearnerConfig,
    OnlineUpdateResult,
    _compute_fisher_diagonal,
    _ewc_penalty,
)
from tests.conftest import make_synthetic_batch, make_synthetic_detector_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_store(
    store: TrainingStore,
    n_samples: int = 20,
    n_phases: int = 3,
    include_eds: bool = True,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Add synthetic samples to a store and return phase names."""
    if rng is None:
        rng = np.random.default_rng(42)

    batch = make_synthetic_batch(
        n_samples=n_samples,
        n_phases=n_phases,
        include_eds=include_eds,
        rng=rng,
    )
    confirmed_phases = [
        batch["phase_names"][pid] for pid in batch["phase_ids"]
    ]
    store.add_batch(
        patterns=batch["patterns"],
        confirmed_phases=confirmed_phases,
        detector_info=batch["detector_info"],
        orientations=batch["orientations"],
        confidence_scores=batch["confidence_scores"],
        eds_data=batch["eds_data"],
        data_source=DataSource.AUTO_INDEXED,
        source_file="test",
    )
    return batch["phase_names"]


def _make_model(phase_names: list[str]) -> PhaseClassifier:
    """Create a small PhaseClassifier for testing."""
    config = ModelConfig(
        n_phases=len(phase_names),
        pattern_feature_dim=32,
        eds_feature_dim=16,
        fused_feature_dim=32,
        dropout=0.0,
    )
    return PhaseClassifier(config=config, phase_names=phase_names)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestOnlineLearnerConfig:
    """Tests for OnlineLearnerConfig."""

    def test_defaults(self) -> None:
        """Default config has reasonable values."""
        cfg = OnlineLearnerConfig()
        assert cfg.learning_rate == 1e-4
        assert cfg.steps_per_update == 50
        assert cfg.batch_size == 32
        assert cfg.replay_fraction == 0.5
        assert cfg.ewc_lambda == 0.0
        assert cfg.eds_dropout_rate == 0.3
        assert cfg.max_samples_per_update == 5000

    def test_to_dict_roundtrip(self) -> None:
        """Config survives serialization roundtrip."""
        cfg = OnlineLearnerConfig(
            learning_rate=5e-5,
            steps_per_update=100,
            ewc_lambda=0.5,
        )
        d = cfg.to_dict()
        restored = OnlineLearnerConfig.from_dict(d)
        assert restored.learning_rate == cfg.learning_rate
        assert restored.steps_per_update == cfg.steps_per_update
        assert restored.ewc_lambda == cfg.ewc_lambda


class TestOnlineUpdateResult:
    """Tests for OnlineUpdateResult."""

    def test_defaults(self) -> None:
        """Default result is empty."""
        result = OnlineUpdateResult()
        assert result.steps_completed == 0
        assert result.mean_loss == 0.0
        assert result.new_phases_added == []
        assert result.samples_used == 0


# ---------------------------------------------------------------------------
# OnlineLearner init tests
# ---------------------------------------------------------------------------


class TestOnlineLearnerInit:
    """Tests for OnlineLearner initialization."""

    def test_basic_init(self, tmp_path: Path) -> None:
        """OnlineLearner initializes without error."""
        store = TrainingStore(tmp_path / "store")
        phase_names = ["Alpha", "Beta"]
        model = _make_model(phase_names)
        learner = OnlineLearner(
            model=model,
            store=store,
            device="cpu",
        )
        assert learner.update_count == 0
        assert learner.device == torch.device("cpu")

    def test_auto_device(self, tmp_path: Path) -> None:
        """Auto device selection works (CPU in test env)."""
        store = TrainingStore(tmp_path / "store")
        model = _make_model(["A", "B"])
        learner = OnlineLearner(model=model, store=store, device="auto")
        assert learner.device.type in ("cpu", "cuda")

    def test_custom_config(self, tmp_path: Path) -> None:
        """Custom config is applied."""
        store = TrainingStore(tmp_path / "store")
        model = _make_model(["A", "B"])
        cfg = OnlineLearnerConfig(learning_rate=1e-5, steps_per_update=10)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        assert learner.config.learning_rate == 1e-5
        assert learner.config.steps_per_update == 10


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------


class TestOnlineLearnerUpdate:
    """Tests for the update() method."""

    def test_empty_store_returns_empty_result(self, tmp_path: Path) -> None:
        """Update on empty store returns zero-step result."""
        store = TrainingStore(tmp_path / "store")
        model = _make_model(["A", "B"])
        learner = OnlineLearner(model=model, store=store, device="cpu")
        result = learner.update()
        assert result.steps_completed == 0
        assert result.mean_loss == 0.0

    def test_basic_update(self, tmp_path: Path) -> None:
        """Update runs and produces finite loss."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=30, n_phases=3)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(steps_per_update=5, batch_size=8)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        result = learner.update()

        assert result.steps_completed == 5
        assert result.mean_loss > 0
        assert np.isfinite(result.mean_loss)
        assert result.samples_used > 0

    def test_update_increments_count(self, tmp_path: Path) -> None:
        """Each update increments the counter."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=2)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=8)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        learner.update()
        assert learner.update_count == 1
        learner.update()
        assert learner.update_count == 2

    def test_update_with_new_indices(self, tmp_path: Path) -> None:
        """Update with explicit new_indices runs experience replay."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=30, n_phases=3)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(
            steps_per_update=5, batch_size=8, replay_fraction=0.5
        )
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        # Pretend samples 20-29 are new
        result = learner.update(new_indices=list(range(20, 30)))

        assert result.steps_completed == 5
        assert result.mean_loss > 0

    def test_multiple_updates_no_crash(self, tmp_path: Path) -> None:
        """Multiple sequential updates don't crash."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=30, n_phases=3)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=8)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )

        losses = []
        for _ in range(3):
            result = learner.update()
            losses.append(result.mean_loss)

        # All losses should be finite
        assert all(np.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# Phase expansion tests
# ---------------------------------------------------------------------------


class TestPhaseExpansion:
    """Tests for automatic phase expansion."""

    def test_new_phase_added(self, tmp_path: Path) -> None:
        """Model expands when store has phases the model doesn't know."""
        store = TrainingStore(tmp_path / "store")
        # Start with 2 phases in the model
        _populate_store(store, n_samples=20, n_phases=2)
        model = _make_model(["Phase_0", "Phase_1"])
        assert model.n_phases == 2

        # Add a new phase to the store
        det = make_synthetic_detector_info()
        rng = np.random.default_rng(99)
        store.add_sample(
            pattern=rng.integers(0, 256, (60, 80), dtype=np.uint8),
            confirmed_phase="Phase_2",
            detector_info=det,
        )

        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=8)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        result = learner.update()

        assert "Phase_2" in result.new_phases_added
        assert model.n_phases == 3
        assert "Phase_2" in model.phase_names

    def test_no_duplicate_phases(self, tmp_path: Path) -> None:
        """Existing phases are not duplicated on update."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=3)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=8)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        result = learner.update()

        assert result.new_phases_added == []
        assert model.n_phases == 3


# ---------------------------------------------------------------------------
# EWC tests
# ---------------------------------------------------------------------------


class TestEWC:
    """Tests for Elastic Weight Consolidation."""

    def test_compute_fisher(self, tmp_path: Path) -> None:
        """Fisher diagonal computation produces non-negative values."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=3)
        model = _make_model(phase_names)
        dataset = EBSDPhaseDataset(
            store=store, phase_names=phase_names, eds_dropout_rate=0.0
        )

        fisher = _compute_fisher_diagonal(
            model, dataset, torch.device("cpu"), n_samples=10
        )

        assert len(fisher) > 0
        for name, values in fisher.items():
            assert (values >= 0).all(), f"Negative Fisher for {name}"

    def test_ewc_penalty_zero_at_start(self, tmp_path: Path) -> None:
        """EWC penalty is zero when model hasn't changed."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=3)
        model = _make_model(phase_names)
        dataset = EBSDPhaseDataset(
            store=store, phase_names=phase_names, eds_dropout_rate=0.0
        )

        fisher = _compute_fisher_diagonal(
            model, dataset, torch.device("cpu"), n_samples=10
        )
        old_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        penalty = _ewc_penalty(model, fisher, old_params)
        assert penalty.item() == pytest.approx(0.0, abs=1e-7)

    def test_ewc_penalty_nonzero_after_change(self, tmp_path: Path) -> None:
        """EWC penalty is positive after model parameters change."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=3)
        model = _make_model(phase_names)
        dataset = EBSDPhaseDataset(
            store=store, phase_names=phase_names, eds_dropout_rate=0.0
        )

        fisher = _compute_fisher_diagonal(
            model, dataset, torch.device("cpu"), n_samples=10
        )
        old_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        # Perturb model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        penalty = _ewc_penalty(model, fisher, old_params)
        assert penalty.item() > 0

    def test_update_with_ewc(self, tmp_path: Path) -> None:
        """Update with EWC enabled runs without error."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=30, n_phases=3)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(
            steps_per_update=5,
            batch_size=8,
            ewc_lambda=0.5,
        )
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        result = learner.update()

        assert result.steps_completed == 5
        assert np.isfinite(result.mean_loss)

    def test_ewc_state_updated_after_each_update(
        self, tmp_path: Path
    ) -> None:
        """EWC anchor is refreshed after each update."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=30, n_phases=3)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(
            steps_per_update=3, batch_size=8, ewc_lambda=0.5
        )
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )

        learner.update()
        fisher_1 = learner._fisher
        assert fisher_1 is not None

        learner.update()
        fisher_2 = learner._fisher
        assert fisher_2 is not None

        # Fisher estimates should differ (model changed between updates)
        # Check at least one parameter differs
        any_diff = False
        for name in fisher_1:
            if name in fisher_2:
                if not torch.allclose(fisher_1[name], fisher_2[name]):
                    any_diff = True
                    break
        assert any_diff, "Fisher should change between updates"


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for checkpoint saving after online updates."""

    def test_checkpoint_saved(self, tmp_path: Path) -> None:
        """Checkpoint file is created after update."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=2)
        model = _make_model(phase_names)

        ckpt_dir = tmp_path / "checkpoints"
        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=8)
        learner = OnlineLearner(
            model=model,
            store=store,
            config=cfg,
            device="cpu",
            checkpoint_dir=ckpt_dir,
        )
        learner.update()

        assert (ckpt_dir / "online_latest.pt").exists()

    def test_checkpoint_loadable(self, tmp_path: Path) -> None:
        """Saved checkpoint can be loaded."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=2)
        model = _make_model(phase_names)

        ckpt_dir = tmp_path / "checkpoints"
        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=8)
        learner = OnlineLearner(
            model=model,
            store=store,
            config=cfg,
            device="cpu",
            checkpoint_dir=ckpt_dir,
        )
        learner.update()

        # Load the checkpoint
        ckpt = torch.load(
            ckpt_dir / "online_latest.pt",
            map_location="cpu",
            weights_only=False,
        )
        restored = PhaseClassifier.from_save_dict(ckpt["model"])
        assert restored.n_phases == model.n_phases
        assert restored.phase_names == model.phase_names

    def test_no_checkpoint_without_dir(self, tmp_path: Path) -> None:
        """No checkpoint is saved when checkpoint_dir is None."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=2)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=8)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        learner.update()

        # No crash, no checkpoint_dir set
        assert learner.checkpoint_dir is None


# ---------------------------------------------------------------------------
# Progress callback tests
# ---------------------------------------------------------------------------


class TestProgressCallback:
    """Tests for progress callback invocation."""

    def test_callback_called(self, tmp_path: Path) -> None:
        """Progress callback is called for each step."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=2)
        model = _make_model(phase_names)

        calls: list[tuple[int, int, float]] = []

        def on_progress(step: int, total: int, loss: float) -> None:
            calls.append((step, total, loss))

        cfg = OnlineLearnerConfig(steps_per_update=5, batch_size=8)
        learner = OnlineLearner(
            model=model,
            store=store,
            config=cfg,
            device="cpu",
            progress_callback=on_progress,
        )
        learner.update()

        assert len(calls) == 5
        # Steps should be 1, 2, 3, 4, 5
        assert [c[0] for c in calls] == [1, 2, 3, 4, 5]
        # Total should be 5
        assert all(c[1] == 5 for c in calls)
        # Losses should be finite
        assert all(np.isfinite(c[2]) for c in calls)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample_store(self, tmp_path: Path) -> None:
        """Update works with just one sample in the store."""
        store = TrainingStore(tmp_path / "store")
        det = make_synthetic_detector_info()
        rng = np.random.default_rng(42)
        store.add_sample(
            pattern=rng.integers(0, 256, (60, 80), dtype=np.uint8),
            confirmed_phase="Alpha",
            detector_info=det,
        )

        model = _make_model(["Alpha", "Beta"])
        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=4)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        result = learner.update()

        assert result.steps_completed == 3
        assert np.isfinite(result.mean_loss)

    def test_update_no_eds(self, tmp_path: Path) -> None:
        """Update works with samples that have no EDS data."""
        store = TrainingStore(tmp_path / "store")
        _populate_store(
            store, n_samples=20, n_phases=2, include_eds=False
        )
        model = _make_model(["Phase_0", "Phase_1"])

        cfg = OnlineLearnerConfig(steps_per_update=3, batch_size=8)
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        result = learner.update()

        assert result.steps_completed == 3
        assert np.isfinite(result.mean_loss)

    def test_max_samples_cap(self, tmp_path: Path) -> None:
        """max_samples_per_update caps the new indices used."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=30, n_phases=2)
        model = _make_model(phase_names)

        cfg = OnlineLearnerConfig(
            steps_per_update=3,
            batch_size=8,
            max_samples_per_update=5,
        )
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        # Pass many new indices but cap should limit
        result = learner.update(new_indices=list(range(30)))

        assert result.steps_completed == 3
        assert np.isfinite(result.mean_loss)

    def test_compute_ewc_state_explicit(self, tmp_path: Path) -> None:
        """compute_ewc_state can be called explicitly before update."""
        store = TrainingStore(tmp_path / "store")
        phase_names = _populate_store(store, n_samples=20, n_phases=2)
        model = _make_model(phase_names)
        dataset = EBSDPhaseDataset(
            store=store, phase_names=phase_names, eds_dropout_rate=0.0
        )

        cfg = OnlineLearnerConfig(
            steps_per_update=3, batch_size=8, ewc_lambda=1.0
        )
        learner = OnlineLearner(
            model=model, store=store, config=cfg, device="cpu"
        )
        learner.compute_ewc_state(dataset, n_samples=10)

        assert learner._fisher is not None
        assert learner._old_params is not None

        result = learner.update()
        assert result.steps_completed == 3
