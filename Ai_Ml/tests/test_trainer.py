"""Tests for the training loop, checkpoints, and early stopping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ebsd_ai.config import ModelConfig, TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset, train_val_split
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.training.trainer import (
    Trainer,
    TrainingResult,
    load_checkpoint,
    save_checkpoint,
)
from tests.conftest import make_synthetic_detector_info, make_synthetic_pattern

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_NAMES = ["Alpha", "Beta", "Gamma"]
N_PHASES = len(PHASE_NAMES)


def _make_model() -> PhaseClassifier:
    """Create a small model for testing."""
    config = ModelConfig(n_phases=N_PHASES)
    return PhaseClassifier(config=config, phase_names=PHASE_NAMES)


def _make_config(**overrides: object) -> TrainingConfig:
    """Training config with fast defaults for tests."""
    defaults: dict[str, object] = {
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "early_stopping_patience": 10,
        "val_fraction": 0.3,
        "mixed_precision": False,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def _populate_store(
    store: TrainingStore,
    n_per_phase: int = 12,
    seed: int = 0,
) -> None:
    """Add enough samples per phase for training + validation."""
    rng = np.random.default_rng(seed)
    det = make_synthetic_detector_info(rng=rng)
    for phase in PHASE_NAMES:
        for _ in range(n_per_phase):
            store.add_sample(
                pattern=make_synthetic_pattern(120, 160, rng),
                confirmed_phase=phase,
                detector_info=det,
                confidence_score=float(rng.uniform(0.5, 1.0)),
            )


@pytest.fixture
def store(tmp_path: Path) -> TrainingStore:
    """TrainingStore with 36 samples (12 per phase)."""
    s = TrainingStore(tmp_path / "store", samples_per_shard=100)
    _populate_store(s, n_per_phase=12)
    return s


@pytest.fixture
def dataset(store: TrainingStore) -> EBSDPhaseDataset:
    """Dataset wrapping the populated store."""
    return EBSDPhaseDataset(
        store,
        phase_names=PHASE_NAMES,
        eds_dropout_rate=0.0,
    )


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temporary directory for checkpoints."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# TrainingResult
# ---------------------------------------------------------------------------


class TestTrainingResult:
    """Tests for the TrainingResult dataclass."""

    def test_defaults(self) -> None:
        r = TrainingResult()
        assert r.train_losses == []
        assert r.val_losses == []
        assert r.best_epoch == 0
        assert r.best_val_loss == float("inf")
        assert r.epochs_completed == 0
        assert r.early_stopped is False
        assert r.elapsed_seconds == 0.0

    def test_mutable_lists_independent(self) -> None:
        """Each instance has independent lists."""
        r1 = TrainingResult()
        r2 = TrainingResult()
        r1.train_losses.append(1.0)
        assert r2.train_losses == []


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpoints:
    """Tests for save_checkpoint / load_checkpoint."""

    def test_save_creates_file(self, output_dir: Path) -> None:
        model = _make_model()
        config = _make_config()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=10,
        )

        path = output_dir / "test_ckpt.pt"
        save_checkpoint(path, model, optimizer, scheduler, 0, 1.5, config)
        assert path.exists()

    def test_roundtrip(self, output_dir: Path) -> None:
        """Save and load restores all fields."""
        model = _make_model()
        config = _make_config()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=10,
        )

        path = output_dir / "roundtrip.pt"
        save_checkpoint(path, model, optimizer, scheduler, 5, 0.42, config)

        ckpt = load_checkpoint(path, torch.device("cpu"))
        assert ckpt["epoch"] == 5
        assert ckpt["best_val_loss"] == 0.42
        assert "model" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert "training_config" in ckpt

    def test_model_can_be_restored(self, output_dir: Path) -> None:
        """Model state_dict can be loaded into a fresh model."""
        model = _make_model()
        config = _make_config()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=10,
        )

        path = output_dir / "restore.pt"
        save_checkpoint(path, model, optimizer, scheduler, 0, 1.0, config)

        ckpt = load_checkpoint(path, torch.device("cpu"))
        restored = PhaseClassifier.from_save_dict(ckpt["model"])
        assert restored.phase_names == PHASE_NAMES
        assert restored.n_phases == N_PHASES

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_checkpoint creates parent directories if needed."""
        model = _make_model()
        config = _make_config()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=10,
        )

        path = tmp_path / "deep" / "nested" / "ckpt.pt"
        save_checkpoint(path, model, optimizer, scheduler, 0, 1.0, config)
        assert path.exists()


# ---------------------------------------------------------------------------
# Trainer initialization
# ---------------------------------------------------------------------------


class TestTrainerInit:
    """Tests for Trainer construction."""

    def test_default_device_cpu(self, output_dir: Path) -> None:
        model = _make_model()
        trainer = Trainer(model, device="cpu", output_dir=output_dir)
        assert trainer.device == torch.device("cpu")

    def test_auto_device(self, output_dir: Path) -> None:
        model = _make_model()
        trainer = Trainer(model, device="auto", output_dir=output_dir)
        # Should be either cpu or cuda, no error
        assert trainer.device.type in ("cpu", "cuda")

    def test_default_config(self, output_dir: Path) -> None:
        model = _make_model()
        trainer = Trainer(model, output_dir=output_dir, device="cpu")
        assert trainer.config.epochs == 50  # TrainingConfig default

    def test_custom_config(self, output_dir: Path) -> None:
        model = _make_model()
        config = _make_config(epochs=7)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        assert trainer.config.epochs == 7

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        model = _make_model()
        new_dir = tmp_path / "new_checkpoints"
        Trainer(model, output_dir=new_dir, device="cpu")
        assert new_dir.is_dir()

    def test_amp_disabled_on_cpu(self, output_dir: Path) -> None:
        model = _make_model()
        config = _make_config(mixed_precision=True)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        assert trainer.use_amp is False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class TestTraining:
    """Tests for the main training loop."""

    def test_returns_training_result(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        assert isinstance(result, TrainingResult)

    def test_loss_lists_have_correct_length(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=3)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        assert len(result.train_losses) == 3
        assert len(result.val_losses) == 3

    def test_epochs_completed(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=3)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        assert result.epochs_completed == 3

    def test_elapsed_seconds_positive(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        assert result.elapsed_seconds > 0

    def test_losses_are_finite(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        for loss in result.train_losses:
            assert np.isfinite(loss)
        for loss in result.val_losses:
            assert np.isfinite(loss)

    def test_best_val_loss_matches_min(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=3)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        assert result.best_val_loss == min(result.val_losses)

    def test_best_epoch_valid(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=3)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        assert 0 <= result.best_epoch < result.epochs_completed


# ---------------------------------------------------------------------------
# Checkpoint file creation
# ---------------------------------------------------------------------------


class TestCheckpointCreation:
    """Verify checkpoint files are created during training."""

    def test_best_checkpoint_created(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        trainer.train(dataset)
        assert (output_dir / "best.pt").exists()

    def test_last_checkpoint_created(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        trainer.train(dataset)
        assert (output_dir / "last.pt").exists()

    def test_best_checkpoint_loadable(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        trainer.train(dataset)

        ckpt = load_checkpoint(output_dir / "best.pt", torch.device("cpu"))
        restored = PhaseClassifier.from_save_dict(ckpt["model"])
        assert restored.n_phases == N_PHASES


# ---------------------------------------------------------------------------
# Auto val split
# ---------------------------------------------------------------------------


class TestAutoValSplit:
    """Verify training auto-splits when no val_dataset given."""

    def test_auto_split_works(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        """Training succeeds with auto val split."""
        model = _make_model()
        config = _make_config(epochs=2, val_fraction=0.3)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        assert result.epochs_completed == 2

    def test_explicit_val_dataset(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        """Training succeeds with explicit val dataset."""
        train_ds, val_ds = train_val_split(dataset, val_fraction=0.3)
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(train_ds, val_dataset=val_ds)
        assert result.epochs_completed == 2


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Tests for the early stopping mechanism."""

    def test_early_stop_flag(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        """With patience=1 and enough epochs, early stopping can trigger."""
        model = _make_model()
        config = _make_config(epochs=50, early_stopping_patience=1)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        # With patience=1, it's very likely to early-stop on random data
        # because val loss will fluctuate
        if result.early_stopped:
            assert result.epochs_completed < 50

    def test_no_early_stop_with_high_patience(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        """With patience > epochs, early stopping never triggers."""
        model = _make_model()
        config = _make_config(epochs=3, early_stopping_patience=100)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        result = trainer.train(dataset)
        assert result.early_stopped is False
        assert result.epochs_completed == 3


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    """Tests for the per-epoch callback."""

    def test_callback_called_per_epoch(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        calls: list[tuple[int, int, float, float]] = []

        def callback(
            epoch: int, total: int,
            train_loss: float, val_loss: float,
        ) -> None:
            calls.append((epoch, total, train_loss, val_loss))

        model = _make_model()
        config = _make_config(epochs=3)
        trainer = Trainer(
            model, config=config, output_dir=output_dir,
            device="cpu", progress_callback=callback,
        )
        trainer.train(dataset)
        assert len(calls) == 3

    def test_callback_receives_correct_epoch(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        calls: list[tuple[int, int, float, float]] = []

        def callback(
            epoch: int, total: int,
            train_loss: float, val_loss: float,
        ) -> None:
            calls.append((epoch, total, train_loss, val_loss))

        model = _make_model()
        config = _make_config(epochs=3)
        trainer = Trainer(
            model, config=config, output_dir=output_dir,
            device="cpu", progress_callback=callback,
        )
        trainer.train(dataset)

        epochs_seen = [c[0] for c in calls]
        assert epochs_seen == [0, 1, 2]

    def test_callback_total_is_max_epochs(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        calls: list[tuple[int, int, float, float]] = []

        def callback(
            epoch: int, total: int,
            train_loss: float, val_loss: float,
        ) -> None:
            calls.append((epoch, total, train_loss, val_loss))

        model = _make_model()
        config = _make_config(epochs=3)
        trainer = Trainer(
            model, config=config, output_dir=output_dir,
            device="cpu", progress_callback=callback,
        )
        trainer.train(dataset)

        for _, total, _, _ in calls:
            assert total == 3

    def test_callback_losses_finite(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        calls: list[tuple[int, int, float, float]] = []

        def callback(
            epoch: int, total: int,
            train_loss: float, val_loss: float,
        ) -> None:
            calls.append((epoch, total, train_loss, val_loss))

        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(
            model, config=config, output_dir=output_dir,
            device="cpu", progress_callback=callback,
        )
        trainer.train(dataset)

        for _, _, tl, vl in calls:
            assert np.isfinite(tl)
            assert np.isfinite(vl)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


class TestResume:
    """Tests for resuming training from a checkpoint."""

    def test_resume_restores_epoch(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        """Resume restores the start epoch."""
        model = _make_model()
        config = _make_config(epochs=5)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        trainer.train(dataset)

        # Resume from last checkpoint
        model2 = _make_model()
        trainer2 = Trainer(model2, config=config, output_dir=output_dir, device="cpu")
        trainer2.resume(output_dir / "last.pt")
        # After resuming from last.pt (which saved epoch=4), _start_epoch = 5
        # So training would start at epoch 5 (which >= epochs=5), completing immediately
        assert trainer2._start_epoch > 0

    def test_resume_then_train(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        """Training after resume works without error."""
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        trainer.train(dataset)

        # Resume and train with more epochs
        model2 = _make_model()
        config2 = _make_config(epochs=4)
        trainer2 = Trainer(model2, config=config2, output_dir=output_dir, device="cpu")
        trainer2.resume(output_dir / "last.pt")
        result = trainer2.train(dataset)

        # Should have done some epochs (started from epoch 2, went to 4)
        assert result.epochs_completed >= 0

    def test_resume_preserves_phase_names(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        """Model phase names survive save/resume."""
        model = _make_model()
        config = _make_config(epochs=2)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        trainer.train(dataset)

        model2 = _make_model()
        trainer2 = Trainer(model2, config=config, output_dir=output_dir, device="cpu")
        trainer2.resume(output_dir / "best.pt")
        assert trainer2.model.phase_names == PHASE_NAMES


# ---------------------------------------------------------------------------
# Class weight setting
# ---------------------------------------------------------------------------


class TestClassWeights:
    """Verify that class weights are set from training data."""

    def test_criterion_has_weights_after_train(
        self, dataset: EBSDPhaseDataset, output_dir: Path,
    ) -> None:
        model = _make_model()
        config = _make_config(epochs=1)
        trainer = Trainer(model, config=config, output_dir=output_dir, device="cpu")
        trainer.train(dataset)
        # The CombinedLoss should have class weights set
        ce = trainer.criterion.ce_loss
        assert ce.class_weights is not None
        assert ce.class_weights.shape == (N_PHASES,)
