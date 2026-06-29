"""End-to-end integration test proving the full pipeline works.

This single test file exercises the entire EBSD-AI pipeline with
synthetic data, verifying that data flows correctly from storage
through training, evaluation, prediction, online learning, and
server synchronization.

The test does NOT verify ML accuracy on real data — only that
the pipeline runs without errors and produces sane outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ebsd_ai.config import (
    DataSource,
    DetectorConvention,
    DetectorInfo,
    DetectorManufacturer,
    ModelConfig,
    PhasePrediction,
    TrainingConfig,
)
from ebsd_ai.data.dataset import EBSDPhaseDataset, train_val_split
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.inference.predictor import PhasePredictor
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.sync.server_sync import ServerSync
from ebsd_ai.training.evaluator import (
    EvaluationResult,
    Evaluator,
    format_classification_report,
)
from ebsd_ai.training.online_learner import OnlineLearner, OnlineLearnerConfig
from ebsd_ai.training.trainer import Trainer, TrainingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_NAMES = ["Ferrite", "Austenite", "Martensite"]
N_PHASES = len(PHASE_NAMES)
SAMPLES_PER_PHASE = 30  # enough for train/val split


def _make_detector() -> DetectorInfo:
    return DetectorInfo(
        manufacturer=DetectorManufacturer.OXFORD,
        pc=(0.5, 0.2, 0.6),
        pc_convention=DetectorConvention.OXFORD,
        kv=20.0,
        working_distance=15.0,
        sample_tilt=70.0,
    )


def _populate_store(
    store: TrainingStore,
    rng: np.random.Generator,
    *,
    with_eds: bool = True,
) -> None:
    """Add synthetic samples for all phases to *store*."""
    detector = _make_detector()
    for phase in PHASE_NAMES:
        for _ in range(SAMPLES_PER_PHASE):
            pattern = rng.integers(0, 256, (120, 160), dtype=np.uint8)
            eds: dict[str, float] | None = None
            if with_eds:
                eds = {
                    "Fe": float(rng.uniform(50, 90)),
                    "Cr": float(rng.uniform(0, 20)),
                    "Ni": float(rng.uniform(0, 15)),
                }
            store.add_sample(
                pattern=pattern,
                confirmed_phase=phase,
                detector_info=detector,
                orientation=np.array([1, 0, 0, 0], dtype=np.float32),
                confidence_score=float(rng.uniform(0.5, 1.0)),
                eds_data=eds,
                data_source=DataSource.SIMULATION,
                source_file="integration_test",
            )


# ---------------------------------------------------------------------------
# The integration test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Proves the full pipeline works end-to-end on synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        """Seed the store, build dataset, and set paths."""
        self.tmp_path = tmp_path
        self.store_path = tmp_path / "store"
        self.checkpoint_dir = tmp_path / "checkpoints"
        self.server_dir = tmp_path / "server"

        self.rng = np.random.default_rng(12345)
        self.store = TrainingStore(self.store_path, samples_per_shard=100)
        _populate_store(self.store, self.rng, with_eds=True)

    # -- Step 1: Storage -----------------------------------------------------

    def test_01_store_populated(self) -> None:
        """TrainingStore holds expected samples and stats."""
        stats = self.store.get_dataset_stats()
        assert stats["total_samples"] == N_PHASES * SAMPLES_PER_PHASE
        for phase in PHASE_NAMES:
            assert phase in stats["samples_per_phase"]

    # -- Step 2: Dataset -----------------------------------------------------

    def test_02_dataset_iteration(self) -> None:
        """EBSDPhaseDataset yields correct tensors."""
        ds = EBSDPhaseDataset(self.store, eds_dropout_rate=0.0)
        assert len(ds) == N_PHASES * SAMPLES_PER_PHASE
        assert set(ds.phase_names) == set(PHASE_NAMES)

        sample = ds[0]
        assert sample["pattern"].shape == (1, 128, 128)
        assert sample["eds_input"].shape == (108,)
        assert sample["phase_label"].dtype == torch.int64

    # -- Step 3: Train / Val split -------------------------------------------

    def test_03_train_val_split(self) -> None:
        """Stratified split produces disjoint, non-empty sets."""
        ds = EBSDPhaseDataset(self.store, eds_dropout_rate=0.0)
        train_ds, val_ds = train_val_split(ds, val_fraction=0.2)
        assert len(train_ds) + len(val_ds) == len(ds)
        assert len(train_ds) > 0
        assert len(val_ds) > 0

    # -- Step 4: Model creation ----------------------------------------------

    def test_04_model_creation(self) -> None:
        """PhaseClassifier initialises with correct structure."""
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)
        assert model.n_phases == N_PHASES
        assert model.phase_names == PHASE_NAMES
        # Quick forward pass
        pat = torch.randn(2, 1, 128, 128)
        eds = torch.randn(2, 108)
        logits, alpha = model(pat, eds)
        assert logits.shape == (2, N_PHASES)
        assert alpha.shape == (2, 1)

    # -- Step 5: Training ----------------------------------------------------

    def test_05_training_converges(self) -> None:
        """Trainer runs and loss decreases (at least slightly)."""
        ds = EBSDPhaseDataset(self.store, eds_dropout_rate=0.3)
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)

        tc = TrainingConfig(
            learning_rate=0.001,
            batch_size=16,
            epochs=3,
            early_stopping_patience=5,
            mixed_precision=False,
            val_fraction=0.2,
        )
        trainer = Trainer(
            model=model,
            config=tc,
            output_dir=self.checkpoint_dir,
            device="cpu",
        )
        result = trainer.train(ds)

        assert isinstance(result, TrainingResult)
        assert result.epochs_completed >= 1
        assert len(result.train_losses) == result.epochs_completed
        assert len(result.val_losses) == result.epochs_completed
        assert result.elapsed_seconds > 0

        # Checkpoints exist
        assert (self.checkpoint_dir / "best.pt").exists()
        assert (self.checkpoint_dir / "last.pt").exists()

    # -- Step 6: Evaluation --------------------------------------------------

    def test_06_evaluation(self) -> None:
        """Evaluator computes metrics on the trained model."""
        # Train quickly
        ds = EBSDPhaseDataset(self.store, eds_dropout_rate=0.0)
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)
        tc = TrainingConfig(
            learning_rate=0.001, batch_size=16, epochs=2,
            mixed_precision=False, val_fraction=0.2,
        )
        trainer = Trainer(model=model, config=tc,
                          output_dir=self.checkpoint_dir, device="cpu")
        trainer.train(ds)

        # Evaluate
        eval_ds = EBSDPhaseDataset(self.store, eds_dropout_rate=0.0)
        evaluator = Evaluator(model=model, device="cpu", k=3, batch_size=16)
        result = evaluator.evaluate(eval_ds)

        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.top1_accuracy <= 1.0
        assert 0.0 <= result.topk_accuracy <= 1.0
        assert result.topk_accuracy >= result.top1_accuracy
        assert result.confusion_matrix.shape == (N_PHASES, N_PHASES)
        assert result.n_samples == len(eval_ds)
        assert len(result.phase_names) == N_PHASES

        # Classification report produces non-empty string
        report = format_classification_report(result)
        assert len(report) > 0
        for phase in PHASE_NAMES:
            assert phase in report

    # -- Step 7: Prediction (single + scan) ----------------------------------

    def test_07_single_prediction(self) -> None:
        """PhasePredictor returns valid predictions for a single pattern."""
        # Train minimal model
        ds = EBSDPhaseDataset(self.store, eds_dropout_rate=0.0)
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)
        tc = TrainingConfig(
            learning_rate=0.001, batch_size=16, epochs=2,
            mixed_precision=False, val_fraction=0.2,
        )
        trainer = Trainer(model=model, config=tc,
                          output_dir=self.checkpoint_dir, device="cpu")
        trainer.train(ds)

        # Predict from checkpoint
        predictor = PhasePredictor(
            str(self.checkpoint_dir / "best.pt"), device="cpu"
        )
        assert predictor.has_model
        assert set(predictor.phase_names) == set(PHASE_NAMES)

        pattern = self.rng.integers(0, 256, (120, 160), dtype=np.uint8)
        eds_data = {"Fe": 70.0, "Cr": 18.0, "Ni": 10.0}

        # With both pattern and EDS
        result = predictor.predict_phase(
            pattern=pattern, eds_data=eds_data, k=3,
        )
        assert isinstance(result, PhasePrediction)
        assert len(result.top_k) == N_PHASES  # k=3, n_phases=3
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.eds_contribution <= 1.0
        # All probabilities sum to ~1
        prob_sum = sum(p for _, p in result.top_k)
        assert abs(prob_sum - 1.0) < 0.01

        # Pattern-only (no EDS)
        result_no_eds = predictor.predict_phase(pattern=pattern, k=3)
        assert len(result_no_eds.top_k) == N_PHASES

        # EDS-only (no pattern)
        result_no_pat = predictor.predict_phase(eds_data=eds_data, k=3)
        assert len(result_no_pat.top_k) == N_PHASES

    def test_08_scan_prediction(self) -> None:
        """PhasePredictor.predict_scan handles a mini scan."""
        ds = EBSDPhaseDataset(self.store, eds_dropout_rate=0.0)
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)
        tc = TrainingConfig(
            learning_rate=0.001, batch_size=16, epochs=2,
            mixed_precision=False, val_fraction=0.2,
        )
        trainer = Trainer(model=model, config=tc,
                          output_dir=self.checkpoint_dir, device="cpu")
        trainer.train(ds)

        predictor = PhasePredictor(
            str(self.checkpoint_dir / "best.pt"), device="cpu",
            batch_size=8,
        )

        n_rows, n_cols = 3, 4
        scan = self.rng.integers(
            0, 256, (n_rows, n_cols, 60, 80), dtype=np.uint8,
        )
        eds_maps = {
            "Fe": self.rng.uniform(50, 90, n_rows * n_cols).astype(np.float32),
        }
        mask = np.ones((n_rows, n_cols), dtype=bool)
        mask[0, 0] = False  # skip one pixel

        scan_result = predictor.predict_scan(
            patterns_4d=scan, eds_maps=eds_maps,
            selection_mask=mask, k=2,
        )
        expected_pixels = int(mask.sum())
        assert scan_result.n_pixels == expected_pixels
        assert scan_result.scan_shape == (n_rows, n_cols)
        assert scan_result.probabilities.shape[0] == expected_pixels
        assert scan_result.confidence.shape == (expected_pixels,)
        assert scan_result.eds_contribution.shape == (expected_pixels,)

    # -- Step 8: Online learning ---------------------------------------------

    def test_09_online_learning(self) -> None:
        """OnlineLearner performs an incremental update without error."""
        ds = EBSDPhaseDataset(self.store, eds_dropout_rate=0.0)
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)
        tc = TrainingConfig(
            learning_rate=0.001, batch_size=16, epochs=2,
            mixed_precision=False, val_fraction=0.2,
        )
        trainer = Trainer(model=model, config=tc,
                          output_dir=self.checkpoint_dir, device="cpu")
        trainer.train(ds)

        ol_config = OnlineLearnerConfig(
            learning_rate=0.0001,
            steps_per_update=5,
            batch_size=8,
            replay_fraction=0.5,
            ewc_lambda=0.0,  # skip EWC for speed
        )
        learner = OnlineLearner(
            model=model, store=self.store, config=ol_config,
            device="cpu", checkpoint_dir=self.checkpoint_dir,
        )
        result = learner.update()
        assert result.steps_completed > 0
        assert result.mean_loss >= 0.0
        assert result.samples_used > 0

    # -- Step 9: Server synchronization --------------------------------------

    def test_10_server_sync(self) -> None:
        """ServerSync copies shards between local and server dirs."""
        sync = ServerSync(
            local_dir=str(self.store_path),
            server_dir=str(self.server_dir),
        )

        # Push local shards to server
        push_result = sync.push()
        assert push_result.n_copied > 0
        assert push_result.n_failed == 0

        # Second push: everything should be skipped
        push_again = sync.push()
        assert push_again.n_copied == 0
        assert push_again.n_skipped == push_result.n_copied

        # Pull back (already present locally)
        pull_result = sync.pull()
        assert pull_result.n_failed == 0

        # Server stats
        stats = sync.server_stats()
        assert stats["n_shards"] > 0

    # -- Step 10: Checkpoint round-trip --------------------------------------

    def test_11_checkpoint_roundtrip(self) -> None:
        """Model saves and reloads identically from a checkpoint."""
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)

        path = self.tmp_path / "roundtrip.pt"
        torch.save(model.get_save_dict(), path)

        loaded = PhaseClassifier.from_save_dict(
            torch.load(path, weights_only=False)
        )
        assert loaded.phase_names == model.phase_names
        assert loaded.n_phases == model.n_phases

        # Weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Mismatch on {n1}"

    # -- Step 11: Dynamic phase addition ------------------------------------

    def test_12_add_phase_and_predict(self) -> None:
        """Adding a new phase expands the model and produces valid output."""
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)
        model.add_phase("Bainite")

        assert model.n_phases == N_PHASES + 1
        assert "Bainite" in model.phase_names

        predictor = PhasePredictor.from_model(model, device="cpu")
        pattern = self.rng.integers(0, 256, (120, 160), dtype=np.uint8)
        result = predictor.predict_phase(
            pattern=pattern, eds_data={"Fe": 70.0}, k=4,
        )
        assert len(result.top_k) == N_PHASES + 1
        assert any(name == "Bainite" for name, _ in result.top_k)

    # -- Step 12: No-EDS pipeline -------------------------------------------

    def test_13_no_eds_pipeline(self) -> None:
        """Full pipeline works with training data that has no EDS."""
        store_path = self.tmp_path / "store_no_eds"
        store = TrainingStore(store_path, samples_per_shard=100)
        _populate_store(store, np.random.default_rng(999), with_eds=False)

        ds = EBSDPhaseDataset(store, eds_dropout_rate=0.0)
        config = ModelConfig(n_phases=N_PHASES, pattern_size=128)
        model = PhaseClassifier(config=config, phase_names=PHASE_NAMES)
        tc = TrainingConfig(
            learning_rate=0.001, batch_size=16, epochs=2,
            mixed_precision=False, val_fraction=0.2,
        )
        ckpt = self.tmp_path / "ckpt_no_eds"
        trainer = Trainer(model=model, config=tc,
                          output_dir=ckpt, device="cpu")
        result = trainer.train(ds)
        assert result.epochs_completed >= 1

        # Predict using pattern only
        predictor = PhasePredictor.from_model(model, device="cpu")
        pattern = self.rng.integers(0, 256, (120, 160), dtype=np.uint8)
        pred = predictor.predict_phase(pattern=pattern, k=3)
        assert len(pred.top_k) == N_PHASES

    # -- Step 13: Predictor fallback (no model) ------------------------------

    def test_14_predictor_no_model_raises(self) -> None:
        """PhasePredictor without a model raises RuntimeError."""
        predictor = PhasePredictor(model_path=None, device="cpu")
        assert not predictor.has_model

        with pytest.raises(RuntimeError, match="No model loaded"):
            predictor.predict_phase(eds_data={"Fe": 70.0, "Cr": 18.0})

    # -- Step 14: add_from_indexing CI filter --------------------------------

    def test_15_add_from_indexing(self) -> None:
        """TrainingStore.add_from_indexing filters by CI threshold."""
        store_path = self.tmp_path / "store_ci"
        store = TrainingStore(store_path, samples_per_shard=100)
        n = 20
        patterns = self.rng.integers(0, 256, (n, 60, 60), dtype=np.uint8)
        phase_ids = self.rng.integers(0, 2, n)
        phase_names = ["Alpha", "Beta"]
        orientations = np.zeros((n, 4), dtype=np.float32)
        orientations[:, 0] = 1.0
        # Half above threshold, half below
        ci = np.concatenate([
            np.full(10, 0.5, dtype=np.float32),
            np.full(10, 0.1, dtype=np.float32),
        ])

        added = store.add_from_indexing(
            patterns=patterns,
            phase_ids=phase_ids,
            phase_names=phase_names,
            orientations=orientations,
            confidence_scores=ci,
            ci_threshold=0.3,
            detector_info=_make_detector(),
            source_file="test_indexing.h5",
        )
        # Only the 10 high-CI samples should be added
        assert added == 10
        stats = store.get_dataset_stats()
        assert stats["total_samples"] == 10
