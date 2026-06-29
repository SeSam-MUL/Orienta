"""Tests for evaluation metrics, confusion matrix, and calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ebsd_ai.config import ModelConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.training.evaluator import (
    EvaluationResult,
    Evaluator,
    format_classification_report,
)
from tests.conftest import (
    make_synthetic_detector_info,
    make_synthetic_pattern,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_NAMES = ["Alpha", "Beta", "Gamma"]
N_PHASES = len(PHASE_NAMES)


def _make_model() -> PhaseClassifier:
    """Create a small model for testing."""
    config = ModelConfig(n_phases=N_PHASES)
    return PhaseClassifier(config=config, phase_names=PHASE_NAMES)


def _populate_store(
    store: TrainingStore,
    n_per_phase: int = 10,
    seed: int = 0,
) -> None:
    """Add samples to the store."""
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
    """TrainingStore with 30 samples (10 per phase)."""
    s = TrainingStore(tmp_path / "store", samples_per_shard=100)
    _populate_store(s, n_per_phase=10)
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
def model() -> PhaseClassifier:
    """Untrained model."""
    return _make_model()


@pytest.fixture
def evaluator(model: PhaseClassifier) -> Evaluator:
    """Evaluator wrapping the model."""
    return Evaluator(model, device="cpu", k=3, batch_size=8)


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    """Tests for the EvaluationResult dataclass."""

    def test_defaults(self) -> None:
        r = EvaluationResult()
        assert r.top1_accuracy == 0.0
        assert r.topk_accuracy == 0.0
        assert r.k == 3
        assert r.per_phase_precision == {}
        assert r.per_phase_recall == {}
        assert r.per_phase_f1 == {}
        assert r.n_samples == 0

    def test_normalized_confusion_matrix_shape(self) -> None:
        cm = np.array([[5, 1, 0], [0, 8, 2], [1, 0, 6]], dtype=np.int64)
        r = EvaluationResult(confusion_matrix=cm)
        norm = r.normalized_confusion_matrix()
        assert norm.shape == (3, 3)

    def test_normalized_rows_sum_to_one(self) -> None:
        cm = np.array([[5, 1, 0], [0, 8, 2], [1, 0, 6]], dtype=np.int64)
        r = EvaluationResult(confusion_matrix=cm)
        norm = r.normalized_confusion_matrix()
        for row in range(3):
            assert abs(norm[row].sum() - 1.0) < 1e-10

    def test_normalized_zero_row(self) -> None:
        """Row with zero support stays all zeros."""
        cm = np.array([[5, 1], [0, 0]], dtype=np.int64)
        r = EvaluationResult(confusion_matrix=cm)
        norm = r.normalized_confusion_matrix()
        assert norm[1].sum() == 0.0

    def test_mutable_dicts_independent(self) -> None:
        r1 = EvaluationResult()
        r2 = EvaluationResult()
        r1.per_phase_precision["test"] = 0.5
        assert "test" not in r2.per_phase_precision


# ---------------------------------------------------------------------------
# Evaluator init
# ---------------------------------------------------------------------------


class TestEvaluatorInit:
    """Tests for Evaluator construction."""

    def test_default_device_cpu(self, model: PhaseClassifier) -> None:
        ev = Evaluator(model, device="cpu")
        assert ev.device == torch.device("cpu")

    def test_auto_device(self, model: PhaseClassifier) -> None:
        ev = Evaluator(model, device="auto")
        assert ev.device.type in ("cpu", "cuda")

    def test_custom_k(self, model: PhaseClassifier) -> None:
        ev = Evaluator(model, device="cpu", k=5)
        assert ev.k == 5

    def test_custom_bins(self, model: PhaseClassifier) -> None:
        ev = Evaluator(model, device="cpu", n_calibration_bins=20)
        assert ev.n_calibration_bins == 20


# ---------------------------------------------------------------------------
# Evaluation on dataset
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Tests for the evaluate() method."""

    def test_returns_evaluation_result(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert isinstance(result, EvaluationResult)

    def test_n_samples(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert result.n_samples == len(dataset)

    def test_phase_names(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert result.phase_names == PHASE_NAMES

    def test_top1_accuracy_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert 0.0 <= result.top1_accuracy <= 1.0

    def test_topk_accuracy_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert 0.0 <= result.topk_accuracy <= 1.0

    def test_topk_ge_top1(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        """Top-k accuracy is always >= top-1 accuracy."""
        result = evaluator.evaluate(dataset)
        assert result.topk_accuracy >= result.top1_accuracy - 1e-10

    def test_confusion_matrix_shape(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert result.confusion_matrix.shape == (N_PHASES, N_PHASES)

    def test_confusion_matrix_sums_to_n_samples(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert result.confusion_matrix.sum() == result.n_samples

    def test_confusion_matrix_non_negative(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert (result.confusion_matrix >= 0).all()

    def test_per_phase_metrics_keys(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        for name in PHASE_NAMES:
            assert name in result.per_phase_precision
            assert name in result.per_phase_recall
            assert name in result.per_phase_f1
            assert name in result.per_phase_support

    def test_precision_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        for p in result.per_phase_precision.values():
            assert 0.0 <= p <= 1.0

    def test_recall_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        for r in result.per_phase_recall.values():
            assert 0.0 <= r <= 1.0

    def test_f1_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        for f in result.per_phase_f1.values():
            assert 0.0 <= f <= 1.0

    def test_support_sums_to_n_samples(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        total = sum(result.per_phase_support.values())
        assert total == result.n_samples

    def test_mean_confidence_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert 0.0 <= result.mean_confidence <= 1.0

    def test_mean_attention_alpha_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert 0.0 <= result.mean_attention_alpha <= 1.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    """Tests for calibration analysis."""

    def test_calibration_bins_shape(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        # n_bins + 1 edges
        assert result.calibration_bins.shape == (11,)

    def test_calibration_bins_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert result.calibration_bins[0] == 0.0
        assert result.calibration_bins[-1] == 1.0

    def test_calibration_accuracy_shape(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert result.calibration_accuracy.shape == (10,)

    def test_calibration_accuracy_range(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        for acc in result.calibration_accuracy:
            assert 0.0 <= acc <= 1.0

    def test_calibration_counts_sum(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        assert result.calibration_counts.sum() == result.n_samples

    def test_custom_bins(
        self,
        model: PhaseClassifier,
        dataset: EBSDPhaseDataset,
    ) -> None:
        ev = Evaluator(model, device="cpu", n_calibration_bins=5)
        result = ev.evaluate(dataset)
        assert result.calibration_bins.shape == (6,)
        assert result.calibration_accuracy.shape == (5,)


# ---------------------------------------------------------------------------
# K handling
# ---------------------------------------------------------------------------


class TestKHandling:
    """Tests for top-k with various k values."""

    def test_k_larger_than_n_phases(
        self,
        model: PhaseClassifier,
        dataset: EBSDPhaseDataset,
    ) -> None:
        """k > n_phases is clamped to n_phases."""
        ev = Evaluator(model, device="cpu", k=10)
        result = ev.evaluate(dataset)
        # All labels are in top-n_phases, so topk == 1.0
        assert result.topk_accuracy == 1.0
        assert result.k == N_PHASES

    def test_k_equals_1(
        self,
        model: PhaseClassifier,
        dataset: EBSDPhaseDataset,
    ) -> None:
        """k=1 means top-k == top-1."""
        ev = Evaluator(model, device="cpu", k=1)
        result = ev.evaluate(dataset)
        assert abs(result.topk_accuracy - result.top1_accuracy) < 1e-10

    def test_k_equals_2(
        self,
        model: PhaseClassifier,
        dataset: EBSDPhaseDataset,
    ) -> None:
        ev = Evaluator(model, device="cpu", k=2)
        result = ev.evaluate(dataset)
        assert result.k == 2
        assert result.topk_accuracy >= result.top1_accuracy - 1e-10


# ---------------------------------------------------------------------------
# Perfect classifier
# ---------------------------------------------------------------------------


class TestPerfectClassifier:
    """Verify metrics with a model forced to predict correctly."""

    def test_perfect_accuracy(
        self, store: TrainingStore, tmp_path: Path,
    ) -> None:
        """A model that always predicts the right class gets 100%."""
        # Build a dataset with only 1 phase so any prediction is correct
        single_store = TrainingStore(
            tmp_path / "single", samples_per_shard=100,
        )
        rng = np.random.default_rng(7)
        det = make_synthetic_detector_info(rng=rng)
        for _ in range(10):
            single_store.add_sample(
                make_synthetic_pattern(rng=rng),
                "OnlyPhase",
                det,
            )
        ds = EBSDPhaseDataset(
            single_store, phase_names=["OnlyPhase"],
            eds_dropout_rate=0.0,
        )
        config = ModelConfig(n_phases=1)
        model = PhaseClassifier(config=config, phase_names=["OnlyPhase"])
        ev = Evaluator(model, device="cpu")
        result = ev.evaluate(ds)
        assert result.top1_accuracy == 1.0
        assert result.topk_accuracy == 1.0
        assert result.per_phase_precision["OnlyPhase"] == 1.0
        assert result.per_phase_recall["OnlyPhase"] == 1.0
        assert result.per_phase_f1["OnlyPhase"] == 1.0


# ---------------------------------------------------------------------------
# format_classification_report
# ---------------------------------------------------------------------------


class TestReport:
    """Tests for the text report formatter."""

    def test_returns_string(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        report = format_classification_report(result)
        assert isinstance(report, str)

    def test_contains_phase_names(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        report = format_classification_report(result)
        for name in PHASE_NAMES:
            assert name in report

    def test_contains_accuracies(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        report = format_classification_report(result)
        assert "Top-1 Accuracy" in report
        assert "Top-3 Accuracy" in report

    def test_contains_sample_count(
        self,
        evaluator: Evaluator,
        dataset: EBSDPhaseDataset,
    ) -> None:
        result = evaluator.evaluate(dataset)
        report = format_classification_report(result)
        assert str(result.n_samples) in report

    def test_empty_result(self) -> None:
        """Report works on empty result without error."""
        r = EvaluationResult(phase_names=["A"])
        report = format_classification_report(r)
        assert "A" in report
