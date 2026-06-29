"""Tests for the PhasePredictor prediction interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ebsd_ai.config import (
    DetectorConvention,
    DetectorInfo,
    ModelConfig,
    PhasePrediction,
)
from ebsd_ai.inference.predictor import PhasePredictor, ScanPredictionResult
from ebsd_ai.models.phase_classifier import PhaseClassifier
from tests.conftest import (
    make_synthetic_detector_info,
    make_synthetic_eds,
    make_synthetic_pattern,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_NAMES = ["Ferrite", "Austenite", "Cementite"]
N_PHASES = len(PHASE_NAMES)


def _make_model() -> PhaseClassifier:
    """Create a small model for testing."""
    config = ModelConfig(n_phases=N_PHASES)
    return PhaseClassifier(config=config, phase_names=PHASE_NAMES)


def _save_checkpoint(model: PhaseClassifier, path: Path) -> Path:
    """Save a model as a Trainer-style checkpoint."""
    ckpt_path = path / "best.pt"
    torch.save({"model": model.get_save_dict()}, ckpt_path)
    return ckpt_path


def _save_direct_checkpoint(model: PhaseClassifier, path: Path) -> Path:
    """Save a model as a direct get_save_dict() output."""
    ckpt_path = path / "direct.pt"
    torch.save(model.get_save_dict(), ckpt_path)
    return ckpt_path


@pytest.fixture
def model() -> PhaseClassifier:
    """Untrained model."""
    return _make_model()


@pytest.fixture
def checkpoint_path(model: PhaseClassifier, tmp_path: Path) -> Path:
    """Path to a saved Trainer-style checkpoint."""
    return _save_checkpoint(model, tmp_path)


@pytest.fixture
def direct_checkpoint_path(model: PhaseClassifier, tmp_path: Path) -> Path:
    """Path to a direct save dict checkpoint."""
    return _save_direct_checkpoint(model, tmp_path)


@pytest.fixture
def predictor(model: PhaseClassifier) -> PhasePredictor:
    """Predictor wrapping a model (no file loading)."""
    return PhasePredictor.from_model(model, device="cpu")


@pytest.fixture
def pattern() -> np.ndarray:
    """A single synthetic pattern."""
    return make_synthetic_pattern(120, 160, np.random.default_rng(0))


@pytest.fixture
def eds_data() -> dict[str, float]:
    """Synthetic EDS data."""
    return make_synthetic_eds(rng=np.random.default_rng(0))


@pytest.fixture
def detector_info() -> DetectorInfo:
    """Synthetic detector info."""
    return make_synthetic_detector_info(rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# PhasePredictor construction
# ---------------------------------------------------------------------------


class TestPredictorInit:
    """Tests for PhasePredictor construction."""

    def test_from_model(self, model: PhaseClassifier) -> None:
        pred = PhasePredictor.from_model(model, device="cpu")
        assert pred.has_model
        assert pred.phase_names == PHASE_NAMES

    def test_no_model(self) -> None:
        pred = PhasePredictor(model_path=None, device="cpu")
        assert not pred.has_model
        assert pred.phase_names == []

    def test_load_trainer_checkpoint(self, checkpoint_path: Path) -> None:
        pred = PhasePredictor(str(checkpoint_path), device="cpu")
        assert pred.has_model
        assert pred.phase_names == PHASE_NAMES

    def test_load_direct_checkpoint(self, direct_checkpoint_path: Path) -> None:
        pred = PhasePredictor(str(direct_checkpoint_path), device="cpu")
        assert pred.has_model
        assert pred.phase_names == PHASE_NAMES

    def test_bad_checkpoint_format(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.pt"
        torch.save({"random_key": 42}, bad_path)
        with pytest.raises(ValueError, match="Unrecognised checkpoint"):
            PhasePredictor(str(bad_path), device="cpu")

    def test_auto_device(self, model: PhaseClassifier) -> None:
        pred = PhasePredictor.from_model(model, device="auto")
        assert pred.device.type in ("cpu", "cuda")

    def test_cpu_device(self, model: PhaseClassifier) -> None:
        pred = PhasePredictor.from_model(model, device="cpu")
        assert pred.device == torch.device("cpu")

    def test_custom_batch_size(self, model: PhaseClassifier) -> None:
        pred = PhasePredictor.from_model(model, device="cpu", batch_size=16)
        assert pred.batch_size == 16


# ---------------------------------------------------------------------------
# Single-pattern prediction
# ---------------------------------------------------------------------------


class TestPredictPhase:
    """Tests for predict_phase()."""

    def test_returns_prediction(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
        eds_data: dict[str, float],
        detector_info: DetectorInfo,
    ) -> None:
        result = predictor.predict_phase(
            pattern=pattern, eds_data=eds_data, detector_info=detector_info,
        )
        assert isinstance(result, PhasePrediction)

    def test_top_k_count(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        result = predictor.predict_phase(pattern=pattern, k=3)
        assert len(result.top_k) == N_PHASES  # k clamped to n_phases

    def test_top_k_clamped(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        result = predictor.predict_phase(pattern=pattern, k=10)
        assert len(result.top_k) == N_PHASES

    def test_top_k_smaller(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        result = predictor.predict_phase(pattern=pattern, k=1)
        assert len(result.top_k) == 1

    def test_confidence_range(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        result = predictor.predict_phase(pattern=pattern)
        assert 0.0 <= result.confidence <= 1.0

    def test_eds_contribution_range(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
        eds_data: dict[str, float],
    ) -> None:
        result = predictor.predict_phase(
            pattern=pattern, eds_data=eds_data,
        )
        assert 0.0 <= result.eds_contribution <= 1.0

    def test_probabilities_sum_to_one(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        """Top-k probs should be <= 1 total (they're a subset)."""
        result = predictor.predict_phase(pattern=pattern, k=N_PHASES)
        total = sum(p for _, p in result.top_k)
        assert abs(total - 1.0) < 1e-5

    def test_top_k_sorted_descending(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        result = predictor.predict_phase(pattern=pattern, k=N_PHASES)
        probs = [p for _, p in result.top_k]
        assert probs == sorted(probs, reverse=True)

    def test_phase_names_valid(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        result = predictor.predict_phase(pattern=pattern)
        for name, _ in result.top_k:
            assert name in PHASE_NAMES

    def test_confidence_matches_top1(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        result = predictor.predict_phase(pattern=pattern, k=3)
        if result.top_k:
            assert abs(result.confidence - result.top_k[0][1]) < 1e-7

    def test_pattern_various_sizes(
        self,
        predictor: PhasePredictor,
    ) -> None:
        """Prediction works with various input pattern sizes."""
        rng = np.random.default_rng(42)
        for h, w in [(60, 60), (80, 60), (320, 240), (640, 480)]:
            pat = make_synthetic_pattern(h, w, rng)
            result = predictor.predict_phase(pattern=pat)
            assert len(result.top_k) > 0


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for missing inputs and fallback behavior."""

    def test_pattern_only(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        """Works without EDS data."""
        result = predictor.predict_phase(pattern=pattern, eds_data=None)
        assert isinstance(result, PhasePrediction)
        assert len(result.top_k) > 0

    def test_eds_only(
        self,
        predictor: PhasePredictor,
        eds_data: dict[str, float],
        detector_info: DetectorInfo,
    ) -> None:
        """Works without pattern (EDS-only prediction)."""
        result = predictor.predict_phase(
            pattern=None, eds_data=eds_data, detector_info=detector_info,
        )
        assert isinstance(result, PhasePrediction)
        assert len(result.top_k) > 0

    def test_no_detector_info(
        self,
        predictor: PhasePredictor,
        pattern: np.ndarray,
    ) -> None:
        """Works without detector info (uses defaults)."""
        result = predictor.predict_phase(
            pattern=pattern, detector_info=None,
        )
        assert isinstance(result, PhasePrediction)

    def test_neither_pattern_nor_eds(
        self,
        predictor: PhasePredictor,
    ) -> None:
        """Raises ValueError when both pattern and EDS are missing."""
        with pytest.raises(ValueError, match="At least one"):
            predictor.predict_phase(pattern=None, eds_data=None)

    def test_no_model_raises(self) -> None:
        """predict_phase raises RuntimeError when no model is loaded."""
        pred = PhasePredictor(model_path=None, device="cpu")
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match="No model loaded"):
            pred.predict_phase(pattern=make_synthetic_pattern(rng=rng))

    def test_no_model_with_eds_raises(self) -> None:
        """predict_phase raises RuntimeError even with EDS data."""
        pred = PhasePredictor(model_path=None, device="cpu")
        with pytest.raises(RuntimeError, match="No model loaded"):
            pred.predict_phase(eds_data={"Fe": 70.0, "C": 2.0})


# ---------------------------------------------------------------------------
# Scan prediction
# ---------------------------------------------------------------------------


class TestPredictScan:
    """Tests for predict_scan() batch prediction."""

    def _make_scan(
        self,
        n_rows: int = 4,
        n_cols: int = 5,
        pat_h: int = 60,
        pat_w: int = 80,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Create a synthetic 4D scan."""
        if rng is None:
            rng = np.random.default_rng(0)
        return rng.integers(
            0, 256, size=(n_rows, n_cols, pat_h, pat_w), dtype=np.uint8,
        )

    def test_returns_scan_result(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan()
        result = predictor.predict_scan(scan)
        assert isinstance(result, ScanPredictionResult)

    def test_scan_shape_stored(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan(n_rows=3, n_cols=7)
        result = predictor.predict_scan(scan)
        assert result.scan_shape == (3, 7)

    def test_n_pixels(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan(n_rows=3, n_cols=5)
        result = predictor.predict_scan(scan)
        assert result.n_pixels == 15

    def test_output_shapes(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan(n_rows=2, n_cols=3)
        result = predictor.predict_scan(scan, k=2)
        assert result.phase_names.shape == (6, 2)
        assert result.probabilities.shape == (6, 2)
        assert result.confidence.shape == (6,)
        assert result.eds_contribution.shape == (6,)

    def test_confidence_range(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan(n_rows=2, n_cols=2)
        result = predictor.predict_scan(scan)
        assert (result.confidence >= 0.0).all()
        assert (result.confidence <= 1.0).all()

    def test_eds_contribution_range(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan(n_rows=2, n_cols=2)
        result = predictor.predict_scan(scan)
        assert (result.eds_contribution >= 0.0).all()
        assert (result.eds_contribution <= 1.0).all()

    def test_with_eds_maps(
        self,
        predictor: PhasePredictor,
    ) -> None:
        rng = np.random.default_rng(1)
        scan = self._make_scan(n_rows=2, n_cols=3, rng=rng)
        eds_maps = {
            "Fe": rng.uniform(0, 80, size=6).astype(np.float32),
            "C": rng.uniform(0, 5, size=6).astype(np.float32),
        }
        result = predictor.predict_scan(scan, eds_maps=eds_maps)
        assert result.n_pixels == 6

    def test_with_detector_info(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan(n_rows=2, n_cols=2)
        det = make_synthetic_detector_info(rng=np.random.default_rng(0))
        result = predictor.predict_scan(scan, detector_info=det)
        assert result.n_pixels == 4

    def test_selection_mask(
        self,
        predictor: PhasePredictor,
    ) -> None:
        """Only predicts for masked pixels."""
        scan = self._make_scan(n_rows=3, n_cols=4)
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 0] = True
        mask[1, 2] = True
        mask[2, 3] = True
        result = predictor.predict_scan(scan, selection_mask=mask)
        assert result.n_pixels == 3

    def test_empty_mask(
        self,
        predictor: PhasePredictor,
    ) -> None:
        """No pixels selected → empty result."""
        scan = self._make_scan(n_rows=2, n_cols=2)
        mask = np.zeros((2, 2), dtype=bool)
        result = predictor.predict_scan(scan, selection_mask=mask)
        assert result.n_pixels == 0

    def test_wrong_scan_ndim(
        self,
        predictor: PhasePredictor,
    ) -> None:
        """Raises ValueError for non-4D input."""
        with pytest.raises(ValueError, match="4-D"):
            predictor.predict_scan(np.zeros((10, 60, 80)))

    def test_wrong_mask_shape(
        self,
        predictor: PhasePredictor,
    ) -> None:
        """Raises ValueError if mask shape doesn't match scan."""
        scan = self._make_scan(n_rows=3, n_cols=4)
        mask = np.ones((2, 2), dtype=bool)
        with pytest.raises(ValueError, match="selection_mask shape"):
            predictor.predict_scan(scan, selection_mask=mask)

    def test_no_model_raises(self) -> None:
        """Scan prediction requires a model."""
        pred = PhasePredictor(model_path=None, device="cpu")
        scan = np.zeros((2, 2, 60, 80), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="No model loaded"):
            pred.predict_scan(scan)

    def test_phase_names_valid(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan(n_rows=2, n_cols=2)
        result = predictor.predict_scan(scan)
        for name in result.phase_names.ravel():
            assert name in PHASE_NAMES

    def test_k_clamped(
        self,
        predictor: PhasePredictor,
    ) -> None:
        scan = self._make_scan(n_rows=2, n_cols=2)
        result = predictor.predict_scan(scan, k=100)
        assert result.k == N_PHASES

    def test_batch_processing(self) -> None:
        """Verify that small batch_size processes all pixels."""
        model = _make_model()
        pred = PhasePredictor.from_model(model, device="cpu", batch_size=2)
        rng = np.random.default_rng(0)
        scan = rng.integers(0, 256, size=(3, 3, 60, 80), dtype=np.uint8)
        result = pred.predict_scan(scan)
        assert result.n_pixels == 9


# ---------------------------------------------------------------------------
# ScanPredictionResult
# ---------------------------------------------------------------------------


class TestScanPredictionResult:
    """Tests for the ScanPredictionResult container."""

    def test_n_pixels(self) -> None:
        result = ScanPredictionResult(
            phase_names=np.array([["A", "B"]] * 5, dtype=object),
            probabilities=np.zeros((5, 2), dtype=np.float32),
            confidence=np.zeros(5, dtype=np.float32),
            eds_contribution=np.zeros(5, dtype=np.float32),
            scan_shape=(2, 3),
            k=2,
        )
        assert result.n_pixels == 5

    def test_scan_shape(self) -> None:
        result = ScanPredictionResult(
            phase_names=np.empty((0, 3), dtype=object),
            probabilities=np.empty((0, 3), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            eds_contribution=np.empty(0, dtype=np.float32),
            scan_shape=(10, 20),
            k=3,
        )
        assert result.scan_shape == (10, 20)
        assert result.n_pixels == 0
