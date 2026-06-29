"""Tests for public API exports, error messages, and MAD mapping.

Covers TASK-33: clean __init__.py exports, RuntimeError on no model,
configurable MAD-to-confidence mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Public API exports via ebsd_ai.__init__
# ---------------------------------------------------------------------------


class TestPackageExports:
    """All key classes are importable from the top-level package."""

    def test_version(self) -> None:
        import ebsd_ai

        assert hasattr(ebsd_ai, "__version__")
        assert isinstance(ebsd_ai.__version__, str)

    def test_phase_predictor(self) -> None:
        from ebsd_ai import PhasePredictor

        assert PhasePredictor is not None

    def test_training_store(self) -> None:
        from ebsd_ai import TrainingStore

        assert TrainingStore is not None

    def test_phase_classifier(self) -> None:
        from ebsd_ai import PhaseClassifier

        assert PhaseClassifier is not None

    def test_pattern_enhancer(self) -> None:
        from ebsd_ai import PatternEnhancer

        assert PatternEnhancer is not None

    def test_trainer(self) -> None:
        from ebsd_ai import Trainer

        assert Trainer is not None

    def test_evaluator(self) -> None:
        from ebsd_ai import Evaluator

        assert Evaluator is not None

    def test_online_learner(self) -> None:
        from ebsd_ai import OnlineLearner

        assert OnlineLearner is not None

    def test_server_sync(self) -> None:
        from ebsd_ai import ServerSync

        assert ServerSync is not None

    def test_config_classes(self) -> None:
        from ebsd_ai import (
            DataConfig,
            DataSource,
            DetectorConvention,
            DetectorInfo,
            DetectorManufacturer,
            ModelConfig,
            PhasePrediction,
            TrainingConfig,
        )

        assert ModelConfig is not None
        assert TrainingConfig is not None
        assert DataConfig is not None
        assert DetectorInfo is not None
        assert DetectorConvention is not None
        assert DetectorManufacturer is not None
        assert DataSource is not None
        assert PhasePrediction is not None

    def test_data_classes(self) -> None:
        from ebsd_ai import (
            EBSDPhaseDataset,
            ImportResult,
            ScanData,
            ScanPredictionResult,
            parse_scan,
        )

        assert EBSDPhaseDataset is not None
        assert ScanData is not None
        assert ImportResult is not None
        assert ScanPredictionResult is not None
        assert parse_scan is not None

    def test_combined_loss(self) -> None:
        from ebsd_ai import CombinedLoss

        assert CombinedLoss is not None

    def test_mad_to_confidence_export(self) -> None:
        from ebsd_ai import mad_to_confidence

        assert callable(mad_to_confidence)

    def test_all_list(self) -> None:
        import ebsd_ai

        assert hasattr(ebsd_ai, "__all__")
        assert len(ebsd_ai.__all__) > 10
        for name in ebsd_ai.__all__:
            assert hasattr(ebsd_ai, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# RuntimeError on predict without model
# ---------------------------------------------------------------------------


class TestPredictNoModel:
    """predict_phase and predict_scan raise RuntimeError without model."""

    def test_predict_phase_no_model(self) -> None:
        from ebsd_ai import PhasePredictor

        pred = PhasePredictor(model_path=None, device="cpu")
        pattern = np.random.default_rng(0).integers(
            0, 255, (64, 64), dtype=np.uint8
        )
        with pytest.raises(RuntimeError, match="No model loaded"):
            pred.predict_phase(pattern=pattern)

    def test_predict_phase_no_model_with_eds(self) -> None:
        from ebsd_ai import PhasePredictor

        pred = PhasePredictor(model_path=None, device="cpu")
        with pytest.raises(RuntimeError, match="No model loaded"):
            pred.predict_phase(eds_data={"Fe": 70.0})

    def test_predict_scan_no_model(self) -> None:
        from ebsd_ai import PhasePredictor

        pred = PhasePredictor(model_path=None, device="cpu")
        scan = np.zeros((2, 2, 32, 32), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="No model loaded"):
            pred.predict_scan(scan)

    def test_error_message_is_helpful(self) -> None:
        from ebsd_ai import PhasePredictor

        pred = PhasePredictor(model_path=None, device="cpu")
        with pytest.raises(RuntimeError, match="PhasePredictor"):
            pred.predict_phase(
                pattern=np.zeros((32, 32), dtype=np.uint8)
            )


# ---------------------------------------------------------------------------
# Configurable MAD-to-confidence mapping
# ---------------------------------------------------------------------------


class TestMadToConfidence:
    """mad_to_confidence with default and custom decay."""

    def test_zero_mad_is_full_confidence(self) -> None:
        from ebsd_ai.config import mad_to_confidence

        result = mad_to_confidence(np.array([0.0]))
        assert result[0] == pytest.approx(1.0)

    def test_high_mad_is_low_confidence(self) -> None:
        from ebsd_ai.config import mad_to_confidence

        result = mad_to_confidence(np.array([5.0]))
        assert result[0] < 0.01

    def test_default_decay(self) -> None:
        from ebsd_ai.config import MAD_DECAY_DEFAULT, mad_to_confidence

        mad = np.array([1.0])
        result = mad_to_confidence(mad)
        expected = np.exp(-1.0 / MAD_DECAY_DEFAULT)
        assert result[0] == pytest.approx(expected)

    def test_custom_decay_larger(self) -> None:
        """Larger decay → higher confidence for same MAD."""
        from ebsd_ai.config import mad_to_confidence

        mad = np.array([2.0])
        default = mad_to_confidence(mad, decay=1.0)
        larger = mad_to_confidence(mad, decay=2.0)
        assert larger[0] > default[0]

    def test_custom_decay_smaller(self) -> None:
        """Smaller decay → lower confidence for same MAD."""
        from ebsd_ai.config import mad_to_confidence

        mad = np.array([1.0])
        default = mad_to_confidence(mad, decay=1.0)
        smaller = mad_to_confidence(mad, decay=0.5)
        assert smaller[0] < default[0]

    def test_negative_mad_clamped(self) -> None:
        from ebsd_ai.config import mad_to_confidence

        result = mad_to_confidence(np.array([-1.0, -5.0]))
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_batch(self) -> None:
        from ebsd_ai.config import mad_to_confidence

        mad = np.array([0.0, 1.0, 2.0, 3.0])
        result = mad_to_confidence(mad)
        assert result.shape == (4,)
        assert result[0] > result[1] > result[2] > result[3]
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_ctf_parser_uses_central_function(self) -> None:
        """CTF parser should use the central mad_to_confidence."""
        import ebsd_ai.data.ctf_parser as ctf_mod

        # The old _mad_to_confidence should be removed
        assert not hasattr(ctf_mod, "_mad_to_confidence")

    def test_hdf5_parser_uses_central_function(self) -> None:
        """HDF5 parser should import mad_to_confidence from config."""
        import ebsd_ai.data.hdf5_parser as hdf5_mod

        assert hasattr(hdf5_mod, "mad_to_confidence")
