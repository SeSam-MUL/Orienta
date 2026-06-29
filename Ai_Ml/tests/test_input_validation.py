"""Tests for input validation hardening (TASK-34).

Covers __post_init__ validators on config dataclasses, predictor input
validation, TrainingStore batch validation, and Trainer empty-dataset check.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# DetectorInfo validation
# ---------------------------------------------------------------------------


class TestDetectorInfoValidation:
    """DetectorInfo __post_init__ checks."""

    def test_valid_defaults(self) -> None:
        from ebsd_ai.config import DetectorInfo

        d = DetectorInfo()
        assert d.kv == 20.0

    def test_invalid_pc_length(self) -> None:
        from ebsd_ai.config import DetectorInfo

        with pytest.raises(ValueError, match="3-tuple"):
            DetectorInfo(pc=(0.5, 0.5))  # type: ignore[arg-type]

    def test_invalid_pc_type(self) -> None:
        from ebsd_ai.config import DetectorInfo

        with pytest.raises(ValueError, match="3-tuple"):
            DetectorInfo(pc=[0.5, 0.5, 0.5])  # type: ignore[arg-type]

    def test_invalid_manufacturer_type(self) -> None:
        from ebsd_ai.config import DetectorInfo

        with pytest.raises(TypeError, match="DetectorManufacturer"):
            DetectorInfo(manufacturer="OXFORD")  # type: ignore[arg-type]

    def test_invalid_convention_type(self) -> None:
        from ebsd_ai.config import DetectorInfo

        with pytest.raises(TypeError, match="DetectorConvention"):
            DetectorInfo(pc_convention="BRUKER")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


class TestModelConfigValidation:
    """ModelConfig __post_init__ checks."""

    def test_valid_defaults(self) -> None:
        from ebsd_ai.config import ModelConfig

        m = ModelConfig()
        assert m.pattern_feature_dim == 256

    def test_negative_feature_dim(self) -> None:
        from ebsd_ai.config import ModelConfig

        with pytest.raises(ValueError, match="pattern_feature_dim"):
            ModelConfig(pattern_feature_dim=-1)

    def test_zero_n_phases(self) -> None:
        from ebsd_ai.config import ModelConfig

        with pytest.raises(ValueError, match="n_phases"):
            ModelConfig(n_phases=0)

    def test_dropout_too_high(self) -> None:
        from ebsd_ai.config import ModelConfig

        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=1.5)

    def test_dropout_negative(self) -> None:
        from ebsd_ai.config import ModelConfig

        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=-0.1)

    def test_zero_pattern_size(self) -> None:
        from ebsd_ai.config import ModelConfig

        with pytest.raises(ValueError, match="pattern_size"):
            ModelConfig(pattern_size=0)

    def test_from_dict_validates(self) -> None:
        """from_dict should trigger __post_init__ validation."""
        from ebsd_ai.config import ModelConfig

        with pytest.raises(ValueError, match="dropout"):
            ModelConfig.from_dict({"dropout": 2.0})


# ---------------------------------------------------------------------------
# TrainingConfig validation
# ---------------------------------------------------------------------------


class TestTrainingConfigValidation:
    """TrainingConfig __post_init__ checks."""

    def test_valid_defaults(self) -> None:
        from ebsd_ai.config import TrainingConfig

        t = TrainingConfig()
        assert t.epochs == 50

    def test_negative_learning_rate(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=-0.01)

    def test_zero_learning_rate(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=0.0)

    def test_zero_batch_size(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_zero_epochs(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_eds_dropout_rate_out_of_range(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="eds_dropout_rate"):
            TrainingConfig(eds_dropout_rate=1.5)

    def test_ci_threshold_negative(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="ci_threshold"):
            TrainingConfig(ci_threshold=-0.1)

    def test_val_fraction_too_high(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="val_fraction"):
            TrainingConfig(val_fraction=1.1)

    def test_negative_weight_decay(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="weight_decay"):
            TrainingConfig(weight_decay=-1e-4)

    def test_from_dict_validates(self) -> None:
        from ebsd_ai.config import TrainingConfig

        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig.from_dict({"learning_rate": -1.0})


# ---------------------------------------------------------------------------
# DataConfig validation
# ---------------------------------------------------------------------------


class TestDataConfigValidation:
    """DataConfig __post_init__ checks."""

    def test_valid_defaults(self) -> None:
        from ebsd_ai.config import DataConfig

        d = DataConfig()
        assert d.pattern_size == 128

    def test_zero_pattern_size(self) -> None:
        from ebsd_ai.config import DataConfig

        with pytest.raises(ValueError, match="pattern_size"):
            DataConfig(pattern_size=0)

    def test_negative_rotation(self) -> None:
        from ebsd_ai.config import DataConfig

        with pytest.raises(ValueError, match="max_rotation_deg"):
            DataConfig(max_rotation_deg=-1.0)

    def test_negative_noise_std(self) -> None:
        from ebsd_ai.config import DataConfig

        with pytest.raises(ValueError, match="noise_std"):
            DataConfig(noise_std=-0.01)

    def test_negative_brightness(self) -> None:
        from ebsd_ai.config import DataConfig

        with pytest.raises(ValueError, match="brightness_range"):
            DataConfig(brightness_range=-0.1)


# ---------------------------------------------------------------------------
# PhasePredictor.predict_phase() input validation
# ---------------------------------------------------------------------------


class TestPredictPhaseValidation:
    """predict_phase() validates pattern, EDS, and k."""

    def _make_predictor(self):  # noqa: ANN202
        from ebsd_ai.config import ModelConfig
        from ebsd_ai.inference.predictor import PhasePredictor
        from ebsd_ai.models.phase_classifier import PhaseClassifier

        model = PhaseClassifier(
            config=ModelConfig(n_phases=3, pattern_size=32),
            phase_names=["A", "B", "C"],
        )
        return PhasePredictor.from_model(model, device="cpu")

    def test_pattern_wrong_type(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(TypeError, match="numpy array"):
            pred.predict_phase(pattern=[[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_pattern_wrong_ndim(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(ValueError, match="2-D"):
            pred.predict_phase(pattern=np.zeros((2, 3, 4), dtype=np.uint8))

    def test_pattern_empty(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(ValueError, match="empty"):
            pred.predict_phase(pattern=np.zeros((0, 0), dtype=np.uint8))

    def test_eds_wrong_type(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(TypeError, match="dict"):
            pred.predict_phase(
                pattern=np.zeros((32, 32), dtype=np.uint8),
                eds_data=[1.0, 2.0],  # type: ignore[arg-type]
            )

    def test_k_zero(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(ValueError, match="positive integer"):
            pred.predict_phase(
                pattern=np.zeros((32, 32), dtype=np.uint8), k=0
            )

    def test_k_negative(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(ValueError, match="positive integer"):
            pred.predict_phase(
                pattern=np.zeros((32, 32), dtype=np.uint8), k=-1
            )

    def test_valid_pattern_passes(self) -> None:
        pred = self._make_predictor()
        result = pred.predict_phase(
            pattern=np.random.default_rng(0).integers(
                0, 255, (32, 32), dtype=np.uint8
            )
        )
        assert len(result.top_k) > 0


# ---------------------------------------------------------------------------
# PhasePredictor.predict_scan() input validation
# ---------------------------------------------------------------------------


class TestPredictScanValidation:
    """predict_scan() validates patterns_4d, EDS maps, and k."""

    def _make_predictor(self):  # noqa: ANN202
        from ebsd_ai.config import ModelConfig
        from ebsd_ai.inference.predictor import PhasePredictor
        from ebsd_ai.models.phase_classifier import PhaseClassifier

        model = PhaseClassifier(
            config=ModelConfig(n_phases=2, pattern_size=32),
            phase_names=["X", "Y"],
        )
        return PhasePredictor.from_model(model, device="cpu")

    def test_patterns_wrong_type(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(TypeError, match="numpy array"):
            pred.predict_scan(patterns_4d=[[[[1]]]])  # type: ignore[arg-type]

    def test_k_zero_scan(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(ValueError, match="positive integer"):
            pred.predict_scan(
                np.zeros((2, 2, 32, 32), dtype=np.uint8), k=0
            )

    def test_eds_maps_wrong_type(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(TypeError, match="dict"):
            pred.predict_scan(
                np.zeros((2, 2, 32, 32), dtype=np.uint8),
                eds_maps=[1, 2, 3],  # type: ignore[arg-type]
            )

    def test_eds_maps_value_wrong_type(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(TypeError, match="numpy array"):
            pred.predict_scan(
                np.zeros((2, 2, 32, 32), dtype=np.uint8),
                eds_maps={"Fe": [70.0, 65.0, 60.0, 55.0]},  # type: ignore[dict-item]
            )

    def test_eds_maps_value_too_short(self) -> None:
        pred = self._make_predictor()
        with pytest.raises(ValueError, match="elements"):
            pred.predict_scan(
                np.zeros((2, 2, 32, 32), dtype=np.uint8),
                eds_maps={"Fe": np.array([70.0, 65.0])},  # need 4
            )

    def test_valid_scan_passes(self) -> None:
        pred = self._make_predictor()
        result = pred.predict_scan(
            np.random.default_rng(0).integers(
                0, 255, (2, 2, 32, 32), dtype=np.uint8
            )
        )
        assert result.n_pixels == 4


# ---------------------------------------------------------------------------
# TrainingStore.add_batch() validation
# ---------------------------------------------------------------------------


class TestTrainingStoreBatchValidation:
    """add_batch() validates shapes and consistency."""

    def test_patterns_wrong_ndim(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        with pytest.raises(ValueError, match="3-D"):
            store.add_batch(
                patterns=np.zeros((64, 64), dtype=np.uint8),
                confirmed_phases=["A"],
                detector_info=DetectorInfo(),
            )

    def test_phase_count_mismatch(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        with pytest.raises(ValueError, match="confirmed_phases length"):
            store.add_batch(
                patterns=np.zeros((3, 32, 32), dtype=np.uint8),
                confirmed_phases=["A", "B"],  # 2 != 3
                detector_info=DetectorInfo(),
            )

    def test_orientation_count_mismatch(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        with pytest.raises(ValueError, match="orientations count"):
            store.add_batch(
                patterns=np.zeros((3, 32, 32), dtype=np.uint8),
                confirmed_phases=["A", "B", "C"],
                detector_info=DetectorInfo(),
                orientations=np.zeros((2, 4), dtype=np.float32),
            )

    def test_confidence_count_mismatch(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        with pytest.raises(ValueError, match="confidence_scores count"):
            store.add_batch(
                patterns=np.zeros((3, 32, 32), dtype=np.uint8),
                confirmed_phases=["A", "B", "C"],
                detector_info=DetectorInfo(),
                confidence_scores=np.ones(2, dtype=np.float32),
            )

    def test_valid_batch_passes(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        n = store.add_batch(
            patterns=np.zeros((3, 32, 32), dtype=np.uint8),
            confirmed_phases=["A", "B", "C"],
            detector_info=DetectorInfo(),
        )
        assert n == 3


# ---------------------------------------------------------------------------
# TrainingStore.add_sample() validation
# ---------------------------------------------------------------------------


class TestTrainingStoreAddSampleValidation:
    """add_sample() validates phase name."""

    def test_empty_phase_name(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        with pytest.raises(ValueError, match="non-empty"):
            store.add_sample(
                pattern=np.zeros((32, 32), dtype=np.uint8),
                confirmed_phase="",
                detector_info=DetectorInfo(),
            )

    def test_whitespace_phase_name(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        with pytest.raises(ValueError, match="non-empty"):
            store.add_sample(
                pattern=np.zeros((32, 32), dtype=np.uint8),
                confirmed_phase="   ",
                detector_info=DetectorInfo(),
            )


# ---------------------------------------------------------------------------
# TrainingStore.add_from_indexing() validation
# ---------------------------------------------------------------------------


class TestAddFromIndexingValidation:
    """add_from_indexing() validates ci_threshold and phase_ids."""

    def test_ci_threshold_out_of_range(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        with pytest.raises(ValueError, match="ci_threshold"):
            store.add_from_indexing(
                patterns=np.zeros((2, 32, 32), dtype=np.uint8),
                phase_ids=np.array([0, 0]),
                phase_names=["A"],
                orientations=np.zeros((2, 4), dtype=np.float32),
                confidence_scores=np.ones(2, dtype=np.float32),
                detector_info=DetectorInfo(),
                ci_threshold=1.5,
            )

    def test_phase_ids_out_of_range(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        store = TrainingStore(tmp_path / "store")
        with pytest.raises(ValueError, match="phase_ids"):
            store.add_from_indexing(
                patterns=np.zeros((2, 32, 32), dtype=np.uint8),
                phase_ids=np.array([0, 5]),  # 5 out of range
                phase_names=["A", "B"],
                orientations=np.zeros((2, 4), dtype=np.float32),
                confidence_scores=np.ones(2, dtype=np.float32),
                detector_info=DetectorInfo(),
            )


# ---------------------------------------------------------------------------
# Trainer empty-dataset check
# ---------------------------------------------------------------------------


class TestTrainerValidation:
    """Trainer.train() validates dataset is non-empty."""

    def test_empty_dataset_raises(self, tmp_path) -> None:  # noqa: ANN001
        from ebsd_ai.config import ModelConfig
        from ebsd_ai.data.dataset import EBSDPhaseDataset
        from ebsd_ai.data.training_store import TrainingStore
        from ebsd_ai.models.phase_classifier import PhaseClassifier
        from ebsd_ai.training.trainer import Trainer

        store = TrainingStore(tmp_path / "store")
        model = PhaseClassifier(
            config=ModelConfig(n_phases=2, pattern_size=32),
            phase_names=["A", "B"],
        )
        ds = EBSDPhaseDataset(store, phase_names=["A", "B"])
        assert len(ds) == 0

        trainer = Trainer(
            model=model,
            output_dir=tmp_path / "ckpt",
            device="cpu",
        )
        with pytest.raises(ValueError, match="empty dataset"):
            trainer.train(ds)
