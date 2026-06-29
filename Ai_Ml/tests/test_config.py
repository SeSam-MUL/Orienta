"""Tests for ebsd_ai.config module."""

import numpy as np
import pytest

from ebsd_ai.config import (
    NUM_ELEMENTS,
    DataConfig,
    DataSource,
    DetectorConvention,
    DetectorInfo,
    DetectorManufacturer,
    ELEMENT_INDEX,
    ELEMENT_SYMBOLS,
    ModelConfig,
    PhasePrediction,
    TrainingConfig,
    eds_dict_to_vector,
    has_eds,
)


class TestConstants:
    def test_element_count(self) -> None:
        assert len(ELEMENT_SYMBOLS) == NUM_ELEMENTS

    def test_element_index_matches_symbols(self) -> None:
        for i, sym in enumerate(ELEMENT_SYMBOLS):
            assert ELEMENT_INDEX[sym] == i

    def test_known_elements(self) -> None:
        assert ELEMENT_INDEX["Fe"] == 25
        assert ELEMENT_INDEX["C"] == 5
        assert ELEMENT_INDEX["Si"] == 13
        assert ELEMENT_INDEX["O"] == 7


class TestEnums:
    def test_detector_conventions(self) -> None:
        names = [c.value for c in DetectorConvention]
        assert "OXFORD" in names
        assert "BRUKER" in names
        assert "EDAX" in names
        assert "EMSOFT" in names
        assert "KIKUCHIPY" in names
        assert len(names) == 5

    def test_detector_manufacturers(self) -> None:
        names = [m.value for m in DetectorManufacturer]
        assert len(names) == 4

    def test_data_sources(self) -> None:
        names = [s.value for s in DataSource]
        assert set(names) == {"SIMULATION", "AUTO_INDEXED", "USER_CONFIRMED"}


class TestDetectorInfo:
    def test_defaults(self) -> None:
        info = DetectorInfo()
        assert info.kv == 20.0
        assert info.sample_tilt == 70.0
        assert info.manufacturer == DetectorManufacturer.OTHER

    def test_roundtrip_dict(self) -> None:
        info = DetectorInfo(
            manufacturer=DetectorManufacturer.OXFORD,
            pc=(0.4, 0.3, 0.6),
            pc_convention=DetectorConvention.OXFORD,
            kv=15.0,
            working_distance=12.0,
            sample_tilt=65.0,
        )
        d = info.to_dict()
        restored = DetectorInfo.from_dict(d)
        assert restored.manufacturer == info.manufacturer
        assert restored.pc == info.pc
        assert restored.pc_convention == info.pc_convention
        assert restored.kv == info.kv
        assert restored.working_distance == info.working_distance
        assert restored.sample_tilt == info.sample_tilt

    def test_encode_shape(self) -> None:
        info = DetectorInfo()
        vec = info.encode()
        assert vec.shape == (15,)
        assert vec.dtype == np.float32

    def test_encode_convention_onehot(self) -> None:
        for conv in DetectorConvention:
            info = DetectorInfo(pc_convention=conv)
            vec = info.encode()
            conv_slice = vec[3:8]
            assert conv_slice.sum() == pytest.approx(1.0)
            idx = list(DetectorConvention).index(conv)
            assert conv_slice[idx] == 1.0

    def test_encode_manufacturer_onehot(self) -> None:
        for mfr in DetectorManufacturer:
            info = DetectorInfo(manufacturer=mfr)
            vec = info.encode()
            mfr_slice = vec[8:12]
            assert mfr_slice.sum() == pytest.approx(1.0)

    def test_encode_normalized_values(self) -> None:
        info = DetectorInfo(kv=5.0, working_distance=5.0, sample_tilt=50.0)
        vec = info.encode()
        assert vec[12] == pytest.approx(0.0)
        assert vec[13] == pytest.approx(0.0)
        assert vec[14] == pytest.approx(0.0)

        info2 = DetectorInfo(kv=30.0, working_distance=25.0, sample_tilt=80.0)
        vec2 = info2.encode()
        assert vec2[12] == pytest.approx(1.0)
        assert vec2[13] == pytest.approx(1.0)
        assert vec2[14] == pytest.approx(1.0)

    def test_encoded_length(self) -> None:
        assert DetectorInfo.encoded_length() == 15


class TestPhasePrediction:
    def test_defaults(self) -> None:
        pred = PhasePrediction()
        assert pred.top_k == []
        assert pred.confidence == 0.0
        assert pred.eds_contribution == 0.0


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.pattern_feature_dim == 256
        assert cfg.eds_feature_dim == 64
        assert cfg.n_phases == 2

    def test_roundtrip_dict(self) -> None:
        cfg = ModelConfig(n_phases=5, dropout=0.5)
        restored = ModelConfig.from_dict(cfg.to_dict())
        assert restored.n_phases == 5
        assert restored.dropout == 0.5


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.eds_dropout_rate == 0.3
        assert cfg.ci_threshold == 0.3
        assert cfg.batch_size == 64

    def test_roundtrip_dict(self) -> None:
        cfg = TrainingConfig(epochs=100, learning_rate=5e-4)
        restored = TrainingConfig.from_dict(cfg.to_dict())
        assert restored.epochs == 100
        assert restored.learning_rate == 5e-4


class TestDataConfig:
    def test_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.pattern_size == 128
        assert cfg.num_elements == 92

    def test_roundtrip_dict(self) -> None:
        cfg = DataConfig(pattern_size=64)
        restored = DataConfig.from_dict(cfg.to_dict())
        assert restored.pattern_size == 64


class TestEdsDictToVector:
    def test_none_returns_zeros(self) -> None:
        vec = eds_dict_to_vector(None)
        assert vec.shape == (NUM_ELEMENTS,)
        assert np.all(vec == 0)

    def test_known_elements(self) -> None:
        eds = {"Fe": 65.2, "C": 8.1}
        vec = eds_dict_to_vector(eds)
        assert vec[ELEMENT_INDEX["Fe"]] == pytest.approx(65.2)
        assert vec[ELEMENT_INDEX["C"]] == pytest.approx(8.1)
        assert vec[ELEMENT_INDEX["O"]] == 0.0

    def test_xray_line_stripping(self) -> None:
        eds = {"Fe Ka1": 50.0, "Cr Kb": 10.0}
        vec = eds_dict_to_vector(eds)
        assert vec[ELEMENT_INDEX["Fe"]] == pytest.approx(50.0)
        assert vec[ELEMENT_INDEX["Cr"]] == pytest.approx(10.0)

    def test_case_insensitive(self) -> None:
        eds = {"fe": 30.0, "FE": 20.0}
        vec = eds_dict_to_vector(eds)
        assert vec[ELEMENT_INDEX["Fe"]] == pytest.approx(20.0)

    def test_unknown_element_ignored(self) -> None:
        eds = {"Xx": 5.0, "Fe": 10.0}
        vec = eds_dict_to_vector(eds)
        assert vec[ELEMENT_INDEX["Fe"]] == pytest.approx(10.0)

    def test_empty_dict(self) -> None:
        vec = eds_dict_to_vector({})
        assert np.all(vec == 0)


class TestHasEds:
    def test_zero_vector(self) -> None:
        assert has_eds(np.zeros(NUM_ELEMENTS)) is False

    def test_nonzero_vector(self) -> None:
        vec = np.zeros(NUM_ELEMENTS)
        vec[25] = 50.0
        assert has_eds(vec) is True
