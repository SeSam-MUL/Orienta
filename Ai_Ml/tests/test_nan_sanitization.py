"""Tests for NaN/Inf sanitization across the pipeline.

Verifies that non-finite values (NaN, Inf, -Inf) are handled gracefully
at every entry point and computation stage, preventing silent corruption.
"""

from __future__ import annotations

import numpy as np
import pytest

from ebsd_ai.config import (
    DetectorInfo,
    DetectorManufacturer,
    DetectorConvention,
    eds_dict_to_vector,
    sanitize_array,
)
from ebsd_ai.data.pattern_io import normalize_pattern, prepare_pattern


# ---------------------------------------------------------------------------
# sanitize_array
# ---------------------------------------------------------------------------


class TestSanitizeArray:
    """Tests for the sanitize_array utility."""

    def test_clean_array_unchanged(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = sanitize_array(arr, name="test")
        np.testing.assert_array_equal(result, arr)

    def test_nan_replaced(self) -> None:
        arr = np.array([1.0, np.nan, 3.0])
        result = sanitize_array(arr, name="test")
        assert np.isfinite(result).all()
        assert result[1] == 0.0

    def test_inf_replaced(self) -> None:
        arr = np.array([np.inf, 2.0, -np.inf])
        result = sanitize_array(arr, name="test")
        assert np.isfinite(result).all()
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_custom_replace_value(self) -> None:
        arr = np.array([np.nan, 1.0])
        result = sanitize_array(arr, replace=-1.0, name="test")
        assert result[0] == -1.0

    def test_2d_array(self) -> None:
        arr = np.array([[1.0, np.nan], [np.inf, 2.0]])
        result = sanitize_array(arr, name="test")
        assert np.isfinite(result).all()
        assert result.shape == (2, 2)

    def test_integer_array_passthrough(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = sanitize_array(arr, name="test")
        np.testing.assert_array_equal(result, arr)

    def test_non_numeric_passthrough(self) -> None:
        arr = np.array(["a", "b", "c"])
        result = sanitize_array(arr, name="test")
        np.testing.assert_array_equal(result, arr)

    def test_empty_array(self) -> None:
        arr = np.array([], dtype=np.float64)
        result = sanitize_array(arr, name="test")
        assert len(result) == 0

    def test_all_nan(self) -> None:
        arr = np.array([np.nan, np.nan, np.nan])
        result = sanitize_array(arr, name="test")
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_does_not_modify_original(self) -> None:
        arr = np.array([1.0, np.nan, 3.0])
        _ = sanitize_array(arr, name="test")
        assert np.isnan(arr[1])


# ---------------------------------------------------------------------------
# eds_dict_to_vector with non-finite values
# ---------------------------------------------------------------------------


class TestEdsDictNaN:
    """EDS dict conversion silently drops non-finite values."""

    def test_nan_value_ignored(self) -> None:
        vec = eds_dict_to_vector({"Fe": float("nan"), "Cu": 5.0})
        assert vec[25] == 0.0  # Fe index
        assert vec[28] == 5.0  # Cu index

    def test_inf_value_ignored(self) -> None:
        vec = eds_dict_to_vector({"Fe": float("inf")})
        assert vec[25] == 0.0

    def test_negative_inf_ignored(self) -> None:
        vec = eds_dict_to_vector({"Fe": float("-inf")})
        assert vec[25] == 0.0

    def test_normal_values_pass(self) -> None:
        vec = eds_dict_to_vector({"Fe": 65.2, "C": 8.1})
        assert vec[25] == pytest.approx(65.2)
        assert vec[5] == pytest.approx(8.1)


# ---------------------------------------------------------------------------
# DetectorInfo.encode with non-finite PC
# ---------------------------------------------------------------------------


class TestDetectorInfoNaN:
    """DetectorInfo.encode() sanitizes non-finite PC values."""

    def test_nan_pc_replaced(self) -> None:
        info = DetectorInfo(
            pc=(float("nan"), 0.5, float("inf")),
            manufacturer=DetectorManufacturer.OXFORD,
            pc_convention=DetectorConvention.OXFORD,
        )
        encoded = info.encode()
        assert np.isfinite(encoded).all()
        assert encoded[0] == 0.0  # NaN → 0
        assert encoded[1] == 0.5  # normal
        assert encoded[2] == 0.0  # Inf → 0

    def test_nan_kv_handled(self) -> None:
        info = DetectorInfo(kv=float("nan"))
        encoded = info.encode()
        assert np.isfinite(encoded).all()
        assert encoded[12] == 0.0  # _normalize returns 0 for NaN


# ---------------------------------------------------------------------------
# Pattern normalization with non-finite values
# ---------------------------------------------------------------------------


class TestPatternNormNaN:
    """Pattern normalization sanitizes non-finite pixel values."""

    def test_nan_in_pattern_sanitized(self) -> None:
        pattern = np.array([[1.0, np.nan], [3.0, 4.0]])
        normalized, stats = normalize_pattern(pattern)
        assert np.isfinite(normalized).all()

    def test_inf_in_pattern_sanitized(self) -> None:
        pattern = np.array([[1.0, np.inf], [3.0, 4.0]])
        normalized, stats = normalize_pattern(pattern)
        assert np.isfinite(normalized).all()

    def test_prepare_pattern_sanitizes(self) -> None:
        pattern = np.random.randint(0, 255, (64, 64), dtype=np.uint8).astype(
            np.float64
        )
        pattern[10, 10] = np.nan
        pattern[20, 20] = np.inf
        normalized, stats = prepare_pattern(pattern, target_size=32)
        assert np.isfinite(normalized).all()


# ---------------------------------------------------------------------------
# ScanData.validate detects non-finite
# ---------------------------------------------------------------------------


class TestScanDataValidateNaN:
    """ScanData.validate() reports non-finite values."""

    def test_nan_in_euler_reported(self) -> None:
        from ebsd_ai.data.scan_import import ScanData

        scan = ScanData(
            phase_ids=np.array([0, 1], dtype=np.int32),
            phase_names=["Not indexed", "Ferrite"],
            euler_angles=np.array(
                [[0.1, 0.2, np.nan], [0.4, 0.5, 0.6]], dtype=np.float64
            ),
            confidence_scores=np.array([0.5, 0.8], dtype=np.float32),
        )
        issues = scan.validate()
        assert any("euler_angles" in i and "non-finite" in i for i in issues)

    def test_nan_in_confidence_reported(self) -> None:
        from ebsd_ai.data.scan_import import ScanData

        scan = ScanData(
            phase_ids=np.array([0, 1], dtype=np.int32),
            phase_names=["Not indexed", "Ferrite"],
            euler_angles=np.zeros((2, 3), dtype=np.float64),
            confidence_scores=np.array([np.nan, 0.8], dtype=np.float32),
        )
        issues = scan.validate()
        assert any(
            "confidence_scores" in i and "non-finite" in i for i in issues
        )

    def test_clean_data_no_issues(self) -> None:
        from ebsd_ai.data.scan_import import ScanData

        scan = ScanData(
            phase_ids=np.array([0, 1], dtype=np.int32),
            phase_names=["Not indexed", "Ferrite"],
            euler_angles=np.zeros((2, 3), dtype=np.float64),
            confidence_scores=np.array([0.5, 0.8], dtype=np.float32),
        )
        issues = scan.validate()
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# HDF5 _quaternion_to_euler with degenerate input
# ---------------------------------------------------------------------------


class TestQuaternionToEulerNaN:
    """_quaternion_to_euler handles degenerate quaternions."""

    def test_zero_quaternion(self) -> None:
        from ebsd_ai.data.hdf5_parser import _quaternion_to_euler

        quats = np.array([[0.0, 0.0, 0.0, 0.0]])
        result = _quaternion_to_euler(quats)
        assert np.isfinite(result).all()

    def test_nan_quaternion(self) -> None:
        from ebsd_ai.data.hdf5_parser import _quaternion_to_euler

        quats = np.array([[np.nan, 0.0, 0.0, 0.0]])
        # Input is sanitized before calling; here we test the function
        # itself replaces residual NaN
        result = _quaternion_to_euler(quats)
        assert np.isfinite(result).all()

    def test_normal_quaternion_unchanged(self) -> None:
        from ebsd_ai.data.hdf5_parser import _quaternion_to_euler

        quats = np.array([[1.0, 0.0, 0.0, 0.0]])  # Identity
        result = _quaternion_to_euler(quats)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Dataset __getitem__ sanitization
# ---------------------------------------------------------------------------


class TestDatasetNaN:
    """Dataset.__getitem__ produces finite tensors even with bad data."""

    def test_getitem_finite_output(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Ensure tensors from dataset are always finite."""
        import torch
        from ebsd_ai.data.training_store import TrainingStore
        from ebsd_ai.data.dataset import EBSDPhaseDataset
        from ebsd_ai.config import DetectorInfo

        store = TrainingStore(tmp_path / "store")
        pattern = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        detector = DetectorInfo()
        store.add_sample(
            pattern=pattern,
            eds_data={"Fe": 65.0},
            detector_info=detector,
            confirmed_phase="Ferrite",
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            confidence_score=0.9,
            source_file="test.h5",
        )

        ds = EBSDPhaseDataset(
            store=store,
            phase_names=["Ferrite"],
            eds_dropout_rate=0.0,
        )
        sample = ds[0]
        assert torch.isfinite(sample["pattern"]).all()
        assert torch.isfinite(sample["eds_input"]).all()
