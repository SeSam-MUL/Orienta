"""Tests for ebsd_ai.data.pattern_io module."""

from __future__ import annotations

import numpy as np
import pytest

from ebsd_ai.config import DEFAULT_PATTERN_SIZE
from ebsd_ai.data.pattern_io import (
    NormStats,
    denormalize_pattern,
    normalize_pattern,
    prepare_pattern,
    resize_pattern,
    validate_pattern,
)
from tests.conftest import make_synthetic_pattern


# ---------------------------------------------------------------------------
# validate_pattern
# ---------------------------------------------------------------------------


class TestValidatePattern:
    def test_valid_uint8(self, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(60, 60, rng)
        validate_pattern(pat)  # should not raise

    def test_valid_float(self, rng: np.random.Generator) -> None:
        pat = rng.random((100, 100)).astype(np.float32)
        validate_pattern(pat)

    def test_not_array(self) -> None:
        with pytest.raises(TypeError, match="np.ndarray"):
            validate_pattern([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_wrong_ndim_1d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            validate_pattern(np.zeros(10))

    def test_wrong_ndim_3d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            validate_pattern(np.zeros((10, 10, 3)))

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_pattern(np.zeros((0, 10)))

    def test_non_numeric(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            validate_pattern(np.array([["a", "b"], ["c", "d"]]))


# ---------------------------------------------------------------------------
# resize_pattern
# ---------------------------------------------------------------------------


class TestResizePattern:
    @pytest.mark.parametrize(
        "h,w",
        [(60, 60), (80, 60), (120, 120), (160, 120), (320, 240), (640, 480)],
    )
    def test_output_shape(self, h: int, w: int, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(h, w, rng)
        resized = resize_pattern(pat)
        assert resized.shape == (DEFAULT_PATTERN_SIZE, DEFAULT_PATTERN_SIZE)

    def test_custom_target_size(self, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(100, 100, rng)
        resized = resize_pattern(pat, target_size=64)
        assert resized.shape == (64, 64)

    def test_output_dtype(self, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(60, 80, rng)
        resized = resize_pattern(pat)
        assert resized.dtype == np.float64

    def test_output_range(self, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(100, 100, rng)
        resized = resize_pattern(pat)
        assert resized.min() >= 0.0
        assert resized.max() <= 1.0

    def test_no_nan(self, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(120, 160, rng)
        resized = resize_pattern(pat)
        assert not np.any(np.isnan(resized))


# ---------------------------------------------------------------------------
# normalize_pattern
# ---------------------------------------------------------------------------


class TestNormalizePattern:
    def test_zero_mean(self, rng: np.random.Generator) -> None:
        pat = rng.random((128, 128)).astype(np.float64)
        normalized, _ = normalize_pattern(pat)
        assert abs(normalized.mean()) < 1e-5

    def test_unit_variance(self, rng: np.random.Generator) -> None:
        pat = rng.random((128, 128)).astype(np.float64)
        normalized, _ = normalize_pattern(pat)
        assert abs(normalized.std() - 1.0) < 1e-5

    def test_output_dtype(self, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(60, 60, rng)
        normalized, _ = normalize_pattern(pat)
        assert normalized.dtype == np.float32

    def test_stats_returned(self, rng: np.random.Generator) -> None:
        pat = rng.random((50, 50))
        _, stats = normalize_pattern(pat)
        assert isinstance(stats, NormStats)
        assert stats.std > 0

    def test_constant_pattern(self) -> None:
        pat = np.full((64, 64), 128.0)
        normalized, stats = normalize_pattern(pat)
        assert np.all(normalized == 0)
        assert stats.mean == 128.0


# ---------------------------------------------------------------------------
# denormalize_pattern
# ---------------------------------------------------------------------------


class TestDenormalizePattern:
    def test_roundtrip(self, rng: np.random.Generator) -> None:
        pat = rng.random((100, 100)).astype(np.float64)
        normalized, stats = normalize_pattern(pat)
        restored = denormalize_pattern(normalized, stats)
        np.testing.assert_allclose(restored, pat, atol=1e-4)

    def test_roundtrip_uint8(self, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(80, 80, rng)
        normalized, stats = normalize_pattern(pat)
        restored = denormalize_pattern(normalized, stats)
        np.testing.assert_allclose(restored, pat.astype(np.float64), atol=1e-1)


# ---------------------------------------------------------------------------
# prepare_pattern (combined pipeline)
# ---------------------------------------------------------------------------


class TestPreparePattern:
    @pytest.mark.parametrize(
        "h,w",
        [(60, 60), (80, 60), (120, 120), (320, 240)],
    )
    def test_output_shape(self, h: int, w: int, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(h, w, rng)
        normalized, stats = prepare_pattern(pat)
        assert normalized.shape == (DEFAULT_PATTERN_SIZE, DEFAULT_PATTERN_SIZE)
        assert normalized.dtype == np.float32

    def test_zero_mean_unit_var(self, rng: np.random.Generator) -> None:
        pat = make_synthetic_pattern(160, 120, rng)
        normalized, _ = prepare_pattern(pat)
        assert abs(normalized.mean()) < 0.01
        assert abs(normalized.std() - 1.0) < 0.01
