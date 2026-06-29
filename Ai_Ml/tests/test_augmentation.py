"""Tests for data augmentation of EBSD patterns and EDS data."""

from __future__ import annotations

import numpy as np
import pytest

from ebsd_ai.config import NUM_ELEMENTS
from ebsd_ai.data.augmentation import (
    AugmentationConfig,
    augment_eds,
    augment_pattern,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aug_config() -> AugmentationConfig:
    """Default augmentation config."""
    return AugmentationConfig()


@pytest.fixture
def disabled_config() -> AugmentationConfig:
    """Augmentation disabled."""
    return AugmentationConfig(enabled=False)


@pytest.fixture
def norm_pattern(rng: np.random.Generator) -> np.ndarray:
    """A normalized (zero-mean, unit-var) 128x128 pattern."""
    p = rng.standard_normal((128, 128)).astype(np.float32)
    return p


@pytest.fixture
def eds_vector(rng: np.random.Generator) -> np.ndarray:
    """A realistic EDS vector with a few nonzero elements."""
    vec = np.zeros(NUM_ELEMENTS, dtype=np.float32)
    # Fe, C, Cr, Ni, Mn at indices 25, 5, 23, 27, 24
    vec[25] = 65.0  # Fe
    vec[5] = 8.0    # C
    vec[23] = 12.0  # Cr
    vec[27] = 5.0   # Ni
    vec[24] = 2.0   # Mn
    return vec


# ---------------------------------------------------------------------------
# Pattern augmentation tests
# ---------------------------------------------------------------------------


class TestAugmentPattern:
    """Tests for pattern augmentation."""

    def test_output_shape_unchanged(
        self, norm_pattern: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Augmented pattern has the same shape as input."""
        result = augment_pattern(norm_pattern, aug_config, rng=np.random.default_rng(0))
        assert result.shape == norm_pattern.shape

    def test_output_dtype_float32(
        self, norm_pattern: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Augmented pattern is float32."""
        result = augment_pattern(norm_pattern, aug_config, rng=np.random.default_rng(0))
        assert result.dtype == np.float32

    def test_disabled_returns_copy(
        self, norm_pattern: np.ndarray, disabled_config: AugmentationConfig,
    ) -> None:
        """When disabled, returns an identical copy (not the same object)."""
        result = augment_pattern(norm_pattern, disabled_config)
        np.testing.assert_array_equal(result, norm_pattern)
        assert result is not norm_pattern

    def test_augmentation_is_stochastic(
        self, norm_pattern: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Two augmentations with different seeds produce different results."""
        r1 = augment_pattern(norm_pattern, aug_config, rng=np.random.default_rng(1))
        r2 = augment_pattern(norm_pattern, aug_config, rng=np.random.default_rng(2))
        assert not np.allclose(r1, r2)

    def test_augmentation_is_reproducible(
        self, norm_pattern: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Same seed produces identical results."""
        r1 = augment_pattern(norm_pattern, aug_config, rng=np.random.default_rng(42))
        r2 = augment_pattern(norm_pattern, aug_config, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_no_nan_or_inf(
        self, norm_pattern: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Augmented patterns contain no NaN or Inf."""
        for seed in range(10):
            result = augment_pattern(
                norm_pattern, aug_config, rng=np.random.default_rng(seed)
            )
            assert np.all(np.isfinite(result)), f"Non-finite values at seed={seed}"

    def test_does_not_modify_input(
        self, norm_pattern: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Input array is not modified in place."""
        original = norm_pattern.copy()
        augment_pattern(norm_pattern, aug_config, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(norm_pattern, original)

    def test_rotation_only(self, norm_pattern: np.ndarray) -> None:
        """Rotation-only config changes the pattern."""
        config = AugmentationConfig(
            max_rotation_deg=2.0,
            noise_std=0.0,
            brightness_range=0.0,
            contrast_range=0.0,
            flip_horizontal=False,
        )
        result = augment_pattern(norm_pattern, config, rng=np.random.default_rng(7))
        assert not np.allclose(result, norm_pattern)
        assert result.shape == norm_pattern.shape

    def test_noise_only(self, norm_pattern: np.ndarray) -> None:
        """Noise-only config adds noise but preserves shape."""
        config = AugmentationConfig(
            max_rotation_deg=0.0,
            noise_std=0.1,
            brightness_range=0.0,
            contrast_range=0.0,
            flip_horizontal=False,
        )
        result = augment_pattern(norm_pattern, config, rng=np.random.default_rng(0))
        diff = np.abs(result - norm_pattern)
        assert diff.mean() > 0.01, "Noise should visibly change the pattern"
        assert result.shape == norm_pattern.shape

    def test_brightness_only(self, norm_pattern: np.ndarray) -> None:
        """Brightness jitter shifts the mean."""
        config = AugmentationConfig(
            max_rotation_deg=0.0,
            noise_std=0.0,
            brightness_range=0.1,
            contrast_range=0.0,
            flip_horizontal=False,
        )
        result = augment_pattern(norm_pattern, config, rng=np.random.default_rng(3))
        # Mean should shift
        mean_diff = abs(result.mean() - norm_pattern.mean())
        assert mean_diff > 0, "Brightness should shift the mean"

    def test_contrast_only(self, norm_pattern: np.ndarray) -> None:
        """Contrast jitter changes the std."""
        config = AugmentationConfig(
            max_rotation_deg=0.0,
            noise_std=0.0,
            brightness_range=0.0,
            contrast_range=0.1,
            flip_horizontal=False,
        )
        result = augment_pattern(norm_pattern, config, rng=np.random.default_rng(5))
        std_diff = abs(result.std() - norm_pattern.std())
        assert std_diff > 0, "Contrast should change the std"

    def test_flip_only(self, norm_pattern: np.ndarray) -> None:
        """Horizontal flip reverses columns."""
        config = AugmentationConfig(
            max_rotation_deg=0.0,
            noise_std=0.0,
            brightness_range=0.0,
            contrast_range=0.0,
            flip_horizontal=True,
        )
        # Run many seeds until we get a flip (50% chance per call)
        found_flip = False
        for seed in range(20):
            result = augment_pattern(
                norm_pattern, config, rng=np.random.default_rng(seed)
            )
            if np.allclose(result, norm_pattern[:, ::-1]):
                found_flip = True
                break
        assert found_flip, "Should find at least one flip in 20 trials"

    def test_various_sizes(self, aug_config: AugmentationConfig) -> None:
        """Augmentation works on various pattern sizes."""
        rng = np.random.default_rng(0)
        for h, w in [(60, 60), (80, 60), (128, 128), (64, 64)]:
            pattern = rng.standard_normal((h, w)).astype(np.float32)
            result = augment_pattern(pattern, aug_config, rng=np.random.default_rng(0))
            assert result.shape == (h, w)
            assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# EDS augmentation tests
# ---------------------------------------------------------------------------


class TestAugmentEds:
    """Tests for EDS vector augmentation."""

    def test_output_shape_unchanged(
        self, eds_vector: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Augmented EDS has the same shape."""
        result = augment_eds(eds_vector, aug_config, rng=np.random.default_rng(0))
        assert result.shape == eds_vector.shape

    def test_output_dtype_float32(
        self, eds_vector: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Augmented EDS is float32."""
        result = augment_eds(eds_vector, aug_config, rng=np.random.default_rng(0))
        assert result.dtype == np.float32

    def test_disabled_returns_copy(
        self, eds_vector: np.ndarray, disabled_config: AugmentationConfig,
    ) -> None:
        """When disabled, returns an identical copy."""
        result = augment_eds(eds_vector, disabled_config)
        np.testing.assert_array_equal(result, eds_vector)
        assert result is not eds_vector

    def test_zero_elements_stay_zero(
        self, eds_vector: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Elements that were zero (unmeasured) remain zero after augmentation."""
        result = augment_eds(eds_vector, aug_config, rng=np.random.default_rng(0))
        zero_mask = eds_vector == 0
        assert np.all(result[zero_mask] == 0), "Unmeasured elements must stay zero"

    def test_nonzero_elements_change(
        self, eds_vector: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Nonzero elements are perturbed."""
        result = augment_eds(eds_vector, aug_config, rng=np.random.default_rng(0))
        nonzero_mask = eds_vector > 0
        assert not np.allclose(
            result[nonzero_mask], eds_vector[nonzero_mask]
        ), "Nonzero elements should be perturbed"

    def test_values_non_negative(
        self, eds_vector: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """All values are non-negative after augmentation."""
        for seed in range(20):
            result = augment_eds(
                eds_vector, aug_config, rng=np.random.default_rng(seed)
            )
            assert np.all(result >= 0), f"Negative values at seed={seed}"

    def test_augmentation_is_stochastic(
        self, eds_vector: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Different seeds produce different results."""
        r1 = augment_eds(eds_vector, aug_config, rng=np.random.default_rng(1))
        r2 = augment_eds(eds_vector, aug_config, rng=np.random.default_rng(2))
        assert not np.allclose(r1, r2)

    def test_augmentation_is_reproducible(
        self, eds_vector: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Same seed produces identical results."""
        r1 = augment_eds(eds_vector, aug_config, rng=np.random.default_rng(42))
        r2 = augment_eds(eds_vector, aug_config, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_all_zero_eds_stays_zero(
        self, aug_config: AugmentationConfig,
    ) -> None:
        """An all-zero EDS vector (no EDS data) stays all zero."""
        zero_eds = np.zeros(NUM_ELEMENTS, dtype=np.float32)
        result = augment_eds(zero_eds, aug_config, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result, zero_eds)

    def test_does_not_modify_input(
        self, eds_vector: np.ndarray, aug_config: AugmentationConfig,
    ) -> None:
        """Input array is not modified in place."""
        original = eds_vector.copy()
        augment_eds(eds_vector, aug_config, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(eds_vector, original)


# ---------------------------------------------------------------------------
# AugmentationConfig tests
# ---------------------------------------------------------------------------


class TestAugmentationConfig:
    """Tests for AugmentationConfig."""

    def test_defaults(self) -> None:
        """Default config has sensible values."""
        cfg = AugmentationConfig()
        assert cfg.enabled is True
        assert cfg.max_rotation_deg == 2.0
        assert cfg.noise_std == 0.05
        assert cfg.flip_horizontal is True

    def test_from_data_config(self) -> None:
        """Conversion from DataConfig preserves values."""
        from ebsd_ai.config import DataConfig
        dcfg = DataConfig(
            augmentation_enabled=False,
            max_rotation_deg=5.0,
            noise_std=0.1,
            brightness_range=0.2,
        )
        acfg = AugmentationConfig.from_data_config(dcfg)
        assert acfg.enabled is False
        assert acfg.max_rotation_deg == 5.0
        assert acfg.noise_std == 0.1
        assert acfg.brightness_range == 0.2
