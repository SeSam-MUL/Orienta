"""Data augmentation for EBSD patterns and EDS data.

Pattern augmentations that preserve crystallographic information:
- Small random rotation (±2° by default)
- Brightness and contrast jitter
- Additive Gaussian noise
- Random horizontal flip (optional, disable for non-centrosymmetric phases)

EDS augmentations:
- Small Gaussian noise on concentrations (simulating measurement variation)

All augmentations operate on normalized float32 patterns (zero-mean,
unit-variance) and return the same format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

from ebsd_ai.config import DataConfig


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation.

    Parameters
    ----------
    enabled : bool
        Master switch for all augmentations.
    max_rotation_deg : float
        Maximum random rotation angle in degrees (applied ± uniformly).
    noise_std : float
        Standard deviation of additive Gaussian noise for patterns.
    brightness_range : float
        Maximum brightness shift as fraction of pattern std.
    contrast_range : float
        Maximum contrast scaling factor deviation from 1.0.
        E.g. 0.1 means contrast scales uniformly in [0.9, 1.1].
    flip_horizontal : bool
        Whether to apply random horizontal flips.
    eds_noise_std : float
        Standard deviation of Gaussian noise added to EDS concentrations
        (in atomic percent units).
    """

    enabled: bool = True
    max_rotation_deg: float = 2.0
    noise_std: float = 0.05
    brightness_range: float = 0.1
    contrast_range: float = 0.1
    flip_horizontal: bool = True
    eds_noise_std: float = 0.5

    @classmethod
    def from_data_config(cls, cfg: DataConfig) -> AugmentationConfig:
        """Create from a ``DataConfig`` instance."""
        return cls(
            enabled=cfg.augmentation_enabled,
            max_rotation_deg=cfg.max_rotation_deg,
            noise_std=cfg.noise_std,
            brightness_range=cfg.brightness_range,
        )


# ---------------------------------------------------------------------------
# Pattern augmentations
# ---------------------------------------------------------------------------


def augment_pattern(
    pattern: np.ndarray,
    config: AugmentationConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random augmentations to a normalized pattern.

    Parameters
    ----------
    pattern : np.ndarray
        2-D float32 pattern (expected zero-mean, unit-variance).
    config : AugmentationConfig
        Augmentation settings.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Augmented float32 pattern of the same shape.
    """
    if not config.enabled:
        return np.array(pattern, copy=True)

    if rng is None:
        rng = np.random.default_rng()

    result = pattern.astype(np.float32, copy=True)

    # Random rotation
    if config.max_rotation_deg > 0:
        result = _random_rotation(result, config.max_rotation_deg, rng)

    # Brightness jitter (additive shift)
    if config.brightness_range > 0:
        result = _brightness_jitter(result, config.brightness_range, rng)

    # Contrast jitter (multiplicative scaling)
    if config.contrast_range > 0:
        result = _contrast_jitter(result, config.contrast_range, rng)

    # Additive Gaussian noise
    if config.noise_std > 0:
        result = _additive_noise(result, config.noise_std, rng)

    # Horizontal flip
    if config.flip_horizontal and rng.random() < 0.5:
        result = np.ascontiguousarray(result[:, ::-1])

    return result


def _random_rotation(
    pattern: np.ndarray,
    max_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rotate pattern by a small random angle."""
    angle = float(rng.uniform(-max_deg, max_deg))
    rotated = scipy_rotate(
        pattern,
        angle,
        reshape=False,
        order=1,
        mode="reflect",
    )
    return np.asarray(rotated, dtype=np.float32)


def _brightness_jitter(
    pattern: np.ndarray,
    brightness_range: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add a random brightness offset."""
    shift = float(rng.uniform(-brightness_range, brightness_range))
    return pattern + shift


def _contrast_jitter(
    pattern: np.ndarray,
    contrast_range: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scale contrast around the mean."""
    factor = float(rng.uniform(1.0 - contrast_range, 1.0 + contrast_range))
    mean = pattern.mean()
    return np.asarray((pattern - mean) * factor + mean, dtype=np.float32)


def _additive_noise(
    pattern: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise to the pattern."""
    noise = rng.normal(0.0, noise_std, size=pattern.shape).astype(np.float32)
    return pattern + noise


# ---------------------------------------------------------------------------
# EDS augmentations
# ---------------------------------------------------------------------------


def augment_eds(
    eds_vector: np.ndarray,
    config: AugmentationConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply noise augmentation to an EDS concentration vector.

    Non-zero concentrations get small additive Gaussian noise to simulate
    measurement variation. Values are clamped to be non-negative.
    Zero-valued elements (not measured) remain zero.

    Parameters
    ----------
    eds_vector : np.ndarray
        Float32 array of shape ``(NUM_ELEMENTS,)`` with atomic percent values.
    config : AugmentationConfig
        Augmentation settings.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Augmented float32 EDS vector of the same shape.
    """
    if not config.enabled or config.eds_noise_std <= 0:
        return np.array(eds_vector, copy=True)

    if rng is None:
        rng = np.random.default_rng()

    result = eds_vector.astype(np.float32, copy=True)

    # Only add noise to elements that have measured values
    nonzero_mask = result > 0
    if nonzero_mask.any():
        noise = rng.normal(
            0.0, config.eds_noise_std, size=result.shape
        ).astype(np.float32)
        result[nonzero_mask] += noise[nonzero_mask]
        # Clamp to non-negative
        np.clip(result, 0.0, None, out=result)

    return result
