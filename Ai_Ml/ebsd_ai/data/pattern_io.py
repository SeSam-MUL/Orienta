"""Pattern loading, resizing, and normalization.

Functions for preprocessing EBSD diffraction patterns:
- Resize any input size to the model's target size (default 128x128)
- Normalize to zero-mean, unit-variance float32
- Inverse-normalize back to original statistics
- Validate pattern arrays
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.transform import resize as sk_resize

from ebsd_ai.config import DEFAULT_PATTERN_SIZE, sanitize_array

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_pattern(pattern: np.ndarray) -> None:
    """Check that *pattern* is a valid 2-D numeric array.

    Parameters
    ----------
    pattern : np.ndarray
        Array to validate.

    Raises
    ------
    ValueError
        If the pattern is not 2-D, is empty, or contains non-finite values.
    TypeError
        If the array is not numeric.
    """
    if not isinstance(pattern, np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(pattern).__name__}"
        )
    if pattern.ndim != 2:
        raise ValueError(
            f"Pattern must be 2-D, got {pattern.ndim}-D with shape {pattern.shape}"
        )
    if pattern.size == 0:
        raise ValueError("Pattern is empty (zero-size array)")
    if not np.issubdtype(pattern.dtype, np.number):
        raise TypeError(
            f"Pattern must have a numeric dtype, got {pattern.dtype}"
        )


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


def resize_pattern(
    pattern: np.ndarray,
    target_size: int = DEFAULT_PATTERN_SIZE,
) -> np.ndarray:
    """Resize a pattern to a square ``(target_size, target_size)`` image.

    Parameters
    ----------
    pattern : np.ndarray
        2-D grayscale pattern of any size.
    target_size : int
        Side length of the output square image.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(target_size, target_size)`` with values
        in [0, 1] range (scikit-image convention).
    """
    validate_pattern(pattern)
    # skimage.resize normalizes to [0,1] for integer inputs automatically
    resized = sk_resize(
        pattern,
        (target_size, target_size),
        anti_aliasing=True,
        preserve_range=False,
    )
    result: np.ndarray = resized.astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@dataclass
class NormStats:
    """Statistics used for normalization, needed for inverse transform.

    Parameters
    ----------
    mean : float
        Mean of the original pattern.
    std : float
        Standard deviation of the original pattern.
    """

    mean: float
    std: float


def normalize_pattern(pattern: np.ndarray) -> tuple[np.ndarray, NormStats]:
    """Normalize a pattern to zero-mean, unit-variance float32.

    Parameters
    ----------
    pattern : np.ndarray
        2-D pattern (any dtype, any size — typically the output of
        ``resize_pattern``).

    Returns
    -------
    normalized : np.ndarray
        Float32 array with mean ~0 and std ~1.
    stats : NormStats
        Original mean and std for later inverse transform.
    """
    validate_pattern(pattern)
    arr = sanitize_array(
        pattern.astype(np.float64), name="pattern (normalize)"
    )
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        # Constant pattern — return zeros
        normalized = np.zeros_like(arr, dtype=np.float32)
    else:
        normalized = ((arr - mean) / std).astype(np.float32)
    return normalized, NormStats(mean=mean, std=std)


def denormalize_pattern(
    normalized: np.ndarray,
    stats: NormStats,
) -> np.ndarray:
    """Inverse of ``normalize_pattern``.

    Parameters
    ----------
    normalized : np.ndarray
        Zero-mean, unit-variance pattern.
    stats : NormStats
        Original statistics from ``normalize_pattern``.

    Returns
    -------
    np.ndarray
        Float64 array with original mean and variance restored.
    """
    return (normalized.astype(np.float64) * stats.std + stats.mean)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def prepare_pattern(
    pattern: np.ndarray,
    target_size: int = DEFAULT_PATTERN_SIZE,
) -> tuple[np.ndarray, NormStats]:
    """Resize and normalize a pattern in one step.

    Parameters
    ----------
    pattern : np.ndarray
        Raw 2-D pattern of any size and dtype.
    target_size : int
        Square output size.

    Returns
    -------
    normalized : np.ndarray
        Float32 ``(target_size, target_size)`` with zero-mean, unit-variance.
    stats : NormStats
        For inverse transform.
    """
    resized = resize_pattern(pattern, target_size)
    return normalize_pattern(resized)
