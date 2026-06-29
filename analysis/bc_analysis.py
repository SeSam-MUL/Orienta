"""
BC Analysis — Band Contrast histogram and 3-Gaussian GMM fitting.

This module provides:
- calculate_bc_histogram(): Histogram of Band Contrast values
- fit_bc_gmm(): 3-Gaussian Mixture Model fit (Deformed/Recovered/RX populations)
- grain_average_bc(): Grain-average Band Contrast values

The 3-component GMM identifies three microstructural states:
- Low BC (component 1): Heavily deformed grains
- Mid BC (component 2): Recovered grains
- High BC (component 3): Recrystallized grains

Reference: MATLAB EBSDanalysis_frame.m lines 260-332
         FEAT-12 gap analysis (BC histogram + GMM)
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.ndimage import labeled_comprehension
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from analysis.ebsd_dataset import EBSDDataset, GrainSet


@dataclass
class BCHistogramResult:
    """
    Band Contrast histogram result.

    Attributes:
        bin_edges: Histogram bin edges (length n_bins + 1)
        bin_centers: Histogram bin centers (length n_bins)
        counts: Probability-normalized counts per bin (length n_bins)
        bin_width: Width of each bin
        bc_values: Raw BC values used for histogram (for validation)

    Reference: MATLAB EBSDanalysis_frame.m lines 260-273
    """
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts: np.ndarray
    bin_width: float
    bc_values: np.ndarray


@dataclass
class BCGMMResult:
    """
    3-Gaussian Mixture Model fit result for Band Contrast.

    Components are sorted by mean BC (ascending):
    - Component 0: Low BC (deformed)
    - Component 1: Mid BC (recovered)
    - Component 2: High BC (recrystallized)

    Attributes:
        weights: Component weights (area fractions), length 3
        means: Component means (BC centers), length 3
        stds: Component standard deviations, length 3

        bc_low_center: Mean BC of deformed component
        bc_low_frac: Area fraction of deformed component
        bc_mid_center: Mean BC of recovered component
        bc_mid_frac: Area fraction of recovered component
        bc_high_center: Mean BC of recrystallized component
        bc_high_frac: Area fraction of recrystallized component

        n_samples: Number of BC values used for fit
        converged: Whether GMM fit converged

    Reference: MATLAB EBSDanalysis_frame.m lines 275-331
    """
    weights: np.ndarray  # length 3
    means: np.ndarray    # length 3
    stds: np.ndarray     # length 3

    bc_low_center: float
    bc_low_frac: float
    bc_mid_center: float
    bc_mid_frac: float
    bc_high_center: float
    bc_high_frac: float

    n_samples: int
    converged: bool


def calculate_bc_histogram(
    dataset: EBSDDataset,
    bin_width: float = 10.0,
    bin_range: Tuple[float, float] = (0.0, 230.0),
) -> BCHistogramResult:
    """
    Calculate Band Contrast histogram.

    Args:
        dataset: EBSDDataset with BC values
        bin_width: Width of each bin (default 10, matching MATLAB)
        bin_range: (min, max) range for bins (default (0, 230))

    Returns:
        BCHistogramResult with bin edges, centers, and probability-normalized counts

    Reference: MATLAB EBSDanalysis_frame.m lines 260-273
    """
    # Get BC values, remove NaN/Inf
    bc = dataset.bc
    bc = bc[np.isfinite(bc)]

    # Create bin edges
    bin_edges = np.arange(bin_range[0], bin_range[1] + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Calculate histogram (probability normalized)
    counts, _ = np.histogram(bc, bins=bin_edges, density=False)
    total = counts.sum()
    counts = counts / total if total > 0 else counts.astype(float)

    return BCHistogramResult(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts=counts,
        bin_width=bin_width,
        bc_values=bc,
    )


def fit_bc_gmm(
    dataset: EBSDDataset,
    n_init: int = 10,
    max_iter: int = 2000,
    reg_covar: float = 1e-6,
    random_state: int = 42,
) -> Optional[BCGMMResult]:
    """
    Fit 3-Gaussian Mixture Model to Band Contrast distribution.

    Identifies three microstructural states:
    - Low BC: Heavily deformed
    - Mid BC: Recovered
    - High BC: Recrystallized

    Component count is fixed at 3 because BCGMMResult has hard-coded
    bc_low/mid/high fields. Callers who need a different component count
    should use ``sklearn.mixture.GaussianMixture`` directly.

    Args:
        dataset: EBSDDataset with BC values
        n_init: Number of random initializations (default 10, MATLAB uses "Replicates")
        max_iter: Maximum iterations (default 2000)
        reg_covar: Regularization value for covariance (default 1e-6)
        random_state: Random seed for reproducibility

    Returns:
        BCGMMResult with component parameters, or None if fit fails

    Reference: MATLAB EBSDanalysis_frame.m lines 275-331
    """
    N_COMPONENTS = 3
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for GMM fitting. Install with: pip install scikit-learn")

    # Get BC values, remove NaN/Inf
    bc = dataset.bc
    bc = bc[np.isfinite(bc)]

    if len(bc) < 10:
        return None  # Not enough data

    # Clamp BC to valid range (MATLAB uses bcClamped)
    bc = np.clip(bc, 0, 255)

    # Reshape for sklearn (needs 2D array)
    X = bc.reshape(-1, 1)

    # Fit GMM
    try:
        gmm = GaussianMixture(
            n_components=N_COMPONENTS,
            covariance_type='spherical',  # Single variance per component
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        gmm.fit(X)

        # Extract parameters
        weights = gmm.weights_
        means = gmm.means_.ravel()
        # For spherical covariance, gmm.covariances_ is 1D array of variances
        stds = np.sqrt(gmm.covariances_)

        # Sort by mean BC (ascending): low → mid → high
        sort_idx = np.argsort(means)
        weights = weights[sort_idx]
        means = means[sort_idx]
        stds = stds[sort_idx]

        # Create result
        result = BCGMMResult(
            weights=weights,
            means=means,
            stds=stds,
            bc_low_center=means[0],
            bc_low_frac=weights[0],
            bc_mid_center=means[1],
            bc_mid_frac=weights[1],
            bc_high_center=means[2],
            bc_high_frac=weights[2],
            n_samples=len(bc),
            converged=gmm.converged_,
        )

        return result

    except Exception as e:
        logger.warning("BC GMM fit failed: %s", e)
        return None


def grain_average_bc(
    dataset: EBSDDataset,
    grains: GrainSet,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate grain-average Band Contrast values.

    Args:
        dataset: EBSDDataset with BC values
        grains: GrainSet with grain_ids

    Returns:
        Tuple of (gBC, grain_ids):
        - gBC: Grain-average BC values (length n_grains)
        - grain_ids: Corresponding grain IDs (1..n_grains)

    Reference: MATLAB gBC.m (grain-average accumarray pattern)
    """
    if not SCIPY_AVAILABLE:
        # Fallback: manual calculation
        bc_flat = dataset.bc
        grain_ids_flat = grains.grain_ids.ravel()

        unique_grain_ids = np.arange(1, grains.n_grains + 1)
        gBC = np.zeros(grains.n_grains)

        for i, gid in enumerate(unique_grain_ids):
            mask = grain_ids_flat == gid
            bc_grain = bc_flat[mask]
            bc_grain = bc_grain[np.isfinite(bc_grain)]
            if len(bc_grain) > 0:
                gBC[i] = np.mean(bc_grain)
            else:
                gBC[i] = np.nan

        return gBC, unique_grain_ids

    # Fast scipy implementation
    bc_flat = dataset.bc
    grain_ids_flat = grains.grain_ids.ravel()

    # Only consider indexed pixels with valid grain IDs
    valid_mask = (grain_ids_flat > 0) & np.isfinite(bc_flat)
    bc_valid = bc_flat[valid_mask]
    gids_valid = grain_ids_flat[valid_mask]

    unique_grain_ids = np.arange(1, grains.n_grains + 1)

    # Use labeled_comprehension for grain-average
    gBC = labeled_comprehension(
        bc_valid,
        gids_valid,
        unique_grain_ids,
        np.nanmean,
        float,
        np.nan,
    )

    return gBC, unique_grain_ids
