"""
Recrystallization (RX) Analysis Module for EBSD Data.

Implements:
- Grain-average KAM (gKAM) calculation
- RX grain classification (GOS/gBC/gKAM criteria)
- RX fraction calculation
- Area-weighted GOS and gKAM histograms

Reference: MATLAB/MTEX RxxAnalysis.m
Architecture: the MTEX-equivalence design notes §2.5
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    from scipy.ndimage import labeled_comprehension
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class RXResult:
    """Results from recrystallization analysis.

    Attributes:
        rx_grain_ids: Array of grain IDs classified as RX (recrystallized)
        rx_fraction: Percentage of phase pixels in RX grains (0-100)

        # Grain-level metrics
        g_kam: Array of grain-average KAM values (degrees) per grain
        g_bc: Array of grain-average BC values per grain

        # Summary metrics (for Excel export)
        high_bc_fraction: Percentage of pixels with BC > 0.7*max(BC)
        low_gkam_fraction: Percentage of grain pixels with gKAM < 0.55°
        low_kam_fraction: Percentage of all pixels with KAM < 0.55°

        # Histograms
        gos_histogram: Tuple of (counts, bin_centers) for area-weighted GOS
        gkam_histogram: Tuple of (counts, bin_centers) for area-weighted gKAM

        # Classification thresholds used
        gos_threshold: float
        gbc_min_fraction: float
        gkam_threshold: float
        min_grain_radius_um: float
    """
    rx_grain_ids: np.ndarray
    rx_fraction: float
    g_kam: np.ndarray
    g_bc: np.ndarray
    high_bc_fraction: float
    low_gkam_fraction: float
    low_kam_fraction: float
    gos_histogram: Tuple[np.ndarray, np.ndarray]
    gkam_histogram: Tuple[np.ndarray, np.ndarray]
    gos_threshold: float
    gbc_min_fraction: float
    gkam_threshold: float
    min_grain_radius_um: float


def classify_rx(dataset, grains,
                rx_threshold_deg: float = 1.5,
                sub_threshold_deg: float = 7.5) -> np.ndarray:
    """Classify each pixel by recrystallization state based on GOS per grain.

    Categories:
        0 = Recrystallized  (GOS < rx_threshold_deg)
        1 = Substructured   (rx_threshold_deg <= GOS < sub_threshold_deg)
        2 = Deformed        (GOS >= sub_threshold_deg)

    Args:
        dataset: EBSDDataset instance
        grains: GrainSet instance (must have gos and grain_ids attributes)
        rx_threshold_deg: GOS threshold below which a grain is RX (default 1.5°)
        sub_threshold_deg: GOS threshold below which a grain is Substructured (default 7.5°)

    Returns:
        2-D array of shape dataset.shape with values 0, 1, or 2.
    """
    n_grains = grains.n_grains
    if n_grains == 0 or grains.gos is None:
        return np.zeros(dataset.shape, dtype=float)

    gos_deg = np.degrees(grains.gos)  # per-grain, length n_grains
    grain_ids = grains.grain_ids      # 2-D pixel map, 1-indexed (0 = no grain)
    grain_class = np.zeros(n_grains + 1, dtype=np.uint8)  # index 0 unused
    for idx in range(n_grains):
        g = gos_deg[idx]
        if g < rx_threshold_deg:
            grain_class[idx + 1] = 0  # RX
        elif g < sub_threshold_deg:
            grain_class[idx + 1] = 1  # Substructured
        else:
            grain_class[idx + 1] = 2  # Deformed

    # Map back to pixel grid
    data = grain_class[grain_ids]
    return data.astype(float)


def calculate_grain_average_kam(kam: np.ndarray, grain_ids: np.ndarray) -> np.ndarray:
    """Calculate grain-average KAM (gKAM) for each grain.

    Args:
        kam: Pixel-wise KAM values (2D array, degrees)
        grain_ids: Grain ID per pixel (2D array, same shape as kam)

    Returns:
        Array of gKAM values per grain (length = number of grains)
        Grain IDs are assumed to be 1-indexed (ID=0 is no grain)

    Reference: RxxAnalysis.m lines 14-15 (gKAM.m helper function)
    """
    if not SCIPY_AVAILABLE:
        # Fallback: manual loop
        unique_ids = np.unique(grain_ids[grain_ids > 0])
        g_kam = np.zeros(len(unique_ids))
        for i, gid in enumerate(unique_ids):
            mask = grain_ids == gid
            g_kam[i] = np.nanmean(kam[mask])
        return g_kam

    # Fast vectorized version
    unique_ids = np.unique(grain_ids[grain_ids > 0])
    g_kam = labeled_comprehension(
        kam.ravel(),
        grain_ids.ravel(),
        unique_ids,
        np.nanmean,
        float,
        np.nan
    )
    return g_kam


def calculate_area_weighted_histogram(values_per_grain: np.ndarray, grain_sizes: np.ndarray,
                                       bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate area-weighted histogram for grain-level values.

    Each grain's contribution to a bin is weighted by its size (number of pixels).

    Args:
        values_per_grain: Values for each grain (e.g., GOS, gKAM)
        grain_sizes: Number of pixels per grain
        bin_edges: Histogram bin edges (e.g., np.arange(0, 15.25, 0.25) for GOS)

    Returns:
        Tuple of (counts, bin_centers)
        - counts: Area-weighted counts per bin
        - bin_centers: Center of each bin

    Reference: RxxAnalysis.m lines 104, 117 (histogram function with grain weights)
    """
    # Filter NaN values
    valid = np.isfinite(values_per_grain)
    if not valid.any():
        n_bins = len(bin_edges) - 1
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        return np.zeros(n_bins), bin_centers

    # Vectorized weighted histogram
    counts, _ = np.histogram(
        values_per_grain[valid], bins=bin_edges, weights=grain_sizes[valid]
    )

    # Bin centers
    bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

    return counts.astype(float), bin_centers


def classify_rx_grains(grains, dataset, g_kam: np.ndarray, g_bc: np.ndarray,
                       gos_threshold_deg: float = 1.25,
                       gbc_min_fraction: float = 0.7,
                       gkam_threshold_deg: float = 0.55,
                       min_grain_radius_um: float = 0.4,
                       edx_fe_channel: Optional[np.ndarray] = None) -> np.ndarray:
    """Classify grains as RX (recrystallized) based on combined threshold query.

    WITHOUT EDX (Fe):
      RX = grains where:
        GOS < gos_threshold AND
        gBC > gbc_min_fraction * max(gBC) AND
        gKAM < gkam_threshold

    WITH EDX (Fe):
      RX = grains where:
        GOS < 1.15° AND
        gBC > 0.55 * max(gBC) AND
        gFe < 2.2 * mean(gFe) AND
        gKAM < 0.7°
      Then additionally remove: gFe > mean(gFe)

    Args:
        grains: GrainSet instance (must have gos attribute)
        dataset: EBSDDataset instance
        g_kam: Grain-average KAM (degrees) per grain
        g_bc: Grain-average BC per grain
        gos_threshold_deg: Maximum GOS for RX classification (degrees)
        gbc_min_fraction: Minimum gBC as fraction of max(gBC)
        gkam_threshold_deg: Maximum gKAM for RX classification (degrees)
        min_grain_radius_um: Minimum grain radius for RX classification (µm)
        edx_fe_channel: Optional grain-average Fe counts (if available)

    Returns:
        Array of grain IDs classified as RX

    Reference: RxxAnalysis.m lines 48-63
    """
    n_grains = grains.n_grains

    # GOS is stored in radians (grain_analysis.py line 251) — always convert
    gos_deg = np.degrees(grains.gos)

    # Grain radii (from area)
    grain_radii_um = np.sqrt(grains.area / np.pi)

    # Thresholds — guard against all-NaN arrays
    g_bc_max = np.nanmax(g_bc) if np.any(np.isfinite(g_bc)) else 1.0
    gbc_threshold = gbc_min_fraction * g_bc_max

    # Boolean mask for RX classification
    if edx_fe_channel is not None:
        # WITH EDX criteria (stricter)
        g_fe_mean = np.nanmean(edx_fe_channel)
        rx_mask = (
            (gos_deg < 1.15) &
            (g_bc > 0.55 * g_bc_max) &
            (edx_fe_channel < 2.2 * g_fe_mean) &
            (g_kam < 0.7) &
            (grain_radii_um > min_grain_radius_um)
        )

        # Additional removal: gFe > mean (line 56)
        rx_mask = rx_mask & (edx_fe_channel <= g_fe_mean)
    else:
        # WITHOUT EDX criteria (standard)
        rx_mask = (
            (gos_deg < gos_threshold_deg) &
            (g_bc > gbc_threshold) &
            (g_kam < gkam_threshold_deg) &
            (grain_radii_um > min_grain_radius_um)
        )

    # Get grain IDs that pass criteria
    # Grain IDs are 1-indexed in grain_ids array
    rx_grain_indices = np.where(rx_mask)[0]

    # Return actual grain IDs (if grains have an ID array, use that; otherwise use indices+1)
    # For now, assume grain IDs are sequential starting from 1
    rx_grain_ids = rx_grain_indices + 1

    return rx_grain_ids


def calculate_rx_fraction(rx_grain_ids: np.ndarray, grain_ids: np.ndarray,
                         phase_mask: np.ndarray) -> float:
    """Calculate fraction of phase pixels in RX grains.

    Args:
        rx_grain_ids: Array of grain IDs classified as RX
        grain_ids: Per-pixel grain ID map (2D array)
        phase_mask: Boolean mask of pixels belonging to the phase

    Returns:
        RX fraction in percent (0-100)

    Reference: RxxAnalysis.m line 69
    ```matlab
    RXedGrainFraction=sum(RXedGrains(ph).numPixel)/sum(grains_smoothed(ph).numPixel)*100
    ```
    """
    # Count pixels in RX grains — ensure shapes match
    grain_flat = grain_ids.ravel()
    mask_flat = np.asarray(phase_mask).ravel()
    rx_pixels = np.isin(grain_flat, rx_grain_ids) & mask_flat
    n_rx_pixels = np.sum(rx_pixels)

    # Count total phase pixels in grains (exclude grain_id=0)
    phase_grain_pixels = mask_flat & (grain_flat > 0)
    n_phase_pixels = np.sum(phase_grain_pixels)

    if n_phase_pixels == 0:
        return 0.0

    rx_fraction = (n_rx_pixels / n_phase_pixels) * 100.0
    return rx_fraction


def analyze_recrystallization(dataset, grains, kam: np.ndarray,
                               gos_threshold_deg: float = 1.25,
                               gbc_min_fraction: float = 0.7,
                               gkam_threshold_deg: float = 0.55,
                               min_grain_radius_um: float = 0.4,
                               edx_fe_channel: Optional[str] = None) -> RXResult:
    """Full recrystallization analysis workflow.

    Args:
        dataset: EBSDDataset instance
        grains: GrainSet instance (must have gos attribute)
        kam: Pixel-wise KAM values (2D array, degrees)
        gos_threshold_deg: Maximum GOS for RX classification (degrees)
        gbc_min_fraction: Minimum gBC as fraction of max(gBC)
        gkam_threshold_deg: Maximum gKAM for RX classification (degrees)
        min_grain_radius_um: Minimum grain radius for RX classification (µm)
        edx_fe_channel: Optional EDX Fe channel name (if available in dataset)

    Returns:
        RXResult with all metrics

    Reference: RxxAnalysis.m full workflow
    """
    # Handle zero-grain case gracefully
    if grains.n_grains == 0 or grains.gos is None:
        empty = np.array([], dtype=float)
        gos_bin_edges = np.arange(0, 15.25, 0.25)
        gkam_bin_edges = np.arange(0, 5.2, 0.2)
        empty_gos_hist = np.zeros(len(gos_bin_edges) - 1)
        empty_gkam_hist = np.zeros(len(gkam_bin_edges) - 1)
        gos_bins = gos_bin_edges[:-1] + 0.125
        gkam_bins = gkam_bin_edges[:-1] + 0.1
        return RXResult(
            rx_grain_ids=np.array([], dtype=int),
            rx_fraction=0.0,
            g_kam=empty, g_bc=empty,
            high_bc_fraction=0.0, low_gkam_fraction=0.0, low_kam_fraction=0.0,
            gos_histogram=(empty_gos_hist, gos_bins),
            gkam_histogram=(empty_gkam_hist, gkam_bins),
            gos_threshold=gos_threshold_deg,
            gbc_min_fraction=gbc_min_fraction,
            gkam_threshold=gkam_threshold_deg,
            min_grain_radius_um=min_grain_radius_um,
        )

    # 1. Calculate grain-average KAM
    grain_ids_2d = grains.grain_ids
    g_kam = calculate_grain_average_kam(kam, grain_ids_2d)

    # 2. Calculate grain-average BC
    from analysis.bc_analysis import grain_average_bc
    g_bc, _ = grain_average_bc(dataset, grains)

    # 3. Extract grain-average Fe (if available)
    g_fe = None
    if edx_fe_channel is not None and edx_fe_channel in dataset.edx_channels:
        fe_data = dataset.edx_channels[edx_fe_channel]
        g_fe = labeled_comprehension(
            fe_data.ravel(),
            grain_ids_2d.ravel(),
            np.arange(1, grains.n_grains + 1),
            np.nanmean,
            float,
            np.nan
        ) if SCIPY_AVAILABLE else None

    # 4. Classify RX grains
    rx_grain_ids = classify_rx_grains(
        grains, dataset, g_kam, g_bc,
        gos_threshold_deg, gbc_min_fraction, gkam_threshold_deg,
        min_grain_radius_um, g_fe
    )

    # 5. Calculate RX fraction
    phase_mask = dataset.indexed_mask_2d
    rx_fraction = calculate_rx_fraction(rx_grain_ids, grain_ids_2d, phase_mask)

    # 6. Calculate summary metrics (for Excel export)
    bc_flat = dataset.bc
    bc_max = np.nanmax(bc_flat)
    n_indexed = dataset.indexed_mask.sum()
    high_bc_fraction = (np.sum(bc_flat > 0.7 * bc_max) / n_indexed * 100.0) if n_indexed > 0 else 0.0

    # Low gKAM fraction (grain pixels with gKAM < 0.55°)
    low_gkam_grain_ids = np.where(g_kam < 0.55)[0] + 1  # 1-indexed
    low_gkam_pixels = np.isin(grain_ids_2d, low_gkam_grain_ids) & phase_mask
    n_phase_grain_pixels = np.sum(phase_mask & (grain_ids_2d > 0))
    low_gkam_fraction = (np.sum(low_gkam_pixels) / n_phase_grain_pixels * 100.0) if n_phase_grain_pixels > 0 else 0.0

    # Low KAM fraction (indexed pixels with KAM < 0.55°)
    kam_finite = np.isfinite(kam)
    n_kam_valid = np.sum(kam_finite)
    low_kam_fraction = (np.sum((kam < 0.55) & kam_finite) / n_kam_valid * 100.0) if n_kam_valid > 0 else 0.0

    # 7. Calculate histograms
    # GOS histogram (area-weighted, bin_width=0.25, range 0-15)
    gos_bin_edges = np.arange(0, 15.25, 0.25)
    gos_deg = np.degrees(grains.gos)
    gos_hist, gos_bins = calculate_area_weighted_histogram(
        gos_deg, grains.grain_size, gos_bin_edges
    )

    # gKAM histogram (area-weighted, bin_width=0.2, range 0-5)
    gkam_bin_edges = np.arange(0, 5.2, 0.2)
    gkam_hist, gkam_bins = calculate_area_weighted_histogram(
        g_kam, grains.grain_size, gkam_bin_edges
    )

    return RXResult(
        rx_grain_ids=rx_grain_ids,
        rx_fraction=rx_fraction,
        g_kam=g_kam,
        g_bc=g_bc,
        high_bc_fraction=high_bc_fraction,
        low_gkam_fraction=low_gkam_fraction,
        low_kam_fraction=low_kam_fraction,
        gos_histogram=(gos_hist, gos_bins),
        gkam_histogram=(gkam_hist, gkam_bins),
        gos_threshold=gos_threshold_deg,
        gbc_min_fraction=gbc_min_fraction,
        gkam_threshold=gkam_threshold_deg,
        min_grain_radius_um=min_grain_radius_um,
    )
