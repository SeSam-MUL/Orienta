"""
Recrystallization (RX) Analysis Module for EBSD Data.

Implements:
- Grain-average KAM (gKAM) calculation
- RX grain classification (GOS/gBC/gKAM criteria)
- RX fraction calculation
- Area-weighted GOS and gKAM histograms

Reference: MATLAB/MTEX RxxAnalysis.m
Architecture: the MTEX-equivalence design notes §2.5

The RX analysis is made on ONE phase, the RX phase (``resolve_rx_phase``: the phase with the most indexed pixels of the
map by default, or a phase id the caller names). The script does the same (``ph = 'Aluminium'``,
``RXedGrains = RXedGrains(ph)``, ``RXedGrainFraction = sum(RXedGrains(ph).numPixel) / sum(grains_smoothed(ph).numPixel)``):
a grain of another phase is never RX, its pixels are not in the denominator of the RX fraction, and the RX map draws it
neutral. The grains' average band contrast is still scaled by the largest gBC of ALL grains, as the script scales it
(``gBC_smoothed > 0.7 * max(gBC_smoothed)``).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    from scipy.ndimage import labeled_comprehension
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


NO_GRAIN = -1     # class of a pixel without a grain in ``classify_rx``
RX_BY_ANALYSIS = 3   # class of a pixel in a grain that ``analyze_recrystallization`` classified as RX

# The four RX criteria under the names the page, the routes and the batch use for them, with the values of the
# script (RxxAnalysis.m:63-67) and the range each is accepted in (inclusive; the unit is in the comment).
# ``analyze_recrystallization`` takes them as ``gos_threshold_deg``, ``gbc_min_fraction``, ``gkam_threshold_deg``
# and ``min_grain_radius_um`` and does not check them; ``check_rx_criteria`` is for what a user typed.
DEFAULT_RX_CRITERIA = {
    "rx_gos_threshold": 1.25,    # degrees: a grain is RX below this GOS
    "rx_gbc_fraction": 0.7,      # fraction of the largest gBC: a grain is RX above this gBC
    "rx_gkam_threshold": 0.55,   # degrees: a grain is RX below this gKAM
    "rx_min_radius_um": 0.4,     # micrometres: a grain is RX above this equivalent radius
}
RX_CRITERIA_RANGES = {
    "rx_gos_threshold": (0.0, 20.0),
    "rx_gbc_fraction": (0.0, 1.0),
    "rx_gkam_threshold": (0.0, 10.0),
    "rx_min_radius_um": (0.0, 100.0),
}


def check_rx_criteria(**values) -> None:
    """Raise ``ValueError`` naming the first criterion that is not a number inside its range.

    Args:
        **values: criteria by their ``DEFAULT_RX_CRITERIA`` names; a name that is not one is an error too
    """
    for name, value in values.items():
        if name not in RX_CRITERIA_RANGES:
            raise ValueError(f"{name} is not an RX criterion; they are {sorted(RX_CRITERIA_RANGES)}")
        low, high = RX_CRITERIA_RANGES[name]
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {value!r}") from None
        if not (low <= number <= high):          # also false for NaN
            raise ValueError(f"{name} must be between {low:g} and {high:g}, got {value!r}")


@dataclass
class RXResult:
    """Results from recrystallization analysis.

    Attributes:
        rx_grain_ids: Array of grain IDs classified as RX (recrystallized)
        rx_fraction: Percentage of the RX phase's pixels in RX grains (0-100); None when the phase has no grain

        # Grain-level metrics
        g_kam: Array of grain-average KAM values (degrees) per grain
        g_bc: Array of grain-average BC values per grain

        # Summary metrics (for Excel export)
        high_bc_fraction: Percentage of the RX phase's pixels with BC > 0.7*max(BC of all pixels); None without pixels
        low_gkam_fraction: Percentage of grain pixels with gKAM < 0.55°; None without grain pixels
        low_kam_fraction: Percentage of all pixels with KAM < 0.55°; None without a finite KAM

        # Histograms
        gos_histogram: Tuple of (counts, bin_centers) for area-weighted GOS
        gkam_histogram: Tuple of (counts, bin_centers) for area-weighted gKAM

        # Classification thresholds used
        gos_threshold: float
        gbc_min_fraction: float
        gkam_threshold: float
        min_grain_radius_um: float

        # The phase of the analysis
        phase_id, phase_name: the RX phase (``resolve_rx_phase``)

        # Denominator of the area-weighted histograms
        total_grain_area: Sum of the cell areas of all grains (um^2, closed holes included; None: not known);
            the histogram counts are cell areas, as MTEX weighs its grain histogram
    """
    rx_grain_ids: np.ndarray
    rx_fraction: Optional[float]
    g_kam: np.ndarray
    g_bc: np.ndarray
    high_bc_fraction: Optional[float]
    low_gkam_fraction: Optional[float]
    low_kam_fraction: Optional[float]
    gos_histogram: Tuple[np.ndarray, np.ndarray]
    gkam_histogram: Tuple[np.ndarray, np.ndarray]
    gos_threshold: float
    gbc_min_fraction: float
    gkam_threshold: float
    min_grain_radius_um: float
    total_grain_area: Optional[float] = None
    # the phase the analysis was made on (see the module docstring)
    phase_id: Optional[int] = None
    phase_name: Optional[str] = None


def resolve_rx_phase(dataset, rx_phase="most_measured") -> int:
    """The id of the phase the RX analysis is made on.

    ``rx_phase``: ``"most_measured"`` (default) = the phase with the most INDEXED pixels of the map (the lowest id on a
    tie; the same rule as the texture phase, ``analysis.texture_analysis.resolve_texture_phase``, but any crystal
    system is allowed), or a phase id. ``ValueError`` for anything else and for an id the map does not have; the
    message lists the ids.
    """
    from analysis.texture_analysis import MOST_MEASURED, _phase_id_of
    if isinstance(rx_phase, str) and rx_phase != MOST_MEASURED:
        raise ValueError(f"rx_phase must be a phase id or {MOST_MEASURED!r}, got {rx_phase!r}")
    return _phase_id_of(dataset, rx_phase, what="rx")


def rx_phase_info(dataset, rx_phase="most_measured") -> dict:
    """``{"id", "name"}`` of the phase ``rx_phase`` resolves to (see ``resolve_rx_phase``)."""
    from analysis.texture_analysis import _phase_label
    pid = resolve_rx_phase(dataset, rx_phase)
    return {"id": pid, "name": _phase_label(dataset, pid)}


def grain_phase_ids(dataset, grains) -> np.ndarray:
    """Phase id of each grain, length ``grains.n_grains`` (grain ``k`` at index ``k - 1``).

    ``grains.grain_phase`` when the grains carry it (every reconstruction sets it); a GrainSet built without it gets
    the phase of the pixels it covers.
    """
    n = int(grains.n_grains)
    gp = getattr(grains, "grain_phase", None)
    if gp is not None:
        return np.asarray(gp, dtype=np.int64).reshape(-1)[:n]
    out = np.full(n, -1, dtype=np.int64)
    gid = np.asarray(grains.grain_ids).reshape(-1)
    pid = np.asarray(dataset.xmap.phase_id).reshape(-1)
    sel = (gid > 0) & (pid >= 0)
    out[gid[sel] - 1] = pid[sel]
    return out


def classify_rx(dataset, grains,
                rx_threshold_deg: float = 1.5,
                sub_threshold_deg: float = 7.5) -> np.ndarray:
    """Classify each pixel by recrystallization state based on GOS per grain.

    Categories:
        -1 = No grain       (not indexed and not inside a grain, or a grain that was removed)
        0 = Recrystallized  (GOS < rx_threshold_deg)
        1 = Substructured   (rx_threshold_deg <= GOS < sub_threshold_deg)
        2 = Deformed        (GOS >= sub_threshold_deg)

    A pixel without a grain is never recrystallized; it has its own class so that a map can draw it neutral.

    Args:
        dataset: EBSDDataset instance
        grains: GrainSet instance (must have gos and grain_ids attributes)
        rx_threshold_deg: GOS threshold below which a grain is RX (default 1.5°)
        sub_threshold_deg: GOS threshold below which a grain is Substructured (default 7.5°)

    Returns:
        2-D array of shape dataset.shape with values -1, 0, 1, or 2.
    """
    n_grains = grains.n_grains
    if n_grains == 0 or grains.gos is None:
        return np.full(dataset.shape, NO_GRAIN, dtype=float)       # nothing was classified

    gos_deg = np.degrees(grains.gos)  # per-grain, length n_grains
    grain_ids = grains.grain_ids      # 2-D pixel map, 1-indexed (0 = no grain)
    grain_class = np.full(n_grains + 1, NO_GRAIN, dtype=np.int8)  # index 0: pixels without a grain
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


def rx_map_classes(dataset, grains, rx_grain_ids=None, phase_id=None) -> np.ndarray:
    """Per-pixel classes for the RX map: the GOS bands of ``classify_rx`` with the analysis' RX grains on top.

    Classes:
        -1 = no grain (``NO_GRAIN``)
         0 = GOS below 1.5 degrees, but not RX by the analysis
         1 = substructured (GOS 1.5 to 7.5 degrees)
         2 = deformed (GOS from 7.5 degrees)
         3 = RX by the analysis (``RX_BY_ANALYSIS``): the grains of ``rx_grain_ids``

    ``rx_grain_ids`` are the grain ids ``analyze_recrystallization`` returns (``RXResult.rx_grain_ids``, with the
    parameters of the panel and the workbook), so the red of the map is exactly the grains the numbers count.
    Closed not-indexed pixels carry the id of their grain and take its class. With ``rx_grain_ids=None`` (the
    analysis cannot run: no native band contrast) nothing is class 3, and the map shows the GOS bands only.

    ``phase_id``: the RX phase. The grains of every other phase are not part of the analysis and are drawn as
    ``NO_GRAIN`` (neutral), whatever their GOS. ``None`` (default) draws the grains of all phases.
    """
    data = classify_rx(dataset, grains)
    if phase_id is not None and grains.n_grains > 0:
        gphase = np.concatenate([[-1], grain_phase_ids(dataset, grains)])    # index 0: pixels without a grain
        data[gphase[np.asarray(grains.grain_ids)] != int(phase_id)] = NO_GRAIN
    if rx_grain_ids is None or len(rx_grain_ids) == 0:
        return data
    data[np.isin(grains.grain_ids, np.asarray(rx_grain_ids, dtype=int))] = RX_BY_ANALYSIS
    return data


def calculate_grain_average_edx(values: np.ndarray, grains) -> np.ndarray:
    """Grain-average of a per-pixel EDX channel over the grain's measured pixels only.

    Closed not-indexed pixels carry the grain's id but are not measurements of it
   ; chemistry read there is not the grain's.

    Args:
        values: Per-pixel channel values (any shape holding one value per grid pixel)
        grains: GrainSet with grain_ids, n_grains and (optional) closed_mask

    Returns:
        Array of length ``grains.n_grains``, NaN for a grain without a finite value.
    """
    from analysis.bc_analysis import measured_grain_ids

    return labeled_comprehension(
        np.asarray(values).ravel(),
        measured_grain_ids(grains),
        np.arange(1, grains.n_grains + 1),
        np.nanmean,
        float,
        np.nan,
    )


def calculate_grain_average_kam(kam: np.ndarray, grain_ids: np.ndarray) -> np.ndarray:
    """Calculate grain-average KAM (gKAM) for each grain.

    Args:
        kam: Pixel-wise KAM values (2D array, degrees)
        grain_ids: Grain ID per pixel (2D array, same shape as kam)

    Returns:
        Array of gKAM values per grain (length = number of grains)
        Grain IDs are assumed to be 1-indexed (ID=0 is no grain)

    Reference: RxxAnalysis.m lines 18-19 (gKAM.m helper function)
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

    Each grain's contribution to a bin is weighted by its area. ``analyze_recrystallization`` passes the cell
    area of the grain (``GrainSet.area``, closed not-indexed holes included), which is what MTEX's
    ``histogram(grains, values, edges)`` weighs by.

    Args:
        values_per_grain: Values for each grain (e.g., GOS, gKAM)
        grain_sizes: Weight of each grain (its area)
        bin_edges: Histogram bin edges (e.g., np.arange(0, 15.25, 0.25) for GOS)

    Returns:
        Tuple of (counts, bin_centers)
        - counts: Area-weighted counts per bin
        - bin_centers: Center of each bin

    Reference: RxxAnalysis.m lines 108, 121 (histogram function with grain weights)
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
                       edx_fe_channel: Optional[np.ndarray] = None,
                       in_phase: Optional[np.ndarray] = None) -> np.ndarray:
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
        in_phase: Optional boolean per grain: only these grains can be RX (the grains of the RX phase,
            ``RXedGrains = RXedGrains(ph)``). The gBC maximum still comes from all grains. None: every grain.

    Returns:
        Array of grain IDs classified as RX

    Reference: RxxAnalysis.m lines 52-67
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

    if in_phase is not None:
        rx_mask = rx_mask & np.asarray(in_phase, dtype=bool)

    # Get grain IDs that pass criteria
    # Grain IDs are 1-indexed in grain_ids array
    rx_grain_indices = np.where(rx_mask)[0]

    # Return actual grain IDs (if grains have an ID array, use that; otherwise use indices+1)
    # For now, assume grain IDs are sequential starting from 1
    rx_grain_ids = rx_grain_indices + 1

    return rx_grain_ids


def calculate_rx_fraction(rx_grain_ids: np.ndarray, grain_ids: np.ndarray,
                         phase_mask: np.ndarray) -> Optional[float]:
    """Calculate fraction of phase pixels in RX grains.

    Args:
        rx_grain_ids: Array of grain IDs classified as RX
        grain_ids: Per-pixel grain ID map (2D array)
        phase_mask: Boolean mask of pixels belonging to the phase

    Returns:
        RX fraction in percent (0-100); None when no pixel of the phase is in a grain (nothing to divide by)

    Reference: RxxAnalysis.m line 73
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
        return None

    rx_fraction = (n_rx_pixels / n_phase_pixels) * 100.0
    return rx_fraction


def analyze_recrystallization(dataset, grains, kam: np.ndarray,
                               gos_threshold_deg: float = 1.25,
                               gbc_min_fraction: float = 0.7,
                               gkam_threshold_deg: float = 0.55,
                               min_grain_radius_um: float = 0.4,
                               edx_fe_channel: Optional[str] = None,
                               rx_phase="most_measured") -> RXResult:
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
        rx_phase: the phase the analysis is made on: ``"most_measured"`` (default, the phase with the most indexed
            pixels) or a phase id (``resolve_rx_phase``). Only its grains can be RX; the RX fraction is the share of
            its pixels; "high BC" counts its pixels against 0.7 x the largest band contrast of all pixels
            (RxxAnalysis.m:47).

    Returns:
        RXResult with all metrics

    Raises:
        ValueError: ``rx_phase`` is not a phase of the map

    Reference: RxxAnalysis.m full workflow
    """
    phase = rx_phase_info(dataset, rx_phase)          # fails loud for an id the map does not have
    pid = phase["id"]

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
            rx_fraction=None,                 # no grain, nothing to divide: a missing number, not a 0 %
            g_kam=empty, g_bc=empty,
            high_bc_fraction=None, low_gkam_fraction=None, low_kam_fraction=None,
            gos_histogram=(empty_gos_hist, gos_bins),
            gkam_histogram=(empty_gkam_hist, gkam_bins),
            gos_threshold=gos_threshold_deg,
            gbc_min_fraction=gbc_min_fraction,
            gkam_threshold=gkam_threshold_deg,
            min_grain_radius_um=min_grain_radius_um,
            total_grain_area=0.0,
            phase_id=pid, phase_name=phase["name"],
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
        g_fe = calculate_grain_average_edx(fe_data, grains) if SCIPY_AVAILABLE else None

    # 4. Classify RX grains
    # only the grains of the RX phase can be RX
    grain_phase = grain_phase_ids(dataset, grains)
    rx_grain_ids = classify_rx_grains(
        grains, dataset, g_kam, g_bc,
        gos_threshold_deg, gbc_min_fraction, gkam_threshold_deg,
        min_grain_radius_um, g_fe, in_phase=(grain_phase == pid)
    )

    # 5. Calculate RX fraction: the measured pixels of the grains of the RX phase are the denominator
    grain_phase_by_pixel = np.concatenate([[-1], grain_phase])[grain_ids_2d]      # -1 where there is no grain
    phase_mask = dataset.indexed_mask_2d & (grain_phase_by_pixel == pid)
    rx_fraction = calculate_rx_fraction(rx_grain_ids, grain_ids_2d, phase_mask)

    # 6. Calculate summary metrics (for Excel export)
    # highBC as RxxAnalysis.m:47: the pixels of the phase above 0.7 x the largest band contrast of ALL pixels,
    # as a share of the pixels of the phase. A pixel without a finite band contrast is in the denominator only.
    bc_flat = np.asarray(dataset.bc, dtype=float).reshape(-1)
    bc_max = np.nanmax(bc_flat) if np.any(np.isfinite(bc_flat)) else np.nan
    phase_px = np.asarray(dataset.xmap.phase_id).reshape(-1) == pid
    n_phase_px = int(phase_px.sum())
    if n_phase_px > 0 and np.isfinite(bc_max):
        with np.errstate(invalid="ignore"):
            high_bc_fraction = float(np.sum(phase_px & (bc_flat > 0.7 * bc_max)) / n_phase_px * 100.0)
    else:
        high_bc_fraction = None

    # Low gKAM fraction (grain pixels with gKAM < 0.55°)
    low_gkam_grain_ids = np.where(g_kam < 0.55)[0] + 1  # 1-indexed
    low_gkam_pixels = np.isin(grain_ids_2d, low_gkam_grain_ids) & phase_mask
    n_phase_grain_pixels = np.sum(phase_mask & (grain_ids_2d > 0))
    low_gkam_fraction = (float(np.sum(low_gkam_pixels) / n_phase_grain_pixels * 100.0)
                         if n_phase_grain_pixels > 0 else None)

    # Low KAM fraction (indexed pixels with KAM < 0.55°)
    kam_finite = np.isfinite(kam)
    n_kam_valid = np.sum(kam_finite)
    low_kam_fraction = (float(np.sum((kam < 0.55) & kam_finite) / n_kam_valid * 100.0)
                        if n_kam_valid > 0 else None)

    # 7. Calculate histograms, every grain weighted by its cell area (MTEX: closed holes included)
    grain_area = np.asarray(grains.area, dtype=float)
    # GOS histogram (area-weighted, bin_width=0.25, range 0-15)
    gos_bin_edges = np.arange(0, 15.25, 0.25)
    gos_deg = np.degrees(grains.gos)
    gos_hist, gos_bins = calculate_area_weighted_histogram(
        gos_deg, grain_area, gos_bin_edges
    )

    # gKAM histogram (area-weighted, bin_width=0.2, range 0-5)
    gkam_bin_edges = np.arange(0, 5.2, 0.2)
    gkam_hist, gkam_bins = calculate_area_weighted_histogram(
        g_kam, grain_area, gkam_bin_edges
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
        total_grain_area=float(np.sum(grain_area)),
        phase_id=pid, phase_name=phase["name"],
    )
