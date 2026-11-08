"""
Deformation Analysis Module for EBSD Data.

Implements:
- KAM (Kernel Average Misorientation) calculation
- Grain boundary classification (SAGB/HAGB)
- Grain boundary metrics and histograms
- Sphericity calculation

Reference: MATLAB/MTEX DeformationAnalysis.m
Architecture: the MTEX-equivalence design notes §2.3
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# The deformation parameters the page, the routes and the batch use, with the values of the script
# (DeformationAnalysis.m:15-17, :142-147) and the range each is accepted in (the unit is in the comment).
# SAGB is 3 degrees up to the HAGB limit, so the HAGB limit is the only boundary parameter.
DEFAULT_DEFORMATION_PARAMS = {
    "kam_order": 2,              # neighbour order of the KAM (|dy| + |dx| <= order)
    "kam_threshold_deg": 7.0,    # degrees: neighbours further apart are not part of the KAM
    "hagb_min_deg": 15.0,        # degrees: a boundary from this angle is HAGB, from 3 degrees up to it SAGB
}
SAGB_MIN_DEG = 3.0


@dataclass
class DeformationResult:
    """Results from deformation analysis.

    Attributes:
        kam: KAM map (ny, nx) in degrees
        kam_hist_counts: Histogram counts (probability normalized)
        kam_hist_edges: Histogram bin edges
        kam_order: KAM calculation order (2 = 2nd nearest neighbors)
        kam_threshold: KAM threshold in degrees (7.0)

        # Grain boundary metrics (None when the reconstruction found no grain: a missing number, not a 0)
        total_gb_length: Total grain boundary length (µm)
        hagb_length: High-angle GB length (from ``hagb_min``, 15° by default) (µm)
        sagb_length: Sub-grain/low-angle GB length (3° up to ``hagb_min``) (µm)
        gb_per_phase_area: GB length per phase area (µm/µm²)
        gb_per_field_area: GB length per field area (µm/µm²)
        hagb_per_phase_area: HAGB length per phase area
        hagb_per_field_area: HAGB length per field area
        sagb_per_phase_area: SAGB length per phase area
        sagb_per_field_area: SAGB length per field area

        # Grain boundary misorientation histogram
        gb_seg_hist_counts: Segment length histogram (µm) for miso angles 3-63°
        gb_seg_hist_bins: Bin centers (3, 5, 7, ..., 63 degrees)

        # Sphericity per grain
        sphericity: Sphericity per grain, area / (perimeter including inclusions * equivalent radius)
            as in the MATLAB reference; 0.5 for a smooth circle (pi r^2 / (2 pi r * r)), pi / 8 for a
            disc of square pixels (its staircase outline is 8 r long)
    """
    kam: np.ndarray
    kam_hist_counts: np.ndarray
    kam_hist_edges: np.ndarray
    kam_order: int
    kam_threshold: float

    total_gb_length: float
    hagb_length: float
    sagb_length: float
    gb_per_phase_area: float
    gb_per_field_area: float
    hagb_per_phase_area: float
    hagb_per_field_area: float
    sagb_per_phase_area: float
    sagb_per_field_area: float

    gb_seg_hist_counts: np.ndarray
    gb_seg_hist_bins: np.ndarray

    sphericity: np.ndarray

    # the HAGB limit the boundary classification used (degrees); SAGB runs from 3 degrees up to it
    hagb_min: float = DEFAULT_DEFORMATION_PARAMS["hagb_min_deg"]


def check_deformation_params(**values) -> None:
    """Raise ``ValueError`` naming the first deformation parameter that is not a number inside its range.

    Args:
        **values: ``kam_order`` (integer, 1 to 5), ``kam_threshold_deg`` (more than 0, at most 30) and
            ``hagb_min_deg`` (more than 3, at most 90), by the names of ``DEFAULT_DEFORMATION_PARAMS``
    """
    ranges = {"kam_order": "1 to 5", "kam_threshold_deg": "more than 0 and at most 30",
              "hagb_min_deg": "more than 3 and at most 90"}
    for name, value in values.items():
        if name not in ranges:
            raise ValueError(f"{name} is not a deformation parameter; they are {sorted(ranges)}")
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if name == "kam_order":
            ok = isinstance(value, (int, np.integer)) and 1 <= value <= 5
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a number, got {value!r}") from None
            ok = (0 < number <= 30) if name == "kam_threshold_deg" else (SAGB_MIN_DEG < number <= 90)
        if not ok:
            raise ValueError(f"{name} must be {ranges[name]}, got {value!r}")


def calculate_kam(dataset, order: int = 2, threshold_deg: float = 7.0, grains=None) -> np.ndarray:
    """Kernel average misorientation per pixel (degrees).

    Mean disorientation to the neighbours up to ``order`` steps (|dy| + |dx| <=
    order) that are below ``threshold_deg``, counting only neighbours of the
    same phase that are measurements and, when ``grains`` (a GrainSet) is
    given, lie in the same grain and were not culled. NaN where no neighbour
    counts and on every pixel that is not a measurement.

    Reference: DeformationAnalysis.m lines 142-147 (KAM after calcGrains).
    """
    from analysis.grain_reconstruction import grid_arrays, phase_ops
    from analysis.orientation_math import disorientation_angle

    quats, phase = grid_arrays(dataset)
    ops = phase_ops(dataset.xmap)
    ny, nx = phase.shape
    ok_px = (phase >= 0) & np.isfinite(quats).all(axis=-1)
    gid = None
    if grains is not None:
        gid = grains.grain_ids
        ok_px &= gid > 0
        if grains.culled_mask is not None:
            ok_px &= ~grains.culled_mask
    thr = np.radians(threshold_deg)
    acc = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx), dtype=np.int32)
    for dy in range(-order, order + 1):
        for dx in range(-order, order + 1):
            if (dy == 0 and dx == 0) or abs(dx) + abs(dy) > order:
                continue
            c = (slice(max(0, -dy), min(ny, ny - dy)), slice(max(0, -dx), min(nx, nx - dx)))
            b = (slice(max(0, dy), min(ny, ny + dy)), slice(max(0, dx), min(nx, nx + dx)))
            ok = ok_px[c] & ok_px[b] & (phase[c] == phase[b])
            if gid is not None:
                ok &= gid[c] == gid[b]
            ang = np.full(ok.shape, np.inf)
            pc = phase[c]
            for pid in np.unique(pc[ok]):
                if int(pid) not in ops:
                    raise ValueError(f"no point group for phase id {int(pid)}; cannot compute KAM")
                s = ok & (pc == pid)
                ang[s] = disorientation_angle(quats[c][s], quats[b][s], ops[int(pid)])
            valid = ok & (ang < thr)
            acc[c] += np.where(valid, np.degrees(np.where(valid, ang, 0.0)), 0.0)
            cnt[c] += valid
    kam = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    kam[~ok_px] = np.nan
    return kam


def calculate_kam_histogram(kam: np.ndarray, bin_width: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate KAM histogram with probability normalization.

    Args:
        kam: KAM map (ny, nx) in degrees
        bin_width: Histogram bin width in degrees (default=0.25)

    Returns:
        counts: Histogram counts (probability normalized, sums to 1.0)
        edges: Bin edges

    Reference: DeformationAnalysis.m line 147
    """
    # Filter out NaN and invalid values
    kam_valid = kam[np.isfinite(kam)]

    if len(kam_valid) == 0:
        return np.array([]), np.array([0.0])

    # Don't fabricate a distribution for a degenerate (all-zero) KAM.
    # np.arange(0, 0+bin_width, bin_width) yields just [0.0], which would
    # crash np.histogram with "bins must increase monotonically". Returning
    # a single synthetic bin would mislead plotters into rendering a bar at
    # KAM=0 as if it were a measurement. Return empty arrays so the caller
    # can decide whether to show "N/A" or skip the plot.
    kam_max = float(kam_valid.max())
    if kam_max <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Calculate histogram
    counts, edges = np.histogram(kam_valid, bins=np.arange(0, kam_max + bin_width, bin_width))

    # Normalize to probability
    total = counts.sum()
    counts = counts.astype(np.float64) / total if total > 0 else counts.astype(np.float64)

    return counts, edges


def _segment_lengths(grains, dataset) -> np.ndarray:
    """Length in um of every entry of ``grains.boundary_type``: ``grains.boundary_length`` when present,
    otherwise one step per segment (a square-grid segment of a GrainSet built without cells)."""
    n = len(np.asarray(grains.boundary_type))
    length = getattr(grains, "boundary_length", None)
    if length is None:
        return np.full(n, float(dataset.step_size))
    length = np.asarray(length, dtype=float)
    if length.shape != (n,):
        raise ValueError(f"boundary_length has {length.shape} entries for {n} boundary segments")
    return length


def classify_grain_boundaries(grains, sagb_min_deg: float = 3.0, sagb_max_deg: float = 15.0,
                              hagb_min_deg: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify grain boundaries into SAGB (sub-grain) and HAGB (high-angle).

    Args:
        grains: GrainSet instance with boundary_segments and boundary_type
        sagb_min_deg: Minimum angle for SAGB (default=3.0°)
        sagb_max_deg: Maximum angle for SAGB (default=15.0°)
        hagb_min_deg: Minimum angle for HAGB (default=15.0°)

    Returns:
        sagb_mask: Boolean mask for SAGB segments
        hagb_mask: Boolean mask for HAGB segments
        all_gb_mask: Boolean mask for all grain boundaries (>sagb_min_deg)

    Reference: DeformationAnalysis.m lines 14-18
    """
    if grains.boundary_type is None:
        raise ValueError("Grain boundary types not calculated. Run reconstruct_grains first.")

    angles = grains.boundary_type  # Misorientation angles in degrees

    sagb_mask = (angles >= sagb_min_deg) & (angles < sagb_max_deg)
    hagb_mask = angles >= hagb_min_deg
    all_gb_mask = angles >= sagb_min_deg

    return sagb_mask, hagb_mask, all_gb_mask


def calculate_gb_metrics(grains, dataset, sagb_min_deg: float = 3.0,
                        sagb_max_deg: float = 15.0, hagb_min_deg: float = 15.0) -> dict:
    """Calculate grain boundary length metrics.

    Args:
        grains: GrainSet instance
        dataset: EBSDDataset instance (for step_size and dimensions)
        sagb_min_deg: Minimum angle for SAGB (default=3.0°)
        sagb_max_deg: Maximum angle for SAGB (default=15.0°)
        hagb_min_deg: Minimum angle for HAGB (default=15.0°)

    Returns:
        Dictionary with:
            total_gb_length, hagb_length, sagb_length (in µm)
            gb_per_phase_area, gb_per_field_area (µm/µm²)
            hagb_per_phase_area, hagb_per_field_area
            sagb_per_phase_area, sagb_per_field_area

    Reference: DeformationAnalysis.m lines 14-35
    """
    sagb_mask, hagb_mask, all_gb_mask = classify_grain_boundaries(
        grains, sagb_min_deg, sagb_max_deg, hagb_min_deg
    )

    # Lengths of the boundary segments (um): the edge lengths of the grain cells when the grains carry
    # them, one pixel step per segment for a GrainSet built without
    segment_length = _segment_lengths(grains, dataset)

    # Total lengths
    total_gb_length = float(segment_length[all_gb_mask].sum())
    hagb_length = float(segment_length[hagb_mask].sum())
    sagb_length = float(segment_length[sagb_mask].sum())

    # Phase area: number of indexed pixels * step_size²
    n_phase_pixels = dataset.indexed_mask.sum()
    phase_area = n_phase_pixels * dataset.step_size ** 2  # µm²

    # Field area: total scan extent
    ny, nx = dataset.xmap.shape
    field_area = (ny * dataset.step_size) * (nx * dataset.step_size)  # µm²

    # Normalize
    gb_per_phase_area = total_gb_length / phase_area if phase_area > 0 else 0.0
    gb_per_field_area = total_gb_length / field_area if field_area > 0 else 0.0

    hagb_per_phase_area = hagb_length / phase_area if phase_area > 0 else 0.0
    hagb_per_field_area = hagb_length / field_area if field_area > 0 else 0.0

    sagb_per_phase_area = sagb_length / phase_area if phase_area > 0 else 0.0
    sagb_per_field_area = sagb_length / field_area if field_area > 0 else 0.0

    return {
        'total_gb_length': total_gb_length,
        'hagb_length': hagb_length,
        'sagb_length': sagb_length,
        'gb_per_phase_area': gb_per_phase_area,
        'gb_per_field_area': gb_per_field_area,
        'hagb_per_phase_area': hagb_per_phase_area,
        'hagb_per_field_area': hagb_per_field_area,
        'sagb_per_phase_area': sagb_per_phase_area,
        'sagb_per_field_area': sagb_per_field_area,
    }


def calculate_gb_segment_histogram(grains, dataset,
                                   min_angle: float = 3.0, max_angle: float = 63.0,
                                   bin_width: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate grain boundary segment length histogram by misorientation angle.

    Args:
        grains: GrainSet instance
        dataset: EBSDDataset instance (for step_size)
        min_angle: Minimum misorientation angle (default=3.0°)
        max_angle: Maximum misorientation angle (default=63.0°)
        bin_width: Bin width in degrees (default=2.0°)

    Returns:
        counts: Sum of segment lengths per bin (µm)
        bins: Bin centers (degrees)

    Reference: DeformationAnalysis.m lines 107-117
    """
    if grains.boundary_type is None:
        raise ValueError("Grain boundary types not calculated")

    angles = grains.boundary_type  # degrees

    # Create bin edges
    bin_edges = np.arange(min_angle, max_angle + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Sum of the segment lengths (um) per bin — vectorized
    counts, _ = np.histogram(angles, bins=bin_edges, weights=_segment_lengths(grains, dataset))
    counts = counts.astype(float)

    return counts, bin_centers


def calculate_sphericity(grains) -> np.ndarray:
    """Calculate sphericity for each grain, as the MATLAB reference does.

    Sphericity = area / perimeter / equivalent_radius, with the perimeter including inclusions
    (``GrainSet.perimeter_with_inclusions``: the outer boundary plus the boundary of every hole or
    enclosed grain, MTEX ``perimeter('withInclusion')``). A GrainSet without it falls back to the
    outer ``perimeter``, and one without any perimeter to the circle estimate 2 * pi * r.

    The value is not 1.0 for a circle: a smooth circle gives pi r^2 / (2 pi r * r) = 0.5, and a disc
    of square pixels, whose staircase outline is 8 r long, gives pi / 8 = 0.393. Smaller values mean
    a longer boundary for the same area (elongated, ragged, or with inclusions). The formula is kept
    so the numbers stay comparable with the reference Documentation.xlsx.

    Args:
        grains: GrainSet instance with area, ecd and, if available, the perimeters

    Returns:
        Sphericity array (length = n_grains)

    Reference: DeformationAnalysis.m line 54
    """
    if grains.area is None or grains.ecd is None:
        raise ValueError("Grain area and ECD not calculated")

    # Equivalent radius = ECD / 2
    equiv_radius = grains.ecd / 2.0

    # Perimeter including inclusions as in the reference; then the outer one; then the circle estimate
    perimeter = getattr(grains, 'perimeter_with_inclusions', None)
    if perimeter is None:
        perimeter = getattr(grains, 'perimeter', None)
    if perimeter is None:
        perimeter = 2 * np.pi * equiv_radius

    # Sphericity
    with np.errstate(divide='ignore', invalid='ignore'):
        sphericity = grains.area / (perimeter * equiv_radius)
        sphericity = np.where(np.isfinite(sphericity), sphericity, np.nan)

    return sphericity


def analyze_deformation(dataset, grains, kam_order: int = 2, kam_threshold: float = 7.0,
                       sagb_min: float = 3.0, sagb_max: Optional[float] = None, hagb_min: float = 15.0,
                       kam_bin_width: float = 0.25, gb_bin_width: float = 2.0) -> DeformationResult:
    """Full deformation analysis workflow.

    Args:
        dataset: EBSDDataset instance
        grains: GrainSet instance (from reconstruct_grains)
        kam_order: KAM neighbor order (default=2)
        kam_threshold: KAM threshold in degrees (default=7.0)
        sagb_min: SAGB minimum angle (default=3.0°)
        sagb_max: SAGB maximum angle; default None = ``hagb_min``, as the script classifies (a boundary is SAGB
            from 3° up to the HAGB limit and HAGB from it)
        hagb_min: HAGB minimum angle (default=15.0°)
        kam_bin_width: KAM histogram bin width (default=0.25°)
        gb_bin_width: GB histogram bin width (default=2.0°)

    Returns:
        DeformationResult with all metrics

    Reference: DeformationAnalysis.m full workflow
    """
    if sagb_max is None:
        sagb_max = hagb_min

    # 1. Calculate KAM
    kam = calculate_kam(dataset, order=kam_order, threshold_deg=kam_threshold,
                        grains=grains if grains is not None and grains.n_grains > 0 else None)
    kam_hist_counts, kam_hist_edges = calculate_kam_histogram(kam, bin_width=kam_bin_width)

    # 2-4: GB-dependent metrics — skip gracefully when no grains exist
    _no_gb = (grains.n_grains == 0 or grains.boundary_type is None)

    if _no_gb:
        # no grain, no boundary: the numbers are missing (None, an empty cell in the workbook), not 0
        gb_metrics = {key: None for key in (
            'total_gb_length', 'hagb_length', 'sagb_length', 'gb_per_phase_area', 'gb_per_field_area',
            'hagb_per_phase_area', 'hagb_per_field_area', 'sagb_per_phase_area', 'sagb_per_field_area')}
        gb_seg_counts = np.array([], dtype=float)
        gb_seg_bins = np.array([], dtype=float)
        sphericity = np.array([], dtype=float)
    else:
        # 2. Calculate GB metrics
        gb_metrics = calculate_gb_metrics(grains, dataset, sagb_min, sagb_max, hagb_min)

        # 3. GB segment histogram
        gb_seg_counts, gb_seg_bins = calculate_gb_segment_histogram(
            grains, dataset, min_angle=sagb_min, max_angle=63.0, bin_width=gb_bin_width
        )

        # 4. Sphericity
        sphericity = calculate_sphericity(grains)

    return DeformationResult(
        kam=kam,
        kam_hist_counts=kam_hist_counts,
        kam_hist_edges=kam_hist_edges,
        kam_order=kam_order,
        kam_threshold=kam_threshold,

        total_gb_length=gb_metrics['total_gb_length'],
        hagb_length=gb_metrics['hagb_length'],
        sagb_length=gb_metrics['sagb_length'],
        gb_per_phase_area=gb_metrics['gb_per_phase_area'],
        gb_per_field_area=gb_metrics['gb_per_field_area'],
        hagb_per_phase_area=gb_metrics['hagb_per_phase_area'],
        hagb_per_field_area=gb_metrics['hagb_per_field_area'],
        sagb_per_phase_area=gb_metrics['sagb_per_phase_area'],
        sagb_per_field_area=gb_metrics['sagb_per_field_area'],

        gb_seg_hist_counts=gb_seg_counts,
        gb_seg_hist_bins=gb_seg_bins,

        sphericity=sphericity,
        hagb_min=float(hagb_min),
    )
