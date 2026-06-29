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
from typing import Tuple
import numpy as np

try:
    from orix.quaternion import Misorientation
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False


@dataclass
class DeformationResult:
    """Results from deformation analysis.

    Attributes:
        kam: KAM map (ny, nx) in degrees
        kam_hist_counts: Histogram counts (probability normalized)
        kam_hist_edges: Histogram bin edges
        kam_order: KAM calculation order (2 = 2nd nearest neighbors)
        kam_threshold: KAM threshold in degrees (7.0)

        # Grain boundary metrics
        total_gb_length: Total grain boundary length (µm)
        hagb_length: High-angle GB length (>15°) (µm)
        sagb_length: Sub-grain/low-angle GB length (3-15°) (µm)
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
        sphericity: Sphericity values per grain (area / (perimeter * equiv_radius))
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


def calculate_kam(dataset, order: int = 2, threshold_deg: float = 7.0) -> np.ndarray:
    """Calculate Kernel Average Misorientation (KAM) for each pixel.

    KAM measures local misorientation by averaging the misorientation angles
    between each pixel and its neighbors up to 'order' distance (diamond neighborhood).

    Args:
        dataset: EBSDDataset instance
        order: Neighbor order (1 = 1st nearest, 2 = 2nd nearest, default=2)
        threshold_deg: Maximum misorientation angle to include (default=7.0°)
                      Angles > threshold are excluded (grain boundary artifact removal)

    Returns:
        KAM map of shape (ny, nx) in degrees

    Reference: DeformationAnalysis.m lines 138-143
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for KAM calculation")

    # Get 2D orientation map
    ori_2d = dataset.orientations_2d  # shape (ny, nx)
    ny, nx = ori_2d.shape

    # Initialize KAM map
    kam = np.zeros((ny, nx), dtype=np.float32)
    count = np.zeros((ny, nx), dtype=np.int32)  # Number of valid neighbors

    threshold_rad = np.radians(threshold_deg)

    # Diamond neighborhood: all (dy, dx) where abs(dy) + abs(dx) <= order
    for dy in range(-order, order + 1):
        for dx in range(-order, order + 1):
            if dx == 0 and dy == 0:
                continue
            if abs(dx) + abs(dy) > order:  # Outside diamond
                continue

            # Create shifted arrays (handle boundary with valid indexing)
            y_min = max(0, dy)
            y_max = min(ny, ny + dy)
            x_min = max(0, dx)
            x_max = min(nx, nx + dx)

            y_src_min = max(0, -dy)
            y_src_max = min(ny, ny - dy)
            x_src_min = max(0, -dx)
            x_src_max = min(nx, nx - dx)

            # Extract overlapping regions
            ori_center = ori_2d[y_src_min:y_src_max, x_src_min:x_src_max]
            ori_neighbor = ori_2d[y_min:y_max, x_min:x_max]

            # Calculate misorientation angles with crystal symmetry
            symmetry = dataset.phase.point_group
            miso = Misorientation(
                ori_center.reshape(-1) * (~ori_neighbor.reshape(-1)),
                symmetry=(symmetry, symmetry)
            )
            miso = miso.map_into_symmetry_reduced_zone()
            angles = miso.angle.reshape(ori_center.shape)

            # Filter by threshold
            valid_mask = angles < threshold_rad

            # Accumulate
            kam[y_src_min:y_src_max, x_src_min:x_src_max] += np.where(
                valid_mask, np.degrees(angles), 0.0
            )
            count[y_src_min:y_src_max, x_src_min:x_src_max] += valid_mask.astype(np.int32)

    # Average KAM (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        kam = np.where(count > 0, kam / count, np.nan)

    return kam


def calculate_kam_histogram(kam: np.ndarray, bin_width: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate KAM histogram with probability normalization.

    Args:
        kam: KAM map (ny, nx) in degrees
        bin_width: Histogram bin width in degrees (default=0.25)

    Returns:
        counts: Histogram counts (probability normalized, sums to 1.0)
        edges: Bin edges

    Reference: DeformationAnalysis.m line 143
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

    Reference: DeformationAnalysis.m lines 10-14
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

    Reference: DeformationAnalysis.m lines 10-31
    """
    sagb_mask, hagb_mask, all_gb_mask = classify_grain_boundaries(
        grains, sagb_min_deg, sagb_max_deg, hagb_min_deg
    )

    # Segment length = step_size (square grid assumption)
    segment_length = dataset.step_size  # µm

    # Total lengths
    total_gb_length = all_gb_mask.sum() * segment_length
    hagb_length = hagb_mask.sum() * segment_length
    sagb_length = sagb_mask.sum() * segment_length

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

    Reference: DeformationAnalysis.m lines 103-113
    """
    if grains.boundary_type is None:
        raise ValueError("Grain boundary types not calculated")

    angles = grains.boundary_type  # degrees
    segment_length = dataset.step_size  # µm

    # Create bin edges
    bin_edges = np.arange(min_angle, max_angle + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Calculate histogram of segment lengths — vectorized
    raw_counts, _ = np.histogram(angles, bins=bin_edges)
    counts = raw_counts.astype(float) * segment_length

    return counts, bin_centers


def calculate_sphericity(grains) -> np.ndarray:
    """Calculate sphericity for each grain.

    Sphericity = area / (perimeter * equivalent_radius)

    Sphericity measures how close a grain's shape is to a circle:
    - Sphericity ≈ 1.0: circular grain
    - Sphericity < 1.0: elongated or irregular grain

    Args:
        grains: GrainSet instance with area, perimeter, and ecd

    Returns:
        Sphericity array (length = n_grains)

    Reference: DeformationAnalysis.m line 47
    """
    if grains.area is None or grains.ecd is None:
        raise ValueError("Grain area and ECD not calculated")

    # Equivalent radius = ECD / 2
    equiv_radius = grains.ecd / 2.0

    # Calculate perimeter if not available
    if hasattr(grains, 'perimeter') and grains.perimeter is not None:
        perimeter = grains.perimeter
    else:
        # Estimate perimeter from area and ECD (circle approximation)
        perimeter = 2 * np.pi * equiv_radius

    # Sphericity
    with np.errstate(divide='ignore', invalid='ignore'):
        sphericity = grains.area / (perimeter * equiv_radius)
        sphericity = np.where(np.isfinite(sphericity), sphericity, np.nan)

    return sphericity


def analyze_deformation(dataset, grains, kam_order: int = 2, kam_threshold: float = 7.0,
                       sagb_min: float = 3.0, sagb_max: float = 15.0, hagb_min: float = 15.0,
                       kam_bin_width: float = 0.25, gb_bin_width: float = 2.0) -> DeformationResult:
    """Full deformation analysis workflow.

    Args:
        dataset: EBSDDataset instance
        grains: GrainSet instance (from reconstruct_grains)
        kam_order: KAM neighbor order (default=2)
        kam_threshold: KAM threshold in degrees (default=7.0)
        sagb_min: SAGB minimum angle (default=3.0°)
        sagb_max: SAGB maximum angle (default=15.0°)
        hagb_min: HAGB minimum angle (default=15.0°)
        kam_bin_width: KAM histogram bin width (default=0.25°)
        gb_bin_width: GB histogram bin width (default=2.0°)

    Returns:
        DeformationResult with all metrics

    Reference: DeformationAnalysis.m full workflow
    """
    # 1. Calculate KAM
    kam = calculate_kam(dataset, order=kam_order, threshold_deg=kam_threshold)
    kam_hist_counts, kam_hist_edges = calculate_kam_histogram(kam, bin_width=kam_bin_width)

    # 2-4: GB-dependent metrics — skip gracefully when no grains exist
    _no_gb = (grains.n_grains == 0 or grains.boundary_type is None)

    if _no_gb:
        gb_metrics = {
            'total_gb_length': 0.0, 'hagb_length': 0.0, 'sagb_length': 0.0,
            'gb_per_phase_area': 0.0, 'gb_per_field_area': 0.0,
            'hagb_per_phase_area': 0.0, 'hagb_per_field_area': 0.0,
            'sagb_per_phase_area': 0.0, 'sagb_per_field_area': 0.0,
        }
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
    )
