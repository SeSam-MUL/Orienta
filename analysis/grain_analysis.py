"""
Grain Analysis — Grain reconstruction and grain size analysis.

This module provides:
- reconstruct_grains(): Misorientation-based connected-component segmentation
- analyze_grain_size(): ECD, MaxDim, Aspect Ratio, area-weighted statistics
- calculate_grain_properties(): Mean orientations, GOS per grain

Algorithm:
1. Build boundary image: mark edges where misorientation > threshold
2. Connected components on inverse boundary image → grain labels
3. Remove grains with < min_pixels
4. Calculate per-grain properties (mean orientation, size, area, ECD, GOS)

Reference: the MTEX-equivalence design notes §2.1, §2.2
         matlab_testskripts/ImportAndModify_MTEX6.m lines 166-167, 260-261
"""

from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import label as scipy_label
    from scipy.ndimage import labeled_comprehension
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from orix.quaternion import Misorientation, Orientation
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

from analysis.ebsd_dataset import EBSDDataset, GrainSet


@dataclass
class GrainSizeResult:
    """
    Grain size analysis results.

    Attributes:
        avg_maxDimX: Mean maximum X dimension (µm)
        std_maxDimX: Std deviation of maxDimX
        avg_maxDimY: Mean maximum Y dimension (µm)
        std_maxDimY: Std deviation of maxDimY
        avg_ECD: Mean equivalent circular diameter (µm)
        std_ECD: Std deviation of ECD
        avg_AR: Mean aspect ratio (width/height)
        std_AR: Std deviation of aspect ratio
        area_weighted_ECD: Area-weighted mean ECD (µm)

        Distributions (for histograms):
        ecd_per_grain: ECD per grain (µm), length n_grains
        maxDimX_per_grain: MaxDimX per grain (µm)
        maxDimY_per_grain: MaxDimY per grain (µm)
        ar_per_grain: Aspect ratio per grain

    Reference: the MTEX-equivalence design notes §2.2
    """
    avg_maxDimX: float
    std_maxDimX: float
    avg_maxDimY: float
    std_maxDimY: float
    avg_ECD: float
    std_ECD: float
    avg_AR: float
    std_AR: float
    area_weighted_ECD: float

    # Distributions
    ecd_per_grain: np.ndarray
    maxDimX_per_grain: np.ndarray
    maxDimY_per_grain: np.ndarray
    ar_per_grain: np.ndarray


def reconstruct_grains(
    dataset: EBSDDataset,
    angle_threshold_deg: float = 5.0,
    min_pixels: int = 3,
) -> GrainSet:
    """
    Reconstruct grains via misorientation-based connected-component segmentation.

    Algorithm:
    1. Build boundary image: for each 4-neighbor pair, mark as boundary if
       misorientation > threshold
    2. Invert boundary image and apply connected components → grain labels
    3. Remove grains with < min_pixels
    4. Calculate per-grain properties

    Args:
        dataset: EBSDDataset with orientations
        angle_threshold_deg: Misorientation threshold in degrees (default 5°)
        min_pixels: Minimum pixels per grain (default 3)

    Returns:
        GrainSet with grain_ids, n_grains, and per-grain properties

    Reference: the MTEX-equivalence design notes §2.1
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for grain reconstruction. Install with: pip install scipy")
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for grain reconstruction. Install with: pip install orix")

    ny, nx = dataset.shape
    threshold_rad = np.deg2rad(angle_threshold_deg)

    # Get orientations as 2D array
    ori_2d = dataset.orientations_2d
    symmetry = dataset.phase.point_group

    # Build edge image: True where misorientation > threshold
    # Vectorized approach: compute misorientations for all neighbor pairs at once
    edges_h = np.zeros((ny, nx), dtype=bool)  # horizontal edges
    edges_v = np.zeros((ny, nx), dtype=bool)  # vertical edges

    indexed = dataset.indexed_mask_2d

    # Horizontal edges: compare (i, j) with (i, j+1)
    both_indexed_h = indexed[:, :-1] & indexed[:, 1:]
    # Non-indexed pairs are boundaries
    edges_h[:, :-1] = ~both_indexed_h

    if both_indexed_h.any():
        try:
            ori_left = ori_2d[:, :-1][both_indexed_h]
            ori_right = ori_2d[:, 1:][both_indexed_h]
            mori_h = Misorientation(ori_right * ~ori_left, symmetry=(symmetry, symmetry))
            mori_h = mori_h.map_into_symmetry_reduced_zone()
            angles_h = mori_h.angle.data if hasattr(mori_h.angle, 'data') else np.asarray(mori_h.angle)
            angles_h = np.atleast_1d(angles_h).ravel()
            # Write back: mark boundaries where angle exceeds threshold
            mask_h_flat = np.zeros(both_indexed_h.shape, dtype=bool)
            mask_h_flat[both_indexed_h] = angles_h > threshold_rad
            edges_h[:, :-1] |= mask_h_flat
        except Exception as e:
            logger.warning("Horizontal misorientation failed, marking all as boundary: %s", e)
            edges_h[:, :-1] = True

    # Vertical edges: compare (i, j) with (i+1, j)
    both_indexed_v = indexed[:-1, :] & indexed[1:, :]
    edges_v[:-1, :] = ~both_indexed_v

    if both_indexed_v.any():
        try:
            ori_top = ori_2d[:-1, :][both_indexed_v]
            ori_bottom = ori_2d[1:, :][both_indexed_v]
            mori_v = Misorientation(ori_bottom * ~ori_top, symmetry=(symmetry, symmetry))
            mori_v = mori_v.map_into_symmetry_reduced_zone()
            angles_v = mori_v.angle.data if hasattr(mori_v.angle, 'data') else np.asarray(mori_v.angle)
            angles_v = np.atleast_1d(angles_v).ravel()
            mask_v_flat = np.zeros(both_indexed_v.shape, dtype=bool)
            mask_v_flat[both_indexed_v] = angles_v > threshold_rad
            edges_v[:-1, :] |= mask_v_flat
        except Exception as e:
            logger.warning("Vertical misorientation failed, marking all as boundary: %s", e)
            edges_v[:-1, :] = True

    # Combine edges: a pixel is on the boundary if any of its edges are boundaries
    boundary = edges_h | edges_v | np.roll(edges_h, 1, axis=1) | np.roll(edges_v, 1, axis=0)

    # Non-indexed pixels are not part of any grain
    boundary[~dataset.indexed_mask_2d] = True

    # Connected components on inverse boundary image
    # Structure for 4-connectivity
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)

    # Invert: grains are where boundary=False
    grain_regions = ~boundary
    # Also ensure only indexed pixels are included
    grain_regions = grain_regions & dataset.indexed_mask_2d

    grain_ids, n_grains_raw = scipy_label(grain_regions, structure=structure)

    # Expand grain IDs into boundary (0) pixels via nearest-neighbor fill
    # This ensures direct grain-to-grain adjacency exists for boundary angle calculation
    if n_grains_raw > 0:
        from scipy.ndimage import distance_transform_edt
        unlabeled = grain_ids == 0
        if unlabeled.any():
            # For each 0-pixel, find the nearest labeled grain pixel
            _, nearest_indices = distance_transform_edt(unlabeled, return_indices=True)
            grain_ids[unlabeled] = grain_ids[nearest_indices[0][unlabeled], nearest_indices[1][unlabeled]]
        # Re-zero non-indexed pixels (they should stay 0)
        grain_ids[~dataset.indexed_mask_2d] = 0

    # Remove grains with < min_pixels and renumber — vectorized via lookup table
    if min_pixels > 1:
        counts = np.bincount(grain_ids.ravel(), minlength=n_grains_raw + 1)
        # Build remap: old_id → new_id (0 for small grains)
        remap = np.zeros(n_grains_raw + 1, dtype=grain_ids.dtype)
        new_id = 1
        for old_id in range(1, n_grains_raw + 1):
            if counts[old_id] >= min_pixels:
                remap[old_id] = new_id
                new_id += 1
        grain_ids = remap[grain_ids]
        n_grains = new_id - 1
    else:
        n_grains = n_grains_raw

    # Calculate per-grain properties
    grains = GrainSet(grain_ids=grain_ids, n_grains=n_grains)

    if n_grains > 0:
        grains = calculate_grain_properties(dataset, grains)

    return grains


def calculate_grain_properties(dataset: EBSDDataset, grains: GrainSet) -> GrainSet:
    """
    Calculate per-grain properties: mean orientations, size, area, ECD, GOS.

    Args:
        dataset: EBSDDataset with orientations and step_size
        grains: GrainSet with grain_ids

    Returns:
        Updated GrainSet with populated properties

    Reference: the MTEX-equivalence design notes §2.1
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix required for grain properties")

    grain_ids_flat = grains.grain_ids.ravel()
    ori_flat = dataset.orientations

    unique_grain_ids = np.arange(1, grains.n_grains + 1)

    # Grain size (number of pixels) — vectorized via bincount
    counts = np.bincount(grain_ids_flat, minlength=grains.n_grains + 1)
    grain_size = counts[1:grains.n_grains + 1]  # skip grain_id=0

    # Area (µm²)
    area = grain_size * (dataset.step_size ** 2)

    # ECD (µm)
    ecd = 2 * np.sqrt(area / np.pi)

    # Mean orientations per grain — proper quaternion averaging
    symmetry = dataset.phase.point_group
    mean_quats = np.zeros((grains.n_grains, 4))
    for i, gid in enumerate(unique_grain_ids):
        mask = grain_ids_flat == gid
        grain_ori = ori_flat[mask]
        if grain_ori.size == 0:
            mean_quats[i] = ori_flat[0].data.ravel()
            continue
        if grain_ori.size == 1:
            mean_quats[i] = grain_ori[0].data.ravel()
            continue
        # Average quaternions with hemisphere consistency.
        # Previous implementation looped over every quaternion in Python.
        # Vectorize: dot products with ref in one matmul, then sign-flip
        # in a single fancy-index assignment. Cuts per-grain work from
        # O(n_pixels) Python overhead to a handful of numpy ops.
        quats = grain_ori.data.copy()
        if quats.ndim > 2:
            quats = quats.reshape(-1, 4)
        ref = quats[0]
        # Vectorized hemisphere flip
        dots = quats @ ref  # (N,)
        quats[dots < 0] *= -1
        mean_q = np.mean(quats, axis=0)
        norm = np.linalg.norm(mean_q)
        if norm > 0:
            mean_q /= norm
        mean_quats[i] = mean_q

    mean_orientations = Orientation(mean_quats, symmetry=symmetry)

    # GOS (Grain Orientation Spread) in radians
    gos = np.zeros(grains.n_grains)
    for i, gid in enumerate(unique_grain_ids):
        mask = grain_ids_flat == gid
        grain_ori = ori_flat[mask]
        if grain_ori.size > 1:
            mean_ori = mean_orientations[i]
            # Misorientation from each pixel to grain mean
            mori = Misorientation(grain_ori * ~mean_ori, symmetry=(dataset.phase.point_group, dataset.phase.point_group))
            mori = mori.map_into_symmetry_reduced_zone()
            angles = mori.angle.data if hasattr(mori.angle, 'data') else mori.angle
            gos[i] = np.mean(angles)
        else:
            gos[i] = 0.0

    # Aspect ratio from bounding box — vectorized via find_objects
    from scipy.ndimage import find_objects
    aspect_ratio = np.ones(grains.n_grains)
    slices = find_objects(grains.grain_ids)
    for i, sl in enumerate(slices):
        if sl is not None:
            height = sl[0].stop - sl[0].start
            width = sl[1].stop - sl[1].start
            if height > 0:
                aspect_ratio[i] = width / height

    # Compute grain boundary misorientation angles (degrees)
    # Find all neighbor pairs where grain IDs differ
    gids = grains.grain_ids
    ny, nx = gids.shape
    boundary_angles = []

    # Horizontal neighbors
    h_diff = (gids[:, :-1] != gids[:, 1:]) & (gids[:, :-1] > 0) & (gids[:, 1:] > 0)
    if h_diff.any():
        try:
            ori_2d = dataset.orientations_2d
            symmetry = dataset.phase.point_group
            ori_l = ori_2d[:, :-1][h_diff]
            ori_r = ori_2d[:, 1:][h_diff]
            mori = Misorientation(ori_r * ~ori_l, symmetry=(symmetry, symmetry))
            mori = mori.map_into_symmetry_reduced_zone()
            ang = mori.angle.data if hasattr(mori.angle, 'data') else np.asarray(mori.angle)
            boundary_angles.append(np.rad2deg(np.atleast_1d(ang).ravel()))
        except Exception as e:
            logger.warning("Horizontal boundary angle calculation failed: %s", e)

    # Vertical neighbors
    v_diff = (gids[:-1, :] != gids[1:, :]) & (gids[:-1, :] > 0) & (gids[1:, :] > 0)
    if v_diff.any():
        try:
            if not h_diff.any():
                ori_2d = dataset.orientations_2d
                symmetry = dataset.phase.point_group
            ori_t = ori_2d[:-1, :][v_diff]
            ori_b = ori_2d[1:, :][v_diff]
            mori = Misorientation(ori_b * ~ori_t, symmetry=(symmetry, symmetry))
            mori = mori.map_into_symmetry_reduced_zone()
            ang = mori.angle.data if hasattr(mori.angle, 'data') else np.asarray(mori.angle)
            boundary_angles.append(np.rad2deg(np.atleast_1d(ang).ravel()))
        except Exception as e:
            logger.warning("Vertical boundary angle calculation failed: %s", e)

    if boundary_angles:
        grains.boundary_type = np.concatenate(boundary_angles)
    else:
        grains.boundary_type = np.array([], dtype=float)

    # Perimeter per grain: count boundary segments touching each grain
    # Each boundary segment contributes step_size to the perimeters of both adjacent grains
    perimeter = np.zeros(grains.n_grains)
    gids_flat = grains.grain_ids
    # Horizontal boundaries
    h_boundary = gids_flat[:, :-1] != gids_flat[:, 1:]
    if h_boundary.any():
        left_ids = gids_flat[:, :-1][h_boundary]
        right_ids = gids_flat[:, 1:][h_boundary]
        for ids in (left_ids, right_ids):
            valid = ids > 0
            np.add.at(perimeter, ids[valid] - 1, dataset.step_size)
    # Vertical boundaries
    v_boundary = gids_flat[:-1, :] != gids_flat[1:, :]
    if v_boundary.any():
        top_ids = gids_flat[:-1, :][v_boundary]
        bottom_ids = gids_flat[1:, :][v_boundary]
        for ids in (top_ids, bottom_ids):
            valid = ids > 0
            np.add.at(perimeter, ids[valid] - 1, dataset.step_size)
    # Map edges also count as perimeter
    for edge in (gids_flat[0, :], gids_flat[-1, :], gids_flat[:, 0], gids_flat[:, -1]):
        valid = edge > 0
        np.add.at(perimeter, edge[valid] - 1, dataset.step_size)
    grains.perimeter = perimeter

    # Update GrainSet
    grains.mean_orientations = mean_orientations
    grains.grain_size = grain_size
    grains.area = area
    grains.ecd = ecd
    grains.aspect_ratio = aspect_ratio
    grains.gos = gos

    return grains


def analyze_grain_size(
    dataset: EBSDDataset,
    grains: GrainSet,
    remove_edge_grains: bool = True,
    edge_tolerance_fraction: float = 0.85
) -> GrainSizeResult:
    """
    Analyze grain size: ECD, MaxDim, Aspect Ratio, area-weighted statistics.

    Workflow:
    1. Optionally remove edge-touching grains (with safety check)
    2. Calculate MaxDimX, MaxDimY per grain
    3. Compute summary statistics (mean, std)
    4. Compute area-weighted ECD

    Args:
        dataset: EBSDDataset with step_size
        grains: GrainSet with calculated properties
        remove_edge_grains: Remove grains touching map edges (default True)
        edge_tolerance_fraction: If > this fraction removed, skip removal (default 0.85)

    Returns:
        GrainSizeResult with statistics and distributions

    Reference: the MTEX-equivalence design notes §2.2
    """
    # Handle zero-grain case gracefully (e.g. fully random data with no boundaries)
    if grains.n_grains == 0 or grains.ecd is None or grains.area is None:
        empty = np.array([], dtype=float)
        return GrainSizeResult(
            avg_maxDimX=0.0, std_maxDimX=0.0,
            avg_maxDimY=0.0, std_maxDimY=0.0,
            avg_ECD=0.0, std_ECD=0.0,
            avg_AR=0.0, std_AR=0.0,
            area_weighted_ECD=0.0,
            ecd_per_grain=empty,
            maxDimX_per_grain=empty,
            maxDimY_per_grain=empty,
            ar_per_grain=empty,
        )

    ny, nx = grains.grain_ids.shape
    grain_ids_flat = grains.grain_ids.ravel()
    unique_grain_ids = np.arange(1, grains.n_grains + 1)

    # Identify edge-touching grains
    edge_grain_ids = set()
    if remove_edge_grains:
        # Top and bottom rows
        edge_grain_ids.update(grains.grain_ids[0, :])
        edge_grain_ids.update(grains.grain_ids[-1, :])
        # Left and right columns
        edge_grain_ids.update(grains.grain_ids[:, 0])
        edge_grain_ids.update(grains.grain_ids[:, -1])
        edge_grain_ids.discard(0)  # Remove background

        # Safety check: if > 85% would be removed, skip
        n_edge_grains = len(edge_grain_ids)
        if grains.n_grains == 0 or n_edge_grains / grains.n_grains > edge_tolerance_fraction:
            logger.warning("%d/%d grains touch edges (>%.0f%%). Skipping edge removal.",
                          n_edge_grains, grains.n_grains, edge_tolerance_fraction * 100)
            edge_grain_ids = set()

    # Filter grain IDs
    if edge_grain_ids:
        valid_mask = np.array([gid not in edge_grain_ids for gid in unique_grain_ids])
        valid_grain_ids = unique_grain_ids[valid_mask]
    else:
        valid_grain_ids = unique_grain_ids
        valid_mask = np.ones(len(unique_grain_ids), dtype=bool)

    # Calculate MaxDimX, MaxDimY per grain — vectorized via find_objects
    from scipy.ndimage import find_objects
    maxDimX = np.zeros(grains.n_grains)
    maxDimY = np.zeros(grains.n_grains)

    slices = find_objects(grains.grain_ids)
    for i, sl in enumerate(slices):
        if sl is not None:
            maxDimX[i] = (sl[1].stop - sl[1].start) * dataset.step_size
            maxDimY[i] = (sl[0].stop - sl[0].start) * dataset.step_size

    # Filter to valid grains
    ecd_valid = grains.ecd[valid_mask]
    maxDimX_valid = maxDimX[valid_mask]
    maxDimY_valid = maxDimY[valid_mask]
    ar_valid = grains.aspect_ratio[valid_mask]
    area_valid = grains.area[valid_mask]

    # Summary statistics
    avg_ecd = np.mean(ecd_valid) if len(ecd_valid) > 0 else 0.0
    std_ecd = np.std(ecd_valid) if len(ecd_valid) > 0 else 0.0

    avg_maxDimX = np.mean(maxDimX_valid) if len(maxDimX_valid) > 0 else 0.0
    std_maxDimX = np.std(maxDimX_valid) if len(maxDimX_valid) > 0 else 0.0

    avg_maxDimY = np.mean(maxDimY_valid) if len(maxDimY_valid) > 0 else 0.0
    std_maxDimY = np.std(maxDimY_valid) if len(maxDimY_valid) > 0 else 0.0

    avg_ar = np.mean(ar_valid) if len(ar_valid) > 0 else 0.0
    std_ar = np.std(ar_valid) if len(ar_valid) > 0 else 0.0

    # Area-weighted ECD
    if len(ecd_valid) > 0 and area_valid.sum() > 0:
        area_weighted_ecd = np.sum(ecd_valid * area_valid) / np.sum(area_valid)
    else:
        area_weighted_ecd = 0.0

    return GrainSizeResult(
        avg_maxDimX=avg_maxDimX,
        std_maxDimX=std_maxDimX,
        avg_maxDimY=avg_maxDimY,
        std_maxDimY=std_maxDimY,
        avg_ECD=avg_ecd,
        std_ECD=std_ecd,
        avg_AR=avg_ar,
        std_AR=std_ar,
        area_weighted_ECD=area_weighted_ecd,
        ecd_per_grain=ecd_valid,
        maxDimX_per_grain=maxDimX_valid,
        maxDimY_per_grain=maxDimY_valid,
        ar_per_grain=ar_valid,
    )
