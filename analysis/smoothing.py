"""
Orientation Smoothing — Preprocessing filters for EBSD orientation data.

Provides smoothing filters that reduce noise in orientation maps while
preserving grain boundaries. These should be applied before grain reconstruction.

Algorithms:
- Median filter: Replace each pixel with the median orientation of its neighborhood
- Mean filter: Replace each pixel with the mean orientation of its neighborhood
- Quality-weighted mean: Weight contributions by band contrast / quality metric

Reference:
    MTEX: splineFilter, halfQuadraticFilter, medianFilter
    matlab_testskripts/ImportAndModify_MTEX6.m lines 111-120
"""

import numpy as np

try:
    from orix.quaternion import Orientation, Rotation, Misorientation
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False
    Orientation = None

from analysis.ebsd_dataset import EBSDDataset


def _kernel_offsets(kernel_size: int) -> list:
    """Get (dy, dx) offsets for a square kernel, excluding center.

    Parameters
    ----------
    kernel_size : int
        Kernel size (3 = 3x3, 5 = 5x5).

    Returns
    -------
    list of (dy, dx) tuples
    """
    half = kernel_size // 2
    offsets = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))
    return offsets


def _overlap_slices(dy: int, dx: int, ny: int, nx: int):
    """Get source and destination slices for a shift offset.

    Returns (src_y, src_x, dst_y, dst_x) slices for the overlapping region
    when shifting by (dy, dx).
    """
    src_y = slice(max(0, -dy), min(ny, ny - dy))
    src_x = slice(max(0, -dx), min(nx, nx - dx))
    dst_y = slice(max(0, dy), min(ny, ny + dy))
    dst_x = slice(max(0, dx), min(nx, nx + dx))
    return src_y, src_x, dst_y, dst_x


def quality_weighted_mean_filter(
    dataset: EBSDDataset,
    kernel_size: int = 3,
    max_misorientation_deg: float = 5.0,
) -> np.ndarray:
    """Apply quality-weighted mean orientation filter (vectorized).

    For each pixel, accumulates quality-weighted quaternion contributions
    from neighbors within the misorientation threshold. Uses the same
    vectorized shift approach as KAM calculation for performance.

    Parameters
    ----------
    dataset : EBSDDataset
        Input EBSD data with orientations and quality metric.
    kernel_size : int
        Neighborhood size (default 3 for 3x3).
    max_misorientation_deg : float
        Maximum misorientation in degrees to include a neighbor.

    Returns
    -------
    np.ndarray
        Smoothed rotation quaternions, shape (n_pixels, 4).
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for orientation smoothing")

    ny, nx = dataset.shape
    ori_2d = dataset.orientations_2d
    quality = dataset.quality.reshape(ny, nx)
    max_misor_rad = np.deg2rad(max_misorientation_deg)
    symmetry = dataset.phase.point_group

    # Work with raw quaternion data in 2D layout: (ny, nx, 4)
    quat_2d = dataset.orientations.data.copy().reshape(ny, nx, 4)

    # Accumulators for weighted quaternion average
    accum = np.zeros((ny, nx, 4), dtype=np.float64)
    weight_sum = np.zeros((ny, nx), dtype=np.float64)

    # Self-contribution (center pixel always contributes)
    center_weight = quality.copy()
    accum += quat_2d * center_weight[..., None]
    weight_sum += center_weight

    offsets = _kernel_offsets(kernel_size)

    for dy, dx in offsets:
        src_y, src_x, dst_y, dst_x = _overlap_slices(dy, dx, ny, nx)

        # Center and neighbor orientations for the overlap region
        # NOTE: Use reshape(-1) not flatten() — orix flatten() scrambles
        # quaternion data for non-contiguous 2D Orientation slices.
        center_ori = ori_2d[src_y, src_x].reshape(-1)
        neighbor_ori = ori_2d[dst_y, dst_x].reshape(-1)

        if center_ori.size == 0:
            continue

        # Vectorized misorientation calculation with crystal symmetry
        miso = Misorientation(
            center_ori * (~neighbor_ori),
            symmetry=(symmetry, symmetry)
        )
        angles = miso.angle
        h = src_y.stop - src_y.start
        w = src_x.stop - src_x.start
        angles_2d = np.asarray(angles).reshape(h, w)

        # Valid mask: within threshold
        valid = angles_2d < max_misor_rad

        # Get neighbor quaternions and ensure hemisphere consistency
        center_q = quat_2d[src_y, src_x]
        neighbor_q = quat_2d[dst_y, dst_x].copy()
        dots = np.sum(center_q * neighbor_q, axis=-1)
        # Flip quaternions in wrong hemisphere
        flip_mask = dots < 0
        neighbor_q[flip_mask] *= -1

        # Weight by quality and validity
        w_neighbor = quality[dst_y, dst_x] * valid

        # Accumulate
        accum[src_y, src_x] += neighbor_q * w_neighbor[..., None]
        weight_sum[src_y, src_x] += w_neighbor

    # Normalize: weighted mean quaternion
    # Avoid division by zero for pixels with no valid neighbors
    safe_weight = np.where(weight_sum > 0, weight_sum, 1.0)
    result = accum / safe_weight[..., None]

    # Normalize quaternion to unit length
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    result = result / norms

    return result.reshape(-1, 4)


def median_orientation_filter(
    dataset: EBSDDataset,
    kernel_size: int = 3,
    max_misorientation_deg: float = 5.0,
) -> np.ndarray:
    """Apply median orientation filter to reduce noise (vectorized).

    For each pixel, selects the neighbor orientation that is most central
    (closest to the weighted mean) among those within the misorientation
    threshold. This preserves actual measured orientations while reducing noise.

    Strategy: compute weighted mean first (fast, vectorized), then for each
    kernel offset, compute distance from neighbor to the mean and track the
    minimum-distance neighbor per pixel.

    Parameters
    ----------
    dataset : EBSDDataset
        Input EBSD data with orientations.
    kernel_size : int
        Neighborhood size (default 3 for 3x3).
    max_misorientation_deg : float
        Maximum misorientation in degrees to include a neighbor.

    Returns
    -------
    np.ndarray
        Smoothed rotation quaternions, shape (n_pixels, 4).
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for orientation smoothing")

    ny, nx = dataset.shape
    ori_2d = dataset.orientations_2d
    max_misor_rad = np.deg2rad(max_misorientation_deg)
    symmetry = dataset.phase.point_group

    quat_2d = dataset.orientations.data.copy().reshape(ny, nx, 4)

    # Step 1: Compute weighted mean orientation (reuse the mean filter logic)
    mean_quats = quality_weighted_mean_filter(
        dataset, kernel_size, max_misorientation_deg
    ).reshape(ny, nx, 4)

    # Step 2: For each pixel, find the neighbor (or self) closest to the mean
    # Track best (minimum distance) quaternion per pixel
    best_quats = quat_2d.copy()  # Start with self
    best_dist = np.full((ny, nx), np.inf)

    # Mean orientations for distance computation
    mean_ori_flat = Orientation(mean_quats.reshape(-1, 4), symmetry=symmetry)
    mean_ori_2d = mean_ori_flat.reshape(ny, nx)

    # Self distance to mean
    self_ori_flat = ori_2d.reshape(-1)
    miso_self = Misorientation(
        self_ori_flat * (~mean_ori_flat),
        symmetry=(symmetry, symmetry)
    )
    best_dist = np.asarray(miso_self.angle).reshape(ny, nx)

    offsets = _kernel_offsets(kernel_size)

    for dy, dx in offsets:
        src_y, src_x, dst_y, dst_x = _overlap_slices(dy, dx, ny, nx)

        h = src_y.stop - src_y.start
        w = src_x.stop - src_x.start
        if h == 0 or w == 0:
            continue

        # Center-to-neighbor misorientation (for threshold filtering)
        center_ori = ori_2d[src_y, src_x].reshape(-1)
        neighbor_ori = ori_2d[dst_y, dst_x].reshape(-1)

        miso_cn = Misorientation(
            center_ori * (~neighbor_ori),
            symmetry=(symmetry, symmetry)
        )
        cn_angles = np.asarray(miso_cn.angle).reshape(h, w)
        valid = cn_angles < max_misor_rad

        # Neighbor-to-mean misorientation (for "most central" selection)
        mean_sub = mean_ori_2d[src_y, src_x].reshape(-1)
        miso_nm = Misorientation(
            neighbor_ori * (~mean_sub),
            symmetry=(symmetry, symmetry)
        )
        nm_dist = np.asarray(miso_nm.angle).reshape(h, w)

        # Only consider valid neighbors; invalid get infinite distance
        nm_dist = np.where(valid, nm_dist, np.inf)

        # Update best where this neighbor is closer to mean
        better = nm_dist < best_dist[src_y, src_x]
        neighbor_q = quat_2d[dst_y, dst_x]

        # Apply update using boolean indexing
        update_region = best_quats[src_y, src_x]
        dist_region = best_dist[src_y, src_x]
        update_region[better] = neighbor_q[better]
        dist_region[better] = nm_dist[better]
        best_quats[src_y, src_x] = update_region
        best_dist[src_y, src_x] = dist_region

    return best_quats.reshape(-1, 4)


def apply_smoothing(
    dataset: EBSDDataset,
    method: str = "median",
    kernel_size: int = 3,
    max_misorientation_deg: float = 5.0,
) -> EBSDDataset:
    """Apply orientation smoothing and return a new EBSDDataset.

    Parameters
    ----------
    dataset : EBSDDataset
        Input EBSD data.
    method : str
        Smoothing method: 'median' or 'weighted_mean'.
    kernel_size : int
        Neighborhood size (3 or 5).
    max_misorientation_deg : float
        Grain boundary protection threshold in degrees.

    Returns
    -------
    EBSDDataset
        New dataset with smoothed orientations.
    """
    if not ORIX_AVAILABLE:
        raise ImportError("orix is required for orientation smoothing")

    from orix.crystal_map import CrystalMap

    if method == "median":
        smoothed_quats = median_orientation_filter(
            dataset, kernel_size, max_misorientation_deg
        )
    elif method == "weighted_mean":
        smoothed_quats = quality_weighted_mean_filter(
            dataset, kernel_size, max_misorientation_deg
        )
    else:
        raise ValueError(f"Unknown smoothing method: {method!r}. Use 'median' or 'weighted_mean'.")

    # Build new CrystalMap with smoothed orientations
    smoothed_rotations = Rotation(smoothed_quats)
    new_xmap = CrystalMap(
        rotations=smoothed_rotations,
        phase_id=dataset.xmap.phase_id,
        x=dataset.xmap.x,
        y=dataset.xmap.y,
        phase_list=dataset.xmap.phases,
        prop=dict(dataset.xmap.prop),
    )

    return EBSDDataset(new_xmap, dataset.step_size, dataset.source)
