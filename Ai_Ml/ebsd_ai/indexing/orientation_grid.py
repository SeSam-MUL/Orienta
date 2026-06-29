"""Cubochoric orientation sampling on SO(3) with symmetry reduction.

Generates a uniform grid of orientations in the fundamental zone for a
given crystal symmetry.  Used to build the FAISS index reference set.
"""
from __future__ import annotations

import numpy as np


def sample_orientations(
    n_cubochoric: int = 50,
    symmetry: str = "m-3m",
) -> np.ndarray:
    """Sample orientations uniformly on SO(3) within the fundamental zone.

    Uses a regular cubochoric grid, converts to quaternions, and reduces
    to the fundamental zone for the given crystal symmetry.

    Parameters
    ----------
    n_cubochoric : int
        Number of grid points along each cubochoric axis.
        Total samples before symmetry reduction ~ n^3.
    symmetry : str
        Hermann-Mauguin symbol for the crystal point group
        (e.g., "m-3m" for cubic, "6/mmm" for hexagonal).

    Returns
    -------
    np.ndarray
        (N, 4) array of unit quaternions in the fundamental zone.
        N depends on symmetry and grid density.
    """
    from orix.quaternion import Rotation, symmetry as sym_module
    from orix.sampling import get_sample_fundamental

    # Map common symmetry strings to orix symmetry objects
    sym_map = {
        "m-3m": sym_module.Oh,
        "6/mmm": sym_module.D6h,
        "4/mmm": sym_module.D4h,
        "mmm": sym_module.D2h,
        "-3m": sym_module.D3d,
        "m-3": sym_module.Th,
        "2/m": sym_module.C2h,
        "-1": sym_module.Ci,
        "1": sym_module.C1,
    }

    sym_obj = sym_map.get(symmetry)
    if sym_obj is None:
        raise ValueError(
            f"Unknown symmetry '{symmetry}'. "
            f"Supported: {list(sym_map.keys())}"
        )

    # Use orix's built-in fundamental zone sampling
    # Resolution in degrees ~ 180 / n_cubochoric (approximate)
    resolution = max(1.0, 180.0 / n_cubochoric)
    rotations = get_sample_fundamental(
        resolution=resolution,
        point_group=sym_obj,
    )

    # Convert to numpy quaternion array (N, 4)
    quats = rotations.to_euler(degrees=False)  # Not needed, use .data directly
    quat_array = np.array(rotations.data, dtype=np.float32)

    # Ensure unit quaternions
    norms = np.linalg.norm(quat_array, axis=1, keepdims=True)
    quat_array = quat_array / np.clip(norms, 1e-8, None)

    return quat_array
