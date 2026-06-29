"""master_to_dict — Path A default, Path B escape hatch.

Path A: real GPU forward projection (translated kikuchipy reference math).
        See ./_projection/ subpackage. Lands as the default in this commit.
Path B: CPU wrapper around mp.get_patterns() with upload — the MVP path,
        retained as an emergency fallback under `_use_path_b=True`.

Source repo: https://github.com/ZacharyVarley/pcadi  (commit 88f676e)
License: MIT (see ../__SOURCE.md)
Path A translation: see ./_projection/__SOURCE.md
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch

from ._projection.direction_cosines import compute_direction_cosines
from ._projection.project import project_master_to_detector

if TYPE_CHECKING:
    import kikuchipy
    import orix.quaternion


def gpu_master_to_dict(
    master_pattern,
    rotations,
    detector,
    *,
    energy: float = 20.0,
    device: "torch.device | str" = "cuda",
    dtype: torch.dtype = torch.float32,
    _use_path_b: bool = False,
) -> torch.Tensor:
    """Generate (n, det_h, det_w) dictionary patterns on GPU.

    Defaults to Path A (real GPU projection). Set `_use_path_b=True` to
    fall back to the CPU wrapper used by the MVP.
    """
    if _use_path_b:
        return _path_b(master_pattern, rotations, detector,
                       energy=energy, device=device, dtype=dtype)
    return _path_a(master_pattern, rotations, detector,
                   energy=energy, device=device, dtype=dtype)


def _path_a(master_pattern, rotations, detector, *, energy, device, dtype):
    """Real GPU forward projection (Path A)."""
    device = torch.device(device) if isinstance(device, str) else device

    mp_data = _select_energy_slice(master_pattern, energy=energy)
    mp_t = torch.from_numpy(mp_data).to(device=device, dtype=dtype).contiguous()

    q = np.asarray(rotations.data, dtype=np.float32)
    if q.ndim > 2:
        q = q.reshape(-1, 4)
    rot_q = torch.from_numpy(q).to(device=device, dtype=dtype)

    dirs = compute_direction_cosines(detector, device=device, dtype=dtype)

    out = project_master_to_detector(mp_t, rot_q, dirs)
    return out.contiguous()


def _path_b(master_pattern, rotations, detector, *, energy, device, dtype):
    """Legacy MVP wrapper: CPU mp.get_patterns + upload. Kept as escape hatch."""
    sig = master_pattern.get_patterns(
        rotations=rotations, detector=detector, energy=energy, compute=True
    )
    arr = np.asarray(sig.data, dtype=np.float32)
    out = torch.from_numpy(arr).to(device=device, dtype=dtype)
    if out.ndim == 4 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out.contiguous()


def _select_energy_slice(mp, *, energy: float) -> np.ndarray:
    """Return the (2, npx, npx) single-energy hemisphere stack."""
    data = np.asarray(mp.data, dtype=np.float32)
    if data.ndim == 4:
        if hasattr(mp.axes_manager, "navigation_axes") and mp.axes_manager.navigation_axes:
            ax = mp.axes_manager.navigation_axes[0]
            energies = np.array([ax.offset + i * ax.scale for i in range(ax.size)])
            idx = int(np.argmin(np.abs(energies - energy)))
        else:
            idx = 0
        data = data[idx]
    if data.shape[0] != 2:
        raise ValueError(f"expected master data shape (2, npx, npx), got {data.shape}")
    return data
