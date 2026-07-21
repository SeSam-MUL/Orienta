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
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from ._projection.direction_cosines import compute_direction_cosines
from ._projection.project import project_master_to_detector
from backend.dict_gpu.exceptions import MasterPatternError

logger = logging.getLogger(__name__)

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
    """Return a C-contiguous (2, npx, npx) both-hemisphere, single-energy stack.

    The GPU forward projection needs the master in the **square Lambert**
    projection with **both hemispheres**. The indexing route loads masters
    with a plain ``kp.load(path)``, which gives the *stereographic* upper
    hemisphere — shape ``(n_energy, npx, npx)``. Sampling that with the
    Lambert direction-cosine grid is geometrically wrong (and the shape
    check would fail anyway). Reload as Lambert + both hemispheres from the
    master's source file whenever either is missing. Verified: with a
    Lambert/both master, Path A matches the kikuchipy CPU reference (Path B)
    at NCC = 1.00000.

    Also returns a **contiguous copy**: kikuchipy master slices can be
    negative-strided, and ``torch.from_numpy`` rejects negative strides.
    """
    if (getattr(mp, "hemisphere", None) != "both"
            or getattr(mp, "projection", None) != "lambert"):
        mp = _reload_both_hemispheres(mp)

    data = np.asarray(mp.data, dtype=np.float32)

    # A both-hemisphere master still carries an energy axis when loaded without
    # an ``energy=`` filter. kikuchipy orders it (hemisphere=2, energy, npx,
    # npx) — collapse the energy axis by picking the slice nearest ``energy``.
    if data.ndim == 4:
        e_idx = _nearest_energy_index(mp, energy, n_energy=data.shape[1])
        data = data[:, e_idx]

    if data.ndim != 3 or data.shape[0] != 2:
        raise ValueError(f"expected master data shape (2, npx, npx), got {data.shape}")
    # ascontiguousarray drops any negative strides from the slicing above.
    return np.ascontiguousarray(data)


def _reload_both_hemispheres(mp):
    """Reload a master as square-Lambert + BOTH hemispheres from its file.

    The master object retains its source path via kikuchipy's
    ``tmp_parameters`` (folder + filename). Reloading with
    ``projection='lambert', hemisphere='both'`` yields the (2, ...) Lambert
    stack the GPU projection needs.
    """
    import kikuchipy as kp

    path = _recover_master_path(mp)
    if path is None:
        raise MasterPatternError(
            "GPU dictionary projection needs the master in the square-Lambert "
            "projection with both hemispheres, but it was loaded as "
            f"{getattr(mp, 'projection', None)!r}/{getattr(mp, 'hemisphere', None)!r} "
            "and its source file could not be located to reload. Re-select the "
            "master pattern file and try again."
        )
    logger.info("Reloading master %s as Lambert + both hemispheres for GPU projection", path)
    return kp.load(path, projection="lambert", hemisphere="both")


def _recover_master_path(mp):
    """Best-effort recovery of the master's on-disk path from kikuchipy meta."""
    tp = getattr(mp, "tmp_parameters", None)
    folder = getattr(tp, "folder", "") if tp is not None else ""
    filename = getattr(tp, "filename", "") if tp is not None else ""
    if filename:
        for ext in ("", ".h5", ".hdf5"):
            cand = os.path.join(folder or ".", str(filename) + ext)
            if os.path.isfile(cand):
                return cand
    # Fallback: metadata original_filename (may be a bare basename).
    try:
        orig = mp.metadata.General.get_item("original_filename", "")
    except Exception:
        orig = ""
    if orig and os.path.isfile(orig):
        return orig
    return None


def _nearest_energy_index(mp, energy: float, *, n_energy: int) -> int:
    """Index of the master energy slice closest to ``energy`` (kV)."""
    try:
        for ax in mp.axes_manager.navigation_axes:
            if getattr(ax, "name", "") == "energy":
                energies = ax.offset + np.arange(ax.size) * ax.scale
                return int(np.argmin(np.abs(energies - energy)))
    except Exception:
        logger.debug("Could not read master energy axis; using last slice", exc_info=True)
    # No usable energy axis: highest energy is the last slice (EMsoft order).
    return n_energy - 1
