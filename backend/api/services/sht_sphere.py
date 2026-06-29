"""Reconstruct an EMSphInx ``.sht`` master pattern onto a sphere grid for 3D display.

Pure helper for the Database Browser's interactive master-pattern sphere viewer
(see ``docs/superpowers/specs/2026-06-22-sht-sphere-viewer-design.md``).  It reuses
the EXACT inverse-SHT the indexer/forward path already runs
(:func:`backend.api.services.sht_pattern_renderer.load_or_get_phase` →
``LambertGrid.grid``, a full-sphere Driscoll-Healy grid), downsamples it for the web,
and returns plain JSON-able data plus crystal metadata.

No new physics: the grid is the same reconstruction the renderer caches; we only
sample angles and pack arrays.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

# Project root: this file is backend/api/services/sht_sphere.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SHT_DB = _PROJECT_ROOT / "Database" / "EBSD_SHT_Database"


class SphereError(RuntimeError):
    """Raised when a ``.sht`` cannot be turned into a sphere (bad/missing file)."""


def _resolve_sht(filename: str) -> Path:
    """Resolve ``filename`` to an existing ``.sht`` under the SHT database.

    Mirrors the resolution used by ``database.preview_file``: try the path as given
    (relative to the SHT DB), then a recursive basename match.  Fail loud if absent.
    """
    if not filename:
        raise FileNotFoundError("empty filename")
    # Absolute path that is actually inside the SHT DB (defensive: stay in-tree).
    cand = (_SHT_DB / filename)
    if cand.is_file():
        return cand
    # Fall back to a literal basename match. NB: do NOT use rglob(name) — the SHT
    # filenames contain glob-special characters (``[Pearson]``, ``{kV}``), which the
    # glob engine would (mis)interpret as character classes / patterns. Iterate over
    # all .sht and compare the literal name instead.
    if _SHT_DB.is_dir():
        target = Path(filename).name
        for p in _SHT_DB.rglob("*.sht"):
            if p.name == target:
                return p
    raise FileNotFoundError(f"SHT file not found under EBSD_SHT_Database: {filename}")


def _downsample_stride(n: int, target: int) -> int:
    """Largest integer stride that keeps ≥ ``target`` samples along an ``n``-axis."""
    if target <= 0 or n <= target:
        return 1
    return max(1, n // target)


@lru_cache(maxsize=8)
def _sphere_payload_cached(resolved_path: str, max_bandwidth: int, target_size: int) -> dict:
    """Heavy part, cached on (path, bw, target).  Returns the JSON-able payload."""
    # Imports are local so importing this module is cheap (no torch at import time).
    from backend.api.services.sht_pattern_renderer import (  # noqa: PLC0415
        load_or_get_phase, SHTRenderError,
    )
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master  # noqa: PLC0415

    p = Path(resolved_path)
    try:
        lg = load_or_get_phase(str(p), max_bandwidth=max_bandwidth)
    except SHTRenderError as e:
        raise SphereError(f"could not reconstruct sphere from {p.name}: {e}") from e

    grid = lg.grid.detach().cpu().numpy().astype(np.float64)  # (2L, 2L): rows=azimuth, cols=polar
    n_phi, n_theta = grid.shape

    # Downsample for the web surface (keep ~target_size per axis).
    sp = _downsample_stride(n_phi, target_size)
    st = _downsample_stride(n_theta, target_size)
    g = grid[::sp, ::st]
    np_, nt_ = g.shape

    # Driscoll-Healy angle convention (see forward.py): row i → azimuth φ = π·i/L over
    # the ORIGINAL grid; col j → polar θ = (2j+1)·π/(4L).  L = n_phi/2.
    L = n_phi / 2.0
    rows = np.arange(0, n_phi, sp)[:np_]
    cols = np.arange(0, n_theta, st)[:nt_]
    phi = (math.pi * rows / L).tolist()                    # azimuth ∈ [0, 2π)
    theta = ((2.0 * cols + 1.0) * math.pi / (4.0 * L)).tolist()  # polar ∈ (0, π)

    meta: dict = {}
    try:
        m = read_sht_master(str(p), device="cpu")
        meta = {
            "formula": m.formula or "",
            "space_group": int(m.space_group),
            "point_group": m.point_group,
            "voltage_kV": round(float(m.voltage_kv), 2),
        }
    except Exception:
        meta = {}

    return {
        "filename": p.name,
        "grid": g.tolist(),            # (n_phi, n_theta) downsampled DH grid
        "phi": phi,                    # length n_phi (azimuth)
        "theta": theta,                # length n_theta (polar)
        "bandwidth": int(lg.bandwidth),
        "file_bandwidth": int(lg.file_bandwidth),
        "meta": meta,
    }


def sphere_from_sht(
    filename: str,
    *,
    max_bandwidth: int = 128,
    target_size: int = 128,
) -> dict:
    """Return JSON-able sphere data for an EMSphInx ``.sht`` master pattern.

    Parameters
    ----------
    filename : str
        Path (relative to ``Database/EBSD_SHT_Database``) or bare basename of a ``.sht``.
    max_bandwidth : int
        Inverse-SHT truncation bandwidth for the reconstruction (default 128 → a
        256×256 sphere grid; higher = sharper bands, slower).
    target_size : int
        Downsample the grid to ≈ this many samples per axis for the web surface.

    Returns
    -------
    dict
        ``{filename, grid (n_phi×n_theta), phi[], theta[], bandwidth, file_bandwidth,
        meta:{formula, space_group, point_group, voltage_kV}}``.

    Raises
    ------
    FileNotFoundError
        If the ``.sht`` does not exist under the SHT database.
    SphereError
        If the file exists but cannot be reconstructed (corrupt / unreadable).
    """
    p = _resolve_sht(filename)
    return _sphere_payload_cached(str(p), int(max_bandwidth), int(target_size))
