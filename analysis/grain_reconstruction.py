"""Grain reconstruction on a square EBSD grid, with narrow not-indexed areas closed.

Own implementation of the published method (no code taken from MTEX):
Bachmann, Hielscher, Schaeben, "Grain detection from 2d and 3d EBSD data --
Specification of the MTEX algorithm", Ultramicroscopy 111 (2011) 1720-1733,
doi:10.1016/j.ultramic.2011.08.002. Every measurement owns a cell of the map;
cells of similar measurements are joined; areas without a measurement are
covered by the surrounding cells only where they are narrow.

On a square grid that is three steps:

1. a morphological closing of the measured mask with a disc of radius
   ``alpha`` pixels decides which not-indexed pixels are covered (narrow areas,
   scratches) and which stay open; outside the map counts as measured, so the
   border never opens anything;
2. the cells are the exact Voronoi cells of the remaining pixels (the
   measurements and the open pixels, ``analysis/grain_cells.py``): a covered
   pixel is no generator, its area belongs to the cells around it;
3. two measurements are neighbours when their cells share an edge;
   neighbours of the same phase with a disorientation below the threshold are
   joined (connected components). A measurement in a grain below
   ``min_pixels`` is treated as not indexed and the map is segmented once more.
   Every pixel without a measurement belongs to the grain whose cell holds its
   centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from analysis.grain_cells import Cells, build_cells, pixel_hosts
from analysis.orientation_math import disorientation_angle, proper_ops


@dataclass
class Segmentation:
    grain_ids: np.ndarray    # (ny, nx) int64, 0 = no grain
    n_grains: int
    measured: np.ndarray     # (ny, nx) bool, the measurements the grains are made of
    closed: np.ndarray       # (ny, nx) bool, in a grain but not one of its measurements
    culled: np.ndarray       # (ny, nx) bool, measurements dropped by min_pixels
    grain_phase: np.ndarray  # (n_grains,) int64 phase id of each grain
    cells: Cells             # cells of the last segmentation pass; the grains' values are read from them
    dropped: np.ndarray      # (ny, nx) bool, the part of ``culled`` dropped after the second pass (still generators)


def grid_arrays(dataset):
    """Unit quaternions (ny, nx, 4) and phase ids (ny, nx) of a full square-grid dataset.

    The quaternions are normalised: data that went through float32 is off by
    about 1e-7 in norm, which gives ~0.07 deg false angles between identical
    orientations. A row that is not finite or has norm 0 is not a measurement (it
    would be 180 deg from everything): it is set to NaN, like a not-indexed pixel.
    """
    ny, nx = dataset.shape
    q = np.asarray(dataset.xmap.rotations.data, dtype=np.float64).reshape(-1, 4).copy()
    pid = np.asarray(dataset.xmap.phase_id).reshape(-1).astype(np.int64)
    if len(q) != ny * nx or len(pid) != ny * nx:
        raise ValueError(
            f"the map holds {len(q)} orientations for a {ny} x {nx} grid; "
            "grain analysis needs every grid point")
    norm = np.linalg.norm(q, axis=1)
    ok = np.isfinite(norm) & (norm > 0)
    q[ok] /= norm[ok, None]
    q[~ok] = np.nan
    return q.reshape(ny, nx, 4), pid.reshape(ny, nx)


def phase_ops(xmap) -> dict:
    """Proper rotations per phase id (frame-aware point groups); phases without one are left out."""
    from crystal_symmetry import phase_point_group

    out = {}
    for pid in xmap.phases.ids:
        if int(pid) < 0:
            continue
        pg = phase_point_group(xmap.phases[pid])
        if pg is not None:
            out[int(pid)] = proper_ops(pg)
    return out


def _disc(alpha: float) -> np.ndarray:
    r = int(np.floor(alpha))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= alpha * alpha + 1e-9


def covered_mask(measured, alpha: float) -> np.ndarray:
    """Measured pixels plus the not-indexed pixels a closing of radius ``alpha`` covers."""
    measured = np.asarray(measured, dtype=bool)
    if alpha <= 0 or measured.all() or not measured.any():
        return measured.copy()
    disc = _disc(alpha)
    pad = disc.shape[0]
    padded = np.pad(measured, pad, mode="constant", constant_values=True)
    closed = ndi.binary_closing(padded, structure=disc)
    return closed[pad:-pad, pad:-pad] | measured


def _measured_pairs(cells, measured_flat) -> np.ndarray:
    """Pairs (a < b) of measurements whose cells share an edge, unique."""
    ec = cells.edge_cells
    ok = (ec[:, 0] >= 0) & (ec[:, 1] >= 0)
    ok[ok] = measured_flat[ec[ok, 0]] & measured_flat[ec[ok, 1]]
    pairs = ec[ok]
    if len(pairs) == 0:
        return pairs.reshape(0, 2)
    return np.unique(pairs, axis=0)


def _join(pairs, q, phase, symmetries, threshold) -> np.ndarray:
    pa, pb = phase[pairs[:, 0]], phase[pairs[:, 1]]
    join = np.zeros(len(pairs), dtype=bool)
    same = pa == pb
    for pid in np.unique(pa[same]):
        sel = same & (pa == pid)
        ang = disorientation_angle(q[pairs[sel, 0]], q[pairs[sel, 1]], symmetries[int(pid)])
        join[sel] = ang < threshold
    return join


def _number(n, pairs, join, measured_flat):
    p = pairs[join]
    graph = coo_matrix((np.ones(len(p), dtype=np.int8), (p[:, 0], p[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    labels = np.zeros(n, dtype=np.int64)
    idx = np.flatnonzero(measured_flat)
    if len(idx) == 0:
        return labels, 0
    _, first, inverse = np.unique(comp[idx], return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(1, len(first) + 1)
    labels[idx] = rank[inverse.ravel()]
    return labels, len(first)


def _once(q, phase, measured, symmetries, threshold, alpha):
    """One pass: cells of the measurements and the open pixels, joined and numbered measurements."""
    covered = covered_mask(measured, alpha)
    cells = build_cells(~(covered & ~measured))
    pairs = _measured_pairs(cells, measured.ravel())
    join = _join(pairs, q, phase, symmetries, threshold)
    labels, n = _number(measured.size, pairs, join, measured.ravel())
    return cells, labels, n


def _renumber(labels):
    used = np.unique(labels[labels > 0])
    lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int64)
    lut[used] = np.arange(1, len(used) + 1)
    return lut[labels], len(used)


def segment(quats, phase_id, symmetries, threshold, min_pixels: int = 1, alpha: float = 3.1) -> Segmentation:
    """Grains of a square map. ``threshold`` in radians, ``alpha`` in pixels."""
    quats = np.asarray(quats, dtype=np.float64)
    phase_id = np.asarray(phase_id)
    if quats.ndim != 3 or quats.shape[-1] != 4 or phase_id.shape != quats.shape[:2]:
        raise ValueError("quats must be (ny, nx, 4) and phase_id (ny, nx)")
    if not threshold > 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if min_pixels < 1:
        raise ValueError(f"min_pixels must be >= 1, got {min_pixels}")
    ny, nx = phase_id.shape
    q = quats.reshape(-1, 4)
    ph = phase_id.reshape(-1).astype(np.int64)
    measured0 = ((ph >= 0) & np.isfinite(q).all(axis=1)).reshape(ny, nx)
    present = set(np.unique(ph[measured0.ravel()]).tolist())
    missing = sorted(present - set(symmetries))
    if missing:
        raise ValueError(f"no point group for phase id(s) {missing}; cannot reconstruct grains")
    empty = np.zeros((ny, nx), dtype=bool)
    if not measured0.any():
        return Segmentation(np.zeros((ny, nx), np.int64), 0, empty, empty.copy(), empty.copy(),
                            np.zeros(0, np.int64), build_cells(np.ones((ny, nx), dtype=bool)), empty.copy())

    measured = measured0.copy()
    dropped = empty.copy()
    cells, labels, n = _once(q, ph, measured, symmetries, threshold, alpha)
    if min_pixels > 1:
        counts = np.bincount(labels, minlength=n + 1)
        small = (labels > 0) & (counts[labels] < min_pixels)
        if small.any():
            measured &= ~small.reshape(ny, nx)
            cells, labels, n = _once(q, ph, measured, symmetries, threshold, alpha)
            counts = np.bincount(labels, minlength=n + 1)
            small = (labels > 0) & (counts[labels] < min_pixels)
            if small.any():
                # still too small after the second pass: dropped without a third pass; the cells of the
                # second pass stay as they are (the reference measures its grain areas on them)
                dropped = small.reshape(ny, nx)
                measured &= ~dropped
                labels[small] = 0
                labels, n = _renumber(labels)

    # a measurement keeps its grain, every other pixel takes the grain whose cell holds its centre
    gid = pixel_hosts(cells, labels).reshape(ny, nx)
    grain_phase = np.zeros(n, dtype=np.int64)
    idx = np.flatnonzero(labels)
    grain_phase[labels[idx] - 1] = ph[idx]
    closed = (gid > 0) & ~measured
    culled = measured0 & ~measured
    return Segmentation(gid.astype(np.int64), int(n), measured, closed, culled, grain_phase, cells, dropped)
