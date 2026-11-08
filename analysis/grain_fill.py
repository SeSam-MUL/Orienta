"""Orientations for the closed pixels of a grain, grown inward from its measurements.

Display only: the filled orientations never enter a statistic. A closed pixel
takes the symmetry-aware mean of its already valued 4-neighbours of the same
grain; nothing crosses a grain boundary and open areas (grain 0) stay empty. A
closed pixel that no 4-path of its grain connects to a measurement takes the
mean orientation of the grain's measurements.
"""
from __future__ import annotations

import logging

import numpy as np

from analysis.orientation_math import label_means, nearest_equivalent

logger = logging.getLogger(__name__)

_SHIFTS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbour(a, dy, dx, fill):
    """out[y, x] = a[y + dy, x + dx], ``fill`` outside the map."""
    out = np.full(a.shape, fill, dtype=a.dtype)
    ny, nx = a.shape[:2]
    out[max(-dy, 0):ny - max(dy, 0), max(-dx, 0):nx - max(dx, 0)] = \
        a[max(dy, 0):ny - max(-dy, 0), max(dx, 0):nx - max(-dx, 0)]
    return out


def _grain_mean_fill(val, todo, quats, grain_ids, measured, grain_phase, phase_px, symmetries):
    """Give the pixels still ``todo`` the mean orientation of their grain's measurements (in place in
    ``val``); returns the mask of the pixels that received one."""
    quats = np.asarray(quats, dtype=np.float64)
    got = np.zeros(todo.shape, dtype=bool)
    n = len(grain_phase)
    use = measured & (grain_ids > 0) & np.isfinite(quats).all(axis=-1)
    for pid in np.unique(phase_px[todo]):
        want = todo & (phase_px == pid)
        px = use & np.isin(grain_ids, np.unique(grain_ids[want]))
        means = label_means(quats[px], grain_ids[px], n, symmetries[int(pid)])
        mean = means[np.maximum(grain_ids, 1) - 1]
        ok = want & np.isfinite(mean).all(axis=-1)
        val[ok] = mean[ok]
        got |= ok
    return got


def fill_closed_pixels(quats, grain_ids, measured, grain_phase, symmetries):
    """(filled (ny, nx, 4) NaN where not filled, filled_mask (ny, nx))."""
    grain_ids = np.asarray(grain_ids, dtype=np.int64)
    measured = np.asarray(measured, dtype=bool)
    ny, nx = grain_ids.shape
    in_grain = grain_ids > 0
    val = np.where((measured & in_grain)[..., None], np.asarray(quats, dtype=np.float64), np.nan)
    todo = in_grain & ~measured
    filled = np.zeros((ny, nx), dtype=bool)
    phase_px = np.full((ny, nx), -1, dtype=np.int64)
    phase_px[in_grain] = np.asarray(grain_phase)[grain_ids[in_grain] - 1]
    while todo.any():
        acc = np.zeros((ny, nx, 4))
        ref = np.full((ny, nx, 4), np.nan)
        cnt = np.zeros((ny, nx), dtype=np.int64)
        for dy, dx in _SHIFTS:
            nb_val = _neighbour(val, dy, dx, np.nan)
            nb_gid = _neighbour(grain_ids, dy, dx, 0)
            ok = todo & (nb_gid == grain_ids) & np.isfinite(nb_val).all(axis=-1)
            if not ok.any():
                continue
            first = ok & (cnt == 0)
            ref[first] = nb_val[first]
            for pid in np.unique(phase_px[ok]):
                s = ok & (phase_px == pid)
                acc[s] += nearest_equivalent(nb_val[s], ref[s], symmetries[int(pid)])
            cnt[ok] += 1
        front = todo & (cnt > 0)
        if not front.any():
            # a closed pixel can sit in a pocket of its grain's cells that no 4-path of the grain
            # connects to a measurement (a pixel only diagonal to its grain); it takes the grain mean
            n_left = int(todo.sum())
            got = _grain_mean_fill(val, todo, quats, grain_ids, measured, grain_phase, phase_px, symmetries)
            filled[got] = True
            logger.info("fill: %d closed pixels have no 4-path to a measurement of their grain; "
                        "%d took the mean orientation of their grain", n_left, int(got.sum()))
            break
        vals = acc[front]
        val[front] = vals / np.linalg.norm(vals, axis=1, keepdims=True)
        filled[front] = True
        todo &= ~front
    out = np.where(filled[..., None], val, np.nan)
    return out, filled
