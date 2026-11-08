"""Synthetic square EBSD maps with known grains, for the grain-reconstruction tests.

Pure numpy in the generator (``make_map``, ``add_defects``, ``scratch_mask``) so
the MTEX reference script builds exactly
the same input. Quaternions (w, x, y, z) in orix's convention: crystal symmetry
acts from the left, a crystal-frame rotation multiplies from the left.
"""
from __future__ import annotations

import numpy as np


def _qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a, float), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def _axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = np.asarray(angle, float)[..., None] / 2.0
    return np.concatenate([np.cos(half), np.sin(half) * axis], axis=-1)


def random_quaternions(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q


def make_map(seed=3, shape=(60, 80), n_seeds=30, noise_deg=0.2, gradient_deg=3.0):
    """Voronoi grains with random orientations, a small gradient and pixel noise.

    The default seed was chosen so that every truth cell is 4-connected (seed 7
    produced a cell with a pixel that touches it only at a corner, which any
    pixel-edge segmentation correctly makes a grain of its own).

    grain 1 and grain 2 are neighbours related by 60 deg about <111> (Sigma3);
    grain 3 sits at phi2 = 45 deg where cubic equivalents meet; every fifth
    grain from grain 5 on is the tetragonal phase 1, all others phase 0.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    seeds = np.column_stack([rng.uniform(0, ny, n_seeds), rng.uniform(0, nx, n_seeds)])
    special = n_seeds >= 3
    if special:
        seeds[1] = seeds[0] + np.array([0.0, 9.0])
    yy, xx = np.mgrid[0:ny, 0:nx]
    d2 = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    truth = d2.argmin(axis=-1) + 1
    grain_q = random_quaternions(rng, n_seeds)
    if special:
        grain_q[1] = _qmul(_axis_angle([1, 1, 1], np.deg2rad(60.0)), grain_q[0])
        grain_q[2] = _axis_angle([0, 0, 1], np.deg2rad(45.0))
    phase_of_grain = np.where(np.arange(n_seeds) % 5 == 4, 1, 0)
    phase_of_grain[:3] = 0
    g_axis = rng.normal(size=(n_seeds, 3))
    rel_x = (xx - xx.mean()) / nx
    grad = _axis_angle(g_axis[truth - 1], np.deg2rad(gradient_deg) * rel_x)
    noise = _axis_angle(rng.normal(size=(ny, nx, 3)), np.deg2rad(rng.normal(0.0, noise_deg, (ny, nx))))
    clean = _qmul(grad, grain_q[truth - 1])
    q = _qmul(noise, clean)
    return {
        "quats": q,
        "clean": clean,             # noiseless orientation per pixel, one branch per grain
        "seed": seed,
        "phase_id": phase_of_grain[truth - 1].astype(np.int64),
        "truth": truth.astype(np.int64),
        "phase_names": np.array(["Al", "Al7FeCu2"]),
        "point_groups": np.array(["m-3m", "4/mmm"]),
        "step": 0.5,
    }


def scratch_mask(shape, p0, p1, width):
    """Pixels within width/2 of the segment p0-p1 (row, col)."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    (y0, x0), (y1, x1) = p0, p1
    d = np.array([y1 - y0, x1 - x0], float)
    length = np.hypot(*d)
    normal = np.array([-d[1], d[0]]) / length
    dist = np.abs((yy - y0) * normal[0] + (xx - x0) * normal[1])
    t = ((yy - y0) * d[0] + (xx - x0) * d[1]) / length ** 2
    return (dist <= width / 2.0) & (t >= 0.0) & (t <= 1.0)


def add_defects(m, seed=11, n_holes=15, n_outliers=40, scratch=True, wide=True, border=True):
    """Small holes, a 2 px scratch across the map, a wide square, a 1 px border strip, outliers."""
    rng = np.random.default_rng(seed)
    q = m["quats"].copy()
    phase = m["phase_id"].copy()
    ny, nx = phase.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    holes = np.zeros((ny, nx), bool)
    for _ in range(n_holes):
        cy, cx, r = rng.uniform(0, ny), rng.uniform(0, nx), rng.uniform(0.5, 1.5)
        holes |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    if scratch:
        holes |= scratch_mask((ny, nx), (5, 0), (ny - 6, nx - 1), width=2.0)
    if wide:
        holes[30:42, 50:62] = True
    if border:
        holes[10:30, 0] = True
    candidates = np.flatnonzero(~holes.ravel())
    outliers = rng.choice(candidates, n_outliers, replace=False)
    q.reshape(-1, 4)[outliers] = random_quaternions(rng, n_outliers)
    q[holes] = np.array([1.0, 0.0, 0.0, 0.0])
    phase[holes] = -1
    return {**m, "quats": q, "phase_id": phase, "holes": holes, "outliers": np.sort(outliers)}


def to_xmap(m):
    """orix CrystalMap of a synthetic map (test process only)."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation

    ny, nx = m["phase_id"].shape
    phases = PhaseList(names=list(m["phase_names"]), point_groups=list(m["point_groups"]), ids=[0, 1])
    return CrystalMap(
        rotations=Rotation(m["quats"].reshape(-1, 4)),
        phase_id=m["phase_id"].reshape(-1),
        x=np.tile(np.arange(nx, dtype=float) * m["step"], ny),
        y=np.repeat(np.arange(ny, dtype=float) * m["step"], nx),
        phase_list=phases,
    )
