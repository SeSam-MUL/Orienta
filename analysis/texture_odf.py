"""Texture index and entropy of the kernel density of the orientations, as the MTEX script computes them.

``TextureAnalysis_mtex6.m`` estimates the ODF of the pixels of the texture phase with
``calcDensity(ori, 'halfwidth', 5*degree)`` and writes ``norm(odf)^2`` (the texture index) and ``entropy(odf)``
to the workbook. This module computes the same two numbers from the same definition.

Definitions
-----------
* Haar measure normalised to 1. For a function of the rotation angle ``w`` alone
  ``int F(w(g)) dg = (2/pi) int_0^pi F(w) sin^2(w/2) dw``.
* de la Vallee Poussin kernel (Schaeben 1997, Phys. Stat. Sol. (b) 200, 367, doi:10.1002/1521-3951(199704)200:2<367::AID-PSSB367>3.0.CO;2-Z):
  ``psi(w) = C cos^(2 kappa)(w/2)``. The halfwidth ``h`` is the angle with ``psi(h) = psi(0)/2``, hence
  ``kappa = ln(1/2) / (2 ln cos(h/2))`` (363.959... for 5 degrees) and, from ``int psi = 1``,
  ``1/C = (2/pi) Gamma(kappa+1/2) Gamma(3/2) / Gamma(kappa+2)`` (12345.11... for 5 degrees).
* Density of ``N`` orientations ``q_i`` of a phase with the proper point group ``S``, triclinic specimen symmetry:
  ``f(g) = 1/(N |S|) sum_i sum_{s in S} psi(w(g, s q_i))``, ``int f = 1``. With unit quaternions
  ``cos(w/2) = |q_g . q_copy|``. The crystal symmetry acts from the LEFT (orix convention, as in
  ``analysis.orientation_math``), so ``f(t g) = f(g)`` for every ``t`` in ``S`` and a specimen rotation
  (from the right in orix) leaves ``f`` and both numbers unchanged.
* Texture index ``J = int f^2 dg`` (``norm(odf)^2``) and entropy ``S = -int f ln f dg`` (<= 0, uniform = 0).

Method
------
``J`` and ``S`` are integrals over SO(3) of smooth functions with the width of the kernel (about 4 degrees), so they
are computed by quadrature, with the density at each node summed exactly over the pixel copies that lie within the
cut-off of the kernel (the angle where ``psi`` is below 1e-9 of its peak, 27.6 degrees for 5 degrees; two different
copies of one pixel are at least 60 degrees apart, so at most one of them reaches a node).

* Nodes: the Hopf-type product grid of the unit quaternions ``(cos t cos a, cos t sin a, sin t cos b, sin t sin b)``
  with Gauss-Legendre nodes in ``sin^2 t`` and equally spaced angles ``a`` and ``b``. The Haar measure is
  ``dt da db`` in these variables, the angles are periodic (trapezoid rule, geometrically convergent) and ``q``
  and ``-q`` (the same rotation) are generated once.
* Symmetry: ``f`` is invariant under ``S``, so the integral over SO(3) is ``|S|`` times an integral over a fundamental
  zone. A hard fundamental-zone cut makes the jagged boundary the dominant error where a peak sits on a face; the
  integral is instead taken with the smooth partition of unity
  ``w(g) = rho(g) / sum_{s in S} rho(s g)`` with a bump ``rho`` that is a smooth function of ``g0^2`` (``g0`` the scalar
  part of the quaternion; a bump written in the rotation angle has a kink at ``w = pi`` that is inside its support
  for the groups with covering radius pi, see ``_partition``), compactly supported where the covering radius of the
  group allows it. ``sum_{s in S} w(s g) = 1`` gives
  ``int f^2 = |S| int w f^2``, and ``|S| int w = 1`` makes ``J = sum w f^2 / sum w`` independent of the node count.
* Neighbours: a cell hash of the (q1, q2, q3) coordinates of the copies, cell size a fraction of the cut-off chord,
  copies near the equator ``q0 = 0`` also hashed with the opposite sign. numba ``njit(parallel=True, cache=True)``.
"""
from __future__ import annotations

import math
import threading

import numpy as np
from numba import njit, prange

DEFAULT_HALFWIDTH_DEG = 5.0
# node spacing in the rotation angle as a fraction of the halfwidth; chosen from the measured convergence
DEFAULT_RESOLUTION_FRACTION = 0.8
# the kernel is truncated where it is below this fraction of its peak
_CUT_REL = 1e-9
# cell size = cut-off chord / this
_CELL_SPLIT = 3
# nodes with a partition weight below this are dropped
_W_MIN = 1e-13
# largest number of quadrature nodes (before the partition drops those outside the support): keeps the peak memory of
# the node construction near 1 GB. It bounds the resolution and so, with the default resolution of 0.8 x halfwidth,
# the halfwidth from below (about 1.8 degrees).
MAX_NODES = 15_000_000
# largest number of pixel copies (pixels x symmetry operators) that are hashed at once. The density at a node is a sum
# over the pixels, so the pixels are processed in blocks of this many copies and the partial densities added: the
# peak memory (about 260 bytes per copy, measured) stays under 1 GB for any number of pixels, and the result is the
# same up to the order of the floating-point additions. A map that fits in one block is computed exactly as before.
MAX_CHUNK_COPIES = 1_500_000
# support of the partition bump relative to the covering radius (in sin^2(w/2))
_SPAN_MARGIN = 2.0
# numba's workqueue threading layer (the one a build without TBB or OpenMP uses) aborts the process on concurrent
# launches of parallel functions, so only one call at a time runs the parallel kernels
_LAUNCH_LOCK = threading.Lock()


# ------------------------------------------------------------------------------------------------------------------
# the kernel
# ------------------------------------------------------------------------------------------------------------------
def kernel_constants(halfwidth_deg: float) -> tuple[float, float, float]:
    """``(kappa, log C, cut-off rotation angle in rad)`` of the de la Vallee Poussin kernel of a halfwidth."""
    h = math.radians(float(halfwidth_deg))
    if not np.isfinite(h) or h <= 0.0 or h >= math.pi:
        raise ValueError(f"halfwidth must be in (0, 180) degrees, got {halfwidth_deg!r}")
    kappa = math.log(0.5) / (2.0 * math.log(math.cos(h / 2.0)))
    log_inv_c = (math.log(2.0 / math.pi) + math.lgamma(kappa + 0.5) + math.lgamma(1.5) - math.lgamma(kappa + 2.0))
    log_c = -log_inv_c
    # cos^(2 kappa)(w/2) = _CUT_REL
    cut = 2.0 * math.acos(math.exp(math.log(_CUT_REL) / (2.0 * kappa)))
    return kappa, log_c, cut


# ------------------------------------------------------------------------------------------------------------------
# numba kernels
# ------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _copies(quats, ops, out):
    """``out[i * ns + k] = ops[k] * quats[i]`` as a unit quaternion with w >= 0 (Hamilton product, symmetry left)."""
    n = quats.shape[0]
    ns = ops.shape[0]
    for i in prange(n):
        a0 = quats[i, 0]
        a1 = quats[i, 1]
        a2 = quats[i, 2]
        a3 = quats[i, 3]
        nrm = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3)
        a0 /= nrm
        a1 /= nrm
        a2 /= nrm
        a3 /= nrm
        for k in range(ns):
            s0 = ops[k, 0]
            s1 = ops[k, 1]
            s2 = ops[k, 2]
            s3 = ops[k, 3]
            w = s0 * a0 - s1 * a1 - s2 * a2 - s3 * a3
            x = s0 * a1 + s1 * a0 + s2 * a3 - s3 * a2
            y = s0 * a2 - s1 * a3 + s2 * a0 + s3 * a1
            z = s0 * a3 + s1 * a2 - s2 * a1 + s3 * a0
            sg = 1.0
            if w < 0.0:
                sg = -1.0
            j = i * ns + k
            out[j, 0] = sg * w
            out[j, 1] = sg * x
            out[j, 2] = sg * y
            out[j, 3] = sg * z


@njit(cache=True)
def _cell_of(x, cs, dim):
    i = int((x + 1.0) / cs)
    if i < 0:
        i = 0
    elif i >= dim:
        i = dim - 1
    return i


@njit(cache=True)
def _build_hash(pts, cs, dim):
    """Counting sort of the points into cells of the (x, y, z) = (q1, q2, q3) coordinates.

    Returns ``(start, sorted_pts)``: the points of cell ``c`` are ``sorted_pts[start[c]:start[c + 1]]``.
    """
    m = pts.shape[0]
    ncell = dim * dim * dim
    start = np.zeros(ncell + 1, dtype=np.int64)
    cid = np.empty(m, dtype=np.int64)
    for j in range(m):
        c = (_cell_of(pts[j, 1], cs, dim) * dim + _cell_of(pts[j, 2], cs, dim)) * dim + _cell_of(pts[j, 3], cs, dim)
        cid[j] = c
        start[c + 1] += 1
    for c in range(ncell):
        start[c + 1] += start[c]
    fill = start[:-1].copy()
    out = np.empty((m, 4), dtype=np.float64)
    for j in range(m):
        c = cid[j]
        p = fill[c]
        fill[c] = p + 1
        for k in range(4):
            out[p, k] = pts[j, k]
    return start, out


@njit(parallel=True, cache=True)
def _density(grid, start, pts, cs, dim, reach, dcut, two_kappa, scale, out):
    """``out[k] = scale * sum over hashed points p with grid[k] . p >= dcut of (grid[k] . p)^(2 kappa)``."""
    m = grid.shape[0]
    for k in prange(m):
        g0 = grid[k, 0]
        g1 = grid[k, 1]
        g2 = grid[k, 2]
        g3 = grid[k, 3]
        if g0 < 0.0:
            g0 = -g0
            g1 = -g1
            g2 = -g2
            g3 = -g3
        ix = _cell_of(g1, cs, dim)
        iy = _cell_of(g2, cs, dim)
        iz = _cell_of(g3, cs, dim)
        x0 = max(ix - reach, 0)
        x1 = min(ix + reach, dim - 1)
        y0 = max(iy - reach, 0)
        y1 = min(iy + reach, dim - 1)
        z0 = max(iz - reach, 0)
        z1 = min(iz + reach, dim - 1)
        acc = 0.0
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                base = (cx * dim + cy) * dim
                for p in range(start[base + z0], start[base + z1 + 1]):
                    d = g0 * pts[p, 0] + g1 * pts[p, 1] + g2 * pts[p, 2] + g3 * pts[p, 3]
                    if d >= dcut:
                        acc += math.exp(two_kappa * math.log(d))
        out[k] = scale * acc


@njit(parallel=True, cache=True)
def _partition(grid, ops, span, out):
    """``out[k] = rho(g) / sum_s rho(s g)`` with ``rho = exp(1 - 1 / (1 - x^2))``, ``x = (1 - g0^2) / span``.

    ``rho`` is a function of ``g0^2`` only, where ``g0 = cos(w/2)`` is the scalar part of the quaternion ``g``
    (``1 - g0^2 = sin^2(w/2)``), and ``rho = 0`` for ``x >= 1``. A bump written in the rotation angle ``w`` itself is
    not smooth in the quaternion for a group whose covering radius is pi (C2, C3, C4, C6 and every group that has
    one of them as its proper part): its support then reaches ``w = pi``, ``g0 = 0``, where ``w = 2 acos|g0|`` has a
    V-shaped kink, and the trapezoid rule on a kink converges only algebraically (relative error 3e-3 in J for C3,
    C4, C6 at the default spacing). A function of ``g0^2`` is analytic there.
    """
    m = grid.shape[0]
    ns = ops.shape[0]
    for k in prange(m):
        g0 = grid[k, 0]
        g1 = grid[k, 1]
        g2 = grid[k, 2]
        g3 = grid[k, 3]
        den = 0.0
        for s in range(ns):
            # (s g)_0 = <conj(s), g>
            c = ops[s, 0] * g0 + ops[s, 1] * g1 + ops[s, 2] * g2 + ops[s, 3] * g3
            x = (1.0 - c * c) / span
            if x < 1.0:
                den += math.exp(1.0 - 1.0 / (1.0 - x * x))
        x = (1.0 - g0 * g0) / span
        num = 0.0
        if x < 1.0:
            num = math.exp(1.0 - 1.0 / (1.0 - x * x))
        if den > 0.0:
            out[k] = num / den
        else:
            out[k] = 0.0


@njit(parallel=True, cache=True)
def _min_angle_to_group(samples, ops, out):
    """Smallest rotation angle (rad) from each sample to the group orbit: ``min_s w(s q)``."""
    m = samples.shape[0]
    ns = ops.shape[0]
    for k in prange(m):
        best = 0.0
        for s in range(ns):
            c = abs(ops[s, 0] * samples[k, 0] + ops[s, 1] * samples[k, 1]
                    + ops[s, 2] * samples[k, 2] + ops[s, 3] * samples[k, 3])
            if c > best:
                best = c
        out[k] = 2.0 * math.acos(min(best, 1.0))


# ------------------------------------------------------------------------------------------------------------------
# nodes and weights (cached per point group and resolution)
# ------------------------------------------------------------------------------------------------------------------
_NODE_CACHE: dict = {}


def _grid_shape(resolution_deg: float) -> tuple[int, int]:
    """``(n_t, n_ang)`` of the product grid; it has ``n_t * n_ang / 2 * n_ang`` nodes."""
    dq = math.radians(resolution_deg) / 2.0                     # spacing in the quaternion angle
    n_t = max(6, int(math.ceil((math.pi / 2.0) / dq)))
    n_ang = 2 * max(4, int(math.ceil(math.pi / dq)))            # even: the shift by pi in (a, b) maps the grid to itself
    return n_t, n_ang


def _grid_size(resolution_deg: float) -> int:
    n_t, n_ang = _grid_shape(resolution_deg)
    return n_t * (n_ang // 2) * n_ang


def _check_grid_size(halfwidth_deg: float, resolution_deg: float) -> None:
    if not (np.isfinite(resolution_deg) and resolution_deg > 0.0):
        raise ValueError(f"resolution must be positive, got {resolution_deg!r}")
    n = _grid_size(resolution_deg)
    if n > MAX_NODES:
        res_min = resolution_deg
        while _grid_size(res_min) > MAX_NODES:
            res_min *= 1.01
        raise ValueError(
            f"a resolution of {resolution_deg:g} degrees needs {n:,} quadrature nodes, above the limit of {MAX_NODES:,}; "
            f"use a resolution of at least {res_min:.2f} degrees (with the default 0.8 x halfwidth that means a "
            f"halfwidth of at least {res_min / DEFAULT_RESOLUTION_FRACTION:.1f} degrees, here {halfwidth_deg:g})")


def _hopf_grid(resolution_deg: float):
    """Product quadrature on the unit quaternions modulo sign: ``(nodes (M, 4), weights (M,))``, weights sum to 1."""
    n_t, n_ang = _grid_shape(resolution_deg)
    x, wx = np.polynomial.legendre.leggauss(n_t)
    t = (x + 1.0) / 2.0
    wt = wx / 2.0
    ct, st = np.sqrt(1.0 - t), np.sqrt(t)
    a = (np.arange(n_ang // 2) + 0.5) * (2.0 * np.pi / n_ang)   # a in (0, pi): one of q, -q
    b = (np.arange(n_ang) + 0.5) * (2.0 * np.pi / n_ang)
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    grid = np.empty((n_t, len(a), len(b), 4))
    grid[..., 0] = ct[:, None, None] * ca[None, :, None]
    grid[..., 1] = ct[:, None, None] * sa[None, :, None]
    grid[..., 2] = st[:, None, None] * cb[None, None, :]
    grid[..., 3] = st[:, None, None] * sb[None, None, :]
    w = np.broadcast_to(wt[:, None, None], grid.shape[:3]) / (len(a) * len(b))
    return grid.reshape(-1, 4), np.ascontiguousarray(w.reshape(-1))


def _covering_radius(ops: np.ndarray) -> float:
    """Largest rotation angle (rad) from any rotation to the nearest element of its orbit (sampled, rounded up)."""
    rng = np.random.default_rng(12345)
    q = rng.normal(size=(300000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = np.empty(len(q))
    _min_angle_to_group(q, ops, out)
    return float(min(out.max() * 1.03 + math.radians(1.0), math.pi))


def _nodes(ops: np.ndarray, resolution_deg: float):
    """Quadrature nodes ``(Q (M, 4), W (M,))`` with the partition-of-unity weights of the group folded in."""
    key = (np.round(ops, 9).tobytes(), round(float(resolution_deg), 6))
    hit = _NODE_CACHE.get(key)
    if hit is not None:
        return hit
    grid, gw = _hopf_grid(resolution_deg)
    # support up to 2 x sin^2(R / 2) of the covering radius R (_SPAN_MARGIN): positive everywhere on the covering of SO(3) by the
    # orbit, and the whole sphere (span > 1) when R is pi
    span = _SPAN_MARGIN * math.sin(_covering_radius(ops) / 2.0) ** 2
    part = np.empty(len(grid))
    _partition(grid, ops, span, part)
    keep = part > _W_MIN
    nodes = np.ascontiguousarray(grid[keep])
    weights = np.ascontiguousarray(gw[keep] * part[keep])
    nodes.setflags(write=False)
    weights.setflags(write=False)
    if len(_NODE_CACHE) > 8:
        _NODE_CACHE.clear()
    _NODE_CACHE[key] = (nodes, weights)
    return nodes, weights


# ------------------------------------------------------------------------------------------------------------------
# public API
# ------------------------------------------------------------------------------------------------------------------
def _check(quats, ops, halfwidth_deg):
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats must have shape (n, 4), got {q.shape}")
    if len(q) == 0:
        raise ValueError("no orientations: the density of an empty set is undefined")
    if not np.all(np.isfinite(q)):
        raise ValueError("quats contains non-finite values; filter the unindexed pixels before calling")
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("quats contains a zero quaternion")
    o = np.asarray(ops, dtype=np.float64)
    if o.ndim != 2 or o.shape[1] != 4 or len(o) == 0 or not np.all(np.isfinite(o)):
        raise ValueError("ops must be a non-empty (k, 4) array of finite quaternions")
    if not (np.isfinite(halfwidth_deg) and halfwidth_deg > 0.0):
        raise ValueError(f"halfwidth must be positive, got {halfwidth_deg!r}")
    return np.ascontiguousarray(q / norms[:, None]), np.ascontiguousarray(o)


def _resolution(halfwidth_deg, resolution_deg):
    return DEFAULT_RESOLUTION_FRACTION * halfwidth_deg if resolution_deg is None else float(resolution_deg)


def kde_density_on_nodes(quats, ops, halfwidth_deg: float = DEFAULT_HALFWIDTH_DEG, resolution_deg=None):
    """The density ``f`` at the quadrature nodes: ``(f (M,), weights (M,))`` with ``int g dg ~ sum(weights * g) / sum(weights)``.

    ``weights`` is a read-only view of the cached node weights. One call at a time runs the parallel numba kernels
    (``_LAUNCH_LOCK``): numba's workqueue threading layer aborts the process on concurrent launches.

    The pixels are taken in blocks of at most ``MAX_CHUNK_COPIES`` copies (pixels x operators) and the partial
    densities are added, so the memory does not grow with the number of pixels (a million pixels of a cubic phase peak
    below 1 GB); the result differs from a single pass only by the order of the additions (relative 1e-15).
    """
    q, o = _check(quats, ops, halfwidth_deg)
    kappa, log_c, cut = kernel_constants(halfwidth_deg)
    resolution = _resolution(halfwidth_deg, resolution_deg)
    _check_grid_size(halfwidth_deg, resolution)
    ns = len(o)
    n = len(q)
    with _LAUNCH_LOCK:
        nodes, weights = _nodes(o, resolution)

        theta = cut / 2.0                                       # cut-off as a quaternion angle
        chord = 2.0 * math.sin(theta / 2.0)
        cs = chord / _CELL_SPLIT
        dim = int(math.ceil(2.0 / cs)) + 1
        scale = math.exp(log_c - math.log(n * ns))              # of ALL pixels: the blocks add up to one density
        block = max(1, MAX_CHUNK_COPIES // ns)                  # pixels per block

        f = None
        for first in range(0, n, block):
            copies = np.empty((min(block, n - first) * ns, 4))
            _copies(np.ascontiguousarray(q[first:first + block]), o, copies)
            # the opposite sign of the copies near the equator q0 = 0, where a node q0 >= 0 can sit next to -copy
            near_equator = copies[:, 0] <= math.sin(theta) * (1.0 + 1e-9)
            hashed = np.concatenate([copies, -copies[near_equator]]) if near_equator.any() else copies
            del copies
            start, sorted_pts = _build_hash(hashed, cs, dim)
            del hashed
            part = np.empty(len(nodes))
            _density(nodes, start, sorted_pts, cs, dim, _CELL_SPLIT, math.cos(theta), 2.0 * kappa, scale, part)
            del start, sorted_pts
            f = part if f is None else f + part
    return f, weights


def kde_texture_index_and_entropy(quats, ops, halfwidth_deg: float = DEFAULT_HALFWIDTH_DEG,
                                  resolution_deg=None) -> tuple[float, float]:
    """Texture index ``J = int f^2`` and entropy ``S = -int f ln f`` of the kernel density of orientations.

    ``quats``: (n, 4) orix-convention quaternions of the pixels (finite; filter unindexed pixels first).
    ``ops``: the proper rotations of the phase (k, 4), acting from the left (``analysis.orientation_math.proper_ops``).
    ``halfwidth_deg``: halfwidth of the de la Vallee Poussin kernel (the script uses 5 degrees).
    ``resolution_deg``: spacing of the quadrature nodes in the rotation angle (default 0.8 x the halfwidth).

    Raises ``ValueError`` for no orientation, non-finite quaternions or a non-positive halfwidth.
    """
    f, w = kde_density_on_nodes(quats, ops, halfwidth_deg, resolution_deg)
    wsum = w.sum()
    j = float(np.dot(w, f * f) / wsum)
    pos = f > 0.0
    s = float(-np.dot(w[pos], f[pos] * np.log(f[pos])) / wsum)
    return j, s


def kde_mass(quats, ops, halfwidth_deg: float = DEFAULT_HALFWIDTH_DEG, resolution_deg=None) -> float:
    """``int f dg`` by the same quadrature (1 up to the quadrature and cut-off error); a check of the normalisation."""
    f, w = kde_density_on_nodes(quats, ops, halfwidth_deg, resolution_deg)
    return float(np.dot(w, f) / w.sum())


def texture_index_and_entropy(dataset, pixel_mask, halfwidth_deg: float = DEFAULT_HALFWIDTH_DEG,
                             resolution_deg=None) -> tuple[float, float]:
    """``(texture index, entropy)`` of the orientations of the masked pixels, with the symmetry of THEIR phase.

    ``pixel_mask``: bool (size,) of the pixels that enter (the pixels of the texture phase, which can be any phase
    of the map). Every masked pixel must be indexed and all of them of ONE phase id, whose proper point group
    (``analysis.grain_reconstruction.phase_ops``) is the symmetry used; otherwise ``ValueError`` names what was found.
    Masked pixels whose quaternion is not finite are left out (``ValueError`` if nothing remains).
    """
    from analysis.grain_reconstruction import phase_ops

    mask = np.asarray(pixel_mask, dtype=bool).reshape(-1)
    xmap = dataset.xmap
    quats = np.asarray(xmap.rotations.data, dtype=np.float64).reshape(-1, 4)
    ids = np.asarray(xmap.phase_id).reshape(-1)
    if len(mask) != len(quats) or len(ids) != len(quats):
        raise ValueError(f"pixel_mask has {len(mask)} entries, the map has {len(quats)} pixels")
    if not mask.any():
        raise ValueError("pixel_mask selects no pixel")
    found = np.unique(ids[mask])
    if np.any(found < 0):
        raise ValueError(f"pixel_mask contains not-indexed pixels (phase ids found: {found.tolist()})")
    if len(found) != 1:
        raise ValueError(f"pixel_mask mixes phases (phase ids found: {found.tolist()}); "
                         "the texture index is defined for the pixels of one phase")
    pid = int(found[0])
    ops = phase_ops(xmap).get(pid)
    if ops is None:
        raise ValueError(f"phase id {pid} has no point group; its proper rotations are unknown")
    q = quats[mask]
    q = q[np.all(np.isfinite(q), axis=1)]
    if len(q) == 0:
        raise ValueError("no pixel with a finite orientation in the mask")
    return kde_texture_index_and_entropy(q, ops, halfwidth_deg, resolution_deg)
