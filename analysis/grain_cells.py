"""Exact grain cells on a square pixel grid: edges, areas, moments, boundary loops and pixel hosts.

Method: the cell of a pixel is its Voronoi cell among all generator pixels plus a ring of exterior
generators one pixel row/column outside the map (Bachmann, Hielscher, Schaeben, Ultramicroscopy 111
(2011) 1720-1733, doi:10.1016/j.ultramic.2011.08.002). Which pixels are generators is decided by the
caller (see ``grain_reconstruction.covered_mask``); where nothing is missing a cell is the pixel square.

The conventions used here - the exterior ring, edge lengths, cell areas and area moments, the boundary
loops with the grain on the left and the sharpest-left-turn rule at pinch vertices, and the crossing
test for pixel centres that lie on a cell edge - were measured equal to MTEX for Python as a black box;
no MTEX code was used.

Coordinates: x = column, y = row, pixel centres at integer positions, lengths in pixel units.
Labels are flat ``(ny * nx,)`` integer arrays indexed by pixel; only the entries of generator pixels
are read, 0 means "no grain".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Voronoi, cKDTree

# Tolerances. Qhull returns Voronoi vertices with ~1e-12 of rounding noise (pixel units). The vertices of
# integer generator points are rational numbers whose distinct values differ by far more than 1e-7
# within the circumradius of a cell, so merging vertices closer than _VERTEX_MERGE only joins the
# copies of one point, and an edge not longer than _MIN_EDGE is a rounding artefact of a four-fold
# vertex. The regime where this could fail is an empty area hundreds of pixels wide (huge cells whose
# vertex coordinates are ill-conditioned): two distinct vertices could then be merged or an edge be
# dropped. That breaks the closed boundary, which the successor check in ``boundary_lengths`` reports as
# a RuntimeError; otherwise the cells would carry a vertex merged by mistake.
_MIN_EDGE = 1e-9        # edges not longer than this are dropped
_VERTEX_MERGE = 1e-7    # Voronoi vertices closer than this are one point

# Peak memory of the Qhull path, per generator pixel. Measured on synthetic maps with random 20 %
# not-indexed speckle: 3.2 million generators (a 2000 x 2000
# map) peaked at 6.35 GB resident, which is about 2 GB per 10^6 generators; 2.2 GB per 10^6 is that
# figure with a margin of 10 %. The cost is the ``Voronoi`` object (Python lists of ridges and
# vertices), so it scales with the number of generators, not with the grain count. The full-grid path
# needs no Qhull and is never refused.
QHULL_BYTES_PER_GENERATOR = 2.2e9 / 1e6


def available_memory_bytes() -> int:
    """Memory the process can still take, in bytes (psutil's ``available``); tests monkeypatch it."""
    import psutil

    return int(psutil.virtual_memory().available)


@dataclass(frozen=True)
class Cells:
    """Cells of the generator pixels of a ``shape = (ny, nx)`` map.

    ``edge_cells``: flat pixel indices of the two cells of each edge, ``-1`` = exterior. When both are
    map cells the first is the smaller; an edge to the exterior has ``-1`` in the second column. Two
    cells can share several edges, and a border cell can have several ``(a, -1)`` edges (one per
    exterior ring generator it touches), so consumers must sum ``edge_length`` per pair.
    ``edge_xy``: x0, y0, x1, y1 of the edge; ``edge_vertex``: ids of its end points (coincident points
    share one id). ``area`` and ``moments`` are per flat pixel and 0 for non-generators; ``moments`` are
    the integrals over the cell of dx, dy, dx*dx, dy*dy, dx*dy in coordinates relative to the cell's own
    generator pixel centre (dx = x - column, dy = y - row), which keeps them exact far from the origin
    (``grain_moments`` shifts them by integer pixel offsets).
    """
    shape: tuple
    generator: np.ndarray
    edge_cells: np.ndarray
    edge_length: np.ndarray
    edge_xy: np.ndarray
    edge_vertex: np.ndarray
    area: np.ndarray
    moments: np.ndarray


def _empty_cells(generator):
    ny, nx = generator.shape
    return Cells(shape=(ny, nx), generator=generator.copy(),
                 edge_cells=np.zeros((0, 2), np.int64), edge_length=np.zeros(0),
                 edge_xy=np.zeros((0, 4)), edge_vertex=np.zeros((0, 2), np.int64),
                 area=np.zeros(ny * nx), moments=np.zeros((ny * nx, 5)))


def _full_grid_cells(generator):
    """All pixels are generators: every cell is the pixel square (no Voronoi needed)."""
    ny, nx = generator.shape
    flat = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    vid = lambda i, j: i * (nx + 1) + j  # noqa: E731  corner (i, j) sits at x = j - 0.5, y = i - 0.5
    rows = np.arange(ny)[:, None]
    cols = np.arange(nx)[None, :]
    parts = []       # (cell_a, cell_b, corner_i0, corner_j0, corner_i1, corner_j1)

    def add(a, b, i0, j0, i1, j1):
        parts.append(np.stack([np.ravel(x) for x in np.broadcast_arrays(a, b, i0, j0, i1, j1)], axis=1))

    if nx > 1:      # left/right neighbours share the vertical line x = c + 0.5
        c = np.arange(nx - 1)[None, :]
        add(flat[:, :-1], flat[:, 1:], rows, c + 1, rows + 1, c + 1)
    if ny > 1:      # upper/lower neighbours share the horizontal line y = r + 0.5
        r = np.arange(ny - 1)[:, None]
        add(flat[:-1, :], flat[1:, :], r + 1, cols, r + 1, cols + 1)
    add(flat[:, 0], -1, np.arange(ny), 0, np.arange(ny) + 1, 0)                    # left border
    add(flat[:, -1], -1, np.arange(ny), nx, np.arange(ny) + 1, nx)                 # right border
    add(flat[0, :], -1, 0, np.arange(nx), 0, np.arange(nx) + 1)                    # top border
    add(flat[-1, :], -1, ny, np.arange(nx), ny, np.arange(nx) + 1)                 # bottom border
    e = np.concatenate(parts).astype(np.int64)
    edge_cells = e[:, :2]
    v0, v1 = vid(e[:, 2], e[:, 3]), vid(e[:, 4], e[:, 5])
    xy = np.stack([e[:, 3] - 0.5, e[:, 2] - 0.5, e[:, 5] - 0.5, e[:, 4] - 0.5], axis=1)
    mom = np.tile([0.0, 0.0, 1 / 12, 1 / 12, 0.0], (ny * nx, 1))     # unit square about its own centre
    return Cells(shape=(ny, nx), generator=generator.copy(), edge_cells=edge_cells,
                 edge_length=np.ones(len(e)), edge_xy=xy,
                 edge_vertex=np.stack([v0, v1], axis=1).astype(np.int64),
                 area=np.ones(ny * nx), moments=mom)


def _exterior_ring(ny, nx):
    c = np.arange(-1, nx + 1)
    r = np.arange(0, ny)
    ring = np.concatenate([
        np.column_stack([c, np.full_like(c, -1)]), np.column_stack([c, np.full_like(c, ny)]),
        np.column_stack([np.full_like(r, -1), r]), np.column_stack([np.full_like(r, nx), r]),
    ])
    return ring.astype(np.float64)


def _triangle_moments(p, v0, v1):
    """(area, 5 integrals of dx, dy, dx*dx, dy*dy, dx*dy) over the triangles (p, v0, v1), in coordinates
    relative to p; p, v0, v1 are (m, 2) arrays."""
    bx, by = v0[:, 0] - p[:, 0], v0[:, 1] - p[:, 1]
    cx, cy = v1[:, 0] - p[:, 0], v1[:, 1] - p[:, 1]
    area = 0.5 * np.abs(bx * cy - cx * by)
    sx, sy = bx + cx, by + cy
    return area, np.stack([area * sx / 3, area * sy / 3, area * (bx * bx + cx * cx + sx * sx) / 12,
                           area * (by * by + cy * cy + sy * sy) / 12,
                           area * (bx * by + cx * cy + sx * sy) / 12], axis=1)


def _merge_vertices(xy):
    """Ids for points; points closer than ``_VERTEX_MERGE`` share one id."""
    m = len(xy)
    pairs = cKDTree(xy).query_pairs(_VERTEX_MERGE, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(m, dtype=np.int64)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    return connected_components(g, directed=False)[1].astype(np.int64)


def build_cells(generator: np.ndarray) -> Cells:
    """Cells of all generator pixels (True) of a map, including the exterior ring (see module doc).

    A map in which every pixel is a generator needs no Voronoi diagram (every cell is the pixel
    square) and is built at once. Any other map goes through Qhull, whose peak memory was measured at
    about 2 GB per million generators (6.35 GB resident for 3.2 million, a 2000 x 2000 map with 20 %
    not-indexed speckle; see ``QHULL_BYTES_PER_GENERATOR``). Before Qhull runs, the estimate
    (``QHULL_BYTES_PER_GENERATOR`` times the number of generators) is compared with the memory the
    machine has available; if it does not fit, a ``MemoryError`` names the map size, the estimate and
    the available memory instead of letting the operating system start swapping or kill the process.
    """
    generator = np.asarray(generator, dtype=bool)
    if generator.ndim != 2:
        raise ValueError("generator must be a 2D boolean array")
    ny, nx = generator.shape
    if generator.all():
        return _full_grid_cells(generator)
    ys, xs = np.nonzero(generator)
    n = len(xs)
    if n == 0:
        return _empty_cells(generator)
    need = n * QHULL_BYTES_PER_GENERATOR
    have = available_memory_bytes()
    if need > have:
        raise MemoryError(
            f"Grain reconstruction needs about {need / 1e9:.2f} GB of memory for this {ny} x {nx} map "
            f"with {n:,} measured or uncovered pixels (about 2.2 GB per million), but only "
            f"{have / 1e9:.2f} GB available. Crop the map to a smaller region, or use a machine with "
            f"more memory.")
    pts = np.column_stack([xs, ys]).astype(np.float64)
    flat = (ys * nx + xs).astype(np.int64)
    vor = Voronoi(np.concatenate([pts, _exterior_ring(ny, nx)]))
    rp = np.asarray(vor.ridge_points)
    rv = np.asarray(vor.ridge_vertices, dtype=np.int64)
    keep = (rp[:, 0] < n) | (rp[:, 1] < n)
    rp, rv = rp[keep], rv[keep]
    if (rv < 0).any():
        raise RuntimeError("unbounded Voronoi ridge at a map cell")
    seg = np.concatenate([vor.vertices[rv[:, 0]], vor.vertices[rv[:, 1]]], axis=1)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    cell = np.where(rp < n, flat[np.minimum(rp, n - 1)], -1)
    swap = (cell[:, 0] < 0) | ((cell[:, 1] >= 0) & (cell[:, 0] > cell[:, 1]))
    cell[swap] = cell[swap][:, ::-1]

    # area and moments from EVERY bounded ridge (also the ones dropped from the edge list below, so no
    # area is lost): each ridge adds the triangle (generator, v0, v1) to each map cell it borders
    area = np.zeros(ny * nx)
    mom = np.zeros((ny * nx, 5))
    gx, gy = cell % nx, cell // nx
    for side in (0, 1):
        sel = cell[:, side] >= 0
        if not sel.any():
            continue
        p = np.column_stack([gx[sel, side], gy[sel, side]]).astype(np.float64)
        a, m5 = _triangle_moments(p, seg[sel, :2], seg[sel, 2:])
        idx = cell[sel, side]
        area += np.bincount(idx, weights=a, minlength=ny * nx)
        for k in range(5):
            mom[:, k] += np.bincount(idx, weights=m5[:, k], minlength=ny * nx)

    # edge list: drop ridges not longer than _MIN_EDGE and ridges that collapse to one point
    ok = length > _MIN_EDGE
    rv, cell, seg, length = rv[ok], cell[ok], seg[ok], length[ok]
    used, inverse = np.unique(rv.ravel(), return_inverse=True)
    vid = _merge_vertices(vor.vertices[used])[inverse].reshape(-1, 2)
    distinct = vid[:, 0] != vid[:, 1]
    cell, seg, length, vid = cell[distinct], seg[distinct], length[distinct], vid[distinct]
    order = np.lexsort((cell[:, 1], cell[:, 0]))
    cell, seg, length, vid = cell[order], seg[order], length[order], vid[order]
    return Cells(shape=(ny, nx), generator=generator.copy(), edge_cells=cell, edge_length=length,
                 edge_xy=seg, edge_vertex=vid, area=area, moments=mom)


# ---------------------------------------------------------------------------------------------
# boundary segments and loops
# ---------------------------------------------------------------------------------------------
def _check_labels(cells, labels):
    """Flat int64 labels; integer dtype, length ny*nx, no negative label on a generator pixel (a negative
    label would collide with the internal -1 sentinel). Other entries are never read."""
    labels = np.asarray(labels)
    if labels.shape != (cells.shape[0] * cells.shape[1],):
        raise ValueError("labels must be a flat (ny * nx,) array")
    if labels.dtype.kind not in "iu":
        raise ValueError("labels must be an integer array")
    labels = labels.astype(np.int64, copy=False)
    gl = labels[cells.generator.ravel()]
    if gl.size and int(gl.min()) < 0:
        raise ValueError("labels of generator pixels must be >= 0")
    return labels


def _directed_segments(cells, labels):
    """Boundary segments of every label > 0, directed so that the labelled cell's generator is on the
    left (``cross(v1 - v0, p - v0) > 0``). Returns label, x0, y0, x1, y1, vertex0, vertex1 arrays."""
    nx = cells.shape[1]
    ec = cells.edge_cells
    lab = np.where(ec >= 0, labels[np.maximum(ec, 0)], 0)
    diff = lab[:, 0] != lab[:, 1]
    parts = []
    for side in (0, 1):
        sel = diff & (ec[:, side] >= 0) & (lab[:, side] > 0)
        if not sel.any():
            continue
        own = ec[sel, side]
        px, py = own % nx, own // nx
        x0, y0, x1, y1 = cells.edge_xy[sel].T
        u, w = cells.edge_vertex[sel].T
        fwd = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) > 0
        parts.append((lab[sel, side], np.where(fwd, x0, x1), np.where(fwd, y0, y1),
                      np.where(fwd, x1, x0), np.where(fwd, y1, y0), np.where(fwd, u, w), np.where(fwd, w, u)))
    if not parts:
        e = np.zeros(0)
        return (e.astype(np.int64), e, e, e, e, e.astype(np.int64), e.astype(np.int64))
    return tuple(np.concatenate(c) for c in zip(*parts))


def _successors(lab, x0, y0, x1, y1, v0, v1):
    """Next segment of the same label at the end vertex; at a pinch vertex (several outgoing segments)
    the one with the largest ``atan2(cross(d_in, d_out), dot(d_in, d_out))``."""
    m = len(lab)
    nv = int(max(v0.max(), v1.max())) + 1
    start_key = lab * nv + v0
    order = np.argsort(start_key, kind="stable")
    sorted_keys = start_key[order]
    end_key = lab * nv + v1
    left = np.searchsorted(sorted_keys, end_key, side="left")
    right = np.searchsorted(sorted_keys, end_key, side="right")
    cnt = right - left
    if (cnt == 0).any():
        raise RuntimeError("open grain boundary: a segment has no successor")
    succ = order[np.minimum(left, m - 1)]
    multi = np.flatnonzero(cnt > 1)
    if multi.size:
        c = cnt[multi]
        group_start = np.cumsum(c) - c
        total = int(c.sum())
        rep_in = np.repeat(multi, c)
        out = order[np.repeat(left[multi], c) + np.arange(total) - np.repeat(group_start, c)]
        dix, diy = (x1 - x0)[rep_in], (y1 - y0)[rep_in]
        dox, doy = (x1 - x0)[out], (y1 - y0)[out]
        turn = np.arctan2(dix * doy - diy * dox, dix * dox + diy * doy)
        best = np.maximum.reduceat(turn, group_start)
        pos = np.where(turn == np.repeat(best, c), np.arange(total), total)
        succ[multi] = out[np.minimum.reduceat(pos, group_start)]
    if np.unique(succ).size != m:
        raise RuntimeError("inconsistent grain boundary at a pinch vertex")
    return succ


def boundary_lengths(cells: Cells, labels: np.ndarray, n: int):
    """``(outer, all_loops)``: for the labels 1..n (index 0 = label 1) the length of the outer boundary
    loop (the loop with the largest absolute signed area) and the summed length of all loops, in pixel
    units. The exterior counts as label 0; a label without cells has length 0. Equal absolute areas
    (within 1e-9 * max(1, largest |area| of the label)) are decided by the longer loop, then by the
    smallest vertex id on the loop."""
    labels = _check_labels(cells, labels)
    gl = labels[cells.generator.ravel()]
    if gl.size and int(gl.max()) > n:
        raise ValueError("a label exceeds n")
    outer = np.zeros(n)
    allc = np.zeros(n)
    lab, x0, y0, x1, y1, v0, v1 = _directed_segments(cells, labels)
    m = len(lab)
    if m == 0:
        return outer, allc
    succ = _successors(lab, x0, y0, x1, y1, v0, v1)
    g = coo_matrix((np.ones(m), (np.arange(m), succ)), shape=(m, m))
    n_loops, loop = connected_components(g, directed=False)
    seg_len = np.hypot(x1 - x0, y1 - y0)
    loop_len = np.bincount(loop, weights=seg_len, minlength=n_loops)
    # signed area by the shoelace sum about the loop's first vertex (not the origin): less cancellation
    first_seg = np.full(n_loops, m)
    np.minimum.at(first_seg, loop, np.arange(m))
    rx, ry = x0[first_seg][loop], y0[first_seg][loop]
    loop_area = np.bincount(loop, weights=((x0 - rx) * (y1 - ry) - (x1 - rx) * (y0 - ry)) / 2,
                            minlength=n_loops)
    loop_label = np.zeros(n_loops, np.int64)
    loop_label[loop] = lab
    allc[:] = np.bincount(loop_label, weights=loop_len, minlength=n + 1)[1:n + 1]
    loop_min_vertex = np.full(n_loops, np.iinfo(np.int64).max)
    np.minimum.at(loop_min_vertex, loop, v0)
    abs_area = np.abs(loop_area)
    largest = np.zeros(n + 1)
    np.maximum.at(largest, loop_label, abs_area)
    tied_for_largest = abs_area >= largest[loop_label] - 1e-9 * np.maximum(1.0, largest[loop_label])
    order = np.lexsort((loop_min_vertex, -loop_len, ~tied_for_largest, loop_label))
    first = np.r_[True, loop_label[order][1:] != loop_label[order][:-1]]
    pick = order[first]
    outer[loop_label[pick] - 1] = loop_len[pick]
    return outer, allc


def grain_moments(cells: Cells, labels: np.ndarray, n: int) -> dict:
    """Area and centred second moments of the labels 1..n, summed over their generator pixels.

    Returns arrays of length n (index 0 = label 1): ``area``, centroid ``cx``, ``cy`` and the central
    second moments per unit area ``cxx``, ``cyy``, ``cxy`` (variance-like integrals about the centroid
    divided by the area, pixel units). Labels without cells give area 0 and NaN for the others; a label
    above n raises ValueError. The cell moments (relative to each cell's own pixel centre) are shifted to
    a per-label reference pixel (the label's first generator pixel) by the integer pixel offset before
    they are summed, so no large coordinates enter the sums.
    """
    labels = _check_labels(cells, labels)
    nx = cells.shape[1]
    isgen = cells.generator.ravel()
    if isgen.any() and int(labels[isgen].max()) > n:
        raise ValueError("a label exceeds n")
    gen = np.flatnonzero(isgen & (labels >= 1))
    lab = labels[gen]
    ref = np.zeros((n + 1, 2))
    present = np.unique(lab, return_index=True)
    first_pixel = gen[present[1]]
    ref[present[0], 0] = first_pixel % nx
    ref[present[0], 1] = first_pixel // nx
    # exact parallel-axis step: the cell moments are relative to the cell's own pixel centre, so the shift
    # to the label's reference pixel is the integer offset (ox, oy) between the two pixel centres
    ox, oy = gen % nx - ref[lab, 0], gen // nx - ref[lab, 1]
    a = cells.area[gen]
    m1x, m1y, mxx, myy, mxy = (cells.moments[gen, k] for k in range(5))
    dx, dy = m1x + ox * a, m1y + oy * a
    dxx = mxx + 2 * ox * m1x + ox * ox * a
    dyy = myy + 2 * oy * m1y + oy * oy * a
    dxy = mxy + ox * m1y + oy * m1x + ox * oy * a
    tot = lambda w: np.bincount(lab, weights=w, minlength=n + 1)[1:]  # noqa: E731
    area = tot(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = tot(dx) / area, tot(dy) / area
        return dict(area=area, cx=ref[1:, 0] + mx, cy=ref[1:, 1] + my,
                    cxx=tot(dxx) / area - mx * mx, cyy=tot(dyy) / area - my * my,
                    cxy=tot(dxy) / area - mx * my)


# ---------------------------------------------------------------------------------------------
# pixel hosts
# ---------------------------------------------------------------------------------------------
def _offsets_at(need):
    """All integer (dy, dx) with dy^2 + dx^2 == need."""
    r = int(np.floor(np.sqrt(need)))
    out = []
    for dy in range(-r, r + 1):
        rest = need - dy * dy
        dx = int(round(np.sqrt(rest)))
        if dx * dx == rest:
            out.append((dy, dx))
            if dx:
                out.append((dy, -dx))
    return out


_CROSS_EPS = 1e-9


def _odd_crossings(px, py, seg):
    """Parity of the crossings of the +x ray from (px, py) with the segments seg (4, k); chunked.

    Half-open rule: a segment straddles when (y0 > py) != (y1 > py) and counts when px < x_int. Voronoi
    vertices carry ~1e-12 of rounding noise and many pixel centres lie exactly on a segment end or on
    a segment, so both comparisons are made with a margin ``_CROSS_EPS`` (pixel units): the rule then
    behaves as in exact arithmetic (distinct rational vertex coordinates here differ by far more)."""
    x0, y0, x1, y1 = seg
    inside = np.zeros(len(px), bool)
    step = max(1, 4_000_000 // max(1, x0.size))
    for s in range(0, len(px), step):
        qx, qy = px[s:s + step, None], py[s:s + step, None]
        straddle = (y0 > qy + _CROSS_EPS) != (y1 > qy + _CROSS_EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x0 + (qy - y0) * (x1 - x0) / (y1 - y0)
        inside[s:s + step] = (straddle & (qx < xint - _CROSS_EPS)).sum(axis=1) % 2 == 1
    return inside


def pixel_hosts(cells: Cells, labels: np.ndarray) -> np.ndarray:
    """Label of the cell that holds the centre of every pixel, as a flat ``(ny * nx,)`` array.

    A generator pixel returns its own label. Another pixel takes the label of its unique nearest
    generator (the exterior ring gives 0); when several generators are equally near (exact integer
    squared distances) a crossing test decides: a +x ray from the centre against the boundary segments
    of each candidate label > 0, odd = inside, the smallest inside label wins, none gives 0.

    The tie rule depends on the direction of the ray, by design and as in MTEX: a single
    hole in a 1 x 7 row of pixels can be closed into its grain while the same hole in a 7 x 1 column
    stays open, because the step to the right leaves the map there. That is not a defect.
    """
    labels = _check_labels(cells, labels)
    ny, nx = cells.shape
    gen = cells.generator
    out = np.where(gen.ravel(), labels, 0).astype(np.int64)
    if gen.all():
        return out
    G = np.pad(gen, 1, constant_values=True)
    Lp = np.pad(np.where(gen, labels.reshape(ny, nx), 0), 1, constant_values=0)
    d2 = np.rint(ndi.distance_transform_edt(~G) ** 2).astype(np.int64)
    ty, tx = np.nonzero(~np.pad(gen, 1, constant_values=True))
    need = d2[ty, tx]
    result = np.zeros(len(ty), np.int64)
    tie_pix, tie_lab = [], []
    for value in np.unique(need):
        sel = np.flatnonzero(need == value)
        offs = _offsets_at(int(value))
        cand = np.full((len(sel), len(offs)), -1, np.int64)
        for j, (dy, dx) in enumerate(offs):
            yy, xx = ty[sel] + dy, tx[sel] + dx
            ok = (yy >= 0) & (yy < ny + 2) & (xx >= 0) & (xx < nx + 2)
            ok[ok] = G[yy[ok], xx[ok]]
            cand[ok, j] = Lp[yy[ok], xx[ok]]
        hi = cand.max(axis=1)
        lo = np.where(cand < 0, np.iinfo(np.int64).max, cand).min(axis=1)
        single = lo == hi
        result[sel[single]] = hi[single]
        tie = ~single
        if tie.any():
            rows, cols = np.nonzero(cand[tie] > 0)
            pairs = np.unique(np.stack([sel[tie][rows], cand[tie][rows, cols]], axis=1), axis=0)
            tie_pix.append(pairs[:, 0])
            tie_lab.append(pairs[:, 1])
    if tie_pix:
        tp, tl = np.concatenate(tie_pix), np.concatenate(tie_lab)
        slab, sx0, sy0, sx1, sy1, _, _ = _directed_segments(cells, labels)
        order = np.argsort(slab, kind="stable")
        slab = slab[order]
        seg = np.stack([sx0, sy0, sx1, sy1])[:, order]
        winner = np.full(len(ty), np.iinfo(np.int64).max)
        # Each label's tie pixels are one run of the label-sorted pairs: sort once, slice the runs.
        by_label = np.argsort(tl, kind="stable")
        tp, tl = tp[by_label], tl[by_label]
        first = np.flatnonzero(np.r_[True, tl[1:] != tl[:-1]])
        for a, b in zip(first, np.r_[first[1:], len(tl)]):
            g = tl[a]
            lo_, hi_ = np.searchsorted(slab, g, "left"), np.searchsorted(slab, g, "right")
            pix = tp[a:b]
            inside = _odd_crossings(tx[pix] - 1.0, ty[pix] - 1.0, seg[:, lo_:hi_])
            np.minimum.at(winner, pix[inside], g)
        has = np.zeros(len(ty), bool)
        has[tp] = True
        result[has] = np.where(winner[has] == np.iinfo(np.int64).max, 0, winner[has])
    out[(ty - 1) * nx + (tx - 1)] = result
    return out
