"""analysis/grain_cells.py: exact cells, edges, areas, boundary loops and pixel hosts.

Method source: Bachmann, Hielscher, Schaeben, Ultramicroscopy 111 (2011) 1720-1733 (Voronoi cells of the
measurements); the cell conventions were measured equal to MTEX as a black box, no MTEX code is used.

The probe maps in ``fixtures/analysis_mtex_parity/cells_probes.npz`` hold numbers from MTEX for Python
run as a separate program (see the fixture README); nothing here imports it.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from analysis.grain_cells import Cells, boundary_lengths, build_cells, grain_moments, pixel_hosts
from analysis.grain_reconstruction import covered_mask

FIX = Path(__file__).parent / "fixtures" / "analysis_mtex_parity" / "cells_probes.npz"
EXACT_HOST_MAPS = ("band_w2", "junction_hole3", "rect_8x8", "island", "corner4")


def _edge_dict(cells, keep=None):
    """{(a, b): summed length} for edges between two map cells (optionally only cells in ``keep``)."""
    out = {}
    for (a, b), length in zip(cells.edge_cells.tolist(), cells.edge_length):
        if a < 0 or b < 0:
            continue
        if keep is not None and not (keep[a] and keep[b]):
            continue
        out[(a, b)] = out.get((a, b), 0.0) + length
    return out


# ---------------------------------------------------------------------------------------------
# full grid and invariants
# ---------------------------------------------------------------------------------------------
def test_full_grid_edges_areas_and_moments():
    ny, nx = 4, 6
    cells = build_cells(np.ones((ny, nx), bool))
    assert isinstance(cells, Cells) and cells.shape == (ny, nx)
    inner = {}
    border = 0
    for (a, b), length in zip(cells.edge_cells.tolist(), cells.edge_length):
        if a < 0 or b < 0:
            border += 1
        else:
            assert a < b
            inner[(a, b)] = length
        assert length == pytest.approx(1.0, abs=1e-12)
    expected = {}
    for r in range(ny):
        for c in range(nx):
            if c + 1 < nx:
                expected[(r * nx + c, r * nx + c + 1)] = 1.0
            if r + 1 < ny:
                expected[(r * nx + c, (r + 1) * nx + c)] = 1.0
    assert inner == expected
    assert border == 2 * ny + 2 * nx
    np.testing.assert_allclose(cells.area, 1.0, atol=1e-12)
    # moments are integrals in coordinates relative to the cell's own pixel centre
    np.testing.assert_allclose(cells.moments, np.tile([0, 0, 1 / 12, 1 / 12, 0], (ny * nx, 1)), atol=1e-12)
    assert cells.edge_xy.shape == (len(cells.edge_length), 4)
    assert cells.edge_vertex.shape == (len(cells.edge_length), 2)
    np.testing.assert_allclose(np.hypot(cells.edge_xy[:, 2] - cells.edge_xy[:, 0],
                                        cells.edge_xy[:, 3] - cells.edge_xy[:, 1]), cells.edge_length, atol=1e-12)


def _corner_ids(cells):
    """{corner (x, y) rounded: vertex id}; fails if one corner carries two ids."""
    ids = {}
    for part, col in ((slice(0, 2), 0), (slice(2, 4), 1)):
        for xy, vid in zip(cells.edge_xy[:, part].round(6).tolist(), cells.edge_vertex[:, col].tolist()):
            assert ids.setdefault(tuple(xy), vid) == vid
    return ids


def test_vertex_ids_of_qhull_path_agree_with_fast_path_on_unaffected_corners():
    """One gap in a 13x13 map sends build_cells through Voronoi/Qhull; the pixel squares farther than 2.5 px
    from the gap keep their corners. In both paths every corner carries one id and distinct corners carry
    distinct ids, so the corner <-> id correspondence is the same up to renaming."""
    gap = (6, 6)
    gen = np.ones((13, 13), bool)
    gen[gap] = False
    fast = _corner_ids(build_cells(np.ones((13, 13), bool)))
    qhull = _corner_ids(build_cells(gen))
    far = [xy for xy in fast if max(abs(xy[0] - gap[1]), abs(xy[1] - gap[0])) > 2.5]
    assert len(far) > 100 and all(xy in qhull for xy in far)
    for ids in (fast, qhull):
        assert len({ids[xy] for xy in far}) == len(far)


@pytest.mark.parametrize("seed,shape,p_missing", [(0, (9, 11), 0.15), (1, (12, 7), 0.3), (2, (8, 8), 0.05),
                                                  (3, (5, 9), 0.2), (4, (6, 3), 0.4), (5, (15, 15), 0.5)])
def test_cells_tile_the_map_rectangle(seed, shape, p_missing):
    """No generator is missing in the border rows/columns, so the exterior ring takes no area."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    gen = rng.random(shape) >= p_missing
    gen[0, :] = gen[-1, :] = gen[:, 0] = gen[:, -1] = True
    gen[ny // 2, nx // 2] = False
    cells = build_cells(gen)
    assert cells.area.sum() == pytest.approx(ny * nx, abs=1e-9)
    assert (cells.area[~gen.ravel()] == 0).all()
    assert (cells.area[gen.ravel()] > 0).all()
    x0, x1, y0, y1 = -0.5, nx - 0.5, -0.5, ny - 0.5
    wx, wy = x1 - x0, y1 - y0
    expect = [wy * (x1 ** 2 - x0 ** 2) / 2, wx * (y1 ** 2 - y0 ** 2) / 2, wy * (x1 ** 3 - x0 ** 3) / 3,
              wx * (y1 ** 3 - y0 ** 3) / 3, (x1 ** 2 - x0 ** 2) / 2 * (y1 ** 2 - y0 ** 2) / 2]
    rr, cc = np.divmod(np.arange(ny * nx), nx)
    m, A = cells.moments, cells.area
    absolute = [m[:, 0] + cc * A, m[:, 1] + rr * A, m[:, 2] + 2 * cc * m[:, 0] + cc * cc * A,
                m[:, 3] + 2 * rr * m[:, 1] + rr * rr * A, m[:, 4] + cc * m[:, 1] + rr * m[:, 0] + cc * rr * A]
    np.testing.assert_allclose([v.sum() for v in absolute], expect, atol=1e-9)
    assert (cells.edge_length > 1e-9).all()
    both = cells.edge_cells[(cells.edge_cells >= 0).all(axis=1)]
    assert (both[:, 0] < both[:, 1]).all()
    assert gen.ravel()[cells.edge_cells[cells.edge_cells >= 0]].all()


def test_missing_border_pixel_gives_a_share_to_the_exterior_ring():
    """A gap in the border row is shared by its two map neighbours, the cell below and the ring generator
    above it: the ring cell takes a quarter of the gap square, so the map cells hold 1/4 less."""
    gen = np.ones((5, 5), bool)
    gen[0, 2] = False
    cells = build_cells(gen)
    assert cells.area.sum() == pytest.approx(25 - 0.25, abs=1e-9)
    assert cells.area[0 * 5 + 1] == pytest.approx(1.25, abs=1e-9)
    assert cells.area[1 * 5 + 2] == pytest.approx(1.25, abs=1e-9)


def test_one_missing_pixel_in_5x5():
    gen = np.ones((5, 5), bool)
    gen[2, 2] = False
    cells = build_cells(gen)
    d = _edge_dict(cells)
    pix = lambda r, c: r * 5 + c  # noqa: E731
    key = lambda p, q: (min(p, q), max(p, q))  # noqa: E731
    assert key(pix(1, 2), pix(3, 2)) not in d      # above / below the gap
    assert key(pix(2, 1), pix(2, 3)) not in d      # left / right of the gap
    for p, q in ((pix(1, 2), pix(2, 1)), (pix(1, 2), pix(2, 3)), (pix(3, 2), pix(2, 1)), (pix(3, 2), pix(2, 3))):
        assert d[key(p, q)] == pytest.approx(np.sqrt(2) / 2, abs=1e-9)
    assert cells.area[pix(2, 2)] == 0
    # the gap square is split into four triangles of area 1/4 among the four edge neighbours
    for q in (pix(1, 2), pix(3, 2), pix(2, 1), pix(2, 3)):
        assert cells.area[q] == pytest.approx(1.25, abs=1e-9)


def test_no_generator_gives_empty_cells():
    cells = build_cells(np.zeros((3, 4), bool))
    assert len(cells.edge_length) == 0 and cells.area.sum() == 0


# ---------------------------------------------------------------------------------------------
# boundary loops
# ---------------------------------------------------------------------------------------------
def test_pinch_two_grains_touching_at_a_vertex():
    """A and B form a 2x2 checkerboard inside the field C of a 6x6 map.

    A (and B) are two unit squares that touch in one corner; at the touching vertex each incoming
    segment turns as sharply as possible towards the left, which is into the square it came from
    (turn +90 deg against -90 deg), so the boundary falls into two loops of length 4: outer 4, all 8.
    C: the 24 border pixels of the map form the outer loop (length 24), the 2x2 hole has length 8.
    """
    ny = nx = 6
    labels = np.full((ny, nx), 3, np.int64)
    labels[2, 2] = labels[3, 3] = 1
    labels[2, 3] = labels[3, 2] = 2
    cells = build_cells(np.ones((ny, nx), bool))
    outer, allc = boundary_lengths(cells, labels.ravel(), 3)
    np.testing.assert_allclose(outer, [4.0, 4.0, 24.0], atol=1e-12)
    np.testing.assert_allclose(allc, [8.0, 8.0, 32.0], atol=1e-12)


def test_boundary_lengths_square_with_hole_and_two_grains():
    labels = np.zeros((5, 6), np.int64)
    labels[:, :3] = 1
    labels[:, 3:] = 2
    labels[2, 1] = 0          # not in a grain (hole inside grain 1)
    cells = build_cells(np.ones((5, 6), bool))
    outer, allc = boundary_lengths(cells, labels.ravel(), 2)
    np.testing.assert_allclose(outer, [16.0, 16.0], atol=1e-12)
    np.testing.assert_allclose(allc, [20.0, 16.0], atol=1e-12)


def test_pixel_hosts_rejects_invalid_labels():
    cells = build_cells(np.ones((3, 3), bool))
    bad = np.ones(9, np.int64)
    bad[4] = -1
    with pytest.raises(ValueError):
        pixel_hosts(cells, bad)
    with pytest.raises(ValueError):
        pixel_hosts(cells, np.ones(8, np.int64))
    with pytest.raises(ValueError):
        pixel_hosts(cells, np.ones(9, float))
    # a negative entry on a pixel that is not a generator is never read
    gen = np.ones((3, 3), bool)
    gen[1, 1] = False
    lab = np.ones(9, np.int64)
    lab[4] = -5
    assert pixel_hosts(build_cells(gen), lab)[4] == 1


def test_grain_moments_unit_squares_and_rectangle():
    ny, nx = 3, 4
    cells = build_cells(np.ones((ny, nx), bool))
    gm = grain_moments(cells, np.ones(ny * nx, np.int64), 1)
    assert gm["area"][0] == pytest.approx(12.0, abs=1e-12)
    assert gm["cx"][0] == pytest.approx((nx - 1) / 2, abs=1e-12)
    assert gm["cy"][0] == pytest.approx((ny - 1) / 2, abs=1e-12)
    assert gm["cxx"][0] == pytest.approx(nx ** 2 / 12, abs=1e-12)     # uniform on [-0.5, nx - 0.5]
    assert gm["cyy"][0] == pytest.approx(ny ** 2 / 12, abs=1e-12)
    assert gm["cxy"][0] == pytest.approx(0.0, abs=1e-12)


def test_grain_moments_two_grains():
    ny, nx = 4, 6
    labels = np.zeros((ny, nx), np.int64)
    labels[:, :2] = 1          # 4 x 2 block: columns 0..1
    labels[:2, 2:] = 2         # 2 x 4 block: rows 0..1, columns 2..5
    cells = build_cells(np.ones((ny, nx), bool))
    gm = grain_moments(cells, labels.ravel(), 2)
    np.testing.assert_allclose(gm["area"], [8.0, 8.0], atol=1e-12)
    np.testing.assert_allclose(gm["cx"], [0.5, 3.5], atol=1e-12)
    np.testing.assert_allclose(gm["cy"], [1.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(gm["cxx"], [2 ** 2 / 12, 4 ** 2 / 12], atol=1e-12)
    np.testing.assert_allclose(gm["cyy"], [4 ** 2 / 12, 2 ** 2 / 12], atol=1e-12)
    np.testing.assert_allclose(gm["cxy"], [0.0, 0.0], atol=1e-12)


def test_grain_moments_rejects_label_above_n():
    cells = build_cells(np.ones((2, 2), bool))
    with pytest.raises(ValueError):
        grain_moments(cells, np.array([1, 2, 3, 1]), 2)


def test_grain_moments_far_from_the_origin_equal_those_near_it():
    """A 3x3 grain at x ~ 1190 gives the same central moments as at x ~ 5 (exact parallel-axis shift)."""
    def moments_of(col0, nx):
        gen = np.ones((7, nx), bool)
        gen[0, 3] = False                        # a gap sends build_cells through Voronoi/Qhull
        labels = np.zeros((7, nx), np.int64)
        labels[2:5, col0:col0 + 3] = 1
        return grain_moments(build_cells(gen), labels.ravel(), 1)

    near, far = moments_of(4, 20), moments_of(1189, 1195)
    for key in ("cxx", "cyy", "cxy"):
        assert far[key][0] == pytest.approx(near[key][0], rel=1e-12, abs=1e-12)
    assert far["cxx"][0] == pytest.approx(0.75, rel=1e-12)
    assert far["cyy"][0] == pytest.approx(0.75, rel=1e-12)
    assert far["cx"][0] == pytest.approx(1190.0, abs=1e-9)


def test_grain_moments_label_without_cells_is_nan():
    cells = build_cells(np.ones((2, 2), bool))
    gm = grain_moments(cells, np.ones(4, np.int64), 2)
    assert gm["area"][1] == 0 and np.isnan(gm["cx"][1]) and np.isnan(gm["cxy"][1])


def test_boundary_lengths_rejects_negative_and_too_large_labels():
    cells = build_cells(np.ones((3, 3), bool))
    bad = np.ones(9, np.int64)
    bad[4] = -1
    with pytest.raises(ValueError):
        boundary_lengths(cells, bad, 1)
    with pytest.raises(ValueError):
        boundary_lengths(cells, np.full(9, 2, np.int64), 1)


def test_outer_loop_of_equal_area_is_the_longer_one():
    """Label 1 = a 2x2 block and a 1x4 strip that touch in one corner: two loops of area 4, lengths 8
    and 10. Equal absolute areas are decided by the longer loop, so outer = 10."""
    labels = np.full((6, 6), 2, np.int64)
    labels[0:2, 0:2] = 1
    labels[2, 2:6] = 1
    cells = build_cells(np.ones((6, 6), bool))
    outer, allc = boundary_lengths(cells, labels.ravel(), 2)
    assert outer[0] == pytest.approx(10.0, abs=1e-12)
    assert allc[0] == pytest.approx(18.0, abs=1e-12)


def test_ridges_shorter_than_the_edge_threshold_still_count_for_area():
    """The edge list is filtered, areas are not: the cells tile the rectangle without gaps."""
    rng = np.random.default_rng(7)
    gen = rng.random((20, 20)) >= 0.3
    gen[0, :] = gen[-1, :] = gen[:, 0] = gen[:, -1] = True
    assert build_cells(gen).area.sum() == pytest.approx(400.0, abs=1e-9)


def test_boundary_lengths_vanishing_label_gives_zero():
    labels = np.zeros((3, 3), np.int64)
    labels[:] = 1
    cells = build_cells(np.ones((3, 3), bool))
    outer, allc = boundary_lengths(cells, labels.ravel(), 2)
    np.testing.assert_allclose(outer, [12.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(allc, [12.0, 0.0], atol=1e-12)


def test_pixel_hosts_generators_keep_their_label_and_ring_gives_zero():
    gen = np.ones((5, 5), bool)
    gen[2, 2] = False
    labels = np.zeros((5, 5), np.int64)
    labels[:, :2] = 1
    labels[:, 3:] = 2
    labels[:, 2] = 3
    labels[2, 2] = 0
    cells = build_cells(gen)
    host = pixel_hosts(cells, labels.ravel())
    expect = labels.ravel().copy()
    # The missing centre (x = 2, y = 2) is the common corner of four cells: the gap square is cut into
    # four triangles by its diagonals, given to the neighbours above (grain 3), below (grain 3), left
    # (grain 1) and right (grain 2). All four are equally near, so the +x ray from (2, 2) decides, with
    # the half-open rule (y0 > py) != (y1 > py) and px < x_int:
    #   grain 3: the only segments that straddle y = 2 are (2, 2)-(2.5, 2.5) and (2, 2)-(1.5, 2.5); both
    #            meet y = 2 at x = 2, and px < 2 is false -> 0 crossings, outside;
    #   grain 1: its boundary lies at x <= 2 (the left triangle ends in (2, 2)); same two touching
    #            segments, none counts -> 0 crossings, outside;
    #   grain 2: the right border of the map, x = 4.5, straddles y = 2 and lies right of the point -> 1
    #            crossing; the two segments that end in (2, 2) do not count -> odd, inside.
    # Host = 2.
    assert host[2 * 5 + 2] == 2
    mask = np.ones(25, bool)
    mask[2 * 5 + 2] = False
    np.testing.assert_array_equal(host[mask], expect[mask])


def test_pixel_hosts_all_nearest_generators_share_a_label():
    """Both closed pixels have three equally near generators, all of one grain: no ray test needed."""
    gen = np.ones((5, 8), bool)
    gen[2, 3:5] = False
    labels = np.zeros((5, 8), np.int64)
    labels[:, :4] = 1
    labels[:, 4:] = 2
    cells = build_cells(gen)
    host = pixel_hosts(cells, labels.ravel()).reshape(5, 8)
    assert host[2, 3] == 1 and host[2, 4] == 2


def test_pixel_hosts_tie_decided_by_ray_crossing():
    """Pixel (2, 3) of a 1-wide scratch lies on the edge x = 3 between the cells of grain 1 (left) and
    grain 2 (right): it is a point of the ridge x = 3, y = 1.5..2.5, between the cells of (2, 2) and (2, 4)
    (the whole grain-1/grain-2 border along x = 3 runs from y = 1 to y = 3 in three pieces; above and below
    the open pixels (0, 3) and (4, 3) own the cells). The +x ray from (3, 2):
      grain 1: every boundary segment lies at x <= 3; the only one that straddles y = 2 is the edge
               x = 3 itself, and px < 3 is false -> 0 crossings, outside;
      grain 2: that same edge does not count either, but the right border x = 6.5 straddles y = 2 -> 1
               crossing, inside.
    Host = 2."""
    gen = np.ones((5, 7), bool)
    gen[1:4, 3] = False
    labels = np.zeros((5, 7), np.int64)
    labels[:, :3] = 1
    labels[:, 4:] = 2
    cells = build_cells(gen)
    host = pixel_hosts(cells, labels.ravel()).reshape(5, 7)
    assert host[2, 3] == 2


def _exact_hosts(cells, labels, gen):
    """Hosts of the non-generator pixels in exact rational arithmetic (brute force, no shared code)."""
    from fractions import Fraction

    ny, nx = gen.shape
    lab2 = labels.reshape(ny, nx)
    gens = [(c, r, int(lab2[r, c])) for r in range(ny) for c in range(nx) if gen[r, c]]
    gens += [(c, -1, 0) for c in range(-1, nx + 1)] + [(c, ny, 0) for c in range(-1, nx + 1)]
    gens += [(-1, r, 0) for r in range(ny)] + [(nx, r, 0) for r in range(ny)]
    frac = lambda v: Fraction(float(v)).limit_denominator(10 ** 7)  # noqa: E731
    segs = {}
    for (a, b), xy in zip(cells.edge_cells.tolist(), cells.edge_xy):
        la = int(labels[a]) if a >= 0 else 0
        lb = int(labels[b]) if b >= 0 else 0
        if la == lb:
            continue
        seg = tuple(frac(v) for v in xy)
        for g in (la, lb):
            if g > 0:
                segs.setdefault(g, []).append(seg)
    out, ties = {}, 0
    for r in range(ny):
        for c in range(nx):
            if gen[r, c]:
                continue
            d2 = [(gx - c) ** 2 + (gy - r) ** 2 for gx, gy, _ in gens]
            m = min(d2)
            cand = {g for d, (_, _, g) in zip(d2, gens) if d == m}
            if len(cand) == 1:
                out[(r, c)] = cand.pop()
                continue
            ties += 1
            inside = []
            for g in sorted(k for k in cand if k > 0):
                n_cross = 0
                for x0, y0, x1, y1 in segs.get(g, []):
                    if (y0 > r) != (y1 > r) and c < x0 + (r - y0) * (x1 - x0) / (y1 - y0):
                        n_cross += 1
                if n_cross % 2 == 1:
                    inside.append(g)
            out[(r, c)] = inside[0] if inside else 0
    return out, ties


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pixel_hosts_ties_equal_exact_rational_reference(seed):
    """Qhull vertices carry ~1e-12 noise; many tie pixels lie exactly on a boundary, so the crossing test
    must behave as it would in exact arithmetic."""
    rng = np.random.default_rng(seed)
    ny = nx = 30
    gen = rng.random((ny, nx)) >= 0.2
    rr, cc = np.divmod(np.arange(ny * nx), nx)
    labels = np.where(gen.ravel(), ((rr // 6) + (cc // 9)) % 3 + 1, 0).astype(np.int64)
    cells = build_cells(gen)
    host = pixel_hosts(cells, labels).reshape(ny, nx)
    expect, ties = _exact_hosts(cells, labels, gen)
    assert ties >= 10
    bad = [(k, int(host[k]), v) for k, v in expect.items() if host[k] != v]
    assert not bad, bad[:5]


# ---------------------------------------------------------------------------------------------
# MTEX probe maps
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def probes():
    F = np.load(FIX)
    names = sorted({k.split("__")[0] for k in F.files if "__" in k})
    out = {}
    for name in names:
        ni = F[name + "__notindexed"].astype(bool)
        alpha = float(F[name + "__alpha"])
        ind = ~ni
        gen = ~(covered_mask(ind, alpha) & ni)
        cells = build_cells(gen)
        ny, nx = ni.shape
        m = ind.ravel()
        # one grain per connected part of the measurements (all orientations equal)
        ec = cells.edge_cells
        ok = (ec >= 0).all(axis=1)
        ok &= m[np.maximum(ec[:, 0], 0)] & m[np.maximum(ec[:, 1], 0)]
        g = coo_matrix((np.ones(ok.sum()), (ec[ok, 0], ec[ok, 1])), shape=(ny * nx, ny * nx))
        _, comp = connected_components(g, directed=False)
        lab_flat = np.zeros(ny * nx, np.int64)
        # number grains 1..n in order of first appearance among measured pixels
        _, first = np.unique(comp[m], return_inverse=True)
        lab_flat[m] = first + 1
        n = int(lab_flat.max())
        M = F[name + "__grain_ids"].astype(np.int64).ravel()
        lut = np.zeros(n + 1, np.int64)
        lut[lab_flat[m]] = M[m]
        gl = [int(v) for v in F[name + "__grain_list"]]
        rows = np.array([gl.index(int(lut[k])) for k in range(1, n + 1)])
        out[name] = dict(F=F, ni=ni, gen=gen, cells=cells, lab=lab_flat, n=n, M=M, rows=rows, nx=nx, ny=ny)
    return out


def test_probe_pairs_and_lengths_equal_mtex(probes):
    for name, p in probes.items():
        F = p["F"]
        m = ~p["ni"].ravel()
        mine = _edge_dict(p["cells"], keep=m)
        ref = {tuple(r): l for r, l in zip(F[name + "__pairs"].tolist(), F[name + "__pair_length"])}
        assert set(mine) == set(ref), name
        worst = max(abs(mine[k] - ref[k]) for k in ref)
        assert worst < 1e-9, (name, worst)


def test_probe_area_perimeters_and_aspect_equal_mtex(probes):
    for name, p in probes.items():
        F, cells, lab, n, rows = p["F"], p["cells"], p["lab"], p["n"], p["rows"]
        assert n == len(F[name + "__grain_list"]), name
        area = np.bincount(lab, weights=cells.area, minlength=n + 1)[1:]
        ref = F[name + "__area"][rows]
        assert np.max(np.abs(area - ref) / ref) < 1e-9, name
        outer, allc = boundary_lengths(cells, lab, n)
        assert np.max(np.abs(outer - F[name + "__perimeter"][rows])) < 1e-9, name
        assert np.max(np.abs(allc - F[name + "__perimeter_with_inclusions"][rows])) < 1e-9, name
        gm = grain_moments(cells, lab, n)
        np.testing.assert_allclose(gm["area"], area, rtol=1e-12)
        half_tr = (gm["cxx"] + gm["cyy"]) / 2
        root = np.hypot((gm["cxx"] - gm["cyy"]) / 2, gm["cxy"])      # stable near isotropy
        aspect = np.sqrt((half_tr + root) / (half_tr - root))
        refa = F[name + "__aspect"][rows]
        assert np.max(np.abs(aspect - refa) / refa) < 1e-9, name


def test_probe_pixel_hosts_equal_mtex_fill(probes):
    agree = total = 0
    for name, p in probes.items():
        F, ni = p["F"], p["ni"]
        labels = np.where(~ni.ravel(), p["M"], 0)
        host = pixel_hosts(p["cells"], labels).reshape(ni.shape)
        ref = F[name + "__fill_grain"].astype(np.int64)
        same = host[ni] == ref[ni]
        agree += int(same.sum())
        total += int(ni.sum())
        if name in EXACT_HOST_MAPS:
            assert same.all(), (name, int((~same).sum()))
        gen = p["gen"]
        np.testing.assert_array_equal(host[gen], labels.reshape(ni.shape)[gen])
    assert agree / total >= 0.99, (agree, total)


# ---------------------------------------------------------------------------------------------
# pixel_hosts: the tie pass resolves every label from one label-sorted run
# ---------------------------------------------------------------------------------------------
def _pixel_hosts_per_label_scan(cells, labels):
    """The first algorithm of ``pixel_hosts``, kept as a reference: its tie loop scans every tie pair once
    per label (O(labels x tie pairs)). The result of the current implementation must equal it bit for bit."""
    from analysis import grain_cells as gc

    labels = gc._check_labels(cells, labels)
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
        offs = gc._offsets_at(int(value))
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
        slab, sx0, sy0, sx1, sy1, _, _ = gc._directed_segments(cells, labels)
        order = np.argsort(slab, kind="stable")
        slab = slab[order]
        seg = np.stack([sx0, sy0, sx1, sy1])[:, order]
        winner = np.full(len(ty), np.iinfo(np.int64).max)
        for g in np.unique(tl):
            lo_, hi_ = np.searchsorted(slab, g, "left"), np.searchsorted(slab, g, "right")
            pix = tp[tl == g]
            inside = gc._odd_crossings(tx[pix] - 1.0, ty[pix] - 1.0, seg[:, lo_:hi_])
            np.minimum.at(winner, pix[inside], g)
        has = np.zeros(len(ty), bool)
        has[tp] = True
        result[has] = np.where(winner[has] == np.iinfo(np.int64).max, 0, winner[has])
    out[(ty - 1) * nx + (tx - 1)] = result
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pixel_hosts_equal_the_per_label_scan_on_the_exact_reference_maps(seed):
    rng = np.random.default_rng(seed)
    ny = nx = 30
    gen = rng.random((ny, nx)) >= 0.2
    rr, cc = np.divmod(np.arange(ny * nx), nx)
    labels = np.where(gen.ravel(), ((rr // 6) + (cc // 9)) % 3 + 1, 0).astype(np.int64)
    cells = build_cells(gen)
    np.testing.assert_array_equal(pixel_hosts(cells, labels), _pixel_hosts_per_label_scan(cells, labels))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pixel_hosts_equal_the_per_label_scan_with_many_small_labels(seed):
    """4 x 4 patches, each its own label: many labels, each with few tie pairs."""
    rng = np.random.default_rng(seed)
    ny, nx = 40, 48
    gen = rng.random((ny, nx)) >= 0.25
    rr, cc = np.divmod(np.arange(ny * nx), nx)
    patch = (rr // 4) * (nx // 4) + cc // 4 + 1
    labels = np.where(gen.ravel(), patch, 0).astype(np.int64)
    cells = build_cells(gen)
    host = pixel_hosts(cells, labels)
    np.testing.assert_array_equal(host, _pixel_hosts_per_label_scan(cells, labels))
    assert len(np.unique(host)) > 50


def test_pixel_hosts_equal_the_per_label_scan_on_the_mtex_probe_maps(probes):
    for name, p in probes.items():
        labels = np.where(~p["ni"].ravel(), p["M"], 0)
        np.testing.assert_array_equal(pixel_hosts(p["cells"], labels),
                                      _pixel_hosts_per_label_scan(p["cells"], labels), err_msg=name)
        # the same cells, one label per connected part of the measurements
        lab = np.where(p["gen"].ravel(), p["lab"], 0)
        np.testing.assert_array_equal(pixel_hosts(p["cells"], lab),
                                      _pixel_hosts_per_label_scan(p["cells"], lab), err_msg=name)


def test_pixel_hosts_equal_the_per_label_scan_on_the_fill_fixture():
    """The map of the fill tests: grains, holes, a scratch, a wide square, a border strip and outliers."""
    from tests.synthetic_grain_map import add_defects, make_map

    m = add_defects(make_map())
    measured = m["phase_id"] >= 0
    gen = ~(covered_mask(measured, 3.1) & ~measured)
    labels = np.where(gen.ravel(), m["truth"].ravel(), 0).astype(np.int64)
    cells = build_cells(gen)
    host = pixel_hosts(cells, labels)
    np.testing.assert_array_equal(host, _pixel_hosts_per_label_scan(cells, labels))
    assert (host[~gen.ravel()] > 0).any()
