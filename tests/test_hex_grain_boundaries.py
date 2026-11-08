"""Grain boundaries of a hexagonal scan: the angle across every neighbouring pair, drawn along the shared hexagon edge."""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import grain_boundaries as gb  # noqa: E402
from backend.api.services import hex_raster  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402
from tests import hex_raster_oracle as oracle  # noqa: E402
from tests.test_phase_map_layer_endpoint import _make_fake_result  # noqa: E402

N_ROWS, N_COLS, STEP = 5, 5, 2.0
SIGNS = (0.5, -0.5)
WHITE_FROM_15 = [{"id": "hagb", "min": 15.0, "max": None, "color": "#ffffff", "width": 2, "on": True}]


def _qz(deg):
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def _classes():
    """Columns 0-1 are one orientation, columns 2-4 another one, 30 degrees about z."""
    return np.tile((np.arange(N_COLS) >= 2).astype(int), N_ROWS)


def _xmap(classes, phases=None):
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    x, y = np.meshgrid(np.arange(N_COLS, dtype=float), np.arange(N_ROWS, dtype=float))
    quats = np.stack([_qz(30.0 * k) for k in classes])
    return CrystalMap(rotations=Rotation(quats), phase_id=np.zeros(N_ROWS * N_COLS, int) if phases is None else phases,
                      x=x.ravel(), y=y.ravel(),
                      phase_list=PhaseList([Phase(name="a", point_group="m-3m"), Phase(name="b", point_group="m-3m")]))


def _lattice(shift):
    g = ScanGeometry.hex(N_ROWS, N_COLS, STEP, odd_row_shift=shift)
    x, y = g.positions()
    return g, np.stack([x.ravel(), y.ravel()], axis=1)


def _oracle_pairs(pts):
    """Every pair of points one step apart, from the positions alone."""
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    a, b = np.nonzero(np.triu((dist > 1e-9) & (dist <= STEP * (1 + 1e-6))))
    return a, b


def _angles(shift):
    topology = Topology("hex", N_ROWS, N_COLS, shift)
    return gb.boundary_angles(_xmap(_classes()), N_ROWS, N_COLS, topology=topology)


@pytest.mark.parametrize("shift", SIGNS)
def test_the_pairs_are_the_neighbouring_pairs_of_the_lattice_and_carry_the_misorientation(shift):
    _, pts = _lattice(shift)
    got = _angles(shift)
    a, b = got["pairs"]
    want_a, want_b = _oracle_pairs(pts)
    assert sorted(zip(a.tolist(), b.tolist())) == sorted(zip(want_a.tolist(), want_b.tolist()))
    classes = _classes()
    for pa, pb, angle in zip(a, b, got["angle"]):
        assert angle == pytest.approx(30.0 if classes[pa] != classes[pb] else 0.0, abs=1e-3)


def test_two_phases_are_no_boundary_angle_on_a_hexagonal_scan_either():
    phases = np.tile((np.arange(N_COLS) >= 2).astype(int), N_ROWS)
    got = gb.boundary_angles(_xmap(_classes(), phases), N_ROWS, N_COLS, topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    a, b = got["pairs"]
    across = phases[a] != phases[b]
    assert across.any() and np.isnan(got["angle"][across]).all() and np.isfinite(got["angle"][~across]).all()


def test_a_pair_with_an_unindexed_point_has_no_angle():
    pids = np.zeros(N_ROWS * N_COLS, dtype=int)
    pids[7] = -1
    got = gb.boundary_angles(_xmap(_classes(), pids), N_ROWS, N_COLS, topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    a, b = got["pairs"]
    touching = (a == 7) | (b == 7)
    assert touching.sum() == 6 and np.isnan(got["angle"][touching]).all() and np.isfinite(got["angle"][~touching]).all()


def test_the_square_angles_are_what_they_were():
    got = gb.boundary_angles(_xmap(_classes()), N_ROWS, N_COLS)
    assert set(got) == {"h", "v"} and got["h"].shape == (N_ROWS, N_COLS - 1)


@pytest.mark.parametrize("shift", SIGNS)
def test_the_cache_keeps_the_two_topologies_apart(shift):
    xm = _xmap(_classes())
    square = gb.cached_boundary_angles(xm, N_ROWS, N_COLS)
    hexagonal = gb.cached_boundary_angles(xm, N_ROWS, N_COLS, topology=Topology("hex", N_ROWS, N_COLS, shift))
    assert "h" in square and "pairs" in hexagonal


# ------------------------------------------------------------------- the drawing

def _drawn(shift, bands=None, width=None):
    g, pts = _lattice(shift)
    spec = hex_raster.raster_spec(g)
    angles = _angles(shift)
    bands = WHITE_FROM_15 if bands is None else bands
    if width is not None:
        bands = [{**bands[0], "width": width}]
    return g, pts, spec, gb.render_boundaries_hex(angles, g, spec, bands)


def _pixel(spec, x, y):
    return int(np.floor((y - spec.y0) / spec.step_um)), int(np.floor((x - spec.x0) / spec.step_um))


def _white_near(out, spec, point):
    """Is a white pixel within one pixel of the one that holds ``point``? (The midpoint of a slanted edge can fall
    exactly on the line between two pixels, so the pixel that holds it is not the one that is drawn.)"""
    i, j = _pixel(spec, *point)
    window = out[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].reshape(-1, 4)
    return bool((window == (255, 255, 255, 255)).all(axis=1).any())


@pytest.mark.parametrize("shift", SIGNS)
def test_the_overlay_has_the_shape_of_the_display_raster(shift):
    _, _, spec, out = _drawn(shift)
    assert out.shape == (spec.height, spec.width, 4) and out.dtype == np.uint8


@pytest.mark.parametrize("shift", SIGNS)
def test_an_edge_between_two_grains_is_drawn_where_the_hexagons_meet(shift):
    g, pts, spec, out = _drawn(shift)
    classes = _classes()
    a, b = _oracle_pairs(pts)
    n_between = 0
    for pa, pb in zip(a, b):
        if classes[pa] == classes[pb]:
            continue
        n_between += 1
        assert _white_near(out, spec, (pts[pa] + pts[pb]) / 2), (pa, pb)
    assert n_between > 0


def test_a_scan_of_one_orientation_draws_no_boundary():
    g, _ = _lattice(0.5)
    spec = hex_raster.raster_spec(g)
    angles = gb.boundary_angles(_xmap(np.zeros(N_ROWS * N_COLS, dtype=int)), N_ROWS, N_COLS,
                                topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    assert not gb.render_boundaries_hex(angles, g, spec, WHITE_FROM_15).any()


@pytest.mark.parametrize("shift", SIGNS)
def test_the_drawing_is_a_thick_line_along_each_boundary_edge_and_nothing_else(shift):
    g, pts, spec, out = _drawn(shift, width=2)
    classes = _classes()
    a, b = _oracle_pairs(pts)
    keep = classes[a] != classes[b]
    rho = STEP / np.sqrt(3.0)                                   # the length of an edge of a hexagon
    mid = (pts[a[keep]] + pts[b[keep]]) / 2
    tangent = np.stack([-(pts[b[keep]] - pts[a[keep]])[:, 1], (pts[b[keep]] - pts[a[keep]])[:, 0]], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    ii, jj = np.meshgrid(np.arange(spec.height), np.arange(spec.width), indexing="ij")
    centres = np.stack([spec.x0 + (jj + 0.5) * spec.step_um, spec.y0 + (ii + 0.5) * spec.step_um], axis=-1)
    rel = centres[:, :, None, :] - mid[None, None, :, :]
    along = np.clip((rel * tangent[None, None]).sum(-1), -rho / 2, rho / 2)
    dist = np.linalg.norm(rel - along[..., None] * tangent[None, None], axis=-1).min(axis=-1) / spec.step_um   # pixels
    expected = dist <= 1.5                                      # width 2 on this raster (factor 4) is 3 pixels
    drawn = out[..., 3] > 0
    assert expected.any()
    assert (drawn & expected).sum() / (drawn | expected).sum() >= 0.6       # the same set, to a pixel of rounding
    assert dist[drawn].max() <= 2.0                              # and nothing drawn more than a pixel from the line


@pytest.mark.parametrize("shift", SIGNS)
def test_a_wider_band_draws_more_pixels(shift):
    thin = int((_drawn(shift, width=2)[3][..., 3] > 0).sum())
    thick = int((_drawn(shift, width=4)[3][..., 3] > 0).sum())
    assert 0 < thin < thick


def test_a_band_that_does_not_contain_the_angle_or_is_off_draws_nothing():
    for bands in ([{"min": 0.5, "max": 5.0, "color": "#ffffff", "width": 2, "on": True}],
                  [{"min": 15.0, "max": None, "color": "#ffffff", "width": 2, "on": False}]):
        assert not _drawn(0.5, bands=bands)[3].any()


def test_an_angle_on_the_cut_belongs_to_the_band_above_it():
    g, _ = _lattice(0.5)
    spec = hex_raster.raster_spec(g)
    a, b = Topology("hex", N_ROWS, N_COLS, 0.5).pairs()
    angles = {"pairs": (a, b), "angle": np.full(a.shape, 15.0, dtype=np.float32)}
    bands = [{"min": 15.0, "max": None, "color": "#00ff00", "width": 2, "on": True},
             {"min": 5.0, "max": 15.0, "color": "#ff0000", "width": 2, "on": True}]     # drawn last: must not take 15.0
    out = gb.render_boundaries_hex(angles, g, spec, bands)
    drawn = out[out[..., 3] > 0]
    assert drawn.size and (drawn[:, 1] == 255).all() and (drawn[:, 0] == 0).all()


def test_a_later_band_paints_over_an_earlier_one():
    bands = [{"min": 15.0, "max": None, "color": "#ff0000", "width": 2, "on": True},
             {"min": 15.0, "max": None, "color": "#00ff00", "width": 2, "on": True}]
    out = _drawn(0.5, bands=bands)[3]
    drawn = out[out[..., 3] > 0]
    assert (drawn[:, 1] == 255).all() and (drawn[:, 0] == 0).all()


# ----------------------------------------- where a line lies, how long and how thick it is, and the scan's outline
#
# Every expectation below comes from the lattice (tests/hex_raster_oracle.py): the middle of an edge is the middle
# between the two points, its direction is perpendicular to the line between them, its length is the side of a
# hexagon, STEP / sqrt(3), and a pixel is where the oracle's raster puts it.

LAT_ROWS = LAT_COLS = 7
SIDE = STEP / np.sqrt(3.0)
BAND_20_DEG = {"min": 15.0, "max": None, "color": "#ffffff", "on": True}


class _Lattice:
    """A 7 x 7 scan of step STEP, its oracle raster of one factor, and one-pair renderings of it."""

    def __init__(self, shift, factor):
        self.geometry = ScanGeometry.hex(LAT_ROWS, LAT_COLS, STEP, odd_row_shift=shift)
        og = oracle.geometry_dict(LAT_ROWS, LAT_COLS, STEP, shift)
        px, py = oracle.positions(og)
        self.points = np.stack([px.ravel(), py.ravel()], axis=1)
        sp = oracle.spec(og, factor)
        self.spec = hex_raster.raster_spec(self.geometry, factor=factor)
        assert (self.spec.height, self.spec.width) == (sp["height"], sp["width"])
        assert np.allclose([self.spec.x0, self.spec.y0, self.spec.step_um], [sp["x0"], sp["y0"], sp["step_um"]])
        self.d = sp["step_um"]
        ii, jj = np.meshgrid(np.arange(sp["height"]), np.arange(sp["width"]), indexing="ij")
        self.cx, self.cy = sp["x0"] + (jj + 0.5) * self.d, sp["y0"] + (ii + 0.5) * self.d    # pixel centres, um
        self.inside = oracle.brute_cell_index(og, sp) >= 0
        a, b = _oracle_pairs(self.points)
        self.pairs = list(zip(a.tolist(), b.tolist()))

    def interior(self, k):
        """Points at least two rows and columns from the border: a thick line there is not cut by the outline."""
        r, c = divmod(k, LAT_COLS)
        return 2 <= r <= LAT_ROWS - 3 and 2 <= c <= LAT_COLS - 3

    def interior_pairs(self):
        return [(a, b) for a, b in self.pairs if self.interior(a) and self.interior(b)]

    def kind(self, a, b):
        if abs(self.points[b][1] - self.points[a][1]) < 1e-9:
            return "same row"
        return "down right" if self.points[b][0] > self.points[a][0] else "down left"

    def draw(self, pairs, width=2):
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        angles = {"pairs": (a, b), "angle": np.full(len(pairs), 20.0, dtype=np.float32)}
        out = gb.render_boundaries_hex(angles, self.geometry, self.spec, [{**BAND_20_DEG, "width": width}])
        return out[..., 3] > 0

    def frame(self, a, b):
        """Unit vector from a to b, the one along the edge, and the middle of the edge."""
        n = (self.points[b] - self.points[a]) / STEP
        return n, np.array([-n[1], n[0]]), (self.points[a] + self.points[b]) / 2

    def offsets(self, drawn, a, b):
        """(across, along) of the pixel centres of ``drawn`` against the edge, in pixels."""
        n, t, mid = self.frame(a, b)
        rx, ry = self.cx[drawn] - mid[0], self.cy[drawn] - mid[1]
        return (rx * n[0] + ry * n[1]) / self.d, (rx * t[0] + ry * t[1]) / self.d


@pytest.mark.parametrize("factor", (4, 2, 1))
@pytest.mark.parametrize("shift", SIGNS)
def test_a_line_lies_on_the_edge_not_beside_it_in_every_direction(shift, factor):
    # The middle of the painted pixels, across the edge. PIL cuts the end points of a line to whole pixels, and a
    # plain cut left the line up to 1.5 pixels towards the upper left (one direction of three: -1.08 on average).
    lat = _Lattice(shift, factor)
    across = {}
    for a, b in lat.interior_pairs():
        s, _ = lat.offsets(lat.draw([(a, b)]), a, b)
        across.setdefault(lat.kind(a, b), []).append(s.mean())
    assert set(across) == {"same row", "down right", "down left"}
    for kind, values in across.items():
        assert abs(np.mean(values)) <= 0.6, (kind, np.mean(values))     # a whole pixel can be no more than half off
        assert np.abs(values).max() <= 1.0, (kind, np.abs(values).max())


def test_a_line_is_as_long_as_the_edge():
    # The raster is too coarse to see 15 % of an edge (2.3 pixels long at factor 4), so this one is measured on a
    # raster of factor 64: the painted pixels whose centres lie on the axis of the line reach from end to end of
    # the edge, less the rounding of the end points. Half an edge or an edge 15 % too long both fall outside.
    factor = 64
    for shift in SIGNS:
        lat = _Lattice(shift, factor)
        want = SIDE / lat.d
        for a, b in lat.interior_pairs():
            s, along = lat.offsets(lat.draw([(a, b)], width=1), a, b)
            on_axis = np.abs(s) <= 0.5
            reach = along[on_axis].max() - along[on_axis].min()
            assert abs(reach + 0.5 - want) <= 1.0, (shift, lat.kind(a, b), reach, want)


def _triples(lat):
    pairs = set(lat.pairs)
    return [(a, b, c) for a, b in lat.pairs for c in range(len(lat.points)) if c > b
            and (a, c) in pairs and (b, c) in pairs and all(lat.interior(k) for k in (a, b, c))]


@pytest.mark.parametrize("factor", (4, 2, 1))
@pytest.mark.parametrize("shift", SIGNS)
def test_three_boundaries_meet_at_the_point_where_three_hexagons_do_without_a_gap(shift, factor):
    from scipy import ndimage
    lat = _Lattice(shift, factor)
    triples = _triples(lat)
    assert len(triples) >= 8
    for a, b, c in triples:
        vertex = (lat.points[a] + lat.points[b] + lat.points[c]) / 3.0
        edges = [(a, b), (a, c), (b, c)]
        masks = [lat.draw([e]) for e in edges]
        for mask, (p, q) in zip(masks, edges):
            near = np.hypot(lat.cx[mask] - vertex[0], lat.cy[mask] - vertex[1]).min() / lat.d
            assert near <= 1.5, ((a, b, c), (p, q), near)               # each line runs up to the point
        _, count = ndimage.label(np.logical_or.reduce(masks), structure=np.ones((3, 3)))
        assert count == 1, ((a, b, c), count)                           # and the three are one piece


@pytest.mark.parametrize("band_width,thickness", [
    (1, {4: 2, 2: 1, 1: 1}), (2, {4: 3, 2: 2, 1: 1}), (3, {4: 4, 2: 2, 1: 1}),
    (4, {4: 5, 2: 3, 1: 1}), (8, {4: 9, 2: 5, 1: 2}),
])
def test_a_band_is_as_thick_as_the_same_band_on_the_square_overlay(band_width, thickness):
    # square: width + 1 overlay pixels of a quarter step, i.e. (width + 1) / 4 scan steps; on a raster of
    # factor f a step is f pixels, so (width + 1) * f / 4 pixels, a half rounded up, at least one
    for factor, want in thickness.items():
        lat = _Lattice(0.5, factor)
        a, b = next((a, b) for a, b in lat.interior_pairs() if lat.kind(a, b) == "same row")
        mask = lat.draw([(a, b)], width=band_width)
        i, j = np.nonzero(mask)
        row = np.bincount(i)[i.min() + (i.max() - i.min()) // 2]      # pixels across the vertical edge, middle row
        assert row == want, (band_width, factor, row)


def test_the_square_overlay_thickness_is_what_the_hexagonal_one_is_held_to():
    got = gb.render_boundaries({"h": np.full((6, 6), 20.0, dtype=np.float32), "v": np.full((5, 7), np.nan)}, 6, 7,
                               [{**BAND_20_DEG, "width": 2}])
    edge = 3 * gb.SS                                                  # the edge between columns 2 and 3
    assert (got[2 * gb.SS + 1, edge - 2:edge + 3, 3] > 0).sum() == 3  # width 2 is 3 pixels of a quarter step


@pytest.mark.parametrize("shift", SIGNS)
def test_nothing_is_drawn_outside_the_hexagons_of_the_scan(shift):
    # the boundary between two grains runs out to the outline of the scan, and a thick line there reaches over it
    lat = _Lattice(shift, 4)
    for width in (2, 8):
        every_pair = lat.draw(lat.pairs, width=width)
        assert every_pair.any()
        assert not (every_pair & ~lat.inside).any(), (width, int((every_pair & ~lat.inside).sum()))
        # and it is the oracle's outline, not the whole raster, that is left: pixels of the scan are painted
        assert (every_pair & lat.inside).sum() > 0.2 * lat.inside.sum()


@pytest.mark.parametrize("shift", SIGNS)
def test_no_interior_edge_vanishes_at_factor_1_with_a_step_that_is_not_exact_in_binary(shift):
    # 21 x 23 points of step 1.7 um at factor 1: a pixel on the border between two cells used to belong to
    # none, the mask of the overlay cleared it, and some of the edges between points lost every pixel.
    rows, cols, step = 21, 23, 1.7
    d = oracle.geometry_dict(rows, cols, step, shift)
    g = ScanGeometry.from_dict(d)
    spec = hex_raster.raster_spec(g, factor=1)
    sp = oracle.spec(d, 1)
    assert (spec.height, spec.width) == (sp["height"], sp["width"])
    px, py = oracle.positions(d)
    pts = np.stack([px.ravel(), py.ravel()], axis=1)
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    a, b = np.nonzero(np.triu((dist > 1e-9) & (dist <= step * (1 + 1e-6))))
    assert len(a) == 21 * 22 + 20 * (2 * 23 - 1)             # 22 along each of 21 rows, 45 between each of 20 row pairs
    index = hex_raster.cell_index(g, spec)
    assert not (oracle.covered_mask(d, sp) & (index < 0)).any()        # no hole in the raster
    empty = []
    for k in range(len(a)):
        angles = {"pairs": (a[k:k + 1], b[k:k + 1]), "angle": np.full(1, 20.0, dtype=np.float32)}
        out = gb.render_boundaries_hex(angles, g, spec, [{**BAND_20_DEG, "width": 2}])
        if not (out[..., 3] > 0).any():
            empty.append((int(a[k]), int(b[k])))
    assert empty == []


# --------------------------------------------------------------------- the route

@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _result(stamp):
    fake = _make_fake_result(n_rows=N_ROWS, n_cols=N_COLS)
    fake.xmap.phase_id = np.zeros(N_ROWS * N_COLS, dtype=np.int16)
    fake.xmap.rotations.data = np.stack([_qz(30.0 * k) for k in _classes()])
    fake.metadata = {"scan_geometry": stamp} if stamp is not None else {}
    return fake


def _arm(monkeypatch, result):
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: None)


def test_the_layer_of_a_hexagonal_result_is_drawn_on_the_raster(client, monkeypatch):
    _arm(monkeypatch, _result(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    r = client.get("/api/phasemap/layer", params={"kind": "grain-boundaries", "gb_bands": json.dumps(WHITE_FROM_15)})
    assert r.status_code == 200, r.text
    body = r.json()
    g, pts = _lattice(0.5)
    spec = hex_raster.raster_spec(g)
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["image"]))))
    assert img.shape == (spec.height, spec.width, 4) and body["shape"] == [spec.height, spec.width]
    classes = _classes()
    a, b = _oracle_pairs(pts)
    for pa, pb in zip(a, b):
        if classes[pa] != classes[pb]:
            assert _white_near(img, spec, (pts[pa] + pts[pb]) / 2)
    direct = gb.render_boundaries_hex(_angles(0.5), g, spec, WHITE_FROM_15)
    np.testing.assert_array_equal(img, direct)


def test_display_index_cannot_undo_the_raster_of_the_boundaries(client, monkeypatch):
    # an index-grid picture of these boundaries does not exist: the answer is the raster and says so
    _arm(monkeypatch, _result(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    body = client.get("/api/phasemap/layer", params={"kind": "grain-boundaries", "display": "index",
                                                     "gb_bands": json.dumps(WHITE_FROM_15)}).json()
    spec = hex_raster.raster_spec(_lattice(0.5)[0])
    assert body["shape"] == [spec.height, spec.width] and body["display"]["shape"] == [spec.height, spec.width]


def test_a_square_result_gets_the_square_overlay(client, monkeypatch):
    _arm(monkeypatch, _result(ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict()))
    body = client.get("/api/phasemap/layer",
                      params={"kind": "grain-boundaries", "gb_bands": json.dumps(WHITE_FROM_15)}).json()
    assert body["shape"] == [N_ROWS * gb.SS, N_COLS * gb.SS] and "display" not in body


# The route draws the analysis dataset when no result holds a crystal map, and the gate used to look there too:
# the boundaries must be drawn from the same source (`_hex_topology_of(result, dataset, ...)`).
def _dataset(stamp):
    from types import SimpleNamespace
    fake = _result(None)
    return SimpleNamespace(xmap=fake.xmap, shape=(N_ROWS, N_COLS), scan_geometry=stamp, per_phase_data={})


def test_a_hexagonal_analysis_dataset_gets_its_boundaries_on_the_raster_when_there_is_no_result(client, monkeypatch):
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: None)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset",
                        lambda: _dataset(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    body = client.get("/api/phasemap/layer",
                      params={"kind": "grain-boundaries", "gb_bands": json.dumps(WHITE_FROM_15)}).json()
    g, _ = _lattice(0.5)
    spec = hex_raster.raster_spec(g)
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["image"]))))
    assert body["shape"] == [spec.height, spec.width] and img.shape == (spec.height, spec.width, 4)
    np.testing.assert_array_equal(img, gb.render_boundaries_hex(_angles(0.5), g, spec, WHITE_FROM_15))


def test_a_hexagonal_analysis_dataset_next_to_a_result_without_a_crystal_map_is_drawn_on_the_raster(client, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result",
                        lambda: SimpleNamespace(xmap=None, metadata={}, original_shape=(N_ROWS, N_COLS)))
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset",
                        lambda: _dataset(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    body = client.get("/api/phasemap/layer",
                      params={"kind": "grain-boundaries", "gb_bands": json.dumps(WHITE_FROM_15)}).json()
    spec = hex_raster.raster_spec(_lattice(0.5)[0])
    assert body["shape"] == [spec.height, spec.width]


def test_a_square_result_next_to_a_hexagonal_dataset_keeps_the_square_overlay(client, monkeypatch):
    _arm(monkeypatch, _result(ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict()))
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset",
                        lambda: _dataset(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    body = client.get("/api/phasemap/layer",
                      params={"kind": "grain-boundaries", "gb_bands": json.dumps(WHITE_FROM_15)}).json()
    assert body["shape"] == [N_ROWS * gb.SS, N_COLS * gb.SS] and "display" not in body
