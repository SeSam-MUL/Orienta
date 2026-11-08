"""The display raster of a hexagonal scan: spec, cell index, conversions."""
from __future__ import annotations

import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import hex_raster  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from tests import hex_raster_oracle as oracle  # noqa: E402

FIXTURE = json.loads(oracle.FIXTURE.read_text(encoding="utf-8"))
CASES = FIXTURE["geometries"]
IDS = [c["name"] for c in CASES]
N_A = next(c for c in CASES if c["name"] == "a_3x3_right")["n_inside"]
SQRT3 = math.sqrt(3.0)


def _geometry(case):
    return ScanGeometry.from_dict(case["geometry"])


def test_the_fixture_is_what_the_generator_writes():
    assert json.loads(json.dumps(oracle.build_cases())) == FIXTURE


# ------------------------------------------------------------------- the raster spec

def test_the_raster_of_a_3x3_scan_by_hand():
    # a = 2 um, odd rows +1 um, rows sqrt(3) um apart. x: -1 .. 6 = 7 um = 14 pixels of 0.5 um.
    # y: -2/3*sqrt(3) .. 2*sqrt(3) + 2/3*sqrt(3) = 5.77 um = 11.55 pixels -> 12.
    spec = hex_raster.raster_spec(ScanGeometry.hex(3, 3, 2.0))
    assert (spec.factor, spec.width, spec.height) == (4, 14, 12)
    assert spec.step_um == 0.5 and spec.x0 == -1.0
    assert spec.y0 == pytest.approx(-2 * SQRT3 / 3)


def test_a_negative_offset_extends_the_raster_to_the_left():
    # odd rows sit at -a/2: leftmost cell edge -a/2 - a/2 = -0.5 um; right edge 4*0.5 + 0 + 0.25 = 2.25 um
    spec = hex_raster.raster_spec(ScanGeometry.hex(4, 5, 0.5, odd_row_shift=-0.5))
    assert spec.x0 == -0.5 and spec.width == 22          # 2.75 um / 0.125 um


def test_a_single_row_has_no_offset_to_make_room_for():
    spec = hex_raster.raster_spec(ScanGeometry.hex(1, 4, 1.0))
    assert (spec.width, spec.height) == (16, 5)          # 4 cells * 4 px; 4/3 * b = 1.155 um = 4.62 px -> 5


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_spec_matches_the_fixture(case):
    want = case["spec"]
    spec = hex_raster.raster_spec(_geometry(case), factor=want["factor"])
    assert (spec.factor, spec.height, spec.width) == (want["factor"], want["height"], want["width"])
    assert spec.step_um == want["step_um"]
    assert spec.x0 == pytest.approx(want["x0"], abs=1e-12) and spec.y0 == pytest.approx(want["y0"], abs=1e-12)


@pytest.mark.parametrize("row", FIXTURE["factors"])
def test_the_factor_falls_with_the_size_of_the_scan(row):
    g = ScanGeometry.hex(row["n_rows"], row["n_cols"], row["step_x"])
    assert hex_raster.choose_factor(g) == row["factor"]


def test_a_square_scan_has_no_raster():
    square = ScanGeometry.square(3, 3, 1.0)
    with pytest.raises(ValueError):
        hex_raster.raster_spec(square)
    with pytest.raises(ValueError):                      # an explicit factor skips choose_factor's own check
        hex_raster.raster_spec(square, factor=4)
    assert hex_raster.display_info(square) is None
    assert hex_raster.geometry_from_stamp(square.to_dict()) is None
    assert hex_raster.geometry_from_stamp(None) is None


def test_display_info_is_what_the_interface_cross_checks():
    info = hex_raster.display_info(ScanGeometry.hex(3, 3, 2.0))
    assert info["kind"] == "hex_raster" and info["factor"] == 4
    assert info["shape"] == [12, 14] and info["step_um"] == 0.5


def test_geometry_from_stamp_rebuilds_a_hexagonal_geometry_and_refuses_a_mask():
    g = ScanGeometry.hex(3, 3, 2.0)
    assert hex_raster.geometry_from_stamp(g.to_dict()) == g
    with pytest.raises(ValueError):
        hex_raster.geometry_from_stamp({**g.to_dict(), "has_valid_mask": True})


# ------------------------------------------------------------------ the cells

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_cell_index_equals_the_brute_force_argmin(case):
    expected = oracle.brute_cell_index(case["geometry"], case["spec"])
    g = _geometry(case)
    got = hex_raster.cell_index(g, hex_raster.raster_spec(g, factor=case["spec"]["factor"]))
    assert got.shape == expected.shape and got.dtype == np.int32 and not got.flags.writeable
    np.testing.assert_array_equal(got, expected)
    assert int((got >= 0).sum()) == case["n_inside"]


def test_a_pixel_inside_a_hexagon_belongs_to_it():
    g = ScanGeometry.hex(3, 3, 2.0)
    spec = hex_raster.raster_spec(g)
    # pixel (2, 2): centre (0.25, 0.095) um, 0.27 um from point (0, 0) and inside its hexagon
    assert hex_raster.cell_at_pixel(g, spec, 2, 2) == (0, 0)


def test_pixels_no_hexagon_covers_belong_to_no_cell():
    g = ScanGeometry.hex(3, 3, 2.0)
    spec = hex_raster.raster_spec(g)
    # pixel (0, 0): centre (-0.75, -0.905) um. Its nearest point (0, 0) is dx = 0.75 <= a/2 = 1 away in x,
    # but the hexagon is only rho * (1 - 0.75/2) = 0.722 um high at that x and dy = 0.905.
    assert hex_raster.cell_at_pixel(g, spec, 0, 0) is None


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_sampled_pixels_match_the_fixture(case):
    g = _geometry(case)
    spec = hex_raster.raster_spec(g, factor=case["spec"]["factor"])
    for p in case["pixels"]:
        want = None if p["cell"] is None else tuple(p["cell"])
        assert hex_raster.cell_at_pixel(g, spec, p["i"], p["j"]) == want, (p["i"], p["j"])


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_cell_owns_the_pixel_that_holds_its_centre(case):
    # holds for every case of the fixture; at factor 1 in general it does not (see pixel_of_cell)
    g = _geometry(case)
    spec = hex_raster.raster_spec(g, factor=case["spec"]["factor"])
    for c in case["cells"]:
        i, j = hex_raster.pixel_of_cell(g, spec, c["row"], c["col"])
        assert [i, j] == c["pixel"]
        assert hex_raster.cell_at_pixel(g, spec, i, j) == (c["row"], c["col"])


def test_a_tie_between_two_columns_goes_to_the_lower_one_and_the_border_belongs_to_the_cell():
    # a = 2 um (exact in binary), odd rows at -1 um, factor 1: pixels of 2 um, and every pixel centre meets
    # the lattice exactly. Rows are b = sqrt(3) um apart, the hexagon is rho = 2b/3 = 1.155 um tall at dx = 0.
    # x: -2 .. 5 um -> 4 pixels, centres -1, 1, 3, 5. y: -1.155 .. 4.619 um -> 3 pixels, centres
    # -0.155, 1.845, 3.845. Row 0 points sit at x = 0, 2, 4; row 1 at x = -1, 1, 3; row 2 at 0, 2, 4.
    g = ScanGeometry.hex(3, 3, 2.0, odd_row_shift=-0.5)
    spec = hex_raster.raster_spec(g, factor=1)
    assert (spec.height, spec.width, spec.step_um, spec.x0) == (3, 4, 2.0, -2.0)
    want = [
        # centre x = -1: on the border dx = a/2 of the first column, which keeps it (row 1: on the point).
        # centre x = 1 and 3 in rows 0 and 2: halfway between two points, a tie, to the lower column.
        # row 1, x = 5: the nearest point is (2, 2), whose hexagon is 1.619 um away in y: no cell.
        [(0, 0), (0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2), None],
        [(2, 0), (2, 0), (2, 1), (2, 2)],
    ]
    for i in range(3):
        for j in range(4):
            assert hex_raster.cell_at_pixel(g, spec, i, j) == want[i][j], (i, j)
    idx = hex_raster.cell_index(g, spec)
    np.testing.assert_array_equal(idx, [[w[0] * 3 + w[1] if w else -1 for w in r] for r in want])


def test_a_spec_of_another_geometry_is_refused():
    g = ScanGeometry.hex(4, 3, 2.0)
    other = hex_raster.raster_spec(ScanGeometry.hex(3, 3, 2.0))      # 12 x 14 px; this scan needs 16 x 14
    assert hex_raster.raster_spec(g).height == 16
    with pytest.raises(ValueError):
        hex_raster.cell_index(g, other)
    with pytest.raises(ValueError):
        hex_raster.to_display(np.zeros((4, 3)), g, spec=other)
    assert hex_raster.cell_index(g, hex_raster.raster_spec(g)).shape == (16, 14)


def test_the_hexagons_tile_the_raster_with_their_own_area():
    # A hexagon of a lattice of spacing a has the area a * b; a pixel is (a/4)^2.
    g = ScanGeometry.hex(12, 12, 0.5)
    idx = hex_raster.cell_index(g)
    assert (idx >= 0).sum() == pytest.approx(144 * 16 * (g.step_y / g.step_x), rel=0.02)
    counts = np.bincount(idx[idx >= 0], minlength=144)
    assert counts.min() >= 12 and counts.max() <= 16


def test_a_cropped_geometry_gets_the_raster_of_its_reversed_offset():
    parent = ScanGeometry.hex(6, 5, 1.0)
    crop = parent.cropped(1, 0, 3, 5)                    # starts on an odd row: the offset reverses
    assert crop.odd_row_shift == -0.5
    spec = hex_raster.raster_spec(crop)
    assert spec.x0 == -1.0                               # odd rows at -0.5 um, minus a/2
    # the cut-out's row 0 is the parent's row 1, so crop cell (0, 0) sits half a step (2 pixels of
    # 0.25 um) to the RIGHT of crop cell (1, 0)
    j0 = hex_raster.pixel_of_cell(crop, spec, 0, 0)[1]
    j1 = hex_raster.pixel_of_cell(crop, spec, 1, 0)[1]
    assert j0 - j1 == 2


def test_the_cell_index_does_not_depend_on_the_chunk_size(monkeypatch):
    g = ScanGeometry.hex(9, 8, 0.7, odd_row_shift=-0.5)
    whole = hex_raster.cell_index(g).copy()
    hex_raster._cell_index_cached.cache_clear()
    monkeypatch.setattr(hex_raster, "CHUNK_PIXELS", 100)           # width 33 -> 3 rows at a time
    np.testing.assert_array_equal(hex_raster.cell_index(g), whole)
    hex_raster._cell_index_cached.cache_clear()


@pytest.mark.parametrize("chunk_pixels, rows_per_chunk", [(4 * 802, 4), (10, 1)])
def test_a_chunk_is_bounded_by_pixels_not_by_rows(monkeypatch, chunk_pixels, rows_per_chunk):
    # hex(4, 200, 0.5) at factor 4, by hand: x from -0.25 to 199 * 0.5 + 0.25 + 0.25 = 100.0 um, i.e.
    # 100.25 um / 0.125 um = 802 px; y: 3 * 0.433 + 2 * 0.2887 = 1.877 um / 0.125 um = 15.01 -> 16 px high.
    g = ScanGeometry.hex(4, 200, 0.5, odd_row_shift=0.5)
    spec = hex_raster.raster_spec(g)
    assert (spec.factor, spec.width, spec.height) == (4, 802, 16)
    whole = hex_raster.cell_index(g).copy()
    hex_raster._cell_index_cached.cache_clear()
    sizes = []
    real_assign = hex_raster._assign

    def counting(geometry, x, y):
        sizes.append(x.size)
        return real_assign(geometry, x, y)

    monkeypatch.setattr(hex_raster, "CHUNK_PIXELS", chunk_pixels)
    monkeypatch.setattr(hex_raster, "_assign", counting)
    got = hex_raster.cell_index(g)
    hex_raster._cell_index_cached.cache_clear()
    np.testing.assert_array_equal(got, whole)
    assert len(sizes) == math.ceil(16 / rows_per_chunk)            # 4 chunks of 4 rows; 16 chunks of one row
    assert max(sizes) == 802 * rows_per_chunk
    assert rows_per_chunk == 1 or max(sizes) <= chunk_pixels      # a wider raster than the bound gets one row


# --------------------------------------------------------------- conversions

def test_to_display_places_every_value_on_its_hexagon_and_fills_the_rest():
    g = ScanGeometry.hex(3, 3, 2.0)
    spec = hex_raster.raster_spec(g)
    values = np.arange(9, dtype=np.int16).reshape(3, 3) + 1           # 1..9, never the fill 0
    out = hex_raster.to_display(values, g)
    assert out.shape == (spec.height, spec.width) and out.dtype == np.int16
    assert int((out > 0).sum()) == N_A
    i, j = hex_raster.pixel_of_cell(g, spec, 2, 1)
    assert out[i, j] == 8                                             # flat index 2*3 + 1, plus one
    rgba = np.zeros((3, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    shown = hex_raster.to_display(rgba, g)
    assert shown.shape == (spec.height, spec.width, 4)
    assert int((shown[..., 3] == 255).sum()) == N_A
    assert int(np.isnan(hex_raster.to_display(values.astype(float), g, fill=np.nan)).sum()) \
        == spec.height * spec.width - N_A
    with pytest.raises(ValueError):
        hex_raster.to_display(np.zeros((3, 4)), g)                    # not this scan's index grid
    with pytest.raises(ValueError):
        hex_raster.to_display(np.zeros((9, 1)), g)                    # 9 values, but not the 3 x 3 grid


def test_png_to_display_returns_an_rgba_png_of_the_raster():
    from PIL import Image
    g = ScanGeometry.hex(3, 3, 2.0)
    spec = hex_raster.raster_spec(g)
    grey = np.arange(9, dtype=np.uint8).reshape(3, 3) * 20 + 10
    buf = io.BytesIO()
    Image.fromarray(grey).save(buf, format="PNG")
    out = hex_raster.png_to_display(base64.b64encode(buf.getvalue()).decode(), g)
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(out))))
    assert img.shape == (spec.height, spec.width, 4)
    i, j = hex_raster.pixel_of_cell(g, spec, 1, 2)
    assert tuple(img[i, j]) == (110, 110, 110, 255)                   # cell 1*3 + 2 = 5: 5*20 + 10
    assert int((img[..., 3] == 255).sum()) == N_A and int(img[img[..., 3] == 0].sum()) == 0


# ------------------------------------------- a step with rounding noise leaves no hole in the raster

COVERAGE = FIXTURE["coverage"]


def _lattice_raster(rows, cols, step, shift, factor):
    d = oracle.geometry_dict(rows, cols, step, shift)
    g = ScanGeometry.from_dict(d)
    return d, g, hex_raster.raster_spec(g, factor=factor)


@pytest.mark.parametrize("case", COVERAGE, ids=[c["name"] for c in COVERAGE])
def test_every_pixel_of_the_union_of_the_hexagons_belongs_to_a_cell_and_no_other_does(case):
    # A pixel centre on the border between two cells (factor 1, a step that is not exact in binary) used to
    # fall into none: the nearest point was the one whose border the noise had pushed 1e-16 away.
    g = ScanGeometry.from_dict(case["geometry"])
    spec = hex_raster.raster_spec(g, factor=case["spec"]["factor"])
    want = np.array([[c == "#" for c in row] for row in case["covered"]])
    assert want.shape == (spec.height, spec.width) and want.any() and not want.all()
    idx = hex_raster.cell_index(g, spec)
    np.testing.assert_array_equal(idx >= 0, want)
    got = np.array([[hex_raster.cell_at_pixel(g, spec, i, j) is not None for j in range(spec.width)]
                    for i in range(spec.height)])
    np.testing.assert_array_equal(got, want)
    assert _pixels_outside_their_cell(g, spec, idx) == 0


def _pixels_outside_their_cell(g, spec, idx):
    """How many pixels with a cell lie outside the closed hexagon (plus the oracle's tolerance) around the
    position of the cell they were given. Found from ScanGeometry.positions() and the hexagon alone, without
    asking which point is nearest: a mask of covered pixels cannot tell a cell shifted by a column."""
    px, py = g.positions()
    a, rho = float(g.step_x), 2 * float(g.step_y) / 3
    tol = oracle.COVER_TOLERANCE * a
    i, j = np.nonzero(idx >= 0)
    row, col = np.divmod(idx[i, j], g.n_cols)
    dx = np.abs(spec.x0 + (j + 0.5) * spec.step_um - px[row, col])
    dy = np.abs(spec.y0 + (i + 0.5) * spec.step_um - py[row, col])
    return int((~((dx <= a / 2 + tol) & (dy <= rho * (1 - dx / a) + tol))).sum())


@pytest.mark.parametrize("factor", (4, 2, 1))
@pytest.mark.parametrize("shift", (0.5, -0.5))
@pytest.mark.parametrize("step", (2.0, 1.7, 1.0, 0.7, 3.3, 0.5, 0.35))
def test_the_cell_index_has_no_hole_and_no_pixel_outside_the_scan(step, shift, factor):
    d, g, spec = _lattice_raster(21, 23, step, shift, factor)
    sp = oracle.spec(d, factor)
    assert (spec.height, spec.width) == (sp["height"], sp["width"])
    idx = hex_raster.cell_index(g, spec)
    covered = oracle.covered_mask(d, sp)
    holes = int((covered & (idx < 0)).sum())
    outside = int((~covered & (idx >= 0)).sum())
    assert (holes, outside) == (0, 0)
    # and the cell a pixel is given is one whose hexagon holds it, not merely some cell
    assert _pixels_outside_their_cell(g, spec, idx) == 0
    # the area is the lattice's: a * b per cell, a pixel being (a / factor) squared
    assert (idx >= 0).sum() == pytest.approx(g.n_points * factor ** 2 * g.step_y / g.step_x, rel=0.1)


def test_a_pixel_on_the_border_between_two_cells_goes_to_one_of_them_by_hand():
    # a = 1.7 um, factor 1, odd rows +a/2, 6 x 6. x0 = -0.85 um, so the centre of pixel column j is at
    # -0.85 + (j + 0.5) * 1.7 = 1.7 * j. Pixel (1, 2): x = 3.4 um, y = -2b/3 + 1.5 * 1.7 = 1.569 um with
    # b = 1.7 * sqrt(3) / 2 = 1.472 um, so next to the odd row 1 (y = 1.472 um), whose points sit at
    # 0.85, 2.55, 4.25 um: x = 3.4 um is halfway between cells (1, 1) and (1, 2), on the border of both.
    d, g, spec = _lattice_raster(6, 6, 1.7, 0.5, 1)
    assert hex_raster.cell_at_pixel(g, spec, 1, 2) in {(1, 1), (1, 2)}
    assert hex_raster.cell_index(g, spec)[1, 2] in {1 * 6 + 1, 1 * 6 + 2}


@pytest.mark.parametrize("x, y, inside", [
    (-1.0 - 1e-13, 0.0, True),                    # vertical border of cell (0, 0), a hair beyond it: rounding noise
    (-1.0 - 1e-6, 0.0, False),                    # ... a millionth of a micrometre beyond it: outside
    (0.5, SQRT3 / 2 + 1e-13, True),               # slanted border, a hair beyond it
    (0.5, SQRT3 / 2 + 1e-6, False),
    (0.5, SQRT3 / 2 - 1e-6, True),
])
def test_the_border_of_a_hexagon_holds_a_hair_beyond_it_and_not_a_millionth_of_a_micrometre(x, y, inside):
    # One row of two points, a = 2 um at x = 0 and 2, rows b = sqrt(3) um apart, so rho = 2b/3 = 2/sqrt(3) um.
    # The vertical border of cell (0, 0) is at x = -1. The slanted one at dx = 0.5 is at dy = rho * (1 - 0.5 / 2)
    # = 2/sqrt(3) * 3/4 = sqrt(3)/2 um. A raster of one pixel of 1 um whose centre is (x, y).
    g = ScanGeometry.hex(1, 2, 2.0)
    spec = hex_raster.RasterSpec(1, 1, 1, 1.0, x - 0.5, y - 0.5)
    assert hex_raster.cell_at_pixel(g, spec, 0, 0) == ((0, 0) if inside else None)


def test_the_border_tolerance_is_the_one_the_fixture_carries():
    # one number for both implementations: hexGeometry.test.js holds EDGE_TOLERANCE of hexGeometry.js to the same
    assert FIXTURE["edge_tolerance"] == oracle.COVER_TOLERANCE
    assert hex_raster.EDGE_TOLERANCE == FIXTURE["edge_tolerance"]


@pytest.mark.parametrize("case", FIXTURE["borders"], ids=[c["name"] for c in FIXTURE["borders"]])
def test_the_border_tolerance_is_a_fraction_of_the_step_whatever_the_step(case):
    # A tolerance in micrometres instead of in steps fails at a step of 0.0017 um (too wide: a pixel beyond the
    # border is taken in) and at 1700 um (too narrow: offsets of a few steps' millionths are refused). The
    # offsets are fractions of the step, so the same fixture row means the same thing at every step.
    g = ScanGeometry.from_dict(case["geometry"])
    a = case["geometry"]["step_x"]
    spec = hex_raster.RasterSpec(1, 1, 1, a, case["x"] - a / 2, case["y"] - a / 2)
    assert hex_raster.cell_at_pixel(g, spec, 0, 0) == ((0, 0) if case["inside"] else None)
