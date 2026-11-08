"""Reading a hexagonal ``.ang`` in Orienta: the scan stamp is kept, or the file is refused by name,
and it is never read as a square map.

Hand values (3 rows, 4 columns, step_x 2.0, step_y = 2 * sqrt(3) / 2 = 1.7320508), the same files the
writer test writes:

* odd rows +0.5 step_x: x = 0 2 4 6 | 1 3 5 7 | 0 2 4 6. x[0,0] = 0 and x[1,0] = 1, so the
  shift is (1 - 0) / 2.0 = +0.5.
* odd rows -0.5 step_x, written with every x lifted by half a step (1.0) so none is negative:
  x = 1 3 5 7 | 0 2 4 6 | 1 3 5 7. x[0,0] = 1 and x[1,0] = 0, so the shift is (0 - 1) / 2.0 = -0.5
  (the lift cancels in the difference).
* y = row * step_y = 0.00000 (x4), 1.73205 (x4), 3.46410 (x4)
* OIM's own form of a 5-row scan with 6 and 5 points in alternate rows: 6+5+6+5+6 = 28 points.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from analysis import ang_hexgrid
from analysis.ang_hexgrid import HexAngRefused, load_ang_xmap, read_ang_grid_header
from backend.api.main import app
from backend.api.routes import analysis as analysis_routes
from backend.api.services import ang_export
from backend.api.services.scan_geometry import ScanGeometry

STEP_X = 2.0
SQRT3 = math.sqrt(3.0)

X_PLUS = [0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6]
X_MINUS = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7]
Y_HEX = [0.0] * 4 + [SQRT3] * 4 + [2 * SQRT3] * 4
X_SQUARE = [0, 2, 4, 6] * 3
Y_SQUARE = [0.0] * 4 + [2.0] * 4 + [4.0] * 4

SCAN_GRID_LINE = "# ORIENTA_SCAN_GRID: HexGrid; rows=3; columns=4 in every row; " \
                 "odd rows at {shift}0.5 step_x; direction={direction}"


def _phi1(n):
    """phi1 of the k-th data line is 0.1 * (k + 1) rad: the order the lines were written in."""
    return [0.1 * (k + 1) for k in range(n)]


def hand_ang(path, x, y, *, grid="HexGrid", ncols_odd=4, ncols_even=4, nrows=3, xstep="2.000000",
             ystep="1.732051", extra=(), not_indexed=(5,), drop=()):
    """Write a TSL-style ``.ang`` by hand: one phase, ten columns, ``len(x)`` lines."""
    lines = ["# TEM_PIXperUM          1.000000"]
    if grid is not None:
        lines.append(f"# GRID: {grid}")
    for key, value in (("XSTEP", xstep), ("YSTEP", ystep), ("NCOLS_ODD", ncols_odd),
                       ("NCOLS_EVEN", ncols_even), ("NROWS", nrows)):
        if key not in drop and value is not None:
            lines.append(f"# {key}: {value}")
    lines += list(extra)
    lines += ["#", "# Phase 1", "# MaterialName      Al", "# Formula       Al",
              "# Symmetry              43",
              "# LatticeConstants      4.049 4.049 4.049  90.000  90.000  90.000", "#"]
    phi1 = _phi1(len(x))
    for k, (xk, yk) in enumerate(zip(x, y)):
        ci = -1.0 if k in not_indexed else 0.8
        lines.append(f"{phi1[k]:.6f} 0.500000 0.250000 {float(xk):.5f} {float(yk):.5f} 100.0 {ci:.3f} 1 0 0.1")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


@pytest.fixture(autouse=True)
def restore_analysis_dataset():
    saved = analysis_routes._dataset
    try:
        yield
    finally:
        analysis_routes._dataset = saved


def _load(path):
    return TestClient(app).post("/api/analysis/load", json={"xmap_path": str(path)})


# ---------------------------------------------------------------------------
# a hexagonal file keeps its stamp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, shift", [(X_PLUS, 0.5), (X_MINUS, -0.5)])
def test_both_signs_through_the_load_route(tmp_path, x, shift):
    path = hand_ang(tmp_path / "hex.ang", x, Y_HEX)
    analysis_routes._dataset = None
    r = _load(path)
    assert r.status_code == 200, r.text
    assert r.json()["shape"] == [3, 4]
    ds = analysis_routes._dataset
    stamp = ds.scan_geometry
    assert (stamp["kind"], stamp["n_rows"], stamp["n_cols"]) == ("hex", 3, 4)
    assert stamp["step_x"] == 2.0 and stamp["step_y"] == pytest.approx(1.732051)
    assert stamp["odd_row_shift"] == shift
    assert stamp["shift_confirmed"] is True            # the direction is read from the x column
    assert ds.step_size == 2.0                          # the analysis basis of hex_guard sees step_x
    assert ds.shape == (3, 4) and ds.size == 12
    # the map is the index-space rectangle, in the order of the data lines
    assert ds.xmap.x.tolist() == [0.0, 2.0, 4.0, 6.0] * 3
    assert ds.xmap.y == pytest.approx([0.0] * 4 + [1.732051] * 4 + [2 * 1.732051] * 4)
    phi1 = ds.xmap.rotations.to_euler()[:, 0]
    assert phi1 == pytest.approx(_phi1(12), abs=1e-5)


def _phase_summary(xmap):
    """(id, name, class, point group name) of every phase of the map; the not-indexed entry has none."""
    return [(i, p.name, type(p), None if p.point_group is None else p.point_group.name)
            for i, p in xmap.phases]


def test_the_map_keeps_what_orix_read(tmp_path):
    """Rotations, phase ids (the not-indexed point stays not indexed), properties and phases are those
    orix reads from the same file; only the coordinates are rebuilt."""
    from orix.io import load as orix_load
    path = hand_ang(tmp_path / "hex.ang", X_PLUS, Y_HEX, not_indexed=(5, 9))
    plain = orix_load(str(path))
    assert plain.size == 12                              # orix itself: 12 points (not its shape)
    xmap, grid = load_ang_xmap(path)
    assert xmap.shape == (3, 4)
    assert np.array_equal(xmap.phase_id, plain.phase_id) and (xmap.phase_id == -1).sum() == 2
    assert np.allclose(xmap.rotations.data, plain.rotations.data)
    assert sorted(xmap.prop.keys()) == sorted(plain.prop.keys())
    for key in plain.prop:
        assert np.array_equal(xmap.prop[key], plain.prop[key]), key
    assert _phase_summary(xmap) == _phase_summary(plain)
    assert [i for i, _ in xmap.phases] == [-1, 1]
    assert xmap.scan_unit == plain.scan_unit


def test_a_frame_phase_stays_a_frame_phase(tmp_path, monkeypatch):
    """Where orix keeps a ``FramePhase`` on the map it read, the rebuilt map holds the same kind of
    phase with the same symmetry, not a plain ``Phase``."""
    import orix.io
    from crystal_symmetry import FramePhase, frame_xmap
    path = hand_ang(tmp_path / "hex.ang", X_PLUS, Y_HEX)
    real = orix.io.load
    monkeypatch.setattr(orix.io, "load", lambda p, *a, **k: frame_xmap(real(p, *a, **k)))
    # the "not indexed" entry (id -1) is made again by orix for every CrystalMap it builds, so it is
    # a plain Phase in the rebuilt map whatever the map before held; the phases of the data are kept
    expected = [row for row in _phase_summary(frame_xmap(real(str(path)))) if row[0] != -1]
    assert [kind for _i, _n, kind, _g in expected] == [FramePhase]
    xmap, _ = load_ang_xmap(path)
    assert [row for row in _phase_summary(xmap) if row[0] != -1] == expected


@pytest.mark.parametrize("direction, confirmed", [("assumed", False), ("confirmed", True)])
def test_the_direction_comes_from_the_orienta_line(tmp_path, direction, confirmed):
    line = SCAN_GRID_LINE.format(shift="+", direction=direction)
    path = hand_ang(tmp_path / "hex.ang", X_PLUS, Y_HEX, extra=[line])
    _xmap, grid = load_ang_xmap(path)
    assert grid["confirmed"] is confirmed and grid["shift"] == 0.5


@pytest.mark.parametrize("shift, confirmed", [(0.5, False), (0.5, True), (-0.5, False), (-0.5, True)])
def test_what_orienta_writes_is_read_back(tmp_path, shift, confirmed):
    """A file written by the exporter: grid and direction come back as they were stated."""
    geometry = ScanGeometry.hex(3, 4, STEP_X, odd_row_shift=shift, shift_confirmed=confirmed)
    n = 12
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(np.array([[10.0 + k, 20.0, 30.0] for k in range(n)]))),
        phase_id=np.zeros(n, dtype=int),
        # the index-space map of an export: x = column * step_x, y = row * step_y (orix reads
        # XSTEP and YSTEP from it)
        x=np.tile(np.arange(4, dtype=float), 3) * STEP_X,
        y=np.repeat(np.arange(3, dtype=float), 4) * geometry.step_y,
        phase_list=PhaseList(phases=[Phase(name="Al", point_group="m-3m")]))
    out = tmp_path / "written.ang"
    comments = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "x"}, step_um=STEP_X, grid=geometry)
    ang_export.write_ang(out, xmap, geometry=geometry, comments=comments)
    # by hand: the file's first x are 0/1 (+0.5) or 1/0 (-0.5)
    x = [float(ln.split()[3]) for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert x[0] == (0.0 if shift > 0 else 1.0) and x[4] == (1.0 if shift > 0 else 0.0)
    got, grid = load_ang_xmap(out)
    assert grid == {"n_rows": 3, "n_cols": 4, "step_x": 2.0, "step_y": pytest.approx(SQRT3, abs=1e-6),
                    "shift": shift, "confirmed": confirmed}
    assert got.shape == (3, 4)
    assert got.rotations.to_euler()[:, 0] == pytest.approx(np.deg2rad([10.0 + k for k in range(n)]),
                                                           abs=1e-5)


@pytest.mark.parametrize("shift, x_expected", [(0.5, X_PLUS), (-0.5, X_MINUS)])
def test_plain_orix_reads_the_written_file_in_file_order(tmp_path, shift, x_expected):
    """A second reader, orix itself without this module: 12 points (``size``, not ``xmap.shape``, which
    is wrong for a hexagonal file), x and y as the hand values in the order of the lines, phi1 in the
    order written."""
    from orix.io import load as orix_load
    geometry = ScanGeometry.hex(3, 4, STEP_X, odd_row_shift=shift, shift_confirmed=True)
    n = 12
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(np.array([[10.0 + k, 20.0, 30.0] for k in range(n)]))),
        phase_id=np.zeros(n, dtype=int),
        x=np.tile(np.arange(4, dtype=float), 3) * STEP_X,
        y=np.repeat(np.arange(3, dtype=float), 4) * geometry.step_y,
        phase_list=PhaseList(phases=[Phase(name="Al", point_group="m-3m")]))
    out = tmp_path / "written.ang"
    comments = ang_export.header_comments(
        method=None, vendor="edax", provenance={"text": "x"}, step_um=STEP_X, grid=geometry)
    ang_export.write_ang(out, xmap, geometry=geometry, comments=comments)
    plain = orix_load(str(out))
    assert plain.size == 12
    assert plain.x == pytest.approx(x_expected, abs=1e-5)
    assert plain.y == pytest.approx(Y_HEX, abs=1e-5)
    assert plain.rotations.to_euler(degrees=True)[:, 0] == pytest.approx([10.0 + k for k in range(n)], abs=1e-3)


def test_a_fine_step_with_thousands_of_rows_is_read(tmp_path):
    """The header states YSTEP with six decimals (0.043301 for the true 0.04330127), the positions
    have five: the last of 2500 rows is 6.8e-4 um from 2499 * YSTEP, far above 1e-3 of the step
    (5e-5 um) and still a regular file."""
    n_rows, n_cols = 2500, 3
    step_x, step_y_true = 0.05, 0.05 * SQRT3 / 2
    x = [(c * step_x + (0.5 * step_x if r % 2 else 0.0)) for r in range(n_rows) for c in range(n_cols)]
    y = [r * step_y_true for r in range(n_rows) for _ in range(n_cols)]
    path = hand_ang(tmp_path / "fine.ang", x, y, ncols_odd=n_cols, ncols_even=n_cols, nrows=n_rows,
                    xstep="0.050000", ystep="0.043301", not_indexed=())
    xmap, grid = load_ang_xmap(path)
    assert xmap.shape == (n_rows, n_cols) and grid["shift"] == 0.5 and grid["n_rows"] == n_rows


# ---------------------------------------------------------------------------
# a hexagonal file that is not read is refused by name
# ---------------------------------------------------------------------------

def _oim_short_rows():
    """OIM's form: rows of 6 and 5 points alternately, 5 rows, 28 points."""
    x, y = [], []
    for r in range(5):
        n = 6 if r % 2 == 0 else 5
        x += [2.0 * c + (1.0 if r % 2 else 0.0) for c in range(n)]
        y += [r * SQRT3] * n
    assert len(x) == 28
    return x, y


def _case(name):
    """(x, y, header keywords) of a hexagonal file that must be refused."""
    if name == "short rows (OIM form)":
        x, y = _oim_short_rows()
        return x, y, dict(ncols_odd=5, ncols_even=6, nrows=5)
    if name == "wrong point count":
        return X_PLUS[:11], Y_HEX[:11], {}
    if name == "one irregular x":
        x = list(X_PLUS)
        x[6] += 0.3                                     # 0.3 um is 15 % of a step
        return x, Y_HEX, {}
    if name == "rows not at multiples of YSTEP":
        y = list(Y_HEX)
        y[8:] = [v + 0.2 for v in y[8:]]
        return X_PLUS, y, {}
    if name == "odd rows a quarter step":
        return [0, 2, 4, 6, 0.5, 2.5, 4.5, 6.5, 0, 2, 4, 6], Y_HEX, {}
    if name == "odd rows alike but one":
        # rows 1 and 3 are odd: row 1 is at +1.0, row 3 at +1.4 (0.4 um is 20 % of a step)
        x = list(X_PLUS) + [1.4, 3.4, 5.4, 7.4] + [0, 2, 4, 6]
        return x, Y_HEX + [3 * SQRT3] * 4 + [4 * SQRT3] * 4, dict(nrows=5)
    if name == "not a hexagonal step ratio":
        return X_PLUS, [0.0] * 4 + [2.0] * 4 + [4.0] * 4, dict(ystep="2.000000")
    if name == "one row":
        return X_PLUS[:4], Y_HEX[:4], dict(nrows=1)
    if name == "no NROWS line":
        return X_PLUS, Y_HEX, dict(drop=("NROWS",))
    if name == "unequal NCOLS":
        return X_PLUS, Y_HEX, dict(ncols_odd=3)
    raise KeyError(name)


CASES = ["short rows (OIM form)", "wrong point count", "one irregular x",
         "rows not at multiples of YSTEP", "odd rows a quarter step", "odd rows alike but one",
         "not a hexagonal step ratio", "one row", "no NROWS line", "unequal NCOLS"]

# A non-finite position passes "abs(position - expected) > tolerance" (False for NaN), and a NaN at the
# start of a row makes the offset of the odd rows NaN, which reads as -0.5: a direction nobody stated.
NON_FINITE = {
    "NaN at the start of the first row": (["nan"] + X_PLUS[1:], Y_HEX),
    "NaN at the start of the second row": (X_PLUS[:4] + ["nan"] + X_PLUS[5:], Y_HEX),
    "NaN x inside a row": (X_PLUS[:6] + ["nan"] + X_PLUS[7:], Y_HEX),
    "NaN y": (X_PLUS, Y_HEX[:9] + ["nan"] + Y_HEX[10:]),
    "infinite x": (X_PLUS[:10] + ["inf"] + X_PLUS[11:], Y_HEX),
    "infinite y": (X_PLUS, Y_HEX[:9] + ["-inf"] + Y_HEX[10:]),
}


@pytest.mark.parametrize("name", CASES)
def test_refusals_by_name(tmp_path, name):
    x, y, kw = _case(name)
    path = hand_ang(tmp_path / "hex.ang", x, y, not_indexed=(), **kw)
    with pytest.raises(HexAngRefused) as err:
        load_ang_xmap(path)
    assert err.value.code == "hex_ang_unreadable"
    assert "HexGrid" in str(err.value)
    _says_what_the_user_can_do(str(err.value))


def _says_what_the_user_can_do(text):
    """Both actions: convert the scan to a square grid in OIM Analysis, or load the scan's .h5 file."""
    assert "square grid in OIM Analysis" in text
    assert "load the scan's .h5 file instead" in text


@pytest.mark.parametrize("name", CASES)
def test_refusals_through_the_load_route(tmp_path, name):
    """400 with the code in the header and a plain English sentence; the dataset that was loaded
    before stays."""
    x, y, kw = _case(name)
    path = hand_ang(tmp_path / "hex.ang", x, y, not_indexed=(), **kw)
    before = object()
    analysis_routes._dataset = before
    r = _load(path)
    assert r.status_code == 400, r.text
    assert r.headers["X-Orienta-Code"] == "hex_ang_unreadable"
    assert "HexGrid" in r.json()["detail"] and "square" in r.json()["detail"]
    _says_what_the_user_can_do(r.json()["detail"])
    assert analysis_routes._dataset is before


@pytest.mark.parametrize("name", list(NON_FINITE))
def test_non_finite_positions_are_refused_by_name(tmp_path, name):
    """The grid checks, on the x/y columns of a hand-made file read as plain numbers (orix 0.14 and
    0.15 stop on such a file before this module sees it, see the next test)."""
    x, y = NON_FINITE[name]
    path = hand_ang(tmp_path / "hex.ang", x, y, not_indexed=())
    columns = np.loadtxt(path, comments="#", usecols=(3, 4))
    assert columns.shape == (12, 2) and not np.isfinite(columns).all()
    with pytest.raises(HexAngRefused) as err:
        ang_hexgrid._grid_of(read_ang_grid_header(path), columns[:, 0], columns[:, 1])
    assert err.value.code == "hex_ang_unreadable" and "not finite" in str(err.value)


@pytest.mark.parametrize("name", list(NON_FINITE))
def test_a_hand_made_file_with_non_finite_positions_gives_no_map(tmp_path, name):
    """orix 0.14 and 0.15 stop on a NaN (ValueError) or infinite (OverflowError) position before the
    grid checks run; for a HexGrid file that becomes the named refusal, through the load route a 400
    with its code, and the dataset that was loaded stays."""
    x, y = NON_FINITE[name]
    path = hand_ang(tmp_path / "hex.ang", x, y, not_indexed=())
    with pytest.raises(HexAngRefused) as err:
        load_ang_xmap(path)
    assert type(err.value) is HexAngRefused and err.value.code == "hex_ang_unreadable"
    assert isinstance(err.value.__cause__, (ValueError, OverflowError))     # the orix error is kept
    assert str(err.value.__cause__) in str(err.value)
    before = object()
    analysis_routes._dataset = before
    r = _load(path)
    assert r.status_code == 400, r.text
    assert r.headers["X-Orienta-Code"] == "hex_ang_unreadable"
    assert "HexGrid" in r.json()["detail"]
    assert analysis_routes._dataset is before


def test_a_square_file_with_a_non_finite_position_keeps_orixs_own_error(tmp_path):
    """The wrap is for HexGrid files only: a SqrGrid file fails as it did before, with orix's error."""
    x, y = [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6], Y_SQUARE
    x[1] = "nan"
    path = hand_ang(tmp_path / "sq.ang", x, y, grid="SqrGrid", ystep="2.000000", not_indexed=())
    with pytest.raises((ValueError, OverflowError)) as err:
        load_ang_xmap(path)
    assert not isinstance(err.value, HexAngRefused)


def test_the_dataset_a_hex_ang_gives_is_refused_by_the_grain_guard(tmp_path):
    """The stamp that ``/api/analysis/load`` puts on a HexGrid file is the one ``hex_guard`` reads: the
    grain-size route (a four-neighbour calculation) refuses it by name, with the existing guard."""
    from backend.api.services.hex_guard import NEIGHBOURHOOD
    path = hand_ang(tmp_path / "hex.ang", X_PLUS, Y_HEX)
    analysis_routes._dataset = None
    assert _load(path).status_code == 200
    r = TestClient(app, raise_server_exceptions=False).post("/api/analysis/grain-size", json={})
    assert r.status_code == 400, r.text
    assert r.headers["X-Orienta-Code"] == NEIGHBOURHOOD
    assert "grain-size analysis" in r.json()["detail"]


# ---------------------------------------------------------------------------
# a square file is read as before
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", ["SqrGrid", None])
def test_square_and_unstated_grids_are_read_as_before(tmp_path, grid):
    from orix.io import load as orix_load
    path = hand_ang(tmp_path / "sq.ang", X_SQUARE, Y_SQUARE, grid=grid, ystep="2.000000")
    plain = orix_load(str(path))
    xmap, stamp = load_ang_xmap(path)
    assert stamp is None
    assert xmap.shape == plain.shape == (3, 4)
    assert np.array_equal(xmap.x, plain.x) and np.array_equal(xmap.y, plain.y)
    assert np.array_equal(xmap.phase_id, plain.phase_id)
    assert np.allclose(xmap.rotations.data, plain.rotations.data)
    analysis_routes._dataset = None
    r = _load(path)
    assert r.status_code == 200, r.text
    assert analysis_routes._dataset.scan_geometry is None      # unknown, as for every file before
    assert analysis_routes._dataset.shape == (3, 4)


def test_a_missing_file_fails_as_before(tmp_path):
    r = _load(tmp_path / "nothing.ang")
    assert r.status_code == 400
    assert "orix.io.load failed" in r.json()["detail"]
    assert "X-Orienta-Code" not in r.headers


def test_the_header_reader_gives_the_grid_lines(tmp_path):
    path = hand_ang(tmp_path / "hex.ang", X_PLUS, Y_HEX,
                    extra=[SCAN_GRID_LINE.format(shift="+", direction="assumed")])
    assert read_ang_grid_header(path) == {
        "GRID": "HexGrid", "XSTEP": "2.000000", "YSTEP": "1.732051", "NCOLS_ODD": "4",
        "NCOLS_EVEN": "4", "NROWS": "3",
        "ORIENTA_SCAN_GRID": "HexGrid; rows=3; columns=4 in every row; odd rows at +0.5 step_x; "
                             "direction=assumed"}


# ---------------------------------------------------------------------------
# the batch of /api/analysis/batch
# ---------------------------------------------------------------------------

try:
    import openpyxl  # noqa: F401
    HAVE_OPENPYXL = True
except ImportError:
    HAVE_OPENPYXL = False
needs_openpyxl = pytest.mark.skipif(not HAVE_OPENPYXL, reason="openpyxl is not installed")


def _square_h5(path):
    """A 10 x 10 map in two grains, as tests/test_batch_processor.py makes them."""
    from orix.io import save
    from orix.quaternion import Orientation
    a, b = Orientation.random(1), Orientation.random(1)
    ori = Orientation.stack([a if c < 5 else b for _ in range(10) for c in range(10)]).flatten()
    xmap = CrystalMap(
        rotations=ori, phase_id=np.zeros(100, dtype=int),
        x=np.tile(np.arange(10), 10) * 0.5, y=np.repeat(np.arange(10), 10) * 0.5,
        phase_list=PhaseList(phases=[Phase(name="Aluminium", space_group=225, structure=None)]),
        prop={"bc": np.random.rand(100) * 200})
    save(str(path), xmap)


@needs_openpyxl
def test_batch_processor_refuses_hex_file_only(tmp_path):
    """A folder with a square map and a hexagonal .ang: the grains and KAM of the batch use four
    neighbours, so the hexagonal file fails alone, with its code, and the folder goes on."""
    from analysis.batch_processor import BatchProcessor
    folder = tmp_path / "folder"
    folder.mkdir()
    _square_h5(folder / "a_square.h5")
    hex_file = hand_ang(folder / "b_hex.ang", X_PLUS, Y_HEX, not_indexed=())
    p = BatchProcessor(str(folder), str(tmp_path / "out.xlsx"))
    assert p.process_all() == 1
    assert p.failed_files == [str(hex_file)]
    assert p.failed_codes == {str(hex_file): "hex_neighbourhood_unsupported"}
    assert (tmp_path / "out.xlsx").exists()


@needs_openpyxl
def test_batch_processor_names_an_unreadable_hex_file(tmp_path):
    from analysis.batch_processor import BatchProcessor
    folder = tmp_path / "folder"
    folder.mkdir()
    _square_h5(folder / "a_square.h5")
    x, y = _oim_short_rows()
    bad = hand_ang(folder / "b_oim.ang", x, y, ncols_odd=5, ncols_even=6, nrows=5, not_indexed=())
    p = BatchProcessor(str(folder), str(tmp_path / "out.xlsx"))
    assert p.process_all() == 1
    assert p.failed_codes == {str(bad): "hex_ang_unreadable"}


@needs_openpyxl
def test_batch_processor_still_reads_a_square_ang(tmp_path):
    """A square .ang of the folder is read through the same door and unchanged."""
    from analysis.batch_processor import BatchProcessor
    from orix.io import save
    from orix.quaternion import Orientation
    folder = tmp_path / "folder"
    folder.mkdir()
    a, b = Orientation.random(1), Orientation.random(1)
    ori = Orientation.stack([a if c < 5 else b for _ in range(10) for c in range(10)]).flatten()
    xmap = CrystalMap(
        rotations=ori, phase_id=np.zeros(100, dtype=int),
        x=np.tile(np.arange(10), 10) * 0.5, y=np.repeat(np.arange(10), 10) * 0.5,
        phase_list=PhaseList(phases=[Phase(name="Al", point_group="m-3m")]),
        prop={"bc": np.random.rand(100) * 200, "ci": np.full(100, 0.9)})
    save(str(folder / "sq.ang"), xmap)
    p = BatchProcessor(str(folder), str(tmp_path / "out.xlsx"))
    assert p.process_all() == 1 and p.failed_files == [] and p.failed_codes == {}


# ---------------------------------------------------------------------------
# the module
# ---------------------------------------------------------------------------

def test_constants_are_those_of_their_origin():
    import edax_hex
    from backend.api.services import scan_geometry
    assert ang_hexgrid.POS_RTOL == edax_hex._POS_RTOL
    assert ang_hexgrid.HEX_ROW_RATIO == scan_geometry.HEX_ROW_RATIO
    assert ang_hexgrid.POS_RTOL == scan_geometry.HEX_RATIO_RTOL


def test_the_package_imports_nothing_from_backend():
    source = Path(ang_hexgrid.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+backend", source, re.M)
