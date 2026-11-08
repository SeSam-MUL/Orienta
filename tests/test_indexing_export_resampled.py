"""``POST /api/indexing/export`` with ``ctf`` and ``ang_square`` for a hexagonal result.

Both formats hold the scan resampled to a square grid by OIM's nearest-neighbour rule. The scan used
throughout is 3 rows x 4 columns, step_x 2 um, odd rows half a step to the right (step_y = sqrt(3)),
the one ``tests/test_hex_resample.py`` works through by hand (with step 1):

* the square grid is 3 x 4 with step 2: x 0 2 4 6, y 0 2 4 (round(2 * sqrt(3) / 2) = 2, plus 1);
* cell (y 0) takes points 0 1 2 3; cell (y 2) takes row 1 (y 1.732, 0.268 away): x 0 -> point 4 (x 1),
  x 2 ties between points 4 (x 1) and 5 (x 3) and the lower column wins (4), x 4 -> 5, x 6 -> 6;
  cell (y 4) takes row 2: 8 9 10 11. Point 7 is nobody's nearest.
  So ``point_index = [0 1 2 3 | 4 4 5 6 | 8 9 10 11]``, 12 source points, 12 cells, 1 unused.
* point k has Euler angles (10 + 11 k, 20 + k, 30 + 2 k) degrees and confidence 0.3 + 0.6 k / 11;
  points 0, 3, 6, 9 are the second phase, the others Al.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from diffpy.structure import Lattice, Structure
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from backend.api.main import app
from backend.api.problem import CODE_HEADER
from backend.api.routes import indexing as indexing_mod
from backend.api.services import ang_export
from backend.api.services.scan_geometry import ScanGeometry
from indexing_controller import IndexingMethod, IndexingResult

N_ROWS, N_COLS = 3, 4
STEP_X = 2.0
POINT_INDEX = [0, 1, 2, 3, 4, 4, 5, 6, 8, 9, 10, 11]
SECOND_PHASE_POINTS = {0, 3, 6, 9}
NOTE = ("Resampled from a hexagonal scan to a square grid by nearest neighbour at step 2 um "
        "(source points 12, points written 12, source points unused 1); nothing was interpolated.")
ASSUMPTION = (" The direction of the row offset is an assumption (no file states it): the odd rows "
              "are placed half a step to the right of the even rows.")
HEADER_JSON = {"step_um": 2.0, "rows": 3, "cols": 4, "source_points": 12, "points": 12,
               "unused": 1, "shift_confirmed": False}


@pytest.fixture(autouse=True)
def restore_indexing_registry():
    saved = dict(indexing_mod._result_registry)
    saved_active = indexing_mod._active_result_id
    try:
        yield
    finally:
        indexing_mod._result_registry.clear()
        indexing_mod._result_registry.update(saved)
        indexing_mod._active_result_id = saved_active


def _second_phase(kind):
    if kind == "162":
        return Phase(name="Hx", space_group=162,
                     structure=Structure(lattice=Lattice(4.0, 4.0, 5.0, 90, 90, 120)))
    if kind == "Ti":
        return Phase(name="Ti", space_group=194,
                     structure=Structure(lattice=Lattice(2.95, 2.95, 4.68, 90, 90, 120)))
    raise AssertionError(kind)


def _make_result(*, second="162", geometry="hex", hex_resampled=None, phase_of_point=None,
                 confirmed=False, n_rows=N_ROWS):
    n = n_rows * N_COLS
    eulers = np.deg2rad(np.array([[10.0 + 11.0 * k, 20.0 + k, 30.0 + 2.0 * k] for k in range(n)]))
    al = Phase(name="Al", space_group=225,
               structure=Structure(lattice=Lattice(4.049, 4.049, 4.049, 90, 90, 90)))
    other = _second_phase(second)
    if phase_of_point is None:
        pid = np.array([2 if k in SECOND_PHASE_POINTS else 1 for k in range(n)])
    else:
        pid = np.asarray(phase_of_point)
    xs = (np.arange(n) % N_COLS).astype(float)           # pixels, as the real map carries
    ys = (np.arange(n) // N_COLS).astype(float)
    xmap = CrystalMap(rotations=Rotation.from_euler(eulers), phase_id=pid, x=xs, y=ys,
                      phase_list=PhaseList(phases=[al, other], ids=[1, 2]))
    md = {"step_size_um": STEP_X, "source_vendor": "edax"}
    if geometry == "hex":
        md["scan_geometry"] = ScanGeometry.hex(
            n_rows, N_COLS, STEP_X, odd_row_shift=0.5, shift_confirmed=confirmed).to_dict()
    elif geometry == "square":
        md["scan_geometry"] = ScanGeometry.square(n_rows, N_COLS, STEP_X).to_dict()
    if hex_resampled is not None:
        md["hex_resampled"] = hex_resampled
    return IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool), original_shape=(n_rows, N_COLS),
        method=IndexingMethod.SPHERICAL, confidence_scores=(0.3 + 0.6 * np.arange(n) / 11.0).astype(
            np.float32),
        metadata=md)


def _activate(monkeypatch, result):
    monkeypatch.setitem(indexing_mod._result_registry, "test_resampled_export", result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", "test_resampled_export")
    monkeypatch.setattr("backend.api.routes.ebsd_viewer._ebsd_file_path", None, raising=False)


def _post(body, **kw):
    return TestClient(app).post("/api/indexing/export", json=body, **kw)


def _export(monkeypatch, tmp_path, fmt, result=None, name=None):
    _activate(monkeypatch, result if result is not None else _make_result())
    out = tmp_path / (name or ("r.ctf" if fmt == "ctf" else "r_square.ang"))
    resp = _post({"format": fmt, "filename": str(out)})
    assert resp.status_code == 200, resp.text
    return out, resp


def _ctf_parts(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = lines.index("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS")
    return lines[:i], [ln.split("\t") for ln in lines[i + 1:]]


def _ang_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def _ang_rows(path):
    return [ln.split() for ln in _ang_lines(path) if ln.strip() and not ln.startswith("#")]


def _ang_field(path, key):
    pat = re.compile(r"^#\s*" + re.escape(key) + r"\s+(.*)$")
    return [m.group(1).strip() for ln in _ang_lines(path) if ln.startswith("#") and (m := pat.match(ln))]


def _ang_symmetry_of(path, name):
    blocks, cur = [], {}
    for ln in _ang_lines(path):
        if not ln.startswith("#"):
            break
        if re.match(r"^#\s*Phase\s+\d+\s*$", ln):
            cur = {}
            blocks.append(cur)
        m = re.match(r"^#\s*(MaterialName|Symmetry)\s+(.*)$", ln)
        if m and blocks:
            blocks[-1][m.group(1)] = m.group(2).strip()
    return [b["Symmetry"] for b in blocks if b.get("MaterialName") == name]


# ---------------------------------------------------------------------------
# .ctf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("second, laue, sg, lattice", [
    # 162 is P-31m: Laue class -3m = Oxford 7. Ti is P63/mmc: 6/mmm = Oxford 9.
    ("162", 7, 162, "4.000;4.000;5.000\t90.0;90.0;120.0"),
    ("Ti", 9, 194, "2.950;2.950;4.680\t90.0;90.0;120.0"),
])
def test_the_ctf_phase_table_keeps_laue_and_space_group(monkeypatch, tmp_path, second, laue, sg, lattice):
    out, _ = _export(monkeypatch, tmp_path, "ctf", _make_result(second=second))
    head, _rows = _ctf_parts(out)
    name = {"162": "Hx", "Ti": "Ti"}[second]
    assert "Phases\t2" in head
    i = head.index("Phases\t2")
    # Al: space group 225 (Fm-3m), Laue class m-3m = Oxford 11, cubic cell of 4.049
    assert head[i + 1] == "4.049;4.049;4.049\t90.0;90.0;90.0\tAl\t11\t225"
    assert head[i + 2] == f"{lattice}\t{name}\t{laue}\t{sg}"


def test_the_ctf_holds_the_square_grid_by_hand(monkeypatch, tmp_path):
    out, _ = _export(monkeypatch, tmp_path, "ctf")
    head, rows = _ctf_parts(out)
    assert "XCells\t4" in head and "YCells\t3" in head
    assert "XStep\t2.0" in head and "YStep\t2.0" in head          # the step of the square grid
    assert len(rows) == 12
    assert [r[1] for r in rows] == ["0.0000", "2.0000", "4.0000", "6.0000"] * 3
    assert [r[2] for r in rows] == ["0.0000"] * 4 + ["2.0000"] * 4 + ["4.0000"] * 4
    # the phase of the point each cell takes: points 0 3 6 9 are the second phase (2), the rest Al (1)
    assert [int(r[0]) for r in rows] == [2 if p in SECOND_PHASE_POINTS else 1 for p in POINT_INDEX]
    # Euler angles of the taken point, in degrees (EDAX source: the native frame, no conversion)
    for cell, point in enumerate(POINT_INDEX):
        got = [float(v) for v in rows[cell][5:8]]
        np.testing.assert_allclose(got, [10.0 + 11.0 * point, 20.0 + point, 30.0 + 2.0 * point],
                                   atol=2e-4, err_msg=f"cell {cell} takes point {point}")
    # no band contrast was exported: honest zeros, and the neutral band count
    assert {r[9] for r in rows} == {"0"} and {r[3] for r in rows} == {"8"}


def test_the_ctf_project_line_says_it_was_resampled(monkeypatch, tmp_path):
    out, _ = _export(monkeypatch, tmp_path, "ctf")
    head, _rows = _ctf_parts(out)
    prj = [ln for ln in head if ln.startswith("Prj\t")][0]
    assert "Note: " + NOTE + ASSUMPTION in prj
    assert "resampled from a hexagonal grid" in prj                 # the scan-origin sentence


def _frame_lines(path):
    return [ln for ln in _ang_lines(path) if ln.startswith("# ORIENTA_CRYSTAL_FRAME: ")]


@pytest.mark.parametrize("second, y_line", [("162", True), ("Ti", False)])
def test_the_resampled_files_state_the_crystal_frame(monkeypatch, tmp_path, second, y_line):
    result = _make_result(second=second)
    ctf, _ = _export(monkeypatch, tmp_path, "ctf", result)
    sq, _ = _export(monkeypatch, tmp_path, "ang_square", result)
    prj = [ln for ln in _ctf_parts(ctf)[0] if ln.startswith("Prj\t")][0]
    lines = _frame_lines(sq)
    assert len(lines) == (2 if y_line else 1)
    assert "X||a" in lines[0] and "Z||c*" in lines[0] and "X||a" in prj
    if y_line:
        assert "Phase Hx: space group 162, point group -31m" in lines[1] and "along Y" in lines[1]
        assert "Phase Hx: point group -31m" in prj and "along Y" in prj
    else:
        assert "along Y" not in prj


def test_the_resampled_files_of_a_cubic_phase_state_no_crystal_frame(monkeypatch, tmp_path):
    # every point is Al (phase 1): the second phase is not in the resampled map at all
    result = _make_result(phase_of_point=[1] * 12)
    ctf, _ = _export(monkeypatch, tmp_path, "ctf", result)
    sq, _ = _export(monkeypatch, tmp_path, "ang_square", result)
    assert _frame_lines(sq) == []
    assert "X||a" not in [ln for ln in _ctf_parts(ctf)[0] if ln.startswith("Prj\t")][0]


def test_a_confirmed_row_offset_leaves_the_assumption_out(monkeypatch, tmp_path):
    out, _ = _export(monkeypatch, tmp_path, "ctf", _make_result(confirmed=True))
    head, _rows = _ctf_parts(out)
    prj = [ln for ln in head if ln.startswith("Prj\t")][0]
    assert "Note: " + NOTE in prj and "assumption" not in prj


# ---------------------------------------------------------------------------
# square .ang
# ---------------------------------------------------------------------------

def test_the_square_ang_holds_the_square_grid_by_hand(monkeypatch, tmp_path):
    out, _ = _export(monkeypatch, tmp_path, "ang_square")
    assert _ang_field(out, "GRID:") == ["SqrGrid"]
    assert _ang_field(out, "XSTEP:") == ["2.000000"] and _ang_field(out, "YSTEP:") == ["2.000000"]
    assert (_ang_field(out, "NCOLS_ODD:"), _ang_field(out, "NCOLS_EVEN:"), _ang_field(out, "NROWS:")) == (
        ["4"], ["4"], ["3"])
    rows = _ang_rows(out)
    assert len(rows) == 12
    assert [r[3] for r in rows] == ["0.00000", "2.00000", "4.00000", "6.00000"] * 3
    assert [r[4] for r in rows] == ["0.00000"] * 4 + ["2.00000"] * 4 + ["4.00000"] * 4
    # phase column: the position of the cell's phase in the file (Al 1, second phase 2)
    assert [int(r[7]) for r in rows] == [2 if p in SECOND_PHASE_POINTS else 1 for p in POINT_INDEX]
    for cell, point in enumerate(POINT_INDEX):
        np.testing.assert_allclose(
            [float(v) for v in rows[cell][0:3]],
            np.deg2rad([10.0 + 11.0 * point, 20.0 + point, 30.0 + 2.0 * point]), atol=2e-5)
        assert float(rows[cell][6]) == pytest.approx(0.3 + 0.6 * point / 11.0, abs=2e-5)  # CI
        assert float(rows[cell][5]) == 0.0                                                # IQ


def test_the_square_ang_names_the_symmetry_by_tsl_code(monkeypatch, tmp_path):
    out, _ = _export(monkeypatch, tmp_path, "ang_square", _make_result(second="162"))
    assert _ang_symmetry_of(out, "Al") == ["43"]
    assert _ang_symmetry_of(out, "Hx") == ["32"]            # -3m
    out_ti, _ = _export(monkeypatch, tmp_path, "ang_square", _make_result(second="Ti"), name="ti.ang")
    assert _ang_symmetry_of(out_ti, "Ti") == ["62"]         # 6/mmm


def test_the_square_ang_header_states_the_resampling(monkeypatch, tmp_path):
    out, _ = _export(monkeypatch, tmp_path, "ang_square")
    text = "\n".join(_ang_lines(out))
    assert "# ORIENTA_NOTE: " + NOTE + ASSUMPTION in text
    assert "resampled from a hexagonal grid" in text
    assert "HexGrid" not in text                           # not the hexagonal layout


def test_a_phase_no_cell_takes_is_not_numbered(monkeypatch, tmp_path):
    # Only point 7 is the second phase, and no cell takes point 7: the resampled map has one phase.
    pid = [2 if k == 7 else 1 for k in range(12)]
    out, _ = _export(monkeypatch, tmp_path, "ang_square", _make_result(phase_of_point=pid))
    assert _ang_symmetry_of(out, "Al") == ["43"]
    assert _ang_symmetry_of(out, "Hx") == []
    assert [ln for ln in _ang_lines(out) if re.match(r"^#\s*Phase\s+\d+\s*$", ln)] == ["# Phase 1"]
    ctf, _ = _export(monkeypatch, tmp_path, "ctf", _make_result(phase_of_point=pid))
    head, rows = _ctf_parts(ctf)
    assert "Phases\t1" in head and {r[0] for r in rows} == {"1"}


# ---------------------------------------------------------------------------
# the note rides as a header, names, frame warning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["ctf", "ang_square"])
def test_the_resampling_travels_in_a_header(monkeypatch, tmp_path, fmt):
    _out, resp = _export(monkeypatch, tmp_path, fmt)
    raw = resp.headers["x-orienta-resampled"]
    raw.encode("ascii")
    assert json.loads(raw) == HEADER_JSON
    # the booleans are JSON ones and the count fields integers
    assert isinstance(json.loads(raw)["shift_confirmed"], bool)
    assert all(type(json.loads(raw)[k]) is int for k in ("rows", "cols", "source_points", "points", "unused"))


def test_a_confirmed_offset_is_stated_in_the_header(monkeypatch, tmp_path):
    _out, resp = _export(monkeypatch, tmp_path, "ctf", _make_result(confirmed=True))
    assert json.loads(resp.headers["x-orienta-resampled"])["shift_confirmed"] is True


def test_the_other_formats_carry_no_resampled_header(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result())
    for fmt in ("ang", "h5_light"):
        resp = _post({"format": fmt, "filename": str(tmp_path / f"o.{fmt}")})
        assert resp.status_code == 200, resp.text
        assert "x-orienta-resampled" not in resp.headers, fmt


def test_the_header_is_exposed_to_the_dev_server(monkeypatch, tmp_path):
    """A frontend on port 5173 reads a header of a cross-origin answer only if it is exposed."""
    _activate(monkeypatch, _make_result())
    resp = _post({"format": "ctf", "filename": str(tmp_path / "e.ctf")},
                 headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200, resp.text
    exposed = [h.strip().lower() for h in resp.headers["access-control-expose-headers"].split(",")]
    assert "x-orienta-resampled" in exposed
    assert CODE_HEADER.lower() in exposed                   # the existing entry stays


@pytest.mark.parametrize("fmt, tail", [("ctf", "indexing_result_spherical.ctf"),
                                       ("ang_square", "indexing_result_spherical_square.ang")])
def test_the_default_names(monkeypatch, fmt, tail):
    _activate(monkeypatch, _make_result())
    resp = _post({"format": fmt})
    assert resp.status_code == 200, resp.text
    assert tail in resp.headers["content-disposition"]


@pytest.mark.parametrize("fmt", ["ctf", "ang_square"])
def test_the_frame_warning_applies_as_for_ang(monkeypatch, tmp_path, fmt):
    from backend.api.services import reference_frame_state as rfs
    monkeypatch.setattr(rfs, "get_frame",
                        lambda *_a, **_k: rfs.normalise_spec({"apply_to_export": True}))
    _activate(monkeypatch, _make_result())
    resp = _post({"format": fmt, "filename": str(tmp_path / ("w.ctf" if fmt == "ctf" else "w.ang"))})
    assert resp.status_code == 200, resp.text
    warning = resp.headers["x-frame-warning"]
    assert warning.startswith(".ctf export" if fmt == "ctf" else ".ang export")
    assert "does not apply the coordinate-system rotation" in warning
    # and without the opt-in there is none
    monkeypatch.setattr(rfs, "get_frame", lambda *_a, **_k: {"apply_to_export": False})
    resp = _post({"format": fmt, "filename": str(tmp_path / ("w2.ctf" if fmt == "ctf" else "w2.ang"))})
    assert "x-frame-warning" not in resp.headers


def test_the_ang_frame_warning_text_is_unchanged(monkeypatch, tmp_path):
    from backend.api.services import reference_frame_state as rfs
    monkeypatch.setattr(rfs, "get_frame",
                        lambda *_a, **_k: rfs.normalise_spec({"apply_to_export": True}))
    _activate(monkeypatch, _make_result(geometry="square"))
    resp = _post({"format": "ang", "filename": str(tmp_path / "a.ang")})
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-frame-warning"] == (
        ".ang export does not apply the coordinate-system rotation; use Light .h5 for "
        "coordinate-system-correct orientations, or disable 'Apply to exports'.")


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

CODE = "export_needs_hex_scan"


@pytest.mark.parametrize("fmt", ["ctf", "ang_square"])
@pytest.mark.parametrize("kind", ["square stamp", "no stamp", "resampled from hex"])
def test_a_result_that_is_not_hexagonal_gets_a_coded_400(monkeypatch, tmp_path, fmt, kind):
    result = {"square stamp": lambda: _make_result(geometry="square"),
              "no stamp": lambda: _make_result(geometry=None),
              "resampled from hex": lambda: _make_result(geometry="hex", hex_resampled=True)}[kind]()
    _activate(monkeypatch, result)
    out = tmp_path / "never"
    resp = _post({"format": fmt, "filename": str(out)})
    assert resp.status_code == 400, resp.text
    assert resp.headers[CODE_HEADER.lower()] == CODE
    assert "hexagonal" in resp.json()["detail"] and "Nothing was written" in resp.json()["detail"]
    # a map resampled on load is refused too, so the text must not say it is not from a hex scan alone
    assert ("was already resampled to a square grid, or is not from a hexagonal scan"
            in resp.json()["detail"])
    assert not out.exists()


def test_an_unusable_stamp_is_refused_before_a_file_is_made(monkeypatch, tmp_path):
    result = _make_result()
    result.metadata["scan_geometry"] = ScanGeometry.hex(
        N_ROWS + 1, N_COLS, STEP_X, odd_row_shift=0.5).to_dict()      # a grid that is not the result's
    _activate(monkeypatch, result)
    out = tmp_path / "never.ctf"
    resp = _post({"format": "ctf", "filename": str(out)})
    assert resp.status_code == 400, resp.text
    assert "Nothing was written" in resp.json()["detail"]
    assert not out.exists()


@pytest.mark.parametrize("lang", ["en", "de", "ja", "zh"])
def test_the_refusal_code_has_a_text_in_every_language(lang):
    root = Path(__file__).resolve().parents[1]
    locale = json.loads((root / f"frontend/src/locales/{lang}/common.json").read_text(encoding="utf-8"))
    assert locale["errors"][CODE].strip(), lang


@pytest.mark.parametrize("lang, phrase", [
    ("en", "already resampled to a square grid, or is not from a hexagonal scan"),
    ("de", "bereits auf ein quadratisches Raster umgerechnet oder stammt nicht aus einem hexagonalen Scan"),
    ("ja", "すでに正方格子に再サンプリング済み"),
    ("zh", "已经重采样到方形网格"),
])
def test_the_refusal_text_names_both_causes(lang, phrase):
    root = Path(__file__).resolve().parents[1]
    locale = json.loads((root / f"frontend/src/locales/{lang}/common.json").read_text(encoding="utf-8"))
    assert phrase in locale["errors"][CODE], lang


@pytest.mark.parametrize("fmt, name", [("ctf", "never.ctf"), ("ang_square", "never.ang")])
def test_a_one_row_scan_is_refused_by_both(monkeypatch, tmp_path, fmt, name):
    """One row of 4 points is a 1 x 4 square grid. orix takes such a map for a one-dimensional one and
    keeps no y position, which neither a .ctf nor a .ang can state: a 400, not a 500, and no file."""
    _activate(monkeypatch, _make_result(n_rows=1, phase_of_point=[1, 2, 1, 1]))
    never = tmp_path / name
    resp = _post({"format": fmt, "filename": str(never)})
    assert resp.status_code == 400, resp.text
    assert "1 x 4" in resp.json()["detail"] and "Nothing was written" in resp.json()["detail"]
    assert not never.exists()


def test_the_unsupported_format_text_lists_every_format(monkeypatch):
    _activate(monkeypatch, _make_result())
    resp = _post({"format": "bogus"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    for fmt in ("ang", "ang_square", "ctf", "h5", "h5_light"):
        assert f"'{fmt}'" in detail, fmt


def test_the_request_comment_lists_every_format():
    src = (Path(__file__).resolve().parents[1] / "backend/api/routes/indexing.py").read_text(encoding="utf-8")
    block = src.split("class ExportRequest(BaseModel):", 1)[1].split("\n\n", 1)[0]
    for fmt in ('"ang"', '"ang_square"', '"ctf"', '"h5"', '"h5_light"'):
        assert fmt in block, (fmt, block)


# ---------------------------------------------------------------------------
# a square .ang stays what it was: the phase construction of build_export_xmap
# ---------------------------------------------------------------------------

def _square_phases():
    from crystal_symmetry import FramePhase
    return [
        Phase(name="cubic_pg", point_group="m-3m",
              structure=Structure(lattice=Lattice(4.049, 4.049, 4.049, 90, 90, 90))),
        Phase(name="cubic_sg", space_group=225,
              structure=Structure(lattice=Lattice(4.049, 4.049, 4.049, 90, 90, 90))),
        Phase(name="Ti", space_group=194,
              structure=Structure(lattice=Lattice(2.95, 2.95, 4.68, 90, 90, 120))),
        Phase(name="Hx", space_group=162,
              structure=Structure(lattice=Lattice(4.0, 4.0, 5.0, 90, 90, 120))),
        FramePhase(name="Fr", space_group=162,
                   structure=Structure(lattice=Lattice(4.0, 4.0, 5.0, 90, 90, 120))),
        Phase(name="mono", space_group=14,
              structure=Structure(lattice=Lattice(5.0, 6.0, 7.0, 90, 100, 90))),
        Phase(name="ortho", space_group=62,
              structure=Structure(lattice=Lattice(5.0, 6.0, 7.0, 90, 90, 90))),
        # the two trigonal groups with the three-fold axis and a mirror or a two-fold axis in the
        # other direction: P312 (point group 312) and P31m (31m); both are Laue class -3m, TSL 32
        Phase(name="P312", space_group=149,
              structure=Structure(lattice=Lattice(4.0, 4.0, 5.0, 90, 90, 120))),
        Phase(name="P31m", space_group=157,
              structure=Structure(lattice=Lattice(4.0, 4.0, 5.0, 90, 90, 120))),
    ]


def test_a_square_ang_is_byte_identical_to_the_old_phase_construction(tmp_path):
    """``build_export_xmap`` used to build ``Phase(name, point_group=<name>, structure)``; it now builds
    a ``FramePhase`` that also keeps the space group. The .ang a square result gives must not move: the
    file written from the new map equals, byte for byte, the file written from the same map with the
    phases built the old way, and the TSL codes of the phases are the same.

    The old construction is rebuilt in this test (the loop below), because the code that made it is
    gone: a ``Phase`` with the point group of the source phase by name and no space group."""
    phases = _square_phases()
    n = len(phases) * 2
    eulers = np.deg2rad(np.array([[10.0 + 7 * k, 20.0 + k, 30.0 + 2 * k] for k in range(n)]))
    xmap = CrystalMap(
        rotations=Rotation.from_euler(eulers), phase_id=np.repeat(np.arange(len(phases)), 2),
        x=(np.arange(n) % 2).astype(float), y=(np.arange(n) // 2).astype(float),
        phase_list=PhaseList(phases=phases))
    new, codes = ang_export.build_export_xmap(
        xmap, (len(phases), 2), np.ones(n, dtype=bool), 0.5, np.linspace(0.2, 0.9, n))

    old_phases = []
    for (_pid, source), (_nid, built) in zip(xmap.phases, new.phases):
        pg = source.point_group
        old_phases.append(Phase(name=source.name,
                                point_group=None if pg is None else getattr(pg, "name", str(pg)),
                                structure=built.structure))
    old = CrystalMap(rotations=new.rotations, phase_id=new.phase_id, x=new.x, y=new.y,
                     phase_list=PhaseList(old_phases), prop={k: v for k, v in new.prop.items()},
                     scan_unit="um")
    p_new, p_old = tmp_path / "new.ang", tmp_path / "old.ang"
    ang_export.write_ang(p_new, new, symmetry_codes=codes)
    ang_export.write_ang(p_old, old, symmetry_codes=codes)
    assert p_new.read_bytes() == p_old.read_bytes()

    # the codes the TSL header got are those of the source phases; the map's own phases give the same
    from backend.api.services.result_exporter import _tsl_codes_of_map
    assert _tsl_codes_of_map(new) == codes
    # by hand: m-3m 43, 6/mmm 62, -3m / 312 / 31m are Laue class -3m: 32, 2/m (b unique) 20, mmm 22
    assert codes == {1: "43", 2: "43", 3: "62", 4: "32", 5: "32", 6: "20", 7: "22", 8: "32", 9: "32"}


def test_the_new_phases_keep_the_space_group():
    phases = _square_phases()
    n = len(phases)
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.zeros((n, 3))), phase_id=np.arange(n),
        x=np.arange(n, dtype=float), y=np.zeros(n), phase_list=PhaseList(phases=phases))
    new, _codes = ang_export.build_export_xmap(
        xmap, (1, n), np.ones(n, dtype=bool), 0.5, None)
    numbers = {p.name: (None if p.space_group is None else int(p.space_group.number))
               for _i, p in new.phases}
    assert numbers == {"cubic_pg": None, "cubic_sg": 225, "Ti": 194, "Hx": 162, "Fr": 162,
                       "mono": 14, "ortho": 62, "P312": 149, "P31m": 157}
