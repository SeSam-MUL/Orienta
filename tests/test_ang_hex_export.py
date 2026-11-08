"""The ``.ang`` of a hexagonal scan: ``GRID: HexGrid``, every point of the index rectangle in
index order, x/y from the scan geometry, and no other column changed.

Hand values (3 rows, 4 columns, step_x 2.0, step_y = 2 * sqrt(3) / 2 = 1.7320508):

* odd rows +0.5 step_x: x = 0 2 4 6 | 1 3 5 7 | 0 2 4 6
* odd rows -0.5 step_x: the geometry's x is 0 2 4 6 | -1 1 3 5 | 0 2 4 6, the file lifts every x
  by half a step (1.0) so that none is negative: 1 3 5 7 | 0 2 4 6 | 1 3 5 7
* y = row * step_y = 0.00000 (x4), 1.73205 (x4), 3.46410 (x4)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from backend.api.main import app
from backend.api.routes import indexing as indexing_mod
from backend.api.services import ang_export
from backend.api.services.scan_geometry import ScanGeometry
from indexing_controller import IndexingMethod, IndexingResult

N_ROWS, N_COLS = 3, 4
STEP_X = 2.0
SQRT3 = math.sqrt(3.0)          # step_y = STEP_X * sqrt(3) / 2 = sqrt(3)

X_PLUS = ["0.00000", "2.00000", "4.00000", "6.00000",
          "1.00000", "3.00000", "5.00000", "7.00000",
          "0.00000", "2.00000", "4.00000", "6.00000"]
X_MINUS = ["1.00000", "3.00000", "5.00000", "7.00000",
           "0.00000", "2.00000", "4.00000", "6.00000",
           "1.00000", "3.00000", "5.00000", "7.00000"]
Y_HEX = ["0.00000"] * 4 + ["1.73205"] * 4 + ["3.46410"] * 4


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


def _hex(shift=0.5, confirmed=False, rows=N_ROWS, cols=N_COLS):
    return ScanGeometry.hex(rows, cols, STEP_X, odd_row_shift=shift, shift_confirmed=confirmed)


def _make_result(geometry=None, *, mask=None, hex_resampled=None, step_um=STEP_X, scan_offset=None,
                 hex_phase=False, shape=(N_ROWS, N_COLS), hex_space_group=162):
    """A spherical-style result on an index-space rectangle (pixel coordinates, 1-based phase ids).
    ``geometry`` is stamped on it the way a run stamps it; ``mask`` selects the indexed points."""
    from diffpy.structure import Lattice, Structure
    n = shape[0] * shape[1]
    sel = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).ravel()
    n_pts = int(sel.sum())
    eulers = np.deg2rad(np.array([[10.0 + i * 11.0, 20.0 + i, 30.0 + 2 * i] for i in range(n_pts)]))
    al = Phase(name="Al", point_group="m-3m",
               structure=Structure(lattice=Lattice(4.049, 4.049, 4.049, 90, 90, 90)))
    if hex_phase:
        other = Phase(name="Hx", space_group=hex_space_group,
                      structure=Structure(lattice=Lattice(4.0, 4.0, 5.0, 90, 90, 120)))
    else:
        other = Phase(name="Fe", point_group="m-3m",
                      structure=Structure(lattice=Lattice(2.87, 2.87, 2.87, 90, 90, 90)))
    pid = np.where(np.arange(n_pts) % 3 == 0, 2, 1)
    idx = np.flatnonzero(sel)
    xs = (idx % shape[1]).astype(float)                    # pixels, as the real map carries
    ys = (idx // shape[1]).astype(float)
    xmap = CrystalMap(rotations=Rotation.from_euler(eulers), phase_id=pid, x=xs, y=ys,
                      phase_list=PhaseList(phases=[al, other], ids=[1, 2]))
    md = {"step_size_um": step_um, "source_vendor": "oxford"}
    if geometry is not None:
        md["scan_geometry"] = geometry.to_dict()
    if hex_resampled is not None:
        md["hex_resampled"] = hex_resampled
    if scan_offset is not None:
        md.update({"scan_row_offset": scan_offset[0], "scan_col_offset": scan_offset[1],
                   "scan_shape": [90, 120]})
    return IndexingResult(
        xmap=xmap, selection_mask=sel, original_shape=shape, method=IndexingMethod.SPHERICAL,
        confidence_scores=np.linspace(0.3, 0.9, n_pts).astype(np.float32), metadata=md)


def _activate(monkeypatch, result):
    rid = "test_ang_hex_export"
    monkeypatch.setitem(indexing_mod._result_registry, rid, result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", rid)
    monkeypatch.setattr("backend.api.routes.ebsd_viewer._ebsd_file_path", None, raising=False)


def _export(out, fmt="ang"):
    return TestClient(app).post("/api/indexing/export", json={"format": fmt, "filename": str(out)})


def _lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def _header(path):
    return [ln for ln in _lines(path) if ln.startswith("#")]


def _rows(path):
    """The data lines as lists of the text tokens (no float conversion)."""
    return [ln.split() for ln in _lines(path) if ln.strip() and not ln.startswith("#")]


def _col(rows, i):
    return [r[i] for r in rows]


def _kv(header, key):
    pat = re.compile(r"^#\s*" + re.escape(key) + r"\s*:\s*(.*)$")
    return [m.group(1) for ln in header if (m := pat.match(ln))]


def _field(header, key):
    pat = re.compile(r"^#\s*" + re.escape(key) + r"\s+(.*)$")
    return [m.group(1).strip() for ln in header if (m := pat.match(ln))]


def _export_ok(monkeypatch, tmp_path, result, name="r.ang"):
    _activate(monkeypatch, result)
    out = tmp_path / name
    r = _export(out)
    assert r.status_code == 200, r.text
    return out


# ---------------------------------------------------------------------------
# positions and header, by hand
# ---------------------------------------------------------------------------

def test_hex_positions_by_hand_plus_half(monkeypatch, tmp_path):
    out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(0.5)))
    hdr, rows = _header(out), _rows(out)
    assert _field(hdr, "GRID:") == ["HexGrid"]
    assert _field(hdr, "XSTEP:") == ["2.000000"]
    assert _field(hdr, "YSTEP:") == ["1.732051"]
    assert (_field(hdr, "NCOLS_ODD:"), _field(hdr, "NCOLS_EVEN:"), _field(hdr, "NROWS:")) == (
        ["4"], ["4"], ["3"])
    assert len(rows) == 12                                    # the whole rectangle, in index order
    assert _col(rows, 3) == X_PLUS
    assert _col(rows, 4) == Y_HEX


def test_hex_positions_by_hand_minus_half(monkeypatch, tmp_path):
    out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(-0.5)))
    rows = _rows(out)
    assert len(rows) == 12
    assert _col(rows, 3) == X_MINUS                           # lifted by half a step: none negative
    assert _col(rows, 4) == Y_HEX
    assert _field(_header(out), "GRID:") == ["HexGrid"]


# ---------------------------------------------------------------------------
# nothing else changes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [0.5, -0.5])
def test_hex_other_columns_equal_square_output(monkeypatch, tmp_path, shift):
    """The same result, once stamped hexagonal and once not stamped: columns 0-2 (Euler angles) and
    5-9 (IQ, CI, phase, detector signal, fit) are the same text; so is every header line but the
    grid, the y step and the lines the hexagonal form adds. Three of twelve points are not indexed,
    so the values orix writes for those are covered too."""
    mask = np.ones(N_ROWS * N_COLS, dtype=bool)
    mask[[1, 6, 11]] = False
    hex_out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(shift), mask=mask), "hex.ang")
    sq_out = _export_ok(monkeypatch, tmp_path, _make_result(None, mask=mask), "sq.ang")

    h_rows, s_rows = _rows(hex_out), _rows(sq_out)
    assert len(h_rows) == len(s_rows) == 12
    for c in (0, 1, 2, 5, 6, 7, 8, 9):
        assert _col(h_rows, c) == _col(s_rows, c), f"column {c} differs"
    # the unindexed points are in the file at their place, with orix's values
    assert _col(h_rows, 0)[1] == "12.56637"

    h_hdr, s_hdr = _header(hex_out), _header(sq_out)
    only_hex = [ln for ln in h_hdr if ln not in s_hdr]
    only_sq = [ln for ln in s_hdr if ln not in h_hdr]
    assert sorted(ln.split(":")[0] for ln in only_sq) == ["# GRID", "# YSTEP"]
    assert sorted({ln.split(":")[0] for ln in only_hex}) == [
        "# GRID", "# ORIENTA_NOTE", "# ORIENTA_SCAN_GRID", "# YSTEP"]
    assert [ln for ln in h_hdr if ln in s_hdr] == [ln for ln in s_hdr if ln in h_hdr]   # same order


def test_hex_header_keys(monkeypatch, tmp_path):
    out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(0.5, confirmed=False)))
    hdr = _header(out)
    assert _kv(hdr, "ORIENTA_SCAN_GRID") == [
        "HexGrid; rows=3; columns=4 in every row; odd rows at +0.5 step_x; direction=assumed"]
    notes = " ".join(_kv(hdr, "ORIENTA_NOTE"))
    assert "equal row length" in notes and "one point less in every second row" in notes
    assert "direction of the row offset is an assumption" in notes
    assert "raised by half a step" not in notes               # only for -0.5

    out2 = _export_ok(monkeypatch, tmp_path, _make_result(_hex(-0.5, confirmed=True)), "m.ang")
    hdr2 = _header(out2)
    assert _kv(hdr2, "ORIENTA_SCAN_GRID") == [
        "HexGrid; rows=3; columns=4 in every row; odd rows at -0.5 step_x; direction=confirmed"]
    notes2 = " ".join(_kv(hdr2, "ORIENTA_NOTE"))
    assert "x is raised by half a step" in notes2
    assert "direction of the row offset is an assumption" not in notes2
    assert "equal row length" in notes2


def test_hex_roi_keeps_rectangle(monkeypatch, tmp_path):
    """An ROI is placed on the rectangle as for a square scan: 12 lines, positions of the whole
    rectangle, the points outside the ROI as orix writes them."""
    mask = np.zeros(N_ROWS * N_COLS, dtype=bool)
    mask[[0, 2, 3, 5, 8]] = True
    out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(0.5), mask=mask))
    rows = _rows(out)
    assert len(rows) == 12
    assert _col(rows, 3) == X_PLUS and _col(rows, 4) == Y_HEX
    assert [r[0] == "12.56637" for r in rows] == [not m for m in mask]


# ---------------------------------------------------------------------------
# where the geometry comes from
# ---------------------------------------------------------------------------

def test_hex_stamp_beats_live_dataset(monkeypatch, tmp_path):
    """The geometry is the result's own stamp. A square dataset active now does not turn a hexagonal
    result into a square file, and a hexagonal dataset active now does not turn a square one hexagonal."""
    from backend.api.routes import ebsd_viewer
    from backend.api.services import scan_geometry
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "live_dataset", raising=False)
    try:
        scan_geometry.set_geometry("live_dataset", ScanGeometry.square(N_ROWS, N_COLS, STEP_X))
        out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(0.5)), "a.ang")
        assert _field(_header(out), "GRID:") == ["HexGrid"]
        assert _col(_rows(out), 3) == X_PLUS

        scan_geometry.set_geometry("live_dataset", _hex(0.5))
        out2 = _export_ok(monkeypatch, tmp_path, _make_result(None), "b.ang")
        assert _field(_header(out2), "GRID:") == ["SqrGrid"]
        assert _col(_rows(out2), 4) == ["0.00000"] * 4 + ["2.00000"] * 4 + ["4.00000"] * 4
    finally:
        scan_geometry.clear("live_dataset")


def test_hex_refused_on_shape_or_step_mismatch(monkeypatch, tmp_path):
    # the stamp is 3 x 4, the result's grid is 4 x 3
    _activate(monkeypatch, _make_result(_hex(0.5), shape=(4, 3)))
    out = tmp_path / "shape.ang"
    r = _export(out)
    assert r.status_code == 400, r.text
    assert "3 x 4" in r.json()["detail"] or "(3, 4)" in r.json()["detail"]
    assert not out.exists()

    # the step the result states is not the stamp's
    _activate(monkeypatch, _make_result(_hex(0.5), step_um=2.5))
    out2 = tmp_path / "step.ang"
    r2 = _export(out2)
    assert r2.status_code == 400, r2.text
    assert "step" in r2.json()["detail"].lower()
    assert not out2.exists()

    # within 1e-4 relative it is the same step
    _activate(monkeypatch, _make_result(_hex(0.5), step_um=STEP_X * (1 + 5e-5)))
    assert _export(tmp_path / "ok.ang").status_code == 200


def test_hex_geometry_for_export_by_case():
    f = ang_export.hex_geometry_for_export
    shape = (N_ROWS, N_COLS)
    assert f(None, shape, STEP_X) is None                                    # no stamp
    assert f(ScanGeometry.square(N_ROWS, N_COLS, STEP_X).to_dict(), shape, STEP_X) is None
    assert f(_hex().to_dict(), shape, STEP_X, hex_resampled=True) is None   # resampled to square
    g = f(_hex(-0.5, True).to_dict(), shape, STEP_X)
    assert (g.kind, g.odd_row_shift, g.shift_confirmed, g.grid_shape) == ("hex", -0.5, True, shape)
    assert f(_hex().to_dict(), shape, None).kind == "hex"                   # no step stated: the stamp's
    with pytest.raises(ang_export.ExportRefused):
        f(_hex().to_dict(), (4, 3), STEP_X)
    with pytest.raises(ang_export.ExportRefused):
        f(_hex().to_dict(), shape, 2.001)
    with pytest.raises(ang_export.ExportRefused):                           # one column: orix makes it 1-D
        f(_hex(rows=3, cols=1).to_dict(), (3, 1), STEP_X)
    masked = {**_hex().to_dict(), "has_valid_mask": True}
    with pytest.raises(ang_export.ExportRefused):
        f(masked, shape, STEP_X)


# ---------------------------------------------------------------------------
# square stays square
# ---------------------------------------------------------------------------

def test_square_and_unstamped_unchanged(monkeypatch, tmp_path):
    plain = _export_ok(monkeypatch, tmp_path, _make_result(None), "plain.ang")
    sq = _export_ok(monkeypatch, tmp_path,
                    _make_result(ScanGeometry.square(N_ROWS, N_COLS, STEP_X)), "sq.ang")
    resampled = _export_ok(monkeypatch, tmp_path,
                           _make_result(_hex(0.5), hex_resampled=True), "res.ang")
    assert plain.read_bytes() == sq.read_bytes() == resampled.read_bytes()
    hdr = _header(plain)
    assert _field(hdr, "GRID:") == ["SqrGrid"]
    assert _field(hdr, "YSTEP:") == ["2.000000"]
    assert _kv(hdr, "ORIENTA_SCAN_GRID") == []
    rows = _rows(plain)
    assert _col(rows, 3) == ["0.00000", "2.00000", "4.00000", "6.00000"] * 3
    assert _col(rows, 4) == ["0.00000"] * 4 + ["2.00000"] * 4 + ["4.00000"] * 4


# ---------------------------------------------------------------------------
# orix is not changed, and a changed orix is noticed
# ---------------------------------------------------------------------------

def test_orix_module_untouched(monkeypatch, tmp_path):
    """The module-level ``create_coordinate_arrays`` and ``file_writer`` of orix are the same objects
    during and after a hexagonal write: the writer is a private copy, nothing is patched."""
    from orix.io.plugins import ang as orix_ang
    before = (orix_ang.create_coordinate_arrays, orix_ang.file_writer)
    seen = []
    real_savetxt = np.savetxt

    def spy(*a, **k):
        seen.append((orix_ang.create_coordinate_arrays is before[0], orix_ang.file_writer is before[1]))
        return real_savetxt(*a, **k)

    monkeypatch.setattr(np, "savetxt", spy)
    _export_ok(monkeypatch, tmp_path, _make_result(_hex(0.5)))
    assert seen == [(True, True)]                                # during the write
    assert (orix_ang.create_coordinate_arrays, orix_ang.file_writer) == before
    # and a square write after it still uses the original positions
    out = _export_ok(monkeypatch, tmp_path, _make_result(None), "sq.ang")
    assert _field(_header(out), "GRID:") == ["SqrGrid"]


def _xmap_for_seam(n_rows, n_cols):
    n = n_rows * n_cols
    return CrystalMap(
        rotations=Rotation.from_euler(np.zeros((n, 3))), phase_id=np.zeros(n, dtype=int),
        x=np.tile(np.arange(n_cols, dtype=float), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=float), n_cols),
        phase_list=PhaseList(phases=[Phase(name="Al", point_group="m-3m")]))


def test_seam_breaks_loudly(monkeypatch, tmp_path):
    from orix.io.plugins import ang as orix_ang
    x, y = _hex(0.5).positions()
    # works against the installed orix
    out = tmp_path / "ok.ang"
    ang_export.orix_ang_writer_with(x, y)(str(out), _xmap_for_seam(N_ROWS, N_COLS))
    assert _col(_rows(out), 3) == X_PLUS

    # a map of another shape than the geometry's: refused, not written with wrong positions
    with pytest.raises(ang_export.ExportRefused):
        ang_export.orix_ang_writer_with(x, y)(str(tmp_path / "shape.ang"), _xmap_for_seam(4, 3))
    # one column: orix takes the map for a 1-D one and cannot write a grid
    x1, y1 = ScanGeometry.hex(3, 1, STEP_X).positions()
    with pytest.raises(ang_export.ExportRefused):
        ang_export.orix_ang_writer_with(x1, y1)

    # an orix whose writer no longer asks the function: the file would carry square positions
    def no_seam(filename, xmap, **kw):
        return None

    monkeypatch.setattr(orix_ang, "file_writer", no_seam)
    with pytest.raises(ang_export.ExportRefused):
        ang_export.orix_ang_writer_with(x, y)


def _asks_twice(filename, xmap, **kw):
    # `create_coordinate_arrays` is a global of orix's module; the private copy supplies it
    create_coordinate_arrays((N_ROWS, N_COLS), (1.0, 1.0))  # noqa: F821
    create_coordinate_arrays((N_ROWS, N_COLS), (1.0, 1.0))  # noqa: F821


def _never_asks(filename, xmap, **kw):
    # names the function, but a path that does not run: the writer would use its own positions
    if kw.get("positions_from_the_seam"):
        create_coordinate_arrays((N_ROWS, N_COLS), (1.0, 1.0))  # noqa: F821
    Path(filename).write_text("# not a file of the scan\n")


def test_seam_never_asked_breaks_loudly(monkeypatch, tmp_path):
    """An orix whose writer mentions the function but does not call it: the file would carry the
    square positions of orix, so the write is refused (and, through ``write_ang``, removed)."""
    from orix.io.plugins import ang as orix_ang
    monkeypatch.setattr(orix_ang, "file_writer", _never_asks)
    x, y = _hex(0.5).positions()
    with pytest.raises(ang_export.ExportRefused):
        ang_export.orix_ang_writer_with(x, y)(str(tmp_path / "raw.ang"),
                                              _xmap_for_seam(N_ROWS, N_COLS))
    out = tmp_path / "never.ang"
    with pytest.raises(ang_export.ExportRefused):
        ang_export.write_ang(out, _xmap_for_seam(N_ROWS, N_COLS), geometry=_hex(0.5))
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw.ang"]   # no temp file behind


def test_seam_called_twice_breaks_loudly(monkeypatch, tmp_path):
    """An orix whose writer asks for the coordinates twice: refused on the second call."""
    from orix.io.plugins import ang as orix_ang
    monkeypatch.setattr(orix_ang, "file_writer", _asks_twice)
    x, y = _hex(0.5).positions()
    with pytest.raises(ang_export.ExportRefused):
        ang_export.orix_ang_writer_with(x, y)(str(tmp_path / "twice.ang"),
                                              _xmap_for_seam(N_ROWS, N_COLS))


OLD_BYTES = b"# an earlier export\r\n0.1 0.2 0.3\n\xc3\xa4\n"


def _only(tmp_path, names):
    return sorted(p.name for p in tmp_path.iterdir()) == sorted(names)


def test_a_refused_write_keeps_the_existing_file(monkeypatch, tmp_path):
    """Every refusal says "Nothing was written": the file that was there stays, byte for byte,
    and no temporary file stays behind."""
    from orix.io.plugins import ang as orix_ang
    out = tmp_path / "keep.ang"
    out.write_bytes(OLD_BYTES)
    # 3 x 4 geometry, 4 x 3 map: refused inside the writer
    with pytest.raises(ang_export.ExportRefused):
        ang_export.write_ang(out, _xmap_for_seam(4, 3), geometry=_hex(0.5))
    assert out.read_bytes() == OLD_BYTES and _only(tmp_path, ["keep.ang"])
    # a writer that never asks for the coordinates: it has written a file, then it is refused
    monkeypatch.setattr(orix_ang, "file_writer", _never_asks)
    with pytest.raises(ang_export.ExportRefused):
        ang_export.write_ang(out, _xmap_for_seam(N_ROWS, N_COLS), geometry=_hex(0.5))
    assert out.read_bytes() == OLD_BYTES and _only(tmp_path, ["keep.ang"])
    # and without an earlier file nothing appears
    gone = tmp_path / "new.ang"
    with pytest.raises(ang_export.ExportRefused):
        ang_export.write_ang(gone, _xmap_for_seam(4, 3), geometry=_hex(0.5))
    assert _only(tmp_path, ["keep.ang"])


def test_a_header_without_the_grid_line_is_refused(monkeypatch, tmp_path):
    """If orix words its ``# GRID:`` line differently, the replacement finds nothing: the file
    would hold hexagonal positions under a square header. It is refused, not written."""
    monkeypatch.setattr(ang_export, "HEX_GRID_REPLACE", {"# GRID: SquareGridX": "# GRID: HexGrid"})
    out = tmp_path / "grid.ang"
    out.write_bytes(OLD_BYTES)
    with pytest.raises(ang_export.ExportRefused, match="grid line"):
        ang_export.write_ang(out, _xmap_for_seam(N_ROWS, N_COLS), geometry=_hex(0.5))
    assert out.read_bytes() == OLD_BYTES and _only(tmp_path, ["grid.ang"])
    fresh = tmp_path / "fresh.ang"
    with pytest.raises(ang_export.ExportRefused):
        ang_export.write_ang(fresh, _xmap_for_seam(N_ROWS, N_COLS), geometry=_hex(0.5))
    assert _only(tmp_path, ["grid.ang"])


def test_a_hex_scan_of_one_row_is_refused(monkeypatch, tmp_path):
    """orix takes a one-row map for a 1-D one and takes YSTEP from its ``dy``: refused, as for
    one column."""
    one_row = _hex(0.5, rows=1)
    f = ang_export.hex_geometry_for_export
    with pytest.raises(ang_export.ExportRefused, match="row"):
        f(one_row.to_dict(), (1, N_COLS), STEP_X)
    x, y = one_row.positions()
    with pytest.raises(ang_export.ExportRefused, match="row"):
        ang_export.orix_ang_writer_with(x, y)
    _activate(monkeypatch, _make_result(one_row, shape=(1, N_COLS)))
    out = tmp_path / "row.ang"
    r = _export(out)
    assert r.status_code == 400, r.text
    assert "row" in r.json()["detail"]
    assert not out.exists()


def test_the_hex_path_rejects_a_name_that_is_not_ang(tmp_path):
    """The square path raises IOError (orix.io.save) for a name no writer knows; so does this."""
    bad = tmp_path / "scan.txt"
    with pytest.raises(IOError):
        ang_export.write_ang(bad, _xmap_for_seam(N_ROWS, N_COLS), geometry=_hex(0.5))
    with pytest.raises(IOError):
        ang_export.save_hex_ang(bad, _xmap_for_seam(N_ROWS, N_COLS), _hex(0.5))
    assert list(tmp_path.iterdir()) == []
    # the extension is read as orix reads it (any case); the temporary name does not trip it
    upper = tmp_path / "SCAN.ANG"
    ang_export.write_ang(upper, _xmap_for_seam(N_ROWS, N_COLS), geometry=_hex(0.5))
    assert _only(tmp_path, ["SCAN.ANG"]) and len(_rows(upper)) == 12


def test_write_ang_replaces_an_existing_hex_file(tmp_path):
    out = tmp_path / "again.ang"
    for _ in range(2):
        ang_export.write_ang(out, _xmap_for_seam(N_ROWS, N_COLS), geometry=_hex(0.5))
    assert len(_rows(out)) == 12 and _field(_header(out), "GRID:") == ["HexGrid"]


# ---------------------------------------------------------------------------
# the origin of a crop
# ---------------------------------------------------------------------------

def test_origin_text_hex_uses_step_y():
    # crop at row 3, column 2 of a 90 x 120 scan; step_x 2.0, step_y sqrt(3):
    # x = 2 * 2.0 = 4.000, y = 3 * sqrt(3) = 5.196 (not 3 * 2.0 = 6.000)
    text = ang_export.scan_origin_text(3, 2, (90, 120), STEP_X, step_y_um=SQRT3)
    assert "x 4.000 um, y 5.196 um" in text
    # without a y step it is the square rule, as before
    assert "x 4.000 um, y 6.000 um" in ang_export.scan_origin_text(3, 2, (90, 120), STEP_X)


def test_hex_crop_states_its_origin_with_step_y(monkeypatch, tmp_path):
    out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(-0.5), scan_offset=(3, 2)))
    line = " ".join(_kv(_header(out), "ORIENTA_SCAN_ORIGIN"))
    assert "row 3, column 2" in line and "x 4.000 um, y 5.196 um" in line


# ---------------------------------------------------------------------------
# the batch: a checkpoint that states a hexagonal grid
# ---------------------------------------------------------------------------

def _checkpoint(tmp_path, geometry, *, hex_phase_attrs):
    from backend.api.services.checkpoint_writer import CheckpointWriter
    source = tmp_path / "FakeSample.h5oina"
    with h5py.File(source, "w") as f:
        f.create_group("1/EBSD/Data").create_dataset(
            "Band Contrast", data=np.arange(10, 22, dtype=np.uint8))
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("Beam Voltage", data=np.float32(20.0))
        hdr.create_dataset("Tilt Angle", data=np.float32(70.0))
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="hex-test", method="spherical",
                     step_size_um=STEP_X, geometry=geometry)
    rng = np.random.default_rng(0)
    ci_a = np.full((3, 4), 0.4, dtype=np.float32)
    ci_b = np.full((3, 4), 0.4, dtype=np.float32)
    ci_a[:2] = 0.8
    ci_b[2:] = 0.8                                   # the second phase wins the bottom row
    cw.write_phase_result(
        "Al", ci_a, rng.uniform(0, 0.1, size=(3, 4, 3)).astype(np.float32),
        {"phase_file": "Al.cif", "ci_mean": 0.75, "duration_sec": 1.0, "space_group": 225,
         "point_group": "m-3m", "lattice_constants": [4.049, 4.049, 4.049, 90.0, 90.0, 90.0]})
    cw.write_phase_result(
        "Hx", ci_b, rng.uniform(0.4, 0.5, size=(3, 4, 3)).astype(np.float32),
        {"phase_file": "Hx.cif", "ci_mean": 0.45, "duration_sec": 1.0,
         "lattice_constants": [4.0, 4.0, 5.0, 90.0, 90.0, 120.0], **hex_phase_attrs})
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


SG162 = {"space_group": 162, "point_group": "-3m"}


def _symmetry_of(path, name):
    """``# Symmetry`` value of the phase block whose MaterialName is ``name``."""
    cur = {}
    blocks = []
    for ln in _header(path):
        m = re.match(r"^#\s*Phase\s+(\d+)\s*$", ln)
        if m:
            cur = {}
            blocks.append(cur)
        m = re.match(r"^#\s*(MaterialName|Symmetry)\s+(.*)$", ln)
        if m and blocks:
            blocks[-1][m.group(1)] = m.group(2).strip()
    return [b["Symmetry"] for b in blocks if b.get("MaterialName") == name]


def test_hex_checkpoint_exports_hex(tmp_path):
    from backend.api.services.result_exporter import export_ang_ctf
    source, cw = _checkpoint(tmp_path, _hex(0.5, confirmed=True), hex_phase_attrs=SG162)
    out = tmp_path / "out"
    ang, _ctf = export_ang_ctf(cw.checkpoint_path, str(out), write_ctf=False,
                               source_h5_path=str(source))
    assert ang, "the hexagonal checkpoint was not exported"
    hdr, rows = _header(ang), _rows(ang)
    assert _field(hdr, "GRID:") == ["HexGrid"]
    assert _field(hdr, "XSTEP:") == ["2.000000"] and _field(hdr, "YSTEP:") == ["1.732051"]
    assert (_field(hdr, "NCOLS_ODD:"), _field(hdr, "NCOLS_EVEN:"), _field(hdr, "NROWS:")) == (
        ["4"], ["4"], ["3"])
    assert len(rows) == 12
    assert _col(rows, 3) == X_PLUS and _col(rows, 4) == Y_HEX
    assert _kv(hdr, "ORIENTA_SCAN_GRID") == [
        "HexGrid; rows=3; columns=4 in every row; odd rows at +0.5 step_x; direction=confirmed"]
    # space group 162 is TSL symmetry 32 on this path as on the route
    assert _symmetry_of(ang, "Hx") == ["32"]


def test_hex_checkpoint_refuses_an_explicit_step_of_another_grid(tmp_path):
    """The checkpoint states step_x = 2.0 um. A step given on top of it is allowed to differ by
    1e-4 relative (2.0 * 1e-4 = 0.0002 um): 2.0001 is the same grid, 2.0003 and 2.5 are not."""
    from backend.api.services.result_exporter import export_ang_ctf
    source, cw = _checkpoint(tmp_path, _hex(0.5, confirmed=True), hex_phase_attrs=SG162)
    for k, step in enumerate((2.5, 2.0003)):
        out = tmp_path / f"refused{k}"
        with pytest.raises(ValueError, match="step_x"):
            export_ang_ctf(cw.checkpoint_path, str(out), write_ctf=False,
                           source_h5_path=str(source), step_size=step)
        assert not out.exists()                                         # refused before any write
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "same"), write_ctf=False,
                            source_h5_path=str(source), step_size=2.0001)
    assert ang and _field(_header(ang), "XSTEP:") == ["2.000000"]       # the checkpoint's step


def test_hex_checkpoint_minus_half_and_legacy_square(tmp_path):
    from backend.api.services.result_exporter import export_ang_ctf
    (tmp_path / "a").mkdir()
    source, cw = _checkpoint(tmp_path / "a", _hex(-0.5), hex_phase_attrs=SG162)
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "outa"), write_ctf=False,
                            source_h5_path=str(source))
    assert _col(_rows(ang), 3) == X_MINUS and _col(_rows(ang), 4) == Y_HEX

    (tmp_path / "b").mkdir()
    source_b, cw_b = _checkpoint(tmp_path / "b", None, hex_phase_attrs=SG162)   # a legacy checkpoint
    ang_b, _ = export_ang_ctf(cw_b.checkpoint_path, str(tmp_path / "outb"), write_ctf=False,
                              source_h5_path=str(source_b))
    hdr = _header(ang_b)
    assert _field(hdr, "GRID:") == ["SqrGrid"] and _field(hdr, "YSTEP:") == ["2.000000"]
    assert _kv(hdr, "ORIENTA_SCAN_GRID") == []
    assert _col(_rows(ang_b), 3) == ["0.00000", "2.00000", "4.00000", "6.00000"] * 3


def test_hex_checkpoint_symmetry_is_the_tsl_code(tmp_path):
    """A point group recorded as ``312`` is written by orix as ``312``, which no TSL software reads;
    on the hexagonal path it becomes 32, the code the route writes for the same phase."""
    from backend.api.services.result_exporter import export_ang_ctf
    source, cw = _checkpoint(tmp_path, _hex(0.5), hex_phase_attrs={"point_group": "312"})
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "out"), write_ctf=False,
                            source_h5_path=str(source))
    assert _symmetry_of(ang, "Hx") == ["32"]
    assert _symmetry_of(ang, "Al") == ["43"]


def test_route_hex_ang_states_the_crystal_frame_and_the_y_setting(monkeypatch, tmp_path):
    out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(0.5), hex_phase=True))
    lines = [ln for ln in _header(out) if ln.startswith("# ORIENTA_CRYSTAL_FRAME: ")]
    assert len(lines) == 2 and "X||a" in lines[0]
    assert "Phase Hx: space group 162, point group -31m" in lines[1] and "along Y" in lines[1]
    # a hexagonal file of cubic phases states nothing
    cubic = _export_ok(monkeypatch, tmp_path, _make_result(_hex(0.5)), name="cubic.ang")
    assert not [ln for ln in _header(cubic) if "CRYSTAL_FRAME" in ln]


@pytest.mark.parametrize("space_group, point_group", [(162, "-3m"), (149, "312")])
def test_orix_loads_the_hex_ang_of_a_phase_with_the_y_setting(monkeypatch, tmp_path, space_group,
                                                              point_group):
    """Route and batch: a comment that holds orix' own header patterns is read as a phase line."""
    from orix.io import load
    from backend.api.services.result_exporter import export_ang_ctf
    result = _make_result(_hex(0.5), hex_phase=True, hex_space_group=space_group)
    out = _export_ok(monkeypatch, tmp_path, result)
    assert len([ln for ln in _header(out) if ln.startswith("# ORIENTA_CRYSTAL_FRAME: ")]) == 2
    assert load(str(out)).size == 12
    (tmp_path / "b").mkdir()
    source, cw = _checkpoint(tmp_path / "b", _hex(0.5, confirmed=True),
                             hex_phase_attrs={"space_group": space_group, "point_group": point_group})
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), write_ctf=False,
                            source_h5_path=str(source))
    assert len([ln for ln in _header(ang) if ln.startswith("# ORIENTA_CRYSTAL_FRAME: ")]) == 2
    assert load(ang).size == 12


def test_hex_checkpoint_states_the_crystal_frame(tmp_path):
    from backend.api.services.result_exporter import export_ang_ctf
    source, cw = _checkpoint(tmp_path, _hex(0.5, confirmed=True), hex_phase_attrs=SG162)
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "out"), write_ctf=False,
                            source_h5_path=str(source))
    lines = [ln for ln in _header(ang) if ln.startswith("# ORIENTA_CRYSTAL_FRAME: ")]
    assert len(lines) == 2 and "space group 162, point group -31m" in lines[1]


def test_route_symmetry_of_space_group_162_is_32(monkeypatch, tmp_path):
    out = _export_ok(monkeypatch, tmp_path, _make_result(_hex(0.5), hex_phase=True))
    assert _symmetry_of(out, "Hx") == ["32"]
