"""The batch page pairs a hexagonal .up1/.up2 file with its APEX measurement and says so.

A pattern file stores its patterns in acquisition order with no positions; the grid lives in the APEX
metadata ``.h5`` of the measurement. The batch loads a file by path, and the loader finds the measurement
by the GUID in the file's header among the ``.h5`` files of the same folder, never by name. The page
shows the pairing when the file is added; a hexagonal file without metadata is refused by code right
then, and an APEX metadata file (which holds no patterns) is neither listed nor taken.

Real files made by ``tests/apex_helpers.py``, the real loader and the real routes. Only the indexer is a
recording stand-in. Expected values come from the numbers written into the helper files, by hand:

* one session ``.h5`` holds A (GUID_A, hex 3 x 5, step 0.5) and B (GUID_B, hex 3 x 5, step 0.8);
* ``a.up1`` carries GUID_B and ``b.up1`` carries GUID_A (the crossed names are the point);
* a.up1 on measurement B: odd rows shifted +0.5 step, so row 1 column 0 is at x = 0.5 * 0.8 = 0.4 and row 2
  is at y = 2 * 0.8 * sqrt(3) / 2 = 1.3856406;
* b.up1 on measurement A: x = 0.5 * 0.5 = 0.25 and y = 2 * 0.5 * sqrt(3) / 2 = 0.8660254.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import batch_v2 as B
from backend.api.services import batch_manager as M
from backend.api.services.calibration_store import CalibrationStore
from backend.api.services.checkpoint_writer import read_checkpoint_geometry
from tests.apex_helpers import (GUID_A, GUID_A_BYTES, GUID_B, GUID_B_BYTES, make_apex_h5, make_osc, make_up,
                                measurement_spec)
from tests.test_batch_export_both_grids import _header, _rows
from tests.test_batch_hex_run import _Indexer

N_ROWS, N_COLS = 3, 5


@pytest.fixture
def store(monkeypatch):
    s = CalibrationStore()
    monkeypatch.setattr(B, "calibration_store", s)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", s)
    B._quick_loaded.clear()
    yield s
    B._quick_loaded.clear()


def _two_measurements(folder: Path) -> None:
    """The session file with A and B, and the two pattern files that name them crosswise."""
    make_apex_h5(folder / "session.h5", [
        measurement_spec(id=GUID_A, name="OIM Map 1", n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5,
                         pc=(0.5, 0.4, 0.6)),
        measurement_spec(id=GUID_B, name="OIM Map 2", n_rows=N_ROWS, n_cols=N_COLS, step_x=0.8,
                         pc=(0.45, 0.35, 0.55)),
    ])
    make_up(folder / "a.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45, guid_bytes=GUID_B_BYTES)
    make_up(folder / "b.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45, guid_bytes=GUID_A_BYTES)


def _post(path):
    return TestClient(app).post("/api/batch-v2/quick-load", json={"file_path": str(path)})


# ------------------------------------------------------------------ the run pairs by GUID

@pytest.fixture
def run_batch(monkeypatch, tmp_path):
    """``run_batch(files)`` -> (manager, batch id, output folder): the real loader and exporter on real
    files, a recording stand-in for the indexer."""
    monkeypatch.setattr(M, "_batch_names", {})
    monkeypatch.setattr(M, "_register_signal_calibration", lambda signal, file_path: None)
    monkeypatch.setattr(M, "run_single_indexing_job", _Indexer())

    def go(files, formats=("ang",)):
        mgr = M.BatchManager(db_path=str(tmp_path / "jobs.db"))
        cif = tmp_path / "Ni.cif"
        cif.write_text("data_Ni\n", encoding="utf-8")
        out = tmp_path / "out"
        config = {"auto_export": True, "export_dir": str(out), "include_eds": False,
                  "export_formats": list(formats)}
        batch_id = mgr.queue.create_batch([str(f) for f in files],
                                          [{"name": "Ni", "path": str(cif), "method": "hough"}], config)
        mgr._run_batch_sync_impl(batch_id)
        return mgr, batch_id, out

    return go


def test_batch_pairs_by_guid_not_name(run_batch, tmp_path):
    """a.up1 names measurement B and b.up1 measurement A: each file is written on the grid of the
    measurement its header names, whatever the file or measurement is called and whichever comes first."""
    _two_measurements(tmp_path)
    mgr, batch_id, out = run_batch([tmp_path / "a.up1", tmp_path / "b.up1"])
    assert sorted(j["status"] for j in mgr.get_jobs(batch_id)) == ["done", "done"]

    a, b = mgr.exports[batch_id][str(tmp_path / "a.up1")], mgr.exports[batch_id][str(tmp_path / "b.up1")]
    assert a["pairing"] == {"h5_name": "session.h5", "measurement_id": GUID_B,
                            "measurement_name": "Area 1 / OIM Map 2"}
    assert b["pairing"] == {"h5_name": "session.h5", "measurement_id": GUID_A,
                            "measurement_name": "Area 1 / OIM Map 1"}
    assert (a["grid_kind"], b["grid_kind"]) == ("hex", "hex")

    # the checkpoints hold the steps of the measurement, not of the other one
    assert read_checkpoint_geometry(tmp_path / "a_multiphase.h5").step_x == pytest.approx(0.8)
    assert read_checkpoint_geometry(tmp_path / "b_multiphase.h5").step_x == pytest.approx(0.5)

    assert "# GRID: HexGrid" in _header(out / "result_a.ang")
    rows_a, rows_b = _rows(out / "result_a.ang"), _rows(out / "result_b.ang")
    assert (rows_a[5][3], rows_a[10][4]) == pytest.approx((0.4, 1.3856406), abs=1e-5)       # row 1 col 0 x; row 2 y
    assert (rows_b[5][3], rows_b[10][4]) == pytest.approx((0.25, 0.8660254), abs=1e-5)


def test_batch_hex_up_without_metadata_fails_that_file_with_its_code(run_batch, tmp_path):
    """A hexagonal .up1 whose measurement no .h5 beside it names fails alone, with the code the page
    translates; the file that has its metadata goes on."""
    _two_measurements(tmp_path)
    lonely = tmp_path / "sub"
    lonely.mkdir()
    make_up(lonely / "c.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45, guid_bytes=GUID_B_BYTES)   # GUID_B, but no .h5 here
    mgr, batch_id, _out = run_batch([lonely / "c.up1", tmp_path / "a.up1"])
    jobs = {Path(j["file_path"]).name: j for j in mgr.get_jobs(batch_id)}
    assert (jobs["c.up1"]["status"], jobs["c.up1"]["error_code"]) == ("failed", "edaxHexUpUnsupported")
    assert jobs["a.up1"]["status"] == "done"


# ------------------------------------------------------------- quick-load names the pairing

def test_quick_load_names_the_pairing(store, tmp_path):
    _two_measurements(tmp_path)
    resp = _post(tmp_path / "a.up1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grid_kind"] == "hex"
    assert body["grid_shape"] == [N_ROWS, N_COLS]
    geometry = body["scan_geometry"]
    assert (geometry["kind"], geometry["grid_shape"]) == ("hex", [N_ROWS, N_COLS])
    assert geometry["step_x"] == pytest.approx(0.8) and geometry["step_y"] == pytest.approx(0.8 * 3 ** 0.5 / 2)
    apex = body["apex"]
    assert apex["measurement_id"] == GUID_B and apex["measurement_name"] == "Area 1 / OIM Map 2"
    assert Path(apex["h5_path"]).name == "session.h5" and apex["pairing"] == "guid"
    # the kept signal does not carry the loader's copy of the geometry (it would go stale on a crop)
    kept = B._quick_loaded[body["dataset_name"]]["signal"]
    assert kept.metadata.get_item("Signal.scan_geometry", None) is None


def test_quick_load_of_the_other_file_names_the_other_measurement(store, tmp_path):
    _two_measurements(tmp_path)
    body = _post(tmp_path / "b.up1").json()
    assert body["apex"]["measurement_id"] == GUID_A and body["apex"]["measurement_name"] == "Area 1 / OIM Map 1"
    assert body["scan_geometry"]["step_x"] == pytest.approx(0.5)


def test_quick_load_of_a_square_file_without_metadata_says_square_and_no_pairing(store, tmp_path):
    """A square .up1 takes the .osc beside it, as on a path load; it has no APEX measurement."""
    path = make_up(tmp_path / "sq.up1", n=16, nx=4, ny=4, hex_flag=0, guid_bytes=None)
    make_osc(tmp_path / "sq.osc", ncols=4, nrows=4, xstep=0.5, ystep=0.5, pc=(0.51, 0.52, 0.53))
    resp = _post(path)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grid_kind"] == "square" and body["apex"] is None
    assert body["scan_geometry"]["kind"] == "square" and body["grid_shape"] == [4, 4]
    assert body["pc"] is not None and body["pc_source"] != "none"        # the .osc gave the PC


@pytest.mark.parametrize("case", ["no_metadata", "metadata_elsewhere", "header_without_guid"])
def test_batch_hex_up_without_metadata_coded_400(store, tmp_path, case):
    """Nothing names where the points lie, so the file is refused when it is added, by code, and
    nothing of it is remembered."""
    from backend.api.services.batch_manager import store_name_for_file
    folder = tmp_path / "scans"
    folder.mkdir()
    if case == "metadata_elsewhere":
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        _two_measurements(elsewhere)
        path = make_up(folder / "h.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45, guid_bytes=GUID_B_BYTES)
    elif case == "header_without_guid":
        _two_measurements(folder)
        path = make_up(folder / "h.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45, guid_bytes=None)
    else:
        path = make_up(folder / "h.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45, guid_bytes=GUID_B_BYTES)
    resp = _post(path)
    assert resp.status_code == 400, resp.text
    assert resp.headers["x-orienta-code"] == "edaxHexUpUnsupported"
    assert "hexagonal grid" in resp.json()["detail"]
    assert store.get_entry(store_name_for_file(path)) is None
    assert B._quick_loaded == {}


def test_batch_apex_h5_refused_by_code(store, tmp_path):
    """The metadata file lists measurements and holds no patterns; added as a scan it is refused by
    code, not by an uncoded 500 the page skips in silence."""
    _two_measurements(tmp_path)
    resp = _post(tmp_path / "session.h5")
    assert resp.status_code == 400, resp.text
    assert resp.headers["x-orienta-code"] == "apexNeedsMeasurement"
    assert B._quick_loaded == {}


def test_a_failure_without_a_code_stays_a_500(store, tmp_path):
    """Only the coded refusals change: a file that is no scan at all is still the old 500."""
    path = tmp_path / "broken.h5oina"
    path.write_bytes(b"not an hdf5 file")
    resp = _post(path)
    assert resp.status_code == 500, resp.text
    assert "x-orienta-code" not in resp.headers


# ------------------------------------------------------------------- scan-folder

def _plain_h5(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.zeros(3))


def test_scan_folder_lists_up_not_metadata(tmp_path):
    """.up1 and .up2 pattern files are listed; the APEX metadata .h5 beside them is not, and a plain
    .h5 still is. Checkpoints of earlier runs stay out as before."""
    _two_measurements(tmp_path)
    make_up(tmp_path / "c.UP2", n=N_ROWS * N_COLS, nx=N_COLS, ny=45, guid_bytes=GUID_B_BYTES)
    _plain_h5(tmp_path / "plain.h5")
    _plain_h5(tmp_path / "a_multiphase.h5")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    resp = TestClient(app).post("/api/batch-v2/scan-folder", json={"folder": str(tmp_path), "recursive": False})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert sorted(Path(p).name for p in body["files"]) == ["a.up1", "b.up1", "c.UP2", "plain.h5"]
    assert body["count"] == 4
    assert {d["extension"] for d in body["details"]} == {".up1", ".up2", ".h5"}
