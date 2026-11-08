"""A batch failure carries a code the interface translates, not only English prose.

A file with no PC fails its batch jobs ("carries no pattern centre ..."). The message
was stored as English text only, in a window that may be German or Japanese. The legacy
batch result and the backend batch job now also carry ``error_code`` (``pc_missing``);
the pages translate it. Real batch code on a real UP1 without an .osc.
"""
import sqlite3
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import indexing as I
from backend.api.services.batch_queue import BatchQueue
from backend.api.services.calibration_store import CalibrationStore

SIDE = 24


def _write_up1(path, n_side=4):
    rng = np.random.default_rng(3)
    pats = rng.integers(0, 250, (n_side * n_side, SIDE, SIDE)).astype("u1")
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, SIDE, SIDE, 16))
        f.write(pats.tobytes())
    return str(path)


NI_CIF = (
    "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
    "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
    "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n")


@pytest.fixture
def stores(monkeypatch):
    from backend.api.routes import ebsd_viewer as ev
    store = CalibrationStore()
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(I, "calibration_store", store)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    return store


def test_the_legacy_batch_result_carries_the_code(stores, tmp_path, monkeypatch):
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    path = _write_up1(tmp_path / "scan_a.up1")
    saved = dict(I._batch_state)
    try:
        resp = TestClient(app).post("/api/indexing/batch/start", json={"datasets": [
            {"file_path": path, "method": "hough"}]})
        assert resp.status_code == 200, resp.text
        entry = I._batch_state["results"][-1]
        assert entry["error_code"] == "pc_missing" and "carries no pattern centre" in entry["error"]
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved)


def test_the_backend_batch_job_carries_the_code(stores, tmp_path, monkeypatch):
    from backend.api.services.batch_manager import BatchManager
    path = _write_up1(tmp_path / "scan_b.up1")
    cif = tmp_path / "Ni.cif"
    cif.write_text(NI_CIF, encoding="utf-8")
    mgr = BatchManager(db_path=str(tmp_path / "jobs.db"))
    batch_id = mgr.queue.create_batch([path], [{"name": "Ni", "path": str(cif), "method": "hough"}], {})
    mgr._run_batch_sync_impl(batch_id)
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "failed", job
    assert job["error_code"] == "pc_missing" and "carries no pattern centre" in job["error_msg"]


def test_the_queue_stores_the_code_and_a_retry_clears_it(tmp_path):
    q = BatchQueue(str(tmp_path / "q.db"))
    batch = q.create_batch(["a.up1"], [{"name": "Ni", "path": "Ni.cif", "method": "hough"}], {})
    (job,) = q.get_jobs(batch)
    q.update_job(job["id"], status="failed", error_msg="boom", error_code="pc_missing")
    assert q.get_jobs(batch)[0]["error_code"] == "pc_missing"
    q.retry_failed(batch)
    again = q.get_jobs(batch)[0]
    assert again["status"] == "pending" and again["error_code"] is None


def test_a_database_from_before_the_column_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE batches (id TEXT PRIMARY KEY, created_at TEXT, status TEXT DEFAULT 'pending',
                config_json TEXT, total_jobs INTEGER DEFAULT 0, completed INTEGER DEFAULT 0,
                failed INTEGER DEFAULT 0);
            CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, file_path TEXT,
                file_name TEXT, phase_name TEXT, phase_path TEXT, method TEXT,
                status TEXT DEFAULT 'pending', ci_mean REAL, ci_median REAL, error_msg TEXT,
                started_at TEXT, finished_at TEXT, duration_sec REAL, retry_count INTEGER DEFAULT 0,
                sort_order INTEGER DEFAULT 0);
        """)
    q = BatchQueue(str(db))
    batch = q.create_batch(["a.up1"], [{"name": "Ni", "path": "Ni.cif", "method": "hough"}], {})
    (job,) = q.get_jobs(batch)
    q.update_job(job["id"], status="failed", error_msg="x", error_code="pc_missing")
    assert q.get_jobs(batch)[0]["error_code"] == "pc_missing"


def _refused_hex():
    from edax_up1 import EdaxHexUpError
    return EdaxHexUpError("scan.up1")


def _apex_needs_measurement():
    from edax_apex import ApexError
    return ApexError("apexNeedsMeasurement", "this file holds several measurements")


@pytest.mark.parametrize("make_error, expected_code", [
    (_refused_hex, "edaxHexUpUnsupported"),
    (_apex_needs_measurement, "apexNeedsMeasurement"),
    (lambda: OSError("the disk went away"), None),
], ids=["hexagonal pattern file without metadata", "apex file needs a measurement", "plain OSError has no code"])
def test_a_batch_job_whose_file_fails_to_load_keeps_the_errors_code(
        tmp_path, monkeypatch, make_error, expected_code):
    """The load failure is recorded on every job of the file; the code the interface translates
    on (``EdaxHexUpError.code``, ``ApexError.code``) must not be lost on the way."""
    from backend.api.services import batch_manager
    from backend.api.services.batch_manager import BatchManager

    def refuse(_path):
        raise make_error()

    monkeypatch.setattr(batch_manager, "_load_signal", refuse)
    mgr = BatchManager(db_path=str(tmp_path / "jobs.db"))
    batch_id = mgr.queue.create_batch(
        [str(tmp_path / "scan.up1")], [{"name": "Ni", "path": "Ni.cif", "method": "hough"}], {})
    mgr._run_batch_sync_impl(batch_id)
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "failed", job
    assert job["error_msg"].startswith("Load failed: ")
    assert job["error_code"] == expected_code
