"""A refused Hough library is a coded problem on the two background paths that lacked it.

The PC optimisation (``/api/pc/optimize``) and the batch-v2 job used to end a refused library
as English text only: the page could not word it in the user's language and had no phase,
row count or one-click answer to offer. They now carry what the other routes carry
(``HoughLibraryTooBig.as_problem``): the code ``hough_library_too_big`` and its params.
Real routes, real indexer build on a real Mg CIF under a small memory budget.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_hough_refusal import small_budget  # noqa: E402,F401
from tests.test_hough_reflector_routes import cifs, clean_registry, client  # noqa: E402,F401


def _check_problem(problem, phase="Mg"):
    assert problem["code"] == "hough_library_too_big"
    params = problem["params"]
    (entry,) = params["phases"]
    assert entry["name"] == phase and entry["rows"] == 70
    assert entry["fit_rows"] == 50 and entry["own_selection"] is False
    assert params["min_rows"] == 40 and params["min_families"] == 4
    assert params["budget_bytes"] > 0


def test_the_pc_optimisation_ends_with_the_coded_problem(monkeypatch, small_budget, cifs, client):
    import kikuchipy as kp
    import torch

    from backend.api.routes import ebsd_viewer

    c, pcr = client
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    signal = kp.data.nickel_ebsd_small()
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: signal)
    assert c.post("/api/pc/phase/add", json={"cif_path": cifs["Mg"]}).status_code == 200
    r = c.post("/api/pc/optimize", json={"patterns": [0], "method": "Nelder-Mead"})
    assert r.status_code == 200, r.text
    task = c.get(f"/api/pc/optimize/{r.json()['task_id']}").json()      # the background task has run
    assert task["status"] == "failed" and task["message"] == "Hough library too big"
    _check_problem(task["problem"])
    # the English text is still there for a reader that does not translate
    assert "Reflector families" in task["error"]
    assert pcr._optimization_active is False          # the next run is not blocked


def test_the_batch_job_carries_the_code_and_the_params(monkeypatch, small_budget, cifs, tmp_path):
    import kikuchipy as kp
    import safe_loader
    import torch

    from backend.api.routes import batch_v2 as B
    from backend.api.services import batch_manager as M
    from backend.api.services.calibration_store import CalibrationStore

    store = CalibrationStore()
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(B, "calibration_store", store)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    signal = kp.data.nickel_ebsd_small()
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path, *a, **k: signal.deepcopy())
    scan = tmp_path / "Scan9.h5oina"
    scan.write_bytes(b"x")                       # only the path matters: the loader is the seam
    mgr = M.BatchManager(db_path=str(tmp_path / "jobs.db"))
    batch_id = mgr.queue.create_batch(
        [str(scan)], [{"name": "Mg", "path": cifs["Mg"], "method": "hough"}], {})
    mgr._run_batch_sync_impl(batch_id)
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "failed", job
    assert job["error_code"] == "hough_library_too_big"
    assert "Reflector families" in job["error_msg"]
    _check_problem({"code": job["error_code"], "params": job["error_params"]})


def test_the_queue_stores_the_params_and_a_retry_clears_them(tmp_path):
    from backend.api.services.batch_queue import BatchQueue

    q = BatchQueue(str(tmp_path / "q.db"))
    batch = q.create_batch(["a.up1"], [{"name": "Ni", "path": "Ni.cif", "method": "hough"}], {})
    (job,) = q.get_jobs(batch)
    assert job["error_params"] is None
    q.update_job(job["id"], status="failed", error_msg="boom", error_code="x",
                 error_params={"phases": [{"name": "Ni", "rows": 70}], "budget_bytes": 7})
    assert q.get_jobs(batch)[0]["error_params"] == {"phases": [{"name": "Ni", "rows": 70}],
                                                   "budget_bytes": 7}
    q.retry_failed(batch)
    assert q.get_jobs(batch)[0]["error_params"] is None


def test_a_database_from_before_the_column_is_migrated(tmp_path):
    import sqlite3

    from backend.api.services.batch_queue import BatchQueue

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE batches (id TEXT PRIMARY KEY, created_at TEXT, status TEXT DEFAULT 'pending',
                config_json TEXT, total_jobs INTEGER DEFAULT 0, completed INTEGER DEFAULT 0,
                failed INTEGER DEFAULT 0);
            CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, file_path TEXT,
                file_name TEXT, phase_name TEXT, phase_path TEXT, method TEXT,
                status TEXT DEFAULT 'pending', ci_mean REAL, ci_median REAL, error_msg TEXT,
                error_code TEXT, started_at TEXT, finished_at TEXT, duration_sec REAL,
                retry_count INTEGER DEFAULT 0, sort_order INTEGER DEFAULT 0);
        """)
    q = BatchQueue(str(db))
    batch = q.create_batch(["a.up1"], [{"name": "Ni", "path": "Ni.cif", "method": "hough"}], {})
    (job,) = q.get_jobs(batch)
    q.update_job(job["id"], status="failed", error_msg="x", error_code="c", error_params={"a": 1})
    assert q.get_jobs(batch)[0]["error_params"] == {"a": 1}


def test_the_pixelwise_calibration_ends_with_the_coded_problem(monkeypatch, small_budget, cifs, client):
    import kikuchipy as kp
    import torch

    from backend.api.routes import ebsd_viewer

    c, pcr = client
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    signal = kp.data.nickel_ebsd_small()
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: signal)
    assert c.post("/api/pc/phase/add", json={"cif_path": cifs["Mg"]}).status_code == 200
    r = c.post("/api/pc/calibrate/grid", json={"grid_step": 1, "mode": "plane"})
    assert r.status_code == 200, r.text
    task = c.get(f"/api/pc/optimize/{r.json()['task_id']}").json()      # the background task has run
    assert task["status"] == "failed" and task["message"] == "Hough library too big"
    _check_problem(task["problem"])
    assert "Reflector families" in task["error"]
    assert pcr._optimization_active is False
