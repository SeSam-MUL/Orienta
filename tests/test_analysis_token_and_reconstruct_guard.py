"""GET /status: ``dataset_token`` and ``reconstruct``; POST /grains/reconstruct: 409 while one runs."""
import asyncio
import uuid

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset
from tests.synthetic_grain_map import make_map, to_xmap

BUSY = "A grain reconstruction is already running; wait for it to finish."
BODY = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 0.0, "fill": False}


def _ds():
    m = make_map(shape=(20, 24), n_seeds=6)
    return EBSDDataset(to_xmap(m), step_size=m["step"])


@pytest.fixture
def ar():
    import backend.api.routes.analysis as mod
    saved = mod._dataset, dict(mod._grain_stats_settings), dict(mod._dataset_by_file)
    mod._dataset = _ds()
    mod._reset_grain_stats_settings()
    mod._clear_reconstruct_marker()
    try:
        yield mod
    finally:
        mod._dataset = saved[0]
        mod._grain_stats_settings.clear()
        mod._grain_stats_settings.update(saved[1])
        mod._dataset_by_file.clear()
        mod._dataset_by_file.update(saved[2])
        mod._clear_reconstruct_marker()


def _status(ar):
    return asyncio.run(ar.analysis_status())


def _start(ar, **extra):
    """Start a reconstruction without running it: the handler's answer and the background task still to run."""
    bt = BackgroundTasks()
    out = asyncio.run(ar.reconstruct_grains(ar.GrainReconstructionRequest(**{**BODY, **extra}), bt))
    return out["task_id"], bt


def _run(bt):
    for t in bt.tasks:
        t.func(*t.args, **t.kwargs)


# ---------------------------------------------------------------------------------------------
# dataset_token
# ---------------------------------------------------------------------------------------------
def test_a_dataset_gets_a_uuid4_hex_token_at_construction():
    a, b = _ds(), _ds()
    assert uuid.UUID(a.dataset_token).version == 4 and a.dataset_token == uuid.UUID(a.dataset_token).hex
    assert a.dataset_token != b.dataset_token


def test_status_carries_the_token_and_it_follows_the_object(ar):
    first = _status(ar)["dataset_token"]
    assert first == ar._dataset.dataset_token and _status(ar)["dataset_token"] == first      # same object, same token
    ar._dataset = _ds()
    assert _status(ar)["dataset_token"] != first                                              # another map


def test_the_token_survives_stash_and_restore_of_the_same_object(ar):
    token = _status(ar)["dataset_token"]
    ar.stash_dataset_for_file("fileA")
    ar._dataset = _ds()
    assert _status(ar)["dataset_token"] != token
    assert ar.restore_dataset_for_file("fileA") is True
    assert _status(ar)["dataset_token"] == token


def test_loading_a_file_again_is_another_map_with_another_token(ar, tmp_path):
    from tests.test_analysis_page_sync import _write_ctf
    _write_ctf(tmp_path / "t.ctf")
    from backend.api.main import app
    c = TestClient(app)
    assert c.post("/api/analysis/load", json={"xmap_path": str(tmp_path / "t.ctf")}).status_code == 200
    t1 = c.get("/api/analysis/status").json()["dataset_token"]
    assert c.post("/api/analysis/load", json={"xmap_path": str(tmp_path / "t.ctf")}).status_code == 200
    assert c.get("/api/analysis/status").json()["dataset_token"] != t1


def test_status_without_a_dataset_has_no_token(ar):
    ar._dataset = None
    assert _status(ar) == {"loaded": False}


# ---------------------------------------------------------------------------------------------
# reconstruct in /status
# ---------------------------------------------------------------------------------------------
def test_status_has_no_reconstruct_when_nothing_runs(ar):
    assert _status(ar)["reconstruct"] is None


def test_status_names_the_running_reconstruction_and_forgets_it_when_done(ar):
    task_id, bt = _start(ar)
    assert _status(ar)["reconstruct"] == {"task_id": task_id, "status": "running", "progress": 0.0}
    _run(bt)
    assert ar._analysis_tasks[task_id]["status"] == "completed"
    assert _status(ar)["reconstruct"] is None


def test_status_reconstruct_is_about_the_loaded_map_only(ar):
    _start(ar)
    ar._dataset = _ds()                     # another map: the run of the old one is no business of its page
    assert _status(ar)["reconstruct"] is None


def test_a_vanished_task_is_not_reported(ar):
    _start(ar)
    ar._analysis_tasks.clear()
    assert _status(ar)["reconstruct"] is None


# ---------------------------------------------------------------------------------------------
# 409
# ---------------------------------------------------------------------------------------------
def test_a_second_reconstruction_on_the_same_dataset_is_refused(ar):
    _, bt = _start(ar)
    with pytest.raises(HTTPException) as e:
        _start(ar)
    assert e.value.status_code == 409 and e.value.detail == BUSY
    _run(bt)
    task_id, bt2 = _start(ar)               # finished: free again
    _run(bt2)
    assert ar._analysis_tasks[task_id]["status"] == "completed"


def test_the_409_goes_over_the_wire(ar):
    from backend.api.main import app
    _start(ar)
    r = TestClient(app).post("/api/analysis/grains/reconstruct", json=BODY)
    assert r.status_code == 409 and r.json()["detail"] == BUSY


def test_a_run_on_another_dataset_does_not_block(ar):
    _start(ar)
    ar._dataset = _ds()
    task_id, bt = _start(ar)
    _run(bt)
    assert ar._analysis_tasks[task_id]["status"] == "completed"


def test_a_vanished_task_frees_the_marker(ar):
    _start(ar)
    ar._analysis_tasks.clear()
    task_id, bt = _start(ar)
    assert task_id in ar._analysis_tasks


def test_a_failing_task_frees_the_marker(ar, monkeypatch):
    import analysis.grain_analysis as ga
    monkeypatch.setattr(ga, "reconstruct_grains", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    task_id, bt = _start(ar)
    _run(bt)
    assert ar._analysis_tasks[task_id]["status"] == "failed" and _status(ar)["reconstruct"] is None
    monkeypatch.undo()
    _, bt2 = _start(ar)                     # not blocked
    _run(bt2)


def test_the_marker_is_released_in_the_finally_even_when_the_failure_handler_itself_fails(ar, monkeypatch):
    """The run's own bookkeeping can fail too (the registry entry is gone): the marker must not stay."""
    import analysis.grain_analysis as ga

    def boom(*a, **k):
        ar._analysis_tasks.clear()
        raise RuntimeError("boom")

    monkeypatch.setattr(ga, "reconstruct_grains", boom)
    _, bt = _start(ar)
    with pytest.raises(KeyError):
        _run(bt)
    assert not ar._reconstruct_runs                  # the entry is gone, not merely ignored


def test_a_finished_task_does_not_clear_the_marker_of_a_newer_run(ar):
    """Two datasets, two runs: the older one finishing must leave the newer one's marker alone."""
    old_dataset = ar._dataset
    _, bt_old = _start(ar)
    ar._dataset = _ds()
    new_id, bt_new = _start(ar)
    _run(bt_old)
    assert _status(ar)["reconstruct"]["task_id"] == new_id
    ar._dataset = old_dataset
    _run(bt_new)


# ---------------------------------------------------------------------------------------------
# the task cap and concurrent posts
# ---------------------------------------------------------------------------------------------
from tests.test_analysis_page_sync import (  # noqa: E402,F401  (fixtures and helpers of the page-sync tests)
    _mixed_folder, _reconstruct, _stub_slow_steps, client_map, client_two_phase,
)


@pytest.fixture
def clean_tasks():
    import backend.api.routes.analysis as mod
    saved = dict(mod._analysis_tasks)
    mod._analysis_tasks.clear()
    try:
        yield mod
    finally:
        mod._analysis_tasks.clear()
        mod._analysis_tasks.update(saved)


def test_the_task_cap_never_evicts_a_running_task(clean_tasks):
    ar = clean_tasks
    ar._track_task("long-run", {"status": "running", "progress": 0.0})       # the oldest, and still running
    for i in range(ar.MAX_TRACKED_TASKS + 5):
        ar._track_task(f"done-{i}", {"status": "completed", "progress": 1.0})
    assert "long-run" in ar._analysis_tasks and len(ar._analysis_tasks) == ar.MAX_TRACKED_TASKS
    assert "done-0" not in ar._analysis_tasks                                  # the oldest finished one went
    assert f"done-{ar.MAX_TRACKED_TASKS + 4}" in ar._analysis_tasks


def test_the_task_cap_keeps_every_task_when_all_are_running(clean_tasks):
    ar = clean_tasks
    for i in range(ar.MAX_TRACKED_TASKS + 3):
        ar._track_task(f"run-{i}", {"status": "running", "progress": 0.0})
    assert len(ar._analysis_tasks) == ar.MAX_TRACKED_TASKS + 3


def test_a_run_survives_fifty_newer_tasks_and_still_blocks_a_second_one(ar, clean_tasks):
    task_id, bt = _start(ar)
    for i in range(ar.MAX_TRACKED_TASKS + 2):
        ar._track_task(f"other-{i}", {"status": "completed", "progress": 1.0})
    assert _status(ar)["reconstruct"]["task_id"] == task_id
    with pytest.raises(HTTPException) as e:
        _start(ar)
    assert e.value.status_code == 409
    _run(bt)


def test_two_real_concurrent_posts_give_exactly_one_409(ar, monkeypatch):
    import threading
    import analysis.grain_analysis as ga
    from backend.api.main import app
    release, started = threading.Event(), threading.Event()
    real = ga.reconstruct_grains

    def slow(*a, **k):
        started.set()
        assert release.wait(20), "the test never released the run"
        return real(*a, **k)

    monkeypatch.setattr(ga, "reconstruct_grains", slow)
    barrier = threading.Barrier(2)
    codes, lock = [], threading.Lock()

    def post():
        barrier.wait(10)
        r = TestClient(app).post("/api/analysis/grains/reconstruct", json=BODY)      # returns when its run is over
        with lock:
            codes.append((r.status_code, r.json().get("detail")))

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
    try:
        deadline = __import__("time").time() + 20
        while not codes and __import__("time").time() < deadline:      # the loser answers at once, the winner waits
            __import__("time").sleep(0.02)
        assert started.wait(10)
    finally:
        release.set()
        for t in threads:
            t.join(30)
    assert sorted(c for c, _ in codes) == [200, 409], codes
    assert [d for c, d in codes if c == 409] == [BUSY]


def test_the_export_reports_a_mixed_slash_path_as_the_same_file_not_a_rename(client_map, monkeypatch, tmp_path):
    import os
    import analysis.excel_exporter as ee
    client, ar = client_map
    monkeypatch.setattr(ar, "_collect_export_results", lambda warnings: {})
    monkeypatch.setattr(ee, "export_to_excel", lambda path, name, results: open(path, "wb").write(b"PK"))
    (tmp_path / "out").mkdir()
    mixed = str(tmp_path).replace(os.sep, "/") + "/out" + os.sep + "Doc.xlsx"       # forward slashes, then one backslash
    r = client.post("/api/analysis/export/excel", json={"output_path": mixed, "dataset_name": "d"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    res = task["result"]
    assert res["path"] == res["requested_path"] == str(tmp_path / "out" / "Doc.xlsx")
    assert "message" not in task and "already exists" not in str(task.get("message", ""))
    # the same file again really is a rename, and says so with the OS spelling
    r = client.post("/api/analysis/export/excel", json={"output_path": mixed, "dataset_name": "d"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["result"]["path"].endswith("Doc_2.xlsx") and task["result"]["requested_path"] == res["requested_path"]
    assert "already exists" in task["message"]


def test_the_batch_does_not_claim_a_rename_for_a_mixed_slash_output_path(client_map, monkeypatch, tmp_path):
    import os
    client, _ = client_map
    _mixed_folder(tmp_path, monkeypatch)
    mixed = str(tmp_path).replace(os.sep, "/") + os.sep + "doc.xlsx"
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": mixed})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    assert "Workbook:" in task["message"] and "already existed" not in task["message"]


def test_the_rx_map_sends_the_low_kam_fractions_of_the_rx_analysis(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    want = ar._collect_export_results([])["rx"]
    assert want.low_gkam_fraction > 0.0 and want.low_kam_fraction > 0.0
    body = client.get("/api/analysis/map/rx").json()
    assert body["low_gkam_fraction"] == pytest.approx(want.low_gkam_fraction)
    assert body["low_kam_fraction"] == pytest.approx(want.low_kam_fraction)


def test_the_rx_map_leaves_the_low_kam_fractions_out_without_native_band_contrast(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    del ar._dataset.xmap.prop["bc"]
    body = client.get("/api/analysis/map/rx").json()
    assert "low_gkam_fraction" not in body and "low_kam_fraction" not in body      # not 0.0
    assert "rx_fraction" not in body and "rx_grain_count" not in body        # a GOS-only number would pass for the analysis
