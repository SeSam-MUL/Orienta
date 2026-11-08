"""/api/jobs: start, watch, wait for and cancel compute-worker jobs."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.worker import manager as wm  # noqa: E402
from tests.worker_helpers import gone_within, kill_quietly  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_JOBS", "1")
    monkeypatch.delenv("ORIENTA_NO_WORKER", raising=False)
    m = wm.WorkerManager(idle_s=600)
    monkeypatch.setattr(wm, "_MANAGER", m)
    from backend.api.main import app
    # No `with`: the lifespan would run the startup reaper, which kills every
    # EMsoft process in WSL.
    yield TestClient(app)
    pid = m.worker_pid
    m.shutdown()
    kill_quietly(pid)


def test_a_test_job_runs_to_the_end(client):
    jid = client.post("/api/jobs", json={"job": "_echo", "args": {"x": "Ä"}}).json()["id"]
    snap = client.get(f"/api/jobs/{jid}/wait", params={"timeout": 30}).json()
    assert snap["status"] == "done" and snap["result"] == {"x": "Ä"}


def test_an_unknown_job_is_refused_with_a_code(client):
    r = client.post("/api/jobs", json={"job": "sphere", "args": {}})
    assert r.status_code == 400 and r.headers["x-orienta-code"] == "unknownJob"


def test_an_unknown_id_is_404_with_a_code(client):
    for r in (client.get("/api/jobs/nope"), client.get("/api/jobs/nope/wait"),
              client.post("/api/jobs/nope/cancel")):
        assert r.status_code == 404 and r.headers["x-orienta-code"] == "jobNotFound"


def test_the_active_list_filters_by_job(client):
    jid = client.post("/api/jobs", json={"job": "_sleep", "args": {"s": 30}}).json()["id"]
    time.sleep(0.5)
    names = [j["job"] for j in client.get("/api/jobs", params={"job": "_sleep"}).json()["jobs"]]
    assert names == ["_sleep"]
    assert client.get("/api/jobs", params={"job": "_echo"}).json()["jobs"] == []
    client.post(f"/api/jobs/{jid}/cancel")


def test_cancel_over_http_kills_the_worker(client):
    jid = client.post("/api/jobs", json={"job": "_sleep", "args": {"s": 60}}).json()["id"]
    end = time.monotonic() + 30
    while client.get(f"/api/jobs/{jid}").json()["status"] != "running":
        assert time.monotonic() < end
        time.sleep(0.05)
    pid = wm._MANAGER.worker_pid
    assert client.post(f"/api/jobs/{jid}/cancel").json() == {"cancelled": True}
    snap = client.get(f"/api/jobs/{jid}/wait", params={"timeout": 10}).json()
    assert snap["status"] == "cancelled"
    assert gone_within(pid, 3.0)


def test_wait_never_blocks_longer_than_thirty_seconds(client):
    jid = client.post("/api/jobs", json={"job": "_sleep", "args": {"s": 5}}).json()["id"]
    t0 = time.monotonic()
    snap = client.get(f"/api/jobs/{jid}/wait", params={"timeout": 1}).json()
    assert time.monotonic() - t0 < 5 and snap["status"] in ("queued", "starting", "running")


def test_shutting_the_manager_down_kills_a_running_worker(monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_JOBS", "1")
    m = wm.WorkerManager(idle_s=600)
    jid = m.submit("_sleep", {"s": 60})
    end = time.monotonic() + 30
    while m.get(jid)["status"] != "running":
        assert time.monotonic() < end
        time.sleep(0.05)
    pid = m.worker_pid
    monkeypatch.setattr(wm, "_MANAGER", m)
    wm.shutdown_manager()
    assert gone_within(pid, 3.0)


def test_the_lifespan_shuts_the_manager_down(monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_JOBS", "1")
    called = []
    monkeypatch.setattr(wm, "shutdown_manager", lambda: called.append(True))
    from backend.api import main as backend_main

    async def _no_reaper():
        return None
    # The real reaper would kill every EMsoft process in WSL.
    monkeypatch.setattr(backend_main, "_reap_orphaned_emsoft", _no_reaper)
    with TestClient(backend_main.app):
        pass
    assert called == [True]
