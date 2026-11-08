"""The sphere preview hands its expensive step to the compute worker."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import sht_sphere  # noqa: E402
from backend.worker import manager as wm  # noqa: E402
from tests.worker_helpers import gone_within, kill_quietly  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(sht_sphere, "_COMPUTE_LOCK", threading.Lock())
    monkeypatch.setattr(sht_sphere, "_INFLIGHT", {})
    sht_sphere.clear_cache()


class _FakeManager:
    def __init__(self, result=None, error=None):
        self.calls, self.result, self.error = [], result, error

    def run(self, job, args, cancelled=None, poll=0.2):
        self.calls.append((job, dict(args)))
        if self.error is not None:
            raise self.error
        return self.result


def _with_worker(monkeypatch, fake):
    monkeypatch.delenv("ORIENTA_NO_WORKER", raising=False)
    monkeypatch.setattr(wm, "get_manager", lambda: fake)
    # "cuda" skips the backend's CPU RAM check (only a name here: no CUDA
    # context is created), so these tests behave the same on a CPU-only runner.
    # The RAM test below switches to "cpu" on purpose.
    monkeypatch.setenv("SHT_FORWARD_DEVICE", "cuda")


def _a_sht(tmp_path) -> str:
    f = tmp_path / "Größe [Pearson] {20kV}.sht"
    f.write_bytes(b"not read in these tests")
    return str(f)


def test_the_expensive_step_goes_to_the_worker_not_the_backend(monkeypatch, tmp_path):
    fake = _FakeManager(result={"filename": "x"})
    _with_worker(monkeypatch, fake)

    def _must_not_run(*a):
        raise AssertionError("computed in the backend")
    monkeypatch.setattr(sht_sphere, "_build_payload", _must_not_run)
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    path = _a_sht(tmp_path)
    assert sht_sphere.sphere_from_sht(path, max_bandwidth=384, target_size=256) == {"filename": "x"}
    assert fake.calls == [("sphere", {"path": str(Path(path)), "max_bandwidth": 384, "target_size": 256})]


@pytest.mark.parametrize("code, exc", [
    ("sphereOutOfMemory", sht_sphere.SphereOutOfMemoryError),
    ("sphereFailed", sht_sphere.SphereError),
    ("notFound", FileNotFoundError),
])
def test_worker_codes_become_the_old_exceptions(monkeypatch, tmp_path, code, exc):
    _with_worker(monkeypatch, _FakeManager(error=wm.JobFailed("msg", code=code)))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    with pytest.raises(exc) as info:
        sht_sphere.sphere_from_sht(_a_sht(tmp_path))
    # SphereOutOfMemoryError is a SphereError subclass, so pin the exact type:
    # the OOM code must stay the specific one, the generic code the generic one.
    assert type(info.value) is exc


def test_a_cancelled_job_is_a_cancelled_sphere(monkeypatch, tmp_path):
    _with_worker(monkeypatch, _FakeManager(error=wm.JobCancelled("cancelled")))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    with pytest.raises(sht_sphere.SphereCancelled):
        sht_sphere.sphere_from_sht(_a_sht(tmp_path))


def test_a_crashed_worker_is_not_disguised(monkeypatch, tmp_path):
    _with_worker(monkeypatch, _FakeManager(error=wm.WorkerCrashed("exit code 3")))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    with pytest.raises(wm.WorkerCrashed):
        sht_sphere.sphere_from_sht(_a_sht(tmp_path))


def test_the_route_answers_worker_crashed_with_its_code(monkeypatch):
    _with_worker(monkeypatch, _FakeManager(error=wm.WorkerCrashed("exit code 3")))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    monkeypatch.setattr(sht_sphere, "cached_sphere", lambda *a: None)
    from fastapi.testclient import TestClient
    from backend.api.main import app
    r = TestClient(app).get("/api/database/sphere/x.sht")      # no lifespan, no reaper
    assert r.status_code == 422 and r.headers["x-orienta-code"] == "workerCrashed"


def test_the_route_answers_a_worker_that_does_not_start_with_503(monkeypatch):
    _with_worker(monkeypatch, _FakeManager(error=wm.WorkerStartFailed("never became ready")))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    monkeypatch.setattr(sht_sphere, "cached_sphere", lambda *a: None)
    from fastapi.testclient import TestClient
    from backend.api.main import app
    r = TestClient(app).get("/api/database/sphere/x.sht")
    assert r.status_code == 503 and r.headers["x-orienta-code"] == "workerStartFailed"


def test_a_cpu_sphere_without_ram_is_refused_before_a_worker_starts(monkeypatch, tmp_path):
    """The RAM check stays in the backend and decides WHETHER a job goes
    to the worker; a refusal must not cost a worker start first."""
    fake = _FakeManager(result={"never": True})
    _with_worker(monkeypatch, fake)
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    monkeypatch.setenv("SHT_FORWARD_DEVICE", "cpu")
    import psutil
    from backend.api.services import sht_pattern_renderer as spr
    # No credit from an earlier CPU load in this test process (as in test_sphere_ram_guard).
    monkeypatch.setattr(spr, "_RAM_RETAINED_GB", {})
    monkeypatch.setattr(spr, "_RAM_RSS_BASELINE_GB", None)

    class _VM:
        available = 1 * 2**30
        total = 16 * 2**30
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    with pytest.raises(sht_sphere.SphereOutOfMemoryError):
        sht_sphere.sphere_from_sht(_a_sht(tmp_path), max_bandwidth=384, target_size=256)
    assert fake.calls == []


def test_the_cpu_only_ram_check_does_not_apply_to_a_gpu_sphere(monkeypatch, tmp_path):
    """On a GPU the sphere does not live in RAM: 1 GB free must not stop a bw-384 job."""
    fake = _FakeManager(result={"ok": True})
    _with_worker(monkeypatch, fake)          # device "cuda"
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    import psutil
    from backend.api.services import sht_pattern_renderer as spr
    monkeypatch.setattr(spr, "_RAM_RETAINED_GB", {})
    monkeypatch.setattr(spr, "_RAM_RSS_BASELINE_GB", None)

    class _VM:
        available = 1 * 2**30
        total = 16 * 2**30
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    assert sht_sphere.sphere_from_sht(_a_sht(tmp_path), max_bandwidth=384, target_size=256) == {"ok": True}
    assert len(fake.calls) == 1


def test_the_ram_refusal_logs_the_file_name_not_the_path(monkeypatch, tmp_path, caplog):
    """The log goes into the diagnostics zip: no local path in it."""
    import logging
    _with_worker(monkeypatch, _FakeManager(result={"never": True}))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    monkeypatch.setenv("SHT_FORWARD_DEVICE", "cpu")
    import psutil
    from backend.api.services import sht_pattern_renderer as spr
    monkeypatch.setattr(spr, "_RAM_RETAINED_GB", {})
    monkeypatch.setattr(spr, "_RAM_RSS_BASELINE_GB", None)

    class _VM:
        available = 1 * 2**30
        total = 16 * 2**30
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    path = _a_sht(tmp_path)
    with caplog.at_level(logging.WARNING, logger=sht_sphere.logger.name):
        with pytest.raises(sht_sphere.SphereOutOfMemoryError):
            sht_sphere.sphere_from_sht(path, max_bandwidth=384, target_size=256)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert Path(path).name in text
    assert str(tmp_path) not in text


def test_an_unexpected_worker_failure_keeps_the_old_wording(monkeypatch, tmp_path):
    _with_worker(monkeypatch, _FakeManager(
        error=wm.JobFailed("ValueError: boom\nTraceback line two", code="jobFailed")))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    with pytest.raises(sht_sphere.SphereError) as info:
        sht_sphere.sphere_from_sht(_a_sht(tmp_path))
    assert str(info.value) == "sphere reconstruction failed: ValueError: boom"


def test_a_failure_the_job_named_is_not_prefixed_again(monkeypatch, tmp_path):
    _with_worker(monkeypatch, _FakeManager(
        error=wm.JobFailed("could not reconstruct sphere from x.sht: bad", code="sphereFailed")))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    with pytest.raises(sht_sphere.SphereError) as info:
        sht_sphere.sphere_from_sht(_a_sht(tmp_path))
    assert str(info.value) == "could not reconstruct sphere from x.sht: bad"


@pytest.mark.parametrize("error", [
    wm.WorkerError("The compute process manager has stopped.", code="workerStopped"),   # from submit()
    wm.JobFailed("The compute process manager has stopped.", code="workerStopped"),     # from _outcome()
])
def test_the_route_answers_a_stopping_backend_with_503(monkeypatch, error):
    _with_worker(monkeypatch, _FakeManager(error=error))
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    monkeypatch.setattr(sht_sphere, "cached_sphere", lambda *a: None)
    from fastapi.testclient import TestClient
    from backend.api.main import app
    r = TestClient(app).get("/api/database/sphere/x.sht")
    assert r.status_code == 503 and r.headers["x-orienta-code"] == "workerStopped"


class _SlowManager:
    """run() works until released; it notes whether it was asked to cancel."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_seen = False
        self.calls = 0

    def run(self, job, args, cancelled=None, poll=0.2):
        self.calls += 1
        self.started.set()
        while not self.release.is_set():
            if cancelled is not None and cancelled():
                self.cancel_seen = True
                raise wm.JobCancelled("cancelled")
            time.sleep(0.05)
        return {"filename": "done"}


def _two_clients(monkeypatch, tmp_path):
    """A leader and a follower asking for the same sphere; returns the pieces."""
    slow = _SlowManager()
    _with_worker(monkeypatch, slow)
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    path = _a_sht(tmp_path)
    key = (str(Path(path)), 128, 128)
    out = {}
    leader_gone, follower_gone = threading.Event(), threading.Event()

    def _go(name, gone):
        try:
            out[name] = sht_sphere.sphere_from_sht(path, cancelled=gone.is_set)
        except BaseException as e:      # noqa: BLE001 -- the test inspects it
            out[name] = e

    t1 = threading.Thread(target=_go, args=("leader", leader_gone), daemon=True)
    t1.start()
    assert slow.started.wait(10)
    t2 = threading.Thread(target=_go, args=("follower", follower_gone), daemon=True)
    t2.start()
    end = time.monotonic() + 10
    while time.monotonic() < end:
        fl = sht_sphere._INFLIGHT.get(key)
        if fl is not None and fl.followers == 1:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("the follower never joined the flight")
    return slow, out, leader_gone, follower_gone, (t1, t2)


def test_the_job_keeps_running_when_only_the_leaders_client_leaves(monkeypatch, tmp_path):
    slow, out, leader_gone, follower_gone, threads = _two_clients(monkeypatch, tmp_path)
    leader_gone.set()
    time.sleep(1.0)                     # several polls of both sides
    assert not slow.cancel_seen, "a follower still waits for this job"
    slow.release.set()
    for t in threads:
        t.join(10)
    assert out["follower"] == {"filename": "done"}
    assert slow.calls == 1              # the follower did not start a second job


def test_the_job_is_cancelled_once_nobody_waits_for_it(monkeypatch, tmp_path):
    slow, out, leader_gone, follower_gone, threads = _two_clients(monkeypatch, tmp_path)
    leader_gone.set()
    time.sleep(0.6)
    assert not slow.cancel_seen
    follower_gone.set()                 # the last waiting client leaves
    for t in threads:
        t.join(10)
    assert slow.cancel_seen
    assert isinstance(out["leader"], sht_sphere.SphereCancelled)
    assert isinstance(out["follower"], sht_sphere.SphereCancelled)


def test_a_client_that_leaves_kills_the_real_worker(monkeypatch, tmp_path):
    """End to end with a real worker; a sleeping stand-in replaces the sphere job."""
    monkeypatch.setenv("ORIENTA_WORKER_TEST_JOBS", "1")
    monkeypatch.delenv("ORIENTA_NO_WORKER", raising=False)
    m = wm.WorkerManager(idle_s=600)
    monkeypatch.setattr(wm, "get_manager", lambda: m)
    monkeypatch.setattr(sht_sphere, "_SPHERE_JOB", "_sleep_heavy")
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda name: Path(name))
    t0 = time.monotonic()
    pids = []

    def _gone():
        if m.worker_pid:
            pids.append(m.worker_pid)
        return time.monotonic() - t0 > 2.0

    try:
        with pytest.raises(sht_sphere.SphereCancelled):
            sht_sphere.sphere_from_sht(_a_sht(tmp_path), cancelled=_gone)
        assert pids and gone_within(pids[-1], 3.0)
    finally:
        pid = m.worker_pid
        m.shutdown()
        kill_quietly(pid)


class _Ctx:
    def progress(self, frac, msg=""):
        pass


@pytest.mark.parametrize("raised, code", [
    (sht_sphere.SphereOutOfMemoryError("oom"), "sphereOutOfMemory"),
    (sht_sphere.SphereError("bad file"), "sphereFailed"),
    (FileNotFoundError("gone"), "notFound"),
    # The wording of a real allocation failure that never became a SphereError:
    # only the job can recognise it, the cause chain does not cross the pipe.
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "sphereOutOfMemory"),
])
def test_the_job_names_its_failures(monkeypatch, raised, code):
    from backend.worker.jobs import JobError
    from backend.worker.jobs import sphere as sphere_job

    def _raise(*a):
        raise raised
    monkeypatch.setattr(sht_sphere, "_build_payload", _raise)
    with pytest.raises(JobError) as info:
        sphere_job.run({"path": "x", "max_bandwidth": 128, "target_size": 128}, _Ctx())
    assert info.value.code == code


def test_the_job_names_its_heavy_imports_for_the_worker():
    """The worker runs `preload` before it tells the backend it has the job, so
    torch's import counts as starting; without it the slow part of a cold start
    would be invisible to the slow-start note and the start limit."""
    from backend.worker.jobs import sphere as sphere_job
    sphere_job.preload()
    assert "torch" in sys.modules
    assert "backend.api.services.sht_pattern_renderer" in sys.modules


def test_the_job_does_not_dress_up_an_unrelated_error(monkeypatch):
    from backend.worker.jobs import sphere as sphere_job

    def _raise(*a):
        raise ValueError("a bug, not a memory problem")
    monkeypatch.setattr(sht_sphere, "_build_payload", _raise)
    with pytest.raises(ValueError):
        sphere_job.run({"path": "x", "max_bandwidth": 128, "target_size": 128}, _Ctx())


def _library_sht():
    for p in sorted(sht_sphere._SHT_DB.rglob("*.sht")):
        return p
    return None


@pytest.mark.parametrize("bw, tgt", [(128, 128), (256, 192)])
def test_the_worker_sphere_is_bit_identical_to_the_backend_sphere(monkeypatch, bw, tgt):
    sht = _library_sht()
    if sht is None:
        pytest.skip("no .sht in Database/EBSD_SHT_Database (a local library is needed)")
    local = sht_sphere._build_payload(str(sht), bw, tgt)
    monkeypatch.delenv("ORIENTA_NO_WORKER", raising=False)
    m = wm.WorkerManager(idle_s=600)
    try:
        remote = m.run("sphere", {"path": str(sht), "max_bandwidth": bw, "target_size": tgt})
    finally:
        pid = m.worker_pid
        m.shutdown()
        kill_quietly(pid)
    assert remote == local
