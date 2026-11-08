"""The backend side of the compute worker: start, reuse, retire, cancel, crash."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.worker.manager import (  # noqa: E402
    JobCancelled, JobFailed, UnknownJob, WorkerManager,
)
from tests.worker_helpers import gone_within, kill_quietly  # noqa: E402


@pytest.fixture
def mk(monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_JOBS", "1")
    made = []

    def _make(**kw):
        kw.setdefault("idle_s", 600)
        m = WorkerManager(**kw)
        made.append(m)
        return m

    yield _make
    for m in made:
        pid = m.worker_pid
        m.shutdown()
        kill_quietly(pid)


def _wait_status(m, jid, status, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        snap = m.get(jid)
        if snap and snap["status"] == status:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"{jid} never reached {status}: {m.get(jid)}")


def test_a_job_runs_and_returns_its_result(mk):
    m = mk()
    assert m.run("_echo", {"a": 1}) == {"a": 1}


def test_an_unknown_job_is_refused_at_submit(mk):
    with pytest.raises(UnknownJob):
        mk().submit("no-such-job", {})


def test_a_stopped_manager_refuses_instead_of_waiting_for_ever(mk):
    from backend.worker.manager import WorkerError
    m = mk()
    m.shutdown()
    with pytest.raises(WorkerError) as e:
        m.run("_echo", {})
    assert e.value.code == "workerStopped"


def test_a_light_job_reuses_the_warm_worker(mk):
    m = mk()
    first = m.run("_pid", {})["pid"]
    second = m.run("_pid", {})["pid"]
    assert first == second == m.worker_pid


def test_a_heavy_job_retires_the_worker(mk):
    m = mk()
    m.run("_echo", {})
    pid = m.worker_pid
    m.run("_sleep_heavy", {"s": 0.1})
    assert gone_within(pid, 5.0)
    assert m.run("_pid", {})["pid"] != pid


def test_an_idle_worker_is_retired(mk):
    m = mk(idle_s=1)
    m.run("_echo", {})
    pid = m.worker_pid
    assert gone_within(pid, 6.0)


def test_cancel_kills_a_running_job_and_the_next_one_runs(mk):
    m = mk()
    jid = m.submit("_sleep", {"s": 60})
    _wait_status(m, jid, "running")
    pid = m.worker_pid
    assert m.cancel(jid) is True
    snap = _wait_status(m, jid, "cancelled", timeout=10)
    assert snap["code"] is None
    assert gone_within(pid, 3.0)
    assert m.run("_echo", {"after": True}) == {"after": True}


def test_a_cooperative_job_is_cancelled_without_killing_the_worker(mk):
    m = mk()
    jid = m.submit("_sleep", {"s": 60, "cooperative": True})
    _wait_status(m, jid, "running")
    pid = m.worker_pid
    m.cancel(jid)
    _wait_status(m, jid, "cancelled", timeout=5)
    assert psutil.pid_exists(pid)


def test_a_queued_job_can_be_cancelled_before_it_starts(mk):
    m = mk()
    first = m.submit("_sleep", {"s": 2})
    second = m.submit("_echo", {})
    assert m.cancel(second) is True
    assert m.get(second)["status"] == "cancelled"
    _wait_status(m, first, "done", timeout=30)


def test_a_cancelled_heavy_job_does_not_block_the_next(mk):
    m = mk()
    heavy = m.submit("_sleep_heavy", {"s": 60})
    waiting = m.submit("_echo", {"n": 2})
    _wait_status(m, heavy, "running")
    pid = m.worker_pid
    m.cancel(heavy)
    assert _wait_status(m, waiting, "done", timeout=60)["result"] == {"n": 2}
    assert gone_within(pid, 3.0)


def test_run_with_a_cancelled_callback_raises_job_cancelled(mk):
    m = mk()
    t0 = time.monotonic()
    with pytest.raises(JobCancelled):
        m.run("_sleep", {"s": 60}, cancelled=lambda: time.monotonic() - t0 > 1.0)
    assert time.monotonic() - t0 < 10


def test_a_crash_is_reported_and_the_backend_goes_on(mk, caplog):
    m = mk()
    with caplog.at_level(logging.INFO):
        jid = m.submit("_crash", {"code": 3})
        snap = _wait_status(m, jid, "error", timeout=30)
    assert snap["code"] == "workerCrashed"
    assert "exit code 3" in snap["error"]
    crash_reports = [r for r in caplog.records
                     if r.levelno == logging.ERROR and "died (exit code 3)" in r.getMessage()]
    assert crash_reports and "about to crash on purpose" in crash_reports[0].getMessage()
    assert m.run("_echo", {"ok": 1}) == {"ok": 1}


def test_worker_logs_land_in_the_log_file_with_their_name(mk, tmp_path):
    """A record from the worker appears in the log FILE under its own
    logger name. orienta.log is a handler on the root logger; so is this one."""
    log_file = tmp_path / "worker.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        mk().run("_log", {"logger": "backend.worker.jobs.sphere", "msg": "Kugel fertig"})
        time.sleep(0.2)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()
    text = log_file.read_text(encoding="utf-8")
    assert "backend.worker.jobs.sphere | [worker " in text and "Kugel fertig" in text


def test_a_coded_job_failure_keeps_its_code(mk):
    m = mk()
    with pytest.raises(JobFailed) as e:
        m.run("_raise", {"code": "sphereOutOfMemory", "msg": "too big"})
    assert e.value.code == "sphereOutOfMemory" and e.value.message == "too big"


def test_worker_logs_reach_the_backend_under_their_own_name(mk, caplog):
    m = mk()
    with caplog.at_level(logging.INFO):
        m.run("_log", {"logger": "backend.api.services.sht_sphere", "msg": "from the process"})
        time.sleep(0.2)
    recs = [r for r in caplog.records if r.name == "backend.api.services.sht_sphere"]
    assert recs and "from the process" in recs[0].getMessage()


def test_progress_is_visible_while_it_runs(mk):
    m = mk()
    jid = m.submit("_progress", {"n": 20, "dt": 0.1})
    end = time.monotonic() + 30
    seen = None
    while time.monotonic() < end:
        snap = m.get(jid)
        if snap["progress"] is not None and snap["status"] == "running":
            seen = snap
            break
        time.sleep(0.02)
    assert seen is not None and seen["message"].startswith("step ")


def test_a_big_result_arrives_through_a_file(mk):
    m = mk()
    from backend.worker.protocol import INLINE_LIMIT
    out = m.run("_big", {"bytes": INLINE_LIMIT + 1000})
    assert len(out["blob"]) == INLINE_LIMIT + 1000


def test_a_path_with_umlauts_arrives_intact(mk, tmp_path):
    m = mk()
    f = tmp_path / "Größe [Pearson] {20kV} Ä.sht"
    f.write_bytes(b"x")
    assert m.run("_stat", {"path": str(f)}) == {"path": str(f), "exists": True}


def test_a_slow_start_is_reported_while_waiting(mk, monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_START_DELAY", "3")
    m = mk(slow_start_s=1)
    jid = m.submit("_echo", {})
    end = time.monotonic() + 10
    while time.monotonic() < end:
        snap = m.get(jid)
        if snap["message_code"] == "workerSlowStart":
            break
        time.sleep(0.05)
    assert m.get(jid)["message_code"] == "workerSlowStart"
    _wait_status(m, jid, "done", timeout=30)


def test_a_worker_that_never_starts_is_refused_with_a_code(mk, monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_START_DELAY", "10")
    m = mk(slow_start_s=0.5, start_limit_s=2)
    jid = m.submit("_echo", {})
    snap = _wait_status(m, jid, "error", timeout=20)
    assert snap["code"] == "workerStartFailed"


def _wait_slow_note(m, jid, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        snap = m.get(jid)
        if snap["message_code"] == "workerSlowStart" or snap["status"] in ("done", "error"):
            return snap
        time.sleep(0.05)
    return m.get(jid)


def test_a_slow_import_after_ready_is_reported_as_a_slow_start(mk, monkeypatch):
    """`ready` is sent before any job module is imported; the slow part of a cold
    start (torch behind a virus scanner) comes after it and must still count."""
    monkeypatch.setenv("ORIENTA_WORKER_TEST_IMPORT_S", "4")
    m = mk(slow_start_s=1.5)
    jid = m.submit("_slow_import", {"x": 1})
    snap = _wait_slow_note(m, jid)
    # "starting" if the worker took longer than slow_start_s to say ready on a busy runner
    assert snap["message_code"] == "workerSlowStart" and snap["status"] in ("starting", "running"), snap
    done = _wait_status(m, jid, "done", timeout=30)
    assert done["message_code"] is None          # the note goes away once the job has started
    assert done["result"] == {"x": 1}


def test_a_job_that_computes_longer_than_the_start_limit_is_not_killed(mk):
    """`imported` closes the start window: a long computation without progress
    messages is a running job, not a failed start."""
    m = mk(slow_start_s=0.5, start_limit_s=2.0)
    jid = m.submit("_sleep", {"s": 5})
    snap = _wait_status(m, jid, "done", timeout=30)
    assert snap["result"] == {"slept": 5}
    assert snap["message_code"] is None


def test_the_start_window_of_a_warm_worker_opens_when_it_gets_the_job(mk, monkeypatch):
    """Not when the worker was launched: an idle worker older than the start limit
    must still be allowed its job's imports."""
    monkeypatch.setenv("ORIENTA_WORKER_TEST_PRELOAD_S", "1.0")
    m = mk(slow_start_s=0.5, start_limit_s=2.0)
    assert m.run("_echo", {}) == {}
    time.sleep(3.0)                               # the warm worker is now older than the limit
    jid = m.submit("_slow_import", {"a": 1})
    assert _wait_status(m, jid, "done", timeout=20)["result"] == {"a": 1}


def test_a_cancel_that_lands_at_the_start_limit_is_a_cancel_not_a_failed_start(mk, monkeypatch):
    """The decision at the limit is taken under the lock cancel() takes. The cancel
    is forced into the gap between the loop's check and that decision."""
    from backend.worker import manager as manager_module
    monkeypatch.setenv("ORIENTA_WORKER_TEST_PRELOAD_S", "60")
    original = manager_module.WorkerManager._watch_start_window
    fired = []

    def _cancel_then_decide(self, rec, worker, window_start):
        if time.monotonic() - window_start > self.start_limit_s and not fired:
            fired.append(self.cancel(rec.id))
        return original(self, rec, worker, window_start)

    monkeypatch.setattr(manager_module.WorkerManager, "_watch_start_window", _cancel_then_decide)
    m = mk(slow_start_s=0.3, start_limit_s=2.0)
    jid = m.submit("_slow_import", {})
    snap = _wait_status(m, jid, "cancelled", timeout=30)
    assert fired == [True] and snap["code"] is None, (fired, snap)


def test_a_first_message_at_the_start_limit_is_not_a_failed_start(mk, monkeypatch):
    from backend.worker import manager as manager_module
    monkeypatch.setenv("ORIENTA_WORKER_TEST_PRELOAD_S", "3")
    original = manager_module.WorkerManager._watch_start_window
    fired = []

    def _started_then_decide(self, rec, worker, window_start):
        if time.monotonic() - window_start > self.start_limit_s and not fired:
            fired.append(True)
            rec.started.set()                     # the worker's `imported` arrives right now
        return original(self, rec, worker, window_start)

    monkeypatch.setattr(manager_module.WorkerManager, "_watch_start_window", _started_then_decide)
    m = mk(slow_start_s=0.3, start_limit_s=2.0)
    jid = m.submit("_slow_import", {"b": 2})
    snap = _wait_status(m, jid, "done", timeout=30)
    assert fired and snap["result"] == {"b": 2}, (fired, snap)


def test_the_worker_does_not_run_in_utf8_mode(mk, monkeypatch):
    """PYTHONUTF8 would change what open() reads by default in the worker and in
    everything it starts; only the standard streams are forced to UTF-8."""
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    out = mk().run("_encodings", {})
    assert out["utf8_mode"] == 0, out
    assert out["stderr"].lower().replace("-", "") == "utf8", out


def test_slow_preload_imports_count_as_starting_too(mk, monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_PRELOAD_S", "4")
    m = mk(slow_start_s=1.5)
    jid = m.submit("_slow_import", {})
    assert _wait_slow_note(m, jid)["message_code"] == "workerSlowStart"
    _wait_status(m, jid, "done", timeout=30)


def test_an_import_that_never_finishes_is_a_failed_start(mk, monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_IMPORT_S", "60")
    m = mk(slow_start_s=0.5, start_limit_s=3)
    jid = m.submit("_slow_import", {})
    end = time.monotonic() + 15
    while time.monotonic() < end and not m.worker_pid:
        time.sleep(0.05)
    pid = m.worker_pid
    snap = _wait_status(m, jid, "error", timeout=30)
    assert snap["code"] == "workerStartFailed", snap
    assert pid and gone_within(pid, 5.0)          # the stuck worker is not left behind


def test_a_warm_worker_never_shows_the_slow_start_note(mk):
    m = mk(slow_start_s=0.4)
    m.run("_echo", {})                            # now the worker is warm
    jid = m.submit("_sleep", {"s": 1.6, "cooperative": True})
    end = time.monotonic() + 1.4
    while time.monotonic() < end:                 # well past slow_start_s, job running
        assert m.get(jid)["message_code"] is None
        time.sleep(0.05)
    m.cancel(jid)
    _wait_status(m, jid, "cancelled", timeout=15)


def test_worker_stderr_is_read_as_utf8(mk, caplog):
    """Windows pipes carry the ANSI code page unless the worker is told otherwise."""
    m = mk()
    with caplog.at_level(logging.INFO, logger="backend.worker.stderr"):
        m.run("_umlaut", {})
        end = time.monotonic() + 5
        while time.monotonic() < end and not any("Größe" in r.getMessage() for r in caplog.records):
            time.sleep(0.05)
    assert any("Größe [Pearson] {20kV} ä ö ü" in r.getMessage() for r in caplog.records)


def test_a_worker_that_died_while_idle_is_replaced(mk):
    m = mk()
    m.run("_echo", {})
    pid = m.worker_pid
    psutil.Process(pid).kill()
    assert gone_within(pid, 3.0)
    assert m.run("_echo", {"again": 1}) == {"again": 1}
    assert m.worker_pid != pid


def test_a_finished_heavy_job_gives_its_memory_back(mk):
    m = mk()
    me = psutil.Process()
    rss_before = me.memory_info().rss
    out = m.run("_alloc", {"mb": 1500})
    assert out["rss_mb"] > 1400                    # the worker really held it
    pid = m.worker_pid
    assert gone_within(pid, 5.0)                   # heavy: retired, so the OS has it back
    assert me.memory_info().rss - rss_before < 150 * 2**20   # the backend did not grow


def _record_popen(monkeypatch):
    """Wrap ``subprocess.Popen`` as the manager sees it; return the list of keyword arguments."""
    import backend.worker.manager as manager_module
    real = subprocess.Popen
    calls = []

    def recorder(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(manager_module.subprocess, "Popen", recorder)
    return calls


@pytest.mark.skipif(sys.platform != "win32", reason="creation flags are a Windows thing")
def test_the_worker_is_started_suspended_when_the_job_object_is_used(mk, monkeypatch):
    """R7: a launcher must not outrun the job, so the worker starts suspended and is resumed."""
    from backend.worker import winjob
    calls = _record_popen(monkeypatch)
    m = mk()
    assert m.run("_echo", {"x": 1}) == {"x": 1}          # resumed: it answered
    flags = calls[0]["creationflags"]
    assert flags & winjob.CREATE_SUSPENDED
    assert flags & subprocess.CREATE_NO_WINDOW
    assert winjob.is_in_job(m.worker_pid)


@pytest.mark.skipif(sys.platform != "win32", reason="creation flags are a Windows thing")
def test_without_the_job_object_the_worker_is_never_started_suspended(mk, monkeypatch):
    """Nobody would resume it: a suspended start without the job would hang for ever."""
    from backend.worker import winjob
    monkeypatch.setenv("ORIENTA_WORKER_TEST_DISABLE", "jobobject")
    calls = _record_popen(monkeypatch)
    m = mk()
    assert m.run("_echo", {"x": 2}) == {"x": 2}
    flags = calls[0]["creationflags"]
    assert not flags & winjob.CREATE_SUSPENDED
    assert flags & subprocess.CREATE_NO_WINDOW


@pytest.mark.skipif(sys.platform != "win32", reason="the venv launcher is a Windows thing")
def test_a_launcher_in_between_is_seen_through(mk, tmp_path):
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import subprocess, sys\n"
        "sys.exit(subprocess.call([sys.executable, '-m', 'backend.worker']))\n",
        encoding="utf-8")
    m = mk(argv=[sys.executable, str(launcher)])
    jid = m.submit("_sleep", {"s": 60})
    _wait_status(m, jid, "running")
    real = m.worker_pid
    from backend.worker import winjob
    assert winjob.is_in_job(real)
    m.cancel(jid)
    _wait_status(m, jid, "cancelled", timeout=10)
    assert gone_within(real, 3.0)


# ---- races between cancel, hand-over, kill and shutdown ----------------------------

def test_a_job_cancelled_while_queued_is_not_resurrected(mk, monkeypatch):
    """The dispatcher has picked the job up; a cancel lands before it takes the lock."""
    import backend.worker.manager as mm
    entered, gate = threading.Event(), threading.Event()
    real_run_one = mm.WorkerManager._run_one

    def held(self, rec):
        entered.set()
        gate.wait(15)
        return real_run_one(self, rec)

    monkeypatch.setattr(mm.WorkerManager, "_run_one", held)
    m = mk()
    jid = m.submit("_echo", {"must": "not run"})
    assert entered.wait(15)
    assert m.cancel(jid) is True
    gate.set()
    assert m.run("_echo", {"next": 1}) == {"next": 1}      # the dispatcher is past the first one
    snap = m.get(jid)
    assert snap["status"] == "cancelled" and "result" not in snap


def test_a_job_is_not_running_until_the_worker_has_it(mk, monkeypatch):
    """A cancel that sees "running" must not be able to overtake the job on its way to the worker."""
    import backend.worker.manager as mm
    at_send, go = threading.Event(), threading.Event()
    real_send = mm._Worker.send

    def held_send(self, msg):
        if "job" in msg:
            at_send.set()
            go.wait(15)
        return real_send(self, msg)

    monkeypatch.setattr(mm._Worker, "send", held_send)
    m = mk()
    jid = m.submit("_sleep", {"s": 60})
    assert at_send.wait(30)
    assert m.get(jid)["status"] == "starting"              # not "running": the worker has no job yet
    assert m.cancel(jid) is True                           # the cancel message goes first
    pid = m.worker_pid
    go.set()
    _wait_status(m, jid, "cancelled", timeout=10)
    assert psutil.pid_exists(pid)                          # answered by the worker, no kill needed


@pytest.mark.skipif(sys.platform != "win32", reason="the venv launcher is a Windows thing")
def test_a_kill_before_ready_takes_the_whole_launcher_tree(mk, tmp_path, monkeypatch):
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import subprocess, sys\n"
        "sys.exit(subprocess.call([sys.executable, '-m', 'backend.worker']))\n",
        encoding="utf-8")
    monkeypatch.setenv("ORIENTA_WORKER_TEST_START_DELAY", "10")
    m = mk(argv=[sys.executable, str(launcher)], slow_start_s=0.5, start_limit_s=3)
    jid = m.submit("_echo", {})
    tree = []
    end = time.monotonic() + 15
    try:
        while time.monotonic() < end and not tree:        # before `ready`: only the launcher is known
            w = m._worker
            if w is not None:
                try:
                    tree = psutil.Process(w.proc.pid).children(recursive=True)
                except psutil.Error:
                    pass
            time.sleep(0.05)
        assert tree, "the launcher never started its child"
        launcher_pid = m._worker.proc.pid
        assert _wait_status(m, jid, "error", timeout=20)["code"] == "workerStartFailed"
        assert gone_within(launcher_pid, 3.0)
        for p in tree:
            assert gone_within(p.pid, 3.0), f"{p.pid} survived the start failure"
    finally:
        for p in tree:
            kill_quietly(p.pid)


def test_shutdown_finishes_every_job_it_did_not_run(mk):
    m = mk()
    running = m.submit("_sleep", {"s": 60})
    _wait_status(m, running, "running")
    queued = [m.submit("_echo", {"n": i}) for i in range(3)]
    m.shutdown()
    for jid in queued:
        snap = m.wait(jid, 15)
        assert snap["status"] == "error" and snap["code"] == "workerStopped", snap
    assert m.wait(running, 15)["status"] in ("error", "cancelled")


def test_run_never_waits_for_ever_across_shutdown(mk):
    from backend.worker.manager import WorkerError
    m = mk()
    m.run("_echo", {})                                     # a warm worker: submits are fast
    outcomes, stop = [], threading.Event()

    def caller():
        while not stop.is_set():
            try:
                outcomes.append(m.run("_echo", {}))
            except WorkerError as e:
                outcomes.append(e.code)
                return

    threads = [threading.Thread(target=caller, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    m.shutdown()
    deadline = time.monotonic() + 20
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    stop.set()
    assert not [t for t in threads if t.is_alive()], "a run() is still waiting after shutdown"


class _HookedLock:
    """The manager's lock, with one hook that runs just before the next acquisition."""

    def __init__(self, real):
        self.real = real
        self.hook = None

    def __enter__(self):
        hook, self.hook = self.hook, None
        if hook is not None:
            hook()
        return self.real.__enter__()

    def __exit__(self, *exc):
        return self.real.__exit__(*exc)


def test_a_shutdown_between_the_stopped_check_and_the_queueing_cannot_strand_a_job(mk):
    from backend.worker.manager import WorkerError
    m = mk()
    m.run("_echo", {})
    lock = _HookedLock(m._lock)
    m._lock = lock

    def stop_completely():                    # the whole shutdown, just before submit takes the lock
        m.shutdown()
        m._thread.join(15)

    lock.hook = stop_completely
    try:
        jid = m.submit("_echo", {})
    except WorkerError as e:
        assert e.code == "workerStopped"
        return
    assert m.wait(jid, 5)["status"] in ("done", "error", "cancelled")


def test_a_job_that_finishes_before_the_hand_over_returns_stays_done(mk, monkeypatch):
    """The reader thread may finish the job while the dispatcher is still in send()'s caller."""
    import backend.worker.manager as mm
    real_send = mm._Worker.send
    box = {}

    def send_then_wait_for_the_answer(self, msg):
        ok = real_send(self, msg)
        if "job" in msg:
            end = time.monotonic() + 10
            while time.monotonic() < end and box["m"].get(msg["id"])["status"] != "done":
                time.sleep(0.01)
        return ok

    monkeypatch.setattr(mm._Worker, "send", send_then_wait_for_the_answer)
    m = mk()
    box["m"] = m
    assert m.run("_echo", {"fast": 1}) == {"fast": 1}
    time.sleep(0.2)                           # the dispatcher is past the hand-over by now
    snap = m.get(next(iter(m._records)))
    assert snap["status"] == "done" and snap["result"] == {"fast": 1}


# ---- "Release GPU" reaches the warm worker -------------------------------------------

def test_release_ends_a_warm_worker_with_no_job_and_the_next_job_runs(mk):
    m = mk()
    pid = m.run("_pid", {})["pid"]
    assert m.worker_pid == pid                   # warm, no job
    assert m.release_idle_worker() is True
    assert gone_within(pid, 5.0)
    assert m.worker_pid is None
    assert m.run("_pid", {})["pid"] != pid       # a fresh one takes the next job


def test_release_leaves_a_worker_alone_that_is_computing(mk):
    m = mk()
    jid = m.submit("_sleep", {"s": 2})
    _wait_status(m, jid, "running")
    pid = m.worker_pid
    assert m.release_idle_worker() is False
    assert m.worker_pid == pid and psutil.pid_exists(pid)
    snap = _wait_status(m, jid, "done", timeout=30)      # the job ended on its own
    assert snap["code"] is None
    assert m.worker_pid == pid                           # and the worker is still the same one


def test_release_of_a_worker_that_had_already_ended_tidies_up_but_says_false(mk):
    """Nothing was freed by it, so the page must not say "worker released"."""
    m = mk()
    pid = m.run("_pid", {})["pid"]
    w = m._worker
    kill_quietly(pid)
    assert w.exited.wait(10)                     # dead, but still the manager's current worker
    assert m._worker is w
    from backend.worker.manager import _ReleaseRequest
    req = _ReleaseRequest()
    m._release(req)                              # what the dispatcher does with a request it picks up
    assert req.done.is_set()
    assert req.released is False
    assert m._worker is None                     # tidied up all the same


def test_release_without_a_worker_starts_none(mk):
    m = mk()
    assert m.release_idle_worker() is False
    assert m.worker_pid is None
    time.sleep(0.3)
    assert m.worker_pid is None


def test_the_module_function_does_nothing_without_a_manager(monkeypatch):
    import backend.worker.manager as mm
    monkeypatch.delenv("ORIENTA_NO_WORKER", raising=False)
    monkeypatch.setattr(mm, "_MANAGER", None)
    assert mm.release_idle_worker() is False
    assert mm._MANAGER is None                   # and it did not create one


def test_the_module_function_does_nothing_when_the_worker_is_off(mk, monkeypatch):
    import backend.worker.manager as mm
    m = mk()
    pid = m.run("_pid", {})["pid"]
    monkeypatch.setattr(mm, "_MANAGER", m)
    monkeypatch.setenv("ORIENTA_NO_WORKER", "1")
    assert mm.release_idle_worker() is False
    assert psutil.pid_exists(pid) and m.worker_pid == pid


def test_the_module_function_ends_the_idle_worker_of_the_manager(mk, monkeypatch):
    import backend.worker.manager as mm
    m = mk()
    pid = m.run("_pid", {})["pid"]
    monkeypatch.setattr(mm, "_MANAGER", m)
    monkeypatch.delenv("ORIENTA_NO_WORKER", raising=False)
    assert mm.release_idle_worker() is True
    assert gone_within(pid, 5.0)


def test_a_job_that_arrived_after_the_request_keeps_the_worker(mk, monkeypatch):
    """The check is repeated in the dispatcher: only there is it final."""
    import backend.worker.manager as mm
    entered, gate = threading.Event(), threading.Event()
    real_run_one = mm.WorkerManager._run_one

    def held(self, rec):
        entered.set()
        gate.wait(15)
        return real_run_one(self, rec)

    m = mk()
    pid = m.run("_pid", {})["pid"]
    monkeypatch.setattr(mm.WorkerManager, "_run_one", held)
    jid = m.submit("_echo", {"n": 1})
    assert entered.wait(15)                      # the job is the dispatcher's, not yet at the worker
    req = mm._ReleaseRequest()
    m._release(req)                              # what the dispatcher does with a request it picks up now
    assert req.done.is_set() and req.released is False
    assert psutil.pid_exists(pid)
    gate.set()
    assert _wait_status(m, jid, "done")["result"] == {"n": 1}
    assert m.worker_pid == pid


def test_a_release_request_still_queued_at_shutdown_is_answered(mk, monkeypatch):
    """Whoever waits in release_idle_worker() must not wait for its whole timeout
    for a dispatcher that is gone: the dispatcher answers what is left in its queue."""
    import backend.worker.manager as mm
    entered, gate = threading.Event(), threading.Event()

    def held(self, rec):
        entered.set()
        gate.wait(15)

    monkeypatch.setattr(mm.WorkerManager, "_run_one", held)
    m = mk()
    jid = m.submit("_echo", {})
    assert entered.wait(15)                      # the dispatcher is busy with the job
    req = mm._ReleaseRequest()
    m._queue.put(req)                            # as release_idle_worker() would have queued it
    m.shutdown()
    assert not req.done.is_set()                 # premise: nobody has looked at it yet
    gate.set()
    assert req.done.wait(5)                      # answered, not left hanging
    assert req.released is False
    assert _wait_status(m, jid, "error")["code"] == "workerStopped"


# ---- a job cancelled while the worker is still starting ------------------------------

def _wait_starting_worker(m, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        w = m._worker
        if w is not None and not w.ready.is_set():
            return w
        time.sleep(0.02)
    raise AssertionError("no worker was starting")


def test_a_cancel_during_the_start_ends_the_job_but_not_the_worker(mk, monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_START_DELAY", "3")
    m = mk()
    first = m.submit("_echo", {"n": 1})
    w = _wait_starting_worker(m)
    assert m.cancel(first) is True
    _wait_status(m, first, "cancelled", timeout=10)
    assert m._worker is w and w.proc.poll() is None        # the import goes on
    # The next job is taken by the same process.
    assert m.run("_echo", {"n": 2}) == {"n": 2}
    assert m._worker is w and w.proc.poll() is None
    assert m.get(first)["status"] == "cancelled"


def test_a_worker_left_starting_by_a_cancel_is_retired_when_idle(mk, monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_START_DELAY", "2")
    m = mk(idle_s=1)
    jid = m.submit("_echo", {})
    w = _wait_starting_worker(m)
    m.cancel(jid)
    _wait_status(m, jid, "cancelled", timeout=10)
    assert w.exited.wait(20)                               # the idle rule still applies
    end = time.monotonic() + 5
    while m._worker is not None and time.monotonic() < end:
        time.sleep(0.02)
    assert m._worker is None


def _leave_a_worker_stuck_starting(m, monkeypatch):
    """A cancelled job leaves a worker that will not say ready for 30 s. Afterwards
    new workers start without a delay."""
    monkeypatch.setenv("ORIENTA_WORKER_TEST_START_DELAY", "30")
    jid = m.submit("_echo", {})
    w = _wait_starting_worker(m)
    m.cancel(jid)
    _wait_status(m, jid, "cancelled", timeout=10)
    assert m._worker is w
    monkeypatch.setenv("ORIENTA_WORKER_TEST_START_DELAY", "0")
    return w


class _CountingEvent(threading.Event):
    """An Event that counts how often it is waited on."""

    def __init__(self):
        super().__init__()
        self.waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        return super().wait(timeout)


def test_a_worker_left_starting_still_has_a_start_limit(mk, monkeypatch):
    """The limit counts from the worker's launch and is not stretched by the next
    job: past it, the worker is ended at once instead of being waited for again."""
    m = mk(start_limit_s=2)
    w = _leave_a_worker_stuck_starting(m, monkeypatch)
    time.sleep(2.2)                                        # past the limit, counted from the launch
    # The dispatcher looks at the worker every 0.1 s (`ready.wait(0.1)`). Counting
    # those looks tells how long it waited without a clock: one look when the
    # limit is already passed, about 20 more when the limit starts at the job.
    looks = _CountingEvent()
    w.ready = looks
    second = m.submit("_echo", {"n": 2})
    assert _wait_status(m, second, "done", timeout=20)["result"] == {"n": 2}
    assert w.exited.is_set()                               # ended, not waited for another 2 s
    assert m._worker is not w and m._worker.proc.pid != w.proc.pid
    assert 1 <= looks.waits <= 3, f"the dispatcher waited {looks.waits} times for the old worker"


def test_a_worker_left_starting_that_dies_costs_the_next_job_nothing(mk, monkeypatch):
    m = mk(start_limit_s=60)
    w = _leave_a_worker_stuck_starting(m, monkeypatch)
    second = m.submit("_pid", {})
    time.sleep(0.5)                                        # the job is waiting for that worker
    assert m.get(second)["status"] == "starting"
    w.kill()
    snap = _wait_status(m, second, "done", timeout=20)     # the second attempt, on a fresh worker
    assert snap["result"]["pid"] not in (w.proc.pid, w.real_pid)
    assert m._worker is not w


class _GatedEvent(threading.Event):
    """An Event whose `wait` first waits for ``gate``: it holds the dispatcher at
    its next look at the worker, so the test decides what has happened by then."""

    def __init__(self, gate):
        super().__init__()
        self._gate = gate

    def wait(self, timeout=None):
        self._gate.wait(30)
        return super().wait(timeout)


def _hold_the_dispatcher(w):
    """Hold the dispatcher at the next look at the starting worker ``w``; return
    the gate that lets it go on. Returns once the dispatcher is held."""
    gate = threading.Event()
    w.ready = _GatedEvent(gate)
    time.sleep(0.4)             # its current 0.1 s wait has ended, it is at the gate
    return gate


def test_a_job_cancelled_when_the_inherited_worker_dies_starts_no_worker(mk, monkeypatch):
    """The cancel and the death of the worker left starting by an earlier job land
    together: the job is cancelled, and no second worker is started for it (it
    would sit there until the idle limit)."""
    calls = _record_popen(monkeypatch)
    m = mk(start_limit_s=60)
    w = _leave_a_worker_stuck_starting(m, monkeypatch)
    started = len(calls)                                   # the stuck worker's own launch
    second = m.submit("_pid", {})
    time.sleep(0.5)
    assert m.get(second)["status"] == "starting"           # waiting for that worker
    gate = _hold_the_dispatcher(w)
    m.cancel(second)
    w.kill()
    assert w.exited.wait(10)
    gate.set()                                             # now the dispatcher sees both
    snap = _wait_status(m, second, "cancelled", timeout=5)
    assert snap["status"] == "cancelled"
    time.sleep(0.5)                                        # room for a start that should not come
    assert len(calls) == started                           # no worker was launched
    assert m._worker is None


def test_a_shutdown_while_the_job_waits_for_an_inherited_worker_starts_no_worker(mk, monkeypatch):
    """`shutdown()` ends the inherited worker the job waits for. The job gets the
    answer a stopped manager gives, at once; no new worker is started, which would
    take seconds and then run the job."""
    calls = _record_popen(monkeypatch)
    m = mk(start_limit_s=60)
    w = _leave_a_worker_stuck_starting(m, monkeypatch)
    started = len(calls)
    second = m.submit("_sleep", {"s": 20})
    time.sleep(0.5)
    assert m.get(second)["status"] == "starting"
    t0 = time.monotonic()
    m.shutdown()
    snap = _wait_status(m, second, "error", timeout=3)     # a worker start alone takes longer
    assert snap["code"] == "workerStopped"
    assert time.monotonic() - t0 < 3
    time.sleep(0.5)
    assert len(calls) == started                           # no worker was launched
    assert w.exited.wait(5)


def _fail_the_send_of_the_job(m, w, before_failing):
    """Make the first attempt's write of the job to the warm worker ``w`` fail for
    real (its stdin is closed first), after ``before_failing()`` ran: that is the
    moment between the manager's look at the job and the write, which is where a
    cancel or another answer can land."""
    real_send = w.send

    def send(msg):
        if "job" in msg:
            w.close_stdin()
            before_failing()
            return real_send(msg)          # False: the pipe is closed
        return real_send(msg)

    w.send = send


def test_a_job_cancelled_while_its_send_fails_starts_no_second_worker(mk, monkeypatch):
    """The retry after a failed send (not the `_RETRY` of `_await_ready`): the write
    of the job to a warm worker fails, and a cancel landed since the manager last
    looked. The job is cancelled and no new worker is started for it (it would sit
    there until the idle limit)."""
    calls = _record_popen(monkeypatch)
    m = mk()
    m.run("_echo", {})                                     # a warm worker
    w = m._worker
    started = len(calls)
    box = {}
    _fail_the_send_of_the_job(m, w, lambda: m.cancel(box["jid"]))
    box["jid"] = m.submit("_echo", {"n": 1})
    snap = _wait_status(m, box["jid"], "cancelled", timeout=15)
    assert snap["status"] == "cancelled"
    assert w.exited.wait(10)
    time.sleep(0.5)                                        # room for a start that should not come
    assert len(calls) == started                           # no worker was launched
    assert m._worker is None


def test_a_job_that_is_answered_while_its_send_fails_starts_no_second_worker(mk, monkeypatch):
    """Same moment, but the job already has its answer (here an error from another
    path). The answer stays as it is and no worker is started for a finished job."""
    calls = _record_popen(monkeypatch)
    m = mk()
    m.run("_echo", {})
    w = m._worker
    started = len(calls)
    box = {}

    def answer_elsewhere():
        m._finish(m._records[box["jid"]], "error", code="answeredElsewhere",
                  error="answered by another path")

    _fail_the_send_of_the_job(m, w, answer_elsewhere)
    box["jid"] = m.submit("_echo", {"n": 1})
    snap = _wait_status(m, box["jid"], "error", timeout=15)
    assert w.exited.wait(10)
    time.sleep(0.5)
    assert snap["code"] == "answeredElsewhere"
    assert snap["error"] == "answered by another path"
    final = m.get(box["jid"])
    assert (final["status"], final["code"], final["error"]) == (
        "error", "answeredElsewhere", "answered by another path")
    assert len(calls) == started
    assert m._worker is None


def test_a_worker_the_job_started_itself_still_fails_the_job(mk, monkeypatch):
    """Only an inherited worker gets the second attempt: here the fresh worker is as
    stuck as the old one, and runs into a start limit of its own."""
    monkeypatch.setenv("ORIENTA_WORKER_TEST_START_DELAY", "30")
    m = mk(start_limit_s=2)
    jid = m.submit("_echo", {})
    w = _wait_starting_worker(m)
    m.cancel(jid)
    _wait_status(m, jid, "cancelled", timeout=10)
    time.sleep(2.2)
    t0 = time.monotonic()
    second = m.submit("_echo", {})
    snap = _wait_status(m, second, "error", timeout=20)
    assert snap["code"] == "workerStartFailed"
    assert time.monotonic() - t0 > 1.8                     # the fresh worker had its own 2 s
    assert w.exited.is_set()
    assert m._worker is None


def test_a_job_that_computes_in_the_worker_is_still_ended_hard(mk):
    m = mk()
    jid = m.submit("_sleep", {"s": 60})
    _wait_status(m, jid, "running")
    w = m._worker
    m.cancel(jid)
    _wait_status(m, jid, "cancelled", timeout=10)
    assert w.exited.wait(5)


def test_the_worker_gets_the_same_start_value_that_it_compares_with(mk, monkeypatch, tmp_path):
    """The backend pins itself with process_watch.started_at, so the worker, which
    compares with process_watch too, reads the same kind of number. Here the Linux
    route is faked (tick count 123456 by hand), where create_time() would differ."""
    import backend.process_watch as pw
    folder = tmp_path / str(os.getpid())
    folder.mkdir()
    (folder / "stat").write_bytes(
        (f"{os.getpid()} (x) S " + " ".join(str(n) for n in range(4, 22)) + " 123456 9\n").encode())
    monkeypatch.setattr(pw, "_PROC_ROOT", str(tmp_path))
    monkeypatch.setattr(pw, "_uses_proc", lambda: True)
    calls = _record_popen(monkeypatch)
    m = mk(argv=[sys.executable, "-c", "import sys; sys.exit(0)"])
    jid = m.submit("_echo", {})
    _wait_status(m, jid, "error", timeout=20)              # this "worker" never says ready
    assert calls[0]["env"]["ORIENTA_WORKER_PARENT_START"] == "123456.0"


# ---- kill() leaves a worker alone that is already gone ---------------------------------

class _FakeProc:
    """Enough of subprocess.Popen for _Worker; `poll()` answers what the test says."""

    def __init__(self, pid, returncode):
        self.pid = pid
        self.returncode = returncode
        self.stdout = self.stderr = iter(())

    def poll(self):
        return self.returncode


def _quiet_worker(monkeypatch, pid, returncode):
    """A _Worker whose reader threads do nothing, so `exited` stays as the test sets it."""
    import backend.worker.manager as mm
    monkeypatch.setattr(mm._Worker, "_read_stdout", lambda self, on_message: None)
    monkeypatch.setattr(mm._Worker, "_read_stderr", lambda self: None)
    return mm._Worker(_FakeProc(pid, returncode), lambda *a: None)


class _RecordedProcess:
    """Stands for psutil.Process in the manager: records whom it was asked about."""
    asked: list = []
    killed: list = []

    def __init__(self, pid):
        type(self).asked.append(pid)
        self.pid = pid

    def children(self, recursive=False):
        return []

    def kill(self):
        type(self).killed.append(self.pid)


@pytest.fixture
def recorded_psutil(monkeypatch):
    import backend.worker.manager as mm
    _RecordedProcess.asked, _RecordedProcess.killed = [], []
    monkeypatch.setattr(mm.psutil, "Process", _RecordedProcess)
    return _RecordedProcess


def test_kill_sends_no_signal_to_a_worker_that_has_exited(monkeypatch, recorded_psutil):
    w = _quiet_worker(monkeypatch, 4_000_001, None)         # poll() would say "running"
    w.real_pid = 4_000_002
    w.exited.set()
    w.kill()
    assert recorded_psutil.asked == [] and recorded_psutil.killed == []


def test_kill_sends_no_signal_when_the_process_has_a_return_code(monkeypatch, recorded_psutil):
    w = _quiet_worker(monkeypatch, 4_000_003, 1)            # reaped, `exited` not yet set
    w.real_pid = 4_000_003
    w.kill()
    assert recorded_psutil.asked == [] and recorded_psutil.killed == []


def test_kill_still_ends_the_child_of_a_launcher_that_is_gone(monkeypatch, recorded_psutil):
    w = _quiet_worker(monkeypatch, 4_000_004, 0)            # the launcher is dead ...
    w.real_pid = 4_000_005                                  # ... the real worker may not be
    w.kill()
    assert recorded_psutil.killed == [4_000_005]


def test_kill_ends_a_running_worker(monkeypatch, recorded_psutil):
    w = _quiet_worker(monkeypatch, 4_000_006, None)
    w.real_pid = 4_000_006
    w.kill()
    assert recorded_psutil.killed == [4_000_006]
