"""A compute worker never outlives the backend that started it.

A stand-in backend starts a worker running a long job and is then killed
hard. Each guard is proven ON ITS OWN by switching the others off: the
end-of-stdin check and the PID poll would otherwise hide a broken OS guard.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.worker_helpers import ROOT, gone_within, kill_quietly  # noqa: E402

# ORIENTA_PROXY_SPIN=n keeps n pure-Python threads busy while the worker is
# being started. A backend is never idle at that moment, and GIL contention
# is what let a launcher's real interpreter run before it was inside the job
# object (measured: 25 of 25 starts without CREATE_SUSPENDED).
# Detecting that mutation is PROBABILISTIC: with eight threads, 15 of 15 runs
# without CREATE_SUSPENDED failed and 0 of 15 runs of the unchanged code did.
# Fewer threads detect it less often (four: 9 of 15), more make the test flaky.
SPINNERS = 8
READ_TIMEOUT_S = 90.0     # longer than the stand-in's own deadline, so its sentinel arrives first
PROXY = r'''
import os, sys, threading, time
sys.path.insert(0, sys.argv[1])
from backend.worker.manager import WorkerManager
argv = sys.argv[2:] or None
stop = threading.Event()
def spin():
    while not stop.is_set():
        pass
for _ in range(int(os.environ.get("ORIENTA_PROXY_SPIN", "0"))):
    threading.Thread(target=spin, daemon=True).start()
m = WorkerManager(idle_s=600, argv=argv)
jid = m.submit("_sleep", {"s": 120})
end = time.time() + float(os.environ.get("ORIENTA_PROXY_DEADLINE", "60"))
while time.time() < end:
    snap = m.get(jid)
    if snap and snap["status"] == "running" and m.worker_pid:
        stop.set()
        print(m.worker_pid, flush=True)
        break
    time.sleep(0.05)
else:
    stop.set()
    w = getattr(m, "_worker", None)
    print("NO-WORKER", w.proc.pid if w else 0, flush=True)
    m.shutdown()
    sys.exit(3)
time.sleep(600)
'''


def _launch(tmp_path, disable: str, launcher: bool = False, spin: int = 0,
            deadline: float = 60.0, start_delay: float = 0.0,
            read_timeout: float = READ_TIMEOUT_S):
    """Start the stand-in backend; return (proxy, first output line or None).

    None means the stand-in said nothing within `read_timeout` or closed its
    output; the caller decides what that means and must reap the proxy.
    """
    script = tmp_path / "proxy.py"
    script.write_text(PROXY, encoding="utf-8")
    args = [sys.executable, str(script), str(ROOT)]
    if launcher:
        lp = tmp_path / "launcher.py"
        lp.write_text("import subprocess, sys\n"
                      "sys.exit(subprocess.call([sys.executable, '-m', 'backend.worker']))\n",
                      encoding="utf-8")
        args += [sys.executable, str(lp)]
    env = dict(os.environ)
    env["ORIENTA_WORKER_TEST_JOBS"] = "1"
    env["ORIENTA_WORKER_TEST_DISABLE"] = disable
    env["ORIENTA_PROXY_SPIN"] = str(spin)
    env["ORIENTA_PROXY_DEADLINE"] = str(deadline)
    env["ORIENTA_WORKER_TEST_START_DELAY"] = str(start_delay)
    proxy = subprocess.Popen(args, stdout=subprocess.PIPE, cwd=str(ROOT), env=env)
    box: "queue.Queue" = queue.Queue()
    threading.Thread(target=lambda: box.put(proxy.stdout.readline()), daemon=True).start()
    try:
        line = box.get(timeout=read_timeout)
    except queue.Empty:
        return proxy, None
    return proxy, (line.decode("utf-8", "replace").strip() or None)


def _reap(proxy) -> None:
    """Kill the stand-in and everything below it; leave nothing running."""
    try:
        victims = psutil.Process(proxy.pid).children(recursive=True)
    except psutil.Error:
        victims = []
    if proxy.poll() is None:
        try:
            victims.append(psutil.Process(proxy.pid))
        except psutil.Error:
            pass
    for p in victims:
        try:
            p.kill()
        except psutil.Error:
            pass
    proxy.wait(timeout=10)
    try:
        proxy.stdout.close()
    except OSError:
        pass


def _run_proxy(tmp_path, disable: str, launcher: bool = False, spin: int = 0, **launch_kw):
    proxy, line = _launch(tmp_path, disable, launcher, spin, **launch_kw)
    try:
        if line is None:
            pytest.fail("the stand-in backend said nothing within "
                        f"{READ_TIMEOUT_S:.0f} s or closed its output")
        if line.startswith("NO-WORKER"):
            pytest.fail("the stand-in backend gave up: its worker never became ready "
                        f"({line!r})")
        try:
            return proxy, int(line)
        except ValueError:
            pytest.fail(f"the stand-in backend printed {line!r} instead of a worker PID")
    except BaseException:
        _reap(proxy)
        raise

def _kill_proxy_and_check(proxy, worker_pid, limit=2.0):
    try:
        proxy.kill()                 # TerminateProcess / SIGKILL: no cleanup runs
        proxy.wait(timeout=10)
        assert gone_within(worker_pid, limit), "the worker outlived the backend"
    finally:
        kill_quietly(worker_pid)
        proxy.stdout.close()


def test_with_every_guard(tmp_path):
    proxy, pid = _run_proxy(tmp_path, disable="")
    _kill_proxy_and_check(proxy, pid)


def test_only_the_end_of_stdin(tmp_path):
    proxy, pid = _run_proxy(tmp_path, disable="os,poll,jobobject")
    _kill_proxy_and_check(proxy, pid)


def test_only_the_pid_poll(tmp_path):
    proxy, pid = _run_proxy(tmp_path, disable="os,eof,jobobject")
    _kill_proxy_and_check(proxy, pid, limit=3.0)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows guard")
def test_only_the_job_object(tmp_path):
    proxy, pid = _run_proxy(tmp_path, disable="os,eof,poll")
    _kill_proxy_and_check(proxy, pid)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows guard")
def test_only_the_job_object_with_a_launcher(tmp_path):
    # Busy backend: without CREATE_SUSPENDED the launcher's real interpreter
    # escapes the job object while the backend is slow to assign it.
    proxy, pid = _run_proxy(tmp_path, disable="os,eof,poll", launcher=True, spin=SPINNERS)
    _kill_proxy_and_check(proxy, pid)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux guard")
def test_only_the_parent_death_signal(tmp_path):
    proxy, pid = _run_proxy(tmp_path, disable="eof,poll")
    _kill_proxy_and_check(proxy, pid)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS guard")
def test_only_the_kqueue_watch(tmp_path):
    proxy, pid = _run_proxy(tmp_path, disable="eof,poll")
    _kill_proxy_and_check(proxy, pid)


def test_a_stand_in_that_gives_up_reports_its_worker_and_leaves_none(tmp_path):
    t0 = time.monotonic()
    proxy, line = _launch(tmp_path, disable="", deadline=2.0, start_delay=20.0, read_timeout=30.0)
    worker_pid = None
    try:
        assert line is not None and line.startswith("NO-WORKER"), line
        worker_pid = int(line.split()[1])
        assert worker_pid > 0
        assert time.monotonic() - t0 < 25.0, "the sentinel took as long as the start delay"
        assert proxy.wait(timeout=10) == 3
        assert gone_within(worker_pid, 5.0), "the stand-in left its worker running"
    finally:
        _reap(proxy)
        kill_quietly(worker_pid)


def test_a_slow_start_fails_fast_with_a_clear_message(tmp_path):
    t0 = time.monotonic()
    with pytest.raises(pytest.fail.Exception, match="never became ready"):
        _run_proxy(tmp_path, disable="", deadline=2.0, start_delay=20.0, read_timeout=30.0)
    assert time.monotonic() - t0 < 25.0


def test_a_silent_stand_in_is_killed_after_the_read_timeout(tmp_path):
    # Worker start takes longer than the stand-in's deadline AND the read
    # timeout: the reader must give up, and the helper must leave no process.
    t0 = time.monotonic()
    with pytest.raises(pytest.fail.Exception, match="said nothing"):
        _run_proxy(tmp_path, disable="", deadline=60.0, start_delay=30.0, read_timeout=3.0)
    assert time.monotonic() - t0 < 20.0