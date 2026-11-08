"""The Windows Job Object that ties every compute worker to the backend."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Job Objects exist only on Windows")

# A launcher-like process: it starts the "real" worker as its child and reports that child's PID.
_LAUNCHER_CODE = (
    "import subprocess, sys; "
    "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "print(c.pid, flush=True); c.wait()"
)


def _sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def _suspended_launcher():
    from backend.worker import winjob
    return subprocess.Popen([sys.executable, "-c", _LAUNCHER_CODE], stdout=subprocess.PIPE,
                            creationflags=winjob.CREATE_SUSPENDED)


def _readline(p, timeout=20):
    """A line from the child's stdout, or an AssertionError instead of a hang if it never comes
    (a worker left suspended must fail the test, not abort the whole run through the timeout)."""
    box = []
    t = threading.Thread(target=lambda: box.append(p.stdout.readline()), daemon=True)
    t.start()
    t.join(timeout)
    assert box, "the process printed nothing: still suspended?"
    return box[0]


def _kill_tree(p):
    import psutil
    try:
        procs = [psutil.Process(p.pid)]
        procs += procs[0].children(recursive=True)
    except psutil.Error:
        procs = []
    for proc in procs:
        try:
            proc.kill()
        except psutil.Error:
            pass
    p.wait(timeout=10)
    if p.stdout is not None:
        p.stdout.close()


class _GilContention:
    """Four pure-Python spinner threads: a busy backend that holds the GIL."""

    def __enter__(self):
        self._stop = threading.Event()

        def spin():
            while not self._stop.is_set():
                sum(range(1000))

        self._threads = [threading.Thread(target=spin, daemon=True) for _ in range(4)]
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)


def test_struct_sizes_match_the_windows_x64_layout():
    """Sizes measured against the Windows headers, not taken from the code."""
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("the expected sizes are the x64 ones")
    from backend.worker import winjob
    assert ctypes.sizeof(winjob._BASIC_LIMIT) == 64
    assert ctypes.sizeof(winjob._IO_COUNTERS) == 48
    assert ctypes.sizeof(winjob._EXTENDED_LIMIT) == 144


def test_a_started_process_is_put_into_the_job():
    from backend.worker import winjob
    p = _sleeper()
    try:
        assert winjob.kill_with_this_process(p) is True
        assert winjob.is_in_job(p.pid) is True
    finally:
        p.kill()


def test_a_child_of_a_suspended_started_member_is_in_the_job_without_being_added():
    """Membership is inherited by processes created after the assignment. Started suspended, the
    launcher cannot have created its child before it is in the job, so this does not depend on timing
    or on whether sys.executable is itself a launcher."""
    from backend.worker import winjob
    p = _suspended_launcher()
    try:
        assert winjob.kill_with_this_process(p, resume=True) is True
        child = int(_readline(p))
        assert winjob.is_in_job(child) is True
    finally:
        _kill_tree(p)


def test_the_launcher_child_is_in_the_job_even_with_a_busy_parent():
    """The race the suspended start closes: under GIL contention in the parent a running process is
    assigned late (measured: 25 of 25 real interpreters outside the job)."""
    from backend.worker import winjob
    with _GilContention():
        for _ in range(8):
            p = _suspended_launcher()
            try:
                assert winjob.kill_with_this_process(p, resume=True) is True
                child = int(_readline(p))
                assert winjob.is_in_job(child) is True
            finally:
                _kill_tree(p)


def test_resume_lets_a_suspended_process_run():
    from backend.worker import winjob
    p = subprocess.Popen([sys.executable, "-c", "print('alive', flush=True)"],
                         stdout=subprocess.PIPE, creationflags=winjob.CREATE_SUSPENDED)
    try:
        assert winjob.kill_with_this_process(p, resume=True) is True
        assert _readline(p).strip() == b"alive"
    finally:
        _kill_tree(p)


def test_a_failed_assignment_still_resumes_the_process(monkeypatch):
    """A worker left suspended would hang until the manager's start limit."""
    from backend.worker import winjob
    monkeypatch.setattr(winjob, "_job", lambda k32: None)
    p = subprocess.Popen([sys.executable, "-c", "print('alive', flush=True)"],
                         stdout=subprocess.PIPE, creationflags=winjob.CREATE_SUSPENDED)
    try:
        assert winjob.kill_with_this_process(p, resume=True) is False
        assert _readline(p).strip() == b"alive"
    finally:
        _kill_tree(p)


def test_a_process_outside_is_not_in_the_job():
    from backend.worker import winjob
    first = _sleeper()
    try:
        winjob.kill_with_this_process(first)       # make sure the job exists
        p = _sleeper()
        try:
            assert winjob.is_in_job(p.pid) is False
        finally:
            p.kill()
    finally:
        first.kill()
