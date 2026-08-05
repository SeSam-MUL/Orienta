"""The desktop backend must not outlive the app that launched it.

Orphaned backends were a recurring, user-visible failure: the process kept
port 8000, the next session silently attached to it instead of starting a
fresh one, and a backend that had outlived a USB replug then served dead HDF5
handles — every load failing with `errno 22 Invalid argument` at fixed offsets
while the same file read fine in any new process.

Three separate defects let it survive; this file covers the backend half.
``_parent_watchdog`` polls the PID Electron passes in ``KIKUCHIPY_PARENT_PID``
and exits when it is gone — the only mechanism that also covers Electron being
killed outright or the hosting console being closed.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("KIKUCHIPY_PARENT_PID", raising=False)


def _run(coro, timeout=5):
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


class _Exited(Exception):
    """Stand-in for os._exit, which never returns."""

    def __init__(self, code):
        self.code = code


def _patch_exit(monkeypatch, backend_main):
    """Replace os._exit with something that stops execution like the real one.

    A recorder that merely returns would let the watchdog run on into its poll
    loop, which is not how os._exit behaves and makes the test hang.
    """
    def _fake_exit(code):
        raise _Exited(code)

    monkeypatch.setattr(backend_main.os, "_exit", _fake_exit)


def test_disabled_without_parent_pid():
    """start_app.py / --headless / plain uvicorn have no parent to watch."""
    from backend.api.main import _parent_watchdog
    _run(_parent_watchdog())  # returns immediately, must not exit the process


def test_disabled_on_garbage_parent_pid(monkeypatch):
    from backend.api.main import _parent_watchdog
    monkeypatch.setenv("KIKUCHIPY_PARENT_PID", "not-a-pid")
    _run(_parent_watchdog())


def test_exits_when_parent_is_already_gone(monkeypatch):
    """A backend must never outlive a parent that died before it looked."""
    import psutil
    from backend.api import main as backend_main

    # A PID that is guaranteed not to exist.
    dead = 999_999
    while psutil.pid_exists(dead):
        dead -= 1
    monkeypatch.setenv("KIKUCHIPY_PARENT_PID", str(dead))

    _patch_exit(monkeypatch, backend_main)

    with pytest.raises(_Exited) as exc:
        _run(backend_main._parent_watchdog())
    assert exc.value.code == 0, "expected a clean exit when the parent is absent"


def test_exits_when_parent_pid_is_reused(monkeypatch):
    """PIDs get recycled — identity is (pid, create_time), not pid alone."""
    import psutil
    from backend.api import main as backend_main

    monkeypatch.setenv("KIKUCHIPY_PARENT_PID", str(os.getpid()))

    real_process = psutil.Process
    _patch_exit(monkeypatch, backend_main)

    state = {"first": True}

    class _ShiftingProcess:
        """Same PID, different start time on the second look."""

        def __init__(self, pid):
            self._p = real_process(pid)

        def create_time(self):
            if state["first"]:
                state["first"] = False
                return self._p.create_time()
            return self._p.create_time() + 1000.0

    monkeypatch.setattr(psutil, "Process", _ShiftingProcess)
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)

    with pytest.raises(_Exited) as exc:
        _run(backend_main._parent_watchdog(), timeout=10)
    assert exc.value.code == 0, "expected a clean exit when the PID was reused"
