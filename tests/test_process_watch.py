"""The one question both the backend's parent watchdog and the compute worker
ask: is the process that started us still the same process?"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import process_watch  # noqa: E402


def _dead_pid() -> int:
    pid = 999_999
    while psutil.pid_exists(pid):
        pid -= 1
    return pid


def test_a_living_process_with_its_own_start_time_is_not_gone():
    me = os.getpid()
    assert process_watch.gone_reason(me, process_watch.started_at(me)) is None


def test_a_pid_that_does_not_exist_is_gone():
    assert process_watch.gone_reason(_dead_pid(), 0.0) == "process no longer exists"


def test_a_reused_pid_is_gone():
    me = os.getpid()
    assert process_watch.gone_reason(me, process_watch.started_at(me) + 1000.0) == (
        "PID was reused by a different process")


def test_a_process_that_cannot_be_inspected_counts_as_gone(monkeypatch):
    def _boom(pid):
        raise psutil.AccessDenied(pid)
    monkeypatch.setattr(process_watch, "_uses_proc", lambda: False)    # psutil's route, on every platform
    monkeypatch.setattr(psutil, "Process", _boom)
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    assert process_watch.gone_reason(os.getpid(), 1.0) == "parent process is no longer inspectable"


def test_started_at_raises_for_a_missing_process():
    with pytest.raises(psutil.Error):
        process_watch.started_at(_dead_pid())


# ---- Linux: the start is read in clock ticks since boot, not as a wall-clock time -----
# The Linux branch cannot run for real on the machines these tests are written on,
# so it is driven through a fake /proc directory; the expected numbers are written
# out by hand below.

def _fake_proc(monkeypatch, tmp_path, pid, stat_text):
    folder = tmp_path / str(pid)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stat").write_bytes(stat_text.encode("ascii"))
    monkeypatch.setattr(process_watch, "_PROC_ROOT", str(tmp_path))
    monkeypatch.setattr(process_watch, "_uses_proc", lambda: True)


# Field 1 pid, field 2 the command in parentheses (here with a space and a ")"
# inside), field 3 state, fields 4..21 the numbers 4..21, field 22 = 123456.
def _stat_line(pid, ticks):
    return f"{pid} (a b) c) S " + " ".join(str(n) for n in range(4, 22)) + f" {ticks} 999 1 2\n"


def test_the_linux_start_is_field_22_of_proc_stat(monkeypatch, tmp_path):
    _fake_proc(monkeypatch, tmp_path, 4242, _stat_line(4242, 123456))
    assert process_watch.started_at(4242) == 123456.0


def test_a_clock_jump_does_not_make_the_parent_look_like_a_reused_pid(monkeypatch, tmp_path):
    me = os.getpid()
    _fake_proc(monkeypatch, tmp_path, me, _stat_line(me, 123456))
    started = process_watch.started_at(me)
    real_process = psutil.Process
    shift = {"s": 0.0}

    class _ClockJumped:
        """psutil's create_time() after the clock was set: same process, other number."""

        def __init__(self, pid):
            self._p = real_process(pid)

        def create_time(self):
            return self._p.create_time() + shift["s"]

    monkeypatch.setattr(psutil, "Process", _ClockJumped)
    shift["s"] = 3600.0
    assert process_watch.gone_reason(me, started) is None


def test_a_different_start_tick_is_still_a_reused_pid(monkeypatch, tmp_path):
    me = os.getpid()
    _fake_proc(monkeypatch, tmp_path, me, _stat_line(me, 123456))
    started = process_watch.started_at(me)
    _fake_proc(monkeypatch, tmp_path, me, _stat_line(me, 123457))
    assert process_watch.gone_reason(me, started) == "PID was reused by a different process"


def test_a_pid_without_a_proc_entry_raises_no_such_process(monkeypatch, tmp_path):
    monkeypatch.setattr(process_watch, "_PROC_ROOT", str(tmp_path))
    monkeypatch.setattr(process_watch, "_uses_proc", lambda: True)
    with pytest.raises(psutil.NoSuchProcess):
        process_watch.started_at(4243)


def test_an_unreadable_proc_stat_falls_back_to_the_creation_time(monkeypatch, tmp_path):
    me = os.getpid()
    _fake_proc(monkeypatch, tmp_path, me, "garbage without a bracket\n")
    assert process_watch.started_at(me) == psutil.Process(me).create_time()


def test_other_systems_keep_the_creation_time(monkeypatch):
    monkeypatch.setattr(process_watch, "_uses_proc", lambda: False)
    me = os.getpid()
    assert process_watch.started_at(me) == psutil.Process(me).create_time()
