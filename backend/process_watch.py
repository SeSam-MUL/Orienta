"""Is the process that started us still the same process?

Asked by the backend's parent watchdog (``backend.api.main._parent_watchdog``)
and by the compute worker (``backend.worker.orphan``). A PID alone is not an
identity: the operating system reuses PIDs, so the start time is pinned too.

The value that is pinned comes from ``started_at`` on both sides (the backend
hands it to its worker in the environment), and ``gone_reason`` compares with
the same function, so the two always agree on what the number means.
"""
from __future__ import annotations

import sys
from typing import Optional

import psutil

_PROC_ROOT = "/proc"


def _uses_proc() -> bool:
    return sys.platform.startswith("linux")


def _linux_start_ticks(pid: int) -> float:
    """Field 22 of ``/proc/<pid>/stat``: the start in clock ticks since boot.

    psutil's ``create_time()`` turns that into a wall-clock time with the
    *current* boot time, so after the clock is set (NTP, a VM resume) the same
    process reads differently and looks like a reused PID. The tick count does
    not depend on the clock."""
    try:
        with open(f"{_PROC_ROOT}/{pid}/stat", "rb") as f:
            raw = f.read().decode("ascii", "replace")
    except (FileNotFoundError, ProcessLookupError):
        raise psutil.NoSuchProcess(pid) from None
    # The command name (field 2) is in parentheses and may hold spaces and
    # parentheses itself; the fields after the LAST ")" are 3, 4, ... so 22 is [19].
    return float(raw[raw.rindex(")") + 1:].split()[19])


def _identity(pid: int) -> float:
    if _uses_proc():
        try:
            return _linux_start_ticks(pid)
        except psutil.NoSuchProcess:
            raise
        except (OSError, ValueError, IndexError):
            pass                    # /proc not readable here: the wall-clock time will do
    return psutil.Process(pid).create_time()


def started_at(pid: int) -> float:
    """What pins ``pid``'s identity; raises ``psutil.Error`` if there is no such process."""
    return _identity(pid)


def gone_reason(pid: int, started: float) -> Optional[str]:
    """Why ``pid`` (started at ``started``) is gone, or None while it is still there."""
    try:
        if not psutil.pid_exists(pid):
            return "process no longer exists"
        if _identity(pid) != started:
            return "PID was reused by a different process"
        return None
    except Exception:  # noqa: BLE001 -- cannot look at it any more: treat as gone
        return "parent process is no longer inspectable"
