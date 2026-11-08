"""Child-side guards: a compute worker never outlives the backend that started it.

Four layers. Three of them live here, in the worker:
- Linux: ``prctl(PR_SET_PDEATHSIG, SIGKILL)``. The kernel kills the worker when
  the thread that forked it ends; the manager forks from its dispatcher thread,
  which lives as long as the backend.
- macOS: a kqueue ``EVFILT_PROC/NOTE_EXIT`` watch on the backend's PID.
- Everywhere: a poll of the backend's PID and start time once a second.
The fourth (end of stdin = backend gone) is in ``__main__``; the Windows Job
Object is set by the parent in ``winjob``.

``ORIENTA_WORKER_TEST_DISABLE`` (only with ``ORIENTA_WORKER_TEST_JOBS=1``)
switches single layers off so each can be proven on its own.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

from backend.process_watch import gone_reason
from backend.worker.jobs import test_switches

logger = logging.getLogger(__name__)

POLL_S = 1.0


def eof_exits() -> bool:
    return "eof" not in test_switches()


def arm_child(parent_pid: int, parent_started_at: float) -> None:
    off = test_switches()
    if "os" not in off:
        if sys.platform.startswith("linux"):
            _linux(parent_pid, parent_started_at)
        elif sys.platform == "darwin":
            _macos(parent_pid)
    if "poll" not in off:
        threading.Thread(target=_poll, args=(parent_pid, parent_started_at),
                         daemon=True, name="worker-parent-poll").start()


def _linux(parent_pid: int, parent_started_at: float) -> None:
    import ctypes
    import signal
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        logger.warning("compute worker: prctl(PR_SET_PDEATHSIG) failed, errno %d",
                       ctypes.get_errno())
        return
    # The backend may have died between fork and prctl; then no signal will come.
    if gone_reason(parent_pid, parent_started_at) is not None:
        os._exit(0)


def _macos(parent_pid: int) -> None:
    import select
    kq = select.kqueue()
    ev = select.kevent(parent_pid, filter=select.KQ_FILTER_PROC,
                       flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                       fflags=select.KQ_NOTE_EXIT)
    try:
        kq.control([ev], 0)
    except OSError:                       # the backend is already gone
        os._exit(0)

    def _wait() -> None:
        kq.control(None, 1)
        os._exit(0)

    threading.Thread(target=_wait, daemon=True, name="worker-parent-kqueue").start()


def _poll(parent_pid: int, parent_started_at: float) -> None:
    while True:
        time.sleep(POLL_S)
        if gone_reason(parent_pid, parent_started_at) is not None:
            os._exit(0)
