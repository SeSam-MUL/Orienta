"""Say in the log when the event loop stops answering, and where it was.

Every request of the page goes through one asyncio event loop. When something
holds it (synchronous work in an ``async`` route, a lock, an import another
thread is in the middle of), the health poll and every page request wait, the
page reports the backend as gone, and nothing in the log says why. Measured on
2026-11-07: the settings answered 18.8 s after "Backend ready" and
``/api/system/log-info`` gave up after 8 s.

A daemon thread asks the loop for a sign of life every ``interval_s``. If the
loop does not answer within ``threshold_s`` it takes the stack of the loop's
thread and of every other thread at that moment, and once the loop answers
again it logs one warning: how long the loop was held and where. One line per
stall, nothing while the loop is healthy.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import traceback

logger = logging.getLogger("backend.loop_watch")


def _stack(frame, limit: int) -> str:
    lines = traceback.format_stack(frame)[-limit:]
    return "".join(lines).rstrip()


def _snapshot(loop_tid: int, limit: int) -> str:
    frames = sys._current_frames()
    names = {t.ident: t.name for t in threading.enumerate()}
    parts = []
    loop_frame = frames.get(loop_tid)
    if loop_frame is not None:
        parts.append(f"event loop thread:\n{_stack(loop_frame, limit)}")
    for tid, frame in frames.items():
        if tid in (loop_tid, threading.get_ident()):
            continue
        name = names.get(tid, f"thread {tid}")
        parts.append(f"{name}:\n{_stack(frame, 4)}")
    return "\n".join(parts)


class LoopWatch:
    def __init__(self, loop: asyncio.AbstractEventLoop, threshold_s: float = 2.0,
                 interval_s: float = 0.5, stack_limit: int = 14):
        self.loop = loop
        self.threshold_s = threshold_s
        self.interval_s = interval_s
        self.stack_limit = stack_limit
        self.loop_tid = threading.get_ident()      # created on the loop's thread
        self.stalls: list = []                     # (seconds, snapshot) for tests and the state file
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="loop-watch", daemon=True)

    def start(self) -> "LoopWatch":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            answered = threading.Event()
            sent = time.monotonic()
            try:
                self.loop.call_soon_threadsafe(answered.set)
            except RuntimeError:      # the loop is closed
                return
            if answered.wait(self.threshold_s):
                continue
            snapshot = _snapshot(self.loop_tid, self.stack_limit)
            while not answered.wait(1.0):
                if self._stop.is_set() or self.loop.is_closed():
                    return
            held = time.monotonic() - sent
            self.stalls.append((held, snapshot))
            logger.warning("The event loop did not answer for %.1f s (requests waited that long). "
                           "Where the threads were %.0f s into it:\n%s",
                           held, self.threshold_s, snapshot)


def start_loop_watch(threshold_s: float = 2.0) -> LoopWatch:
    """Start watching the running loop (call from inside it, e.g. the lifespan)."""
    return LoopWatch(asyncio.get_running_loop(), threshold_s=threshold_s).start()
