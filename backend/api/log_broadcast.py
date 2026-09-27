"""
Broadcast Python log records to connected WebSocket clients.

Usage (in main.py):
    from backend.api.log_broadcast import install_ws_log_handler
    install_ws_log_handler(ws_manager)
"""

import asyncio
import logging
import time
from contextlib import contextmanager

# Will be set by install_ws_log_handler()
_ws_manager = None
_loop = None

# Loggers to suppress (too noisy for the dev panel)
#
# "py.warnings" is not here for noise. logging.captureWarnings(True)
# (file_log.py) turns every Python warning into a record on this logger, so a
# warning raised WHILE broadcasting comes straight back into this handler and
# can warn again. That is a closed cycle, and one record was measured to be
# enough to run away (measured before the fix, on Windows: 3001 records from
# one; the tests pin the FIXED behaviour, they do not re-run the runaway).
# The dev panel
# loses Python warnings; the log FILE still records them, which is where they
# belong — deliberately not added to the file handler's suppression list.
_SUPPRESSED_LOGGERS = frozenset({
    "uvicorn.access",
    "watchfiles.main",
    "httpcore",
    "httpx",
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    "py.warnings",
    # Not a measured path — a review tried to construct one and could not.
    # It is here because "asyncio" is the logger that reports failures OF the
    # very loop we schedule onto, which is the one cycle shape left, and one
    # line is cheaper than finding out the hard way. asyncio's own errors
    # still reach the log file.
    "asyncio",
    # The broadcast prune reports itself (main.py ConnectionManager.broadcast).
    # That record must never come back through here: it would be broadcast to
    # the same dead socket, raise again, and log again. The file log keeps it.
    "backend.api.ws_prune",
})

# A guard on a `_from_ws_broadcast` record attribute used to sit in emit(),
# described as stopping records that originate from the broadcast itself.
# Nothing in this repository has ever set that attribute (only the design note
# it was copied from, docs/superpowers/plans/2026-03-27-global-dev-panel.md),
# so it never fired once. Removed rather than left standing: a brake that does
# nothing, next to brakes that do, is how the next reader mis-reads this file.


def _schedule(payload) -> bool:
    """Hand a payload to the event loop; never leave a coroutine unawaited.

    `run_coroutine_threadsafe(manager.broadcast(payload), loop)` builds the
    coroutine as an ARGUMENT, so it already exists when the call raises on a
    closed loop. Dropping it there is what produced the runaway: an un-awaited
    coroutine emits a RuntimeWarning, captureWarnings turns that into a log
    record, and the record arrives back here. Build it first, close it by hand
    if it cannot be scheduled.

    Returns True when the loop accepted it.
    """
    # Read each global ONCE. `uninstall_ws_log_handler` clears them from the
    # loop thread while worker threads (the prewarm, the reaper, the thread
    # excepthook, the simulation heartbeat) may be inside this function. Read
    # twice, a thread can pass the `is None` guard and then call
    # `run_coroutine_threadsafe(coro, None)`, which raises AttributeError, not
    # RuntimeError — measured — so the coroutine escapes un-awaited and the
    # cascade is armed again.
    manager, loop = _ws_manager, _loop
    if manager is None or loop is None:
        return False
    coro = manager.broadcast(payload)
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
        return True
    except Exception:
        # Loop closed during shutdown, or anything else. Closing the coroutine
        # marks it finished, so garbage collection does not warn about it.
        # Deliberately broad: what must never happen here is an un-awaited
        # coroutine, and every escape route leads to one.
        coro.close()
        return False


class WebSocketLogHandler(logging.Handler):
    """Logging handler that broadcasts records via WebSocket."""

    def emit(self, record: logging.LogRecord):
        # Every handler owes its caller this try/except: logging.Handler.handle
        # does NOT guard emit, so anything raised here comes out of an ordinary
        # logger.info() somewhere else entirely. Measured before this was added:
        # a bad format string ("a %d", "not-an-int") raised TypeError into the
        # caller, while every other root handler survived the same record.
        # handleError only writes to stderr and emits no record, so it cannot
        # re-enter this handler.
        try:
            if record.name in _SUPPRESSED_LOGGERS:
                return
            if _ws_manager is None or _loop is None:
                return

            msg = self.format(record)
            payload = {
                "type": "dev_log",
                "category": "log",
                "level": record.levelname,
                "source": record.name,
                "message": msg,
                "timestamp": time.strftime("%H:%M:%S", time.localtime(record.created))
                             + f".{int(record.msecs):03d}",
            }
            _schedule(payload)
        except Exception:
            self.handleError(record)


def _remove_own_handlers() -> int:
    """Detach every handler of ours from the root logger. Returns how many."""
    root = logging.getLogger()
    doomed = [h for h in root.handlers if isinstance(h, WebSocketLogHandler)]
    for h in doomed:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:  # pragma: no cover - closing must never raise upward
            pass
    return len(doomed)


def install_ws_log_handler(ws_manager, level=logging.DEBUG):
    """Attach the WebSocket handler to the root logger.

    Call once at application startup, after the event loop is running.

    A handler installed by an earlier lifespan is removed first. Without that,
    a second install (a test client, a reload, two apps in one process) leaves
    the first handler on the root logger pointing at a loop that is already
    closed, and every log record in the process then goes through it.
    """
    global _ws_manager, _loop
    _remove_own_handlers()
    _ws_manager = ws_manager
    _loop = asyncio.get_running_loop()

    handler = WebSocketLogHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.addHandler(handler)
    # Ensure root logger level allows DEBUG through
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)


def uninstall_ws_log_handler() -> int:
    """Detach the handler and forget the loop. Call from lifespan shutdown.

    The loop closes right after the lifespan ends. A handler left behind holds
    a reference to it and turns every later log record into a scheduling
    attempt against a dead loop, so the process keeps paying for a WebSocket
    that no longer exists. Clearing `_loop` also makes `emit` return early,
    which is a second brake if anything still holds a handler.
    """
    global _ws_manager, _loop
    removed = _remove_own_handlers()
    _ws_manager = None
    _loop = None
    return removed


@contextmanager
def timed_step(name: str):
    """Context manager that logs step duration via WebSocket.

    Usage:
        with timed_step("Hough: prepare reflectors"):
            reflectors = prepare_reflectors(phase_list)
    """
    start = time.perf_counter()
    logger = logging.getLogger("timing")
    logger.info("[START] %s", name)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Log as normal log (gets picked up by WebSocketLogHandler too)
        logger.info("[DONE]  %s — %.0fms", name, elapsed_ms)
        # Also send structured timing message
        if _ws_manager is not None and _loop is not None:
            payload = {
                "type": "dev_log",
                "category": "timing",
                "step": name,
                "duration_ms": round(elapsed_ms),
                "timestamp": time.strftime("%H:%M:%S", time.localtime())
                             + f".{int((time.perf_counter() % 1) * 1000):03d}",
            }
            # Same construct, same trap — this second site was not in the
            # report, and an un-awaited coroutine here warns just as loudly.
            _schedule(payload)
