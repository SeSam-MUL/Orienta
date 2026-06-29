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
_SUPPRESSED_LOGGERS = frozenset({
    "uvicorn.access",
    "watchfiles.main",
    "httpcore",
    "httpx",
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
})


class WebSocketLogHandler(logging.Handler):
    """Logging handler that broadcasts records via WebSocket."""

    def emit(self, record: logging.LogRecord):
        if record.name in _SUPPRESSED_LOGGERS:
            return
        if _ws_manager is None or _loop is None:
            return
        # Don't broadcast logs that originate from the broadcast itself
        if getattr(record, "_from_ws_broadcast", False):
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
        try:
            asyncio.run_coroutine_threadsafe(_ws_manager.broadcast(payload), _loop)
        except RuntimeError:
            pass  # Loop closed during shutdown — ignore


def install_ws_log_handler(ws_manager, level=logging.DEBUG):
    """Attach the WebSocket handler to the root logger.

    Call once at application startup, after the event loop is running.
    """
    global _ws_manager, _loop
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
            try:
                asyncio.run_coroutine_threadsafe(
                    _ws_manager.broadcast(payload), _loop
                )
            except RuntimeError:
                pass
