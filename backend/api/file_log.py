"""Persistent rotating log file for the backend.

Until now the backend had no log file at all: records went to the Dev-Panel
WebSocket (in-memory, 500 entries) and to a console that vanishes when its
window closes.  Bug reports arrived as screenshots with the actual error long
gone.  This module gives every session a durable trace in ``logs/orienta.log``.

Installed at *import time* of ``backend.api.main`` (NOT in the lifespan) so the
import/prewarm phase is captured too.  Design notes, each the result of a
codebase audit (2026-08-26):

- uvicorn's default logging config sets ``propagate: False`` on the ``uvicorn``
  logger, so unhandled-route tracebacks ("Exception in ASGI application")
  NEVER reach the root logger.  The handler is therefore attached to the
  ``uvicorn`` logger as well.
- ``warnings.warn`` output (ours + kikuchipy/hyperspy/numpy deprecations)
  bypasses logging entirely → ``logging.captureWarnings(True)``.
- Raw ``threading.Thread`` bodies (e.g. the emsphinx upload worker) die via
  ``threading.excepthook`` straight to stderr → hooked into logging here.
- The WS dev-panel handler forces the root logger to DEBUG; the file handler
  guards itself with level INFO plus a noisy-logger filter, otherwise
  matplotlib/PIL/numba chatter would blow through the rotation in minutes.

Opt-out for tests / special runs: set ``ORIENTA_NO_FILE_LOG=1``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
LOG_BASENAME = "orienta.log"

_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# Loggers whose records never belong in the file (same spirit as the
# Dev-Panel suppression list in log_broadcast.py, plus known INFO-spammers).
_NOISY_LOGGERS = (
    "uvicorn.access",
    "watchfiles.main",
    "httpcore",
    "httpx",
    "matplotlib",
    "PIL",
    "numba",
)


class _DropNoisyLoggers(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        name = record.name
        for prefix in _NOISY_LOGGERS:
            if name == prefix or name.startswith(prefix + "."):
                return False
        return True


_install_lock = threading.Lock()
_installed_path: Path | None = None


def install_file_logging(log_dir: Path | None = None) -> Path | None:
    """Attach a rotating file handler; return the log-file path.

    Idempotent; never raises (a broken log setup must not kill the backend).
    Returns None when disabled via ORIENTA_NO_FILE_LOG=1 or on failure.
    """
    global _installed_path
    if os.environ.get("ORIENTA_NO_FILE_LOG", "").strip() == "1":
        return None
    with _install_lock:
        if _installed_path is not None:
            return _installed_path
        try:
            directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / LOG_BASENAME

            handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
            )
            handler.setLevel(logging.INFO)
            handler.addFilter(_DropNoisyLoggers())
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
            )

            root = logging.getLogger()
            root.addHandler(handler)
            # Root defaults to WARNING; open it up so INFO records reach the
            # file (the WS handler forces DEBUG later anyway — our handler's
            # own INFO level is the file's guard).
            if root.level > logging.INFO:
                root.setLevel(logging.INFO)

            # Root now has a handler, so logging.lastResort no longer prints
            # WARNINGs to stderr.  Preserve that console behaviour explicitly.
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(logging.WARNING)
            console.addFilter(_DropNoisyLoggers())
            console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            root.addHandler(console)

            # uvicorn detaches from root (propagate=False) — attach the file
            # handler there too so ASGI exception tracebacks are persisted.
            # (uvicorn applies its dictConfig before importing the app, so
            # this attachment survives.)
            logging.getLogger("uvicorn").addHandler(handler)

            logging.captureWarnings(True)
            _hook_thread_exceptions()
            _hook_sys_excepthook()

            _installed_path = path
            return path
        except Exception:  # pragma: no cover - last-ditch safety
            try:
                logging.getLogger(__name__).exception("File logging setup failed")
            except Exception:
                pass
            return None


def _hook_thread_exceptions() -> None:
    original = threading.excepthook

    def hook(args):
        try:
            name = args.thread.name if args.thread is not None else "?"
            logging.getLogger("threading").error(
                "Uncaught exception in thread %r",
                name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        except Exception:
            pass
        original(args)

    threading.excepthook = hook


def _hook_sys_excepthook() -> None:
    original = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        try:
            logging.getLogger("uncaught").error(
                "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
            )
        except Exception:
            pass
        original(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


def get_log_paths() -> list[Path]:
    """All existing log files (current + rotated), newest first."""
    if _installed_path is None:
        base = DEFAULT_LOG_DIR / LOG_BASENAME
    else:
        base = _installed_path
    paths = [base] + [base.with_name(f"{base.name}.{i}") for i in range(1, _BACKUP_COUNT + 1)]
    return [p for p in paths if p.exists()]
