"""Why the backend ended, written down before it ends.

Until now a backend session in ``logs/orienta.log`` simply stopped: 40 starts
between 2026-09-09 and 2026-11-08 and not one line that said why the process
went away. The ways out of this process and what each leaves behind:

* An orderly shutdown (uvicorn on Ctrl+C or a signal, the lifespan's tear-down)
  and a plain interpreter exit run ``atexit``: :func:`_log_end` writes one line
  with the reason, the uptime, the last request, the requests still running and
  the memory.
* ``os._exit`` (the watchdogs, ``/api/shutdown``) skips ``atexit``. Those callers
  go through :func:`exit_now`, which writes the same line first.
* A native crash (access violation, stack overflow in a C extension) runs no
  Python at all. :func:`install` points ``faulthandler`` at
  ``logs/backend-fault.log``, which receives the Python stack of every thread.
* A hard kill from outside (``taskkill /F``, the OOM killer) leaves nothing in
  this process. The state file (:func:`write_state`, every few seconds) holds the
  last request, the requests in flight and the memory, so the launcher can say
  what the backend was doing when it disappeared; the launcher logs the exit code.

Nothing here may stop the backend from starting or running: every step is
wrapped, and a failure is at most a missing line.
"""
from __future__ import annotations

import atexit
import faulthandler
import itertools
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("backend.lifecycle")

#: Seconds between two writes of the state file.
STATE_INTERVAL_S = 5.0
STATE_BASENAME = "backend-state.json"
FAULT_BASENAME = "backend-fault.log"

_started = time.monotonic()
_lock = threading.Lock()
# One writer of the state file at a time (the 5-s loop and the end of the process); a
# separate lock, so a write never waits for a request being noted.
_state_lock = threading.Lock()
_reason: Optional[str] = None
_inflight: dict = {}
_ids = itertools.count(1)
_last_request: Optional[dict] = None
_state_file: Optional[Path] = None
_fault_stream = None
_ended = False


def memory_mib() -> dict:
    """``{"commit": MiB, "rss": MiB}`` of this process; empty when psutil cannot tell.

    ``commit`` is the private bytes on Windows (what counts against the commit
    limit, the limit that ran out on 2026-11-01), the VMS elsewhere."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        commit = getattr(mi, "private", None)
        if commit is None:
            commit = mi.vms
        return {"commit": round(commit / 2**20), "rss": round(mi.rss / 2**20)}
    except Exception:
        return {}


def request_started(method: str, path: str) -> int:
    """Note that a request began; returns the token for :func:`request_finished`."""
    global _last_request
    token = next(_ids)
    entry = {"method": method, "path": path, "started": time.monotonic(),
             "at": time.strftime("%H:%M:%S")}
    with _lock:
        _inflight[token] = entry
        _last_request = entry
    return token


def request_finished(token: int) -> None:
    with _lock:
        _inflight.pop(token, None)


def note_reason(reason: str) -> None:
    """The reason the process is about to end; the first one stated wins."""
    global _reason
    with _lock:
        if _reason is None:
            _reason = reason


def state() -> dict:
    """What the process is doing now: the content of the state file."""
    now = time.monotonic()
    with _lock:
        inflight = sorted(_inflight.values(), key=lambda e: e["started"])
        last = dict(_last_request) if _last_request else None
        reason = _reason
    def _req(e):
        return {"method": e["method"], "path": e["path"], "at": e["at"],
                "running_s": round(now - e["started"], 1)}
    return {
        "pid": os.getpid(),
        "uptime_s": round(now - _started, 1),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "last_request": _req(last) if last else None,
        "in_flight": [_req(e) for e in inflight[:10]],
        "n_in_flight": len(inflight),
        "memory_mib": memory_mib(),
    }


def end_line(reason: Optional[str] = None) -> str:
    """The one line that says why and in which state the process ends."""
    s = state()
    why = reason or s["reason"] or "the interpreter exited (no reason was stated)"
    last = s["last_request"]
    last_txt = f'{last["method"]} {last["path"]} at {last["at"]}' if last else "none"
    run = ", ".join(f'{r["method"]} {r["path"]} ({r["running_s"]} s)' for r in s["in_flight"]) or "none"
    mem = s["memory_mib"]
    mem_txt = f'commit {mem["commit"]} MiB, RSS {mem["rss"]} MiB' if mem else "unknown"
    return (f"Backend process ends (pid {s['pid']}): {why}. Uptime {s['uptime_s']} s; "
            f"last request {last_txt}; still running: {run}; memory {mem_txt}.")


def _flush_logs() -> None:
    for h in list(logging.getLogger().handlers) + list(logging.getLogger("uvicorn").handlers):
        try:
            h.flush()
        except Exception:
            pass


def _log_end(reason: Optional[str] = None) -> None:
    global _ended
    with _lock:
        if _ended:
            return
        _ended = True
    try:
        logger.warning(end_line(reason))
        _flush_logs()
    except Exception:
        pass
    try:
        write_state(final_reason=reason or _reason or "the interpreter exited")
    except Exception:
        pass


def exit_now(code: int, reason: str) -> None:
    """``os._exit`` with the reason in the log first (``atexit`` does not run there).

    The exit is guaranteed: a logging handler that hangs (a stalled stderr pipe, a lock
    held by a stuck thread) cannot keep a watchdog from ending the process; after 2 s
    the process ends without the line."""
    timer = threading.Timer(2.0, os._exit, (code,))
    timer.daemon = True
    timer.start()
    note_reason(reason)
    _log_end(reason)
    os._exit(code)


def write_state(final_reason: Optional[str] = None) -> None:
    """Write :func:`state` to the state file, atomically (a reader never sees half a file)."""
    if _state_file is None:
        return
    with _state_lock:
        # Once the end is written, the loop's next tick must not overwrite it with a
        # state that no longer says the process ended.
        if final_reason is None and _ended:
            return
        s = state()
        if final_reason is not None:
            s["ended"] = final_reason
        tmp = _state_file.with_name(_state_file.name + ".tmp")
        tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
        os.replace(tmp, _state_file)


def _state_loop() -> None:
    while True:
        time.sleep(STATE_INTERVAL_S)
        try:
            write_state()
        except Exception:
            pass


def install(log_dir: Optional[Path]) -> None:
    """Fault log, end line and state file for this process. Idempotent, never raises.

    ``ORIENTA_BACKEND_STATE_FILE`` (set by the desktop app) names the state file,
    so the app can read it after the process is gone; otherwise it sits next to
    the log file. ``log_dir`` None (file logging switched off) leaves out the
    fault log and the default state file, not the end line."""
    global _state_file, _fault_stream
    if getattr(install, "_done", False):
        return
    install._done = True
    try:
        atexit.register(_log_end)
    except Exception:
        pass
    try:
        explicit = os.environ.get("ORIENTA_BACKEND_STATE_FILE", "").strip()
        if explicit:
            _state_file = Path(explicit)
        elif log_dir is not None:
            _state_file = Path(log_dir) / STATE_BASENAME
        if _state_file is not None:
            _state_file.parent.mkdir(parents=True, exist_ok=True)
            write_state()
            threading.Thread(target=_state_loop, name="backend-state", daemon=True).start()
    except Exception:
        logger.debug("backend state file not available", exc_info=True)
        _state_file = None
    try:
        if log_dir is not None:
            _fault_stream = open(Path(log_dir) / FAULT_BASENAME, "a", encoding="utf-8")
            _fault_stream.write(f"\n===== backend pid {os.getpid()} started "
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            _fault_stream.flush()
            faulthandler.enable(file=_fault_stream, all_threads=True)
    except Exception:
        logger.debug("fault log not available", exc_info=True)
