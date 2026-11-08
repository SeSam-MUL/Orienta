"""The compute worker process: ``python -m backend.worker``.

Started by ``backend.worker.manager``, never by hand. It reads jobs from stdin,
one JSON line each, runs them one after the other and answers on its protocol
stream. It ends when stdin ends (the backend closed it, or the backend died).
"""
from __future__ import annotations

import importlib
import logging
import os
import queue
import sys
import threading
import time
import traceback


def _protocol_stream():
    """The real stdout, kept for protocol lines; fd 1 is pointed at stderr.

    Libraries print: tqdm, numba, C code. One stray line on the protocol
    channel would break it, so everything that writes to "stdout" in this
    process goes to stderr, which the backend mirrors into its log. On Windows
    the process's standard output handle is moved as well, because a C runtime
    other than Python's writes through that handle and not through fd 1.
    """
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32")
        k32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
        STD_OUTPUT_HANDLE = wintypes.DWORD(-11 & 0xFFFFFFFF)
        k32.SetStdHandle(STD_OUTPUT_HANDLE, wintypes.HANDLE(msvcrt.get_osfhandle(2)))
    return out


def main() -> None:
    out = _protocol_stream()
    from backend.worker import orphan, protocol
    from backend.worker.jobs import JobContext, JobError, lookup

    lock = threading.Lock()

    def send(msg: dict) -> None:
        data = protocol.encode(msg)
        with lock:
            out.write(data)
            out.flush()

    orphan.arm_child(int(os.environ["ORIENTA_WORKER_PARENT_PID"]),
                     float(os.environ["ORIENTA_WORKER_PARENT_START"]))
    _forward_logging(send)

    jobs: "queue.Queue[dict]" = queue.Queue()
    cancelled_ids: set = set()

    def _read_stdin() -> None:
        for line in sys.stdin.buffer:
            try:
                msg = protocol.decode(line)
                if not isinstance(msg, dict):
                    raise ValueError("not a JSON object")
                hash(msg.get("id"))               # an id that cannot be a set member is unusable
                if msg.get("ev") == "cancel":
                    cancelled_ids.add(msg.get("id"))
                elif "job" in msg:
                    jobs.put(msg)
            except Exception:  # noqa: BLE001 -- one bad line must never end the reader thread
                logging.getLogger("backend.worker").warning("unreadable input line %r", line[:200])
                continue
        # stdin ended: the backend closed it on purpose, or the backend is gone.
        if orphan.eof_exits():
            os._exit(0)

    threading.Thread(target=_read_stdin, daemon=True, name="worker-stdin").start()

    if os.environ.get("ORIENTA_WORKER_TEST_JOBS", "").strip() == "1":
        time.sleep(float(os.environ.get("ORIENTA_WORKER_TEST_START_DELAY", "0") or 0))
    send({"ev": "ready", "pid": os.getpid()})

    while True:
        req = jobs.get()
        _run(req, send, cancelled_ids, lookup, JobContext, JobError, protocol)


def _run(req, send, cancelled_ids, lookup, JobContext, JobError, protocol) -> None:
    job_id = req.get("id")
    name = req.get("job")
    try:
        spec = lookup(name)
    except (KeyError, TypeError):             # TypeError: a name that is not even hashable
        _send_error(send, job_id, "unknownJob", f"unknown job {name!r}")
        return
    ctx = JobContext(job_id, send, cancelled_ids)
    if ctx.cancelled():
        # Cancelled while it waited in the queue: answer at once, so the
        # backend does not have to kill a worker that never started it.
        send({"ev": "cancelled", "id": job_id})
        return
    try:
        module = importlib.import_module(spec.module)
        # A job names its heavy imports (torch, kikuchipy) in `preload`, so they
        # are done HERE and the backend can tell "still importing" from "computing":
        # `ready` is sent before any job module is imported, and the slow part of a
        # cold start (a virus scanner reading torch's DLLs) comes after it.
        preload = getattr(module, "preload", None)
        if preload is not None:
            preload()
        send({"ev": "imported", "id": job_id})
        result = getattr(module, spec.func)(req.get("args") or {}, ctx)
    except JobError as e:
        _send_error(send, job_id, e.code, e.message)
        return
    except Exception as e:  # noqa: BLE001 -- report every failure, keep the worker
        logging.getLogger("backend.worker").error(
            "job %s failed:\n%s", name, traceback.format_exc())
        first = (str(e).strip().splitlines() or [type(e).__name__])[0][:300]
        _send_error(send, job_id, "jobFailed", first)
        return
    if ctx.cancelled():
        send({"ev": "cancelled", "id": job_id})
        return
    try:
        message = protocol.result_message(job_id, result)
    except Exception as e:  # noqa: BLE001 -- a result the protocol cannot carry must not end the worker
        logging.getLogger("backend.worker").error(
            "job %s returned a result that cannot be sent:\n%s", name, traceback.format_exc())
        first = (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
        _send_error(send, job_id, "jobFailed",
                    f"the result of job {name!r} is not JSON-serialisable "
                    f"({type(e).__name__}: {first})")
        return
    send(message)


def _send_error(send, job_id, code, message) -> None:
    """Answer a failed job. A text that cannot be encoded as UTF-8 (a lone surrogate)
    is sent in its ``ascii()`` form: the backend must get a reply, and an error text
    must never be able to end the worker."""
    try:
        send({"ev": "error", "id": job_id, "code": code, "message": message})
    except (UnicodeError, ValueError):
        send({"ev": "error", "id": job_id, "code": ascii(code)[1:-1],
              "message": ascii(message)[1:-1]})


class _PipeHandler(logging.Handler):
    """Log records travel to the backend; the worker never opens orienta.log."""

    def __init__(self, send):
        super().__init__(logging.INFO)
        self._send = send

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._send({"ev": "log", "name": record.name, "level": record.levelno,
                        "msg": self.format(record)})
        except Exception:  # noqa: BLE001 -- a log line must never take a job down
            pass


def _forward_logging(send) -> None:
    handler = _PipeHandler(send)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)


if __name__ == "__main__":
    main()
