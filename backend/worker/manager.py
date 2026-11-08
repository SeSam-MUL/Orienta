"""The backend's compute worker: start it, keep it warm, retire it, cancel it.

One worker process at a time, one job at a time. A dispatcher thread
owns the worker: it starts it (so on Linux the parent-death signal is tied to a
thread that lives as long as the backend), sends it one job, waits for the
outcome, and retires the worker after a heavy job or after ``idle_s`` without
work. Cancelling a job the worker has been given sends it a cancel message and,
after ``grace_s``, kills the worker; the next job starts a fresh one. Cancelling
a job that is still waiting for a starting worker only answers the job: the
worker goes on starting and is taken by the next job, or ended as idle. A worker
that dies without an answer gives the job the code ``workerCrashed``; its stderr
is in the log.
"""
from __future__ import annotations

import collections
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import psutil

from backend.process_watch import started_at
from backend.worker import protocol
from backend.worker.jobs import lookup, test_switches

logger = logging.getLogger(__name__)
_stderr_log = logging.getLogger("backend.worker.stderr")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TERMINAL = ("done", "error", "cancelled")
_STOPPED_TEXT = "The compute process manager has stopped."
# What `_await_ready` tells `_run_one`.
_READY, _ANSWERED, _RETRY = "ready", "answered", "retry"


def worker_enabled() -> bool:
    """False only under ORIENTA_NO_WORKER=1, which exists for the unit tests."""
    return os.environ.get("ORIENTA_NO_WORKER", "").strip() != "1"


class WorkerError(Exception):
    code = "workerError"

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class WorkerCrashed(WorkerError):
    code = "workerCrashed"


class WorkerStartFailed(WorkerError):
    code = "workerStartFailed"


class JobCancelled(WorkerError):
    code = "cancelled"


class JobFailed(WorkerError):
    code = "jobFailed"


class UnknownJob(WorkerError):
    code = "unknownJob"


@dataclass
class _Record:
    id: str
    job: str
    args: dict
    heavy: bool
    status: str = "queued"
    progress: Optional[float] = None
    message: str = ""
    message_code: Optional[str] = None
    result: object = None
    code: Optional[str] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    cancel_at: Optional[float] = None
    done: threading.Event = field(default_factory=threading.Event)
    #: Set by the first message the worker sends for this job. Until then the job
    #: is still "starting" -- the worker may still be importing torch.
    started: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict:
        snap = {"id": self.id, "job": self.job, "status": self.status,
                "progress": self.progress, "message": self.message,
                "message_code": self.message_code, "code": self.code,
                "error": self.error, "created": self.created, "finished": self.finished}
        if self.status == "done":
            snap["result"] = self.result
        return snap


class _ReleaseRequest:
    """A request in the dispatcher's queue to end the warm worker if it is idle."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.released = False


class _Worker:
    """One worker process and the two threads that read its output."""

    def __init__(self, proc: subprocess.Popen, on_message: Callable):
        self.proc = proc
        self.spawned_at = time.monotonic()
        self.real_pid: Optional[int] = None
        self.ready = threading.Event()
        self.exited = threading.Event()
        self.stderr_done = threading.Event()
        self.stderr_tail: "collections.deque[str]" = collections.deque(maxlen=200)
        self._write_lock = threading.Lock()
        threading.Thread(target=self._read_stdout, args=(on_message,), daemon=True,
                         name=f"worker-{proc.pid}-out").start()
        threading.Thread(target=self._read_stderr, daemon=True,
                         name=f"worker-{proc.pid}-err").start()

    def send(self, msg: dict) -> bool:
        with self._write_lock:
            try:
                self.proc.stdin.write(protocol.encode(msg))
                self.proc.stdin.flush()
                return True
            except (OSError, ValueError):
                return False

    def close_stdin(self) -> None:
        with self._write_lock:
            try:
                self.proc.stdin.close()
            except (OSError, ValueError):
                pass

    def kill(self) -> None:
        # The process Popen started may be a launcher (Windows venv python.exe);
        # the one that computes is the PID it reported in `ready`. Before `ready`
        # that PID is not known, so the launcher's children are collected first:
        # once the launcher is dead they no longer show up as its children.
        #
        # A worker that is already gone gets no signal: its PID may belong to an
        # unrelated process by now. Gone is `exited` (its output ended and it was
        # waited for) or, for the process Popen started, a return code -- that one
        # has been reaped, so on POSIX its PID is free. A dead launcher with a
        # live child (`real_pid` is another PID, `exited` still unset) still has
        # that child ended.
        if self.exited.is_set():
            return
        launcher_gone = self.proc.poll() is not None
        victims = []
        if not launcher_gone:
            try:
                victims += psutil.Process(self.proc.pid).children(recursive=True)
            except psutil.Error:
                pass
        pids = {self.real_pid, self.proc.pid} - {None}
        if launcher_gone:
            pids.discard(self.proc.pid)
        for pid in pids:
            try:
                victims.append(psutil.Process(pid))
            except psutil.Error:
                pass
        for victim in victims:
            try:
                victim.kill()
            except psutil.Error:
                pass

    def _read_stdout(self, on_message: Callable) -> None:
        for line in self.proc.stdout:
            try:
                msg = protocol.decode(line)
            except ValueError:
                _stderr_log.warning("[worker %d] unreadable protocol line %r", self.proc.pid, line[:200])
                continue
            if not isinstance(msg, dict):
                _stderr_log.warning("[worker %d] protocol line is not an object: %r",
                                    self.proc.pid, line[:200])
                continue
            if msg.get("ev") == "ready":
                self.real_pid = int(msg.get("pid") or self.proc.pid)
                self.ready.set()
                continue
            try:
                on_message(self, msg)
            except Exception:  # noqa: BLE001 -- one bad message must not end the reader
                logger.exception("compute worker %d: handling %r failed", self.proc.pid, msg.get("ev"))
        self.proc.wait()
        self.exited.set()

    def _read_stderr(self) -> None:
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.stderr_tail.append(line)
                _stderr_log.info("[worker %d] %s", self.proc.pid, line)
        self.stderr_done.set()


class WorkerManager:
    def __init__(self, *, idle_s=None, slow_start_s=None, start_limit_s=None,
                 grace_s: float = 1.0, keep_s: float = 600.0, argv=None):
        env = os.environ.get
        self.idle_s = float(idle_s if idle_s is not None else env("ORIENTA_WORKER_IDLE_S", "180"))
        self.slow_start_s = float(slow_start_s if slow_start_s is not None
                                  else env("ORIENTA_WORKER_SLOW_START_S", "15"))
        self.start_limit_s = float(start_limit_s if start_limit_s is not None
                                   else env("ORIENTA_WORKER_START_LIMIT_S", "120"))
        self.grace_s = float(grace_s)
        self.keep_s = float(keep_s)
        self.argv = list(argv) if argv else [sys.executable, "-m", "backend.worker"]
        self._records: "dict[str, _Record]" = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[_Worker] = None
        self._stopped = False
        self._thread = threading.Thread(target=self._dispatch, daemon=True, name="worker-dispatch")
        self._thread.start()

    # ---- public -------------------------------------------------------
    @property
    def worker_pid(self) -> Optional[int]:
        w = self._worker
        return None if w is None else (w.real_pid or w.proc.pid)

    def submit(self, job: str, args: Optional[dict] = None) -> str:
        try:
            spec = lookup(job)
        except KeyError:
            raise UnknownJob(f"unknown job {job!r}") from None
        args = dict(args or {})
        rec = _Record(id=uuid.uuid4().hex[:12], job=job, args=args, heavy=spec.is_heavy(args))
        with self._lock:
            if self._stopped:
                # Otherwise run() would wait for ever on a job nobody dispatches.
                # Checked under the lock that shutdown() takes to set the flag, so a
                # job is either refused here or in the records before the sweep.
                raise WorkerError(_STOPPED_TEXT, code="workerStopped")
            self._prune()
            self._records[rec.id] = rec
            self._queue.put(rec.id)
        return rec.id

    def get(self, job_id: str) -> Optional[dict]:
        rec = self._records.get(job_id)
        return None if rec is None else rec.snapshot()

    def wait(self, job_id: str, timeout: float) -> Optional[dict]:
        rec = self._records.get(job_id)
        if rec is None:
            return None
        rec.done.wait(max(0.0, float(timeout)))
        return rec.snapshot()

    def active(self, job: Optional[str] = None) -> list:
        with self._lock:
            recs = [r for r in self._records.values() if r.status not in TERMINAL]
        out = []
        for r in recs:
            if job is None or r.job == job:
                snap = r.snapshot()
                snap.pop("result", None)
                out.append(snap)
        return out

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None or rec.status in TERMINAL:
                return False
            if rec.status == "queued":
                self._finish_locked(rec, "cancelled")
                return True
            if rec.cancel_at is None:
                rec.cancel_at = time.monotonic()
            worker = self._worker
        if worker is not None:
            worker.send({"ev": "cancel", "id": job_id})
        return True

    def run(self, job: str, args: Optional[dict] = None,
            cancelled: Optional[Callable[[], bool]] = None, poll: float = 0.2):
        jid = self.submit(job, args)
        rec = self._records[jid]
        while not rec.done.wait(poll):
            if cancelled is not None and rec.cancel_at is None and cancelled():
                self.cancel(jid)
        return _outcome(rec)

    def release_idle_worker(self, timeout: float = 15.0) -> bool:
        """Ends a warm worker that has no job (running, starting or waiting) so the
        memory it holds (CUDA context, grids) goes back to the system. Leaves a
        worker that is busy alone; never starts one. True when a worker was ended.

        The dispatcher thread owns the worker, so the request goes through its
        queue and is looked at again there: a job that arrived in between keeps
        the worker. Waits up to ``timeout`` s for the worker to be gone."""
        req = _ReleaseRequest()
        with self._lock:
            if self._stopped or self._worker is None or self._has_active_locked():
                return False
            self._queue.put(req)
        req.done.wait(max(0.0, float(timeout)))
        return req.released

    def shutdown(self) -> None:
        with self._lock:
            self._stopped = True
        self._queue.put(None)
        w = self._worker
        if w is not None:
            w.kill()

    # ---- dispatcher thread --------------------------------------------
    def _dispatch(self) -> None:
        while not self._stopped:
            try:
                jid = self._queue.get(timeout=self.idle_s)
            except queue.Empty:
                self._retire("idle")
                continue
            if jid is None:
                break
            if isinstance(jid, _ReleaseRequest):
                self._release(jid)
                continue
            rec = self._records.get(jid)
            if rec is None or rec.status in TERMINAL:
                continue
            try:
                self._run_one(rec)
            except Exception:  # noqa: BLE001 -- the dispatcher must survive any job
                logger.exception("compute worker: dispatching %s failed", rec.job)
                self._finish(rec, "error", code="workerCrashed",
                             error="The compute process failed unexpectedly.")
        self._retire("shutdown")
        while True:                 # a release request nobody will answer any more
            try:
                left = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(left, _ReleaseRequest):
                left.done.set()
        # Whatever was still waiting will never run: answer it, so no run()/wait()
        # hangs. submit() refuses under the same lock, so nothing is added after this.
        with self._lock:
            for rec in self._records.values():
                self._finish_locked(rec, "error", code="workerStopped", error=_STOPPED_TEXT)

    def _run_one(self, rec: _Record) -> None:
        with self._lock:
            if rec.status in TERMINAL:
                return          # cancelled between the dispatcher's look and this lock
            rec.status = "starting"
        # Two attempts: a warm worker may have died while idle (killed by the
        # OS or a user) without its exit being noticed yet. Writing the job
        # then fails, and a fresh worker takes it instead -- unless the job was
        # cancelled or the manager stopped meanwhile (see the second attempt).
        window_start = None
        for _attempt in (1, 2):
            if _attempt == 2 and self._answer_instead_of_starting(rec):
                return
            worker = self._worker
            window_start = time.monotonic()         # a warm worker: from the moment it gets the job
            inherited = True        # a worker this job did not start
            if worker is None or worker.exited.is_set():
                self._worker = None
                worker = self._spawn(rec)
                if worker is None:
                    return
                inherited = False
            if not worker.ready.is_set():
                # A fresh worker, or one that an earlier job's cancel left starting.
                window_start = worker.spawned_at    # from its launch
                outcome = self._await_ready(worker, rec, inherited)
                if outcome == _ANSWERED:
                    return
                if outcome == _RETRY:
                    continue
            with self._lock:
                if rec.status in TERMINAL:
                    return
                if rec.cancel_at is not None:
                    # Cancelled while the worker was starting: never send it.
                    self._finish_locked(rec, "cancelled")
                    return
            if worker.send({"job": rec.job, "id": rec.id, "args": rec.args}):
                # "running" means the worker has the job. A cancel that lands before
                # this point sends its message first; the worker answers it at once.
                # The job may already be finished (a fast one): never go back.
                with self._lock:
                    if rec.status == "starting":
                        rec.status = "running"
                break
            worker.kill()
            worker.exited.wait(2.0)
            self._worker = None
        else:
            self._finish(rec, "error", code="workerCrashed",
                         error="The compute process could not take the job.")
            return
        while not rec.done.wait(0.05):
            if worker.exited.is_set():
                if rec.cancel_at is not None:
                    self._finish(rec, "cancelled")
                else:
                    self._crashed(rec, worker)
                break
            if rec.cancel_at is not None and time.monotonic() - rec.cancel_at > self.grace_s:
                worker.kill()
                worker.exited.wait(5.0)
            if rec.cancel_at is None and not rec.started.is_set():
                if self._watch_start_window(rec, worker, window_start):
                    break
        if rec.heavy or worker.exited.is_set():
            self._retire("heavy job" if rec.heavy else "worker ended")

    def _answer_instead_of_starting(self, rec: _Record) -> bool:
        """Before the second attempt: whether the job is answered now, so that no
        new worker is started for it. The first attempt's worker is gone (it died,
        or it was ended by `shutdown()`), and the job may have been cancelled or the
        manager stopped since; a worker started now would outlive the job and sit
        until the idle limit (a stopped manager would even run the job).
        Decided under the lock `cancel()` and `shutdown()` take, like the check
        before the job is sent."""
        with self._lock:
            if rec.status in TERMINAL:
                return True
            if self._stopped:
                self._finish_locked(rec, "error", code="workerStopped", error=_STOPPED_TEXT)
                return True
            if rec.cancel_at is not None:
                self._finish_locked(rec, "cancelled")
                return True
        return False

    def _watch_start_window(self, rec: _Record, worker: _Worker, window_start: float) -> bool:
        """Applies the slow-start note and the start limit while the job is still
        "starting": from the worker's launch (or, for a warm worker, from the moment
        it got the job) to the job's first message. That first message follows the
        job's heavy imports, so a cold start's slow part (a virus scanner reading
        torch's DLLs) is inside this window. True once the job was given up."""
        elapsed = time.monotonic() - window_start
        if elapsed > self.start_limit_s:
            with self._lock:
                # Decided under the lock the first message and cancel() take: a
                # first message or a cancel that landed since the caller looked is
                # not a failed start. A cancelled job goes through the grace kill.
                if rec.started.is_set() or rec.cancel_at is not None or rec.status in TERMINAL:
                    return False
                self._finish_locked(rec, "error", code="workerStartFailed",
                                    error=f"The compute process did not start within "
                                          f"{self.start_limit_s:.0f} s.")
            logger.error("compute worker %s: job %s sent nothing within %.0f s; ending the worker",
                         worker.real_pid or worker.proc.pid, rec.job, self.start_limit_s)
            worker.kill()
            worker.exited.wait(5.0)
            return True
        if elapsed > self.slow_start_s:
            with self._lock:
                # Re-checked under the lock the first message takes: a note must
                # not appear after the job has started.
                if not rec.started.is_set() and rec.message_code != "workerSlowStart":
                    rec.message_code = "workerSlowStart"
                    rec.message = "The compute process is starting slowly."
                    logger.warning("compute worker %s: job %s not started after %.0f s",
                                   worker.real_pid or worker.proc.pid, rec.job, elapsed)
        return False

    def _spawn(self, rec: _Record) -> Optional[_Worker]:
        """Launches a worker and makes it the current one; None (the job is
        answered) when the process cannot be started at all."""
        env = dict(os.environ)
        env["ORIENTA_WORKER_PARENT_PID"] = str(os.getpid())
        # The same function the worker compares with (see backend.process_watch).
        env["ORIENTA_WORKER_PARENT_START"] = repr(started_at(os.getpid()))
        env["PYTHONUNBUFFERED"] = "1"
        # The worker's stderr is read as UTF-8 below; on Windows a pipe would
        # otherwise carry the ANSI code page and every umlaut in a trace or a path
        # would arrive as a replacement character. Only the standard streams:
        # PYTHONUTF8 would also change what open() reads by default, in the worker
        # and in everything it starts.
        env["PYTHONIOENCODING"] = "utf-8"
        kwargs = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      cwd=str(PROJECT_ROOT), env=env)
        # Decided before Popen: a suspended start is only safe when the job
        # object code below will resume the process. Suspended, a venv launcher
        # cannot spawn the real interpreter before it is inside the job, so the
        # interpreter inherits the membership (see winjob.kill_with_this_process).
        use_job = sys.platform == "win32" and "jobobject" not in test_switches()
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW
            if use_job:
                from backend.worker import winjob
                flags |= winjob.CREATE_SUSPENDED
            kwargs["creationflags"] = flags
        try:
            proc = subprocess.Popen(self.argv, **kwargs)
        except OSError as e:
            self._finish(rec, "error", code="workerStartFailed",
                         error=f"The compute process could not be started: {e}")
            return None
        if use_job:
            try:
                tied = winjob.kill_with_this_process(proc, resume=True)
            except Exception:  # noqa: BLE001 -- degraded, not fatal; the other guards remain
                logger.exception("compute worker %d: the job object failed", proc.pid)
                tied = False
            if not tied:
                logger.warning("compute worker %d is not tied to the backend by a job "
                               "object; the other guards remain", proc.pid)
        worker = _Worker(proc, self._on_message)
        self._worker = worker
        return worker

    def _await_ready(self, worker: _Worker, rec: _Record, inherited: bool = False) -> str:
        """Waits for a starting worker to say `ready`, for ``rec``. `_READY` when it
        did; `_ANSWERED` when the job was answered instead (the worker ended, the
        start limit passed, or the job was cancelled).

        ``inherited``: the worker was left starting by an earlier, cancelled job.
        Its start limit still counts from its launch, but a failure of it is not
        this job's: the worker is ended and `_RETRY` tells `_run_one` to take a
        fresh worker, which has a start limit of its own."""
        proc = worker.proc
        t0 = worker.spawned_at
        slow = False
        while not worker.ready.wait(0.1):
            elapsed = time.monotonic() - t0
            if worker.exited.is_set():
                worker.stderr_done.wait(2.0)
                logger.error("compute worker %d ended before it was ready (exit code %s):\n%s",
                             proc.pid, proc.returncode, "\n".join(worker.stderr_tail))
                self._worker = None
                if inherited:
                    return _RETRY
                self._finish(rec, "error", code="workerStartFailed",
                             error="The compute process ended while starting.")
                return _ANSWERED
            if rec.cancel_at is not None:
                # The job has not reached the worker, so nothing computes there:
                # only the job ends. This is looked at only before `ready`, and the
                # heavy imports (torch) run after it, so the worker has only just
                # begun; it goes on starting for the next job, or is retired as idle.
                self._finish(rec, "cancelled")
                return _ANSWERED
            if not slow and elapsed > self.slow_start_s:
                slow = True
                with self._lock:
                    rec.message_code = "workerSlowStart"
                    rec.message = "The compute process is starting slowly."
                logger.warning("compute worker %d: not ready after %.0f s", proc.pid, elapsed)
            if elapsed > self.start_limit_s:
                worker.kill()
                self._worker = None
                if inherited:
                    logger.warning("compute worker %d left starting by an earlier job did not "
                                   "become ready within %.0f s; ending it for a fresh one",
                                   proc.pid, self.start_limit_s)
                    worker.exited.wait(5.0)
                    return _RETRY
                self._finish(rec, "error", code="workerStartFailed",
                             error=f"The compute process did not start within "
                                   f"{self.start_limit_s:.0f} s.")
                return _ANSWERED
        # A launcher's child (the real worker) is in the job by inheritance;
        # what it needs from us is that kill() targets real_pid (see _Worker).
        # A slow-start note set above stays: "starting" ends with the job's first
        # message (see _run_one), not with `ready`, which is sent before the job's
        # heavy imports.
        logger.info("compute worker %d ready after %.1f s", worker.real_pid, time.monotonic() - t0)
        return _READY

    def _has_active_locked(self) -> bool:
        return any(r.status not in TERMINAL for r in self._records.values())

    def _release(self, req: "_ReleaseRequest") -> None:
        """Dispatcher thread: the queued release request. Only here is the check
        final, because only this thread starts workers and hands them jobs."""
        try:
            with self._lock:
                busy = self._has_active_locked()
            w = self._worker
            if not busy and w is not None:
                # A worker that had already ended is tidied up too, but nothing was
                # freed by it: the answer is False, not "released".
                alive = not w.exited.is_set()
                self._retire("release requested")
                req.released = alive
        finally:
            req.done.set()

    def _retire(self, reason: str) -> None:
        w = self._worker
        if w is None:
            return
        if not w.exited.is_set():
            w.close_stdin()
            if not w.exited.wait(5.0):
                w.kill()
                w.exited.wait(5.0)
        logger.info("compute worker %s retired (%s)", w.real_pid or w.proc.pid, reason)
        self._worker = None

    # ---- reader-thread callbacks --------------------------------------
    def _on_message(self, worker: _Worker, msg: dict) -> None:
        ev = msg.get("ev")
        if ev == "log":
            logging.getLogger(msg.get("name") or "backend.worker").log(
                int(msg.get("level") or logging.INFO), "[worker %s] %s",
                worker.real_pid or worker.proc.pid, msg.get("msg", ""))
            return
        rec = self._records.get(msg.get("id"))
        if rec is None:
            return
        if not rec.started.is_set():
            # The job's first message: the worker has imported what the job needs.
            with self._lock:
                rec.started.set()
                if rec.message_code == "workerSlowStart":
                    rec.message_code = None
                    rec.message = ""
        if ev == "progress":
            with self._lock:
                rec.progress = msg.get("frac")
                rec.message = msg.get("msg", "")
        elif ev == "result":
            try:
                rec.result = protocol.read_result(msg)
            except (OSError, ValueError) as e:
                self._finish(rec, "error", code="jobFailed", error=f"unreadable result: {e}")
                return
            self._finish(rec, "done")
        elif ev == "error":
            self._finish(rec, "error", code=msg.get("code") or "jobFailed",
                         error=msg.get("message", ""))
        elif ev == "cancelled":
            self._finish(rec, "cancelled")

    # ---- bookkeeping --------------------------------------------------
    def _finish(self, rec: _Record, status: str, code=None, error=None) -> None:
        with self._lock:
            self._finish_locked(rec, status, code, error)

    def _finish_locked(self, rec: _Record, status: str, code=None, error=None) -> None:
        if rec.status in TERMINAL:
            return
        rec.status = status
        rec.code = code
        rec.error = error
        rec.finished = time.time()
        rec.done.set()

    def _crashed(self, rec: _Record, worker: _Worker) -> None:
        worker.stderr_done.wait(2.0)        # the last lines before the death belong in the report
        rc = worker.proc.returncode
        logger.error("compute worker %s died (exit code %s) while running %s %s:\n%s",
                     worker.real_pid or worker.proc.pid, rc, rec.job, rec.id,
                     "\n".join(worker.stderr_tail))
        self._finish(rec, "error", code="workerCrashed",
                     error=f"The compute process ended unexpectedly (exit code {rc}).")

    def _prune(self) -> None:
        now = time.time()
        stale = [k for k, r in self._records.items()
                 if r.status in TERMINAL and r.finished and now - r.finished > self.keep_s]
        for k in stale:
            del self._records[k]


def _outcome(rec: _Record):
    if rec.status == "done":
        return rec.result
    if rec.status == "cancelled":
        raise JobCancelled("cancelled")
    msg = rec.error or "the job failed"
    if rec.code == "workerCrashed":
        raise WorkerCrashed(msg)
    if rec.code == "workerStartFailed":
        raise WorkerStartFailed(msg)
    raise JobFailed(msg, code=rec.code or "jobFailed")


_MANAGER: Optional[WorkerManager] = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> WorkerManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = WorkerManager()
        return _MANAGER


def release_idle_worker(timeout: float = 15.0) -> bool:
    """Ends the warm compute worker when it has no job, so its GPU memory is free.

    For the "Release GPU" paths. Never starts a worker; does nothing when there is
    no manager yet, when the worker is off (ORIENTA_NO_WORKER=1), or when a job is
    running or waiting. True when a worker was ended."""
    if not worker_enabled():
        return False
    with _MANAGER_LOCK:
        m = _MANAGER
    return False if m is None else m.release_idle_worker(timeout)


def shutdown_manager() -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        m, _MANAGER = _MANAGER, None
    if m is not None:
        m.shutdown()
