"""What the compute worker can run.

A job is a function ``run(args: dict, ctx: JobContext) -> JSON-able``, named
here and imported only when it is asked for, so a worker loads only the chain
of the job it runs. ``args`` is plain JSON (paths, numbers). Whether a job is
heavy decides whether the worker is retired after it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


class JobError(Exception):
    """A failure the job names itself; ``code`` reaches the interface."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class JobContext:
    """What a running job can do: report progress, ask whether it was cancelled."""

    def __init__(self, job_id: str, send: Callable[[dict], None], cancelled_ids: set):
        self.id = job_id
        self._send = send
        self._cancelled = cancelled_ids

    def progress(self, frac, msg: str = "") -> None:
        self._send({"ev": "progress", "id": self.id,
                    "frac": None if frac is None else float(frac), "msg": str(msg)})

    def cancelled(self) -> bool:
        return self.id in self._cancelled


def _never(args: dict) -> bool:
    return False


def _always(args: dict) -> bool:
    return True


def _sphere_heavy(args: dict) -> bool:
    # High (256) and Max (384) are the computations that leave gigabytes behind.
    return int(args.get("max_bandwidth") or 128) >= 256


@dataclass(frozen=True)
class JobSpec:
    name: str
    module: str
    func: str
    heavy: Callable[[dict], bool]

    def is_heavy(self, args: dict) -> bool:
        return bool(self.heavy(args or {}))


JOBS = {
    "sphere": JobSpec("sphere", "backend.worker.jobs.sphere", "run", _sphere_heavy),
}

_T = "backend.worker.jobs._testing"
TEST_JOBS = {
    "_echo": JobSpec("_echo", _T, "echo", _never),
    "_sleep": JobSpec("_sleep", _T, "sleep", _never),
    "_sleep_heavy": JobSpec("_sleep_heavy", _T, "sleep", _always),
    "_crash": JobSpec("_crash", _T, "crash", _never),
    "_alloc": JobSpec("_alloc", _T, "alloc", _always),
    "_log": JobSpec("_log", _T, "log", _never),
    "_stat": JobSpec("_stat", _T, "stat", _never),
    "_progress": JobSpec("_progress", _T, "progress", _never),
    "_raise": JobSpec("_raise", _T, "raise_coded", _never),
    "_big": JobSpec("_big", _T, "big", _never),
    "_print": JobSpec("_print", _T, "print_everywhere", _never),
    "_pid": JobSpec("_pid", _T, "pid", _never),
    "_bad_result": JobSpec("_bad_result", _T, "bad_result", _never),
    "_encodings": JobSpec("_encodings", _T, "encodings", _never),
    "_umlaut": JobSpec("_umlaut", _T, "umlaut", _never),
    "_slow_import": JobSpec("_slow_import", "backend.worker.jobs._testing_slow", "echo", _never),
}


def _test_jobs_enabled() -> bool:
    return os.environ.get("ORIENTA_WORKER_TEST_JOBS", "").strip() == "1"


def lookup(name: str) -> JobSpec:
    """The job ``name``; test jobs only with ``ORIENTA_WORKER_TEST_JOBS=1``."""
    if name in JOBS:
        return JOBS[name]
    if _test_jobs_enabled() and name in TEST_JOBS:
        return TEST_JOBS[name]
    raise KeyError(name)


def test_switches() -> set:
    """Guards a test has switched off (``ORIENTA_WORKER_TEST_DISABLE``), only with test jobs on."""
    if not _test_jobs_enabled():
        return set()
    raw = os.environ.get("ORIENTA_WORKER_TEST_DISABLE", "")
    return {s.strip() for s in raw.split(",") if s.strip()}
