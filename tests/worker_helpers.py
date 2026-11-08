"""Helpers for tests that run a real compute worker."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from backend.process_watch import started_at

ROOT = Path(__file__).resolve().parents[1]


def worker_env(**extra) -> dict:
    env = dict(os.environ)
    env["ORIENTA_WORKER_TEST_JOBS"] = "1"
    env["ORIENTA_WORKER_PARENT_PID"] = str(os.getpid())
    env["ORIENTA_WORKER_PARENT_START"] = repr(started_at(os.getpid()))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def spawn_raw_worker(**extra) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "backend.worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(ROOT), env=worker_env(**extra))


def send(proc: subprocess.Popen, msg: dict) -> None:
    proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
    proc.stdin.flush()


class LineReader:
    """Reads lines from a pipe in a thread so a test can wait with a timeout."""

    def __init__(self, stream):
        self._q: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._pump, args=(stream,), daemon=True).start()

    def _pump(self, stream):
        for line in stream:
            self._q.put(line)
        self._q.put(None)

    def next(self, timeout: float = 30.0):
        line = self._q.get(timeout=timeout)
        return None if line is None else json.loads(line.decode("utf-8"))

    def until(self, pred, timeout: float = 30.0) -> dict:
        end = time.monotonic() + timeout
        while True:
            msg = self.next(max(0.01, end - time.monotonic()))
            if msg is None:
                raise EOFError("the worker closed its output")
            if pred(msg):
                return msg


def gone_within(pid: int, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def kill_quietly(pid) -> None:
    if pid is None:
        return
    try:
        psutil.Process(pid).kill()
    except psutil.Error:
        pass
