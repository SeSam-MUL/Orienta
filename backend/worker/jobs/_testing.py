"""Jobs for the worker's own tests. Reachable only with ORIENTA_WORKER_TEST_JOBS=1."""
from __future__ import annotations

import logging
import os
import sys
import time

from backend.worker.jobs import JobError


def echo(args, ctx):
    return args


def sleep(args, ctx):
    if args.get("cooperative"):
        ctx.progress(0.0, "started")      # lets a test cancel only once the job is running
    end = time.monotonic() + float(args.get("s", 60))
    while time.monotonic() < end:
        if args.get("cooperative") and ctx.cancelled():
            return {"stopped": True}
        time.sleep(0.05)
    return {"slept": args.get("s", 60)}


def crash(args, ctx):
    sys.stderr.write("about to crash on purpose\n")
    sys.stderr.flush()
    os._exit(int(args.get("code", 3)))


def alloc(args, ctx):
    import numpy as np
    import psutil
    a = np.ones(int(args["mb"]) * 2**20 // 8)        # np.ones writes every page
    return {"rss_mb": psutil.Process().memory_info().rss / 2**20, "first": float(a[0])}


def log(args, ctx):
    logging.getLogger(args.get("logger", "backend.worker.jobs._testing")).warning(
        args.get("msg", "hello from the worker"))
    return {}


def stat(args, ctx):
    path = args["path"]
    return {"path": path, "exists": os.path.isfile(path)}


def progress(args, ctx):
    n = int(args.get("n", 5))
    for i in range(n):
        ctx.progress(i / n, f"step {i}")
        time.sleep(float(args.get("dt", 0.05)))
    return {"steps": n}


def raise_coded(args, ctx):
    if args.get("surrogate"):                     # text that cannot be encoded as UTF-8
        args = dict(args, msg=args.get("msg", "") + "\udcff")
    if args.get("plain"):
        raise ValueError(args["msg"])
    raise JobError(args.get("code", "testCode"), args.get("msg", "a coded failure"))


def big(args, ctx):
    return {"blob": "x" * int(args["bytes"])}


def print_everywhere(args, ctx):
    # No flush of sys.stdout: if it were not pointed at stderr, this text would sit
    # in a block buffer and the test would never see it.
    print("printed by python")
    sys.stdout.write("written to sys.stdout\n")
    os.write(1, b"written to fd 1\n")
    if sys.platform == "win32":
        _write_to_stdout_handle(b"written to the stdout handle\n")
    return {"ok": True}


def _write_to_stdout_handle(data: bytes) -> None:
    """A C runtime other than Python's writes through the Win32 handle, not through fd 1."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetStdHandle.argtypes = [wintypes.DWORD]
    k32.GetStdHandle.restype = wintypes.HANDLE
    k32.WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_char_p, wintypes.DWORD,
                              ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    k32.WriteFile.restype = wintypes.BOOL
    handle = k32.GetStdHandle(wintypes.DWORD(-11 & 0xFFFFFFFF))
    written = wintypes.DWORD(0)
    if not k32.WriteFile(handle, data, len(data), ctypes.byref(written), None):
        raise OSError(ctypes.get_last_error(), "WriteFile on the standard output handle failed")


def encodings(args, ctx):
    return {"utf8_mode": sys.flags.utf8_mode, "stderr": sys.stderr.encoding}


def umlaut(args, ctx):
    sys.stderr.write("Größe [Pearson] {20kV} ä ö ü\n")
    sys.stderr.flush()
    return {}


def pid(args, ctx):
    return {"pid": os.getpid()}


def bad_result(args, ctx):
    """A result the protocol cannot carry: not JSON, or text that is not valid UTF-8."""
    if args.get("kind") == "surrogate":
        return {"x": "lone surrogate \ud800"}
    return {"x": object()}
