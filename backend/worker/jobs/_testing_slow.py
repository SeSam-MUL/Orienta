"""A job module whose import is slow, like torch on a machine with a virus scanner.

Reachable only with ORIENTA_WORKER_TEST_JOBS=1. The delay is read at import time.
"""
from __future__ import annotations

import os
import time

time.sleep(float(os.environ.get("ORIENTA_WORKER_TEST_IMPORT_S", "0") or 0))


def preload():
    """Heavy imports a job names (see backend.worker.__main__._run)."""
    time.sleep(float(os.environ.get("ORIENTA_WORKER_TEST_PRELOAD_S", "0") or 0))


def echo(args, ctx):
    return args
