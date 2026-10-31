"""Helpers for tests that drive the legacy batch route (``POST /api/indexing/batch/start``).

The route exports every result next to its source file unless the REQUEST says
``"auto_export": false``. Three tests sent ``"export_result": false`` inside the
dataset instead, a key ``BatchDatasetConfig`` does not have; pydantic dropped it
without a word, and every run wrote ``<stem>_indexed.h5`` into ``Test_data``.
``untouched(folder)`` turns such a write into a failure.

The route submits the batch to the executor and returns, so the batch is still
running when the POST answers. ``wait_for_batch`` returns only once
``/api/indexing/batch/status`` reports the run as over; a test that looks at the
folder or the registry should do so after that, not straight after the POST.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path


def _files(folder: Path) -> dict:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in folder.iterdir() if p.is_file()}


@contextmanager
def untouched(folder):
    """Fail if the body adds a file to ``folder`` or changes one already there."""
    folder = Path(folder)
    before = _files(folder)
    yield
    after = _files(folder)
    added = sorted(set(after) - set(before))
    changed = sorted(n for n in set(before) & set(after) if before[n] != after[n])
    assert not added and not changed, (
        f"the test wrote into {folder.name}: added {added}, changed {changed}")


def wait_for_batch(client, timeout: float = 300.0) -> dict:
    """Poll ``/api/indexing/batch/status`` until it reports the batch is no longer running.

    Returns that final status. Raises ``AssertionError`` if the batch is still running
    after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        status = client.get("/api/indexing/batch/status").json()
        if not status.get("running"):
            return status
        if time.monotonic() > deadline:
            raise AssertionError(f"batch still running after {timeout:.0f} s: {status}")
        time.sleep(0.05)
