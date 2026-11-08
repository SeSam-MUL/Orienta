"""JSON lines between the backend and its compute worker.

One JSON object per line, UTF-8, in both directions. Paths and texts travel
only here, never on a command line. A result larger than ``INLINE_LIMIT`` is
written to a file in a temp folder named after the worker's PID, and only the
path travels; the reader removes the file.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

#: Results above this many bytes of JSON travel as a file.
INLINE_LIMIT = 4 * 1024 * 1024


def encode(msg: dict) -> bytes:
    return (json.dumps(msg, ensure_ascii=False, allow_nan=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes) -> dict:
    return json.loads(line.decode("utf-8"))


def spill_dir() -> Path:
    d = Path(tempfile.gettempdir()) / f"orienta-worker-{os.getpid()}"
    d.mkdir(exist_ok=True)
    return d


def result_message(job_id: str, result) -> dict:
    """The ``result`` message for ``result``: inline, or as a file above the limit."""
    body = json.dumps(result, ensure_ascii=False, allow_nan=True, separators=(",", ":"))
    if len(body.encode("utf-8")) <= INLINE_LIMIT:
        return {"ev": "result", "id": job_id, "result": result}
    path = spill_dir() / f"{job_id}.json"
    path.write_text(body, encoding="utf-8")
    return {"ev": "result", "id": job_id, "result_file": str(path)}


def read_result(msg: dict):
    """The result a ``result`` message carries; a spilled file is removed after reading."""
    if "result_file" in msg:
        path = Path(msg["result_file"])
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        finally:
            path.unlink(missing_ok=True)
            try:
                path.parent.rmdir()          # the worker's folder, once empty
            except OSError:
                pass
    return msg.get("result")
