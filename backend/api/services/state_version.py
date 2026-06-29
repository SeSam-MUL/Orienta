"""Global monotonic state-version counter.

Bumped on any backend change that affects how the ACTIVE indexing result
renders (new/activated result, cleanup, refinement, frame change, file switch).
The detached pole-figure window polls GET /api/state-version and refetches only
when this number (or the active result id) changes. In-memory only — a backend
restart resets it to 0, which is fine: results are lost on restart anyway.
"""
from __future__ import annotations

import threading

_version = 0
_lock = threading.Lock()


def bump() -> int:
    """Increment the version and return the new value (thread-safe)."""
    global _version
    with _lock:
        _version += 1
        return _version


def get() -> int:
    """Return the current version."""
    return _version
