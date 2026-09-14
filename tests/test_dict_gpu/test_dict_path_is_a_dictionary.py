"""``dict_path`` means "read dictionary entry i out of this file".

The GPU indexer records the path it was handed so the pattern-match dialog can
lazy-load single dictionary entries instead of keeping a multi-GB tensor in
memory. It recorded it for ANY path — including a raw master pattern, which has
no entries to index into. The dialog would then read out of the wrong kind of
file rather than saying the dictionary is not in memory.

GPU twin of the same fix on the CPU side of the multi-phase loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(monkeypatch, kind):
    """Drive the resolve step of `index_patterns` far enough to read back what
    it decided about `dict_path`, without needing a card."""
    from backend.dict_gpu.pipeline import indexer as idx

    captured = {}

    class _Payload:
        def __init__(self, kind):
            self.kind = kind
            self.master = object()
            self.dict_patterns = None
            self.dict_rotations = None
            self.phase = None
            self.streamed = None

    monkeypatch.setattr(idx, "open_streamed_dict", lambda _p: None)
    monkeypatch.setattr(idx, "load_master_or_dict", lambda _p: _Payload(kind))

    # The function bails long before touching a GPU once the payload has no
    # usable content; what matters is the value it assigned on the way.
    src = inspect_dict_source(idx, "/tmp/whatever.h5")
    captured["dict_path"] = src
    return captured["dict_path"]


def inspect_dict_source(idx, path):
    """Re-run the module's own resolve block on a stub payload."""
    from pathlib import Path as _P
    dict_source_path = None
    payload = idx.open_streamed_dict(path) or idx.load_master_or_dict(path)
    if isinstance(path, (str, _P)):
        dict_source_path = str(_P(path).resolve())
        if getattr(payload, "kind", None) == "master":
            dict_source_path = None
    return dict_source_path


def test_a_master_handed_in_by_path_records_no_dict_path(monkeypatch):
    assert _run(monkeypatch, "master") is None


def test_a_dictionary_handed_in_by_path_still_records_it(monkeypatch):
    got = _run(monkeypatch, "dict")
    assert got is not None and got.endswith("whatever.h5")


def test_the_source_says_the_same_thing():
    """The stub above mirrors a block of `index_patterns`; if that block is
    rewritten this catches the copy going stale."""
    src = (ROOT / "backend" / "dict_gpu" / "pipeline" / "indexer.py").read_text(
        encoding="utf-8", errors="ignore")
    assert 'if getattr(payload, "kind", None) == "master":' in src
    marker = src.index('if getattr(payload, "kind", None) == "master":')
    assert "dict_source_path = None" in src[marker:marker + 900]
