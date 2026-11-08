"""What pymatgen read from a CIF, kept on disk across backend starts.

Every start parsed the whole crystal library with pymatgen twice before the
Indexing page could list a phase: the phase-file discovery for its compositions
(about 12 s for 46 CIFs, measured 2026-11-08) and the Crystal Hint index for its
cells (11.2 s), plus the same again for the second indexing method the page
asks for. The files had not changed since the last start; only the in-memory
caches were gone.

An entry is reused only for the same file (resolved path), the same size and
the same modification time to the nanosecond, the same pymatgen version and the
same ``kind`` (which names the reading code and its version: change the reading,
change the kind). Anything else is parsed again. A cache that cannot be read or
written costs the time, never a wrong answer: every failure falls back to
parsing.

``ORIENTA_CIF_PARSE_CACHE`` names the cache file; ``0`` switches the cache off
(the test suite does, so a test that patches the parser sees its patch).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_entries: dict | None = None


def _cache_file() -> Path | None:
    raw = os.environ.get("ORIENTA_CIF_PARSE_CACHE", "").strip()
    if raw == "0":
        return None
    if raw:
        return Path(raw)
    return Path.home() / ".orienta" / "cache" / "cif_parse.json"


def _pymatgen_version() -> str:
    try:
        from importlib.metadata import version
        return version("pymatgen")
    except Exception:
        return "unknown"


def _load(file: Path) -> dict:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("pymatgen") == _pymatgen_version():
            return data.get("entries") or {}
    except FileNotFoundError:
        pass
    except Exception:
        logger.info("CIF parse cache %s unreadable; starting a new one", file, exc_info=True)
    return {}


def _save(file: Path, entries: dict) -> None:
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"pymatgen": _pymatgen_version(), "entries": entries}),
                       encoding="utf-8")
        os.replace(tmp, file)
    except Exception:
        logger.info("CIF parse cache %s not written", file, exc_info=True)


def cached(kind: str, cif_path, compute: Callable[[Path], Any],
           keep: Callable[[Any], bool] = lambda value: True) -> Any:
    """``compute(path)`` for this file, from the cache when the file is unchanged.

    ``compute`` must return something JSON can hold; a tuple comes back as a list. A value
    ``keep`` rejects (a failed parse, which may be a missing library rather than the file)
    is returned and not stored."""
    path = Path(cif_path)
    file = _cache_file()
    if file is None:
        return compute(path)
    try:
        st = path.stat()
        key = f"{kind}|{os.path.normcase(str(path.resolve()))}"
        stamp = [st.st_size, st.st_mtime_ns]
    except OSError:
        return compute(path)
    global _entries
    with _lock:
        if _entries is None:
            _entries = _load(file)
        hit = _entries.get(key)
    if hit is not None and hit.get("stamp") == stamp:
        return hit.get("value")
    value = compute(path)
    if not keep(value):
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return value                    # not storable: computed, not cached
    with _lock:                         # one writer at a time: the file is replaced whole
        _entries[key] = {"stamp": stamp, "value": value}
        _save(file, _entries)
    return value


def _reset_for_tests() -> None:
    global _entries
    with _lock:
        _entries = None
