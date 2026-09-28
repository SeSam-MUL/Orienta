"""Names people give phases: one display name, any number of search terms.

Spec §2.4. A library key is a filename (`sd_0302719`, `Al4Fe1.7Si (τ11)`) and a
filename is not a name anyone says out loud. So a phase may carry:

* **one display name**, shown wherever the key is shown today, and
* **any number of further search terms**, which find the phase and appear
  nowhere.

THE FILENAME STAYS FINDABLE. A synonym stands in front of the identity, it does
not replace it: search still matches the key, and the card still shows it. A
rename that hid the file would make two people looking at one library unable to
talk about the same phase.

WHO AND WHEN, ALWAYS. In a shared library a name without an author is a silent
edit to someone else's work -- the store therefore records both, and the card
shows them. That is also why this lives in `Database/Synonyms/` and not in the
browser: the names travel with the library they describe, the way collections do.

SYNONYMS NEVER REACH AN EXPORT. `.ang`, `.ctf` and the light `.h5` carry the
phase names the indexing result was built with, and those come from the HDF5
group keys (`result_exporter`), not from this store. A nickname in a provenance
field would make a file's phase unfindable in anyone else's library.
`test_the_export_never_sees_a_synonym` pins it.

SEEDED, NOT REPLACED. `phase_metadata._PHASE_NICKNAMES` held five literature
nicknames in code. They are copied in on first read, authored
"Orienta (predefined)", and `phase_nickname` reads the store from then on. The
code table is NOT deleted: it is the seed, and deleting it would mean the
nicknames vanish the day someone wires up `build_canonical_label` -- which two
comments in this codebase already claim is wired, while the field they describe is
set from something else entirely.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = 1
_DIRNAME = "Synonyms"
_FILENAME = "synonyms.json"

#: Author recorded for the seeded literature nicknames, so the card can say the
#: name did not come from a person here.
SEED_AUTHOR = "Orienta (predefined)"

_DIR_OVERRIDE: Optional[Path] = None


def synonyms_dir() -> Path:
    if _DIR_OVERRIDE is not None:
        return _DIR_OVERRIDE
    from path_utils import get_local_database_path
    return get_local_database_path() / _DIRNAME


def set_synonyms_dir_for_test(p: Optional[Path]) -> None:
    """Test-only: redirect the store. ``None`` restores.

    Not a convenience: ``tests/conftest.py`` fails any test that writes under
    ``Database/``, and rightly -- a test must not add names to the maintainer's
    library. Same hook as the collections service.
    """
    global _DIR_OVERRIDE
    _DIR_OVERRIDE = Path(p) if p is not None else None


def _write_atomic(path: Path, text: str) -> None:
    """Temp file beside the target, then replace.

    ``newline="\\n"`` is the point, not tidiness: without it Python turns every
    ``\\n`` into CRLF on Windows and a store copied between machines differs on
    every line with no name changed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".synonyms_", suffix=".json.tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty() -> dict:
    return {"schema": SCHEMA, "phases": {}}


def _read_raw() -> dict:
    p = synonyms_dir() / _FILENAME
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or not isinstance(d.get("phases"), dict):
            raise ValueError("not a synonym store")
        return d
    except FileNotFoundError:
        return _empty()
    except Exception:
        # A broken store must not cost the names it still holds silently: say so
        # and carry on with none, rather than crashing the page that asked.
        logger.warning("synonym store at %s is unreadable; continuing without it",
                       p, exc_info=True)
        return _empty()


def _seed_entries() -> dict:
    """The literature nicknames, as store entries.

    Keyed by LIBRARY KEY, which the code table is not -- it is keyed by
    `(space group, sorted elements)`. So the seed is resolved against the real
    library, and a nickname whose pair matches no phase is simply not seeded.
    """
    from backend.api.services import crystal_hint_local_library as chl
    from phase_metadata import _PHASE_NICKNAMES

    out: dict = {}
    try:
        idx = chl.get_index()
    except Exception:
        logger.debug("cannot seed synonyms: no library index", exc_info=True)
        return out
    for key, entry in idx.items():
        sg = (entry.space_group or "").replace(" ", "")
        nick = _PHASE_NICKNAMES.get((sg, tuple(sorted(entry.elements))))
        if nick:
            out[key] = {"display_name": nick, "search_terms": [],
                        "author": SEED_AUTHOR, "updated": _now(),
                        "seeded": True}
    return out


def load() -> dict:
    """``{library key: {display_name, search_terms, author, updated, seeded}}``.

    Seeds the literature nicknames the first time, and WRITES that seeding, so
    the card can show an author for them and a user can edit or remove one like
    any other name. A failed write is not fatal -- the names are still returned,
    they are just recomputed next time.
    """
    raw = _read_raw()
    phases = raw.get("phases") or {}
    if not raw.get("seeded_at"):
        seeds = _seed_entries()
        for key, val in seeds.items():
            phases.setdefault(key, val)
        raw = {"schema": SCHEMA, "phases": phases, "seeded_at": _now()}
        try:
            _write_atomic(synonyms_dir() / _FILENAME,
                          json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("could not write the seeded synonym store",
                           exc_info=True)
    return phases


def get(key: str) -> dict:
    """One phase's names, or empty defaults."""
    return load().get(key) or {"display_name": None, "search_terms": [],
                               "author": None, "updated": None, "seeded": False}


def display_name(key: str) -> Optional[str]:
    return (get(key) or {}).get("display_name") or None


def search_terms(key: str) -> list[str]:
    return list((get(key) or {}).get("search_terms") or [])


def set_names(key: str, *, display_name: Optional[str] = None,
              search_terms: Optional[list[str]] = None,
              author: str = "") -> dict:
    """Give a phase a display name and/or search terms. Returns the new entry.

    `author` is required in substance: an empty one is stored as "unknown", never
    silently as the previous author, because inheriting an author would attribute
    one person's name to another.
    """
    if not str(key).strip():
        raise ValueError("a synonym needs a phase key")
    raw = _read_raw()
    if not raw.get("seeded_at"):
        load()                      # seed first, so a seed is not overwritten blind
        raw = _read_raw()
    phases = raw.get("phases") or {}
    entry = dict(phases.get(key) or {})

    if display_name is not None:
        name = str(display_name).strip()
        entry["display_name"] = name or None
    if search_terms is not None:
        seen, terms = set(), []
        for t in search_terms:
            t = str(t).strip()
            if t and t.casefold() not in seen:
                seen.add(t.casefold())
                terms.append(t)
        entry["search_terms"] = terms
    entry.setdefault("display_name", None)
    entry.setdefault("search_terms", [])
    entry["author"] = str(author).strip() or "unknown"
    entry["updated"] = _now()
    entry["seeded"] = False        # a person touched it; it is no longer the seed

    if not entry["display_name"] and not entry["search_terms"]:
        phases.pop(key, None)      # nothing left to remember
    else:
        phases[key] = entry
    raw = {"schema": SCHEMA, "phases": phases,
           "seeded_at": raw.get("seeded_at") or _now()}
    _write_atomic(synonyms_dir() / _FILENAME,
                  json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    return phases.get(key) or {"display_name": None, "search_terms": [],
                               "author": None, "updated": None, "seeded": False}
