"""Phase collections: the user's own filing of the crystal library.

WHY THIS EXISTS. The library is one flat list. Picking phases for a run means
reading 36 names, six of which are called `sd_1802610`, and the cost of getting
it wrong is not a slow run but a plausible wrong answer — ICDD publishes the
same reasoning for their `subfiles`: a curated category cuts the false
positives before the correlation ever runs.

WHY ONE JSON FILE PER COLLECTION, next to the library rather than in the config
directory. A collection points AT library content, so it belongs beside it and
travels when the library is copied. And it has to be readable and mailable: the
question after a phase map is "which phases did you even consider?", and that is
answered by a file you can open.

WHAT A COLLECTION IS NOT. It is not a run configuration. No reflector limits,
no per-method file choice, no colours — those belong to the run and to the EDS
presets. A collection answers one question: which phases belong together.

THE KEY IS A FILENAME STEM, and that is the weak point, stated plainly. There
is no content-based phase identifier in this app, so renaming a CIF breaks the
pointer. Each member therefore also records formula and space group, which are
enough to PROPOSE a repair. They are never used for display: the label shown is
always recomputed from the library, so this file cannot become a second, stale
truth.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Bumped only when the on-disk shape changes incompatibly. A file carrying any
#: other number is refused rather than guessed at — a collection read wrong
#: silently changes which phases a run considers.
COLLECTION_SCHEMA = 1

_DIR_OVERRIDE: Optional[Path] = None      # tests inject this


#: The folder name. Deliberately NOT an entry in
#: ``path_utils.DATABASE_SUBFOLDERS``, although that is where the other five
#: live. Measured: ``simulation/server_mode_manager.py:207`` indexes that dict
#: through a five-key ``_attr_map`` with no default, so a sixth key raises
#: ``KeyError`` inside ``_ensure_server_directories()``, whose caller catches
#: ``Exception``, logs "Failed to initialize server mode" and carries on with
#: the feature switched off. Anyone using server mode would have lost it
#: without a word. ``tests/test_path_utils.py:132-136`` also pins the key set
#: exactly. Collections hold no crystal data and are not part of the server
#: sync, so they have no business in that dict anyway.
_COLLECTIONS_DIRNAME = "Collections"


def collections_dir() -> Path:
    """Where the user's collections live: ``Database/Collections``.

    Beside the library, not in ``%APPDATA%`` where the EDS presets live: a
    preset is a recipe and belongs to the machine, a collection points at
    library content and belongs next to it.
    """
    if _DIR_OVERRIDE is not None:
        return _DIR_OVERRIDE
    from path_utils import get_local_database_path
    return get_local_database_path() / _COLLECTIONS_DIRNAME


def set_collections_dir_for_test(p: Optional[Path]) -> None:
    """Test-only: redirect the collections directory. ``None`` restores.

    Not a convenience. ``get_local_database_path()`` is hardwired to the
    project tree and ``tests/conftest.py`` fails any test that writes under
    ``Database/``, so without this hook the service cannot be tested at all
    without touching the user's phase library.
    """
    global _DIR_OVERRIDE
    _DIR_OVERRIDE = Path(p) if p is not None else None


# --- model ------------------------------------------------------------------

@dataclass
class PhaseMember:
    """One phase in a collection.

    ``key`` is the library stem and the only thing looked up. The other three
    are the fingerprint used to propose a repair after a rename (§10.2); they
    are never displayed.
    """
    key: str
    cif_filename: str = ""
    formula: str = ""
    space_group: str = ""

    def to_dict(self) -> dict:
        d = {"key": self.key}
        for f in ("cif_filename", "formula", "space_group"):
            if getattr(self, f):
                d[f] = getattr(self, f)
        return d

    @classmethod
    def from_dict(cls, payload: Any) -> "PhaseMember":
        if isinstance(payload, str):          # tolerated shorthand
            return cls(key=payload)
        if not isinstance(payload, dict):
            raise ValueError(f"member must be an object or a string, got {payload!r}")
        key = str(payload.get("key") or "").strip()
        if not key:
            raise ValueError("member has no key")
        return cls(
            key=key,
            cif_filename=str(payload.get("cif_filename") or ""),
            formula=str(payload.get("formula") or ""),
            space_group=str(payload.get("space_group") or ""),
        )


@dataclass
class PhaseCollection:
    name: str
    parent: Optional[str] = None
    exclusive: bool = True
    description: str = ""
    created: str = ""
    members: list[PhaseMember] = field(default_factory=list)
    schema: int = COLLECTION_SCHEMA

    def to_dict(self) -> dict:
        return {
            "schema": COLLECTION_SCHEMA,
            "name": self.name,
            "parent": self.parent,
            "exclusive": bool(self.exclusive),
            "description": self.description,
            "created": self.created,
            "members": [m.to_dict() for m in self.members],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "PhaseCollection":
        if not isinstance(payload, dict):
            raise ValueError("collection must be a JSON object")
        schema = payload.get("schema")
        if not isinstance(schema, int) or isinstance(schema, bool):
            raise ValueError(
                f"collection has no usable 'schema' field (got {schema!r}); "
                f"this build writes schema {COLLECTION_SCHEMA}"
            )
        if schema != COLLECTION_SCHEMA:
            raise ValueError(
                f"collection schema {schema} cannot be read by this build "
                f"(expected {COLLECTION_SCHEMA})"
            )
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("collection has no name")
        raw = payload.get("members") or []
        if not isinstance(raw, list):
            raise ValueError(f"collection {name!r}: 'members' must be a list")
        parent = payload.get("parent")
        return cls(
            name=name,
            parent=str(parent).strip() or None if parent else None,
            exclusive=bool(payload.get("exclusive", True)),
            description=str(payload.get("description") or ""),
            created=str(payload.get("created") or ""),
            members=[PhaseMember.from_dict(m) for m in raw],
        )


# --- files ------------------------------------------------------------------

_RESERVED_STEMS = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_LOCAL_STATE = "_local.json"


def safe_filename(name: str) -> str:
    """A filesystem-safe stem for a name the user typed.

    Phase keys in this library contain dots, parentheses and Greek letters
    (`Al4Fe1.7Si (tau11)`), and collection names are free text. The true name
    lives inside the JSON and is what every lookup matches on; this only
    decides where the bytes go.
    """
    stem = _UNSAFE.sub("_", str(name).strip())
    stem = re.sub(r"_+", "_", stem).strip("._-")
    if stem.lower() in _RESERVED_STEMS:
        stem = f"{stem}_collection"
    if not stem:
        stem = "collection"
    return stem[:80]


def _write_atomic(path: Path, text: str) -> None:
    """Temp file in the same directory, then replace.

    ``newline="\\n"`` is the point, not tidiness: without it Python translates
    every ``\\n`` to CRLF on Windows, and a mailed collection would differ from
    the saved one on every line with no content changed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".collection_", suffix=".json.tmp",
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


def _target_path(name: str) -> Path:
    """Where a collection of this name goes.

    Two different names can sanitise to the same stem (``Al/Si`` and
    ``Al:Si``), so a stem already held by a DIFFERENT name gets a short hash
    suffix rather than overwriting somebody else's filing.
    """
    d = collections_dir()
    stem = safe_filename(name)
    path = d / f"{stem}.json"
    if path.exists():
        try:
            existing = PhaseCollection.from_dict(
                json.loads(path.read_text(encoding="utf-8")))
            same = existing.name.casefold() == str(name).strip().casefold()
        except Exception:
            same = False          # unreadable: step aside rather than clobber
        if not same:
            suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
            return d / f"{stem}_{suffix}.json"
    return path


def _files() -> list[Path]:
    d = collections_dir()
    try:
        return sorted(p for p in d.glob("*.json") if p.name != _LOCAL_STATE)
    except OSError:
        return []


def load_all() -> tuple[list[PhaseCollection], list[dict]]:
    """Every readable collection, plus everything that is wrong with the set.

    Never raises. One bad file must not take the whole filing with it, and a
    problem is reported rather than repaired: the app does not know which of
    two collections was meant to hold a phase.
    """
    cols: list[PhaseCollection] = []
    problems: list[dict] = []
    for f in _files():
        try:
            cols.append(PhaseCollection.from_dict(
                json.loads(f.read_text(encoding="utf-8"))))
        except Exception as exc:
            problems.append({"kind": "bad_schema", "detail": f"{f.name}: {exc}"})

    names = {c.name for c in cols}
    for c in cols:
        if c.parent and c.parent not in names:
            problems.append({"kind": "orphan_parent", "detail": c.name,
                             "parent": c.parent})
            c.parent = None

    seen: dict[str, list[str]] = {}
    for c in cols:
        if not c.exclusive:
            continue
        for m in c.members:
            seen.setdefault(m.key, []).append(c.name)
    for key, owners in seen.items():
        if len(owners) > 1:
            problems.append({"kind": "duplicate_key", "key": key,
                             "collections": sorted(owners),
                             "detail": f"{key} is in {', '.join(sorted(owners))}"})
    return cols, problems


def _by_name() -> dict[str, PhaseCollection]:
    return {c.name: c for c in load_all()[0]}


def _path_of(name: str) -> Optional[Path]:
    key = str(name).strip().casefold()
    for f in _files():
        try:
            c = PhaseCollection.from_dict(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
        if c.name.casefold() == key:
            return f
    return None


def save(c: PhaseCollection) -> Path:
    """Write a collection. Enforces the one-level rule and exclusivity."""
    if not str(c.name).strip():
        raise ValueError("a collection needs a name")
    existing = _by_name()
    # `_path_of` matches casefold, so "matrix" and "Matrix" resolve to one
    # file. Without this guard, saving the second one writes over the first
    # and its members are gone with nothing said. A change of casing is a
    # rename and goes through rename(), which knows to move the file.
    for other in existing.values():
        if other.name != c.name and other.name.casefold() == c.name.casefold():
            raise ValueError(
                f"a collection named {other.name!r} already exists; "
                f"names that differ only in case would share one file"
            )
    if c.parent:
        if c.parent == c.name:
            raise ValueError(f"{c.name!r} cannot be its own parent")
        p = existing.get(c.parent)
        if p is None:
            raise ValueError(f"no collection named {c.parent!r}")
        if p.parent:
            raise ValueError(
                "collections nest one level only: "
                f"{c.parent!r} is itself inside {p.parent!r}"
            )
    if any(o.parent == c.name for o in existing.values()) and c.parent:
        raise ValueError(
            f"collections nest one level only: {c.name!r} already has children"
        )
    if c.exclusive:
        mine = {m.key for m in c.members}
        for other in existing.values():
            if other.name == c.name or not other.exclusive:
                continue
            clash = mine & {m.key for m in other.members}
            if clash:
                other.members = [m for m in other.members if m.key not in clash]
                _write_atomic(_path_of(other.name) or _target_path(other.name),
                              json.dumps(other.to_dict(), indent=2,
                                         ensure_ascii=False) + "\n")
    path = _path_of(c.name) or _target_path(c.name)
    _write_atomic(path, json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return path


def rename(old: str, new: str) -> None:
    """Rename a collection, and follow it everywhere it is referenced."""
    new = str(new).strip()
    if not new:
        raise ValueError("a collection needs a name")
    cols = _by_name()
    if old not in cols:
        raise KeyError(f"no collection named {old!r}")
    # Casefold, because the file lookup is casefold: an exact-match guard let
    # "b" -> "A" through while "a" existed, and the write then destroyed "a".
    # Renaming a collection to a different casing of ITSELF is legitimate and
    # is what the first clause allows.
    if new.casefold() != old.casefold() and any(
            n.casefold() == new.casefold() for n in cols):
        raise ValueError(f"a collection named {new!r} already exists")
    target = cols[old]
    src = _path_of(old)
    target.name = new
    _write_atomic(_target_path(new),
                  json.dumps(target.to_dict(), indent=2, ensure_ascii=False) + "\n")
    if src is not None and src != _target_path(new):
        try:
            src.unlink()
        except OSError as exc:
            logger.warning("phase_collections: could not remove %s: %s", src, exc)
    for c in cols.values():
        if c.parent == old:
            c.parent = new
            _write_atomic(_path_of(c.name) or _target_path(c.name),
                          json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n")
    st = load_state()
    changed = False
    if st.get("active") == old:
        st["active"] = new
        changed = True
    if old in (st.get("hidden") or []):
        st["hidden"] = [new if h == old else h for h in st["hidden"]]
        changed = True
    if changed:
        save_state(st)


def delete(name: str) -> bool:
    """Delete a collection. Children are promoted; NO crystal file is touched."""
    path = _path_of(name)
    if path is None:
        return False
    for c in _by_name().values():
        if c.parent == name:
            c.parent = None
            _write_atomic(_path_of(c.name) or _target_path(c.name),
                          json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n")
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("phase_collections: could not delete %s: %s", path, exc)
        return False
    st = load_state()
    changed = False
    if st.get("active") == name:
        st["active"] = None
        changed = True
    if name in (st.get("hidden") or []):
        st["hidden"] = [h for h in st["hidden"] if h != name]
        changed = True
    if changed:
        save_state(st)
    return True


def assign(keys: list[str], to_name: Optional[str],
           position: Optional[int] = None) -> None:
    """Move phases into a collection, or out of every one (``to_name=None``).

    Filing into an EXCLUSIVE collection (or unfiling everywhere,
    ``to_name=None``) strips the key from every OTHER exclusive collection --
    a phase has exactly one exclusive "home", and the shape of this whole
    feature is folders-plus-a-working-set: one home each, and one collection
    that deliberately cuts across them.

    Filing into a NON-exclusive collection (the working set) is additive
    ONLY. A star must not evict a phase from the folder it already lives in --
    that was a real bug here (not a wording issue): the strip loop below used
    to run unconditionally on every OTHER exclusive collection regardless of
    whether the TARGET was exclusive, so `assign(["Al"], "Arbeitsauswahl")`
    silently emptied "Matrix" of "Al" even though "Arbeitsauswahl" is
    `exclusive=False`. `tests/test_phase_collections.py::
    test_assign_into_working_set_leaves_exclusive_home_intact` pins the fix.
    """
    wanted = [str(k) for k in keys if str(k)]
    if not wanted:
        return
    cols = _by_name()
    if to_name is not None and to_name not in cols:
        raise KeyError(f"no collection named {to_name!r}")
    target = cols.get(to_name) if to_name else None

    # A metadata donor for each wanted key, wherever it already lives --
    # read-only, and independent of whether that collection gets stripped
    # below, so filing into the working set still carries formula/space_group
    # over from an existing membership instead of inventing a bare member.
    carried: dict[str, PhaseMember] = {}
    for c in cols.values():
        for m in c.members:
            if m.key in wanted:
                carried.setdefault(m.key, m)

    # See the docstring: only strip when the target is exclusive (or there
    # is no target at all -- "unfile everywhere"). A non-exclusive target
    # must leave every other collection's membership untouched.
    if target is None or target.exclusive:
        for c in cols.values():
            if target is not None and c.name == target.name:
                continue
            if not c.exclusive:
                continue
            keep = [m for m in c.members if m.key not in wanted]
            if len(keep) != len(c.members):
                c.members = keep
                _write_atomic(_path_of(c.name) or _target_path(c.name),
                              json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n")
    if target is None:
        return
    have = {m.key for m in target.members}
    added = [carried.get(k) or PhaseMember(key=k) for k in wanted if k not in have]
    if position is None or position >= len(target.members):
        target.members.extend(added)
    else:
        i = max(0, int(position))
        target.members[i:i] = added
    save(target)


def unassign_from(name: str, keys: list[str]) -> None:
    """Take phases out of ONE collection (spec 6.1).

    Distinct from ``assign(keys, None)``, which un-files them everywhere. The
    route needs this one: a user removing a phase from A must not also lose it
    from B, and with the working set in play a key legitimately sits in two
    places at once.
    """
    cols = _by_name()
    if name not in cols:
        raise KeyError(f"no collection named {name!r}")
    drop = {str(k) for k in keys}
    c = cols[name]
    c.members = [m for m in c.members if m.key not in drop]
    _write_atomic(_path_of(c.name) or _target_path(c.name),
                  json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n")


def effective_members(name: str,
                      cols: Optional[dict] = None) -> list[PhaseMember]:
    """Own members in order, then each child's members in child order (§4.2).

    ``cols`` lets a caller that already loaded every collection pass the map
    in; otherwise this reloads all of them, once per call.
    """
    if cols is None:
        cols = _by_name()
    if name not in cols:
        raise KeyError(f"no collection named {name!r}")
    out = list(cols[name].members)
    for child in sorted((c for c in cols.values() if c.parent == name),
                        key=lambda c: c.name.lower()):
        out.extend(child.members)
    return out


# --- per-installation state, deliberately NOT in a collection file ----------

_DEFAULT_STATE = {"schema": COLLECTION_SCHEMA, "active": None, "hidden": []}


def load_state() -> dict:
    """Which collection is active and which are hidden.

    Separate from the collection files on purpose: mail a collection and it
    must not arrive hidden at the other end, and "active" is a statement about
    this sitting, not about the content. Missing or broken: rebuilt from
    defaults, because losing it costs nothing.
    """
    p = collections_dir() / _LOCAL_STATE
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("not an object")
    except Exception:
        return dict(_DEFAULT_STATE)
    return {
        "schema": COLLECTION_SCHEMA,
        "active": d.get("active") or None,
        "hidden": [str(h) for h in (d.get("hidden") or []) if str(h)],
    }


def save_state(state: dict) -> None:
    payload = {
        "schema": COLLECTION_SCHEMA,
        "active": state.get("active") or None,
        "hidden": [str(h) for h in (state.get("hidden") or []) if str(h)],
    }
    _write_atomic(collections_dir() / _LOCAL_STATE,
                  json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


# --- what the library says about a member ---

def _index() -> dict:
    """The unified CIF+XTAL+SHT index, or an empty one if it cannot be built.

    Imported inside the function: the library scan parses every CIF with
    pymatgen, and a module-level import would pay that at import time for
    callers that only want to read a JSON file.
    """
    try:
        from backend.api.services.crystal_hint_local_library import get_index
        return get_index()
    except Exception:
        logger.warning("phase_collections: library index unavailable",
                       exc_info=True)
        return {}


def annotate(members: list[PhaseMember], idx: Optional[dict] = None) -> list[dict]:
    """Per member: what the library currently has for it.

    A member the index does not know is returned with ``present: False`` and
    every capability false. It is NEVER dropped — the caller has to be able to
    tell "this collection is empty" from "this collection's library is gone".

    ``idx`` lets a caller that already holds the index pass it in. Without it,
    listing 30 collections rebuilt the index 34 times; measured at roughly
    0.34 s per request, because ``get_index()`` re-walks CIF+XTAL+SHT and
    stats every file before it consults its own cache.
    """
    if idx is None:
        idx = _index()
    out = []
    for m in members:
        e = idx.get(m.key)
        label = ""
        if e is not None:
            label = getattr(e, "display_label", "") or getattr(e, "formula", "")
        out.append({
            "key": m.key,
            "label": label or m.key,
            "formula": getattr(e, "formula", "") if e is not None else m.formula,
            "space_group": (getattr(e, "space_group", "") if e is not None
                            else m.space_group),
            "present": e is not None,
            "has_cif": bool(getattr(e, "cif_path", None)) if e is not None else False,
            "has_xtal": bool(getattr(e, "xtal_path", None)) if e is not None else False,
            "has_sht": bool(getattr(e, "sht_path", None)) if e is not None else False,
        })
    return out


def unassigned_keys(idx: Optional[dict] = None,
                    cols: Optional[list] = None) -> list[str]:
    """Library phases that are in no exclusive collection, in index order."""
    if idx is None:
        idx = _index()
    if cols is None:
        cols = load_all()[0]
    filed = set()
    for c in cols:
        if c.exclusive:
            filed.update(m.key for m in c.members)
    return [k for k in idx if k not in filed]


def propose_repairs(idx: Optional[dict] = None,
                    cols: Optional[list] = None) -> list[dict]:
    """For each member the library has lost, library phases that could be it.

    Matched on formula AND space group, and only against phases that are in no
    collection. PROPOSED, never applied: `Al2Cu_mp-985806_conventional_standard`
    and its `.P1-backup` twin agree on both, and only the user knows which was
    meant.
    """
    if idx is None:
        idx = _index()
    if cols is None:
        cols = load_all()[0]
    free = list(unassigned_keys(idx, cols))
    out = []
    for c in cols:
        for m in c.members:
            if m.key in idx or not (m.formula and m.space_group):
                continue
            cands = [
                k for k in free
                if (getattr(idx[k], "formula", "") or "") == m.formula
                and (getattr(idx[k], "space_group", "") or "") == m.space_group
            ]
            if cands:
                out.append({"collection": c.name, "missing_key": m.key,
                            "candidates": sorted(cands)})
    return out


# --- a starting point, made from the library that is actually present -------

#: Transition metals, for "an Al phase with a transition metal in it".
_TRANSITION_METALS = frozenset(
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Ru Rh Pd Ag Cd Hf Ta W Re Os Ir "
    "Pt Au".split()
)

#: Evaluated in order; the first rule that matches takes the phase. English
#: names: they are seeds the user renames, and a German default in an English
#: UI would be worse than a plain one.
_SUGGESTION_RULES: list[tuple[str, Any]] = [
    ("Matrix and pure metals", lambda els: len(els) == 1),
    ("Intermetallics in Al",
     lambda els: "Al" in els and bool(set(els) & _TRANSITION_METALS)),
    ("Mg phases", lambda els: "Mg" in els),
    ("Zn phases", lambda els: "Zn" in els),
    ("Carbides and nitrides",
     lambda els: bool({"C", "N"} & set(els)) and bool(set(els) & _TRANSITION_METALS)),
]


def suggest() -> list[dict]:
    """Collections proposed from the phases on this machine. Writes nothing.

    Not shipped as files: ``Database/`` ships empty, so a delivered collection
    would point at phases that do not exist — the same class of mistake as a
    default filter that hides the entry you need. A phase whose formula yields
    no elements is left out rather than filed under a guess; it shows up as
    unassigned, which is the honest answer.
    """
    try:
        from phase_metadata import extract_elements
    except Exception:
        logger.warning("phase_collections: phase_metadata unavailable",
                       exc_info=True)
        return []
    already = set()
    for c in load_all()[0]:
        if c.exclusive:
            already.update(m.key for m in c.members)
    buckets: dict[str, list[str]] = {name: [] for name, _ in _SUGGESTION_RULES}
    for key, entry in _index().items():
        if key in already:
            continue
        els = list(getattr(entry, "elements", ()) or
                   extract_elements(getattr(entry, "formula", "") or ""))
        if not els:
            continue
        for name, rule in _SUGGESTION_RULES:
            if rule(els):
                buckets[name].append(key)
                break
    return [{"name": n, "keys": sorted(k)} for n, k in buckets.items() if k]
