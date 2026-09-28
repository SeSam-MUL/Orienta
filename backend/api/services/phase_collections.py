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

import contextlib
import functools
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Bumped when the on-disk shape changes incompatibly. Schema 2 (2026-09-27)
#: added a stable `id`, an author and a timestamp, and made membership a TAG
#: rather than a folder.
#:
#: OLDER FILES ARE READ, NEWER ONES ARE NOT. A schema-1 file loads and gains an
#: `id` from its filename — refusing it would have made every collection on disk
#: unreadable the moment this number changed, including the two live ones. A file
#: from a NEWER build is still refused: a collection read wrong silently changes
#: which phases a run considers, and guessing at a shape nobody here knows is how
#: that happens.
COLLECTION_SCHEMA = 2

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
    """A named set of phases. Membership is a TAG, not a folder (spec §2.5).

    `id` is the identity and `name` is a label. They were one thing until schema
    2, and that is what broke on two machines: the file is named after the name,
    so a rename on one machine and a rename on the other produced two files for
    one group with no way to tell they were the same.

    `exclusive` is still read and still written, and is IGNORED. Not deleted,
    because a collection file that arrives from a schema-1 machine must still
    load -- and two such files are on disk right now. Nothing may branch on it
    again: with tags, a phase has no single home, and the code that assumed one
    made "not yet filed" mean the whole library.
    """

    name: str
    id: str = ""
    parent: Optional[str] = None
    exclusive: bool = True          # read, written, never acted on
    description: str = ""
    created: str = ""
    author: str = ""
    updated: str = ""
    members: list[PhaseMember] = field(default_factory=list)
    schema: int = COLLECTION_SCHEMA

    #: The file this was READ from, when it was read from one. Never serialised
    #: (`to_dict` omits it) and never part of the format.
    #:
    #: It exists because deriving the write path back from the NAME is how a
    #: write lands on the wrong file. `_path_of` returns the first casefold match
    #: in the folder, so with two files carrying one name -- a hand copy, a merged
    #: share, the very workflow §2.5.3 describes -- every writer overwrote
    #: whichever came first, not the one it had loaded. Measured on the version
    #: before schema 2 and on this one, identically: `assign(['NEW'], 'Matrix')`
    #: rewrote `twinA.json` with twinB's members, and twinA's own member was gone.
    #: Knowing where a collection came from is the only thing that cannot alias.
    source_path: Optional[Path] = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict:
        return {
            "schema": COLLECTION_SCHEMA,
            "id": self.id,
            "name": self.name,
            "parent": self.parent,
            # WITHDRAWN 2026-09-27: this used to say "kept so a schema-2 file
            # stays readable by a schema-1 build". That is FALSE, and measured
            # against the pre-change reader: it raises on `schema != 1` before it
            # ever looks at this field. Writing schema 2 is a ONE-WAY DOOR PER
            # FILE: the first edit after an update makes THAT collection
            # unreadable to any earlier Orienta.
            #
            # NARROWED 2026-09-27 after b9 and 03 checked my wording against the
            # old reader's code. I had written "shows an empty library", measured
            # on a folder with ONE file. `load_all` catches per file and carries
            # on, so an older build keeps every untouched group and reports only
            # the edited one. Measured with three groups, one edited: the old
            # build lists the other two and reports `bad_schema` for the third.
            # "Empty" holds only if every group was edited, or there was one.
            # Worth a release-note line in that narrower form; it is not what this
            # field is for.
            #
            # The real reason to keep it: files written BEFORE schema 2 carry
            # `exclusive: false` -- `livetest-2026-09-27-workingset.json` in this
            # tree does -- `from_dict` preserves it, and
            # `databaseGrouping.js:134,140` still branches on it. Dropping the
            # field would silently change how the Database browser groups a phase
            # that sits in two collections.
            "exclusive": bool(self.exclusive),
            "description": self.description,
            "created": self.created,
            "author": self.author,
            "updated": self.updated,
            "members": [m.to_dict() for m in self.members],
        }

    @classmethod
    def from_dict(cls, payload: Any, *, fallback_id: str = "") -> "PhaseCollection":
        """Read schema 1 or 2. A schema-1 file gains an `id` on first read.

        The old check refused any number but its own, so bumping the schema
        without this would have made every collection on disk unreadable --
        including the two live ones. Reading an older file is not a favour, it is
        the difference between a migration and a data loss.
        """
        if not isinstance(payload, dict):
            raise ValueError("collection must be a JSON object")
        schema = payload.get("schema")
        if not isinstance(schema, int) or isinstance(schema, bool):
            raise ValueError(
                f"collection has no usable 'schema' field (got {schema!r}); "
                f"this build writes schema {COLLECTION_SCHEMA}"
            )
        if schema > COLLECTION_SCHEMA:
            raise ValueError(
                f"collection schema {schema} was written by a newer build than "
                f"this one (which reads up to {COLLECTION_SCHEMA}); it is not "
                f"read rather than read wrongly"
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
            # Schema 1 had none. The FILE STEM is used, because it is the only
            # stable thing such a file carries -- and it is what every existing
            # reference (`active`, `hidden`, `parent`) was keyed on by way of the
            # name, so ids minted this way keep those references working.
            id=str(payload.get("id") or "").strip() or fallback_id,
            parent=str(parent).strip() or None if parent else None,
            exclusive=bool(payload.get("exclusive", True)),
            description=str(payload.get("description") or ""),
            created=str(payload.get("created") or ""),
            author=str(payload.get("author") or ""),
            updated=str(payload.get("updated") or ""),
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
                json.loads(path.read_text(encoding="utf-8")), fallback_id=stem)
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
            c = PhaseCollection.from_dict(
                json.loads(f.read_text(encoding="utf-8")), fallback_id=f.stem)
        except Exception as exc:
            problems.append({"kind": "bad_schema", "detail": f"{f.name}: {exc}"})
            continue
        c.source_path = f          # so no writer has to guess it back from a name
        cols.append(c)

    # TWO FILES, ONE NAME is now REPORTED instead of silently resolved. It was
    # silent in both directions before: `_by_name()` collapses them, so one
    # collection simply vanished from the app, and every writer overwrote
    # whichever file `_path_of` matched first. Measured before and after schema 2,
    # identically: `assign(['NEW'], 'Matrix')` rewrote twinA's file with twinB's
    # members. The app cannot know which was meant, so it says so -- the same
    # reason `bad_schema` is reported rather than repaired.
    #
    # Reachable only through a folder that was copied, merged or shared, which is
    # exactly the workflow identities exist for (§2.5.3).
    seen: dict[str, list[str]] = {}
    for c in cols:
        seen.setdefault(c.name.casefold(), []).append(
            c.source_path.name if c.source_path else "?")
    for folded, files in seen.items():
        if len(files) > 1:
            problems.append({"kind": "duplicate_name",
                             "detail": next(c.name for c in cols
                                            if c.name.casefold() == folded),
                             "files": sorted(files)})

    # Parents are normalised to IDENTITIES here, once, so that everything
    # downstream (children, effective members, the cycle guard, delete) compares
    # ids and never names. A schema-1 file names its parent; it is resolved by
    # the name fallback and rewritten in memory, so the next save of that file
    # stores the id without anyone having to run a migration step.
    for c in cols:
        if not c.parent:
            continue
        parent = resolve_ref(c.parent, cols)
        if parent is None:
            problems.append({"kind": "orphan_parent", "detail": c.name,
                             "parent": c.parent})
            c.parent = None
        elif parent is c:
            # A file edited by hand can say this.
            #
            # CORRECTED 2026-09-27: the original comment here claimed "infinite
            # recursion in `effective_members`". That is FALSE, and measured to be
            # false -- `children_of` filters `c is not parent`, and
            # `effective_members` does not recurse into grandchildren at all, so a
            # self-parent yields `children_of == []` and the members come back
            # normally. The guard is still right, for the reason below.
            #
            # What actually happens without it: the collection DISAPPEARS. The
            # manager renders `topLevel = collections.filter(c => !c.parent)` and
            # reaches children only underneath a top-level row, so a collection
            # that is its own parent is excluded from the top level and nested
            # under a row that is never drawn.
            problems.append({"kind": "self_parent", "detail": c.name,
                             "parent": c.parent})
            c.parent = None
        else:
            c.parent = parent.id or parent.name

    # NO `duplicate_key` PROBLEM ANY MORE. It reported a phase that was in two
    # exclusive collections, which was a contradiction when a phase had one home.
    # With tags (§2.5) it is the normal case -- `Al2Cu` belongs in "Al systems"
    # and in "Cu systems" -- so reporting it would put a permanent warning on
    # correct filing. The old code could not even reach the check: it skipped
    # every non-exclusive collection first, and with tags there are none.
    return cols, problems


def _by_name() -> dict[str, PhaseCollection]:
    return {c.name: c for c in load_all()[0]}


#: One lock for the whole folder, not one per file. Reentrant because the
#: read-modify-write helpers call `save()` inside their own critical section.
_WRITE_LOCK = threading.RLock()


@contextlib.contextmanager
def _write_lock():
    """Serialise read-modify-write on the collections folder.

    WHY A LOCK AT ALL. `_write_atomic` makes each individual write atomic; it does
    not make read-modify-write atomic, and every mutating function here is
    read-modify-write. FastAPI runs a sync `def` endpoint in a threadpool, so two
    requests genuinely overlap in one process. Found in f7's user loop: an
    Alt-drag over twelve phases fired twelve parallel `POST /members`, they all
    read the same "before", and the last write won -- **twelve on screen, nine in
    the folder, every response 200**. Reproduced here; on Windows the racing
    `os.replace` additionally raises `WinError 5`, so one missing lock shows up as
    a lost update on one platform and an error on another.

    WHY FOLDER-WIDE AND NOT PER FILE. `assign` reads EVERY collection (`_by_name`)
    and may write several of them, so a per-file lock would not cover its own
    critical section. Measurably harmless: the folder holds a handful of small
    JSON files and only WRITES take the lock.

    WHAT IT DOES NOT DO. It is process-wide. Two Orienta processes, or the app
    plus a script, still race -- that needs a lock file, and `Database/` may sit on
    a network share where those behave differently. Out of scope here, and named
    rather than implied.

    Reads deliberately do NOT take it: holding it across `load_all` would put
    every listing behind every write for no benefit, since a read of a
    half-updated folder still sees whole files.
    """
    with _WRITE_LOCK:
        yield


def _serialised(fn):
    """Take the folder write lock for the whole call.

    A decorator rather than a `with` inside each body: six functions mutate the
    folder, and one that forgets the block is exactly the silent lost update this
    exists to stop. `_guarded_writers` below reads this list back off the module
    and fails if a writer loses its decorator.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _WRITE_LOCK:
            return fn(*args, **kwargs)
    wrapper._serialised = True          # noqa: SLF001 - read by the guard test
    return wrapper


def _now_iso() -> str:
    """UTC, to the second. Same shape `eds_presets._now_iso` writes."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_path_of(c: PhaseCollection) -> Path:
    """Where to write this collection, without asking its NAME.

    The file it was read from, when we know it. Only a collection built from
    scratch falls back to the name-derived path, and that one has no file to
    alias with.
    """
    if c.source_path is not None:
        return c.source_path
    return _path_of(c.name) or _target_path(c.name)


def resolve_ref(ref: Any, cols) -> Optional[PhaseCollection]:
    """Find a collection by IDENTITY, falling back to its NAME for one version.

    Every cross-reference -- `parent`, and `active`/`hidden` in the state file --
    used to be a name, which is why `rename()` had to walk the whole folder and
    rewrite them. A name is what the user edits; it is the worst possible key.

    The fallback is not politeness, it is the only thing between an update and a
    quiet loss. Measured on a real folder written before this change: `_local.json`
    held `active: "_my-group"` while that file's stem -- and so its minted id --
    was `my-group`, without the leading underscore. An id-only lookup would drop
    the active collection on first start after the update, and nothing would say
    so.

    NAMES WIN OVER IDS. I had this the other way round and wrote "the identity is
    the one that cannot have been typed by accident" -- which is true and is not
    the point. f7's closing review found the consequence: a group literally NAMED
    `Al_systems`, next to a group "Al systems" whose minted id is `Al_systems`,
    was resolved to the second. `delete("Al_systems")` then deleted the group the
    user did NOT mean, members and all, and nothing said so.

    A name is what the user typed and what the UI shows; an id is an internal
    handle. When the two collide, the typed thing is the one a human meant. The
    collision SOURCE is closed separately: `_mint_id` may no longer mint an id
    that already exists as a name (see `_taken_refs`), so from this code's side
    the two spaces cannot overlap at all -- only a user naming a group exactly
    like an existing id can produce it, and then this order is the sane reading.

    Names are matched casefold, ids exactly: `save()` already refuses two names
    differing only in case, so the casefold match cannot be ambiguous, while an
    id is machine-made and never needs folding.
    """
    ref = str(ref or "").strip()
    if not ref:
        return None
    cols = list(cols)
    folded = ref.casefold()
    for c in cols:
        if c.name.casefold() == folded:
            return c
    for c in cols:
        if c.id and c.id == ref:
            return c
    return None


def _taken_refs() -> set[str]:
    """Every string a reference could already resolve to: ids AND names.

    A minted id must collide with NEITHER. Ids alone was not enough, and f7's
    closing review found the hole: minting `Al_systems` for "Al systems" while a
    group was literally NAMED `Al_systems` put one string in front of two
    collections, and `delete("Al_systems")` removed the wrong one with its
    members. Closing it here means the two spaces cannot overlap from this code's
    side at all; `resolve_ref` decides the remaining case, where a user names a
    group exactly like an existing id.

    Names are folded because `resolve_ref` matches them folded: a minted id that
    differs from a name only in case would still be ambiguous to the lookup.
    """
    out: set[str] = set()
    for f in _files():
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc = doc or {}
        got = str(doc.get("id") or "").strip() or f.stem
        if got:
            out.add(got)
        name = str(doc.get("name") or "").strip()
        if name:
            out.add(name)
            out.add(name.casefold())
    return out


def _all_ids() -> set[str]:
    """Every id on disk, read file by file.

    Deliberately not `{c.id for c in load_all()[0]}` via `_by_name()`: that map
    is keyed by name and drops one of two files sharing one, hiding its id from
    a uniqueness check.
    """
    out: set[str] = set()
    for f in _files():
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        got = str((doc or {}).get("id") or "").strip() or f.stem
        if got:
            out.add(got)
    return out


def _mint_id(name: str, taken) -> str:
    """A readable, stable id for a collection that has none yet.

    Readable on purpose rather than a uuid: this folder is copied and mailed by
    hand (§2.5.3 -- nothing syncs yet), and a human opening `Al systems.json`
    should be able to see what `parent` points at. It is minted ONCE and then
    travels inside the payload, so it survives every later rename -- which is
    the entire point.
    """
    base = safe_filename(str(name).strip()) or "collection"
    taken = set(taken)

    def _free(candidate: str) -> bool:
        # Folded as well as literal: `resolve_ref` matches names folded, so an id
        # that differs from an existing name only in case is still ambiguous to
        # the lookup and is not free.
        return candidate not in taken and candidate.casefold() not in taken

    if _free(base):
        return base
    n = 2
    while not _free(f"{base}-{n}"):
        n += 1
    return f"{base}-{n}"


def _path_of(name: str) -> Optional[Path]:
    key = str(name).strip().casefold()
    for f in _files():
        try:
            c = PhaseCollection.from_dict(
                json.loads(f.read_text(encoding="utf-8")), fallback_id=f.stem)
        except Exception:
            continue
        if c.name.casefold() == key:
            return f
    return None


@_serialised
def save(c: PhaseCollection) -> Path:
    """Write a collection. Enforces the one-level rule.

    Stamps `created` once and `updated` on every write. §2.5.3 says this folder
    is copied and mailed between machines, so "who and when" has to be IN the
    file -- and until now no code path filled either field, which made a claim in
    schema 2's own commit message ("they already answer 'who was that' when the
    folder is copied by hand") simply false. `author` is only ever set by a caller
    that knows one; it is never invented here.
    """
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
    # An identity is minted once, on first save, and then travels in the payload.
    # Until this point a new collection has none, so everything that references
    # it -- and the duplicate check below -- would be keyed on a name the user
    # can change a minute later.
    if not c.id:
        # Against EVERY id on disk, not against `existing` -- that is keyed by
        # name, so two files sharing a name collapse and one of their ids becomes
        # invisible to the check. Measured: with `twinA` holding id `Matrix`,
        # creating a collection named "Matrix" minted `Matrix` a second time, and
        # `resolve_ref("Matrix")` then answered the NEW EMPTY collection. That is
        # the part schema 2 added to the pre-existing duplicate-name mess: a
        # `parent`, `active` or `hidden` reference resolving to the wrong group.
        c.id = _mint_id(c.name, _taken_refs())
    if c.parent:
        p = resolve_ref(c.parent, existing.values())
        if p is None:
            raise ValueError(f"no collection named {c.parent!r}")
        if p.id == c.id or p.name == c.name:
            raise ValueError(f"{c.name!r} cannot be its own parent")
        if p.parent:
            raise ValueError(
                "collections nest one level only: "
                f"{p.name!r} is itself inside {p.parent!r}"
            )
        # Stored as the identity, so renaming the parent leaves this untouched.
        c.parent = p.id or p.name
    if c.parent and children_of(c, existing.values()):
        raise ValueError(
            f"collections nest one level only: {c.name!r} already has children"
        )
    # SAVING A GROUP NO LONGER TOUCHES ANY OTHER GROUP. Until schema 2 this
    # stripped every member it shared with another "exclusive" collection, and
    # wrote those files as a side effect of saving this one. Under tags that is
    # simply wrong -- `Al2Cu` belongs in "Al systems" AND in "Cu systems" -- and
    # it was destructive in the quietest possible way: a save removed phases from
    # a group the user was not looking at. Removing from another group is now only
    # ever the explicit `assign(..., move=True)` (§2.5: dragging adds; Alt moves).
    # Stamped here rather than at each of the five callers, so no write can
    # forget. `created` only if it is empty -- a collection is created once, and
    # re-stamping it on every member change would erase the one date that says
    # when the group came into being.
    now = _now_iso()
    if not c.created:
        c.created = now
    c.updated = now
    # "unknown" rather than "", the same convention the synonym store uses
    # (`phase_synonyms.set_names`), so both stores in `Database/` read the same
    # way. This app has no user identity, so an author is only ever what a caller
    # states -- and a write by somebody who did not say who they were is a fact
    # worth recording, not an empty field to guess at later.
    if not str(c.author).strip():
        c.author = "unknown"
    path = _write_path_of(c)
    _write_atomic(path, json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return path


@_serialised
def rename(old: str, new: str) -> None:
    """Rename a collection, and follow it everywhere it is referenced."""
    new = str(new).strip()
    if not new:
        raise ValueError("a collection needs a name")
    cols = _by_name()
    target = resolve_ref(old, cols.values())
    if target is None:
        raise KeyError(f"no collection named {old!r}")
    # The collection's REAL current name. `old` may have been an identity, and
    # everything below compares against the old name: the duplicate-name guard,
    # the file to remove, and the state reference to migrate.
    old = target.name
    # Casefold, because the file lookup is casefold: an exact-match guard let
    # "b" -> "A" through while "a" existed, and the write then destroyed "a".
    # Renaming a collection to a different casing of ITSELF is legitimate and
    # is what the first clause allows.
    if new.casefold() != old.casefold() and any(
            n.casefold() == new.casefold() for n in cols):
        raise ValueError(f"a collection named {new!r} already exists")
    src = _path_of(old)
    target.name = new
    _write_atomic(_target_path(new),
                  json.dumps(target.to_dict(), indent=2, ensure_ascii=False) + "\n")
    if src is not None and src != _target_path(new):
        try:
            src.unlink()
        except OSError as exc:
            logger.warning("phase_collections: could not remove %s: %s", src, exc)
    # NOTHING ELSE IS REWRITTEN ANY MORE. A rename used to walk every other
    # collection file and the state file to repair references to the old name --
    # the tax for using a name as a key. Children point at `target.id`, which a
    # rename does not touch, so their files are already correct.
    #
    # The state file is the one place a stale NAME can still be on disk (written
    # by a build before schema 2). It is not repaired but MIGRATED: if the
    # reference resolves to the collection being renamed, it is rewritten as the
    # identity, so this is the last rename that has to think about it.
    ref = target.id or new
    # Matched against the OLD name on purpose: `target.name` is already `new` by
    # here, so resolving through the collection would find nothing and the
    # active collection would go quietly blank -- the exact failure this whole
    # change exists to prevent.
    def _points_here(r: Any) -> bool:
        r = str(r or "").strip()
        return bool(r) and (r == target.id or r.casefold() == old.casefold())

    st = load_state()
    changed = False
    if _points_here(st.get("active")) and st.get("active") != ref:
        st["active"] = ref
        changed = True
    hidden = [ref if _points_here(h) else h for h in (st.get("hidden") or [])]
    if hidden != (st.get("hidden") or []):
        st["hidden"] = hidden
        changed = True
    if changed:
        save_state(st)


@_serialised
def delete(ref: str) -> bool:
    """Delete a collection by IDENTITY or name. Children promoted; no crystal
    file is touched.

    Resolution comes FIRST, and that order is the fix: this used to start with
    `_path_of(name)`, which matches on the name only, so an id returned None and
    the function answered `False`. `False` means "there is no such collection",
    so a caller holding a perfectly good identity was told its collection did
    not exist -- a silent failure dressed as a normal answer, which is worse
    than the KeyError the other three raised.
    """
    cols = list(_by_name().values())
    target = resolve_ref(ref, cols)
    if target is None:
        return False
    # By the resolved collection's own name: it came out of a file, and two
    # names differing only in case cannot both exist (`save` refuses them).
    path = target.source_path or _path_of(target.name)
    if path is None:
        return False
    for c in (children_of(target, cols) if target is not None else []):
        c.parent = None
        _write_atomic(_write_path_of(c),
                      json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n")
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("phase_collections: could not delete %s: %s", path, exc)
        return False
    # By identity OR by name: the state file may still hold either (§2.5.1's one
    # version of fallback). A reference left pointing at a deleted collection is
    # not inert -- `active` would name something that is gone, and `hidden`
    # would silently hide the next collection that happened to reuse the name.
    gone = {target.id, target.name}
    gone = {str(g).strip().casefold() for g in gone if str(g).strip()}

    def _points_here(r: Any) -> bool:
        return str(r or "").strip().casefold() in gone

    st = load_state()
    changed = False
    if _points_here(st.get("active")):
        st["active"] = None
        changed = True
    hidden = [h for h in (st.get("hidden") or []) if not _points_here(h)]
    if hidden != (st.get("hidden") or []):
        st["hidden"] = hidden
        changed = True
    if changed:
        save_state(st)
    return True


@_serialised
def assign(keys: list[str], to_name: Optional[str],
           position: Optional[int] = None, *, move: bool = False) -> None:
    """Add phases to a collection, or remove them from every one (``to_name=None``).

    ADDING IS THE DEFAULT (spec §2.5): membership is a tag, so putting `Al2Cu`
    in "Cu systems" leaves it in "Al systems". `move=True` is the named action
    "move here (remove from others)" -- the ⋯ menu entry, and Alt-drag.

    Until schema 2 this behaviour was derived from the target's `exclusive` flag,
    which meant the caller could not ask for one or the other: a drag into a
    "folder" always evicted, and only the single non-exclusive "working set" was
    additive. Under tags every group is additive unless the user says otherwise,
    so the decision moved from the DATA to the CALLER, where it belongs.

    (The older bug behind that flag is still worth knowing: the strip loop once
    ran regardless of the target, so `assign(["Al"], "Arbeitsauswahl")` silently
    emptied "Matrix" of "Al".
    `tests/test_phase_collections.py::test_assign_into_working_set_leaves_exclusive_home_intact`
    pinned the fix and still passes -- with `move` defaulting to False, an
    additive assign cannot touch another group at all.)
    """
    wanted = [str(k) for k in keys if str(k)]
    if not wanted:
        return
    cols = _by_name()
    # An IDENTITY or a name (§2.5.1). The frontend has both in hand, and an id
    # is the one that survives a rename between the click and the request.
    target = resolve_ref(to_name, cols.values()) if to_name else None
    if to_name is not None and target is None:
        raise KeyError(f"no collection named {to_name!r}")

    # A metadata donor for each wanted key, wherever it already lives --
    # read-only, and independent of whether that collection gets stripped
    # below, so filing into the working set still carries formula/space_group
    # over from an existing membership instead of inventing a bare member.
    carried: dict[str, PhaseMember] = {}
    for c in cols.values():
        for m in c.members:
            if m.key in wanted:
                carried.setdefault(m.key, m)

    # Only when the caller ASKED to move, or when there is no target at all
    # ("remove from every group"). Never because of a flag on the data.
    def _strip_the_others() -> None:
        for c in cols.values():
            if target is not None and c.name == target.name:
                continue
            keep = [m for m in c.members if m.key not in wanted]
            if len(keep) != len(c.members):
                c.members = keep
                _write_atomic(_write_path_of(c),
                              json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n")

    if target is None:
        _strip_the_others()          # `to_name=None` means: un-file everywhere
        return

    have = {m.key for m in target.members}
    added = [carried.get(k) or PhaseMember(key=k) for k in wanted if k not in have]
    if position is None or position >= len(target.members):
        target.members.extend(added)
    else:
        i = max(0, int(position))
        target.members[i:i] = added

    # THE TARGET IS WRITTEN FIRST, and that order is the point. `save()` is the
    # step that can refuse -- its casefold guard raises when two names would share
    # one file -- and the strip loop used to run before it. Measured: a
    # `move=True` into a casefold twin stripped the phase from its old group,
    # THEN raised, so the caller got an error stating nothing had happened while
    # the phase was already gone. A move that fails must move nothing.
    save(target)
    if move:
        _strip_the_others()


@_serialised
def unassign_from(name: str, keys: list[str]) -> None:
    """Take phases out of ONE collection (spec 6.1).

    Distinct from ``assign(keys, None)``, which un-files them everywhere. The
    route needs this one: a user removing a phase from A must not also lose it
    from B, and with the working set in play a key legitimately sits in two
    places at once.
    """
    cols = _by_name()
    c = resolve_ref(name, cols.values())
    if c is None:
        raise KeyError(f"no collection named {name!r}")
    drop = {str(k) for k in keys}
    c.members = [m for m in c.members if m.key not in drop]
    _write_atomic(_write_path_of(c),
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
    parent = cols[name]
    out = list(parent.members)
    for child in sorted(children_of(parent, cols.values()),
                        key=lambda c: c.name.lower()):
        out.extend(child.members)
    return out


def children_of(parent: PhaseCollection, cols) -> list[PhaseCollection]:
    """The collections filed under `parent`, matched on its IDENTITY.

    One function, because this comparison was written out in four places and
    each was a `c.parent == name`; a rename used to have to repair all four by
    rewriting the files.
    """
    ref = parent.id or parent.name
    return [c for c in cols if c is not parent and c.parent == ref]


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


def resolve_state(state: dict, cols) -> dict:
    """The same state with `active` and `hidden` as IDENTITIES.

    `load_state` stays deliberately dumb -- it does not read the collections, so
    it cannot fail because of them, and losing the state costs nothing. This is
    where a reference written by an older build (a name) becomes the id every
    caller compares against, including `c.parent`, which is always an id.

    A reference that resolves to nothing is KEPT, not dropped: a collection file
    that is temporarily missing (a half-finished copy of the folder, a share not
    mounted yet) must not cost the user their active collection permanently.
    `delete()` is the one thing that clears a reference, because that is the one
    case where the collection is known to be gone.
    """
    cols = list(cols)

    def _id(ref):
        c = resolve_ref(ref, cols)
        return (c.id or c.name) if c is not None else (str(ref or "").strip() or None)

    seen, hidden = set(), []
    for h in (state.get("hidden") or []):
        r = _id(h)
        if r and r not in seen:      # two spellings of one collection collapse
            seen.add(r)
            hidden.append(r)
    return {**state, "active": _id(state.get("active")), "hidden": hidden}


@_serialised
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
    """Library phases that are in NO collection at all, in index order.

    Schema 2: "filed" means a member of any group, because membership is a tag
    and a phase has no single home. The old version counted only EXCLUSIVE
    collections, which with tags is none of them -- so `filed` would always be
    empty and "not yet filed" would permanently list the whole library, while the
    suggestion engine offered phases that were already grouped. One of the four
    places §2.5.1 names.
    """
    if idx is None:
        idx = _index()
    if cols is None:
        cols = load_all()[0]
    filed = {m.key for c in cols for m in c.members}
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
    # A phase already in ANY group is not a suggestion. Gating on `exclusive`
    # meant `already` was empty under tags, so the engine proposed phases the
    # user had filed long ago -- the second of the four places §2.5.1 names.
    already = {m.key for c in load_all()[0] for m in c.members}
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
