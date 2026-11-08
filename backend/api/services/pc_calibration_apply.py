"""Applying a calibration's PC to datasets, and knowing afterwards which dataset runs on which version.

THE RULE: a dataset gets a calibration's PC only because a caller NAMED it (``apply``), and a
dataset whose settings differ from the calibration's gets it only if the caller names it a second
time as accepted (``accept_mismatch``). Nothing here is called by a load, a refine or a suggestion.

The PC itself is written by the one routine that writes a PC to a dataset
(``pcrefinement._writeback_refined_pc``: the dataset, its parent and the copies of its family that
never had a PC of their own, and the PC page's session of each but the named one) plus
``pcrefinement._session_takes_pc`` for the named dataset's own PC page. This module only decides WHETHER to
write, and remembers WHAT was written (``usage``): which calibration and version a file's dataset
was given. "Runs on a calibration" is never stored as a fact: ``current_ref`` compares the dataset's
PC with that version's PC at the moment it is asked, so a PC typed or refined afterwards ends it
without anyone having to remember to update a copy.

FILES THAT ARE NOT OPEN. A load empties the calibration store, so the store holds the open file's datasets only. The
files the user left behind keep their calibration in the viewer's stash (``ebsd_viewer.stashed_file_entries``); every
function here that lists files (``candidates``, ``followers``) or writes to one (``apply``) goes over both, and
reads and writes a stashed file through a ``CalibrationStore.over`` its stash. The page names a file by ``dataset``:
the name of its dataset, or, when that name is also an open dataset's or another stashed file's, the name and the
folder (``label``). ``resolve_target`` is the one place that turns such a name into a store and a dataset.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional

import numpy as np

from backend.api.services import pc_calibrations as cal

logger = logging.getLogger(__name__)

#: Dataset name (the root of its family) -> {"id", "version"}: the calibration version it was given.
_USAGE: Dict[str, dict] = {}
_LOCK = threading.Lock()
PC_ATOL = 1e-9


def _store(store=None):
    if store is not None:
        return store
    from backend.api.services.calibration_store import calibration_store
    return calibration_store


def _memory_key(st, dataset: str) -> str:
    """What the memory of this module (the calibration a file was given, the search that made its PC) is kept
    under: the path of the file the family's root belongs to and the root's name, so that two files of one name
    (the same stem in two folders) and two measurements of one file do not share it. The measurement a pattern file
    was paired with is part of the path (``ebsd_viewer.file_scope``): the root's name is the same for both. A root
    without a path (a store filled by hand) is kept under its name."""
    root = st.family_root(dataset)
    entry = st.get_entry(root)
    key = getattr(entry, "file_key", None)
    if not key:
        return str(root)
    scope = getattr(st, "scope", None)
    if scope is None:
        from backend.api.routes import ebsd_viewer
        scope = ebsd_viewer.open_file_scope(str(key))
    return f"{scope}|{root}"


class Target(NamedTuple):
    """One file of ``candidates``/``followers``/``apply``: the name the page knows it by (``label``), its dataset
    and the store that holds it (``name``, ``store``: the open file's store, or one made over a stash), its entry,
    the path it was loaded from (None when not known) and whether it is a file that is not open."""
    label: str
    name: str
    store: Any
    entry: Any
    path: Optional[str]
    stashed: bool


def _stashed_targets(st) -> List[Target]:
    """The files that were left behind (see the module text), except a file that has an entry in the store: the live
    entry wins. Only for the application's own store: a store handed in by a test is the whole world it is asked
    about."""
    from backend.api.services.calibration_store import CalibrationStore, calibration_store
    if st is not calibration_store:
        return []
    from backend.api.routes import ebsd_viewer
    # a file that is live in the store (the viewer's, or a quick-load's) is that entry's row: its stash is older
    live = {e.file_key for e in st.get_all_entries().values() if e.file_key}
    items = sorted((i for i in ebsd_viewer.stashed_file_entries() if i["canonical"] not in live),
                   key=lambda i: i["canonical"])
    taken = set(st.get_all_entries())
    counts = Counter(i["dataset"] for i in items)
    clash = {i["canonical"] for i in items if i["dataset"] in taken or counts[i["dataset"]] > 1}
    labels = {}
    for i in items:
        name = i["dataset"]
        labels[i["canonical"]] = name if i["canonical"] not in clash else f"{name} ({Path(i['path']).parent.name})"
    again = Counter(labels.values())
    for i in items:                        # two folders of one name: the whole folder says which
        if again[labels[i["canonical"]]] > 1:
            labels[i["canonical"]] = f"{i['dataset']} ({Path(i['path']).parent})"
    return [Target(labels[i["canonical"]], i["dataset"],
                   CalibrationStore.over(i["calibration"], ebsd_viewer.file_scope(i["canonical"], _variant_of(i["key"]))),
                   i["entry"], i["path"], True) for i in items]


def _variant_of(key):
    """The variant part of a stash key (``None`` for the default variant)."""
    return key[1] if isinstance(key, tuple) else None


def resolve_target(label: str, store=None) -> Optional[Target]:
    """The file the page calls ``label``: a dataset of the open file's store first (the existing name stays for the
    open file), else a file that was left behind. ``None`` when there is none."""
    st = _store(store)
    entry = st.get_entry(label)
    if entry is not None:
        return Target(label, label, st, entry, None, False)
    return next((t for t in _stashed_targets(st) if t.label == label), None)


def current_ref_of(label: str, store=None) -> Optional[dict]:
    """``current_ref`` of the file the page calls ``label`` (``None`` for none): see ``resolve_target``."""
    target = resolve_target(label, store)
    return None if target is None else current_ref(target.name, target.store)


#: Family root -> the report of the last PC search that set the PC of its family (``SearchReport.to_dict()``),
#: kept so a calibration saved from a dataset can say how its PC was found. A search on a copy is the search
#: of the raw dataset it was made from, as the PC it found is written to both.
_LAST_SEARCH: Dict[str, dict] = {}


def note_search(dataset: str, report: dict, store=None) -> None:
    key = _memory_key(_store(store), dataset)
    with _LOCK:
        _LAST_SEARCH[key] = dict(report)


def search_of(dataset: str, store=None) -> Optional[dict]:
    """The noted search of ``dataset``'s family (see ``note_search``), or ``None``."""
    key = _memory_key(_store(store), dataset)
    with _LOCK:
        found = _LAST_SEARCH.get(key)
        return None if found is None else dict(found)


def restore_search(dataset: str, report: Optional[dict], store=None) -> None:
    """Make the noted search of ``dataset``'s family ``report`` again, or none (``None``): what "Put it back" does."""
    if report is not None:
        note_search(dataset, report, store)
        return
    key = _memory_key(_store(store), dataset)
    with _LOCK:
        _LAST_SEARCH.pop(key, None)


def memory_usage_of(dataset: str, store=None) -> Optional[dict]:
    """The calibration version this process remembers ``dataset``'s family was given (``None``: none). Only the
    memory, not the member lists: it is what a step back keeps (``restore_usage``)."""
    key = _memory_key(_store(store), dataset)
    with _LOCK:
        entry = _USAGE.get(key)
        return None if entry is None else dict(entry)


def restore_usage(dataset: str, usage: Optional[dict], store=None) -> None:
    """Make the remembered usage of ``dataset``'s family ``usage`` again, or none (``None``): what "Put it back" does,
    so that the family is on the calibration it was on before the write whose step is taken back."""
    key = _memory_key(_store(store), dataset)
    with _LOCK:
        if usage is None:
            _USAGE.pop(key, None)
        else:
            _USAGE[key] = dict(usage)


def forget_all() -> None:
    """Test hook."""
    with _LOCK:
        _USAGE.clear()
        _LAST_SEARCH.clear()


def usage_of(dataset: str, store=None) -> Optional[dict]:
    """The calibration version ``dataset`` (its family root) was given, whatever its PC is now.

    The memory of this process first; when it has nothing (after a restart) the calibration files' own
    member lists (``_member_usage``): a file that is a member of a calibration was given that member's
    version. A file that never was a member has no usage. Both are reads: nothing is applied."""
    st = _store(store)
    root = st.family_root(dataset)
    with _LOCK:
        entry = _USAGE.get(_memory_key(st, dataset))
        if entry is not None:
            return dict(entry)
    return _member_usage(root, st)


def _member_usage(root: str, st) -> Optional[dict]:
    """``{"id", "version"}`` of the calibration whose member list holds the file of ``root`` (name, size and
    header id: ``same_file``); the one it was last applied to when it is a member of several."""
    entry = st.get_entry(root)
    fingerprint = _fingerprint_of_entry(entry) if entry is not None else None
    if fingerprint is None:
        return None
    best = None
    found, _problems = cal.load_all()
    for calibration in found:
        for member in calibration["members"]:
            if cal.same_file(member, fingerprint) and (best is None or str(member.get("applied") or "") > best[0]):
                best = (str(member.get("applied") or ""), calibration["id"], int(member["version"]))
    return None if best is None else {"id": best[1], "version": best[2]}


def _remember(st, dataset: str, cal_id: str, version: int) -> None:
    key = _memory_key(st, dataset)
    with _LOCK:
        _USAGE[key] = {"id": cal_id, "version": int(version)}


def current_ref(dataset: str, store=None) -> Optional[dict]:
    """``{"id", "name", "version"}`` when ``dataset`` was given a calibration's PC and still has exactly
    that PC; ``None`` otherwise (never given one, or its PC was typed, refined or replaced since)."""
    st = _store(store)
    used = usage_of(dataset, st)
    if used is None:
        return None
    pc = st.get_pc(dataset)
    if pc is None:
        return None
    try:
        calibration = cal.get(used["id"])
        if calibration is None:
            return None
        wanted = cal.version_pc(calibration, used["version"])
    except cal.CalibrationError:
        return None
    if not np.allclose(np.asarray(pc, dtype=float).reshape(-1)[:3], wanted, rtol=0, atol=PC_ATOL):
        return None
    return {"id": calibration["id"], "name": calibration["name"], "version": int(used["version"])}


def _fingerprint_of_entry(entry) -> Optional[dict]:
    """The file an entry belongs to, by name, size and header id; ``None`` when it has no readable path."""
    if not getattr(entry, "file_key", None):
        return None
    try:
        return cal.fingerprint_of(entry.file_key)
    except cal.CalibrationError:
        logger.warning("calibration: could not fingerprint the file of %r", entry.dataset_name, exc_info=True)
        return None


def _compare(calibration: dict, features: Optional[dict]) -> dict:
    """The comparison of a file's settings with a calibration's, for ``candidates`` and ``apply``. A file whose
    settings could not be read at all, and a file that states nothing the calibration states (so that nothing
    but the pattern shape could be compared), are ``unverified``: they need the second naming."""
    if not features:
        return {"match": False, "differences": [], "unknown": list(cal.SETTING_KEYS), "checked": [],
                "unverified": True}
    return cal.compare_features(calibration["features"], features)


def candidates(cal_id: str, store=None) -> List[dict]:
    """Every dataset that is the entry of a FILE (not a copy of one) in the calibration store, and every file that was
    left behind (the module text), with whether its settings match the calibration's and what differs: the list a
    person ticks from. A row of a file that is not open says so (``stashed``) and carries the path it is loaded from."""
    st = _store(store)
    calibration = cal.get(cal_id)
    if calibration is None:
        raise cal.CalibrationError("unknownCalibration", "no such calibration", id=cal_id)
    top = cal.latest(calibration)
    out = []
    targets = [Target(name, name, st, entry, None, False) for name, entry in sorted(st.get_all_entries().items())]
    for target in targets + _stashed_targets(st):
        entry = target.entry
        if getattr(entry, "is_view_of_parent", False):
            continue
        compared = _compare(calibration, getattr(entry, "features", None))
        ref = current_ref(target.name, target.store)
        has_pc = entry.pc_source != "missing"
        pc = target.store.get_pc(target.name) if has_pc else None
        row = {
            "dataset": target.label,
            "has_pc": has_pc,
            # what the file has now (the table "old PC"), the calibration version it still runs on (None when its PC
            # is no longer that version's) and the settings the file states
            "pc": None if pc is None else [float(v) for v in np.asarray(pc, dtype=float).reshape(-1)[:3]],
            "ref": ref,
            "settings": {"file": cal.settings_of(getattr(entry, "features", None))},
            "match": bool(compared["match"] and not compared["unverified"]), "unverified": compared["unverified"],
            "differences": compared["differences"], "unknown": compared["unknown"],
            "checked": compared["checked"],
            "on_this_version": bool(ref and ref["id"] == cal_id and ref["version"] == int(top["version"])),
            "on_calibration": bool(ref and ref["id"] == cal_id),
        }
        if target.stashed:
            row["stashed"] = True
            row["path"] = target.path
        out.append(row)
    return out


def apply(cal_id: str, datasets: Iterable[str], *, accept_mismatch: Iterable[str] = (), store=None,
          version: Optional[int] = None) -> List[dict]:
    """Give the PC of a calibration (its latest version, or ``version``) to the named datasets.

    One answer per name: ``applied``, ``mismatch`` (the settings differ and the name was not accepted;
    nothing written), ``notLoaded``, ``isCopy`` or ``notWritable`` (the calibration file could not be written, e.g.
    another program holds it; nothing written). At least one name is required. An applied answer says what
    happened to a per-pixel PC map (``pc_map_action``: "replaced" | "shifted" | "none"; ``pc_map_actions``: the
    action for every dataset written) and ``previous`` (``{"pc", "had_pc"}``: the PC it had before).

    This function takes no lock: the route that calls it holds the one that keeps a refine and an apply apart."""
    names = [str(n) for n in datasets]
    if not names:
        raise cal.CalibrationError("needsTargets", "name the datasets the calibration is to be applied to")
    st = _store(store)
    calibration = cal.get(cal_id)
    if calibration is None:
        raise cal.CalibrationError("unknownCalibration", "no such calibration", id=cal_id)
    number = int(version) if version is not None else int(cal.latest(calibration)["version"])
    pc = cal.version_pc(calibration, number)
    origin = next(v for v in calibration["versions"] if int(v["version"]) == number).get("origin") or {}
    accepted = {str(n) for n in accept_mismatch}
    refinement = None
    if origin.get("refinement"):
        refinement = dict(origin["refinement"])
    from backend.api.routes import pcrefinement          # the one writer of a PC to a dataset
    results = []
    for name in names:
        target = resolve_target(name, st)
        if target is None:
            results.append({"dataset": name, "status": "notLoaded"})
            continue
        entry = target.entry
        if getattr(entry, "is_view_of_parent", False):
            results.append({"dataset": name, "status": "isCopy"})
            continue
        compared = _compare(calibration, getattr(entry, "features", None))
        mismatch = compared["unverified"] or (not compared["match"])
        if mismatch and name not in accepted:
            results.append({"dataset": name, "status": "mismatch", "unverified": compared["unverified"],
                            "differences": compared["differences"]})
            continue
        # The calibration file can fail (another program holds it); the PC in memory cannot. So the member
        # is written first: a failure writes nothing, and the dataset keeps the PC it had.
        fingerprint = _fingerprint_of_entry(entry)
        if fingerprint is not None:
            try:
                cal.set_member(cal_id, fingerprint, number)
            except OSError:
                logger.warning("calibration: could not write the member list of %r", cal_id, exc_info=True)
                results.append({"dataset": name, "status": "notWritable"})
                continue
        written: Dict[str, Optional[str]] = {}
        # A file that is not open is written in its stash and touches no PC page: a page of that dataset name is
        # the open file's. Its own page follows the store when the file is opened.
        store_action = pcrefinement._writeback_refined_pc(
            target.name, pc, store=target.store, source="calibration", refinement=refinement,
            on_refined_map="replace", report=written, touch_sessions=not target.stashed)
        # the write refreshes the PC pages of the parent, the root and the followers; the named dataset's own
        # page is refreshed here (a refine updates its own controller, a calibration does not)
        controller_action = (None if target.stashed
                             else pcrefinement._session_takes_pc(target.name, pc, store_action))
        _remember(target.store, target.name, cal_id, number)
        if target.stashed:                  # the page knows the file by its label, not by the name in its stash
            written = {(target.label if k == target.name else k): v for k, v in written.items()}
        results.append({"dataset": name, "status": "applied", "pc": [float(v) for v in pc],
                        "version": number, "mismatch_accepted": bool(mismatch),
                        "pc_map_action": pcrefinement._pc_map_action(controller_action, store_action, written),
                        "pc_map_actions": written,
                        # what the dataset's PC was before (None, or "had_pc" False): "Put it back" restores it
                        "previous": pcrefinement._previous_of(target.name, target.store)})
    return results


def note_dataset(cal_id: str, dataset: str, version: int, store=None) -> None:
    """The dataset's PC IS this calibration version (it was just made from it): remember that."""
    st = _store(store)
    _remember(st, dataset, cal_id, version)
    entry = st.get_entry(st.family_root(dataset))
    fingerprint = _fingerprint_of_entry(entry) if entry is not None else None
    if fingerprint is not None:
        try:
            cal.set_member(cal_id, fingerprint, version)
        except OSError:
            # the calibration file is held: the memory above answers for this process, and the member list (what a
            # restart reads) is written by the next apply or version
            logger.warning("calibration: could not write the member list of %r", cal_id, exc_info=True)


def followers(dataset: str, store=None) -> dict:
    """For the dataset a refine just changed: the calibration it was on, whether its PC still is that
    calibration's, and the other datasets that run on exactly that version."""
    st = _store(store)
    used = usage_of(dataset, st)
    if used is None:
        return {"calibration": None, "diverged": False, "followers": []}
    try:
        calibration = cal.get(used["id"])
    except cal.CalibrationError:                 # its file became unreadable or newer: nothing to follow
        calibration = None
    if calibration is None:
        return {"calibration": None, "diverged": False, "followers": []}
    root = st.family_root(dataset)
    others = []
    # every file in the store and every file that was left behind, not only the ones this process gave something:
    # after a restart the memory is empty and ``usage_of`` answers from the calibration files' member lists
    targets = [Target(n, n, st, e, None, False) for n, e in st.get_all_entries().items() if n != root]
    for target in targets + _stashed_targets(st):
        if getattr(target.entry, "is_view_of_parent", False):
            continue
        ref = current_ref(target.name, target.store)
        if ref and ref["id"] == used["id"] and ref["version"] == used["version"]:
            others.append(follower_of(target.label, target.entry))
    return {
        "calibration": {"id": calibration["id"], "name": calibration["name"], "version": int(used["version"]),
                        "latest": int(cal.latest(calibration)["version"])},
        "diverged": current_ref(dataset, st) is None,
        "followers": sorted(others, key=lambda f: f["dataset"]),
    }


def follower_of(name: str, entry) -> dict:
    """One follower of ``followers``: the file, whether it holds a PC for every map point, and what kind of map
    that is. A new calibration version REPLACES a map the grid calibration made (``"grid"``: the map is dropped,
    one PC remains) and MOVES a map the file brought along (``"header"``: its shape stays, its mean becomes the PC);
    ``map_kind`` is ``None`` for a file with a single PC."""
    # a file left behind that dropped its header map still has one (``map_dropped``)
    has_map = getattr(entry, "pc_map", None) is not None or bool(getattr(entry, "map_dropped", False))
    kind = None
    if has_map:
        kind = "grid" if getattr(entry, "pc_source", None) == "refined_map" else "header"
    return {"dataset": name, "per_pixel": has_map, "map_kind": kind}
