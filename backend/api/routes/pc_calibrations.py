"""Named pattern-centre calibrations: list, save, suggest, apply (always on request), versions.

Nothing here changes a dataset unless the request names it (``apply``, ``versions``); ``suggest``,
``candidates``, ``from-file`` and ``results`` only read. See ``services/pc_calibrations`` (storage and
the "same settings" rule) and ``services/pc_calibration_apply`` (applying, and which dataset runs on
which version).

The routes are plain ``def``: FastAPI runs them in a worker thread, which is where they belong, because
a calibration file is read under a lock with retries and a file another program holds would otherwise
stall the event loop. The two routes that change a PC (``apply``, ``versions``) are kept apart from a
PC search the way the PC page's own edits are: refused while one runs, and run under the page's state
lock, because a PC given to a dataset meanwhile would be overwritten when the search ends.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.problem import problem
from backend.api.routes.pcrefinement import _refuse_while_optimizing, _state_lock_or_busy
from backend.api.services import pc_calibration_apply as use
from backend.api.services import pc_calibrations as cal
from backend.api.services import pc_sources
from backend.api.services.pc_search import quality_of

logger = logging.getLogger(__name__)
router = APIRouter()

_STATUS = {"unknownCalibration": 404, "unknownVersion": 404, "nameTaken": 409, "unknownDataset": 404,
           "notWritable": 409}
_CHANGING = "applying a calibration"


def _raise(exc: cal.CalibrationError):
    raise problem(_STATUS.get(exc.code, 400), exc.code, str(exc))


def _not_writable(exc: OSError):
    """The calibration file could not be written (another program may hold it): the code ``apply`` answers with."""
    logger.warning("calibration: could not write a calibration file", exc_info=exc)
    raise problem(409, "notWritable", "the calibration file could not be written (another program may hold it), "
                                      "so nothing was changed")


def _store():
    from backend.api.services.calibration_store import calibration_store
    return calibration_store


def _file_root(store, dataset: str) -> str:
    """The dataset of the FILE ``dataset`` belongs to: ``dataset`` itself, or, for a view of a file (a crop or a copy
    made in the viewer: ``is_view_of_parent``), the file it views, walking parents through views only. A file that
    took the PC of another one (``copy-pc`` sets its ``parent_name``) is a file of its own, not a view."""
    seen = {dataset}
    while True:
        entry = store.get_entry(dataset)
        parent = getattr(entry, "parent_name", None)
        if entry is None or not entry.is_view_of_parent or not parent or parent in seen:
            return dataset
        seen.add(parent)
        dataset = parent


def _entry_of(dataset: str):
    """The store entry of the FILE a dataset belongs to (see ``_file_root``), or a 404."""
    store = _store()
    root = _file_root(store, dataset)
    entry = store.get_entry(root)
    if entry is None:
        raise problem(404, "unknownDataset", f"{dataset!r} is not loaded")
    return root, entry


def _require_pc(store, dataset: str):
    """The PC of ``dataset`` when it has one of its own to save: the same check for saving a calibration and for
    making a new version. A dataset whose ``pc_source`` is ``missing`` has no pattern centre: its detector holds
    kikuchipy's placeholder (0.5, 0.5, 0.5), which must never become a version. ``CalibrationStore.get_pc`` answers
    ``None`` for such a dataset today; the source is checked as well so that this does not depend on that."""
    own = store.get_entry(dataset)
    pc = store.get_pc(dataset)
    if own is None or own.pc_source == "missing" or pc is None:
        raise problem(400, "noPcToSave", "this file has no pattern centre to save: find or set one first")
    return pc


def _require_targets(store, names) -> None:
    """Every name must be a loaded file (open, or left behind: ``use.resolve_target``) and not a copy of one;
    checked BEFORE anything is written."""
    for name in names:
        target = use.resolve_target(name, store)
        if target is None:
            raise problem(404, "unknownDataset", f"{name!r} is not loaded")
        if getattr(target.entry, "is_view_of_parent", False):
            raise problem(400, "isCopy", f"{name!r} is a copy of a file: name the file itself")


def _libraries() -> dict:
    out = {}
    for name in ("kikuchipy", "pyebsdindex", "orix"):
        try:
            from importlib.metadata import version
            out[name] = version(name)
        except Exception:  # noqa: BLE001 - a missing distribution is not an error here
            out[name] = None
    return out


def _loaded_path_of(file_key) -> Optional[str]:
    """The path the file was loaded with (as the user gave it), if the loaded-files list holds it."""
    if not file_key:
        return None
    from backend.api.routes import ebsd_viewer as ev
    entry = ev._loaded_entry(file_key)
    return entry["path"] if entry else None


def _origin_of(dataset: str) -> dict:
    """How the dataset's PC came about: the search that made it when that is what it still is."""
    store = _store()
    entry = store.get_entry(dataset)
    pc = np.asarray(store.get_pc(dataset), dtype=float).reshape(-1)[:3]
    origin = {"dataset": dataset, "pc_source": entry.pc_source, "libraries": _libraries(),
              "refinement": dict(entry.pc_refinement) if entry.pc_refinement else None,
              # the file the PC was made on, by name (a copy of a file has its parent's file key)
              # (as written on disk: the file key is lower case on Windows)
              "source_name": cal.real_file_name(entry.file_key, _loaded_path_of(entry.file_key)),
              "quality": None}
    search = use.search_of(dataset, store)
    if search and np.allclose(search["pc"], pc, rtol=0, atol=1e-9):
        origin["quality"] = quality_of(search)
        origin["method"] = search["mode"]
        origin["search"] = {k: search.get(k) for k in ("mode", "method", "cost", "n_patterns", "recentred",
                                                          "far_box_used")}
        origin["search"]["confidence"] = (search.get("confidence") or {}).get("state")
        origin["search"]["contrast"] = (search.get("confidence") or {}).get("contrast")
    elif entry.pc_source == "manual":
        origin["method"] = "typed"
    elif entry.pc_source == "calibration":
        origin["method"] = "calibration"
    elif entry.pc_source == "header":
        origin["method"] = "file"
    else:
        origin["method"] = str(entry.pc_source)
    return origin


# ------------------------------------------------------------------ requests

class CreateRequest(BaseModel):
    name: str
    dataset: str
    author: str = ""          # free text; empty means "unknown" (never the name of the machine's user)


class RenameRequest(BaseModel):
    name: str


class ApplyRequest(BaseModel):
    datasets: List[str]
    accept_mismatch: List[str] = []
    version: Optional[int] = None


class VersionRequest(BaseModel):
    dataset: str
    apply_to: List[str] = []
    accept_mismatch: List[str] = []
    #: The version of this calibration the question was about (the version the files in ``apply_to`` were on when it
    #: was asked). A file in ``apply_to`` that is not on exactly that version any more is left as it is
    #: (``status: "left"``). Without it, nothing is compared.
    from_version: Optional[int] = None


class FromFileRequest(BaseModel):
    path: str
    dataset: str


# ------------------------------------------------------------------ reads

@router.get("")
def list_calibrations():
    found, problems = cal.load_all()
    return {"calibrations": [cal.summary(c) for c in found], "problems": problems}


@router.get("/suggest")
def suggest(dataset: str, all: bool = False):
    """The calibrations that fit the file ``dataset`` belongs to. Read-only: the file's PC is not touched.

    ``target`` is the file the offer is applied to: ``dataset`` itself, or the file a copy was made from (``apply``
    refuses a copy with ``isCopy``). The copy follows unless it has a PC of its own, which the answer of ``apply``
    shows (``pc_map_actions`` lists the datasets written).

    ``all=true`` lists every calibration with its comparison to the file, fitting or not (the "Take PC from..."
    window): a file whose settings could not be read is then compared by its pattern shape alone, so every
    calibration of the same size reads "cannot be checked" (``unverified``) and every other one "differs".

    A name the store does not know (the page asked for the file it has just left) is not an error: the answer is
    200 with no suggestions and ``unknown_dataset: true``."""
    store = _store()
    if store.get_entry(_file_root(store, dataset)) is None:
        return {"dataset": dataset, "target": dataset, "suggestions": [], "current": None, "unverified": False,
                "unknown_dataset": True}
    root, entry = _entry_of(dataset)
    ref = use.current_ref(root, _store())
    features = getattr(entry, "features", None)
    if all and not features:
        features = {"pattern_shape": [int(v) for v in entry.detector_shape]}
    if not all and (not features or not cal.stated_settings(features)):
        # settings that could not be read, or a file that states none of them (an .ebsp, an .up1): nothing
        # but the pattern size could be compared, which is not a check
        return {"dataset": dataset, "target": root, "suggestions": [], "current": ref, "unverified": True}
    fingerprint = None
    if entry.file_key:
        try:
            fingerprint = cal.fingerprint_of(entry.file_key)
        except cal.CalibrationError:
            logger.info("suggest: no fingerprint for %r", dataset, exc_info=True)
    from backend.api.services.pc_in_use import pc_in_use_of
    in_use = pc_in_use_of(root, _store())
    found = cal.suggest(features, fingerprint, replaces={"pc": in_use["pc"], "kind": in_use["kind"]},
                        only_fitting=not all)
    if ref is not None and not all:
        # already on this very version: nothing to propose
        found = [s for s in found if not (s["id"] == ref["id"] and s["version"] == ref["version"])]
    return {"dataset": dataset, "target": root, "suggestions": found, "current": ref, "unverified": False}


@router.get("/for-dataset/{dataset}")
def for_dataset(dataset: str):
    """The calibration a dataset was given, whether its PC still is that calibration's, and the other
    datasets that run on exactly that version (what a refine at this dataset would leave behind)."""
    return use.followers(dataset, _store())


@router.post("/from-file")
def from_file(req: FromFileRequest):
    """Read the PC another file holds and propose it for ``dataset``. Writes nothing.

    ``notes`` are the English sentences; ``note_items`` are the same with a stable ``code`` and ``params``
    for the interface to translate."""
    root, entry = _entry_of(req.dataset)
    try:
        got = pc_sources.read_pc(Path(req.path), shape=tuple(entry.detector_shape))
    except pc_sources.PcSourceError as exc:
        raise problem(400, exc.code, str(exc))
    target = [int(v) for v in entry.detector_shape]
    out = got.to_dict()
    out.update(dataset=req.dataset, target_shape=target,
               shape_matches=(got.shape is None or list(got.shape) == target),
               current_pc=None if entry.pc_source == "missing" else [float(v) for v in np.asarray(
                   _store().get_pc(root)).reshape(-1)[:3]])
    return out


@router.get("/{cal_id}")
def get_calibration(cal_id: str):
    try:
        calibration = cal.get(cal_id)
    except cal.CalibrationError as exc:
        _raise(exc)
    if calibration is None:
        raise problem(404, "unknownCalibration", "no such calibration")
    return {**cal.summary(calibration), "versions": calibration["versions"],
            "members": [{"name": m["name"], "version": m["version"], "applied": m.get("applied")}
                        for m in calibration["members"]]}


@router.get("/{cal_id}/candidates")
def candidates(cal_id: str):
    store = _store()
    try:
        found = use.candidates(cal_id, store)
    except cal.CalibrationError as exc:
        _raise(exc)
    for item in found:
        if item.get("path") is not None:               # a file that is not open: its row says where it is
            continue
        # where the file is loaded from: the page opens it through the app's own file switch
        entry = store.get_entry(item["dataset"])
        item["path"] = _loaded_path_of(entry.file_key) if entry is not None and entry.file_key else None
    return {"candidates": found}


@router.get("/{cal_id}/results")
def results_state(cal_id: str):
    """The stored results that were computed at an older version of this calibration."""
    try:
        calibration = cal.get(cal_id)
    except cal.CalibrationError as exc:
        _raise(exc)
    if calibration is None:
        raise problem(404, "unknownCalibration", "no such calibration")
    latest = int(cal.latest(calibration)["version"])
    from backend.api.routes import indexing
    from backend.api.services import pc_record
    out = []
    for rid, res in list(indexing._result_registry.items()):
        record = pc_record.record_of(res)
        ref = (record or {}).get("calibration")
        if not ref or ref.get("id") != cal_id:
            continue
        meta = res.metadata if isinstance(getattr(res, "metadata", None), dict) else {}
        out.append({"result_id": rid, "source_file": Path(str(meta.get("source_file") or "")).name or None,
                    "version": int(ref["version"]), "latest": latest,
                    "state": "current" if int(ref["version"]) == latest else "older"})
    return {"calibration": cal_id, "latest": latest, "results": out}


# ------------------------------------------------------------------ writes

@router.post("")
def create_calibration(req: CreateRequest):
    """Save the PC of ``req.dataset`` as a named calibration (version 1), with the settings of its file."""
    store = _store()
    root, entry = _entry_of(req.dataset)
    pc = _require_pc(store, req.dataset)
    features = getattr(entry, "features", None)
    if not features:
        raise problem(400, "needsFeatures", "the acquisition settings of this file could not be read")
    fingerprint = None
    try:
        fingerprint = cal.fingerprint_of(entry.file_key) if entry.file_key else None
    except cal.CalibrationError:
        logger.warning("create: the file of %r cannot be fingerprinted", req.dataset, exc_info=True)
    try:
        made = cal.create(name=req.name, pc=pc, features=features,
                          origin=_origin_of(req.dataset), author=req.author, member=fingerprint)
    except cal.CalibrationError as exc:
        _raise(exc)
    except OSError as exc:
        _not_writable(exc)
    use.note_dataset(made["id"], req.dataset, 1, store)
    return cal.summary(made)


@router.put("/{cal_id}")
def rename_calibration(cal_id: str, req: RenameRequest):
    try:
        return cal.summary(cal.rename(cal_id, req.name))
    except cal.CalibrationError as exc:
        _raise(exc)


@router.delete("/{cal_id}")
def delete_calibration(cal_id: str):
    try:
        removed = cal.delete(cal_id)
    except cal.CalibrationError as exc:
        _raise(exc)
    if not removed:
        raise problem(404, "unknownCalibration", "no such calibration")
    return {"deleted": cal_id}


@router.post("/{cal_id}/apply")
def apply_calibration(cal_id: str, req: ApplyRequest):
    """Give the calibration's PC to the datasets NAMED in the request. A dataset whose settings differ is
    refused unless it is named again in ``accept_mismatch``. Every applied answer says what happened to a
    per-pixel PC map (``pc_map_action``, and ``pc_map_actions`` for every dataset written).

    Refused (409) while a PC search runs or another PC job holds the page's lock."""
    _refuse_while_optimizing(_CHANGING)
    with _state_lock_or_busy(_CHANGING):
        _refuse_while_optimizing(_CHANGING)               # one may have started while the lock was awaited
        try:
            results = use.apply(cal_id, req.datasets, accept_mismatch=req.accept_mismatch, store=_store(),
                                version=req.version)
        except cal.CalibrationError as exc:
            _raise(exc)
    return {"calibration": cal_id, "results": results}


@router.post("/{cal_id}/versions")
def new_version(cal_id: str, req: VersionRequest):
    """The PC ``req.dataset`` has now becomes the next version of the calibration; the dataset and the
    datasets named in ``apply_to`` are put on it. The old version stays readable.

    Everything is checked before anything is written: the dataset must have a PC of its own and every name in
    ``apply_to`` must be a loaded file and not a copy. A file in ``apply_to`` is put on the new version only while it
    still is on version ``from_version`` of this calibration (its PC is that version's): the list the page asked
    about can be older than the files (one got another PC since). Such a file is not written; its answer is
    ``{"dataset", "status": "left"}``. Refused (409) while a PC search runs."""
    _refuse_while_optimizing(_CHANGING)
    store = _store()
    with _state_lock_or_busy(_CHANGING):
        _refuse_while_optimizing(_CHANGING)
        _entry_of(req.dataset)
        pc = _require_pc(store, req.dataset)              # the same check as saving: never the placeholder
        _require_targets(store, req.apply_to)             # before anything is written
        # who is still on the version the question was about: read now, before anything is written
        still_on = set(req.apply_to)
        if req.from_version is not None:
            still_on = set()
            for name in req.apply_to:
                ref = use.current_ref_of(name, store)
                if ref and ref["id"] == cal_id and ref["version"] == int(req.from_version):
                    still_on.add(name)
        try:
            made = cal.add_version(cal_id, pc=pc, origin=_origin_of(req.dataset))
            number = int(cal.latest(made)["version"])
            use.note_dataset(cal_id, req.dataset, number, store)
            targets = [n for n in req.apply_to if n in still_on]
            written = {a["dataset"]: a for a in use.apply(
                cal_id, targets, accept_mismatch=req.accept_mismatch, store=store)} if targets else {}
            applied = [written.get(n) or {"dataset": n, "status": "left"} for n in req.apply_to]
        except cal.CalibrationError as exc:
            _raise(exc)
        except OSError as exc:
            _not_writable(exc)
    return {"calibration": cal.summary(made), "applied": applied}
