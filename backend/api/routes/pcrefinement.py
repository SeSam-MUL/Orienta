"""
PC Refinement API Routes

Wraps PCController for:
- Phase loading from CIF
- Indexing parameter configuration
- PC optimization (background task)
- Result retrieval
"""

import asyncio
import contextlib
import logging
import os
import threading
import uuid
from typing import Literal, Optional, List, Union

import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, model_validator

from backend.api.services.image_utils import array_to_base64_raw
from pc_controller import PhaseSetError, pc_is_per_pixel
from hough_reflectors import HoughLibraryTooBig
from backend.api.services.calibration_store import calibration_store
from backend.api.services.detector_convention import for_kikuchipy_projection

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level state
from dataclasses import dataclass, field as _dc_field

# Cap _optimization_tasks at MAX_TRACKED_TASKS so a long backend session
# that runs many PC optimizations / grid calibrations doesn't leak memory.
from collections import OrderedDict as _OrderedDict
MAX_TRACKED_TASKS = 50
_optimization_tasks: _OrderedDict = _OrderedDict()


def _track_task(task_id: str, initial_state: dict) -> None:
    """Insert a new task while enforcing MAX_TRACKED_TASKS LRU cap."""
    _optimization_tasks[task_id] = initial_state
    while len(_optimization_tasks) > MAX_TRACKED_TASKS:
        _optimization_tasks.popitem(last=False)


def _refinement_record(n_patterns: int, method, mode: Optional[str] = None) -> dict:
    """What the calibration store keeps about a PC that came from refinement.

    ``method`` is what the page sent to the optimiser: "PSO" or "Nelder-Mead". ``mode`` is the search that made
    it, "find" ("Find PC") or "refine" ("Global PC Refine"): the page names where the PC in use came from by
    it. A record without one (the grid calibration writes none, and records saved before it existed) is a refine.
    """
    name = "PSO" if str(method).strip().upper() == "PSO" else "Nelder-Mead"
    record = {"n_patterns": int(n_patterns), "method": name}
    if mode in ("find", "refine"):
        record["mode"] = mode
    return record


def _session_takes_pc(name: str, pc, action) -> Optional[str]:
    """The PC page's session of dataset ``name``, if it has one with a detector, takes ``pc``.

    ``action`` is what the dataset's store entry did with its map: a replaced map is replaced in the
    session's controller too, anything else is shifted. The session's simulation cache is emptied.
    Returns the controller's action ("single" | "shifted" | "replaced"), or None without a session."""
    session = _sessions.get(name)
    if session is None or session.ctrl.detector is None:
        return None
    if action == "replaced":
        controller_action = session.ctrl.set_global_pc(pc)
    else:
        controller_action = session.ctrl.shift_pc_map(pc)
    session.sim_cache.clear()
    return controller_action


def _writeback_refined_pc(dataset_name: str, pc, store=None, source: str = "refined",
                          refinement: Optional[dict] = None,
                          on_refined_map: str = "keep",
                          report: Optional[dict] = None,
                          touch_sessions: bool = True) -> Optional[str]:
    """Write a single PC to the dataset AND its parent (F2).

    A Global PC Refine almost always runs on a *derived* copy (e.g.
    ``..._bg_dyn_avg``), so writing only the active dataset strands the refined
    PC there while the raw parent — the dataset the user phase-tests, indexes and
    exports — keeps the stale header PC. We therefore also write the refined PC to
    the single explicit ``parent_name`` link, and to the copies of that family that
    never had a PC of their own (``CalibrationStore.following_family``: a copy made
    before the refine would otherwise keep the old PC); a copy whose PC was set on
    its own is left alone. The writes go through the F1-gated ``update_pc`` (a
    header/inherited per-pixel map is rigid-shifted; a genuine ``refined_map`` is
    preserved), so this is safe to call unconditionally. ``store`` is injectable
    for testing.

    ``refinement`` (``{"n_patterns", "method"}``) is given by the refine routes
    and recorded with the PC on both entries, so a run indexed at it can credit
    the software that refined it; a manual PC passes none and clears it.

    ``on_refined_map`` is what a grid-calibrated PC map does ("keep" | "shift" |
    "replace", see ``CalibrationStore.update_pc``), for the active dataset AND for
    every dataset the PC is passed on to: a root that kept its old map would show the
    new PC while its indexing, export and phase tests read the old map. Returns what
    happened to the active dataset's map ("single" | "shifted" | "replaced" |
    "kept"), or None when it is not in the store; ``report``, when given, is filled
    with the same for every dataset written ({name: action}).

    Before each dataset is written its PC state is kept (``_pc_before``, one step per family):
    ``put_back`` restores it.

    ``touch_sessions`` False for a ``store`` that is not the open file's (``CalibrationStore.over`` the stash of a
    file that is not open): the PC pages are keyed by dataset name, and a page of that name belongs to the file
    that is open. A page of the stashed file itself follows the store when the file is opened.
    """
    store = store or calibration_store
    # The step back (``_pc_before``): the PC state of every dataset this write really changed, taken
    # just before it was written. A store without ``snapshot`` (a stand-in in a test) has none.
    take = getattr(store, "snapshot", None)
    step = _PCStep()
    step_key = _step_key(store, dataset_name) if take is not None else dataset_name
    if take is not None:
        # the search the family's PC was noted to come from, as it is just before this write
        from backend.api.services import pc_calibration_apply
        step.search = pc_calibration_apply.search_of(dataset_name, store)
        # the calibration the family was given, as it is just before this write (a calibration applied by this
        # write is remembered after it)
        step.usage = pc_calibration_apply.memory_usage_of(dataset_name, store)

    def write(name, **kw):
        """``update_pc_detail`` that keeps the entry as it was. A dataset that is not in the store has no
        snapshot, and an update that raises never reaches the line that keeps it: neither adds to the step."""
        old = take([name]) if take is not None else {}
        done_action = store.update_pc_detail(name, pc, refinement=refinement,
                                             on_refined_map=on_refined_map, **kw)
        step.update(old)
        return done_action

    try:
        action = write(dataset_name, source=source)
        entry = store.get_entry(dataset_name)
        if report is not None:
            report[dataset_name] = action

        def follow(name):
            """Another dataset takes this PC: its store entry, and its PC page if opened."""
            done_action = write(name, source="propagated")
            if report is not None:
                report[name] = done_action
            if touch_sessions:
                _session_takes_pc(name, pc, done_action)

        done = [dataset_name]
        if entry is not None and entry.parent_name:
            follow(entry.parent_name)
            done.append(entry.parent_name)
            # the raw dataset the chain of copies was made from is what the user phase-tests,
            # indexes and exports, however many copies of copies lie between
            root = store.family_root(dataset_name)
            if root not in done:
                follow(root)
                done.append(root)
        # A copy made before this refine (another copy of the same raw dataset, a copy
        # of the copy) still holds the old PC: those that never had one of their own
        # follow it.
        for name in store.following_family(dataset_name, exclude=done):
            follow(name)
        return action
    finally:
        # One step: this write replaces the one before. A write that changed nothing keeps it.
        if step:
            _pc_before[step_key] = step


class _PCStep(dict):
    """One step back: ``{dataset name: its store entry before the write}`` (``CalibrationStore.snapshot``), plus
    ``search``, the noted PC search of the family at that moment (``None``: there was none), which "Put it back"
    makes the noted search again, and ``usage``, the calibration version the family was given (``None``: none),
    which "Put it back" makes the remembered one again, as the PC it puts back is that version's."""
    search: Optional[dict] = None
    usage: Optional[dict] = None


#: The sentence for a PC the calibration store did not take (code ``pcNotStored`` on the page).
_PC_NOT_STORED_TEXT = "This PC is on this page only: indexing will not use it. Load the file again."

#: The PC state of every dataset the last PC write touched, as it was just before it, per family root
#: (``CalibrationStore.snapshot`` entries). One step: the next write of that family replaces it. It is
#: what "Put it back" restores; see :func:`put_back`.
_pc_before: dict = {}


def _mean_pc_of(entry) -> list:
    arr = entry.pc_map if entry.pc_map is not None else entry.pc_single
    return [float(v) for v in np.asarray(arr, dtype=float).reshape(-1, 3).mean(axis=0)]


def _step_key(store, name: str) -> str:
    """Where the step back of ``name``'s family is kept: the root's name, and for the store of a file that is not
    open (``CalibrationStore.over``) the file's path as well, so that it is not the step of an open file of the
    same name."""
    root = store.family_root(name)
    scope = getattr(store, "scope", None)
    return root if scope is None else f"{scope}|{root}"


def _step_of(name: str, store=None):
    """The snapshot entry of ``name`` in the stored step of its family, if it still applies: the live
    entry must be the entry of the same file (the store is keyed by name, and two files can share one)."""
    store = store or calibration_store
    entries = _pc_before.get(_step_key(store, name))
    old = entries.get(name) if entries else None
    live = store.get_entry(name)
    if old is None or live is None or live.file_key != old.file_key:
        return None
    return old


def _previous_of(name: str, store=None) -> Optional[dict]:
    """What the dataset's PC was before the last write, or ``None`` when no step is stored for it.

    ``{"pc": [x, y, z] | None, "had_pc": bool}``: ``had_pc`` is False when the file carried none
    (``pc_source`` "missing": a placeholder is not a PC) and ``pc`` is then None. A per-pixel map
    gives its mean."""
    old = _step_of(name, store)
    if old is None:
        return None
    if old.pc_source == "missing":
        return {"pc": None, "had_pc": False}
    return {"pc": _mean_pc_of(old), "had_pc": True}


def _writeback_geometry(dataset_name: str, store=None, **tilts) -> None:
    """Write detector tilts (``sample_tilt``, ``tilt``, ``azimuthal``) to the
    calibration store, on the dataset AND its explicit parent (as the PC is).

    The PC page's detector is the controller's own copy: a tilt changed there that
    does not reach the store never reaches indexing, which builds its detector from
    the store.
    """
    store = store or calibration_store
    store.update_geometry(dataset_name, **tilts)
    entry = store.get_entry(dataset_name)
    if entry is not None and entry.parent_name:
        store.update_geometry(entry.parent_name, **tilts)


def _pc_map_action(controller_action, store_action, others=None) -> str:
    """What the page is told happened to a per-pixel PC map: "replaced", "shifted"
    or "none". The map indexing reads is the store's, the one the page edits is the
    controller's; whichever of them was changed is reported."""
    both = (controller_action, store_action, *(others or {}).values())
    if "replaced" in both:
        return "replaced"
    if "shifted" in both:
        return "shifted"
    return "none"


@dataclass
class _PCSession:
    """Per-file PC Refinement state.

    Each loaded EBSD dataset gets its own session. Switching files
    swaps the active session; switching back restores the previous
    session exactly as it was (patterns, detector, phase, indexer
    cache, simulation cache). Lives until backend restart.
    """
    ctrl: "object"                  # pc_controller.PCController
    sim_cache: dict = _dc_field(default_factory=dict)
    #: Version of the per-phase reflector registry the sim cache was filled under.
    spec_version: int = -1
    #: The file this session belongs to (canonical path). Two files can have the same
    #: dataset name (the same stem in two folders); a session of one must never serve
    #: the other (its calibration patterns, its PC, a refine written to its entry).
    file_key: str = ""


# Per-dataset sessions. Key = dataset name (matches ``_active_dataset``
# in ebsd_viewer). Sessions are lazily created on first access for a
# dataset by ``_get_session`` and persist until backend restart.
_sessions: dict[str, _PCSession] = {}
# Sessions of a file that is not the one loaded now, under (file key, dataset name):
# switching back to that file gets its session back.
_parked_sessions: dict[tuple, _PCSession] = {}


def _current_file_key() -> str:
    """Canonical path of the file the viewer has loaded now ("" when none); for a pattern file the
    user paired with a measurement, the path and that measurement (the same file with another
    measurement has other patterns and another grid, so it has its own session)."""
    from backend.api.routes import ebsd_viewer as _ev
    path = getattr(_ev, "_ebsd_file_path", None)
    try:
        if not path:
            return ""
        key = _ev._canonical_path(path)
        variant = getattr(_ev, "_ebsd_variant_key", None)
        return key if not variant or len(variant) < 3 else f"{key}|{variant[2]}"
    except Exception:  # noqa: BLE001
        return ""


def _session_of_this_file(key: str, file_key: str):
    """The session of dataset ``key`` for the file ``file_key``; a session of another
    file with the same dataset name is parked, that file's own session is put back."""
    sess = _sessions.get(key)
    mine = getattr(sess, "file_key", "") if sess is not None else ""
    if sess is not None and file_key and mine and mine != file_key:
        _parked_sessions[(mine, key)] = _sessions.pop(key)
        sess = None
    if sess is None and file_key:
        back = _parked_sessions.pop((file_key, key), None)
        if back is not None:
            _sessions[key] = back
            sess = back
    return sess


def drop_session(name: str) -> bool:
    """Forget the PC session of dataset ``name`` of the loaded file (it was deleted).

    Dataset names are recycled (a deepcopy or a crop of the same source gets its
    name again), so a session left behind would be served to a different, new
    dataset: its calibration patterns, PC and simulated results. A session of
    another file with the same name is not this dataset's and stays (parked for its
    file). Plain dict operations, no ``_state_lock``: an Index All can hold that for
    minutes, and a job that is running keeps its own controller. Returns whether a
    session of this dataset was dropped.
    """
    file_key = _current_file_key()
    _pc_before.pop(name, None)           # the step back of a deleted dataset goes with it
    dropped = _session_of_this_file(name, file_key) is not None
    _sessions.pop(name, None)
    if file_key:
        dropped = (_parked_sessions.pop((file_key, name), None) is not None) or dropped
    return dropped


def resync_sessions_with_store() -> list:
    """After a load, a restore or an import: make the PC page agree with the store.

    The calibration store is where indexing and exports read the PC from; the PC page
    works on its own controller, which was built from the store once. A load can
    change the store behind its back (a restored calibration, a header PC that now
    wins, an .osc put beside the file), and a page that kept the old PC showed one
    number while indexing used another. For every session of the loaded file the
    controller's detector takes the store's PC (single or map) and tilts, its pixel
    size and binning stay; a session of another file with the same dataset name is
    parked. Returns the names resynced.
    """
    file_key = _current_file_key()
    done = []
    # The sessions this file parked when another one took their name come back first:
    # they are the ones whose controller is oldest, and the store may have changed while
    # they were away (an .osc put beside the file, a calibration restored).
    for fk, name in list(_parked_sessions):
        if file_key and fk == file_key:
            _session_of_this_file(name, file_key)
    for name in list(_sessions):
        sess = _session_of_this_file(name, file_key)
        if _session_follows_store(name, sess):
            done.append(name)
    return done


def _session_follows_store(name: str, sess, hold_lock: bool = True) -> bool:
    """The PC page's ``sess`` of dataset ``name`` takes the store's PC (single or map) and tilts when
    its controller's detector differs from them; its pixel size and binning stay, its simulation cache
    is emptied. Returns whether it changed. ``hold_lock`` False when the caller already holds
    ``_state_lock`` (it is not re-entrant)."""
    from kikuchipy.detectors import EBSDDetector
    if sess is None or sess.ctrl.detector is None:
        return False
    entry = calibration_store.get_entry(name)
    if entry is None:
        return False
    det = sess.ctrl.detector
    want = entry.pc_map if entry.pc_map is not None else entry.pc_single
    want = np.asarray(want, dtype=float)
    have = np.asarray(det.pc, dtype=float)
    tilts_same = (np.isclose(float(det.sample_tilt), entry.sample_tilt)
                  and np.isclose(float(det.tilt), entry.tilt)
                  and np.isclose(float(det.azimuthal), entry.azimuthal))
    same_pc = (have.size == want.size
               and bool(np.allclose(have.reshape(-1), want.reshape(-1), rtol=0, atol=1e-12)))
    if same_pc and tilts_same:
        return False
    new = EBSDDetector(
        shape=tuple(int(v) for v in det.shape), px_size=det.px_size, binning=det.binning,
        tilt=entry.tilt, azimuthal=entry.azimuthal, sample_tilt=entry.sample_tilt, pc=want)
    with (_state_lock if hold_lock else contextlib.nullcontext()):
        sess.ctrl.apply_pixelwise_pc(new)
        sess.sim_cache.clear()
    logger.info("PC page of %r follows the calibration store", name)
    return True

# Thread safety: protects all shared state (sessions, optimization_tasks)
_state_lock = threading.Lock()
# Flag to prevent concurrent optimizations
_optimization_active = False


def _get_session() -> _PCSession:
    """Return (lazy-create) the PC session for the active dataset.

    Per-file state isolation: each dataset has its own controller +
    sim_cache. Switching files restores the previous session for that
    file rather than starting over (per user request, 2026-05-26).
    """
    from backend.api.routes.ebsd_viewer import _active_dataset
    if not _active_dataset:
        # Fallback: an unnamed shared session so endpoints called
        # before any file is loaded still work. This shouldn't happen
        # in normal flow.
        key = ""
    else:
        key = _active_dataset
    file_key = _current_file_key()
    sess = _session_of_this_file(key, file_key)
    if sess is None:
        from pc_controller import PCController
        sess = _sessions[key] = _PCSession(ctrl=PCController(), file_key=file_key)
    elif not getattr(sess, "file_key", "") and file_key:
        sess.file_key = file_key         # a session made before the file was known
    # A change of a phase's reflector selection (either page, or a run) makes the
    # bands simulated with the old one out of date. The indexer itself is rebuilt
    # where it is used, under _state_lock (`_index_and_simulate`, the optimiser,
    # `PCController._ensure_indexer`), never from here: this runs on every request,
    # including while another one is indexing.
    from hough_reflectors import registry_version
    version = registry_version()
    if sess.spec_version != version:
        sess.spec_version = version
        sess.sim_cache.clear()
    return sess


def _get_controller():
    return _get_session().ctrl


def _active_dataset_name():
    """The dataset that is active in the viewer now (read when asked, not when started)."""
    from backend.api.routes.ebsd_viewer import _active_dataset
    return _active_dataset


class _ActiveSimCacheProxy:
    """Module-level proxy that routes every operation to the active
    session's sim_cache. Lets existing call sites (``_sim_cache[idx]``,
    ``_sim_cache.clear()``, ``_sim_cache.pop(...)``) keep working
    unchanged after the per-file refactor.
    """
    def __getitem__(self, k):
        return _get_session().sim_cache[k]

    def __setitem__(self, k, v):
        _get_session().sim_cache[k] = v

    def __delitem__(self, k):
        del _get_session().sim_cache[k]

    def __contains__(self, k):
        return k in _get_session().sim_cache

    def __iter__(self):
        return iter(_get_session().sim_cache)

    def __len__(self):
        return len(_get_session().sim_cache)

    def get(self, k, default=None):
        return _get_session().sim_cache.get(k, default)

    def pop(self, k, *args):
        return _get_session().sim_cache.pop(k, *args)

    def clear(self):
        _get_session().sim_cache.clear()

    def items(self):
        return _get_session().sim_cache.items()

    def keys(self):
        return _get_session().sim_cache.keys()

    def values(self):
        return _get_session().sim_cache.values()


_sim_cache = _ActiveSimCacheProxy()


class LoadPhaseRequest(BaseModel):
    cif_path: str

    @model_validator(mode="after")
    def _not_too_large_to_be_a_cif(self):
        # Checked here so that `/phase/load` and `/phase/add` both refuse a huge
        # file (a renamed 210 MB binary took 12.6 s and ~800 MB to fail) before any
        # parser sees it. An HTTPException passes through pydantic unchanged.
        _check_cif_file(self.cif_path)
        return self


class SetDetectorRequest(BaseModel):
    shape: List[int]        # [height, width]
    pc: List[float]         # [pcx, pcy, pcz]
    sample_tilt: float = 70.0
    camera_tilt: float = 0.0   # camera / detector tilt (deg)
    binning: int = 1
    detector_tilt: float = 0.0  # additional detector tilt (deg)
    azimuthal: float = 0.0      # azimuthal rotation (deg)
    pixel_size: Optional[float] = None  # microns per pixel (physical pixel size)
    # PC values entering this endpoint are in BRUKER convention because
    # kikuchipy.EBSDDetector.pc stores Bruker internally and that's what
    # /api/calibration and /api/pc/detector/info both return to the
    # frontend. Defaulting to 'tsl' here used to silently flip PCy on
    # every Apply (PCy -> 1-PCy via the TSL->Bruker conversion inside
    # kikuchipy's EBSDDetector ctor). Investigation 2026-05-22.
    convention: str = "bruker"  # PC convention: "bruker", "tsl", "oxford", "emsoft"


class UpdateParamsRequest(BaseModel):
    min_d: Optional[float] = None
    f_threshold: Optional[float] = None
    max_reflectors: Optional[int] = None
    n_bands: Optional[int] = None
    # "Apply pixel size": the detector's binning and its physical width (mm). Both
    # are needed to fix the pixel size; either alone changes nothing.
    binning: Optional[int] = None
    detector_width_mm: Optional[float] = None


class RenderPreviewRequest(BaseModel):
    """Trial-geometry forward-sim preview for the PC Refinement page.

    The frontend posts the current PC + tilts; the backend renders a
    simulated EBSP for the selected calibration pattern using these
    geometry values plus the Hough-indexed orientation at the same
    pixel. NCC vs the experimental pattern is returned so the user
    can see in real time whether they've found the right geometry.

    All PC values are in BRUKER convention (matches the rest of the
    app — see SetDetectorRequest comment for context).
    """
    pattern_idx: int
    sht_path: str
    pc: List[float]              # [PCx, PCy, PCz] Bruker
    sample_tilt: float           # deg, typically 65-75
    detector_tilt: float = 0.0   # deg, typically 0-10
    azimuthal: float = 0.0       # deg
    binning: int = 1
    pixel_size: Optional[float] = None  # microns
    max_bandwidth: int = 128     # SHT truncation: 128 = fast, 256 = sharper
    # Optional orientation override. If omitted, Hough is run at this
    # pattern under the trial geometry to obtain one.
    orientation_euler_deg: Optional[List[float]] = None


class AddPatternRequest(BaseModel):
    row: int
    col: int


class OptimizePCRequest(BaseModel):
    patterns: List[int]  # Flat indices of patterns to optimize
    method: str = "PSO"
    search_limit: float = 0.05
    #: "refine": Global PC Refine, starts at the PC the page holds. "find": Find PC, searches a
    #: wide box and needs no start (at least ``pc_search.MIN_PATTERNS_FIND`` patterns).
    mode: Literal["refine", "find"] = "refine"


def _pixel_geometry_from_file(det) -> None:
    """Give the controller's detector the pixel size and binning of the FILE.

    The calibration store keeps only the PC and the tilts, so a detector built from
    it has kikuchipy's default pixel size (1 um) and binning 1. ``extrapolate_pc``
    turns a beam step into a PC shift with ``px_size * binning``: with the default
    the PC drift over a 45 um map came out as 0.56 of the detector width instead
    of 0.0015. The active signal's detector still holds the file's values.
    """
    try:
        from backend.api.routes.ebsd_viewer import _get_active_signal
        sig_det = getattr(_get_active_signal(), "detector", None)
        if sig_det is None or tuple(sig_det.shape) != tuple(det.shape):
            return
        det.px_size = float(sig_det.px_size)
        det.binning = int(sig_det.binning)
    except Exception:  # noqa: BLE001 - the default pixel size is what it was before
        logger.debug("pixel size and binning of the file not carried over", exc_info=True)


def _auto_attach_detector(ctrl):
    """Auto-attach detector from CalibrationStore if controller has none.

    Uses CalibrationStore as single source of truth — immune to the
    kikuchipy deepcopy bug that resets PC to [0.5, 0.5, 0.5].

    The store always returns a single-PC detector (mean PC for per-pixel
    datasets), which is what the PC Refinement controller needs.

    Per-file sessions (see ``_get_session``) mean ``ctrl`` is bound by
    construction to the active dataset, so we only attach when the
    session-local controller has no detector yet.
    """
    if ctrl.detector is not None:
        return  # Already set for this session

    from backend.api.routes.ebsd_viewer import _active_dataset

    # 1. Try CalibrationStore (preferred — always has correct PC)
    det = calibration_store.get_detector(_active_dataset)
    if det is not None:
        # Store may return per-pixel PC; controller needs single PC
        pc_raw = np.array(det.pc)
        if pc_raw.ndim > 1:
            mean_pc = pc_raw.reshape(-1, 3).mean(axis=0)
            from kikuchipy.detectors import EBSDDetector
            det = EBSDDetector(
                shape=tuple(det.shape),
                pc=list(mean_pc),
                sample_tilt=float(det.sample_tilt),
                tilt=float(getattr(det, 'tilt', 0.0)),
                azimuthal=float(getattr(det, 'azimuthal', 0.0)),
            )
        _pixel_geometry_from_file(det)
        ctrl.attach_detector(det)
        logger.info("Auto-attached detector from CalibrationStore: PC=%s",
                     list(np.array(det.pc).flatten()[:3]))
        return

    # 2. Fallback: signal's own detector
    from backend.api.routes.ebsd_viewer import _get_active_signal
    signal = _get_active_signal()
    if signal is None:
        return
    sig_det = getattr(signal, 'detector', None)
    if sig_det is None:
        return

    try:
        pc_raw = np.array(sig_det.pc)
        mean_pc = pc_raw.reshape(-1, 3).mean(axis=0) if pc_raw.ndim > 1 else pc_raw.flatten()[:3]
    except Exception:
        mean_pc = [0.5, 0.5, 0.5]

    from kikuchipy.detectors import EBSDDetector
    new_det = EBSDDetector(
        shape=tuple(sig_det.shape),
        pc=list(mean_pc),
        sample_tilt=float(getattr(sig_det, 'sample_tilt', 70.0)),
        tilt=float(getattr(sig_det, 'tilt', 0.0)),
        azimuthal=float(getattr(sig_det, 'azimuthal', 0.0)),
        px_size=float(getattr(sig_det, 'px_size', 1.0)),
        binning=int(getattr(sig_det, 'binning', 1)),
    )
    ctrl.attach_detector(new_det)
    logger.info("Auto-attached detector from signal (fallback): PC=%s", list(mean_pc))


def _winning_phase_position(ctrl, xmap):
    """Position in ``ctrl.phase_list`` of the phase that won ``xmap``'s pixel 0.

    The crystal map carries the winner's phase id; the reflector sets of a
    multi-phase list are in phase-list order. ``None`` when no phase fitted
    (id -1) or the id is not one of the list's.
    """
    try:
        pid = int(np.asarray(xmap.phase_id).ravel()[0])
    except Exception:
        return None
    ids = [int(i) for i in ctrl.phase_list.ids]
    return ids.index(pid) if pid in ids else None


def _global_ci(cis):
    """The page's "Global CI": the mean CI over ALL calibration patterns.

    Every pattern counts with the CI of its winning phase (``index_data_ci``),
    indexed at the detector's current PC. ``None`` when ``cis`` is empty or a
    pattern has no CI yet: a mean over some of the patterns is not a global value.
    """
    cis = list(cis)
    if not cis or any(c is None for c in cis):
        return None
    return float(sum(cis) / len(cis))


def _same_patterns(ctrl, patterns_data) -> bool:
    """True when ``patterns_data`` are, in order, the calibration patterns' arrays."""
    if len(patterns_data) != len(ctrl.patterns):
        return False
    try:
        return all(np.array_equal(np.asarray(p), np.asarray(c[1]))
                   for p, c in zip(patterns_data, ctrl.patterns))
    except Exception:
        return False


def _index_and_simulate(ctrl, pattern_idx, sim_cache=None):
    """Index a single calibration pattern and simulate Kikuchi line segments.

    Returns dict with {ci, phase_name, segments, n_bands}.
    Results are cached in ``sim_cache`` keyed by pattern_idx; by default the one of
    the active dataset's session. A background job passes the cache of the session
    it started on: the active dataset may have changed while it ran.
    """
    import kikuchipy as kp
    from kikuchipy.signals import EBSD
    from ebsd_utils import create_indexer
    from hough_reflectors import prepare_reflectors

    # The reflector selection of a phase may have changed since the indexer was
    # built: drop it (and what was indexed with it) so it is rebuilt below.
    ctrl.reflector_specs_changed()

    coords, pat = ctrl.patterns[pattern_idx]

    # Verify detector shape matches pattern shape — mismatch causes overlay shifting
    det_shape = tuple(ctrl.detector.shape)
    pat_shape = tuple(pat.shape)
    if det_shape != pat_shape:
        logger.warning("Detector shape %s != pattern shape %s — adjusting detector",
                       det_shape, pat_shape)
        from kikuchipy.detectors import EBSDDetector
        # One PC: the pattern's own pixel when the detector holds a PC map.
        ctrl.detector = EBSDDetector(
            shape=pat_shape,
            pc=[float(v) for v in ctrl.pc_at(coords)],
            sample_tilt=float(ctrl.detector.sample_tilt),
            tilt=float(ctrl.detector.tilt),
            azimuthal=float(ctrl.detector.azimuthal),
        )
        ctrl.indexer = None  # Force indexer recreation with correct detector

    # The detector for THIS pattern: with a per-pixel PC map (grid calibration)
    # PyEBSDIndex and the line simulation take the PC of the pattern's own pixel.
    det = ctrl.detector_at(coords)

    # An indexer copies the PC it was built at; a refine or a typed PC that came
    # after it, or another pixel's, must not be indexed with it.
    if ctrl.indexer is not None and ctrl._indexer_pc_is_stale(det):
        ctrl.indexer = None

    # Ensure indexer
    if ctrl.indexer is None:
        reflectors = prepare_reflectors(
            ctrl.phase_list,
            min_d=getattr(ctrl, 'min_d', 1.0),
            f_threshold=getattr(ctrl, 'f_threshold', 0.1),
            max_reflectors=getattr(ctrl, 'max_reflectors', 70),
        )
        ctrl.reflectors = reflectors
        ctrl.indexer = create_indexer(
            det, ctrl.phase_list, reflectors,
            nBands=getattr(ctrl, 'nBands', 12),
        )

    # Hough index
    # pat is a 2D array (H, W).  hough_indexing requires at least one navigation
    # axis so we add two leading singleton dimensions → shape (1, 1, H, W).
    # Using EBSD(Signal2D(pat)) gives navigation_shape=() which crashes pyebsdindex.
    from backend.api.services.load_choices import for_pyebsdindex
    ebsd = EBSD(for_pyebsdindex(pat)[np.newaxis, np.newaxis], detector=det)
    xmap, index_data, band_data = ebsd.hough_indexing(
        ctrl.phase_list, ctrl.indexer,
        return_index_data=True,
        return_band_data=True,
        verbose=0,
    )

    from pc_controller import index_data_ci, phase_fits
    n_phases = len(ctrl.phase_list.ids)
    try:
        ci = index_data_ci(index_data, n_phases)
    except Exception:
        ci = float(index_data['cm'].flat[0])

    # `phases_in_data` is a PhaseList and `[0]` on it looks up phase ID 0, which
    # raises as soon as the winner is any other phase of a multi-phase list.
    phase_name = ""
    if hasattr(xmap, 'phases_in_data'):
        _won = list(xmap.phases_in_data.names)
        phase_name = str(_won[0]) if _won else ""
    # With several phases loaded, which one won this pattern (position in the
    # phase list; None when no phase fitted). Not set for a single phase.
    multi_phase = len(ctrl.phase_list.ids) > 1
    phase_index = _winning_phase_position(ctrl, xmap) if multi_phase else None

    # Cache in controller too
    ctrl.cache[pattern_idx] = {'ci': ci, 'xmap': xmap, 'band_data': band_data}

    # Simulate Kikuchi lines
    segments = []
    try:
        if multi_phase:
            # One reflector set per phase: simulate the lines of the phase this
            # pattern was indexed as, not of a phase it did not match. The
            # controller keeps its reflectors current (phase changes and
            # indexing-parameter changes rebuild them, and the indexer built
            # above sets them), so they are not computed again per pattern: for
            # a large cell that costs seconds each.
            if phase_index is None:
                raise RuntimeError("no phase fitted this pattern")
            reflectors = ctrl.reflectors[phase_index]
        else:
            reflectors = prepare_reflectors(
                ctrl.phase_list,
                min_d=getattr(ctrl, 'min_d', 1.0),
                f_threshold=getattr(ctrl, 'f_threshold', 0.1),
                max_reflectors=getattr(ctrl, 'max_reflectors', 70),
            )
        simulator = kp.simulations.KikuchiPatternSimulator(reflectors)
        rots = xmap.rotations[0:1]
        sim = simulator.on_detector(for_kikuchipy_projection(det), rots)
        marker = sim.as_markers()[0]
        m_kwargs = marker.get_current_kwargs()
        raw_segs = m_kwargs.get('segments', [])
        for seg in raw_segs:
            (x0, y0), (x1, y1) = seg
            if not (np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1)):
                segments.append([[float(x0), float(y0)], [float(x1), float(y1)]])
    except Exception as e:
        logger.warning("Kikuchi simulation failed for pattern %d: %s", pattern_idx, e)

    result = {
        "ci": ci,
        "phase_name": phase_name,
        "segments": segments,
        "n_bands": len(segments),
        # The PC this pattern was indexed at: its own pixel's with a PC map.
        "pc": [float(v) for v in np.asarray(det.pc, dtype=float).ravel()[:3]],
    }
    if multi_phase:
        result["phase_index"] = phase_index
        # What every loaded phase made of this pattern (the winner is chosen by
        # fit and matched bands, see pc_controller.phase_fits).
        try:
            result["phase_fits"] = phase_fits(index_data, ctrl.phase_list)
        except Exception:
            logger.warning("per-phase fits unavailable for pattern %d", pattern_idx,
                           exc_info=True)
    (_sim_cache if sim_cache is None else sim_cache)[pattern_idx] = result
    return result


@router.post("/pattern/add")
async def add_pattern(req: AddPatternRequest):
    """Add a pattern from the EBSD signal to the PC calibration set."""
    from backend.api.routes.ebsd_viewer import _get_active_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    ctrl = _get_controller()
    # Auto-attach detector from signal (like PyQt5 add_pattern)
    _auto_attach_detector(ctrl)

    # Bounds-check before signal.data[row,col] — same silent-numpy-wrap
    # class as the other bounds-check fixes. Without this, a stale
    # frontend selection (negative row from a previous file) silently
    # adds a pattern from the wrong end of the array as a calibration
    # point, then PC refinement converges on garbage.
    nav_shape = signal.axes_manager.navigation_shape
    n_cols = int(nav_shape[0]) if len(nav_shape) >= 1 else 1
    n_rows = int(nav_shape[1]) if len(nav_shape) >= 2 else 1
    if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Position ({req.row},{req.col}) out of bounds for {n_rows}x{n_cols} grid",
        )

    dataset = _active_dataset_name()

    # The list is changed under _state_lock, in a worker (the lock can be held for
    # minutes by an Index All; the event loop must not wait for it). The check for
    # a running refine is repeated inside: one may have started while this waited.
    def _add_locked():
        # Stored as the map run sees the pixel (an int16 scan rescaled by the range of the
        # whole scan): every indexer the calibration feeds from this list sees those values.
        # Read before the lock: the first pattern of a signal reads the scan's range.
        from backend.api.services.load_choices import signal_pattern
        pattern = signal_pattern(signal, req.row, req.col).copy()
        # The aperture is the detector's: decide it from the pattern as recorded,
        # not from one the viewer has processed (that one's corners are noise).
        recorded = None
        if not getattr(ctrl, "aperture_circular", False):
            recorded = _unprocessed_pattern(dataset, req.row, req.col, pattern.shape)
        with _state_lock_or_busy("adding or removing calibration patterns"):
            _refuse_while_optimizing("adding or removing calibration patterns")
            # Reject duplicates (same row, col already in list)
            for coords, _ in ctrl.patterns:
                if coords[0] == req.row and coords[1] == req.col:
                    return {
                        "success": True,
                        "n_patterns": len(ctrl.patterns),
                        "added": {"row": req.row, "col": req.col},
                        "duplicate": True,
                    }
            # A standalone copy, so the pattern survives a later file switch /
            # signal reload: a view into signal.data
            # becomes invalid once _raw_signals.clear() drops the source signal in
            # load_ebsd. A switch back to this dataset would then crash or render
            # garbage.
            logger.info(f"add_pattern ({req.row},{req.col}) from dataset '{dataset}'")
            if not getattr(ctrl, "aperture_circular", False):
                ctrl.aperture_circular = _has_black_corners(
                    recorded if recorded is not None else pattern)
            ctrl.add_pattern((req.row, req.col), pattern)
            return {
                "success": True,
                "n_patterns": len(ctrl.patterns),
                "added": {"row": req.row, "col": req.col},
                "dataset": dataset,
            }

    _refuse_while_optimizing("adding or removing calibration patterns")
    try:
        return await asyncio.to_thread(_add_locked)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _unprocessed_pattern(dataset, row: int, col: int, shape) -> Optional[np.ndarray]:
    """The pattern at (row, col) as the detector recorded it, if the viewer still holds it.

    A dataset the viewer processed (background removal, frame average, CLAHE...) is
    flagged dirty; a processed copy is a derived dataset whose parent is recorded in the
    calibration store. Walk that chain to the first dataset that is not dirty and read its
    pattern - only while the navigation grid and pattern shape stay the same (a crop
    re-indexes the grid, so its coordinates would point elsewhere). None when there is no
    such dataset; the caller then has only the pattern it stored.
    """
    try:
        from backend.api.routes import ebsd_viewer as ev
    except Exception:
        return None
    seen = set()
    name = dataset
    first = ev._raw_signals.get(dataset)
    nav = tuple(first.axes_manager.navigation_shape) if first is not None else None
    while name and name not in seen:
        seen.add(name)
        sig = ev._raw_signals.get(name)
        if sig is None or nav is None or tuple(sig.axes_manager.navigation_shape) != nav:
            return None
        if name not in ev._dirty_datasets:
            from backend.api.services.load_choices import signal_pattern
            pat = signal_pattern(sig, row, col)     # int16 rescaled: no negative values
            return pat if pat.shape == tuple(shape) else None
        entry = calibration_store.get_entry(name)
        name = getattr(entry, "parent_name", None)
    return None


def _has_black_corners(pattern) -> bool:
    """``detect_circular_aperture`` for evidence only a recorded pattern can give.

    That check compares the corner means with 5 % of the maximum, so on signed
    data (a background-subtracted float pattern, mean near zero) the corners are
    negative and pass - a full-frame detector would read as circular, and the
    session flag would then mask every pattern. A pattern with negative values
    is processed data and says nothing about the aperture.
    """
    from tools.pattern_comparison import detect_circular_aperture
    p = np.asarray(pattern)
    if p.ndim != 2 or p.size == 0 or not np.isfinite(p).all() or float(p.min()) < 0:
        return False
    return bool(detect_circular_aperture(p))


def _session_is_circular(ctrl) -> bool:
    """Whether this session's detector records through a circular (EDAX-style) aperture.

    One session is one dataset and one detector, so the answer holds for every pattern:
    it is decided at add time from the unprocessed pattern (``aperture_circular``) and,
    for a session stored before that existed, from any stored pattern whose corners are
    still black. Processing in the viewer turns the black corners into noise, which is
    why a single processed pattern cannot answer it (patterns added
    after BG Dynamic were shown square, the first one round).
    """
    if getattr(ctrl, "aperture_circular", False):
        return True
    return any(_has_black_corners(p) for _, p in list(ctrl.patterns))


class AutoPickRequest(BaseModel):
    #: True: the calibration list becomes the picked patterns. False: they are added to it (a
    #: position that is already in the list is not added twice).
    replace: bool = True
    #: ``[row, col]`` positions that must not be chosen ("other patterns"): in each block the best
    #: position that is not in this list is taken. The pick is otherwise deterministic.
    exclude: List[List[int]] = []

    @model_validator(mode="after")
    def _positions_are_pairs(self):
        if any(len(p) != 2 for p in self.exclude):
            raise ValueError("every excluded position is [row, col]")
        return self


@router.post("/pattern/auto-pick")
async def auto_pick_patterns(req: AutoPickRequest):
    """Choose calibration patterns for "Find PC" from the whole scan and put them in the list.

    One pattern from each of 3 x 3 blocks of the scan, the best image quality of a regular sample
    of the scan (``services/pc_patterns``). The patterns are stored as copies, with their background
    removed unless the user processed the signal in the viewer (then they are taken as they are);
    the answer says which. A scan of nine patterns or fewer is used completely.

    Like ``/pattern/add`` it changes the calibration list, so it is refused (409) while a PC job
    runs, changes the list under ``_state_lock`` and does its work in a worker thread.
    """
    from backend.api.routes.ebsd_viewer import _get_active_signal, is_active_signal_dirty
    from backend.api.services import pc_patterns
    from backend.api.services.load_choices import for_signal_indexer

    _refuse_while_optimizing("adding or removing calibration patterns")
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")
    ctrl = _get_controller()
    sim_cache = _get_session().sim_cache
    _auto_attach_detector(ctrl)

    nav = signal.axes_manager.navigation_shape
    n_cols = int(nav[0]) if len(nav) >= 1 else 1
    n_rows = int(nav[1]) if len(nav) >= 2 else 1
    prepare = not is_active_signal_dirty()
    file_background = getattr(signal, "static_background", None)
    dataset = _active_dataset_name()

    def read(rows, cols):
        # Only the sampled positions are read (a lazy scan stays on disk), as the map run sees
        # them: an int16 scan is rescaled by the range of the whole scan, as in /pattern/add.
        data = signal.data
        if data.ndim == 2:                       # no navigation axis: the one pattern
            block = data[None]
        elif len(nav) < 2:                       # one navigation axis: the patterns in a row
            block = data.vindex[cols] if hasattr(data, "vindex") else data[cols]
        else:
            block = data.vindex[rows, cols] if hasattr(data, "vindex") else data[rows, cols]
        block = block.compute() if hasattr(block, "compute") else block
        return for_signal_indexer(signal, np.asarray(block))

    def _pick_and_store():
        picked = pc_patterns.pick(read, n_rows, n_cols, prepare=prepare,
                                  file_background=file_background, exclude=req.exclude)
        if not picked.coords:
            # Refused before the list is touched: "replace" must not leave the page without patterns.
            if req.exclude:
                raise _coded(400, "noPatternLeft",
                             "No calibration pattern is left to pick: every sampled position that "
                             "could be chosen is in the list of positions to leave out.")
            raise _coded(400, "noPatternPicked",
                         "No calibration pattern could be picked: the image quality of the "
                         "sampled patterns could not be measured.")
        from backend.api.services.pc_search import MIN_PATTERNS_FIND
        if req.exclude and len(picked.coords) < MIN_PATTERNS_FIND:
            # Other patterns for "Find PC" that are fewer than the search needs: said here, before the
            # list is replaced by a set the search would refuse (pc_search.SearchError tooFewPatterns).
            raise _coded(400, "tooFewPatterns",
                         f"Find PC needs at least {MIN_PATTERNS_FIND} calibration patterns, but "
                         f"only {len(picked.coords)} could be picked once the positions to leave out are "
                         "excluded.", have=len(picked.coords), need=MIN_PATTERNS_FIND)
        # The aperture is the detector's, decided as in /pattern/add from the pattern as recorded
        # (the picked ones have their background removed: no evidence). One pattern with black
        # corners decides; at most the nine picked ones are read, and none once it is known.
        circular = bool(getattr(ctrl, "aperture_circular", False))
        if not circular:
            shape = np.asarray(picked.patterns[0]).shape
            for row, col in picked.coords:
                recorded = _unprocessed_pattern(dataset, row, col, shape)
                if recorded is None:
                    recorded = read(np.array([row]), np.array([col]))[0]
                if _has_black_corners(recorded):
                    circular = True
                    break
        with _state_lock_or_busy("adding or removing calibration patterns"):
            _refuse_while_optimizing("adding or removing calibration patterns")
            if not getattr(ctrl, "aperture_circular", False):
                ctrl.aperture_circular = circular
            if req.replace:
                ctrl.patterns.clear()
            have = {(int(c[0]), int(c[1])) for c, _ in ctrl.patterns}
            added = []
            for (row, col), pattern, quality in zip(picked.coords, picked.patterns, picked.quality):
                if (row, col) in have:
                    continue
                # A standalone copy: it must not change when the viewer's signal is processed later.
                index = ctrl.add_pattern((row, col), np.array(pattern, copy=True))
                added.append({"index": index, "row": row, "col": col,
                              "quality": None if quality is None else float(quality)})
            ctrl.clear_cache()
            sim_cache.clear()
            return {
                "success": True, "n_patterns": len(ctrl.patterns), "patterns": added,
                "prepared": picked.prepared, "background": picked.background,
                "n_sampled": picked.n_sampled, "stride": picked.stride,
                "notes": list(picked.notes),
            }

    try:
        return await asyncio.to_thread(_pick_and_store)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("auto-pick failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pattern/remove")
async def remove_pattern(idx: int = 0):
    """Remove a calibration pattern by index."""
    _refuse_while_optimizing("adding or removing calibration patterns")
    ctrl = _get_controller()
    if idx < 0 or idx >= len(ctrl.patterns):
        raise HTTPException(status_code=400, detail=f"Pattern index {idx} out of range (have {len(ctrl.patterns)})")
    sim_cache = _get_session().sim_cache

    def _remove_locked():
        with _state_lock_or_busy("adding or removing calibration patterns"):
            _refuse_while_optimizing("adding or removing calibration patterns")
            # The list may have changed while this waited for the lock.
            if idx >= len(ctrl.patterns):
                raise HTTPException(
                    status_code=400,
                    detail=f"Pattern index {idx} out of range (have {len(ctrl.patterns)})")
            ctrl.patterns.pop(idx)
            ctrl.clear_cache()
            sim_cache.clear()
            return {"success": True, "n_patterns": len(ctrl.patterns)}

    return await asyncio.to_thread(_remove_locked)


@router.post("/phase/load")
async def load_phase(req: LoadPhaseRequest):
    """Load a crystal phase from a CIF file."""
    ctrl = _get_controller()

    # Off the event loop: the first read of a two-origin CIF decides its origin
    # choice (2-4 s). In a worker thread the controller is changed under
    # _state_lock, as _index_pattern_locked does.
    sim_cache = _get_session().sim_cache

    def _load_phase_locked():
        with _state_lock:
            phase = ctrl.load_phase(req.cif_path)
            sim_cache.clear()           # simulated lines belong to the old phase
            return phase

    try:
        phase = await asyncio.to_thread(_load_phase_locked)
        return {
            "success": True,
            "phase_name": str(phase.name),
            "space_group": str(phase.space_group) if hasattr(phase, 'space_group') else "",
            "lattice": {
                "a": float(phase.structure.lattice.a),
                "b": float(phase.structure.lattice.b),
                "c": float(phase.structure.lattice.c),
            } if hasattr(phase, 'structure') else {},
        }
    except Exception as e:
        logger.exception("Failed to load phase")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/detector/set")
async def set_detector(req: SetDetectorRequest):
    """Set the EBSD detector parameters."""
    ctrl = _get_controller()
    try:
        from kikuchipy.detectors import EBSDDetector

        # The page shows ONE PC: the mean of a PC map, when there is one. An Apply
        # whose PC fields still hold that value (it was pressed for a tilt, the
        # binning, the pixel size) changes nothing about the PC and must not turn
        # the map into that one value; only a PC the user actually changed does.
        pc_was_map = pc_is_per_pixel(ctrl.detector) if ctrl.detector is not None else False
        keep_map = False
        if pc_was_map:
            from kikuchipy.detectors import EBSDDetector as _D
            applied_bruker = np.asarray(_D(shape=tuple(req.shape), pc=req.pc,
                                           convention=req.convention).pc, dtype=float).ravel()[:3]
            shown = np.asarray(ctrl.detector.pc, dtype=float).reshape(-1, 3).mean(axis=0)
            keep_map = bool(np.allclose(applied_bruker, shown, atol=1e-9))

        # Build keyword arguments — only pass what EBSDDetector accepts
        kwargs = {
            "shape": tuple(req.shape),
            "pc": req.pc,
            "sample_tilt": req.sample_tilt,
            "tilt": req.camera_tilt,
            "binning": req.binning,
            "azimuthal": req.azimuthal,
        }

        convention = req.convention
        if keep_map:
            kwargs["pc"] = np.asarray(ctrl.detector.pc, dtype=float)   # the map, Bruker
            convention = "bruker"

        # convention parameter name varies by kikuchipy version
        try:
            det = EBSDDetector(**kwargs, convention=convention)
        except TypeError:
            try:
                det = EBSDDetector(**kwargs, pc_convention=req.convention)
            except TypeError:
                det = EBSDDetector(**kwargs)

        # The physical pixel pitch of a STORED pattern pixel (um) is what the page
        # shows as the pixel size; kikuchipy's px_size is the UNBINNED one, so with
        # a binning b the pitch is px_size * b. (This used to set an attribute
        # kikuchipy never reads, `pixel_size`, so the value changed nothing.)
        if req.pixel_size is not None:
            det.px_size = float(req.pixel_size) / max(int(req.binning), 1)

        ctrl.attach_detector(det)
        _sim_cache.clear()              # bands and CI were simulated for the old detector

        # "Apply" defines the detector: its PC and tilts go to the store as well, or
        # indexing would keep reading the old ones. A grid-calibrated PC map of the
        # dataset is replaced by this one PC, as the controller's was just now.
        pc_map_action = "none"
        try:
            from backend.api.routes.ebsd_viewer import _active_dataset
            applied_pc = [float(v) for v in np.asarray(det.pc, dtype=float).reshape(-1, 3).mean(axis=0)]
            entry = calibration_store.get_entry(_active_dataset)
            # Apply with the PC the store already has (to set a tilt, say) is not
            # supplying a PC: a placeholder stays "missing", a grid map stays, and a
            # header PC does not become "manual".
            unchanged = (entry is not None
                         and np.allclose(entry.pc_single, applied_pc, atol=1e-9))
            store_action = None if unchanged else _writeback_refined_pc(
                _active_dataset, applied_pc, source="manual", on_refined_map="replace")
            _writeback_geometry(_active_dataset, sample_tilt=req.sample_tilt,
                                tilt=req.camera_tilt, azimuthal=req.azimuthal)
            controller_action = "single" if keep_map or not pc_was_map else "replaced"
            pc_map_action = _pc_map_action(controller_action, store_action)
        except Exception:
            logger.warning("detector store write-back skipped", exc_info=True)

        # Build repr string for display
        from backend.api.routes.ebsd_viewer import _build_detector_dict, _build_axes_repr, _get_active_signal
        det_info = _build_detector_dict(det)
        axes_repr = ""
        try:
            sig = _get_active_signal()
            if sig is not None:
                axes_repr = _build_axes_repr(sig)
        except Exception:
            pass

        return {
            "success": True,
            "pc": req.pc,
            "shape": req.shape,
            "binning": req.binning,
            "sample_tilt": req.sample_tilt,
            "camera_tilt": req.camera_tilt,
            "azimuthal": req.azimuthal,
            "pixel_size": req.pixel_size,
            "convention": req.convention,
            "repr": det_info.get("repr", ""),
            "axes_repr": axes_repr,
            "pc_map_action": pc_map_action,
        }
    except Exception as e:
        logger.exception("Failed to set detector")
        raise HTTPException(status_code=400, detail=str(e))


def _pc_report(ctrl, coords=None) -> dict:
    """What the page may call "the PC" of the controller's detector.

    Three numbers, always: the PC of the detector when it has one; for a
    per-pixel detector (after a grid calibration) the PC of the pixel ``coords``
    when it lies on the map, else the MEAN of the map. ``pc_basis`` says which
    ("single" | "pixel" | "mean"); a per-pixel detector also reports the map
    shape and the spread of each PC component (max - min), so the page can say
    what it is looking at instead of presenting pixel (0, 0) as the PC. The map
    itself (rows x cols x 3 floats) is never sent.

    ``det`` is the single-PC detector the numbers describe (the controller's own
    when it has one PC), for conventions and geometry.
    """
    from pc_controller import pc_is_per_pixel
    det = ctrl.detector
    if not pc_is_per_pixel(det):
        pc = [float(v) for v in np.asarray(det.pc, dtype=float).reshape(-1, 3)[0]]
        return {"pc": pc, "pc_basis": "single", "pc_per_pixel": False,
                "pc_map_shape": None, "pc_span": None, "det": det}
    arr = np.asarray(det.pc, dtype=float)
    flat = arr.reshape(-1, 3)
    on_map = (coords is not None and arr.ndim == 3
              and 0 <= int(coords[0]) < arr.shape[0] and 0 <= int(coords[1]) < arr.shape[1])
    if on_map:
        pc = [float(v) for v in arr[int(coords[0]), int(coords[1])]]
    else:
        pc = [float(v) for v in flat.mean(axis=0)]
    from pc_controller import single_pc_detector
    return {
        "pc": pc,
        "pc_basis": "pixel" if on_map else "mean",
        "pc_per_pixel": True,
        "pc_map_shape": [int(v) for v in arr.shape[:-1]] if arr.ndim == 3 else [int(flat.shape[0])],
        "pc_span": [float(v) for v in (flat.max(axis=0) - flat.min(axis=0))],
        "det": single_pc_detector(det, pc),
    }


@router.get("/detector/info")
async def detector_info(row: Optional[int] = None, col: Optional[int] = None):
    """Return full detector info including binning, pixel size, and all geometry params.

    ``pc`` is always three numbers (see :func:`_pc_report`): with a per-pixel PC
    map the PC at scan position ``(row, col)`` if given, else the map mean.
    """
    ctrl = _get_controller()
    det = ctrl.detector
    if det is None:
        return {"has_detector": False}

    report = _pc_report(ctrl, None if row is None or col is None else (row, col))
    det = report.pop("det")
    info: dict = {"has_detector": True, **report}

    # Core geometry — coerce to native Python ints/floats so FastAPI's
    # jsonable_encoder can serialize. kikuchipy detector .shape is a tuple
    # of numpy.int32 and .pc is a numpy.float64 ndarray; both crash the
    # encoder if passed through as-is (see commit 3a305a3 for the same
    # bug class in CalibrationEntry.to_dict).
    try:
        info["shape"] = [int(v) for v in det.shape]
    except Exception:
        info["shape"] = []

    try:
        info["sample_tilt"] = float(det.sample_tilt)
    except Exception:
        info["sample_tilt"] = 70.0

    try:
        info["camera_tilt"] = float(det.tilt)
    except Exception:
        info["camera_tilt"] = 0.0

    # The geometry the app computes with, not the camera's own binning: the
    # displayed detector width must match what indexing and the forward
    # renderer use. The file's value is reported separately.
    from backend.api.routes.ebsd_viewer import _display_binning_and_pixel_size
    info["binning"], info["file_binning"], info["pixel_size"] = (
        _display_binning_and_pixel_size(det))

    try:
        info["azimuthal"] = float(det.azimuthal)
    except Exception:
        info["azimuthal"] = 0.0

    # PC in various conventions — same numpy-coercion concern as above
    try:
        info["pc_tsl"] = [float(v) for v in det.pc_tsl().flatten()]
    except Exception:
        pass

    try:
        info["pc_bruker"] = [float(v) for v in det.pc_bruker().flatten()]
    except Exception:
        pass

    try:
        info["pc_oxford"] = [float(v) for v in det.pc_oxford().flatten()]
    except Exception:
        pass

    return info


class ConvertPcRequest(BaseModel):
    pc: List[float]
    source: str
    target: str
    shape: List[int]


@router.post("/convert")
async def convert_pc(req: ConvertPcRequest):
    """A pattern centre in another vendor's convention (Bruker, EDAX/TSL, Oxford).

    The page keeps every PC in Bruker's convention; this is how it shows and takes the numbers
    of the other two. Both directions are kikuchipy's own (``services/pc_convention.py``), and
    the pattern ``shape`` (rows, columns) is required: the conventions differ by it.
    """
    from backend.api.problem import problem
    from backend.api.services import pc_convention
    try:
        out = pc_convention.convert(req.pc, req.source, req.target, req.shape)
    except pc_convention.ConventionError as exc:
        raise problem(400, exc.code, str(exc))
    return {"pc": [float(v) for v in out], "source": pc_convention.canonical(req.source),
            "target": pc_convention.canonical(req.target)}


class IndexPatternRequest(BaseModel):
    row: int
    col: int


@router.post("/index-pattern")
async def index_pattern(req: IndexPatternRequest):
    """Run Hough indexing on a single pattern and return CI + Kikuchi line segments.

    Uses _index_and_simulate (same code path as Index All / update_pc) to ensure
    identical results.  Temporarily adds the pattern to ctrl.patterns if needed.
    """
    from backend.api.routes.ebsd_viewer import _get_active_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available — load an EBSD file first")
    if ctrl.phase is None:
        raise HTTPException(status_code=400, detail="No phase loaded — load a CIF file first")

    # Bounds-check BEFORE hitting signal.data[row, col] — numpy would
    # silently wrap a negative row into the far end of the array and
    # index the wrong pattern. Match the guards in h5_viewer.py.
    nav_shape = signal.axes_manager.navigation_shape
    n_cols = int(nav_shape[0]) if len(nav_shape) >= 1 else 1
    n_rows = int(nav_shape[1]) if len(nav_shape) >= 2 else 1
    if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Position ({req.row},{req.col}) out of bounds for {n_rows}x{n_cols} grid",
        )

    # BUG-G mitigation: the lazy pyebsdindex build inside _index_and_simulate
    # (reflectors -> indexer) and the add/pop of ctrl.patterns / _sim_cache are
    # not thread-safe. React Strict-Mode in dev fires this endpoint twice on
    # mount/navigate, and two concurrent cold-start calls could both enter the
    # indexer-build path and corrupt ctrl.indexer. Serialize the whole block
    # under the existing _state_lock so the second caller waits for the
    # already-in-progress build instead of racing.
    def _index_pattern_locked():
        # The value range of an int16 scan is read once per signal; on a big lazy scan the first click
        # spends minutes in that pass. Take it before the lock, so that the other routes that wait
        # for _state_lock are not held up by it (the pattern read below then finds it kept).
        from backend.api.services.load_choices import scan_value_range
        scan_value_range(signal)
        with _state_lock:
            # Find if this (row, col) is already a calibration pattern
            pat_idx = None
            for i, (coords, _) in enumerate(ctrl.patterns):
                if coords[0] == req.row and coords[1] == req.col:
                    pat_idx = i
                    break

            if pat_idx is None:
                # Pattern not in calibration list — fetch from current active signal
                from backend.api.services.load_choices import signal_pattern
                pattern = signal_pattern(signal, req.row, req.col)
                pat_idx = ctrl.add_pattern((req.row, req.col), pattern)
                temp_added = True
            else:
                # Use the STORED pattern (from add_pattern time, possibly processed)
                # Do NOT re-fetch from active signal — user may have switched datasets
                temp_added = False

            # Clear caches so we get a fresh result with current detector
            ctrl.clear_cache()
            _sim_cache.clear()

            result = _index_and_simulate(ctrl, pat_idx)

            # Remove temp pattern to keep calibration list clean
            if temp_added:
                ctrl.patterns.pop(pat_idx)
                _sim_cache.pop(pat_idx, None)
                ctrl.cache.pop(pat_idx, None)
            return result

    try:
        # Hough indexing + Kikuchi simulation is CPU-heavy (and the lazy
        # pyebsdindex indexer build is slow on first call) — run it off the
        # event loop so the whole backend doesn't freeze. _state_lock now
        # serializes inside the worker thread instead of on the loop.
        result = await asyncio.to_thread(_index_pattern_locked)
        return {
            "success": True,
            **result,
        }
    except HoughLibraryTooBig as e:
        # The reflector list needs more memory than may be spent on it: a coded problem
        # (not an English 500) so the page can word it and offer the one-click answer.
        logger.warning("Hough library refused: %s", e)
        raise HTTPException(status_code=400, detail=e.as_problem())
    except ImportError as e:
        # Missing optional dependency (e.g. pyebsdindex)
        logger.exception("Index pattern failed — missing dependency")
        raise HTTPException(
            status_code=503,
            detail=f"Missing dependency for Hough indexing: {e}. "
                   "Ensure pyebsdindex is installed in the backend Python environment.",
        )
    except Exception as e:
        logger.exception("Index pattern failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/index-all")
async def index_all_patterns():
    """Index ALL loaded calibration patterns and return per-pattern CI + segments.

    Mirrors PyQt5 IndexAllWorker: indexes every pattern, caches results,
    returns CI + segments for each so frontend can color-code the pattern list
    and show overlays instantly on navigation.
    """
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available — load an EBSD file first")
    if ctrl.phase is None:
        raise HTTPException(status_code=400, detail="No phase loaded — load a CIF file first")
    if not ctrl.patterns:
        raise HTTPException(status_code=400, detail="No calibration patterns loaded")

    global _sim_cache

    # Serialize the whole batch under _state_lock. Same reasoning as
    # /index-pattern (BUG-G, commit 6a65c69): ctrl.indexer rebuild and
    # _sim_cache / ctrl.cache mutation aren't thread-safe. Without this,
    # a concurrent /index-pattern click mid-batch could corrupt the
    # pyebsdindex indexer state.
    def _index_all_locked():
        results = []
        with _state_lock:
            _sim_cache.clear()
            ctrl.clear_cache()  # Force fresh indexer with current detector

            for idx in range(len(ctrl.patterns)):
                try:
                    result = _index_and_simulate(ctrl, idx)
                    results.append({
                        "index": idx,
                        "row": int(ctrl.patterns[idx][0][0]),
                        "col": int(ctrl.patterns[idx][0][1]),
                        **result,
                    })
                except HoughLibraryTooBig:
                    raise                      # every pattern would be refused: stop, say why
                except Exception as pat_err:
                    logger.warning("Index pattern %d failed: %s", idx, pat_err, exc_info=True)
                    # Retry once — PyEBSDIndex can fail intermittently
                    try:
                        ctrl.indexer = None
                        result = _index_and_simulate(ctrl, idx)
                        results.append({
                            "index": idx,
                            "row": int(ctrl.patterns[idx][0][0]),
                            "col": int(ctrl.patterns[idx][0][1]),
                            **result,
                        })
                    except Exception:
                        results.append({
                            "index": idx,
                            "row": int(ctrl.patterns[idx][0][0]),
                            "col": int(ctrl.patterns[idx][0][1]),
                            "ci": 0.0,
                            "phase_name": "",
                            "segments": [],
                            "n_bands": 0,
                            "error": str(pat_err),
                        })

            # Global CI: the mean over the patterns that were indexed (a pattern
            # that failed twice has no CI and is not counted).
            ci_vals = [r["ci"] for r in results if "error" not in r]
            global_ci = sum(ci_vals) / len(ci_vals) if ci_vals else 0.0

            return {
                "success": True,
                "results": results,
                "global_ci": global_ci,
                "n_indexed": len([r for r in results if "error" not in r]),
            }

    try:
        # Indexing every calibration pattern (Hough + simulation) is heavy;
        # run the locked batch off the event loop so the backend stays
        # responsive to other requests (and the lock serializes off-loop).
        return await asyncio.to_thread(_index_all_locked)
    except HoughLibraryTooBig as e:
        logger.warning("Hough library refused: %s", e)
        raise HTTPException(status_code=400, detail=e.as_problem())
    except Exception as e:
        logger.exception("Index all patterns failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pattern/{idx}/image")
async def get_pattern_image(idx: int):
    """Return the STORED pattern image for a calibration pattern.

    This returns the pattern data that was captured at add_pattern time,
    NOT the current active signal. Critical for DeepCopy/FrameAvg workflows
    where the active signal may have changed.
    """
    ctrl = _get_controller()
    if idx < 0 or idx >= len(ctrl.patterns):
        raise HTTPException(status_code=400, detail=f"Pattern index {idx} out of range")
    coords, pat = ctrl.patterns[idx]
    # One image pixel per pattern pixel. The Kikuchi lines of /index-pattern,
    # /index-all and /pattern/{idx}/result are in detector pixels and the page
    # draws them on this image unscaled, so the image must have the pattern's
    # own size (the matplotlib renderer enlarges it to a figure size, which
    # left the lines in the top-left corner of the picture).
    pat = np.asarray(pat)
    # Every pattern of a circular-aperture detector is shown through the same disc,
    # also one stored after viewer processing (its corners are noise, not signal).
    # Display only: the stored pattern, and so every CI, is left as it was.
    circular = _session_is_circular(ctrl) and pat.ndim == 2
    inside = None
    if circular:
        from tools.pattern_comparison import circular_mask
        inside = circular_mask(pat.shape)
    # Stretched to the full grey range, as the enlarged picture was - over the disc
    # only, so noise outside the aperture does not set the contrast.
    ref = pat[inside] if inside is not None else pat
    lo, hi = float(np.nanmin(ref)), float(np.nanmax(ref))
    shown = (np.zeros(pat.shape, dtype=np.uint8) if not hi > lo else
             (np.clip((pat.astype(np.float64) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))
    if inside is not None:
        shown[~inside] = 0
    image_b64 = array_to_base64_raw(shown)
    return {
        "success": True,
        "image": image_b64,
        "row": coords[0],
        "col": coords[1],
        "shape": [int(v) for v in pat.shape[:2]],
        "circular": bool(circular),
    }


@router.get("/pattern/{idx}/result")
async def get_pattern_result(idx: int):
    """Get cached indexing result for a calibration pattern.

    Returns cached CI + segments if available, or indexes on-demand.
    Mirrors PyQt5 on_pattern_changed() cache-first logic.
    """
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)

    if idx < 0 or idx >= len(ctrl.patterns):
        raise HTTPException(status_code=400, detail=f"Pattern index {idx} out of range")

    # The session's own cache, read once: worker threads clear it concurrently, so
    # "is it there?" and "give it to me" must be one look (.get), not two.
    sim_cache = _get_session().sim_cache
    hit = sim_cache.get(idx)
    if hit is not None:
        return {"success": True, "cached": True, **hit}

    # Not cached — index on demand if we have detector + phase
    if ctrl.detector is None or ctrl.phase is None:
        return {"success": False, "detail": "Detector or phase not set"}

    # Hough indexing (and the indexer build it may need) takes seconds to minutes:
    # off the event loop, under _state_lock like /index-pattern. The result goes to
    # the cache of the session the request came in on, not of a dataset that
    # became active while it ran.
    def _index_locked():
        # A plain wait, not the 2 s "busy" of the short edits: this IS the long work
        # (a cold indexer takes a minute), and a second click must queue, not fail.
        with _state_lock:
            # The list may have changed while this waited for the lock (a Remove),
            # and another request may have indexed the pattern meanwhile.
            if idx >= len(ctrl.patterns):
                raise HTTPException(status_code=400, detail=f"Pattern index {idx} out of range")
            cached = sim_cache.get(idx)
            if cached is not None:
                return cached, True
            return _index_and_simulate(ctrl, idx, sim_cache=sim_cache), False

    try:
        result, was_cached = await asyncio.to_thread(_index_locked)
        return {"success": True, "cached": was_cached, **result}
    except HTTPException:
        raise
    except HoughLibraryTooBig as e:
        logger.warning("Hough library refused: %s", e)
        raise HTTPException(status_code=400, detail=e.as_problem())
    except Exception as e:
        logger.exception("On-demand indexing failed for pattern %d", idx)
        raise HTTPException(status_code=500, detail=str(e))


class UpdatePCRequest(BaseModel):
    pcx: float
    pcy: float
    pcz: float
    current_pattern_idx: Optional[int] = None


@router.post("/detector/update-pc")
async def update_pc(req: UpdatePCRequest):
    """Update detector PC values, clear caches, re-index current pattern.

    Mirrors PyQt5 apply_detector_pc(): spinbox change → update det.pc →
    clear cache → re-index current → re-overlay.
    """
    _refuse_while_optimizing("changing the PC")
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available")

    # The dataset and its sim cache are read here, on the request's own thread: the
    # work below runs in a worker, and the active dataset may change meanwhile.
    dataset = _active_dataset_name()
    sim_cache = _get_session().sim_cache

    # Off the event loop (re-indexing builds the Hough indexer: seconds to minutes)
    # and under _state_lock, like /index-pattern.
    def _update_locked():
        with _state_lock_or_busy("changing the PC"):
            _refuse_while_optimizing("changing the PC")   # one may have started meanwhile
            # Update PC. With a per-pixel PC map (grid calibration) the typed PC is the
            # new MEAN: the whole map moves rigidly (its shape is kept), nothing
            # collapses to one value behind the user's back. The response says so
            # (pc_map_action).
            ctrl_action = ctrl.shift_pc_map((req.pcx, req.pcy, req.pcz))

            # Also write the calibration store (+ parent), so a MANUAL PC reaches the
            # phase test / indexing / export — not just the PC-Refinement controller.
            # (The manual endpoint used to write only ctrl.detector, like
            # /optimize did.) A grid-calibrated map of the active dataset is
            # shifted the same way.
            store_action = None
            written = {}              # {dataset: what happened to its PC map}
            try:
                store_action = _writeback_refined_pc(
                    dataset, [req.pcx, req.pcy, req.pcz], source="manual",
                    on_refined_map="shift", report=written)
            except Exception:
                logger.warning("manual PC store write-back skipped", exc_info=True)

            # Clear caches (like PyQt5: controller.clear_cache() + sim_cache.clear())
            ctrl.clear_cache()
            sim_cache.clear()

            result = {"success": True, "pc": [req.pcx, req.pcy, req.pcz],
                      "pc_map_action": _pc_map_action(ctrl_action, store_action, written),
                      "pc_map_actions": written,
                      # False when the calibration store does not know the dataset (or the write
                      # failed): the PC is on the page only, indexing will not use it
                      "pc_stored": store_action is not None,
                      "previous": _previous_of(dataset)}

            # Re-index current pattern if specified and prerequisites met
            if req.current_pattern_idx is not None and ctrl.phase is not None:
                idx = req.current_pattern_idx
                if 0 <= idx < len(ctrl.patterns):
                    try:
                        sim_result = _index_and_simulate(ctrl, idx, sim_cache=sim_cache)
                        result["ci"] = sim_result["ci"]
                        result["segments"] = sim_result["segments"]
                        result["n_bands"] = sim_result["n_bands"]
                        result["phase_name"] = sim_result["phase_name"]
                        # No "global_ci": only the selected pattern was indexed at the
                        # new PC, and a mean over all patterns needs all of them (Index
                        # All or Global PC Refine give it).
                    except Exception as e:
                        logger.warning("Re-index after PC update failed: %s", e)
                        result["index_error"] = str(e)
            return result

    return await asyncio.to_thread(_update_locked)


class PutBackRequest(BaseModel):
    dataset: Optional[str] = None


def _store_and_name(dataset: str):
    """``(store, name, is_open)``: the open file's store for a name it holds, else the stash of a file that is not
    open (``dataset`` is then the label ``pc_calibration_apply.candidates`` gave that file) and the name of its
    dataset in that stash."""
    if calibration_store.get_entry(dataset) is not None:
        return calibration_store, dataset, True
    from backend.api.services import pc_calibration_apply
    target = pc_calibration_apply.resolve_target(dataset, calibration_store)
    if target is None or not target.stashed:
        return calibration_store, dataset, True
    return target.store, target.name, False


@router.get("/previous")
def previous_pc(dataset: Optional[str] = None):
    """What the PC of ``dataset`` (default: the active one) was before the last PC write, or
    ``previous: null`` when there is nothing to put back for it."""
    name = dataset or _active_dataset_name()
    store, own, _is_open = _store_and_name(name)
    return {"dataset": name, "previous": _previous_of(own, store)}


@router.post("/put-back")
def put_back(req: Optional[PutBackRequest] = None):
    """Undo the last PC write: the PC of every dataset it touched (the dataset, its parent and root,
    the copies that followed) is what it was before, and the PC pages of those follow the store.

    One step: a second call has nothing to put back (409 ``nothingToPutBack``), as has a dataset whose
    step belongs to another file with the same name. Only the calibration store changes; nothing is
    written to the file on disk. Refused (409) while a PC search runs or another PC job holds the
    state.

    A file that is not open (the ``dataset`` is the name ``candidates`` gave it) goes back the same way in the
    stash it is kept in; its PC page, if it has one, follows the store when the file is opened.

    The answer lists ``restored`` (the datasets that went back), ``not_restored`` (copies and crops
    of the same family made after the write: they keep the PC the write gave them) and
    ``pc_per_pixel`` (whether the dataset has a per-pixel map now)."""
    from backend.api.problem import problem
    what = "putting the PC back"
    _refuse_while_optimizing(what)
    asked = (req.dataset if req is not None else None) or _active_dataset_name()
    with _state_lock_or_busy(what):
        _refuse_while_optimizing(what)           # one may have started meanwhile
        store, dataset, is_open = _store_and_name(asked)
        if _step_of(dataset, store) is None:
            raise problem(409, "nothingToPutBack", "There is no earlier PC of this dataset to put back.")
        step_key = _step_key(store, dataset)
        entries = _pc_before[step_key]
        previous = _previous_of(dataset, store)
        restored = store.restore_pc(entries)
        from backend.api.services import pc_calibration_apply
        pc_calibration_apply.restore_search(dataset, getattr(entries, "search", None), store)
        pc_calibration_apply.restore_usage(dataset, getattr(entries, "usage", None), store)
        _pc_before.pop(step_key, None)
        file_key = _current_file_key()
        for name in restored if is_open else []:
            sess = _session_of_this_file(name, file_key)
            if sess is None:
                continue
            _session_follows_store(name, sess, hold_lock=False)
            sess.sim_cache.clear()
            sess.ctrl.clear_cache()              # the per-pattern results and the indexer belong to the old PC
        # A copy or crop made after the write took the new PC from its parent and is in no step: it
        # keeps that PC (source "inherited") while its parent goes back. The caller says so.
        root = store.family_root(dataset)
        not_restored = sorted(
            n for n, e in store.get_all().items()
            if n != root and n not in entries and e.pc_source == "inherited"
            and store.family_root(n) == root)
        has_map = store.get_entry(dataset).pc_map is not None
    return {"dataset": asked, "pc": previous["pc"], "had_pc": previous["had_pc"],
            "restored": restored, "not_restored": not_restored, "pc_per_pixel": has_map}


class UpdateTiltRequest(BaseModel):
    sample_tilt: Optional[float] = None
    detector_tilt: Optional[float] = None
    azimuthal: Optional[float] = None
    current_pattern_idx: Optional[int] = None


@router.post("/detector/update-tilt")
async def update_tilt(req: UpdateTiltRequest):
    """Update detector tilt settings, clear caches, re-index current pattern.

    Mirrors PyQt5 apply_tilt_settings(): tilt change → update det tilts →
    clear cache → re-index current → re-overlay.
    """
    _refuse_while_optimizing("changing the tilts")
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available")

    # Read on the request's own thread: the work below runs in a worker.
    dataset = _active_dataset_name()
    sim_cache = _get_session().sim_cache

    # Off the event loop and under _state_lock, like /index-pattern.
    def _update_locked():
        with _state_lock_or_busy("changing the tilts"):
            _refuse_while_optimizing("changing the tilts")   # one may have started meanwhile
            det = ctrl.detector
            if req.sample_tilt is not None:
                det.sample_tilt = req.sample_tilt
            if req.detector_tilt is not None:
                det.tilt = req.detector_tilt
            if req.azimuthal is not None:
                det.azimuthal = req.azimuthal
            try:
                _writeback_geometry(dataset, sample_tilt=req.sample_tilt,
                                    tilt=req.detector_tilt, azimuthal=req.azimuthal)
            except Exception:
                logger.warning("tilt store write-back skipped", exc_info=True)

            # Clear caches
            ctrl.clear_cache()
            sim_cache.clear()

            result = {
                "success": True,
                "sample_tilt": float(det.sample_tilt),
                "detector_tilt": float(det.tilt),
                "azimuthal": float(det.azimuthal),
            }

            # Re-index current pattern if specified
            if req.current_pattern_idx is not None and ctrl.phase is not None:
                idx = req.current_pattern_idx
                if 0 <= idx < len(ctrl.patterns):
                    try:
                        sim_result = _index_and_simulate(ctrl, idx, sim_cache=sim_cache)
                        result["ci"] = sim_result["ci"]
                        result["segments"] = sim_result["segments"]
                        result["n_bands"] = sim_result["n_bands"]
                        result["phase_name"] = sim_result["phase_name"]
                    except Exception as e:
                        logger.warning("Re-index after tilt update failed: %s", e)
                        result["index_error"] = str(e)
            return result

    return await asyncio.to_thread(_update_locked)


@router.post("/params/update")
async def update_params(req: UpdateParamsRequest):
    """Update indexing parameters and clear caches."""
    ctrl = _get_controller()
    ctrl.update_indexing_params(
        min_d=req.min_d,
        f_threshold=req.f_threshold,
        max_reflectors=req.max_reflectors,
        nBands=req.n_bands,
    )
    global _sim_cache
    _sim_cache.clear()

    result = {"success": True}
    if req.binning is not None and req.detector_width_mm is not None and ctrl.detector is not None:
        if req.binning < 1 or req.detector_width_mm <= 0:
            raise HTTPException(status_code=400, detail="binning must be at least 1 and the "
                                "detector width above 0 mm")
        ncols = int(ctrl.detector.shape[1])
        # width = columns * binning * px_size  (kikuchipy), so px_size is the UNBINNED
        # pixel size in um; it fixes how far a beam step moves the PC (extrapolate_pc).
        ctrl.detector.px_size = req.detector_width_mm * 1000.0 / (ncols * req.binning)
        ctrl.detector.binning = int(req.binning)
        result.update({"px_size_um": float(ctrl.detector.px_size), "binning": int(ctrl.detector.binning)})
    return result


@router.get("/status")
async def pc_status(pattern: Optional[int] = None):
    """Get current PC controller state including per-pattern CI from cache.

    ``pc`` is three numbers (see :func:`_pc_report`): with a per-pixel PC map the
    PC of calibration pattern ``pattern`` if given, else the map mean; the map's
    own numbers are never sent.
    """
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)

    pattern_list = []
    # One look at each: worker threads add, remove and clear these concurrently
    # (a snapshot of the list, and .get instead of "in" followed by "[]").
    sim_cache = _get_session().sim_cache
    for i, (coords, _) in enumerate(list(ctrl.patterns)):
        entry = {"index": i, "row": int(coords[0]), "col": int(coords[1])}
        # Include CI from sim_cache if available
        hit = sim_cache.get(i) or ctrl.cache.get(i)
        if hit is not None:
            entry["ci"] = hit.get("ci")
        pattern_list.append(entry)

    # Global CI: a mean over ALL patterns, only when every one has a CI.
    global_ci = _global_ci([p.get("ci") for p in pattern_list])

    from backend.api.routes.ebsd_viewer import _active_dataset
    from backend.api.services.pc_in_use import pc_in_use_of

    def read_pc_state():
        """The PC the page shows, where the dataset's PC comes from and whether it still runs on a calibration, in
        one go: separate looks (one on the loop, one in a worker) would let a PC written in between leave the header
        with numbers that do not belong together. ``pc_in_use`` reads the calibration files (under a lock, with
        retries) and may hash a data file: off the event loop."""
        info = {"pc": [], "pc_basis": None, "pc_per_pixel": False,
                "pc_map_shape": None, "pc_span": None}
        if ctrl.detector is not None:
            where = (ctrl.patterns[pattern][0]
                     if pattern is not None and 0 <= pattern < len(ctrl.patterns) else None)
            info = _pc_report(ctrl, where)
            info.pop("det")
        # Where the active dataset's PC comes from ("header", "manual", "refined", ...;
        # "missing" = the file carries none and nobody set one: the page says so).
        entry = calibration_store.get_entry(_active_dataset) if _active_dataset else None
        return info, entry, pc_in_use_of(_active_dataset, calibration_store)

    pc_info, _entry, pc_in_use = await asyncio.to_thread(read_pc_state)

    return {
        "has_detector": ctrl.detector is not None,
        "has_phase": ctrl.phase is not None,
        "phase_name": str(ctrl.phase.name) if ctrl.phase else None,
        "phase_names": ctrl.phase_names(),
        "has_indexer": ctrl.indexer is not None,
        "n_patterns": len(ctrl.patterns),
        "pc_source": _entry.pc_source if _entry is not None else None,
        "pc_origin": _entry.pc_origin if _entry is not None else None,
        # the PC the dataset is using now and where it came from (kind), for the header of the page
        "pc_in_use": pc_in_use,
        **pc_info,
        "patterns": pattern_list,
        "global_ci": global_ci,
    }


def _phase_path_of(phase):
    """The file a loaded phase came from (None when unknown)."""
    ctrl = _get_controller()
    return getattr(ctrl, "phase_paths", {}).get(str(phase.name))


def _coded(status, code, message, **params):
    return HTTPException(status_code=status, detail={"code": code, "message": message,
                                                     "params": params})


#: How long a page edit waits for _state_lock before it answers "busy". An Index All
#: holds the lock for the whole batch (minutes) without being a refine; an Add from
#: the viewer, a Remove or a typed PC then hung silently up to the client's timeout.
_BUSY_WAIT_S = 2.0


@contextlib.contextmanager
def _state_lock_or_busy(what: str):
    """Hold _state_lock for a short page edit, or answer 409 pc_job_running when a
    long job (Index All, the re-index at the end of a refine) holds it."""
    if not _state_lock.acquire(timeout=_BUSY_WAIT_S):
        raise _coded(409, "pc_job_running",
                     f"A PC job is running. Wait for it to finish before {what}.",
                     action=what)
    try:
        yield
    finally:
        _state_lock.release()


def _refuse_while_optimizing(what: str) -> None:
    """409 while a PC refine or grid calibration runs: it iterates the calibration
    patterns and works on the controller's detector for minutes, and a pattern added
    or removed, or a PC or tilt typed, meanwhile shifts the indices under it."""
    if _optimization_active:
        raise _coded(409, "pc_job_running",
                     f"A PC optimization is running. Wait for it to finish before {what}.",
                     action=what)


def _check_cif_file(path):
    """Refuse a file too large to be a CIF before anything reads it."""
    from pathlib import Path
    from backend.api.services.phase_path import PhasePathError, check_cif_size
    try:
        check_cif_size(Path(str(path)))
    except PhasePathError as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail())


def _phase_summary(phase) -> dict:
    """The part of a phase the page shows: name, space group, lattice."""
    out = {"name": str(phase.name),
           "space_group": str(phase.space_group) if hasattr(phase, 'space_group') else ""}
    out["path"] = _phase_path_of(phase)
    # What identifies the file (a link on the way does not make it another one).
    out["real_path"] = os.path.realpath(out["path"]) if out["path"] else None
    try:
        out["lattice"] = {
            "a": float(phase.structure.lattice.a),
            "b": float(phase.structure.lattice.b),
            "c": float(phase.structure.lattice.c),
        }
    except Exception:
        out["lattice"] = {}
    return out


def _loaded_phases(ctrl) -> list:
    if ctrl.phase_list is None:
        return []
    return [_phase_summary(ctrl.phase_list[pid]) for pid in ctrl.phase_list.ids]


class RemovePhaseRequest(BaseModel):
    phase_name: str


@router.post("/phase/add")
async def add_phase(req: LoadPhaseRequest):
    """Add a crystal phase (CIF) to the ones already loaded.

    PC refinement can use several phases at once (e.g. austenite and ferrite of
    a duplex steel): every calibration pattern is Hough-indexed against all of
    them and the best-fitting phase wins, per pattern. With nothing loaded this
    is the same as ``/phase/load``.
    """
    if _optimization_active:
        raise _coded(409, "optimization_running",
                     "A PC optimization is running. Wait for it to finish "
                     "before changing the phases.")
    ctrl = _get_controller()

    # Off the event loop (reading the CIF and its reflectors takes a moment) and
    # under the state lock, as the other controller-changing endpoints do.
    def _add_locked():
        with _state_lock:
            phase = ctrl.add_phase(req.cif_path)
            _sim_cache.clear()          # simulated lines belong to the old phase set
            return phase

    try:
        phase = await asyncio.to_thread(_add_locked)
    except PhaseSetError as e:
        raise HTTPException(status_code=400, detail=e.as_detail())
    except Exception as e:
        logger.exception("Failed to add phase")
        raise _coded(400, "phase_unreadable", str(e), reason=str(e)[:300])
    return {
        "success": True,
        "phase_name": str(phase.name),
        **{k: v for k, v in _phase_summary(phase).items() if k != "name"},
        "phases": _loaded_phases(ctrl),
        "n_phases": len(_loaded_phases(ctrl)),
    }


@router.post("/phase/remove")
async def remove_phase(req: RemovePhaseRequest):
    """Remove one loaded phase by name; the rest stay loaded."""
    if _optimization_active:
        raise _coded(409, "optimization_running",
                     "A PC optimization is running. Wait for it to finish "
                     "before changing the phases.")
    ctrl = _get_controller()

    def _remove_locked():
        with _state_lock:
            ctrl.remove_phase(req.phase_name)
            _sim_cache.clear()          # simulated lines belong to the old phase set

    try:
        await asyncio.to_thread(_remove_locked)
    except KeyError:
        raise _coded(404, "phase_not_loaded", f"Phase '{req.phase_name}' is not loaded.",
                     name=req.phase_name)
    except Exception as e:
        logger.exception("Failed to remove phase")
        raise _coded(400, "phase_unreadable", str(e), reason=str(e)[:300])
    phases = _loaded_phases(ctrl)
    return {"success": True, "phases": phases, "n_phases": len(phases),
            "phase_name": phases[0]["name"] if phases else None}


# ---------------------------------------------------------------------------
# Reflector families of the loaded phases
#
# The same four calls as `/api/indexing/hough/reflectors`, for a phase that is
# already loaded here (named, since this page has no CIF path to give). Both
# pages read and write one registry (`hough_reflectors`), so what is chosen here
# is what a run on the Indexing page uses, and the other way round.
# ---------------------------------------------------------------------------

class PhaseReflectorChange(BaseModel):
    phase_name: str = ""
    spec: Optional[dict] = None
    extended: bool = False


class PhaseReflectorValidate(BaseModel):
    phase_name: str = ""
    hkl: Union[str, List[float]]
    spec: Optional[dict] = None


def _loaded_phase(ctrl, phase_name):
    """The loaded phase called ``phase_name`` (the only one when blank)."""
    from backend.api.services.hough_reflector_service import ReflectorError
    if ctrl.phase_list is None:
        raise ReflectorError("no_phase", "No phase is loaded.", status=404)
    phases = [ctrl.phase_list[pid] for pid in ctrl.phase_list.ids]
    if not phase_name:
        if len(phases) == 1:
            return phases[0]
        raise ReflectorError("phase_name_required",
                             "Several phases are loaded; say which one.")
    for p in phases:
        if str(p.name) == str(phase_name):
            return p
    raise ReflectorError("no_phase", f"Phase '{phase_name}' is not loaded.",
                         status=404, phase=str(phase_name))


def _reflector_http(exc):
    from backend.api.routes.hough_reflectors import raise_http
    raise_http(exc)


@router.get("/phase/reflectors")
async def pc_phase_reflectors(phase_name: str = "", extended: bool = False):
    """Family table of a loaded phase (see ``GET /api/indexing/hough/reflectors``)."""
    from backend.api.services import hough_reflector_service as svc
    ctrl = _get_controller()

    def work():
        phase = _loaded_phase(ctrl, phase_name)
        return svc.table(phase, phase.name, extended)

    try:
        return await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        _reflector_http(exc)


@router.put("/phase/reflectors")
async def pc_put_phase_reflectors(req: PhaseReflectorChange):
    """Change which families a loaded phase is indexed with (null = default).

    The next indexing step on this page rebuilds the indexer; the simulated
    bands are drawn from the same list.
    """
    from backend.api.services import hough_reflector_service as svc
    if _optimization_active:
        raise HTTPException(
            status_code=409,
            detail="A PC optimization is running. Wait for it to finish "
                   "before changing the reflectors.",
        )
    ctrl = _get_controller()

    def work():
        phase = _loaded_phase(ctrl, req.phase_name)
        return svc.change(phase, phase.name, req.spec, req.extended)

    try:
        out = await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        _reflector_http(exc)
    _get_controller()     # notice the change now: indexer and simulated bands are dropped
    return out


@router.post("/phase/reflectors/fit")
async def pc_fit_phase_reflectors(req: PhaseReflectorChange):
    """The one-click answer to a refusal (see ``POST /api/indexing/hough/reflectors/fit``)."""
    from backend.api.services import hough_reflector_service as svc
    if _optimization_active:
        raise HTTPException(
            status_code=409,
            detail="A PC optimization is running. Wait for it to finish "
                   "before changing the reflectors.",
        )
    ctrl = _get_controller()

    def work():
        phase = _loaded_phase(ctrl, req.phase_name)
        return svc.fit(phase, phase.name, req.extended)

    try:
        out = await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        _reflector_http(exc)
    _get_controller()     # notice the change now: indexer and simulated bands are dropped
    return out


@router.post("/phase/reflectors/validate")
async def pc_validate_phase_reflector(req: PhaseReflectorValidate):
    from backend.api.services import hough_reflector_service as svc
    ctrl = _get_controller()

    def work():
        phase = _loaded_phase(ctrl, req.phase_name)
        return svc.check_family(phase, phase.name, req.hkl, req.spec)

    try:
        return await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        _reflector_http(exc)


@router.get("/phase/reflectors/cost")
async def pc_phase_reflector_cost(phase_name: str = "", n_bands: int = 12):
    from backend.api.services import hough_reflector_service as svc
    ctrl = _get_controller()

    def work():
        phase = _loaded_phase(ctrl, phase_name)
        return svc.cost(phase, phase.name, n_bands)

    try:
        return await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        _reflector_http(exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reflector cost failed", exc_info=True)
        code = getattr(exc, "code", "cost_failed")
        raise HTTPException(status_code=400, detail={
            "code": code, "message": str(exc), "params": getattr(exc, "params", {})})


#: Background tasks whose Cancel was pressed. A PC search is one call over all the calibration
#: patterns: it ends at the search's next progress call (see ``_progress`` in
#: ``_run_optimization``) or, failing that, before the run writes anything. The grid calibration
#: stops between two grid points. A cancelled run leaves the PC, the store and the caches as
#: they were.
_cancel_requested: set = set()


class _RunCancelled(Exception):
    """Raised inside a background run when its Cancel was pressed."""


class _FileChanged(Exception):
    """Raised inside a background run whose dataset name now belongs to another file."""


def _store_file_state(dataset):
    """``(has an entry, its file_key)`` of ``dataset`` in the calibration store. Two files of the same name in
    two folders share one dataset name; the key is what tells them apart."""
    entry = calibration_store.get_entry(dataset) if dataset else None
    return (entry is not None, getattr(entry, "file_key", None))


def _check_same_file(dataset, started_on) -> None:
    """Raise ``_FileChanged`` when the store entry of ``dataset`` is no longer the one the run started on (another
    file was loaded under the same name, or the entry is gone). Asked under the lock, just before a write."""
    if _store_file_state(dataset) != started_on:
        raise _FileChanged(dataset)


def _fail_task_file_changed(task_id, dataset) -> None:
    """End a background run whose file was replaced: nothing was written, the page says so by the code."""
    logger.warning("PC background task %s: the file of %r was replaced while it ran; nothing was written",
                   task_id, dataset)
    task = _optimization_tasks[task_id]
    task["status"] = "failed"
    task["code"] = "fileChanged"
    task["result"] = None
    task["error"] = "Another file was opened during the search. Nothing was written."


def _check_cancelled(task_id: str) -> None:
    if task_id in _cancel_requested:
        raise _RunCancelled()


def _mark_cancelled(task_id: str) -> None:
    _cancel_requested.discard(task_id)
    task = _optimization_tasks.get(task_id)
    if task is not None:
        task["status"] = "cancelled"
        task["error"] = None
        task["result"] = None
    logger.info("PC background task %s cancelled; nothing was written", task_id)


def _start_pc(ctrl, coords, n_patterns):
    """The PC a refine starts at: the PC the detector has at the calibration patterns' pixels
    (their mean when the detector holds a per-pixel map), else the detector's one PC."""
    if coords:
        pcs = [ctrl.pc_at(c) for c in list(coords)[:n_patterns]]
    else:
        pcs = [ctrl.pc_at(None)]
    return np.mean(np.asarray(pcs, dtype=float), axis=0)


def _file_pc_state(dataset):
    """``(pc, missing)``: the PC the FILE itself gave (``None`` when it gave none) and whether the
    store says the file carries no PC at all. A dataset the store does not know is not "missing"."""
    entry = calibration_store.get_entry(dataset) if dataset else None
    if entry is None:
        return None, False
    if entry.pc_source == "missing":
        return None, True
    file_pc = getattr(entry, "file_pc", None)
    return (None if file_pc is None else np.asarray(file_pc, dtype=float).reshape(3)), False


def _calibration_ref_now(dataset):
    """``{"id", "name", "version"}`` of the calibration version the dataset runs on AT THIS MOMENT (its PC still is
    that version's), or ``None``. Asked just before a run writes its PC: only a dataset that was on a calibration
    when the run started leaves it by this run. Never raises."""
    try:
        from backend.api.services import pc_calibration_apply
        return pc_calibration_apply.current_ref(dataset, calibration_store)
    except Exception:  # noqa: BLE001
        logger.warning("could not tell which calibration the dataset is on", exc_info=True)
        return None


def _calibration_after_run(dataset, report, was_on=None):
    """Remember how the dataset's PC was found (a calibration saved from it says so) and report which
    calibration it was on and who else is still on it. Never raises: this is information, and the PC
    is already written when it is asked.

    ``was_on`` is ``_calibration_ref_now`` as it was BEFORE the run wrote its PC. A dataset that was not on a
    calibration then (it left it earlier: a PC typed or searched since) did not leave it by this run, so the block
    is empty: no question, however many files are still on that calibration."""
    empty = {"calibration": None, "diverged": False, "followers": []}
    try:
        from backend.api.services import pc_calibration_apply
    except Exception:  # noqa: BLE001
        logger.warning("could not tell which calibration the dataset was on", exc_info=True)
        return empty
    try:
        pc_calibration_apply.note_search(dataset, report.to_dict(), calibration_store)
    except Exception:  # noqa: BLE001
        logger.warning("could not remember the search of %r", dataset, exc_info=True)
    if was_on is None:
        return empty
    try:
        return pc_calibration_apply.followers(dataset, calibration_store)
    except Exception:  # noqa: BLE001
        logger.warning("could not tell which calibration the dataset was on", exc_info=True)
        return empty


def _run_optimization(task_id: str, patterns_data, method: str, search_limit: float,
                      coords=None, dataset=None, ctrl=None, sim_cache=None, mode: str = "refine",
                      started_on=None):
    """Background task for PC optimization.

    Thread-safe: acquires _state_lock around all shared state modifications.

    ``dataset`` and ``ctrl`` are the dataset the run was started on and its
    controller. The run takes minutes; the active dataset may be another one by
    the time it writes its result, and it must land on the one it was run for.

    ``coords`` are the ``(row, col)`` scan positions of ``patterns_data``. A refine starts at
    the PC the detector has at those pixels (their mean when it holds a per-pixel map).

    ``mode`` is "refine" ("Global PC Refine": starts at the PC the page holds) or "find" ("PC
    finden": needs no start, searches a wide box). Both minimise ONE cost over all the calibration
    patterns together (``services/pc_search``); what each pattern prefers on its own is kept only to
    show how far they disagree (``pc_values``, ``search.spread``), it is not averaged into the PC.

    ``started_on`` is ``_store_file_state(dataset)`` as the route saw it when it started the run: the run
    writes nothing, and fails with the code ``fileChanged``, when the dataset name belongs to another file
    by the time it writes.
    """
    global _optimization_active
    if dataset is None:
        from backend.api.routes.ebsd_viewer import _active_dataset as dataset
    if started_on is None:
        started_on = _store_file_state(dataset)
    if ctrl is None:
        ctrl = _get_controller()
    # The route hands over the cache of the session it started on. Looked up here
    # instead, a file switch between the reply and the thread start parked that
    # session, and the fallback below wrote into the now-active dataset's cache.
    if sim_cache is not None:
        job_sim_cache = sim_cache
    else:
        sess = _sessions.get(dataset)
        job_sim_cache = sess.sim_cache if sess is not None else _sim_cache
    try:
        from backend.api.services import pc_search

        with _state_lock:
            ctrl.reflector_specs_changed()
            if ctrl.indexer is None:
                if ctrl.reflectors is None:
                    from hough_reflectors import prepare_reflectors
                    ctrl.reflectors = prepare_reflectors(ctrl.phase_list)
                from ebsd_utils import create_indexer
                # The optimiser sets its own trial PCs; a per-pixel detector is
                # reduced to one PC for the indexer (see PCController).
                ctrl.indexer = create_indexer(
                    ctrl.detector_at(coords[0] if coords else None), ctrl.phase_list,
                    ctrl.reflectors, nBands=ctrl.nBands,
                )
            # Snapshot the indexer for this run
            indexer = ctrl.indexer

        # What the page held before this run: where a refine starts, and what the drift warning
        # compares with. A file that carries no PC has none to compare with: kikuchipy's
        # placeholder (0.5, 0.5, 0.5) is not a reference, so no drift is reported against it.
        # A find does not start anywhere.
        file_pc, pc_missing = _file_pc_state(dataset)
        start_pc = _start_pc(ctrl, coords, len(patterns_data))
        _ref_pc = None if (pc_missing or mode == "find") else start_pc
        try:
            _ph, _pw = np.asarray(patterns_data[0]).shape[-2:]
        except Exception:
            _ph = _pw = None

        def _progress(fraction):
            # Raises out of the search when Cancel was pressed: the search has no steps of its own to
            # stop between, its progress calls are where a run can end.
            _check_cancelled(task_id)
            _optimization_tasks[task_id]["progress"] = float(fraction)

        _check_cancelled(task_id)
        report = pc_search.run_global(
            indexer, patterns_data, mode=mode, method=method, search_limit=search_limit,
            start=None if mode == "find" else start_pc, progress=_progress)
        # what each calibration pattern prefers on its own: kept to show how far they disagree
        results = ([[float(v) for v in row] for row in report.per_pattern]
                   if report.per_pattern is not None else [])
        mean_pc = [float(v) for v in report.pc]       # the minimiser of the joint cost (not a mean)

        # --- Warnings -------------------------------------------------------------
        # Each warning travels as a stable code plus its numbers, so the page can say it in the
        # reader's language; the English sentence stays as the fallback and as what goes into the log.
        _PC_DRIFT_TOL = 0.05
        _warns = []
        _warn_codes = []
        _pc_dev = (None if _ref_pc is None
                   else float(np.max(np.abs(np.asarray(mean_pc, dtype=float) - _ref_pc))))
        if _pc_dev is not None and _pc_dev > _PC_DRIFT_TOL:
            _refined_txt = ", ".join(f"{float(v):.3f}" for v in mean_pc)
            _start_txt = ", ".join(f"{float(v):.3f}" for v in _ref_pc)
            _warns.append(
                f"Refined PC moved {_pc_dev:.3f} from the starting PC "
                f"(refined {[round(float(v), 3) for v in mean_pc]} vs start "
                f"{[round(float(v), 3) for v in _ref_pc]}). On low-quality / low-res "
                "patterns this is usually refinement noise — verify before trusting it.")
            _warn_codes.append({"code": "pcDrift",
                                # Raw number, locale-formatted on the page.
                                "params": {"deviation": round(_pc_dev, 3),
                                           "refined": _refined_txt,
                                           "start": _start_txt}})
        if mode == "refine" and pc_missing:
            _warns.append(
                "This file carries no pattern centre, so the refine started at kikuchipy's "
                "placeholder (0.5, 0.5, 0.5). \"Find PC\" searches without a start and is the "
                "better first step for such a file.")
            _warn_codes.append({"code": "refineFromPlaceholder", "params": {}})
        # While the search has no threshold its state is "notJudged" (pc_search.CONTRAST_MIN is None):
        # it fires no note about clarity on every result. A number set there brings the note back.
        _c = report.confidence
        if _c.get("state") == "unclear":
            _warns.append(
                "The calibration patterns do not fix this PC clearly: the cost at the result is only "
                f"{float(_c.get('contrast') or 0.0) * 100:.0f}% below the typical cost of plausible "
                "PCs. Treat it as approximate, check it against a known PC, or add patterns with "
                "clear bands from other parts of the scan.")
            _warn_codes.append({"code": "pcUnclear", "params": {
                "contrast": round(float(_c.get("contrast") or 0.0), 2)}})
        if "notAMinimum" in (_c.get("reasons") or []):
            _warns.append(
                "A point right next to the result has a lower cost, so the search did not end in a "
                "minimum of the cost. Treat this PC as approximate.")
            _warn_codes.append({"code": "pcNotAMinimum", "params": {}})
        if report.on_edge:
            _axes = ", ".join(report.on_edge)
            _warns.append(
                f"The result lies at the edge of the search box in {_axes}: the PC may lie "
                "outside it, or the patterns may not constrain it there.")
            _warn_codes.append({"code": "pcOnBoxEdge", "params": {"axes": _axes}})
        if report.second_position:
            _second_txt = ", ".join(f"{float(v):.3f}" for v in report.second_position["pc"])
            _second_cost = round(float(report.second_position["cost"]), 3)
            _best_cost = round(float(report.cost), 3)
            _warns.append(
                "Two about equally good positions. These patterns may not fix the PC well. Check the bands.")
            _warn_codes.append({"code": "pcSecondPosition",
                                "params": {"pc": _second_txt, "cost": _second_cost, "best_cost": _best_cost}})
        _pc_warning = "  ".join(_warns) if _warns else None
        if _pc_warning:
            logger.warning("[pc-optimize] %s", _pc_warning)

        # Apply optimized PC to detector and re-index — all under lock
        with _state_lock:
            # Last point at which Cancel still means "nothing changed".
            _check_cancelled(task_id)
            _check_same_file(dataset, started_on)
            # The calibration the dataset is on, asked BEFORE the PC is written: the write is what ends it.
            was_on = _calibration_ref_now(dataset)
            # Apply optimized PC — mirrors PyQt5: self.controller.detector.pc = tuple(mean_pc).
            # A global refine yields ONE PC: after a grid calibration it replaces
            # the controller's per-pixel detector (the controller's own copy only).
            ctrl_action = ctrl.set_global_pc(mean_pc)
            ctrl.clear_cache()
            job_sim_cache.clear()

            # Update CalibrationStore so indexing and other modules see the refined
            # PC — on the active dataset AND its parent (F2), so a refine on a
            # derived copy reaches the raw dataset the user phase-tests/indexes/exports.
            # A global refine is asking for ONE PC: a grid-calibrated map of the
            # active dataset is replaced by it (a map from the file keeps its
            # structure and is shifted, as before). Otherwise the page would show
            # the refined PC while indexing kept reading the old map.
            written = {}
            store_action = _writeback_refined_pc(
                dataset, mean_pc,
                refinement=_refinement_record(
                    len(patterns_data), "PSO" if "PSO" in report.method else "Nelder-Mead", mode=mode),
                on_refined_map="replace", report=written)
            # A dataset the store does not know took nothing: the page has the PC, indexing will not.
            # No refusal (a signal without a store entry may still use the page), but it is said.
            pc_stored = store_action is not None
            if not pc_stored:
                _warns.append(_PC_NOT_STORED_TEXT)
                _warn_codes.append({"code": "pcNotStored", "params": {}})
                _pc_warning = "  ".join(_warns)
                logger.warning("[pc-optimize] %s", _PC_NOT_STORED_TEXT)

            # Index every calibration pattern at the refined PC (the same code
            # as Index All): the CI and the winning phase of each, and from them
            # the Global CI. The first pattern's segments redraw the overlay.
            ci = None
            segments = []
            pattern_results = []
            global_ci = None
            if len(ctrl.patterns) > 0 and ctrl.phase is not None:
                for idx in range(len(ctrl.patterns)):
                    sim_result = None
                    for attempt in range(2):
                        try:
                            if attempt > 0:
                                ctrl.indexer = None
                            sim_result = _index_and_simulate(ctrl, idx, job_sim_cache)
                            break
                        except Exception as e:
                            logger.warning("Re-index of pattern %d after optimization attempt %d failed: %s",
                                           idx, attempt + 1, e)
                    if sim_result is None:
                        continue
                    if idx == 0:
                        ci = sim_result.get("ci")
                        segments = sim_result.get("segments", [])
                    # `segments`: the page draws the lines of the pattern selected
                    # when the run FINISHES, which is not necessarily the first.
                    entry = {"index": idx, "ci": sim_result.get("ci"),
                             "phase_name": sim_result.get("phase_name"),
                             "segments": sim_result.get("segments", [])}
                    if "phase_index" in sim_result:
                        entry["phase_index"] = sim_result["phase_index"]
                    if "phase_fits" in sim_result:
                        entry["phase_fits"] = sim_result["phase_fits"]
                    pattern_results.append(entry)
                # Over the patterns that could be indexed, as Index All does.
                global_ci = _global_ci([e["ci"] for e in pattern_results]
                                       if pattern_results else [])

            # Several phases: say which one each optimised pattern is indexed
            # as at the refined PC. A one-phase refine reports nothing extra.
            _multi = {}
            if ctrl.phase_list is not None and len(ctrl.phase_list.ids) > 1:
                _multi = {"phase_names": ctrl.phase_names()}
                try:
                    if _same_patterns(ctrl, patterns_data) and len(pattern_results) == len(patterns_data):
                        # The optimised patterns ARE the calibration patterns, just
                        # indexed above at the refined PC: one answer, not two.
                        _per_pattern = [
                            {"phase_name": (e["phase_name"] or None) if e.get("phase_index") is not None else None,
                             "phase_index": e.get("phase_index"), "ci": e["ci"],
                             "phase_fits": e.get("phase_fits")}
                            for e in pattern_results]
                    else:
                        _per_pattern = ctrl.phases_of_patterns(patterns_data, coords=coords)
                    _multi["pattern_phases"] = [{"index": i, **e}
                                                for i, e in enumerate(_per_pattern)]
                except Exception:
                    # The PC is already refined and applied; failing to label
                    # the patterns must not discard it.
                    logger.warning("per-pattern phase lookup failed after PC refine",
                                   exc_info=True)

            _optimization_tasks[task_id]["status"] = "completed"
            _optimization_tasks[task_id]["result"] = {
                # What each calibration pattern prefers on its own, started at the joint result.
                # It shows how far the patterns disagree; the PC below is NOT their mean.
                "pc_values": results,
                # The PC that minimises the cost of all calibration patterns together (the key keeps
                # its old name: the page and the export read it).
                "mean_pc": mean_pc,
                "mode": mode,
                "search": report.to_dict(),
                # How good the result is, in the few numbers the page shows (contrast, warning, axes,
                # patterns), and for a refine the contrast of the PC it started from
                "quality": {**pc_search.quality_of(report.to_dict()),
                            "contrast_before": report.to_dict().get("contrast_at_start")},
                # The PC the file itself gave (None when it gave none) and how far the result is
                # from it: for "Find PC" on a file that has a PC, the two numbers to compare.
                "file_pc": None if file_pc is None else [float(v) for v in file_pc],
                "file_pc_deviation": (None if file_pc is None else float(
                    np.max(np.abs(np.asarray(mean_pc, dtype=float) - file_pc)))),
                # The calibration this dataset was on before the run (if any), whether it left it, and the
                # other datasets still on that version: the page asks what to do with them, nothing
                # is passed on here.
                "calibration": _calibration_after_run(dataset, report, was_on),
                # the dataset the run was for, and whether the active one is another by
                # now: the result is stored for ITS dataset, and the page must not show
                # it as the PC of the file in front of the user
                "dataset": dataset,
                "dataset_changed": _active_dataset_name() != dataset,
                "ci": ci,
                "global_ci": global_ci,
                "pattern_results": pattern_results,
                "segments": segments,
                "pc_warning": _pc_warning,
                "pc_warning_codes": _warn_codes,
                "pc_deviation": _pc_dev,
                # What happened to a per-pixel PC map: "replaced" | "shifted" | "none"
                # (indexing reads the store's map: when the store knows the dataset, its action is the
                # one told, however the controller's own copy of the map was set)
                "pc_map_action": _pc_map_action(None if pc_stored else ctrl_action, store_action, written),
                # per dataset the PC was written to (this one, its parent, the raw root, copies)
                "pc_map_actions": written,
                # False when the calibration store does not know the dataset: the PC is on the page only
                "pc_stored": pc_stored,
                # what the dataset's PC was before this run wrote its own (None when nothing was written)
                "previous": _previous_of(dataset),
                "pattern_size": [int(_ph), int(_pw)] if _ph is not None else None,
                **_multi,
            }
    except _RunCancelled:
        _mark_cancelled(task_id)
    except _FileChanged:
        _fail_task_file_changed(task_id, dataset)
    except HoughLibraryTooBig as e:
        _fail_task_refused(task_id, e)
    except Exception as e:
        _optimization_tasks[task_id]["status"] = "failed"
        _optimization_tasks[task_id]["error"] = str(e)
    finally:
        _cancel_requested.discard(task_id)
        _optimization_active = False


def _fail_task_refused(task_id, exc):
    """End a background task whose Hough library was refused for memory.

    The same coded problem the other routes answer with (``exc.as_problem()``: code,
    English message, the phases with their rows and the one-click count), so the page
    words it in the user's language and offers the answer where the failure is shown,
    instead of the English text of ``str(exc)`` alone.
    """
    logger.warning("Hough library refused: %s", exc)
    task = _optimization_tasks[task_id]
    task["status"] = "failed"
    task["error"] = str(exc)
    task["problem"] = exc.as_problem()
    task["message"] = "Hough library too big"


@router.post("/optimize")
async def optimize_pc(req: OptimizePCRequest, background_tasks: BackgroundTasks):
    """Start PC optimization as a background task.

    Rejects concurrent requests — only one optimization can run at a time.
    """
    global _optimization_active

    # Reject if optimization is already running
    if _optimization_active:
        raise HTTPException(
            status_code=409,
            detail="An optimization is already running. Please wait for it to complete.",
        )

    # Get patterns from EBSD signal
    from backend.api.routes.ebsd_viewer import _get_active_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available")
    if ctrl.phase is None:
        raise HTTPException(status_code=400, detail="No phase loaded")

    nav_shape = signal.axes_manager.navigation_shape
    n_cols = nav_shape[0] if len(nav_shape) >= 1 else 1

    coords = [(int(idx // n_cols), int(idx % n_cols)) for idx in req.patterns]

    # The file the search is for, as it is NOW: taken before the first await below, so that a load or a switch
    # while the patterns are read cannot name another dataset, or another file of the same name, for the job.
    from backend.api.routes.ebsd_viewer import _active_dataset
    _sess = _sessions.get(_active_dataset)
    started_on = _store_file_state(_active_dataset)

    # What the search is given is refused here, with its code, before a job exists.
    from backend.api.problem import problem
    from backend.api.services import pc_search
    if req.mode == "find":
        if len(req.patterns) < pc_search.MIN_PATTERNS_FIND:
            raise problem(
                400, "tooFewPatterns",
                f"Find PC needs at least {pc_search.MIN_PATTERNS_FIND} calibration patterns, "
                f"got {len(req.patterns)}")
    else:
        if not req.patterns:
            # Nothing to fit: the job used to average an empty list and write NaN as the PC.
            raise problem(400, "noPatterns", "A refine needs at least one calibration pattern")
        try:
            pc_search.check_refine(req.method, _start_pc(ctrl, coords, len(coords)), req.search_limit)
        except pc_search.SearchError as exc:
            raise problem(400, exc.code, str(exc))

    # The calibration pattern the page shows and indexes is the copy stored when it was added (or
    # picked): the live signal may have been processed since, and a search must see the pattern
    # the user sees. A requested position that is not among the stored patterns is read from the signal.
    stored = {(int(c[0]), int(c[1])): np.asarray(p) for c, p in ctrl.patterns}

    def _read_patterns():
        # As the map run sees each pixel (an int16 scan rescaled by the range of the whole scan).
        from backend.api.services.load_choices import signal_pattern
        return [stored[rc] if rc in stored else signal_pattern(signal, *rc) for rc in coords]

    # The first pattern of a signal reads the scan's range: off the event loop.
    patterns_data = await asyncio.to_thread(_read_patterns)

    # Clean up old completed/failed tasks to prevent memory leaks
    stale_keys = [k for k, v in _optimization_tasks.items()
                  if v["status"] in ("completed", "failed", "cancelled")]
    for k in stale_keys:
        del _optimization_tasks[k]

    task_id = str(uuid.uuid4())
    _optimization_active = True
    _track_task(task_id, {
        "status": "running",
        "progress": 0.0,
        "result": None,
        "error": None,
    })

    background_tasks.add_task(_run_optimization, task_id, patterns_data, req.method,
                              req.search_limit, coords, _active_dataset, ctrl, mode=req.mode,
                              sim_cache=_sess.sim_cache if _sess is not None else None,
                              started_on=started_on)

    return {"task_id": task_id, "status": "running"}


@router.get("/optimize/{task_id}")
async def get_optimization_status(task_id: str):
    """Check optimization task status."""
    if task_id not in _optimization_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _optimization_tasks[task_id]


@router.post("/optimize/{task_id}/cancel")
async def cancel_optimization(task_id: str):
    """Ask a running Global PC Refine or grid calibration to stop.

    A PC search stops at its next progress call or before it writes, a grid
    calibration between two grid points; the run writes nothing. The answer is "cancelling" while it winds down; the status
    route then reports "cancelled". A run that already finished keeps its result
    and its status is returned as it is.
    """
    task = _optimization_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.get("status") != "running":
        return {"task_id": task_id, "status": task.get("status")}
    _cancel_requested.add(task_id)
    return {"task_id": task_id, "status": "cancelling"}


class GridCalibrationRequest(BaseModel):
    grid_step: int = 10
    # "plane" = smooth PC-field fit (detector.fit_pc); "extrapolate" = mean +
    # spatial model (detector.extrapolate_pc). The "Global (single PC)" UI
    # mode never calls this endpoint (global PC uses /optimize instead).
    mode: str = "plane"
    method: str = "PSO"        # PC optimiser at each grid point (PSO | NM)
    search_limit: float = 0.05
    # Plane-fit model: "affine" (default — robust to PSO PC noise; needs >=3
    # valid grid points) or "projective" (needs >=4; ~4x lower RMS only on
    # low-noise DENSE grids per Winkelmann 2020, but overfits/blows up on noisy
    # PSO PCs — verified 15x worse on real LoGainNi). Output is range-checked.
    transformation: str = "affine"


@router.post("/calibrate/grid")
async def calibrate_grid(req: GridCalibrationRequest, background_tasks: BackgroundTasks):
    """Run pixel-wise PC grid calibration.

    One PC job at a time (this, a Global PC Refine, ...): the run uses the
    controller's detector and indexer for minutes and then replaces its detector.
    """
    global _optimization_active
    if _optimization_active:
        raise HTTPException(
            status_code=409,
            detail="A PC optimization is already running. Please wait for it to complete.",
        )
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available")
    if ctrl.phase is None:
        raise HTTPException(status_code=400, detail="No phase loaded")

    # Need the full EBSD signal to refine PC at each grid point (the
    # controller only holds the sparse user-added calibration patterns).
    from backend.api.routes.ebsd_viewer import _get_active_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")
    # The run takes minutes: its result is written to the dataset it was started
    # on, not to whichever is active when it ends.
    from backend.api.routes.ebsd_viewer import _active_dataset as grid_dataset
    # ... and the results simulated for it are dropped from its session's cache.
    grid_sess = _sessions.get(grid_dataset)
    grid_sim_cache = grid_sess.sim_cache if grid_sess is not None else _sim_cache
    grid_started_on = _store_file_state(grid_dataset)       # the file the run is for

    # The extrapolation turns a beam step into a PC shift with the detector's pixel
    # size (extrapolate_pc divides by px_size * binning). kikuchipy's default of 1 um
    # means nobody set one (Oxford and EDAX scans and the nickel gain series load with it): the drift
    # would come out hundreds of times too large, silently.
    if req.mode == "extrapolate" and not float(ctrl.detector.px_size) > 1.0:
        from backend.api.problem import problem
        raise problem(
            400, "pixel_size_unknown",
            "Extrapolating the PC over the scan needs the detector's pixel size, and none "
            "is set (the default 1 um would make the PC drift hundreds of times too "
            "large). Enter the detector width and the binning under Pixel Size & Binning "
            "and press Apply Pixel Size, then run the extrapolation again.")

    task_id = str(uuid.uuid4())
    _optimization_active = True
    _track_task(task_id, {
        "status": "running",
        "progress": 0.0,
        "result": None,
        "error": None,
    })

    def run_calibration():
        global _optimization_active
        try:
            import numpy as _np
            nr, nc = int(signal.data.shape[0]), int(signal.data.shape[1])
            # 1) sparse grid of calibration points across the whole scan
            grid = ctrl.generate_calibration_grid(nr, nc, step=int(req.grid_step))
            n_pts = len(grid)

            def _prog(done, total):
                _optimization_tasks[task_id]["progress"] = 0.05 + 0.9 * (done / max(total, 1))
                _check_cancelled(task_id)       # Cancel stops between two grid points

            # 2) refine PC at each grid point (the slow part — PSO/NM per point)
            refined = ctrl.refine_pc_at_points(
                signal, grid, method=req.method,
                search_limit=req.search_limit, progress_callback=_prog,
            )

            # 3) fit/extrapolate a per-pixel PC across the full scan
            if req.mode == "extrapolate":
                step_um = None
                try:
                    sx = float(signal.axes_manager[0].scale)
                    step_um = (sx, sx)
                except Exception:
                    step_um = None
                new_det = ctrl.extrapolate_pc_from_points(grid, refined, nr, nc, step_sizes=step_um)
            else:
                new_det = ctrl.fit_pc_from_grid(
                    grid, refined, nr, nc, transformation=req.transformation,
                )

            # 4) make the per-pixel detector the active one for indexing
            with _state_lock:
                _check_cancelled(task_id)       # last point where Cancel changes nothing
                _check_same_file(grid_dataset, grid_started_on)
                ctrl.apply_pixelwise_pc(new_det)
                grid_sim_cache.clear()       # the bands and CI were made at the old PC

            # P3-A: land the fitted per-pixel PC field in the calibration store so the
            # phase test / indexing / export actually USE it (a genuine "refined_map",
            # which F1 spares and the single-pixel phase test reads at the clicked pixel
            # via F3). Previously the field lived only in the PC-Refinement controller
            # and was invisible to every store consumer.
            try:
                calibration_store.update_pc_map(
                    grid_dataset, _np.asarray(new_det.pc).reshape(nr, nc, 3),
                    refinement=_refinement_record(n_pts, req.method))
            except Exception:
                logger.warning("grid-calibration store write-back skipped", exc_info=True)

            # 5) summarise the per-pixel PC field for the UI ("Show PC Map")
            pc = _np.asarray(new_det.pc).reshape(-1, 3)
            valid = _np.asarray(refined).reshape(-1, 3)
            valid = valid[_np.all(_np.isfinite(valid), axis=1)]
            comp = ["pcx", "pcy", "pcz"]
            pc_stats = {
                comp[i]: {
                    "min": float(_np.nanmin(pc[:, i])),
                    "max": float(_np.nanmax(pc[:, i])),
                    "mean": float(_np.nanmean(pc[:, i])),
                    "span": float(_np.nanmax(pc[:, i]) - _np.nanmin(pc[:, i])),
                }
                for i in range(3)
            }
            # Noise-floor honesty check: per-pixel PC only helps when the real
            # spatial drift exceeds the per-point refinement scatter. SNR per
            # component = field_span / point_scatter. If NO component varies
            # clearly (>=2x) above its measurement noise, the plane is fitting
            # noise -> applying it can DEGRADE indexing vs the global PC
            # (verified on LoGainNi 2026-06-03: SNR max 1.3, result 1.35deg ->
            # 2.00deg vs Hough). We still apply it (user asked), but flag it.
            scatter = _np.std(valid, axis=0) if len(valid) >= 2 else _np.zeros(3)
            field_span = _np.array([pc_stats[comp[i]]["span"] for i in range(3)])
            snr = field_span / _np.where(scatter > 1e-9, scatter, _np.nan)
            snr_max = float(_np.nanmax(snr)) if _np.isfinite(snr).any() else 0.0
            noise_dominated = bool(len(valid) >= 2 and snr_max < 2.0)
            noise_msg = None
            if noise_dominated:
                noise_msg = (
                    "Per-pixel PC drift is at/below the refinement noise floor "
                    f"(best SNR {snr_max:.1f}x; field span {field_span.round(4).tolist()} "
                    f"vs point scatter {scatter.round(4).tolist()}). The fit is "
                    "dominated by noise; the GLOBAL PC is likely better here. Consider "
                    "Global (single PC) mode, more patterns per grid point, or a "
                    "coarser grid."
                )
                logger.warning("[grid-calib] %s", noise_msg)
            _optimization_tasks[task_id]["status"] = "completed"
            _optimization_tasks[task_id]["progress"] = 1.0
            _optimization_tasks[task_id]["result"] = {
                "mode": req.mode,
                "grid_step": int(req.grid_step),
                "n_calibration_points": n_pts,
                "n_valid_points": int(len(valid)),
                "grid_shape": [nr, nc],
                "pc_per_pixel_shape": list(pc.shape),
                "pc_stats": pc_stats,
                "point_scatter": {comp[i]: float(scatter[i]) for i in range(3)},
                "pc_snr": {comp[i]: (float(snr[i]) if _np.isfinite(snr[i]) else None) for i in range(3)},
                "noise_dominated": noise_dominated,
                "noise_warning": noise_msg,
                # kikuchipy's plane fit estimates the sample tilt too; it is NOT applied
                # (the tilt set on the page stays), and reported to compare with it.
                "dataset": grid_dataset,
                "dataset_changed": _active_dataset_name() != grid_dataset,
                "fitted_sample_tilt": (ctrl.last_fitted_sample_tilt
                                       if req.mode != "extrapolate" else None),
                "sample_tilt": float(new_det.sample_tilt),
                "applied": True,
            }
            logger.info("Pixel-wise PC grid calibration done: %d pts, mode=%s, "
                        "PC span x=%.4f y=%.4f z=%.4f", n_pts, req.mode,
                        pc_stats["pcx"]["span"], pc_stats["pcy"]["span"],
                        pc_stats["pcz"]["span"])
        except _RunCancelled:
            _mark_cancelled(task_id)
        except _FileChanged:
            _fail_task_file_changed(task_id, grid_dataset)
        except HoughLibraryTooBig as e:
            _fail_task_refused(task_id, e)
        except Exception as e:
            logger.exception("Pixel-wise PC grid calibration failed")
            _optimization_tasks[task_id]["status"] = "failed"
            _optimization_tasks[task_id]["error"] = str(e)
        finally:
            _cancel_requested.discard(task_id)
            _optimization_active = False

    background_tasks.add_task(run_calibration)
    return {"task_id": task_id, "status": "running"}


def _render_preview_sync(req: "RenderPreviewRequest"):
    """Render a forward-simulated EBSP for one calibration pattern using
    the trial geometry, return both PNGs + NCC vs experimental.

    Heavy + synchronous (Hough indexing under the trial geometry + GPU SHT
    render + NCC). Runs in a worker thread (see the async wrapper below) so a
    geometry-slider drag doesn't freeze the whole backend.

    Closes the sample-tilt + det-tilt + azimuthal degeneracies that
    Hough band-fitting can't break. The user changes a parameter, this
    endpoint regenerates the simulated, and the NCC value gives an
    objective fitness signal even where the Hough lines look fine.

    Pipeline:
      1. Fetch experimental pattern from controller.patterns[idx].
      2. Run Hough indexing under the TRIAL geometry to get an
         orientation (or use the user-supplied one).
      3. Build a kikuchipy.EBSDDetector with the trial geometry,
         pc_convention='bruker' (matches the rest of the app).
      4. Render via the existing SHT forward renderer.
      5. NCC between experimental + simulated.
    """
    import io as _io
    import base64 as _b64
    from pathlib import Path as _Path
    import numpy as _np
    from PIL import Image as _Image

    ctrl = _get_controller()
    if req.pattern_idx < 0 or req.pattern_idx >= len(ctrl.patterns):
        raise HTTPException(
            status_code=400,
            detail=f"Pattern index {req.pattern_idx} out of range (have {len(ctrl.patterns)})",
        )
    if not _Path(req.sht_path).is_file():
        raise HTTPException(status_code=404, detail=f"SHT file not found: {req.sht_path}")
    if len(req.pc) != 3:
        raise HTTPException(status_code=400, detail="pc must have 3 values [PCx, PCy, PCz]")

    coords, exp_pattern = ctrl.patterns[req.pattern_idx]
    exp_arr = _np.asarray(exp_pattern, dtype=_np.float32)
    if exp_arr.ndim != 2:
        raise HTTPException(
            status_code=500,
            detail=f"experimental pattern must be 2D, got shape {exp_arr.shape}",
        )
    H, W = exp_arr.shape

    # --- Step 1: build kikuchipy detector with TRIAL geometry (Bruker conv) ---
    try:
        from kikuchipy.detectors import EBSDDetector
        det = EBSDDetector(
            shape=(H, W),
            sample_tilt=float(req.sample_tilt),
            tilt=float(req.detector_tilt),
            azimuthal=float(req.azimuthal),
            pc=req.pc,
            convention="bruker",
            binning=int(req.binning),
        )
        if req.pixel_size is not None:
            try:
                det.pixel_size = float(req.pixel_size)
            except AttributeError:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detector build failed: {e}")

    # --- Step 2: orientation source ---
    # Priority: (1) user-supplied → (2) Hough → (3) identity.
    # NOTE (2026-06-27): an attempt to source the orientation from the
    # SHT-spherical (Tier1) indexer here was REVERTED. Empirically Hough gives
    # the CORRECT band overlay + a perfect Forward-Sim match on this app's data,
    # whereas spherical Tier1 lands in the WRONG BASIN for some phases (e.g.
    # alpha-AlFeMnSi → NCC ~0.18 vs Hough's >0.4) and rendered a visibly wrong
    # pattern. The spherical-vs-Hough disagreement is a real indexer bug under
    # investigation; until it's fixed the preview must keep using Hough.
    from backend.spherical_gpu.pipeline.detector import (
        DEFAULT_PIXEL_SIZE_UM, stored_pixel_size,
    )
    # One grid, one pixel size: the renderer and the PC conversion both work on
    # the stored (binned) pattern. The default is already a stored-pixel size;
    # an explicit pixel size is the unbinned sensor pixel.
    _px = stored_pixel_size(
        req.pixel_size if req.pixel_size is not None else DEFAULT_PIXEL_SIZE_UM,
        req.binning, pixel_size_is_stored_pixel=req.pixel_size is None)
    rot_obj = None
    orientation_euler_deg = None
    orientation_source = None
    indexed_phase = None      # several phases loaded: the one Hough picked here

    if req.orientation_euler_deg is not None and len(req.orientation_euler_deg) == 3:
        from orix.quaternion import Rotation
        eu_rad = _np.deg2rad(req.orientation_euler_deg)
        rot_obj = Rotation.from_euler(eu_rad.reshape(1, 3))
        orientation_euler_deg = list(req.orientation_euler_deg)
        orientation_source = "supplied"

    if rot_obj is None:
        try:
            from kikuchipy.signals import EBSD as _EBSD
            try:
                from indexing.indexing_utils import create_indexer  # type: ignore
            except Exception:
                from ebsd_utils import create_indexer
            if ctrl.phase_list is None:
                raise RuntimeError("no phase loaded — call /api/pc/phase/load first")
            ebsd = _EBSD(exp_arr[_np.newaxis, _np.newaxis], detector=det)
            # Build a TRANSIENT indexer for this trial geometry (don't mutate
            # the cached one in ctrl — that's tied to its own detector).
            indexer = create_indexer(det, ctrl.phase_list, ctrl.reflectors, nBands=ctrl.nBands)
            xmap, _idx, _bands = ebsd.hough_indexing(
                ctrl.phase_list, indexer,
                return_index_data=True, return_band_data=True, verbose=0,
            )
            rot_obj = xmap.rotations
            orientation_euler_deg = [
                float(x) for x in _np.rad2deg(rot_obj.to_euler()[0])
            ]
            orientation_source = "hough"
            if len(ctrl.phase_list.ids) > 1:
                _won = list(xmap.phases_in_data.names)
                indexed_phase = str(_won[0]) if _won else None
        except Exception as e:
            logger.warning("Hough at trial geometry failed: %s — falling back to identity", e)
            from orix.quaternion import Rotation
            rot_obj = Rotation.identity()
            orientation_euler_deg = [0.0, 0.0, 0.0]
            orientation_source = "identity"

    # --- Step 3: render simulated via SHT forward renderer ---
    try:
        from backend.api.services.sht_pattern_renderer import (
            render_pattern_to_png_b64,
        )
        from backend.spherical_gpu.pipeline.detector import convert_pc_to_emsoft
        import torch as _torch

        # _px was already resolved in Step 2 (shared with the spherical helper).
        xpc, ypc, L_um = convert_pc_to_emsoft(
            pc=tuple(req.pc),
            vendor="Bruker",  # PC is in Bruker convention (see SetDetectorRequest)
            pat_width=W,
            pat_height=H,
            pixel_size=_px,
            binning=1,
        )

        q_arr = _np.asarray(rot_obj.data).reshape(-1)[:4]
        q_t = _torch.tensor(q_arr, dtype=_torch.float64)
        # det_tilt_deg MUST be passed alongside sample_tilt. The Tier1
        # indexer uses alpha = 90 - sample_tilt + det_tilt; omitting
        # det_tilt here makes the renderer use 0, which produces a
        # pattern rotated by exactly req.detector_tilt from what the
        # experimental shows. (Bug 2026-05-22: broke the Forward-Sim
        # preview for any geometry with non-zero detector tilt.)
        sim_b64 = render_pattern_to_png_b64(
            sht_path=str(_Path(req.sht_path).resolve()),
            orientation_quat=q_t,
            pc_emsoft=(float(xpc), float(ypc), float(L_um)),
            detector_shape=(H, W),
            pixel_size_um=_px,
            tilt_deg=float(req.sample_tilt),
            det_tilt_deg=float(req.detector_tilt),
            max_bandwidth=int(req.max_bandwidth),
        )
    except Exception as e:
        logger.exception("render-preview SHT render failed")
        raise HTTPException(status_code=500, detail=f"render failed: {e}")

    # --- Step 4: NCC between experimental and simulated ---
    # EDAX patterns are recorded through a circular aperture (black corners);
    # the SHT renderer fills the full square, so the corner mismatch would
    # crater the NCC. Auto-detect the aperture from the experimental pattern
    # and, for EDAX only, score over the inscribed disc and mask the simulated
    # thumbnail to match. Oxford full-frame patterns carry real signal in the
    # corners (session is not circular) → unchanged full-frame NCC.
    sim_b64_display = sim_b64
    try:
        sim_arr = _np.asarray(
            _Image.open(_io.BytesIO(_b64.b64decode(sim_b64))).convert("L"),
            dtype=_np.float32,
        )
        from tools.pattern_comparison import (
            circular_mask as _circ_mask,
            compute_ncc_scalar_masked as _ncc_masked,
        )
        # The aperture is the detector's, decided once per session (as the image route does):
        # the stored pattern may be processed, signed data whose corners say nothing.
        if _session_is_circular(ctrl) and sim_arr.shape == exp_arr.shape:
            _mask = _circ_mask(exp_arr.shape, 1.0)
            ncc = _ncc_masked(exp_arr, sim_arr, _mask)
            # Black out the simulated corners so the preview matches the
            # experimental, WITHOUT re-normalising — preserve the renderer's
            # exact grayscale inside the disc (sim_arr is already 0–255 from the
            # decoded PNG, so a uint8 cast + direct PIL re-encode is lossless).
            _sim_u8 = sim_arr.astype(_np.uint8).copy()
            _sim_u8[~_mask] = 0
            _buf = _io.BytesIO()
            _Image.fromarray(_sim_u8, "L").save(_buf, format="PNG")
            sim_b64_display = _b64.b64encode(_buf.getvalue()).decode("utf-8")
        else:
            a0 = exp_arr - exp_arr.mean()
            b0 = sim_arr - sim_arr.mean()
            denom = float(_np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum()))
            ncc = float((a0 * b0).sum() / denom) if denom > 1e-12 else 0.0
    except Exception as e:
        logger.warning("ncc computation failed: %s", e)
        ncc = None

    # An "identity" orientation is the emergency fallback when no orientation could
    # be indexed at this geometry: the pattern rendered from it matches the
    # experimental one by chance, so its NCC says nothing about the geometry (it
    # used to be reported, and read as "this PC is bad"). No number, and the reason.
    ncc_reason = None
    if orientation_source == "identity":
        ncc, ncc_reason = None, "no_orientation"

    # --- Step 5: encode experimental for the frontend (mirror simulated path) ---
    from backend.api.services.image_utils import array_to_base64_png as _arr2b64
    exp_b64 = _arr2b64(exp_pattern)

    return {
        "success": True,
        "experimental_b64": exp_b64,
        "simulated_b64": sim_b64_display,
        "ncc": ncc,
        "ncc_reason": ncc_reason,
        "orientation_euler_deg": orientation_euler_deg,
        "orientation_source": orientation_source,
        "pattern_row": int(coords[0]),
        "pattern_col": int(coords[1]),
        "detector_shape": [H, W],
        "trial_geometry": {
            "pc": req.pc,
            "sample_tilt": req.sample_tilt,
            "detector_tilt": req.detector_tilt,
            "azimuthal": req.azimuthal,
            "binning": req.binning,
            "pixel_size": req.pixel_size,
            "max_bandwidth": req.max_bandwidth,
        },
        # Only with several phases loaded; a one-phase reply is unchanged.
        **({"indexed_phase": indexed_phase} if indexed_phase is not None else {}),
    }


@router.post("/render-preview")
async def render_preview(req: RenderPreviewRequest):
    """Forward-sim preview for the PC page — offloaded to a worker thread so
    the geometry sliders stay responsive (HTTPExceptions raised inside the
    sync renderer propagate through to_thread unchanged)."""
    return await asyncio.to_thread(_render_preview_sync, req)
