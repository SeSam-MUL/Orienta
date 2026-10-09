"""
PC Refinement API Routes

Wraps PCController for:
- Phase loading from CIF
- Indexing parameter configuration
- PC optimization (background task)
- Result retrieval
"""

import asyncio
import logging
import os
import threading
import uuid
from typing import Optional, List, Union

import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, model_validator

from backend.api.services.image_utils import array_to_base64_raw
from pc_controller import PhaseSetError
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


def _refinement_record(n_patterns: int, method) -> dict:
    """What the calibration store keeps about a PC that came from refinement.

    ``method`` is what the page sent to the optimiser: "PSO" or "Nelder-Mead".
    """
    name = "PSO" if str(method).strip().upper() == "PSO" else "Nelder-Mead"
    return {"n_patterns": int(n_patterns), "method": name}


def _writeback_refined_pc(dataset_name: str, pc, store=None, source: str = "refined",
                          refinement: Optional[dict] = None) -> None:
    """Write a single PC to the dataset AND its parent (F2).

    A Global PC Refine almost always runs on a *derived* copy (e.g.
    ``..._bg_dyn_avg``), so writing only the active dataset strands the refined
    PC there while the raw parent — the dataset the user phase-tests, indexes and
    exports — keeps the stale header PC. We therefore also write the refined PC to
    the single explicit ``parent_name`` link (never siblings). Both writes go
    through the F1-gated ``update_pc`` (a header/inherited per-pixel map is
    rigid-shifted; a genuine ``refined_map`` is preserved), so this is safe to
    call unconditionally. ``store`` is injectable for testing.

    ``refinement`` (``{"n_patterns", "method"}``) is given by the refine routes
    and recorded with the PC on both entries, so a run indexed at it can credit
    the software that refined it; a manual PC passes none and clears it.
    """
    store = store or calibration_store
    store.update_pc(dataset_name, pc, source=source, refinement=refinement)
    entry = store.get_entry(dataset_name)
    if entry is not None and entry.parent_name:
        store.update_pc(entry.parent_name, pc, source="propagated",
                        refinement=refinement)


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


# Per-dataset sessions. Key = dataset name (matches ``_active_dataset``
# in ebsd_viewer). Sessions are lazily created on first access for a
# dataset by ``_get_session`` and persist until backend restart.
_sessions: dict[str, _PCSession] = {}

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
    if key not in _sessions:
        from pc_controller import PCController
        _sessions[key] = _PCSession(ctrl=PCController())
    sess = _sessions[key]
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


def _index_and_simulate(ctrl, pattern_idx):
    """Index a single calibration pattern and simulate Kikuchi line segments.

    Returns dict with {ci, phase_name, segments, n_bands}.
    Results are cached in _sim_cache keyed by pattern_idx.
    """
    import kikuchipy as kp
    from kikuchipy.signals import EBSD
    from ebsd_utils import create_indexer
    from hough_reflectors import prepare_reflectors

    # The reflector selection of a phase may have changed since the indexer was
    # built: drop it (and what was indexed with it) so it is rebuilt below.
    ctrl.reflector_specs_changed()

    # An indexer copies the PC it was built at; a refine or a typed PC that came
    # after it must not be indexed with it.
    if ctrl.indexer is not None and ctrl._indexer_pc_is_stale():
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
            ctrl.detector, ctrl.phase_list, reflectors,
            nBands=getattr(ctrl, 'nBands', 12),
        )

    coords, pat = ctrl.patterns[pattern_idx]

    # Verify detector shape matches pattern shape — mismatch causes overlay shifting
    det_shape = tuple(ctrl.detector.shape)
    pat_shape = tuple(pat.shape)
    if det_shape != pat_shape:
        logger.warning("Detector shape %s != pattern shape %s — adjusting detector",
                       det_shape, pat_shape)
        from kikuchipy.detectors import EBSDDetector
        ctrl.detector = EBSDDetector(
            shape=pat_shape,
            pc=[float(v) for v in ctrl.detector.pc.flatten()],
            sample_tilt=float(ctrl.detector.sample_tilt),
            tilt=float(ctrl.detector.tilt),
            azimuthal=float(ctrl.detector.azimuthal),
        )
        ctrl.indexer = None  # Force indexer recreation with correct detector

    # Hough index
    # pat is a 2D array (H, W).  hough_indexing requires at least one navigation
    # axis so we add two leading singleton dimensions → shape (1, 1, H, W).
    # Using EBSD(Signal2D(pat)) gives navigation_shape=() which crashes pyebsdindex.
    ebsd = EBSD(pat[np.newaxis, np.newaxis], detector=ctrl.detector)
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
        sim = simulator.on_detector(for_kikuchipy_projection(ctrl.detector), rots)
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
    _sim_cache[pattern_idx] = result
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

    try:
        # Reject duplicates (same row, col already in list)
        for coords, _ in ctrl.patterns:
            if coords[0] == req.row and coords[1] == req.col:
                return {
                    "success": True,
                    "n_patterns": len(ctrl.patterns),
                    "added": {"row": req.row, "col": req.col},
                    "duplicate": True,
                }
        # Force a standalone copy so the pattern survives a later
        # file switch / signal reload (per-file sessions, 2026-05-26).
        # Without .copy() this may be a view into signal.data, which
        # becomes invalid once _raw_signals.clear() drops the source
        # signal in load_ebsd. A switch back to this dataset would
        # then crash or render garbage.
        pattern = np.asarray(signal.data[req.row, req.col]).copy()
        from backend.api.routes.ebsd_viewer import _active_dataset
        logger.info(f"add_pattern ({req.row},{req.col}) from dataset '{_active_dataset}'")
        ctrl.add_pattern((req.row, req.col), pattern)
        return {
            "success": True,
            "n_patterns": len(ctrl.patterns),
            "added": {"row": req.row, "col": req.col},
            "dataset": _active_dataset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pattern/remove")
async def remove_pattern(idx: int = 0):
    """Remove a calibration pattern by index."""
    ctrl = _get_controller()
    if idx < 0 or idx >= len(ctrl.patterns):
        raise HTTPException(status_code=400, detail=f"Pattern index {idx} out of range (have {len(ctrl.patterns)})")
    ctrl.patterns.pop(idx)
    ctrl.clear_cache()
    global _sim_cache
    _sim_cache.clear()
    return {"success": True, "n_patterns": len(ctrl.patterns)}


@router.post("/phase/load")
async def load_phase(req: LoadPhaseRequest):
    """Load a crystal phase from a CIF file."""
    ctrl = _get_controller()

    # Off the event loop: the first read of a two-origin CIF decides its origin
    # choice (2-4 s). In a worker thread the controller is changed under
    # _state_lock, as _index_pattern_locked does.
    def _load_phase_locked():
        with _state_lock:
            return ctrl.load_phase(req.cif_path)

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

        # Build keyword arguments — only pass what EBSDDetector accepts
        kwargs = {
            "shape": tuple(req.shape),
            "pc": req.pc,
            "sample_tilt": req.sample_tilt,
            "tilt": req.camera_tilt,
            "binning": req.binning,
            "azimuthal": req.azimuthal,
        }

        # convention parameter name varies by kikuchipy version
        try:
            det = EBSDDetector(**kwargs, convention=req.convention)
        except TypeError:
            try:
                det = EBSDDetector(**kwargs, pc_convention=req.convention)
            except TypeError:
                det = EBSDDetector(**kwargs)

        # Store pixel_size as metadata if provided
        if req.pixel_size is not None:
            try:
                det.pixel_size = req.pixel_size
            except AttributeError:
                pass  # Some versions don't have this attribute

        ctrl.attach_detector(det)

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
        }
    except Exception as e:
        logger.exception("Failed to set detector")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/detector/info")
async def detector_info():
    """Return full detector info including binning, pixel size, and all geometry params."""
    ctrl = _get_controller()
    det = ctrl.detector
    if det is None:
        return {"has_detector": False}

    info: dict = {"has_detector": True}

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
        info["pc"] = [float(v) for v in det.pc.flatten()]
    except Exception:
        info["pc"] = []

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
        with _state_lock:
            # Find if this (row, col) is already a calibration pattern
            pat_idx = None
            for i, (coords, _) in enumerate(ctrl.patterns):
                if coords[0] == req.row and coords[1] == req.col:
                    pat_idx = i
                    break

            if pat_idx is None:
                # Pattern not in calibration list — fetch from current active signal
                pattern = signal.data[req.row, req.col]
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
    # Stretched to the full grey range, as the enlarged picture was.
    lo, hi = float(np.nanmin(pat)), float(np.nanmax(pat))
    shown = (np.zeros(pat.shape, dtype=np.uint8) if not hi > lo else
             ((pat.astype(np.float64) - lo) / (hi - lo) * 255).astype(np.uint8))
    image_b64 = array_to_base64_raw(shown)
    return {
        "success": True,
        "image": image_b64,
        "row": coords[0],
        "col": coords[1],
        "shape": [int(v) for v in pat.shape[:2]],
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

    # Check sim_cache first (has segments)
    if idx in _sim_cache:
        return {"success": True, "cached": True, **_sim_cache[idx]}

    # Not cached — index on demand if we have detector + phase
    if ctrl.detector is None or ctrl.phase is None:
        return {"success": False, "detail": "Detector or phase not set"}

    try:
        result = _index_and_simulate(ctrl, idx)
        return {"success": True, "cached": False, **result}
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
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available")

    # Update PC
    ctrl.detector.pc = (req.pcx, req.pcy, req.pcz)

    # Also write the calibration store (+ parent), so a MANUAL PC reaches the phase
    # test / indexing / export — not just the PC-Refinement controller. (P2-A: the
    # manual endpoint previously wrote only ctrl.detector, like /optimize before F2.)
    try:
        from backend.api.routes.ebsd_viewer import _active_dataset
        _writeback_refined_pc(_active_dataset, [req.pcx, req.pcy, req.pcz], source="manual")
    except Exception:
        logger.warning("manual PC store write-back skipped", exc_info=True)

    # Clear caches (like PyQt5: controller.clear_cache() + sim_cache.clear())
    ctrl.clear_cache()
    global _sim_cache
    _sim_cache.clear()

    result = {"success": True, "pc": [req.pcx, req.pcy, req.pcz]}

    # Re-index current pattern if specified and prerequisites met
    if req.current_pattern_idx is not None and ctrl.phase is not None:
        idx = req.current_pattern_idx
        if 0 <= idx < len(ctrl.patterns):
            try:
                sim_result = _index_and_simulate(ctrl, idx)
                result["ci"] = sim_result["ci"]
                result["segments"] = sim_result["segments"]
                result["n_bands"] = sim_result["n_bands"]
                result["phase_name"] = sim_result["phase_name"]
                # No "global_ci": only the selected pattern was indexed at the new
                # PC, and a mean over all patterns needs all of them (Index All or
                # Global PC Refine give it).
            except Exception as e:
                logger.warning("Re-index after PC update failed: %s", e)
                result["index_error"] = str(e)

    return result


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
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available")

    det = ctrl.detector
    if req.sample_tilt is not None:
        det.sample_tilt = req.sample_tilt
    if req.detector_tilt is not None:
        det.tilt = req.detector_tilt
    if req.azimuthal is not None:
        det.azimuthal = req.azimuthal

    # Clear caches
    ctrl.clear_cache()
    global _sim_cache
    _sim_cache.clear()

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
                sim_result = _index_and_simulate(ctrl, idx)
                result["ci"] = sim_result["ci"]
                result["segments"] = sim_result["segments"]
                result["n_bands"] = sim_result["n_bands"]
                result["phase_name"] = sim_result["phase_name"]
            except Exception as e:
                logger.warning("Re-index after tilt update failed: %s", e)
                result["index_error"] = str(e)

    return result


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
    return {"success": True}


@router.get("/status")
async def pc_status():
    """Get current PC controller state including per-pattern CI from cache."""
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)

    pattern_list = []
    for i, (coords, _) in enumerate(ctrl.patterns):
        entry = {"index": i, "row": int(coords[0]), "col": int(coords[1])}
        # Include CI from sim_cache if available
        if i in _sim_cache:
            entry["ci"] = _sim_cache[i].get("ci")
        elif i in ctrl.cache:
            entry["ci"] = ctrl.cache[i].get("ci")
        pattern_list.append(entry)

    # Global CI: a mean over ALL patterns, only when every one has a CI.
    global_ci = _global_ci([p.get("ci") for p in pattern_list])

    return {
        "has_detector": ctrl.detector is not None,
        "has_phase": ctrl.phase is not None,
        "phase_name": str(ctrl.phase.name) if ctrl.phase else None,
        "phase_names": ctrl.phase_names(),
        "has_indexer": ctrl.indexer is not None,
        "n_patterns": len(ctrl.patterns),
        "pc": [float(v) for v in ctrl.detector.pc.flatten()] if ctrl.detector is not None else [],
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


def _run_optimization(task_id: str, patterns_data, method: str, search_limit: float):
    """Background task for PC optimization.

    Thread-safe: acquires _state_lock around all shared state modifications.
    """
    global _optimization_active
    ctrl = _get_controller()
    try:
        from ebsd_utils import optimize_pc, compute_ci

        with _state_lock:
            ctrl.reflector_specs_changed()
            if ctrl.indexer is None:
                if ctrl.reflectors is None:
                    from hough_reflectors import prepare_reflectors
                    ctrl.reflectors = prepare_reflectors(ctrl.phase_list)
                from ebsd_utils import create_indexer
                ctrl.indexer = create_indexer(
                    ctrl.detector, ctrl.phase_list,
                    ctrl.reflectors, nBands=ctrl.nBands,
                )
            # Snapshot the indexer and detector for this optimization run
            indexer = ctrl.indexer
            detector = ctrl.detector

        # Reference PC (the value BEFORE this refine — e.g. the vendor/.osc PC)
        # + pattern size, for the reliability guard below.
        _ref_pc = np.asarray(detector.pc, dtype=float).reshape(-1, 3).mean(axis=0)
        try:
            _ph, _pw = np.asarray(patterns_data[0]).shape[-2:]
            _pat_min = int(min(int(_ph), int(_pw)))
        except Exception:
            _ph = _pw = _pat_min = None

        results = []
        for i, pat in enumerate(patterns_data):
            mean_pc, _ = optimize_pc(
                detector=detector,
                indexer=indexer,
                pattern=pat,
                method=method,
                search_limit=search_limit,
                batch=True,
            )
            results.append(list(mean_pc))
            _optimization_tasks[task_id]["progress"] = (i + 1) / len(patterns_data)

        mean_pc = list(np.mean(results, axis=0))

        # --- Reliability guard --------------------------------------------------
        # This refine is Hough/Radon band-fitting based (PyEBSDIndex optimize_pc).
        # On small patterns that band-fitting is unreliable, so the PC can drift
        # to a wrong value. Warn when (a) patterns are small (<~100 px) or (b) the
        # refined PC moved far from the starting (vendor/.osc) PC — on low-res
        # data that is usually refinement noise, not real drift. Generally useful
        # whenever patterns are small (user request 2026-07-23).
        _SMALL_PAT_PX = 100
        _PC_DRIFT_TOL = 0.05
        _warns = []
        # Each warning travels as a stable code plus its numbers, so the page
        # can say it in the reader's language; the English sentence stays as
        # the fallback and as what goes into the log.
        _warn_codes = []
        if _pat_min is not None and _pat_min < _SMALL_PAT_PX:
            _warns.append(
                f"Small patterns ({_ph}×{_pw} px): the Hough/Radon band-fitting this "
                f"PC refine relies on is unreliable below ~{_SMALL_PAT_PX} px, so the "
                "refined PC can be noisy. Cross-check it against your vendor/.osc PC.")
            _warn_codes.append({"code": "smallPatterns",
                                "params": {"h": _ph, "w": _pw, "limit": _SMALL_PAT_PX}})
        _pc_dev = float(np.max(np.abs(np.asarray(mean_pc, dtype=float) - _ref_pc)))
        if _pc_dev > _PC_DRIFT_TOL:
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
        _pc_warning = "  ".join(_warns) if _warns else None
        if _pc_warning:
            logger.warning("[pc-optimize] %s", _pc_warning)

        # Apply optimized PC to detector and re-index — all under lock
        with _state_lock:
            # Apply optimized PC — mirrors PyQt5: self.controller.detector.pc = tuple(mean_pc)
            ctrl.detector.pc = tuple(mean_pc)
            ctrl.clear_cache()
            global _sim_cache
            _sim_cache.clear()

            # Update CalibrationStore so indexing and other modules see the refined
            # PC — on the active dataset AND its parent (F2), so a refine on a
            # derived copy reaches the raw dataset the user phase-tests/indexes/exports.
            from backend.api.routes.ebsd_viewer import _active_dataset
            _writeback_refined_pc(
                _active_dataset, mean_pc,
                refinement=_refinement_record(len(patterns_data), method))

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
                            sim_result = _index_and_simulate(ctrl, idx)
                            break
                        except Exception as e:
                            logger.warning("Re-index of pattern %d after optimization attempt %d failed: %s",
                                           idx, attempt + 1, e)
                    if sim_result is None:
                        continue
                    if idx == 0:
                        ci = sim_result.get("ci")
                        segments = sim_result.get("segments", [])
                    entry = {"index": idx, "ci": sim_result.get("ci"),
                             "phase_name": sim_result.get("phase_name")}
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
                        _per_pattern = ctrl.phases_of_patterns(patterns_data)
                    _multi["pattern_phases"] = [{"index": i, **e}
                                                for i, e in enumerate(_per_pattern)]
                except Exception:
                    # The PC is already refined and applied; failing to label
                    # the patterns must not discard it.
                    logger.warning("per-pattern phase lookup failed after PC refine",
                                   exc_info=True)

            _optimization_tasks[task_id]["status"] = "completed"
            _optimization_tasks[task_id]["result"] = {
                "pc_values": results,
                "mean_pc": mean_pc,
                "ci": ci,
                "global_ci": global_ci,
                "pattern_results": pattern_results,
                "segments": segments,
                "pc_warning": _pc_warning,
                "pc_warning_codes": _warn_codes,
                "pc_deviation": _pc_dev,
                "pattern_size": [int(_ph), int(_pw)] if _ph is not None else None,
                **_multi,
            }
    except Exception as e:
        _optimization_tasks[task_id]["status"] = "failed"
        _optimization_tasks[task_id]["error"] = str(e)
    finally:
        _optimization_active = False


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

    patterns_data = []
    for idx in req.patterns:
        row = idx // n_cols
        col = idx % n_cols
        patterns_data.append(signal.data[row, col])

    # Clean up old completed/failed tasks to prevent memory leaks
    stale_keys = [k for k, v in _optimization_tasks.items()
                  if v["status"] in ("completed", "failed")]
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

    background_tasks.add_task(_run_optimization, task_id, patterns_data, req.method, req.search_limit)

    return {"task_id": task_id, "status": "running"}


@router.get("/optimize/{task_id}")
async def get_optimization_status(task_id: str):
    """Check optimization task status."""
    if task_id not in _optimization_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _optimization_tasks[task_id]


class DriftRequest(BaseModel):
    indices: List[int] = []


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


@router.post("/drift")
async def analyze_drift(req: DriftRequest):
    """Analyze PC drift across calibration patterns."""
    ctrl = _get_controller()
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector configured")

    try:
        if hasattr(ctrl, 'analyze_pc_drift'):
            result = ctrl.analyze_pc_drift(req.indices)
            return {"success": True, "drift": result}

        # Fallback: compute PC variation from optimization results
        return {
            "success": True,
            "drift": {
                "message": "PC drift analysis requires multiple calibration patterns",
                "n_patterns": len(req.indices),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibrate/grid")
async def calibrate_grid(req: GridCalibrationRequest, background_tasks: BackgroundTasks):
    """Run pixel-wise PC grid calibration."""
    ctrl = _get_controller()
    _auto_attach_detector(ctrl)
    if ctrl.detector is None:
        raise HTTPException(status_code=400, detail="No detector available")
    if ctrl.phase is None:
        raise HTTPException(status_code=400, detail="No phase loaded")

    task_id = str(uuid.uuid4())
    _track_task(task_id, {
        "status": "running",
        "progress": 0.0,
        "result": None,
        "error": None,
    })

    # Need the full EBSD signal to refine PC at each grid point (the
    # controller only holds the sparse user-added calibration patterns).
    from backend.api.routes.ebsd_viewer import _get_active_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    def run_calibration():
        try:
            import numpy as _np
            nr, nc = int(signal.data.shape[0]), int(signal.data.shape[1])
            # 1) sparse grid of calibration points across the whole scan
            grid = ctrl.generate_calibration_grid(nr, nc, step=int(req.grid_step))
            n_pts = len(grid)

            def _prog(done, total):
                _optimization_tasks[task_id]["progress"] = 0.05 + 0.9 * (done / max(total, 1))

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
            ctrl.apply_pixelwise_pc(new_det)

            # P3-A: land the fitted per-pixel PC field in the calibration store so the
            # phase test / indexing / export actually USE it (a genuine "refined_map",
            # which F1 spares and the single-pixel phase test reads at the clicked pixel
            # via F3). Previously the field lived only in the PC-Refinement controller
            # and was invisible to every store consumer.
            try:
                from backend.api.routes.ebsd_viewer import _active_dataset as _grid_ds
                calibration_store.update_pc_map(
                    _grid_ds, _np.asarray(new_det.pc).reshape(nr, nc, 3),
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
                "applied": True,
            }
            logger.info("Pixel-wise PC grid calibration done: %d pts, mode=%s, "
                        "PC span x=%.4f y=%.4f z=%.4f", n_pts, req.mode,
                        pc_stats["pcx"]["span"], pc_stats["pcy"]["span"],
                        pc_stats["pcz"]["span"])
        except Exception as e:
            logger.exception("Pixel-wise PC grid calibration failed")
            _optimization_tasks[task_id]["status"] = "failed"
            _optimization_tasks[task_id]["error"] = str(e)

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
    # corners (detector returns False) → unchanged full-frame NCC.
    sim_b64_display = sim_b64
    try:
        sim_arr = _np.asarray(
            _Image.open(_io.BytesIO(_b64.b64decode(sim_b64))).convert("L"),
            dtype=_np.float32,
        )
        from tools.pattern_comparison import (
            detect_circular_aperture as _detect_circ,
            circular_mask as _circ_mask,
            compute_ncc_scalar_masked as _ncc_masked,
        )
        if _detect_circ(exp_arr) and sim_arr.shape == exp_arr.shape:
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

    # --- Step 5: encode experimental for the frontend (mirror simulated path) ---
    from backend.api.services.image_utils import array_to_base64_png as _arr2b64
    exp_b64 = _arr2b64(exp_pattern)

    return {
        "success": True,
        "experimental_b64": exp_b64,
        "simulated_b64": sim_b64_display,
        "ncc": ncc,
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
