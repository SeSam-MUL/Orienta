"""
EBSD Viewer API Routes

Wraps the EBSD loading pipeline (safe_loader -> kikuchipy EBSD signal).
Provides pattern viewing, enhancement, and basic metadata.
Supports multiple named datasets with deepcopy and processing operations.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Optional, List
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.services.image_utils import array_to_base64_raw, array_to_base64_png
from backend.api.services.calibration_store import calibration_store
from backend.api.services import crop_window as crop_window_service
from backend.api.services import state_version

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level state: primary signal (for backward compat) + named datasets
_ebsd_signal = None
_ebsd_file_path = None

# Multi-dataset state
_raw_signals: dict = {}   # name -> kikuchipy EBSD signal
_positions: dict = {}     # name -> (row, col) current nav position
_active_dataset: str = "" # name of currently active dataset

# Loaded-files registry — remembers every file path the user has loaded so
# the UI can offer a switcher. Switching re-loads from disk (see /switch-file).
# Each entry: {"path": str, "name": str}. Order = load order.
_loaded_files: list = []

# Datasets whose in-memory patterns were modified in the viewer (frame
# averaging / background removal / autocontrast) and therefore differ from
# the raw file on disk. Spherical indexing reads patterns straight from the
# file, so it checks this flag to use the processed patterns instead.
_dirty_datasets: set = set()

# Per-dataset circular signal mask. Used by /autocontrast (percentile range),
# dynamic-BG halo cleanup, pattern PNG rendering and (when wired) downstream
# indexing/refinement. Default: disabled, fraction=1.0 (inscribed circle).
# Convention: ``mask_include`` shape (H, W) bool, True = pixel inside circle
# = used. kikuchipy's ``signal_mask`` is the inverse (True = excluded).
_signal_masks: dict = {}  # name -> {"enabled": bool, "radius_fraction": float}

# Per-file stash of the in-memory processed-dataset registry. Keyed by
# CANONICAL file path. Mirrors ``analysis._dataset_by_file`` /
# ``indexing._result_registry``: a file switch STASHES the outgoing file's
# datasets here instead of destroying them, and restores them on switch-back —
# so a round-trip A -> B -> A is non-destructive and the user's BG-removal /
# CLAHE / frame-average work (minutes of compute, NOT on disk) survives.
# Each value: {"raw_signals", "positions", "dirty", "masks", "active",
# "calibration", "crops"}. Signal objects are stored by reference; a lazily-loaded
# kikuchipy signal keeps its own h5 handle open, so stashed datasets stay
# readable on restore.
_registry_by_file: dict = {}

# Stage-based load progress, keyed by client-supplied request_id (UUID4).
# Updated by _load_ebsd_blocking as it advances through 4 named stages,
# read by GET /load/progress/{request_id}. Protected by a threading.Lock
# because writes happen from the to_thread worker, reads from the asyncio
# event-loop thread. dict ops are atomic in CPython but we want a consistent
# snapshot for the API response.
_load_progress: dict = {}
_load_progress_lock = threading.Lock()
_LOAD_PROGRESS_TTL_S = 60.0  # drop completed/errored entries after this

# Progress for long pattern-processing ops (CLAHE etc.), keyed by a client-
# supplied request_id. The op runs in a worker thread (so the event loop can
# still serve the progress poll) and a dask callback ticks this dict per
# completed chunk. Read by GET /processing-progress/{request_id}.
_processing_progress: dict = {}
_processing_progress_lock = threading.Lock()
_PROCESSING_PROGRESS_TTL_S = 60.0


def _set_processing_progress(request_id: Optional[str], **state) -> None:
    """Merge a progress snapshot for a processing op; sweep stale entries.

    Fields: done, total, fraction (0..1), stage ('running'|'complete'|'error'),
    started_at, message, error. No-op when request_id is None.
    """
    if not request_id:
        return
    now = time.time()
    with _processing_progress_lock:
        stale = [
            rid for rid, e in _processing_progress.items()
            if e.get("stage") in ("complete", "error")
            and (now - e.get("started_at", now)) > _PROCESSING_PROGRESS_TTL_S
        ]
        for rid in stale:
            _processing_progress.pop(rid, None)
        entry = _processing_progress.get(request_id, {})
        entry.update(state)
        _processing_progress[request_id] = entry


def _make_dask_progress_callback(request_id: str, started_at: float):
    """Build a dask Callback that reports chunk completion to
    _processing_progress[request_id]. Returns a context manager; if dask isn't
    importable for some reason, returns a no-op context so processing still
    runs (just without granular progress)."""
    try:
        from dask.callbacks import Callback
    except Exception:
        import contextlib
        return contextlib.nullcontext()

    class _Cb(Callback):
        # Use _start_state (not len(dsk)) so the total counts only schedulable
        # tasks — matches dask's own ProgressBar so `done` actually reaches
        # `total` (len(dsk) overcounts fused/cached tasks, making the bar stall
        # at ~50% then jump to 100%).
        def _start_state(self, dsk, state):
            self._total = max(1, sum(len(state[k]) for k in
                                     ("ready", "waiting", "running", "finished")))
            _set_processing_progress(
                request_id, done=0, total=self._total, fraction=0.0,
                stage="running", started_at=started_at,
            )

        def _posttask(self, key, result, dsk, state, worker_id):
            done = len(state["finished"])
            _set_processing_progress(
                request_id, done=done, total=self._total,
                fraction=min(1.0, done / self._total),
                stage="running", started_at=started_at,
            )

    return _Cb()

# Per-dataset overview-image cache. Computing the navigation map (mean / std
# / max etc. across all patterns) on a lazy-loaded 25k-pattern dataset takes
# minutes on slow disks because every pattern has to be read from h5. Recomputing
# every call is wasteful — the underlying signal data doesn't change between
# overview requests for the same dataset+mode unless the user runs a processing
# op (BG removal, autocontrast, etc.), which dirties the dataset. We use the
# existing ``_dirty_datasets`` set to invalidate so the cache stays correct.
# Keyed by (dataset_name, mode). Cleared on /load and /switch-file via
# _raw_signals.clear() coupling.
_overview_cache: dict = {}


def _get_mask_state(dataset_name: str) -> dict:
    """Return the mask state for ``dataset_name``, materialising the default
    on first access (disabled, fraction=1.0)."""
    state = _signal_masks.get(dataset_name)
    if state is None:
        state = {"enabled": False, "radius_fraction": 1.0}
        _signal_masks[dataset_name] = state
    return state


def _compute_include_mask(signal, dataset_name: str) -> Optional[np.ndarray]:
    """Build the (H, W) bool inclusion mask for the given dataset.

    Returns ``None`` if masking is disabled for this dataset — callers should
    short-circuit on that to preserve bit-identical behaviour to before the
    mask feature existed.
    """
    state = _signal_masks.get(dataset_name)
    if not state or not state.get("enabled"):
        return None
    sig_shape = signal.axes_manager.signal_shape  # (cols, rows) hyperspy
    if len(sig_shape) < 2:
        return None
    pat_w = int(sig_shape[0])
    pat_h = int(sig_shape[1])
    cy = (pat_h - 1) / 2.0
    cx = (pat_w - 1) / 2.0
    # Inscribed circle = min(H, W) / 2. radius_fraction scales it.
    r_inscribed = min(pat_h, pat_w) / 2.0
    radius = float(state.get("radius_fraction", 1.0)) * r_inscribed
    radius = max(0.0, radius)
    yy, xx = np.ogrid[:pat_h, :pat_w]
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist_sq <= (radius * radius)


def get_active_include_mask() -> Optional[np.ndarray]:
    """Return the (H, W) bool inclusion mask for the active dataset, or None
    if masking is disabled. Exported so downstream indexing/refinement routes
    can opt-in to the same mask the viewer is using."""
    signal = _get_active_signal()
    if signal is None or not _active_dataset:
        return None
    return _compute_include_mask(signal, _active_dataset)


def _canonical_path(path: str) -> str:
    """Normalise a file path to a canonical key for de-duplication.

    The same file can reach the registry via different path strings — a
    relative ``Test_data/x.h5oina`` from one load and an absolute
    ``E:/.../Test_data/x.h5oina`` from another, or mixed ``/`` vs ``\\``
    separators. Without normalisation each spelling becomes a *separate*
    registry entry, so the file-switcher shows the same file two or three
    times. ``abspath`` + ``normpath`` + ``normcase`` collapses all of those
    spellings to one key (case-insensitive on Windows, which is correct for
    NTFS). Pure string/cwd math — no disk I/O, no symlink resolution — so it
    is safe to call on every comparison.
    """
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
    except Exception:
        return str(path)


def _register_loaded_file(path: str) -> None:
    """Add a file path to the loaded-files registry (idempotent, order-preserving).

    De-duplication is by *canonical* path (see ``_canonical_path``) so the
    same file loaded under different spellings is registered once.
    """
    from pathlib import Path as _P
    canon = _canonical_path(path)
    for entry in _loaded_files:
        if _canonical_path(entry["path"]) == canon:
            return
    _loaded_files.append({"path": path, "name": _P(path).stem})


def _update_progress(request_id: Optional[str], **state) -> None:
    """Write a progress snapshot for the given request_id.

    Snapshot fields: stage (str), stage_idx (int), stage_total (int),
    started_at (float, set by caller on first write), message (str).
    Optional: error (str) when stage='error'.

    No-op if request_id is None (caller didn't ask for progress tracking).
    Sweeps stale entries on each write.
    """
    if not request_id:
        return
    now = time.time()
    with _load_progress_lock:
        # Sweep stale entries first
        stale = [
            rid for rid, entry in _load_progress.items()
            if entry.get("stage") in ("complete", "error")
            and (now - entry.get("started_at", now)) > _LOAD_PROGRESS_TTL_S
        ]
        for rid in stale:
            _load_progress.pop(rid, None)
        # Apply the update — merge with any existing entry for this rid so
        # started_at is preserved across stage transitions.
        existing = _load_progress.get(request_id, {})
        existing.update(state)
        _load_progress[request_id] = existing


def _read_progress(request_id: str) -> Optional[dict]:
    """Return a snapshot dict for the given request_id, or None if unknown."""
    with _load_progress_lock:
        entry = _load_progress.get(request_id)
        if entry is None:
            return None
        snapshot = dict(entry)  # copy under the lock
    snapshot["elapsed_seconds"] = max(0.0, time.time() - snapshot.get("started_at", time.time()))
    return snapshot


def load_ebsd_file(path: str) -> bool:
    """Synchronous helper to load an EBSD file into the module state.

    Used by batch indexing to load files without going through the async route.
    Returns True on success, False on failure.
    """
    global _ebsd_signal, _ebsd_file_path, _active_dataset
    try:
        from safe_loader import load_ebsd_safe
        signal = load_ebsd_safe(path)
        # Same non-destructive contract as the interactive load path: stash the
        # outgoing file's processed datasets before clearing so batch indexing
        # can never silently wipe a user's BG/CLAHE/frame-average work. The
        # stash guard above skips raw-only files, so a pure batch run (raw
        # files, no processing) stays lightweight.
        _prev = _ebsd_file_path
        try:
            _stash_registry_for_file(_prev)
        except Exception:
            logger.warning("Could not stash registry for %s", _prev, exc_info=True)
        _ebsd_signal = signal
        _ebsd_file_path = path
        _raw_signals.clear()
        _positions.clear()
        _dirty_datasets.clear()
        _signal_masks.clear()
        dataset_name = Path(path).stem
        _raw_signals[dataset_name] = signal
        _positions[dataset_name] = (0, 0)
        _active_dataset = dataset_name
        try:
            _restore_registry_for_file(path, dataset_name)
        except Exception:
            logger.warning("Could not restore stashed datasets for %s", path, exc_info=True)
            _active_dataset = dataset_name
            _ebsd_signal = signal
        _register_loaded_file(path)
        return True
    except Exception as e:
        logger.exception(f"Failed to load EBSD file: {path}")
        return False


def _get_active_signal():
    """Return the active signal, falling back to legacy _ebsd_signal."""
    global _ebsd_signal
    if _active_dataset and _active_dataset in _raw_signals:
        return _raw_signals[_active_dataset]
    return _ebsd_signal


def get_active_crop_window():
    """The crop window of the ACTIVE dataset, or None if it is not a crop.

    This is the single question every consumer asks: "is what I am looking at
    a cut-out, and if so, where from?"
    """
    if not _active_dataset:
        return None
    return crop_window_service.get_crop(_active_dataset)


def is_active_signal_dirty() -> bool:
    """True if the active in-memory signal was modified in the viewer and so
    differs from the raw file on disk. Spherical indexing uses this to decide
    whether to index the processed patterns or re-read the original H5."""
    return bool(_active_dataset) and _active_dataset in _dirty_datasets


def _mark_active_dirty() -> None:
    """Flag the active dataset as modified in-memory.

    Also invalidates the overview cache for this dataset — any in-memory
    edit (BG removal, autocontrast, CLAHE) changes the pixel values so
    the cached navigation map is no longer correct.
    """
    if _active_dataset:
        _dirty_datasets.add(_active_dataset)
        # Drop every cached overview that's keyed to this dataset (one
        # per mode). Done explicitly here rather than at cache-read time
        # because invalidation-on-read would also need a "dataset dirty"
        # flag check, which is just this same work done lazily.
        for key in [k for k in _overview_cache if k[0] == _active_dataset]:
            _overview_cache.pop(key, None)


class LoadEBSDRequest(BaseModel):
    path: str
    use_kikuchipy: bool = True
    request_id: Optional[str] = None  # client-supplied UUID for progress tracking


class SwitchFileRequest(BaseModel):
    path: str


class PatternSelectionRequest(BaseModel):
    indices: List[int]  # Flat indices of selected patterns


class DeepCopyRequest(BaseModel):
    name: str = ""  # Name for the new dataset; auto-generated if empty


class CropRequest(BaseModel):
    shape: str = "rect"           # "rect" | "ellipse" | "lasso"
    row0: int
    col0: int
    rows: int
    cols: int
    mask: Optional[List[bool]] = None   # rows*cols, row-major, window-local
    name: str = ""
    materialise: bool = True


class FrameAverageRequest(BaseModel):
    window_size: int = 3


class BackgroundRemovalRequest(BaseModel):
    method: str = "dynamic"  # "dynamic" or "static"
    # Static background params
    static_bg_row: int = 0
    static_bg_col: int = 0


class SelectDatasetRequest(BaseModel):
    name: str


class OverviewRequest(BaseModel):
    mode: str = "mean"  # "mean", "std", "max"


class SignalMaskRequest(BaseModel):
    """Per-dataset circular detector mask state."""
    enabled: bool = False
    radius_fraction: float = 1.0  # 0.0–1.0 of the inscribed circle radius


class ClaheRequest(BaseModel):
    kernel_size: int = 8  # passed to kikuchipy as ``(kernel_size, kernel_size)``
    # Optional client-supplied UUID4. When given, the long CLAHE compute
    # reports chunk-level progress to _processing_progress[request_id], which
    # the frontend polls via GET /processing-progress/{request_id} to show a
    # real progress bar instead of an indeterminate spinner.
    request_id: Optional[str] = None


# A processed dataset bigger than this is NOT force-materialised into the
# stash — we keep it lazy and warn instead of risking an out-of-memory crash
# of the whole backend. 4 GiB comfortably covers realistic interactive EBSD
# scans (the giant 27 GB Oxford files aren't processed full-frame in the UI).
_STASH_MATERIALIZE_MAX_BYTES = 4 * 1024 ** 3


def _materialise_for_stash(name: str, sig):
    """Return a FILE-INDEPENDENT version of a processed signal.

    Derived datasets (deepcopy + in-place CLAHE / BG / frame-average) are
    usually still LAZY — their dask graph reads from the source file's open h5
    handle. If we stash such a signal and then switch files, the source raw
    signal is dropped and garbage-collected, its h5 file closes, and every
    later read of the derived signal fails — the "I switched once and nothing
    loads anymore" bug. Computing the lazy signal now (while the source handle
    is still open) snapshots it into a plain in-memory numpy array that no
    longer depends on any file, so it stays readable forever.

    Eager signals are returned untouched. Oversized lazy signals are kept lazy
    with a loud warning (better a possibly-stale large dataset than an OOM).
    Compute is done in-place on the same object that is about to be cleared
    from the live registry, so there is no extra copy.
    """
    if not getattr(sig, "_lazy", False):
        return sig
    try:
        nbytes = int(getattr(sig.data, "nbytes", 0))
    except Exception:
        nbytes = 0
    if nbytes and nbytes > _STASH_MATERIALIZE_MAX_BYTES:
        logger.warning(
            "Dataset '%s' (%.1f GB) exceeds the %.0f GB stash-materialise "
            "budget — keeping it lazy. It may not survive a file switch.",
            name, nbytes / 1024 ** 3,
            _STASH_MATERIALIZE_MAX_BYTES / 1024 ** 3,
        )
        return sig
    try:
        t0 = time.perf_counter()
        try:
            sig.compute(show_progressbar=False)
        except TypeError:
            sig.compute()  # older hyperspy without the kwarg
        logger.info("Materialised '%s' for stash in %.1fs (now file-independent)",
                    name, time.perf_counter() - t0)
    except Exception:
        logger.warning("Could not materialise '%s' for stash; keeping it lazy "
                        "(may not survive a switch)", name, exc_info=True)
    return sig


def _stash_registry_for_file(file_path) -> None:
    """Snapshot the current in-memory dataset registry under ``file_path``.

    Called just before the load pipeline clears the live registries, so a
    later switch BACK to this file can restore its processed datasets instead
    of finding them gone. No-op when the path is empty or nothing is loaded
    (so we never stash an empty registry that would later 'restore' over real
    data). Stores signal objects by reference — cheap, and they stay alive
    (and their h5 handles open) for as long as the stash holds them.
    """
    if not file_path or not _raw_signals:
        return
    # Only worth stashing when there is NON-REPRODUCIBLE work to preserve —
    # i.e. derived/processed datasets beyond the raw one (BG removal, CLAHE,
    # frame-average). A raw-only file just reloads from disk on switch-back,
    # so skip it; that also stops batch indexing (which loads many raw files
    # in a loop) from piling up open h5 handles in the stash.
    has_derived = len(_raw_signals) > 1 or bool(_dirty_datasets)
    if not has_derived:
        return
    key = _canonical_path(file_path)
    # Materialise the PROCESSED (dirty) datasets so they survive the source
    # file being closed on switch. The raw dataset is NOT materialised — it is
    # dropped on restore and re-read fresh from disk anyway.
    raw_snapshot = {}
    for name, sig in _raw_signals.items():
        if name in _dirty_datasets:
            raw_snapshot[name] = _materialise_for_stash(name, sig)
        else:
            raw_snapshot[name] = sig
    # Crop windows travel with the datasets they belong to. The load path
    # clears the whole crop registry (the live datasets go with it), so a
    # cropped dataset restored on switch-back would otherwise come back
    # looking like a full scan — and every consumer would map its pixels to
    # the wrong place in the original grid.
    crops = {}
    for name in _raw_signals:
        window = crop_window_service.get_crop(name)
        if window is not None:
            crops[name] = window
    _registry_by_file[key] = {
        "raw_signals": raw_snapshot,
        "positions": dict(_positions),
        "dirty": set(_dirty_datasets),
        "masks": {k: dict(v) for k, v in _signal_masks.items()},
        "active": _active_dataset,
        "calibration": calibration_store.snapshot(list(_raw_signals.keys())),
        "crops": crops,
    }
    logger.info(
        "Stashed %d dataset(s) for %s (derived: %s)",
        len(raw_snapshot), key,
        [n for n in _raw_signals if n in _dirty_datasets],
    )


def _restore_registry_for_file(file_path, fresh_raw_name: str) -> bool:
    """Restore a previously stashed registry for ``file_path``.

    Merges the stashed derived datasets back on top of the freshly re-read raw
    dataset (already registered under ``fresh_raw_name``). The fresh raw signal
    WINS over any stashed copy of the same name — it has live file handles and
    correct metadata, whereas the stashed raw may be a stale lazy signal. The
    valuable derived datasets (BG/CLAHE/frame-average) come back from the stash.

    Returns True if a stash existed for this file (and was consumed), False
    otherwise (a first-time-visited file — nothing to restore).
    """
    global _active_dataset, _ebsd_signal
    key = _canonical_path(file_path)
    snap = _registry_by_file.pop(key, None)
    if not snap:
        return False

    for name, sig in snap["raw_signals"].items():
        if name == fresh_raw_name:
            continue  # keep the freshly re-read raw signal
        _raw_signals[name] = sig
    for name, pos in snap["positions"].items():
        # Don't clobber the fresh raw's reset position; fill in the rest.
        _positions.setdefault(name, pos)
    _dirty_datasets.update(n for n in snap["dirty"] if n != fresh_raw_name)
    for name, mask in snap["masks"].items():
        _signal_masks[name] = dict(mask)
    calibration_store.restore(snap["calibration"], skip=fresh_raw_name)
    for name, window in snap["crops"].items():
        if name != fresh_raw_name:
            crop_window_service.set_crop(name, window)

    # Restore the previously-active dataset selection if it still exists
    # (e.g. the user was viewing 'Scan1_bg_clahe' when they switched away).
    if snap["active"] in _raw_signals:
        _active_dataset = snap["active"]
        _ebsd_signal = _raw_signals[_active_dataset]

    logger.info(
        "Restored %d stashed dataset(s) for %s; active='%s'",
        len(snap["raw_signals"]), key, _active_dataset,
    )
    return True


def _load_eds_only_blocking(path: str, probe: dict, request_id, started_at) -> dict:
    """Open a file that has EDS / electron images but no diffraction patterns.

    Everything the EDS page needs comes from ``h5_session``; the EBSD signal
    registries are cleared so no stale patterns from the previously loaded file
    can be mistaken for this one's. The response says ``has_patterns: False``
    and ``content_mode: "eds_only"`` so the UI can explain the missing pattern
    views instead of looking broken.
    """
    global _ebsd_signal, _ebsd_file_path, _active_dataset

    _update_progress(request_id, stage="building_signal", stage_idx=2,
                     stage_total=4, started_at=started_at,
                     message="No diffraction patterns — opening EDS data")

    _prev_file_path = _ebsd_file_path
    try:
        _stash_registry_for_file(_prev_file_path)
    except Exception:
        logger.warning("Could not stash dataset registry for %s", _prev_file_path,
                       exc_info=True)

    # No EBSD signal exists for this file. Clearing rather than keeping the
    # previous file's signal is deliberate: a stale pattern grid of the wrong
    # shape is far worse than an empty one.
    _ebsd_signal = None
    _ebsd_file_path = path
    _raw_signals.clear()
    _positions.clear()
    _dirty_datasets.clear()
    _signal_masks.clear()
    _overview_cache.clear()
    crop_window_service.clear_all()
    _active_dataset = None
    calibration_store.clear()

    try:
        from backend.api.routes import indexing as _idx_mod
        _idx_mod.reactivate_result_for_source(path)
    except Exception:
        logger.warning("Could not reactivate indexing result on EDS-only load",
                       exc_info=True)
    try:
        from backend.api.routes import analysis as _ana_mod
        _ana_mod.stash_dataset_for_file(_prev_file_path)
        _ana_mod.restore_dataset_for_file(path)
    except Exception:
        logger.warning("Could not stash/restore analysis dataset on EDS-only load",
                       exc_info=True)

    _update_progress(request_id, stage="detecting_features", stage_idx=3,
                     stage_total=4, started_at=started_at,
                     message="Reading EDS elements and electron images")

    has_eds = probe.get("has_eds", False)
    has_electron = probe.get("has_electron_images", False)
    eds_elements = list(probe.get("eds_elements", []))
    electron_images = list(probe.get("electron_images", []))
    try:
        from backend.api.services.h5_session import (
            open_file as h5_open, is_open as h5_is_open,
            get_active_extractor, close_file as h5_close, get_current_path,
        )
        if h5_is_open() and get_current_path() != path:
            h5_close()
        if not h5_is_open():
            h5_open(path)
        features = get_active_extractor().detect_available_features()
        ext_elements = [str(e) for e in features.get('eds_elements', [])]
        ext_images = [str(e) for e in features.get('electron_images', [])]
        # The extractor is the authority once open — it is also what the EDS
        # page itself queries. If it comes back empty while the probe saw data,
        # the EDS page will be empty too, so say so loudly rather than
        # reporting a rosy element list the viewer cannot actually serve.
        if not ext_elements and eds_elements:
            logger.warning(
                "h5_session found no EDS elements in %s although the file "
                "contains %d (%s) — the EDS page will be empty",
                Path(path).name, len(eds_elements), ", ".join(eds_elements[:5]),
            )
        else:
            eds_elements = ext_elements
        if not ext_images and electron_images:
            logger.warning(
                "h5_session found no electron images in %s although the file "
                "contains %s", Path(path).name, ", ".join(electron_images),
            )
        else:
            electron_images = ext_images
        has_eds = bool(eds_elements) or features.get('has_eds', has_eds)
        has_electron = bool(electron_images) or features.get('has_electron_images', has_electron)
    except Exception:
        logger.exception("h5_session setup for EDS-only file %s failed", path)

    _update_progress(request_id, stage="finalising", stage_idx=4,
                     stage_total=4, started_at=started_at, message="Finalising")
    _register_loaded_file(path)

    response = {
        "success": True,
        "dataset_name": Path(path).stem,
        "file_path": path,
        "format_type": "Oxford" if 'h5oina' in path.lower() else "EDAX",
        "pc_source": None,
        "pc_defaulted": False,
        "content_mode": "eds_only",
        "data_shape": [],
        "navigation_shape": [],
        "signal_shape": [],
        "grid_shape": [0, 0],
        "pattern_shape": [0, 0],
        "pattern_count": 0,
        "n_patterns": 0,
        "has_patterns": False,
        "has_raw_patterns": False,
        "has_eds": has_eds,
        "has_electron_images": has_electron,
        "eds_elements": eds_elements,
        "electron_images": electron_images,
        "request_id": request_id,
    }

    _update_progress(request_id, stage="complete", stage_idx=4, stage_total=4,
                     started_at=started_at, message="Complete")
    state_version.bump()
    return response


def _load_ebsd_blocking(path: str, request_id: Optional[str] = None) -> dict:
    """Synchronous portion of the EBSD load.

    Extracted from the route so it can be dispatched via asyncio.to_thread
    and not block the event loop. Mutates module-level state
    (_ebsd_signal, _raw_signals, _positions, _active_dataset, calibration_store,
    indexing._active_result_id, analysis._dataset) and returns the response
    dict that the route should send back to the client.

    If ``request_id`` is provided, stage-based progress snapshots are written
    to ``_load_progress[request_id]`` at four checkpoints (reading_metadata,
    building_signal, detecting_features, finalising) plus a final 'complete'
    state. Read by GET /load/progress/{request_id}.
    """
    global _ebsd_signal, _ebsd_file_path, _active_dataset

    started_at = time.time()
    _update_progress(request_id, stage="reading_metadata", stage_idx=1,
                     stage_total=4, started_at=started_at,
                     message="Reading file headers")

    # Some Aztec "Elementverteilungsdaten" acquisitions carry EDS maps and
    # SE/FSE images but no /<n>/EBSD group at all. Both loaders reject those
    # ("no top groups with subgroup name 'EBSD'"), which used to make the file
    # unopenable — even though the EDS viewer reads everything it needs through
    # h5_session and never touches the EBSD signal. Take the EDS-only route
    # instead of failing.
    from safe_loader import load_ebsd_safe, probe_ebsd_content

    _probe = probe_ebsd_content(path)
    if _probe.get("eds_only"):
        logger.info(
            "%s has no EBSD patterns — loading as EDS-only (%d elements, images: %s)",
            Path(path).name, len(_probe["eds_elements"]),
            ", ".join(_probe["electron_images"]) or "none",
        )
        return _load_eds_only_blocking(path, _probe, request_id, started_at)

    signal = load_ebsd_safe(path)

    _update_progress(request_id, stage="building_signal", stage_idx=2,
                     stage_total=4, started_at=started_at,
                     message="Building lazy signal")

    _prev_file_path = _ebsd_file_path  # capture BEFORE overwrite (for analysis stash)

    # STASH the outgoing file's processed-dataset registry BEFORE clearing it,
    # so a switch-back restores the user's BG-removal / CLAHE / frame-average
    # work instead of finding it garbage-collected. No-op on the very first
    # load (nothing loaded yet). This is the fix for the "all my processing
    # disappeared after switching files" data-loss bug. Guarded so a stash
    # failure can never break the load itself.
    try:
        _stash_registry_for_file(_prev_file_path)
    except Exception:
        logger.warning("Could not stash dataset registry for %s — that file's "
                       "processed datasets may be lost", _prev_file_path,
                       exc_info=True)

    _ebsd_signal = signal
    _ebsd_file_path = path

    # Clear the live registries and register the freshly-read raw dataset.
    # (The previous file's datasets are NOT lost — they were stashed above.)
    _raw_signals.clear()
    _positions.clear()
    _dirty_datasets.clear()
    _signal_masks.clear()
    _overview_cache.clear()
    crop_window_service.clear_all()
    dataset_name = Path(path).stem
    _raw_signals[dataset_name] = signal
    _positions[dataset_name] = (0, 0)
    _active_dataset = dataset_name

    # Register calibration (PC + detector geometry) in central store
    calibration_store.clear()  # New file → fresh store
    calibration_store.register(dataset_name, signal)

    # RESTORE this file's previously-stashed processed datasets (if we've
    # visited it before). Merges the derived datasets + their calibration +
    # the active selection back on top of the freshly re-read raw signal.
    # First visit → no-op, so a brand-new file shows only its raw dataset.
    # Guarded + fail-safe: if restore throws, fall back to the freshly-loaded
    # raw so the viewer ALWAYS has a readable active dataset (never "nothing
    # loads"). Worst case is the old behaviour (raw only), never a broken load.
    try:
        _restore_registry_for_file(path, dataset_name)
    except Exception:
        logger.warning("Could not restore stashed datasets for %s — showing "
                       "the raw dataset only", path, exc_info=True)
        _active_dataset = dataset_name
        _ebsd_signal = signal

    # Deactivate (but DON'T clear) the indexing result registry on file
    # load. The _active_result_id is reset so /pattern-match / /render
    # and friends fall back to the new file's defaults instead of using
    # an xmap from the previous file (wrong shape → "cannot reshape").
    # The registry itself is preserved so results indexed on the
    # previous file remain available; the user can re-activate them by
    # switching back to that file via the FileSwitcher and clicking the
    # gallery entry. Lazy-import to avoid a hard dependency cycle
    # between ebsd_viewer and indexing.
    # Re-activate the most recent indexing result that belongs to THIS file
    # (if any) so returning to a file you already indexed restores its phase
    # map automatically, instead of showing an empty one until you re-pick it
    # from the gallery. Results for other files stay in the registry but
    # deactivated; if the new file has none, this sets None (old behaviour).
    try:
        from backend.api.routes import indexing as _idx_mod
        _idx_mod.reactivate_result_for_source(path)
    except Exception:
        logger.warning("Could not reactivate indexing result on file load", exc_info=True)

    # Analysis dataset carries grain reconstruction and per-pixel results
    # keyed to a specific file's xmap. Rather than DESTROYING it on every file
    # change (which permanently lost grain results — the "my analysis
    # disappeared" symptom), stash it under the PREVIOUS file and restore any
    # dataset previously computed for the NEW file. This makes a file
    # round-trip (A -> B -> A) non-destructive, matching how the indexing
    # registry is preserved above. A fresh file with no prior analysis
    # restores to None, so /render still falls back to the new file's
    # defaults instead of a stale (wrong-shape) xmap.
    try:
        from backend.api.routes import analysis as _ana_mod
        _ana_mod.stash_dataset_for_file(_prev_file_path)
        _ana_mod.restore_dataset_for_file(path)
    except Exception:
        logger.warning("Could not stash/restore analysis dataset on file load", exc_info=True)

    # PC refinement state is now PER-FILE (see ``pcrefinement._sessions``,
    # 2026-05-26). We do NOT clear it on /load — switching files swaps
    # to that file's own session (patterns, detector, phase) which was
    # preserved from any previous time the user worked with the file.
    # The session-bound design means there is no risk of cross-file
    # state leak that the old global-controller path had.

    shape = signal.data.shape
    nav_shape = signal.axes_manager.navigation_shape
    sig_shape = signal.axes_manager.signal_shape

    # navigation_shape is (cols, rows) in hyperspy convention
    # grid_shape is (rows, cols) for the frontend
    n_rows = nav_shape[1] if len(nav_shape) >= 2 else 1
    n_cols = nav_shape[0] if len(nav_shape) >= 1 else 1
    pat_h = sig_shape[1] if len(sig_shape) >= 2 else sig_shape[0] if sig_shape else shape[-2]
    pat_w = sig_shape[0] if len(sig_shape) >= 2 else sig_shape[0] if sig_shape else shape[-1]

    _update_progress(request_id, stage="detecting_features", stage_idx=3,
                     stage_total=4, started_at=started_at,
                     message="Detecting EDS and electron-image features")

    # Try to detect EDS/electron features from H5 session
    has_eds = False
    has_electron = False
    eds_elements = []
    electron_images = []
    # EDAX UP1/UP2 are raw-pattern-only files (no HDF5 container, so no EDS or
    # electron images by definition). Skip the h5_session probe for them — it
    # would only fail on a non-HDF5 file and log an alarming traceback.
    _is_hdf5 = Path(path).suffix.lower() in (".h5", ".hdf5", ".h5oina")
    if _is_hdf5:
        try:
            from backend.api.services.h5_session import (
                open_file as h5_open, is_open as h5_is_open,
                get_active_extractor, close_file as h5_close, get_current_path,
            )
            # Reopen h5_session if it's pointing at a different file — otherwise
            # EDS / element / electron-image queries keep returning data for
            # the PREVIOUSLY loaded file. User-visible: load file A with EDS,
            # load file B, EDS page shows A's elements (BUG, 2026-04-21).
            if h5_is_open() and get_current_path() != path:
                h5_close()
            if not h5_is_open():
                h5_open(path)
            ext = get_active_extractor()
            features = ext.detect_available_features()
            has_eds = features.get('has_eds', False)
            has_electron = features.get('has_electron_images', False)
            eds_elements = [str(e) for e in features.get('eds_elements', [])]
            electron_images = [str(e) for e in features.get('electron_images', [])]
        except Exception:
            logger.exception("h5_session setup for %s failed — EDS/electron features may be unavailable", path)
    else:
        # Non-HDF5 (EDAX UP1/UP2): also make sure any h5_session left open by a
        # previously-loaded HDF5 file is closed, so stale EDS/element queries
        # don't return the old file's data after switching to a UP file.
        try:
            from backend.api.services.h5_session import (
                is_open as h5_is_open, close_file as h5_close,
                get_current_path,
            )
            if h5_is_open() and get_current_path() != path:
                h5_close()
        except Exception:
            logger.warning("Could not close stale h5_session on UP-file load", exc_info=True)

    _update_progress(request_id, stage="finalising", stage_idx=4,
                     stage_total=4, started_at=started_at,
                     message="Finalising")

    _register_loaded_file(path)

    # Pattern-centre provenance: EDAX UP1/UP2 files carry no PC, so we either
    # recovered the real one from the .osc sidecar ('osc') or fell back to
    # kikuchipy's placeholder (0.5,0.5,0.5) ('default'). The UI warns loudly on
    # 'default' so the user knows indexing is uncalibrated. HDF5 files carry
    # their own PC, so pc_source stays None (no banner).
    pc_source = None
    try:
        pc_source = signal.metadata.get_item("Signal.pc_source")
    except Exception:
        pc_source = None

    response = {
        "success": True,
        "dataset_name": dataset_name,
        "file_path": path,
        "format_type": "Oxford" if 'h5oina' in path.lower() else "EDAX",
        "pc_source": pc_source,
        "pc_defaulted": pc_source == "default",
        "data_shape": list(shape),
        "navigation_shape": list(nav_shape),
        "signal_shape": list(sig_shape),
        "grid_shape": [int(n_rows), int(n_cols)],
        "pattern_shape": [int(pat_h), int(pat_w)],
        "pattern_count": int(n_rows * n_cols),
        "n_patterns": int(n_rows * n_cols),
        "has_patterns": True,
        "has_raw_patterns": True,
        "has_eds": has_eds,
        "has_electron_images": has_electron,
        "eds_elements": eds_elements,
        "electron_images": electron_images,
        "request_id": request_id,
    }

    _update_progress(request_id, stage="complete", stage_idx=4,
                     stage_total=4, started_at=started_at,
                     message="Complete")

    # A successful load = a file switch → the active result/render changed.
    # Tell polling clients (e.g. the detached pole-figure window) to refetch.
    state_version.bump()

    return response


@router.post("/load")
async def load_ebsd(req: LoadEBSDRequest):
    """Load EBSD data from H5OINA/H5 file.

    Heavy work runs in a thread via asyncio.to_thread so the event loop
    stays responsive. If req.request_id is set, progress is reported via
    GET /load/progress/{request_id}.

    Note: other read endpoints may briefly observe intermediate state
    while this is running (e.g. /loaded-files between path-set and
    register, calibration_store between clear() and register()). This
    is intentional — locking would re-block the loop.
    """
    try:
        return await asyncio.to_thread(_load_ebsd_blocking, req.path, req.request_id)
    except Exception as e:
        if req.request_id:
            # Preserve started_at from earlier stages if it exists; only seed it
            # if this is the first write (error before stage 1 ever fired). Without
            # this, an error mid-load resets started_at to now() → elapsed_seconds
            # reads as ~0 and the caller can't see how long the load actually ran
            # before failing.
            with _load_progress_lock:
                had_started = (
                    req.request_id in _load_progress
                    and "started_at" in _load_progress[req.request_id]
                )
            err_kwargs = dict(stage="error", error=str(e),
                              stage_idx=0, stage_total=4,
                              message=f"Load failed: {e}")
            if not had_started:
                err_kwargs["started_at"] = time.time()
            _update_progress(req.request_id, **err_kwargs)
        logger.exception("Failed to load EBSD: %s", req.path)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/info")
async def ebsd_info():
    """Get info about currently loaded EBSD data.

    Returns ``{"loaded": false}`` with status 200 when no file is loaded —
    the frontend calls this on mount (pre-file-load) and treating that
    normal startup state as 400 produced console noise on every app start.
    """
    signal = _get_active_signal()
    if signal is None:
        return {"loaded": False}

    return {
        "loaded": True,
        "file_path": _ebsd_file_path,
        "active_dataset": _active_dataset,
        "data_shape": [int(v) for v in signal.data.shape],
        "dtype": str(signal.data.dtype),
    }


# Above this final-atlas size (uint8 bytes) we refuse to build the tiled atlas
# for a LAZY signal — materialising every pattern off disk would spike RAM and
# stall other requests. Eager (small, <2 GB) signals are already in RAM and are
# never gated. The frontend transparently falls back to per-pattern fetch.
_ATLAS_MAX_BYTES = 96 * 1024 * 1024  # 96 MB


@router.get("/pattern-atlas")
async def get_pattern_atlas(step: int = 4):
    """Return all patterns as a single tiled JPEG image (thumbnail atlas).

    Each pattern is subsampled by *step* (default 4 → 32×39 from 128×156).
    The atlas arranges thumbnails in a grid: rows top-to-bottom, cols left-to-right.
    Response includes atlas image, thumbnail dimensions, and grid shape so the
    frontend can extract any pattern with a simple canvas crop — zero latency.
    """
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    def _build_atlas() -> dict:
        from PIL import Image
        import io
        import base64

        data = signal.data  # expected (nrows, ncols, pat_h, pat_w)

        # The atlas is a drag-navigation OPTIMISATION only — the frontend
        # treats a missing atlas as "fall back to per-pattern fetch"
        # (EBSDViewer.fetchAtlas: `if (!d?.atlas) return`). So whenever we
        # can't or shouldn't build it, return ``atlas: None`` (HTTP 200)
        # instead of raising — the viewer stays fully functional.
        if data.ndim != 4:
            # 1-D-nav / 3-D signals (unified_loader fallback) don't tile into a
            # row/col grid here; skip rather than IndexError on data.shape[3].
            return {"atlas": None, "reason": f"unsupported nav shape {tuple(data.shape)}"}

        nrows, ncols = int(data.shape[0]), int(data.shape[1])
        pat_h, pat_w = int(data.shape[2]), int(data.shape[3])
        th = len(range(0, pat_h, step))
        tw = len(range(0, pat_w, step))

        # CRITICAL: never materialise a multi-GB lazy signal here. Building the
        # full atlas reads EVERY pattern off disk; on a large lazy dataset that
        # blew up RAM and (when this ran synchronously) jammed the event loop,
        # so every other page's request queued behind it. The per-pattern path
        # (`/pattern/{row}/{col}`, already to_thread'd, one chunk per read) is
        # the correct path for big data — signal the frontend to use it.
        is_lazy = hasattr(data, "chunks")
        atlas_bytes = nrows * ncols * th * tw  # final uint8 atlas footprint
        if is_lazy and atlas_bytes > _ATLAS_MAX_BYTES:
            return {
                "atlas": None,
                "reason": (
                    f"dataset too large for atlas "
                    f"({atlas_bytes / 1024**2:.0f} MB > "
                    f"{_ATLAS_MAX_BYTES / 1024**2:.0f} MB cap); "
                    "using per-pattern fetch"
                ),
            }

        # Materialise only the subsampled grid (np.asarray forces dask compute
        # for lazy signals; no-op for eager numpy).
        thumbs = np.asarray(data[:, :, ::step, ::step])  # (nrows, ncols, th, tw)

        if thumbs.dtype != np.uint8:
            tmin, tmax = float(np.nanmin(thumbs)), float(np.nanmax(thumbs))
            if tmax > tmin:
                thumbs = ((thumbs.astype(np.float32) - tmin) / (tmax - tmin) * 255).astype(np.uint8)
            else:
                thumbs = np.zeros_like(thumbs, dtype=np.uint8)

        # Rearrange into a single atlas image: (nrows*th, ncols*tw)
        atlas = thumbs.transpose(0, 2, 1, 3).reshape(nrows * th, ncols * tw)

        img = Image.fromarray(atlas, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        atlas_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "atlas": atlas_b64,
            "thumb_h": th,
            "thumb_w": tw,
            "grid_rows": nrows,
            "grid_cols": ncols,
            "pat_h": pat_h,
            "pat_w": pat_w,
        }

    try:
        # Off the event loop: the materialise+encode is pure CPU/IO and would
        # otherwise block every concurrent request (this is why switching into
        # the Indexing page hung while the viewer's atlas was building).
        return await asyncio.to_thread(_build_atlas)
    except Exception as e:
        logger.exception("Failed to build pattern atlas")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pattern/{row}/{col}")
async def get_ebsd_pattern(row: int, col: int, display_filter: str = "None"):
    """Get a single EBSD pattern from the loaded signal.

    Wrapped in ``asyncio.to_thread`` because the ``signal.data[row, col]``
    slice + ``np.asarray`` inside ``array_to_base64_raw`` triggers a dask
    chunk read from h5 on lazy-loaded signals. That can take seconds on a
    slow disk and would otherwise block every other endpoint (most
    importantly the progress-poll GET that the LoadProgressModal depends
    on). Bounds-checking stays on the sync side so a bad request fails
    fast without spinning up a thread.
    """
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    # Bounds-check before the numpy indexing. Otherwise signal.data[-1, col]
    # silently wraps to the last row and returns a pattern from the wrong
    # position. Same class as 2baa4d5 / 345c59a.
    nav_shape = signal.axes_manager.navigation_shape
    n_cols = int(nav_shape[0]) if len(nav_shape) >= 1 else 1
    n_rows = int(nav_shape[1]) if len(nav_shape) >= 2 else 1
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Position ({row},{col}) out of bounds for {n_rows}x{n_cols} grid",
        )

    active_dataset = _active_dataset

    def _read_pattern() -> dict:
        # Materialise the slice up front. On a lazy (dask) signal
        # ``signal.data[row, col]`` is an un-computed dask array; the display
        # filters (e.g. Dynamic BG → kikuchipy.remove_dynamic_background) and
        # scipy ops downstream need a real numpy array. Without this the
        # filtered-preview path 500'd on big lazy files (the Single-Pixel Phase
        # Test then showed "no pattern"). No-op for already-numpy signals.
        pattern = np.asarray(signal.data[row, col])
        if display_filter and display_filter != "None":
            from backend.api.services.image_utils import apply_display_filter
            pattern = apply_display_filter(pattern, display_filter)
        # If the user has the circular signal mask active, zero out the
        # corners in the rendered preview so what they see matches what
        # gets fed to autocontrast / indexing.
        include_mask = _compute_include_mask(signal, active_dataset)
        if include_mask is not None:
            pattern = np.where(include_mask, pattern, 0)
        return {
            "image": array_to_base64_raw(pattern),
            "row": row,
            "col": col,
            "shape": list(np.asarray(pattern).shape),
        }

    try:
        return await asyncio.to_thread(_read_pattern)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Pattern at ({row},{col}) not found")


@router.post("/select")
async def select_patterns(req: PatternSelectionRequest):
    """Select specific patterns for further processing (e.g., PC refinement)."""
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    nav_shape = signal.axes_manager.navigation_shape
    n_cols = int(nav_shape[0]) if len(nav_shape) >= 1 else 1
    n_rows = int(nav_shape[1]) if len(nav_shape) >= 2 else 1
    n_total = n_rows * n_cols

    patterns = []
    for idx in req.indices:
        # Skip negative or oversized indices upfront — without this check
        # numpy silently wraps idx=-1 → last row, returning a different
        # pattern than requested.
        if not (0 <= idx < n_total):
            continue
        row = idx // n_cols
        col = idx % n_cols
        try:
            pat = signal.data[row, col]
            patterns.append({
                "index": idx,
                "row": row,
                "col": col,
                "image": array_to_base64_raw(pat),
            })
        except IndexError:
            continue

    return {"patterns": patterns, "count": len(patterns)}


def _build_detector_dict(det):
    """Extract all detector fields into a dict."""
    try:
        raw_shape = det.shape if hasattr(det, 'shape') else ()
        shape = [int(x) for x in (raw_shape if hasattr(raw_shape, '__iter__') else [raw_shape])]
    except (TypeError, ValueError):
        shape = []
    try:
        raw_pc = det.pc if hasattr(det, 'pc') else None
        if raw_pc is None:
            pc = []
        else:
            import numpy as np
            arr = np.asarray(raw_pc)
            # If per-pixel PC (shape (rows, cols, 3) or (n, 3)), return mean PC
            if arr.ndim > 1:
                flat = arr.reshape(-1, arr.shape[-1])
                pc = [float(x) for x in flat.mean(axis=0)]
            else:
                pc = [float(x) for x in arr.flatten()]
    except (TypeError, ValueError):
        pc = []
    sample_tilt = float(det.sample_tilt) if hasattr(det, 'sample_tilt') else 70.0
    camera_tilt = float(det.tilt) if hasattr(det, 'tilt') else 0.0
    azimuthal = float(det.azimuthal) if hasattr(det, 'azimuthal') else 0.0
    binning = int(det.binning) if hasattr(det, 'binning') else 1
    pixel_size = None
    try:
        pixel_size = float(det.pixel_size) if hasattr(det, 'pixel_size') else None
    except Exception:
        pass

    # Build human-readable repr
    lines = []
    if len(shape) >= 2:
        lines.append(f"Shape: {shape[0]} x {shape[1]} px")
    elif len(shape) == 1:
        lines.append(f"Shape: {shape[0]} px")
    if len(pc) == 3:
        lines.append(f"PC: ({pc[0]:.5f}, {pc[1]:.5f}, {pc[2]:.5f})")
    lines.append(f"Sample tilt: {sample_tilt:.4f} deg")
    lines.append(f"Camera tilt: {camera_tilt:.2f} deg")
    if azimuthal != 0.0:
        lines.append(f"Azimuthal: {azimuthal:.1f} deg")
    lines.append(f"Binning: {binning}")
    if pixel_size is not None:
        lines.append(f"Pixel size: {pixel_size:.2f} um")
    repr_str = "\n".join(lines)

    return {
        "has_detector": True,
        "shape": shape,
        "pc": pc,
        "sample_tilt": sample_tilt,
        "camera_tilt": camera_tilt,
        "azimuthal": azimuthal,
        "binning": binning,
        "pixel_size": pixel_size,
        "repr": repr_str,
    }


def _build_axes_repr(signal):
    """Build human-readable axes manager string."""
    lines = []
    try:
        am = signal.axes_manager
        for ax in am.navigation_axes[::-1]:  # reversed: y first, then x
            name = getattr(ax, 'name', '?')
            size = int(getattr(ax, 'size', 0))
            scale = float(getattr(ax, 'scale', 1.0))
            units = getattr(ax, 'units', 'px') or 'px'
            lines.append(f"{name}: {size} @ {scale:.4f} {units}/px")
        for ax in am.signal_axes[::-1]:
            name = getattr(ax, 'name', '?')
            size = int(getattr(ax, 'size', 0))
            scale = float(getattr(ax, 'scale', 1.0))
            units = getattr(ax, 'units', 'px') or 'px'
            lines.append(f"{name}: {size} @ {scale:.4f} {units}/px")
    except Exception:
        pass
    return "\n".join(lines) if lines else "No axes info"


def _extract_step_size(signal, file_path=None):
    """Extract step size in microns from axes manager, with H5OINA fallback."""
    UNDEFINED = ("<undefined>", "undefined", "", None)

    def _valid(val, units):
        return val is not None and float(val) > 0 and float(val) != 1.0 or (
            str(units) not in UNDEFINED and val not in (None, 0)
        )

    try:
        am = signal.axes_manager
        nav = am.navigation_axes
        if len(nav) >= 2:
            sx = float(nav[0].scale)
            sy = float(nav[1].scale)
            units = str(getattr(nav[0], 'units', 'um') or 'um')
            if str(units) not in UNDEFINED and sx > 0:
                return {"x": sx, "y": sy, "units": units}
        elif len(nav) == 1:
            sx = float(nav[0].scale)
            units = str(getattr(nav[0], 'units', 'um') or 'um')
            if str(units) not in UNDEFINED and sx > 0:
                return {"x": sx, "y": sx, "units": units}
    except Exception:
        pass

    # Fallback: read directly from H5OINA file. EDS-only acquisitions have no
    # EBSD header at all, so try the EDS and Electron Image headers too —
    # otherwise the scale bar stays greyed out on a file that plainly carries
    # its pixel size (X Step = 0.1815 um).
    fp = file_path or _ebsd_file_path
    if fp and fp.lower().endswith('.h5oina'):
        try:
            import h5py
            with h5py.File(fp, 'r') as f:
                for area in ('EBSD', 'EDS', 'Electron Image'):
                    for entry in f.keys():
                        hdr = f.get(f'{entry}/{area}/Header')
                        if hdr is None:
                            continue
                        for xkey in ['X Step', 'Step X', 'StepX', 'x_step']:
                            if xkey in hdr:
                                sx = float(np.ravel(hdr[xkey][()])[0])
                                sy_val = hdr.get('Y Step', hdr.get('Step Y', hdr.get('StepY')))
                                sy = float(np.ravel(sy_val[()])[0]) if sy_val is not None else sx
                                if sx > 0:
                                    return {"x": sx, "y": sy, "units": "um"}
        except Exception:
            pass

    # No geometry anywhere. Returning 1 um/px here used to look like a valid
    # scale and produced silently wrong scale bars; None makes the UI disable
    # the scale bar instead.
    return None


def _pixel_sizes_from_session():
    """Per-area pixel size from the open h5 session, or {} when unavailable."""
    try:
        from backend.api.services.h5_session import is_open as h5_is_open, get_active_extractor
        if h5_is_open():
            return get_active_extractor().get_pixel_sizes() or {}
    except Exception:
        logger.warning("Could not read per-area pixel sizes", exc_info=True)
    return {}


def _eds_only_metadata(path: str) -> dict:
    """Metadata for a file that carries EDS / electron images but no patterns.

    Same keys the normal response uses, so the store and every consumer work
    unchanged — just with empty pattern geometry.
    """
    pixel_sizes = _pixel_sizes_from_session()
    step = _extract_step_size(None, path)
    if step is None:
        # Prefer whichever area does report a scale.
        for key in ("eds", "electron_image", "ebsd"):
            ps = pixel_sizes.get(key)
            if ps:
                step = {"x": ps["x"], "y": ps["y"], "units": ps.get("units", "um")}
                break

    has_eds, eds_elements = False, []
    grid = [0, 0]
    try:
        from backend.api.services.h5_session import is_open as h5_is_open, get_active_extractor
        if h5_is_open():
            ext = get_active_extractor()
            features = ext.detect_available_features()
            has_eds = features.get("has_eds", False)
            eds_elements = [str(e) for e in features.get("eds_elements", [])]
            grid = [int(v) for v in features.get("grid_shape", (0, 0))]
    except Exception:
        logger.warning("Could not read EDS features for %s", path, exc_info=True)

    return {
        "loaded": True,
        "content_mode": "eds_only",
        "detector": {"has_detector": False},
        "axes_repr": "",
        "step_size": step,
        "pixel_sizes": pixel_sizes,
        "grid_shape": grid,
        "pattern_shape": [0, 0],
        "pattern_count": 0,
        "beam_energy": _extract_beam_energy(path),
        "format_type": "Oxford" if 'h5oina' in path.lower() else "EDAX",
        "file_path": path,
        "has_eds": has_eds,
        "eds_elements": eds_elements,
        "phases": [],
    }


def _extract_beam_energy(file_path):
    """Try to extract beam energy (kV) from H5OINA header."""
    try:
        import h5py
        with h5py.File(file_path, 'r') as f:
            # Oxford H5OINA: /1/EBSD/Header/Beam Voltage or similar
            for key in f.keys():
                header = f.get(f'{key}/EBSD/Header')
                if header is None:
                    continue
                for bv_key in ['Beam Voltage', 'BeamVoltage', 'Accelerating Voltage']:
                    if bv_key in header:
                        val = header[bv_key][()]
                        if hasattr(val, 'item'):
                            val = val.item()
                        return float(val)
    except Exception:
        pass
    return None


@router.get("/detector")
async def get_detector():
    """Get detector info from loaded EBSD signal.

    Returns ``{"has_detector": false}`` with status 200 when no file is loaded
    (consistent with the case where a file is loaded but has no detector).
    Previously returned 400 which produced console noise on every app start.
    """
    import json as _json
    try:
        signal = _get_active_signal()
        if signal is None:
            return {"has_detector": False}

        # Prefer CalibrationStore (survives deepcopy), fallback to signal.detector
        det = calibration_store.get_detector(_active_dataset)
        if det is None:
            det = getattr(signal, 'detector', None)
        if det is None:
            return {"has_detector": False}

        result = _build_detector_dict(det)
        result["axes_repr"] = _build_axes_repr(signal)
        # Force all values to JSON-safe native Python types (avoid numpy scalar serialization errors)
        return _json.loads(_json.dumps(result, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Detector endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata")
async def get_metadata():
    """Central metadata endpoint — returns all known info about the loaded EBSD dataset.

    This is the single source of truth for all modules (PC Refinement, Simulation,
    Analysis, Phase Maps, Indexing) to read detector geometry, step sizes, beam energy, etc.

    Returns ``{"loaded": false}`` with status 200 when no file is loaded, matching
    the pattern of /info and /detector so frontend mount-time calls produce no
    console noise.
    """
    signal = _get_active_signal()
    if signal is None:
        # An EDS-only acquisition has no EBSD signal but still has geometry,
        # elements and images. Returning a bare {"loaded": false} left the
        # store without a step size, which greys out the export scale bar.
        if _ebsd_file_path:
            return _eds_only_metadata(_ebsd_file_path)
        return {"loaded": False, "detector": {"has_detector": False}}

    # Detector — prefer CalibrationStore (survives deepcopy)
    det = calibration_store.get_detector(_active_dataset)
    if det is None:
        det = getattr(signal, 'detector', None)
    detector = _build_detector_dict(det) if det else {"has_detector": False}

    # Axes
    axes_repr = _build_axes_repr(signal)
    step_size = _extract_step_size(signal, _ebsd_file_path)

    # Grid & pattern shape
    nav_shape = signal.axes_manager.navigation_shape
    sig_shape = signal.axes_manager.signal_shape
    n_rows = nav_shape[1] if len(nav_shape) >= 2 else 1
    n_cols = nav_shape[0] if len(nav_shape) >= 1 else 1
    pat_h = sig_shape[1] if len(sig_shape) >= 2 else sig_shape[0] if sig_shape else 0
    pat_w = sig_shape[0] if len(sig_shape) >= 2 else 0

    # Beam energy from H5 file
    beam_energy = _extract_beam_energy(_ebsd_file_path) if _ebsd_file_path else None

    # EDS / features from H5 session
    has_eds = False
    eds_elements = []
    try:
        from backend.api.services.h5_session import is_open as h5_is_open, get_active_extractor
        if h5_is_open():
            ext = get_active_extractor()
            features = ext.detect_available_features()
            has_eds = features.get('has_eds', False)
            eds_elements = [str(e) for e in features.get('eds_elements', [])]
    except Exception:
        # get_active_extractor() raises on a crop/file grid mismatch. Without
        # this line that fail-loud degrades to a silent "this file has no EDS".
        logger.warning("Could not read EDS features for the active dataset — "
                       "reporting no EDS", exc_info=True)

    # Phases from signal if available
    phases = []
    try:
        if hasattr(signal, 'xmap') and signal.xmap is not None:
            for p in signal.xmap.phases:
                phases.append(str(p.name))
    except Exception:
        pass

    return {
        "detector": detector,
        "axes_repr": axes_repr,
        "step_size": step_size,
        # Per-area pixel size. The electron images live on the SEM raster, the
        # maps on the scan raster — measured 10.6x apart on a real file — so a
        # scale bar must use the area its image came from, not `step_size`.
        "pixel_sizes": _pixel_sizes_from_session(),
        "grid_shape": [int(n_rows), int(n_cols)],
        "pattern_shape": [int(pat_h), int(pat_w)],
        "pattern_count": int(n_rows * n_cols),
        "beam_energy": beam_energy,
        "format_type": "Oxford" if _ebsd_file_path and 'h5oina' in _ebsd_file_path.lower() else "EDAX",
        "file_path": _ebsd_file_path,
        "has_eds": has_eds,
        "eds_elements": eds_elements,
        "phases": phases,
    }


# --- Multi-dataset endpoints ---

@router.get("/loaded-files")
async def get_loaded_files():
    """List every H5OINA/H5 file the user has loaded this session.

    The UI uses this to offer a file switcher. `active` marks the file
    currently open. Switching is done via POST /switch-file.
    """
    active_canon = _canonical_path(_ebsd_file_path) if _ebsd_file_path else None
    return {
        "files": [
            {**entry, "active": _canonical_path(entry["path"]) == active_canon}
            for entry in _loaded_files
        ],
        "active_path": _ebsd_file_path,
    }


@router.post("/switch-file")
async def switch_file(req: SwitchFileRequest):
    """Switch the active file by re-loading it from disk.

    Re-uses the full load_ebsd pipeline (which resets indexing results,
    analysis dataset, calibration, PC controller, and the h5_session).
    The file must already be in the loaded-files registry — otherwise
    use POST /load.
    """
    req_canon = _canonical_path(req.path)
    known = any(_canonical_path(e["path"]) == req_canon for e in _loaded_files)
    if not known:
        raise HTTPException(
            status_code=404,
            detail=f"File not in loaded-files registry: {req.path}. Use /load for new files.",
        )
    return await load_ebsd(LoadEBSDRequest(path=req.path))


@router.post("/loaded-files/remove")
async def remove_loaded_file(req: SwitchFileRequest):
    """Remove a single file from the loaded-files registry.

    Lets the user prune the file-switcher list (it otherwise only grows for
    the lifetime of the backend process). The *active* file cannot be
    removed — the user must switch to another file first, since removing the
    open file would leave the viewer pointing at nothing. Matching is by
    canonical path so any spelling of the same file resolves correctly.
    """
    req_canon = _canonical_path(req.path)
    if _ebsd_file_path and _canonical_path(_ebsd_file_path) == req_canon:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove the active file. Switch to another file first.",
        )
    before = len(_loaded_files)
    _loaded_files[:] = [
        e for e in _loaded_files if _canonical_path(e["path"]) != req_canon
    ]
    if len(_loaded_files) == before:
        raise HTTPException(
            status_code=404,
            detail=f"File not in loaded-files registry: {req.path}",
        )
    return {"success": True, "removed": req.path, "remaining": len(_loaded_files)}


@router.post("/loaded-files/clear")
async def clear_loaded_files():
    """Drop every entry from the loaded-files registry except the active file.

    The active file is kept so the viewer stays usable; everything else is
    cleared in one click instead of removing entries one by one.
    """
    active_canon = _canonical_path(_ebsd_file_path) if _ebsd_file_path else None
    _loaded_files[:] = [
        e for e in _loaded_files if _canonical_path(e["path"]) == active_canon
    ]
    return {"success": True, "remaining": len(_loaded_files)}


@router.post("/loaded-files/clear-all")
async def clear_all_loaded_files():
    """Empty the entire loaded-files registry, including the active file.

    Unlike /loaded-files/clear (which keeps the active file so the switcher
    stays usable), this drops *every* entry — the one-click "reset" for when
    the list has accumulated files the user no longer cares about, without
    having to kill and restart the backend process.

    The active file stays *open* in the viewer (its in-memory signal, indexing
    results, etc. are untouched); it simply no longer appears in the switcher
    list until it is loaded again.
    """
    _loaded_files.clear()
    return {"success": True, "remaining": 0}


@router.get("/datasets")
async def list_datasets():
    """List all loaded datasets."""
    # Check if EDS data is available from the H5 session
    has_eds = False
    eds_elements = []
    try:
        from backend.api.services.h5_session import is_open, get_active_extractor
        if is_open():
            ext = get_active_extractor()
            elems = ext.get_available_elements()
            has_eds = len(elems) > 0
            eds_elements = [str(e) for e in elems]
    except Exception:
        # Same reason as in get_metadata(): a crop/file grid mismatch must not
        # look like "this file has no EDS" with nothing in the log.
        logger.warning("Could not read EDS elements for the active dataset — "
                       "reporting no EDS", exc_info=True)

    datasets = []
    for name, sig in _raw_signals.items():
        nav_shape = sig.axes_manager.navigation_shape
        sig_shape = sig.axes_manager.signal_shape
        is_active = name == _active_dataset
        datasets.append({
            "name": name,
            "active": is_active,
            "navigation_shape": list(nav_shape),
            "signal_shape": list(sig_shape),
            "n_patterns": int(np.prod(nav_shape)) if nav_shape else sig.data.shape[0],
            "dtype": str(sig.data.dtype),
            "has_eds": has_eds and is_active,
            "eds_elements": eds_elements if is_active else [],
        })
    return {"datasets": datasets, "active": _active_dataset, "file_path": _ebsd_file_path}


@router.post("/select-dataset")
async def select_dataset(req: SelectDatasetRequest):
    """Switch to a different loaded dataset."""
    global _ebsd_signal, _active_dataset
    if req.name not in _raw_signals:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.name}' not found")
    _active_dataset = req.name
    _ebsd_signal = _raw_signals[req.name]
    return {"success": True, "active_dataset": _active_dataset}


@router.post("/deepcopy")
async def deepcopy_dataset(req: DeepCopyRequest):
    """Create a deep copy of the current signal as a new named dataset."""
    global _active_dataset
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    try:
        import copy

        parent_name = _active_dataset
        new_signal = copy.deepcopy(signal)

        # Auto-generate a name if not provided
        base_name = req.name.strip() if req.name.strip() else (_active_dataset or "dataset")
        new_name = base_name
        counter = 1
        while new_name in _raw_signals:
            new_name = f"{base_name}_copy{counter}"
            counter += 1

        _raw_signals[new_name] = new_signal
        _positions[new_name] = (0, 0)
        # Inherit mask state from parent so visual + processing behavior stays
        # consistent across the copy.
        if parent_name and parent_name in _signal_masks:
            _signal_masks[new_name] = dict(_signal_masks[parent_name])

        # Inherit calibration from parent (CalibrationStore is immune to deepcopy bug)
        calibration_store.register_derived(new_name, parent_name)

        nav_shape = new_signal.axes_manager.navigation_shape
        logger.info("Created deepcopy: '%s' from '%s'", new_name, _active_dataset)
        return {
            "success": True,
            "name": new_name,
            "parent": _active_dataset,
            "navigation_shape": list(nav_shape),
            "n_patterns": int(np.prod(nav_shape)) if nav_shape else new_signal.data.shape[0],
        }
    except Exception as e:
        logger.exception("Deepcopy failed")
        raise HTTPException(status_code=500, detail=str(e))


# A crop pulled into RAM must not blow the process up. Same ceiling the
# per-file stash uses, from the same constant, so the two limits cannot drift.
_CROP_MATERIALIZE_MAX_BYTES = _STASH_MATERIALIZE_MAX_BYTES


@router.post("/crop")
async def crop_dataset(req: CropRequest):
    """Cut the active dataset down to a drawn selection.

    The EBSD side is one ``inav`` slice — kikuchipy carries the per-pixel PC
    map and the xmap along with it (see ``_update_custom_attributes`` in
    kikuchipy/signals/ebsd.py). Everything else this endpoint does is
    bookkeeping so the rest of the app knows where the cut-out came from.
    """
    global _active_dataset, _ebsd_signal

    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    nav_shape = signal.axes_manager.navigation_shape
    n_cols = int(nav_shape[0]) if len(nav_shape) >= 1 else 1
    n_rows = int(nav_shape[1]) if len(nav_shape) >= 2 else 1

    r0, c0, rows, cols = int(req.row0), int(req.col0), int(req.rows), int(req.cols)
    if rows <= 0 or cols <= 0:
        raise HTTPException(
            status_code=400, detail=f"Empty selection: {rows}x{cols} pixels")
    if r0 < 0 or c0 < 0 or r0 + rows > n_rows or c0 + cols > n_cols:
        raise HTTPException(
            status_code=400,
            detail=(f"The selection (rows {r0}-{r0 + rows}, cols {c0}-{c0 + cols}) "
                    f"does not fit the scan ({n_rows}x{n_cols})"),
        )

    nav_mask = None
    if req.mask is not None:
        arr = np.asarray(req.mask, dtype=bool)
        if arr.size != rows * cols:
            raise HTTPException(
                status_code=400,
                detail=(f"The mask has {arr.size} entries but the selection is "
                        f"{rows}x{cols} = {rows * cols} pixels"),
            )
        nav_mask = arr.reshape(rows, cols)
        if not nav_mask.any():
            raise HTTPException(
                status_code=400, detail="The selection contains no pixels")

    parent_name = _active_dataset
    parent_window = crop_window_service.get_crop(parent_name)

    def _cut():
        # hyperspy indexes navigation as [x, y] — columns first.
        new = signal.inav[c0:c0 + cols, r0:r0 + rows]
        nbytes = int(np.prod(new.data.shape)) * int(new.data.dtype.itemsize)
        materialised = False
        if req.materialise and nbytes <= _CROP_MATERIALIZE_MAX_BYTES:
            if hasattr(new, "compute"):
                try:
                    new.compute(show_progressbar=False)
                    materialised = True
                except TypeError:
                    new.compute()
                    materialised = True
            else:
                materialised = True   # already in memory
        return new, nbytes, materialised

    try:
        new_signal, nbytes, materialised = await asyncio.to_thread(_cut)
    except Exception as e:
        logger.exception("Crop failed")
        raise HTTPException(status_code=500, detail=str(e))

    base_name = req.name.strip() or f"{parent_name or 'dataset'}_crop"
    new_name = base_name
    counter = 1
    while new_name in _raw_signals:
        new_name = f"{base_name}{counter}"
        counter += 1

    local_window = crop_window_service.CropWindow(
        source_file=str(_ebsd_file_path or ""),
        row0=r0, col0=c0, rows=rows, cols=cols,
        original_shape=(n_rows, n_cols),
        shape_kind=req.shape,
        nav_mask=nav_mask,
    )
    window = (crop_window_service.compose(parent_window, local_window)
              if parent_window is not None else local_window)

    # Register BEFORE publishing the dataset. register_cropped cuts the
    # parent's per-pixel PC map and raises when that map is not on the grid
    # this window was cut from; publishing first would leave a dataset in
    # _raw_signals that IS a crop but carries no crop window — the "looks
    # like a full scan" state the whole window registry exists to prevent.
    # Everything after this point is dict writes, which cannot fail.
    try:
        calibration_store.register_cropped(new_name, parent_name, local_window)
    except ValueError as e:
        parent_entry = calibration_store.get_entry(parent_name)
        pc_grid = (tuple(np.shape(parent_entry.pc_map)[:2])
                   if parent_entry is not None and parent_entry.pc_map is not None
                   else None)
        logger.warning("Crop registration refused for '%s': %s", new_name, e)
        raise HTTPException(
            status_code=400,
            detail=(f"The per-pixel pattern-centre map of '{parent_name}' is on a "
                    f"{pc_grid} grid but the selection was drawn on a "
                    f"{tuple(local_window.original_shape)} grid — reload the file "
                    f"before cropping"),
        )
    crop_window_service.set_crop(new_name, window)

    _raw_signals[new_name] = new_signal
    _positions[new_name] = (0, 0)
    if parent_name and parent_name in _signal_masks:
        _signal_masks[new_name] = dict(_signal_masks[parent_name])
    # A crop is ALWAYS dirty, even when its parent is clean: "dirty" means the
    # in-memory patterns differ from the raw file on disk, and a crop's do — by
    # grid, if not by value. Without this, is_active_signal_dirty() is False and
    # indexing re-reads the file, i.e. indexes the full scan instead of the
    # cut-out the user drew.
    _dirty_datasets.add(new_name)

    _active_dataset = new_name
    _ebsd_signal = new_signal
    # The name is fresh in _raw_signals, but delete_dataset does not purge the
    # overview cache, so a recycled name can still find a stale entry here.
    for key in [k for k in _overview_cache if k[0] == new_name]:
        _overview_cache.pop(key, None)

    logger.info(
        "Cropped '%s' -> '%s': %dx%d px (%d selected), %.1f MB, materialised=%s",
        parent_name, new_name, rows, cols, window.n_selected,
        nbytes / 1e6, materialised,
    )
    return {
        "success": True,
        "name": new_name,
        "parent": parent_name,
        "window": window.to_dict(),
        "n_patterns": rows * cols,
        "n_selected": window.n_selected,
        "materialised": materialised,
        "bytes": nbytes,
    }


@router.get("/crop")
async def get_crop_window():
    """The active dataset's crop window, or null when it is not a crop."""
    window = get_active_crop_window()
    return {
        "dataset": _active_dataset,
        "window": window.to_dict() if window is not None else None,
    }


@router.delete("/dataset/{name}")
async def delete_dataset(name: str):
    """Remove a dataset from memory to free RAM."""
    global _active_dataset, _ebsd_signal
    if name not in _raw_signals:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    del _raw_signals[name]
    _positions.pop(name, None)
    _signal_masks.pop(name, None)
    _dirty_datasets.discard(name)
    calibration_store.remove(name)
    crop_window_service.clear_crop(name)
    logger.info("Deleted dataset: '%s'", name)

    # If deleted the active dataset, switch to the first remaining one
    if _active_dataset == name:
        if _raw_signals:
            _active_dataset = next(iter(_raw_signals))
            _ebsd_signal = _raw_signals[_active_dataset]
        else:
            _active_dataset = ""
            _ebsd_signal = None

    remaining = [
        {"name": n, "navigation_shape": list(s.axes_manager.navigation_shape)}
        for n, s in _raw_signals.items()
    ]
    return {"success": True, "deleted": name, "active": _active_dataset, "datasets": remaining}


@router.post("/frame-average")
async def frame_average(req: FrameAverageRequest):
    """Apply frame (neighbor pattern) averaging to the active dataset in-place."""
    global _ebsd_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    size = max(1, req.window_size)

    def _run():
        # kikuchipy API: average_neighbour_patterns (British spelling in kp 0.11+)
        if hasattr(signal, 'average_neighbour_patterns'):
            signal.average_neighbour_patterns(window="rectangular", window_shape=(size, size), show_progressbar=False)
        else:
            signal.average_neighbor_patterns(window="rectangular", window_shape=(size, size), show_progressbar=False)

    try:
        # Full-dataset neighbour averaging — run off the event loop.
        await asyncio.to_thread(_run)

        # Keep legacy reference in sync
        if _active_dataset and _active_dataset in _raw_signals:
            _ebsd_signal = _raw_signals[_active_dataset]

        _mark_active_dirty()
        return {"success": True, "window_size": size}
    except Exception as e:
        logger.exception("Frame averaging failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/background-removal")
async def background_removal(req: BackgroundRemovalRequest):
    """Apply background removal to the active dataset in-place."""
    global _ebsd_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    import time
    t0 = time.perf_counter()
    n_patterns = int(np.prod(signal.axes_manager.navigation_shape))
    logger.info("BG-%s start: %d patterns", req.method, n_patterns)

    # Validate up front (fast, on the event loop). For static BG, the
    # reference pattern must be inside the scan — previously an out-of-range
    # (r,c) or any kikuchipy failure was silently swapped for the mean of ALL
    # patterns, so the data fed to indexing was processed differently than the
    # user asked, with a green "success". Fail loud instead.
    if req.method == "static":
        nav = signal.axes_manager.navigation_shape  # (n_cols, n_rows)
        n_cols = int(nav[0]) if len(nav) >= 1 else 1
        n_rows = int(nav[1]) if len(nav) >= 2 else 1
        if not (0 <= req.static_bg_row < n_rows and 0 <= req.static_bg_col < n_cols):
            raise HTTPException(
                status_code=400,
                detail=f"Static-background reference pattern "
                       f"({req.static_bg_row},{req.static_bg_col}) is outside the "
                       f"scan {n_rows}x{n_cols}.",
            )
    elif req.method != "dynamic":
        raise HTTPException(status_code=400, detail=f"Unknown background removal method: {req.method}")

    def _run():
        if req.method == "dynamic":
            signal.remove_dynamic_background(
                operation="subtract",
                filter_domain="frequency",
                show_progressbar=False,
            )
            # When the user has a circular mask active, the dynamic-BG
            # Gaussian leaves a bright halo at the disc edge because the
            # dark corners bleed into the convolution. Zero them out here so
            # the halo doesn't end up in the displayed pattern or in the
            # NCC computation downstream.
            include_mask = _compute_include_mask(signal, _active_dataset)
            if include_mask is not None:
                exclude = ~include_mask
                if exclude.any():
                    # signal.data is a dask array when loaded lazily and dask
                    # arrays are read-only. Materialise into a numpy array
                    # in-place (hyperspy's compute() flips ._lazy and replaces
                    # .data) before the in-place write below.
                    if getattr(signal, '_lazy', False):
                        signal.compute()
                    signal.data[..., exclude] = 0
        else:  # static — reference validated above; no silent mean fallback
            static_bg = signal.data[req.static_bg_row, req.static_bg_col].astype(signal.data.dtype)
            signal.remove_static_background(
                operation="subtract",
                static_bg=static_bg,
                show_progressbar=False,
            )

    try:
        # Full-dataset dask op — run off the event loop (mirrors /clahe).
        await asyncio.to_thread(_run)

        if _active_dataset and _active_dataset in _raw_signals:
            _ebsd_signal = _raw_signals[_active_dataset]

        _mark_active_dirty()
        elapsed = time.perf_counter() - t0
        rate = n_patterns / max(elapsed, 1e-6)
        logger.info("BG-%s done: %d patterns in %.1fs (%.0f patterns/s)",
                    req.method, n_patterns, elapsed, rate)
        return {"success": True, "method": req.method,
                "elapsed_seconds": round(elapsed, 2),
                "n_patterns": n_patterns}
    except Exception as e:
        logger.exception("Background removal failed after %.1fs",
                         time.perf_counter() - t0)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/autocontrast")
async def autocontrast():
    """Apply batch auto-contrast (rescale intensities) to the active dataset in-place.

    When a signal mask is active the contrast stretch is driven by the
    P2/P98 percentiles of pixels INSIDE the mask only. That stops the dark
    detector corners from dragging the lower percentile down and wasting
    most of the output range.
    """
    global _ebsd_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    include_mask = _compute_include_mask(signal, _active_dataset)

    def _run():
        if include_mask is not None and include_mask.any():
            # Drive the stretch from percentiles inside the disc only.
            # Use the global P2/P98 of masked pixels for relative scaling —
            # matches ``relative=True`` semantics (preserves inter-pattern
            # intensity differences) but skips the dark corners.
            data = signal.data
            # data is (rows, cols, H, W); broadcast mask across nav axes
            masked = data[..., include_mask]
            p_lo = float(np.percentile(masked, 2))
            p_hi = float(np.percentile(masked, 98))
            if p_hi > p_lo:
                signal.rescale_intensity(
                    in_range=(p_lo, p_hi),
                    show_progressbar=False,
                )
            else:
                # Degenerate range (e.g. constant patterns) — fall back to
                # plain relative rescale; this also matches legacy behaviour.
                signal.rescale_intensity(relative=True, show_progressbar=False)
        else:
            signal.rescale_intensity(relative=True, show_progressbar=False)

    try:
        # Full-dataset percentile + rescale — run off the event loop.
        await asyncio.to_thread(_run)

        if _active_dataset and _active_dataset in _raw_signals:
            _ebsd_signal = _raw_signals[_active_dataset]

        _mark_active_dirty()
        return {"success": True, "masked": include_mask is not None}
    except Exception as e:
        logger.exception("Autocontrast failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clahe")
async def clahe(req: ClaheRequest):
    """Apply Contrast-Limited Adaptive Histogram Equalization to every pattern.

    Sharpens Kikuchi bands by spreading intensity values per-tile. This is
    the kikuchipy-recommended final step in the preprocessing pipeline:
    static BG → dynamic BG → CLAHE.

    Progress is reported via `logger.info` (auto-broadcast to the dev-panel
    log via `install_ws_log_handler` in main.py) — the frontend's
    `doProcessing` also logs start/finish per step in the bottom log so
    the user has two channels for progress.
    """
    global _ebsd_signal
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")
    k = int(req.kernel_size)
    if k < 2 or k > 64:
        raise HTTPException(status_code=400, detail=f"kernel_size must be 2–64; got {k}")
    t0 = time.perf_counter()
    started_at = time.time()
    n_patterns = int(np.prod(signal.axes_manager.navigation_shape))
    logger.info("CLAHE start: %d patterns, kernel=%d", n_patterns, k)
    _set_processing_progress(req.request_id, done=0, total=1, fraction=0.0,
                             stage="running", started_at=started_at,
                             message=f"CLAHE (kernel={k})")

    # CLAHE goes through dask (signal.map, chunked). Run it in a worker thread
    # so the event loop stays free to answer the progress poll, and hook a
    # dask callback to tick chunk-level progress into _processing_progress.
    def _run():
        if req.request_id:
            with _make_dask_progress_callback(req.request_id, started_at):
                signal.adaptive_histogram_equalization(kernel_size=(k, k))
        else:
            signal.adaptive_histogram_equalization(kernel_size=(k, k))

    try:
        await asyncio.to_thread(_run)
        if _active_dataset and _active_dataset in _raw_signals:
            _ebsd_signal = _raw_signals[_active_dataset]
        _mark_active_dirty()
        elapsed = time.perf_counter() - t0
        rate = n_patterns / max(elapsed, 1e-6)
        logger.info("CLAHE done: %d patterns in %.1fs (%.0f patterns/s)",
                    n_patterns, elapsed, rate)
        _set_processing_progress(req.request_id, fraction=1.0, stage="complete",
                                 started_at=started_at,
                                 elapsed_seconds=round(elapsed, 2))
        return {"success": True, "kernel_size": k,
                "elapsed_seconds": round(elapsed, 2),
                "n_patterns": n_patterns}
    except Exception as e:
        logger.exception("CLAHE failed after %.1fs", time.perf_counter() - t0)
        _set_processing_progress(req.request_id, stage="error", error=str(e),
                                 started_at=started_at)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing-progress/{request_id}")
async def get_processing_progress(request_id: str):
    """Snapshot of a long processing op's progress (CLAHE etc.).

    Returns {found, stage, done, total, fraction, elapsed_seconds, message,
    error}. `found=False` before the first tick or after the TTL sweep — the
    frontend treats that as "no granular progress yet" and keeps the spinner.
    """
    with _processing_progress_lock:
        entry = _processing_progress.get(request_id)
        snap = dict(entry) if entry else None
    if snap is None:
        return {"found": False}
    snap["found"] = True
    snap["elapsed_seconds"] = round(max(0.0, time.time() - snap.get("started_at", time.time())), 1)
    return snap


@router.get("/signal-mask")
async def get_signal_mask():
    """Return the current mask state for the active dataset."""
    if not _active_dataset:
        return {"enabled": False, "radius_fraction": 1.0}
    state = _get_mask_state(_active_dataset)
    return {
        "enabled": bool(state.get("enabled", False)),
        "radius_fraction": float(state.get("radius_fraction", 1.0)),
        "dataset": _active_dataset,
    }


@router.post("/signal-mask")
async def set_signal_mask(req: SignalMaskRequest):
    """Configure the circular detector mask for the active dataset."""
    if not _active_dataset:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")
    rf = float(req.radius_fraction)
    if not (0.0 <= rf <= 1.5):  # allow slight overflow for safety, clamp below
        raise HTTPException(status_code=400, detail=f"radius_fraction must be 0.0–1.5; got {rf}")
    rf = min(rf, 1.5)
    rf = max(rf, 0.0)
    state = _get_mask_state(_active_dataset)
    state["enabled"] = bool(req.enabled)
    state["radius_fraction"] = rf
    return {"success": True, "enabled": state["enabled"], "radius_fraction": rf, "dataset": _active_dataset}


# Overview modes that map to a per-pixel quality array Aztec already stored
# in the H5OINA (tiny contiguous datasets, ~175 KB for a 175k-pixel scan).
# Reading these is ~3 ms regardless of media speed, versus minutes of
# streaming every pattern off a lazy signal (one HDF5 chunk per pattern, so
# the reduction reads the whole multi-GB pattern array as ~175k tiny dask
# tasks). These metrics are intrinsic to the measurement and unaffected by
# our in-app preprocessing, so they're safe to read straight from disk.
# Each value is a list of candidate dataset names tried in priority order.
# See session 2026-06-01 (overview never appeared on a 3.3 GB lazy file).
_STORED_OVERVIEW_DATASETS = {
    "bc": ["Band Contrast"],
    "std": ["Band Slope", "Pattern Quality"],
    "snr": ["Pattern Quality"],
}

# Hard cap on how many patterns the lazy compute path reads for an overview
# when no stored array exists (mean / max / sharpness / entropy / ncc). Above
# this the scan is strided down to ~this many patterns and the coarse result
# upscaled — bounds the cost to a few seconds instead of minutes.
_OVERVIEW_PATTERN_CAP = 4000


def _read_stored_quality_map(file_path, mode, n_rows, n_cols):
    """Read Aztec's precomputed per-pixel quality array for ``mode`` from the
    source H5OINA and return it shaped ``(n_rows, n_cols)`` as float.

    Returns ``None`` (never raises) when the file/vendor/dataset doesn't
    provide it or the shape doesn't match — callers fall back to computing
    the overview from the patterns.
    """
    candidates = _STORED_OVERVIEW_DATASETS.get(mode)
    if not candidates or not file_path:
        return None
    try:
        import h5py
        with h5py.File(file_path, "r") as f:
            # Header group name varies by vendor/version ('1', '2', ...).
            for top in f:
                grp = f.get(f"{top}/EBSD/Data")
                if grp is None:
                    continue
                for name in candidates:
                    if name in grp:
                        arr = np.asarray(grp[name][:], dtype=float)
                        if arr.size != n_rows * n_cols:
                            return None
                        return arr.reshape(n_rows, n_cols)
        return None
    except Exception:
        logger.debug("stored quality map read failed", exc_info=True)
        return None


def _reduce_nav(data, mode):
    """Reduce the pattern stack ``data`` (..., sig_h, sig_w) to a per-pixel
    navigation map for the given overview ``mode``. Works on numpy or dask
    arrays; on dask the result is still lazy until materialised by the caller.
    """
    if mode == "mean":
        return data.mean(axis=(-2, -1))
    elif mode == "std":
        return data.std(axis=(-2, -1))
    elif mode == "max":
        return data.max(axis=(-2, -1))
    elif mode == "sharpness":
        # Sharpness approximation: horizontal gradient variance (fast)
        dx = np.diff(data.astype(np.float32), axis=-1)
        return dx.var(axis=(-2, -1))
    elif mode == "snr":
        # Signal-to-Noise ratio via MAD (vectorized)
        median_map = np.median(data, axis=(-2, -1))
        mad = np.median(np.abs(data - median_map[..., None, None]), axis=(-2, -1))
        mad[mad == 0] = 1
        return data.mean(axis=(-2, -1)) / (mad * 1.4826)
    elif mode == "entropy":
        # Fast entropy proxy: std (true histogram entropy too slow for 10k+)
        return data.std(axis=(-2, -1)).astype(float)
    return data.mean(axis=(-2, -1))


def _upscale_nearest(small, n_rows, n_cols):
    """Nearest-neighbour upscale a coarse (rr, cc) nav map to (n_rows, n_cols)."""
    rr, cc = small.shape
    ri = np.minimum((np.arange(n_rows) * rr) // n_rows, rr - 1)
    ci = np.minimum((np.arange(n_cols) * cc) // n_cols, cc - 1)
    return small[np.ix_(ri, ci)]


@router.get("/overview")
async def overview(mode: str = "mean"):
    """
    Get an overview map of the scan (mean/std/max intensity per pixel).
    Returns a base64-encoded PNG image of the navigation map.

    For lazy-loaded signals the reductions below read every pattern from
    disk (350 MB+ for a 25k-pattern dataset) and force a dask compute.
    Without ``asyncio.to_thread`` this happens on the event-loop thread
    and blocks every other endpoint — most importantly the progress-poll
    GET that the LoadProgressModal depends on — for the full duration
    (3+ minutes was observed on a slow USB disk, session 2026-05-27).
    Running it in a thread releases the event loop so progress polls,
    /pattern, /metadata and friends stay responsive while the overview
    cooks in the background.
    """
    signal = _get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")

    active_dataset = _active_dataset

    # Cache lookup — invalidated by _mark_active_dirty() on every
    # processing op and by _overview_cache.clear() on /load + /switch-file.
    cache_key = (active_dataset, mode)
    cached = _overview_cache.get(cache_key)
    if cached is not None:
        return cached

    def _compute_overview() -> dict:
        nav_shape = signal.axes_manager.navigation_shape  # (cols, rows)
        n_cols = int(nav_shape[0]) if len(nav_shape) >= 1 else 1
        n_rows = int(nav_shape[1]) if len(nav_shape) >= 2 else 1

        # --- Fast path: Aztec's precomputed per-pixel quality array ---
        # Only for the primary, unmodified dataset (a processed copy or an
        # intensity-derived mode falls through to live compute below).
        nav_map = None
        sampled = False
        if (_ebsd_file_path
                and active_dataset == Path(_ebsd_file_path).stem
                and active_dataset not in _dirty_datasets):
            nav_map = _read_stored_quality_map(_ebsd_file_path, mode, n_rows, n_cols)
        native_hit = nav_map is not None

        # --- Computed Band Contrast: native BC else kikuchipy FFT image quality.
        # No std/mean CoV fake. bc is intercepted here so it never reaches
        # _reduce_nav (which has no bc branch anymore).
        if nav_map is None and mode == "bc":
            from backend.api.services.pattern_quality import compute_image_quality
            data = signal.data
            is_lazy = hasattr(data, "chunks")
            n_pat = n_rows * n_cols
            if is_lazy and n_pat > _OVERVIEW_PATTERN_CAP:
                stride = int(np.ceil((n_pat / _OVERVIEW_PATTERN_CAP) ** 0.5))
                sub = signal.inav[::stride, ::stride]
                small = np.asarray(compute_image_quality(sub)).astype(float)
                nav_map = _upscale_nearest(small, n_rows, n_cols)
                sampled = True
            else:
                nav_map = np.asarray(compute_image_quality(signal)).astype(float)

        if nav_map is None:
            data = signal.data  # (n_rows, n_cols, sig_h, sig_w)
            is_lazy = hasattr(data, "chunks")
            n_pat = n_rows * n_cols
            if is_lazy and n_pat > _OVERVIEW_PATTERN_CAP:
                # Stride the scan down to ~_OVERVIEW_PATTERN_CAP patterns so a
                # lazy signal (one HDF5 chunk per pattern) reads a small subset
                # instead of the whole multi-GB array. Coarse but bounded; the
                # result is upscaled back to the full grid.
                stride = int(np.ceil((n_pat / _OVERVIEW_PATTERN_CAP) ** 0.5))
                small = _reduce_nav(data[::stride, ::stride], mode)
                small = np.asarray(small).astype(float)
                nav_map = _upscale_nearest(small, n_rows, n_cols)
                sampled = True
            else:
                nav_map = _reduce_nav(data, mode)

        # Normalise to uint8 for PNG. ``.min()`` / ``.max()`` force the
        # dask compute when ``nav_map`` is still a dask array (lazy load
        # path); on eager-loaded signals this is just a numpy reduction.
        nav_f = np.asarray(nav_map).astype(float)
        mn, mx = nav_f.min(), nav_f.max()
        if mx > mn:
            nav_u8 = ((nav_f - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            nav_u8 = np.zeros_like(nav_f, dtype=np.uint8)

        return {
            "image": array_to_base64_raw(nav_u8),
            "mode": mode,
            "shape": list(nav_f.shape),
            "min": float(mn),
            "max": float(mx),
            "best_pos": [int(x) for x in np.unravel_index(nav_f.argmax(), nav_f.shape)],
            "worst_pos": [int(x) for x in np.unravel_index(nav_f.argmin(), nav_f.shape)],
            "dataset": active_dataset,
            "sampled": sampled,
            "source": "native" if native_hit else "computed",
            "metric": (
                "band_contrast" if (native_hit and mode == "bc")
                else ("image_quality" if mode == "bc" else mode)
            ),
        }

    try:
        result = await asyncio.to_thread(_compute_overview)
        # Only cache successful results — failures shouldn't get pinned.
        _overview_cache[cache_key] = result
        return result
    except Exception as e:
        logger.exception("Overview failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load/progress/{request_id}")
async def get_load_progress(request_id: str):
    """Return current stage of an in-flight or recently-completed load.

    Stages: reading_metadata, building_signal, detecting_features,
    finalising, complete, error.

    Returns 404 if request_id is unknown (load never started or entry
    evicted after 60 s TTL).
    """
    snapshot = _read_progress(request_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}")
    return snapshot
