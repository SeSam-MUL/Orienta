"""
Manages the open HDF5 file session.

This is a singleton-like session that holds the currently open H5 file
and its data extractor. The GUI only works with one file at a time.
Includes an LRU pattern cache for fast navigation.
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Serialises session swaps (open/close) against the accessors. Reentrant so
# open_file() -> close_file() nesting is safe. Held only briefly (ref swaps +
# the file open), never during the actual data reads, so it can't deadlock a
# slow reader.
_lock = threading.RLock()

# When a new file is opened, the previous h5py.File is NOT closed immediately:
# a concurrent request (e.g. an EDS probe running in a worker thread) may still
# hold the old extractor/file and be mid-read. Closing it underneath them threw
# "invalid location identifier". Instead we hand the old handle off here; it
# stays open until the last reference is dropped (h5py closes it on GC). We keep
# a small bounded backlog and close the oldest — by then, several opens later,
# no live request can still hold it.
_pending_close: list = []
_PENDING_CLOSE_MAX = 4

# Lazy-load h5py to speed up backend startup (~150ms saved)
h5py = None

def _h5py():
    global h5py
    if h5py is None:
        import h5py as _mod
        h5py = _mod
    return h5py

# Pattern cache — avoids re-reading from disk on repeated navigation
_pattern_cache: OrderedDict = OrderedDict()
_PATTERN_CACHE_SIZE = 500  # Cache up to 500 patterns

# Global session state
_current_file = None
_current_extractor = None
_current_path: Optional[str] = None
# measurement -> (the h5py.File it was made on, its extractor); see get_variant_extractor
_variant_extractors: dict = {}


def get_format_type(h5file) -> str:
    """Detect if file is Oxford H5OINA or EDAX HDF5."""
    for key in h5file.keys():
        if isinstance(h5file[key], _h5py().Group):
            if f'{key}/EBSD' in h5file:
                # Check for Oxford-specific markers
                header_path = f'{key}/EBSD/Header'
                if header_path in h5file:
                    header = h5file[header_path]
                    if 'X Cells' in header:
                        return 'Oxford'
                    elif 'nColumns' in header:
                        return 'EDAX'
    logger.warning("Could not detect HDF5 format from file structure, defaulting to 'Oxford'")
    return 'Oxford'


def open_file(path: str):
    """Open an HDF5 file and create the data extractor."""
    global _current_file, _current_extractor, _current_path

    from tools.h5_viewer_backend import H5OINADataExtractor

    # Open + build the extractor OUTSIDE the lock (disk I/O can be slow), then
    # swap the session refs atomically under the lock.
    h5file = _h5py().File(path, 'r')
    fmt = get_format_type(h5file)
    try:
        extractor = H5OINADataExtractor(h5file, fmt)
    except Exception:
        h5file.close()
        raise

    with _lock:
        close_file()  # defers the old file's physical close (reentrant lock)
        _current_file = h5file
        _current_extractor = extractor
        _current_path = path

    logger.info("Opened HDF5 file: %s (format: %s)", path, fmt)

    # Restore any phase-map classification the user produced for this
    # file (and the measurement the viewer holds) in a previous session.
    # Best-effort — a corrupted sidecar
    # just leaves the store empty (the user can re-run auto-classify),
    # it must NEVER fail the file open.
    try:
        global _phase_map_ctx
        from backend.api.services.phase_map_store import get_phase_map_store
        measurement = active_measurement()
        _phase_map_ctx = (_context_path(path), measurement)
        if get_phase_map_store().load_from_disk(path, measurement):
            logger.info("Phase-map sidecar restored for %s", path)
    except Exception:
        logger.debug("phase_map_store.load_from_disk() failed during open_file", exc_info=True)

    # Start evaluating the spectrum cube, if the file has one, in the
    # background (eds_spectrum_service). Best-effort like the sidecar above.
    start_spectrum_fit(path)

    return extractor, fmt


def start_spectrum_fit(path: str) -> None:
    """Start (or retry) the EDS spectrum fit of ``path``. Never raises.

    ``open_file`` calls this, and so must every loader that finds the session
    already holding ``path`` and therefore skips ``open_file``: loading a file
    again is the user's retry of a fit that failed. A no-op for a job that is
    running or ready, and for a file without a cube. A failure to START is
    logged and never fails the load -- the first EDS read starts the job
    again -- while a failure INSIDE the job is recorded there and served to
    every EDS read as a 500 with code eds_spectrum_fit_failed.
    """
    try:
        from backend.api.services import eds_spectrum_service
        eds_spectrum_service.start_for_file(path)
    except Exception:
        logger.warning("could not start the EDS spectrum fit for %s", path,
                       exc_info=True)


def active_measurement() -> Optional[str]:
    """The measurement of the open file the EBSD viewer's active dataset reads (None: the file's
    default measurement, the slot the session itself reads, or nothing is open).

    What derived per-measurement state is keyed on: a phase map made of one measurement's EDS
    chemistry is not the phase map of another."""
    with _lock:
        extractor, path = _current_extractor, _current_path
    if extractor is None or not path:
        return None
    try:
        from backend.api.routes.ebsd_viewer import active_variant
    except ImportError:
        return None
    variant = active_variant(path)
    measurement = None if variant is None else variant.measurement
    if measurement is None or measurement == extractor.root_key:
        return None
    return str(measurement)


#: (file, measurement) the phase-map store was last matched to, by ``open_file`` or by
#: :func:`sync_phase_map_store`. It tells "the store is empty because the user cleared it" from "the store
#: is empty because it was never matched to what is loaded"; only the second reads a sidecar.
_phase_map_ctx: Optional[tuple] = None


def _phase_map_ctx_reset() -> None:
    global _phase_map_ctx
    _phase_map_ctx = None


def _context_path(path) -> str:
    import os
    try:
        return os.path.normcase(os.path.realpath(str(path)))
    except (OSError, ValueError):
        return str(path)


def sync_phase_map_store() -> bool:
    """Make the phase-map store hold the map of the file and the measurement the active dataset reads.

    The store keeps one map. Loading another measurement of the SAME file does not reopen the session
    (the path is unchanged), so the map of the measurement that was just left stayed on screen, was
    paired with the new measurement's at-% maps by the export, and a first open of the new measurement
    restored the sidecar the old one had left. Called after every load; a no-op when the store already
    holds the map for what is loaded. The map left behind is not lost: every edit is saved to the
    sidecar of its own measurement, which is read again when the user comes back to it.

    Returns True when the store was changed."""
    path = get_current_path()
    if not path:
        return False
    try:
        from backend.api.routes import ebsd_viewer
        from backend.api.services.phase_map_store import get_phase_map_store
        from backend.api.services.variant_registry import same_file
        viewer_path = ebsd_viewer._ebsd_file_path
        if not viewer_path or not same_file(viewer_path, path):
            return False            # the session was opened for something else than the viewer's file
        global _phase_map_ctx
        store = get_phase_map_store()
        wanted = active_measurement()
        held = store.file_path()
        context = (_context_path(path), wanted)
        if store.is_set():
            if not held:
                return False        # a map made without a file (nothing to compare): not ours to drop
            if same_file(held, path) and store.measurement() == wanted:
                _phase_map_ctx = context
                return False
        elif _phase_map_ctx == context:
            # Nothing held, and the store was already matched to this very file and measurement: the
            # user cleared the map on purpose ("start from scratch"); the sidecar is not read back.
            return False
        store.clear()
        _phase_map_ctx = context
        if store.load_from_disk(path, wanted):
            logger.info("Phase-map sidecar restored for %s (measurement %s)", path, wanted or "default")
        return True
    except Exception:
        logger.debug("phase-map store sync failed (ignored)", exc_info=True)
        return False


def get_extractor():
    """Get the current data extractor. Raises if no file is open."""
    with _lock:
        if _current_extractor is None:
            raise RuntimeError("No HDF5 file is currently open")
        return _current_extractor


def _extractor_path(extractor) -> Optional[str]:
    """The file an extractor reads, taken from the extractor itself.

    Read off the very object being wrapped rather than from
    ``get_current_path()``: those are two separate lock acquisitions, and a
    file opened between them would pair file B's extractor with file A's
    spectrum fit. Anything without a real h5py handle (test doubles) has no
    path, and so no spectrum fit.
    """
    h5 = getattr(extractor, "h5file", None)
    if h5 is None:
        return None
    try:
        name = h5.filename
    except Exception:
        return None
    return name if isinstance(name, str) and name else None


def _with_spectrum_source(extractor):
    """Wrap the raw extractor in a SpectrumNetExtractor when its file has a
    spectrum cube, so every EDS map read gets net intensities.

    Not wrapped: a file without a cube (window path, unchanged), and anything
    run inside ``eds_spectrum_service.force_window_source()``.
    """
    from backend.api.services import eds_spectrum_service as spectrum
    if spectrum.window_source_forced():
        return extractor
    path = _extractor_path(extractor)
    if path is None or not spectrum.spectrum_applies(path):
        return extractor
    return spectrum.SpectrumNetExtractor(extractor, path)


def get_variant_extractor():
    """The session's extractor read as the MEASUREMENT the EBSD viewer's active dataset holds.

    The session opens a file by path and its extractor reads the first slot that holds EBSD data.
    A file can offer several measurements (groups ``/1``, ``/2``, ...) and the viewer holds one of
    them: its EDS maps, electron images, Euler/Phase, headers and static backgrounds are that
    measurement's, not the first one's. When the viewer's file is the open file and its dataset
    names a measurement other than the session's slot, this returns an extractor on the SAME open
    handle for that measurement (cached; the shared session itself is not touched, so the HDF5
    viewer page, which opens files on its own, keeps what it opened). Otherwise it is
    ``get_extractor()`` itself, the same object: a dataset loaded without a choice, or on the
    default measurement, reads as it always did.
    """
    extractor = get_extractor()
    try:
        from backend.api.routes.ebsd_viewer import active_variant
    except ImportError:
        logger.debug("ebsd_viewer not importable — no loaded variant", exc_info=True)
        return extractor
    with _lock:
        path, h5file = _current_path, _current_file
    variant = active_variant(path) if path else None
    measurement = None if variant is None else variant.measurement
    if measurement is None or measurement == extractor.root_key:
        return extractor
    with _lock:
        cached = _variant_extractors.get(measurement)
        if cached is not None and cached[0] is h5file:
            return cached[1]
    from tools.h5_viewer_backend import H5OINADataExtractor
    made = H5OINADataExtractor(h5file, extractor.format, measurement=measurement)
    with _lock:
        _variant_extractors[measurement] = (h5file, made)
    return made


def get_active_extractor():
    """The extractor as the ACTIVE DATASET sees it.

    Two optional layers on the raw extractor, in this order:

    1. a ``SpectrumNetExtractor`` when the file has a per-pixel spectrum cube
       (``eds_spectrum_service``): EDS element maps are net line intensities,
       and while the fit runs every map read answers HTTP 503. The fit is made of
       the file's default measurement (the slot the session reads); a dataset
       loaded as another measurement of the file (``get_variant_extractor``)
       reads that measurement's window maps, never the default one's nets;
    2. a ``CroppedExtractor`` when the active dataset is a crop, cutting every
       grid-shaped read -- net maps included -- to the crop window.

    Without either (no cube, no crop, default measurement) it is the very same
    object ``get_extractor()`` returns -- that path is unchanged, not merely
    equivalent.

    Use this from anything that works on "the dataset the user is looking at".
    Use ``get_extractor()`` from anything that must show the FILE as it is —
    the h5 viewer, format probing, preflight checks.
    """
    raw = get_extractor()
    extractor = get_variant_extractor()
    if extractor is raw:
        extractor = _with_spectrum_source(extractor)

    # Lazy import: ebsd_viewer imports services, so a module-level import here
    # would close a cycle. Same pattern as close_file's phase_map_store import.
    #
    # ONLY the import is allowed to fail soft. Swallowing an error out of
    # get_active_crop_window() would return the FULL-SCAN extractor for a
    # dataset that IS a crop, and every switched route would then serve
    # full-scan data with nothing in the log — exactly the silently-wrong
    # mode this design exists to prevent. Let anything else propagate.
    try:
        from backend.api.routes.ebsd_viewer import (
            get_active_crop_window, _canonical_path,
        )
    except ImportError:
        logger.debug("ebsd_viewer not importable — no crop window", exc_info=True)
        return extractor

    window = get_active_crop_window()

    if window is None:
        return extractor

    # Two independent ways the open file can be the wrong one, and the shape
    # check alone catches only the first. This project ships LoGainNi and
    # HiGainNi — different measurements, both 28 086 pixels — so a cockpit
    # `POST /api/h5/open` of the sibling file would have passed the grid test
    # and handed back a cropped view of the WRONG MEASUREMENT. The message
    # names which check failed: "same size, different file" and "different
    # size" call for different things from the user.
    grid = tuple(extractor.get_grid_dimensions())
    want_grid = tuple(int(v) for v in window.original_shape)

    source = str(getattr(window, "source_file", "") or "")
    open_path = get_current_path()
    # Only a comparison of two KNOWN paths can fail. An unrecorded source or a
    # session with no path is "unknown", not "mismatched" — refusing there
    # would break every in-memory-only path that never had a file name.
    wrong_file = bool(
        source and open_path
        and _canonical_path(source) != _canonical_path(open_path)
    )

    if wrong_file or grid != want_grid:
        # Fail loud rather than cutting the wrong scan.
        if wrong_file and grid == want_grid:
            reason = (
                f"the grids match ({grid}) but the files do not: the window "
                f"was cut from {source!r} and the open file is {open_path!r}"
            )
        elif wrong_file:
            reason = (
                f"neither the file nor the grid matches: the window was cut "
                f"from {source!r} at {want_grid} and the open file is "
                f"{open_path!r} at {grid}"
            )
        else:
            reason = (
                f"the window was cut from a {want_grid} scan but the open "
                f"file is {grid}"
            )
        raise RuntimeError(f"crop window does not belong to the open file — {reason}")

    from backend.api.services.cropped_extractor import CroppedExtractor
    return CroppedExtractor(extractor, window)


def get_h5_file():
    """Return the open h5py.File handle, or raise RuntimeError if no file is open.

    Use this from routes that need raw HDF5 access (tree, attributes).
    Most routes should use get_extractor() instead.
    """
    with _lock:
        if _current_file is None:
            raise RuntimeError("No HDF5 file is open")
        return _current_file


def get_current_path() -> Optional[str]:
    """Get the path of the currently open file."""
    with _lock:
        return _current_path


def is_open() -> bool:
    """Check if a file is currently open."""
    with _lock:
        return _current_file is not None


def close_file():
    """Close the current file.

    The previous h5py.File is NOT closed synchronously — a concurrent reader
    may still hold it. It's parked in _pending_close (kept open) and the oldest
    beyond the cap is closed once it can no longer be in use. h5py also closes
    on GC, so nothing leaks indefinitely.
    """
    global _current_file, _current_extractor, _current_path

    with _lock:
        if _current_file is not None:
            _pending_close.append(_current_file)
            while len(_pending_close) > _PENDING_CLOSE_MAX:
                old = _pending_close.pop(0)
                try:
                    old.close()
                except Exception:
                    pass
            _current_file = None
            _current_extractor = None
            _current_path = None
        _pattern_cache.clear()
        _variant_extractors.clear()
        _phase_map_ctx_reset()

    # Drop any per-file derived state so it cannot leak into the next
    # file's session. Lazy-imported to keep the dependency one-way
    # (phase_map_store doesn't know about h5_session). Failure here is
    # never fatal — the worst case is a stale phase map, not a broken
    # file close.
    try:
        from backend.api.services.phase_map_store import get_phase_map_store
        get_phase_map_store().clear()
    except Exception:
        logger.debug("phase_map_store.clear() failed during close_file (ignored)", exc_info=True)


def get_cached_pattern(index: int, pattern_type: str = "processed"):
    """Get a pattern with LRU caching for fast repeated access."""
    cache_key = (index, pattern_type)
    if cache_key in _pattern_cache:
        _pattern_cache.move_to_end(cache_key)
        return _pattern_cache[cache_key]

    ext = get_extractor()
    pattern = ext.get_pattern_at_index(index, pattern_type)
    if pattern is not None:
        _pattern_cache[cache_key] = pattern
        while len(_pattern_cache) > _PATTERN_CACHE_SIZE:
            _pattern_cache.popitem(last=False)
    return pattern


def get_loaded_pattern(index: int, pattern_type: str = "processed"):
    """The pattern at flat ``index`` of the file as the EBSD viewer's dataset holds it.

    For a dataset loaded as a variant (measurement x pattern source) that is the variant's
    measurement and its pattern source; ``pattern_type`` is then not a choice, the loaded source
    decides (it is what the viewer shows and what an analysis has to look at). A dataset loaded
    without a choice reads exactly as :func:`get_cached_pattern` does with ``pattern_type``."""
    try:
        from backend.api.routes.ebsd_viewer import active_variant
    except ImportError:
        logger.debug("ebsd_viewer not importable — no loaded variant", exc_info=True)
        return get_cached_pattern(index, pattern_type)
    variant = active_variant(get_current_path())
    if variant is None:
        return get_cached_pattern(index, pattern_type)
    ext = get_variant_extractor()
    return ext.get_pattern_at_index(index, "raw" if variant.pattern_source == "unprocessed" else "processed")
