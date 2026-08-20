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
    # file in a previous session. Best-effort — a corrupted sidecar
    # just leaves the store empty (the user can re-run auto-classify),
    # it must NEVER fail the file open.
    try:
        from backend.api.services.phase_map_store import get_phase_map_store
        if get_phase_map_store().load_from_disk(path):
            logger.info("Phase-map sidecar restored for %s", path)
    except Exception:
        logger.debug("phase_map_store.load_from_disk() failed during open_file", exc_info=True)

    return extractor, fmt


def get_extractor():
    """Get the current data extractor. Raises if no file is open."""
    with _lock:
        if _current_extractor is None:
            raise RuntimeError("No HDF5 file is currently open")
        return _current_extractor


def get_active_extractor():
    """The extractor as the ACTIVE DATASET sees it.

    If the active dataset is a crop, this is a ``CroppedExtractor`` view whose
    every grid-shaped read is cut to the crop window. Otherwise it is the very
    same object ``get_extractor()`` returns — the non-crop path is unchanged,
    not merely equivalent.

    Use this from anything that works on "the dataset the user is looking at".
    Use ``get_extractor()`` from anything that must show the FILE as it is —
    the h5 viewer, format probing, preflight checks.
    """
    extractor = get_extractor()

    # Lazy import: ebsd_viewer imports services, so a module-level import here
    # would close a cycle. Same pattern as close_file's phase_map_store import.
    #
    # ONLY the import is allowed to fail soft. Swallowing an error out of
    # get_active_crop_window() would return the FULL-SCAN extractor for a
    # dataset that IS a crop, and every switched route would then serve
    # full-scan data with nothing in the log — exactly the silently-wrong
    # mode this design exists to prevent. Let anything else propagate.
    try:
        from backend.api.routes.ebsd_viewer import get_active_crop_window
    except ImportError:
        logger.debug("ebsd_viewer not importable — no crop window", exc_info=True)
        return extractor

    window = get_active_crop_window()

    if window is None:
        return extractor

    if tuple(extractor.get_grid_dimensions()) != tuple(window.original_shape):
        # The open file is not the one this window was cut from. Fail loud
        # rather than cutting the wrong scan.
        raise RuntimeError(
            f"crop window was cut from {tuple(window.original_shape)} but the "
            f"open file is {tuple(extractor.get_grid_dimensions())}"
        )

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
