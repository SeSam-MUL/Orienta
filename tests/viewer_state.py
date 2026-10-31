"""Shared by the tests that load a dataset through ``POST /api/ebsd/load`` with the file
loader stubbed (``test_loaded_files_registry.py``, ``test_state_reset_on_load.py``).

A load writes a lot of module state outside the router it is called through. A test
that drives one has to put all of it back, or the next file finds a MagicMock signal
as the active dataset and a calibration entry with an empty detector shape.
``stubbed_viewer`` does that; ``snapshot_load_state`` / ``restore_load_state`` are the
two halves, public so the restore can be tested by hand.

What a load changes and these put back: the viewer's dataset registries and scalars,
the calibration store, the analysis dataset and its per-file stash, the crop windows
(``clear_all``), and the h5 session (a load of another file closes the open one).

What they leave out, deliberately:

* the indexing result registry and the active result id: ``tests/conftest.py`` restores
  them for every test (one mechanism);
* the static-background cache, the h5 pattern cache and the phase-map store the h5
  close clears, the EDS warning de-duplication, the ``state_version`` counter and
  ``_load_progress``: caches, log de-duplication and a monotonic counter. Nothing a
  later test could read as data, and a load of a new file rebuilds each of them.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import ebsd_viewer


def make_fake_signal(rows: int = 4, cols: int = 5, pat_h: int = 60, pat_w: int = 60):
    """Build a MagicMock EBSD signal with just the attributes load_ebsd reads."""
    sig = MagicMock()
    sig.data.shape = (rows, cols, pat_h, pat_w)
    sig.data.dtype = "uint8"
    # hyperspy: navigation_shape is (cols, rows), signal_shape is (w, h)
    sig.axes_manager.navigation_shape = (cols, rows)
    sig.axes_manager.signal_shape = (pat_w, pat_h)
    return sig


#: Module state of the viewer that a load writes: scalars, and containers changed in place.
VIEWER_SCALARS = ("_ebsd_signal", "_ebsd_file_path", "_active_dataset")
VIEWER_CONTAINERS = ("_loaded_files", "_raw_signals", "_positions", "_dirty_datasets",
                     "_signal_masks", "_overview_cache", "_registry_by_file")


def snapshot_load_state() -> dict:
    """Everything ``/load`` changes outside this router that a later test could see."""
    from backend.api.routes import analysis
    from backend.api.services import crop_window, h5_session
    from backend.api.services.calibration_store import calibration_store
    return {
        "scalars": {n: getattr(ebsd_viewer, n) for n in VIEWER_SCALARS},
        "containers": {n: getattr(ebsd_viewer, n).copy() for n in VIEWER_CONTAINERS},
        "calibration": calibration_store.snapshot(list(calibration_store.get_all())),
        "analysis": (analysis._dataset, dict(analysis._dataset_by_file)),
        "crops": (dict(crop_window._windows), dict(crop_window._stashed_masks)),
        "h5_session": (h5_session._current_file, h5_session._current_extractor,
                       h5_session._current_path, list(h5_session._pending_close)),
    }


def restore_load_state(saved: dict) -> None:
    from backend.api.routes import analysis
    from backend.api.services import crop_window, h5_session
    from backend.api.services.calibration_store import calibration_store
    for name, value in saved["scalars"].items():
        setattr(ebsd_viewer, name, value)
    for name, value in saved["containers"].items():
        live = getattr(ebsd_viewer, name)
        if isinstance(live, list):
            live[:] = value
        else:
            live.clear()
            live.update(value)
    calibration_store.clear()
    calibration_store.restore(saved["calibration"])
    analysis._dataset = saved["analysis"][0]
    analysis._dataset_by_file.clear()
    analysis._dataset_by_file.update(saved["analysis"][1])

    windows, masks = saved["crops"]
    crop_window._windows.clear()
    crop_window._windows.update(windows)
    crop_window._stashed_masks.clear()
    crop_window._stashed_masks.update(masks)

    # A load of another file closes the open h5 session. Put it back only if it can
    # still be read: ``close_file`` parks the handle open, so it normally can.
    file_, extractor, path, pending = saved["h5_session"]
    if file_ is not None and h5_session._current_file is not file_ and bool(file_):
        h5_session._current_file = file_
        h5_session._current_extractor = extractor
        h5_session._current_path = path
        h5_session._pending_close[:] = [f for f in pending if f is not file_]


@contextmanager
def stubbed_viewer(monkeypatch):
    """A TestClient over an app that mounts only the ebsd_viewer router.

    Starts from a blank viewer (no dataset, no loaded files, no crop windows), stubs the
    file loader so no real H5 file is needed, and puts back on exit everything the
    loads changed. The tests used to leave a MagicMock signal as the active dataset,
    with a calibration entry whose detector shape is empty, and a later test asking
    the active dataset for its detector got that entry.
    """
    import safe_loader
    from backend.api.routes import analysis
    from backend.api.services import crop_window
    from backend.api.services.calibration_store import calibration_store

    saved = snapshot_load_state()
    try:
        # Reset module state so tests are independent of order.
        for name in VIEWER_CONTAINERS:
            getattr(ebsd_viewer, name).clear()
        ebsd_viewer._ebsd_signal = None
        ebsd_viewer._ebsd_file_path = None
        ebsd_viewer._active_dataset = ""
        crop_window.clear_all()
        calibration_store.clear()
        analysis._dataset = None
        analysis._dataset_by_file.clear()

        # Stub the actual disk load - return a fresh fake signal for any path.
        monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path: make_fake_signal())

        app = FastAPI()
        app.include_router(ebsd_viewer.router, prefix="/api/ebsd")
        yield TestClient(app)
    finally:
        restore_load_state(saved)
