"""Tests for lazy EBSD loading + stage-based progress + non-blocking route.

Companion to test_ebsd_viewer_mask_clahe.py — uses the same MagicMock signal
pattern for fast in-memory tests. The large-file regression test against the
real 27 GB H5OINA file lives in tests/test_ebsd_viewer_large_file.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _restore_viewer_state():
    """Several tests below install a MagicMock signal as the viewer's active
    signal. Put the module state back afterwards so it cannot leak into the
    next test file (a leaked mock reports an empty detector shape)."""
    from backend.api.routes import ebsd_viewer as ev
    scalars = ("_ebsd_signal", "_ebsd_file_path", "_active_dataset")
    dicts = ("_loaded_files", "_raw_signals", "_positions",
             "_dirty_datasets", "_signal_masks")
    saved_scalars = {n: getattr(ev, n) for n in scalars}
    saved_dicts = {n: getattr(ev, n).copy() for n in dicts}
    from backend.api.services.calibration_store import calibration_store
    saved_cal = dict(calibration_store._entries)
    yield
    for n, v in saved_scalars.items():
        setattr(ev, n, v)
    for n, v in saved_dicts.items():
        live = getattr(ev, n)
        if isinstance(live, list):
            live[:] = v
        else:
            live.clear()
            live.update(v)
    # /load with a MagicMock signal registers an entry whose detector shape is ().
    calibration_store._entries.clear()
    calibration_store._entries.update(saved_cal)


# Small real file shipped with the repo. Used because we want to verify the
# actual kikuchipy lazy-load path, not a mock.
SMALL_TEST_FILE = Path(__file__).resolve().parents[1] / "Test_data" / "LoGainNi.h5"

# SampleB is the smallest Oxford H5OINA file shipped with the repo. Used to
# verify the Camera-Binning-Mode workaround in safe_loader: without the
# workaround, kp.load crashes with AttributeError on every Aztec 6.2+ file
# in this project (header omits "Camera Binning Mode" — kikuchipy 0.11.3
# calls None.split(...) on the missing value and the bare except clause
# misses AttributeError).
SMALL_OXFORD_FILE = (
    Path(__file__).resolve().parents[1]
    / "Test_data"
    / "EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 "
      "Elementverteilungsdaten 1.h5oina"
)


@pytest.mark.skipif(not SMALL_TEST_FILE.exists(), reason="LoGainNi.h5 not present")
def test_load_returns_lazy_signal(monkeypatch):
    """safe_loader.load_ebsd_safe must return a lazy hyperspy signal
    when the lazy threshold is set so the file qualifies.

    Updated 2026-05-27 for the size-based lazy/eager decision (session
    diagnostic: lazy-load was a regression for small files because every
    subsequent pattern/overview access had to read from disk, while
    eager loading a 339 MB file into RAM is trivial and makes everything
    instant). With KIKUCHIPY_LAZY_THRESHOLD_BYTES=0 every file goes
    lazy, which is the behaviour this test was written to verify.
    """
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "0")
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(str(SMALL_TEST_FILE))
    # hyperspy lazy signals expose ._lazy and have a dask-backed .data
    assert sig._lazy is True, "expected lazy signal — got eager"
    assert hasattr(sig.data, 'compute'), "expected dask array — got numpy"


@pytest.mark.skipif(not SMALL_TEST_FILE.exists(), reason="LoGainNi.h5 not present")
def test_load_small_file_returns_eager_signal_by_default():
    """The new size-based decision: files under 2 GB load eagerly so
    every subsequent pattern / overview / atlas access is instant from
    RAM. LoGainNi.h5 (101 MB) is comfortably below the threshold.

    Companion to ``test_load_returns_lazy_signal`` — that one verifies
    the lazy path is still wired up when explicitly opted into via the
    env var; this one verifies the default behaviour for the common
    small-file case.
    """
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(str(SMALL_TEST_FILE))
    assert sig._lazy is False, (
        "expected eager signal for a 101 MB file (under 2 GB default "
        "threshold) — got lazy"
    )
    # Eager signals expose numpy data directly; no .compute() needed.
    import numpy as _np
    assert isinstance(sig.data, _np.ndarray), (
        f"expected numpy.ndarray, got {type(sig.data).__name__}"
    )


@pytest.mark.skipif(
    not SMALL_OXFORD_FILE.exists(),
    reason="SampleB H5OINA not present in Test_data/",
)
def test_load_oxford_h5oina_uses_native_kikuchipy_path(monkeypatch):
    """Regression test for the Camera-Binning-Mode workaround in safe_loader.

    Previously, kp.load on any Oxford H5OINA crashed with AttributeError
    (header_group.get('Camera Binning Mode') is None, .split() fails on
    None, and the bare except (IndexError, ValueError) does not catch
    AttributeError) — and safe_loader silently fell back to unified_loader.
    Net effect: T1's lazy=True was a no-op for every Oxford file.

    After the A1 workaround
    (``_kikuchipy_oxford_camera_binning_workaround``), kp.load succeeds.
    To verify the native path (not the unified_loader fallback) we force
    lazy=True via the threshold env var: only the native kikuchipy path
    returns a lazy signal, the unified_loader fallback always returns
    eager. So sig._lazy == True ↔ native path took the load.
    """
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "0")
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(str(SMALL_OXFORD_FILE))
    assert sig._lazy is True, (
        "Oxford H5OINA must now return lazy via the workaround — if this "
        "fails, safe_loader probably fell through to unified_loader again"
    )
    assert hasattr(sig.data, 'compute'), (
        "expected dask array (lazy) — got numpy (eager fallback fired)"
    )


def test_oxford_workaround_restores_original_method():
    """The context manager must put the original kikuchipy method back even
    if the wrapped call succeeds — otherwise we leak monkey-patches into
    every kikuchipy code path in the same Python process.
    """
    from safe_loader import (
        _kikuchipy_needs_binning_patch,
        _kikuchipy_oxford_camera_binning_workaround,
    )
    # Ensure kikuchipy is loaded so the symbol exists
    import kikuchipy  # noqa: F401
    from kikuchipy.io.plugins.oxford_h5ebsd import _api as _ox

    original = _ox.OxfordH5EBSDReader.scan2dict
    with _kikuchipy_oxford_camera_binning_workaround():
        patched = _ox.OxfordH5EBSDReader.scan2dict
        if _kikuchipy_needs_binning_patch():
            assert patched is not original, "context manager did not install patch"
        else:
            assert patched is original, (
                "kikuchipy >= 0.12 reads the binning itself — overriding its "
                "reader with the 0.11.3 copy would discard that logic"
            )
    restored = _ox.OxfordH5EBSDReader.scan2dict
    assert restored is original, (
        "context manager did not restore original scan2dict on exit — "
        "monkey-patch is leaking into the rest of the process"
    )


@pytest.mark.parametrize("version, needed", [
    ("0.11.3", True), ("0.11.0", True), ("0.11.3+local", True),
    ("0.12.0", False), ("0.12.1", False), ("0.13.1", False), ("1.0.0", False),
    ("0.12.0rc1", False), ("0.13.0.dev0", False),
    # unreadable: never raise inside the load, leave kikuchipy's reader alone
    ("unknown", False), ("", False), ("1", False),
])
def test_binning_patch_only_below_kikuchipy_0_12(monkeypatch, version, needed):
    """kikuchipy 0.12.0 fixed the missing-binning crash upstream and removed
    ``kikuchipy.detectors.ebsd_detector``, which the 0.11.3 copy imported.
    Patching on >= 0.12 raised ImportError inside the load, and every H5OINA
    fell back to the eager loader without a word."""
    import safe_loader
    monkeypatch.setattr(safe_loader._kp(), "__version__", version)
    assert safe_loader._kikuchipy_needs_binning_patch() is needed


@pytest.mark.skipif(
    not SMALL_OXFORD_FILE.exists(),
    reason="SampleB H5OINA not present in Test_data/",
)
def test_h5oina_fallback_to_eager_is_a_warning(monkeypatch, caplog):
    """A native-loader failure on an H5OINA must be visible. It used to be an
    INFO line only, which is how kikuchipy 0.12 switched the lazy path off
    for every installation without anyone noticing."""
    import logging
    import safe_loader

    def boom(*a, **k):
        raise ImportError("simulated reader break")

    monkeypatch.setattr(safe_loader._kp(), "load", boom)
    with caplog.at_level(logging.WARNING, logger="safe_loader"):
        try:
            safe_loader.load_ebsd_safe(str(SMALL_OXFORD_FILE), verbose=False)
        except Exception:
            pass  # only the warning matters here, not whether a fallback loads
    assert any(
        r.levelno >= logging.WARNING and "eager loader" in r.getMessage()
        for r in caplog.records
    ), "native-loader failure on an H5OINA was not logged as a warning"


from unittest.mock import MagicMock

import numpy as np
import dask.array as da
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_lazy_fake_signal(rows: int = 3, cols: int = 4, pat_h: int = 16, pat_w: int = 20):
    """MagicMock signal whose .data is a dask array — exercises the lazy path."""
    sig = MagicMock()
    # Real dask-backed data so .compute() actually does something observable.
    numpy_data = np.full((rows, cols, pat_h, pat_w), 128, dtype=np.uint8)
    sig.data = da.from_array(numpy_data, chunks=(1, 1, pat_h, pat_w))
    sig._lazy = True
    sig.axes_manager.navigation_shape = (cols, rows)
    sig.axes_manager.signal_shape = (pat_w, pat_h)

    def _compute_in_place():
        # Mimic hyperspy's signal.compute() — flips _lazy + replaces .data with numpy
        sig.data = np.asarray(sig.data)
        sig._lazy = False
    sig.compute = _compute_in_place
    return sig


@pytest.fixture
def lazy_client(monkeypatch):
    """TestClient with a lazy MagicMock signal pre-loaded + mask enabled."""
    from backend.api.routes import ebsd_viewer
    ebsd_viewer._loaded_files.clear()
    ebsd_viewer._raw_signals.clear()
    ebsd_viewer._positions.clear()
    ebsd_viewer._dirty_datasets.clear()
    ebsd_viewer._signal_masks.clear()
    ebsd_viewer._ebsd_signal = None
    ebsd_viewer._ebsd_file_path = None
    ebsd_viewer._active_dataset = ""

    sig = _make_lazy_fake_signal()
    ebsd_viewer._raw_signals["lazy_fake"] = sig
    ebsd_viewer._positions["lazy_fake"] = (0, 0)
    ebsd_viewer._active_dataset = "lazy_fake"
    ebsd_viewer._ebsd_signal = sig
    # Enable the circular mask so the in-place write at line 941 fires
    ebsd_viewer._signal_masks["lazy_fake"] = {"enabled": True, "radius_fraction": 0.8}

    app = FastAPI()
    app.include_router(ebsd_viewer.router, prefix="/api/ebsd")
    return TestClient(app), sig


def test_background_removal_dynamic_with_mask_materialises_lazy_signal(lazy_client):
    """Dynamic BG with a mask writes signal.data[..., exclude] = 0 — dask arrays
    are read-only, so the route must materialise the lazy signal first."""
    client, sig = lazy_client
    assert sig._lazy is True, "fixture setup wrong: signal should start lazy"

    r = client.post("/api/ebsd/background-removal", json={"method": "dynamic"})

    assert r.status_code == 200, r.text
    assert sig._lazy is False, "signal must have been materialised before in-place write"
    # And the in-place zero outside the mask must have been applied
    assert isinstance(sig.data, np.ndarray)


import asyncio
import time
import httpx


def _build_app_with_health():
    """Build a FastAPI app that mounts the ebsd_viewer router and adds /health.

    Resets module-level signal state so each test starts clean. The slow loader
    behaviour is supplied per-test by monkeypatching ``safe_loader.load_ebsd_safe``.
    """
    from backend.api.routes import ebsd_viewer
    ebsd_viewer._loaded_files.clear()
    ebsd_viewer._raw_signals.clear()
    ebsd_viewer._positions.clear()
    ebsd_viewer._dirty_datasets.clear()
    ebsd_viewer._signal_masks.clear()
    ebsd_viewer._ebsd_signal = None
    ebsd_viewer._ebsd_file_path = None
    ebsd_viewer._active_dataset = ""

    app = FastAPI()
    app.include_router(ebsd_viewer.router, prefix="/api/ebsd")

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


def test_load_does_not_block_event_loop(monkeypatch, tmp_path):
    """While /load is in flight, /health must answer in <200 ms.

    With the old `async def load_ebsd` calling sync code directly, the event
    loop was blocked for the entire load duration (10 min on the 27 GB file).
    Wrapping the blocking call in asyncio.to_thread keeps the loop free.

    Detection strategy: we issue /load as a background task, then sleep on the
    event loop. With a blocking implementation, the sleep takes ~1.5 s
    wall-clock (the loop is frozen during the load and our sleep callback
    can't fire). With asyncio.to_thread the load runs in a threadpool,
    the loop stays responsive, and the sleep returns in its nominal time.
    We also issue /health WHILE the load is still in flight as a second
    check.
    """
    # Slow loader: sleep 1.5 s then return a tiny fake signal
    def slow_safe_loader(path, *args, **kwargs):
        time.sleep(1.5)
        return _make_lazy_fake_signal()
    monkeypatch.setattr("safe_loader.load_ebsd_safe", slow_safe_loader)
    # h5_session calls would otherwise try to open a real file — stub them out
    monkeypatch.setattr(
        "backend.api.services.h5_session.is_open", lambda: False
    )
    monkeypatch.setattr(
        "backend.api.services.h5_session.open_file", lambda path: None
    )
    monkeypatch.setattr(
        "backend.api.services.h5_session.get_extractor",
        lambda: type("E", (), {"detect_available_features": lambda self: {
            "has_eds": False, "has_electron_images": False,
            "eds_elements": [], "electron_images": []
        }})()
    )

    app = _build_app_with_health()
    fake_path = str(tmp_path / "fake.h5oina")
    Path(fake_path).touch()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            load_task = asyncio.create_task(
                ac.post("/api/ebsd/load", json={"path": fake_path})
            )
            # Measurement 1: how long does a 0.3 s sleep actually take while
            # /load is in flight? With a blocked loop, this will be ~1.5 s.
            # With asyncio.to_thread, this stays close to 0.3 s.
            t_sleep_start = time.perf_counter()
            await asyncio.sleep(0.3)
            sleep_elapsed = time.perf_counter() - t_sleep_start

            # Measurement 2: is the /load task still in flight after the
            # sleep, AND can /health answer quickly?  If the loader was
            # threadpool-dispatched the load is still running (~1.2 s of
            # the 1.5 s remain), and /health answers in ms.
            load_in_flight_after_sleep = not load_task.done()
            t_health = time.perf_counter()
            health_r = await ac.get("/health")
            health_elapsed = time.perf_counter() - t_health
            load_r = await load_task
            return (
                health_r, health_elapsed, load_r,
                sleep_elapsed, load_in_flight_after_sleep,
            )

    (
        health_r, health_elapsed, load_r,
        sleep_elapsed, load_in_flight_after_sleep,
    ) = asyncio.run(scenario())

    assert health_r.status_code == 200
    assert load_r.status_code == 200
    # The primary signal of a non-blocking implementation: the event loop's
    # own 0.3 s sleep returns close to 0.3 s, not 1.5 s.
    assert sleep_elapsed < 0.6, (
        f"event loop was blocked for {sleep_elapsed*1000:.1f} ms during a "
        f"0.3 s asyncio.sleep — the loader is running in the event-loop "
        f"coroutine instead of being dispatched to a thread"
    )
    # Corollary: with the loop responsive, /load must still be running
    # after our 0.3 s sleep (it takes 1.5 s total).
    assert load_in_flight_after_sleep, (
        "load_task completed during a 0.3 s sleep — that would only happen "
        "if the loader was synchronous AND blocking the event loop "
        "(in which case our 0.3 s sleep ran for 1.5 s)"
    )
    # And /health must answer fast while /load is still pending.
    assert health_elapsed < 0.2, (
        f"/health took {health_elapsed*1000:.1f} ms while /load was in "
        f"flight — event loop is still being blocked"
    )


import uuid


def test_progress_endpoint_404_for_unknown_id():
    """Unknown request_id → 404 from the explicit endpoint handler,
    not FastAPI auto-404 (which would also return 404 for a missing route)."""
    from backend.api.routes import ebsd_viewer
    app = FastAPI()
    app.include_router(ebsd_viewer.router, prefix="/api/ebsd")
    client = TestClient(app)

    rid = str(uuid.uuid4())
    r = client.get(f"/api/ebsd/load/progress/{rid}")
    assert r.status_code == 404
    # Explicit endpoint message → catches accidental endpoint removal regression
    assert "Unknown request_id" in r.json().get("detail", ""), (
        f"expected explicit endpoint 404 message, got: {r.json()}"
    )


def test_progress_error_preserves_started_at(monkeypatch, tmp_path):
    """When a load fails mid-way, elapsed_seconds must still reflect the
    real load duration, not be reset to ~0 by the error path."""
    from backend.api.routes import ebsd_viewer
    ebsd_viewer._load_progress.clear()
    ebsd_viewer._raw_signals.clear()

    # Loader sleeps then raises — stage 1 fires, then we fail.
    def failing_load(path, *args, **kwargs):
        time.sleep(0.4)
        raise RuntimeError("simulated load failure")
    monkeypatch.setattr("safe_loader.load_ebsd_safe", failing_load)

    app = _build_app_with_health()
    fake_path = str(tmp_path / "fake.h5oina")
    Path(fake_path).touch()
    rid = str(uuid.uuid4())

    client = TestClient(app)
    r = client.post("/api/ebsd/load", json={"path": fake_path, "request_id": rid})
    assert r.status_code == 400, r.text

    progress_r = client.get(f"/api/ebsd/load/progress/{rid}")
    assert progress_r.status_code == 200
    snap = progress_r.json()
    assert snap["stage"] == "error"
    assert snap["elapsed_seconds"] >= 0.3, (
        f"elapsed_seconds={snap['elapsed_seconds']:.3f} — error path is "
        f"resetting started_at, hiding how long the load ran before failing"
    )


def test_progress_endpoint_reports_all_stages(monkeypatch, tmp_path):
    """Poll /progress during a slow load — must observe all 4 named stages
    plus the final 'complete' state.

    Each stage gets a small artificial delay (200 ms safe_loader, 200 ms
    calibration_store.register, 200 ms h5_session feature detection, 200 ms
    _register_loaded_file) so a 50 ms-interval poll loop is guaranteed to
    observe each transition. In production every stage has real wallclock
    time (multi-second loads on multi-GB files); this test just makes the
    MagicMock environment match that reality at coarse granularity.
    """
    from backend.api.routes import ebsd_viewer
    ebsd_viewer._load_progress.clear()
    ebsd_viewer._raw_signals.clear()

    # Stage 1 → 2: safe_loader runs during reading_metadata, returns into
    # building_signal.
    def slow_load(path, *args, **kwargs):
        time.sleep(0.2)
        return _make_lazy_fake_signal()
    monkeypatch.setattr("safe_loader.load_ebsd_safe", slow_load)

    # Stage 2 → 3: calibration_store.register runs after building_signal is
    # set and before detecting_features. Slow it to make building_signal
    # observable.
    from backend.api.services.calibration_store import calibration_store
    def slow_register(dataset_name, signal):
        time.sleep(0.2)
        # The signal is a MagicMock so we can't pass it to the real register
        # without side effects — just no-op here, our test doesn't read it.
        return None
    monkeypatch.setattr(calibration_store, "register", slow_register)

    # Stage 3: h5_session.detect_available_features runs during
    # detecting_features. Slow it for observability.
    monkeypatch.setattr("backend.api.services.h5_session.is_open", lambda: False)
    monkeypatch.setattr("backend.api.services.h5_session.open_file", lambda path: None)

    class _SlowExtractor:
        def detect_available_features(self):
            time.sleep(0.2)
            return {
                "has_eds": False, "has_electron_images": False,
                "eds_elements": [], "electron_images": []
            }

    monkeypatch.setattr(
        "backend.api.services.h5_session.get_extractor",
        lambda: _SlowExtractor()
    )

    # Stage 4 → complete: _register_loaded_file runs after finalising is
    # set and before complete. Slow it for observability.
    _orig_register_file = ebsd_viewer._register_loaded_file
    def slow_register_file(path):
        time.sleep(0.2)
        return _orig_register_file(path)
    monkeypatch.setattr(ebsd_viewer, "_register_loaded_file", slow_register_file)

    app = _build_app_with_health()
    fake_path = str(tmp_path / "fake.h5oina")
    Path(fake_path).touch()
    rid = str(uuid.uuid4())

    seen_stages = []

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            load_task = asyncio.create_task(
                ac.post("/api/ebsd/load", json={"path": fake_path, "request_id": rid})
            )
            # Poll while load runs — load takes ~800 ms total (4 × 200 ms
            # observability delays), so 40 polls × 50 ms = 2 s covers it.
            for _ in range(40):
                await asyncio.sleep(0.05)
                r = await ac.get(f"/api/ebsd/load/progress/{rid}")
                if r.status_code == 200:
                    body = r.json()
                    if body["stage"] not in seen_stages:
                        seen_stages.append(body["stage"])
                    if body["stage"] in ("complete", "error"):
                        break
            await load_task

    asyncio.run(scenario())

    expected = {"reading_metadata", "building_signal", "detecting_features",
                "finalising", "complete"}
    missing = expected - set(seen_stages)
    assert not missing, f"never observed stages: {missing}; saw: {seen_stages}"


def test_progress_entry_evicted_after_ttl(monkeypatch):
    """Entries with stage in (complete, error) and age > TTL are dropped on
    the next progress write."""
    from backend.api.routes import ebsd_viewer
    ebsd_viewer._load_progress.clear()
    # Synthesise a stale complete entry (start time 120 s ago)
    rid = str(uuid.uuid4())
    ebsd_viewer._load_progress[rid] = {
        "stage": "complete",
        "stage_idx": 4,
        "stage_total": 4,
        "started_at": time.time() - 120.0,
        "message": "done",
    }
    # Trigger the sweep via a fresh progress write
    new_rid = str(uuid.uuid4())
    ebsd_viewer._update_progress(new_rid, stage="reading_metadata", stage_idx=1,
                                  stage_total=4, message="…", started_at=time.time())

    assert rid not in ebsd_viewer._load_progress, "stale complete entry should be evicted"
    assert new_rid in ebsd_viewer._load_progress


# ---------------------------------------------------------------------------
# pattern-atlas guards (2026-06-01)
#
# The atlas endpoint used to (a) run synchronously on the event loop and
# (b) materialise EVERY pattern of a lazy multi-GB signal into RAM, then
# (c) assume a 4-D shape — so on big/odd datasets it OOM'd or raised and
# returned 500, which broke the EBSD-viewer image AND, because it blocked
# the loop, made switching into the Indexing page hang. The atlas is only a
# drag-navigation optimisation; the frontend falls back to per-pattern fetch
# when ``atlas`` is None, so these guards return 200 + null instead of 500.
# ---------------------------------------------------------------------------

def _make_eager_fake_signal(rows=3, cols=4, pat_h=16, pat_w=20, ndim4=True):
    """Eager (numpy) MagicMock signal. ndim4=False yields a 3-D nav layout."""
    sig = MagicMock()
    if ndim4:
        sig.data = np.full((rows, cols, pat_h, pat_w), 128, dtype=np.uint8)
    else:
        sig.data = np.full((rows * cols, pat_h, pat_w), 128, dtype=np.uint8)
    sig._lazy = False
    sig.axes_manager.navigation_shape = (cols, rows)
    sig.axes_manager.signal_shape = (pat_w, pat_h)
    return sig


def _atlas_client(sig):
    from backend.api.routes import ebsd_viewer
    ebsd_viewer._raw_signals.clear()
    ebsd_viewer._raw_signals["fake"] = sig
    ebsd_viewer._active_dataset = "fake"
    ebsd_viewer._ebsd_signal = sig
    app = FastAPI()
    app.include_router(ebsd_viewer.router, prefix="/api/ebsd")
    return TestClient(app)


def test_pattern_atlas_skips_oversized_lazy_dataset(monkeypatch):
    """A lazy signal whose atlas footprint exceeds the cap must return
    ``atlas: None`` (HTTP 200) WITHOUT materialising the lazy signal."""
    from backend.api.routes import ebsd_viewer
    monkeypatch.setattr(ebsd_viewer, "_ATLAS_MAX_BYTES", 1)  # force the cap
    sig = _make_lazy_fake_signal()
    client = _atlas_client(sig)

    r = client.get("/api/ebsd/pattern-atlas?step=4")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["atlas"] is None
    assert "too large" in body["reason"]
    assert sig._lazy is True, "oversized lazy signal must NOT be materialised"


def test_pattern_atlas_3d_signal_returns_none_not_500():
    """A 3-D nav signal (unified_loader fallback) must not 500 on
    ``data.shape[3]`` — it returns ``atlas: None`` so the viewer falls back."""
    sig = _make_eager_fake_signal(ndim4=False)
    client = _atlas_client(sig)

    r = client.get("/api/ebsd/pattern-atlas?step=4")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["atlas"] is None
    assert "unsupported nav shape" in body["reason"]


def test_pattern_atlas_small_dataset_builds():
    """A small (under-cap) dataset still builds a real atlas image."""
    sig = _make_eager_fake_signal()
    client = _atlas_client(sig)

    r = client.get("/api/ebsd/pattern-atlas?step=4")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["atlas"] is not None
    assert body["grid_rows"] == 3 and body["grid_cols"] == 4
