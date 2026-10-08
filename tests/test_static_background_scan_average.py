"""Scan average for static background removal: block reads, cache, event loop, progress.

A 27 GB scan (485k patterns, one h5 chunk per pattern) took 530 s and ~4.5 GB of
RAM when the mean was built from one dask task per pattern, and the viewer's POST
sat on it for that long. The average is now read in large sequential slabs,
cached per loaded dataset, reported to the progress poll, and computed off the
event loop.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import dask.array as da
import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402

from backend.api.services import static_background as sb  # noqa: E402

N, H, W = 61, 12, 16          # 61 patterns: deliberately not a multiple of any block size


@pytest.fixture(autouse=True)
def _restore_viewer_state():
    """These tests put a real signal into the viewer's module state. Restore it
    afterwards so it cannot leak into the next test file: a leaked signal makes
    a later export write its patterns and a later import keep that file open."""
    from backend.api.routes import ebsd_viewer as ev
    scalars = ("_ebsd_signal", "_ebsd_file_path", "_active_dataset")
    dicts = ("_loaded_files", "_raw_signals", "_positions", "_dirty_datasets",
             "_signal_masks", "_processing_progress", "_registry_by_file",
             "_overview_cache")
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
    # /load registers a calibration entry for the fake signal it was handed.
    calibration_store._entries.clear()
    calibration_store._entries.update(saved_cal)


class CountingDataset:
    """h5py-like source that records every read (and can be slow)."""

    def __init__(self, arr, chunks0=1, delay=0.0):
        self._a = arr
        self.shape, self.dtype, self.ndim = arr.shape, arr.dtype, arr.ndim
        self.chunks = (chunks0,) + arr.shape[1:]
        self.calls = []              # slab reads: a plain slice along the pattern axis
        self.other_calls = []        # dask's zero-size meta probe, single-pattern reads
        self.delay = delay
        self.lock = threading.Lock()

    def __getitem__(self, key):
        with self.lock:
            (self.calls if isinstance(key, slice) else self.other_calls).append(key)
        if self.delay:
            time.sleep(self.delay)
        return self._a[key]


def _patterns(seed=0, n=N, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256 if dtype == np.uint8 else 4000, size=(n, H, W)).astype(dtype)


def _lazy_signal(ds, rows, cols):
    """Lazy EBSD signal over ``ds`` reshaped to a (rows, cols) navigation grid,
    built like kikuchipy builds it from an h5 dataset (from_array -> reshape)."""
    import kikuchipy as kp

    arr = da.from_array(ds, chunks=(1,) + ds.shape[1:]).reshape(rows, cols, H, W)
    return kp.signals.LazyEBSD(arr)


def _old_average(a):
    mean = np.asarray(a.reshape(-1, H, W).mean(axis=0, dtype=np.float64))
    return np.rint(mean).astype(a.dtype) if np.issubdtype(a.dtype, np.integer) else mean.astype(a.dtype)


@pytest.fixture(autouse=True)
def _clean_cache():
    sb.clear_cache()
    yield
    sb.clear_cache()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
@pytest.mark.parametrize("block_bytes", [1, H * W * 4, H * W * 7 + 13, 10 ** 9])
def test_block_average_is_bit_identical_to_the_per_pattern_mean(monkeypatch, dtype, block_bytes):
    monkeypatch.setattr(sb, "BLOCK_TARGET_BYTES", block_bytes)
    a = _patterns(dtype=dtype)
    ds = CountingDataset(a, chunks0=4)
    out = sb.scan_average(_lazy_signal(ds, 1, N))
    assert out.dtype == a.dtype
    assert np.array_equal(out, _old_average(a))


def test_pristine_h5_graph_is_read_in_slabs_not_per_pattern(monkeypatch):
    monkeypatch.setattr(sb, "BLOCK_TARGET_BYTES", H * W * 20)  # ~20 patterns per slab
    ds = CountingDataset(_patterns(), chunks0=4)
    sig = _lazy_signal(ds, 1, N)
    assert sb._pristine_source(sig.data) is ds, "fixture should look like kikuchipy's h5 graph"
    sb.scan_average(sig)
    # 61 patterns / 20 per slab -> 4 reads, each spanning many patterns. The
    # one-task-per-pattern graph would issue 61 single-pattern reads.
    assert 1 <= len(ds.calls) <= 5, len(ds.calls)
    assert all(isinstance(k, slice) for k in ds.calls)
    assert all((k.stop - k.start) > 1 for k in ds.calls[:-1])  # only the tail slab may be short
    # slab starts are aligned to the dataset's own chunking
    assert all(k.start % 4 == 0 for k in ds.calls)


def test_4d_memmap_source_is_read_in_row_blocks_within_the_byte_bound(monkeypatch, tmp_path):
    """A 4D (ny, nx, h, w) source (kikuchipy's lazy up1/up2 reader wraps a memmap)
    slices ROWS of patterns along axis 0: the block size has to count a row, not a
    pattern, or one block swallows the whole file and progress never advances."""
    import kikuchipy as kp

    ny, nx = 40, 6
    a = _patterns(n=ny * nx).reshape(ny, nx, H, W)
    path = tmp_path / "patterns.dat"
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=a.shape)
    mm[...] = a
    mm.flush()
    ds = CountingDataset(np.memmap(path, dtype=np.uint8, mode="r", shape=a.shape), chunks0=1)

    row_bytes = nx * H * W
    bound = row_bytes * 5                                    # five rows per block
    monkeypatch.setattr(sb, "BLOCK_TARGET_BYTES", bound)
    arr = da.from_array(ds, chunks=(1, nx, H, W))
    sig = kp.signals.LazyEBSD(arr)
    assert sb._pristine_source(sig.data) is ds, "4D memmap source must take the slab path"

    seen = []
    out = sb.scan_average(sig, progress=lambda d, t: seen.append((d, t)))

    assert np.array_equal(out, _old_average(a.reshape(-1, H, W)))
    assert len(ds.calls) == ny // 5, ds.calls
    assert all((k.stop - k.start) * row_bytes <= bound for k in ds.calls)
    assert seen[-1] == (ny // 5, ny // 5) and len(seen) > 2


def test_processed_lazy_data_is_not_mistaken_for_the_file(monkeypatch):
    """After a processing step the mean must be of the PROCESSED data."""
    a = _patterns()
    ds = CountingDataset(a)
    sig = _lazy_signal(ds, 1, N)
    sig.remove_dynamic_background(show_progressbar=False)
    assert sb._pristine_source(sig.data) is None
    out = sb.scan_average(sig)
    processed = np.asarray(sig.data)
    assert np.array_equal(out, _old_average(processed))
    assert not np.array_equal(out, _old_average(a))


def test_cropped_lazy_data_is_averaged_over_the_crop_only():
    a = _patterns()
    sig = _lazy_signal(CountingDataset(a), 1, N)
    crop = sig.inav[10:30, :]
    assert sb._pristine_source(crop.data) is None
    assert np.array_equal(sb.scan_average(crop), _old_average(a[10:30]))


def test_eager_numpy_signal_unchanged_and_not_cached():
    import kikuchipy as kp

    a = _patterns()
    sig = kp.signals.EBSD(a.reshape(1, N, H, W).copy())
    assert np.array_equal(sb.scan_average(sig), _old_average(a))
    assert sb._cache == {}


def test_progress_is_reported_per_block_and_ends_complete(monkeypatch):
    monkeypatch.setattr(sb, "BLOCK_TARGET_BYTES", H * W * 10)
    seen = []
    sb.scan_average(_lazy_signal(CountingDataset(_patterns()), 1, N), progress=lambda d, t: seen.append((d, t)))
    assert seen[0] == (0, seen[0][1]) and seen[-1][0] == seen[-1][1] > 1
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_second_average_of_the_same_data_is_served_from_cache():
    ds = CountingDataset(_patterns())
    sig = _lazy_signal(ds, 1, N)
    first = sb.scan_average(sig, source_path=__file__)
    n_reads = len(ds.calls)
    second = sb.scan_average(sig, source_path=__file__)
    assert len(ds.calls) == n_reads, "second call must not read again"
    assert np.array_equal(first, second)
    second[...] = 0                      # callers may not poison the cache
    assert np.array_equal(sb.scan_average(sig, source_path=__file__), first)


def test_cache_is_not_shared_with_another_load_other_file_or_processed_data():
    a = _patterns()
    ds1 = CountingDataset(a)
    sig1 = _lazy_signal(ds1, 1, N)
    sb.scan_average(sig1, source_path=__file__)

    ds2 = CountingDataset(a)                     # same bytes, freshly loaded
    sb.scan_average(_lazy_signal(ds2, 1, N), source_path=__file__)
    assert ds2.calls, "a new load must be read, not served from the previous load's entry"

    n = len(ds1.calls) + len(ds1.other_calls)
    sig1.remove_dynamic_background(show_progressbar=False)   # processing changes the data
    out = sb.scan_average(sig1, source_path=__file__)
    assert len(ds1.calls) + len(ds1.other_calls) > n, "processed data must be read again"
    assert np.array_equal(out, _old_average(np.asarray(sig1.data)))


def test_clear_cache_forces_a_new_read():
    ds = CountingDataset(_patterns())
    sig = _lazy_signal(ds, 1, N)
    sb.scan_average(sig)
    n = len(ds.calls)
    sb.clear_cache()
    sb.scan_average(sig)
    assert len(ds.calls) > n


def test_cache_is_bounded():
    for i in range(sb._CACHE_MAX_ENTRIES + 3):
        sb.scan_average(_lazy_signal(CountingDataset(_patterns(seed=i)), 1, N))
    assert len(sb._cache) <= sb._CACHE_MAX_ENTRIES


# ---------------------------------------------------------------------------
# Route: event loop stays free, progress is reported, cache is dropped on load
# ---------------------------------------------------------------------------
def _app_with(signal, name="scanavg"):
    from backend.api.routes import ebsd_viewer

    ebsd_viewer._loaded_files.clear()
    ebsd_viewer._raw_signals.clear()
    ebsd_viewer._positions.clear()
    ebsd_viewer._dirty_datasets.clear()
    ebsd_viewer._signal_masks.clear()
    ebsd_viewer._processing_progress.clear()
    ebsd_viewer._raw_signals[name] = signal
    ebsd_viewer._positions[name] = (0, 0)
    ebsd_viewer._active_dataset = name
    ebsd_viewer._ebsd_signal = signal
    ebsd_viewer._ebsd_file_path = None
    app = FastAPI()
    app.include_router(ebsd_viewer.router, prefix="/api/ebsd")

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app, ebsd_viewer


def test_scan_average_does_not_block_the_event_loop():
    """/health must keep answering within milliseconds while the average is read."""
    ds = CountingDataset(_patterns(), chunks0=1, delay=0.15)   # each slab read takes 150 ms
    sig = _lazy_signal(ds, 1, N)
    app, _ = _app_with(sig)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            task = asyncio.create_task(ac.post("/api/ebsd/background-removal", json={"method": "static"}))
            await asyncio.sleep(0.05)
            lat = []
            while not task.done():
                t0 = time.perf_counter()
                r = await ac.get("/health")
                lat.append(time.perf_counter() - t0)
                assert r.status_code == 200
                await asyncio.sleep(0.05)
            return await task, lat

    old = sb.BLOCK_TARGET_BYTES
    sb.BLOCK_TARGET_BYTES = H * W * 8          # several slabs -> the POST lasts about a second
    try:
        resp, lat = asyncio.run(scenario())
    finally:
        sb.BLOCK_TARGET_BYTES = old
    assert resp.status_code == 200, resp.text
    assert len(lat) >= 5, "the request should have been in flight long enough to poll"
    assert max(lat) < 0.1, f"event loop blocked: worst /health latency {max(lat) * 1000:.0f} ms"


def test_route_reports_scan_average_progress_and_completion(monkeypatch):
    monkeypatch.setattr(sb, "BLOCK_TARGET_BYTES", H * W * 8)
    ds = CountingDataset(_patterns())
    app, ev = _app_with(_lazy_signal(ds, 1, N))
    snaps = []
    orig = ev._set_processing_progress

    def spy(rid, **state):
        snaps.append(dict(state))
        return orig(rid, **state)

    monkeypatch.setattr(ev, "_set_processing_progress", spy)
    from fastapi.testclient import TestClient

    c = TestClient(app)
    r = c.post("/api/ebsd/background-removal", json={"method": "static", "request_id": "rid-1"})
    assert r.status_code == 200, r.text
    running = [s for s in snaps if s.get("phase") == "scan_average"]
    assert running and running[0]["n_patterns"] == N and running[-1]["fraction"] == 1.0
    assert snaps[-1]["stage"] == "complete"
    p = c.get("/api/ebsd/processing-progress/rid-1").json()
    assert p["found"] and p["stage"] == "complete"


def test_route_second_static_on_a_copy_of_unchanged_data_uses_the_cache():
    """The Recommended Pipeline / a second click on identical data must not re-read."""
    import copy
    from fastapi.testclient import TestClient

    ds = CountingDataset(_patterns())
    sig = _lazy_signal(ds, 1, N)
    app, ev = _app_with(sig)
    ev._ebsd_file_path = __file__
    twin = copy.deepcopy(sig)                    # what /deepcopy gives: same graph
    c = TestClient(app)
    assert c.post("/api/ebsd/background-removal", json={"method": "static"}).status_code == 200
    n = len(ds.calls)
    ev._raw_signals["scanavg"] = twin
    ev._ebsd_signal = twin
    assert c.post("/api/ebsd/background-removal", json={"method": "static"}).status_code == 200
    assert len(ds.calls) == n, "identical data must not be averaged twice"


def test_loading_a_file_drops_the_cache(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    app, ev = _app_with(_lazy_signal(CountingDataset(_patterns()), 1, N))
    sb.scan_average(_lazy_signal(CountingDataset(_patterns()), 1, N))
    assert sb._cache

    fake = _lazy_signal(CountingDataset(_patterns(seed=3)), 1, N)
    monkeypatch.setattr("safe_loader.load_ebsd_safe", lambda path, *a, **k: fake)
    monkeypatch.setattr("backend.api.services.h5_session.is_open", lambda: False)
    monkeypatch.setattr("backend.api.services.h5_session.open_file", lambda path: None)
    monkeypatch.setattr(
        "backend.api.services.h5_session.get_extractor",
        lambda: type("E", (), {"detect_available_features": lambda self: {
            "has_eds": False, "has_electron_images": False,
            "eds_elements": [], "electron_images": []}})(),
    )
    f = tmp_path / "x.h5oina"
    f.touch()
    r = TestClient(app).post("/api/ebsd/load", json={"path": str(f)})
    assert r.status_code == 200, r.text
    assert sb._cache == {}


def test_crop_and_switch_paths_drop_the_cache():
    """Source-level guard: every place that resets the overview cache for a new
    file / a crop also drops the scan-average cache."""
    src = (Path(__file__).resolve().parents[1] / "backend" / "api" / "routes" / "ebsd_viewer.py").read_text(
        encoding="utf-8")
    lines = src.splitlines()
    resets = [i for i, ln in enumerate(lines) if ln.strip() == "_overview_cache.clear()"]
    assert len(resets) == 2, "load and switch-file"
    for i in resets:
        assert lines[i + 1].strip() == "static_background.clear_cache()"
    crop_body = src[src.index("async def crop_dataset"):src.index("@router.get(\"/crop\")")]
    assert "static_background.clear_cache()" in crop_body
