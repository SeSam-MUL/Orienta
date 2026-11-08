"""POST /api/pc/pattern/auto-pick: the calibration patterns "Find PC" starts from."""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROWS, COLS, SHAPE = 12, 15, (16, 16)
URL = "/api/pc/pattern/auto-pick"


class _Signal:
    """A scan of 12 x 15 patterns of 16 x 16 pixels; every pattern is a texture plus its own offset."""

    def __init__(self, n_rows=ROWS, n_cols=COLS, one_dimensional=False):
        rng = np.random.default_rng(3)
        self.texture = rng.normal(size=SHAPE).astype(np.float32)
        offsets = rng.normal(size=(n_rows, n_cols, 1, 1)).astype(np.float32) * 5
        data = self.texture + offsets
        if one_dimensional:
            data = data.reshape(n_rows * n_cols, *SHAPE)
            self.axes_manager = SimpleNamespace(navigation_shape=(n_rows * n_cols,))
        else:
            self.axes_manager = SimpleNamespace(navigation_shape=(n_cols, n_rows))
        self.data = data
        self.static_background = None


@pytest.fixture
def route(monkeypatch):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from backend.api.services import pc_patterns
    from backend.api.services.calibration_store import CalibrationStore

    monkeypatch.setattr(R, "_sessions", {})
    monkeypatch.setattr(R, "_parked_sessions", {})
    monkeypatch.setattr(R, "calibration_store", CalibrationStore())
    monkeypatch.setattr(R, "_optimization_active", False)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", None, raising=False)
    ctrl = R._get_controller()
    # state.quality(rows, cols) -> one number per sampled position; the default is the image
    # quality of a position = 100 * row + column, so the best pattern of a block is the one in the
    # block's last sampled row and column
    state = SimpleNamespace(dirty=False, signal=_Signal(), quality=None, seen_thread=None)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: state.signal)
    monkeypatch.setattr(ebsd_viewer, "is_active_signal_dirty", lambda: state.dirty)
    sampled = {}

    def quality(stack):
        state.seen_thread = threading.current_thread()
        if not sampled:                      # a scan of nine patterns or fewer is not sampled
            return np.zeros(len(stack))
        rows, cols = sampled["rows"], sampled["cols"]
        if state.quality is not None:
            return np.asarray(state.quality(rows, cols), dtype=float)
        return (rows * 100 + cols).astype(float)

    real = pc_patterns.sample_positions

    def spy(n_rows, n_cols, max_sample=pc_patterns.MAX_SAMPLE):
        rows, cols, k = real(n_rows, n_cols, max_sample)
        sampled.update(rows=rows, cols=cols)
        return rows, cols, k

    monkeypatch.setattr(pc_patterns, "sample_positions", spy)
    monkeypatch.setattr(pc_patterns, "_kikuchipy_quality", quality)
    app = FastAPI()
    app.include_router(R.router, prefix="/api/pc")
    return R, ctrl, state, TestClient(app), app


def _pos(out):
    return sorted((p["row"], p["col"]) for p in out["patterns"])


def test_nine_patterns_one_from_each_block_of_the_scan(route):
    R, ctrl, state, client, _ = route
    resp = client.post(URL, json={})
    assert resp.status_code == 200, resp.text
    out = resp.json()
    # 12 x 15 = 180 patterns <= 225: all looked at. Blocks cut the scan at 4 / 8 rows and 5 / 10
    # columns (by hand); with quality 100 * row + column the best of a block is its last row and column.
    assert _pos(out) == [
        (3, 4), (3, 9), (3, 14), (7, 4), (7, 9), (7, 14), (11, 4), (11, 9), (11, 14)]
    assert out["n_patterns"] == 9 == len(ctrl.patterns)
    assert out["n_sampled"] == 180 and out["stride"] == 1 and out["notes"] == []
    assert [(c[0], c[1]) for c, _ in ctrl.patterns] == [(p["row"], p["col"]) for p in out["patterns"]]
    assert [p["index"] for p in out["patterns"]] == list(range(9))
    assert out["patterns"][0]["quality"] == 304.0       # row 3, column 4


def test_the_stored_patterns_are_copies_the_page_can_keep(route):
    R, ctrl, state, client, _ = route
    state.dirty = True                      # taken as it is: nothing but the copy separates it from the viewer
    client.post(URL, json={})
    (row, col), stored = ctrl.patterns[0]
    kept = stored.copy()
    # the viewer processes its signal afterwards: what the page holds does not change
    state.signal.data[...] = 0.0
    np.testing.assert_array_equal(stored, kept)
    assert not np.shares_memory(stored, state.signal.data)
    # each pattern owns its memory, not a slice of the one array the pick returned
    assert all(p.base is None for _, p in ctrl.patterns)


def test_a_processed_signal_is_taken_as_it_is(route):
    """The user processed the patterns in the viewer: the picked patterns are the viewer's, bit for bit."""
    R, ctrl, state, client, _ = route
    state.dirty = True
    out = client.post(URL, json={}).json()
    assert out["prepared"] is False and out["background"] == "none"
    assert len(ctrl.patterns) == 9
    for (row, col), stored in ctrl.patterns:
        np.testing.assert_array_equal(stored, state.signal.data[row, col])


def test_an_untouched_signal_has_its_background_removed_and_says_so(route):
    R, ctrl, state, client, _ = route
    out = client.post(URL, json={}).json()
    assert out["prepared"] is True and out["background"] == "mean"
    (row, col), stored = ctrl.patterns[0]
    assert stored.dtype == np.float32 and not np.array_equal(stored, state.signal.data[row, col])


def test_the_background_of_the_file_is_used_when_the_signal_has_one(route):
    R, ctrl, state, client, _ = route
    state.signal.static_background = np.full(SHAPE, 2.0, np.float32)
    out = client.post(URL, json={}).json()
    assert out["prepared"] is True and out["background"] == "file"


def test_replace_false_adds_and_skips_a_position_already_in_the_list(route):
    R, ctrl, state, client, _ = route
    ctrl.add_pattern((3, 4), np.zeros(SHAPE, np.float32))              # one of the nine is already there
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.float32))
    out = client.post(URL, json={"replace": False}).json()
    assert len(ctrl.patterns) == 2 + 8 and out["n_patterns"] == 10
    assert (3, 4) not in [(p["row"], p["col"]) for p in out["patterns"]]
    assert np.array_equal(ctrl.patterns[0][1], np.zeros(SHAPE))        # the user's own is untouched


def test_replace_true_drops_the_old_list(route):
    R, ctrl, state, client, _ = route
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.float32))
    client.post(URL, json={})
    assert (0, 0) not in [c for c, _ in ctrl.patterns]


def test_the_cached_results_of_the_old_list_are_dropped(route):
    R, ctrl, state, client, _ = route
    sim_cache = R._get_session().sim_cache
    sim_cache[0] = {"ci": 0.5}
    ctrl.cache = {0: "old"}
    client.post(URL, json={})
    assert len(sim_cache) == 0 and ctrl.cache == {}


# ----- locking and refusal: the same as /pattern/add ------------------------------------------

@pytest.mark.parametrize("replace", [True, False])
def test_it_is_refused_while_a_search_runs(route, monkeypatch, replace):
    R, ctrl, state, client, _ = route
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.float32))
    monkeypatch.setattr(R, "_optimization_active", True)
    resp = client.post(URL, json={"replace": replace})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "pc_job_running"
    assert [c for c, _ in ctrl.patterns] == [(0, 0)]          # not even "replace" emptied the list
    assert state.seen_thread is None                          # refused before the scan was looked at


def test_a_search_that_starts_while_the_pick_waits_for_the_lock_refuses_it(route, monkeypatch):
    """The check is repeated inside the lock: a refine may start while the pick worked."""
    R, ctrl, state, client, _ = route
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.float32))

    @contextlib.contextmanager
    def lock_then_a_search_has_started(what):
        monkeypatch.setattr(R, "_optimization_active", True)
        yield

    monkeypatch.setattr(R, "_state_lock_or_busy", lock_then_a_search_has_started)
    resp = client.post(URL, json={})
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "pc_job_running"
    assert [c for c, _ in ctrl.patterns] == [(0, 0)]


def test_a_held_state_lock_answers_busy_and_changes_nothing(route, monkeypatch):
    """An Index All holds _state_lock for minutes: the pick answers 409 instead of hanging."""
    R, ctrl, state, client, _ = route
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.float32))
    monkeypatch.setattr(R, "_BUSY_WAIT_S", 0.05)
    held, release = threading.Event(), threading.Event()

    def holder():
        with R._state_lock:
            held.set()
            release.wait(10)

    t = threading.Thread(target=holder)
    t.start()
    try:
        assert held.wait(5)
        resp = client.post(URL, json={})
    finally:
        release.set()
        t.join()
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "pc_job_running"
    assert [c for c, _ in ctrl.patterns] == [(0, 0)]


def test_the_list_is_changed_under_the_lock_and_the_pick_leaves_the_event_loop_free(route, monkeypatch):
    R, ctrl, state, client, app = route
    seen = {}
    real_add = ctrl.add_pattern

    def spy_add(coords, pattern):
        seen["lock_held"] = R._state_lock.locked()
        seen["add_thread"] = threading.current_thread()
        return real_add(coords, pattern)

    monkeypatch.setattr(ctrl, "add_pattern", spy_add)
    from backend.api.services import pc_patterns
    fast = pc_patterns._kikuchipy_quality

    def slow(stack):
        time.sleep(1.2)
        return fast(stack)

    monkeypatch.setattr(pc_patterns, "_kikuchipy_quality", slow)

    async def scenario():
        ticks = 0
        stop = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.05)
                ticks += 1

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            tick_task = asyncio.create_task(ticker())
            resp = await ac.post(URL, json={})
            stop.set()
            await tick_task
        return resp, ticks

    resp, ticks = asyncio.run(scenario())
    assert resp.status_code == 200, resp.text
    # a free loop ticks ~24 times over the 1.2 s of the measurement; a blocked one (almost) not at all
    assert ticks >= 8, f"the event loop was blocked: {ticks} ticks"
    assert state.seen_thread is not threading.main_thread()
    assert seen["lock_held"] is True and seen["add_thread"] is not threading.main_thread()


def test_no_scan_loaded_is_a_400(route):
    R, ctrl, state, client, _ = route
    state.signal = None
    resp = client.post(URL, json={})
    assert resp.status_code == 400


# ----- reading the scan -----------------------------------------------------------------------

def test_a_scan_with_one_navigation_axis_is_read_along_it(route):
    """The five calibration patterns of a NORDIF Setting.txt arrive as one row; kikuchipy hands other
    one-dimensional scans as (n, rows, cols): every pattern of a scan of nine or fewer is used."""
    R, ctrl, state, client, _ = route
    state.signal = _Signal(1, 5, one_dimensional=True)
    out = client.post(URL, json={}).json()
    assert out["n_patterns"] == 5 and _pos(out) == [(0, i) for i in range(5)]


def test_an_int16_scan_is_stored_as_the_map_run_sees_it(route):
    """Rescaled by the range of the whole scan, like /pattern/add (signal_pattern)."""
    R, ctrl, state, client, _ = route
    rng = np.random.default_rng(5)
    raw = rng.integers(-100, 156, size=(ROWS, COLS) + SHAPE).astype(np.int16)
    raw[0, 0, 0, 0], raw[0, 0, 0, 1] = -100, 155      # the scan's range is exactly 255 wide
    sig = _Signal()
    sig.data = raw
    state.signal = sig
    state.dirty = True                                # taken as it is: only the rescale remains
    client.post(URL, json={})
    from backend.api.services.load_choices import signal_pattern
    assert len(ctrl.patterns) == 9
    for (row, col), stored in ctrl.patterns:
        # by hand: (v - (-100)) * 255 / 255 = v + 100
        np.testing.assert_array_equal(stored, (raw[row, col].astype(np.float32) + 100.0))
        np.testing.assert_array_equal(stored, signal_pattern(sig, row, col).astype(np.float32))
        assert stored.min() >= 0.0


class _Counting:
    """An array on "disk" that counts the chunks asked of it. Module level, with the list on the
    class: dask pickles a class defined inside a function by value, which would count on a copy."""
    reads: list = []

    def __init__(self, data):
        self.data = data
        self.shape, self.dtype, self.ndim = data.shape, data.dtype, data.ndim

    def __getitem__(self, key):
        out = self.data[key]
        if out.size:
            _Counting.reads.append(1)
        return out


def _counting_dask(data, reads):
    import dask.array as da

    _Counting.reads = reads
    return da.from_array(_Counting(data), chunks=(1,) * (data.ndim - 2) + SHAPE)


def test_a_lazy_scan_is_read_at_the_sampled_positions_only(route):
    R, ctrl, state, client, _ = route
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(40, 40) + SHAPE).astype(np.float32)
    reads = []
    sig = _Signal()
    sig.data = _counting_dask(raw, reads)
    sig.axes_manager = SimpleNamespace(navigation_shape=(40, 40))
    state.signal = sig
    state.dirty = True
    ctrl.aperture_circular = True           # known: no pattern is read for it, only the sample
    out = client.post(URL, json={}).json()
    # by hand: 40 x 40 = 1600 > 225 -> stride ceil(sqrt(1600 / 225)) = 3 -> 14 x 14 = 196 positions
    assert out["stride"] == 3 and out["n_sampled"] == 196
    assert len(reads) == 196
    for (row, col), stored in ctrl.patterns:
        np.testing.assert_array_equal(stored, raw[row, col])


def test_the_aperture_costs_one_read_per_picked_pattern_at_most(route):
    """Not known yet and no black corner anywhere: the nine picked patterns are read once more each, no more."""
    R, ctrl, state, client, _ = route
    raw = np.random.default_rng(1).normal(size=(40, 40) + SHAPE).astype(np.float32)
    reads = []
    sig = _Signal()
    sig.data = _counting_dask(raw, reads)
    sig.axes_manager = SimpleNamespace(navigation_shape=(40, 40))
    state.signal = sig
    state.dirty = True
    client.post(URL, json={})
    assert len(reads) == 196 + 9            # the 196 sampled positions of the test above, plus nine


def test_a_lazy_scan_with_one_navigation_axis(route):
    R, ctrl, state, client, _ = route
    raw = np.random.default_rng(2).normal(size=(5,) + SHAPE).astype(np.float32)
    reads = []
    sig = _Signal()
    sig.data = _counting_dask(raw, reads)
    sig.axes_manager = SimpleNamespace(navigation_shape=(5,))
    state.signal = sig
    state.dirty = True
    ctrl.aperture_circular = True           # known: no pattern is read for it, only the scan
    out = client.post(URL, json={}).json()
    assert _pos(out) == [(0, i) for i in range(5)] and len(reads) == 5
    for (row, col), stored in ctrl.patterns:
        np.testing.assert_array_equal(stored, raw[col])


# ----- what the pick says about itself ---------------------------------------------------------

def test_a_note_of_a_scan_too_small_for_a_mean_background_reaches_the_answer(route):
    R, ctrl, state, client, _ = route
    state.signal = _Signal(1, 2, one_dimensional=True)
    out = client.post(URL, json={}).json()
    assert out["notes"] == ["meanBackgroundSkipped"] and out["background"] == "none"
    assert out["n_patterns"] == 2


def test_a_note_of_a_mean_from_few_patterns_reaches_the_answer(route):
    R, ctrl, state, client, _ = route
    state.signal = _Signal(1, 5, one_dimensional=True)
    out = client.post(URL, json={}).json()
    assert out["notes"] == ["meanFromFewPatterns"] and out["background"] == "mean"


def test_a_note_of_blocks_without_a_quality_reaches_the_answer(route):
    R, ctrl, state, client, _ = route
    # no quality at all in the right-hand column of blocks (columns 10 and up)
    state.quality = lambda rows, cols: np.where(cols >= 10, np.nan, rows * 100.0 + cols)
    out = client.post(URL, json={}).json()
    assert out["notes"] == ["blockWithoutQuality"]
    assert out["n_patterns"] == 6 and all(p["col"] < 10 for p in out["patterns"])


def test_a_quality_that_is_not_finite_is_null_in_the_answer(route):
    """NaN and infinity are not valid JSON: the pick has no number for that pattern."""
    R, ctrl, state, client, _ = route
    state.quality = lambda rows, cols: np.where((rows == 3) & (cols == 4), np.inf, rows * 100.0 + cols)
    resp = client.post(URL, json={})
    assert resp.status_code == 200
    by_pos = {(p["row"], p["col"]): p["quality"] for p in resp.json()["patterns"]}
    assert by_pos[(3, 4)] is None and by_pos[(3, 9)] == 309.0


# ----- the aperture of the detector ------------------------------------------------------------

DISC = (60, 60)       # the corner blocks of the aperture check are 8 x 8: the disc needs a frame this big


def _disc_scan(circular):
    """12 x 15 recorded patterns of 60 x 60, values 80..199 (no negative value); with ``circular`` the
    pixels outside the inscribed disc are 0, as an EDAX detector records them."""
    from tools.pattern_comparison import circular_mask
    rng = np.random.default_rng(11)
    data = rng.integers(80, 200, size=(ROWS, COLS) + DISC).astype(np.float32)
    if circular:
        data[..., ~circular_mask(DISC)] = 0.0
    sig = _Signal()
    sig.data = data
    return sig


def _corner_max(png_b64):
    import base64
    import io
    from PIL import Image
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))))
    b = 6
    return max(int(img[:b, :b].max()), int(img[:b, -b:].max()), int(img[-b:, :b].max()), int(img[-b:, -b:].max()))


def test_a_circular_detector_makes_the_picked_patterns_round(route):
    """The picked patterns have their background removed (signed values); the aperture is read from the
    patterns as recorded, so the page shows them through the disc."""
    R, ctrl, state, client, _ = route
    state.signal = _disc_scan(circular=True)
    out = client.post(URL, json={}).json()
    assert out["prepared"] is True and out["n_patterns"] == 9
    assert min(float(p.min()) for _, p in ctrl.patterns) < 0       # the trap: the stored ones are signed
    assert ctrl.aperture_circular is True
    body = client.get("/api/pc/pattern/0/image").json()
    assert body["circular"] is True and _corner_max(body["image"]) == 0


def test_a_full_frame_detector_stays_square_after_a_pick(route):
    R, ctrl, state, client, _ = route
    state.signal = _disc_scan(circular=False)
    client.post(URL, json={})
    assert ctrl.aperture_circular is False
    body = client.get("/api/pc/pattern/0/image").json()
    assert body["circular"] is False and _corner_max(body["image"]) > 0


def test_signed_data_of_a_full_frame_scan_does_not_make_it_round(route):
    """The default scan is signed noise with no black corner anywhere."""
    R, ctrl, state, client, _ = route
    client.post(URL, json={})
    assert not getattr(ctrl, "aperture_circular", False)


def test_one_pattern_with_black_corners_decides_and_a_later_pick_keeps_it(route):
    R, ctrl, state, client, _ = route
    scan = _disc_scan(circular=False)
    scan.data[3, 4] = _disc_scan(circular=True).data[3, 4]      # only the pattern at (3, 4) is round
    state.signal = scan
    out = client.post(URL, json={}).json()
    assert (3, 4) in _pos(out) and ctrl.aperture_circular is True
    state.signal = _disc_scan(circular=False)                   # a later pick from a square scan
    client.post(URL, json={"replace": False})
    assert ctrl.aperture_circular is True


def test_nothing_is_read_for_the_aperture_when_it_is_known_and_never_more_than_nine(route, monkeypatch):
    R, ctrl, state, client, _ = route
    asked = []
    real = R._unprocessed_pattern

    def spy(dataset, row, col, shape):
        asked.append((row, col))
        return real(dataset, row, col, shape)

    monkeypatch.setattr(R, "_unprocessed_pattern", spy)
    state.signal = _disc_scan(circular=False)
    client.post(URL, json={})
    assert len(asked) == 9                      # square: all nine looked at, none twice
    asked.clear()
    ctrl.aperture_circular = True
    client.post(URL, json={})
    assert asked == []


def test_the_aperture_is_read_from_the_viewers_unprocessed_parent(route, monkeypatch):
    """A processed dataset is taken as it is; its recorded parent still shows the black corners."""
    from backend.api.routes import ebsd_viewer as ev
    import kikuchipy as kp
    R, ctrl, state, client, _ = route
    recorded = _disc_scan(circular=True).data
    processed = np.random.default_rng(2).normal(size=recorded.shape).astype(np.float32)   # signed noise
    parent, child = kp.signals.EBSD(recorded), kp.signals.EBSD(processed)
    sig = _Signal()
    sig.data = processed
    state.signal, state.dirty = sig, True
    monkeypatch.setattr(ev, "_raw_signals", {"scan": parent, "ds": child})
    monkeypatch.setattr(ev, "_dirty_datasets", {"ds"})
    monkeypatch.setattr(R.calibration_store, "get_entry",
                        lambda name: SimpleNamespace(parent_name="scan" if name == "ds" else None))
    client.post(URL, json={})
    assert ctrl.aperture_circular is True


def test_when_no_pattern_can_be_picked_the_old_list_stays(route):
    R, ctrl, state, client, _ = route
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.float32))
    state.quality = lambda rows, cols: np.full(len(rows), np.nan)
    resp = client.post(URL, json={})
    assert resp.status_code == 400 and resp.json()["detail"]["code"] == "noPatternPicked"
    assert [c for c, _ in ctrl.patterns] == [(0, 0)]


# ----- other patterns: exclude ------------------------------------------------------------------

FIRST_NINE = [(3, 4), (3, 9), (3, 14), (7, 4), (7, 9), (7, 14), (11, 4), (11, 9), (11, 14)]


def test_excluding_the_first_nine_gives_the_next_best_of_each_block(route):
    """Quality 100 * row + column, blocks cut at rows 4 / 8 and columns 5 / 10 (by hand): with the nine
    corners left out the best of every block is the position one column to the left."""
    R, ctrl, state, client, _ = route
    resp = client.post(URL, json={"exclude": [list(p) for p in FIRST_NINE]})
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert _pos(out) == [(3, 3), (3, 8), (3, 13), (7, 3), (7, 8), (7, 13), (11, 3), (11, 8), (11, 13)]
    assert [(c[0], c[1]) for c, _ in ctrl.patterns] == [(p["row"], p["col"]) for p in out["patterns"]]
    assert out["patterns"][0]["quality"] == 303.0


def test_without_exclude_the_pick_is_the_same_as_before(route):
    R, ctrl, state, client, _ = route
    assert _pos(client.post(URL, json={}).json()) == sorted(FIRST_NINE)
    assert _pos(client.post(URL, json={"exclude": []}).json()) == sorted(FIRST_NINE)


def test_fewer_patterns_than_the_search_needs_are_refused_in_words_and_the_list_stays(route):
    """Every sampled position from row 4 on is left out: only the three blocks of rows 0-3 have a
    pattern, and Find PC needs five."""
    R, ctrl, state, client, _ = route
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.float32))
    exclude = [[r, c] for r in range(4, ROWS) for c in range(COLS)]
    resp = client.post(URL, json={"exclude": exclude})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["code"] == "tooFewPatterns" and detail["params"] == {"have": 3, "need": 5}
    assert "at least 5" in detail["message"] and "only 3" in detail["message"]
    assert [c for c, _ in ctrl.patterns] == [(0, 0)]


def test_when_everything_is_left_out_nothing_is_picked_and_the_list_stays(route):
    R, ctrl, state, client, _ = route
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.float32))
    exclude = [[r, c] for r in range(ROWS) for c in range(COLS)]
    resp = client.post(URL, json={"exclude": exclude})
    assert resp.status_code == 400 and resp.json()["detail"]["code"] == "noPatternLeft"
    assert [c for c, _ in ctrl.patterns] == [(0, 0)]


def test_a_position_that_is_not_a_pair_is_refused(route):
    R, ctrl, state, client, _ = route
    assert client.post(URL, json={"exclude": [[1, 2, 3]]}).status_code == 422
    assert client.post(URL, json={"exclude": [[1]]}).status_code == 422
    assert client.post(URL, json={"exclude": ["a"]}).status_code == 422
