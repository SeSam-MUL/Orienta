"""POST /api/ebsd/nlpar: NLPAR on the viewer's active dataset.

Run against a real kikuchipy signal put into the viewer's module state, eager
and lazy, through the real router. What must hold afterwards is what the rest
of the app reads: the dataset is flagged dirty (indexing and export use the
in-memory patterns), the overview cache is dropped, the provenance mark is on
the signal, and a cancelled or failed run leaves the patterns bit-identical.
"""

import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pyebsdindex")
pytest.importorskip("numba")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.services import nlpar as N  # noqa: E402
from tests.test_nlpar_service import make_scan  # noqa: E402

NAME = "nlpar_test"


@pytest.fixture(autouse=True)
def _restore_viewer_state():
    """The tests put a real signal into the viewer's module state. Restore it
    afterwards so it cannot leak into the next test file."""
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import nlpar as route
    scalars = ("_ebsd_signal", "_ebsd_file_path", "_active_dataset")
    dicts = ("_loaded_files", "_raw_signals", "_positions", "_dirty_datasets",
             "_signal_masks", "_processing_progress", "_registry_by_file",
             "_overview_cache")
    saved_scalars = {n: getattr(ev, n) for n in scalars}
    saved_dicts = {n: getattr(ev, n).copy() for n in dicts}
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
    route._running.clear()
    route._cancel_events.clear()


def _install(signal, name=NAME):
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import nlpar as route

    ev._loaded_files.clear()
    ev._raw_signals.clear()
    ev._positions.clear()
    ev._dirty_datasets.clear()
    ev._signal_masks.clear()
    ev._overview_cache.clear()
    ev._raw_signals[name] = signal
    ev._positions[name] = (0, 0)
    ev._active_dataset = name
    ev._ebsd_signal = signal
    ev._ebsd_file_path = None
    app = FastAPI()
    app.include_router(ev.router, prefix="/api/ebsd")
    app.include_router(route.router, prefix="/api/ebsd")
    return TestClient(app)


@pytest.fixture(scope="module")
def raw():
    return make_scan()


@pytest.fixture(params=["eager", "lazy"])
def setup(request, raw):
    import kikuchipy as kp

    sig = kp.signals.EBSD(raw.copy())
    if request.param == "lazy":
        sig = sig.as_lazy()
    return _install(sig), sig


def _data(sig):
    return np.asarray(sig.data)


# --------------------------------------------------------------------------- #

def test_denoises_the_active_dataset_in_place(setup, raw):
    client, sig = setup
    r = client.post("/api/ebsd/nlpar", json={"radius": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True and body["radius"] == 3
    assert body["lam"] > 0 and body["n_patterns"] == raw.shape[0] * raw.shape[1]

    ref = N.denoise(raw, radius=3)
    assert np.array_equal(_data(sig), ref.data)
    assert not np.array_equal(_data(sig), raw)
    assert body["lam"] == pytest.approx(ref.lam, abs=1e-4)
    # the signal is a normal eager EBSD afterwards
    assert not sig._lazy and isinstance(sig.data, np.ndarray)


def test_dataset_is_flagged_dirty_and_the_overview_cache_is_dropped(setup):
    from backend.api.routes import ebsd_viewer as ev

    client, _ = setup
    ev._overview_cache[(NAME, "mean")] = ("stale", None)
    ev._overview_cache[("other", "mean")] = ("keep", None)
    assert not ev.is_active_signal_dirty()
    assert client.post("/api/ebsd/nlpar", json={"radius": 2}).status_code == 200
    assert ev.is_active_signal_dirty()
    assert (NAME, "mean") not in ev._overview_cache
    assert ("other", "mean") in ev._overview_cache


def test_provenance_is_written_to_the_signal_metadata(setup):
    client, sig = setup
    body = client.post("/api/ebsd/nlpar", json={"radius": 2}).json()
    info = N.applied_info(sig)
    assert info["radius"] == 2
    assert info["lam"] == pytest.approx(body["lam"], abs=1e-4)
    assert info["sigma_median"] > 0
    assert info["pyebsdindex"]


def test_second_run_on_a_denoised_dataset_is_refused_unless_forced(setup):
    client, sig = setup
    assert client.post("/api/ebsd/nlpar", json={"radius": 2}).status_code == 200
    once = _data(sig).copy()

    r = client.post("/api/ebsd/nlpar", json={"radius": 2})
    assert r.status_code == 409 and "already denoised" in r.json()["detail"]
    assert np.array_equal(_data(sig), once), "a refused request must not touch the data"

    assert client.post("/api/ebsd/nlpar", json={"radius": 2, "force": True}).status_code == 200
    assert not np.array_equal(_data(sig), once)


def test_a_copy_of_a_denoised_dataset_carries_the_mark(setup):
    client, _ = setup
    assert client.post("/api/ebsd/nlpar", json={"radius": 1}).status_code == 200
    copy_name = client.post("/api/ebsd/deepcopy", json={"name": "copy_of_denoised"}).json()["name"]
    assert client.post("/api/ebsd/select-dataset", json={"name": copy_name}).status_code == 200
    assert client.post("/api/ebsd/nlpar", json={"radius": 1}).status_code == 409


def test_progress_ends_complete_and_names_its_stages(setup):
    client, _ = setup
    seen = []
    from backend.api.routes import ebsd_viewer as ev
    real = ev._set_processing_progress

    def spy(rid, **state):
        if rid == "rid-1" and "phase" in state:
            seen.append(state["phase"])
        return real(rid, **state)

    ev._set_processing_progress = spy
    try:
        r = client.post("/api/ebsd/nlpar", json={"radius": 2, "request_id": "rid-1"})
    finally:
        ev._set_processing_progress = real
    assert r.status_code == 200
    snap = client.get("/api/ebsd/processing-progress/rid-1").json()
    assert snap["found"] and snap["stage"] == "complete" and snap["fraction"] == 1.0
    for stage in ("compile", "sigma", "lambda", "filter"):
        assert stage in seen, f"no progress report for stage {stage}: {seen}"


@pytest.mark.parametrize("radius", [0, 4, -1])
def test_radius_outside_1_to_3_is_rejected(setup, raw, radius):
    client, sig = setup
    assert client.post("/api/ebsd/nlpar", json={"radius": radius}).status_code == 400
    assert np.array_equal(_data(sig), raw)


def test_no_data_loaded_is_a_400():
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import nlpar as route

    saved = (ev._ebsd_signal, ev._active_dataset)
    ev._ebsd_signal, ev._active_dataset = None, ""
    try:
        app = FastAPI()
        app.include_router(route.router, prefix="/api/ebsd")
        assert TestClient(app).post("/api/ebsd/nlpar", json={}).status_code == 400
    finally:
        ev._ebsd_signal, ev._active_dataset = saved


def test_scan_smaller_than_3x3_is_a_400():
    import kikuchipy as kp

    small = make_scan(rows=2, cols=6)
    client = _install(kp.signals.EBSD(small.copy()))
    r = client.post("/api/ebsd/nlpar", json={"radius": 1})
    assert r.status_code == 400 and "too small" in r.json()["detail"]


def test_hexagonal_edax_scan_is_refused(setup, raw, monkeypatch):
    from backend.api.routes import ebsd_viewer as ev

    client, sig = setup
    fake = types.ModuleType("edax_hex")
    fake.is_edax_hex_file = lambda path: str(path).endswith("hex.h5")
    monkeypatch.setitem(sys.modules, "edax_hex", fake)
    monkeypatch.setattr(ev, "_ebsd_file_path", "scan_hex.h5")
    r = client.post("/api/ebsd/nlpar", json={"radius": 2})
    assert r.status_code == 400 and "hexagonal" in r.json()["detail"]
    assert np.array_equal(_data(sig), raw)
    # a square-grid file is not caught by the same check
    monkeypatch.setattr(ev, "_ebsd_file_path", "scan_square.h5")
    assert client.post("/api/ebsd/nlpar", json={"radius": 2}).status_code == 200


def test_not_enough_room_for_a_second_copy_is_a_413_and_says_what_to_do(setup, raw, monkeypatch):
    from backend.api.routes import nlpar as route

    client, sig = setup
    monkeypatch.setattr(route, "_available_ram", lambda: 1000)
    r = client.post("/api/ebsd/nlpar", json={"radius": 2})
    assert r.status_code == 413 and "Crop" in r.json()["detail"]
    assert np.array_equal(_data(sig), raw)


def test_the_strip_working_set_counts_towards_the_room_needed(setup, raw, monkeypatch):
    """A tiny scan with plenty of RAM is refused when its strips alone would
    take every free byte: the route must hand the strip cost to the check."""
    from backend.api.routes import nlpar as route
    from backend.api.services import nlpar as svc

    client, sig = setup
    monkeypatch.setattr(route, "_available_ram", lambda: 8 * 1024 ** 3)
    monkeypatch.setattr(svc, "strip_working_bytes", lambda *a, **k: 8 * 1024 ** 3)
    r = client.post("/api/ebsd/nlpar", json={"radius": 2})
    assert r.status_code == 413 and "Crop" in r.json()["detail"]
    assert np.array_equal(_data(sig), raw)


def test_above_the_stash_limit_is_a_413(setup, raw, monkeypatch):
    from backend.api.routes import ebsd_viewer as ev

    client, sig = setup
    monkeypatch.setattr(ev, "_STASH_MATERIALIZE_MAX_BYTES", 1000)
    assert client.post("/api/ebsd/nlpar", json={"radius": 2}).status_code == 413
    assert np.array_equal(_data(sig), raw)


def test_a_run_already_going_on_the_dataset_is_a_409(setup, raw):
    from backend.api.routes import nlpar as route

    client, sig = setup
    route._running.add(NAME)
    r = client.post("/api/ebsd/nlpar", json={"radius": 2})
    assert r.status_code == 409 and "already running" in r.json()["detail"]
    assert np.array_equal(_data(sig), raw)


def test_cancel_leaves_dataset_bit_identical_and_the_next_run_works(setup, raw, monkeypatch):
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import nlpar as route

    client, sig = setup
    real = N.denoise

    def cancelling(data, **kw):
        inner = kw["progress"]

        def progress(stage, frac, done, total):
            inner(stage, frac, done, total)
            if stage == "sigma" and done == 1:
                # through the endpoint, from another thread, as the UI would
                other = TestClient(client.app)
                assert other.post("/api/ebsd/nlpar/cancel/rid-c").json() == {"cancelled": True}

        return real(data, **{**kw, "progress": progress, "strip_rows": 3})

    monkeypatch.setattr(N, "denoise", cancelling)
    r = client.post("/api/ebsd/nlpar", json={"radius": 3, "request_id": "rid-c"})
    assert r.status_code == 200 and r.json() == {"success": False, "cancelled": True}
    assert np.array_equal(_data(sig), raw)
    assert N.applied_info(sig) is None
    assert not ev.is_active_signal_dirty()
    snap = client.get("/api/ebsd/processing-progress/rid-c").json()
    assert snap["stage"] == "error" and snap["error"] == "cancelled"
    assert not route._running and "rid-c" not in route._cancel_events

    monkeypatch.setattr(N, "denoise", real)
    assert client.post("/api/ebsd/nlpar", json={"radius": 3}).status_code == 200


def test_cancel_of_an_unknown_request_is_harmless(setup):
    client, _ = setup
    assert client.post("/api/ebsd/nlpar/cancel/never-started").json() == {"cancelled": False}


def test_result_is_discarded_when_the_active_dataset_changed_meanwhile(setup, raw, monkeypatch):
    from backend.api.routes import ebsd_viewer as ev

    client, sig = setup
    real = N.denoise

    def switching(data, **kw):
        res = real(data, **kw)
        ev._raw_signals["elsewhere"] = ev._raw_signals[NAME]
        ev._active_dataset = "elsewhere"
        return res

    monkeypatch.setattr(N, "denoise", switching)
    r = client.post("/api/ebsd/nlpar", json={"radius": 2})
    assert r.status_code == 409 and "discarded" in r.json()["detail"]
    assert np.array_equal(_data(sig), raw)
    assert N.applied_info(sig) is None


def test_a_failing_run_is_a_500_and_leaves_the_data_alone(setup, raw, monkeypatch):
    from backend.api.routes import ebsd_viewer as ev

    client, sig = setup

    def boom(data, **kw):
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(N, "denoise", boom)
    r = client.post("/api/ebsd/nlpar", json={"radius": 2})
    assert r.status_code == 500 and "kernel exploded" in r.json()["detail"]
    assert np.array_equal(_data(sig), raw)
    assert not ev.is_active_signal_dirty()


def test_the_router_is_mounted_in_the_application():
    from backend.api.main import app

    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/ebsd/nlpar" in paths
    assert "/api/ebsd/nlpar/cancel/{request_id}" in paths


# --------------------------------------------------------------------------- #
# export and indexing read the same signal
# --------------------------------------------------------------------------- #

def test_export_writes_the_denoised_patterns(setup, tmp_path):
    """The rich export overwrites the pattern dataset with the ACTIVE signal's data."""
    import h5py
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes.indexing import _overwrite_patterns_with_processed

    client, sig = setup
    assert client.post("/api/ebsd/nlpar", json={"radius": 2}).status_code == 200
    assert ev.is_active_signal_dirty()          # the condition under which export uses it

    ny, nx, sy, sx = sig.data.shape
    out = tmp_path / "ox.h5"
    with h5py.File(str(out), "w") as h:
        h.create_group("1/EBSD/Data").create_dataset(
            "Processed Patterns", data=np.zeros((ny * nx, sy, sx), np.uint8))
    assert _overwrite_patterns_with_processed(str(out), ev._get_active_signal()) is True
    with h5py.File(str(out), "r") as h:
        written = np.asarray(h["1/EBSD/Data/Processed Patterns"])
    assert np.array_equal(written, _data(sig).reshape(ny * nx, sy, sx))


def test_other_ops_still_work_after_nlpar(setup):
    """Static BG after NLPAR (the recommended order) on the swapped-in array."""
    client, sig = setup
    assert client.post("/api/ebsd/nlpar", json={"radius": 2}).status_code == 200
    r = client.post("/api/ebsd/background-removal", json={"method": "static"})
    assert r.status_code == 200, r.text
    r = client.post("/api/ebsd/background-removal", json={"method": "dynamic"})
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
# inputs a user can reach that must stop before the numba kernels
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("nav", [(1, 9), (9, 1), (2, 2), (2, 9), (1, 1)])
def test_degenerate_scan_shapes_are_a_400_and_untouched(nav):
    import kikuchipy as kp

    data = make_scan(rows=nav[0], cols=nav[1])
    client = _install(kp.signals.EBSD(data.copy()))
    r = client.post("/api/ebsd/nlpar", json={"radius": 1})
    assert r.status_code == 400, r.text
    from backend.api.routes import ebsd_viewer as ev
    assert np.array_equal(np.asarray(ev._get_active_signal().data), data)


def test_a_one_dimensional_navigation_is_a_400():
    import kikuchipy as kp

    data = make_scan(rows=1, cols=12)[0]            # (12, H, W): a line scan
    client = _install(kp.signals.EBSD(data.copy()))
    r = client.post("/api/ebsd/nlpar", json={"radius": 1})
    assert r.status_code == 400 and "2-D scan" in r.json()["detail"]


def test_a_crop_smaller_than_the_window_is_refused_not_run(setup):
    client, _ = setup
    r = client.post("/api/ebsd/crop", json={"shape": "rect", "row0": 1, "col0": 1, "rows": 2, "cols": 5})
    assert r.status_code == 200, r.text
    r = client.post("/api/ebsd/nlpar", json={"radius": 1})
    assert r.status_code == 400 and "too small" in r.json()["detail"]


def test_a_radius_larger_than_the_scan_allows_is_clamped_not_run_as_asked():
    import kikuchipy as kp

    data = make_scan(rows=5, cols=40)               # (2r+1) <= 5  ->  r <= 2
    client = _install(kp.signals.EBSD(data.copy()))
    r = client.post("/api/ebsd/nlpar", json={"radius": 3})
    assert r.status_code == 200, r.text
    assert r.json()["radius"] == 2 and r.json()["radius_requested"] == 3


def test_a_lasso_mask_does_not_change_what_nlpar_sees(setup, raw):
    """A non-rectangular crop is a bounding-box crop plus a navigation mask; the mask only
    limits indexing, the neighbour patterns are real, so the patterns are denoised as a whole."""
    client, _ = setup
    rows, cols = 8, 9
    mask = [bool((r + c) % 2) for r in range(rows) for c in range(cols)]
    r = client.post("/api/ebsd/crop", json={"shape": "lasso", "row0": 2, "col0": 2,
                                             "rows": rows, "cols": cols, "mask": mask})
    assert r.status_code == 200, r.text
    from backend.api.routes import ebsd_viewer as ev
    cropped = np.asarray(ev._get_active_signal().data).copy()
    assert client.post("/api/ebsd/nlpar", json={"radius": 2}).status_code == 200
    expected = N.denoise(cropped, radius=2).data
    assert np.array_equal(np.asarray(ev._get_active_signal().data), expected)


# --------------------------------------------------------------------------- #
# GET /api/ebsd/nlpar/advice
# --------------------------------------------------------------------------- #

def _install_scan(data, lazy=False):
    import kikuchipy as kp

    sig = kp.signals.EBSD(data.copy())
    return _install(sig.as_lazy() if lazy else sig), sig


@pytest.mark.parametrize("lazy", [False, True])
def test_advice_recommends_for_a_noisy_scan_and_advises_against_for_a_clean_one(lazy):
    from tests.test_nlpar_service import _drifting_scan

    client, sig = _install_scan(_drifting_scan(30.0), lazy)
    noisy = client.get("/api/ebsd/nlpar/advice").json()
    assert noisy["recommendation"] == "recommended" and noisy["applied"] is False
    assert noisy["lam"] < noisy["threshold"] and noisy["sigma"] > 0 and noisy["n_patterns"] > 500

    client, sig = _install_scan(_drifting_scan(1.0), lazy)
    clean = client.get("/api/ebsd/nlpar/advice").json()
    assert clean["recommendation"] == "not_recommended"
    assert clean["lam"] >= clean["threshold"]


def test_advice_changes_nothing(setup, raw):
    from backend.api.routes import ebsd_viewer as ev

    client, sig = setup
    assert client.get("/api/ebsd/nlpar/advice").status_code == 200
    assert np.array_equal(_data(sig), raw)
    assert not ev.is_active_signal_dirty() and N.applied_info(sig) is None


def test_advice_says_applied_after_a_run_and_does_not_estimate_again(setup, monkeypatch):
    client, sig = setup
    assert client.post("/api/ebsd/nlpar", json={"radius": 2}).status_code == 200

    def boom(*a, **k):
        raise AssertionError("an applied dataset must not be estimated again")

    monkeypatch.setattr(N, "estimate_noise", boom)
    body = client.get("/api/ebsd/nlpar/advice").json()
    assert body == {"recommendation": "applied", "applied": True, "applied_radius": 2}


def test_advice_follows_the_active_dataset_not_the_file(setup):
    """The viewer copies before every op: the original stays untouched and keeps its advice, the
    copy that was denoised says applied, and a copy of that copy carries the mark on."""
    client, _ = setup
    orig = client.post("/api/ebsd/deepcopy", json={"name": "orig"}).json()["name"]
    assert client.get("/api/ebsd/nlpar/advice").json()["recommendation"] in ("recommended", "not_recommended", "unavailable")
    assert client.post("/api/ebsd/nlpar", json={"radius": 1}).status_code == 200
    assert client.get("/api/ebsd/nlpar/advice").json()["recommendation"] == "applied"
    copy = client.post("/api/ebsd/deepcopy", json={"name": "cp"}).json()["name"]
    client.post("/api/ebsd/select-dataset", json={"name": copy})
    assert client.get("/api/ebsd/nlpar/advice").json()["recommendation"] == "applied"
    client.post("/api/ebsd/select-dataset", json={"name": orig})
    assert client.get("/api/ebsd/nlpar/advice").json()["recommendation"] != "applied"


def test_advice_unavailable_reasons(monkeypatch):
    import kikuchipy as kp
    from backend.api.routes import ebsd_viewer as ev

    client = _install(kp.signals.EBSD(make_scan(rows=4, cols=30)))
    assert client.get("/api/ebsd/nlpar/advice").json()["reason"] == "too_small"

    client = _install(kp.signals.EBSD(np.full((20, 20, 8, 8), 100, np.uint8)))
    body = client.get("/api/ebsd/nlpar/advice").json()
    assert body["recommendation"] == "unavailable" and body["reason"] == "uniform"

    from tests.test_nlpar_service import _drifting_scan
    client = _install(kp.signals.EBSD(_drifting_scan(10.0)))
    fake = types.ModuleType("edax_hex")
    fake.is_edax_hex_file = lambda path: True
    monkeypatch.setitem(sys.modules, "edax_hex", fake)
    monkeypatch.setattr(ev, "_ebsd_file_path", "scan_hex.h5")
    assert client.get("/api/ebsd/nlpar/advice").json()["reason"] == "hex"


def test_advice_without_data_is_a_400():
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import nlpar as route

    saved = (ev._ebsd_signal, ev._active_dataset)
    ev._ebsd_signal, ev._active_dataset = None, ""
    try:
        app = FastAPI()
        app.include_router(route.router, prefix="/api/ebsd")
        assert TestClient(app).get("/api/ebsd/nlpar/advice").status_code == 400
    finally:
        ev._ebsd_signal, ev._active_dataset = saved


def test_result_is_discarded_when_the_data_array_was_replaced_meanwhile(setup, raw, monkeypatch):
    """Another request (background removal, an in-place crop) swapped `signal.data` while the
    worker ran; writing the denoised copy of the OLD array back would silently undo it."""
    client, sig = setup
    real = N.denoise
    replaced = {}

    def replacing(data, **kw):
        res = real(data, **kw)
        replaced["new"] = np.asarray(sig.data).copy() + 1        # what the other request left behind
        sig.data = replaced["new"]
        return res

    monkeypatch.setattr(N, "denoise", replacing)
    r = client.post("/api/ebsd/nlpar", json={"radius": 2})
    assert r.status_code == 409 and "discarded" in r.json()["detail"]
    assert sig.data is replaced["new"], "the other request's data must survive"
    assert N.applied_info(sig) is None
