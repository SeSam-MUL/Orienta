"""The spectrum-fit job and the net-intensity extractor.

Real data: the Oxford reference scan (90 x 120, in this checkout's ``Test_data``) and, when
``ORIENTA_TEST_DATA`` points at a ``Test_data`` that has it, the aluminium-alloy R AB1
scan. Edge cases run on small synthetic H5OINA files written here: an
EDS-only acquisition with window maps and, unless stated otherwise, a
per-pixel spectrum cube drawn from the forward model.

Every test that opens a file closes it and clears the job registry on the way
out, so no background fit and no session leaks into the next test.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np
import psutil
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api.services import eds_spectrum_service as svc
from backend.api.services import h5_session
from backend.eds_spectrum import fit as spectrum_fit
from backend.eds_spectrum import job as spectrum_job
from backend.eds_spectrum.cube import read_meta, sum_spectrum
from backend.eds_spectrum.model import NonlinearParams, default_families
from tests.data_deps import TEST_DATA_DIR, need
from tests.test_eds_spectrum_cube import write_cube
from tests.test_eds_spectrum_fit import AREAS, FAMS, META, truth

SAMPLE_B = ("EBSD_SampleB_extrusion_withPattern Sample_B "
            "Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina")
AB1_7050 = ("batch_test/7050EBSD 70502_R Arbeitsbereich 1 "
            "Elementverteilungsdaten 5.h5oina")
WAIT = 120.0          # generous: a loaded test machine can slow the fit down

NX, NY = 6, 4
N_PX = NX * NY
#: Line areas of the synthetic spectra. O is in the sample but NOT among the
#: exported windows: the spectrum shows it, and the composition must still
#: leave it out (C, N, O only ever come from AZtec's export).
SYNTH_AREAS = {**AREAS, "O K": 3e5}
#: AZtec window labels of the synthetic files, with the energy band (keV)
#: their window maps integrate (any band does; the values only have to be
#: window-like and different from the net maps).
WINDOWS = {
    "Al Kα1": (1.40, 1.58), "Si Kα1": (1.67, 1.82), "Mn Kα1": (5.80, 5.99),
    "Fe Kα1": (6.30, 6.50), "Cu Lα1,2": (0.88, 1.00), "Zn Kα1": (8.53, 8.74),
}


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_session():
    """No open file, no crop, no job before and after every test."""
    from backend.api.services import crop_window as crop_window_service
    from backend.api.services.phase_map_store import get_phase_map_store

    def _reset():
        h5_session.close_file()
        svc.reset()
        crop_window_service.clear_all()
        get_phase_map_store().clear()

    _reset()
    yield
    _reset()


# ---------------------------------------------------------------------------
# synthetic files
# ---------------------------------------------------------------------------
_UNIT_CACHE = {}


def _unit_spectra():
    """Base spectrum (continuum, tails, pile-up) and one unit spectrum per family."""
    if not _UNIT_CACHE:
        ff = truth(META, FAMS, AREAS, NonlinearParams(0.0, 1.0, 0.13, 3e-4))
        base = spectrum_fit.forward(ff, {})
        _UNIT_CACHE["base"] = base
        _UNIT_CACHE["unit"] = {k: spectrum_fit.forward(ff, {k: 1.0}) - base
                               for k in SYNTH_AREAS}
    return _UNIT_CACHE["base"], _UNIT_CACHE["unit"]


def write_eds_file(path, fe_scale=1.0, seed=0, with_cube=True):
    """An EDS-only H5OINA-like file: header, window maps and (optionally) a cube."""
    base, unit = _unit_spectra()
    rng = np.random.default_rng(seed)
    spectra = np.zeros((N_PX, META.n_channels))
    for i in range(N_PX):
        mean = base.copy()
        for k, v in SYNTH_AREAS.items():
            scale = fe_scale if k.startswith("Fe") else 1.0
            mean += 0.25 * v * scale * rng.uniform(0.5, 1.5) * unit[k]
        spectra[i] = rng.poisson(np.clip(mean, 0, None))
    live = 3e-4 * rng.uniform(0.5, 1.5, N_PX)
    write_cube(path, spectra, live, nx=NX, ny=NY)
    energies = META.energies()
    with h5py.File(path, "a") as f:
        w = f.create_group("1/EDS/Data/Window Integral")
        for lab, (lo, hi) in WINDOWS.items():
            band = (energies >= lo) & (energies <= hi)
            w.create_dataset(lab, data=(spectra[:, band].sum(axis=1) / live)
                             .astype(np.float32))
        if not with_cube:
            del f["1/EDS/Data/Spectrum"]
    return Path(path)


@pytest.fixture(scope="module")
def synth_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("eds_service")
    return {
        "A": write_eds_file(d / "A.h5oina", fe_scale=1.0, seed=1),
        "B": write_eds_file(d / "B.h5oina", fe_scale=0.05, seed=2),
        "C": write_eds_file(d / "C.h5oina", fe_scale=0.5, seed=3),
        "nocube": write_eds_file(d / "nocube.h5oina", seed=4, with_cube=False),
    }


def direct_net_maps(path):
    """The net maps computed straight from the fit package, no service involved."""
    with h5py.File(path, "r") as f:
        exported = list(f["1/EDS/Data/Window Integral"].keys())
    meta = read_meta(path)
    y = sum_spectrum(path)
    ff = spectrum_fit.fit_sum_spectrum(y, meta, default_families(exported, meta))
    return ff, spectrum_fit.project_cube(path, ff, y_sum=y), y


def window_map(path, label):
    with h5py.File(path, "r") as f:
        return f[f"1/EDS/Data/Window Integral/{label}"][()]


@pytest.fixture
def sampleb():
    data_dir = Path(os.environ.get("ORIENTA_TEST_DATA") or TEST_DATA_DIR)
    path = data_dir / SAMPLE_B
    if not path.is_file():
        path = need(TEST_DATA_DIR / SAMPLE_B)
    return path


def _open_and_wait(path):
    h5_session.open_file(str(path))
    return svc.get_result(str(path), wait_s=WAIT)


@pytest.fixture
def in_thread(monkeypatch):
    """Run the job in a thread of THIS process, as the job ran before it moved to a child.

    The product runs it in a child process (``svc.RUN_IN_CHILD``), which a
    monkeypatch of the fit's functions cannot reach. Tests that hold or break
    the fit by patching ``svc.fit_sum_spectrum`` / ``svc.project_cube`` ask
    for this; everything else runs the child, the product path.
    """
    monkeypatch.setattr(svc, "RUN_IN_CHILD", False)


@pytest.fixture
def blocked_fit(monkeypatch):
    """Make the fit step of the job wait until the test lets it go.

    In a thread of this process (see ``in_thread``; set here directly so the
    modules that import this fixture need nothing else).
    """
    monkeypatch.setattr(svc, "RUN_IN_CHILD", False)
    gate = threading.Event()
    real = svc.fit_sum_spectrum

    def slow(*args, **kwargs):
        if not gate.wait(WAIT):
            raise TimeoutError("test never released the fit")
        return real(*args, **kwargs)

    monkeypatch.setattr(svc, "fit_sum_spectrum", slow)
    yield gate
    gate.set()


@pytest.fixture
def slow_job(blocked_fit, synth_files):
    """File A open, its fit held in the 'fitting' stage."""
    path = synth_files["A"]
    h5_session.open_file(str(path))
    t0 = time.monotonic()
    while svc.status(str(path))["stage"] != "fitting":
        assert time.monotonic() - t0 < WAIT, svc.status(str(path))
        time.sleep(0.01)
    yield path
    blocked_fit.set()
    svc.get_result(str(path), wait_s=WAIT)


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------
def test_open_starts_job_and_status_reaches_ready(sampleb, client):
    t0 = time.perf_counter()
    h5_session.open_file(str(sampleb))
    first = svc.status(str(sampleb))
    assert first["state"] in ("running", "ready"), first

    res = svc.get_result(str(sampleb), wait_s=WAIT)
    elapsed = time.perf_counter() - t0
    st = svc.status(str(sampleb))
    assert res is not None
    assert (st["state"], st["stage"], st["progress"], st["error"]) == (
        "ready", "done", 1.0, None)
    fit = st["fit"]
    assert fit["model_version"] == "eds-spectrum-fit/1"
    assert np.isfinite(fit["chi2_red"]) and fit["chi2_red"] < 8  # 5.18 measured (T3)
    assert set(fit["params"]) == {"offset_kev", "gain", "fwhm_mnka_kev",
                                  "absorption_a"}
    # the line per element AFTER the spectrum-source policy: with the file's
    # geometry (matrix correction) Cu through K like every other element
    assert fit["lines"]["Cu"]["family"] == "K"
    assert fit["lines"]["Cu"]["label"] == "Cu Kα1"
    assert {fit["lines"][el]["family"] for el in ("Al", "Si", "Fe", "Mn", "Zn")} == {"K"}

    # detections and quality warnings are the fit package's own verdicts
    ff, _maps, y = direct_net_maps(sampleb)
    exported = {"Al", "C", "Cu", "Fe", "Mn", "O", "Si", "Zn"}
    expected = sorted((d.element, d.family) for d in spectrum_fit.detect(ff)
                      if d.element not in exported | {"C", "N", "O"})
    assert sorted((d["element"], d["family"])
                  for d in fit["detected_not_exported"]) == expected
    assert all(d["z"] >= 5 for d in fit["detected_not_exported"])
    assert sorted(q["family"] for q in fit["quality_warnings"]) == sorted(
        q.family for q in spectrum_fit.quality(ff, y))

    # the route says the same thing, as JSON
    r = client.get("/api/eds/spectrum-fit/status")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "ready"
    assert r.json()["fit"]["lines"]["Cu"]["family"] == "K"
    print(f"\nSampleB open -> ready: {elapsed:.2f} s "
          f"(job reports {st['elapsed_s']:.2f} s)")


def test_one_job_per_key(synth_files):
    path = str(synth_files["A"])
    svc.start_for_file(path)
    job = svc._job_for(path, start=False)
    svc.start_for_file(path)
    # the same file spelled differently is the same key
    svc.start_for_file(os.path.dirname(path).replace("\\", "/") + "/./A.h5oina")
    assert svc._job_for(path, start=False) is job
    assert svc.get_result(path, wait_s=WAIT) is not None


def test_registry_keeps_at_most_two_files(synth_files):
    for name in ("A", "B", "C"):
        svc.start_for_file(str(synth_files[name]))
    assert len(svc._registry) <= 2
    assert svc._job_for(str(synth_files["A"]), start=False) is None
    assert svc.get_result(str(synth_files["C"]), wait_s=WAIT) is not None


def test_failure_inside_the_job_is_reported(monkeypatch, in_thread, synth_files,
                                            client):
    def boom(*_a, **_k):
        raise ValueError("synthetic fit failure")

    monkeypatch.setattr(svc, "fit_sum_spectrum", boom)
    path = str(synth_files["A"])
    h5_session.open_file(path)
    with pytest.raises(svc.SpectrumFitFailed):
        svc.get_result(path, wait_s=WAIT)
    st = svc.status(path)
    assert (st["state"], st["stage"]) == ("failed", "failed")
    assert "synthetic fit failure" in st["error"]

    r = client.post("/api/eds/quantify/pixel", json={"row": 1, "col": 1})
    assert r.status_code == 500, r.text
    assert r.json()["detail"]["code"] == "eds_spectrum_fit_failed"
    assert "synthetic fit failure" in r.json()["detail"]["error"]


def test_a_failure_to_start_never_fails_the_open(monkeypatch, synth_files, caplog):
    def boom(_path):
        raise RuntimeError("cannot start")

    monkeypatch.setattr(svc, "start_for_file", boom)
    extractor, fmt = h5_session.open_file(str(synth_files["A"]))
    assert extractor is h5_session.get_extractor()
    assert "cannot start" in caplog.text


def test_pending_and_failed_are_http_exceptions():
    p = svc.SpectrumFitPending(0.42, "fitting")
    assert isinstance(p, HTTPException) and p.status_code == 503
    assert p.detail["code"] == "eds_spectrum_fit_running"
    assert p.detail["progress"] == pytest.approx(0.42) and p.progress == 0.42
    assert p.detail["message"]
    f = svc.SpectrumFitFailed("x went wrong")
    assert isinstance(f, HTTPException) and f.status_code == 500
    assert f.detail["code"] == "eds_spectrum_fit_failed"
    assert f.detail["error"] == "x went wrong" and f.detail["message"]


# ---------------------------------------------------------------------------
# A request during the fit answers 503, never window numbers
# ---------------------------------------------------------------------------
def test_route_during_fit_answers_503_running(slow_job, client):
    r = client.post("/api/eds/quantify/pixel", json={"row": 1, "col": 1})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "eds_spectrum_fit_running"
    assert 0.0 < r.json()["detail"]["progress"] < 1.0

    fe = quote("Fe Kα1")
    # The element list is not a number: its labels come from the file's
    # header, so it answers at once and says the numbers are still coming
    # (test_elements_answer_during_the_fit_with_the_finished_labels).
    r = client.get("/api/eds/elements")
    assert r.status_code == 200, r.text
    assert r.json()["eds_status"]["code"] == "eds_spectrum_fit_running"
    for method, url, body in [
        ("get", f"/api/eds/map/{fe}?mode=at_pct", None),
        ("get", f"/api/eds/map/{fe}?mode=counts", None),
        ("post", "/api/eds/region-quantify",
         {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1}),
        ("post", "/api/eds/suggest-phases", {"row": 1, "col": 1}),
        ("post", "/api/eds/chemistry-mask",
         {"filters": [{"element": "Fe", "operator": ">", "min_val": 1.0}]}),
        ("post", "/api/eds/linescan",
         {"start_row": 0, "start_col": 0, "end_row": 3, "end_col": 5,
          "layers": ["eds-Fe Kα1"]}),
        ("get", f"/api/h5/eds/map/{fe}?scope=dataset", None),
    ]:
        r = getattr(client, method)(url, **({"json": body} if body else {}))
        assert r.status_code == 503, (url, r.status_code, r.text)
        assert r.json()["detail"]["code"] == "eds_spectrum_fit_running", url

    st = client.get("/api/eds/spectrum-fit/status").json()
    assert (st["state"], st["stage"]) == ("running", "fitting")
    assert 0.0 < st["progress"] < 1.0 and st["fit"] is None

    # The hover probe is a composite (EDS, band contrast, phase): it keeps
    # its other layers and says why the EDS part is missing.
    r = client.post("/api/eds/probe", json={"row": 1, "col": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["elements"] == {}
    assert {"bc", "phase", "row", "col"} <= set(body)
    assert body["eds_status"]["state"] == "pending"
    assert body["eds_status"]["code"] == "eds_spectrum_fit_running"
    assert 0.0 < body["eds_status"]["progress"] < 1.0

    # the H5 viewer still shows the file as it is
    r = client.get(f"/api/h5/eds/map/{fe}")
    assert r.status_code == 200, r.text
    assert r.json()["max_val"] == pytest.approx(
        float(window_map(slow_job, "Fe Kα1").max()), rel=1e-6)


def _seed_region_map(file_path=None, phase_index=-1):
    """A classification with one region over the whole synthetic grid, so the
    region payload has a composition to (not) compute."""
    from backend.api.services.cif_phase_library import CifPhaseEntry
    from backend.api.services.phase_map_store import get_phase_map_store

    entry = CifPhaseEntry(
        key="test-phase", cif_filename="test.cif", formula="Al",
        space_group="Fm-3m", space_group_number=225, crystal_system="cubic",
        composition={"Al": 100.0}, elements=["Al"],
    )
    store = get_phase_map_store()
    store.set_classification(
        phase_grid=np.full((NY, NX), phase_index, dtype=np.int32),
        score_grid=np.zeros((NY, NX), dtype=np.float32),
        phase_entries=[entry], tolerance=15.0, min_score=0.3,
        file_path=file_path,
        region_grid=np.zeros((NY, NX), dtype=np.int32),
        region_phase=[phase_index],
    )
    return store


def test_composite_payloads_degrade_and_eds_endpoints_raise(slow_job):
    """The phase-map payload is not an EDS number: it degrades with a marker.
    The region inspector and a preset's portability record ARE EDS numbers:
    they raise, so "still running" never reads as "no composition"."""
    from types import SimpleNamespace

    from backend.api.routes import eds, eds_export

    state = SimpleNamespace(region_grid=np.zeros((NY, NX), dtype=np.int32),
                            n_rows=NY, n_cols=NX, region_phase=[-1],
                            phase_entries=[])
    notes = {}
    regions = eds._regions_payload(state, notes)
    assert [r["region_id"] for r in regions] == [0]
    assert regions[0]["mean_at_pct"] == {}
    assert notes["eds_status"]["code"] == "eds_spectrum_fit_running"
    with pytest.raises(svc.SpectrumFitPending):
        eds._region_detail(state, 0)
    with pytest.raises(svc.SpectrumFitPending):
        eds_export._authored_against_loaded_file([], None)


def test_a_phase_map_write_during_the_fit_is_applied_once(slow_job, client):
    store = _seed_region_map()
    r = client.post("/api/eds/phase-map/assign-region",
                    json={"row_start": 0, "row_end": 2, "col_start": 0,
                          "col_end": 2, "phase_index": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_overwritten"] == 9
    assert body["eds_status"]["code"] == "eds_spectrum_fit_running"
    assert body["regions"][0]["mean_at_pct"] == {}
    assert int((store.get_state().phase_grid == 0).sum()) == 9
    # one write, one undo step: undoing it once restores the map
    assert store.undo()
    assert int((store.get_state().phase_grid == 0).sum()) == 0

    r = client.get("/api/eds/phase-map")
    assert r.status_code == 200, r.text
    assert r.json()["eds_status"]["state"] == "pending"


def test_phase_map_after_a_failed_fit_still_shows_the_saved_map(
        monkeypatch, in_thread, tmp_path, client):
    from backend.api.services.phase_map_store import _sidecar_path_for

    def boom(*_a, **_k):
        raise ValueError("synthetic fit failure")

    monkeypatch.setattr(svc, "fit_sum_spectrum", boom)
    path = write_eds_file(tmp_path / "saved.h5oina", seed=7)
    h5_session.open_file(str(path))
    with pytest.raises(svc.SpectrumFitFailed):
        svc.get_result(str(path), wait_s=WAIT)
    _seed_region_map(file_path=str(path), phase_index=0)   # autosaves
    assert _sidecar_path_for(str(path)).is_file()

    h5_session.close_file()
    h5_session.open_file(str(path))          # restores the saved map
    with pytest.raises(svc.SpectrumFitFailed):
        svc.get_result(str(path), wait_s=WAIT)
    r = client.get("/api/eds/phase-map")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["loaded"] is True and body["n_classified"] == N_PX
    assert body["eds_status"]["state"] == "failed"
    assert body["eds_status"]["code"] == "eds_spectrum_fit_failed"
    assert "synthetic fit failure" in body["eds_status"]["error"]


def test_preset_save_during_the_fit_answers_503_and_writes_nothing(
        slow_job, client, tmp_path):
    from backend.api.services import eds_presets as presets_svc

    presets = tmp_path / "presets"
    presets.mkdir()
    presets_svc.set_preset_dir_for_test(presets)
    try:
        r = client.post("/api/eds/presets", json={
            "name": "during the fit",
            "settings": {"mode": "cluster", "min_score": 0.4},
            # the UI sends these, so the save needs no map of its own --
            # and is still refused while the numbers are being made
            "authored_elements": ["Al", "Fe"], "authored_step_um": 0.5})
        assert r.status_code == 503, r.text
        assert r.json()["detail"]["code"] == "eds_spectrum_fit_running"
        assert list(presets.iterdir()) == []
    finally:
        presets_svc.set_preset_dir_for_test(None)


def test_auto_classify_during_the_fit_refuses_before_anything_changes(
        slow_job, client):
    """Auto-classify replaces the map from EDS numbers: it refuses first --
    before loading the CIF library, long before touching the map."""
    store = _seed_region_map(phase_index=0)
    before = store.get_state().phase_grid.copy()
    r = client.post("/api/eds/auto-classify", json={})
    assert r.status_code == 503, r.text
    assert r.json()["detail"]["code"] == "eds_spectrum_fit_running"
    assert np.array_equal(store.get_state().phase_grid, before)


# ---------------------------------------------------------------------------
# a failed fit is not permanent
# ---------------------------------------------------------------------------
def test_a_failed_fit_is_retried_when_the_file_is_opened_again(
        monkeypatch, in_thread, tmp_path):
    real = svc.fit_sum_spectrum
    calls = []

    def fails_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("synthetic: out of memory once")
        return real(*args, **kwargs)

    monkeypatch.setattr(svc, "fit_sum_spectrum", fails_once)
    path = str(write_eds_file(tmp_path / "retry.h5oina", seed=8))
    h5_session.open_file(path)
    with pytest.raises(svc.SpectrumFitFailed):
        svc.get_result(path, wait_s=WAIT)
    # a read does not retry (that would refit on every request) ...
    with pytest.raises(svc.SpectrumFitFailed):
        svc.get_result(path, wait_s=WAIT)
    assert len(calls) == 1
    # ... opening the file again does
    h5_session.open_file(path)
    assert svc.get_result(path, wait_s=WAIT) is not None
    assert svc.status(path)["state"] == "ready"


def test_a_failed_probe_is_retried_when_the_file_is_opened_again(
        monkeypatch, tmp_path):
    real = svc._probe
    calls = []

    def fails_once(p):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("synthetic: drive not ready")
        return real(p)

    monkeypatch.setattr(svc, "_probe", fails_once)
    path = str(write_eds_file(tmp_path / "probe.h5oina", seed=9))
    h5_session.open_file(path)
    assert svc.status(path)["state"] == "failed"
    h5_session.open_file(path)
    assert svc.get_result(path, wait_s=WAIT) is not None


@pytest.fixture
def fresh_viewer():
    """The EBSD viewer's module state, empty before and after the test."""
    from backend.api.routes import ebsd_viewer

    def _reset():
        ebsd_viewer._registry_by_file.clear()
        ebsd_viewer._raw_signals.clear()
        ebsd_viewer._positions.clear()
        ebsd_viewer._dirty_datasets.clear()
        ebsd_viewer._signal_masks.clear()
        ebsd_viewer._overview_cache.clear()
        ebsd_viewer._loaded_files.clear()
        ebsd_viewer._active_dataset = ""
        ebsd_viewer._ebsd_file_path = None
        ebsd_viewer._ebsd_signal = None

    _reset()
    yield
    _reset()


def _fit_fails_once(monkeypatch):
    monkeypatch.setattr(svc, "RUN_IN_CHILD", False)     # the patch must reach it
    real = svc.fit_sum_spectrum
    calls = []

    def fails_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("synthetic: out of memory once")
        return real(*args, **kwargs)

    monkeypatch.setattr(svc, "fit_sum_spectrum", fails_once)
    return calls


def _reload_retries(client, path, again_url, calls):
    r = client.post("/api/ebsd/load", json={"path": str(path)})
    assert r.status_code == 200, r.text
    with pytest.raises(svc.SpectrumFitFailed):
        svc.get_result(str(path), wait_s=WAIT)
    # the session still holds the file, so the loader skips open_file
    assert h5_session.get_current_path() == str(path)
    r = client.post(again_url, json={"path": str(path)})
    assert r.status_code == 200, r.text
    assert svc.get_result(str(path), wait_s=WAIT) is not None
    assert svc.status(str(path))["state"] == "ready" and len(calls) == 2


@pytest.mark.parametrize("again", ["/api/ebsd/load", "/api/ebsd/switch-file"])
def test_loading_an_eds_only_file_again_retries_its_failed_fit(
        again, monkeypatch, tmp_path, client, fresh_viewer):
    calls = _fit_fails_once(monkeypatch)
    path = write_eds_file(tmp_path / "reload.h5oina", seed=10)
    _reload_retries(client, path, again, calls)


@pytest.mark.parametrize("again", ["/api/ebsd/load", "/api/ebsd/switch-file"])
def test_loading_an_ebsd_file_again_retries_its_failed_fit(
        again, monkeypatch, sampleb, client, fresh_viewer):
    calls = _fit_fails_once(monkeypatch)
    _reload_retries(client, sampleb, again, calls)


def test_old_versions_of_the_open_file_are_evicted(tmp_path):
    """The open file is protected by its CURRENT key; a version of it with an
    older mtime is just an old entry (``_lookup`` can never reach it)."""
    path = str(write_eds_file(tmp_path / "mtime.h5oina", seed=11))
    h5_session.open_file(path)
    st = os.stat(path)
    for i in range(1, 5):
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + i * 10**9))
        svc.start_for_file(path)
    assert len(svc._registry) <= svc.MAX_FILES
    assert svc._job_for(path, start=False) is not None


def test_eviction_never_drops_the_open_file(synth_files):
    path_a = str(synth_files["A"])
    h5_session.open_file(path_a)                 # the session's file
    svc.get_result(path_a, wait_s=WAIT)
    for name in ("B", "C"):                      # e.g. the compare script
        svc.start_for_file(str(synth_files[name]))
    assert len(svc._registry) <= 2
    assert svc._job_for(path_a, start=False) is not None
    assert svc._job_for(str(synth_files["B"]), start=False) is None


def test_a_stale_extractor_does_not_restart_its_fit(synth_files):
    path_a = str(synth_files["A"])
    h5_session.open_file(path_a)
    svc.get_result(path_a, wait_s=WAIT)
    ext_a = h5_session.get_active_extractor()
    h5_session.open_file(str(synth_files["B"]))
    h5_session.open_file(str(synth_files["C"]))  # A is evicted now
    assert svc._job_for(path_a, start=False) is None
    with pytest.raises(svc.SpectrumFitFailed) as exc:
        ext_a.get_element_map("Fe Kα1")
    assert "no longer" in exc.value.detail["error"]
    assert svc._job_for(path_a, start=False) is None


def test_an_ebsd_load_during_the_fit_still_reports_eds(blocked_fit, sampleb,
                                                        client):
    """Loading a file is not an EDS read: it must not wait for, or fail on,
    the spectrum fit it starts, and "has EDS" is true from the first moment."""
    from backend.api.routes import ebsd_viewer

    def _reset():
        ebsd_viewer._registry_by_file.clear()
        ebsd_viewer._raw_signals.clear()
        ebsd_viewer._positions.clear()
        ebsd_viewer._dirty_datasets.clear()
        ebsd_viewer._signal_masks.clear()
        ebsd_viewer._overview_cache.clear()
        ebsd_viewer._active_dataset = ""
        ebsd_viewer._ebsd_file_path = None
        ebsd_viewer._ebsd_signal = None

    _reset()
    try:
        r = client.post("/api/ebsd/load", json={"path": str(sampleb)})
        assert r.status_code == 200, r.text
        assert svc.status(str(sampleb))["state"] == "running"
        with h5py.File(sampleb, "r") as f:
            exported = sorted(f["1/EDS/Data/Window Integral"].keys())
        active = [d for d in client.get("/api/ebsd/datasets").json()["datasets"]
                  if d["active"]]
        assert active and active[0]["has_eds"] is True
        assert sorted(active[0]["eds_elements"]) == exported
        r = client.get(f"/api/eds/map/{quote('Fe Kα1')}")
        assert r.status_code == 503, r.text
    finally:
        blocked_fit.set()
        svc.get_result(str(sampleb), wait_s=WAIT)
        _reset()


def test_get_result_never_blocks_by_default(slow_job):
    t0 = time.perf_counter()
    with pytest.raises(svc.SpectrumFitPending) as exc:
        svc.get_result(str(slow_job))
    assert time.perf_counter() - t0 < 0.5
    assert exc.value.detail["code"] == "eds_spectrum_fit_running"
    ext = h5_session.get_active_extractor()
    with pytest.raises(svc.SpectrumFitPending):
        ext.get_element_map("Fe Kα1")
    with pytest.raises(svc.SpectrumFitPending):
        ext.get_detected_only_elements()
    # the labels come from the header and need no fit
    assert set(WINDOWS) <= set(ext.get_available_elements())
    # raw window data needs no fit
    assert np.array_equal(ext.get_window_map("Fe Kα1"),
                          window_map(slow_job, "Fe Kα1"))


# ---------------------------------------------------------------------------
# Crops cut net maps exactly like window maps
# ---------------------------------------------------------------------------
def test_crop_cuts_net_maps_like_window_maps(sampleb, monkeypatch, client):
    from backend.api.services.crop_window import CropWindow
    from backend.api.services.cropped_extractor import CroppedExtractor

    res = _open_and_wait(sampleb)
    rows, cols = h5_session.get_extractor().get_grid_dimensions()
    window = CropWindow(str(sampleb), 10, 20, 30, 40, (rows, cols))
    monkeypatch.setattr("backend.api.routes.ebsd_viewer.get_active_crop_window",
                        lambda: window)

    ext = h5_session.get_active_extractor()
    assert isinstance(ext, CroppedExtractor)
    assert isinstance(ext.raw, svc.SpectrumNetExtractor)
    assert ext.raw.raw is h5_session.get_extractor()
    assert ext.quant_source == "spectrum"

    net_crop = ext.get_element_map("Fe Kα1")
    window_crop = ext.get_window_map("Fe Kα1")
    assert net_crop.shape == window_crop.shape == (30 * 40,)
    full_net = res.maps["Fe K"].reshape(rows, cols)
    full_win = window_map(sampleb, "Fe Kα1").reshape(rows, cols)
    assert np.array_equal(net_crop, full_net[10:40, 20:60].ravel())
    assert np.array_equal(window_crop, full_win[10:40, 20:60].ravel())
    assert ext.get_element_map_2d("Fe Kα1").shape == (30, 40)

    r = client.get(f"/api/eds/map/{quote('Fe Kα1')}")
    assert r.status_code == 200, r.text
    assert r.json()["shape"] == [30, 40]


# ---------------------------------------------------------------------------
# A job for file A never delivers its maps to file B
# ---------------------------------------------------------------------------
def test_file_switch_never_leaks_maps(monkeypatch, in_thread, synth_files):
    path_a, path_b = str(synth_files["A"]), str(synth_files["B"])
    release_a = threading.Event()
    real = svc.project_cube

    def held_for_a(h5file, *args, **kwargs):
        if Path(h5file.filename).name == "A.h5oina":
            if not release_a.wait(WAIT):
                raise TimeoutError("test never released A")
        return real(h5file, *args, **kwargs)

    monkeypatch.setattr(svc, "project_cube", held_for_a)
    _ff_a, maps_a, _ = direct_net_maps(path_a)
    _ff_b, maps_b, _ = direct_net_maps(path_b)
    assert not np.allclose(maps_a["Fe K"], maps_b["Fe K"])  # the test can bite

    h5_session.open_file(path_a)
    ext_a = h5_session.get_active_extractor()
    h5_session.open_file(path_b)                       # A still running
    assert svc.get_result(path_b, wait_s=WAIT) is not None
    ext_b = h5_session.get_active_extractor()
    assert np.allclose(ext_b.get_element_map("Fe Kα1"), maps_b["Fe K"],
                       rtol=1e-5, atol=1e-3)
    with pytest.raises(svc.SpectrumFitPending):         # A's view: A's own job
        ext_a.get_element_map("Fe Kα1")

    release_a.set()                                    # A finishes AFTER B
    assert svc.get_result(path_a, wait_s=WAIT) is not None
    for ext in (ext_b, h5_session.get_active_extractor()):
        assert np.allclose(ext.get_element_map("Fe Kα1"), maps_b["Fe K"],
                           rtol=1e-5, atol=1e-3)
    assert np.allclose(ext_a.get_element_map("Fe Kα1"), maps_a["Fe K"],
                       rtol=1e-5, atol=1e-3)


# ---------------------------------------------------------------------------
# files without a cube, the raw extractor, the forced window source
# ---------------------------------------------------------------------------
def test_file_without_cube_keeps_window_path(synth_files, client):
    path = synth_files["nocube"]
    h5_session.open_file(str(path))
    assert svc.status(str(path))["state"] == "none"
    assert svc.get_result(str(path)) is None
    ext = h5_session.get_active_extractor()
    assert getattr(ext, "quant_source", "window") == "window"
    assert ext is h5_session.get_extractor()

    r = client.post("/api/eds/quantify/pixel", json={"row": 1, "col": 1})
    assert r.status_code == 200, r.text
    fe = r.json()["data"]["Fe"]["counts"]
    assert fe == pytest.approx(float(window_map(path, "Fe Kα1")[1 * NX + 1]),
                               rel=1e-6)


def test_raw_extractor_unchanged(sampleb, client):
    res = _open_and_wait(sampleb)
    raw = h5_session.get_extractor()
    assert type(raw).__name__ == "H5OINADataExtractor"
    assert np.array_equal(raw.get_element_map("Fe Kα1"),
                          window_map(sampleb, "Fe Kα1"))
    with h5py.File(sampleb, "r") as f:
        exported = sorted(f["1/EDS/Data/Window Integral"].keys())
    assert raw.get_available_elements() == exported

    active = h5_session.get_active_extractor()
    assert isinstance(active, svc.SpectrumNetExtractor) and active.raw is raw
    assert active.quant_source == "spectrum"
    _ff, maps, _y = direct_net_maps(sampleb)
    net = active.get_element_map("Fe Kα1")
    assert net.dtype == np.float32 and net.shape == (90 * 120,)
    assert np.allclose(net, maps["Fe K"], rtol=1e-5, atol=1e-3)
    assert np.array_equal(net, res.maps["Fe K"])
    assert not np.allclose(net, window_map(sampleb, "Fe Kα1"))
    assert np.array_equal(active.get_window_map("Fe Kα1"),
                          window_map(sampleb, "Fe Kα1"))
    # the returned map is a copy: a caller that clips in place hurts nobody
    net[:] = -1
    assert not np.array_equal(active.get_element_map("Fe Kα1"), net)

    r = client.get(f"/api/h5/eds/map/{quote('Fe Kα1')}")
    assert r.json()["max_val"] == pytest.approx(
        float(window_map(sampleb, "Fe Kα1").max()), rel=1e-6)


def test_force_window_source(synth_files):
    path = str(synth_files["A"])
    _open_and_wait(path)
    assert isinstance(h5_session.get_active_extractor(), svc.SpectrumNetExtractor)
    with svc.force_window_source():
        assert h5_session.get_active_extractor() is h5_session.get_extractor()
    assert isinstance(h5_session.get_active_extractor(), svc.SpectrumNetExtractor)


def test_labels_offer_every_fitted_family(synth_files):
    path = str(synth_files["A"])
    _open_and_wait(path)
    ext = h5_session.get_active_extractor()
    labels = ext.get_available_elements()
    # AZtec's exact labels for exported families ...
    for lab in WINDOWS:
        assert lab in labels, (lab, labels)
    # ... plus the fitted families AZtec did not export, for the same elements
    for lab in ("Cu Kα1", "Fe Lα1,2", "Mn Lα1,2", "Zn Lα1,2"):
        assert lab in labels, (lab, labels)
    # O is detected (it is in the spectra) but was not exported: no label
    fit = svc.status(path)["fit"]
    assert ("O", "K") in {(d.element, d.family)
                          for d in svc.get_result(path).detections
                          if d.z >= 5}
    assert not any(d["element"] == "O" for d in fit["detected_not_exported"])
    assert not any(lab.split()[0] in ("C", "N", "O") for lab in labels)
    assert np.array_equal(ext.get_element_map("Cu Kα1"),
                          svc.get_result(path).maps["Cu K"])
    assert ext.get_element_map("Pb Mα") is None


@pytest.mark.parametrize("which", ["sampleb", "7050"])
def test_detected_element_is_available(which, sampleb):
    if which == "sampleb":
        path = sampleb
    else:
        data_dir = os.environ.get("ORIENTA_TEST_DATA")
        if not data_dir:
            pytest.skip("set ORIENTA_TEST_DATA to a Test_data folder with "
                        f"{AB1_7050}")
        path = need(Path(data_dir) / AB1_7050)
    res = _open_and_wait(path)
    ext = h5_session.get_active_extractor()
    labels = ext.get_available_elements()
    with h5py.File(path, "r") as f:
        exported = list(f["1/EDS/Data/Window Integral"].keys())
    for lab in exported:                     # every exported window survives
        assert lab in labels, (lab, labels)
    exported_symbols = {lab.split()[0] for lab in exported}
    fit = svc.status(str(path))["fit"]
    # The composition is AZtec's exported elements only: a detection adds no label to it ...
    assert {lab.split()[0] for lab in labels} <= exported_symbols
    detected = ext.get_detected_only_elements()
    assert svc.detected_only_labels(ext) == detected
    for d in fit["detected_not_exported"]:
        assert d["element"] not in exported_symbols | {"C", "N", "O"}
        lab = {"K": f"{d['element']} Kα1", "L": f"{d['element']} Lα1,2",
               "M": f"{d['element']} Mα"}[d["family"]]
        # ... it is listed with its map label and its size in the map sum,
        assert d["label"] == lab
        assert d["at_pct_map_sum"] is not None and d["at_pct_map_sum"] > 0
        # ... and its net map stays reachable ("add as map").
        assert lab in detected and lab not in labels, (lab, detected, labels)
        assert np.array_equal(ext.get_element_map(lab),
                              res.maps[f"{d['element']} {d['family']}"])
    assert {lab.split()[0] for lab in detected} == {
        d["element"] for d in fit["detected_not_exported"]}
    assert not any(str(el) in {d["element"] for d in fit["detected_not_exported"]}
                   for el in res.lines)
    if which == "sampleb":
        # Ti K at 5.2 sigma is the one element AZtec did
        # not export on this scan
        assert "Ti Kα1" in detected


# ---------------------------------------------------------------------------
# The job runs in a child process
#
# In the backend's own process the cube read stalled every request: on the
# 29 GB scan /api/health took 9.6-11.4 s and a pattern request 20.8 s while
# it ran. A child shares neither the
# interpreter lock nor h5py's lock with the request threads.
# ---------------------------------------------------------------------------
def _job(path):
    job = svc._job_for(str(path), start=False)
    assert job is not None, path
    return job


def _child(path):
    """``path``'s child process (a ``Popen``) as soon as it has been started."""
    job = _job(path)
    t0 = time.monotonic()
    while job.process is None:
        assert job.state == "running", svc.status(str(path))
        assert time.monotonic() - t0 < WAIT, "the child was never started"
        time.sleep(0.002)
    return job.process


def _hold(path):
    """Suspend ``path``'s child at once -- it is still importing then, so it
    is held before it has read anything. Returns the ``psutil.Process``."""
    held = psutil.Process(_child(path).pid)
    held.suspend()
    return held


def _gone(proc, timeout=WAIT):
    """True once the child ``proc`` has exited."""
    t0 = time.monotonic()
    while proc.poll() is None:
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.02)
    return True


def _max_rel_diff(a, b):
    scale = float(np.max(np.abs(b))) or 1.0
    return float(np.max(np.abs(np.asarray(a, np.float64) - b))) / scale


def test_the_job_runs_in_a_child_and_delivers_the_in_process_maps(
        synth_files, monkeypatch):
    path = str(synth_files["A"])
    assert svc.RUN_IN_CHILD                        # the product's setting
    h5_session.open_file(path)
    proc = _child(path)
    assert proc.pid != os.getpid()
    res = svc.get_result(path, wait_s=WAIT)
    assert _gone(proc) and proc.returncode == 0
    fit_child = svc.status(path)["fit"]

    # the fit package, straight, in this process
    _ff, maps, _y = direct_net_maps(path)
    assert sorted(res.maps) == sorted(maps)
    for fam, m in maps.items():
        assert res.maps[fam].dtype == m.dtype and res.maps[fam].shape == m.shape
        assert _max_rel_diff(res.maps[fam], m) <= 1e-6, fam

    # the same job in a thread of this process
    h5_session.close_file()
    svc.reset()
    monkeypatch.setattr(svc, "RUN_IN_CHILD", False)
    h5_session.open_file(path)
    res_t = svc.get_result(path, wait_s=WAIT)
    assert _job(path).process is None
    for fam in maps:
        assert _max_rel_diff(res.maps[fam], res_t.maps[fam]) <= 1e-6, fam
    np.testing.assert_array_equal(res.live_time, res_t.live_time)
    assert res.labels == res_t.labels and res.lines == res_t.lines
    assert res.detected_labels == res_t.detected_labels
    fit_thread = svc.status(path)["fit"]
    assert fit_child["chi2_red"] == pytest.approx(fit_thread["chi2_red"], rel=1e-6)
    assert {k: v for k, v in fit_child.items() if k not in ("chi2_red", "params")} == \
        {k: v for k, v in fit_thread.items() if k not in ("chi2_red", "params")}


def _mismatched_file(path):
    """Window maps with one pixel more than the spectrum cube: the job refuses
    the file -- and it does so inside the child."""
    write_eds_file(path, seed=12)
    lab = "1/EDS/Data/Window Integral/Fe Kα1"
    with h5py.File(path, "a") as f:
        del f[lab]
        f.create_dataset(lab, data=np.ones(N_PX + 1, dtype=np.float32))
    return str(path)


def test_a_failure_in_the_child_is_reported(tmp_path, client):
    path = _mismatched_file(tmp_path / "mismatch.h5oina")
    h5_session.open_file(path)
    proc = _child(path)
    with pytest.raises(svc.SpectrumFitFailed) as exc:
        svc.get_result(path, wait_s=WAIT)
    assert exc.value.error.startswith("ValueError: the spectrum cube has 24 pixels")
    assert _gone(proc) and proc.returncode != 0
    st = svc.status(path)
    assert (st["state"], st["stage"]) == ("failed", "failed")
    r = client.post("/api/eds/quantify/pixel", json={"row": 1, "col": 1})
    assert r.status_code == 500, r.text
    assert r.json()["detail"]["code"] == "eds_spectrum_fit_failed"
    assert "the spectrum cube has 24 pixels" in r.json()["detail"]["error"]


def test_a_child_that_dies_is_reported_and_retried_on_open(synth_files):
    path = str(synth_files["A"])
    h5_session.open_file(path)
    _hold(path).kill()                          # the OS takes it, no message
    with pytest.raises(svc.SpectrumFitFailed) as exc:
        svc.get_result(path, wait_s=WAIT)
    assert "exit code" in exc.value.error
    h5_session.open_file(path)                  # opening it again is the retry
    assert svc.get_result(path, wait_s=WAIT) is not None
    assert svc.status(path)["state"] == "ready"


def test_a_child_that_cannot_start_falls_back_to_this_process(
        synth_files, monkeypatch, caplog, client):
    """No child at all (the interpreter cannot be started) is a problem of
    the machine, not of the file: the numbers are still made, here, as before
    the child process existed -- and the log AND the status say so (the backend stalls while it
    reads the cube; the page has to be able to tell why)."""
    def no_child(_job):
        raise OSError("synthetic: cannot start a process")

    monkeypatch.setattr(svc, "_spawn", no_child)
    path = str(synth_files["A"])
    h5_session.open_file(path)
    res = svc.get_result(path, wait_s=WAIT)
    assert res is not None and svc.status(path)["state"] == "ready"
    assert _job(path).process is None
    _ff, maps, _y = direct_net_maps(path)
    assert _max_rel_diff(res.maps["Fe K"], maps["Fe K"]) <= 1e-6
    assert "could not start its process" in caplog.text
    assert "cannot start a process" in caplog.text
    st = client.get("/api/eds/spectrum-fit/status").json()
    assert st["runner"] == "in_process_fallback"
    assert "OSError: synthetic: cannot start a process" in st["runner_reason"]


def test_the_status_says_where_the_fit_runs(synth_files, monkeypatch, client):
    path = str(synth_files["A"])
    assert svc.status(path)["runner"] is None             # no job yet
    h5_session.open_file(path)
    assert svc.get_result(path, wait_s=WAIT) is not None
    st = client.get("/api/eds/spectrum-fit/status").json()
    assert (st["runner"], st["runner_reason"]) == ("child", None)

    h5_session.close_file()
    svc.reset()
    monkeypatch.setattr(svc, "RUN_IN_CHILD", False)
    h5_session.open_file(path)
    assert svc.get_result(path, wait_s=WAIT) is not None
    assert svc.status(path)["runner"] == "in_process"

    nocube = str(synth_files["nocube"])
    h5_session.open_file(nocube)
    assert (svc.status(nocube)["state"], svc.status(nocube)["runner"]) == \
        ("none", None)


_DIES_BEFORE_ANSWERING = (
    "import sys; "
    "sys.stderr.write('starting up\\n'); "
    "sys.stderr.write('ModuleNotFoundError: No module named synthetic_dep\\n'); "
    "sys.exit(3)")


def test_a_child_that_dies_before_answering_says_why(synth_files, monkeypatch):
    """A child that never writes a protocol line (an import that fails under
    this interpreter) leaves its reason only on stderr. Its last line is in
    the error the page shows, not only in the log."""
    def dying(_job):
        return subprocess.Popen(
            [sys.executable, "-c", _DIES_BEFORE_ANSWERING],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    monkeypatch.setattr(svc, "_spawn", dying)
    path = str(synth_files["A"])
    h5_session.open_file(path)
    with pytest.raises(svc.SpectrumFitFailed) as exc:
        svc.get_result(path, wait_s=WAIT)
    assert "ended without an answer (exit code 3)" in exc.value.error
    assert exc.value.error.endswith(
        "ModuleNotFoundError: No module named synthetic_dep")
    assert svc.status(path)["runner"] == "child"


def test_eviction_terminates_the_child(synth_files):
    path_a = str(synth_files["A"])
    svc.start_for_file(path_a)                  # no session: A can be evicted
    held = _hold(path_a)
    proc_a = _job(path_a).process
    for name in ("B", "C"):
        svc.start_for_file(str(synth_files[name]))
    assert svc._job_for(path_a, start=False) is None
    assert _gone(proc_a, timeout=10)
    assert not held.is_running()
    assert svc.get_result(str(synth_files["C"]), wait_s=WAIT) is not None


def test_file_switch_never_leaks_maps_from_the_child(synth_files):
    path_a, path_b = str(synth_files["A"]), str(synth_files["B"])
    _ff_a, maps_a, _ = direct_net_maps(path_a)
    _ff_b, maps_b, _ = direct_net_maps(path_b)
    assert not np.allclose(maps_a["Fe K"], maps_b["Fe K"])  # the test can bite

    h5_session.open_file(path_a)
    ext_a = h5_session.get_active_extractor()
    held = _hold(path_a)
    try:
        h5_session.open_file(path_b)                       # A still running
        assert svc.get_result(path_b, wait_s=WAIT) is not None
        ext_b = h5_session.get_active_extractor()
        assert np.allclose(ext_b.get_element_map("Fe Kα1"), maps_b["Fe K"],
                           rtol=1e-5, atol=1e-3)
        with pytest.raises(svc.SpectrumFitPending):         # A's view: A's job
            ext_a.get_element_map("Fe Kα1")
        # a switch is not a close: A's child keeps running past the grace
        time.sleep(svc.CLOSE_GRACE_S + 0.5)
        assert held.is_running() and _job(path_a).state == "running"
    finally:
        held.resume()
    assert svc.get_result(path_a, wait_s=WAIT) is not None   # A after B
    for ext in (ext_b, h5_session.get_active_extractor()):
        assert np.allclose(ext.get_element_map("Fe Kα1"), maps_b["Fe K"],
                           rtol=1e-5, atol=1e-3)
    assert np.allclose(ext_a.get_element_map("Fe Kα1"), maps_a["Fe K"],
                       rtol=1e-5, atol=1e-3)


def test_closing_the_file_stops_its_child(synth_files):
    path = str(synth_files["A"])
    h5_session.open_file(path)
    held = _hold(path)
    proc = _job(path).process
    h5_session.close_file()
    assert _gone(proc, timeout=svc.CLOSE_GRACE_S + 10)
    assert not held.is_running()
    st = svc.status(path)
    assert (st["state"], st["error"]) == ("failed", "cancelled")
    h5_session.open_file(path)                  # opening it again fits it again
    assert svc.get_result(path, wait_s=WAIT) is not None


def test_closing_one_file_stops_only_its_own_fit(synth_files):
    """The close rule is per file: A opened, B opened (a switch -- A runs
    on), B closed with nothing reopened. B's fit stops after the grace; A's
    goes on to ready. Before, every job the session had ever opened was
    bound to the session for good, so closing B stopped A as well."""
    path_a, path_b = str(synth_files["A"]), str(synth_files["B"])
    h5_session.open_file(path_a)
    held_a = _hold(path_a)
    try:
        h5_session.open_file(path_b)
        held_b = _hold(path_b)
        proc_b = _job(path_b).process
        h5_session.close_file()
        assert _gone(proc_b, timeout=svc.CLOSE_GRACE_S + 10)
        assert not held_b.is_running()
        st_b = svc.status(path_b)
        assert (st_b["state"], st_b["error"]) == ("failed", "cancelled")
        # A's watcher looks every _POLL_S; give it several looks past B's end
        time.sleep(svc.CLOSE_GRACE_S)
        assert held_a.is_running() and _job(path_a).state == "running"
    finally:
        held_a.resume()
    assert svc.get_result(path_a, wait_s=WAIT) is not None
    assert svc.status(path_a)["state"] == "ready"


def test_reopening_a_file_binds_it_again(synth_files):
    """A, then B, then A again (its job kept): A is the session's file once
    more, so closing it now stops A -- and B, switched away from, runs on."""
    path_a, path_b = str(synth_files["A"]), str(synth_files["B"])
    h5_session.open_file(path_a)
    held_a = _hold(path_a)
    h5_session.open_file(path_b)
    held_b = _hold(path_b)
    try:
        h5_session.open_file(path_a)               # A's job is kept, re-bound
        assert _job(path_a).session_bound and not _job(path_b).session_bound
        proc_a = _job(path_a).process
        h5_session.close_file()
        assert _gone(proc_a, timeout=svc.CLOSE_GRACE_S + 10)
        assert not held_a.is_running()
        time.sleep(svc.CLOSE_GRACE_S)
        assert held_b.is_running() and _job(path_b).state == "running"
    finally:
        held_b.resume()
    assert svc.get_result(path_b, wait_s=WAIT) is not None


def test_shutdown_stops_the_children(synth_files):
    path = str(synth_files["A"])
    h5_session.open_file(path)
    held = _hold(path)
    proc = _job(path).process
    svc._stop_at_exit()                         # what atexit runs
    assert _gone(proc, timeout=10)
    assert not held.is_running()


_BACKEND_THAT_DIES = """
import sys, time
from backend.api.services import eds_spectrum_service as svc
svc.start_for_file(sys.argv[1])
job = svc._job_for(sys.argv[1], start=False)
while job.process is None:
    time.sleep(0.002)
print(job.process.pid, flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process handles")
def test_the_child_exits_when_its_backend_dies(synth_files):
    """The backend leaves by ``os._exit`` in its watchdogs and can be killed;
    no atexit runs then. The child must go with it, by itself."""
    import _winapi
    root = Path(__file__).resolve().parents[1]
    backend = subprocess.Popen(
        [sys.executable, "-c", _BACKEND_THAT_DIES, str(synth_files["A"])],
        cwd=str(root), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    handle = None
    try:
        pid = int(backend.stdout.readline())
        # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION: keeps the exit code
        handle = _winapi.OpenProcess(0x00100000 | 0x1000, False, pid)
        backend.kill()
        backend.wait(30)
        assert _winapi.WaitForSingleObject(handle, 30_000) == 0, \
            "the child outlived its backend"
        assert _winapi.GetExitCodeProcess(handle) == spectrum_job.EXIT_PARENT_GONE
    finally:
        if backend.poll() is None:
            backend.kill()
        if handle is not None:
            _winapi.CloseHandle(handle)


# ---------------------------------------------------------------------------
# The element list does not wait for the fit
# ---------------------------------------------------------------------------
def test_elements_answer_during_the_fit_with_the_finished_labels(
        blocked_fit, sampleb, client):
    h5_session.open_file(str(sampleb))
    assert svc.status(str(sampleb))["state"] == "running"
    r = client.get("/api/eds/elements")
    assert r.status_code == 200, r.text
    pending = r.json()
    assert pending["elements"] and pending["detected_only"] == []
    assert pending["eds_status"]["state"] == "pending"
    assert pending["eds_status"]["code"] == "eds_spectrum_fit_running"
    wrapper = h5_session.get_active_extractor().get_available_elements()
    h5 = client.get("/api/h5/eds/elements", params={"scope": "dataset"})
    assert h5.status_code == 200, h5.text
    # ... and the numbers still wait
    fe = quote("Fe Kα1")
    assert client.get(f"/api/eds/map/{fe}?mode=at_pct").status_code == 503

    blocked_fit.set()
    assert svc.get_result(str(sampleb), wait_s=WAIT) is not None
    r = client.get("/api/eds/elements")
    assert r.status_code == 200, r.text
    ready = r.json()
    assert ready["elements"] == pending["elements"] == wrapper
    assert h5.json()["elements"] == ready["elements"]
    assert "eds_status" not in ready
    assert "Ti Kα1" in ready["detected_only"]


def test_elements_answer_after_a_failed_fit(monkeypatch, synth_files, client):
    path_ok, path_bad = str(synth_files["C"]), str(synth_files["A"])
    h5_session.open_file(path_ok)                 # same header as A
    assert svc.get_result(path_ok, wait_s=WAIT) is not None
    labels = client.get("/api/eds/elements").json()["elements"]

    def boom(*_a, **_k):
        raise ValueError("synthetic fit failure")

    monkeypatch.setattr(svc, "RUN_IN_CHILD", False)
    monkeypatch.setattr(svc, "fit_sum_spectrum", boom)
    h5_session.open_file(path_bad)
    with pytest.raises(svc.SpectrumFitFailed):
        svc.get_result(path_bad, wait_s=WAIT)
    r = client.get("/api/eds/elements")
    assert r.status_code == 200, r.text
    assert r.json()["elements"] == labels and r.json()["detected_only"] == []
    assert r.json()["eds_status"]["state"] == "failed"
    assert "synthetic fit failure" in r.json()["eds_status"]["error"]
