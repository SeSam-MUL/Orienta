"""The phase test says WHY a pixel has no chemistry while the spectrum fit runs.

``pixel_at_pct: null`` has always meant "this scan has no EDS" to the
single-pixel phase test dialog, which then shows no chemistry line at all. On
a file with a spectrum cube the composition is null for a different reason for
the first seconds after a load -- the spectra are still being evaluated -- or
for good when the fit failed. The typed reason used to reach only the log.
These tests pin ``pixel_chemistry_reason`` on every response that carries
``pixel_at_pct``: the live pixel-chemistry lookup, the synchronous test, the
job start, the job result and the re-mask.
"""
import asyncio
import time
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.api.routes.indexing as idx
import backend.api.services.eds_pixel_chemistry as chem
from backend.api.main import app
from backend.api.services.eds_pixel_chemistry import ChemistryUnavailable

client = TestClient(app)

_CHEM_URL = "/api/indexing/single-pixel-phase-test/pixel-chemistry"


def _refusal(code):
    return lambda ext=None: ChemistryUnavailable(code, f"{code} detail",
                                                 expected=False)


class _Entry:
    def __init__(self, key, formula, sht):
        self.key = key; self.formula = formula; self.display_formula = formula
        self.elements = tuple(); self.space_group = "Fm-3m"
        self.crystal_system = "cubic"; self.sht_path = sht


@pytest.fixture
def _wired(monkeypatch, tmp_path):
    """The phase test without a GPU: same stubs as the route tests."""
    sht = tmp_path / "Al.sht"; sht.write_bytes(b"x")
    sig = types.SimpleNamespace()
    sig.data = np.random.rand(3, 4, 10, 10).astype(np.float32)
    monkeypatch.setattr(idx, "_get_active_signal_for_phase_test",
                        lambda: sig, raising=False)
    monkeypatch.setattr(
        idx, "_get_detector_for_phase_test",
        lambda *a, **k: (types.SimpleNamespace(pc_source="header"),
                         {"pat_height": 10, "pat_width": 10, "n_cols": 4,
                          "n_rows": 3, "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.6,
                          "pixel_size": 50.0, "binning": 1, "vendor": "Bruker",
                          "sample_tilt": 70.0, "tilt": 10.0}),
        raising=False)
    monkeypatch.setattr(idx, "get_pixel_at_pct_for_phase_test",
                        lambda i: None, raising=False)
    monkeypatch.setattr(idx, "_phase_test_library_entries",
                        lambda keys: ([_Entry("Al", "Al", sht)], []),
                        raising=False)
    monkeypatch.setattr(
        idx, "_compute_phase_compare_results",
        lambda **kw: [{"phase_id": pid, "phase_name": n, "r_score": 0.5,
                       "ncc_score": 0.6, "euler_deg": [1, 2, 3],
                       "simulated_b64": "S", "ncc_diff_b64": "D",
                       "rank": pid + 1}
                      for pid, n in kw["phase_names"].items()])
    return sht


@pytest.fixture(autouse=True)
def _chemistry_on_the_file_source(request, monkeypatch):
    """The reason plumbing exists for pixel chemistry that READS the spectrum
    fit. It now reads the window numbers
    (``eds_indexing_prior.PRIOR_EDS_SOURCE``) and never waits; these tests
    pin the plumbing for the other setting. Tests named ``test_window_*``
    run with the product's own setting."""
    if request.node.name.startswith("test_window_"):
        return
    import backend.api.services.eds_indexing_prior as prior
    monkeypatch.setattr(prior, "PRIOR_EDS_SOURCE", None)


# --- the live lookup ---------------------------------------------------------
def test_window_chemistry_names_no_spectrum_fit_reason(monkeypatch):
    """On the window numbers the chemistry does not wait for the fit, so a
    null during a running fit has another cause and the dialog must not say
    "spectra still being evaluated" -- the refusal is not even asked."""
    assert chem.chemistry_source() == "window"
    monkeypatch.setattr(idx, "get_pixel_at_pct_for_phase_test",
                        lambda i: None, raising=False)
    asked = []
    monkeypatch.setattr(chem, "spectrum_fit_refusal",
                        lambda ext=None: asked.append(1) or ChemistryUnavailable(
                            "spectrum_fit_running", "x", expected=False))
    body = client.get(f"{_CHEM_URL}/0").json()
    assert body["pixel_chemistry_reason"] is None
    assert asked == []


def test_running_fit_is_named_not_mistaken_for_no_eds(monkeypatch):
    monkeypatch.setattr(idx, "get_pixel_at_pct_for_phase_test",
                        lambda i: None, raising=False)
    monkeypatch.setattr(chem, "spectrum_fit_refusal",
                        _refusal("spectrum_fit_running"))
    body = client.get(f"{_CHEM_URL}/0").json()
    assert body["pixel_at_pct"] is None
    assert body["pixel_chemistry_reason"] == "spectrum_fit_running"


def test_failed_fit_is_named(monkeypatch):
    monkeypatch.setattr(idx, "get_pixel_at_pct_for_phase_test",
                        lambda i: None, raising=False)
    monkeypatch.setattr(chem, "spectrum_fit_refusal",
                        _refusal("spectrum_fit_failed"))
    body = client.get(f"{_CHEM_URL}/0").json()
    assert body["pixel_chemistry_reason"] == "spectrum_fit_failed"


def test_no_reason_when_there_simply_is_no_eds(monkeypatch):
    # The real refusal check: no file open, so not a spectrum-fit question.
    monkeypatch.setattr(idx, "get_pixel_at_pct_for_phase_test",
                        lambda i: None, raising=False)
    body = client.get(f"{_CHEM_URL}/0").json()
    assert body["pixel_at_pct"] is None
    assert body["pixel_chemistry_reason"] is None


def test_no_reason_when_the_chemistry_is_there(monkeypatch):
    # A composition in hand is never second-guessed by a later fit question.
    monkeypatch.setattr(idx, "get_pixel_at_pct_for_phase_test",
                        lambda i: {"Al": 99.0}, raising=False)
    asked = []
    monkeypatch.setattr(chem, "spectrum_fit_refusal",
                        lambda ext=None: asked.append(1))
    body = client.get(f"{_CHEM_URL}/0").json()
    assert body["pixel_at_pct"] == {"Al": 99.0}
    assert body["pixel_chemistry_reason"] is None
    assert asked == []


# --- the runs ------------------------------------------------------------------
def test_sync_test_carries_the_reason(_wired, monkeypatch):
    monkeypatch.setattr(chem, "spectrum_fit_refusal",
                        _refusal("spectrum_fit_running"))
    r = client.post("/api/indexing/single-pixel-phase-test",
                    json={"pixel_index": 0, "eds_weighting": "soft"})
    assert r.status_code == 200
    body = r.json()
    assert body["pixel_at_pct"] is None
    assert body["pixel_chemistry_reason"] == "spectrum_fit_running"


def test_job_start_and_result_carry_the_reason(_wired, monkeypatch):
    monkeypatch.setattr(chem, "spectrum_fit_refusal",
                        _refusal("spectrum_fit_running"))
    r = client.post("/api/indexing/single-pixel-phase-test/start",
                    json={"pixel_index": 0, "eds_weighting": "soft"})
    assert r.status_code == 200
    start = r.json()
    assert start["pixel_chemistry_reason"] == "spectrum_fit_running"

    # Wait on the clock, not on a number of requests: the job runs in a thread,
    # and on a busy machine 50 back-to-back polls can all come before it ends.
    body = None
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        body = client.get("/api/indexing/single-pixel-phase-test/progress/"
                          f"{start['job_id']}").json()
        if body["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    assert body["status"] == "done", body
    assert body["result"]["pixel_chemistry_reason"] == "spectrum_fit_running"


def test_job_coroutine_records_the_reason_it_is_given(_wired, tmp_path):
    sht = tmp_path / "Al.sht"; sht.write_bytes(b"x")
    jid = idx._new_phase_test_job()
    asyncio.run(idx._run_phase_test_job(
        jid, exp=np.random.rand(10, 10).astype(np.float32),
        det_params={"pat_height": 10, "pat_width": 10},
        sht_paths_by_phase={0: str(sht)}, phase_names={0: "Al"},
        mask=None, max_bandwidth=128,
        id_to_entry={0: _Entry("Al", "Al", sht)}, chem_by_key={},
        eff_mode="off", excluded=[], row=0, col=0, pixel_index=0,
        exp_display="EXP", pixel_at_pct=None,
        pixel_chemistry_reason="spectrum_fit_failed"))
    job = idx._phase_test_jobs[jid]
    assert job["status"] == "done"
    assert job["result"]["pixel_chemistry_reason"] == "spectrum_fit_failed"


def test_remask_keeps_the_reason_of_its_run():
    cache = {
        "measured": np.random.rand(10, 10).astype(np.float32),
        "sims": {}, "rows": [], "id_to_entry": {}, "chem_by_key": {},
        "eff_mode": "off", "excluded": [], "row": 0, "col": 0,
        "pixel_index": 0, "pixel_at_pct": None,
        "pixel_chemistry_reason": "spectrum_fit_running", "bandwidth": 128,
    }
    out = idx._recompute_remask(cache, "full", 1.0)
    assert out["pixel_chemistry_reason"] == "spectrum_fit_running"
