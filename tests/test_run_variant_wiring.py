"""How a run hands the variant on, and what its failure carries.

* The Spherical run hands the variant of the dataset it runs on to the indexer (this was pinned only
  by a test that reads the route's source text): the real ``POST /api/indexing/start`` is driven with
  the indexer stubbed to record its arguments.
* A refusal raised inside the run (a re-reader that cannot follow the variant) ends the task with its
  code, like a batch item does, not as a generic failure with a traceback.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.api.services import variant_registry
from backend.api.services.load_choices import LoadChoiceError
from tests.test_variant_rereads import RAW, RAW2, two_sources  # noqa: F401  (two_sources: fixture)


@pytest.fixture
def drive(monkeypatch, two_sources):
    """``drive(variant, stub, body, entry)``: POST /api/indexing/start (by default Spherical, GPU backend,
    one phase) for the active dataset 'ds', which is a load of ``two_sources`` as ``variant`` (None: no
    choice). The run is executed in place and ``indexing_controller.<entry>`` (by default
    ``spherical_gpu_index_patterns``) is replaced by ``stub``. Returns the task state."""
    import kikuchipy as kp
    from fastapi.testclient import TestClient
    import indexing_controller
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from backend.api.services import scan_geometry
    from backend.api.services.calibration_store import CalibrationStore
    from kikuchipy.detectors import EBSDDetector
    from tests.test_result_keeps_its_run_pc import RUN_PC, SHAPE

    sig = kp.data.nickel_ebsd_small()
    sig.detector = EBSDDetector(shape=SHAPE, pc=RUN_PC, sample_tilt=70.0, tilt=3.5)
    store = CalibrationStore()
    store.register("ds", sig)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: sig)
    monkeypatch.setattr(ev, "is_active_signal_dirty", lambda: False)
    monkeypatch.setattr(ev, "get_active_include_mask", lambda: None)
    monkeypatch.setattr(ev, "_ebsd_file_path", two_sources, raising=False)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="indexing": fn())

    def run(variant, stub, body=None, entry="spherical_gpu_index_patterns"):
        monkeypatch.setattr(indexing_controller, entry, stub)
        if variant is None:
            variant_registry.drop("ds")
        else:
            variant_registry.register("ds", variant)
        scan_geometry.clear_all()
        saved, saved_active = dict(I._result_registry), I._active_result_id
        try:
            resp = TestClient(app).post("/api/indexing/start", json=body or {
                "method": "spherical", "backend": "spherical_gpu", "sht_paths": ["a.sht"]})
            assert resp.status_code == 200, resp.text
            return I._indexing_tasks[resp.json()["task_id"]]
        finally:
            variant_registry.drop("ds")
            scan_geometry.clear_all()
            I._result_registry.clear()
            I._result_registry.update(saved)
            I._active_result_id = saved_active
    return run


class _Recorder:
    def __init__(self, raises):
        self.calls, self.raises = [], raises

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        raise self.raises


# ============================================================== the run hands the variant on

@pytest.mark.parametrize("variant", [RAW, RAW2, None], ids=["measurement-1-unprocessed",
                                                          "measurement-2-unprocessed", "no-choice"])
def test_the_spherical_run_hands_the_datasets_variant_to_the_indexer(drive, two_sources, variant):
    rec = _Recorder(RuntimeError("stop after the call"))
    drive(variant, rec)
    assert len(rec.calls) == 1
    assert rec.calls[0]["variant"] == variant
    assert rec.calls[0]["h5_path"] == two_sources


SPHERICAL = {"method": "spherical", "backend": "spherical_gpu"}


@pytest.mark.parametrize("variant", [RAW2, None], ids=["measurement-2-unprocessed", "no-choice"])
def test_the_multi_phase_gpu_run_hands_the_datasets_variant_to_the_indexer(drive, two_sources, variant):
    rec = _Recorder(RuntimeError("stop after the call"))
    drive(variant, rec, {**SPHERICAL, "sht_paths": ["a.sht", "b.sht"]})
    assert len(rec.calls) == 1 and rec.calls[0]["variant"] == variant


@pytest.mark.parametrize("variant", [RAW2, None], ids=["measurement-2-unprocessed", "no-choice"])
def test_the_emsphinx_run_hands_the_datasets_variant_to_the_indexer(drive, two_sources, variant):
    rec = _Recorder(RuntimeError("stop after the call"))
    drive(variant, rec, {"method": "spherical", "backend": "emsphinx", "sht_paths": ["a.sht"]},
          entry="spherical_index_patterns")
    assert len(rec.calls) == 1 and rec.calls[0]["variant"] == variant


# ====================================================== what a refusal inside the run ends as

def test_a_refusal_inside_the_run_ends_the_task_with_its_code(drive):
    refusal = LoadChoiceError("variantNotSupported", "This step cannot read the unprocessed patterns",
                              {"where": "somewhere"})
    task = drive(RAW, _Recorder(refusal))
    assert task["status"] == "failed"
    assert task["error_code"] == "variantNotSupported"
    assert task["error_params"] == {"where": "somewhere"}
    assert task["error"] == "This step cannot read the unprocessed patterns"      # prose only: no traceback
    assert "Traceback" not in task["error"]


def test_any_other_failure_of_the_run_stays_the_generic_one_without_a_code(drive):
    task = drive(RAW, _Recorder(RuntimeError("something else")))
    assert task["status"] == "failed"
    assert "error_code" not in task
    assert "something else" in task["error"] and "Traceback" in task["error"]


def test_the_status_route_hands_the_code_on(drive):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.api.routes import indexing as I
    refusal = LoadChoiceError("loadVariantUnknown", "no such source")
    task = drive(RAW, _Recorder(refusal))
    I._indexing_tasks["task-with-a-code"] = task
    try:
        got = TestClient(app).get("/api/indexing/status/task-with-a-code").json()
    finally:
        I._indexing_tasks.pop("task-with-a-code", None)
    assert got["status"] == "failed" and got["error_code"] == "loadVariantUnknown"


# ===================================== the phase-map routing and the CPU multi-phase loop hand it on too

@pytest.mark.parametrize("variant", [RAW2, None], ids=["measurement-2-unprocessed", "no-choice"])
def test_the_multi_phase_cpu_loop_hands_the_datasets_variant_to_the_indexer(drive, two_sources, variant):
    rec = _Recorder(RuntimeError("stop after the call"))
    drive(variant, rec, {"method": "spherical", "backend": "emsphinx", "sht_paths": ["a.sht", "b.sht"]},
          entry="run_single_phase_method")
    assert rec.calls and all(c["variant"] == variant for c in rec.calls)


@pytest.mark.parametrize("variant", [RAW2, None], ids=["measurement-2-unprocessed", "no-choice"])
def test_the_phase_map_routing_hands_the_datasets_variant_to_the_indexer(
        drive, monkeypatch, tmp_path, two_sources, variant):
    from pathlib import Path
    from types import SimpleNamespace
    import phase_metadata
    from backend.api.services import phase_map_store
    from tests.test_hough_cif_origin import _si_choice_2
    cif = _si_choice_2(tmp_path)
    state = SimpleNamespace(phase_entries=[SimpleNamespace(cif_filename=cif.name)],
                            phase_grid=np.zeros((2, 3), dtype=int))
    monkeypatch.setattr(phase_map_store, "get_phase_map_store",
                        lambda: SimpleNamespace(get_state=lambda: state))
    monkeypatch.setattr(phase_metadata, "strict_linked_cif", lambda _file, lib: Path(lib) / cif.name)
    rec = _Recorder(RuntimeError("stop after the call"))
    drive(variant, rec, {"method": "spherical", "backend": "spherical_gpu",
                         "sht_paths": [str(tmp_path / "si.sht")], "use_phase_map_routing": True},
          entry="run_per_phase_indexing")
    assert len(rec.calls) == 1 and rec.calls[0]["variant"] == variant
