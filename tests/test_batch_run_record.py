"""A batch-v2 export carries the same record of the run as every other export path.

Every result file written from the indexing routes holds ``hough_reflectors`` (which
reflector list each Hough phase was indexed with) and the citation steps of the run.
The batch-v2 exports went through ``result_exporter.export_all``, which passed neither
to the two .h5 writers, so a batch result said nothing about its reflector lists.

Real batch (create, start, jobs, checkpoint, export), real Hough indexing of a real Ni
CIF on a synthetic 4x4 UP1 with an .osc; the files are read back with h5py.
"""
import json
import time

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.calibration_store import CalibrationStore
from tests.test_batch_same_stem import NI_CIF


@pytest.fixture
def world(monkeypatch, tmp_path):
    import kikuchipy as kp
    import safe_loader
    from backend.api.routes import batch_v2 as B
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services import batch_manager as M
    import ebsd_utils as eu
    import hough_reflectors as hr
    hr.clear_specs()
    eu.clear_phase_reflector_limits()
    store = CalibrationStore()
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(B, "calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr(M, "_batch_names", {})
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    # the loader seam: the file on disk is only the source the exports refer to
    signal = kp.data.nickel_ebsd_small()
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path, *a, **k: signal.deepcopy())
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mgr = M.BatchManager(db_path=str(tmp_path / "jobs.db"))
    monkeypatch.setattr(B, "_get_manager", lambda: mgr)
    cif = tmp_path / "Ni.cif"
    cif.write_text(NI_CIF, encoding="utf-8")
    scan = tmp_path / "Scan7.h5oina"
    with h5py.File(scan, "w") as f:                 # the shape the exporters need of a source
        f.create_group("1/EBSD/Data")
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("Beam Voltage", data=np.float32(20.0))
        hdr.create_dataset("Tilt Angle", data=np.float32(70.0))
    yield mgr, str(scan), str(cif), tmp_path
    hr.clear_specs()
    eu.clear_phase_reflector_limits()


def _run(mgr, scan, cif, tmp_path, formats):
    client = TestClient(app)
    out = tmp_path / "out"
    resp = client.post("/api/batch-v2/create", json={
        "files": [scan], "phases": [{"name": "Ni", "path": cif, "method": "hough"}],
        "config": {"auto_export": True, "export_dir": str(out), "export_formats": formats,
                   "include_eds": False}})
    assert resp.status_code == 200, resp.text
    batch_id = resp.json()["batch_id"]
    resp = client.post(f"/api/batch-v2/{batch_id}/start")
    assert resp.status_code == 200, resp.text
    for _ in range(600):
        time.sleep(0.2)
        if mgr.get_status(batch_id)["status"] in ("completed", "paused"):
            break
    return batch_id, out


def _light(out):
    (light,) = out.glob("result_Scan7*_light.h5")
    return light


def test_the_batch_light_export_records_the_reflector_list(world):
    mgr, scan, cif, tmp = world
    batch_id, out = _run(mgr, scan, cif, tmp, ["h5_light"])
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "done", job
    with h5py.File(_light(out), "r") as f:
        rec = json.loads(f["Indexing/Citations"].attrs["hough_reflectors"])
    assert list(rec) == ["Ni"]
    entry = rec["Ni"]
    # the default list of a cubic phase: its rows and where they come from
    assert entry["mode"] == "default" and entry["source"]
    assert entry["n_rows"] > 0 and entry["families_used"]


def _direct_hough_run(cif):
    """The same indexing as the batch job, called directly: the result whose record an
    export of any other path would write."""
    import kikuchipy as kp
    from orix.crystal_map import PhaseList

    import ebsd_utils as eu
    from indexing_controller import IndexingConfig, IndexingMethod, hough_index_patterns

    signal = kp.data.nickel_ebsd_small()
    phase = eu.hough_phase_from_cif(cif)
    phase.name = "Ni"
    return hough_index_patterns(
        signal=signal, phase_list=PhaseList([phase]), detector=signal.detector,
        config=IndexingConfig(method=IndexingMethod.HOUGH, n_bands=12, t_sigma=2.0, r_sigma=2.0))


def test_the_batch_export_carries_the_record_and_the_steps_of_the_direct_run(world):
    from backend.api.services.citations.provenance import get_steps

    mgr, scan, cif, tmp = world
    batch_id, out = _run(mgr, scan, cif, tmp, ["h5_light"])
    assert mgr.get_jobs(batch_id)[0]["status"] == "done"
    direct = _direct_hough_run(cif)
    with h5py.File(_light(out), "r") as f:
        cit = f["Indexing/Citations"]
        rec = json.loads(cit.attrs["hough_reflectors"])
        steps = json.loads(cit.attrs["steps"])
    # JSON round trip of the direct run's record: the same fields, the same values (the
    # memory budget is the free memory of the moment of each run, so it is not compared)
    want = json.loads(json.dumps(direct.metadata["hough_reflectors"]))
    for entry in (rec["Ni"], want["Ni"]):
        entry.pop("budget_bytes")
    assert rec == want
    assert steps == json.loads(json.dumps(get_steps(direct)))
    assert "indexing.hough" in [s["key"] for s in steps]


def _two_phase_checkpoint(tmp_path):
    from backend.api.services.checkpoint_writer import CheckpointWriter

    source = tmp_path / "Two.h5oina"
    with h5py.File(source, "w") as f:
        f.create_group("1/EBSD/Data")
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("Beam Voltage", data=np.float32(20.0))
        hdr.create_dataset("Tilt Angle", data=np.float32(70.0))
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="rec", method="hough", step_size_um=0.5)
    ci = np.full((3, 4), 0.6, dtype=np.float32)
    ori = np.zeros((3, 4, 3), dtype=np.float32)
    common = {"ci_mean": 0.6, "duration_sec": 1.0, "space_group": 225, "point_group": "m-3m",
              "lattice_constants": [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]}
    shared = {"key": "indexing.hough", "params": {"orienta_version": "x"}}
    cw.write_phase_result("Ni", ci, ori, dict(
        common, phase_file="Ni.cif", citation_steps=[shared],
        hough_reflectors={"Ni": {"mode": "default", "n_rows": 58}}))
    cw.write_phase_result("Mg", ci, ori, dict(
        common, phase_file="Mg.cif",
        citation_steps=[shared, {"key": "indexing.hough_reflectors",
                                 "params": {"phase": "Mg", "n_rows": 50}}],
        hough_reflectors={"Mg": {"mode": "custom", "n_rows": 50}}))
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


def test_the_checkpoint_merges_the_records_of_its_phases(tmp_path):
    from backend.api.services.checkpoint_writer import CheckpointWriter

    source, cw = _two_phase_checkpoint(tmp_path)
    record = CheckpointWriter(str(source)).read_run_record()      # a fresh reader: a resumed batch
    assert [s["key"] for s in record["citation_steps"]] == [
        "indexing.hough", "indexing.hough_reflectors"]            # the shared step once
    assert record["hough_reflectors"] == {"Ni": {"mode": "default", "n_rows": 58},
                                          "Mg": {"mode": "custom", "n_rows": 50}}


def test_a_checkpoint_from_before_the_record_gives_none(tmp_path):
    from backend.api.services.checkpoint_writer import CheckpointWriter

    source = tmp_path / "Old.h5oina"
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="rec", method="hough", step_size_um=0.5)
    cw.write_phase_result("Ni", np.zeros((3, 4), np.float32), np.zeros((3, 4, 3), np.float32),
                          {"phase_file": "Ni.cif", "ci_mean": 0.1, "duration_sec": 1.0})
    assert cw.read_run_record() == {"citation_steps": None, "hough_reflectors": None}


@pytest.mark.parametrize("fmt,key", [("h5_light", "h5_light"), ("h5_rich", "h5_rich")])
def test_export_all_hands_both_records_to_each_h5_writer(tmp_path, fmt, key):
    from backend.api.services.result_exporter import export_all

    source, cw = _two_phase_checkpoint(tmp_path)
    record = cw.read_run_record()
    out = tmp_path / "out"
    out.mkdir()
    results = export_all(
        source_h5_path=str(source), checkpoint_path=cw.checkpoint_path, output_dir=str(out),
        formats=[fmt], include_eds=False, **record)
    assert results.get(key), results.get("_errors")
    with h5py.File(results[key], "r") as f:
        cit = f["Indexing/Citations"]
        assert json.loads(cit.attrs["hough_reflectors"]) == record["hough_reflectors"]
        assert [s["key"] for s in json.loads(cit.attrs["steps"])] == [
            "indexing.hough", "indexing.hough_reflectors"]


def test_without_a_record_the_file_says_nothing(tmp_path):
    from backend.api.services.result_exporter import export_all

    source, cw = _two_phase_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    results = export_all(source_h5_path=str(source), checkpoint_path=cw.checkpoint_path,
                         output_dir=str(out), formats=["h5_light"], include_eds=False)
    with h5py.File(results["h5_light"], "r") as f:
        assert "Citations" not in f["Indexing"]
