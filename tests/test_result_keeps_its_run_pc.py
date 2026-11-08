"""A result carries the PC it was indexed at, and its exports state that one.

The exports read the calibration store when they were written, so a Hough result indexed
at (0.48, 0.52, 0.60) and exported again after (0.50, 0.50, 0.62) had been typed carried the
new PC in the .ang x/y/z-star and in /Detector/pc, with byte-identical orientations: two
exports of one result with two pattern centres. The run now records PC, its source and the
tilts when it STARTS (``result.metadata["pc_at_run"]``, every method), the exports state
that, and a result without the record (indexed before it existed) says "not recorded"
instead of taking a PC from the store.

The route is the real ``POST /api/indexing/start`` and its real background task; the one
seam is the Hough indexer itself (a 3 x 3 map is returned), the thing under test is what
the route records and what the exports write.
"""
import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from kikuchipy.detectors import EBSDDetector
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from backend.api.main import app
from backend.api.routes import indexing as I
from backend.api.services.calibration_store import CalibrationStore
from indexing_controller import IndexingMethod, IndexingResult

SHAPE = (60, 60)
RUN_PC = (0.48, 0.52, 0.60)          # what the run was indexed at
TYPED_LATER = (0.50, 0.50, 0.62)     # typed after the run


@pytest.fixture(autouse=True)
def restore_registry():
    saved = dict(I._result_registry)
    saved_active = I._active_result_id
    try:
        yield
    finally:
        I._result_registry.clear()
        I._result_registry.update(saved)
        I._active_result_id = saved_active


def _ni_cif(tmp_path):
    cif = tmp_path / "Ni.cif"
    cif.write_text(
        "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding="utf-8")
    return str(cif)


def _fake_hough(signal, phase_list, detector, config, selection_mask, progress_callback=None,
                cancel_check=None, **kw):
    n_rows, n_cols = 3, 3
    n = n_rows * n_cols
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(np.array([[i * 9.0, 20.0, 30.0] for i in range(n)]))),
        phase_id=np.zeros(n, dtype=np.int32),
        x=np.tile(np.arange(n_cols, dtype=np.float32), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=np.float32), n_cols),
        phase_list=PhaseList(phases=[Phase(name="Ni", space_group=225, point_group="m-3m")]))
    return IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool), original_shape=(n_rows, n_cols),
        method=IndexingMethod.HOUGH, confidence_scores=np.full(n, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.5, "recorded_lattices": {0: [3.52] * 3 + [90, 90, 90]}})


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Index the dataset 'ds' (PC = RUN_PC) through the real route; returns (client, store, signal)."""
    import indexing_controller
    import kikuchipy as kp
    from backend.api.routes import ebsd_viewer as ev
    sig = kp.data.nickel_ebsd_small()
    sig.detector = EBSDDetector(shape=SHAPE, pc=RUN_PC, sample_tilt=70.0, tilt=3.5)
    store = CalibrationStore()
    store.register("ds", sig)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: sig)
    up1 = tmp_path / "scan.up1"
    up1.write_bytes(b"not an hdf5 file")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(up1), raising=False)
    monkeypatch.setattr(indexing_controller, "hough_index_patterns", _fake_hough)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="indexing": fn())
    client = TestClient(app)
    resp = client.post("/api/indexing/start", json={"method": "hough", "cif_paths": [_ni_cif(tmp_path)]})
    assert resp.status_code == 200, resp.text
    task = I._indexing_tasks[resp.json()["task_id"]]
    assert task["status"] == "completed", task
    return client, store, sig


def _export_ang(client, path):
    resp = client.post("/api/indexing/export", json={"format": "ang", "filename": str(path)})
    assert resp.status_code == 200, resp.text
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [float([ln for ln in lines if ln.startswith(f"# {k}")][0].split()[-1])
            for k in ("x-star", "y-star", "z-star")]


def test_the_run_records_the_pc_and_tilts_it_was_indexed_at(run):
    client, store, sig = run
    rec = I._get_result().metadata["pc_at_run"]
    np.testing.assert_allclose(rec["pc"], RUN_PC, atol=1e-12)
    assert rec["pc_source"] == "header" and rec["per_pixel"] is False
    assert (rec["sample_tilt"], rec["tilt"]) == (70.0, 3.5)


def test_an_export_after_a_later_pc_change_still_states_the_pc_of_the_run(run, tmp_path):
    client, store, sig = run
    before = _export_ang(client, tmp_path / "before.ang")
    store.update_pc("ds", list(TYPED_LATER), source="manual")          # typed after the run
    after = _export_ang(client, tmp_path / "after.ang")
    # by hand, TSL: x* = PCx, y* = 1 - PCy, z* = PCz * rows / min(rows, cols) (square: 1)
    expected = [RUN_PC[0], 1 - RUN_PC[1], RUN_PC[2]]
    np.testing.assert_allclose(before, expected, atol=1e-6)
    np.testing.assert_allclose(after, expected, atol=1e-6)             # not (0.50, 0.50, 0.62)
    # the rich and the light h5 say the same
    for fmt in ("h5", "h5_light"):
        out = tmp_path / f"{fmt}.h5"
        resp = client.post("/api/indexing/export", json={"format": fmt, "filename": str(out)})
        assert resp.status_code == 200, resp.text
        with h5py.File(out, "r") as f:
            np.testing.assert_allclose(f["Detector/pc"][()], RUN_PC, atol=1e-12)
            assert f["Detector"].attrs["pc_source"] == "header"


def test_a_tilt_edited_after_the_run_is_not_stated_either(run, tmp_path):
    client, store, sig = run
    store.update_geometry("ds", sample_tilt=75.0, tilt=9.0)
    out = tmp_path / "rich.h5"
    assert client.post("/api/indexing/export", json={"format": "h5_light", "filename": str(out)}).status_code == 200
    with h5py.File(out, "r") as f:
        assert f["Detector"].attrs["sample_tilt"] == pytest.approx(70.0)
        assert f["Detector"].attrs["camera_tilt"] == pytest.approx(3.5)


def test_the_ang_header_states_the_tilts_of_the_run_not_a_later_edit(run, tmp_path):
    client, store, sig = run
    store.update_geometry("ds", sample_tilt=75.0, tilt=9.0)
    out = tmp_path / "tilt.ang"
    assert client.post("/api/indexing/export", json={"format": "ang", "filename": str(out)}).status_code == 200
    text = out.read_text(encoding="utf-8", errors="replace")
    assert "sample_tilt_deg=70.000" in text and "detector_tilt_deg=3.500" in text
    assert "sample_tilt_deg=75" not in text


def test_a_per_pixel_map_is_stated_by_its_mean_and_flagged(monkeypatch, run, tmp_path):
    """Indexed on a grid-calibrated map: the export states the map's mean and says so."""
    client, store, sig = run
    grid = np.stack([np.full((3, 3), 0.47), np.full((3, 3), 0.41), np.full((3, 3), 0.77)], axis=-1)
    grid[0, 0] = (0.40, 0.30, 0.70)
    store.update_pc_map("ds", grid)
    resp = client.post("/api/indexing/start", json={"method": "hough", "cif_paths": [_ni_cif(tmp_path)]})
    assert resp.status_code == 200, resp.text
    rec = I._get_result().metadata["pc_at_run"]
    assert rec["per_pixel"] is True and rec["map_shape"] == [3, 3] and rec["pc_source"] == "refined_map"
    out = tmp_path / "map.h5"
    assert client.post("/api/indexing/export", json={"format": "h5_light", "filename": str(out)}).status_code == 200
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["Detector/pc"][()], grid.reshape(-1, 3).mean(axis=0), atol=1e-12)
        assert bool(f["Detector"].attrs["pc_is_map_mean"]) is True


def test_a_result_without_the_record_says_not_recorded_instead_of_using_the_store(run, tmp_path):
    client, store, sig = run
    I._get_result().metadata.pop("pc_at_run")                          # indexed before the record existed
    store.update_pc("ds", list(TYPED_LATER), source="manual")
    out = tmp_path / "old.h5"
    assert client.post("/api/indexing/export", json={"format": "h5_light", "filename": str(out)}).status_code == 200
    with h5py.File(out, "r") as f:
        g = f["Detector"]
        assert "pc" not in g
        assert bool(g.attrs["pc_not_recorded"]) is True and bool(g.attrs["geometry_not_recorded"]) is True
        assert "sample_tilt" not in g.attrs
    ang = tmp_path / "old.ang"
    assert client.post("/api/indexing/export", json={"format": "ang", "filename": str(ang)}).status_code == 200
    text = ang.read_text(encoding="utf-8", errors="replace")
    assert "was not recorded with it" in text
    assert _export_ang(client, tmp_path / "old2.ang") == [0.0, 0.0, 0.0]
    # the machine-readable state and the note stand DIRECTLY above the three zero lines
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# x-star"))
    assert lines[i - 2] == "# ORIENTA_PC: not recorded"
    assert lines[i - 1].startswith("# ORIENTA_NOTE:") and "was not recorded with it" in lines[i - 1]
    assert [float(lines[i + k].split()[-1]) for k in range(3)] == [0.0, 0.0, 0.0]
    assert lines[i + 1].startswith("# y-star") and lines[i + 2].startswith("# z-star")
    from orix.io import load                                       # a parser takes the file as before
    assert load(str(ang)).size == 9


# --- the batches: a PC typed while the batch runs is not the PC of its orientations ----

def _write_up1(path, n_side=4):
    import struct
    rng = np.random.default_rng(3)
    pats = rng.integers(0, 250, (n_side * n_side, 24, 24)).astype("u1")
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, 24, 24, 16))
        f.write(pats.tobytes())
    return str(path)


def _write_osc(path, pc_tsl, n_side=4):
    import struct
    from edax_up1 import _OSC_PC_OFFSET
    raw = bytearray(_OSC_PC_OFFSET + 64)
    struct.pack_into("<3I", raw, 16, n_side - 1, n_side - 1, n_side * n_side)
    struct.pack_into("<3f", raw, _OSC_PC_OFFSET, *pc_tsl)
    with open(path, "wb") as f:
        f.write(bytes(raw))


def test_the_legacy_batch_export_states_the_pc_the_file_was_indexed_at(monkeypatch, tmp_path):
    import indexing_controller
    from backend.api.routes import ebsd_viewer as ev
    store = CalibrationStore()
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    path = _write_up1(tmp_path / "scan.up1")
    _write_osc(tmp_path / "scan.osc", (0.55, 0.50, 0.70))        # the file's own PC (TSL)

    def indexer_that_gets_a_pc_typed(*a, **k):
        store.update_pc("scan", list(TYPED_LATER), source="manual")   # the user types while it runs
        return _fake_hough(*a, **k)

    monkeypatch.setattr(indexing_controller, "hough_index_patterns", indexer_that_gets_a_pc_typed)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    saved = dict(I._batch_state)
    try:
        resp = TestClient(app).post("/api/indexing/batch/start", json={"datasets": [
            {"file_path": path, "method": "hough", "cif_paths": [_ni_cif(tmp_path)]}],
            "auto_export": True, "export_dir": str(tmp_path)})
        assert resp.status_code == 200, resp.text
        entry = I._batch_state["results"][-1]
        assert entry["error"] is None, entry
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved)
    run_pc = I._get_result().metadata["pc_at_run"]["pc"]
    np.testing.assert_allclose(run_pc, store.get_entry("scan").file_pc, atol=1e-12)   # the file's, not the typed one
    with h5py.File(tmp_path / "scan_indexed.h5", "r") as f:
        np.testing.assert_allclose(f["Detector/pc"][()], run_pc, atol=1e-12)
        assert not np.allclose(f["Detector/pc"][()], TYPED_LATER, atol=1e-3)


def test_the_backend_batch_exports_the_pc_it_loaded_the_file_with(monkeypatch, tmp_path):
    """BatchManager takes the record when the file is loaded and hands THAT to export_all."""
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services import batch_manager as M
    from backend.api.services import result_exporter
    store = CalibrationStore()
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    path = _write_up1(tmp_path / "scan_b.up1")
    _write_osc(tmp_path / "scan_b.osc", (0.55, 0.50, 0.70))

    def fake_job(signal, file_path, phase_name, phase_path, method, grid_shape, job_config=None,
                 progress_callback=None, scan_geometry=None):
        store.update_pc("scan_b", list(TYPED_LATER), source="manual")      # typed while the batch runs
        ci = np.full(grid_shape, 0.5)
        return ci, np.zeros(grid_shape + (3,)), {"ci_mean": 0.5, "ci_median": 0.5, "duration_sec": 0.1}

    seen = {}
    monkeypatch.setattr(M, "run_single_indexing_job", fake_job)
    monkeypatch.setattr(result_exporter, "export_all",
                        lambda **kw: seen.update(kw) or {})
    mgr = M.BatchManager(db_path=str(tmp_path / "jobs.db"))
    cif = _ni_cif(tmp_path)
    batch_id = mgr.queue.create_batch([path], [{"name": "Ni", "path": cif, "method": "hough"}],
                                      {"auto_export": True, "export_dir": str(tmp_path)})
    mgr._run_batch_sync_impl(batch_id)
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "done", job
    file_pc = store.get_entry("scan_b").file_pc
    np.testing.assert_allclose(seen["pc"], file_pc, atol=1e-12)           # what it loaded with
    assert not np.allclose(seen["pc"], TYPED_LATER, atol=1e-3)


# --- the dataset a run is for is the one it was requested for ------------------------

def test_a_run_whose_dataset_changed_before_it_started_is_refused_not_run_on_the_other_file(
        monkeypatch, tmp_path):
    """The PC guard ran for 'ds' at the request; the task started after another file had
    become active. It must not index that file's patterns at 'ds''s PC."""
    import indexing_controller
    import kikuchipy as kp
    from backend.api.routes import ebsd_viewer as ev
    sig = kp.data.nickel_ebsd_small()
    sig.detector = EBSDDetector(shape=SHAPE, pc=RUN_PC, sample_tilt=70.0)
    store = CalibrationStore()
    store.register("ds", sig)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: sig)
    ran = []
    monkeypatch.setattr(indexing_controller, "hough_index_patterns",
                        lambda *a, **k: ran.append(1) or _fake_hough(*a, **k))

    def start_after_the_switch(fn, kind="indexing"):
        monkeypatch.setattr(ev, "_active_dataset", "other_file")     # the user loaded another file
        fn()

    monkeypatch.setattr(I, "_submit_run_guarded", start_after_the_switch)
    saved = dict(I._result_registry)
    try:
        resp = TestClient(app).post("/api/indexing/start", json={"method": "hough", "cif_paths": [_ni_cif(tmp_path)]})
        assert resp.status_code == 200, resp.text
        task = I._indexing_tasks[resp.json()["task_id"]]
        assert task["status"] == "failed" and "changed from 'ds' to 'other_file'" in task["error"]
        assert ran == []                                             # nothing was indexed
        assert I._result_registry == saved                           # and no result was stored
    finally:
        I._result_registry.clear()
        I._result_registry.update(saved)


def test_the_run_takes_the_pc_of_the_dataset_it_was_requested_for(run):
    """Same dataset throughout: the record is the requested dataset's (a normal run is unchanged)."""
    client, store, sig = run
    assert I._get_result().metadata["pc_at_run"]["pc_source"] == "header"


def test_exports_of_a_result_that_knows_its_pc_but_not_its_camera_tilt(run, tmp_path):
    """The record of an imported old export: PC and sample tilt, camera tilt unknown (None)."""
    client, store, sig = run
    rec = I._get_result().metadata["pc_at_run"]
    rec.update(tilt=None, azimuthal=None, pc_source="export")
    ang = tmp_path / "partial.ang"
    assert client.post("/api/indexing/export", json={"format": "ang", "filename": str(ang)}).status_code == 200
    text = ang.read_text(encoding="utf-8", errors="replace")
    np.testing.assert_allclose(_export_ang(client, tmp_path / "p2.ang"),
                               [RUN_PC[0], 1 - RUN_PC[1], RUN_PC[2]], atol=1e-6)   # the PC is stated
    assert "detector_tilt_deg" not in text                                          # the unknown tilt is not invented
    out = tmp_path / "partial.h5"
    assert client.post("/api/indexing/export", json={"format": "h5_light", "filename": str(out)}).status_code == 200
    with h5py.File(out, "r") as f:
        g = f["Detector"]
        np.testing.assert_allclose(g["pc"][()], RUN_PC, atol=1e-12)
        assert g.attrs["sample_tilt"] == pytest.approx(70.0)
        assert "camera_tilt" not in g.attrs and bool(g.attrs["camera_tilt_not_recorded"]) is True
