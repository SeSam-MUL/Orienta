"""POST /api/indexing/export (rich h5) of an EDAX UP1 source states the run's PC.

The route builds a fresh EDAX-format h5 for a source that is not HDF5. The PC it
writes into the header must be the one the run used, i.e. the calibration store's
(a Global PC Refine writes the store, not the signal's detector), and a file whose
PC nobody supplied must get none. The writer itself is covered in
``test_rich_export_keeps_pc``; this is the call in the route.
"""
import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from backend.api.main import app
from backend.api.routes import indexing as indexing_mod
from backend.api.services.calibration_store import CalibrationStore
from indexing_controller import IndexingMethod, IndexingResult

SHAPE = (60, 40)
SIGNAL_PC = (0.5, 0.5, 0.5)          # what the signal's own detector holds
STORE_PC = (0.5312, 0.4207, 0.6498)  # what the run used (refined)


@pytest.fixture(autouse=True)
def restore_indexing_registry():
    saved = dict(indexing_mod._result_registry)
    saved_active = indexing_mod._active_result_id
    try:
        yield
    finally:
        indexing_mod._result_registry.clear()
        indexing_mod._result_registry.update(saved)
        indexing_mod._active_result_id = saved_active


def _signal():
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    rng = np.random.default_rng(5)
    sig = kp.signals.EBSD(rng.integers(0, 255, (3, 4) + SHAPE).astype(np.uint8))
    sig.detector = EBSDDetector(shape=SHAPE, pc=SIGNAL_PC, sample_tilt=70.0)
    return sig


@pytest.fixture
def route(monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer
    n_rows, n_cols = 3, 4
    n = n_rows * n_cols
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(np.array([[i * 9.0, 20.0, 30.0] for i in range(n)]))),
        phase_id=np.zeros(n, dtype=np.int32),
        x=np.tile(np.arange(n_cols, dtype=np.float32), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=np.float32), n_cols),
        phase_list=PhaseList(phases=[Phase(name="Al", space_group=225, point_group="m-3m")]),
    )
    result = IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool), original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL, confidence_scores=np.full(n, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.5, "recorded_lattices": {0: [4.05, 4.05, 4.05, 90, 90, 90], 1: [4.05, 4.05, 4.05, 90, 90, 90]}})
    monkeypatch.setitem(indexing_mod._result_registry, "rich_pc", result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", "rich_pc")

    up1 = tmp_path / "scan.up1"
    up1.write_bytes(b"not an hdf5 file")               # a source that is not HDF5
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", str(up1), raising=False)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: _signal())
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    return tmp_path


def _use(monkeypatch, store):
    """The state of the store when the run STARTS: it is patched in and the registered
    result takes its record (``pc_at_run``) as the indexing route does at its start."""
    from backend.api.services.pc_record import pc_run_record
    monkeypatch.setattr(indexing_mod, "calibration_store", store)
    result = indexing_mod._result_registry["rich_pc"]
    result.metadata["pc_at_run"] = pc_run_record("ds", store)


def _store(pc):
    store = CalibrationStore()
    store.register("ds", type("S", (), {"detector": _signal().detector})())
    if pc is not None:
        store.update_pc("ds", pc, source="refined")
    else:
        store.get_entry("ds").pc_source = "missing"
    return store


def _export(tmp_path):
    out = tmp_path / "rich.h5"
    resp = TestClient(app).post("/api/indexing/export", json={"format": "h5", "filename": str(out)})
    assert resp.status_code == 200, resp.text
    return out


def test_the_header_states_the_pc_the_run_used(route, monkeypatch):
    _use(monkeypatch, _store(STORE_PC))
    out = _export(route)
    # in the HEADER (the /Detector group would give the same PC back on load): the
    # run's PC in the TSL convention, by hand: x* = PCx, y* = 1 - PCy, z* = PCz * rows / min
    with h5py.File(out, "r") as f:
        g = f["Scan1/EBSD/Header/Pattern Center Calibration"]
        assert g["x-star"][0] == pytest.approx(STORE_PC[0])
        assert g["y-star"][0] == pytest.approx(1 - STORE_PC[1])
        assert g["z-star"][0] == pytest.approx(STORE_PC[2] * 60 / 40)
    from safe_loader import load_ebsd_safe
    back = load_ebsd_safe(str(out), verbose=False)
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), STORE_PC, atol=1e-6)


def test_no_header_pc_is_written_for_a_file_that_has_none(route, monkeypatch):
    _use(monkeypatch, _store(None))
    out = _export(route)
    with h5py.File(out, "r") as f:
        assert "Pattern Center Calibration" not in f["Scan1/EBSD/Header"]
        # nor does /Detector state the signal's placeholder as if it were one
        assert "pc" not in f["Detector"]
        assert bool(f["Detector"].attrs["pc_missing"]) is True
    # and loading the export again does not turn the placeholder into a header PC
    from safe_loader import load_ebsd_safe
    from backend.api.services.calibration_store import CalibrationStore
    back = load_ebsd_safe(str(out), verbose=False)
    assert CalibrationStore().register("again", back).pc_source == "missing"


def test_light_export_states_the_run_pc_and_not_a_placeholder(route, monkeypatch):
    out = route / "light.h5"
    client = TestClient(app)

    _use(monkeypatch, _store(STORE_PC))
    assert client.post("/api/indexing/export",
                       json={"format": "h5_light", "filename": str(out)}).status_code == 200
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["Detector/pc"][()], STORE_PC, atol=1e-12)

    out2 = route / "light_missing.h5"
    _use(monkeypatch, _store(None))
    assert client.post("/api/indexing/export",
                       json={"format": "h5_light", "filename": str(out2)}).status_code == 200
    with h5py.File(out2, "r") as f:
        assert "pc" not in f["Detector"]
        assert bool(f["Detector"].attrs["pc_missing"]) is True


# --- the tilts an export states are the ones the run used ---------------------------

def _edited_store():
    store = _store(STORE_PC)
    store.update_geometry("ds", sample_tilt=72.5, tilt=3.0, azimuthal=0.5)   # typed on the PC page
    return store


def test_the_rich_export_states_the_edited_tilts(route, monkeypatch):
    """The signal's own detector still holds 70 / 0 / 0; the run used the store's."""
    _use(monkeypatch, _edited_store())
    out = _export(route)
    with h5py.File(out, "r") as f:
        hdr = f["Scan1/EBSD/Header"]
        assert hdr["Sample Tilt"][0] == pytest.approx(72.5)
        assert hdr["Camera Elevation Angle"][0] == pytest.approx(3.0)
        assert hdr["Camera Azimuthal Angle"][0] == pytest.approx(0.5)
        assert f["Detector"].attrs["sample_tilt"] == pytest.approx(72.5)
        assert f["Detector"].attrs["camera_tilt"] == pytest.approx(3.0)


def test_the_light_export_states_the_edited_tilts(route, monkeypatch):
    _use(monkeypatch, _edited_store())
    out = route / "light_tilt.h5"
    assert TestClient(app).post("/api/indexing/export",
                                json={"format": "h5_light", "filename": str(out)}).status_code == 200
    with h5py.File(out, "r") as f:
        assert f["Detector"].attrs["sample_tilt"] == pytest.approx(72.5)
        assert f["Detector"].attrs["camera_tilt"] == pytest.approx(3.0)


def test_the_ang_header_states_the_edited_tilts(route, monkeypatch):
    _use(monkeypatch, _edited_store())
    out = route / "tilt.ang"
    resp = TestClient(app).post("/api/indexing/export", json={"format": "ang", "filename": str(out)})
    assert resp.status_code == 200, resp.text
    text = out.read_text(encoding="utf-8", errors="replace")
    assert "sample_tilt_deg=72.500" in text and "detector_tilt_deg=3.000" in text


def test_without_an_edit_the_exports_still_state_the_files_tilts(route, monkeypatch):
    _use(monkeypatch, _store(STORE_PC))
    out = _export(route)
    with h5py.File(out, "r") as f:
        assert f["Scan1/EBSD/Header/Sample Tilt"][0] == pytest.approx(70.0)
        assert f["Scan1/EBSD/Header/Camera Elevation Angle"][0] == pytest.approx(0.0)


def test_the_shared_detector_writer_states_pc_and_tilts_from_the_store(route, monkeypatch, tmp_path):
    """The routine the interactive exports AND the batch export call."""
    result = indexing_mod._result_registry["rich_pc"]
    _use(monkeypatch, _edited_store())
    with h5py.File(tmp_path / "d.h5", "w") as f:
        g = f.create_group("Detector")
        indexing_mod._write_detector_facts(g, "ds", _signal().detector, result)
        np.testing.assert_allclose(g["pc"][()], STORE_PC, atol=1e-12)
        assert g.attrs["sample_tilt"] == pytest.approx(72.5)
        assert g.attrs["camera_tilt"] == pytest.approx(3.0)
    _use(monkeypatch, _store(None))
    with h5py.File(tmp_path / "m.h5", "w") as f:
        g = f.create_group("Detector")
        indexing_mod._write_detector_facts(g, "ds", _signal().detector, result)
        assert "pc" not in g and bool(g.attrs["pc_missing"]) is True


_AL_CIF = "\n".join([
    "data_Al", "_cell_length_a 4.05", "_cell_length_b 4.05", "_cell_length_c 4.05",
    "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
    "_symmetry_space_group_name_H-M 'F m -3 m'", "_symmetry_Int_Tables_number 225",
    "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
    "_atom_site_fract_y", "_atom_site_fract_z", "Al1 Al 0 0 0", ""])


def test_the_legacy_batch_export_states_pc_and_tilts_from_the_store(route, monkeypatch):
    """The batch run reloads the file, indexes it (stubbed here to the synthetic result)
    and exports; its /Detector group must state the store's PC and tilts too."""
    import indexing_controller
    from backend.api.routes import ebsd_viewer
    store = _edited_store()
    _use(monkeypatch, store)
    monkeypatch.setattr(ebsd_viewer, "load_ebsd_file", lambda path: {"ok": True})
    sig = _signal()
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: sig)
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", None, raising=False)
    result = indexing_mod._result_registry["rich_pc"]
    monkeypatch.setattr(indexing_controller, "hough_index_patterns", lambda **kw: result)
    monkeypatch.setattr(indexing_mod, "_submit_run_guarded", lambda fn, kind="": fn())
    out_dir = route / "batch_out"
    out_dir.mkdir()
    (route / "Al.cif").write_text(_AL_CIF, encoding="utf-8")
    saved = dict(indexing_mod._batch_state)
    try:
        resp = TestClient(app).post("/api/indexing/batch/start", json={
            "datasets": [{"file_path": "C:/data/ds.up1", "method": "hough",
                          "cif_paths": [str(route / "Al.cif")]}],
            "auto_export": True, "export_dir": str(out_dir)})
        assert resp.status_code == 200, resp.text
        entry = indexing_mod._batch_state["results"][-1]
        assert not entry["error"], entry["error"]
    finally:
        indexing_mod._batch_state.clear()
        indexing_mod._batch_state.update(saved)
    files = list(out_dir.glob("*.h5"))
    assert files, entry
    with h5py.File(files[0], "r") as f:
        np.testing.assert_allclose(f["Detector/pc"][()], STORE_PC, atol=1e-6)
        assert f["Detector"].attrs["sample_tilt"] == pytest.approx(72.5)
        assert f["Detector"].attrs["camera_tilt"] == pytest.approx(3.0)


# --- the .ang export of a file without a PC says so ---------------------------------

def _export_ang(tmp_path):
    out = tmp_path / "result.ang"
    resp = TestClient(app).post("/api/indexing/export", json={"format": "ang", "filename": str(out)})
    assert resp.status_code == 200, resp.text
    return out.read_text(encoding="utf-8", errors="replace").splitlines()


def _header_value(lines, prefix):
    (line,) = [ln for ln in lines if ln.startswith(prefix)]
    return float(line.split()[-1])


def test_the_ang_of_a_file_without_a_pc_says_that_none_was_calibrated(route, monkeypatch):
    """x/y/z-star stay 0 (orix's placeholder) and a comment line explains why: a
    reader must not take them for a measured PC, and 0.5 must not appear either."""
    _use(monkeypatch, _store(None))
    lines = _export_ang(route)
    assert [_header_value(lines, f"# {k}") for k in ("x-star", "y-star", "z-star")] == [0.0, 0.0, 0.0]
    notes = [ln for ln in lines if "never calibrated" in ln]
    assert len(notes) == 1 and notes[0].startswith("# ")
    assert not any(ln.startswith("# ORIENTA_PATTERN_CENTRE_BRUKER") for ln in lines)   # no PC line is stated
    # the state stands right above the three lines, machine-readably and in words
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# x-star"))
    assert lines[i - 2] == "# ORIENTA_PC: not calibrated"
    assert lines[i - 1].startswith("# ORIENTA_NOTE: x-star, y-star, z-star are zero")


def test_the_ang_of_a_calibrated_file_states_the_tsl_pc_and_no_such_note(route, monkeypatch):
    _use(monkeypatch, _store(STORE_PC))
    lines = _export_ang(route)
    assert _header_value(lines, "# x-star") == pytest.approx(STORE_PC[0], abs=1e-6)
    assert _header_value(lines, "# y-star") == pytest.approx(1 - STORE_PC[1], abs=1e-6)   # TSL: 1 - PCy
    assert _header_value(lines, "# z-star") == pytest.approx(STORE_PC[2] * 60 / 40, abs=1e-6)
    assert not any("never calibrated" in ln for ln in lines)
