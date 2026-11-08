"""An imported result of a file that carries no PC is not rendered at the placeholder.

``import-h5`` rebuilt the render geometry (``detector_geometry``) of a result from the
loaded file's detector. For a file with no PC that detector holds kikuchipy's
placeholder (0.5, 0.5, 0.5), so the pattern match and the forward diagnostics drew the
simulated pattern at a PC nobody measured, as if it were the run's. The result is now
marked ``pc_missing`` with no geometry, the pattern match says why it cannot render, and
the forward diagnostics refuse with the coded 400. Real export -> real import.
"""
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


@pytest.fixture(autouse=True)
def restore_registry():
    saved = dict(indexing_mod._result_registry)
    saved_active = indexing_mod._active_result_id
    try:
        yield
    finally:
        indexing_mod._result_registry.clear()
        indexing_mod._result_registry.update(saved)
        indexing_mod._active_result_id = saved_active


def _signal(tag="default"):
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    rng = np.random.default_rng(5)
    sig = kp.signals.EBSD(rng.integers(0, 255, (3, 4) + SHAPE).astype(np.uint8))
    sig.detector = EBSDDetector(shape=SHAPE, pc=(0.5, 0.5, 0.5), sample_tilt=70.0)
    if tag:
        sig.metadata.set_item("Signal.pc_source", tag)        # "default": the loader found no PC in the file
    return sig


@pytest.fixture(params=["missing"])
def imported(request, monkeypatch, tmp_path):
    """A result of a file (without a PC, or with one), exported (rich h5) and imported again."""
    from backend.api.routes import ebsd_viewer as ev
    real_get_active_signal = ev._get_active_signal
    n_rows, n_cols = 3, 4
    n = n_rows * n_cols
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(np.array([[i * 9.0, 20.0, 30.0] for i in range(n)]))),
        phase_id=np.zeros(n, dtype=np.int32),
        x=np.tile(np.arange(n_cols, dtype=np.float32), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=np.float32), n_cols),
        phase_list=PhaseList(phases=[Phase(name="Al", space_group=225, point_group="m-3m")]))
    result = IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool), original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL, confidence_scores=np.full(n, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.5, "indexing_method": "spherical",
                  "recorded_lattices": {0: [4.05, 4.05, 4.05, 90, 90, 90]}})
    monkeypatch.setitem(indexing_mod._result_registry, "src", result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", "src")

    store = CalibrationStore()
    sig = _signal("default" if request.param == "missing" else None)
    store.register("ds", sig)
    assert store.get_entry("ds").pc_source == ("missing" if request.param == "missing" else "header")
    from backend.api.services.pc_record import pc_run_record
    result.metadata["pc_at_run"] = pc_run_record("ds", store)       # as the run records it at its start
    monkeypatch.setattr(indexing_mod, "calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    up1 = tmp_path / "scan.up1"
    up1.write_bytes(b"not an hdf5 file")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(up1), raising=False)
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: sig)
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    out = tmp_path / "rich.h5"
    client = TestClient(app)
    assert client.post("/api/indexing/export", json={"format": "h5", "filename": str(out)}).status_code == 200
    # a fresh viewer: the import loads the export as the active file
    monkeypatch.setattr(ev, "_get_active_signal", real_get_active_signal)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    resp = client.post("/api/indexing/import-h5", json={"path": str(out)})
    assert resp.status_code == 200, resp.text
    return client, resp.json()["result_id"], store


def test_the_import_marks_the_result_and_builds_no_placeholder_geometry(imported):
    client, rid, store = imported
    md = indexing_mod._result_registry[rid].metadata
    assert md.get("pc_missing") is True
    assert "detector_geometry" not in md
    assert store.get_entry("rich").pc_source == "missing"        # the imported file is a file without a PC


@pytest.mark.parametrize("library_has_master", [False, True])
def test_the_pattern_match_says_why_it_cannot_render(imported, library_has_master):
    # The import looks the phase's master up in the library; whether it finds one depends
    # on the machine. Both cases are pinned here, so the answer cannot depend on it.
    client, rid, _ = imported
    md = indexing_mod._result_registry[rid].metadata
    md["sht_paths_by_phase"] = {0: "Al.sht"} if library_has_master else {}
    resp = client.get("/api/indexing/pattern-match", params={"row": 0, "col": 0, "result_id": rid})
    assert "carries no pattern centre" in resp.text         # not a pattern simulated at (0.5, 0.5, 0.5)


def test_the_pattern_match_carries_the_code_a_refused_source_file_gave(imported, monkeypatch):
    """The experimental panel is empty when the result's own file was refused with a code; the
    answer names the code so the window can say why instead of showing a bare field."""
    import tools.pattern_comparison as pattern_comparison
    client, rid, _ = imported
    params = {"row": 0, "col": 0, "result_id": rid}
    assert client.get("/api/indexing/pattern-match", params=params).json()["experimental_error_code"] is None

    def refused(result, row, col, *, for_indexer=False, refusals=None):
        if refusals is not None:
            refusals.append("loadCameUpAsOther")
        return None

    monkeypatch.setattr(pattern_comparison, "get_experimental_pattern", refused)
    body = client.get("/api/indexing/pattern-match", params=params).json()
    assert body["experimental_error_code"] == "loadCameUpAsOther"
    assert body["experimental"] is None


def test_the_forward_diagnostics_refuse_with_the_code(imported):
    client, rid, _ = imported
    resp = client.post("/api/forward-diagnostics/compute", json={"result_id": rid, "max_bandwidth": 128})
    assert resp.status_code == 400
    assert resp.headers["x-orienta-code"] == "pc_missing"


@pytest.mark.parametrize("imported", ["header"], indirect=True)
def test_a_result_of_a_file_with_a_pc_still_gets_its_geometry(imported):
    """The guard is for files without a PC only: the geometry is rebuilt as before."""
    client, rid, store = imported
    md = indexing_mod._result_registry[rid].metadata
    assert "pc_missing" not in md
    assert md.get("detector_geometry"), "the render geometry of a file with a PC is rebuilt"
    assert store.get_entry("rich").pc_source == "header"
