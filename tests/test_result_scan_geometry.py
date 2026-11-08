"""A result carries the geometry of the scan it ran on: stamped, listed, never re-read live."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from tests.apex_helpers import make_apex_h5, make_up, measurement_spec, wipe_backend  # noqa: E402
from tests.test_crop_provenance import (  # noqa: E402,F401
    _export_light_result, _import_and_read_back, _store_dummy_result, cropped_result,
)

N_ROWS, N_COLS = 6, 5


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    """Each test starts with an empty result registry and leaves behind the one it found. GET /results
    lists every entry, and tests/test_pattern_match_source_file.py stores SimpleNamespace results there
    that have no ``.method``: in one process after that file, four tests here failed."""
    from backend.api.routes import indexing
    saved, saved_active = list(indexing._result_registry.items()), indexing._active_result_id
    indexing._result_registry.clear()
    wipe_backend()
    yield
    wipe_backend()
    indexing._result_registry.clear()
    indexing._result_registry.update(saved)
    indexing._active_result_id = saved_active


def _load_hex(client, tmp_path):
    folder = tmp_path / "hex"
    folder.mkdir()
    make_apex_h5(folder / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5)])
    up = make_up(folder / "scan_hex.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS)
    r = client.post("/api/ebsd/load", json={"path": str(up)})
    assert r.status_code == 200, r.text


def _results(client):
    return {e["id"]: e for e in client.get("/api/indexing/results").json()["results"]}


def test_a_result_is_stamped_with_the_geometry_of_the_dataset_it_ran_on(client, tmp_path):
    from backend.api.routes import indexing
    _load_hex(client, tmp_path)
    rid, result = _store_dummy_result()
    try:
        assert result.metadata["scan_geometry"]["kind"] == "hex"
        assert result.metadata["scan_geometry"]["odd_row_shift"] == 0.5
        assert _results(client)[rid]["scan_geometry"] == result.metadata["scan_geometry"]
    finally:
        indexing._result_registry.pop(rid, None)


def test_a_result_keeps_the_geometry_it_ran_on_when_the_dataset_changes_afterwards(client, tmp_path):
    from backend.api.routes import indexing
    _load_hex(client, tmp_path)
    first, _ = _store_dummy_result()
    crop = client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 1, "col0": 1, "rows": 3, "cols": 3}).json()
    second, _ = _store_dummy_result()
    try:
        listed = _results(client)
        assert listed[first]["scan_geometry"]["odd_row_shift"] == 0.5        # stamped, not re-read
        assert listed[second]["scan_geometry"]["odd_row_shift"] == -0.5
        assert listed[second]["scan_geometry"]["grid_shape"] == [3, 3] and crop["success"]
    finally:
        indexing._result_registry.pop(first, None)
        indexing._result_registry.pop(second, None)


def test_a_result_stored_before_the_stamp_existed_lists_with_none(client, tmp_path):
    from backend.api.routes import indexing
    _load_hex(client, tmp_path)
    rid, result = _store_dummy_result()
    try:
        result.metadata.pop("scan_geometry", None)
        assert _results(client)[rid]["scan_geometry"] is None
    finally:
        indexing._result_registry.pop(rid, None)


def test_an_imported_result_does_not_get_the_geometry_of_whatever_is_loaded(client, tmp_path):
    """The file does not state a geometry yet. Unseeded, _store_result's setdefault would stamp the
    hexagonal scan that happens to be loaded on a result that came from somewhere else."""
    _load_hex(client, tmp_path)
    exported = _export_light_result(tmp_path, {"scan_row_offset": 0, "scan_col_offset": 0, "scan_shape": None})
    entry = _import_and_read_back(client, exported)
    assert entry["scan_geometry"] is None


def test_the_exporters_still_write_the_three_attributes_and_nothing_else(client, cropped_result, tmp_path):
    """A dict cannot be an HDF5 attribute: the geometry must not reach _write_scan_provenance."""
    import h5py
    from backend.api.routes import indexing
    cropped_result.metadata["scan_geometry"] = ScanGeometry.square(4, 4, 0.5).to_dict()
    assert set(indexing._stored_scan_provenance(cropped_result)) == {
        "scan_row_offset", "scan_col_offset", "scan_shape"}
    out = tmp_path / "result_light.h5"
    r = client.post("/api/indexing/export", json={"format": "h5_light", "filename": str(out)})
    assert r.status_code == 200, r.text
    with h5py.File(out, "r") as f:
        assert "scan_geometry" not in f["Indexing"].attrs


def test_the_analysis_dataset_made_from_a_result_carries_its_geometry(client, cropped_result):
    from backend.api.routes import analysis
    hex_dict = ScanGeometry.hex(4, 4, 0.5).to_dict()
    cropped_result.metadata["scan_geometry"] = hex_dict
    try:
        r = client.post("/api/analysis/load", json={"xmap_path": "__from_indexing__"})
        assert r.status_code == 200, r.text
        assert analysis._dataset.scan_geometry == hex_dict
    finally:
        analysis._dataset = None


# --- hex_resampled: a result on a map that was resampled from a hexagonal scan says so -----------

def _tag_active_signal_resampled():
    """What ``edax_hex`` does to the square map it builds from an OIM-form hexagonal file."""
    from backend.api.routes import ebsd_viewer
    signal = ebsd_viewer._get_active_signal()
    assert signal is not None
    signal.metadata.set_item("Signal.hex_resampled", True)


def test_a_result_on_a_resampled_map_is_stamped_and_listed_as_resampled(client, tmp_path):
    from backend.api.routes import indexing
    _load_hex(client, tmp_path)
    _tag_active_signal_resampled()
    rid, result = _store_dummy_result()
    try:
        assert result.metadata["hex_resampled"] is True
        assert _results(client)[rid]["hex_resampled"] is True
    finally:
        indexing._result_registry.pop(rid, None)


def test_a_result_on_an_ordinary_scan_is_stamped_not_resampled(client, tmp_path):
    """The loaded APEX scan is a native hexagonal one: hexagonal is not the same as resampled."""
    from backend.api.routes import indexing
    _load_hex(client, tmp_path)
    rid, result = _store_dummy_result()
    try:
        assert result.metadata["hex_resampled"] is False
        assert _results(client)[rid]["hex_resampled"] is False
    finally:
        indexing._result_registry.pop(rid, None)


def test_a_result_keeps_its_resampled_stamp_when_another_scan_is_loaded_afterwards(client, tmp_path):
    from backend.api.routes import indexing
    _load_hex(client, tmp_path)
    _tag_active_signal_resampled()
    rid, _ = _store_dummy_result()
    wipe_backend()                                  # the dataset changes: nothing is tagged any more
    try:
        assert _results(client)[rid]["hex_resampled"] is True
    finally:
        indexing._result_registry.pop(rid, None)


def test_a_result_stored_without_a_signal_lists_hex_resampled_as_unknown(client):
    from backend.api.routes import indexing
    rid, result = _store_dummy_result()
    try:
        assert result.metadata["hex_resampled"] is None
        assert _results(client)[rid]["hex_resampled"] is None
    finally:
        indexing._result_registry.pop(rid, None)


def test_a_result_stored_before_the_resampled_stamp_existed_lists_with_none(client, tmp_path):
    from backend.api.routes import indexing
    _load_hex(client, tmp_path)
    rid, result = _store_dummy_result()
    try:
        result.metadata.pop("hex_resampled", None)
        assert _results(client)[rid]["hex_resampled"] is None
    finally:
        indexing._result_registry.pop(rid, None)


def test_an_imported_result_is_not_stamped_with_the_resampled_state_of_whatever_is_loaded(client, tmp_path):
    """No signal stands behind an imported result: unknown, even while a tagged scan is loaded."""
    _load_hex(client, tmp_path)
    _tag_active_signal_resampled()
    exported = _export_light_result(tmp_path, {"scan_row_offset": 0, "scan_col_offset": 0, "scan_shape": None})
    entry = _import_and_read_back(client, exported)
    assert entry["hex_resampled"] is None


# --- the stamp's key sets are pinned: the geometry never joins what the exporters write ------------

OFFSET_KEYS = {"scan_row_offset", "scan_col_offset", "scan_shape"}
GEOMETRY_KEYS = {"scan_geometry", "hex_resampled"}


def test_the_provenance_dict_holds_the_three_offset_keys_and_never_the_geometry(client, tmp_path):
    """``_scan_provenance_fields`` is what ``_write_scan_provenance`` turns into HDF5 attributes, and
    a dict cannot be one. Both of its returns are checked: the no-crop one and the windowed one."""
    from backend.api.routes import indexing
    assert set(indexing._scan_provenance_fields()) == OFFSET_KEYS            # nothing loaded: no window
    _load_hex(client, tmp_path)
    assert set(indexing._scan_provenance_fields()) == OFFSET_KEYS            # loaded, not cropped
    assert client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 1, "col0": 1, "rows": 3, "cols": 3}).json()["success"]
    windowed = indexing._scan_provenance_fields()
    assert windowed["scan_row_offset"] == 1 and windowed["scan_col_offset"] == 1   # premise: a real window
    assert set(windowed) == OFFSET_KEYS


def test_the_geometry_readers_and_the_run_stamp_hold_exactly_their_keys(client, tmp_path):
    from backend.api.routes import ebsd_viewer, indexing
    _load_hex(client, tmp_path)
    assert set(indexing._scan_geometry_fields()) == GEOMETRY_KEYS
    signal = ebsd_viewer._get_active_signal()
    assert set(indexing._scan_geometry_fields_of(ebsd_viewer._active_dataset, signal)) == GEOMETRY_KEYS
    # the run stamp also holds the vendor of the file the run reads (the Euler frame of its exports)
    assert set(indexing._scan_stamp_at_run_start(ebsd_viewer._active_dataset, signal)) == (
        OFFSET_KEYS | GEOMETRY_KEYS | {"source_vendor"})


# --- the geometry and the signal are read independently, a failure leaves a trace -----------------

def test_a_failing_geometry_lookup_still_yields_the_resampled_flag(client, tmp_path, monkeypatch, caplog):
    from backend.api.routes import ebsd_viewer, indexing
    from backend.api.services import scan_geometry
    _load_hex(client, tmp_path)
    _tag_active_signal_resampled()

    def broken(_name):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(scan_geometry, "geometry_for", broken)
    with caplog.at_level("WARNING", logger=indexing.logger.name):
        fields = indexing._scan_geometry_fields_of(
            ebsd_viewer._active_dataset, ebsd_viewer._get_active_signal())
    assert fields == {"scan_geometry": None, "hex_resampled": True}
    assert any("scan geometry" in r.getMessage() and r.exc_info for r in caplog.records)


def test_a_failing_signal_read_still_yields_the_geometry(client, tmp_path, monkeypatch, caplog):
    from backend.api.routes import ebsd_viewer, indexing

    _load_hex(client, tmp_path)

    def broken(_signal):
        raise RuntimeError("signal unreadable")

    monkeypatch.setattr(ebsd_viewer, "_signal_is_hex_resampled", broken)
    with caplog.at_level("WARNING", logger=indexing.logger.name):
        fields = indexing._scan_geometry_fields_of(
            ebsd_viewer._active_dataset, ebsd_viewer._get_active_signal())
    assert fields["scan_geometry"]["kind"] == "hex"
    assert fields["hex_resampled"] is None
    assert any("hexagonal" in r.getMessage() and r.exc_info for r in caplog.records)


# --- the holders own their copy of the geometry ---------------------------------------------------

def test_what_stored_scan_geometry_returns_is_a_copy(client, tmp_path):
    from backend.api.routes import indexing
    _load_hex(client, tmp_path)
    rid, result = _store_dummy_result()
    try:
        returned = indexing._stored_scan_geometry(result)
        assert returned == result.metadata["scan_geometry"] and returned is not result.metadata["scan_geometry"]
        returned["odd_row_shift"] = 99.0                 # a consumer edits what it was handed
        assert result.metadata["scan_geometry"]["odd_row_shift"] == 0.5
    finally:
        indexing._result_registry.pop(rid, None)


def test_the_analysis_dataset_owns_its_copy_of_the_results_geometry(client, cropped_result):
    from backend.api.routes import analysis
    cropped_result.metadata["scan_geometry"] = ScanGeometry.hex(4, 4, 0.5).to_dict()
    try:
        r = client.post("/api/analysis/load", json={"xmap_path": "__from_indexing__"})
        assert r.status_code == 200, r.text
        held = analysis._dataset.scan_geometry
        assert held == cropped_result.metadata["scan_geometry"]
        assert held is not cropped_result.metadata["scan_geometry"]
        held["odd_row_shift"] = 99.0
        assert cropped_result.metadata["scan_geometry"]["odd_row_shift"] == 0.5
    finally:
        analysis._dataset = None


# --- every analysis dataset says which grid it sits on --------------------------------------------

HEX_DICT = ScanGeometry.hex(5, 5, 0.5).to_dict()


def test_a_fresh_analysis_dataset_says_unknown_not_square():
    from tests.test_smoothing import _make_test_dataset
    assert _make_test_dataset().scan_geometry is None


def test_smoothing_keeps_the_geometry_of_the_dataset_it_smooths():
    from analysis.smoothing import apply_smoothing
    from tests.test_smoothing import _make_test_dataset
    dataset = _make_test_dataset()
    dataset.scan_geometry = dict(HEX_DICT)
    smoothed = apply_smoothing(dataset)
    assert smoothed is not dataset
    assert smoothed.scan_geometry == HEX_DICT and smoothed.scan_geometry is not dataset.scan_geometry


def test_crop_to_indexed_keeps_the_geometry_of_the_dataset_it_crops():
    from tests.test_smoothing import _make_test_dataset
    dataset = _make_test_dataset()
    dataset.scan_geometry = dict(HEX_DICT)
    cropped = dataset.crop_to_indexed()
    assert cropped is not dataset
    assert cropped.scan_geometry == HEX_DICT and cropped.scan_geometry is not dataset.scan_geometry


def test_an_analysis_dataset_made_from_a_result_without_a_stamp_says_unknown(client, cropped_result):
    from backend.api.routes import analysis
    cropped_result.metadata.pop("scan_geometry", None)
    try:
        r = client.post("/api/analysis/load", json={"xmap_path": "__from_indexing__"})
        assert r.status_code == 200, r.text
        assert analysis._dataset.scan_geometry is None
    finally:
        analysis._dataset = None


def test_a_file_load_after_a_result_load_drops_the_results_geometry(client, cropped_result, monkeypatch, tmp_path):
    """A file does not state its grid yet: the geometry of the result loaded before must not carry over."""
    import orix.io
    from backend.api.routes import analysis
    from tests.test_smoothing import _make_test_dataset
    cropped_result.metadata["scan_geometry"] = dict(HEX_DICT)
    path = tmp_path / "scan.ang"
    path.write_text("not read: orix.io.load is replaced\n")
    monkeypatch.setattr(orix.io, "load", lambda _path: _make_test_dataset().xmap)
    try:
        assert client.post("/api/analysis/load", json={"xmap_path": "__from_indexing__"}).status_code == 200
        assert analysis._dataset.scan_geometry == HEX_DICT                    # premise
        r = client.post("/api/analysis/load", json={"xmap_path": str(path)})
        assert r.status_code == 200, r.text
        assert analysis._dataset.source == str(path)                          # premise: the file branch ran
        assert analysis._dataset.scan_geometry is None
    finally:
        analysis._dataset = None


# --- a run stamps the dataset it indexed, taken when it starts, not when the result is stored -----

def _a_window(row0, col0):
    """What ``_scan_provenance_fields`` reads off a crop window (hand constants)."""
    from types import SimpleNamespace
    return SimpleNamespace(row0=row0, col0=col0, original_shape=(90, 120), nav_mask=None)


def test_a_run_stamps_the_dataset_it_indexed_even_when_another_is_activated_during_the_run(
        monkeypatch, tmp_path):
    """The stamp is the run's, not whatever is active when the result is stored: a crop or a file
    switch during the run makes a new dataset active, and the finished result must still say hex,
    resampled, offset (7, 9)."""
    import indexing_controller
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from backend.api.services import scan_geometry
    from backend.api.services.calibration_store import CalibrationStore
    from tests.test_result_keeps_its_run_pc import RUN_PC, SHAPE, _fake_hough, _ni_cif

    sig = kp.data.nickel_ebsd_small()
    sig.detector = EBSDDetector(shape=SHAPE, pc=RUN_PC, sample_tilt=70.0, tilt=3.5)
    sig.metadata.set_item("Signal.hex_resampled", True)
    other_sig = kp.data.nickel_ebsd_small()                  # untagged: what a switch would show
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
    monkeypatch.setattr(I, "_active_crop_window", lambda: _a_window(7, 9))
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="indexing": fn())

    def indexer_during_which_another_dataset_becomes_active(*a, **k):
        monkeypatch.setattr(ev, "_active_dataset", "other")
        monkeypatch.setattr(ev, "_get_active_signal", lambda *a2, **k2: other_sig)
        monkeypatch.setattr(I, "_active_crop_window", lambda: None)
        return _fake_hough(*a, **k)

    monkeypatch.setattr(indexing_controller, "hough_index_patterns",
                        indexer_during_which_another_dataset_becomes_active)
    scan_geometry.set_geometry("ds", ScanGeometry.hex(3, 3, 0.5))
    scan_geometry.set_geometry("other", ScanGeometry.square(3, 3, 0.5))
    saved, saved_active = dict(I._result_registry), I._active_result_id
    try:
        from backend.api.main import app
        resp = TestClient(app).post(
            "/api/indexing/start", json={"method": "hough", "cif_paths": [_ni_cif(tmp_path)]})
        assert resp.status_code == 200, resp.text
        task = I._indexing_tasks[resp.json()["task_id"]]
        assert task["status"] == "completed", task
        assert ev._active_dataset == "other"                 # premise: the switch really happened
        meta = I._get_result().metadata
        assert meta["scan_geometry"]["kind"] == "hex"        # not the square one of "other"
        assert meta["hex_resampled"] is True                 # not the untagged signal's False
        assert (meta["scan_row_offset"], meta["scan_col_offset"]) == (7, 9)
        assert meta["scan_shape"] == [90, 120]
    finally:
        scan_geometry.clear("ds")
        scan_geometry.clear("other")
        I._result_registry.clear()
        I._result_registry.update(saved)
        I._active_result_id = saved_active


def test_a_batch_item_stamps_the_dataset_it_indexed_even_when_another_is_activated_during_it(
        client, monkeypatch, tmp_path):
    """Same rule in the batch loop: the stamp is taken right after the item's own load."""
    import numpy as np
    import indexing_controller
    from indexing_controller import IndexingMethod, IndexingResult
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from backend.api.services import scan_geometry

    folder = tmp_path / "hexbatch"
    folder.mkdir()
    make_apex_h5(folder / "s.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5)])
    up = make_up(folder / "scan_hex.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS)
    seen = {}

    def spherical_during_which_another_dataset_becomes_active(h5_path, config, detector_params,
                                                              selection_mask, **kwargs):
        seen["loaded_as"] = ev._active_dataset
        scan_geometry.set_geometry("other", ScanGeometry.square(2, 2, 0.25))
        monkeypatch.setattr(ev, "_active_dataset", "other")
        mask = np.asarray(selection_mask, dtype=bool)
        return IndexingResult(
            xmap=None, selection_mask=mask, original_shape=mask.shape,
            method=IndexingMethod.SPHERICAL,
            confidence_scores=np.full(mask.size, 0.5, dtype=np.float32))

    for name in ("spherical_index_patterns", "spherical_gpu_index_patterns"):
        monkeypatch.setattr(indexing_controller, name, spherical_during_which_another_dataset_becomes_active)
    monkeypatch.setattr(I, "_attach_indexing_metadata", lambda *a, **k: None)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    saved_batch, saved_results = dict(I._batch_state), dict(I._result_registry)
    try:
        resp = client.post("/api/indexing/batch/start", json={"datasets": [{
            "file_path": str(up), "method": "spherical", "selection_mode": "full",
            "export_result": False}]})
        assert resp.status_code == 200, resp.text
        entry = I._batch_state["results"][-1]
        assert entry["error"] is None, entry
        assert ev._active_dataset == "other"                 # premise: the switch really happened
        meta = I._get_result().metadata
        assert meta["scan_geometry"]["kind"] == "hex"        # the file's, not the square one of "other"
        assert meta["scan_geometry"]["grid_shape"] == [N_ROWS, N_COLS]
        assert meta["scan_geometry"]["odd_row_shift"] == 0.5
        assert meta["hex_resampled"] is False                # a signal was there to ask
    finally:
        scan_geometry.clear("other")
        I._batch_state.clear()
        I._batch_state.update(saved_batch)
        I._result_registry.clear()
        I._result_registry.update(saved_results)


# --- a refinement covers its parent's pixels: it reports its parent's geometry --------------------

def _refine(parent):
    """Run the real refinement over a fake parent; return the refined result's id."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    import torch
    from backend.api.services import refinement as rf

    renderer = MagicMock()
    renderer.render_batch.return_value = torch.ones(16, 8, 8, dtype=torch.float32)
    renderer.render.return_value = torch.ones(8, 8, dtype=torch.float32)
    with patch.object(rf, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(rf, "get_renderer", return_value=renderer), \
         patch.object(rf, "_get_experimental_pattern", return_value=np.ones((8, 8), dtype=np.float32)):
        rf.compute_full_refinement(parent, smoothness_lambda=0.01)
    return parent.metadata["refinement"]["refined_result_id"]


def test_a_refinement_inherits_its_parents_geometry(client, tmp_path):
    from backend.api.routes import indexing
    from tests.test_refinement_service import _make_fake_indexed_result

    _load_hex(client, tmp_path)
    _tag_active_signal_resampled()
    parent = _make_fake_indexed_result()
    parent_rid = indexing._store_result(parent, "spherical")
    assert parent.metadata["scan_geometry"]["kind"] == "hex"               # premise
    assert parent.metadata["scan_geometry"]["odd_row_shift"] == 0.5
    assert parent.metadata["hex_resampled"] is True
    # The live state moves on before the user refines: a crop starting on an odd row flips the
    # shift (see the test above), and wiping the backend leaves no geometry and no signal at all.
    assert client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 1, "col0": 1, "rows": 3, "cols": 3}).json()["success"]
    wipe_backend()
    refined_rid = None
    try:
        refined_rid = _refine(parent)
        refined = indexing._result_registry[refined_rid]
        assert refined.metadata["scan_geometry"] == parent.metadata["scan_geometry"]
        assert refined.metadata["scan_geometry"]["kind"] == "hex"
        assert refined.metadata["scan_geometry"]["odd_row_shift"] == 0.5
        assert refined.metadata["hex_resampled"] is True
    finally:
        indexing._result_registry.pop(refined_rid, None)
        indexing._result_registry.pop(parent_rid, None)


def test_a_refinement_of_a_parent_without_a_stamp_says_unknown_not_the_live_geometry(client, tmp_path):
    from backend.api.routes import indexing
    from tests.test_refinement_service import _make_fake_indexed_result

    _load_hex(client, tmp_path)
    _tag_active_signal_resampled()
    parent = _make_fake_indexed_result()
    parent_rid = indexing._store_result(parent, "spherical")
    parent.metadata.pop("scan_geometry")                                   # a result from before the stamp
    parent.metadata.pop("hex_resampled")
    refined_rid = None
    try:
        refined_rid = _refine(parent)                                      # the hex scan is still active
        refined = indexing._result_registry[refined_rid]
        assert refined.metadata["scan_geometry"] is None
        assert refined.metadata["hex_resampled"] is None
    finally:
        indexing._result_registry.pop(refined_rid, None)
        indexing._result_registry.pop(parent_rid, None)
