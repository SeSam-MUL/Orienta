"""A result from a cropped dataset knows where in the original it sits."""
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    """Start every test from a backend that has never loaded a file.

    Same reason as ``tests/test_crop_endpoint.py``: the viewer's registries are
    module-level and re-loading an already-visited file restores that file's
    derived datasets AND re-selects the dataset that was active before. Without
    this, the "no crop is registered" test would see the previous test's crop.
    """
    from backend.api.routes import ebsd_viewer
    from backend.api.services import crop_window as cw

    def _wipe():
        ebsd_viewer._raw_signals.clear()
        ebsd_viewer._positions.clear()
        ebsd_viewer._dirty_datasets.clear()
        ebsd_viewer._signal_masks.clear()
        ebsd_viewer._overview_cache.clear()
        ebsd_viewer._registry_by_file.clear()
        ebsd_viewer._active_dataset = ""
        ebsd_viewer._ebsd_signal = None
        ebsd_viewer._ebsd_file_path = None
        cw.clear_all()

    _wipe()
    yield
    _wipe()


def _store_dummy_result(metadata=None):
    """Register a minimal result through the real store path.

    Returns ``(result_id, result)``. The caller is responsible for popping the
    id out of the registry again — it is module-global state.
    """
    from indexing_controller import IndexingMethod, IndexingResult
    from backend.api.routes import indexing

    result = IndexingResult(
        xmap=None,
        selection_mask=np.ones((4, 4), dtype=bool),
        original_shape=(4, 4),
        method=IndexingMethod.HOUGH,
        metadata=dict(metadata or {}),
    )
    return indexing._store_result(result, "hough"), result


@pytest.mark.integration
def test_results_report_the_crop_offset(client, h5oina_path):
    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 7, "col0": 9, "rows": 4, "cols": 4})

    # Store a result the way the indexing route does, then read it back.
    from backend.api.routes import indexing
    from backend.api.routes.ebsd_viewer import get_active_crop_window

    window = get_active_crop_window()
    assert (window.row0, window.col0) == (7, 9)

    entry = indexing._scan_provenance_fields()
    assert entry["scan_row_offset"] == 7
    assert entry["scan_col_offset"] == 9
    assert entry["scan_shape"] is not None


@pytest.mark.integration
def test_without_a_crop_the_offsets_are_zero(client, h5oina_path):
    from backend.api.routes import indexing

    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    entry = indexing._scan_provenance_fields()
    assert entry == {"scan_row_offset": 0, "scan_col_offset": 0,
                     "scan_shape": None}


@pytest.mark.integration
def test_results_endpoint_carries_the_fields(client, h5oina_path):
    """Every entry must carry the three keys, and a result stored while a crop
    was active must report that crop's offset — not a fresh reading of
    whatever happens to be loaded when /results is called."""
    from backend.api.routes import indexing

    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 7, "col0": 9, "rows": 4, "cols": 4})
    result_id, _ = _store_dummy_result()
    try:
        body = client.get("/api/indexing/results").json()
        entries = body.get("results", [])
        assert entries, "the stored result must show up in /results"
        for entry in entries:
            assert "scan_row_offset" in entry
            assert "scan_col_offset" in entry
            assert "scan_shape" in entry
        mine = next(e for e in entries if e["id"] == result_id)
        assert mine["scan_row_offset"] == 7
        assert mine["scan_col_offset"] == 9
        assert mine["scan_shape"] is not None
    finally:
        indexing._result_registry.pop(result_id, None)


@pytest.mark.integration
def test_a_result_stored_before_this_change_still_lists(client, h5oina_path):
    """An entry written before crop provenance existed carries no such keys.
    /results must fall back rather than raise."""
    from backend.api.routes import indexing

    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    result_id, result = _store_dummy_result()
    try:
        for key in ("scan_row_offset", "scan_col_offset", "scan_shape"):
            result.metadata.pop(key, None)
        body = client.get("/api/indexing/results").json()
        mine = next(e for e in body["results"] if e["id"] == result_id)
        assert mine["scan_row_offset"] == 0
        assert mine["scan_col_offset"] == 0
        assert mine["scan_shape"] is None
    finally:
        indexing._result_registry.pop(result_id, None)


def test_the_exporter_writes_the_three_attributes(tmp_path):
    """The exporter helper is what puts the provenance into a result file."""
    import h5py
    from backend.api.services.result_exporter import _write_scan_provenance

    prov = {"scan_row_offset": 7, "scan_col_offset": 9,
            "scan_shape": [64, 80]}
    with h5py.File(tmp_path / "prov.h5", "w") as f:
        _write_scan_provenance(f.create_group("Indexing"), prov)
    with h5py.File(tmp_path / "prov.h5", "r") as f:
        idx = f["Indexing"]
        assert int(idx.attrs["scan_row_offset"]) == 7
        assert int(idx.attrs["scan_col_offset"]) == 9
        assert list(idx.attrs["scan_shape"]) == [64, 80]


def test_the_exporter_writes_nothing_without_provenance(tmp_path):
    """Default None must leave an exported file exactly as it was before."""
    import h5py
    from backend.api.services.result_exporter import _write_scan_provenance

    with h5py.File(tmp_path / "plain.h5", "w") as f:
        g = f.create_group("Indexing")
        _write_scan_provenance(g, None)
        # A no-crop dict carries a None shape; h5py cannot store that and
        # "no original shape" is the absent attribute.
        _write_scan_provenance(g, {"scan_row_offset": 0, "scan_col_offset": 0,
                                   "scan_shape": None})
    with h5py.File(tmp_path / "plain.h5", "r") as f:
        assert "scan_shape" not in f["Indexing"].attrs
        assert int(f["Indexing"].attrs["scan_row_offset"]) == 0


# --- An imported result's provenance belongs to its file, not to this session.


def _export_light_result(tmp_path, provenance):
    """Produce a REAL light result file through ``export_all``.

    Deliberately not hand-built: the whole point of the round trip is that the
    writer and ``indexing._read_scan_provenance`` agree on WHERE the attributes
    live, and a hand-made /Indexing group would keep agreeing with the reader
    after the exporter moved them. Checkpoint shape borrowed from
    ``tests/batch/test_export_step_size_round_trip.py::cubic_checkpoint``.
    """
    import h5py
    from backend.api.services.checkpoint_writer import CheckpointWriter
    from backend.api.services.result_exporter import export_all

    source = tmp_path / "FakeSample.h5oina"
    with h5py.File(source, "w") as f:
        f.create_group("1/EBSD/Data").create_dataset(
            "Band Contrast", data=np.arange(12, dtype=np.uint8))
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("Beam Voltage", data=np.float32(20.0))
        hdr.create_dataset("Tilt Angle", data=np.float32(70.0))

    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="scan-prov",
                     method="spherical", step_size_um=0.5)
    cw.write_phase_result(
        "Al",
        np.full((3, 4), 0.8, dtype=np.float32),
        np.zeros((3, 4, 3), dtype=np.float32),
        {"phase_file": "Al.cif", "ci_mean": 0.8, "duration_sec": 1.0,
         "space_group": 225, "point_group": "m-3m"},
    )
    cw.compute_auto_assignment(confidence_threshold=0.0)

    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    results = export_all(
        source_h5_path=str(source), checkpoint_path=cw.checkpoint_path,
        output_dir=str(out), formats=["h5_light"], include_eds=False,
        scan_provenance=provenance,
    )
    assert results.get("h5_light"), f"light export failed: {results.get('_errors')}"
    return results["h5_light"]


def _import_and_read_back(client, fpath):
    from backend.api.routes import indexing

    resp = client.post("/api/indexing/import-h5", json={"path": str(fpath)})
    assert resp.status_code == 200, resp.text
    rid = resp.json()["result_id"]
    try:
        body = client.get("/api/indexing/results").json()
        return next(e for e in body["results"] if e["id"] == rid)
    finally:
        indexing._result_registry.pop(rid, None)


@pytest.mark.integration
def test_an_imported_result_takes_its_provenance_from_its_file(
        client, h5oina_path, tmp_path):
    """A file that records an offset keeps its own, not the session's."""
    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 7, "col0": 9, "rows": 4, "cols": 4})

    f = _export_light_result(tmp_path, {
        "scan_row_offset": 20, "scan_col_offset": 30, "scan_shape": [64, 80]})
    entry = _import_and_read_back(client, f)
    assert entry["scan_row_offset"] == 20
    assert entry["scan_col_offset"] == 30
    assert list(entry["scan_shape"]) == [64, 80]


@pytest.mark.integration
def test_an_imported_full_scan_result_never_inherits_the_live_crop(
        client, h5oina_path, tmp_path):
    """The bug this guards: a crop is active, the user imports a result whose
    file records no offset, and it must NOT claim the session's crop. Without
    the seeding in import_h5_result, _store_result's setdefault stamps 7 / 9
    onto a result that was never cropped."""
    from backend.api.routes.ebsd_viewer import get_active_crop_window

    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 7, "col0": 9, "rows": 4, "cols": 4})
    assert get_active_crop_window() is not None, "the crop must really be live"

    f = _export_light_result(tmp_path, None)
    entry = _import_and_read_back(client, f)
    assert entry["scan_row_offset"] == 0
    assert entry["scan_col_offset"] == 0
    assert entry["scan_shape"] is None


# --- Two more _store_result callers that must not stamp the live crop.


def _run_stubbed_batch(client, monkeypatch, file_path):
    """Drive the real batch route over ``file_path`` with only the indexer
    stubbed, and return the /results entry for the result it stored.

    Spherical rather than Hough for no deeper reason than that it was the
    branch that worked when this was written: the Hough and Dictionary
    branches could not reach _store_result at all, because ``det_params`` was
    bound only in the spherical branch yet read unconditionally afterwards.
    That pre-existing UnboundLocalError was fixed one commit later (3c3d934),
    so all three branches reach _store_result now; this test simply never
    moved.
    """
    import indexing_controller
    from indexing_controller import IndexingMethod, IndexingResult
    from backend.api.routes import indexing

    def _fake_spherical(h5_path, config, detector_params, selection_mask,
                        **kwargs):
        mask = np.asarray(selection_mask, dtype=bool)
        return IndexingResult(
            xmap=None, selection_mask=mask, original_shape=mask.shape,
            method=IndexingMethod.SPHERICAL,
            confidence_scores=np.full(mask.size, 0.5, dtype=np.float32),
        )

    # The batch route imports its indexers from indexing_controller inside the
    # loop, so that is the module the stubs have to land on.
    for name in ("spherical_index_patterns", "spherical_gpu_index_patterns"):
        monkeypatch.setattr(indexing_controller, name, _fake_spherical)
    monkeypatch.setattr(indexing, "_attach_indexing_metadata",
                        lambda *a, **k: None)

    from pathlib import Path
    from tests.batch_helpers import untouched, wait_for_batch

    before = set(indexing._result_registry)
    with untouched(Path(file_path).parent):
        resp = client.post("/api/indexing/batch/start", json={"datasets": [{
            "file_path": str(file_path), "method": "spherical",
            "selection_mode": "full",
        }], "auto_export": False})
        assert resp.status_code == 200, resp.text
        status = wait_for_batch(client)
    new_ids = [r for r in indexing._result_registry if r not in before]
    assert new_ids, f"the batch stored no result: {status}"
    try:
        body = client.get("/api/indexing/results").json()
        return next(e for e in body["results"] if e["id"] == new_ids[-1])
    finally:
        for rid in new_ids:
            indexing._result_registry.pop(rid, None)


@pytest.mark.integration
def test_a_batch_over_another_file_reports_that_file(
        client, h5oina_path, edax_h5_path, monkeypatch):
    """File A is cropped in the viewer, then a batch runs over file B.

    B was never cropped, so its result must not claim A's window. This holds
    because the batch's own load_ebsd_file re-points the active dataset at B
    (measured: window None) — NOT because anything forces zeros."""
    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 7, "col0": 9, "rows": 4, "cols": 4})

    entry = _run_stubbed_batch(client, monkeypatch, edax_h5_path)
    assert entry["scan_row_offset"] == 0
    assert entry["scan_col_offset"] == 0
    assert entry["scan_shape"] is None


@pytest.mark.integration
def test_a_batch_over_a_cropped_file_keeps_that_crop(
        client, h5oina_path, monkeypatch):
    """The batch queues the file the user already cropped.

    Loading it restores the crop as the active dataset, so the batch really
    indexes the 4x4 cut-out — and must say so. This is the test that fails if
    anyone "hardens" the batch path by forcing the no-crop answer: that would
    erase provenance the result genuinely has."""
    from backend.api.routes import ebsd_viewer as ev

    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 7, "col0": 9, "rows": 4, "cols": 4})

    entry = _run_stubbed_batch(client, monkeypatch, h5oina_path)
    # The batch indexed the crop, not the whole scan — 16 pixels, not 10800.
    assert entry["n_indexed"] == 16, (
        "premise broken: the batch did not index the crop, so this test is "
        "no longer about what it claims"
    )
    assert entry["scan_row_offset"] == 7
    assert entry["scan_col_offset"] == 9
    assert list(entry["scan_shape"]) == list(ev.get_active_crop_window().original_shape)


@pytest.mark.integration
def test_a_refinement_inherits_its_parents_origin(client, h5oina_path):
    """A refinement covers its parent's pixels, so it reports its parent's
    origin — even though the crop was cleared between the two steps."""
    from unittest.mock import MagicMock, patch
    import torch
    from backend.api.routes import indexing
    from backend.api.services import refinement as rf
    from tests.test_refinement_service import _make_fake_indexed_result

    client.post("/api/ebsd/load", json={"path": str(h5oina_path)})
    client.post("/api/ebsd/crop", json={
        "shape": "rect", "row0": 7, "col0": 9, "rows": 4, "cols": 4})

    parent = _make_fake_indexed_result()
    parent.metadata.update(indexing._scan_provenance_fields())
    assert parent.metadata["scan_row_offset"] == 7
    parent_rid = indexing._store_result(parent, "spherical")

    # The user goes back to the whole scan before refining. Read live, the
    # refinement would now report 0 / 0 / None while its parent reports 7 / 9
    # for the very same pixels.
    from backend.api.services import crop_window as cw
    cw.clear_all()
    assert indexing._scan_provenance_fields()["scan_row_offset"] == 0

    grid, renderer = MagicMock(), MagicMock()
    renderer.render_batch.return_value = torch.ones(16, 8, 8, dtype=torch.float32)
    renderer.render.return_value = torch.ones(8, 8, dtype=torch.float32)
    with patch.object(rf, "load_or_get_phase", return_value=grid),          patch.object(rf, "get_renderer", return_value=renderer),          patch.object(rf, "_get_experimental_pattern",
                      return_value=np.ones((8, 8), dtype=np.float32)):
        rf.compute_full_refinement(parent, smoothness_lambda=0.01)

    refined_rid = parent.metadata["refinement"]["refined_result_id"]
    try:
        refined = indexing._result_registry[refined_rid]
        assert refined.metadata["scan_row_offset"] == 7
        assert refined.metadata["scan_col_offset"] == 9
        assert list(refined.metadata["scan_shape"]) ==             list(parent.metadata["scan_shape"])
    finally:
        indexing._result_registry.pop(refined_rid, None)
        indexing._result_registry.pop(parent_rid, None)


# --- The interactive export route must WRITE the provenance it stored.
#
# `result_exporter._write_scan_provenance` exists and is called from
# `export_result_h5` / `export_light_h5`, but the only caller of those is
# `export_all` from the batch manager, which must pass None (its files are not
# the active dataset). The interactive route duplicates the /Indexing writers
# inline, by design, and neither branch called it — so crop -> index -> "Save
# result" -> re-import came back NO_SCAN_OFFSET: the file claimed the result
# covered the whole scan.


@pytest.fixture
def cropped_result(monkeypatch):
    """A stored result that records a 4x4 window at (7, 9) of a 90x120 scan.

    The provenance is set on the RESULT, the way ``_store_result`` sets it at
    run time — not read live at export time. That is the distinction the export
    has to honour: a result indexed before the user cropped must not be
    exported claiming the window that is active now.
    """
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from indexing_controller import IndexingMethod, IndexingResult
    from backend.api.routes import indexing as indexing_mod

    n_rows, n_cols = 4, 4
    n_pix = n_rows * n_cols
    rotations = Rotation.from_euler(np.deg2rad(np.array(
        [[i * 7.0, 20.0, 30.0] for i in range(n_pix)], dtype=np.float64)))
    xmap = CrystalMap(
        rotations=rotations,
        phase_id=np.zeros(n_pix, dtype=np.int32),
        x=np.tile(np.arange(n_cols, dtype=np.float32), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=np.float32), n_cols),
        phase_list=PhaseList(phases=[
            Phase(name="Al", space_group=225, point_group="m-3m")]),
    )
    result = IndexingResult(
        xmap=xmap,
        selection_mask=np.ones(n_pix, dtype=bool),
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL,
        confidence_scores=np.full(n_pix, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.5, "scan_row_offset": 7,
                  "scan_col_offset": 9, "scan_shape": [90, 120]},
    )
    rid = "test_crop_export_provenance"
    monkeypatch.setitem(indexing_mod._result_registry, rid, result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", rid)
    monkeypatch.setattr("backend.api.routes.ebsd_viewer._ebsd_file_path",
                        None, raising=False)
    return result


@pytest.mark.parametrize("fmt", ["h5_light", "h5"])
def test_the_export_route_writes_the_stored_provenance(
        client, cropped_result, tmp_path, fmt):
    import h5py

    out = tmp_path / f"result_{fmt}.h5"
    resp = client.post("/api/indexing/export",
                       json={"format": fmt, "filename": str(out)})
    assert resp.status_code == 200, resp.text
    with h5py.File(out, "r") as f:
        idx = f["Indexing"]
        assert int(idx.attrs["scan_row_offset"]) == 7
        assert int(idx.attrs["scan_col_offset"]) == 9
        assert list(idx.attrs["scan_shape"]) == [90, 120]


@pytest.mark.parametrize("fmt", ["h5_light", "h5"])
def test_an_uncropped_result_still_exports_exactly_as_before(
        client, cropped_result, tmp_path, fmt, monkeypatch):
    """The no-crop path must be untouched: zeros, and NO scan_shape attribute
    (h5py cannot store None, and "no original shape" is the absent attr)."""
    import h5py

    cropped_result.metadata.update({"scan_row_offset": 0, "scan_col_offset": 0,
                                    "scan_shape": None})
    out = tmp_path / f"plain_{fmt}.h5"
    assert client.post("/api/indexing/export",
                       json={"format": fmt,
                             "filename": str(out)}).status_code == 200
    with h5py.File(out, "r") as f:
        idx = f["Indexing"]
        assert int(idx.attrs["scan_row_offset"]) == 0
        assert int(idx.attrs["scan_col_offset"]) == 0
        assert "scan_shape" not in idx.attrs


def test_export_then_reimport_round_trips_the_window(
        client, cropped_result, tmp_path):
    """The whole point: what the writer put in is what /results reads back.

    Hand-checking the attributes is not enough — the reader
    (``indexing._read_scan_provenance``) has to look where the writer wrote.
    """
    out = tmp_path / "roundtrip_light.h5"
    assert client.post("/api/indexing/export",
                       json={"format": "h5_light",
                             "filename": str(out)}).status_code == 200

    entry = _import_and_read_back(client, out)
    assert entry["scan_row_offset"] == 7
    assert entry["scan_col_offset"] == 9
    assert list(entry["scan_shape"]) == [90, 120]
