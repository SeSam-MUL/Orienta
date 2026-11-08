"""The ``_indexed.h5`` of ``POST /api/indexing/batch/start`` is a file the app can read back.

The inline writer used to store flat arrays, the raw orix phase ids, a dataset ``confidence`` that no reader
asked for and no ``Phases`` group, so ``_load_kikuchipy_rich_h5`` could not rebuild the map (a square scan
as little as a hexagonal one), and a hexagonal scan was not exported at all. It now writes the grid-shaped
arrays of the light export, and a hexagonal scan carries its ``/Indexing`` stamp.

The route, its loader, the exporter and the reader are real; the indexer is a stand-in that returns an
orix map built here from numbers written out below.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from tests.test_hex_guard import client, fresh_backend  # noqa: F401  (fixtures)

N_ROWS, N_COLS = 3, 5

#: By hand, for the flat point f of the 3 x 5 scan: phase 0 ("Trig", space group 162) when f % 3 == 0,
#: phase 1 ("Al", 225) when f % 3 == 1, not indexed (-1) when f % 3 == 2. On disk the ids are 1-based and 0
#: is unindexed: f % 3 == 0 -> 1, 1 -> 2, 2 -> 0.
DISK_PHASE = [1, 2, 0]
#: CI of point f is f / 20 (0.0 .. 0.7); Euler angles (0.05 f, 0.4, 0.6) rad.


def _euler(flat):
    return np.array([[0.05 * f, 0.4, 0.6] for f in flat])


def _stub_for(mask):
    """The result a spherical run over ``mask`` returns: only the selected points have rows."""
    from indexing_controller import IndexingMethod, IndexingResult
    flat = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    pid = np.array([[0, 1, -1][f % 3] for f in flat])
    xmap = CrystalMap(
        rotations=Rotation.from_euler(_euler(flat)), phase_id=pid,
        x=(flat % N_COLS) * 0.5, y=(flat // N_COLS) * 0.5,
        phase_list=PhaseList([Phase(name="Trig", space_group=162), Phase(name="Al", space_group=225)]))
    return IndexingResult(
        xmap=xmap, selection_mask=np.asarray(mask, dtype=bool), original_shape=(N_ROWS, N_COLS),
        method=IndexingMethod.SPHERICAL, confidence_scores=(flat / 20.0).astype(np.float32))


def _legacy_batch(client, monkeypatch, tmp_path, *, hex_flag, dataset=None, earlier_export=None):
    """Drive the real route over a synthetic UP1 file with auto-export on; returns (entry, export folder).
    ``earlier_export``: bytes of a file already standing under the name the export will take."""
    import indexing_controller
    from backend.api.routes import indexing as I
    from tests.apex_helpers import make_apex_h5, make_up, measurement_spec

    folder = tmp_path / ("hex" if hex_flag else "square")
    folder.mkdir()
    make_apex_h5(folder / "s.h5", [measurement_spec(
        grid="HexGrid" if hex_flag else "SqrGrid", n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5)])
    up = make_up(folder / "scan.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS, hex_flag=1 if hex_flag else 0)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    if earlier_export is not None:
        (export_dir / "scan_indexed.h5").write_bytes(earlier_export)

    def stub_spherical(h5_path, config, detector_params, selection_mask, **kwargs):
        return _stub_for(selection_mask)

    for name in ("spherical_index_patterns", "spherical_gpu_index_patterns"):
        monkeypatch.setattr(indexing_controller, name, stub_spherical)
    monkeypatch.setattr(I, "_attach_indexing_metadata", lambda *a, **k: None)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    saved_batch, saved_results = dict(I._batch_state), dict(I._result_registry)
    try:
        resp = client.post("/api/indexing/batch/start", json={
            "datasets": [{"file_path": str(up), "method": "spherical", **(dataset or {"selection_mode": "full"})}],
            "auto_export": True, "export_dir": str(export_dir), "cleanup_after_export": False})
        assert resp.status_code == 200, resp.text
        entry = dict(I._batch_state["results"][-1])
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved_batch)
        I._result_registry.clear()
        I._result_registry.update(saved_results)
    return entry, export_dir


def _reload(entry):
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5
    assert entry["error"] is None, entry
    assert entry["export_path"] is not None, entry
    return entry["export_path"], _load_kikuchipy_rich_h5(entry["export_path"])


def _angle_between(got, want):
    return (got * ~want).angle


def _check_full(path, xmap):
    # hand values for all 15 points
    flat = np.arange(N_ROWS * N_COLS)
    assert xmap.size == 15
    assert xmap.phase_id.tolist() == [[0, 1, -1][f % 3] for f in flat]          # orix: 0-based, -1 unindexed
    np.testing.assert_allclose(xmap.prop["ci"], flat / 20.0, atol=1e-6)
    assert np.all(_angle_between(xmap.rotations, Rotation.from_euler(_euler(flat))) < 1e-5)
    from crystal_symmetry import frame_symmetry
    assert [p.name for i, p in xmap.phases if i >= 0] == ["Trig", "Al"]
    assert xmap.phases["Trig"].point_group is frame_symmetry("-31m")              # space group 162
    with h5py.File(path, "r") as f:
        idx = f["Indexing"]
        assert idx["phase_id"].shape == (N_ROWS, N_COLS) and idx["phase_id"].dtype == np.uint8
        assert idx["phase_id"][()].ravel().tolist() == [DISK_PHASE[f % 3] for f in flat]
        assert idx["euler_angles"].shape == (N_ROWS, N_COLS, 3)
        assert idx["confidence_index"].shape == (N_ROWS, N_COLS)
        assert "confidence" not in idx                                            # the name nobody read
        assert list(idx.attrs["grid_shape"]) == [N_ROWS, N_COLS]
        assert idx["Phases/1"].attrs["space_group"] == 162 and idx["Phases/2"].attrs["space_group"] == 225


def test_legacy_batch_square_export_reloads(client, monkeypatch, tmp_path):
    entry, _ = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=False)
    path, xmap = _reload(entry)
    _check_full(path, xmap)
    with h5py.File(path, "r") as f:
        assert "grid_type" not in f["Indexing"].attrs                              # a square scan has no hex stamp
        assert f["Indexing"].attrs["step_size_um"] == pytest.approx(0.5)
    np.testing.assert_allclose(xmap.x[:5], [0.0, 0.5, 1.0, 1.5, 2.0], atol=1e-6)    # column * step


def test_legacy_batch_hex_export_reloads_hex(client, monkeypatch, tmp_path):
    entry, _ = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=True)
    assert entry["export_error"] is None
    path, xmap = _reload(entry)
    _check_full(path, xmap)
    from backend.api.routes.analysis import read_index_stamp
    stamp = read_index_stamp(path)
    assert stamp is not None and stamp.kind == "hex" and stamp.grid_shape == (N_ROWS, N_COLS)
    assert stamp.step_x == pytest.approx(0.5) and stamp.step_y == pytest.approx(0.5 * 3 ** 0.5 / 2)
    with h5py.File(path, "r") as f:
        idx = f["Indexing"]
        # APEX states HexGrid with the odd rows half a step to the right (+0.5): the sign is written down
        # here, not taken from the file, and so are the positions below.
        assert float(idx.attrs["odd_row_shift"]) == 0.5
        x = idx["X"][()]
        # row 0 at column * 0.5; the odd row 1 half a step (0.25 um) to the right: 0.25, 0.75, ... 2.25;
        # y = row * 0.5 * sqrt(3) / 2
        np.testing.assert_allclose(x[0], [0.0, 0.5, 1.0, 1.5, 2.0], atol=1e-9)
        np.testing.assert_allclose(x[1], [0.25, 0.75, 1.25, 1.75, 2.25], atol=1e-9)
        np.testing.assert_allclose(x[2], [0.0, 0.5, 1.0, 1.5, 2.0], atol=1e-9)
        np.testing.assert_allclose(idx["Y"][()][:, 0], [0.0, 0.4330127, 0.8660254], atol=1e-6)


def test_legacy_batch_region_item_reloads(client, monkeypatch, tmp_path):
    """Rows 1-2, columns 1-3 (end exclusive) = the flat points 6, 7, 8, 11, 12, 13. The other nine points
    are unindexed on disk: phase 0, CI 0."""
    entry, _ = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=False, dataset={
        "selection_mode": "region", "row_start": 1, "row_end": 3, "col_start": 1, "col_end": 4})
    assert entry["n_indexed"] == 6
    path, xmap = _reload(entry)
    inside = [6, 7, 8, 11, 12, 13]
    assert xmap.size == 15
    want_pid = [[0, 1, -1][f % 3] if f in inside else -1 for f in range(15)]
    assert xmap.phase_id.tolist() == want_pid
    np.testing.assert_allclose(xmap.prop["ci"], [f / 20.0 if f in inside else 0.0 for f in range(15)], atol=1e-6)
    got = Rotation(xmap.rotations.data[inside])
    assert np.all(_angle_between(got, Rotation.from_euler(_euler(inside))) < 1e-5)
    with h5py.File(path, "r") as f:
        assert f["Indexing/selection_mask"][()].sum() == 6
        assert f["Indexing/phase_id"][()].ravel().tolist() == [
            DISK_PHASE[g % 3] if g in inside else 0 for g in range(15)]


def test_legacy_batch_export_keeps_the_detector_and_the_scan(client, monkeypatch, tmp_path):
    """What the earlier writer stated stays: a /Detector group, the scan it came from. (The caveats
    stamp is pinned by tests/test_hough_trigonal_caveat.py.)"""
    entry, _ = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=False)
    path, _xmap = _reload(entry)
    with h5py.File(path, "r") as f:
        assert "Detector" in f and f["Indexing"].attrs["method"] == "spherical"
        assert f["Indexing"].attrs["n_indexed"] == 15
        assert Path(f["Scan"].attrs["source_file"]).name == "scan.up1"


# ----------------------------------------------------------------------- a mask item

#: An irregular selection (no rectangle): the flat points 1, 4, 5, 9 and 14 of the 3 x 5 scan.
MASKED = [1, 4, 5, 9, 14]


def _check_masked(path, xmap, inside):
    """By hand, as the region test: the points outside the selection are unindexed on disk (phase 0, CI 0);
    inside it the values of point f are those of ``_stub_for`` (phase [0, 1, -1][f % 3], CI f / 20)."""
    assert xmap.size == 15
    assert xmap.phase_id.tolist() == [[0, 1, -1][f % 3] if f in inside else -1 for f in range(15)]
    np.testing.assert_allclose(xmap.prop["ci"], [f / 20.0 if f in inside else 0.0 for f in range(15)], atol=1e-6)
    got = Rotation(xmap.rotations.data[inside])
    assert np.all(_angle_between(got, Rotation.from_euler(_euler(inside))) < 1e-5)
    with h5py.File(path, "r") as f:
        assert f["Indexing/selection_mask"][()].ravel().tolist() == [1 if g in inside else 0 for g in range(15)]
        assert f["Indexing/phase_id"][()].ravel().tolist() == [
            DISK_PHASE[g % 3] if g in inside else 0 for g in range(15)]


def test_the_writer_places_the_rows_of_an_irregular_selection_by_its_mask(tmp_path):
    """A result that ran on a mask (a lasso crop leaves one) has one row per selected point; they go to their
    grid points and every other point is unindexed."""
    from kikuchipy.detectors import EBSDDetector

    from backend.api.routes import analysis, indexing as I
    mask = np.zeros(N_ROWS * N_COLS, dtype=bool)
    mask[MASKED] = True
    result = _stub_for(mask.reshape(N_ROWS, N_COLS))
    assert result.xmap.size == len(MASKED)                      # rows only for the selected points
    path = tmp_path / "masked_indexed.h5"
    I._write_batch_indexed_h5(
        str(path), result, dataset_name="masked", detector=EBSDDetector(shape=(8, 8), pc=[0.5, 0.5, 0.5]),
        source_file="scan.up1", n_indexed=len(MASKED))
    _check_masked(str(path), analysis._load_kikuchipy_rich_h5(str(path)), MASKED)


def test_legacy_batch_mask_item_without_a_mask_fails_that_item_and_writes_nothing(client, monkeypatch, tmp_path):
    """``selection_mode: "mask"`` needs the mask of a chemistry filter and the legacy batch request carries
    none: the item fails with the reason, before anything is indexed or written."""
    entry, export_dir = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=False,
                                      dataset={"selection_mode": "mask"})
    assert "Chemistry mask must be provided" in entry["error"]
    assert entry.get("export_path") is None
    assert list(export_dir.iterdir()) == []


# ---------------------------------------------------------------- a failed write leaves no file

def test_a_failed_write_leaves_neither_a_truncated_file_nor_a_part_file(client, monkeypatch, tmp_path):
    """The writer builds the file under a ``.part`` name and renames it when it is complete: a failure half
    way leaves nothing under the final name, and the entry says the file is missing."""
    from backend.api.routes import indexing as I

    def explode(*a, **k):
        raise OSError("the disk is full")

    monkeypatch.setattr(I, "_write_detector_facts", explode)      # runs after the arrays are written
    entry, export_dir = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=False)
    assert entry.get("export_path") is None and "the disk is full" in entry["export_error"]
    assert list(export_dir.iterdir()) == []


def test_a_phase_table_that_cannot_be_written_fails_the_file_and_is_the_export_error(client, monkeypatch, tmp_path):
    """The table used to be skipped with a log line, and the file reloaded triclinic or with too few phases."""
    from backend.api.services import ang_export

    def explode(*a, **k):
        raise ValueError("no lattice for this phase")

    monkeypatch.setattr(ang_export, "phase_table_attrs", explode)
    entry, export_dir = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=False)
    assert entry.get("export_path") is None
    assert "phase table" in entry["export_error"] and "no lattice for this phase" in entry["export_error"]
    assert list(export_dir.iterdir()) == []


def test_a_failed_write_keeps_the_file_an_earlier_export_left(client, monkeypatch, tmp_path):
    from backend.api.routes import indexing as I

    def explode(*a, **k):
        raise OSError("the disk is full")

    monkeypatch.setattr(I, "_write_detector_facts", explode)
    entry, export_dir = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=False, earlier_export=b"EARLIER RESULT")
    assert "export_error" in entry and entry.get("export_path") is None
    assert [p.name for p in export_dir.iterdir()] == ["scan_indexed.h5"]
    assert (export_dir / "scan_indexed.h5").read_bytes() == b"EARLIER RESULT"


def test_a_good_write_replaces_the_earlier_file_and_leaves_no_part_file(client, monkeypatch, tmp_path):
    entry, export_dir = _legacy_batch(client, monkeypatch, tmp_path, hex_flag=False, earlier_export=b"EARLIER RESULT")
    assert [p.name for p in export_dir.iterdir()] == ["scan_indexed.h5"]
    path, xmap = _reload(entry)
    _check_full(path, xmap)
