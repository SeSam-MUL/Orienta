"""Two files with one stem in two folders: a batch keeps their calibrations and outputs apart.

The calibration store is keyed by file stem. A batch-v2 run over ``A/Scan5.up1`` (no PC)
and ``B/Scan5.up1`` (PC in its .osc) used ONE entry for both: A was indexed at B's PC
(CI 0.18 instead of the refusal ``pc_missing``), both wrote ``result_Scan5.*`` into one
folder (B's PC in the .ang), and a ``.part`` fragment of the failed light export stayed.
Within a batch the files whose stems collide are now named by their stem plus a hash of
their canonical path; the outputs carry it (and the batch says so); a writer that fails
leaves no ``.part``. The legacy batch clears and registers per file, shown here too.
Real routes, real loader, real checkpoints and ``.ang`` writer on synthetic UP1 files; the
one seam is the indexer of a job (``run_single_indexing_job``), which calls the REAL detector
lookup of the batch (``_get_detector_for_file``) and returns small maps.
"""
import struct
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.calibration_store import CalibrationStore

SIDE = 24
OSC_A = (0.55, 0.50, 0.70)
OSC_B = (0.60, 0.45, 0.75)


def _write_up1(path, seed=3, n_side=4):
    rng = np.random.default_rng(seed)
    pats = rng.integers(0, 250, (n_side * n_side, SIDE, SIDE)).astype("u1")
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, SIDE, SIDE, 16))
        f.write(pats.tobytes())
    return str(path)


def _write_osc(path, pc_tsl, n_side=4):
    from edax_up1 import _OSC_PC_OFFSET
    raw = bytearray(_OSC_PC_OFFSET + 64)
    struct.pack_into("<3I", raw, 16, n_side - 1, n_side - 1, n_side * n_side)
    struct.pack_into("<3f", raw, _OSC_PC_OFFSET, *pc_tsl)
    with open(path, "wb") as f:
        f.write(bytes(raw))


NI_CIF = (
    "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
    "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
    "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n")


@pytest.fixture
def world(monkeypatch, tmp_path):
    from backend.api.routes import batch_v2 as B
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services import batch_manager as M
    store = CalibrationStore()
    for target in ("backend.api.services.calibration_store.calibration_store",):
        monkeypatch.setattr(target, store)
    monkeypatch.setattr(B, "calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr(M, "_batch_names", {})
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    mgr = M.BatchManager(db_path=str(tmp_path / "jobs.db"))
    monkeypatch.setattr(B, "_get_manager", lambda: mgr)
    return store, mgr, tmp_path


def _fake_job(used):
    """The indexer of a job: it asks the batch's REAL detector lookup, as the real one does."""
    from backend.api.services import batch_manager as M

    def job(signal, file_path, phase_name, phase_path, method, grid_shape, job_config=None,
            progress_callback=None, scan_geometry=None):
        det = M._get_detector_for_file(file_path, signal)
        used[file_path] = np.asarray(det.pc, dtype=float).reshape(-1, 3).mean(axis=0)
        ci = np.full(grid_shape, 0.5)
        ori = np.zeros(tuple(grid_shape) + (3,))
        return ci, ori, {"ci_mean": 0.5, "ci_median": 0.5, "duration_sec": 0.1, "step_size_um": 0.5,
                         "space_group": 225, "point_group": "m-3m",
                         "lattice_constants": [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]}

    return job


def _run_batch(client, mgr, files, tmp_path, formats):
    cif = tmp_path / "Ni.cif"
    cif.write_text(NI_CIF, encoding="utf-8")
    export_dir = tmp_path / "out"
    resp = client.post("/api/batch-v2/create", json={
        "files": files, "phases": [{"name": "Ni", "path": str(cif), "method": "hough"}],
        "config": {"auto_export": True, "export_dir": str(export_dir), "export_formats": formats}})
    assert resp.status_code == 200, resp.text
    batch_id = resp.json()["batch_id"]
    resp = client.post(f"/api/batch-v2/{batch_id}/start")
    assert resp.status_code == 200, resp.text
    for _ in range(300):
        time.sleep(0.2)
        if mgr.get_status(batch_id)["status"] in ("completed", "paused"):
            break
    return batch_id, export_dir


def _ang_pc(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return [float([ln for ln in lines if ln.startswith(f"# {k}")][0].split()[-1])
            for k in ("x-star", "y-star", "z-star")]


def test_a_file_without_a_pc_is_refused_even_when_a_file_of_the_same_stem_has_one(world, monkeypatch):
    """Not quick-loaded: A is decided when the job loads it, from its own header."""
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)                    # no .osc: no PC
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    used = {}
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job(used))
    client = TestClient(app)
    assert client.post("/api/batch-v2/quick-load", json={"file_path": b}).status_code == 200   # only B is known
    batch_id, _ = _run_batch(client, mgr, [a, b], tmp, ["ang"])
    jobs = {j["file_path"]: j for j in mgr.get_jobs(batch_id)}
    assert jobs[a]["status"] == "failed" and jobs[a]["error_code"] == "pc_missing"
    assert jobs[b]["status"] == "done"
    assert a not in used                                                # A was never indexed at B's PC
    # B at its own PC: by hand, TSL (0.60, 0.45, 0.75) -> Bruker x, 1 - y, z (square pattern)
    np.testing.assert_allclose(used[b], [OSC_B[0], 1 - OSC_B[1], OSC_B[2]], atol=1e-5)


def test_quick_loaded_files_each_have_their_own_entry_and_the_batch_refuses_the_one_without_a_pc(world, monkeypatch):
    """Both quick-loaded: A (no PC) is registered as missing under its own name, and the
    start of the batch is refused for it with the coded 400 - never A at B's PC."""
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    used = {}
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job(used))
    client = TestClient(app)
    names = {}
    for f in (a, b):
        resp = client.post("/api/batch-v2/quick-load", json={"file_path": f})
        assert resp.status_code == 200
        names[f] = resp.json()["dataset_name"]
    assert names[a] == "Scan5" and names[b].startswith("Scan5__")
    assert store.get_entry(names[a]).pc_source == "missing" and store.get_entry(names[b]).pc_source == "header"
    cif = tmp / "Ni.cif"
    cif.write_text(NI_CIF, encoding="utf-8")
    resp = client.post("/api/batch-v2/create", json={
        "files": [a, b], "phases": [{"name": "Ni", "path": str(cif), "method": "hough"}],
        "config": {"auto_export": False}})
    resp = client.post(f"/api/batch-v2/{resp.json()['batch_id']}/start")
    assert resp.status_code == 400 and resp.headers["x-orienta-code"] == "pc_missing"
    assert used == {}


def test_two_files_with_one_stem_export_under_different_names(world, monkeypatch):
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    _write_osc(tmp / "A" / "Scan5.osc", OSC_A)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    used = {}
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job(used))
    client = TestClient(app)
    batch_id, out = _run_batch(client, mgr, [a, b], tmp, ["ang"])
    assert all(j["status"] == "done" for j in mgr.get_jobs(batch_id))
    # each file at its own PC
    np.testing.assert_allclose(used[a], [OSC_A[0], 1 - OSC_A[1], OSC_A[2]], atol=1e-5)
    np.testing.assert_allclose(used[b], [OSC_B[0], 1 - OSC_B[1], OSC_B[2]], atol=1e-5)
    # two .ang files, none overwritten, each stating its own file's PC (TSL, by hand)
    angs = sorted(p for p in Path(out).glob("result_Scan5__*.ang"))
    assert len(angs) == 2 and not list(Path(out).glob("result_Scan5.ang"))
    pcs = sorted(tuple(round(v, 4) for v in _ang_pc(p)) for p in angs)
    assert pcs == sorted([OSC_A, OSC_B])
    # and the batch says which suffix belongs to which file, deterministically
    listing = client.get(f"/api/batch-v2/{batch_id}/exports").json()["exports"]
    by_file = {e["file_path"]: e for e in listing}
    assert set(by_file) == {a, b}
    suffixes = {e["name_suffix"] for e in listing}
    assert len(suffixes) == 2 and all(s.startswith("__") and len(s) == 10 for s in suffixes)
    for e in listing:
        assert Path(e["outputs"]["ang"]).name == f"result_Scan5{e['name_suffix']}.ang"
    assert M.assign_batch_dataset_names([a, b])[M._canonical_file(a)][1] == by_file[a]["name_suffix"]


def test_a_batch_without_a_collision_keeps_the_plain_names(world, monkeypatch):
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    c = _write_up1(tmp / "B" / "Other.up1", seed=2)
    _write_osc(tmp / "A" / "Scan5.osc", OSC_A)
    _write_osc(tmp / "B" / "Other.osc", OSC_B)
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job({}))
    batch_id, out = _run_batch(TestClient(app), mgr, [a, c], tmp, ["ang"])
    assert sorted(p.name for p in Path(out).glob("*.ang")) == ["result_Other.ang", "result_Scan5.ang"]


def test_a_failed_writer_leaves_no_part_file(tmp_path):
    """The light export of a source that is not HDF5 (a UP1) starts writing its file and
    fails half way; the fragment is removed, not left behind as ``...h5.part``."""
    from backend.api.services.checkpoint_writer import CheckpointWriter
    from backend.api.services.result_exporter import export_all
    src = tmp_path / "scan.up1"
    src.write_bytes(b"not an hdf5 file")
    cw = CheckpointWriter(str(src))
    cw.init_metadata(grid_shape=(4, 4), batch_id="b", method="hough", step_size_um=0.5)
    cw.write_phase_result("Ni", np.full((4, 4), 0.5), np.zeros((4, 4, 3)),
                          {"ci_mean": 0.5, "ci_median": 0.5, "duration_sec": 0.1,
                           "space_group": 225, "point_group": "m-3m",
                           "lattice_constants": [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]})
    out = tmp_path / "out"
    results = export_all(str(src), cw.checkpoint_path, str(out), formats=["h5_light"], step_size=0.5)
    assert results["h5_light"] is None and "h5_light" in results["_errors"]
    assert not list(out.glob("*.part")) and not list(tmp_path.glob("*.part"))


def test_the_legacy_batch_keeps_two_same_stem_files_apart(world, monkeypatch):
    """Not affected: it clears and registers the calibration per file. Shown, both orders."""
    import indexing_controller
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from tests.test_result_keeps_its_run_pc import _fake_hough as fake_hough
    store, mgr, tmp = world
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(indexing_controller, "hough_index_patterns", fake_hough)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)                    # no PC
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    cif = tmp / "Ni.cif"
    cif.write_text(NI_CIF, encoding="utf-8")
    saved = dict(I._batch_state)
    try:
        for order in ([b, a], [a, b]):
            store.clear()                                               # a fresh session each time:
            ev._raw_signals.clear()                                     # nothing loaded, nothing registered
            ev._ebsd_file_path = None
            resp = TestClient(app).post("/api/indexing/batch/start", json={"datasets": [
                {"file_path": f, "method": "hough", "cif_paths": [str(cif)]} for f in order],
                "auto_export": False})
            assert resp.status_code == 200, resp.text
            results = {r["file_path"]: r for r in I._batch_state["results"][-2:]}
            assert results[a]["error_code"] == "pc_missing" and results[a]["n_indexed"] == 0
            assert results[b]["error"] is None and results[b]["n_indexed"] > 0
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved)


def test_a_batch_file_does_not_take_over_the_entry_of_the_viewers_file_of_the_same_stem(world, monkeypatch):
    """The viewer holds A/Scan5 with a typed PC; a batch over B/Scan5 uses B's own PC and
    leaves the viewer's entry (and its typed PC) alone."""
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    client = TestClient(app)
    assert client.post("/api/ebsd/load", json={"path": a}).status_code == 200      # A is the viewer's file
    store.update_pc("Scan5", [0.48, 0.52, 0.60], source="manual")
    used = {}
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job(used))
    batch_id, _ = _run_batch(client, mgr, [b], tmp, ["ang"])
    assert [j["status"] for j in mgr.get_jobs(batch_id)] == ["done"]
    np.testing.assert_allclose(used[b], [OSC_B[0], 1 - OSC_B[1], OSC_B[2]], atol=1e-5)   # B's own, not 0.48...
    entry = store.get_entry("Scan5")                                                # the viewer's entry is intact
    assert entry.pc_source == "manual"
    np.testing.assert_allclose(store.get_pc("Scan5"), [0.48, 0.52, 0.60], atol=1e-12)
    assert M.dataset_name_for(b).startswith("Scan5__")


def test_a_second_batch_over_another_file_of_the_same_stem_does_not_reuse_the_first_ones_entry(world, monkeypatch):
    """Run 1 over A/Scan5 (no PC, refused) leaves its entry; run 2 over B/Scan5 alone is neither
    refused at the start because of A's missing entry nor indexed at it."""
    store, mgr, tmp = world
    from backend.api.services import batch_manager as M
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    used = {}
    monkeypatch.setattr(M, "run_single_indexing_job", _fake_job(used))
    client = TestClient(app)
    batch_a, _ = _run_batch(client, mgr, [a], tmp, ["ang"])
    assert [j["status"] for j in mgr.get_jobs(batch_a)] == ["failed"]
    assert store.get_entry("Scan5").pc_source == "missing" and store.get_entry("Scan5").file_key
    batch_b, _ = _run_batch(client, mgr, [b], tmp, ["ang"])
    assert [j["status"] for j in mgr.get_jobs(batch_b)] == ["done"]
    np.testing.assert_allclose(used[b], [OSC_B[0], 1 - OSC_B[1], OSC_B[2]], atol=1e-5)


def test_the_legacy_request_guard_does_not_refuse_a_file_because_of_another_files_entry(world, monkeypatch):
    import indexing_controller
    from backend.api.routes import indexing as I
    from tests.test_result_keeps_its_run_pc import _fake_hough as fake_hough
    store, mgr, tmp = world
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(indexing_controller, "hough_index_patterns", fake_hough)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    assert TestClient(app).post("/api/batch-v2/quick-load", json={"file_path": a}).status_code == 200   # entry: A, missing
    assert store.get_entry("Scan5").pc_source == "missing"
    cif = tmp / "Ni.cif"
    cif.write_text(NI_CIF, encoding="utf-8")
    saved = dict(I._batch_state)
    try:
        resp = TestClient(app).post("/api/indexing/batch/start", json={"datasets": [
            {"file_path": b, "method": "hough", "cif_paths": [str(cif)]}], "auto_export": False})
        assert resp.status_code == 200, resp.text                      # not refused for A's entry
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved)


def test_typing_a_pc_for_one_file_of_a_shared_stem_does_not_reach_the_others_entry(world):
    """The Batch page's "set PC manually" dialog, addressed by file path like copy-pc:
    two files of the same stem are two different entries, and only the one the dialog
    was open on gets the typed PC."""
    store, mgr, tmp = world
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)                    # no .osc: no PC
    b = _write_up1(tmp / "B" / "Scan5.up1", seed=2)
    _write_osc(tmp / "B" / "Scan5.osc", OSC_B)
    client = TestClient(app)
    assert client.post("/api/batch-v2/quick-load", json={"file_path": a}).status_code == 200
    assert client.post("/api/batch-v2/quick-load", json={"file_path": b}).status_code == 200
    typed_pc = [0.51, 0.34, 0.84]
    resp = client.post("/api/batch-v2/set-pc", json={"file_path": a, "pc": typed_pc})
    assert resp.status_code == 200, resp.text
    from backend.api.services.batch_manager import default_variant_entry_of_file
    key_a = default_variant_entry_of_file(a, store)
    key_b = default_variant_entry_of_file(b, store)
    assert key_a != key_b                                               # two entries, same stem
    np.testing.assert_allclose(store.get_pc(key_a), typed_pc, atol=1e-12)
    assert store.get_entry(key_a).pc_source == "manual"
    np.testing.assert_allclose(store.get_pc(key_b), [OSC_B[0], 1 - OSC_B[1], OSC_B[2]], atol=1e-5)
    assert store.get_entry(key_b).pc_source == "header"                 # B's own PC is untouched

    # The other direction: typing a PC for B (the one that did NOT get the plain stem name)
    # must reach B's own entry, not fall back onto A's just because A holds the bare "Scan5".
    typed_pc_b = [0.22, 0.33, 0.77]
    resp = client.post("/api/batch-v2/set-pc", json={"file_path": b, "pc": typed_pc_b})
    assert resp.status_code == 200, resp.text
    np.testing.assert_allclose(store.get_pc(key_b), typed_pc_b, atol=1e-12)
    assert store.get_entry(key_b).pc_source == "manual"
    np.testing.assert_allclose(store.get_pc(key_a), typed_pc, atol=1e-12)   # A's PC from above, untouched


def test_set_pc_shifts_a_grid_calibrated_map_instead_of_leaving_it_on_its_old_mean(world):
    """A file with a per-pixel PC map (a grid calibration) that then gets a typed PC must have
    the map SHIFT (mean = the typed PC, shape kept) — the same choice the PC-Refinement page's
    own manual-PC endpoint makes. Leaving the map on its old mean would mean indexing keeps
    reading it while the page says the typed PC is in use."""
    store, _mgr, tmp = world
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    _write_osc(tmp / "A" / "Scan5.osc", OSC_A)
    client = TestClient(app)
    assert client.post("/api/batch-v2/quick-load", json={"file_path": a}).status_code == 200
    from backend.api.services.batch_manager import default_variant_entry_of_file
    key = default_variant_entry_of_file(a, store)
    rows, cols = 3, 4
    plane = np.stack(np.broadcast_arrays(
        0.50 + 1e-3 * np.arange(cols), (0.40 + 2e-3 * np.arange(rows))[:, None], 0.60), axis=-1)
    store.update_pc_map(key, plane)

    typed_pc = [0.5321, 0.3987, 0.7123]
    resp = client.post("/api/batch-v2/set-pc", json={"file_path": a, "pc": typed_pc})
    assert resp.status_code == 200, resp.text

    entry = store.get_entry(key)
    assert entry.pc_map is not None                                     # shifted, not dropped
    map_flat = np.asarray(entry.pc_map).reshape(-1, 3)
    plane_flat = plane.reshape(-1, 3)
    np.testing.assert_allclose(map_flat.mean(axis=0), typed_pc, atol=1e-12)
    np.testing.assert_allclose(map_flat - np.asarray(typed_pc),
                               plane_flat - plane_flat.mean(axis=0), atol=1e-12)   # shape kept


def test_quick_load_names_the_files_own_pc_convention(world):
    """For the Batch page to preselect the right convention in its PC dialog (an EDAX .up1
    writes TSL numbers, not Bruker's) instead of assuming Bruker for every vendor."""
    _store, _mgr, tmp = world
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    resp = TestClient(app).post("/api/batch-v2/quick-load", json={"file_path": a})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["convention"] == "tsl" and body["convention_known"] is True


def test_set_pc_refuses_a_file_that_was_never_quick_loaded(world):
    """A real error, not a quiet checkmark: the file must be in the store first."""
    store, mgr, tmp = world
    a = _write_up1(tmp / "A" / "Scan5.up1", seed=1)
    resp = TestClient(app).post("/api/batch-v2/set-pc", json={"file_path": a, "pc": [0.5, 0.5, 0.5]})
    assert resp.status_code == 404
