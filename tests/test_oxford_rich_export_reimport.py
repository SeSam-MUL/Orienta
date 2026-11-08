"""Re-importing the rich export of an Oxford scan keeps the geometry and says what the PC is.

An Orienta rich export of an Oxford scan is a copy of the .h5oina with our groups added,
saved under the name the user gave it (``.h5`` by default). ``kikuchipy.load`` picks its
reader by the file EXTENSION: ``.h5oina`` goes to the Oxford reader, ``.h5`` is offered to
the EDAX / Bruker / kikuchipy / EMsoft readers only, and none takes an Oxford file
("Could not read ..."). The loader then fell back to the eager unified loader, which
kept neither the camera tilt (4.3 deg became 0) nor the per-pixel PC map, and never
reached the code that says "/Detector/pc differs from the header PC" - so a refined PC
was dropped without a word. The Oxford reader is now called directly for an Oxford-layout
file under any extension, the fallback applies the same hint, and ``import-h5`` reports the
PC state in its answer, with the result carrying the PC its run was indexed at.

The scan is a small synthetic file in the layout of an Oxford H5OINA (3 x 4 patterns of
24 x 24 px, per-pixel PC map, camera tilt 4.3 deg) that kikuchipy's own Oxford reader takes.
"""
import shutil

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from backend.api.main import app
from backend.api.routes import indexing as I
from backend.api.services.calibration_store import CalibrationStore
from indexing_controller import IndexingMethod, IndexingResult

N_ROWS, N_COLS, SIDE = 3, 4, 24
# the means of the per-pixel map written below, in kikuchipy's (Bruker) convention, by hand:
# PCx 0.51, PCz = DD 0.695, PCy = 1 - 0.555 for a square pattern (Oxford counts y from the top)
HEADER_PC_MEAN = (0.51, 0.445, 0.695)
REFINED = [0.5105, 0.3300, 0.8400]           # a refined PC, far from the header's


def write_oxford(path):
    n = N_ROWS * N_COLS
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"Oxford Instruments"]))
        f.create_dataset("Format Version", data=np.array([b"8.0"]))
        f.create_dataset("Software Version", data=np.array([b"6.3.9481.6"]))
        f.create_dataset("Index", data=np.array([1]))
        h = f.create_group("1/EBSD/Header")
        h.create_dataset("X Cells", data=np.int32(N_COLS))
        h.create_dataset("Y Cells", data=np.int32(N_ROWS))
        h.create_dataset("X Step", data=np.float32(0.5))
        h.create_dataset("Y Step", data=np.float32(0.5))
        h.create_dataset("Pattern Height", data=np.int32(SIDE))
        h.create_dataset("Pattern Width", data=np.int32(SIDE))
        h.create_dataset("Tilt Angle", data=np.float32(np.deg2rad(70.0)))
        h.create_dataset("Detector Orientation Euler",
                         data=np.array([0.0, np.deg2rad(94.3), 0.0], np.float32))   # camera tilt 4.3 deg
        h.create_dataset("Camera Binning Mode", data=np.bytes_("4x4 (156x128 px)"))
        d = f.create_group("1/EBSD/Data")
        d.create_dataset("Processed Patterns", data=rng.integers(0, 250, (n, SIDE, SIDE)).astype(np.uint8))
        # a map around the header values: PCx 0.50..0.52, PCy 0.55..0.56 (mean 0.555 is by hand below)
        d.create_dataset("Pattern Center X", data=np.linspace(0.50, 0.52, n).astype(np.float32))
        d.create_dataset("Pattern Center Y", data=np.linspace(0.55, 0.56, n).astype(np.float32))
        d.create_dataset("Detector Distance", data=np.linspace(0.69, 0.70, n).astype(np.float32))
        d.create_dataset("Euler", data=np.zeros((n, 3), np.float32))
        d.create_dataset("Phase", data=np.ones(n, np.int32))
        d.create_dataset("X", data=np.tile(np.arange(N_COLS) * 0.5, N_ROWS).astype(np.float32))
        d.create_dataset("Y", data=np.repeat(np.arange(N_ROWS) * 0.5, N_COLS).astype(np.float32))
        d.create_dataset("Band Contrast", data=np.full(n, 100, np.uint8))
        d.create_dataset("Mean Angular Deviation", data=np.full(n, 0.5, np.float32))
        d.create_dataset("Bands", data=np.full(n, 8, np.uint8))
        d.create_dataset("Error", data=np.zeros(n, np.uint8))
    return str(path)


@pytest.fixture
def oxford(tmp_path):
    return write_oxford(tmp_path / "scan.h5oina")


# --- why kikuchipy refuses it, and the fix -----------------------------------------

def test_kikuchipy_picks_its_reader_by_extension_so_an_h5_copy_is_refused(oxford, tmp_path):
    import kikuchipy as kp
    copy = tmp_path / "copy.h5"
    shutil.copy2(oxford, copy)
    with pytest.raises(OSError, match="Could not read"):
        kp.load(str(copy))                                       # the plugin lookup, by extension


def test_an_oxford_file_named_h5_keeps_its_tilt_and_its_pc_map(oxford, tmp_path):
    from safe_loader import load_ebsd_safe
    copy = tmp_path / "copy.h5"
    shutil.copy2(oxford, copy)
    sig = load_ebsd_safe(str(copy), verbose=False)
    ref = load_ebsd_safe(oxford, verbose=False)                  # the same file under its own name
    assert float(sig.detector.tilt) == pytest.approx(4.3, abs=1e-3)       # 94.3 - 90 by hand; not 0
    assert np.asarray(sig.detector.pc).shape == (N_ROWS, N_COLS, 3)       # the map, not one PC
    np.testing.assert_allclose(np.asarray(sig.detector.pc), np.asarray(ref.detector.pc), atol=1e-12)


def test_the_unified_loader_fallback_applies_the_hint_too(oxford, tmp_path, monkeypatch):
    """Force the fallback (the Oxford reader refuses): /Detector/pc still gets its hint."""
    import safe_loader
    out = tmp_path / "fallback.h5"
    shutil.copy2(oxford, out)
    with h5py.File(out, "a") as f:
        f.create_group("Detector").create_dataset("pc", data=np.array(REFINED))
    monkeypatch.setattr(safe_loader, "_load_oxford_by_layout", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("refused")))
    sig = safe_loader.load_ebsd_safe(str(out), verbose=False)
    # the unified loader's header PC wins over /Detector/pc, and it says so (they differ)
    header = np.asarray(sig.detector.pc, dtype=float).reshape(-1, 3).mean(axis=0)
    assert not np.allclose(header, REFINED, atol=1e-6)
    assert sig.metadata.get_item("Signal.export_pc_ignored") == pytest.approx(REFINED)


# --- the whole way: refined PC -> run -> rich export (.h5) -> import-h5 ------------

def _fake_hough(signal, phase_list, detector, config, selection_mask, progress_callback=None,
                cancel_check=None, **kw):
    n = N_ROWS * N_COLS
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(np.array([[i * 9.0, 20.0, 30.0] for i in range(n)]))),
        phase_id=np.zeros(n, dtype=np.int32),
        x=np.tile(np.arange(N_COLS, dtype=np.float32), N_ROWS),
        y=np.repeat(np.arange(N_ROWS, dtype=np.float32), N_COLS),
        phase_list=PhaseList(phases=[Phase(name="Ni", space_group=225, point_group="m-3m")]))
    return IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool), original_shape=(N_ROWS, N_COLS),
        method=IndexingMethod.HOUGH, confidence_scores=np.full(n, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.5, "recorded_lattices": {0: [3.52] * 3 + [90, 90, 90]}})


def _ni_cif(tmp_path):
    cif = tmp_path / "Ni.cif"
    cif.write_text(
        "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding="utf-8")
    return str(cif)


@pytest.fixture
def session(monkeypatch, oxford, tmp_path):
    import indexing_controller
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import pcrefinement as R
    store = CalibrationStore()
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    monkeypatch.setattr(R, "_sessions", {})
    monkeypatch.setattr(R, "_parked_sessions", {})
    monkeypatch.setattr(indexing_controller, "hough_index_patterns", _fake_hough)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="indexing": fn())
    saved = dict(I._result_registry)
    saved_active = I._active_result_id
    client = TestClient(app)
    assert client.post("/api/ebsd/load", json={"path": oxford}).status_code == 200
    yield client, store, tmp_path
    I._result_registry.clear()
    I._result_registry.update(saved)
    I._active_result_id = saved_active


def test_a_refined_pc_survives_the_rich_export_and_the_import_says_what_happened(session, oxford):
    client, store, tmp = session
    entry = store.get_entry("scan")
    assert entry.pc_map is not None and entry.pc_source == "header"
    np.testing.assert_allclose(entry.pc_single, HEADER_PC_MEAN, atol=2e-3)     # by hand: the means of the map
    store.update_pc("scan", REFINED, source="refined")                           # the user refined the PC
    resp = client.post("/api/indexing/start", json={"method": "hough", "cif_paths": [_ni_cif(tmp)]})
    assert resp.status_code == 200, resp.text
    run = I._get_result().metadata["pc_at_run"]
    np.testing.assert_allclose(run["pc"], REFINED, atol=1e-6)
    assert run["pc_source"] == "refined"

    out = tmp / "rich.h5"                                                        # the default extension
    resp = client.post("/api/indexing/export", json={"format": "h5", "filename": str(out)})
    assert resp.status_code == 200, resp.text
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["Detector/pc"][()], REFINED, atol=1e-6)     # the export states the run's PC

    resp = client.post("/api/indexing/import-h5", json={"path": str(out)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # the answer says what the live PC is, and that the header's won over /Detector/pc
    assert body["pc_notice"] == "export_pc_ignored"
    np.testing.assert_allclose(body["pc_export_ignored"], REFINED, atol=1e-6)
    np.testing.assert_allclose(body["pc"], HEADER_PC_MEAN, atol=2e-3)
    # the imported RESULT carries the PC of its run, so its own exports state it again
    np.testing.assert_allclose(body["pc_at_run"]["pc"], REFINED, atol=1e-6)
    np.testing.assert_allclose(I._result_registry[body["result_id"]].metadata["pc_at_run"]["pc"],
                               REFINED, atol=1e-6)
    # and the geometry of the file survived: camera tilt and the per-pixel map
    entry = store.get_entry("rich")
    from backend.api.routes.ebsd_viewer import _canonical_path
    assert entry.file_key == _canonical_path(str(out))             # import-h5 says whose entry it is
    assert entry.tilt == pytest.approx(4.3, abs=1e-3)
    assert entry.pc_map is not None and entry.pc_map.shape == (N_ROWS, N_COLS, 3)
    assert body["pc_store_source"] == "header"


def test_an_export_without_a_run_pc_gives_the_imported_result_none(session, oxford):
    client, store, tmp = session
    resp = client.post("/api/indexing/start", json={"method": "hough", "cif_paths": [_ni_cif(tmp)]})
    assert resp.status_code == 200, resp.text
    I._get_result().metadata.pop("pc_at_run")                       # a result from before the record
    out = tmp / "old.h5"
    assert client.post("/api/indexing/export", json={"format": "h5", "filename": str(out)}).status_code == 200
    body = client.post("/api/indexing/import-h5", json={"path": str(out)}).json()
    assert body["pc_at_run"] is None                                # not made up from the store or the header
    assert "pc_at_run" not in I._result_registry[body["result_id"]].metadata


def test_the_reader_of_a_run_record_takes_nothing_from_an_export_that_says_not_recorded(tmp_path):
    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("Detector")
        g.create_dataset("pc", data=np.array(REFINED))
        g.attrs["sample_tilt"] = 70.0
        g.attrs["camera_tilt"] = 4.3
        g.attrs["pc_not_recorded"] = True
    assert I._pc_record_from_export(str(path)) is None
    with h5py.File(path, "a") as f:
        del f["Detector"].attrs["pc_not_recorded"]
    rec = I._pc_record_from_export(str(path))
    np.testing.assert_allclose(rec["pc"], REFINED)
    assert (rec["sample_tilt"], rec["tilt"]) == (70.0, 4.3)


# --- the Aztec seed map after a file switch (it is stored as "header") --------------

def test_the_aztec_seed_map_is_the_files_own_map_and_comes_back_with_the_file(session, oxford, tmp_path):
    """"Use as seed" stores the Aztec per-pixel PC as a header map. After a file switch the
    store is rebuilt from the file, and an Oxford file carries exactly that map in its own
    header, so the seed is regenerated (nothing is lost); this pins it. By hand: the values
    of the file's Pattern Center X / Pattern Center Y / Detector Distance datasets."""
    import asyncio
    from types import SimpleNamespace
    from backend.api.routes import analysis
    client, store, tmp = session
    with h5py.File(oxford, "r") as f:
        d = f["1/EBSD/Data"]
        prop = {"pc_x": d["Pattern Center X"][()], "pc_y": d["Pattern Center Y"][()],
                "dd": d["Detector Distance"][()]}
    n = N_ROWS * N_COLS
    store.update_pc_map("scan", np.full((N_ROWS, N_COLS, 3), 0.5), source="header")   # not the file's map
    analysis._dataset = SimpleNamespace(xmap=SimpleNamespace(prop=prop), size=n, shape=(N_ROWS, N_COLS))
    try:
        asyncio.run(analysis.pc_drift_to_seed(analysis.PcSeedRequest(target_dataset="scan")))
    finally:
        analysis._dataset = None
    seeded = store.get_entry("scan").pc_map.copy()
    assert not np.allclose(seeded, 0.5, atol=1e-3)                  # the seed replaced the flat map
    assert store.get_entry("scan").pc_source == "header"
    other = write_oxford(tmp_path / "other.h5oina")
    assert client.post("/api/ebsd/load", json={"path": other}).status_code == 200      # switch away
    assert client.post("/api/ebsd/load", json={"path": oxford}).status_code == 200     # and back
    back = store.get_entry("scan")
    assert back.pc_source == "header"
    np.testing.assert_allclose(back.pc_map, seeded, atol=1e-6)                          # regenerated, same map


# --- exports written before the record existed: PC and sample tilt, no camera tilt ---------

def _legacy_export(path, pc=(0.5026, 0.3273, 0.8458), sample_tilt=75.7):
    """What every export before the run record wrote in /Detector: pc, sample_tilt, shape."""
    with h5py.File(path, "w") as f:
        g = f.create_group("Detector")
        g.create_dataset("pc", data=np.array(pc))
        g.attrs["sample_tilt"] = sample_tilt
        g.attrs["shape"] = [60, 60]
    return str(path)


def test_an_old_export_keeps_its_pc_and_marks_the_tilt_it_does_not_have(tmp_path):
    rec = I._pc_record_from_export(_legacy_export(tmp_path / "old.h5"))
    assert rec is not None, "an export with a PC is not 'not recorded'"
    np.testing.assert_allclose(rec["pc"], (0.5026, 0.3273, 0.8458))
    assert rec["sample_tilt"] == pytest.approx(75.7)
    assert rec["tilt"] is None and rec["azimuthal"] is None          # camera tilt: not recorded
    assert rec["pc_source"] == "export"


def test_a_record_with_an_unknown_tilt_is_exported_as_not_recorded_not_as_the_stores(tmp_path):
    from types import SimpleNamespace
    rec = I._pc_record_from_export(_legacy_export(tmp_path / "old.h5"))
    result = SimpleNamespace(metadata={"pc_at_run": rec})
    with h5py.File(tmp_path / "out.h5", "w") as f:
        g = f.create_group("Detector")
        I._write_detector_facts(g, "ds", SimpleNamespace(pc=np.array([[0.1, 0.2, 0.3]])), result)
        np.testing.assert_allclose(g["pc"][()], (0.5026, 0.3273, 0.8458))
        assert g.attrs["sample_tilt"] == pytest.approx(75.7)
        assert "camera_tilt" not in g.attrs and bool(g.attrs["camera_tilt_not_recorded"]) is True
        assert "geometry_not_recorded" not in g.attrs
    # and read back, the same record
    again = I._pc_record_from_export(str(tmp_path / "out.h5"))
    assert again["tilt"] is None and again["sample_tilt"] == pytest.approx(75.7)


def test_the_ang_of_a_result_with_a_pc_but_no_camera_tilt_states_the_pc(tmp_path):
    from types import SimpleNamespace
    rec = I._pc_record_from_export(_legacy_export(tmp_path / "old.h5"))
    result = SimpleNamespace(metadata={"pc_at_run": rec})
    assert I._pc_for_export("ds", None, result).tolist() == pytest.approx([0.5026, 0.3273, 0.8458])
    assert I._geometry_for_export("ds", None, result) == {"sample_tilt": 75.7, "tilt": None, "azimuthal": None}


@pytest.mark.skipif(not __import__("os").environ.get("ORIENTA_TEST_OLD_EXPORTS"),
                    reason="ORIENTA_TEST_OLD_EXPORTS not set (a folder of exports written before the run record)")
def test_real_old_exports_keep_their_pc():
    """Every .h5 of the folder that has a /Detector/pc."""
    import os
    from pathlib import Path
    seen = 0
    for p in sorted(Path(os.environ["ORIENTA_TEST_OLD_EXPORTS"]).glob("*.h5")):
        try:
            with h5py.File(p, "r") as f:
                if "Detector/pc" not in f:
                    continue
                want = np.asarray(f["Detector/pc"][()], dtype=float).ravel()[:3]
        except OSError:
            continue
        rec = I._pc_record_from_export(str(p))
        assert rec is not None and np.allclose(rec["pc"], want), p.name
        seen += 1
    assert seen >= 1, "no export with a /Detector/pc in the folder"


def test_an_older_spherical_export_without_detector_keeps_the_pc_of_its_render_geometry(tmp_path):
    """No /Detector, but /Indexing.attrs["detector_geometry"] (what the spherical run rendered at):
    the record takes PC and tilts from it and says where it got them."""
    import json
    path = tmp_path / "old_spherical.h5"
    with h5py.File(path, "w") as f:
        f.create_group("Indexing").attrs["detector_geometry"] = json.dumps({
            "pc_x": 0.5469, "pc_y": 0.465, "pc_z": 0.6086, "tilt": 3.44, "sample_tilt": 70.0,
            "pat_width": 118, "pat_height": 118})
    rec = I._pc_record_from_export(str(path))
    np.testing.assert_allclose(rec["pc"], (0.5469, 0.465, 0.6086))
    assert (rec["sample_tilt"], rec["tilt"], rec["azimuthal"]) == (70.0, 3.44, None)
    assert rec["pc_source"] == "render_geometry"
    # nothing in the file at all: nothing is made up
    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as f:
        f.create_group("Indexing")
    assert I._pc_record_from_export(str(empty)) is None
    # an export that says "not recorded" is not rescued by a geometry it happens to carry
    said = tmp_path / "said.h5"
    with h5py.File(said, "w") as f:
        f.create_group("Detector").attrs["pc_not_recorded"] = True
        f.create_group("Indexing").attrs["detector_geometry"] = json.dumps({"pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5})
    assert I._pc_record_from_export(str(said)) is None
