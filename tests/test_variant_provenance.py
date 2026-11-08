"""A result knows which patterns it was indexed on: the measurement and the pattern source of the file.

The stamp is ``metadata["load_variant"]`` (``{"measurement", "pattern_source"}``, None = the file's default
variant, missing = the same). It is taken when a run STARTS, written to every export as the attributes
``measurement`` and ``pattern_source`` (only when not the default), read back on import, used to reload the
right variant when a result is activated, and named in the methods sentence when the patterns were the raw ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import ebsd_viewer as ev  # noqa: E402
from backend.api.routes import indexing as I  # noqa: E402
from backend.api.services.citations.provenance import get_steps  # noqa: E402
from backend.api.services.citations.render import render_methods  # noqa: E402
from backend.api.services.citations.steps import STEP_REGISTRY  # noqa: E402
from indexing_controller import IndexingMethod, IndexingResult  # noqa: E402
from tests.test_load_variant import _file, _unprocessed  # noqa: E402
from tests.test_load_variant_routes import _reset, _v  # noqa: E402

UNPROCESSED = {"measurement": "1", "pattern_source": "unprocessed"}
STEP_KEY = "indexing.pattern_source"


@pytest.fixture
def client():
    from backend.api.main import app
    _reset()
    yield TestClient(app)
    _reset()


@pytest.fixture(autouse=True)
def restore_registry():
    saved, saved_active = dict(I._result_registry), I._active_result_id
    I._result_registry.clear()
    I._active_result_id = None
    try:
        yield
    finally:
        I._result_registry.clear()
        I._result_registry.update(saved)
        I._active_result_id = saved_active


@pytest.fixture
def multi(tmp_path):
    return _file(tmp_path, "multi.h5oina", [(1, {}), (2, {})])


def _load(client, path, variant=None):
    body = {"path": path}
    if variant is not None:
        body["variant"] = variant
    r = client.post("/api/ebsd/load", json=body)
    assert r.status_code == 200, r.text
    return r


def _result(**meta) -> IndexingResult:
    n_rows, n_cols = 4, 3
    n = n_rows * n_cols
    eulers = np.deg2rad(np.array([[i * 11.0, 20.0 + i, 30.0 + 2 * i] for i in range(n)]))
    xmap = CrystalMap(
        rotations=Rotation.from_euler(eulers), phase_id=np.zeros(n, dtype=np.int32),
        x=np.tile(np.arange(n_cols, dtype=np.float32), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=np.float32), n_cols),
        phase_list=PhaseList(phases=[Phase(name="Al", space_group=225, point_group="m-3m")]))
    return IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool), original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL, confidence_scores=np.full(n, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.5, **meta})


def _store(**meta) -> str:
    return I._store_result(_result(**meta), "spherical")


def _export(client, path: Path, fmt="h5_light") -> Path:
    r = client.post("/api/indexing/export", json={"format": fmt, "filename": str(path)})
    assert r.status_code == 200, r.text
    return path


def _import(client, path: Path) -> dict:
    r = client.post("/api/indexing/import-h5", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return I._result_registry[I._active_result_id].metadata


def _attrs(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return dict(f["Indexing"].attrs)


# ------------------------------------------------------------------------------------ the run start


def test_result_stamps_variant_at_run_start(monkeypatch, tmp_path):
    """A run on the unprocessed patterns that sees another variant loaded while it runs still stores
    the stamp of its own start; a run on the default stores none. The stamp also puts the sentence
    about the raw patterns on the result, and none on the default one."""
    import indexing_controller
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    from backend.api.services.calibration_store import CalibrationStore
    from tests.test_result_keeps_its_run_pc import RUN_PC, SHAPE, _fake_hough, _ni_cif

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
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="indexing": fn())

    def indexer_during_which_the_default_is_loaded(*a, **k):
        monkeypatch.setattr(ev, "_ebsd_variant_key", None)
        return _fake_hough(*a, **k)

    monkeypatch.setattr(indexing_controller, "hough_index_patterns", indexer_during_which_the_default_is_loaded)
    from backend.api.main import app
    cif = _ni_cif(tmp_path)

    def run(variant_key):
        monkeypatch.setattr(ev, "_ebsd_variant_key", variant_key)
        resp = TestClient(app).post("/api/indexing/start", json={"method": "hough", "cif_paths": [cif]})
        assert resp.status_code == 200, resp.text
        assert I._indexing_tasks[resp.json()["task_id"]]["status"] == "completed"
        assert ev._ebsd_variant_key is None                       # premise: the switch really happened
        return I._get_result().metadata

    meta = run(("1", "unprocessed"))
    assert meta["load_variant"] == UNPROCESSED
    steps = [s for s in get_steps(I._get_result()) if s["key"] == STEP_KEY]
    assert [s["params"] for s in steps] == [{"pattern_source": "unprocessed"}]
    assert "unprocessed patterns" in render_methods(get_steps(I._get_result()))

    meta = run(None)
    assert meta["load_variant"] is None
    assert STEP_KEY not in [s["key"] for s in get_steps(I._get_result())]
    assert "unprocessed" not in render_methods(get_steps(I._get_result()))


def test_the_stamp_is_not_lost_when_the_scan_stamp_fails(client, multi, monkeypatch):
    """The variant is stamped on its own: a failing read of the crop window must not skip it."""
    _load(client, multi, _v("1", "unprocessed"))

    def boom():
        raise RuntimeError("no scan provenance today")

    monkeypatch.setattr(I, "_scan_provenance_fields", boom)
    rid = _store()
    assert I._result_registry[rid].metadata["load_variant"] == UNPROCESSED


def test_a_seeded_stamp_survives_the_store(client, multi):
    """setdefault: what a run seeded at its start wins over whatever is loaded when it is stored."""
    _load(client, multi, _v("2", "unprocessed"))
    res = _result()
    I._seed_variant_stamp(res, {"measurement": "1", "pattern_source": "processed"})
    rid = I._store_result(res, "spherical")
    assert I._result_registry[rid].metadata["load_variant"] == {"measurement": "1", "pattern_source": "processed"}


def test_the_default_stamp_is_none_and_adds_no_step(client, multi):
    _load(client, multi)
    res = _result()
    I._seed_variant_stamp(res, I._variant_stamp_at_run_start())
    assert res.metadata["load_variant"] is None
    assert get_steps(res) == []


def test_a_batch_item_stamps_the_variant_it_loaded(client, multi, monkeypatch, tmp_path):
    """The batch loop loads each file as it was last loaded (the entry's variant) and stamps that, taken
    right after its own load, even when another variant becomes the loaded one during the item. Its
    automatic export says so too."""
    import indexing_controller

    _load(client, multi, _v("1", "unprocessed"))
    I._result_registry.clear()

    def spherical_during_which_the_default_is_loaded(h5_path, config, detector_params, selection_mask, **kwargs):
        monkeypatch.setattr(ev, "_ebsd_variant_key", None)
        mask = np.asarray(selection_mask, dtype=bool)
        return IndexingResult(
            xmap=None, selection_mask=mask, original_shape=mask.shape, method=IndexingMethod.SPHERICAL,
            confidence_scores=np.full(mask.size, 0.5, dtype=np.float32))

    for name in ("spherical_index_patterns", "spherical_gpu_index_patterns"):
        monkeypatch.setattr(indexing_controller, name, spherical_during_which_the_default_is_loaded)
    monkeypatch.setattr(I, "_attach_indexing_metadata", lambda *a, **k: None)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    saved_batch = dict(I._batch_state)
    out_dir = tmp_path / "batch_out"
    out_dir.mkdir()
    try:
        resp = client.post("/api/indexing/batch/start", json={
            "auto_export": True, "export_dir": str(out_dir),
            "datasets": [{"file_path": multi, "method": "spherical", "selection_mode": "full",
                          "export_result": False}]})
        assert resp.status_code == 200, resp.text
        entry = I._batch_state["results"][-1]
        assert entry["error"] is None, entry
        assert ev._ebsd_variant_key is None                              # premise: the switch really happened
        meta = I._get_result().metadata
        assert meta["load_variant"] == UNPROCESSED
        assert STEP_KEY in [s["key"] for s in get_steps(I._get_result())]
        (exported,) = list(out_dir.glob("*_indexed.h5"))
        attrs = _attrs(exported)
        assert attrs["pattern_source"] == "unprocessed" and attrs["measurement"] == "1"
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved_batch)


# ------------------------------------------------------------------------------------------ exports


@pytest.mark.parametrize("fmt", ["h5_light", "h5"])
def test_export_round_trip_keeps_variant(client, multi, tmp_path, fmt):
    _load(client, multi, _v("2", "unprocessed"))
    _store()
    out = _export(client, tmp_path / f"result_{fmt}.h5", fmt)
    attrs = _attrs(out)
    assert attrs["pattern_source"] == "unprocessed" and attrs["measurement"] == "2"

    I._result_registry.clear()
    meta = _import(client, out)
    assert meta["indexed_variant"] == {"measurement": "2", "pattern_source": "unprocessed"}
    again = _export(client, tmp_path / f"again_{fmt}.h5", fmt)             # the imported result re-exports it
    assert _attrs(again)["pattern_source"] == "unprocessed" and _attrs(again)["measurement"] == "2"


def test_an_imported_result_stays_active_on_the_file_it_names(client, multi, tmp_path):
    """The imported stamp describes the original scan; ``load_variant`` still says which variant of
    ``source_file`` (the exported file) the result belongs to, so nothing deactivates it."""
    _load(client, multi, _v("1", "unprocessed"))
    _store()
    out = _export(client, tmp_path / "result.h5")
    I._result_registry.clear()
    meta = _import(client, out)
    assert meta["load_variant"] is None
    assert I._active_result_id is not None


def test_the_default_export_has_no_variant_attributes(client, multi, tmp_path):
    _load(client, multi)
    _store()
    for fmt in ("h5_light", "h5"):
        out = _export(client, tmp_path / f"default_{fmt}.h5", fmt)
        assert "pattern_source" not in _attrs(out) and "measurement" not in _attrs(out)


def test_old_result_without_stamp_is_default(client, multi, tmp_path):
    """A result stored before the stamp existed (no key at all) exports and imports as the default and is
    activated without a reload."""
    _load(client, multi)
    rid = _store()
    I._result_registry[rid].metadata.pop("load_variant", None)
    out = _export(client, tmp_path / "old.h5")
    assert "pattern_source" not in _attrs(out)
    assert I._read_variant_provenance(out) is None
    assert "indexed_variant" not in _import(client, out)

    calls = []
    real = ev.load_ebsd

    async def counting(req):
        calls.append(req)
        return await real(req)

    ev.load_ebsd, saved = counting, ev.load_ebsd
    try:
        r = client.post(f"/api/indexing/results/activate/{rid}")
    finally:
        ev.load_ebsd = saved
    assert r.status_code == 200, r.text
    assert calls == []


# -------------------------------------------------------------------------------------- activation


def test_activate_result_reloads_its_variant(client, multi):
    _load(client, multi, _v("1", "unprocessed"))
    rid = _store()
    _load(client, multi)                                                    # the default replaces it
    assert I._active_result_id is None and ev._ebsd_variant_key is None
    r = client.post(f"/api/indexing/results/activate/{rid}")
    assert r.status_code == 200, r.text
    assert ev._ebsd_variant_key == ("1", "unprocessed")
    data = np.asarray(ev._raw_signals[ev._active_dataset].data)
    assert data.dtype == np.int16
    assert np.array_equal(data, _unprocessed(1).reshape(data.shape))
    assert I._active_result_id == rid


def test_activate_a_default_result_reloads_the_default(client, multi):
    _load(client, multi)
    rid = _store()
    _load(client, multi, _v("2", "unprocessed"))
    r = client.post(f"/api/indexing/results/activate/{rid}")
    assert r.status_code == 200, r.text
    assert ev._ebsd_variant_key is None
    assert I._active_result_id == rid


def test_activate_result_compares_the_variant_not_only_the_path(client, multi):
    """Same file loaded, other variant: the file is loaded again. Same variant: it is not."""
    _load(client, multi, _v("1", "unprocessed"))
    unprocessed = _store()
    _load(client, multi, _v("2", "processed"))
    other_measurement = _store()
    loads = []
    real = ev.load_ebsd

    async def counting(req):
        loads.append(req.variant.model_dump() if req.variant else None)
        return await real(req)

    ev.load_ebsd = counting
    try:
        assert client.post(f"/api/indexing/results/activate/{other_measurement}").status_code == 200
        assert loads == []                                                  # already the loaded variant
        assert client.post(f"/api/indexing/results/activate/{unprocessed}").status_code == 200
        assert loads == [UNPROCESSED]
    finally:
        ev.load_ebsd = real
    assert ev._ebsd_variant_key == ("1", "unprocessed")


def test_activate_result_says_why_when_the_variant_is_gone(client, multi):
    _load(client, multi, _v("1", "unprocessed"))
    rid = _store()
    I._result_registry[rid].metadata["load_variant"] = {"measurement": "99", "pattern_source": "unprocessed"}
    _load(client, multi)
    r = client.post(f"/api/indexing/results/activate/{rid}")
    assert r.status_code == 400 and r.headers["x-orienta-code"] == "loadVariantUnknown"


# ----------------------------------------------- a refinement covers its parent's pixels and patterns


def _refine_after_loading(client, path, parent, variant=None):
    """Store ``parent``, load ``variant`` of ``path`` (what the user has open when refining), refine."""
    from tests.test_result_scan_geometry import _refine
    I._store_result(parent, "spherical")
    _load(client, path, variant)
    return I._result_registry[_refine(parent)]


def test_a_refinement_reports_its_parents_variant_not_the_loaded_one(client, multi):
    from tests.test_refinement_service import _make_fake_indexed_result
    parent = _make_fake_indexed_result()
    parent.metadata["load_variant"] = dict(UNPROCESSED)
    refined = _refine_after_loading(client, multi, parent)                 # the default is the loaded one
    assert refined.metadata["load_variant"] == UNPROCESSED
    assert [s["params"] for s in get_steps(refined) if s["key"] == STEP_KEY] == [
        {"pattern_source": "unprocessed"}]


def test_a_refinement_of_an_import_keeps_what_the_file_said(client, multi):
    from tests.test_refinement_service import _make_fake_indexed_result
    parent = _make_fake_indexed_result()
    parent.metadata.update(indexed_variant=dict(UNPROCESSED), load_variant=None)
    refined = _refine_after_loading(client, multi, parent)
    assert refined.metadata["indexed_variant"] == UNPROCESSED
    assert refined.metadata["load_variant"] is None
    assert I._stored_variant(refined) == UNPROCESSED


def test_a_refinement_of_a_default_parent_never_takes_the_loaded_variant(client, multi):
    from tests.test_refinement_service import _make_fake_indexed_result
    parent = _make_fake_indexed_result()
    parent.metadata["load_variant"] = None
    refined = _refine_after_loading(client, multi, parent, _v("1", "unprocessed"))
    assert ev._ebsd_variant_key is not None                                # premise: the other one is live
    assert refined.metadata["load_variant"] is None
    assert I._stored_variant(refined) is None
    assert [s for s in get_steps(refined) if s["key"] == STEP_KEY] == []


# ------------------------------------------------------------------------------------ the sentence


def test_methods_sentence_names_unprocessed():
    step = STEP_REGISTRY[STEP_KEY]
    assert step.citation_ids
    text = render_methods([{"key": STEP_KEY, "params": {"pattern_source": "unprocessed"}}])
    assert "unprocessed patterns" in text and "{" not in text


def test_the_step_never_records_a_path(client, multi):
    _load(client, multi, _v("1", "unprocessed"))
    res = _result()
    I._seed_variant_stamp(res, I._variant_stamp_at_run_start())
    (step,) = [s for s in get_steps(res) if s["key"] == STEP_KEY]
    assert step["params"] == {"pattern_source": "unprocessed"}
    assert "multi" not in str(step)                                        # neither file name nor path
    I._seed_variant_stamp(res, I._variant_stamp_at_run_start())            # seeding twice records once
    assert len([s for s in get_steps(res) if s["key"] == STEP_KEY]) == 1
