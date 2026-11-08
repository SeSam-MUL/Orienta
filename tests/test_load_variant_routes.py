"""``/load`` and ``/switch-file`` with a chosen variant (measurement x pattern source): the loaded-files
registry, the per-file stash, the answer (``variant``, ``has_raw_patterns``, ``format_type``) and the
dataset names. Everything runs the real loader on small synthetic files."""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safe_loader  # noqa: E402
from backend.api.routes import ebsd_viewer as ev  # noqa: E402
from backend.api.services import crop_window as crop_window_service  # noqa: E402
from backend.api.services import scan_geometry as scan_geometry_service  # noqa: E402
from backend.api.services import variant_registry  # noqa: E402
from backend.api.services.calibration_store import calibration_store  # noqa: E402
from backend.api.services.load_choices import LOAD_CODES, LoadChoiceError, LoadVariant  # noqa: E402
from tests.test_load_choices import _sampleb, _text  # noqa: E402
from tests.test_load_variant import NX, NY, SX, SY, _file, _processed, _unprocessed  # noqa: E402

DOT = "\u00b7"
CODE_HEADER = "x-orienta-code"
ROOT = Path(__file__).resolve().parents[1]


def _name(stem: str, *parts: str) -> str:
    return stem + "".join(f" {DOT} {p}" for p in parts)


def _reset() -> None:
    ev._loaded_files.clear()
    ev._raw_signals.clear()
    ev._positions.clear()
    ev._dirty_datasets.clear()
    ev._signal_masks.clear()
    ev._registry_by_file.clear()
    ev._overview_cache.clear()
    ev._pc_load_notice.clear()
    ev._ebsd_signal = None
    ev._ebsd_file_path = None
    ev._active_dataset = ""
    ev._ebsd_variant_key = None
    ev._ebsd_apex = None
    variant_registry.clear_all()
    crop_window_service.clear_all()
    scan_geometry_service.clear_all()
    calibration_store.clear()
    try:
        from backend.api.services import h5_session
        h5_session.close_file()
    except Exception:  # noqa: BLE001 - nothing open
        pass


@pytest.fixture
def client():
    _reset()
    app = FastAPI()
    app.include_router(ev.router, prefix="/api/ebsd")
    yield TestClient(app)
    _reset()


@pytest.fixture
def multi(tmp_path):
    """Groups 1 and 2, each with processed and unprocessed patterns."""
    return _file(tmp_path, "multi.h5oina", [(1, {}), (2, {})])


@pytest.fixture
def other(tmp_path):
    return _file(tmp_path, "other.h5oina", [(1, {})])


@pytest.fixture
def processed_only(tmp_path):
    return _file(tmp_path, "proc.h5oina", [(1, {"unprocessed": False, "unprocessed_bg": False})])


@pytest.fixture
def bruker(tmp_path):
    """Two scan groups, one pattern dataset each; loadable by kikuchipy's Bruker reader."""
    path = tmp_path / "bruker.h5"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Bruker Nano")
        _text(f, "Version", "1.0")
        for name, offset in (("Scan 1", 0), ("Scan 2", 50)):
            hdr = f.create_group(f"{name}/EBSD/Header")
            data = f.create_group(f"{name}/EBSD/Data")
            _text(hdr, "Grid Type", "isometric")
            for key, value in (("NCOLS", NX), ("NROWS", NY), ("PatternHeight", SY), ("PatternWidth", SX),
                               ("XSTEP", 0.5), ("YSTEP", 0.5)):
                hdr.create_dataset(key, data=value)
            patterns = ((np.arange(NY * NX * SY * SX) % 100) + offset).astype(np.uint8)
            data.create_dataset("RawPatterns", data=patterns.reshape(NY * NX, SY, SX))
            for key in ("PCX", "PCY", "DD"):
                data.create_dataset(key, data=np.full(NY * NX, 0.5, np.float32))
    return str(path)


def _bruker_patterns(scan: int) -> np.ndarray:
    return (((np.arange(NY * NX * SY * SX) % 100) + (0, 0, 50)[scan]).astype(np.uint8)
            .reshape(NY, NX, SY, SX))


def _load(client, path, variant=None, **extra):
    body = {"path": path, **extra}
    if variant is not None:
        body["variant"] = variant
    return client.post("/api/ebsd/load", json=body)


def _v(measurement, source):
    return {"measurement": measurement, "pattern_source": source}


def _active_data():
    return np.asarray(ev._raw_signals[ev._active_dataset].data)


def _entry(path):
    return ev._loaded_entry(path)


# ------------------------------------------------------------------------------ the chosen variant


def test_load_unprocessed_registers_variant(client, multi):
    r = _load(client, multi, _v("1", "unprocessed"))
    assert r.status_code == 200, r.text
    body = r.json()
    name = _name("multi", "unprocessed")
    assert body["dataset_name"] == name
    assert body["variant"] == {"measurement": "1", "pattern_source": "unprocessed", "is_default": False}
    assert _entry(multi)["variant"] == _v("1", "unprocessed")
    assert _entry(multi)["name"] == name
    assert variant_registry.variant_for(name) == LoadVariant("1", "unprocessed")
    assert ev._active_dataset == name
    assert _active_data().dtype == np.int16
    assert np.array_equal(_active_data(), _unprocessed(1).reshape(NY, NX, SY, SX))
    listed = client.get("/api/ebsd/loaded-files").json()["files"]
    assert listed[0]["variant"] == _v("1", "unprocessed")


def test_switch_keeps_variant(client, multi, other):
    _load(client, multi, _v("2", "unprocessed"))
    assert _load(client, other).status_code == 200
    assert variant_registry.variant_for(_name("multi", "2", "unprocessed")) is None   # cleared by the load
    r = client.post("/api/ebsd/switch-file", json={"path": multi})
    assert r.status_code == 200, r.text
    assert r.json()["dataset_name"] == _name("multi", "2", "unprocessed")
    assert r.json()["variant"]["pattern_source"] == "unprocessed"
    assert np.array_equal(_active_data(), _unprocessed(2).reshape(NY, NX, SY, SX))
    assert _entry(multi)["variant"] == _v("2", "unprocessed")
    assert variant_registry.variant_for(ev._active_dataset) == LoadVariant("2", "unprocessed")


def test_other_variant_replaces_entry(client, multi):
    _load(client, multi, _v("1", "processed"))
    assert len(ev._loaded_files) == 1
    _load(client, multi, _v("1", "unprocessed"))
    assert len(ev._loaded_files) == 1
    assert _entry(multi)["variant"] == _v("1", "unprocessed")
    r = _load(client, multi, _v("2", "processed"))
    assert r.json()["dataset_name"] == _name("multi", "2")
    assert len(ev._loaded_files) == 1
    assert _entry(multi)["variant"] == _v("2", "processed")
    assert _entry(multi)["name"] == _name("multi", "2")
    assert set(ev._raw_signals) == {_name("multi", "2")}


def test_default_name_unchanged(client, multi):
    r = _load(client, multi)
    assert r.json()["dataset_name"] == "multi"
    assert "variant" not in _entry(multi)
    assert _entry(multi)["name"] == "multi"
    assert np.array_equal(_active_data(), _processed(1).reshape(NY, NX, SY, SX))
    # the default named in full is the default: same name, same data
    r = _load(client, multi, _v("1", "processed"))
    assert r.json()["dataset_name"] == "multi"
    assert r.json()["variant"]["is_default"] is True
    assert np.array_equal(_active_data(), _processed(1).reshape(NY, NX, SY, SX))


def test_plain_load_says_which_variant_it_took(client, multi, tmp_path):
    body = _load(client, multi).json()
    assert body["variant"] == {"measurement": "1", "pattern_source": "processed", "is_default": True}
    only = _file(tmp_path, "unproc.h5oina", [(1, {"processed": False, "processed_bg": False})])
    body = _load(client, only).json()
    assert body["variant"]["pattern_source"] == "unprocessed" and body["variant"]["is_default"] is True
    assert body["dataset_name"] == "unproc"


def test_stash_never_restores_other_variant(client, multi, other):
    """Processed + background removal, load unprocessed, switch away and back."""
    assert _load(client, multi).status_code == 200
    copy = client.post("/api/ebsd/deepcopy", json={"name": "multi_bg"}).json()["name"]
    client.post("/api/ebsd/select-dataset", json={"name": copy})
    r = client.post("/api/ebsd/background-removal", json={"method": "static", "static_bg_row": 0, "static_bg_col": 0})
    assert r.status_code == 200, r.text
    corrected = _active_data().copy()
    assert copy in ev._dirty_datasets
    assert not np.array_equal(corrected, _processed(1).reshape(NY, NX, SY, SX))

    unproc = _name("multi", "unprocessed")
    assert _load(client, multi, _v("1", "unprocessed")).json()["dataset_name"] == unproc
    assert set(ev._raw_signals) == {unproc}
    assert _load(client, other).status_code == 200                   # away ...
    r = client.post("/api/ebsd/switch-file", json={"path": multi})   # ... and back
    assert r.status_code == 200, r.text
    assert ev._active_dataset == unproc
    assert set(ev._raw_signals) == {unproc}, "a processed copy came back into the unprocessed variant"
    assert not ev._dirty_datasets
    assert _active_data().dtype == np.int16
    assert np.array_equal(_active_data(), _unprocessed(1).reshape(NY, NX, SY, SX))

    # the processed work is not lost: it is kept for the processed variant
    assert _load(client, multi, _v("1", "processed")).status_code == 200
    assert copy in ev._raw_signals and copy in ev._dirty_datasets
    assert np.array_equal(np.asarray(ev._raw_signals[copy].data), corrected)


def test_stash_of_the_unprocessed_variant_comes_back_to_it(client, multi, other):
    _load(client, multi, _v("1", "unprocessed"))
    copy = client.post("/api/ebsd/deepcopy", json={"name": "u_copy"}).json()["name"]
    client.post("/api/ebsd/select-dataset", json={"name": copy})
    assert variant_registry.variant_for(copy) == LoadVariant("1", "unprocessed")
    _load(client, other)
    client.post("/api/ebsd/switch-file", json={"path": multi})
    assert copy in ev._raw_signals
    assert variant_registry.variant_for(copy) == LoadVariant("1", "unprocessed")
    key = ev._canonical_path(multi)
    assert key not in ev._registry_by_file          # consumed by the restore


def test_stash_keys_hold_the_variant(client, multi, other):
    _load(client, multi, _v("2", "unprocessed"))
    client.post("/api/ebsd/deepcopy", json={"name": "c"})
    _load(client, other)
    assert list(ev._registry_by_file) == [(ev._canonical_path(multi), ("2", "unprocessed"))]


def test_explicit_default_variant_shares_the_stash_with_a_plain_load(client, multi, other):
    _load(client, multi)
    client.post("/api/ebsd/deepcopy", json={"name": "keep"})
    _load(client, other)
    assert ev._canonical_path(multi) in ev._registry_by_file     # the default variant keeps today's key
    _load(client, multi, _v("1", "processed"))
    assert "keep" in ev._raw_signals


def test_registry_follows_copy_crop_and_delete(client, multi):
    _load(client, multi, _v("2", "unprocessed"))
    want = LoadVariant("2", "unprocessed")
    copy = client.post("/api/ebsd/deepcopy", json={"name": "cp"}).json()["name"]
    assert variant_registry.variant_for(copy) == want
    r = client.post("/api/ebsd/crop", json={"shape": "rect", "row0": 0, "col0": 0, "rows": 1, "cols": 2})
    assert r.status_code == 200, r.text
    crop = r.json()["name"]
    assert variant_registry.variant_for(crop) == want
    assert client.delete(f"/api/ebsd/dataset/{copy}").status_code == 200
    assert variant_registry.variant_for(copy) is None
    assert variant_registry.variant_for(crop) == want


def test_batch_loader_takes_the_remembered_variant(client, multi):
    _load(client, multi, _v("1", "unprocessed"))
    assert ev.load_ebsd_file(multi) is True
    name = _name("multi", "unprocessed")
    assert ev._active_dataset == name
    assert _active_data().dtype == np.int16
    assert variant_registry.variant_for(name) == LoadVariant("1", "unprocessed")
    assert _entry(multi)["variant"] == _v("1", "unprocessed")
    assert ev.loaded_dataset_name(multi) == name


def test_loaded_dataset_name_is_the_plain_stem_for_the_default(client, multi):
    _load(client, multi)
    assert ev.loaded_dataset_name(multi) == "multi"
    assert ev.loaded_dataset_name("C:/never/loaded.h5oina") == "loaded"


# ----------------------------------------------------------------------- the answer's other fields


def test_has_raw_patterns_is_true_only_with_unprocessed(client, multi, processed_only):
    assert _load(client, multi).json()["has_raw_patterns"] is True
    assert _load(client, multi, _v("1", "unprocessed")).json()["has_raw_patterns"] is True
    assert _load(client, processed_only).json()["has_raw_patterns"] is False


def test_has_raw_patterns_follows_the_loaded_measurement(client, tmp_path):
    path = _file(tmp_path, "mixed.h5oina", [(1, {}), (2, {"unprocessed": False, "unprocessed_bg": False})])
    assert _load(client, path, _v("1", "processed")).json()["has_raw_patterns"] is True
    assert _load(client, path, _v("2", "processed")).json()["has_raw_patterns"] is False


def test_format_type_bruker_is_not_edax(client, bruker):
    body = _load(client, bruker).json()
    assert body["format_type"] == "Bruker"
    assert body["has_raw_patterns"] is False
    assert client.get("/api/ebsd/metadata").json()["format_type"] == "Bruker"


def test_format_type_of_oxford_files_is_unchanged(client, multi, tmp_path):
    assert _load(client, multi).json()["format_type"] == "Oxford"
    assert client.get("/api/ebsd/metadata").json()["format_type"] == "Oxford"
    # an Oxford layout under another extension was called EDAX (the mask followed): it is Oxford
    odd = _file(tmp_path, "oxford_layout.h5", [(1, {})])
    assert _load(client, odd).json()["format_type"] == "Oxford"


def test_format_type_of_edax_files_is_unchanged(client, tmp_path):
    up1 = tmp_path / "scan.up1"
    pats = np.random.default_rng(1).integers(0, 250, (16, 24, 24)).astype("u1")
    with open(up1, "wb") as f:
        f.write(struct.pack("<4I", 1, 24, 24, 16))
        f.write(pats.tobytes())
    body = _load(client, str(up1)).json()
    assert body["format_type"] == "EDAX"
    assert body["has_raw_patterns"] is False
    assert body["variant"] is None


def test_variant_is_ignored_for_a_file_without_choices(client, tmp_path):
    up1 = tmp_path / "scan.up1"
    pats = np.random.default_rng(1).integers(0, 250, (16, 24, 24)).astype("u1")
    with open(up1, "wb") as f:
        f.write(struct.pack("<4I", 1, 24, 24, 16))
        f.write(pats.tobytes())
    r = _load(client, str(up1), _v("1", "unprocessed"))
    assert r.status_code == 200, r.text
    assert r.json()["dataset_name"] == "scan"
    assert "variant" not in _entry(str(up1))


# ----------------------------------------------------------- a format with one source, several scans


def test_single_source_second_scan_loads_through_load(client, bruker):
    r = _load(client, bruker, _v("Scan 2", "single"))
    assert r.status_code == 200, r.text
    assert r.json()["dataset_name"] == _name("bruker", "Scan 2")
    assert r.json()["variant"] == {"measurement": "Scan 2", "pattern_source": "single", "is_default": False}
    assert np.array_equal(_active_data(), _bruker_patterns(2))
    _load(client, bruker)                      # the default: the first scan
    assert ev._active_dataset == "bruker"
    assert np.array_equal(_active_data(), _bruker_patterns(0))
    r = client.post("/api/ebsd/switch-file", json={"path": bruker})
    assert r.json()["dataset_name"] == "bruker"


def test_single_source_scan_survives_a_switch(client, bruker, other):
    _load(client, bruker, _v("Scan 2", "single"))
    _load(client, other)
    r = client.post("/api/ebsd/switch-file", json={"path": bruker})
    assert r.json()["dataset_name"] == _name("bruker", "Scan 2")
    assert np.array_equal(_active_data(), _bruker_patterns(2))


# ----------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("variant,code", [
    (_v("1", "bogus"), "loadVariantUnknown"),
    (_v("99", "processed"), "loadVariantUnknown"),
    (_v("1", "single"), "loadVariantUnknown"),
])
def test_unknown_variant_is_a_coded_400(client, multi, variant, code):
    r = _load(client, multi, variant)
    assert r.status_code == 400
    assert r.headers[CODE_HEADER] == code


def test_a_measurement_without_patterns_is_a_coded_400(client, tmp_path):
    path = _file(tmp_path, "nopat.h5oina", [(1, {}), (2, {"processed": False, "unprocessed": False})])
    r = _load(client, path, _v("2", "processed"))
    assert r.status_code == 400
    assert r.headers[CODE_HEADER] in ("loadNoPatterns", "loadVariantUnknown")


def test_a_refused_load_leaves_the_loaded_file_alone(client, multi, other):
    _load(client, other)
    assert _load(client, multi, _v("1", "bogus")).status_code == 400
    assert ev._active_dataset == "other"
    assert [e["path"] for e in ev._loaded_files] == [other]


# ---------------------------------------------------------- safe_loader: errors are not swallowed


def test_with_variant_error_is_not_swallowed_by_strategy_1(multi, monkeypatch):
    def boom(sig, path, resolved):
        raise LoadChoiceError("loadVariantUnknown", "from _with_variant")

    monkeypatch.setattr(safe_loader, "_with_variant", boom)
    fell_back = []
    import unified_loader
    real = unified_loader.EBSDDataLoader
    monkeypatch.setattr(unified_loader, "EBSDDataLoader",
                        lambda *a, **k: (fell_back.append(1), real(*a, **k))[1])
    with pytest.raises(LoadChoiceError) as ei:
        safe_loader.load_ebsd_safe(multi, verbose=False, variant=LoadVariant("1", "processed"))
    assert ei.value.code == "loadVariantUnknown"
    assert not fell_back, "the load fell back to the eager loader"


def test_other_errors_of_with_variant_surface_too(multi, monkeypatch):
    def boom(sig, path, resolved):
        raise OSError("disk went away")

    monkeypatch.setattr(safe_loader, "_with_variant", boom)
    with pytest.raises(OSError, match="disk went away"):
        safe_loader.load_ebsd_safe(multi, verbose=False, variant=LoadVariant("1", "processed"))


def test_with_variant_error_is_not_swallowed_by_the_layout_reader(tmp_path, monkeypatch):
    path = _file(tmp_path, "layout.h5", [(1, {})])      # Oxford layout under another extension
    calls = []

    def boom(sig, p, resolved):
        calls.append(1)
        raise LoadChoiceError("loadVariantUnknown", "from _with_variant")

    monkeypatch.setattr(safe_loader, "_with_variant", boom)
    real = safe_loader._kp().load

    def kp_fails(*a, **k):
        raise ValueError("kikuchipy refuses")

    monkeypatch.setattr(safe_loader._kp(), "load", kp_fails)
    with pytest.raises(LoadChoiceError):
        safe_loader.load_ebsd_safe(path, verbose=False, variant=LoadVariant("1", "processed"))
    assert calls == [1], "_with_variant ran again on another strategy"
    monkeypatch.setattr(safe_loader._kp(), "load", real)


# ------------------------------------------------------------------------------- codes and texts


def _literal_codes() -> set:
    found = set()
    for rel in ("backend/api/services/load_choices.py", "safe_loader.py", "backend/api/routes/ebsd_viewer.py",
                "backend/api/routes/indexing.py"):
        found |= set(re.findall(r'"(load[A-Z][A-Za-z]+|variantNotSupported)"', (ROOT / rel).read_text(encoding="utf-8")))
    return found


def test_load_codes_are_the_codes_the_backend_raises():
    assert set(LOAD_CODES) == _literal_codes()
    assert len(set(LOAD_CODES)) == len(LOAD_CODES)
    assert "loadVariantNeedsNativeReader" in LOAD_CODES


def test_load_codes_fixture_matches_the_backend():
    fixture = ROOT / "frontend/src/components/EBSDViewer/__fixtures__/loadCodes.json"
    assert json.loads(fixture.read_text(encoding="utf-8")) == {"codes": list(LOAD_CODES)}, (
        "load_choices.LOAD_CODES changed: regenerate loadCodes.json ({'codes': list(LOAD_CODES)}) "
        "and add the texts under errors.* of the four ebsdviewer.json files")


# ------------------------------------------------------------------------------------- the Oxford reference scan


def test_sampleb_unprocessed_through_load(client, monkeypatch):
    path = _sampleb()
    if path is None:
        pytest.skip("EBSD_SampleB real file is not available here")
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "1")
    r = _load(client, str(path), _v("1", "unprocessed"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_raw_patterns"] is True
    assert body["format_type"] == "Oxford"
    assert body["dataset_name"].endswith(f" {DOT} unprocessed")
    assert ev._raw_signals[body["dataset_name"]].data.dtype == np.int16
    body = _load(client, str(path)).json()
    assert body["has_raw_patterns"] is True and "unprocessed" not in body["dataset_name"]


# ------------------------------------------------------- results belong to a variant


@pytest.fixture
def results():
    from backend.api.routes import indexing
    saved, saved_active = list(indexing._result_registry.items()), indexing._active_result_id
    indexing._result_registry.clear()
    indexing._active_result_id = None
    yield indexing
    indexing._result_registry.clear()
    indexing._result_registry.update(saved)
    indexing._active_result_id = saved_active


def _store():
    from tests.test_crop_provenance import _store_dummy_result
    return _store_dummy_result()[0]


def test_a_result_is_reactivated_only_for_the_variant_it_ran_on(client, multi, results):
    assert _load(client, multi).status_code == 200
    rid = _store()                                              # a result on the default variant
    assert results._result_registry[rid].metadata.get("load_variant") is None
    _load(client, multi, _v("1", "unprocessed"))
    assert results._active_result_id is None, "a result of the processed patterns became active"
    _load(client, multi)
    assert results._active_result_id == rid
    # the explicit default is the default
    _load(client, multi, _v("1", "unprocessed"))
    _load(client, multi, _v("1", "processed"))
    assert results._active_result_id == rid


def test_a_result_on_the_unprocessed_variant_is_not_offered_to_the_default(client, multi, results):
    _load(client, multi, _v("1", "unprocessed"))
    rid = _store()
    assert results._result_registry[rid].metadata["load_variant"] == _v("1", "unprocessed")
    _load(client, multi)
    assert results._active_result_id is None
    r = client.post("/api/ebsd/switch-file", json={"path": multi})      # back to the default: still none
    assert results._active_result_id is None
    _load(client, multi, _v("1", "unprocessed"))
    assert results._active_result_id == rid


def test_another_measurement_does_not_pick_up_the_results_of_the_first(client, multi, results):
    _load(client, multi)
    first = _store()
    _load(client, multi, _v("2", "processed"))
    assert results._active_result_id is None
    second = _store()
    assert results._result_registry[second].metadata["load_variant"] == _v("2", "processed")
    _load(client, multi)
    assert results._active_result_id == first
    _load(client, multi, _v("2", "processed"))
    assert results._active_result_id == second


def test_a_result_without_the_stamp_counts_as_the_default(client, multi, results):
    _load(client, multi)
    rid = _store()
    results._result_registry[rid].metadata.pop("load_variant", None)    # a result of an older session
    _load(client, multi, _v("1", "unprocessed"))
    assert results._active_result_id is None
    _load(client, multi)
    assert results._active_result_id == rid


def test_an_imported_result_keeps_the_stamp_it_brought(client, multi, other, results):
    from tests.test_crop_provenance import _store_dummy_result
    _load(client, multi, _v("1", "unprocessed"))
    rid, res = _store_dummy_result({"source_file": other})              # another file's result
    assert res.metadata.get("load_variant") is None


# ------------------------------------------------ a refused switch drops the variant


def test_a_refused_switch_drops_the_stale_variant(client, multi, other, tmp_path):
    _load(client, multi, _v("2", "unprocessed"))
    _load(client, other)
    # the file no longer holds what the entry remembers (the entry is what a changed file leaves behind)
    _entry(multi)["variant"] = _v("99", "unprocessed")
    r = client.post("/api/ebsd/switch-file", json={"path": multi})
    assert r.status_code == 400 and r.headers[CODE_HEADER] == "loadVariantUnknown"
    assert "variant" not in _entry(multi)
    r = client.post("/api/ebsd/switch-file", json={"path": multi})
    assert r.status_code == 200, r.text
    assert r.json()["dataset_name"] == "multi"
    assert r.json()["variant"] == {"measurement": "1", "pattern_source": "processed", "is_default": True}

def test_info_tells_whether_the_loaded_measurement_holds_unprocessed_patterns(client, tmp_path):
    """/info is what the interface reads after a reload of the page: it said nothing, and the store
    filled in "yes" for every file."""
    path = _file(tmp_path, "mixed_info.h5oina", [(1, {}), (2, {"unprocessed": False, "unprocessed_bg": False})])
    _load(client, path, _v("1", "processed"))
    assert client.get("/api/ebsd/info").json()["has_raw_patterns"] is True
    _load(client, path, _v("2", "processed"))
    assert client.get("/api/ebsd/info").json()["has_raw_patterns"] is False
