"""A result imported from an export keeps its stored band contrast, whatever measurement it was indexed on.

The stamp (``indexed_variant``) of an imported result names a measurement of the ORIGINAL scan. The file the
result now has as ``source_file`` is the export: a light export holds no scan group at all, so a reader that
applied the stamp to it asked for a group that is not there and the quality map vanished ("no quality map").
A full export is a copy of the scan and does hold the group, so there the stamp still applies.

One place decides it: ``variant_registry.variant_for_source(result, path)``. These tests drive the real
export and the real import on the synthetic two-measurement scan of ``tests/two_measurements.py``, then the
readers that follow a result.
"""
from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api.services.load_choices import LoadChoiceError
from tests.test_indexing_export_light import restore_indexing_registry  # noqa: F401  (autouse)
from tests.two_measurements import *  # noqa: F401,F403  (constants and fixtures)
from tests.two_measurements import _grid, _xmap  # noqa: F401


def _result(stamp, source_file):
    from indexing_controller import IndexingMethod, IndexingResult
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    xmap = CrystalMap(rotations=Rotation.identity(N), phase_id=np.zeros(N, int),
                      x=np.tile(np.arange(NX, dtype=np.float32), NY),
                      y=np.repeat(np.arange(NY, dtype=np.float32), NX),
                      phase_list=PhaseList(Phase(name="Al", space_group=225, point_group="m-3m")))
    return IndexingResult(
        xmap=xmap, selection_mask=np.ones(N, bool), original_shape=(NY, NX),
        method=IndexingMethod.SPHERICAL, confidence_scores=np.full(N, 0.8, np.float32),
        metadata={"step_size_um": 0.5, "source_vendor": "oxford", "source_file": source_file,
                  "load_variant": stamp,
                  # per-phase data makes the export write /Indexing/Assignment, where the quality fields go
                  "per_phase_data": {"Al": {"ci": np.full(N, 0.8, np.float32), "ci_mean": 0.8}}})


@pytest.fixture
def imported(two, tmp_path, monkeypatch):
    """``do(fmt)``: a result indexed on measurement 2 of ``two``, exported as ``fmt`` and imported again.
    Returns the imported result and the exported file."""
    from backend.api.main import app
    import backend.api.routes.indexing as ix
    from backend.api.routes import ebsd_viewer as ev

    def do(fmt):
        monkeypatch.setattr(ev, "_ebsd_file_path", two, raising=False)
        ix._result_registry.clear()
        ix._result_registry["indexed"] = _result(V2.to_dict(), two)
        ix._active_result_id = "indexed"
        out = tmp_path / f"exported_{fmt}.h5"
        client = TestClient(app)
        resp = client.post("/api/indexing/export", json={"format": fmt, "filename": str(out)})
        assert resp.status_code == 200, resp.text
        ix._result_registry.clear()
        ix._active_result_id = None
        resp = client.post("/api/indexing/import-h5", json={"path": str(out)})
        assert resp.status_code == 200, resp.text
        res = ix._result_registry[ix._active_result_id]
        # premises: it is an import, it was indexed on measurement 2, and its source is the export
        assert res.metadata["imported_from_h5"] and res.metadata["source_file"] == str(out)
        assert res.metadata["indexed_variant"]["measurement"] == "2"
        return res, out

    return do


def test_a_light_export_holds_no_scan_group(imported):
    """The premise of all of it: the stamp names a group the file it is read from does not have."""
    _res, out = imported("h5_light")
    with h5py.File(out, "r") as f:
        assert "2" not in f and "1" not in f


def test_the_rule_drops_the_stamp_for_a_file_that_is_not_the_scan(imported):
    from backend.api.services import variant_registry as vr
    res, out = imported("h5_light")
    assert vr.variant_of_result(res) == V2                                # what the patterns were: unchanged
    assert vr.variant_for_source(res, str(out)) is None                  # the light file: the default reader
    assert vr.stamp_of_result(res)["measurement"] == "2"                 # and what an export states: unchanged


def test_the_rule_keeps_the_stamp_for_a_file_that_holds_the_scan(imported, two, tmp_path):
    from backend.api.services import variant_registry as vr
    res, out = imported("h5")
    with h5py.File(out, "r") as f:
        assert "2" in f
    assert vr.variant_for_source(res, str(out)) == V2
    # a path that is not the result's source is never touched by the rule, also when it lacks the
    # measurement (a reader of that file has to refuse, not read another one)
    assert vr.variant_for_source(res, two) == V2
    unrelated = tmp_path / "unrelated.h5"
    with h5py.File(unrelated, "w") as f:
        f.create_group("1")
    assert vr.variant_for_source(res, str(unrelated)) == V2
    # ... and the same for a light export asked about a file that is not it
    light, light_out = imported("h5_light")
    assert vr.variant_for_source(light, str(unrelated)) == V2


def test_the_rule_leaves_a_result_that_was_not_imported_alone(two):
    from backend.api.services import variant_registry as vr
    own = _result(V2.to_dict(), two)
    assert vr.variant_for_source(own, two) == V2
    own.metadata["source_file"] = two + ".gone"
    assert vr.variant_for_source(own, two + ".gone") == V2               # missing file: the reader refuses, as before
    assert vr.variant_for_source(_result(None, two), two) is None


def test_the_quality_map_of_an_imported_light_result_is_its_stored_band_contrast(imported):
    """get_quality_map with what the rule says: BC of measurement 2 from the xmap, not 'no quality map'."""
    from backend.api.services import variant_registry as vr
    from backend.api.services.pattern_quality import LABEL_NATIVE, get_quality_map
    res, out = imported("h5_light")
    qm = get_quality_map(NY, NX, source_file=res.metadata["source_file"], xmap=res.xmap, allow_compute=False,
                         variant=vr.variant_for_source(res, res.metadata["source_file"]))
    assert qm is not None and qm.label == LABEL_NATIVE
    np.testing.assert_array_equal(qm.array, _grid(BC[2]))


def test_the_phase_map_draws_the_stored_band_contrast_of_an_imported_light_result(imported):
    """The light import carries the band contrast on its xmap: the layer draws it, and does not go to a
    scan group the file lacks."""
    from unittest.mock import patch
    from backend.api.routes.phase_map import _compute_layer_rgba, _get_scalar_array_2d
    imported("h5_light")
    scale = {}
    with patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        _compute_layer_rgba(kind="bc", scale_out=scale)
        got = _get_scalar_array_2d("bc")
    assert (scale["min"], scale["max"]) == (0.0, 1.0)                    # the stored map, as a brightness factor
    assert got is not None and got.shape == (NY, NX)


def test_the_phase_map_reads_the_band_contrast_of_a_full_export_from_its_measurement(imported):
    """A full export is a copy of the scan: the stamp applies, and measurement 2's group is read."""
    from unittest.mock import patch
    from backend.api.routes.phase_map import _compute_layer_rgba
    res, _out = imported("h5")
    res.xmap.prop.pop("bc", None)                                        # nothing stored on the xmap
    scale = {}
    with patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        _compute_layer_rgba(kind="bc", scale_out=scale)
    assert (scale["min"], scale["max"]) == (min(BC[2]), max(BC[2]))


def test_the_addon_context_quality_of_an_imported_light_result_is_kept(imported):
    from backend.api.routes.addons import _context_extras
    res, _out = imported("h5_light")
    extras = _context_extras(res)
    np.testing.assert_array_equal(extras["quality"], _grid(BC[2]))
    assert extras["quality_source"] == "Band Contrast (native)"


def test_every_file_reader_of_a_result_asks_the_rule_for_its_variant(imported, monkeypatch):
    """The readers that open a result's own source file get the variant from the rule: for the light import
    that is none (the default reader), for the full export it is the measurement the result was indexed on."""
    from backend.api.main import app
    import backend.api.routes.analysis as ar
    import backend.api.routes.indexing as ix
    from backend.api.services import pattern_quality as pq
    seen = []

    def spy(src, n_rows=None, n_cols=None, variant=None):
        seen.append(variant)
        return None

    monkeypatch.setattr(pq, "read_native_band_contrast", spy)
    for fmt, expected in (("h5_light", None), ("h5", V2)):
        res, _out = imported(fmt)
        res.xmap.prop.pop("bc", None)
        saved = ar._dataset
        del seen[:]
        try:
            ar._dataset = None
            monkeypatch.setattr(ix, "get_last_indexing_result", lambda r=res: r)
            resp = TestClient(app).post("/api/analysis/load", json={"xmap_path": "__from_indexing__"})
            assert resp.status_code == 200, resp.text
        finally:
            ar._dataset = saved
        assert seen == [expected], (fmt, seen)


def test_a_light_export_of_an_imported_result_still_states_the_original_measurement(imported, tmp_path, monkeypatch):
    """The imported result is exported again with the light file open as the viewer's file: it must not be
    refused for lacking the group, and it must say measurement 2 (the statement is the stamp's)."""
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    res, out = imported("h5_light")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(out), raising=False)
    again = tmp_path / "again.h5"
    resp = TestClient(app).post("/api/indexing/export", json={"format": "h5_light", "filename": str(again)})
    assert resp.status_code == 200, resp.text
    with h5py.File(again, "r") as f:
        assert f["Indexing"].attrs["measurement"] == "2"


def test_a_result_that_is_not_imported_still_refuses_a_scan_without_its_measurement(two):
    """The rule is not a fall-through: the file of a result that ran on it must hold the measurement."""
    from backend.api.services import variant_registry as vr
    from backend.api.services.pattern_quality import read_native_band_contrast
    res = _result(V7.to_dict(), two)
    with pytest.raises(LoadChoiceError) as e:
        read_native_band_contrast(two, NY, NX, vr.variant_for_source(res, two))
    assert e.value.code == "variantNotSupported"


def test_the_phase_map_reads_its_own_source_with_the_variant_the_rule_gives(imported, monkeypatch):
    """The Phase Map BC layer, when the xmap carries nothing: the file it opens is the result's source, and
    the variant it passes is the rule's (none for a light export, the measurement for a full one)."""
    from unittest.mock import patch
    from backend.api.routes.phase_map import _compute_layer_rgba
    from backend.api.services import pattern_quality as pq
    seen = []
    real = pq.read_native_band_contrast

    def spy(src, n_rows=None, n_cols=None, variant=None):
        seen.append(variant)
        return real(src, n_rows, n_cols, variant)

    monkeypatch.setattr(pq, "read_native_band_contrast", spy)
    for fmt, expected in (("h5_light", None), ("h5", V2)):
        res, _out = imported(fmt)
        res.xmap.prop.pop("bc", None)
        del seen[:]
        with patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
            try:
                _compute_layer_rgba(kind="bc", scale_out={})
            except HTTPException:
                pass                                   # a light export with nothing stored has nothing to draw
        assert seen and seen[0] == expected, (fmt, seen)


def test_the_export_context_reads_the_header_the_rule_names(imported):
    """An imported result exported again with its own export as the viewer's file: no refusal, and the
    header of the scan is read where the scan is (the full export), not asked of a file without it."""
    from backend.api.routes.indexing import _export_context
    res, out = imported("h5_light")
    ctx = _export_context(res, str(out), "oxford")                       # was: LoadChoiceError
    assert ctx["geometry"] is None or ctx["geometry"].get("beam_voltage_kv") is None
    res, out = imported("h5")
    ctx = _export_context(res, str(out), "oxford")
    assert ctx["geometry"]["beam_voltage_kv"] == KV[2]


def test_the_session_fallback_reads_the_open_file_only_when_it_is_the_results_source(two, session, tmp_path):
    """The shared HDF5 session is read for a result's band contrast with the result's stamp only when the
    open file is the file the stamp describes."""
    from unittest.mock import patch
    from backend.api.routes.phase_map import _get_scalar_array_2d
    from tests.test_phase_map_layer_endpoint import _make_fake_result
    fake = _make_fake_result(n_rows=NY, n_cols=NX, with_bc=False)
    # another file than the open one is the result's: the open one is not read (the stamp names a
    # measurement of a file it does not describe)
    fake.metadata = {"source_file": str(tmp_path / "elsewhere.h5oina"), "load_variant": V2.to_dict()}
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        assert _get_scalar_array_2d("bc") is None
        # the Phase Map layer reads the same way: nothing to draw, rather than another file's map
        from backend.api.routes.phase_map import _compute_layer_rgba
        with pytest.raises(HTTPException):
            _compute_layer_rgba(kind="bc", scale_out={})
    # the open file is the result's source: its measurement 2 is read
    fake.metadata = {"source_file": two, "load_variant": V2.to_dict()}
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        np.testing.assert_array_equal(_get_scalar_array_2d("bc"), _grid(BC[2]))


# ------------------------------------------------------------------------------ activation, one rule

STAMPS = [None, V1.to_dict(), V2.to_dict(), V2_RAW.to_dict(), {"measurement": None, "pattern_source": "processed"}]


@pytest.mark.parametrize("stamp", STAMPS, ids=lambda s: "default" if s is None else f"{s['measurement']}-{s['pattern_source']}")
def test_for_a_result_that_ran_on_its_file_both_questions_have_one_answer(two, stamp):
    """What the patterns were and which variant of the file is loaded for the result: the same stamp."""
    from backend.api.services import variant_registry as vr
    res = _result(stamp, two)
    assert vr.file_stamp_of_result(res) == vr.stamp_of_result(res)


def test_for_an_imported_result_they_differ_on_purpose(imported):
    """The export is loaded as the default variant; the stamp says what its original scan was."""
    from backend.api.services import variant_registry as vr
    res, _out = imported("h5_light")
    assert vr.file_stamp_of_result(res) is None
    assert vr.stamp_of_result(res)["measurement"] == "2"


def test_activation_and_reactivation_ask_the_one_function(two, monkeypatch):
    """Both go through ``file_stamp_of_result``: with it saying measurement 2, a file loaded as the default
    is not the result's file (re-activation finds nothing, activation loads measurement 2); the key the
    viewer keeps for measurement 2 matches."""
    import backend.api.routes.indexing as ix
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services import variant_registry as vr
    from fastapi.testclient import TestClient
    from backend.api.main import app

    ix._result_registry.clear()
    ix._result_registry["r"] = _result(None, two)
    monkeypatch.setattr(vr, "file_stamp_of_result", lambda res: V2.to_dict())
    monkeypatch.setattr(ev, "_ebsd_file_path", two, raising=False)
    monkeypatch.setattr(ev, "_ebsd_variant_key", None, raising=False)
    assert ix.reactivate_result_for_source(two) is None
    assert ix.reactivate_result_for_source(two, ("2", "processed")) == "r"

    loaded = []

    async def fake_load(req):
        loaded.append(req.variant)
        monkeypatch.setattr(ev, "_ebsd_variant_key", ("2", "processed"), raising=False)   # what a real load leaves
        return {}

    monkeypatch.setattr(ev, "load_ebsd", fake_load)
    resp = TestClient(app).post("/api/indexing/results/activate/r")
    assert resp.status_code == 200, resp.text
    assert [(v.measurement, v.pattern_source) for v in loaded] == [("2", "processed")]


def test_an_activation_that_loads_hands_the_load_answer_on(two, monkeypatch):
    """The page that activated has to bring the stores to the file that came up: it needs the answer."""
    import backend.api.routes.indexing as ix
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app

    ix._result_registry.clear()
    ix._result_registry["r"] = _result(V2_RAW.to_dict(), two)
    monkeypatch.setattr(ev, "_ebsd_file_path", two, raising=False)
    monkeypatch.setattr(ev, "_ebsd_variant_key", ("2", "unprocessed"), raising=False)
    client = TestClient(app)
    assert client.post("/api/indexing/results/activate/r").json()["reloaded"] is None   # nothing was loaded

    async def fake_load(req):
        monkeypatch.setattr(ev, "_ebsd_variant_key", ("2", "unprocessed"), raising=False)
        return {"grid_shape": [NY, NX]}

    monkeypatch.setattr(ev, "_ebsd_variant_key", None, raising=False)
    monkeypatch.setattr(ev, "load_ebsd", fake_load)
    answer = client.post("/api/indexing/results/activate/r").json()
    assert answer["reloaded"] == {"grid_shape": [NY, NX]}


def test_a_light_export_activates_without_loading_anything(imported, two, monkeypatch):
    """A light export holds no patterns. Activating it a second time used to load it as a scan and
    end in a 500; the viewer stays on what it shows, as it did when the file was imported."""
    import backend.api.routes.indexing as ix
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app
    res, out = imported("h5_light")
    rid = ix._active_result_id
    monkeypatch.setattr(ev, "_ebsd_file_path", two, raising=False)
    ix._active_result_id = None
    resp = TestClient(app).post(f"/api/indexing/results/activate/{rid}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["reloaded"] is None
    assert ix._active_result_id == rid and ev._ebsd_file_path == two


@pytest.fixture
def clean_viewer():
    from tests.test_load_variant_routes import _reset
    _reset()
    yield
    _reset()


def test_an_imported_export_of_a_variant_activates_with_a_real_load(clean_viewer, imported, two):
    """The export marks measurement 2, so a load without a choice gives measurement 2, and that is what
    the result belongs to. Real export, real import, real load: it used to end in loadCameUpAsOther and
    an active result of nothing."""
    import backend.api.routes.indexing as ix
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app
    res, out = imported("h5")
    rid = ix._active_result_id
    assert not res.metadata.get("imported_without_patterns")
    client = TestClient(app)
    resp = client.post(f"/api/indexing/results/activate/{rid}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["reloaded"] is None                             # the import brought it up already
    assert client.post("/api/ebsd/load", json={"path": two}).status_code == 200
    resp = client.post(f"/api/indexing/results/activate/{rid}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["reloaded"]["file_path"] == str(out)
    assert ix._active_result_id == rid and ev._ebsd_file_path == str(out)


def test_a_pc_typed_after_the_import_of_a_variant_reaches_the_store(clean_viewer, imported):
    """The calibration was registered under the bare stem, the dataset on screen is named with its
    variant: the PC route answered 200, nothing was stored, and the next run took the file's PC."""
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app
    from backend.api.services.calibration_store import calibration_store
    imported("h5")
    assert " \u00b7 " in ev._active_dataset
    resp = TestClient(app).post("/api/pc/detector/update-pc", json={"pcx": 0.3, "pcy": 0.31, "pcz": 0.32})
    assert resp.status_code == 200, resp.text
    assert calibration_store.get_pc(ev._active_dataset) == pytest.approx([0.3, 0.31, 0.32])


def test_the_import_registers_the_calibration_with_the_file_it_loaded(clean_viewer, imported):
    """A stem names two files in two folders: an entry without its file is no file's, and a batch
    run that asks by path would not find the PC of this one."""
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services.calibration_store import calibration_store
    _, out = imported("h5")
    assert calibration_store.name_of_file(ev._canonical_path(str(out))) == ev._active_dataset


def test_a_default_result_on_a_marked_file_comes_back(clean_viewer, imported):
    """No stamp is the file's default variant. A plain load of a marked file gives the marked one, so
    the activation has to ask for the default by name."""
    import backend.api.routes.indexing as ix
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app
    _res, out = imported("h5")
    ix._result_registry.clear()
    ix._active_result_id = None
    client = TestClient(app)
    assert client.post("/api/ebsd/load", json={"path": str(out), "variant": V1.to_dict()}).status_code == 200
    ix._result_registry["own"] = _result(ix._loaded_variant_stamp(str(out)), str(out))
    assert client.post("/api/ebsd/load", json={"path": str(out), "variant": V2.to_dict()}).status_code == 200
    resp = client.post("/api/indexing/results/activate/own")
    assert resp.status_code == 200, resp.text
    assert ev._ebsd_variant_key is None and ix._active_result_id == "own"


def test_the_list_says_when_an_activation_would_load(clean_viewer, two):
    """A page visit may adopt a stored result, but not by loading the variant the user just left."""
    import backend.api.routes.indexing as ix
    from backend.api.main import app
    client = TestClient(app)
    assert client.post("/api/ebsd/load", json={"path": two}).status_code == 200
    ix._result_registry.clear()
    ix._result_registry["default"] = _result(None, two)
    ix._result_registry["measurement-2"] = _result(V2_RAW.to_dict(), two)

    def loads():
        listed = client.get("/api/indexing/results").json()["results"]
        return {r["id"]: r["activation_loads"] for r in listed}

    assert loads() == {"default": False, "measurement-2": True}
    assert client.post("/api/ebsd/load", json={"path": two, "variant": V2_RAW.to_dict()}).status_code == 200
    assert loads() == {"default": True, "measurement-2": False}


def test_info_names_an_open_file_that_has_no_signal(monkeypatch):
    """An EDS-only file is open without a signal. A page that must not replace an open file has to
    see it; ``file_path`` stays what it was for the readers that take it for a signal's file."""
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: None)
    monkeypatch.setattr(ev, "_ebsd_file_path", None, raising=False)
    assert TestClient(app).get("/api/ebsd/info").json() == {"loaded": False}
    monkeypatch.setattr(ev, "_ebsd_file_path", "D:/eds-only.h5oina", raising=False)
    assert TestClient(app).get("/api/ebsd/info").json() == {"loaded": False, "open_file": "D:/eds-only.h5oina"}


def test_the_import_asks_the_load_choices_whether_a_file_holds_patterns(imported, monkeypatch):
    """The import's own look knows two layouts. What it answers decides whether the viewer follows."""
    import backend.api.routes.indexing as ix
    res, out = imported("h5_light")
    assert res.metadata["imported_without_patterns"]
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app
    monkeypatch.setattr(ix, "_file_holds_patterns", lambda path, found: True)
    monkeypatch.setattr(ev, "load_ebsd_file", lambda path: False)
    answer = TestClient(app).post("/api/indexing/import-h5", json={"path": str(out)}).json()
    assert answer["capabilities"]["patterns"] is True
    assert not ix._result_registry[ix._active_result_id].metadata["imported_without_patterns"]


def test_a_damaged_file_does_not_take_the_list_with_it(tmp_path):
    import backend.api.routes.indexing as ix
    from backend.api.main import app
    path = tmp_path / "damaged.h5oina"
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("Oxford Instruments"))
        f.create_group("1/EBSD/Data").create_dataset("Processed Patterns", data=np.zeros((6, 4, 5), np.uint8))
        f["1/EBSD"].create_dataset("Header", data=1)
    ix._result_registry.clear()
    ix._result_registry["r"] = _result(V2_RAW.to_dict(), str(path))
    resp = TestClient(app, raise_server_exceptions=False).get("/api/indexing/results")
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"][0]["variant_parts"] == ["unprocessed"]


@pytest.mark.parametrize("top,dataset,manufacturer,shape", [
    ("Scan 1", "RawPatterns", "Bruker Nano", (6, 4, 5)),
    ("Scan 1", "patterns", "kikuchipy", (6, 4, 5)),
    ("1", "Processed Patterns", "Oxford Instruments", (2, 3, 4, 5)),
], ids=["bruker", "kikuchipy", "oxford-4d"])
def test_an_export_of_another_layout_counts_as_holding_patterns(tmp_path, top, dataset, manufacturer, shape):
    from backend.api.routes.indexing import _file_holds_patterns
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_(manufacturer))
        f.create_group(f"{top}/EBSD/Header")
        f.create_dataset(f"{top}/EBSD/Data/{dataset}", data=np.zeros(shape, np.uint8))
    assert _file_holds_patterns(str(path), False)
    assert not _file_holds_patterns(str(tmp_path / "gone.h5"), False)
    assert _file_holds_patterns(str(tmp_path / "gone.h5"), True)


def test_the_list_says_which_patterns_a_result_ran_on(two, tmp_path, monkeypatch):
    """The gallery shows results of one file side by side; only this tells them apart. The parts are
    those of the dataset name: nothing for the default variant, what differs from it otherwise."""
    import backend.api.routes.indexing as ix
    from backend.api.main import app

    def parts(stamp, source):
        ix._result_registry.clear()
        ix._result_registry["r"] = _result(stamp, source)
        return TestClient(app).get("/api/indexing/results").json()["results"][0]["variant_parts"]

    assert parts(None, two) == []
    assert parts(V2.to_dict(), two) == ["2"]
    assert parts(V2_RAW.to_dict(), two) == ["2", "unprocessed"]
    assert parts({"measurement": "1", "pattern_source": "unprocessed"}, two) == ["unprocessed"]
    # a file that no longer lists its variants (gone, or a light export): the stamp still says the source
    assert parts(V2_RAW.to_dict(), str(tmp_path / "gone.h5oina")) == ["unprocessed"]
    assert parts(V2.to_dict(), str(tmp_path / "gone.h5oina")) == []


def test_the_open_file_of_an_imported_light_result_is_read_without_a_measurement(imported, monkeypatch):
    """The viewer's session has the result's own (light) export open: it is read with the variant the
    rule gives for that file, which is none, not with the stamp of a scan the file does not hold."""
    from backend.api.routes import phase_map, virtual_images
    from backend.api.services import h5_session
    res, out = imported("h5_light")
    seen = []
    monkeypatch.setattr(virtual_images, "_read_native_band_contrast",
                        lambda variant=None: seen.append(variant))
    h5_session.open_file(str(out))
    try:
        phase_map._band_contrast_of_open_file(res)
    finally:
        h5_session.close_file()
    assert seen == [None]


# ------------------------------------------------------------------ the PC paths built since the fix

TYPED_PC = [0.3, 0.31, 0.32]


@pytest.fixture
def calibrations(tmp_path, monkeypatch):
    """Calibrations only in a temporary folder; the step back and the memory of applications start empty."""
    from backend.api.routes import pcrefinement
    from backend.api.services import pc_calibration_apply as use
    from backend.api.services import pc_calibrations as cal
    monkeypatch.setattr(pcrefinement, "_optimization_active", False)
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    use.forget_all()
    pcrefinement._pc_before.clear()
    yield cal
    use.forget_all()
    pcrefinement._pc_before.clear()
    cal.set_calibrations_dir_for_test(None)


def _calibration_of_the_variant_on_screen(client, name):
    """Save the PC the store holds for ``name`` as a calibration; returns (id, that PC)."""
    from backend.api.services.calibration_store import calibration_store
    file_pc = [float(v) for v in calibration_store.get_pc(name)]
    resp = client.post("/api/pc-calibrations", json={"name": "import test", "dataset": name})
    assert resp.status_code == 200, resp.text
    assert resp.json()["pc"] == pytest.approx(file_pc)
    return resp.json()["id"], file_pc


def _type_pc(client, name):
    from backend.api.services.calibration_store import calibration_store
    resp = client.post("/api/pc/detector/update-pc", json={"pcx": TYPED_PC[0], "pcy": TYPED_PC[1], "pcz": TYPED_PC[2]})
    assert resp.status_code == 200, resp.text
    assert calibration_store.get_pc(name) == pytest.approx(TYPED_PC)


def test_a_calibration_applied_after_the_import_of_a_variant_reaches_the_store(clean_viewer, calibrations, imported):
    """Save, type, apply: all three name the dataset on screen, which carries the variant. The apply answered
    ``applied`` only for a name the store knows; the store has to hold the calibration's PC afterwards."""
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app
    from backend.api.services.calibration_store import calibration_store
    imported("h5")
    name = ev._active_dataset
    assert " \u00b7 " in name
    client = TestClient(app)
    cal_id, file_pc = _calibration_of_the_variant_on_screen(client, name)
    _type_pc(client, name)
    assert calibration_store.get_pc(name) != pytest.approx(file_pc)               # the apply has something to change
    resp = client.post(f"/api/pc-calibrations/{cal_id}/apply", json={"datasets": [name], "accept_mismatch": [name]})
    assert resp.status_code == 200, resp.text
    got = resp.json()["results"]
    assert [(r["dataset"], r["status"]) for r in got] == [(name, "applied")], got
    assert got[0]["previous"]["pc"] == pytest.approx(TYPED_PC)
    assert calibration_store.get_pc(name) == pytest.approx(file_pc)
    assert calibration_store.get_entry(name).pc_source == "calibration"


def test_put_back_after_a_calibration_applied_to_a_variant_returns_the_typed_pc(clean_viewer, calibrations, imported):
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.main import app
    from backend.api.services.calibration_store import calibration_store
    imported("h5")
    name = ev._active_dataset
    client = TestClient(app)
    cal_id, _file_pc = _calibration_of_the_variant_on_screen(client, name)
    _type_pc(client, name)
    resp = client.post(f"/api/pc-calibrations/{cal_id}/apply", json={"datasets": [name], "accept_mismatch": [name]})
    assert resp.status_code == 200 and resp.json()["results"][0]["status"] == "applied", resp.text
    assert calibration_store.get_pc(name) != pytest.approx(TYPED_PC)
    resp = client.post("/api/pc/put-back")
    assert resp.status_code == 200, resp.text
    assert resp.json()["restored"] == [name]
    assert calibration_store.get_pc(name) == pytest.approx(TYPED_PC)
    assert calibration_store.get_entry(name).pc_source == "manual"


def test_a_calibration_applied_to_a_name_the_store_does_not_know_claims_nothing(clean_viewer, calibrations, imported):
    """The bare stem is what the import used to register. Naming it now is naming nothing: ``notLoaded``,
    and the PC, the step back, the calibration's members and the memory of applications stay as they were."""
    from backend.api.routes import ebsd_viewer as ev, pcrefinement
    from backend.api.main import app
    from backend.api.services import pc_calibration_apply as use
    from backend.api.services.calibration_store import calibration_store
    imported("h5")
    name = ev._active_dataset
    bare = name.split(" \u00b7 ")[0]
    assert bare != name and calibration_store.get_entry(bare) is None             # premise
    client = TestClient(app)
    cal_id, file_pc = _calibration_of_the_variant_on_screen(client, name)
    _type_pc(client, name)
    members = calibrations.get(cal_id)["members"]
    step = dict(pcrefinement._pc_before)
    names = sorted(calibration_store.get_all())
    got = use.apply(cal_id, [bare], accept_mismatch=[bare])
    assert got == [{"dataset": bare, "status": "notLoaded"}]
    assert calibration_store.get_pc(name) == pytest.approx(TYPED_PC)
    assert sorted(calibration_store.get_all()) == names
    assert pcrefinement._pc_before == step
    assert calibrations.get(cal_id)["members"] == members
    assert use.usage_of(bare) is None
