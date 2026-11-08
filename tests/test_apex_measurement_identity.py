"""Which measurement of which file: an EDAX pattern file paired by hand with a measurement.

A pattern file that names no measurement (a header without a GUID) is paired with a measurement of a
metadata file by the user, and the same pattern file can be paired with another measurement of the same
metadata file later. "This file with measurement 2" is then another variant of the file than "this file
with measurement 1", exactly as "this file, unprocessed" is another variant than "this file": its stash
(crops, derived datasets, a typed PC), its analysis dataset, its results, its dataset name and its
loaded-files entry are its own. Everything runs through the real routes on small synthetic files.

By hand: the metadata file holds measurement 1 (hexagonal 3 rows x 5 columns, PC of the file
(0.5, 0.4, 0.6)) and measurement 2 (hexagonal 5 rows x 3 columns, PC (0.45, 0.35, 0.7)); both want 15
patterns, so the one pattern file fits both. The file states the PC in EDAX/TSL form; on a square 8 x 8
detector Bruker's is (x, 1 - y, z): (0.5, 0.6, 0.6) and (0.45, 0.65, 0.7).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import analysis as analysis_routes  # noqa: E402
from backend.api.routes import ebsd_viewer as ev  # noqa: E402
from backend.api.routes import indexing as ind  # noqa: E402
from backend.api.services import crop_window, variant_registry  # noqa: E402
from backend.api.services.calibration_store import calibration_store  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_A_BYTES, GUID_B, make_apex_h5, make_up, measurement_spec, wipe_backend,
)

PC_1 = [0.5, 0.6, 0.6]
PC_2 = [0.45, 0.65, 0.7]
#: The dataset of a pattern file is called after the file; the pairing is not part of the name (a file
#: with one pairing is named as it always was), the stash key and the loaded-files entry carry it.
NAME_1 = NAME_2 = "noguid"


def _reset() -> None:
    wipe_backend()
    ev._ebsd_variant_key = None
    ev._ebsd_apex = None
    ev._pc_load_notice.clear()
    variant_registry.clear_all()
    ind._result_registry.clear()
    ind._active_result_id = None
    analysis_routes._dataset = None
    analysis_routes._dataset_by_file.clear()


@pytest.fixture(autouse=True)
def fresh_backend():
    _reset()
    yield
    _reset()


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture
def files(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    h5 = make_apex_h5(d / "s.h5", [
        measurement_spec(n_rows=3, n_cols=5, step_x=0.5),                                   # measurement 1
        measurement_spec(id=GUID_B, name="OIM Map 2", pc=(0.45, 0.35, 0.7), grid="HexGrid",
                         n_rows=5, n_cols=3, step_x=0.5),                                   # measurement 2
    ])
    return SimpleNamespace(
        h5=str(h5),
        up=str(make_up(d / "noguid.up1", n=15, guid_bytes=None, version=3, nx=5, ny=3, hex_flag=1)),
        other=str(make_up(d / "other.up1", n=6, guid_bytes=None, version=3, nx=3, ny=2, hex_flag=0)),
        named=str(make_up(d / "named.up1", n=15, guid_bytes=GUID_A_BYTES, version=4, nx=5, ny=3,
                          hex_flag=1)),
    )


def _load(client, path, h5=None, mid=None):
    body = {"path": path}
    if h5:
        body["apex"] = {"h5_path": h5, "measurement_id": mid}
    r = client.post("/api/ebsd/load", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _m1(client, f):
    return _load(client, f.up, f.h5, GUID_A)


def _m2(client, f):
    return _load(client, f.up, f.h5, GUID_B)


def _datasets(client):
    body = client.get("/api/ebsd/datasets").json()
    return {d["name"]: d for d in body["datasets"]}, body["active"]


def _pc(name):
    return [round(float(v), 6) for v in calibration_store.get_entry(name).pc_single]


def _entry(client, path):
    return next(e for e in client.get("/api/ebsd/loaded-files").json()["files"] if e["path"] == path)


def _crop(client, name="c1"):
    r = client.post("/api/ebsd/crop", json={"shape": "rect", "row0": 1, "col0": 0, "rows": 2, "cols": 4,
                                            "name": name})
    assert r.status_code == 200, r.text


def _store():
    from tests.test_crop_provenance import _store_dummy_result
    return _store_dummy_result()[0]


# ----------------------------------------------------------------- reloading a file with another measurement

def test_reloading_the_file_with_another_measurement_starts_from_that_measurement(client, files):
    first = _m1(client, files)
    assert first["dataset_name"] == NAME_1 and first["grid_shape"] == [3, 5]
    _crop(client)
    assert crop_window.get_crop("c1") is not None
    _load(client, files.other)
    second = _m2(client, files)
    # what the file says, what the registries hold and what indexing would use are one measurement
    assert second["dataset_name"] == NAME_2 and second["apex"]["measurement_id"] == GUID_B
    assert second["grid_shape"] == [5, 3] and second["scan_geometry"]["grid_shape"] == [5, 3]
    assert second["pc"] == pytest.approx(PC_2)
    datasets, active = _datasets(client)
    assert list(datasets) == [NAME_2] and active == NAME_2
    assert datasets[NAME_2]["scan_geometry"]["grid_shape"] == [5, 3]
    assert datasets[NAME_2]["navigation_shape"] == [3, 5]
    assert crop_window.get_crop("c1") is None                 # measurement 1's crop is not here
    assert _pc(NAME_2) == pytest.approx(PC_2)
    assert client.get("/api/ebsd/info").json()["active_dataset"] == NAME_2
    assert _entry(client, files.up)["apex"]["measurement_id"] == GUID_B


def test_going_back_to_the_first_measurement_brings_its_crop_and_its_pc_back(client, files):
    _m1(client, files)
    calibration_store.update_pc(NAME_1, np.array([0.31, 0.42, 0.53]), source="manual")   # typed by the user
    _crop(client)
    _load(client, files.other)
    _m2(client, files)
    _m2_pc_before = _pc(NAME_2)
    back = _m1(client, files)
    datasets, active = _datasets(client)
    assert set(datasets) == {NAME_1, "c1"} and active == "c1"
    # the answer describes the dataset that is active after the restore: the crop, on its own grid
    assert back["dataset_name"] == "c1" and back["grid_shape"] == [2, 4]
    window = crop_window.get_crop("c1")
    assert (window.row0, window.col0, window.rows, window.cols) == (1, 0, 2, 4)
    assert tuple(window.original_shape) == (3, 5)             # cut from the 3 x 5 grid of measurement 1
    assert _pc(NAME_1) == pytest.approx([0.31, 0.42, 0.53])   # the typed PC, not the file's
    assert _pc("c1") == pytest.approx([0.31, 0.42, 0.53])
    assert _m2_pc_before == pytest.approx(PC_2)
    # and measurement 2 again: nothing of measurement 1 in it
    _load(client, files.other)
    _m2(client, files)
    datasets, active = _datasets(client)
    assert list(datasets) == [NAME_2] and _pc(NAME_2) == pytest.approx(PC_2)


@pytest.mark.parametrize("which", ["clear", "clear-all"])
def test_clearing_the_list_keeps_what_the_other_measurement_of_the_active_file_left_behind(client, files, which):
    """"The active file is not affected" (the user guide): the crop of measurement 1 belongs to the file
    that stays open while measurement 2 is the one on screen. "Clear all" discarded it with the files
    that really left the list."""
    _m1(client, files)
    _crop(client)                                                     # c1, cut from measurement 1
    _m2(client, files)
    assert any(isinstance(k, tuple) and k[1] == (None, None, GUID_A) for k in ev._registry_by_file), \
        "premise: measurement 1 sits in the stash"
    assert client.post(f"/api/ebsd/loaded-files/{which}").status_code == 200
    _m1(client, files)
    datasets, active = _datasets(client)
    assert set(datasets) == {NAME_1, "c1"} and active == "c1", "the crop of measurement 1 was discarded"


def test_the_loaded_files_entry_tells_the_two_measurements_apart(client, files):
    _m1(client, files)
    one = dict(_entry(client, files.up))
    _m2(client, files)
    two = dict(_entry(client, files.up))
    assert (one["apex"]["measurement_id"], two["apex"]["measurement_id"]) == (GUID_A, GUID_B)
    assert (one["apex"]["measurement_name"], two["apex"]["measurement_name"]) == (
        "Area 1 / OIM Map 1", "Area 1 / OIM Map 2")
    assert one["apex"] != two["apex"] and one["apex"]["pairing"] == two["apex"]["pairing"] == "count"
    # a switch back reloads the measurement the file was last loaded with
    _load(client, files.other)
    back = client.post("/api/ebsd/switch-file", json={"path": files.up})
    assert back.status_code == 200, back.text
    assert back.json()["dataset_name"] == NAME_2 and back.json()["grid_shape"] == [5, 3]
    assert len(client.get("/api/ebsd/loaded-files").json()["files"]) == 2     # one entry per file


# ------------------------------------------------------------------------- results and analysis

def test_a_result_of_the_first_measurement_is_not_reactivated_under_the_second(client, files):
    _m1(client, files)
    first = _store()
    assert ev._ebsd_variant_key == (None, None, GUID_A)
    assert ind._result_registry[first].metadata["load_variant"]["apex"]["measurement_id"] == GUID_A
    _m2(client, files)
    assert ind._active_result_id is None
    second = _store()
    assert ind._result_registry[second].metadata["load_variant"]["apex"]["measurement_id"] == GUID_B
    _m1(client, files)
    assert ind._active_result_id == first
    _m2(client, files)
    assert ind._active_result_id == second
    _load(client, files.other)
    assert ind._active_result_id is None


def test_activating_the_result_of_the_first_measurement_loads_the_first_measurement(client, files):
    _m1(client, files)
    first = _store()
    _m2(client, files)
    assert ind._active_result_id is None
    r = client.post(f"/api/indexing/results/activate/{first}")
    assert r.status_code == 200, r.text
    assert ind._active_result_id == first
    assert _entry(client, files.up)["apex"]["measurement_id"] == GUID_A
    datasets, active = _datasets(client)
    assert active == NAME_1 and datasets[NAME_1]["scan_geometry"]["grid_shape"] == [3, 5]
    assert _pc(NAME_1) == pytest.approx(PC_1)


def test_the_analysis_dataset_of_the_first_measurement_is_not_restored_under_the_second(client, files):
    sentinel = object()
    _m1(client, files)
    analysis_routes._dataset = sentinel
    _load(client, files.other)
    _m2(client, files)
    assert analysis_routes._dataset is None
    other_sentinel = object()
    analysis_routes._dataset = other_sentinel
    _load(client, files.other)
    _m1(client, files)
    assert analysis_routes._dataset is sentinel
    _load(client, files.other)
    _m2(client, files)
    assert analysis_routes._dataset is other_sentinel


def test_the_batch_loader_keeps_the_pairing_apart_too(client, files):
    _m1(client, files)
    _crop(client)
    _load(client, files.other)
    assert ev.load_ebsd_file(files.up) is True               # the remembered pairing: measurement 1
    assert ev._ebsd_variant_key == (None, None, GUID_A)
    assert ev._active_dataset == "c1"                        # its stash came back
    _m2(client, files)
    assert ev._ebsd_variant_key == (None, None, GUID_B)
    _load(client, files.other)
    assert ev.load_ebsd_file(files.up) is True               # now the file was last loaded with 2
    assert ev._ebsd_variant_key == (None, None, GUID_B) and ev._active_dataset == NAME_2


def test_the_pairing_of_a_file_that_was_left_is_not_kept_for_an_eds_only_file(client, files, tmp_path):
    """The loaded pairing describes the loaded file, and a file without patterns has none."""
    import h5py
    eds_only = tmp_path / "eds_only.h5oina"
    with h5py.File(eds_only, "w") as f:
        g = f.create_group("1")
        g.create_group("EDS/Data/Window Integral").create_dataset("Al Ka1", data=np.zeros(12, np.float32))
        g.create_group("Electron Image/Data").create_dataset("SE", data=np.zeros((4, 4), np.uint8))
    _m2(client, files)
    assert ev._ebsd_apex is not None and ev._ebsd_apex["measurement_id"] == GUID_B     # premise
    _load(client, str(eds_only))
    assert ev._ebsd_variant_key is None and ev._ebsd_apex is None


def test_a_batch_loaded_file_stamps_its_results_with_the_pairing_it_was_loaded_with(client, files):
    """The batch loader sets the loaded pairing beside the key, as the interactive load does: the file
    before it (another file, no pairing) must not be what the stamp reads."""
    _m2(client, files)
    h5 = _entry(client, files.up)["apex"]["h5_path"]
    _load(client, files.other)                               # no pairing is loaded now
    assert ev.load_ebsd_file(files.up) is True               # the remembered one: measurement 2
    assert _stamp_of(_store())["apex"] == {"h5_path": h5, "measurement_id": GUID_B}


# --------------------------------------------------------------------- what keeps its keys

def test_a_pattern_file_that_names_its_measurement_keeps_its_plain_keys(client, files):
    answer = _load(client, files.named, files.h5, GUID_A)
    assert answer["apex"]["pairing"] == "guid"
    assert answer["dataset_name"] == "named" and ev._ebsd_variant_key is None
    rid = _store()
    assert ind._result_registry[rid].metadata["load_variant"] is None
    _load(client, files.other)
    _load(client, files.named, files.h5, GUID_A)
    assert ind._active_result_id == rid
    assert _entry(client, files.named)["name"] == "named"


def test_a_pattern_file_loaded_without_metadata_keeps_its_plain_keys(client, files):
    answer = _load(client, files.other)
    assert answer["dataset_name"] == "other" and ev._ebsd_variant_key is None
    assert ind._result_registry == {} and _entry(client, files.other)["name"] == "other"


def test_an_export_writes_the_variant_of_a_result_and_never_a_pairing_as_a_pattern_source():
    paired = {"measurement": None, "pattern_source": None, "apex": {"h5_path": "x.h5", "measurement_id": "g"}}
    group = SimpleNamespace(attrs={})
    ind._write_variant_attrs(group, SimpleNamespace(metadata={"load_variant": paired}))
    assert group.attrs == {}                                  # a metadata pairing is no HDF5 attribute
    group = SimpleNamespace(attrs={})
    ind._write_variant_attrs(group, SimpleNamespace(metadata={"load_variant": {
        "measurement": "2", "pattern_source": "unprocessed"}}))
    assert group.attrs == {"measurement": "2", "pattern_source": "unprocessed"}


def test_a_pairing_that_the_file_names_is_not_part_of_a_stamp_and_a_made_one_is():
    plain = {"measurement": "1", "pattern_source": "unprocessed"}
    assert ind._stamp_key(plain) == ("1", "unprocessed")
    assert ind._stamp_key(None) is None
    paired = {"measurement": None, "pattern_source": None, "apex": {"h5_path": "x.h5", "measurement_id": "g"}}
    assert ind._stamp_key(paired) == (None, None, "g")
    both = {**plain, "apex": {"h5_path": "x.h5", "measurement_id": "g"}}
    assert ind._stamp_key(both) == ("1", "unprocessed", "g")
    # the stamp of a pairing alone is a stamp (the default variant of nothing), and no variant to load
    result = SimpleNamespace(metadata={"load_variant": paired})
    assert variant_registry.stamp_of_result(result) == paired
    assert variant_registry.file_stamp_of_result(result) == paired
    assert variant_registry.variant_of_result(result) is None


def test_the_cross_file_pattern_match_reads_the_pairing_the_result_ran_on(client, files):
    """The file remembers the pairing it was LAST loaded with (measurement 2); a result of measurement 1
    is compared with the patterns of measurement 1. Pattern i of the made-up file is the constant
    (7 * i) % 256; the hexagonal grid of measurement 1 is 3 x 5, that of measurement 2 is 5 x 3."""
    from tools.pattern_comparison import _source_signal_cache, get_experimental_pattern
    _source_signal_cache.clear()
    _m1(client, files)
    first = _store()
    _m2(client, files)
    second = _store()
    _load(client, files.other)                               # another file is active: the match must re-read
    one, two = ind._result_registry[first], ind._result_registry[second]
    p1 = get_experimental_pattern(one, 0, 4)                 # row 0, column 4 of 3 x 5: pattern 4
    assert p1 is not None and np.all(p1 == (7 * 4) % 256)
    p2 = get_experimental_pattern(two, 4, 2)                 # row 4, column 2 of 5 x 3: pattern 14
    assert p2 is not None and np.all(p2 == (7 * 14) % 256)
    assert get_experimental_pattern(two, 0, 4) is None       # column 4 does not exist in 5 x 3
    _source_signal_cache.clear()


def test_the_pc_page_keeps_one_session_per_measurement_of_the_file(client, files):
    """Both datasets are called "noguid" and sit on the same file; a PC-page session of one must not
    serve the other (its calibration patterns), and comes back with its measurement. Through the two
    places that decide it, ``_get_session`` (every request of the page) and ``resync_sessions_with_store``
    (after every load): ignoring the pairing in either turns this red."""
    from kikuchipy.detectors import EBSDDetector

    from backend.api.routes import pcrefinement as pcr
    pcr._sessions.clear()
    pcr._parked_sessions.clear()
    try:
        _m1(client, files)
        mine = pcr._get_session()
        key_1 = pcr._current_file_key()
        assert key_1.endswith(f"|{GUID_A}") and mine.file_key == key_1
        stale = [0.3, 0.3, 0.3]                         # a PC the page holds that the store does not
        mine.ctrl.detector = EBSDDetector(shape=(8, 8), pc=stale, sample_tilt=70.0)
        _m2(client, files)
        key_2 = pcr._current_file_key()
        assert key_1 != key_2 and key_1.split("|")[0] == key_2.split("|")[0]    # one file, two keys
        # the load resynced the sessions: the one of measurement 1 is not measurement 2's, untouched
        assert np.allclose(np.asarray(mine.ctrl.detector.pc).ravel(), stale)
        theirs = pcr._get_session()                     # the request of the page after the load
        assert theirs is not mine and theirs.file_key == key_2
        _m1(client, files)
        assert pcr._get_session() is mine               # comes back with its measurement ...
        # ... and follows the store (the resync after the load saw it as the loaded measurement's)
        assert np.allclose(np.asarray(mine.ctrl.detector.pc).ravel(), PC_1)
        _load(client, files.other)
        assert "|" not in pcr._current_file_key()                               # no pairing, plain key
    finally:
        pcr._sessions.clear()
        pcr._parked_sessions.clear()


def test_a_paired_pattern_file_has_no_phase_map_store_to_mix_up(client, files):
    """The phase-map store is keyed by file and HDF5 measurement; a pattern file opens no HDF5 session,
    so a pairing never reaches it."""
    from backend.api.services import h5_session
    _m1(client, files)
    _m2(client, files)
    assert h5_session.is_open() is False and h5_session.active_measurement() is None
    assert h5_session.sync_phase_map_store() is False


# ------------------------------------------------- a stamp takes its pairing from what is LOADED

def _stamp_of(rid):
    return ind._result_registry[rid].metadata["load_variant"]


def _clear_all(client):
    assert client.post("/api/ebsd/loaded-files/clear-all").status_code == 200
    assert client.get("/api/ebsd/loaded-files").json()["files"] == []          # premise: no entry left


def test_a_result_started_after_clear_all_still_names_the_metadata_file(client, files):
    """"Clear all" empties the list and keeps the active file open; the entry that held the metadata
    file was the stamp's only source, so the stamp named the measurement and no file."""
    _m1(client, files)
    h5 = _entry(client, files.up)["apex"]["h5_path"]
    _clear_all(client)
    assert _stamp_of(_store())["apex"] == {"h5_path": h5, "measurement_id": GUID_A}


def test_a_result_started_while_another_pairing_is_being_loaded_names_that_pairings_file(
        client, files, monkeypatch):
    """The key is set early in a switch from measurement 1 to 2 of one path, the entry at the end of
    the load: in between, the entry still says measurement 1."""
    _m1(client, files)
    h5 = _entry(client, files.up)["apex"]["h5_path"]
    monkeypatch.setattr(ev, "_remember_apex_pairing", lambda *a, **k: None)     # the entry is written late
    _m2(client, files)
    assert ev._loaded_entry(files.up)["apex"]["measurement_id"] == GUID_A       # premise
    assert _stamp_of(_store())["apex"] == {"h5_path": h5, "measurement_id": GUID_B}


def _refusal_code(err):
    assert err.value.status_code == 400
    return err.value.headers["X-Orienta-Code"]


def test_a_pairing_without_its_metadata_file_takes_the_file_the_same_measurement_was_loaded_with(client, files):
    _m1(client, files)
    h5 = ev._loaded_entry(files.up)["apex"]["h5_path"]
    got = ev.remembered_apex_pairing(files.up, prefer={"h5_path": None, "measurement_id": GUID_A})
    assert got == {"h5_path": h5, "measurement_id": GUID_A}


@pytest.mark.parametrize("remembered", ["other measurement", "nothing"])
def test_a_pairing_without_its_metadata_file_is_refused_with_a_code_not_a_type_error(client, files, remembered):
    from fastapi import HTTPException
    _m2(client, files)
    if remembered == "nothing":
        _clear_all(client)
    # the pairing of measurement 1, whose file is lost, against an entry of measurement 2 (or none)
    with pytest.raises(HTTPException) as err:
        ev.remembered_apex_pairing(files.up, prefer={"h5_path": None, "measurement_id": GUID_A})
    assert _refusal_code(err) == "apexPairingLost"


def test_activating_a_result_whose_stamp_lost_its_metadata_file_is_refused_and_activates_nothing(client, files):
    _m1(client, files)
    first = _store()
    _stamp_of(first)["apex"]["h5_path"] = None            # a stamp as it was written after "Clear all"
    _load(client, files.other)
    _clear_all(client)
    r = client.post(f"/api/indexing/results/activate/{first}")
    assert r.status_code == 400 and r.headers["x-orienta-code"] == "apexPairingLost", r.text
    assert ind._active_result_id is None
    assert ev._ebsd_file_path == files.other                # nothing was switched


def test_a_result_is_never_activated_on_a_file_loaded_as_another_pairing(client, files, monkeypatch):
    _m2(client, files)
    second = _store()
    _load(client, files.other)
    real = ev.load_ebsd

    async def loads_the_other_measurement(req):
        return await real(ev.LoadEBSDRequest(
            path=req.path, apex=ev.ApexRef(h5_path=files.h5, measurement_id=GUID_A)))

    monkeypatch.setattr(ev, "load_ebsd", loads_the_other_measurement)
    r = client.post(f"/api/indexing/results/activate/{second}")
    assert ev._ebsd_variant_key == (None, None, GUID_A), "premise: the file came up as measurement 1"
    # the metadata file IS known; the file merely came up as another measurement: not "pairing lost"
    assert r.status_code == 400 and r.headers["x-orienta-code"] == "loadCameUpAsOther", r.text
    assert ind._active_result_id is None
    assert ev._ebsd_variant_key == (None, None, GUID_A), "the viewer stays on the variant that came up"


def test_a_result_is_never_activated_on_a_file_loaded_as_another_variant(client, files, monkeypatch):
    _load(client, files.other)
    rid = _store()
    ind._result_registry[rid].metadata.update(
        source_file=files.up, load_variant={"measurement": "2", "pattern_source": "unprocessed"})

    async def loads_nothing_new(req):
        return {}

    monkeypatch.setattr(ev, "load_ebsd", loads_nothing_new)
    ind._active_result_id = None                            # storing a result activates it; the question is the click
    r = client.post(f"/api/indexing/results/activate/{rid}")
    assert r.status_code == 400 and r.headers["x-orienta-code"] == "loadCameUpAsOther", r.text
    assert ind._active_result_id is None
