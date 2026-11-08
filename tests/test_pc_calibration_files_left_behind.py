"""A calibration reaches the files a load left behind.

Every load empties the calibration store, so the store holds the open file only. The files the user
left behind keep their calibration in the viewer's stash; the lists a person ticks from
(``candidates``, ``followers``) and the writers (``apply``, ``versions``, "Put it back") go over them
too. These tests go through the real routes with real files of the Test_data folder
(``ORIENTA_TEST_DATA``): two Ni files of one maker with the same settings, one Oxford file with other
settings, and a copy of one Ni file in another folder (the same stem). Nothing fills the store by hand.

Expected values are chosen inputs (the PCs typed below) or what a file's header states (read from the
file by ``probe``: the two Ni files state the same PC).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ORIENTA_ALLOWED_HOSTS", "testserver")

from tests.apex_helpers import wipe_backend  # noqa: E402
from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402
from tests.data_deps import local_name  # noqa: E402

A_FILE = "HiGainNi.h5"
B_FILE = "LoGainNi.h5"
FAR_FILE = local_name("oxford_pc_map_crop")   # an alias, see tests/data_deps.py
#: the PC both Ni headers state
HEADER_NI = (0.5072619, 0.2620761, 0.5584886)
TYPED_B = (0.55, 0.30, 0.60)         # typed on B, saved as the calibration
TYPED_B2 = (0.56, 0.31, 0.61)        # typed on B afterwards (the next version)
TYPED_A = (0.52, 0.27, 0.57)         # typed on A, its own
TYPED_COPY = (0.54, 0.29, 0.59)      # typed on the copy of A


def _data(name: str) -> Path:
    base = Path(os.environ.get("ORIENTA_TEST_DATA") or Path(__file__).resolve().parents[1] / "Test_data")
    path = base / name
    if not path.exists():
        pytest.skip(f"{name} is not in {base} (set ORIENTA_TEST_DATA to the folder that holds the test data)")
    return path


@pytest.fixture(autouse=True)
def clean(tmp_path):
    from backend.api.routes import pcrefinement
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")

    def wipe():
        wipe_backend()
        use.forget_all()
        pcrefinement._pc_before.clear()
        pcrefinement._sessions.clear()
        pcrefinement._parked_sessions.clear()
    wipe()
    yield
    wipe()
    cal.set_calibrations_dir_for_test(None)


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, client=("127.0.0.1", 5000))


def _load(client, path):
    r = client.post("/api/ebsd/load", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _switch(client, path):
    r = client.post("/api/ebsd/switch-file", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _type_pc(client, pc):
    r = client.post("/api/pc/detector/update-pc", json={"pcx": pc[0], "pcy": pc[1], "pcz": pc[2]})
    assert r.status_code == 200, r.text
    assert r.json()["pc_stored"] is True


def _save(client, dataset, name="Ni REM"):
    r = client.post("/api/pc-calibrations", json={"name": name, "dataset": dataset})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _rows(client, cal_id):
    r = client.get(f"/api/pc-calibrations/{cal_id}/candidates")
    assert r.status_code == 200, r.text
    return {c["dataset"]: c for c in r.json()["candidates"]}


def _apply(client, cal_id, names, accept=()):
    r = client.post(f"/api/pc-calibrations/{cal_id}/apply", json={"datasets": list(names), "accept_mismatch": list(accept)})
    assert r.status_code == 200, r.text
    return {a["dataset"]: a for a in r.json()["results"]}


def _store():
    from backend.api.services.calibration_store import calibration_store
    return calibration_store


@pytest.fixture
def a_path():
    return _data(A_FILE)


@pytest.fixture
def b_path():
    return _data(B_FILE)


@pytest.fixture
def world(client, a_path, b_path):
    """A loaded, then B loaded: A is left behind. A PC typed on B is saved as a calibration (version 1)."""
    _load(client, a_path)
    _load(client, b_path)
    _type_pc(client, TYPED_B)
    return client, _save(client, "LoGainNi")


def test_a_file_left_behind_is_listed_with_its_path_settings_and_pc(world, a_path):
    client, cal_id = world
    rows = _rows(client, cal_id)
    assert set(rows) == {"LoGainNi", "HiGainNi"}
    a = rows["HiGainNi"]
    assert a["stashed"] is True and a["path"] == str(a_path)
    assert a["match"] is True and a["has_pc"] is True and a["ref"] is None and a["on_this_version"] is False
    assert a["pc"] == pytest.approx(HEADER_NI, abs=1e-6)
    assert "stashed" not in rows["LoGainNi"]
    assert rows["LoGainNi"]["ref"]["version"] == 1 and rows["LoGainNi"]["on_this_version"] is True


def test_the_calibration_applied_to_a_file_left_behind_is_its_pc_when_it_is_opened(world, a_path):
    client, cal_id = world
    out = _apply(client, cal_id, ["HiGainNi"])["HiGainNi"]
    assert out["status"] == "applied" and out["pc"] == pytest.approx(TYPED_B)
    assert out["previous"] == {"pc": pytest.approx(HEADER_NI, abs=1e-6), "had_pc": True}
    # the open file is not touched
    assert _store().get_pc("LoGainNi") == pytest.approx(TYPED_B)
    answer = _switch(client, a_path)
    assert answer["pc"] == pytest.approx(TYPED_B) and answer["pc_source"] == "calibration"
    assert answer["pc_notice"] == "user_pc_kept"
    store = _store()
    assert store.get_entry("HiGainNi").pc_source == "calibration"
    ref = use.current_ref("HiGainNi", store)
    assert ref["id"] == cal_id and ref["version"] == 1
    # the detector indexing builds from the store
    assert np.asarray(store.get_detector("HiGainNi").pc).reshape(-1)[:3] == pytest.approx(TYPED_B)
    f = client.get("/api/pc-calibrations/for-dataset/HiGainNi").json()
    assert f["calibration"]["id"] == cal_id and f["diverged"] is False


def test_a_refine_at_b_asks_about_the_file_left_behind_that_is_on_the_same_version(world):
    client, cal_id = world
    assert client.get("/api/pc-calibrations/for-dataset/LoGainNi").json()["followers"] == []
    _apply(client, cal_id, ["HiGainNi"])
    f = client.get("/api/pc-calibrations/for-dataset/LoGainNi").json()
    assert f["diverged"] is False
    assert f["followers"] == [{"dataset": "HiGainNi", "per_pixel": False, "map_kind": None}]
    _type_pc(client, TYPED_B2)                              # B leaves the calibration
    f = client.get("/api/pc-calibrations/for-dataset/LoGainNi").json()
    assert f["diverged"] is True
    assert [x["dataset"] for x in f["followers"]] == ["HiGainNi"]


def test_a_new_version_moves_the_file_left_behind(world, a_path):
    client, cal_id = world
    _apply(client, cal_id, ["HiGainNi"])
    _type_pc(client, TYPED_B2)
    r = client.post(f"/api/pc-calibrations/{cal_id}/versions",
                    json={"dataset": "LoGainNi", "apply_to": ["HiGainNi"], "from_version": 1})
    assert r.status_code == 200, r.text
    assert [(a["dataset"], a["status"], a["version"]) for a in r.json()["applied"]] == [("HiGainNi", "applied", 2)]
    answer = _switch(client, a_path)
    assert answer["pc"] == pytest.approx(TYPED_B2)
    ref = use.current_ref("HiGainNi", _store())
    assert ref["version"] == 2


def test_a_file_left_with_a_pc_of_its_own_is_left_alone_by_a_new_version(world, a_path, b_path):
    client, cal_id = world
    _apply(client, cal_id, ["HiGainNi"])
    _switch(client, a_path)
    _type_pc(client, TYPED_A)                               # A leaves version 1 by its own hand
    _switch(client, b_path)
    _type_pc(client, TYPED_B2)
    r = client.post(f"/api/pc-calibrations/{cal_id}/versions",
                    json={"dataset": "LoGainNi", "apply_to": ["HiGainNi"], "from_version": 1})
    assert r.status_code == 200, r.text
    assert r.json()["applied"] == [{"dataset": "HiGainNi", "status": "left"}]
    answer = _switch(client, a_path)
    assert answer["pc"] == pytest.approx(TYPED_A) and answer["pc_source"] == "manual"


def test_a_file_that_was_dropped_from_the_list_is_gone_from_the_rows(world, a_path):
    from backend.api.routes import ebsd_viewer
    client, cal_id = world
    _apply(client, cal_id, ["HiGainNi"])                    # a PC of the user's: its stash outlives the list entry
    r = client.post("/api/ebsd/loaded-files/remove", json={"path": str(a_path)})
    assert r.status_code == 200, r.text
    assert ebsd_viewer._canonical_path(str(a_path)) in ebsd_viewer._registry_by_file, "the premise: a stash is left"
    assert set(_rows(client, cal_id)) == {"LoGainNi"}
    assert _apply(client, cal_id, ["HiGainNi"])["HiGainNi"]["status"] == "notLoaded"
    assert client.get("/api/pc-calibrations/for-dataset/LoGainNi").json()["followers"] == []


def test_a_file_with_other_settings_needs_the_second_naming_also_when_it_is_left_behind(client, b_path):
    far = _data(FAR_FILE)
    _load(client, far)
    _load(client, b_path)
    _type_pc(client, TYPED_B)
    cal_id = _save(client, "LoGainNi")
    name = far.stem
    row = _rows(client, cal_id)[name]
    assert row["stashed"] is True and row["match"] is False and row["differences"], row
    before = row["pc"]
    refused = _apply(client, cal_id, [name])[name]
    assert refused["status"] == "mismatch"
    assert _rows(client, cal_id)[name]["pc"] == before      # nothing was written
    applied = _apply(client, cal_id, [name], accept=[name])[name]
    assert applied["status"] == "applied" and applied["mismatch_accepted"] is True
    assert _switch(client, far)["pc"] == pytest.approx(TYPED_B)


def test_put_it_back_for_a_file_that_is_not_open(world, a_path, b_path):
    client, cal_id = world
    assert client.get("/api/pc/previous", params={"dataset": "HiGainNi"}).json()["previous"] is None
    _apply(client, cal_id, ["HiGainNi"])
    prev = client.get("/api/pc/previous", params={"dataset": "HiGainNi"}).json()["previous"]
    assert prev == {"pc": pytest.approx(HEADER_NI, abs=1e-6), "had_pc": True}
    r = client.post("/api/pc/put-back", json={"dataset": "HiGainNi"})
    assert r.status_code == 200, r.text
    assert r.json()["restored"] == ["HiGainNi"] and r.json()["pc"] == pytest.approx(HEADER_NI, abs=1e-6)
    assert _store().get_pc("LoGainNi") == pytest.approx(TYPED_B)            # the open file is not touched
    assert _rows(client, cal_id)["HiGainNi"]["pc"] == pytest.approx(HEADER_NI, abs=1e-6)
    assert _rows(client, cal_id)["HiGainNi"]["ref"] is None
    again = client.post("/api/pc/put-back", json={"dataset": "HiGainNi"})
    assert again.status_code == 409 and again.headers.get("X-Orienta-Code") == "nothingToPutBack"
    assert _switch(client, a_path)["pc"] == pytest.approx(HEADER_NI, abs=1e-6)


def test_the_answer_of_put_it_back_for_a_file_that_is_not_open_has_the_shape_the_page_reads(world):
    """The page reads `pc`, `had_pc` and `not_restored` of this answer to say what is back; the key set is pinned."""
    client, cal_id = world
    _apply(client, cal_id, ["HiGainNi"])
    r = client.post("/api/pc/put-back", json={"dataset": "HiGainNi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"dataset", "pc", "had_pc", "restored", "not_restored", "pc_per_pixel"}
    assert body["dataset"] == "HiGainNi" and body["had_pc"] is True
    assert body["pc"] == pytest.approx(HEADER_NI, abs=1e-6)
    assert body["restored"] == ["HiGainNi"] and body["not_restored"] == [] and body["pc_per_pixel"] is False


def test_a_load_ends_the_step_of_a_file_that_is_not_open(world, a_path, b_path):
    client, cal_id = world
    _apply(client, cal_id, ["HiGainNi"])
    _switch(client, a_path)
    _switch(client, b_path)                                  # A is left behind again, with the calibration
    r = client.post("/api/pc/put-back", json={"dataset": "HiGainNi"})
    assert r.status_code == 409 and r.headers.get("X-Orienta-Code") == "nothingToPutBack"
    assert _rows(client, cal_id)["HiGainNi"]["ref"]["version"] == 1


def test_two_files_of_one_stem_in_two_folders_do_not_mix(client, a_path, b_path, tmp_path):
    from backend.api.routes import pcrefinement
    other = tmp_path / "other_folder"
    other.mkdir()
    copy = other / A_FILE
    shutil.copy2(a_path, copy)
    _load(client, b_path)
    _type_pc(client, TYPED_B)
    cal_id = _save(client, "LoGainNi")
    _load(client, a_path)                                    # the original: left behind below
    _load(client, copy)                                      # the copy of the same stem is open
    _type_pc(client, TYPED_COPY)
    label = f"HiGainNi ({a_path.parent.name})"
    rows = _rows(client, cal_id)
    assert set(rows) == {"LoGainNi", "HiGainNi", label}
    assert rows["HiGainNi"]["pc"] == pytest.approx(TYPED_COPY) and "stashed" not in rows["HiGainNi"]
    assert rows[label]["stashed"] is True and rows[label]["path"] == str(a_path)
    assert rows[label]["pc"] == pytest.approx(HEADER_NI, abs=1e-6)
    out = _apply(client, cal_id, [label])[label]
    assert out["status"] == "applied"
    # the open file of the same name: its store entry, its PC page and the calibration it runs on are its own
    assert _store().get_pc("HiGainNi") == pytest.approx(TYPED_COPY)
    session = pcrefinement._sessions["HiGainNi"]
    assert np.asarray(session.ctrl.detector.pc).reshape(-1)[:3] == pytest.approx(TYPED_COPY)
    assert use.current_ref("HiGainNi", _store()) is None
    assert _rows(client, cal_id)["HiGainNi"]["ref"] is None
    assert _rows(client, cal_id)[label]["ref"]["version"] == 1
    # the plain name still means the open file
    assert _apply(client, cal_id, ["HiGainNi"])["HiGainNi"]["status"] == "applied"
    assert _rows(client, cal_id)["HiGainNi"]["ref"]["version"] == 1
    # and the original, opened, has what was given to it
    assert _switch(client, a_path)["pc"] == pytest.approx(TYPED_B)
    assert _switch(client, copy)["pc"] == pytest.approx(TYPED_B)


def _same_stem_world(client, a_path, b_path, tmp_path, distinct=False):
    """B saved as a calibration; the original of A left behind; a copy of A of the same stem open. A byte copy is
    the same file by the rule that recognises a file again (name, size, header id); ``distinct`` appends bytes to it
    (the pattern file is unchanged), which makes it another file of the same stem."""
    other = tmp_path / "other_folder"
    other.mkdir()
    copy = other / A_FILE
    shutil.copy2(a_path, copy)
    if distinct:
        with open(copy, "ab") as handle:
            handle.write(bytes(16))
    _load(client, b_path)
    _type_pc(client, TYPED_B)
    cal_id = _save(client, "LoGainNi")
    _load(client, a_path)
    _load(client, copy)
    return cal_id, copy, f"HiGainNi ({a_path.parent.name})"


def test_a_pc_typed_on_the_open_file_is_not_the_calibration_its_namesake_was_given(client, a_path, b_path, tmp_path):
    cal_id, copy, label = _same_stem_world(client, a_path, b_path, tmp_path, distinct=True)
    assert _apply(client, cal_id, [label])[label]["status"] == "applied"
    _type_pc(client, TYPED_B)                      # the open copy types the very PC of the calibration
    assert use.current_ref("HiGainNi", _store()) is None
    assert _rows(client, cal_id)["HiGainNi"]["ref"] is None
    assert _rows(client, cal_id)[label]["ref"]["version"] == 1
    f = client.get("/api/pc-calibrations/for-dataset/HiGainNi").json()
    assert f["calibration"] is None and f["followers"] == []


def test_the_step_of_the_open_file_survives_a_write_to_a_stashed_file_of_the_same_name(client, a_path, b_path, tmp_path):
    cal_id, copy, label = _same_stem_world(client, a_path, b_path, tmp_path)
    _type_pc(client, TYPED_COPY)                   # the open copy: its step back holds the header PC
    assert _apply(client, cal_id, [label])[label]["status"] == "applied"
    r = client.post("/api/pc/put-back", json={"dataset": "HiGainNi"})
    assert r.status_code == 200, r.text
    assert r.json()["pc"] == pytest.approx(HEADER_NI, abs=1e-6)
    assert _store().get_pc("HiGainNi") == pytest.approx(HEADER_NI, abs=1e-6)
    # the stashed file keeps its calibration and has its own step, and putting it back leaves the page of the open
    # file of the same name alone (a marker in its cache is not cleared)
    from backend.api.routes import pcrefinement
    pcrefinement._sessions["HiGainNi"].sim_cache["marker"] = 1
    assert _rows(client, cal_id)[label]["ref"]["version"] == 1
    assert client.post("/api/pc/put-back", json={"dataset": label}).status_code == 200
    assert pcrefinement._sessions["HiGainNi"].sim_cache.get("marker") == 1
    assert _rows(client, cal_id)[label]["ref"] is None


def test_a_write_to_a_stashed_file_with_a_copy_touches_no_page_of_the_open_file(client, a_path, b_path, tmp_path):
    """Both files hold a crop of the name "odd", which follows the PC of its file; only the stashed one's does."""
    from backend.api.routes import pcrefinement
    other = tmp_path / "other_folder"
    other.mkdir()
    copy = other / A_FILE
    shutil.copy2(a_path, copy)
    crop = dict(shape="rect", row0=0, col0=0, rows=4, cols=4, name="odd")
    _load(client, b_path)
    _type_pc(client, TYPED_B)
    cal_id = _save(client, "LoGainNi")
    _load(client, a_path)
    assert client.post("/api/ebsd/crop", json=crop).status_code == 200
    _load(client, copy)
    assert client.post("/api/ebsd/crop", json=crop).status_code == 200
    _type_pc(client, TYPED_COPY)                   # on the crop of the open copy: this makes its page
    label = f"HiGainNi ({a_path.parent.name})"
    assert _apply(client, cal_id, [label])[label]["status"] == "applied"
    page = pcrefinement._sessions["odd"]
    assert np.asarray(page.ctrl.detector.pc).reshape(-1)[:3] == pytest.approx(TYPED_COPY)
    assert _store().get_pc("odd") == pytest.approx(TYPED_COPY)
    # the crop of the stashed file did follow, and has the PC when its file is opened
    _switch(client, a_path)
    assert _store().get_pc("HiGainNi") == pytest.approx(TYPED_B)
    assert _store().get_pc("odd") == pytest.approx(TYPED_B)


def test_the_files_the_viewer_hands_out_are_left_behind_files_of_the_list_in_their_last_variant(client, a_path, b_path):
    """Hand-made state on top of real files: a stash of the open file, and a stash of another variant."""
    import copy as _copy
    from backend.api.routes import ebsd_viewer as ev
    _load(client, a_path)
    _load(client, b_path)
    (found,) = ev.stashed_file_entries()
    assert found["dataset"] == "HiGainNi" and found["path"] == str(a_path)
    # another variant of A left behind earlier: its dataset has another name, it is not the file's last variant
    other = _copy.deepcopy(found["entry"])
    other.dataset_name = "HiGainNi · 2"
    ev._registry_by_file[(found["canonical"], ("2", "unprocessed"))] = {"calibration": {"HiGainNi · 2": other}}
    assert [f["dataset"] for f in ev.stashed_file_entries()] == ["HiGainNi"]
    # a stash of the OPEN file (B) is not a file that was left behind
    ev._stash_registry_for_file(str(b_path))
    assert [f["dataset"] for f in ev.stashed_file_entries()] == ["HiGainNi"]


def test_a_file_that_is_quick_loaded_is_one_row_the_live_one(world, a_path):
    """A was left behind and is then looked at by a quick-load: its entry is in the store now, and that entry (the
    one a write reaches first) is the file's row; the stash of the same file gives none."""
    client, cal_id = world
    ql = client.post("/api/batch-v2/quick-load", json={"file_path": str(a_path)})
    assert ql.status_code == 200, ql.text
    name = ql.json()["dataset_name"]
    from backend.api.routes import ebsd_viewer
    assert _store().get_entry(name).file_key == ebsd_viewer._canonical_path(str(a_path)), "the premise: the file is live"
    rows = client.get(f"/api/pc-calibrations/{cal_id}/candidates").json()["candidates"]
    of_a = [r for r in rows if r.get("path") == str(a_path) or r["dataset"].startswith("HiGainNi")]
    assert [(r["dataset"], r.get("stashed", False)) for r in of_a] == [(name, False)]
    out = _apply(client, cal_id, [name])[name]
    assert out["status"] == "applied" and _store().get_pc(name) == pytest.approx(TYPED_B)
