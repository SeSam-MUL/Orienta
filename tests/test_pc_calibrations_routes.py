"""/api/pc-calibrations: save a PC as a calibration, be offered one, apply it only when asked."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402

PC_A = (0.5026, 0.3272, 0.8458)


def _signal(wd=15.0, shape=(128, 156), pc=(0.5, 0.4, 0.6)):
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=shape, pc=pc, sample_tilt=70.0, tilt=4.32)
    sig = kp.signals.EBSD(np.zeros((2, 2) + shape, np.uint8), detector=det)
    if wd is not None:
        sig.metadata.set_item("Acquisition_instrument.SEM.working_distance", wd)
    return sig


@pytest.fixture
def world(tmp_path, monkeypatch):
    from backend.api.main import app
    from backend.api.routes import pcrefinement
    from backend.api.services.calibration_store import calibration_store as store
    monkeypatch.setattr(pcrefinement, "_optimization_active", False)
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    store.clear()
    use.forget_all()
    paths = {}
    for name, wd in (("scan_a", 15.0), ("scan_b", 15.0), ("scan_far", 20.0)):
        paths[name] = tmp_path / f"{name}.h5oina"
        paths[name].write_bytes(name.encode() * 40)
        store.register(name, _signal(wd=wd), file_key=str(paths[name]))
    store.update_pc("scan_a", PC_A, source="refined", refinement={"n_patterns": 9, "method": "PSO"})
    yield TestClient(app), store, paths
    store.clear()
    use.forget_all()
    cal.set_calibrations_dir_for_test(None)


def _save(client, name="REM 2026-10-14", dataset="scan_a", **extra):
    r = client.post("/api/pc-calibrations", json={"name": name, "dataset": dataset, **extra})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ saving

def test_saving_a_dataset_makes_version_one_with_its_settings_and_origin(world):
    client, store, paths = world
    made = _save(client)
    assert made["version"] == 1 and made["pc"] == pytest.approx(PC_A) and made["n_members"] == 1
    assert made["features"]["pattern_shape"] == [128, 156] and made["features"]["working_distance_mm"] == 15.0
    assert made["origin"]["pc_source"] == "refined" and made["origin"]["refinement"] == {"n_patterns": 9, "method": "PSO"}
    assert made["origin"]["dataset"] == "scan_a" and "kikuchipy" in made["origin"]["libraries"]
    assert use.current_ref("scan_a", store)["id"] == made["id"]          # scan_a IS this calibration now
    listing = client.get("/api/pc-calibrations").json()
    assert [c["name"] for c in listing["calibrations"]] == ["REM 2026-10-14"] and listing["problems"] == []


def test_a_calibration_made_from_a_search_says_how_the_pc_was_found(world):
    client, store, _ = world
    use.note_search("scan_a", {"pc": list(PC_A), "mode": "find", "method": "PSO+NM", "cost": 0.9, "n_patterns": 9,
                               "recentred": 1, "far_box_used": False,
                               "confidence": {"state": "clear", "contrast": 0.8}})
    made = _save(client)
    assert made["origin"]["method"] == "find"
    assert made["origin"]["search"]["confidence"] == "clear" and made["origin"]["search"]["contrast"] == 0.8
    # a search whose PC is no longer the dataset's is not attached to a PC it did not make
    use.note_search("scan_b", {"pc": [0.1, 0.1, 0.1], "mode": "find", "confidence": {}})
    store.update_pc("scan_b", (0.5, 0.31, 0.7), source="manual")
    other = _save(client, "Other", "scan_b")
    assert other["origin"]["method"] == "typed" and "search" not in other["origin"]


def test_a_file_without_a_pc_cannot_be_saved(world):
    client, store, _ = world
    entry = store.get_entry("scan_b")
    entry.pc_source = "missing"
    r = client.post("/api/pc-calibrations", json={"name": "x", "dataset": "scan_b"})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "noPcToSave"
    assert client.get("/api/pc-calibrations").json()["calibrations"] == []


def test_unknown_dataset_and_taken_name_are_refused_with_codes(world):
    client, *_ = world
    r = client.post("/api/pc-calibrations", json={"name": "x", "dataset": "nope"})
    assert r.status_code == 404 and r.headers["X-Orienta-Code"] == "unknownDataset"
    _save(client)
    r = client.post("/api/pc-calibrations", json={"name": "rem 2026-10-14", "dataset": "scan_b"})
    assert r.status_code == 409 and r.headers["X-Orienta-Code"] == "nameTaken"


def test_rename_and_delete(world):
    client, *_ = world
    made = _save(client)
    r = client.put(f"/api/pc-calibrations/{made['id']}", json={"name": "Renamed"})
    assert r.status_code == 200 and r.json()["name"] == "Renamed" and r.json()["id"] == made["id"]
    assert client.delete(f"/api/pc-calibrations/{made['id']}").status_code == 200
    r = client.get(f"/api/pc-calibrations/{made['id']}")
    assert r.status_code == 404 and r.headers["X-Orienta-Code"] == "unknownCalibration"
    assert client.delete(f"/api/pc-calibrations/{made['id']}").status_code == 404


# ------------------------------------------------------------------ suggesting never changes anything

def test_suggest_offers_a_matching_calibration_and_changes_nothing(world):
    client, store, _ = world
    made = _save(client)
    before = {n: store.get_pc(n).copy() for n in ("scan_a", "scan_b", "scan_far")}
    files = {p.name: p.read_bytes() for p in cal.calibrations_dir().glob("*.json")}
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).json()
    assert [s["id"] for s in got["suggestions"]] == [made["id"]]
    assert got["suggestions"][0]["match"] is True and got["suggestions"][0]["via_member"] is False
    assert got["current"] is None and got["unverified"] is False
    far = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_far"}).json()
    assert far["suggestions"] == []                                  # 20 mm against 15 mm
    for name, pc in before.items():
        np.testing.assert_allclose(store.get_pc(name), pc)
    assert {p.name: p.read_bytes() for p in cal.calibrations_dir().glob("*.json")} == files


@pytest.mark.parametrize("params", [{"dataset": "gone_file"}, {"dataset": "gone_file", "all": "true"}])
def test_suggest_for_a_name_the_store_does_not_know_answers_200_and_says_so(world, params):
    client, store, _ = world
    made = _save(client)
    got = client.get("/api/pc-calibrations/suggest", params=params)
    assert got.status_code == 200
    assert got.json() == {"dataset": "gone_file", "target": "gone_file", "suggestions": [], "current": None,
                          "unverified": False, "unknown_dataset": True}
    # a known name is not marked, so the page can tell the two apart
    known = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).json()
    assert "unknown_dataset" not in known and [s["id"] for s in known["suggestions"]] == [made["id"]]


def test_the_other_reads_of_the_panel_on_a_stale_name_do_not_answer_404(world):
    client, *_ = world
    made = _save(client)
    # for-dataset answers "no calibration, nobody follows" for a name that is not loaded
    got = client.get("/api/pc-calibrations/for-dataset/gone_file")
    assert got.status_code == 200
    assert got.json() == {"calibration": None, "diverged": False, "followers": []}
    # results and candidates are asked by calibration id, never by file name; a known id never answers 404
    assert client.get(f"/api/pc-calibrations/{made['id']}/results").status_code == 200
    assert client.get(f"/api/pc-calibrations/{made['id']}/candidates").status_code == 200


def test_a_dataset_already_on_the_latest_version_is_not_offered_it_again(world):
    client, store, _ = world
    made = _save(client)
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_a"}).json()
    assert got["suggestions"] == [] and got["current"]["id"] == made["id"]


def test_a_remembered_file_is_offered_its_calibration_even_when_a_setting_changed(world, tmp_path):
    client, store, paths = world
    made = _save(client)
    store.get_entry("scan_a").features["working_distance_mm"] = 20.0     # the header changed since
    store.update_pc("scan_a", (0.5, 0.4, 0.6), source="manual")          # and the PC was set by hand
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_a"}).json()
    assert len(got["suggestions"]) == 1 and got["suggestions"][0]["via_member"] is True
    assert got["suggestions"][0]["match"] is False and got["suggestions"][0]["differences"][0]["key"] == "working_distance_mm"


def test_a_file_whose_settings_could_not_be_read_gets_no_suggestion_and_is_told_so(world):
    client, store, _ = world
    _save(client)
    store.get_entry("scan_b").features = None
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).json()
    assert got["suggestions"] == [] and got["unverified"] is True and got["target"] == "scan_b"


def test_a_file_that_states_none_of_its_settings_gets_no_suggestion_and_is_told_so(world, tmp_path):
    """An .ebsp-style signal (``pc_calibrations.settings_stated``): the store's entry has the pattern shape and
    nothing else, so a calibration of the same size says nothing about it."""
    client, store, _ = world
    _save(client)
    path = tmp_path / "pat.ebsp"
    path.write_bytes(b"pat" * 50)
    sig = _signal(wd=None)
    sig.metadata.set_item("General.original_filename", str(path))
    store.register("pat", sig, file_key=str(path))
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "pat"}).json()
    assert got["suggestions"] == [] and got["unverified"] is True


def test_an_offer_carries_the_settings_of_both_sides_also_where_they_agree(world):
    """The settings table of the page shows every setting with the value of the file and of the calibration."""
    client, *_ = world
    _save(client)
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).json()["suggestions"][0]
    # scan_b and the calibration (made on scan_a) were built with working distance 15 mm and a 128 x 156 pattern
    assert got["settings"]["file"]["pattern_shape"] == [128, 156]
    assert got["settings"]["calibration"]["pattern_shape"] == [128, 156]
    assert got["settings"]["file"]["working_distance_mm"] == 15.0
    assert got["settings"]["calibration"]["working_distance_mm"] == 15.0
    # the helper signal has no detector insertion header: not stated on either side, never 0
    assert got["settings"]["file"]["detector_insertion_mm"] is None
    assert got["settings"]["calibration"]["detector_insertion_mm"] is None


def test_all_true_lists_the_calibration_of_another_working_distance_with_what_differs(world):
    client, *_ = world
    made = _save(client)
    only = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_far"}).json()
    assert only["suggestions"] == []
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_far", "all": "true"}).json()
    [one] = got["suggestions"]
    assert one["id"] == made["id"] and one["match"] is False
    assert [(d["key"], d["file"], d["calibration"]) for d in one["differences"]] == [("working_distance_mm", 20.0, 15.0)]
    assert one["settings"]["file"]["working_distance_mm"] == 20.0
    assert one["settings"]["calibration"]["working_distance_mm"] == 15.0


def test_all_true_also_lists_the_calibration_the_file_is_on_already(world):
    client, *_ = world
    made = _save(client)                               # scan_a is on this very version now
    assert client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_a"}).json()["suggestions"] == []
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_a", "all": "true"}).json()
    assert [s["id"] for s in got["suggestions"]] == [made["id"]] and got["suggestions"][0]["match"] is True


def test_all_true_for_a_file_whose_settings_could_not_be_read_compares_the_pattern_shape_only(world):
    client, store, _ = world
    made = _save(client)
    store.get_entry("scan_b").features = None
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b", "all": "true"}).json()
    [one] = got["suggestions"]
    assert one["id"] == made["id"] and one["unverified"] is True and one["match"] is False
    assert one["differences"] == [] and one["settings"]["file"]["working_distance_mm"] is None
    assert one["settings"]["file"]["pattern_shape"] == [128, 156]


# ------------------------------------------------------------------ applying

def test_apply_changes_only_what_is_named_and_says_what_it_did(world):
    client, store, _ = world
    made = _save(client)
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b", "scan_far"]})
    assert r.status_code == 200
    by = {x["dataset"]: x for x in r.json()["results"]}
    assert by["scan_b"]["status"] == "applied" and by["scan_far"]["status"] == "mismatch"
    np.testing.assert_allclose(store.get_pc("scan_b"), PC_A, atol=1e-12)
    assert not np.allclose(store.get_pc("scan_far"), PC_A)
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply",
                    json={"datasets": ["scan_far"], "accept_mismatch": ["scan_far"]})
    assert r.json()["results"][0]["status"] == "applied" and r.json()["results"][0]["mismatch_accepted"] is True


def test_apply_without_a_name_is_refused(world):
    client, *_ = world
    made = _save(client)
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": []})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "needsTargets"


def test_candidates_list_the_loaded_files_with_what_differs(world):
    client, *_ = world
    made = _save(client)
    got = {c["dataset"]: c for c in client.get(f"/api/pc-calibrations/{made['id']}/candidates").json()["candidates"]}
    assert got["scan_b"]["match"] is True and got["scan_far"]["match"] is False
    assert got["scan_far"]["differences"][0]["key"] == "working_distance_mm"
    # the apply table reads the PC of the file, its settings and where it is loaded from (these files are not in the loaded list)
    assert got["scan_far"]["pc"] == pytest.approx([0.5, 0.4, 0.6]) and got["scan_far"]["ref"] is None
    assert got["scan_far"]["settings"]["file"]["working_distance_mm"] == 20.0
    assert got["scan_far"]["path"] is None


# ------------------------------------------------------------------ taking a PC from another file

def test_a_pc_from_an_ang_file_is_proposed_not_applied(world, tmp_path):
    client, store, _ = world
    ang = tmp_path / "other.ang"
    ang.write_text("# x-star 0.500000\n# y-star 0.600000\n# z-star 0.700000\n#\n 0 0 0 0 0\n", encoding="utf-8")
    before = store.get_pc("scan_b").copy()
    r = client.post("/api/pc-calibrations/from-file", json={"path": str(ang), "dataset": "scan_b"})
    assert r.status_code == 200, r.text
    body = r.json()
    # TSL (0.5, 0.6, 0.7) on 128 x 156 (the dataset's pattern shape): y = 1 - 0.6; z * min(128, 156) / 128 = 0.7
    assert body["pc"] == pytest.approx([0.5, 0.4, 0.7], abs=1e-6)
    assert body["shape_assumed"] is True and body["target_shape"] == [128, 156] and body["shape_matches"] is True
    # the English sentence stays in ``notes``; ``note_items`` carry the code and the numbers the page translates with
    assert body["note_items"][0]["code"] == "assumedShape"
    assert body["note_items"][0]["params"] == {"rows": 128, "cols": 156, "convention": "tsl"}
    assert body["note_items"][0]["text"] == body["notes"][0]
    np.testing.assert_allclose(store.get_pc("scan_b"), before)


def test_a_file_that_cannot_give_a_pc_is_refused_with_its_code(world, tmp_path):
    client, *_ = world
    ang = tmp_path / "empty.ang"
    ang.write_text("# TEM_PIXperUM 1.0\n", encoding="utf-8")
    r = client.post("/api/pc-calibrations/from-file", json={"path": str(ang), "dataset": "scan_b"})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "noPcInFile"
    r = client.post("/api/pc-calibrations/from-file", json={"path": str(tmp_path / "x.ang"), "dataset": "scan_b"})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "fileNotFound"


def test_the_shape_a_file_states_is_compared_with_the_datasets(world, tmp_path):
    import h5py
    client, *_ = world
    p = tmp_path / "other.h5"
    with h5py.File(p, "w") as f:
        det = f.create_group("Detector")
        det["pc"] = np.array([0.5, 0.4, 0.7])
        det.attrs["shape"] = np.array([60, 60])
    body = client.post("/api/pc-calibrations/from-file", json={"path": str(p), "dataset": "scan_b"}).json()
    assert body["shape"] == [60, 60] and body["shape_matches"] is False


# ------------------------------------------------------------------ versions, followers, results

def test_a_refine_at_a_member_can_become_the_next_version_for_everyone_it_names(world):
    client, store, _ = world
    made = _save(client)
    client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]})
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    f = client.get("/api/pc-calibrations/for-dataset/scan_a").json()
    assert f["diverged"] is True and f["followers"] == [{"dataset": "scan_b", "per_pixel": False, "map_kind": None}]
    assert f["calibration"]["version"] == 1
    r = client.post(f"/api/pc-calibrations/{made['id']}/versions", json={"dataset": "scan_a", "apply_to": ["scan_b"]})
    assert r.status_code == 200, r.text
    assert r.json()["calibration"]["version"] == 2 and r.json()["applied"][0]["status"] == "applied"
    np.testing.assert_allclose(store.get_pc("scan_b"), (0.51, 0.33, 0.85), atol=1e-12)
    assert use.current_ref("scan_a", store)["version"] == 2 and use.current_ref("scan_b", store)["version"] == 2
    assert client.get(f"/api/pc-calibrations/{made['id']}").json()["versions"][0]["pc"] == pytest.approx(PC_A)


def _two_on_one_version(client, store):
    """scan_a and scan_b on version 1; scan_a then searched (0.51, 0.33, 0.85): the question is about scan_b."""
    made = _save(client)
    client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]})
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    return made


def test_a_follower_that_got_another_pc_since_the_question_is_left_alone(world):
    """The page asked about scan_b while it was on version 1. Then scan_b got its own PC (0.47, 0.37, 0.57). The old
    list must not overwrite it: its answer is ``left``, nothing is written to it, the version is still made."""
    client, store, _ = world
    made = _two_on_one_version(client, store)
    store.update_pc("scan_b", (0.47, 0.37, 0.57), source="manual")
    r = client.post(f"/api/pc-calibrations/{made['id']}/versions",
                    json={"dataset": "scan_a", "apply_to": ["scan_b"], "from_version": 1})
    assert r.status_code == 200, r.text
    assert r.json()["applied"] == [{"dataset": "scan_b", "status": "left"}]
    np.testing.assert_allclose(store.get_pc("scan_b"), (0.47, 0.37, 0.57), atol=1e-12)
    assert r.json()["calibration"]["version"] == 2                           # the version for scan_a was made
    assert use.current_ref("scan_b", store) is None


def test_a_follower_still_on_the_version_asked_about_is_moved_and_one_that_left_is_not(world):
    client, store, paths = world
    made = _save(client)
    client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b", "scan_far"],
                                                                 "accept_mismatch": ["scan_far"]})
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    store.update_pc("scan_far", (0.47, 0.37, 0.57), source="manual")
    r = client.post(f"/api/pc-calibrations/{made['id']}/versions",
                    json={"dataset": "scan_a", "apply_to": ["scan_b", "scan_far"], "from_version": 1,
                          "accept_mismatch": ["scan_far"]})
    assert r.status_code == 200, r.text
    got = r.json()["applied"]
    assert [(a["dataset"], a["status"]) for a in got] == [("scan_b", "applied"), ("scan_far", "left")]   # the order asked
    np.testing.assert_allclose(store.get_pc("scan_b"), (0.51, 0.33, 0.85), atol=1e-12)
    np.testing.assert_allclose(store.get_pc("scan_far"), (0.47, 0.37, 0.57), atol=1e-12)


def test_a_follower_on_another_version_than_the_question_was_about_is_left_alone(world):
    client, store, _ = world
    made = _two_on_one_version(client, store)
    r = client.post(f"/api/pc-calibrations/{made['id']}/versions",
                    json={"dataset": "scan_a", "apply_to": ["scan_b"], "from_version": 2})
    assert r.status_code == 200, r.text
    assert r.json()["applied"] == [{"dataset": "scan_b", "status": "left"}]
    np.testing.assert_allclose(store.get_pc("scan_b"), PC_A, atol=1e-12)     # still the PC of version 1


def _versions_of(client, cal_id):
    return client.get(f"/api/pc-calibrations/{cal_id}").json()["versions"]


def test_a_dataset_without_a_pc_cannot_become_a_version_the_placeholder_is_not_a_pc(world, monkeypatch):
    """A file whose ``pc_source`` is ``missing`` has no pattern centre, whatever number its detector holds. Today the
    store answers ``None`` for it (``CalibrationStore.get_pc``), so a check on ``get_pc(...) is None`` happens to
    work; the route must not depend on that. Second half: a store that answered with kikuchipy's placeholder
    (0.5, 0.5, 0.5) must not turn it into version 2 of a real calibration."""
    client, store, _ = world
    made = _save(client)
    store.get_entry("scan_b").pc_source = "missing"
    url = f"/api/pc-calibrations/{made['id']}/versions"
    r = client.post(url, json={"dataset": "scan_b"})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "noPcToSave"
    real = store.get_pc
    monkeypatch.setattr(store, "get_pc",
                        lambda name: np.array([0.5, 0.5, 0.5]) if name == "scan_b" else real(name))
    assert store.get_pc("scan_b") is not None                               # a store that does hand out the placeholder
    r = client.post(url, json={"dataset": "scan_b"})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "noPcToSave"
    assert len(_versions_of(client, made["id"])) == 1                       # nothing was written


def test_a_new_version_names_only_loaded_files_and_writes_nothing_when_one_name_is_wrong(world):
    client, store, _ = world
    made = _save(client)
    store.register_derived("scan_a_copy", "scan_a")
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    for name, status, code in (("nope", 404, "unknownDataset"), ("scan_a_copy", 400, "isCopy")):
        r = client.post(f"/api/pc-calibrations/{made['id']}/versions",
                        json={"dataset": "scan_a", "apply_to": ["scan_b", name]})
        assert r.status_code == status and r.headers["X-Orienta-Code"] == code
        assert len(_versions_of(client, made["id"])) == 1                   # the version was not written first
        np.testing.assert_allclose(store.get_pc("scan_b"), (0.5, 0.4, 0.6))  # and nobody was moved


# ------------------------------------------------------------------ further cases of the stored values

def test_every_route_is_a_plain_def_not_a_coroutine(world):
    """A calibration file is read under a lock with retries: on the event loop a held file would stall every request."""
    import inspect
    from backend.api.routes import pc_calibrations as routes
    found = [r for r in routes.router.routes if hasattr(r, "endpoint")]
    assert len(found) >= 12
    assert [r.path for r in found if inspect.iscoroutinefunction(r.endpoint)] == []


def test_an_author_is_free_text_and_unknown_by_default_never_the_machines_user(world):
    import getpass
    client, *_ = world
    first = _save(client)
    assert first["author"] == "unknown" and first["author"] != getpass.getuser()
    second = _save(client, "Second", "scan_b", author="  A. Example, Lab 3 ")
    assert second["author"] == "A. Example, Lab 3"
    assert client.get(f"/api/pc-calibrations/{first['id']}").json()["author"] == "unknown"


def test_apply_reports_what_happened_to_a_per_pixel_map_of_each_dataset(world):
    """scan_b holds a grid-calibrated 2 x 2 map: a calibration replaces it and the answer says so; scan_a has a
    single PC, so nothing happened to a map (``none``, its own entry ``single``)."""
    client, store, _ = world
    made = _save(client)
    grid = np.array([[[0.49, 0.39, 0.59], [0.51, 0.41, 0.61]], [[0.49, 0.39, 0.59], [0.51, 0.41, 0.61]]])
    store.update_pc_map("scan_b", grid)
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b", "scan_a"]})
    by = {x["dataset"]: x for x in r.json()["results"]}
    assert by["scan_b"]["pc_map_action"] == "replaced" and by["scan_b"]["pc_map_actions"] == {"scan_b": "replaced"}
    assert by["scan_a"]["pc_map_action"] == "none" and by["scan_a"]["pc_map_actions"] == {"scan_a": "single"}
    assert store.get_entry("scan_b").pc_map is None
    np.testing.assert_allclose(store.get_pc("scan_b"), PC_A, atol=1e-12)


def _refine_at_a_member(client, store):
    made = _save(client)
    client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]})
    store.update_pc("scan_a", (0.51, 0.33, 0.85), source="refined")
    return made


def test_a_calibration_is_not_applied_nor_versioned_while_a_pc_search_runs(world, monkeypatch):
    from backend.api.routes import pcrefinement
    client, store, _ = world
    made = _refine_at_a_member(client, store)
    store.update_pc("scan_b", (0.55, 0.45, 0.65), source="manual")
    monkeypatch.setattr(pcrefinement, "_optimization_active", True)
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "pc_job_running"
    r = client.post(f"/api/pc-calibrations/{made['id']}/versions", json={"dataset": "scan_a", "apply_to": ["scan_b"]})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "pc_job_running"
    np.testing.assert_allclose(store.get_pc("scan_b"), (0.55, 0.45, 0.65))          # nobody was moved
    assert len(_versions_of(client, made["id"])) == 1                               # and nothing was written
    # reading is not changing: the page may still ask while the search runs
    assert client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).status_code == 200
    assert client.get(f"/api/pc-calibrations/{made['id']}/candidates").status_code == 200
    monkeypatch.setattr(pcrefinement, "_optimization_active", False)
    assert client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]}).status_code == 200


@pytest.mark.parametrize("which", ["apply", "versions"])
def test_a_calibration_waits_for_the_page_lock_and_answers_busy_when_it_is_held(world, monkeypatch, which):
    """An Index All holds the lock for minutes: the request answers 409 after the wait instead of hanging, and
    changes nothing."""
    from backend.api.routes import pcrefinement
    client, store, _ = world
    made = _refine_at_a_member(client, store)
    monkeypatch.setattr(pcrefinement, "_BUSY_WAIT_S", 0.05)
    body = {"datasets": ["scan_b"]} if which == "apply" else {"dataset": "scan_a", "apply_to": ["scan_b"]}
    store.update_pc("scan_b", (0.55, 0.45, 0.65), source="manual")
    pcrefinement._state_lock.acquire()
    try:
        r = client.post(f"/api/pc-calibrations/{made['id']}/{which}", json=body)
    finally:
        pcrefinement._state_lock.release()
    assert r.status_code == 409 and r.json()["detail"]["code"] == "pc_job_running"
    np.testing.assert_allclose(store.get_pc("scan_b"), (0.55, 0.45, 0.65))
    assert len(_versions_of(client, made["id"])) == 1


@pytest.mark.parametrize("which", ["apply", "versions"])
def test_a_search_that_starts_while_the_request_waits_for_the_lock_still_refuses_it(world, monkeypatch, which):
    import threading
    import time
    from backend.api.routes import pcrefinement
    client, store, _ = world
    made = _refine_at_a_member(client, store)
    store.update_pc("scan_b", (0.55, 0.45, 0.65), source="manual")
    body = {"datasets": ["scan_b"]} if which == "apply" else {"dataset": "scan_a", "apply_to": ["scan_b"]}
    out = {}
    pcrefinement._state_lock.acquire()
    try:
        worker = threading.Thread(target=lambda: out.update(
            r=client.post(f"/api/pc-calibrations/{made['id']}/{which}", json=body)))
        worker.start()
        time.sleep(0.4)                                    # the request is waiting for the lock (it gives up after 2 s)
        monkeypatch.setattr(pcrefinement, "_optimization_active", True)       # a search is accepted meanwhile
    finally:
        pcrefinement._state_lock.release()
    worker.join(10)
    assert out["r"].status_code == 409 and out["r"].json()["detail"]["code"] == "pc_job_running"
    np.testing.assert_allclose(store.get_pc("scan_b"), (0.55, 0.45, 0.65))
    assert len(_versions_of(client, made["id"])) == 1


def test_an_unknown_version_and_an_unknown_calibration_are_refused_with_codes(world):
    client, *_ = world
    made = _save(client)
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"], "version": 7})
    assert r.status_code == 404 and r.headers["X-Orienta-Code"] == "unknownVersion"
    r = client.post("/api/pc-calibrations/nope-123456/apply", json={"datasets": ["scan_b"]})
    assert r.status_code == 404 and r.headers["X-Orienta-Code"] == "unknownCalibration"
    r = client.post("/api/pc-calibrations/nope-123456/versions", json={"dataset": "scan_a"})
    assert r.status_code == 404 and r.headers["X-Orienta-Code"] == "unknownCalibration"


# ------------------------------------------------------------------ files and requests at the edges

def _plant(tmp_path, name, text):
    folder = tmp_path / "Calibrations"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


def _planted_copy(tmp_path, cal_id, mutate):
    """A second calibration file made from the saved one: ``mutate(doc)`` changes it, ``id`` follows the file name."""
    doc = json.loads((tmp_path / "Calibrations" / f"{cal_id}.json").read_text(encoding="utf-8"))
    doc["id"], doc["name"] = "edge-abc123", "Edge"
    mutate(doc)
    _plant(tmp_path, "edge-abc123.json", json.dumps(doc))


def _raw_json(client, method, url, body):
    """The body as JSON text with ensure_ascii, so a lone surrogate travels as the six characters backslash-u-d800."""
    return client.request(method, url, content=json.dumps(body), headers={"Content-Type": "application/json"})


def test_a_file_nested_deeper_than_the_parser_goes_is_a_problem_not_a_500(world, tmp_path):
    client, *_ = world
    made = _save(client)
    _plant(tmp_path, "zz.json", "[" * 200000)
    listing = client.get("/api/pc-calibrations")
    assert listing.status_code == 200
    assert [c["id"] for c in listing.json()["calibrations"]] == [made["id"]]
    assert [(p["file"], p["code"]) for p in listing.json()["problems"]] == [("zz.json", "unreadable")]
    assert client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).status_code == 200
    assert client.post("/api/pc-calibrations", json={"name": "Second", "dataset": "scan_b"}).status_code == 200


@pytest.mark.parametrize("key", ["camera_tilt", "sample_tilt", "working_distance_mm", "binning"])
@pytest.mark.parametrize("literal", ["1e400", "-1e400", str(10 ** 400)])
def test_a_setting_that_is_infinite_or_beyond_a_float_makes_the_file_a_problem(world, tmp_path, key, literal):
    client, *_ = world
    made = _save(client)
    _planted_copy(tmp_path, made["id"], lambda doc: doc["features"].update({key: "__BIG__"}))
    path = tmp_path / "Calibrations" / "edge-abc123.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"__BIG__"', literal), encoding="utf-8")
    listing = client.get("/api/pc-calibrations")
    assert listing.status_code == 200
    assert [c["id"] for c in listing.json()["calibrations"]] == [made["id"]]
    assert [(p["file"], p["code"]) for p in listing.json()["problems"]] == [("edge-abc123.json", "unreadable")]
    for url in ("/api/pc-calibrations/edge-abc123", "/api/pc-calibrations/edge-abc123/candidates"):
        r = client.get(url)
        assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "unreadable", url
    assert client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).status_code == 200


@pytest.mark.parametrize("name", ["\ud800x", "a\udfffb"])
def test_a_name_that_cannot_be_written_is_refused_with_a_code_on_save_and_rename(world, name):
    client, *_ = world
    made = _save(client)
    r = _raw_json(client, "POST", "/api/pc-calibrations", {"name": name, "dataset": "scan_b"})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "badName"
    r = _raw_json(client, "PUT", f"/api/pc-calibrations/{made['id']}", {"name": name})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "badName"
    assert [c["name"] for c in client.get("/api/pc-calibrations").json()["calibrations"]] == ["REM 2026-10-14"]


def test_an_author_that_cannot_be_written_is_refused_with_a_code(world):
    client, *_ = world
    r = _raw_json(client, "POST", "/api/pc-calibrations", {"name": "N", "dataset": "scan_b", "author": "\ud800"})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "badName"
    assert client.get("/api/pc-calibrations").json()["calibrations"] == []


def test_name_and_author_have_a_limit_of_200_characters(world):
    client, *_ = world
    assert cal.MAX_NAME_CHARS == 200 and cal.MAX_AUTHOR_CHARS == 200
    r = client.post("/api/pc-calibrations", json={"name": "n" * 201, "dataset": "scan_b"})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "nameTooLong"
    r = client.post("/api/pc-calibrations", json={"name": "n", "dataset": "scan_b", "author": "a" * 201})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "nameTooLong"
    assert client.get("/api/pc-calibrations").json()["calibrations"] == []
    made = _save(client, "m" * 200, "scan_b", author="a" * 200)           # exactly 200 is allowed
    assert len(made["name"]) == 200 and made["author"] == "a" * 200
    r = client.put(f"/api/pc-calibrations/{made['id']}", json={"name": "n" * 201})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "nameTooLong"
    assert client.get(f"/api/pc-calibrations/{made['id']}").json()["name"] == "m" * 200


def test_an_oxford_map_whose_spread_overflows_gives_the_pc_without_a_spread_note(world, tmp_path):
    """x = 1e155 and 3e155: the mean 2e155 is a float, the deviations 1e155 square to 1e310, which is not."""
    import h5py
    client, *_ = world
    p = tmp_path / "huge_map.h5oina"
    with h5py.File(p, "w") as f:
        data = f.create_group("1/EBSD/Data")
        data["Pattern Center X"] = np.array([1e155, 3e155])
        data["Pattern Center Y"] = np.array([0.5, 0.5])
        data["Detector Distance"] = np.array([0.7, 0.7])
        header = f.create_group("1/EBSD/Header")
        header["Pattern Width"] = np.array([156])
        header["Pattern Height"] = np.array([128])
    r = client.post("/api/pc-calibrations/from-file", json={"path": str(p), "dataset": "scan_b"})
    assert r.status_code == 200, r.text
    body = r.json()
    codes = [n["code"] for n in body["note_items"]]
    assert "stdOverMap" not in codes and "meanOfMap" in codes and "outsideDetectorRange" in codes
    assert body["pc"][0] == pytest.approx(2e155, rel=1e-12)
    assert body["pc"][1:] == pytest.approx([1 - 0.5 * 1.21875, 0.7 * 1.21875])


def test_a_member_stored_in_another_case_is_still_the_file(world, tmp_path):
    """A library moved to a case-sensitive system: the stored member name differs in case from the file's name."""
    client, store, _ = world
    made = _save(client)
    path = tmp_path / "Calibrations" / f"{made['id']}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert [m["name"].lower() for m in doc["members"]] == ["scan_a.h5oina"]
    doc["members"][0]["name"] = "SCAN_A.H5OINA"
    path.write_text(json.dumps(doc), encoding="utf-8")
    store.get_entry("scan_a").features["working_distance_mm"] = 20.0      # settings differ: only membership offers it
    store.update_pc("scan_a", (0.5, 0.4, 0.6), source="manual")          # no longer on the calibration's version
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_a"}).json()
    assert [(s["id"], s["via_member"]) for s in got["suggestions"]] == [(made["id"], True)]


def test_a_result_shows_which_version_of_the_calibration_it_was_computed_at(world):
    from backend.api.routes import indexing
    client, store, _ = world
    made = _save(client)
    from backend.api.services import pc_record
    rec = pc_record.pc_run_record("scan_a", store)
    assert rec["calibration"] == {"id": made["id"], "name": "REM 2026-10-14", "version": 1}
    indexing._result_registry["r-old"] = SimpleNamespace(metadata={"pc_at_run": rec, "source_file": "C:/x/scan_a.h5oina"})
    indexing._result_registry["r-none"] = SimpleNamespace(metadata={"pc_at_run": dict(rec, calibration=None)})
    try:
        client.post(f"/api/pc-calibrations/{made['id']}/versions", json={"dataset": "scan_a"})   # still the same PC
        cal.add_version(made["id"], pc=(0.5, 0.33, 0.85), origin={"method": "refine"})
        got = client.get(f"/api/pc-calibrations/{made['id']}/results").json()
        assert got["latest"] == 3
        assert got["results"] == [{"result_id": "r-old", "source_file": "scan_a.h5oina", "version": 1,
                                   "latest": 3, "state": "older"}]
    finally:
        indexing._result_registry.pop("r-old", None)
        indexing._result_registry.pop("r-none", None)


def test_a_run_at_a_typed_pc_has_no_calibration_reference(world):
    from backend.api.services import pc_record
    client, store, _ = world
    _save(client)
    rec = pc_record.pc_run_record("scan_a", store, typed_pc=(0.5, 0.3, 0.8))
    assert rec["calibration"] is None and rec["pc_source"] == "typed_for_run"


@pytest.mark.parametrize("what,mutate", [
    ("a text for the origin", lambda doc: doc["versions"][0].update({"origin": "typed"})),
    ("a list for the origin", lambda doc: doc["versions"][0].update({"origin": ["file"]})),
    ("a number for the origin", lambda doc: doc["versions"][0].update({"origin": 3})),
    ("a text of digits for the PC", lambda doc: doc["versions"][0].update({"pc": "123"})),
    ("a text of digits in the PC", lambda doc: doc["versions"][0].update({"pc": ["0.5", "0.4", "0.6"]})),
    ("a record for the PC", lambda doc: doc["versions"][0].update({"pc": {"a": 1, "b": 2, "c": 3}})),
    ("a truth value in the PC", lambda doc: doc["versions"][0].update({"pc": [True, 0.4, 0.6]})),
])
def test_a_file_whose_version_has_a_wrong_origin_or_pc_type_is_a_problem_and_the_list_stays_200(world, tmp_path, what, mutate):
    client, *_ = world
    made = _save(client)
    _planted_copy(tmp_path, made["id"], mutate)
    listing = client.get("/api/pc-calibrations")
    assert listing.status_code == 200, what
    assert [c["id"] for c in listing.json()["calibrations"]] == [made["id"]]
    assert [p["file"] for p in listing.json()["problems"]] == ["edge-abc123.json"]
    assert client.get("/api/pc-calibrations/edge-abc123").status_code == 400
    assert client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).status_code == 200


def test_a_version_without_an_origin_or_with_none_is_still_a_calibration(world, tmp_path):
    client, *_ = world
    made = _save(client)
    _planted_copy(tmp_path, made["id"], lambda doc: doc["versions"][0].update({"origin": None}))
    listing = client.get("/api/pc-calibrations").json()
    assert sorted(c["id"] for c in listing["calibrations"]) == sorted([made["id"], "edge-abc123"])
    assert listing["problems"] == []


# ------------------------------------------------------------------ a copy is offered the calibration for its file

def test_a_suggestion_for_a_copy_names_the_file_it_is_to_be_applied_to_and_apply_refuses_the_copy(world):
    client, store, _ = world
    made = _save(client)
    store.register_derived("scan_b_copy", "scan_b")
    got = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b_copy"}).json()
    assert [s["id"] for s in got["suggestions"]] == [made["id"]]
    assert got["dataset"] == "scan_b_copy" and got["target"] == "scan_b"            # the file, not the copy
    plain = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).json()
    assert plain["target"] == "scan_b"                                              # a file is its own target
    everything = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b_copy", "all": True}).json()
    assert everything["target"] == "scan_b"
    # the copy itself is refused by apply, with its status: nothing is written to it
    refused = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b_copy"]}).json()
    assert refused["results"] == [{"dataset": "scan_b_copy", "status": "isCopy"}]
    np.testing.assert_allclose(store.get_pc("scan_b"), (0.5, 0.4, 0.6))
    # applied to the target, the copy that never had a PC of its own follows and the answer lists it
    applied = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": [got["target"]]}).json()
    assert applied["results"][0]["status"] == "applied"
    assert set(applied["results"][0]["pc_map_actions"]) == {"scan_b", "scan_b_copy"}


def test_a_copy_that_has_a_pc_of_its_own_is_not_in_the_answer_of_the_apply_to_its_file(world):
    client, store, _ = world
    made = _save(client)
    store.register_derived("scan_b_copy", "scan_b")
    store.update_pc("scan_b_copy", (0.45, 0.35, 0.55), source="manual")           # typed on the copy
    applied = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]}).json()
    assert set(applied["results"][0]["pc_map_actions"]) == {"scan_b"}
    np.testing.assert_allclose(store.get_pc("scan_b_copy"), (0.45, 0.35, 0.55))
