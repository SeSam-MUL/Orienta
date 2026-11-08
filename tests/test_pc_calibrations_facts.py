"""What a calibration answer says about the version it shows: its date, the file it was made on, how good the
search was, and (in an offer) what the offer would replace."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402
from tests.test_pc_calibrations_routes import PC_A, _save, world  # noqa: E402,F401  (the fixture)

SEARCH = {"pc": list(PC_A), "mode": "find", "method": "PSO+NM", "cost": 0.9, "n_patterns": 9, "recentred": 0,
          "far_box_used": False, "on_edge": [],
          "confidence": {"state": "notJudged", "contrast": 0.8, "typical_cost": 8.0, "reasons": []}}
GOOD = {"contrast": 0.8, "warning": None, "axes": [], "n_patterns": 9}


def test_a_calibration_made_from_a_search_carries_its_quality_date_and_source_file(world):
    client, store, _ = world
    use.note_search("scan_a", SEARCH)
    made = _save(client)
    assert made["quality"] == GOOD and made["source_name"] == "scan_a.h5oina"
    stored = cal.get(made["id"])
    assert stored["versions"][0]["origin"]["quality"] == GOOD
    assert stored["versions"][0]["origin"]["source_name"] == "scan_a.h5oina"
    # the date of the version is the one written into the version; it is a date
    assert made["version_created"] == stored["versions"][0]["created"]
    datetime.fromisoformat(made["version_created"])
    shown = client.get(f"/api/pc-calibrations/{made['id']}").json()
    assert shown["quality"] == GOOD and shown["source_name"] == "scan_a.h5oina"
    assert shown["versions"][0]["origin"]["quality"] == GOOD
    listed = client.get("/api/pc-calibrations").json()["calibrations"][0]
    assert listed["quality"] == GOOD and listed["source_name"] == "scan_a.h5oina"


def test_the_warning_of_the_search_is_stored_with_the_calibration(world):
    client, store, _ = world
    use.note_search("scan_a", {**SEARCH, "on_edge": ["x", "z"]})
    made = _save(client)
    assert made["quality"] == {"contrast": 0.8, "warning": "edge", "axes": ["x", "z"], "n_patterns": 9}


def test_a_calibration_made_from_a_typed_pc_has_no_quality_but_still_a_source_file(world):
    client, store, _ = world
    store.update_pc("scan_b", (0.5, 0.31, 0.7), source="manual")
    made = _save(client, "Typed", "scan_b")
    assert made["quality"] is None and made["source_name"] == "scan_b.h5oina"


def test_a_search_that_did_not_make_the_pc_gives_no_quality(world):
    client, store, _ = world
    use.note_search("scan_b", {**SEARCH, "pc": [0.1, 0.1, 0.1]})
    assert _save(client, "Other search", "scan_b")["quality"] is None


def test_a_new_version_shows_its_own_date_file_and_quality(world):
    client, store, _ = world
    use.note_search("scan_a", SEARCH)
    made = _save(client)
    use.note_search("scan_b", {**SEARCH, "pc": [0.51, 0.33, 0.85], "on_edge": ["y"],
                               "confidence": {"contrast": 0.6, "typical_cost": 8.0, "reasons": []}})
    store.update_pc("scan_b", (0.51, 0.33, 0.85), source="refined", refinement={"n_patterns": 9, "method": "PSO"})
    r = client.post(f"/api/pc-calibrations/{made['id']}/versions", json={"dataset": "scan_b"})
    assert r.status_code == 200, r.text
    shown = r.json()["calibration"]
    assert shown["version"] == 2 and shown["source_name"] == "scan_b.h5oina"
    assert shown["quality"] == {"contrast": 0.6, "warning": "edge", "axes": ["y"], "n_patterns": 9}
    stored = cal.get(made["id"])
    assert shown["version_created"] == stored["versions"][1]["created"]
    assert stored["versions"][0]["origin"]["source_name"] == "scan_a.h5oina"        # version 1 keeps its own
    assert stored["versions"][0]["origin"]["quality"] == GOOD


OTHER_POSITION = {"pc": [0.40, 0.30, 1.30], "cost": 1.0}


def test_a_calibration_saved_after_a_find_with_a_second_position_carries_it(world):
    """The search found PC_A at cost 0.9 and another position at cost 1.0: saved with both, in the version's origin
    and in every answer for it."""
    client, store, _ = world
    use.note_search("scan_a", {**SEARCH, "second_position": OTHER_POSITION})
    made = _save(client)
    want = {**GOOD, "warning": "secondPosition", "second_position": OTHER_POSITION, "best_cost": 0.9}
    assert made["quality"] == want
    assert cal.get(made["id"])["versions"][0]["origin"]["quality"] == want
    assert client.get(f"/api/pc-calibrations/{made['id']}").json()["quality"] == want
    assert client.get("/api/pc-calibrations").json()["calibrations"][0]["quality"] == want


def test_a_version_saved_before_these_facts_were_recorded_reads_none(world):
    client, store, _ = world
    made = _save(client)
    path = cal.calibrations_dir() / f"{made['id']}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for key in ("source_name", "quality"):
        doc["versions"][0]["origin"].pop(key)
    path.write_text(json.dumps(doc), encoding="utf-8")
    shown = client.get(f"/api/pc-calibrations/{made['id']}").json()
    assert shown["source_name"] is None and shown["quality"] is None
    assert shown["version_created"] == doc["versions"][0]["created"]


def test_the_source_file_is_a_name_never_a_path(tmp_path):
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")       # never the library under Database/
    try:
        _source_file_is_a_name()
    finally:
        cal.set_calibrations_dir_for_test(None)


def _source_file_is_a_name():
    made = cal.create(name="p", pc=(0.5, 0.4, 0.7), features={"pattern_shape": [60, 60]},
                      origin={"source_name": "C:\\Users\\someone\\data\\scan.h5oina"})
    assert made["versions"][0]["origin"]["source_name"] == "scan.h5oina"
    assert cal.summary(made)["source_name"] == "scan.h5oina"
    more = cal.add_version(made["id"], pc=(0.5, 0.4, 0.71), origin={"source_name": "/home/x/other.h5"})
    assert more["versions"][1]["origin"]["source_name"] == "other.h5"


# ------------------------------------------------------------------ an offer says what it would replace

def test_an_offer_says_what_it_would_replace_and_the_dates(world):
    client, store, _ = world
    made = _save(client)
    store.get_entry("scan_b").features["hints"]["date"] = "2026-10-01"
    offer = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).json()["suggestions"][0]
    assert offer["id"] == made["id"]
    assert offer["replaces"] == {"pc": pytest.approx([0.5, 0.4, 0.6]), "kind": "file"}      # scan_b's header PC
    assert offer["file_date"] == "2026-10-01"
    assert offer["version_created"] == cal.get(made["id"])["versions"][0]["created"]
    assert offer["source_name"] == "scan_a.h5oina"
    assert offer["checked"][0] == "pattern_shape" and "working_distance_mm" in offer["checked"]


@pytest.mark.parametrize("source,record,kind", [
    ("manual", None, "typed"),
    ("refined", {"n_patterns": 9, "method": "PSO", "mode": "find"}, "find"),
    ("refined", {"n_patterns": 9, "method": "Nelder-Mead", "mode": "refine"}, "refine"),
])
def test_an_offer_names_the_kind_of_the_pc_it_would_replace(world, source, record, kind):
    client, store, _ = world
    _save(client)
    store.update_pc("scan_b", (0.45, 0.25, 0.55), source=source, refinement=record)
    offer = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).json()["suggestions"][0]
    assert offer["replaces"] == {"pc": pytest.approx([0.45, 0.25, 0.55]), "kind": kind}


def test_an_offer_for_a_file_without_a_pc_replaces_nothing(world):
    client, store, _ = world
    _save(client)
    store.get_entry("scan_b").pc_source = "missing"
    offer = client.get("/api/pc-calibrations/suggest", params={"dataset": "scan_b"}).json()["suggestions"][0]
    assert offer["replaces"] == {"pc": None, "kind": "none"}


# ------------------------------------------------------------------ the source file reads as it is written on disk

def test_the_source_name_keeps_the_capitals_of_the_file_though_its_key_is_lower_case(world):
    from tests.test_pc_calibrations_routes import _signal
    client, store, paths = world
    folder = paths["scan_a"].parent
    (folder / "Scan_Cap.h5oina").write_bytes(b"x" * 40)
    # the key as a Windows load makes it: the whole path normcased, here only the name is enough
    store.register("scan_cap", _signal(), file_key=str(folder / "scan_cap.h5oina"))
    store.update_pc("scan_cap", PC_A, source="refined", refinement={"n_patterns": 9, "method": "PSO"})
    made = _save(client, name="capitals", dataset="scan_cap")
    assert made["source_name"] == "Scan_Cap.h5oina"
    assert cal.get(made["id"])["versions"][0]["origin"]["source_name"] == "Scan_Cap.h5oina"


def test_the_real_file_name_falls_back_to_the_name_the_caller_holds_then_to_the_key(tmp_path):
    key = str(tmp_path / "gone" / "scan_x.h5oina")                      # no such folder: nothing to list
    assert cal.real_file_name(key, "D:\data\Scan_X.h5oina") == "Scan_X.h5oina"
    assert cal.real_file_name(key, "D:\data\other.h5oina") == "scan_x.h5oina"     # another file's name
    assert cal.real_file_name(key) == "scan_x.h5oina"
    assert cal.real_file_name(None, "Scan_X.h5oina") is None
