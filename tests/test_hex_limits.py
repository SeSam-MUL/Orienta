"""``GET /api/scan-geometry/hex-limits``: the list of what a hexagonal scan cannot do yet.

The list is built from the table that enforces the limits, so these tests hold the answer against
that table (every entry present, nothing else); the batches are not in it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import hex_guard  # noqa: E402
from backend.api.services.hex_guard import GUARDED  # noqa: E402
from tests.apex_helpers import wipe_backend  # noqa: E402
from tests.test_hex_guard import (  # noqa: E402
    HEX, _fake_signal, IDS,
)

URL = "/api/scan-geometry/hex-limits"


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


@pytest.fixture
def answer(client):
    r = client.get(URL)
    assert r.status_code == 200, r.text
    return r.json()


def test_the_answer_has_two_groups_and_the_unlimited_things(answer):
    assert set(answer) == {"groups", "categories", "not_limited"}
    assert list(answer["groups"]) == ["before_indexing", "after_indexing"]
    assert answer["categories"] == ["preprocessing", "grains_and_boundaries", "maps", "cleanup",
                                    "refinement"]
    # image export, saving a result as a file and global pattern-centre refinement are NOT limited,
    # and the answer says so
    assert [(n["key"], bool(n["what"])) for n in answer["not_limited"]] == [
        ("image_export", True), ("file_export", True), ("global_pc_refinement", True)]
    assert ".ang" in answer["not_limited"][1]["what"] and ".h5" in answer["not_limited"][1]["what"]


def test_the_table_has_ten_keys_in_four_filled_categories_and_the_counts_are_written_by_hand(answer):
    """Hand count from the table: nlpar + 3 crystal-hint routes (1 key) = 2 keys in preprocessing; grain phase
    assignment, reconstruction, size, deformation = 4 in grains_and_boundaries; analysis_map = 1 in maps;
    pixel-wise PC calibration, refinement compute, resmooth = 3 in refinement; 12 routes in all."""
    items = _all_items(answer)
    assert len(items) == 12
    per_category = {}
    for i in items:
        per_category.setdefault(i["category"], set()).add(i["key"])
    assert {c: len(k) for c, k in per_category.items()} == {
        "preprocessing": 2, "grains_and_boundaries": 4, "maps": 1, "refinement": 3}
    assert {i["code"] for i in items} == {hex_guard.NEIGHBOURHOOD, hex_guard.PC_GRID,
                                          hex_guard.ANALYSIS_PAGE}


def test_nothing_is_left_of_the_export_refusal():
    assert not hasattr(hex_guard, "EXPORT")
    assert "hex_export_pending" not in hex_guard.MESSAGES
    assert "export" not in hex_guard.CATEGORIES and "batch" not in hex_guard.CATEGORIES


@pytest.mark.parametrize("entry", GUARDED, ids=IDS)
def test_every_guarded_entry_is_in_the_answer_in_its_group(answer, entry):
    items = answer["groups"][entry.stage]
    found = [i for i in items if (i["method"], i["path"]) == (entry.method, entry.path)]
    assert len(found) == 1, f"{entry.method} {entry.path} is not in the {entry.stage} group"
    item = found[0]
    assert (item["what"], item["code"]) == (entry.what, entry.code)
    assert item["message"] == hex_guard.MESSAGES[entry.code].format(what=entry.what)
    assert item["partial"] is (entry.when is not None)
    assert (item["key"], item["category"]) == (entry.key, entry.category)
    assert set(item) == {"key", "category", "method", "path", "what", "code", "message", "partial",
                         "affects", "parameters"}


def test_the_groups_hold_the_table_and_nothing_else(answer):
    listed = [(i["method"], i["path"]) for g in answer["groups"] for i in answer["groups"][g]]
    assert sorted(listed) == sorted((g.method, g.path) for g in GUARDED)      # no entry twice, none invented


def test_which_limits_are_met_before_indexing_is_written_down_by_hand(answer):
    before = {i["path"] for i in answer["groups"]["before_indexing"]}
    assert before == {"/api/ebsd/nlpar", "/api/pc/calibrate/grid",
                      "/api/crystal-hint/analyze-pixel", "/api/crystal-hint/analyze-region",
                      "/api/crystal-hint/quality-check"}
    after = {i["path"] for i in answer["groups"]["after_indexing"]}
    assert "/api/indexing/grain-phase-assign" in after and "/api/analysis/grain-size" in after
    assert "/api/indexing/export" not in after and "/api/indexing/export" not in before   # the .ang is written
    assert not (before & after)


def test_only_the_requests_that_need_neighbours_are_partial(answer):
    partial = {i["path"] for g in ("before_indexing", "after_indexing") for i in answer["groups"][g]
               if i["partial"]}
    assert partial == {"/api/crystal-hint/analyze-pixel", "/api/crystal-hint/analyze-region",
                       "/api/crystal-hint/quality-check"}


def _item(answer, path):
    return next(i for i in _all_items(answer) if i["path"] == path)


def test_a_partial_limit_says_what_it_refuses_as_data_written_down_by_hand(answer):
    for path in ("/api/crystal-hint/analyze-pixel", "/api/crystal-hint/analyze-region",
                 "/api/crystal-hint/quality-check"):
        averaging = _item(answer, path)
        assert (averaging["affects"], averaging["parameters"]) == ([], ["avg_radius"])


def test_a_limit_that_is_always_refused_has_nothing_to_list(answer):
    for item in _all_items(answer):
        if not item["partial"]:
            assert (item["affects"], item["parameters"]) == ([], []), item["path"]


#: Change a constant of the gate: the answer follows, because the answer is built from it, and so does
#: the gate. One test per constant, so that a mutant is caught by the test that names it.

def test_the_averaging_parameter_listed_is_the_parameter_the_gate_reads(client, monkeypatch):
    monkeypatch.setattr(hex_guard, "AVERAGING_PARAMETER", "radius")
    assert _item(client.get(URL).json(), "/api/crystal-hint/analyze-pixel")["parameters"] == ["radius"]
    assert hex_guard.averaging_asked(hex_guard.Ask({}, {"radius": 1}, {})) is True
    assert hex_guard.averaging_asked(hex_guard.Ask({}, {"avg_radius": 1}, {})) is False


def test_every_entry_has_a_stage_the_list_knows():
    assert all(g.stage in hex_guard.STAGES for g in GUARDED)


# ----------------------------------------------------------- keys and categories (the load dialog translates by key)

#: Written out by hand: the load dialog has one text per key, so a rename is a decision, not an accident.
EXPECTED_KEYS = {
    "nlpar", "pixelwise_pc_calibration", "grain_phase_assignment",
    "grain_reconstruction", "grain_size", "deformation_analysis", "analysis_map",
    "refinement_compute", "refinement_resmooth",
    "crystal_hint_averaging",
}

#: The keys that several routes share: they are one limit to the user (the crystal hint's
#: neighbourhood average).
SHARED_KEYS = {"crystal_hint_averaging": {"/api/crystal-hint/analyze-pixel",
                                          "/api/crystal-hint/analyze-region",
                                          "/api/crystal-hint/quality-check"}}


def _all_items(answer):
    return [i for items in answer["groups"].values() for i in items]


def test_every_limit_has_a_snake_case_key_and_a_known_category_and_only_the_crystal_hint_averaging_shares_one(answer):
    import re
    items = _all_items(answer)
    keys = [i["key"] for i in items]
    assert set(keys) == EXPECTED_KEYS
    assert all(re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", k) for k in keys), keys
    assert all(i["category"] in answer["categories"] for i in items)
    sharing = {k: {i["path"] for i in items if i["key"] == k} for k in set(keys) if keys.count(k) > 1}
    assert sharing == SHARED_KEYS


def test_routes_that_share_a_key_agree_in_everything_but_their_route(answer):
    """One limit to the user: one sentence, one category, one stage, the same triggers."""
    items = _all_items(answer)
    for key in SHARED_KEYS:
        group = [i for i in items if i["key"] == key]
        assert len(group) == 3
        rest = [{k: v for k, v in i.items() if k not in ("method", "path")} for i in group]
        assert all(r == rest[0] for r in rest), key


@pytest.mark.parametrize("entry", GUARDED, ids=IDS)
def test_a_guarded_entry_without_a_key_or_with_an_unknown_category_fails(entry):
    """The check that a NEW ``GUARDED`` entry cannot skip: it must name itself and sort itself."""
    assert entry.key, f"{entry.method} {entry.path} has no key"
    assert entry.category in hex_guard.CATEGORIES, f"{entry.method} {entry.path}: {entry.category!r}"


# ------------------------------------------------------------------------------ the batch

def test_the_batch_has_no_limit_of_its_own(answer):
    """Both batches (the page's ``/api/batch-v2`` and ``/api/indexing/batch/start``) index and write a
    hexagonal scan, so the list has no batch group, no batch category and no entry for them."""
    assert "batch" not in answer["groups"] and "batch" not in answer["categories"]
    assert not any("batch" in i["key"] for i in _all_items(answer))
    for gone in ("BATCH", "BATCH_GROUP", "BATCH_RUN", "BATCH_EXPORT", "BatchLimit", "HexScanRefused",
                 "refuse_hex_signal"):
        assert not hasattr(hex_guard, gone), gone
