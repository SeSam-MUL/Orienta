"""The wire carries NAMES; identities stay on disk.

The defect this file exists for: making `parent`, `active` and `hidden`
identities (eb284a48) turned the phase filter silently OFF. Every consumer
compares names -- `collectionFilter.js:25` is `list.find(c => c.name ===
activeName)` and `:26` is `if (!self) return null; // renamed or deleted: filter
nothing` -- so an id took that branch and the Indexing page, the Phase Tester and
the EDS phase map all widened to the WHOLE LIBRARY while the toolbar still read
"Collection: …". Measured before the fix:

    server: state.active = 'Al_systems'   (name was 'Al systems')
    frontend activeKeySet(...)  -> None   = filter nothing
    childrenOf('Al systems')    -> []     = the child collection vanishes

EVERY TEST HERE USES A NAME WITH A SPACE, and asserts up front that the id and
the name differ. That is the whole reason the defect shipped: the only
round-trip test used the name "M", and `safe_filename("M") == "M"`, so id and
name coincided and the assertion could not fail. Measured:

    PUT active='M'           -> GET 'M'           round-trips
    PUT active='Al systems'  -> GET 'Al_systems'  did not

Same class as the 2026-08-28 `syncResultContract` lesson: a green test is
exactly as strong as its fixture.

These go through the ROUTE functions, not the service. The service was correct
throughout -- it was the translation at the boundary that was missing, and a
service-level test cannot see a boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.routes import phase_collections as route  # noqa: E402
from backend.api.services import phase_collections as pc   # noqa: E402

#: A space is the trigger: `safe_filename` turns it into `_`, so the id and the
#: name differ. Any character outside [A-Za-z0-9._-] does it, which includes
#: every name our own suggestion engine proposes ("Matrix and pure metals").
PARENT = "Al systems"
CHILD = "Al-Fe phases"


class _Entry:
    def __init__(self, key):
        self.key, self.formula, self.space_group = key, key, ""
        self.display_label = key
        self.cif_path = self.xtal_path = self.sht_path = None
        self.elements = ()
        self.parse_error = None


@pytest.fixture
def client(tmp_path, monkeypatch):
    pc.set_collections_dir_for_test(tmp_path / "Collections")
    import backend.api.services.crystal_hint_local_library as lib
    monkeypatch.setattr(lib, "get_index", lambda: {
        k: _Entry(k) for k in ("Al", "Al13Fe4", "Si")})
    monkeypatch.setattr(route, "_master_map", lambda idx: {})
    app = FastAPI()
    app.include_router(route.router, prefix="/api/phase-collections")
    yield TestClient(app)
    pc.set_collections_dir_for_test(None)


def _frontend_active_key_set(collections, active_name):
    """`collectionFilter.js:22-32`, transcribed.

    Kept as a transcription rather than a description so that this test fails
    if the server stops speaking the language that function reads.
    """
    if not active_name:
        return None
    self_ = next((c for c in collections if c["name"] == active_name), None)
    if self_ is None:
        return None                      # "renamed or deleted: filter nothing"
    keys = {m["key"] for m in (self_.get("members") or [])}
    for child in collections:
        if child.get("parent") == active_name:
            keys |= {m["key"] for m in (child.get("members") or [])}
    return keys


def _build(client):
    """A parent with a child, both named with a space, and the id proven distinct."""
    assert client.post("/api/phase-collections/",
                       json={"name": PARENT}).status_code == 200
    assert client.post("/api/phase-collections/",
                       json={"name": CHILD, "parent": PARENT}).status_code == 200
    client.post("/api/phase-collections/members",
                json={"name": PARENT, "keys": ["Al"]})
    client.post("/api/phase-collections/members",
                json={"name": CHILD, "keys": ["Al13Fe4"]})
    body = client.get("/api/phase-collections/").json()
    by = {c["name"]: c for c in body["collections"]}
    assert by[PARENT]["id"] != PARENT, (
        "this whole file depends on the id differing from the name; "
        f"got id={by[PARENT]['id']!r} for {PARENT!r}")
    return body, by


def test_parent_is_answered_as_a_name(client):
    _, by = _build(client)
    assert by[CHILD]["parent"] == PARENT, (
        "an id here makes childrenOf() return [] and the child vanishes from "
        "both collection UIs")


def test_active_is_answered_as_a_name_and_the_filter_stays_on(client):
    body, _ = _build(client)
    assert client.put("/api/phase-collections/state",
                      json={"active": PARENT, "hidden": []}).status_code == 200
    body = client.get("/api/phase-collections/").json()

    assert body["state"]["active"] == PARENT

    # The claim that matters: the shipped filter narrows, and to the parent's
    # own members PLUS the child's.
    keys = _frontend_active_key_set(body["collections"], body["state"]["active"])
    assert keys is not None, "FILTER OFF: the whole library would be offered"
    assert keys == {"Al", "Al13Fe4"}


def test_a_hidden_collection_is_reported_hidden_by_name(client):
    _build(client)
    client.put("/api/phase-collections/state",
               json={"active": None, "hidden": [PARENT]})
    body = client.get("/api/phase-collections/").json()
    by = {c["name"]: c for c in body["collections"]}
    assert body["state"]["hidden"] == [PARENT]
    assert by[PARENT]["hidden"] is True
    assert by[CHILD]["hidden"] is True, "a child inherits its parent's hiding"


def test_put_state_answers_the_same_shape_as_the_listing(client):
    """It used to drop `schema`: `resolve_state`'s `{**state, ...}` carries only
    what it is handed. One thing must not have two shapes depending on which
    endpoint you read it from."""
    _build(client)
    put = client.put("/api/phase-collections/state",
                     json={"active": PARENT, "hidden": []}).json()["state"]
    get = client.get("/api/phase-collections/").json()["state"]
    assert set(put) == set(get) == {"schema", "active", "hidden"}
    assert put == get


def test_the_identity_is_still_what_is_stored_on_disk(client):
    """The other half. Names on the wire must not mean names at rest -- the
    point of identities is that a rename cannot break a stored reference."""
    _, by = _build(client)
    parent_id = by[PARENT]["id"]
    client.put("/api/phase-collections/state",
               json={"active": PARENT, "hidden": [PARENT]})

    raw = pc.load_state()                      # straight off the file
    assert raw["active"] == parent_id
    assert raw["hidden"] == [parent_id]
    child = {c.name: c for c in pc.load_all()[0]}[CHILD]
    assert child.parent == parent_id


def test_a_rename_keeps_the_child_and_the_active_state(client):
    """What identities are FOR, asserted end to end: after a rename nothing had
    to be repaired, and the answer is consistent in the new name."""
    _, by = _build(client)
    client.put("/api/phase-collections/state",
               json={"active": PARENT, "hidden": []})

    assert client.patch("/api/phase-collections/rename",
                        json={"name": PARENT,
                              "new_name": "Aluminium systems"}).status_code == 200

    body = client.get("/api/phase-collections/").json()
    by2 = {c["name"]: c for c in body["collections"]}
    assert set(by2) == {"Aluminium systems", CHILD}
    assert by2[CHILD]["parent"] == "Aluminium systems", "the link followed"
    assert body["state"]["active"] == "Aluminium systems", "and so did active"
    keys = _frontend_active_key_set(body["collections"], body["state"]["active"])
    assert keys == {"Al", "Al13Fe4"}


def test_a_child_follows_its_parent_in_the_listing(client):
    """`databaseGrouping.js:98-101` relies on that adjacency instead of tracking
    parents itself. Sorting on a key that mixes ids and names breaks it:
    measured ['Al systems', 'Al then', 'Al-Fe phases'] with ids against
    ['Al systems', 'Al-Fe phases', 'Al then'] with names."""
    _build(client)
    client.post("/api/phase-collections/", json={"name": "Al then"})
    names = [c["name"] for c in
             client.get("/api/phase-collections/").json()["collections"]]
    assert names.index(CHILD) == names.index(PARENT) + 1, names


def test_a_parent_comes_before_a_child_that_sorts_ahead_of_it(client):
    """The tie-break, with a fixture that can actually fail.

    A parent and its child share the first sort key -- the parent's name -- so
    without an explicit "parents first" term the order falls to whatever comes
    next. The test above does NOT catch that: `"al systems"` happens to sort
    before `"al-fe phases"`, so the parent led by luck (checked by removing the
    term: that test still passed).

    Here the child is named so that it sorts AHEAD of its parent. Measured with
    the term removed: ['Al AAA phases', 'Al systems'] -- the child precedes the
    parent it belongs to, and `databaseGrouping.js` reads it as a top-level row.
    """
    assert client.post("/api/phase-collections/",
                       json={"name": PARENT}).status_code == 200
    early_child = "Al AAA phases"
    assert early_child.lower() < PARENT.lower(), "the fixture's whole point"
    assert client.post("/api/phase-collections/",
                       json={"name": early_child,
                             "parent": PARENT}).status_code == 200

    names = [c["name"] for c in
             client.get("/api/phase-collections/").json()["collections"]]
    assert names == [PARENT, early_child], names
