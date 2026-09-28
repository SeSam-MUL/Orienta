"""The collections route. Mounted on a bare app, so main.py's import-time work
(model prewarm, log handlers) stays out of the test.

Every endpoint takes the collection name in the BODY, or in the query string
for the one read that needs it. There are no path parameters, and the first
two tests exist to keep it that way: the first draft had `PUT /state` declared
after `PUT /{name}`, which shadowed it completely and left the whole
active/hidden mechanism dead while every other test stayed green.
"""
import json
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


class _Entry:
    """A stand-in for `LocalEntry`. It has to carry every field the routes read.

    `parse_error` was added on 2026-09-27: `resolve(method="hough")` consults it
    to refuse a CIF whose structure could not be read uniquely, and this stub
    raised `AttributeError` instead. A stub that lags the real dataclass turns a
    route change into a test failure that looks like a route bug.
    """

    def __init__(self, key, sht=None, cif=None, parse_error=None):
        self.key, self.formula, self.space_group = key, key, ""
        self.display_label = key
        self.cif_path, self.xtal_path, self.sht_path = cif, None, sht
        self.elements = ()
        self.parse_error = parse_error


@pytest.fixture(autouse=True)
def client(tmp_path, monkeypatch):
    """autouse ON PURPOSE. Without it a later test added to this file would
    run against the real Database/Collections; the conftest guard would fail
    it, but only after it had tried."""
    pc.set_collections_dir_for_test(tmp_path / "Collections")
    import backend.api.services.crystal_hint_local_library as lib
    monkeypatch.setattr(lib, "get_index", lambda: {
        "Al": _Entry("Al", sht=tmp_path / "Al.sht", cif=tmp_path / "Al.cif"),
        "Si": _Entry("Si", cif=tmp_path / "Si.cif"),
    })
    monkeypatch.setattr(route, "_master_map", lambda idx: {})
    app = FastAPI()
    app.include_router(route.router, prefix="/api/phase-collections")
    yield TestClient(app)
    pc.set_collections_dir_for_test(None)


def test_no_route_has_a_path_parameter(client):
    """The structural guard. A `{name}` segment reintroduced anywhere would
    shadow one of the literal routes depending on declaration order, and the
    symptom is a 404 that names a collection nobody asked for."""
    paths = [r.path for r in route.router.routes]
    assert paths, "router has no routes"
    assert not [q for q in paths if "{" in q], paths


def test_state_round_trips(client):
    client.post("/api/phase-collections/", json={"name": "M"})
    r = client.put("/api/phase-collections/state",
                   json={"active": "M", "hidden": ["M"]})
    assert r.status_code == 200, r.text
    assert client.get("/api/phase-collections/").json()["state"]["active"] == "M"


def test_empty_library_returns_a_usable_shape(client):
    r = client.get("/api/phase-collections/")
    assert r.status_code == 200
    body = r.json()
    assert body["collections"] == []
    assert sorted(m["key"] for m in body["unassigned"]) == ["Al", "Si"]
    assert body["problems"] == []
    # The schema travels with the state file, so it moves with the format.
    assert body["state"] == {"schema": pc.COLLECTION_SCHEMA,
                             "active": None, "hidden": []}


def test_a_created_collection_carries_who_and_when(client):
    """The measurement the 2026-09-27 review made: `create(name=...)` produced
    `author: ''`, `updated: ''`, `created: ''`. No code path filled any of them,
    while schema 2's commit message claimed they "already answer 'who was that'
    when the folder is copied by hand".

    Through the ROUTE, because that is where the gap was -- the service could
    carry the fields and did, which is exactly what the old service-level test
    proved and why it could not fail.
    """
    assert client.post("/api/phase-collections/",
                       json={"name": "Matrix", "author": "irmgard"}
                       ).status_code == 200
    doc = json.loads((pc.collections_dir() / "Matrix.json")
                     .read_text(encoding="utf-8"))
    assert doc["author"] == "irmgard"
    assert doc["created"] and doc["updated"], doc
    assert doc["created"].endswith("+00:00")


def test_a_created_collection_without_an_author_says_unknown(client):
    """The frontend sends none today. "unknown" is what the synonym store writes
    in the same situation, and it is honest: somebody did this and did not say."""
    client.post("/api/phase-collections/", json={"name": "Matrix"})
    doc = json.loads((pc.collections_dir() / "Matrix.json")
                     .read_text(encoding="utf-8"))
    assert doc["author"] == "unknown"


def test_update_can_set_the_author_and_leaves_it_alone_otherwise(client):
    client.post("/api/phase-collections/", json={"name": "Matrix",
                                                 "author": "irmgard"})
    # No `author` in the body: the stored one survives, as with `description`.
    assert client.put("/api/phase-collections/update",
                      json={"name": "Matrix", "description": "d"}
                      ).status_code == 200
    doc = json.loads((pc.collections_dir() / "Matrix.json")
                     .read_text(encoding="utf-8"))
    assert doc["author"] == "irmgard" and doc["description"] == "d"

    assert client.put("/api/phase-collections/update",
                      json={"name": "Matrix", "author": "sebastian"}
                      ).status_code == 200
    doc = json.loads((pc.collections_dir() / "Matrix.json")
                     .read_text(encoding="utf-8"))
    assert doc["author"] == "sebastian"


def test_create_then_list_carries_counts_and_capabilities(client):
    client.post("/api/phase-collections/", json={"name": "Matrix"})
    client.post("/api/phase-collections/members",
                json={"name": "Matrix", "keys": ["Al", "Si"]})
    body = client.get("/api/phase-collections/").json()
    col = body["collections"][0]
    assert col["name"] == "Matrix"
    assert col["member_count"] == 2 and col["effective_member_count"] == 2
    caps = {m["key"]: (m["has_sht"], m["has_master"]) for m in col["members"]}
    assert caps == {"Al": (True, False), "Si": (False, False)}
    assert body["unassigned"] == []


def test_creating_a_name_that_exists_is_refused_and_changes_nothing(client):
    client.post("/api/phase-collections/", json={"name": "Matrix"})
    client.post("/api/phase-collections/members",
                json={"name": "Matrix", "keys": ["Al"]})
    r = client.post("/api/phase-collections/", json={"name": "matrix"})
    assert r.status_code == 409, r.text
    cols = client.get("/api/phase-collections/").json()["collections"]
    assert len(cols) == 1
    assert [m["key"] for m in cols[0]["members"]] == ["Al"]


def test_removing_a_member_affects_only_the_named_collection(client):
    # Whole-branch review (2026-09-26), "tests that do not bite": this test
    # used to pair "A" with a NON-exclusive "Arbeitsauswahl" second
    # collection. `assign(keys, None)` (the bug this test exists to catch —
    # see the sibling service-level test's own docstring) skips non-exclusive
    # collections unconditionally, so it left "Arbeitsauswahl" holding Si
    # either way and the test green under the bug. Two EXCLUSIVE collections
    # are needed to tell `unassign_from` (touches only the named collection)
    # apart from `assign(keys, None)` (strips every exclusive collection) —
    # and since an exclusive collection cannot legitimately share a member
    # through the normal API (both `save()` and `assign()` cross-strip), B's
    # copy of Si is written directly to the collection file on disk, the same
    # hand-edit a duplicate-key state (spec 10.3) always is.
    client.post("/api/phase-collections/", json={"name": "A"})
    client.post("/api/phase-collections/", json={"name": "B"})
    client.post("/api/phase-collections/members",
                json={"name": "A", "keys": ["Al", "Si"]})
    p = pc.collections_dir() / "B.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["members"] = [{"key": "Si"}]
    p.write_text(json.dumps(d), encoding="utf-8", newline="\n")

    r = client.request("DELETE", "/api/phase-collections/members",
                       json={"name": "A", "keys": ["Si"]})
    assert r.status_code == 200, r.text
    cols = {c["name"]: [m["key"] for m in c["members"]]
            for c in client.get("/api/phase-collections/").json()["collections"]}
    assert cols["A"] == ["Al"]
    # This is the assertion the old, non-exclusive fixture could never fail:
    # under assign(keys, None), B (now exclusive) would ALSO lose Si here.
    assert cols["B"] == ["Si"]


def test_removing_a_member_from_an_unknown_collection_is_a_404(client):
    r = client.request("DELETE", "/api/phase-collections/members",
                       json={"name": "Nope", "keys": ["Al"]})
    assert r.status_code == 404


def test_effective_count_includes_children(client):
    client.post("/api/phase-collections/", json={"name": "P"})
    client.post("/api/phase-collections/", json={"name": "C", "parent": "P"})
    client.post("/api/phase-collections/members", json={"name": "P", "keys": ["Al"]})
    client.post("/api/phase-collections/members", json={"name": "C", "keys": ["Si"]})
    cols = {c["name"]: c for c in
            client.get("/api/phase-collections/").json()["collections"]}
    assert cols["P"]["member_count"] == 1
    assert cols["P"]["effective_member_count"] == 2


def test_update_can_promote_a_child_to_the_top_level(client):
    client.post("/api/phase-collections/", json={"name": "P"})
    client.post("/api/phase-collections/", json={"name": "C", "parent": "P"})
    r = client.put("/api/phase-collections/update",
                   json={"name": "C", "clear_parent": True})
    assert r.status_code == 200, r.text
    cols = {c["name"]: c for c in
            client.get("/api/phase-collections/").json()["collections"]}
    assert cols["C"]["parent"] is None


def test_resolve_returns_paths_in_collection_order_and_names_the_misses(client):
    client.post("/api/phase-collections/", json={"name": "M"})
    client.post("/api/phase-collections/members",
                json={"name": "M", "keys": ["Al", "Si"]})
    body = client.get("/api/phase-collections/resolve",
                      params={"name": "M", "method": "spherical"}).json()
    assert len(body["paths"]) == 1 and body["paths"][0].endswith("Al.sht")
    assert body["missing"] == [{"key": "Si", "reason": "no_sht"}]


def test_resolve_rejects_an_unknown_method(client):
    client.post("/api/phase-collections/", json={"name": "M"})
    r = client.get("/api/phase-collections/resolve",
                   params={"name": "M", "method": "magic"})
    assert r.status_code == 422


def test_a_name_with_a_slash_works_everywhere(client):
    """"Al / Si" is a legitimate name. It is why there are no path parameters."""
    assert client.post("/api/phase-collections/",
                       json={"name": "Al / Si"}).status_code == 200
    assert client.post("/api/phase-collections/members",
                       json={"name": "Al / Si", "keys": ["Al"]}).status_code == 200
    assert client.get("/api/phase-collections/resolve",
                      params={"name": "Al / Si", "method": "hough"}
                      ).status_code == 200
    assert client.patch("/api/phase-collections/rename",
                        json={"name": "Al / Si", "new_name": "AlSi"}
                        ).status_code == 200
    names = [c["name"] for c in
             client.get("/api/phase-collections/").json()["collections"]]
    assert names == ["AlSi"]


def test_deleting_a_collection_reports_it_and_keeps_the_phases(client):
    client.post("/api/phase-collections/", json={"name": "M"})
    client.post("/api/phase-collections/members", json={"name": "M", "keys": ["Al"]})
    r = client.request("DELETE", "/api/phase-collections/",
                       json={"name": "M"})
    assert r.status_code == 200, r.text
    body = client.get("/api/phase-collections/").json()
    assert body["collections"] == []
    assert sorted(m["key"] for m in body["unassigned"]) == ["Al", "Si"]


def test_suggest_proposes_from_the_fake_library_and_writes_nothing(client):
    r = client.post("/api/phase-collections/suggest")
    assert r.status_code == 200, r.text
    suggestions = {s["name"]: s["keys"] for s in r.json()["suggestions"]}
    assert suggestions["Matrix and pure metals"] == ["Al", "Si"]
    # Nothing was created by asking for a proposal.
    assert client.get("/api/phase-collections/").json()["collections"] == []


def test_suggest_apply_skips_a_name_that_already_exists_without_overwriting_it(client):
    client.post("/api/phase-collections/", json={"name": "Matrix and pure metals"})
    client.post("/api/phase-collections/members",
               json={"name": "Matrix and pure metals", "keys": ["Al"]})
    r = client.post("/api/phase-collections/suggest/apply",
                    json={"accepted": ["Matrix and pure metals"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] == []
    assert body["skipped"] == ["Matrix and pure metals"]
    cols = {c["name"]: [m["key"] for m in c["members"]]
            for c in client.get("/api/phase-collections/").json()["collections"]}
    # The suggestion would have proposed ["Al", "Si"]; the existing collection
    # keeps exactly what the user put in it.
    assert cols["Matrix and pure metals"] == ["Al"]
