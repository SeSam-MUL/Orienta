"""Every write endpoint takes an IDENTITY as well as a name (spec §2.5.1).

Half the API was made identity-aware and half was not, and the summary I sent
said "send the id from now on" -- which would have made every group
unmanageable. Measured before the fix, on a collection named "Al systems"
(minted id `Al_systems`):

    assign         with the id -> KeyError
    unassign_from  with the id -> KeyError
    rename         with the id -> KeyError
    delete         with the id -> False        <- the worse one

`False` is the worse one because it means "there is no such collection": a
caller holding a good identity is told its collection does not exist, and
nothing looks broken. The other three at least fail loudly.

EVERY test here asserts the id and the name DIFFER first. Without that the
suite would pass just as well with no resolution at all -- `_mint_id` turns
"Al systems" into `Al_systems`, but a single-word name mints an id equal to
itself, and a test built on one of those proves nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services import phase_collections as pc  # noqa: E402


@pytest.fixture(autouse=True)
def store(tmp_path):
    pc.set_collections_dir_for_test(tmp_path / "Collections")
    try:
        yield
    finally:
        pc.set_collections_dir_for_test(None)


def _made(name: str, keys=()) -> pc.PhaseCollection:
    """Create a collection whose id is NOT its name, and prove it."""
    pc.save(pc.PhaseCollection(
        name=name, members=[pc.PhaseMember(key=k) for k in keys]))
    c = {x.name: x for x in pc.load_all()[0]}[name]
    assert c.id and c.id != c.name, (
        f"this test needs an id distinct from the name; got {c.id!r} for "
        f"{c.name!r} -- pick a name with a space")
    return c


# --- delete: b9 asked for this one first ----------------------------------

def test_delete_by_identity_removes_that_collections_file():
    a = _made("Al systems", ["Al"])
    b = _made("Cu systems", ["Al2Cu"])
    before = {p.name for p in pc.collections_dir().glob("*.json")}

    assert pc.delete(a.id) is True, "an identity must not read as 'no such collection'"

    after = {p.name for p in pc.collections_dir().glob("*.json")}
    assert len(before - after) == 1, f"exactly one file goes: {before - after}"
    assert [c.name for c in pc.load_all()[0]] == [b.name], "and it is the right one"
    assert [m.key for m in pc.load_all()[0][0].members] == ["Al2Cu"], (
        "the survivor keeps its members")


def test_delete_by_identity_deletes_the_resolved_collections_path():
    """Not merely 'a file disappeared': the file that disappeared must be the
    one belonging to the resolved collection."""
    a = _made("Al systems", ["Al"])
    _made("Cu systems", ["Al2Cu"])
    a_path = pc._path_of(a.name)
    assert a_path is not None and a_path.is_file()

    assert pc.delete(a.id) is True
    assert not a_path.exists(), "the resolved collection's own file"
    assert pc._path_of("Cu systems").is_file(), "and nothing else"


def test_delete_of_an_unknown_reference_still_answers_false():
    """Unchanged, and it must stay that way: `False` is the honest answer when
    there really is no such collection. The defect was that a VALID id got the
    same answer."""
    _made("Al systems", ["Al"])
    assert pc.delete("no-such-thing") is False
    assert [c.name for c in pc.load_all()[0]] == ["Al systems"]


def test_delete_by_identity_clears_the_state_that_pointed_at_it():
    a = _made("Al systems", ["Al"])
    pc.save_state({"active": a.id, "hidden": [a.name]})
    assert pc.delete(a.id) is True
    st = pc.load_state()
    assert st["active"] is None and st["hidden"] == []


# --- the other three -------------------------------------------------------

def test_assign_by_identity_files_the_phase():
    a = _made("Al systems", [])
    pc.assign(["Al2Cu"], a.id)
    assert [m.key for m in pc.load_all()[0][0].members] == ["Al2Cu"]


def test_assign_by_identity_can_still_move():
    a = _made("Al systems", ["Al2Cu"])
    b = _made("Cu systems", [])
    pc.assign(["Al2Cu"], b.id, move=True)
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {a.name: [], b.name: ["Al2Cu"]}


def test_unassign_from_by_identity_touches_only_that_collection():
    a = _made("Al systems", ["Al", "Al2Cu"])
    b = _made("Cu systems", ["Al2Cu"])
    pc.unassign_from(a.id, ["Al2Cu"])
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {a.name: ["Al"], b.name: ["Al2Cu"]}


def test_rename_by_identity_keeps_the_identity_and_the_members():
    a = _made("Al systems", ["Al"])
    pc.rename(a.id, "Aluminium systems")
    (c,) = pc.load_all()[0]
    assert c.name == "Aluminium systems"
    assert c.id == a.id, "a rename must never mint a new identity"
    assert [m.key for m in c.members] == ["Al"]


def test_rename_by_identity_leaves_no_second_file_behind():
    """The old file must go. `rename` removes `_path_of(old)`, and `old` is the
    caller's argument -- an id there would have deleted nothing and left the
    collection on disk twice."""
    a = _made("Al systems", ["Al"])
    pc.rename(a.id, "Aluminium systems")
    assert len(list(pc.collections_dir().glob("*.json"))) == 1
    assert [c.name for c in pc.load_all()[0]] == ["Aluminium systems"]


def test_unknown_reference_still_raises_for_the_three_that_raise():
    _made("Al systems", ["Al"])
    for call in (lambda: pc.assign(["Al"], "ghost"),
                 lambda: pc.unassign_from("ghost", ["Al"]),
                 lambda: pc.rename("ghost", "x")):
        with pytest.raises(KeyError):
            call()


# --- names keep working ----------------------------------------------------

def test_every_entry_point_still_takes_a_name():
    """The fallback the frontend uses today. It is NOT a formality: f7 sends
    names, and a change that made ids work by breaking names would pass every
    test above."""
    a = _made("Al systems", ["Al"])
    pc.assign(["Al2Cu"], a.name)
    pc.unassign_from(a.name, ["Al"])
    pc.rename(a.name, "Renamed systems")
    assert [m.key for m in pc.load_all()[0][0].members] == ["Al2Cu"]
    assert pc.delete("Renamed systems") is True
    assert pc.load_all()[0] == []


# --- through the route, which is what the frontend actually calls ----------

class _Entry:
    def __init__(self, key):
        self.key, self.formula, self.space_group = key, key, ""
        self.display_label = key
        self.cif_path = self.xtal_path = self.sht_path = None
        self.elements = ()
        self.parse_error = None


@pytest.fixture
def client(tmp_path, monkeypatch):
    from backend.api.routes import phase_collections as route
    import backend.api.services.crystal_hint_local_library as lib
    monkeypatch.setattr(lib, "get_index",
                        lambda: {"Al": _Entry("Al"), "Al2Cu": _Entry("Al2Cu")})
    monkeypatch.setattr(route, "_master_map", lambda idx: {})
    app = FastAPI()
    app.include_router(route.router, prefix="/api/phase-collections")
    return TestClient(app)


def test_post_members_adds_by_default_and_moves_on_request(client):
    """b9's decision: `move` is a flag on the route, default false.

    Both halves, because a default of true and a default of false both make the
    happy path work -- only the other group's members tell them apart.
    """
    a = _made("Al systems", ["Al2Cu"])
    b = _made("Cu systems", [])

    # Default: adds, and does NOT evict.
    assert client.post("/api/phase-collections/members",
                       json={"name": b.name, "keys": ["Al2Cu"]}
                       ).status_code == 200
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {a.name: ["Al2Cu"], b.name: ["Al2Cu"]}, (
        "a tag is not a folder: adding must not evict")

    # move: the named action.
    assert client.post("/api/phase-collections/members",
                       json={"name": b.name, "keys": ["Al2Cu"], "move": True}
                       ).status_code == 200
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {a.name: [], b.name: ["Al2Cu"]}


def test_delete_members_has_no_move_flag(client):
    """`DELETE /members` keeps the model without `move`. A field a second endpoint
    accepts and ignores reads to the caller as a capability, and this project has
    shipped exactly that twice."""
    from backend.api.routes import phase_collections as route
    assert "move" in route.AddMembersRequest.model_fields
    assert "move" not in route.MembersRequest.model_fields
    # And pydantic must not quietly accept it on the DELETE body either way:
    # whatever it does, the phase must still only leave the named collection.
    a = _made("Al systems", ["Al", "Al2Cu"])
    b = _made("Cu systems", ["Al2Cu"])
    client.request("DELETE", "/api/phase-collections/members",
                   json={"name": a.name, "keys": ["Al2Cu"], "move": True})
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {a.name: ["Al"], b.name: ["Al2Cu"]}, (
        "removal touches only the named collection, flag or no flag")


def test_a_literal_name_beats_a_minted_identity():
    """f7's closing review, and it was a data-loss path I created.

    I had `resolve_ref` prefer the id, with the reasoning "an identity cannot have
    been typed by accident". True, and not the point: a group literally NAMED
    `Al_systems`, beside "Al systems" whose minted id IS `Al_systems`, resolved to
    the second. `delete("Al_systems")` then removed the group the user did not
    mean, members and all, silently.

    A name is what the user typed and what the UI shows. When the two spaces
    collide, the typed thing wins.
    """
    pc.save(pc.PhaseCollection(name="Al systems",
                               members=[pc.PhaseMember(key="Al")]))
    spaced = {c.name: c for c in pc.load_all()[0]}["Al systems"]
    assert spaced.id == "Al_systems", (
        "this test needs the minted id to equal the other group's name")

    # Planted directly: `save()` now refuses to MINT this collision, and that is
    # the other half of the fix -- but a file can still arrive by hand.
    d = pc.collections_dir()
    (d / "literal.json").write_text(json.dumps({
        "schema": 2, "id": "literal", "name": "Al_systems", "parent": None,
        "members": [{"key": "Si"}]}), encoding="utf-8", newline="\n")

    hit = pc.resolve_ref("Al_systems", pc.load_all()[0])
    assert hit is not None and hit.name == "Al_systems", "the typed name wins"
    assert hit.id == "literal"

    assert pc.delete("Al_systems") is True
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {"Al systems": ["Al"]}, (
        "the group that merely had that string as its id must survive, with its "
        f"members: {by}")


def test_an_identity_still_resolves_when_no_name_matches():
    """The other direction. Names winning must not mean ids stop working -- the
    live `_local.json` reference depends on the id path."""
    pc.save(pc.PhaseCollection(name="Al systems", members=[pc.PhaseMember(key="Al")]))
    c = pc.load_all()[0][0]
    hit = pc.resolve_ref(c.id, pc.load_all()[0])
    assert hit is not None and hit.name == "Al systems"


def test_a_minted_identity_never_equals_an_existing_name():
    """The collision SOURCE, closed. Minting only checked ids, so it happily
    produced a string that already named another group."""
    pc.save(pc.PhaseCollection(name="Al_systems"))          # a literal name
    pc.save(pc.PhaseCollection(name="Al systems"))          # would mint Al_systems
    ids = {c.name: c.id for c in pc.load_all()[0]}
    assert ids["Al systems"] != "Al_systems", ids
    assert ids["Al systems"] == "Al_systems-2", ids
    # And nothing is now ambiguous:
    assert pc.resolve_ref("Al_systems", pc.load_all()[0]).name == "Al_systems"


def test_a_minted_identity_avoids_a_name_that_is_no_ones_identity():
    """The general case, and the one the test above does NOT cover.

    There the existing group is named `Al_systems` and its own id is `Al_systems`
    too, so checking ids alone already blocks the collision — I only know because
    the mutation "check ids only" survived that test and was caught by the
    case-variant one. Here the name belongs to NO id, so nothing but a name check
    can see it.
    """
    d = pc.collections_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "f1.json").write_text(json.dumps({
        "schema": 2, "id": "f1", "name": "Al_systems", "parent": None,
        "members": []}), encoding="utf-8", newline="\n")

    pc.save(pc.PhaseCollection(name="Al systems"))

    ids = {c.name: c.id for c in pc.load_all()[0]}
    assert ids == {"Al_systems": "f1", "Al systems": "Al_systems-2"}, ids
    assert pc.resolve_ref("Al_systems", pc.load_all()[0]).id == "f1"


def test_a_minted_identity_avoids_a_name_differing_only_in_case():
    """`resolve_ref` folds names, so an id that differs from a name only in case
    is still ambiguous to the lookup and is not free either."""
    pc.save(pc.PhaseCollection(name="AL_SYSTEMS"))
    pc.save(pc.PhaseCollection(name="Al systems"))
    ids = {c.name: c.id for c in pc.load_all()[0]}
    assert ids["Al systems"] == "Al_systems-2", ids


def test_update_takes_an_identity_like_the_other_write_routes(client):
    """f7's closing review: `PUT /update` was the ONE write route still doing an
    exact, case-sensitive dict lookup -- not even casefold -- so it answered 404
    for an identity while `PATCH /rename` on the same collection accepted both.
    Nesting is what this route does, so the mismatch landed on the action f7 had
    just built."""
    a = _made("Al systems", [])
    assert client.put("/api/phase-collections/update",
                      json={"name": a.id, "description": "by id"}
                      ).status_code == 200
    assert {c.name: c.description for c in pc.load_all()[0]} == {
        "Al systems": "by id"}

    # And a name whose casing differs, which it also used to refuse.
    assert client.put("/api/phase-collections/update",
                      json={"name": "al SYSTEMS", "description": "by casing"}
                      ).status_code == 200
    assert {c.description for c in pc.load_all()[0]} == {"by casing"}


def test_update_still_404s_for_something_that_is_neither(client):
    _made("Al systems", [])
    assert client.put("/api/phase-collections/update",
                      json={"name": "ghost", "description": "x"}
                      ).status_code == 404


def test_the_routes_accept_an_identity_in_the_body(client):
    a = _made("Al systems", [])
    assert client.post("/api/phase-collections/members",
                       json={"name": a.id, "keys": ["Al"]}).status_code == 200
    assert client.request("DELETE", "/api/phase-collections/members",
                          json={"name": a.id, "keys": ["Al"]}).status_code == 200
    assert client.patch("/api/phase-collections/rename",
                        json={"name": a.id, "new_name": "Neu"}).status_code == 200
    assert client.request("DELETE", "/api/phase-collections/",
                          json={"name": a.id}).status_code == 200
    assert client.get("/api/phase-collections/").json()["collections"] == []
