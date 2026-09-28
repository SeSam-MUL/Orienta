"""Schema 2: a stable identity, tags instead of folders, and older files still load.

Spec §2.5.1 named four places where "the server stays as it is" was not true, and
the first is the one that could have destroyed data: `from_dict` refused every
schema but its own, so bumping the number would have made every collection on disk
unreadable -- including the two live ones written during the phase-collections
work, one of them carrying `exclusive: true`.

The other three are semantic: `unassigned_keys` and `suggest` both gated on
`exclusive`, which under tags is never set, so "not yet filed" would have become
the whole library permanently while the suggestion engine offered phases that were
already grouped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services import phase_collections as pc  # noqa: E402

#: A schema-1 file exactly as the live ones on disk are written.
SCHEMA_1 = {
    "schema": 1,
    "name": "crop1",
    "parent": None,
    "exclusive": True,
    "description": "",
    "created": "2026-09-27T10:00:00+00:00",
    "members": [{"key": "Al"}, {"key": "sd_0302719", "formula": "Mn0.5Fe0.5Al5Si0.68"}],
}


@pytest.fixture(autouse=True)
def store(tmp_path):
    pc.set_collections_dir_for_test(tmp_path / "Collections")
    try:
        yield
    finally:
        pc.set_collections_dir_for_test(None)


def _write(stem: str, payload: dict) -> Path:
    d = pc.collections_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    return p


# --- migration ------------------------------------------------------------

def test_a_schema_1_file_still_loads_and_keeps_its_members():
    """The one that would have been data loss.

    Refusing it would not have corrupted anything -- it would have made every
    existing collection vanish from the app while sitting intact on disk, which
    is worse, because nothing looks broken.
    """
    _write("crop1", SCHEMA_1)
    cols, problems = pc.load_all()
    assert [c.name for c in cols] == ["crop1"]
    assert [m.key for m in cols[0].members] == ["Al", "sd_0302719"]
    assert cols[0].members[1].formula == "Mn0.5Fe0.5Al5Si0.68"
    assert problems == []


def test_a_schema_1_file_gains_an_id_from_its_filename():
    """The file stem is the only stable thing such a file carries.

    And it is what every existing reference was keyed on by way of the name, so
    ids minted this way keep `active`, `hidden` and `parent` working.
    """
    _write("crop1", SCHEMA_1)
    assert pc.load_all()[0][0].id == "crop1"


def test_a_file_from_a_newer_build_is_still_refused():
    """Forward compatibility is not symmetric with backward.

    Reading a shape nobody here knows would silently change which phases a run
    considers; the file is reported and skipped instead.
    """
    _write("good", SCHEMA_1)
    _write("future", {**SCHEMA_1, "schema": 99, "name": "future"})
    cols, problems = pc.load_all()
    assert [c.name for c in cols] == ["crop1"]
    assert [p["kind"] for p in problems] == ["bad_schema"]
    assert "99" in problems[0]["detail"]


def test_saving_writes_schema_2_and_keeps_exclusive_on_disk():
    """CORRECTED 2026-09-27: `exclusive` is NOT kept for backward readability.

    Measured against the pre-change reader: it raises on `schema != 1` before it
    ever looks at the field, so writing schema 2 is a one-way door regardless. The
    field is kept because files written BEFORE schema 2 carry `exclusive: false`
    (`livetest-2026-09-27-workingset.json` in this tree does), `from_dict`
    preserves it, and `databaseGrouping.js:134,140` still branches on it.
    """
    pc.save(pc.PhaseCollection(name="A", members=[pc.PhaseMember(key="Al")]))
    doc = json.loads((pc.collections_dir() / "A.json").read_text(encoding="utf-8"))
    assert doc["schema"] == 2
    assert "exclusive" in doc, "a schema-1 build checks this field strictly"
    assert "id" in doc


# --- the three semantic places -------------------------------------------

def test_a_phase_in_any_group_counts_as_filed():
    """Otherwise "not yet filed" is the whole library, forever.

    `unassigned_keys` gated on `exclusive`; with tags nothing is exclusive, so
    `filed` would always have been empty.
    """
    idx = {"Al": object(), "Si": object(), "Ni": object()}
    cols = [pc.PhaseCollection(name="A", exclusive=False,
                               members=[pc.PhaseMember(key="Al")])]
    assert pc.unassigned_keys(idx, cols) == ["Si", "Ni"]


def test_being_in_two_groups_is_not_reported_as_a_problem():
    cols = [pc.PhaseCollection(name="A", members=[pc.PhaseMember(key="Al")]),
            pc.PhaseCollection(name="B", members=[pc.PhaseMember(key="Al")])]
    assert pc.unassigned_keys({"Al": object()}, cols) == []
    _write("A", {**SCHEMA_1, "name": "A", "members": [{"key": "Al"}]})
    _write("B", {**SCHEMA_1, "name": "B", "members": [{"key": "Al"}]})
    assert [p["kind"] for p in pc.load_all()[1]] == []


def test_saving_one_group_never_edits_another():
    """The quietest possible destruction: a save that empties a group you are
    not looking at."""
    pc.save(pc.PhaseCollection(name="Al systems",
                               members=[pc.PhaseMember(key="Al2Cu")]))
    pc.save(pc.PhaseCollection(name="Cu systems",
                               members=[pc.PhaseMember(key="Al2Cu")]))
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {"Al systems": ["Al2Cu"], "Cu systems": ["Al2Cu"]}


def test_move_is_the_only_way_to_remove_from_the_others():
    pc.save(pc.PhaseCollection(name="A", members=[pc.PhaseMember(key="Al")]))
    pc.save(pc.PhaseCollection(name="B", members=[]))
    pc.assign(["Al"], "B")
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by["A"] == ["Al"] and by["B"] == ["Al"]
    pc.assign(["Al"], "B", move=True)
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by["A"] == [] and by["B"] == ["Al"]


def test_unfiling_everywhere_still_works():
    """`to_name=None` means "remove from every group" and always did."""
    pc.save(pc.PhaseCollection(name="A", members=[pc.PhaseMember(key="Al")]))
    pc.save(pc.PhaseCollection(name="B", members=[pc.PhaseMember(key="Al")]))
    pc.assign(["Al"], None)
    by = {c.name: [m.key for m in c.members] for c in pc.load_all()[0]}
    assert by == {"A": [], "B": []}


# --- references are identities, with a name fallback for one version ------

def test_a_reference_resolves_by_name_first_then_by_identity():
    """REVERSED 2026-09-27 by f7's closing review, and it was a data-loss path.

    This used to assert that the IDENTITY wins, with the reasoning "it cannot
    have been typed by accident". True, and not the point: a group literally NAMED
    `Al_systems`, beside "Al systems" whose minted id IS `Al_systems`, resolved to
    the second — so `delete("Al_systems")` removed the group the user did not mean,
    members and all, silently. A name is what the user typed and what the UI
    shows. When the two spaces collide, the typed thing wins.

    The collision SOURCE is closed separately: `_mint_id` may no longer produce an
    id that already exists as a name, so from this code's side the spaces cannot
    overlap at all.
    """
    a = pc.PhaseCollection(name="Alpha", id="k1")
    b = pc.PhaseCollection(name="k1", id="k2")
    # "k1" is BOTH a's id and b's name.
    assert pc.resolve_ref("k1", [a, b]) is b, "the collection NAMED k1"
    # An id still resolves when no name matches -- the live `_local.json`
    # reference depends on that path.
    assert pc.resolve_ref("k2", [a, b]) is b
    assert pc.resolve_ref("Alpha", [a, b]) is a
    assert pc.resolve_ref("alpha", [a, b]) is a, "names match case-insensitively"
    assert pc.resolve_ref("nothing", [a, b]) is None
    assert pc.resolve_ref("", [a, b]) is None
    assert pc.resolve_ref(None, [a, b]) is None


def test_the_real_local_state_file_still_finds_its_collection():
    """The case that made the fallback non-negotiable, reproduced exactly.

    `Database/Collections/_local.json` in this tree holds
    `active: "_livetest-2026-09-27-crop1"` -- the NAME, with a leading
    underscore -- while the collection file is `livetest-2026-09-27-crop1.json`,
    so its minted id has no underscore. Resolving ids only would have dropped
    the active collection on first start after the update, silently.
    """
    _write("livetest-2026-09-27-crop1",
           {**SCHEMA_1, "name": "_livetest-2026-09-27-crop1"})
    cols, _ = pc.load_all()
    (c,) = cols
    assert c.id == "livetest-2026-09-27-crop1"
    assert c.name == "_livetest-2026-09-27-crop1"
    state = pc.resolve_state({"active": "_livetest-2026-09-27-crop1",
                              "hidden": []}, cols)
    assert state["active"] == "livetest-2026-09-27-crop1", (
        "the name must resolve, and be stored as the identity")


def test_a_parent_named_in_an_old_file_becomes_an_identity_on_read():
    _write("parent", {**SCHEMA_1, "name": "Parent", "parent": None})
    _write("child", {**SCHEMA_1, "name": "Child", "parent": "Parent"})
    cols, problems = pc.load_all()
    by = {c.name: c for c in cols}
    assert problems == []
    assert by["Child"].parent == by["Parent"].id == "parent"
    assert [c.name for c in pc.children_of(by["Parent"], cols)] == ["Child"]


def test_a_parent_that_is_nobody_is_reported_and_dropped():
    _write("child", {**SCHEMA_1, "name": "Child", "parent": "Ghost"})
    cols, problems = pc.load_all()
    assert cols[0].parent is None
    assert [p["kind"] for p in problems] == ["orphan_parent"]


def test_a_hand_edited_file_cannot_be_its_own_parent():
    """CORRECTED 2026-09-27: this used to say "infinite recursion in
    `effective_members`". It is not, and the review measured it: `children_of`
    filters `c is not parent` and `effective_members` never recurses into
    grandchildren, so a self-parent gives `children_of == []` and the members come
    back normally. With the guard removed the old test failed on
    `assert 'solo' is None` in 1.67 s -- not on a hang.

    The guard stays, for the real reason: without it the collection DISAPPEARS
    from the manager, which renders only top-level rows and reaches children
    beneath them, so one that is its own parent is excluded from the top level and
    nested under a row that is never drawn.

    The `effective_members` assertion below now documents the absence of recursion
    instead of implying it was ever there.
    """
    _write("solo", {**SCHEMA_1, "name": "Solo", "parent": "Solo"})
    cols, problems = pc.load_all()
    assert cols[0].parent is None
    assert [p["kind"] for p in problems] == ["self_parent"]
    assert [m.key for m in pc.effective_members("Solo")] == ["Al", "sd_0302719"]


def test_an_unresolvable_state_reference_is_kept_not_dropped():
    """A collection file that is momentarily missing -- a half-copied folder, a
    share not mounted yet -- must not cost the user their active collection
    permanently. Only `delete()` clears a reference, because only there is the
    collection known to be gone."""
    state = pc.resolve_state({"active": "Not here yet", "hidden": ["Nor this"]}, [])
    assert state["active"] == "Not here yet"
    assert state["hidden"] == ["Nor this"]


def test_two_spellings_of_one_collection_collapse_in_hidden():
    c = pc.PhaseCollection(name="Alpha", id="k1")
    state = pc.resolve_state({"active": None, "hidden": ["Alpha", "k1"]}, [c])
    assert state["hidden"] == ["k1"], "one collection, hidden once"


def test_deleting_clears_the_state_written_in_either_spelling():
    pc.save(pc.PhaseCollection(name="Gone", members=[pc.PhaseMember(key="Al")]))
    ident = pc.load_all()[0][0].id
    pc.save_state({"active": "Gone", "hidden": [ident]})
    assert pc.delete("Gone") is True
    st = pc.load_state()
    assert st["active"] is None, "a name reference must be cleared too"
    assert st["hidden"] == [], "and an identity reference"


# --- author and time ------------------------------------------------------

def test_the_author_and_timestamp_round_trip():
    """REWRITTEN after the 2026-09-27 review: this used to be the only test of
    these fields, and it could not fail.

    It set both by hand on the dataclass and read them back, so it proved that
    `to_dict`/`from_dict` carry two strings -- and it passed in a world where NO
    CODE PATH ever filled either. Measured then: `create(name="Al systems")`
    produced `author: ''`, `updated: ''`, `created: ''`, which made schema 2's own
    commit message ("they already answer 'who was that' when the folder is copied
    by hand") false. The tests below are the ones that would have caught that.
    """
    c = pc.PhaseCollection(name="A", author="sebastian")
    pc.save(c)
    doc = json.loads((pc.collections_dir() / "A.json").read_text(encoding="utf-8"))
    assert doc["author"] == "sebastian"
    assert pc.load_all()[0][0].author == "sebastian"


def test_every_save_stamps_updated():
    """Not "the field exists" -- the field has a VALUE, put there by the code
    under test rather than by the test."""
    pc.save(pc.PhaseCollection(name="A", members=[pc.PhaseMember(key="Al")]))
    first = pc.load_all()[0][0]
    assert first.updated, "a saved collection must carry a timestamp"
    assert first.updated.endswith("+00:00"), f"UTC, got {first.updated!r}"
    assert first.created == first.updated, "on the first write they agree"


def test_created_is_stamped_once_and_updated_moves(monkeypatch):
    """Re-stamping `created` on every member change would erase the one date that
    says when the group came into being.

    THE CLOCK IS PINNED, and it has to be: `_now_iso` truncates to the second, so
    two saves inside one test share a timestamp and `created == born` holds even
    with `created` re-stamped every time. I know because I mutated it that way and
    the first version of this test SURVIVED -- the third time this shift that a
    guard could not express the difference it was named for.
    """
    clock = iter(["2026-01-01T00:00:00+00:00", "2026-06-15T12:30:00+00:00"])
    monkeypatch.setattr(pc, "_now_iso", lambda: next(clock))

    pc.save(pc.PhaseCollection(name="A", members=[pc.PhaseMember(key="Al")]))
    born = pc.load_all()[0][0].created
    assert born == "2026-01-01T00:00:00+00:00"

    c = pc.load_all()[0][0]
    c.members.append(pc.PhaseMember(key="Si"))
    pc.save(c)

    again = pc.load_all()[0][0]
    assert again.created == born, "created never moves"
    assert again.updated == "2026-06-15T12:30:00+00:00", "updated does"


def test_an_unstated_author_is_recorded_as_unknown():
    """The same convention `phase_synonyms.set_names` uses, so both stores in
    `Database/` read the same way. A write by somebody who did not say who they
    were is a fact worth recording, not an empty field to guess at later."""
    pc.save(pc.PhaseCollection(name="A"))
    assert pc.load_all()[0][0].author == "unknown"


def test_a_stated_author_is_never_overwritten():
    """`save()` fills the field only when it is empty -- otherwise every member
    change would erase whoever made the group."""
    pc.save(pc.PhaseCollection(name="A", author="irmgard"))
    c = pc.load_all()[0][0]
    c.members.append(pc.PhaseMember(key="Al"))
    pc.save(c)
    assert pc.load_all()[0][0].author == "irmgard"
