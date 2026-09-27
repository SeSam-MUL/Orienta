"""Phase collections: the filing system, and its refusal to guess.

A collection points at library phases by filename stem. Files get renamed and
drives get unmounted, so most of what is tested here is what happens when the
pointer no longer resolves — and the rule is always the same: say so, keep the
entry, never silently drop it. A collection that quietly loses members is worse
than no collection, because the user goes on trusting the count.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services import phase_collections as pc
from backend.api.services.phase_collections import (
    COLLECTION_SCHEMA,
    PhaseCollection,
    PhaseMember,
)


@pytest.fixture(autouse=True)
def collections_dir(tmp_path):
    """Every test gets its own directory. The user's Database/ is never touched
    — tests/conftest.py fails the test if it is."""
    pc.set_collections_dir_for_test(tmp_path / "Collections")
    yield tmp_path / "Collections"
    pc.set_collections_dir_for_test(None)


def _c(name, keys, parent=None, exclusive=True):
    return PhaseCollection(
        name=name, parent=parent, exclusive=exclusive,
        members=[PhaseMember(key=k) for k in keys],
    )


def test_round_trip_keeps_member_order():
    # Order is not cosmetic: PyEBSDIndex stops checking further phases once
    # seven bands match, so the order a collection stores changes results.
    pc.save(_c("Intermetallics", ["Al7FeCu2", "Al13Fe4", "Al6Fe"]))
    got = {c.name: c for c in pc.load_all()[0]}["Intermetallics"]
    assert [m.key for m in got.members] == ["Al7FeCu2", "Al13Fe4", "Al6Fe"]


def test_written_file_is_lf_and_utf8():
    pc.save(_c("Grieche", ["\u03b1-(AlMnSi)"]))
    raw = (pc.collections_dir() / "Grieche.json").read_bytes()
    assert b"\r\n" not in raw
    assert "\u03b1-(AlMnSi)" in json.loads(raw.decode("utf-8"))["members"][0]["key"]


def test_key_with_dots_parens_and_greek_survives_but_never_becomes_a_filename():
    key = "Al4Fe1.7Si (\u03c411)"
    pc.save(_c("Tau", [key]))
    files = list(pc.collections_dir().glob("*.json"))
    assert [f.name for f in files] == ["Tau.json"]
    assert [m.key for m in pc.load_all()[0][0].members] == [key]


def test_two_names_that_sanitise_alike_do_not_overwrite_each_other():
    # REVIEW FOCUS 2. "Al/Si" and "Al:Si" both sanitise to "Al_Si".
    pc.save(_c("Al/Si", ["Al"]))
    pc.save(_c("Al:Si", ["Si"]))
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert set(by_name) == {"Al/Si", "Al:Si"}
    assert [m.key for m in by_name["Al/Si"].members] == ["Al"]
    assert [m.key for m in by_name["Al:Si"].members] == ["Si"]


def test_a_second_collection_whose_name_differs_only_in_case_is_refused():
    # The file lookup matches casefold, so without this guard the second save
    # writes over the first one's file and its members are gone with no
    # message. Measured on this plan's first draft.
    pc.save(_c("Matrix", ["Al"]))
    with pytest.raises(ValueError, match="already exists"):
        pc.save(_c("matrix", ["Si"]))
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert set(by_name) == {"Matrix"}
    assert [m.key for m in by_name["Matrix"].members] == ["Al"]


def test_saving_the_same_collection_again_is_still_an_update():
    pc.save(_c("Matrix", ["Al"]))
    pc.save(_c("Matrix", ["Al", "Si"]))
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert [m.key for m in by_name["Matrix"].members] == ["Al", "Si"]


def test_renaming_onto_a_case_variant_of_another_collection_is_refused():
    pc.save(_c("a", ["Al"]))
    pc.save(_c("b", ["Si"]))
    with pytest.raises(ValueError, match="already exists"):
        pc.rename("b", "A")
    assert {c.name for c in pc.load_all()[0]} == {"a", "b"}


def test_renaming_a_collection_to_another_casing_of_itself_is_allowed():
    pc.save(_c("matrix", ["Al"]))
    pc.rename("matrix", "Matrix")
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert set(by_name) == {"Matrix"}
    assert [m.key for m in by_name["Matrix"].members] == ["Al"]


def test_unassign_from_takes_a_phase_out_of_one_collection_only():
    # The first draft's route called assign(keys, None), which strips the key
    # from EVERY collection: removing Si from A also removed it from B.
    #
    # Whole-branch review (2026-09-26), finding "tests that do not bite":
    # this test used to pair A with a NON-exclusive ("Arbeitsauswahl",
    # exclusive=False) second collection. assign(keys, None)'s strip loop
    # skips non-exclusive collections unconditionally (see assign()'s own
    # docstring), so reinstating the assign(keys, None) bug left this test
    # green: B keeps Si either way, purely because it is non-exclusive, not
    # because unassign_from() was called correctly. The working set's own
    # immunity to assign() is already pinned by
    # test_assign_into_working_set_leaves_exclusive_home_intact above; this
    # test's job is a DIFFERENT property (unassign_from touches only the
    # NAMED collection) and needs a second EXCLUSIVE collection to be able to
    # tell the two implementations apart at all.
    #
    # An exclusive collection cannot legitimately share a member through
    # save()/assign() (both cross-strip on save) — so B's copy of Si is
    # written directly to disk, the same hand-edit bypass
    # test_a_key_in_two_collections_is_reported_not_resolved uses. That is a
    # real state the app must tolerate (spec 10.3): a duplicate key, exactly
    # what a hand-edited file produces.
    pc.save(_c("A", ["Al", "Si"]))
    pc.save(_c("B", []))
    p = pc.collections_dir() / "B.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["members"] = [{"key": "Si"}]
    p.write_text(json.dumps(d), encoding="utf-8", newline="\n")

    pc.unassign_from("A", ["Si"])
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert [m.key for m in by_name["A"].members] == ["Al"]
    # This is the assertion the old, non-exclusive fixture could never fail:
    # under assign(keys, None), B (now exclusive) would ALSO lose Si here.
    assert [m.key for m in by_name["B"].members] == ["Si"]


def test_unassign_from_an_unknown_collection_raises_rather_than_mutating():
    pc.save(_c("A", ["Al"]))
    with pytest.raises(KeyError):
        pc.unassign_from("Nope", ["Al"])
    assert [m.key for m in pc.load_all()[0][0].members] == ["Al"]


def test_assign_moves_a_key_out_of_its_previous_collection():
    pc.save(_c("A", ["Al", "Si"]))
    pc.save(_c("B", []))
    pc.assign(["Si"], "B")
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert [m.key for m in by_name["A"].members] == ["Al"]
    assert [m.key for m in by_name["B"].members] == ["Si"]


def test_a_key_in_two_collections_is_reported_not_resolved():
    # REVIEW FOCUS 3. Hand-edited files; the app must not pick a winner.
    pc.save(_c("A", ["Al"]))
    pc.save(_c("B", ["Al"]))
    # Write A's file directly. It must be A, not B: saving B already stripped
    # Al out of A, so rewriting B changes nothing and the test would pass for
    # the wrong reason. Writing behind save()'s back is exactly what a hand
    # edit or a copied collection file does.
    p = pc.collections_dir() / "A.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["members"] = [{"key": "Al"}]
    p.write_text(json.dumps(d), encoding="utf-8", newline="\n")
    _cols, problems = pc.load_all()
    dupes = [pr for pr in problems if pr["kind"] == "duplicate_key"]
    assert len(dupes) == 1 and dupes[0]["key"] == "Al"
    assert set(dupes[0]["collections"]) == {"A", "B"}


def test_unknown_schema_is_refused_and_the_others_still_load():
    pc.save(_c("Good", ["Al"]))
    bad = pc.collections_dir() / "Bad.json"
    bad.write_text(json.dumps({"schema": 99, "name": "Bad", "members": []}),
                   encoding="utf-8", newline="\n")
    cols, problems = pc.load_all()
    assert [c.name for c in cols] == ["Good"]
    assert any(pr["kind"] == "bad_schema" and "Bad" in pr["detail"] for pr in problems)


def test_broken_json_is_refused_named_and_left_on_disk():
    pc.save(_c("Good", ["Al"]))
    bad = pc.collections_dir() / "Bad.json"
    bad.write_text("{not json", encoding="utf-8", newline="\n")
    cols, problems = pc.load_all()
    assert [c.name for c in cols] == ["Good"]
    assert any(pr["kind"] == "bad_schema" for pr in problems)
    assert bad.read_text(encoding="utf-8") == "{not json"   # user content, untouched


def test_one_nesting_level_only():
    pc.save(_c("Parent", []))
    pc.save(_c("Child", [], parent="Parent"))
    with pytest.raises(ValueError, match="one level"):
        pc.save(_c("Grandchild", [], parent="Child"))


def test_effective_members_are_own_then_children_in_order():
    pc.save(_c("Parent", ["Al", "Si"]))
    pc.save(_c("Fe-rich", ["Al13Fe4"], parent="Parent"))
    pc.save(_c("Mg-rich", ["Mg2Si"], parent="Parent"))
    assert [m.key for m in pc.effective_members("Parent")] == \
        ["Al", "Si", "Al13Fe4", "Mg2Si"]


def test_renaming_a_parent_follows_into_children_and_into_the_active_state():
    # REVIEW FOCUS 5.
    pc.save(_c("Parent", []))
    pc.save(_c("Child", [], parent="Parent"))
    pc.save_state({"schema": 1, "active": "Parent", "hidden": ["Parent"]})
    pc.rename("Parent", "Umbenannt")
    cols = {c.name: c for c in pc.load_all()[0]}
    assert cols["Child"].parent == "Umbenannt"
    state = pc.load_state()
    assert state["active"] == "Umbenannt"
    assert state["hidden"] == ["Umbenannt"]


def test_deleting_a_parent_promotes_children_and_clears_the_active_state():
    # Whole-branch review (2026-09-26), "tests that do not bite": this test
    # used to assert `cols["Child"].parent is None` from `pc.load_all()[0]`
    # ALONE. `load_all()` itself repairs an orphaned `parent` reference IN
    # MEMORY on every call (the loop right above `load_all()` in this same
    # module, appending an `orphan_parent` problem and setting `c.parent =
    # None` on the object it returns) -- disabling delete()'s OWN promotion
    # loop (the one that writes the promotion to Child.json) left this test
    # green anyway, because load_all()'s own repair was masking its absence.
    # A test that only reads through load_all() cannot tell "delete()
    # promoted and persisted this" from "delete() did nothing and load_all()
    # is quietly patching it up on every read forever after".
    pc.save(_c("Parent", ["Al"]))
    pc.save(_c("Child", ["Si"], parent="Parent"))
    pc.save_state({"schema": 1, "active": "Parent", "hidden": []})
    pc.delete("Parent")

    # (1) What's actually on disk: Child.json's own `parent` field, read
    # directly, bypassing load_all()'s repair-on-read entirely.
    child_path = pc.collections_dir() / "Child.json"
    on_disk = json.loads(child_path.read_text(encoding="utf-8"))
    assert on_disk["parent"] is None

    # (2) The absence of an orphan_parent problem on a fresh load — if
    # delete() had NOT persisted the promotion, every subsequent load_all()
    # call would keep re-detecting "Child" as orphaned (its disk file still
    # names the now-deleted "Parent") and keep reporting it.
    cols, problems = pc.load_all()
    assert not [pr for pr in problems if pr["kind"] == "orphan_parent"]

    by_name = {c.name: c for c in cols}
    assert set(by_name) == {"Child"}
    assert by_name["Child"].parent is None
    assert pc.load_state()["active"] is None


def test_the_working_set_is_exempt_from_exclusivity():
    pc.save(_c("Intermetallics", ["Al7FeCu2"]))
    pc.save(_c("Arbeitsauswahl", ["Al7FeCu2"], exclusive=False))
    problems = pc.load_all()[1]
    assert not [pr for pr in problems if pr["kind"] == "duplicate_key"]
    # Widened: the two collections above were built directly with `save()`,
    # which never exercises `assign()`'s own stripping logic at all -- that
    # gap is exactly why the CRITICAL bug below survived every earlier round.
    # Reached through the real code path, filing into the working set must
    # not touch "Intermetallics".
    pc.assign(["Al7FeCu2"], "Arbeitsauswahl")
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert [m.key for m in by_name["Intermetallics"].members] == ["Al7FeCu2"]


def test_assign_into_working_set_leaves_exclusive_home_intact():
    """The CRITICAL fix. A star must not evict a phase from its folder.

    ``assign()`` used to strip a key from every OTHER exclusive collection
    regardless of whether the TARGET was exclusive -- filing "Al" into the
    non-exclusive working set silently emptied "Matrix" of "Al" too. The
    working set is designed to cut ACROSS the partition (folders, one home
    each, plus one collection that deliberately ignores that rule for
    itself) -- not to replace a phase's home when it is added.
    """
    pc.save(_c("Matrix", ["Al"]))
    pc.save(_c("Arbeitsauswahl", [], exclusive=False))
    pc.assign(["Al"], "Arbeitsauswahl")
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert [m.key for m in by_name["Matrix"].members] == ["Al"]           # untouched
    assert [m.key for m in by_name["Arbeitsauswahl"].members] == ["Al"]   # added


def test_assign_into_exclusive_collection_leaves_the_working_set_intact():
    """The direction that already worked. Pinned so it cannot regress while
    fixing the direction above (`test_assign_into_working_set_leaves_
    exclusive_home_intact`) -- filing into an EXCLUSIVE collection must
    still leave the working set's own copy alone, since the working set
    legitimately holds a key that also lives in an exclusive collection."""
    pc.save(_c("Matrix", []))
    pc.save(_c("Arbeitsauswahl", ["Al"], exclusive=False))
    pc.assign(["Al"], "Matrix")
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert [m.key for m in by_name["Matrix"].members] == ["Al"]
    assert [m.key for m in by_name["Arbeitsauswahl"].members] == ["Al"]   # untouched


def test_assign_into_exclusive_collection_still_strips_other_exclusive_ones():
    """A phase still has exactly one EXCLUSIVE home -- unchanged behaviour
    (also covered by `test_assign_moves_a_key_out_of_its_previous_collection`
    above), named explicitly here so the asymmetry with the two working-set
    tests above is legible from the three test names alone."""
    pc.save(_c("Matrix", ["Al"]))
    pc.save(_c("Intermetallics", []))
    pc.assign(["Al"], "Intermetallics")
    by_name = {c.name: c for c in pc.load_all()[0]}
    assert [m.key for m in by_name["Matrix"].members] == []               # stripped
    assert [m.key for m in by_name["Intermetallics"].members] == ["Al"]


def test_delete_removes_the_collection_file_and_nothing_else():
    # An earlier version of this test put a .cif into a directory no code under
    # test knows about and asserted it survived; it could not have failed. What
    # can be checked is that delete() takes exactly one file out of the
    # collections directory and leaves the rest alone.
    pc.save(_c("A", ["Al"]))
    pc.save(_c("B", ["Si"]))
    pc.save_state({"schema": 1, "active": "B", "hidden": []})
    before = {f.name for f in pc.collections_dir().iterdir()}
    pc.delete("A")
    after = {f.name for f in pc.collections_dir().iterdir()}
    assert before - after == {"A.json"}
    assert "_local.json" in after and "B.json" in after


def test_schema_constant_is_written_into_every_file():
    pc.save(_c("A", ["Al"]))
    d = json.loads((pc.collections_dir() / "A.json").read_text(encoding="utf-8"))
    assert d["schema"] == COLLECTION_SCHEMA


class _Entry:
    """Stand-in for crystal_hint_local_library.LocalEntry — only the fields
    annotate() reads. Built by hand rather than mocked so the test states the
    contract it depends on."""
    def __init__(self, key, cif=None, xtal=None, sht=None,
                 formula="", space_group="", display_label=""):
        self.key, self.formula, self.space_group = key, formula, space_group
        self.cif_path, self.xtal_path, self.sht_path = cif, xtal, sht
        self.display_label = display_label
        self.elements = ()


def _fake_index(monkeypatch, entries):
    import backend.api.services.crystal_hint_local_library as lib
    monkeypatch.setattr(lib, "get_index", lambda: {e.key: e for e in entries})


def test_annotate_reports_what_each_member_can_do(monkeypatch, tmp_path):
    _fake_index(monkeypatch, [
        _Entry("Al", cif=tmp_path / "Al.cif", sht=tmp_path / "Al.sht",
               formula="Al", space_group="Fm-3m", display_label="Al - cF4 (Fm-3m)"),
        _Entry("Mg2Si", cif=tmp_path / "Mg2Si.cif", formula="Mg2Si"),
    ])
    out = pc.annotate([PhaseMember(key="Al"), PhaseMember(key="Mg2Si")])
    assert [o["key"] for o in out] == ["Al", "Mg2Si"]
    assert out[0]["present"] and out[0]["has_sht"] and out[0]["has_cif"]
    assert out[0]["label"] == "Al - cF4 (Fm-3m)"
    assert out[1]["present"] and not out[1]["has_sht"]


def test_a_member_the_library_does_not_have_is_kept_and_marked(monkeypatch):
    # REVIEW FOCUS 1, single member. Never dropped: a collection that quietly
    # shrinks while a drive is unmounted is worse than one that says so.
    _fake_index(monkeypatch, [])
    out = pc.annotate([PhaseMember(key="Al7FeCu2", formula="Al7Cu2Fe")])
    assert len(out) == 1
    assert out[0]["present"] is False
    assert out[0]["has_cif"] is False and out[0]["has_sht"] is False
    assert out[0]["label"]        # falls back to the key, never empty


def test_a_collection_whose_members_have_all_vanished_reports_every_one(monkeypatch):
    # REVIEW FOCUS 1, whole collection. The caller must be able to tell
    # "nothing here" from "everything is missing".
    _fake_index(monkeypatch, [])
    pc.save(_c("Hartmetalle", ["WC", "TiC", "TaC"]))
    out = pc.annotate(pc.effective_members("Hartmetalle"))
    assert len(out) == 3
    assert all(o["present"] is False for o in out)


def test_label_is_recomputed_from_the_library_not_read_from_the_file(monkeypatch, tmp_path):
    # The stored fingerprint must never become a second truth on screen.
    _fake_index(monkeypatch, [
        _Entry("Al", cif=tmp_path / "Al.cif", formula="Al",
               display_label="Al - cF4 (Fm-3m)"),
    ])
    out = pc.annotate([PhaseMember(key="Al", formula="STALE", space_group="STALE")])
    assert "STALE" not in out[0]["label"]


def test_unassigned_lists_library_phases_in_no_collection(monkeypatch, tmp_path):
    _fake_index(monkeypatch, [
        _Entry("Al", cif=tmp_path / "Al.cif"),
        _Entry("Si", cif=tmp_path / "Si.cif"),
        _Entry("Ni", cif=tmp_path / "Ni.cif"),
    ])
    pc.save(_c("Matrix", ["Al", "Si"]))
    assert pc.unassigned_keys() == ["Ni"]


def test_repair_is_proposed_never_applied(monkeypatch, tmp_path):
    # REVIEW FOCUS, rename case. Al2Cu and its .P1-backup twin share formula
    # and space group, so the app cannot know which was meant.
    _fake_index(monkeypatch, [
        _Entry("Al2Cu_renamed", cif=tmp_path / "x.cif",
               formula="Al2Cu", space_group="I4/mcm"),
    ])
    pc.save(PhaseCollection(name="A", members=[
        PhaseMember(key="Al2Cu", formula="Al2Cu", space_group="I4/mcm")]))
    proposals = pc.propose_repairs()
    assert proposals == [{"collection": "A", "missing_key": "Al2Cu",
                          "candidates": ["Al2Cu_renamed"]}]
    # and nothing was written
    after = {c.name: c for c in pc.load_all()[0]}["A"]
    assert [m.key for m in after.members] == ["Al2Cu"]


def test_suggest_sorts_the_library_that_is_there_and_writes_nothing(monkeypatch, tmp_path):
    def E(key, formula):
        e = _Entry(key, cif=tmp_path / f"{key}.cif", formula=formula)
        return e
    _fake_index(monkeypatch, [
        E("Al", "Al"), E("Si", "Si"),
        E("Al13Fe4", "Al13Fe4"), E("Al7FeCu2", "Al7Cu2Fe"),
        E("Mg2Si", "Mg2Si"), E("MgZn2", "MgZn2"),
        E("WC", "WC"),
        E("sd_1802610", ""),
    ])
    out = {s["name"]: s["keys"] for s in pc.suggest()}
    assert set(out["Matrix and pure metals"]) == {"Al", "Si"}
    assert set(out["Intermetallics in Al"]) == {"Al13Fe4", "Al7FeCu2"}
    assert "Mg2Si" in out["Mg phases"] and "MgZn2" in out["Mg phases"]
    assert out["Carbides and nitrides"] == ["WC"]
    # A phase whose formula could not be parsed is left alone, not guessed at.
    assert not any("sd_1802610" in keys for keys in out.values())
    assert pc.load_all()[0] == []          # nothing written


def test_suggest_never_proposes_a_phase_that_is_already_filed(monkeypatch, tmp_path):
    _fake_index(monkeypatch, [
        _Entry("Al", cif=tmp_path / "Al.cif", formula="Al"),
        _Entry("Si", cif=tmp_path / "Si.cif", formula="Si"),
    ])
    pc.save(_c("Mine", ["Al"]))
    out = {s["name"]: s["keys"] for s in pc.suggest()}
    assert out.get("Matrix and pure metals") == ["Si"]
