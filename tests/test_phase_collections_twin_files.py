"""Two collection files carrying one `name`.

Review HIGH 4 (2026-09-27). Reachable only through a folder that was COPIED,
MERGED or SHARED -- which is precisely the workflow identities exist for (§2.5.3,
"copied and mailed by hand"), so it is the case schema 2 has to survive rather
than an exotic one. `create` and `rename` both refuse casefold-duplicate names,
so the app cannot produce it inside one folder.

WHAT WAS PRE-EXISTING AND WHAT SCHEMA 2 ADDED -- measured by running the same
probes against `516bd3d1^` and against HEAD, not argued:

  pre-existing, identical on both sides
    * `assign(['NEW'], 'Matrix')` rewrote twinA's FILE with twinB's members, so
      twinA's own member was gone. `_path_of` returns the first casefold match in
      the folder, and every writer re-derived its path from the NAME.
    * `unassign_from` the same.
    * a `move` into a casefold twin stripped the phase, THEN raised -- the caller
      got an error stating nothing had happened.
    * `load_all` reported no problem for any of it.

  added by schema 2 (1cc75b9a)
    * two files ended up carrying the SAME `id` after such a write.
    * `_mint_id` could mint an id that already existed, because its `taken` set
      came from `_by_name()`, which is keyed by name and drops one of a pair.
      Measured: ids on disk `['Matrix', 'Matrix', 'zzz']`, and
      `resolve_ref('Matrix')` then answered the NEW EMPTY collection -- so a
      `parent`, `active` or `hidden` reference resolved to the wrong group.

The destructive write is therefore NOT a regression; what schema 2 did was make
references unreliable on top of it. Both are fixed here, and the tests say which
is which so nobody has to re-derive it.
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


@pytest.fixture(autouse=True)
def store(tmp_path):
    pc.set_collections_dir_for_test(tmp_path / "Collections")
    pc.collections_dir().mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        pc.set_collections_dir_for_test(None)


def _plant(stem: str, name: str, members, **extra) -> Path:
    """Write a collection file directly -- the hand copy this test is about."""
    doc = {"schema": 2, "name": name, "parent": None, "exclusive": True,
           "description": "", "created": "", "author": "", "updated": "",
           "members": [{"key": k} for k in members]}
    doc.update(extra)
    p = pc.collections_dir() / f"{stem}.json"
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8", newline="\n")
    return p


def _on_disk() -> dict:
    out = {}
    for p in sorted(pc.collections_dir().glob("*.json")):
        if p.name == "_local.json":
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        out[p.name] = (doc.get("id"), [m["key"] for m in doc.get("members", [])])
    return out


# --- the duplicate is REPORTED, not silently resolved ----------------------

def test_two_files_with_one_name_are_reported():
    """It was silent in both directions: `_by_name()` collapses them, so one
    collection vanished from the app entirely, and the writers overwrote whichever
    file came first. The app cannot know which was meant -- the same reason
    `bad_schema` is reported rather than repaired."""
    _plant("twinA", "Matrix", ["a1"], id="twinA")
    _plant("twinB", "Matrix", ["b1"], id="twinB")
    _cols, problems = pc.load_all()
    dupes = [p for p in problems if p["kind"] == "duplicate_name"]
    assert len(dupes) == 1
    assert dupes[0]["detail"] == "Matrix"
    assert dupes[0]["files"] == ["twinA.json", "twinB.json"]


def test_a_casefold_duplicate_is_reported_too():
    """`_path_of` matches casefold, so `Matrix` and `matrix` share one file as
    far as every lookup is concerned."""
    _plant("MatrixUpper", "Matrix", ["Al"], id="MatrixUpper")
    _plant("matrixlower", "matrix", ["Si"], id="matrixlower")
    problems = pc.load_all()[1]
    assert [p["kind"] for p in problems] == ["duplicate_name"]


def test_a_normal_folder_reports_nothing():
    """The guard must not cry on the ordinary case -- a permanent warning on
    correct filing is how a report gets ignored."""
    pc.save(pc.PhaseCollection(name="Al systems", members=[pc.PhaseMember(key="Al")]))
    pc.save(pc.PhaseCollection(name="Cu systems", members=[pc.PhaseMember(key="Al2Cu")]))
    assert pc.load_all()[1] == []


# --- PRE-EXISTING: a write must not land on another file -------------------

def test_assign_does_not_overwrite_the_other_twins_file():
    """The pre-existing defect, measured identically before schema 2: writing to
    the collection named "Matrix" rewrote the OTHER file, and its own member was
    gone. Nothing announced it."""
    _plant("twinA", "Matrix", ["only-in-twinA"], id="twinA")
    _plant("twinB", "Matrix", ["only-in-twinB"], id="twinB")

    pc.assign(["NEW"], "Matrix")

    disk = _on_disk()
    # Whichever twin the name resolved to got the new member; the other is
    # untouched, and no member was lost from either.
    assert "only-in-twinA" in disk["twinA.json"][1], "twinA's member survives"
    assert "only-in-twinB" in disk["twinB.json"][1], "twinB's member survives"
    assert sum("NEW" in m for _i, m in disk.values()) == 1, "written once"
    assert disk["twinA.json"][0] == "twinA" and disk["twinB.json"][0] == "twinB", (
        "and neither file took the other's identity")


def test_unassign_from_does_not_overwrite_the_other_twins_file():
    _plant("twinA", "Matrix", ["a1"], id="twinA")
    _plant("twinB", "Matrix", ["b1"], id="twinB")

    pc.unassign_from("Matrix", ["a1"])

    disk = _on_disk()
    assert disk["twinA.json"][0] == "twinA"
    assert disk["twinB.json"][0] == "twinB"
    assert disk["twinB.json"][1] == ["b1"], "the untargeted twin is unchanged"


def test_a_move_that_fails_moves_nothing():
    """The pre-existing partial write. `save()` refuses a casefold duplicate, and
    the strip loop used to run BEFORE it: the phase left its old group and then
    the caller was told nothing had happened. A failing move must move nothing."""
    _plant("MatrixUpper", "Matrix", ["Al", "Si"], id="MatrixUpper")
    _plant("matrixlower", "matrix", [], id="matrixlower")
    before = _on_disk()

    with pytest.raises(ValueError):
        pc.assign(["Al"], "matrix", move=True)

    assert _on_disk() == before, "not one byte may have moved"


# --- ADDED BY SCHEMA 2: identities must stay unique ------------------------

def test_a_minted_identity_never_collides_with_one_on_disk():
    """What schema 2 added. `_mint_id` took its `taken` set from `_by_name()`,
    which is keyed by NAME and therefore hides one of a pair -- so it minted an id
    that already existed, and `resolve_ref` then answered the wrong collection.
    A duplicate id is worse than a duplicate name: a name is a label the user
    reads, an id is what `parent`, `active` and `hidden` point at."""
    _plant("twinA", "M", [], id="Matrix")
    _plant("twinB", "M", [], id="zzz")

    pc.save(pc.PhaseCollection(name="Matrix"))

    ids = [i for i, _m in _on_disk().values()]
    assert len(ids) == len(set(ids)), f"ids must stay unique: {ids}"
    assert "Matrix-2" in ids, "the newcomer takes the suffixed id"


def test_the_new_collection_does_not_steal_the_existing_identity():
    """NARROWED 2026-09-27: what this test is FOR is that the pre-existing holder
    keeps its members. What it used to also assert -- that `resolve_ref("Matrix")`
    answers the holder of the ID -- is reversed: names now win over ids (f7's
    review found that ids-win deleted the wrong group), and here the newcomer is
    literally NAMED "Matrix", so it is the one a user asking for "Matrix" means.

    The no-data-loss claim is unchanged and is the half that matters.
    """
    _plant("twinA", "M", ["kept"], id="Matrix")
    _plant("twinB", "M", [], id="zzz")
    pc.save(pc.PhaseCollection(name="Matrix"))

    by_id = {c.id: c for c in pc.load_all()[0]}
    assert "Matrix" in by_id, "the existing identity is still held by its owner"
    assert [m.key for m in by_id["Matrix"].members] == ["kept"], (
        "and nothing took its members")
    # The newcomer could not take that id, which is the other half of the fix.
    assert by_id["Matrix"].name == "M"
    assert {c.id for c in pc.load_all()[0]} == {"Matrix", "zzz", "Matrix-2"}

    hit = pc.resolve_ref("Matrix", pc.load_all()[0])
    assert hit is not None and hit.name == "Matrix" and hit.id == "Matrix-2", (
        "asking for 'Matrix' means the group CALLED Matrix")


# --- the mechanism behind all of it ----------------------------------------

def test_a_loaded_collection_remembers_the_file_it_came_from():
    """`source_path` is what makes a write unable to alias. Not serialised: it is
    a fact about this process, not part of the format."""
    p = _plant("oddly-named-file", "Matrix", ["Al"], id="m1")
    (c,) = pc.load_all()[0]
    assert c.source_path == p
    assert "source_path" not in c.to_dict()

    pc.assign(["Si"], "Matrix")
    assert json.loads(p.read_text(encoding="utf-8"))["members"] == [
        {"key": "Al"}, {"key": "Si"}], "written back to its own file"
    assert list(pc.collections_dir().glob("*.json")) == [p], "and no second file"
