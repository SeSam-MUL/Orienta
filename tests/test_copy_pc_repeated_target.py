"""A file can be the target of a PC copy more than once.

``copy-pc`` sets ``parent_name`` on the entry of the TARGET FILE (whose PC it took), and
``CalibrationStore.name_of_file`` skipped every entry with a ``parent_name`` because
deepcopies and crops (views of their parent's file, carrying its ``file_key``) must not be
taken for the entry of the file. So after A -> B, the file B was "not in the store": C -> B
(or the same copy again) answered "this file is not in the store (quick-load it first)".
The two meanings are now apart: ``is_view_of_parent`` marks a view, ``parent_name`` keeps
its family meaning for both.
"""
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import batch_v2
from backend.api.services.batch_manager import _canonical_file
from backend.api.services.calibration_store import CalibrationStore
from backend.api.services.crop_window import CropWindow

PCS = {"A": (0.51, 0.31, 0.81), "B": (0.50, 0.30, 0.80), "C": (0.55, 0.35, 0.85)}
PATHS = {k: f"/d/rep_{k}.up1" for k in PCS}


class _Det:
    shape = (60, 60)
    sample_tilt = 70.0
    tilt = 0.0
    azimuthal = 0.0

    def __init__(self, pc):
        self.pc = np.array(pc)


class _Sig:
    def __init__(self, pc):
        self.detector = _Det(pc)


@pytest.fixture
def world(monkeypatch):
    store = CalibrationStore()
    monkeypatch.setattr(batch_v2, "calibration_store", store)
    for name, pc in PCS.items():
        store.register(name, _Sig(pc), file_key=_canonical_file(PATHS[name]))
    app = FastAPI()
    app.include_router(batch_v2.router, prefix="/api/batch-v2")
    return TestClient(app), store


def _copy(client, source, target):
    return client.post("/api/batch-v2/copy-pc", json={
        "source_dataset": source, "target_datasets": [target],
        "source_file_path": PATHS[source], "target_file_paths": [PATHS[target]]})


def _ok(resp):
    assert resp.status_code == 200, resp.text
    (result,) = resp.json()["results"]
    assert result["success"] is True, result
    return result


def test_a_second_copy_to_the_same_file_takes_the_second_sources_pc(world):
    client, store = world
    _ok(_copy(client, "A", "B"))
    np.testing.assert_allclose(store.get_pc("B"), PCS["A"])
    _ok(_copy(client, "C", "B"))                       # B is a copy target now, and still B's file
    np.testing.assert_allclose(store.get_pc("B"), PCS["C"])
    assert store.get_entry("B").pc_source == "inherited"


def test_the_same_copy_can_be_repeated(world):
    client, store = world
    _ok(_copy(client, "A", "B"))
    _ok(_copy(client, "A", "B"))
    np.testing.assert_allclose(store.get_pc("B"), PCS["A"])


def test_the_origin_shown_after_a_copy_is_the_last_source(world):
    client, store = world
    _ok(_copy(client, "A", "B"))
    shown = client.get("/api/batch-v2/pc-status", params={"files": "B"}).json()[0]
    assert shown["pc_source"] == "inherited" and shown["parent_name"] == "A"
    _ok(_copy(client, "C", "B"))
    shown = client.get("/api/batch-v2/pc-status", params={"files": "B"}).json()[0]
    assert shown["pc_source"] == "inherited" and shown["parent_name"] == "C"
    np.testing.assert_allclose(shown["pc"], PCS["C"])


def test_a_copy_target_is_its_files_entry_but_a_view_of_it_is_not():
    store = CalibrationStore()
    store.register("A", _Sig(PCS["A"]), file_key="/d/a.up1")
    store.register("B", _Sig(PCS["B"]), file_key="/d/b.up1")
    store.get_entry("B").parent_name = "A"                     # what copy-pc does to the target
    store.register_derived("B_bg", "B")                        # views of B: B's key, B as parent
    store.register_cropped("B_crop", "B", CropWindow("b.h5oina", 2, 3, 4, 5, (10, 12)))
    assert store.get_entry("B_bg").is_view_of_parent and store.get_entry("B_crop").is_view_of_parent
    assert not store.get_entry("B").is_view_of_parent
    # the views come FIRST in the table: B's entry is still the answer
    store._entries = {k: store._entries[k] for k in ("B_bg", "B_crop", "A", "B")}
    assert store.name_of_file("/d/b.up1") == "B"
    assert store.name_of_file("/d/a.up1") == "A"


# ---------------------------------------------------------------- a copy back: A -> B, B -> A
#
# The copy back is not refused: refining B and copying the result back to A is the natural
# flow. But parent_name must never form a cycle: the family walks start at the refined file
# and, on a cycle, name a different "root" for each start, so a refine on A missed the copy
# of B and the other way round. The source of a copy therefore becomes the root of its
# family when the target is one of its ancestors: the family stays a tree.

def _add_views(store):
    """A deepcopy of each file, as the viewer makes them (the parent's PC, 'inherited')."""
    for name in list(store.get_all()):
        if name in PCS:
            store.register_derived(f"{name}_bg", name)


def _pc_of_all(store, expected, names):
    for name in names:
        np.testing.assert_allclose(store.get_pc(name), expected, err_msg=name)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("refined", ["A", "B"])
def test_after_a_copy_back_a_refine_on_either_file_reaches_every_copy(world, refined):
    from backend.api.routes.pcrefinement import _writeback_refined_pc

    client, store = world
    _add_views(store)
    _ok(_copy(client, "A", "B"))
    store.update_pc("B", [0.6, 0.4, 0.9], source="refined")
    _ok(_copy(client, "B", "A"))
    # a tree: B is the root, A took its PC, nothing is its own ancestor
    assert store.get_entry("B").parent_name is None
    assert store.get_entry("A").parent_name == "B"
    np.testing.assert_allclose(store.get_pc("A"), [0.6, 0.4, 0.9])
    c_before = store.get_pc("C").copy()
    written = {}
    new_pc = [0.57, 0.37, 0.87]
    _writeback_refined_pc(refined, new_pc, store=store,
                          refinement={"n_patterns": 3, "method": "NM"}, report=written)
    _pc_of_all(store, new_pc, ["A", "B", "A_bg", "B_bg"])
    assert store.get_entry(refined).pc_source == "refined"
    assert set(written) == {"A", "B", "A_bg", "B_bg"}
    np.testing.assert_allclose(store.get_pc("C"), c_before)          # nobody else is written


@pytest.mark.timeout(60)
def test_a_chain_closed_by_a_third_copy_is_still_a_tree(world):
    from backend.api.routes.pcrefinement import _writeback_refined_pc

    client, store = world
    _add_views(store)
    _ok(_copy(client, "A", "B"))
    _ok(_copy(client, "B", "C"))
    _ok(_copy(client, "C", "A"))               # A is an ancestor of C: C becomes the root
    assert store.get_entry("C").parent_name is None
    roots = {store.family_root(n) for n in ("A", "B", "C", "A_bg", "B_bg")}
    assert roots == {"C"}                       # (c) one root for every member
    new_pc = [0.58, 0.38, 0.88]
    _writeback_refined_pc("B", new_pc, store=store, refinement=None)
    _pc_of_all(store, new_pc, ["A", "B", "C", "A_bg", "B_bg"])


@pytest.mark.timeout(60)
def test_no_sequence_of_copies_makes_a_cycle(world):
    client, store = world
    _add_views(store)
    for source, target in (("A", "B"), ("B", "A"), ("A", "B"), ("B", "C"), ("C", "A"), ("A", "C")):
        _ok(_copy(client, source, target))
        for name in store.get_all():
            assert not store.is_ancestor(name, name), (source, target, name)
        # the root of every entry is a real root (on a cycle the walk ends on an entry
        # that still has a parent)
        for name in store.get_all():
            assert store.get_entry(store.family_root(name)).parent_name is None, (source, target, name)


# ------------------------------------------------------------------ a copy onto itself

def test_a_file_copied_onto_itself_is_skipped_and_the_other_targets_still_get_the_pc(world):
    client, store = world
    resp = client.post("/api/batch-v2/copy-pc", json={
        "source_dataset": "A", "target_datasets": ["A", "B"],
        "source_file_path": PATHS["A"], "target_file_paths": [PATHS["A"], PATHS["B"]]})
    assert resp.status_code == 200, resp.text
    own, other = resp.json()["results"]
    assert own["skipped"] is True and own["success"] is True        # not a failure for the page
    assert other["success"] is True and "skipped" not in other
    assert store.get_entry("A").parent_name is None                  # not its own parent
    assert store.get_entry("A").pc_source == "header"                # nothing was written to it
    np.testing.assert_allclose(store.get_pc("B"), PCS["A"])
    assert store.get_entry("B").parent_name == "A"
