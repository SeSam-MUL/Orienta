"""Structures: group the map by chemistry first, name it afterwards.

The user's framing, and the reason this layer exists: "vorclustern zu einer
Phase einfach nur deswegen weil die Pixel zusammengehoeren und vom
Elementverhaeltnis gleich sind. dann kann ich fuer jede dieser Phasen eine cif
zuordnen." Two consequences these tests pin:

* ``phase_grid`` stays authoritative. It has five consumers outside the store
  (phase suggestion, summary, grid guard, "Send to Indexing", the consensus
  xmap), so a structure assignment must WRITE INTO it, not replace it.
* Hand edits win over structures. A boundary move must never silently undo a
  decision the user made explicitly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.phase_map_store import PhaseMapStore


def _entry(name, comp):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="",
        space_group_number=None, crystal_system="", composition=comp,
        elements=sorted(comp),
    )


AL = _entry("Al.cif", {"Al": 100.0})
SI = _entry("Si.cif", {"Si": 100.0})
FE = _entry("Fe.cif", {"Fe": 100.0})


def _store(structure_grid=None, structure_phase=None, features=None):
    """A 4x4 map whose left half is structure 0 and right half structure 1."""
    st = PhaseMapStore()
    if structure_grid is None:
        structure_grid = np.zeros((4, 4), dtype=np.int32)
        structure_grid[:, 2:] = 1
    if structure_phase is None:
        structure_phase = [-1] * (int(structure_grid.max()) + 1)
    if features is None:
        # One feature: the column index, so a split cuts left/right predictably.
        features = np.repeat(
            np.arange(structure_grid.shape[1], dtype=float),
            1).reshape(1, -1).repeat(structure_grid.shape[0], axis=0).reshape(-1, 1)
    st.set_classification(
        phase_grid=np.full(structure_grid.shape, -1, dtype=np.int32),
        score_grid=np.zeros(structure_grid.shape, dtype=np.float32),
        phase_entries=[AL, SI, FE], tolerance=15.0, min_score=0.3,
        structure_grid=structure_grid,
        structure_phase=structure_phase,
        structure_features=features,
    )
    return st


# --- the model ---------------------------------------------------------------

def test_structures_live_beside_phase_grid_not_instead_of_it():
    st = _store()
    state = st.get_state()
    assert state.phase_grid.shape == state.structure_grid.shape
    assert state.n_structures == 2


def test_naming_a_structure_paints_every_pixel_of_it():
    st = _store()
    n = st.assign_structure(1, 0)          # right half -> Al
    assert n == 8
    grid = st.get_state().phase_grid
    assert (grid[:, 2:] == 0).all()
    assert (grid[:, :2] == -1).all(), "the other structure must not move"


def test_naming_does_not_lock_the_pixels():
    """The structure IS the record of the decision, so a later boundary
    change keeps following it. Locking would freeze the pixels in place."""
    st = _store()
    st.assign_structure(0, 1)
    lm = st.get_state().locked_mask
    assert lm is None or not lm.any()


def test_naming_an_unknown_structure_raises():
    st = _store()
    with pytest.raises(ValueError):
        st.assign_structure(9, 0)


def test_naming_with_an_unknown_phase_raises():
    st = _store()
    with pytest.raises(ValueError):
        st.assign_structure(0, 99)


def test_naming_without_structures_says_so_rather_than_guessing():
    st = PhaseMapStore()
    st.set_classification(
        phase_grid=np.zeros((2, 2), dtype=np.int32),
        score_grid=np.zeros((2, 2), dtype=np.float32),
        phase_entries=[AL], tolerance=15.0, min_score=0.3,
    )
    with pytest.raises(RuntimeError):
        st.assign_structure(0, 0)


# --- merge -------------------------------------------------------------------

def test_merge_folds_one_structure_into_another():
    st = _store()
    st.assign_structure(0, 0)
    moved = st.merge_structures(0, 1)
    assert moved == 8
    state = st.get_state()
    assert state.n_structures == 1
    assert (state.structure_grid == 0).all()
    assert (state.phase_grid == 0).all(), "the merged pixels take the kept name"


def test_merge_keeps_the_id_list_dense():
    """Ids are positions; a hole would misname every structure after it."""
    grid = np.array([[0, 1], [2, 2]], dtype=np.int32)
    st = _store(grid, [-1, 0, 1])
    st.merge_structures(0, 1)
    state = st.get_state()
    assert state.n_structures == 2
    assert set(np.unique(state.structure_grid)) == {0, 1}
    assert state.structure_phase == [-1, 1], "structure 2's name must follow it down"


def test_merge_onto_itself_is_a_no_op():
    st = _store()
    assert st.merge_structures(0, 0) == 0
    assert st.get_state().n_structures == 2


def test_merge_is_undoable():
    st = _store()
    before = st.get_state().structure_grid.copy()
    st.merge_structures(0, 1)
    st.undo()
    assert (st.get_state().structure_grid == before).all()
    assert st.get_state().n_structures == 2


def test_merge_rejects_an_unknown_structure():
    st = _store()
    with pytest.raises(ValueError):
        st.merge_structures(0, 7)


# --- split -------------------------------------------------------------------

def test_split_cuts_one_structure_and_leaves_the_rest_alone():
    """Raising the global cluster count would re-cut every structure; that is
    why splitting is local."""
    st = _store()
    other_before = (st.get_state().structure_grid == 1).copy()
    n = st.split_structure(0, 2)
    assert n == 2
    state = st.get_state()
    assert state.n_structures == 3
    assert ((state.structure_grid == 1) == other_before).all()


def test_split_parts_inherit_the_original_name():
    st = _store()
    st.assign_structure(0, 2)
    st.split_structure(0, 2)
    state = st.get_state()
    assert state.structure_phase[0] == 2
    assert state.structure_phase[-1] == 2


def test_split_of_a_structure_smaller_than_the_parts_is_a_no_op():
    grid = np.array([[0, 1], [1, 1]], dtype=np.int32)
    st = _store(grid, [-1, -1])
    assert st.split_structure(0, 3) == 0


def test_split_without_the_composition_fails_loud():
    """After a backend restart the sidecar has the structures but not the
    composition. Guessing would split on the wrong data."""
    st = _store()
    st._structure_features = None
    with pytest.raises(RuntimeError, match="re-run the classification"):
        st.split_structure(0, 2)


# --- grow / shrink -----------------------------------------------------------

def test_grow_pushes_the_boundary_out():
    st = _store()
    before = int((st.get_state().structure_grid == 0).sum())
    st.grow_structure(0, 1)
    assert int((st.get_state().structure_grid == 0).sum()) > before


def test_shrink_hands_the_pixels_to_a_neighbour_not_to_nothing():
    """Vacated pixels must never become unclassified - that would punch
    holes in the map."""
    st = _store()
    st.grow_structure(0, -1)
    grid = st.get_state().structure_grid
    assert (grid >= 0).all(), "no pixel may be left without a structure"


def test_grow_by_zero_changes_nothing():
    st = _store()
    assert st.grow_structure(0, 0) == 0


def test_grow_is_undoable():
    st = _store()
    before = st.get_state().structure_grid.copy()
    st.grow_structure(0, 1)
    st.undo()
    assert (st.get_state().structure_grid == before).all()


# --- hand edits win ----------------------------------------------------------

def test_a_boundary_move_does_not_undo_a_hand_edit():
    st = _store()
    st.assign_structure(0, 0)
    st.assign_structure(1, 1)
    # The user overrules one pixel by hand.
    hand = np.zeros((4, 4), dtype=bool)
    hand[0, 0] = True
    st.assign_mask(hand, 2)
    assert st.get_state().phase_grid[0, 0] == 2

    st.grow_structure(1, 1)
    assert st.get_state().phase_grid[0, 0] == 2, (
        "a boundary move must not silently revert an explicit decision")


def test_renaming_a_structure_does_not_undo_a_hand_edit():
    st = _store()
    hand = np.zeros((4, 4), dtype=bool)
    hand[3, 3] = True
    st.assign_mask(hand, 2)
    st.assign_structure(1, 0)
    assert st.get_state().phase_grid[3, 3] == 2


# --- snap --------------------------------------------------------------------

def test_snap_keeps_every_structure_alive():
    """A watershed seeded by interiors can move a boundary but must never
    let a structure vanish or swap identity."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[:, 4:] = 1
    feats = np.zeros((64, 1))
    feats[grid.ravel() == 1] = 1.0        # a clean chemistry edge at column 4
    st = _store(grid, [-1, -1], feats)
    st.snap_structure_edges(strength=1.0)
    out = st.get_state().structure_grid
    assert set(np.unique(out)) == {0, 1}


def test_snap_with_zero_strength_changes_nothing():
    st = _store()
    assert st.snap_structure_edges(strength=0.0) == 0


def test_snap_without_the_composition_fails_loud():
    st = _store()
    st._structure_features = None
    with pytest.raises(RuntimeError, match="re-run the classification"):
        st.snap_structure_edges(strength=1.0)


def test_snap_moves_a_boundary_onto_the_real_chemistry_edge():
    """The point of the tool: the label boundary starts in the wrong place
    and the chemistry says where it belongs."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[:, 6:] = 1                       # label edge at column 6 ...
    feats = np.zeros((64, 1))
    cols = np.tile(np.arange(8), 8)
    feats[cols >= 4] = 1.0                # ... chemistry edge at column 4
    st = _store(grid, [-1, -1], feats)
    st.snap_structure_edges(strength=2.0)
    out = st.get_state().structure_grid
    boundary = int(np.argmax(out[0] == 1))
    assert boundary < 6, f"boundary should move toward the chemistry edge, got {boundary}"
