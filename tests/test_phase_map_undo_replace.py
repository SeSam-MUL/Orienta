"""One-level undo and the map-wide phase replacement.

The two corrections a seeded selection cannot make: taking back a mistake,
and fixing a classifier error that covers half the map. Lassoing 55 % of a
scan by hand is not a workflow.
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


def _store(grid=None):
    st = PhaseMapStore()
    g = np.zeros((4, 4), dtype=np.int32) if grid is None else grid
    st.set_classification(
        phase_grid=g, score_grid=np.full(g.shape, 0.9, dtype=np.float32),
        phase_entries=[AL, SI, FE], tolerance=15.0, min_score=0.3,
    )
    return st


# --- undo -------------------------------------------------------------------

def test_undo_takes_back_an_assignment():
    st = _store()
    st.assign_region(1, 2, 1, 2, 1)
    assert (st.get_state().phase_grid[1:3, 1:3] == 1).all()
    assert st.undo() is True
    assert (st.get_state().phase_grid == 0).all()


def test_undo_restores_the_locked_mask_too():
    """Restoring the grid alone would leave the map claiming pixels are
    hand-set that no longer are."""
    st = _store()
    st.assign_region(0, 0, 0, 0, 1)
    assert st.get_state().locked_mask is not None
    st.undo()
    lm = st.get_state().locked_mask
    assert lm is None or not lm.any()


def test_undo_of_an_undo_is_a_redo():
    st = _store()
    st.assign_region(1, 1, 1, 1, 2)
    st.undo()
    assert st.get_state().phase_grid[1, 1] == 0
    st.undo()
    assert st.get_state().phase_grid[1, 1] == 2


def test_undo_with_nothing_to_undo_says_so():
    st = PhaseMapStore()
    assert st.undo() is False


def test_undo_takes_back_a_reclassify():
    st = _store()
    st.assign_region(0, 3, 0, 3, 1)          # whole map -> Si
    st.set_classification(
        phase_grid=np.full((4, 4), 2, dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=[AL, SI, FE], tolerance=15.0, min_score=0.3,
        preserve_locked=False,
    )
    assert (st.get_state().phase_grid == 2).all()
    st.undo()
    assert (st.get_state().phase_grid == 1).all()


def test_a_rejected_write_does_not_consume_the_undo_slot():
    """The one slot must hold the last thing that actually happened."""
    st = _store()
    st.assign_region(1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        st.assign_region(0, 0, 0, 0, 99)     # out of range
    st.undo()
    assert (st.get_state().phase_grid == 0).all()


def test_undo_label_names_the_last_change():
    st = _store()
    # The FIRST classification has nothing before it to go back to.
    assert st.undo_label is None
    st.assign_region(0, 0, 0, 0, 1)
    assert st.undo_label == "assign region"


# --- replace phase ----------------------------------------------------------

def test_replace_repoints_every_pixel_of_one_phase():
    g = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 1, 1],
                  [2, 2, 1, 1]], dtype=np.int32)
    st = _store(g)
    n = st.replace_phase(1, 0)
    assert n == 8
    grid = st.get_state().phase_grid
    assert not (grid == 1).any()
    assert (grid == 0).sum() == 12
    assert (grid == 2).sum() == 4


def test_replace_marks_the_changed_pixels_as_hand_set():
    g = np.array([[0, 1], [1, 0]], dtype=np.int32)
    st = _store(g)
    st.replace_phase(1, 0)
    lm = st.get_state().locked_mask
    assert lm[0, 1] and lm[1, 0]
    assert not lm[0, 0] and not lm[1, 1]


def test_replace_can_target_and_source_unclassified():
    g = np.array([[-1, -1], [0, 0]], dtype=np.int32)
    st = _store(g)
    assert st.replace_phase(-1, 2) == 2
    assert (st.get_state().phase_grid[0] == 2).all()
    assert st.replace_phase(2, -1) == 2
    assert (st.get_state().phase_grid[0] == -1).all()


def test_replace_is_undoable():
    g = np.array([[0, 1], [1, 0]], dtype=np.int32)
    st = _store(g)
    st.replace_phase(1, 2)
    assert (st.get_state().phase_grid == np.array([[0, 2], [2, 0]])).all()
    st.undo()
    assert (st.get_state().phase_grid == g).all()


def test_replace_with_nothing_to_change_is_a_no_op():
    st = _store()
    st.assign_region(0, 0, 0, 0, 1)
    assert st.replace_phase(2, 1) == 0          # no pixel holds phase 2
    assert st.undo_label == "assign region", "a no-op must not eat the undo slot"


def test_replace_onto_itself_is_a_no_op():
    st = _store()
    assert st.replace_phase(0, 0) == 0


def test_replace_rejects_an_unknown_phase():
    st = _store()
    with pytest.raises(ValueError):
        st.replace_phase(0, 99)
    with pytest.raises(ValueError):
        st.replace_phase(99, 0)


def test_replace_without_a_map_raises():
    with pytest.raises(RuntimeError):
        PhaseMapStore().replace_phase(0, 1)
