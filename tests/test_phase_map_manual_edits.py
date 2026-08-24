"""Hand-assigned pixels must survive a re-classify.

`locked_mask` has been written on every manual edit since M4 and read by
nothing — `set_classification` hardcoded `locked_mask=None`, so every
re-classify silently destroyed the corrections the mask exists to protect.
"""
import sys
from pathlib import Path

import numpy as np

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


def _store(entries=(AL, SI), fill=0):
    st = PhaseMapStore()
    st.set_classification(
        phase_grid=np.full((4, 4), fill, dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=list(entries), tolerance=15.0, min_score=0.3,
    )
    return st


def test_a_hand_assigned_pixel_survives_reclassify():
    st = _store()
    st.assign_region(1, 1, 1, 1, 1)                 # hand-set one pixel to Si
    assert st.get_state().phase_grid[1, 1] == 1

    st.set_classification(                          # the classifier says Al everywhere
        phase_grid=np.zeros((4, 4), dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=[AL, SI], tolerance=15.0, min_score=0.3,
    )
    s = st.get_state()
    assert s.phase_grid[1, 1] == 1, "the hand-assigned pixel was overwritten"
    assert s.phase_grid[0, 0] == 0, "the rest must take the new classification"
    assert s.locked_mask is not None and s.locked_mask[1, 1]


def test_start_over_discards_them_on_request():
    st = _store()
    st.assign_region(1, 1, 1, 1, 1)
    st.set_classification(
        phase_grid=np.zeros((4, 4), dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=[AL, SI], tolerance=15.0, min_score=0.3,
        preserve_locked=False,
    )
    s = st.get_state()
    assert s.phase_grid[1, 1] == 0
    assert s.locked_mask is None


def test_pixels_are_carried_by_phase_identity_not_by_index():
    """Phase ids are POSITIONS in the candidate list. A re-run over a
    different list re-numbers them, so carrying the raw index would silently
    repoint a hand-assigned pixel at an unrelated phase."""
    st = _store(entries=(AL, SI))
    st.assign_region(2, 2, 2, 2, 1)                 # Si.cif, index 1
    assert st.get_state().phase_entries[1].cif_filename == "Si.cif"

    st.set_classification(                          # Si is now index 2
        phase_grid=np.zeros((4, 4), dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=[AL, FE, SI], tolerance=15.0, min_score=0.3,
    )
    s = st.get_state()
    assert s.phase_entries[s.phase_grid[2, 2]].cif_filename == "Si.cif"


def test_a_phase_that_left_the_candidate_list_is_dropped_not_repointed():
    st = _store(entries=(AL, SI))
    st.assign_region(3, 3, 3, 3, 1)                 # Si.cif
    st.set_classification(                          # Si is gone from the library
        phase_grid=np.zeros((4, 4), dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=[AL, FE], tolerance=15.0, min_score=0.3,
    )
    s = st.get_state()
    assert s.phase_entries[s.phase_grid[3, 3]].cif_filename != "Fe.cif", (
        "the pixel was repointed at whatever now sits at index 1"
    )
    assert s.phase_grid[3, 3] == 0


def test_hand_marked_unclassified_is_also_a_decision():
    st = _store()
    st.assign_region(0, 0, 0, 0, -1)                # user says "not a phase"
    st.set_classification(
        phase_grid=np.zeros((4, 4), dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=[AL, SI], tolerance=15.0, min_score=0.3,
    )
    assert st.get_state().phase_grid[0, 0] == -1


def test_untouched_map_reclassifies_completely():
    st = _store()
    st.set_classification(
        phase_grid=np.ones((4, 4), dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=[AL, SI], tolerance=15.0, min_score=0.3,
    )
    assert (st.get_state().phase_grid == 1).all()
