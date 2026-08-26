"""A hand-built segmentation has to survive a reload.

The user asked for "full manual customization of region definitions and phase
assignments should the user prefer that instead". Definitions that live only
in the browser tab make that unusable for real work: a refresh, a backend
restart, or coming back tomorrow would lose the windows you spent an hour
tuning, and the map would still be there without anything explaining how it
was made.

So the definitions and the element weights ride in the sidecar alongside the
map. They also make the map self-describing, which is the second half of the
same problem: an editor showing nothing next to a map it cannot explain is
worse than no editor.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.phase_map_store import PhaseMapStore

DEFS = [{
    "name": "pure Si",
    "phase_key": "Si.cif",
    "elements": [{"element": "Si", "min_at_pct": 20.0, "max_at_pct": None},
                 {"element": "Fe", "min_at_pct": None, "max_at_pct": 1.0}],
    "ratios": [],
    "enrichment": [],
}]
WEIGHTS = {"Fe": 3.0}


def _entry(name):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="",
        space_group_number=None, crystal_system="", composition={"Al": 100.0},
        elements=["Al"],
    )


def _classify(st, **kw):
    st.set_classification(
        phase_grid=np.zeros((4, 4), dtype=np.int32),
        score_grid=np.zeros((4, 4), dtype=np.float32),
        phase_entries=[_entry("Al.cif")], tolerance=15.0, min_score=0.3, **kw)


def test_the_map_remembers_how_it_was_made():
    st = PhaseMapStore()
    _classify(st, region_defs=DEFS, element_weights=WEIGHTS)
    state = st.get_state()
    assert state.region_defs == DEFS
    assert state.element_weights == WEIGHTS


def test_a_plain_run_carries_none_of_it():
    """The automatic path must not start claiming it was hand-built."""
    st = PhaseMapStore()
    _classify(st)
    state = st.get_state()
    assert state.region_defs == []
    assert state.element_weights == {}


def test_definitions_and_weights_survive_a_round_trip(tmp_path):
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"not a real file, only a key for the sidecar")
    st = PhaseMapStore()
    _classify(st, region_defs=DEFS, element_weights=WEIGHTS)
    assert st.save_to_disk(str(f)) is not None

    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True
    state = fresh.get_state()
    assert state.region_defs == DEFS
    assert state.element_weights == WEIGHTS


def test_a_sidecar_written_before_this_feature_still_loads(tmp_path):
    """Absence must read as "automatic", not as a corrupt sidecar."""
    f = tmp_path / "scan.h5oina"
    f.write_bytes(b"key")
    st = PhaseMapStore()
    _classify(st)                       # no defs, no weights
    st.save_to_disk(str(f))

    fresh = PhaseMapStore()
    assert fresh.load_from_disk(str(f)) is True
    assert fresh.get_state().region_defs == []
    assert fresh.get_state().element_weights == {}


def test_a_later_plain_reclassify_drops_the_provenance(tmp_path):
    """Re-running without definitions means the map is no longer built from
    them, and it must stop saying it was."""
    st = PhaseMapStore()
    _classify(st, region_defs=DEFS, element_weights=WEIGHTS)
    _classify(st)
    assert st.get_state().region_defs == []
