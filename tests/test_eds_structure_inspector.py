"""What the inspector reports about one structure.

These are measurements a user acts on — merge or not, this phase or that one —
so the failure that matters is not a crash but a number that looks right and
is not. The first version divided the structure's at% mean by a background
held as a RENORMALISED FRACTION and reported "Al 49.68x enriched" on a map
whose background is aluminium. That is what these pin.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.routes.eds import _structure_detail
from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.phase_map_store import PhaseMapState


def _entry(name, comp):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="",
        space_group_number=None, crystal_system="", composition=comp,
        elements=sorted(comp),
    )


def _state(structure_grid, structure_phase=None, entries=None):
    shape = structure_grid.shape
    return PhaseMapState(
        phase_grid=np.full(shape, -1, dtype=np.int32),
        score_grid=np.zeros(shape, dtype=np.float32),
        phase_entries=entries or [_entry("Al.cif", {"Al": 100.0}),
                                  _entry("Si.cif", {"Si": 100.0}),
                                  _entry("AlSi.cif", {"Al": 50.0, "Si": 50.0})],
        n_rows=shape[0], n_cols=shape[1], tolerance=15.0, min_score=0.3,
        structure_grid=structure_grid.astype(np.int32),
        structure_phase=structure_phase or [-1] * (int(structure_grid.max()) + 1),
    )


@pytest.fixture
def sample(monkeypatch):
    """An 8x8 map: an Al matrix with a 2x2 Si particle in it."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[3:5, 3:5] = 1
    al = np.full(64, 95.0)
    si = np.full(64, 5.0)
    flat = grid.ravel() == 1
    al[flat] = 40.0
    si[flat] = 60.0
    at_maps = {"Al": al, "Si": si}
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: (at_maps, 8, 8, "/tmp/x.h5oina"),
    )
    return _state(grid)


# --- composition -------------------------------------------------------------

def test_reports_the_measured_composition(sample):
    d = _structure_detail(sample, 1)
    got = {e["element"]: e["at_pct"] for e in d["elements"]}
    assert got["Si"] == pytest.approx(60.0)
    assert got["Al"] == pytest.approx(40.0)


def test_enrichment_is_a_factor_not_a_ratio_of_mismatched_units(sample):
    """The bug this exists for: Si really is enriched here, Al really is
    depleted, and both must land near a plausible magnitude rather than in
    the hundreds."""
    d = _structure_detail(sample, 1)
    e = {x["element"]: x["enrichment"] for x in d["elements"]}
    assert e["Si"] > 1.3, "a 60 at% Si particle in a 5 at% Si background is enriched"
    assert e["Al"] < 1.0, "aluminium is depleted inside the particle"
    assert e["Si"] < 100, f"implausible magnitude {e['Si']} - units are mismatched"


def test_enrichment_matches_the_wand_reporter(sample):
    """One definition. Two would let the inspector and the selection readout
    disagree about the same pixels."""
    from backend.api.services.chemistry_score import background_levels
    from backend.api.services.eds_wand import selection_stats

    at_maps, _r, _c, _f = __import__(
        "backend.api.routes.eds", fromlist=["x"]
    )._build_at_pct_maps_for_loaded_file()
    mask = sample.structure_grid == 1
    ref = selection_stats(at_maps, mask, background=background_levels(at_maps))
    d = _structure_detail(sample, 1)
    for e in d["elements"]:
        assert e["enrichment"] == ref["enrichment"].get(e["element"])


def test_spread_is_zero_for_a_uniform_structure(sample):
    d = _structure_detail(sample, 1)
    assert all(e["spread"] == 0.0 for e in d["elements"])


def test_elements_come_back_strongest_first(sample):
    d = _structure_detail(sample, 1)
    vals = [e["at_pct"] for e in d["elements"]]
    assert vals == sorted(vals, reverse=True)


# --- pieces ------------------------------------------------------------------

def test_counts_connected_pieces(sample):
    d = _structure_detail(sample, 1)
    assert d["n_pieces"] == 1
    assert d["pieces"] == [4]


def test_two_separate_blobs_of_one_structure_count_as_two(monkeypatch):
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[1, 1] = 1
    grid[6, 6] = 1
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": np.full(64, 100.0)}, 8, 8, "/tmp/x.h5oina"),
    )
    d = _structure_detail(_state(grid), 1)
    assert d["n_pieces"] == 2


# --- neighbours --------------------------------------------------------------

def test_lists_the_structures_it_touches_with_their_distance(sample):
    d = _structure_detail(sample, 1)
    assert [n["structure_id"] for n in d["neighbours"]] == [0]
    # Largest single-element difference: Si 60 vs 5.
    assert d["neighbours"][0]["gap_at_pct"] == pytest.approx(55.0, abs=0.5)
    assert d["neighbours"][0]["shared_edge_px"] > 0


def test_neighbours_are_ordered_by_how_close_they_are(monkeypatch):
    """Closest first, because that is the merge candidate."""
    grid = np.zeros((3, 9), dtype=np.int32)
    grid[:, 3:6] = 1
    grid[:, 6:] = 2
    si = np.zeros(27)
    si[grid.ravel() == 0] = 5.0
    si[grid.ravel() == 1] = 7.0      # close to structure 0
    si[grid.ravel() == 2] = 60.0     # far
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": 100.0 - si, "Si": si}, 3, 9, "/tmp/x.h5oina"),
    )
    d = _structure_detail(_state(grid), 1)
    gaps = [n["gap_at_pct"] for n in d["neighbours"]]
    assert gaps == sorted(gaps)
    assert d["neighbours"][0]["structure_id"] == 0


def test_a_structure_that_touches_nothing_reports_no_neighbours(monkeypatch):
    grid = np.zeros((4, 4), dtype=np.int32)
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": np.full(16, 100.0)}, 4, 4, "/tmp/x.h5oina"),
    )
    assert _structure_detail(_state(grid), 0)["neighbours"] == []


# --- candidates --------------------------------------------------------------

def test_candidates_are_ranked_by_how_far_off_they_are(sample):
    d = _structure_detail(sample, 1)
    gaps = [c["gap_at_pct"] for c in d["candidates"]]
    assert gaps == sorted(gaps)


def test_the_closest_candidate_is_the_one_a_person_would_pick(sample):
    """40 at% Al / 60 at% Si is nearest AlSi, not pure Al or pure Si."""
    d = _structure_detail(sample, 1)
    assert d["candidates"][0]["cif_filename"] == "AlSi.cif"


def test_every_candidate_is_offered_even_when_none_fits(sample):
    """A structure whose chemistry matches nothing must still be nameable -
    that is the case a manual correction exists for."""
    d = _structure_detail(sample, 1)
    assert len(d["candidates"]) == len(sample.phase_entries)


# --- degenerate cases --------------------------------------------------------

def test_an_empty_structure_returns_a_shape_not_an_error(monkeypatch):
    grid = np.zeros((4, 4), dtype=np.int32)
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": np.full(16, 100.0)}, 4, 4, "/tmp/x.h5oina"),
    )
    st = _state(grid, structure_phase=[-1, -1])
    d = _structure_detail(st, 1)          # id exists, no pixels
    assert d["n_pixels"] == 0
    assert d["elements"] == [] and d["neighbours"] == []


def test_without_the_composition_it_reports_geometry_and_stops(monkeypatch):
    """After a backend restart the structures survive but the composition
    does not. Reporting the pixels honestly beats inventing chemistry."""
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[0, 0] = 1

    def boom():
        raise RuntimeError("no file open")

    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file", boom)
    d = _structure_detail(_state(grid), 1)
    assert d["n_pixels"] == 1
    assert d["n_pieces"] == 1
    assert d["elements"] == []
    assert d["candidates"] == []


def test_a_composition_from_a_different_sized_map_is_refused(monkeypatch):
    """Mixing a stale at% map into a structure's mean would silently report
    another file's chemistry."""
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[0, 0] = 1
    monkeypatch.setattr(
        "backend.api.routes.eds._build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": np.full(100, 100.0)}, 10, 10, "/other.h5oina"),
    )
    d = _structure_detail(_state(grid), 1)
    assert d["elements"] == []


def test_the_detail_names_the_phase_not_only_its_index(sample):
    """A rule is keyed on the phase NAME, never on a position in the candidate
    list. Without the name here, "rule from this structure" has nothing to key
    on — which is exactly how it shipped broken the first time: the button was
    permanently disabled and said "select a structure first" while one was
    selected."""
    st = _state(sample.structure_grid, structure_phase=[-1, 2])
    d = _structure_detail(st, 1)
    assert d["phase_index"] == 2
    assert d["cif_filename"] == "AlSi.cif"
    # The fixture's `_entry` uses the filename as the formula too.
    assert d["formula"] == "AlSi.cif"


def test_an_unnamed_structure_reports_no_phase_rather_than_a_wrong_one(sample):
    st = _state(sample.structure_grid, structure_phase=[-1, -1])
    d = _structure_detail(st, 1)
    assert d["phase_index"] == -1
    assert d["cif_filename"] is None
