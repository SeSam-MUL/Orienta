"""Hand-declared regions, and telling the grouping which elements matter.

Both come from one user report:

    "pure Silicon and AlFeMnSi phases are not possible to separate from each
     other because the automatic definition of regions based on composition
     lumps them into the same category. The high amount of Al background
     signal is likely the issue here ... The option to manually define and
     edit the number of regions, their composition ranges etc. would allow
     the user to decide themselves what the ideal segmentation looks like."

The two answers are deliberately separate tools: weights keep the grouping
automatic but change what it cares about; a definition takes the decision away
from the fit entirely. What neither may do is change the automatic path when
it is not in use, so that is the first thing tested here.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.eds_clustering import (
    _feature_matrix, choose_k_by_distinctness, cluster_and_match,
)
from backend.api.services.phase_rules import (
    ElementRange, RatioRange, RegionDefinition, region_defs_from_list,
    region_defs_to_list, region_labels,
)


def _entry(name, comp):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="",
        space_group_number=None, crystal_system="", composition=comp,
        elements=sorted(comp),
    )


N_ROWS = N_COLS = 12


def _synthetic():
    """An Al matrix with a silicon particle and an Fe-bearing particle.

    Built to reproduce the reported failure mode: the two particles differ
    from each other mostly in iron, a couple of at% on top of a dominant
    aluminium background.
    """
    n = N_ROWS * N_COLS
    al = np.full(n, 92.0)
    si = np.full(n, 6.0)
    fe = np.full(n, 1.0)
    mn = np.full(n, 1.0)
    g = np.arange(n).reshape(N_ROWS, N_COLS)
    si_particle = ((g // N_COLS < 3) & (g % N_COLS < 3)).ravel()
    fe_particle = ((g // N_COLS > 8) & (g % N_COLS > 8)).ravel()
    al[si_particle], si[si_particle], fe[si_particle] = 55.0, 44.0, 0.5
    al[fe_particle], si[fe_particle], fe[fe_particle] = 84.0, 6.0, 7.0
    return {"Al": al, "Si": si, "Fe": fe, "Mn": mn}, si_particle, fe_particle


# --- the automatic path must not move ---------------------------------------

def test_no_weights_is_the_same_feature_matrix():
    at, _, _ = _synthetic()
    _, plain = _feature_matrix(at, N_ROWS * N_COLS)
    _, unit = _feature_matrix(at, N_ROWS * N_COLS, {"Fe": 1.0, "Al": 1.0})
    assert np.array_equal(plain, unit)


def test_empty_weights_leave_the_run_bit_identical():
    at, _, _ = _synthetic()
    a = cluster_and_match(at, N_ROWS, N_COLS, [], k=4)
    b = cluster_and_match(at, N_ROWS, N_COLS, [], k=4, element_weights={})
    assert np.array_equal(a[1], b[1])


def test_no_definitions_leaves_the_run_bit_identical():
    at, _, _ = _synthetic()
    a = cluster_and_match(at, N_ROWS, N_COLS, [], k=4)
    b = cluster_and_match(at, N_ROWS, N_COLS, [], k=4, region_defs=[])
    assert np.array_equal(a[1], b[1])


# --- element weights ---------------------------------------------------------

def test_a_weight_scales_only_its_own_column():
    at, _, _ = _synthetic()
    els, plain = _feature_matrix(at, N_ROWS * N_COLS)
    _, w = _feature_matrix(at, N_ROWS * N_COLS, {"Fe": 4.0})
    fe = els.index("Fe")
    assert np.allclose(w[:, fe], plain[:, fe] * 4.0)
    other = [i for i in range(len(els)) if i != fe]
    assert np.allclose(w[:, other], plain[:, other])


def test_a_negative_or_nonfinite_weight_is_ignored_rather_than_obeyed():
    """A weight is a user-typed number; a stray minus must not fold the
    feature space through the origin and silently regroup the map."""
    at, _, _ = _synthetic()
    _, plain = _feature_matrix(at, N_ROWS * N_COLS)
    _, bad = _feature_matrix(at, N_ROWS * N_COLS,
                             {"Fe": -3.0, "Al": float("nan")})
    assert np.array_equal(plain, bad)


def test_the_distinctness_gap_is_measured_in_at_pct_not_in_weighted_units():
    """Weights say WHICH elements group the map; min_gap says how far apart
    two chemistries must be to count as two. Measuring the gap on weighted
    values couples them - measured on SampleB that pushed k from 7 to 12 and
    coherence from 0.933 to 0.800 without separating anything."""
    at, _, _ = _synthetic()
    n = N_ROWS * N_COLS
    _, plain = _feature_matrix(at, n)
    _, heavy = _feature_matrix(at, n, {"Fe": 8.0})
    k_phys = choose_k_by_distinctness(heavy, (2, 8), gap_X=plain)
    k_weighted = choose_k_by_distinctness(heavy, (2, 8))
    assert k_phys <= k_weighted


# --- definitions claim pixels ------------------------------------------------

def test_a_definition_claims_exactly_the_pixels_in_its_window():
    at, si_particle, _ = _synthetic()
    d = RegionDefinition("Si", elements=(ElementRange("Si", 20.0, None),))
    asg = region_labels([d], at)
    assert np.array_equal(asg.labels == 0, si_particle)


def test_two_particles_that_differ_only_in_iron_land_in_different_regions():
    """The reported failure, stated as a test."""
    at, si_particle, fe_particle = _synthetic()
    defs = [
        RegionDefinition("Si", elements=(ElementRange("Si", 20.0, None),
                                         ElementRange("Fe", None, 2.0))),
        RegionDefinition("AlFeMnSi", elements=(ElementRange("Fe", 3.0, None),)),
    ]
    asg = region_labels(defs, at)
    assert np.array_equal(asg.labels == 0, si_particle)
    assert np.array_equal(asg.labels == 1, fe_particle)


def test_an_empty_definition_claims_nothing_rather_than_everything():
    """A window with no clauses is a half-finished edit. Reading it as "the
    whole map" would wipe the map on the way to typing the first threshold."""
    at, _, _ = _synthetic()
    asg = region_labels([RegionDefinition("wip")], at)
    assert asg.counts == [0]
    assert (asg.labels < 0).all()
    assert asg.reasons[0]


def test_the_first_definition_wins_an_overlap_and_the_loss_is_reported():
    at, si_particle, _ = _synthetic()
    a = RegionDefinition("first", elements=(ElementRange("Si", 20.0, None),))
    b = RegionDefinition("second", elements=(ElementRange("Si", 10.0, None),))
    asg = region_labels([a, b], at)
    assert np.array_equal(asg.labels == 0, si_particle)
    assert asg.counts[1] == 0
    assert asg.overlaps[1] == int(si_particle.sum())


def test_a_definition_naming_an_unmeasured_element_claims_nothing_and_says_so():
    at, _, _ = _synthetic()
    d = RegionDefinition("Mg", elements=(ElementRange("Mg", 5.0, None),))
    asg = region_labels([d], at)
    assert asg.counts == [0]
    assert "not measured" in (asg.reasons[0] or "")


def test_ratio_clauses_work_in_a_definition_too():
    at, si_particle, _ = _synthetic()
    d = RegionDefinition("Si rich", ratios=(RatioRange("Si", "Al", 0.5, None),))
    asg = region_labels([d], at)
    assert np.array_equal(asg.labels == 0, si_particle)


# --- definitions inside the classification ----------------------------------

def test_declared_regions_come_first_and_the_rest_is_still_grouped():
    at, si_particle, _ = _synthetic()
    defs = [RegionDefinition("Si", elements=(ElementRange("Si", 20.0, None),))]
    _, grid, _, k = cluster_and_match(at, N_ROWS, N_COLS, [], k=3,
                                      region_defs=defs, scale=0)
    assert np.array_equal(grid.ravel() == 0, si_particle)
    assert k > 1, "the undeclared remainder must still be grouped"


def test_cluster_remainder_false_gives_exactly_the_declared_regions_plus_one():
    at, _, _ = _synthetic()
    defs = [
        RegionDefinition("Si", elements=(ElementRange("Si", 20.0, None),)),
        RegionDefinition("Fe", elements=(ElementRange("Fe", 3.0, None),)),
    ]
    _, _, _, k = cluster_and_match(at, N_ROWS, N_COLS, [], k=None,
                                   region_defs=defs, cluster_remainder=False,
                                   scale=0)
    assert k == 3, "two declared regions and one for everything else"


def test_the_leftover_is_a_region_not_a_hole():
    """Leaving undeclared pixels at -1 would drop them out of the map instead
    of showing what is still unaccounted for."""
    at, _, _ = _synthetic()
    defs = [RegionDefinition("Si", elements=(ElementRange("Si", 20.0, None),))]
    _, grid, _, _ = cluster_and_match(at, N_ROWS, N_COLS, [], k=None,
                                      region_defs=defs, cluster_remainder=False,
                                      scale=0)
    assert (grid >= 0).all()


def test_region_ids_stay_dense_when_a_definition_claims_nothing():
    """Ids are positions; a gap would misname every region after it."""
    at, _, _ = _synthetic()
    defs = [
        RegionDefinition("nothing", elements=(ElementRange("Si", 99.0, None),)),
        RegionDefinition("Si", elements=(ElementRange("Si", 20.0, None),)),
    ]
    _, grid, _, k = cluster_and_match(at, N_ROWS, N_COLS, [], k=None,
                                      region_defs=defs, cluster_remainder=False,
                                      scale=0)
    assert sorted(int(v) for v in np.unique(grid)) == list(range(k))


def test_a_declared_phase_key_survives_the_matcher():
    """The matcher advises; it must not overrule a name the user chose."""
    at, si_particle, _ = _synthetic()
    cands = [_entry("Al.cif", {"Al": 100.0}), _entry("Si.cif", {"Si": 100.0})]
    defs = [RegionDefinition(
        "Si", elements=(ElementRange("Si", 20.0, None),), phase_key="Al.cif")]
    phase_grid, _, _, _ = cluster_and_match(at, N_ROWS, N_COLS, cands, k=2,
                                            region_defs=defs, scale=0)
    assert (phase_grid.ravel()[si_particle] == 0).all(), (
        "the declared phase must win over the library's own preference")


def test_a_definition_without_a_phase_key_is_still_matched_normally():
    at, _, _ = _synthetic()
    cands = [_entry("Al.cif", {"Al": 100.0}), _entry("Si.cif", {"Si": 100.0})]
    defs = [RegionDefinition("Si", elements=(ElementRange("Si", 20.0, None),))]
    _, _, matches, _ = cluster_and_match(at, N_ROWS, N_COLS, cands, k=2,
                                         region_defs=defs, scale=0)
    assert any(m.phase_index >= 0 for m in matches)


# --- round trip --------------------------------------------------------------

def test_definitions_round_trip_through_the_wire():
    defs = [RegionDefinition(
        "Si", elements=(ElementRange("Si", 20.0, 60.0),),
        ratios=(RatioRange("Si", "Al", 0.5, None),), phase_key="Si.cif")]
    back = region_defs_from_list(region_defs_to_list(defs))
    assert back == defs


def test_a_definition_with_no_name_still_parses():
    back = region_defs_from_list([{"elements": [{"element": "Si",
                                                 "min_at_pct": 20}]}])
    assert len(back) == 1 and back[0].name
