"""Clustering must recover compositional regions and match them on the
cluster mean, not on individual noisy pixels.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.eds_clustering import cluster_and_match, phase_coherence


def _entry(name, comp):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="",
        space_group_number=None, crystal_system="", composition=comp,
        elements=sorted(comp),
    )


CANDS = [
    _entry("Al.cif", {"Al": 100.0}),
    _entry("sd_0302719.cif", {"Al": 75.8, "Fe": 11.6, "Si": 9.9, "Mn": 2.7}),
]


def _two_region_map(n_rows=40, n_cols=40, noise=2.0, seed=0):
    """Left half pure Al, right half an Fe-Mn-Si intermetallic, plus noise."""
    rng = np.random.default_rng(seed)
    n = n_rows * n_cols
    col = np.tile(np.arange(n_cols), n_rows)
    right = col >= n_cols // 2
    at = {
        "Al": np.where(right, 75.8, 99.0) + rng.normal(0, noise, n),
        "Fe": np.where(right, 11.6, 0.4) + rng.normal(0, noise * 0.3, n),
        "Si": np.where(right, 9.9, 0.4) + rng.normal(0, noise * 0.3, n),
        "Mn": np.where(right, 2.7, 0.2) + rng.normal(0, noise * 0.2, n),
    }
    return {k: np.clip(v, 0, None) for k, v in at.items()}, right, n_rows, n_cols


def test_recovers_two_regions_and_matches_each():
    at, right, n_rows, n_cols = _two_region_map()
    _grid, clusters, matches, _k = cluster_and_match(
        at, n_rows, n_cols, CANDS, k=2)
    names = {m.cluster_id: CANDS[m.phase_index].cif_filename
             for m in matches if m.phase_index >= 0}
    labels = np.array([names.get(int(c), "?") for c in clusters.ravel()])
    assert (labels[~right] == "Al.cif").mean() > 0.95
    assert (labels[right] == "sd_0302719.cif").mean() > 0.95


def test_matching_uses_the_cluster_mean_not_single_pixels():
    """Noise that would defeat per-pixel matching must not defeat the cluster."""
    at, _right, n_rows, n_cols = _two_region_map(noise=6.0, seed=3)
    _grid, _clusters, matches, _k = cluster_and_match(
        at, n_rows, n_cols, CANDS, k=2)
    al = [m for m in matches if m.phase_index >= 0
          and CANDS[m.phase_index].cif_filename == "Al.cif"]
    assert al, "the aluminium matrix cluster must still match Al.cif"
    assert al[0].mean_at_pct["Al"] > 95.0, (
        f"eroded-interior mean should be clean, got {al[0].mean_at_pct['Al']:.1f}"
    )


def test_phase_grid_paints_every_pixel_of_a_matched_cluster():
    at, _right, n_rows, n_cols = _two_region_map()
    grid, clusters, matches, _k = cluster_and_match(
        at, n_rows, n_cols, CANDS, k=2)
    for m in matches:
        if m.phase_index >= 0:
            assert (grid.ravel()[clusters.ravel() == m.cluster_id]
                    == m.phase_index).all()


def test_chooses_k_automatically_within_range():
    at, _right, n_rows, n_cols = _two_region_map()
    _grid, _clusters, _matches, k = cluster_and_match(
        at, n_rows, n_cols, CANDS, k=None, k_range=(2, 6))
    assert 2 <= k <= 6


def test_auto_k_prefers_a_spatially_coherent_map():
    """The chosen k must not shatter the map into confetti.

    This is the regression for the measured failure of BIC: on SampleB it
    chose k=12, which split one physical phase's noise distribution across
    clusters that straddle the phase boundary and interleave spatially
    (18.6 % of pixels disagreeing with their neighbourhood, vs 1.5 % at k=4).
    """
    at, _right, n_rows, n_cols = _two_region_map(noise=4.0, seed=7)
    grid, _clusters, _matches, _k = cluster_and_match(
        at, n_rows, n_cols, CANDS, k=None, k_range=(2, 8))
    assert phase_coherence(grid) > 0.95


def test_auto_k_still_resolves_more_than_one_phase():
    """Collapsing everything to one phase would score perfect coherence."""
    at, _right, n_rows, n_cols = _two_region_map()
    grid, _clusters, _matches, _k = cluster_and_match(
        at, n_rows, n_cols, CANDS, k=None, k_range=(2, 8))
    assert len({int(v) for v in np.unique(grid) if v >= 0}) >= 2


def test_phase_coherence_bounds():
    assert phase_coherence(np.zeros((5, 5), dtype=np.int32)) == 1.0
    checker = np.indices((6, 6)).sum(axis=0) % 2
    assert phase_coherence(checker.astype(np.int32)) == 0.0


def test_cluster_with_no_acceptable_match_is_reported_not_forced():
    at, _right, n_rows, n_cols = _two_region_map()
    only_silicon = [_entry("Si.cif", {"Si": 100.0})]
    grid, _clusters, matches, _k = cluster_and_match(
        at, n_rows, n_cols, only_silicon, k=2, min_score=0.3)
    assert all(m.phase_index == -1 for m in matches)
    assert (grid == -1).all(), "unmatched clusters must not be forced onto a phase"


def test_empty_candidate_list_is_safe():
    at, _right, n_rows, n_cols = _two_region_map()
    grid, _clusters, matches, _k = cluster_and_match(at, n_rows, n_cols, [], k=2)
    assert (grid == -1).all()
    assert all(m.phase_index == -1 for m in matches)


def test_no_chemistry_at_all_is_safe():
    grid, clusters, matches, k = cluster_and_match({}, 4, 4, CANDS, k=2)
    assert (grid == -1).all() and (clusters == -1).all()
    assert matches == [] and k == 0


def test_runners_up_are_reported():
    at, _right, n_rows, n_cols = _two_region_map()
    _grid, _clusters, matches, _k = cluster_and_match(
        at, n_rows, n_cols, CANDS, k=2)
    assert any(m.runners_up for m in matches), "expected a runner-up to be listed"
    for m in matches:
        for idx, score in m.runners_up:
            assert 0 <= idx < len(CANDS)
            assert 0.0 <= score <= 1.0
