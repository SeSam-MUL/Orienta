"""Seeded selection on the EDS composition maps."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.eds_wand import (
    flood_from, selection_stats, wand_field,
)


def _two_features(n_rows=40, n_cols=40, noise=0.0, seed=0):
    """Al matrix with one Fe-rich square and one Si-rich square."""
    rng = np.random.default_rng(seed)
    al = np.full((n_rows, n_cols), 96.0)
    fe = np.full((n_rows, n_cols), 0.5)
    si = np.full((n_rows, n_cols), 0.5)
    al[5:15, 5:15], fe[5:15, 5:15] = 70.0, 26.0      # feature A
    al[25:35, 25:35], si[25:35, 25:35] = 60.0, 36.0  # feature B
    if noise:
        al += rng.normal(0, noise, al.shape)
        fe += rng.normal(0, noise, fe.shape)
        si += rng.normal(0, noise, si.shape)
    return {k: np.clip(v, 0, None).ravel() for k, v in
            (("Al", al), ("Fe", fe), ("Si", si))}, n_rows, n_cols


def _threshold_for(growth, target):
    """The growth-curve step closest to a target pixel count — this is how
    the UI drives the slider, so the tests drive it the same way."""
    return min(growth, key=lambda g: abs(g["n_pixels"] - target))["threshold"]


def test_the_fill_grows_from_the_core_outwards():
    """Smoothing gives the feature a blurred rim, so a tight threshold
    selects its core and a looser one its full extent. That IS the slider."""
    at, nr, nc = _two_features()
    field, _scale, growth, _seed = wand_field(at, nr, nc, 10, 10)
    core = flood_from(field, 10, 10, growth[0]["threshold"])
    assert 0 < core.sum() < 100, "the tightest step should be the core only"
    assert core[8:12, 8:12].all(), "the core must sit inside the feature"


def test_the_fill_covers_the_feature_and_stops_at_its_edge():
    """Ask the slider for a little more than the feature and you get the
    feature — not the map. Smoothing blurs the rim, so asking for exactly
    the nominal area lands just inside it."""
    at, nr, nc = _two_features()
    field, _scale, growth, _seed = wand_field(at, nr, nc, 10, 10)
    mask = flood_from(field, 10, 10, _threshold_for(growth, 130))
    covered = mask[5:15, 5:15].mean()
    assert covered > 0.9, f"only {covered*100:.0f}% of the feature selected"
    assert not mask[25:35, 25:35].any(), "it leaked into the other feature"
    assert mask.sum() < 400, f"it leaked into the matrix ({mask.sum()} px)"


def test_the_growth_curve_exposes_the_leak_before_the_user_commits():
    """There is always a threshold at which the fill escapes into the
    matrix. The curve must SHOW that jump, because the pixel count is the
    only warning the user gets before flooding half the map."""
    at, nr, nc = _two_features()
    _f, _s, growth, _seed = wand_field(at, nr, nc, 10, 10)
    counts = [g["n_pixels"] for g in growth]
    jumps = [b / max(a, 1) for a, b in zip(counts, counts[1:])]
    assert max(jumps) > 3, "the escape into the matrix is not visible in the curve"


def test_seeding_the_matrix_selects_the_matrix_not_the_features():
    at, nr, nc = _two_features()
    field, _s, _g, _seed = wand_field(at, nr, nc, 0, 0)
    mask = flood_from(field, 0, 0, threshold=20)
    assert not mask[5:15, 5:15].any()
    assert not mask[25:35, 25:35].any()


def test_smoothing_is_what_makes_it_work_on_noisy_data():
    """Raw EDS shatters; the module smooths on purpose. Without it a fill
    on realistic noise grows a fragment instead of the feature."""
    at, nr, nc = _two_features(noise=6.0, seed=3)
    smoothed, _s, _g, _seed = wand_field(at, nr, nc, 10, 10, smooth=5)
    raw, _s2, _g2, _seed2 = wand_field(at, nr, nc, 10, 10, smooth=1)
    n_smooth = flood_from(smoothed, 10, 10, 30).sum()
    n_raw = flood_from(raw, 10, 10, 30).sum()
    assert n_smooth > n_raw * 2, (
        f"smoothing did not help: {n_smooth} px smoothed vs {n_raw} px raw"
    )


def test_growth_curve_is_monotone_and_has_no_duplicate_plateaus():
    at, nr, nc = _two_features()
    _f, _s, growth, _seed = wand_field(at, nr, nc, 10, 10)
    counts = [g["n_pixels"] for g in growth]
    assert counts == sorted(counts), "growth must be monotone"
    assert len(counts) == len(set(counts)), "plateaus must be collapsed"
    assert counts[0] >= 1


def test_unmeasured_pixels_are_unreachable():
    at, nr, nc = _two_features()
    for v in at.values():
        v.reshape(nr, nc)[:, 30:] = 0.0        # dead strip
    field, _s, _g, _seed = wand_field(at, nr, nc, 10, 10)
    assert (field.reshape(nr, nc)[:, 30:] == 255).all()
    assert not flood_from(field, 10, 10, 254)[:, 30:].any()


def test_seeding_a_dead_pixel_selects_nothing():
    at, nr, nc = _two_features()
    for v in at.values():
        v.reshape(nr, nc)[0, 0] = 0.0
    field, _s, _g, _seed = wand_field(at, nr, nc, 0, 0)
    assert flood_from(field, 0, 0, 254).sum() == 0


def test_distance_is_readable_in_at_pct():
    at, nr, nc = _two_features()
    field, scale, _g, seed = wand_field(at, nr, nc, 10, 10)
    assert seed["Fe"] == pytest.approx(26.0, abs=1.0)
    # the matrix differs from the seed by roughly |96-70|+|0.5-26|+0 ~ 51 at%
    far = float(field[0, 0]) * scale
    assert 30.0 < far < 80.0, f"distance to the matrix reads {far:.1f} at%"


def test_seed_outside_the_grid_is_refused():
    at, nr, nc = _two_features()
    with pytest.raises(ValueError):
        wand_field(at, nr, nc, nr + 5, 0)


def test_selection_stats_report_composition_and_enrichment():
    at, nr, nc = _two_features()
    field, _s, _g, _seed = wand_field(at, nr, nc, 10, 10)
    mask = flood_from(field, 10, 10, 20)
    stats = selection_stats(at, mask, background={"Al": 0.96, "Fe": 0.005, "Si": 0.005})
    assert stats["n_pixels"] == int(mask.sum())
    assert stats["mean_at_pct"]["Fe"] > 20
    assert stats["enrichment"]["Fe"] > 5, "an Fe feature must read as Fe-enriched"


def test_empty_selection_is_safe():
    at, nr, nc = _two_features()
    assert selection_stats(at, np.zeros((nr, nc), bool))["n_pixels"] == 0
