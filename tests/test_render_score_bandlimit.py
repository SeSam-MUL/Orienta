"""The render comparison must not spend its norm on noise the model cannot have.

An experimental pattern carries detector noise at the pixel scale; an SHT render
does not. Correlated across the full frequency range, most of the norm goes into
content with no counterpart in the model and the score collapses. Measured on a
real 7050 map (40 pixels, with geometry, pattern centre and orientation each
verified to be AT their optimum first, so this compensates for no mis-set
parameter):

    as shipped (no band limit)   median render-NCC 0.196,  30% above 0.25
    band-limit both sides        median render-NCC 0.417,  90% above 0.25
                                 higher on 40 of 40 pixels

and it improves the DECISION, not just the level — contrast against the noise
floor 0.096 -> 0.221, phase gap 0.119 -> 0.244, while the peak stays as sharp
(5 degrees off the answer the score is already gone in both recipes).

These tests use a stand-in renderer so they pin the RULE without a GPU: the same
band limit must reach both sides, and it must be the noisy comparison that gains.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H, W = 40, 48
DET = {
    "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.6,
    "pat_height": H, "pat_width": W,
    "pixel_size": 70.0, "tilt": 0.0, "sample_tilt": 70.0,
    "binning": 1, "vendor": "Bruker",
}
IDENT = np.array([[1.0, 0.0, 0.0, 0.0]])


def _bands(seed=0, noise=0.0):
    """A band-like pattern: smooth structure plus optional pixel-scale noise."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = (np.sin(xx / 3.0) + np.sin((xx + yy) / 5.0) + 0.5 * np.cos(yy / 4.0))
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img.astype(np.float32)


CLEAN = _bands()


@pytest.fixture
def stub_renderer(monkeypatch):
    """Renderer that always returns the CLEAN pattern — i.e. a perfect model."""
    from backend.api.services import sht_pattern_renderer as svc

    class _Sim:
        def numpy(self):
            return CLEAN

    class _Rnd:
        def render(self, *_a, **_kw):
            return _Sim()

    monkeypatch.setattr(svc, "get_renderer", lambda: _Rnd())
    monkeypatch.setattr(svc, "load_or_get_phase", lambda *a, **kw: object())


def _score(get_pattern, **kw):
    from backend.spherical_gpu.pipeline.variant_unification import build_render_score_fn
    fn = build_render_score_fn("stub.sht", DET, get_pattern, **kw)
    return float(np.asarray(fn(np.array([0]), IDENT)).reshape(-1)[0])


def test_defaults_are_the_measured_ones():
    from backend.spherical_gpu.pipeline.variant_unification import (
        RENDER_APERTURE_FRAC, RENDER_LOWPASS_SIGMA,
    )
    assert RENDER_LOWPASS_SIGMA == 2.0
    assert RENDER_APERTURE_FRAC == 0.80


def test_a_noisy_pattern_scores_higher_once_both_sides_are_band_limited(stub_renderer):
    """The whole claim, in one comparison.

    The experiment is the model plus pixel-scale noise — the real situation.
    Band-limiting both sides must raise the score, because the noise it drops
    had no counterpart in the model to begin with.
    """
    noisy = _bands(noise=0.8)
    without = _score(lambda _f: noisy, lowpass_sigma=0.0)
    with_bl = _score(lambda _f: noisy, lowpass_sigma=2.0)
    assert with_bl > without + 0.05, (
        f"band limit gained nothing: {without:.3f} -> {with_bl:.3f}"
    )


def test_it_does_not_manufacture_agreement_where_there_is_none(stub_renderer):
    """A blur that made everything match would be worse than useless.

    Correlated against an UNRELATED pattern the score must stay near zero with
    the band limit on — otherwise the gain above is just smoothing two things
    into the same blob.
    """
    rng = np.random.default_rng(5)
    unrelated = rng.normal(0, 1, (H, W)).astype(np.float32)
    assert abs(_score(lambda _f: unrelated, lowpass_sigma=2.0)) < 0.25


def test_sigma_zero_is_the_previous_behaviour(stub_renderer):
    """An escape hatch that really is off, so an old calibration can be
    reproduced exactly."""
    noisy = _bands(noise=0.8)
    a = _score(lambda _f: noisy, lowpass_sigma=0.0)
    b = _score(lambda _f: noisy, lowpass_sigma=0.0)
    assert a == b
    assert a != _score(lambda _f: noisy, lowpass_sigma=2.0)


def test_the_band_limit_reaches_BOTH_sides(stub_renderer):
    """Filtering only the experiment would be a different, weaker recipe.

    Measured on real data: experiment only 0.343, both sides 0.421. If a future
    edit dropped the filter from the render side this test is what notices —
    with an unfiltered model the perfect-model case cannot reach ~1.
    """
    # Experiment IS the model here, so a correctly symmetric pipeline must
    # score essentially 1: both sides get the same treatment, whatever it is.
    assert _score(lambda _f: CLEAN, lowpass_sigma=2.0) > 0.99


def test_a_missing_pattern_still_scores_minus_inf(stub_renderer):
    """Unchanged contract: a pixel with no pattern never decides anything."""
    assert _score(lambda _f: None, lowpass_sigma=2.0) == float("-inf")


def test_scoring_a_pixel_does_not_corrupt_the_caller_s_pattern(stub_renderer):
    """The scorer must not write into the array it was handed.

    `np.asarray` does not copy a float32 array, so the EBSD signal wrapped the
    caller's buffer and `remove_dynamic_background` subtracted IN PLACE. The
    phase check scores the stored phase and then EVERY candidate on the same
    pixels, so each candidate was compared against a pattern that had already
    been background-corrected once per earlier phase. Two identical calls
    returned 0.715 and 0.623 (2026-09-06).
    """
    pattern = _bands(noise=0.5)
    untouched = pattern.copy()

    first = _score(lambda _f: pattern, lowpass_sigma=2.0)
    assert np.array_equal(pattern, untouched), "the scorer wrote into its input"

    second = _score(lambda _f: pattern, lowpass_sigma=2.0)
    assert first == pytest.approx(second, abs=1e-12), (
        "the same pixel scored differently the second time — state leaked "
        "between calls"
    )
