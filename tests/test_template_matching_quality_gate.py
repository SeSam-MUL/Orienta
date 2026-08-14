"""Dictionary indexing must say when the patterns are too noisy for it.

Image quality as THIS function reports it — after the background removal it
performs, which is what the thresholds are compared against:

    LoGainNi.h5    image quality 0.3616  ->  NCC 0.469   works
    HiGainNi.h5                  0.0285  ->  NCC 0.115   noise
    AL_SI_x3000 raw             ~0.022   ->  NCC 0.045   noise

The LoGainNi NCC is the full-map 2 deg run against Hough (28,086 px). An
earlier version of this docstring quoted 0.374 / 0.037 — those are the RAW
image qualities, measured without the background removal, so they were not the
numbers the gate actually sees.

Hough and spherical indexing produce correct grain maps on all of these,
because they integrate over band positions. Template matching correlates whole
patterns pixel-by-pixel, so on the noisy ones it returns a near-uniform map of
meaningless orientations WITHOUT any error — which is how a whole day was lost
chasing a non-existent bug in the dictionary code.
"""
from pathlib import Path

import numpy as np
import pytest

from backend.api.services.pattern_quality import (
    DICT_IQ_GOOD,
    DICT_IQ_MINIMUM,
    assess_for_template_matching,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "Test_data"
LOGAIN = TEST_DATA / "LoGainNi.h5"
HIGAIN = TEST_DATA / "HiGainNi.h5"


class _FakeSignal:
    def __init__(self, data):
        self.data = data


def test_thresholds_are_ordered():
    assert 0 < DICT_IQ_MINIMUM < DICT_IQ_GOOD < 1


def test_non_4d_signal_is_unknown_not_a_crash():
    out = assess_for_template_matching(_FakeSignal(np.zeros((4, 4))))
    assert out["verdict"] == "unknown"
    assert out["image_quality"] is None


def test_pure_noise_is_reported_as_too_noisy():
    rng = np.random.default_rng(0)
    noise = rng.normal(128, 20, size=(8, 8, 60, 60)).astype(np.float32)
    out = assess_for_template_matching(_FakeSignal(noise))
    assert out["verdict"] == "too_noisy", out
    assert "Hough" in out["detail"], "must point at a method that does work"


def test_assessment_never_raises_on_broken_input():
    """It informs; it must never be able to block an indexing run."""
    class _Boom:
        @property
        def data(self):
            raise RuntimeError("nope")

    out = assess_for_template_matching(_Boom())
    assert out["verdict"] == "unknown"


@pytest.mark.skipif(not LOGAIN.is_file(), reason="LoGainNi.h5 not available")
def test_logain_ni_is_good_enough():
    """The positive control: this one reaches NCC 0.43 in a real run."""
    import kikuchipy as kp

    out = assess_for_template_matching(kp.load(str(LOGAIN), lazy=True))
    assert out["verdict"] == "good", out
    assert out["image_quality"] > DICT_IQ_GOOD


@pytest.mark.skipif(not HIGAIN.is_file(), reason="HiGainNi.h5 not available")
def test_higain_ni_is_flagged():
    """Despite the name, HiGain = high camera gain = short exposure = noisy.
    Its best achievable NCC is 0.115."""
    import kikuchipy as kp

    out = assess_for_template_matching(kp.load(str(HIGAIN), lazy=True))
    assert out["verdict"] == "too_noisy", out
    assert out["image_quality"] < DICT_IQ_MINIMUM


@pytest.mark.skipif(not (LOGAIN.is_file() and HIGAIN.is_file()),
                    reason="both Ni gain datasets needed")
def test_the_two_gain_datasets_are_an_order_of_magnitude_apart():
    """Guards the threshold against drift in the quality measure itself."""
    import kikuchipy as kp

    lo = assess_for_template_matching(kp.load(str(LOGAIN), lazy=True))["image_quality"]
    hi = assess_for_template_matching(kp.load(str(HIGAIN), lazy=True))["image_quality"]
    assert lo > 5 * hi, f"LoGain {lo} vs HiGain {hi}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
