"""The three pattern-centre conventions: every expectation is a hand calculation or a number
that a vendor's own file and kikuchipy's loader agree on (written next to the assertion)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_convention as pcc  # noqa: E402


# ------------------------------------------------------------------ hand calculations

def test_oxford_to_bruker_on_a_landscape_detector_matches_the_values_kikuchipy_loads_from_the_file():
    """An Oxford file of 128 x 156 patterns stores (0.5026, 0.5520, 0.6940) in
    ``/1/EBSD/Data/Pattern Center X, Y, Detector Distance`` (mean over the scan); kikuchipy's
    loader hands the application (0.5026, 0.3273, 0.8458). By hand, aspect = 156 / 128 = 1.21875:
    y = 1 - 0.5520 * 1.21875 = 0.32725 and z = 0.6940 * 1.21875 = 0.845813."""
    out = pcc.to_bruker((0.5026, 0.5520, 0.6940), "oxford", (128, 156))
    assert out == pytest.approx((0.5026, 0.32725, 0.845813), abs=1e-5)
    # and the rounded numbers the loader showed
    assert out == pytest.approx((0.5026, 0.3273, 0.8458), abs=2e-4)


def test_tsl_to_bruker_on_a_square_detector_only_turns_y():
    """EDAX Scan 5 (79 x 79): the .osc states x* = 0.5499, y* = 0.5026, z* = 0.7020 and the
    application works with Bruker (0.5499, 0.4974, 0.7020) (the harness log). By hand: a square
    pattern has min(rows, cols) / rows = 1, so z is unchanged and y = 1 - 0.5026 = 0.4974."""
    out = pcc.to_bruker((0.5499, 0.5026, 0.7020), "tsl", (79, 79))
    assert out == pytest.approx((0.5499, 0.4974, 0.7020), abs=1e-9)


def test_tsl_z_is_in_units_of_the_shorter_side_on_a_portrait_detector():
    """160 rows x 120 columns: min(160, 120) / 160 = 0.75. TSL (0.5, 0.6, 0.8) is Bruker
    (0.5, 1 - 0.6, 0.8 * 0.75) = (0.5, 0.4, 0.6)."""
    assert pcc.to_bruker((0.5, 0.6, 0.8), "tsl", (160, 120)) == pytest.approx((0.5, 0.4, 0.6), abs=1e-9)


def test_tsl_z_is_unchanged_on_a_landscape_detector():
    """128 rows x 156 columns: min(128, 156) / 128 = 1, so only y turns."""
    assert pcc.to_bruker((0.5, 0.6, 0.7), "tsl", (128, 156)) == pytest.approx((0.5, 0.4, 0.7), abs=1e-9)


def test_oxford_on_a_portrait_detector():
    """160 x 120: aspect = 120 / 160 = 0.75. Oxford (0.5, 0.4, 0.8) is Bruker
    (0.5, 1 - 0.4 * 0.75, 0.8 * 0.75) = (0.5, 0.7, 0.6)."""
    assert pcc.to_bruker((0.5, 0.4, 0.8), "oxford", (160, 120)) == pytest.approx((0.5, 0.7, 0.6), abs=1e-9)


def test_from_bruker_is_the_inverse_by_hand():
    """Bruker (0.5, 0.4, 0.6) on 160 x 120 (aspect 0.75, min side 120): TSL y = 1 - 0.4 = 0.6 and
    z = 0.6 / 0.75 = 0.8; Oxford y = (1 - 0.4) / 0.75 = 0.8 and z = 0.6 / 0.75 = 0.8."""
    assert pcc.from_bruker((0.5, 0.4, 0.6), "tsl", (160, 120)) == pytest.approx((0.5, 0.6, 0.8), abs=1e-9)
    # Oxford: y = (1 - 0.4) / 0.75 = 0.8, z = 0.6 / 0.75 = 0.8
    assert pcc.from_bruker((0.5, 0.4, 0.6), "oxford", (160, 120)) == pytest.approx((0.5, 0.8, 0.8), abs=1e-9)


def test_edax_is_another_name_for_tsl_and_names_ignore_case():
    assert pcc.canonical("EDAX") == "tsl" == pcc.canonical(" Tsl ")
    assert pcc.canonical("Bruker") == "bruker"
    assert pcc.to_bruker((0.5, 0.6, 0.7), "EDAX", (60, 60)) == pcc.to_bruker((0.5, 0.6, 0.7), "tsl", (60, 60))


def test_the_same_convention_returns_the_numbers_untouched():
    assert pcc.convert((0.1, 0.2, 0.3), "bruker", "bruker", (60, 60)) == (0.1, 0.2, 0.3)
    assert pcc.convert((0.1, 0.2, 0.3), "oxford", "oxford", (60, 60)) == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("shape", [(60, 60), (128, 156), (160, 120)])
@pytest.mark.parametrize("a,b", [("bruker", "tsl"), ("bruker", "oxford"), ("tsl", "oxford")])
def test_every_pair_round_trips(a, b, shape):
    pc = (0.47, 0.31, 0.58)
    there = pcc.convert(pc, a, b, shape)
    back = pcc.convert(there, b, a, shape)
    assert back == pytest.approx(pc, abs=1e-12)


# ------------------------------------------------------------------ refusals (never a guess)

@pytest.mark.parametrize("bad,code", [
    ({"source": "bruker", "target": "klingon"}, "unknownConvention"),
    ({"source": "", "target": "tsl"}, "unknownConvention"),
    ({"source": "bruker", "target": "tsl", "shape": None}, "needsShape"),
    ({"source": "bruker", "target": "tsl", "shape": (0, 60)}, "needsShape"),
    ({"source": "bruker", "target": "tsl", "pc": (0.5, float("nan"), 0.5)}, "badPc"),
    ({"source": "bruker", "target": "tsl", "pc": (0.5, 0.5)}, "badPc"),
])
def test_what_cannot_be_converted_is_refused_with_a_code(bad, code):
    args = {"pc": (0.5, 0.5, 0.5), "shape": (60, 60)}
    args.update(bad)
    with pytest.raises(pcc.ConventionError) as exc:
        pcc.convert(args["pc"], args["source"], args["target"], args["shape"])
    assert exc.value.code == code


@pytest.mark.parametrize("shape", [
    (float("inf"), 60), (60, float("inf")), (float("nan"), 60), {}, (), (60,), "66",
    (60, 60, 3), (10, 60, 60), [[60, 60]], (None, 60), ("a", "b"),
    (60.5, 60), (60, 0.5), (np.float64(60.7), 60),     # int() would cut these to (60, 60) or (60, 0)
])
def test_a_shape_that_is_not_exactly_rows_and_columns_is_refused_not_guessed(shape):
    """A stack shape (N, H, W) or (H, W, 3) would be read from its first two numbers: a guess."""
    with pytest.raises(pcc.ConventionError) as exc:
        pcc.convert((0.5, 0.5, 0.5), "bruker", "tsl", shape)
    assert exc.value.code == "needsShape"


def test_a_whole_number_given_as_a_float_is_a_valid_shape():
    """(160.0, 120.0) is 160 x 120: min(160, 120) / 160 = 0.75, so TSL (0.5, 0.6, 0.8) is
    Bruker (0.5, 1 - 0.6, 0.8 * 0.75) = (0.5, 0.4, 0.6), as for the integer shape."""
    assert pcc.to_bruker((0.5, 0.6, 0.8), "tsl", (160.0, 120.0)) == pytest.approx((0.5, 0.4, 0.6), abs=1e-9)
    assert pcc.to_bruker((0.5, 0.6, 0.8), "tsl", (np.float64(160.0), np.int64(120))) == pytest.approx(
        (0.5, 0.4, 0.6), abs=1e-9)


@pytest.mark.parametrize("pc", [
    None, "abc", "123", 5, (0.5, None, 0.5), (0.5, (0.5,), 0.5), [[0.5, 0.5, 0.5]],
    ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)), (0.5, "x", 0.5), (10 ** 400, 0.5, 0.5),
    (True, False, True), (0.5, True, 0.5), (0.5, 0.5, 0.5, 0.5), (),
])
def test_a_pattern_centre_that_is_not_three_plain_numbers_is_refused_with_a_code(pc):
    with pytest.raises(pcc.ConventionError) as exc:
        pcc.convert(pc, "bruker", "tsl", (60, 60))
    assert exc.value.code == "badPc"


def test_a_finite_pc_whose_conversion_overflows_is_refused_not_returned_as_inf():
    """1e308 is finite, but the Oxford PCz of a 60 x 120 pattern is 1e308 * 120 / 60: infinity."""
    with pytest.raises(pcc.ConventionError) as exc:
        pcc.convert((0.5, 0.5, 1e308), "oxford", "bruker", (60, 120))
    assert exc.value.code == "badPc"


@pytest.mark.parametrize("pc", [(float("nan"), 0.5, 0.5), (0.5, float("inf"), 0.5), (0.5, 0.5, float("-inf"))])
def test_a_non_finite_pc_is_refused_even_when_no_conversion_is_needed(pc):
    """The same-convention path returns the numbers as they are, so it must not pass a nan on."""
    with pytest.raises(pcc.ConventionError) as exc:
        pcc.convert(pc, "oxford", "oxford", (60, 60))
    assert exc.value.code == "badPc"


# ------------------------------------------------------------------ the route

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    return TestClient(app)


def test_the_route_converts(client):
    r = client.post("/api/pc/convert", json={
        "pc": [0.5026, 0.5520, 0.6940], "source": "oxford", "target": "bruker", "shape": [128, 156]})
    assert r.status_code == 200, r.text
    assert r.json()["pc"] == pytest.approx([0.5026, 0.32725, 0.845813], abs=1e-5)
    assert r.json()["target"] == "bruker"


def test_the_route_refuses_with_the_code_in_the_header(client):
    r = client.post("/api/pc/convert", json={
        "pc": [0.5, 0.5, 0.5], "source": "klingon", "target": "bruker", "shape": [60, 60]})
    assert r.status_code == 400
    assert r.headers["X-Orienta-Code"] == "unknownConvention"
    r = client.post("/api/pc/convert", json={
        "pc": [0.5, 0.5, 0.5], "source": "tsl", "target": "bruker", "shape": [0, 0]})
    assert r.status_code == 400 and r.headers["X-Orienta-Code"] == "needsShape"
