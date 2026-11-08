"""A header that states only part of a pattern centre is not a pattern centre.

kikuchipy's EDAX reader takes each of x-star, y-star and z-star the header lacks from its
placeholder 0.5, and says nothing. A file with a partial calibration group used to be registered as
having a measured PC. These tests build the headers by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edax_hex  # noqa: E402
import safe_loader  # noqa: E402


def _edax_file(path, **stars):
    """An EDAX-layout .h5: ``Manufacturer`` EDAX and a calibration group holding ``stars``."""
    with h5py.File(path, "w") as f:
        f["Manufacturer"] = np.array([b"EDAX"])
        grp = f.create_group("Scan 1/EBSD/Header/Pattern Center Calibration")
        for key, value in stars.items():
            grp[key] = np.array([value], dtype=np.float32)
    return path


def _has_pc(path):
    with h5py.File(path, "r") as f:
        return safe_loader._header_has_pc(f)


def test_a_complete_calibration_group_is_a_pc(tmp_path):
    assert _has_pc(_edax_file(tmp_path / "a.h5", **{"x-star": 0.5, "y-star": 0.7, "z-star": 0.6})) is True


@pytest.mark.parametrize("stars", [
    {"x-star": 0.5, "y-star": 0.7},                         # no z-star
    {"y-star": 0.7, "z-star": 0.6},                         # no x-star
    {"x-star": 0.5, "y-star": float("nan"), "z-star": 0.6},
    {"x-star": 0.5, "y-star": float("inf"), "z-star": 0.6},
    {},
])
def test_a_partial_or_non_finite_group_is_not_a_pc(tmp_path, stars):
    assert _has_pc(_edax_file(tmp_path / "b.h5", **stars)) is False


def test_an_oxford_per_pixel_pc_is_still_a_pc(tmp_path):
    p = tmp_path / "o.h5oina"
    with h5py.File(p, "w") as f:
        f["1/EBSD/Data/Pattern Center X"] = np.array([0.5, 0.5], dtype=np.float32)
    assert _has_pc(p) is True


def test_a_file_without_any_calibration_is_not_a_pc(tmp_path):
    p = tmp_path / "n.h5"
    with h5py.File(p, "w") as f:
        f["Scan 1/EBSD/Header/Pattern Width"] = np.array([60])
    assert _has_pc(p) is False


def _signal():
    import kikuchipy as kp
    return kp.signals.EBSD(np.zeros((2, 2, 8, 8), np.uint8))


def test_the_loader_tags_a_partial_edax_header_as_having_no_pc(tmp_path):
    path = _edax_file(tmp_path / "partial.h5", **{"x-star": 0.5, "y-star": 0.7})
    sig = _signal()
    safe_loader._restore_pc_from_orienta_export(sig, str(path), verbose=False)
    assert sig.metadata.get_item("Signal.pc_source", None) == "default"


def test_the_loader_leaves_a_complete_edax_header_alone(tmp_path):
    path = _edax_file(tmp_path / "full.h5", **{"x-star": 0.5, "y-star": 0.7, "z-star": 0.6})
    sig = _signal()
    safe_loader._restore_pc_from_orienta_export(sig, str(path), verbose=False)
    assert sig.metadata.get_item("Signal.pc_source", None) is None


# ------------------------------------------------------------------ the hexagonal EDAX reader

def test_the_hex_header_pc_is_complete_only_with_all_three_finite_components():
    values, complete = edax_hex._header_pc({"x-star": np.array([0.51]), "y-star": 0.74, "z-star": 0.56})
    assert complete is True and values == pytest.approx((0.51, 0.74, 0.56))


@pytest.mark.parametrize("cal", [{}, {"x-star": 0.5, "y-star": 0.7}, {"x-star": 0.5, "y-star": None, "z-star": 0.6},
                                  {"x-star": 0.5, "y-star": 0.7, "z-star": float("nan")}])
def test_a_missing_component_is_the_placeholder_and_the_header_is_incomplete(cal):
    values, complete = edax_hex._header_pc(cal)
    assert complete is False
    for key, v in zip(("x-star", "y-star", "z-star"), values):
        if key not in cal or cal[key] is None or not np.isfinite(cal[key]):
            assert v == 0.5


def test_an_incomplete_header_tags_the_signal_like_every_loader_without_a_pc():
    import kikuchipy as kp
    tagged = kp.signals.EBSD(np.zeros((1, 1, 4, 4), np.uint8))
    edax_hex._tag_incomplete_pc(tagged, False, "x.h5")
    assert tagged.metadata.get_item("Signal.pc_source") == "default"
    complete = kp.signals.EBSD(np.zeros((1, 1, 4, 4), np.uint8))
    edax_hex._tag_incomplete_pc(complete, True, "x.h5")
    assert complete.metadata.get_item("Signal.pc_source", None) is None


def test_the_hex_loader_asks_the_header_and_tags_with_the_answer():
    """A synthetic HexGrid file is a lot of layout for one question; the loader's two lines are pinned
    structurally, and the real-file test below runs the whole path when a hex file is available."""
    import inspect
    src = inspect.getsource(edax_hex.load_edax_hex)
    assert "pc_values, pc_complete = _header_pc(pc_cal)" in src
    assert "pc=pc_values," in src
    assert "_tag_incomplete_pc(signal, pc_complete, file_name)" in src


def test_a_real_hex_file_with_a_pc_is_not_tagged():
    import os
    hex_file = os.environ.get("KIKUCHIPY_EDAX_HEX_FILE")
    if not (hex_file and os.path.exists(hex_file)):
        pytest.skip("KIKUCHIPY_EDAX_HEX_FILE is not set (a real EDAX HexGrid .h5)")
    sig = edax_hex.load_edax_hex(hex_file, lazy=True, verbose=False)
    assert sig.metadata.get_item("Signal.pc_source", None) is None
