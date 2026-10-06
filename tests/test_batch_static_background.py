"""Batch preprocessing "static" background: scan average per file, not pattern (0,0).

Same semantics as the EBSD Viewer's BG Static (see
test_ebsd_viewer_static_background.py): no reference named -> mean of all
patterns of THAT file; row and column named -> that pattern; only one of the two
-> error recorded for the file and the data left untouched.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROWS, COLS, H, W = 5, 7, 40, 48


def _make_data(seed: int, illum_level: float = 110.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    illum = illum_level + 20.0 * np.exp(-(((yy - H * 0.7) / H) ** 2 + ((xx - W / 2) / W) ** 2) * 4)
    data = np.empty((ROWS, COLS, H, W), dtype=np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            pat = illum.copy()
            for _ in range(4):
                theta = rng.uniform(0, np.pi)
                offset = rng.uniform(-8, 8)
                dist = (xx - W / 2) * np.sin(theta) - (yy - H / 2) * np.cos(theta) - offset
                pat += 50.0 * np.exp(-(dist / 1.5) ** 2)
            pat += rng.normal(0, 3, size=pat.shape)
            data[r, c] = np.clip(pat, 0, 255).astype(np.uint8)
    return data


def _cc(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _signal(data, lazy):
    import kikuchipy as kp

    sig = kp.signals.EBSD(data.copy())
    return sig.as_lazy() if lazy else sig


def _reference(data, bg):
    import kikuchipy as kp

    ref = kp.signals.EBSD(data.copy())
    ref.remove_static_background(operation="subtract", static_bg=bg, show_progressbar=False)
    return np.asarray(ref.data)


CFG = {"background_removal": True, "background_method": "static"}


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_static_without_reference_uses_scan_average_not_pattern_00(lazy):
    from backend.api.services.batch_manager import _apply_preprocessing

    raw = _make_data(0)
    sig, applied = _apply_preprocessing(_signal(raw, lazy), dict(CFG))
    out = np.asarray(sig.data)

    others = [(i, j) for i in range(ROWS) for j in range(COLS) if (i, j) != (0, 0)]
    ghost = np.mean([_cc(out[p], raw[0, 0]) for p in others])
    baseline = np.mean([_cc(raw[p], raw[0, 0]) for p in others])
    assert ghost > baseline - 0.3, f"pattern (0,0) imprinted (cc {ghost:+.3f}, raw {baseline:+.3f})"

    mean_bg = np.rint(raw.reshape(-1, H, W).mean(axis=0)).astype(np.uint8)
    assert np.abs(out.astype(int) - _reference(raw, mean_bg).astype(int)).max() <= 2
    assert applied["background_removal"] == "static@scan_average"


def test_each_file_gets_its_own_average():
    from backend.api.services.batch_manager import _apply_preprocessing

    a, b = _make_data(1, 90.0), _make_data(2, 150.0)
    out_a = np.asarray(_apply_preprocessing(_signal(a, False), dict(CFG))[0].data)
    out_b = np.asarray(_apply_preprocessing(_signal(b, False), dict(CFG))[0].data)
    for raw, out in ((a, out_a), (b, out_b)):
        mean_bg = np.rint(raw.reshape(-1, H, W).mean(axis=0)).astype(np.uint8)
        assert np.abs(out.astype(int) - _reference(raw, mean_bg).astype(int)).max() <= 2


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_explicit_reference_pattern_is_that_pattern(lazy):
    from backend.api.services.batch_manager import _apply_preprocessing

    raw = _make_data(0)
    cfg = dict(CFG, static_bg_row=3, static_bg_col=5)
    sig, applied = _apply_preprocessing(_signal(raw, lazy), cfg)
    assert np.array_equal(np.asarray(sig.data), _reference(raw, raw[3, 5]))
    assert applied["background_removal"] == "static@3,5"


@pytest.mark.parametrize("cfg_extra", [{"static_bg_row": 2}, {"static_bg_col": 2},
                                       {"static_bg_row": ROWS, "static_bg_col": 0}])
def test_incomplete_or_out_of_range_reference_is_an_error_and_data_untouched(cfg_extra):
    from backend.api.services.batch_manager import _apply_preprocessing

    raw = _make_data(0)
    sig, applied = _apply_preprocessing(_signal(raw, False), dict(CFG, **cfg_extra))
    assert str(applied["background_removal"]).startswith("FAILED")
    assert np.array_equal(np.asarray(sig.data), raw)


def test_null_reference_in_config_means_scan_average():
    """A client that sends explicit nulls (or an old default of null) = no reference."""
    from backend.api.services.batch_manager import _apply_preprocessing

    raw = _make_data(0)
    _, applied = _apply_preprocessing(
        _signal(raw, False), dict(CFG, static_bg_row=None, static_bg_col=None))
    assert applied["background_removal"] == "static@scan_average"
