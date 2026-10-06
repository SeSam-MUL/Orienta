"""EBSD Viewer "BG Static": the reference must be the scan average, not pattern (0,0).

The viewer's "BG Static" button (and the first step of the "Recommended
Pipeline") posts ``{"method": "static"}`` without naming a reference pattern.
The route used to fill in ``static_bg_row = static_bg_col = 0`` and subtract
the pattern at the scan origin from every pattern, so the bands of that one
pattern appeared, inverted, in the whole map.  The documented behaviour is the
scan-average background, which is also what kikuchipy recommends when the file
carries no usable static background.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ROWS, COLS, H, W = 5, 7, 40, 48  # deliberately not square, both in map and detector


def _make_data(seed: int = 0) -> np.ndarray:
    """Patterns = shared smooth illumination + a few bands per pattern.

    Every pattern gets its own random band orientations, so the scan average
    contains the illumination but (almost) none of the bands, like a real map
    over many grains.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    illum = 110.0 + 20.0 * np.exp(-(((yy - H * 0.7) / H) ** 2 + ((xx - W / 2) / W) ** 2) * 4)
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


def _install(signal, name="bg_static_test"):
    from backend.api.routes import ebsd_viewer

    ebsd_viewer._loaded_files.clear()
    ebsd_viewer._raw_signals.clear()
    ebsd_viewer._positions.clear()
    ebsd_viewer._dirty_datasets.clear()
    ebsd_viewer._signal_masks.clear()
    ebsd_viewer._raw_signals[name] = signal
    ebsd_viewer._positions[name] = (0, 0)
    ebsd_viewer._active_dataset = name
    ebsd_viewer._ebsd_signal = signal
    app = FastAPI()
    app.include_router(ebsd_viewer.router, prefix="/api/ebsd")
    return TestClient(app)


@pytest.fixture(params=["eager", "lazy"])
def setup(request):
    import kikuchipy as kp

    data = _make_data()
    sig = kp.signals.EBSD(data.copy())
    if request.param == "lazy":
        sig = sig.as_lazy()
    client = _install(sig)
    return client, sig, data


def _processed(sig) -> np.ndarray:
    return np.asarray(sig.data)


def test_static_without_reference_does_not_imprint_pattern_00(setup):
    client, sig, raw = setup
    r = client.post("/api/ebsd/background-removal", json={"method": "static"})
    assert r.status_code == 200, r.text

    out = _processed(sig)
    others = [(i, j) for i in range(ROWS) for j in range(COLS) if (i, j) != (0, 0)]
    ghost = np.mean([_cc(out[p], raw[0, 0]) for p in others])
    baseline = np.mean([_cc(raw[p], raw[0, 0]) for p in others])
    # Before the fix the processed patterns carried the inverted bands of
    # pattern (0,0): strongly negative correlation with it (about -0.7 on real
    # data). Unrelated patterns correlate around zero.
    # Processing should not make the patterns MORE anti-correlated with (0,0)
    # than they were raw (illumination shared by all patterns correlates them
    # positively in the raw data).
    assert ghost > baseline - 0.3, (
        f"pattern (0,0) is imprinted on the map (mean cc {ghost:+.3f}, raw {baseline:+.3f})"
    )


def test_static_without_reference_equals_scan_average_subtraction(setup):
    import kikuchipy as kp

    client, sig, raw = setup
    client.post("/api/ebsd/background-removal", json={"method": "static"})

    ref = kp.signals.EBSD(raw.copy())
    mean_bg = np.rint(raw.reshape(-1, H, W).mean(axis=0)).astype(np.uint8)
    ref.remove_static_background(operation="subtract", static_bg=mean_bg, show_progressbar=False)

    diff = np.abs(_processed(sig).astype(int) - ref.data.astype(int))
    assert diff.max() <= 2, f"not the scan-average result (max grey-level diff {diff.max()})"


def test_static_response_names_the_reference_used(setup):
    client, _, _ = setup
    body = client.post("/api/ebsd/background-removal", json={"method": "static"}).json()
    assert body["static_reference"] == "scan_average"


def test_static_with_explicit_reference_pattern_is_unchanged(setup):
    """The API keeps accepting an explicit reference pattern."""
    import kikuchipy as kp

    client, sig, raw = setup
    body = client.post(
        "/api/ebsd/background-removal",
        json={"method": "static", "static_bg_row": 3, "static_bg_col": 5},
    ).json()
    assert body["static_reference"] == "pattern(3,5)"

    ref = kp.signals.EBSD(raw.copy())
    ref.remove_static_background(operation="subtract", static_bg=raw[3, 5], show_progressbar=False)
    assert np.array_equal(_processed(sig), ref.data)


def test_static_reference_must_name_both_row_and_col(setup):
    client, sig, raw = setup
    r = client.post("/api/ebsd/background-removal", json={"method": "static", "static_bg_row": 2})
    assert r.status_code == 400
    assert np.array_equal(_processed(sig), raw), "a rejected request must not touch the data"


def test_static_reference_outside_scan_is_rejected(setup):
    client, _, _ = setup
    r = client.post(
        "/api/ebsd/background-removal",
        json={"method": "static", "static_bg_row": ROWS, "static_bg_col": 0},
    )
    assert r.status_code == 400


def test_dynamic_is_per_pattern_and_independent_of_other_patterns(setup):
    """BG Dynamic filters each pattern on its own: no other pattern leaks in."""
    import kikuchipy as kp

    client, sig, raw = setup
    r = client.post("/api/ebsd/background-removal", json={"method": "dynamic"})
    assert r.status_code == 200, r.text

    ref = kp.signals.EBSD(raw.copy())
    ref.remove_dynamic_background(operation="subtract", filter_domain="frequency", show_progressbar=False)
    assert np.array_equal(_processed(sig), ref.data)

    # Same pattern processed alone gives the same result as inside the map.
    alone = kp.signals.EBSD(raw[2:3, 4:5].copy())
    alone.remove_dynamic_background(operation="subtract", filter_domain="frequency", show_progressbar=False)
    assert np.array_equal(_processed(sig)[2, 4], alone.data[0, 0])
