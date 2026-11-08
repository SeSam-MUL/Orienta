"""Calibration patterns of a circular-aperture detector are all shown round.

On the PC Refinement page the first selected pattern was
round (black corners outside the EDAX aperture) and later ones were square with
noisy corners. The later ones had been added after background removal in the
viewer, which turns the black corners into noise, and the page decided "round or
not" for each pattern from its own corners.

The aperture belongs to the detector, so it is decided once per session from the
pattern as recorded (the unprocessed dataset, when the viewer still holds it) and
every stored pattern is shown through the same disc. Display only: the stored
patterns themselves are not changed.
"""
import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

SHAPE = (60, 60)


def _disc_pattern(seed=0):
    """A recorded EDAX-style pattern: signal inside the disc, zeros outside."""
    from tools.pattern_comparison import circular_mask
    rng = np.random.default_rng(seed)
    pat = rng.integers(80, 200, size=SHAPE).astype(np.uint8)
    pat[~circular_mask(SHAPE)] = 0
    return pat


def _processed(pat, seed=1):
    """What background removal leaves: rescaled back to the full grey range, so the
    corners outside the aperture become mid-grey noise (not black, not negative)."""
    rng = np.random.default_rng(seed)
    out = pat.astype(np.float32) - pat.mean() + rng.normal(0, 25, size=pat.shape).astype(np.float32)
    out = (out - out.min()) / (out.max() - out.min()) * 255
    return out.astype(np.float32)


def test_the_processed_stand_in_really_has_lost_its_black_corners():
    # Otherwise the cases below could pass on the processed pattern alone.
    from tools.pattern_comparison import detect_circular_aperture
    assert detect_circular_aperture(_disc_pattern()) is True
    assert detect_circular_aperture(_processed(_disc_pattern())) is False


def _png(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


@pytest.fixture
def client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import backend.api.routes.pcrefinement as pcr
    from backend.api.routes import ebsd_viewer as ev
    monkeypatch.setattr(pcr, "_sessions", {})
    monkeypatch.setattr(pcr, "_parked_sessions", {})
    monkeypatch.setattr(ev, "_active_dataset", None)
    app = FastAPI()
    app.include_router(pcr.router, prefix="/api/pc")
    return TestClient(app), pcr


def _corner_max(img):
    b = 6
    return max(int(img[:b, :b].max()), int(img[:b, -b:].max()), int(img[-b:, :b].max()), int(img[-b:, -b:].max()))


def test_a_processed_pattern_is_shown_through_the_session_aperture(client):
    c, pcr = client
    ctrl = pcr._get_controller()
    raw = _disc_pattern()
    ctrl.add_pattern((1, 1), raw)                 # added before processing: black corners
    ctrl.add_pattern((2, 2), _processed(raw))     # added after BG removal: noisy corners

    first = c.get("/api/pc/pattern/0/image").json()
    later = c.get("/api/pc/pattern/1/image").json()

    assert first["circular"] is True and later["circular"] is True
    assert _corner_max(_png(later["image"])) == 0
    assert _png(later["image"]).max() == 255      # contrast set inside the disc


def test_contrast_is_set_inside_the_disc_not_by_the_noise_outside(client):
    from tools.pattern_comparison import circular_mask
    c, pcr = client
    ctrl = pcr._get_controller()
    raw = _disc_pattern()
    ctrl.add_pattern((1, 1), raw)
    proc = _processed(raw, seed=7)
    inside = circular_mask(SHAPE)
    proc[inside] = np.clip(proc[inside], 0, 150)   # the pattern is dim ...
    proc[:4, :4] = 255                              # ... the brightest pixels sit outside
    ctrl.add_pattern((2, 2), proc)
    img = _png(c.get("/api/pc/pattern/1/image").json()["image"])
    assert img[inside].max() == 255                 # stretched over the disc


def test_the_stored_pattern_itself_is_not_changed(client):
    c, pcr = client
    ctrl = pcr._get_controller()
    raw = _disc_pattern()
    proc = _processed(raw)
    ctrl.add_pattern((1, 1), raw)
    ctrl.add_pattern((2, 2), proc.copy())
    c.get("/api/pc/pattern/1/image")
    np.testing.assert_array_equal(ctrl.patterns[1][1], proc)


def test_signed_processed_data_is_no_evidence_of_an_aperture(client):
    """A background-subtracted float pattern of a FULL-FRAME detector has negative
    corners; they must not make the session "circular"."""
    c, pcr = client
    ctrl = pcr._get_controller()
    # A full-frame pattern with the usual vignetting: signal everywhere, the
    # centre brighter than the corners. Background-subtracted to zero mean, its
    # corners are clearly negative - every corner block below 5 % of the max.
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    r2 = (yy - SHAPE[0] / 2) ** 2 + (xx - SHAPE[1] / 2) ** 2
    full = (120 + 80 * np.exp(-r2 / (2 * (SHAPE[0] / 4) ** 2))).astype(np.float32)
    signed = full - full.mean()
    from tools.pattern_comparison import detect_circular_aperture
    assert detect_circular_aperture(signed)         # the trap: the raw check is fooled
    ctrl.add_pattern((0, 0), signed)
    body = c.get("/api/pc/pattern/0/image").json()
    assert body["circular"] is False
    assert _corner_max(_png(body["image"])) > 0


def test_a_full_frame_detector_stays_square(client):
    c, pcr = client
    ctrl = pcr._get_controller()
    pat = np.random.default_rng(3).integers(60, 220, size=SHAPE).astype(np.uint8)
    ctrl.add_pattern((0, 0), pat)
    body = c.get("/api/pc/pattern/0/image").json()
    assert body["circular"] is False
    assert _corner_max(_png(body["image"])) > 0


def test_add_decides_the_aperture_from_the_unprocessed_parent(monkeypatch, client):
    """All patterns added after processing: the parent dataset still holds the
    recorded patterns, and the aperture is read from there."""
    import kikuchipy as kp
    from backend.api.routes import ebsd_viewer as ev
    c, pcr = client

    raw = np.stack([np.stack([_disc_pattern(r * 3 + k) for k in range(3)]) for r in range(3)])
    proc = np.stack([np.stack([_processed(raw[r, k], r * 3 + k) for k in range(3)]) for r in range(3)])
    parent, child = kp.signals.EBSD(raw), kp.signals.EBSD(proc)

    monkeypatch.setattr(ev, "_raw_signals", {"scan": parent, "scan_bg_dyn": child})
    monkeypatch.setattr(ev, "_dirty_datasets", {"scan_bg_dyn"})
    monkeypatch.setattr(ev, "_active_dataset", "scan_bg_dyn")
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: child)
    monkeypatch.setattr(pcr, "_auto_attach_detector", lambda ctrl: None)
    entries = {"scan_bg_dyn": SimpleNamespace(parent_name="scan"), "scan": SimpleNamespace(parent_name=None)}
    monkeypatch.setattr(pcr.calibration_store, "get_entry", lambda name: entries.get(name))

    r = c.post("/api/pc/pattern/add", json={"row": 1, "col": 2})
    assert r.status_code == 200, r.text
    ctrl = pcr._get_controller()
    assert ctrl.aperture_circular is True
    body = c.get("/api/pc/pattern/0/image").json()
    assert body["circular"] is True
    assert _corner_max(_png(body["image"])) == 0


def test_a_recorded_scan_with_negative_values_still_shows_its_aperture(monkeypatch, client):
    """An int16 scan is recorded data, and its black corners sit at the bottom of a range that
    starts below zero. Read as the map run reads it they are 0; read raw they are negative, and
    the check takes that for processed data and leaves the session square."""
    import kikuchipy as kp
    from backend.api.routes import ebsd_viewer as ev
    c, pcr = client

    # corners 0 * 16 - 1000 = -1000, the disc at least 80 * 16 - 1000 = 280
    raw = np.stack([np.stack([_disc_pattern(r * 3 + k) for k in range(3)]) for r in range(3)])
    scan = kp.signals.EBSD(raw.astype(np.int16) * 16 - 1000)
    assert scan.data.min() == -1000

    monkeypatch.setattr(ev, "_raw_signals", {"scan": scan})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_active_dataset", "scan")
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: scan)
    monkeypatch.setattr(pcr, "_auto_attach_detector", lambda ctrl: None)
    monkeypatch.setattr(pcr.calibration_store, "get_entry", lambda name: None)

    r = c.post("/api/pc/pattern/add", json={"row": 1, "col": 2})
    assert r.status_code == 200, r.text
    assert pcr._get_controller().aperture_circular is True


def test_unprocessed_pattern_stops_at_a_different_grid(monkeypatch):
    """A crop re-indexes the grid: the parent's (row, col) is another pixel."""
    import kikuchipy as kp
    from backend.api.routes import ebsd_viewer as ev
    import backend.api.routes.pcrefinement as pcr
    parent = kp.signals.EBSD(np.zeros((4, 4) + SHAPE, dtype=np.uint8))
    crop = kp.signals.EBSD(np.ones((2, 2) + SHAPE, dtype=np.float32))
    monkeypatch.setattr(ev, "_raw_signals", {"scan": parent, "scan_crop": crop})
    monkeypatch.setattr(ev, "_dirty_datasets", {"scan_crop"})
    monkeypatch.setattr(pcr.calibration_store, "get_entry",
                        lambda name: SimpleNamespace(parent_name="scan" if name == "scan_crop" else None))
    assert pcr._unprocessed_pattern("scan_crop", 1, 1, SHAPE) is None
