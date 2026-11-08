"""The forward preview reports an NCC only for an orientation that was found.

When no orientation can be indexed at the trial geometry the preview falls back to
the identity orientation. A pattern rendered from it matches the experimental one
by chance, so its NCC says nothing about the geometry - and it was reported as if
it did (a low number next to a possibly good PC). Now there is no number and a
reason; a supplied orientation keeps its NCC.
"""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from pc_controller import PCController

SHAPE = (60, 60)


def _png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def preview(monkeypatch, tmp_path):
    from backend.api.routes import pcrefinement as R
    from backend.api.services import sht_pattern_renderer

    rng = np.random.default_rng(1)
    exp = rng.integers(0, 255, SHAPE).astype(np.uint8)
    ctrl = PCController()                       # no phase loaded: Hough cannot run
    ctrl.add_pattern((0, 0), exp)
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    # the renderer needs a GPU and a master pattern; its picture is not what is tested.
    # It returns the experimental pattern itself, so a computed NCC would be 1.0.
    monkeypatch.setattr(sht_pattern_renderer, "render_pattern_to_png_b64",
                        lambda **kw: _png_b64(exp))
    sht = tmp_path / "Ni.sht"
    sht.write_bytes(b"x")

    def run(**kw):
        req = R.RenderPreviewRequest(pattern_idx=0, sht_path=str(sht), pc=[0.5, 0.4, 0.6],
                                     sample_tilt=70.0, **kw)
        return R._render_preview_sync(req)
    return run


def test_the_fallback_orientation_gets_no_ncc(preview):
    res = preview()
    assert res["orientation_source"] == "identity"
    assert res["ncc"] is None
    assert res["ncc_reason"] == "no_orientation"


def test_a_supplied_orientation_keeps_its_ncc(preview):
    res = preview(orientation_euler_deg=[10.0, 20.0, 30.0])
    assert res["orientation_source"] == "supplied"
    assert res["ncc"] == pytest.approx(1.0, abs=1e-6)       # the stub renders the pattern itself
    assert res["ncc_reason"] is None


# ----- the aperture: the detector's, decided once per session ------------------------------------

SIM_CORNER = 200                  # the stub renders a full frame: every corner pixel is 200


@pytest.fixture
def aperture_preview(monkeypatch, tmp_path):
    """run(stored, circular_flag) -> the preview of ``stored`` as the one pattern of a session."""
    from backend.api.routes import pcrefinement as R
    from backend.api.services import sht_pattern_renderer

    sim = np.full(SHAPE, SIM_CORNER, np.uint8)
    sim[20:40, 20:40] = 90                                  # something inside, to tell the picture apart
    monkeypatch.setattr(sht_pattern_renderer, "render_pattern_to_png_b64", lambda **kw: _png_b64(sim))
    sht = tmp_path / "Ni.sht"
    sht.write_bytes(b"x")

    def run(stored, circular_flag=None):
        ctrl = PCController()
        ctrl.add_pattern((0, 0), stored)
        if circular_flag is not None:
            ctrl.aperture_circular = circular_flag
        monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
        req = R.RenderPreviewRequest(pattern_idx=0, sht_path=str(sht), pc=[0.5, 0.4, 0.6],
                                     sample_tilt=70.0, orientation_euler_deg=[10.0, 20.0, 30.0])
        return R._render_preview_sync(req)
    return run


def _sim_png(res):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(res["simulated_b64"]))))


def _vignetted_signed():
    """A full-frame detector's pattern after background subtraction: the centre brighter than the
    corners, zero mean, so every corner block is far below 5 % of the maximum."""
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    r2 = (yy - SHAPE[0] / 2) ** 2 + (xx - SHAPE[1] / 2) ** 2
    full = (120 + 80 * np.exp(-r2 / (2 * (SHAPE[0] / 4) ** 2))).astype(np.float32)
    return full - full.mean()


def _disc():
    from tools.pattern_comparison import circular_mask
    pat = np.random.default_rng(0).integers(80, 200, size=SHAPE).astype(np.uint8)
    pat[~circular_mask(SHAPE)] = 0
    return pat


def test_a_signed_pattern_of_a_full_frame_detector_is_not_masked(aperture_preview):
    signed = _vignetted_signed()
    from tools.pattern_comparison import detect_circular_aperture
    assert detect_circular_aperture(signed)            # the trap: the raw check takes it for a disc
    sim = _sim_png(aperture_preview(signed))
    assert sim[0, 0] == sim[0, -1] == sim[-1, 0] == sim[-1, -1] == SIM_CORNER


def test_a_circular_session_masks_the_preview_of_its_signed_pattern(aperture_preview):
    sim = _sim_png(aperture_preview(_vignetted_signed(), circular_flag=True))
    assert sim[0, 0] == sim[0, -1] == sim[-1, 0] == sim[-1, -1] == 0
    assert sim[30, 30] == 90                           # inside the disc the render is untouched


def test_a_recorded_circular_pattern_is_masked_without_a_session_flag(aperture_preview):
    sim = _sim_png(aperture_preview(_disc()))
    assert sim[0, 0] == sim[-1, -1] == 0 and sim[30, 30] == 90
