"""Detector geometry must not depend on how kikuchipy reports Oxford binning.

kikuchipy 0.11.3 reports ``binning = 1`` for Oxford H5OINA files, kikuchipy
>= 0.12 reports the true factor (8 for a "Speed 2 (156x128 px)" camera mode).
The app substitutes a pixel size of ``DEFAULT_PIXEL_SIZE_UM`` because the file
carries only the placeholder 1.0. That substitute is the size of the pixels of
the stored (already binned) patterns, so the binning must not multiply it a
second time in ``convert_pc_to_emsoft``. Both kikuchipy versions have to end
up with the same detector distance.
"""
import os
import types

import numpy as np
import pytest

from backend.api.routes.indexing import build_spherical_det_params
from backend.spherical_gpu.pipeline.detector import (
    DEFAULT_PIXEL_SIZE_UM,
    DetectorGeometry,
    convert_pc_to_emsoft,
    pc_conversion_binning,
)


def _stub_signal(shape=(156, 128)):
    am = types.SimpleNamespace(
        signal_shape=shape,
        navigation_shape=(4, 3),
        navigation_axes=[types.SimpleNamespace(scale=0.5),
                         types.SimpleNamespace(scale=0.5)],
    )
    return types.SimpleNamespace(axes_manager=am)


def _stub_detector(binning, px_size=1.0, shape=(128, 156)):
    return types.SimpleNamespace(
        pc=np.array([[0.5026, 0.3273, 0.8458]]),
        shape=shape, px_size=px_size, tilt=4.3, sample_tilt=70.0,
        binning=binning,
    )


def _emsoft(dp):
    return convert_pc_to_emsoft(
        (dp["pc_x"], dp["pc_y"], dp["pc_z"]), dp["vendor"],
        dp["pat_width"], dp["pat_height"], dp["pixel_size"], dp["binning"],
    )


def test_pc_conversion_binning_helper():
    assert pc_conversion_binning(8, pixel_size_is_stored_pixel=True) == 1
    assert pc_conversion_binning(8, pixel_size_is_stored_pixel=False) == 8
    assert pc_conversion_binning(1, pixel_size_is_stored_pixel=False) == 1
    assert pc_conversion_binning(0, pixel_size_is_stored_pixel=False) == 1


def test_substituted_pixel_size_gives_same_geometry_for_binning_1_and_8():
    dp1 = build_spherical_det_params(_stub_signal(), _stub_detector(1), "")
    dp8 = build_spherical_det_params(_stub_signal(), _stub_detector(8), "")
    assert dp1["pixel_size"] == dp8["pixel_size"] == DEFAULT_PIXEL_SIZE_UM
    assert _emsoft(dp8) == _emsoft(dp1)          # bit-identical, not approx
    assert dp8["binning"] == 1                   # fold: pixel_size is the stored pixel
    assert dp8["file_binning"] == 8              # the file's value stays visible
    assert dp1["file_binning"] == 1


def test_substituted_geometry_is_plausible_for_binning_8():
    dp = build_spherical_det_params(_stub_signal(), _stub_detector(8), "")
    xpc, ypc, L = _emsoft(dp)
    width_mm = dp["pixel_size"] * dp["pat_width"] / 1000.0
    assert 5.0 <= width_mm <= 90.0
    assert 5_000.0 < L < 20_000.0, L             # um; 60 000 would be the double count


def test_autoscaled_pixel_size_is_also_a_stored_pixel_size():
    # 60 px wide -> 70 um gives 4.2 mm, below EMSphInx' range -> autoscaled.
    sig = _stub_signal(shape=(60, 60))
    d1 = build_spherical_det_params(sig, _stub_detector(1, shape=(60, 60)), "")
    d8 = build_spherical_det_params(sig, _stub_detector(8, shape=(60, 60)), "")
    assert d1["pixel_size"] == d8["pixel_size"] == 250.0
    assert _emsoft(d8) == _emsoft(d1)


def test_pixel_size_from_the_file_keeps_its_binning():
    # A file that really carries a pixel size keeps today's behaviour.
    dp = build_spherical_det_params(
        _stub_signal(), _stub_detector(2, px_size=55.0), "")
    assert dp["pixel_size"] == 55.0
    assert dp["binning"] == 2
    assert dp["file_binning"] == 2


def test_geometry_object_sees_folded_binning():
    torch = pytest.importorskip("torch")
    dp = build_spherical_det_params(_stub_signal(), _stub_detector(8), "")
    dp1 = build_spherical_det_params(_stub_signal(), _stub_detector(1), "")
    g = DetectorGeometry.from_params(dp, torch.device("cpu"))
    g1 = DetectorGeometry.from_params(dp1, torch.device("cpu"))
    assert float(g.L) == float(g1.L)
    assert float(g.xpc) == float(g1.xpc)


# ---- the app's own geometry function on the real Oxford file -----------------
SAMPLEB = ("EBSD_SampleB_extrusion_withPattern Sample_B "
           "Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina")

# What the app computed under kikuchipy 0.11.3 (binning reported as 1) before
# this change; stored results were indexed with exactly this geometry.
SAMPLEB_L_UM_0113 = 7578.707629164061
SAMPLEB_XPC_0113 = -0.4088957065343912
SAMPLEB_YPC_0113 = 22.110701713164644


def test_real_sampleb_geometry_is_independent_of_kikuchipy_version(
        test_data_dir):
    path = os.path.join(str(test_data_dir), SAMPLEB)
    if not os.path.isfile(path):
        pytest.skip("SampleB test file not present")
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(path)
    dp = build_spherical_det_params(sig, sig.detector, path)
    xpc, ypc, L = _emsoft(dp)
    assert dp["pixel_size"] == DEFAULT_PIXEL_SIZE_UM
    assert (xpc, ypc, L) == pytest.approx(
        (SAMPLEB_XPC_0113, SAMPLEB_YPC_0113, SAMPLEB_L_UM_0113), rel=1e-12)
    # unbinned 1244 x 1024 sensor -> active width 10.9 mm: a real detector.
    assert dp["pixel_size"] * dp["pat_width"] / 1000.0 == pytest.approx(10.92)


# ---- PC-refinement forward-sim preview --------------------------------------
@pytest.mark.parametrize("pixel_size, binning, expected", [
    (None, 8, 1),     # default pixel size = stored pixel -> never binned again
    (None, 1, 1),
    (55.0, 8, 8),     # an explicit (unbinned) pixel size keeps its binning
])
def test_render_preview_binning_passed_to_pc_conversion(
        monkeypatch, tmp_path, pixel_size, binning, expected):
    from fastapi import HTTPException
    from backend.api.routes import pcrefinement as pcr
    import backend.spherical_gpu.pipeline.detector as det_mod

    sht = tmp_path / "x.sht"
    sht.write_bytes(b"x")                       # only is_file() is checked
    ctrl = types.SimpleNamespace(
        patterns=[((0, 0), np.ones((128, 156), dtype=np.uint8))],
        phase_list=None, reflectors=None, nBands=9,
    )
    monkeypatch.setattr(pcr, "_get_controller", lambda: ctrl)
    seen = {}

    def _record(**kw):
        seen.update(kw)
        raise RuntimeError("stop after the conversion")

    monkeypatch.setattr(det_mod, "convert_pc_to_emsoft", _record)
    req = pcr.RenderPreviewRequest(
        pattern_idx=0, sht_path=str(sht), pc=[0.5, 0.33, 0.85],
        sample_tilt=70.0, binning=binning, pixel_size=pixel_size,
        orientation_euler_deg=[0.0, 0.0, 0.0],
    )
    with pytest.raises(HTTPException):
        pcr._render_preview_sync(req)
    assert seen["binning"] == expected
