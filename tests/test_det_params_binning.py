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
    stored_pixel_size,
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


def test_stored_pixel_size_helper():
    # a size the app substituted already refers to the stored pixel
    assert stored_pixel_size(70.0, 8, pixel_size_is_stored_pixel=True) == 70.0
    # a size from a file is kikuchipy's unbinned px_size
    assert stored_pixel_size(55.0, 2, pixel_size_is_stored_pixel=False) == 110.0
    assert stored_pixel_size(55.0, 1, pixel_size_is_stored_pixel=False) == 55.0
    assert stored_pixel_size(55.0, 0, pixel_size_is_stored_pixel=False) == 55.0


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


def test_pixel_size_from_the_file_is_folded_into_the_stored_pixel():
    # px_size > 1 is the UNBINNED size; the stored grid is binned, so the
    # pixel size handed to every consumer is px_size * binning and the
    # conversion gets binning 1 - same physical detector as a binning-1 file
    # whose px_size is already the stored-pixel size.
    shape = (128, 156)
    dp2 = build_spherical_det_params(
        _stub_signal(), _stub_detector(2, px_size=55.0, shape=shape), "")
    dp1 = build_spherical_det_params(
        _stub_signal(), _stub_detector(1, px_size=110.0, shape=shape), "")
    assert dp2["pixel_size"] == 110.0
    assert dp2["binning"] == 1
    assert dp2["file_binning"] == 2
    assert _emsoft(dp2) == _emsoft(dp1)
    for k in ("pixel_size", "pat_width", "pat_height", "binning"):
        assert dp2[k] == dp1[k]


def test_folded_pixel_size_goes_through_the_width_check():
    # 8 x 60 um = 480 um per stored pixel, 156 px = 74.9 mm: inside the
    # [5, 90] mm range, so the folded value is kept (no auto-scale).
    dp = build_spherical_det_params(
        _stub_signal(), _stub_detector(8, px_size=60.0), "")
    assert dp["pixel_size"] == 480.0


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
@pytest.mark.parametrize("pixel_size, binning, expected_px", [
    (None, 8, DEFAULT_PIXEL_SIZE_UM),   # default = stored pixel, never binned again
    (None, 1, DEFAULT_PIXEL_SIZE_UM),
    (55.0, 8, 440.0),                   # explicit (unbinned) size folded to the stored pixel
])
def test_render_preview_binning_passed_to_pc_conversion(
        monkeypatch, tmp_path, pixel_size, binning, expected_px):
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
    assert seen["binning"] == 1
    assert seen["pixel_size"] == expected_px


# ---- the detector the UI shows must be the geometry the computation uses -----
def test_detector_display_reports_folded_geometry():
    from backend.api.routes.ebsd_viewer import _build_detector_dict
    d = _build_detector_dict(types.SimpleNamespace(
        shape=(128, 156), pc=np.array([0.5, 0.33, 0.85]), sample_tilt=70.0,
        tilt=4.3, azimuthal=0.0, binning=8, px_size=1.0))
    assert d["binning"] == 1 and d["file_binning"] == 8
    assert d["pixel_size"] is None               # placeholder: no real size known
    d = _build_detector_dict(types.SimpleNamespace(
        shape=(128, 156), pc=np.array([0.5, 0.33, 0.85]), sample_tilt=70.0,
        tilt=4.3, azimuthal=0.0, binning=2, px_size=55.0))
    assert d["binning"] == 1 and d["file_binning"] == 2
    assert d["pixel_size"] == 110.0
    assert "File binning: 2" in d["repr"]


def test_pc_detector_info_matches_the_display_dict(monkeypatch):
    import asyncio
    from backend.api.routes import pcrefinement as pcr
    det = types.SimpleNamespace(
        shape=(128, 156), pc=np.array([[0.5, 0.33, 0.85]]), sample_tilt=70.0,
        tilt=4.3, azimuthal=0.0, binning=8, px_size=1.0)
    monkeypatch.setattr(pcr, "_get_controller",
                        lambda: types.SimpleNamespace(detector=det))
    info = asyncio.run(pcr.detector_info())
    assert info["binning"] == 1 and info["file_binning"] == 8


# ---- exports written before the fold carry the doubled geometry --------------
def _write_geometry(path, dg, source_vendor=None):
    import json
    import h5py
    with h5py.File(path, "w") as f:
        g = f.create_group("Indexing")
        g.attrs["detector_geometry"] = json.dumps(dg)
        if source_vendor is not None:
            g.attrs["source_vendor"] = source_vendor


def _legacy(**over):
    dg = {"pc_x": 0.5, "pc_y": 0.33, "pc_z": 0.85, "pat_width": 156,
          "pat_height": 128, "pixel_size": DEFAULT_PIXEL_SIZE_UM,
          "binning": 8, "tilt": 4.3, "vendor": "Bruker"}
    dg.update(over)
    return dg


@pytest.mark.parametrize("vendor", ["oxford", "edax", "Oxford"])
def test_legacy_export_with_default_pixel_size_is_corrected(tmp_path, caplog, vendor):
    """Oxford and EDAX files carry only the placeholder pixel size, so a 70 um in
    their export IS the app's substitute."""
    from backend.api.routes.indexing import _restore_render_geometry
    p = tmp_path / "old.h5"
    _write_geometry(p, _legacy(), source_vendor=vendor)
    with caplog.at_level("WARNING"):
        dg = _restore_render_geometry(p)["detector_geometry"]
    assert dg["binning"] == 1 and dg["file_binning"] == 8
    assert dg["pixel_size"] == DEFAULT_PIXEL_SIZE_UM
    assert any("corrected" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.parametrize("vendor", ["bruker", "unknown", None, ""])
def test_a_70_um_that_may_be_the_files_own_is_not_corrected(tmp_path, caplog, vendor):
    """A reader that supplied a real pixel size of exactly 70 um (Bruker files do)
    looks identical to the substitute in the export. Only vendors whose files never
    carry a pixel size (Oxford, EDAX) justify the correction; for the rest it is a
    warning, and the geometry is used as stored."""
    from backend.api.routes.indexing import _restore_render_geometry
    p = tmp_path / "old.h5"
    _write_geometry(p, _legacy(), source_vendor=vendor)
    with caplog.at_level("WARNING"):
        dg = _restore_render_geometry(p)["detector_geometry"]
    assert dg["binning"] == 8 and "file_binning" not in dg
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert msgs and not any("corrected" in m.lower() and "not corrected" not in m.lower() for m in msgs)
    assert any("could be the file's own" in m or "cannot tell" in m for m in msgs)


def test_legacy_export_with_other_pixel_size_is_only_warned_about(tmp_path, caplog):
    from backend.api.routes.indexing import _restore_render_geometry
    p = tmp_path / "old.h5"
    _write_geometry(p, _legacy(pixel_size=55.0))
    with caplog.at_level("WARNING"):
        dg = _restore_render_geometry(p)["detector_geometry"]
    assert dg["binning"] == 8 and "file_binning" not in dg
    assert any("binning" in r.getMessage() and r.levelname == "WARNING"
               for r in caplog.records)


@pytest.mark.parametrize("dg", [
    _legacy(binning=1),                          # 0.11.3 export: nothing to do
    _legacy(binning=1, file_binning=8),          # written after the fold
])
def test_current_and_binning_1_exports_are_untouched(tmp_path, caplog, dg):
    from backend.api.routes.indexing import _restore_render_geometry
    p = tmp_path / "x.h5"
    _write_geometry(p, dg)
    with caplog.at_level("WARNING"):
        out = _restore_render_geometry(p)["detector_geometry"]
    assert out == dg
    assert not caplog.records


# ---- EMSphInx: delta is the pixel of the stored pattern ----------------------
@pytest.mark.parametrize("binning, px_size, expected_delta", [
    (1, 1.0, DEFAULT_PIXEL_SIZE_UM),   # placeholder: default, whatever the camera says
    (8, 1.0, DEFAULT_PIXEL_SIZE_UM),
    (2, 55.0, 110.0),                  # file value: unbinned size x binning
])
def test_emsphinx_nml_delta_is_the_stored_pixel_size(
        tmp_path, binning, px_size, expected_delta):
    import re
    from pathlib import Path
    from indexing_controller import IndexingConfig, generate_emsphinx_nml
    dp = build_spherical_det_params(
        _stub_signal(), _stub_detector(binning, px_size=px_size), "")
    nml = generate_emsphinx_nml(
        IndexingConfig(sht_file=str(tmp_path / "m.sht")),
        str(tmp_path / "absent.h5oina"), dp, Path(tmp_path))
    delta = float(re.search(r"delta\s*=\s*([0-9.eE+-]+)",
                            nml.read_text(encoding="utf-8")).group(1))
    assert delta == expected_delta
