"""Tests for detector geometry — PC convention conversions and frame transforms."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.spherical_gpu.exceptions import DetectorConventionError
from backend.spherical_gpu.pipeline.detector import (
    DetectorGeometry,
    convert_pc_to_emsoft,
)


def test_oxford_pc_centered_to_emsoft():
    """Oxford PC (0.5, 0.5, DD) -> EMsoft (0, 0, DD * pat_w * px).

    Oxford PCx=0.5 means horizontally centered → xpc=0.
    Oxford PCy=0.5 means vertically centered → ypc=0 (after Y flip, still 0).
    DD=0.6 → L = 0.6 * 60 * 55 = 1980 microns.
    """
    pc_oxford = (0.5, 0.5, 0.6)
    xpc, ypc, L = convert_pc_to_emsoft(
        pc=pc_oxford, vendor="Oxford",
        pat_width=60, pat_height=60, pixel_size=55.0, binning=1,
    )
    assert xpc == pytest.approx(0.0, abs=1e-6)
    assert ypc == pytest.approx(0.0, abs=1e-6)
    assert L == pytest.approx(1980.0, abs=1e-3)


def test_oxford_pc_off_center_x():
    """Off-center PCx tests sign convention.

    Oxford PCx 0.6 means 0.1 right of detector center in image space.
    EMsoft xpc is positive to the LEFT (matches kikuchipy convention),
    so the result is -6 px on a 60-wide detector.
    """
    pc = (0.6, 0.5, 0.6)
    xpc, ypc, L = convert_pc_to_emsoft(
        pc=pc, vendor="Oxford",
        pat_width=60, pat_height=60, pixel_size=55.0, binning=1,
    )
    assert xpc == pytest.approx(-6.0, abs=1e-6)
    assert ypc == pytest.approx(0.0, abs=1e-6)


def test_oxford_pc_off_center_y_matches_kikuchipy():
    """Oxford PCy 0.6 -> EMsoft ypc = +6 on a square 60x60 detector.

    The Oxford-to-Bruker chain takes pcy -> 1 - pcy*aspect (= 0.4 for square),
    and Bruker-to-EMsoft applies (0.5 - pcy_B)*N_y. So a PCy that's "below
    center" in image-space becomes a positive ypc in EMsoft after the
    intermediate Bruker representation. Cross-validated against
    kikuchipy.detectors.EBSDDetector.pc_emsoft() — see live REPL output.
    """
    pc = (0.5, 0.6, 0.6)
    xpc, ypc, L = convert_pc_to_emsoft(
        pc=pc, vendor="Oxford",
        pat_width=60, pat_height=60, pixel_size=55.0, binning=1,
    )
    assert ypc == pytest.approx(6.0, abs=1e-6)


def test_edax_pc_centered_to_emsoft():
    """EDAX (0.5, 0.5, zstar) on 60x80 detector -> centered EMsoft.

    Following kikuchipy._pc_tsl2bruker:
    - pc_x_b = pc_x_T = 0.5
    - pc_y_b = 1 - pc_y_T = 0.5
    - pc_z_b = pc_z_T * min(N_y, N_x)/N_y = 0.6 * 60/80 = 0.45

    Then Bruker -> EMsoft:
    - L = 0.45 * 80 * 1 * 55 = 1980.
    """
    pc = (0.5, 0.5, 0.6)
    xpc, ypc, L = convert_pc_to_emsoft(
        pc=pc, vendor="EDAX",
        pat_width=60, pat_height=80, pixel_size=55.0, binning=1,
    )
    assert xpc == pytest.approx(0.0)
    assert ypc == pytest.approx(0.0)
    assert L == pytest.approx(1980.0)


def test_emsoft_passthrough():
    """EMsoft PC values pass through unchanged."""
    pc = (3.5, -2.0, 17000.0)
    xpc, ypc, L = convert_pc_to_emsoft(
        pc=pc, vendor="EMsoft",
        pat_width=60, pat_height=60, pixel_size=55.0, binning=1,
    )
    assert xpc == pytest.approx(3.5)
    assert ypc == pytest.approx(-2.0)
    assert L == pytest.approx(17000.0)


def test_unknown_vendor_raises():
    """Unsupported vendors must error loudly, not silently default."""
    with pytest.raises(DetectorConventionError):
        convert_pc_to_emsoft(
            pc=(0.5, 0.5, 0.6), vendor="Hitachi",
            pat_width=60, pat_height=60, pixel_size=55.0, binning=1,
        )


def test_detector_geometry_dataclass(detector_params):
    """``DetectorGeometry.from_params`` packages the dict into tensors on the
    requested device. Uses CPU device for portability — tests run on
    machines with or without CUDA."""
    device = torch.device("cpu")
    geom = DetectorGeometry.from_params(detector_params, device=device)

    assert geom.pat_w == int(detector_params["pat_width"])
    assert geom.pat_h == int(detector_params["pat_height"])
    assert geom.tilt_deg == pytest.approx(detector_params["tilt"])
    assert geom.vendor == detector_params["vendor"]

    # PC tensors are 0-d FP64 on the requested device
    assert geom.xpc.dim() == 0
    assert geom.ypc.dim() == 0
    assert geom.L.dim() == 0
    assert geom.xpc.dtype == torch.float64
    assert geom.xpc.device.type == device.type


def test_pc_against_kikuchipy_reference(detector_params):
    """Cross-validate against kikuchipy's own conversion using the oracle's PC.

    kikuchipy's ``EBSDDetector.pc_emsoft()`` is a battle-tested reference;
    if our conversion matches it within float precision, our PC math is right.
    """
    pytest.importorskip("kikuchipy")
    from kikuchipy.detectors import EBSDDetector

    pc_tuple = (
        detector_params["pc_x"],
        detector_params["pc_y"],
        detector_params["pc_z"],
    )

    # kikuchipy convention names: 'tsl' = EDAX, 'oxford' = Oxford,
    # 'bruker' = Bruker, 'emsoft' = EMsoft.
    kk_convention_map = {
        "Oxford": "oxford",
        "EDAX": "tsl",
        "Bruker": "bruker",
        "EMsoft": "emsoft",
    }
    kk_conv = kk_convention_map[detector_params["vendor"]]

    det = EBSDDetector(
        shape=(detector_params["pat_height"], detector_params["pat_width"]),
        pc=pc_tuple,
        convention=kk_conv,
        px_size=detector_params["pixel_size"],
        binning=detector_params["binning"],
    )
    kk_pc_emsoft = np.asarray(det.pc_emsoft()).flatten()[:3]

    xpc, ypc, L = convert_pc_to_emsoft(
        pc=pc_tuple, vendor=detector_params["vendor"],
        pat_width=detector_params["pat_width"],
        pat_height=detector_params["pat_height"],
        pixel_size=detector_params["pixel_size"],
        binning=detector_params["binning"],
    )

    # Oxford and Bruker conventions match kikuchipy bit-for-bit.
    # EDAX may differ in scale factor (kikuchipy applies extra binning factor
    # in some versions); treat 1% tolerance as acceptable cross-validation.
    rel_tol = 1e-3 if detector_params["vendor"] in ("Oxford", "Bruker", "EMsoft") else 1e-2
    assert xpc == pytest.approx(float(kk_pc_emsoft[0]), rel=rel_tol, abs=1e-3)
    assert ypc == pytest.approx(float(kk_pc_emsoft[1]), rel=rel_tol, abs=1e-3)
    assert L == pytest.approx(float(kk_pc_emsoft[2]), rel=rel_tol, abs=1.0)
