"""Convert detector pixel grid + projection center to 3D directions.

Vectorized port of kikuchipy's
``kikuchipy.signals.util._master_pattern._get_direction_cosines_for_fixed_pc``
(adapted from EMsoft).

Conventions
-----------
- PC is given in the Bruker / kikuchipy convention: ``(pcx, pcy, pcz)``,
  fractional coordinates in [0, 1] of the detector (pcx, pcy) and
  ``pcz = DD / nrows_in_detector_height_units`` (kikuchipy / EMsoft v5
  uses ``zpc = nrows * pcz`` internally).
- ``sample_tilt`` is the sample tilt from horizontal in degrees
  (default 70 deg). Detector tilt ``tilt`` and ``azimuthal`` default to
  0, matching kikuchipy's ``EBSDDetector(...)`` defaults.
- Output directions live in the *sample frame* (the same frame kikuchipy
  feeds into ``rotate_vector`` before sampling the master pattern), so
  applying ``R . d_sample`` after this lands in the crystal frame.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def detector_pixel_directions(
    shape: Tuple[int, int],
    pc: Tuple[float, float, float],
    tilt_deg: float = 70.0,
    detector_tilt_deg: float = 0.0,
    azimuthal_deg: float = 0.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return (H*W, 3) unit-length direction cosines in the sample frame.

    Mirrors kikuchipy's ``_get_direction_cosines_for_fixed_pc`` exactly,
    so the output matches what ``kp.signals.EBSDMasterPattern.get_patterns``
    feeds into its rotation step.

    Parameters
    ----------
    shape
        ``(nrows, ncols)`` detector shape.
    pc
        ``(pcx, pcy, pcz)`` Bruker / kikuchipy PC.
    tilt_deg
        Sample tilt from horizontal in degrees. Defaults to 70 deg.
    detector_tilt_deg
        Detector tilt from horizontal in degrees. Defaults to 0 (matches
        ``EBSDDetector(...)``).
    azimuthal_deg
        Sample tilt about RD in degrees. Defaults to 0.
    """
    H, W = shape
    nrows, ncols = H, W
    pcx, pcy, pcz = pc

    # kikuchipy: Bruker -> EMsoft v5 PC (in pixel units, not normalized)
    xpc = ncols * (0.5 - pcx)
    ypc = nrows * (0.5 - pcy)
    zpc = nrows * pcz

    # det_x, det_y (1D) in pixel-centred coordinates.
    # kikuchipy uses `nrows_array = arange(nrows - 1, -1, -1)` (reversed),
    #            and `det_y = ypc - (1 - nrows)*0.5 - nrows_array`.
    nrows_array = torch.arange(nrows - 1, -1, -1, device=device, dtype=dtype)
    ncols_array = torch.arange(ncols, device=device, dtype=dtype)

    det_x = xpc + (1 - ncols) * 0.5 + ncols_array            # (W,)
    det_y = ypc - (1 - nrows) * 0.5 - nrows_array            # (H,)

    # Combined sample + detector tilt + azimuthal:
    # alpha = pi/2 - sample_tilt + detector_tilt
    alpha = math.pi / 2.0 - math.radians(tilt_deg) + math.radians(detector_tilt_deg)
    azimuthal = math.radians(azimuthal_deg)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cw = math.cos(azimuthal)
    sw = math.sin(azimuthal)

    Ls = -sw * det_x + zpc * cw                              # (W,)
    Lc = cw * det_x + zpc * sw                               # (W,)

    # Build (H, W, 3): r[i,j,0] = det_y[i]*ca + sa*Ls[j], r[i,j,1] = Lc[j],
    # r[i,j,2] = -sa*det_y[i] + ca*Ls[j].
    det_y_col = det_y.unsqueeze(1)                           # (H, 1)
    Ls_row = Ls.unsqueeze(0)                                 # (1, W)
    Lc_row = Lc.unsqueeze(0).expand(H, -1)                   # (H, W)

    rx = det_y_col * ca + sa * Ls_row                        # (H, W)
    ry = Lc_row                                              # (H, W)
    rz = -sa * det_y_col + ca * Ls_row                       # (H, W)

    r = torch.stack((rx, ry, rz), dim=-1).reshape(-1, 3)     # (H*W, 3)
    norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    return r / norm
