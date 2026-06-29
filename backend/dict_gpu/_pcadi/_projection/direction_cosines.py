"""Detector-pixel -> 3D direction in sample frame.

Translation of kikuchipy._get_direction_cosines_for_fixed_pc (lines 149-235
of kikuchipy/signals/util/_master_pattern.py), vectorised in PyTorch.

Critical conventions (see tasks/kikuchipy_projection_reference.md):
  - xpc = ncols * (0.5 - pcx)
  - ypc = nrows * (0.5 - pcy)   <-- uses nrows
  - zpc = nrows * pcz           <-- uses nrows
  - nrows_array runs from nrows-1 down to 0 (reversed)
  - det_x = xpc + (1 - ncols) * 0.5 + ncols_array
  - det_y = ypc - (1 - nrows) * 0.5 - nrows_array
  - Ls = -sw * det_x + zpc * cw
  - Lc =  cw * det_x + zpc * sw
  - r[row,col,0] = det_y[row] * ca + sa * Ls[col]
  - r[row,col,1] = Lc[col]
  - r[row,col,2] = -sa * det_y[row] + ca * Ls[col]
  - alpha = pi/2 - sample_tilt_rad + tilt_rad
"""
from __future__ import annotations
import math
import torch


def compute_direction_cosines(detector, *, device="cuda", dtype=torch.float32) -> torch.Tensor:
    """Direction-cosine grid as (det_h, det_w, 3) unit vectors on `device`.

    Mirrors kikuchipy's `_get_direction_cosines_for_fixed_pc` element-wise,
    but returned as a 2D pixel grid rather than a flattened 1D vector.

    Parameters
    ----------
    detector
        A `kikuchipy.detectors.EBSDDetector`-like object with attributes
        `shape` (nrows, ncols), `pcx[0]`, `pcy[0]`, `pcz[0]`, `tilt`,
        `azimuthal`, `sample_tilt` (all degrees for tilts).
    device
        Torch device string.
    dtype
        Output dtype. Internal math is performed in this dtype; tests
        currently use float32.

    Returns
    -------
    torch.Tensor
        Shape `(nrows, ncols, 3)`, unit-normalised in the sample frame.
    """
    nrows, ncols = int(detector.shape[0]), int(detector.shape[1])
    pcx = float(detector.pcx[0])
    pcy = float(detector.pcy[0])
    pcz = float(detector.pcz[0])
    tilt_deg = float(detector.tilt)
    azimuthal_deg = float(detector.azimuthal)
    sample_tilt_deg = float(detector.sample_tilt)

    # kikuchipy._get_cosine_sine_of_alpha_and_azimuthal
    alpha = math.pi / 2.0 - math.radians(sample_tilt_deg) + math.radians(tilt_deg)
    azimuthal = math.radians(azimuthal_deg)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cw = math.cos(azimuthal)
    sw = math.sin(azimuthal)

    # Bruker -> EMsoft v5 PC convention (asymmetric: ypc and zpc both use nrows)
    xpc = ncols * (0.5 - pcx)
    ypc = nrows * (0.5 - pcy)
    zpc = nrows * pcz

    # nrows_array = arange(nrows-1, -1, -1) — reversed
    # ncols_array = arange(ncols)
    nrows_array = torch.arange(nrows - 1, -1, -1, device=device, dtype=dtype)
    ncols_array = torch.arange(ncols, device=device, dtype=dtype)

    # det_x: shape (ncols,)
    det_x = xpc + (1.0 - ncols) * 0.5 + ncols_array
    # det_y: shape (nrows,)
    det_y = ypc - (1.0 - nrows) * 0.5 - nrows_array

    # Ls, Lc: shape (ncols,)
    Ls = -sw * det_x + zpc * cw
    Lc = cw * det_x + zpc * sw

    # Build (nrows, ncols) components.
    # det_y depends only on row -> shape (nrows, 1)
    # Ls/Lc depend only on col -> shape (1, ncols)
    det_y_col = det_y.unsqueeze(1)  # (nrows, 1)
    Ls_row = Ls.unsqueeze(0)         # (1, ncols)
    Lc_row = Lc.unsqueeze(0)         # (1, ncols)

    r0 = det_y_col * ca + sa * Ls_row                 # (nrows, ncols)
    r1 = Lc_row.expand(nrows, ncols)                  # (nrows, ncols)
    r2 = -sa * det_y_col + ca * Ls_row                # (nrows, ncols)

    r = torch.stack([r0, r1, r2], dim=-1)             # (nrows, ncols, 3)

    # Normalise to unit vectors (kikuchipy normalises at the end)
    norm = r.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r = r / norm

    return r.contiguous()
