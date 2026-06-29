"""Pattern Center convention conversions and detector geometry.

References
----------
- EMSphInx ``include/modality/ebsd/detector.hpp``
- https://kikuchipy.org/en/stable/tutorials/pattern_center.html
- ``kikuchipy.detectors.EBSDDetector`` (MIT-licensed)

PC convention notes
-------------------
- **Oxford**: ``(PCx, PCy, DD)`` all in [0,1], normalized to detector width.
  ``PCx`` is from LEFT edge, ``PCy`` from TOP edge (image space, Y down).
  ``DD`` is detector distance / detector width.
- **EDAX**: ``(xstar, ystar, zstar)``. Origin at bottom; ``zstar`` is
  detector distance / detector height.
- **EMsoft**: ``(xpc, ypc, L)``. ``xpc, ypc`` are pixels (centered, Y up);
  ``L`` is detector distance in microns.
- **Bruker**: same numerical layout as Oxford here (Aztec exports
  the Bruker variant identically), so we treat it equivalently.

The EMSphInx NML accepts EMsoft convention via the ``vendor='EMsoft'`` flag
AND its own native convention. Internally we always work in EMsoft units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from ..exceptions import DetectorConventionError


# Single source of truth for the detector pixel size assumed when a dataset
# carries no explicit value (Aztec/Bruker H5OINA often omit it). BOTH the
# Single-Pixel Phase Test (build_spherical_det_params) and the PC-Refinement
# Forward-Sim preview MUST use this so the two paths build the SAME detector
# distance L for the same PC — otherwise the same phase renders differently in
# the two tools (root-caused 2026-06-27: Phase Test used 55, PC-Ref used 70 →
# 27 % different L → divergent simulated patterns at an identical PC).
DEFAULT_PIXEL_SIZE_UM = 70.0


def convert_pc_to_emsphinx(
    pc: Tuple[float, float, float],
    vendor: str,
    pat_width: int,
    pat_height: int,
    pixel_size: float,
) -> Tuple[float, float, float]:
    """Convert a pattern center to EMSphInx-native (cX, cY, sDst).

    EMSphInx native convention (detector.hpp):
    - ``cX``: pixel offset from image center in x (positive = right)
    - ``cY``: pixel offset from image center in y (positive = down in image)
    - ``sDst``: scintillator distance in microns

    Oxford vendor formulas (detector.hpp lines 248-265):
    - ``cX = (PCx - 0.5) * w``    (PCx normalized to image width w)
    - ``cY = (PCy - 0.5) * h``    (PCy normalized to image height h, from top)
    - ``sDst = PCz * w * pixel_size``  (PCz normalized to image width)
    """
    PCx, PCy, PCz = float(pc[0]), float(pc[1]), float(pc[2])
    w = int(pat_width)
    h = int(pat_height)

    if vendor in ("Oxford", "Bruker"):
        if vendor == "Oxford":
            # Oxford H5OINA: PCx and PCy are BOTH normalized to detector
            # WIDTH, not height. PCy is measured from the top of the image.
            # DD is also normalized to width. This matches the kikuchipy
            # Oxford convention and the AzTec H5OINA spec.
            #
            # Bug history: previously we used `(PCy - 0.5) * h` which is
            # correct only for square patterns. On non-square detectors
            # (e.g. 128x156) this produces a ~19 px Y error and gives
            # essentially random orientations. Verified against the
            # kikuchipy Oxford -> Bruker -> EMsoft chain on a real 7050
            # H5OINA: ypc must be -43.5 px for PCy=0.689, w=156, h=128.
            cX = (PCx - 0.5) * w
            cY = PCy * w - 0.5 * h
            sDst = PCz * w * pixel_size
        else:  # Bruker
            # Bruker (kikuchipy native): PCx normalized to width from
            # LEFT, PCy normalized to HEIGHT from BOTTOM. DD normalized to
            # height. So:
            #   y_pix_from_top = (1 - PCy) * h
            #   cY (= y_pix_from_top - h/2) = (0.5 - PCy) * h
            # Verified against kikuchipy: Oxford PCy=0.689 -> Bruker
            # PCy=0.160, both giving y_pix_from_top=107.5, cY=+43.5 with
            # the formula below.
            cX = (PCx - 0.5) * w
            cY = (0.5 - PCy) * h
            sDst = PCz * h * pixel_size
    elif vendor == "EDAX":
        # EDAX/TSL convention — match EMSphInx detector.hpp:250 exactly:
        #     cX = xStar * w - w/2
        #     cY = yStar * w - h/2
        #     sDst = zStar * w * pixel_size
        # iter-15G validation: with this formula, vendor='EDAX' matches
        # vendor='Oxford' for square detectors, and HiGainNi vs EMSphInx
        # alt-oracle gives 97.5 percent agreement < 5 deg.
        cX = PCx * w - 0.5 * w
        cY = PCy * w - 0.5 * h
        sDst = PCz * w * pixel_size
    elif vendor == "EMsoft":
        # EMsoft: xpc = -cX, ypc = cY (in unbinned pixels)
        # sDst = L (in microns)
        cX = -PCx
        cY = PCy
        sDst = PCz
    else:
        raise DetectorConventionError(
            f"Unknown vendor '{vendor}'. "
            "Supported: Oxford, EDAX, Bruker, EMsoft."
        )
    return float(cX), float(cY), float(sDst)


def convert_pc_to_emsoft(
    pc: Tuple[float, float, float],
    vendor: str,
    pat_width: int,
    pat_height: int,
    pixel_size: float,
    binning: int,
) -> Tuple[float, float, float]:
    """Convert a pattern center from any supported vendor convention to EMsoft.

    Implementation strategy: chain ``vendor -> Bruker (canonical) -> EMsoft``.
    This mirrors :class:`kikuchipy.detectors.EBSDDetector` exactly so we
    can cross-validate bit-for-bit. Kikuchipy stores the PC internally in
    Bruker convention; their published transforms are the source of truth.

    EMsoft v5 convention (default in current EMsoft):
    - ``xpc, ypc`` in *unbinned* pixels, centered on the detector
    - ``xpc`` positive to the LEFT of center (sign flips between v4 and v5)
    - ``ypc`` positive UPWARD
    - ``L`` is the detector distance in microns (unbinned)

    Parameters
    ----------
    pc : (PCx, PCy, DD) — vendor-specific normalized form
    vendor : 'Oxford', 'EDAX', 'Bruker', or 'EMsoft'
    pat_width, pat_height : detector size in pixels (post-binning, the
        shape of the actual loaded pattern array)
    pixel_size : pixel size in microns (post-binning)
    binning : binning factor

    Returns
    -------
    (xpc, ypc, L) in EMsoft units

    Raises
    ------
    DetectorConventionError
        For unsupported vendors.
    """
    PCx, PCy, DD = float(pc[0]), float(pc[1]), float(pc[2])
    nx = int(pat_width)
    ny = int(pat_height)
    aspect = nx / ny  # = ncols / nrows

    # Step 1: vendor -> Bruker (canonical internal form)
    if vendor in ("Oxford",):
        # Oxford: PCx unchanged, PCy is 1 - pcy*aspect, DD is DD*aspect
        # (matches kikuchipy._pc_oxford2bruker).
        pc_x_b = PCx
        pc_y_b = 1.0 - PCy * aspect
        pc_z_b = DD * aspect
    elif vendor == "EDAX":
        # EDAX (TSL): PCx unchanged, PCy = 1 - pcy, DD scaled by
        # min(nrows, ncols) / nrows  (matches kikuchipy._pc_tsl2bruker).
        pc_x_b = PCx
        pc_y_b = 1.0 - PCy
        pc_z_b = DD * (min(ny, nx) / ny)
    elif vendor == "Bruker":
        # Already canonical
        pc_x_b, pc_y_b, pc_z_b = PCx, PCy, DD
    elif vendor == "EMsoft":
        # Skip the Bruker step entirely — already in target units
        return PCx, PCy, DD
    else:
        raise DetectorConventionError(
            f"Unknown vendor '{vendor}'. "
            "Supported: Oxford, EDAX, Bruker, EMsoft."
        )

    # Step 2: Bruker -> EMsoft v5 (the version flag in current EMsoft).
    # Reference: kikuchipy._pc_bruker2emsoft
    xpc = (0.5 - pc_x_b) * nx * binning
    ypc = (0.5 - pc_y_b) * ny * binning
    L = pc_z_b * ny * binning * pixel_size

    return float(xpc), float(ypc), float(L)


@dataclass
class DetectorGeometry:
    """All detector parameters needed by the indexing pipeline.

    PC values are stored as 0-d torch tensors on the requested device for
    direct use in tensor ops; scalar fields stay as Python floats so the
    object remains JSON-serializable for logging.

    Two PC representations are stored:
    - ``xpc, ypc, L`` — EMsoft v5 convention (legacy, kept for compatibility)
    - ``cx, cy, L``  — EMSphInx native convention (used by Tier1Indexer)
      ``cx``: pixel offset from center, positive = right
      ``cy``: pixel offset from center, positive = down (image convention)
    """
    pat_w: int
    pat_h: int
    pixel_size: float
    binning: int
    tilt_deg: float
    xpc: torch.Tensor
    ypc: torch.Tensor
    L: torch.Tensor
    vendor: str
    # EMSphInx-native pattern center (set by from_params; may be None for
    # DetectorGeometry constructed manually without vendor info)
    cx: torch.Tensor = None
    cy: torch.Tensor = None
    # Experimental sample tilt in degrees, propagated from
    # detector_params["sample_tilt"]. Tier1Indexer uses this as the
    # second-priority source for the sample tilt (after an explicit
    # ``sample_tilt_deg`` constructor argument) to avoid the silent
    # default to ``master.primary_tilt_deg`` (the MC simulation tilt,
    # which is metadata, not the experimental geometry).
    sample_tilt_deg: float = None

    @classmethod
    def from_params(
        cls, params: dict, device: torch.device,
    ) -> "DetectorGeometry":
        """Build a DetectorGeometry from a flat dict (the format used by
        ``indexing_controller`` and the conftest ``detector_params`` fixture)."""
        xpc, ypc, L = convert_pc_to_emsoft(
            pc=(params["pc_x"], params["pc_y"], params["pc_z"]),
            vendor=params["vendor"],
            pat_width=params["pat_width"],
            pat_height=params["pat_height"],
            pixel_size=params["pixel_size"],
            binning=params["binning"],
        )
        cx, cy, _ = convert_pc_to_emsphinx(
            pc=(params["pc_x"], params["pc_y"], params["pc_z"]),
            vendor=params["vendor"],
            pat_width=params["pat_width"],
            pat_height=params["pat_height"],
            pixel_size=params["pixel_size"],
        )
        sample_tilt = params.get("sample_tilt")
        return cls(
            pat_w=int(params["pat_width"]),
            pat_h=int(params["pat_height"]),
            pixel_size=float(params["pixel_size"]),
            binning=int(params["binning"]),
            tilt_deg=float(params["tilt"]),
            xpc=torch.tensor(xpc, dtype=torch.float64, device=device),
            ypc=torch.tensor(ypc, dtype=torch.float64, device=device),
            L=torch.tensor(L, dtype=torch.float64, device=device),
            vendor=str(params["vendor"]),
            cx=torch.tensor(cx, dtype=torch.float64, device=device),
            cy=torch.tensor(cy, dtype=torch.float64, device=device),
            sample_tilt_deg=float(sample_tilt) if sample_tilt is not None else None,
        )
