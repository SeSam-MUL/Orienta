"""Vendor-copied math primitives from ebsdtorch.

See ``__SOURCE.md`` for attribution, license, and patch policy.

Public surface used by `backend.spherical_gpu.pipeline`:

- ``rosca_lambert``, ``inv_rosca_lambert`` — sphere/square projection
- ``theta_phi_to_xyz``, ``xyz_to_theta_phi`` — spherical/Cartesian conversion
- ``wigner_d`` — Wigner-d matrix dispatcher (logspace + xnum)
- ``RSHT`` — real spherical harmonic transform module
- ``rs2cc_``, ``cs2cc_`` — SHT cross-correlation kernels
"""
from .lambert import (
    inv_rosca_lambert,
    rosca_lambert,
    rosca_lambert_side_by_side,
    theta_phi_to_xyz,
    xyz_to_theta_phi,
)
from .sht import (
    RSHT, CSHT, DLT, RDLT,
    grid_DriscollHealy,
    theta_phi_to_xyz,
)  # noqa: F401
from .sht_cc import cs2cc_, rs2cc_, rs2cc_fast_
from .wigner_d import wigner_d

__all__ = [
    # Lambert
    "rosca_lambert",
    "inv_rosca_lambert",
    "rosca_lambert_side_by_side",
    "theta_phi_to_xyz",
    "xyz_to_theta_phi",
    # SHT
    "RSHT",
    "CSHT",
    "DLT",
    "RDLT",
    "grid_DriscollHealy",
    # Cross-correlation
    "rs2cc_",
    "rs2cc_fast_",
    "cs2cc_",
    # Wigner
    "wigner_d",
]
