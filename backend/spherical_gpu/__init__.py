"""GPU-accelerated spherical indexing for EBSD patterns.

Public API: :func:`backend.spherical_gpu.api.gpu_spherical_index_patterns`

The implementation parallels EMSphInx's IndexEBSD pipeline but runs on
GPU via PyTorch.
"""
from .exceptions import (
    DetectorConventionError,
    GPUUnavailableError,
    OracleIntegrityError,
    SHTReadError,
    SphericalGPUError,
)

__all__ = [
    "SphericalGPUError",
    "SHTReadError",
    "DetectorConventionError",
    "GPUUnavailableError",
    "OracleIntegrityError",
]
