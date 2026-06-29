"""Custom exceptions for the GPU spherical indexing pipeline."""


class SphericalGPUError(Exception):
    """Base class for all GPU spherical indexing errors."""


class SHTReadError(SphericalGPUError):
    """Raised when an EMsoft .sht binary master file cannot be parsed."""


class DetectorConventionError(SphericalGPUError):
    """Raised when an unknown vendor or invalid PC values are passed."""


class GPUUnavailableError(SphericalGPUError):
    """Raised when CUDA is requested but unavailable, and CPU fallback is too slow."""


class OracleIntegrityError(SphericalGPUError):
    """Raised when the validation oracle's sha256 does not match its lock file."""
