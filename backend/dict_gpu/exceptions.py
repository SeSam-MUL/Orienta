"""Exceptions for the GPU dictionary indexing module.

These are *contract* errors - they fire on user-controllable conditions
(no GPU when GPU was forced, malformed master pattern, OOM beyond budget).
The controller surfaces them verbatim to the frontend toast.
"""

class GpuDictError(Exception):
    """Base class for all dict_gpu errors."""

class VramExhaustedError(GpuDictError):
    """The dictionary cannot be tiled small enough to fit the budget."""

class MasterPatternError(GpuDictError):
    """The master pattern is missing, malformed, or geometrically incompatible."""
