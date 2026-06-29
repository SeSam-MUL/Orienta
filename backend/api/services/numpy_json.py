"""Global numpy-aware JSON encoder registration for FastAPI.

FastAPI's default ``jsonable_encoder`` does not handle numpy scalars
(``numpy.int32``, ``numpy.float64``, etc.) or numpy arrays. Code-paths
that return detector shapes, pattern centers, or anything touching
kikuchipy structures easily leak numpy types into responses and produce
500 Internal Server Errors at serialization time (see commits 3a305a3,
c3d742d for two concrete bugs that hit the same bug class).

Rather than pepper every route with ``[int(v) for v in ...]`` and
``[float(v) for v in ...]`` coercions, register numpy scalar/array
conversions once, here, via FastAPI's ``ENCODERS_BY_TYPE`` registry.
Every endpoint returning dicts/lists that happen to contain numpy
values will serialize cleanly.

Call :func:`register_numpy_encoders` exactly once at app startup
(from ``backend.api.main``).
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi.encoders import ENCODERS_BY_TYPE

logger = logging.getLogger(__name__)


def _ndarray_to_list(arr: np.ndarray):
    """Convert ndarray to native Python list (tolist recursively casts)."""
    return arr.tolist()


def register_numpy_encoders() -> None:
    """Register numpy types with FastAPI's jsonable_encoder.

    FastAPI's ENCODERS_BY_TYPE is consulted via exact-type match in some
    code paths and via isinstance in others (depending on FastAPI version).
    To be robust against both, register the abstract bases AND every
    concrete scalar dtype we're likely to see from kikuchipy/h5py/orix.

    Idempotent — safe to call multiple times.
    """
    # Concrete integer dtypes (exact-type matches in FastAPI's encoder).
    int_types = (
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
    )
    for t in int_types:
        ENCODERS_BY_TYPE[t] = int
    # Also register the abstract base for any numpy int subclass we miss.
    ENCODERS_BY_TYPE[np.integer] = int

    # Concrete float dtypes.
    float_types = (np.float16, np.float32, np.float64)
    for t in float_types:
        ENCODERS_BY_TYPE[t] = float
    ENCODERS_BY_TYPE[np.floating] = float

    # Booleans.
    ENCODERS_BY_TYPE[np.bool_] = bool

    # Complex (rare, but defensive).
    ENCODERS_BY_TYPE[np.complex64] = lambda z: {"real": float(z.real), "imag": float(z.imag)}
    ENCODERS_BY_TYPE[np.complex128] = lambda z: {"real": float(z.real), "imag": float(z.imag)}
    ENCODERS_BY_TYPE[np.complexfloating] = lambda z: {"real": float(z.real), "imag": float(z.imag)}

    # Arrays — use tolist() which recursively converts to native types.
    ENCODERS_BY_TYPE[np.ndarray] = _ndarray_to_list

    # Datetime / timedelta — h5py occasionally returns these from time-tagged
    # attributes in HDF5 files. FastAPI's default jsonable_encoder couldn't
    # handle them and produced a 500 with "object is not iterable".
    # Convert to ISO-8601 string / float seconds — the formats every JSON
    # consumer can parse.
    ENCODERS_BY_TYPE[np.datetime64] = lambda d: str(np.datetime_as_string(d))
    ENCODERS_BY_TYPE[np.timedelta64] = lambda t: float(t / np.timedelta64(1, 's'))

    logger.debug(
        "numpy encoders registered for %d types",
        len(int_types) + len(float_types) + 7,  # + bases, bool, 2 complex, ndarray, datetime, timedelta
    )
