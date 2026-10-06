"""Static-background reference for the Viewer's and the batch's "static" removal.

Semantics shared by both callers:

* no reference pattern named  -> the mean of ALL patterns of the signal at hand
  (every batch file gets its own average);
* a reference row AND column  -> that single pattern;
* only one of the two         -> error (never guess the other one).

Defaulting the reference to pattern (0, 0) subtracts that one pattern from the
whole map, so its bands show up inverted everywhere.
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def validate_reference(row: Optional[int], col: Optional[int]) -> Optional[Tuple[int, int]]:
    """Return ``(row, col)`` for an explicit reference, ``None`` for the scan average.

    Raises ``ValueError`` when only one of the two is given.
    """
    if (row is None) != (col is None):
        raise ValueError(
            "Static-background reference pattern needs both static_bg_row and "
            "static_bg_col (or neither, for the scan average)."
        )
    if row is None:
        return None
    return int(row), int(col)


def scan_average(signal) -> np.ndarray:
    """Mean of all patterns of ``signal`` as an array of the pattern dtype.

    Accumulates in float64 over every navigation axis and rounds to the pattern
    dtype for integer data (kikuchipy requires the background dtype to equal the
    pattern dtype). On a lazy (dask) signal the reduction streams chunk by chunk,
    so peak memory is a few chunks, not the dataset.
    """
    data = signal.data
    nav_axes = tuple(range(data.ndim - 2))
    n_patterns = int(np.prod([data.shape[a] for a in nav_axes])) if nav_axes else 1
    lazy = hasattr(data, "compute")
    t0 = time.perf_counter()
    logger.info(
        "Static background: averaging %d patterns%s ...",
        n_patterns, " (streaming from disk)" if lazy else "",
    )
    mean_bg = np.asarray(data.mean(axis=nav_axes, dtype=np.float64))
    if np.issubdtype(data.dtype, np.integer):
        mean_bg = np.rint(mean_bg)
    logger.info("Static background: scan average of %d patterns done in %.1fs",
                n_patterns, time.perf_counter() - t0)
    return mean_bg.astype(data.dtype)
