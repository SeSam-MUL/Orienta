"""Static-background reference for the Viewer's and the batch's "static" removal.

Semantics shared by both callers:

* no reference pattern named  -> the mean of ALL patterns of the signal at hand
  (every batch file gets its own average);
* a reference row AND column  -> that single pattern;
* only one of the two         -> error (never guess the other one).

Defaulting the reference to pattern (0, 0) subtracts that one pattern from the
whole map, so its bands show up inverted everywhere.

The scan average of a lazy signal is computed in large sequential blocks, not
as one dask task per pattern: a 27 GB scan (485k patterns) took 530 s as 485k
single-pattern tasks, and the task graph alone cost about 4.5 GB of RAM.
"""

import logging
import os
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Target size of one read block. A block is held in RAM while it is summed, so
# this is (about) the peak working set of the average, independent of file size.
BLOCK_TARGET_BYTES = 128 * 1024 ** 2

ProgressFn = Callable[[int, int], None]  # (blocks_done, blocks_total)


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


# ---------------------------------------------------------------------------
# Cache: scan average per loaded dataset
# ---------------------------------------------------------------------------
# Key: (dask graph name, source file path, size, mtime). A dask array's name
# identifies its immutable graph, and a new load / crop / in-place processing
# step always produces a new array, so a hit can only be the very same data
# (a deepcopy of a dataset shares the graph, which is exactly when a second
# average would be wasted). Numpy-backed (eager) data is never cached: it is
# averaged in memory in a fraction of a second and may be mutated in place.
_CACHE_MAX_ENTRIES = 8
_cache: "dict[tuple, np.ndarray]" = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Forget every cached average (call when a file is loaded or a crop applied)."""
    with _cache_lock:
        _cache.clear()


def _source_identity(path: Optional[str]) -> tuple:
    if not path:
        return (None, None, None)
    try:
        st = os.stat(path)
        return (str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return (str(path), None, None)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _pristine_source(data):
    """The h5py-style dataset behind ``data``, if ``data`` is just a reshaped /
    rechunked view of it; else ``None``.

    The mean over ALL patterns does not depend on how the patterns are arranged
    on the navigation grid, so a graph made only of ``from_array`` / rechunk /
    reshape layers can be averaged straight from the dataset, in big
    sequential slabs, bypassing the per-pattern task graph.  Anything else
    (crop, processing, frame averaging, ...) makes the graph unequal to the
    stored data and returns ``None``.
    """
    try:
        graph = data.__dask_graph__()
        layers = getattr(graph, "layers", None)
        if not layers:
            return None
        sources = []
        for name in layers:
            if name.startswith("original-array"):
                for v in layers[name].values():
                    sources.append(v)
            elif not name.startswith(("array-", "rechunk", "reshape")):
                return None
        if len(sources) != 1:
            return None
        src = sources[0]
        if not (hasattr(src, "shape") and hasattr(src, "dtype") and hasattr(src, "__getitem__")):
            return None
        if len(src.shape) < 3 or tuple(src.shape[-2:]) != tuple(data.shape[-2:]):
            return None
        if int(np.prod(src.shape)) != int(np.prod(data.shape)) or src.dtype != data.dtype:
            return None
        return src
    except Exception:
        return None


def _block_rows(n_items: int, item_bytes: int, align: int = 1) -> int:
    rows = max(1, BLOCK_TARGET_BYTES // max(1, item_bytes))
    if align > 1:
        rows = max(align, (rows // align) * align)
    return int(min(rows, n_items))


def _sum_blocks(read, n_items: int, item_bytes: int, align: int, progress: Optional[ProgressFn]) -> np.ndarray:
    """Sum ``read(i, j)`` (items i..j along axis 0) over all items, float64."""
    step = _block_rows(n_items, item_bytes, align)
    n_blocks = -(-n_items // step)
    total = None
    if progress:
        progress(0, n_blocks)
    for b, i in enumerate(range(0, n_items, step)):
        block = np.asarray(read(i, min(i + step, n_items)))
        # float64 accumulation of integer pixels is exact (< 2**53), so the
        # result does not depend on the block size.
        part = block.reshape(-1, *block.shape[-2:]).sum(axis=0, dtype=np.float64)
        total = part if total is None else total + part
        if progress:
            progress(b + 1, n_blocks)
    return total


def scan_average(
    signal,
    progress: Optional[ProgressFn] = None,
    source_path: Optional[str] = None,
) -> np.ndarray:
    """Mean of all patterns of ``signal`` as an array of the pattern dtype.

    Accumulates in float64 and rounds to the pattern dtype for integer data
    (kikuchipy requires the background dtype to equal the pattern dtype).
    A lazy signal is streamed in blocks of about ``BLOCK_TARGET_BYTES``, so peak
    memory does not grow with the dataset; ``progress(done, total)`` is called
    after every block.  Lazy results are cached per dataset (see ``_cache``).
    """
    data = signal.data
    dtype = data.dtype
    n_patterns = int(np.prod(data.shape[:-2])) if data.ndim > 2 else 1
    lazy = hasattr(data, "compute")

    key = None
    if lazy:
        key = (getattr(data, "name", None),) + _source_identity(source_path)
        with _cache_lock:
            hit = _cache.get(key)
        if hit is not None:
            logger.info("Static background: scan average of %d patterns taken from cache", n_patterns)
            if progress:
                progress(1, 1)
            return hit.copy()

    t0 = time.perf_counter()
    if not lazy:
        total = data.reshape(-1, *data.shape[-2:]).sum(axis=0, dtype=np.float64)
        logger.info("Static background: averaging %d patterns (in memory) ...", n_patterns)
    else:
        item_bytes = int(np.prod(data.shape[-2:])) * dtype.itemsize
        src = _pristine_source(data)
        if src is not None:
            how = "sequential slabs from the file"
            align = int(src.chunks[0]) if getattr(src, "chunks", None) else 1
            logger.info("Static background: averaging %d patterns (%s) ...", n_patterns, how)
            total = _sum_blocks(lambda i, j: src[i:j], int(src.shape[0]), item_bytes, align, progress)
        else:
            how = "block-wise through the dask graph"
            logger.info("Static background: averaging %d patterns (%s) ...", n_patterns, how)
            # Walk the first navigation axis; each block is a small sub-graph.
            row_items = int(np.prod(data.shape[1:-2])) if data.ndim > 3 else 1
            total = _sum_blocks(lambda i, j: data[i:j], int(data.shape[0]), item_bytes * row_items, 1, progress)

    mean = total / float(n_patterns)
    if np.issubdtype(dtype, np.integer):
        mean = np.rint(mean)
    logger.info("Static background: scan average of %d patterns done in %.1fs",
                n_patterns, time.perf_counter() - t0)
    out = mean.astype(dtype)
    if key is not None:
        with _cache_lock:
            if len(_cache) >= _CACHE_MAX_ENTRIES:
                _cache.pop(next(iter(_cache)))
            _cache[key] = out.copy()
    return out
