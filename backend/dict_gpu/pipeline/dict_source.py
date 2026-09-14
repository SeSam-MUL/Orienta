"""Where the dictionary entries come from, one block at a time.

The indexer used to take the dictionary as one array and put all of it on the
GPU. That caps the dictionary at roughly a quarter of the card (measured peak:
3.76x the dictionary), which is why the user's 16 GB and 24 GB dictionaries
died with CUDA-OOM on a 12.88 GB card while the 8 GB one only ran through the
Windows driver's system-memory fallback.

A source hands out contiguous blocks of entries instead, so the indexer can
normalise and correlate one tile at a time and never hold more than a tile.
Two of them:

``ArrayDictSource``
    The dictionary is already a host array — which is what the Indexing page
    produces, because it loads each phase's dictionary with an eager
    ``kp.load`` before calling the indexer. Blocks are plain slices; the cost
    is the host-to-device copy the resident path pays as well.

``H5DictSource``
    The dictionary is a file. Blocks are read straight from the HDF5 dataset,
    so the entries never exist in host RAM either. Measured on the user's
    8 GB Al dictionary (uncompressed, chunked 12 entries deep):

        sequential block of  6,000 entries   0.48 GB   3.71 GB/s
        sequential block of 24,000 entries   1.92 GB   3.78 GB/s
        sequential block of 48,000 entries   3.83 GB   3.52 GB/s
        sorted fancy-index gather of 5,000   0.40 GB   0.11 GB/s

    Sequential is what the indexer does; the 35x penalty on scattered reads is
    why nothing in the streaming path gathers single entries from the file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Dataset names that have held dictionary patterns in this project: kikuchipy
# writes ``Scan <n>/EBSD/Data/patterns``; the legacy cache writer used a
# top-level ``/patterns``.
_PATTERN_NAMES = ("patterns", "dictionary", "data")


class DictSource:
    """Read-only, block-at-a-time view of a dictionary of patterns.

    Subclasses return ``(m, pat_h, pat_w)`` float32 C-contiguous blocks from
    :meth:`read_block` so the caller can hand them to ``torch.from_numpy``
    without a further copy.
    """

    n_entries: int = 0
    pat_h: int = 0
    pat_w: int = 0
    #: Short description for log lines ("host array", "file <name>").
    origin: str = ""

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_entries, self.pat_h, self.pat_w)

    def read_block(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        """Release whatever the source holds open. Idempotent."""

    def __enter__(self) -> "DictSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ArrayDictSource(DictSource):
    """Blocks are slices of an array that is already in host RAM."""

    def __init__(self, array: np.ndarray):
        a = np.asarray(array)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 3:
            raise ValueError(
                f"dictionary array must be (n, h, w); got {a.shape}")
        self._a = a
        self.n_entries, self.pat_h, self.pat_w = (int(v) for v in a.shape)
        self.origin = "host array"

    def read_block(self, start: int, stop: int) -> np.ndarray:
        block = self._a[start:stop]
        # float32 + contiguous is the contract; both are already true for a
        # kikuchipy-loaded dictionary, so this is a no-op there rather than a
        # gigabyte of needless copying.
        if block.dtype != np.float32 or not block.flags["C_CONTIGUOUS"]:
            block = np.ascontiguousarray(block, dtype=np.float32)
        return block


class H5DictSource(DictSource):
    """Blocks are read from an HDF5 dataset, never held whole in RAM."""

    def __init__(self, path: str | Path, dataset_key: str):
        import h5py

        self._path = Path(path)
        self._key = dataset_key
        self._f = h5py.File(str(self._path), "r")
        ds = self._f[dataset_key]
        if ds.ndim != 3:
            self._f.close()
            raise ValueError(
                f"{self._path.name}:{dataset_key} is {ds.ndim}-D, expected "
                f"(n, h, w)")
        self.n_entries, self.pat_h, self.pat_w = (int(v) for v in ds.shape)
        self._ds = ds
        #: Entries per HDF5 chunk — tiles are rounded to this so a block never
        #: straddles a chunk it does not need.
        self.chunk_entries = int(ds.chunks[0]) if ds.chunks else 1
        self.origin = f"file {self._path.name}"

    def read_block(self, start: int, stop: int) -> np.ndarray:
        block = self._ds[start:stop]
        if block.dtype != np.float32:
            block = block.astype(np.float32)
        return np.ascontiguousarray(block)

    def close(self) -> None:
        f = getattr(self, "_f", None)
        if f is not None:
            try:
                f.close()
            finally:
                self._f = None
                self._ds = None

    def __del__(self):  # pragma: no cover — safety net for a failed run
        # An indexing run that raises half way through skips the explicit
        # close; without this the handle would live until the process ends,
        # which on Windows keeps the file locked against a regeneration.
        try:
            self.close()
        except Exception:
            pass


def find_pattern_dataset(path: str | Path) -> Optional[str]:
    """Name of the 3-D patterns dataset in an HDF5 dictionary, or None.

    Returns None — rather than raising — when the file is not an HDF5
    dictionary at all, so the caller can fall back to loading it whole.
    """
    try:
        import h5py
    except ImportError:  # pragma: no cover — h5py is a hard dependency
        return None
    found: list[str] = []
    try:
        with h5py.File(str(path), "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3 \
                        and name.rsplit("/", 1)[-1] in _PATTERN_NAMES \
                        and np.issubdtype(obj.dtype, np.number):
                    found.append(name)
            f.visititems(visit)
    except Exception:
        return None
    # More than one candidate means we would be guessing which is the
    # dictionary; let the caller take the loud, whole-file route instead.
    return found[0] if len(found) == 1 else None
