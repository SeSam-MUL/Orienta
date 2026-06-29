"""FAISS index wrapper for EBSD embedding retrieval.

Wraps FAISS indices with orientation and phase metadata for fast
nearest-neighbor lookup of crystal orientations from embeddings.
Auto-selects Flat for small datasets, IVF+PQ for larger ones.
"""
from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_IVF_THRESHOLD = 50_000


class EBSDFaissIndex:
    """FAISS-backed embedding index with orientation metadata.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._index: faiss.Index | None = None
        self._orientations: np.ndarray | None = None
        self._phase_ids: np.ndarray | None = None
        self.index_type: str = ""

    def build(
        self,
        embeddings: np.ndarray,
        orientations: np.ndarray,
        phase_ids: np.ndarray,
        use_gpu: bool = False,
    ) -> None:
        """Build the FAISS index.

        Parameters
        ----------
        embeddings : np.ndarray
            (N, dim) L2-normalized float32 embeddings.
        orientations : np.ndarray
            (N, 4) unit quaternions.
        phase_ids : np.ndarray
            (N,) integer phase IDs.
        use_gpu : bool
            Whether to move index to GPU (requires faiss-gpu).
        """
        n = len(embeddings)
        assert embeddings.shape == (n, self.dim)
        assert orientations.shape == (n, 4)
        assert phase_ids.shape == (n,)

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if n < _IVF_THRESHOLD:
            self._index = faiss.IndexFlatIP(self.dim)
            self.index_type = "Flat"
        else:
            n_list = min(int(np.sqrt(n)), 256)
            quantizer = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIVFPQ(
                quantizer, self.dim, n_list, 16, 8,
            )
            self._index.train(embeddings)
            self.index_type = "IVF+PQ"

        self._index.add(embeddings)
        self._orientations = orientations.copy()
        self._phase_ids = phase_ids.copy()

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("FAISS index moved to GPU")
            except Exception:
                logger.warning("GPU transfer failed, using CPU index")

        logger.info(
            "Built %s index with %d entries (dim=%d)",
            self.index_type, n, self.dim,
        )

    def query(
        self,
        embeddings: np.ndarray,
        k: int = 5,
    ) -> dict[str, np.ndarray]:
        """Query nearest neighbors.

        Parameters
        ----------
        embeddings : np.ndarray
            (Q, dim) query embeddings.
        k : int
            Number of neighbors to return.

        Returns
        -------
        dict
            Keys: indices (Q,k), distances (Q,k),
            orientations (Q,k,4), phase_ids (Q,k).
        """
        assert self._index is not None, "Index not built yet"
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self.index_type == "IVF+PQ":
            self._index.nprobe = min(16, self._index.nlist)

        distances, indices = self._index.search(embeddings, k)

        return {
            "indices": indices,
            "distances": distances,
            "orientations": self._orientations[indices],
            "phase_ids": self._phase_ids[indices],
        }

    def save(self, directory: str) -> None:
        """Save index and metadata to directory."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        # For GPU index, convert back to CPU for saving
        cpu_index = faiss.index_gpu_to_cpu(self._index) if hasattr(self._index, "getDevice") else self._index
        faiss.write_index(cpu_index, str(d / "faiss.index"))
        np.save(d / "orientations.npy", self._orientations)
        np.save(d / "phase_ids.npy", self._phase_ids)
        np.save(d / "meta.npy", np.array([self.dim], dtype=np.int64))

    @classmethod
    def load(cls, directory: str) -> "EBSDFaissIndex":
        """Load index from directory."""
        d = Path(directory)
        meta = np.load(d / "meta.npy")
        dim = int(meta[0])

        obj = cls(dim=dim)
        obj._index = faiss.read_index(str(d / "faiss.index"))
        obj._orientations = np.load(d / "orientations.npy")
        obj._phase_ids = np.load(d / "phase_ids.npy")
        obj.index_type = "loaded"
        return obj
