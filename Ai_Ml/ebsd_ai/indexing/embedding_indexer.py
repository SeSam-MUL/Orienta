"""High-level embedding-based EBSD indexer.

Chains: encoder -> FAISS query -> optional refinement -> result dict.
Provides the same interface pattern as other indexing methods.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ebsd_ai.indexing.faiss_index import EBSDFaissIndex
from ebsd_ai.indexing.local_refinement import refine_orientations
from ebsd_ai.models.embedding_encoder import EmbeddingEncoder


class EmbeddingIndexer:
    """Embedding-based EBSD pattern indexer.

    Parameters
    ----------
    model : EmbeddingEncoder
        Trained embedding encoder.
    faiss_index : EBSDFaissIndex
        Built FAISS index with reference embeddings.
    device : torch.device or None
        Device for inference. Defaults to CPU.
    """

    def __init__(
        self,
        model: EmbeddingEncoder,
        faiss_index: EBSDFaissIndex,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.faiss_index = faiss_index
        self.device = device or torch.device("cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def _encode_patterns(
        self,
        patterns: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode patterns to embeddings in batches.

        Parameters
        ----------
        patterns : np.ndarray
            (N, H, W) float32 patterns.
        batch_size : int
            Batch size for encoding.

        Returns
        -------
        np.ndarray
            (N, D) L2-normalized embeddings.
        """
        n = len(patterns)
        all_embeddings = []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = patterns[start:end]
                # Add channel dim: (B, H, W) -> (B, 1, H, W)
                tensor = torch.from_numpy(batch).unsqueeze(1).float().to(self.device)
                emb = self.model(tensor).cpu().numpy()
                all_embeddings.append(emb)

        return np.concatenate(all_embeddings, axis=0)

    def index_scan(
        self,
        patterns: np.ndarray,
        k: int = 5,
        batch_size: int = 64,
        refine: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Index a scan of EBSD patterns.

        Parameters
        ----------
        patterns : np.ndarray
            (N, H, W) float32 patterns.
        k : int
            Number of nearest neighbors to retrieve.
        batch_size : int
            Batch size for CNN encoding.
        refine : bool
            Whether to apply quaternion refinement.
        progress_callback : callable or None
            Called with (current, total) during encoding.

        Returns
        -------
        dict
            Keys: orientations (N,4), phase_ids (N,), confidence (N,),
            top_k_orientations (N,k,4), top_k_distances (N,k).
        """
        # Step 1: Encode patterns to embeddings
        embeddings = self._encode_patterns(patterns, batch_size)

        # Step 2: FAISS query
        query_results = self.faiss_index.query(embeddings, k=k)

        top_k_oris = query_results["orientations"]  # (N, k, 4)
        top_k_dists = query_results["distances"]  # (N, k)
        top_k_phases = query_results["phase_ids"]  # (N, k)

        # Step 3: Best orientation (optionally refined)
        if refine and k > 1:
            orientations = refine_orientations(top_k_oris, top_k_dists)
        else:
            orientations = top_k_oris[:, 0, :]

        # Step 4: Phase ID from nearest neighbor
        phase_ids = top_k_phases[:, 0]

        # Step 5: Confidence from distance ratio
        if k >= 2:
            d1 = top_k_dists[:, 0]
            d2 = top_k_dists[:, 1]
            # For inner product: higher = more similar
            # Confidence = how much better the best match is vs second best
            confidence = np.clip(d1 - d2, 0, None)
        else:
            confidence = top_k_dists[:, 0]

        return {
            "orientations": orientations,
            "phase_ids": phase_ids,
            "confidence": confidence,
            "top_k_orientations": top_k_oris,
            "top_k_distances": top_k_dists,
        }
