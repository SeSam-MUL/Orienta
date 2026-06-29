"""GpuPCA — SVD-based PCA on GPU tensors.

Inspired by ZacharyVarley/pcadi (utils.py OnlineCovMatrix, lines ~2080–2400)
but reimplemented as a direct SVD on the centered data matrix because:
  - our use case fits the dictionary in VRAM in one shot (no streaming need)
  - SVD on the centered data is the standard PCA path and is obviously correct
  - we avoid the complexity of accumulator state for MVP

Source repo: https://github.com/ZacharyVarley/pcadi  (commit 88f676e)
License: MIT (see ../__SOURCE.md)
"""
from __future__ import annotations
import torch


class GpuPCA:
    """SVD-based PCA on GPU tensors.

    Fit consumes a (n, d) tensor and stores the top n_components singular
    directions. Transform projects onto that basis; inverse_transform maps
    back into the original space (lossy when k < min(n, d)).
    """

    def __init__(self, n_components: int):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = n_components
        self.mean_: torch.Tensor | None = None
        self.components_: torch.Tensor | None = None  # (k, d)
        self.singular_values_: torch.Tensor | None = None  # (k,)

    def fit(self, X: torch.Tensor) -> "GpuPCA":
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {tuple(X.shape)}")
        n, d = X.shape
        if self.n_components > min(n, d):
            raise ValueError(
                f"n_components={self.n_components} > min(n,d)={min(n, d)}"
            )

        self.mean_ = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean_

        # PyTorch's default cusolver SVD (cusolverDnSgesvdj_bufferSize)
        # fails on the large rectangular matrices we hit in dict-gpu —
        # 100k × 20k float32 trips CUSOLVER_STATUS_INVALID_VALUE on some
        # cards/driver combos even when the input is clean. Three-tier
        # fallback ladder, fastest → most robust:
        #
        # 1. torch.svd_lowrank — randomised top-k SVD. Only computes the
        #    n_components we'll actually use (orders of magnitude cheaper
        #    than a full SVD when k << min(n,d), which is always true for
        #    PCA pre-processing). Much smaller workspace.
        # 2. torch.linalg.svd via the magma backend — full SVD but
        #    different solver; works when cusolver chokes.
        # 3. torch.linalg.svd via cusolver (the original path) — last
        #    resort, kept so we surface the real error if both above fail.
        k = self.n_components
        errors = []
        try:
            # svd_lowrank uses random projection; q controls oversampling,
            # niter=2 gives well-converged top-k singular vectors.
            U, S, V = torch.svd_lowrank(Xc, q=min(k + 10, min(n, d)), niter=2)
            self.components_ = V[:, :k].T.contiguous()  # V is (d, q) → (k, d)
            self.singular_values_ = S[:k].contiguous()
            return self
        except Exception as e:
            errors.append(f"svd_lowrank: {type(e).__name__}: {e}")

        for backend_name in ("magma", "cusolver"):
            try:
                torch.backends.cuda.preferred_linalg_library(backend_name)
                U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
                self.components_ = Vh[:k].contiguous()
                self.singular_values_ = S[:k].contiguous()
                return self
            except Exception as e:
                errors.append(f"linalg.svd({backend_name}): {type(e).__name__}: {e}")

        raise RuntimeError(
            "All SVD backends failed for PCA fit. Tried, in order: "
            + " | ".join(errors)
            + f". Matrix shape: ({n}, {d}), k={k}, dtype={Xc.dtype}, "
            f"device={Xc.device}."
        )

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("GpuPCA.transform called before fit")
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Y: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("GpuPCA.inverse_transform called before fit")
        return Y @ self.components_ + self.mean_
