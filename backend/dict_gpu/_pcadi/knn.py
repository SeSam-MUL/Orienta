"""GEMM-based top-k NCC search.

Inspired by ZacharyVarley/pcadi ChunkedKNN (utils.py ~4720, utils_knn.py ~110)
but trimmed to the inner GEMM + top-k step. Tiling/chunking lives in
backend/dict_gpu/pipeline/tiling.py (Task 13). Caller is responsible for
mean-centering and L2-normalising both query and dictionary so the GEMM
inner product equals NCC.

Source repo: https://github.com/ZacharyVarley/pcadi  (commit 88f676e)
License: MIT (see ../__SOURCE.md)
"""
from __future__ import annotations
import torch


def gemm_topk_ncc(
    query: torch.Tensor,
    dictionary: torch.Tensor,
    k: int = 1,
    *,
    use_fp16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k NCC search via a single GEMM.

    Parameters
    ----------
    query : (n_query, d) tensor — mean-centred + L2-normed by caller
    dictionary : (n_dict, d) tensor — same preprocessing
    k : how many top matches to keep per query
    use_fp16 : run the GEMM in FP16 (default True). Top-k is always done
        in FP32 to keep the index ordering stable.

    Returns
    -------
    scores : (n_query, k) float32, in [-1, 1], non-increasing along dim=1
    indices : (n_query, k) int64
    """
    if query.ndim != 2 or dictionary.ndim != 2:
        raise ValueError(
            f"query/dictionary must be 2D, got {tuple(query.shape)} / "
            f"{tuple(dictionary.shape)}"
        )
    if query.shape[1] != dictionary.shape[1]:
        raise ValueError(
            f"feature dim mismatch: query {query.shape[1]} vs "
            f"dictionary {dictionary.shape[1]}"
        )
    n_dict = dictionary.shape[0]
    if not (1 <= k <= n_dict):
        raise ValueError(f"k must be in [1, n_dict={n_dict}], got {k}")

    if use_fp16 and query.is_cuda:
        # Run the GEMM in FP16 under autocast for speed; cast back for topk
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            scores = query.half() @ dictionary.half().T
        scores = scores.float()
    else:
        scores = (query @ dictionary.T).float()

    # torch.topk returns descending by default
    top_scores, top_idx = torch.topk(scores, k=k, dim=1)
    return top_scores.contiguous(), top_idx.contiguous().to(torch.int64)
