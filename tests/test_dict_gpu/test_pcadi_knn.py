import pytest, torch
import numpy as np
from backend.dict_gpu._pcadi.knn import gemm_topk_ncc


def _normalise(X):
    """Mean-center and L2-normalise rows."""
    X = X - X.mean(dim=1, keepdim=True)
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)


@pytest.mark.gpu
def test_gemm_topk_ncc_top1_matches_numpy_argmax():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(0)
    Q = torch.from_numpy(rng.standard_normal((50, 256)).astype(np.float32)).cuda()
    D = torch.from_numpy(rng.standard_normal((1000, 256)).astype(np.float32)).cuda()
    Qn = _normalise(Q)
    Dn = _normalise(D)

    scores, indices = gemm_topk_ncc(Qn, Dn, k=1, use_fp16=False)
    assert scores.shape == (50, 1)
    assert indices.shape == (50, 1)
    assert scores.dtype == torch.float32
    assert indices.dtype == torch.int64

    # Reference via plain numpy
    ref_scores_full = (Qn @ Dn.T).cpu().numpy()
    ref_top1_idx = ref_scores_full.argmax(axis=1)
    ref_top1_score = ref_scores_full.max(axis=1)

    np.testing.assert_array_equal(indices[:, 0].cpu().numpy(), ref_top1_idx)
    np.testing.assert_allclose(scores[:, 0].cpu().numpy(), ref_top1_score, atol=1e-5)


@pytest.mark.gpu
def test_gemm_topk_ncc_top5_returns_descending_scores():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(1)
    Q = torch.from_numpy(rng.standard_normal((10, 64)).astype(np.float32)).cuda()
    D = torch.from_numpy(rng.standard_normal((200, 64)).astype(np.float32)).cuda()
    Qn = _normalise(Q)
    Dn = _normalise(D)

    scores, indices = gemm_topk_ncc(Qn, Dn, k=5, use_fp16=False)
    assert scores.shape == (10, 5)
    assert indices.shape == (10, 5)
    # Each row's scores must be non-increasing
    diffs = scores[:, 1:] - scores[:, :-1]
    assert (diffs <= 1e-5).all(), f"scores not descending: max diff {diffs.max().item()}"
    # Indices in each row must be unique
    for r in range(10):
        assert len(set(indices[r].cpu().tolist())) == 5


@pytest.mark.gpu
def test_gemm_topk_ncc_self_match_yields_unity():
    """When the query equals the dictionary, top-1 must be score 1.0 at the diagonal."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(2)
    X = torch.from_numpy(rng.standard_normal((30, 100)).astype(np.float32)).cuda()
    Xn = _normalise(X)

    scores, indices = gemm_topk_ncc(Xn, Xn, k=1, use_fp16=False)
    assert torch.allclose(scores[:, 0], torch.ones(30, device="cuda"), atol=1e-5)
    assert (indices[:, 0] == torch.arange(30, device="cuda")).all()


@pytest.mark.gpu
def test_gemm_topk_ncc_fp16_close_to_fp32():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(3)
    Q = torch.from_numpy(rng.standard_normal((30, 256)).astype(np.float32)).cuda()
    D = torch.from_numpy(rng.standard_normal((500, 256)).astype(np.float32)).cuda()
    Qn = _normalise(Q)
    Dn = _normalise(D)

    s32, i32 = gemm_topk_ncc(Qn, Dn, k=1, use_fp16=False)
    s16, i16 = gemm_topk_ncc(Qn, Dn, k=1, use_fp16=True)

    # Top-1 indices should agree on the vast majority (>95%) of queries
    agree = (i32[:, 0] == i16[:, 0]).float().mean().item()
    assert agree >= 0.95, f"FP16 vs FP32 top-1 agreement only {agree:.3f}"
    # Scores should be close
    np.testing.assert_allclose(
        s32[:, 0].cpu().numpy(), s16[:, 0].cpu().numpy(), atol=1e-2
    )
