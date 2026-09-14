import pytest


def test_compute_tile_size_returns_positive_within_budget():
    from backend.dict_gpu.pipeline.tiling import compute_tile_size
    # 500k dictionary entries, 3600 floats, 2 bytes (FP16), 8 GB budget
    tile = compute_tile_size(
        n_dict=500_000, pattern_dim=3600, dtype_bytes=2, vram_budget_bytes=8_000_000_000
    )
    assert 1 <= tile <= 500_000
    # The dictionary slab `tile * 3600 * 2` bytes must leave >=30% of budget free
    used = tile * 3600 * 2
    assert used <= 8_000_000_000 * 0.70 + 1, (
        f"tile {tile} uses {used / 1e9:.2f} GB of {8_000_000_000 / 1e9:.1f} GB "
        f"budget, exceeding the 70% target"
    )


def test_compute_tile_size_caps_at_n_dict_when_budget_is_huge():
    from backend.dict_gpu.pipeline.tiling import compute_tile_size
    tile = compute_tile_size(
        n_dict=1000, pattern_dim=64, dtype_bytes=4, vram_budget_bytes=100_000_000_000
    )
    assert tile == 1000  # whole dict fits trivially


def test_compute_tile_size_returns_at_least_one_even_for_tiny_budget():
    from backend.dict_gpu.pipeline.tiling import compute_tile_size
    tile = compute_tile_size(
        n_dict=1_000_000, pattern_dim=10_000, dtype_bytes=4, vram_budget_bytes=1024
    )
    assert tile >= 1


def test_iter_tiles_covers_all_indices_no_overlap():
    from backend.dict_gpu.pipeline.tiling import iter_tiles
    slices = list(iter_tiles(1000, 300))
    assert slices == [slice(0, 300), slice(300, 600), slice(600, 900), slice(900, 1000)]


def test_iter_tiles_handles_exact_division():
    from backend.dict_gpu.pipeline.tiling import iter_tiles
    slices = list(iter_tiles(900, 300))
    assert slices == [slice(0, 300), slice(300, 600), slice(600, 900)]


def test_iter_tiles_single_tile_when_total_le_size():
    from backend.dict_gpu.pipeline.tiling import iter_tiles
    slices = list(iter_tiles(50, 1000))
    assert slices == [slice(0, 50)]


def test_iter_tiles_zero_or_negative_raises():
    from backend.dict_gpu.pipeline.tiling import iter_tiles
    with pytest.raises(ValueError):
        list(iter_tiles(100, 0))
    with pytest.raises(ValueError):
        list(iter_tiles(100, -5))


@pytest.mark.gpu
def test_tiled_gemm_topk_matches_single_pass():
    """Forced small tile vs single-pass produces identical top-1 indices."""
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")

    from backend.dict_gpu._pcadi.knn import gemm_topk_ncc
    from backend.dict_gpu.pipeline.tiling import iter_tiles

    rng = np.random.default_rng(0)
    Q = torch.from_numpy(rng.standard_normal((50, 256)).astype(np.float32)).cuda()
    D = torch.from_numpy(rng.standard_normal((1000, 256)).astype(np.float32)).cuda()

    def _norm(X):
        X = X - X.mean(dim=1, keepdim=True)
        return X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)

    Qn, Dn = _norm(Q), _norm(D)

    # Single-pass
    s_full, i_full = gemm_topk_ncc(Qn, Dn, k=1, use_fp16=False)

    # Tiled - for each tile of D, run topk(k=1), then aggregate the global max
    best_score = torch.full((50,), -float("inf"), device="cuda")
    best_idx = torch.full((50,), -1, dtype=torch.int64, device="cuda")
    for sl in iter_tiles(1000, 250):
        s_tile, i_tile = gemm_topk_ncc(Qn, Dn[sl], k=1, use_fp16=False)
        s_tile = s_tile[:, 0]
        i_tile = i_tile[:, 0] + sl.start  # offset to global indexing
        beats = s_tile > best_score
        best_score = torch.where(beats, s_tile, best_score)
        best_idx = torch.where(beats, i_tile, best_idx)

    # The aggregated tiled result must match the single-pass result
    assert (best_idx == i_full[:, 0]).all(), "tiled top-1 indices diverged from single pass"
    assert torch.allclose(best_score, s_full[:, 0], atol=1e-5)
