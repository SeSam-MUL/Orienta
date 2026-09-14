"""Verify bilinear sampling on Lambert hemispheres matches the reference."""
import numpy as np
import pytest
import torch


@pytest.mark.gpu
def test_sample_at_north_pole_returns_central_pixel():
    """Direction (0,0,+1) -> Lambert (0,0) -> north hemisphere centre pixel."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu._pcadi._projection.sample import sample_master

    npx = 21
    rng = np.random.default_rng(0)
    mp_north = torch.from_numpy(rng.standard_normal((npx, npx)).astype(np.float32)).cuda()
    mp_south = torch.from_numpy(rng.standard_normal((npx, npx)).astype(np.float32)).cuda()

    dirs = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    out = sample_master(mp_north, mp_south, dirs)

    centre = mp_north[npx // 2, npx // 2].item()
    np.testing.assert_allclose(out.cpu().numpy(), [centre], atol=1e-3)


@pytest.mark.gpu
def test_sample_hemisphere_selection():
    """+z -> north, -z -> south."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu._pcadi._projection.sample import sample_master

    npx = 11
    mp_north = torch.zeros((npx, npx), device="cuda")
    mp_south = torch.ones((npx, npx), device="cuda") * 3.7

    dirs = torch.tensor([
        [0.0, 0.0, +1.0],
        [0.0, 0.0, -1.0],
    ], device="cuda")
    out = sample_master(mp_north, mp_south, dirs).cpu().numpy()
    np.testing.assert_allclose(out, [0.0, 3.7], atol=1e-3)


@pytest.mark.gpu
def test_sample_against_kikuchipy_reference():
    """50 random directions: compare to kikuchipy's full helper chain."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu._pcadi._projection.sample import sample_master
    from kikuchipy.signals.util._master_pattern import (
        _get_lambert_interpolation_parameters,
        _get_pixel_from_master_pattern,
    )

    npx = 51
    rng = np.random.default_rng(42)
    mp_north_np = rng.standard_normal((npx, npx)).astype(np.float32)
    mp_south_np = rng.standard_normal((npx, npx)).astype(np.float32)
    mp_north = torch.from_numpy(mp_north_np).cuda()
    mp_south = torch.from_numpy(mp_south_np).cuda()

    v = rng.standard_normal((50, 3)).astype(np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    # kikuchipy's _get_lambert_interpolation_parameters takes (v, npx, npy, scale)
    # and is batched. It returns 8 arrays. We then dispatch per-vector to the
    # right hemisphere and call _get_pixel_from_master_pattern.
    scale = (npx - 1) / 2.0
    nii, nij, niip, nijp, di, dj, dim, djm = _get_lambert_interpolation_parameters(
        v, npx, npx, scale
    )
    ref = np.zeros(50, dtype=np.float32)
    for i in range(50):
        hemi_np = mp_north_np if v[i, 2] >= 0 else mp_south_np
        ref[i] = _get_pixel_from_master_pattern(
            hemi_np,
            nii[i], nij[i], niip[i], nijp[i],
            di[i], dj[i], dim[i], djm[i],
        )

    ours = sample_master(mp_north, mp_south, torch.from_numpy(v).cuda().float()).cpu().numpy()
    # grid_sample's bilinear convention may produce ~1e-3 differences vs the
    # hand-rolled four-corner weighted-sum reference at boundary pixels.
    np.testing.assert_allclose(ours, ref, atol=5e-3)
