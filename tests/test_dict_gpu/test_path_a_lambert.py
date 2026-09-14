"""Verify vector_to_lambert and lambert_to_vector mirror kikuchipy's numpy
reference."""
import numpy as np
import pytest
import torch


@pytest.mark.gpu
def test_vector_to_lambert_matches_kikuchipy():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu._pcadi._projection.lambert import vector_to_lambert
    from kikuchipy.signals.util._master_pattern import _vector2lambert

    rng = np.random.default_rng(0)
    v = rng.standard_normal((200, 3)).astype(np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    # kikuchipy._vector2lambert is batched: takes (n, 3) returns (n, 2)
    ref = _vector2lambert(v)

    ours = vector_to_lambert(torch.from_numpy(v).cuda().float()).cpu().numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5)


@pytest.mark.gpu
def test_lambert_to_vector_matches_kikuchipy():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu._pcadi._projection.lambert import lambert_to_vector
    from kikuchipy.signals.util._master_pattern import _lambert2vector

    rng = np.random.default_rng(0)
    # _lambert2vector internally multiplies inputs by sqrt(pi/2). Inputs are
    # square-grid normalized coords in [-1, 1].
    uv = rng.uniform(-1.0, 1.0, size=(200, 2)).astype(np.float64)

    # kikuchipy._lambert2vector is batched: takes x[n], y[n] returns (n, 3)
    ref = _lambert2vector(uv[:, 0].copy(), uv[:, 1].copy())

    ours = lambert_to_vector(torch.from_numpy(uv).cuda().float()).cpu().numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-4)


@pytest.mark.gpu
def test_lambert_roundtrip_on_upper_hemisphere():
    """Forward then inverse Lambert recovers the original unit vector.

    kikuchipy's conventions differ between the two halves:
      * `_vector2lambert` outputs in `[-sqrt(pi)/2, +sqrt(pi)/2]`
      * `_lambert2vector` expects `[-1, 1]` and rescales by `sqrt(pi/2)`
    so to chain them we must divide the forward output by `SQRT_PI_HALF`
    (= sqrt(pi/2)). This is exactly what
    `_get_lambert_interpolation_parameters` does in kikuchipy.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import math
    from backend.dict_gpu._pcadi._projection.lambert import (
        vector_to_lambert, lambert_to_vector,
    )
    SQRT_PI_HALF = math.sqrt(math.pi / 2.0)
    rng = np.random.default_rng(0)
    v = rng.standard_normal((100, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[:, 2] = np.abs(v[:, 2])     # force upper hemisphere
    v = torch.from_numpy(v).cuda()
    uv = vector_to_lambert(v) / SQRT_PI_HALF
    v_back = lambert_to_vector(uv)
    # lambert_to_vector returns unnormalised vectors; normalise before comparing
    v_back = v_back / v_back.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    np.testing.assert_allclose(v_back.cpu().numpy(), v.cpu().numpy(), atol=1e-4)
