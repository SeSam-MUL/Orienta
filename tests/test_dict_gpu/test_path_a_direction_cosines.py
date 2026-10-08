"""Verify our PyTorch direction-cosine grid matches kikuchipy's numpy reference
element-wise."""
import numpy as np
import pytest
import torch


@pytest.fixture
def detector():
    import kikuchipy as kp
    return kp.detectors.EBSDDetector(
        shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0, tilt=0.0
    )


@pytest.mark.gpu
def test_direction_cosines_shape_and_dtype(detector):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu._pcadi._projection.direction_cosines import (
        compute_direction_cosines,
    )
    grid = compute_direction_cosines(detector, device="cuda", dtype=torch.float32)
    assert grid.shape == (60, 60, 3)
    assert grid.dtype == torch.float32
    assert grid.is_cuda
    norms = grid.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


@pytest.mark.gpu
def test_direction_cosines_matches_kikuchipy_reference(detector):
    """Element-wise check against kikuchipy's own direction cosines for this detector."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu._pcadi._projection.direction_cosines import (
        compute_direction_cosines,
    )
    # The private per-PC function changed its signature between kikuchipy 0.11
    # (pcx, pcy, pcz, nrows, ncols, tilt, ...) and 0.13 (gnomonic bounds and an
    # orientation matrix), so call the wrapper that takes the detector: it has
    # the same signature in both and picks the right function itself.
    from kikuchipy.signals.util._master_pattern import (
        _get_direction_cosines_from_detector,
    )

    nrows, ncols = detector.shape
    ref = _get_direction_cosines_from_detector(detector)
    if ref.ndim == 2:
        ref = ref.reshape(nrows, ncols, 3)
    ref = ref.astype(np.float32)

    ours = compute_direction_cosines(detector, device="cuda", dtype=torch.float32).cpu().numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5, rtol=1e-5)
