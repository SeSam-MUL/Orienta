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
    """Element-wise check against kikuchipy._get_direction_cosines_for_fixed_pc."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from backend.dict_gpu._pcadi._projection.direction_cosines import (
        compute_direction_cosines,
    )
    from kikuchipy.signals.util._master_pattern import (
        _get_direction_cosines_for_fixed_pc,
    )

    nrows, ncols = detector.shape
    pcx = float(detector.pcx[0])
    pcy = float(detector.pcy[0])
    pcz = float(detector.pcz[0])
    tilt = float(detector.tilt)
    azimuthal = float(detector.azimuthal)
    sample_tilt = float(detector.sample_tilt)

    ref = _get_direction_cosines_for_fixed_pc(
        pcx=pcx, pcy=pcy, pcz=pcz, nrows=nrows, ncols=ncols,
        tilt=tilt, azimuthal=azimuthal, sample_tilt=sample_tilt,
        signal_mask=np.ones(nrows * ncols, dtype=bool),
    )
    if ref.ndim == 2:
        ref = ref.reshape(nrows, ncols, 3)
    ref = ref.astype(np.float32)

    ours = compute_direction_cosines(detector, device="cuda", dtype=torch.float32).cpu().numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5, rtol=1e-5)
