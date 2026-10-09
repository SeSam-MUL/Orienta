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


# (shape, pc, sample_tilt, tilt, azimuthal). The centred PC on a square detector
# cannot tell a flipped or swapped axis from the right one, so the others are
# off-centre, non-square and tilted.
_REFERENCE_DETECTORS = [
    ((60, 60), (0.5, 0.5, 0.5), 70.0, 0.0, 0.0),
    ((48, 64), (0.43, 0.61, 0.52), 70.0, 10.0, 0.0),
    ((60, 80), (0.55, 0.40, 0.65), 68.0, 5.0, 0.0),
    # Orienta keeps the EMsoft / kikuchipy 0.11 meaning of the azimuthal angle on
    # every path. kikuchipy >= 0.12.1 reads it the other way round, so the
    # reference is taken through ``for_kikuchipy_projection``, which hands the
    # installed kikuchipy the sign it expects (a no-op on 0.11).
    ((60, 80), (0.55, 0.40, 0.65), 68.0, 5.0, 7.0),
    ((60, 80), (0.55, 0.40, 0.65), 68.0, 5.0, -7.0),
    ((48, 64), (0.43, 0.61, 0.52), 70.0, 10.0, 3.0),
]


@pytest.mark.gpu
@pytest.mark.parametrize("shape,pc,sample_tilt,tilt,azimuthal", _REFERENCE_DETECTORS)
def test_direction_cosines_matches_kikuchipy_reference(
        shape, pc, sample_tilt, tilt, azimuthal):
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

    import kikuchipy as kp
    detector = kp.detectors.EBSDDetector(
        shape=shape, pc=pc, sample_tilt=sample_tilt, tilt=tilt, azimuthal=azimuthal
    )
    nrows, ncols = detector.shape
    from backend.api.services.detector_convention import for_kikuchipy_projection
    ref = _get_direction_cosines_from_detector(for_kikuchipy_projection(detector))
    if ref.ndim == 2:
        ref = ref.reshape(nrows, ncols, 3)
    ref = ref.astype(np.float32)

    ours = compute_direction_cosines(detector, device="cuda", dtype=torch.float32).cpu().numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5, rtol=1e-5)
