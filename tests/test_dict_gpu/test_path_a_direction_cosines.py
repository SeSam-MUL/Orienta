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


def _kikuchipy_flips_azimuthal() -> bool:
    import kikuchipy
    from packaging.version import Version
    return Version(kikuchipy.__version__) >= Version("0.13")


# kikuchipy 0.13 reversed the meaning of the detector's azimuthal angle (0.11: the
# convention our projection uses; 0.13: "a positive angle means features on the
# detector appear to move toward the right"). Measured on both: our grid at
# azimuthal=+7 matches kikuchipy 0.11 at +7 and kikuchipy 0.13 at -7 (max
# deviation 1e-7), and is 14 deg (twice the azimuthal) off kikuchipy 0.13 at +7.
# Our GPU projection keeps the earlier (EMsoft) sign. Which sign matches vendor
# headers is undecided and left for a later release; until then this case
# documents the difference and fails loudly (strict) if the two ever agree, so
# the marker gets removed.
_AZIMUTHAL_SIGN_OPEN = pytest.mark.xfail(
    _kikuchipy_flips_azimuthal(), strict=True,
    reason=("kikuchipy >= 0.13 reversed the meaning of the azimuthal angle; "
            "Orienta's GPU projection keeps the earlier (EMsoft) sign; which one "
            "matches vendor headers is undecided and left for a later release"),
)

# (shape, pc, sample_tilt, tilt, azimuthal). The centred PC on a square detector
# cannot tell a flipped or swapped axis from the right one, so the others are
# off-centre, non-square and tilted.
_REFERENCE_DETECTORS = [
    ((60, 60), (0.5, 0.5, 0.5), 70.0, 0.0, 0.0),
    ((48, 64), (0.43, 0.61, 0.52), 70.0, 10.0, 0.0),
    ((60, 80), (0.55, 0.40, 0.65), 68.0, 5.0, 0.0),
    pytest.param((60, 80), (0.55, 0.40, 0.65), 68.0, 5.0, 7.0,
                 marks=_AZIMUTHAL_SIGN_OPEN),
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
    ref = _get_direction_cosines_from_detector(detector)
    if ref.ndim == 2:
        ref = ref.reshape(nrows, ncols, 3)
    ref = ref.astype(np.float32)

    ours = compute_direction_cosines(detector, device="cuda", dtype=torch.float32).cpu().numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5, rtol=1e-5)
