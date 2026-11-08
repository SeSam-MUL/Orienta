"""Tests for analysis.smoothing module."""

import numpy as np
import pytest

try:
    from orix.quaternion import Orientation, Rotation
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from diffpy.structure import Lattice, Structure
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not available")


def _make_test_dataset(ny=5, nx=5, noise_deg=1.0):
    """Create a small synthetic EBSDDataset for smoothing tests.

    Returns a 5x5 map with a single grain (uniform orientation + noise).
    """
    from analysis.ebsd_dataset import EBSDDataset

    phase = Phase(
        name="Al",
        space_group=225,
        structure=Structure(lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)),
    )

    # Base orientation (identity rotation)
    n_pixels = ny * nx
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.tile(base_quat, (n_pixels, 1))

    # Add small random noise
    rng = np.random.default_rng(42)
    noise_rad = np.deg2rad(noise_deg)
    for i in range(n_pixels):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, noise_rad)
        q_noise = np.array([
            np.cos(angle / 2),
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2),
        ])
        # Quaternion multiplication
        a, b = quats[i], q_noise
        quats[i] = np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ])
        quats[i] /= np.linalg.norm(quats[i])

    rotations = Rotation(quats)
    x = np.tile(np.arange(nx), ny).astype(float)
    y = np.repeat(np.arange(ny), nx).astype(float)

    xmap = CrystalMap(
        rotations=rotations,
        phase_id=np.zeros(n_pixels, dtype=int),
        x=x,
        y=y,
        phase_list=PhaseList([phase]),
        prop={"bc": rng.integers(100, 230, size=n_pixels).astype(float)},
    )

    return EBSDDataset(xmap, step_size=0.5, source="test")


def _make_bicrystal_dataset(ny=6, nx=6):
    """Create a 6x6 map with two grains (left half vs right half, 30 degrees apart)."""
    from analysis.ebsd_dataset import EBSDDataset

    phase = Phase(
        name="Al",
        space_group=225,
        structure=Structure(lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)),
    )

    n_pixels = ny * nx
    quats = np.zeros((n_pixels, 4))

    # Grain 1: identity
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    # Grain 2: 30 degrees around z-axis
    angle = np.deg2rad(30)
    q2 = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

    for row in range(ny):
        for col in range(nx):
            idx = row * nx + col
            quats[idx] = q1 if col < nx // 2 else q2

    rotations = Rotation(quats)
    x = np.tile(np.arange(nx), ny).astype(float)
    y = np.repeat(np.arange(ny), nx).astype(float)

    xmap = CrystalMap(
        rotations=rotations,
        phase_id=np.zeros(n_pixels, dtype=int),
        x=x,
        y=y,
        phase_list=PhaseList([phase]),
        prop={"bc": np.full(n_pixels, 200.0)},
    )

    return EBSDDataset(xmap, step_size=1.0, source="test")


class TestKernelHelpers:
    def test_kernel_offsets_3x3(self):
        from analysis.smoothing import _kernel_offsets
        offsets = _kernel_offsets(3)
        assert len(offsets) == 8  # 3x3 minus center

    def test_kernel_offsets_5x5(self):
        from analysis.smoothing import _kernel_offsets
        offsets = _kernel_offsets(5)
        assert len(offsets) == 24  # 5x5 minus center

    def test_overlap_slices(self):
        from analysis.smoothing import _overlap_slices
        src_y, src_x, dst_y, dst_x = _overlap_slices(1, 0, 5, 5)
        assert src_y == slice(0, 4)
        assert dst_y == slice(1, 5)

    def test_overlap_slices_negative(self):
        from analysis.smoothing import _overlap_slices
        src_y, src_x, dst_y, dst_x = _overlap_slices(-1, -1, 5, 5)
        assert src_y == slice(1, 5)
        assert dst_y == slice(0, 4)


class TestMedianOrientationFilter:
    def test_reduces_noise(self):
        """Smoothing should reduce orientation spread in a single grain."""
        from analysis.smoothing import median_orientation_filter

        dataset = _make_test_dataset(noise_deg=2.0)

        # Measure spread before
        ori_before = dataset.orientations
        ref = Orientation(ori_before.data[0], symmetry=dataset.phase.point_group)
        from orix.quaternion import Misorientation
        misor_before = Misorientation(
            ~ref * ori_before, symmetry=(dataset.phase.point_group,) * 2
        )
        spread_before = np.mean(misor_before.angle.data)

        # Apply filter
        smoothed_quats = median_orientation_filter(dataset, kernel_size=3)

        # Measure spread after
        ori_after = Orientation(smoothed_quats, symmetry=dataset.phase.point_group)
        misor_after = Misorientation(
            ~ref * ori_after, symmetry=(dataset.phase.point_group,) * 2
        )
        spread_after = np.mean(misor_after.angle.data)

        assert spread_after <= spread_before

    def test_output_shape(self):
        from analysis.smoothing import median_orientation_filter
        dataset = _make_test_dataset()
        result = median_orientation_filter(dataset)
        assert result.shape == (dataset.size, 4)

    def test_preserves_grain_boundary(self):
        """Filter should NOT smooth across a 30-degree grain boundary."""
        from analysis.smoothing import median_orientation_filter

        dataset = _make_bicrystal_dataset()
        result = median_orientation_filter(dataset, max_misorientation_deg=5.0)

        # Check that left half and right half are still distinct
        ny, nx = dataset.shape
        left_quats = result[:nx // 2]   # first row, left half
        right_quats = result[nx // 2:nx]  # first row, right half

        # They should still be ~30 degrees apart
        left_ori = Orientation(left_quats[0], symmetry=dataset.phase.point_group)
        right_ori = Orientation(right_quats[0], symmetry=dataset.phase.point_group)
        from orix.quaternion import Misorientation
        misor = Misorientation(
            ~left_ori * right_ori,
            symmetry=(dataset.phase.point_group,) * 2,
        )
        angle_deg = np.rad2deg(misor.angle.data[0])
        assert angle_deg > 20.0  # Should still be large (boundary preserved)


class TestQualityWeightedMeanFilter:
    def test_output_shape(self):
        from analysis.smoothing import quality_weighted_mean_filter
        dataset = _make_test_dataset()
        result = quality_weighted_mean_filter(dataset)
        assert result.shape == (dataset.size, 4)

    def test_unit_quaternions(self):
        """All output quaternions should have unit norm."""
        from analysis.smoothing import quality_weighted_mean_filter
        dataset = _make_test_dataset()
        result = quality_weighted_mean_filter(dataset)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestApplySmoothing:
    def test_returns_dataset(self):
        from analysis.smoothing import apply_smoothing
        dataset = _make_test_dataset()
        result = apply_smoothing(dataset, method="median")
        from analysis.ebsd_dataset import EBSDDataset
        assert isinstance(result, EBSDDataset)

    def test_preserves_shape(self):
        from analysis.smoothing import apply_smoothing
        dataset = _make_test_dataset()
        result = apply_smoothing(dataset, method="median")
        assert result.shape == dataset.shape

    def test_preserves_step_size(self):
        from analysis.smoothing import apply_smoothing
        dataset = _make_test_dataset()
        result = apply_smoothing(dataset, method="weighted_mean")
        assert result.step_size == dataset.step_size

    def test_invalid_method_raises(self):
        from analysis.smoothing import apply_smoothing
        dataset = _make_test_dataset()
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            apply_smoothing(dataset, method="invalid")

    def test_weighted_mean_method(self):
        from analysis.smoothing import apply_smoothing
        dataset = _make_test_dataset()
        result = apply_smoothing(dataset, method="weighted_mean")
        assert result.size == dataset.size
