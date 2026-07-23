"""Tests for analysis.ebsd_dataset module."""

import numpy as np
import pytest

try:
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation, Orientation
    from diffpy.structure import Lattice, Structure
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not available")


def _make_xmap(ny=10, nx=10, with_bc=True, bc_key="bc"):
    """Create a minimal CrystalMap for testing."""
    n_pixels = ny * nx
    phase = Phase(
        name="Al",
        space_group=225,
        structure=Structure(lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)),
    )
    rotations = Rotation.identity(n_pixels)
    x = np.tile(np.arange(nx), ny).astype(float)
    y = np.repeat(np.arange(ny), nx).astype(float)

    prop = {}
    if with_bc:
        rng = np.random.default_rng(42)
        prop[bc_key] = rng.integers(50, 230, size=n_pixels).astype(float)

    return CrystalMap(
        rotations=rotations,
        phase_id=np.zeros(n_pixels, dtype=int),
        x=x, y=y,
        phase_list=PhaseList([phase]),
        prop=prop,
    )


class TestEBSDDatasetInit:
    def test_basic_creation(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap()
        ds = EBSDDataset(xmap, step_size=0.5, source="test")
        assert ds.step_size == 0.5
        assert ds.source == "test"

    def test_shape(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(ny=8, nx=12)
        ds = EBSDDataset(xmap, step_size=1.0)
        assert ds.shape == (8, 12)
        assert ds.ny == 8
        assert ds.nx == 12
        assert ds.size == 96

    def test_non_crystalmap_raises(self):
        from analysis.ebsd_dataset import EBSDDataset
        with pytest.raises(TypeError, match="orix.CrystalMap"):
            EBSDDataset("not a crystal map", step_size=1.0)

    def test_grains_initially_none(self):
        from analysis.ebsd_dataset import EBSDDataset
        ds = EBSDDataset(_make_xmap(), step_size=1.0)
        assert ds.grains is None


class TestQualityMetrics:
    def test_bc_property(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(ny=5, nx=5)
        ds = EBSDDataset(xmap, step_size=1.0)
        bc = ds.bc
        assert bc.shape == (25,)
        assert np.all(bc >= 50)
        assert np.all(bc <= 230)

    def test_quality_normalized(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(ny=5, nx=5)
        ds = EBSDDataset(xmap, step_size=1.0)
        q = ds.quality
        assert q.shape == (25,)
        assert np.all(q >= 0)
        assert np.all(q <= 1.0)

    def test_bc_2d(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(ny=4, nx=6)
        ds = EBSDDataset(xmap, step_size=1.0)
        bc_2d = ds.bc_2d
        assert bc_2d.shape == (4, 6)

    def test_fallback_without_bc(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(with_bc=False)
        ds = EBSDDataset(xmap, step_size=1.0)
        bc = ds.bc
        # Fallback should return 128
        assert np.all(bc == 128)

    def test_quality_fallback_without_bc(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(with_bc=False)
        ds = EBSDDataset(xmap, step_size=1.0)
        q = ds.quality
        assert np.all(q == 1.0)


class TestHasNativeBC:
    """has_native_bc gates the BC-calibrated GMM / quality-filter. It must key
    STRICTLY on native 'bc' — pq (Hough pattern quality) / scores (NCC match
    confidence) are NOT native Oxford Band Contrast."""

    def test_true_when_native_bc_matches_size(self):
        from analysis.ebsd_dataset import EBSDDataset
        ds = EBSDDataset(_make_xmap(ny=3, nx=2, with_bc=True, bc_key="bc"), step_size=1.0)
        assert ds.has_native_bc is True

    def test_false_when_bc_absent(self):
        from analysis.ebsd_dataset import EBSDDataset
        ds = EBSDDataset(_make_xmap(ny=3, nx=2, with_bc=False), step_size=1.0)
        assert ds.has_native_bc is False

    def test_false_when_only_pq(self):
        from analysis.ebsd_dataset import EBSDDataset
        ds = EBSDDataset(_make_xmap(ny=3, nx=2, with_bc=True, bc_key="pq"), step_size=1.0)
        assert ds.has_native_bc is False

    def test_false_when_bc_wrong_size(self):
        from types import SimpleNamespace
        from analysis.ebsd_dataset import EBSDDataset
        ds = EBSDDataset(_make_xmap(ny=3, nx=2, with_bc=False), step_size=1.0)
        # A native 'bc' whose length does not match the map size is not usable.
        ds.xmap = SimpleNamespace(prop={"bc": np.arange(3, dtype=float)}, size=6)
        assert ds.has_native_bc is False


class TestOrientations:
    def test_orientations_shape(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(ny=3, nx=4)
        ds = EBSDDataset(xmap, step_size=1.0)
        ori = ds.orientations
        assert ori.size == 12

    def test_phase_property(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap()
        ds = EBSDDataset(xmap, step_size=1.0)
        phase = ds.phase
        assert phase.name == "Al"


class TestIndexedMask:
    def test_all_indexed(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(ny=5, nx=5)
        ds = EBSDDataset(xmap, step_size=1.0)
        mask = ds.indexed_mask
        assert mask.shape == (25,)
        assert np.all(mask)  # All indexed since all phase_id=0

    def test_indexed_mask_2d(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(ny=4, nx=6)
        ds = EBSDDataset(xmap, step_size=1.0)
        mask_2d = ds.indexed_mask_2d
        assert mask_2d.shape == (4, 6)


class TestPhaseNames:
    def test_phase_names(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap()
        ds = EBSDDataset(xmap, step_size=1.0)
        names = ds.phase_names
        assert "Al" in names


class TestRepr:
    def test_repr_contains_info(self):
        from analysis.ebsd_dataset import EBSDDataset
        xmap = _make_xmap(ny=10, nx=10)
        ds = EBSDDataset(xmap, step_size=0.5, source="hough")
        r = repr(ds)
        assert "hough" in r
        assert "0.5" in r
        assert "10" in r


class TestGrainSet:
    def test_valid_grain_set(self):
        from analysis.ebsd_dataset import GrainSet
        ids = np.array([[1, 1, 2], [2, 3, 3]])
        gs = GrainSet(grain_ids=ids, n_grains=3)
        assert gs.n_grains == 3

    def test_non_ndarray_raises(self):
        from analysis.ebsd_dataset import GrainSet
        with pytest.raises(TypeError, match="np.ndarray"):
            GrainSet(grain_ids=[[1, 2], [3, 4]], n_grains=4)

    def test_1d_raises(self):
        from analysis.ebsd_dataset import GrainSet
        with pytest.raises(ValueError, match="2D"):
            GrainSet(grain_ids=np.array([1, 2, 3]), n_grains=3)
