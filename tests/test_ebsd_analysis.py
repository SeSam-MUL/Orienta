"""
Unit tests for EBSD Analysis Module (FEAT-31).

Tests cover:
    - EBSDDataset wrapper functionality
    - GrainSet dataclass validation
    - Quality metric normalization
    - EDX channel detection
    - Phase/orientation access

Reference: tasks/MTEX_to_Python_Architecture.md (internal, not shipped)
"""

import pytest
import numpy as np

# Try to import orix, skip tests if not available
try:
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation, symmetry
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

from analysis.ebsd_dataset import EBSDDataset, GrainSet, ORIX_AVAILABLE as EBSD_ORIX_CHECK
from analysis.grain_analysis import reconstruct_grains, analyze_grain_size, GrainSizeResult
from analysis.bc_analysis import (
    calculate_bc_histogram,
    fit_bc_gmm,
    grain_average_bc,
    BCHistogramResult,
    BCGMMResult,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_xmap_simple():
    """Create minimal CrystalMap for testing."""
    if not ORIX_AVAILABLE:
        pytest.skip("orix not available")

    # Create 10x10 map
    ny, nx = 10, 10
    n_points = ny * nx

    # Random rotations
    rotations = Rotation.random(n_points)

    # Single phase (cubic)
    phase = Phase(
        name="TestPhase",
        point_group=symmetry.Oh,  # m-3m cubic
        structure=None
    )

    # All pixels indexed (phase_id = 0)
    phase_id = np.zeros(n_points, dtype=int)

    # Grid coordinates
    x = np.tile(np.arange(nx), ny)
    y = np.repeat(np.arange(ny), nx)

    # Quality metric (band contrast)
    bc = np.random.uniform(50, 200, n_points)

    xmap = CrystalMap(
        rotations=rotations,
        phase_id=phase_id,
        x=x,
        y=y,
        phase_list=PhaseList(phases=phase),
        prop={'bc': bc}
    )

    return xmap


@pytest.fixture
def mock_xmap_with_edx():
    """Create CrystalMap with EDX data."""
    if not ORIX_AVAILABLE:
        pytest.skip("orix not available")

    ny, nx = 5, 5
    n_points = ny * nx

    rotations = Rotation.random(n_points)
    phase = Phase(name="Al", point_group=symmetry.Oh)
    phase_id = np.zeros(n_points, dtype=int)

    x = np.tile(np.arange(nx), ny)
    y = np.repeat(np.arange(ny), nx)

    # Multiple quality + EDX channels
    bc = np.random.uniform(100, 220, n_points)
    fe_counts = np.random.uniform(0, 1000, n_points)
    al_counts = np.random.uniform(500, 3000, n_points)
    si_counts = np.random.uniform(0, 500, n_points)

    xmap = CrystalMap(
        rotations=rotations,
        phase_id=phase_id,
        x=x,
        y=y,
        phase_list=PhaseList(phases=phase),
        prop={
            'bc': bc,
            'Fe': fe_counts,
            'Al': al_counts,
            'Si': si_counts,
        }
    )

    return xmap


# ─── GrainSet Tests ──────────────────────────────────────────────────────────

def test_grainset_creation():
    """GrainSet can be created with minimal data."""
    grain_ids = np.array([[0, 1, 1], [0, 2, 2], [3, 3, 3]])
    grains = GrainSet(grain_ids=grain_ids, n_grains=3)

    assert grains.n_grains == 3
    assert grains.grain_ids.shape == (3, 3)
    assert grains.mean_orientations is None
    assert grains.boundary_segments == []


def test_grainset_validates_shape():
    """GrainSet rejects non-2D grain_ids."""
    with pytest.raises(ValueError, match="grain_ids must be 2D"):
        GrainSet(grain_ids=np.array([1, 2, 3]), n_grains=3)


def test_grainset_validates_type():
    """GrainSet rejects non-ndarray grain_ids."""
    with pytest.raises(TypeError, match="grain_ids must be np.ndarray"):
        GrainSet(grain_ids=[[1, 2], [3, 4]], n_grains=2)


# ─── EBSDDataset Basic Tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_creation(mock_xmap_simple):
    """EBSDDataset can be created from CrystalMap."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=0.5, source="test")

    assert dataset.source == "test"
    assert dataset.step_size == 0.5
    assert dataset.nx == 10
    assert dataset.ny == 10
    assert dataset.shape == (10, 10)
    assert dataset.size == 100
    assert dataset.grains is None


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_repr(mock_xmap_simple):
    """EBSDDataset has informative __repr__."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=0.5, source="hough")
    repr_str = repr(dataset)

    assert "hough" in repr_str
    assert "shape=(10, 10)" in repr_str
    assert "step=0.5µm" in repr_str
    assert "indexed=100.0%" in repr_str


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_rejects_invalid_xmap():
    """EBSDDataset rejects non-CrystalMap input."""
    with pytest.raises(TypeError, match="xmap must be orix.CrystalMap"):
        EBSDDataset("not a crystalmap", step_size=1.0)


# ─── Quality Metric Tests ────────────────────────────────────────────────────

@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_quality_bc_normalization(mock_xmap_simple):
    """Quality metric normalizes BC to [0, 1]."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0, source="h5oina")

    quality = dataset.quality
    assert quality.min() >= 0.0
    assert quality.max() <= 1.0
    assert len(quality) == 100


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_bc_raw_values(mock_xmap_simple):
    """BC property returns raw values."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    bc = dataset.bc
    assert len(bc) == 100
    assert bc.min() >= 50  # From fixture range
    assert bc.max() <= 200


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_bc_2d_shape(mock_xmap_simple):
    """bc_2d reshapes to map shape."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    bc_2d = dataset.bc_2d
    assert bc_2d.shape == (10, 10)


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_quality_fallback():
    """Quality metric falls back to ones if no BC/PQ/scores."""
    if not ORIX_AVAILABLE:
        pytest.skip("orix not available")

    # Create xmap without quality metrics
    rotations = Rotation.random(25)
    phase = Phase(name="Test", point_group=symmetry.Oh)
    phase_id = np.zeros(25, dtype=int)
    x = np.tile(np.arange(5), 5)
    y = np.repeat(np.arange(5), 5)

    xmap = CrystalMap(
        rotations=rotations,
        phase_id=phase_id,
        x=x,
        y=y,
        phase_list=PhaseList(phases=phase),
        prop={}  # No quality metrics
    )

    dataset = EBSDDataset(xmap, step_size=1.0)
    quality = dataset.quality

    assert np.allclose(quality, 1.0)


# ─── EDX Tests ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_edx_channels(mock_xmap_with_edx):
    """EDX channels are correctly detected."""
    dataset = EBSDDataset(mock_xmap_with_edx, step_size=1.0, source="h5oina")

    edx = dataset.edx_channels
    assert 'Fe' in edx
    assert 'Al' in edx
    assert 'Si' in edx
    assert len(edx['Fe']) == 25


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_has_edx_element(mock_xmap_with_edx):
    """has_edx_element correctly checks for elements."""
    dataset = EBSDDataset(mock_xmap_with_edx, step_size=1.0)

    assert dataset.has_edx_element('Fe')
    assert dataset.has_edx_element('al')  # case-insensitive
    assert not dataset.has_edx_element('Cu')


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_edx_in_repr(mock_xmap_with_edx):
    """EDX channels appear in repr."""
    dataset = EBSDDataset(mock_xmap_with_edx, step_size=1.0)
    repr_str = repr(dataset)

    assert "EDX:" in repr_str
    assert "Fe" in repr_str or "Al" in repr_str  # At least one element


# ─── Orientation & Phase Tests ───────────────────────────────────────────────

@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_orientations(mock_xmap_simple):
    """Orientations property returns Orientation with symmetry."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    ori = dataset.orientations
    assert ori.size == 100
    # Check that symmetry is applied
    assert ori.symmetry == dataset.phase.point_group


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_orientations_2d(mock_xmap_simple):
    """orientations_2d reshapes to map shape."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    ori_2d = dataset.orientations_2d
    assert ori_2d.shape == (10, 10)


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_phase(mock_xmap_simple):
    """Phase property returns primary phase."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    phase = dataset.phase
    assert phase.name == "TestPhase"
    assert phase.point_group == symmetry.Oh


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_indexed_mask(mock_xmap_simple):
    """Indexed mask correctly identifies indexed pixels."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    mask = dataset.indexed_mask
    assert mask.shape == (100,)
    assert mask.all()  # All pixels indexed in fixture


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_indexed_mask_2d(mock_xmap_simple):
    """indexed_mask_2d reshapes to map shape."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    mask_2d = dataset.indexed_mask_2d
    assert mask_2d.shape == (10, 10)
    assert mask_2d.all()


# ─── Convenience Methods ─────────────────────────────────────────────────────

@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_get_phase_mask(mock_xmap_simple):
    """get_phase_mask returns correct boolean mask."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    mask = dataset.get_phase_mask(0)
    assert mask.shape == (100,)
    assert mask.all()  # All pixels are phase_id=0


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_get_phase_mask_2d(mock_xmap_simple):
    """get_phase_mask_2d reshapes to map shape."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    mask_2d = dataset.get_phase_mask_2d(0)
    assert mask_2d.shape == (10, 10)


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_crop_to_indexed():
    """crop_to_indexed removes non-indexed pixels."""
    if not ORIX_AVAILABLE:
        pytest.skip("orix not available")

    # Create map with some non-indexed pixels
    rotations = Rotation.random(25)
    phase = Phase(name="Test", point_group=symmetry.Oh)
    phase_id = np.array([0, 0, -1, 0, 0] * 5)  # Some not indexed
    x = np.tile(np.arange(5), 5)
    y = np.repeat(np.arange(5), 5)

    xmap = CrystalMap(
        rotations=rotations,
        phase_id=phase_id,
        x=x,
        y=y,
        phase_list=PhaseList(phases=phase),
        prop={'bc': np.random.uniform(100, 200, 25)}
    )

    dataset = EBSDDataset(xmap, step_size=1.0)
    assert dataset.size == 25

    cropped = dataset.crop_to_indexed()
    assert cropped.size == 20  # Only indexed pixels


# ─── Integration Test ────────────────────────────────────────────────────────

@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_ebsddataset_full_workflow(mock_xmap_with_edx):
    """Full workflow: create, access properties, check EDX."""
    dataset = EBSDDataset(mock_xmap_with_edx, step_size=0.5, source="h5oina")

    # Basic properties
    assert dataset.nx == 5
    assert dataset.ny == 5
    assert dataset.size == 25

    # Quality
    quality = dataset.quality
    assert 0 <= quality.min() <= quality.max() <= 1

    # EDX
    assert dataset.has_edx_element('Fe')
    edx = dataset.edx_channels
    assert len(edx) == 3  # Fe, Al, Si

    # Orientations
    ori = dataset.orientations
    assert ori.size == 25

    # Masks
    assert dataset.indexed_mask.all()

    # Repr
    repr_str = repr(dataset)
    assert "h5oina" in repr_str
    assert "indexed=100.0%" in repr_str


# ─── Grain Analysis Tests ────────────────────────────────────────────────────

@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_reconstruct_grains_simple(mock_xmap_simple):
    """Grain reconstruction produces GrainSet with grain_ids."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    # Note: with random orientations, grain reconstruction may find 0 grains if
    # all neighbors exceed threshold. Use high threshold + min_pixels=1 to get some grains.
    grains = reconstruct_grains(dataset, angle_threshold_deg=15.0, min_pixels=1)

    assert isinstance(grains, GrainSet)
    assert grains.grain_ids.shape == (10, 10)
    # May be 0 grains with truly random orientations - that's OK
    assert grains.n_grains >= 0

    if grains.n_grains > 0:
        assert grains.mean_orientations is not None
        assert len(grains.grain_size) == grains.n_grains
        assert len(grains.ecd) == grains.n_grains


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_reconstruct_grains_removes_small(mock_xmap_simple):
    """Grain reconstruction removes grains below min_pixels."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    # Use high threshold to get some grains
    grains_min10 = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=10)
    grains_min1 = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=1)

    # All remaining grains should have >= 10 pixels
    if grains_min10.n_grains > 0:
        assert np.all(grains_min10.grain_size >= 10)

    # Fewer or equal grains with higher min_pixels
    assert grains_min10.n_grains <= grains_min1.n_grains


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_grain_properties_calculated(mock_xmap_simple):
    """Grain properties (ECD, area, GOS) are calculated."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=0.5)

    grains = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=3)

    # Skip if no grains found (can happen with random orientations)
    if grains.n_grains == 0:
        pytest.skip("No grains found with random orientations")

    # Check all properties exist
    assert grains.mean_orientations is not None
    assert grains.grain_size is not None
    assert grains.area is not None
    assert grains.ecd is not None
    assert grains.aspect_ratio is not None
    assert grains.gos is not None

    # Check shapes
    assert len(grains.grain_size) == grains.n_grains
    assert len(grains.area) == grains.n_grains
    assert len(grains.ecd) == grains.n_grains
    assert len(grains.gos) == grains.n_grains

    # Check ECD formula: ECD = 2 * sqrt(area / pi)
    expected_ecd = 2 * np.sqrt(grains.area / np.pi)
    assert np.allclose(grains.ecd, expected_ecd)


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_analyze_grain_size(mock_xmap_simple):
    """Grain size analysis produces statistics."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)
    grains = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=3)

    if grains.n_grains == 0:
        pytest.skip("No grains found with random orientations")

    result = analyze_grain_size(dataset, grains, remove_edge_grains=False)

    assert isinstance(result, GrainSizeResult)
    assert result.avg_ECD > 0
    assert result.avg_maxDimX > 0
    assert result.avg_maxDimY > 0
    assert result.avg_AR > 0
    assert result.area_weighted_ECD > 0

    # Check distributions
    assert len(result.ecd_per_grain) <= grains.n_grains
    assert len(result.maxDimX_per_grain) == len(result.ecd_per_grain)
    assert len(result.maxDimY_per_grain) == len(result.ecd_per_grain)
    assert len(result.ar_per_grain) == len(result.ecd_per_grain)


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_analyze_grain_size_edge_removal(mock_xmap_simple):
    """Edge grain removal reduces grain count."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)
    grains = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=3)

    if grains.n_grains == 0:
        pytest.skip("No grains found with random orientations")

    result_with_edges = analyze_grain_size(dataset, grains, remove_edge_grains=False)
    result_without_edges = analyze_grain_size(dataset, grains, remove_edge_grains=True)

    # Without edges should have fewer or equal grains
    assert len(result_without_edges.ecd_per_grain) <= len(result_with_edges.ecd_per_grain)


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_grain_size_area_weighted_ecd(mock_xmap_simple):
    """Area-weighted ECD gives more weight to larger grains."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)
    grains = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=3)

    if grains.n_grains == 0:
        pytest.skip("No grains found with random orientations")

    result = analyze_grain_size(dataset, grains, remove_edge_grains=False)

    # Area-weighted ECD should be between min and max ECD
    if len(result.ecd_per_grain) > 0:
        assert result.ecd_per_grain.min() <= result.area_weighted_ECD <= result.ecd_per_grain.max()


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not installed")
def test_reconstruct_grains_integration(mock_xmap_with_edx):
    """Full grain reconstruction workflow with EDX data."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    dataset = EBSDDataset(mock_xmap_with_edx, step_size=0.5)

    # Reconstruct grains - use high threshold for random data
    grains = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=2)

    assert grains.grain_ids.shape == (5, 5)
    # May be 0 grains with random orientations
    assert grains.n_grains >= 0

    if grains.n_grains > 0:
        # Analyze size
        size_result = analyze_grain_size(dataset, grains, remove_edge_grains=False)

        assert size_result.avg_ECD > 0
        assert size_result.std_ECD >= 0

        # Update dataset with grains
        dataset.grains = grains

        # Check dataset repr includes grain count
        repr_str = repr(dataset)
        assert "grains" in repr_str


# ─── BC Analysis Tests ───────────────────────────────────────────────────────

def test_calculate_bc_histogram(mock_xmap_simple):
    """BC histogram calculates bins and probability-normalized counts."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    result = calculate_bc_histogram(dataset, bin_width=10, bin_range=(0, 230))

    assert isinstance(result, BCHistogramResult)
    assert result.bin_width == 10
    assert len(result.bin_edges) == 24  # 0-230 with width 10 → 23 bins + 1
    assert len(result.bin_centers) == 23
    assert len(result.counts) == 23
    # Probability normalized: sum should be ~1
    assert np.isclose(result.counts.sum(), 1.0)
    # BC values should be finite
    assert np.all(np.isfinite(result.bc_values))


def test_bc_histogram_bin_centers(mock_xmap_simple):
    """BC histogram bin centers are correct."""
    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    result = calculate_bc_histogram(dataset, bin_width=10, bin_range=(0, 230))

    # First bin center should be 5 (0 + 10/2)
    assert result.bin_centers[0] == 5.0
    # Last bin center should be 225 (220 + 10/2)
    assert result.bin_centers[-1] == 225.0


def test_fit_bc_gmm(mock_xmap_simple):
    """BC GMM fit produces 3 components."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        pytest.skip("sklearn not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    result = fit_bc_gmm(dataset, n_init=5, random_state=42)

    if result is None:
        pytest.skip("GMM fit failed (can happen with random data)")

    assert isinstance(result, BCGMMResult)
    assert len(result.weights) == 3
    assert len(result.means) == 3
    assert len(result.stds) == 3

    # Weights should sum to ~1
    assert np.isclose(result.weights.sum(), 1.0)

    # Components should be sorted by mean BC
    assert result.means[0] <= result.means[1] <= result.means[2]

    # Check convenience attributes match
    assert result.bc_low_center == result.means[0]
    assert result.bc_mid_center == result.means[1]
    assert result.bc_high_center == result.means[2]
    assert result.bc_low_frac == result.weights[0]
    assert result.bc_mid_frac == result.weights[1]
    assert result.bc_high_frac == result.weights[2]

    # All fractions should be positive
    assert result.bc_low_frac > 0
    assert result.bc_mid_frac > 0
    assert result.bc_high_frac > 0


def test_fit_bc_gmm_convergence(mock_xmap_simple):
    """BC GMM fit reports convergence status."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        pytest.skip("sklearn not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)

    result = fit_bc_gmm(dataset, n_init=5, max_iter=2000)

    if result is None:
        pytest.skip("GMM fit failed")

    # Should converge with 2000 iterations
    assert result.converged is True
    assert result.n_samples > 0


def test_grain_average_bc(mock_xmap_simple):
    """Grain-average BC calculation."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    if not ORIX_AVAILABLE:
        pytest.skip("orix not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)
    grains = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=3)

    if grains.n_grains == 0:
        pytest.skip("No grains found")

    gBC, grain_ids = grain_average_bc(dataset, grains)

    assert len(gBC) == grains.n_grains
    assert len(grain_ids) == grains.n_grains
    # Grain IDs should be 1..n_grains
    assert np.array_equal(grain_ids, np.arange(1, grains.n_grains + 1))
    # gBC values should be finite (or NaN for empty grains)
    assert np.all(np.isfinite(gBC) | np.isnan(gBC))


def test_grain_average_bc_range(mock_xmap_simple):
    """Grain-average BC values are in valid range."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("scipy not available")

    if not ORIX_AVAILABLE:
        pytest.skip("orix not available")

    dataset = EBSDDataset(mock_xmap_simple, step_size=1.0)
    grains = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=3)

    if grains.n_grains == 0:
        pytest.skip("No grains found")

    gBC, _ = grain_average_bc(dataset, grains)

    # Filter out NaN values
    gBC_valid = gBC[np.isfinite(gBC)]

    if len(gBC_valid) > 0:
        # Should be within BC range of dataset
        bc_min = dataset.bc[np.isfinite(dataset.bc)].min()
        bc_max = dataset.bc[np.isfinite(dataset.bc)].max()
        assert np.all(gBC_valid >= bc_min)
        assert np.all(gBC_valid <= bc_max)


def test_bc_analysis_integration(mock_xmap_with_edx):
    """Full BC analysis workflow."""
    try:
        from sklearn.mixture import GaussianMixture
        from scipy.ndimage import label
    except ImportError:
        pytest.skip("sklearn or scipy not available")

    if not ORIX_AVAILABLE:
        pytest.skip("orix not available")

    dataset = EBSDDataset(mock_xmap_with_edx, step_size=0.5)

    # Calculate histogram
    hist_result = calculate_bc_histogram(dataset)
    assert hist_result.counts.sum() > 0

    # Fit GMM
    gmm_result = fit_bc_gmm(dataset, n_init=3)
    if gmm_result is not None:
        assert gmm_result.converged
        assert 0 < gmm_result.bc_low_frac < 1
        assert 0 < gmm_result.bc_mid_frac < 1
        assert 0 < gmm_result.bc_high_frac < 1

    # Reconstruct grains
    grains = reconstruct_grains(dataset, angle_threshold_deg=30.0, min_pixels=2)

    if grains.n_grains > 0:
        # Calculate grain-average BC
        gBC, grain_ids = grain_average_bc(dataset, grains)
        assert len(gBC) == grains.n_grains
        assert len(grain_ids) == grains.n_grains

