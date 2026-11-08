"""Per-grain values: sizes from the grain area, orientation values from measurements."""
import numpy as np
import pytest
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation

from analysis.ebsd_dataset import EBSDDataset, GrainSet
from analysis.grain_analysis import calculate_grain_properties, reconstruct_grains
from analysis.orientation_math import disorientation_angle, proper_ops
from orix.quaternion.symmetry import Oh
from tests.synthetic_grain_map import add_defects, make_map, to_xmap


def _ds(m):
    return EBSDDataset(to_xmap(m), step_size=m["step"], source="synthetic")


def _block_map(ny=12, nx=16, blocks=(), unindexed=()):
    """Background orientation everywhere, each (y0, y1, x0, x1, deg) block rotated about z.

    ``unindexed``: flat pixel indices built in as phase -1 (orix's ``phase_id``
    property returns a copy, so it cannot be changed after construction).
    """
    q = np.tile(Rotation.identity(1).data, (ny * nx, 1)).reshape(ny, nx, 4)
    for y0, y1, x0, x1, deg in blocks:
        q[y0:y1, x0:x1] = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(deg)).data[0]
    pid = np.zeros(ny * nx, int)
    pid[list(unindexed)] = -1
    xmap = CrystalMap(rotations=Rotation(q.reshape(-1, 4)), phase_id=pid,
                      x=np.tile(np.arange(nx, dtype=float), ny), y=np.repeat(np.arange(ny, dtype=float), nx),
                      phase_list=PhaseList(names=["Al"], point_groups=["m-3m"]))
    return EBSDDataset(xmap, step_size=1.0)


def _grain_of(grains, y, x):
    return int(grains.grain_ids[y, x]) - 1


def test_aspect_ratio_of_a_rectangle_is_length_over_width():
    ds = _block_map(blocks=[(4, 6, 3, 9, 30.0)])          # 2 rows x 6 columns
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    i = _grain_of(g, 4, 3)
    assert g.aspect_ratio[i] == pytest.approx(3.0, rel=1e-9)
    assert np.all(g.aspect_ratio >= 1.0)


def test_perimeter_and_area_of_an_interior_block():
    ds = _block_map(blocks=[(3, 6, 5, 9, 30.0)])          # 3 x 4 block
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    i = _grain_of(g, 3, 5)
    assert g.perimeter[i] == pytest.approx(2 * (3 + 4))
    assert g.area[i] == pytest.approx(12.0)
    assert g.ecd[i] == pytest.approx(2 * np.sqrt(12.0 / np.pi))


def test_area_counts_a_closed_hole_but_grain_size_does_not():
    ds = _block_map(ny=10, nx=10, unindexed=[55])           # one not-indexed pixel
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    assert g.n_grains == 1
    assert g.grain_size[0] == 99 and g.n_closed[0] == 1
    assert g.area[0] == pytest.approx(100.0)       # the hole's area is shared by the cells around it
    # a covered hole has no boundary of its own: both perimeters are the map border
    assert g.perimeter[0] == pytest.approx(40.0) and g.perimeter_with_inclusions[0] == pytest.approx(40.0)


def test_gos_of_a_tight_grain_on_a_symmetry_branch():
    """Grain 3 sits at phi2 = 45 deg; its pixels are stored on random symmetry branches,
    as files store reduced orientations. The old mean broke on exactly this."""
    from analysis.orientation_math import qmul
    m = make_map()
    ops = proper_ops(Oh)
    sel = m["truth"] == 3
    rng = np.random.default_rng(1)
    q = m["quats"].copy()
    q[sel] = qmul(ops[rng.integers(0, len(ops), sel.sum())], q[sel])
    ds = _ds({**m, "quats": q})
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    i = int(np.bincount(g.grain_ids[sel]).argmax()) - 1
    assert np.degrees(g.gos[i]) < 0.5
    # expectation from the generator, independent of the code under test:
    # the noiseless orientations of the grain lie on ONE branch, so a plain average is the truth
    clean = m["clean"][sel]
    expected = clean.mean(axis=0)
    expected /= np.linalg.norm(expected)
    mean = g.mean_orientations.data[i:i + 1]
    assert np.degrees(disorientation_angle(mean, expected[None], ops))[0] < 0.1


def test_gos_never_exceeds_the_largest_pair_angle():
    d = add_defects(make_map())
    ds = _ds(d)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, fill=False)
    q = d["quats"]
    meas = (g.grain_ids > 0) & ~g.closed_mask
    for i in range(g.n_grains):
        px = q[(g.grain_ids == i + 1) & meas]
        if len(px) < 2 or len(px) > 400:
            continue
        ops = proper_ops(Oh) if g.grain_phase[i] == 0 else proper_ops(ds.xmap.phases[1].point_group)
        pair = max(disorientation_angle(np.broadcast_to(p, px.shape), px, ops).max() for p in px)
        assert g.gos[i] <= pair + 1e-9


def test_no_grain_holds_two_phases_and_grain_phase_matches():
    d = add_defects(make_map())
    g = reconstruct_grains(_ds(d), angle_threshold_deg=5.0, min_pixels=3, fill=False)
    ph = d["phase_id"]
    meas = (g.grain_ids > 0) & ~g.closed_mask
    for i in range(g.n_grains):
        phases = np.unique(ph[(g.grain_ids == i + 1) & meas])
        assert phases.tolist() == [g.grain_phase[i]]


def test_sigma3_boundary_angle_and_no_phase_boundary_angles():
    m = make_map()
    g = reconstruct_grains(_ds(m), angle_threshold_deg=5.0, min_pixels=1, fill=False)
    angles = np.asarray(g.boundary_type)
    assert np.any(np.abs(angles - 60.0) < 0.5)              # grains 1 | 2
    # same-phase boundaries only, each reduced by its own group: cubic <= 62.8, tetragonal <= 98.42
    assert angles.max() <= 98.42 + 1e-6


def test_legacy_grainset_without_new_fields():
    m = make_map()
    ds = _ds(m)
    g = GrainSet(grain_ids=m["truth"].copy(), n_grains=int(m["truth"].max()))
    g = calculate_grain_properties(ds, g)
    assert len(g.gos) == g.n_grains and np.all(np.isfinite(g.gos))
    # no cells given: every pixel owns its square, so the area is the pixel count times the step squared
    assert g.cells is not None and g.cells.generator.all()
    np.testing.assert_allclose(g.area, np.bincount(m["truth"].ravel())[1:] * m["step"] ** 2, rtol=1e-12)
    assert len(g.boundary_type) == len(g.boundary_length) == len(g.boundary_pairs) > 0
    assert (g.boundary_pairs[:, 0] < g.boundary_pairs[:, 1]).all()


@pytest.mark.parametrize("kw", [{"angle_threshold_deg": 0.0}, {"alpha": -0.1}, {"min_pixels": 0}])
def test_reconstruct_grains_rejects_invalid_values(kw):
    with pytest.raises(ValueError):
        reconstruct_grains(_ds(make_map()), **{"fill": False, **kw})


def test_a_label_above_n_grains_is_refused():
    m = make_map()
    g = GrainSet(grain_ids=m["truth"].copy(), n_grains=int(m["truth"].max()) - 1)
    with pytest.raises(ValueError, match="label above n_grains"):
        calculate_grain_properties(_ds(m), g)


def _scaled(ds, step):
    return EBSDDataset(ds.xmap, step_size=step)


def test_lengths_areas_and_boundary_lengths_scale_with_the_step():
    ds = _scaled(_block_map(blocks=[(3, 6, 5, 9, 30.0)]), 2.0)          # 3 x 4 block, 2 um pixels
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    i = _grain_of(g, 3, 5)
    assert g.area[i] == pytest.approx(12 * 4.0)
    assert g.perimeter[i] == pytest.approx(14 * 2.0)
    assert g.ecd[i] == pytest.approx(2 * np.sqrt(48.0 / np.pi))
    # the block touches the background along 14 pixel edges: 14 segments of 2 um, 30 deg about z
    assert len(g.boundary_type) == len(g.boundary_length) == 14
    assert g.boundary_length.sum() == pytest.approx(28.0)
    np.testing.assert_allclose(g.boundary_type, 30.0, atol=1e-9)


def test_a_slender_grain_has_the_exact_aspect_ratio():
    """2 rows x 400 columns: second moments 400^2/12 and 2^2/12, so the aspect ratio is exactly 200."""
    ds = _block_map(ny=6, nx=420, blocks=[(2, 4, 10, 410, 30.0)])
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    assert g.aspect_ratio[_grain_of(g, 2, 10)] == pytest.approx(200.0, rel=1e-12)


def test_perimeter_with_inclusions_adds_the_loop_of_an_open_hole():
    """alpha 0 leaves the 8 x 8 not-indexed block open: its 32 edges are an inner loop of the one grain."""
    ny = nx = 24
    hole = [y * nx + x for y in range(8, 16) for x in range(8, 16)]
    g = reconstruct_grains(_block_map(ny=ny, nx=nx, unindexed=hole), angle_threshold_deg=5.0,
                           min_pixels=1, alpha=0.0, fill=False)
    assert g.n_grains == 1
    assert g.perimeter[0] == pytest.approx(96.0)                    # the map border
    assert g.perimeter_with_inclusions[0] == pytest.approx(96.0 + 32.0)
    assert g.area[0] == pytest.approx(24 * 24 - 64.0) and g.n_closed[0] == 0


def test_boundary_segments_sum_the_edges_of_a_pair():
    """Two cells can share several edges; the pair is one segment with their summed length."""
    from dataclasses import replace

    from analysis.grain_analysis import _boundary_segments
    from analysis.grain_cells import build_cells

    cells = build_cells(np.ones((1, 2), dtype=bool))
    shared = np.flatnonzero((cells.edge_cells == [0, 1]).all(axis=1))
    assert len(shared) == 1
    cells = replace(cells,
                    edge_cells=np.vstack([cells.edge_cells, [[0, 1]]]),
                    edge_length=np.append(cells.edge_length, 0.5),
                    edge_xy=np.vstack([cells.edge_xy, cells.edge_xy[shared]]),
                    edge_vertex=np.vstack([cells.edge_vertex, cells.edge_vertex[shared]]))
    q = np.array([[1.0, 0, 0, 0], [np.cos(np.deg2rad(10)), 0, 0, np.sin(np.deg2rad(10))]])
    pairs, angle, length = _boundary_segments(cells, np.array([1, 2]), np.array([True, True]),
                                              np.array([0, 0]), q, {0: proper_ops(Oh)}, 3.0)
    np.testing.assert_array_equal(pairs, [[0, 1]])
    assert length == pytest.approx([(1.0 + 0.5) * 3.0])
    assert angle == pytest.approx([20.0])


def test_boundary_metrics_sum_the_segment_lengths_and_fall_back_to_one_step():
    from analysis.deformation_analysis import calculate_gb_metrics, calculate_gb_segment_histogram

    ds = _scaled(_block_map(blocks=[(3, 6, 5, 9, 30.0)]), 2.0)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    assert calculate_gb_metrics(g, ds)["hagb_length"] == pytest.approx(28.0)       # 14 segments of 2 um
    g.boundary_length = g.boundary_length * 0.5                                      # lengths now matter
    assert calculate_gb_metrics(g, ds)["hagb_length"] == pytest.approx(14.0)
    assert calculate_gb_metrics(g, ds)["sagb_length"] == 0.0
    counts, _ = calculate_gb_segment_histogram(g, ds)
    assert counts.sum() == pytest.approx(14.0)
    g.boundary_length = None                                                         # none: one step each
    assert calculate_gb_metrics(g, ds)["hagb_length"] == pytest.approx(14 * 2.0)
    counts, _ = calculate_gb_segment_histogram(g, ds)
    assert counts.sum() == pytest.approx(14 * 2.0)
