"""KAM: same phase, measured, and - once grains exist - same grain."""
import numpy as np
from orix.quaternion import Misorientation, Orientation

from analysis.deformation_analysis import calculate_kam
from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from tests.synthetic_grain_map import add_defects, make_map, to_xmap


def _ds(m):
    return EBSDDataset(to_xmap(m), step_size=m["step"])


def _orix_kam(m, order, thr_deg, phase_filter=0):
    """Independent reference with orix, single phase, every pixel measured."""
    q, ph = m["quats"], m["phase_id"]
    ny, nx = ph.shape
    sym = to_xmap(m).phases[phase_filter].point_group
    acc, cnt = np.zeros((ny, nx)), np.zeros((ny, nx))
    for dy in range(-order, order + 1):
        for dx in range(-order, order + 1):
            if (dy, dx) == (0, 0) or abs(dy) + abs(dx) > order:
                continue
            c = (slice(max(0, -dy), min(ny, ny - dy)), slice(max(0, -dx), min(nx, nx - dx)))
            nb = (slice(max(0, dy), min(ny, ny + dy)), slice(max(0, dx), min(nx, nx + dx)))
            shape = ph[c].shape
            a = Orientation(q[c].reshape(-1, 4), symmetry=sym)
            b = Orientation(q[nb].reshape(-1, 4), symmetry=sym)
            ang = np.degrees(np.asarray(Misorientation(b * ~a, symmetry=(sym, sym))
                                        .reduce().angle)).reshape(shape)
            valid = ang < thr_deg
            acc[c] += np.where(valid, ang, 0.0)
            cnt[c] += valid
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def test_single_phase_kam_equals_orix_reference():
    m = make_map(shape=(12, 14), n_seeds=4)
    m["phase_id"][:] = 0
    np.testing.assert_allclose(calculate_kam(_ds(m), order=2, threshold_deg=7.0),
                               _orix_kam(m, 2, 7.0), atol=1e-6, equal_nan=True)


def test_other_phase_and_unindexed_neighbours_do_not_count():
    m = make_map(shape=(10, 10), n_seeds=1)
    m["phase_id"][:, 5:] = 1
    m["phase_id"][0, 0] = -1
    kam = calculate_kam(_ds(m), order=1, threshold_deg=7.0)
    assert np.isnan(kam[0, 0])
    m2 = {**m, "phase_id": m["phase_id"].copy()}
    m2["phase_id"][:, 5:] = -1                       # the other phase removed altogether
    kam2 = calculate_kam(_ds(m2), order=1, threshold_deg=7.0)
    np.testing.assert_allclose(kam[:, :5], kam2[:, :5], atol=1e-9, equal_nan=True)


def test_with_grains_pairs_across_a_grain_boundary_are_left_out():
    m = make_map()
    ds = _ds(m)
    g = reconstruct_grains(ds, angle_threshold_deg=0.3, min_pixels=1, fill=False)   # tiny threshold: many grains
    without = calculate_kam(ds, order=1, threshold_deg=7.0)
    with_g = calculate_kam(ds, order=1, threshold_deg=7.0, grains=g)
    finite = np.isfinite(with_g)
    assert np.nanmax(np.abs(with_g[finite] - without[finite])) > 0  # something was excluded
    # a pixel alone in its grain has no neighbour left
    sizes = np.bincount(g.grain_ids.ravel())
    alone = sizes[g.grain_ids] == 1
    assert alone.any()
    assert np.isnan(with_g[alone]).all()


def test_culled_pixels_are_nan_with_grains():
    d = add_defects(make_map(), n_holes=0, scratch=False, wide=False, border=False)
    ds = _ds(d)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, fill=False)
    kam = calculate_kam(ds, order=2, threshold_deg=7.0, grains=g)
    assert np.isnan(kam[g.culled_mask]).all()
