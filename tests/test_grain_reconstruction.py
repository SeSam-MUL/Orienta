"""Segmentation with narrow not-indexed areas closed."""
import numpy as np
import pytest
from scipy import ndimage as ndi
from sklearn.metrics import adjusted_rand_score

from pathlib import Path

from analysis import grain_reconstruction as gr
from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_cells import build_cells
from analysis.grain_reconstruction import covered_mask, grid_arrays, phase_ops, segment
from analysis.orientation_math import disorientation_angle
from tests.synthetic_grain_map import add_defects, make_map, scratch_mask, to_xmap

THR = np.deg2rad(5.0)
FIXTURE = Path(__file__).parent / "fixtures" / "analysis_mtex_parity" / "synthetic_two_phase.npz"


def _ops(m):
    return phase_ops(to_xmap(m))


def _ari_on(mask, a, b):
    return adjusted_rand_score(a[mask], b[mask])


def test_clean_map_reproduces_the_true_grains():
    m = make_map()
    seg = segment(m["quats"], m["phase_id"], _ops(m), THR, min_pixels=1)
    assert seg.n_grains == m["truth"].max()
    assert _ari_on(np.ones_like(seg.measured), seg.grain_ids, m["truth"]) == 1.0
    assert not seg.closed.any() and not seg.culled.any()


def test_holes_scratch_and_border_strip_do_not_change_the_grains():
    m = make_map()
    d = add_defects(m, n_outliers=0, wide=False)
    seg = segment(d["quats"], d["phase_id"], _ops(m), THR, min_pixels=1, alpha=3.1)
    measured = d["phase_id"] >= 0
    assert _ari_on(measured, seg.grain_ids, m["truth"]) == 1.0
    # every hole pixel is closed except the end of the scratch at the right border: (53, 79) and (54, 79)
    # lie next to the map border, where the exterior ring's cell reaches them, so they stay open.
    # Expected pixels from MTEX's number on the same scratch: the scratch alone (no other defect) is
    # not indexed in the fixture map too. That map differs from this one by a wide area (rows 30-41,
    # columns 50-61, which the scratch crosses) and by outliers, so only the scratch part right of
    # column 65 is compared; there the scratch pixels with fixture p0_fill_grain == 0 are the open ones
    # (cells_probes band_w2 shows the same: 0 at both border pixels of its band)
    scratch = add_defects(m, n_holes=0, n_outliers=0, wide=False, border=False)["holes"]
    scratch[:, :66] = False
    with np.load(FIXTURE) as fx:
        assert (fx["phase_id"][scratch] < 0).all()
        mtex_open = scratch & (fx["p0_fill_grain"] == 0)
    assert sorted(zip(*np.nonzero(mtex_open))) == [(53, 79), (54, 79)]
    assert ((seg.grain_ids[scratch] == 0) == mtex_open[scratch]).all()
    assert (seg.grain_ids[d["holes"] & ~mtex_open] > 0).all()
    assert seg.n_grains == m["truth"].max()


def test_alpha_zero_lets_a_scratch_split_grains():
    m = make_map()
    d = add_defects(m, n_holes=0, n_outliers=0, wide=False, border=False)
    split = segment(d["quats"], d["phase_id"], _ops(m), THR, min_pixels=1, alpha=0.0)
    joined = segment(d["quats"], d["phase_id"], _ops(m), THR, min_pixels=1, alpha=3.1)
    assert split.n_grains > joined.n_grains == m["truth"].max()
    assert (split.grain_ids[d["holes"]] == 0).all()


@pytest.mark.parametrize("width,closed", [(4.0, True), (8.0, False)])
def test_the_width_rule(width, closed):
    m = make_map()
    band = scratch_mask(m["phase_id"].shape, (0, 40), (59, 40), width)
    q, ph = m["quats"].copy(), m["phase_id"].copy()
    ph[band] = -1
    seg = segment(q, ph, _ops(m), THR, min_pixels=1, alpha=3.1)
    # the two rows at each end of the band touch the map border and take the exterior's cell there
    # (MTEX leaves such border pixels open too, cells_probes band_w2: fill_grain 0 at (0, 10), (19, 10));
    # the width rule is read away from the ends
    interior = band.copy()
    interior[:2] = False
    interior[-2:] = False
    assert (seg.grain_ids[interior] > 0).all() == closed
    if closed:
        assert set(np.nonzero(band & (seg.grain_ids == 0))[0]) <= {0, 1, 58, 59}


def test_a_wide_area_stays_open_also_at_the_border():
    m = make_map()
    q, ph = m["quats"].copy(), m["phase_id"].copy()
    ph[20:40, 0:14] = -1                      # wide, touching the left border
    seg = segment(q, ph, _ops(m), THR, min_pixels=1, alpha=3.1)
    assert (seg.grain_ids[24:36, 0:8] == 0).all()


def test_outliers_are_culled_and_closed_into_their_grain():
    m = make_map()
    d = add_defects(m, n_holes=0, scratch=False, wide=False, border=False)
    seg = segment(d["quats"], d["phase_id"], _ops(m), THR, min_pixels=3, alpha=3.1)
    culled = np.flatnonzero(seg.culled.ravel())
    np.testing.assert_array_equal(culled, d["outliers"])
    keep = ~seg.culled
    assert _ari_on(keep, seg.grain_ids, m["truth"]) == 1.0
    assert (seg.grain_ids.ravel()[culled] > 0).all()


def test_phases_are_never_joined():
    m = make_map()
    q = np.broadcast_to(m["quats"][0, 0], m["quats"].shape).copy()   # one orientation everywhere
    ph = np.zeros(m["phase_id"].shape, np.int64)
    ph[:, 40:] = 1
    seg = segment(q, ph, _ops(m), THR, min_pixels=1)
    assert seg.n_grains == 2
    for g in range(1, seg.n_grains + 1):
        assert len(np.unique(ph[seg.grain_ids == g])) == 1


def test_no_measured_pair_below_threshold_is_split():
    m = make_map()
    d = add_defects(m)
    ops = _ops(m)
    seg = segment(d["quats"], d["phase_id"], ops, THR, min_pixels=3)
    q, ph, g, meas = d["quats"], d["phase_id"], seg.grain_ids, seg.measured
    for a, b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                 ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        both = meas[a] & meas[b] & (ph[a] == ph[b])
        for pid in np.unique(ph[a][both]):
            s = both & (ph[a] == pid)
            ang = disorientation_angle(q[a][s], q[b][s], ops[int(pid)])
            assert not np.any((ang < THR) & (g[a][s] != g[b][s]))


def test_deterministic():
    d = add_defects(make_map())
    ops = _ops(d)
    a = segment(d["quats"], d["phase_id"], ops, THR, min_pixels=3)
    b = segment(d["quats"], d["phase_id"], ops, THR, min_pixels=3)
    np.testing.assert_array_equal(a.grain_ids, b.grain_ids)


@pytest.mark.parametrize("kw", [{"threshold": 0.0}, {"alpha": -1.0}, {"min_pixels": 0}])
def test_invalid_parameters_raise(kw):
    m = make_map()
    args = {"threshold": THR, "alpha": 3.1, "min_pixels": 1, **kw}
    with pytest.raises(ValueError):
        segment(m["quats"], m["phase_id"], _ops(m), args["threshold"],
                min_pixels=args["min_pixels"], alpha=args["alpha"])


def test_a_phase_without_point_group_is_named():
    m = make_map()
    ops = _ops(m)
    del ops[1]
    with pytest.raises(ValueError, match="1"):
        segment(m["quats"], m["phase_id"], ops, THR)


def test_covered_mask_keeps_measured_and_alpha_zero_is_identity():
    rng = np.random.default_rng(0)
    meas = rng.random((30, 30)) > 0.2
    np.testing.assert_array_equal(covered_mask(meas, 0.0), meas)
    assert (covered_mask(meas, 3.1) >= meas).all()


def test_grid_arrays_returns_unit_quaternions():
    # float32-sourced data is off by ~1e-7 in norm; unnormalised that gives false angles.
    # orix renormalises on construction; grid_arrays normalises too, which guards other xmap sources.
    m = make_map()
    scaled = {**m, "quats": m["quats"] * (1.0 + 2e-7)}
    q, _ = grid_arrays(EBSDDataset(to_xmap(scaled), step_size=m["step"]))
    assert np.abs(np.linalg.norm(q, axis=-1) - 1.0).max() < 1e-12


def test_generator_truth_cells_are_4_connected():
    truth = make_map()["truth"]
    cross = ndi.generate_binary_structure(2, 1)
    for g in np.unique(truth):
        assert ndi.label(truth == g, structure=cross)[1] == 1, f"truth cell {g} is not 4-connected"


def test_covered_pixels_are_no_generators_and_open_ones_are():
    m = make_map()
    q, ph = m["quats"].copy(), m["phase_id"].copy()
    ph[30, 40] = -1                                    # a hole: covered, its area goes to the cells around
    ph[5:25, 60:75] = -1                               # a wide area: open, its pixels own cells
    seg = segment(q, ph, _ops(m), THR, min_pixels=1, alpha=3.1)
    gen = seg.cells.generator
    assert not gen[30, 40]
    assert gen[10:20, 65:70].all() and (seg.grain_ids[10:20, 65:70] == 0).all()   # the core stays open
    assert gen[ph >= 0].all()
    assert seg.cells.area[30 * 80 + 40] == 0
    assert seg.cells.area.sum() == pytest.approx(ph.size, abs=1e-9)      # the cells tile the map


def _two_class_map(seed, ny=14, nx=14, p_not_indexed=0.25):
    """Random pixels of two orientations (0 and 40 deg about z), a quarter of them not indexed."""
    r = np.random.default_rng(seed)
    cls = r.integers(0, 2, (ny, nx))
    q = np.zeros((ny, nx, 4))
    for k, deg in enumerate((0.0, 40.0)):
        a = np.deg2rad(deg) / 2
        q[cls == k] = [np.cos(a), 0, 0, np.sin(a)]
    ph = np.zeros((ny, nx), np.int64)
    ph[r.random((ny, nx)) < p_not_indexed] = -1
    return q, ph


def test_segmentation_cells_are_the_second_pass_and_the_final_drop_does_not_rebuild_them():
    """min_pixels: pass 1 culls, pass 2 segments again on the reduced map, the final check drops what is
    still too small WITHOUT a third pass and keeps the cells of pass 2 (MTEX's grain areas equal the
    second-pass cells to 9.5e-12 on a real map; cells rebuilt after the drop gave 543 false mismatches)."""
    q, ph = _two_class_map(seed=7)
    ops = phase_ops(to_xmap(make_map()))
    alpha, min_pixels = 1.5, 3
    measured0 = ph >= 0
    _, labels1, n1 = gr._once(q.reshape(-1, 4), ph.reshape(-1), measured0, ops, THR, alpha)
    counts1 = np.bincount(labels1, minlength=n1 + 1)
    measured1 = measured0 & ~((labels1 > 0) & (counts1[labels1] < min_pixels)).reshape(ph.shape)
    _, labels2, n2 = gr._once(q.reshape(-1, 4), ph.reshape(-1), measured1, ops, THR, alpha)
    assert (n1, n2) == (28, 8)            # measured on this seed: pass 1 finds 28 grains, pass 2 finds 8

    seg = segment(q, ph, ops, THR, min_pixels=min_pixels, alpha=alpha)
    assert seg.n_grains == 7              # one grain of pass 2 is still below min_pixels and is dropped
    cells_pass2 = build_cells(~(covered_mask(measured1, alpha) & ~measured1))
    np.testing.assert_array_equal(seg.cells.generator, cells_pass2.generator)
    np.testing.assert_array_equal(seg.cells.area, cells_pass2.area)
    after_drop = ~(covered_mask(seg.measured, alpha) & ~seg.measured)
    assert not np.array_equal(seg.cells.generator, after_drop)       # a rebuilt cell set would differ
    dropped = measured1 & ~seg.measured
    assert dropped.any() and (seg.grain_ids[dropped] == 0).all()     # dropped pixels: no grain, measured removed
    assert np.array_equal(seg.culled, measured0 & ~seg.measured)
