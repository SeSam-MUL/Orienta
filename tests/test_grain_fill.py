"""Filling closed pixels from their own grain."""
import numpy as np
from scipy import ndimage as ndi

from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from analysis.orientation_math import disorientation_angle, proper_ops
from orix.quaternion.symmetry import Oh
from tests.synthetic_grain_map import make_map, scratch_mask, to_xmap


def _scratched():
    m = make_map()
    band = scratch_mask(m["phase_id"].shape, (3, 0), (56, 79), 3.0)
    d = {**m, "quats": m["quats"].copy(), "phase_id": m["phase_id"].copy()}
    d["quats"][band] = [1.0, 0.0, 0.0, 0.0]
    d["phase_id"][band] = -1
    return m, d, band


def _same_grain_as_clean(g, clean, measured):
    """Per pixel: does the scratched run put it in the grain it had without the scratch?

    Grain numbers differ between runs, so each scratched-run grain is mapped to the
    clean-run grain that holds most of its measured pixels.
    """
    lut = np.full(g.n_grains + 1, -1, dtype=np.int64)
    for gid in range(1, g.n_grains + 1):
        own = measured & (g.grain_ids == gid)
        if own.any():
            lut[gid] = np.bincount(clean.grain_ids[own]).argmax()
    return lut[g.grain_ids] == clean.grain_ids


def test_scratch_is_filled_with_the_orientation_it_hid():
    m, d, band = _scratched()
    clean = reconstruct_grains(EBSDDataset(to_xmap(m), step_size=m["step"]), min_pixels=1, fill=False)
    g = reconstruct_grains(EBSDDataset(to_xmap(d), step_size=d["step"]), min_pixels=1, fill=True)
    # Which scratch pixels can stay unfilled, derived from this map alone. A pixel that is no generator
    # is held by the cell of its nearest generator, and the exterior ring (one row/column outside the
    # map) counts as a generator that grows no grain. So only a scratch pixel next to the map border
    # can stay open: one whose ring generator (distance = its distance to the border + 1) is at most as
    # far as the nearest map generator. The scratch touches the border at three pixels, (3, 0), (4, 0)
    # at its start and (55, 79) at its end; each has the ring generator at distance 1 and a map
    # generator at distance 1 too, so all three are ties. pixel_hosts decides a tie with a +x ray: the
    # two at the left border go to the grain (they are filled), (55, 79) - where the two map
    # generators (54, 79), (56, 79) and the ring generator (55, 80) meet at the pixel centre - goes to
    # the exterior ring, so it stays open. The same right-border outcome is MTEX's on the scratch end
    # (fixture p0_fill_grain is 0 at (53, 79), (54, 79), see test_grain_reconstruction); the left-border
    # tie outcome is the code's tie rule, checked against MTEX only in aggregate (test_grain_cells).
    ny, nx = band.shape
    gen = g.cells.generator
    d_map = np.rint(ndi.distance_transform_edt(~gen) ** 2)
    rr, cc = np.mgrid[0:ny, 0:nx]
    d_ring = np.minimum(np.minimum(cc + 1, nx - cc), np.minimum(rr + 1, ny - rr)) ** 2
    ring_may_hold = band & ~gen & (d_ring <= d_map)
    assert sorted(zip(*np.nonzero(ring_may_hold))) == [(3, 0), (4, 0), (55, 79)]
    unfilled = band & ~g.filled_mask
    assert sorted(zip(*np.nonzero(unfilled))) == [(55, 79)]       # a second unfilled pixel fails here
    assert g.grain_ids[55, 79] == 0
    cubic = band & (m["phase_id"] == 0)
    # inside the scratch the cell boundaries may sit a pixel off the true grain boundary;
    # those pixels carry the neighbour grain's orientation by design and are left out
    agree = cubic & _same_grain_as_clean(g, clean, ~band)
    assert agree.sum() > 0.8 * cubic.sum()
    err = np.degrees(disorientation_angle(g.filled_rotations[agree], m["quats"][agree], proper_ops(Oh)))
    assert np.median(err) < 0.5


def test_fill_never_crosses_a_grain_boundary():
    m, d, band = _scratched()
    g = reconstruct_grains(EBSDDataset(to_xmap(d), step_size=d["step"]), min_pixels=1, fill=True)
    ops = proper_ops(Oh)
    cubic = g.filled_mask & (g.grain_phase[np.maximum(g.grain_ids, 1) - 1] == 0)
    gid = g.grain_ids[cubic]
    mean = g.mean_orientations.data[gid - 1]
    ang = np.degrees(disorientation_angle(g.filled_rotations[cubic], mean, ops))
    assert ang.max() < 5.0                     # within the grain's own spread


def test_open_areas_and_measurements_are_not_filled():
    m = make_map()
    d = {**m, "phase_id": m["phase_id"].copy()}
    d["phase_id"][20:40, 30:50] = -1           # wide area
    g = reconstruct_grains(EBSDDataset(to_xmap(d), step_size=d["step"]), min_pixels=1, fill=True)
    assert not g.filled_mask[26:34, 36:44].any()
    assert not (g.filled_mask & (d["phase_id"] >= 0)).any()
    assert np.isnan(g.filled_rotations[~g.filled_mask]).all()


def test_a_closed_pixel_without_a_path_takes_the_grain_mean(caplog):
    """The closed pixel (1, 1) of grain 1 has only grain 2 around it in 4 directions, so growth cannot
    reach it; it takes the mean orientation of grain 1's measurement and the run says so at info level."""
    import logging

    from analysis.grain_fill import fill_closed_pixels

    ids = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 2]])
    measured = np.ones((3, 3), dtype=bool)
    measured[1, 1] = False
    quats = np.tile([1.0, 0, 0, 0], (3, 3, 1))
    a = np.deg2rad(7.0) / 2
    quats[0, 0] = [np.cos(a), 0, 0, np.sin(a)]                      # grain 1: 7 deg about z
    with caplog.at_level(logging.INFO, logger="analysis.grain_fill"):
        filled, mask = fill_closed_pixels(quats, ids, measured, np.array([0, 0]), {0: proper_ops(Oh)})
    assert mask[1, 1] and mask.sum() == 1
    assert np.degrees(disorientation_angle(filled[1:2, 1], quats[0:1, 0], proper_ops(Oh)))[0] < 1e-9
    assert [r for r in caplog.records if "no 4-path" in r.getMessage() and r.levelno == logging.INFO]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_pockets_of_two_phases_take_the_mean_with_their_own_symmetry():
    """Left block: cubic grain 1 with measurements 0 and 90 deg about z (equivalent under m-3m); right
    block: hexagonal grain 2 with 0 and 60 deg (equivalent under 6/mmm). Each has a closed pixel in the
    middle with no 4-path to its grain. The mean of a grain is only the common orientation when taken
    with the grain's own group (cubic on the hexagonal grain would give 30 deg to both measurements)."""
    from analysis.grain_fill import fill_closed_pixels
    from orix.quaternion.symmetry import D6h

    left = np.array([[1, 3, 3], [3, 1, 3], [3, 3, 1]])
    right = np.array([[2, 4, 4], [4, 2, 4], [4, 4, 2]])
    ids = np.hstack([left, right])
    measured = np.ones(ids.shape, dtype=bool)
    measured[1, 1] = measured[1, 4] = False
    quats = np.tile([1.0, 0, 0, 0], (3, 6, 1))

    def about_z(deg):
        a = np.deg2rad(deg) / 2
        return [np.cos(a), 0, 0, np.sin(a)]

    quats[2, 2] = about_z(90.0)
    quats[2, 5] = about_z(60.0)
    ops = {0: proper_ops(Oh), 1: proper_ops(D6h)}
    filled, mask = fill_closed_pixels(quats, ids, measured, np.array([0, 1, 0, 1]), ops)
    assert mask.sum() == 2 and mask[1, 1] and mask[1, 4]
    for (r, c), own, members in (((1, 1), ops[0], [(0, 0), (2, 2)]), ((1, 4), ops[1], [(0, 3), (2, 5)])):
        for mr, mc in members:
            assert np.degrees(disorientation_angle(filled[r:r + 1, c], quats[mr:mr + 1, mc], own))[0] < 1e-9


def test_fill_off_leaves_nothing():
    m, d, band = _scratched()
    g = reconstruct_grains(EBSDDataset(to_xmap(d), step_size=d["step"]), min_pixels=1, fill=False)
    assert g.filled_mask is None and g.filled_rotations is None
