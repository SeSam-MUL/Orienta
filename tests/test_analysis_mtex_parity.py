"""analysis/ against MTEX for Python on a synthetic two-phase map with holes, a scratch,
a wide not-indexed area, a border strip and outliers.

The reference numbers come from MTEX run as a separate program; see the fixture README.
"""
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from analysis.deformation_analysis import calculate_kam
from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from tests.synthetic_grain_map import to_xmap

FIX = Path(__file__).parent / "fixtures" / "analysis_mtex_parity" / "synthetic_two_phase.npz"
PARAMS = {"p0": (5.0, 3, 3.1), "p1": (5.0, 3, 1.5), "p2": (15.0, 5, 3.1)}


@pytest.fixture(scope="module")
def fx():
    with np.load(FIX) as f:
        return {k: f[k] for k in f.files}


def _ours(fx, p):
    m = {k: fx[k] for k in ("quats", "phase_id", "phase_names", "point_groups")}
    m["step"] = float(fx["step"])
    ds = EBSDDataset(to_xmap(m), step_size=m["step"])
    thr, minpix, alpha = PARAMS[p]
    return ds, reconstruct_grains(ds, angle_threshold_deg=thr, min_pixels=minpix, alpha=alpha, fill=False)


def _measured(g):
    """The measurements the grains are made of (in a grain, not closed)."""
    return (g.grain_ids > 0) & ~g.closed_mask


def _mtex_numbers(g, theirs):
    """Our grain number -> MTEX grain id, read through the measured pixels (0 stays 0)."""
    lut = np.zeros(g.n_grains + 1, dtype=np.int64)
    meas = _measured(g)
    lut[g.grain_ids[meas]] = theirs[meas]
    return lut


@pytest.mark.parametrize("p", list(PARAMS))
def test_partition_equals_mtex(fx, p):
    """The measured pixels fall into exactly the grains MTEX finds: same number of grains,
    ARI 1.0, and the measurements MTEX leaves in grains are exactly the ones we keep."""
    _, g = _ours(fx, p)
    theirs = fx[f"{p}_grain_ids"]
    meas = _measured(g)
    assert g.n_grains == int(np.sum(fx[f"{p}_phase"] >= 0))      # measured: 32 / 31 / 30 grains
    assert (theirs[meas] > 0).all()
    assert adjusted_rand_score(g.grain_ids[meas], theirs[meas]) == 1.0
    # every indexed pixel is a measurement of a grain or one of our culled ones
    np.testing.assert_array_equal((fx["phase_id"] >= 0) & ~meas, g.culled_mask)
    lut = _mtex_numbers(g, theirs)
    assert len(np.unique(lut[1:])) == g.n_grains                  # one MTEX grain per grain of ours


@pytest.mark.parametrize("p", list(PARAMS))
def test_pixels_without_a_measurement_take_the_grain_mtex_fills_them_with(fx, p):
    """Host of every originally not-indexed pixel against `fill_grain` (the bound is >= 99 %;
    measured 100 % on all three parameter sets)."""
    _, g = _ours(fx, p)
    mine = _mtex_numbers(g, fx[f"{p}_grain_ids"])[g.grain_ids]
    not_indexed = fx["phase_id"] < 0
    agree = np.mean(mine[not_indexed] == fx[f"{p}_fill_grain"][not_indexed])
    assert agree >= 0.99
    assert g.closed_mask[not_indexed].sum() == np.count_nonzero(fx[f"{p}_fill_grain"][not_indexed])


def test_p0_reproduces_the_generator_truth(fx):
    """Independent of MTEX: the measured pixels land in exactly the generator's grains.

    On the map without the wide not-indexed area. At the corners of a wide area MTEX - and now
    Orienta - counts fragments that touch only around the corner as separate grains (fixture p0: 32
    grains, generator truth 30), so the full map is compared with MTEX above, not with the truth.
    """
    from tests.synthetic_grain_map import add_defects, make_map

    m = make_map()
    d = add_defects(m, wide=False)
    g = reconstruct_grains(EBSDDataset(to_xmap(d), step_size=d["step"]),
                           angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    assert g.n_grains == m["truth"].max()
    measured = (g.grain_ids > 0) & ~g.closed_mask
    assert adjusted_rand_score(g.grain_ids[measured], m["truth"][measured]) == 1.0


@pytest.mark.parametrize("p", list(PARAMS))
def test_grain_values_equal_mtex(fx, p):
    """Every grain of every parameter set, against the MTEX grain it is made of: pixel count,
    area, both perimeters and aspect ratio relative < 1e-9, GOS < 1e-6 deg, phase.
    Measured worst deviations: area 3.6e-15, perimeters 4.4e-16, aspect 7.4e-12, GOS 2.7e-11 deg."""
    _, g = _ours(fx, p)
    t = _mtex_numbers(g, fx[f"{p}_grain_ids"])[1:] - 1          # MTEX array index of each grain of ours
    np.testing.assert_array_equal(g.grain_size, fx[f"{p}_num_pixel"][t])
    for ours, key in ((g.area, "area"), (g.perimeter, "perimeter"),
                      (g.perimeter_with_inclusions, "perimeter_with_inclusions"), (g.aspect_ratio, "aspect")):
        assert np.max(np.abs(ours / fx[f"{p}_{key}"][t] - 1)) < 1e-9, key
    assert np.degrees(np.abs(g.gos - fx[f"{p}_gos"][t])).max() < 1e-6
    np.testing.assert_array_equal(g.grain_phase, fx[f"{p}_phase"][t])


@pytest.mark.parametrize("p", list(PARAMS))
def test_boundary_segments_equal_mtex(fx, p):
    """Pairs, summed lengths and angles of the grain boundary segments: the pairs are equal,
    lengths < 1e-9 um, angles < 1e-4 deg. The fixture lists phase pairs with NaN; ours has none."""
    _, g = _ours(fx, p)
    pairs, angle, length = g.boundary_pairs, g.boundary_type, g.boundary_length
    ref_pairs, ref_length, ref_angle = (fx[f"{p}_gb_pairs"], fx[f"{p}_gb_length"], fx[f"{p}_gb_angle"])
    same_phase = np.isfinite(ref_angle)
    order = np.lexsort((ref_pairs[same_phase][:, 1], ref_pairs[same_phase][:, 0]))
    np.testing.assert_array_equal(pairs, ref_pairs[same_phase][order])
    assert np.abs(length - ref_length[same_phase][order]).max() < 1e-9
    assert np.abs(angle - ref_angle[same_phase][order]).max() < 1e-4
    assert len(pairs) == len(angle) == len(length)


@pytest.mark.parametrize("p", list(PARAMS))
def test_fill_marks_the_pixels_mtex_fills(fx, p):
    ds, _ = _ours(fx, p)
    thr, minpix, alpha = PARAMS[p]
    g = reconstruct_grains(ds, angle_threshold_deg=thr, min_pixels=minpix, alpha=alpha, fill=True)
    not_indexed = fx["phase_id"] < 0
    np.testing.assert_array_equal(g.filled_mask[not_indexed], fx[f"{p}_fill_grain"][not_indexed] > 0)
    assert np.isfinite(g.filled_rotations[g.filled_mask]).all()


def test_kam_without_and_with_grains(fx):
    ds, g = _ours(fx, "p0")
    for ours, theirs in ((calculate_kam(ds, 2, 7.0), fx["kam_nograin"]),
                         (calculate_kam(ds, 2, 7.0, grains=g), fx["kam_grain_p0"])):
        np.testing.assert_array_equal(np.isfinite(ours), np.isfinite(theirs))   # same NaN pattern
        both = np.isfinite(ours) & np.isfinite(theirs)
        assert both.sum() > 0.5 * ours.size
        assert np.abs(ours[both] - theirs[both]).max() < 1e-4


PROBES = Path(__file__).parent / "fixtures" / "analysis_mtex_parity" / "cells_probes.npz"


def _probe_dataset(not_indexed):
    """One phase, every pixel its own orientation within 0.8 deg of the identity (the stored MTEX numbers
    do not depend on the values: with a 5 deg threshold every connected part is one grain)."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation

    ny, nx = not_indexed.shape
    deg = (np.arange(ny * nx) % 3) * 0.4
    xmap = CrystalMap(rotations=Rotation.from_axes_angles([0, 0, 1], np.deg2rad(deg)),
                      phase_id=np.where(not_indexed.ravel(), -1, 0),
                      x=np.tile(np.arange(nx, dtype=float), ny), y=np.repeat(np.arange(ny, dtype=float), nx),
                      phase_list=PhaseList(names=["Al"], point_groups=["m-3m"]))
    return EBSDDataset(xmap, step_size=1.0)


def _probe_names():
    with np.load(PROBES) as f:
        return sorted({k.split("__")[0] for k in f.files if "__" in k})


@pytest.mark.parametrize("name", _probe_names())
def test_probe_grain_values_equal_mtex(name):
    """19 small maps with bands, crosses, notches, discs, an island and random speckle: area, perimeter,
    perimeter with inclusions, aspect ratio and pixel count of every grain through reconstruct_grains
    (measured worst: area 1e-15, perimeters 1e-13, aspect 7e-14 relative)."""
    with np.load(PROBES) as f:
        f = {k: f[k] for k in f.files if k.startswith(name + "__")}
    not_indexed = f[name + "__notindexed"].astype(bool)
    g = reconstruct_grains(_probe_dataset(not_indexed), angle_threshold_deg=5.0, min_pixels=1,
                           alpha=float(f[name + "__alpha"]), fill=False)
    mtex = f[name + "__grain_ids"].astype(np.int64)
    grain_list = [int(v) for v in f[name + "__grain_list"]]
    assert g.n_grains == len(grain_list)
    meas = _measured(g)
    lut = np.zeros(g.n_grains + 1, dtype=np.int64)
    lut[g.grain_ids[meas]] = mtex[meas]
    rows = np.array([grain_list.index(int(lut[k])) for k in range(1, g.n_grains + 1)])
    np.testing.assert_array_equal(g.grain_size, f[name + "__num_pixel"][rows])
    for ours, key in ((g.area, "area"), (g.perimeter, "perimeter"),
                      (g.perimeter_with_inclusions, "perimeter_with_inclusions"), (g.aspect_ratio, "aspect")):
        assert np.max(np.abs(ours / f[f"{name}__{key}"][rows] - 1)) < 1e-9, (name, key)
