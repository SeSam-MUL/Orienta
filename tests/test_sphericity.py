"""Sphericity as in the MATLAB reference (DeformationAnalysis.m:54):
area / perimeter('withInclusion') / equivalentRadius.

A smooth circle gives 0.5 (pi r^2 / (2 pi r * r)). A grain made of square pixels has a staircase
outline whose length is the Manhattan length, 8 r for a disc of radius r (the bounding square's
perimeter), so its sphericity is pi / 8 = 0.393, not 0.5. That is the value MTEX gives on a square
grid as well; the cell outline is what both measure.
"""
import numpy as np
import pytest
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation

from analysis.deformation_analysis import calculate_sphericity
from analysis.ebsd_dataset import EBSDDataset, GrainSet
from analysis.grain_analysis import reconstruct_grains


def _disc(r, hole=None, inner_deg=None, step=1.0):
    """A disc of radius r (30 deg about z) in a background of orientation identity.

    ``hole``: radius of a concentric region that is not indexed (``inner_deg`` None) or a grain of
    its own orientation (``inner_deg`` degrees about z).
    """
    n = int(2 * r + 10)
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.hypot(yy - c, xx - c)
    q = np.tile(Rotation.identity(1).data, (n * n, 1)).reshape(n, n, 4)
    q[d <= r] = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(30.0)).data[0]
    pid = np.zeros((n, n), int)
    if hole is not None:
        h = d <= hole
        if inner_deg is None:
            pid[h] = -1
        else:
            q[h] = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(inner_deg)).data[0]
    xmap = CrystalMap(rotations=Rotation(q.reshape(-1, 4)), phase_id=pid.ravel(),
                      x=np.tile(np.arange(n, dtype=float), n) * step,
                      y=np.repeat(np.arange(n, dtype=float), n) * step,
                      phase_list=PhaseList(names=["Al"], point_groups=["m-3m"]))
    return EBSDDataset(xmap, step_size=step), n


def _ring_grain(grains, n, r, hole):
    """Index of the disc grain, read on the row through the centre between hole and rim."""
    return int(grains.grain_ids[n // 2, int((n - 1) / 2 - (r + hole) / 2)]) - 1


def _staircase(r):
    """Outline of a disc of radius r about a pixel corner, in pixels, from the geometry alone.

    The centre sits between pixels, so a row through it holds the centres at +-0.5, +-1.5, ... within
    r: 2 * floor(r + 0.5) pixels. A convex staircase is as long as its bounding square's perimeter.
    """
    return 4.0 * 2 * np.floor(r + 0.5)


def _by_hand(g, i):
    return g.area[i] / (g.perimeter_with_inclusions[i] * g.ecd[i] / 2.0)


def test_a_smooth_circle_has_sphericity_one_half():
    r = 7.5
    area = np.pi * r ** 2
    g = GrainSet(grain_ids=np.ones((1, 1), int), n_grains=1, area=np.array([area]),
                 ecd=np.array([2 * r]), perimeter=np.array([2 * np.pi * r]),
                 perimeter_with_inclusions=np.array([2 * np.pi * r]))
    assert calculate_sphericity(g)[0] == pytest.approx(0.5, rel=1e-12)


@pytest.mark.parametrize("r", [10, 20, 40])
def test_a_pixel_disc_measures_its_staircase_outline(r):
    ds, n = _disc(r)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    i = int(g.grain_ids[n // 2, n // 2]) - 1
    # measured: the outline of a convex staircase is the bounding square's perimeter, 8 r exactly
    assert g.perimeter[i] == pytest.approx(8.0 * r) and g.perimeter_with_inclusions[i] == pytest.approx(8.0 * r)
    s = calculate_sphericity(g)[i]
    # measured 0.3938 / 0.3938 / 0.3926 for r = 10 / 20 / 40: pi / 8 within 0.002
    assert s == pytest.approx(np.pi / 8.0, abs=2e-3)
    assert s == pytest.approx(_by_hand(g, i), rel=1e-12)


def test_the_step_scales_area_perimeter_and_radius_but_not_the_sphericity():
    """Area in um^2, perimeter and ECD in um: 0.25, 0.5 and 0.5 of the step-1 values at step 0.5."""
    values = {}
    for step in (1.0, 0.5):
        ds, n = _disc(20, step=step)
        g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
        i = int(g.grain_ids[n // 2, n // 2]) - 1
        values[step] = (g.area[i], g.perimeter_with_inclusions[i], g.ecd[i], calculate_sphericity(g)[i])
    assert values[1.0][:3] == pytest.approx((1264.0, 160.0, 2 * np.sqrt(1264.0 / np.pi)))
    assert values[0.5][0] == pytest.approx(0.25 * values[1.0][0], rel=1e-12)
    assert values[0.5][1] == pytest.approx(0.5 * values[1.0][1], rel=1e-12)
    assert values[0.5][2] == pytest.approx(0.5 * values[1.0][2], rel=1e-12)
    assert values[0.5][3] == pytest.approx(values[1.0][3], rel=1e-12)
    assert values[0.5][3] == pytest.approx(np.pi / 8.0, abs=2e-3)


def test_an_open_hole_counts_its_loop_in_the_perimeter():
    """The not-indexed core (radius 6, diameter beyond 2 alpha) stays open: its edges are a second loop."""
    ds, n = _disc(20, hole=6)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    i = _ring_grain(g, n, 20, 6)
    assert g.perimeter[i] == pytest.approx(_staircase(20))
    assert g.perimeter_with_inclusions[i] == pytest.approx(_staircase(20) + _staircase(6))   # 160 + 48
    outer_only = g.area[i] / (g.perimeter[i] * g.ecd[i] / 2.0)
    s = calculate_sphericity(g)[i]
    assert s < outer_only
    assert s == pytest.approx(_by_hand(g, i), rel=1e-12)


def test_an_enclosed_grain_counts_its_loop_in_the_perimeter():
    ds, n = _disc(20, hole=6, inner_deg=40.0)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    assert g.n_grains == 3
    i = _ring_grain(g, n, 20, 6)
    assert g.perimeter[i] == pytest.approx(_staircase(20))
    assert g.perimeter_with_inclusions[i] == pytest.approx(_staircase(20) + _staircase(6))   # 160 + 48
    outer_only = g.area[i] / (g.perimeter[i] * g.ecd[i] / 2.0)
    s = calculate_sphericity(g)[i]
    assert s < outer_only
    assert s == pytest.approx(_by_hand(g, i), rel=1e-12)


def test_a_closed_hole_has_no_loop_so_both_perimeters_agree():
    ds, n = _disc(20, hole=1.5)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, fill=False)
    i = int(g.grain_ids[n // 2, n // 2 - 8]) - 1
    assert g.n_closed[i] > 0
    assert g.perimeter_with_inclusions[i] == pytest.approx(g.perimeter[i])
    assert calculate_sphericity(g)[i] == pytest.approx(_by_hand(g, i), rel=1e-12)


def test_fallbacks_without_inclusion_perimeter_and_without_any_perimeter():
    area, ecd = np.array([100.0, 30.0]), np.array([2.0, 2.0]) * np.array([5.0, 3.0])
    ids = np.ones((1, 1), int)
    outer = np.array([40.0, 25.0])
    g = GrainSet(grain_ids=ids, n_grains=2, area=area, ecd=ecd, perimeter=outer)
    np.testing.assert_allclose(calculate_sphericity(g), area / (outer * ecd / 2.0))
    g = GrainSet(grain_ids=ids, n_grains=2, area=area, ecd=ecd)
    np.testing.assert_allclose(calculate_sphericity(g), area / (2 * np.pi * (ecd / 2.0) ** 2))
