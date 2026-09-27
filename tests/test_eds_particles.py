"""Known-answer tests for backend.api.services.eds_particles.

Every geometric assertion here has a closed-form answer written next to it.
Smoke tests would not catch the failures this module can actually have - a
perimeter that silently reverts to a staircase count, a Feret taken over pixel
centres, ids that shuffle between two exports - and each of those produces a
plausible-looking number rather than an exception.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from backend.api.services.eds_particles import (
    CONNECTIVITY,
    ParticleGeometry,
    composition_for_mask,
    find_particles,
    particle_masks,
    scale_to_um,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _grid(shape, blobs):
    """Label grid of ``shape``; ``blobs`` is a list of ``(region_id, mask)``."""
    g = np.full(shape, -1, dtype=np.int32)
    for region_id, mask in blobs:
        g[mask] = region_id
    return g


def _disc_mask(shape, centre_row, centre_col, radius):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((yy - centre_row) ** 2 + (xx - centre_col) ** 2) <= radius ** 2


def _staircase_perimeter(mask):
    """The naive boundary-step count the module deliberately does NOT use."""
    p = np.pad(mask, 1)
    c = p[1:-1, 1:-1]
    return int(
        np.count_nonzero(c & ~p[1:-1, :-2]) + np.count_nonzero(c & ~p[1:-1, 2:])
        + np.count_nonzero(c & ~p[:-2, 1:-1]) + np.count_nonzero(c & ~p[2:, 1:-1])
    )


# ---------------------------------------------------------------------------
# degenerate and tiny shapes
# ---------------------------------------------------------------------------

def test_single_pixel_is_a_unit_square_not_a_point():
    g = _grid((7, 7), [])
    g[3, 3] = 0
    parts = find_particles(g)

    assert len(parts) == 1
    p = parts[0]
    assert p.particle_id == 1
    assert p.region_id == 0
    assert p.n_px == 1
    # A one-pixel particle has no interior. That is information, not an error.
    assert p.n_px_core == 0
    assert (p.centroid_row, p.centroid_col) == (3.0, 3.0)
    assert (p.bbox_row_min, p.bbox_row_max) == (3, 3)
    assert (p.bbox_col_min, p.bbox_col_max) == (3, 3)
    assert p.touches_edge is False

    # Corners, not centres: a point hull would give 0.0 for both and then
    # divide by it in aspect_ratio.
    assert p.feret_min_px == pytest.approx(1.0)
    assert p.feret_max_px == pytest.approx(math.sqrt(2.0))
    assert p.convex_area_px == pytest.approx(1.0)
    # Crofton on one pixel is not the true 4.0; documented, and the reason the
    # export carries n_px next to every shape column.
    assert p.perimeter_px == pytest.approx(2.6815170613, rel=1e-9)


def test_3x3_square():
    g = _grid((9, 9), [])
    g[3:6, 3:6] = 0
    p = find_particles(g)[0]

    assert p.n_px == 9
    assert p.n_px_core == 1                       # only the centre survives
    assert (p.centroid_row, p.centroid_col) == (4.0, 4.0)
    assert (p.bbox_row_min, p.bbox_row_max, p.bbox_col_min, p.bbox_col_max) == (3, 5, 3, 5)
    assert p.convex_area_px == pytest.approx(9.0)  # hull area == pixel area
    assert p.feret_min_px == pytest.approx(3.0)
    assert p.feret_max_px == pytest.approx(3.0 * math.sqrt(2.0))

    m = scale_to_um(p, 1.0, 1.0)
    assert m["area_um2"] == pytest.approx(9.0)
    assert m["solidity"] == pytest.approx(1.0)


def test_line_1xN_does_not_raise_and_measures_its_box():
    """A 1xN line is the classic degenerate hull - over pixel CENTRES."""
    g = _grid((9, 12), [])
    g[4, 2:7] = 0                                  # 1 x 5
    p = find_particles(g)[0]

    assert p.n_px == 5
    assert p.n_px_core == 0                        # a one-pixel-wide line has no core
    assert p.feret_min_px == pytest.approx(1.0)    # the pixel width
    assert p.feret_max_px == pytest.approx(math.hypot(1.0, 5.0))
    assert p.convex_area_px == pytest.approx(5.0)
    assert math.isfinite(p.perimeter_px)

    m = scale_to_um(p, 1.0, 1.0)
    assert m["aspect_ratio"] == pytest.approx(math.hypot(1.0, 5.0))
    assert m["solidity"] == pytest.approx(1.0)


def test_empty_grid_returns_no_particles():
    assert find_particles(np.full((20, 20), -1, dtype=np.int32)) == []


# ---------------------------------------------------------------------------
# the disc: the shape whose answers are known analytically
# ---------------------------------------------------------------------------

def test_disc_area_ecd_perimeter_and_circularity():
    r = 50
    shape = (2 * r + 4, 2 * r + 4)
    centre = (shape[0] - 1) / 2.0
    mask = _disc_mask(shape, centre, centre, r)
    g = _grid(shape, [(0, mask)])

    p = find_particles(g)[0]
    n_px = int(mask.sum())
    assert p.n_px == n_px

    # Effective radius of the rasterised disc; the r=50 ideal is 7853.98 px.
    r_eff = math.sqrt(n_px / math.pi)
    m = scale_to_um(p, 1.0, 1.0)

    assert m["area_um2"] == pytest.approx(float(n_px))
    assert m["ecd_um"] == pytest.approx(2.0 * r_eff)
    assert m["ecd_um"] == pytest.approx(2.0 * r, rel=0.01)

    # The point of Crofton: within a few percent of 2*pi*r.
    assert m["perimeter_um"] == pytest.approx(2.0 * math.pi * r_eff, rel=0.02)
    assert m["circularity"] == pytest.approx(1.0, abs=0.02)
    assert m["aspect_ratio"] == pytest.approx(1.0, abs=0.05)
    assert m["solidity"] > 0.95


def test_crofton_beats_the_staircase_count_by_the_expected_margin():
    """The staircase overestimates by ~4/pi; that is why it is not used.

    A user asked specifically whether the perimeter is corrected. It is, and
    this test pins the size of the correction so a silent revert is caught.
    """
    r = 50
    shape = (2 * r + 4, 2 * r + 4)
    centre = (shape[0] - 1) / 2.0
    mask = _disc_mask(shape, centre, centre, r)
    p = find_particles(_grid(shape, [(0, mask)]))[0]

    true_perimeter = 2.0 * math.pi * math.sqrt(int(mask.sum()) / math.pi)
    staircase = _staircase_perimeter(mask)

    assert staircase / true_perimeter == pytest.approx(4.0 / math.pi, rel=0.01)
    assert p.perimeter_px / true_perimeter == pytest.approx(1.0, abs=0.01)
    # ... and the staircase would have made this disc read as 62% round.
    assert 4.0 * math.pi * mask.sum() / staircase ** 2 < 0.65


# ---------------------------------------------------------------------------
# elongation
# ---------------------------------------------------------------------------

def test_two_to_one_rectangle_feret_and_aspect_ratio():
    g = _grid((40, 60), [])
    g[10:20, 10:30] = 0                            # 10 rows x 20 cols

    p = find_particles(g)[0]
    assert p.n_px == 200
    # The maximum Feret of a rectangle is its DIAGONAL, so a 2:1 box is
    # sqrt(5) = 2.236 by the Feret definition, not 2. Several competing
    # definitions exist; scale_to_um's docstring names the one used.
    assert p.feret_max_px == pytest.approx(math.hypot(10.0, 20.0))
    assert p.feret_min_px == pytest.approx(10.0)
    assert p.convex_area_px == pytest.approx(200.0)

    m = scale_to_um(p, 1.0, 1.0)
    assert m["aspect_ratio"] == pytest.approx(math.sqrt(5.0))
    assert m["solidity"] == pytest.approx(1.0)
    assert m["circularity"] < 0.9                  # not round


def test_solidity_drops_for_a_concave_particle():
    """An L: half the bounding box, so a hull far larger than the pixels."""
    g = _grid((30, 30), [])
    g[5:25, 5:15] = 0
    g[15:25, 15:25] = 0
    p = find_particles(g)[0]

    assert p.n_px == 20 * 10 + 10 * 10
    assert p.convex_area_px > p.n_px
    assert scale_to_um(p, 1.0, 1.0)["solidity"] < 0.9


# ---------------------------------------------------------------------------
# components, regions, edges
# ---------------------------------------------------------------------------

def test_two_disconnected_blobs_in_one_region_are_two_particles():
    g = _grid((30, 60), [])
    g[5:15, 5:15] = 0                              # 100 px
    g[5:11, 40:46] = 0                             # 36 px, same region

    parts = find_particles(g)
    assert len(parts) == 2
    assert {p.region_id for p in parts} == {0}
    assert [p.n_px for p in parts] == [100, 36]    # sorted largest first
    assert [p.particle_id for p in parts] == [1, 2]


def test_particles_are_split_per_region_even_when_regions_touch():
    g = _grid((20, 20), [])
    g[5:15, 5:10] = 0
    g[5:15, 10:15] = 1                             # shares a border with region 0

    parts = find_particles(g)
    assert len(parts) == 2
    assert [p.region_id for p in parts] == [0, 1]
    assert all(p.n_px == 50 for p in parts)


def test_touches_edge_true_and_false():
    g = _grid((20, 20), [])
    g[0:5, 0:5] = 0                                # flush against row 0 / col 0
    g[10:14, 10:14] = 0                            # interior
    g[16:20, 2:6] = 0                              # flush against the last row

    parts = {p.n_px: p for p in find_particles(g)}
    assert parts[25].touches_edge is True
    assert parts[16].touches_edge is True
    # the 4x4 interior blob is 16 px too, so identify it by position instead
    interior = [p for p in find_particles(g) if p.bbox_row_min == 10][0]
    assert interior.touches_edge is False


def test_connectivity_4_vs_8_on_a_diagonal_link():
    g = _grid((10, 10), [])
    g[3, 3] = 0
    g[4, 4] = 0                                    # diagonal neighbour only

    assert len(find_particles(g, connectivity=8)) == 1
    assert len(find_particles(g, connectivity=4)) == 2
    assert CONNECTIVITY == 8                       # the module default


def test_include_regions_filters_and_ignores_absent_ids():
    g = _grid((20, 20), [])
    g[2:6, 2:6] = 0
    g[10:14, 10:14] = 1
    g[15:18, 2:5] = 2

    assert {p.region_id for p in find_particles(g)} == {0, 1, 2}
    assert {p.region_id for p in find_particles(g, include_regions=[1])} == {1}
    assert {p.region_id for p in find_particles(g, include_regions=[0, 2])} == {0, 2}
    # An id that is not on this map is ignored, so a batch driver can pass a
    # fixed list of regions across a folder of scans.
    assert find_particles(g, include_regions=[99]) == []


def test_n_px_core_erodes_one_pixel_off_every_face():
    g = _grid((20, 20), [])
    g[5:12, 5:14] = 0                              # 7 x 9
    p = find_particles(g)[0]
    assert p.n_px == 63
    assert p.n_px_core == 5 * 7                    # (7-2) x (9-2)


# ---------------------------------------------------------------------------
# id stability
# ---------------------------------------------------------------------------

def test_ids_are_stable_across_runs_and_ordered_by_size():
    g = _grid((40, 80), [])
    g[2:12, 2:12] = 0                              # 100 px, region 0
    g[20:24, 20:24] = 0                            # 16 px,  region 0
    g[2:8, 40:46] = 0                              # 36 px,  region 0
    g[20:30, 50:60] = 1                            # 100 px, region 1

    first = find_particles(g)
    second = find_particles(g.copy())

    key = lambda ps: [(p.particle_id, p.region_id, p.n_px) for p in ps]
    assert key(first) == key(second)

    # Ordered by region, then by descending size.
    assert [(p.region_id, p.n_px) for p in first] == [
        (0, 100), (0, 36), (0, 16), (1, 100)]
    assert [p.particle_id for p in first] == [1, 2, 3, 4]

    # And the rule that matters for a figure legend: within a region, a bigger
    # particle always carries a lower id than a smaller one.
    for a, b in zip(first, first[1:]):
        if a.region_id == b.region_id:
            assert a.n_px >= b.n_px


def test_ids_do_not_shuffle_when_a_particle_below_them_grows():
    """Raster-order labelling would renumber; the size sort must not."""
    base = _grid((40, 40), [])
    base[2:12, 2:12] = 0                           # 100 px, near the top
    base[30:34, 30:34] = 0                         # 16 px,  near the bottom

    grown = base.copy()
    grown[30:35, 30:35] = 0                        # now 25 px, still smaller

    big_before = [p for p in find_particles(base) if p.n_px == 100][0]
    big_after = [p for p in find_particles(grown) if p.n_px == 100][0]
    assert big_before.particle_id == big_after.particle_id == 1


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def test_particle_masks_yields_the_right_pixels_in_input_order():
    g = _grid((30, 60), [])
    g[5:15, 5:15] = 0
    g[5:11, 40:46] = 0
    g[20:26, 20:26] = 1

    parts = find_particles(g)
    pairs = list(particle_masks(g, parts))

    assert [p.particle_id for p, _ in pairs] == [p.particle_id for p in parts]
    total = np.zeros(g.shape, dtype=bool)
    for p, mask in pairs:
        assert mask.shape == g.shape
        assert mask.dtype == bool
        assert int(mask.sum()) == p.n_px
        assert np.all(g[mask] == p.region_id)
        assert not (total & mask).any()            # particles never overlap
        total |= mask

    assert np.array_equal(total, g >= 0)           # and they tile the regions


def test_particle_masks_is_a_generator_not_a_list():
    g = _grid((10, 10), [])
    g[2:5, 2:5] = 0
    it = particle_masks(g, find_particles(g))
    assert hasattr(it, "__next__")


def test_particle_masks_refuses_a_hand_built_geometry():
    """Fail loud: a mask for the wrong particle is a silently wrong export."""
    g = _grid((10, 10), [])
    g[2:5, 2:5] = 0
    fake = ParticleGeometry(
        particle_id=1, region_id=0, n_px=9, n_px_core=1,
        centroid_row=3.0, centroid_col=3.0,
        bbox_row_min=2, bbox_row_max=4, bbox_col_min=2, bbox_col_max=4,
        touches_edge=False, perimeter_px=10.0,
        feret_max_px=4.2, feret_min_px=3.0, convex_area_px=9.0,
    )
    with pytest.raises(ValueError, match="component_label"):
        list(particle_masks(g, [fake]))


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def test_composition_for_mask_returns_mean_and_population_sd():
    """Two pixels reading 10 and 20 at%: mean 15, POPULATION sd 5.0.

    The sample sd (ddof=1) would be 7.0711. The module documents ddof=0 and
    the export header states it, so this number is pinned deliberately.
    """
    at_maps = {
        "Al": np.array([[10.0, 20.0], [99.0, 99.0]]),
        "Si": np.array([[3.0, 3.0], [0.0, 0.0]]),
    }
    mask = np.array([[True, True], [False, False]])

    comp = composition_for_mask(at_maps, mask)
    assert set(comp) == {"Al", "Si"}
    assert comp["Al"][0] == pytest.approx(15.0)
    assert comp["Al"][1] == pytest.approx(5.0)          # not 7.0710678
    assert comp["Al"][1] != pytest.approx(math.sqrt(50.0))
    assert comp["Si"] == pytest.approx((3.0, 0.0))


def test_composition_for_mask_on_an_empty_mask_returns_empty_dict():
    at_maps = {"Al": np.ones((4, 4))}
    assert composition_for_mask(at_maps, np.zeros((4, 4), dtype=bool)) == {}


def test_composition_for_mask_accepts_flat_maps_and_skips_mismatched_ones():
    at_maps = {
        "Al": np.array([1.0, 2.0, 3.0, 4.0]),           # flat, right size
        "Fe": np.array([1.0, 2.0]),                     # wrong size entirely
    }
    mask = np.array([[True, True], [False, False]])
    comp = composition_for_mask(at_maps, mask)
    assert comp["Al"] == pytest.approx((1.5, 0.5))
    assert "Fe" not in comp


def test_core_composition_differs_from_all_pixel_composition():
    """The reason n_px_core exists: rim pixels are mixtures."""
    g = _grid((20, 20), [])
    g[5:12, 5:12] = 0
    p = find_particles(g)[0]
    _, mask = next(iter(particle_masks(g, [p])))

    si = np.zeros((20, 20))
    si[5:12, 5:12] = 20.0                              # rim: diluted
    si[6:11, 6:11] = 40.0                              # core: the real value

    core = np.zeros((20, 20), dtype=bool)
    core[6:11, 6:11] = True
    assert int(core.sum()) == p.n_px_core

    all_px = composition_for_mask({"Si": si}, mask)["Si"]
    core_px = composition_for_mask({"Si": si}, core)["Si"]
    assert core_px[0] == pytest.approx(40.0)
    assert all_px[0] < core_px[0]                      # dragged toward the matrix
    assert core_px[1] == pytest.approx(0.0)
    assert all_px[1] > 0.0


# ---------------------------------------------------------------------------
# physical scaling
# ---------------------------------------------------------------------------

def _one_particle(n_rows=10, n_cols=20):
    g = np.full((n_rows + 10, n_cols + 10), -1, dtype=np.int32)
    g[3:3 + n_rows, 4:4 + n_cols] = 0
    return find_particles(g)[0]


@pytest.mark.parametrize("bad", [None, 0, 0.0, -1.0, -0.25, float("nan"), float("inf")])
def test_scale_to_um_raises_on_an_unusable_step(bad):
    p = _one_particle()
    with pytest.raises(ValueError):
        scale_to_um(p, bad, 1.0)
    with pytest.raises(ValueError):
        scale_to_um(p, 1.0, bad)


def test_scale_to_um_error_names_the_axis():
    p = _one_particle()
    with pytest.raises(ValueError, match="step_x_um"):
        scale_to_um(p, None, 0.5)
    with pytest.raises(ValueError, match="step_y_um"):
        scale_to_um(p, 0.5, None)


def test_isotropic_scaling_is_exact():
    p = _one_particle(n_rows=10, n_cols=20)         # 200 px
    step = 0.25
    m = scale_to_um(p, step, step)

    assert m["area_um2"] == pytest.approx(200 * step * step)
    assert m["feret_max_um"] == pytest.approx(math.hypot(10, 20) * step)
    assert m["feret_min_um"] == pytest.approx(10 * step)
    assert m["ecd_um"] == pytest.approx(2 * math.sqrt(200 * step * step / math.pi))
    # Dimensionless measures must not move with the scale.
    ref = scale_to_um(p, 1.0, 1.0)
    for k in ("circularity", "solidity", "aspect_ratio"):
        assert m[k] == pytest.approx(ref[k])


def test_non_square_pixels_give_the_right_area_and_feret():
    p = _one_particle(n_rows=10, n_cols=20)         # 10 rows x 20 cols
    sx, sy = 0.5, 2.0                               # x = column step, y = row step
    m = scale_to_um(p, sx, sy)

    # 200 px * 0.5 * 2.0 = 200 um^2. The box is 20 cols * 0.5 = 10 um wide and
    # 10 rows * 2.0 = 20 um tall: the anisotropy turns a wide particle into a
    # tall one, so the Feret diameters must come from the RESCALED hull and not
    # from a single factor on the pixel values (which would give 22.36 * 1.0
    # by luck here and be wrong for any other step pair).
    assert m["area_um2"] == pytest.approx(200.0)
    assert m["ecd_um"] == pytest.approx(2.0 * math.sqrt(200.0 / math.pi))
    assert m["feret_max_um"] == pytest.approx(math.hypot(10.0 * sy, 20.0 * sx))
    assert m["feret_min_um"] == pytest.approx(10.0)
    assert m["aspect_ratio"] == pytest.approx(math.hypot(20.0, 10.0) / 10.0)

    # A step pair that is not a mirror of the pixel shape, to make sure the
    # agreement above is not arithmetic coincidence: 10 rows * 1.0 = 10 um tall,
    # 20 cols * 3.0 = 60 um wide.
    m2 = scale_to_um(p, 3.0, 1.0)
    assert m2["area_um2"] == pytest.approx(600.0)
    assert m2["feret_max_um"] == pytest.approx(math.hypot(10.0, 60.0))
    assert m2["feret_min_um"] == pytest.approx(10.0)


def test_centroid_is_measured_from_the_centre_of_pixel_zero():
    g = _grid((20, 20), [])
    g[4:7, 8:11] = 0                                # centroid at row 5, col 9
    p = find_particles(g)[0]
    assert (p.centroid_row, p.centroid_col) == (5.0, 9.0)

    m = scale_to_um(p, 0.5, 0.25)
    assert m["centroid_x_um"] == pytest.approx(9.0 * 0.5)
    assert m["centroid_y_um"] == pytest.approx(5.0 * 0.25)


def test_scale_to_um_returns_exactly_the_contracted_keys():
    m = scale_to_um(_one_particle(), 1.0, 1.0)
    assert set(m) == {
        "area_um2", "ecd_um", "feret_max_um", "feret_min_um", "perimeter_um",
        "circularity", "solidity", "aspect_ratio", "centroid_x_um",
        "centroid_y_um",
    }


# ---------------------------------------------------------------------------
# input validation and scale
# ---------------------------------------------------------------------------

def test_bad_label_grid_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        find_particles(np.zeros((4, 4, 4), dtype=np.int32))
    with pytest.raises(ValueError, match="integer"):
        find_particles(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="connectivity"):
        find_particles(np.zeros((4, 4), dtype=np.int32), connectivity=6)


def test_rectangle_fast_path_agrees_with_qhull_exactly():
    """The bbox shortcut in _hull_measures must be exact, not close.

    A particle that fills its bounding box has a hull known in closed form, so
    Qhull is skipped - that is what makes 10 000 small particles take under a
    second instead of eleven. An optimisation that quietly disagreed with the
    general path would show up as shape numbers that depend on whether a
    particle happened to be rectangular, which no user would ever diagnose.
    """
    from scipy.spatial import ConvexHull

    from backend.api.services import eds_particles as ep

    for h in range(1, 9):
        for w in range(1, 9):
            rows, cols = np.nonzero(np.ones((h, w), dtype=bool))
            fast = ep._hull_measures(rows, cols, (0, h - 1, 0, w - 1))

            pts = ep._corner_candidates(rows, cols)
            hull = ConvexHull(pts)
            verts = pts[hull.vertices]
            d = verts[:, None, :] - verts[None, :, :]

            assert fast[0] == pytest.approx(float(np.sqrt((d * d).sum(-1)).max()))
            assert fast[1] == pytest.approx(ep._min_caliper_width(verts))
            assert fast[2] == pytest.approx(float(hull.volume))


def test_half_a_megapixel_with_thousands_of_particles_is_quick():
    """500 000 px, >10 000 particles. Measured at 0.85 s on the dev box.

    The bound is loose because this asserts "not pathological", not a
    benchmark; the failure it guards against is a per-particle full-size array
    or a lost fast path, both of which cost orders of magnitude, not percent.
    """
    n_rows, n_cols = 500, 1000
    g = np.full((n_rows, n_cols), -1, dtype=np.int32)
    rows = np.arange(n_rows)[:, None]
    cols = np.arange(n_cols)[None, :]
    # A grid of 3x3 blobs on a 7 px pitch, alternating between two regions.
    blob = ((rows % 7) < 3) & ((cols % 7) < 3)
    g[blob] = 0
    g[blob & (((rows // 7) + (cols // 7)) % 2 == 1)] = 1

    t0 = time.perf_counter()
    parts = find_particles(g)
    elapsed = time.perf_counter() - t0

    assert len(parts) > 3000
    assert {p.region_id for p in parts} == {0, 1}
    assert all(p.n_px <= 9 for p in parts)
    assert elapsed < 20.0, f"find_particles took {elapsed:.1f}s"
