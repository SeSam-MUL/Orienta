"""Segment an orientation field into grains, chemistry-free.

Deliberately chemistry-free: this segmentation exists to correct per-pixel
phase labels, so it must not depend on the phase labels it is used to correct.
The only input is the orientation field.

Three failures cost real time while this was being designed, and ALL THREE
produced a plausible-looking wrong answer instead of an error:

* Euler angles exported in RADIANS were read as degrees. Every orientation
  collapsed onto the identity and the whole 121k-pixel map became one grain.
* ``(~o1) * o2`` was used instead of ``angle_with``, skipping the symmetry
  reduction. Disorientations came out up to 179 deg where m-3m allows 62.8.
* Unindexed pixels carry NaN Euler angles, and orix's ``angle_with`` runs its
  result through ``np.nan_to_num``. Every bond touching an unindexed pixel
  therefore reported 0.0 deg -- "identical orientation" -- so unindexed pixels
  were welded to their neighbours, and a line of them welded two grains into
  one.
* Isolating those unindexed pixels then disarmed the degenerate-result guard,
  which asked for "exactly one grain". Every unindexed pixel takes its own
  label, so on a real map that count is in the thousands and can never be one.
  A 301x402 map with 5% unindexed pixels and the radian bug above came back
  with 6072 "grains" and no complaint, the 30 deg boundary silently gone. The
  guard now measures the share of indexed pixels that landed in the largest
  label, not the number of labels.

All four are now hard guards. A wrong number that looks right is worse than a
crash, because nobody checks it.

Note the pattern in the last two: each was introduced by the fix for the one
before it. Guards here need testing against the *combination* of failures, at
realistic map size, not one at a time on a 10x10.
"""
from __future__ import annotations

import numpy as np

# Theoretical maximum disorientation angle per point group (degrees).
#
# The maximum is a property of the PROPER ROTATION SUBGROUP, not of the point
# group's name or its Laue class: an inversion or a mirror is not a rotation,
# so adding one cannot bring two orientations closer together. Reading the name
# instead of the proper subgroup is what once put m-3 and -43m at the cubic
# 62.8 deg; both reduce to T, not to O.
#
# EVERY value below was reproduced numerically rather than copied. Method:
# maximise the two-sided symmetry-reduced disorientation angle over SO(3) --
# 300k random rotations, then hill-climb the best 64 with a shrinking step.
# The method was validated first against five settled literature values (O
# 62.80, D3 104.48, D4 98.42, D6 93.84, D2 120.00) and matched all five to
# better than 0.005 deg before any of the others were believed.
#
# The reproduced column is a numerical LOWER bound: hill-climbing approaches
# the maximum from below and never quite reaches it, so a value a few
# thousandths under the tabulated one is convergence, not disagreement.
#
#   point group  proper subgroup   tabulated   reproduced (lower bound)
#   m-3m, 432    O  (432)            62.8       62.798
#   m-3, -43m    T  (23)             90.0       89.999
#   6/mmm        D6 (622)            93.8       93.840
#   4/mmm        D4 (422)            98.4       98.420
#   -3m          D3 (32)            104.5      104.477
#   mmm          D2 (222)           120.0      119.999
#   2/m          C2 (2)             180.0      180.000
#   -1           C1 (1)             180.0      180.000
#
# T is 90.0, not the 104.5 it was briefly given. 104.5 is D3's value, one row
# down, and copying it up is the error to guard against. What establishes 90.0
# is the measurement above, reproduced three ways: the search in this comment
# (89.999), an independent sampler written without orix (89.9998), and an exact
# argument that needs no sampling at all -- T's Rodrigues fundamental zone is
# |r_i| <= 1 intersected with |+-x+-y+-z| <= 1, whose farthest point is the
# vertex (1, 0, 0), giving theta = 2*arctan(1) = 90 deg exactly.
#
# Do NOT try to re-derive these by counting group elements. Order does not
# bound the maximum: measured, C6 has 6 proper rotations and reaches 180.0,
# while D2 has 4 and reaches 120.0; T and D6 have 12 each and reach 90.0 and
# 93.84. The rule that IS valid is the subgroup rule -- if H is a subgroup of
# G then max(G) <= max(H) -- but it does not settle T against D3, because D3 is
# NOT a subgroup of T (checked: 4 of D3's 6 elements are absent from T; T's
# only 2-folds are the three <100> axes, none perpendicular to a 3-fold). The
# subgroup of T that does apply is D2 at 120.0, which does not separate 90 from
# 104.5 at all.
#
# The tabulated values are rounded to one decimal, in both directions, so the
# guard below needs its 0.5 deg of slack: 6/mmm's true maximum is about 93.841
# against a tabulated 93.8, and 4/mmm's about 98.421 against 98.4. A perfectly
# legitimate boundary in either group measures ABOVE the number in this table
# -- the 6/mmm pair pinned in the tests measures 93.8338, itself only a lower
# bound on that group's maximum, and already 0.034 above the tabulated value.
# Do not tighten that slack to zero.
#
# Sampling under-reaches these maxima, since they are attained on a
# near-measure-zero set, but only mildly: over 200k random pairs per group the
# shortfall ranged from 0.0001 deg (-1) to 1.06 deg (mmm). A shortfall much
# larger than that is evidence the tabulated value is wrong, not that sampling
# is weak -- it is what should have exposed the 104.5 error, which sampling
# missed by 15.6 deg.
#
# At 180.0 deg, 2/m and -1 make the disorientation guard a structural no-op:
# no disorientation can exceed 180 deg, so monoclinic and triclinic get no
# protection from it. That is inherent to those symmetries, not a defect.
MAX_DISORIENTATION_DEG: dict[str, float] = {
    "m-3m": 62.8, "m-3": 90.0, "432": 62.8, "-43m": 90.0,
    "6/mmm": 93.8, "-3m": 104.5, "4/mmm": 98.4, "mmm": 120.0,
    "2/m": 180.0, "-1": 180.0,
}

_ORIX_SYMMETRY = {
    "m-3m": "Oh", "m-3": "Th", "432": "O", "-43m": "Td",
    "6/mmm": "D6h", "-3m": "D3d", "4/mmm": "D4h", "mmm": "D2h",
    "2/m": "C2h", "-1": "Ci",
}

#: Below this, a neighbour disorientation is floating-point noise rather than a
#: boundary, and a map whose largest one stays under it genuinely is a single
#: orientation -- so one grain is the answer, not a bug.
#:
#: orix computes the angle as ``arccos(2 * dot**2 - 1)``, and arccos near 1
#: amplifies double-precision error, so symmetry-equivalent orientations come
#: back exactly 0 deg apart only when they happen to be axis-aligned. Measured
#: over 2000 random m-3m orientations against a random symmetry-equivalent
#: partner, round-tripped through Euler angles as this function does: 1.3% land
#: more than 1e-6 deg apart, the worst 2.41e-6 deg, none as far as 1e-4 deg.
#: A 1e-6 tolerance would therefore refuse roughly 1 uniform map in 75.
#:
#: 1e-4 sits ~40x above that measured floor and ~5000x below the 0.52 deg left
#: behind by the double-conversion bug, so it separates "one orientation" from
#: "one grain by mistake" with margin either way. It is also far below any real
#: misorientation: EBSD angular resolution is ~0.1 deg, a million times coarser.
_SINGLE_ORIENTATION_TOL_DEG = 1e-4

#: Share of the INDEXED pixels that one label has to cover before the
#: segmentation counts as merged-into-one-grain.
#:
#: This started as a count -- "exactly one grain" -- and a count does not
#: survive contact with a real map. Unindexed pixels each take their own label,
#: so on a 301x402 map with 5% unindexed the count is ~6000 and the guard is
#: dead exactly where it matters. Counting labels over the indexed pixels only
#: is still not enough: measured on such a map with the units bug applied, the
#: count came to 2 rather than 1, because a single indexed pixel happened to be
#: ringed by unindexed pixels on all four sides. One stray pixel must not
#: disarm the guard, so it asks how much of the map merged, not how many pieces
#: are left.
#:
#: The trade is deliberate and one-directional: a genuine map in which one
#: grain legitimately dominates -- a single-grain calibration region, or a scan
#: with one large grain and a small second one -- will be refused. That costs a
#: user one explicit message they can act on, whereas the failure this catches
#: costs a silently merged map that looks like an ordinary result. A caller who
#: knows the scan is grain-dominated says so with ``allow_dominant_grain=True``.
#:
#: No constant can separate the two populations, which is why the escape is a
#: caller assertion and not a better threshold. Measured: a doubly-converted
#: 301x402 map lands at 0.9808, while a legitimate map with a real 1.44%
#: second grain lands at 0.9856 -- ABOVE it. Any cut that refuses the first
#: also refuses the second.
#:
#: 0.95 rather than 0.99, because 0.99 was measured to miss the real case. The
#: units bug does NOT simply divide every disorientation by 180/pi -- Euler
#: differences do not map linearly onto disorientation. Measured over 2000
#: random m-3m pairs, converting a second time takes the median from 42.4 to
#: 2.85 deg and the maximum from 61.8 to 11.2 deg. So a tail of boundaries
#: survives a 5 deg threshold, and a 301x402 map of ~500 grains collapses to a
#: largest share of 0.9808, not 1.0 -- under a 0.99 cut and silently returned.
#: A 10x10 misses it from the other direction: one stray indexed pixel out of
#: 96 is already 1.04% of the map.
#:
#: There is very little room here, so do not nudge this number without
#: re-measuring. The two cases it has to catch sit at 0.9808 (the realistic map)
#: and 0.9896 (one stray pixel on a 10x10) -- only 0.009 apart. Anything above
#: 0.9808 misses the realistic case, which is the one that matters; 0.95 clears
#: both by about 0.03, and every legitimate multi-grain map measured sits below
#: 0.01.
_MERGED_SHARE = 0.95


def segment_grains(euler_rad: np.ndarray, point_group_name: str = "m-3m",
                   threshold_deg: float = 5.0,
                   allow_dominant_grain: bool = False) -> np.ndarray:
    """Label connected regions whose neighbour disorientation stays below the
    threshold. ``euler_rad`` is (rows, cols, 3) Bunge angles in RADIANS.

    Returns an int32 (rows, cols) label map starting at 0.

    Pixels with a non-finite Euler triple are unindexed: they have no
    orientation, cannot be shown to lie inside any grain, and so each becomes
    its own single-pixel label. They never join or bridge their neighbours.

    ``allow_dominant_grain`` is the caller asserting "one grain legitimately
    dominates this scan" -- a pattern-centre calibration region inside a single
    grain is the normal case, and so is a scan with one large grain and a small
    second one. It is needed because the ambiguity cannot be resolved from the
    data: a single crystal with 0.5 deg of intra-grain scatter and a map whose
    Euler angles were converted to radians twice look identical from in here,
    and the two populations interleave, so no threshold separates them.

    **The cost is that the unit bug is not detectable on a call with this flag
    set.** Leave it off unless the scan is known to be grain-dominated.

    The flag is narrow in exactly one way, and that is the guarantee worth
    relying on: it stands down the merged-into-one-grain check and nothing
    else. The radian check, the point-group check, the maximum-disorientation
    check and the every-pixel-its-own-grain check all stay live, because the
    flag says a dominant grain is expected, not that the input is above
    suspicion.

    Raises ``ValueError`` rather than returning a plausible-looking label map
    when the shape is wrong, when no pixel is indexed at all, when the angles
    are too large to be radians, when the point group is unknown, when a
    disorientation exceeds the group maximum (which means the symmetry
    reduction did not run), or when the segmentation comes out degenerate.
    """
    from orix.quaternion import Orientation, symmetry as _sym

    euler_rad = np.asarray(euler_rad, dtype=np.float64)
    if euler_rad.ndim != 3 or euler_rad.shape[2] != 3:
        raise ValueError(f"expected (rows, cols, 3), got {euler_rad.shape}")

    # Unindexed pixels are legitimate input; a map made entirely of them is not
    # an orientation field and every guard below would be reading NaN.
    indexed = np.isfinite(euler_rad).all(axis=2)
    if not indexed.any():
        raise ValueError(
            "no indexed pixels: every Euler triple is non-finite, so there is "
            "no orientation field to segment")

    # Measured over the indexed pixels only, so an inf is reported as the
    # non-finite value it is rather than misdiagnosed as degrees.
    hottest = float(np.abs(euler_rad[indexed]).max())
    if hottest > 2.0 * np.pi + 1e-3:
        raise ValueError(
            "euler angles must be in radians -- the largest value "
            f"({hottest:.2f}) exceeds 2*pi, which means degrees were passed")
    if point_group_name not in _ORIX_SYMMETRY:
        raise ValueError(f"unknown point group {point_group_name!r}; "
                         f"known: {sorted(_ORIX_SYMMETRY)}")

    R, C, _ = euler_rad.shape
    sym = getattr(_sym, _ORIX_SYMMETRY[point_group_name])
    ori = Orientation.from_euler(euler_rad.reshape(-1, 3), symmetry=sym)

    is_indexed = indexed.ravel()
    idx = np.arange(R * C).reshape(R, C)
    bonds = []
    limit = MAX_DISORIENTATION_DEG[point_group_name]
    worst_overall = 0.0
    for a, b in ((idx[:-1, :].ravel(), idx[1:, :].ravel()),
                 (idx[:, :-1].ravel(), idx[:, 1:].ravel())):
        if a.size == 0:
            continue
        ang = np.asarray(ori[a].angle_with(ori[b], degrees=True)).ravel()
        # orix's angle_with ends in np.nan_to_num, so a bond touching an
        # unindexed pixel arrives as 0.0 -- the value meaning "identical".
        # Restore the NaN, which compares False against the threshold below and
        # is skipped by the nanmax guards, so the pixel isolates.
        ang[~(is_indexed[a] & is_indexed[b])] = np.nan
        if np.any(np.isfinite(ang)):
            worst = float(np.nanmax(ang))
            if worst > limit + 0.5:
                raise ValueError(
                    f"disorientation {worst:.2f} deg exceeds the maximum "
                    f"{limit:.1f} deg for {point_group_name} -- the symmetry "
                    "reduction did not run")
            worst_overall = max(worst_overall, worst)
        bonds.append((ang, a, b))

    parent = np.arange(R * C)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ang, a, b in bonds:
        keep = ang < threshold_deg
        for x, y in zip(a[keep], b[keep]):
            rx, ry = find(int(x)), find(int(y))
            if rx != ry:
                parent[rx] = ry

    raw = np.fromiter((find(i) for i in range(R * C)), dtype=np.int64,
                      count=R * C)
    _, lab = np.unique(raw, return_inverse=True)
    lab = lab.reshape(R, C).astype(np.int32)

    # Judged over the indexed pixels only. Unindexed pixels are individually
    # isolated by design, so counting them here would report thousands of
    # "grains" on any real map and hide both degenerate cases.
    sizes = np.bincount(lab[indexed].ravel())
    n_indexed = int(indexed.sum())
    n_grains = int(np.count_nonzero(sizes))
    largest_share = float(sizes.max()) / n_indexed

    # Merging everything is only a bug if the map really holds more than one
    # orientation. A uniform map -- or one whose halves differ by a symmetry
    # operation, which is the same orientation -- is correctly a single grain,
    # and that escape is the only thing this guard forgives.
    uniform = worst_overall <= _SINGLE_ORIENTATION_TOL_DEG
    merged = (largest_share >= _MERGED_SHARE and not uniform
              and not allow_dominant_grain)
    # Deliberately NOT gated on allow_dominant_grain: that flag says one grain
    # dominates, which is the opposite of every pixel standing alone.
    shattered = n_grains == n_indexed
    if R * C > 4 and (merged or shattered):
        # The escape only applies to the merged case. Offering it for a
        # shattered map would send the operator after the wrong thing.
        remedy = (
            "If one grain really does dominate this scan -- a pattern-centre "
            "calibration region, say -- pass allow_dominant_grain=True, which "
            "accepts the result and gives up the unit check on that call."
            if merged else
            "Every indexed pixel came out as its own grain: no neighbour bond "
            "survived the threshold. Raising threshold_deg is the lever here; "
            "allow_dominant_grain does not apply to this case.")
        raise ValueError(
            f"degenerate segmentation: {n_grains} grain(s) over {n_indexed} "
            f"indexed pixels ({R * C} total), the largest covering "
            f"{100.0 * largest_share:.2f}% of them; the largest neighbour "
            f"disorientation is {worst_overall:.4f} deg against a threshold of "
            f"{threshold_deg} deg. Check the threshold and the Euler-angle "
            "units first: Euler angles already in radians and converted to "
            f"radians a second time collapse the disorientation range. {remedy}")
    return lab
