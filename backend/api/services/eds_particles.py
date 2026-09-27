"""Connected components of an EDS region: per-particle shape and composition.

WHY THIS EXISTS. A *region* in this app is a chemistry group that spans the
whole map; a *particle* is one connected component of one region. They answer
different questions and only the first one could be answered. "How much
silicon is in this scan" is a region question. "How big are the silicon
particles, how round are they, how many are cut off at the edge" is a particle
question, and it is the one a size distribution, a dispersoid count or a
failure analysis is actually made of.

The data was already being computed and thrown away: the region inspector runs
``scipy.ndimage.label`` for its "pieces" readout and keeps the twelve largest
sizes (``backend/api/routes/eds.py``, ``_region_detail``). This module keeps
all of it and measures it.

WHY CROFTON AND NOT A BOUNDARY-PIXEL COUNT. Counting boundary steps measures
the staircase, not the object. Measured here on rasterised discs, the
staircase count is 1.262 / 1.270 / 1.269 / 1.273 / 1.273 times the true
perimeter at r = 5 / 10 / 25 / 50 / 100 - it converges to 4/pi = 1.2732 and
never improves with size, because the staircase length of a straight diagonal
is genuinely sqrt(2) times its length. Since circularity goes as
1/perimeter^2, a 27 % perimeter overestimate reads as a 38 % circularity
UNDERESTIMATE: every particle in the export would look less round than it is,
and a user comparing our numbers against ImageJ or MTEX would conclude the
classification was wrong. The 4-direction Crofton estimate on the same discs
gives ratios 1.021 / 1.010 / 0.999 / 0.998 / 1.000. It is used here, and
``scale_to_um`` says so in the column documentation.

The implementation was verified bit-identical to
``skimage.measure.perimeter_crofton(mask, directions=4)`` on discs of r =
5..100, a single pixel, a 3x3 square and a 1x5 line. scikit-image is not
imported: this module is deliberately numpy/scipy only so a future batch
driver can call it with no extra dependency.

Crofton is unbiased for a convex set of random orientation, which is what a
particle is. It is NOT unbiased for an axis-aligned polygon - a 50x50 square
measures 188.5 against a true 200 - and objects of a few pixels are not
measurable at all (a single pixel measures 2.68 against a true 4, which makes
its circularity read 1.75). That is why the export carries ``n_px`` and
``below_size_limit`` next to every shape number.

WHY PIXEL CORNERS AND NOT PIXEL CENTRES. A hull over pixel centres gives a
single pixel a Feret diameter of zero and a two-pixel particle a width of
zero, which then divides into ``aspect_ratio``. Hulling the four corners of
each pixel treats a pixel as the unit square it represents, so a single pixel
measures 1.0 px across its flats and sqrt(2) px across its diagonal, and the
hull area of any convex pixel set equals its pixel count exactly (so solidity
is exactly 1, not 1.4).

WHY CORE-ONLY COMPOSITION EXISTS (``n_px_core``). The EDS interaction volume
during an EBSD session is larger than the features being mapped, so the rim
pixels of a small particle are particle-plus-matrix mixtures. Averaging a
particle over all its pixels therefore drags every small particle toward the
matrix and manufactures a spurious "composition depends on particle size"
trend. ``composition_for_mask`` is called twice by the exporter - once on the
full mask, once on the eroded core - so the user can see and defend the
choice. Small particles legitimately erode to nothing; ``n_px_core == 0`` is
information, not an error.

WHY THE IDS ARE SORTED AND NOT LABELLING ORDER. ``scipy.ndimage.label``
numbers components in raster order, so inserting one pixel at the top of the
map renumbers everything below it. A user who builds a figure with particle
labels, changes a threshold and re-exports would find every label pointing at
a different particle. Ids here are assigned after a total order on
``(region_id, -n_px, centroid_row, centroid_col)``, which is stable under
anything that does not change the particles themselves.

Pure numpy/scipy: no request, no session, no loaded file, so this is unit
testable without data and reusable from a batch driver.

Spec: docs/superpowers/specs/2026-08-27-eds-presets-and-export-design.md
      (sections 3.4 and 6)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

__all__ = [
    "CONNECTIVITY",
    "ParticleGeometry",
    "find_particles",
    "particle_masks",
    "composition_for_mask",
    "scale_to_um",
]

#: Pixel connectivity used to split a region into particles.
#:
#: 8 rather than 4, because these masks come out of a smoothed, clustered
#: composition field and not out of a clean segmentation: a thin feature that
#: runs diagonally across the raster is one particle to a human and, under
#: 4-connectivity, a chain of separate one-pixel particles to the counter.
#: Splitting a real dispersoid into fourteen "particles" is the worse error
#: here - it corrupts the count and every size percentile - so the diagonal
#: link is kept.
#:
#: Callers MUST record the value they used in the export provenance: the
#: particle count is not comparable across the two settings, and neither is
#: any distribution derived from it.
CONNECTIVITY = 8

#: Value in ``label_grid`` meaning "this pixel belongs to no region".
UNASSIGNED = -1


@dataclass
class ParticleGeometry:
    """One connected component of one region, measured in pixels.

    Everything here is in raster units; nothing in this object knows the step
    size. :func:`scale_to_um` is the only place physical units appear, and it
    refuses to guess a missing step size.

    Coordinates are 0-based, row-major, row 0 at the top - the same convention
    as the label grid and as every other map in this app.
    """

    particle_id: int          # 1-based, stable ordering (see module docstring)
    region_id: int
    n_px: int
    n_px_core: int            # after a one-pixel 4-connected erosion; may be 0
    centroid_row: float
    centroid_col: float
    bbox_row_min: int         # all four bounds INCLUSIVE
    bbox_row_max: int
    bbox_col_min: int
    bbox_col_max: int
    touches_edge: bool        # any pixel sits on the raster border
    perimeter_px: float       # 4-direction Crofton estimate
    feret_max_px: float       # longest caliper distance over the pixel corners
    feret_min_px: float       # rotating-calipers minimum width
    convex_area_px: float     # area of the pixel-corner convex hull

    # --- bookkeeping, not measurements -------------------------------------
    #: Label this component carried in the per-region ``ndimage.label`` pass.
    #: :func:`particle_masks` re-runs that labelling and selects on this, which
    #: is exact; matching on bounding box and size would be ambiguous for two
    #: interleaved components of equal extent.
    component_label: int = 0
    #: Convex-hull vertices as ``(V, 2)`` float array in ``(row, col)``
    #: pixel-corner coordinates. Kept so :func:`scale_to_um` can be EXACT when
    #: the pixel is not square - a Feret diameter is not a length that can be
    #: rescaled by a single factor once the two axes have different steps.
    #: ``compare=False`` because an ndarray has no truth value and would break
    #: dataclass equality.
    hull_px: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")


#: 4-connected element for the core erosion. Deliberately NOT tied to
#: ``connectivity``: connectivity decides what counts as one particle,
#: erosion decides how deep the interaction-volume rim reaches, and they are
#: unrelated questions. A 4-connected erosion peels exactly one pixel off
#: every face, which is the rim the mixing argument is about.
_EROSION_STRUCT = ndimage.generate_binary_structure(2, 1)


def _crofton_perimeter(mask: np.ndarray) -> float:
    """4-direction Crofton perimeter estimate, in pixels.

    Cauchy-Crofton in 2D is ``P = 1/2 * integral over theta of the intercept
    count``. Discretised over the four directions a square raster offers, with
    the diagonal line families spaced 1/sqrt(2) apart instead of 1::

        P = pi/4 * [ R_0 + R_90 + (R_45 + R_135) / sqrt(2) ]

    where ``R_d`` is the number of foreground runs along direction ``d``,
    counted as "foreground pixel whose predecessor along d is background".

    Verified bit-identical to ``skimage.measure.perimeter_crofton(mask, 4)``.
    See the module docstring for why this and not a staircase count.
    """
    if not mask.any():
        return 0.0
    p = np.pad(mask, 1)
    c = p[1:-1, 1:-1]
    r_0 = int(np.count_nonzero(c & ~p[1:-1, :-2]))    # runs along (0, +1)
    r_90 = int(np.count_nonzero(c & ~p[:-2, 1:-1]))   # runs along (+1, 0)
    r_45 = int(np.count_nonzero(c & ~p[:-2, :-2]))    # runs along (+1, +1)
    r_135 = int(np.count_nonzero(c & ~p[:-2, 2:]))    # runs along (+1, -1)
    return (math.pi / 4.0) * (r_0 + r_90 + (r_45 + r_135) / math.sqrt(2.0))


def _corner_candidates(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pixel-corner points that can possibly be convex-hull vertices.

    Only the leftmost and rightmost pixel of a raster row can contribute a
    hull vertex - any corner between them lies on the segment joining the two.
    So 4 points per occupied row suffice instead of 4 per pixel: on a 30 000 px
    blob that is ~700 points instead of 120 000, and the hull is the only
    super-linear step in this module.

    ``rows``/``cols`` must be in row-major (``np.nonzero``) order, which
    guarantees rows ascend and columns ascend within a row.
    """
    uniq_rows, starts = np.unique(rows, return_index=True)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = rows.size
    c_min = cols[starts]
    c_max = cols[ends - 1]

    # A pixel (r, c) occupies the unit square with corners (r, c) .. (r+1, c+1).
    top = uniq_rows.astype(np.float64)
    bot = top + 1.0
    left = c_min.astype(np.float64)
    right = c_max.astype(np.float64) + 1.0
    return np.concatenate([
        np.stack([top, left], axis=1),
        np.stack([top, right], axis=1),
        np.stack([bot, left], axis=1),
        np.stack([bot, right], axis=1),
    ])


def _min_caliper_width(verts: np.ndarray) -> float:
    """Rotating-calipers minimum width of a convex polygon.

    The minimum width of a convex polygon is always attained with one caliper
    flush against an edge, so it is the smallest over edges of the largest
    perpendicular distance from that edge to any vertex.
    """
    n = len(verts)
    if n < 3:
        return 0.0
    best = math.inf
    for i in range(n):
        a = verts[i]
        d = verts[(i + 1) % n] - a
        length = math.hypot(float(d[0]), float(d[1]))
        if length < 1e-12:
            continue                      # duplicate vertex, no edge direction
        rel = verts - a
        # 2D cross product by hand: np.cross on 2-vectors is removed in numpy 2.
        dist = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / length
        w = float(dist.max())
        if w < best:
            best = w
    return best if math.isfinite(best) else 0.0


def _hull_measures(
    rows: np.ndarray, cols: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    """``(feret_max_px, feret_min_px, convex_area_px, hull_vertices)``.

    Hulling pixel CORNERS rather than centres, so a single pixel is a unit
    square (min width 1.0, max caliper sqrt(2)) rather than a point. That also
    means the point set always spans two dimensions and Qhull cannot degenerate
    - the fallback below exists for defence, not for a known case.
    """
    r0, r1, c0, c1 = bbox
    h = float(r1 - r0 + 1)
    w = float(c1 - c0 + 1)

    # A particle that fills its bounding box IS its bounding box, so the hull
    # is known in closed form and Qhull can be skipped. This is not an
    # approximation, and it is not rare: single pixels, pairs and small
    # compact blobs are most of what a real map contains. It matters because
    # scipy's Qhull binding opens a temp file per call on Windows - measured
    # at 3.8 s of temp-file churn alone over 10 296 particles, more than a
    # third of the total runtime.
    if len(rows) == int(h) * int(w):
        verts = np.array([[r0, c0], [r0, c1 + 1.0],
                          [r1 + 1.0, c1 + 1.0], [r1 + 1.0, c0]], dtype=np.float64)
        return math.hypot(h, w), min(h, w), h * w, verts

    pts = _corner_candidates(rows, cols)
    try:
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        convex_area = float(hull.volume)     # 2D: .volume is area, .area is perimeter
    except Exception:
        # Defensive only. Fall back to the bounding box, which bounds all three
        # quantities from above and is never silently wrong-by-a-lot.
        return math.hypot(h, w), min(h, w), h * w, None

    diff = verts[:, None, :] - verts[None, :, :]
    feret_max = float(np.sqrt((diff * diff).sum(axis=-1)).max())
    feret_min = _min_caliper_width(verts)
    if feret_min <= 0.0:
        feret_min = feret_max                # a genuinely degenerate hull
    return feret_max, feret_min, convex_area, verts


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _as_grid(label_grid) -> np.ndarray:
    grid = np.asarray(label_grid)
    if grid.ndim != 2:
        raise ValueError(f"label_grid must be 2-D, got shape {grid.shape}")
    if grid.dtype.kind not in "iu":
        raise ValueError(
            f"label_grid must hold integer region ids, got dtype {grid.dtype}")
    return grid


def find_particles(
    label_grid,
    *,
    connectivity: int = CONNECTIVITY,
    include_regions: Optional[Iterable[int]] = None,
) -> List[ParticleGeometry]:
    """Every connected component of every region, measured.

    Parameters
    ----------
    label_grid
        ``(n_rows, n_cols)`` integer raster. Each distinct value ``>= 0`` is a
        region; ``-1`` means unassigned. A particle is a connected component
        WITHIN one region - two regions that touch are never one particle,
        however similar their chemistry.
    connectivity
        4 or 8. See :data:`CONNECTIVITY`; record it in the provenance.
    include_regions
        Restrict to these region ids. ``None`` means every region present.
        Ids that are not present are ignored rather than raising, so a caller
        can pass a fixed list across a batch of scans.

    Returns
    -------
    list of ParticleGeometry
        Ordered by ``(region_id, -n_px, centroid_row, centroid_col)`` and
        numbered ``1..N`` in that order, so the ids survive a re-export.

    Notes
    -----
    Works on bounding-box sub-arrays throughout; the only full-size arrays are
    one boolean and one label array per REGION (not per particle), so a
    500 000-pixel map with thousands of particles stays in the hundreds of
    milliseconds.
    """
    grid = _as_grid(label_grid)
    structure = _structure(connectivity)
    n_rows, n_cols = grid.shape

    present = np.unique(grid)
    region_ids = [int(v) for v in present if v >= 0]
    if include_regions is not None:
        wanted = {int(r) for r in include_regions}
        region_ids = [r for r in region_ids if r in wanted]

    out: List[ParticleGeometry] = []
    for region_id in region_ids:
        labelled, n_comp = ndimage.label(grid == region_id, structure=structure)
        if n_comp == 0:
            continue
        for comp, sl in enumerate(ndimage.find_objects(labelled), start=1):
            if sl is None:                    # label numbers are never sparse
                continue                      # here, but find_objects may say so
            r0, r1 = sl[0].start, sl[0].stop - 1
            c0, c1 = sl[1].start, sl[1].stop - 1
            sub = labelled[sl] == comp

            local_rows, local_cols = np.nonzero(sub)
            n_px = int(local_rows.size)
            if n_px == 0:
                continue

            rows = local_rows + r0
            cols = local_cols + c0

            # binary_erosion's default border_value=0 treats everything outside
            # the crop as background, which is exactly right: the bbox is tight,
            # so outside it there is by definition no pixel of this particle.
            core = ndimage.binary_erosion(sub, structure=_EROSION_STRUCT)

            feret_max, feret_min, convex_area, hull = _hull_measures(
                rows, cols, (r0, r1, c0, c1))

            out.append(ParticleGeometry(
                particle_id=0,                # assigned after the sort
                region_id=region_id,
                n_px=n_px,
                n_px_core=int(np.count_nonzero(core)),
                centroid_row=float(rows.mean()),
                centroid_col=float(cols.mean()),
                bbox_row_min=int(r0), bbox_row_max=int(r1),
                bbox_col_min=int(c0), bbox_col_max=int(c1),
                # The bbox is tight, so a bbox edge on the raster border implies
                # a real pixel there; no separate scan of the border is needed.
                touches_edge=bool(
                    r0 == 0 or c0 == 0 or r1 == n_rows - 1 or c1 == n_cols - 1),
                perimeter_px=_crofton_perimeter(sub),
                feret_max_px=feret_max,
                feret_min_px=feret_min,
                convex_area_px=convex_area,
                component_label=comp,
                hull_px=hull,
            ))

    out.sort(key=lambda p: (p.region_id, -p.n_px, p.centroid_row, p.centroid_col))
    for i, p in enumerate(out, start=1):
        p.particle_id = i
    return out


def particle_masks(
    label_grid,
    particles: Sequence[ParticleGeometry],
    *,
    connectivity: int = CONNECTIVITY,
) -> Iterator[Tuple[ParticleGeometry, np.ndarray]]:
    """Yield each particle together with its full-grid boolean mask.

    A generator on purpose: a map with 10 000 particles would otherwise hold
    10 000 full-size boolean arrays at once (5 GB on a 500 000-pixel map).
    Consumers - per-particle composition, core composition, per-particle
    margins - need one mask at a time.

    ``label_grid`` and ``connectivity`` MUST be the same ones
    :func:`find_particles` saw; the components are re-derived from them and
    selected by ``component_label``. Passing a different grid yields masks for
    different particles, silently. The label array of one region at a time is
    cached, so the natural order (the sorted output of ``find_particles``,
    which is grouped by region) relabels once per region.

    Yields in the order of ``particles``.
    """
    grid = _as_grid(label_grid)
    structure = _structure(connectivity)

    cached_region: Optional[int] = None
    cached_labels: Optional[np.ndarray] = None

    for p in particles:
        if p.component_label <= 0:
            raise ValueError(
                f"particle {p.particle_id} carries no component_label; only "
                "ParticleGeometry objects produced by find_particles() can be "
                "turned back into masks")
        if cached_region != p.region_id:
            cached_labels, _ = ndimage.label(
                grid == p.region_id, structure=structure)
            cached_region = p.region_id

        mask = np.zeros(grid.shape, dtype=bool)
        sl = (slice(p.bbox_row_min, p.bbox_row_max + 1),
              slice(p.bbox_col_min, p.bbox_col_max + 1))
        mask[sl] = cached_labels[sl] == p.component_label
        yield p, mask


def composition_for_mask(
    at_maps: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    """``{element: (mean_at_pct, sd_at_pct)}`` over the masked pixels.

    The spread is the POPULATION standard deviation (``ddof=0``). Every pixel
    of the particle is measured, so this is the whole population and not a
    sample drawn from a larger one; ``ddof=1`` would be answering "how well do
    these pixels estimate some other particle's composition", which is not the
    question. On two pixels reading 10 and 20 at% this returns ``sd = 5.0``,
    not 7.071. The exporter states the convention in the column header.

    The sd is a real number in the export, not decoration: it is how a user
    tells a homogeneous particle from a gradient or from two things that the
    smoothing merged.

    An empty mask returns ``{}`` rather than raising - a caller iterating
    particles must not have to special-case the degenerate one - and elements
    whose map does not match the mask shape are skipped rather than crashing
    the whole export.
    """
    flat = np.asarray(mask, dtype=bool).ravel()
    n = int(np.count_nonzero(flat))
    if n == 0:
        return {}

    out: Dict[str, Tuple[float, float]] = {}
    for element, values in at_maps.items():
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size != flat.size:
            continue
        sel = arr[flat]
        out[element] = (float(sel.mean()), float(sel.std()))
    return out


def _check_step(value, axis: str) -> float:
    if value is None:
        raise ValueError(
            f"step_{axis}_um is unknown; physical measures must be omitted "
            "entirely rather than computed on a guessed scale")
    step = float(value)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(
            f"step_{axis}_um must be a positive finite length, got {value!r}")
    return step


def scale_to_um(
    p: ParticleGeometry,
    step_x_um: float,
    step_y_um: float,
) -> Dict[str, float]:
    """Physical measures for one particle.

    ``step_x_um`` is the column (X) step, ``step_y_um`` the row (Y) step.

    Raises ``ValueError`` if either step is ``None``, non-finite or ``<= 0``.
    That is deliberate and the caller is expected to catch it and OMIT the
    micrometre columns: ``get_pixel_sizes()`` legitimately returns ``None``,
    and a size distribution silently computed on a scale of 1 um/px when the
    scan was 0.2 is a wrong number in a report - worse than a missing column,
    because it looks like an answer.

    Returned keys
    -------------
    area_um2
        ``n_px * step_x * step_y``. Exact; no shape model involved.
    ecd_um
        Equivalent circle diameter, ``2 * sqrt(area / pi)`` - the diameter of
        the circle with the same area. The size measure to prefer for
        distributions, since it does not depend on the perimeter estimate.
    feret_max_um, feret_min_um
        Longest caliper distance and rotating-calipers minimum width over the
        pixel-corner convex hull. Note that for a rectangle the maximum Feret
        is the DIAGONAL, so a 2:1 rectangle has ``feret_max/feret_min =
        sqrt(5) = 2.236``, not 2.
    aspect_ratio
        ``feret_max / feret_min``. There are several competing definitions in
        the literature (major/minor axis of the equivalent ellipse, bounding
        box ratio, Feret ratio); this is the Feret one and the export header
        says so.
    perimeter_um
        4-direction Crofton estimate, i.e. corrected for the staircase - see
        the module docstring. Unreliable below a few tens of pixels.
    circularity
        ``4 * pi * area / perimeter^2``. 1.0 for a disc. Not clamped: values
        slightly above 1 occur for objects of a few pixels, where Crofton
        underestimates the perimeter, and hiding that would hide the fact that
        the shape numbers are meaningless there.
    solidity
        ``n_px / convex_area``. 1.0 for a convex particle, lower for a concave
        or a branched one.
    centroid_x_um, centroid_y_um
        Centroid measured from the CENTRE of pixel (0, 0) - the same origin
        the vendor "X Position"/"Y Position" fields use - so a particle can be
        found again in Aztec or in the EBSD map. Y increases downward, as the
        raster does.

    Anisotropic pixels are handled exactly. Area, ECD and the centroid scale
    trivially, but a Feret diameter does not: it is the extent of the hull in
    some direction, and once the two axes have different steps that direction
    changes. The stored hull is therefore rescaled and re-measured. The
    perimeter is the one quantity that cannot be recovered from pixel units
    alone; when the steps differ it is scaled by their geometric mean and is
    approximate to the same degree the aspect ratio departs from 1. Square
    pixels - which is what an EBSD scan almost always has - are exact
    throughout.
    """
    sx = _check_step(step_x_um, "x")
    sy = _check_step(step_y_um, "y")

    area_um2 = float(p.n_px) * sx * sy
    ecd_um = 2.0 * math.sqrt(area_um2 / math.pi) if area_um2 > 0 else 0.0

    if abs(sx - sy) <= 1e-12 * max(sx, sy):
        feret_max = p.feret_max_px * sx
        feret_min = p.feret_min_px * sx
        perimeter = p.perimeter_px * sx
    else:
        perimeter = p.perimeter_px * math.sqrt(sx * sy)
        if p.hull_px is not None and len(p.hull_px) >= 3:
            # hull_px is (row, col); rows scale with the Y step, cols with X.
            scaled = np.empty_like(p.hull_px, dtype=np.float64)
            scaled[:, 0] = p.hull_px[:, 0] * sy
            scaled[:, 1] = p.hull_px[:, 1] * sx
            diff = scaled[:, None, :] - scaled[None, :, :]
            feret_max = float(np.sqrt((diff * diff).sum(axis=-1)).max())
            feret_min = _min_caliper_width(scaled)
            if feret_min <= 0.0:
                feret_min = feret_max
        else:
            mean_step = math.sqrt(sx * sy)
            feret_max = p.feret_max_px * mean_step
            feret_min = p.feret_min_px * mean_step

    circularity = (
        4.0 * math.pi * area_um2 / (perimeter * perimeter)
        if perimeter > 0 else 0.0)
    solidity = (
        min(1.0, float(p.n_px) / p.convex_area_px)
        if p.convex_area_px > 0 else 1.0)
    aspect_ratio = feret_max / feret_min if feret_min > 0 else 1.0

    return {
        "area_um2": area_um2,
        "ecd_um": ecd_um,
        "feret_max_um": feret_max,
        "feret_min_um": feret_min,
        "perimeter_um": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "centroid_x_um": p.centroid_col * sx,
        "centroid_y_um": p.centroid_row * sy,
    }
