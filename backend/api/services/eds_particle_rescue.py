"""Particle rescue for pattern-degenerate phase pairs (Al/Si and the like).

Why this exists (measured 2026-09-13, Probe B Arbeitsbereich 1, 80x48 crop,
Spherical Al+Si with the EDS prior on):

* The pattern correlation prefers Al on EVERY pixel of the scan, including
  pixels with 93 at% Si (sphere raw Al 0.59 / Si 0.53; render-NCC 0.63 / 0.60).
  A synthetic Si pattern rendered from the Si master IS recognised as Si
  (16/16), so the method can in principle -- on real patterns the margin is
  below the noise. Al and Si share the fcc band geometry; the Al fit and the
  Si fit of the same pattern land on the same orientation (0.1 deg apart).
  Only band widths and intensities differ, and those do not survive the
  correlation. On such a pair the pattern contributes NOTHING to the phase
  decision: the Si phase in the map is 100 % the EDS prior.
* The prior's rule is a symmetric composition distance, so it calls Si only
  when more Si than Al is measured (crossover ~48 at%). EDS integrates over a
  2-3 um volume; a Si particle smaller than that reads 20-40 at% Si and stays
  Al. Five such particles (37 px) on the crop, none of them anything but Al+Si.
* There is no lateral EDS/EBSD offset (best shift +1 row, gain < 0.5 px). The
  "offset" is the sampling volume: a pixel can show 30 at% Si while the
  diffracting surface is matrix, or while it is the particle.
* What the pattern CAN say is whether the surface belongs to the matrix
  grain: the Al-fitted orientation of an enriched blob against the ring of
  matrix pixels around it was 37-55 deg on the four real particles and
  0.3 deg on the one blob where the beam sits on matrix beside a particle.

So: a post-pass on the phase map, for phase pairs whose masters cannot be
told apart by the pattern (same point group). Take the element the particle
phase has and the matrix phase lacks; find connected blobs where it is
enriched; and decide EACH candidate pixel by orientation continuity with the
true matrix around it:

* a matrix-assigned pixel in the blob that matches NO nearby matrix pixel
  (min disorientation > ``angle_deg``) is not matrix -> it becomes the
  particle phase, keeping its measured orientation (the two fits agree);
* a particle-assigned pixel in the blob that DOES match a nearby matrix pixel
  is matrix at the surface -> it goes back to the matrix phase.

Pure functions, no I/O, no FastAPI. The route glues them to a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional

import numpy as np

#: An element counts as "defining" for the particle phase from this nominal at%.
DEFAULT_MAJOR_AT_PCT = 15.0
#: ... and as "absent" in the matrix phase up to this nominal at%.
DEFAULT_ABSENT_AT_PCT = 5.0
#: A pixel is enriched when the element is at least this many at% ...
DEFAULT_ENRICH_AT_PCT = 15.0
#: ... AND at least this multiple of the matrix background (median over the
#: matrix-assigned pixels). Si background on the crop: 1.6 at%.
DEFAULT_ENRICH_FACTOR = 5.0
#: Continuity: a pixel is attached to the matrix when some true-matrix pixel
#: within ``ring_px`` (Chebyshev) is closer than this in orientation.
DEFAULT_ANGLE_DEG = 5.0
DEFAULT_RING_PX = 3
#: Enriched blobs below this many pixels are not decided (see the loop).
DEFAULT_MIN_BLOB_PX = 3


def _min_angles_within(o_a, ra, ca, o_b, rb, cb, radius: int, inner_radius=None):
    """Per a-pixel, the smallest disorientation (deg, symmetry-reduced) to any
    b-pixel within Chebyshev ``radius``; ``inf`` where there is none. With
    ``inner_radius`` a second array restricted to that smaller radius comes
    back (else None). Pairs come from a k-d tree, so the cost is the number
    of pairs actually in reach, not len(a) x len(b)."""
    from scipy.spatial import cKDTree
    ra = np.asarray(ra); ca = np.asarray(ca); rb = np.asarray(rb); cb = np.asarray(cb)
    n_a = ra.size
    out = np.full(n_a, np.inf); inner = np.full(n_a, np.inf) if inner_radius is not None else None
    if n_a == 0 or rb.size == 0:
        return out, inner
    lists = cKDTree(np.c_[rb, cb]).query_ball_point(np.c_[ra, ca], r=radius, p=np.inf)
    counts = np.fromiter((len(l) for l in lists), dtype=int, count=n_a)
    if counts.sum() == 0:
        return out, inner
    ia = np.repeat(np.arange(n_a), counts)
    ib = np.concatenate([np.asarray(l, dtype=int) for l in lists if len(l)])
    ang = np.asarray(o_a[ia].angle_with(o_b[ib], degrees=True), dtype=np.float64).reshape(-1)
    np.minimum.at(out, ia, ang)
    if inner_radius is not None:
        close = (np.abs(ra[ia] - rb[ib]) <= inner_radius) & (np.abs(ca[ia] - cb[ib]) <= inner_radius)
        np.minimum.at(inner, ia[close], ang[close])
    return out, inner


@dataclass(frozen=True)
class RescuePair:
    particle_pid: int
    matrix_pid: int
    element: str
    particle_name: str = ""
    matrix_name: str = ""


@dataclass
class RescueResult:
    phase_id_2d: np.ndarray
    to_particle: np.ndarray          # (H, W) bool -- matrix -> particle
    to_matrix: np.ndarray            # (H, W) bool -- particle -> matrix
    report: dict = field(default_factory=dict)

    @property
    def n_changed(self) -> int:
        return int(self.to_particle.sum() + self.to_matrix.sum())


def select_rescue_pairs(
    phase_ids: list[int],
    expected_at: dict[int, dict[str, float]],
    point_groups: dict[int, Optional[str]],
    names: Optional[dict[int, str]] = None,
    *,
    major_at_pct: float = DEFAULT_MAJOR_AT_PCT,
    absent_at_pct: float = DEFAULT_ABSENT_AT_PCT,
) -> list[RescuePair]:
    """Ordered (particle, matrix) pairs the rescue may act on.

    A pair qualifies when both phases carry the SAME point group (the pattern
    cannot separate them -- this is the whole premise; for pairs the pattern
    can separate, the pattern is the better judge) and the particle phase
    has an element at >= ``major_at_pct`` that the matrix phase has at
    <= ``absent_at_pct``. With several such elements the most abundant one
    is used. Phases without a point group or composition never pair.
    """
    names = names or {}
    out: list[RescuePair] = []
    for p, q in permutations(phase_ids, 2):
        pg_p = point_groups.get(p)
        pg_q = point_groups.get(q)
        if not pg_p or not pg_q or str(pg_p) != str(pg_q):
            continue
        exp_p = expected_at.get(p) or {}
        exp_q = expected_at.get(q) or {}
        if not exp_p or not exp_q:
            continue
        candidates = [
            (float(v), el) for el, v in exp_p.items()
            if float(v) >= major_at_pct and float(exp_q.get(el, 0.0)) <= absent_at_pct
        ]
        if not candidates:
            continue
        _, element = max(candidates)
        out.append(RescuePair(
            particle_pid=int(p), matrix_pid=int(q), element=str(element),
            particle_name=str(names.get(p, "")), matrix_name=str(names.get(q, "")),
        ))
    return out


def _orientations(quats: np.ndarray, symmetry):
    from orix.quaternion import Orientation
    return Orientation(np.asarray(quats, dtype=np.float64), symmetry=symmetry)


def rescue_particles(
    phase_id_2d: np.ndarray,
    quats_2d: np.ndarray,
    element_2d: np.ndarray,
    pair: RescuePair,
    symmetry,
    *,
    enrich_at_pct: float = DEFAULT_ENRICH_AT_PCT,
    enrich_factor: float = DEFAULT_ENRICH_FACTOR,
    angle_deg: float = DEFAULT_ANGLE_DEG,
    ring_px: int = DEFAULT_RING_PX,
    min_blob_px: int = DEFAULT_MIN_BLOB_PX,
) -> RescueResult:
    """Apply the rescue for one pair. Pure: returns a NEW phase grid.

    Parameters
    ----------
    phase_id_2d : (H, W) int
        Phase id per pixel, negative = unindexed.
    quats_2d : (H, W, 4) float
        Orientation quaternion per pixel (w, x, y, z). Any frame, as long as
        every pixel is in the same one -- only disorientations are used.
    element_2d : (H, W) float
        Measured at% of ``pair.element``; NaN where unmeasured.
    pair : RescuePair
    symmetry : orix Symmetry
        Point group shared by both phases of the pair.
    """
    from scipy import ndimage as ndi

    pid = np.asarray(phase_id_2d)
    if pid.ndim != 2:
        raise ValueError(f"phase_id_2d must be 2-D, got {pid.shape}")
    H, W = pid.shape
    q = np.asarray(quats_2d, dtype=np.float64)
    if q.shape != (H, W, 4):
        raise ValueError(f"quats_2d must be {(H, W, 4)}, got {q.shape}")
    el = np.asarray(element_2d, dtype=np.float64)
    if el.shape != (H, W):
        raise ValueError(f"element_2d must be {(H, W)}, got {el.shape}")
    ring_px = int(ring_px)
    if ring_px < 1:
        raise ValueError("ring_px must be >= 1")

    new_pid = pid.copy()
    to_particle = np.zeros((H, W), dtype=bool)
    to_matrix = np.zeros((H, W), dtype=bool)
    report: dict = {
        "element": pair.element,
        "particle_pid": pair.particle_pid, "matrix_pid": pair.matrix_pid,
        "particle_name": pair.particle_name, "matrix_name": pair.matrix_name,
        "enrich_at_pct": float(enrich_at_pct), "enrich_factor": float(enrich_factor),
        "angle_deg": float(angle_deg), "ring_px": ring_px, "min_blob_px": int(min_blob_px),
        "n_blobs": 0, "n_to_particle": 0, "n_to_matrix": 0, "n_undecidable": 0,
        "n_too_small": 0, "blobs": [], "skipped": None,
    }

    valid = (pid >= 0) & np.isfinite(el)
    is_matrix = valid & (pid == pair.matrix_pid)
    is_particle = valid & (pid == pair.particle_pid)
    if not is_matrix.any():
        report["skipped"] = "no matrix pixels"
        return RescueResult(new_pid, to_particle, to_matrix, report)

    background = float(np.median(el[is_matrix]))
    report["background_at_pct"] = background
    if background >= enrich_at_pct:
        # The element is not a minority in the matrix -- "enriched" would
        # mean nothing. (Al vs Si seen from the Al side lands here.)
        report["skipped"] = (
            f"{pair.element} background {background:.1f} at% is not below "
            f"{enrich_at_pct:.1f} at%")
        return RescueResult(new_pid, to_particle, to_matrix, report)

    threshold = max(float(enrich_at_pct), float(enrich_factor) * background)
    report["threshold_at_pct"] = threshold
    enriched = valid & (el >= threshold)
    true_matrix = is_matrix & ~enriched
    if not enriched.any():
        return RescueResult(new_pid, to_particle, to_matrix, report)

    labels, n_blobs = ndi.label(enriched, structure=np.ones((3, 3), dtype=int))
    report["n_blobs"] = int(n_blobs)
    struct = np.ones((2 * ring_px + 1, 2 * ring_px + 1), dtype=bool)
    # Only pixels within ring_px of the true matrix can be decided at all;
    # the interior of a large blob is undecidable by construction and would
    # only make the pairwise angle matrices quadratic in the blob size
    # (measured: 2025 px -> 1.1 GB). Particle pixels one ring further in are
    # kept as the "core zone" the backward guard compares against.
    reach = ndi.binary_dilation(true_matrix, structure=struct)
    reach2 = ndi.binary_dilation(reach, structure=struct)
    rr, cc = np.mgrid[0:H, 0:W]

    for b in range(1, n_blobs + 1):
        blob = labels == b
        assigned = blob & (is_matrix | is_particle)
        if not assigned.any():
            continue
        if int(blob.sum()) < int(min_blob_px):
            # A lone enriched pixel with a non-matrix orientation is as likely
            # a bad pattern as a particle; the real particles on the crop were
            # 3 px and up. Counted, not decided.
            report["n_too_small"] += int(assigned.sum())
            continue
        ring = ndi.binary_dilation(blob, structure=struct) & true_matrix & ~blob
        cand = assigned & reach
        core_zone = assigned & is_particle & reach2 & ~reach
        brr, bcc = np.where(blob)
        entry = {
            "row_min": int(brr.min()), "row_max": int(brr.max()),
            "col_min": int(bcc.min()), "col_max": int(bcc.max()),
            "n_px": int(blob.sum()),
            "n_matrix_assigned": int((blob & is_matrix).sum()),
            "n_particle_assigned": int((blob & is_particle).sum()),
            "n_ring": int(ring.sum()),
            "n_to_particle": 0, "n_to_matrix": 0,
            "n_undecidable": int((assigned & ~cand).sum()),
            "n_core_free": 0, "median_min_angle_deg": None,
        }
        if not ring.any() or not cand.any():
            entry["n_undecidable"] = int(assigned.sum())
            report["n_undecidable"] += int(assigned.sum())
            report["blobs"].append(entry)
            continue

        crr, ccc = rr[cand], cc[cand]
        rrr, rcc = rr[ring], cc[ring]
        o_cand = _orientations(q[cand], symmetry)
        o_ring = _orientations(q[ring], symmetry)
        # Pairwise only over the pairs that are actually within reach (a
        # k-d tree, never a candidates x ring matrix: a connected eutectic
        # network is tens of thousands of pixels and the dense matrix was
        # measured at 6.6 GB for 48 400 px).
        min_angle, rim_min = _min_angles_within(
            o_cand, crr, ccc, o_ring, rrr, rcc, ring_px, inner_radius=1)
        decidable = np.isfinite(min_angle)
        attached = decidable & (min_angle < float(angle_deg))
        cand_is_matrix = pid[cand] == pair.matrix_pid

        flip_up = decidable & ~attached & cand_is_matrix        # matrix -> particle

        # Backward direction. A particle-assigned pixel goes back to the
        # matrix only when it is attached to the matrix AND not part of the
        # particle's own crystal. The "free core" is the set of particle
        # pixels that do NOT continue the matrix: candidates that are not
        # attached, plus the core zone one ring further in (no matrix in
        # reach, core by definition). Measured on the crop: one 92-px Si
        # particle shares the orientation of the grain below it to 0.5 deg --
        # there continuity says nothing, and without this guard its 77 at% Si
        # core would have gone to Al. With an empty free core the blob's
        # orientation is indistinguishable from the matrix and no pixel goes
        # back.
        core_free = ~cand_is_matrix & ~attached
        zrr, zcc = rr[core_zone], cc[core_zone]
        n_core_free = int(core_free.sum()) + int(core_zone.sum())
        flip_down = np.zeros_like(attached)
        if n_core_free:
            q_core = np.concatenate([q[cand][core_free], q[core_zone]], axis=0)
            krr = np.concatenate([crr[core_free], zrr]); kcc = np.concatenate([ccc[core_free], zcc])
            to_core, _ = _min_angles_within(
                o_cand, crr, ccc, _orientations(q_core, symmetry), krr, kcc, ring_px)
            in_core = to_core < float(angle_deg)
            # ... and only at the rim: the matching matrix pixel must be a
            # direct neighbour. A matrix-oriented pixel 2-3 px inside an
            # enriched blob (77 at% Si on the crop) is not "the beam on the
            # matrix beside the particle"; whatever it is, this rule does not
            # know, so it leaves it.
            rim_attached = rim_min < float(angle_deg)
            flip_down = rim_attached & ~cand_is_matrix & ~in_core   # particle -> matrix
        entry["n_core_free"] = n_core_free
        new_pid[crr[flip_up], ccc[flip_up]] = pair.particle_pid
        new_pid[crr[flip_down], ccc[flip_down]] = pair.matrix_pid
        to_particle[crr[flip_up], ccc[flip_up]] = True
        to_matrix[crr[flip_down], ccc[flip_down]] = True

        entry["n_to_particle"] = int(flip_up.sum())
        entry["n_to_matrix"] = int(flip_down.sum())
        entry["n_undecidable"] += int((~decidable).sum())
        if decidable.any():
            entry["median_min_angle_deg"] = float(np.median(min_angle[decidable]))
        report["n_to_particle"] += entry["n_to_particle"]
        report["n_to_matrix"] += entry["n_to_matrix"]
        report["n_undecidable"] += entry["n_undecidable"]
        report["blobs"].append(entry)

    return RescueResult(new_pid, to_particle, to_matrix, report)
