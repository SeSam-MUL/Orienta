"""Pseudo-symmetry resolution for the spherical SHT indexer.

Cubic-approximant intermetallics (point groups 23, m-3) have Kikuchi-band geometry
with the full cubic (m-3m / 432) symmetry while the crystal is only m-3 / 23. The
SHT cross-correlation therefore lands on a WRONG pseudo-symmetric variant. This
module expands the spherical cc peaks by the pseudo-symmetry coset (supergroup \
group) and selects the variant consistent with a rough Hough band-geometry
orientation. The selected orientation has spherical (cc-peak) precision on the
Hough-correct variant; selection is pure quaternion math (no rendering).

See tasks/golive/spherical-vs-emsphinx-vs-hough-2026-06-27.md and
tasks/golive/pseudosym-resolver-plan.md.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _sym_quats(point_group: str) -> np.ndarray:
    """(M, 4) array of the crystal symmetry quaternions (w,x,y,z) for a point group."""
    from orix.quaternion.symmetry import _groups
    g = next((gr for gr in _groups if gr.name == point_group), None)
    if g is None:
        raise ValueError(f"Unknown point group {point_group!r}")
    return np.asarray(g.data, dtype=np.float64)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w,x,y,z), broadcasting over leading dims."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _qconj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


# Crystal point group -> pseudo-symmetry holohedry. A cubic-approximant crystal
# (these lower-symmetry cubic classes) has Kikuchi-band geometry with the FULL
# cubic m-3m symmetry, so the extra operations (m-3m \ group) are the pseudo-
# symmetric variants the SHT correlation can confuse.
# Only the two cubic classes whose PROPER-ROTATION group is a genuine subgroup of
# the band holohedry's rotations (so the coset is non-empty) belong here: m-3 (Th,
# rotations = T) and 23 (T). 432 (O) already has the full cubic rotation group, and
# -43m (Td) collapses to the same quaternion set as m-3m under the centrosymmetric
# holohedry used here — both yield an EMPTY coset, so declaring them pseudo-symmetric
# would advertise a resolution that can never run. They are deliberately excluded.
# (A -43m cubic approximant could be added later via a proper-rotation 432\T coset.)
_PSEUDO_HOLOHEDRY: dict[str, str] = {
    "23": "m-3m",
    "m-3": "m-3m",
}


def is_pseudosymmetric(point_group: str) -> bool:
    """True if `point_group` is a cubic-approximant class whose band geometry has a
    higher (pseudo-)symmetry than the crystal, so the AUTOMATIC SHT indexing path
    substitutes Hough orientations. This is a deliberately narrow set (only the
    classes we auto-handle). The user-driven manual variant tool
    (:func:`pseudosym_variant_quats`) covers a much wider range — see there."""
    return point_group in _PSEUDO_HOLOHEDRY


# Point groups whose master pattern has z-rotational symmetry order 2 (`z_rot==2`),
# for which the SHT-spherical SO(3) cross-correlation CANNOT form a sharp
# orientation peak — it lands in a wrong basin (root-caused 2026-06-27/28 over 16
# diagnostic harnesses; the true orientation is not even among the top cc bins).
#
# THE AUTHORITATIVE DETERMINANT IS THE MASTER'S `z_rot == 2` (use
# :func:`spherical_unreliable`). This name set is only a FALLBACK for when the
# z_rot value is unavailable. It must list every point group the SHT loader can
# emit (`sht_io._build_sg_to_pg_table` stores the crystal point group, NOT the
# Laue class) whose master is z_rot=2 — verified by reading every library .sht:
#   cubic        m-3, 23, -43m   (full m-3m band geometry, lower crystal symmetry)
#   orthorhombic mmm, 222, mm2   (2-fold about c → z_rot=2)
# Everything else is reliable: triclinic `-1` / monoclinic `2/m` are z_rot=1
# (WORK); tetragonal `4/mmm`, trigonal `-3m`/`-6m2`, hexagonal `6/mmm` and
# full-cubic `m-3m` / `432` are z_rot>=3 (WORK). For ALL z_rot=2 masters the
# orientation is taken from Hough band-geometry indexing.
# (`-43m` is z_rot=2 — e.g. Mg17Al12 — so it MUST be here, even though it is
# excluded from `is_pseudosymmetric` for a separate reason: its coset under the
# centrosymmetric holohedry is empty. The spherical correlation fails for it
# regardless of coset emptiness.)
_SPHERICAL_UNRELIABLE_PG: frozenset[str] = frozenset(
    {"m-3", "23", "-43m", "mmm", "222", "mm2"})


def spherical_unreliable_pointgroup(point_group) -> bool:
    """Name-based fallback for :func:`spherical_unreliable` — True for the point
    groups whose master is z_rot=2 (see :data:`_SPHERICAL_UNRELIABLE_PG`). Prefer
    :func:`spherical_unreliable` with the master's actual ``z_rot`` when available.
    Whitespace-tolerant because point-group strings arrive from file metadata."""
    return str(point_group).strip() in _SPHERICAL_UNRELIABLE_PG


def spherical_unreliable(z_rot=None, point_group=None) -> bool:
    """True if the SHT-spherical SO(3) correlation cannot index this master (its
    z-rotational symmetry order is 2 → flat cc volume, wrong basin) and the
    AUTOMATIC map-indexing path should take orientations from Hough instead.

    Prefers the master's actual ``z_rot`` (the EXACT determinant, identical to the
    per-pattern phase-test path in ``backend/api/routes/indexing.py`` which gates on
    ``z_rot == 2``); falls back to the point-group name (:func:`
    spherical_unreliable_pointgroup`) only when ``z_rot`` is unavailable. A SUPERSET
    of :func:`is_pseudosymmetric` (which is narrowly the cubic approximants m-3/23):
    it also covers orthorhombic ``mmm``/``222``/``mm2`` and cubic ``-43m`` — all
    z_rot=2, all defeating the capped spherical correlation."""
    if z_rot is not None:
        try:
            return int(z_rot) == 2
        except (TypeError, ValueError):
            pass
    return spherical_unreliable_pointgroup(point_group)


# Crystal-system holohedry used to generate pseudo-symmetric *variant candidates*
# for the USER-DRIVEN manual flip tool. This is broader than `_PSEUDO_HOLOHEDRY`
# (which gates the automatic Hough substitution): for the manual tool we want to
# offer the user every plausible pseudo-variant for ANY phase, and let render-NCC +
# the user's eye pick the right one. Trigonal classes map to the HEXAGONAL
# holohedry 6/mmm — that is the pseudo-symmetry that confuses quartz/corundum (the
# 60°-about-c ambiguity), not the trigonal holohedry. Metric-driven pseudo-symmetry
# (e.g. tetragonal pseudo-cubic at c/a≈1) is NOT captured by a point-group map; for
# those the Hough candidate (added separately by the route) is what helps.
_SYSTEM_HOLOHEDRY: dict[str, str] = {
    "1": "-1", "-1": "-1",
    "2": "2/m", "m": "2/m", "2/m": "2/m",
    "222": "mmm", "mm2": "mmm", "mmm": "mmm",
    "4": "4/mmm", "-4": "4/mmm", "4/m": "4/mmm", "422": "4/mmm",
    "4mm": "4/mmm", "-42m": "4/mmm", "-4m2": "4/mmm", "4/mmm": "4/mmm",
    "3": "6/mmm", "-3": "6/mmm", "32": "6/mmm", "3m": "6/mmm", "-3m": "6/mmm",
    "6": "6/mmm", "-6": "6/mmm", "6/m": "6/mmm", "622": "6/mmm",
    "6mm": "6/mmm", "-6m2": "6/mmm", "-62m": "6/mmm", "6/mmm": "6/mmm",
    "23": "m-3m", "m-3": "m-3m", "432": "m-3m", "-43m": "m-3m", "m-3m": "m-3m",
}


def pseudosym_holohedry(point_group: str):
    """The pseudo-symmetry holohedry used to generate variant candidates for
    `point_group` (or None if unknown). See :data:`_SYSTEM_HOLOHEDRY`."""
    return _SYSTEM_HOLOHEDRY.get(point_group)


def pseudosym_variant_quats(q, point_group: str, min_sep_deg: float = 3.0,
                            max_variants: int = 24) -> np.ndarray:
    """All distinct pseudo-symmetric variant orientations of `q` for `point_group`,
    for the user-driven manual flip tool — universal across crystal systems.

    Generates candidates by applying the holohedry operators (both crystal-frame
    ``q·h`` and sample-frame ``h·q``, so the right physical variant is included
    regardless of frame convention), then deduplicates under the TRUE crystal point
    group (symmetry-equivalents are the same orientation). The current orientation
    is returned FIRST (variant 0). The caller renders each and ranks by render-NCC;
    a phase that is already holohedral (or unknown) yields just ``[q]`` (no
    pseudo-variants — the route still offers the Hough orientation as a candidate).

    Parameters
    ----------
    q : (4,) array — current orientation quaternion (w, x, y, z).
    point_group : crystal point-group name.

    Returns
    -------
    (K, 4) ndarray — distinct variant quaternions, current orientation first.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    holo = pseudosym_holohedry(point_group)
    if holo is None or holo == point_group:
        return q[None, :]
    H = _sym_quats(holo)
    cands = np.vstack([q[None, :],
                       _qmul(H, q[None, :]),        # h · q  (sample frame)
                       _qmul(q[None, :], H)])       # q · h  (crystal frame)
    kept = [q]
    for cq in cands[1:]:
        ang = same_orientation_angle_deg(np.asarray(kept), cq, point_group)
        if float(np.min(ang)) > min_sep_deg:
            kept.append(cq)
        if len(kept) >= max_variants:
            break
    return np.asarray(kept, dtype=np.float64)


def _qkey(q: np.ndarray, ndigits: int = 4, tol: float = 1e-4) -> tuple:
    """Sign-canonical hashable key for a quaternion (q and -q are the same rotation)."""
    q = np.asarray(q, dtype=np.float64)
    for v in q:                      # make the first significant component positive
        if abs(v) > tol:
            if v < 0:
                q = -q
            break
    return tuple(np.round(q, ndigits))


@lru_cache(maxsize=None)
def coset_operators(point_group: str) -> np.ndarray:
    """Pseudo-symmetry coset operators (K, 4) = holohedry symmetry \\ crystal
    symmetry, as quaternions (w,x,y,z). Applying these to a cc-peak orientation
    generates the pseudo-symmetric variant candidates. m-3 -> 24 ops."""
    if point_group not in _PSEUDO_HOLOHEDRY:
        raise ValueError(f"{point_group!r} is not pseudo-symmetric (no holohedry mapping)")
    sup = _sym_quats(_PSEUDO_HOLOHEDRY[point_group])
    grp_keys = {_qkey(g) for g in _sym_quats(point_group)}
    coset = [s for s in sup if _qkey(s) not in grp_keys]
    # Always return a (K, 4) array, even when empty, so callers can index [:, :]
    # without a shape surprise.
    if not coset:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(coset, dtype=np.float64)


def disorientation_deg(qa, qb, point_group: str = "m-3"):
    """Disorientation angle in degrees between orientation quaternion(s) qa and qb
    under the crystal point-group symmetry (reduced on BOTH sides).

    Parameters
    ----------
    qa : (4,) or (N, 4) array — orientation quaternion(s) (w, x, y, z)
    qb : (4,) array — single reference orientation quaternion
    point_group : orix point-group name (e.g. "m-3", "m-3m")

    Returns
    -------
    float if qa is (4,) or (1,4); else (N,) ndarray.
    """
    S = _sym_quats(point_group)                      # (M, 4)
    qa = np.atleast_2d(np.asarray(qa, dtype=np.float64))   # (N, 4)
    qb = np.atleast_2d(np.asarray(qb, dtype=np.float64))   # (1, 4)
    m = _qmul(qa, _qconj(qb))                         # (N, 4) misorientation
    # both-sided symmetry reduction: min over s1, s2 of angle(s1 * m * s2)
    t = _qmul(S[None, :, :], m[:, None, :])          # (N, M, 4)  = s1 * m
    f = _qmul(t[:, :, None, :], S[None, None, :, :])  # (N, M, M, 4) = s1 * m * s2
    w = np.abs(f[..., 0]).reshape(m.shape[0], -1).max(axis=1)
    out = np.degrees(2.0 * np.arccos(np.clip(w, 0.0, 1.0)))
    return float(out[0]) if out.shape[0] == 1 else out


def same_orientation_angle_deg(qa, qb, point_group: str = "m-3"):
    """Smallest rotation angle (deg) taking each of `qa` to a crystal-symmetry
    equivalent of `qb` — the ONE-SIDED symmetry-reduced angle.

    For comparing two *orientations* of the same phase this equals the full
    :func:`disorientation_deg` (both give the disorientation angle), but it
    reduces by the point group on a single side (M ops, not M*M), so it is ~M
    times cheaper — which matters because the dedup runs per pixel over a whole
    map. The two functions only diverge when reducing a misorientation in the
    misorientation fundamental zone, which this resolver never does; we keep the
    both-sided `disorientation_deg` for the (rare, small) variant-selection step
    and use this fast one-sided form for the hot dedup loop.

    Parameters
    ----------
    qa : (N, 4) array — orientation quaternion(s) (w, x, y, z).
    qb : (4,) array — single reference orientation quaternion.
    point_group : orix point-group name.

    Returns
    -------
    (N,) ndarray of angles in degrees.
    """
    S = _sym_quats(point_group)                          # (M, 4)
    qa = np.atleast_2d(np.asarray(qa, dtype=np.float64))   # (N, 4)
    qb = np.asarray(qb, dtype=np.float64)                  # (4,)
    m = _qmul(qa, _qconj(qb))                              # (N, 4) misorientation
    w = np.abs(_qmul(S[None, :, :], m[:, None, :])[..., 0]).max(axis=1)  # (N,)
    return np.degrees(2.0 * np.arccos(np.clip(w, 0.0, 1.0)))


def resolve_variant(peak_quats, hough_quat, point_group: str):
    """Pick the pseudo-symmetric variant of the spherical cc peaks that is
    consistent with a rough Hough band-geometry orientation.

    Expands the cc peaks by the pseudo-symmetry coset (both compositions s*P and
    P*s) and returns the candidate closest to `hough_quat` under the crystal
    point-group symmetry. The returned orientation keeps the spherical cc-peak
    precision but on the Hough-correct pseudo-variant. No rendering.

    Parameters
    ----------
    peak_quats : (K, 4) array — top-K spherical cc-peak orientation quaternions.
    hough_quat : (4,) array — rough Hough orientation (only needs ~20 deg accuracy;
        the variants are 40-70 deg apart).
    point_group : crystal point group (must be pseudo-symmetric).

    Returns
    -------
    resolved : (4,) ndarray — the selected variant.
    candidates : (M, 4) ndarray — all expanded candidates (for a manual flip UI).
    diso_to_hough_deg : float — disorientation of `resolved` to Hough.
    """
    coset = coset_operators(point_group)                 # (C, 4)
    P = np.atleast_2d(np.asarray(peak_quats, dtype=np.float64))   # (K, 4)
    H = np.asarray(hough_quat, dtype=np.float64)
    cands = [P]
    if coset.shape[0]:                                   # skip if coset is empty
        cands.append(_qmul(coset[None, :, :], P[:, None, :]).reshape(-1, 4))  # s * P
        cands.append(_qmul(P[:, None, :], coset[None, :, :]).reshape(-1, 4))  # P * s
    C = np.concatenate(cands, axis=0)                    # (M, 4)
    d = np.atleast_1d(disorientation_deg(C, H, point_group))   # (M,) — atleast_1d
    i = int(np.argmin(d))                                # guards the M==1 scalar case
    return C[i], C, float(d[i])
