"""Orientation arithmetic for the analysis package.

Quaternions are (..., 4) float arrays (w, x, y, z) multiplied with the Hamilton
product, the convention orix uses. Crystal symmetry acts from the LEFT in orix:
an orientation ``o`` and ``s * o`` (``s`` a proper rotation of the phase's point
group) are the same crystal orientation. ``o * s`` is a different orientation;
averaging representatives chosen that way is what broke the old grain mean.

The disorientation angle is the smallest rotation angle among the symmetry
equivalent misorientations. With symmetry from the left the equivalents of
``m = qb * qa^-1`` are ``s2 m s1^-1``, whose angles are those of ``(s1^-1 s2) m``,
so a one-sided maximum over the group suffices (the smallest angle has the largest
``|w|``): ``max_s |w(s m)| = max_s |<m, conj(s)>|``.
The grain mean averages quaternions after moving each to the equivalent nearest
the current mean, as described for grain mean orientations in Bachmann,
Hielscher, Schaeben, Ultramicroscopy 111 (2011) 1720-1733,
doi:10.1016/j.ultramic.2011.08.002.
"""
from __future__ import annotations

import numpy as np

_CONJ = np.array([1.0, -1.0, -1.0, -1.0])


def qmul(a, b) -> np.ndarray:
    """Hamilton product ``a * b`` of broadcastable (..., 4) arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def qconj(q) -> np.ndarray:
    """Conjugate (inverse of a unit quaternion)."""
    return np.asarray(q, dtype=np.float64) * _CONJ


def proper_ops(symmetry) -> np.ndarray:
    """The proper rotations of an orix ``Symmetry`` as an (n, 4) array."""
    ops = np.asarray(symmetry.proper_subgroup.data, dtype=np.float64).reshape(-1, 4)
    if len(ops) == 0:
        raise ValueError(f"symmetry {symmetry!r} has no proper rotations")
    return ops


def disorientation_angle(qa, qb, ops, chunk: int = 262144) -> np.ndarray:
    """Smallest rotation angle (radians) between orientations ``qa`` and ``qb`` of one phase.

    ``qa``, ``qb``: (n, 4); ``ops``: the phase's proper rotations (k, 4). A NaN
    quaternion gives a NaN angle (orix's ``angle_with`` turns NaN into 0 deg,
    which once welded unindexed pixels to their neighbours). Pairs are processed
    ``chunk`` at a time so the (n, k) work array stays small for large n.
    """
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    qa, qb = np.broadcast_arrays(qa, qb)
    cops = qconj(ops)
    out = np.empty(qa.shape[:-1], dtype=np.float64)
    flat_a, flat_b, flat_out = qa.reshape(-1, 4), qb.reshape(-1, 4), out.reshape(-1)
    for i in range(0, len(flat_a), chunk):
        m = qmul(flat_b[i:i + chunk], qconj(flat_a[i:i + chunk]))
        # one operator at a time with plain element-wise products: O(chunk) memory
        # and a result that does not depend on the chunk size (a BLAS matmul does)
        w = np.zeros(len(m))
        for s in cops:
            w = np.maximum(w, np.abs(m[:, 0] * s[0] + m[:, 1] * s[1] + m[:, 2] * s[2] + m[:, 3] * s[3]))
        flat_out[i:i + chunk] = 2.0 * np.arccos(np.clip(w, 0.0, 1.0))
    return out


def min_disorientation_to_references(q, refs, ops, chunk: int = 16384) -> np.ndarray:
    """Per orientation ``q[i]`` the smallest disorientation angle (radians) to any of the reference orientations.

    ``q``: (n, 4); ``refs``: (m, 4), orientations of the same phase (for example the specimen-symmetry
    variants of one texture component); ``ops``: the phase's proper rotations (k, 4). The result equals
    ``min_j disorientation_angle(refs[j], q[i], ops)`` without pairing every orientation with every
    reference: with unit quaternions ``<q r^-1, s^-1> = <q, s^-1 r>``, so the largest ``|w|`` over all
    operators and references is the largest ``|<q, c>|`` over the (k * m) quaternions ``c = s^-1 r``, one
    matrix product per chunk. A NaN quaternion gives a NaN angle.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    refs = np.asarray(refs, dtype=np.float64).reshape(-1, 4)
    cand = qmul(qconj(ops)[:, None, :], refs[None, :, :]).reshape(-1, 4)       # s^-1 * r for every pair
    out = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), chunk):
        w = np.abs(q[i:i + chunk] @ cand.T).max(axis=1)
        out[i:i + chunk] = 2.0 * np.arccos(np.clip(w, 0.0, 1.0))
    return out


def nearest_equivalent(q, ref, ops, chunk: int = 65536) -> np.ndarray:
    """For every ``q[i]`` the equivalent ``s * q[i]`` closest to ``ref[i]``, sign-aligned with it."""
    q = np.asarray(q, dtype=np.float64)
    ref = np.broadcast_to(np.asarray(ref, dtype=np.float64), q.shape)
    out = np.empty_like(q)
    for i in range(0, len(q), chunk):
        qq, rr = q[i:i + chunk], ref[i:i + chunk]
        cand = qmul(ops[:, None, :], qq[None, :, :])              # (k, m, 4) = s * q
        dots = np.einsum("skc,kc->sk", cand, rr)
        best = np.abs(dots).argmax(axis=0)
        cols = np.arange(len(qq))
        sel = cand[best, cols]
        sign = np.where(dots[best, cols] < 0, -1.0, 1.0)
        out[i:i + chunk] = sel * sign[:, None]
    return out


def label_means(q, labels, n_labels: int, ops, iterations: int = 3) -> np.ndarray:
    """Mean orientation (n_labels, 4) of the pixels labelled 1..n_labels; label 0 ignored.

    Each label starts from its first pixel; every iteration moves all pixels to
    their equivalent nearest the current mean and averages. Labels without a
    finite pixel give NaN rows.
    """
    q = np.asarray(q, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    keep = (labels > 0) & np.isfinite(q).all(axis=1)
    q, labels = q[keep], labels[keep]
    means = np.full((n_labels + 1, 4), np.nan)
    if len(q) == 0:
        return means[1:]
    order = np.argsort(labels, kind="stable")
    starts = order[np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1]]
    means[labels[starts]] = q[starts]
    for _ in range(iterations):
        aligned = nearest_equivalent(q, means[labels], ops)
        sums = np.stack([np.bincount(labels, weights=aligned[:, c], minlength=n_labels + 1)
                         for c in range(4)], axis=1)
        norm = np.linalg.norm(sums, axis=1)
        ok = norm > 0
        means[ok] = sums[ok] / norm[ok, None]
    return means[1:]


def label_spread(q, labels, means, ops) -> np.ndarray:
    """Mean angle (radians) of each label's pixels to its mean (GOS); NaN for empty labels."""
    q = np.asarray(q, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    means = np.asarray(means, dtype=np.float64)
    n = len(means)
    keep = (labels > 0) & np.isfinite(q).all(axis=1)
    ang = disorientation_angle(q[keep], means[labels[keep] - 1], ops)
    total = np.bincount(labels[keep], weights=ang, minlength=n + 1)[1:]
    count = np.bincount(labels[keep], minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)
