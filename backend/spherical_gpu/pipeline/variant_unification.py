"""Map-wide pseudo-symmetry variant unification.

For phases whose orientations come from the (variant-blind) Hough anchor,
physical grains appear as salt-and-pepper mixtures of pseudo-symmetric
variants in the IPF map. This module removes that speckle map-wide, for ANY
point group, without corrupting real pseudo-merohedral twins:

1. **Segment grains modulo the SUPERGROUP** (system holohedry): variants
   collapse under the supergroup, so same-grain pixels re-unite regardless of
   which wrong variant they carry — pure quaternion math, no pattern NCC.
2. **Class labels + spatial-coherence policy** per grain:
   - salt-and-pepper class labels = mis-indexing (physically impossible) →
     ONE aggregated render-NCC decision, applied to the whole grain;
   - two-plus compact coherent domains = plausible pseudo-merohedral twin →
     each domain verified INDEPENDENTLY and flipped only on a CLEAR margin
     (hysteresis; default keep). Experimental-pattern NCC cannot make this
     distinction (twins share band positions) — coherence can.
3. **Rescue**: tiny orphan grains (Hough total failures that fit no neighbour
   even modulo the supergroup) adopt the enclosing grain's orientation.

Every pixel keeps ITS OWN measured orientation, snapped between variant
classes by left-multiplication (`q' = v_k · v_c⁻¹ · q` — Bunge/orix crystal
action), so intra-grain distortion survives (same math as grain-flip v2).

Scoring is injected (``score_fn``) so the core stays pure and unit-testable;
the wiring layers provide render-NCC scorers (pipeline: in-memory patterns;
endpoint: lazy per-pixel fetch). See
docs/superpowers/specs/2026-07-08-map-variant-unification-design.md.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from ..pseudosym import (
    _qconj,
    _qmul,
    _sym_quats,
    pseudosym_holohedry,
    same_orientation_angle_deg,
)


# ---------------------------------------------------------------------------
# Variant classes (generic over ALL point groups)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def proper_coset_class_reps(point_group: str) -> np.ndarray:
    """Representative quaternions of the distinct pseudo-variant classes for
    `point_group` — left cosets S\\H of the crystal group S in its system
    holohedry H, deduplicated numerically (improper operators share their
    quaternion with a proper one and collapse automatically).

    Class 0 is always the identity ("current variant"). A holohedral /
    unknown group returns just the identity (K=1 → nothing to unify).

    The dedup happens at the LAUE level automatically: orix stores improper
    operators with the quaternion of their proper part, so classes that only
    differ by an improper operation collapse. That is the physically correct
    granularity — Kikuchi patterns obey Friedel's law, so only Laue-distinct
    variants produce different patterns and ANY pattern-based method can only
    ever distinguish those. Consequences: m-3/23 → 2, trigonal -3m/3m/32 → 2,
    tetragonal 4/-4/4/m → 2, BUT -43m/4mm/mm2 → 1 (their "variants" are
    pattern-identical — nothing to decide, and no method could). Polar groups
    may still show IPF colour splits for pattern-identical orientations; that
    is an IPF-colouring convention matter, not an indexing error, and is out
    of scope here.
    """
    ident = np.array([[1.0, 0.0, 0.0, 0.0]])
    holo = pseudosym_holohedry(point_group)
    if holo is None or holo == point_group:
        return ident
    reps = [ident[0]]
    for h in _sym_quats(holo):
        d = same_orientation_angle_deg(np.asarray(reps), h, point_group)
        if float(np.min(d)) > 1.0:      # h·r⁻¹ ∉ S for every kept rep
            reps.append(np.asarray(h, dtype=np.float64))
    return np.asarray(reps, dtype=np.float64)


def metric_supergroup_ops(lattice, *, tol_ratio: float = 0.02,
                          tol_angle_deg: float = 1.0):
    """Extra candidate operators for METRIC pseudo-symmetry: a lattice whose
    metric is (within tolerance) that of a higher crystal system confuses band
    geometry beyond the point-group holohedry. v1 detects the dominant case,
    pseudo-CUBIC metric (a≈b≈c, all angles ≈90°) — e.g. tetragonal c/a≈1 —
    and returns the cubic holohedry operators. Returns None when the metric is
    not pseudo-cubic or `lattice` is unusable.

    `lattice` = (a, b, c, alpha, beta, gamma) in any consistent length unit /
    degrees. Friedel (inversion) polarity is NOT expressible as a rotation and
    is out of scope by physics.
    """
    try:
        a, b, c, al, be, ga = [float(x) for x in lattice]
    except (TypeError, ValueError):
        return None
    if a <= 0 or b <= 0 or c <= 0:
        return None
    lengths_close = (abs(b / a - 1.0) < tol_ratio and abs(c / a - 1.0) < tol_ratio)
    angles_right = all(abs(x - 90.0) < tol_angle_deg for x in (al, be, ga))
    if lengths_close and angles_right:
        return _sym_quats("m-3m")
    return None


def class_reps_for_phase(point_group: str, lattice=None) -> np.ndarray:
    """Variant class representatives for a phase: point-group holohedry coset
    classes, EXTENDED by metric-supergroup classes when the lattice metric is
    pseudo-cubic (dedup under the true group, identity first)."""
    reps = proper_coset_class_reps(point_group)
    extra = metric_supergroup_ops(lattice) if lattice is not None else None
    if extra is None:
        return reps
    out = [r for r in reps]
    for h in extra:
        d = same_orientation_angle_deg(np.asarray(out), h, point_group)
        if float(np.min(d)) > 1.0:
            out.append(np.asarray(h, dtype=np.float64))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Vectorised pairwise disorientation + supergroup grain segmentation
# ---------------------------------------------------------------------------

def pair_orientation_angle_deg(qa: np.ndarray, qb: np.ndarray,
                               sym: np.ndarray) -> np.ndarray:
    """Element-wise smallest rotation angle (deg) between orientation pairs
    ``qa[i] ↔ qb[i]`` reduced by the symmetry quaternions `sym` (left
    reduction, same convention as :func:`same_orientation_angle_deg`).
    Shapes: qa, qb (N,4); sym (M,4) → (N,)."""
    m = _qmul(np.atleast_2d(qa), _qconj(np.atleast_2d(qb)))       # (N,4)
    w = np.abs(_qmul(sym[None, :, :], m[:, None, :])[..., 0]).max(axis=1)
    return np.degrees(2.0 * np.arccos(np.clip(w, 0.0, 1.0)))


class _UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, i: int) -> int:
        p = self.p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[rj] = ri


def segment_supergroup_grains(full_q, phase_full, n_rows: int, n_cols: int,
                              phase_id, point_group: str,
                              threshold_deg: float = 5.0) -> np.ndarray:
    """Grain labels (n_rows*n_cols,) for one phase, segmented modulo the
    SUPERGROUP holohedry — variant-split grains come out as ONE grain, while a
    generically-rotated neighbour grain stays separate. -1 = not this phase /
    unindexed. Vectorised neighbour disorientations + union-find.
    """
    holo = pseudosym_holohedry(point_group) or point_group
    sym = _sym_quats(holo)
    n = n_rows * n_cols
    q = np.asarray(full_q, dtype=np.float64).reshape(n, 4)
    valid = (~np.isnan(q[:, 0])) & (np.asarray(phase_full).reshape(n) == int(phase_id))

    uf = _UnionFind(n)
    idx = np.arange(n).reshape(n_rows, n_cols)
    for axis_pairs in (
        (idx[:, :-1].ravel(), idx[:, 1:].ravel()),   # horizontal neighbours
        (idx[:-1, :].ravel(), idx[1:, :].ravel()),   # vertical neighbours
    ):
        a, b = axis_pairs
        ok = valid[a] & valid[b]
        a, b = a[ok], b[ok]
        if a.size == 0:
            continue
        d = pair_orientation_angle_deg(q[a], q[b], sym)
        for i, j in zip(a[d < threshold_deg], b[d < threshold_deg]):
            uf.union(int(i), int(j))

    labels = np.full(n, -1, dtype=np.int64)
    roots = {}
    for i in np.flatnonzero(valid):
        r = uf.find(int(i))
        labels[i] = roots.setdefault(r, len(roots))
    return labels


# ---------------------------------------------------------------------------
# Class labels + coherence analysis per grain
# ---------------------------------------------------------------------------

def class_label_pixels(q_pixels: np.ndarray, q_ref: np.ndarray,
                       class_reps: np.ndarray, point_group: str) -> np.ndarray:
    """Variant class index per pixel, relative to `q_ref`: pixel is in class k
    when its orientation is (modulo the TRUE group) closest to ``v_k · q_ref``.
    Classes are 40–90° apart; intra-grain spread is a few degrees, so argmin
    is unambiguous. Shapes: q_pixels (N,4) → (N,) int."""
    qp = np.atleast_2d(np.asarray(q_pixels, dtype=np.float64))
    d = np.stack([
        same_orientation_angle_deg(qp, _qmul(v[None, :], np.asarray(q_ref)[None, :])[0],
                                   point_group)
        for v in class_reps
    ], axis=1)                                        # (N, K)
    return np.argmin(d, axis=1).astype(np.int64)


def _components(pix_flat: np.ndarray, cls: np.ndarray, n_cols: int):
    """4-connected components of same-class pixels. Returns a list of
    (class_k, [flat indices]) sorted largest-first."""
    cls_of = {int(f): int(c) for f, c in zip(pix_flat, cls)}
    seen: set[int] = set()
    comps = []
    for f in pix_flat:
        f = int(f)
        if f in seen:
            continue
        k = cls_of[f]
        comp = [f]
        seen.add(f)
        dq = deque([f])
        while dq:
            cur = dq.popleft()
            r, c = divmod(cur, n_cols)
            for nb in (cur - n_cols, cur + n_cols,
                       cur - 1 if c > 0 else -1, cur + 1 if c < n_cols - 1 else -1):
                if nb in cls_of and nb not in seen and cls_of[nb] == k:
                    seen.add(nb)
                    comp.append(nb)
                    dq.append(nb)
        comps.append((k, comp))
    comps.sort(key=lambda t: -len(t[1]))
    return comps


def plan_grain_units(pix_flat: np.ndarray, cls: np.ndarray, n_cols: int,
                     *, domain_min_px: int = 8, speckle_frac_max: float = 0.2):
    """Decide how a supergroup-grain is treated (the twin-protection policy).

    Returns ``(mode, units)`` where each unit is
    ``{"pixels": ndarray flat, "current_class": int}``:

    - ``"speckle"``: class labels are spatially incoherent (or only one class
      present with noise) — the WHOLE grain is one decision unit; a physical
      grain cannot alternate variants per pixel, so unifying is always right.
    - ``"domains"``: ≥2 compact coherent domains of different classes with few
      speckle pixels — plausible pseudo-merohedral twin. Each large domain is
      its own unit (verified independently, conservative flip); small islands
      are absorbed into their surrounding domain's unit.
    """
    comps = _components(np.asarray(pix_flat), np.asarray(cls), n_cols)
    total = sum(len(c) for _k, c in comps)
    large = [(k, c) for k, c in comps if len(c) >= domain_min_px]
    small_px = total - sum(len(c) for _k, c in large)
    large_classes = {k for k, _c in large}
    if len(large) >= 2 and len(large_classes) >= 2 and small_px / max(total, 1) <= speckle_frac_max:
        units = []
        # absorb each small component into the large unit that surrounds it
        # (majority class among adjacent large-component pixels; fallback:
        # biggest domain).
        large_units = [{"pixels": list(c), "current_class": k} for k, c in large]
        owner_of = {}
        for ui, u in enumerate(large_units):
            for f in u["pixels"]:
                owner_of[f] = ui
        for k, c in comps:
            if len(c) >= domain_min_px:
                continue
            votes: dict[int, int] = {}
            for f in c:
                r, col = divmod(f, n_cols)
                for nb in (f - n_cols, f + n_cols,
                           f - 1 if col > 0 else -1, f + 1 if col < n_cols - 1 else -1):
                    if nb in owner_of:
                        votes[owner_of[nb]] = votes.get(owner_of[nb], 0) + 1
            ui = max(votes, key=votes.get) if votes else 0
            large_units[ui]["pixels"].extend(c)
        for u in large_units:
            u["pixels"] = np.asarray(sorted(u["pixels"]), dtype=np.int64)
            units.append(u)
        return "domains", units
    dominant = comps[0][0] if comps else 0
    return "speckle", [{
        "pixels": np.asarray(sorted(int(f) for f in pix_flat), dtype=np.int64),
        "current_class": int(dominant),
    }]


# ---------------------------------------------------------------------------
# Snap + top-level unification
# ---------------------------------------------------------------------------

def snap_to_class(q_pixels: np.ndarray, cls: np.ndarray, target_k: int,
                  class_reps: np.ndarray) -> np.ndarray:
    """Move each pixel from its current class to `target_k` by the LEFT
    operator ``v_k · v_c⁻¹`` (crystal action) — each pixel keeps its own
    measured deviation, so intra-grain texture is preserved."""
    qp = np.atleast_2d(np.asarray(q_pixels, dtype=np.float64)).copy()
    for c in np.unique(cls):
        if int(c) == int(target_k):
            continue
        op = _qmul(class_reps[int(target_k)][None, :],
                   _qconj(class_reps[int(c)][None, :]))[0]
        m = cls == c
        qp[m] = _qmul(op[None, :], qp[m])
    n = np.linalg.norm(qp, axis=1, keepdims=True)
    return qp / np.maximum(n, 1e-12)


def unify_map(full_q, phase_full, n_rows: int, n_cols: int, phase_id,
              point_group: str, score_fn, *,
              lattice=None,
              threshold_deg: float = 5.0,
              domain_min_px: int = 8,
              speckle_frac_max: float = 0.2,
              score_pixels_max: int = 8,
              margin_clear: float = 0.03,
              margin_ambiguous: float = 0.01,
              rescue_max_px: int = 2,
              min_grain_px: int = 3,
              adopt_max_px: int = 128,
              progress=None):
    """Unify pseudo-variant speckle for ONE phase across the whole map.

    ``score_fn(flat_indices (n,), quats (n,4)) -> (n,) float`` returns the
    render-NCC of each pixel's experimental pattern vs the SHT forward render
    at the given orientation (injected — see module docstring).

    Returns ``(new_full_q, report)``; ``new_full_q`` is a modified COPY of
    `full_q` (or None when the phase has only one variant class → no-op).
    ``report`` carries per-grain decisions for provenance / the UI:
    ``{"n_grains", "n_flipped_units", "n_ambiguous", "n_rescued",
    "grains": [{"pixels", "mode", "decision", "margin", "centroid"}...]}``.
    """
    class_reps = class_reps_for_phase(point_group, lattice)
    K = class_reps.shape[0]
    if K < 2:
        return None, {"n_grains": 0, "skipped": "single variant class",
                      "n_flipped_units": 0, "n_ambiguous": 0, "n_rescued": 0,
                      "grains": []}

    n = n_rows * n_cols
    q = np.asarray(full_q, dtype=np.float64).reshape(n, 4)
    new_q = q.copy()
    labels = segment_supergroup_grains(q, phase_full, n_rows, n_cols,
                                       phase_id, point_group, threshold_deg)
    grain_ids = [g for g in np.unique(labels) if g >= 0]
    report = {"n_grains": len(grain_ids), "n_flipped_units": 0,
              "n_ambiguous": 0, "n_rescued": 0, "grains": []}

    def _emit(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    tiny: list[np.ndarray] = []
    for gi, g in enumerate(grain_ids):
        pix = np.flatnonzero(labels == g)
        if pix.size <= rescue_max_px:
            tiny.append(pix)
            continue
        if pix.size < min_grain_px:
            continue
        q_ref = q[pix[0]]
        cls = class_label_pixels(q[pix], q_ref, class_reps, point_group)
        mode, units = plan_grain_units(pix, cls, n_cols,
                                       domain_min_px=domain_min_px,
                                       speckle_frac_max=speckle_frac_max)
        cls_of = dict(zip(pix.tolist(), cls.tolist()))
        for unit in units:
            upix = unit["pixels"]
            ucls = np.asarray([cls_of[int(f)] for f in upix], dtype=np.int64)
            # sample pixels for scoring, evenly across the unit
            ns = int(min(score_pixels_max, max(3, upix.size // 10), upix.size))
            step = max(1, upix.size // ns)
            sample = upix[::step][:ns]
            scls = np.asarray([cls_of[int(f)] for f in sample], dtype=np.int64)
            medians = np.empty(K)
            for k in range(K):
                cand = snap_to_class(q[sample], scls, k, class_reps)
                s = np.asarray(score_fn(sample, cand), dtype=np.float64)
                medians[k] = float(np.median(s)) if s.size else float("-inf")
            order = np.argsort(medians)[::-1]
            best, second = int(order[0]), int(order[1])
            with np.errstate(invalid="ignore"):
                # -inf − -inf == NaN when every candidate failed to render;
                # the not-finite guard below handles it (margin is reset there).
                margin = float(medians[best] - medians[second])
            cur = int(unit["current_class"])
            decision = "keep"
            if not np.isfinite(medians[best]):
                # Every scored candidate failed to render (e.g. persistent GPU
                # OOM → score_fn returned -inf; -inf − -inf is also NaN, which
                # would slip past the margin comparisons below). A -inf score
                # must NEVER pick a winner: keep the current/dominant variant
                # and flag the grain ambiguous.
                target = cur
                margin = 0.0
                report["n_ambiguous"] += 1
                decision = "keep_ambiguous"
            elif mode == "speckle":
                # unify ALWAYS (speckle is unphysical). Trust in the winner is
                # the BEST-vs-SECOND margin: a clear margin → the score winner;
                # a flat signal → the DOMINANT current class (least change, no
                # arbitrary tie-break flip) and flag the grain as ambiguous.
                if margin < margin_ambiguous:
                    target = cur
                    report["n_ambiguous"] += 1
                    decision = "unified_ambiguous"
                else:
                    target = best
                    decision = "unified"
            else:
                # coherent domain (possible twin): conservative — flip only on
                # a CLEAR margin in favour of a different class
                if best != cur and margin >= margin_clear:
                    target = best
                    decision = "flipped"
                else:
                    target = cur
                    if best != cur:
                        report["n_ambiguous"] += 1
                        decision = "keep_ambiguous"
            # ALWAYS snap the unit onto its target class: even a kept coherent
            # domain may contain absorbed speckle islands of the other class —
            # snap_to_class is a no-op for pixels already in the target class.
            new_q[upix] = snap_to_class(q[upix], ucls, target, class_reps)
            if int(np.count_nonzero(ucls != target)):
                report["n_flipped_units"] += 1
            rr, cc = np.divmod(upix, n_cols)
            report["grains"].append({
                "pixels": int(upix.size), "mode": mode, "decision": decision,
                "margin": round(margin, 4), "class": int(target),
                "centroid": [int(round(rr.mean())), int(round(cc.mean()))],
            })
        _emit(f"Variant unification: grain {gi + 1}/{len(grain_ids)} "
              f"({pix.size} px, {mode})")

    # Stage 2.5 — render-verified SMALL-GRAIN ADOPTION. Observed on real data
    # (Scan1 alpha-AlFeMnSi): ~20 small "grains" all sitting at the IDENTICAL
    # misorientation (71.9° under m-3) to the surrounding big grain — a
    # systematic second Hough basin (band-coincidence rotation) that is NOT in
    # the holohedry coset, so the class machinery above correctly leaves it
    # alone. Generic remedy that needs no knowledge of the specific operator:
    # for each small grain g adjacent to a much bigger grain G, build the
    # constant map C = mean(G)·mean(g)⁻¹ (left — preserves g's internal
    # texture) and let render-NCC decide: adopt G's branch only on a CLEAR
    # margin. A REAL small grain renders better in its own orientation and is
    # kept bit-identical — same primum-non-nocere policy as the twin domains.
    grain_pix = {int(g): np.flatnonzero(labels == g) for g in grain_ids}
    grain_size = {g: p.size for g, p in grain_pix.items()}
    holo_sym = _sym_quats(pseudosym_holohedry(point_group) or point_group)

    def _branch_mean(pix):
        ref = new_q[pix[0]]
        qs = np.empty((pix.size, 4))
        for k, i in enumerate(pix):
            cands = _qmul(holo_sym, new_q[i][None, :])
            b = cands[int(np.argmax(np.abs(cands @ ref)))]
            qs[k] = -b if float(np.dot(b, ref)) < 0 else b
        m = qs.mean(axis=0)
        return m / max(float(np.linalg.norm(m)), 1e-12)

    adopted_grains: set[int] = set()
    for g, pix in grain_pix.items():
        if pix.size <= rescue_max_px or pix.size > adopt_max_px:
            continue
        # boundary-adjacency census → the dominant, much-bigger neighbour grain
        contact: dict[int, int] = {}
        for f in pix:
            r, c = divmod(int(f), n_cols)
            for nb in (f - n_cols, f + n_cols,
                       f - 1 if c > 0 else -1, f + 1 if c < n_cols - 1 else -1):
                nb = int(nb)
                if 0 <= nb < n and labels[nb] >= 0 and labels[nb] != g:
                    contact[int(labels[nb])] = contact.get(int(labels[nb]), 0) + 1
        donors = [G for G in contact if grain_size.get(G, 0) >= max(
            3 * pix.size, domain_min_px)]
        if not donors:
            continue
        G = max(donors, key=lambda d: contact[d])
        C = _qmul(_branch_mean(grain_pix[G])[None, :],
                  _qconj(_branch_mean(pix)[None, :]))[0]
        ns = int(min(score_pixels_max, pix.size))
        step = max(1, pix.size // ns)
        sample = pix[::step][:ns]
        s_own = np.asarray(score_fn(sample, new_q[sample]), dtype=np.float64)
        mapped = _qmul(C[None, :], new_q[sample])
        s_map = np.asarray(score_fn(sample, mapped), dtype=np.float64)
        med_own = float(np.median(s_own)) if s_own.size else float("-inf")
        med_map = float(np.median(s_map)) if s_map.size else float("-inf")
        rr, cc = np.divmod(pix, n_cols)
        entry = {"pixels": int(pix.size), "mode": "adoption",
                 "class": -1, "centroid": [int(round(rr.mean())), int(round(cc.mean()))]}
        if np.isfinite(med_map) and (med_map - (med_own if np.isfinite(med_own)
                                                else float("-inf"))) >= margin_clear:
            allq = _qmul(C[None, :], new_q[pix])
            new_q[pix] = allq / np.maximum(
                np.linalg.norm(allq, axis=1, keepdims=True), 1e-12)
            adopted_grains.add(g)
            report["n_flipped_units"] += 1
            entry["decision"] = "adopted"
            entry["margin"] = round(med_map - med_own, 4) if np.isfinite(med_own) else None
        else:
            entry["decision"] = "kept"
            entry["margin"] = (round(med_map - med_own, 4)
                               if np.isfinite(med_map) and np.isfinite(med_own) else None)
            if np.isfinite(med_map) and np.isfinite(med_own) and \
                    abs(med_map - med_own) < margin_ambiguous:
                report["n_ambiguous"] += 1
                entry["decision"] = "kept_ambiguous"
        report["grains"].append(entry)
    report["n_adopted"] = len(adopted_grains)

    # Stage 3 — rescue tiny orphans (Hough total failures): adopt the
    # orientation of the adjacent unified grain (majority neighbour).
    for pix in tiny:
        votes: dict[int, int] = {}
        donor: dict[int, int] = {}
        for f in pix:
            r, c = divmod(int(f), n_cols)
            for nb in (f - n_cols, f + n_cols,
                       f - 1 if c > 0 else -1, f + 1 if c < n_cols - 1 else -1):
                nb = int(nb)
                if 0 <= nb < n and labels[nb] >= 0 and labels[nb] != labels[int(f)]:
                    g = int(labels[nb])
                    votes[g] = votes.get(g, 0) + 1
                    donor[g] = nb
        if not votes:
            continue
        g = max(votes, key=votes.get)
        new_q[pix] = new_q[donor[g]]
        report["n_rescued"] += int(pix.size)

    return new_q, report


# ---------------------------------------------------------------------------
# Wiring — render-NCC scorer + pipeline entry point (lazy heavy imports)
# ---------------------------------------------------------------------------

def build_render_score_fn(sht_path: str, det_params: dict, get_pattern,
                          max_bandwidth: int = 128):
    """Render-NCC scorer for :func:`unify_map`.

    ``get_pattern(flat_idx) -> (H, W) float array | None`` supplies the
    experimental pattern (in-memory batch for the pipeline; lazy per-pixel
    fetch for the endpoint — only the few scored pixels per grain are read).

    Renders through the cached SHT renderer service (bw=`max_bandwidth`),
    removes the dynamic background from BOTH sides and compares inside the
    inscribed detector disc — the same recipe as the validated SampleB
    render-NCC ground-truth harness. Pixels without a pattern score -inf so
    they never decide a class.

    GPU-OOM resilience: all torch work runs under ``torch.no_grad()`` (scoring
    never needs autograd graphs), and a per-pixel render that raises a CUDA
    out-of-memory error triggers a ONE-TIME cache release + renderer rebuild
    (shared across every scored pixel via a small mutable closure state) and a
    single retry. If the retry still OOMs the pixel scores -inf — it never
    decides a class (see :func:`unify_map`'s -inf guard).
    """
    import logging
    import torch
    import kikuchipy as kp
    # Reference the renderer service through the MODULE (not `from ... import`)
    # so get_renderer / load_or_get_phase / release_gpu_caches stay patchable
    # seams and the OOM rebuild picks up a freshly-cleared singleton.
    from backend.api.services import sht_pattern_renderer as _sht_service
    from .detector import convert_pc_to_emsoft

    log = logging.getLogger(__name__)

    H = int(det_params["pat_height"])
    W = int(det_params["pat_width"])
    xpc, ypc, L_um = convert_pc_to_emsoft(
        pc=(float(det_params["pc_x"]), float(det_params["pc_y"]),
            float(det_params["pc_z"])),
        vendor=str(det_params.get("vendor", "Bruker")),
        pat_width=W, pat_height=H,
        pixel_size=float(det_params.get("pixel_size", 70.0)),
        binning=int(det_params.get("binning", 1)),
    )
    pc = (float(xpc), float(ypc), float(L_um))
    pixel_size = float(det_params.get("pixel_size", 70.0))
    sample_tilt = float(det_params.get("sample_tilt", 70.0))
    det_tilt = float(det_params.get("tilt", 0.0))

    # Renderer + phase grid live in a small mutable closure state so an OOM
    # recovery rebuilds them ONCE, shared across every scored pixel/call, rather
    # than per pixel.
    state = {
        "rnd": _sht_service.get_renderer(),
        "pgrid": _sht_service.load_or_get_phase(
            str(sht_path), max_bandwidth=int(max_bandwidth)),
    }

    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (min(cy, cx) * 0.96) ** 2

    def _dynbg(im):
        s = kp.signals.EBSD(np.asarray(im, dtype=np.float32)[None, None])
        s.remove_dynamic_background(operation="subtract", filter_domain="frequency")
        return s.data[0, 0].astype(np.float32)

    def _ncc(a, b):
        av = a[disc].astype(np.float64)
        bv = b[disc].astype(np.float64)
        av -= av.mean()
        bv -= bv.mean()
        den = np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12
        return float(np.dot(av, bv) / den)

    def _is_oom(exc):
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _reacquire_renderer():
        """Free GPU caches and rebuild the renderer + phase grid in-place after
        a CUDA OOM, so the retry (and every later pixel) uses fresh handles."""
        try:
            _sht_service.release_gpu_caches()
        except Exception:
            log.debug("variant-unify OOM recovery: cache release failed",
                      exc_info=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            log.debug("variant-unify OOM recovery: empty_cache failed",
                      exc_info=True)
        state["rnd"] = _sht_service.get_renderer()
        state["pgrid"] = _sht_service.load_or_get_phase(
            str(sht_path), max_bandwidth=int(max_bandwidth))

    def _render(qt):
        sim = state["rnd"].render(state["pgrid"], qt, pc, (H, W), pixel_size,
                                  tilt_deg=sample_tilt, det_tilt_deg=det_tilt)
        return _dynbg(np.asarray(sim.numpy(), dtype=np.float32))

    exp_cache: dict[int, np.ndarray | None] = {}

    def score(flat_idx, quats):
        out = []
        with torch.no_grad():
            for f, q in zip(np.atleast_1d(flat_idx), np.atleast_2d(quats)):
                f = int(f)
                if f not in exp_cache:
                    p = get_pattern(f)
                    exp_cache[f] = None if p is None else _dynbg(p)
                exp = exp_cache[f]
                if exp is None:
                    out.append(float("-inf"))
                    continue
                qt = torch.tensor(np.asarray(q, dtype=np.float64)[:4],
                                  dtype=torch.float64)
                try:
                    rendered = _render(qt)
                except RuntimeError as e:
                    if not _is_oom(e):
                        raise
                    # First OOM for this pixel: free the interactive VRAM,
                    # rebuild the renderer once, retry ONCE. Persistent OOM →
                    # -inf so this pixel never decides a variant class.
                    try:
                        _reacquire_renderer()
                        rendered = _render(qt)
                    except RuntimeError as e2:
                        if _is_oom(e2):
                            log.warning(
                                "variant-unify render OOM persisted after cache "
                                "release for pixel %d — scoring it -inf", f)
                            out.append(float("-inf"))
                            continue
                        raise
                out.append(_ncc(exp, rendered))
        return np.asarray(out, dtype=np.float64)

    return score


def unify_after_hough_resolve(eulers, phase_id, patterns, sht_paths,
                              masters_meta, det_params, selection_mask,
                              roi_mode: bool, resolved_phase_ids,
                              progress=None):
    """Pipeline entry: run map-wide variant unification for every phase whose
    orientations were just substituted from (variant-blind) Hough.

    `eulers` (N,3) Bunge-ZXZ radians in result order; `patterns` (N,H,W) same
    order; `phase_id` (N,) 1-indexed. Returns ``(eulers_new | None, reports)``
    — None when nothing changed. Fail-safe per phase: an error leaves that
    phase's orientations as delivered by the resolver.
    """
    from orix.quaternion import Rotation

    n_rows = int(det_params.get("n_rows") or 0)
    n_cols = int(det_params.get("n_cols") or 0)
    N = int(np.asarray(eulers).shape[0])
    reports: dict[int, dict] = {}

    # Map result rows -> full-grid flat indices (same convention as the
    # CrystalMap placement below the call site).
    if roi_mode and selection_mask is not None:
        sel = np.asarray(selection_mask, dtype=bool)
        n_rows, n_cols = sel.shape
        flat_of_row = np.flatnonzero(sel.ravel())
        if flat_of_row.size != N:
            return None, {"skipped": "selection mask does not match result size"}
    else:
        if n_rows * n_cols != N:
            return None, {"skipped": "no 2D grid (streamed/1D result)"}
        flat_of_row = np.arange(N)

    row_of_flat = np.full(n_rows * n_cols, -1, dtype=np.int64)
    row_of_flat[flat_of_row] = np.arange(N)

    q_rows = np.asarray(Rotation.from_euler(np.asarray(eulers)).data,
                        dtype=np.float64).reshape(N, 4)
    full_q = np.full((n_rows * n_cols, 4), np.nan)
    full_q[flat_of_row] = q_rows
    phase_full = np.full(n_rows * n_cols, -1, dtype=np.int64)
    phase_full[flat_of_row] = np.asarray(phase_id).reshape(-1)

    pats = np.asarray(patterns)

    def _get_pattern(flat):
        r = int(row_of_flat[flat])
        return None if r < 0 else pats[r]

    changed = False
    for pid in sorted(resolved_phase_ids):
        meta = masters_meta[pid - 1] if 0 <= pid - 1 < len(masters_meta) else {}
        pg = (meta or {}).get("point_group")
        sht = sht_paths[pid - 1] if 0 <= pid - 1 < len(sht_paths) else None
        if not pg or not sht:
            continue
        try:
            score_fn = build_render_score_fn(sht, det_params, _get_pattern)
            new_full, rep = unify_map(full_q, phase_full, n_rows, n_cols,
                                      pid, pg, score_fn, progress=progress)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "variant unification failed for phase %s — keeping resolver "
                "output for it", pid, exc_info=True)
            continue
        reports[pid] = rep
        if new_full is not None and not np.allclose(new_full, full_q, equal_nan=True):
            full_q = new_full
            changed = True

    if not changed:
        return None, reports
    eulers_new = np.asarray(
        Rotation(full_q[flat_of_row]).to_euler(), dtype=np.float64)
    return eulers_new, reports
