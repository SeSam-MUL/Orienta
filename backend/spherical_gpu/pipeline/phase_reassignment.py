"""Render-verified phase check + reassignment (grain-based).

Problem: multi-phase indexing assigns each pixel to the phase with the best
*internal* correlation score, but per-phase scores from different SHT
libraries are NOT mutually comparable (different master normalisation, band
structure, symmetry degeneracy). A chemically/structurally degenerate phase
(e.g. a cubic m-3 approximant on an Al m-3m matrix) can therefore "steal"
pixels of another phase while looking confident on its own scale.

The only cross-phase comparable signal is the forward render: simulate the
pattern for each candidate phase at that phase's best orientation and
correlate with the experiment (render-NCC) — the same lesson as the
pseudo-symmetry variant work ("judge by render, never by internal scores").

Stage A (:func:`check_map`) is read-only: per grain, score the STORED phase
at the stored orientations and every candidate phase at a Hough-anchored
orientation, over a few core sample pixels. Produces a per-grain report and
a per-pixel margin map (stored − best alternative; negative = the stored
phase loses) for a diagnostic layer.

Stage B (:func:`apply_reassignment`) flips only grains whose check decision
was ``reassign`` (margin beats a hysteresis, default 0.05 — stricter than
variant unification because a phase change is the bigger claim), whole grain
or nothing, with per-pixel Hough orientations for the winning phase and
nearest-neighbour fill for pixels where Hough fails. Callers store undo data.

Candidate orientations come from HOUGH with the candidate's reflectors —
never from a per-phase spherical re-index, which systematically
underestimates z_rot==2 phases (their SO(3) correlation lands on a
high-symmetry pole; root-caused 2026-06-27/28). The stored phase is scored
at its stored orientation (for z_rot==2 phases that is already
Hough-anchored by the pipeline). Candidates where Hough fails are skipped —
fail-safe = keep the stored phase.
"""
from __future__ import annotations

import logging

import numpy as np

from .variant_unification import segment_supergroup_grains

logger = logging.getLogger(__name__)

# Grains whose stored phase renders at least this well are never questioned
# (fast path — no candidate scoring, no margin entry). On validated data,
# correct assignments render ≥ ~0.3 and wrong-phase assignments ~0.17.
SCORE_FLOOR = 0.25
# A candidate must beat the stored phase by this much (aggregated render-NCC)
# before a grain is offered for reassignment. Stricter than the 0.03 used for
# variant unification: changing the phase is a bigger claim than changing the
# variant.
MARGIN_CLEAR = 0.05
MIN_GRAIN_PX = 5
SAMPLE_PX_MAX = 16


def _grain_core_sample(pix: np.ndarray, n_rows: int, n_cols: int,
                       k: int = SAMPLE_PX_MAX) -> np.ndarray:
    """Pick up to ``k`` flat indices from a grain, preferring CORE pixels
    (all 4-neighbours inside the grain) — boundary pixels carry mixed
    signal from the neighbouring grain/phase and must not decide a phase.
    Evenly spread over the (sorted) core so one corner can't dominate.
    """
    pix = np.asarray(sorted(int(p) for p in pix), dtype=np.int64)
    inset = set(pix.tolist())
    n = n_rows * n_cols
    core = []
    for p in pix:
        r, c = divmod(int(p), n_cols)
        nbs = []
        if r > 0:
            nbs.append(p - n_cols)
        if r < n_rows - 1:
            nbs.append(p + n_cols)
        if c > 0:
            nbs.append(p - 1)
        if c < n_cols - 1:
            nbs.append(p + 1)
        if len(nbs) == 4 and all(int(nb) in inset for nb in nbs):
            core.append(int(p))
    base = core if len(core) >= max(3, k // 2) else pix.tolist()
    if len(base) <= k:
        return np.asarray(base, dtype=np.int64)
    step = len(base) / float(k)
    return np.asarray([base[int(i * step)] for i in range(k)], dtype=np.int64)


def _agg(scores: np.ndarray) -> float:
    """Aggregate per-pixel render-NCC scores for one grain: median over the
    finite values (same estimator as variant unification). -inf when nothing
    scored — never wins."""
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    return float(np.median(s)) if s.size else float("-inf")


def check_map(full_q, phase_full, n_rows: int, n_cols: int,
              phases: dict, score_fns: dict, hough_quats_fn, *,
              sample_px_max: int = SAMPLE_PX_MAX,
              score_floor: float = SCORE_FLOOR,
              margin_clear: float = MARGIN_CLEAR,
              min_grain_px: int = MIN_GRAIN_PX,
              threshold_deg: float = 5.0,
              progress=None):
    """Stage A — render-verified phase check over the whole map.

    Parameters
    ----------
    full_q : (n_rows*n_cols, 4) quaternions, NaN rows = unindexed.
    phase_full : (n_rows*n_cols,) int phase ids, -1 = unindexed.
    phases : {pid: point_group_name} for every phase present.
    score_fns : {pid: score_fn(flat_idx (n,), quats (n,4)) -> (n,)} — render
        scorer per phase (only phases present here can be scored; a stored
        phase without a scorer is reported ``no-sht`` and skipped).
    hough_quats_fn : (candidate_pid, flat_idx (n,)) -> (n, 4) quats with NaN
        rows where Hough failed. Injected so the core stays free of route /
        pattern-IO concerns (and trivially testable).

    Returns
    -------
    (report, margin_full)
        report: {"grains": [entry...], "n_grains", "n_checked", "n_suspect",
                 "n_reassign", "params": {...}}
        margin_full: (n_rows*n_cols,) float64 — ``stored − best alternative``
        broadcast over each SUSPECT grain's pixels (positive = stored phase
        wins, negative = a candidate renders better). NaN for healthy /
        unchecked pixels, so the diagnostic layer only lights up where a
        question was actually asked and answered.
    """
    n = n_rows * n_cols
    q = np.asarray(full_q, dtype=np.float64).reshape(n, 4)
    pf = np.asarray(phase_full).reshape(n)
    margin_full = np.full(n, np.nan, dtype=np.float64)

    entries: list[dict] = []
    n_grains = n_checked = n_suspect = n_reassign = 0

    pids = sorted(int(p) for p in np.unique(pf) if int(p) >= 0)
    for pid in pids:
        pg = phases.get(pid)
        if pg is None:
            continue
        labels = segment_supergroup_grains(
            q, pf, n_rows, n_cols, pid, pg, threshold_deg=threshold_deg)
        grain_ids = [g for g in np.unique(labels) if g >= 0]
        for g in grain_ids:
            pix = np.flatnonzero(labels == g)
            if pix.size < min_grain_px:
                continue
            n_grains += 1
            rr, cc = np.divmod(pix, n_cols)
            entry = {
                "grain_id": int(g),
                "phase_id": pid,
                "pixels": int(pix.size),
                "centroid": [float(rr.mean()), float(cc.mean())],
            }
            if pid not in score_fns:
                entry["decision"] = "no-sht"
                entries.append(entry)
                continue

            sample = _grain_core_sample(pix, n_rows, n_cols, k=sample_px_max)
            stored = _agg(score_fns[pid](sample, q[sample]))
            entry["stored_score"] = None if not np.isfinite(stored) else round(stored, 4)
            n_checked += 1
            if progress is not None:
                try:
                    progress(pid, int(g), len(grain_ids))
                except Exception:
                    pass

            if not np.isfinite(stored):
                # The STORED phase's render failed (e.g. transient GPU OOM) —
                # that is a scoring failure, not evidence against the phase.
                # Never let a candidate win by default: fail-safe = keep.
                entry["decision"] = "no-score"
                entries.append(entry)
                continue
            if stored >= score_floor:
                entry["decision"] = "ok"
                entries.append(entry)
                continue

            # Suspect: score every candidate phase at its Hough orientation.
            n_suspect += 1
            alt_scores: dict[int, float] = {}
            for cand in pids:
                if cand == pid or cand not in score_fns:
                    continue
                try:
                    cq = np.asarray(hough_quats_fn(cand, sample),
                                    dtype=np.float64).reshape(-1, 4)
                except Exception:
                    logger.warning("[phase-check] hough for candidate %s failed",
                                   cand, exc_info=True)
                    continue
                ok = np.isfinite(cq[:, 0])
                # A candidate scored on a tiny Hough-lucky subset would be
                # compared against the stored phase's FULL-sample median —
                # asymmetric and biased. Require a minimum successful count.
                if int(ok.sum()) < min(3, sample.size):
                    continue
                s = _agg(score_fns[cand](sample[ok], cq[ok]))
                if np.isfinite(s):
                    alt_scores[cand] = round(float(s), 4)

            if alt_scores:
                best_alt = max(alt_scores, key=lambda k_: alt_scores[k_])
                best_alt_score = alt_scores[best_alt]
                margin = (stored if np.isfinite(stored) else -1.0) - best_alt_score
                entry.update({
                    "alt_scores": alt_scores,
                    "best_alt_phase": int(best_alt),
                    "best_alt_score": best_alt_score,
                    "margin": round(float(margin), 4),
                })
                margin_full[pix] = margin
                if margin <= -margin_clear:
                    entry["decision"] = "reassign"
                    n_reassign += 1
                else:
                    entry["decision"] = "keep"
            else:
                entry["decision"] = "keep"  # nothing comparable → conservative
            entries.append(entry)

    report = {
        "grains": entries,
        "n_grains": n_grains,
        "n_checked": n_checked,
        "n_suspect": n_suspect,
        "n_reassign": n_reassign,
        "params": {
            "score_floor": score_floor,
            "margin_clear": margin_clear,
            "sample_px_max": sample_px_max,
            "min_grain_px": min_grain_px,
            "threshold_deg": threshold_deg,
        },
    }
    return report, margin_full


def apply_reassignment(full_q, phase_full, n_rows: int, n_cols: int,
                       report: dict, phases: dict, hough_quats_fn, *,
                       progress=None):
    """Stage B — apply the ``reassign`` decisions from :func:`check_map`.

    For every grain with ``decision == "reassign"``: set its pixels to the
    winning phase and give each pixel its own Hough orientation from that
    phase (per-pixel, so intra-grain texture is real, not a rigid copy).
    Pixels where Hough fails inherit the quat of the nearest (flat-order)
    successful pixel; if Hough fails for the WHOLE grain the grain is left
    untouched (fail-safe = stored phase survives).

    Grain identity is re-derived by re-segmenting the CURRENT orientations,
    and grain ids are only stable while the map is unchanged. Two layers of
    protection: the routes invalidate ``phase_check`` metadata on every
    orientation edit (grain flip / unify / undo), AND each re-derived grain
    is verified against the report entry's pixel count + centroid — a
    mismatch skips the grain (``grain changed since check``) instead of
    flipping whatever now happens to carry that id.

    Returns ``(new_phase_full, new_full_q, applied, skipped)`` where
    ``applied`` / ``skipped`` are lists of grain entries (with
    ``"n_hough_filled"`` on applied entries counting nearest-fill pixels).
    """
    n = n_rows * n_cols
    q = np.asarray(full_q, dtype=np.float64).reshape(n, 4).copy()
    pf = np.asarray(phase_full).reshape(n).copy()

    applied: list[dict] = []
    skipped: list[dict] = []
    todo = [e for e in report.get("grains", []) if e.get("decision") == "reassign"]

    # Re-derive each grain's pixels the same way check_map found them: the
    # report doesn't store pixel lists (they can be thousands of indices), so
    # re-segment per phase once and match grains by id.
    labels_by_phase: dict[int, np.ndarray] = {}
    for e in todo:
        pid = int(e["phase_id"])
        if pid not in labels_by_phase:
            pg = phases.get(pid)
            # pg missing → store None; the apply loop below skips the grain
            # exactly once (appending to `skipped` here too would duplicate it).
            labels_by_phase[pid] = None if pg is None else segment_supergroup_grains(
                q, pf, n_rows, n_cols, pid, pg,
                threshold_deg=float(report.get("params", {}).get("threshold_deg", 5.0)))

    for e in todo:
        pid = int(e["phase_id"])
        target = int(e.get("best_alt_phase", -1))
        labels = labels_by_phase.get(pid)
        if labels is None or target < 0:
            skipped.append({**e, "skip_reason": (
                "phase symmetry missing" if labels is None else "no target phase")})
            continue
        pix = np.flatnonzero(labels == int(e["grain_id"]))
        if pix.size == 0:
            skipped.append({**e, "skip_reason": "grain no longer found"})
            continue
        # Identity guard: grain ids are first-encounter ordinals of the
        # segmentation — after ANY orientation edit the same id can point at
        # a DIFFERENT physical grain. Verify count + centroid against what
        # the check saw; on mismatch skip rather than corrupt a good grain.
        rr, cc = np.divmod(pix, n_cols)
        cen = e.get("centroid") or [np.nan, np.nan]
        if (pix.size != int(e.get("pixels", -1))
                or abs(float(rr.mean()) - float(cen[0])) > 0.5
                or abs(float(cc.mean()) - float(cen[1])) > 0.5):
            skipped.append({**e, "skip_reason": "grain changed since check"})
            continue
        try:
            cq = np.asarray(hough_quats_fn(target, pix),
                            dtype=np.float64).reshape(-1, 4)
        except Exception:
            logger.warning("[phase-reassign] hough failed for grain %s → phase %s",
                           e.get("grain_id"), target, exc_info=True)
            skipped.append({**e, "skip_reason": "hough failed"})
            continue
        ok = np.isfinite(cq[:, 0])
        if not ok.any():
            skipped.append({**e, "skip_reason": "hough failed on every pixel"})
            continue
        # Nearest-successful fill (flat order) for failed pixels.
        n_filled = 0
        if not ok.all():
            good_pos = np.flatnonzero(ok)
            for bad in np.flatnonzero(~ok):
                nearest = good_pos[np.argmin(np.abs(good_pos - bad))]
                cq[bad] = cq[nearest]
                n_filled += 1
        pf[pix] = target
        q[pix] = cq
        applied.append({**e, "n_hough_filled": int(n_filled)})
        if progress is not None:
            try:
                progress(len(applied), len(todo))
            except Exception:
                pass

    return pf, q, applied, skipped
