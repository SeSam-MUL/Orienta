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
# What counts as a grain. This is a materials-science judgement, not a
# tuning knob: the user set it at 9 px (2026-09-07). Everything smaller is
# an island and belongs to island_check, whose MAX_ISLAND_PX is derived
# from this so the two can never leave a gap.
MIN_GRAIN_PX = 9
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
    # Suspect grains, parked until every candidate's Hough orientations have
    # been fetched in one batch per candidate phase (see below).
    suspects: list[dict] = []
    n_grains = n_checked = n_suspect = n_reassign = 0
    n_undecided = n_rescued = 0

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

            # Suspect. The candidate scoring does NOT happen here: every
            # candidate's Hough orientations for EVERY suspect grain are
            # fetched together further down, in one call per candidate phase.
            #
            # Why: each `hough_indexing` call leaks ~0.24 GiB of Windows commit
            # that is never returned (measured 2026-09-04: linear over 24 calls,
            # RSS rising with it, no plateau — it is a real leak in
            # kikuchipy/PyEBSDIndex, not our indexer, since one reused indexer
            # leaks just the same). Per grain per candidate that came to 172
            # calls and ~40 GiB on one map, which is what kept taking the
            # machine to its commit limit. Batching makes it one call per
            # candidate phase.
            n_suspect += 1
            suspects.append({"entry": entry, "pid": pid, "pix": pix,
                             "sample": sample, "stored": stored})
            entries.append(entry)

    # --- Candidate scoring, batched per candidate phase --------------------
    #
    # Each suspect grain keeps its OWN slice of the batch, so the scores are
    # per grain exactly as before: `hough_indexing` treats every pattern
    # independently, so concatenating the samples cannot change a result.
    for cand in pids:
        if cand not in score_fns:
            continue
        todo = [sp for sp in suspects if sp["pid"] != cand]
        if not todo:
            continue
        flats = np.concatenate([sp["sample"] for sp in todo])
        try:
            cq_all = np.asarray(hough_quats_fn(cand, flats),
                                dtype=np.float64).reshape(-1, 4)
        except Exception as exc:  # noqa: BLE001 — one candidate, not the run
            logger.warning("[phase-check] hough for candidate %s failed",
                           cand, exc_info=True)
            reason = str(exc) or "hough failed"
            for sp in todo:
                sp["entry"].setdefault("unevaluated", {})[int(cand)] = reason
            continue

        at = 0
        for sp in todo:
            sample = sp["sample"]
            cq = cq_all[at:at + sample.size]
            at += sample.size
            ok = np.isfinite(cq[:, 0])
            # A candidate scored on a tiny Hough-lucky subset would be
            # compared against the stored phase's FULL-sample median —
            # asymmetric and biased. Require a minimum successful count.
            if int(ok.sum()) < min(3, sample.size):
                continue
            sc = _agg(score_fns[cand](sample[ok], cq[ok]))
            if np.isfinite(sc):
                sp.setdefault("alt_scores", {})[int(cand)] = round(float(sc), 4)

    # --- Fairness: the STORED phase gets a Hough orientation too -----------
    #
    # A suspect grain is "suspect" because its stored phase rendered badly AT
    # ITS STORED ORIENTATION. For a phase whose spherical orientation is
    # unreliable (z_rot == 2: mmm, -43m, m-3 ...) that is not evidence against
    # the phase at all -- it is the indexer's orientation being wrong. On the
    # real 7050 the whole MgCuAl2 particle (one grain, ~250 px) was flipped to
    # Al this way: MgCuAl2 rendered 0.207 at its stored orientation, but 0.420
    # at a fair one, the best of the three phases; Al's Hough-oriented 0.35
    # only won because the comparison was rigged (2026-09-07). Same mechanism
    # the island check already guards against; this is the grain-sized twin.
    #
    # So every stored phase is Hough-oriented on its suspect grains' samples,
    # ONE batched call per stored phase (same leak arithmetic as above), and
    # the grain is judged at the better of stored@stored and stored@hough.
    by_stored: dict = {}
    for sp in suspects:
        by_stored.setdefault(int(sp["pid"]), []).append(sp)
    for spid, group in by_stored.items():
        flats = np.concatenate([sp["sample"] for sp in group])
        reason = None
        try:
            fq_all = np.asarray(hough_quats_fn(spid, flats),
                                dtype=np.float64).reshape(-1, 4)
            if fq_all.shape[0] != flats.size:
                raise ValueError("hough returned %d rows for %d pixels"
                                 % (fq_all.shape[0], flats.size))
        except Exception as exc:  # noqa: BLE001 -- fall through to "no fair"
            logger.warning("[phase-check] hough for STORED phase %s failed",
                           spid, exc_info=True)
            fq_all = np.full((flats.size, 4), np.nan)
            reason = str(exc) or "hough failed"
        at = 0
        for sp in group:
            sample = sp["sample"]
            fq = fq_all[at:at + sample.size]
            at += sample.size
            ok = np.isfinite(fq[:, 0])
            if int(ok.sum()) >= min(3, sample.size):
                fair = _agg(score_fns[spid](sample[ok], fq[ok]))
            else:
                fair = float("nan")
            sp["stored_fair"] = fair
            sp["stored_fair_reason"] = reason
            if np.isfinite(fair):
                sp["entry"]["stored_score_fair"] = round(float(fair), 4)

    # --- Decision per suspect grain ---------------------------------------
    for sp in suspects:
        entry = sp["entry"]
        alt_scores = sp.get("alt_scores") or {}
        stored = sp["stored"]
        fair = sp.get("stored_fair", float("nan"))
        fair_ok = np.isfinite(fair)
        stored_best = np.nanmax([stored if np.isfinite(stored) else -1.0,
                                 fair if fair_ok else -np.inf])
        entry["fair_checked"] = bool(fair_ok)
        if alt_scores:
            best_alt = max(alt_scores, key=lambda k_: alt_scores[k_])
            best_alt_score = alt_scores[best_alt]
            margin_raw = (stored if np.isfinite(stored) else -1.0) - best_alt_score
            margin = float(stored_best) - best_alt_score
            entry.update({
                "alt_scores": alt_scores,
                "best_alt_phase": int(best_alt),
                "best_alt_score": best_alt_score,
                "margin": round(float(margin), 4),
            })
            if margin <= -margin_clear and not fair_ok:
                # Would reassign, but the stored phase never got a fair
                # orientation: wrong phase and wrong orientation are
                # indistinguishable here. Say so; never guess. (Same rule as
                # the island check.)
                entry["decision"] = "undecided"
                entry["undecided_reason"] = sp.get("stored_fair_reason") or (
                    "no Hough orientation for the stored phase on this grain")
                n_undecided += 1
            elif margin <= -margin_clear:
                margin_full[sp["pix"]] = margin
                entry["decision"] = "reassign"
                n_reassign += 1
            else:
                margin_full[sp["pix"]] = margin
                entry["decision"] = "keep"
                if margin_raw <= -margin_clear:
                    # Lost at the stored orientation, wins fairly oriented:
                    # the artefact this step exists to stop.
                    entry["rescued_by_fair_orientation"] = True
                    n_rescued += 1
        else:
            entry["decision"] = "keep"  # nothing comparable -> conservative

    # Grains where at least one candidate could not be evaluated at all. Kept
    # apart from n_reassign on purpose: "nothing to reassign" and "nothing could
    # be compared" look identical in a count and mean opposite things.
    n_unevaluated = sum(1 for e in entries if e.get("unevaluated"))
    unevaluated_reasons = sorted({
        str(r) for e in entries for r in (e.get("unevaluated") or {}).values()
    })

    report = {
        "grains": entries,
        "n_grains": n_grains,
        "n_checked": n_checked,
        "n_suspect": n_suspect,
        "n_unevaluated": n_unevaluated,
        "unevaluated_reasons": unevaluated_reasons[:5],
        "n_reassign": n_reassign,
        # Suspect grains that lost at their stored orientation but got no fair
        # (Hough) orientation for the stored phase -- reported, never applied.
        "n_undecided": n_undecided,
        # Suspect grains that lost at their stored orientation and WON once
        # fairly oriented: a bad orientation, not a bad phase.
        "n_rescued": n_rescued,
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
                       score_fns=None, score_floor: float = SCORE_FLOOR,
                       margin_clear: float = MARGIN_CLEAR,
                       progress=None):
    """Stage B — apply the ``reassign`` decisions from :func:`check_map`.

    For every grain with ``decision == "reassign"``: set its pixels to the
    winning phase and give each pixel its own Hough orientation from that
    phase (per-pixel, so intra-grain texture is real, not a rigid copy).

    The grain decides the candidate; the PIXEL decides whether it is written
    (2026-09-07). With ``score_fns`` given, every pixel of a reassigned grain
    is verified on its own: the winner at its Hough orientation must render
    at least ``score_floor`` -- what a correct assignment renders -- AND beat
    the stored phase on that pixel by ``margin_clear``. Pixels that fail stay
    as they are and are counted in ``n_px_refused``; pixels without a Hough
    orientation of their own are refused too (no nearest-fill in this mode),
    because an orientation copied from a neighbour was never verified.

    Why: a reassign on a deformed 7050 map moved 531 px in one press. Per
    pixel, the winner beat the stored phase on 76 %% of the Al->Al7FeCu2
    pixels -- the 301 that passed had band contrast 110 (particle), the 176
    that failed had 83 (matrix) -- and on only 16 %% of the 44 Al->MgCuAl2
    pixels, where the written orientation rendered at 0.09, worse than the
    phase it replaced. A 16-sample grain median cannot tell those apart; the
    per-pixel evidence can.

    Without ``score_fns`` (older callers, unit tests) the grain is written
    whole and pixels where Hough fails inherit the quat of the nearest
    (flat-order) successful pixel; if Hough fails for the WHOLE grain the
    grain is left untouched (fail-safe = stored phase survives).

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
        except Exception as exc:  # noqa: BLE001 — one grain must not stop the rest
            logger.warning("[phase-reassign] hough failed for grain %s → phase %s",
                           e.get("grain_id"), target, exc_info=True)
            # Carry the reason. "hough failed" on its own sent the user looking
            # at their data for a fault that was in the machine's memory.
            skipped.append({**e, "skip_reason": str(exc) or "hough failed"})
            continue
        ok = np.isfinite(cq[:, 0])
        if not ok.any():
            # Hough RAN and rejected every pixel at its own quality gate
            # (fit > 3 deg or nmatch < 4) -- a statement about how well this
            # phase's bands fit these patterns, NOT a crash and NOT a memory
            # problem. Those arrive as an exception above and carry their own
            # text. Saying "hough failed" for both sent a user hunting an
            # OpenCL error that was not happening in that session
            # (2026-09-07), so the two now read differently.
            skipped.append({**e, "skip_reason": (
                "Hough ran but could not fit this phase's bands to any pixel "
                "of the grain (every pixel missed its fit/nmatch gate). This "
                "is about how the phase matches these patterns, not a crash "
                "or a memory problem.")})
            continue
        gated = (score_fns is not None and target in score_fns and pid in score_fns)
        if gated:
            # Per-pixel evidence gate -- see the docstring. Only pixels with
            # their OWN Hough orientation are even considered.
            idx_ok = pix[ok]
            win_r = np.asarray(score_fns[target](idx_ok, cq[ok]), dtype=np.float64).reshape(-1)
            st_r = np.asarray(score_fns[pid](idx_ok, q[idx_ok]), dtype=np.float64).reshape(-1)
            passed = (np.isfinite(win_r) & (win_r >= float(score_floor))
                      & np.isfinite(st_r) & ((win_r - st_r) >= float(margin_clear)))
            write = idx_ok[passed]
            n_refused = int(pix.size - write.size)
            if write.size == 0:
                skipped.append({**e, "skip_reason": (
                    "no pixel of the grain carried enough evidence for the new "
                    "phase on its own (winner render below %.2f, or not clearly "
                    "better than the stored phase)" % float(score_floor))})
                continue
            pf[write] = target
            q[write] = cq[ok][passed]
            applied.append({**e, "n_hough_filled": 0,
                            "n_px_applied": int(write.size),
                            "n_px_refused": n_refused,
                            "n_px_no_hough": int((~ok).sum())})
        else:
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
            applied.append({**e, "n_hough_filled": int(n_filled),
                            "n_px_applied": int(pix.size), "n_px_refused": 0})
        if progress is not None:
            try:
                progress(len(applied), len(todo))
            except Exception:
                pass

    return pf, q, applied, skipped


def rigid_grain_quats(q_stored_grain, q_click, q_seed):
    """Carry one known orientation across a grain as a RIGID correction.

    ``q_new(i) = (q_seed · q_click⁻¹) · q_stored(i)``

    The fallback for a manual phase assignment when Hough cannot supply
    per-pixel orientations for the target phase. The user has already been
    shown one good orientation for that phase AT the clicked pixel (the
    Compare-phases panel re-indexes the pixel per phase and prints its
    render-NCC), so the information the assignment needs is on screen; asking
    Hough for it again is what fails.

    Why a rigid correction and not one orientation copied onto every pixel:
    inside one grain the stored orientations carry a real lattice rotation
    field — sub-grain rotation, the deformation the sample actually has. A
    constant orientation would flatten it and hand back a suspiciously perfect
    grain. The relative rotation between two pixels is a rotation in SAMPLE
    space, so it stays meaningful when the phase label changes; only the
    absolute anchor comes from the new phase.

    This is the same correction `pseudosym.grain_snap_floodfill` applies for a
    same-phase grain flip (``C = q_target·q_click⁻¹``), and it is deliberately
    NOT a substitute for Hough: per-pixel Hough orientations are independent
    measurements, this one propagates a single measurement. It is used only
    when Hough produced nothing at all, and the caller reports which of the two
    it used so the result is never silently the weaker one.
    """
    from backend.spherical_gpu.pseudosym import _qconj, _qmul

    q = np.asarray(q_stored_grain, dtype=np.float64).reshape(-1, 4)
    click = np.asarray(q_click, dtype=np.float64).reshape(4)
    seed = np.asarray(q_seed, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(click)) or not np.all(np.isfinite(seed)):
        raise ValueError("click and seed orientations must both be finite")
    n_click = np.linalg.norm(click)
    n_seed = np.linalg.norm(seed)
    if n_click < 1e-12 or n_seed < 1e-12:
        raise ValueError("click and seed orientations must be non-zero")
    corr = _qmul(seed / n_seed, _qconj(click / n_click))
    out = _qmul(np.broadcast_to(corr, q.shape), q)
    return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)
