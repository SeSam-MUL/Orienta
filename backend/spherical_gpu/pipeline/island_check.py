"""Render-verified check for ENCLOSED ISLANDS — the wrong pixels inside a grain.

The problem this exists for
---------------------------
In a heavily deformed region the patterns go diffuse, and the fine detail that
identifies a low-symmetry intermetallic is the first thing to go. What survives
is a few broad bands, which fit any cubic phase — so on those pixels the
highest-symmetry phase in the list wins. The result is salt-and-pepper matrix
pixels sitting inside an intermetallic particle. Reported on a 7050 sample
2026-09-05, and visible in that result's own numbers: Al carried the LOWEST
median CI (0.41) of the three phases while owning 84.63% of the map — it was
not winning where it fits, it was winning where nothing fits.

Why the existing per-grain check cannot see it
----------------------------------------------
`check_map` segments per phase and skips anything below `MIN_GRAIN_PX` (5), so
a one-pixel island is not merely unchecked — it is never counted. It was built
for whole grains stolen by a degenerate phase, which is a different failure.

Why this can be done cheaply, and without Hough
-----------------------------------------------
To ask "does phase B fit here?" you need an orientation for B at that pixel,
and the per-grain check gets one from Hough — expensive, and it leaks ~0.29 GiB
per call (measured 2026-09-04). But an island INSIDE a grain is surrounded by
pixels whose orientation is already known, so the candidate orientation is free:
the neighbour's. Pseudo-symmetric variants are likewise generated from the
pixel's OWN orientation. So this whole check runs on renders alone — no Hough,
no leak.

The neighbour orientation is taken per island pixel from its own nearest
enclosing neighbour, NOT as one average for the island: this runs on deformed
material, where orientation drifts across a grain, and an averaged orientation
would be wrong at exactly the pixels that matter. Quaternion averaging would
also have to deal with sign ambiguity for no benefit here.

Nothing is written. This module reports; the caller decides.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Islands larger than this are the per-grain check's business. Derived, not
#: chosen: an island is whatever is too small to be a grain, so the two
#: numbers must meet with nothing in between. The grain size itself is the
#: user's call (>= 9 px, 2026-09-07) and lives in phase_reassignment.
from .phase_reassignment import MIN_GRAIN_PX  # noqa: E402
MAX_ISLAND_PX = MIN_GRAIN_PX - 1

#: A candidate must beat the stored phase's render by this much before the
#: island is offered. Same spirit as MARGIN_CLEAR in the per-grain check: a
#: near-tie is not evidence, and these are single pixels.
MARGIN_CLEAR = 0.05

#: Cap on pseudo-symmetric variants scored per pixel — the cost is one render
#: each and the physically relevant ones come first.
MAX_VARIANTS = 8

_NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbours(flat: int, n_rows: int, n_cols: int):
    r, c = divmod(int(flat), n_cols)
    for dr, dc in _NEIGHBOUR_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n_rows and 0 <= cc < n_cols:
            yield rr * n_cols + cc


def find_enclosed_islands(phase_full, n_rows: int, n_cols: int, *,
                          max_island_px: int = MAX_ISLAND_PX):
    """Small same-phase islands that are completely surrounded by ONE other phase.

    Returns a list of
    ``{"pixels": (k,) int64, "stored_pid": int, "enclosing_pid": int,
       "neighbour_of": {pixel: neighbour_pixel}}``.

    "Enclosed" means every INDEXED neighbour outside the island belongs to the
    same other phase. Unindexed neighbours are ignored rather than disqualifying
    — a hole in the map should not protect a wrong pixel — but an island with no
    indexed neighbour at all is skipped, since there is then nothing to compare
    against.
    """
    pf = np.asarray(phase_full).reshape(-1)
    n = n_rows * n_cols
    if pf.size != n:
        raise ValueError(f"phase_full has {pf.size} entries, expected {n}")

    seen = np.zeros(n, dtype=bool)
    islands = []
    for start in range(n):
        pid = int(pf[start])
        if pid < 0 or seen[start]:
            continue
        # Flood fill this phase's connected component, but stop as soon as it
        # is too big to be an island — a full grain would otherwise be walked
        # in its entirety for nothing.
        comp = [start]
        seen[start] = True
        stack = [start]
        too_big = False
        while stack:
            cur = stack.pop()
            for nb in _neighbours(cur, n_rows, n_cols):
                if seen[nb] or int(pf[nb]) != pid:
                    continue
                seen[nb] = True
                comp.append(nb)
                stack.append(nb)
                if len(comp) > max_island_px:
                    too_big = True
                    break
            if too_big:
                break
        if too_big:
            # Mark the rest of this component seen so it is not re-walked from
            # another of its pixels.
            while stack:
                cur = stack.pop()
                for nb in _neighbours(cur, n_rows, n_cols):
                    if not seen[nb] and int(pf[nb]) == pid:
                        seen[nb] = True
                        stack.append(nb)
            continue

        inside = set(comp)
        ring_pid = None
        neighbour_of: dict[int, int] = {}
        enclosed = True
        for px in comp:
            for nb in _neighbours(px, n_rows, n_cols):
                if nb in inside:
                    continue
                nb_pid = int(pf[nb])
                if nb_pid < 0:
                    continue
                if ring_pid is None:
                    ring_pid = nb_pid
                elif nb_pid != ring_pid:
                    enclosed = False
                    break
                neighbour_of.setdefault(px, nb)
            if not enclosed:
                break
        if not enclosed or ring_pid is None or not neighbour_of:
            continue

        # Every island pixel must end up with a neighbour to borrow an
        # orientation from. A pixel whose own outside neighbours are ALL
        # unindexed has none of its own (found on a real 7050 map, where it
        # crashed the run) — give it the geometrically nearest ring pixel
        # instead. Still a local orientation, just one step further away.
        missing = [int(p) for p in comp if int(p) not in neighbour_of]
        if missing:
            ring = sorted(set(neighbour_of.values()))
            ring_rc = np.array([[r // n_cols, r % n_cols] for r in ring])
            for px in missing:
                pr, pc = divmod(px, n_cols)
                d = np.abs(ring_rc[:, 0] - pr) + np.abs(ring_rc[:, 1] - pc)
                neighbour_of[px] = int(ring[int(np.argmin(d))])

        islands.append({
            "pixels": np.asarray(sorted(comp), dtype=np.int64),
            "stored_pid": pid,
            "enclosing_pid": int(ring_pid),
            "neighbour_of": neighbour_of,
        })
    return islands


def _agg(scores) -> float:
    """Median over the finite scores; -inf when nothing scored (never wins)."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    return float(np.median(s)) if s.size else float("-inf")


def check_islands(full_q, phase_full, n_rows: int, n_cols: int,
                  phases: dict, score_fns: dict, *,
                  variants_fn=None,
                  fair_orientation_fn=None,
                  max_island_px: int = MAX_ISLAND_PX,
                  margin_clear: float = MARGIN_CLEAR,
                  max_variants: int = MAX_VARIANTS,
                  progress=None):
    """Score every enclosed island against its neighbours and its own variants.

    `score_fns` is ``{pid: fn(flat_idx (n,), quats (n,4)) -> (n,)}`` — the same
    render-NCC scorers the per-grain check uses. `variants_fn` is
    ``(q (4,), point_group) -> (m, 4)`` pseudo-symmetric variants of one
    orientation; omitted (or None) skips the variant stage entirely.

    Two stages, because renders are the whole cost:

      1. stored phase at its stored orientation  vs  the ENCLOSING phase at the
         neighbour's orientation.
      2. only if the stored phase is still ahead, its pseudo-symmetric variants.

    So the common case — a pixel that simply belongs to the surrounding phase —
    costs two renders, and the variant search is paid only where it can still
    change something.

    The fairness step (2026-09-07) -- why a stage-1 loss is not yet a finding
    ----------------------------------------------------------------------
    Stage 1 judges the stored phase at its STORED orientation. On a 1-8 px
    island that orientation is unreliable almost by definition: these are the
    pixels where indexing already went wrong, and a wrong orientation renders
    badly whatever the phase is. Measured on the real 7050 map: 8 of 24 stage-1
    findings were ARTEFACTS -- the stored phase, given a fair orientation, beat
    the proposal (MgCuAl2 at its stored orientation 0.207, at a fair one 0.420,
    the best of the three phases). Neither the stored score nor the margin
    separates artefacts from real findings (both overlap completely), and a
    random orientation search cannot find the ~2-degree-wide render peak (256
    samples still miss 5 of 8). So the stored phase is re-oriented by a
    DIRECTED method -- `fair_orientation_fn`, Hough in production -- and judged
    at the better of the two orientations. One batched call per stored phase,
    so the cost is a few calls per map, not one per island.

    Outcomes for a stage-1 loser, in order:
      * fair orientation available and the phase STILL loses -> "phase"
      * fair orientation available and the phase now wins/ties -> "keep"
        (entry carries "rescued_by_fair_orientation": True)
      * fair orientation unavailable (Hough rejected every pixel) -> "undecided"
        (the check cannot tell wrong phase from wrong orientation, and says so)
      * no `fair_orientation_fn` at all -> "suspect": a report-only verdict
        that `apply_island_findings` never applies, because a third of them
        would be wrong.

    Returns ``{"islands": [...], "n_islands", "n_phase_swap", "n_variant_flip",
    "n_unresolved", "n_undecided", "n_suspect", "n_rescued"}``. Nothing is
    modified.
    """
    q = np.asarray(full_q, dtype=np.float64).reshape(-1, 4)
    islands = find_enclosed_islands(phase_full, n_rows, n_cols,
                                    max_island_px=max_island_px)

    out = []
    pending = []          # stage-1 losers awaiting the fairness step
    n_variant_flip = 0
    for i, isl in enumerate(islands):
        pix = isl["pixels"]
        stored_pid = isl["stored_pid"]
        enc_pid = isl["enclosing_pid"]
        entry = {
            "pixels": [int(p) for p in pix],
            "n_px": int(pix.size),
            "stored_phase": int(stored_pid),
            "enclosing_phase": int(enc_pid),
            "centroid": [float(np.mean(pix // n_cols)), float(np.mean(pix % n_cols))],
        }
        if progress is not None:
            try:
                progress(i, len(islands))
            except Exception:
                pass

        if stored_pid not in score_fns or enc_pid not in score_fns:
            # Without a renderer for both sides there is no comparison to make.
            entry["decision"] = "no-scorer"
            out.append(entry)
            continue

        stored = _agg(score_fns[stored_pid](pix, q[pix]))
        entry["stored_score"] = None if not np.isfinite(stored) else round(stored, 4)

        # Stage 1 — the enclosing phase, at each island pixel's OWN neighbour's
        # orientation. Deformed grains drift; one average would be wrong exactly
        # where it matters.
        nb_q = np.stack([q[isl["neighbour_of"][int(p)]] for p in pix])
        enc = _agg(score_fns[enc_pid](pix, nb_q))
        entry["enclosing_score"] = None if not np.isfinite(enc) else round(enc, 4)

        if np.isfinite(enc) and enc - stored >= margin_clear:
            entry.update({"decision": "phase", "winner_phase": int(enc_pid),
                          "margin": round(float(enc - stored), 4),
                          # The orientations that WON, one per pixel, aligned
                          # with `pixels`. Carried in the report so applying a
                          # finding needs no re-derivation (and cannot silently
                          # drift from what was scored).
                          "new_quats": [[float(v) for v in row] for row in nb_q],
                          "_stored_pix": pix})
            # PROVISIONAL: a loss at the stored orientation is not yet
            # evidence against the phase -- the fairness step below decides.
            pending.append(entry)
            out.append(entry)
            continue

        # Stage 2 — pseudo-symmetry, only now that it can still change the call.
        best_var, best_var_score = None, float("-inf")
        if variants_fn is not None and np.isfinite(stored):
            pg = phases.get(int(stored_pid))
            for p in pix:
                try:
                    variants = np.asarray(variants_fn(q[int(p)], pg),
                                          dtype=np.float64).reshape(-1, 4)
                except Exception:  # noqa: BLE001 — variants are optional evidence
                    logger.debug("variant generation failed for pixel %s", p,
                                 exc_info=True)
                    continue
                for vq in variants[:max_variants]:
                    s = _agg(score_fns[stored_pid](np.asarray([p], dtype=np.int64),
                                                   vq.reshape(1, 4)))
                    if s > best_var_score:
                        best_var_score, best_var = s, vq
        if best_var is not None and np.isfinite(best_var_score):
            entry["best_variant_score"] = round(float(best_var_score), 4)
            if best_var_score - max(stored, enc) >= margin_clear:
                entry.update({
                    "decision": "variant",
                    "winner_phase": int(stored_pid),
                    "variant": [float(x) for x in best_var],
                    "margin": round(float(best_var_score - max(stored, enc)), 4),
                })
                n_variant_flip += 1
                out.append(entry)
                continue

        entry["decision"] = "keep"
        out.append(entry)

    # ---- fairness step: re-orient the stored phase before believing a loss ----
    n_rescued = 0
    if pending and fair_orientation_fn is None:
        for e in pending:
            e["decision"] = "suspect"
            e["fair_checked"] = False
    elif pending:
        by_phase: dict = {}
        for e in pending:
            by_phase.setdefault(int(e["stored_phase"]), []).append(e)
        for spid, entries in by_phase.items():
            flats = np.concatenate([e["_stored_pix"] for e in entries])
            reason = None
            try:
                fair_q = np.asarray(fair_orientation_fn(spid, flats),
                                    dtype=np.float64).reshape(-1, 4)
                if fair_q.shape[0] != flats.size:
                    raise ValueError("fair_orientation_fn returned %d rows for "
                                     "%d pixels" % (fair_q.shape[0], flats.size))
            except Exception as exc:  # noqa: BLE001 -- the run failed; say so
                logger.warning("[island-check] fair orientation for phase %s "
                               "failed: %s", spid, exc)
                fair_q = np.full((flats.size, 4), np.nan)
                reason = str(exc)
            pos = 0
            for e in entries:
                pix = e["_stored_pix"]
                fq = fair_q[pos:pos + pix.size]
                pos += pix.size
                good = np.all(np.isfinite(fq), axis=1)
                e["fair_checked"] = True
                if not good.any():
                    # Hough rejected every pixel (or could not run): the check
                    # cannot separate "wrong phase" from "wrong orientation".
                    e["decision"] = "undecided"
                    e["undecided_reason"] = reason or (
                        "no fair orientation for the stored phase on any pixel")
                    for k in ("winner_phase", "new_quats", "margin"):
                        e.pop(k, None)
                    continue
                fair = _agg(score_fns[spid](pix[good], fq[good]))
                e["stored_score_fair"] = None if not np.isfinite(fair) else round(fair, 4)
                stored0 = e.get("stored_score")
                stored_best = max(
                    float(stored0) if stored0 is not None else float("-inf"), fair)
                enc = float(e["enclosing_score"])
                if enc - stored_best >= margin_clear:
                    e["margin"] = round(float(enc - stored_best), 4)   # final
                else:
                    e["decision"] = "keep"
                    e["rescued_by_fair_orientation"] = True
                    for k in ("winner_phase", "new_quats", "margin"):
                        e.pop(k, None)
                    n_rescued += 1
    for e in out:
        e.pop("_stored_pix", None)

    return {
        "islands": out,
        "n_islands": len(out),
        "n_phase_swap": sum(1 for e in out if e["decision"] == "phase"),
        "n_variant_flip": n_variant_flip,
        # Islands that could not be judged at all -- kept separate from "keep",
        # because "nothing was wrong" and "nothing could be compared" are
        # different answers and must never share a number.
        "n_unresolved": sum(1 for e in out if e["decision"] == "no-scorer"),
        # Stage-1 losers the fairness step could not settle (no fair
        # orientation) -- reported, never applied.
        "n_undecided": sum(1 for e in out if e["decision"] == "undecided"),
        # Stage-1 losers with NO fairness step available at all.
        "n_suspect": sum(1 for e in out if e["decision"] == "suspect"),
        # Stage-1 losers that turned out to be a bad ORIENTATION, not a bad
        # phase -- the artefacts this step exists to stop.
        "n_rescued": n_rescued,
    }


def apply_island_findings(full_q, phase_full, n_rows: int, n_cols: int, report):
    """Write the accepted island findings into copies of the phase/orientation
    arrays. Pure: returns ``(new_phase_full, new_full_q, applied)``.

    Only ``decision`` of ``"phase"`` or ``"variant"`` is applied, i.e. exactly
    what `check_islands` was willing to stand behind. Everything needed is
    already in the report — the winning phase id and the winning orientations —
    so this needs no scorers, no renders and no Hough, and it cannot disagree
    with what the check reported.
    """
    n = n_rows * n_cols
    q = np.asarray(full_q, dtype=np.float64).reshape(n, 4).copy()
    pf = np.asarray(phase_full).reshape(n).copy()
    applied = []
    for e in report.get("islands", []):
        dec = e.get("decision")
        if dec not in ("phase", "variant"):
            continue
        pix = np.asarray(e["pixels"], dtype=np.int64)
        if pix.size == 0 or pix.min() < 0 or pix.max() >= n:
            raise ValueError(f"island pixel out of range: {e.get('pixels')}")
        if dec == "phase":
            quats = np.asarray(e["new_quats"], dtype=np.float64).reshape(-1, 4)
            if quats.shape[0] != pix.size:
                raise ValueError(
                    f"island has {pix.size} pixels but {quats.shape[0]} quats")
        else:
            # One variant of the island's own orientation, for every pixel.
            quats = np.repeat(
                np.asarray(e["variant"], dtype=np.float64).reshape(1, 4),
                pix.size, axis=0)
        pf[pix] = int(e["winner_phase"])
        q[pix] = quats
        applied.append({"pixels": [int(p) for p in pix],
                        "winner_phase": int(e["winner_phase"]),
                        "decision": dec})
    return pf, q, applied
