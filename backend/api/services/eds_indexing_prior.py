"""EDS chemistry prior for multi-phase indexing.

Turns per-pixel EDS composition + per-phase strengths into a (P, N) multiplier
applied to each phase's pattern score before the winner is chosen. Reuses the
Phase-Test chemistry so 'consistency' matches the rest of the app. Off (all
strengths 0) or no EDS => the caller passes None and indexing is bit-identical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from backend.api.services.crystal_hint_phase_fit import (
    chemistry_fit, phase_nominal_at_pct,
)

logger = logging.getLogger(__name__)

#: Causes already reported this session. A systematic reason is a property of
#: the FILE, not of a pixel, and the prior is asked for a whole map — without
#: this, one line per cause would become one line per run and then be ignored.
#:
#: Keyed on the code AND what it is about: two different unpriceable windows
#: are two different things to tell the user, and a key of just "excluded_window"
#: would report the first and swallow the second.
_WARNED: set = set()


def _warn_once(code: str, message: str, subject=()) -> None:
    key = (code, tuple(sorted(str(s) for s in subject)))
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning("%s", message)


def reset_prior_warning_dedupe() -> None:
    """Forget what has been reported — for a file change, and for tests."""
    _WARNED.clear()


#: Two sources of a phase's composition are "the same answer" below this.
#: Above it the difference is worth a line in the log, because it means the
#: name on the file and the structure behind it disagree about the phase.
_COMPOSITION_NOTICE_AT_PCT = 2.0

#: Above this the two sources are not two descriptions of one compound, and
#: the structure is REFUSED rather than adopted.
#:
#: Calibrated on two cases, measured by comparing each CIF with its file name
#: over the 30-master library:
#: alpha-Al(Fe,Mn)Si at 4.8 at% (the CIF refines the mixed site the file name
#: idealises -- adopt) and `MgCu2 (sd_1816951)` at 46.7 (stored as Mg 80 /
#: Cu 20 for a compound that is Cu 66.7 / Mg 33.3 -- refuse). Adopting the
#: second moved the prior's weight on a genuine MgCu2 pixel from 0.922 to 0.499
#: and RAISED it on a Mg-rich Cu-poor one: a confident wrong answer for any
#: 7xxx workflow, announced only by a log line.
#:
#: 15.0 is the geometric middle of that empty gap (sqrt(4.8 x 46.7) = 14.97),
#: i.e. 3.1x above the largest disagreement that IS a refinement and 3.1x below
#: the smallest that is not. It also sits below the smallest stoichiometric step
#: that changes a binary's identity (AB -> A2B is 16.7 at%), which is the
#: physical reading of the same number.
#:
#: THE CALIBRATING CASE NO LONGER REACHES THIS LINE, and the script now prints
#: 28 agree / 1 element-set disagreement / 1 no-CIF-resolved / 1 split above
#: 2 at%. Not a change of fact: `_entry_from_cif` refuses sd_1816951 outright
#: since the multi-block guard, so `composition_from_structure` returns None
#: for it and the prior keeps the file name through the unreadable-structure
#: path -- the same right answer, one step earlier, with its own warning. This
#: threshold is what catches the next entry where the two sources disagree that
#: far and pymatgen has no complaint.
_COMPOSITION_REFUSE_AT_PCT = 15.0


def composition_from_structure(path: str) -> Optional[dict[str, float]]:
    """At.% from the refined atom sites of the CIF ``path`` was built from.

    WHY THE FILE NAME IS NOT THE ANSWER. The prior read a phase's composition
    off the master's file name. For the alpha approximant that name is
    ``Mn0.5Fe0.5Al5Si0.68`` — the Pauling File's *prototype label*, not the
    refinement. The CIF behind it (``sd_0302719.cif``) writes the shared site
    as ``0.812Fe + 0.188Mn`` and its own header calls the compound
    ``(FeMn)3Si2Al15, Fe/(Mn+Fe) = 0.90``: Fe and Mn substitute for each other
    on one site, so the split is a property of the ALLOY, not of the structure,
    and the 1:1 in the name is an idealisation nobody measured. SampleB reads
    Fe:Mn = 3.79 (Aztec, independently: 3.78) where the name demands 1.0, and
    that one number cost the prior its whole alpha region — measured per pixel,
    9.95 % of it named correctly.

    The structure file is the better source because it is what was actually
    SIMULATED: ``sd_0302719.xtal``, EMsoft's input for the master pattern,
    carries the same 0.812/0.188 occupancies. The master IS that composition.

    Returns None when no CIF resolves or it cannot be read — the caller keeps
    the formula. The composition is produced by the same
    ``cif_phase_library`` reader the EDS phase map uses.

    THAT IS NOT THE SAME AS "one phase, one number everywhere", and saying so
    would be false. Two places disagree today and both are known:
    ``beta-AlFeSi`` — where :func:`_reconcile` keeps the file name because the
    CIF has no silicon and the map uses the CIF's composition — and
    ``sd_1816951`` (MgCu2), where the map reads a spreadsheet row written from
    a parse this reader now refuses, and the prior refuses it too. In both the
    prior is the one that does not use the broken value, and in both the
    disagreement is reported rather than resolved.

    THE LINK IS BEST-EFFORT. ``phase_metadata.find_linked_cif`` falls back to
    searching the whole CIF folder for a matching formula, so a master with no
    same-named CIF can be linked to a different file that happens to agree on a
    formula string. That is why :func:`_reconcile` never takes this answer on
    trust: a mis-linked CIF almost always disagrees with the master's own name
    in the element set or by a large split, and both are refused out loud.

    Never raises — the prior degrades to the file name rather than costing the
    run — but it never degrades QUIETLY either. On a machine without the CIF
    library every phase falls back, alpha's per-pixel rate returns to 9.95 %,
    and the only way to know is this warning.
    """
    try:
        from backend.api.services.cif_phase_library import _cached_entry_from_cif
        from path_utils import DATABASE_SUBFOLDERS, get_local_database_path
        from phase_metadata import get_phase_metadata
        cif_dir = get_local_database_path() / DATABASE_SUBFOLDERS["cif_library"]
        meta = get_phase_metadata(Path(path), cif_library_dir=cif_dir)
        cif_path = getattr(meta, "cif_path", "") or ""
        if not cif_path:
            logger.warning(
                "Phase %s: no CIF resolved from %s, so the EDS prior keeps the "
                "composition in the file name. For a phase with a mixed site "
                "that name can be an idealisation -- set the expected "
                "composition by hand if the chemistry looks wrong.",
                Path(path).name, cif_dir)
            return None
        entry, reason = _cached_entry_from_cif(Path(cif_path))
        if entry is None or not entry.composition:
            logger.warning(
                "Phase %s: %s could not be read into a composition (%s), so "
                "the EDS prior keeps the composition in the file name.",
                Path(path).name, Path(cif_path).name,
                reason or "no reason given")
            return None
        return dict(entry.composition)
    except Exception as exc:                      # never break a run over this
        logger.warning(
            "Phase %s: could not reach its structure (%s: %s), so the EDS "
            "prior keeps the composition in the file name.",
            Path(path).name, type(exc).__name__, exc)
        return None


def _reconcile(formula_at: dict, structure_at: dict, path: str) -> dict:
    """Pick between the file name's formula and the structure's refinement.

    The structure wins ONLY when the two agree on which ELEMENTS the phase is
    made of. A disagreement there is not a refinement, it is a broken library
    entry, and the prior must not resolve it silently in either direction:
    ``beta-AlFeSi.cif`` reduces to Al11Fe2 — no silicon at all — while the
    master it produced is named ``Al4FeSi (beta-AlFeSi)``, and the reference
    region of that phase measures 17.1 at% Si. Adopting the CIF there would
    delete an element the phase is named after; adopting the name hides that
    the simulated structure never had it. So: keep the name, say both out loud.

    A disagreement about the SPLIT above :data:`_COMPOSITION_REFUSE_AT_PCT` is
    refused for the same reason: at that size the two are not two descriptions
    of one compound. ``MgCu2 (sd_1816951)`` is stored as Mg 80 / Cu 20 for a
    Laves phase that is Cu 66.7 / Mg 33.3, and adopting it would tell the prior
    that a genuine MgCu2 pixel is the wrong phase.

    Scanned over the whole 30-master library
    (CIF-versus-file-name comparison, re-run 2026-09-13): 28
    element sets agree, 1 disagrees (beta), 1 resolves no readable CIF
    (``MgCu2``, refused by the multi-block guard before it reaches here) and 1
    of the 28 differs in the SPLIT by more than 2 at% — alpha by 4.8, the
    mixed site, adopted. Everything above 2 at% is logged either way.
    """
    if not structure_at:
        return formula_at
    if not formula_at:
        return structure_at
    if set(formula_at) != set(structure_at):
        logger.warning(
            "Phase %s: its file name says %s but the structure it was built "
            "from is %s. The element sets disagree, so this is a mislabelled "
            "library entry, not a refinement - the EDS prior keeps the name's "
            "composition and does not use the structure.",
            Path(path).name,
            _fmt(formula_at), _fmt(structure_at))
        return formula_at
    worst = max(abs(formula_at[el] - structure_at[el]) for el in formula_at)
    if worst >= _COMPOSITION_REFUSE_AT_PCT:
        logger.warning(
            "Phase %s: its file name says %s and the structure it was built "
            "from says %s -- %.1f at%% apart. That is too far for two "
            "descriptions of one compound, so one of the two library files is "
            "wrong and the EDS prior will not guess: it keeps the name's "
            "composition. Check the CIF and the .xtal for this phase.",
            Path(path).name, _fmt(formula_at), _fmt(structure_at), worst)
        return formula_at
    if worst >= _COMPOSITION_NOTICE_AT_PCT:
        logger.warning(
            "Phase %s: the EDS prior uses the composition of the structure it "
            "was built from, %s, not the one in its file name, %s (they differ "
            "by up to %.1f at%%). The structure is what was simulated.",
            Path(path).name, _fmt(structure_at), _fmt(formula_at), worst)
    return structure_at


def _fmt(at: dict) -> str:
    return " ".join(f"{el}={v:.1f}" for el, v in sorted(at.items()))


def expected_at_pct_for_phases(
    formulas: list[str], overrides: Optional[list[dict]] = None,
    paths: Optional[list[str]] = None,
) -> list[dict]:
    """Expected At.% (0-100) per phase, in order of authority.

    1. ``overrides[i]`` — what the user typed for this phase, per element.
    2. the refined composition of the structure ``paths[i]`` was built from,
       when it names the same elements as the formula (:func:`_reconcile`).
    3. the formula, i.e. whatever the file name says.

    ``paths`` is optional so callers that only ever had formulas keep working;
    without it the behaviour is exactly (1) then (3), as before. An
    unparseable/empty formula with no override and no structure yields {}
    (neutral).
    """
    out: list[dict] = []
    for i, f in enumerate(formulas):
        exp = dict(phase_nominal_at_pct(f or ""))
        if paths and i < len(paths) and paths[i]:
            structure = composition_from_structure(str(paths[i]))
            if structure:
                exp = _reconcile(exp, structure, str(paths[i]))
        if overrides and i < len(overrides) and overrides[i]:
            exp.update({str(k): float(v) for k, v in overrides[i].items()})
        out.append(exp)
    return out


def chemistry_weight_matrix(
    measured: list[Optional[dict]],
    expected: list[dict],
    strengths: list[float],
) -> np.ndarray:
    """(P, N) multiplier, weight_p(i) = (1 - s_p) + s_p * chemistry_fit(measured[i],
    expected[p]). Rows with s_p <= 0 or empty expected are all-ones (no-op). Pixels
    with empty/None measured get 1.0 via chemistry_fit's fail-soft. Reuses the scalar
    chemistry_fit per (pixel, phase) for parity; only phases with s_p > 0 pay the cost."""
    P = len(expected)
    N = len(measured)
    W = np.ones((P, N), dtype=np.float64)
    for p in range(P):
        s = float(strengths[p]) if p < len(strengths) else 0.0
        exp = expected[p]
        if s <= 0.0 or not exp:
            continue
        for i in range(N):
            meas = measured[i]
            if not meas:
                continue  # fail-soft neutral (1.0)
            fit = chemistry_fit(meas, exp)
            W[p, i] = (1.0 - s) + s * fit
    return W


#: The EDS numbers the prior is made of -- and with it the particle rescue
#: (``routes.indexing._apply_particle_rescue``) and the indexing pre-flight
#: that predicts the prior (``eds_preflight.run_preflight``), so the three can
#: never read different sources -- and every other consumer that ranks with
#: the same ``chemistry_fit`` on measured at%: the single-pixel phase test and
#: the crystal hint (``eds_pixel_chemistry.chemistry_source``, measured the
#: same way: 89.37 -> 74.55 % on the alpha region of the reference scan) and grain phase
#: assignment (``routes.indexing._grain_chemistry_maps``).
#: ``"window"``: AZtec's raw window maps with the
#: window calibration, whatever the open file serves; ``None``: the open
#: file's own source (net line intensities when it has a spectrum cube).
#:
#: KEPT ON THE WINDOW PATH, measured window vs spectrum
#: on the AZtec reference scan (a scan not included in the repository). Used as a per-pixel
#: classifier against the pattern-only 4-phase map of the AZtec reference scan,
#: the prior's chemistry (``chemistry_fit``) picks alpha-Al(Fe,Mn)Si on
#: 89.37 % of its pixels from the window numbers and on 74.55 % from the
#: spectrum numbers (all four phases 88.90 -> 87.03 %), and Fe >= 11.6 at%
#: pixels of that scan keep an Fe phase on 100 -> 98.46 %. Not noise (5x5
#: smoothing leaves 91.99 vs 77.36 %): alpha and beta score within 0.04 of each
#: other, and the spectrum numbers -- closer to AZtec -- move the alpha
#: pixels' Si and Fe toward beta and drop the window's spurious Zn, which had
#: been charging beta more than alpha. The scorer's floors were tuned on the
#: window numbers. ``None`` is only valid once they are re-measured on the
#: spectrum at% (veto-constant and joint sweeps) and the alpha rate on the
#: reference scan is at least 89.37 %.
PRIOR_EDS_SOURCE: Optional[str] = "window"


def _build_at_pct_maps_for_loaded_file():
    # Imported lazily so the pure functions above have no FastAPI/route dependency
    # and the test can monkeypatch this symbol.
    from backend.api.routes.eds import _build_at_pct_maps_for_loaded_file as _f
    return _f(source=PRIOR_EDS_SOURCE)


@dataclass(frozen=True)
class RunChemistry:
    """The EDS chemistry of the dataset a run started on, read ONCE when the run started.

    A run lasts minutes to hours. The chemistry that steers it must be the chemistry of the dataset
    and measurement it runs on; every reader of "the loaded file" follows the viewer, and the viewer
    can be switched to another measurement (or another file) while the run is going. A prior built
    after the indexer, or a rescue after the run, would then pair the run's patterns with another
    measurement's chemistry; equal grids pass the grid guard.

    ``at_maps`` is None when the chemistry could not be read at the start (no EDS, a refused window,
    ...): the prior then behaves as it did when the read failed late, no weights, and the active state
    is NOT asked again. ``nav_shape`` is the navigation grid (rows, cols) of the run's own signal, the
    shape the EDS grid is checked against."""

    at_maps: Optional[dict]
    n_rows: int = 0
    n_cols: int = 0
    nav_shape: Optional[tuple] = None
    #: why a snapshot that was configured was dropped (a read that failed, a viewer that was switched
    #: during it); the run says it in its progress. None for chemistry that was never there.
    dropped: Optional[str] = None


def read_run_chemistry(nav_shape: Optional[tuple]) -> RunChemistry:
    """Read the chemistry of the active dataset now, as the snapshot a run holds. Never raises."""
    try:
        at_maps, n_rows, n_cols, _ = _build_at_pct_maps_for_loaded_file()
    except Exception as exc:  # noqa: BLE001 - the run goes on without the prior, but says why
        logger.warning("EDS chemistry could not be read for this run (%s: %s): the prior is off",
                       type(exc).__name__, exc, exc_info=True)
        return RunChemistry(None, nav_shape=nav_shape,
                            dropped=f"the chemistry could not be read ({type(exc).__name__}: {exc})")
    if not at_maps:
        return RunChemistry(None, nav_shape=nav_shape)
    return RunChemistry(at_maps, int(n_rows), int(n_cols), nav_shape)


class EdsGridMismatch(ValueError):
    """EDS map grid differs from the EBSD navigation grid.

    Raised instead of silently slicing: with different shapes every pattern
    would be paired with a DIFFERENT pixel's chemistry, so the prior would be
    confidently wrong rather than merely unhelpful. Fail loud (2026-08-05).
    """


def measured_atpct_per_pixel(
    selection_mask: Optional[np.ndarray] = None,
    expected_shape: Optional[tuple] = None,
    chemistry: Optional[RunChemistry] = None,
) -> Optional[list]:
    """Per RESULT-pixel At.% dicts (0-100), row-major over the nav grid. With
    selection_mask, only selected pixels in row-major (ascending-flat) order — the
    same order the pattern stack / result use. Returns None if no EDS is available.

    ``expected_shape`` is the (n_rows, n_cols) of the EBSD navigation grid. When
    given — or derivable from ``selection_mask`` — it is compared against the EDS
    grid and a mismatch raises :class:`EdsGridMismatch`.

    ``chemistry`` is the snapshot a run took when it started (:func:`read_run_chemistry`): the maps
    come from it and the loaded file is not asked. Without it, the loaded file is read now (the
    request-time callers).
    """
    if chemistry is not None:
        at_maps, n_rows, n_cols = chemistry.at_maps, chemistry.n_rows, chemistry.n_cols
    else:
        from backend.api.services.eds_spectrum_service import SPECTRUM_FIT_ERRORS
        try:
            at_maps, n_rows, n_cols, _ = _build_at_pct_maps_for_loaded_file()
        except SPECTRUM_FIT_ERRORS:
            # Only reachable with PRIOR_EDS_SOURCE = None (the window numbers
            # need no fit). The run decided at its start that this dataset's
            # chemistry is there (indexing._decide_eds_prior refuses the prior for
            # the WHOLE run while a fit the prior needs runs). Losing it half-way
            # -- the file was switched, its fit dropped -- must not switch the
            # prior off for the rest of the run: that is the off-then-on run the
            # decision exists to prevent. Fail loud instead.
            raise
        except Exception:
            return None
    if not at_maps:
        return None

    # A window the quantification could not price is left OUT of the maps and
    # the rest renormalise without it. For a DISPLAY that is the right trade;
    # for the PRIOR it is not, and the difference is not a matter of taste:
    # `chemistry_fit` reads an element that is not in the measured dict as
    # `p.get(el, 0.0)` -- a measured ZERO -- and its missing-major veto then
    # floors every phase whose major element is the one we dropped. A hole in
    # the composition does not merely weaken the ranking, it rules out exactly
    # the phases that contain the missing element, over the whole map, at 20x.
    #
    # `eds_pixel_chemistry` already refuses this on the single-pixel path
    # (2026-09-13). This is the same guard at the map boundary, because
    # `_build_eds_phase_weights` never looks at the provenance. Reachable
    # today: the beam voltage comes from the file, so a low-kV scan cannot
    # price Cu K and the prior would suppress every Cu-bearing phase.
    #
    # OFF, not degraded. The classifier can be taught that an element is
    # unmeasured rather than absent because it scores a whole map at once and
    # can renormalise both sides; the prior multiplies a per-pixel weight onto
    # a pattern score, and a weight computed on a different element set is not
    # comparable with the ones beside it.
    excluded = list((getattr(at_maps, "provenance", None) or {}).get(
        "excluded_windows", []))
    if excluded:
        names = ", ".join(str(e.get("element")) for e in excluded)
        _warn_once(
            "excluded_window",
            f"EDS chemistry prior OFF for this dataset: window(s) {names} "
            f"cannot be quantified, so the composition would have a hole the "
            f"phase scorer reads as a measured zero "
            f"({excluded[0].get('reason')})",
            subject=[e.get("element") for e in excluded])
        return None

    want = expected_shape
    if want is None and selection_mask is not None:
        want = tuple(np.asarray(selection_mask).shape)
    if want is not None and tuple(int(v) for v in want) != (int(n_rows), int(n_cols)):
        raise EdsGridMismatch(
            f"EDS grid {int(n_rows)}x{int(n_cols)} does not match the EBSD "
            f"navigation grid {tuple(int(v) for v in want)}. Pairing them would "
            f"give every pattern the wrong pixel's chemistry."
        )

    elements = list(at_maps.keys())
    n_full = n_rows * n_cols
    # (N_full, E)
    arr = np.stack([np.asarray(at_maps[el]).reshape(-1)[:n_full] for el in elements], axis=1)
    if selection_mask is not None:
        flat = np.asarray(selection_mask, dtype=bool).reshape(-1)
        idx = np.where(flat)[0]
    else:
        idx = np.arange(n_full)
    out = []
    for i in idx:
        row = arr[i]
        out.append({el: float(row[j]) for j, el in enumerate(elements) if row[j] > 0.0})
    return out
