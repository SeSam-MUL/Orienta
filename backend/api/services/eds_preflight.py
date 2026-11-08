"""Pre-flight check: is the EDS chemistry of the loaded file actually usable
for the phases the user is about to index with?

Motivation (measured on ProbeB, 2026-08-04). The old UI gated the whole EDS
prior on a single boolean ``has_eds``. That is not enough:

* the EDS map and the EBSD navigation grid were never compared, so a size
  mismatch silently paired each pattern with the WRONG pixel's chemistry;
* a phase whose defining elements are not even measured got a chemistry score
  anyway (fail-soft 1.0), i.e. the prior "confirmed" a phase it could not see;
* nothing told the user that a phase is chemically unsupported ANYWHERE in the
  map before spending a full indexing run on it. In the case that triggered
  this work, alpha-Al(Fe,Mn)Si took 10 % of the map while its defining Fe/Mn
  are present on only 1.8 % of it.

The plausibility test deliberately reuses the exact rule the veto in
``chemistry_fit`` applies (defining element = nominal fraction >= _MAJOR_REQ,
present = measured >= _ABSENT). A phase reported at 0 % here is a phase the
prior will veto at every pixel — the two can never disagree.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from backend.api.services.crystal_hint_phase_fit import phase_nominal_at_pct

# Same constants the veto uses, imported by value so the two stay in step.
# (crystal_hint_phase_fit keeps them local to the function; re-declaring them
# here with an explicit test that pins the pair is the least-coupling option.)
DEFINING_FRACTION = 0.05   # nominal at% fraction that makes an element defining
PRESENT_AT_PCT = 2.0       # measured at% below which an element counts as absent

# Below this many total X-ray counts per pixel the standardless Cliff-Lorimer
# at% is dominated by counting noise. Oxford avoid quantification entirely for
# this reason; we quantify, so we at least say when the counts are thin.
LOW_COUNTS_PER_PX = 500.0

Severity = str  # "ok" | "warn" | "block"


def defining_elements(formula: str) -> dict[str, float]:
    """Elements whose ABSENCE makes the phase impossible, with nominal at%.

    Mirrors the veto rule: nominal fraction >= DEFINING_FRACTION.
    """
    nominal = phase_nominal_at_pct(formula or "")
    if not nominal:
        return {}
    total = sum(nominal.values())
    if total <= 1e-9:
        return {}
    return {el: v for el, v in nominal.items()
            if (v / total) >= DEFINING_FRACTION and el not in ("O", "C")}


def _ebsd_nav_shape() -> Optional[tuple[int, int]]:
    """(n_rows, n_cols) of the loaded EBSD navigation grid, or None."""
    try:
        from backend.api.routes.ebsd_viewer import _get_active_signal
        signal = _get_active_signal()
        if signal is None:
            return None
        nav = signal.axes_manager.navigation_shape  # (cols, rows)
        if len(nav) < 2:
            return None
        return int(nav[1]), int(nav[0])
    except Exception:
        return None


def _median_window_counts() -> Optional[float]:
    """Median over pixels of the summed RAW window counts, or None.

    at% is closed to 100 %, so it cannot show weak signal; counts can.
    LOW_COUNTS_PER_PX is a threshold on counts as AZtec integrates them
    (background included), so these are AZtec's windows even on a file whose
    EDS numbers come from the spectrum fit -- read through
    ``eds_spectrum_service.window_maps``, which needs no fit.
    """
    try:
        from backend.api.routes.eds import get_extractor  # type: ignore
        from backend.api.services.eds_spectrum_service import window_maps
        labels, read = window_maps(get_extractor())
        tot = None
        for name in labels:
            d = read(name)
            if d is None:
                continue
            a = np.asarray(d, dtype=np.float64).reshape(-1)
            tot = a if tot is None else tot + a
        if tot is not None and tot.size:
            return float(np.median(tot))
    except Exception:
        return None
    return None


def _signal_check() -> dict:
    median_counts = _median_window_counts()
    if median_counts is None:
        return {"severity": "warn", "ok": True,
                "median_counts_per_px": None,
                "code": "countsUnavailable", "params": {},
                "detail": "raw counts unavailable — cannot judge signal"}
    weak = median_counts < LOW_COUNTS_PER_PX
    return {
        "severity": "warn" if weak else "ok", "ok": True,
        "median_counts_per_px": round(median_counts, 1),
        "detail": (f"median {median_counts:,.0f} counts/pixel — thin, the "
                   f"at% conversion is noise-dominated"
                   if weak else f"median {median_counts:,.0f} counts/pixel"),
        "code": "signalThin" if weak else "signalOk",
        # Raw, not pre-formatted: "12,345" reads as twelve point three
        # four five in German. The page formats it for its own locale.
        "params": {"median": round(median_counts)},
    }


def run_preflight(phase_files: list[str],
                  phase_formulas: Optional[list[str]] = None) -> dict:
    """Evaluate EDS readiness for ``phase_files``.

    ``phase_formulas`` may be supplied by the caller (it already resolves them
    for the prior); when omitted they are looked up from the file metadata.
    Never raises — every failure becomes a reported check so the UI can show a
    reason instead of an empty panel.
    """
    checks: dict = {}
    blocking: list[str] = []

    # ---- 1. EDS present -------------------------------------------------
    at_maps = None
    eds_rows = eds_cols = None
    try:
        # The PRIOR's numbers (eds_indexing_prior.PRIOR_EDS_SOURCE): this
        # check predicts what the prior will veto, so it must read the same
        # source -- the window maps while the prior is kept on them.
        from backend.api.services import eds_indexing_prior as _prior
        at_maps, eds_rows, eds_cols, _ = _prior._build_at_pct_maps_for_loaded_file()
    except Exception as exc:
        from backend.api.services.eds_spectrum_service import SPECTRUM_FIT_ERRORS
        detail = getattr(exc, "detail", exc)
        if isinstance(detail, dict):      # the spectrum fit's 503 / 500
            detail = detail.get("message") or detail
        checks["eds_present"] = {
            "severity": "block", "ok": False,
            "detail": str(detail)[:200], "elements": [],
            "code": "edsUnavailable", "params": {},
        }
        blocking.append("eds_present")
        if isinstance(exc, SPECTRUM_FIT_ERRORS):
            # The at% wait for the spectrum fit (only when the prior reads the
            # file's own source); the counts per pixel are AZtec's raw
            # windows and do not.
            checks["signal"] = _signal_check()
        return {"checks": checks, "phases": [], "can_index": False,
                "blocking": blocking, "eds_usable": False}

    elements = sorted(at_maps.keys())
    checks["eds_present"] = {"severity": "ok", "ok": True,
                             "detail": f"{len(elements)} elements",
                             "code": "elementsFound", "params": {"count": len(elements)},
                             "elements": elements}

    # ---- 2. grid match (BLOCKING) ---------------------------------------
    nav = _ebsd_nav_shape()
    grid_ok = nav is not None and (int(eds_rows), int(eds_cols)) == nav
    checks["grid"] = {
        "severity": "ok" if grid_ok else "block",
        "ok": bool(grid_ok),
        "eds_shape": [int(eds_rows), int(eds_cols)],
        "ebsd_shape": list(nav) if nav else None,
        "detail": ("EDS and EBSD grids match"
                   if grid_ok else
                   "EDS grid does not match the EBSD navigation grid — every "
                   "pattern would be paired with the wrong pixel's chemistry"),
        "code": "gridMatch" if grid_ok else "gridMismatch",
        "params": {},
    }
    if not grid_ok:
        blocking.append("grid")

    # ---- 3. signal strength ---------------------------------------------
    checks["signal"] = _signal_check()

    # ---- 4 + 5. per-phase coverage and plausibility ----------------------
    n_px = int(eds_rows) * int(eds_cols)
    arr = {el: np.asarray(at_maps[el], dtype=np.float64).reshape(-1)[:n_px]
           for el in elements}

    if phase_formulas is None:
        phase_formulas = []
        for p in phase_files:
            try:
                from pathlib import Path as _P
                from phase_metadata import get_phase_metadata
                phase_formulas.append(
                    getattr(get_phase_metadata(_P(p)), "formula", "") or "")
            except Exception:
                phase_formulas.append("")

    phases: list[dict] = []
    any_missing = False
    for path, formula in zip(phase_files, phase_formulas):
        defining = defining_elements(formula)
        # The displayed name of a phase file the request named — separator
        # rules must not depend on the platform. See display_names.
        from display_names import display_stem
        entry: dict = {"path": path, "name": display_stem(path), "formula": formula,
                       "defining": {k: round(v, 1) for k, v in defining.items()}}
        if not defining:
            entry.update({"severity": "warn", "max_area_pct": None,
                          "missing_elements": [],
                          "code": "noFormula", "params": {},
                          "detail": "no parseable formula — chemistry cannot "
                                    "judge this phase (it will pass unchanged)"})
            any_missing = True
            phases.append(entry)
            continue

        missing = [el for el in defining if el not in arr]
        entry["missing_elements"] = missing
        if missing:
            any_missing = True
            entry.update({"severity": "warn", "max_area_pct": None,
                          "code": "notMeasured",
                          "params": {"elements": ", ".join(missing)},
                          "detail": "not measured: " + ", ".join(missing)
                                    + " — chemistry is blind to this phase"})
            phases.append(entry)
            continue

        # NOT an estimate of the phase's area — an UPPER BOUND on it. The mask
        # is "chemistry does not veto this phase here", i.e. every defining
        # element measured at or above the absence threshold. The phase's true
        # area can only be smaller. For a one-element phase (Al, Si) the bound
        # is weak by construction (trace levels of Al are everywhere); for the
        # multi-element intermetallics it is the number that matters, and it is
        # exactly the one that was missing: alpha bounded at ~1 % would have
        # exposed the 10 % it actually got.
        ok_mask = np.ones(n_px, dtype=bool)
        for el in defining:
            ok_mask &= (arr[el] >= PRESENT_AT_PCT)
        frac = 100.0 * float(ok_mask.sum()) / max(1, n_px)
        entry["max_area_pct"] = round(frac, 2)
        entry["threshold_at_pct"] = PRESENT_AT_PCT
        entry["is_upper_bound"] = True
        entry["weak_bound"] = len(defining) < 2
        if frac <= 0.0:
            entry.update({"severity": "warn",
                          "code": "vetoed",
                          "params": {"elements": ", ".join(sorted(defining)),
                                     "threshold": PRESENT_AT_PCT},
                          "detail": "chemistry vetoes this phase on the whole "
                                    "map — no pixel carries "
                                    + ", ".join(sorted(defining))
                                    + f" at >= {PRESENT_AT_PCT:g} at%"})
        else:
            entry["severity"] = "ok"
            entry["code"] = "notRuledOut"
            entry["params"] = {"pct": round(frac, 2),
                               "elements": ", ".join(sorted(defining)),
                               "threshold": PRESENT_AT_PCT}
            entry["detail"] = (
                f"not ruled out on {frac:.2f} % of the map "
                f"(needs " + ", ".join(sorted(defining))
                + f" >= {PRESENT_AT_PCT:g} at%) — upper bound on its area")
        phases.append(entry)

    checks["coverage"] = {
        "severity": "warn" if any_missing else "ok", "ok": True,
        "code": "coverageIncomplete" if any_missing else "coverageComplete",
        "params": {},
        "detail": ("some phases cannot be judged chemically"
                   if any_missing else "all defining elements are measured"),
    }

    return {
        "checks": checks,
        "phases": phases,
        "blocking": blocking,
        "can_index": not blocking,
        "eds_usable": not blocking,
    }
