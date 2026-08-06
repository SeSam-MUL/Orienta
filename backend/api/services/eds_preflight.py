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
        from backend.api.routes.eds import _build_at_pct_maps_for_loaded_file
        at_maps, eds_rows, eds_cols, _ = _build_at_pct_maps_for_loaded_file()
    except Exception as exc:
        checks["eds_present"] = {
            "severity": "block", "ok": False,
            "detail": str(getattr(exc, "detail", exc))[:200], "elements": [],
        }
        blocking.append("eds_present")
        return {"checks": checks, "phases": [], "can_index": False,
                "blocking": blocking, "eds_usable": False}

    elements = sorted(at_maps.keys())
    checks["eds_present"] = {"severity": "ok", "ok": True,
                             "detail": f"{len(elements)} elements",
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
    }
    if not grid_ok:
        blocking.append("grid")

    # ---- 3. signal strength ---------------------------------------------
    # at% is closed to 100 %, so it cannot show weak signal. Use raw counts.
    median_counts = None
    try:
        from backend.api.routes.eds import get_extractor  # type: ignore
        ext = get_extractor()
        tot = None
        for name in ext.get_available_elements():
            d = ext.get_element_map(name)
            if d is None:
                continue
            a = np.asarray(d, dtype=np.float64).reshape(-1)
            tot = a if tot is None else tot + a
        if tot is not None and tot.size:
            median_counts = float(np.median(tot))
    except Exception:
        median_counts = None
    if median_counts is None:
        checks["signal"] = {"severity": "warn", "ok": True,
                            "median_counts_per_px": None,
                            "detail": "raw counts unavailable — cannot judge signal"}
    else:
        weak = median_counts < LOW_COUNTS_PER_PX
        checks["signal"] = {
            "severity": "warn" if weak else "ok", "ok": True,
            "median_counts_per_px": round(median_counts, 1),
            "detail": (f"median {median_counts:,.0f} counts/pixel — thin, the "
                       f"at% conversion is noise-dominated"
                       if weak else f"median {median_counts:,.0f} counts/pixel"),
        }

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
        from pathlib import Path as _P
        entry: dict = {"path": path, "name": _P(path).stem, "formula": formula,
                       "defining": {k: round(v, 1) for k, v in defining.items()}}
        if not defining:
            entry.update({"severity": "warn", "max_area_pct": None,
                          "missing_elements": [],
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
                          "detail": "chemistry vetoes this phase on the whole "
                                    "map — no pixel carries "
                                    + ", ".join(sorted(defining))
                                    + f" at >= {PRESENT_AT_PCT:g} at%"})
        else:
            entry["severity"] = "ok"
            entry["detail"] = (
                f"not ruled out on {frac:.2f} % of the map "
                f"(needs " + ", ".join(sorted(defining))
                + f" >= {PRESENT_AT_PCT:g} at%) — upper bound on its area")
        phases.append(entry)

    checks["coverage"] = {
        "severity": "warn" if any_missing else "ok", "ok": True,
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
