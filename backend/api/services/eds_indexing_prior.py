"""EDS chemistry prior for multi-phase indexing.

Turns per-pixel EDS composition + per-phase strengths into a (P, N) multiplier
applied to each phase's pattern score before the winner is chosen. Reuses the
Phase-Test chemistry so 'consistency' matches the rest of the app. Off (all
strengths 0) or no EDS => the caller passes None and indexing is bit-identical.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from backend.api.services.crystal_hint_phase_fit import (
    chemistry_fit, phase_nominal_at_pct,
)


def expected_at_pct_for_phases(
    formulas: list[str], overrides: Optional[list[dict]] = None,
) -> list[dict]:
    """Expected At.% (0-100) per phase from its formula, merged with an optional
    per-phase override dict ({element: at_pct}) that wins over the stoichiometric
    value. An unparseable/empty formula with no override yields {} (neutral)."""
    out: list[dict] = []
    for i, f in enumerate(formulas):
        exp = dict(phase_nominal_at_pct(f or ""))
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


def _build_at_pct_maps_for_loaded_file():
    # Imported lazily so the pure functions above have no FastAPI/route dependency
    # and the test can monkeypatch this symbol.
    from backend.api.routes.eds import _build_at_pct_maps_for_loaded_file as _f
    return _f()


def measured_atpct_per_pixel(
    selection_mask: Optional[np.ndarray] = None,
) -> Optional[list]:
    """Per RESULT-pixel At.% dicts (0-100), row-major over the nav grid. With
    selection_mask, only selected pixels in row-major (ascending-flat) order — the
    same order the pattern stack / result use. Returns None if no EDS is available."""
    try:
        at_maps, n_rows, n_cols, _ = _build_at_pct_maps_for_loaded_file()
    except Exception:
        return None
    if not at_maps:
        return None
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
