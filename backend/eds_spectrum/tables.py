"""Vendored X-ray tables: line energies/weights, edges, yields, attenuation.

The data come from ``xray_tables.json``, written offline by
``scripts/build_xray_tables.py`` (sources in :data:`SOURCES`).  Nothing here
imports exspy or xraylib.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_FAMILIES = ("K", "L", "M")

with open(Path(__file__).with_name("xray_tables.json"), encoding="utf-8") as _fh:
    _RAW = json.load(_fh)

SOURCES: Dict[str, str] = dict(_RAW["sources"])
_ELEMENTS: Dict[str, dict] = _RAW["elements"]

# Per-element log-log interpolation arrays, built lazily.
_MU_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _element(symbol: str) -> dict:
    try:
        return _ELEMENTS[symbol]
    except KeyError:
        raise KeyError(f"no X-ray table for element {symbol!r}") from None


def family_lines(symbol: str, family: str) -> List[Tuple[float, float]]:
    """(energy keV, relative weight) of every line of a family, by energy.

    The main line has weight 1.0.  Empty if the element has none (or is
    unknown).
    """
    el = _ELEMENTS.get(symbol)
    if el is None:
        return []
    return [(float(e), float(w)) for e, w in el["lines"].get(family, [])]


def main_line(symbol: str, family: str) -> Tuple[float, float]:
    """The weight-1.0 line of a family as (energy keV, 1.0)."""
    lines = family_lines(symbol, family)
    if not lines:
        raise KeyError(f"no {family} lines for {symbol!r}")
    return max(lines, key=lambda ew: ew[1])


def edge_kev(symbol: str, family: str) -> Optional[float]:
    """Absorption edge of the family (K -> K, L -> L3, M -> M5), keV."""
    return _element(symbol)["edges"].get(family)


def fluor_yield_k(symbol: str) -> float:
    """K-shell fluorescence yield (0.0 where the element has none)."""
    return float(_element(symbol)["fluor_yield_k"])


def mu_rho(symbol: str, energy_kev: np.ndarray) -> np.ndarray:
    """Total mass attenuation coefficient in cm^2/g.

    Log-log interpolation on the stored edge-aware grid (0.1-30 keV);
    energies outside the grid take the value at the nearest end.
    """
    grid = _MU_CACHE.get(symbol)
    if grid is None:
        el = _element(symbol)
        grid = (np.log(np.asarray(el["mu_energy"], dtype=float)),
                np.log(np.asarray(el["mu_rho"], dtype=float)))
        _MU_CACHE[symbol] = grid
    e = np.asarray(energy_kev, dtype=float)
    return np.exp(np.interp(np.log(e), grid[0], grid[1]))


def mu_rho_mix(mass_fractions: Dict[str, float], energy_kev: np.ndarray) -> np.ndarray:
    """Mass-fraction-weighted mu/rho of a mixture (cm^2/g)."""
    e = np.asarray(energy_kev, dtype=float)
    out = np.zeros_like(e)
    for symbol, w in mass_fractions.items():
        out = out + float(w) * mu_rho(symbol, e)
    return out


def mu_steps(symbol: str) -> List[Tuple[float, float]]:
    """(below, above) energies bracketing each resolved step of mu/rho, keV."""
    return [(float(lo), float(hi)) for lo, hi in _element(symbol)["mu_steps"]]
