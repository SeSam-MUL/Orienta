"""Offline generator for ``backend/eds_spectrum/xray_tables.json`` (dev-only).

Reads the X-ray line energies and relative weights from exspy's element
database and the absorption edges, K fluorescence yields and total mass
attenuation coefficients from xraylib, and writes one small JSON that the
product code loads (``backend/eds_spectrum/tables.py``).  Neither exspy nor
xraylib is ever imported by product code and neither is a runtime dependency.

Usage (both libraries live outside the project environment)::

    ORIENTA_EXSPY_PATH=<dir containing exspy/ and xraylib.py> \
        python scripts/build_xray_tables.py [output.json]

``ORIENTA_EXSPY_PATH`` may hold several directories separated by ``os.pathsep``.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path

N_LOG_GRID = 200
E_MIN_KEV = 0.1
E_MAX_KEV = 30.0
EDGE_DELTA_KEV = 1e-4
Z_MAX = 92

SOURCES = {
    "lines": (
        "X-ray line energies and relative weights: exspy element database "
        "(exspy._misc.elements, GPL-3.0; derived from the HyperSpy/EDSpy "
        "tables, Bearden 1967 and Zschornack 2007)."
    ),
    "edges": (
        "K, L3 and M5 absorption edge energies: xraylib EdgeEnergy "
        "(BSD-3-Clause; Elam, Ravel & Sieber 2002, Radiat. Phys. Chem. 63, 121)."
    ),
    "fluor_yield": (
        "K-shell fluorescence yields: xraylib FluorYield (BSD-3-Clause; "
        "Krause 1979, J. Phys. Chem. Ref. Data 8, 307)."
    ),
    "mu_rho": (
        "Total mass attenuation coefficient mu/rho in cm^2/g, 0.1-30 keV, "
        "200 log-spaced points, refined by bisection wherever log-log interpolation "
        "would be off by more than 0.3 %, plus points on both sides of every discontinuity "
        "of the xraylib table that is larger than 3 % between adjacent scan "
        "samples (0.5 eV spacing up to 2 keV, 0.05 % above), i.e. K, L and M "
        "edges with a visible step; the steps are located in the table itself "
        "because they lie up to ~10 eV from the nominal edge energy. Smaller "
        "steps (most heavy-element M edges) and N-shell edges are not resolved. "
        "Source: xraylib CS_Total (BSD-3-Clause; Schoonjans et al. 2011, "
        "Spectrochim. Acta B 66, 776)."
    ),
}


def _sig(x: float, n: int) -> float:
    """Round to ``n`` significant digits."""
    if x == 0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{n - 1}e}")


def _load_libs():
    raw = os.environ.get("ORIENTA_EXSPY_PATH")
    if not raw:
        print(
            "ORIENTA_EXSPY_PATH is not set; point it at the directory that holds "
            "exspy and xraylib (they are not project dependencies).",
            file=sys.stderr,
        )
        sys.exit(2)
    for part in reversed([p for p in raw.split(os.pathsep) if p]):
        sys.path.insert(0, part)
    warnings.filterwarnings("ignore")
    from exspy._misc.elements import elements_db  # noqa: PLC0415
    import xraylib  # noqa: PLC0415

    return elements_db, xraylib


def _edge(xraylib, z: int, shell: int):
    try:
        e = float(xraylib.EdgeEnergy(z, shell))
    except ValueError:
        return None
    return _sig(e, 6) if e > 0 else None


JUMP_MIN_RATIO = 1.03      # adjacent-sample ratio that counts as a step
SCAN_FINE_STEP_KEV = 5e-4  # 0.5 eV steps below SCAN_FINE_MAX_KEV ...
SCAN_FINE_MAX_KEV = 2.0
SCAN_REL_STEP = 5e-4       # ... then 0.05 % steps up to E_MAX_KEV
BISECT_TOL_KEV = 1e-7
E_DIGITS = 10              # energies stored with 10 significant digits
REFINE_TOL = 3e-3          # max log-log interpolation error at refined midpoints
REFINE_MIN_KEV = 3e-4
REFINE_MIN_REL = 1e-3
REFINE_MAX_ROUNDS = 20


def _scan_energies():
    es = []
    e = E_MIN_KEV
    while e < SCAN_FINE_MAX_KEV:
        es.append(e)
        e += SCAN_FINE_STEP_KEV
    while e < E_MAX_KEV:
        es.append(e)
        e *= 1.0 + SCAN_REL_STEP
    es.append(E_MAX_KEV)
    return es


def _cs_total(xraylib, z: int, e: float):
    try:
        return float(xraylib.CS_Total(z, e))
    except ValueError:  # outside the tabulated range
        return None


def _find_steps(xraylib, z: int):
    """Every discontinuity of CS_Total(z, E) in [E_MIN_KEV, E_MAX_KEV].

    Absorption edges are found in the table itself rather than at the nominal
    ``EdgeEnergy``: the steps sit up to ~10 eV away from it (Al K: nominal
    1.5596 keV, step near 1.559; O K: nominal 0.532, step near 0.543), and
    neighbouring L/M edges can lie within a few eV of each other (Cr L3/L2).
    Returns ``[(lo, hi), ...]``, each bracketing one step to ``BISECT_TOL_KEV``.
    Steps smaller than ``JUMP_MIN_RATIO`` (e.g. most M edges of heavy
    elements) are not resolved.
    """
    es = _scan_energies()
    mus = [_cs_total(xraylib, z, e) for e in es]
    steps = []
    for i in range(len(es) - 1):
        if mus[i] is None or mus[i + 1] is None:
            continue
        if mus[i + 1] / mus[i] < JUMP_MIN_RATIO:
            continue
        lo, hi = es[i], es[i + 1]
        mid_mu = 0.5 * (mus[i] + mus[i + 1])
        while hi - lo > BISECT_TOL_KEV:
            mid = 0.5 * (lo + hi)
            if xraylib.CS_Total(z, mid) < mid_mu:
                lo = mid
            else:
                hi = mid
        steps.append((lo, hi))
    return steps


def _refine(xraylib, z: int, energies, mu):
    """Insert log-midpoints until log-log interpolation is within REFINE_TOL.

    Smooth but sharp features (white lines just above an L edge) are narrower
    than the base grid spacing.  Intervals narrower than ``REFINE_MIN_REL``
    (relative) or ``REFINE_MIN_KEV`` are left alone: they are the bracket
    pairs around a step, or a step too small to resolve.
    """
    energies, mu = list(energies), list(mu)
    for _ in range(REFINE_MAX_ROUNDS):
        new = []
        for i in range(len(energies) - 1):
            e0, e1 = energies[i], energies[i + 1]
            if e1 - e0 < REFINE_MIN_KEV or e1 / e0 - 1.0 < REFINE_MIN_REL:
                continue
            em = _sig(math.sqrt(e0 * e1), E_DIGITS)
            true = _cs_total(xraylib, z, em)
            if true is None:
                continue
            t = (math.log(em) - math.log(e0)) / (math.log(e1) - math.log(e0))
            pred = math.exp((1 - t) * math.log(mu[i]) + t * math.log(mu[i + 1]))
            if abs(pred / true - 1.0) > REFINE_TOL:
                new.append((em, true))
        if not new:
            break
        merged = sorted(list(zip(energies, mu)) + new)
        energies = [e for e, _ in merged]
        mu = [m for _, m in merged]
    return energies, mu


def _build_element(db, xraylib, symbol: str):
    z = int(xraylib.SymbolToAtomicNumber(symbol))

    lines = {"K": [], "L": [], "M": []}
    try:
        raw = db[symbol]["Atomic_properties"]["Xray_lines"].as_dictionary()
    except (AttributeError, KeyError):  # H, He: no tabulated lines
        raw = {}
    for key, val in raw.items():
        fam = key[0]
        if fam in lines:
            lines[fam].append(
                [round(float(val["energy (keV)"]), 5), _sig(float(val["weight"]), 6)]
            )
    for fam in lines:
        lines[fam].sort()

    shells = {
        "K": xraylib.K_SHELL,
        "L": xraylib.L3_SHELL,
        "M": xraylib.M5_SHELL,
    }
    edges = {fam: _edge(xraylib, z, sh) for fam, sh in shells.items()}

    try:
        fy = float(xraylib.FluorYield(z, xraylib.K_SHELL))
    except ValueError:
        fy = 0.0

    # Grid: log-spaced plus points on both sides of every step found.
    grid = [E_MIN_KEV * (E_MAX_KEV / E_MIN_KEV) ** (i / (N_LOG_GRID - 1)) for i in range(N_LOG_GRID)]
    steps = _find_steps(xraylib, z)
    for lo, hi in steps:
        grid.extend([lo - EDGE_DELTA_KEV, lo, hi, hi + EDGE_DELTA_KEV])
    energies, mu = [], []
    for e in sorted({_sig(g, E_DIGITS) for g in grid}):
        m = _cs_total(xraylib, z, e)
        if m is None:
            continue
        if not (m > 0 and math.isfinite(m)):
            raise ValueError(f"non-positive mu/rho for {symbol} at {e} keV")
        energies.append(e)
        mu.append(m)
    energies, mu = _refine(xraylib, z, energies, mu)
    mu = [_sig(m, 6) for m in mu]
    steps_out = [[_sig(lo, E_DIGITS), _sig(hi, E_DIGITS)] for lo, hi in steps]

    return {
        "Z": z,
        "lines": lines,
        "edges": edges,
        "fluor_yield_k": _sig(fy, 6),
        "mu_energy": energies,
        "mu_rho": mu,
        "mu_steps": steps_out,
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = (
        Path(argv[0])
        if argv
        else Path(__file__).resolve().parents[1] / "backend" / "eds_spectrum" / "xray_tables.json"
    )
    db, xraylib = _load_libs()

    elements = {}
    for z in range(1, Z_MAX + 1):
        symbol = xraylib.AtomicNumberToSymbol(z)
        elements[symbol] = _build_element(db, xraylib, symbol)

    out.write_text(
        json.dumps({"sources": SOURCES, "elements": elements}, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(elements)} elements)")
    print(f"xraylib CS_Total(Al, 1.740 keV) = {xraylib.CS_Total(13, 1.740):.2f} cm^2/g")
    return 0


if __name__ == "__main__":
    sys.exit(main())
