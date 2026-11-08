"""Linear EDS spectrum model: components, design matrix, pile-up rebuild.

The model is the validated bench model (a non-negative least-squares fit) on the vendored tables of :mod:`backend.eds_spectrum.tables`.
Every column is non-negative and enters linearly:

* one block per line family (``"Fe K"``): Gaussians at the table energies
  with the table weights, width from the SDD law
  ``FWHM^2 = f0^2 + 2.5e-3 keV * (E - 5.899)``, plus Si escape peaks
  (E - 1.740 keV, Reed & Ware fraction);
* incomplete-charge-collection tails on Al K and Si K (two decay lengths);
* pulse pile-up (``pile lo``, ``pile hi``, ``pile 3``);
* the continuum: Kramers x detector efficiency x sample self-absorption x ten
  non-negative cubic B-splines.

The four non-linear parameters are in :class:`NonlinearParams`.  Nothing here
reads global mutable state; a :class:`SpectrumModel` only caches things that
are functions of its own constructor arguments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import BSpline

from . import tables

MODEL_VERSION = "eds-spectrum-fit/1"
E_LO = 0.25
E_HI = 14.0

#: Si K absorption edge and K-alpha line used for the escape peaks, keV.
ESC_EDGE = 1.839
ESC_LINE = 1.740
#: Si K jump ratio in the Reed & Ware escape fraction.
SI_JUMP_RATIO = 10.9
#: Smallest U = E0 / E_edge for which a family is modelled.
MIN_OVERVOLTAGE = 1.5
#: Pile-up "lo" split: both photons below this energy, keV.
PILE_LO_KEV = 2.5
#: ICC tails: decay lengths and cut-off, keV.
TAIL_DECAYS = (0.02, 0.08)
TAIL_CUT = 0.6
N_SPLINES = 10

#: Detector window: 0.3 um polyimide (C22H10N2O5, 1.42 g/cm3), 30 nm Al
#: (2.70 g/cm3), 50 nm Si dead layer (2.33 g/cm3); thicknesses in cm.
POLYIMIDE_FORMULA = {"C": 22, "H": 10, "N": 2, "O": 5}
POLY_RHO, POLY_T = 1.42, 0.3e-4
AL_RHO, AL_T = 2.70, 0.03e-4
SI_RHO, SI_T = 2.33, 0.05e-4

#: Elements always modelled when excited: (symbol, families).
CANDIDATES: List[Tuple[str, Tuple[str, ...]]] = [
    ("C", ("K",)), ("N", ("K",)), ("O", ("K",)),
    ("Mg", ("K",)), ("Al", ("K",)), ("Si", ("K",)),
    ("Ti", ("K", "L")), ("Cr", ("K", "L")), ("Mn", ("K", "L")),
    ("Fe", ("K", "L")), ("Ni", ("K", "L")), ("Cu", ("K", "L")),
    ("Zn", ("K", "L")), ("Zr", ("L",)),
]


@dataclass(frozen=True)
class SpectrumMeta:
    """Channel grid and acquisition conditions of one spectrum."""
    start_kev: float
    width_kev: float
    n_channels: int
    beam_kv: float
    detector: str = ""
    tilt_deg: Optional[float] = None

    def energies(self) -> np.ndarray:
        return self.start_kev + self.width_kev * np.arange(self.n_channels)


@dataclass
class NonlinearParams:
    """The four non-linear parameters of the model."""
    offset_kev: float = 0.0
    gain: float = 1.0
    fwhm_mnka_kev: float = 0.13
    absorption_a: float = 3e-4

    def as_array(self) -> np.ndarray:
        return np.array([self.offset_kev, self.gain, self.fwhm_mnka_kev,
                         self.absorption_a], dtype=float)

    @classmethod
    def from_array(cls, p) -> "NonlinearParams":
        return cls(float(p[0]), float(p[1]), float(p[2]), float(p[3]))


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
_LABEL = re.compile(r"^\s*([A-Z][a-z]?)(?:\s+([KLM]))?")


def default_families(exported: List[str],
                     meta: Optional[SpectrumMeta] = None) -> List[Tuple[str, str]]:
    """Families to model: the exported elements plus the candidate list.

    ``exported`` holds element symbols (``"Fe"``) or window names that start
    with one (``"Fe Kα1"``, ``"Cu Lα1,2"``); a bare symbol brings every
    family the tables know.  A family is kept only if the tables have lines
    for it and, when ``meta`` is given, the beam excites its edge
    (overvoltage >= :data:`MIN_OVERVOLTAGE`).  Candidates come first in the
    order of :data:`CANDIDATES`, further exported families follow.
    """
    wanted: List[Tuple[str, str]] = []
    for sym, fams in CANDIDATES:
        wanted.extend((sym, f) for f in fams)
    for label in exported:
        m = _LABEL.match(str(label))
        if not m:
            continue
        sym, fam = m.group(1), m.group(2)
        for f in ((fam,) if fam else ("K", "L", "M")):
            if (sym, f) not in wanted:
                wanted.append((sym, f))
    out = []
    for sym, fam in wanted:
        if not tables.family_lines(sym, fam):
            continue
        if meta is not None:
            edge = tables.edge_kev(sym, fam)
            if edge is None or meta.beam_kv / edge < MIN_OVERVOLTAGE:
                continue
        out.append((sym, fam))
    return out


def label(family: Tuple[str, str]) -> str:
    return f"{family[0]} {family[1]}"


# ---------------------------------------------------------------------------
# physics building blocks
# ---------------------------------------------------------------------------
def fwhm(E, f0):
    """Detector FWHM (keV) at energy E for FWHM(Mn K-alpha) = f0."""
    return np.sqrt(np.maximum(f0 ** 2 + 2.5e-3 * (E - 5.899), 1e-4))


def gauss(E, c, f0, w):
    """Unit-area Gaussian of weight w centred at c on the grid E."""
    s = fwhm(c, f0) / 2.3548
    return w * np.exp(-0.5 * ((E - c) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def _centred_kernel(dE: float, sigma: float, n_sigma: float = 4.0) -> np.ndarray:
    """Unit-sum Gaussian on +-n_sigma, centred at the middle of the array.

    ``np.convolve(..., mode="same")`` keeps the kernel's middle element at
    zero shift; a kernel sampled on the energy grid (peak at the line) would
    move the result far off the array.
    """
    k = int(np.ceil(n_sigma * sigma / dE))
    g = np.exp(-0.5 * (np.arange(-k, k + 1) * dE / sigma) ** 2)
    return g / g.sum()


@lru_cache(maxsize=None)
def escape_frac(e: float) -> float:
    """Reed & Ware (1972) Si escape probability at normal incidence."""
    if e <= ESC_EDGE:
        return 0.0
    mu_i = float(tables.mu_rho("Si", np.array([e]))[0])
    mu_k = float(tables.mu_rho("Si", np.array([ESC_LINE]))[0])
    x = mu_k / mu_i
    return float(0.5 * tables.fluor_yield_k("Si") * (1 - 1 / SI_JUMP_RATIO)
                 * (1 - x * np.log(1 + 1 / x)))


@lru_cache(maxsize=None)
def _polyimide_mass_fractions() -> Dict[str, float]:
    from eds_utils import ATOMIC_MASSES  # root module, same as backend routes
    mass = {s: n * ATOMIC_MASSES[s] for s, n in POLYIMIDE_FORMULA.items()}
    tot = sum(mass.values())
    return {s: m / tot for s, m in mass.items()}


def det_eff(E) -> np.ndarray:
    """Detector efficiency: polyimide window, Al layer and Si dead layer."""
    E = np.maximum(np.asarray(E, dtype=float), 0.1)
    poly = tables.mu_rho_mix(_polyimide_mass_fractions(), E) * POLY_RHO * POLY_T
    al = tables.mu_rho("Al", E) * AL_RHO * AL_T
    si = tables.mu_rho("Si", E) * SI_RHO * SI_T
    return np.exp(-(poly + al + si))


def mu_mix(E, comp: Dict[str, float]) -> np.ndarray:
    """mu/rho (cm2/g) of a composition given as any positive weights."""
    tot = sum(comp.values())
    return tables.mu_rho_mix({k: v / tot for k, v in comp.items()}, E)


def bspline_basis(E, n: int = N_SPLINES) -> np.ndarray:
    """``n`` cubic B-splines on [E_LO, E_HI] with clamped ends (zero outside)."""
    t_in = np.linspace(E_LO, E_HI, n - 2)
    t = np.r_[[E_LO] * 3, t_in, [E_HI] * 3]
    cols = []
    for i in range(len(t) - 4):
        c = np.zeros(len(t) - 4)
        c[i] = 1
        cols.append(BSpline(t, c, 3, extrapolate=False)(E))
    return np.nan_to_num(np.array(cols).T)


def measured_pileup(y: np.ndarray, meta: SpectrumMeta) -> Dict[str, np.ndarray]:
    """First-pass pile-up shapes: the measured spectrum convolved with itself."""
    n = y.size
    E0 = meta.energies()
    yy = np.where(E0 > 0.15, np.asarray(y, dtype=float), 0)
    lo = np.where(E0 < PILE_LO_KEV, yy, 0)
    conv_all, conv_lo = np.convolve(yy, yy), np.convolve(lo, lo)
    shift = meta.start_kev / meta.width_kev  # E(i)+E(j) -> channel i+j+start/w
    idx = np.arange(n) - shift
    k = np.arange(conv_all.size)
    return {"pile lo": np.interp(idx, k, conv_lo, left=0, right=0),
            "pile hi": np.interp(idx, k, conv_all - conv_lo, left=0, right=0)}


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class SpectrumModel:
    """Design matrix of the linear model for one channel grid and family list.

    ``pile_cols`` are the pile-up shapes on the full channel grid; set them
    from :func:`measured_pileup` for a first pass and from
    :meth:`rebuild_pileup` once a fit exists.  Without them the matrix has no
    pile-up columns.
    """

    def __init__(self, meta: SpectrumMeta, families: List[Tuple[str, str]],
                 pile_cols: Optional[Dict[str, np.ndarray]] = None):
        self.meta = meta
        self.families = [tuple(f) for f in families]
        self.w = meta.width_kev
        self.E0 = meta.energies()
        self.lines = {f: tables.family_lines(*f) for f in self.families}
        for f, ls in self.lines.items():
            if not ls:
                raise ValueError(f"no lines for family {label(f)!r}")
        self.main_weight = {f: tables.main_line(*f)[1] for f in self.families}
        self.pile_cols: Dict[str, np.ndarray] = dict(pile_cols or {})
        self._B: Optional[np.ndarray] = None
        self._cont_key = None
        self._kd = self._mu = None

    # -- continuum -------------------------------------------------------
    def _continuum_terms(self, comp: Dict[str, float]):
        ck = tuple(sorted((k, round(v, 6)) for k, v in comp.items()))
        if self._cont_key != ck:
            Eg = np.maximum(self.E0, 0.05)
            sel = (self.E0 >= E_LO - 0.05) & (self.E0 <= E_HI + 0.05)
            kd = np.zeros_like(Eg)
            mu = np.ones_like(Eg)
            kram = np.clip((self.meta.beam_kv - Eg[sel]) / Eg[sel], 0, None)
            kd[sel] = kram * det_eff(Eg[sel])
            mu[sel] = mu_mix(Eg[sel], comp)
            self._kd, self._mu, self._cont_key = kd, mu, ck
            if self._B is None:
                self._B = bspline_basis(self.E0)
        return self._kd, self._mu, self._B

    # -- design matrix ---------------------------------------------------
    def design(self, params: NonlinearParams, comp: Dict[str, float]
               ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Matrix over the fit range, column names, boolean channel mask."""
        E = params.offset_kev + params.gain * self.E0
        m = (E >= E_LO) & (E <= E_HI)
        Em = E[m]
        f0 = params.fwhm_mnka_kev
        cols, names = [], []
        for fam in self.families:
            col = np.zeros_like(Em)
            for c, wgt in self.lines[fam]:
                col += gauss(Em, c, f0, wgt)
                ef = escape_frac(c)
                if ef > 0:
                    col += gauss(Em, c - ESC_LINE, f0, wgt * ef)
            cols.append(col * self.w)
            names.append(label(fam))
        for el in ("Al", "Si"):  # ICC tails, low side only, two decay lengths
            c = tables.main_line(el, "K")[0]
            d = c - Em
            g = _centred_kernel(Em[1] - Em[0], fwhm(c, f0) / 2.3548)
            for lam in TAIL_DECAYS:
                tail = np.where(d > 0, np.exp(-d / lam), 0) * (d < TAIL_CUT)
                tail = np.convolve(tail, g, mode="same")
                cols.append(tail / max(tail.sum(), 1e-12))
                names.append(f"{el} tail{int(lam * 1000)}")
        for nm, pc in self.pile_cols.items():
            cols.append(pc[m] / max(pc[m].sum(), 1e-12))
            names.append(nm)
        # continuum on the uncorrected grid (its shape barely changes with off/gain)
        kd, mu_all, B_all = self._continuum_terms(comp)
        mu = mu_all[m]
        absf = (1 - np.exp(-params.absorption_a * mu)) / np.maximum(
            params.absorption_a * mu, 1e-9)
        cont = kd[m] * absf
        B = B_all[m]
        for j in range(B.shape[1]):
            cols.append(cont * B[:, j])
            names.append(f"bg{j}")
        return np.array(cols).T, names, m

    # -- pile-up ---------------------------------------------------------
    def rebuild_pileup(self, params: NonlinearParams, coef: np.ndarray,
                       names: List[str], A: np.ndarray, mask: np.ndarray) -> None:
        """Replace ``pile_cols`` by shapes built from the fitted spectrum.

        The intrinsic spectrum (line areas incl. escapes, tails, continuum) is
        convolved with itself (two and three photons), split into low+low and
        the rest, and broadened with the detector width at the SUM energy.
        """
        n = self.E0.size
        Etrue = params.offset_kev + params.gain * self.E0
        dE = Etrue[1] - Etrue[0]
        I = np.zeros(n)

        def put(e, area):
            x = (e - Etrue[0]) / dE
            i = int(np.floor(x))
            f = x - i
            if 0 <= i < n - 1:
                I[i] += area * (1 - f)
                I[i + 1] += area * f

        cmap = dict(zip(names, coef))
        for fam in self.families:
            a_f = cmap[label(fam)]
            if a_f <= 0:
                continue
            for c, wgt in self.lines[fam]:
                put(c, a_f * wgt)
                ef = escape_frac(c)
                if ef > 0:
                    put(c - ESC_LINE, a_f * wgt * ef)
        smooth = np.zeros(n)
        for j, nm in enumerate(names):  # continuum and tails are already smooth
            if nm.startswith("bg") or "tail" in nm:
                smooth[mask] += A[:, j] * coef[j]
        I += smooth
        lo = np.where(Etrue < PILE_LO_KEV, I, 0)
        shift = Etrue[0] / dE
        c_all, c_lo = np.convolve(I, I), np.convolve(lo, lo)
        k2 = np.arange(2 * n - 1)
        idx2 = np.arange(n) - shift
        P_all = np.interp(idx2, k2, c_all, left=0, right=0)
        P_lo = np.interp(idx2, k2, c_lo, left=0, right=0)
        c3 = np.convolve(c_all, I)  # i+j+k -> + 2*start/w
        P3 = np.interp(np.arange(n) - 2 * shift, np.arange(c3.size), c3,
                       left=0, right=0)
        s = fwhm(np.maximum(Etrue, 0.05), params.fwhm_mnka_kev) / 2.3548
        G = np.exp(-0.5 * ((Etrue[:, None] - Etrue[None, :]) / s[None, :]) ** 2) \
            / (s[None, :] * np.sqrt(2 * np.pi))
        G *= dE
        self.pile_cols = {"pile lo": G @ P_lo, "pile hi": G @ (P_all - P_lo),
                          "pile 3": G @ P3}
