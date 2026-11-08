"""phi(rho z) matrix correction for bulk SEM-EDS: XPP with specimen tilt, plus Reed fluorescence.

What this computes
------------------
For an X-ray line ``i`` of element ``e`` in a homogeneous bulk specimen of mass-fraction
composition ``C``, the detected net intensity is modelled as

    I_i = G * R(E_i) * K_i * C_e * Q_i(E0) * ZA_i(C) * Fl_i(C)

* ``ZA_i(C)``  -- the integral of the depth distribution of ionisations phi(rho z) times the
  absorption on the way out, ``int phi(rho z) exp(-chi rho z) d(rho z)`` in g/cm^2, with
  ``chi = (mu/rho)_C(E_i) / sin(psi)``.  phi(rho z) is Pouchou & Pichoir's XPP model.  For a
  tilted specimen the product (``tilt_model="scaled"``) takes the ionisation area of the tilted
  specimen (backscatter loss of Pouchou, Pichoir & Boivin, ICXOM-12, 1991) and the
  normal-incidence curve compressed in depth by ``cos(tilt)**0.5``; the full tilt extension of
  that paper (``"xpp_ext"``) is kept for comparison and fails at 70 deg (see ``TILT_MODELS``).
  This carries the atomic-number (Z) and absorption (A) effects together.
* ``Fl_i(C)``  -- 1 + characteristic secondary fluorescence (Reed) of K secondaries, as in EPQ's
  ``Fluorescence.compute`` (one representative primary line per element, L preferred over K).
  L and M secondaries get no fluorescence term (see :func:`fluorescence_rows`).
* ``Q_i(E0)``  -- Pouchou & Pichoir's proportional ionisation cross-section at the beam energy,
  ``ln(U0) / (E_c^2 U0^m)``; it multiplies ``ZA`` back into an absolute ionisation integral.
* ``K_i``      -- omega * p * n_shell * b_shell / A: fluorescence yield, fraction of the shell's
  emission in the fitted main line, electrons and Bethe constant of the shell, atomic mass
  (the same physical constants as ``eds_utils._emission_k``).
* ``R(E_i)``   -- the instrument response (detector efficiency and whatever the physics above
  leaves out), measured once against reference regions *after* the matrix correction, so it is
  free of the matrix it was measured in.  See ``eds_utils``.
* ``G``        -- beam current x solid angle; cancels in the 100 % normalisation.

Standardless quantification inverts this by fixed-point iteration on ``C`` and normalises to
100 % (:func:`quantify` for one spectrum, :func:`quantify_map` for many pixels at once).

Everything is computed on arrays of compositions (one row per pixel); the scalar functions are
thin wrappers over the same code, so the EPQ oracle test checks exactly what the maps use.

Provenance of the equations
---------------------------
Transcribed from NIST's EPQ library (DTSA-II), which is in the public domain (title 17 U.S.C.
105, see ``licenses/EPQ-NIST.txt``): ``XPP1991.java`` (XPP), ``XPP1989Ext.java`` (tilt),
``BackscatterFactor.PouchouBackscatterFactor``, ``BackscatterCoefficient.PouchouAndPichoir91``,
``SurfaceIonization.Pouchou1991``, ``StoppingPower.Pouchou1991`` with
``MeanIonizationPotential.Zeller75``, ``ProportionalIonizationCrossSection.Pouchou86``,
``Fluorescence.ReedFluorescence`` with ``IonizationDepthRatio.Reed1990`` and
``LenardCoefficient.Heinrich``.  Mass attenuation coefficients, edges and K yields come from
``tables`` (xraylib, BSD-3), not from EPQ's own MAC tables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import tables

logger = logging.getLogger(__name__)

SQRT2 = math.sqrt(2.0)
_TINY = 1.0e-6
#: below this tilt the plain (normal-incidence) XPP is used, as in EPQ
_MIN_TILT_RAD = math.radians(1.0)
#: How a tilted specimen is described (see :func:`phi_rho_z_rows`):
#: ``"scaled"`` (default) -- the tilted XPP's AREA (backscatter loss at tilt), the
#: normal-incidence CURVE compressed in depth by s = cos(tilt)**DEPTH_SCALE_EXPONENT;
#: ``"xpp_ext"`` -- the full tilt extension of Pouchou, Pichoir & Boivin as in EPQ.
#: At 70 deg the extension puts the ionisation peak 9x shallower than at normal
#: incidence. Against EPQ's Monte Carlo (NISTMonte Gen3, 20 000 electrons, 70 deg tilt,
#: 81 deg exit angle, 5 compositions x up to 9 lines, k-ratios to the pure element) the
#: extension errs by 18.0 % RMS (log), the depth-scaled curve by 4.7 % with s fitted (0.57)
#: and 4.8 % with s = sqrt(cos 70) = 0.585, that fit in a form that is 1 at normal incidence.
#: The fitted s depends on the geometry (0.37 at a 9.54 deg exit angle, where sqrt(cos) errs by
#: 18.3 %): sqrt(cos) is tuned to the 81 deg geometry the L/K ratios selected. Checked only at
#: 70 deg tilt and 81 deg exit angle; another geometry needs its own Monte Carlo check.
TILT_MODELS = ("scaled", "xpp_ext")
DEPTH_SCALE_EXPONENT = 0.5
#: A primary line more than this above the K secondary's edge is ignored (EPQ: 5 keV for K)
_FLUOR_WINDOW_K_KEV = 5.0
#: EPQ ReedFluorescence.familyFactor for a K secondary, by primary family
_FLUOR_FAMILY_FACTOR_K = {"K": 1.0, "L": 1.0 / 0.24, "M": 0.02}
#: Lowest overvoltage a quantified line may have (the same gate as eds_utils.MIN_OVERVOLTAGE):
#: ln(U) -> 0 as U -> 1, so a line just above its edge takes the whole composition
MIN_OVERVOLTAGE = 1.5


# --------------------------------------------------------------------------- element data
@lru_cache(maxsize=None)
def _z(el: str) -> int:
    return int(tables._element(el)["Z"])


@lru_cache(maxsize=None)
def _a(el: str) -> float:
    from eds_utils import ATOMIC_MASSES  # root module (same pattern as model.py)
    return float(ATOMIC_MASSES[el])


@lru_cache(maxsize=None)
def _line(el: str, fam: str) -> Tuple[float, float, float, float]:
    """(edge keV, main line keV, omega, p) from eds_utils._XRAY_LINES."""
    from eds_utils import _XRAY_LINES
    entry = _XRAY_LINES.get(fam, {}).get(el)
    if entry is None:
        raise KeyError(f"no {fam} line data for {el}")
    return tuple(float(x) for x in entry)  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _shell_constants(fam: str) -> Tuple[float, float]:
    from eds_utils import _SHELL_B, _SHELL_ELECTRONS
    return float(_SHELL_ELECTRONS[fam]), float(_SHELL_B[fam])


def _mean_ionisation_kev(el: str) -> float:
    """Zeller 1975: J = (10.04 + 8.25 exp(-Z/11.22)) Z eV."""
    z = _z(el)
    return (10.04 + 8.25 * math.exp(-z / 11.22)) * z * 1.0e-3


@lru_cache(maxsize=None)
def _mu(el: str, energy_kev: float) -> float:
    return float(tables.mu_rho(el, np.array([energy_kev]))[0])


def mu_line(comp: Mapping[str, float], energy_kev: float) -> float:
    """Mass attenuation coefficient (cm^2/g) of a mass-fraction composition at one energy."""
    return float(sum(w * _mu(el, energy_kev) for el, w in comp.items()))


# --------------------------------------------------------------------------- composition helpers
def normalise(comp: Mapping[str, float]) -> Dict[str, float]:
    tot = float(sum(max(v, 0.0) for v in comp.values()))
    if tot <= 0:
        raise ValueError("composition has no positive mass fraction")
    return {k: max(float(v), 0.0) / tot for k, v in comp.items()}


def icx_exponent(el: str, fam: str) -> float:
    if fam == "K":
        z = _z(el)
        return 0.86 + 0.12 * math.exp(-z * z / 25.0)
    if fam == "L":
        return 0.82
    if fam == "M":
        return 0.78
    raise ValueError(fam)


def icx(el: str, fam: str, e0: float) -> float:
    """Pouchou 1986 proportional cross-section ln(U)/(Ec^2 U^m)."""
    ec = _line(el, fam)[0]
    u = e0 / ec
    if u <= 1.0:
        return 0.0
    return math.log(u) / (ec * ec * u ** icx_exponent(el, fam))


# --------------------------------------------------------------------------- the element basis
class Basis:
    """Per-element constants of one element list, for vectorised evaluation.

    Compositions are arrays ``C`` of shape (n, m), mass fractions, columns in ``elements`` order.
    """

    def __init__(self, elements: Sequence[str]):
        self.elements: List[str] = list(elements)
        self.index = {e: i for i, e in enumerate(self.elements)}
        self.Z = np.array([_z(e) for e in self.elements], float)
        self.A = np.array([_a(e) for e in self.elements], float)
        self.sqrtZ = np.sqrt(self.Z)
        self.ZoA = self.Z / self.A
        self.ZoA_lnJ = self.ZoA * np.log([_mean_ionisation_kev(e) for e in self.elements])

    def mu_vec(self, energy_kev: float) -> np.ndarray:
        return np.array([_mu(e, energy_kev) for e in self.elements], float)


def _as_rows(C) -> np.ndarray:
    C = np.asarray(C, float)
    return C[None, :] if C.ndim == 1 else C


def _normalise_rows(C: np.ndarray) -> np.ndarray:
    C = np.maximum(C, 0.0)
    s = C.sum(axis=1, keepdims=True)
    s[s <= 0] = np.inf
    return C / s


# --------------------------------------------------------------------------- phi(rho z)
@dataclass(frozen=True)
class PhiRhoZ:
    """XPP curve phi(rz) = A e^{-a rz} + (B rz + phi0 - A) e^{-b rz}, rz in g/cm^2 (arrays)."""
    phi0: np.ndarray
    big_a: np.ndarray
    big_b: np.ndarray
    a: np.ndarray
    b: np.ndarray
    eps: np.ndarray
    area: np.ndarray          # F, g/cm^2

    def curve(self, rz) -> np.ndarray:
        rz = np.asarray(rz, float)
        return (self.big_a * np.exp(-self.a * rz)
                + ((self.big_b * rz + self.phi0) - self.big_a) * np.exp(-self.b * rz))

    def emitted(self, chi) -> np.ndarray:
        """int phi(rz) exp(-chi rz) d rz  (XPP1991.computeZACorrection), g/cm^2."""
        b, eps = self.b, self.eps
        return (((self.phi0 + self.big_b / (b + chi))
                 - self.big_a * b * eps / (b * (1.0 + eps) + chi)) / (b + chi))


class _MatrixState:
    """Composition-only quantities shared by every line of one composition array."""

    def __init__(self, basis: "Basis", C: np.ndarray, tilt_rad: float):
        self.C = C
        self.zb = (C @ basis.sqrtZ) ** 2                       # PAP mean Z
        self.zbar = C @ basis.Z                                # weight-averaged Z
        self.eta = 1.75e-3 * self.zb + 0.37 * (1.0 - np.exp(-0.015 * self.zb ** 1.3))
        self.big_m = C @ basis.ZoA
        self.j = np.exp((C @ basis.ZoA_lnJ) / self.big_m)
        self.w0 = 0.595 + self.eta / 3.7 + self.eta ** 4.55
        self.tilted = tilt_rad > _MIN_TILT_RAD
        self.cos_b = math.cos(tilt_rad)
        if self.tilted:
            cb = self.cos_b
            self.eta_t = self.eta ** (cb ** (1.14 - 0.4 * (1.0 - np.exp(-self.zb / 25.0))))
            self.w_t = self.w0 ** (cb ** (0.69 - 0.21 * (1.0 - np.exp(-self.zb / 17.0))))
            self.h_t = 0.2 + 2.3 / np.sqrt(self.zbar)
            self.uu_t = 8.0e-3 * np.exp(-(self.zbar ** 0.3))
        self.x = 1.0 + 1.3 * np.log(self.zb)
        self.y = 0.2 + self.zb / 200.0
        self.log1y = np.log(1.0 + self.y)
        self.log4zb = np.log(4.0 * self.zb)
        self.mu_cache: Dict[float, np.ndarray] = {}

    def mu(self, basis: "Basis", energy_kev: float) -> np.ndarray:
        v = self.mu_cache.get(energy_kev)
        if v is None:
            v = self.C @ basis.mu_vec(energy_kev)
            self.mu_cache[energy_kev] = v
        return v


def phi_rho_z_rows(basis: Basis, C, el: str, fam: str, e0: float,
                   tilt_rad: float = 0.0, state: Optional[_MatrixState] = None,
                   tilt_model: str = "scaled") -> PhiRhoZ:
    """XPP phi(rho z) of the ``fam`` shell of ``el`` for every composition row of ``C``.

    ``tilt_rad`` is the angle between the beam and the surface normal (0 = normal incidence).
    ``tilt_model`` (:data:`TILT_MODELS`) chooses how a tilt enters.
    ``state`` (internal) carries the composition-only terms when several lines share ``C``.
    """
    if tilt_model not in TILT_MODELS:
        raise ValueError(f"tilt_model must be one of {TILT_MODELS}")
    if state is None:
        state = _MatrixState(basis, _normalise_rows(_as_rows(C)), tilt_rad)
    st = state
    ec = _line(el, fam)[0]
    u0 = e0 / ec
    if u0 <= 1.0:
        raise ValueError(f"{el} {fam}: edge {ec} keV not excited at {e0} kV")
    m = icx_exponent(el, fam)
    q0 = icx(el, fam, e0)
    zb, eta, j = st.zb, st.eta, st.j
    # 1/S, Pouchou & Pichoir 1991 (StoppingPower.Pouchou1991.computeInv)
    v0 = e0 / j
    vu = v0 / u0
    lnu = math.log(u0)
    inv_s = np.zeros_like(j)
    for dk, pk in ((6.6e-6, 0.78), (1.12e-5 * (1.35 - 0.45 * j * j), 0.1), (2.2e-6 / j, 0.25 * (j - 2.0))):
        tk = 1.0 + pk - m
        ut = u0 ** tk
        inv_s = inv_s + dk * vu ** pk * (tk * ut * lnu - ut + 1.0) / (tk * tk)
    inv_s = (u0 / (v0 * st.big_m)) * inv_s
    ju = 1.0 + u0 * (lnu - 1.0)

    def backscatter_factor(eta_, w_):
        q = (2.0 * w_ - 1.0) / (1.0 - w_)
        gu = (u0 - 1.0 - (1.0 - u0 ** (-(1.0 + q))) / (1.0 + q)) / ((2.0 + q) * ju)
        return 1.0 - eta_ * w_ * (1.0 - gu)

    phi0 = 1.0 + 3.3 * (1.0 - u0 ** (-(2.0 - 2.3 * eta))) * eta ** 1.2
    # the curve's shape: tilted (xpp_ext) or normal incidence (scaled, compressed below)
    tilted = st.tilted and tilt_model == "xpp_ext"
    cos_b = st.cos_b
    f_area_tilt = None
    if st.tilted and tilt_model == "scaled":
        f_area_tilt = cos_b * backscatter_factor(st.eta_t, st.w_t) * inv_s / q0
    if not tilted:
        f_area = backscatter_factor(eta, st.w0) * inv_s / q0
    else:
        f_area = cos_b * backscatter_factor(st.eta_t, st.w_t) * inv_s / q0
        qq = 1.0 + st.h_t * (1.0 - math.exp(-((u0 - 1.0) ** 0.3)))
        phi0 = qq * (phi0 / qq) ** (cos_b ** 0.7)
    # mean depth of ionisation
    rbar = f_area / (1.0 + st.x * np.log(1.0 + st.y * (1.0 - u0 ** -0.42)) / st.log1y)
    rbar = np.where(f_area / rbar < phi0, f_area / phi0, rbar)
    # initial slope
    g = 0.22 * st.log4zb * (1.0 - 2.0 * np.exp(zb * (1.0 - u0) / 15.0))
    hh = 1.0 - 10.0 * (1.0 - 1.0 / (1.0 + u0 / 10.0)) / (zb * zb)
    gh4 = g * hh ** 4
    b = SQRT2 * (1.0 + np.sqrt(np.maximum(1.0 - rbar * phi0 / f_area, 0.0))) / (rbar * (cos_b if tilted else 1.0))
    limit = 0.9 * b * rbar * rbar * (b - 2.0 * phi0 / f_area)
    gh4 = np.minimum(gh4, limit)
    p = gh4 * f_area / (rbar * rbar)
    if tilted:
        t_deg = math.degrees(tilt_rad)
        p = p * ((1.0 + st.uu_t * t_deg ** 1.7) - 27.0 * math.exp(-(90.0 - t_deg) / 7.0))
    a = (p + b * (2.0 * phi0 - b * f_area)) / (b * f_area * (2.0 - b * rbar) - phi0)
    eps = (a - b) / b
    if tilted:
        small = eps < _TINY
        eps = np.where(small, _TINY, eps)
    else:
        small = np.abs(eps) < _TINY
        eps = np.where(small, np.where(eps < 0, -_TINY, _TINY), eps)
    a = np.where(small, b * (1.0 + eps), a)
    big_b = (b * b * f_area * (1.0 + eps) - p - phi0 * b * (2.0 + eps)) / eps
    big_a = ((big_b / b + phi0) - b * f_area) * (1.0 + eps) / eps
    if f_area_tilt is not None:
        # phi_t(rz) = (F_t / F_0) / s * phi_0(rz / s), s = cos(tilt)^0.5: area F_t, depth x s
        k = f_area_tilt / f_area
        sc = cos_b ** DEPTH_SCALE_EXPONENT
        return PhiRhoZ(phi0=phi0 * k / sc, big_a=big_a * k / sc, big_b=big_b * k / (sc * sc),
                       a=a / sc, b=b / sc, eps=eps, area=f_area_tilt)
    return PhiRhoZ(phi0=phi0, big_a=big_a, big_b=big_b, a=a, b=b, eps=eps, area=f_area)


def phi_rho_z(comp: Mapping[str, float], el: str, fam: str, e0: float,
              tilt_rad: float = 0.0, tilt_model: str = "scaled") -> PhiRhoZ:
    """Scalar convenience wrapper: ``comp`` a {element: mass fraction} mapping."""
    comp = normalise(comp)
    if el not in comp:
        raise ValueError(f"{el} not in composition")
    basis = Basis(list(comp))
    return phi_rho_z_rows(basis, [comp[e] for e in basis.elements], el, fam, e0, tilt_rad,
                          tilt_model=tilt_model)


# --------------------------------------------------------------------------- fluorescence (Reed)
@lru_cache(maxsize=None)
def _jump_fraction(el: str, fam: str) -> float:
    """(r - 1)/r for the absorption edge of the ``fam`` shell, from the attenuation table."""
    ec = _line(el, fam)[0]
    lo = _mu(el, ec * 0.995)
    hi = _mu(el, ec * 1.005)
    r = hi / lo if lo > 0 else 1.0
    return (r - 1.0) / r if r > 1.0 else 0.0


def _family_lines_norm(el: str, fam: str):
    lines = tables.family_lines(el, fam)
    tot = sum(w for _e, w in lines)
    return [(e, w / tot) for e, w in lines] if tot > 0 else []


@lru_cache(maxsize=None)
def _primary_line(el_b: str, ec_a: float, e0: float):
    """EPQ ``Fluorescence.primaryExcitingLine`` + the family weight of ``compute``.

    Among ``el_b``'s lines above the secondary edge ``ec_a`` take the highest family (L before
    K, M before L) and in it the strongest line; ``w`` sums that family's normalised weights of
    the lines above the edge. Returns ``(family, energy, w)`` or ``None``.
    """
    best = None
    for fam in ("K", "L", "M"):
        lines = [(e, w) for e, w in _family_lines_norm(el_b, fam) if e > ec_a]
        if not lines:
            continue
        try:
            edge_b = _line(el_b, fam)[0]
        except KeyError:
            continue
        if edge_b >= e0:
            continue
        rep_e = max(lines, key=lambda ew: ew[1])[0]
        best = (fam, rep_e, sum(w for _e, w in lines))
    return best


def fluorescence_rows(basis: Basis, C, el: str, fam: str, e0: float, psi_rad: float,
                      state: Optional[_MatrixState] = None) -> np.ndarray:
    """1 + Reed characteristic fluorescence of ``el``'s ``fam`` line, per composition row.

    EPQ's ``Fluorescence.compute`` with ``ReedFluorescence``: one representative primary line
    per other element (:func:`_primary_line`), ignored more than 5 keV above the secondary edge,
    weighted with its family's summed weight. Our attenuation coefficients and jump ratios come
    from ``tables`` (EPQ: Heinrich), so the term differs from EPQ's by a few percent of itself
    (``tests/test_eds_matrix_correction.py`` pins it against EPQ).

    K secondaries only. For L (and M) secondaries Reed's formula with EPQ's family factor
    (1/0.24) made our k-ratios WORSE than no fluorescence term at all on the Pouchou & Pichoir
    k-ratio database (Cu L in Al +39 %; measured during development with a variant that has
    since been removed).
    """
    if state is None:
        state = _MatrixState(basis, _normalise_rows(_as_rows(C)), 0.0)
    C = state.C
    out = np.ones(C.shape[0])
    if fam != "K":
        return out
    ec_a, e_sec, _om, _p = _line(el, fam)
    u_a = e0 / ec_a
    if u_a <= 1.0:
        return out
    mu_sec = state.mu(basis, e_sec)
    den_a = u_a * math.log(u_a) - u_a + 1.0
    lenard = 4.5e5 / (e0 ** 1.65 - ec_a ** 1.65)
    jump = _jump_fraction(el, fam)
    for el_b in basis.elements:
        if el_b == el:
            continue
        prim = _primary_line(el_b, ec_a, e0)
        if prim is None:
            continue
        pfam, e_prim, w = prim
        if e_prim >= ec_a + _FLUOR_WINDOW_K_KEV:
            continue
        ec_b, _eb, omega_b, _pb = _line(el_b, pfam)
        c_b = C[:, basis.index[el_b]]
        u_b = e0 / ec_b
        ss = (u_b * math.log(u_b) - u_b + 1.0) / den_a
        const = (_FLUOR_FAMILY_FACTOR_K[pfam] * 0.5 * jump * omega_b
                 * (_a(el) / _a(el_b)) * ss)
        mu_b_a = _mu(el, e_prim)
        mu_b = state.mu(basis, e_prim)
        u = mu_sec / (math.sin(psi_rad) * mu_b)
        v = lenard / mu_b
        out = out + w * const * c_b * (mu_b_a / mu_b) * (np.log1p(u) / u + np.log1p(v) / v)
    return out


def fluorescence_factor(comp: Mapping[str, float], el: str, fam: str, e0: float,
                        psi_rad: float) -> float:
    comp = normalise(comp)
    basis = Basis(list(comp))
    return float(fluorescence_rows(basis, [comp[e] for e in basis.elements], el, fam, e0, psi_rad)[0])


# --------------------------------------------------------------------------- geometry
def take_off_angle(elevation_deg: float, azimuth_deg: float, tilt_deg: float,
                   convention: str = "toward", tilt_axis_deg: float = 0.0) -> float:
    """Exit (take-off) angle above the specimen surface, radians.

    Detector direction ``d = (cos el cos az, cos el sin az, sin el)`` (z up, towards the column).
    ``convention="toward"`` (default): the surface normal leans towards azimuth
    ``tilt_axis_deg``, so with tilt axis 0 an azimuth of 0 means the specimen is tilted
    towards the detector (the reading exspy documents for its azimuth, per research note R1;
    AZtec's own definition was not found). Elevation 29, azimuth 0, tilt 70: 81.0 deg.
    ``convention="axis"``: the normal leans perpendicular to the tilt axis direction, azimuth
    0 = along the tilt axis (``sin psi = sin(el) cos(tilt)``, 9.54 deg for the same header).
    How azimuth and tilt axis combine is verified for neither convention when either is
    non-zero (every reference file has both 0).
    Evidence for "toward" on the reference instrument: EPQ Monte Carlo predicts the measured
    Cu/Fe/Zn L/K ratios of 8 regions to 4.5 % RMS at 81 deg against 8.2 % at 9.54 deg; a
    second test of all lines with a free response per line does not discriminate (14.4 % vs
    14.7 %): the free per-line responses absorb the geometry.
    """
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    t = math.radians(tilt_deg)
    ax = math.radians(tilt_axis_deg)
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    lean = ax + (math.pi / 2.0 if convention == "axis" else 0.0)
    n = np.array([math.sin(t) * math.cos(lean), math.sin(t) * math.sin(lean), math.cos(t)])
    return float(math.asin(float(np.clip(n @ d, -1.0, 1.0))))


# --------------------------------------------------------------------------- intensities
@dataclass(frozen=True)
class Geometry:
    beam_kv: float
    psi_rad: float       # take-off (exit) angle above the specimen surface
    tilt_rad: float      # beam to surface-normal angle
    #: why this geometry is less certain than the reference instrument's ("" when it is not)
    note: str = ""

    def __post_init__(self):
        if not (0.0 <= self.tilt_rad < math.pi / 2):
            raise ValueError(f"specimen tilt {math.degrees(self.tilt_rad):.2f} deg is outside [0, 90)")
        if not (0.0 < self.psi_rad <= math.pi / 2):
            raise ValueError(f"exit angle {math.degrees(self.psi_rad):.2f} deg is outside (0, 90]")
        if not self.beam_kv > 0:
            raise ValueError(f"beam voltage {self.beam_kv} kV")

    @classmethod
    def from_degrees(cls, beam_kv: float, psi_deg: float, tilt_deg: float) -> "Geometry":
        return cls(float(beam_kv), math.radians(psi_deg), math.radians(tilt_deg))

    @property
    def psi_deg(self) -> float:
        return math.degrees(self.psi_rad)

    @property
    def tilt_deg(self) -> float:
        return math.degrees(self.tilt_rad)


def line_constant(el: str, fam: str) -> float:
    """K_i = omega * p * n * b / A (per unit mass, arbitrary common scale)."""
    _ec, _e, omega, p = _line(el, fam)
    n, b = _shell_constants(fam)
    return omega * p * n * b / _a(el)


def emitted_rows(basis: Basis, C, el: str, fam: str, geo: Geometry,
                 fluorescence: bool = True, state: Optional[_MatrixState] = None) -> np.ndarray:
    """Q(E0) * ZA(C) * Fl(C) per composition row (emitted intensity per unit mass fraction)."""
    if state is None:
        state = _MatrixState(basis, _normalise_rows(_as_rows(C)), geo.tilt_rad)
    prz = phi_rho_z_rows(basis, state.C, el, fam, geo.beam_kv, geo.tilt_rad, state)
    chi = state.mu(basis, _line(el, fam)[1]) / math.sin(geo.psi_rad)
    za = prz.emitted(chi)
    if fluorescence:
        za = za * fluorescence_rows(basis, state.C, el, fam, geo.beam_kv, geo.psi_rad, state)
    return icx(el, fam, geo.beam_kv) * za


def emitted_per_mass(comp: Mapping[str, float], el: str, fam: str, geo: Geometry,
                     fluorescence: bool = True) -> float:
    """Scalar wrapper of :func:`emitted_rows`."""
    comp = normalise(comp)
    if el not in comp:
        comp = dict(comp)
        comp[el] = 0.0
    basis = Basis(list(comp))
    return float(emitted_rows(basis, [comp[e] for e in basis.elements], el, fam, geo, fluorescence)[0])


def predicted_intensities(comp: Mapping[str, float], lines: Iterable[Tuple[str, str]],
                          geo: Geometry, response: Optional[Mapping[Tuple[str, str], float]] = None,
                          fluorescence: bool = True) -> Dict[Tuple[str, str], float]:
    """Relative detected intensity of each line for a known composition (G = 1)."""
    comp = normalise(comp)
    out = {}
    for el, fam in lines:
        if comp.get(el, 0.0) <= 0:
            out[(el, fam)] = 0.0
            continue
        r = 1.0 if response is None else float(response[(el, fam)])
        out[(el, fam)] = (r * line_constant(el, fam) * comp[el]
                          * emitted_per_mass(comp, el, fam, geo, fluorescence))
    return out


def quantify_map(intensities: Mapping[str, Tuple[str, np.ndarray]], geo: Geometry,
                 response: Mapping[Tuple[str, str], float], fluorescence: bool = True,
                 max_iter: int = 50, tol: float = 1.0e-6,
                 stats: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], int]:
    """Standardless, normalised mass fractions for many spectra at once.

    ``intensities``: {element: (family, array of net intensities)}, all arrays the same shape.
    Fixed point per pixel: C_e proportional to I_e / (R K Q ZA(C) Fl(C)), normalised to 1.
    Negative intensities enter as zero, non-finite ones too (counted); a pixel with no positive
    intensity returns zeros. A line below :data:`MIN_OVERVOLTAGE` raises ``ValueError``.
    ``stats`` (optional dict) receives ``converged`` (bool), ``iterations``, ``max_change``
    (last largest change of a mass fraction) and ``non_finite`` (values set to 0).
    Returns ({element: mass-fraction array}, iterations).
    """
    elements = list(intensities)
    if not elements:
        if stats is not None:
            stats.update(converged=True, iterations=0, max_change=0.0, non_finite=0)
        return {}, 0
    shape = np.shape(intensities[elements[0]][1])
    fams = [intensities[e][0] for e in elements]
    for e, f in zip(elements, fams):
        u0 = geo.beam_kv / _line(e, f)[0]
        if u0 < MIN_OVERVOLTAGE:
            raise ValueError(f"{e} {f}: overvoltage {u0:.2f} at {geo.beam_kv:g} kV is below "
                             f"{MIN_OVERVOLTAGE}; the line cannot be quantified")
    raw = np.stack([np.asarray(intensities[e][1], float).ravel() for e in elements], axis=1)
    finite = np.isfinite(raw)
    non_finite = int(raw.size - np.count_nonzero(finite))
    I = np.maximum(np.where(finite, raw, 0.0), 0.0)
    basis = Basis(elements)
    pre = np.array([float(response[(e, f)]) * line_constant(e, f) for e, f in zip(elements, fams)])
    q0 = np.array([max(icx(e, f, geo.beam_kv), 1e-30) for e, f in zip(elements, fams)])
    alive = I.sum(axis=1) > 0
    C = _normalise_rows(I / (pre * q0))
    it = 0
    if alive.any():
        Ca = C[alive]
        Ia = I[alive]
        delta = 0.0
        for it in range(1, max_iter + 1):
            st = _MatrixState(basis, Ca, geo.tilt_rad)
            denom = np.stack([emitted_rows(basis, Ca, e, f, geo, fluorescence, st)
                              for e, f in zip(elements, fams)], axis=1) * pre
            new = _normalise_rows(Ia / denom)
            delta = float(np.max(np.abs(new - Ca))) if new.size else 0.0
            Ca = new
            if delta < tol:
                break
        C[alive] = Ca
        converged = delta < tol
    else:
        delta, converged = 0.0, True
    C[~alive] = 0.0
    if stats is not None:
        stats.update(converged=bool(converged), iterations=int(it), max_change=float(delta),
                     non_finite=non_finite)
    return {e: C[:, i].reshape(shape) for i, e in enumerate(elements)}, it


def quantify(intensities: Mapping[Tuple[str, str], float], geo: Geometry,
             response: Mapping[Tuple[str, str], float], fluorescence: bool = True,
             max_iter: int = 100, tol: float = 1.0e-7) -> Tuple[Dict[str, float], int]:
    """One spectrum: {(element, family): net intensity} -> ({element: mass fraction}, iterations)."""
    by_el = {}
    for (el, fam), v in intensities.items():
        if el in by_el:
            raise ValueError(f"two lines given for {el}")
        by_el[el] = (fam, np.array([float(v)]))
    comp, it = quantify_map(by_el, geo, response, fluorescence, max_iter, tol)
    return {e: float(v[0]) for e, v in comp.items()}, it


def mass_to_atomic(comp: Mapping[str, float]) -> Dict[str, float]:
    mol = {el: c / _a(el) for el, c in comp.items()}
    tot = sum(mol.values())
    return {el: m / tot for el, m in mol.items()} if tot > 0 else dict(mol)


def atomic_to_mass(comp: Mapping[str, float]) -> Dict[str, float]:
    mass = {el: c * _a(el) for el, c in comp.items()}
    tot = sum(mass.values())
    return {el: m / tot for el, m in mass.items()}


# --------------------------------------------------------------------------- geometry of a file
def geometry_from_meta(meta) -> Optional[Geometry]:
    """The quantification geometry of a spectrum's acquisition, or ``None``.

    ``meta`` is a :class:`backend.eds_spectrum.model.SpectrumMeta`.  Needs tilt and detector
    elevation from the file header; azimuth and tilt axis default to 0.  The exit angle uses the
    reading of AZtec's azimuth the stored calibration was made with (``geometry_convention``,
    "toward": elevation 29, azimuth 0, tilt 70 give 81.0 deg; see :func:`take_off_angle`).
    ``None`` -- and so the uncorrected curve, which says why in its provenance -- for a header
    without tilt or elevation, a tilt outside [0, 89.5] deg (no sign convention for a negative
    tilt is known), or a detector shadowed by the specimen (exit angle <= 1 deg). A non-zero
    azimuth or tilt axis gives a geometry with a ``note``: that combination is unverified.
    """
    tilt = getattr(meta, "tilt_deg", None)
    elev = getattr(meta, "elevation_deg", None)
    if tilt is None or elev is None:
        return None
    tilt = float(tilt)
    if not (0.0 <= tilt <= 89.5):
        logger.warning("EDS matrix correction: specimen tilt %.2f deg in the header is outside "
                       "[0, 89.5]; quantifying without matrix correction", tilt)
        return None
    az = float(getattr(meta, "azimuth_deg", None) or 0.0)
    axis = float(getattr(meta, "tilt_axis_deg", None) or 0.0)
    convention = stored_calibration().get("geometry_convention", "toward")
    psi = take_off_angle(float(elev), az, tilt, convention, axis)
    if psi <= math.radians(1.0):
        return None          # detector shadowed by the specimen: nothing to correct with
    note = ""
    if abs(az) > 1e-6 or abs(axis) > 1e-6:
        note = (f"detector azimuth {az:g} deg / tilt axis {axis:g} deg: the exit angle "
                f"({math.degrees(psi):.1f} deg) uses a convention verified only for azimuth 0 "
                "and tilt axis 0")
    return Geometry(float(meta.beam_kv), psi, math.radians(tilt), note)


# --------------------------------------------------------------------------- instrument response
#: fallback family for lines whose family has no calibrated factor (M lines -> L factor)
_FAMILY_FALLBACK = {"M": "L"}


@dataclass(frozen=True)
class MatrixResponse:
    """R(line) = eps(E) * exp(family factor + per-line deviation).

    ``eps`` is the detector-window transmission of :func:`backend.eds_spectrum.model.det_eff`;
    the family factor carries what the physical constants get wrong for a whole shell family;
    the per-line deviation exists only for lines with a reference (``anchored``), every other
    line is physics + family factor.
    """
    family_log: Mapping[str, float]
    delta: Mapping[Tuple[str, str], float]
    prior_sigma: float
    label: str = ""

    @property
    def anchored(self) -> List[Tuple[str, str]]:
        return sorted(self.delta)

    def family_for(self, fam: str) -> str:
        return fam if fam in self.family_log else _FAMILY_FALLBACK.get(fam, "K")

    def __call__(self, el: str, fam: str) -> float:
        from .model import det_eff
        eps = float(det_eff(np.array([_line(el, fam)[1]]))[0])
        return eps * math.exp(self.family_log[self.family_for(fam)] + self.delta.get((el, fam), 0.0))

    def to_json(self) -> dict:
        return {"family_log": dict(self.family_log),
                "delta": {f"{e} {f}": v for (e, f), v in sorted(self.delta.items())},
                "prior_sigma": self.prior_sigma, "label": self.label}

    @classmethod
    def from_json(cls, d: Mapping) -> "MatrixResponse":
        return cls(family_log={k: float(v) for k, v in d["family_log"].items()},
                   delta={tuple(k.split()): float(v) for k, v in d["delta"].items()},
                   prior_sigma=float(d["prior_sigma"]), label=str(d.get("label", "")))


def reference_sigma(at_pct: float) -> float:
    """Relative uncertainty given to a reference composition: 5 % relative (Oxford's claim for
    Tru-Q above 1 wt%) combined with a 0.3 at% absolute floor for minor elements."""
    return math.sqrt(0.05 ** 2 + (0.3 / max(float(at_pct), 1e-3)) ** 2)


def fit_response(references: Sequence[Mapping], anchors: Sequence[Tuple[str, str]],
                 prior_sigma: float = 0.10, label: str = "") -> Tuple[MatrixResponse, float, int]:
    """Weighted least squares for the response, in logs, in two stages.

    ``references``: dicts with ``nets`` ({"Fe K": net cps}), ``ref_at_pct`` ({"Fe": at%}) and
    ``geometry`` (:class:`Geometry`).  Model: log I = log G_region + family_log[fam]
    + delta[line] + log(eps K c Q ZA Fl) with each delta pulled to 0 with ``prior_sigma``.
    Stage 1 fits the K lines (what the quantification uses) together with the region scales
    G; stage 2 fits the other families with G held, so the less well modelled L lines cannot
    pull the K calibration.  Returns (response, chi2 of the K reference rows, their dof).
    """
    k_anchors = [l for l in anchors if l[1] == "K"]
    resp_k, chi2, dof, g = _fit_stage(references, k_anchors, prior_sigma, None)
    other = [l for l in anchors if l[1] != "K"]
    fam_log, delta = dict(resp_k.family_log), dict(resp_k.delta)
    if other:
        resp_o, _c, _d, _g = _fit_stage(references, other, prior_sigma, g)
        fam_log.update(resp_o.family_log)
        delta.update(resp_o.delta)
    return (MatrixResponse(family_log=fam_log, delta=delta, prior_sigma=float(prior_sigma),
                           label=label), chi2, dof)


def _fit_stage(references, anchors, prior_sigma, fixed_g):
    """One least-squares stage of :func:`fit_response`; ``fixed_g`` holds the region scales."""
    from .model import det_eff
    regions = list(range(len(references)))
    fams = sorted({f for _e, f in anchors})
    lid = {l: i for i, l in enumerate(anchors)}
    nr, nf, nl = len(regions), len(fams), len(anchors)
    npar = nr + nf + nl
    rows, b, w = [], [], []
    for r, ref in enumerate(references):
        at = {k: float(v) for k, v in ref["ref_at_pct"].items() if float(v) > 0}
        comp = atomic_to_mass(at)
        geo = ref["geometry"]
        for line in anchors:
            el, fam = line
            if el not in comp:
                continue
            i_meas = float(ref["nets"].get(f"{el} {fam}", 0.0))
            if i_meas <= 0:
                continue
            pred = (line_constant(el, fam) * comp[el] * emitted_per_mass(comp, el, fam, geo)
                    * float(det_eff(np.array([_line(el, fam)[1]]))[0]))
            row = np.zeros(npar)
            row[r] = 1.0
            row[nr + fams.index(fam)] = 1.0
            row[nr + nf + lid[line]] = 1.0
            rows.append(row); b.append(math.log(i_meas) - math.log(pred))
            w.append(1.0 / reference_sigma(at[el]))
    n_obs = len(rows)
    for line in anchors:
        row = np.zeros(npar); row[nr + nf + lid[line]] = 1.0
        rows.append(row); b.append(0.0); w.append(1.0 / prior_sigma)
    if fixed_g is None:
        gauge = np.zeros(npar); gauge[0] = 1.0
        rows.append(gauge); b.append(0.0); w.append(1e3)
    else:
        for r in regions:
            row = np.zeros(npar); row[r] = 1.0
            rows.append(row); b.append(float(fixed_g[r])); w.append(1e3)
    A, bb, ww = np.array(rows), np.array(b), np.array(w)
    x, *_ = np.linalg.lstsq(A * ww[:, None], bb * ww, rcond=None)
    res = (A @ x - bb) * ww
    chi2 = float(np.sum(res[:n_obs] ** 2))
    resp = MatrixResponse(family_log={f: float(x[nr + i]) for i, f in enumerate(fams)},
                          delta={l: float(x[nr + nf + lid[l]]) for l in anchors},
                          prior_sigma=float(prior_sigma))
    dof = n_obs - (nr - 1 if fixed_g is None else 0) - nf
    return resp, chi2, dof, [float(v) for v in x[:nr]]


# --------------------------------------------------------------------------- stored calibration
_CALIBRATION_FILE = "matrix_calibration.json"


@lru_cache(maxsize=1)
def stored_calibration() -> dict:
    """``matrix_calibration.json``: anchors, references and the fitted response."""
    import json
    from pathlib import Path
    with open(Path(__file__).with_name(_CALIBRATION_FILE), encoding="utf-8") as fh:
        return json.load(fh)


def stored_response() -> MatrixResponse:
    return MatrixResponse.from_json(stored_calibration()["response"])
