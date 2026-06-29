"""SP2 — elastic scattering physics for the EBSD Monte Carlo.

Two elastic cross-section models are provided, selectable via the
:class:`ElasticModel` enum:

``ElasticModel.JOY`` (default — EMsoft-faithful)
    Screened-Rutherford as in Joy 1995 ch.3, ported verbatim from ebsdtorch
    ``simulation/monte_carlo.py`` (Z. Varley, MIT).  This is the model used by
    EMsoft's own GPU kernel (EMMC.cl L224-228) and the CPU kernel (EMMC.f90
    L490-493).  A whole-tree grep of the EMsoft source returned **zero** hits
    for Mott/Browning/Czyzewski/ELSEPA — EMsoft is pure screened-Rutherford.
    MFP(Ni, 20 keV) ≈ 5.17 nm.

``ElasticModel.BROWNING`` (selectable — true Mott-fitted analytic)
    Browning et al. 1994 empirical total elastic cross-section fitted to
    Czyzewski relativistic Hartree-Fock Mott tables (J. Appl. Phys. 76(4):2016).

    Total cross-section (E in keV, sigma in cm²/atom, bare — NOT Avogadro-grouped)::

        sigma_el = 3.0e-18 · Z^1.7 / (E + 0.005·Z^1.7·E^0.5 + 0.0007·Z²/E^0.5)

    MFP uses the standard number-density path::

        N = 6.022e23 · rho / A    [atoms/cm³]
        mfp = 1e7 / (N · sigma)   [nm]

    This gives MFP(Ni, 20 keV) ≈ 3.36 nm  (vs Joy 5.17 nm — 1.54× shorter).

    Angular sampling (two-component screened-Rutherford + isotropic mix, see
    Browning 1995 Scanning 17:250; open description in arXiv:1712.01104 and
    CASINO Drouin 1997):

    * Anisotropic fraction: ``f = 1 − 0.9 / Z^0.5``
    * Anisotropic screening: ``alpha_B = 7.0e-3 · E^{-0.5}`` (energy-only,
      not Z-dependent — distinct from Joy's ``3.4e-3·Z^{2/3}/E``).
    * Draw R1, R2, R3 ∈ (0,1):
      - if R1 ≤ f: ``cos φ = 1 − 2·alpha_B·R2 / (1 + alpha_B − R2)``  (Rutherford CDF)
      - else:      ``cos φ = 1 − 2·R2``                                (isotropic)
    * ``ψ = 2π·R3``

    NOTE: Since EMsoft itself uses screened-Rutherford, switching to Browning
    makes our MC **diverge** from EMsoft rather than converge to it.  Browning
    is provided for absolute-fidelity experiments vs true Mott DCS.  The
    NCC=0.68 plateau is NOT an elastic-model issue (see commit 5ccb583 note on
    missing i-factor in build_A).

    Validity: Z ≤ 92, E = 0.1–30 keV.  Fit accuracy ~5–15% at high angle/high Z;
    use ELSEPA/Czyzewski tables if sub-5% accuracy is needed.

UNITS (shared between models):
* E in keV
* A (atomic weight) in g/mol
* rho in g/cm^3
* sigma_E in cm^2/atom
* mfp / step / depth in nm (1e7 factor converts cm → nm)
* dE/ds in keV per (g/cm²) — multiply by step_nm * rho to get dE per step
* angles in radians internally

References:
    Joy, D.C. 1995, *Monte Carlo Modeling for Electron Microscopy and
    Microanalysis*, Oxford University Press, ch.3.

    Browning, R. et al. 1994, *Empirical forms for electron/atom elastic
    scattering cross sections in the range 0.1–30 keV*,
    J. Appl. Phys. 76(4):2016.

    Browning, R. 1995, *Low-energy electron/atom elastic scattering cross
    sections from 0.1–30 keV*, Scanning 17:250.

    Drouin, D. et al. 1997, *CASINO Part II: Tabulated values of the Mott cross
    section*, Scanning 19:20.

    arXiv:1712.01104 — open implementation confirming Browning two-component
    sampling.

    ebsdtorch (Z. Varley, MIT):
    https://github.com/ZacharyVarley/ebsdtorch (simulation/monte_carlo.py)

    EMsoft GPU MC kernel (screened-Rutherford, NOT Mott):
    EMsoft/opencl/EMMC.cl L224-228.
"""
from __future__ import annotations

import math
from enum import Enum


# ---------------------------------------------------------------------------
# Elastic model selector
# ---------------------------------------------------------------------------

class ElasticModel(str, Enum):
    """Elastic cross-section model for the GPU Monte Carlo.

    Attributes:
        JOY: Screened-Rutherford (Joy 1995) — EMsoft-faithful default.
            MFP(Ni, 20 keV) ≈ 5.17 nm.
        BROWNING: Browning 1994 analytic Mott fit.
            MFP(Ni, 20 keV) ≈ 3.36 nm.
    """

    JOY = "joy"
    BROWNING = "browning"


def mean_ionisation_potential_kev(Z: float) -> float:
    """Berger-Seltzer / Joy mean ionisation potential in keV.

    J = (9.76·Z + 58.5·Z^{-0.19}) × 10^{-3}

    The classic expression gives eV; the ``1e-3`` converts to keV to match
    the energy unit used throughout the MC (E in keV).

    Args:
        Z: atomic number (may be fractional for mean-Z targets).

    Returns:
        J in keV.

    The constant 0.9911 inside the Bethe log (used in :func:`bethe_loss_rate_kev_per_gcm2`
    and in :func:`run_gpu_mc`) is the Joy-Luo value as encoded in ebsdtorch and EMsoft's
    OpenCL kernel — NOT the 0.85 value from Joy's book (1995, eq 3.21 text).  The
    ebsdtorch source uses ``0.9911`` verbatim; this module matches that choice.

    Examples:
        >>> round(mean_ionisation_potential_kev(13), 4)   # Al
        0.1628
        >>> round(mean_ionisation_potential_kev(28), 4)   # Ni
        0.3043
    """
    return (9.76 * Z + 58.5 * Z ** (-0.19)) * 1.0e-3


def screening_alpha(Z: float, E_keV: float) -> float:
    """Screening parameter alpha for the screened-Rutherford cross-section.

    alpha = 3.4e-3 · Z^{2/3} / E   (Joy 1995 eq 3.2, dimensionless)

    Args:
        Z: atomic number (or mean Z).
        E_keV: current electron energy in keV.

    Returns:
        alpha (dimensionless, > 0).

    Raises:
        ValueError: if E_keV <= 0 (electron has no energy).
    """
    if E_keV <= 0.0:
        raise ValueError(f"E_keV must be positive, got {E_keV}")
    return 3.4e-3 * (Z ** (2.0 / 3.0)) / E_keV


def screened_rutherford_xsection_cm2(Z: float, E_keV: float) -> float:
    """Screened-Rutherford total elastic cross-section (Joy 1995 eqs 3.1 & 3.3).

    sigma_E = 5.21 · 602.2 · (Z/E)^2 · (4π / (alpha·(1+alpha)))
              · ((511 + E) / (1024 + E))^2

    Constants:
        5.21   — Joy's constant (1e-21 cm^2 units, absorbed into the formula)
        602.2  — Avogadro-grouping that keeps mfp = 1e7·A/(rho·sigma_E) in nm
        511    — electron rest-mass energy in keV
        1024   = 2·511 (relativistic correction denominator)

    Args:
        Z: atomic number (or mean Z).
        E_keV: current electron energy in keV.

    Returns:
        Total elastic cross-section in cm^2/atom.

    Raises:
        ValueError: if E_keV <= 0.
    """
    if E_keV <= 0.0:
        raise ValueError(f"E_keV must be positive, got {E_keV}")
    alpha = screening_alpha(Z, E_keV)
    return (
        5.21
        * 602.2
        * (Z / E_keV) ** 2
        * (4.0 * math.pi / (alpha * (1.0 + alpha)))
        * ((511.0 + E_keV) / (1024.0 + E_keV)) ** 2
    )


def elastic_mfp_nm(Z: float, A: float, rho: float, E_keV: float) -> float:
    """Elastic mean free path in nm (Joy 1995 eq 3.3 + cm->nm conversion).

    mfp = 1e7 · A / (rho · sigma_E)

    The factor ``1e7`` converts from cm to nm (1 cm = 1e7 nm).

    Args:
        Z: atomic number (or mean Z).
        A: atomic weight in g/mol.
        rho: density in g/cm^3.
        E_keV: current electron energy in keV.

    Returns:
        Elastic mean free path in nm.

    Raises:
        ValueError: if E_keV <= 0, rho <= 0, or A <= 0.
    """
    if rho <= 0.0:
        raise ValueError(f"rho must be positive, got {rho}")
    if A <= 0.0:
        raise ValueError(f"A must be positive, got {A}")
    sigma = screened_rutherford_xsection_cm2(Z, E_keV)
    return 1.0e7 * A / (rho * sigma)


def bethe_loss_rate_kev_per_gcm2(Z: float, A: float, E_keV: float, J_keV: float) -> float:
    """Joy-Luo modified Bethe energy-loss rate dE/ds (Joy 1995 eq 3.21).

    dE/ds = -0.00785 · (Z / (A · E)) · ln(1.166 · E / J + 0.9911)

    The ``+0.9911`` inside the log is the Joy-Luo low-energy modification that
    keeps dE/ds finite and negative as E approaches J (prevents ``ln(0)``).

    Units: keV per (g/cm^2).  The energy loss over a step is then::

        dE = step_nm * rho * (dE/ds)        (step_nm in nm, rho in g/cm^3)

    since step_nm * rho has units of nm·g/cm^3 = 1e-7 g/cm^2 — BUT NOTE: the
    ``1e-7`` is *not* applied explicitly here, matching the ebsdtorch source
    (``simulation/monte_carlo.py`` line ``energies += step_nm * rho * de_ds``
    where step_nm is in nm, rho in g/cm^3, de_ds in keV/(g/cm^2)).  The result
    is approximately keV/step because nm·g/cm^3 ≃ g/cm^2 × 1e-7 and 0.00785
    already encodes the Bethe constant in Joy's mixed-unit scheme.

    Args:
        Z: atomic number (or mean Z).
        A: atomic weight in g/mol.
        E_keV: current electron energy in keV.
        J_keV: mean ionisation potential in keV (see
            :func:`mean_ionisation_potential_kev`).

    Returns:
        dE/ds in keV per (g/cm^2).  Always negative (energy loss).

    Raises:
        ValueError: if E_keV <= 0 or A <= 0.
    """
    if E_keV <= 0.0:
        raise ValueError(f"E_keV must be positive, got {E_keV}")
    if A <= 0.0:
        raise ValueError(f"A must be positive, got {A}")
    return -0.00785 * (Z / (A * E_keV)) * math.log(1.166 * E_keV / J_keV + 0.9911)


def sample_polar_angle_rad(alpha: float, u: float) -> float:
    """Inverse-CDF polar scattering angle from screened-Rutherford (Joy eqs 3.10-3.11).

    phi = arccos(1 - 2·alpha·U / (1 + alpha - U))

    Args:
        alpha: screening parameter from :func:`screening_alpha`.
        u: uniform random variate in (0, 1).

    Returns:
        Polar scattering angle in radians in [0, pi].
    """
    cos_phi = 1.0 - 2.0 * alpha * u / (1.0 + alpha - u)
    cos_phi = max(-1.0, min(1.0, cos_phi))  # numerical guard
    return math.acos(cos_phi)


# ---------------------------------------------------------------------------
# Browning 1994 elastic cross-section model
# ---------------------------------------------------------------------------

def browning_xsection_cm2(Z: float, E_keV: float) -> float:
    """Browning 1994 total elastic cross-section (bare cm^2/atom, NOT Avogadro-grouped).

    sigma_el = 3.0e-18 · Z^1.7 / (E + 0.005·Z^1.7·E^0.5 + 0.0007·Z²/E^0.5)

    Fitted to Czyzewski relativistic Hartree-Fock Mott tables.
    Valid for Z ≤ 92 and E = 0.1–30 keV; accuracy ~5–15% at high angle/high Z.

    NOTE: This returns a **bare** cm^2/atom value (not the 5.21*602.2-grouped
    quantity used by :func:`screened_rutherford_xsection_cm2`).  Use
    :func:`browning_mfp_nm` which applies the correct N=6.022e23*rho/A path.

    Args:
        Z: atomic number (or mean Z).  Must satisfy 1 ≤ Z ≤ 92.
        E_keV: current electron energy in keV.  Must be in [0.1, 30].

    Returns:
        Total elastic cross-section in cm^2/atom (bare, Avogadro-ungrouped).

    Raises:
        ValueError: if E_keV <= 0.

    Example (Ni at 20 keV): browning_xsection_cm2(28, 20.0) ~= 3.26e-17 cm^2
    (matches Browning 1994 Table values within ~5%).
    """
    if E_keV <= 0.0:
        raise ValueError(f"E_keV must be positive, got {E_keV}")
    z17 = Z ** 1.7
    e_sqrt = E_keV ** 0.5
    denom = E_keV + 0.005 * z17 * e_sqrt + 0.0007 * Z * Z / e_sqrt
    return 3.0e-18 * z17 / denom


def browning_mfp_nm(Z: float, A: float, rho: float, E_keV: float) -> float:
    """Elastic mean free path in nm using the Browning 1994 cross-section.

    Uses the number-density path (Option A from the research):
        N = 6.022e23 · rho / A   [atoms/cm³]
        mfp = 1e7 / (N · sigma)  [nm]

    This is NOT interchangeable with :func:`elastic_mfp_nm` (Joy) because the
    Browning sigma is a bare cm^2 value while Joy's
    :func:`screened_rutherford_xsection_cm2` returns the 5.21*602.2-grouped
    quantity.

    Reference values (Browning 1994, numerically verified):
        Ni (Z=28, A=58.693, rho=8.908) at 20 keV → MFP ≈ 3.36 nm
        Al (Z=13, A=26.982, rho=2.702) at 20 keV → MFP ≈ 15.4 nm

    Args:
        Z: atomic number (or mean Z).
        A: atomic weight in g/mol.
        rho: density in g/cm^3.
        E_keV: current electron energy in keV.

    Returns:
        Elastic mean free path in nm.

    Raises:
        ValueError: if E_keV <= 0, rho <= 0, or A <= 0.
    """
    if rho <= 0.0:
        raise ValueError(f"rho must be positive, got {rho}")
    if A <= 0.0:
        raise ValueError(f"A must be positive, got {A}")
    sigma = browning_xsection_cm2(Z, E_keV)
    N_atoms_per_cm3 = 6.022e23 * rho / A  # atoms/cm³
    return 1.0e7 / (N_atoms_per_cm3 * sigma)  # nm


def browning_screening_alpha(E_keV: float) -> float:
    """Browning angular-sampling screening parameter (energy-only, no Z).

    alpha_B = 7.0e-3 · E^{-0.5}

    Unlike Joy's Z-dependent alpha = 3.4e-3·Z^{2/3}/E, Browning's parameter
    depends only on energy.  Used exclusively in :func:`browning_sample_polar_angle_rad`.

    Args:
        E_keV: current electron energy in keV.

    Returns:
        Dimensionless screening parameter alpha_B.

    Raises:
        ValueError: if E_keV <= 0.
    """
    if E_keV <= 0.0:
        raise ValueError(f"E_keV must be positive, got {E_keV}")
    return 7.0e-3 / (E_keV ** 0.5)


def browning_anisotropic_fraction(Z: float) -> float:
    """Browning 1994 anisotropic mixing fraction (Z-dependent).

    f = 1 − 0.9 / Z^0.5

    Controls the weight of the forward-peaked (anisotropic/Rutherford) component
    vs the isotropic component in :func:`browning_sample_polar_angle_rad`.
    For large Z the distribution is nearly all anisotropic (f→1); for small Z
    (Z≈1) the isotropic component dominates.

    Args:
        Z: atomic number.  Must be ≥ 1.

    Returns:
        Mixing fraction f ∈ [0, 1] (clamped to [0, 1] for numerical safety).
    """
    f = 1.0 - 0.9 / (Z ** 0.5)
    return max(0.0, min(1.0, f))


def browning_sample_polar_angle_rad(Z: float, E_keV: float, r1: float, r2: float) -> float:
    """Two-component Browning 1994 polar scattering angle sampler.

    Draws a polar angle from the Browning two-component model:
    * Anisotropic (forward-peaked) branch — Rutherford inverse CDF with
      ``alpha_B = 7.0e-3·E^{-0.5}``:
      ``cos φ = 1 − 2·alpha_B·r2 / (1 + alpha_B − r2)``
    * Isotropic branch:
      ``cos φ = 1 − 2·r2``

    The branch is selected by comparing r1 against f = 1 − 0.9/Z^0.5.
    When r1 ≤ f the anisotropic (forward-peaked) branch is used.

    Args:
        Z: atomic number (or mean Z).
        E_keV: current electron energy in keV.
        r1: uniform variate in (0, 1) — branch selector.
        r2: uniform variate in (0, 1) — angle sampler.

    Returns:
        Polar scattering angle in radians in [0, pi].

    Raises:
        ValueError: if E_keV <= 0.
    """
    if E_keV <= 0.0:
        raise ValueError(f"E_keV must be positive, got {E_keV}")
    f = browning_anisotropic_fraction(Z)
    alpha_B = browning_screening_alpha(E_keV)
    if r1 <= f:
        # Anisotropic (forward-peaked) branch — same Rutherford inverse-CDF
        # algebra as Joy, but with alpha_B instead of Z-dependent alpha.
        cos_phi = 1.0 - 2.0 * alpha_B * r2 / (1.0 + alpha_B - r2)
    else:
        # Isotropic branch
        cos_phi = 1.0 - 2.0 * r2
    cos_phi = max(-1.0, min(1.0, cos_phi))  # numerical guard
    return math.acos(cos_phi)
