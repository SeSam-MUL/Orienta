"""SP2 — derive Monte-Carlo target composition (Z, A, rho) from CrystalStructure.

The GPU MC requires three scalar material properties for the trajectory equations:
* ``mean_Z``  — occupancy-weighted mean atomic number
* ``mean_A``  — occupancy-weighted mean atomic weight in g/mol
* ``rho``     — mass density in g/cm^3

All three are derived from a :class:`~backend.forward_sim.crystal.xtal_io.CrystalStructure`
(read via :func:`~backend.forward_sim.crystal.xtal_io.read_crystal_structure`).  The
atom basis is first expanded to the full conventional cell using
:func:`~backend.forward_sim.crystal.structure_matrix._expand_symmetry_orbit` (same
expansion used by the master builder), then occupancy-weighted.

Density is computed from the formula-unit mass and the unit-cell volume::

    rho = (sum_i occ_i * A_Zi) * amu / V_cell_cm3
        = (sum_i occ_i * A_Zi) / (N_A * V_cell_cm3)

where ``V_cell_cm3`` is the direct-cell volume in cm^3 (from
:func:`~backend.forward_sim.crystal.structure_matrix._cell_volume_nm3` scaled by
``1e-21`` since 1 nm^3 = 1e-21 cm^3) and ``N_A`` is Avogadro's number.

NOTE — mean-Z/A/rho Monte Carlo IS EMsoft's own method (not an approximation):
    The MC reduces the target to a single occupancy-weighted mean Z, mean A and
    density (rho).  This is EXACTLY what EMsoft's ``CalcDensity`` computes for
    both its CPU (``EMMC.f90``) and GPU (``EMMCOpenCL.f90``) Monte Carlo — the
    same scalar averaging feeding the same screened-Rutherford + Joy-Luo
    single-scattering model (constants verified term-for-term, 2026-06-26).  So
    for multi-element compounds we are methodologically EQUIVALENT to EMsoft, not
    "approximate": the per-element crystallography (scattering factors, occupancy,
    Debye-Waller) enters the SEPARATE dynamical master-pattern stage, NOT the MC,
    and the MC only supplies the energy/depth distribution (mean-Z-dominated).
    The only genuine caution is heavily DISORDERED cells (low site occupancy),
    which EMsoft's dynamical theory also struggles with; those should be filtered
    upstream (see the disorder warning in the changelog).

Atomic-weight table:
    A periodic-table subset covering EBSD-relevant elements (Z 1-96), sourced
    from IUPAC/CODATA 2021 standard atomic weights.  Only elements with
    tabulated values are included; an unrecognised Z raises :class:`ValueError`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from backend.forward_sim.crystal.structure_matrix import (
    _cell_volume_nm3,
    _expand_symmetry_orbit,
)
from backend.forward_sim.crystal.xtal_io import CrystalStructure

# Avogadro's number (mol^-1)
_AVOGADRO = 6.02214076e23
# 1 nm^3 in cm^3
_NM3_TO_CM3 = 1.0e-21

# IUPAC/CODATA 2021 standard atomic weights [g/mol], indexed by Z.
# Elements covered: H(1) .. Cm(96).  Only entries actually needed for EBSD
# targets are required; missing entries raise ValueError.
# fmt: off
_ATOMIC_WEIGHT: dict[int, float] = {
    1:  1.008,    # H
    2:  4.0026,   # He
    3:  6.941,    # Li
    4:  9.0122,   # Be
    5:  10.811,   # B
    6:  12.011,   # C
    7:  14.007,   # N
    8:  15.999,   # O
    9:  18.998,   # F
    10: 20.180,   # Ne
    11: 22.990,   # Na
    12: 24.305,   # Mg
    13: 26.982,   # Al
    14: 28.086,   # Si
    15: 30.974,   # P
    16: 32.060,   # S
    17: 35.450,   # Cl
    18: 39.948,   # Ar
    19: 39.098,   # K
    20: 40.078,   # Ca
    21: 44.956,   # Sc
    22: 47.867,   # Ti
    23: 50.942,   # V
    24: 51.996,   # Cr
    25: 54.938,   # Mn
    26: 55.845,   # Fe
    27: 58.933,   # Co
    28: 58.693,   # Ni  <-- gate material
    29: 63.546,   # Cu
    30: 65.380,   # Zn
    31: 69.723,   # Ga
    32: 72.630,   # Ge
    33: 74.922,   # As
    34: 78.971,   # Se
    35: 79.904,   # Br
    36: 83.798,   # Kr
    37: 85.468,   # Rb
    38: 87.620,   # Sr
    39: 88.906,   # Y
    40: 91.224,   # Zr
    41: 92.906,   # Nb
    42: 95.950,   # Mo
    43: 98.000,   # Tc (no stable isotope; use common value)
    44: 101.07,   # Ru
    45: 102.91,   # Rh
    46: 106.42,   # Pd
    47: 107.87,   # Ag
    48: 112.41,   # Cd
    49: 114.82,   # In
    50: 118.71,   # Sn
    51: 121.76,   # Sb
    52: 127.60,   # Te
    53: 126.90,   # I
    54: 131.29,   # Xe
    55: 132.91,   # Cs
    56: 137.33,   # Ba
    57: 138.91,   # La
    58: 140.12,   # Ce
    59: 140.91,   # Pr
    60: 144.24,   # Nd
    61: 145.00,   # Pm (radioactive; use 145)
    62: 150.36,   # Sm
    63: 151.96,   # Eu
    64: 157.25,   # Gd
    65: 158.93,   # Tb
    66: 162.50,   # Dy
    67: 164.93,   # Ho
    68: 167.26,   # Er
    69: 168.93,   # Tm
    70: 173.05,   # Yb
    71: 174.97,   # Lu
    72: 178.49,   # Hf
    73: 180.95,   # Ta
    74: 183.84,   # W
    75: 186.21,   # Re
    76: 190.23,   # Os
    77: 192.22,   # Ir
    78: 195.08,   # Pt
    79: 196.97,   # Au
    80: 200.59,   # Hg
    81: 204.38,   # Tl
    82: 207.20,   # Pb
    83: 208.98,   # Bi
    90: 232.04,   # Th
    92: 238.03,   # U
    96: 247.00,   # Cm
}
# fmt: on


def _atomic_weight(Z: int) -> float:
    """Return the standard atomic weight for element Z.

    Args:
        Z: atomic number.

    Returns:
        Atomic weight in g/mol.

    Raises:
        ValueError: if Z is not in the table.
    """
    try:
        return _ATOMIC_WEIGHT[Z]
    except KeyError:
        raise ValueError(
            f"Atomic weight for Z={Z} not found in the IUPAC table; "
            "add the element to backend/forward_sim/mc/composition.py::_ATOMIC_WEIGHT"
        )


@dataclass(frozen=True)
class MCComposition:
    """Effective scalar material properties for the Joy-1995 MC.

    Attributes:
        mean_Z: occupancy-weighted mean atomic number.
        mean_A: occupancy-weighted mean atomic weight in g/mol.
        rho: mass density in g/cm^3.
        n_distinct_elements: number of distinct atomic species in the cell.
    """

    mean_Z: float
    mean_A: float
    rho: float
    n_distinct_elements: int


def mc_composition_from_structure(structure: CrystalStructure) -> MCComposition:
    """Derive Joy-MC effective composition from a :class:`CrystalStructure`.

    Expands the asymmetric unit to the full conventional cell, computes
    occupancy-weighted mean Z and mean A, then derives density from the
    formula-unit mass and the direct-cell volume.

    Args:
        structure: crystal structure from
            :func:`~backend.forward_sim.crystal.xtal_io.read_crystal_structure`.

    Returns:
        :class:`MCComposition` with ``(mean_Z, mean_A, rho)`` ready for the
        GPU MC.

    Raises:
        ValueError: if any atom's Z is not in the atomic-weight table, or if
            the expansion yields no atoms, or if any site occupancy exceeds 1
            (physically invalid).  Sites with occ <= 0 are silently skipped.
            Low occupancy values (0 < occ < 0.5) are accepted and weighted
            proportionally — no fail-loud guard is applied.

    Warns:
        UserWarning: when the target has more than one distinct element
            (single-scattering approximation only exact for mono-elemental).
    """
    expanded = _expand_symmetry_orbit(structure)
    if not expanded:
        raise ValueError("_expand_symmetry_orbit returned an empty list; bad structure?")

    # Validate occupancies and collect per-site (Z, occ, A).
    z_sum = 0.0
    a_sum = 0.0
    occ_total = 0.0
    distinct_z: set[int] = set()

    for Z, occ, _B, _r in expanded:
        if occ <= 0.0:
            continue  # zero-occupancy sites contribute nothing
        if occ > 1.0 + 1e-6:
            raise ValueError(
                f"Site occupancy {occ} > 1 for Z={Z}; structure invalid"
            )
        A_z = _atomic_weight(Z)
        z_sum += occ * Z
        a_sum += occ * A_z
        occ_total += occ
        distinct_z.add(Z)

    if occ_total <= 0.0:
        raise ValueError("Total occupancy is zero; no atoms in expanded cell")

    mean_Z = z_sum / occ_total
    mean_A = a_sum / occ_total

    # Density: rho = (sum occ_i * A_Zi) / (N_A * V_cell_cm3)
    # a_sum is the total formula-unit mass in (occ-scaled) g/mol units.
    # For a full conventional cell, a_sum / N_A is the cell mass in grams.
    V_cell_nm3 = _cell_volume_nm3(structure)
    V_cell_cm3 = V_cell_nm3 * _NM3_TO_CM3
    # cell mass = a_sum [g/mol] / N_A [atoms/mol]  (a_sum already sums over all atoms)
    rho = a_sum / (_AVOGADRO * V_cell_cm3)

    n_distinct = len(distinct_z)
    if n_distinct > 1:
        warnings.warn(
            f"MCComposition: target has {n_distinct} distinct elements "
            f"({sorted(distinct_z)}). The MC uses an occupancy-weighted mean "
            "Z/A/rho — this is EXACTLY EMsoft's CalcDensity approach "
            "(EMMC.f90 / EMMCOpenCL.f90), NOT an approximation relative to "
            "EMsoft. The per-element crystallography enters the separate "
            "dynamical master stage. Only heavily disordered (low-occupancy) "
            "cells warrant caution.",
            UserWarning,
            stacklevel=2,
        )

    return MCComposition(
        mean_Z=mean_Z,
        mean_A=mean_A,
        rho=rho,
        n_distinct_elements=n_distinct,
    )
