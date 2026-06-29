"""SP0 — read EMsoft ``CrystalData`` into a :class:`CrystalStructure`.

The reader works against either an EMsoft ``.xtal`` file or an EMsoft master
``.h5`` (both expose the same ``CrystalData`` HDF5 group layout).

Confirmed layout (verified against the real Ni/Al oracle ``.h5`` files,
2026-06-10):

* ``CrystalData/AtomData`` — shape ``(5, N)`` float, rows ``[x, y, z, occ, B]``
  (fractional coordinates, site occupancy, Debye-Waller B factor in nm**2 —
  EMsoft's AtomData convention).
* ``CrystalData/Atomtypes`` — shape ``(N,)`` int, atomic numbers.
* ``CrystalData/LatticeParameters`` — shape ``(6,)`` ``[a, b, c, alpha, beta,
  gamma]`` with ``a, b, c`` in nm and angles in degrees.
* ``CrystalData/SpaceGroupNumber`` — int (scalar or length-1 array).
* ``CrystalData/CrystalSystem`` — int (scalar or length-1 array).
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

# The 92 centrosymmetric space groups (those whose point group is one of the 11
# Laue / centrosymmetric crystal classes — i.e. the space group contains the
# inversion centre ``-1``).  Authoritative list keyed by international space-group
# number (1..230); used to decide whether the southern Lambert hemisphere can be
# derived from the northern one by inversion (NH ⟂ SH symmetry).  Reference: the
# International Tables for Crystallography, Vol. A — the space groups belonging to
# the centrosymmetric point groups -1, 2/m, mmm, 4/m, 4/mmm, -3, -3m, 6/m, 6/mmm,
# m-3, m-3m.
_CENTROSYMMETRIC_SPACE_GROUPS = frozenset(
    {
        2,                                                      # -1
        10, 11, 12, 13, 14, 15,                                 # 2/m
        47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
        73, 74,                                                 # mmm
        83, 84, 85, 86, 87, 88,                                 # 4/m
        123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
        134, 135, 136, 137, 138, 139, 140, 141, 142,           # 4/mmm
        147, 148,                                               # -3
        162, 163, 164, 165, 166, 167,                           # -3m
        175, 176,                                               # 6/m
        191, 192, 193, 194,                                     # 6/mmm
        200, 201, 202, 203, 204, 205, 206,                      # m-3
        221, 222, 223, 224, 225, 226, 227, 228, 229, 230,       # m-3m
    }
)


def is_centrosymmetric_space_group(space_group: int) -> bool:
    """True if international space-group number ``space_group`` is centrosymmetric.

    A centrosymmetric space group contains the inversion centre ``-1`` (its point
    group is one of the 11 Laue classes).  For such a crystal the EBSD master is
    inversion-symmetric (``I(k) == I(-k)``), so the southern Lambert hemisphere is
    fully determined by the northern one (no independent build is needed).

    Args:
        space_group: international space-group number (1..230).

    Returns:
        ``True`` iff the space group is one of the 92 centrosymmetric groups.

    Raises:
        ValueError: if ``space_group`` is outside ``[1, 230]``.
    """
    sg = int(space_group)
    if not 1 <= sg <= 230:
        raise ValueError(f"space_group {sg} out of range [1, 230]")
    return sg in _CENTROSYMMETRIC_SPACE_GROUPS


@dataclass(frozen=True)
class Atom:
    """A single atom site read from ``CrystalData/AtomData``."""

    Z: int  # atomic number
    xyz: tuple  # fractional coordinates (x, y, z)
    occ: float  # site occupancy
    B: float  # Debye-Waller B factor (nm**2, EMsoft AtomData convention)


@dataclass(frozen=True)
class CrystalStructure:
    """An EMsoft crystal structure (lattice + atom basis + space group)."""

    lattice: tuple  # (a, b, c, alpha, beta, gamma); a,b,c in nm, angles in deg
    atoms: list  # list[Atom]
    space_group: int
    crystal_system: int

    @property
    def is_centrosymmetric(self) -> bool:
        """True if the crystal's space group contains an inversion centre.

        For a centrosymmetric crystal the EBSD master obeys ``I(k) == I(-k)``, so
        the southern Lambert hemisphere is fully determined by the northern one
        (see :func:`is_centrosymmetric_space_group`).
        """
        return is_centrosymmetric_space_group(self.space_group)

    @property
    def reciprocal_metric(self) -> np.ndarray:
        """Reciprocal metric tensor ``g*`` (3x3, float64).

        Built as the inverse of the direct metric tensor ``g`` formed from the
        lattice parameters.  For a reciprocal-lattice vector with Miller indices
        ``(h, k, l)`` the squared length is ``[h k l] g* [h k l]^T`` (units
        1/nm**2), so ``1/d_hkl = sqrt([h k l] g* [h k l]^T)``.
        """
        a, b, c, alpha, beta, gamma = self.lattice
        ca = np.cos(np.deg2rad(alpha))
        cb = np.cos(np.deg2rad(beta))
        cg = np.cos(np.deg2rad(gamma))
        # Direct metric tensor g_ij = a_i . a_j.
        g = np.array(
            [
                [a * a, a * b * cg, a * c * cb],
                [a * b * cg, b * b, b * c * ca],
                [a * c * cb, b * c * ca, c * c],
            ],
            dtype=np.float64,
        )
        # Reciprocal metric tensor is the inverse of the direct metric tensor.
        return np.linalg.inv(g)

    @property
    def structure_matrix(self) -> np.ndarray:
        """Direct structure matrix ``dsm`` (3x3, float64) — EMsoft ``CalcMatrices``.

        Columns of ``dsm`` are the crystallographic basis vectors ``a, b, c``
        expressed in the standard IUCr Cartesian setting ("``a`` along x, ``c*``
        toward z"): ``a`` lies along Cartesian x, ``b`` is in the x-y plane, and the
        third Cartesian axis is along ``c*`` (EMsoft ``crystal.f90::CalcMatrices``,
        eq. 1.64).  For ``(a, b, c, α, β, γ)`` with ``ca = cos α`` etc.,
        ``sg = sin γ`` and ``vol = a·b·c·√(1 − ca² − cb² − cg² + 2·ca·cb·cg)``::

            dsm = [[a,  b·cg,  c·cb               ],
                   [0,  b·sg, -c·(cb·cg − ca)/sg  ],
                   [0,  0,     vol/(a·b·sg)        ]]

        It maps a **Cartesian** vector ``t`` to **reciprocal-fractional** (Miller)
        coordinates via EMsoft's ``TransSpace('c'→'r')`` = ``dsmᵀ · t`` — the recipe
        used by the master-pattern direction map.  The defining identities hold for
        every crystal system: ``dsmᵀ·dsm == g`` (direct metric) and
        ``dsm·rsmᵀ == I`` with ``rsm = (dsm⁻¹)ᵀ`` the reciprocal structure matrix
        (so ``rsmᵀ·rsm == g*`` = :pyattr:`reciprocal_metric`).  For a **cubic** cell
        ``dsm == a·I``, so ``dsmᵀ·t == a·t`` — exactly the scalar cubic map.

        Raises:
            ValueError: if ``sin γ`` is (near-)zero (degenerate cell) or the cell
                volume term is non-positive (an impossible lattice).
        """
        a, b, c, alpha, beta, gamma = self.lattice
        ca = np.cos(np.deg2rad(alpha))
        cb = np.cos(np.deg2rad(beta))
        cg = np.cos(np.deg2rad(gamma))
        sg = np.sin(np.deg2rad(gamma))
        # EMsoft FatalError guard: gamma must not be 0 / 180 deg (sin gamma in the
        # denominators).  Fail loud rather than silently returning inf/nan.
        if abs(sg) < 1e-12:
            raise ValueError(
                f"degenerate lattice: sin(gamma) ~ 0 (gamma={gamma}); structure "
                "matrix is undefined"
            )
        vol_term = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
        if vol_term <= 0.0:
            raise ValueError(
                f"impossible lattice angles (a,b,c,al,be,ga)={self.lattice}: cell "
                f"volume term {vol_term!r} <= 0"
            )
        vol = a * b * c * np.sqrt(vol_term)
        dsm = np.array(
            [
                [a, b * cg, c * cb],
                [0.0, b * sg, -c * (cb * cg - ca) / sg],
                [0.0, 0.0, vol / (a * b * sg)],
            ],
            dtype=np.float64,
        )
        return dsm


def _scalar_int(value) -> int:
    """Coerce an HDF5 scalar / length-1 array to a Python int."""
    arr = np.asarray(value).ravel()
    return int(arr[0])


def read_crystal_structure(path: str) -> CrystalStructure:
    """Read ``CrystalData/*`` from an EMsoft ``.xtal`` or master ``.h5``.

    ``AtomData`` is ``(5, N)`` with rows ``[x, y, z, occ, B]``; ``Atomtypes`` is
    ``(N,)`` atomic numbers.

    Raises:
        ValueError: if the ``CrystalData`` group is missing or ``AtomData`` is
            not shaped ``(5, N)``.
    """
    with h5py.File(path, "r") as f:
        if "CrystalData" not in f:
            raise ValueError(f"CrystalData group missing in {path!r}")
        cd = f["CrystalData"]

        atom_data = np.asarray(cd["AtomData"][()])
        if atom_data.ndim != 2 or atom_data.shape[0] != 5:
            raise ValueError(
                "CrystalData/AtomData must have shape (5, N) "
                f"[rows x,y,z,occ,B], got {atom_data.shape}"
            )

        atomtypes = np.asarray(cd["Atomtypes"][()]).ravel()
        n_atoms = atom_data.shape[1]
        if atomtypes.shape[0] != n_atoms:
            raise ValueError(
                f"Atomtypes length {atomtypes.shape[0]} does not match "
                f"AtomData atom count {n_atoms}"
            )

        lattice_raw = np.asarray(cd["LatticeParameters"][()]).ravel()
        if lattice_raw.shape[0] != 6:
            raise ValueError(
                "CrystalData/LatticeParameters must have 6 entries "
                f"[a,b,c,alpha,beta,gamma], got {lattice_raw.shape[0]}"
            )
        lattice = tuple(float(v) for v in lattice_raw)

        space_group = _scalar_int(cd["SpaceGroupNumber"][()])
        crystal_system = _scalar_int(cd["CrystalSystem"][()])

    atoms = [
        Atom(
            Z=int(atomtypes[i]),
            xyz=(
                float(atom_data[0, i]),
                float(atom_data[1, i]),
                float(atom_data[2, i]),
            ),
            occ=float(atom_data[3, i]),
            B=float(atom_data[4, i]),
        )
        for i in range(n_atoms)
    ]

    return CrystalStructure(
        lattice=lattice,
        atoms=atoms,
        space_group=space_group,
        crystal_system=crystal_system,
    )
