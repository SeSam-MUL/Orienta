"""SP1 — ``Sgh`` per-atom back-scatter structure weighting (Task 10).

The EBSD master-pattern value at a direction ``k`` is
``I(k) = Σ_{g,h strong} Lgh(k)[g,h] · Sgh[g,h]`` (design §2 step 5), where
``Lgh`` is the depth-integrated dynamical intensity (Task 9) and ``Sgh`` is the
**direction-independent** structure weighting that encodes which reflection pairs
back-scatter and with what amplitude — per atom type, summed over the full
symmetry orbit.

This module implements :func:`compute_Sgh`, EMsoft's ``CalcSgh``.

EMsoft formula (``MBmodule.f90::CalcSgh``, verified verbatim 2026-06-10)
-----------------------------------------------------------------------
For each atom type ``ip`` (1 .. numset), with the full symmetry orbit
``apos(ip, 1..n, :)`` of its asymmetric-unit position::

    Znsq = Z_ip² · occ_ip                           ! cell%ATOM_pos(ip,4) = occupancy
    do ir = 1, nn:                                  ! over strong reflections (rows)
      do ic = 1, nn:                                ! over strong reflections (cols)
        kkk  = hkl_ic − hkl_ir                      ! difference vector g_c − g_r
        kkl  = 0.25 · CalcLength(kkk, 'r')²         ! |g_c − g_r|² / 4   (nm⁻²)
        DBWF = Znsq · exp(−B_ip · kkl)              ! cell%ATOM_pos(ip,5) = B (nm²)
        do ikk = 1, n:                              ! over the symmetry orbit
          arg  = 2π · kkk · apos(ip, ikk, :)
          Sgh(ir,ic,ip) += cmplx(cos arg, sin arg) · DBWF

We collapse EMsoft's per-atom-type third axis ``Sgh(:,:,ip)`` into a **summed**
``(n, n)`` matrix (single-site combining): for Ni/Al there is one atom type, and
the master builder uses the total weighting ``Σ_ip Sgh(:,:,ip)``.

Key physics notes
------------------
* **Z², not a scattering factor.**  EMsoft weights the back-scatter by the bare
  ``Z²·occ`` (the elastic atomic number squared), *not* the Weickenmeier-Kohl
  ``f(s)``.  ``Sgh`` is therefore **voltage-independent** in this v1 — the
  ``voltage_kV`` argument is kept for signature/contract parity (design §4.6) and
  a future core-loss/energy weighting, but does not enter the formula.
* **Units.**  ``CalcLength(kkk, 'r')`` is in nm⁻¹ (EMsoft lattice in nm), so
  ``kkl`` is nm⁻², and ``ATOM_pos(ip,5)`` (= :pyattr:`Atom.B`) is nm² — the
  product ``B·kkl`` is dimensionless.  We use ``structure.reciprocal_metric``
  (nm⁻²) for ``|g_c − g_r|²``, matching :func:`compute_Ug_table`.
* **Diagonal.**  At ``ir == ic`` the difference vanishes (``kkl = 0``,
  ``phase = 1``), so ``Sgh[i,i] = Σ_orbit Z²·occ = n_orbit · Z² · occ`` — real,
  positive, the per-direction self back-scatter weight.
* **Hermiticity.**  Swapping ``(ir, ic)`` sends ``kkk → −kkk``; ``kkl`` is even
  (DBWF unchanged) and the phase conjugates, so ``Sgh[ic,ir] = conj(Sgh[ir,ic])``
  — ``Sgh`` is Hermitian.

The asymmetric unit is expanded by the space-group orbit via
:func:`backend.forward_sim.crystal.structure_matrix._expand_symmetry_orbit`,
exactly as :func:`compute_Ug_table` does, so the oracle's single origin atom
becomes the 4-atom FCC cell.

Source: EMsoft ``MBmodule.f90::CalcSgh`` (+ ``getSghfromLUT`` fast path, which we
do not need: we compute on the small strong-reflection set directly).
https://github.com/EMsoft-org/EMsoft/blob/develop/Source/EMsoftLib/MBmodule.f90
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

from ..crystal.structure_matrix import _expand_symmetry_orbit, _BUILD_UG_LUT_M_THRESHOLD
from ..crystal.xtal_io import CrystalStructure

_TWOPI = 2.0 * math.pi

# Threshold above which compute_Sgh_for_build_master returns a lazy proxy
# instead of the dense (M, M) matrix.  Matched to _BUILD_UG_LUT_M_THRESHOLD
# (≈11 000) so both switches fire at the same cell-size boundary.
# Dense (M, M) complex128 = M² × 16 bytes; at M=11000 this is ~1.9 GB.
_SGH_DENSE_M_THRESHOLD: int = _BUILD_UG_LUT_M_THRESHOLD


# --------------------------------------------------------------------------- #
# Lever 2b-2 — Sgh difference-BOX gather (drop the dense (M, M) build)         #
# --------------------------------------------------------------------------- #
#
# KEY FINDING (PERF lever2bc_profile §2 Lever 2b-2): Sgh[ir,ic] depends ONLY on
# the difference vector ``d = hkl[ic] − hkl[ir]`` — exactly like U_g.  In
# ``compute_Sgh``:
#
#     diffs[ir, ic] = hkl[ic] − hkl[ir]
#     kkl  = 0.25·|diffs|²                         (reciprocal metric, nm⁻²)
#     dbwf = Z²·occ·exp(−B·kkl)                    (per atom)
#     Sgh[ir,ic] = Σ_atoms dbwf · exp(2πi·diffs·r)
#
# So Sgh is a pure function of ``d`` and can be precomputed over the SAME dense
# (D, D, D) difference box ``build_diff_lut`` uses for U (``H = 2·max|index|``):
#
#     Sgh_box(d) = Σ_atoms Z²·occ·exp(−B·0.25|d|²)·exp(2πi·d·r)
#
# Then the per-direction strong sub-block is a pure index GATHER (no (M,M) build,
# no per-direction Python atom loop).  The box build reuses the EXACT same per-atom
# expression as ``compute_Sgh`` (same orbit, same dbwf, same phase sign), so the
# gathered value is **bit-identical** to ``compute_Sgh`` for every difference (the
# difference d for a strong pair (ir,ic) always lies in the box, since both g lie in
# the reflection list whose max abs index ≤ H/2).  Proven Δ = 0.0 on Ni + tau2.


def build_sgh_box(
    structure: CrystalStructure,
    reflections: torch.Tensor,
    voltage_kV: float,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.complex128,
) -> tuple[torch.Tensor, int]:
    """Dense complex Sgh difference-box ``Sgh_box(d)`` for GPU gather (lever 2b-2).

    Builds, ONCE for the whole reflection list, a dense ``(D, D, D)`` complex table
    with ``D = 2·H + 1`` and ``H = 2·max|index|`` (the SAME box geometry as
    :func:`backend.forward_sim.dynamical.scattering_matrix.build_diff_lut`), such
    that

        sgh_box[dh + H, dk + H, dl + H]
            = Σ_atoms Z²·occ·exp(−B·0.25·|d|²)·exp(2πi·d·r)

    for every difference vector ``d = (dh, dk, dl)`` that can arise as
    ``g_ic − g_ir`` between two reflections (all lie in ``[−H, H]³``).  This is the
    direction-independent ``Sgh`` of EMsoft ``CalcSgh`` evaluated on the difference
    lattice rather than the dense ``(M, M)`` pair grid — the per-strong-block value
    is then a pure index gather (:func:`_gather_sgh_block`).

    The per-atom expression is **term-for-term identical** to :func:`compute_Sgh`
    (same orbit expansion, same ``Z²·occ·exp(−B·kkl)`` Debye-Waller weight, same
    ``exp(2πi·d·r)`` phase sign), so a gathered sub-block is **bit-identical** to a
    direct ``compute_Sgh`` on the same reflections.

    Args:
        structure: the crystal (lattice + asymmetric-unit atoms + space group);
            provides ``reciprocal_metric`` (nm⁻²) and the per-atom ``Z``/``occ``/``B``.
        reflections: ``(M, 3)`` int Miller indices (defines the box via the max abs
            index).  Index 0 is the transmitted beam ``(0, 0, 0)``.
        voltage_kV: accelerating voltage (parity with :func:`compute_Sgh`; does not
            enter the v1 formula).
        device: target device for the returned box (e.g. ``cuda``).
        dtype: box complex dtype (``complex128`` default).

    Returns:
        ``(sgh_box, H)`` where ``sgh_box`` is the ``(D, D, D)`` complex tensor on
        ``device`` and ``H`` the half-box offset (a difference ``d`` maps to
        ``sgh_box[d + H]``).
    """
    hkl = np.asarray(reflections.detach().cpu(), dtype=np.int64).reshape(-1, 3)
    a_max = int(np.abs(hkl).max()) if hkl.size else 0
    H = 2 * a_max
    D = 2 * H + 1

    gstar = np.asarray(structure.reciprocal_metric, dtype=np.float64)  # nm⁻²

    # All offset cells of the box (D³) — vectorised over the whole box at once.
    rng = np.arange(-H, H + 1, dtype=np.float64)
    dh, dk, dl = np.meshgrid(rng, rng, rng, indexing="ij")
    cells = np.stack([dh.ravel(), dk.ravel(), dl.ravel()], axis=1)  # (D³, 3)

    # kkl = 0.25·|d|²  (reciprocal-cartesian length through g*), one per cell — same
    # quantity compute_Sgh forms per (ir,ic) pair.
    kkl = 0.25 * np.einsum("mi,ij,mj->m", cells, gstar, cells)        # (D³,)

    atoms = _expand_symmetry_orbit(structure)

    sgh = np.zeros(cells.shape[0], dtype=np.complex128)
    for Z, occ, B, r in atoms:
        znsq = float(Z) * float(Z) * float(occ)
        dbwf = znsq * np.exp(-float(B) * kkl)                          # (D³,) real
        arg = _TWOPI * (cells @ np.asarray(r, dtype=np.float64))       # (D³,)
        sgh += dbwf * np.exp(1j * arg)

    box = torch.from_numpy(np.ascontiguousarray(sgh)).reshape(D, D, D)
    return box.to(device=device, dtype=dtype), H


def _gather_sgh_block(
    sgh_box: torch.Tensor, H: int, strong_hkl: torch.Tensor
) -> torch.Tensor:
    """Gather the per-direction strong ``(B, n, n)`` Sgh sub-block from the box.

    ``strong_hkl`` is ``(B, n, 3)`` int Miller indices (on ``sgh_box``'s device).
    The pair difference is ``d[b, ir, ic] = hkl[b, ic] − hkl[b, ir]`` (the
    :func:`compute_Sgh` convention: ``diffs[ir, ic] = hkl[ic] − hkl[ir]``); each
    maps to ``sgh_box[d + H]`` via a flat-index gather (pure device op).  Returns a
    complex ``(B, n, n)`` tensor — bit-identical to stacking
    ``compute_Sgh(structure, hkl[b])`` per direction.
    """
    D = sgh_box.shape[-1]
    g = strong_hkl.to(torch.long)                              # (B, n, 3)
    # diffs[b, ir, ic] = g[b, ic] − g[b, ir]  (matches compute_Sgh's diffs convention)
    diffs = g[:, None, :, :] - g[:, :, None, :]               # (B, n, n, 3)
    idx = diffs + H                                            # (B, n, n, 3)
    flat = (idx[..., 0] * D + idx[..., 1]) * D + idx[..., 2]   # (B, n, n)
    return sgh_box.reshape(-1).index_select(0, flat.reshape(-1)).reshape(flat.shape)


def compute_Sgh(
    structure: CrystalStructure,
    strong_refl: torch.Tensor,
    voltage_kV: float,
) -> torch.Tensor:
    """Per-atom back-scatter structure weighting ``Sgh`` (EMsoft ``CalcSgh``).

    For an ``(n, 3)`` list of strong reflections ``hkl`` returns the ``(n, n)``
    complex matrix::

        Sgh[ir, ic] = Σ_ip Σ_{r in orbit(ip)}
                          Z_ip² · occ_ip · exp(−B_ip · kkl) · exp(2πi·kkk·r)

    with ``kkk = hkl[ic] − hkl[ir]`` and ``kkl = 0.25·|kkk|²`` (reciprocal metric,
    nm⁻²).  The sum runs over **every** atom type ``ip`` and the full space-group
    symmetry orbit of its asymmetric-unit position (so a single oracle atom at the
    origin expands to the 4-atom FCC cell for Ni/Al).

    ``Sgh`` is **direction-independent** (it depends only on the crystal and the
    strong-reflection set) and **voltage-independent** in this v1 (EMsoft weights
    the back-scatter by the bare ``Z²·occ``, not the WK scattering factor).

    Args:
        structure: the crystal (lattice + asymmetric-unit atoms + space group);
            provides ``reciprocal_metric`` (nm⁻²) for the difference-vector length
            and the per-atom ``Z``, ``occ`` and Debye-Waller ``B`` (nm²).
        strong_refl: ``(n, 3)`` integer (or float) Miller indices of the strong
            reflections entering the dynamical matrix (index 0 is conventionally
            the transmitted beam ``g = 0``).
        voltage_kV: accelerating voltage in kV — accepted for signature/contract
            parity with the other SP1 primitives and reserved for a future
            energy-dependent weighting; it does **not** enter the v1 formula.

    Returns:
        ``(n, n)`` ``complex128`` tensor — the structure weighting ``Sgh`` summed
        over all atom types.  Hermitian for a centrosymmetric crystal, with a
        real-positive diagonal equal to ``n_orbit · Σ_ip Z_ip² · occ_ip``.

    Raises:
        ValueError: if ``strong_refl`` is not ``(n, 3)``-shaped.
    """
    if strong_refl.dim() != 2 or strong_refl.shape[-1] != 3:
        raise ValueError(
            f"strong_refl must be (n, 3), got {tuple(strong_refl.shape)}"
        )

    hkl = np.asarray(strong_refl.detach().cpu(), dtype=np.float64).reshape(-1, 3)
    n = hkl.shape[0]
    gstar = np.asarray(structure.reciprocal_metric, dtype=np.float64)  # nm⁻²

    # Difference vectors kkk = g_ic − g_ir for every ordered pair (ir, ic).
    # diffs[ir, ic, :] = hkl[ic] − hkl[ir].
    diffs = hkl[None, :, :] - hkl[:, None, :]            # (n, n, 3)
    flat = diffs.reshape(-1, 3)                          # (n·n, 3)

    # kkl = 0.25 · |kkk|²  (reciprocal-cartesian length through g*), one per pair.
    kkl = 0.25 * np.einsum("mi,ij,mj->m", flat, gstar, flat)   # (n·n,)
    kkl = kkl.reshape(n, n)

    # Expand the asymmetric unit into the full conventional-cell atom basis
    # (same orbit expansion compute_Ug_table uses); each entry is (Z, occ, B, r).
    atoms = _expand_symmetry_orbit(structure)

    sgh = torch.zeros(n, n, dtype=torch.complex128)
    for Z, occ, B, r in atoms:
        # DBWF = Z²·occ·exp(−B·kkl); B in nm², kkl in nm⁻² → dimensionless arg.
        znsq = float(Z) * float(Z) * float(occ)
        dbwf = znsq * np.exp(-float(B) * kkl)            # (n, n) real
        # Phase exp(2πi·kkk·r) for this orbit atom.
        arg = _TWOPI * (diffs @ np.asarray(r, dtype=np.float64))  # (n, n)
        term = dbwf * np.exp(1j * arg)                   # (n, n) complex
        sgh = sgh + torch.from_numpy(np.ascontiguousarray(term)).to(torch.complex128)

    return sgh


# --------------------------------------------------------------------------- #
# Large-cell lazy Sgh proxy (iter-16 productionisation)                       #
# --------------------------------------------------------------------------- #


class _LazySghMatrix:
    """Lazy proxy for the direction-independent Sgh matrix for large cells.

    The full ``(M, M)`` complex128 Sgh matrix is never materialised.  Instead,
    ``compute_Sgh`` is called on demand for the per-direction STRONG subset
    (``index_select(0, idx).index_select(1, idx)`` where the two index tensors
    are equal — the only access pattern ``build_master`` uses).

    This proxy supports exactly the interface ``build_master`` requires:
    * ``.to(device)`` — returns a proxy on that device (deferred compute)
    * ``.detach()`` — returns self (no gradient graph)
    * ``.cpu()`` / ``.to(torch.complex128)`` — chainable, return new proxy
    * ``.index_select(dim, idx)`` — records one index; when BOTH row and column
      indices are set AND equal, computes the real sub-matrix via
      ``compute_Sgh(structure, all_refl[idx], voltage_kV)`` and returns it
      (a real dense tensor on ``self._device``).

    For the serial fallback (``use_batched=False``), which calls
    ``sgh_cpu = sgh_full.detach().cpu().to(torch.complex128)`` and then
    materialises the full matrix, the proxy raises ``RuntimeError`` — large
    cells MUST use the batched lever-3 path.
    """

    def __init__(
        self,
        structure: CrystalStructure,
        all_refl: torch.Tensor,
        voltage_kV: float,
        device: torch.device,
        _row_idx: torch.Tensor | None = None,
        _col_idx: torch.Tensor | None = None,
    ) -> None:
        self._structure = structure
        self._all_refl = all_refl.detach().cpu()
        self._voltage_kV = voltage_kV
        self._device = device
        self._row_idx = _row_idx
        self._col_idx = _col_idx

    # ---- chainable device/dtype ops ----------------------------------------

    def detach(self) -> "_LazySghMatrix":
        return self

    def cpu(self) -> "_LazySghMatrix":
        return _LazySghMatrix(
            self._structure, self._all_refl, self._voltage_kV,
            torch.device("cpu"), self._row_idx, self._col_idx,
        )

    def to(self, *args, **kwargs) -> "_LazySghMatrix":
        dev = self._device
        for a in args:
            if isinstance(a, (torch.device, str)):
                dev = torch.device(a)
            elif isinstance(a, torch.dtype):
                pass  # dtype cast — no-op for a lazy proxy
        if "device" in kwargs:
            dev = torch.device(kwargs["device"])
        return _LazySghMatrix(
            self._structure, self._all_refl, self._voltage_kV,
            dev, self._row_idx, self._col_idx,
        )

    # ---- indexing ----------------------------------------------------------

    def index_select(self, dim: int, idx: torch.Tensor) -> Union["_LazySghMatrix", torch.Tensor]:
        """Record one index dimension; materialise when both dimensions are set.

        ``build_master`` always calls:
            ``sgh_full.index_select(0, sidx).index_select(1, sidx)``
        with ``sidx`` the same tensor for both dimensions.  We detect this
        (equal row == col index after both are set) and compute ``compute_Sgh``
        on the selected sub-reflection list — the result is a real dense tensor.
        """
        idx_cpu = idx.detach().cpu()
        if dim == 0:
            new_row, new_col = idx_cpu, self._col_idx
        elif dim == 1:
            new_row, new_col = self._row_idx, idx_cpu
        else:
            raise ValueError(f"_LazySghMatrix: index_select dim must be 0 or 1, got {dim}")

        if new_row is not None and new_col is not None:
            r = new_row.numpy()
            c = new_col.numpy()
            if r.shape != c.shape or not bool(np.array_equal(r, c)):
                raise RuntimeError(
                    "_LazySghMatrix: row and column index sets differ — "
                    "build_master is expected to use the same strong-reflection "
                    "index for both axes of Sgh."
                )
            # Materialise only the requested sub-block.
            sub_refl = self._all_refl[new_row]
            sgh_sub = compute_Sgh(self._structure, sub_refl, self._voltage_kV)
            return sgh_sub.to(self._device)

        return _LazySghMatrix(
            self._structure, self._all_refl, self._voltage_kV,
            self._device, new_row, new_col,
        )

    # ---- device attribute (used by build_master serial path) ---------------

    @property
    def device(self) -> torch.device:
        """Expose the target device so ``sidx.to(sgh_full.device)`` works."""
        return self._device

    # ---- full-materialise guard (serial path) -----------------------------

    def __repr__(self) -> str:
        m = self._all_refl.shape[0]
        return (
            f"_LazySghMatrix(M={m}, device={self._device}, "
            f"row_idx={'set' if self._row_idx is not None else 'none'}, "
            f"col_idx={'set' if self._col_idx is not None else 'none'})"
        )


def compute_Sgh_for_build_master(
    structure: CrystalStructure,
    all_refl: torch.Tensor,
    voltage_kV: float,
    device: torch.device,
) -> "Union[torch.Tensor, _LazySghMatrix]":
    """Return the Sgh matrix or a lazy proxy, depending on cell size.

    For ``M <= _SGH_DENSE_M_THRESHOLD`` (small cells — Ni/Al/most phases):
    computes and returns the dense ``(M, M)`` complex128 tensor on ``device``
    — **bit-identical** to the previous ``compute_Sgh(…).to(dev)`` call.

    For ``M > _SGH_DENSE_M_THRESHOLD`` (large cells — T-phase etc.):
    returns a :class:`_LazySghMatrix` proxy that computes only the per-direction
    strong subset on demand via ``compute_Sgh``, never allocating the full
    ``(M, M)`` matrix.

    Args:
        structure: the crystal (for ``compute_Sgh``).
        all_refl: ``(M, 3)`` full reflection list (transmitted beam at index 0).
        voltage_kV: accelerating voltage (passed through to ``compute_Sgh``).
        device: target device for the dense matrix (large-cell proxy stores
            the device and uses it when materialising each sub-block).

    Returns:
        Dense ``(M, M)`` complex128 tensor on ``device`` (small M) or a
        :class:`_LazySghMatrix` proxy (large M).
    """
    m = all_refl.shape[0]
    if m <= _SGH_DENSE_M_THRESHOLD:
        # Small-cell path — BIT-IDENTICAL to the previous direct call.
        return compute_Sgh(structure, all_refl, voltage_kV).to(device)
    # Large-cell path — lazy proxy, zero peak-RAM allocation.
    return _LazySghMatrix(structure, all_refl, voltage_kV, device)
