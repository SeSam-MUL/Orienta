"""SP0 — reflection list, ``U_g`` table and Bethe partition (Task 4+).

Task 4 implements :func:`reflection_list`: the *geometric* list of reflections
``(h, k, l)`` whose interplanar spacing satisfies ``d_hkl >= dmin``.  Equivalently
``|g| = 1/d_hkl <= 1/dmin``, where ``|g|`` is computed from the crystal's
reciprocal metric tensor (see :pyattr:`CrystalStructure.reciprocal_metric`).

Structure-factor selection rules (systematic absences) are *not* applied in
:func:`reflection_list` — that is the purely geometric set bounded by ``dmin``;
the absences emerge from :func:`compute_Ug_table` (Task 5) once the full
symmetry-expanded atom basis is summed.
"""
from __future__ import annotations

import math
from typing import Callable

import h5py
import numpy as np
import torch

from .scattering_factors import wk_scattering_factor
from .xtal_io import CrystalStructure

# --- EMsoft CalcUcg normalisation constants (diffraction.f90, WK branch) -------
#
# Reconciled against the EMsoft ``CalcUcg`` source (others.f90 ``FSCATT`` +
# diffraction.f90 ``CalcUcg``), verbatim, on 2026-06-10:
#
#   pref = 0.04787801 / cell%vol / (4π)        ! cell%vol is in **nm³** (lattice
#                                              !   a,b,c are stored in nm)
#   preg = 0.664840340614319 = 2·m_e·e/h²·1e-18  (V → nm⁻² conversion)
#   pre  = pref · preg
#   ff  += occ · phase · Re(sf);  gg += occ · phase · Im(sf)   ! sf = FSCATT (Å)
#   rlp%Vmod = pref · |ff|                      ! the potential V_g  [volts]
#   rlp%Umod = preg · rlp%Vmod                  ! |U_g|              [nm⁻²]
#   rlp%Ucg  = pre  · cmplx(Re(ff)-Im(gg), Im(ff)+Re(gg))   ! complex U_g [nm⁻²]
#
# CRITICAL reconciliation of the 4π:
#   EMsoft's ``FSCATT`` returns ``FREAL = 4π · DEWA · WEKO · γ`` — i.e. the
#   scattering factor *carries an explicit 4π*, and ``pref`` divides it back out
#   via the ``/(4π)``.  Our :func:`wk_scattering_factor` deliberately returns the
#   physical ``f_e`` in Å **without** that 4π (so it can be cross-checked against
#   published f_e tables — see scattering_factors.py).  We therefore fold the 4π
#   into the prefactor here: ``PREF_EFF = 0.04787801 / vol_nm3 = pref · 4π`` and
#   multiply our (4π-free) ``f`` by it.  The 4π cancellation is exact, so the
#   resulting U_g matches EMsoft's convention bit-for-bit in the limit of equal
#   scattering factors.
#
#   (NB: an earlier lessons.md draft wrote ``pref = .../vol_A3`` — that was wrong;
#   EMsoft ``cell%vol`` is in nm³, confirmed in crystal.f90::CalcMatrices.)
PREF_BARE = 0.04787801          # EMsoft WK numerator; with /vol_nm3/(4π) gives pref
PREG = 0.664840340614319        # = 2·m_e·e/h²·1e-18  (V → nm⁻²)
_TWOPI = 2.0 * math.pi
_FOURPI = 4.0 * math.pi

# Large-cell OOM guard threshold for build_ug_lut.
# The dense path allocates (M, M, 3) int64 + (M, M) float64 host arrays.
# Peak RAM ≈ M² × (24 + 8) bytes.  Cap at ~3 GB → M ≈ sqrt(3e9 / 32) ≈ 9682.
# We round to 11_000 for margin (at M=11000: 32 × 11000² ≈ 3.9 GB).
_BUILD_UG_LUT_M_THRESHOLD: int = 11_000


def reflection_list(structure: CrystalStructure, dmin: float) -> torch.Tensor:
    """Enumerate all reflections ``(h, k, l)`` with ``d_hkl >= dmin``.

    A reflection is kept when ``|g| = 1/d_hkl <= 1/dmin``, with ``|g|`` taken
    from ``structure.reciprocal_metric``.  The origin ``(0, 0, 0)`` is excluded.

    The search box is bounded from ``a / dmin``: the shortest reciprocal axis
    fixes how far any single index can range before ``|g|`` necessarily exceeds
    ``1/dmin``.  Using ``ceil(a_max / dmin)`` for every axis is a safe (slightly
    generous) bound that guarantees no in-sphere reflection is missed for any
    lattice; out-of-sphere candidates are then filtered by the exact ``|g|``
    test, so the result is independent of the box being generous.

    Args:
        structure: the crystal structure (provides ``reciprocal_metric`` and the
            direct lattice parameters in nm).
        dmin: resolution limit in nm; reflections with ``d_hkl >= dmin`` are
            kept.

    Returns:
        An ``(M, 3)`` ``int64`` tensor of Miller indices, ``(0, 0, 0)`` excluded.

    Raises:
        ValueError: if ``dmin`` is not strictly positive.
    """
    if dmin <= 0.0:
        raise ValueError(f"dmin must be > 0, got {dmin!r}")

    gstar = structure.reciprocal_metric  # (3, 3) reciprocal metric, 1/nm**2
    g_max = 1.0 / dmin  # |g| limit in 1/nm

    # Bound the search box from a_max / dmin (a in nm).  The longest direct-axis
    # length gives the largest index range any axis could need.
    a, b, c = structure.lattice[0], structure.lattice[1], structure.lattice[2]
    a_max = max(a, b, c)
    h_bound = int(np.ceil(a_max / dmin))

    rng = np.arange(-h_bound, h_bound + 1)
    hh, kk, ll = np.meshgrid(rng, rng, rng, indexing="ij")
    hkl = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1).astype(np.float64)

    # Exclude the origin.
    nonzero = np.any(hkl != 0.0, axis=1)
    hkl = hkl[nonzero]

    # |g|^2 = [h k l] g* [h k l]^T  (vectorised over all candidates).
    inv_d_sq = np.einsum("ni,ij,nj->n", hkl, gstar, hkl)
    keep = inv_d_sq <= g_max * g_max + 1e-9  # small slack for boundary rounding

    hkl_keep = hkl[keep].astype(np.int64)
    return torch.from_numpy(np.ascontiguousarray(hkl_keep))


def _cell_volume_nm3(structure: CrystalStructure) -> float:
    """Direct-cell volume in nm³ from the lattice parameters.

    ``V = sqrt(det(g))`` with ``g`` the direct metric tensor (and the reciprocal
    metric ``g*`` is its inverse), so ``V = 1/sqrt(det(g*))``.
    """
    gstar = structure.reciprocal_metric
    return float(1.0 / math.sqrt(np.linalg.det(gstar)))


def _expand_symmetry_orbit(structure: CrystalStructure):
    """Expand the asymmetric unit into the full unit-cell atom basis.

    The EMsoft oracle ``.h5`` stores only the *asymmetric unit* (one atom at the
    origin for FCC Ni/Al); the FCC ``(½½0)``-type sites — and hence the FCC
    selection rules — appear only after applying the space-group symmetry orbit.
    Uses :func:`diffpy.structure.spacegroups.GetSpaceGroup`, which maps an integer
    space-group number directly to its symmetry operations in the standard
    setting (matches the EMsoft ``CalcOrbit`` expansion for these cubic cells).

    Returns:
        list of ``(Z, occ, B, r)`` where ``r`` is an ``(3,)`` float64 array of
        fractional coordinates, one entry per atom in the conventional cell.
    """
    from diffpy.structure.spacegroups import GetSpaceGroup

    sg = GetSpaceGroup(int(structure.space_group))
    out = []
    for atom in structure.atoms:
        base = np.asarray(atom.xyz, dtype=np.float64)
        seen = set()
        for op in sg.symop_list:
            r = (op.R @ base + op.t) % 1.0
            key = tuple(np.round(r, 6))
            if key in seen:
                continue
            seen.add(key)
            out.append((int(atom.Z), float(atom.occ), float(atom.B), r))
    return out


def compute_Ug_table(
    structure: CrystalStructure,
    hkl: torch.Tensor,
    voltage_kV: float,
    *,
    absorptive: bool = True,
    absflg: int = 1,
) -> torch.Tensor:
    """Fourier coefficients ``U_g`` of the complex crystal potential.

    For each reciprocal-lattice vector ``g = (h, k, l)``::

        V_g = (0.04787801 / V_nm3) · Σ_atoms occ · f_WK(Z, |g|, B) · e^{2πi g·r}
        U_g = preg · V_g                                              [nm⁻²]

    where the sum runs over **every** atom in the conventional cell (the
    asymmetric unit expanded by the space-group symmetry orbit — see
    :func:`_expand_symmetry_orbit`), ``f_WK`` is the complex Weickenmeier-Kohl
    scattering factor (Å, relativistic + Debye-Waller + absorptive) from
    :func:`wk_scattering_factor`, and ``V_nm3`` is the direct-cell volume in nm³.

    This follows EMsoft ``diffraction.f90::CalcUcg`` (WK branch).  EMsoft assembles
    the complex potential as ``cmplx(Re(ff)-Im(gg), Im(ff)+Re(gg))`` where
    ``ff`` accumulates the elastic (real) scattering factor weighted by the atomic
    phase and ``gg`` the absorptive (imaginary) part; because each atom's complex
    scattering factor ``sf = f_el + i·f_abs`` is multiplied by the (complex) phase
    ``e^{-2πi g·r}`` (the EMsoft ``CalcUcg`` phase sign), the term
    ``occ · sf · phase`` already realises exactly that real/imag re-mixing.  We
    therefore sum ``occ · sf · phase`` directly and multiply by the (real,
    positive) prefactor — algebraically identical to the EMsoft ``cmplx(...)``
    assembly.  (For centrosymmetric Ni/Al the phase sign is immaterial; matching
    EMsoft's ``-i`` removes a latent non-centrosymmetric trap.)

    The 4π that EMsoft carries inside ``FSCATT`` (``FREAL = 4π·DEWA·WEKO``) and
    divides out in ``pref`` is folded into ``PREF_BARE/V_nm3 = pref·4π`` here,
    because :func:`wk_scattering_factor` returns the 4π-free physical ``f_e``
    (the cancellation is exact — see the module-level constant comment).

    The mean inner potential ``U_0`` (the ``g = 0`` beam) is **not** specially
    branched in EMsoft: at ``|g| = 0`` all phases are 1, so the *elastic* part of
    ``U_0`` comes out real and positive automatically (V_0 = Re(U_0)/preg ≈ tens of
    volts) — we let the same formula handle it.  The absorptive part contributes a
    finite positive imaginary ``U_0'`` (the mean absorptive potential / inelastic
    mean free path term) even at ``g = 0``; that is correct physics, not a bug.

    Hermiticity.  The crystal potential ``V(r)`` is a *real* field, so its elastic
    Fourier coefficients obey ``U^el_{-g} = conj(U^el_g)`` (call with
    ``absorptive=False`` to see this exactly).  For a **centrosymmetric** crystal
    (e.g. FCC Fm-3m) the absorptive coefficients are also real-and-even, so the
    *full* complex ``U_g`` satisfies ``U_{-g} = U_g`` (g-even) and equals
    ``conj(U_g)`` only for the elastic part — both are standard dynamical-theory
    facts and match EMsoft's ``CalcUcg`` output.

    Args:
        structure: the crystal (lattice + asymmetric-unit atoms + space group).
        hkl: ``(M, 3)`` integer tensor of Miller indices (``(0, 0, 0)`` allowed →
            yields ``U_0``).
        voltage_kV: accelerating voltage in kV (relativistic γ in ``f_WK``).
        absorptive: if True (default) include the WK imaginary part; if False
            return the purely elastic (real-potential) ``U_g`` for which
            ``U_{-g} = conj(U_g)`` holds exactly.
        absflg: EMsoft absorption flag passed to :func:`wk_scattering_factor`
            (``1`` = phonon only, default = no behaviour change; ``3`` = phonon +
            core-loss).  Only affects the imaginary channel and only when
            ``absorptive`` is True.

    Returns:
        ``(M,)`` ``complex128`` tensor of ``U_g`` in nm⁻², one per input ``g``.

    Performance.  The scattering factor ``f_WK`` depends on ``g`` only through
    ``|g|`` (``s = |g|``) and the fixed per-species ``(Z, B)`` — its absorptive
    channel (``_fphon``/``_fcore``) is a scalar-by-scalar scipy integral.  We
    therefore evaluate ``f`` once per UNIQUE ``|g|`` per species and gather (a pure
    memoisation over ``|g|`` — bit-identical to a per-pair evaluation).  On large
    cells at small ``dmin`` the difference-vector grid has high lattice
    multiplicity, so this collapses the dominant one-time potential-table setup
    cost from ``O(n_atoms · M)`` to ``O(#unique|g|)`` absorptive evaluations.
    """
    hkl_np = np.asarray(hkl.detach().cpu(), dtype=np.float64).reshape(-1, 3)
    gstar = structure.reciprocal_metric  # 1/nm²

    vol_nm3 = _cell_volume_nm3(structure)
    pref_eff = PREF_BARE / vol_nm3  # = pref · 4π (compensates 4π-free f_e)

    atoms = _expand_symmetry_orbit(structure)

    # |g| (nm⁻¹) → scattering-factor argument s = |g| (the EMsoft CalcUcg WK
    # convention: ``rlp%g = CalcLength(hkl,'r') = |g|`` is fed directly, and the
    # internal ``S = S_SCALE·s = 0.05·|g|`` is the physical sin(θ)/λ in Å⁻¹ — see
    # the AMP_2 root-cause note in :func:`wk_scattering_factor`).
    #
    # AMP_2 FIX (2026-06-18).  Previously this passed ``s = |g|/2`` (intended as
    # the physical sin(θ)/λ in nm⁻¹), but ``S_SCALE = SWK/4π = 0.05`` ALREADY folds
    # in BOTH the nm⁻¹→Å⁻¹ factor (×0.1) and the |g|→sin(θ)/λ factor (×0.5):
    # ``0.05 = 0.1·0.5``.  Feeding ``|g|/2`` applied the ÷2 a second time, so the
    # WEKO variable came out ``S = 0.05·(|g|/2) = 0.025·|g|`` — HALF the EMsoft
    # value ``S = rlp%g·swk/4π = 0.05·|g|`` (diffraction.f90::CalcUcg WK branch,
    # ``s = rlp%g*swk``).  Because WEKO falls as S², the half-angle inflated U_g in a
    # |g|-dependent way (verified vs Doyle-Turner: ratio 1.0 at s=|g|, rising 1.07→3.9
    # at s=|g|/2), compressing dynamical band contrast on dense cells.  Passing
    # ``s = |g|`` corrects the elastic WEKO S AND the absorptive FPHON/DEWA/FCORE
    # ``G = swk·s`` simultaneously.
    inv_d = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", hkl_np, gstar, hkl_np), 0.0))
    s_var = inv_d  # nm⁻¹ (= EMsoft rlp%g; internal S = 0.05·|g| = sin θ/λ in Å⁻¹)

    # --- |g|-dedup of the scattering factor (one-time-setup speedup) -------------
    # ``wk_scattering_factor`` depends on g ONLY through ``s = |g|`` (plus the
    # fixed per-species Z/B and the voltage/flags) — NOT on the pair identity (h,k,l)
    # or the atom position r.  Its absorptive channel (``_fphon``/``_fcore``) loops
    # scalar-by-scalar over EVERY s through scipy's exponential integral, so at
    # dmin=0.05 on a large cell (M ~ 1e5 difference vectors, but only a few thousand
    # DISTINCT |g| magnitudes by lattice multiplicity) it re-evaluates the same f
    # thousands of times.  We collapse s to its UNIQUE values, evaluate f once per
    # unique s per atomic species (Z, B), and gather back via the inverse index.
    # This is pure memoisation over |g| → BIT-IDENTICAL to the per-pair evaluation
    # (identical s ⇒ identical f for a given species; ``np.unique`` preserves the
    # exact float values, no rounding).  Caching across atom species means the FCC
    # 4-atom orbit (all one Z/B) evaluates f only ONCE for the whole table.
    s_uniq, s_inv = np.unique(s_var, return_inverse=True)
    s_uniq_t = torch.as_tensor(s_uniq, dtype=torch.float64)
    # ``.reshape(-1)`` guards the NumPy-2 ``return_inverse`` shape change so the
    # gather index is always 1-D (M,) regardless of the installed NumPy.
    s_inv_t = torch.as_tensor(np.asarray(s_inv).reshape(-1), dtype=torch.long)
    _f_species: dict[tuple[int, float], torch.Tensor] = {}

    ug = torch.zeros(hkl_np.shape[0], dtype=torch.complex128)
    for Z, occ, B, r in atoms:
        # f_WK is g-dependent only through |g| (s); evaluate once per unique s per
        # (Z, B) species, then gather to the full reflection list.
        species_key = (int(Z), float(B))
        f_uniq = _f_species.get(species_key)
        if f_uniq is None:
            f_uniq = wk_scattering_factor(
                Z, s_uniq_t, B, voltage_kV, absorptive=absorptive, absflg=absflg
            ).to(torch.complex128)
            _f_species[species_key] = f_uniq
        f = f_uniq[s_inv_t]  # (M,) gather: identical s → identical f (bit-exact)
        phase_arg = _TWOPI * (hkl_np @ r)  # 2π g·r, shape (M,)
        # EMsoft CalcUcg uses the e^{-i 2π g·r} phase sign (p1 = Σ exp(-i2π g·r));
        # for centrosymmetric Ni/Al cells the sign is immaterial, but matching
        # EMsoft removes a latent non-centrosymmetric trap.
        phase = torch.as_tensor(
            np.exp(-1j * phase_arg), dtype=torch.complex128
        )
        ug = ug + (occ * f * phase)

    ug = ug * (pref_eff * PREG)
    return ug


# ---------------------------------------------------------------------------
# Task 6 — Bethe strong/weak partition (excitation error + EMsoft c1/c2 split)
# ---------------------------------------------------------------------------
#
# Verified verbatim against EMsoft ``gvectors.f90::Apply_BethePotentials`` and
# ``diffraction.f90::CalcsgSingle`` (develop source, 2026-06-10).  Full physics
# write-up with citations in ``tasks/forward_sim/lessons.md`` §11.
#
# Excitation error (CalcsgSingle), all dot/length in reciprocal-cartesian ('r'):
#
#     s_g = -g·(2k+g) / [2·|k+g|·cos(angle(k+g, FN))]
#
# where ``k`` is the incident wavevector (|k| = 1/λ in the reciprocal metric),
# ``g`` the reflection, and ``FN`` the foil normal.  For the master pattern the
# beam direction *is* the foil normal, so we take ``FN = k`` (default).
#
# Bethe strong/weak (Apply_BethePotentials):
#
#     sgp   = (1/λ)·|s_g|
#     rh_h  = sgp / |U_{g-h}|     for every reflection h (10000 if |U_{g-h}| = 0)
#     m     = min_h rh_h
#     m  > c2          → IGNORE
#     c1 < m ≤ c2      → WEAK
#     m ≤ c1           → STRONG
#
# The first reflection (transmitted/incident beam, conventionally g=0) is always
# strong.  The parameters ``[c1, c2, c3, sgdbdiff]`` come from the FILE
# (``EMData/EBSDmaster/BetheParameters``), not the EMsoft code defaults.


def read_bethe_parameters(path: str) -> np.ndarray:
    """Read ``EMData/EBSDmaster/BetheParameters`` ``(4,)`` from an EMsoft ``.h5``.

    Returns the stored ``[c1, c2, c3, sgdbdiff]`` float array (the Ni/Al oracle
    stores ``[4, 8, 50, 1]``).  These are *authoritative* — they may be tighter
    than the EMsoft code defaults (8/50), so always read them from the file.

    Raises:
        ValueError: if the ``BetheParameters`` dataset is missing or not length-4.
    """
    with h5py.File(path, "r") as f:
        key = "EMData/EBSDmaster/BetheParameters"
        if key not in f:
            raise ValueError(f"{key!r} missing in {path!r}")
        bp = np.asarray(f[key][()], dtype=np.float64).ravel()
    if bp.shape[0] != 4:
        raise ValueError(
            f"BetheParameters must have 4 entries [c1,c2,c3,sgdbdiff], "
            f"got {bp.shape[0]}"
        )
    return bp


def excitation_error(
    reflections: torch.Tensor,
    direction: torch.Tensor,
    reciprocal_metric: np.ndarray,
    foil_normal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Excitation error ``s_g`` for a batch of reflections (EMsoft ``CalcsgSingle``).

    ``s_g = -g·(2k+g) / [2·|k+g|·cos(angle(k+g, FN))]`` with all dot products and
    lengths taken in **reciprocal-cartesian** space, i.e. through the reciprocal
    metric tensor ``g*`` (``u·v = uᵀ g* v``).  ``s_g = 0`` exactly when the
    reflection lies on the Ewald sphere (``|k+g| = |k|`` ⟺ numerator vanishes).

    Args:
        reflections: ``(M, 3)`` integer (or float) Miller indices ``g``.  The
            transmitted beam ``g = (0,0,0)`` yields ``s_g = 0`` (it is on the
            sphere by definition).
        direction: ``(3,)`` incident wavevector ``k`` in the same Miller/reciprocal
            coordinates, with ``|k| = 1/λ`` under the reciprocal metric.
        reciprocal_metric: ``(3, 3)`` reciprocal metric tensor ``g*`` (1/nm²).
        foil_normal: ``(3,)`` foil normal in reciprocal coordinates; defaults to
            ``direction`` (master-pattern convention: beam ∥ foil normal).

    Returns:
        ``(M,)`` float64 tensor of ``s_g`` (nm⁻¹).
    """
    gstar = np.asarray(reciprocal_metric, dtype=np.float64)
    g = np.asarray(reflections.detach().cpu(), dtype=np.float64).reshape(-1, 3)
    k = np.asarray(direction.detach().cpu(), dtype=np.float64).reshape(3)
    fn = k if foil_normal is None else np.asarray(
        foil_normal.detach().cpu(), dtype=np.float64
    ).reshape(3)

    kpg = k[None, :] + g                # (M, 3)  k + g
    tkpg = 2.0 * k[None, :] + g         # (M, 3)  2k + g

    # xnom = -g·(2k+g)  (reciprocal-cartesian dot through g*).
    xnom = -np.einsum("mi,ij,mj->m", g, gstar, tkpg)

    # q1 = |k+g| ; cos(angle(k+g, FN)) = (k+g)·FN / (|k+g|·|FN|).
    q1 = np.sqrt(np.maximum(np.einsum("mi,ij,mj->m", kpg, gstar, kpg), 0.0))
    fn_len = math.sqrt(float(fn @ gstar @ fn))
    kpg_dot_fn = np.einsum("mi,ij,j->m", kpg, gstar, fn)
    # xden = 2·|k+g|·cos(α) = 2·(k+g)·FN / |FN|.
    xden = 2.0 * kpg_dot_fn / fn_len

    # Guard the transmitted beam (g=0): xnom=0 and xden=2|k|cos(0)=2|k| → s_g=0.
    sg = np.zeros_like(xnom)
    nz = np.abs(xden) > 0.0
    sg[nz] = xnom[nz] / xden[nz]
    return torch.from_numpy(np.ascontiguousarray(sg))


def _apply_n_cap(
    strong: np.ndarray,
    weak: np.ndarray,
    m: np.ndarray,
    n_cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive Bethe strong-beam cap (iter-15) — SHARED by serial + batched paths.

    Given a ``(B, M)`` strong/weak partition and the ``(B, M)`` Bethe ratio
    ``m = sgp / max_h|U|`` the ``c1`` test ranks by, DEMOTE the **marginal** strong
    beams (largest ``m`` → closest to the ``c1`` boundary) into the weak
    (perturbative) set on every direction whose strong count exceeds ``n_cap``.  The
    logic is **identical** for both call sites (the serial :func:`bethe_partition`
    passes a 1-row batch, the vectorised :func:`bethe_partition_batched` the full
    grid), so a capped serial build is bit-faithful to a capped batched build:

    * the transmitted beam (column 0) is **always kept** strong (ranked first via
      ``m = −inf``);
    * directions **at or under** the cap are left **untouched**;
    * over-cap directions are pinned to **exactly** ``n_cap`` strong beams — the
      ``n_cap`` most strongly-coupled (smallest ``m``) survive;
    * the demoted beams move to the **weak** set (the strong+weak union is
      preserved — nothing is dropped, only re-labelled) and are **not** re-gated
      against ``c2`` (they were already *strong*, well inside the perturbation
      regime); their signed excitation error is retained downstream.

    Args:
        strong: ``(B, M)`` bool — the post-``c1``/``c3`` strong mask (with the
            transmitted beam already forced strong at column 0).
        weak: ``(B, M)`` bool — the post-``c1``/``c3`` weak mask.
        m: ``(B, M)`` float — the Bethe ratio ``sgp / max_h|U|`` (``big`` on
            all-``|U|=0`` rows), the strong/weak boundary distance.
        n_cap: int ``≥ 1`` — the maximum strong-beam count per direction.

    Returns:
        ``(strong, weak)`` ``(B, M)`` bool arrays with the demotion applied
        in place on copies (the inputs are not mutated).

    Raises:
        ValueError: if ``n_cap < 1``.
    """
    if int(n_cap) < 1:
        raise ValueError(f"n_cap must be >= 1, got {n_cap!r}")
    cap = int(n_cap)
    m_refl = strong.shape[1]
    if m_refl <= 1:
        return strong, weak  # nothing demotable (only the transmitted beam)

    strong = strong.copy()
    weak = weak.copy()
    n_strong = strong.sum(axis=1)                            # (B,)
    over = n_strong > cap
    if over.any():
        # Rank order m ASC → marginal (large-m) strong beams come LAST.  Force the
        # transmitted beam (col 0) to rank first by giving it m = -inf, and push
        # non-strong beams past the keep window with m = +inf so only currently-
        # strong beams compete for the cap.
        rank = np.where(strong, m, np.inf)                   # (B, M)
        rank[:, 0] = -np.inf                                 # col 0 ranks first
        order = np.argsort(rank, axis=1, kind="stable")      # (B, M) asc by m
        # keep_rank[b, j] = position of reflection j in b's ascending order.
        keep_rank = np.empty_like(order)
        cols = np.arange(m_refl)[None, :]
        np.put_along_axis(
            keep_rank, order, np.broadcast_to(cols, order.shape), axis=1
        )
        # Beams ranked at position >= cap among the strong set are demoted.
        demote = over[:, None] & strong & (keep_rank >= cap)
        strong = strong & ~demote
        weak = weak | demote
    return strong, weak


def bethe_partition(
    reflections: torch.Tensor,
    direction: torch.Tensor,
    Ug_table: Callable[[torch.Tensor], torch.Tensor],
    bethe_params,
    *,
    reciprocal_metric: np.ndarray,
    wavelength_nm: float,
    foil_normal: torch.Tensor | None = None,
    apply_c3_prefilter: bool = True,
    n_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition reflections into strong / weak via the EMsoft Bethe criterion.

    Implements ``gvectors.f90::Apply_BethePotentials`` for a single beam
    ``direction`` — **including** the per-direction reflection-list construction of
    ``initializers.f90::Initialize_ReflectionList`` that precedes it.  For each
    reflection ``g`` (index ``ir``):

    * compute the excitation error ``s_g`` (see :func:`excitation_error`) and
      ``sgp = |s_g| / λ``;
    * **EMsoft list construction (the c3 pre-filter).**  EMsoft only adds ``g`` to
      the per-direction reflection list when its **self** Bethe ratio
      ``m_self = sgp / |U_g|`` (``U_g = U_{g-0}``, the reflection's own structure
      factor) satisfies ``m_self ≤ c3`` — or, for a double-diffraction reflection
      (``|U_g| = 0``), when ``|s_g| ≤ sgdbdiff``.  Reflections failing this gate are
      **never listed** (and so are neither strong nor weak); see the lessons.md §
      "c3 reflection-list pre-filter".  This bounds the list to reflections actually
      near the Ewald sphere — a pure geometric ``|g| ≤ 1/dmin`` sphere (our
      :func:`reflection_list`) is far larger and would inject reflections with
      astronomically large ``s_g`` that EMsoft never solves.
    * for every **listed** reflection ``h`` (index ``ih``, including ``ir`` itself
      and the transmitted ``g = 0``) form the ratio ``rh = sgp / |U_{g-h}|`` — or
      ``10000`` when ``|U_{g-h}| = 0`` (a double-diffraction difference vector);
    * ``m = min_h rh``;  then ``m ≤ c1`` → **strong**, ``c1 < m ≤ c2`` → **weak**,
      ``m > c2`` → **ignored** (neither mask set).

    The single-pass self-gate here (drop ``m_self > c3`` from the partition over the
    full list) is **bit-identical** to EMsoft's two-stage build-then-partition: a
    reflection failing the gate has such a large ``s_g`` that its coupling
    ``|U_{g-h}|`` to any surviving ``h`` is negligible vs ``sgp``, so it never
    provides a surviving reflection's row-minimum (verified across directions at
    dmin = 0.05 and 0.10, Ni — see the iter-10 probe).

    The **first** reflection in ``reflections`` is treated as the transmitted /
    incident beam and is always strong (matches EMsoft's unconditional
    ``listroot%next%strong = .TRUE.``), and is never dropped by the c3 gate.

    Args:
        reflections: ``(M, 3)`` integer Miller indices.  Index 0 is the
            transmitted beam (conventionally ``(0, 0, 0)``).
        direction: ``(3,)`` incident wavevector ``k`` (``|k| = 1/λ`` under the
            reciprocal metric), in Miller/reciprocal coordinates.
        Ug_table: a callable mapping an ``(N, 3)`` int tensor of *difference*
            Miller indices ``g - h`` to an ``(N,)`` complex tensor of ``U_{g-h}``
            (EMsoft's ``cell%LUT``).  Typically a closure over
            :func:`compute_Ug_table`.
        bethe_params: length-≥2 sequence ``[c1, c2, c3, sgdbdiff]`` — the strong
            cutoff ``c1``, weak cutoff ``c2``, list-construction cutoff ``c3``
            (``sgmaxval``) and double-diffraction ``|s_g|`` cutoff ``sgdbdiff``
            (e.g. from :func:`read_bethe_parameters`).  ``c3`` / ``sgdbdiff`` are
            only used when ``apply_c3_prefilter`` is True; when they are absent
            (length-2 ``bethe_params``) the c3 gate is skipped.
        reciprocal_metric: ``(3, 3)`` reciprocal metric tensor ``g*`` (1/nm²).
        wavelength_nm: relativistic electron wavelength ``λ`` in nm.
        foil_normal: optional ``(3,)`` foil normal; defaults to ``direction``.
        apply_c3_prefilter: when True (default), apply the EMsoft
            ``Initialize_ReflectionList`` c3 self-ratio reflection-list gate
            described above (drop ``m_self > c3`` reflections, keeping
            double-diffraction ``|s_g| ≤ sgdbdiff`` ones).  Requires ``c3`` (and
            ``sgdbdiff``) in ``bethe_params``; if they are missing the gate is a
            no-op.  Set False to partition the **full** input list with no list
            construction (the pre-iter-10 behaviour — for callers that have already
            filtered, or for the bit-faithfulness oracle).
        n_cap: **adaptive Bethe strong-beam cap (iter-15) — opt-in, default
            ``None`` = OFF (bit-identical to the un-capped partition).**  When an
            int ``≥ 1``, after the standard ``c1`` strong/weak partition (and the
            c3 gate) the marginal strong beams (largest Bethe ratio
            ``m = sgp / max_h|U|``, closest to the ``c1`` boundary) are DEMOTED into
            the weak (perturbative) set until the strong count is ``n_cap`` (the
            transmitted beam, column 0, is always kept strong).  The demotion is
            performed by the shared :func:`_apply_n_cap` helper — **byte-for-byte the
            same logic** the batched :func:`bethe_partition_batched` applies — so a
            capped serial build is bit-faithful to a capped batched build (the
            autopilot's batched==serial correctness gate).  ``None`` performs NO
            demotion.  ``ValueError`` if ``< 1``.

    Returns:
        ``(strong_mask, weak_mask)``, each a ``(M,)`` bool tensor over the input
        reflection list.  ``strong_mask & weak_mask`` is all-False; the remaining
        (ignored, incl. c3-pruned) reflections are ``~(strong_mask | weak_mask)``.

    Raises:
        ValueError: if ``wavelength_nm`` is not strictly positive, ``bethe_params``
            has fewer than two entries, or ``n_cap`` is ``< 1``.
    """
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be > 0, got {wavelength_nm!r}")
    if n_cap is not None and int(n_cap) < 1:
        raise ValueError(f"n_cap must be >= 1 or None, got {n_cap!r}")
    bp = np.asarray(bethe_params, dtype=np.float64).ravel()
    if bp.shape[0] < 2:
        raise ValueError("bethe_params must provide at least [c1, c2]")
    c1, c2 = float(bp[0]), float(bp[1])
    c3 = float(bp[2]) if bp.shape[0] >= 3 else None
    sgdbdiff = float(bp[3]) if bp.shape[0] >= 4 else 0.0

    hkl = np.asarray(reflections.detach().cpu(), dtype=np.int64).reshape(-1, 3)
    m_refl = hkl.shape[0]

    # |U_{g-h}| for every (ir, ih) pair.  Build the unique difference vectors once,
    # query the LUT in a single batched call, then scatter back to the (M, M) grid.
    diffs = hkl[:, None, :] - hkl[None, :, :]            # (M, M, 3)
    flat = diffs.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    u_uniq = Ug_table(torch.from_numpy(np.ascontiguousarray(uniq)))
    u_abs_uniq = torch.abs(u_uniq).detach().cpu().numpy().astype(np.float64)
    u_abs = u_abs_uniq[inv].reshape(m_refl, m_refl)      # |U_{g-h}| grid

    # Excitation error → sgp = |s_g| / λ  (one per reflection ir).
    sg = excitation_error(
        reflections, direction, reciprocal_metric, foil_normal=foil_normal
    ).detach().cpu().numpy()
    sgp = np.abs(sg) / wavelength_nm                     # (M,)

    # rh[ir, ih] = sgp[ir] / |U_{g_ir - g_ih}| ; 10000 where |U| == 0 (dbdiff).
    big = 10000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = sgp[:, None] / u_abs
    rh = np.where(u_abs > 0.0, rh, big)
    m = rh.min(axis=1)                                   # (M,)  min over ih

    strong = m <= c1
    weak = (m > c1) & (m <= c2)

    # EMsoft list construction (Initialize_ReflectionList): keep a reflection only
    # if its SELF ratio m_self = sgp/|U_g| ≤ c3 (or, for |U_g|=0 double diffraction,
    # |s_g| ≤ sgdbdiff).  Reflections failing this gate are not listed → drop them
    # from BOTH strong and weak (single-pass equivalent of EMsoft's build-then-split,
    # see the docstring + iter-10 probe).  ``u_abs[:, 0]`` = |U_{g-0}| = |U_g|.
    if apply_c3_prefilter and c3 is not None and m_refl > 0:
        u_g = u_abs[:, 0]                                # |U_g| (column 0 = h=g=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            m_self = sgp / u_g
        # Listed: |U_g|>0 and m_self ≤ c3; OR double-diffraction |U_g|=0 and |s_g| ≤ sgdbdiff.
        listed = np.where(
            u_g > 0.0, m_self <= c3, np.abs(sg) <= sgdbdiff
        )
        listed[0] = True                                # transmitted beam always listed
        strong = strong & listed
        weak = weak & listed

    # The transmitted beam (index 0) is unconditionally strong.
    if m_refl > 0:
        strong[0] = True
        weak[0] = False

    # --- Adaptive Bethe n-cap (iter-15): demote the marginal strong beams --------
    # Identical demotion to bethe_partition_batched (the SHARED _apply_n_cap helper),
    # applied AFTER the transmitted-beam force so the input strong set matches the
    # batched path exactly → a capped serial build is bit-faithful to a capped
    # batched build.  Reshape the (M,) masks/ratio to a 1-row batch.
    if n_cap is not None and m_refl > 1:
        strong, weak = _apply_n_cap(
            strong[None, :], weak[None, :], m[None, :], int(n_cap)
        )
        strong = strong[0]
        weak = weak[0]

    return (
        torch.from_numpy(np.ascontiguousarray(strong)),
        torch.from_numpy(np.ascontiguousarray(weak)),
    )


# ---------------------------------------------------------------------------
# SP4 lever 1+2 — U_g difference-LUT precompute + vectorised Bethe partition
# ---------------------------------------------------------------------------
#
# The serial :func:`bethe_partition` rebuilds the full ``(M, M)`` ``|U_{g-h}|``
# grid (a numpy ``np.unique`` + ``Ug_table`` query over ``M²`` pairs) for EVERY
# direction, even though that grid is **direction-independent** — paying the
# Python/numpy overhead once per direction (profiled at 72.7 % of wall time).
#
# These two functions hoist that grid out of the loop and vectorise the
# excitation-error classification across a BATCH of directions:
#
# * :func:`build_ug_lut` computes the ``(M+1,)`` per-row maximum
#   ``max_h |U_{g_ir − g_ih}|`` ONCE (a single batched ``Ug_table`` over the
#   unique difference vectors), plus the row indices with no non-zero ``|U|``
#   (the "all-double-diffraction" rows that always map to the ``big`` ratio).
#
# * :func:`bethe_partition_batched` evaluates ``s_g`` for all ``B`` directions at
#   once (a ``(B, M)`` einsum on the reciprocal metric) and classifies
#   strong/weak with the EMsoft ``c1/c2`` criterion as ``(B, M)`` booleans.
#
# Bit-faithfulness — algebraic equivalence to the serial min.  The serial code
# forms ``m[ir] = min_h rh[ir, h]`` with ``rh[ir, h] = sgp[ir] / |U[ir, h]|``
# (and ``10000`` where ``|U|==0``).  Because ``sgp[ir] ≥ 0``, dividing by a
# LARGER ``|U|`` gives a SMALLER ratio, so the min over ``h`` is reached at the
# row's **maximum** ``|U|``::
#
#     m[ir] = sgp[ir] / max_h |U[ir, h]|        (when some |U[ir, h]| > 0)
#     m[ir] = 10000                              (when all |U[ir, h]| == 0)
#
# i.e. the per-direction ``(M, M)`` min collapses to a precomputed ``(M,)``
# vector ``max_h |U|`` and a ``(B, M)`` elementwise divide — verified
# bit-identical to the serial masks at npx≤15 for Ni/Al (FORWARD_SIM_DEVICE=cpu).
#
# The transmitted beam (row 0) is forced strong / not-weak exactly as the serial
# path does.


def build_ug_lut(
    reflections: torch.Tensor,
    Ug_table: Callable[[torch.Tensor], torch.Tensor],
    *,
    _large_cell_row_chunk: int = 1024,
    diff_lut: torch.Tensor | None = None,
    diff_lut_H: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute the direction-independent Bethe ``|U_{g-h}|`` row-maxima (lever 1).

    Builds, ONCE for the whole direction grid, the per-row maximum of the
    ``(M, M)`` absolute potential grid ``|U_{g_ir − g_ih}|`` over all column
    reflections ``h`` — exactly the quantity the Bethe min reduces to (see the
    module comment for the algebraic equivalence) — **and** the per-reflection
    self potential ``|U_g| = |U_{g_ir − 0}|`` (column 0) used by the EMsoft c3
    reflection-list pre-filter.  All ``M²`` difference vectors are reduced to their
    **unique** set and queried through ``Ug_table`` in a single batched call
    (EMsoft ``cell%LUT``), then scattered back; the row maximum, an "all-zero row"
    mask, and the self ``|U_g|`` column are returned.

    **Large-cell OOM guard (iter-16 productionisation).**  For ``M`` above
    ``_BUILD_UG_LUT_M_THRESHOLD`` (≈ 11 000) the dense ``(M, M, 3)`` int64
    difference array and the resulting ``(M, M)`` float64 ``|U|`` grid would
    exceed ~3 GB host RAM.  Above that threshold the function automatically
    switches to a memory-safe chunked path: it builds the dense difference-box LUT
    once via :func:`build_diff_lut` (a ``(D, D, D)`` complex box, tiny even at
    dmin=0.05) and then gathers ``|U_{g−h}|`` per row-chunk of size
    ``_large_cell_row_chunk``, reducing the per-chunk peak to
    ``row_chunk × M × 8 bytes`` (float64).  The result is **bit-identical** to
    the dense path (verified in the iter-16 harness).

    For ``M < _BUILD_UG_LUT_M_THRESHOLD`` the original dense code path runs
    **unchanged** — Ni/Al and other small cells are guaranteed bit-identical to
    all prior validated results.

    Args:
        reflections: ``(M, 3)`` integer Miller indices.  Index 0 is the
            transmitted beam (conventionally ``(0, 0, 0)``).
        Ug_table: callable mapping an ``(N, 3)`` int tensor of *difference* Miller
            indices ``g − h`` to an ``(N,)`` complex tensor of ``U_{g-h}``.
        _large_cell_row_chunk: rows processed per chunk in the large-cell path
            (memory knob; does not affect the result).
        diff_lut: **Lever 2b-1/2b-3 — optional precomputed difference box.**  When a
            ``(D, D, D)`` complex ``U_{g−h}`` box (from :func:`build_diff_lut`) is
            supplied, the row-maxima / all-zero / self-``|U|`` are gathered from it
            directly — **dropping the ``np.unique((M², 3), axis=0)`` dedup** that
            dominates the dense path on mid-size cells (tau2 M=1459, mono M=6249,
            both below ``_BUILD_UG_LUT_M_THRESHOLD`` so they otherwise took the slow
            ``np.unique`` path).  The gather runs on the box's device (GPU when the
            box is on CUDA), per row-chunk so the ``(B, M)`` ``|U|`` working set
            stays bounded.  **Bit-identical** to the dense ``np.unique`` path (the
            box holds exactly the same ``U_{g−h}``; proven max|Δ| = 0.0 on
            tau2/mono).  ``None`` (default) keeps the original M-threshold behaviour
            (small cell → dense ``np.unique``; large cell → internal box).  Sharing
            ONE box with :func:`backend.forward_sim.dynamical.master_builder._lever3_eval`
            also avoids building the difference box twice (Lever 2b-3).
        diff_lut_H: the box half-offset (a difference ``d`` maps to ``diff_lut[d + H]``);
            required when ``diff_lut`` is given.

    Returns:
        ``(max_u_per_row, all_zero_row, u_self)`` where ``max_u_per_row`` is an
        ``(M,)`` float64 tensor of ``max_h |U_{g_ir − g_ih}|``, ``all_zero_row`` an
        ``(M,)`` bool tensor flagging rows whose entire ``|U|`` is zero (those
        map to the EMsoft ``big`` ratio in the partition), and ``u_self`` an
        ``(M,)`` float64 tensor of ``|U_{g_ir}| = |U_{g_ir − 0}|`` (the c3
        self-ratio denominator; index 0 along the column axis is the transmitted
        beam, present in the EMsoft master list at index 0).  All three are returned
        on CPU (the Bethe partition consumes them as numpy).
    """
    hkl = np.asarray(reflections.detach().cpu(), dtype=np.int64).reshape(-1, 3)
    m_refl = hkl.shape[0]

    # ---------------------------------------------------------------------- #
    # Lever 2b-1/2b-3 — SHARED PRECOMPUTED BOX gather path.                    #
    # When the caller passes a difference box, gather row-maxima / all-zero /  #
    # self-|U| from it (no np.unique(M²)).  Runs on the box's device (GPU when #
    # the box is CUDA); per-row-chunk to bound the (B, M) |U| working set.     #
    # Bit-identical to the dense np.unique path (same U values).               #
    # ---------------------------------------------------------------------- #
    if diff_lut is not None:
        dev = diff_lut.device
        D = diff_lut.shape[-1]
        H = int(diff_lut_H)
        lut_abs = diff_lut.abs()                              # (D, D, D) float
        lut_flat = lut_abs.reshape(-1)
        hkl_t = torch.from_numpy(hkl).to(dev)                # (M, 3) int64
        zero_row = hkl_t[0:1, :]                              # (1, 3) = g=0
        max_u = torch.empty(m_refl, dtype=torch.float64, device=dev)
        all_zero = torch.empty(m_refl, dtype=torch.bool, device=dev)
        u_self = torch.empty(m_refl, dtype=torch.float64, device=dev)
        for i0 in range(0, m_refl, _large_cell_row_chunk):
            i1 = min(i0 + _large_cell_row_chunk, m_refl)
            rows = hkl_t[i0:i1, :]                            # (B, 3)
            diffs = rows[:, None, :] - hkl_t[None, :, :]      # (B, M, 3)
            idx = diffs + H
            flat = (idx[..., 0] * D + idx[..., 1]) * D + idx[..., 2]  # (B, M)
            ua = lut_flat.index_select(0, flat.reshape(-1)).reshape(
                i1 - i0, m_refl
            ).to(torch.float64)
            max_u[i0:i1] = ua.max(dim=1).values
            all_zero[i0:i1] = ~(ua > 0.0).any(dim=1)
            sd = rows - zero_row                              # (B, 3)
            si = sd + H
            sf = (si[:, 0] * D + si[:, 1]) * D + si[:, 2]    # (B,)
            u_self[i0:i1] = lut_flat.index_select(0, sf).to(torch.float64)
        return (
            max_u.detach().cpu(),
            all_zero.detach().cpu(),
            u_self.detach().cpu(),
        )

    if m_refl <= _BUILD_UG_LUT_M_THRESHOLD:
        # ------------------------------------------------------------------ #
        # SMALL-CELL PATH — UNCHANGED (guaranteed bit-identical to all prior  #
        # validated results for Ni/Al and cells with M <= threshold).         #
        # ------------------------------------------------------------------ #
        diffs = hkl[:, None, :] - hkl[None, :, :]            # (M, M, 3)
        flat = diffs.reshape(-1, 3)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        u_uniq = Ug_table(torch.from_numpy(np.ascontiguousarray(uniq)))
        u_abs_uniq = torch.abs(u_uniq).detach().cpu().numpy().astype(np.float64)
        u_abs = u_abs_uniq[inv].reshape(m_refl, m_refl)      # |U_{g-h}| grid
        max_u = u_abs.max(axis=1)                            # (M,) row maximum
        all_zero = ~(u_abs > 0.0).any(axis=1)                # (M,) no non-zero |U|
        # |U_g| = |U_{g-0}|.  Index 0 is the transmitted beam (g=0), so column 0
        # of the difference grid is g_ir − g_0 = g_ir → the self structure factor.
        u_self = u_abs[:, 0].copy() if m_refl > 0 else u_abs[:, :0].reshape(0)
        return (
            torch.from_numpy(np.ascontiguousarray(max_u)),
            torch.from_numpy(np.ascontiguousarray(all_zero)),
            torch.from_numpy(np.ascontiguousarray(u_self)),
        )

    # ---------------------------------------------------------------------- #
    # LARGE-CELL PATH — chunked, memory-safe, bit-identical to the dense path #
    # Dense (M,M,3) int64 + (M,M) float64 would exceed ~3 GB above threshold. #
    # Instead: build the box-LUT once (tiny), then gather per row-chunk.      #
    # ---------------------------------------------------------------------- #
    from ..dynamical.scattering_matrix import build_diff_lut  # lazy import avoids circularity

    # build_diff_lut needs a device; use CPU (this function is host-side).
    _cpu = torch.device("cpu")
    lut, H = build_diff_lut(reflections, Ug_table, _cpu, dtype=torch.complex128)
    lut_abs = lut.abs()   # (D, D, D) float64
    D = lut_abs.shape[-1]

    hkl_t = torch.from_numpy(hkl)                            # (M, 3) int64 on CPU
    zero_row = hkl_t[0:1, :]                                  # (1, 3) = g=0 (transmitted)

    max_u = torch.empty(m_refl, dtype=torch.float64)
    all_zero = torch.empty(m_refl, dtype=torch.bool)
    u_self = torch.empty(m_refl, dtype=torch.float64)

    for i0 in range(0, m_refl, _large_cell_row_chunk):
        i1 = min(i0 + _large_cell_row_chunk, m_refl)
        rows = hkl_t[i0:i1, :]                                # (B, 3)
        # Differences: rows[:, None, :] - hkl_t[None, :, :]  → (B, M, 3)
        diffs = rows[:, None, :] - hkl_t[None, :, :]          # (B, M, 3)
        idx = (diffs + H).to(torch.long)                       # (B, M, 3)
        flat = (idx[..., 0] * D + idx[..., 1]) * D + idx[..., 2]  # (B, M)
        ua = lut_abs.reshape(-1).index_select(0, flat.reshape(-1)).reshape(i1 - i0, m_refl)
        max_u[i0:i1] = ua.max(dim=1).values
        all_zero[i0:i1] = ~(ua > 0.0).any(dim=1)
        # u_self[i] = |U_{g_i − 0}| — difference of row i with the transmitted beam.
        sd = rows - zero_row                                   # (B, 3)
        si = (sd + H).to(torch.long)                           # (B, 3)
        sf = (si[:, 0] * D + si[:, 1]) * D + si[:, 2]         # (B,)
        u_self[i0:i1] = lut_abs.reshape(-1).index_select(0, sf)

    return max_u.to(torch.float64), all_zero, u_self.to(torch.float64)


def bethe_partition_batched(
    reflections: torch.Tensor,
    directions: torch.Tensor,
    max_u_per_row: torch.Tensor,
    all_zero_row: torch.Tensor,
    bethe_params,
    *,
    reciprocal_metric: np.ndarray,
    wavelength_nm: float,
    dir_chunk: int = 32_768,
    u_self: torch.Tensor | None = None,
    apply_c3_prefilter: bool = True,
    n_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strong/weak Bethe partition for a BATCH of directions (lever 2).

    Vectorises :func:`bethe_partition` over ``B`` beam directions: the excitation
    error ``s_g`` is computed for all directions at once as a ``(B, M)`` tensor
    (einsum on the reciprocal metric, foil normal = beam direction — the
    master-pattern convention), then classified strong/weak with the EMsoft
    ``c1/c2`` criterion using the precomputed direction-independent row maxima
    from :func:`build_ug_lut`.

    The result is **bit-identical** to calling :func:`bethe_partition` per
    direction (verified for Ni/Al at npx≤15): the only change is that the
    per-direction ``(M, M)`` ``|U|`` rebuild and ``min`` are replaced by the
    algebraically-equal ``sgp / max_h|U|`` form with ``max_h|U|`` hoisted out of
    the loop.

    **SP4 iter-5 OOM fix.**  The per-direction excitation-error einsum materialises
    ``(B, M, 3)`` working arrays (``k+g``, ``2k+g``, the broadcast ``g``).  At the
    1:1 target (npx=500 ⇒ ``B`` ≈ 785k inside-disc directions, dmin=0.05 ⇒ ``M`` ≈
    1419) a single ``(B, M, 3)`` float64 array is ``785349·1419·3·8`` ≈ **24.9 GiB**
    — an out-of-memory crash.  We therefore stream the directions in chunks of
    ``dir_chunk`` (default 32 768), running the **same** ``(B', M)`` batched math
    per chunk and concatenating the strong/weak masks.  Each chunk's working set is
    ``dir_chunk·M·3`` float64 (≈ 1.1 GiB for the defaults) — bounded, and the result
    is **bit-identical** to the un-chunked computation (the chunks are independent
    rows of the same ``(B, M)`` classification; concatenation reassembles them in
    order).

    Args:
        reflections: ``(M, 3)`` integer Miller indices; index 0 is the
            transmitted beam.
        directions: ``(B, 3)`` incident wavevectors ``k`` (``|k| = 1/λ`` under the
            reciprocal metric), in Miller/reciprocal coordinates.
        max_u_per_row: ``(M,)`` ``max_h |U_{g_ir − g_ih}|`` from :func:`build_ug_lut`.
        all_zero_row: ``(M,)`` bool — rows with no non-zero ``|U|`` (→ ``big``).
        bethe_params: length-≥2 sequence ``[c1, c2, c3, sgdbdiff]`` Bethe cutoffs.
        reciprocal_metric: ``(3, 3)`` reciprocal metric tensor ``g*`` (1/nm²).
        wavelength_nm: relativistic electron wavelength ``λ`` (nm).
        dir_chunk: max directions processed per chunk (memory knob; does not
            affect the result).  The ``(B', M, 3)`` excitation-error working set is
            ``dir_chunk·M·3`` float64; lower it on tight RAM, raise it for fewer
            chunks.  ``<= 0`` processes all directions in one chunk (the legacy,
            OOM-prone path — only for tiny grids).
        u_self: ``(M,)`` ``|U_g| = |U_{g−0}|`` from :func:`build_ug_lut` (the third
            return value).  Required when ``apply_c3_prefilter`` is True — it is the
            denominator of the EMsoft c3 reflection-list self-ratio.  ``None``
            disables the c3 gate (pre-iter-10 behaviour) regardless of the flag.
        apply_c3_prefilter: when True (default) and ``u_self`` is given and
            ``bethe_params`` has ``c3``, apply the EMsoft
            ``Initialize_ReflectionList`` reflection-list gate per direction: keep a
            reflection only if its SELF ratio ``sgp/|U_g| ≤ c3`` (or, for
            ``|U_g|=0`` double diffraction, ``|s_g| ≤ sgdbdiff``); otherwise it is
            **not listed** → neither strong nor weak.  Bit-identical to calling
            :func:`bethe_partition` with the same flag per direction (the
            single-pass self-gate equals EMsoft's build-then-split; see the
            :func:`bethe_partition` docstring + iter-10 probe).
        n_cap: **adaptive Bethe strong-beam cap (iter-15) — opt-in, default
            ``None`` = OFF (bit-identical to the un-capped partition).**  When an
            int ``≥ 1``, after the standard ``c1`` strong/weak partition (and the
            c3 gate) any direction whose strong-beam count exceeds ``n_cap`` has its
            **marginal** strong beams DEMOTED into the weak (perturbative) set: the
            transmitted beam (column 0) is always kept, then the remaining strong
            beams are ranked by their Bethe ratio ``m = sgp / max_h|U|`` (the
            distance-to-the-strong/weak-boundary the ``c1`` test uses), the
            ``n_cap − 1`` with the SMALLEST ``m`` (most strongly coupled, deepest in
            the strong regime) are kept strong, and the rest — the ones closest to
            the ``c1`` boundary — are moved to the weak set.  Bethe perturbation
            already handles those marginal beams accurately, so the master NCC stays
            ~unchanged while the dense ``A`` shrinks (the per-direction ``n³``
            ``matrix_exp``/``eig`` is the wall on large cells).  A demoted beam keeps
            its real signed excitation error for the weak-beam correction; it is NOT
            re-gated against ``c2`` (it was already a *strong* beam, well inside the
            perturbation regime).  ``None`` performs NO demotion — every output mask
            is exactly the un-capped result.  ``ValueError`` if ``< 1``.

    Returns:
        ``(strong_masks, weak_masks)``, each a ``(B, M)`` bool tensor.

    Raises:
        ValueError: if ``wavelength_nm`` is not strictly positive,
            ``bethe_params`` has fewer than two entries, or ``n_cap`` is ``< 1``.
    """
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be > 0, got {wavelength_nm!r}")
    if n_cap is not None and int(n_cap) < 1:
        raise ValueError(f"n_cap must be >= 1 or None, got {n_cap!r}")
    bp = np.asarray(bethe_params, dtype=np.float64).ravel()
    if bp.shape[0] < 2:
        raise ValueError("bethe_params must provide at least [c1, c2]")
    c1, c2 = float(bp[0]), float(bp[1])
    c3 = float(bp[2]) if bp.shape[0] >= 3 else None
    sgdbdiff = float(bp[3]) if bp.shape[0] >= 4 else 0.0

    gstar = np.asarray(reciprocal_metric, dtype=np.float64)
    g = np.asarray(reflections.detach().cpu(), dtype=np.float64).reshape(-1, 3)
    k_all = np.asarray(directions.detach().cpu(), dtype=np.float64).reshape(-1, 3)
    b_dir = k_all.shape[0]
    m_refl = g.shape[0]

    # Direction-independent classification helpers (computed once).
    max_u = np.asarray(max_u_per_row.detach().cpu(), dtype=np.float64).reshape(-1)
    azr = np.asarray(all_zero_row.detach().cpu(), dtype=bool).reshape(-1)
    big = 10000.0
    safe_max = np.where(max_u > 0.0, max_u, 1.0)

    # EMsoft c3 reflection-list pre-filter (Initialize_ReflectionList): the per-
    # reflection SELF ratio ``sgp/|U_g|`` (direction-dependent through ``sgp``) is
    # gated against c3 below; ``|U_g|`` is direction-independent → prepared once.
    do_c3 = bool(apply_c3_prefilter) and c3 is not None and u_self is not None
    if do_c3:
        u_g = np.asarray(u_self.detach().cpu(), dtype=np.float64).reshape(-1)
        u_g_pos = u_g > 0.0
        safe_u_g = np.where(u_g_pos, u_g, 1.0)

    # Chunk the directions so the (B', M, 3) excitation-error working set stays
    # bounded (the OOM fix — see the docstring).  Each chunk produces the same
    # (B', M) masks the un-chunked path would; concatenation in direction order
    # is bit-identical.
    step = b_dir if dir_chunk is None or dir_chunk <= 0 else int(dir_chunk)
    step = max(1, step)
    strong_parts: list[np.ndarray] = []
    weak_parts: list[np.ndarray] = []
    for c0 in range(0, b_dir, step):
        k = k_all[c0:c0 + step]                                  # (B', 3)
        b = k.shape[0]

        # --- Batched excitation error s_g (B', M) ------------------------------
        # Identical math to excitation_error() but vectorised over directions,
        # with foil_normal = beam direction (the master-pattern convention).
        #   s_g = -g·(2k+g) / [2·(k+g)·FN / |FN|]
        kpg = k[:, None, :] + g[None, :, :]          # (B', M, 3)  k + g
        tkpg = 2.0 * k[:, None, :] + g[None, :, :]    # (B', M, 3)  2k + g
        g_b = np.broadcast_to(g[None, :, :], (b, m_refl, 3))
        xnom = -np.einsum("bmi,ij,bmj->bm", g_b, gstar, tkpg)    # (B', M)
        fn_len = np.sqrt(np.einsum("bi,ij,bj->b", k, gstar, k))  # (B',)
        kpg_dot_fn = np.einsum("bmi,ij,bj->bm", kpg, gstar, k)   # (B', M)
        xden = 2.0 * kpg_dot_fn / fn_len[:, None]                # (B', M)
        sg = np.zeros((b, m_refl), dtype=np.float64)
        nz = np.abs(xden) > 0.0
        sg[nz] = xnom[nz] / xden[nz]

        sgp = np.abs(sg) / wavelength_nm                         # (B', M)

        # --- Classify via m = sgp / max_h|U| (big on all-zero |U| rows) --------
        m = sgp / safe_max[None, :]                              # (B', M)
        m = np.where(azr[None, :], big, m)

        strong = m <= c1
        weak = (m > c1) & (m <= c2)

        # EMsoft c3 reflection-list gate (per direction): keep a reflection only if
        # m_self = sgp/|U_g| ≤ c3 (or, for |U_g|=0 double diffraction, |s_g| ≤
        # sgdbdiff).  Reflections failing the gate are not listed → drop from BOTH
        # strong and weak.  Bit-identical to the serial single-pass gate.
        if do_c3:
            m_self = sgp / safe_u_g[None, :]                     # (B', M)
            listed = np.where(
                u_g_pos[None, :], m_self <= c3, np.abs(sg) <= sgdbdiff
            )
            if m_refl > 0:
                listed[:, 0] = True                             # transmitted beam
            strong = strong & listed
            weak = weak & listed

        # Transmitted beam (column 0) is unconditionally strong for every dir.
        if m_refl > 0:
            strong[:, 0] = True
            weak[:, 0] = False

        # --- Adaptive Bethe n-cap (iter-15): demote the marginal strong beams ----
        # When a direction's strong count exceeds n_cap, move the strong beams
        # closest to the c1 boundary (LARGEST Bethe ratio m) into the weak set,
        # keeping the n_cap most strongly-coupled (smallest m) + the transmitted
        # beam (column 0, always kept).  Bethe perturbation accurately handles the
        # demoted marginal beams → master NCC ~unchanged, dense A shrinks.  Delegated
        # to the SHARED :func:`_apply_n_cap` helper — byte-for-byte the same logic
        # the serial :func:`bethe_partition` applies, so a capped batched build is
        # bit-faithful to a capped serial build.
        if n_cap is not None and m_refl > 1:
            strong, weak = _apply_n_cap(strong, weak, m, int(n_cap))

        strong_parts.append(strong)
        weak_parts.append(weak)

    if not strong_parts:  # b_dir == 0
        strong = np.zeros((0, m_refl), dtype=bool)
        weak = np.zeros((0, m_refl), dtype=bool)
    else:
        strong = np.concatenate(strong_parts, axis=0)
        weak = np.concatenate(weak_parts, axis=0)

    return (
        torch.from_numpy(np.ascontiguousarray(strong)),
        torch.from_numpy(np.ascontiguousarray(weak)),
    )
