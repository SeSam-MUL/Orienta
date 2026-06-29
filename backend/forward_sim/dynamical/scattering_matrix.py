"""SP1 — dynamical matrix ``A`` and scattering matrix ``S`` (Task 8).

This module assembles the complex **dynamical matrix** ``A`` for a batch of beam
directions and exponentiates it into the **scattering matrix** ``S(z)``.

GetDynMat mode-``'D'`` parity (verified vs EMsoft develop ``MBmodule.f90``)
--------------------------------------------------------------------------
:func:`build_A` reproduces EMsoft's ``GetDynMat`` **mode ``'D'``** (the standard
Bloch-wave path the EBSD master uses; ``EMEBSDmaster.f90`` calls ``GetDynMat``
*without* the optional ``BlochMode`` argument → ``AorD = 'D'``).  For a strong
reflection list (row index ``ir``, column index ``ic``) and a per-direction weak
set ``{w}`` (the Bethe perturbation beams), with ``λ = mLambda`` (nm), ``U`` =
``cell%LUT`` (the complex potential coefficient table, nm⁻²), and ``Upz =
rlp%Upmod`` (the normal-absorption coefficient = ``Im(U_0)``, nm⁻²):

* **off-diagonal** (``MBmodule.f90`` GetDynMat ``'D'``)::

      DynMat(ir,ic) = U_{g_ir − g_ic}
                      − cmplx(0.5·λ, 0)·Σ_w  U_{g_ir − w}·U_{w − g_ic} / s_w

  i.e. **plain** ``U_{g−h}`` (NO ``π``, NO ``i`` prefactor) minus the Bethe
  weak-beam correction ``weaksum``.  ``s_w`` is the (signed) excitation error of
  weak beam ``w`` (nm⁻¹).  With an empty weak set the off-diagonal is bare
  ``U_{g−h}``.

* **diagonal** (``MBmodule.f90`` GetDynMat ``'D'``)::

      DynMat(ir,ir) = cmplx( 2·s_{g_ir}/λ − weaksgsum_ir ,  Upz )
      weaksgsum_ir  = (λ/2)·Σ_w  |U_{g_ir − w}|² / s_w

  the excitation-error term ``2·s_g/λ`` (real), reduced by the weak-beam
  ``weaksgsum``, with the **positive imaginary** normal-absorption term ``Upz``.

This ``A`` (= EMsoft ``DynMat``) is consumed directly by
:func:`backend.forward_sim.dynamical.depth_integral.depth_integrated_Lgh`
(EMsoft ``CalcLgh``), which eigen-decomposes it and rescales ``W/(2·kn)`` — that
math is unchanged; only the matrix it is fed here is the corrected one.

> **The historical ``π·U`` off-diagonal was wrong** and has been removed; the
> ``i·π·λ`` / real-``π`` prefactors belong to the *expm scattering-matrix* and
> ``GetDynMat`` mode-``'A'`` (Struc) conventions, NOT the CalcLgh eigen-integral
> the EBSD master uses (see ``tasks/forward_sim/lessons.md`` §"DynMat off-diagonal
> i*pi*U DIAGNOSIS REFUTED").

:func:`scattering_matrix` (``S(z) = expm(2πi·A·z)``) is a *separate* convention
used only by its own spike test, kept for the validated ``torch.matrix_exp``
primitive (SPIKE Task 1: complex + batched + CUDA support confirmed).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from ..crystal.structure_matrix import excitation_error

_PI = math.pi
_TWOPI = 2.0 * math.pi

# Default Bethe-perturbation-validity threshold τ for the weak-beam correction.
# A per-pair single-weak-beam contribution is kept only when its magnitude is
# ≤ τ·(the direct coupling it perturbs).  τ = 0.1 (≲10 %) is the standard
# "the correction must be small vs the direct term" perturbation bound — see
# the module/``build_A`` docstrings and ``tasks/forward_sim/lessons.md`` §iter-12.
_WEAK_PERT_TAU = 0.1


def _lookup_grid(
    Ug_lookup: Callable[[torch.Tensor], torch.Tensor], diffs_np: np.ndarray
) -> torch.Tensor:
    """Look up ``U`` over an array of difference vectors, reusing the cache.

    ``diffs_np`` is an ``(..., 3)`` int array of Miller difference vectors; the
    unique vectors are queried once through ``Ug_lookup`` (EMsoft ``cell%LUT``)
    and scattered back to the input shape as a ``complex128`` tensor of shape
    ``diffs_np.shape[:-1]``.
    """
    shape = diffs_np.shape[:-1]
    flat = diffs_np.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    u_uniq = Ug_lookup(
        torch.from_numpy(np.ascontiguousarray(uniq))
    ).to(torch.complex128)
    u_flat = u_uniq[torch.from_numpy(np.ascontiguousarray(inv.reshape(-1)))]
    return u_flat.reshape(*shape)


def _weak_chunk_bounds(w: int, weak_chunk: int):
    """Yield ``(w0, w1)`` half-open weak-axis chunk bounds.

    ``weak_chunk <= 0`` (or ``>= w``) yields a SINGLE chunk covering ``[0, w)`` —
    the unchunked path (so the result is bit-identical to the pre-chunking code).
    Otherwise yields contiguous chunks of ``weak_chunk`` beams.  Chunking only
    bounds the working-set memory of the ``off_terms`` tensor; the masked terms it
    accumulates are identical regardless of the chunk size (same per-term guard,
    same Σ_w).
    """
    if w <= 0:
        return
    step = w if weak_chunk <= 0 else min(int(weak_chunk), w)
    for w0 in range(0, w, step):
        yield w0, min(w0 + step, w)


def build_A(
    directions: torch.Tensor,
    strong_refl: torch.Tensor,
    Ug_lookup: Callable[[torch.Tensor], torch.Tensor],
    *,
    reciprocal_metric: np.ndarray,
    wavelength_nm: float,
    Upz: float = 0.0,
    weak_refl: torch.Tensor | None = None,
    weak_sg: torch.Tensor | None = None,
    foil_normal: torch.Tensor | None = None,
    weak_pert_tau: float = 0.1,
    weak_chunk: int = 0,
) -> torch.Tensor:
    """Assemble the ``(B, n, n)`` complex dynamical matrix ``A`` (GetDynMat ``'D'``).

    For each beam direction ``k`` (row of ``directions``) and each ordered pair of
    strong reflections ``(g_ir, g_ic)``::

        A[b, ir, ic] = U_{g_ir − g_ic}                                  (ir ≠ ic)
                       − (λ/2)·Σ_w U_{g_ir − w}·U_{w − g_ic}/s_w(k_b)
        A[b, ir, ir] = cmplx( 2·s_{g_ir}(k_b)/λ − weaksgsum_ir(k_b), Upz )
        weaksgsum_ir = (λ/2)·Σ_w |U_{g_ir − w}|²/s_w(k_b)

    where ``s_g`` is the EMsoft excitation error (see
    :func:`backend.forward_sim.crystal.structure_matrix.excitation_error`),
    ``U_{·}`` the Fourier coefficient of the complex crystal potential from
    ``Ug_lookup`` (EMsoft ``cell%LUT``), ``{w}`` the Bethe weak-beam set, and
    ``Upz`` the normal-absorption coefficient (= ``Im(U_0)``).

    The bare ``U_{g−h}`` off-diagonal block is direction-independent (the unique
    difference vectors are looked up once, cached); the per-direction
    excitation-error diagonal, and — when a weak set is supplied — the
    direction-dependent ``weaksum`` / ``weaksgsum`` corrections, are applied in a
    per-direction loop.

    Args:
        directions: ``(B, 3)`` (or ``(3,)`` for a single direction) incident
            wavevectors ``k`` in Miller/reciprocal coordinates, ``|k| = 1/λ``.
        strong_refl: ``(n, 3)`` Miller indices of the strong reflections (index 0
            is conventionally the transmitted beam ``g = 0``).
        Ug_lookup: callable mapping an ``(M, 3)`` int tensor of *difference* Miller
            indices to an ``(M,)`` complex tensor of ``U`` (EMsoft ``cell%LUT``).
        reciprocal_metric: ``(3, 3)`` reciprocal metric tensor ``g*`` (1/nm²).
        wavelength_nm: relativistic electron wavelength ``λ`` in nm.
        Upz: normal-absorption coefficient ``Im(U_0)`` (nm⁻²) placed on every
            diagonal imaginary part (EMsoft ``rlp%Upmod``).  Default ``0`` (no
            absorption — the bare structural matrix).
        weak_refl: optional ``(W, 3)`` Miller indices of the per-direction Bethe
            **weak** beams.  When ``None`` / empty, no weak-beam correction is
            applied (bare strong-beam matrix).
        weak_sg: excitation errors ``s_w`` (nm⁻¹) of the weak beams — shape
            ``(W,)`` (shared across the batch, the master-pattern per-direction
            call) or ``(B, W)`` (one row per direction).  Required when
            ``weak_refl`` is non-empty.
        foil_normal: optional ``(3,)`` foil normal; defaults per-direction to the
            beam direction (master-pattern convention, beam ∥ foil normal).
        weak_pert_tau: Bethe-**perturbation-validity** threshold ``τ`` for the
            weak-beam correction (default ``0.1`` = ON).  A single weak beam ``w``
            contributes its off-diagonal term ``(λ/2)·U_{g_ir−w}·U_{w−g_ic}/s_w``
            to ``weaksum[ir,ic]`` **only when** that term's magnitude does not
            exceed ``τ·|U_{g_ir−g_ic}|`` (the direct coupling it perturbs);
            likewise its diagonal term ``(λ/2)·|U_{g_ir−w}|²/s_w`` enters
            ``weaksgsum[ir]`` only when ≤ ``τ·|2·s_{g_ir}/λ|`` (the direct
            excitation-error term).  This drops the out-of-regime contributions a
            dense (fine-``dmin``) reflection list admits — many weak beams with
            small ``|s_w|`` whose collective ``weaksum`` leaves the perturbation
            regime and *over-corrects* the off-diagonal couplings (the iter-8
            full-res NCC regression).  When the direct term is zero (a forbidden
            ``|U_{g−h}|=0`` difference, or ``s_g≈0`` on the Ewald sphere) the
            term is kept (no direct coupling to be perturbative against — that is
            the genuine Umweganregung term, not an over-correction).  Set
            ``0.0`` to disable the guard and apply the raw EMsoft GetDynMat
            ``weaksum``/``weaksgsum`` over the full weak set (the pre-iter-12
            behaviour).
        weak_chunk: iter-13 OOM fix — chunk size (number of weak beams ``w``)
            for the per-pair weak-term reduction.  The guard (iter-12) keeps the
            per-(ir,ic,w) ``off_terms`` tensor explicit so it can drop
            out-of-regime contributions *before* the ``Σ_w`` reduction, but that
            ``(B, n, n, W)`` tensor blows up at fine ``dmin`` (npx=500/dmin=0.05:
            ~41 GiB → CUDA OOM).  ``weak_chunk > 0`` processes the weak axis in
            chunks of that many beams, applying the guard mask per chunk and
            accumulating the **masked** ``weaksum`` / ``weaksgsum`` incrementally
            — the full ``(B, n, n, W)`` tensor is never materialised.  ``0``
            (default) processes all ``W`` weak beams at once (the pre-iter-13
            path, unchanged for small weak sets).  Equivalent math; bit-faithful
            to the unchunked guard (a single chunk ``weak_chunk ≥ W`` is the
            unchunked path exactly; smaller chunks match to FP precision).

    Returns:
        ``(B, n, n)`` ``complex128`` dynamical matrix.

    Raises:
        ValueError: if ``wavelength_nm`` is not strictly positive,
            ``strong_refl`` / ``directions`` are not ``(*, 3)``-shaped, or
            ``weak_refl`` is given without ``weak_sg``.
    """
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be > 0, got {wavelength_nm!r}")

    dirs = directions
    if dirs.dim() == 1:
        dirs = dirs.unsqueeze(0)
    if dirs.dim() != 2 or dirs.shape[-1] != 3:
        raise ValueError(
            f"directions must be (B, 3) or (3,), got {tuple(directions.shape)}"
        )
    if strong_refl.dim() != 2 or strong_refl.shape[-1] != 3:
        raise ValueError(
            f"strong_refl must be (n, 3), got {tuple(strong_refl.shape)}"
        )

    n = strong_refl.shape[0]
    b = dirs.shape[0]
    lam = float(wavelength_nm)
    half_lam = 0.5 * lam

    # --- Off-diagonal block: plain U_{g−h}, direction-independent ---------------
    # Difference vectors g_ir − g_ic for all ordered pairs (n, n, 3); the diagonal
    # (ir == ic) entries are placeholders (U_0) overwritten below.
    hkl = np.asarray(strong_refl.detach().cpu(), dtype=np.int64).reshape(-1, 3)
    diffs = hkl[:, None, :] - hkl[None, :, :]            # (n, n, 3)
    u_grid = _lookup_grid(Ug_lookup, diffs)              # (n, n) complex128

    # Off-diagonal A[ir,ic] = U_{g_ir − g_ic}; broadcast over the B directions.
    A = u_grid.unsqueeze(0).expand(b, n, n).clone()      # (B, n, n) complex128

    # --- Bethe weak-beam lookups (direction-independent U parts) ----------------
    has_weak = (
        weak_refl is not None
        and weak_refl.dim() == 2
        and weak_refl.shape[0] > 0
    )
    if has_weak:
        if weak_sg is None:
            raise ValueError("weak_sg is required when weak_refl is non-empty")
        w_hkl = np.asarray(weak_refl.detach().cpu(), dtype=np.int64).reshape(-1, 3)
        w = w_hkl.shape[0]
        # Ugw[ir, w] = U_{g_ir − w}   (n, W)
        ugw = _lookup_grid(
            Ug_lookup, hkl[:, None, :] - w_hkl[None, :, :]
        )                                                # (n, W) complex128
        # Uwh[w, ic] = U_{w − g_ic}   (W, n)
        uwh = _lookup_grid(
            Ug_lookup, w_hkl[:, None, :] - hkl[None, :, :]
        )                                                # (W, n) complex128
        ugw_abs2 = (ugw.real * ugw.real + ugw.imag * ugw.imag)  # (n, W) float64
        weak_sg_t = weak_sg.detach().cpu().to(torch.float64)
        if weak_sg_t.dim() not in (1, 2):
            raise ValueError(
                f"weak_sg must be (W,) or (B, W), got {tuple(weak_sg.shape)}"
            )

    eye = torch.eye(n, dtype=torch.bool)
    fn = None if foil_normal is None else foil_normal
    upz_col = torch.full((n,), float(Upz), dtype=torch.float64)

    # --- Per-direction diagonal (+ weak-beam corrections) -----------------------
    for bi in range(b):
        sg = excitation_error(
            strong_refl, dirs[bi], reciprocal_metric, foil_normal=fn
        ).to(torch.float64)                              # (n,)
        diag_real = 2.0 * sg / lam                       # (n,) float64

        if has_weak:
            sw = weak_sg_t if weak_sg_t.dim() == 1 else weak_sg_t[bi]   # (W,)
            # 1/s_w with a guard for an (unphysical) on-sphere weak beam s_w==0.
            inv_sw = torch.where(
                sw != 0.0, 1.0 / sw, torch.zeros_like(sw)
            )                                            # (W,) float64
            inv_sw_c = inv_sw.to(torch.complex128)

            # --- Off-diagonal weak correction (weaksum) ------------------------
            #   weaksum[ir,ic] = Σ_w U_{g_ir−w}·U_{w−g_ic}/s_w
            # Per-(ir,ic,w) single-weak-beam term (kept explicit so the
            # perturbation-validity guard can drop individual out-of-regime
            # contributions before the Σ_w reduction) — but reduced over the weak
            # axis in CHUNKS so the full (n, n, W) tensor is never materialised
            # (iter-13 OOM fix).  The guard mask is applied per chunk; the masked
            # weaksum accumulates incrementally.
            u_dir = torch.abs(u_grid)                    # (n, n) float64
            thresh = float(weak_pert_tau) * u_dir.unsqueeze(-1)  # (n, n, 1)
            weaksum = torch.zeros((n, n), dtype=torch.complex128)
            for w0, w1 in _weak_chunk_bounds(w, weak_chunk):
                off_terms = (
                    ugw[:, w0:w1].unsqueeze(1)               # (n, 1, c)
                    * inv_sw_c[w0:w1].reshape(1, 1, -1)      # (1, 1, c)
                    * uwh[w0:w1].transpose(0, 1).unsqueeze(0)  # (1, n, c)
                )                                            # (n, n, c) complex128
                if weak_pert_tau > 0.0:
                    # Keep term w only if |(λ/2)·term| ≤ τ·|U_{g_ir−g_ic}| (the
                    # direct coupling).  Zero direct coupling → keep (genuine
                    # Umweganregung).
                    term_mag = half_lam * torch.abs(off_terms)       # (n, n, c)
                    keep = (term_mag <= thresh) | (u_dir.unsqueeze(-1) == 0.0)
                    off_terms = torch.where(
                        keep, off_terms, torch.zeros_like(off_terms)
                    )
                weaksum = weaksum + off_terms.sum(dim=-1)    # (n, n) complex128
            A[bi] = u_grid - half_lam * weaksum

            # --- Diagonal weak correction (weaksgsum) --------------------------
            #   weaksgsum[ir] = (λ/2)·Σ_w |U_{g_ir−w}|²/s_w
            # Chunked over w as well (the (n, W) diag tensor is small, but keep the
            # accumulation aligned with the off-diagonal chunking for consistency).
            d_dir = torch.abs(diag_real)                 # (n,) float64
            d_thresh = float(weak_pert_tau) * d_dir.unsqueeze(-1)  # (n, 1)
            weaksgsum = torch.zeros((n,), dtype=torch.float64)
            for w0, w1 in _weak_chunk_bounds(w, weak_chunk):
                diag_terms = ugw_abs2[:, w0:w1] * inv_sw[w0:w1].unsqueeze(0)  # (n, c)
                if weak_pert_tau > 0.0:
                    # Keep term w only if |(λ/2)·term| ≤ τ·|2·s_g/λ| (the direct
                    # diagonal magnitude).  Zero direct term (s_g≈0) → keep.
                    d_mag = half_lam * torch.abs(diag_terms)             # (n, c)
                    d_keep = (d_mag <= d_thresh) | (d_dir.unsqueeze(-1) == 0.0)
                    diag_terms = torch.where(
                        d_keep, diag_terms, torch.zeros_like(diag_terms)
                    )
                weaksgsum = weaksgsum + diag_terms.sum(dim=1)  # (n,) float64
            weaksgsum = half_lam * weaksgsum             # (n,) float64
            diag_real = diag_real - weaksgsum            # (n,) float64

        diag = torch.complex(diag_real, upz_col).to(torch.complex128)
        A[bi][eye] = diag

    return A


def _lookup_grid_global(
    Ug_lookup: Callable[[torch.Tensor], torch.Tensor], diffs_np: np.ndarray
) -> np.ndarray:
    """Look up ``U`` over an ``(..., 3)`` int difference array → ``complex128`` numpy.

    Like :func:`_lookup_grid` but returns a **numpy** ``complex128`` array of shape
    ``diffs_np.shape[:-1]`` and reduces the ENTIRE (possibly large, multi-direction)
    difference array to its unique vectors in a single ``np.unique`` + one batched
    ``Ug_lookup`` call.  Used by :func:`build_A_batched` to look up the
    ``(B', n, n)`` / ``(B', n, W)`` difference grids of a whole direction group at
    once (the unique difference vectors are heavily shared across directions).
    """
    shape = diffs_np.shape[:-1]
    flat = diffs_np.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    u_uniq = (
        Ug_lookup(torch.from_numpy(np.ascontiguousarray(uniq)))
        .to(torch.complex128)
        .detach()
        .cpu()
        .numpy()
    )
    return u_uniq[inv.reshape(-1)].reshape(*shape)


def build_A_batched(
    directions: torch.Tensor,
    strong_hkl: torch.Tensor,
    Ug_lookup: Callable[[torch.Tensor], torch.Tensor],
    *,
    reciprocal_metric: np.ndarray,
    wavelength_nm: float,
    Upz: float = 0.0,
    weak_hkl: torch.Tensor | None = None,
    weak_mask: torch.Tensor | None = None,
    weak_sg: torch.Tensor | None = None,
    weak_pert_tau: float = 0.1,
    weak_chunk: int = 0,
) -> torch.Tensor:
    """Batched ``(B, n, n)`` dynamical matrix ``A`` for a group of SAME-``n`` directions.

    This is the **SP4 lever-3** vectorised counterpart of :func:`build_A`.  Unlike
    :func:`build_A` (which broadcasts ONE direction-independent strong set over a
    direction batch and loops over directions for the diagonal/weak corrections),
    here every direction ``b`` has its **own** ``n`` strong reflections
    ``strong_hkl[b]`` and its own (zero-padded) weak set ``weak_hkl[b]`` — the
    ragged-by-direction Bethe partition, grouped by strong-beam count ``n``.

    The assembled matrix is **term-for-term identical** to calling :func:`build_A`
    once per direction (GetDynMat mode ``'D'``)::

        A[b, ir, ic] = U_{g_ir − g_ic}                                  (ir ≠ ic)
                       − (λ/2)·Σ_w U_{g_ir − w}·U_{w − g_ic}/s_w
        A[b, ir, ir] = cmplx( 2·s_{g_ir}/λ − weaksgsum_ir,  Upz )
        weaksgsum_ir = (λ/2)·Σ_w |U_{g_ir − w}|²/s_w

    with the per-direction strong set ``g`` = ``strong_hkl[b]`` and weak set
    ``w`` = ``weak_hkl[b]`` (padded rows masked out via ``weak_mask`` →
    ``1/s_w = 0`` so they contribute nothing — exactly as an empty/shorter weak
    set would in the serial path).

    Args:
        directions: ``(B, 3)`` incident wavevectors ``k`` (Miller/reciprocal,
            ``|k| = 1/λ``).
        strong_hkl: ``(B, n, 3)`` int Miller indices of the per-direction strong
            reflections; index 0 along ``n`` is the transmitted beam ``g = 0``.
        Ug_lookup: callable ``(N, 3) int → (N,) complex`` (EMsoft ``cell%LUT``).
        reciprocal_metric: ``(3, 3)`` reciprocal metric ``g*`` (1/nm²).
        wavelength_nm: relativistic electron wavelength ``λ`` (nm).
        Upz: normal-absorption coefficient ``Im(U_0)`` on every diagonal (nm⁻²).
        weak_hkl: optional ``(B, W, 3)`` int Miller indices of the per-direction
            Bethe weak beams, **zero-padded** to the group-max ``W``.  ``None`` →
            no weak-beam correction (bare strong matrix).
        weak_mask: ``(B, W)`` bool — True for the real (non-pad) weak entries.
            Required when ``weak_hkl`` is given.
        weak_sg: ``(B, W)`` float — signed excitation errors ``s_w`` of the weak
            beams (pad entries arbitrary; masked out).  Required when ``weak_hkl``
            is given.
        weak_pert_tau: Bethe-perturbation-validity threshold ``τ`` (default
            ``0.1`` = ON) — identical semantics to :func:`build_A` (drop a weak
            beam's per-pair ``weaksum``/``weaksgsum`` term when its magnitude
            exceeds ``τ`` × the direct coupling it perturbs; keep when the direct
            term is zero).  ``0.0`` disables the guard.  Applied per-term so this
            stays **bit-faithful** to :func:`build_A` (same τ, same kept terms).
        weak_chunk: iter-13 OOM fix — chunk size (number of weak beams ``w``) for
            the per-pair weak-term reduction.  The guard keeps the per-(b,ir,ic,w)
            ``off_terms`` tensor explicit before the ``Σ_w`` reduction; at fine
            ``dmin`` that ``(B, n, n, W)`` tensor OOMs (npx=500/dmin=0.05:
            ~41 GiB).  ``weak_chunk > 0`` reduces the weak axis in chunks of that
            many beams — applying the guard mask per chunk and accumulating the
            masked ``weaksum`` / ``weaksgsum`` incrementally — so the full
            ``(B, n, n, W)`` tensor is never materialised.  ``0`` (default)
            reduces all ``W`` at once (pre-iter-13 path).  Bit-faithful to the
            unchunked guard (``weak_chunk ≥ W`` is the unchunked path exactly).

    Returns:
        ``(B, n, n)`` ``complex128`` dynamical matrix (CPU).

    Raises:
        ValueError: if shapes are inconsistent or ``wavelength_nm`` ≤ 0.
    """
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be > 0, got {wavelength_nm!r}")
    if strong_hkl.dim() != 3 or strong_hkl.shape[-1] != 3:
        raise ValueError(
            f"strong_hkl must be (B, n, 3), got {tuple(strong_hkl.shape)}"
        )
    dirs = directions
    if dirs.dim() != 2 or dirs.shape[-1] != 3:
        raise ValueError(
            f"directions must be (B, 3), got {tuple(directions.shape)}"
        )

    b = strong_hkl.shape[0]
    n = strong_hkl.shape[1]
    if dirs.shape[0] != b:
        raise ValueError(
            f"directions B={dirs.shape[0]} != strong_hkl B={b}"
        )

    lam = float(wavelength_nm)
    half_lam = 0.5 * lam
    gstar = np.asarray(reciprocal_metric, dtype=np.float64)

    hkl = np.asarray(strong_hkl.detach().cpu(), dtype=np.int64).reshape(b, n, 3)
    k_np = np.asarray(dirs.detach().cpu(), dtype=np.float64).reshape(b, 3)

    # --- Off-diagonal block: U_{g_ir − g_ic} per direction (B, n, n) -----------
    # diffs[b, ir, ic] = g_{b,ir} − g_{b,ic}.
    diffs = hkl[:, :, None, :] - hkl[:, None, :, :]          # (B, n, n, 3)
    u_grid = _lookup_grid_global(Ug_lookup, diffs)           # (B, n, n) complex128 np
    A = torch.from_numpy(np.ascontiguousarray(u_grid)).to(torch.complex128)

    # --- Batched excitation error s_g for the diagonal (B, n) -------------------
    # Same math as excitation_error(), foil normal = beam direction (master conv).
    g_b = hkl.astype(np.float64)                             # (B, n, 3)
    kpg = k_np[:, None, :] + g_b                             # (B, n, 3)  k + g
    tkpg = 2.0 * k_np[:, None, :] + g_b                      # (B, n, 3)  2k + g
    xnom = -np.einsum("bni,ij,bnj->bn", g_b, gstar, tkpg)    # (B, n)
    fn_len = np.sqrt(np.einsum("bi,ij,bj->b", k_np, gstar, k_np))  # (B,)
    kpg_dot_fn = np.einsum("bni,ij,bj->bn", kpg, gstar, k_np)      # (B, n)
    xden = 2.0 * kpg_dot_fn / fn_len[:, None]               # (B, n)
    sg = np.zeros((b, n), dtype=np.float64)
    nz = np.abs(xden) > 0.0
    sg[nz] = xnom[nz] / xden[nz]
    diag_real = 2.0 * sg / lam                              # (B, n) float64

    # --- Weak-beam corrections (weaksum off-diag overwrite + weaksgsum diag) ----
    has_weak = (
        weak_hkl is not None
        and weak_hkl.dim() == 3
        and weak_hkl.shape[1] > 0
    )
    if has_weak:
        if weak_mask is None or weak_sg is None:
            raise ValueError(
                "weak_mask and weak_sg are required when weak_hkl is given"
            )
        w_hkl = np.asarray(weak_hkl.detach().cpu(), dtype=np.int64).reshape(
            b, -1, 3
        )
        w = w_hkl.shape[1]
        wmask = np.asarray(weak_mask.detach().cpu(), dtype=bool).reshape(b, w)
        wsg = np.asarray(weak_sg.detach().cpu(), dtype=np.float64).reshape(b, w)
        # 1/s_w, zeroed where padded OR where (unphysical) s_w == 0 — both make
        # the corresponding weak beam contribute nothing (matches serial guard).
        safe = (wmask) & (wsg != 0.0)
        inv_sw = np.zeros((b, w), dtype=np.float64)
        inv_sw[safe] = 1.0 / wsg[safe]                             # (B, W)

        # Ugw[b, ir, w] = U_{g_{b,ir} − w_{b,w}}   (B, n, W)
        diffs_gw = hkl[:, :, None, :] - w_hkl[:, None, :, :]        # (B, n, W, 3)
        ugw = torch.from_numpy(
            np.ascontiguousarray(_lookup_grid_global(Ug_lookup, diffs_gw))
        ).to(torch.complex128)                                     # (B, n, W)
        # Uwh[b, w, ic] = U_{w_{b,w} − g_{b,ic}}   (B, W, n)
        diffs_wh = w_hkl[:, :, None, :] - hkl[:, None, :, :]        # (B, W, n, 3)
        uwh = torch.from_numpy(
            np.ascontiguousarray(_lookup_grid_global(Ug_lookup, diffs_wh))
        ).to(torch.complex128)                                     # (B, W, n)

        inv_sw_c = torch.from_numpy(np.ascontiguousarray(inv_sw)).to(
            torch.complex128
        )                                                          # (B, W)
        inv_sw_t = torch.from_numpy(np.ascontiguousarray(inv_sw)).to(
            torch.float64
        )                                                          # (B, W)
        ugw_abs2 = (ugw.real * ugw.real + ugw.imag * ugw.imag)     # (B, n, W) f64

        # --- Off-diagonal weak correction (weaksum) ---------------------------
        # Per-(b,ir,ic,w) single-weak-beam term so the perturbation-validity
        # guard can drop out-of-regime contributions before the Σ_w reduction —
        # reduced over the weak axis in CHUNKS so the full (B, n, n, W) tensor is
        # never materialised (iter-13 OOM fix; ~41 GiB at npx=500/dmin=0.05).
        #   off_terms[b,ir,ic,w] = U_{g_ir−w}·U_{w−g_ic}/s_w
        u_dir = torch.abs(A)                                      # (B, n, n) f64
        thresh = float(weak_pert_tau) * u_dir.unsqueeze(-1)       # (B, n, n, 1)
        uwh_t = uwh.transpose(1, 2)                               # (B, n, W) (=Uwh[w→ic])
        weaksum = torch.zeros((b, n, n), dtype=torch.complex128)
        for w0, w1 in _weak_chunk_bounds(w, weak_chunk):
            off_terms = (
                ugw[:, :, w0:w1].unsqueeze(2)                     # (B, n, 1, c)
                * inv_sw_c[:, w0:w1].reshape(b, 1, 1, w1 - w0)    # (B, 1, 1, c)
                * uwh_t[:, :, w0:w1].unsqueeze(1)                 # (B, 1, n, c)
            )                                                     # (B, n, n, c)
            if weak_pert_tau > 0.0:
                term_mag = half_lam * torch.abs(off_terms)       # (B, n, n, c)
                keep = (term_mag <= thresh) | (u_dir.unsqueeze(-1) == 0.0)
                off_terms = torch.where(
                    keep, off_terms, torch.zeros_like(off_terms)
                )
            weaksum = weaksum + off_terms.sum(dim=-1)            # (B, n, n)
        A = A - half_lam * weaksum

        # --- Diagonal weak correction (weaksgsum) -----------------------------
        #   weaksgsum[b, ir] = (λ/2)·Σ_w |U_{g_ir−w}|²/s_w  (chunked over w too)
        d_dir = np.abs(diag_real)                                # (B, n) np f64
        d_dir_t = torch.from_numpy(np.ascontiguousarray(d_dir))
        d_thresh = float(weak_pert_tau) * d_dir_t.unsqueeze(-1)   # (B, n, 1)
        weaksgsum = torch.zeros((b, n), dtype=torch.float64)
        for w0, w1 in _weak_chunk_bounds(w, weak_chunk):
            diag_terms = ugw_abs2[:, :, w0:w1] * inv_sw_t[:, w0:w1].unsqueeze(1)  # (B, n, c)
            if weak_pert_tau > 0.0:
                d_mag = half_lam * torch.abs(diag_terms)         # (B, n, c)
                d_keep = (d_mag <= d_thresh) | (d_dir_t.unsqueeze(-1) == 0.0)
                diag_terms = torch.where(
                    d_keep, diag_terms, torch.zeros_like(diag_terms)
                )
            weaksgsum = weaksgsum + diag_terms.sum(dim=-1)       # (B, n)
        weaksgsum = half_lam * weaksgsum                         # (B, n)
        diag_real = diag_real - weaksgsum.detach().cpu().numpy()  # (B, n)

    # --- Write the diagonal: cmplx(2·s_g/λ − weaksgsum, Upz) --------------------
    diag = torch.complex(
        torch.from_numpy(np.ascontiguousarray(diag_real)).to(torch.float64),
        torch.full((b, n), float(Upz), dtype=torch.float64),
    ).to(torch.complex128)                                         # (B, n)
    eye = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(b, n, n)
    A = A.clone()
    A[eye] = diag.reshape(-1)
    return A


# ---------------------------------------------------------------------------
# SP4 iter-5 — GPU build_A: dense difference-LUT gather (no np.unique/argsort)
# ---------------------------------------------------------------------------
#
# Profiling (SP4 iter-4 → iter-5) showed ``build_A_batched`` is the wall-time
# bottleneck (~70–80 %), and **92 % of its own time is ``np.unique``'s
# ``argsort``** over the ``(B', n, n, 3)`` / ``(B', n, W, 3)`` difference grids
# (deduplicating before the dict ``Ug_lookup``).  At npx=500 those grids are
# enormous, so the CPU sort dominates.
#
# The fix: precompute a **dense complex difference-LUT** once per master — a
# ``(D, D, D)`` table indexed by the *offset* difference vector ``g − h`` (all
# such vectors lie in the box ``[−2·a_max, 2·a_max]³`` where ``a_max`` is the
# largest abs Miller index in the reflection list).  The table is tiny (≤ 0.6 MB
# complex128 even at dmin=0.05) and lives on the GPU.  Then the off-diagonal /
# weak-beam ``U`` gathers become **pure GPU index ops** (a flat-index gather into
# the LUT) — no ``np.unique``, no ``argsort``, no per-call dict lookups — and the
# whole ``(B', n, n)`` assembly (off-diagonal, diagonal excitation error, the
# weak-beam ``weaksum``/``weaksgsum`` corrections) runs as batched CUDA tensor
# ops.  Bit-faithful: the LUT holds exactly the same ``U_{g−h}`` the dict path
# returns, and the assembly math is term-for-term identical to ``build_A_batched``.


def build_diff_lut(
    reflections: torch.Tensor,
    Ug_lookup: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.complex128,
) -> tuple[torch.Tensor, int]:
    """Dense complex difference-LUT ``U_{g−h}`` for GPU gather (SP4 iter-5).

    Builds, ONCE for the whole reflection list, a dense ``(D, D, D)`` complex
    table ``lut`` with ``D = 2·H + 1`` and ``H = 2·max|index|``, such that

        lut[dh + H, dk + H, dl + H] = U_{(dh, dk, dl)}

    for every difference vector ``(dh, dk, dl)`` that can arise as ``g − h``
    between two reflections (they all lie in ``[−H, H]³``).  The table is queried
    from ``Ug_lookup`` (EMsoft ``cell%LUT``) over the **full box** in a single
    batched call (≤ 0.6 MB complex128 even at dmin=0.05), then moved to ``device``.

    Args:
        reflections: ``(M, 3)`` int Miller indices (defines the box via the max
            abs index).  Index 0 is the transmitted beam ``(0,0,0)``.
        Ug_lookup: callable ``(N, 3) int → (N,) complex`` (EMsoft ``cell%LUT``).
        device: target device for the returned LUT (e.g. ``cuda``).
        dtype: LUT complex dtype (``complex128`` default; ``complex64`` halves
            the gather/assembly memory traffic — the caller promotes back as
            needed).

    Returns:
        ``(lut, H)`` where ``lut`` is the ``(D, D, D)`` complex tensor on
        ``device`` and ``H`` the half-box offset (a difference ``d`` maps to
        ``lut[d + H]``).
    """
    hkl = np.asarray(reflections.detach().cpu(), dtype=np.int64).reshape(-1, 3)
    a_max = int(np.abs(hkl).max()) if hkl.size else 0
    H = 2 * a_max
    D = 2 * H + 1

    # Lever 2b-1c: query ``Ug_lookup`` only for the difference vectors that can
    # ACTUALLY occur as ``g − h`` between two reflections (the unique ``hkl[i] −
    # hkl[j]`` set), NOT every cell of the dense ``[−H, H]³`` box.  Every gather
    # into the box (build_A off-diagonal / weak; build_ug_lut row-maxima) addresses
    # a difference of two listed reflections, so any never-occurring box cell is
    # NEVER read — leaving it 0 is bit-faithful.  The dense box covered |g| up to
    # ``2·max_index·|b*|`` (~46 nm⁻¹ on tau2), but real differences only reach
    # ~2× the reflection-list radius (~20 nm⁻¹); the dense box therefore drove the
    # scipy phonon integral (``compute_Ug_table``) over ~4× more unique ``|g|``
    # than occur (tau2: 3135 vs 799), and that physics cost was the residual SETUP
    # wall once the (M,M) Sgh / np.unique(M²) costs were removed.  Restricting to
    # the real differences cuts it ~4× — identical gathered values, proven Δ = 0.0.
    if hkl.size:
        # Reachable box cells = unique FLAT index of every difference d = g_i − g_j.
        # We dedup on the 1-D flat int index (a cheap contiguous-int unique), NOT a
        # row-sort ``np.unique((M², 3), axis=0)`` — the latter is the very cost
        # Lever 2b-1 removed (tau2 ~2.6 s).  The ``g_i − hkl`` rows are formed in
        # CHUNKS so the working set never exceeds ``row_chunk × M`` int64 (the full
        # (M, M, 3) diff array would be 3.4 GB on a T-phase cell); each chunk's flat
        # indices are deduped incrementally via a running unique.  Recover each
        # unique cell's (h, k, l) from its flat index for one batched ``Ug_lookup``.
        DD = D * D
        M = hkl.shape[0]
        seen = np.zeros(0, dtype=np.int64)
        # Bound the per-chunk (row_chunk, M, 3) working set to ~2M int triples.
        _row_chunk = max(1, min(M, 2_000_000 // max(1, M)))
        for i0 in range(0, M, _row_chunk):
            rows = hkl[i0:i0 + _row_chunk]                    # (B, 3)
            d = rows[:, None, :] - hkl[None, :, :]            # (B, M, 3)
            di = d + H
            fl = ((di[..., 0] * D + di[..., 1]) * D + di[..., 2]).reshape(-1)
            seen = np.union1d(seen, np.unique(fl))            # running 1-D dedup
        flat_uniq = seen
        oh = flat_uniq // DD
        ok = (flat_uniq // D) % D
        ol = flat_uniq % D
        uniq = np.stack([oh - H, ok - H, ol - H], axis=1).astype(np.int64)
    else:
        flat_uniq = np.zeros(1, dtype=np.int64)
        uniq = np.zeros((1, 3), dtype=np.int64)              # just (0,0,0)
    u = (
        Ug_lookup(torch.from_numpy(np.ascontiguousarray(uniq)))
        .to(torch.complex128)
        .detach()
        .cpu()
    )
    # Scatter the computed U into a zero box at the offset index (d + H).  Cells
    # without a real difference stay 0 (never gathered → bit-faithful).
    lut = torch.zeros(D * D * D, dtype=torch.complex128)
    lut[torch.from_numpy(np.ascontiguousarray(flat_uniq))] = u
    lut = lut.reshape(D, D, D).to(device=device, dtype=dtype)
    return lut, H


def _gather_lut(
    lut: torch.Tensor, H: int, diffs: torch.Tensor
) -> torch.Tensor:
    """Gather ``U`` from a dense difference-LUT for an ``(..., 3)`` int diff tensor.

    ``diffs`` are integer difference vectors ``g − h`` on the LUT's device; each
    maps to ``lut[d + H]``.  Returns a complex tensor of shape ``diffs.shape[:-1]``
    via a single flat-index gather (pure device op, no host round-trip).
    """
    D = lut.shape[-1]
    idx = (diffs + H).to(torch.long)                       # (..., 3)
    flat = (idx[..., 0] * D + idx[..., 1]) * D + idx[..., 2]  # (...,)
    return lut.reshape(-1).index_select(0, flat.reshape(-1)).reshape(flat.shape)


def build_A_batched_gpu(
    directions: torch.Tensor,
    strong_hkl: torch.Tensor,
    lut: torch.Tensor,
    H: int,
    *,
    reciprocal_metric: np.ndarray,
    wavelength_nm: float,
    Upz: float = 0.0,
    weak_hkl: torch.Tensor | None = None,
    weak_mask: torch.Tensor | None = None,
    weak_sg: torch.Tensor | None = None,
    weak_pert_tau: float = 0.1,
    weak_chunk: int = 0,
) -> torch.Tensor:
    """GPU ``(B, n, n)`` dynamical matrix ``A`` via a dense difference-LUT gather.

    The SP4 **iter-5** GPU counterpart of :func:`build_A_batched`.  Identical
    GetDynMat mode-``'D'`` math (term-for-term), but every ``U_{g−h}`` lookup is a
    **dense-LUT gather** (:func:`_gather_lut`) on ``directions.device`` instead of a
    CPU ``np.unique`` + dict query, and the excitation error, ``weaksum`` /
    ``weaksgsum`` corrections, and diagonal are all batched tensor ops on the
    device.  This removes the ``np.unique``/``argsort`` CPU hot spot (~92 % of the
    old ``build_A_batched`` time) and runs the assembly on the GPU.

    The assembled matrix is **term-for-term identical** to :func:`build_A_batched`
    (and hence to :func:`build_A` per direction)::

        A[b, ir, ic] = U_{g_ir − g_ic}                                  (ir ≠ ic)
                       − (λ/2)·Σ_w U_{g_ir − w}·U_{w − g_ic}/s_w
        A[b, ir, ir] = cmplx( 2·s_{g_ir}/λ − weaksgsum_ir,  Upz )
        weaksgsum_ir = (λ/2)·Σ_w |U_{g_ir − w}|²/s_w

    Args:
        directions: ``(B, 3)`` incident wavevectors ``k`` (Miller/reciprocal,
            ``|k| = 1/λ``) on the compute device (float64).
        strong_hkl: ``(B, n, 3)`` int Miller indices of the per-direction strong
            reflections on the device; index 0 along ``n`` is ``g = 0``.
        lut: ``(D, D, D)`` complex difference-LUT from :func:`build_diff_lut`
            (on the same device as ``directions``).
        H: the LUT half-box offset (a difference ``d`` is at ``lut[d + H]``).
        reciprocal_metric: ``(3, 3)`` reciprocal metric ``g*`` (1/nm²).
        wavelength_nm: relativistic electron wavelength ``λ`` (nm).
        Upz: normal-absorption coefficient ``Im(U_0)`` on every diagonal (nm⁻²).
        weak_hkl: optional ``(B, W, 3)`` int weak-beam Miller indices, **zero-
            padded** to the group-max ``W`` (on the device).  ``None`` → no weak
            correction.
        weak_mask: ``(B, W)`` bool — True for the real (non-pad) weak entries.
            Required when ``weak_hkl`` is given.
        weak_sg: ``(B, W)`` float — signed excitation errors ``s_w`` (pad arbitrary;
            masked out).  Required when ``weak_hkl`` is given.
        weak_pert_tau: Bethe-perturbation-validity threshold ``τ`` (default
            ``0.1`` = ON) — identical semantics to :func:`build_A` /
            :func:`build_A_batched` (drop a weak beam's per-pair
            ``weaksum``/``weaksgsum`` term when its magnitude exceeds ``τ`` × the
            direct coupling; keep when the direct term is zero).  Applied
            per-term on-device so this stays **bit-faithful** to
            :func:`build_A_batched`.  ``0.0`` disables the guard.
        weak_chunk: iter-13 OOM fix — chunk size (number of weak beams ``w``) for
            the per-pair weak-term reduction (the on-device ``(B, n, n, W)``
            ``off_terms`` tensor that OOM'd the 12 GB RTX 4070 at npx=500/dmin=0.05,
            ~41 GiB).  ``weak_chunk > 0`` reduces the weak axis in chunks of that
            many beams — applying the guard mask per chunk and accumulating the
            masked ``weaksum`` / ``weaksgsum`` incrementally on-device — so the
            full ``(B, n, n, W)`` tensor is never materialised.  ``0`` (default)
            reduces all ``W`` at once (pre-iter-13 path).  Bit-faithful to the
            unchunked guard (``weak_chunk ≥ W`` is the unchunked path exactly).

    Returns:
        ``(B, n, n)`` complex dynamical matrix (same dtype as ``lut``,
        ``complex128`` recommended) on ``directions.device``.

    Raises:
        ValueError: if shapes are inconsistent or ``wavelength_nm`` ≤ 0.
    """
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be > 0, got {wavelength_nm!r}")
    if strong_hkl.dim() != 3 or strong_hkl.shape[-1] != 3:
        raise ValueError(
            f"strong_hkl must be (B, n, 3), got {tuple(strong_hkl.shape)}"
        )
    if directions.dim() != 2 or directions.shape[-1] != 3:
        raise ValueError(
            f"directions must be (B, 3), got {tuple(directions.shape)}"
        )
    b = strong_hkl.shape[0]
    n = strong_hkl.shape[1]
    if directions.shape[0] != b:
        raise ValueError(
            f"directions B={directions.shape[0]} != strong_hkl B={b}"
        )

    dev = directions.device
    cdtype = lut.dtype
    fdtype = torch.float64
    lam = float(wavelength_nm)
    half_lam = 0.5 * lam

    gstar = torch.as_tensor(
        np.asarray(reciprocal_metric, dtype=np.float64), dtype=fdtype, device=dev
    )                                                       # (3, 3)
    g = strong_hkl.to(dev).to(torch.long)                   # (B, n, 3)
    k = directions.to(dev).to(fdtype)                       # (B, 3)
    g_f = g.to(fdtype)                                       # (B, n, 3)

    # --- Off-diagonal block: U_{g_ir − g_ic} per direction (B, n, n) -----------
    diffs = g[:, :, None, :] - g[:, None, :, :]             # (B, n, n, 3)
    A = _gather_lut(lut, H, diffs).to(cdtype)               # (B, n, n)

    # --- Batched excitation error s_g for the diagonal (B, n) ------------------
    # Same math as excitation_error(); foil normal = beam direction (master conv).
    kpg = k[:, None, :] + g_f                               # (B, n, 3)  k + g
    tkpg = 2.0 * k[:, None, :] + g_f                        # (B, n, 3)  2k + g
    xnom = -torch.einsum("bni,ij,bnj->bn", g_f, gstar, tkpg)    # (B, n)
    fn_len = torch.sqrt(torch.einsum("bi,ij,bj->b", k, gstar, k))  # (B,)
    kpg_dot_fn = torch.einsum("bni,ij,bj->bn", kpg, gstar, k)      # (B, n)
    xden = 2.0 * kpg_dot_fn / fn_len[:, None]               # (B, n)
    sg = torch.where(
        xden.abs() > 0.0, xnom / xden, torch.zeros_like(xden)
    )                                                       # (B, n)
    diag_real = 2.0 * sg / lam                              # (B, n)

    # --- Weak-beam corrections (weaksum off-diag + weaksgsum diag) -------------
    has_weak = (
        weak_hkl is not None
        and weak_hkl.dim() == 3
        and weak_hkl.shape[1] > 0
    )
    if has_weak:
        if weak_mask is None or weak_sg is None:
            raise ValueError(
                "weak_mask and weak_sg are required when weak_hkl is given"
            )
        w_hkl = weak_hkl.to(dev).to(torch.long)             # (B, W, 3)
        w = w_hkl.shape[1]
        wmask = weak_mask.to(dev).to(torch.bool).reshape(b, w)
        wsg = weak_sg.to(dev).to(fdtype).reshape(b, w)
        # 1/s_w, zeroed where padded OR where (unphysical) s_w == 0 — matches the
        # serial guard (those weak beams contribute nothing).
        safe = wmask & (wsg != 0.0)
        inv_sw = torch.where(
            safe, 1.0 / wsg, torch.zeros_like(wsg)
        )                                                   # (B, W) float64

        # Ugw[b, ir, w] = U_{g_{b,ir} − w_{b,w}}   (B, n, W)
        diffs_gw = g[:, :, None, :] - w_hkl[:, None, :, :]  # (B, n, W, 3)
        ugw = _gather_lut(lut, H, diffs_gw).to(cdtype)      # (B, n, W)
        # Uwh[b, w, ic] = U_{w_{b,w} − g_{b,ic}}   (B, W, n)
        diffs_wh = w_hkl[:, :, None, :] - g[:, None, :, :]  # (B, W, n, 3)
        uwh = _gather_lut(lut, H, diffs_wh).to(cdtype)      # (B, W, n)

        inv_sw_c = inv_sw.to(cdtype)                        # (B, W)
        ugw_abs2 = (ugw.real * ugw.real + ugw.imag * ugw.imag)    # (B, n, W) f64

        # --- Off-diagonal weak correction (weaksum) ---------------------------
        # Per-(b,ir,ic,w) single-weak-beam term so the perturbation-validity
        # guard can drop out-of-regime contributions before the Σ_w reduction —
        # reduced over the weak axis in CHUNKS so the full (B, n, n, W) on-device
        # tensor is never materialised (iter-13 OOM fix; ~41 GiB on the 12 GB
        # RTX 4070 at npx=500/dmin=0.05).
        #   off_terms[b,ir,ic,w] = U_{g_ir−w}·U_{w−g_ic}/s_w
        u_dir = torch.abs(A)                                # (B, n, n) f
        thresh = float(weak_pert_tau) * u_dir.unsqueeze(-1)  # (B, n, n, 1)
        uwh_t = uwh.transpose(1, 2)                         # (B, n, W) (=Uwh[w→ic])
        weaksum = torch.zeros((b, n, n), dtype=cdtype, device=dev)
        for w0, w1 in _weak_chunk_bounds(w, weak_chunk):
            off_terms = (
                ugw[:, :, w0:w1].unsqueeze(2)               # (B, n, 1, c)
                * inv_sw_c[:, w0:w1].reshape(b, 1, 1, w1 - w0)  # (B, 1, 1, c)
                * uwh_t[:, :, w0:w1].unsqueeze(1)           # (B, 1, n, c)
            )                                               # (B, n, n, c)
            if weak_pert_tau > 0.0:
                term_mag = half_lam * torch.abs(off_terms)  # (B, n, n, c)
                keep = (term_mag <= thresh) | (u_dir.unsqueeze(-1) == 0.0)
                off_terms = torch.where(
                    keep, off_terms, torch.zeros_like(off_terms)
                )
            weaksum = weaksum + off_terms.sum(dim=-1)       # (B, n, n)
        A = A - half_lam * weaksum

        # --- Diagonal weak correction (weaksgsum) -----------------------------
        #   weaksgsum[b, ir] = (λ/2)·Σ_w |U_{g_ir−w}|²/s_w  (chunked over w too)
        d_dir = diag_real.abs()                             # (B, n)
        d_thresh = float(weak_pert_tau) * d_dir.unsqueeze(-1)  # (B, n, 1)
        weaksgsum = torch.zeros((b, n), dtype=fdtype, device=dev)
        for w0, w1 in _weak_chunk_bounds(w, weak_chunk):
            diag_terms = ugw_abs2[:, :, w0:w1] * inv_sw[:, w0:w1].unsqueeze(1)  # (B, n, c)
            if weak_pert_tau > 0.0:
                d_mag = half_lam * diag_terms.abs()         # (B, n, c)
                d_keep = (d_mag <= d_thresh) | (d_dir.unsqueeze(-1) == 0.0)
                diag_terms = torch.where(
                    d_keep, diag_terms, torch.zeros_like(diag_terms)
                )
            weaksgsum = weaksgsum + diag_terms.sum(dim=-1)  # (B, n)
        weaksgsum = half_lam * weaksgsum                    # (B, n)
        diag_real = diag_real - weaksgsum                  # (B, n)

    # --- Write the diagonal: cmplx(2·s_g/λ − weaksgsum, Upz) -------------------
    diag = torch.complex(
        diag_real.to(fdtype),
        torch.full((b, n), float(Upz), dtype=fdtype, device=dev),
    ).to(cdtype)                                            # (B, n)
    eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0).expand(b, n, n)
    A = A.clone()
    A[eye] = diag.reshape(-1)
    return A


def scattering_matrix(A: torch.Tensor, z: torch.Tensor | float) -> torch.Tensor:
    """Scattering matrix ``S(z) = expm(2πi · A · z)``.

    Exponentiates the dynamical matrix to the depth-``z`` scattering matrix via
    :func:`torch.matrix_exp` (the SPIKE-validated complex/batched primitive).

    .. note::

       This is a **distinct convention** (the ``S = expm(2πi·A·z)`` propagator)
       used only by its own spike test — the EBSD master path consumes ``build_A``
       through ``CalcLgh`` (eigen-decomposition), not through this exponential.
       See ``tasks/forward_sim/lessons.md`` §"DynMat off-diagonal … REFUTED".

    Args:
        A: ``(..., n, n)`` complex dynamical matrix (any leading batch dims).
        z: depth.  Either a Python scalar / 0-d tensor (applied to the whole
            batch) or a tensor broadcastable to ``A``'s leading batch dims (then
            unsqueezed over the trailing matrix axes).

    Returns:
        ``S`` with the same shape and complex dtype as ``A``.  At ``z = 0`` this
        is the identity matrix (``expm(0) = I``).
    """
    if not torch.is_tensor(z):
        z_t = torch.as_tensor(z, dtype=torch.float64, device=A.device)
    else:
        z_t = z.to(device=A.device)

    # Broadcast a per-batch z over the trailing (n, n) matrix axes.
    if z_t.dim() > 0:
        z_t = z_t.reshape(z_t.shape + (1, 1))

    phase = torch.tensor(_TWOPI, dtype=A.dtype, device=A.device) * 1j
    return torch.matrix_exp(phase * A * z_t.to(A.dtype))
