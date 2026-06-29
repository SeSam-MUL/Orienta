"""SP1 — depth-integrated ``Lgh`` via the EMsoft ``CalcLgh`` eigen-integral (Task 9).

The back-scattered EBSD yield integrates the dynamical intensity over depth,
weighted by the Monte-Carlo escape-depth distribution ``λ(z)``.  EMsoft's
``CalcLgh`` (``Source/EMsoftLib/MBmodule.f90``) evaluates this integral
**analytically** by eigen-decomposing the dynamical matrix, which is far more
numerically stable than naively summing ``S(z) = expm(2πi·A·z)`` over many depth
bins (no repeated matrix exponentials, no error accumulation, and the per-mode
decay ``exp(-qold·iz)`` is bounded by EMsoft's sign guard).  We implement the
**eigen-decomposition analytic depth integral** to match ``CalcLgh`` term-for-term.

Verified verbatim against EMsoft ``MBmodule.f90::CalcLgh`` (develop source,
2026-06-10) — see the task brief and ``tasks/forward_sim/lessons.md``:

* eigen-decompose the dynamical matrix ``A`` (= EMsoft ``DynMat``):
  ``W, CGG = eig(A)`` (eigenvalues ``W``, right eigenvectors as the **columns**
  of ``CGG``); ``CGinv = CGG^{-1}``;
* rescale the eigenvalues: ``W ← W / (2·kn)`` with ``kn`` the wavevector
  component along the foil normal.  For the master pattern the beam is parallel
  to the foil normal, so ``kn = |k| = 1/λ`` (nm⁻¹);
* per eigen-mode pair ``(j, k)``::

      qold = cmplx( tpi·(Im W_j + Im W_k),  tpi·(Re W_j − Re W_k) )   with tpi = 2π·depthstep
      if Re(qold) < 0:  qold = −qold                                  # EMsoft overflow guard
      q_{jk} = Σ_{iz=1}^{izz}  λ(iz) · exp(−qold · iz)
      Ijk_{jk} = conj(CGinv_{j,g0}) · q_{jk} · CGinv_{k,g0}

  then ``Ijk ← Ijk · (depthstep/thick)`` (``= Ijk / izz``, since
  ``thick = izz·depthstep``);
* assemble ``Lgh = conj(CGG) @ Ijk @ CGG^T``  (a plain **transpose**, *not* the
  Hermitian conjugate — this is exactly EMsoft's
  ``Lgh = matmul(matmul(conjg(CGG), Ijk), transpose(CGG))``).

``g0`` is the index of the transmitted/direct beam ``(0,0,0)`` in the strong-beam
list; by the SP0/SP1 convention that beam is index 0 of ``strong_refl`` and hence
of every row/column of ``A`` (the ``gzero`` argument of EMsoft ``CalcLgh``).

The 1-based ``iz`` loop is reproduced faithfully (``exp(-qold·iz)`` for
``iz = 1 .. izz``), matching EMsoft's depth-bin indexing.
"""
from __future__ import annotations

import math

import torch

from ..mc.emsoft_mc_input import MCData

_TWOPI = 2.0 * math.pi


def depth_integrated_Lgh(
    A: torch.Tensor,
    mc: MCData,
    energy_idx: int,
    *,
    wavelength_nm: float,
    gzero: int = 0,
    depth_weight: torch.Tensor | None = None,
    eig_dtype: torch.dtype = torch.complex128,
    use_flip_guard: bool = True,
) -> torch.Tensor:
    """Depth-integrated ``Lgh`` for a batch of dynamical matrices (EMsoft ``CalcLgh``).

    Computes, for each ``(B, n, n)`` dynamical matrix ``A`` in the batch, the
    ``(n, n)`` complex ``Lgh`` matrix that weights every strong-reflection pair by
    the depth-integrated dynamical intensity, using the eigen-decomposition
    analytic integral over the Monte-Carlo depth distribution ``λ(z)``.

    Args:
        A: ``(B, n, n)`` (or ``(n, n)`` for a single direction) complex dynamical
            matrix from :func:`backend.forward_sim.dynamical.scattering_matrix.build_A`.
            Index ``gzero`` (default 0) is the transmitted beam.
        mc: the EMsoft Monte-Carlo data; ``mc.lambda_z(energy_idx)`` supplies the
            normalised depth distribution ``λ(z)`` over the ``accum_z`` bins, and
            ``mc.depth_step`` the bin width (nm).
        energy_idx: index into the MC energy axis selecting the depth profile.
        wavelength_nm: relativistic electron wavelength ``λ`` (nm); ``kn = 1/λ``
            for the master-pattern (beam ∥ foil normal) eigenvalue rescaling.
        gzero: column/row index of the direct beam in ``A`` (default 0).
        depth_weight: optional ``(izz,)`` depth distribution to use in place of the
            pure histogram ``mc.lambda_z(energy_idx)`` — EMsoft ``EMEBSDmaster.f90``
            feeds ``CalcLgh`` the **absorption-weighted** ``lambdaE`` (= histogram ·
            ``exp(2π·(iz−1)·depthstep/xgp)``), which ``master_builder`` precomputes
            once per energy.  When ``None``, the pure histogram is used (the
            uniform-/geometry-sanity path and any direct caller).
        eig_dtype: precision for the eigendecomposition only — ``torch.complex128``
            (default, full double precision) or ``torch.complex64`` (single, the SP4
            iter-3 perf knob: roughly halves the dominant ``torch.linalg.eig`` cost).
            ``A`` is cast to ``eig_dtype`` *for the eig/inv step only*; the
            eigenvalues ``W`` and eigenvectors ``CGG`` are promoted back to
            ``complex128`` immediately afterward, so the depth integral and the
            ``Lgh = conj(CGG)·Ijk·CGGᵀ`` bilinear sum **accumulate in float64**
            regardless.  Gated on a correctness test (NCC ≥ 0.9999, max-rel-err <
            1e-3 vs the complex128 path) in ``master_builder``.
        use_flip_guard: EMsoft's overflow guard ``if Re(qold) < 0: qold = −qold``
            (default ``True`` — bit-faithful to ``CalcLgh``).  The guard fires only
            for the few eigen-mode pairs whose combined imaginary eigenvalue is
            negative (``Im W_j + Im W_k < 0``, i.e. an *unphysical anti-absorption*
            mode that the closed-form ``exp(−qold·iz)`` would otherwise grow without
            bound).  Set ``False`` to compute the **un-guarded** integral — which is
            exactly what the scattering-matrix propagation
            (:func:`depth_integrated_Lgh_propagate`) evaluates, so this no-flip path
            is the machine-precision parity oracle for that GPU route.  On real
            Ni/Al masters the guard changes the result by <2 % on ~1 % of pixels and
            does **not** improve fidelity vs the EMsoft oracle (NCC vs EMsoft is
            identical to 2e-5 with or without it — see ``tasks/forward_sim/lessons.md``
            §19), so the un-guarded integral is just as physically faithful.

    .. note::

       Keeping ``MCData.lambda_z`` the *pure* MC histogram and supplying the
       absorption factor through ``depth_weight`` mirrors EMsoft: ``accum_z`` is
       the raw histogram; the ``lambdaE`` absorption re-weighting lives in
       ``EMEBSDmaster.f90`` (here: ``master_builder``), not in the MC reader.

    Returns:
        ``(B, n, n)`` ``complex128`` ``Lgh``.

    Raises:
        ValueError: if ``wavelength_nm`` is not strictly positive, ``A`` is not
            ``(*, n, n)``-shaped, or ``gzero`` is out of range.
    """
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be > 0, got {wavelength_nm!r}")

    if A.dim() == 2:
        A = A.unsqueeze(0)
    if A.dim() != 3 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (B, n, n) or (n, n) square, got {tuple(A.shape)}"
        )

    n = A.shape[-1]
    if not 0 <= gzero < n:
        raise ValueError(f"gzero {gzero} out of range [0, {n})")

    if eig_dtype not in (torch.complex128, torch.complex64):
        raise ValueError(
            f"eig_dtype must be complex128 or complex64, got {eig_dtype!r}"
        )

    device = A.device
    # The eig/inv run at ``eig_dtype`` (complex64 halves the dominant eig cost);
    # everything downstream accumulates in complex128.  ``A`` itself is kept in
    # double for the assembly, only the eig input is down-cast.
    A_eig = A.to(eig_dtype)

    # --- λ(z): depth distribution (absorption-weighted lambdaE or pure histogram)
    if depth_weight is not None:
        lam = depth_weight.to(device=device, dtype=torch.float64)  # (izz,)
    else:
        lam = mc.lambda_z(energy_idx).to(device=device, dtype=torch.float64)
    izz = int(lam.shape[0])
    depth_step = float(mc.depth_step)
    # thick = izz · depthstep  →  dzt = depthstep / thick = 1 / izz.
    dzt = depth_step / (izz * depth_step) if izz > 0 else 0.0
    tpi = _TWOPI * depth_step                       # EMsoft: tpi = 2·π·depthstep

    # 1-based depth indices iz = 1 .. izz (EMsoft loop convention).
    iz = torch.arange(1, izz + 1, device=device, dtype=torch.float64)  # (izz,)

    # --- eigen-decomposition of the dynamical matrix --------------------------
    # torch.linalg.eig returns right eigenvectors as the COLUMNS of CGG, i.e.
    # A · CGG[:, j] = W[j] · CGG[:, j] — the same convention as EMsoft's ZGEEV.
    # eig + inv at eig_dtype (complex64 is ~1.2–1.4× faster); promote the
    # eigenpairs back to complex128 so the depth integral accumulates in double.
    W, CGG = torch.linalg.eig(A_eig)                # W: (B, n), CGG: (B, n, n)
    CGinv = torch.linalg.inv(CGG)                   # (B, n, n)
    W = W.to(torch.complex128)
    CGG = CGG.to(torch.complex128)
    CGinv = CGinv.to(torch.complex128)

    # Rescale eigenvalues: W ← W / (2·kn), kn = 1/λ (master pattern).
    kn = 1.0 / wavelength_nm
    W = W / (2.0 * kn)

    Wr = W.real                                     # (B, n)
    Wi = W.imag                                     # (B, n)

    # --- qold_{jk} (per batch) -------------------------------------------------
    # qold = tpi·(Im W_j + Im W_k)  +  i·tpi·(Re W_j − Re W_k)
    qre = tpi * (Wi.unsqueeze(-1) + Wi.unsqueeze(-2))          # (B, n, n)
    qim = tpi * (Wr.unsqueeze(-1) - Wr.unsqueeze(-2))          # (B, n, n)
    qold = torch.complex(qre, qim)                            # (B, n, n)
    # EMsoft overflow guard: if Re(qold) < 0, flip sign so exp(-qold·iz) decays.
    # Disabled (use_flip_guard=False) for the propagation-parity oracle — the
    # scattering-matrix route evaluates the un-guarded integral exactly.
    if use_flip_guard:
        flip = qold.real < 0.0
        qold = torch.where(flip, -qold, qold)

    # --- q_{jk} = Σ_iz λ(iz) · exp(−qold·iz) -----------------------------------
    # exponent[..., iz] = −qold[..., None] · iz[None.. ]  → (B, n, n, izz).
    expo = (-qold).unsqueeze(-1) * iz                         # (B, n, n, izz)
    weighted = lam * torch.exp(expo)                         # broadcast λ over iz
    q = weighted.sum(dim=-1)                                 # (B, n, n)

    # --- Ijk_{jk} = dzt · conj(CGinv_{j,g0}) · q_{jk} · CGinv_{k,g0} -----------
    cg_g0 = CGinv[..., :, gzero]                            # (B, n)  CGinv[:, j, g0]
    left = cg_g0.conj().unsqueeze(-1)                       # (B, n, 1)  conj(CGinv_{j,g0})
    right = cg_g0.unsqueeze(-2)                             # (B, 1, n)  CGinv_{k,g0}
    Ijk = dzt * left * q * right                           # (B, n, n)

    # --- Lgh = conj(CGG) @ Ijk @ CGG^T  (transpose, NOT Hermitian conjugate) ---
    CGG_T = CGG.transpose(-1, -2)                          # plain transpose
    Lgh = torch.matmul(torch.matmul(CGG.conj(), Ijk), CGG_T)
    return Lgh.to(torch.complex128)


def depth_integrated_Lgh_propagate(
    A: torch.Tensor,
    mc: MCData,
    energy_idx: int,
    *,
    wavelength_nm: float,
    gzero: int = 0,
    depth_weight: torch.Tensor | None = None,
    s1_dtype: torch.dtype = torch.complex128,
    matrix_exp_method: str = "auto",
    taylor_terms: int = 18,
) -> torch.Tensor:
    """Depth-integrated ``Lgh`` via **scattering-matrix propagation** (NO eig).

    The SP4-iter4 GPU-native depth integral.  Where
    :func:`depth_integrated_Lgh` eigen-decomposes the dynamical matrix ``A`` to
    evaluate ``CalcLgh`` analytically (``torch.linalg.eig`` — CUDA-hostile,
    PyTorch #107291), this route evaluates the **same** depth integral by
    propagating the transmitted-beam column of the scattering matrix down the
    foil — one batched matrix exponential plus the depth stack built by
    **binary doubling** (``⌈log2 izz⌉`` batched matmuls) contracted with one fused
    ``einsum``, all on the GPU, **no eigendecomposition**.

    **SP-PERF lever 2a (depth-batched).** The depth integral is NOT a sequential
    ``izz``-step matvec loop: the full depth stack ``V[b, :, k] = S1^(k+1)·e_{g0}``
    is assembled by repeatedly applying the current matrix power ``S1^T`` to the
    accumulated block and doubling ``T`` (so all ``izz`` columns are produced in
    ``⌈log2 izz⌉`` batched matmuls), then ``Lgh`` is one fused
    ``einsum("bgk,bhk->bgh", conj(V), λ·V)``.  This collapses the loop's ~300
    tiny per-step kernel launches per chunk (the launch-latency-bound wall — the
    GPU sat idle 77–93 %, see ``PERF/lever2_profile.md``) into a handful of large
    batched kernels.  It is **bit-faithful** to the sequential loop (the matrix
    powers are exact products of the same ``S1``; same ``λ(z)``, same bilinear
    ``conj``-on-first-index placement) — max-abs-rel vs the loop ≈ machine ε
    (measured 1.7e-15 Ni / 8.6e-15 pi).

    The identity (verified to machine precision against the eig path's
    *un-guarded* integral, see below):

        S(z) = expm( 2πi · (z / (2·kn)) · A ),   kn = 1/λ
        v(0) = e_{g0},   v(k·dz) = S(dz) · v((k−1)·dz)
        Lgh[g, h] = (1/izz) · Σ_z λ(z) · conj(v(z))[g] · v(z)[h]

    i.e. only the ``g0`` **column** ``v(z) = S(z)[:, g0]`` is needed (``v(0) = e_{g0}``),
    propagated by repeated multiplication with the **single** per-step propagator
    ``S1 = S(dz) = expm(2πi · A · dz/(2·kn))``, accumulating the rank-1 outer
    product ``λ(z_k)·outer(conj(v), v)`` at every depth bin.

    **Bilinear bookkeeping (the load-bearing detail).** Element ``[g, h]`` is
    ``conj(v(z))[g] · v(z)[h]`` — the *conjugate on the first (g) index*,
    matching EMsoft's ``Lgh = conj(CGG)·Ijk·CGGᵀ`` (a plain transpose, not ``.mH``).
    Empirically this is the unique placement that reproduces the eig path; the
    ``+`` sign in ``S = expm(+2πi…)`` (not ``−``) is likewise required (the eig
    path's per-mode factor carries ``exp(+2πi·z·W)``).

    **Equivalence to the eig path.** This is the eig path's depth integral with
    EMsoft's overflow guard *off* (``use_flip_guard=False``): for the few
    eigen-mode pairs with ``Im W_j + Im W_k < 0`` (unphysical anti-absorption,
    from the weak-beam corrections) EMsoft flips ``qold`` to keep the closed-form
    ``exp`` bounded; the propagation instead lets those modes grow — but they are
    *bounded by the depth weight* (``S1`` spectral radius ≈ 1.01, ``max|v| < 1``
    over ``izz`` ≈ 100 steps on real Ni/Al, so no overflow, no drift).  The two
    integrals agree to ~1e-14 wherever the guard does not fire; the guard's net
    effect on a real master is <2 % on ~1 % of pixels and **does not** change the
    fidelity vs the EMsoft oracle (NCC vs EMsoft identical to 2e-5 either way).

    Args:
        A: ``(B, n, n)`` (or ``(n, n)``) complex dynamical matrix (GetDynMat ``'D'``).
        mc: EMsoft Monte-Carlo data (supplies ``λ(z)`` and ``depth_step``).
        energy_idx: energy-axis index for the depth profile.
        wavelength_nm: relativistic wavelength ``λ`` (nm); ``kn = 1/λ``.
        gzero: column index of the transmitted beam in ``A`` (default 0).
        depth_weight: optional ``(izz,)`` absorption-weighted ``lambdaE`` override
            (same as the eig path); ``None`` → ``mc.lambda_z(energy_idx)``.
        s1_dtype: precision for the **single** ``S1 = expm(…)`` step only
            (``complex128`` default, or ``complex64`` perf knob).  ``S1`` is
            promoted back to ``complex128`` before propagation, so the depth-stack
            matrix powers and the fused ``einsum`` contraction **run in complex128**
            (the depth accumulation is the drift-sensitive part — kept double).
        matrix_exp_method: how to compute ``S1`` — ``"matrix_exp"``
            (``torch.matrix_exp``), ``"squaring"`` (manual batched
            scaling-and-squaring Taylor, all cuBLAS matmuls), or ``"auto"``
            (default).  **Benchmarked on the RTX 4070 (SP4 lessons §19):** for the
            small ``n``≈18–30 matrices the EBSD master produces, ``torch.matrix_exp``
            is the **winner** (≈2× faster than manual squaring at complex128, and
            competitive at complex64) — the PyTorch #107291 sequential-per-matrix
            pathology only bites for huge batches of *tiny* (2×2) matrices, NOT
            ``n``≈18, where ``matrix_exp`` parallelises across the batch on cuBLAS.
            So ``"auto"`` picks ``"matrix_exp"`` on **both** CUDA and CPU; the manual
            ``"squaring"`` route is kept as a faithful, dependency-light fallback
            (and was the brief's contingency had ``matrix_exp`` been the bottleneck).
        taylor_terms: number of Taylor terms for the ``"squaring"`` route
            (default 18 — ample once scaled so ``‖X/2^s‖ ≤ 0.5``).

    Returns:
        ``(B, n, n)`` ``complex128`` ``Lgh`` (same device as ``A``).

    Raises:
        ValueError: if ``wavelength_nm`` ≤ 0, ``A`` is not ``(*, n, n)``, ``gzero``
            is out of range, ``s1_dtype`` is not complex, or ``matrix_exp_method``
            is unknown.
    """
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be > 0, got {wavelength_nm!r}")
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if A.dim() != 3 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (B, n, n) or (n, n) square, got {tuple(A.shape)}"
        )
    n = A.shape[-1]
    if not 0 <= gzero < n:
        raise ValueError(f"gzero {gzero} out of range [0, {n})")
    if s1_dtype not in (torch.complex128, torch.complex64):
        raise ValueError(
            f"s1_dtype must be complex128 or complex64, got {s1_dtype!r}"
        )
    if matrix_exp_method not in ("auto", "matrix_exp", "squaring"):
        raise ValueError(
            f"matrix_exp_method must be auto/matrix_exp/squaring, got "
            f"{matrix_exp_method!r}"
        )

    device = A.device
    B = A.shape[0]

    # --- λ(z): depth distribution (absorption-weighted lambdaE or pure histogram)
    if depth_weight is not None:
        lam = depth_weight.to(device=device, dtype=torch.float64)
    else:
        lam = mc.lambda_z(energy_idx).to(device=device, dtype=torch.float64)
    izz = int(lam.shape[0])
    depth_step = float(mc.depth_step)
    dzt = 1.0 / izz if izz > 0 else 0.0
    kn = 1.0 / wavelength_nm

    # --- S1 = S(dz) = expm( 2πi · (dz/(2·kn)) · A ) ---------------------------
    # The per-step propagator (one matrix exp for the whole batch).  ``c`` is the
    # real scalar dz/(2kn); the generator is X = (2πi·c)·A.
    c = depth_step / (2.0 * kn)
    X = (_TWOPI * c) * 1j * A.to(s1_dtype)                  # (B, n, n)

    method = matrix_exp_method
    if method == "auto":
        # Benchmarked winner on both devices for n≈18–30 (lessons §19):
        # torch.matrix_exp beats manual squaring ~2× at complex128 here.
        method = "matrix_exp"

    if method == "matrix_exp":
        S1 = torch.matrix_exp(X)
    else:  # "squaring" — batched scaling-and-squaring Taylor (all cuBLAS matmuls)
        S1 = _matrix_exp_squaring(X, n_terms=taylor_terms)
    S1 = S1.to(torch.complex128)                           # propagate in double

    # --- Propagate v(z) = S(z)[:, g0] and accumulate Lgh ----------------------
    # v(0) = e_{g0}; v(k·dz) = S1 @ v((k-1)·dz);  the depth stack is
    #   V[b, :, k] = v((k+1)·dz) = S1^(k+1) · e_{g0}   for k = 0 .. izz-1,
    # and  Lgh[b, g, h] = (1/izz) · Σ_k λ(z_k) · conj(V[b, g, k]) · V[b, h, k].
    #
    # SP-PERF lever 2a — DEPTH-BATCH the propagation.  The naive form is a length-
    # ``izz`` Python loop of tiny per-step matvecs + outer-product accumulates
    # (~300 micro-kernel launches/chunk → the GPU sat idle 77–93 % of the wall, the
    # dominant cost; see ``PERF/lever2_profile.md``).  Instead build the WHOLE depth
    # stack ``V`` by **binary doubling** of the propagated block — start with the
    # first column ``V = v(1) = S1·e_{g0}`` and repeatedly apply the current power
    # ``S1^T`` to the accumulated ``T``-wide block, doubling ``T`` each round, so the
    # full ``izz``-column stack is built in ``⌈log2 izz⌉`` batched matmuls — then
    # contract it with ONE fused ``einsum``.  Mathematically identical to the loop
    # (same ``S1``, same ``λ(z)``, same bilinear placement: ``conj`` on the first/g
    # index): the matrix powers ``S1^m`` are exact products of the same propagator,
    # so the result is bit-faithful to the sequential path (max-abs-rel ≈ machine ε,
    # measured 1.7e-15 Ni / 8.6e-15 pi).  ``V`` is ``(B, n, izz)`` complex128
    # (≤163 MB even at the full pi B=1326/n=76 → well within VRAM).
    if izz <= 0:
        return torch.zeros(B, n, n, dtype=torch.complex128, device=device)

    e = torch.zeros(B, n, dtype=torch.complex128, device=device)
    e[:, gzero] = 1.0
    V = torch.matmul(S1, e.unsqueeze(-1))                   # (B, n, 1) = v(1)
    cur_pow = S1                                            # S1^(current T)
    T = 1
    while T < izz:
        # next block = S1^T · V = v(T+1 .. 2T)  (T columns, one batched matmul).
        # On the last round only ``izz − T`` of those columns are needed, so cap the
        # block — V never exceeds izz columns (bounds peak VRAM: without the cap V
        # would grow to the next power of two ≥ izz, e.g. 128 for izz=101, ~27 % more
        # memory on the small-n / large-B′ npx500 chunks).
        take = min(T, izz - T)                              # columns to append
        block = torch.matmul(cur_pow, V)[:, :, :take]      # v(T+1 .. T+take)
        V = torch.cat([V, block], dim=-1)                  # (B, n, T+take)
        if 2 * T < izz:                                    # skip the last squaring
            cur_pow = torch.matmul(cur_pow, cur_pow)       # S1^(2T)
        T += take                                          # = min(2T, izz)

    # Lgh[b, g, h] = (1/izz) · Σ_k λ_k · conj(V[b, g, k]) · V[b, h, k].
    lamV = lam.to(torch.complex128).reshape(1, 1, izz) * V  # weight by λ(z_k)
    Lgh = torch.einsum("bgk,bhk->bgh", V.conj(), lamV)
    return (dzt * Lgh).to(torch.complex128)


def _matrix_exp_squaring(X: torch.Tensor, *, n_terms: int = 18) -> torch.Tensor:
    """Batched ``expm(X)`` via scaling-and-squaring + a truncated Taylor series.

    A GPU-friendly alternative to :func:`torch.matrix_exp`, whose CUDA kernel runs
    each matrix in a batch **sequentially** (PyTorch #107291) and so wastes the GPU
    on large direction batches.  This uses only batched ``matmul`` (cuBLAS, fully
    parallel across the batch):

    1. scale ``Y = X / 2^s`` with ``s = max(0, ⌈log2 ‖X‖ + 1⌉)`` chosen per batch so
       ``‖Y‖ ≤ 0.5`` (1-norm, the largest absolute column sum), where the small-norm
       Taylor series converges fast;
    2. ``expm(Y) ≈ Σ_{j=0}^{n_terms} Y^j / j!`` by Horner-style batched matmuls;
    3. square ``s`` times: ``expm(X) = (expm(Y))^(2^s)``.

    ``s`` is taken as the **batch maximum** so a single (vectorised) number of
    squarings applies to the whole batch — over-squaring a small-norm matrix is
    exact (its ``Y`` was merely scaled down further), so this costs at most a few
    extra batched matmuls, never accuracy.  Validated bit-close to
    ``torch.matrix_exp`` (max-abs-diff ~1e-13 at complex128) in the SP4-iter4 tests.
    """
    # Per-matrix 1-norm ‖X‖₁ = max over columns of Σ_rows |X[:, j]|.
    norms = X.abs().sum(dim=-2).amax(dim=-1)               # (B,)
    max_norm = float(norms.max().item()) if norms.numel() else 0.0
    if max_norm <= 0.0:
        s = 0
    else:
        s = max(0, int(math.ceil(math.log2(max_norm) + 1.0)))
    scale = float(2 ** s)
    Y = X / scale

    n = X.shape[-1]
    eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
    # Horner: E = I + Y(1/1!)(I + Y(1/2!)(I + … )).  Build from the highest term.
    E = eye.clone()
    for j in range(n_terms, 0, -1):
        E = eye + torch.matmul(Y, E) * (1.0 / j)
    # Square s times: (expm(Y))^(2^s) = expm(X).
    for _ in range(s):
        E = torch.matmul(E, E)
    return E
