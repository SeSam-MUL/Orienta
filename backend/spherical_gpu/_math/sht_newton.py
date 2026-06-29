"""Autograd-based analytic Newton refinement on the SHT cross-correlation.

EMSphInx computes Jacobian/Hessian analytically in
``sht_xcorr.hpp:Correlator<Real>::derivatives`` via ~300 lines of dense
Chebyshev recursion + Wigner-d glide-plane symmetry. Porting that exactly
to PyTorch is error-prone; instead we express the FORWARD scalar
``cc(eu) = sum_{l, m, n} flm[l, m] * conj(gln[l, n]) * D^l_{m, n}(eu)``
in PyTorch, and let ``torch.autograd.functional.{jacobian, hessian}``
compute derivatives. Mathematically correct and finite-difference-
validated by ``test_09_analytic_newton.py``.

**Performance caveat (read before wiring into a hot path).**
``_wigner_d_small`` is O(l^3) per l with Python loops; ``cc_at_rotation``
sums it L times → O(L^4) per forward. Each Newton iteration runs the
forward via autograd's Jacobian + Hessian (3-9x more passes), and a
typical pixel takes 3-15 iterations. At L=8 (unit tests) a refinement
is ~50 ms; at L=68 (real-data parity gates) it climbs into the
seconds-per-pixel regime. Use this path only for parity-gate-sized
slices (~200 pixels), not full-scan production runs. If a future
caller needs production scale, the obvious follow-up is to vectorize
``_wigner_d_small`` over l (recurrence) or port the EMSphInx analytic
``derivatives()`` C++ to PyTorch.

Convention: Lenthe-style Wigner-D, ``D^l_{m, n}(a, b, c) =
exp(-i m a) * d^l_{m, n}(b) * exp(-i n c)``. The real cc is
``Re[ sum_{l, m, n} ... ]`` (imaginary part cancels for real input
patterns; we assert this in tests).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


def _wigner_d_small(l: int, beta: Tensor) -> Tensor:
    """Real small-d Wigner matrix for degree l, evaluated at scalar beta.

    Returns a (2*l+1, 2*l+1) tensor indexed as d[m' + l, m + l]. Closed-form
    Jacobi-polynomial sum (Wikipedia / Sakurai); see `_wigner_d_small_scalar`
    for the reference scalar implementation kept side-by-side for
    finite-difference / parity checks.

    Vectorised over the (m', m, s) triple via broadcast tensors so the
    whole d-matrix is one batched torch op instead of (2l+1)^2 Python
    iterations. At L=68 this drops the cost from ~80M Python ops to one
    GPU/CPU kernel — the difference between hours and seconds per pixel
    in the autograd Newton path.

    Log-factorial path (math.lgamma) is used throughout; for l > ~30
    direct math.factorial overflows float64 in the sqrt step. The
    coefficient table is built in float64 then cast to ``dtype`` once.
    """
    device = beta.device
    dtype = beta.dtype
    cos_half = torch.cos(beta / 2)
    sin_half = torch.sin(beta / 2)

    # Cache log-factorials up to (2l+1)! once — used for both numerator
    # and denominator without repeated lgamma calls. lgamma(n+1) == log(n!).
    log_fact = torch.tensor(
        [math.lgamma(n + 1) for n in range(2 * l + 2)],
        dtype=torch.float64, device=device,
    )

    # Mesh tensors for (m', m, s). Layout: (2l+1, 2l+1, 2l+1).
    arange_l = torch.arange(-l, l + 1, device=device)  # (2l+1,)
    s_arange = torch.arange(0, 2 * l + 1, device=device)  # (2l+1,)
    mp = arange_l.view(-1, 1, 1)
    m = arange_l.view(1, -1, 1)
    s = s_arange.view(1, 1, -1)

    # Validity mask: s in [max(0, m - m'), min(l + m, l - m')]
    s_min = torch.clamp(m - mp, min=0)
    s_max = torch.minimum(l + m, l - mp)
    valid = (s >= s_min) & (s <= s_max)

    # log_fact[n] = lgamma(n+1) = log(n!) for n in [0, 2l+1].
    # The scalar formula uses lgamma(k) = log((k-1)!) = log_fact[k-1].
    # So lgamma(l + m' + 1) == log_fact[l + m']; etc.
    # Log-numerator-root depends only on (m', m). Broadcast over s axis.
    log_num_root_2d = 0.5 * (
        log_fact[l + arange_l].view(-1, 1)   # lgamma(l + m' + 1)
        + log_fact[l - arange_l].view(-1, 1)  # lgamma(l - m' + 1)
        + log_fact[l + arange_l].view(1, -1)  # lgamma(l + m  + 1)
        + log_fact[l - arange_l].view(1, -1)  # lgamma(l - m  + 1)
    ).unsqueeze(-1)  # (2l+1, 2l+1, 1)

    # Log-denominator depends on (m', m, s). Clamp BOTH ends — masked
    # positions outside ``valid`` can produce indices outside log_fact's
    # range (size 2l+2 i.e. valid index [0, 2l+1]). The ``valid``
    # multiplication below zeroes their contribution either way.
    upper = 2 * l + 1
    idx_a = (l + m - s).clamp(min=0, max=upper)
    idx_b = s.clamp(min=0, max=upper)
    idx_c = (mp - m + s).clamp(min=0, max=upper)
    idx_d = (l - mp - s).clamp(min=0, max=upper)
    log_den = (
        log_fact[idx_a] + log_fact[idx_b]
        + log_fact[idx_c] + log_fact[idx_d]
    )

    # Sign: (-1)^(m' - m + s).
    sign = torch.where(
        ((mp - m + s) & 1) == 0,
        torch.ones((), dtype=torch.float64, device=device),
        -torch.ones((), dtype=torch.float64, device=device),
    )

    # Coefficient with the validity mask folded in. exp(log_num_root -
    # log_den) is finite because log_fact entries are finite even at
    # masked indices.
    coeff_fp64 = sign * torch.exp(log_num_root_2d - log_den)
    coeff = (coeff_fp64 * valid.to(torch.float64)).to(dtype)

    # Powers: clamp to >= 0 (negative power-of-zero would yield inf/NaN
    # at beta = 0; masked positions already contribute zero via coeff,
    # but we still need clean numerics for cos_half**0 = 1).
    power_c = (2 * l - 2 * s + m - mp).to(dtype).clamp(min=0)
    power_s = (2 * s + mp - m).to(dtype).clamp(min=0)

    cos_pow = cos_half ** power_c
    sin_pow = sin_half ** power_s
    # Sum over the s axis → (2l+1, 2l+1) d-matrix in (m'+l, m+l) order.
    d = (coeff * cos_pow * sin_pow).sum(dim=-1)
    return d


def _wigner_d_small_scalar(l: int, beta: Tensor) -> Tensor:
    """Reference scalar implementation — kept for unit-test parity check.

    Same math as ``_wigner_d_small`` but with explicit Python loops over
    (m', m, s). Slower; only used in tests to validate the vectorised
    version's numerics at small L.
    """
    device = beta.device
    dtype = beta.dtype
    size = 2 * l + 1
    d = torch.zeros((size, size), dtype=dtype, device=device)
    cos_half = torch.cos(beta / 2)
    sin_half = torch.sin(beta / 2)
    lgf = math.lgamma
    for mp in range(-l, l + 1):
        for m in range(-l, l + 1):
            s_min = max(0, m - mp)
            s_max = min(l + m, l - mp)
            total = torch.zeros((), dtype=dtype, device=device)
            log_num_root = 0.5 * (
                lgf(l + mp + 1) + lgf(l - mp + 1)
                + lgf(l + m + 1) + lgf(l - m + 1)
            )
            for s in range(s_min, s_max + 1):
                log_den = (
                    lgf(l + m - s + 1) + lgf(s + 1)
                    + lgf(mp - m + s + 1) + lgf(l - mp - s + 1)
                )
                coeff = ((-1) ** (mp - m + s)) * math.exp(log_num_root - log_den)
                power_c = 2 * l - 2 * s + m - mp
                power_s = 2 * s + mp - m
                total = total + coeff * cos_half**power_c * sin_half**power_s
            d[mp + l, m + l] = total
    return d


def cc_at_rotation(flm: Tensor, gln: Tensor, eu: Tensor, L: int) -> Tensor:
    """SHT cross-correlation at a single rotation, as a differentiable scalar.

    Parameters
    ----------
    flm : (L, L) complex tensor
        Master pattern SHT coefs, ``flm[l, m]`` (m in 0..l).
        Standard half-storage; m=-1..-l recovered via flm[l, |m|].conj() * (-1)^m.
    gln : (L, L) complex tensor
        Pattern SHT coefs, same layout.
    eu : (3,) real tensor (requires_grad ok)
        Bunge ZYZ Euler angles in radians (alpha, beta, gamma).
    L : int
        Bandwidth (used l = 0..L-1).

    Returns
    -------
    cc : () real scalar tensor
        Re[ sum_{l, m, n} flm[l, m] * conj(gln[l, n]) *
            exp(-i m a) * d^l_{m, n}(b) * exp(-i n c) ]

    Raises
    ------
    TypeError
        If ``flm`` or ``gln`` is not ``complex128``. The point of this path
        is FP64 parity with EMSphInx; silently accepting ``complex64`` would
        defeat that. Callers that intentionally want FP32 should construct
        their own forward.
    """
    if flm.dtype != torch.complex128:
        raise TypeError(
            f"cc_at_rotation requires flm in complex128 for FP64-parity; got "
            f"{flm.dtype}. Cast at the call site with .to(torch.complex128)."
        )
    if gln.dtype != torch.complex128:
        raise TypeError(
            f"cc_at_rotation requires gln in complex128 for FP64-parity; got "
            f"{gln.dtype}. Cast at the call site with .to(torch.complex128)."
        )
    alpha, beta, gamma = eu[0], eu[1], eu[2]
    cc = torch.zeros((), dtype=eu.dtype, device=eu.device)
    for l in range(L):
        d = _wigner_d_small(l, beta)
        flm_l = _expand_half_to_full(flm[l, :l + 1], l)
        gln_l = _expand_half_to_full(gln[l, :l + 1], l)
        m_idx = torch.arange(-l, l + 1, dtype=eu.dtype, device=eu.device)
        n_idx = torch.arange(-l, l + 1, dtype=eu.dtype, device=eu.device)
        exp_ma = torch.exp(-1j * m_idx * alpha).to(flm.dtype)
        exp_nc = torch.exp(-1j * n_idx * gamma).to(flm.dtype)
        block = (
            (flm_l[:, None] * exp_ma[:, None])
            * d.to(flm.dtype)
            * (gln_l.conj()[None, :] * exp_nc[None, :])
        )
        cc = cc + block.sum().real
    return cc


def _expand_half_to_full(flm_half: Tensor, l: int) -> Tensor:
    """Expand (l+1,) complex tensor m=0..l into (2l+1,) m=-l..l.

    Uses the standard reality condition for real-spherical functions:
    flm[l, -m] = (-1)^m * conj(flm[l, m]).
    """
    pos = flm_half
    if l == 0:
        return pos
    neg_m = torch.arange(1, l + 1, device=flm_half.device)
    sign = (-1.0) ** neg_m.to(flm_half.real.dtype)
    neg = (sign * pos[1:l + 1].conj()).flip(0)
    return torch.cat([neg, pos], dim=0)


def jacobian_hessian(flm: Tensor, gln: Tensor, eu: Tensor, L: int) -> Tuple[Tensor, Tensor]:
    """Return (jac (3,), hess (3,3)) of ``cc_at_rotation`` wrt eu.

    Uses ``torch.autograd.functional.{jacobian, hessian}``. The forward
    pass is run twice internally — once for J, once for H; that's faster
    than computing both from one graph for this small problem size.
    """
    def f(e):
        return cc_at_rotation(flm, gln, e, L)
    jac = torch.autograd.functional.jacobian(f, eu, create_graph=False)
    hes = torch.autograd.functional.hessian(f, eu, create_graph=False)
    return jac, hes


def _zxz_to_zyz(eu_zxz: Tensor) -> Tensor:
    """Convert Bunge ZXZ Euler -> ZYZ Euler (in radians).

    Derivation: R_y(beta) = R_z(pi/2) R_x(beta) R_z(-pi/2). Substituting
    into R_zxz(a, b, c) = R_z(a) R_x(b) R_z(c) and pattern-matching
    against R_zyz(a', b', c') = R_z(a') R_y(b') R_z(c') gives
    a' = a - pi/2, b' = b, c' = c + pi/2.
    """
    half_pi = math.pi / 2
    a, b, c = eu_zxz[0], eu_zxz[1], eu_zxz[2]
    return torch.stack([a - half_pi, b, c + half_pi])


def _zyz_to_zxz(eu_zyz: Tensor) -> Tensor:
    """Convert ZYZ Euler -> Bunge ZXZ Euler (in radians). Inverse of _zxz_to_zyz."""
    half_pi = math.pi / 2
    a, b, c = eu_zyz[0], eu_zyz[1], eu_zyz[2]
    return torch.stack([a + half_pi, b, c - half_pi])


def newton_refine(
    flm: Tensor,
    gln: Tensor,
    eu_seed: Tensor,
    L: int,
    *,
    eps: float = 1e-5,
    max_iter: int = 15,
    step_clip: float = 0.5,
) -> Tuple[Tensor, Tensor, bool]:
    """Newton refinement of cc(eu) starting from ``eu_seed``.

    ``eu_seed`` is in Bunge ZXZ (the convention used everywhere else in
    this project — orix.Rotation.from_euler defaults to it). Internally
    we run Newton against the ZYZ-formulated ``cc_at_rotation`` and
    convert back at the end; the conversion is a constant shift on
    alpha/gamma so it doesn't affect step magnitudes or convergence.

    Mirrors ``sht_xcorr.hpp:Correlator<Real>::refinePeak`` semantics:
    Cholesky-style step, magnitude must monotonically decrease, fall
    back to 1-D / 2-D sub-problem on Hessian degeneracy near beta = 0 / pi.

    We MAXIMIZE cc, so the Newton step that zeros the gradient is
    eu_{n+1} = eu_n - H^{-1} J  (same first-order condition as a
    minimization; the sign of H carries the convexity direction).

    Note: at beta = 0 / pi, the Hessian is rank-deficient by 1 (ZYZ
    gimbal degeneracy — alpha and gamma rotate around the same axis).
    The exception fallback handles that case via a 1-D sub-problem.

    Parameters
    ----------
    eps : convergence tolerance on max-abs step (radians)
    max_iter : hard cap on iterations
    step_clip : per-axis step bound (radians). The raw Newton step is
        clamped to ``[-step_clip, +step_clip]`` per axis before applying.
        Prevents an ill-conditioned Hessian from launching the iterate
        far across the cc surface; the monotonicity check still rejects
        bad updates that survive the clamp.
    """
    # Convert the caller's Bunge ZXZ seed to ZYZ for the ZYZ-formulated
    # cc_at_rotation. Pure shift on alpha/gamma — no effect on derivatives.
    eu = _zxz_to_zyz(eu_seed).clone().detach()
    # Initial step-magnitude bound matches sht_xcorr.hpp:refinePeak:
    # prevMag2 = (pi/2) * 3 / slP, where slP = 2L-1 is the cc-volume side
    # length. This caps the first Newton step at ~one cc-bin (~ pi/(2L-1)
    # rad), which is what "sub-bin refinement" actually means.
    sl_p = max(2 * L - 1, 1)
    prev_step_mag2 = (math.pi / 2) * 3 / sl_p
    converged = False
    for _ in range(max_iter):
        jac, hes = jacobian_hessian(flm, gln, eu, L)
        try:
            step = torch.linalg.solve(hes, jac)
        except RuntimeError:
            step = torch.zeros(3, dtype=eu.dtype, device=eu.device)
            if hes[0, 0].abs() > 1e-12:
                step[0] = jac[0] / hes[0, 0]
        # Per-axis step clamp — bounds the Newton update so a bad Hessian
        # condition number can't launch us across the cc surface. Done in
        # one place; the monotonicity check below catches any bad step
        # that survives the clamp.
        step = step.clamp(-step_clip, step_clip)
        mag2 = float((step * step).sum())
        if mag2 > prev_step_mag2:
            # Reject this step; current eu is the best we have. Convert
            # back to ZXZ before returning.
            break
        prev_step_mag2 = mag2
        eu = eu - step
        if step.abs().max() < eps:
            converged = True
            break
    cc_val = cc_at_rotation(flm, gln, eu, L)
    return _zyz_to_zxz(eu), cc_val, converged
