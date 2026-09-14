"""M1.5 tests: direct Wigner-D evaluation at arbitrary quaternions.

This is the REFERENCE oracle (matrix-exp generators), not the production
batched M2 kernel. Validates:
1. quat_to_zyz_euler correctness (identity, pure z-rot, gimbal lock).
2. Wigner-D matrix elementary properties (unitarity, identity at R=I).
3. cc = sum F[l,m,n] * D^l_{mn}(R) on synthetic spectra with known values.
4. Round-trip via two different rotation parameterizations.
"""
from __future__ import annotations

import math
import torch
import pytest

from backend.spherical_gpu._math.wigner_d_eval import (
    quat_to_zyz_euler,
    wigner_D_matrix,
    evaluate_wigner_D_at_quaternions,
)


# ---------------------------------------------------------------------------
# quat_to_zyz_euler
# ---------------------------------------------------------------------------

def test_quat_to_zyz_identity() -> None:
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a, b, g = quat_to_zyz_euler(q)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-6)
    assert torch.allclose(b, torch.zeros_like(b), atol=1e-6)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_quat_to_zyz_z_rotation_gimbal_lock() -> None:
    """For a pure z-rotation, beta=0 and alpha+gamma = total angle.

    Our gimbal-lock convention puts all rotation in alpha (gamma=0).
    """
    sq2 = (2.0 ** 0.5) / 2.0
    q = torch.tensor([[sq2, 0.0, 0.0, sq2]])  # 90-deg z
    a, b, g = quat_to_zyz_euler(q)
    assert b.item() < 1e-3, f"expected beta=0 at z-rot, got {b.item()}"
    assert g.item() < 1e-6, f"expected gamma=0 at gimbal lock, got {g.item()}"
    assert abs(a.item() - math.pi / 2) < 1e-3, (
        f"expected alpha=pi/2 at 90-deg z-rot, got {a.item()}"
    )


def test_quat_to_zyz_y_rotation() -> None:
    """For a 90-deg y-rotation, beta=pi/2, alpha=0, gamma=0."""
    sq2 = (2.0 ** 0.5) / 2.0
    q = torch.tensor([[sq2, 0.0, sq2, 0.0]])  # 90-deg y
    a, b, g = quat_to_zyz_euler(q)
    # Expected: beta = pi/2, alpha = 0, gamma = 0 (some conventions; our
    # ZYZ with R = R_z(a)*R_y(b)*R_z(g) would extract this)
    assert abs(b.item() - math.pi / 2) < 1e-3, (
        f"expected beta=pi/2, got {b.item()}"
    )


def test_quat_to_zyz_random_round_trip() -> None:
    """Construct a quat from random ZYZ, then extract → must round-trip."""
    torch.manual_seed(42)
    n = 20
    a = torch.rand(n) * (2 * math.pi)
    b = torch.rand(n) * math.pi
    g = torch.rand(n) * (2 * math.pi)

    # Quaternion from ZYZ: q = q_z(a) * q_y(b) * q_z(g)
    def axis_angle_quat(axis_xyz, angle):
        h = angle / 2.0
        return torch.stack([
            torch.cos(h),
            torch.tensor(axis_xyz[0], dtype=h.dtype) * torch.sin(h),
            torch.tensor(axis_xyz[1], dtype=h.dtype) * torch.sin(h),
            torch.tensor(axis_xyz[2], dtype=h.dtype) * torch.sin(h),
        ], dim=-1)

    def qmul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    qa = axis_angle_quat([0.0, 0.0, 1.0], a)
    qb = axis_angle_quat([0.0, 1.0, 0.0], b)
    qg = axis_angle_quat([0.0, 0.0, 1.0], g)
    q = qmul(qmul(qa, qb), qg)
    # Standardize to w >= 0
    q = torch.where(q[:, 0:1] >= 0, q, -q)

    a_out, b_out, g_out = quat_to_zyz_euler(q)

    # Compare. Beta should be the same; alpha and gamma may differ if there's
    # gimbal lock. Sample beta strictly inside (0, pi) to avoid that case.
    safe = (b > 0.05) & (b < math.pi - 0.05)
    err_b = (b[safe] - b_out[safe]).abs().max().item()
    assert err_b < 1e-4, f"beta round-trip err {err_b:.2e}"
    # alpha + gamma should differ by 0 or 2pi (mod 2pi)
    sum_in = (a + g) % (2 * math.pi)
    sum_out = (a_out + g_out) % (2 * math.pi)
    diff = (sum_in[safe] - sum_out[safe]).abs()
    diff = torch.minimum(diff, 2 * math.pi - diff)
    err_sum = diff.max().item()
    assert err_sum < 1e-4, f"alpha+gamma round-trip err {err_sum:.2e}"


# ---------------------------------------------------------------------------
# wigner_D_matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("l", [0, 1, 2, 3, 5])
def test_wigner_d_at_identity(l: int) -> None:
    """D^l(0, 0, 0) = (2l+1) × (2l+1) identity matrix."""
    D = wigner_D_matrix(l, 0.0, 0.0, 0.0)
    expected = torch.eye(2 * l + 1, dtype=torch.complex128)
    assert torch.allclose(D, expected, atol=1e-10), (
        f"l={l}: D(0,0,0) max err {(D - expected).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("l", [1, 2, 3, 5])
def test_wigner_d_unitarity(l: int) -> None:
    """D^l(R) is unitary: D D† = I."""
    torch.manual_seed(l)
    a, b, g = float(torch.rand(1).item()) * 2 * math.pi, float(torch.rand(1).item()) * math.pi, float(torch.rand(1).item()) * 2 * math.pi
    D = wigner_D_matrix(l, a, b, g)
    DDh = D @ D.conj().T
    expected = torch.eye(2 * l + 1, dtype=torch.complex128)
    assert torch.allclose(DDh, expected, atol=1e-9), (
        f"l={l}: D D† max err {(DDh - expected).abs().max().item():.2e}"
    )


def test_wigner_d_z_rotation() -> None:
    """D^l(alpha, 0, 0) is diagonal with entries exp(-i*m*alpha)."""
    l = 2
    alpha = 0.5
    D = wigner_D_matrix(l, alpha, 0.0, 0.0)
    for m_idx in range(2 * l + 1):
        m = m_idx - l
        expected = complex(math.cos(-m * alpha), math.sin(-m * alpha))
        actual = D[m_idx, m_idx].item()
        # compare reasonably loose
        err = abs(actual - expected)
        assert err < 1e-9, f"D^{l}_{m},{m}(alpha=0.5) err {err:.2e}"


# ---------------------------------------------------------------------------
# evaluate_wigner_D_at_quaternions
# ---------------------------------------------------------------------------

def test_cc_identity_with_diag_spectrum() -> None:
    """If F[l, m, m] = 1 for some l, then cc(I) = trace(I) = 2l+1."""
    L = 4
    for j in range(L):
        spectrum = torch.zeros(L, 2 * L - 1, 2 * L - 1, dtype=torch.complex128)
        for m in range(-j, j + 1):
            spectrum[j, m + L - 1, m + L - 1] = 1.0
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        cc = evaluate_wigner_D_at_quaternions(spectrum, q, L)
        expected = float(2 * j + 1)
        assert abs(cc.item() - expected) < 1e-5, (
            f"j={j}: cc(I) = {cc.item()}, expected {expected}"
        )


def test_cc_zrotation_with_diag_spectrum() -> None:
    """For F[1, m, m] = 1, cc(R_z(α)) = sum_m exp(-i*m*α). Real part = sum cos(mα).

    At alpha=pi/2 with l=1: cos(0) + cos(pi/2) + cos(-pi/2) = 1.
    """
    L = 4
    spectrum = torch.zeros(L, 2 * L - 1, 2 * L - 1, dtype=torch.complex128)
    for m in range(-1, 2):
        spectrum[1, m + L - 1, m + L - 1] = 1.0

    sq2 = (2.0 ** 0.5) / 2.0
    q = torch.tensor([[sq2, 0.0, 0.0, sq2]])  # 90-deg z
    cc = evaluate_wigner_D_at_quaternions(spectrum, q, L)
    assert abs(cc.item() - 1.0) < 1e-6


def test_cc_unitarity_invariance() -> None:
    """For diagonal spectrum F[l, m, m] = 1, cc(R) = sum_m D^l_mm(R).

    The trace of D^l(R) is real for any R (rotation has real char.):
    trace(D^l(R)) = sum_{k=-l..l} cos(k*theta) where theta is the rotation
    angle and the sum is the character of the rep at angle theta.

    For l=1, character chi(theta) = 1 + 2*cos(theta).
    """
    L = 4
    spectrum = torch.zeros(L, 2 * L - 1, 2 * L - 1, dtype=torch.complex128)
    for m in range(-1, 2):
        spectrum[1, m + L - 1, m + L - 1] = 1.0

    # Build a quaternion at axis-angle (axis=arbitrary, angle=theta).
    theta = 0.3
    axis = torch.tensor([1.0, 2.0, 3.0])
    axis = axis / axis.norm()
    h = theta / 2.0
    q = torch.cat([
        torch.tensor([math.cos(h)]), math.sin(h) * axis
    ]).unsqueeze(0)
    q = torch.where(q[:, 0:1] >= 0, q, -q)

    cc = evaluate_wigner_D_at_quaternions(spectrum, q, L)
    expected = 1.0 + 2.0 * math.cos(theta)  # SU(2) character at l=1
    assert abs(cc.item() - expected) < 1e-5, (
        f"cc = {cc.item()}, expected character chi^1(theta=0.3) = {expected}"
    )


def test_cc_invalid_shapes_raise() -> None:
    L = 4
    spectrum = torch.zeros(L, 2 * L - 1, 2 * L - 1, dtype=torch.complex128)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    # Wrong L
    with pytest.raises(ValueError):
        evaluate_wigner_D_at_quaternions(spectrum, quats, L + 1)

    # Wrong spectrum shape
    bad = torch.zeros(L, 2 * L, 2 * L - 1, dtype=torch.complex128)
    with pytest.raises(ValueError):
        evaluate_wigner_D_at_quaternions(bad, quats, L)

    # Wrong quat last dim
    bad_q = torch.zeros(5, 3)
    with pytest.raises(ValueError):
        evaluate_wigner_D_at_quaternions(spectrum, bad_q, L)


def test_cc_batched_spectrum() -> None:
    """Spectrum can have a batch dim: shape (B, L, 2L-1, 2L-1)."""
    L = 4
    B = 3
    spectrum = torch.zeros(B, L, 2 * L - 1, 2 * L - 1, dtype=torch.complex128)
    spectrum[0, 1, L - 1, L - 1] = 1.0
    spectrum[1, 1, L - 1, L - 1] = 2.0
    spectrum[2, 1, L - 1, L - 1] = 3.0
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    cc = evaluate_wigner_D_at_quaternions(spectrum, q, L)
    assert cc.shape == (B, 1)
    assert torch.allclose(cc.squeeze(1), torch.tensor([1.0, 2.0, 3.0]), atol=1e-6)
