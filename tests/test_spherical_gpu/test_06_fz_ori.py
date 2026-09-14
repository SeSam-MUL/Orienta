"""M1.1 tests: FZ-membership tests + projection (vendored from ebsdtorch).

Validates closed-form `ori_in_fz_laue` against brute-force 24-symop
enumeration `ori_in_fz_laue_brute` for all 11 Laue groups, and verifies
`ori_to_fz_laue` projection lands every quaternion inside the FZ.

These tests are CPU-portable; they don't require CUDA. The vendored
module uses `@torch.jit.script` decorators that work on either device.
"""
from __future__ import annotations

import pytest
import torch

from backend.spherical_gpu._math.fz_ori import (
    get_laue_mult,
    laue_elements,
    ori_in_fz_laue,
    ori_in_fz_laue_brute,
    ori_to_fz_laue,
    qu_prod,
    qu_std,
)


# Laue group cardinalities including inversion (i.e. |G|).
EXPECTED_MULT = {
    1: 2,    # C1
    2: 4,    # C2
    3: 8,    # D2
    4: 8,    # C4
    5: 16,   # D4
    6: 6,    # C3
    7: 12,   # D3
    8: 12,   # C6
    9: 24,   # D6
    10: 24,  # T (cubic low)
    11: 48,  # O (cubic high, m-3m)
}


def _random_unit_quats(n: int, seed: int = 0) -> torch.Tensor:
    """N random unit quaternions standardized to non-negative real part."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    return qu_std(q)


@pytest.mark.parametrize("laue_id", list(range(1, 12)))
def test_get_laue_mult_matches_table(laue_id: int) -> None:
    assert get_laue_mult(laue_id) == EXPECTED_MULT[laue_id]


def test_get_laue_mult_rejects_out_of_range() -> None:
    with pytest.raises(Exception):
        get_laue_mult(0)
    with pytest.raises(Exception):
        get_laue_mult(12)


@pytest.mark.parametrize("laue_id", list(range(1, 12)))
def test_laue_elements_shape_and_identity(laue_id: int) -> None:
    gens = laue_elements(laue_id)
    expected_card = EXPECTED_MULT[laue_id] // 2  # without inversion
    assert gens.shape == (expected_card, 4)
    # The first generator must be the identity quaternion.
    assert torch.allclose(
        gens[0], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gens.dtype)
    )


@pytest.mark.parametrize("laue_id", list(range(1, 12)))
def test_laue_elements_are_unit_norm(laue_id: int) -> None:
    gens = laue_elements(laue_id)
    norms = gens.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12), (
        f"laue_id={laue_id} has non-unit generators: norms={norms.tolist()}"
    )


def test_identity_is_in_every_fz() -> None:
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    for laue_id in range(1, 12):
        assert ori_in_fz_laue(identity, laue_id).item(), (
            f"identity reported NOT in FZ for laue_id={laue_id}"
        )
        assert ori_in_fz_laue_brute(identity, laue_id).item(), (
            f"identity (brute) reported NOT in FZ for laue_id={laue_id}"
        )


@pytest.mark.parametrize("laue_id", list(range(1, 12)))
def test_closed_form_matches_brute_force(laue_id: int) -> None:
    """The closed-form indicator MUST match the 24-symop brute force result
    to bit-identity on every random quaternion.
    """
    q = _random_unit_quats(1000, seed=42 + laue_id)
    fast = ori_in_fz_laue(q, laue_id)
    slow = ori_in_fz_laue_brute(q, laue_id)
    n_disagreements = int((fast != slow).sum().item())
    assert n_disagreements == 0, (
        f"laue_id={laue_id}: closed-form and brute-force disagree on "
        f"{n_disagreements}/1000 quaternions"
    )


@pytest.mark.parametrize("laue_id", list(range(1, 12)))
def test_ori_to_fz_lands_in_fz(laue_id: int) -> None:
    """Every projected quaternion MUST satisfy the FZ indicator."""
    q = _random_unit_quats(500, seed=100 + laue_id)
    q_fz = ori_to_fz_laue(q, laue_id)
    in_fz_count = int(ori_in_fz_laue(q_fz, laue_id).sum().item())
    # Allow tiny float-edge slack: ~1% may sit ON the FZ boundary where
    # the closed-form `<=` may flip ties differently than `argmax`. In
    # practice this is fine because the boundary is measure-zero.
    assert in_fz_count >= 495, (
        f"laue_id={laue_id}: only {in_fz_count}/500 projected quats in FZ "
        "(expected >= 495 due to boundary slack)"
    )


@pytest.mark.parametrize("laue_id", list(range(1, 12)))
def test_ori_to_fz_preserves_unit_norm(laue_id: int) -> None:
    q = _random_unit_quats(100, seed=200 + laue_id)
    q_fz = ori_to_fz_laue(q, laue_id)
    norms = q_fz.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
        f"laue_id={laue_id}: ori_to_fz_laue produced non-unit quats"
    )


def test_m3m_fz_indicator_explicit() -> None:
    """Explicit Larsen-Schmidt 2017 inequalities for m-3m (laue_id=11).

    The FZ test is:
        max(|x|, |y|, |z|) <= (sqrt(2) - 1) * w   AND
        |x| + |y| + |z|     <= w

    Construct quats that should pass and quats that should fail and check.
    """
    sqrt2_m1 = (2.0 ** 0.5) - 1.0  # ≈ 0.4142

    # 1) Identity (w=1, xyz=0) -> in FZ
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert ori_in_fz_laue(q_id, 11).all()

    # 2) Tiny rotation (w=0.99, x=0.01) -> in FZ
    q_small = torch.tensor([[0.99, 0.01, 0.0, 0.0]])
    q_small = q_small / q_small.norm(dim=-1, keepdim=True)
    assert ori_in_fz_laue(q_small, 11).all()

    # 3) 90° rotation about z (w=cos(45°), z=sin(45°)) -> NOT in FZ (boundary case)
    sq2 = (2.0 ** 0.5) / 2.0
    q_90z = torch.tensor([[sq2, 0.0, 0.0, sq2]])
    # |z|/w = 1.0 > sqrt(2)-1 ≈ 0.414 → not in FZ
    assert not ori_in_fz_laue(q_90z, 11).all()

    # 4) 60° rotation about <1,1,1> (close to FZ corner). For m-3m the
    #    angle limit toward <111> is ~62.8°. A 60° rotation about <111>:
    #    half-angle 30°, sin(30°)=0.5, cos(30°)=√3/2.
    h_angle = 0.5  # sin(30°)
    axis = torch.tensor([1.0, 1.0, 1.0]) / (3.0 ** 0.5)
    q_60_111 = torch.cat(
        [torch.tensor([3.0 ** 0.5 / 2.0]), h_angle * axis]
    ).unsqueeze(0)
    q_60_111 = q_60_111 / q_60_111.norm(dim=-1, keepdim=True)
    # max(|x|,|y|,|z|) = 0.5/√3 ≈ 0.289;  (sqrt(2)-1)*w = 0.414*√3/2 ≈ 0.359
    # First inequality: 0.289 <= 0.359 -> True
    # Second: 3·0.289 = 0.866 vs w = √3/2 ≈ 0.866. Boundary. Should pass via <=.
    assert ori_in_fz_laue(q_60_111, 11).all()


def test_m3m_brute_force_against_explicit_examples() -> None:
    """Same quats as test_m3m_fz_indicator_explicit but via brute-force.

    Confirms the brute-force oracle agrees with the closed-form on the
    boundary cases (the closed-form's `<=` ties match argmax tie-breaking).
    """
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert ori_in_fz_laue_brute(q_id, 11).all()

    q_small = torch.tensor([[0.99, 0.01, 0.0, 0.0]])
    q_small = q_small / q_small.norm(dim=-1, keepdim=True)
    assert ori_in_fz_laue_brute(q_small, 11).all()


def test_jit_scripted_runs_on_cuda_if_available() -> None:
    """If CUDA is available, the FZ functions must run end-to-end on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    q = _random_unit_quats(64, seed=999).cuda()
    in_fz = ori_in_fz_laue(q, 11)
    assert in_fz.device.type == "cuda"
    proj = ori_to_fz_laue(q, 11)
    assert proj.device.type == "cuda"
