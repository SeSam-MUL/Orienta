"""Fundamental-zone primitives for SO(3): orientation FZ membership tests,
generators, and FZ projection — vendored from ebsdtorch by Zachary Varley.

Source repo: https://github.com/ZacharyVarley/ebsdtorch
License (MIT) preserved verbatim below.

Vendored 2026-05-09 for GPU spherical-indexing perf C1c milestone M1.1.
The upstream ebsdtorch is alpha-stage and not yet a stable PyPI dependency,
so we ship the primitives directly. Author attribution (Zachary Varley,
2024) is preserved in the docstrings.

Mathematical reference for the closed-form FZ indicators:
- Larsen, Peter Mahler, and Søren Schmidt. "Improved orientation sampling for
  indexing diffraction patterns of polycrystalline materials." Journal of
  Applied Crystallography 50, no. 6 (2017): 1571-1582.
- Krakow et al. "On three-dimensional misorientation spaces." Proc. R. Soc. A
  473 (2017): 20170274.

This file vendors:
- `LAUE_GROUPS` table (11 Laue groups) and `laue_elements(laue_id)`.
- `get_laue_mult(laue_id)` — group cardinality including inversion.
- `qu_prod`, `qu_prod_pos_real`, `qu_std` — minimal quaternion helpers
  needed by `ori_to_fz_laue` and `ori_in_fz_laue_brute`.
- `ori_to_fz_laue(quats, laue_id)` — project quaternions into the FZ.
- `ori_in_fz_laue_brute(quats, laue_id)` — slow oracle (24 multiplications).
- `ori_in_fz_laue(quats, laue_id)` — fast closed-form indicator (no enumeration).

We do NOT vendor `cube_to_rfz`, `rfz_to_cube`, `disorientation`, etc., to
keep the surface minimal. They can be added in M1.2+ as needed.

----------------------------------------------------------------------
Original ebsdtorch LICENSE (MIT, Copyright (c) 2024 Zachary Varley):

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
----------------------------------------------------------------------
"""
from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Laue group multiplicity (including inversion)
# ---------------------------------------------------------------------------

@torch.jit.script
def get_laue_mult(laue_group: int) -> int:
    """Multiplicity (cardinality including inversion) of a Laue group.

    Laue group ID convention (used everywhere in this module):
       1) C1   Triclinic        (1-, 1)            cardinality 2
       2) C2   Monoclinic        (2/m, m, 2)        cardinality 4
       3) D2   Orthorhombic      (mmm, mm2, 222)    cardinality 8
       4) C4   Tetragonal low    (4/m, 4-, 4)       cardinality 8
       5) D4   Tetragonal high   (4/mmm, ...)       cardinality 16
       6) C3   Trigonal low      (3-, 3)            cardinality 6
       7) D3   Trigonal high     (3-m, 3m, 32)      cardinality 12
       8) C6   Hexagonal low     (6/m, 6-, 6)       cardinality 12
       9) D6   Hexagonal high    (6/mmm, ...)       cardinality 24
      10) T    Cubic low         (m3-, 23)          cardinality 24
      11) O    Cubic high        (m3-m, 4-3m, 432)  cardinality 48
    """
    if laue_group == 1:
        return 2
    elif laue_group == 2:
        return 4
    elif laue_group == 3:
        return 8
    elif laue_group == 4:
        return 8
    elif laue_group == 5:
        return 16
    elif laue_group == 6:
        return 6
    elif laue_group == 7:
        return 12
    elif laue_group == 8:
        return 12
    elif laue_group == 9:
        return 24
    elif laue_group == 10:
        return 24
    elif laue_group == 11:
        return 48
    else:
        raise ValueError(f"laue_group must be in [1, 11], got {laue_group}")


# ---------------------------------------------------------------------------
# Quaternion generators for each Laue group (Larsen & Schmidt 2017)
# ---------------------------------------------------------------------------

@torch.jit.script
def laue_elements(laue_id: int) -> Tensor:
    """Quaternion generators for the given Laue group.

    Returns a tensor of shape (cardinality, 4) in (w, x, y, z) convention.
    The first element is always the identity quaternion (1, 0, 0, 0).
    """
    R2 = 1.0 / (2.0 ** 0.5)         # sqrt(2)/2
    R3 = (3.0 ** 0.5) / 2.0         # sqrt(3)/2

    LAUE_O = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [R2, 0.0, 0.0, R2],
        [R2, 0.0, 0.0, -R2],
        [0.0, R2, R2, 0.0],
        [0.0, -R2, R2, 0.0],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [R2, R2, 0.0, 0.0],
        [R2, -R2, 0.0, 0.0],
        [R2, 0.0, R2, 0.0],
        [R2, 0.0, -R2, 0.0],
        [0.0, R2, 0.0, R2],
        [0.0, -R2, 0.0, R2],
        [0.0, 0.0, R2, R2],
        [0.0, 0.0, -R2, R2],
    ], dtype=torch.float64)

    LAUE_T = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
    ], dtype=torch.float64)

    LAUE_D6 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, R3],
        [0.5, 0.0, 0.0, -R3],
        [0.0, 0.0, 0.0, 1.0],
        [R3, 0.0, 0.0, 0.5],
        [R3, 0.0, 0.0, -0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -0.5, R3, 0.0],
        [0.0, 0.5, R3, 0.0],
        [0.0, R3, 0.5, 0.0],
        [0.0, -R3, 0.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float64)

    LAUE_C6 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, R3],
        [0.5, 0.0, 0.0, -R3],
        [0.0, 0.0, 0.0, 1.0],
        [R3, 0.0, 0.0, 0.5],
        [R3, 0.0, 0.0, -0.5],
    ], dtype=torch.float64)

    LAUE_D3 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, R3],
        [0.5, 0.0, 0.0, -R3],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -0.5, R3, 0.0],
        [0.0, 0.5, R3, 0.0],
    ], dtype=torch.float64)

    LAUE_C3 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, R3],
        [0.5, 0.0, 0.0, -R3],
    ], dtype=torch.float64)

    LAUE_D4 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [R2, 0.0, 0.0, R2],
        [R2, 0.0, 0.0, -R2],
        [0.0, R2, R2, 0.0],
        [0.0, -R2, R2, 0.0],
    ], dtype=torch.float64)

    LAUE_C4 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [R2, 0.0, 0.0, R2],
        [R2, 0.0, 0.0, -R2],
    ], dtype=torch.float64)

    LAUE_D2 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float64)

    LAUE_C2 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float64)

    LAUE_C1 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float64)

    if laue_id == 1:
        return LAUE_C1
    elif laue_id == 2:
        return LAUE_C2
    elif laue_id == 3:
        return LAUE_D2
    elif laue_id == 4:
        return LAUE_C4
    elif laue_id == 5:
        return LAUE_D4
    elif laue_id == 6:
        return LAUE_C3
    elif laue_id == 7:
        return LAUE_D3
    elif laue_id == 8:
        return LAUE_C6
    elif laue_id == 9:
        return LAUE_D6
    elif laue_id == 10:
        return LAUE_T
    elif laue_id == 11:
        return LAUE_O
    else:
        raise ValueError(f"laue_id must be in [1, 11], got {laue_id}")


# ---------------------------------------------------------------------------
# Minimal quaternion helpers
# ---------------------------------------------------------------------------

@torch.jit.script
def qu_std(qu: Tensor) -> Tensor:
    """Standardize unit quaternion to have non-negative real part."""
    return torch.where(qu[..., 0:1] >= 0, qu, -qu)


@torch.jit.script
def qu_prod_raw(a: Tensor, b: Tensor) -> Tensor:
    """Quaternion multiplication a*b (no normalization, no real-positive)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


@torch.jit.script
def qu_prod(a: Tensor, b: Tensor) -> Tensor:
    """Quaternion multiplication, then standardize to non-negative real part."""
    return qu_std(qu_prod_raw(a, b))


@torch.jit.script
def qu_prod_pos_real(a: Tensor, b: Tensor) -> Tensor:
    """Magnitude of the real part of the quaternion product a*b.

    Cheaper than full multiplication when only |Re(a*b)| is needed (e.g.
    for FZ membership tests).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    return ow.abs()


# ---------------------------------------------------------------------------
# FZ projection and membership tests
# ---------------------------------------------------------------------------

@torch.jit.script
def ori_to_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """Project quaternions into the orientation fundamental zone.

    For each quaternion `q`, find the symmetry-equivalent representative
    `q_eq = q * g` (g in the Laue group) that has the largest |w| component.
    That representative is the FZ canonical form.

    Args:
        quats: (..., 4) unit quaternions, w >= 0 by convention.
        laue_id: integer 1..11.

    Returns:
        (..., 4) projected quaternions.
    """
    data_shape = quats.shape
    N = int(torch.prod(torch.tensor(data_shape[:-1])).item())
    card = get_laue_mult(laue_id) // 2
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)

    # Equivalent quaternion magnitudes (real part); shape (N, card)
    equivalent_real = qu_prod_pos_real(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    )
    # For each input quat, the symop g_i that maximizes |Re(q*g_i)| is the
    # FZ representative; multiply through and standardize sign.
    row_max = torch.argmax(equivalent_real, dim=-1)
    output = qu_prod(quats.reshape(N, 4), laue_group[row_max])
    return output.reshape(data_shape)


@torch.jit.script
def ori_in_fz_laue_brute(quats: Tensor, laue_id: int) -> Tensor:
    """Brute-force FZ membership test by enumeration.

    A quaternion is in the FZ iff the IDENTITY symop produces the largest
    |Re(q*g)| among the |G|/2 generators. Used as a slow oracle for
    validating the closed-form `ori_in_fz_laue` indicators below.

    Args:
        quats: (..., 4) unit quaternions, w >= 0.
        laue_id: integer 1..11.

    Returns:
        (...,) bool mask, True if the quaternion is already in the FZ.
    """
    data_shape = quats.shape
    N = int(torch.prod(torch.tensor(data_shape[:-1])).item())
    card = get_laue_mult(laue_id) // 2
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)

    equiv_real = qu_prod_pos_real(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()
    row_max = torch.argmax(equiv_real, dim=-1)
    # Identity is always the first generator in laue_elements(); index 0
    # being the row max means the original quaternion is already in the FZ.
    return (row_max == 0).reshape(data_shape[:-1])


@torch.jit.script
def ori_in_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """Closed-form FZ membership test (no enumeration).

    Per Larsen & Schmidt 2017, each Laue group's FZ has a closed-form
    indicator on the quaternion. For m-3m (laue_id=11):
        max(|x|,|y|,|z|) <= (sqrt(2) - 1) * w   AND   |x| + |y| + |z| <= w
    These are the Rodrigues fundamental-zone inequalities reformulated on
    quaternions: the first encodes the m-3m FZ as a truncated cube in
    Rodrigues space; the second is the cube-diagonal cap.

    Args:
        quats: (..., 4) unit quaternions, w >= 0.
        laue_id: integer 1..11.

    Returns:
        (...,) bool mask, True if in FZ. Functionally equivalent to
        `ori_in_fz_laue_brute` but ~50-100x faster on GPU (no group sum).
    """
    if laue_id == 11:
        # O: cubic high (m-3m). Truncated cube + diagonal cap.
        xyz_abs = torch.abs(quats[..., 1:])
        return (
            torch.max(xyz_abs, dim=-1).values <= (quats[..., 0] * (2 ** 0.5 - 1))
        ) & (torch.sum(xyz_abs, dim=-1) <= quats[..., 0])
    elif laue_id == 10:
        # T: cubic low (m3-).
        return torch.sum(torch.abs(quats[..., 1:]), dim=-1) <= quats[..., 0]
    elif laue_id == 9:
        # D6: hexagonal high (6/mmm).
        x_abs, y_abs = torch.abs(quats[..., 1]), torch.abs(quats[..., 2])
        cond = x_abs > y_abs
        m = torch.where(cond, x_abs, y_abs)
        n = torch.where(cond, y_abs, x_abs)
        rot = torch.where(m > (2 + 3 ** 0.5) * n, m, (3 ** 0.5 / 2) * m + 0.5 * n)
        return (torch.abs(quats[..., 3]) <= (2 - 3 ** 0.5) * quats[..., 0]) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 8:
        # C6: hexagonal low (6/m).
        return torch.abs(quats[..., 3]) <= ((2 - 3 ** 0.5) * quats[..., 0])
    elif laue_id == 7:
        # D3: trigonal high (3-m).
        rot = torch.where(
            torch.abs(quats[..., 1]) >= torch.abs(quats[..., 2]) * (3 ** 0.5),
            torch.abs(quats[..., 1]),
            (3 ** 0.5 / 2) * torch.abs(quats[..., 2]) + 0.5 * torch.abs(quats[..., 1]),
        )
        return (torch.abs(quats[..., 3]) <= ((1.0 / 3 ** 0.5) * quats[..., 0])) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 6:
        # C3: trigonal low (3-).
        return torch.abs(quats[..., 3]) <= (1.0 / 3 ** 0.5) * quats[..., 0]
    elif laue_id == 5:
        # D4: tetragonal high (4/mmm).
        x_abs, y_abs = torch.abs(quats[..., 1]), torch.abs(quats[..., 2])
        cond = x_abs > y_abs
        m = torch.where(cond, x_abs, y_abs)
        n = torch.where(cond, y_abs, x_abs)
        rot = torch.where(
            m > (2 ** 0.5 + 1) * n, m, (1 / 2 ** 0.5) * m + (1 / 2 ** 0.5) * n
        )
        return (torch.abs(quats[..., 3]) <= ((2 ** 0.5 - 1) * quats[..., 0])) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 4:
        # C4: tetragonal low (4/m).
        return torch.abs(quats[..., 3]) <= (2 ** 0.5 - 1) * quats[..., 0]
    elif laue_id == 3:
        # D2: orthorhombic (mmm).
        return torch.max(torch.abs(quats[..., 1:]), dim=-1).values <= quats[..., 0]
    elif laue_id == 2:
        # C2: monoclinic (2/m).
        return torch.abs(quats[..., 3]) <= quats[..., 0]
    elif laue_id == 1:
        # C1: triclinic — every quaternion (with w >= 0) is in FZ.
        return torch.full(
            quats.shape[:-1], True, dtype=torch.bool, device=quats.device
        )
    else:
        raise ValueError(f"laue_id must be in [1, 11], got {laue_id}")


__all__ = [
    "get_laue_mult",
    "laue_elements",
    "qu_std",
    "qu_prod_raw",
    "qu_prod",
    "qu_prod_pos_real",
    "ori_to_fz_laue",
    "ori_in_fz_laue",
    "ori_in_fz_laue_brute",
]
