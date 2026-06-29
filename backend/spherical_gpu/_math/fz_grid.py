"""Cubochoric quaternion grid construction inside SO(3) fundamental zones.

Vendored from ebsdtorch's `s2_and_so3/sampling.py` and `s2_and_so3/laue_fz_ori.py`
(Zachary Varley, MIT — see fz_ori.py for full license text). Source repo:
https://github.com/ZacharyVarley/ebsdtorch.

Functions:
- `so3_cubochoric_grid(edge_length, device)`:
  Build the full cubochoric SO(3) grid of edge_length^3 quaternions covering
  all of SO(3) (one hemisphere of S^3, with w >= 0 standardization).
- `sample_ori_fz_laue_angle(laue_id, angular_resolution_deg, device,
                              permute=True)`:
  Generate a uniform cubochoric SO(3) grid sized for the requested angular
  resolution, then keep only the points lying inside the Laue fundamental
  zone. For m-3m (laue_id=11) at angular_resolution_deg=2.0, this yields
  ~96k quaternions inside the FZ.

Resolution heuristic (Singh & De Graef 2016, MSMSE 24:085013, eq. 12):

    theta_avg(N) ≈ 131.97049 / (N - 0.038551)   degrees

where N is the half-edge of the full SO(3) cube (so the grid has (2N+1)^3
or N_so3 ≈ (2 * 131.97049 / (theta_avg - 0.03732) + 1)^3 sample points).
The FZ subset is approximately N_so3 / (|G|/2) — for m-3m: 1/24.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .cubochoric import cu2qu, CUBE_HALF_EDGE
from .fz_ori import qu_std, ori_in_fz_laue, get_laue_mult


@torch.jit.script
def so3_cubochoric_grid(edge_length: int, device: torch.device) -> Tensor:
    """Cubochoric grid covering SO(3).

    Builds a uniform `edge_length` × `edge_length` × `edge_length` cube of
    cubochoric coordinates (cell-centered), maps every point to a quaternion
    via `cu2qu`, and standardizes to w >= 0.

    Args:
        edge_length: integer N >= 1; the grid is N^3 cells.
        device: torch device.

    Returns:
        (N^3, 4) float32 unit quaternions covering SO(3).
    """
    half_edge = (math.pi ** (2.0 / 3.0)) / 2.0
    # Use cell-centers: linspace on [-half_edge, +half_edge] with edge+1 nodes,
    # drop the last for cell-centers convention.
    cu_axis = torch.linspace(
        -half_edge, half_edge, edge_length + 1, device=device, dtype=torch.float32
    )[:-1]
    # Shift to cell centers:
    step = (2.0 * half_edge) / float(edge_length)
    cu_axis = cu_axis + step / 2.0
    grid_x, grid_y, grid_z = torch.meshgrid(cu_axis, cu_axis, cu_axis, indexing="ij")
    cu = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3)
    return qu_std(cu2qu(cu))


#: Empirical fit constant: avg nearest-neighbor angular gap (deg) × cube edge
#: stays ≈ 183 across the range edge ∈ [22, 132] (m-3m FZ).
#: Determined empirically 2026-05-09 from cubochoric grids at edge ∈ {22, 46, 100, 132}.
_EDGE_TIMES_THETA = 183.0


def edge_length_for_resolution(angular_resolution_deg: float) -> int:
    """Pick a cubochoric cube edge that yields the requested avg angular gap.

    Empirical fit (m-3m, FZ-only nearest-neighbor): edge × theta ≈ 183.
    Tested on edge ∈ {22, 46, 100, 132}; max deviation ~2%.

    Note: upstream ebsdtorch's `sample_ori_fz_laue_angle` uses a Singh-De Graef
    eq. 12-derived formula that has a cube-vs-FZ unit confusion and produces
    grids ~24× too sparse. We override with the empirical fit below.
    """
    if angular_resolution_deg <= 0:
        raise ValueError(
            f"angular_resolution_deg must be > 0, got {angular_resolution_deg}"
        )
    edge = int(round(_EDGE_TIMES_THETA / angular_resolution_deg))
    return max(edge, 2)


def sample_ori_fz_laue_angle(
    laue_id: int,
    angular_resolution_deg: float,
    device: torch.device,
    permute: bool = True,
    seed: int = 0,
) -> Tensor:
    """Cubochoric SO(3) grid restricted to the Laue fundamental zone.

    Builds a full SO(3) cubochoric cube and rejects points outside the FZ.

    Args:
        laue_id: Laue group integer 1..11.
        angular_resolution_deg: target average angular nearest-neighbor
            spacing (degrees) within the FZ. For L=88 spherical NCC
            (cc-bin 360/175 ≈ 2.057°), use 2.057. The actual avg gap
            will be within ~5% of this target.
        device: torch device.
        permute: if True, randomize the order of returned quaternions.
        seed: RNG seed for the permutation. Ignored if permute=False.

    Returns:
        (N_fz, 4) float32 unit quaternions inside the FZ.
        For m-3m at 2.057°: edge=89, N_fz ~ 30,000.
        For m-3m at 1.5°:    edge=122, N_fz ~ 76,000.
        For m-3m at 1.37°:  edge=132, N_fz ~ 96,000.

    Note on upstream divergence: ebsdtorch's `sample_ori_fz_laue_angle` uses
    a different formula that yields ~24× sparser grids than the angular
    resolution suggests. We override the formula with an empirical fit
    `edge × theta ≈ 183` validated against actual FZ nearest-neighbor
    measurements at edge ∈ {22, 46, 100, 132}. See `edge_length_for_resolution`.
    """
    if laue_id < 1 or laue_id > 11:
        raise ValueError(f"laue_id must be in [1, 11], got {laue_id}")

    edge_length = edge_length_for_resolution(angular_resolution_deg)

    quats = so3_cubochoric_grid(edge_length, device)
    in_fz = ori_in_fz_laue(quats, laue_id)
    quats_fz = quats[in_fz]

    if permute:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(quats_fz.shape[0], generator=gen).to(device)
        quats_fz = quats_fz[perm]

    return quats_fz


def sample_ori_fz_laue_edge(
    laue_id: int,
    edge_length: int,
    device: torch.device,
    permute: bool = True,
    seed: int = 0,
) -> Tensor:
    """Direct cubochoric-cube edge control (bypasses angular-resolution fit).

    Args:
        laue_id: Laue group integer 1..11.
        edge_length: cube cells per axis (>= 2). Total cube cells = edge^3,
            of which roughly edge^3 / (|G|/2) lie inside the FZ.
        device: torch device.
        permute: if True, randomize order.
        seed: RNG seed.

    Returns:
        (N_fz, 4) float32 unit quaternions in the FZ.
    """
    if laue_id < 1 or laue_id > 11:
        raise ValueError(f"laue_id must be in [1, 11], got {laue_id}")
    if edge_length < 2:
        raise ValueError(f"edge_length must be >= 2, got {edge_length}")

    quats = so3_cubochoric_grid(edge_length, device)
    in_fz = ori_in_fz_laue(quats, laue_id)
    quats_fz = quats[in_fz]

    if permute:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(quats_fz.shape[0], generator=gen).to(device)
        quats_fz = quats_fz[perm]

    return quats_fz


def build_fz_quaternion_grid(
    laue_id: int,
    angular_resolution_deg: float = 2.057,
    device: torch.device | str = "cuda",
    permute: bool = True,
    seed: int = 0,
) -> Tensor:
    """High-level wrapper for FZ grid construction.

    Same as `sample_ori_fz_laue_angle` with default angular resolution
    matching the L=88 spherical-NCC cc-bin spacing (360°/175 ≈ 2.057°).
    """
    if isinstance(device, str):
        device = torch.device(device)
    return sample_ori_fz_laue_angle(
        laue_id=laue_id,
        angular_resolution_deg=angular_resolution_deg,
        device=device,
        permute=permute,
        seed=seed,
    )


__all__ = [
    "so3_cubochoric_grid",
    "sample_ori_fz_laue_angle",
    "sample_ori_fz_laue_edge",
    "build_fz_quaternion_grid",
    "edge_length_for_resolution",
]
