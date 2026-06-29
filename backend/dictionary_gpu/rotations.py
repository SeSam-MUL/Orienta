"""Apply unit quaternions to 3D direction tensors.

Convention: quaternion (w, x, y, z) with w as scalar.
Rotation: v' = q * v * q^-1, expanded as
    v' = v + 2 * q_vec x (q_vec x v + w * v)
"""
from __future__ import annotations

import torch


def rotate_directions_by_quaternions(
    quats: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Rotate directions by quaternions.

    Parameters
    ----------
    quats : (N, 4)
        Unit quaternions (w, x, y, z).
    directions : (P, 3)
        Unit direction vectors.

    Returns
    -------
    (N, P, 3) tensor of rotated directions.
    """
    if quats.dim() != 2 or quats.shape[-1] != 4:
        raise ValueError(f"quats must be (N, 4), got {tuple(quats.shape)}")
    if directions.dim() != 2 or directions.shape[-1] != 3:
        raise ValueError(f"directions must be (P, 3), got {tuple(directions.shape)}")

    w = quats[:, 0:1]                             # (N, 1)
    qv = quats[:, 1:]                             # (N, 3)

    # Broadcast to (N, P, 3)
    v = directions.unsqueeze(0)                   # (1, P, 3)
    qv_b = qv.unsqueeze(1)                        # (N, 1, 3)
    w_b = w.unsqueeze(1)                          # (N, 1, 1)

    t = 2.0 * torch.cross(qv_b.expand(-1, v.shape[1], -1), v.expand(qv_b.shape[0], -1, -1), dim=-1)
    rotated = v + w_b * t + torch.cross(qv_b.expand(-1, v.shape[1], -1), t, dim=-1)
    return rotated
