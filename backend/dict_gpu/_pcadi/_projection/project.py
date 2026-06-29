"""Compose direction_cosines + rotate + lambert + sample into full forward projection.

Mirrors kikuchipy._project_single_pattern_from_master_pattern (line 496) and its
batched wrapper. Vectorised over rotations in one kernel pass.

Convention notes
----------------
kikuchipy's `_project_single_pattern_from_master_pattern` calls
`rotate_vector(rotation, direction_cosines)` directly — i.e. it applies the
quaternion `q = (a, b, c, d)` itself (not its inverse) to the sample-frame
direction cosines. We verified from `kikuchipy._utils.numba.rotate_vector`
(lines 62-81) that the expanded matrix is

    R(q)[0,0] = a^2 + b^2 - c^2 - d^2
    R(q)[1,1] = a^2 - b^2 + c^2 - d^2
    R(q)[2,2] = a^2 - b^2 - c^2 + d^2

which is the standard ACTIVE quaternion rotation `v' = q v q*`.

Therefore we must apply the same active rotation `R(q)` (NOT R(q^-1)).
The identity vs random-rotation tests in `test_path_a_project.py` discriminate
this: a sign / inverse mistake passes identity (trivially) but fails random.
"""
from __future__ import annotations
import torch

from .sample import sample_master


def _rotate_directions(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Active rotation R(q) @ v for unit Hamilton quaternion `q = (w, x, y, z)`.

    Mirrors `kikuchipy._utils.numba.rotate_vector` exactly (the expanded
    9-term form), vectorised across an arbitrary leading shape.

    Parameters
    ----------
    q : torch.Tensor of shape (n, 4)
        Unit quaternions in (w, x, y, z) order (orix / kikuchipy convention).
    v : torch.Tensor of shape (..., 3)
        Vectors to rotate. Need not be unit.

    Returns
    -------
    torch.Tensor of shape (n, ..., 3)
        Rotated vectors: out[i, ..., :] = R(q[i]) @ v[..., :].
    """
    assert q.shape[-1] == 4, f"quaternion last dim must be 4, got {q.shape}"
    assert v.shape[-1] == 3, f"vector last dim must be 3, got {v.shape}"
    assert q.ndim == 2, f"quaternion must be (n, 4), got {q.shape}"

    a = q[:, 0]
    b = q[:, 1]
    c = q[:, 2]
    d = q[:, 3]
    aa = a * a
    bb = b * b
    cc = c * c
    dd = d * d
    ac = a * c
    ab = a * b
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # 3x3 rotation matrix entries, shape (n,)
    r00 = aa + bb - cc - dd
    r01 = 2.0 * (bc - ad)
    r02 = 2.0 * (ac + bd)
    r10 = 2.0 * (ad + bc)
    r11 = aa - bb + cc - dd
    r12 = 2.0 * (cd - ab)
    r20 = 2.0 * (bd - ac)
    r21 = 2.0 * (ab + cd)
    r22 = aa - bb - cc + dd

    # Build batched (n, 3, 3) and apply via einsum to v shaped (..., 3).
    # We want out[i, ..., :] = R[i] @ v[..., :].
    n = q.shape[0]
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)  # (n, 3, 3)

    # Flatten leading dims of v for a clean matmul, then unflatten.
    lead_shape = v.shape[:-1]
    v_flat = v.reshape(-1, 3)              # (M, 3)
    # out[i, m, :] = R[i] @ v_flat[m] = (R[i] v_flat.T).T[m]
    # Use einsum: 'nij,mj->nmi'
    out = torch.einsum("nij,mj->nmi", R, v_flat)  # (n, M, 3)
    return out.reshape(n, *lead_shape, 3)


def project_master_to_detector(
    mp_data: torch.Tensor,
    rotations_q: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Forward-project the master pattern onto the detector for `n` rotations.

    Parameters
    ----------
    mp_data : torch.Tensor of shape (2, npx, npx)
        Master pattern hemispheres in square Lambert: mp_data[0] = upper
        (north), mp_data[1] = lower (south). Same device and dtype.
    rotations_q : torch.Tensor of shape (n, 4)
        Hamilton quaternions (w, x, y, z), on the same device as mp_data.
        Sign convention matches orix / kikuchipy.
    directions : torch.Tensor of shape (det_h, det_w, 3)
        Precomputed sample-frame direction cosines for each detector pixel
        (output of `compute_direction_cosines`). Same device as mp_data.

    Returns
    -------
    torch.Tensor of shape (n, det_h, det_w)
        EBSD patterns, one per rotation, same dtype as `mp_data`.
    """
    assert mp_data.ndim == 3 and mp_data.shape[0] == 2, (
        f"mp_data must be (2, npx, npx), got {mp_data.shape}"
    )
    assert rotations_q.ndim == 2 and rotations_q.shape[1] == 4, (
        f"rotations_q must be (n, 4), got {rotations_q.shape}"
    )
    assert directions.ndim == 3 and directions.shape[-1] == 3, (
        f"directions must be (h, w, 3), got {directions.shape}"
    )

    # rotated has shape (n, det_h, det_w, 3) — sample_master accepts any
    # leading shape with trailing 3.
    rotated = _rotate_directions(rotations_q, directions)
    return sample_master(mp_data[0], mp_data[1], rotated)
