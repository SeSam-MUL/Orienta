"""SO(3)-aware angular contrastive loss for embedding training.

Uses misorientation angles between crystal orientations to define
positive and negative pairs.  Orientations within ``pos_angle_deg``
are treated as positives; those beyond ``neg_angle_deg`` as negatives.

The loss is an InfoNCE-style formulation scaled by a temperature
parameter, operating on L2-normalized embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _misorientation_angles(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise misorientation angles (radians) between quaternions.

    Parameters
    ----------
    q1, q2 : torch.Tensor
        (N, 4) and (M, 4) unit quaternions (scalar-first: w, x, y, z).

    Returns
    -------
    torch.Tensor
        (N, M) pairwise misorientation angles in radians.
    """
    # Quaternion dot product gives cos(angle/2)
    # q1: (N, 1, 4), q2: (1, M, 4) -> dot: (N, M)
    dot = torch.abs((q1.unsqueeze(1) * q2.unsqueeze(0)).sum(dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


class AngularContrastiveLoss(nn.Module):
    """SO(3)-aware contrastive loss using misorientation angles.

    Parameters
    ----------
    temperature : float
        Scaling temperature for the InfoNCE loss.
    pos_angle_deg : float
        Maximum misorientation (degrees) for positive pairs.
    neg_angle_deg : float
        Minimum misorientation (degrees) for negative pairs.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        pos_angle_deg: float = 2.0,
        neg_angle_deg: float = 10.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.pos_angle_rad = pos_angle_deg * torch.pi / 180.0
        self.neg_angle_rad = neg_angle_deg * torch.pi / 180.0

    def forward(
        self,
        embeddings: torch.Tensor,
        orientations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute angular contrastive loss.

        Parameters
        ----------
        embeddings : torch.Tensor
            (B, D) L2-normalized embedding vectors.
        orientations : torch.Tensor
            (B, 4) unit quaternions.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Pairwise misorientation angles (B, B)
        angles = _misorientation_angles(orientations, orientations)

        # Positive mask: within pos_angle (exclude diagonal)
        pos_mask = (angles < self.pos_angle_rad).float()
        pos_mask.fill_diagonal_(0)

        # Negative mask: beyond neg_angle
        neg_mask = (angles > self.neg_angle_rad).float()

        # Cosine similarity matrix scaled by temperature
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature

        # For numerical stability, subtract max per row
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Mask out self-similarity
        self_mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)

        # Exp of similarities (exclude self)
        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)

        # Denominator: sum of exp over all non-self entries
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8

        # Log probability of positives
        log_prob = sim - torch.log(denom)

        # If no positives for a sample, use all non-self as soft positives
        # weighted by angular proximity
        has_pos = pos_mask.sum(dim=1) > 0

        if has_pos.any():
            # Weighted mean of log probabilities over positive pairs
            pos_log_prob = (pos_mask * log_prob).sum(dim=1)
            pos_count = pos_mask.sum(dim=1).clamp(min=1)
            loss_with_pos = -pos_log_prob / pos_count

            # For samples without explicit positives, use soft weighting
            # based on angular proximity (closer = higher weight)
            soft_weights = torch.exp(-angles / self.pos_angle_rad)
            soft_weights.fill_diagonal_(0)
            soft_weights = soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            loss_without_pos = -(soft_weights * log_prob).sum(dim=1)

            loss = torch.where(has_pos, loss_with_pos, loss_without_pos)
        else:
            # No explicit positive pairs at all — use soft angular weighting
            soft_weights = torch.exp(-angles / self.pos_angle_rad)
            soft_weights.fill_diagonal_(0)
            soft_weights = soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            loss = -(soft_weights * log_prob).sum(dim=1)

        return loss.mean()
