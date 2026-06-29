"""Custom loss functions for phase classification.

Provides:

- :class:`WeightedCrossEntropyLoss` — cross-entropy with automatic
  inverse-frequency class weights from the training label distribution.
- :class:`ConfidencePenaltyLoss` — penalises overconfident *wrong*
  predictions, encouraging the model to say "I don't know" rather
  than guessing confidently.
- :class:`CombinedLoss` — weighted sum of the above two losses.
- :func:`compute_class_weights` — helper to derive class weights from
  label counts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(
    labels: torch.Tensor | list[int],
    n_classes: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a label tensor.

    Parameters
    ----------
    labels : torch.Tensor or list[int]
        1-D integer tensor (or list) of class labels.
    n_classes : int
        Total number of classes (including any that may have zero count).
    smoothing : float
        Additive smoothing applied to counts before inversion so that
        classes with zero samples get a finite (large) weight instead
        of infinity.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(n_classes,)`` with per-class weights
        normalised so the mean weight equals 1.
    """
    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.int64)

    counts = torch.zeros(n_classes, dtype=torch.float64)
    for lbl in labels:
        idx = int(lbl.item())
        if 0 <= idx < n_classes:
            counts[idx] += 1.0

    # Inverse frequency with smoothing
    weights = 1.0 / (counts + smoothing)

    # Normalise so mean weight == 1 (keeps loss scale stable)
    weights = weights / weights.mean()

    return weights.float()


# ---------------------------------------------------------------------------
# WeightedCrossEntropyLoss
# ---------------------------------------------------------------------------


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with per-class inverse-frequency weights.

    If *class_weights* is not provided at construction, call
    :meth:`set_weights_from_labels` before the first forward pass
    (or pass *class_weights* directly).

    Parameters
    ----------
    class_weights : torch.Tensor, optional
        Float tensor of shape ``(n_classes,)``.
    label_smoothing : float
        Label smoothing factor (0 = hard labels, 0.1 = 10 % smoothing).
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def set_weights_from_labels(
        self,
        labels: torch.Tensor | list[int],
        n_classes: int,
        smoothing: float = 0.1,
    ) -> None:
        """Compute and set class weights from label distribution.

        Parameters
        ----------
        labels : torch.Tensor or list[int]
            Training labels.
        n_classes : int
            Total number of classes.
        smoothing : float
            Additive smoothing for count inversion.
        """
        weights = compute_class_weights(labels, n_classes, smoothing)
        self.register_buffer("class_weights", weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(B, C)`` raw logits from the classifier.
        targets : torch.Tensor
            Shape ``(B,)`` integer class labels.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        w: torch.Tensor | None = self.class_weights  # type: ignore[assignment]
        return F.cross_entropy(
            logits,
            targets,
            weight=w,
            label_smoothing=self.label_smoothing,
        )


# ---------------------------------------------------------------------------
# ConfidencePenaltyLoss
# ---------------------------------------------------------------------------


class ConfidencePenaltyLoss(nn.Module):
    """Penalise overconfident wrong predictions.

    For each sample where the model is *wrong* (argmax != target), adds
    a penalty proportional to the confidence (max softmax probability)
    of that wrong prediction.  Correct predictions receive zero penalty.

    This encourages the model to be uncertain when it doesn't know,
    which is valuable for the EBSD use case where "I don't know" is
    better than a confident wrong phase assignment.

    Parameters
    ----------
    penalty_weight : float
        Scaling factor for the confidence penalty term.
    """

    def __init__(self, penalty_weight: float = 0.1) -> None:
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute confidence penalty for wrong predictions.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(B, C)`` raw logits.
        targets : torch.Tensor
            Shape ``(B,)`` integer class labels.

        Returns
        -------
        torch.Tensor
            Scalar penalty (0 if all predictions are correct).
        """
        probs = F.softmax(logits, dim=1)  # (B, C)
        max_probs, preds = probs.max(dim=1)  # (B,), (B,)
        wrong_mask = preds != targets  # (B,) bool

        if not wrong_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Penalty: mean confidence of wrong predictions
        penalty = max_probs[wrong_mask].mean()
        return self.penalty_weight * penalty


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """Weighted combination of cross-entropy and confidence penalty.

    ``loss = ce_weight * CE + cp_weight * ConfidencePenalty``

    Parameters
    ----------
    class_weights : torch.Tensor, optional
        Per-class weights for cross-entropy.
    label_smoothing : float
        Label smoothing for cross-entropy.
    ce_weight : float
        Scaling factor for the cross-entropy term.
    cp_weight : float
        Scaling factor for the confidence penalty term.
    penalty_weight : float
        Internal scaling inside :class:`ConfidencePenaltyLoss`.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        ce_weight: float = 1.0,
        cp_weight: float = 1.0,
        penalty_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.ce_loss = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        self.cp_loss = ConfidencePenaltyLoss(
            penalty_weight=penalty_weight,
        )
        self.ce_weight = ce_weight
        self.cp_weight = cp_weight

    def set_weights_from_labels(
        self,
        labels: torch.Tensor | list[int],
        n_classes: int,
        smoothing: float = 0.1,
    ) -> None:
        """Set class weights for the cross-entropy component.

        Parameters
        ----------
        labels : torch.Tensor or list[int]
            Training labels.
        n_classes : int
            Total number of classes.
        smoothing : float
            Additive smoothing for count inversion.
        """
        self.ce_loss.set_weights_from_labels(labels, n_classes, smoothing)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(B, C)`` raw logits.
        targets : torch.Tensor
            Shape ``(B,)`` integer class labels.

        Returns
        -------
        torch.Tensor
            Scalar combined loss.
        """
        ce: torch.Tensor = self.ce_loss(logits, targets)
        cp: torch.Tensor = self.cp_loss(logits, targets)
        return self.ce_weight * ce + self.cp_weight * cp
