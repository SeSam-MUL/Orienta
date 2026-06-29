"""Tests for custom loss functions."""

from __future__ import annotations

import torch
import pytest

from ebsd_ai.models.losses import (
    CombinedLoss,
    ConfidencePenaltyLoss,
    WeightedCrossEntropyLoss,
    compute_class_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 16
N_CLASSES = 4


@pytest.fixture
def logits() -> torch.Tensor:
    """Random logits (B, C)."""
    torch.manual_seed(42)
    return torch.randn(BATCH, N_CLASSES, requires_grad=True)


@pytest.fixture
def targets() -> torch.Tensor:
    """Random integer labels (B,)."""
    torch.manual_seed(7)
    return torch.randint(0, N_CLASSES, (BATCH,))


@pytest.fixture
def balanced_labels() -> list[int]:
    """Balanced labels: 10 per class for 4 classes."""
    return [i for i in range(N_CLASSES) for _ in range(10)]


@pytest.fixture
def imbalanced_labels() -> list[int]:
    """Imbalanced: 100 of class 0, 10 of class 1, 5 of class 2, 1 of class 3."""
    return [0] * 100 + [1] * 10 + [2] * 5 + [3] * 1


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


class TestComputeClassWeights:
    """Tests for the weight computation helper."""

    def test_output_shape(self, balanced_labels: list[int]) -> None:
        w = compute_class_weights(balanced_labels, N_CLASSES)
        assert w.shape == (N_CLASSES,)

    def test_output_dtype(self, balanced_labels: list[int]) -> None:
        w = compute_class_weights(balanced_labels, N_CLASSES)
        assert w.dtype == torch.float32

    def test_balanced_weights_equal(self, balanced_labels: list[int]) -> None:
        """Equal class distribution → equal weights."""
        w = compute_class_weights(balanced_labels, N_CLASSES)
        assert torch.allclose(w, w[0].expand(N_CLASSES), atol=1e-2)

    def test_mean_is_one(self, balanced_labels: list[int]) -> None:
        """Weights are normalised so mean ≈ 1."""
        w = compute_class_weights(balanced_labels, N_CLASSES)
        assert abs(w.mean().item() - 1.0) < 1e-5

    def test_imbalanced_minority_higher(
        self, imbalanced_labels: list[int]
    ) -> None:
        """Minority class gets higher weight."""
        w = compute_class_weights(imbalanced_labels, N_CLASSES)
        # class 3 has 1 sample, class 0 has 100
        assert w[3].item() > w[0].item()

    def test_all_positive(self, imbalanced_labels: list[int]) -> None:
        w = compute_class_weights(imbalanced_labels, N_CLASSES)
        assert (w > 0).all()

    def test_accepts_tensor(self) -> None:
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        w = compute_class_weights(labels, 3)
        assert w.shape == (3,)

    def test_zero_count_class_gets_finite_weight(self) -> None:
        """Class with no samples still gets a finite weight (via smoothing)."""
        labels = [0, 0, 1, 1]
        w = compute_class_weights(labels, 3)  # class 2 has 0 samples
        assert torch.isfinite(w).all()
        assert w[2].item() > 0

    def test_smoothing_parameter(self) -> None:
        labels = [0, 0, 0]
        w_low = compute_class_weights(labels, 2, smoothing=0.01)
        w_high = compute_class_weights(labels, 2, smoothing=1.0)
        # With higher smoothing, the ratio between weights is smaller
        ratio_low = (w_low[1] / w_low[0]).item()
        ratio_high = (w_high[1] / w_high[0]).item()
        assert ratio_low > ratio_high


# ---------------------------------------------------------------------------
# WeightedCrossEntropyLoss
# ---------------------------------------------------------------------------


class TestWeightedCrossEntropyLoss:
    """Tests for WeightedCrossEntropyLoss."""

    def test_scalar_output(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = WeightedCrossEntropyLoss()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_positive_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = WeightedCrossEntropyLoss()
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_gradient_flows(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = WeightedCrossEntropyLoss()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_with_class_weights(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        weights = torch.tensor([1.0, 2.0, 1.0, 0.5])
        loss_fn = WeightedCrossEntropyLoss(class_weights=weights)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_set_weights_from_labels(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = WeightedCrossEntropyLoss()
        loss_fn.set_weights_from_labels([0, 0, 1, 2, 3], N_CLASSES)
        assert loss_fn.class_weights is not None
        assert loss_fn.class_weights.shape == (N_CLASSES,)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_weighted_differs_from_unweighted(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        unweighted = WeightedCrossEntropyLoss()
        weighted = WeightedCrossEntropyLoss(
            class_weights=torch.tensor([10.0, 1.0, 1.0, 1.0])
        )
        loss_u = unweighted(logits, targets)
        loss_w = weighted(logits, targets)
        assert loss_u.item() != loss_w.item()

    def test_label_smoothing(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        hard = WeightedCrossEntropyLoss(label_smoothing=0.0)
        smooth = WeightedCrossEntropyLoss(label_smoothing=0.1)
        loss_hard = hard(logits, targets)
        loss_smooth = smooth(logits, targets)
        # Label smoothing generally changes the loss
        assert loss_hard.item() != loss_smooth.item()

    def test_no_nan(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = WeightedCrossEntropyLoss()
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# ConfidencePenaltyLoss
# ---------------------------------------------------------------------------


class TestConfidencePenaltyLoss:
    """Tests for ConfidencePenaltyLoss."""

    def test_scalar_output(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = ConfidencePenaltyLoss()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_non_negative(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = ConfidencePenaltyLoss()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_zero_when_all_correct(self) -> None:
        """No penalty when model gets everything right."""
        # Make logits strongly predict the correct class
        logits = torch.zeros(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0])
        logits_data = torch.zeros(4, 3)
        logits_data[0, 0] = 100.0
        logits_data[1, 1] = 100.0
        logits_data[2, 2] = 100.0
        logits_data[3, 0] = 100.0
        logits = logits_data.requires_grad_(True)

        loss_fn = ConfidencePenaltyLoss()
        loss = loss_fn(logits, targets)
        assert loss.item() < 1e-6

    def test_higher_when_wrong_and_confident(self) -> None:
        """More penalty when model is confidently wrong."""
        # Confidently wrong: strong logit on wrong class
        logits_confident = torch.tensor(
            [[10.0, -10.0], [-10.0, 10.0]], requires_grad=True
        )
        targets = torch.tensor([1, 0])  # Both wrong

        # Less confident wrong: weaker logits
        logits_unsure = torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]], requires_grad=True
        )

        loss_fn = ConfidencePenaltyLoss(penalty_weight=1.0)
        loss_confident = loss_fn(logits_confident, targets)
        loss_unsure = loss_fn(logits_unsure, targets)
        assert loss_confident.item() > loss_unsure.item()

    def test_gradient_flows(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = ConfidencePenaltyLoss()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_penalty_weight_scales(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_low = ConfidencePenaltyLoss(penalty_weight=0.1)
        loss_high = ConfidencePenaltyLoss(penalty_weight=1.0)
        l1 = loss_low(logits, targets)
        l2 = loss_high(logits, targets)
        # Higher penalty_weight → higher loss (when there are wrong preds)
        if l1.item() > 0:
            assert l2.item() > l1.item()

    def test_no_nan(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = ConfidencePenaltyLoss()
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------


class TestCombinedLoss:
    """Tests for CombinedLoss."""

    def test_scalar_output(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = CombinedLoss()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_positive(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = CombinedLoss()
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_gradient_flows(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = CombinedLoss()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_set_weights_from_labels(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = CombinedLoss()
        loss_fn.set_weights_from_labels([0, 0, 1, 2, 3], N_CLASSES)
        assert loss_fn.ce_loss.class_weights is not None
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_ce_only(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """With cp_weight=0, equals pure cross-entropy."""
        combined = CombinedLoss(ce_weight=1.0, cp_weight=0.0)
        ce_only = WeightedCrossEntropyLoss()
        l_combined = combined(logits, targets)
        l_ce = ce_only(logits, targets)
        assert abs(l_combined.item() - l_ce.item()) < 1e-5

    def test_with_class_weights(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        weights = torch.tensor([1.0, 2.0, 1.0, 0.5])
        loss_fn = CombinedLoss(class_weights=weights)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_no_nan(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        loss_fn = CombinedLoss()
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)

    def test_ce_weight_scaling(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Higher ce_weight increases the total loss."""
        low = CombinedLoss(ce_weight=0.5, cp_weight=0.0)
        high = CombinedLoss(ce_weight=2.0, cp_weight=0.0)
        l_low = low(logits, targets)
        l_high = high(logits, targets)
        assert l_high.item() > l_low.item()

    def test_both_components_contribute(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Combined loss is greater than CE alone when there are wrong preds."""
        ce_only = CombinedLoss(ce_weight=1.0, cp_weight=0.0)
        both = CombinedLoss(
            ce_weight=1.0, cp_weight=1.0, penalty_weight=1.0
        )
        l_ce = ce_only(logits, targets)
        l_both = both(logits, targets)
        # If any predictions are wrong, the penalty adds to the loss
        assert l_both.item() >= l_ce.item() - 1e-6
