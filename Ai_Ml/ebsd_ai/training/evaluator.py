"""Metrics, confusion matrix, and calibration evaluation.

Provides :class:`Evaluator` — a utility for computing classification metrics
on a trained :class:`PhaseClassifier` against an :class:`EBSDPhaseDataset`.

Key metrics:
- Top-1 and Top-k accuracy (Top-3 is most important for the pre-filter use case)
- Per-phase precision, recall, and F1 score
- Confusion matrix (absolute counts and normalized)
- Calibration analysis (predicted confidence vs observed accuracy)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from ebsd_ai.models.phase_classifier import PhaseClassifier

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics.

    Parameters
    ----------
    top1_accuracy : float
        Fraction of samples where the top-1 prediction is correct.
    topk_accuracy : float
        Fraction of samples where the correct label is in the top-k predictions.
    k : int
        The k used for top-k accuracy.
    per_phase_precision : dict[str, float]
        Phase name → precision (TP / (TP + FP)).
    per_phase_recall : dict[str, float]
        Phase name → recall (TP / (TP + FN)).
    per_phase_f1 : dict[str, float]
        Phase name → F1 score (harmonic mean of precision and recall).
    per_phase_support : dict[str, int]
        Phase name → number of ground truth samples.
    confusion_matrix : np.ndarray
        Shape ``(n_phases, n_phases)`` — rows are true labels,
        columns are predicted labels.
    phase_names : list[str]
        Ordered phase names for interpreting the confusion matrix.
    mean_confidence : float
        Average predicted probability for the top-1 class.
    calibration_bins : np.ndarray
        Bin edges for calibration analysis.
    calibration_accuracy : np.ndarray
        Observed accuracy per calibration bin.
    calibration_confidence : np.ndarray
        Mean predicted confidence per calibration bin.
    calibration_counts : np.ndarray
        Number of samples per calibration bin.
    n_samples : int
        Total number of evaluated samples.
    mean_attention_alpha : float
        Average fusion attention weight (1 = trusts pattern, 0 = trusts EDS).
    """

    top1_accuracy: float = 0.0
    topk_accuracy: float = 0.0
    k: int = 3
    per_phase_precision: dict[str, float] = field(default_factory=dict)
    per_phase_recall: dict[str, float] = field(default_factory=dict)
    per_phase_f1: dict[str, float] = field(default_factory=dict)
    per_phase_support: dict[str, int] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int64)
    )
    phase_names: list[str] = field(default_factory=list)
    mean_confidence: float = 0.0
    calibration_bins: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    calibration_accuracy: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    calibration_confidence: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    calibration_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    n_samples: int = 0
    mean_attention_alpha: float = 0.0

    def normalized_confusion_matrix(self) -> np.ndarray:
        """Row-normalized confusion matrix (each row sums to 1).

        Returns
        -------
        np.ndarray
            Shape ``(n_phases, n_phases)`` with values in [0, 1].
            Rows with zero support are all zeros.
        """
        row_sums = self.confusion_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        safe_sums = np.where(row_sums > 0, row_sums, 1)
        return self.confusion_matrix.astype(np.float64) / safe_sums


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Evaluate a :class:`PhaseClassifier` on a dataset.

    Parameters
    ----------
    model : PhaseClassifier
        Trained model to evaluate.
    device : str or torch.device
        ``"auto"`` picks CUDA if available, else CPU.
    k : int
        Top-k for accuracy computation (default 3).
    n_calibration_bins : int
        Number of bins for calibration analysis (default 10).
    batch_size : int
        Batch size for evaluation DataLoader (default 64).
    """

    def __init__(
        self,
        model: PhaseClassifier,
        device: str | torch.device = "auto",
        k: int = 3,
        n_calibration_bins: int = 10,
        batch_size: int = 64,
    ) -> None:
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.k = k
        self.n_calibration_bins = n_calibration_bins
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate(self, dataset: torch.utils.data.Dataset) -> EvaluationResult:
        """Run evaluation on the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Must yield dicts with ``"pattern"``, ``"eds_input"``, and
            ``"phase_label"`` keys (as produced by :class:`EBSDPhaseDataset`).

        Returns
        -------
        EvaluationResult
            All computed metrics.
        """
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        all_true: list[int] = []
        all_pred: list[int] = []
        all_top1_probs: list[float] = []
        all_topk_correct: list[bool] = []
        all_alphas: list[float] = []

        phase_names = self.model.phase_names
        n_phases = self.model.n_phases
        k = min(self.k, n_phases)

        for batch in loader:
            pattern = batch["pattern"].to(self.device)
            eds_input = batch["eds_input"].to(self.device)
            targets = batch["phase_label"]  # keep on CPU

            logits, alpha = self.model(pattern, eds_input)
            probs = torch.softmax(logits, dim=1).cpu()

            # Top-1 predictions
            top1_prob, top1_idx = probs.max(dim=1)

            # Top-k predictions
            _, topk_idx = probs.topk(k, dim=1)

            for i in range(targets.size(0)):
                true_label = targets[i].item()
                pred_label = top1_idx[i].item()
                all_true.append(int(true_label))
                all_pred.append(int(pred_label))
                all_top1_probs.append(float(top1_prob[i].item()))

                # Check if true label is in top-k
                topk_set = set(topk_idx[i].tolist())
                all_topk_correct.append(true_label in topk_set)

            all_alphas.extend(alpha.cpu().squeeze(-1).tolist())

        n_samples = len(all_true)
        if n_samples == 0:
            return EvaluationResult(phase_names=phase_names, k=k)

        true_arr = np.array(all_true, dtype=np.int64)
        pred_arr = np.array(all_pred, dtype=np.int64)
        prob_arr = np.array(all_top1_probs, dtype=np.float64)

        # --- Top-1 and Top-k accuracy ---
        top1_accuracy = float((true_arr == pred_arr).mean())
        topk_accuracy = float(np.mean(all_topk_correct))

        # --- Confusion matrix ---
        cm = np.zeros((n_phases, n_phases), dtype=np.int64)
        for t, p in zip(true_arr, pred_arr):
            cm[t, p] += 1

        # --- Per-phase precision, recall, F1 ---
        precision: dict[str, float] = {}
        recall: dict[str, float] = {}
        f1: dict[str, float] = {}
        support: dict[str, int] = {}

        for i, name in enumerate(phase_names):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            total_true = tp + fn

            support[name] = total_true

            if tp + fp > 0:
                precision[name] = tp / (tp + fp)
            else:
                precision[name] = 0.0

            if tp + fn > 0:
                recall[name] = tp / (tp + fn)
            else:
                recall[name] = 0.0

            p, r = precision[name], recall[name]
            if p + r > 0:
                f1[name] = 2 * p * r / (p + r)
            else:
                f1[name] = 0.0

        # --- Calibration ---
        bin_edges = np.linspace(0.0, 1.0, self.n_calibration_bins + 1)
        cal_accuracy = np.zeros(self.n_calibration_bins, dtype=np.float64)
        cal_confidence = np.zeros(self.n_calibration_bins, dtype=np.float64)
        cal_counts = np.zeros(self.n_calibration_bins, dtype=np.int64)

        correct_arr = (true_arr == pred_arr).astype(np.float64)

        for b in range(self.n_calibration_bins):
            lo = bin_edges[b]
            hi = bin_edges[b + 1]
            if b == self.n_calibration_bins - 1:
                # Include right edge in last bin
                mask = (prob_arr >= lo) & (prob_arr <= hi)
            else:
                mask = (prob_arr >= lo) & (prob_arr < hi)

            count = int(mask.sum())
            cal_counts[b] = count
            if count > 0:
                cal_accuracy[b] = float(correct_arr[mask].mean())
                cal_confidence[b] = float(prob_arr[mask].mean())

        # --- Attention alpha ---
        mean_alpha = float(np.mean(all_alphas)) if all_alphas else 0.0

        return EvaluationResult(
            top1_accuracy=top1_accuracy,
            topk_accuracy=topk_accuracy,
            k=k,
            per_phase_precision=precision,
            per_phase_recall=recall,
            per_phase_f1=f1,
            per_phase_support=support,
            confusion_matrix=cm,
            phase_names=phase_names,
            mean_confidence=float(prob_arr.mean()),
            calibration_bins=bin_edges,
            calibration_accuracy=cal_accuracy,
            calibration_confidence=cal_confidence,
            calibration_counts=cal_counts,
            n_samples=n_samples,
            mean_attention_alpha=mean_alpha,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def format_classification_report(result: EvaluationResult) -> str:
    """Format evaluation results as a human-readable report.

    Parameters
    ----------
    result : EvaluationResult
        Output of :meth:`Evaluator.evaluate`.

    Returns
    -------
    str
        Multi-line string with per-phase metrics and summary.
    """
    lines: list[str] = []
    lines.append(
        f"{'Phase':<20} {'Precision':>10} {'Recall':>10}"
        f" {'F1':>10} {'Support':>10}"
    )
    lines.append("-" * 62)

    for name in result.phase_names:
        p = result.per_phase_precision.get(name, 0.0)
        r = result.per_phase_recall.get(name, 0.0)
        f = result.per_phase_f1.get(name, 0.0)
        s = result.per_phase_support.get(name, 0)
        lines.append(f"{name:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10d}")

    lines.append("-" * 62)
    lines.append(f"Top-1 Accuracy:  {result.top1_accuracy:.4f}")
    lines.append(f"Top-{result.k} Accuracy:  {result.topk_accuracy:.4f}")
    lines.append(f"Mean Confidence: {result.mean_confidence:.4f}")
    lines.append(f"Mean Alpha:      {result.mean_attention_alpha:.4f}")
    lines.append(f"Samples:         {result.n_samples}")

    return "\n".join(lines)
