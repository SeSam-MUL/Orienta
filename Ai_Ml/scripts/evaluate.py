"""Evaluate a trained EBSD phase classifier on TrainingStore data.

Usage
-----
.. code-block:: bash

    .venv/bin/python scripts/evaluate.py --model best.pt --data ./data
    .venv/bin/python scripts/evaluate.py --model best.pt --data ./data --top-k 5
    .venv/bin/python scripts/evaluate.py --model best.pt --data ./data --split 0.3

The script:
1. Loads a trained :class:`PhaseClassifier` from a checkpoint
2. Loads evaluation data from a :class:`TrainingStore` directory
3. Optionally splits off a test set (or evaluates on the full dataset)
4. Runs the :class:`Evaluator` and prints a detailed report
5. Displays the confusion matrix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from ebsd_ai.data.dataset import EBSDPhaseDataset, train_val_split
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.training.evaluator import Evaluator, format_classification_report
from ebsd_ai.training.trainer import load_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Argument list.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EBSD phase classifier.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (best.pt or last.pt from training).",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to TrainingStore directory containing shard_*.h5 files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k for accuracy computation (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device: "auto", "cpu", or "cuda" (default: auto).',
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.0,
        help=(
            "If > 0, split data and evaluate only on the held-out fraction. "
            "E.g. --split 0.2 evaluates on 20%% of data. "
            "If 0 (default), evaluate on the full dataset."
        ),
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for calibration analysis (default: 10).",
    )

    return parser.parse_args(argv)


def _format_confusion_matrix(
    cm: np.ndarray, phase_names: list[str],
) -> str:
    """Format a confusion matrix as a readable string.

    Parameters
    ----------
    cm : np.ndarray
        Shape ``(n, n)`` integer confusion matrix.
    phase_names : list[str]
        Phase names for row/column headers.

    Returns
    -------
    str
        Multi-line formatted table.
    """
    n = len(phase_names)
    # Determine column width from longest phase name
    col_w = max(len(name) for name in phase_names)
    col_w = max(col_w, 6)  # minimum width

    lines: list[str] = []

    # Header row
    header = " " * (col_w + 2)
    for name in phase_names:
        header += f"{name:>{col_w + 2}}"
    lines.append(header)

    # Data rows
    for i in range(n):
        row = f"{phase_names[i]:>{col_w}}  "
        for j in range(n):
            row += f"{cm[i, j]:>{col_w + 2}d}"
        lines.append(row)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the evaluation script.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    args = parse_args(argv)

    # -- Validate paths ----------------------------------------------------
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    data_path = Path(args.data)
    if not data_path.is_dir():
        print(f"Error: Data directory does not exist: {data_path}", file=sys.stderr)
        return 1

    # -- Resolve device ----------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # -- Load model --------------------------------------------------------
    print(f"Loading model from: {model_path}")
    ckpt = load_checkpoint(model_path, device)
    model = PhaseClassifier.from_save_dict(ckpt["model"], device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Phases: {model.phase_names}")
    print(f"  Parameters: {n_params:,}")
    if "epoch" in ckpt:
        print(f"  Checkpoint epoch: {ckpt['epoch'] + 1}")

    # -- Load data ---------------------------------------------------------
    from ebsd_ai.data.training_store import TrainingStore

    store = TrainingStore(local_path=data_path)
    n_samples = len(store)
    if n_samples == 0:
        print(f"Error: No samples found in {data_path}", file=sys.stderr)
        return 1

    print(f"\nLoaded TrainingStore: {n_samples} samples from {data_path}")
    stats = store.get_dataset_stats()
    print(f"  Phases: {stats['samples_per_phase']}")
    print(f"  Samples with EDS: {stats['samples_with_eds']}")
    print(f"  Samples without EDS: {stats['samples_without_eds']}")

    # No EDS dropout or augmentation for evaluation
    dataset = EBSDPhaseDataset(
        store,
        phase_names=model.phase_names,
        eds_dropout_rate=0.0,
    )

    # -- Optional split ----------------------------------------------------
    if args.split > 0:
        _, eval_ds = train_val_split(dataset, val_fraction=args.split)
        print(f"\nEvaluating on held-out split: {len(eval_ds)} samples "
              f"({args.split:.0%} of {n_samples})")
    else:
        eval_ds = dataset
        print(f"\nEvaluating on full dataset: {len(eval_ds)} samples")

    # -- Evaluate ----------------------------------------------------------
    evaluator = Evaluator(
        model=model,
        device=device,
        k=args.top_k,
        n_calibration_bins=args.calibration_bins,
        batch_size=args.batch_size,
    )
    result = evaluator.evaluate(eval_ds)

    # -- Print report ------------------------------------------------------
    print()
    print(format_classification_report(result))

    # -- Confusion matrix --------------------------------------------------
    if result.n_samples > 0:
        print("\nConfusion Matrix (rows = true, columns = predicted):")
        print(_format_confusion_matrix(result.confusion_matrix, result.phase_names))

        # Normalized confusion matrix
        norm_cm = result.normalized_confusion_matrix()
        print("\nNormalized Confusion Matrix (per-row percentages):")
        n = len(result.phase_names)
        col_w = max(len(name) for name in result.phase_names)
        col_w = max(col_w, 6)

        header = " " * (col_w + 2)
        for name in result.phase_names:
            header += f"{name:>{col_w + 2}}"
        print(header)
        for i in range(n):
            row = f"{result.phase_names[i]:>{col_w}}  "
            for j in range(n):
                row += f"{norm_cm[i, j]:>{col_w + 1}.1%} "
            print(row)

    # -- Calibration summary -----------------------------------------------
    if result.n_samples > 0:
        print("\nCalibration Analysis:")
        print(f"  {'Bin':>12} {'Confidence':>12} {'Accuracy':>12} {'Count':>8}")
        print("  " + "-" * 46)
        for b in range(len(result.calibration_counts)):
            lo = result.calibration_bins[b]
            hi = result.calibration_bins[b + 1]
            count = result.calibration_counts[b]
            if count > 0:
                conf = result.calibration_confidence[b]
                acc = result.calibration_accuracy[b]
                print(
                    f"  {lo:.1f}-{hi:.1f}   "
                    f"  {conf:>10.4f}   {acc:>10.4f}   {count:>6d}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
