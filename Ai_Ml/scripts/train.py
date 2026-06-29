"""Train the EBSD phase classifier from TrainingStore data.

Usage
-----
.. code-block:: bash

    .venv/bin/python scripts/train.py --data ./training_data
    .venv/bin/python scripts/train.py --data ./data --output ckpt
    .venv/bin/python scripts/train.py --data ./data --resume ckpt/last.pt

The script:
1. Loads training data from a :class:`TrainingStore` directory
2. Creates (or resumes) a :class:`PhaseClassifier`
3. Trains with early stopping and checkpoint saving
4. Evaluates on a held-out validation set
5. Prints a classification report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ebsd_ai.config import ModelConfig, TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset, train_val_split
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.training.evaluator import Evaluator, format_classification_report
from ebsd_ai.training.trainer import Trainer, load_checkpoint


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
        description="Train an EBSD phase classifier on TrainingStore data.",
    )

    # Required
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to TrainingStore directory containing shard_*.h5 files.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum training epochs (default: 50)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Peak learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15).",
    )
    parser.add_argument(
        "--eds-dropout",
        type=float,
        default=0.3,
        help="EDS dropout rate during training (default: 0.3).",
    )

    # Model
    parser.add_argument(
        "--pattern-dim",
        type=int,
        default=256,
        help="Pattern encoder output dimension (default: 256).",
    )
    parser.add_argument(
        "--eds-dim",
        type=int,
        default=64,
        help="EDS encoder output dimension (default: 64).",
    )
    parser.add_argument(
        "--fused-dim",
        type=int,
        default=128,
        help="Fused feature dimension (default: 128).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability (default: 0.3).",
    )

    # Output / device
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints (default: checkpoints).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device: "auto", "cpu", or "cuda" (default: auto).',
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )

    # Evaluation
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k for evaluation accuracy (default: 3).",
    )

    return parser.parse_args(argv)


def _progress_callback(
    epoch: int, total: int, train_loss: float, val_loss: float,
) -> None:
    """Print a one-line progress update per epoch."""
    bar_len = 20
    filled = int(bar_len * (epoch + 1) / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(
        f"  [{bar}] Epoch {epoch + 1:>3d}/{total} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the training script.

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

    # -- Validate data path ------------------------------------------------
    data_path = Path(args.data)
    if not data_path.is_dir():
        print(f"Error: Data directory does not exist: {data_path}", file=sys.stderr)
        return 1

    # -- Load data ---------------------------------------------------------
    from ebsd_ai.data.training_store import TrainingStore

    store = TrainingStore(local_path=data_path)
    n_samples = len(store)
    if n_samples == 0:
        print(f"Error: No training samples found in {data_path}", file=sys.stderr)
        return 1

    print(f"Loaded TrainingStore: {n_samples} samples from {data_path}")
    stats = store.get_dataset_stats()
    print(f"  Phases: {stats['samples_per_phase']}")
    print(f"  Samples with EDS: {stats['samples_with_eds']}")
    print(f"  Samples without EDS: {stats['samples_without_eds']}")

    dataset = EBSDPhaseDataset(store, eds_dropout_rate=args.eds_dropout)
    phase_names = dataset.phase_names
    n_phases = dataset.num_phases
    print(f"  Phase mapping: {dict(enumerate(phase_names))}")

    # -- Create or resume model --------------------------------------------
    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        eds_dropout_rate=args.eds_dropout,
        val_fraction=args.val_fraction,
    )

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            print(f"Error: Checkpoint not found: {resume_path}", file=sys.stderr)
            return 1

        print(f"\nResuming from checkpoint: {resume_path}")
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available() else
            args.device if args.device != "auto" else "cpu"
        )
        ckpt = load_checkpoint(resume_path, device)
        model = PhaseClassifier.from_save_dict(ckpt["model"], device=device)
        print(f"  Model phases: {model.phase_names}")
        print(f"  Resuming from epoch {ckpt['epoch'] + 1}")
    else:
        model_config = ModelConfig(
            n_phases=n_phases,
            pattern_feature_dim=args.pattern_dim,
            eds_feature_dim=args.eds_dim,
            fused_feature_dim=args.fused_dim,
            dropout=args.dropout,
        )
        model = PhaseClassifier(config=model_config, phase_names=phase_names)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nCreated model: {n_phases} phases, {n_params:,} parameters")

    # -- Train -------------------------------------------------------------
    trainer = Trainer(
        model=model,
        config=training_config,
        output_dir=args.output,
        device=args.device,
        progress_callback=_progress_callback,
    )

    if args.resume:
        trainer.resume(args.resume)

    print(f"\nTraining (max {args.epochs} epochs, patience {args.patience})...")
    print(f"  Device: {trainer.device}")
    print(f"  Output: {args.output}")
    print()

    # Split dataset (trainer does this internally, but we do it here
    # to also have the val set for post-training evaluation)
    train_ds, val_ds = train_val_split(
        dataset, val_fraction=args.val_fraction
    )
    result = trainer.train(train_ds, val_dataset=val_ds)

    # -- Training summary --------------------------------------------------
    print("\nTraining complete:")
    print(f"  Epochs completed: {result.epochs_completed}")
    print(f"  Best epoch:       {result.best_epoch + 1}")
    print(f"  Best val loss:    {result.best_val_loss:.4f}")
    print(f"  Early stopped:    {'Yes' if result.early_stopped else 'No'}")
    print(f"  Duration:         {result.elapsed_seconds:.1f}s")

    # -- Evaluate on validation set ----------------------------------------
    print(f"\nEvaluating on validation set ({len(val_ds)} samples)...")

    # Load the best checkpoint for evaluation
    best_path = Path(args.output) / "best.pt"
    if best_path.is_file():
        best_ckpt = load_checkpoint(best_path, trainer.device)
        eval_model = PhaseClassifier.from_save_dict(
            best_ckpt["model"], device=trainer.device
        )
    else:
        eval_model = model

    evaluator = Evaluator(
        model=eval_model,
        device=trainer.device,
        k=args.top_k,
        batch_size=args.batch_size,
    )
    eval_result = evaluator.evaluate(val_ds)

    print()
    print(format_classification_report(eval_result))

    print(f"\nCheckpoints saved to: {Path(args.output).resolve()}")
    print("  best.pt  — best validation loss")
    print("  last.pt  — final epoch")

    return 0


if __name__ == "__main__":
    sys.exit(main())
