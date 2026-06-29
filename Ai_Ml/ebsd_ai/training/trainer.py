"""Training loop with checkpointing, early stopping, and mixed precision.

Provides :class:`Trainer` — the main entry point for training a
:class:`PhaseClassifier` on an :class:`EBSDPhaseDataset`.

Features:
- Adam optimizer with OneCycleLR scheduler
- Automatic mixed precision (AMP) on GPU
- Early stopping on validation loss
- Best + last checkpoint saving
- Optional per-epoch callback for GUI progress reporting
- Resume from checkpoint
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from ebsd_ai.config import TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset, train_val_split
from ebsd_ai.models.losses import CombinedLoss
from ebsd_ai.models.phase_classifier import PhaseClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Returned by :meth:`Trainer.train`.

    Parameters
    ----------
    train_losses : list[float]
        Per-epoch mean training loss.
    val_losses : list[float]
        Per-epoch mean validation loss.
    best_epoch : int
        Epoch index with the lowest validation loss.
    best_val_loss : float
        Lowest validation loss achieved.
    epochs_completed : int
        Number of epochs actually run (may be < max if early stopped).
    early_stopped : bool
        Whether training terminated due to early stopping.
    elapsed_seconds : float
        Total wall-clock training time.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_completed: int = 0
    early_stopped: bool = False
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: PhaseClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    training_config: TrainingConfig,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    path : Path
        File path for the checkpoint.
    model : PhaseClassifier
        The model to save.
    optimizer : torch.optim.Optimizer
        Optimizer state.
    scheduler : LRScheduler
        Scheduler state.
    epoch : int
        Current epoch number.
    best_val_loss : float
        Best validation loss so far.
    training_config : TrainingConfig
        Training hyperparameters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.get_save_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "training_config": training_config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:
    """Load a training checkpoint.

    Parameters
    ----------
    path : Path
        Path to the checkpoint file.
    device : torch.device
        Device to map tensors to.

    Returns
    -------
    dict
        Checkpoint contents.
    """
    data: dict = torch.load(path, map_location=device, weights_only=False)
    return data


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

# Type alias for the per-epoch callback
ProgressCallback = Callable[[int, int, float, float], None]
"""Signature: ``(epoch, total_epochs, train_loss, val_loss) -> None``."""


class Trainer:
    """Train a :class:`PhaseClassifier` on EBSD data.

    Parameters
    ----------
    model : PhaseClassifier
        The model to train.
    config : TrainingConfig
        Training hyperparameters.
    output_dir : str or Path
        Directory for checkpoints (``best.pt``, ``last.pt``).
    device : str or torch.device
        ``"auto"`` picks CUDA if available, else CPU.
    progress_callback : callable, optional
        Called after each epoch with ``(epoch, total, train_loss, val_loss)``.
    """

    def __init__(
        self,
        model: PhaseClassifier,
        config: TrainingConfig | None = None,
        output_dir: str | Path = "checkpoints",
        device: str | torch.device = "auto",
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

        # Device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        # Loss
        self.criterion = CombinedLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler placeholder (created per-train call because it depends
        # on the number of steps)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        # AMP
        self.use_amp = (
            self.config.mixed_precision and self.device.type == "cuda"
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Early stopping state
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._start_epoch = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: EBSDPhaseDataset,
        val_dataset: EBSDPhaseDataset | None = None,
    ) -> TrainingResult:
        """Run the full training loop.

        If *val_dataset* is not provided, the dataset is split using
        ``TrainingConfig.val_fraction``.

        Parameters
        ----------
        dataset : EBSDPhaseDataset
            Training data.
        val_dataset : EBSDPhaseDataset, optional
            Validation data.  Split from *dataset* if not given.

        Returns
        -------
        TrainingResult
        """
        if len(dataset) == 0:
            raise ValueError(
                "Cannot train on an empty dataset. Add samples to the "
                "TrainingStore before calling train()."
            )

        # Split if no val set provided
        if val_dataset is None:
            dataset, val_dataset = train_val_split(
                dataset, val_fraction=self.config.val_fraction
            )

        # Set class weights from training labels and move to device
        self.criterion.set_weights_from_labels(
            dataset._labels, dataset.num_phases
        )
        self.criterion = self.criterion.to(self.device)

        # DataLoaders (num_workers=0 because h5py is not fork-safe).
        # BatchNorm1d (in the EDS/feature encoders) raises
        # "Expected more than 1 value per channel" on a size-1 batch in
        # train mode. Drop the last (incomplete) batch whenever we still
        # have at least one full batch, so BN never sees a singleton tail.
        # The sampler reshuffles every epoch, so no sample is permanently
        # excluded. When the whole train split is smaller than one batch we
        # keep it (single batch of size >=2, guarded in train_model).
        _drop_last_train = len(dataset) >= self.config.batch_size
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=dataset.make_sampler(),
            drop_last=_drop_last_train,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # Scheduler (OneCycleLR over total steps)
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=max(total_steps, 1),
        )

        result = TrainingResult()
        t0 = time.monotonic()

        for epoch in range(self._start_epoch, self.config.epochs):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate(val_loader)

            result.train_losses.append(train_loss)
            result.val_losses.append(val_loss)
            result.epochs_completed = epoch + 1

            # Checkpoint: last
            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self._best_val_loss,
                self.config,
            )

            # Best checkpoint
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                result.best_epoch = epoch
                result.best_val_loss = val_loss
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self._best_val_loss,
                    self.config,
                )
            else:
                self._patience_counter += 1

            # Progress callback
            if self.progress_callback is not None:
                self.progress_callback(
                    epoch, self.config.epochs, train_loss, val_loss
                )

            # Early stopping
            if self._patience_counter >= self.config.early_stopping_patience:
                result.early_stopped = True
                break

        result.elapsed_seconds = time.monotonic() - t0
        return result

    def _train_one_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch.  Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            pattern = batch["pattern"].to(self.device)
            eds_input = batch["eds_input"].to(self.device)
            targets = batch["phase_label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                logits, _alpha = self.model(pattern, eds_input)
                loss = self.criterion(logits, targets)

            # Skip batch if loss is non-finite (NaN/Inf)
            loss_val = loss.item()
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss (%s), skipping batch", loss_val)
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()
            # Gradient clipping for numerical stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss_val
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Run one validation pass.  Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            pattern = batch["pattern"].to(self.device)
            eds_input = batch["eds_input"].to(self.device)
            targets = batch["phase_label"].to(self.device)

            logits, _alpha = self.model(pattern, eds_input)
            loss = self.criterion(logits, targets)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume(self, checkpoint_path: str | Path) -> None:
        """Restore trainer state from a checkpoint.

        Call this before :meth:`train` to continue training from a saved
        checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to a checkpoint file saved by :func:`save_checkpoint`.
        """
        ckpt = load_checkpoint(Path(checkpoint_path), self.device)

        # Restore model
        model_dict = ckpt["model"]
        self.model = PhaseClassifier.from_save_dict(
            model_dict, device=self.device
        )

        # Restore optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Restore state
        self._start_epoch = ckpt["epoch"] + 1
        self._best_val_loss = ckpt["best_val_loss"]
        self._patience_counter = 0
