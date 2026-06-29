"""Training loop for EBSD pattern denoising.

Trains a PatternEnhancer U-Net on denoising pairs stored in a
TrainingStore.  Follows the same conventions as the existing
classification Trainer: Adam + OneCycleLR, AMP on GPU, early
stopping, best/last checkpoints, and an optional progress callback.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from ebsd_ai.config import DenoisingTrainingConfig
from ebsd_ai.data.denoising_dataset import DenoisingDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.models.denoising_loss import DenoisingLoss
from ebsd_ai.models.pattern_enhancer import PatternEnhancer

logger = logging.getLogger(__name__)


@dataclass
class DenoisingTrainingResult:
    """Returned by DenoisingTrainer.train()."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_completed: int = 0
    early_stopped: bool = False
    elapsed_seconds: float = 0.0


ProgressCallback = Callable[[int, int, float, float], None]


class DenoisingTrainer:
    """Train a PatternEnhancer for denoising.

    Parameters
    ----------
    config : DenoisingTrainingConfig or None
        Hyperparameters.
    output_dir : str or Path
        Directory for best.pt and last.pt checkpoints.
    device : str or torch.device
        "auto" picks CUDA if available.
    progress_callback : callable, optional
        Called after each epoch with (epoch, total, train_loss, val_loss).
    """

    def __init__(
        self,
        config: DenoisingTrainingConfig | None = None,
        output_dir: str | Path = "checkpoints",
        device: str | torch.device = "auto",
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config or DenoisingTrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    def train(self, store: TrainingStore) -> DenoisingTrainingResult:
        """Run the full training loop.

        Parameters
        ----------
        store : TrainingStore
            Must contain denoising pairs (pair_type="denoising").

        Returns
        -------
        DenoisingTrainingResult
        """
        cfg = self.config
        dataset = DenoisingDataset(store, augment=True)

        if len(dataset) < cfg.min_training_pairs:
            raise ValueError(
                f"Need at least {cfg.min_training_pairs} denoising pairs, "
                f"found {len(dataset)}"
            )

        # Train/val split
        val_size = max(1, int(len(dataset) * cfg.val_fraction))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            drop_last=False, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )

        # Model
        model = PatternEnhancer(residual_learning=cfg.residual_learning)
        model = model.to(self.device)

        # Loss
        criterion = DenoisingLoss(
            l1_weight=cfg.l1_weight, ssim_weight=cfg.ssim_weight,
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * cfg.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.learning_rate,
            total_steps=max(total_steps, 1),
        )

        # AMP
        use_amp = cfg.mixed_precision and self.device.type == "cuda"
        scaler = GradScaler("cuda", enabled=use_amp)

        # Training state
        best_val_loss = float("inf")
        patience_counter = 0
        result = DenoisingTrainingResult()
        t0 = time.time()

        for epoch in range(cfg.epochs):
            # --- Train ---
            model.train()
            train_loss_sum = 0.0
            train_batches = 0

            for batch in train_loader:
                inp = batch["input"].to(self.device)
                target = batch["target"].to(self.device)
                det = batch["detector_encoding"].to(self.device)

                optimizer.zero_grad()
                with autocast("cuda", enabled=use_amp):
                    pred = model(inp, detector_encoding=det)
                    loss = criterion(pred, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss_sum += loss.item()
                train_batches += 1

            avg_train = train_loss_sum / max(train_batches, 1)
            result.train_losses.append(avg_train)

            # --- Validate ---
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    inp = batch["input"].to(self.device)
                    target = batch["target"].to(self.device)
                    det = batch["detector_encoding"].to(self.device)

                    with autocast("cuda", enabled=use_amp):
                        pred = model(inp, detector_encoding=det)
                        loss = criterion(pred, target)

                    val_loss_sum += loss.item()
                    val_batches += 1

            avg_val = val_loss_sum / max(val_batches, 1)
            result.val_losses.append(avg_val)
            result.epochs_completed = epoch + 1

            # Callback
            if self.progress_callback is not None:
                self.progress_callback(epoch + 1, cfg.epochs, avg_train, avg_val)

            logger.info(
                "Epoch %d/%d  train=%.5f  val=%.5f",
                epoch + 1, cfg.epochs, avg_train, avg_val,
            )

            # Checkpoint + early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                result.best_val_loss = avg_val
                result.best_epoch = epoch
                patience_counter = 0
                torch.save(model.get_save_dict(), self.output_dir / "best.pt")
            else:
                patience_counter += 1

            torch.save(model.get_save_dict(), self.output_dir / "last.pt")

            if patience_counter >= cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                result.early_stopped = True
                break

        result.elapsed_seconds = time.time() - t0
        return result
