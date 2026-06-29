"""Training loop for contrastive embedding learning.

Trains an EmbeddingEncoder on simulated EBSD patterns with SO(3)-aware
contrastive loss.  AdamW + CosineAnnealingWarmRestarts, AMP on GPU,
checkpoint saving, and progress callbacks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ebsd_ai.config import EmbeddingConfig
from ebsd_ai.data.simulation_dataset import SimulationDataset
from ebsd_ai.models.contrastive_loss import AngularContrastiveLoss
from ebsd_ai.models.embedding_encoder import EmbeddingEncoder

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingTrainingResult:
    """Result of embedding training."""

    final_loss: float = 0.0
    best_loss: float = float("inf")
    epochs_completed: int = 0
    best_checkpoint: str | None = None


class EmbeddingTrainer:
    """Contrastive embedding trainer.

    Parameters
    ----------
    config : EmbeddingConfig
        Training configuration.
    progress_callback : callable or None
        Called with (epoch, loss) after each epoch.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        progress_callback: Callable[[int, float], None] | None = None,
    ) -> None:
        self.cfg = config
        self.progress_callback = progress_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        patterns: np.ndarray,
        orientations: np.ndarray,
        phase_ids: np.ndarray,
        checkpoint_dir: str | None = None,
    ) -> EmbeddingTrainingResult:
        """Run contrastive training loop.

        Parameters
        ----------
        patterns : np.ndarray
            (N, H, W) simulated patterns.
        orientations : np.ndarray
            (N, 4) unit quaternions.
        phase_ids : np.ndarray
            (N,) phase identifiers.
        checkpoint_dir : str or None
            Directory to save checkpoints.

        Returns
        -------
        EmbeddingTrainingResult
        """
        dataset = SimulationDataset(
            patterns=patterns,
            orientations=orientations,
            phase_ids=phase_ids,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=len(dataset) > self.cfg.batch_size,
            num_workers=0,
        )

        model = EmbeddingEncoder(embedding_dim=self.cfg.embedding_dim).to(self.device)
        loss_fn = AngularContrastiveLoss(
            temperature=self.cfg.temperature,
            pos_angle_deg=self.cfg.pos_angle_deg,
            neg_angle_deg=self.cfg.neg_angle_deg,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, self.cfg.epochs // 15), T_mult=2,
        )

        use_amp = self.cfg.mixed_precision and self.device.type == "cuda"
        scaler = GradScaler("cuda") if use_amp else None

        result = EmbeddingTrainingResult()

        for epoch in range(self.cfg.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                pat = batch["pattern"].to(self.device)
                ori = batch["orientation"].to(self.device)

                optimizer.zero_grad()

                if use_amp:
                    with autocast("cuda"):
                        emb = model(pat)
                        loss = loss_fn(emb, ori)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    emb = model(pat)
                    loss = loss_fn(emb, ori)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            result.final_loss = avg_loss
            result.epochs_completed = epoch + 1

            if self.progress_callback is not None:
                self.progress_callback(epoch + 1, avg_loss)

            # Save best checkpoint
            if checkpoint_dir is not None and avg_loss < result.best_loss:
                result.best_loss = avg_loss
                ckpt_path = Path(checkpoint_dir) / "embedding_best.pt"
                torch.save(model.get_save_dict(), ckpt_path)
                result.best_checkpoint = str(ckpt_path)
                logger.info("Saved best checkpoint at epoch %d (loss=%.4f)", epoch + 1, avg_loss)

        return result
