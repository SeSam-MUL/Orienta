"""Continual learning from new training data.

Provides :class:`OnlineLearner` — a lightweight fine-tuning wrapper around
:class:`PhaseClassifier` that incrementally updates the model as new
high-CI indexing results are added to the :class:`TrainingStore`.

Design goals:

- **No catastrophic forgetting**: Uses experience replay (mixing old and new
  samples in each update batch) and an optional elastic weight consolidation
  (EWC) penalty that discourages large changes to weights that were important
  for previously learned phases.
- **New phase support**: Automatically expands the classifier head when new
  phases appear in the store that the model doesn't know about yet.
- **Small learning rate**: Online updates use a much lower learning rate than
  initial training to avoid overwriting well-learned features.
- **Background-friendly**: Each :meth:`update` call runs a configurable number
  of mini-batches (not full epochs), so it can be called after each indexing
  run without blocking the GUI for long.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from ebsd_ai.config import TrainingConfig
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.models.losses import CombinedLoss
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.training.trainer import save_checkpoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OnlineLearnerConfig:
    """Configuration for online / continual learning updates.

    Parameters
    ----------
    learning_rate : float
        Learning rate for online updates (typically much lower than
        initial training).
    steps_per_update : int
        Number of gradient steps per :meth:`OnlineLearner.update` call.
    batch_size : int
        Mini-batch size for update steps.
    replay_fraction : float
        Fraction of each batch filled with randomly sampled old data
        (experience replay).  ``0.5`` means half old, half new.
    ewc_lambda : float
        Elastic Weight Consolidation penalty strength.  ``0.0`` disables EWC.
    eds_dropout_rate : float
        EDS dropout rate during online training.
    weight_decay : float
        L2 regularization.
    max_samples_per_update : int
        Cap on how many new samples are used per update.  Prevents a single
        huge indexing run from dominating the update.
    """

    learning_rate: float = 1e-4
    steps_per_update: int = 50
    batch_size: int = 32
    replay_fraction: float = 0.5
    ewc_lambda: float = 0.0
    eds_dropout_rate: float = 0.3
    weight_decay: float = 1e-4
    max_samples_per_update: int = 5000

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        return {
            "learning_rate": self.learning_rate,
            "steps_per_update": self.steps_per_update,
            "batch_size": self.batch_size,
            "replay_fraction": self.replay_fraction,
            "ewc_lambda": self.ewc_lambda,
            "eds_dropout_rate": self.eds_dropout_rate,
            "weight_decay": self.weight_decay,
            "max_samples_per_update": self.max_samples_per_update,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> OnlineLearnerConfig:
        """Reconstruct from a plain dict."""
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**cast(dict[str, Any], filtered))


# ---------------------------------------------------------------------------
# Update result
# ---------------------------------------------------------------------------


@dataclass
class OnlineUpdateResult:
    """Returned by :meth:`OnlineLearner.update`.

    Parameters
    ----------
    steps_completed : int
        Number of gradient steps executed.
    mean_loss : float
        Average loss across all steps.
    new_phases_added : list[str]
        Phase names that were added to the model during this update.
    samples_used : int
        Total number of samples seen during the update.
    """

    steps_completed: int = 0
    mean_loss: float = 0.0
    new_phases_added: list[str] = field(default_factory=list)
    samples_used: int = 0


# ---------------------------------------------------------------------------
# EWC helpers
# ---------------------------------------------------------------------------


def _compute_fisher_diagonal(
    model: PhaseClassifier,
    dataset: EBSDPhaseDataset,
    device: torch.device,
    n_samples: int = 200,
) -> dict[str, torch.Tensor]:
    """Estimate diagonal Fisher Information Matrix via empirical samples.

    Parameters
    ----------
    model : PhaseClassifier
        The trained model to compute Fisher for.
    dataset : EBSDPhaseDataset
        Dataset to sample from.
    device : torch.device
        Computation device.
    n_samples : int
        Number of samples to estimate Fisher over.

    Returns
    -------
    dict[str, torch.Tensor]
        Parameter name -> diagonal Fisher estimate.
    """
    model.eval()
    fisher: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)

    n = min(n_samples, len(dataset))
    if n == 0:
        return fisher

    indices = np.random.default_rng(0).choice(len(dataset), size=n, replace=False)

    for idx in indices:
        sample = dataset[int(idx)]
        pattern = sample["pattern"].unsqueeze(0).to(device)
        eds_input = sample["eds_input"].unsqueeze(0).to(device)

        model.zero_grad()
        logits, _ = model(pattern, eds_input)
        # Use log-likelihood of the predicted class
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        loss = -log_probs[0, pred[0]]
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2

    # Normalize
    for name in fisher:
        fisher[name] /= n

    return fisher


def _ewc_penalty(
    model: PhaseClassifier,
    fisher: dict[str, torch.Tensor],
    old_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute EWC penalty: sum of Fisher-weighted squared parameter diffs.

    Parameters
    ----------
    model : PhaseClassifier
        Current model state.
    fisher : dict[str, torch.Tensor]
        Diagonal Fisher from the previous task.
    old_params : dict[str, torch.Tensor]
        Parameter snapshot from before online update.

    Returns
    -------
    torch.Tensor
        Scalar EWC penalty.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in fisher and name in old_params:
            diff = param - old_params[name]
            penalty = penalty + (fisher[name] * diff ** 2).sum()
    return penalty


# Type alias for progress callback
OnlineProgressCallback = Callable[[int, int, float], None]
"""Signature: ``(step, total_steps, step_loss) -> None``."""


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------


class OnlineLearner:
    """Incrementally fine-tune a :class:`PhaseClassifier` on new data.

    Parameters
    ----------
    model : PhaseClassifier
        The model to update in-place.
    store : TrainingStore
        Training data backend.
    config : OnlineLearnerConfig, optional
        Online learning hyperparameters.
    device : str or torch.device
        ``"auto"`` picks CUDA if available, else CPU.
    checkpoint_dir : str or Path, optional
        If provided, save a checkpoint after each update.
    progress_callback : callable, optional
        Called after each gradient step with ``(step, total_steps, loss)``.
    """

    def __init__(
        self,
        model: PhaseClassifier,
        store: TrainingStore,
        config: OnlineLearnerConfig | None = None,
        device: str | torch.device = "auto",
        checkpoint_dir: str | Path | None = None,
        progress_callback: OnlineProgressCallback | None = None,
    ) -> None:
        self.config = config or OnlineLearnerConfig()
        self.store = store
        self.progress_callback = progress_callback

        # Device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        # Checkpoint directory
        self.checkpoint_dir: Path | None = None
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # EWC state (computed lazily on first update if lambda > 0)
        self._fisher: dict[str, torch.Tensor] | None = None
        self._old_params: dict[str, torch.Tensor] | None = None

        # Track how many updates have been performed
        self.update_count: int = 0

    # ------------------------------------------------------------------
    # Phase expansion
    # ------------------------------------------------------------------

    def _expand_phases_if_needed(
        self, phase_names: list[str]
    ) -> list[str]:
        """Add any new phases from the store to the model.

        Parameters
        ----------
        phase_names : list[str]
            All phase names currently in the store.

        Returns
        -------
        list[str]
            List of newly added phase names (may be empty).
        """
        known = set(self.model.phase_names)
        new_phases: list[str] = []
        for name in phase_names:
            if name not in known:
                self.model.add_phase(name)
                new_phases.append(name)
                known.add(name)
        return new_phases

    # ------------------------------------------------------------------
    # EWC
    # ------------------------------------------------------------------

    def compute_ewc_state(
        self, dataset: EBSDPhaseDataset, n_samples: int = 200
    ) -> None:
        """Compute and store Fisher / parameter snapshot for EWC.

        Call this after initial training (or after any update) to
        establish the "anchor" state that EWC will try to preserve.

        Parameters
        ----------
        dataset : EBSDPhaseDataset
            Dataset to estimate Fisher over.
        n_samples : int
            Number of samples for Fisher estimation.
        """
        self._fisher = _compute_fisher_diagonal(
            self.model, dataset, self.device, n_samples
        )
        self._old_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        new_indices: list[int] | None = None,
    ) -> OnlineUpdateResult:
        """Run one round of online learning.

        Parameters
        ----------
        new_indices : list[int], optional
            Store indices of the newly added samples.  If *None*, uses
            all samples in the store (useful for first update or when
            indices are not tracked externally).

        Returns
        -------
        OnlineUpdateResult
            Summary of the update.
        """
        result = OnlineUpdateResult()
        store_size = len(self.store)

        if store_size == 0:
            return result

        # Discover and expand phases
        store_phases = self.store.get_phase_names()
        result.new_phases_added = self._expand_phases_if_needed(store_phases)

        # Build dataset over entire store (needed for replay)
        all_phase_names = self.model.phase_names
        full_dataset = EBSDPhaseDataset(
            store=self.store,
            phase_names=all_phase_names,
            eds_dropout_rate=self.config.eds_dropout_rate,
            augmentation=None,
            seed=42 + self.update_count,
        )

        if len(full_dataset) == 0:
            return result

        # Compute EWC if enabled and not yet computed
        if self.config.ewc_lambda > 0 and self._fisher is None:
            self.compute_ewc_state(full_dataset)

        # Determine new vs old indices
        if new_indices is not None:
            # Cap new indices
            capped = new_indices[: self.config.max_samples_per_update]
            new_set = set(capped)
            old_indices = [i for i in range(store_size) if i not in new_set]
        else:
            # Treat all as "new" (no special replay split)
            new_set = set()
            old_indices = []

        # Build mixed DataLoader with experience replay
        loader = self._build_replay_loader(
            full_dataset, list(new_set), old_indices
        )

        # Setup optimizer (fresh each update for clean state)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Loss
        criterion = CombinedLoss()
        criterion.set_weights_from_labels(
            full_dataset._labels, full_dataset.num_phases
        )

        # Training steps
        self.model.train()
        total_loss = 0.0
        steps = 0
        target_steps = self.config.steps_per_update

        loader_iter = iter(loader)
        while steps < target_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Re-create iterator for another pass
                loader_iter = iter(loader)
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break  # Empty loader

            pattern = batch["pattern"].to(self.device)
            eds_input = batch["eds_input"].to(self.device)
            targets = batch["phase_label"].to(self.device)

            optimizer.zero_grad()
            logits, _alpha = self.model(pattern, eds_input)
            loss = criterion(logits, targets)

            # EWC penalty
            if (
                self.config.ewc_lambda > 0
                and self._fisher is not None
                and self._old_params is not None
            ):
                ewc_loss = _ewc_penalty(
                    self.model, self._fisher, self._old_params
                )
                loss = loss + self.config.ewc_lambda * ewc_loss

            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            total_loss += step_loss
            steps += 1
            result.samples_used += pattern.size(0)

            if self.progress_callback is not None:
                self.progress_callback(steps, target_steps, step_loss)

        result.steps_completed = steps
        result.mean_loss = total_loss / max(steps, 1)

        # Update EWC anchor after this update
        if self.config.ewc_lambda > 0:
            self.compute_ewc_state(full_dataset)

        # Save checkpoint
        if self.checkpoint_dir is not None:
            self._save_online_checkpoint()

        self.update_count += 1
        return result

    # ------------------------------------------------------------------
    # Replay DataLoader
    # ------------------------------------------------------------------

    def _build_replay_loader(
        self,
        full_dataset: EBSDPhaseDataset,
        new_indices: list[int],
        old_indices: list[int],
    ) -> DataLoader:
        """Build a DataLoader that mixes new and old samples.

        If no new/old split is specified (both empty or new covers all),
        returns a loader over the full dataset.

        Parameters
        ----------
        full_dataset : EBSDPhaseDataset
            The complete dataset.
        new_indices : list[int]
            Global store indices of new samples.
        old_indices : list[int]
            Global store indices of old samples.

        Returns
        -------
        DataLoader
        """
        if not new_indices or not old_indices:
            # No meaningful split — use full dataset with balanced sampling
            return DataLoader(
                full_dataset,
                batch_size=self.config.batch_size,
                sampler=full_dataset.make_sampler(
                    num_samples=self.config.steps_per_update
                    * self.config.batch_size
                ),
                drop_last=False,
            )

        # Build mixed sampling weights: oversample new data relative to old
        # to ensure new data is well represented while still replaying old
        replay_frac = self.config.replay_fraction
        n_new = len(new_indices)
        n_old = len(old_indices)

        new_set = set(new_indices)
        sample_weights = []
        for i in range(len(full_dataset)):
            global_idx = full_dataset._indices[i]
            if global_idx in new_set:
                # Weight for new samples: (1 - replay_frac) / n_new
                sample_weights.append(
                    (1.0 - replay_frac) / max(n_new, 1)
                )
            else:
                # Weight for old samples: replay_frac / n_old
                sample_weights.append(
                    replay_frac / max(n_old, 1)
                )

        from torch.utils.data import WeightedRandomSampler

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=self.config.steps_per_update * self.config.batch_size,
            replacement=True,
        )

        return DataLoader(
            full_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_online_checkpoint(self) -> None:
        """Save model checkpoint to the configured directory."""
        if self.checkpoint_dir is None:
            return

        # Use a dummy training config for the checkpoint format
        training_config = TrainingConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
        )

        save_checkpoint(
            path=self.checkpoint_dir / "online_latest.pt",
            model=self.model,
            optimizer=torch.optim.Adam(self.model.parameters()),
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                torch.optim.Adam(self.model.parameters()),
                lr_lambda=lambda _: 1.0,
            ),
            epoch=self.update_count,
            best_val_loss=0.0,
            training_config=training_config,
        )
