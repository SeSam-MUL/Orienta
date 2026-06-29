"""Main phase classification model.

Combines a :class:`PatternEncoder` (CNN), an :class:`EDSEncoder` (MLP), and
a :class:`MultimodalFusion` (attention-gated) module with a linear
classification head to predict crystal phases from EBSD data.

The model supports:
- Variable number of output phases (can grow via :meth:`add_phase`)
- Inference with missing EDS (the fusion attention learns to ignore it)
- Top-k predictions with associated probabilities
- Saving / loading with full config reproducibility
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ebsd_ai.config import ModelConfig
from ebsd_ai.features.eds_encoder import EDSEncoder
from ebsd_ai.features.fusion import MultimodalFusion
from ebsd_ai.features.pattern_encoder import PatternEncoder


class PhaseClassifier(nn.Module):
    """End-to-end phase classification from EBSD pattern + EDS + metadata.

    Parameters
    ----------
    config : ModelConfig
        Model hyperparameters (feature dims, dropout, n_phases, pattern_size).
    phase_names : list[str] or None
        Ordered list of phase names corresponding to class indices.
        If *None*, phases are labelled ``"phase_0"``, ``"phase_1"``, etc.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        phase_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # Phase registry: ordered list of phase names
        if phase_names is not None:
            if len(phase_names) != self.config.n_phases:
                raise ValueError(
                    f"len(phase_names)={len(phase_names)} != "
                    f"config.n_phases={self.config.n_phases}"
                )
            self._phase_names: list[str] = list(phase_names)
        else:
            self._phase_names = [
                f"phase_{i}" for i in range(self.config.n_phases)
            ]

        # Sub-modules
        self.pattern_encoder = PatternEncoder(
            feature_dim=self.config.pattern_feature_dim,
            pattern_size=self.config.pattern_size,
        )
        self.eds_encoder = EDSEncoder(
            feature_dim=self.config.eds_feature_dim,
            dropout=self.config.dropout,
        )
        self.fusion = MultimodalFusion(
            pattern_dim=self.config.pattern_feature_dim,
            eds_dim=self.config.eds_feature_dim,
            fused_dim=self.config.fused_feature_dim,
            dropout=self.config.dropout,
        )

        # Classification head
        self.classifier = nn.Linear(
            self.config.fused_feature_dim, self.config.n_phases
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase_names(self) -> list[str]:
        """Ordered list of known phase names."""
        return list(self._phase_names)

    @property
    def n_phases(self) -> int:
        """Current number of output classes."""
        return len(self._phase_names)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        pattern: torch.Tensor,
        eds_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: pattern + EDS → logits + attention.

        Parameters
        ----------
        pattern : torch.Tensor
            Shape ``(B, 1, H, W)`` normalised grayscale patterns.
        eds_input : torch.Tensor
            Shape ``(B, 108)`` concatenated EDS + detector + has_eds flag
            (as produced by :meth:`EDSEncoder.build_input`).

        Returns
        -------
        logits : torch.Tensor
            Shape ``(B, n_phases)`` raw classification logits.
        alpha : torch.Tensor
            Shape ``(B, 1)`` fusion attention weight (1 → trusts pattern).
        """
        pattern_feat = self.pattern_encoder(pattern)  # (B, pattern_dim)
        eds_feat = self.eds_encoder(eds_input)  # (B, eds_dim)
        fused, alpha = self.fusion(pattern_feat, eds_feat)  # (B, fused_dim)
        logits = self.classifier(fused)  # (B, n_phases)
        return logits, alpha

    # ------------------------------------------------------------------
    # Convenience inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_top_k(
        self,
        pattern: torch.Tensor,
        eds_input: torch.Tensor,
        k: int = 3,
    ) -> list[list[tuple[str, float]]]:
        """Return top-k phase predictions per sample.

        Parameters
        ----------
        pattern : torch.Tensor
            Shape ``(B, 1, H, W)``.
        eds_input : torch.Tensor
            Shape ``(B, 108)``.
        k : int
            Number of top predictions to return per sample.

        Returns
        -------
        list[list[tuple[str, float]]]
            For each sample in the batch, a list of ``(phase_name, prob)``
            pairs sorted by descending probability.
        """
        was_training = self.training
        self.eval()
        try:
            logits, _ = self.forward(pattern, eds_input)
            probs = F.softmax(logits, dim=1)  # (B, n_phases)
            k = min(k, self.n_phases)
            top_probs, top_idx = torch.topk(probs, k, dim=1)

            results: list[list[tuple[str, float]]] = []
            for b in range(pattern.size(0)):
                sample_preds: list[tuple[str, float]] = []
                for j in range(k):
                    idx = int(top_idx[b, j].item())
                    prob = float(top_probs[b, j].item())
                    sample_preds.append((self._phase_names[idx], prob))
                results.append(sample_preds)
            return results
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def get_attention_weights(
        self,
        pattern: torch.Tensor,
        eds_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return fusion attention weights (pattern trust).

        Parameters
        ----------
        pattern : torch.Tensor
            Shape ``(B, 1, H, W)``.
        eds_input : torch.Tensor
            Shape ``(B, 108)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)`` alpha values.  ``eds_contribution = 1 - alpha``.
        """
        was_training = self.training
        self.eval()
        try:
            _, alpha = self.forward(pattern, eds_input)
            return alpha
        finally:
            if was_training:
                self.train()

    # ------------------------------------------------------------------
    # Dynamic phase management
    # ------------------------------------------------------------------

    def add_phase(self, phase_name: str) -> int:
        """Add a new phase class, expanding the classifier head.

        The new class weight row is zero-initialised so the model initially
        assigns near-zero probability to the new phase until fine-tuned.

        Parameters
        ----------
        phase_name : str
            Name of the new phase.

        Returns
        -------
        int
            Index of the newly added phase.

        Raises
        ------
        ValueError
            If *phase_name* already exists.
        """
        if phase_name in self._phase_names:
            raise ValueError(f"Phase '{phase_name}' already exists")

        old_weight = self.classifier.weight.data  # (old_n, fused_dim)
        old_bias = self.classifier.bias.data  # (old_n,)
        old_n = old_weight.size(0)
        fused_dim = old_weight.size(1)

        # Create new linear layer with one more output
        new_classifier = nn.Linear(fused_dim, old_n + 1)
        new_classifier.weight.data[:old_n] = old_weight
        new_classifier.bias.data[:old_n] = old_bias
        # Zero-init the new row so it doesn't disrupt existing predictions
        new_classifier.weight.data[old_n].zero_()
        new_classifier.bias.data[old_n].zero_()

        self.classifier = new_classifier
        self._phase_names.append(phase_name)
        self.config.n_phases = len(self._phase_names)
        return old_n

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_save_dict(self) -> dict[str, object]:
        """Build a checkpoint dictionary for :func:`torch.save`.

        Returns
        -------
        dict
            Contains ``state_dict``, ``config``, and ``phase_names``.
        """
        return {
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "phase_names": list(self._phase_names),
        }

    @classmethod
    def from_save_dict(
        cls,
        save_dict: dict[str, object],
        device: Optional[torch.device] = None,
    ) -> PhaseClassifier:
        """Reconstruct a model from a checkpoint dictionary.

        Parameters
        ----------
        save_dict : dict
            As produced by :meth:`get_save_dict`.
        device : torch.device or None
            Device to place the model on. Defaults to CPU.

        Returns
        -------
        PhaseClassifier
            Restored model with weights loaded.
        """
        config = ModelConfig.from_dict(save_dict["config"])  # type: ignore[arg-type]
        phase_names = list(save_dict["phase_names"])  # type: ignore[call-overload]
        model = cls(config=config, phase_names=phase_names)
        model.load_state_dict(save_dict["state_dict"])  # type: ignore[arg-type]
        if device is not None:
            model = model.to(device)
        return model
