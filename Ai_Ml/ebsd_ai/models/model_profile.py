"""Model profile for per-material-system ML configurations.

Stores metadata about trained models: phase names, architecture
settings, training statistics, and file paths for persistence.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Profile for a trained ML model (encoder or classifier).

    Parameters
    ----------
    name : str
        Profile name (e.g., "Fe-Cr-System").
    phase_names : list of str
        Names of phases this model handles.
    embedding_dim : int
        Embedding dimension.
    training_epochs : int
        Number of training epochs completed.
    best_loss : float
        Best validation/training loss achieved.
    encoder_path : str
        Path to encoder checkpoint.
    faiss_index_path : str
        Path to saved FAISS index directory.
    metrics : dict
        Additional metrics (accuracy, angular error, etc.).
    """

    name: str = ""
    phase_names: List[str] = field(default_factory=list)
    embedding_dim: int = 128
    training_epochs: int = 0
    best_loss: float = float("inf")
    encoder_path: str = ""
    faiss_index_path: str = ""
    metrics: dict = field(default_factory=dict)

    def validate_for_embedding(self) -> bool:
        """Check if profile is valid for embedding indexing (>= 1 phase)."""
        return len(self.phase_names) >= 1

    def validate_for_classification(self) -> bool:
        """Check if profile is valid for classification (>= 2 phases)."""
        return len(self.phase_names) >= 2

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "name": self.name,
            "phase_names": self.phase_names,
            "embedding_dim": self.embedding_dim,
            "training_epochs": self.training_epochs,
            "best_loss": self.best_loss,
            "encoder_path": self.encoder_path,
            "faiss_index_path": self.faiss_index_path,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelProfile":
        """Deserialize from dict."""
        return cls(
            name=d.get("name", ""),
            phase_names=d.get("phase_names", []),
            embedding_dim=d.get("embedding_dim", 128),
            training_epochs=d.get("training_epochs", 0),
            best_loss=d.get("best_loss", float("inf")),
            encoder_path=d.get("encoder_path", ""),
            faiss_index_path=d.get("faiss_index_path", ""),
            metrics=d.get("metrics", {}),
        )

    def save(self, directory: str) -> None:
        """Save profile metadata to directory as JSON."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta_path = d / f"{self.name}.json"
        with open(meta_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved profile '%s' to %s", self.name, meta_path)

    @classmethod
    def load(cls, directory: str) -> "ModelProfile":
        """Load first profile found in directory."""
        d = Path(directory)
        json_files = list(d.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No profile JSON found in {directory}")
        with open(json_files[0]) as f:
            data = json.load(f)
        return cls.from_dict(data)
