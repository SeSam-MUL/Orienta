"""EBSD-AI: Machine Learning Phase Classifier for EBSD Patterns.

Public API — import directly from ``ebsd_ai``:

.. code-block:: python

    from ebsd_ai import (
        PhasePredictor,
        TrainingStore,
        PhaseClassifier,
        PatternEnhancer,
        Trainer,
        Evaluator,
        OnlineLearner,
        ServerSync,
    )
"""

__version__ = "0.1.0"

# -- Core public API (lazy re-exports) ------------------------------------

from ebsd_ai.config import (
    DataConfig,
    DataSource,
    DetectorConvention,
    DetectorInfo,
    DetectorManufacturer,
    ModelConfig,
    PhasePrediction,
    TrainingConfig,
    mad_to_confidence,
)
from ebsd_ai.data.dataset import EBSDPhaseDataset
from ebsd_ai.data.scan_import import ImportResult, ScanData, parse_scan
from ebsd_ai.data.training_store import TrainingStore
from ebsd_ai.inference.predictor import PhasePredictor, ScanPredictionResult
from ebsd_ai.models.losses import CombinedLoss
from ebsd_ai.models.pattern_enhancer import PatternEnhancer
from ebsd_ai.models.phase_classifier import PhaseClassifier
from ebsd_ai.sync.server_sync import ServerSync
from ebsd_ai.training.evaluator import Evaluator
from ebsd_ai.training.online_learner import OnlineLearner
from ebsd_ai.training.trainer import Trainer

__all__ = [
    # Inference
    "PhasePredictor",
    "ScanPredictionResult",
    "PhasePrediction",
    # Models
    "PhaseClassifier",
    "PatternEnhancer",
    "CombinedLoss",
    # Data
    "TrainingStore",
    "EBSDPhaseDataset",
    "ScanData",
    "ImportResult",
    "parse_scan",
    # Training
    "Trainer",
    "Evaluator",
    "OnlineLearner",
    # Config
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "DetectorInfo",
    "DetectorConvention",
    "DetectorManufacturer",
    "DataSource",
    "mad_to_confidence",
    # Sync
    "ServerSync",
]
