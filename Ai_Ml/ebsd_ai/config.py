"""Central configuration with dataclasses for the EBSD-AI system.

All configuration, enums, and result dataclasses used across the package
are defined here to avoid circular imports and provide a single source
of truth for defaults and type definitions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ELEMENTS = 92  # Atomic numbers 1 (H) through 92 (U)

DEFAULT_PATTERN_SIZE = 128  # All patterns resized to this square dimension

ELEMENT_SYMBOLS: list[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",
]

# Lookup from symbol to 0-based index
ELEMENT_INDEX: dict[str, int] = {sym: i for i, sym in enumerate(ELEMENT_SYMBOLS)}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DetectorConvention(enum.Enum):
    """Pattern Center convention used by different EBSD software."""

    OXFORD = "OXFORD"
    BRUKER = "BRUKER"
    EDAX = "EDAX"
    EMSOFT = "EMSOFT"
    KIKUCHIPY = "KIKUCHIPY"


class DetectorManufacturer(enum.Enum):
    """EBSD detector manufacturer."""

    OXFORD = "OXFORD"
    BRUKER = "BRUKER"
    EDAX = "EDAX"
    OTHER = "OTHER"


class DataSource(enum.Enum):
    """Origin of a training sample."""

    SIMULATION = "SIMULATION"
    AUTO_INDEXED = "AUTO_INDEXED"
    USER_CONFIRMED = "USER_CONFIRMED"


# ---------------------------------------------------------------------------
# Detector metadata
# ---------------------------------------------------------------------------


@dataclass
class DetectorInfo:
    """Detector metadata for a single EBSD measurement.

    Parameters
    ----------
    manufacturer : DetectorManufacturer
        Detector hardware manufacturer.
    pc : tuple[float, float, float]
        Pattern center (PCx, PCy, PCz) in the *original* convention.
    pc_convention : DetectorConvention
        Which convention ``pc`` is expressed in.
    kv : float
        Accelerating voltage in kV (typically 5-30).
    working_distance : float
        Working distance in mm (typically 5-25).
    sample_tilt : float
        Sample tilt in degrees (typically 60-70).
    """

    manufacturer: DetectorManufacturer = DetectorManufacturer.OTHER
    pc: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pc_convention: DetectorConvention = DetectorConvention.KIKUCHIPY
    kv: float = 20.0
    working_distance: float = 15.0
    sample_tilt: float = 70.0

    # -- Normalization ranges for encoding --------------------------------

    KV_RANGE: tuple[float, float] = (5.0, 30.0)
    WD_RANGE: tuple[float, float] = (5.0, 25.0)
    TILT_RANGE: tuple[float, float] = (50.0, 80.0)

    def __post_init__(self) -> None:
        if not isinstance(self.pc, tuple) or len(self.pc) != 3:
            raise ValueError(
                f"DetectorInfo.pc must be a 3-tuple (PCx, PCy, PCz), "
                f"got {self.pc!r}"
            )
        if not isinstance(self.manufacturer, DetectorManufacturer):
            raise TypeError(
                f"DetectorInfo.manufacturer must be a DetectorManufacturer "
                f"enum, got {type(self.manufacturer).__name__}"
            )
        if not isinstance(self.pc_convention, DetectorConvention):
            raise TypeError(
                f"DetectorInfo.pc_convention must be a DetectorConvention "
                f"enum, got {type(self.pc_convention).__name__}"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict (for HDF5 attribute storage)."""
        return {
            "manufacturer": self.manufacturer.value,
            "pc": list(self.pc),
            "pc_convention": self.pc_convention.value,
            "kv": self.kv,
            "working_distance": self.working_distance,
            "sample_tilt": self.sample_tilt,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> DetectorInfo:
        """Reconstruct from a plain dict."""
        return cls(
            manufacturer=DetectorManufacturer(d["manufacturer"]),
            pc=tuple(d["pc"]),  # type: ignore[arg-type]
            pc_convention=DetectorConvention(d["pc_convention"]),
            kv=float(d["kv"]),  # type: ignore[arg-type]
            working_distance=float(d["working_distance"]),  # type: ignore[arg-type]
            sample_tilt=float(d["sample_tilt"]),  # type: ignore[arg-type]
        )

    def encode(self) -> np.ndarray:
        """Encode detector info as a fixed-length float32 feature vector.

        Layout (length = ``encoded_length()``):
            [PCx, PCy, PCz,
             convention_one_hot (5),
             manufacturer_one_hot (4),
             kv_norm, wd_norm, tilt_norm]

        Total: 3 + 5 + 4 + 3 = 15 floats.
        """
        vec = np.zeros(self.encoded_length(), dtype=np.float32)

        # Raw PC values (in original convention), sanitize non-finite
        pc = np.array(self.pc, dtype=np.float32)
        pc = np.where(np.isfinite(pc), pc, 0.0)
        vec[0:3] = pc

        # Convention one-hot (5 categories)
        conv_idx = list(DetectorConvention).index(self.pc_convention)
        vec[3 + conv_idx] = 1.0

        # Manufacturer one-hot (4 categories)
        mfr_idx = list(DetectorManufacturer).index(self.manufacturer)
        vec[8 + mfr_idx] = 1.0

        # Normalized continuous values → [0, 1]
        vec[12] = _normalize(self.kv, *self.KV_RANGE)
        vec[13] = _normalize(self.working_distance, *self.WD_RANGE)
        vec[14] = _normalize(self.sample_tilt, *self.TILT_RANGE)

        return vec

    @staticmethod
    def encoded_length() -> int:
        """Number of floats produced by ``encode()``."""
        return 15


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


@dataclass
class PhasePrediction:
    """Result of a single phase prediction.

    Parameters
    ----------
    top_k : list[tuple[str, float]]
        Phase names with associated probabilities, sorted descending.
    confidence : float
        Probability of the top-1 prediction.
    eds_contribution : float
        Attention weight indicating how much EDS influenced the result
        (0 = pattern only, 1 = EDS only).
    """

    top_k: list[tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    eds_contribution: float = 0.0


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Hyperparameters for the phase classification model.

    Parameters
    ----------
    pattern_feature_dim : int
        Output dimension of the pattern encoder.
    eds_feature_dim : int
        Output dimension of the EDS+metadata encoder.
    fused_feature_dim : int
        Dimension after fusion of pattern and EDS features.
    dropout : float
        Dropout probability used across encoders.
    n_phases : int
        Number of output classes (can grow with ``add_phase``).
    pattern_size : int
        Square side length that input patterns are resized to.
    """

    pattern_feature_dim: int = 256
    eds_feature_dim: int = 64
    fused_feature_dim: int = 128
    dropout: float = 0.3
    n_phases: int = 2
    pattern_size: int = DEFAULT_PATTERN_SIZE

    def __post_init__(self) -> None:
        for name in ("pattern_feature_dim", "eds_feature_dim",
                      "fused_feature_dim", "n_phases", "pattern_size"):
            val = getattr(self, name)
            if not isinstance(val, int) or val < 1:
                raise ValueError(
                    f"ModelConfig.{name} must be a positive integer, got {val!r}"
                )
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(
                f"ModelConfig.dropout must be in [0, 1], got {self.dropout}"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        return {
            "pattern_feature_dim": self.pattern_feature_dim,
            "eds_feature_dim": self.eds_feature_dim,
            "fused_feature_dim": self.fused_feature_dim,
            "dropout": self.dropout,
            "n_phases": self.n_phases,
            "pattern_size": self.pattern_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ModelConfig:
        """Reconstruct from a plain dict."""
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**cast(dict[str, Any], filtered))


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop.

    Parameters
    ----------
    learning_rate : float
        Peak learning rate for OneCycleLR scheduler.
    batch_size : int
        Mini-batch size.
    epochs : int
        Maximum number of training epochs.
    early_stopping_patience : int
        Stop after this many epochs without validation loss improvement.
    eds_dropout_rate : float
        Fraction of samples where EDS is randomly zeroed during training
        to force the model to work without EDS.
    ci_threshold : float
        Minimum confidence index for auto-labeling from indexing results.
    val_fraction : float
        Fraction of data reserved for validation.
    weight_decay : float
        L2 regularization strength.
    mixed_precision : bool
        Use automatic mixed precision on GPU.
    """

    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 10
    eds_dropout_rate: float = 0.3
    ci_threshold: float = 0.3
    val_fraction: float = 0.15
    weight_decay: float = 1e-4
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                f"TrainingConfig.learning_rate must be positive, "
                f"got {self.learning_rate}"
            )
        for name in ("batch_size", "epochs", "early_stopping_patience"):
            val = getattr(self, name)
            if not isinstance(val, int) or val < 1:
                raise ValueError(
                    f"TrainingConfig.{name} must be a positive integer, "
                    f"got {val!r}"
                )
        for name in ("eds_dropout_rate", "ci_threshold", "val_fraction"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"TrainingConfig.{name} must be in [0, 1], got {val}"
                )
        if self.weight_decay < 0:
            raise ValueError(
                f"TrainingConfig.weight_decay must be non-negative, "
                f"got {self.weight_decay}"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "eds_dropout_rate": self.eds_dropout_rate,
            "ci_threshold": self.ci_threshold,
            "val_fraction": self.val_fraction,
            "weight_decay": self.weight_decay,
            "mixed_precision": self.mixed_precision,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> TrainingConfig:
        """Reconstruct from a plain dict."""
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**cast(dict[str, Any], filtered))


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Parameters
    ----------
    pattern_size : int
        Square dimension to resize all patterns to.
    num_elements : int
        Number of elements in the EDS feature vector.
    augmentation_enabled : bool
        Whether to apply data augmentation during training.
    max_rotation_deg : float
        Maximum random rotation angle for pattern augmentation.
    noise_std : float
        Standard deviation of additive Gaussian noise for augmentation.
    brightness_range : float
        Maximum brightness jitter as fraction of mean intensity.
    """

    pattern_size: int = DEFAULT_PATTERN_SIZE
    num_elements: int = NUM_ELEMENTS
    augmentation_enabled: bool = True
    max_rotation_deg: float = 2.0
    noise_std: float = 0.05
    brightness_range: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.pattern_size, int) or self.pattern_size < 1:
            raise ValueError(
                f"DataConfig.pattern_size must be a positive integer, "
                f"got {self.pattern_size!r}"
            )
        if self.max_rotation_deg < 0:
            raise ValueError(
                f"DataConfig.max_rotation_deg must be non-negative, "
                f"got {self.max_rotation_deg}"
            )
        if self.noise_std < 0:
            raise ValueError(
                f"DataConfig.noise_std must be non-negative, "
                f"got {self.noise_std}"
            )
        if self.brightness_range < 0:
            raise ValueError(
                f"DataConfig.brightness_range must be non-negative, "
                f"got {self.brightness_range}"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        return {
            "pattern_size": self.pattern_size,
            "num_elements": self.num_elements,
            "augmentation_enabled": self.augmentation_enabled,
            "max_rotation_deg": self.max_rotation_deg,
            "noise_std": self.noise_std,
            "brightness_range": self.brightness_range,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> DataConfig:
        """Reconstruct from a plain dict."""
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**cast(dict[str, Any], filtered))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_array(
    arr: np.ndarray,
    *,
    replace: float = 0.0,
    name: str = "array",
) -> np.ndarray:
    """Replace NaN and Inf values in an array with *replace*.

    Parameters
    ----------
    arr : np.ndarray
        Input array (modified in-place if writable, otherwise a copy).
    replace : float
        Value to substitute for non-finite entries.
    name : str
        Descriptive label for log messages.

    Returns
    -------
    np.ndarray
        Array with all non-finite values replaced.
    """
    if not np.issubdtype(arr.dtype, np.number):
        return arr
    mask = ~np.isfinite(arr)
    if mask.any():
        import logging

        n_bad = int(mask.sum())
        logging.getLogger("ebsd_ai").warning(
            "Sanitized %d non-finite value(s) in %s", n_bad, name
        )
        arr = np.array(arr, copy=True)
        arr[mask] = replace
    return arr


# Default MAD-to-confidence decay constant (degrees).
# confidence = exp(-MAD / MAD_DECAY).
# With decay=1.0: MAD=0 → 1.0, MAD=1 → 0.37, MAD=3 → 0.05.
MAD_DECAY_DEFAULT: float = 1.0


def mad_to_confidence(
    mad: np.ndarray,
    *,
    decay: float = MAD_DECAY_DEFAULT,
) -> np.ndarray:
    """Convert MAD (Mean Angular Deviation) to a confidence score.

    Uses an exponential decay: ``confidence = exp(-MAD / decay)``.

    Parameters
    ----------
    mad : np.ndarray
        MAD values in degrees, shape ``(N,)``.  Negative values are
        clamped to zero.
    decay : float
        Decay constant in degrees.  Larger values produce higher
        confidence for the same MAD.  Default: 1.0.

    Returns
    -------
    np.ndarray
        Confidence scores in [0, 1], same shape as *mad*.
    """
    result: np.ndarray = np.exp(-np.clip(mad, 0, None) / decay)
    return result


def _normalize(value: float, lo: float, hi: float) -> float:
    """Normalize *value* to [0, 1] given range [lo, hi].

    Values outside the range are clamped.
    Non-finite inputs return 0.0.
    """
    if not np.isfinite(value):
        return 0.0
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def eds_dict_to_vector(
    eds_data: Optional[dict[str, float]],
) -> np.ndarray:
    """Convert an element-name → At.% dict to a fixed-length vector.

    Parameters
    ----------
    eds_data : dict[str, float] or None
        Mapping from element symbol (e.g. ``"Fe"``) to atomic percent.
        If *None*, returns an all-zero vector (no EDS available).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(NUM_ELEMENTS,)``.

    Notes
    -----
    Element symbols are matched case-insensitively after stripping any
    X-ray line suffix (e.g. ``"Fe Ka1"`` → ``"Fe"``).
    """
    vec = np.zeros(NUM_ELEMENTS, dtype=np.float32)
    if eds_data is None:
        return vec
    for raw_key, value in eds_data.items():
        symbol = raw_key.split()[0].capitalize()
        if symbol in ELEMENT_INDEX:
            v = float(value)
            if np.isfinite(v):
                vec[ELEMENT_INDEX[symbol]] = v
    return vec


def has_eds(eds_vector: np.ndarray) -> bool:
    """Return True if the EDS vector contains any nonzero values."""
    return bool(np.any(eds_vector != 0))


# ---------------------------------------------------------------------------
# Denoising configuration
# ---------------------------------------------------------------------------


@dataclass
class DenoisingTrainingConfig:
    """Training hyperparameters for pattern denoising.

    Model architecture parameters (depth, channels, FiLM) stay in
    ``EnhancerConfig``.  This dataclass covers only the training loop.

    Parameters
    ----------
    learning_rate : float
        Peak learning rate for OneCycleLR scheduler.
    batch_size : int
        Mini-batch size (U-Net needs more GPU RAM, so smaller than classification).
    epochs : int
        Maximum training epochs.
    early_stopping_patience : int
        Stop after this many epochs without validation loss improvement.
    val_fraction : float
        Fraction of data reserved for validation.
    l1_weight : float
        Weight for L1 (mean absolute error) loss component.
    ssim_weight : float
        Weight for (1 - SSIM) loss component.
    min_training_pairs : int
        Minimum number of denoising pairs before training can start.
    residual_learning : bool
        If True, network learns the noise residual instead of clean image.
    weight_decay : float
        L2 regularization strength.
    mixed_precision : bool
        Use automatic mixed precision on GPU.
    """

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    val_fraction: float = 0.15
    l1_weight: float = 1.0
    ssim_weight: float = 0.5
    min_training_pairs: int = 500
    residual_learning: bool = True
    weight_decay: float = 1e-4
    mixed_precision: bool = True

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "DenoisingTrainingConfig":
        """Reconstruct from a plain dict, ignoring unknown keys."""
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**cast(dict[str, Any], filtered))


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding encoder and contrastive training.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the L2-normalized embedding vector.
    encoder_type : str
        Encoder backbone type (currently only "resnet18").
    temperature : float
        Temperature for InfoNCE contrastive loss.
    pos_angle_deg : float
        Maximum misorientation angle (degrees) for positive pairs.
    neg_angle_deg : float
        Minimum misorientation angle (degrees) for hard negatives.
    learning_rate : float
        Peak learning rate for AdamW.
    weight_decay : float
        L2 regularization.
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs.
    pc_augment_sigma : float
        Standard deviation for pattern center perturbation augmentation.
    pattern_size : int
        Square side length for input patterns.
    mixed_precision : bool
        Use AMP on GPU.
    """

    embedding_dim: int = 128
    encoder_type: str = "resnet18"
    temperature: float = 0.07
    pos_angle_deg: float = 2.0
    neg_angle_deg: float = 10.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 150
    pc_augment_sigma: float = 0.015
    pattern_size: int = 128
    mixed_precision: bool = True

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "EmbeddingConfig":
        """Reconstruct from a plain dict, ignoring unknown keys."""
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**cast(dict[str, Any], filtered))
