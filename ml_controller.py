"""
ML Controller — Bridge between GUI and ebsd_ai ML subsystem.

Provides functions for:
- Loading/saving ML models
- Adding indexing results to the training store
- Getting training store statistics
- Running predictions on EBSD scans

Handles the case where ebsd_ai is not installed gracefully.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Global references to avoid re-creating on every call
_predictor = None
_store = None

# Default training store location (relative to project)
_STORE_DIR = Path(__file__).parent / "ml_training_data"
_PROFILE_DIR = Path(__file__).parent / "ml_profiles"


def _ensure_store():
    """Lazily initialize the TrainingStore."""
    global _store
    if _store is not None:
        return _store
    try:
        import sys
        ai_ml_path = str(Path(__file__).parent / "Ai_Ml")
        if ai_ml_path not in sys.path:
            sys.path.insert(0, ai_ml_path)

        from ebsd_ai import TrainingStore

        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        _store = TrainingStore(str(_STORE_DIR))
        logger.info(f"Training store initialized at {_STORE_DIR}")
        return _store
    except ImportError:
        logger.warning("ebsd_ai package not available. ML features disabled.")
        return None


def is_ml_available() -> bool:
    """Check if the ebsd_ai package is importable."""
    try:
        import sys
        ai_ml_path = str(Path(__file__).parent / "Ai_Ml")
        if ai_ml_path not in sys.path:
            sys.path.insert(0, ai_ml_path)
        import ebsd_ai  # noqa: F401
        return True
    except ImportError:
        return False


def load_ml_model(model_path: str) -> List[str]:
    """Load a trained ML model from a checkpoint file.

    Parameters
    ----------
    model_path : str
        Path to .pt checkpoint file.

    Returns
    -------
    list of str
        Phase names known to the model.

    Raises
    ------
    ImportError
        If ebsd_ai is not installed.
    FileNotFoundError
        If the checkpoint file doesn't exist.
    """
    global _predictor

    import sys
    ai_ml_path = str(Path(__file__).parent / "Ai_Ml")
    if ai_ml_path not in sys.path:
        sys.path.insert(0, ai_ml_path)

    from ebsd_ai import PhasePredictor

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    _predictor = PhasePredictor(model_path=model_path)
    logger.info(f"ML model loaded: {_predictor.phase_names}")
    return list(_predictor.phase_names)


def save_ml_model(save_path: str) -> None:
    """Save the current model to a checkpoint file.

    Raises
    ------
    RuntimeError
        If no model is loaded.
    """
    global _predictor
    if _predictor is None or not _predictor.has_model:
        raise RuntimeError("No model loaded to save.")

    import torch
    save_dict = _predictor._model.get_save_dict()
    torch.save(save_dict, save_path)
    logger.info(f"ML model saved to {save_path}")


def get_predictor():
    """Return the global PhasePredictor instance (or None)."""
    return _predictor


def get_store_stats() -> Dict:
    """Get statistics from the training store.

    Returns
    -------
    dict
        Statistics including total_samples, samples_per_phase, etc.
        Returns empty dict if store is not available.
    """
    store = _ensure_store()
    if store is None:
        return {}
    try:
        return store.get_dataset_stats()
    except Exception as e:
        logger.error(f"Error getting store stats: {e}")
        return {"total_samples": 0}


def add_indexing_results_to_store(
    patterns: np.ndarray,
    phase_ids: np.ndarray,
    phase_names: List[str],
    orientations: np.ndarray,
    confidence_scores: np.ndarray,
    ci_threshold: float = 0.3,
    detector_info=None,
    eds_data: Optional[Dict[str, np.ndarray]] = None,
    source_file: str = "",
) -> int:
    """Add indexing results to the training store.

    Filters by CI threshold — only patterns with confidence >= threshold
    are stored.

    Parameters
    ----------
    patterns : np.ndarray
        Shape (N, H, W) — all indexed patterns.
    phase_ids : np.ndarray
        Shape (N,) — phase index per pixel.
    phase_names : list of str
        Phase name lookup table.
    orientations : np.ndarray
        Shape (N, 4) — quaternion per pixel.
    confidence_scores : np.ndarray
        Shape (N,) — CI per pixel.
    ci_threshold : float
        Minimum CI to accept.
    detector_info : DetectorInfo, optional
        Detector metadata. Creates default if None.
    eds_data : dict, optional
        Element -> array mapping.
    source_file : str
        Source file path for provenance.

    Returns
    -------
    int
        Number of samples added (those that passed threshold).

    Raises
    ------
    ImportError
        If ebsd_ai is not installed.
    """
    store = _ensure_store()
    if store is None:
        raise ImportError(
            "ebsd_ai package not available. Install it from Ai_Ml/ directory."
        )

    import sys
    ai_ml_path = str(Path(__file__).parent / "Ai_Ml")
    if ai_ml_path not in sys.path:
        sys.path.insert(0, ai_ml_path)

    from ebsd_ai import DetectorInfo as DI

    if detector_info is None:
        detector_info = DI()

    count = store.add_from_indexing(
        patterns=patterns,
        phase_ids=phase_ids,
        phase_names=phase_names,
        orientations=orientations,
        confidence_scores=confidence_scores,
        detector_info=detector_info,
        ci_threshold=ci_threshold,
        eds_data=eds_data,
        source_file=source_file,
    )
    logger.info(
        f"Added {count} samples to training store "
        f"(threshold={ci_threshold:.2f})"
    )
    return count


def train_model(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    progress_callback=None,
) -> dict:
    """Train a PhaseClassifier on the current training store.

    Parameters
    ----------
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Peak learning rate for OneCycleLR.
    progress_callback : callable, optional
        Called as (epoch, total_epochs, train_loss, val_loss) -> None.

    Returns
    -------
    dict
        Training result with keys: epochs_completed, best_epoch,
        best_val_loss, early_stopped, elapsed_seconds, phase_names,
        checkpoint_path.

    Raises
    ------
    ImportError
        If torch or ebsd_ai is not available.
    ValueError
        If training store is empty or has < 2 phases.
    RuntimeError
        If training fails for any reason.
    """
    global _predictor

    # Step 1: Verify dependencies with specific error messages
    import sys
    ai_ml_path = str(Path(__file__).parent / "Ai_Ml")
    if ai_ml_path not in sys.path:
        sys.path.insert(0, ai_ml_path)

    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is not installed.\n"
            "Install with: pip install torch torchvision\n"
            "Or with CUDA: pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121"
        )

    try:
        from ebsd_ai import (
            TrainingStore, EBSDPhaseDataset, PhaseClassifier,
            Trainer, TrainingConfig, ModelConfig, PhasePredictor,
        )
    except ImportError as e:
        raise ImportError(
            f"ebsd_ai package not available: {e}\n"
            f"The package should be in: {ai_ml_path}\n"
            f"Check that Ai_Ml/ebsd_ai/__init__.py exists."
        )

    # Step 2: Verify training store has data
    store = _ensure_store()
    if store is None:
        raise RuntimeError(
            "Training store could not be initialized.\n"
            f"Expected at: {_STORE_DIR}"
        )

    stats = store.get_dataset_stats()
    total = stats.get("total_samples", 0)
    if total == 0:
        raise ValueError(
            "Training store is empty. Add indexing results first "
            "using 'Add from Indexing Results'."
        )
    if total < 4:
        raise ValueError(
            f"Need at least 4 samples to train, store has {total}. "
            "Index more pixels (or lower the CI threshold) and add them first."
        )

    # BatchNorm layers in the model require batch size >= 2, and the batch
    # can't exceed the dataset. Clamp loudly so a user-entered batch=1 (or a
    # batch larger than the store) doesn't crash deep in the training loop.
    _bs = max(2, min(int(batch_size), total))
    if _bs != batch_size:
        logger.warning("Clamped batch_size %s -> %s (1 < bs <= n_samples=%d)",
                       batch_size, _bs, total)
        batch_size = _bs

    phase_names = store.get_phase_names()
    n_phases = len(phase_names)
    logger.info(f"Training on {total} samples, {n_phases} phase(s): {phase_names}")

    if n_phases < 1:
        raise ValueError(
            f"Need at least 1 phase to train, found {n_phases}.\n"
            f"Store stats: {stats}"
        )

    # Step 3: Create dataset
    dataset = EBSDPhaseDataset(store=store, phase_names=phase_names)
    logger.info(f"Dataset created: {len(dataset)} samples, {dataset.num_phases} phases")

    # Step 4: Create model
    model_config = ModelConfig(n_phases=max(n_phases, 2))
    model = PhaseClassifier(
        config=model_config,
        phase_names=phase_names if n_phases >= 2 else phase_names + ["_placeholder"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    # Step 5: Create trainer
    checkpoint_dir = _STORE_DIR / "checkpoints"
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        config=training_config,
        output_dir=str(checkpoint_dir),
        device=device,
        progress_callback=progress_callback,
    )

    # Step 6: Train
    result = trainer.train(dataset)

    # Step 7: Load best checkpoint as the active predictor
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        _predictor = PhasePredictor(model_path=str(best_path))
        logger.info(f"Best model loaded from {best_path}")

    return {
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "early_stopped": result.early_stopped,
        "elapsed_seconds": result.elapsed_seconds,
        "phase_names": phase_names,
        "checkpoint_path": str(best_path),
    }


def extract_denoising_pairs(
    experimental_patterns: np.ndarray,
    simulated_patterns: np.ndarray,
    confidence_scores: np.ndarray,
    phase_ids: np.ndarray,
    phase_names: List[str],
    ci_threshold: float = 0.3,
) -> List[Dict]:
    """Extract denoising training pairs from indexing results.

    Filters by confidence threshold and returns a list of dicts, each
    containing the experimental pattern, best-match simulated pattern,
    confidence, and phase name.

    Parameters
    ----------
    experimental_patterns : np.ndarray
        (N, H, W) experimental patterns.
    simulated_patterns : np.ndarray
        (N, H, W) best-match simulated patterns from dictionary indexing.
    confidence_scores : np.ndarray
        (N,) confidence index per pattern.
    phase_ids : np.ndarray
        (N,) phase index per pattern.
    phase_names : list of str
        Phase name lookup table.
    ci_threshold : float
        Minimum confidence to include a pair.

    Returns
    -------
    list of dict
        Each dict has keys: experimental, simulated, confidence, phase.
    """
    mask = confidence_scores >= ci_threshold
    indices = np.where(mask)[0]

    pairs = []
    for idx in indices:
        pairs.append({
            "experimental": experimental_patterns[idx],
            "simulated": simulated_patterns[idx],
            "confidence": float(confidence_scores[idx]),
            "phase": phase_names[phase_ids[idx]],
        })
    return pairs


def add_denoising_pairs_to_store(
    pairs: List[Dict],
    store=None,
    detector_info=None,
    source_file: str = "",
) -> int:
    """Add extracted denoising pairs to the training store.

    Parameters
    ----------
    pairs : list of dict
        As returned by :func:`extract_denoising_pairs`.
    store : TrainingStore, optional
        Store to add to. Uses global store if None.
    detector_info : DetectorInfo, optional
        Detector metadata. Creates default if None.
    source_file : str
        Provenance path.

    Returns
    -------
    int
        Number of pairs added.
    """
    if store is None:
        store = _ensure_store()
    if store is None:
        raise ImportError(
            "ebsd_ai package not available. Install it from Ai_Ml/ directory."
        )

    import sys
    ai_ml_path = str(Path(__file__).parent / "Ai_Ml")
    if ai_ml_path not in sys.path:
        sys.path.insert(0, ai_ml_path)

    from ebsd_ai import DetectorInfo as DI

    if detector_info is None:
        detector_info = DI()

    count = 0
    for pair in pairs:
        store.add_denoising_pair(
            experimental_pattern=pair["experimental"],
            simulated_pattern=pair["simulated"],
            confirmed_phase=pair["phase"],
            detector_info=detector_info,
            confidence_score=pair.get("confidence", 1.0),
            source_file=source_file,
        )
        count += 1

    logger.info(f"Added {count} denoising pairs to training store")
    return count


def predict_single_pattern(
    pattern: np.ndarray,
    eds_data: Optional[Dict[str, float]] = None,
    k: int = 3,
):
    """Predict phase for a single pattern.

    Returns
    -------
    PhasePrediction or None
        Prediction result, or None if no model loaded.
    """
    if _predictor is None or not _predictor.has_model:
        return None

    return _predictor.predict_phase(
        pattern=pattern,
        eds_data=eds_data,
        k=k,
    )


def predict_scan(
    patterns_4d: np.ndarray,
    eds_maps: Optional[Dict[str, np.ndarray]] = None,
    selection_mask: Optional[np.ndarray] = None,
    k: int = 3,
):
    """Predict phases for an entire scan.

    Returns
    -------
    ScanPredictionResult or None
        Prediction result, or None if no model loaded.
    """
    if _predictor is None or not _predictor.has_model:
        return None

    return _predictor.predict_scan(
        patterns_4d=patterns_4d,
        eds_maps=eds_maps,
        selection_mask=selection_mask,
        k=k,
    )


# ---------------------------------------------------------------------------
# Embedding-based indexing helpers
# ---------------------------------------------------------------------------

def build_faiss_index(
    encoder_checkpoint: str,
    patterns: np.ndarray,
    orientations: np.ndarray,
    phase_ids: np.ndarray,
    save_dir: Optional[str] = None,
    embedding_dim: int = 128,
):
    """Build a FAISS index from patterns using a trained encoder.

    Parameters
    ----------
    encoder_checkpoint : str
        Path to encoder .pt checkpoint.
    patterns : np.ndarray
        (N, H, W) reference patterns.
    orientations : np.ndarray
        (N, 4) quaternions.
    phase_ids : np.ndarray
        (N,) phase IDs.
    save_dir : str or None
        Directory to save the index.
    embedding_dim : int
        Embedding dimension.

    Returns
    -------
    EBSDFaissIndex or None
    """
    try:
        import torch
        from ebsd_ai.indexing.faiss_index import EBSDFaissIndex
        from ebsd_ai.models.embedding_encoder import EmbeddingEncoder

        save_dict = torch.load(
            encoder_checkpoint, map_location="cpu", weights_only=True,
        )
        model = EmbeddingEncoder.from_save_dict(save_dict)
        model.eval()

        # Encode in batches
        embeddings_list = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(patterns), batch_size):
                batch = torch.from_numpy(
                    patterns[i:i + batch_size],
                ).unsqueeze(1).float()
                emb = model(batch).numpy()
                embeddings_list.append(emb)
        embeddings = np.concatenate(embeddings_list, axis=0)

        index = EBSDFaissIndex(dim=embedding_dim)
        index.build(embeddings, orientations, phase_ids)

        if save_dir is not None:
            index.save(save_dir)
            logger.info("FAISS index saved to %s", save_dir)

        return index
    except Exception as exc:
        logger.error("Failed to build FAISS index: %s", exc)
        return None


def index_with_embedding(
    encoder_checkpoint: str,
    faiss_index_dir: str,
    patterns: np.ndarray,
    k: int = 5,
    batch_size: int = 64,
    embedding_dim: int = 128,
):
    """Index patterns using embedding + FAISS.

    Parameters
    ----------
    encoder_checkpoint : str
        Path to encoder .pt checkpoint.
    faiss_index_dir : str
        Path to saved FAISS index directory.
    patterns : np.ndarray
        (N, H, W) patterns to index.
    k : int
        Number of nearest neighbors.
    batch_size : int
        Batch size for encoding.
    embedding_dim : int
        Embedding dimension.

    Returns
    -------
    dict or None
        Result dict with orientations, phase_ids, confidence.
    """
    try:
        import torch
        from ebsd_ai.indexing.embedding_indexer import EmbeddingIndexer
        from ebsd_ai.indexing.faiss_index import EBSDFaissIndex
        from ebsd_ai.models.embedding_encoder import EmbeddingEncoder

        save_dict = torch.load(
            encoder_checkpoint, map_location="cpu", weights_only=True,
        )
        model = EmbeddingEncoder.from_save_dict(save_dict)

        faiss_index = EBSDFaissIndex.load(faiss_index_dir)

        indexer = EmbeddingIndexer(model=model, faiss_index=faiss_index)
        return indexer.index_scan(patterns, k=k, batch_size=batch_size)
    except Exception as exc:
        logger.error("Failed to index with embedding: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Model Profile CRUD
# ---------------------------------------------------------------------------

def set_profile_dir(directory: str) -> None:
    """Override the default profile storage directory."""
    global _PROFILE_DIR
    _PROFILE_DIR = Path(directory)


def list_profiles() -> List[str]:
    """List available profile names."""
    if not _PROFILE_DIR.exists():
        return []
    return [p.stem for p in _PROFILE_DIR.glob("*.json")]


def save_profile(profile) -> None:
    """Save a ModelProfile to disk."""
    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile.save(str(_PROFILE_DIR))


def load_profile(name: str):
    """Load a ModelProfile by name."""
    import json
    from ebsd_ai.models.model_profile import ModelProfile

    profile_path = _PROFILE_DIR / f"{name}.json"
    if not profile_path.exists():
        logger.error("Profile '%s' not found at %s", name, profile_path)
        return None
    with open(profile_path) as f:
        data = json.load(f)
    return ModelProfile.from_dict(data)


def delete_profile(name: str) -> bool:
    """Delete a profile by name."""
    profile_path = _PROFILE_DIR / f"{name}.json"
    if profile_path.exists():
        profile_path.unlink()
        logger.info("Deleted profile '%s'", name)
        return True
    return False
