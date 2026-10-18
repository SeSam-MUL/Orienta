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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Global references to avoid re-creating on every call
_predictor = None
# Run ids whose lazy load failed in this process; skipped by later lazy loads.
_unloadable_runs: set = set()
_store = None
# Run id (folder name under checkpoints/) of the model held by ``_predictor``;
# None when the predictor did not come from a stored run.
_loaded_run_id: Optional[str] = None

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
    global _predictor, _loaded_run_id

    import sys
    ai_ml_path = str(Path(__file__).parent / "Ai_Ml")
    if ai_ml_path not in sys.path:
        sys.path.insert(0, ai_ml_path)

    from ebsd_ai import PhasePredictor

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    _predictor = PhasePredictor(model_path=model_path)
    _loaded_run_id = None  # load_run() sets it when the file is a stored run
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


# ---------------------------------------------------------------------------
# Stored training runs: checkpoints/<run_id>/{best.pt,last.pt,meta.json}
# ---------------------------------------------------------------------------

LEGACY_RUN_ID = "legacy"


def _checkpoints_root() -> Path:
    return _STORE_DIR / "checkpoints"


def _new_run_id() -> str:
    """UTC timestamp, lexicographically sortable; suffix on a same-second clash."""
    from datetime import datetime, timezone

    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = _checkpoints_root()
    run_id, n = base, 1
    while (root / run_id).exists():
        n += 1
        run_id = f"{base}-{n}"
    return run_id


def _run_dir(run_id: str) -> Optional[Path]:
    """Folder of a stored run, or None. ``legacy`` is the old flat layout."""
    root = _checkpoints_root()
    if run_id == LEGACY_RUN_ID:
        return root if (root / "best.pt").is_file() else None
    # Only ids that appear in the listing are accepted (no path traversal).
    if run_id in {r["run_id"] for r in list_runs()}:
        return root / run_id
    return None


def list_runs() -> List[Dict]:
    """Stored training runs, newest first, each with its ``meta.json`` content.

    A folder counts as a run only if it holds a ``best.pt`` or ``last.pt``
    (a training that died before its first epoch leaves an empty folder).
    A flat ``checkpoints/best.pt`` from before run folders is listed last as
    run id ``legacy``.
    """
    import json

    root = _checkpoints_root()
    runs: List[Dict] = []
    if root.is_dir():
        for d in sorted((x for x in root.iterdir() if x.is_dir()),
                        key=lambda x: x.name, reverse=True):
            if not ((d / "best.pt").is_file() or (d / "last.pt").is_file()):
                continue
            meta: Dict = {}
            mp = d / "meta.json"
            if mp.is_file():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    logger.warning("Unreadable meta.json in %s", d)
            runs.append({"run_id": d.name, **meta})
        if (root / "best.pt").is_file():
            legacy = {"run_id": LEGACY_RUN_ID, "legacy": True}
            runs.append(legacy)
    for r in runs:
        r["loaded"] = r["run_id"] == _loaded_run_id and _predictor is not None
    return runs


def load_run(run_id: str) -> List[str]:
    """Load a stored run (its best checkpoint) as the live predictor.

    Raises
    ------
    FileNotFoundError
        If no such run exists.
    """
    global _loaded_run_id
    d = _run_dir(run_id)
    if d is None:
        raise FileNotFoundError(f"Unknown ML run: {run_id}")
    ckpt = d / "best.pt"
    if not ckpt.is_file():
        ckpt = d / "last.pt"
    names = load_ml_model(str(ckpt))
    _loaded_run_id = run_id
    return names


def get_loaded_run_id() -> Optional[str]:
    """Run id of the live predictor (None if it is not a stored run)."""
    return _loaded_run_id if _predictor is not None else None


def ensure_predictor_loaded() -> bool:
    """Make sure a model is live, lazily loading the newest stored run.

    Called when a request needs a model and the backend was restarted since
    training. Returns True when a model is live afterwards. Never raises: a
    broken newest run falls through to the next older one.
    """
    if _predictor is not None and getattr(_predictor, "has_model", False):
        return True
    if not is_ml_available():
        return False
    for run in list_runs():
        if run["run_id"] in _unloadable_runs:
            continue  # failed before; /status polls must not retry it every time
        try:
            load_run(run["run_id"])
            logger.info("ML model lazily loaded from run %s", run["run_id"])
            return True
        except Exception as e:  # noqa: BLE001 - try the next older run
            _unloadable_runs.add(run["run_id"])
            logger.warning("Could not load ML run %s: %s", run["run_id"], e)
    return False


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
    source_file: str = "",
    sample_indices: Optional[Sequence[int]] = None,
    sample_sources: Optional[Dict] = None,
) -> dict:
    """Train a PhaseClassifier on the training store (or a subset of it).

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
    source_file : str, optional
        File name of the EBSD data the samples came from (recorded in the
        run's ``meta.json``; informational only).
    sample_indices : sequence of int, optional
        Global store indices to train on. None means the whole store. The
        phase list of the model is taken from these samples only, so phases
        that exist elsewhere in the store do not leak into the run.
    sample_sources : dict, optional
        Where the samples came from (written verbatim to ``meta.json`` as
        ``sample_sources``; informational only).

    Returns
    -------
    dict
        Training result with keys: epochs_completed, best_epoch,
        best_val_loss, early_stopped, elapsed_seconds, phase_names,
        checkpoint_path, run_id. Checkpoints and a ``meta.json`` are written
        to ``checkpoints/<run_id>/``.

    Raises
    ------
    ImportError
        If torch or ebsd_ai is not available.
    ValueError
        If the training store (or the selected subset) is too small.
    RuntimeError
        If training fails for any reason.
    """
    global _predictor, _loaded_run_id

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
    if sample_indices is not None:
        sample_indices = [int(i) for i in sample_indices]
        total = len(sample_indices)
        what = "this request"
    else:
        total = stats.get("total_samples", 0)
        what = "store"
    if total == 0:
        raise ValueError(
            "No training samples. "
            + ("No pixel of this indexing result passed the CI threshold; "
               "lower it, or include the existing store."
               if sample_indices is not None else
               "Training store is empty. Add indexing results first "
               "using 'Add from Indexing Results'.")
        )
    if total < 4:
        raise ValueError(
            f"Need at least 4 samples to train, {what} has {total}. "
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

    if sample_indices is not None:
        # Phases of the selected samples only (store.get_phase_names() would
        # list every phase ever added, including those of other results).
        phase_names = sorted({
            str(store.get_metadata(i)["confirmed_phase"]) for i in sample_indices
        })
    else:
        phase_names = store.get_phase_names()
    n_phases = len(phase_names)
    logger.info(f"Training on {total} samples ({what}), {n_phases} phase(s): {phase_names}")

    if n_phases < 1:
        raise ValueError(
            f"Need at least 1 phase to train, found {n_phases}.\n"
            f"Store stats: {stats}"
        )

    # Step 3: Create dataset
    dataset = EBSDPhaseDataset(store=store, phase_names=phase_names, indices=sample_indices)
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
    run_id = _new_run_id()
    checkpoint_dir = _checkpoints_root() / run_id
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
    _write_run_meta(
        checkpoint_dir,
        phase_names=phase_names,
        source_file=source_file,
        n_samples=total,
        epochs_requested=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        result=result,
        sample_sources=sample_sources,
    )
    if best_path.exists():
        _predictor = PhasePredictor(model_path=str(best_path))
        _loaded_run_id = run_id
        logger.info(f"Best model loaded from {best_path}")

    return {
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "early_stopped": result.early_stopped,
        "elapsed_seconds": result.elapsed_seconds,
        "phase_names": phase_names,
        "checkpoint_path": str(best_path),
        "run_id": run_id,
    }


def _write_run_meta(run_dir: Path, *, phase_names, source_file, n_samples,
                    epochs_requested, learning_rate, batch_size, result,
                    sample_sources=None) -> None:
    """Write ``meta.json`` next to the checkpoints of one run.

    ``n_samples`` is the number of samples the run actually trained on;
    ``sample_sources`` says how many came from this request and how many
    from samples that were already in the store."""
    import json
    from datetime import datetime, timezone

    def _last(xs):
        return float(xs[-1]) if xs else None

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase_names": [str(n) for n in phase_names],
        "source_file": str(source_file or ""),
        "n_samples": int(n_samples),
        "sample_sources": sample_sources,
        "epochs_requested": int(epochs_requested),
        "epochs_completed": int(result.epochs_completed),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "final_train_loss": _last(result.train_losses),
        "final_val_loss": _last(result.val_losses),
        "best_val_loss": float(result.best_val_loss),
        "best_epoch": int(result.best_epoch),
        "early_stopped": bool(result.early_stopped),
        # The Trainer reports losses only; accuracy is not computed.
        "final_train_accuracy": None,
        "final_val_accuracy": None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
