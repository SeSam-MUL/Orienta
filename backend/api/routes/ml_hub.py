"""
ML Hub API Routes

Wraps ml_controller and Ai_Ml/ebsd_ai for:
- Model management (load, train, predict)
- Pattern classification
- Training data management
"""

import asyncio
import logging
import uuid
from collections import OrderedDict
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Background training tasks: task_id -> {status, progress, epoch, total, log,
# result, error}. Capped so a long session doesn't grow unbounded.
_train_tasks: "OrderedDict[str, dict]" = OrderedDict()
_MAX_TRAIN_TASKS = 16


def _extract_phase_labels(xmap):
    """Return (contiguous phase_ids int32 array, phase_names aligned to 0..K-1).

    Maps each pixel's raw phase_id through the xmap's PhaseList to a real
    crystallographic name, remapping ids to a contiguous 0..K-1 range so the
    training store's ``phase_names[phase_id]`` lookup is correct. This is what
    stops every sample collapsing to "Phase_0": the previous code passed a
    bare ``["Phase_0"]`` fallback and indexed it with raw (possibly negative
    or non-contiguous) phase ids.
    """
    import numpy as np
    raw = np.asarray(xmap.phase_id).ravel().astype(int)
    id_to_name: dict[int, str] = {}
    try:
        pl = xmap.phases
        for pid in list(pl.ids):
            try:
                nm = str(pl[pid].name).strip()
            except Exception:
                nm = ""
            id_to_name[int(pid)] = nm or f"phase_{int(pid)}"
    except Exception:
        logger.warning("Could not read PhaseList names from xmap; using phase_<id> labels")
    unique_ids = sorted({int(i) for i in raw.tolist()})
    remap = {pid: k for k, pid in enumerate(unique_ids)}

    def _name(pid: int) -> str:
        if pid < 0:
            return "not_indexed"
        return id_to_name.get(pid, f"phase_{pid}")

    names = [_name(pid) for pid in unique_ids]
    ids = np.array([remap[int(i)] for i in raw.tolist()], dtype=np.int32)
    return ids, names


class PredictRequest(BaseModel):
    pattern_indices: List[int]


class TrainRequest(BaseModel):
    training_data_path: str = "__last_indexing__"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    ci_threshold: float = 0.3


@router.get("/status")
async def ml_status():
    """Check ML module availability and loaded model."""
    try:
        from ml_controller import is_ml_available, get_store_stats, list_profiles, get_predictor
        if not is_ml_available():
            return {"available": False, "reason": "ebsd_ai module not installed"}
        stats = get_store_stats()
        profiles = list_profiles()
        # model_loaded must reflect the LIVE predictor state, not whether
        # a profile JSON exists on disk. Previous code returned
        # `model_loaded = len(profiles) > 0` — true when profiles existed
        # but no model was loaded into _predictor (so /predict would
        # return None) AND false when a model was just trained but no
        # profile saved yet (training only writes a checkpoint).
        predictor = get_predictor()
        model_loaded = bool(predictor is not None and getattr(predictor, 'has_model', False))
        return {
            "available": True,
            "model_loaded": model_loaded,
            "store_stats": stats,
            "profiles": profiles,
            "phase_names": list(getattr(predictor, 'phase_names', [])) if model_loaded else [],
        }
    except ImportError:
        return {"available": False, "reason": "ml_controller not available"}


@router.get("/models")
async def list_models():
    """List available ML models/profiles."""
    try:
        from ml_controller import list_profiles
        profiles = list_profiles()
        return {"models": profiles}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.post("/predict")
async def predict(req: PredictRequest):
    """Run prediction on the currently loaded EBSD signal's patterns.

    pattern_indices are flat pixel indices into the loaded signal.
    """
    try:
        import numpy as np
        from ml_controller import predict_single_pattern
        from backend.api.routes.ebsd_viewer import _get_active_signal

        signal = _get_active_signal()
        if signal is None:
            raise HTTPException(status_code=400, detail="No EBSD data loaded — open a file first")

        # BUG: earlier code passed the flat int index to
        # predict_single_pattern, but that function expects a 2D numpy array
        # (the actual pattern). It silently returned None for every index
        # because _predictor was None. With a trained model it would
        # crash. Resolve the index → pattern here.
        nav_shape = signal.data.shape[:2]
        all_patterns = signal.data.reshape(-1, *signal.data.shape[-2:])
        n_total = all_patterns.shape[0]

        results = []
        for idx in req.pattern_indices:
            if not (0 <= idx < n_total):
                results.append({"index": idx, "error": f"index out of range (0..{n_total - 1})"})
                continue
            pat = np.asarray(all_patterns[idx])
            pred = predict_single_pattern(pat)
            if pred is None:
                results.append({"index": idx, "error": "No trained model loaded — train first or load a checkpoint"})
            else:
                # PhasePrediction (Ai_Ml/ebsd_ai/config.py) has top_k:
                # list[tuple[name, prob]], confidence: float, eds_contribution.
                # There is NO `.phase_name` attribute — earlier code returned
                # phase_name=None always. Derive it from top_k[0].
                top_k = list(getattr(pred, 'top_k', []) or [])
                phase_name = top_k[0][0] if top_k else None
                results.append({
                    "index": idx,
                    "phase_name": phase_name,
                    "confidence": float(getattr(pred, 'confidence', 0.0)),
                    "eds_contribution": float(getattr(pred, 'eds_contribution', 0.0)),
                    "top_k": [{"name": str(n), "prob": float(p)} for n, p in top_k],
                })
        return {"predictions": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ML predict failed")
        raise HTTPException(status_code=500, detail=str(e))


def _add_last_indexing_to_store(req: "TrainRequest") -> int:
    """Extract samples from the last indexing result and add them to the
    training store. Returns the number of samples added. Raises HTTPException
    on missing data so the caller can surface a clean message."""
    import numpy as np
    from ml_controller import add_indexing_results_to_store
    from backend.api.routes.indexing import get_last_indexing_result

    result = get_last_indexing_result()
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available — run indexing first")

    # Dictionary indexing stashes the signal in metadata; Hough/Spherical
    # don't, so fall back to the active viewer signal.
    signal = result.metadata.get('signal') if result.metadata else None
    if signal is None:
        from backend.api.routes.ebsd_viewer import _get_active_signal
        signal = _get_active_signal()
    if signal is None:
        raise HTTPException(
            status_code=400,
            detail="No EBSD signal available (neither in indexing result nor loaded in viewer). "
            "Load an EBSD file first, then re-run indexing.",
        )

    xmap = result.xmap
    mask = result.selection_mask
    flat_mask = mask.ravel() if mask is not None else None
    all_patterns = signal.data.reshape(-1, *signal.data.shape[-2:])
    if flat_mask is not None and flat_mask.size == all_patterns.shape[0]:
        patterns = all_patterns[flat_mask]
    else:
        patterns = all_patterns
    patterns = np.ascontiguousarray(patterns)

    # Real, correctly-mapped phase labels (fixes the "everything is Phase_0" bug).
    phase_ids, phase_names = _extract_phase_labels(xmap)
    if flat_mask is not None and flat_mask.size == phase_ids.size:
        phase_ids = phase_ids[flat_mask]

    rot_data = np.asarray(xmap.rotations.data)
    if rot_data.ndim == 3:
        orientations = rot_data[:, 0, :]
    elif rot_data.ndim == 2 and rot_data.shape[-1] == 4:
        orientations = rot_data
    else:
        orientations = rot_data.reshape(-1, 4)

    ci = (np.asarray(result.confidence_scores).ravel()
          if result.confidence_scores is not None else np.zeros(patterns.shape[0]))

    n = min(len(patterns), len(phase_ids), len(orientations), len(ci))
    added = add_indexing_results_to_store(
        patterns=patterns[:n],
        phase_ids=phase_ids[:n],
        phase_names=phase_names,
        orientations=orientations[:n],
        confidence_scores=ci[:n],
        ci_threshold=req.ci_threshold,
    )
    logger.info("Added %d samples to ML store from last indexing result (phases=%s, ci>=%.2f)",
                added, phase_names, req.ci_threshold)
    return added


@router.post("/train")
async def train(req: TrainRequest):
    """Start model training in the background and return a task id to poll.

    Training is a long, CPU/GPU-bound torch loop (minutes to hours); it MUST
    run off the event loop or the whole backend (every other request) freezes
    for the entire run. We run it via asyncio.to_thread and report progress
    through GET /api/ml/train/{task_id}.
    """
    from ml_controller import train_model

    task_id = uuid.uuid4().hex[:12]
    _train_tasks[task_id] = {
        "status": "running", "progress": 0.0, "epoch": 0,
        "total": req.epochs, "log": [], "result": None, "error": None,
    }
    while len(_train_tasks) > _MAX_TRAIN_TASKS:
        _train_tasks.popitem(last=False)
    task = _train_tasks[task_id]

    def _work():
        try:
            if req.training_data_path == '__last_indexing__':
                added = _add_last_indexing_to_store(req)
                task["log"].append(f"Added {added} samples from last indexing result")

            def _cb(epoch, total, train_loss, val_loss):
                task["epoch"] = int(epoch)
                task["total"] = int(total)
                task["progress"] = epoch / max(total, 1)
                task["log"].append(
                    f"Epoch {epoch}/{total} — train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )

            res = train_model(
                epochs=req.epochs, batch_size=req.batch_size,
                learning_rate=req.learning_rate, progress_callback=_cb,
            )
            task["status"] = "completed"
            task["progress"] = 1.0
            task["result"] = {
                "epochs_completed": res.get("epochs_completed"),
                "best_val_loss": res.get("best_val_loss"),
                "phase_names": list(res.get("phase_names", []) or []),
                "checkpoint_path": str(res.get("checkpoint_path", "")),
                "elapsed_seconds": res.get("elapsed_seconds"),
            }
            task["log"].append("Training complete.")
        except HTTPException as he:
            task["status"] = "failed"
            task["error"] = he.detail
            task["log"].append(f"ERROR: {he.detail}")
        except Exception as e:
            logger.exception("ML train failed")
            task["status"] = "failed"
            task["error"] = str(e)
            task["log"].append(f"ERROR: {e}")

    # Fire-and-forget on a worker thread; keep a ref so it isn't GC'd.
    task["_future"] = asyncio.ensure_future(asyncio.to_thread(_work))
    return {"task_id": task_id, "status": "running"}


@router.get("/train/{task_id}")
async def train_status(task_id: str):
    """Poll a background training task."""
    task = _train_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Unknown training task id")
    # Don't serialise the asyncio future.
    return {k: v for k, v in task.items() if k != "_future"}


@router.post("/store/clear")
async def clear_store():
    """Delete all samples from the training store.

    The store accumulates samples across every 'Add from Indexing' call and
    persists on disk; stale/mislabelled samples (e.g. a run that landed
    everything under 'Phase_0') otherwise poison every future model and make
    training pathologically slow. This gives the user a clean reset.
    """
    try:
        from ml_controller import _ensure_store
        store = _ensure_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Training store unavailable")
        before = 0
        try:
            before = store.get_dataset_stats().get("total_samples", 0)
        except Exception:
            pass
        if not hasattr(store, "clear"):
            raise HTTPException(status_code=501, detail="Training store does not support clear()")
        store.clear()
        return {"cleared": True, "samples_removed": before}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ML store clear failed")
        raise HTTPException(status_code=500, detail=str(e))
