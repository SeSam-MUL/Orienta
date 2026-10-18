"""
ML Hub API Routes

Wraps ml_controller and Ai_Ml/ebsd_ai for:
- Model management (load, train, predict)
- Pattern classification
- Training data management
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from collections import OrderedDict
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

EXPERIMENTAL_ML_ENV = "ORIENTA_EXPERIMENTAL_ML"


def require_experimental_ml() -> None:
    """Router-level gate: the ML Hub is an internal, experimental page.

    Unless the backend was started with ``ORIENTA_EXPERIMENTAL_ML=1`` every
    ``/api/ml`` route answers 404 -- except ``/status``, which always answers
    200 so the browser does not log a console error on every start: it says
    ``{"enabled": false}`` when the switch is off (see ``ml_status``).
    The variable is read on every request (not at import) so a test can flip it.
    """
    if not _experimental_ml_enabled():
        raise HTTPException(status_code=404, detail="Not Found")


def _experimental_ml_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_ML_ENV, "").strip() == "1"


# ``router`` is what main.py mounts. ``/status`` hangs on it directly (ungated);
# every other route hangs on ``_gated``, which carries the 404 gate and is
# included at the bottom of this file. Add new routes to ``_gated``.
router = APIRouter()
_gated = APIRouter(dependencies=[Depends(require_experimental_ml)])

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


class LoadModelRequest(BaseModel):
    run_id: str


class TrainRequest(BaseModel):
    training_data_path: str = "__last_indexing__"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    ci_threshold: float = 0.3
    # The store keeps every sample ever added. By default a run trains only on
    # the samples THIS request adds (from the last indexing result); set this
    # to also train on whatever was already in the store (other results, other
    # files, other phases).
    include_existing_store: bool = False


@router.get("/status")
async def ml_status():
    """Is the ML Hub switched on, and (if so) module availability and loaded model.

    Always 200. Switch off: ``{"enabled": false}`` and nothing else. Switch on:
    the details below plus ``"enabled": true``.
    """
    if not _experimental_ml_enabled():
        return {"enabled": False}
    body = await _ml_status_details()
    body["enabled"] = True
    return body


async def _ml_status_details():
    try:
        from ml_controller import (
            is_ml_available, get_store_stats, get_predictor,
            ensure_predictor_loaded, get_loaded_run_id,
        )
        if not is_ml_available():
            return {"available": False, "reason": "ebsd_ai module not installed"}
        stats = get_store_stats()
        # After a backend restart the predictor is empty although trained
        # runs are on disk: load the newest one now. Off the event loop: the
        # first load imports torch and reads a checkpoint (seconds).
        await asyncio.to_thread(ensure_predictor_loaded)
        # model_loaded reflects the LIVE predictor, not what is on disk.
        predictor = get_predictor()
        model_loaded = bool(predictor is not None and getattr(predictor, 'has_model', False))
        return {
            "available": True,
            "model_loaded": model_loaded,
            "loaded_run_id": get_loaded_run_id() if model_loaded else None,
            "store_stats": stats,
            "phase_names": list(getattr(predictor, 'phase_names', [])) if model_loaded else [],
        }
    except ImportError:
        return {"available": False, "reason": "ml_controller not available"}


@_gated.get("/models")
async def list_models():
    """List the stored training runs, newest first (``legacy`` last)."""
    try:
        from ml_controller import list_runs
        return {"models": list_runs()}
    except Exception as e:
        return {"models": [], "error": str(e)}


@_gated.post("/models/load")
async def load_model(req: LoadModelRequest):
    """Make a stored training run the live predictor."""
    from ml_controller import load_run
    try:
        phase_names = await asyncio.to_thread(load_run, req.run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Unknown model run: {req.run_id}")
    except Exception as e:
        logger.exception("ML model load failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"loaded_run_id": req.run_id, "phase_names": phase_names}


@_gated.post("/predict")
async def predict(req: PredictRequest):
    """Run prediction on the currently loaded EBSD signal's patterns.

    pattern_indices are flat pixel indices into the loaded signal.
    """
    try:
        import numpy as np
        from ml_controller import predict_single_pattern, ensure_predictor_loaded
        from backend.api.routes.ebsd_viewer import _get_active_signal

        signal = _get_active_signal()
        if signal is None:
            raise HTTPException(status_code=400, detail="No EBSD data loaded — open a file first")

        # BUG: earlier code passed the flat int index to
        # predict_single_pattern, but that function expects a 2D numpy array
        # (the actual pattern). It silently returned None for every index
        # because _predictor was None. With a trained model it would
        # crash. Resolve the index → pattern here.
        await asyncio.to_thread(ensure_predictor_loaded)  # after a restart: newest stored run
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


def _add_last_indexing_to_store(req: "TrainRequest") -> tuple:
    """Extract samples from the last indexing result and add them to the
    training store. Returns ``(n_before, added)``: the store size before this
    call and the number of samples added, so the caller knows that the new
    samples occupy store indices ``n_before .. n_before + added - 1``. Raises
    HTTPException on missing data so the caller can surface a clean message."""
    import numpy as np
    from ml_controller import add_indexing_results_to_store, _ensure_store
    from backend.api.routes.indexing import get_last_indexing_result
    from backend.api.routes.ebsd_viewer import _ebsd_file_path

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
    store = _ensure_store()
    n_before = len(store) if store is not None else 0
    added = add_indexing_results_to_store(
        patterns=patterns[:n],
        phase_ids=phase_ids[:n],
        phase_names=phase_names,
        orientations=orientations[:n],
        confidence_scores=ci[:n],
        ci_threshold=req.ci_threshold,
        source_file=Path(str(_ebsd_file_path)).name if _ebsd_file_path else "",
    )
    logger.info("Added %d samples to ML store from last indexing result (phases=%s, ci>=%.2f)",
                added, phase_names, req.ci_threshold)
    return n_before, added


@_gated.post("/train")
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
            source_file = ""
            sample_indices = None  # None = the whole store
            n_before, added = 0, 0
            if req.training_data_path == '__last_indexing__':
                n_before, added = _add_last_indexing_to_store(req)
                from backend.api.routes.ebsd_viewer import _ebsd_file_path
                source_file = Path(str(_ebsd_file_path)).name if _ebsd_file_path else ""
                task["log"].append(f"Added {added} samples from last indexing result")
                if not req.include_existing_store:
                    # Train on this request's samples only; the store may hold
                    # samples of other results, files and phases.
                    sample_indices = list(range(n_before, n_before + added))
                    if n_before:
                        task["log"].append(
                            f"Training on the {added} samples of this request; "
                            f"{n_before} older samples in the store are left out"
                        )
                elif n_before:
                    task["log"].append(
                        f"Training on {added} samples of this request plus "
                        f"{n_before} older samples from the store"
                    )
            else:
                from ml_controller import _ensure_store
                _store = _ensure_store()
                n_before = len(_store) if _store is not None else 0
            sample_sources = {
                "this_request": {"n_samples": int(added), "source_file": source_file},
                "existing_store": {
                    "available": int(n_before),
                    "included": sample_indices is None,
                    "n_samples": int(n_before) if sample_indices is None else 0,
                },
            }

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
                source_file=source_file, sample_indices=sample_indices,
                sample_sources=sample_sources,
            )
            task["status"] = "completed"
            task["progress"] = 1.0
            task["result"] = {
                "epochs_completed": res.get("epochs_completed"),
                "best_val_loss": res.get("best_val_loss"),
                "phase_names": list(res.get("phase_names", []) or []),
                "checkpoint_path": str(res.get("checkpoint_path", "")),
                "run_id": res.get("run_id"),
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


@_gated.get("/train/{task_id}")
async def train_status(task_id: str):
    """Poll a background training task."""
    task = _train_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Unknown training task id")
    # Don't serialise the asyncio future.
    return {k: v for k, v in task.items() if k != "_future"}


@_gated.post("/store/clear")
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


# Last, after every ``@_gated`` route above is registered.
router.include_router(_gated)
