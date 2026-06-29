"""FastAPI endpoints for GPU dictionary generation (stand-alone tool)."""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.dictionary_gpu.pipeline import generate_dictionary_gpu

logger = logging.getLogger(__name__)
router = APIRouter()

# Project root — used to anchor the `tasks/` directory regardless of the
# process CWD. Project convention: no hardcoded/cwd-relative paths.
#   backend/api/routes/dictionary_gpu.py -> parents[3] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASKS_DIR = PROJECT_ROOT / "tasks"


class GenerateRequest(BaseModel):
    master_path: str
    detector_shape: List[int] = Field(..., min_length=2, max_length=2)
    pc: List[float] = Field(..., min_length=3, max_length=3)
    sample_tilt: float = 70.0
    energy_kv: Optional[float] = None
    resolution_deg: float = 5.0
    output_path: Optional[str] = None
    normalize: bool = False


_tasks: dict = {}   # task_id -> {"status", "progress", "message", "result"}
_tasks_lock = threading.Lock()


def _set(task_id: str, **fields) -> None:
    with _tasks_lock:
        _tasks.setdefault(task_id, {})
        _tasks[task_id].update(fields)


def _run_job(task_id: str, payload: GenerateRequest) -> None:
    try:
        def cb(frac: float, msg: str) -> None:
            _set(task_id, progress=frac, message=msg)

        result = generate_dictionary_gpu(
            master_path=payload.master_path,
            detector_shape=tuple(payload.detector_shape),
            pc=tuple(payload.pc),
            sample_tilt=payload.sample_tilt,
            energy_kv=payload.energy_kv,
            resolution_deg=payload.resolution_deg,
            output_path=payload.output_path,
            normalize=payload.normalize,
            progress_callback=cb,
        )
        _set(
            task_id,
            status="done",
            progress=1.0,
            message="complete",
            result={
                "output_path": str(result.output_path),
                "n_orientations": result.metadata.n_orientations,
            },
        )
    except Exception as e:
        logger.exception("dictionary-gpu generation failed")
        _set(task_id, status="error", message=str(e))


@router.post("/generate")
def generate(payload: GenerateRequest):
    if not Path(payload.master_path).is_file():
        raise HTTPException(404, f"master pattern not found: {payload.master_path}")

    task_id = str(uuid.uuid4())
    _set(task_id, status="running", progress=0.0, message="queued")
    threading.Thread(target=_run_job, args=(task_id, payload), daemon=True).start()
    return {"task_id": task_id}


@router.get("/progress/{task_id}")
def progress(task_id: str):
    with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "unknown task_id")
        return dict(_tasks[task_id])


@router.get("/list")
def list_dictionaries():
    """List GPU-generated dictionaries (h5 + json sidecar pairs in tasks/).

    Enriches the raw sidecar JSON with derived display fields the React
    table expects: ``name`` (file stem), ``n_patterns`` (alias for
    ``n_orientations``) and ``size_mb`` (h5 file size).
    """
    import json
    out: List[dict] = []
    for json_path in TASKS_DIR.rglob("dict_gpu_*.json"):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            h5_path = Path(meta.get("dictionary_path", ""))
            if h5_path.is_file():
                meta["name"] = h5_path.stem
                meta["n_patterns"] = meta.get("n_orientations", 0)
                meta["size_mb"] = round(h5_path.stat().st_size / 1048576, 1)
                out.append(meta)
        except Exception as e:
            logger.warning("skipping %s: %s", json_path, e)
    return {"dictionaries": out}


@router.delete("/{name}")
def delete(name: str):
    """Delete a dictionary by stem name. Removes both .h5 and .json."""
    if "/" in name or "\\" in name or ".." in name or Path(name).name != name:
        raise HTTPException(400, "invalid dictionary name")
    h5 = TASKS_DIR / f"{name}.h5"
    js = TASKS_DIR / f"{name}.json"
    deleted = []
    for p in (h5, js):
        if p.is_file():
            p.unlink()
            deleted.append(str(p))
    if not deleted:
        raise HTTPException(404, "no matching dictionary")
    return {"deleted": deleted}
