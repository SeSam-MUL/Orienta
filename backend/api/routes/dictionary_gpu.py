"""FastAPI endpoints for GPU dictionary generation (stand-alone tool)."""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.dictionary_gpu.pipeline import (
    generate_dictionary_cpu,
    generate_dictionary_gpu,
)

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
    # Camera geometry of the loaded dataset. Defaults keep the old behaviour,
    # but the UI must send the real values — a dictionary simulated at
    # detector_tilt 0 for a 3.44 deg detector is unusable (NCC ~0.02).
    detector_tilt_deg: float = 0.0
    azimuthal_deg: float = 0.0
    energy_kv: Optional[float] = None
    resolution_deg: float = 5.0
    output_path: Optional[str] = None
    normalize: bool = False
    # When true (and no explicit output_path), write into
    # Database/Dictionary_Library under the canonical name so the Indexing
    # page's file discovery can find the result and attach it to its phase
    # card. Without this the dictionary lands in tasks/ where nothing looks.
    save_to_library: bool = False
    # "gpu" = in-process PyTorch projection; "cpu" = kikuchipy get_patterns.
    # Both write the same file layout, so the choice is purely about hardware.
    backend: str = "gpu"


def _dictionary_library_dir() -> Path:
    """Root of the local dictionary library. Indirection so tests can redirect."""
    from path_utils import get_local_database_path, DATABASE_SUBFOLDERS

    return get_local_database_path() / DATABASE_SUBFOLDERS["dictionary_library"]


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

        runner = (
            generate_dictionary_cpu
            if payload.backend == "cpu"
            else generate_dictionary_gpu
        )
        result = runner(
            master_path=payload.master_path,
            detector_shape=tuple(payload.detector_shape),
            pc=tuple(payload.pc),
            sample_tilt=payload.sample_tilt,
            detector_tilt_deg=payload.detector_tilt_deg,
            azimuthal_deg=payload.azimuthal_deg,
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

    if payload.save_to_library and not payload.output_path:
        # The energy is part of the filename, and a wrong energy in the name
        # mislabels the file for every later reader. Refuse rather than guess.
        if payload.energy_kv is None:
            raise HTTPException(
                400,
                "save_to_library needs energy_kv — it is part of the library "
                "filename and must not be guessed.",
            )
        from simulation.dictionary_generator import dictionary_library_paths

        h5_path, _ = dictionary_library_paths(
            _dictionary_library_dir(),
            master_path=payload.master_path,
            energy_kv=payload.energy_kv,
            detector_shape=tuple(payload.detector_shape),
            pc=tuple(payload.pc),
            resolution_deg=payload.resolution_deg,
        )
        payload.output_path = str(h5_path)

    task_id = str(uuid.uuid4())
    _set(task_id, status="running", progress=0.0, message="queued")
    threading.Thread(target=_run_job, args=(task_id, payload), daemon=True).start()
    return {"task_id": task_id, "output_path": payload.output_path}


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
