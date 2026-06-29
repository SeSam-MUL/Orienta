"""
Simulation API Routes

Wraps SimulationController for:
- Configuration management
- Simulation start/stop
- Progress monitoring via WebSocket
- System status checks
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU forward-sim .sht bandwidth
# ---------------------------------------------------------------------------
# The shared request default (60) is the EMsoft EMEBSDmasterSHT default.  The GPU
# .sht path uses 384.  resolve_gpu_bandwidth treats the legacy default 60 as
# "unset → use the GPU default", while honouring any explicit non-default value.
GPU_SHT_BANDWIDTH = 384
_EMSOFT_DEFAULT_BANDWIDTH = 60.0


def resolve_gpu_bandwidth(requested: float) -> int:
    """Effective .sht bandwidth for the GPU path.

    The request model shares one ``bandwidth`` field (default 60, the EMsoft
    value) between both engines.  For the GPU path we use 384 unless the caller
    explicitly set a different value (anything other than the EMsoft default 60).
    """
    if float(requested) == _EMSOFT_DEFAULT_BANDWIDTH:
        return GPU_SHT_BANDWIDTH
    return int(requested)


# ---------------------------------------------------------------------------
# Persistent per-job log files
# ---------------------------------------------------------------------------
_SIM_LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sim_logs"


def _get_job_log_path(task_id: str) -> Path:
    _SIM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _SIM_LOG_DIR / f"{task_id}.log"


def _write_job_log(task_id: str, lines: list, crystal: str = "", error: str = ""):
    """Write all log lines for a finished job to a persistent file."""
    try:
        path = _get_job_log_path(task_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Simulation Log — {crystal}\n")
            f.write(f"# Task ID: {task_id}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            if error:
                f.write(f"# ERROR: {error}\n")
            f.write("#" + "-" * 60 + "\n")
            for line in lines:
                f.write(line + "\n")
        logger.info("Saved simulation log: %s", path.name)
    except Exception as e:
        logger.warning("Could not write job log for %s: %s", task_id, e)
router = APIRouter()

_controller = None
_simulation_tasks = {}
_batch_jobs = {}  # {batch_id: {"status", "jobs": [...], "completed", "failed"}}
_tasks_lock = threading.Lock()  # Protects _simulation_tasks and _batch_jobs
_MAX_FINISHED_TASKS = 50  # Keep at most this many finished tasks in memory

# GPU job lock — serialises build_master work on the single GPU.
# A single /start-gpu and a /batch/start-gpu (or two concurrent batches) must
# not run two build_master calls concurrently on the same GPU (VRAM exhaustion /
# incorrect results).  Callers acquire this lock around run_gpu_simulation; the
# lock queues rather than rejects so all requests eventually complete.
_gpu_build_lock = threading.Lock()


def _cleanup_finished_tasks():
    """Remove oldest finished tasks when exceeding _MAX_FINISHED_TASKS.

    MUST be called with _tasks_lock held.
    """
    finished = [
        tid for tid, t in _simulation_tasks.items()
        if t.get("status") in ("completed", "failed", "cancelled")
    ]
    if len(finished) > _MAX_FINISHED_TASKS:
        for tid in finished[:-_MAX_FINISHED_TASKS]:
            _simulation_tasks.pop(tid, None)

    finished_batches = [
        bid for bid, b in _batch_jobs.items()
        if b.get("status") in ("completed", "cancelled")
    ]
    if len(finished_batches) > _MAX_FINISHED_TASKS:
        for bid in finished_batches[:-_MAX_FINISHED_TASKS]:
            _batch_jobs.pop(bid, None)

# ---------------------------------------------------------------------------
# History persistence helpers
# ---------------------------------------------------------------------------
_HISTORY_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "simulation_history.json"


def _load_history() -> list:
    try:
        if _HISTORY_FILE.exists():
            return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_history(entries: list) -> None:
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_FILE.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save simulation history: %s", e)


def _append_history(entry: dict) -> None:
    entries = _load_history()
    entries.append(entry)
    _save_history(entries)


_SERVER_CONFIG_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "server_config.json"


def _load_server_config() -> dict:
    try:
        if _SERVER_CONFIG_FILE.exists():
            return json.loads(_SERVER_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"enabled": False, "database_root": "", "offline_mode": False}


def _save_server_config(cfg: dict) -> None:
    try:
        _SERVER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SERVER_CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save server config: %s", e)


def _get_controller():
    global _controller
    if _controller is None:
        from simulation.simulation_controller import SimulationController
        _controller = SimulationController()
    return _controller


class SimulationStartRequest(BaseModel):
    xtal_path: str
    ekev: float = 20.0
    sig: float = 70.0
    omega: float = 0.0
    numsx: int = 501
    totnum_el: int = 500000000
    dmin: float = 0.05
    # bandwidth: EMsoft EMEBSDmasterSHT spherical-harmonic bandwidth.  The EMsoft
    # default is 60; the GPU forward-sim .sht path uses 384 (see GPU_SHT_BANDWIDTH
    # / resolve_gpu_bandwidth — the EMsoft path is unaffected by that resolution).
    bandwidth: float = 60.0
    npx: int = 500
    nthreads: int = 10
    output_type: str = "sht_only"
    compute_mode: str = "auto"
    platid: int = 1
    devid: int = 1
    globalworkgrpsz: int = 150
    # Advanced Monte Carlo parameters
    depthmax: float = 100.0
    depthstep: float = 1.0
    # Advanced Master Pattern parameters
    combinesites: bool = False
    use_energy_weighting: bool = False
    do_legendre: bool = False
    esel: int = -1
    uniform: bool = False


class ConfigUpdateRequest(BaseModel):
    section: str
    key: str
    value: str


@router.get("/config")
async def get_config():
    """Get current simulation configuration."""
    ctrl = _get_controller()
    issues = ctrl.validate_config()

    config_dict = {}
    if ctrl.config:
        for section in ctrl.config.sections():
            config_dict[section] = dict(ctrl.config[section])

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "config": config_dict,
    }


@router.post("/config/update")
async def update_config(req: ConfigUpdateRequest):
    """Update a config value."""
    ctrl = _get_controller()
    if ctrl.config is None:
        raise HTTPException(status_code=400, detail="No config loaded")

    if not ctrl.config.has_section(req.section):
        ctrl.config.add_section(req.section)
    ctrl.config.set(req.section, req.key, req.value)

    # Save to disk
    with open(ctrl.config_path, 'w') as f:
        ctrl.config.write(f)

    return {"success": True}


# system-status runs WSL + EMsoft + OpenCL checks that take ~5 seconds.
# The StatusBar hits this on every page render, so we cache the result for
# 5 minutes. Force a re-check with ?force=true (Settings page "Re-Check" btn).
import time as _time
_system_status_cache = {"value": None, "expires_at": 0.0}
_SYSTEM_STATUS_TTL_SEC = 300.0  # 5 minutes


def forward_sim_capabilities() -> dict:
    """Report which hardware paths the EMsoft-free "Ours" engine can use.

    The frontend uses this to (1) label the "Ours" hardware path (GPU / CPU) and
    (2) enable/disable the manual ``[EMsoft | Ours]`` engine switch — EMsoft is an
    OPTIONAL engine, disabled in the UI when WSL+EMsoft is not detected; "Ours"
    requires neither WSL nor EMsoft and is always available (GPU or CPU).

    Returns a dict with three booleans:

    * ``gpu_available`` — a CUDA GPU is present (``torch.cuda.is_available``) AND
      the cupy one-thread-per-electron MC kernel compiles.  This is the fast
      "Ours" path; when False, "Ours" still runs on the numba/PyTorch CPU MC.
    * ``numba_available`` — the numba njit CPU MC kernel compiles (the no-GPU
      fast path; when False, "Ours" falls back to the slow PyTorch CPU loop).
    * ``emsoft_available`` — WSL + the EMMCOpenCL binary are reachable (the
      manual EMsoft engine can be selected).  NOT required by "Ours".

    Every probe is wrapped fail-safe (returns False on any error) so a missing
    optional dependency never breaks the status endpoint.
    """
    def _torch_cuda() -> bool:
        try:
            import torch  # noqa: PLC0415
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _gpu_mc() -> bool:
        # cupy MC kernel compiles AND a CUDA device is present.
        if not _torch_cuda():
            return False
        try:
            from backend.forward_sim.mc.gpu_monte_carlo import cupy_mc_available  # noqa: PLC0415
            return bool(cupy_mc_available())
        except Exception:
            return False

    def _numba_mc() -> bool:
        try:
            from backend.forward_sim.mc.numba_mc import numba_mc_available  # noqa: PLC0415
            return bool(numba_mc_available())
        except Exception:
            return False

    def _emsoft() -> bool:
        try:
            from backend.api.services.gpu_sim_runner import _wsl_available  # noqa: PLC0415
            return bool(_wsl_available())
        except Exception:
            return False

    return {
        "gpu_available": _gpu_mc(),
        "numba_available": _numba_mc(),
        "emsoft_available": _emsoft(),
    }


def _build_system_status() -> dict:
    """Run the full check and format the response dict."""
    from simulation.system_check import check_system_status
    status = check_system_status()
    # EMsoft-free "Ours"-engine capability probe (GPU / numba / EMsoft switch).
    caps = forward_sim_capabilities()
    return {
        # emsoft_available: prefer the dedicated WSL+binary probe (what the manual
        # EMsoft switch actually needs); fall back to the system_check value.
        "emsoft_available": caps["emsoft_available"] or getattr(status, 'emsoft_available', False),
        "wsl_installed": getattr(status, 'wsl_installed', False),
        "wsl_distro": getattr(status, 'wsl_distro', ''),
        "emsphinx_available": getattr(status, 'emsphinx_available', False),
        "opencl_available": getattr(status, 'opencl_available', False),
        "has_gpu": getattr(status, 'has_gpu', False),
        "gpu_name": getattr(status, 'gpu_name', ''),
        "gpu_memory_gb": getattr(status, 'gpu_memory_gb', 0),
        "cpu_count": getattr(status, 'cpu_count', 1),
        "recommended_mode": getattr(status, 'recommended_mode', 'cpu'),
        "recommended_settings": getattr(status, 'recommended_settings', {}),
        "errors": getattr(status, 'errors', []),
        "warnings": getattr(status, 'warnings', []),
        # --- "Ours" engine hardware capabilities (EMsoft-free) ---
        "gpu_available": caps["gpu_available"],
        "numba_available": caps["numba_available"],
        "checked_at": _time.time(),
    }


@router.get("/system-status")
async def system_status(force: bool = False):
    """Check system status (EMsoft, WSL, OpenCL).

    Cached for 5 minutes because the underlying checks spawn WSL
    subprocesses and take ~5 seconds. Pass ?force=true to bypass the
    cache (used by the Settings page "Re-Check" button).
    """
    try:
        now = _time.time()
        if not force and _system_status_cache["value"] is not None and now < _system_status_cache["expires_at"]:
            cached = dict(_system_status_cache["value"])
            cached["cached"] = True
            return cached

        value = _build_system_status()
        _system_status_cache["value"] = value
        _system_status_cache["expires_at"] = now + _SYSTEM_STATUS_TTL_SEC
        return {**value, "cached": False}
    except Exception as e:
        return {"error": str(e)}


@router.get("/forward-sim/capabilities")
async def forward_sim_capabilities_endpoint(include_emsoft: bool = True):
    """Report the EMsoft-free "Ours"-engine hardware capabilities.

    Returns ``{gpu_available, numba_available, emsoft_available,
    ours_hardware_path}`` so the frontend can label the "Ours" hardware path and
    enable/disable the manual ``[EMsoft | Ours]`` switch.  ``ours_hardware_path``
    is a human label for the path "Ours" will actually take ("GPU (CUDA + cupy)"
    / "CPU (numba)" / "CPU (PyTorch)") — "Ours" always runs regardless.

    Pass ``?include_emsoft=false`` to skip the WSL probe (a cheap ``test -f``,
    but it does spawn WSL) when only the "Ours" GPU/CPU label is needed; then
    ``emsoft_available`` is reported as ``None`` (not probed).
    """
    try:
        caps = forward_sim_capabilities()
        # Honour include_emsoft=false: report the EMsoft probe result only when
        # asked (forward_sim_capabilities already ran the cheap WSL test -f, so we
        # simply suppress reporting it; the WSL probe is itself fail-safe + fast).
        emsoft = caps["emsoft_available"] if include_emsoft else None
        if caps["gpu_available"]:
            path = "GPU (CUDA + cupy)"
        elif caps["numba_available"]:
            path = "CPU (numba)"
        else:
            path = "CPU (PyTorch)"
        return {
            "gpu_available": caps["gpu_available"],
            "numba_available": caps["numba_available"],
            "emsoft_available": emsoft,
            "ours_hardware_path": path,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/start")
async def start_simulation(req: SimulationStartRequest, background_tasks: BackgroundTasks):
    """Start a simulation as a background task."""
    ctrl = _get_controller()

    task_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    with _tasks_lock:
        _cleanup_finished_tasks()
        _simulation_tasks[task_id] = {
            "status": "running",
            "progress": 0.0,
            "message": "Starting simulation...",
            "result": None,
            "error": None,
            "logLines": [],
        }

    def run_sim():
        try:
            from simulation.simulation_controller import SimulationParameters, OutputType
            # Build params dict with all fields, using getattr for optional ones
            param_kwargs = dict(
                ekev=req.ekev, sig=req.sig, omega=req.omega, numsx=req.numsx,
                totnum_el=req.totnum_el, dmin=req.dmin, bandwidth=req.bandwidth,
                npx=req.npx, nthreads=req.nthreads, output_type=req.output_type,
                compute_mode=req.compute_mode, platid=req.platid, devid=req.devid,
                globalworkgrpsz=req.globalworkgrpsz,
                depthmax=req.depthmax, depthstep=req.depthstep,
                combinesites=req.combinesites, useEnergyWeighting=req.use_energy_weighting,
                doLegendre=req.do_legendre, Esel=req.esel, uniform=req.uniform,
            )
            # Only pass kwargs that SimulationParameters accepts
            import inspect
            valid_keys = set(inspect.signature(SimulationParameters).parameters.keys())
            param_kwargs = {k: v for k, v in param_kwargs.items() if k in valid_keys}
            params = SimulationParameters(**param_kwargs)

            job_id = ctrl.start_simulation(req.xtal_path, params)

            # Store job_id so the stop endpoint can cancel via controller
            _simulation_tasks[task_id]["job_id"] = job_id

            # Monitor job progress (max 12 hours timeout)
            import time
            max_wait = 12 * 3600  # 12 hours
            start_time = time.time()
            while time.time() - start_time < max_wait:
                # Check if cancelled from outside
                if _simulation_tasks[task_id].get("status") == "cancelled":
                    ctrl.cancel_simulation(job_id)
                    break
                job = ctrl.jobs.get(job_id)
                if job is None:
                    break
                _simulation_tasks[task_id]["message"] = job.progress_message
                _simulation_tasks[task_id]["progress"] = round(job.progress_pct, 1)
                _simulation_tasks[task_id]["logLines"] = list(job.log_lines or [])
                if job.status.value in ("completed", "failed", "cancelled"):
                    _simulation_tasks[task_id]["status"] = job.status.value
                    if job.result_files:
                        _simulation_tasks[task_id]["result"] = job.result_files
                    if job.error_message:
                        _simulation_tasks[task_id]["error"] = job.error_message
                    break
                time.sleep(1)
            else:
                _simulation_tasks[task_id]["status"] = "failed"
                _simulation_tasks[task_id]["error"] = "Simulation timed out after 12 hours"

        except Exception as e:
            _simulation_tasks[task_id]["status"] = "failed"
            _simulation_tasks[task_id]["error"] = str(e)
        finally:
            # Write persistent log file
            task_data = _simulation_tasks.get(task_id, {})
            log_lines = task_data.get("logLines", [])
            error_msg = task_data.get("error", "")
            _write_job_log(task_id, log_lines, crystal=req.xtal_path, error=error_msg)

            # Persist completed/failed job to history file (now with error + log path)
            final_status = task_data.get("status", "unknown")
            _append_history({
                "id": task_id,
                "crystal": req.xtal_path,
                "method": req.output_type,
                "status": final_status,
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output_path": str(task_data.get("result") or ""),
                "error": error_msg,
                "log_file": str(_get_job_log_path(task_id)),
                "params": {
                    "ekev": req.ekev,
                    "sig": req.sig,
                    "totnum_el": req.totnum_el,
                    "nthreads": req.nthreads,
                    "compute_mode": req.compute_mode,
                },
            })

    background_tasks.add_task(run_sim)
    return {"task_id": task_id, "status": "running"}


@router.get("/status/{task_id}")
async def get_simulation_status(task_id: str):
    """Check simulation task status."""
    with _tasks_lock:
        if task_id not in _simulation_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return dict(_simulation_tasks[task_id])  # Return copy to avoid mutation


@router.post("/stop/{task_id}")
async def stop_simulation(task_id: str):
    """Stop a running simulation.

    Marks the task as cancelled so the monitoring loop calls ctrl.cancel_simulation,
    and also kills any running WSL/EMsoft subprocess directly.
    """
    with _tasks_lock:
        if task_id not in _simulation_tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task = _simulation_tasks[task_id]
        if task.get("status") not in ("running", "queued"):
            return {"success": False, "detail": f"Task is already {task.get('status')}"}

        # Mark cancelled — the background thread's monitoring loop will pick this up
        task["status"] = "cancelled"
        task["message"] = "Cancelled by user"

    # Also cancel via controller (handles job_id lookup + WSL pkill)
    ctrl = _get_controller()
    job_id = task.get("job_id")
    if job_id:
        try:
            ctrl.cancel_simulation(job_id)
        except Exception as e:
            logger.warning("cancel_simulation(%s) raised: %s", job_id, e)
    else:
        # job_id not stored yet — only kill WSL processes if no OTHER tasks are running
        other_running = any(
            t.get("status") == "running" and tid != task_id
            for tid, t in _simulation_tasks.items()
        )
        if other_running:
            logger.warning("Skipping WSL pkill: other tasks are still running")
        else:
            try:
                ctrl._kill_wsl_simulation_processes()
            except Exception as e:
                logger.warning("_kill_wsl_simulation_processes raised: %s", e)

    return {"success": True, "task_id": task_id}


@router.get("/log/{task_id}", response_class=PlainTextResponse)
async def get_simulation_log(task_id: str):
    """Get persistent log file for a simulation job."""
    log_path = _get_job_log_path(task_id)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    return log_path.read_text(encoding="utf-8")


@router.get("/history")
async def get_history():
    """Get persisted simulation job history."""
    return {"jobs": _load_history()}


@router.post("/history/clear")
async def clear_history():
    """Clear persisted simulation job history."""
    _save_history([])
    return {"success": True}


@router.get("/recommend-engine")
async def recommend_engine_endpoint(xtal: str, dmin: float = 0.05):
    """Auto-route a single job: which engine for this ``.xtal`` at this dmin.

    Query params:
      xtal  — absolute path to the ``.xtal`` file (required).
      dmin  — effective dmin the job will run with (default 0.05).

    Response (200):
      {"engine": "emsoft"|"gpu", "reason": str, "reflections": int,
       "point_group_order": int, "emsoft_available": bool}

    404 if the xtal does not exist; 500 on read/parse error.
    """
    from backend.api.services.engine_router import recommend_engine_for_xtal
    try:
        rec = recommend_engine_for_xtal(xtal, float(dmin))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("recommend_engine failed for %s: %s", xtal, e)
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "engine": rec.engine,
        "reason": rec.reason,
        "reflections": rec.reflections,
        "point_group_order": rec.point_group_order,
        "emsoft_available": rec.emsoft_available,
    }


@router.get("/scan-missing")
async def scan_missing_materials(output_type: str = "both", ekev: float = 20.0):
    """Scan XTAL Library for materials that still need simulation.

    Each entry is enriched with the largest lattice parameter and an adaptive
    dmin recommendation (small cells keep the 0.05 floor; large cells get a
    coarser dmin so the reflection range stays ~15) so the UI can warn about
    days-long large-cell sims and pre-fill a sensible per-phase dmin.
    """
    ctrl = _get_controller()
    try:
        from simulation.simulation_controller import (
            SimulationParameters, recommend_dmin,
            read_xtal_max_lattice_nm, read_xtal_min_occupancy,
        )
        params = SimulationParameters(ekev=ekev)
        missing = ctrl.scan_missing_materials(output_type, params)
        # Per-phase engine routing (additive — existing consumers unaffected).
        # Probe EMsoft/WSL availability ONCE for the whole scan, then reuse it.
        from backend.api.services.engine_router import (
            recommend_engine, point_group_order,
        )
        from backend.forward_sim.crystal.xtal_io import read_crystal_structure
        from backend.forward_sim.crystal.structure_matrix import reflection_list
        from backend.api.services.gpu_sim_runner import _wsl_available
        emsoft_avail = _wsl_available()
        for m in missing:
            a_max = read_xtal_max_lattice_nm(m.get("xtal_path"))
            rec = recommend_dmin(a_max) if a_max else None
            if rec:
                m["a_max_nm"] = rec["a_max_nm"]
                m["recommended_dmin"] = rec["recommended_dmin"]
                m["range_at_floor"] = rec["range_at_floor"]
                m["range_at_recommended"] = rec["range_at_recommended"]
                m["dmin_level"] = rec["level"]
            else:
                m["a_max_nm"] = None
                m["recommended_dmin"] = None
                m["dmin_level"] = "unknown"
            # Disorder check: partially-occupied / mixed sites can hang EMsoft.
            occ = read_xtal_min_occupancy(m.get("xtal_path"))
            m["min_occupancy"] = round(occ, 3) if occ is not None else None
            if occ is None:
                m["occupancy_level"] = "unknown"
            elif occ < 0.5:
                m["occupancy_level"] = "severe"   # likely hangs the master-pattern run
            elif occ < 0.98:
                m["occupancy_level"] = "mild"      # disordered but has run before
            else:
                m["occupancy_level"] = "ok"
            # Engine routing: count reflections at the phase's EFFECTIVE dmin
            # (the already-computed recommended_dmin, falling back to the 0.05
            # floor) so the route matches what the batch will actually run.
            try:
                eff_dmin = m.get("recommended_dmin") or 0.05
                struct = read_crystal_structure(m["xtal_path"])
                n_refl = int(reflection_list(struct, float(eff_dmin)).shape[0])
                pg = point_group_order(struct.space_group)
                rec_eng = recommend_engine(
                    reflections=n_refl, point_group_order=pg,
                    emsoft_available=emsoft_avail,
                )
                m["recommended_engine"] = rec_eng.engine
                m["engine_reason"] = rec_eng.reason
                m["reflections"] = rec_eng.reflections
                m["point_group_order"] = rec_eng.point_group_order
            except Exception as e:  # fail-soft: leave routing unknown, don't 500 the scan
                logger.warning("engine routing failed for %s: %s", m.get("stem"), e)
                m["recommended_engine"] = None
                m["engine_reason"] = f"routing failed: {e}"
        return {"missing": missing, "count": len(missing)}
    except Exception as e:
        logger.error("scan_missing_materials failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class BatchStartRequest(BaseModel):
    xtal_paths: list  # List of .xtal file paths
    dmin_overrides: dict = {}  # {xtal stem: dmin} — per-phase adaptive dmin; falls back to `dmin`
    ekev: float = 20.0
    sig: float = 70.0
    omega: float = 0.0
    numsx: int = 501
    totnum_el: int = 500000000
    dmin: float = 0.05
    # bandwidth: EMsoft default 60; the GPU .sht path uses 384 (resolve_gpu_bandwidth).
    bandwidth: float = 60.0
    npx: int = 500
    nthreads: int = 10
    output_type: str = "both"
    compute_mode: str = "auto"
    platid: int = 1
    devid: int = 1
    globalworkgrpsz: int = 150
    depthmax: float = 100.0
    depthstep: float = 1.0
    combinesites: bool = False
    use_energy_weighting: bool = False
    do_legendre: bool = False
    esel: int = -1
    uniform: bool = False


@router.post("/batch/start")
async def start_batch(req: BatchStartRequest, background_tasks: BackgroundTasks):
    """Start a sequential batch of simulations. Jobs run one at a time."""
    if not req.xtal_paths:
        raise HTTPException(status_code=400, detail="No xtal_paths provided")

    batch_id = str(uuid.uuid4())
    job_entries = [
        {"task_id": str(uuid.uuid4()), "xtal_path": p, "status": "pending", "progress": 0.0, "message": "Pending"}
        for p in req.xtal_paths
    ]
    with _tasks_lock:
        _cleanup_finished_tasks()
    _batch_jobs[batch_id] = {
        "status": "running",
        "jobs": job_entries,
        "completed": 0,
        "failed": 0,
    }

    def run_batch():
        ctrl = _get_controller()
        batch = _batch_jobs[batch_id]
        for entry in batch["jobs"]:
            if batch["status"] == "cancelled":
                entry["status"] = "cancelled"
                continue

            entry["status"] = "running"
            entry["message"] = "Starting..."
            task_id = entry["task_id"]

            try:
                from simulation.simulation_controller import SimulationParameters
                import inspect, time

                # Determine what's actually missing for this specific material
                actual_output_type = req.output_type
                if req.output_type == "both":
                    try:
                        # Compute missing list once (cached outside loop if not yet done)
                        if '_missing_cache' not in batch:
                            check_params = SimulationParameters(ekev=req.ekev)
                            batch['_missing_cache'] = ctrl.scan_missing_materials(req.output_type, check_params)
                        per_mat = batch['_missing_cache']
                        xtal_stem = Path(entry["xtal_path"]).stem
                        match = next((m for m in per_mat if m["stem"] == xtal_stem), None)
                        if match:
                            missing = match["missing"]
                            has_sht = "sht" not in missing
                            has_master = "master" not in missing
                            if has_sht and not has_master:
                                actual_output_type = "master_only"
                                logger.info("Batch: %s — SHT exists, only running master", xtal_stem)
                            elif has_master and not has_sht:
                                actual_output_type = "sht_only"
                                logger.info("Batch: %s — Master exists, only running SHT", xtal_stem)
                            elif has_sht and has_master:
                                logger.info("Batch: %s — SHT+Master both exist, skipping", xtal_stem)
                                entry["status"] = "completed"
                                entry["message"] = "Already complete"
                                batch["completed"] += 1
                                continue
                        else:
                            logger.info("Batch: %s — not in missing list, skipping", xtal_stem)
                            entry["status"] = "completed"
                            entry["message"] = "Already complete"
                            batch["completed"] += 1
                            continue
                    except Exception as e:
                        logger.warning("Batch: could not check missing for %s: %s", entry["xtal_path"], e)

                # Per-phase dmin: adaptive override from the UI keyed by xtal
                # stem, else the batch-wide default. Lets a mixed batch give
                # small cells a fine dmin and large cells a coarser one.
                entry_dmin = req.dmin_overrides.get(Path(entry["xtal_path"]).stem, req.dmin)
                entry["dmin_used"] = entry_dmin

                param_kwargs = dict(
                    ekev=req.ekev, sig=req.sig, omega=req.omega, numsx=req.numsx,
                    totnum_el=req.totnum_el, dmin=entry_dmin, bandwidth=req.bandwidth,
                    npx=req.npx, nthreads=req.nthreads, output_type=actual_output_type,
                    compute_mode=req.compute_mode, platid=req.platid, devid=req.devid,
                    globalworkgrpsz=req.globalworkgrpsz,
                    depthmax=req.depthmax, depthstep=req.depthstep,
                    combinesites=req.combinesites,
                    useEnergyWeighting=req.use_energy_weighting,
                    doLegendre=req.do_legendre, Esel=req.esel, uniform=req.uniform,
                )
                valid_keys = set(inspect.signature(SimulationParameters).parameters.keys())
                param_kwargs = {k: v for k, v in param_kwargs.items() if k in valid_keys}
                params = SimulationParameters(**param_kwargs)

                job_id = ctrl.start_simulation(entry["xtal_path"], params)
                entry["job_id"] = job_id

                max_wait = 12 * 3600
                start_time = time.time()
                while time.time() - start_time < max_wait:
                    if batch["status"] == "cancelled":
                        ctrl.cancel_simulation(job_id)
                        entry["status"] = "cancelled"
                        break
                    job = ctrl.jobs.get(job_id)
                    if job is None:
                        break
                    entry["message"] = job.progress_message
                    entry["progress"] = round(getattr(job, 'progress_pct', 0.0), 1)
                    entry["logLines"] = list(job.log_lines or [])
                    if job.status.value in ("completed", "failed", "cancelled"):
                        entry["status"] = job.status.value
                        if job.result_files:
                            entry["result"] = job.result_files
                        if job.error_message:
                            entry["error"] = job.error_message
                        break
                    time.sleep(1)
                else:
                    entry["status"] = "failed"
                    entry["error"] = "Timed out"

            except Exception as e:
                entry["status"] = "failed"
                entry["error"] = str(e)

            if entry["status"] == "completed":
                batch["completed"] += 1
            elif entry["status"] in ("failed", "cancelled"):
                batch["failed"] += 1

            # Write persistent log file for this job
            job_log_lines = entry.get("logLines", [])
            job_error = entry.get("error", "")
            _write_job_log(task_id, job_log_lines, crystal=entry["xtal_path"], error=job_error)

            _append_history({
                "id": task_id,
                "crystal": entry["xtal_path"],
                "method": req.output_type,
                "status": entry["status"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output_path": str(entry.get("result") or ""),
                "error": job_error,
                "log_file": str(_get_job_log_path(task_id)),
                "params": {"ekev": req.ekev, "sig": req.sig, "totnum_el": req.totnum_el,
                           "nthreads": req.nthreads, "compute_mode": req.compute_mode},
            })

        batch["status"] = "completed"

    background_tasks.add_task(run_batch)
    return {"batch_id": batch_id, "job_count": len(job_entries), "jobs": job_entries}


@router.get("/batch/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of a batch simulation."""
    with _tasks_lock:
        if batch_id not in _batch_jobs:
            raise HTTPException(status_code=404, detail="Batch not found")
        return dict(_batch_jobs[batch_id])


@router.post("/batch/cancel/{batch_id}")
async def cancel_batch(batch_id: str):
    """Cancel remaining jobs in a batch."""
    with _tasks_lock:
        if batch_id not in _batch_jobs:
            raise HTTPException(status_code=404, detail="Batch not found")
        _batch_jobs[batch_id]["status"] = "cancelled"
    return {"success": True}


# ---------------------------------------------------------------------------
# GPU-native batch endpoint — "Simulate All Missing" on the GPU engine
# ---------------------------------------------------------------------------

@router.post("/batch/start-gpu")
async def start_batch_gpu(req: BatchStartRequest, background_tasks: BackgroundTasks):
    """Start a sequential GPU-native batch (the GPU "Simulate All Missing").

    Mirrors :func:`start_batch` exactly — the SAME ``BatchStartRequest`` body, the
    SAME ``_batch_jobs`` dict shape (so ``GET /batch/status/{batch_id}`` and the
    frontend batch polling work unchanged), the SAME per-phase ``scan_missing``
    skip logic and per-phase ``dmin_overrides`` — but runs each phase through the
    GPU forward-sim runner (PyTorch MC + dynamical master) instead of the EMsoft
    controller.  History entries are tagged ``engine='gpu'``.

    Requires CUDA (the GPU runner fails loud via ``get_device`` if no device).
    """
    if not req.xtal_paths:
        raise HTTPException(status_code=400, detail="No xtal_paths provided")

    batch_id = str(uuid.uuid4())
    job_entries = [
        {"task_id": str(uuid.uuid4()), "xtal_path": p, "status": "pending", "progress": 0.0, "message": "Pending"}
        for p in req.xtal_paths
    ]
    with _tasks_lock:
        _cleanup_finished_tasks()
    _batch_jobs[batch_id] = {
        "status": "running",
        "jobs": job_entries,
        "completed": 0,
        "failed": 0,
        "engine": "gpu",
    }

    def run_batch_gpu():
        from backend.api.services.gpu_sim_runner import run_gpu_simulation

        batch = _batch_jobs[batch_id]
        # scan_missing is engine-agnostic (it inspects the SAME Database output
        # paths the GPU runner writes), so reuse the controller's scanner to skip
        # already-complete phases — exactly like the EMsoft batch.
        ctrl = _get_controller()

        for entry in batch["jobs"]:
            if batch["status"] == "cancelled":
                entry["status"] = "cancelled"
                continue

            entry["status"] = "running"
            entry["message"] = "Starting (GPU)…"
            task_id = entry["task_id"]
            log_lines: list = []

            def _progress(pct: float, msg: str, _entry=entry) -> None:
                _entry["progress"] = round(pct, 1)
                _entry["message"] = msg

            def _log(line: str, _entry=entry, _lines=log_lines) -> None:
                _lines.append(line)
                _entry["logLines"] = list(_lines)

            try:
                from simulation.simulation_controller import SimulationParameters

                # Determine what's actually missing for this material (same logic
                # as the EMsoft batch: skip complete phases, narrow output_type).
                actual_output_type = req.output_type
                if req.output_type == "both":
                    try:
                        if '_missing_cache' not in batch:
                            check_params = SimulationParameters(ekev=req.ekev)
                            batch['_missing_cache'] = ctrl.scan_missing_materials(
                                req.output_type, check_params
                            )
                        per_mat = batch['_missing_cache']
                        xtal_stem = Path(entry["xtal_path"]).stem
                        match = next((m for m in per_mat if m["stem"] == xtal_stem), None)
                        if match:
                            missing = match["missing"]
                            has_sht = "sht" not in missing
                            has_master = "master" not in missing
                            if has_sht and not has_master:
                                actual_output_type = "master_only"
                            elif has_master and not has_sht:
                                actual_output_type = "sht_only"
                            elif has_sht and has_master:
                                entry["status"] = "completed"
                                entry["message"] = "Already complete"
                                batch["completed"] += 1
                                continue
                        else:
                            entry["status"] = "completed"
                            entry["message"] = "Already complete"
                            batch["completed"] += 1
                            continue
                    except Exception as e:
                        logger.warning(
                            "GPU batch: could not check missing for %s: %s",
                            entry["xtal_path"], e,
                        )

                # Per-phase adaptive dmin override (same key = xtal stem).
                entry_dmin = req.dmin_overrides.get(Path(entry["xtal_path"]).stem, req.dmin)
                entry["dmin_used"] = entry_dmin

                params = {
                    "ekev": req.ekev,
                    "sig": req.sig,
                    "omega": req.omega,
                    "totnum_el": req.totnum_el,
                    "depthmax": req.depthmax,
                    "depthstep": req.depthstep,
                    "npx": req.npx,
                    "dmin": entry_dmin,
                    "output_type": actual_output_type,
                    "bandwidth": resolve_gpu_bandwidth(req.bandwidth),
                }

                if batch["status"] == "cancelled":
                    entry["status"] = "cancelled"
                    continue

                # Acquire GPU build lock — prevents two concurrent GPU batches
                # (or a /start-gpu racing with this batch) from running
                # build_master simultaneously on the same device.
                _log("[gpu_sim] waiting for GPU build lock…")
                with _gpu_build_lock:
                    _log("[gpu_sim] GPU build lock acquired")
                    result_files = run_gpu_simulation(
                        entry["xtal_path"], params, progress_cb=_progress, log_cb=_log
                    )
                entry["status"] = "completed"
                entry["result"] = result_files
                entry["progress"] = 100.0
                entry["message"] = "Done"

            except Exception as e:
                logger.exception("GPU batch job failed: %s", entry["xtal_path"])
                entry["status"] = "failed"
                entry["error"] = str(e)
                entry["message"] = f"Failed: {e}"

            finally:
                # Free GPU memory between phases so large cells don't OOM the next.
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            if entry["status"] == "completed":
                batch["completed"] += 1
            elif entry["status"] in ("failed", "cancelled"):
                batch["failed"] += 1

            job_error = entry.get("error", "")
            _write_job_log(task_id, log_lines, crystal=entry["xtal_path"], error=job_error)

            _append_history({
                "id": task_id,
                "crystal": entry["xtal_path"],
                "method": req.output_type,
                "status": entry["status"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output_path": str(entry.get("result") or ""),
                "error": job_error,
                "log_file": str(_get_job_log_path(task_id)),
                "params": {"ekev": req.ekev, "sig": req.sig, "totnum_el": req.totnum_el,
                           "npx": req.npx, "compute_mode": "gpu"},
                "engine": "gpu",
            })

        batch["status"] = "completed"

    background_tasks.add_task(run_batch_gpu)
    return {"batch_id": batch_id, "job_count": len(job_entries), "jobs": job_entries}


class ServerConfigRequest(BaseModel):
    enabled: bool = False
    database_root: str = ""
    offline_mode: bool = False


@router.get("/server-config")
async def get_server_config():
    """Get server mode configuration."""
    cfg = _load_server_config()
    discovered = {}
    root = cfg.get("database_root", "")
    if root and Path(root).is_dir():
        root_path = Path(root)
        subfolders = {
            "sht_database": "EBSD_SHT_Database",
            "h5_cache": "EBSD_H5_Cache",
            "dictionary_library": "Dictionary_Library",
            "cif_library": "EBSD_CIF_Library",
            "xtal_library": "EBSD_XTAL_Library",
        }
        for key, folder_name in subfolders.items():
            sub = root_path / folder_name
            discovered[key] = {"path": str(sub), "exists": sub.is_dir()}
    return {**cfg, "discovered": discovered}


@router.post("/server-config")
async def update_server_config(req: ServerConfigRequest):
    """Update server mode configuration."""
    cfg = {
        "enabled": req.enabled,
        "database_root": req.database_root,
        "offline_mode": req.offline_mode,
    }
    _save_server_config(cfg)
    return {"success": True}


@router.post("/server-config/test")
async def test_server_connection():
    """Test if server paths are accessible."""
    cfg = _load_server_config()
    root = cfg.get("database_root", "")
    if not root:
        return {"connected": False, "message": "No database root configured"}
    root_path = Path(root)
    if not root_path.exists():
        return {"connected": False, "message": f"Path not accessible: {root}"}
    checks = {}
    for name in ["EBSD_SHT_Database", "EBSD_H5_Cache"]:
        sub = root_path / name
        checks[name] = sub.is_dir()
    all_ok = all(checks.values())
    return {
        "connected": all_ok,
        "message": "Connected" if all_ok else "Some subfolders missing",
        "checks": checks,
    }


class NmlTemplateRequest(BaseModel):
    xtal_name: str = "Fe"
    ekev: float = 20.0
    sig: float = 70.0
    omega: float = 0.0
    numsx: int = 501
    totnum_el: int = 500000000
    dmin: float = 0.05
    bandwidth: float = 60.0
    npx: int = 500
    nthreads: int = 10
    output_type: str = "sht_only"
    platid: int = 1
    devid: int = 1
    globalworkgrpsz: int = 150
    depthmax: float = 100.0
    depthstep: float = 1.0


def _load_nml_template(name: str) -> str:
    """Load a NML template file from simulation/templates/."""
    template_dir = Path(__file__).parent.parent.parent.parent / "simulation" / "templates"
    path = template_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


@router.get("/nml-template", response_class=PlainTextResponse)
async def get_nml_template(
    xtal_name: str = "Fe",
    ekev: float = 20.0,
    sig: float = 70.0,
    omega: float = 0.0,
    numsx: int = 501,
    totnum_el: int = 500000000,
    dmin: float = 0.05,
    bandwidth: float = 60.0,
    npx: int = 500,
    nthreads: int = 10,
    output_type: str = "sht_only",
    platid: int = 1,
    devid: int = 1,
    globalworkgrpsz: int = 150,
    depthmax: float = 100.0,
    depthstep: float = 1.0,
):
    """Generate NML template content for the given parameters.

    Returns a plain-text string containing the filled NML file(s) separated
    by a comment line, ready to save and pass to EMsoft.
    """
    # Derive base name for output files
    try:
        from simulation.simulation_controller import sanitize_sim_name
        safe_stem = sanitize_sim_name(xtal_name)
    except Exception:
        safe_stem = xtal_name.replace(" ", "_")

    data_name = f"{safe_stem}_E{int(ekev)}kV.h5"
    energy_file = data_name
    bethe_file = "BetheParameters.nml"

    # Common substitutions for EMMCOpenCL
    mc_vars = dict(
        mode="full",
        xtalname=f"{xtal_name}.xtal",
        numsx=numsx,
        sig=sig,
        omega=omega,
        sigstart=2.0,
        sigend=70.0,
        sigstep=2.0,
        ivolx=101,
        ivoly=101,
        ivolz=101,
        ivolstepx=1.0,
        ivolstepy=1.0,
        ivolstepz=1.0,
        num_el=10,
        platid=platid,
        devid=devid,
        globalworkgrpsz=globalworkgrpsz,
        totnum_el=totnum_el,
        multiplier=1,
        EkeV=ekev,
        Ehistmin=5.0,
        Ebinsize=0.5,
        depthmax=depthmax,
        depthstep=depthstep,
        dataname=data_name,
        Notify="Off",
    )

    parts = []

    try:
        mc_tmpl = _load_nml_template("EMMCOpenCL.nml.template")
        mc_nml = mc_tmpl.format(**mc_vars)
        parts.append(f"! === EMMCOpenCL.nml ({xtal_name}, {ekev} kV) ===\n{mc_nml}")
    except Exception as e:
        parts.append(f"! EMMCOpenCL template error: {e}\n")

    if output_type in ("sht_only", "both"):
        sht_vars = dict(
            dmin=dmin,
            nthreads=nthreads,
            energyfile=energy_file,
            BetheParametersFile=bethe_file,
            useDOI="n",
            SHT_formula=xtal_name,
            SHT_name=xtal_name,
            SHT_structuresymbol="",
            SHT_folder="SHT",
            Notify="Off",
        )
        try:
            sht_tmpl = _load_nml_template("EMEBSDmasterSHT.nml.template")
            sht_nml = sht_tmpl.format(**sht_vars)
            parts.append(f"! === EMEBSDmasterSHT.nml ===\n{sht_nml}")
        except Exception as e:
            parts.append(f"! EMEBSDmasterSHT template error: {e}\n")

    if output_type in ("master_only", "both"):
        master_vars = dict(
            dmin=dmin,
            npx=npx,
            nthreads=nthreads,
            energyfile=energy_file,
            BetheParametersFile=bethe_file,
            useDOI="n",
            Notify="Off",
        )
        try:
            master_tmpl = _load_nml_template("EMEBSDmaster.nml.template")
            master_nml = master_tmpl.format(**master_vars)
            parts.append(f"! === EMEBSDmaster.nml ===\n{master_nml}")
        except Exception as e:
            parts.append(f"! EMEBSDmaster template error: {e}\n")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GPU-native forward-simulation endpoint
# ---------------------------------------------------------------------------

@router.post("/start-gpu")
async def start_gpu_simulation(req: SimulationStartRequest, background_tasks: BackgroundTasks):
    """Start a GPU-native forward-simulation as a background task.

    Uses our own PyTorch Monte Carlo + dynamical master builder instead of
    EMsoft/WSL.  The task dict shape and GET /status/{task_id} are identical
    to /start so the frontend Queue + polling work unchanged.

    Requires CUDA (or FORWARD_SIM_DEVICE env var pointing to a device).
    """
    task_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    with _tasks_lock:
        _cleanup_finished_tasks()
        _simulation_tasks[task_id] = {
            "status": "running",
            "progress": 0.0,
            "message": "Starting GPU simulation…",
            "result": None,
            "error": None,
            "logLines": [],
        }

    def run_gpu_sim():
        log_lines: list = []

        def _progress(pct: float, msg: str) -> None:
            _simulation_tasks[task_id]["progress"] = round(pct, 1)
            _simulation_tasks[task_id]["message"] = msg

        def _log(line: str) -> None:
            log_lines.append(line)
            _simulation_tasks[task_id]["logLines"] = list(log_lines)

        try:
            from backend.api.services.gpu_sim_runner import run_gpu_simulation

            params = {
                "ekev": req.ekev,
                "sig": req.sig,
                "omega": req.omega,
                "totnum_el": req.totnum_el,
                "depthmax": req.depthmax,
                "depthstep": req.depthstep,
                "npx": req.npx,
                "dmin": req.dmin,
                "output_type": req.output_type,
                "bandwidth": resolve_gpu_bandwidth(req.bandwidth),
            }

            # Acquire GPU build lock — serialises concurrent /start-gpu and
            # /batch/start-gpu jobs so two build_master calls never run on the
            # same GPU simultaneously (VRAM exhaustion / correctness risk).
            _log("[gpu_sim] waiting for GPU build lock…")
            with _gpu_build_lock:
                _log("[gpu_sim] GPU build lock acquired")
                result_files = run_gpu_simulation(
                    req.xtal_path,
                    params,
                    progress_cb=_progress,
                    log_cb=_log,
                )

            _simulation_tasks[task_id]["status"] = "completed"
            _simulation_tasks[task_id]["result"] = result_files
            _simulation_tasks[task_id]["message"] = "Done"

        except Exception as exc:
            logger.exception("GPU simulation failed for task %s", task_id)
            _simulation_tasks[task_id]["status"] = "failed"
            _simulation_tasks[task_id]["error"] = str(exc)
            _simulation_tasks[task_id]["message"] = f"Failed: {exc}"

        finally:
            # Free GPU memory after the run so a subsequent large-cell job
            # (single or batch) doesn't OOM — matches the per-phase cleanup in
            # the GPU batch path.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            task_data = _simulation_tasks.get(task_id, {})
            error_msg = task_data.get("error", "")
            _write_job_log(task_id, log_lines, crystal=req.xtal_path, error=error_msg)
            final_status = task_data.get("status", "unknown")
            _append_history({
                "id": task_id,
                "crystal": req.xtal_path,
                "method": req.output_type,
                "status": final_status,
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output_path": str(task_data.get("result") or ""),
                "error": error_msg,
                "log_file": str(_get_job_log_path(task_id)),
                "params": {
                    "ekev": req.ekev,
                    "sig": req.sig,
                    "totnum_el": req.totnum_el,
                    "npx": req.npx,
                    "compute_mode": "gpu",
                },
                "engine": "gpu",
            })

    background_tasks.add_task(run_gpu_sim)
    return {"task_id": task_id, "status": "running"}


# ---------------------------------------------------------------------------
# Crystal Picker — scan XTAL database with pipeline status
# ---------------------------------------------------------------------------

@router.get("/crystal-picker")
async def crystal_picker(kv: int = 20):
    """List all .xtal files with MC/Master/SHT pipeline status at given kV.

    Scans:
      - Database/XTAL_Library/ for .xtal files
      - Database/EBSD_H5_Cache/ for MC (_E{kv}kV_) and Master (_master_E{kv}kV_) files
      - Database/EBSD_SHT_Database/ for SHT ({kv}kV) files
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    xtal_dir = project_root / "Database" / "XTAL_Library"
    h5_cache_dir = project_root / "Database" / "EBSD_H5_Cache"
    sht_dir = project_root / "Database" / "EBSD_SHT_Database"

    if not xtal_dir.is_dir():
        return []

    # Collect all H5 and SHT filenames for matching
    h5_files = []
    if h5_cache_dir.is_dir():
        h5_files = [f.name.lower() for f in h5_cache_dir.rglob("*.h5")]

    sht_files = []
    if sht_dir.is_dir():
        sht_files = [f.name.lower() for f in sht_dir.rglob("*.sht") if f.name != "test.sht"]

    kv_tag = f"e{kv}kv"       # For H5 matching: E20kV
    kv_sht = f"{{{kv}kv}}"    # For SHT matching: {20kV}  (with curly braces in filename)
    kv_sht_alt = f"{kv}kv"    # Simpler match without braces

    results = []
    for xtal_file in sorted(xtal_dir.glob("*.xtal")):
        stem = xtal_file.stem
        stem_lower = stem.lower()
        # Normalize stem for matching: replace special chars
        stem_normalized = stem_lower.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")

        # MC: {stem}_E{kv}kV_sig*_n*_o0.h5 — stem appears before _E{kv}kV, NOT _master_
        has_mc = any(
            stem_lower in fn and kv_tag in fn and "_master_" not in fn
            for fn in h5_files
        )

        # Master: {stem}_master_E{kv}kV_npx*.h5
        has_master = any(
            stem_lower in fn and "_master_" in fn and kv_tag in fn
            for fn in h5_files
        )

        # SHT: * ({stem}) [*] {kV}kV.sht — stem appears in parentheses
        # Match by checking if stem (in parens or not) appears AND kV matches
        has_sht = any(
            (f"({stem_lower})" in fn or stem_lower in fn) and kv_sht_alt in fn
            for fn in sht_files
        )

        results.append({
            "name": xtal_file.name,
            "stem": stem,
            "path": str(xtal_file),
            "has_mc": has_mc,
            "has_master": has_master,
            "has_sht": has_sht,
            "pipeline_complete": has_mc and has_master and has_sht,
        })

    # Sort: pipeline_complete first, then has_sht, then by name
    results.sort(key=lambda x: (not x["pipeline_complete"], not x["has_sht"], x["name"].lower()))
    return results
