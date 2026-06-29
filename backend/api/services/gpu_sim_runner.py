"""GPU-native forward-simulation runner service.

This is the "Ours" engine: FULLY SELF-CONTAINED and HARDWARE-ADAPTIVE — it
requires NO EMsoft and NO WSL, and auto-picks the fastest available hardware
path for every step (GPU when CUDA is present, else CPU).  EMsoft is a SEPARATE,
optional engine (the ``/start`` pipeline) the user picks manually; it is never
reached from here.

Wraps the full pipeline:
    read_crystal_structure
    -> [MC SOURCE STRATEGY] obtain MCData (EMsoft-free, hardware-adaptive):
         (a) load from existing MC .h5 (free, ~instant)
         (b) run OUR GPU MC via run_gpu_mc (engine="auto" -> cupy CUDA kernel,
             ~1.4 s / 50M e⁻) when CUDA + cupy are available.  The full-pipeline
             benchmark (tasks/forward_sim/iterations/PIPE2/pipe2.json) proved our
             GPU MC is FASTER than EMsoft EMMCOpenCL (~14 s / 500M default) AND
             master-NCC-equivalent (0.9999 vs the EMsoft oracle).
         (c) run OUR numba CPU MC (engine="auto" -> numba njit prange loop) when
             no CUDA/cupy is present but numba compiles — the no-GPU fast path.
         (d) PyTorch step-major CPU loop (capped) — last resort, emits a LOUD
             warning because it is ~400× slower than the cupy kernel.
       Strategies (b)/(c)/(d) all route through ``run_gpu_mc(engine="auto")``,
       which itself fans out cupy -> numba -> pytorch; the runner only chooses the
       per-host messaging.  EMsoft EMMCOpenCL is NOT in this chain (the "Ours"
       engine must not require WSL/EMsoft, and our MC is faster anyway).
    -> build_master         (dynamical master builder)
    -> write_master_h5      (if output_type in {master_only, both})
    -> write_sht            (if output_type in {sht_only, both})

Output paths follow the same crystal-picker-compatible convention used by
the EMsoft /start endpoint so ``scan_missing`` and the Simulation-page queue
detect the files correctly:

    mc.h5      ->  Database/EBSD_H5_Cache/<stem>/<stem>_E<kv>kV_sig<sig>_n<numsx>_o0.h5
    master.h5  ->  Database/EBSD_H5_Cache/<stem>/<stem>_master_E<kv>kV_npx<npx>.h5
    .sht       ->  Database/EBSD_SHT_Database/<stem>/<formula> (<stem>) [<pearson>] {<kv>kV}.sht
                   (the ``[<pearson>]`` bracket is omitted when the Pearson
                   symbol is unknown; ``(<stem>)`` is always retained so
                   ``scan_missing``'s ``f"({stem})" in name`` guard still matches)

The MC ``.h5`` is ALWAYS written (the Monte Carlo runs regardless of
``output_type``) so a GPU run is detected as ``has_mc`` and "Simulate All
Missing" converges.  The MC name matches scan_missing's
``*/{safe_stem}_E{int(ekev)}kV*.h5`` glob (and is NOT a ``_master_`` file); the
SHT pattern matches scan_missing's ``f"({stem})" in f.name`` guard.

The MC/master filenames use :func:`path_utils.sanitize_filename` — the SAME
single-source-of-truth sanitizer ``scan_missing`` greps with (via
``sanitize_sim_name``) and the EMsoft pipeline names its outputs with.  This is
critical for stems containing spaces, parentheses, commas or Greek letters
(e.g. ``Mg32(Al,Zn)49``, ``α-AlFeSi``): a runner-local sanitizer that only
stripped Windows-illegal characters would diverge from the scan glob and the
batch would re-queue those phases forever.  The SHT filename deliberately uses
the RAW stem on both sides (the ``f"({stem})"`` guard), so it is unaffected.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from path_utils import sanitize_filename, ekev_to_kv_label

logger = logging.getLogger(__name__)

# Project root: this file lives at backend/api/services/gpu_sim_runner.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_H5_CACHE = _PROJECT_ROOT / "Database" / "EBSD_H5_Cache"
_SHT_DB = _PROJECT_ROOT / "Database" / "EBSD_SHT_Database"


def _cif_library_dir() -> Path:
    """Database/CIF_Library, resolved from the project root.

    ``_SHT_DB.parent`` is ``Database/`` (there is no ``_DB_ROOT`` constant in
    this module — the database root is ``_PROJECT_ROOT / "Database"``).
    """
    return _SHT_DB.parent / "CIF_Library"


def _sht_filename_for(stem, xtal_path, energy_kV, sig, cif_library_dir=None):
    """Convention-compliant .sht filename, formula/Pearson read from the CIF.

    Pure + testable: delegates to the Task-1 helpers in ``phase_metadata``.
    The ``({stem})`` token is preserved by ``build_sht_filename`` so
    ``scan_missing`` keeps finding the file via ``f"({stem})" in name``.
    """
    from phase_metadata import resolve_sht_name_fields, build_sht_filename  # noqa: PLC0415
    cif_dir = cif_library_dir if cif_library_dir is not None else _cif_library_dir()
    formula, pearson = resolve_sht_name_fields(
        stem, cif_library_dir=cif_dir, xtal_path=xtal_path
    )
    return build_sht_filename(
        stem, formula, pearson, ekev_to_kv_label(energy_kV), tilt_deg=sig
    )


# The provenance sidecar writer now lives in sht_provenance (engine-neutral, no
# heavy deps) so BOTH the GPU "Ours" runner and the EMsoft controller share it.
# Re-exported here so existing callers/tests keep importing it from gpu_sim_runner.
# (sht_provenance does NOT import gpu_sim_runner -> no import cycle.)
from backend.api.services.sht_provenance import write_provenance_sidecar  # noqa: E402,F401


# The proven EMsoft WSL automation helpers (read EMsoftConfig.json, copy to WSL).
_AUTOMATION_DIR = (
    _PROJECT_ROOT
    / "crystal-structures-for-ebsd-main"
    / "_Phyton_Automization"
    / "windwos_to_WSL"
)

# The GPU Monte Carlo needs FAR fewer electrons than EMsoft's 500M production
# count.  The master pattern consumes only the NORMALISED depth/energy profile
# (lambda_z = accum_z summed over direction, normalised to 1) + the per-energy
# weighting — NOT the absolute backscatter count.  Those profiles converge by
# ~10-25M electrons (SP2 validated master-NCC 1.0 vs the EMsoft-MC master at
# 10M).  Running 500M through the PyTorch MC would take hours for ZERO benefit
# to the master/.sht, so the GPU path defaults to (and caps at) a sufficient
# count.  EMsoft's own MC is already fast; the GPU speedup is the dynamical
# MASTER step, not the MC.
GPU_MC_DEFAULT_ELECTRONS = 25_000_000   # form default when GPU engine is selected
GPU_MC_MAX_ELECTRONS = 50_000_000        # hard safety ceiling (manual overrides)

# EMsoft energy binning defaults — must match the values embedded in the MC .h5
# so load_mc reads the same grid that build_master expects.
_EHISTMIN = 10.0
_EBINSIZE = 1.0


def mc_h5_path(stem: str, ekev_kV: int, sig: float, numsx: int) -> Path:
    """Build the MC ``.h5`` output path the GPU runner writes for ``stem``.

    Uses :func:`path_utils.sanitize_filename` so the produced name is identical
    to what ``SimulationController.scan_missing_materials`` greps for with its
    ``*/{safe_stem}_E{ekev_to_kv_label(ekev)}kV*.h5`` glob (and to the EMsoft
    pipeline's own naming — both use :func:`path_utils.ekev_to_kv_label`).
    Extracted as a helper so the runner→scan_missing naming contract can be
    exercised directly in tests.
    """
    safe = sanitize_filename(stem)
    fname = f"{safe}_E{ekev_kV}kV_sig{int(round(sig))}_n{numsx}_o0.h5"
    return _H5_CACHE / safe / fname


def master_h5_path(stem: str, ekev_kV: int, npx: int) -> Path:
    """Build the master ``.h5`` output path the GPU runner writes for ``stem``.

    Uses :func:`path_utils.sanitize_filename` so the produced name matches
    ``scan_missing``'s ``*/{safe_stem}_master*.h5`` glob.
    """
    safe = sanitize_filename(stem)
    fname = f"{safe}_master_E{ekev_kV}kV_npx{npx}.h5"
    return _H5_CACHE / safe / fname


def _wsl_available() -> bool:
    """Return True if WSL and the EMsoft EMMCOpenCL binary are reachable.

    Runs a cheap ``test -f`` probe via WSL.  Returns False on any error or
    timeout.

    NB: this is a CAPABILITY PROBE only — it is NOT a fallback path for the
    "Ours" pipeline (:func:`run_gpu_simulation` is EMsoft-free; its MC chain is
    cupy -> numba -> pytorch).  It is used by (1) the auto engine-router
    (``engine_router.recommend_engine_for_xtal``) and (2) capability reporting
    so the frontend can enable/disable the manual EMsoft engine switch (EMsoft is
    an optional engine the user picks explicitly, not something "Ours" requires).
    """
    try:
        from simulation.simulation_controller import SimulationController  # noqa: PLC0415
        ctrl = SimulationController()
        emmc_exe = (ctrl.config or {}) and ctrl.config.get(
            "EMsoftPaths", "emmcopencl_executable_wsl", fallback=""
        ) if ctrl.config else ""
        if not emmc_exe:
            # No configured path — probe the well-known location under the WSL
            # user's home. $HOME is expanded INSIDE wsl/bash below, so this works
            # for any username (no hardcoded '/home/<user>').
            emmc_exe = "$HOME/emsoft/builds/EMsoft-Release/Bin/EMMCOpenCL"
        if sys.platform == "win32":
            cmd = ["wsl", "bash", "-lc", f'test -f "{emmc_exe}" && echo yes']
        else:
            cmd = ["bash", "-lc", f'test -f "{emmc_exe}" && echo yes']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() == "yes"
    except Exception:
        return False


def run_gpu_simulation(
    xtal_path: str,
    params: dict,
    progress_cb: Callable[[float, str], None],
    log_cb: Callable[[str], None],
) -> Dict[str, Optional[str]]:
    """Run the full GPU forward-sim pipeline and return paths to produced files.

    This is the "Ours" engine: EMsoft-free and hardware-adaptive.  It NEVER
    invokes EMsoft / WSL — that is a separate, optional engine the user picks
    manually (the ``/start`` pipeline, unchanged).

    MC Source Strategy (EMsoft-free, hardware-adaptive):
    1. If an MC .h5 already exists at this runner's OWN canonical output path
       (:func:`mc_h5_path`, the same name the runner writes and scan_missing
       greps for) → load it immediately via ``emsoft_mc_input.load_mc`` (free,
       instant).  This is a self-consistent cache hit: a prior run for the same
       stem/kV/sig/numsx.
    2. Else run OUR MC via ``run_gpu_mc(engine="auto")``, which fans out to the
       fastest available hardware path with NO EMsoft:
         * cupy one-thread-per-electron CUDA kernel when CUDA + cupy compile
           (~1.4 s / 50M e⁻; the full-pipeline benchmark proved this is FASTER
           than EMsoft EMMCOpenCL and master-NCC-equivalent, 0.9999), else
         * the numba njit prange CPU loop when numba compiles (no-GPU fast path),
           else
         * the validated PyTorch step-major CPU loop (last resort).
       The electron count is capped to ``GPU_MC_MAX_ELECTRONS`` (50M): the master
       only consumes the NORMALISED depth/energy profile, which converges well
       before this.  When neither CUDA+cupy is present (i.e. the slow CPU paths
       will be used) the runner emits a LOUD log warning.

    After obtaining MCData, the pipeline is unchanged: build_master →
    write_master_h5 / write_sht as before.  Every device-bound step
    (build_master, write_sht) auto-resolves to CUDA-if-available-else-CPU, so the
    whole pipeline runs on a GPU OR a CPU-only machine with no env var required.

    Args:
        xtal_path: Absolute path to an EMsoft ``.xtal`` crystal file.
        params: Dict with simulation parameters.  Recognised keys mirror
            ``SimulationStartRequest``:
            ``ekev``, ``sig``, ``omega``, ``totnum_el``, ``depthmax``,
            ``depthstep``, ``npx``, ``dmin``, ``output_type``,
            ``bandwidth`` (for .sht, default 384).
        progress_cb: Called as ``progress_cb(percent, message)`` at each stage.
        log_cb: Called as ``log_cb(line)`` for informational log lines.

    Returns:
        Dict with keys ``"mc"``, ``"master"`` and ``"sht"``, each either an
        absolute path string (if that artifact was produced) or ``None``.  The
        ``"mc"`` MC ``.h5`` is always produced; ``"master"`` / ``"sht"`` depend
        on ``output_type``.

    Raises:
        ValueError: on bad parameters (e.g. ekev <= Ehistmin=10, or an
            unsupported ``output_type``).  ALL crystal systems are supported:
            a cubic cell uses the scalar Cartesian->Miller map, a non-cubic cell
            routes through the general direct-structure-matrix map.  Only a CUBIC
            cell keeps the writer's NH copy-mirror; EVERY non-cubic cell (centro
            or not) builds BOTH hemispheres directly (the identity copy-mirror is
            wrong for low-symmetry centro cells — see the build_sh comment / L3).
        FileNotFoundError: when ``xtal_path`` does not exist.
    """
    from backend.forward_sim.runtime import resolve_device_adaptive  # noqa: PLC0415
    from backend.forward_sim.crystal.xtal_io import read_crystal_structure  # noqa: PLC0415
    from backend.forward_sim.mc.composition import mc_composition_from_structure  # noqa: PLC0415
    from backend.forward_sim.mc.gpu_monte_carlo import (  # noqa: PLC0415
        run_gpu_mc,
        MCConfig,
        cupy_mc_available,
    )
    from backend.forward_sim.mc.emsoft_mc_input import load_mc  # noqa: PLC0415
    from backend.forward_sim.dynamical.master_builder import build_master, _is_cubic  # noqa: PLC0415
    from backend.forward_sim.io.master_h5 import write_master_h5  # noqa: PLC0415
    from backend.forward_sim.io.mc_h5 import write_mc_h5  # noqa: PLC0415
    from backend.forward_sim.io.sht_writer import write_sht  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    xtal = Path(xtal_path)
    if not xtal.exists():
        raise FileNotFoundError(f"xtal file not found: {xtal_path}")

    # --- parameters ---
    ekev: float = float(params.get("ekev", 20.0))
    sig: float = float(params.get("sig", 70.0))
    omega: float = float(params.get("omega", 0.0))
    totnum_el: int = int(params.get("totnum_el", GPU_MC_DEFAULT_ELECTRONS))
    depthmax: float = float(params.get("depthmax", 100.0))
    depthstep: float = float(params.get("depthstep", 1.0))
    npx: int = int(params.get("npx", 500))
    dmin: float = float(params.get("dmin", 0.05))
    output_type: str = str(params.get("output_type", "sht_only"))
    bandwidth: int = int(params.get("bandwidth", 384))

    do_master = output_type in ("master_only", "both")
    do_sht = output_type in ("sht_only", "both")
    if not do_master and not do_sht:
        raise ValueError(
            f"output_type must be one of sht_only / master_only / both, got {output_type!r}"
        )

    # --- EKEV GUARD: fail-loud BEFORE resolving the device, so a low-kV request
    # gives the actionable ekev message even on a machine with no CUDA (where
    # get_device() would otherwise raise ForwardSimError first and mask it). ---
    ehistmin = _EHISTMIN
    if ekev <= ehistmin:
        raise ValueError(
            f"ekev={ekev} kV is <= Ehistmin={ehistmin} kV: the energy histogram "
            f"would have zero counts (lambda_z all-zero), causing a divide-by-zero "
            f"in the master builder.  Use ekev > {ehistmin} kV (the frontend allows "
            f"down to 5 kV which silently breaks the pipeline at this check)."
        )

    # --- resolve device (HARDWARE-ADAPTIVE: CUDA-if-available-else-CPU) ---
    # The "Ours" engine must run on a GPU OR a CPU-only machine with NO env var
    # required.  Unlike get_device() (which fail-loud raises ForwardSimError on a
    # CPU-only host to protect the GPU-native indexing paths),
    # resolve_device_adaptive honours FORWARD_SIM_DEVICE, else uses cuda when
    # available, else gracefully falls back to cpu.  This same device flows into
    # build_master and write_sht below, so every step auto-picks its hardware.
    device = resolve_device_adaptive()
    log_cb(f"[gpu_sim] device: {device}")

    stem = xtal.stem
    safe = sanitize_filename(stem)

    # --- Stage 1: read crystal ---
    progress_cb(5.0, "Reading crystal structure…")
    log_cb(f"[gpu_sim] reading crystal: {xtal_path}")
    structure = read_crystal_structure(xtal_path)
    log_cb(
        f"[gpu_sim] crystal: SG={structure.space_group} "
        f"a={structure.lattice[0]:.4f} nm  atoms={len(structure.atoms)}"
    )

    # --- CRYSTAL-SYSTEM ROUTING (SP6) ---
    # The GPU master builder now supports ALL crystal systems: a cubic cell uses
    # the bit-identical scalar Cartesian→Miller map; a non-cubic cell routes
    # through the general direct-structure-matrix map.  Only a CUBIC cell may keep
    # the writer's identity copy-mirror (mLPSH = mLPNH); every NON-cubic cell needs
    # its true southern hemisphere computed directly — even a centrosymmetric one.
    # (Verified: monoclinic-centro Al13Fe4 has NH that is NOT flip-invariant, so the
    # identity copy gives SH NCC 0.76 vs the correct 1.00 from a direct south build;
    # see tasks/forward_sim/iterations/SIMFIX/backend_report.md L3.)
    structure_is_cubic = _is_cubic(structure)
    if not structure_is_cubic:
        log_cb(
            f"[gpu_sim] non-cubic crystal (lattice {structure.lattice}): using the "
            f"general dsm-based direction map + explicit southern-hemisphere build"
        )

    # --- Stage 2: MC Source Strategy ---
    # Determine canonical MC output path (used for both read and write)
    ebinsize = _EBINSIZE
    nE = max(1, int(round((ekev - ehistmin) / ebinsize)) + 1)
    numsx_mc = 501  # EMsoft default; also used for scan_missing glob

    # Shared kV label (round) — MUST agree with scan_missing's glob and the
    # EMsoft naming so "Simulate All Missing" detects this file (M1 fix).
    kv_int = ekev_to_kv_label(ekev)
    expected_mc_path = mc_h5_path(stem, kv_int, sig, numsx_mc)

    mc_data = None
    mc_source_used: str = "unknown"
    # The ACTUAL electron count that produced the master (for honest provenance):
    # set to the capped n_sim when our MC runs, or read back from a reused MC .h5.
    # Falls back to the requested count only as a last resort (cache .h5 unreadable).
    electrons_actual: int = int(totnum_el)

    # ------------------------------------------------------------------
    # Shared helper: run OUR MC via run_gpu_mc(engine="auto").  This single call
    # is the WHOLE EMsoft-free hardware-adaptive MC: engine="auto" fans out to
    # cupy (CUDA) -> numba (CPU) -> pytorch (CPU) internally.  The runner only
    # picks the surrounding message; there is NO EMsoft branch.
    # ------------------------------------------------------------------
    def _run_our_mc(stage_label: str, progress_msg: str) -> Any:
        nonlocal electrons_actual
        n_sim: int = min(max(1, totnum_el), GPU_MC_MAX_ELECTRONS)
        electrons_actual = n_sim
        if n_sim < totnum_el:
            log_cb(
                f"[gpu_sim] requested {totnum_el:,} electrons — capping to {n_sim:,} "
                f"(the master uses the normalised depth/energy profile, which converges "
                f"well before this; far fewer than EMsoft's 500M are needed). The GPU "
                f"speedup is the master step, not the MC."
            )
        comp = mc_composition_from_structure(structure)
        log_cb(
            f"[gpu_sim] {stage_label} composition: Z={comp.mean_Z:.3f} "
            f"A={comp.mean_A:.3f} rho={comp.rho:.4f} g/cm³"
        )
        mc_cfg = MCConfig(
            starting_E_keV=ekev,
            Ehistmin=ehistmin,
            Ebinsize=ebinsize,
            nE=nE,
            depth_step=depthstep,
            depth_max=depthmax,
            sig_deg=sig,
            omega_deg=omega,
            n_dir=numsx_mc,
            n_dir_z=51,
            n_simulations=n_sim,
            n_electrons_parallel=100_000,
            n_max_steps=300,
            device=str(device),
            # engine="auto" (default): the one-thread-per-electron CuPy RawKernel
            # MC on a CUDA device when cupy is available (~400x faster than the
            # PyTorch loop, master-NCC-equivalent 0.9999), else the numba CPU loop,
            # else the validated PyTorch step-major loop.  See LEVER 1 / MC-1.
        )
        progress_cb(15.0, progress_msg)
        _t = time.perf_counter()
        _mc = run_gpu_mc(comp.mean_Z, comp.mean_A, comp.rho, mc_cfg)
        _wall = time.perf_counter() - _t
        log_cb(
            f"[gpu_sim] {stage_label} done in {_wall:.1f}s  "
            f"EkeVs={_mc.EkeVs.tolist()}  "
            f"eta={float(_mc.accum_e.sum().item()) / n_sim:.4f}"
        )
        return _mc

    # Strategy (a): existing MC .h5 (instant cache hit; keep first).  The cached
    # .h5 may have been written by a prior "Ours" run OR by the separate EMsoft
    # engine — either way it is a valid MC source we just load, no re-run.
    if expected_mc_path.exists():
        progress_cb(10.0, "MC: loading existing MC .h5…")
        log_cb(f"[gpu_sim] MC source (a): loading existing {expected_mc_path}")
        try:
            mc_data = load_mc(str(expected_mc_path))
            mc_source_used = "existing_cache"
            # Recover the real electron count embedded in the reused MC .h5.
            try:
                import h5py as _h5  # noqa: PLC0415
                with _h5.File(str(expected_mc_path), "r") as _f:
                    electrons_actual = int(_f["EMData/MCOpenCL/totnum_el"][()][0])
            except Exception:
                pass  # keep the requested-count default
            # NB: no eta here — for an existing .h5 we did not run the MC, so the
            # incident-electron total is unknown (accum_e is the BSE count only;
            # eta = BSE/incident is not derivable).  The old "sum/sum" was ≈1.0.
            log_cb(
                f"[gpu_sim] MC loaded from existing .h5: "
                f"EkeVs={mc_data.EkeVs.tolist()}  "
                f"bse_counts={int(mc_data.accum_e.sum().item()):,}"
            )
            progress_cb(38.0, "MC: loaded from existing .h5")
        except Exception as e:
            log_cb(f"[gpu_sim] Failed to load existing MC .h5 ({e}); falling through to re-run")
            mc_data = None

    # Strategy (b/c/d): OUR EMsoft-free MC.  ONE call to run_gpu_mc(engine="auto")
    # does the hardware fan-out internally: cupy CUDA kernel -> numba CPU loop ->
    # PyTorch CPU loop.  EMsoft EMMCOpenCL / WSL is intentionally NOT in this
    # chain — the "Ours" engine must be self-contained and never require WSL.
    #
    # ``device`` is hardware-adaptive (cuda-if-available-else-cpu); when it is NOT
    # a CUDA device with a working cupy MC kernel, the run will use the slow CPU
    # paths, so we emit a LOUD warning up front (per fail-loud-on-downgrade).
    if mc_data is None:
        _is_cuda = str(device).startswith("cuda")
        _gpu_mc_ready = _is_cuda and cupy_mc_available()
        if _gpu_mc_ready:
            log_cb(
                f"[gpu_sim] MC source (b): OUR GPU MC (cupy CUDA kernel)  "
                f"ekev={ekev} sig={sig} n_el={totnum_el:,}"
            )
            mc_data = _run_our_mc("GPU MC", "Monte Carlo (GPU, EMsoft-free): running…")
            mc_source_used = "gpu_native"
            progress_cb(38.0, "Monte Carlo (GPU): done")
        else:
            # No CUDA+cupy MC kernel: run_gpu_mc(engine="auto") will use numba
            # (CPU fast path) when it compiles, else the PyTorch loop.  Both are
            # far slower than the cupy kernel — warn loudly, but DO NOT touch
            # EMsoft (the "Ours" engine is EMsoft-free by design).
            n_sim_d: int = min(max(1, totnum_el), GPU_MC_MAX_ELECTRONS)
            if not _is_cuda:
                _cause_phrase = "no CUDA device — CPU MC path"
            else:
                _cause_phrase = "cupy MC kernel unavailable — CPU MC path"
            log_cb(
                f"WARNING: {_cause_phrase} (EMsoft-free): using OUR numba/PyTorch "
                f"CPU MC (the PyTorch loop is ~400x slower than the cupy kernel), "
                f"capped {n_sim_d:,} electrons. For fast MC use a CUDA GPU with cupy."
            )
            logger.warning(
                "[gpu_sim] %s -> OUR CPU MC (numba/pytorch, ~400x slower), capped %d electrons",
                _cause_phrase,
                n_sim_d,
            )
            mc_data = _run_our_mc("CPU MC", "Monte Carlo (CPU, EMsoft-free): running…")
            mc_source_used = "cpu_fallback"
            progress_cb(38.0, "Monte Carlo (CPU): done")

    assert mc_data is not None, "BUG: mc_data is None after all MC strategies"

    # energy index = top bin (highest kV = ekev)
    energy_idx: int = mc_data.n_energy - 1
    energy_kV_actual: float = float(mc_data.EkeVs[energy_idx].item())
    log_cb(f"[gpu_sim] mc_source={mc_source_used}  energy_idx={energy_idx}  energy_kV={energy_kV_actual:.1f}")

    # --- Stage 2b: ALWAYS write the MC .h5 ---
    # For a cache hit (existing .h5) the file is already at expected_mc_path, so we
    # just record it.  For our own freshly-run MC (gpu_native / cpu_fallback) the
    # result is in-memory accumulators that we must serialise so scan_missing sees
    # has_mc.
    kv = ekev_to_kv_label(energy_kV_actual)
    result: Dict[str, Optional[str]] = {"master": None, "sht": None, "mc": None}

    if mc_source_used == "existing_cache":
        # MC .h5 already on disk at the canonical path — just record it
        result["mc"] = str(expected_mc_path)
        log_cb(f"[gpu_sim] MC .h5 already at {expected_mc_path}")
    else:
        # Our own MC (gpu_native / cpu_fallback): write the accumulators to disk
        # so scan_missing sees "has_mc".
        mc_path = mc_h5_path(stem, kv, sig, numsx_mc)
        mc_path.parent.mkdir(parents=True, exist_ok=True)
        progress_cb(40.0, "Writing MC .h5…")
        log_cb(f"[gpu_sim] writing MC .h5: {mc_path}")
        t_mc_h5 = time.perf_counter()
        write_mc_h5(
            str(mc_path),
            mc_data,
            structure,
            sig=sig,
            omega=omega,
            numsx=numsx_mc,
            totnum_el=int(params.get("totnum_el", GPU_MC_DEFAULT_ELECTRONS)),
            Ehistmin=ehistmin,
            Ebinsize=ebinsize,
            EkeV=energy_kV_actual,
            xtalname=xtal.name,
        )
        log_cb(f"[gpu_sim] MC .h5 written in {time.perf_counter() - t_mc_h5:.1f}s")
        result["mc"] = str(mc_path)

    # --- Stage 3: build master pattern ---
    progress_cb(45.0, "Building master pattern…")
    log_cb(f"[gpu_sim] build_master: npx={npx} dmin={dmin}")
    t_build = time.perf_counter()

    # A non-cubic crystal needs its southern hemisphere computed DIRECTLY (NH ≠ SH
    # in general): build both.  Only a CUBIC cell keeps the writer's identity
    # copy-mirror (mLPSH = mLPNH) in write_master_h5, so only NH is built (cubic
    # path unchanged — bit-identical to before).
    #
    # NB: the gate is `not cubic`, NOT `not centrosymmetric`.  The writer's
    # copy-mirror is an IDENTITY copy (mLPSH = mLPNH), which is correct only when
    # NH is invariant under the both-axis Lambert flip — true for cubic, but NOT
    # for low-symmetry centrosymmetric cells.  Measured on monoclinic-centro
    # Al13Fe4: the identity copy yields SH NCC 0.76 vs 1.00 for a direct south
    # build (build_master's own centro short-circuit returns NH.flip, not NH).
    # Relaxing this to `not is_centrosymmetric` would corrupt the SH of every
    # non-cubic centro cell.  (L3 of the SIMFIX pass.)
    build_sh = not structure_is_cubic

    # Map the master build's per-direction progress into the 45-90% band with an
    # elapsed-time message (UI hook only — does NOT change build_master numerics).
    # For a two-hemisphere (non-cubic) build the NH gets the first half of the
    # band and the SH the second half.
    _MASTER_LO, _MASTER_HI = 45.0, 90.0
    _NH_HI = (_MASTER_LO + _MASTER_HI) / 2.0 if build_sh else _MASTER_HI

    def _make_progress(lo: float, hi: float, label: str):
        def _cb(done: int, total: int) -> None:
            if total <= 0:
                return
            frac = max(0.0, min(1.0, done / total))
            pct = lo + (hi - lo) * frac
            elapsed = time.perf_counter() - t_build
            progress_cb(
                pct,
                f"{label}… {done:,}/{total:,} dirs ({frac * 100:.0f}%, {elapsed:.0f}s)",
            )
        return _cb

    master_pattern = build_master(
        structure,
        mc_data,
        npx=npx,
        energy_idx=energy_idx,
        dmin=dmin,
        device=device,
        dir_chunk="auto",   # adaptive: VRAM-safe for large cells (iter-16 OOM guard)
        hemisphere="north",
        progress_cb=_make_progress(_MASTER_LO, _NH_HI, "Building master (NH)"),
    )
    master_pattern_sh = None
    if build_sh:
        log_cb("[gpu_sim] build_master: southern hemisphere (non-cubic, NH != SH)")
        master_pattern_sh = build_master(
            structure,
            mc_data,
            npx=npx,
            energy_idx=energy_idx,
            dmin=dmin,
            device=device,
            dir_chunk="auto",
            hemisphere="south",
            progress_cb=_make_progress(_NH_HI, _MASTER_HI, "Building master (SH)"),
        )
    build_wall = time.perf_counter() - t_build
    log_cb(
        f"[gpu_sim] master built in {build_wall:.1f}s  "
        f"shape={tuple(master_pattern.shape)}  "
        f"finite={bool(master_pattern.isfinite().all().item())}"
        + (
            f"  SH finite={bool(master_pattern_sh.isfinite().all().item())}"
            if master_pattern_sh is not None
            else ""
        )
    )
    progress_cb(90.0, "Master pattern: done")

    # --- Stage 4: write output files ---
    ekevs_np = mc_data.EkeVs.detach().cpu().numpy().astype("float32")

    if do_master:
        progress_cb(91.0, "Writing master.h5…")
        master_path = master_h5_path(stem, ekev_to_kv_label(energy_kV_actual), npx)
        master_path.parent.mkdir(parents=True, exist_ok=True)
        log_cb(f"[gpu_sim] writing master.h5: {master_path}")
        t_h5 = time.perf_counter()
        write_master_h5(
            str(master_path),
            master_pattern,
            structure,
            mc_source_h5=None,
            npx=npx,
            dmin=dmin,
            energy_kV=energy_kV_actual,
            EkeVs=ekevs_np,
            Ebinsize=float(ebinsize),
            xtalname=xtal.name,
            our_master_SH=master_pattern_sh,  # explicit true SH for non-cubic; None=copy-mirror (cubic)
        )
        log_cb(f"[gpu_sim] master.h5 written in {time.perf_counter() - t_h5:.1f}s")
        result["master"] = str(master_path)
        progress_cb(94.0, "master.h5 written")

    if do_sht:
        progress_cb(94.0, "Writing .sht…")
        # Convention-compliant name; stem stays in (parens) for scan-missing.
        # Formula/Pearson are READ from the linked CIF (or the .xtal) by the
        # Task-1 helpers — not synthesised from atom types here.
        sht_dir = _SHT_DB / safe
        sht_dir.mkdir(parents=True, exist_ok=True)
        sht_fname = _sht_filename_for(stem, xtal, energy_kV_actual, sig)
        sht_path = sht_dir / sht_fname
        log_cb(f"[gpu_sim] writing .sht: {sht_path}")
        t_sht = time.perf_counter()
        # Bake the source crystal's literature reference into the .sht binary so
        # the citation survives even if the sidecar + .xtal are later lost.
        _sht_reference = ""
        try:
            from backend.api.services.sht_provenance import read_xtal_reference  # noqa: PLC0415
            _sht_reference = read_xtal_reference(xtal) or ""
        except Exception:
            _sht_reference = ""
        write_sht(
            str(sht_path),
            master_pattern,
            structure,
            # Genuine southern hemisphere for a non-cubic build (NH != SH) → the
            # SHT encodes the full sphere (EMsoft analyze(NH, SH)); None for a
            # cubic/equator-symmetric cell uses the writer's NH copy-mirror
            # (mLPSH = mLPNH).  (NB: the MCSHT reorder changed the default MC
            # *source* to our cupy GPU MC, so the .sht is no longer byte-identical
            # to an EMsoft-MC build — but it is master-NCC-equivalent, 0.9999.)
            our_master_SH=master_pattern_sh,
            bandwidth=bandwidth,
            energy_kV=energy_kV_actual,
            npx=npx,
            dmin=dmin,
            primary_tilt_deg=sig,
            secondary_tilt_deg=omega,
            sg_eff=int(structure.space_group),
            note=f"source_xtal={xtal.name}",
            doi=_sht_reference,
            totnum_el=int(electrons_actual),
        )
        log_cb(f"[gpu_sim] .sht written in {time.perf_counter() - t_sht:.1f}s")
        result["sht"] = str(sht_path)
        # Provenance sidecar: record the source xtal/cif + sim params next to the
        # .sht.  best-effort — a sidecar failure must never fail the simulation.
        # NB: ``bethe`` is omitted (no Bethe-parameters variable is in scope in
        # this GPU path).  ``electrons`` is the ACTUAL simulated count (capped or
        # read back from a reused MC .h5), and ``engine`` marks this as an "Ours"
        # (EMsoft-free) master so provenance can tell the two engines apart.
        try:
            write_provenance_sidecar(
                sht_path, xtal_path=xtal, cif_dir=_cif_library_dir(),
                params={"dmin": dmin, "npx": npx, "voltage_kV": energy_kV_actual,
                        "sig": sig, "omega": omega,
                        "electrons": int(electrons_actual), "engine": "ours",
                        "bandwidth": bandwidth},
            )
        except Exception as exc:  # provenance is best-effort; never fail the sim
            log_cb(f"[gpu_sim] provenance sidecar skipped: {exc}")
        progress_cb(98.0, ".sht written")

    progress_cb(100.0, "Done")
    log_cb(f"[gpu_sim] finished: {result}")
    return result
