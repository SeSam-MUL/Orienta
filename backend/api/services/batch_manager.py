"""Batch Manager — orchestrates batch indexing with memory safety and checkpointing.

Key improvement over v1: Each file is fully loaded → preprocessed → indexed → unloaded.
No more assumption that a signal is pre-loaded.
"""
import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.api.services.batch_queue import BatchQueue
from backend.api.services.checkpoint_writer import CheckpointWriter
from backend.api.services.memory_guardian import MemoryGuardian, MemoryStatus
from backend.api.services.preflight_check import PreFlightChecker, PreFlightReport

logger = logging.getLogger(__name__)


def _load_signal(file_path: str):
    """Load an EBSD signal from file. Returns the kikuchipy EBSD signal."""
    from safe_loader import load_ebsd_safe
    logger.info("Loading signal: %s", Path(file_path).stem)
    return load_ebsd_safe(file_path)


def _extract_step_size_um(signal, file_path: str) -> float:
    """Best-effort µm-step extraction for the checkpoint.

    Order: signal.axes_manager.navigation_axes[0].scale (preferred —
    kikuchipy puts it there) → h5oina header X Step (fallback for files
    where the loader didn't populate the axes manager) → 0.0 sentinel.

    Caller is responsible for refusing to proceed on 0.0; we don't raise
    here because some test fixtures legitimately load synthetic signals
    with no axes scale and a 0.0 must be detectable rather than fatal.
    """
    try:
        nav = signal.axes_manager.navigation_axes
        if nav:
            scale = float(nav[0].scale)
            # kikuchipy fills 1.0 when the source has no scale; treat that as
            # "unknown" so we can still try the h5oina header below.
            if scale > 0 and scale != 1.0:
                return scale
    except Exception:
        pass
    try:
        import h5py
        with h5py.File(file_path, "r") as f:
            for entry in f.keys():
                hdr = f.get(f"{entry}/EBSD/Header")
                if hdr is None:
                    continue
                for key in ("X Step", "Step X", "StepX", "x_step"):
                    if key in hdr:
                        v = float(hdr[key][()])
                        if v > 0:
                            return v
    except Exception:
        pass
    # Last resort: if axes had scale=1.0 and h5oina header was absent, we
    # honour 1.0 (better than 0). The caller will warn.
    try:
        nav = signal.axes_manager.navigation_axes
        if nav and float(nav[0].scale) > 0:
            return float(nav[0].scale)
    except Exception:
        pass
    return 0.0


def _apply_preprocessing(signal, config: Dict):
    """Apply preprocessing steps to the signal in-place.

    Returns
    -------
    tuple (signal, applied)
        ``applied`` is a dict of keys → True/False/error-string, describing
        which steps actually ran. The caller persists this into the
        checkpoint so downstream consumers (report UI, refinement) can
        confirm what was done.

    config : dict with preprocessing options:
        frame_averaging: bool
        frame_averaging_window: int (3, 5, 7)
        background_removal: bool
        background_method: 'dynamic' | 'static'
        static_bg_row: int
        static_bg_col: int
        gauss_background: bool (for spherical)
        nregions_ahe: int (for spherical AHE)
    """
    applied: Dict[str, Any] = {}
    if not config:
        return signal, applied

    # Frame averaging
    if config.get("frame_averaging", False):
        window = config.get("frame_averaging_window", 3)
        try:
            try:
                signal.average_neighbour_patterns(
                    window="rectangular", window_shape=(window, window)
                )
            except AttributeError:
                signal.average_neighbor_patterns(
                    window="rectangular", window_shape=(window, window)
                )
            logger.info("Applied frame averaging (window=%d)", window)
            applied["frame_averaging"] = f"{window}x{window}"
        except Exception as e:
            logger.warning("Frame averaging failed: %s", e)
            applied["frame_averaging"] = f"FAILED: {e}"

    # Background removal
    if config.get("background_removal", False):
        method = config.get("background_method", "dynamic")
        try:
            if method == "dynamic":
                # filter_domain="spatial" uses a Gaussian kernel via scipy
                # instead of dask-backed FFT. Empirically the frequency path
                # has been leaking dask workers across successive batch files
                # on Windows and the uvicorn process dies silently partway
                # through file 2. Spatial is slightly slower per pattern but
                # stable across a multi-file batch.
                signal.remove_dynamic_background(
                    operation="subtract", filter_domain="spatial"
                )
                logger.info("Applied dynamic background removal (spatial)")
                applied["background_removal"] = "dynamic-spatial"
            elif method == "static":
                row = config.get("static_bg_row", 0)
                col = config.get("static_bg_col", 0)
                try:
                    bg = signal.inav[col, row].data.squeeze()
                except Exception:
                    bg = signal.data.reshape(-1, *signal.axes_manager.signal_shape).mean(axis=0)
                signal.remove_static_background(operation="subtract", static_bg=bg)
                logger.info("Applied static background removal (ref: %d,%d)", row, col)
                applied["background_removal"] = f"static@{row},{col}"
        except Exception as e:
            logger.warning("Background removal failed: %s", e)
            applied["background_removal"] = f"FAILED: {e}"

    return signal, applied


def _register_signal_calibration(signal, file_path: str):
    """Register signal's detector/PC in CalibrationStore."""
    from backend.api.services.calibration_store import calibration_store
    dataset_name = Path(file_path).stem
    if dataset_name not in calibration_store:
        calibration_store.register(dataset_name, signal)


#: The spherical engines a batch request may ask for. Re-exported from
#: indexing_controller rather than restated: the interactive route, the
#: per-phase dispatcher and this one all have to agree on what the names are,
#: and two lists of two strings are exactly the kind of thing that drifts.
from indexing_controller import SPHERICAL_BACKENDS as _SPHERICAL_BACKENDS


def _get_detector_for_file(file_path: str, signal=None):
    """Get the best available detector for a file. Never returns None."""
    from backend.api.services.calibration_store import calibration_store
    dataset_name = Path(file_path).stem
    detector = calibration_store.get_detector(dataset_name)
    if detector is None and signal is not None:
        detector = getattr(signal, "detector", None)
    if detector is None:
        # Last resort: a detector nobody measured. PC (0.5, 0.5, 0.5), sample
        # tilt 70, camera elevation 0 — kikuchipy's defaults, not this scan's
        # geometry. Indexing against it produces confident-looking orientations
        # that are simply wrong, so say so once, loudly, here at the only place
        # that can know it happened. (Callers cannot: this function never hands
        # back None, so a `detector is None` guard downstream is unreachable.)
        from kikuchipy.detectors import EBSDDetector
        sig_shape = signal.axes_manager.signal_shape[::-1] if signal else (60, 60)
        detector = EBSDDetector(shape=sig_shape)
        logger.warning(
            "No calibration for %r and the signal carries no detector — "
            "falling back to a FABRICATED geometry (PC 0.5/0.5/0.5, sample tilt "
            "%.1f deg, camera elevation %.1f deg). Orientations from this run "
            "are not trustworthy; set the PC in PC Refinement first.",
            dataset_name, float(detector.sample_tilt), float(detector.tilt),
        )
    return detector


def run_single_indexing_job(
    signal,
    file_path: str,
    phase_name: str,
    phase_path: str,
    method: str,
    grid_shape: Tuple[int, int],
    job_config: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Run indexing for one signal x one phase.

    Parameters
    ----------
    signal : kikuchipy EBSD signal (already loaded and preprocessed)
    file_path : str
    phase_name, phase_path, method : str
    grid_shape : (n_rows, n_cols)
    job_config : dict with method-specific parameters (keep_n, bandwidth, etc.)
    progress_callback : optional

    Returns
    -------
    (ci_map, orientation_map, metadata)
    """
    from indexing_controller import (
        IndexingConfig, IndexingMethod,
        hough_index_patterns, dictionary_index_patterns, spherical_index_patterns,
        spherical_gpu_index_patterns,
    )

    detector = _get_detector_for_file(file_path, signal)
    config_opts = job_config or {}
    start_time = time.time()

    if method == "spherical":
        # Spherical has its own in-EMSphinx background removal (gausbckg)
        # that's separate from kikuchipy preprocessing. The batch used to
        # leave it at default False, so batch-spherical runs produced very
        # noisy orientations while single-file runs (where the user toggles
        # it on) came out clean. Map the batch's preprocessing.background_removal
        # + a dedicated spherical_* config onto the IndexingConfig.
        preproc = (config_opts or {}).get("preprocessing", {})
        sph_gausbckg = bool(
            config_opts.get("spherical_gausbckg",
                preproc.get("background_removal", True))
        )
        sph_bandwidth = int(config_opts.get("spherical_bandwidth", 88))
        sph_nregions  = int(config_opts.get("spherical_nregions", 10))
        sph_refine    = bool(config_opts.get("spherical_refine", True))
        sph_normed    = bool(config_opts.get("spherical_normed", True))
        sph_circmask  = int(config_opts.get("spherical_circmask", -1))
        # The engine choice belongs on the config, like everything else the
        # request sets, so that ONE field decides it — the same rule the
        # interactive route follows (IndexingStartRequest.backend ->
        # config.backend -> dispatch). The default stays "emsphinx", matching
        # IndexingConfig and IndexingStartRequest: the GPU path needs CUDA, and
        # repairing a selector must not also change what a batch that asks for
        # nothing does.
        # Not str(): a JSON null would become the string "None" and the
        # rejection below would read like a typo the caller never made.
        sph_backend   = config_opts.get("backend", "emsphinx")
        config = IndexingConfig(
            method=IndexingMethod.SPHERICAL,
            sht_file=phase_path,
            bandwidth=sph_bandwidth,
            normed=sph_normed,
            refine=sph_refine,
            nregions=sph_nregions,
            circmask=sph_circmask,
            gausbckg=sph_gausbckg,
            backend=sph_backend,
        )
        logger.info(
            "Spherical config: gausbckg=%s, bandwidth=%d, nregions=%d, refine=%s",
            sph_gausbckg, sph_bandwidth, sph_nregions, sph_refine,
        )
        # The detector dict comes from the SAME derivation the interactive
        # route uses — `build_spherical_det_params` — not from a second copy.
        # The copy that used to stand here carried a comment saying parity was
        # critical and had drifted anyway; the divergence that mattered was a
        # missing `sample_tilt` key. `DetectorGeometry.from_params` reads
        # `params.get("sample_tilt")`, so its absence became None and
        # `Tier1Indexer` fell back to `master.primary_tilt_deg` — the master
        # pattern's Monte-Carlo tilt, usually 70 deg (read off Al and Ni; not
        # checked across the whole library) — not the tilt the scan was taken
        # at. On LoGainNi (75.7 deg) that is a fixed 5.7 deg rotation, silent
        # because 70 is a legal tilt. Same defect as 2026-05-22, which was only
        # ever fixed on the interactive side.
        #
        # It was LATENT, never live: the selector below always chose EMSphInx
        # (see the comment there), and the EMSphInx path has no sample-tilt
        # field in its NML and never reads the key. The 5.7 deg is what the
        # batch would have begun paying the moment the selector was repaired
        # (measured 4.998 deg median on a 100 px LoGainNi ROI, L=88).
        #
        # Sharing the derivation also brings the map-mean PC (the batch read
        # pixel (0,0) off a per-pixel PC map), the vendor detection that can
        # answer "edax" (the batch probe could only answer "oxford"), and the
        # EMSphinx [5, 90] mm detector-width guard.
        #
        # `pixel_size` resolves differently than the batch's old
        # `_get_pixel_size` (70 vs 55 um on SampleB) and that does not move the
        # geometry: `convert_pc_to_emsoft` returns xpc/ypc in detector pixels
        # and L proportional to the pixel size, so L/pixel_size is invariant
        # (108.267 either way, measured 2026-09-12). The number only feeds the
        # width guard.
        #
        # Imported inside the function: `routes.indexing` is heavy and imports
        # this module's siblings, and the batch already does its imports lazily.
        # The shared derivation reads the signal's axes (scan shape, step sizes),
        # so unlike the old inline dict it cannot work without one. Say which
        # argument is missing rather than let it fall over inside a route module
        # with an AttributeError on `signal.axes_manager`.
        if signal is None:
            raise ValueError(
                "Spherical indexing needs the loaded EBSD signal to read the scan "
                "geometry (shape and step sizes); run_single_indexing_job was "
                f"called with signal=None for {Path(file_path).name!r}"
            )
        from backend.api.routes.indexing import build_spherical_det_params

        det_params = build_spherical_det_params(signal, detector, file_path)
        # One field decides, read the same way as backend/api/routes/indexing.py.
        # It used to read `getattr(config, "backend", config_opts.get(...))`,
        # and `backend` is a dataclass field with a default, so it is always
        # present and the three-argument getattr never reached its fallback:
        # the request's choice was dead code and every batch spherical job went
        # to EMSphInx. That is why the missing `sample_tilt` above never bit —
        # the EMSphInx path does not read it.
        # job_config is free-form JSON straight off the request, so a typo
        # ("gpu", "spherical-gpu", "emsphnix") would otherwise run the OTHER
        # engine without a word and the user would read the difference as
        # data. The check is indexing_controller.resolve_spherical_backend --
        # the same one the interactive route and the per-phase dispatcher use,
        # so a value one of them refuses cannot be accepted by another. A
        # missing key still defaults to "emsphinx" (see the config build
        # above); a JSON null still reports as "null/missing".
        from indexing_controller import resolve_spherical_backend
        backend_choice = resolve_spherical_backend(config)
        if backend_choice == "spherical_gpu":
            result = spherical_gpu_index_patterns(
                h5_path=file_path,
                config=config,
                detector_params=det_params,
                progress_callback=progress_callback,
            )
        else:
            result = spherical_index_patterns(
                h5_path=file_path,
                config=config,
                detector_params=det_params,
                progress_callback=progress_callback,
            )
    elif method == "dictionary":
        import kikuchipy as kp
        dictionary = kp.load(phase_path)
        keep_n = config_opts.get("keep_n", 20)
        config = IndexingConfig(method=IndexingMethod.DICTIONARY, keep_n=keep_n)
        result = dictionary_index_patterns(
            signal=signal,
            dictionary=dictionary,
            config=config,
            progress_callback=progress_callback,
        )
        del dictionary
    elif method == "hough":
        # Hough needs a phase_list (built from CIF) and a detector.
        # BUG-L (2026-04-21): previous code called hough_index_patterns without
        # these two required positional arguments, so Hough in the batch manager
        # always raised TypeError and the job failed.
        from orix.crystal_map import PhaseList
        from orix.crystal_map import Phase as _Phase
        from ebsd_utils import sanitize_cif
        from pathlib import Path as _P

        n_bands = config_opts.get("n_bands", 12)
        t_sigma = config_opts.get("t_sigma", 2.0)
        r_sigma = config_opts.get("r_sigma", 2.0)
        config = IndexingConfig(
            method=IndexingMethod.HOUGH,
            n_bands=n_bands,
            t_sigma=t_sigma,
            r_sigma=r_sigma,
        )

        phase = _Phase.from_cif(sanitize_cif(phase_path))
        original_stem = _P(phase_path).stem
        if phase.name != original_stem:
            phase.name = original_stem
        phase_list = PhaseList([phase])

        # No `detector is None` guard: _get_detector_for_file never returns
        # None — it builds a default EBSDDetector and warns that the geometry
        # is fabricated. See its docstring.
        result = hough_index_patterns(
            signal=signal,
            phase_list=phase_list,
            detector=detector,
            config=config,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"Unsupported method for batch: {method}")

    duration = time.time() - start_time

    # Extract CI map and orientations from result.
    # BUG-M (2026-04-21): pyebsdindex sometimes returns 2D cm with a
    # solutions-or-phases axis, which survives ravel() as n_rows×n_pixels.
    # pyebsdindex stores cm as shape (n_phases+1, n_pixels) row-major with
    # the CONSENSUS best-per-pixel row at indxData[-1], so the defensive
    # fallback extracts the LAST n_pixels entries (trailing slice), not
    # the first — otherwise multi-phase batches would silently use phase-0
    # CI instead of the consensus best.
    n_pixels = int(np.prod(grid_shape))
    ci_map = np.zeros(grid_shape, dtype=np.float32)
    if result.confidence_scores is not None:
        scores = np.asarray(result.confidence_scores).ravel()
        if scores.size == n_pixels:
            ci_map = scores.reshape(grid_shape).astype(np.float32)
        elif scores.size > n_pixels and scores.size % n_pixels == 0:
            ci_map = scores[-n_pixels:].reshape(grid_shape).astype(np.float32)
        # else: keep zeros; upstream logs the mismatch

    euler = result.xmap.rotations.to_euler(degrees=False)
    euler_arr = np.asarray(euler).reshape(-1, 3)
    if euler_arr.shape[0] == n_pixels:
        orientation_map = euler_arr.reshape((*grid_shape, 3)).astype(np.float32)
    elif euler_arr.shape[0] > n_pixels and euler_arr.shape[0] % n_pixels == 0:
        orientation_map = euler_arr[-n_pixels:].reshape((*grid_shape, 3)).astype(np.float32)
    else:
        orientation_map = np.zeros((*grid_shape, 3), dtype=np.float32)

    ci_valid = ci_map[~np.isnan(ci_map)]
    metadata = {
        "phase_file": phase_path,
        "ci_mean": float(np.nanmean(ci_valid)) if len(ci_valid) > 0 else 0.0,
        "ci_median": float(np.nanmedian(ci_valid)) if len(ci_valid) > 0 else 0.0,
        "duration_sec": duration,
        "started_at": "",
    }

    # Pull the crystallographic symmetry off the indexed xmap so the
    # checkpoint keeps it for later .ang / .ctf / .h5 exports. Without
    # this the exported file loses its point group and IPF coloring
    # collapses to random colour noise (all orientations treated as
    # triclinic fundamental zone).
    try:
        phases_on_xmap = result.xmap.phases
        for pid in phases_on_xmap.ids:
            if pid >= 0:
                p = phases_on_xmap[pid]
                if getattr(p, "space_group", None) is not None:
                    try:
                        metadata["space_group"] = int(p.space_group.number)
                    except Exception:
                        pass
                if getattr(p, "point_group", None) is not None:
                    try:
                        metadata["point_group"] = str(p.point_group.name)
                    except Exception:
                        pass
                break
    except Exception as e:
        logger.debug("Could not extract symmetry from xmap for %s: %s", phase_name, e)

    return ci_map, orientation_map, metadata


class BatchManager:
    """Central orchestrator for multi-phase batch indexing.

    Key workflow per file:
    1. Load signal (safe_loader)
    2. Register in CalibrationStore (preserves any pre-set PC)
    3. Apply preprocessing (frame avg, BG removal)
    4. Index all phases for this file
    5. Write checkpoints
    6. Unload signal + gc.collect()
    """

    def __init__(self, db_path: Optional[str] = None):
        self.queue = BatchQueue(db_path) if db_path else BatchQueue()
        self.guardian = MemoryGuardian()
        self._paused = False
        self._stopped = False

    def create_batch(
        self,
        files: List[str],
        phases: List[Dict[str, str]],
        config: Dict[str, Any],
    ) -> Tuple[str, PreFlightReport]:
        """Create a batch: run pre-flight checks, create jobs in SQLite."""
        checker = PreFlightChecker()
        report = checker.run(
            files, phases,
            export_dir=config.get("export_dir", ""),
            export_formats=config.get("export_formats"),
            auto_export=config.get("auto_export", True),
        )
        batch_id = self.queue.create_batch(files, phases, config)
        return batch_id, report

    def get_status(self, batch_id: str) -> Optional[Dict]:
        return self.queue.get_batch_status(batch_id)

    def get_jobs(self, batch_id: str) -> List[Dict]:
        return self.queue.get_jobs(batch_id)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stopped = True

    def run_batch_sync(
        self,
        batch_id: str,
        progress_callback: Optional[Callable] = None,
    ):
        """Run batch synchronously (called from executor thread).

        Processes all pending jobs, grouped by file for efficiency:
        - Load file once
        - Apply preprocessing once
        - Run all phases for that file
        - Unload and cleanup
        """
        self._paused = False
        self._stopped = False
        # Crash-recovery: any 'running' state from a previous (crashed) backend
        # process is reset to 'paused' BEFORE we mark this run as running.
        # Doing it the other way around (the previous order) immediately reset
        # this batch's freshly-set 'running' state to 'paused', so the dashboard
        # showed 'paused' through the entire run and the final 'completed' was
        # often skipped if export raised an exception.
        self.queue.cleanup_stale()
        self.queue.set_batch_status(batch_id, "running")

        # Get batch config
        batch_status = self.queue.get_batch_status(batch_id)
        batch_config = {}
        if batch_status and batch_status.get("config_json"):
            import json
            try:
                batch_config = json.loads(batch_status["config_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        preprocessing_config = batch_config.get("preprocessing", {})
        checkpoints: Dict[str, CheckpointWriter] = {}
        current_signal = None
        current_file = None

        while True:
            if self._stopped:
                self.queue.set_batch_status(batch_id, "paused")
                logger.info("Batch %s stopped by user", batch_id)
                break

            if self._paused:
                self.queue.set_batch_status(batch_id, "paused")
                logger.info("Batch %s paused", batch_id)
                break

            job = self.queue.next_job(batch_id)
            if job is None:
                break  # All done

            file_path = job["file_path"]
            phase_name = job["phase_name"]

            # --- File grouping: load new file if different from current ---
            if file_path != current_file:
                # Unload previous signal
                if current_signal is not None:
                    logger.info("Unloading previous signal: %s", Path(current_file).stem)
                    del current_signal
                    current_signal = None
                    gc.collect()

                # Load new file
                try:
                    current_signal = _load_signal(file_path)
                    current_file = file_path

                    # Register calibration (preserves any PC set via copy-pc)
                    _register_signal_calibration(current_signal, file_path)

                    # Get grid shape from signal. navigation_shape is (nx, ny)
                    # for 2D scans and (n,) for 1D line scans; handle both so
                    # line-scan datasets don't crash on tuple-index access.
                    nav_shape = current_signal.axes_manager.navigation_shape[::-1]
                    if len(nav_shape) >= 2:
                        grid_shape = (int(nav_shape[0]), int(nav_shape[1]))
                    elif len(nav_shape) == 1:
                        grid_shape = (1, int(nav_shape[0]))
                    else:
                        raise ValueError(
                            f"Signal has no navigation dimensions: {nav_shape}"
                        )

                    # Apply preprocessing (only for hough/dictionary — spherical reads from disk)
                    first_method = job["method"]
                    applied_preprocessing = {}
                    if preprocessing_config and first_method != "spherical":
                        current_signal, applied_preprocessing = _apply_preprocessing(
                            current_signal, preprocessing_config
                        )
                        if applied_preprocessing:
                            logger.info(
                                "Preprocessing applied to %s: %s",
                                Path(file_path).stem, applied_preprocessing,
                            )
                    elif preprocessing_config and first_method == "spherical":
                        logger.info(
                            "Preprocessing skipped for %s: spherical reads patterns "
                            "directly from disk", Path(file_path).stem,
                        )
                        applied_preprocessing = {"_skipped": "spherical-bypass"}

                    logger.info("Loaded: %s (grid: %s, method: %s)", Path(file_path).stem, grid_shape, first_method)

                except Exception as e:
                    logger.exception("Failed to load %s", file_path)
                    # Mark all jobs for this file as failed
                    all_jobs = self.queue.get_jobs(batch_id)
                    for j in all_jobs:
                        if j["file_path"] == file_path and j["status"] == "pending":
                            self.queue.update_job(j["id"], status="failed", error_msg=f"Load failed: {e}")
                    current_file = None
                    current_signal = None
                    continue

            # --- Checkpoint management ---
            if file_path not in checkpoints:
                # Pull the real µm step from the signal's axes_manager so the
                # exporter can stamp .ang/.ctf/.light with the right scale.
                # Without this every downstream tool reconstructs an axes
                # manager with scale=1.0 and µm distances are meaningless.
                step_size_um = _extract_step_size_um(current_signal, file_path)
                if step_size_um <= 0.0:
                    raise ValueError(
                        f"Could not determine step size for {Path(file_path).name}: "
                        "signal axes_manager has no scale and h5oina header "
                        "lacks 'X Step'. Refusing to start batch — exports "
                        "would be unusable for any quantitative analysis."
                    )
                checkpoints[file_path] = CheckpointWriter(file_path)
                checkpoints[file_path].init_metadata(
                    grid_shape=grid_shape, batch_id=batch_id,
                    method=job["method"], step_size_um=step_size_um,
                )
                # Record preprocessing at file level (same preprocessing is
                # applied once before all phases run against that file).
                try:
                    checkpoints[file_path].write_preprocessing_meta(
                        requested=preprocessing_config,
                        applied=applied_preprocessing,
                    )
                except Exception as e:
                    logger.warning("Could not write preprocessing meta: %s", e)
            cw = checkpoints[file_path]

            if cw.phase_already_done(phase_name):
                logger.info("Skipping %s x %s (already in checkpoint)", Path(file_path).stem, phase_name)
                # Mark status="skipped" (not "done"!) so the jobs table can
                # visually distinguish checkpoint-resumed rows from ones
                # indexed in this run. Cached CI values still surface so
                # users see the match quality rather than zeros.
                cached = cw.get_phase_metadata(phase_name) or {}
                self.queue.update_job(
                    job["id"], status="skipped",
                    ci_mean=cached.get("ci_mean", 0.0),
                    ci_median=cached.get("ci_median", 0.0),
                    duration_sec=cached.get("duration_sec", 0.0),
                )
                continue

            # --- Memory check ---
            mem_status = self.guardian.check_can_proceed()
            if mem_status == MemoryStatus.CRITICAL:
                self.guardian.force_cleanup()
                mem_status = self.guardian.check_can_proceed()
                if mem_status == MemoryStatus.CRITICAL:
                    logger.warning("Memory CRITICAL — pausing batch")
                    self.queue.set_batch_status(batch_id, "paused")
                    break

            # --- Run indexing ---
            self.queue.update_job(job["id"], status="running")
            if progress_callback:
                progress_callback(job, f"Indexing {Path(file_path).stem} x {phase_name}")

            try:
                ci_map, orientation_map, metadata = run_single_indexing_job(
                    signal=current_signal,
                    file_path=file_path,
                    phase_name=phase_name,
                    phase_path=job["phase_path"],
                    method=job["method"],
                    grid_shape=grid_shape,
                    job_config=batch_config,
                    progress_callback=lambda msg, pct=None: (
                        progress_callback(job, msg) if progress_callback else None
                    ),
                )

                cw.write_phase_result(phase_name, ci_map, orientation_map, metadata)
                self.queue.update_job(
                    job["id"], status="done",
                    ci_mean=metadata["ci_mean"],
                    ci_median=metadata.get("ci_median"),
                    duration_sec=metadata["duration_sec"],
                )
                logger.info(
                    "Done: %s x %s — CI=%.3f, %.1fs",
                    Path(file_path).stem, phase_name,
                    metadata["ci_mean"], metadata["duration_sec"],
                )

            except Exception as e:
                logger.exception("Failed: %s x %s", Path(file_path).stem, phase_name)
                self.queue.update_job(job["id"], status="failed", error_msg=str(e))

            self.guardian.force_cleanup()

        # --- Cleanup: unload last signal ---
        if current_signal is not None:
            del current_signal
            gc.collect()

        # --- Compute auto-assignment for completed files ---
        postproc = batch_config.get("postprocessing", {}) or {}
        ci_thr   = float(postproc.get("ci_threshold", 0) or 0)
        unc_thr  = float(postproc.get("uncertainty_threshold", 0) or 0)
        min_clst = int(postproc.get("min_cluster_size", 0) or 0)
        for file_path, cw in checkpoints.items():
            jobs_for_file = [j for j in self.queue.get_jobs(batch_id) if j["file_path"] == file_path]
            if all(j["status"] in ("done", "skipped") for j in jobs_for_file):
                try:
                    cw.compute_auto_assignment()
                except Exception as e:
                    logger.warning("Auto-assignment failed for %s: %s", file_path, e)
                # Optional cleanup: applies MTEX-style filters to the winning
                # phase assignment so the exports don't propagate noise-pixel
                # false positives. Nothing happens when all thresholds are 0.
                if ci_thr > 0 or unc_thr > 0 or min_clst > 0:
                    try:
                        stats = cw.apply_cleanup(
                            ci_threshold=ci_thr,
                            uncertainty_threshold=unc_thr,
                            min_cluster_size=min_clst,
                        )
                        logger.info("Cleanup applied to %s: %s", Path(file_path).stem, stats)
                    except Exception as e:
                        logger.warning("Cleanup failed for %s: %s", file_path, e)

        # --- Export results for completed files ---
        export_dir = batch_config.get("export_dir", "")
        auto_export = batch_config.get("auto_export", True)
        export_formats = batch_config.get("export_formats")  # may be None = default set
        if auto_export and not self._paused and not self._stopped:
            from backend.api.services.result_exporter import export_all
            from backend.api.services.calibration_store import calibration_store

            for file_path, cw in checkpoints.items():
                jobs_for_file = [j for j in self.queue.get_jobs(batch_id) if j["file_path"] == file_path]
                if not all(j["status"] in ("done", "skipped") for j in jobs_for_file):
                    continue
                try:
                    out = export_dir or str(Path(file_path).parent)
                    pc_entry = calibration_store.get_entry(Path(file_path).stem)
                    pc_val = [float(v) for v in pc_entry.pc_single] if pc_entry else None
                    s_tilt = pc_entry.sample_tilt if pc_entry else 70.0
                    d_shape = pc_entry.detector_shape if pc_entry else None

                    results = export_all(
                        source_h5_path=file_path,
                        checkpoint_path=cw.checkpoint_path,
                        output_dir=out,
                        indexing_params=batch_config.get("preprocessing", {}),
                        pc=pc_val,
                        sample_tilt=s_tilt,
                        detector_shape=d_shape,
                        formats=export_formats,
                        # Default ON: full EDS in .light is required for the
                        # phase-vorauswahl workflow (FEAT-10) — the .light
                        # is useless for EDS-driven phase filtering without
                        # the original spectra alongside the indexing data.
                        include_eds=bool(batch_config.get("include_eds", True)),
                    )
                    logger.info("Exported %s: %s", Path(file_path).stem, {k: v for k, v in results.items() if v})
                except Exception as e:
                    logger.warning("Export failed for %s: %s", file_path, e)

        # --- Final status ---
        if not self._paused and not self._stopped:
            self.queue.set_batch_status(batch_id, "completed")
