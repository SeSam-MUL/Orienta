# pc_controller.py
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
from kikuchipy.signals import EBSD
from ebsd_utils import create_indexer, optimize_pc, prepare_reflectors
import hough_reflectors

class PhaseSetError(ValueError):
    """The set of phases cannot be changed as asked.

    ``code`` and ``params`` are stable (the page words the problem in the user's
    language); the message is the English sentence for the log.
    """

    def __init__(self, code, message, **params):
        super().__init__(message)
        self.code = code
        self.message = message
        self.params = params

    def as_detail(self):
        return {"code": self.code, "message": self.message, "params": self.params}


def _same_file(a, b):
    """Two paths name the same file: spelling, case on Windows, slashes, and a
    link or junction on the way (the library is often reached through one)."""
    import os
    return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))


class PCController:
    """
    Controller for EBSD workflows: load phases, manage patterns,
    indexing with caching and global Pattern Center refinement.
    """
    def __init__(self):
        self.detector = None
        self.phase = None
        self.phase_list = None
        self.reflectors = None
        self.phase_paths = {}         # phase name -> the file it was loaded from
        self.indexer = None
        self.patterns = []            # list of (coords, pattern)
        self.cache = {}               # idx -> {'ci', 'xmap', 'band_data'}

        # Indexing parameters with defaults
        self.min_d = 1.0
        self.f_threshold = 0.1
        self.max_reflectors = 70
        self.nBands = 12
        # Version of the per-phase reflector registry (`hough_reflectors`) this
        # controller last looked at: when it moves, the indexer and everything
        # computed with it are out of date.
        self._spec_version = hough_reflectors.registry_version()

    def attach_detector(self, detector):
        """Set the global detector and invalidate cache/indexer."""
        self.detector = detector
        self.indexer = None
        self.clear_cache()

    def load_phase(self, path):
        """Load phase from CIF, create PhaseList, reflectors and invalidate cache/indexer."""
        from orix.crystal_map import PhaseList
        from pathlib import Path as P
        from ebsd_utils import hough_phase_from_cif
        phase = hough_phase_from_cif(path)
        # Restore original name if sanitize_cif created a temp file
        original_stem = P(path).stem
        if phase.name != original_stem:
            phase.name = original_stem
        self.phase = phase
        self.phase_list = PhaseList(phase)
        import os
        self.phase_paths = {str(phase.name): os.path.abspath(str(path))}
        # Compute reflectors with current GUI parameters
        self.reflectors = self._compute_reflectors()
        self.indexer = None
        self.clear_cache()
        return phase

    def _compute_reflectors(self):
        """The reflectors of the loaded phases: the default list, or - for a
        phase with a reflector selection of its own - that selection.

        Used for the Kikuchi-line overlay; the indexer reads the same registry
        inside ``create_indexer``, so the bands drawn are the bands used. A
        selection that no longer fits its phase is reported when the indexer is
        built (loudly); here the default list stands in so the phase can still be
        loaded and the selection reset.
        """
        try:
            return hough_reflectors.prepare_reflectors(
                self.phase_list, min_d=self.min_d, f_threshold=self.f_threshold,
                max_reflectors=self.max_reflectors)
        except hough_reflectors.SpecError as exc:
            logger.warning("reflector selection not applied to the overlay: %s", exc)
            return prepare_reflectors(
                self.phase_list, min_d=self.min_d, f_threshold=self.f_threshold,
                max_reflectors=self.max_reflectors)

    def reflector_specs_changed(self):
        """True (once) when the per-phase reflector registry changed since this
        controller last looked; the indexer, the results computed with it and
        the overlay's reflectors are then rebuilt."""
        version = hough_reflectors.registry_version()
        if version == self._spec_version:
            return False
        self._spec_version = version
        if self.phase_list is not None:
            self.reflectors = self._compute_reflectors()
        self.clear_cache()           # also drops the indexer
        return True

    def clear_cache(self):
        """Clears the pattern index cache."""
        self.cache = {}
        self.indexer = None

    # --- Several phases (e.g. austenite + ferrite of a duplex steel) -------
    #
    # PyEBSDIndex indexes each pattern against every phase of the list and keeps
    # the best-fitting one; its PC optimiser scores a trial PC on that
    # best-of-phases result. So holding several phases here is enough for
    # ``index_pattern`` / ``refine_global_pc`` to work with them. ``load_phase``
    # stays "replace everything with this one phase".

    #: Upper bound on phases held at once. Every phase adds a full Hough pass
    #: (and its own band-triplet library) to every optimiser evaluation.
    MAX_PHASES = 8

    def phase_names(self) -> List[str]:
        """Names of the loaded phases, in list order (empty when none)."""
        if self.phase_list is None:
            return []
        return [str(self.phase_list[pid].name) for pid in self.phase_list.ids]

    def _set_phases(self, phases):
        """Make ``phases`` (a non-empty list of orix Phase) the loaded set."""
        from orix.crystal_map import PhaseList
        self.phase = phases[0]
        self.phase_list = PhaseList(phases=list(phases))
        self.reflectors = self._compute_reflectors()
        self.indexer = None
        self.clear_cache()

    def add_phase(self, path):
        """Add one more phase to those already loaded and return it.

        With nothing loaded this is ``load_phase``. A phase whose name (the CIF
        stem; a reflector selection is stored under it too) is already loaded is
        refused - as a duplicate when it is the same file, as "same name, other
        file" when it is not - and a CIF that cannot be read leaves the loaded
        phases exactly as they were. Every refusal is a ``PhaseSetError`` with a
        code.
        """
        import os
        from pathlib import Path as P
        stem = P(str(path)).stem
        if self.phase_list is not None:
            for n, loaded in self.phase_paths.items():
                if n.casefold() == stem.casefold():
                    if _same_file(loaded, path):
                        raise PhaseSetError(
                            "phase_duplicate", f"The phase '{n}' is already loaded.", name=n)
                    raise PhaseSetError(
                        "phase_same_name",
                        f"A different file called '{n}' is already loaded "
                        f"({loaded}); a phase is identified by its file name.",
                        name=n, loaded_path=loaded, path=str(path))
        if self.phase_list is None:
            try:
                return self.load_phase(path)
            except PhaseSetError:
                raise
            except Exception as exc:  # noqa: BLE001 - diffpy raises unrelated types
                raise PhaseSetError(
                    "phase_unreadable", f"{stem} could not be read: {exc}",
                    name=stem, path=str(path), reason=str(exc)[:300]) from exc
        previous = [self.phase_list[pid] for pid in self.phase_list.ids]
        if len(previous) >= self.MAX_PHASES:
            raise PhaseSetError(
                "phase_limit",
                f"At most {self.MAX_PHASES} phases can be used together; "
                "remove one first.", max=self.MAX_PHASES)
        saved = (self.phase, self.phase_list, self.reflectors, dict(self.phase_paths))
        try:
            # ``load_phase`` is the one place a CIF is read into a Phase.
            new_phase = self.load_phase(path)
            if str(new_phase.name) in {str(p.name) for p in previous}:
                raise PhaseSetError(
                    "phase_duplicate",
                    f"A phase named '{new_phase.name}' is already loaded.",
                    name=str(new_phase.name))
            paths = dict(saved[3])
            paths[str(new_phase.name)] = os.path.abspath(str(path))
            self._set_phases(previous + [new_phase])
            self.phase_paths = paths
        except Exception as exc:
            self.phase, self.phase_list, self.reflectors, self.phase_paths = saved
            self.indexer = None
            self.clear_cache()
            if isinstance(exc, PhaseSetError):
                raise
            raise PhaseSetError(
                "phase_unreadable", f"{stem} could not be read: {exc}",
                name=stem, path=str(path), reason=str(exc)[:300]) from exc
        return new_phase

    def remove_phase(self, name):
        """Remove the loaded phase called ``name`` (KeyError if not loaded)."""
        if self.phase_list is None:
            raise KeyError(name)
        phases = [self.phase_list[pid] for pid in self.phase_list.ids]
        keep = [p for p in phases if str(p.name) != str(name)]
        if len(keep) == len(phases):
            raise KeyError(name)
        if not keep:
            self.phase = None
            self.phase_list = None
            self.reflectors = None
            self.phase_paths = {}
            self.indexer = None
            self.clear_cache()
            return
        self._set_phases(keep)
        self.phase_paths = {k: v for k, v in self.phase_paths.items() if k != str(name)}

    def update_indexing_params(self, min_d=None, f_threshold=None,
                               max_reflectors=None, nBands=None):
        """
        Updates indexing parameters and recalculates reflectors if needed.
        Invalidates cache and indexer on changes.
        """
        changed = False

        if min_d is not None and min_d != self.min_d:
            self.min_d = min_d
            changed = True
        if f_threshold is not None and f_threshold != self.f_threshold:
            self.f_threshold = f_threshold
            changed = True
        if max_reflectors is not None and max_reflectors != self.max_reflectors:
            self.max_reflectors = max_reflectors
            changed = True
        if nBands is not None and nBands != self.nBands:
            self.nBands = nBands
            # nBands only affects indexer, not reflectors; what was indexed with
            # the old number is stale
            self.clear_cache()

        if changed:
            # Recalculate reflectors with current parameters
            if self.phase_list is not None:
                self.reflectors = self._compute_reflectors()
            self.indexer = None
            self.clear_cache()

    def _indexer_pc_is_stale(self):
        """True when the cached indexer was built for another PC than the detector's.

        A PyEBSDIndex indexer copies the PC when it is created; assigning
        ``detector.pc`` afterwards (a refine, a typed value) does not reach it, so
        whatever it indexes is indexed at the OLD PC.
        """
        try:
            built = np.asarray(self.indexer.PC, dtype=float).ravel()
            now = np.asarray(self.detector.pc_flattened, dtype=float).ravel()
            return built.shape != now.shape or not np.allclose(built, now, rtol=0, atol=1e-12)
        except Exception:  # noqa: BLE001 - cannot tell: rebuilding is always safe
            return True

    def _ensure_indexer(self):
        """Create indexer if not already cached. Validates prerequisites."""
        if self.detector is None:
            raise RuntimeError("No detector set.")
        if self.phase_list is None:
            raise RuntimeError("No phase loaded.")
        self.reflector_specs_changed()
        if self.indexer is not None and self._indexer_pc_is_stale():
            self.indexer = None
        if self.indexer is None:
            self.indexer = create_indexer(
                self.detector, self.phase_list, self.reflectors,
                nBands=self.nBands
            )
        return self.indexer

    def add_pattern(self, coords, pattern):
        """Add a new pattern. Does not invalidate existing cache entries."""
        idx = len(self.patterns)
        self.patterns.append((coords, pattern))
        # Don't clear cache — existing indexed patterns are still valid.
        # Only the new pattern at idx needs indexing.
        return idx

    def index_pattern(self, idx):
        """Index a single pattern and cache the result."""
        if idx < 0 or idx >= len(self.patterns):
            raise ValueError(f"Pattern index {idx} out of range [0, {len(self.patterns)})")
        self._ensure_indexer()
        coords, pat = self.patterns[idx]
        # pat is 2D (H, W); hough_indexing requires at least one navigation axis.
        # Adding two singleton dims → (1, 1, H, W) avoids navigation_shape=() crash.
        ebsd = EBSD(pat[np.newaxis, np.newaxis], detector=self.detector)
        xmap, index_data, band_data = ebsd.hough_indexing(
            self.phase_list, self.indexer,
            return_index_data=True, return_band_data=True,
            verbose=0
        )
        ci = float(index_data['cm'].mean())
        
        self.cache[idx] = {'ci': ci, 'xmap': xmap, 'band_data': band_data}
        return ci, xmap, band_data

    def phases_of_patterns(self, patterns):
        """Which phase indexes each pattern best, at the current detector.

        ``patterns`` are 2D arrays. Returns one dict per pattern with
        ``phase_name``, ``phase_index`` (position in the phase list) and ``ci``;
        a pattern that no phase fitted (or that failed to index) has ``None``
        for the first two. Does not touch the per-pattern result cache.
        """
        self._ensure_indexer()
        ids = [int(i) for i in self.phase_list.ids]
        out = []
        for pat in patterns:
            try:
                ebsd = EBSD(np.asarray(pat)[np.newaxis, np.newaxis], detector=self.detector)
                xmap, index_data = ebsd.hough_indexing(
                    self.phase_list, self.indexer,
                    return_index_data=True, verbose=0
                )
                pid = int(np.asarray(xmap.phase_id).ravel()[0])
                ci = float(index_data['cm'].mean())
            except Exception as exc:  # one failing pattern must not hide the rest
                logger.warning("phase lookup failed for a calibration pattern: %s", exc)
                out.append({"phase_name": None, "phase_index": None, "ci": None})
                continue
            if pid in ids:
                out.append({"phase_name": str(self.phase_list[pid].name),
                            "phase_index": ids.index(pid), "ci": ci})
            else:
                out.append({"phase_name": None, "phase_index": None, "ci": ci})
        return out

    def index_all_patterns(self):
        """Index all loaded patterns and cache all results."""
        self.clear_cache()
        for idx in range(len(self.patterns)):
            self.index_pattern(idx)
        return
    
    
    @staticmethod
    def _pc_outlier_mask(pcs_arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask (True = outlier) for a (N, 3) PC array, using the same
        physical ranges as :meth:`_validate_pcs`. Used to feed ``is_outlier``
        into kikuchipy's ``fit_pc``/``extrapolate_pc`` so NaN / implausible
        refinements don't corrupt the plane fit.
        """
        pcs_arr = np.asarray(pcs_arr, dtype=float)
        if pcs_arr.ndim != 2 or pcs_arr.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {pcs_arr.shape}")
        valid = ~np.any(np.isnan(pcs_arr), axis=1)
        valid &= (pcs_arr[:, 0] >= -0.5) & (pcs_arr[:, 0] <= 1.5)
        valid &= (pcs_arr[:, 1] >= -0.5) & (pcs_arr[:, 1] <= 1.5)
        valid &= (pcs_arr[:, 2] >= 0.01) & (pcs_arr[:, 2] <= 2.0)
        return ~valid

    @staticmethod
    def _validate_pcs(pcs_arr: np.ndarray) -> np.ndarray:
        """
        Validate and filter PC array. Removes NaN and physically implausible values.

        Valid PC ranges (must match :meth:`_pc_outlier_mask`):
            PCx, PCy: -0.5 - 1.5 (detector coordinates, usually 0-1)
            PCz: 0.01 - 2.0 (detector distance, typically 0.3-0.8)

        Returns filtered array (may be shorter than input).
        """
        if pcs_arr.ndim != 2 or pcs_arr.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {pcs_arr.shape}")

        # NaN filter
        valid = ~np.any(np.isnan(pcs_arr), axis=1)

        # Physical range check
        valid &= (pcs_arr[:, 0] >= -0.5) & (pcs_arr[:, 0] <= 1.5)  # PCx
        valid &= (pcs_arr[:, 1] >= -0.5) & (pcs_arr[:, 1] <= 1.5)  # PCy
        valid &= (pcs_arr[:, 2] >= 0.01) & (pcs_arr[:, 2] <= 2.0)  # PCz

        n_invalid = int((~valid).sum())
        if n_invalid > 0:
            logger.warning(
                "%d/%d PC values rejected (NaN or out of physical range)",
                n_invalid, len(pcs_arr)
            )

        # Outlier detection (IQR-based) on valid values. `filtered` and the
        # filtered->array index map are computed ONCE on the initial valid set;
        # the per-dim loop only flips additional bits in `valid`. (Recomputing
        # np.where(valid) inside the loop shrank it after the first dim while
        # dim_valid stayed at the original length -> IndexError; fixed.)
        filtered = pcs_arr[valid]
        if len(filtered) >= 4:
            valid_indices = np.where(valid)[0]          # len == len(filtered), fixed
            for dim in range(3):
                q1, q3 = np.percentile(filtered[:, dim], [25, 75])
                iqr = q3 - q1
                lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
                dim_valid = (filtered[:, dim] >= lower) & (filtered[:, dim] <= upper)
                if not dim_valid.all():
                    n_outliers = int((~dim_valid).sum())
                    logger.warning(
                        "%d PC values are outliers in dimension %d (IQR: %.4f-%.4f)",
                        n_outliers, dim, lower, upper
                    )
                    valid[valid_indices[~dim_valid]] = False

        return pcs_arr[valid]

    def refine_global_pc(self, method="PSO", search_limit=0.05,
                         progress_callback=None):
        """Run PSO optimization over all patterns and set global PC.

        Args:
            method: Optimization method ("PSO" or "NM")
            search_limit: Search range for each PC component
            progress_callback: Optional callable(int, int) for (current, total) progress
        """
        if not self.patterns:
            raise RuntimeError("No patterns loaded.")
        self._ensure_indexer()
        pcs = []
        n = len(self.patterns)
        for i, (_, pat) in enumerate(self.patterns):
            mean_pc, _ = optimize_pc(
                detector=self.detector,
                indexer=self.indexer,
                pattern=pat,
                method=method,
                search_limit=search_limit,
                batch=True
            )
            pcs.append(mean_pc)
            logger.debug("Pattern %d/%d: PC %s", i + 1, n, mean_pc)
            if progress_callback is not None:
                progress_callback(i + 1, n)
        pcs_arr = np.array(pcs)
        pcs_arr = self._validate_pcs(pcs_arr)
        if len(pcs_arr) == 0:
            raise RuntimeError("All PC optimizations failed (invalid results)")
        global_pc = np.mean(pcs_arr, axis=0)
        self.detector.pc = tuple(global_pc)
        self.indexer = None
        self.clear_cache()
        return tuple(global_pc)

    # === Pixel-wise PC Correction (FEAT-11) ===

    def suggest_grid_step(
        self,
        n_rows: int,
        n_cols: int,
        target_points: int = 50,
    ) -> Tuple[int, int, float]:
        """
        Suggest a grid step that gives approximately target_points calibration points.

        Returns:
            (step, estimated_points, estimated_minutes)
        """
        # Compute step to achieve target_points on a rectangular grid
        area = n_rows * n_cols
        step = max(1, int(np.sqrt(area / max(target_points, 1))))

        # Compute actual number of points with this step
        n_rows_grid = len(range(0, n_rows, step))
        n_cols_grid = len(range(0, n_cols, step))
        # Include edge rows/cols
        if (n_rows - 1) % step != 0:
            n_rows_grid += 1
        if (n_cols - 1) % step != 0:
            n_cols_grid += 1
        estimated_points = n_rows_grid * n_cols_grid

        # Rough time estimates: ~3s per point for NM, ~8s per point for PSO
        time_per_point = 5.0  # average seconds
        estimated_minutes = (estimated_points * time_per_point) / 60.0

        return step, estimated_points, estimated_minutes

    def generate_calibration_grid(
        self,
        n_rows: int,
        n_cols: int,
        step: int = 10,
    ) -> List[Tuple[int, int]]:
        """
        Generate a sparse grid of calibration points across the scan.

        Args:
            n_rows: Total number of rows in the scan
            n_cols: Total number of columns in the scan
            step: Spacing between calibration points (in pixels)

        Returns:
            List of (row, col) tuples for calibration points
        """
        if step < 1:
            raise ValueError(f"Step must be >= 1, got {step}")
        rows = list(range(0, n_rows, step))
        cols = list(range(0, n_cols, step))
        # Ensure last row/col are included
        if rows[-1] != n_rows - 1:
            rows.append(n_rows - 1)
        if cols[-1] != n_cols - 1:
            cols.append(n_cols - 1)
        grid = [(r, c) for r in rows for c in cols]
        return grid

    def refine_pc_at_points(
        self,
        signal: EBSD,
        calibration_points: List[Tuple[int, int]],
        method: str = "PSO",
        search_limit: float = 0.05,
        progress_callback=None,
    ) -> np.ndarray:
        """
        Refine PC at specific calibration points.

        Args:
            signal: Full EBSD signal (4D: n_rows, n_cols, height, width)
            calibration_points: List of (row, col) pixel coordinates
            method: Optimization method (PSO or NM)
            search_limit: Search range for each PC component
            progress_callback: Optional callable(int, int) for (current, total)

        Returns:
            Array of shape (n_points, 3) with refined PC values per point.
            Invalid values (NaN, out-of-range) are logged but kept in place
            so the caller can map results back to calibration_points.
        """
        self._ensure_indexer()

        # Validate calibration points are within signal bounds
        nav_shape = signal.data.shape[:2]
        for i, (row, col) in enumerate(calibration_points):
            if not (0 <= row < nav_shape[0] and 0 <= col < nav_shape[1]):
                raise ValueError(
                    f"Calibration point {i} ({row},{col}) out of bounds "
                    f"for signal shape {nav_shape}"
                )

        n = len(calibration_points)
        refined_pcs = []
        for i, (row, col) in enumerate(calibration_points):
            pat = signal.data[row, col]
            mean_pc, _ = optimize_pc(
                detector=self.detector,
                indexer=self.indexer,
                pattern=pat,
                method=method,
                search_limit=search_limit,
                batch=True,
            )
            refined_pcs.append(mean_pc)
            if progress_callback is not None:
                progress_callback(i + 1, n)

        result = np.array(refined_pcs)
        # Validate and warn (don't filter — caller needs positional mapping)
        valid_result = self._validate_pcs(result)
        n_bad = len(result) - len(valid_result)
        if n_bad > 0:
            logger.warning(
                "%d/%d calibration points produced invalid PC values",
                n_bad, n
            )
        return result

    def fit_pc_from_grid(
        self,
        calibration_points: List[Tuple[int, int]],
        refined_pcs: np.ndarray,
        n_rows: int,
        n_cols: int,
        transformation: str = "affine",
    ) -> 'EBSDDetector':
        """
        Fit a smooth PC field through calibration PCs and interpolate to all pixels.

        Both transformations preserve co-planarity of the detector across the
        scan. The *affine* fit (linear least squares) is the DEFAULT because it
        is robust to the per-point PC noise our PSO/NM grid refinement produces.
        The *projective* fit has more degrees of freedom and, on high-quality
        DENSE PC grids with large drift, is more accurate (Winkelmann et al.
        2020, Materials 13(12):2816: ~4x lower RMS — it is kikuchipy's own
        ``fit_pc`` default). BUT on real LoGainNi data (60x60 detector, PSO PC
        scatter std ~0.01 comparable to the actual drift) the projective
        homography OVERFITS and goes numerically unstable: verified 2026-06-03,
        leave-one-out held-out RMS 0.302 (projective) vs 0.020 (affine) — 15x
        worse, with in-sample PC norms up to 11.5 (garbage). So projective is
        OPT-IN for low-noise dense grids only; affine is the safe default.
        Either way the fitted field is range-checked and a fit that produces
        implausible PCs fails loud (no silent garbage propagation).

        Args:
            calibration_points: List of (row, col) for calibration locations
            refined_pcs: Array of shape (n_points, 3) with refined PCs
            n_rows: Total rows in scan
            n_cols: Total columns in scan
            transformation: "affine" (default, needs >=3 valid points) or
                "projective" (needs >=4; only for low-noise dense grids). No
                silent downgrade — too few valid points raises.

        Returns:
            New EBSDDetector with per-pixel PC values
        """
        from kikuchipy.detectors import EBSDDetector

        transformation = str(transformation).lower()
        if transformation not in ("projective", "affine"):
            raise ValueError(
                f"transformation must be 'projective' or 'affine', got {transformation!r}"
            )
        min_pts = 4 if transformation == "projective" else 3

        refined_pcs = np.asarray(refined_pcs, dtype=float)
        if len(calibration_points) != len(refined_pcs):
            raise ValueError(
                f"calibration_points ({len(calibration_points)}) and "
                f"refined_pcs ({len(refined_pcs)}) must have same length"
            )
        if len(calibration_points) < min_pts:
            raise ValueError(
                f"Need at least {min_pts} calibration points for a {transformation} "
                f"plane fit, got {len(calibration_points)}"
            )

        # kikuchipy's fit_pc fits a plane through the detector's OWN .pc array,
        # so the calibration detector must CARRY the refined per-point PCs —
        # using self.detector (single global PC) would discard them and raise
        # "Fitting requires multiple projection centers".
        is_outlier = self._pc_outlier_mask(refined_pcs)
        if is_outlier.all():
            raise ValueError(
                f"all {len(refined_pcs)} calibration points produced implausible PC "
                "values — grid refinement failed. Try a different method or a larger "
                "search_limit, or check the detector geometry / phase."
            )
        n_valid = int((~is_outlier).sum())
        if n_valid < min_pts:
            raise ValueError(
                f"{transformation} plane fit needs >= {min_pts} valid calibration "
                f"points, but only {n_valid} of {len(calibration_points)} survived "
                "outlier removal. Use a smaller grid_step (more points), "
                "transformation='affine' (needs 3), or check the phase/geometry."
            )
        cal_det = EBSDDetector(
            shape=self.detector.shape,
            px_size=self.detector.px_size,
            binning=self.detector.binning,
            tilt=self.detector.tilt,
            azimuthal=self.detector.azimuthal,
            sample_tilt=self.detector.sample_tilt,
            pc=refined_pcs,  # (N, 3) → navigation_shape (N,)
        )

        # pc_indices: (2, N) row/col of each calibration point (matches nav_shape)
        pc_indices = np.array(calibration_points).T  # (2, N)
        # map_indices: (2, n_rows, n_cols) → returned detector has 2D nav shape
        all_rows, all_cols = np.mgrid[0:n_rows, 0:n_cols]
        map_indices = np.array([all_rows, all_cols])  # (2, n_rows, n_cols)

        new_det = cal_det.fit_pc(
            pc_indices=pc_indices,
            map_indices=map_indices,
            transformation=transformation,
            is_outlier=is_outlier if is_outlier.any() else None,
            plot=False,
        )

        # Fail loud if the fit produced an implausible field. The projective
        # homography can blow up (divide-by-near-zero) on noisy PCs and emit
        # nonsensical values (observed PC norms up to 11.5) — never let that
        # silently become the active per-pixel detector.
        field = np.asarray(new_det.pc).reshape(-1, 3)
        bad = self._pc_outlier_mask(field)
        if bad.any():
            raise ValueError(
                f"{transformation} plane fit produced {int(bad.sum())}/{len(field)} "
                f"implausible per-pixel PC values (out of physical range; max |PC| "
                f"= {np.nanmax(np.abs(field)):.3f}). This typically means the fit "
                "overfit noisy calibration PCs. Use transformation='affine' "
                "(the default), add grid points, or improve the per-point refinement."
            )
        return new_det

    def extrapolate_pc_from_points(
        self,
        calibration_points: List[Tuple[int, int]],
        refined_pcs: np.ndarray,
        n_rows: int,
        n_cols: int,
        step_sizes: Optional[Tuple[float, float]] = None,
    ) -> 'EBSDDetector':
        """
        Extrapolate PC from calibration points using mean + spatial model.

        Simpler than fit_pc — uses the mean PC and accounts for
        spatial position to estimate per-pixel values.

        Args:
            calibration_points: List of (row, col) calibration locations
            refined_pcs: Array of shape (n_points, 3) with refined PCs
            n_rows: Total rows in scan
            n_cols: Total columns in scan
            step_sizes: (step_y_um, step_x_um) physical step sizes in micrometers

        Returns:
            New EBSDDetector with per-pixel PC values
        """
        from kikuchipy.detectors import EBSDDetector

        refined_pcs = np.asarray(refined_pcs, dtype=float)
        if step_sizes is None:
            step_sizes = (1.0, 1.0)  # Default to 1 µm

        # extrapolate_pc averages the detector's OWN .pc, so the calibration
        # detector must carry the refined per-point PCs (not the global PC).
        is_outlier = self._pc_outlier_mask(refined_pcs)
        if is_outlier.all():
            raise ValueError(
                f"all {len(refined_pcs)} calibration points produced implausible PC "
                "values — grid refinement failed. Try a different method or a larger "
                "search_limit, or check the detector geometry / phase."
            )
        cal_det = EBSDDetector(
            shape=self.detector.shape,
            px_size=self.detector.px_size,
            binning=self.detector.binning,
            tilt=self.detector.tilt,
            azimuthal=self.detector.azimuthal,
            sample_tilt=self.detector.sample_tilt,
            pc=refined_pcs,  # (N, 3)
        )

        pc_indices = np.array(calibration_points).T  # (2, N)
        new_det = cal_det.extrapolate_pc(
            pc_indices=pc_indices,
            navigation_shape=(n_rows, n_cols),
            step_sizes=step_sizes,
            is_outlier=is_outlier if is_outlier.any() else None,
        )
        return new_det

    def apply_pixelwise_pc(self, corrected_detector: 'EBSDDetector'):
        """
        Apply a per-pixel corrected detector as the active detector.

        Args:
            corrected_detector: EBSDDetector with per-pixel PC values
        """
        self.detector = corrected_detector
        self.indexer = None
        self.clear_cache()
