"""Production wrapper around ``Tier1Indexer`` exposing the EMSphInx option set.

This is the single entry point upstream code (FastAPI routes, batch jobs,
notebooks) should use to run the GPU spherical indexing pipeline. It
mirrors the option surface of the EMSphInx CLI so callers can swap
backends by changing one flag, without touching any other parameters.

Design goals
------------
- **Drop-in for the EMSphInx flow**: same ``IndexingConfig`` fields, same
  ``detector_params`` dict, same ``IndexResult`` return shape.
- **No surprise behavior**: every EMSphInx-equivalent flag (circmask,
  gausbckg, nregions, normed, refine, bandwidth) is honoured. Where
  semantics differ (notably ``refine``: we do triquadratic sub-bin, not
  EMSphInx's continuous-SHT Newton step), we document the difference.
- **Hardware-adaptive** (Phase 2, runtime.py): GPU memory is queried;
  batch size is auto-picked; OOM falls back to halved batch; no-CUDA
  systems fall back to CPU.
- **Multi-format input** (Phase 4): H5OINA, EDAX H5, in-memory ndarray,
  custom HDF5 paths. Bruker BCF stays out of v1.
- **Multi-phase** (Phase 4): pass a list of ``PhaseConfig`` objects; the
  backend runs each phase independently and selects the highest-scoring
  result per pixel.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import h5py
import numpy as np
import torch

from .pipeline.detector import DetectorGeometry
from .pipeline.indexer import IndexResult, Tier1Indexer
from .pipeline._shared_tables import SharedSphericalTables
from .pipeline.sht_io import read_sht_master
from .runtime import (
    RuntimeInfo,
    compute_safe_batch,
    detect_runtime,
    run_with_oom_retry,
)


@dataclass
class PhaseConfig:
    """Single-phase configuration for the GPU spherical indexing backend.

    A multi-phase run is just a list of these. Each phase is indexed
    independently against the same patterns, and the per-pixel winner is
    chosen by raw cc score (matches EMSphInx multi-phase behaviour).
    """
    sht_file: str                          # Path to the .sht master pattern
    name: str = ""                         # Display name (defaults to file stem)
    bandwidth: int = 68                    # SHT bandwidth used for indexing
    normed: bool = True                    # Std-normalize patterns before SHT
    refine: bool = True                    # Sub-bin triquadratic Newton step
    circmask: int = -1                     # -1 off, 0 inscribed, >0 explicit radius
    gausbckg: bool = True                  # Subtract 2D Gaussian background
    nregions: int = 10                     # Adaptive histogram equalization tiles
    cc_fp64: bool = False                  # FP64 cc-volume (debugging)
    sample_tilt_deg: Optional[float] = None  # Override master.primary_tilt_deg

    def __post_init__(self) -> None:
        if not self.name:
            self.name = Path(self.sht_file).stem


@dataclass
class BackendConfig:
    """Top-level configuration for ``SphericalGPUBackend``.

    Wraps the per-phase configs plus runtime knobs. Built from an
    ``IndexingConfig`` via ``BackendConfig.from_indexing_config``.
    """
    phases: list[PhaseConfig]
    batch_size: Optional[int] = None       # None = auto-detect from GPU memory (Phase 2)
    device: Optional[torch.device] = None  # None = auto-detect CUDA, fall back to CPU
    flip_y: Optional[bool] = None          # None = auto from vendor (Oxford/EDAX → True)

    @classmethod
    def from_indexing_config(
        cls,
        config,                            # IndexingConfig (avoid hard import for circularity)
        sht_files: Optional[Iterable[str]] = None,
    ) -> "BackendConfig":
        """Build a single- or multi-phase backend config from an IndexingConfig.

        ``sht_files`` overrides ``config.sht_file`` and is the multi-phase
        path used by ``batch_v2``. If neither is set, raises ValueError.
        """
        files = list(sht_files) if sht_files else (
            [config.sht_file] if getattr(config, "sht_file", "") else []
        )
        if not files:
            raise ValueError(
                "BackendConfig.from_indexing_config requires at least one "
                ".sht master file (config.sht_file or sht_files= argument)."
            )
        phases = [
            PhaseConfig(
                sht_file=f,
                bandwidth=int(config.bandwidth),
                normed=bool(config.normed),
                refine=bool(config.refine),
                circmask=int(config.circmask),
                gausbckg=bool(config.gausbckg),
                nregions=int(config.nregions),
                cc_fp64=bool(getattr(config, "cc_fp64", False)),
            )
            for f in files
        ]
        return cls(phases=phases)


class SphericalGPUBackend:
    """Production GPU spherical indexing backend.

    Usage
    -----
    >>> bcfg = BackendConfig.from_indexing_config(config)
    >>> backend = SphericalGPUBackend(bcfg)
    >>> result = backend.index_h5(h5_path, detector_params)

    The backend lazily constructs one ``Tier1Indexer`` per phase on the
    first call to ``index_*``; subsequent calls reuse them so the static
    Wigner-d / SHT precomputation only happens once. Detector geometry
    must be the same across calls — if it changes (e.g. PC drift), call
    ``invalidate()`` and reconfigure.
    """

    def __init__(self, config: BackendConfig):
        if not config.phases:
            raise ValueError("BackendConfig must have at least one phase.")
        self.config = config
        self._device = config.device or self._choose_device()
        # Probe runtime once at construction; used by _auto_batch_size and
        # exposed via .runtime for diagnostic logging by callers.
        self.runtime: RuntimeInfo = detect_runtime(self._device)
        # Indexers are built lazily on first index_* call (we need
        # ``detector_params`` to construct ``DetectorGeometry``).
        self._indexers: list[Tier1Indexer] = []
        self._geom: Optional[DetectorGeometry] = None
        self._detector_params_cached: Optional[dict] = None
        # Cooperative cancellation. Caller installs via ``set_cancel_check``;
        # batch loops poll between batches and raise CancelledIndexingError
        # when the callback returns True. None = no cancel installed.
        self._cancel_check: Optional[Callable[[], bool]] = None

    def set_cancel_check(self, fn: Optional[Callable[[], bool]]) -> None:
        """Install a cooperative-cancel callback.

        The callback is polled once per batch in every index loop
        (``_index_h5_dataset_via_indexer``, ``_index_with_phases``,
        ``_index_with_phases_interleaved``). Returning True raises
        ``indexing_controller.CancelledIndexingError`` from inside the
        loop. Pass ``None`` to clear.
        """
        self._cancel_check = fn

    def _raise_if_cancelled(self) -> None:
        if self._cancel_check is not None and self._cancel_check():
            from indexing_controller import CancelledIndexingError
            raise CancelledIndexingError(
                "Spherical-GPU indexing cancelled by user"
            )

    # --- Public API ----------------------------------------------------------

    def index_h5(
        self,
        h5_path: Union[str, Path],
        detector_params: dict,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> IndexResult:
        """Index every pattern in an H5OINA / EDAX H5 file.

        Parameters
        ----------
        h5_path
            Path to an Oxford ``.h5oina`` or EDAX ``.h5`` containing patterns.
            Pattern dataset is auto-discovered (see
            ``Tier1Indexer._open_pattern_dataset`` for the search list).
        detector_params
            Dict with keys: ``n_rows, n_cols, pat_width, pat_height,
            pixel_size, tilt, binning, step_x, step_y, pc_x, pc_y, pc_z,
            vendor``. Same shape that ``generate_emsphinx_nml`` consumes,
            so existing callers can pass it unchanged.
        progress_callback
            ``(message: str, fraction_done: float) -> None``. Forwarded to
            the indexer; called once per batch.
        """
        self._ensure_built(detector_params)
        return self._index_with_phases(
            kind="h5",
            payload=str(h5_path),
            progress_callback=progress_callback,
        )

    def index_array(
        self,
        patterns: np.ndarray,
        detector_params: dict,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> IndexResult:
        """Index a stack of patterns already loaded in memory.

        ``patterns`` is ``(N, H, W)`` float32 or compatible — typical use
        is when patterns came from a non-H5 source or have been ROI-cropped
        in upstream code. Detector params are still required so we can
        build the right SHT geometry.
        """
        if patterns.ndim != 3:
            raise ValueError(
                f"patterns must be (N, H, W); got shape {patterns.shape!r}"
            )
        self._ensure_built(detector_params)
        return self._index_with_phases(
            kind="array",
            payload=patterns,
            progress_callback=progress_callback,
        )

    def index_signal(
        self,
        signal,                                          # kikuchipy.signals.EBSD
        detector_params: dict,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> IndexResult:
        """Index a kikuchipy / hyperspy EBSD signal.

        Accepts the result of ``kp.load()`` or ``hs.load()`` — useful when
        callers have already loaded an .hspy / .h5oina / .ebsp through
        kikuchipy. The signal's ``data`` attribute is reshaped to (N, H, W)
        and forwarded to :meth:`index_array`.

        We don't auto-extract detector params from the signal because the
        kikuchipy EBSD ``detector`` representation may be incomplete (e.g.
        no PC for raw ``.ebsp`` files); the caller passes them explicitly.
        """
        data = np.asarray(signal.data)
        if data.ndim == 4:
            ny, nx, ph, pw = data.shape
            patterns = data.reshape(ny * nx, ph, pw)
        elif data.ndim == 3:
            patterns = data
        else:
            raise ValueError(
                f"signal.data must be 3D (N, H, W) or 4D (ny, nx, H, W); "
                f"got shape {data.shape!r}"
            )
        return self.index_array(patterns, detector_params, progress_callback)

    def index_h5_dataset(
        self,
        h5_path: Union[str, Path],
        dataset_path: str,
        detector_params: dict,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> IndexResult:
        """Index patterns from an arbitrary HDF5 dataset path.

        Use this when the HDF5 layout doesn't match Oxford H5OINA or EDAX
        conventions (e.g. custom internal pipelines). ``dataset_path`` is
        the slash-delimited path inside the file; the dataset must have
        shape ``(N, H, W)`` along the last three axes.

        Streams patterns batch-by-batch — large datasets (28k patterns at
        640x480 = ~33 GB) are NOT materialized in host RAM, only the
        active batch is read at a time. The HDF5 file is reopened per
        phase for multi-phase indexing (cheap; HDF5 has its own caching).
        """
        self._ensure_built(detector_params)
        # Validate the dataset path once before dispatching.
        with h5py.File(str(h5_path), "r") as f:
            if dataset_path not in f:
                raise KeyError(
                    f"dataset {dataset_path!r} not found in {h5_path}; "
                    f"top-level keys: {list(f.keys())}"
                )
            dset = f[dataset_path]
            if not isinstance(dset, h5py.Dataset) or dset.ndim != 3:
                raise ValueError(
                    f"dataset at {dataset_path!r} must be a 3D HDF5 dataset; "
                    f"got {type(dset).__name__} with shape {getattr(dset, 'shape', None)!r}"
                )
        return self._index_with_phases(
            kind="h5_dataset",
            payload=(str(h5_path), dataset_path),
            progress_callback=progress_callback,
        )

    def invalidate(self) -> None:
        """Drop cached indexers; next call will rebuild from current config."""
        self._indexers = []
        self._geom = None
        self._detector_params_cached = None

    @property
    def device(self) -> torch.device:
        return self._device

    # --- Internals -----------------------------------------------------------

    def _choose_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn(
            "SphericalGPUBackend: CUDA not available — falling back to CPU. "
            "This will be ~10-50x slower than GPU; consider using the "
            "EMSphInx CPU backend instead for production CPU-only runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    def _ensure_built(self, detector_params: dict) -> None:
        """Build (or rebuild) per-phase Tier1Indexers if needed.

        Rebuilds when:
        - This is the first call.
        - Detector params changed (PC drift, different file, etc.) — we do
          a deep equality check on the dict.
        """
        if self._indexers and self._detector_params_cached == detector_params:
            return
        self._geom = DetectorGeometry.from_params(detector_params, device=self._device)
        self._indexers = []
        # perf: in multi-phase mode at the same bandwidth AND same
        # sample_tilt, harvest phase 1's L+geom-only precomputations and
        # share with phases 2+ to skip ~7.5 s/phase of redundant build cost.
        # Solo phase, mixed bandwidths, or mixed sample_tilts keep the
        # legacy path. Code-review iter-4 finding: gating on bandwidth
        # alone made cross-tilt configs crash inside Tier1Indexer's
        # assert_compatible instead of falling back to per-phase build.
        first = self.config.phases[0]
        share_eligible = (
            len(self.config.phases) > 1
            and all(ph.bandwidth == first.bandwidth for ph in self.config.phases)
            and all(ph.sample_tilt_deg == first.sample_tilt_deg
                    for ph in self.config.phases)
        )
        shared: Optional[SharedSphericalTables] = None
        for i, ph in enumerate(self.config.phases):
            master = read_sht_master(ph.sht_file, device=self._device)
            ix = Tier1Indexer(
                geom=self._geom,
                master=master,
                device=self._device,
                bandwidth=ph.bandwidth,
                flip_y=self.config.flip_y,
                cc_fp64=ph.cc_fp64,
                sample_tilt_deg=ph.sample_tilt_deg,
                circmask=ph.circmask,
                gausbckg=ph.gausbckg,
                nregions=ph.nregions,
                normed=ph.normed,
                refine=ph.refine,
                shared_tables=shared,  # None for phase 1 (and solo runs)
            )
            self._indexers.append(ix)
            # After phase 1 builds normally, harvest its (L+geom)-only state
            # for phases 2+. SharedSphericalTables.assert_compatible inside
            # subsequent Tier1Indexer.__init__ guards against silent
            # cross-phase sample_tilt mismatch.
            if shared is None and share_eligible and i == 0:
                shared = SharedSphericalTables.from_indexer(ix)
        self._detector_params_cached = dict(detector_params)

    def _index_single_phase(
        self,
        indexer: Tier1Indexer,
        kind: str,
        payload,
        progress_callback,
    ) -> IndexResult:
        batch_size = self.config.batch_size or self._auto_batch_size(indexer)

        def run(b: int) -> IndexResult:
            if kind == "h5":
                return indexer.index_h5oina(
                    payload, batch_size=b,
                    progress_callback=progress_callback,
                    cancel_check=self._cancel_check,
                )
            if kind == "array":
                return self._index_array_via_indexer(
                    indexer, payload, b, progress_callback,
                )
            if kind == "h5_dataset":
                h5_path, dataset_path = payload
                return self._index_h5_dataset_via_indexer(
                    indexer, h5_path, dataset_path, b, progress_callback,
                )
            raise ValueError(f"Unknown indexing kind: {kind!r}")

        # On CUDA, wrap in OOM-retry; on CPU there is no OOM mechanism we
        # can recover from cleanly, so just call directly.
        if self._device.type == "cuda":
            return run_with_oom_retry(run, batch_size)
        return run(batch_size)

    def _index_h5_dataset_via_indexer(
        self,
        indexer: Tier1Indexer,
        h5_path: str,
        dataset_path: str,
        batch_size: int,
        progress_callback,
    ) -> IndexResult:
        """Streaming HDF5 dataset indexer — reads one batch at a time."""
        t0 = time.perf_counter()
        with h5py.File(h5_path, "r") as f:
            dset = f[dataset_path]
            n_total = int(dset.shape[0])
            all_eulers = torch.empty(n_total, 3, dtype=torch.float32)
            all_scores = torch.empty(n_total, dtype=torch.float32)
            for start in range(0, n_total, batch_size):
                self._raise_if_cancelled()
                end = min(start + batch_size, n_total)
                chunk = np.asarray(dset[start:end], dtype=np.float32)
                pat = torch.from_numpy(chunk).to(self._device)
                eulers, scores = indexer._index_batch(pat)
                all_eulers[start:end] = eulers.cpu()
                all_scores[start:end] = scores.cpu()
                if progress_callback:
                    progress_callback(
                        f"Indexed {end}/{n_total} patterns",
                        end / n_total,
                    )
        return IndexResult(
            euler_xyz=all_eulers,
            score=all_scores,
            phase_id=torch.ones(n_total, dtype=torch.int8),
            runtime_seconds=time.perf_counter() - t0,
        )

    def _index_with_phases(
        self,
        kind: str,
        payload,
        progress_callback,
    ) -> IndexResult:
        """Run all phases, select the per-pixel winner by score.

        For single-phase configs this just returns the one indexer's result.

        For multi-phase, interleaves at the BATCH level: read each batch
        from the H5 file ONCE, run it through every phase indexer, then
        move to the next batch. This avoids the 3x H5 read overhead and
        2x preprocessing waste of the previous "loop over phases" design.
        Empirically: 3-phase L=88 throughput went from ~344 pat/s
        (per-phase loop) to ~600 pat/s (batch-level interleave) on the
        7050 dataset.

        The phases must share the same preprocessing flags (circmask,
        gausbckg, nregions) for the SHT-input prep to be reusable across
        them. If they differ, we fall back to the per-phase loop.
        """
        if len(self._indexers) == 1:
            res = self._index_single_phase(
                self._indexers[0], kind, payload, progress_callback,
            )
            return res

        # Decide which strategy to use. Batch-interleaved sharing requires
        # all phases to use the same preprocessing settings — otherwise the
        # ``prep`` tensor produced for one phase wouldn't be valid input
        # for another.
        first = self._indexers[0]
        all_same_preproc = all(
            (ix.circmask == first.circmask
             and ix.gausbckg_on == first.gausbckg_on
             and ix.nregions_n == first.nregions_n
             and ix.normed == first.normed
             and ix.bandwidth == first.bandwidth)
            for ix in self._indexers[1:]
        )

        if all_same_preproc:
            return self._index_with_phases_interleaved(
                kind, payload, progress_callback,
            )

        # Fallback: per-phase loop (heterogeneous preprocessing).
        runtime0 = time.perf_counter()
        per_phase: list[IndexResult] = []
        for i, ix in enumerate(self._indexers):
            self._raise_if_cancelled()
            phase_name = self.config.phases[i].name
            wrapped_cb = (
                (lambda msg, frac, _i=i, _n=phase_name: progress_callback(
                    f"[phase {_i+1}/{len(self._indexers)} {_n}] {msg}", frac,
                ))
                if progress_callback else None
            )
            res = self._index_single_phase(ix, kind, payload, wrapped_cb)
            per_phase.append(res)
        scores = torch.stack([r.score for r in per_phase], dim=0)
        eulers = torch.stack([r.euler_xyz for r in per_phase], dim=0)
        winner = scores.argmax(dim=0)
        n = winner.shape[0]
        idx = torch.arange(n)
        best_eulers = eulers[winner, idx]
        best_scores = scores[winner, idx]
        phase_id = (winner + 1).to(torch.int8)
        return IndexResult(
            euler_xyz=best_eulers, score=best_scores, phase_id=phase_id,
            runtime_seconds=time.perf_counter() - runtime0,
        )

    def _index_with_phases_interleaved(
        self,
        kind: str,
        payload,
        progress_callback,
    ) -> IndexResult:
        """Batch-level multi-phase: read each batch once, run all phases,
        pick winner. Cuts H5 I/O + preprocessing + (full) SHT cost from
        N_phases x to 1 x at L=88, keeping only the per-phase cc + decode.

        Each phase's Tier1Indexer still has its own ``_l_active`` /
        ``_master_coefs_a`` / etc. set up; we just feed them the same
        preprocessed patterns instead of redoing the read+prep cycle.
        """
        import h5py
        from .pipeline.preprocessing import gausbckg, nregions, circmask

        runtime0 = time.perf_counter()
        n_phases = len(self._indexers)
        first = self._indexers[0]
        batch_size = self.config.batch_size or self._auto_batch_size(first)

        # Resolve payload to a streaming pattern source.
        if kind == "h5":
            # _open_pattern_dataset returns (n_total, dset, ds_path, file).
            # Capture the REAL file handle that `dset` belongs to and close
            # that one. The previous code opened a SECOND h5py.File (closed
            # later) while `dset` referenced the helper's handle — leaking one
            # open HDF5 file per whole-file index.
            n_total, dset, _, h5_handle = first._open_pattern_dataset(str(payload))
            close_handle = True
        elif kind == "array":
            patterns_np = np.asarray(payload)
            n_total = int(patterns_np.shape[0])
            dset = patterns_np
            h5_handle = None
            close_handle = False
        elif kind == "h5_dataset":
            h5_path_str, dataset_path = payload
            h5_handle = h5py.File(h5_path_str, "r")
            dset = h5_handle[dataset_path]
            n_total = int(dset.shape[0])
            close_handle = True
        else:
            raise ValueError(f"Unknown indexing kind: {kind!r}")

        # Output buffers (CPU-side) for the per-pixel winner selection.
        all_eulers = torch.empty(n_total, 3, dtype=torch.float32)
        all_scores = torch.empty(n_total, dtype=torch.float32)
        all_phase_id = torch.empty(n_total, dtype=torch.int8)

        # perf A4: prefetch next batch's H5 read on a worker thread so
        # the disk I/O + uint8->float32 conversion overlaps with current
        # batch GPU compute. H5 read alone is ~3-5ms per batch; with GPU
        # compute at ~200+ms per batch (interleaved 3-phase), this is
        # essentially free overlap.
        from concurrent.futures import ThreadPoolExecutor

        def _read_chunk(s, e):
            return np.asarray(dset[s:e], dtype=np.float32)

        ex = ThreadPoolExecutor(max_workers=1)
        starts = list(range(0, n_total, batch_size))
        # Prime: submit the first batch's read.
        first_end = min(starts[0] + batch_size, n_total)
        fut = ex.submit(_read_chunk, starts[0], first_end)

        try:
            for batch_idx, start in enumerate(starts):
                self._raise_if_cancelled()
                end = min(start + batch_size, n_total)
                pat_np = fut.result()
                # Submit next batch's read in background while we compute.
                if batch_idx + 1 < len(starts):
                    next_start = starts[batch_idx + 1]
                    next_end = min(next_start + batch_size, n_total)
                    fut = ex.submit(_read_chunk, next_start, next_end)
                patterns = torch.from_numpy(pat_np).to(self._device)

                # Preprocess ONCE — flags are identical across phases (we
                # checked in the caller). All indexers will see the same
                # ``prep`` input.
                prep = first._run_preprocessing(patterns)

                # perf Phase 5 EXPERIMENT (iter-14, REJECTED):
                # tried CUDA streams to parallelize the 3 phases. Result was
                # CATASTROPHIC: 835s (19x slower) AND oracle G2 FAIL (Al
                # phase fraction drifted from 72% to 44%, far past 2%
                # tolerance). Cause: cuFFT actually serializes on its
                # workspace, AND the per-stream contention triggered a race
                # in the shared rDen / sparse-rs2cc tables (built into
                # SharedSphericalTables in iter-3) when multiple phases
                # accessed them simultaneously. Reverted the original loop.
                # See the perf notes Phase 5 final.
                phase_eulers = []   # list of (b, 3) on device
                phase_scores = []   # list of (b,)
                for i, ix in enumerate(self._indexers):
                    eulers_i, scores_i = ix._index_prepped_batch(prep)
                    phase_eulers.append(eulers_i)
                    phase_scores.append(scores_i)

                stacked_scores = torch.stack(phase_scores, dim=0)   # (P, b)
                stacked_eulers = torch.stack(phase_eulers, dim=0)   # (P, b, 3)
                winner = stacked_scores.argmax(dim=0)               # (b,)
                bsz = winner.shape[0]
                bidx = torch.arange(bsz, device=self._device)
                best_eul = stacked_eulers[winner, bidx]             # (b, 3)
                best_sco = stacked_scores[winner, bidx]             # (b,)

                all_eulers[start:end] = best_eul.cpu()
                all_scores[start:end] = best_sco.cpu()
                all_phase_id[start:end] = (winner + 1).to(torch.int8).cpu()

                if progress_callback:
                    progress_callback(
                        f"Indexed {end}/{n_total} patterns (all {n_phases} phases)",
                        end / n_total,
                    )
        finally:
            # wait=True so any in-flight prefetch finishes BEFORE we close
            # the H5 file. Otherwise a worker thread can read from a closed
            # h5py.Dataset on an exception path and segfault. Cost of the
            # wait is at most one ~5ms read (the worker is processing the
            # next batch while we exit the loop).
            ex.shutdown(wait=True)
            if close_handle and h5_handle is not None:
                h5_handle.close()

        return IndexResult(
            euler_xyz=all_eulers, score=all_scores, phase_id=all_phase_id,
            runtime_seconds=time.perf_counter() - runtime0,
        )

    def _index_array_via_indexer(
        self,
        indexer: Tier1Indexer,
        patterns: np.ndarray,
        batch_size: int,
        progress_callback,
    ) -> IndexResult:
        """Mini-loop equivalent to ``Tier1Indexer.index_h5oina`` for ndarray input."""
        n_total = patterns.shape[0]
        all_eulers = torch.empty(n_total, 3, dtype=torch.float32)
        all_scores = torch.empty(n_total, dtype=torch.float32)
        t0 = time.perf_counter()
        for start in range(0, n_total, batch_size):
            self._raise_if_cancelled()
            end = min(start + batch_size, n_total)
            chunk = np.asarray(patterns[start:end], dtype=np.float32)
            pat = torch.from_numpy(chunk).to(self._device)
            eulers, scores = indexer._index_batch(pat)
            all_eulers[start:end] = eulers.cpu()
            all_scores[start:end] = scores.cpu()
            if progress_callback:
                progress_callback(
                    f"Indexed {end}/{n_total} patterns",
                    end / n_total,
                )
        return IndexResult(
            euler_xyz=all_eulers,
            score=all_scores,
            phase_id=torch.ones(n_total, dtype=torch.int8),
            runtime_seconds=time.perf_counter() - t0,
        )

    def _auto_batch_size(self, indexer: Tier1Indexer) -> int:
        """Pick a safe batch size for the current device + bandwidth.

        Uses ``runtime.compute_safe_batch`` which queries free GPU memory
        and computes a per-bandwidth budget. CPU returns a fixed small
        batch (latency-friendly). On OOM the batch is halved by
        ``run_with_oom_retry`` in ``_index_single_phase``.

        On CPU, also re-emits a clear "you're on CPU, this will be slow"
        warning at index time (the construction-time warning may have
        scrolled off the user's terminal by now). Estimated runtime is
        based on the fixed ~16 pat/s CPU rate observed at L=68.
        """
        if self._device.type != "cuda":
            warnings.warn(
                "SphericalGPUBackend running on CPU — expect ~3-5 pat/s at L=68. "
                "A 10000-pattern scan will take ~30-60 minutes. For production, "
                "use a CUDA-capable GPU or fall back to the EMSphInx CPU backend "
                "(which uses native OpenMP and is significantly faster than "
                "PyTorch CPU for this kernel).",
                RuntimeWarning,
                stacklevel=2,
            )
        return compute_safe_batch(
            bandwidth=indexer.bandwidth,
            runtime=self.runtime,
            user_override=None,
        )
