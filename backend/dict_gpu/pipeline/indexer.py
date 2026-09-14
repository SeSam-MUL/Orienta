"""End-to-end orchestrator for the GPU dictionary indexer.

Wires master loader, grid sampler, dict generator (Path B wrapper for now),
NCC kernel, optional PCA, optional tiling, and CrystalMap back-mapping
into a single function that returns an IndexingResult layout-compatible
with the existing CPU path.
"""
from __future__ import annotations
import functools
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from backend.dict_gpu.exceptions import GpuDictError, MasterPatternError
from backend.dict_gpu.runtime import detect_gpu, vram_budget_bytes
from backend.dict_gpu._pcadi.pca import GpuPCA
from backend.dict_gpu._pcadi.knn import gemm_topk_ncc
from backend.dict_gpu._pcadi.master_to_dict import gpu_master_to_dict
from backend.dict_gpu.pipeline.master_loader import (
    load_master_or_dict,
    open_streamed_dict,
    MasterPayload,
)
from backend.dict_gpu.pipeline.grid import sample_orientations
from backend.dict_gpu.pipeline.tiling import (
    compute_tile_size,
    compute_stream_tile_size,
    iter_tiles,
    resident_peak_bytes,
    should_stream,
    stream_bytes_per_entry,
    MIN_STREAM_TILE,
)
from backend.dict_gpu.pipeline.dict_source import ArrayDictSource, DictSource
from backend.dict_gpu.pipeline.output import build_crystal_map


# Late-import IndexingResult to avoid module-load circularity
def _indexing_result_types():
    from indexing_controller import IndexingResult, IndexingMethod
    return IndexingResult, IndexingMethod


logger = logging.getLogger(__name__)


def _normalise_rows(X: torch.Tensor) -> torch.Tensor:
    X = X - X.mean(dim=1, keepdim=True)
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _normalise_rows_(X: torch.Tensor) -> torch.Tensor:
    """:func:`_normalise_rows` without the two full-size temporaries.

    Same arithmetic in the same order, so the result is bit-identical — the
    difference is that the streamed tile is centred and scaled in the buffer it
    arrived in instead of being copied twice. At a 128x156 detector that is
    160 KB per entry not allocated, which is the difference between a tile of
    19,000 entries and a tile of 8,000.

    Only ever called on a tile this module just created; never on a caller's
    tensor.
    """
    X -= X.mean(dim=1, keepdim=True)
    X /= X.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return X


def _assert_finite_tile(tile: torch.Tensor, start: int, n_dict: int,
                        *, stage: str = "source") -> None:
    """Fail loud on NaN/Inf before the arithmetic hides where it came from.

    The resident path checks the whole dictionary twice — once as loaded and
    once after normalisation — and streaming does the same per tile, which
    covers exactly the same entries and can say which ones.

    Checked through the row sums rather than ``torch.isfinite(tile).all()``,
    which is the same verdict for far less memory: NaN and Inf both survive a
    sum, and a row of 19,968 fp32 intensities cannot overflow into one. The
    elementwise form was measured at 1008 B per entry of transient workspace —
    nearly twice the tile itself, on the exact code path that exists because
    memory is tight. The row sums cost 4.
    """
    row_sums = tile.sum(dim=1)
    if torch.isfinite(row_sums).all():
        return
    bad = ~torch.isfinite(row_sums)
    n_bad = int(bad.sum().item())
    first = start + int(torch.argmax(bad.int()).item())
    where = (f"entries [{start}, {start + tile.shape[0]}) of {n_dict}; "
             f"first at index {first}")
    if stage == "source":
        raise ValueError(
            f"Dictionary contains {n_bad} non-finite patterns in {where}. "
            "Likely cause: a corrupted dictionary file or a master pattern "
            "that projects all-zero patterns at certain orientations. "
            "Regenerate the dictionary, or skip the bad orientations upstream."
        )
    raise ValueError(
        f"Dictionary normalisation produced non-finite values in {where}. "
        "The entries were finite as read, so mean-subtract + L2 normalisation "
        "surfaced something structurally wrong in the source data "
        "(the norm is clamped, so it cannot produce this on its own). "
        "Regenerate the dictionary."
    )


def _iter_streamed_tiles(
    source: DictSource,
    tile: int,
    device,
    *,
    pattern_dim: int,
    keep_cols: Optional[torch.Tensor],
    cancel_check=None,
):
    """Yield ``(slice, normalised tile on device)``, one tile resident at a time.

    The previous tile's storage is released as soon as the consumer moves on,
    so the loop's high-water mark is one tile — not the dictionary.
    """
    n_dict = source.n_entries
    for sl in iter_tiles(n_dict, tile):
        if cancel_check is not None and cancel_check():
            from indexing_controller import CancelledIndexingError
            raise CancelledIndexingError(
                "Dictionary-GPU indexing cancelled by user")
        block = source.read_block(sl.start, sl.stop)
        t = torch.from_numpy(block).to(device, non_blocking=True)
        t = t.reshape(t.shape[0], pattern_dim)
        if keep_cols is not None:
            t = t[:, keep_cols]
        t = t.contiguous()
        _assert_finite_tile(t, sl.start, n_dict)
        t = _normalise_rows_(t)
        _assert_finite_tile(t, sl.start, n_dict, stage="normalised")
        yield sl, t
        # Drop OUR reference the moment the consumer hands control back. A
        # suspended generator keeps its locals alive, so without this the
        # previous tile is still held while the next block lands on the card —
        # measured as two tiles resident at every hand-over, i.e. double the
        # peak the tile size promises.
        t = None


def _is_master_pattern(obj) -> bool:
    """An EBSDMasterPattern has phase, data, and projection (master-specific)."""
    return all(hasattr(obj, a) for a in ("phase", "data", "projection"))


def _is_ebsd_signal_dict(obj) -> bool:
    """A kikuchipy EBSD signal that's been treated as a dictionary
    (i.e. its xmap.rotations are the dictionary rotations).

    Distinguished from EBSDMasterPattern by the absence of `.projection`.
    """
    if not hasattr(obj, "data") or not hasattr(obj, "axes_manager"):
        return False
    return not hasattr(obj, "projection")


def _rotations_from_ebsd_signal(sig):
    """Try several common locations where rotations might live on an
    EBSD-as-dictionary signal."""
    from orix.quaternion import Rotation
    # 1. xmap?
    if hasattr(sig, "xmap") and getattr(sig.xmap, "rotations", None) is not None:
        return sig.xmap.rotations
    # 2. metadata path used by simulation/dictionary_generator.py
    md = getattr(sig, "metadata", None)
    if md is not None:
        rots_data = None
        try:
            rots_data = md.Simulation.EBSD_dictionary.rotations
        except Exception:
            pass
        if rots_data is not None:
            return Rotation(np.asarray(rots_data))
    raise MasterPatternError(
        "EBSD-signal-as-dictionary supplied but rotations could not be located "
        "(checked sig.xmap.rotations and "
        "sig.metadata.Simulation.EBSD_dictionary.rotations)."
    )


def _run_dictionary_index(
    experimental_signal,
    master_pattern_or_path,
    detector,
    *,
    _opened_sources: Optional[list] = None,
    angular_step_deg: float = 1.5,
    metric: str = "ncc",
    keep_n: int = 1,
    selection_mask: Optional[np.ndarray] = None,
    # Detector mask in kikuchipy's convention: True = EXCLUDE (the dark corners
    # outside the phosphor disc), False = keep. Same meaning and same effect as
    # the ``signal_mask`` argument of ``EBSD.dictionary_indexing``.
    signal_mask: Optional[np.ndarray] = None,
    use_pca: Union[str, bool] = "auto",
    # Only reached when memory pressure forces PCA (see below). 256 components
    # keep ~32 % of the dictionary variance; in that truncated subspace the
    # correlation is dominated by structure common to every pattern of the
    # phase, so the score it reports is not comparable to the full one.
    # Measured on LoGainNi (7440 px, 2 deg grid, 100,347 orientations) against
    # Hough as ground truth, which indexes this dataset at 0.27 deg median fit:
    #     no PCA    median 0.95 deg to Hough, 95.9 % <2 deg, reports 0.469
    #     k=256     median 1.06 deg to Hough, 93.9 % <2 deg, reports 0.603
    #     k=1024    median 0.94 deg to Hough, 97.3 % <2 deg, reports 0.564
    # The orientations survive truncation; the reported NCC does not.
    pca_components: int = 1024,
    use_quantization: Union[str, bool] = "auto",
    vram_budget_gb: Optional[float] = None,
    # Stream the dictionary past the GPU one tile at a time instead of
    # uploading all of it. "auto" streams exactly when the resident path would
    # not fit the VRAM budget (see tiling.should_stream), which keeps every
    # dictionary that used to fit on the path it was validated on. True forces
    # streaming; False forbids it and fails loud if the dictionary cannot fit.
    stream_dictionary: Union[str, bool] = "auto",
    # Entries per streamed tile. None = sized from the free VRAM. Present so a
    # test can force an awkward split (a tile that does not divide the
    # dictionary) and so a user with a busy card can pin it.
    stream_tile_entries: Optional[int] = None,
    progress_callback=None,
    cancel_check=None,
):
    if metric != "ncc":
        raise NotImplementedError(f"metric={metric!r}; only 'ncc' is supported in MVP")

    # Phase-timer for "why is dict indexing slow?" diagnostics. Logs each
    # major phase's wall-clock duration so we know whether the bottleneck
    # is file I/O, GPU upload, PCA, the NCC kernel, or post-processing.
    import time
    _t0 = time.perf_counter()
    _last_ts = _t0
    _phase_times: dict[str, float] = {}

    def _phase_done(name: str):
        nonlocal _last_ts
        now = time.perf_counter()
        dt = now - _last_ts
        _phase_times[name] = dt
        if dt >= 0.05:  # don't spam the log with sub-50ms phases
            msg = f"Dict-GPU [timing]: {name} = {dt:.2f}s"
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)
        _last_ts = now

    # Late-import to avoid pulling indexing_controller at module load.
    def _check_cancel():
        if cancel_check is not None and cancel_check():
            from indexing_controller import CancelledIndexingError
            raise CancelledIndexingError("Dictionary-GPU indexing cancelled by user")

    gpu = detect_gpu()
    if not gpu.available:
        raise GpuDictError(
            "gpu_dictionary_index_patterns called but no CUDA device available."
        )
    device = torch.device("cuda")
    # The high-water mark is a per-process statistic, so it has to be reset for
    # "how much did THIS run need" to mean anything — otherwise a big earlier
    # run makes every later one look like it paged to system memory.
    torch.cuda.reset_peak_memory_stats()
    # Snapshot the PRE-UPLOAD VRAM budget. The "PCA for memory pressure"
    # decision below compares this against the dict size — calling
    # vram_budget_bytes() after the dict is already on the GPU would
    # report only ~free-budget and trip PCA on every run regardless of
    # the actual card capacity.
    initial_budget_bytes = int(max(1.0, gpu.vram_free_gb - 2.0) * 1e9)

    def _p(msg):
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)

    # ---- 1. Resolve master / dict-cache --------------------------------------
    # Capture the source file path (when caller supplied one) so the
    # downstream pattern-match dialog can lazy-load specific dictionary
    # entries from disk without needing the multi-GB tensor in memory.
    dict_source_path: str | None = None
    if isinstance(master_pattern_or_path, (str, Path)):
        dict_source_path = str(Path(master_pattern_or_path).resolve())
        # Try to open it without reading the patterns first. A 24 GB
        # dictionary handed over as a path has no business passing through
        # host RAM on its way to a card that will only ever hold a tile of it.
        payload = (open_streamed_dict(master_pattern_or_path)
                   or load_master_or_dict(master_pattern_or_path))
        if getattr(payload, "kind", None) == "master":
            # It was a MASTER, not a dictionary. `dict_path` means "read
            # dictionary entry i out of this file"; pointed at a master there
            # are no entries to read, and the pattern-match dialog would
            # index into the wrong thing instead of saying the dictionary is
            # not in memory. GPU twin of the CPU fix in the multi-phase loop.
            dict_source_path = None
    elif _is_master_pattern(master_pattern_or_path):
        payload = MasterPayload(
            kind="master",
            master=master_pattern_or_path,
            dict_patterns=None,
            dict_rotations=None,
            phase=getattr(master_pattern_or_path, "phase", None),
        )
    elif _is_ebsd_signal_dict(master_pattern_or_path):
        # already-generated dictionary (from mp.get_patterns elsewhere)
        sig = master_pattern_or_path
        # Recover the source path from kikuchipy's tmp_parameters when
        # the caller pre-loaded the dictionary via kp.load(). Without
        # this the pattern-match dialog can't lazy-load patterns from
        # disk and shows "Dictionary not in memory" after the run.
        try:
            tp = getattr(sig, "tmp_parameters", None)
            if tp is not None:
                folder = getattr(tp, "folder", "")
                filename = getattr(tp, "filename", "")
                if filename:
                    base = Path(folder or ".") / filename
                    # kikuchipy strips the extension from `filename`, but
                    # filenames like '..._2.0deg' contain a dot that breaks
                    # Path.with_suffix (which would replace '.0deg' → '.h5').
                    # Append the extension as a plain string instead.
                    for ext in ("", ".h5", ".hdf5"):
                        cand = Path(str(base) + ext)
                        if cand.is_file():
                            dict_source_path = str(cand.resolve())
                            break
        except Exception:
            pass
        patterns = np.asarray(sig.data, dtype=np.float32)
        if patterns.ndim == 4 and patterns.shape[0] == 1:
            patterns = patterns.squeeze(0)
        rots = _rotations_from_ebsd_signal(sig)
        # Extract the phase from the EBSD signal's xmap so the resulting
        # CrystalMap carries proper point_group for IPF colouring.
        # PhaseList has no len() — iterate ids instead.
        sig_phase = None
        try:
            xm = getattr(sig, "xmap", None)
            if xm is not None:
                ids = list(xm.phases_in_data.ids)
                if len(ids) >= 1:
                    sig_phase = xm.phases_in_data[ids[0]]
                else:
                    ids = list(xm.phases.ids)
                    if len(ids) >= 1:
                        sig_phase = xm.phases[ids[0]]
        except Exception:
            pass
        payload = MasterPayload(
            kind="dict_cache",
            master=None,
            dict_patterns=patterns,
            dict_rotations=rots,
            phase=sig_phase,
        )
    else:
        raise MasterPatternError(
            f"master_pattern_or_path must be a path, an EBSDMasterPattern, "
            f"or a kikuchipy EBSD signal containing pre-generated dictionary "
            f"patterns; got {type(master_pattern_or_path).__name__}"
        )
    _phase_done("1. resolve master/dict + load from disk")

    # ---- 2. Sample orientations ----------------------------------------------
    if payload.kind == "master":
        rotations = sample_orientations(
            payload.master.phase.point_group, angular_step_deg
        )
        _p(f"Dict-GPU: sampled {rotations.size} rotations at {angular_step_deg} deg")
    else:
        rotations = payload.dict_rotations
        _p(f"Dict-GPU: legacy dict cache supplied with {rotations.size} rotations")
    _phase_done("2. sample/load orientations")

    # ---- 3. Generate the dictionary, or open it as a stream source -----------
    # A master still has to be projected, and that happens on the device, so a
    # generated dictionary is resident by construction. A pre-generated one is
    # only *opened* here: its shape is all we need to size the run, and the
    # entries themselves are fetched per tile below.
    dict_source: Optional[DictSource] = None
    dict_data = None
    if payload.kind == "master":
        dict_data = gpu_master_to_dict(
            payload.master, rotations, detector, energy=20.0, device=device
        )
        n_dict = dict_data.shape[0]
        pat_h, pat_w = dict_data.shape[-2:]
    else:
        dict_source = (payload.dict_source
                       if getattr(payload, "dict_source", None) is not None
                       else ArrayDictSource(payload.dict_patterns))
        if _opened_sources is not None:
            _opened_sources.append(dict_source)
        n_dict, pat_h, pat_w = dict_source.shape
    pattern_dim = pat_h * pat_w

    # Circular detector mask (kikuchipy convention: True = EXCLUDE). Applied by
    # dropping the masked columns from BOTH the dictionary and the experimental
    # patterns before normalisation, which is exactly what kikuchipy's
    # ``signal_mask`` does on the CPU path — the mean and the L2 norm are then
    # taken over the disc only, instead of being dragged down by the dark
    # corners that carry no diffraction signal.
    keep_cols = None
    feat_dim = pattern_dim
    if signal_mask is not None:
        sm = np.asarray(signal_mask, dtype=bool)
        if sm.shape != (pat_h, pat_w):
            raise GpuDictError(
                f"signal_mask shape {sm.shape} != detector shape {(pat_h, pat_w)}"
            )
        keep_flat = ~sm.reshape(-1)
        feat_dim = int(keep_flat.sum())
        if feat_dim == 0:
            raise GpuDictError("signal_mask excludes every detector pixel")
        if feat_dim < pattern_dim:
            keep_cols = torch.from_numpy(np.flatnonzero(keep_flat)).to(device)
            _p(f"Dict-GPU: signal mask keeps {feat_dim}/{pattern_dim} detector px")
        else:
            signal_mask = None
    # Sync CUDA so the timing reflects the actual GPU upload completion
    # rather than the async-launch return. Without this the upload time
    # gets attributed to the next phase.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dict_data is not None:
        _p(f"Dict-GPU: dictionary tensor {tuple(dict_data.shape)} on {device}")
    else:
        _p(f"Dict-GPU: dictionary {n_dict:,} x {pat_h}x{pat_w} "
           f"({n_dict * pattern_dim * 4 / 1e9:.2f} GB fp32) from "
           f"{dict_source.origin}")
    _phase_done("3. dictionary tensor -> GPU")

    # ---- 4. VRAM budget + auto choices ---------------------------------------
    # Two distinct budgets here:
    #   - decision_budget: how much VRAM was available BEFORE we uploaded
    #     the dictionary. Used for the PCA-memory-pressure question.
    #     Without this snapshot, vram_budget_bytes() called HERE would
    #     return ~free - 2GB AFTER the 8GB dict upload, reporting 1.5GB
    #     and tripping PCA on every run regardless of card size.
    #   - tile_budget: current free VRAM, used to size the matching tiles.
    if vram_budget_gb is not None:
        decision_budget = int(vram_budget_gb * 1e9)
        tile_budget = decision_budget
    else:
        decision_budget = initial_budget_bytes
        tile_budget = vram_budget_bytes()
    budget = tile_budget  # kept name for compute_tile_size compatibility
    fp32_bytes = n_dict * feat_dim * 4

    # Estimate how many experimental patterns this run will match.
    # Used for the PCA cost/benefit decision below.
    n_exp_est = 0
    if selection_mask is not None:
        n_exp_est = int(selection_mask.sum())
    else:
        es_data = getattr(experimental_signal, "data", None)
        if es_data is not None and es_data.ndim >= 4:
            n_exp_est = int(es_data.shape[0] * es_data.shape[1])

    # Resident or streamed? One rule, and it only ever looks at whether the
    # whole-dictionary path would fit. A generated dictionary is already on the
    # device and has nowhere to stream from, so it is resident by definition.
    can_stream = dict_source is not None
    if stream_dictionary is True:
        if not can_stream:
            raise GpuDictError(
                "stream_dictionary=True but this dictionary was generated on "
                "the device from a master pattern, so there is nothing to "
                "stream it from. Pass a pre-generated dictionary file."
            )
        stream_effective = True
    elif stream_dictionary is False:
        stream_effective = False
    else:  # "auto"
        stream_effective = can_stream and should_stream(
            n_dict, pattern_dim, feat_dim, decision_budget)
    if stream_effective:
        _p(f"Dict-GPU: streaming the dictionary — the resident path would "
           f"peak near {resident_peak_bytes(n_dict, pattern_dim, feat_dim) / 1e9:.1f} GB "
           f"for a {fp32_bytes / 1e9:.1f} GB dictionary, budget is "
           f"{decision_budget / 1e9:.1f} GB")
    elif can_stream and stream_dictionary is False and \
            should_stream(n_dict, pattern_dim, feat_dim, decision_budget):
        _p("Dict-GPU: streaming was disabled by the caller, but the resident "
           f"path is expected to need "
           f"{resident_peak_bytes(n_dict, pattern_dim, feat_dim) / 1e9:.1f} GB "
           f"of a {decision_budget / 1e9:.1f} GB budget — expect an OOM or a "
           "crawl through system memory.")

    # PCA decision (two independent triggers; either alone enables PCA):
    #
    # 1. Memory pressure — fp32 dictionary doesn't fit in the VRAM budget
    #    that was available BEFORE we uploaded it. PCA shrinks the
    #    matching footprint from d -> k floats per pattern. Streaming solves
    #    the same problem without approximating anything, so when the
    #    dictionary is streamed this trigger is off and the score the user
    #    reads stays an exact NCC.
    #
    # 2. Speed payoff — matching cost is O(n_exp * n_dict * d) without PCA
    #    vs O(n_exp * n_dict * k) with PCA, but the SVD fit pays a fixed
    #    upfront cost (~30-40s on a 4070 for our dictionary sizes).
    #    Empirically the crossover sits around 1000 experimental patterns;
    #    below that the SVD setup dwarfs the matching savings. We use a
    #    conservative threshold to avoid flipping back and forth near the
    #    boundary.
    #
    # Auto-trigger only when EITHER memory pressure OR clear speed win
    # exists. Explicit True/False from the caller overrides everything.
    # PCA is a LOSSY approximation. It may be traded for memory — without it a
    # large dictionary simply does not fit — but not for speed it does not buy.
    # This threshold used to auto-enable PCA for any selection >= 2000 px,
    # which is every full map, so every full-map dictionary run was
    # approximated whether or not it needed to be. Measured on LoGainNi
    # (7440 px, 100,347 orientations): 37.5 s without PCA, 39.0 s with k=256,
    # 75.4 s with k=1024 — no win to pay for, and the truncated run reports a
    # score that is not comparable to the full one (see pca_components).
    PCA_PAYOFF_MIN_PATTERNS = float("inf")
    need_pca_for_memory = fp32_bytes > decision_budget and not stream_effective
    worth_pca_for_speed = n_exp_est >= PCA_PAYOFF_MIN_PATTERNS

    if use_pca is True:
        use_pca_effective = True
        _p(f"Dict-GPU: PCA forced ON by caller")
    elif use_pca is False:
        use_pca_effective = False
        _p(f"Dict-GPU: PCA forced OFF by caller")
    else:  # "auto"
        use_pca_effective = need_pca_for_memory or worth_pca_for_speed
        if use_pca_effective:
            reasons = []
            if need_pca_for_memory:
                reasons.append(f"memory pressure ({fp32_bytes/1e9:.1f} GB dict > {decision_budget/1e9:.1f} GB pre-upload budget)")
            if worth_pca_for_speed:
                reasons.append(f"large selection ({n_exp_est} patterns)")
            _p(f"Dict-GPU: PCA enabled — " + " AND ".join(reasons))
        else:
            _p(f"Dict-GPU: PCA off — dictionary fits in VRAM "
               f"({fp32_bytes/1e9:.1f} GB <= {decision_budget/1e9:.1f} GB pre-upload budget), "
               f"so the score is an exact NCC.")

    use_quant_effective = (
        use_quantization is True
        or (use_quantization == "auto" and not use_pca_effective and fp32_bytes // 2 > budget)
    )
    if use_quant_effective:
        # MVP: skip quantisation path; log the would-do
        logger.info("Dict-GPU: quantisation skipped in MVP; FP16 GEMM will handle it")
        use_quant_effective = False

    if use_pca_effective and stream_effective:
        raise GpuDictError(
            "PCA was requested together with a streamed dictionary. The PCA "
            "basis is fitted on the whole normalised dictionary at once, and "
            "the dictionary is being streamed precisely because it does not "
            f"fit ({fp32_bytes / 1e9:.1f} GB against a "
            f"{decision_budget / 1e9:.1f} GB budget). Fitting it on a sample "
            "instead would report scores from a basis nobody asked for. Run "
            "with use_pca=False — streaming needs no approximation — or on a "
            "dictionary that fits."
        )

    # ---- 5. Normalise dictionary (and optionally PCA) ------------------------
    # Streamed runs normalise each tile as it lands (see _iter_streamed_tiles),
    # so none of this is reached: there is no whole dictionary to normalise.
    dict_flat = dict_norm = dict_proj = None
    if stream_effective:
        _phase_done("5a. dictionary normalisation + finite check")
        pca = None
    else:
        if dict_data is None:
            # Resident path with a pre-generated dictionary: this is the one
            # upload, the thing streaming exists to avoid.
            dict_data = torch.from_numpy(
                dict_source.read_block(0, n_dict)).to(device).float()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _p(f"Dict-GPU: dictionary tensor {tuple(dict_data.shape)} on {device}")
        dict_flat = dict_data.reshape(n_dict, pattern_dim)
        if keep_cols is not None:
            dict_flat = dict_flat[:, keep_cols]
        dict_flat = dict_flat.contiguous()
        # Fail loud on NaN/Inf in the source dictionary BEFORE the SVD step
        # below — torch.linalg.svd surfaces NaN as a generic CUSOLVER error
        # ("CUSOLVER_STATUS_INVALID_VALUE") with no hint about the real cause.
        # NaN typically comes from a corrupted dictionary file (some entries
        # were never written), or — much more commonly — patterns that were
        # all-zero before _normalise_rows. _normalise_rows clamps the norm so
        # it can't produce NaN itself; if we see NaN here, the upstream data
        # is bad and the user needs to know which patterns.
        if not torch.isfinite(dict_flat).all():
            n_bad = int((~torch.isfinite(dict_flat)).any(dim=1).sum().item())
            raise ValueError(
                f"Dictionary contains {n_bad} non-finite patterns out of "
                f"{n_dict}. Likely cause: a corrupted dictionary file or a "
                "master pattern that projects all-zero patterns at certain "
                "orientations. Regenerate the dictionary, or skip the bad "
                "orientations upstream."
            )
        dict_norm = _normalise_rows(dict_flat)
        if not torch.isfinite(dict_norm).all():
            # Post-normalise NaN means the mean-subtraction surfaced a degenerate
            # row (constant pattern → mean = pattern → result is all-zero, then
            # norm is clamped to 1e-8 and the row is essentially noise but still
            # finite). If we still see NaN here, something is structurally
            # broken in the source data.
            raise ValueError(
                "Dictionary normalisation produced non-finite values after "
                "mean-subtract + L2 normalisation. This indicates patterns "
                "with NaN/Inf intensities in the source dictionary. "
                "Regenerate the dictionary."
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _phase_done("5a. dictionary normalisation + finite check")
        pca = None
        if use_pca_effective:
            k = min(pca_components, n_dict, feat_dim)
            pca = GpuPCA(n_components=k).fit(dict_norm)
            dict_proj = pca.transform(dict_norm)
            # PCA breaks unit-norm; renormalise so NCC interpretation holds
            dict_proj = _normalise_rows(dict_proj)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _p(f"Dict-GPU: PCA reduced dictionary {feat_dim} -> {k}")
            _phase_done("5b. PCA fit + transform")
        else:
            dict_proj = dict_norm

    # ---- 6. Prepare experimental patterns ------------------------------------
    exp_arr = np.asarray(experimental_signal.data, dtype=np.float32)
    if exp_arr.ndim == 4:
        n_rows, n_cols, eh, ew = exp_arr.shape
    else:
        raise GpuDictError(
            f"experimental_signal.data must be 4D (n_rows, n_cols, h, w); got {exp_arr.shape}"
        )
    if (eh, ew) != (pat_h, pat_w):
        raise GpuDictError(
            f"detector shape {(pat_h, pat_w)} does not match experimental "
            f"pattern shape {(eh, ew)}; resize/PC before indexing"
        )

    if selection_mask is None:
        selection_mask = np.ones((n_rows, n_cols), dtype=bool)
    elif selection_mask.shape != (n_rows, n_cols):
        raise GpuDictError(
            f"selection_mask shape {selection_mask.shape} != ({n_rows}, {n_cols})"
        )

    sel_idx = np.argwhere(selection_mask)
    n_sel = sel_idx.shape[0]
    if n_sel == 0:
        raise GpuDictError("selection_mask selects zero pixels")
    _p(f"Dict-GPU: matching {n_sel} experimental patterns against {n_dict} dict")

    exp_selected = exp_arr[sel_idx[:, 0], sel_idx[:, 1]].reshape(n_sel, pattern_dim)
    exp_t = torch.from_numpy(exp_selected).to(device)
    if keep_cols is not None:
        exp_t = exp_t[:, keep_cols].contiguous()
    exp_norm = _normalise_rows(exp_t)
    if pca is not None:
        exp_proj = pca.transform(exp_norm)
        exp_proj = _normalise_rows(exp_proj)
    else:
        exp_proj = exp_norm
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _phase_done("6. experimental patterns -> GPU + normalise + PCA-project")

    # ---- 7. Tiled top-k NCC --------------------------------------------------
    # Both modes feed the same merge loop below; they differ only in where a
    # tile comes from. Resident: a view on the dictionary that is already on
    # the card. Streamed: a block read from the host array or the file,
    # normalised on arrival and dropped when the next one lands.
    dtype_bytes = 4  # FP32; FP16 lives only inside the autocast region of gemm_topk_ncc
    if stream_effective:
        feature_dim = feat_dim
        # Measured HERE and not at step 4, for the same reason the resident
        # branch re-measures below: step 6 has just put the experimental
        # patterns on the card, and both exp_t and exp_norm stay there until
        # the end of the run. At a full 28,086-pixel map with a 128x156
        # detector that is 2 x 2.24 = 4.49 GB, which
        # ``compute_stream_tile_size`` does not model — it counts per-entry
        # costs only. Sized from the step-4 budget, the tile would spend that
        # memory twice and survive only on the 2 GB reserve inside
        # ``vram_budget_bytes``, with no re-tiling fallback if it did not.
        stream_budget = (int(vram_budget_gb * 1e9) if vram_budget_gb is not None
                         else vram_budget_bytes())
        if stream_tile_entries is not None:
            tile = int(stream_tile_entries)
            if tile <= 0:
                raise GpuDictError(
                    f"stream_tile_entries must be > 0, got {stream_tile_entries}")
            tile = min(tile, n_dict)
        else:
            tile = compute_stream_tile_size(
                n_dict, pattern_dim, feat_dim, n_sel, stream_budget,
                chunk_entries=getattr(dict_source, "chunk_entries", 1),
            )
        if tile <= 0:
            # Not even a minimal tile plus the experimental buffers fit. Say
            # so with the numbers instead of letting CUDA report the single
            # allocation that happened to be last. Both numbers are taken at
            # the moment of failure: the free VRAM now (the experimental
            # patterns are already on the card), and what the smallest tile
            # this loop would bother with actually costs — the same arithmetic
            # the tile size was refused by, not a budget divided by a count.
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            min_tile_bytes = MIN_STREAM_TILE * stream_bytes_per_entry(
                pattern_dim, feat_dim, n_sel)
            raise GpuDictError(
                f"Dictionary indexing does not fit on this card even one tile "
                f"at a time. Dictionary {fp32_bytes / 1e9:.2f} GB "
                f"({n_dict:,} entries x {feat_dim} px), "
                f"{n_sel:,} experimental patterns need "
                f"{n_sel * feat_dim * 4 / 1e9:.2f} GB on their own, free VRAM "
                f"{free_gb:.2f} GB, budget {stream_budget / 1e9:.2f} GB, smallest "
                f"usable tile {MIN_STREAM_TILE} entries "
                f"({min_tile_bytes / (1 << 20):.1f} MiB). "
                f"Index a smaller region, close what else is using the GPU, or "
                f"regenerate the dictionary at a coarser angular step."
            )
        _n_tiles_est = max(1, -(-n_dict // tile))
        _p(f"Dict-GPU: streaming {n_dict:,} entries in {_n_tiles_est} tiles of "
           f"{tile:,} ({tile * pattern_dim * 4 / 1e9:.2f} GB per tile, "
           f"{stream_budget / 1e9:.2f} GB budget after the experimental upload)")
        tile_iter = _iter_streamed_tiles(
            dict_source, tile, device, pattern_dim=pattern_dim,
            keep_cols=keep_cols, cancel_check=cancel_check,
        )
    else:
        feature_dim = dict_proj.shape[1]
        # Measured HERE and not at step 4, because in this mode the dictionary
        # is already on the card: what is left for a matching slab is the free
        # VRAM AFTER the upload, which is what this call reports and what the
        # code did before the upload moved. Sizing the slab from the pre-upload
        # budget would hand gemm_topk_ncc a slab whose score matrix does not
        # fit — 14 GB at a full 28,086-pixel map.
        resident_budget = (int(vram_budget_gb * 1e9) if vram_budget_gb is not None
                           else vram_budget_bytes())
        tile = compute_tile_size(n_dict, feature_dim, dtype_bytes, resident_budget)
        tile_iter = ((sl, dict_proj[sl]) for sl in iter_tiles(n_dict, tile))

    # Aggregate global top-k across tiles.
    # Pre-declare all per-iteration variables to None so the post-loop
    # `del` cleanup below is unconditional — see VRAM-release notes there.
    best_scores = torch.full((n_sel, keep_n), -float("inf"), device=device)
    best_indices = torch.full((n_sel, keep_n), -1, dtype=torch.int64, device=device)
    tile_dict = s_tile = i_tile = merged_scores = merged_idx = top_pos = None
    s_pad = i_pad = None
    # Live throughput. Every experimental pattern is compared against every
    # dictionary tile, so after k of n tiles the work done is n_sel * k/n
    # patterns' worth — that is the honest running rate, and it converges to
    # the final one. Reported at most once a second so a fast run does not
    # drown the log.
    _match_t0 = time.perf_counter()
    _n_tiles = max(1, -(-n_dict // tile))   # ceil
    _last_report = _match_t0
    # Counted by hand rather than with enumerate(): enumerate caches the
    # (index, item) tuple it last produced, and that reference keeps the
    # previous tile alive while the generator builds the next one. Measured on
    # the 8 GB Al dictionary, 20,000-entry tiles: peak 3.36 GB with enumerate
    # against 1.79 GB without, on tiles of 1.6 GB — exactly one tile's worth.
    _tile_i = 0
    for sl, tile_dict in tile_iter:
        _tile_i += 1
        _check_cancel()
        # Tile may be smaller than keep_n on the very last slab; clamp k.
        k_tile = min(keep_n, tile_dict.shape[0])
        s_tile, i_tile = gemm_topk_ncc(exp_proj, tile_dict, k=k_tile, use_fp16=True)
        i_tile = i_tile + sl.start  # offset to global indices
        # If the tile produced fewer than keep_n columns, pad to keep_n with -inf so
        # the merge below has consistent shapes.
        if k_tile < keep_n:
            pad = keep_n - k_tile
            s_pad = torch.full((n_sel, pad), -float("inf"), device=device)
            i_pad = torch.full((n_sel, pad), -1, dtype=torch.int64, device=device)
            s_tile = torch.cat([s_tile, s_pad], dim=1)
            i_tile = torch.cat([i_tile, i_pad], dim=1)
        # Merge tile top-k with running best
        merged_scores = torch.cat([best_scores, s_tile], dim=1)
        merged_idx = torch.cat([best_indices, i_tile], dim=1)
        best_scores, top_pos = torch.topk(merged_scores, k=keep_n, dim=1)
        best_indices = merged_idx.gather(1, top_pos)
        # Let go of this tile BEFORE the generator fetches the next one. A
        # streamed tile is a real allocation, not a view, and the loop variable
        # would otherwise still hold it while the next block lands — two tiles
        # resident at the hand-over, i.e. twice the peak the tile size promises.
        tile_dict = s_tile = i_tile = merged_scores = merged_idx = top_pos = None
        _now = time.perf_counter()
        if _now - _last_report >= 1.0 and _tile_i < _n_tiles:
            _last_report = _now
            _done = n_sel * _tile_i / _n_tiles
            _p(f"Dict-GPU: matching {_tile_i}/{_n_tiles} tiles "
               f"· {_done / max(_now - _match_t0, 1e-9):,.0f} pat/s")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _match_dt = time.perf_counter() - _match_t0
    _p(f"Dictionary: {n_sel:,} patterns in {_match_dt:,.1f} s "
       f"· {n_sel / max(_match_dt, 1e-9):,.0f} pat/s (GPU, {n_dict:,} entries)"
       if _match_dt >= 1e-3 else
       f"Dictionary: {n_sel:,} patterns in {_match_dt * 1e3:.1f} ms (GPU)")
    # Did the run actually stay on the card? On Windows the NVIDIA driver
    # silently spills to system memory instead of failing, and the only sign is
    # a run that crawls — the user's 8 GB dictionary "worked" that way. We
    # cannot switch the fallback off from here, but we can say when the
    # high-water mark got close enough to the card that it must have been used.
    if torch.cuda.is_available():
        _peak_gb = torch.cuda.max_memory_allocated() / 1e9
        _total_gb = gpu.vram_total_gb
        if _total_gb > 0 and _peak_gb > 0.95 * _total_gb:
            _p(f"⚠ Dict-GPU: peak allocation {_peak_gb:.2f} GB against "
               f"{_total_gb:.2f} GB of VRAM — the driver was almost certainly "
               f"paging to system memory, which is why this was slow.")
        else:
            logger.info("Dict-GPU [vram]: peak allocation %.2f GB of %.2f GB",
                        _peak_gb, _total_gb)
    _phase_done("7. tiled top-k NCC matching")

    # In the PCA subspace the score is the cosine between the PROJECTIONS: the
    # components orthogonal to the retained ones — exactly where the
    # model-vs-experiment mismatch lives — are dropped from both vectors before
    # the angle is taken, so it reads high. Measured on LoGainNi, 2 deg grid:
    # 0.5669 truncated against 0.4685 for the same winning orientations.
    #
    # PCA switches itself on under memory pressure, which depends on the
    # dictionary size and on what else is using the card. Without this, the
    # same file indexed twice can report scores 0.1 apart with nothing in the
    # output saying why. So re-score the winners at full dimension: keep_n rows
    # per pixel, one gather and one row-wise dot product, and the number the
    # user reads is an NCC again whatever the search ran in.
    if pca is not None:
        _check_cancel()
        flat_idx = best_indices.reshape(-1).clamp_min(0)
        exact = torch.empty_like(best_scores).view(-1)
        # Three (tile, feat_dim) fp32 tensors are live at once — the two
        # gathers and their product — so the tile must be sized from the
        # feature dimension, not from a row count. A fixed 1e6 rows would be
        # 8 GB at feat_dim 3600, on a run that reached this branch precisely
        # because memory was tight. Measured transient with the divisor below:
        # 384 MiB at feat_dim 3600 and 379 MiB at 19,968, i.e. the budget is
        # met, and at 3600 features it is ~5900 rows per pass.
        RESCORE_BUDGET_BYTES = 384 << 20
        rescore_tile = max(1, RESCORE_BUDGET_BYTES // (feat_dim * 4 * 3))
        for s in range(0, flat_idx.numel(), rescore_tile):
            e = min(s + rescore_tile, flat_idx.numel())
            rows = flat_idx[s:e]
            exp_rows = exp_norm[torch.div(
                torch.arange(s, e, device=device), keep_n, rounding_mode="floor")]
            exact[s:e] = (dict_norm[rows] * exp_rows).sum(dim=1)
        exact = exact.reshape(best_scores.shape)
        # -1 marks a padding slot from a short tile; leave those at -inf.
        best_scores = torch.where(best_indices >= 0, exact, best_scores)
        # The candidates came back ordered by the truncated score, which is not
        # the order the exact one puts them in. Re-sort so "best match" means
        # the best by the number we now report. This only re-ranks within the
        # keep_n the search found — it cannot recover a candidate the truncated
        # search never shortlisted.
        best_scores, order = torch.sort(best_scores, dim=1, descending=True)
        best_indices = best_indices.gather(1, order)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _p("Dict-GPU: re-scored top matches at full dimension "
           f"(PCA k={pca.n_components} was used for the search only)")
        _phase_done("7b. exact NCC re-score of top-k")

    # ---- 8. Build CrystalMap (BEFORE releasing the GPU tensors!) -----------
    # build_crystal_map consumes best_indices / best_scores (it converts
    # them to CPU numpy internally) so it must run BEFORE the VRAM
    # cleanup block below. Earlier versions del'd these first → crash.
    from orix.crystal_map import Phase, PhaseList
    src_phase = payload.phase
    if src_phase is None or getattr(src_phase, "point_group", None) is None:
        raise GpuDictError(
            "Dictionary source has no phase metadata with point_group. "
            "IPF colouring requires crystallographic symmetry. "
            "Regenerate the dictionary from a master pattern whose CIF "
            "carries a valid space/point group, or pick a different "
            "dictionary file."
        )
    phase = Phase(
        name=getattr(src_phase, "name", None) or "Phase",
        point_group=src_phase.point_group,
        space_group=getattr(src_phase, "space_group", None),
        structure=getattr(src_phase, "structure", None),
    )
    phase_list = PhaseList(phases=[phase], ids=[1])
    xmap = build_crystal_map(
        top_indices=best_indices,
        top_scores=best_scores,
        rotations=rotations,
        phase_list=phase_list,
        selection_mask=selection_mask,
        original_shape=(n_rows, n_cols),
    )

    # ---- Release the dictionary tensors NOW that build_crystal_map is done.
    # Without these explicit dels the multi-GB dict_data / dict_norm /
    # dict_proj refs survive until function exit, and PyTorch's caching
    # allocator keeps the underlying VRAM blocks in its free list rather
    # than returning them to the driver — the user sees ~1.9 GB still
    # "in use" even after invalidate(). Free them now so the next
    # indexing run starts with the full free-VRAM budget.
    #
    # CRITICAL: the tile-loop variables (tile_dict, s_tile, i_tile,
    # merged_scores, merged_idx, top_pos, s_pad, i_pad) stay bound after
    # the for-loop exits — Python doesn't scope them to the iteration.
    # `tile_dict = dict_proj[sl]` is a *view* on dict_proj's storage, so
    # without releasing tile_dict first the underlying VRAM block survives
    # the `del dict_proj` via the view ref.

    # Capture metadata BEFORE the cleanup block.
    scores_top1_np = best_scores[:, 0].detach().cpu().numpy().astype(np.float32)
    pca_components_count = pca.n_components if pca is not None else None

    # Capture VRAM before the cleanup so we can show whether our del +
    # empty_cache actually returns memory to the driver, or if the
    # residual usage is CUDA-runtime overhead (kernels, cuBLAS, cuSOLVER
    # libraries) that can't be released without ending the process.
    vram_before_cleanup = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free_b, total_b = torch.cuda.mem_get_info()
        vram_before_cleanup = free_b / 1e9

    # A streamed run holds the file open until here; the host array it may
    # wrap instead belongs to the caller and is left alone.
    if dict_source is not None:
        dict_source.close()
        tile_iter = None

    # Release tile-loop view holders FIRST (they pin dict_proj's storage).
    del tile_dict, s_tile, i_tile, merged_scores, merged_idx, top_pos
    del s_pad, i_pad
    # Release the matching results (already drained into scores_top1_np
    # above and into the xmap via build_crystal_map).
    del best_scores, best_indices
    # Release the big dictionary + experimental tensors.
    del dict_data, dict_flat, dict_norm, dict_proj
    del exp_t, exp_norm, exp_proj, exp_arr, exp_selected
    if pca is not None:
        del pca
    if torch.cuda.is_available():
        # Force Python GC before empty_cache so any cyclic refs holding
        # GPU tensors get collected (cuBLAS / autograd graphs sometimes
        # create these).
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_b_after, _ = torch.cuda.mem_get_info()
        vram_after = free_b_after / 1e9
        if vram_before_cleanup is not None:
            _p(f"Dict-GPU [vram]: cleanup released "
               f"{(vram_after - vram_before_cleanup) * 1024:.0f} MiB "
               f"({vram_before_cleanup:.2f} GB -> {vram_after:.2f} GB free)")

    # ---- 9. Build IndexingResult ---------------------------------------------
    IndexingResult, IndexingMethod = _indexing_result_types()
    _phase_done("8. build CrystalMap + result")

    # Print a sorted summary of where the wall-clock went. Helps the user
    # answer "why did this take 50 s?" without rerunning with a profiler.
    total = time.perf_counter() - _t0
    summary_parts = [f"Dict-GPU [timing summary]: total={total:.2f}s ="]
    for name, dt in sorted(_phase_times.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * dt / total if total > 0 else 0.0
        summary_parts.append(f"\n  {pct:5.1f}%  {dt:6.2f}s  {name}")
    summary_msg = "".join(summary_parts)
    if progress_callback:
        progress_callback(summary_msg)
    logger.info(summary_msg)

    return IndexingResult(
        xmap=xmap,
        selection_mask=selection_mask,
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.DICTIONARY,
        confidence_scores=scores_top1_np,
        metadata={
            "compute_mode": "gpu",
            "device_name": gpu.name,
            "vram_total_gb": gpu.vram_total_gb,
            "use_pca": use_pca_effective,
            "pca_components": pca_components_count,
            "use_quantization": use_quant_effective,
            # Was the dictionary streamed past the card, and in what size
            # tiles? Worth recording: it is the difference between a score
            # searched at full dimension and one searched in a PCA subspace,
            # and between a run that fitted and one that paged.
            "streamed": bool(stream_effective),
            "stream_tile_entries": int(tile) if stream_effective else None,
            "n_dictionary": n_dict,
            "angular_step_deg": angular_step_deg,
            "metric": metric,
            "keep_n": keep_n,
            # Source path for lazy pattern reload — pattern-match dialog
            # reads single dictionary entries from this file by index
            # rather than holding the full dictionary in memory.
            "dict_path": dict_source_path,
        },
    )


@functools.wraps(_run_dictionary_index)
def run_dictionary_index(*args, **kwargs):
    # Public entry point; owns nothing but the cleanup.
    #
    # A streamed run holds an HDF5 handle open for the length of the match. The
    # run closes it on the way out, but a raise in the middle of a seven-tile
    # match would leave it to the garbage collector — and on a failing run that
    # means "whenever the traceback holding the frame is released", which this
    # project has already measured to be much later than it looks (task F: the
    # dictionary tensors were held by a traceback). On Windows an open handle
    # also keeps the file locked against regenerating it.
    #
    # So the source registers itself the moment it is opened and this closes
    # whatever was opened, whichever way the run ends. Closing twice is a
    # no-op, so the explicit close inside stays where it is — it still runs
    # before the result is built, on the successful path.
    #
    # functools.wraps is load-bearing, not decoration: inspect.signature and
    # inspect.getsource both follow __wrapped__, and several tests in this
    # project read this function's source and signature.
    opened: list = []
    try:
        return _run_dictionary_index(*args, _opened_sources=opened, **kwargs)
    finally:
        for src in opened:
            try:
                src.close()
            except Exception:  # pragma: no cover — defensive
                logger.debug("could not close dictionary source", exc_info=True)
