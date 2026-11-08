"""Batch Manager — orchestrates batch indexing with memory safety and checkpointing.

Key improvement over v1: Each file is fully loaded → preprocessed → indexed → unloaded.
No more assumption that a signal is pre-loaded.
"""
import gc
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from display_names import display_stem

import numpy as np

from backend.api.services.batch_queue import BatchQueue
from backend.api.services.checkpoint_writer import (
    CheckpointPairingChanged, CheckpointWriter, read_checkpoint_pairing,
)
from backend.api.services import scan_geometry as scan_geometry_service
from backend.api.services.memory_guardian import MemoryGuardian, MemoryStatus
from backend.api.services.pc_sources import Note
from backend.api.services.preflight_check import PreFlightChecker, PreFlightReport

logger = logging.getLogger(__name__)


class HexFrameAverageRefused(ValueError):
    """Frame averaging of a hexagonal scan was refused or failed. ``code`` is the one the interface translates.

    A hexagonal scan must never be indexed unaveraged when the user asked for averaging, so unlike a square
    file (where a failed averaging is noted as ``FAILED`` and the run goes on) this fails the file."""

    code = "hexFrameAverageRefused"


def _load_signal(file_path: str):
    """Load an EBSD signal from file. Returns the kikuchipy EBSD signal.

    The loader's copy of the scan geometry (``Signal.scan_geometry``) stays on the signal: the run reads it
    once (``scan_geometry.geometry_of_signal``) and hands it on. A batch signal is never cropped or copied,
    so the copy cannot go stale; taking it off here would make a hexagonal scan read as a square one from
    its navigation axes."""
    from safe_loader import load_ebsd_safe
    logger.info("Loading signal: %s", display_stem(file_path))
    return load_ebsd_safe(file_path)


#: The formats that are written by ``export_ang_ctf`` (one failure there is reported under ``ang_ctf``).
_ANG_CTF_FORMATS = ("ang", "ctf", "ang_square")


def _export_plan(formats, checkpoint_path: str):
    """What to ask ``export_all`` for one file, and what to tell the user about it.

    Returns ``(formats, grid_kind, notes)``. The grid is read from the checkpoint, as the writers do. The
    resampled square ``.ang`` exists for a hexagonal scan only: for a square file it is left out (asking
    for it would refuse the whole ``.ang``/``.ctf`` export of that file) and the notes say so; for a
    hexagonal scan whose ``.ctf`` or square ``.ang`` is asked for, the notes carry the sentence those
    files carry about the resampling.

    A note is the English sentence (``str``) with a ``code`` and ``params`` (``pc_sources.Note``): the
    sentence is what the files on disk carry, the code and the numbers are what the page words it with."""
    from backend.api.services import ang_export
    from backend.api.services.checkpoint_writer import read_checkpoint_geometry
    from backend.api.services.result_exporter import DEFAULT_EXPORT_FORMATS
    wanted = list(formats) if formats is not None else list(DEFAULT_EXPORT_FORMATS)
    geometry = read_checkpoint_geometry(checkpoint_path)
    notes: List[str] = []
    if geometry is None or not geometry.is_hex:
        if "ang_square" in wanted:
            wanted.remove("ang_square")
            notes.append(Note("No resampled square .ang was written: this scan is already on a square grid.",
                              "noSquareAngForSquareScan"))
        return wanted, "square", notes
    if "ctf" in wanted or "ang_square" in wanted:
        from backend.api.services.hex_resample import square_resampling
        try:
            resampling = square_resampling(geometry)
            assumed = not geometry.shift_confirmed
            notes.append(Note(
                ang_export.resampling_note(resampling, geometry), "resampledFromHex",
                step=float(f"{resampling.step:.10g}"), source_points=int(resampling.n_source),
                written=int(resampling.n_out), unused=int(resampling.n_unused), direction_assumed=assumed,
                # the side the odd rows were placed on, named only when the sentence says it is an assumption
                side=("right" if float(geometry.odd_row_shift) > 0 else "left") if assumed else None))
        except ValueError:
            pass     # the exporter refuses the same scan with the same reason: it is in the record's errors
    return wanted, "hex", notes


def _eds_notes(file_path: str, formats, include_eds: bool) -> List[str]:
    """The note for a light ``.h5`` that cannot hold the EDS data it was asked to hold.

    ``include_eds`` is on by default. A pattern file that is not HDF5 (an EDAX ``.up1`` / ``.up2``, any
    other format) carries no EDS: the light file is written without it, and the record says so (code
    ``edsNotCopied``)."""
    import h5py
    if not (include_eds and "h5_light" in formats):
        return []
    path = Path(file_path)
    if path.is_file() and not h5py.is_hdf5(str(path)):
        return [Note("The light .h5 holds no EDS data: this scan's patterns are in another file format, "
                     "which carries none.", "edsNotCopied")]
    return []


def _unwritten(formats, results: dict, errors: dict) -> Dict[str, str]:
    """The formats asked for whose file does not exist and whose failure no error names.

    ``export_ang_ctf`` turns a failed ``.ang``, ``.ctf`` or square ``.ang`` into a missing path and a log
    line; without this the record would show an output that is simply absent. The reason is a
    ``pc_sources.Note`` with the code ``fileNotWritten``."""
    missing = {}
    for fmt in formats:
        if results.get(fmt) is not None or fmt in errors:
            continue
        if fmt in _ANG_CTF_FORMATS and "ang_ctf" in errors:
            continue
        missing[fmt] = Note("The file was not written; the backend log says why.", "fileNotWritten")
    return missing


def _items(values, default_code: str = "other") -> List[dict]:
    """``{code, params, text}`` of each note: the code and numbers the page words it with, and the English
    sentence it falls back to. A plain string has no code of its own and gets ``default_code``, with itself
    as the ``detail``."""
    out = []
    for value in values:
        code = getattr(value, "code", None)
        if code is None:
            out.append({"code": default_code, "params": {"detail": str(value)}, "text": str(value)})
        else:
            out.append({"code": code, "params": dict(value.params), "text": str(value)})
    return out


def _error_items(errors: Optional[dict]) -> Optional[Dict[str, dict]]:
    """``_items`` of the reasons of a record, by format; ``None`` when nothing failed.

    A reason without a code of its own is an exception's text (``str(e)``): the page puts a translated
    lead-in before it and shows the text as technical detail. A failed export as a whole is ``export``."""
    if not errors:
        return None
    return {fmt: _items([reason], "exportFailed" if fmt == "export" else "writerFailed")[0]
            for fmt, reason in errors.items()}


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


def _average_hex_neighbours(signal, window, geometry, topology) -> int:
    """Average each pattern of a hexagonal scan with the points within ``(window - 1) / 2`` steps, in place.

    The same rules as the viewer's ``/api/ebsd/frame-average`` on a hexagonal dataset: an odd window up to
    ``frame_average.HEX_MAX_WINDOW``, patterns that fit twice into ``frame_average.HEX_MAX_BYTES``, a lazy signal
    computed in place. Returns the number of rings averaged (0 for a window of one: nothing changed). Raises
    :class:`HexFrameAverageRefused` for a refusal and for any failure."""
    from backend.api.services import frame_average
    try:
        size = int(window)
        if size < 1:
            raise HexFrameAverageRefused(f"A window of {size} is not a window.")
        if size > frame_average.HEX_MAX_WINDOW:
            raise HexFrameAverageRefused(
                f"A window of {size} is too large for a hexagonal scan: the largest is "
                f"{frame_average.HEX_MAX_WINDOW}.")
        if size > 1 and size % 2 == 0:
            raise HexFrameAverageRefused(
                f"A window of {size} is not available on a hexagonal scan: the neighbourhood grows by whole "
                f"rings around a point. Use an odd size: 3 averages a point with its 6 neighbours, "
                f"5 with the 18 points within two steps.")
        rings = (size - 1) // 2
        if rings < 1:
            return 0                                               # a window of one averages nothing
        nbytes = int(np.prod(signal.data.shape, dtype=np.int64)) * int(signal.data.dtype.itemsize)
        if nbytes > frame_average.HEX_MAX_BYTES:
            raise HexFrameAverageRefused(
                f"Averaging a hexagonal scan holds its {nbytes / 1024 ** 3:.2f} GiB of patterns twice in "
                f"memory, more than the {frame_average.HEX_MAX_BYTES / 1024 ** 3:.2f} GiB allowed.")
        if getattr(signal, "_lazy", False):
            signal.compute(show_progressbar=False)                 # materialise in place, as the viewer does
        signal.data[...] = frame_average.average_neighbour_patterns(
            signal.data, topology, rings, geometry.valid_mask)
        return rings
    except HexFrameAverageRefused:
        raise
    except Exception as exc:  # noqa: BLE001 - whatever broke, an unaveraged hexagonal file is not indexed
        raise HexFrameAverageRefused(f"Frame averaging failed: {exc}") from exc


def _apply_preprocessing(signal, config: Dict, geometry=None):
    """Apply preprocessing steps to the signal in-place.

    ``geometry`` is the ``ScanGeometry`` of the file (the run reads it once, when it loads the file). A
    hexagonal one makes frame averaging run on the neighbour table of the lattice, and a refusal or a failure
    there raises :class:`HexFrameAverageRefused`; a square or unknown geometry (``None``) takes kikuchipy's
    rectangular window, whose failure is noted as ``FAILED`` while the run goes on.

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
        static_bg_row: int | None  (with static_bg_col: one reference pattern;
        static_bg_col: int | None   both unset: mean of all patterns of the file)
        gauss_background: bool (for spherical)
        nregions_ahe: int (for spherical AHE)
    """
    applied: Dict[str, Any] = {}
    if not config:
        return signal, applied

    # Frame averaging
    if config.get("frame_averaging", False):
        window = config.get("frame_averaging_window", 3)
        from backend.api.services import scan_neighbours
        topology = scan_neighbours.hex_topology(geometry) if geometry is not None else None
        if topology is not None:
            rings = _average_hex_neighbours(signal, window, geometry, topology)
            logger.info("Applied hexagonal frame averaging (window=%s, %d rings)", window, rings)
            # What was averaged is rings of neighbours, not a rectangle of window x window points. The
            # label says so, so that a checkpoint never reads as the rectangular window of a square scan.
            if rings < 1:
                applied["frame_averaging"] = "hex, 0 rings (nothing averaged)"
            else:
                applied["frame_averaging"] = f"hex, {rings} ring{'' if rings == 1 else 's'}"
        else:
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
                # No reference named -> the mean of all patterns of THIS file;
                # row AND col named -> that one pattern; one of them -> error.
                # (Defaulting to pattern (0,0) imprints that pattern's bands,
                # inverted, on every pattern of the map.)
                from backend.api.services.static_background import (
                    scan_average, validate_reference,
                )
                ref = validate_reference(
                    config.get("static_bg_row"), config.get("static_bg_col"))
                if ref is None:
                    bg = scan_average(signal)
                    label = "scan_average"
                else:
                    row, col = ref
                    nav = signal.axes_manager.navigation_shape  # (n_cols, n_rows)
                    n_cols = int(nav[0]) if len(nav) >= 1 else 1
                    n_rows = int(nav[1]) if len(nav) >= 2 else 1
                    if not (0 <= row < n_rows and 0 <= col < n_cols):
                        raise ValueError(
                            f"Static-background reference pattern ({row},{col}) "
                            f"is outside the scan {n_rows}x{n_cols}.")
                    bg = np.asarray(signal.inav[col, row].data).squeeze()
                    label = f"{row},{col}"
                signal.remove_static_background(operation="subtract", static_bg=bg)
                logger.info("Applied static background removal (reference: %s)", label)
                applied["background_removal"] = f"static@{label}"
        except Exception as e:
            logger.warning("Background removal failed: %s", e)
            applied["background_removal"] = f"FAILED: {e}"

    return signal, applied


# --- one calibration entry per FILE, also when two files share a stem -------------------
#
# The calibration store is keyed by dataset name (the file stem). A batch over
# ``A/scan.up1`` and ``B/scan.up1`` would use ONE entry for both: the second file was
# indexed at the first one's PC (or the one without a PC at the other's), and both wrote
# ``result_scan.*`` into the same folder. Within a batch, files whose stems collide get a
# name that carries a short hash of their canonical path; the others keep their stem.
_batch_names: Dict[str, Tuple[str, str]] = {}      # canonical path -> (dataset name, suffix)


def _canonical_file(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _path_suffix(canonical: str) -> str:
    import hashlib
    return "__" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


def _holds_default_variant(name: str, file_path) -> bool:
    """Whether the store entry ``name`` of ``file_path`` describes the file's default variant, the
    patterns batch v2 reads. The viewer registers a file it loaded as another variant (another
    measurement, the unprocessed patterns) under a name that carries it; the registry of loaded
    variants knows those names. No variant on record, or the file's default one (also when it was
    named explicitly), is the default."""
    from backend.api.services import variant_registry
    from backend.api.services.load_choices import inspect_file
    variant = variant_registry.variant_for(name)
    if variant is None:
        return True
    try:
        choices = inspect_file(str(file_path))
    except Exception:  # noqa: BLE001 - cannot be listed: it has no choices to differ by
        return True
    return choices is None or variant.is_default(choices)


def holds_default_measurement(name: str, file_path) -> bool:
    """Whether the store entry ``name`` of ``file_path`` holds the measurement batch v2 reads, in
    whichever pattern source. Two sources of one measurement are one acquisition on one detector;
    another measurement has a PC of its own. A file that cannot be listed answers False."""
    from backend.api.services import variant_registry
    from backend.api.services.load_choices import inspect_file
    variant = variant_registry.variant_for(name)
    if variant is None:
        return True
    try:
        choices = inspect_file(str(file_path))
    except Exception:  # noqa: BLE001 - cannot be listed: nothing says it is the same measurement
        return False
    default = choices.default_variant() if choices is not None else None
    return default is None or variant.measurement == default.measurement


def default_variant_entry_of_file(file_path, store=None) -> Optional[str]:
    """The name of the entry the store holds for ``file_path`` and its default variant, or None.

    A file the viewer holds as another variant has an entry of that variant (named after it); that
    one is not this file's entry for batch v2, whose data are the processed patterns of the first
    measurement. Views of a dataset (copies, crops) are never the file's own entry, as in
    ``CalibrationStore.name_of_file``. ``store``: the calibration store to ask (the app's by default)."""
    if store is None:
        from backend.api.services.calibration_store import calibration_store as store
    c = _canonical_file(file_path)
    for name, entry in store.get_all().items():
        if entry.file_key == c and not entry.is_view_of_parent and _holds_default_variant(name, file_path):
            return name
    return None


def variant_entry_of_file(file_path, store=None) -> Optional[str]:
    """The entry the viewer registered for ``file_path`` as a variant other than the default, or
    None (see :func:`default_variant_entry_of_file`)."""
    if store is None:
        from backend.api.services.calibration_store import calibration_store as store
    c = _canonical_file(file_path)
    for name, entry in store.get_all().items():
        if entry.file_key == c and not entry.is_view_of_parent and not _holds_default_variant(name, file_path):
            return name
    return None


def store_name_for_file(file_path) -> str:
    """The name under which the calibration store holds (or would hold) THIS file for batch v2.

    The store is keyed by name (the file stem), so two files with one stem need two names:
    the entry already registered for the file's canonical path wins (whatever it is
    called); else the stem when no entry has it; else - an entry of that stem belongs to
    another file, or to no file known (registered without a path) - the stem plus a short
    hash of the canonical path. A caller that shows or writes a PC of a file goes through
    this name, never through the bare stem.

    Batch v2 reads the file's default variant (the processed patterns), so the name is that of an
    entry holding it: an entry the viewer made for another variant of the same file (named after
    the variant) is not taken, it would label processed data as the unprocessed ones.
    """
    from backend.api.services.calibration_store import calibration_store
    c = _canonical_file(file_path)
    own = default_variant_entry_of_file(file_path)
    if own is not None:
        return own
    stem = display_stem(file_path)
    if calibration_store.get_entry(stem) is None:
        return stem
    return stem + _path_suffix(c)


def assign_batch_dataset_names(files) -> Dict[str, Tuple[str, str]]:
    """Name every file of a batch (see :func:`store_name_for_file`) and the suffix its
    OUTPUTS carry. Files whose stems collide within the batch, or whose store name is not
    the plain stem, get ``__<hash8>`` in their output names, so no two files overwrite each
    other; the others keep the plain name. Deterministic (the hash is of the canonical
    path) and remembered for the calibration, the exports and the guards. Returns
    ``{canonical path: (dataset name, output suffix)}`` for these files."""
    from collections import Counter
    canon = {}
    for f in files:
        canon.setdefault(_canonical_file(f), str(f))
    stems = Counter(display_stem(f) for f in canon.values())
    out = {}
    taken = set()
    for c, f in canon.items():
        stem = display_stem(f)
        name = store_name_for_file(f)
        if name in taken:                      # two files of the batch, one still-unregistered name
            name = stem + _path_suffix(c)
        taken.add(name)
        suffix = _path_suffix(c) if (stems[stem] > 1 or name != stem) else ""
        out[c] = (name, suffix)
    _batch_names.update(out)
    return out


def dataset_name_for(file_path) -> str:
    """The calibration name of ``file_path`` in the batch that is (or was last) set up."""
    return _batch_names.get(_canonical_file(file_path), (display_stem(file_path), ""))[0]


def name_suffix_for(file_path) -> str:
    return _batch_names.get(_canonical_file(file_path), (None, ""))[1]


def _register_signal_calibration(signal, file_path: str):
    """Register signal's detector/PC in CalibrationStore."""
    from backend.api.services.calibration_store import calibration_store
    dataset_name = dataset_name_for(file_path)
    if dataset_name not in calibration_store:
        calibration_store.register(dataset_name, signal, file_key=_canonical_file(file_path))


#: The spherical engines a batch request may ask for. Re-exported from
#: indexing_controller rather than restated: the interactive route, the
#: per-phase dispatcher and this one all have to agree on what the names are,
#: and two lists of two strings are exactly the kind of thing that drifts.
from indexing_controller import SPHERICAL_BACKENDS as _SPHERICAL_BACKENDS


def _get_detector_for_file(file_path: str, signal=None):
    """Get the best available detector for a file. Never returns None."""
    from backend.api.services.calibration_store import calibration_store
    from backend.api.services.pc_guard import PcMissingError, is_pc_missing, pc_missing_message
    dataset_name = dataset_name_for(file_path)
    if is_pc_missing(dataset_name, calibration_store, signal, _canonical_file(file_path)):
        # The store's detector would hold kikuchipy's placeholder PC.
        raise PcMissingError(pc_missing_message(dataset_name))
    detector = calibration_store.get_detector(dataset_name)
    if detector is not None and signal is not None:
        # The entry has to describe THIS file's patterns: another pattern shape means it
        # is another detector's (an entry of another file), and indexing would die far
        # from here or, with the same shape, use a PC that is not this file's.
        try:
            sig_shape = tuple(int(v) for v in signal.axes_manager.signal_shape[::-1])
            if tuple(int(v) for v in detector.shape) != sig_shape:
                from backend.api.services.pc_guard import DetectorMismatchError
                raise DetectorMismatchError(
                    f"The calibration held for '{dataset_name}' is for a "
                    f"{tuple(detector.shape)} detector, but the patterns of "
                    f"{display_stem(file_path)} are {sig_shape}: it belongs to another file. "
                    "Load this file again, or set its PC.")
        except DetectorMismatchError:
            raise
        except Exception:  # noqa: BLE001 - a stand-in signal without axes
            pass
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
    scan_geometry: Optional[Dict] = None,
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
    scan_geometry : the stamp (``ScanGeometry.to_dict()``) of the grid this file was loaded with, which the
        batch read once at the load. ``None`` (a direct call) reads it off ``signal``.

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
        # The stamp is this job's own file's geometry, not the viewer's dataset's (the default
        # of `scan_geometry` downstream): this file is never the active dataset. The batch reads it
        # once when it loads the file and passes it in; a direct call reads it off the signal.
        # Unknown is `None`, which the run reads as "not hexagonal" and never as a reason to look
        # at the viewer. Both backends get it: EMSphInx refuses a hexagonal scan by it.
        if scan_geometry is not None:
            _scan_stamp = scan_geometry
        else:
            try:
                from backend.api.services import scan_geometry as _scan_geometry
                _scan_stamp = _scan_geometry.geometry_of_signal(signal).to_dict()
            except Exception:  # noqa: BLE001 - a geometry that cannot be read is unknown
                logger.warning("could not read the scan geometry of %s; its hexagonal note is "
                               "left out", display_stem(file_path), exc_info=True)
                _scan_stamp = None
        if backend_choice == "spherical_gpu":
            result = spherical_gpu_index_patterns(
                h5_path=file_path,
                config=config,
                detector_params=det_params,
                progress_callback=progress_callback,
                scan_geometry=_scan_stamp,
                # This job's own signal: batch v2 loads its file into a signal of its own and never
                # activates it in the viewer, so the viewer's patterns are some other dataset's.
                signal=signal,
            )
        else:
            result = spherical_index_patterns(
                h5_path=file_path,
                config=config,
                detector_params=det_params,
                progress_callback=progress_callback,
                scan_geometry=_scan_stamp,
                # This job's file as it is on disk, never the viewer's processed
                # patterns - not even when the viewer shows this same file.
                use_viewer_patterns=False,
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
        from ebsd_utils import hough_phase_from_cif
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

        phase = hough_phase_from_cif(phase_path)
        original_stem = display_stem(phase_path)
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
    # What the run recorded about itself, carried to the checkpoint and from there to
    # the exports (every other export path writes the same two records): the citation
    # steps that ran and, for Hough, the reflector list each phase was indexed with.
    from backend.api.services.citations.provenance import get_steps
    _steps = get_steps(result)
    if _steps:
        metadata["citation_steps"] = [dict(s) for s in _steps]
    _md = getattr(result, "metadata", None)
    _hough = _md.get("hough_reflectors") if isinstance(_md, dict) else None
    if isinstance(_hough, dict) and _hough:
        metadata["hough_reflectors"] = _hough
    _caveats = _md.get("caveats") if isinstance(_md, dict) else None
    if isinstance(_caveats, list) and _caveats:
        metadata["caveats"] = _caveats

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
                # The lattice, from the phase itself or the file it came from.
                # Not finding one is not an error here: the exporter looks again
                # and refuses to write a placeholder if it still cannot.
                try:
                    from backend.api.services import ang_export
                    metadata["lattice_constants"] = list(
                        ang_export.resolve_phase_lattice(
                            p, phase_file=phase_path,
                            # a kikuchipy master's structure is in nm
                            structure_unit=("nm" if str(method).lower() == "dictionary"
                                            else "angstrom"))[0])
                except Exception as e:
                    logger.debug("no lattice for %s: %s", phase_name, e)
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
        #: batch id -> {source file: what was exported for it} (name suffix included)
        self.exports: Dict[str, Dict[str, Dict[str, Any]]] = {}

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

        Marks a run as active while it goes (reflector selections of Hough phases
        cannot be changed meanwhile), whatever way it ends.
        """
        import hough_reflectors
        token = hough_reflectors.begin_run("batch")
        try:
            return self._run_batch_sync_impl(batch_id, progress_callback)
        finally:
            hough_reflectors.end_run(token)

    def _run_batch_sync_impl(
        self,
        batch_id: str,
        progress_callback: Optional[Callable] = None,
    ):
        """The batch itself (see ``run_batch_sync``).

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
        # What this run calls each of its files (stems that collide get a path hash).
        assign_batch_dataset_names([j["file_path"] for j in self.queue.get_jobs(batch_id)])
        #: file -> {"dataset", "name_suffix", "outputs"}: what was exported for it
        self.exports[batch_id] = {}
        # The PC and tilts each file was indexed at, taken when the file is loaded: the
        # export at the end of the batch states THESE, not what the store holds by then
        # (a PC typed while the batch ran is not the PC of its orientations).
        pc_records: Dict[str, Optional[dict]] = {}
        current_signal = None
        current_file = None
        current_geometry = None          # ScanGeometry of current_file, read at its load
        current_stamp = None             # the same as the dict the indexer takes

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
                    logger.info("Unloading previous signal: %s", display_stem(current_file))
                    del current_signal
                    current_signal = None
                    gc.collect()

                # Load new file
                try:
                    current_signal = _load_signal(file_path)
                    current_file = file_path

                    # The grid this file was loaded with, read once, here: everything that follows (the
                    # averaging, the checkpoint's stamp, the indexer) works from THIS, never from a dataset
                    # the viewer may have open. A loader that disagrees with its own signal raises, and the
                    # file fails with that.
                    from backend.api.services import scan_geometry as scan_geometry_service
                    current_geometry = scan_geometry_service.geometry_of_signal(current_signal)
                    current_stamp = current_geometry.to_dict()

                    # Register calibration (preserves any PC set via copy-pc)
                    _register_signal_calibration(current_signal, file_path)
                    from backend.api.services.pc_record import pc_run_record
                    pc_records[file_path] = pc_run_record(dataset_name_for(file_path))

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
                            current_signal, preprocessing_config, current_geometry
                        )
                        if applied_preprocessing:
                            logger.info(
                                "Preprocessing applied to %s: %s",
                                display_stem(file_path), applied_preprocessing,
                            )
                    elif preprocessing_config and first_method == "spherical":
                        logger.info(
                            "Preprocessing skipped for %s: spherical reads patterns "
                            "directly from disk", display_stem(file_path),
                        )
                        applied_preprocessing = {"_skipped": "spherical-bypass"}

                    logger.info("Loaded: %s (grid: %s, method: %s)", display_stem(file_path), grid_shape, first_method)

                except Exception as e:
                    logger.exception("Failed to load %s", file_path)
                    # Mark all jobs for this file as failed
                    all_jobs = self.queue.get_jobs(batch_id)
                    for j in all_jobs:
                        if j["file_path"] == file_path and j["status"] == "pending":
                            # keep the code of a coded refusal (a hexagonal scan, an APEX file that
                            # needs a measurement): the interface translates on it
                            self.queue.update_job(j["id"], status="failed", error_msg=f"Load failed: {e}",
                                                  error_code=getattr(e, "code", None))
                    current_file = None
                    current_signal = None
                    current_geometry = current_stamp = None
                    continue

            # --- Checkpoint management ---
            if file_path not in checkpoints:
                # The checkpoint carries the grid of the file (writers and the resume check work from it). A
                # hexagonal geometry is the truth about the file's grid, and its step is the file's step: it is
                # taken first, so that axes without a scale cannot refuse a file whose geometry states its step.
                # A square one is stamped only when it states the step recorded here: when the axes carry no
                # scale (placeholder 1.0) and the step came from the file's header, the two differ, the
                # checkpoint keeps the step as it always has and gets no stamp (it reads as square).
                #
                # Nothing in this block may end the batch thread (no caller reads its outcome, the status would
                # stay "running" and the other files "pending"): a step that cannot be found, a checkpoint of
                # another pairing and a checkpoint that cannot be read all fail THIS file, with their text.
                stage = "step"
                try:
                    stamp_geometry = current_geometry
                    if current_geometry.is_hex:
                        step_size_um = current_geometry.step_x
                    else:
                        # Pull the real µm step from the signal's axes_manager so the exporter can stamp
                        # .ang/.ctf/.light with the right scale. Without this every downstream tool
                        # reconstructs an axes manager with scale=1.0 and µm distances are meaningless.
                        step_size_um = _extract_step_size_um(current_signal, file_path)
                        if step_size_um <= 0.0:
                            raise ValueError(
                                f"Could not determine step size for {Path(file_path).name}: "
                                "signal axes_manager has no scale and h5oina header "
                                "lacks 'X Step'. This file is not indexed, because its exports "
                                "would be unusable for any quantitative analysis; the batch goes on "
                                "with the other files."
                            )
                        if not math.isclose(step_size_um, current_geometry.step_x, rel_tol=1e-9):
                            stamp_geometry = None
                    stage = "checkpoint"
                    writer = CheckpointWriter(file_path)
                    writer.init_metadata(
                        grid_shape=grid_shape, batch_id=batch_id,
                        method=job["method"], step_size_um=step_size_um,
                        geometry=stamp_geometry, apex=scan_geometry_service.apex_of_signal(current_signal),
                    )
                except (ValueError, OSError, TypeError) as e:     # TypeError: a grid_shape that is no pair
                    # A changed pairing has its own code; a checkpoint that cannot be read (a damaged or missing
                    # stamp, no /metadata, not an HDF5 file) gets "checkpointUnreadable"; a missing step keeps
                    # its text and has no code.
                    code = getattr(e, "code", None)
                    if stage == "checkpoint":
                        code = code or "checkpointUnreadable"
                        message = str(e) if isinstance(e, CheckpointPairingChanged) else (
                            f"The checkpoint of {Path(file_path).name} cannot be read: {e}")
                    else:
                        message = str(e)
                    logger.exception("Checkpoint of %s refused: %s", display_stem(file_path), message)
                    for j in self.queue.get_jobs(batch_id):
                        if j["file_path"] == file_path and j["status"] == "pending":
                            self.queue.update_job(j["id"], status="failed", error_msg=message,
                                                  error_code=code)
                    current_file = None
                    current_signal = None
                    current_geometry = current_stamp = None
                    continue
                checkpoints[file_path] = writer
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
                logger.info("Skipping %s x %s (already in checkpoint)", display_stem(file_path), phase_name)
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
                progress_callback(job, f"Indexing {display_stem(file_path)} x {phase_name}")

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
                    scan_geometry=current_stamp,
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
                    display_stem(file_path), phase_name,
                    metadata["ci_mean"], metadata["duration_sec"],
                )

            except Exception as e:
                logger.exception("Failed: %s x %s", display_stem(file_path), phase_name)
                # a coded failure also carries the values its translated text needs
                # (a refused Hough library: the phases, their rows, the one-click count)
                _params = getattr(e, "params", None)
                self.queue.update_job(job["id"], status="failed", error_msg=str(e),
                                      error_code=getattr(e, "code", None),
                                      error_params=_params if isinstance(_params, dict) else None)

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
        # What went wrong between the runs and the files, by file: the export that follows is made from
        # the map as it stands, so the record of that file must say what was not applied to it.
        postprocessing_notes: Dict[str, List[str]] = {}
        for file_path, cw in checkpoints.items():
            jobs_for_file = [j for j in self.queue.get_jobs(batch_id) if j["file_path"] == file_path]
            if all(j["status"] in ("done", "skipped") for j in jobs_for_file):
                try:
                    cw.compute_auto_assignment()
                except Exception as e:
                    logger.warning("Auto-assignment failed for %s: %s", file_path, e)
                    postprocessing_notes.setdefault(file_path, []).append(Note(
                        f"The best phase of each point could not be assigned ({e}); the files of "
                        "this scan may not hold the assignment you expect.",
                        "autoAssignmentFailed", detail=str(e)))
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
                        logger.info("Cleanup applied to %s: %s", display_stem(file_path), stats)
                    except Exception as e:
                        logger.warning("Cleanup failed for %s: %s", file_path, e)
                        postprocessing_notes.setdefault(file_path, []).append(Note(
                            f"The cleanup of the phase map failed ({e}); the files hold the map "
                            "as it was before the cleanup.", "cleanupFailed", detail=str(e)))

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
                    pc_entry = calibration_store.get_entry(dataset_name_for(file_path))
                    # the PC the file was indexed at (None: the file carries none), see above
                    run_rec = pc_records.get(file_path)
                    pc_val = run_rec["pc"] if run_rec else None
                    s_tilt = run_rec["sample_tilt"] if run_rec else 70.0
                    d_shape = pc_entry.detector_shape if pc_entry else None
                    # what the phases recorded about their runs (read from the checkpoint,
                    # so phases resumed from an earlier run are included)
                    try:
                        record = cw.read_run_record()
                    except Exception as rec_err:    # the record is documentation: never lose the export over it
                        logger.warning("Could not read the run record of %s: %s", file_path, rec_err)
                        record = {"citation_steps": None, "hough_reflectors": None}

                    # the grid the checkpoint was written for decides which files this scan can have
                    file_formats, grid_kind, notes = _export_plan(export_formats, cw.checkpoint_path)
                    include_eds = bool(batch_config.get("include_eds", True))
                    notes = list(notes) + _eds_notes(file_path, file_formats, include_eds)
                    notes += postprocessing_notes.get(file_path, [])
                    results = export_all(
                        source_h5_path=file_path,
                        checkpoint_path=cw.checkpoint_path,
                        output_dir=out,
                        indexing_params=batch_config.get("preprocessing", {}),
                        pc=pc_val,
                        sample_tilt=s_tilt,
                        detector_shape=d_shape,
                        formats=file_formats,
                        # Default ON: full EDS in .light is required for the
                        # phase-vorauswahl workflow (FEAT-10) — the .light
                        # is useless for EDS-driven phase filtering without
                        # the original spectra alongside the indexing data.
                        include_eds=include_eds,
                        # two files with one stem must not overwrite each other's outputs
                        name_suffix=name_suffix_for(file_path),
                        citation_steps=record["citation_steps"],
                        hough_reflectors=record["hough_reflectors"],
                        caveats=record.get("caveats"),
                    )
                    errors = dict(results.get("_errors") or {})
                    errors.update(_unwritten(file_formats, results, errors))
                    self.exports[batch_id][file_path] = {
                        "dataset": dataset_name_for(file_path),
                        "name_suffix": name_suffix_for(file_path) or None,
                        "outputs": {k: v for k, v in results.items()
                                    if k in ("h5_rich", "h5_light", "ang", "ctf", "ang_square")},
                        "errors": {fmt: str(reason) for fmt, reason in errors.items()} or None,
                        # the same reasons with the code and numbers the page words them with
                        "error_items": _error_items(errors),
                        # which grid the files hold, the APEX measurement the checkpoint was written for
                        # (None: the file had none) and what the resampled files did to the scan
                        "grid_kind": grid_kind,
                        "pairing": read_checkpoint_pairing(cw.checkpoint_path),
                        "notes": [str(n) for n in notes],
                        "note_items": _items(notes),
                    }
                    logger.info("Exported %s: %s", display_stem(file_path), {k: v for k, v in results.items() if v})
                except Exception as e:
                    logger.exception("Export failed for %s", file_path)
                    # the file gets a record too: an export that never started is a missing file as well
                    self.exports[batch_id][file_path] = {
                        "dataset": dataset_name_for(file_path),
                        "name_suffix": name_suffix_for(file_path) or None,
                        "outputs": {}, "errors": {"export": str(e)},
                        "error_items": _error_items({"export": str(e)}),
                        "grid_kind": None, "pairing": None, "notes": [], "note_items": [],
                    }

        # --- Final status ---
        if not self._paused and not self._stopped:
            self.queue.set_batch_status(batch_id, "completed")
