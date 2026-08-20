"""Result Exporter — creates rich H5 result files + .ang/.ctf exports.

After batch indexing, this module:
1. Copies the original h5oina into a new file (preserving all original data)
2. Adds /Indexing/ group with per-phase results (Euler, CI, best match pattern)
3. Adds /Indexing/AutoAssignment/ with the winning phase per pixel
4. Adds /Documentation/ with human-readable descriptions
5. Exports .ang and .ctf for MTEX compatibility

File structure:
    result_<name>.h5    — Full results + original data
    result_<name>.ang   — MTEX-compatible (EDAX format)
    result_<name>.ctf   — MTEX-compatible (Oxford format)
"""
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Version of the result file format.
#   1.1 — added step_size_um to /Indexing and /Detector; readers use it to
#         decide whether to trust the file value over an active-signal axes
#         manager.
#   1.2 — added explicit per-pixel /Indexing/X and /Indexing/Y µm coordinate
#         datasets (column·step / row·step), and the interactive Light export
#         route now writes step_size_um too (previously only the batch helper
#         did). Makes the Light .h5 self-sufficient for MTEX import.
#   1.3 — euler_angles are written in the SOURCE VENDOR's stored frame (Aztec /
#         MTEX default import) instead of our native EMsoft/kikuchipy frame, so
#         a raw read matches the vendor solution. /Indexing.attrs carries
#         source_vendor + orientation_reference_frame; the reader inverts it on
#         re-import. (Fixes Irmi's 90deg-about-ND offset, 2026-06-05.)
FORMAT_VERSION = "1.3"


def place_rows_on_grid(rows, original_shape, selection_mask=None,
                       fill=0.0, dtype=None):
    """Reshape per-pixel result ROWS to the full ``(n_rows, n_cols[, k])`` grid.

    Full-coverage results are a plain reshape. ROI/masked results (fewer rows
    than grid pixels) are PLACED via the selection mask — the old blind
    ``.reshape(original_shape)`` raised ``cannot reshape array of size N``
    on every ROI export (user-hit 2026-07-15). Pixels outside the ROI get
    ``fill``.
    """
    import numpy as np
    a = np.asarray(rows)
    n_rows, n_cols = int(original_shape[0]), int(original_shape[1])
    n = n_rows * n_cols
    tail = a.shape[1:]
    if a.shape[0] == n:
        return a.reshape((n_rows, n_cols) + tail)
    if selection_mask is None:
        raise ValueError(
            f"result has {a.shape[0]} rows for a {n_rows}x{n_cols} grid and "
            "no selection mask — cannot place ROI rows")
    flat = np.flatnonzero(np.asarray(selection_mask, dtype=bool).ravel())
    if flat.size != a.shape[0]:
        raise ValueError(
            f"selection mask covers {flat.size} px but the result has "
            f"{a.shape[0]} rows")
    out = np.full((n,) + tail, fill, dtype=dtype if dtype is not None else a.dtype)
    out[flat] = a
    return out.reshape((n_rows, n_cols) + tail)


def xmap_phase_write_table(xmap):
    """(ordered_table, id_mapping) for the on-disk phase convention.

    On disk: ``phase_id`` 0 = unindexed, 1..N ↔ ``/Indexing/Phases/<n>`` in
    written order (the reader maps written id n → the n-th Phases entry).
    xmap ids are NOT reliably 0-based — Hough xmaps use orix-native 0..N-1,
    spherical PhaseLists carry explicit 1-based ids. The old blind
    ``raw_pid + 1`` therefore shifted every spherical export onto the WRONG
    phase name on re-import. Returns ``[(written_id, phase_obj), ...]`` and
    ``{actual_xmap_id: written_id}``.
    """
    table = []
    mapping = {}
    counter = 0
    try:
        entries = list(xmap.phases) if xmap is not None else []
    except Exception:
        entries = []
    for entry in entries:
        if isinstance(entry, tuple) and len(entry) == 2:
            pid, phase_obj = entry
        else:
            phase_obj = entry
            pid = getattr(entry, "id", None)
        try:
            if pid is not None and int(pid) < 0:
                continue
        except Exception:
            pass
        counter += 1
        table.append((counter, phase_obj))
        if pid is not None:
            try:
                mapping[int(pid)] = counter
            except Exception:
                pass
    return table, mapping


def map_phase_ids_for_export(raw_rows, xmap):
    """Per-row xmap phase ids → on-disk 1..N convention (0 = unindexed)."""
    import numpy as np
    raw = np.asarray(raw_rows).reshape(-1).astype(int)
    _table, mapping = xmap_phase_write_table(xmap)
    out = np.zeros(raw.shape, dtype=np.uint8)
    for actual, written in mapping.items():
        out[raw == actual] = written
    return out


def confidence_rows_for_export(confidence_scores, original_shape):
    """Flatten a result's confidence scores to one value per RESULT ROW.

    Handles: 1D per-row scores, a full (rows, cols) grid, and (n, ranks)
    top-k stacks (top-1 taken, matching extract_score_map)."""
    import numpy as np
    cs = np.asarray(confidence_scores)
    if cs.ndim == 2 and cs.shape == tuple(original_shape):
        return cs.reshape(-1)
    if cs.ndim == 2:
        return cs[:, 0].reshape(-1)
    return cs.reshape(-1)


def _read_step_size_from_checkpoint(checkpoint_path: str) -> float:
    """Read step_size_um from /metadata in a multiphase checkpoint.

    Raises ValueError if the attr is missing or non-positive — a missing
    step is a contract violation: every export downstream of this point
    bakes coordinates in µm, so a 0 or 1.0 default would silently
    corrupt every .ang/.ctf/.light produced by the run.
    """
    with h5py.File(checkpoint_path, "r") as cp:
        meta = cp.get("metadata")
        if meta is None:
            raise ValueError(
                f"Checkpoint {Path(checkpoint_path).name} has no /metadata group"
            )
        step = meta.attrs.get("step_size_um", 0.0)
        try:
            step_f = float(step)
        except (TypeError, ValueError):
            step_f = 0.0
        if step_f <= 0.0:
            raise ValueError(
                f"Checkpoint {Path(checkpoint_path).name} has step_size_um="
                f"{step!r}; refusing to export because every consumer of "
                "the .ang/.ctf/.light coordinates would silently use a "
                "1.0 µm default. Re-run the batch with a properly "
                "calibrated EBSD signal."
            )
        return step_f

# Quality / per-pixel measurement fields copied from the source h5oina into
# /Indexing/Assignment/ of the light h5. Mapping: h5oina path → (key in
# light file, dtype). Loader pulls these by key on read.
#
# MAD intentionally absent: we re-index from scratch instead of trusting
# the source's MAD, so carrying it forward only invites confusion (which
# MAD is shown — Aztec's or ours? we have no "ours"). BC is the band
# contrast acquisition signal and survives across re-indexing — keep it.
H5OINA_QUALITY_FIELDS: List[Tuple[str, str, type]] = [
    ("/1/EBSD/Data/Band Contrast",          "band_contrast", np.uint8),
    ("/1/EBSD/Data/Pattern Center X",       "pc_x",          np.float32),
    ("/1/EBSD/Data/Pattern Center Y",       "pc_y",          np.float32),
    ("/1/EBSD/Data/Detector Distance",      "dd",            np.float32),
    ("/1/EBSD/Data/Bands",                  "bands",         np.uint8),
]


def _copy_quality_fields(
    source_h5_path: str, dst_asg_grp: "h5py.Group", grid_shape: Tuple[int, int]
) -> List[str]:
    """Copy per-pixel quality fields (BC, MAD, PC, Bands) from h5oina.

    Each field is reshaped to grid_shape and stored under
    /Indexing/Assignment/<key>. Returns list of keys that were copied so
    the caller can attach a manifest attr.
    """
    copied: List[str] = []
    n_pixels = int(np.prod(grid_shape))
    try:
        with h5py.File(source_h5_path, "r") as src:
            for h5_path, key, dtype in H5OINA_QUALITY_FIELDS:
                if key in dst_asg_grp:
                    continue  # idempotent — don't overwrite if already present
                ds = src.get(h5_path)
                if ds is None:
                    continue
                arr = np.asarray(ds)
                if arr.size != n_pixels:
                    logger.warning(
                        "Quality field %s size %d != grid %s — skipping",
                        key, arr.size, grid_shape,
                    )
                    continue
                dst_asg_grp.create_dataset(
                    key,
                    data=arr.reshape(grid_shape),
                    dtype=dtype,
                    compression="gzip",
                )
                copied.append(key)
    except Exception as e:
        logger.warning("Quality-field copy failed: %s", e)
    return copied


README_TEXT = """EBSD Indexing Results — Orienta
=====================================

This file contains EBSD indexing results combined with original experimental data.

Structure:
  /1/EBSD/              Original experimental data (copied from h5oina source)
  /1/EDS/               Original EDS data (if available in source)
  /Indexing/            Indexing results
  /Indexing/PerPhase/   Results for EACH phase that was tested
  /Indexing/Assignment/ Best phase per pixel (auto-assigned by highest CI)
  /Indexing/Parameters/ Indexing parameters for reproducibility
  /Detector/            Detector geometry and pattern center
  /Documentation/       This README and format description

Grid / coordinates (format_version >= 1.2):
  - /Indexing.attrs[step_size_um]   scan step in micrometres
  - /Indexing.attrs[grid_shape]     [rows, cols]
  - /Indexing/X, /Indexing/Y        per-pixel sample coordinates in µm,
                                    shape (rows, cols). X = column·step,
                                    Y = row·step (same convention as .ang).
                                    Use these to place orientations on a
                                    grid in MTEX without the source h5oina.

Conventions:
  - Euler angles: Bunge convention (ZXZ), unit = radians
  - Phase IDs: 0 = not indexed, 1..N = phases (see /Indexing/Phases/)
  - CI (Confidence Index): [0, 1], higher = better match
  - Patterns: uint8 [0, 255]
  - All datasets are gzip-compressed where applicable

Per-Phase Results (/Indexing/PerPhase/<phase_name>/):
  Each phase that was indexed has its own group with:
  - euler_angles:  orientation at each pixel IF this phase were correct
  - confidence_index: how well this phase matched at each pixel
  - best_match_pattern: the simulated pattern of the best match (if available)
  This allows post-processing to re-assign phases (e.g., switch a pixel
  from Al to Fe4Al13 if EDS reveals iron content).

Auto-Assignment (/Indexing/Assignment/):
  The automatically determined best phase per pixel, based on highest CI.
  - phase_id: winning phase at each pixel
  - euler_angles: orientation of the winning phase
  - confidence_index: CI of the winning phase
  - uncertainty: CI_best - CI_second_best (low = ambiguous)

Software: Orienta — https://github.com/your-repo
Libraries: kikuchipy, orix, diffsims, EMSphInx
"""


def _write_scan_provenance(group, scan_provenance: Optional[Dict]) -> None:
    """Stamp where the indexed dataset sat in its original scan.

    ``scan_provenance`` is the dict ``indexing._scan_provenance_fields``
    returns: ``scan_row_offset``, ``scan_col_offset``, ``scan_shape``.
    ``None`` — the default at every call site — writes nothing, so a file
    exported without it is identical to one exported before this existed.
    ``None`` values inside the dict are skipped too: h5py has no way to store
    one, and "no scan shape" is exactly the absent attribute.
    ``indexing._read_scan_provenance`` reads these back on re-import.
    """
    for key, value in (scan_provenance or {}).items():
        if value is not None:
            group.attrs[key] = value


def export_result_h5(
    source_h5_path: str,
    checkpoint_path: str,
    output_dir: str,
    indexing_params: Optional[Dict] = None,
    pc: Optional[List[float]] = None,
    sample_tilt: float = 70.0,
    detector_shape: Optional[Tuple[int, int]] = None,
    step_size: Optional[float] = None,
    scan_provenance: Optional[Dict] = None,
) -> str:
    """Create a rich H5 result file: original data + indexing results.

    Parameters
    ----------
    source_h5_path : str
        Path to the original h5oina file.
    checkpoint_path : str
        Path to the _multiphase.h5 checkpoint file (from batch indexing).
    output_dir : str
        Directory to write the result file.
    indexing_params : dict, optional
        Method parameters for reproducibility.
    pc : list of 3 floats, optional
        Pattern center [PCx, PCy, PCz].
    sample_tilt : float
    detector_shape : tuple (height, width)
    scan_provenance : dict, optional
        ``{scan_row_offset, scan_col_offset, scan_shape}`` — where the indexed
        dataset sat in the original scan. Written as attributes on /Indexing
        and /Documentation. ``None`` writes nothing.

        ONLY pass this when the checkpoint being exported is the ACTIVE
        dataset's, because the only source of the value is the active crop
        window. A caller that exports files from a queue (the batch manager)
        must leave it ``None``: its files are not the active dataset, so
        stamping the live window on them would claim provenance they do not
        have. That is why the parameter currently has no caller — it is
        waiting for an export path that runs on the active dataset, not an
        oversight to be "finished".

    Returns
    -------
    str
        Path to the created result file.
    """
    source_path = Path(source_h5_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if step_size is None:
        step_size = _read_step_size_from_checkpoint(checkpoint_path)

    result_name = f"result_{source_path.stem}.h5"
    result_path = out_dir / result_name

    logger.info("Creating result file: %s", result_path)

    # Step 1: Copy original h5oina as the base
    logger.info("Copying original data from: %s", source_path)
    shutil.copy2(str(source_path), str(result_path))

    # Step 2: Open checkpoint and read indexing results
    with h5py.File(checkpoint_path, "r") as cp:
        phases_grp = cp.get("phases")
        if phases_grp is None:
            logger.warning("No phases in checkpoint, skipping indexing results")
            return str(result_path)

        # Collect phase data
        phase_names = []
        phase_data = {}
        for name in phases_grp:
            if phases_grp[name].attrs.get("status") != "done":
                continue
            phase_names.append(name)
            phase_data[name] = {
                "ci": np.array(phases_grp[name]["ci"]),
                "orientation": np.array(phases_grp[name]["orientation"]),
                "ci_mean": float(phases_grp[name].attrs.get("ci_mean", 0.0)),
                "ci_median": float(phases_grp[name].attrs.get("ci_median", 0.0)),
                "phase_file": phases_grp[name].attrs.get("phase_file", ""),
                "duration_sec": float(phases_grp[name].attrs.get("duration_sec", 0.0)),
            }

        # Read auto-assignment if available. When /apply-cleanup has run,
        # a ``cleaned_phase_id`` dataset is present — prefer that so exports
        # propagate the filtered assignment rather than the raw noise.
        #
        # Convention in our result files (both rich and light):
        #   phase_id 0   = not indexed (cleanup result)
        #   phase_id 1+  = 1-based phase index (PerPhase/Phases groups)
        # This matches the .ang/.ctf / MTEX / Oxford convention and lets the
        # unindexed state survive the uint8 storage.
        auto_assign = {}
        if "auto_assignment" in cp:
            aa = cp["auto_assignment"]
            for key in aa:
                auto_assign[key] = np.array(aa[key])
            # Always 1-based on disk (0 = not indexed) so the reader never
            # has to guess which convention a given file uses. Prior code
            # kept raw 0-based when no cleanup was applied, which collapsed
            # Al pixels to phase_id=0 and made the reader treat them as
            # unindexed — empty IPF map on reload.
            if "cleaned_phase_id" in aa:
                cleaned = auto_assign["cleaned_phase_id"]
                auto_assign["best_phase_id"] = np.where(
                    cleaned < 0, 0, cleaned + 1
                ).astype(np.uint8)
                auto_assign["_cleaned"] = True
            elif "best_phase_id" in auto_assign:
                auto_assign["best_phase_id"] = (
                    auto_assign["best_phase_id"].astype(np.uint8) + 1
                )
            auto_assign["_one_based"] = True

        grid_shape = tuple(cp["metadata"].attrs.get("grid_shape", [0, 0]))
        method = cp["metadata"].attrs.get("method", "unknown")

    # Step 3: Write indexing results into the result file
    with h5py.File(str(result_path), "a") as f:
        now = datetime.now(timezone.utc).isoformat()

        # Remove existing /Indexing if re-exporting
        if "Indexing" in f:
            del f["Indexing"]

        idx = f.create_group("Indexing")
        idx.attrs["method"] = method
        idx.attrs["software"] = "Orienta"
        idx.attrs["created"] = now
        idx.attrs["source_file"] = str(source_path.name)
        idx.attrs["n_phases"] = len(phase_names)
        idx.attrs["grid_shape"] = list(grid_shape)
        idx.attrs["step_size_um"] = float(step_size)
        idx.attrs["format_version"] = FORMAT_VERSION
        _write_scan_provenance(idx, scan_provenance)

        # --- Per-Phase Results ---
        per_phase = idx.create_group("PerPhase")
        per_phase.attrs["description"] = (
            "Results for each phase that was tested. "
            "Each pixel has orientation + CI for every phase, "
            "allowing post-processing phase reassignment."
        )

        for i, name in enumerate(phase_names):
            pd = phase_data[name]
            pg = per_phase.create_group(name)
            pg.attrs["phase_id"] = i + 1
            pg.attrs["phase_file"] = pd["phase_file"]
            pg.attrs["ci_mean"] = pd["ci_mean"]
            pg.attrs["ci_median"] = pd["ci_median"]
            pg.attrs["duration_sec"] = pd["duration_sec"]

            ds_euler = pg.create_dataset(
                "euler_angles", data=pd["orientation"],
                dtype=np.float32, compression="gzip",
            )
            ds_euler.attrs["unit"] = "radians"
            ds_euler.attrs["convention"] = "Bunge (ZXZ)"
            ds_euler.attrs["description"] = "Euler angles if this phase is the correct assignment"

            ds_ci = pg.create_dataset(
                "confidence_index", data=pd["ci"],
                dtype=np.float32, compression="gzip",
            )
            ds_ci.attrs["description"] = "Confidence Index [0,1] for this phase at each pixel"

        # --- Phase Table ---
        phases_grp = idx.create_group("Phases")
        phases_grp.attrs["description"] = "Phase definitions"
        for i, name in enumerate(phase_names):
            pg = phases_grp.create_group(str(i + 1))
            pg.attrs["name"] = name
            pg.attrs["phase_file"] = phase_data[name]["phase_file"]
            pg.attrs["ci_mean"] = phase_data[name]["ci_mean"]

        # --- Auto-Assignment ---
        if auto_assign:
            aa_grp = idx.create_group("Assignment")
            aa_grp.attrs["description"] = (
                "Automatically determined best phase per pixel (highest CI wins). "
                "Use /Indexing/PerPhase/ to override with a different phase."
            )

            if "best_phase_id" in auto_assign:
                ds = aa_grp.create_dataset(
                    "phase_id", data=auto_assign["best_phase_id"],
                    dtype=np.uint8, compression="gzip",
                )
                ds.attrs["description"] = "Best phase ID per pixel (1-based, 0=not indexed)"
                ds.attrs["phase_names"] = phase_names

            if "best_ci" in auto_assign:
                ds = aa_grp.create_dataset(
                    "confidence_index", data=auto_assign["best_ci"],
                    dtype=np.float32, compression="gzip",
                )
                ds.attrs["description"] = "CI of the winning phase at each pixel"

            if "uncertainty" in auto_assign:
                ds = aa_grp.create_dataset(
                    "uncertainty", data=auto_assign["uncertainty"],
                    dtype=np.float32, compression="gzip",
                )
                ds.attrs["description"] = "CI_best - CI_second_best (low = ambiguous assignment)"

            if "confident_mask" in auto_assign:
                ds = aa_grp.create_dataset(
                    "confident_mask", data=auto_assign["confident_mask"],
                    compression="gzip",
                )
                ds.attrs["description"] = "True where assignment is confident (uncertainty > threshold)"

            # Build the winning euler_angles from per-phase data.
            # best_ids is stored 1-based when cleanup ran (see _one_based
            # flag above) and 0-based otherwise. If we don't honour that,
            # the enumerate(i) counter mismatches and each phase's
            # orientation lands on the next phase's pixels — exactly the
            # "batch IPF is noise" bug.
            if "best_phase_id" in auto_assign and phase_names:
                best_ids = auto_assign["best_phase_id"]
                one_based = bool(auto_assign.get("_one_based", False))
                euler_best = np.zeros((*grid_shape, 3), dtype=np.float32)
                for i, name in enumerate(phase_names):
                    target = (i + 1) if one_based else i
                    mask = best_ids == target
                    euler_best[mask] = phase_data[name]["orientation"][mask]
                ds = aa_grp.create_dataset(
                    "euler_angles", data=euler_best,
                    dtype=np.float32, compression="gzip",
                )
                ds.attrs["unit"] = "radians"
                ds.attrs["convention"] = "Bunge (ZXZ)"
                ds.attrs["description"] = "Euler angles of the winning phase at each pixel"

        # --- Parameters (reproducibility) ---
        params = idx.create_group("Parameters")
        params.attrs["description"] = "Indexing parameters for reproducibility"
        params.attrs["method"] = method
        if indexing_params:
            for k, v in indexing_params.items():
                try:
                    params.attrs[k] = v
                except TypeError:
                    params.attrs[k] = str(v)

        if pc is not None:
            ds = params.create_dataset("pc", data=np.array(pc, dtype=np.float32))
            ds.attrs["description"] = "Pattern Center [PCx, PCy, PCz]"
            ds.attrs["convention"] = "Bruker"

        params.attrs["sample_tilt_deg"] = sample_tilt
        params.attrs["step_size_um"] = float(step_size)
        if detector_shape:
            params.attrs["detector_shape"] = list(detector_shape)

        # --- Detector ---
        if "Detector" in f:
            del f["Detector"]
        det = f.create_group("Detector")
        det.attrs["description"] = "Detector geometry used for indexing"
        if pc is not None:
            det.create_dataset("pc", data=np.array(pc, dtype=np.float32))
        det.attrs["sample_tilt_deg"] = sample_tilt
        det.attrs["step_size_um"] = float(step_size)
        if detector_shape:
            det.attrs["shape"] = list(detector_shape)

        # --- Documentation ---
        if "Documentation" in f:
            del f["Documentation"]
        doc = f.create_group("Documentation")
        doc.attrs["format_version"] = FORMAT_VERSION
        _write_scan_provenance(doc, scan_provenance)
        doc.attrs["description"] = "EBSD indexing results + original experimental data"
        doc.create_dataset("README", data=README_TEXT)

        # Phase table as readable string
        table_lines = ["ID  Name             CI_mean   Phase File"]
        table_lines.append("-" * 70)
        for i, name in enumerate(phase_names):
            pd = phase_data[name]
            table_lines.append(
                f"{i+1:2d}  {name:16s} {pd['ci_mean']:.3f}     {Path(pd['phase_file']).name}"
            )
        doc.create_dataset("phase_table", data="\n".join(table_lines))

    file_size_mb = result_path.stat().st_size / (1024 * 1024)
    logger.info("Result file created: %s (%.1f MB)", result_path, file_size_mb)
    return str(result_path)


def _read_h5oina_acquisition(source_h5_path: Optional[str]) -> Dict[str, float]:
    """Pull real KV / TiltAngle / Mag / WD from h5oina /1/EBSD/Header.

    Returns whatever it could read; missing keys are absent (caller falls
    back to None and writes an empty CTF field rather than a fake value).
    Default values used to be hardcoded (KV=20, TiltAngle=70, Mag=300) and
    silently lied about acquisition geometry whenever the user's setup
    differed.
    """
    out: Dict[str, float] = {}
    if not source_h5_path:
        return out
    try:
        with h5py.File(source_h5_path, "r") as f:
            for entry in f.keys():
                hdr = f.get(f"{entry}/EBSD/Header")
                if hdr is None:
                    continue
                # H5OINA writes scalars as 1-element datasets; coerce to float.
                def _get(*keys):
                    for k in keys:
                        if k in hdr:
                            try: return float(np.asarray(hdr[k]).item())
                            except Exception:
                                try: return float(np.asarray(hdr[k]).flatten()[0])
                                except Exception: return None
                    return None
                kv  = _get("Beam Voltage", "BeamVoltage", "Accelerating Voltage")
                tlt = _get("Tilt Angle", "TiltAngle", "Sample Tilt")
                mag = _get("Magnification")
                wd  = _get("Working Distance", "WorkingDistance", "WD")
                if kv  is not None: out["kv"]  = kv
                if tlt is not None: out["tilt"] = tlt
                if mag is not None: out["mag"] = mag
                if wd  is not None: out["wd"]  = wd
                break
    except Exception as e:
        logger.warning("Could not read h5oina acquisition header: %s", e)
    return out


def _read_h5oina_quality(
    source_h5_path: Optional[str], n_pixels: int
) -> Dict[str, np.ndarray]:
    """Pull real BC and Bands per-pixel arrays from h5oina.

    Returns flattened arrays of size n_pixels. MAD is deliberately not
    pulled — we re-index from scratch so the source's MAD doesn't
    apply to our results, and writing a "MAD" column with somebody
    else's values is misleading.
    """
    out: Dict[str, np.ndarray] = {}
    if not source_h5_path:
        return out
    try:
        with h5py.File(source_h5_path, "r") as f:
            for h5_key, out_key in (
                ("/1/EBSD/Data/Band Contrast", "bc"),
                ("/1/EBSD/Data/Bands", "bands"),
            ):
                ds = f.get(h5_key)
                if ds is None:
                    continue
                arr = np.asarray(ds).reshape(-1)
                if arr.size != n_pixels:
                    logger.warning(
                        "h5oina %s size %d != n_pixels %d — skipping",
                        h5_key, arr.size, n_pixels,
                    )
                    continue
                out[out_key] = arr
    except Exception as e:
        logger.warning("Could not read h5oina quality fields: %s", e)
    return out


def _write_ctf(
    ctf_path: str, xmap, phase_names: List[str], step_size: float,
    source_h5_path: Optional[str] = None,
    sample_tilt: Optional[float] = None,
) -> None:
    """Write an MTEX-compatible Channel Text File (.ctf) from a CrystalMap.

    orix doesn't write .ctf directly (``orix.io.save`` only knows .ang, .h5).
    This is a minimal but complete CTF writer: HKL-style header + phase table
    + per-pixel rows with (phase_id, X, Y, bands, error, euler°, MAD, BC, BS).
    Phase IDs are 1-based; 0 is reserved for "not indexed".

    Quality columns:
    - ``BC``: real per-pixel Band Contrast from the source h5oina if
      ``source_h5_path`` is provided; otherwise an honest all-zero column
      (never a CI surrogate) plus a provenance note on the ``Prj`` header.
    - ``Bands``: real per-pixel band count from the source if available,
      else 8 (a neutral placeholder).
    - ``MAD``: written as 0. We re-index from scratch and our pipeline
      doesn't compute MAD; writing the source's MAD would mix Aztec's
      diagnostic with our orientation result, which is misleading.

    The acquisition header (TiltAngle) prefers the caller-supplied
    ``sample_tilt`` (typically the calibrated value from
    CalibrationStore) over the h5oina header's acquisition value.
    KV/Mag/WD continue to come from the h5oina header — there is no
    "calibrated" version of those.
    """
    import math
    # Access xmap data
    n = xmap.size
    # Coordinates — xmap.x/y are in microns already
    xs = np.asarray(xmap.x).reshape(-1)
    ys = np.asarray(xmap.y).reshape(-1)
    # Phase ids in xmap are 0-based with -1 = unindexed. CTF on disk
    # wants 1-based with 0 = unindexed, so translate just here. Doing
    # the shift earlier (in xmap construction) used to misalign
    # phase_id with PhaseList ids and lose every non-first phase's
    # symmetry — see the comment block in export_ang_ctf for context.
    raw_pid = np.asarray(xmap.phase_id).reshape(-1).astype(int)
    phase_id = np.where(raw_pid < 0, 0, raw_pid + 1)
    # Euler angles — Bunge convention, CTF wants degrees
    eulers = xmap.rotations.to_euler(degrees=True)
    eulers = np.asarray(eulers).reshape(-1, 3)

    # Real BC/Bands from source h5oina if available; when the source has
    # no Band Contrast we write an honest all-zero BC column (never a
    # confidence surrogate). MAD is always 0 — we re-index from scratch so
    # we have no MAD of our own and the source's MAD doesn't apply to our
    # orientations.
    quality = _read_h5oina_quality(source_h5_path, n)
    has_real_bc    = "bc" in quality
    has_real_bands = "bands" in quality
    bands = quality.get("bands", np.full(n, 8, dtype=int)).astype(int)
    error = np.zeros(n, dtype=int)
    mad   = np.zeros(n, dtype=np.float32)  # column kept for CTF spec compliance
    if has_real_bc:
        bc = quality["bc"].astype(int)
    else:
        bc = np.zeros(n, dtype=int)  # no native Band Contrast — honest empty column, never a CI surrogate
    bs = np.full(n, 255, dtype=int)

    # Grid dimensions (infer from coordinate range)
    if n == 0:
        raise ValueError("Empty CrystalMap — nothing to write")
    x_unique = np.unique(xs)
    y_unique = np.unique(ys)
    x_cells = len(x_unique)
    y_cells = len(y_unique)
    if x_cells * y_cells != n:
        # Used to silently fall back to a flat N×1 file. That made every
        # downstream MTEX import look like a 1-pixel-wide line scan with
        # no warning. Refuse instead — the CrystalMap was constructed by
        # us in this same function, so a non-rectangular xs/ys means a
        # real upstream bug we want to see in the logs, not a CTF that
        # silently misrepresents the data.
        raise ValueError(
            f"Cannot infer rectangular CTF grid from xmap: "
            f"{x_cells}×{y_cells} unique coords ≠ {n} pixels. "
            "Coordinates are not on a regular grid; this should never "
            "happen for our exporter and indicates a CrystalMap "
            "construction bug."
        )

    # Map orix point_group.name → Oxford CTF "Laue class" code.
    # Oxford uses a compressed 1..11 code covering Laue classes; we pick the
    # best match per phase so the CTF reader reduces orientations to the
    # right fundamental zone. Previously every phase was hard-coded Laue 11
    # (cubic m-3m), which happened to be right for Al but wrong for any
    # non-cubic phase (e.g. Al7FeCu2 / oC16 / mmm) — IPF then picked
    # different colours than the .ang for those phases.
    _LAUE_FROM_PG = {
        "-1":   1,  "1":    1,
        "2/m":  2,  "2":    2,   "m":   2,
        "mmm":  3,  "222":  3,   "mm2": 3,
        "4/m":  4,  "4":    4,   "-4":  4,
        "4/mmm":5,  "422":  5,   "4mm": 5,  "-42m": 5,
        "-3":   6,  "3":    6,
        "-3m":  7,  "32":   7,   "3m":  7,
        "6/m":  8,  "6":    8,   "-6":  8,
        "6/mmm":9,  "622":  9,   "6mm": 9,  "-6m2": 9,
        "m-3":  10, "23":   10,
        "m-3m": 11, "432":  11,  "-43m":11,
    }

    def _phase_metadata(name):
        """Return (laue_int, a, b, c, alpha, beta, gamma, sg_int) from the
        xmap's Phase object. Raises ValueError when the phase is missing
        a usable point group — silent cubic-default Laue=11 used to slip
        past, making MTEX colour every non-cubic phase as though it were
        cubic. The caller (export_ang_ctf) is responsible for ensuring
        each Phase carries point_group / space_group; if it doesn't,
        the upstream checkpoint is incomplete and re-running the batch
        is the right answer, not papering over with fake symmetry."""
        for pid in xmap.phases.ids:
            if pid < 0:
                continue
            p = xmap.phases[pid]
            if p.name != name:
                continue
            pg_name = None
            try: pg_name = p.point_group.name
            except Exception: pass
            laue = _LAUE_FROM_PG.get(pg_name)
            if laue is None:
                raise ValueError(
                    f"Phase {name!r} has no usable point_group "
                    f"(orix returned {pg_name!r}); refusing to write a "
                    "CTF with a fake cubic Laue class. Re-run the batch "
                    "indexer so the checkpoint captures point_group, or "
                    "edit the phase definition to set its symmetry."
                )
            a = b = c = 1.0
            alpha = beta = gamma = 90.0
            struct = getattr(p, "structure", None)
            lat = getattr(struct, "lattice", None) if struct is not None else None
            if lat is not None:
                try: a, b, c = float(lat.a), float(lat.b), float(lat.c)
                except Exception: pass
                try:
                    alpha, beta, gamma = float(lat.alpha), float(lat.beta), float(lat.gamma)
                except Exception: pass
            sg = 0
            try:
                sg = int(p.space_group.number) if p.space_group is not None else 0
            except Exception: pass
            return laue, a, b, c, alpha, beta, gamma, sg
        raise ValueError(f"Phase {name!r} not found in xmap.phases")

    # Acquisition header: KV/Mag come from the source h5oina (no
    # calibration stage modifies them). TiltAngle prefers the caller's
    # sample_tilt (calibrated value from CalibrationStore) and only
    # falls back to the h5oina header's acquisition tilt when no
    # calibrated value is available.
    acq = _read_h5oina_acquisition(source_h5_path)
    kv_str  = f"{acq['kv']:.1f}"  if "kv"  in acq else "0"
    mag_str = f"{acq['mag']:.0f}" if "mag" in acq else "0"
    if sample_tilt is not None:
        tilt_str = f"{float(sample_tilt):.1f}"
    elif "tilt" in acq:
        tilt_str = f"{acq['tilt']:.1f}"
    else:
        tilt_str = "0"

    with open(ctf_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("Channel Text File\n")
        # Prj is free text; MTEX/HKL parsers ignore its content. Append a
        # provenance note ONLY when there is no native Band Contrast, so the
        # BC column of 0 isn't mistaken for a real (or CI-surrogate) signal.
        # Byte-identical to the original line when native BC IS present.
        prj = "Orienta multi-phase batch"
        if not has_real_bc:
            prj += " — BC column = 0 (no native Band Contrast in source)"
        f.write(f"Prj\t{prj}\n")
        f.write(f"Author\tOrienta\n")
        f.write(f"JobMode\tGrid\n")
        f.write(f"XCells\t{x_cells}\n")
        f.write(f"YCells\t{y_cells}\n")
        f.write(f"XStep\t{step_size}\n")
        f.write(f"YStep\t{step_size}\n")
        f.write("AcqE1\t0\nAcqE2\t0\nAcqE3\t0\n")
        f.write("Euler angles refer to Sample Coordinate system (CS0)!\t"
                f"Mag\t{mag_str}\tCoverage\t100\tDevice\t0\tKV\t{kv_str}"
                f"\tTiltAngle\t{tilt_str}\tTiltAxis\t0\n")
        # Whether BC is real or surrogate is logged at WARNING level
        # rather than written into the file — Channel CTF parsers are
        # picky about non-spec lines and adding "# ..." can cause MTEX
        # import to fail in some versions.
        if not has_real_bc:
            logger.warning(
                "CTF %s: no native Band Contrast available — writing BC "
                "column = 0 (honest empty column, not a CI surrogate). "
                "Pass source_h5_path to get the real Band Contrast values "
                "from the h5oina.", ctf_path,
            )
        f.write(f"Phases\t{len(phase_names)}\n")
        for name in phase_names:
            laue, a, b, c, alpha, beta, gamma, sg = _phase_metadata(name)
            # Column order: <a;b;c>\t<alpha;beta;gamma>\t<name>\t<Laue>\t<SpaceGroup>
            f.write(
                f"{a:.3f};{b:.3f};{c:.3f}\t"
                f"{alpha:.1f};{beta:.1f};{gamma:.1f}\t"
                f"{name}\t{laue}\t{sg}\n"
            )
        # Column header
        f.write("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n")
        # Rows
        for i in range(n):
            f.write(
                f"{phase_id[i]}\t{xs[i]:.4f}\t{ys[i]:.4f}\t"
                f"{bands[i]}\t{error[i]}\t"
                f"{eulers[i,0]:.4f}\t{eulers[i,1]:.4f}\t{eulers[i,2]:.4f}\t"
                f"{mad[i]:.4f}\t{bc[i]}\t{bs[i]}\n"
            )


def _inject_ang_acquisition(
    ang_path: str,
    *,
    sample_tilt: Optional[float] = None,
    pc: Optional[List[float]] = None,
    note: Optional[str] = None,
) -> None:
    """Add/replace TILT and PC lines in an ANG file's header.

    orix.io.save writes the orientation and quality columns but doesn't
    accept detector geometry, so a freshly saved ANG carries default
    (zero) TILT and PC. This function reads the file, splices the
    real values into the header section (any line beginning with ``#``
    before the data rows), and rewrites the file.

    Lines added/replaced (Bruker/EDAX ANG convention):
        # TILT          <deg>
        # x-star        <pcx>
        # y-star        <pcy>
        # z-star        <pcz>
    Existing matching lines are replaced; missing lines are inserted at
    the end of the header.

    ``note`` (optional) is a free-text ``#``-comment line (e.g. a
    provenance note that the IQ column is empty). It is added/replaced in
    the header just like the geometry lines. If nothing is supplied the
    file is left untouched.
    """
    if sample_tilt is None and pc is None and note is None:
        return
    try:
        with open(ang_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as e:
        logger.warning("ANG header inject: cannot read %s (%s)", ang_path, e)
        return

    # Build replacement payload. Order chosen to match how OIM/Bruker
    # readers expect it — TILT after acquisition geometry, x/y/z-star
    # together. PC convention: orix gives Bruker (PCx, PCy, PCz);
    # EDAX ANG x-star/y-star/z-star is Bruker-equivalent for our usage.
    replacements: Dict[str, str] = {}
    if sample_tilt is not None:
        replacements["# TILT"] = f"# TILT          {float(sample_tilt):.6f}\n"
    if pc is not None and len(pc) >= 3:
        replacements["# x-star"] = f"# x-star        {float(pc[0]):.6f}\n"
        replacements["# y-star"] = f"# y-star        {float(pc[1]):.6f}\n"
        replacements["# z-star"] = f"# z-star        {float(pc[2]):.6f}\n"
    if note is not None:
        # Full ``#``-comment line; replaced if already present, else appended.
        replacements["# NOTE:"] = note if note.endswith("\n") else note + "\n"

    out: List[str] = []
    seen = set()
    in_header = True
    last_header_idx = -1
    for line in lines:
        stripped = line.lstrip()
        if in_header and not stripped.startswith("#") and stripped.strip() != "":
            in_header = False
        if in_header:
            last_header_idx = len(out)
        replaced = False
        for key, repl in replacements.items():
            if stripped.startswith(key):
                out.append(repl)
                seen.add(key)
                replaced = True
                break
        if not replaced:
            out.append(line)

    # Insert any missing keys at the tail of the header section.
    missing = [v for k, v in replacements.items() if k not in seen]
    if missing:
        insert_at = last_header_idx + 1 if last_header_idx >= 0 else 0
        out[insert_at:insert_at] = missing

    try:
        with open(ang_path, "w", encoding="utf-8", newline="") as fh:
            fh.writelines(out)
    except OSError as e:
        logger.warning("ANG header inject: cannot write %s (%s)", ang_path, e)


def export_ang_ctf(
    checkpoint_path: str,
    output_dir: str,
    phase_names: Optional[List[str]] = None,
    step_size: Optional[float] = None,
    *, write_ang: bool = True, write_ctf: bool = True,
    source_h5_path: Optional[str] = None,
    sample_tilt: Optional[float] = None,
    pc: Optional[List[float]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Export indexing results as .ang and .ctf for MTEX compatibility.

    Uses the auto-assignment (best phase per pixel) to build a CrystalMap
    and exports via orix.

    Parameters
    ----------
    checkpoint_path : str
        Path to _multiphase.h5 checkpoint.
    output_dir : str
        Directory for output files.
    phase_names : list of str, optional
        Phase names in order. If None, read from checkpoint.
    step_size : float, optional
        Step size in microns for pixel coordinates. If None, read from
        the checkpoint's /metadata attrs. Raises ValueError if neither
        the caller nor the checkpoint provides a positive value.
    source_h5_path : str, optional
        Original h5oina file. When given, the .ctf gets real per-pixel
        Band Contrast / MAD / Bands and the header carries real KV /
        TiltAngle / Mag instead of the previous CI surrogates and
        hardcoded defaults.

    Returns
    -------
    (ang_path, ctf_path) — paths to created files, or None if failed.
    """
    if step_size is None:
        step_size = _read_step_size_from_checkpoint(checkpoint_path)
    elif step_size <= 0.0:
        raise ValueError(
            f"export_ang_ctf called with step_size={step_size!r}; refusing "
            "to write coordinates that would be unusable downstream."
        )
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    cp_path = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = cp_path.stem.replace("_multiphase", "")

    try:
        with h5py.File(str(cp_path), "r") as cp:
            grid_shape = tuple(cp["metadata"].attrs.get("grid_shape", [0, 0]))
            n_rows, n_cols = grid_shape

            # Get phase names
            if phase_names is None:
                phase_names = list(cp["metadata"].attrs.get("phases_list", []))

            if not phase_names:
                logger.warning("No phases in checkpoint for ang/ctf export")
                return None, None

            # Read auto-assignment or build from per-phase CI.
            # Prefer cleaned_phase_id (written by /apply-cleanup or batch
            # postprocessing) over the raw best_phase_id, so the .ang/.ctf
            # exports reflect whatever noise removal the user requested.
            # Keep -1 ("unindexed") as-is here so the downstream CTF/orix
            # mapping can emit "phase 0 = not indexed" rather than silently
            # re-assigning cleaned pixels to the first real phase.
            # Keep best_phase_id as int16 so -1 for unindexed is preserved.
            # Previously we squashed -1 → 0 before the per-phase loop, which
            # made the Al (phase_id 0) branch *also* match every unindexed
            # pixel and stamp Al's orientation onto them. When IPF rendered,
            # those noisy Al orientations leaked through and produced the
            # "batch looks like random speckle" bug.
            if "auto_assignment" in cp and "cleaned_phase_id" in cp["auto_assignment"]:
                best_phase_id = np.array(cp["auto_assignment"]["cleaned_phase_id"]).astype(np.int16)
            elif "auto_assignment" in cp and "best_phase_id" in cp["auto_assignment"]:
                best_phase_id = np.array(cp["auto_assignment"]["best_phase_id"]).astype(np.int16)
            else:
                # Compute from phase CIs — argmax is 0-based, never -1.
                ci_maps = []
                for name in phase_names:
                    ci = np.array(cp[f"phases/{name}/ci"])
                    ci_maps.append(ci)
                ci_stack = np.stack(ci_maps, axis=0)
                best_phase_id = np.argmax(ci_stack, axis=0).astype(np.int16)

            # Build euler angles + CI from winning phase. -1 pixels never
            # match any i (0..N-1) so they stay at the zero initial value.
            euler_angles = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
            ci = np.zeros((n_rows, n_cols), dtype=np.float32)
            for i, name in enumerate(phase_names):
                mask = best_phase_id == i
                if f"phases/{name}/orientation" in cp:
                    euler_angles[mask] = np.array(cp[f"phases/{name}/orientation"])[mask]
                if f"phases/{name}/ci" in cp:
                    ci[mask] = np.array(cp[f"phases/{name}/ci"])[mask]

        # Flatten for CrystalMap.
        #
        # orix's PhaseList auto-assigns ids starting at 0 in construction
        # order, and CrystalMap looks up Phase metadata by matching the
        # per-pixel phase_id against PhaseList.ids. So we MUST keep
        # phase_id 0-based here (0 = first real phase, N-1 = last) and
        # use -1 for unindexed. The 1-based "0=unindexed, 1..N=phase"
        # convention is only for the on-disk CTF rows — we apply that
        # shift in _write_ctf, not here.
        #
        # Previous code shifted to 1-based BEFORE xmap construction,
        # which made orix map "phase 1 in the xmap" to PhaseList[1] (the
        # SECOND phase) instead of the first. The bug was hidden by a
        # silent cubic-default fallback in _phase_metadata; with that
        # fallback removed it now surfaces as "Phase X not found".
        n_pixels = n_rows * n_cols
        euler_flat = euler_angles.reshape(n_pixels, 3)
        phase_id_flat = best_phase_id.flatten().astype(int)  # -1=unindexed, 0..N-1=phase
        ci_flat = ci.flatten()

        # Build coordinates
        y_coords, x_coords = np.mgrid[0:n_rows, 0:n_cols]
        x_flat = (x_coords.flatten() * step_size).astype(float)
        y_flat = (y_coords.flatten() * step_size).astype(float)

        # Build phase list — CRUCIAL that each Phase carries its proper
        # point group / space group. Without it orix falls back to the
        # triclinic fundamental zone and IPF coloring turns into noise
        # (every quaternion maps to a different colour regardless of the
        # actual orientation). We store the symmetry per phase in the
        # checkpoint (see write_phase_result); pick it up here and fall
        # back to Phase.from_cif when only the .cif path is available.
        phases = []
        with h5py.File(str(cp_path), "r") as cp:
            for name in phase_names:
                phase_grp = cp.get(f"phases/{name}")
                sg = None; pg_name = None; phase_file = ""
                if phase_grp is not None:
                    sg = phase_grp.attrs.get("space_group", None)
                    pg_name = phase_grp.attrs.get("point_group", None)
                    if isinstance(pg_name, bytes):
                        pg_name = pg_name.decode("utf-8", errors="replace")
                    phase_file = phase_grp.attrs.get("phase_file", "")
                    if isinstance(phase_file, bytes):
                        phase_file = phase_file.decode("utf-8", errors="replace")
                built = None
                if sg is not None:
                    try:
                        built = Phase(name=name, space_group=int(sg))
                    except Exception as e:
                        logger.warning("Phase %s: space_group=%s rejected by orix (%s)", name, sg, e)
                if built is None and pg_name:
                    try:
                        built = Phase(name=name, point_group=str(pg_name))
                    except Exception as e:
                        logger.warning("Phase %s: point_group=%s rejected (%s)", name, pg_name, e)
                if built is None and phase_file and str(phase_file).lower().endswith(".cif"):
                    # Fallback: parse the original CIF. Handles older
                    # checkpoints that didn't store the symmetry attrs.
                    try:
                        from ebsd_utils import sanitize_cif
                        built = Phase.from_cif(sanitize_cif(phase_file))
                        built.name = name
                    except Exception as e:
                        logger.warning("Phase %s: CIF fallback failed (%s)", name, e)
                if built is None:
                    raise ValueError(
                        f"Phase {name!r}: no symmetry info available "
                        "(space_group / point_group / CIF all missing or "
                        "rejected by orix). Refusing to export — the .ang "
                        "would be saved against the triclinic fundamental "
                        "zone and the .ctf cannot pick a Laue class. Re-run "
                        "the batch with the current indexer so the "
                        "checkpoint captures point_group, or restore the "
                        "phase's CIF file at the recorded path."
                    )
                phases.append(built)
        phase_list = PhaseList(phases)

        # Build CrystalMap. Real per-pixel Band Contrast goes into
        # xmap.prop['iq'] so orix.io.save writes it into the ANG IQ
        # column. ANG has no MAD column we care about; orix fills CI
        # from xmap.prop['ci'] which we set below. Without injecting
        # 'iq', orix writes zeros and the .ang's IQ column is useless
        # in MTEX/EDAX OIM Analysis.
        prop: Dict[str, np.ndarray] = {"ci": ci_flat}
        bc_quality = _read_h5oina_quality(source_h5_path, n_pixels)
        bc_arr = bc_quality.get("bc")
        if bc_arr is not None:
            prop["iq"] = bc_arr.astype(np.float32)
        # When there is no native Band Contrast, orix writes zeros into the
        # ANG IQ column (no fake). Record that honestly in the header so the
        # empty IQ column isn't mistaken for a confidence surrogate.
        ang_iq_note = None
        if bc_arr is None:
            ang_iq_note = (
                "# NOTE: IQ column = 0 (no native Band Contrast in source; "
                "not a confidence surrogate)"
            )

        rotations = Rotation.from_euler(euler_flat, degrees=False)
        xmap = CrystalMap(
            rotations=rotations,
            phase_id=phase_id_flat,
            x=x_flat,
            y=y_flat,
            phase_list=phase_list,
            prop=prop,
            scan_unit="um",
        )

        # Export .ang — orix.io.save prompts on stdin for overwrite when the
        # file exists and a batch re-run would fail. Pass overwrite=True.
        # Skip entirely when the caller didn't ask for it so we don't leave
        # unwanted files on disk just because ctf was requested.
        ang_path = None
        if write_ang:
            ang_path = str(out_dir / f"result_{base_name}.ang")
            try:
                from orix.io import save
                save(ang_path, xmap, overwrite=True)
                # Post-edit the ANG header to inject TILT and PC from
                # the calibration store. orix's ANG writer does not take
                # detector geometry as input — adding it as additional
                # ``# KEY VALUE`` lines is the standard ANG convention
                # and MTEX / EDAX OIM tolerate (and parse) extras.
                _inject_ang_acquisition(
                    ang_path, sample_tilt=sample_tilt, pc=pc, note=ang_iq_note
                )
                logger.info("Exported .ang: %s", ang_path)
            except Exception as e:
                logger.warning("Failed to export .ang: %s", e)
                ang_path = None

        # Export .ctf — orix doesn't support ctf writing, so we emit it
        # manually. Same "write only if requested" rule as .ang so the
        # export_formats selection is honoured on disk.
        ctf_path = None
        if write_ctf:
            ctf_path = str(out_dir / f"result_{base_name}.ctf")
            try:
                _write_ctf(
                    ctf_path, xmap, phase_names, step_size,
                    source_h5_path=source_h5_path,
                    sample_tilt=sample_tilt,
                )
                logger.info("Exported .ctf: %s", ctf_path)
            except Exception as e:
                logger.warning("Failed to export .ctf: %s", e)
                ctf_path = None

        return ang_path, ctf_path

    except ValueError:
        # Bubble out — these are intentional refusals (missing step,
        # missing symmetry) that the caller's error dict captures.
        raise
    except Exception as e:
        logger.exception("Failed to export ang/ctf from checkpoint: %s", e)
        return None, None


def _copy_eds_full(source_h5_path: str, dst_file: "h5py.File") -> bool:
    """Copy the FULL EDS payload from a source h5oina into the light file.

    This includes the per-pixel raw Spectrum dataset — historically we
    skipped it because it is multi-GB and made the "light" name a lie.
    Phase Map's EDS-driven phase preselection (FEAT-10) needs the spectra
    available alongside the indexing data, otherwise the light file is
    only useful when the original h5oina sits in the same place.

    Approach: walk the source EDS group recursively and replicate every
    dataset and group into ``/EDS/`` of the destination, preserving names
    and chunk-friendly shapes. We compress everything (gzip default) so
    the file is still smaller than the raw input.

    Returns True on success, False if the source has no EDS group.
    Raises if the copy itself fails partway — we don't want a silent
    half-EDS file (the user explicitly asked for full data).
    """
    with h5py.File(source_h5_path, "r") as src:
        # H5OINA stores EDS under "1/EDS" (first scan group).
        src_eds = src.get("1/EDS")
        if src_eds is None:
            return False
        dst_eds = dst_file.create_group("EDS")
        dst_eds.attrs["description"] = (
            "Full EDS payload copied from the source h5oina: per-element "
            "Window Integral counts AND per-pixel raw Spectrum data plus "
            "all Header metadata. The .light file is fully self-contained "
            "for EDS-driven phase preselection workflows."
        )
        dst_eds.attrs["source_file"] = source_h5_path

        def _copy_node(src_node, dst_grp):
            for key in src_node:
                item = src_node[key]
                if isinstance(item, h5py.Group):
                    sub = dst_grp.create_group(key)
                    # Carry group attrs across (some H5OINA writers stash
                    # important metadata like quantification factors here).
                    for ak, av in item.attrs.items():
                        try:
                            sub.attrs[ak] = av
                        except Exception:
                            sub.attrs[ak] = str(av)
                    _copy_node(item, sub)
                else:
                    arr = np.asarray(item)
                    # Pick chunking that matches the source's chunk shape
                    # if the source was chunked; otherwise let h5py decide
                    # (auto-chunk is fine for vectors and small arrays).
                    chunks = item.chunks if item.chunks is not None else True
                    try:
                        ds = dst_grp.create_dataset(
                            key,
                            data=arr,
                            compression="gzip",
                            compression_opts=4,
                            chunks=chunks if arr.size > 0 else None,
                        )
                    except (TypeError, ValueError):
                        # Fall back to uncompressed for tiny / scalar
                        # datasets where chunking/compression isn't valid.
                        ds = dst_grp.create_dataset(key, data=arr)
                    for ak, av in item.attrs.items():
                        try:
                            ds.attrs[ak] = av
                        except Exception:
                            ds.attrs[ak] = str(av)

        _copy_node(src_eds, dst_eds)
        return True


def _read_vendor_from_h5(path: str) -> str:
    """Map an EBSD file's Manufacturer to oxford/edax/bruker, or "" if unknown.

    Used to write exported Euler in the vendor's stored frame (see
    orientation_frame.to_vendor_export_frame).
    """
    try:
        with h5py.File(path, "r") as f:
            if "Manufacturer" in f:
                raw = f["Manufacturer"][()]
                s = raw[0] if np.ndim(raw) else raw
                s = (s.decode() if isinstance(s, bytes) else str(s)).lower()
                if "oxford" in s:
                    return "oxford"
                if "edax" in s or "tsl" in s or "ametek" in s:
                    return "edax"
                if "bruker" in s:
                    return "bruker"
    except Exception:
        logger.debug("vendor read from %s failed", path, exc_info=True)
    return ""


def _euler_to_vendor_frame(euler_arr, vendor: str):
    """Right-multiply an (..., 3) Bunge-radian Euler array into the vendor frame."""
    from orix.quaternion import Rotation
    from backend.api.services.orientation_frame import to_vendor_export_frame
    arr = np.asarray(euler_arr, dtype=np.float64)
    shp = arr.shape
    return (
        to_vendor_export_frame(Rotation.from_euler(arr.reshape(-1, 3)), vendor)
        .to_euler()
        .reshape(shp)
        .astype(np.float32)
    )


def export_result_h5_light(
    source_h5_path: str,
    checkpoint_path: str,
    output_dir: str,
    indexing_params: Optional[Dict] = None,
    pc: Optional[List[float]] = None,
    sample_tilt: float = 70.0,
    detector_shape: Optional[Tuple[int, int]] = None,
    include_eds: bool = True,
    step_size: Optional[float] = None,
    scan_provenance: Optional[Dict] = None,
) -> str:
    """Create a lightweight H5 result file: indexing data only, no source copy.

    Unlike :func:`export_result_h5`, this does NOT copy the (potentially
    multi-GB) original h5oina patterns. It only writes:
      /Indexing/       — per-phase CI + orientations + auto-assignment
      /Detector/       — geometry
      /Documentation/  — README + phase table
      /SourceReference/  — file_path, stem, checksum (for re-linking)

    Typical size: 5–50 MB for a 2000×2000 scan with 3 phases, independent
    of whether the source is 200 MB or 200 GB.

    Use this format when you want Phase Map + Analysis + Refinement to work
    on the results without replicating terabytes of raw pattern data.
    """
    source_path = Path(source_h5_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if step_size is None:
        step_size = _read_step_size_from_checkpoint(checkpoint_path)

    # Vendor for export-frame conversion: our indexers emit the EMsoft/kikuchipy
    # common frame; Oxford/Bruker store Euler 90deg about ND from it. Writing in
    # the vendor frame makes a raw MTEX/h5read import match Aztec.
    export_vendor = _read_vendor_from_h5(source_h5_path)

    result_name = f"result_{source_path.stem}_light.h5"
    result_path = out_dir / result_name

    logger.info("Creating light result file: %s", result_path)

    # Read checkpoint data (same as the rich exporter)
    with h5py.File(checkpoint_path, "r") as cp:
        phases_grp = cp.get("phases")
        if phases_grp is None:
            logger.warning("No phases in checkpoint, cannot export")
            raise ValueError("No completed phases in checkpoint")

        phase_names = []
        phase_data = {}
        for name in phases_grp:
            if phases_grp[name].attrs.get("status") != "done":
                continue
            phase_names.append(name)
            _sg = phases_grp[name].attrs.get("space_group", None)
            _pg = phases_grp[name].attrs.get("point_group", None)
            if isinstance(_pg, bytes):
                _pg = _pg.decode("utf-8", errors="replace")
            phase_data[name] = {
                "ci": np.array(phases_grp[name]["ci"]),
                "orientation": np.array(phases_grp[name]["orientation"]),
                "ci_mean": float(phases_grp[name].attrs.get("ci_mean", 0.0)),
                "ci_median": float(phases_grp[name].attrs.get("ci_median", 0.0)),
                "phase_file": phases_grp[name].attrs.get("phase_file", ""),
                "duration_sec": float(phases_grp[name].attrs.get("duration_sec", 0.0)),
                "space_group": int(_sg) if _sg is not None else None,
                "point_group": str(_pg) if _pg else None,
            }

        auto_assign = {}
        if "auto_assignment" in cp:
            aa = cp["auto_assignment"]
            for key in aa:
                auto_assign[key] = np.array(aa[key])
            # Convention for the on-disk file: 1-based phase_id with 0 =
            # "not indexed", 1..N = real phases. This matches the reader in
            # analysis.py (_load_kikuchipy_rich_h5) and .ang/.ctf/MTEX.
            # We used to only translate when cleanup had run; otherwise the
            # raw 0-based best_phase_id slipped through, and on reload every
            # Al pixel (phase 0) was interpreted as unindexed — empty IPF.
            if "cleaned_phase_id" in aa:
                cleaned = auto_assign["cleaned_phase_id"]
                auto_assign["best_phase_id"] = np.where(
                    cleaned < 0, 0, cleaned + 1
                ).astype(np.uint8)
            elif "best_phase_id" in auto_assign:
                # Checkpoint's best_phase_id is always 0-based argmax —
                # shift up by one so the file on disk is 1-based. There is
                # no "unindexed" state in this branch (without cleanup).
                auto_assign["best_phase_id"] = (
                    auto_assign["best_phase_id"].astype(np.uint8) + 1
                )
            auto_assign["_one_based"] = True

        grid_shape = tuple(cp["metadata"].attrs.get("grid_shape", [0, 0]))
        method = cp["metadata"].attrs.get("method", "unknown")

    # Write a NEW (empty) file — no source copy
    with h5py.File(str(result_path), "w") as f:
        now = datetime.now(timezone.utc).isoformat()

        # Reference to the source (NOT the data itself)
        ref = f.create_group("SourceReference")
        ref.attrs["description"] = (
            "Pointer to the original source file. Data was NOT copied to save disk. "
            "Open the source directly if you need raw patterns."
        )
        ref.attrs["source_file_path"] = str(source_path.absolute())
        ref.attrs["source_file_name"] = source_path.name
        ref.attrs["source_file_stem"] = source_path.stem
        try:
            ref.attrs["source_size_mb"] = round(source_path.stat().st_size / (1024**2), 1)
        except OSError:
            pass

        # /Indexing/
        idx = f.create_group("Indexing")
        idx.attrs["method"] = method
        idx.attrs["software"] = "Orienta"
        idx.attrs["created"] = now
        idx.attrs["source_file"] = str(source_path.name)
        idx.attrs["n_phases"] = len(phase_names)
        idx.attrs["grid_shape"] = list(grid_shape)
        idx.attrs["format"] = "light"
        idx.attrs["format_version"] = FORMAT_VERSION
        _write_scan_provenance(idx, scan_provenance)
        # Store the µm step size on /Indexing AND /Detector below so any
        # reader path finds it without guessing. The reader in
        # analysis.py uses this to scale CrystalMap.x/y from pixel units
        # to µm, which is what the PC-Refinement axes manager needs.
        idx.attrs["step_size_um"] = float(step_size)
        # Orientation export-frame tag (see _read_vendor_from_h5). Euler below
        # is written in the vendor frame; the reader inverts via source_vendor.
        from backend.api.services.orientation_frame import (
            export_frame_offset as _exp_off,
        )
        idx.attrs["source_vendor"] = export_vendor or "unknown"
        idx.attrs["orientation_reference_frame"] = (
            "vendor_stored (Aztec/MTEX default import)"
            if float(_exp_off(export_vendor).angle.max()) > 1e-6
            else "native (EMsoft/kikuchipy common)"
        )

        # Per-pixel X/Y µm coordinates so MTEX (and any generic HDF5 reader)
        # can place the orientations on a grid without the source h5oina.
        # X = column · step, Y = row · step — same convention as the .ang
        # writer (export_ang_ctf).
        _n_rows, _n_cols = grid_shape
        _cc, _rr = np.meshgrid(np.arange(_n_cols), np.arange(_n_rows))
        _xds = idx.create_dataset(
            "X", data=(_cc * float(step_size)).astype(np.float32),
            compression="gzip",
        )
        _xds.attrs["unit"] = "um"
        _xds.attrs["description"] = "Sample X (column · step_size_um)"
        _yds = idx.create_dataset(
            "Y", data=(_rr * float(step_size)).astype(np.float32),
            compression="gzip",
        )
        _yds.attrs["unit"] = "um"
        _yds.attrs["description"] = "Sample Y (row · step_size_um)"

        # Per-Phase
        per_phase = idx.create_group("PerPhase")
        for i, name in enumerate(phase_names):
            pd = phase_data[name]
            pg = per_phase.create_group(name)
            pg.attrs["phase_id"] = i + 1
            pg.attrs["phase_file"] = pd["phase_file"]
            pg.attrs["ci_mean"] = pd["ci_mean"]
            pg.attrs["ci_median"] = pd["ci_median"]
            pg.attrs["duration_sec"] = pd["duration_sec"]
            pg.create_dataset(
                "euler_angles",
                data=_euler_to_vendor_frame(pd["orientation"], export_vendor),
                dtype=np.float32, compression="gzip")
            pg.create_dataset("confidence_index", data=pd["ci"],
                              dtype=np.float32, compression="gzip")

        # Phase table — include crystallographic symmetry so readers can
        # rebuild an IPF-capable Phase object on load. Without this the
        # reader falls back to triclinic and the IPF view becomes noise.
        phases_tbl = idx.create_group("Phases")
        for i, name in enumerate(phase_names):
            pg = phases_tbl.create_group(str(i + 1))
            pg.attrs["name"] = name
            pg.attrs["phase_file"] = phase_data[name]["phase_file"]
            pg.attrs["ci_mean"] = phase_data[name]["ci_mean"]
            if phase_data[name].get("space_group") is not None:
                pg.attrs["space_group"] = int(phase_data[name]["space_group"])
            if phase_data[name].get("point_group"):
                pg.attrs["point_group"] = str(phase_data[name]["point_group"])

        # Auto-Assignment
        if auto_assign:
            aa_grp = idx.create_group("Assignment")
            if "best_phase_id" in auto_assign:
                ds = aa_grp.create_dataset("phase_id", data=auto_assign["best_phase_id"],
                                           dtype=np.uint8, compression="gzip")
                ds.attrs["phase_names"] = phase_names
            if "best_ci" in auto_assign:
                aa_grp.create_dataset("confidence_index", data=auto_assign["best_ci"],
                                      dtype=np.float32, compression="gzip")
            if "uncertainty" in auto_assign:
                aa_grp.create_dataset("uncertainty", data=auto_assign["uncertainty"],
                                      dtype=np.float32, compression="gzip")
            if "confident_mask" in auto_assign:
                aa_grp.create_dataset("confident_mask", data=auto_assign["confident_mask"],
                                      compression="gzip")
            if "best_phase_id" in auto_assign and phase_names:
                best_ids = auto_assign["best_phase_id"]
                # CRITICAL: when cleaned_phase_id was applied above we
                # rewrote best_phase_id to the 1-based scheme (0=unindexed,
                # 1..N=phase). That means `best_ids == i` with i enumerating
                # phase_names (0..N-1) points at the WRONG pixels — i=0
                # matches unindexed and every real phase is off by one.
                # Root cause of the "batch IPF looks like noise" bug: each
                # phase's orientation landed on the next phase's pixels.
                # Honour the flag and offset accordingly.
                one_based = bool(auto_assign.get("_one_based", False))
                euler_best = np.zeros((*grid_shape, 3), dtype=np.float32)
                for i, name in enumerate(phase_names):
                    target = (i + 1) if one_based else i
                    mask = best_ids == target
                    euler_best[mask] = phase_data[name]["orientation"][mask]
                aa_grp.create_dataset(
                    "euler_angles",
                    data=_euler_to_vendor_frame(euler_best, export_vendor),
                    dtype=np.float32, compression="gzip")

            # Per-pixel quality fields (BC, MAD, PC X/Y, DD, Bands) copied
            # from the source h5oina. Cheap (a few MB total) but unlocks
            # quality maps, BC-GMM/RX analysis, per-pixel PC refinement,
            # and Aztec-vs-ours diagnostic comparisons — without keeping
            # the (multi-GB) source open.
            copied = _copy_quality_fields(source_h5_path, aa_grp, grid_shape)
            if copied:
                aa_grp.attrs["quality_fields"] = copied
                logger.info("Copied quality fields from source: %s", copied)

        # Parameters
        params = idx.create_group("Parameters")
        params.attrs["method"] = method
        if indexing_params:
            for k, v in indexing_params.items():
                try: params.attrs[k] = v
                except TypeError: params.attrs[k] = str(v)
        if pc is not None:
            params.create_dataset("pc", data=np.array(pc, dtype=np.float32))
        params.attrs["sample_tilt_deg"] = sample_tilt
        params.attrs["step_size_um"] = float(step_size)
        if detector_shape:
            params.attrs["detector_shape"] = list(detector_shape)

        # /Detector/
        det = f.create_group("Detector")
        if pc is not None:
            det.create_dataset("pc", data=np.array(pc, dtype=np.float32))
        det.attrs["sample_tilt_deg"] = sample_tilt
        det.attrs["step_size_um"] = float(step_size)
        if detector_shape:
            det.attrs["shape"] = list(detector_shape)

        # /Documentation/
        doc = f.create_group("Documentation")
        doc.attrs["format_version"] = FORMAT_VERSION
        _write_scan_provenance(doc, scan_provenance)
        doc.attrs["format"] = "light"
        doc.attrs["description"] = (
            "Lightweight indexing result. Original pattern data is NOT included — "
            "see /SourceReference/source_file_path for the source file location."
        )

        # /EDS/ — full payload by default (per-pixel Spectrum + Window
        # Integral counts + Header). Opt-out via include_eds=False if
        # disk space is the constraint and the source file will remain
        # accessible. Default-on so the EDS-driven phase-preselection
        # workflow (FEAT-10) works without manual configuration.
        if include_eds:
            if _copy_eds_full(source_h5_path, f):
                doc.attrs["eds_included"] = True
                doc.attrs["eds_full"] = True
                logger.info("Full EDS payload copied into light result file")
            else:
                doc.attrs["eds_included"] = False
                logger.info("include_eds requested but source has no EDS group")
        else:
            doc.attrs["eds_included"] = False

    file_size_mb = result_path.stat().st_size / (1024 * 1024)
    logger.info("Light result file: %s (%.1f MB)", result_path, file_size_mb)
    return str(result_path)


def export_all(
    source_h5_path: str,
    checkpoint_path: str,
    output_dir: Optional[str] = None,
    indexing_params: Optional[Dict] = None,
    pc: Optional[List[float]] = None,
    sample_tilt: float = 70.0,
    detector_shape: Optional[Tuple[int, int]] = None,
    step_size: Optional[float] = None,
    formats: Optional[List[str]] = None,
    include_eds: bool = True,
    scan_provenance: Optional[Dict] = None,
) -> Dict[str, Optional[str]]:
    """Export selected formats: rich .h5, light .h5, .ang, .ctf.

    Parameters
    ----------
    source_h5_path : str
        Original h5oina file.
    checkpoint_path : str
        _multiphase.h5 from batch indexing.
    output_dir : str, optional
        Output directory. Defaults to same dir as source file.
    scan_provenance : dict, optional
        Where the indexed dataset sat in the original scan; forwarded to the
        two .h5 writers (the .ang/.ctf writers have no place for it). ``None``
        writes nothing, and ``None`` is correct for every caller that exports
        files it did not just index — see the note in :func:`export_result_h5`.
    formats : list of str, optional
        Subset of {"h5_rich", "h5_light", "ang", "ctf"}. Default:
        ["h5_light", "ang", "ctf"] — the small-output set, safe for
        multi-GB sources. Include "h5_rich" explicitly when you need the
        source patterns replicated alongside the indexing data.

    Returns
    -------
    dict with keys: 'h5_rich', 'h5_light', 'ang', 'ctf' — paths or None.
    """
    if output_dir is None:
        output_dir = str(Path(source_h5_path).parent)
    if formats is None:
        formats = ["h5_light", "ang", "ctf"]

    # Resolve step_size once so every output stamps the same scale.
    # Raises ValueError if the checkpoint never recorded a step — better
    # to fail the whole export than to write a mix of right and wrong
    # files. The .ang/.ctf carry coordinates in µm; the .h5_light carries
    # step_size_um in /Indexing attrs; the reader (analysis.py) scales
    # the rebuilt CrystalMap against this. Diverging values would make
    # the symptom "axes manager is gone in PC refinement" come back.
    if step_size is None:
        step_size = _read_step_size_from_checkpoint(checkpoint_path)
    elif step_size <= 0.0:
        raise ValueError(
            f"export_all called with step_size={step_size!r}; refusing "
            "the export to avoid coordinates that would be unusable."
        )

    results: Dict[str, Optional[str]] = {
        "h5_rich": None, "h5_light": None, "ang": None, "ctf": None
    }
    errors: Dict[str, str] = {}

    if "h5_rich" in formats:
        try:
            results["h5_rich"] = export_result_h5(
                source_h5_path, checkpoint_path, output_dir,
                indexing_params=indexing_params, pc=pc,
                sample_tilt=sample_tilt, detector_shape=detector_shape,
                step_size=step_size,
                scan_provenance=scan_provenance,
            )
        except Exception as e:
            logger.exception("Rich H5 export failed: %s", e)
            errors["h5_rich"] = str(e)

    if "h5_light" in formats:
        try:
            results["h5_light"] = export_result_h5_light(
                source_h5_path, checkpoint_path, output_dir,
                indexing_params=indexing_params, pc=pc,
                sample_tilt=sample_tilt, detector_shape=detector_shape,
                include_eds=include_eds,
                step_size=step_size,
                scan_provenance=scan_provenance,
            )
        except Exception as e:
            logger.exception("Light H5 export failed: %s", e)
            errors["h5_light"] = str(e)

    if "ang" in formats or "ctf" in formats:
        try:
            ang, ctf = export_ang_ctf(
                checkpoint_path, output_dir, step_size=step_size,
                write_ang=("ang" in formats),
                write_ctf=("ctf" in formats),
                source_h5_path=source_h5_path,
                # Calibrated values from CalibrationStore land here so
                # the .ang TILT / x-star / y-star / z-star header lines
                # and the .ctf TiltAngle reflect the user's refined PC,
                # not the (often outdated) h5oina acquisition values.
                sample_tilt=sample_tilt,
                pc=pc,
            )
            results["ang"] = ang
            results["ctf"] = ctf
        except Exception as e:
            logger.exception("ANG/CTF export failed: %s", e)
            errors["ang_ctf"] = str(e)

    # Surface per-format error reasons so callers/UI can distinguish
    # "not requested" (path None, no error key) from "failed" (path None
    # plus an error string). Previously failures dropped silently into
    # the backend log and the frontend just saw a missing file.
    if errors:
        results["_errors"] = errors  # type: ignore[assignment]
    return results
