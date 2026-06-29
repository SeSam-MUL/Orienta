"""Checkpoint writer for multi-phase batch indexing results.

Manages _multiphase.h5 files — one per source EBSD file.
Each phase result is appended incrementally for crash recovery.
"""
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]


class CheckpointWriter:
    """Read/write _multiphase.h5 checkpoint files.

    File layout:
      /metadata/   — batch_id, grid_shape, phases_list, etc.
      /phases/<name>/ci          — float32 [rows, cols]
      /phases/<name>/orientation — float32 [rows, cols, 3]
      /auto_assignment/          — computed after all phases
      /manual_override/          — written by refinement UI
      /detector/                 — PC and detector info
    """

    def __init__(self, source_file_path: str):
        self.source_file_path = source_file_path
        self.checkpoint_path = self._derive_path(source_file_path)
        self._cleanup_tmp()

    def _derive_path(self, source_path: str) -> str:
        p = Path(source_path)
        return str(p.parent / f"{p.stem}_multiphase.h5")

    def _cleanup_tmp(self):
        """Remove stale .tmp files from prior crashes."""
        tmp = self.checkpoint_path + ".tmp"
        if os.path.exists(tmp):
            logger.warning("Removing stale tmp file: %s", tmp)
            os.remove(tmp)

    def checkpoint_path_exists(self) -> bool:
        return os.path.isfile(self.checkpoint_path)

    def init_metadata(self, grid_shape: tuple, batch_id: str, method: str,
                      step_size_um: float = 0.0):
        """Initialize the checkpoint file with metadata. Idempotent."""
        if self.checkpoint_path_exists():
            # Already initialized — just update last_modified
            with h5py.File(self.checkpoint_path, "a") as f:
                if "metadata" in f:
                    f["metadata"].attrs["last_modified"] = datetime.now(timezone.utc).isoformat()
            return

        with h5py.File(self.checkpoint_path, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["source_file"] = self.source_file_path
            meta.attrs["source_file_name"] = Path(self.source_file_path).name
            meta.attrs["batch_id"] = batch_id
            meta.attrs["created"] = datetime.now(timezone.utc).isoformat()
            meta.attrs["last_modified"] = datetime.now(timezone.utc).isoformat()
            meta.attrs["grid_shape"] = list(grid_shape)
            meta.attrs["step_size_um"] = step_size_um
            meta.attrs["method"] = method
            meta.attrs["version"] = 1
            f.create_group("phases")

    def phase_already_done(self, phase_name: str) -> bool:
        """Check if a phase has completed results in the checkpoint."""
        if not self.checkpoint_path_exists():
            return False
        try:
            with h5py.File(self.checkpoint_path, "r") as f:
                grp = f.get(f"phases/{phase_name}")
                if grp is None:
                    return False
                return grp.attrs.get("status", "") == "done"
        except Exception:
            return False

    def get_phase_metadata(self, phase_name: str) -> Optional[Dict]:
        """Read cached CI stats for a phase, or None if missing.

        Used by batch_manager to populate the jobs table when a phase is
        skipped because it was already done (BUG-N, 2026-04-21 — previously
        the skip path wrote ci_mean=0.0, making completed rows look empty).
        """
        if not self.checkpoint_path_exists():
            return None
        try:
            with h5py.File(self.checkpoint_path, "r") as f:
                grp = f.get(f"phases/{phase_name}")
                if grp is None or grp.attrs.get("status", "") != "done":
                    return None
                return {
                    "ci_mean": float(grp.attrs.get("ci_mean", 0.0)),
                    "ci_median": float(grp.attrs.get("ci_median", 0.0)),
                    "duration_sec": float(grp.attrs.get("duration_sec", 0.0)),
                }
        except Exception:
            return None

    def get_completed_phases(self) -> List[str]:
        """List all phases with status='done'."""
        if not self.checkpoint_path_exists():
            return []
        result = []
        try:
            with h5py.File(self.checkpoint_path, "r") as f:
                phases_grp = f.get("phases")
                if phases_grp is None:
                    return []
                for name in phases_grp:
                    if phases_grp[name].attrs.get("status", "") == "done":
                        result.append(name)
        except Exception:
            pass
        return result

    def write_phase_result(self, phase_name: str, ci_map: np.ndarray,
                           orientation_map: np.ndarray, metadata: Dict):
        """Write one phase result to the checkpoint file.

        Parameters
        ----------
        phase_name : str
        ci_map : ndarray, float32, shape (rows, cols)
        orientation_map : ndarray, float32, shape (rows, cols, 3) — Euler angles
        metadata : dict with keys: phase_file, ci_mean, duration_sec,
            space_group (int), point_group (str, e.g. "m-3m"), etc.

        The crystallographic symmetry attrs are critical for IPF colouring
        on export: without them orix treats the phase as triclinic and the
        IPF map becomes quasi-random colour noise.
        """
        with h5py.File(self.checkpoint_path, "a") as f:
            phase_path = f"phases/{phase_name}"
            if phase_path in f:
                del f[phase_path]  # Overwrite if re-running
            grp = f.create_group(phase_path)
            grp.create_dataset("ci", data=ci_map.astype(np.float32), compression="gzip")
            grp.create_dataset("orientation", data=orientation_map.astype(np.float32), compression="gzip")
            grp.attrs["status"] = "done"
            grp.attrs["phase_file"] = metadata.get("phase_file", "")
            grp.attrs["ci_mean"] = float(metadata.get("ci_mean", 0.0))
            grp.attrs["ci_median"] = float(metadata.get("ci_median", 0.0))
            grp.attrs["duration_sec"] = float(metadata.get("duration_sec", 0.0))
            grp.attrs["started_at"] = metadata.get("started_at", "")
            grp.attrs["finished_at"] = datetime.now(timezone.utc).isoformat()
            # Persist the symmetry so exports can round-trip it.
            sg = metadata.get("space_group")
            pg = metadata.get("point_group")
            if sg is not None:
                try: grp.attrs["space_group"] = int(sg)
                except Exception: pass
            if pg is not None:
                grp.attrs["point_group"] = str(pg)
            if "indexing_params" in metadata:
                grp.attrs["indexing_params"] = json.dumps(metadata["indexing_params"])

            # Update metadata
            f["metadata"].attrs["last_modified"] = datetime.now(timezone.utc).isoformat()
            # Update phases_list
            phase_names = [n for n in f["phases"]]
            f["metadata"].attrs["phases_list"] = phase_names

        logger.info("Checkpoint: wrote phase '%s' to %s", phase_name, self.checkpoint_path)

    @staticmethod
    def clean_phase_assignment(
        best_phase_id: np.ndarray,
        best_ci: np.ndarray,
        uncertainty: np.ndarray,
        ci_threshold: float = 0.0,
        uncertainty_threshold: float = 0.0,
        min_cluster_size: int = 0,
        fill_unindexed: bool = False,
        modal_filter_size: int = 0,
    ) -> np.ndarray:
        """Apply MTEX-style post-processing filters to a phase assignment map.

        Pure function — takes the raw assignment arrays and returns a filtered
        phase_id map with a sentinel value of -1 for "cleaned-out" pixels.

        Parameters
        ----------
        best_phase_id : (rows, cols) uint8
            Current best-phase assignment, 0-based phase ids (0..N-1).
        best_ci, uncertainty : (rows, cols) float32
            Paired maps used for the CI and uncertainty filters.
        ci_threshold : float
            Pixels with ``best_ci < ci_threshold`` become -1 (unindexed).
        uncertainty_threshold : float
            Pixels with ``uncertainty < uncertainty_threshold`` become -1.
        min_cluster_size : int
            Phase clusters (4-connected same-id regions) with fewer pixels than
            this threshold are relabelled to the majority phase ID of their
            immediate neighbours. Removes single-pixel speckle without leaving
            holes.
        fill_unindexed : bool
            After other filters run, fill remaining -1 pixels with the phase ID
            of the nearest indexed neighbour (distance transform). Closes the
            pinholes left by CI / uncertainty cutoffs.
        modal_filter_size : int
            Size of the majority-vote (modal) filter window. 0 disables, 3 or 5
            applies a 3×3 or 5×5 mode filter. Replaces each pixel with the most
            common phase ID in its neighbourhood — the standard salt-and-pepper
            removal for EBSD phase maps. Only votes from indexed neighbours
            count; pixels with no indexed neighbour are left at -1.
        """
        from scipy import ndimage as ndi

        out = best_phase_id.astype(np.int16).copy()  # int16 so we can store -1

        # 1. CI threshold
        if ci_threshold > 0:
            out[best_ci < ci_threshold] = -1

        # 2. Uncertainty threshold
        if uncertainty_threshold > 0:
            out[uncertainty < uncertainty_threshold] = -1

        # 3. Small-cluster removal (4-connectivity per phase id)
        if min_cluster_size > 0:
            for pid in np.unique(out):
                if pid < 0:
                    continue
                mask = out == pid
                labeled, n_labels = ndi.label(mask)
                if n_labels == 0:
                    continue
                sizes = ndi.sum(mask, labeled, index=np.arange(1, n_labels + 1))
                small_labels = np.where(sizes < min_cluster_size)[0] + 1
                if len(small_labels) == 0:
                    continue
                small_mask = np.isin(labeled, small_labels)
                # Relabel small clusters by dilating the remaining phases;
                # each small-cluster pixel inherits from the nearest non-small
                # neighbour via the "distance transform with labels" trick.
                valid = ~small_mask
                if valid.any():
                    idx_valid = ndi.distance_transform_edt(
                        ~valid, return_distances=False, return_indices=True
                    )
                    out[small_mask] = out[tuple(idx_valid[:, small_mask])]
                else:
                    out[small_mask] = -1

        # 4. Modal (majority-vote) filter — MTEX-style salt-and-pepper removal
        if modal_filter_size and modal_filter_size >= 3:
            size = int(modal_filter_size)
            # Build a per-phase vote stack: one binary map per phase, then
            # convolve each with a uniform kernel to count neighbours of that
            # phase. The phase with the most votes wins.
            unique_ids = np.unique(out)
            valid_ids = unique_ids[unique_ids >= 0]
            if valid_ids.size > 0:
                kernel = np.ones((size, size), dtype=np.float32)
                vote_stack = np.zeros((valid_ids.size, *out.shape), dtype=np.float32)
                for k, pid in enumerate(valid_ids):
                    vote_stack[k] = ndi.convolve(
                        (out == pid).astype(np.float32), kernel, mode="constant", cval=0.0
                    )
                # Argmax across phase axis → new phase id per pixel
                best_phase_idx = np.argmax(vote_stack, axis=0)
                new_out = valid_ids[best_phase_idx]
                # Preserve pixels that had zero indexed neighbours → still -1
                any_vote = vote_stack.sum(axis=0) > 0
                out = np.where(any_vote, new_out, -1).astype(np.int16)

        # 5. Fill remaining unindexed pixels from nearest indexed neighbour
        if fill_unindexed and (out == -1).any() and (out >= 0).any():
            valid = out >= 0
            idx_valid = ndi.distance_transform_edt(
                ~valid, return_distances=False, return_indices=True
            )
            out[~valid] = out[tuple(idx_valid[:, ~valid])]

        return out

    def apply_cleanup(
        self,
        ci_threshold: float = 0.0,
        uncertainty_threshold: float = 0.0,
        min_cluster_size: int = 0,
    ) -> Dict[str, int]:
        """Persist a cleanup pass into the checkpoint's auto_assignment group.

        Writes a new ``cleaned_phase_id`` dataset plus ``cleanup_*`` attrs
        describing the settings. The original ``best_phase_id`` is preserved
        so cleanup can be re-applied with different thresholds without
        cascading information loss.

        Returns counts: {before, after_unindexed, moved}
        """
        if not self.checkpoint_path_exists():
            return {"before": 0, "after_unindexed": 0, "moved": 0}
        with h5py.File(self.checkpoint_path, "a") as f:
            aa = f.get("auto_assignment")
            if aa is None:
                return {"before": 0, "after_unindexed": 0, "moved": 0}
            best_phase_id = np.array(aa["best_phase_id"])
            best_ci       = np.array(aa["best_ci"])
            uncertainty   = np.array(aa["uncertainty"])
            cleaned = self.clean_phase_assignment(
                best_phase_id, best_ci, uncertainty,
                ci_threshold=ci_threshold,
                uncertainty_threshold=uncertainty_threshold,
                min_cluster_size=min_cluster_size,
            )
            if "cleaned_phase_id" in aa:
                del aa["cleaned_phase_id"]
            ds = aa.create_dataset(
                "cleaned_phase_id", data=cleaned, dtype=np.int16, compression="gzip",
            )
            ds.attrs["description"] = (
                "Post-processed best_phase_id: -1 = unindexed/cleaned out. "
                "Use this for visualisation; best_phase_id is the raw data."
            )
            aa.attrs["cleanup_ci_threshold"] = float(ci_threshold)
            aa.attrs["cleanup_uncertainty_threshold"] = float(uncertainty_threshold)
            aa.attrs["cleanup_min_cluster_size"] = int(min_cluster_size)
            aa.attrs["cleanup_last_modified"] = datetime.now(timezone.utc).isoformat()

            before = int((best_phase_id >= 0).sum())
            after_unindexed = int((cleaned < 0).sum())
            moved = int(((best_phase_id != cleaned) & (cleaned >= 0)).sum())

        return {"before": before, "after_unindexed": after_unindexed, "moved": moved}

    def compute_auto_assignment(self, confidence_threshold: float = 0.3):
        """Compute best_phase_id, uncertainty, confident_mask from all phase CIs."""
        all_cis = self.read_all_phase_cis()
        if not all_cis:
            return

        phase_names = list(all_cis.keys())
        grid_shape = list(all_cis.values())[0].shape
        n_phases = len(phase_names)

        # Stack CI maps: shape (n_phases, rows, cols)
        ci_stack = np.stack([all_cis[name] for name in phase_names], axis=0)

        # Best phase per pixel
        best_idx = np.argmax(ci_stack, axis=0).astype(np.uint8)
        best_ci = np.max(ci_stack, axis=0).astype(np.float32)

        # Second best
        if n_phases >= 2:
            # Set best to -inf, find next best.
            # Previous implementation was an O(n_rows * n_cols) Python double
            # loop that took ~30s for a 2000x2000 scan. Vectorized fancy-index
            # assignment does the same thing in one numpy call (<1s).
            ci_masked = ci_stack.copy()
            r_idx, c_idx = np.indices(grid_shape)
            ci_masked[best_idx, r_idx, c_idx] = -np.inf
            second_idx = np.argmax(ci_masked, axis=0).astype(np.uint8)
            second_ci = np.max(ci_masked, axis=0).astype(np.float32)
            second_ci = np.where(np.isinf(second_ci), 0.0, second_ci)
        else:
            second_idx = np.zeros_like(best_idx)
            second_ci = np.zeros_like(best_ci)

        uncertainty = (best_ci - second_ci).astype(np.float32)
        confident = uncertainty > confidence_threshold

        with h5py.File(self.checkpoint_path, "a") as f:
            if "auto_assignment" in f:
                del f["auto_assignment"]
            aa = f.create_group("auto_assignment")
            aa.create_dataset("best_phase_id", data=best_idx, compression="gzip")
            aa.create_dataset("best_ci", data=best_ci, compression="gzip")
            aa.create_dataset("second_phase_id", data=second_idx, compression="gzip")
            aa.create_dataset("second_ci", data=second_ci, compression="gzip")
            aa.create_dataset("uncertainty", data=uncertainty, compression="gzip")
            aa.create_dataset("confident_mask", data=confident, compression="gzip")

        logger.info("Auto-assignment computed for %s (%d phases)", self.checkpoint_path, n_phases)

    def read_phase_ci(self, phase_name: str) -> Optional[np.ndarray]:
        """Read CI map for one phase."""
        with h5py.File(self.checkpoint_path, "r") as f:
            ds = f.get(f"phases/{phase_name}/ci")
            return np.array(ds) if ds is not None else None

    def read_all_phase_cis(self) -> Dict[str, np.ndarray]:
        """Read CI maps for all completed phases."""
        result = {}
        if not self.checkpoint_path_exists():
            return result
        with h5py.File(self.checkpoint_path, "r") as f:
            phases_grp = f.get("phases")
            if phases_grp is None:
                return result
            for name in phases_grp:
                if phases_grp[name].attrs.get("status") == "done":
                    result[name] = np.array(phases_grp[name]["ci"])
        return result

    def write_preprocessing_meta(self, requested: Dict, applied: Dict) -> None:
        """Record which preprocessing steps were requested vs. actually applied.

        Written to ``/metadata/preprocessing/`` as two JSON attrs. This is how
        the Report UI and Refinement page verify what was done to the signal
        before indexing — previously this information was only in the backend
        logs (invisible to the user).
        """
        if not self.checkpoint_path_exists():
            return
        with h5py.File(self.checkpoint_path, "a") as f:
            if "metadata" not in f:
                return
            meta = f["metadata"]
            # Use JSON strings so mixed-type dicts (bool, int, str) survive
            meta.attrs["preprocessing_requested"] = json.dumps(requested or {})
            meta.attrs["preprocessing_applied"] = json.dumps(applied or {})

    def get_preprocessing_meta(self) -> Optional[Dict]:
        """Read back the preprocessing metadata written during the batch run."""
        if not self.checkpoint_path_exists():
            return None
        try:
            with h5py.File(self.checkpoint_path, "r") as f:
                if "metadata" not in f:
                    return None
                meta = f["metadata"]
                req = meta.attrs.get("preprocessing_requested")
                app = meta.attrs.get("preprocessing_applied")
                if req is None and app is None:
                    return None
                return {
                    "requested": json.loads(req) if req else {},
                    "applied":   json.loads(app) if app else {},
                }
        except Exception:
            return None

    def write_manual_override(self, phase_id_map: np.ndarray, source_map: np.ndarray):
        """Write manual overrides from the Refinement UI.

        phase_id_map is stored as int16 (range -32768..32767) instead of
        int8 (-128..127) so projects with > 127 phases don't silently
        overflow into negative ids. -1 still encodes "no override".
        """
        with h5py.File(self.checkpoint_path, "a") as f:
            if "manual_override" in f:
                del f["manual_override"]
            mo = f.create_group("manual_override")
            mo.create_dataset("phase_id", data=phase_id_map.astype(np.int16), compression="gzip")
            mo.create_dataset("override_source", data=source_map.astype(np.uint8), compression="gzip")
            mo.attrs["last_modified"] = datetime.now(timezone.utc).isoformat()

    def validate(self) -> ValidationResult:
        """Check file integrity: HDF5 readable, datasets consistent."""
        errors = []
        if not self.checkpoint_path_exists():
            return ValidationResult(valid=False, errors=["File does not exist"])
        try:
            with h5py.File(self.checkpoint_path, "r") as f:
                if "metadata" not in f:
                    errors.append("Missing /metadata group")
                if "phases" not in f:
                    errors.append("Missing /phases group")
                else:
                    grid_shape = None
                    if "metadata" in f:
                        raw_shape = f["metadata"].attrs.get("grid_shape", None)
                        if raw_shape is not None and len(raw_shape) >= 2:
                            grid_shape = tuple(int(v) for v in raw_shape)
                    for name in f["phases"]:
                        phase_grp = f[f"phases/{name}"]
                        if "ci" not in phase_grp:
                            errors.append(f"Phase '{name}' missing ci dataset")
                        elif grid_shape is not None and phase_grp["ci"].shape != grid_shape:
                            # Only compare when we actually have a valid grid_shape —
                            # earlier code used an empty-tuple fallback which made
                            # every phase's shape "mismatch" with an empty grid.
                            errors.append(f"Phase '{name}' ci shape mismatch: {phase_grp['ci'].shape} vs {grid_shape}")
        except Exception as e:
            errors.append(f"Cannot open HDF5: {e}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)
