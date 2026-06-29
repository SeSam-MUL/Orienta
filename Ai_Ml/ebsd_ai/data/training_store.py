"""Local training data storage backed by HDF5.

The ``TrainingStore`` manages a collection of EBSD training samples
stored as compressed HDF5 files. Each sample contains:
- Original pattern (variable size, uint8/16)
- Normalized pattern (fixed 128x128, float32)
- EDS vector (92 elements, float32; all zeros if no EDS)
- Detector metadata
- Phase label, orientation, confidence score
- Data source and provenance
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from ebsd_ai.config import (
    DataSource,
    DetectorInfo,
    eds_dict_to_vector,
)
from ebsd_ai.data.pattern_io import normalize_pattern, resize_pattern

# ---------------------------------------------------------------------------
# Storage keys (HDF5 dataset names within each sample group)
# ---------------------------------------------------------------------------

_K_PATTERN_ORIG = "pattern_original"
_K_PATTERN_NORM = "pattern_normalized"
_K_EDS = "eds_atomic_pct"
_K_HAS_EDS = "has_eds"
_K_PHASE = "confirmed_phase"
_K_ORIENTATION = "orientation_quaternion"
_K_CONFIDENCE = "confidence_score"
_K_SOURCE = "data_source"
_K_TIMESTAMP = "timestamp"
_K_SOURCE_FILE = "source_file"
_K_PATTERN_TARGET = "pattern_target"
_K_PAIR_TYPE = "pair_type"


def _sample_id(idx: int) -> str:
    """Return a zero-padded sample group name."""
    return f"sample_{idx:08d}"


# ---------------------------------------------------------------------------
# TrainingStore
# ---------------------------------------------------------------------------


class TrainingStore:
    """HDF5-backed store for EBSD training samples.

    Parameters
    ----------
    local_path : str or Path
        Directory where HDF5 shard files are stored. Created if missing.
    samples_per_shard : int
        Maximum samples per HDF5 file before starting a new shard.
    """

    def __init__(
        self,
        local_path: str | Path,
        samples_per_shard: int = 5000,
    ) -> None:
        self.local_path = Path(local_path)
        self.local_path.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self._lock = threading.RLock()

    # -- Internal helpers ---------------------------------------------------

    def _shard_files(self) -> list[Path]:
        """Return sorted list of shard HDF5 files."""
        return sorted(self.local_path.glob("shard_*.h5"))

    def _current_shard(self) -> Path:
        """Return the current (latest) shard file, creating one if needed."""
        shards = self._shard_files()
        if not shards:
            return self._new_shard()
        last = shards[-1]
        with self._lock, h5py.File(last, "r") as f:
            if len(f.keys()) >= self.samples_per_shard:
                return self._new_shard()
        return last

    def _new_shard(self) -> Path:
        """Create a new shard file with a unique name."""
        ts = int(time.time() * 1000)
        h = hashlib.md5(str(ts).encode()).hexdigest()[:8]
        path = self.local_path / f"shard_{ts}_{h}.h5"
        with h5py.File(path, "w") as f:
            f.attrs["created"] = ts
        return path

    def clear(self) -> int:
        """Delete every shard file, emptying the store. Returns the number of
        shard files removed. Used to purge stale/mislabelled samples that would
        otherwise poison future models and slow training."""
        removed = 0
        with self._lock:
            for shard in self._shard_files():
                try:
                    shard.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    # -- Public API: add data -----------------------------------------------

    def add_sample(
        self,
        pattern: np.ndarray,
        confirmed_phase: str,
        detector_info: DetectorInfo,
        orientation: np.ndarray | None = None,
        confidence_score: float = 1.0,
        eds_data: dict[str, float] | None = None,
        data_source: DataSource = DataSource.USER_CONFIRMED,
        source_file: str = "",
        target_size: int = 128,
    ) -> None:
        """Add a single training sample to the store.

        Parameters
        ----------
        pattern : np.ndarray
            2-D raw pattern (any size, uint8/16 or float).
        confirmed_phase : str
            Ground-truth phase name.
        detector_info : DetectorInfo
            Detector metadata.
        orientation : np.ndarray, optional
            Unit quaternion (4,). Zeros if unknown.
        confidence_score : float
            Confidence index (0-1).
        eds_data : dict[str, float], optional
            Element -> At.% mapping. None if no EDS.
        data_source : DataSource
            How this sample was obtained.
        source_file : str
            Original file name for provenance.
        target_size : int
            Size for the normalized pattern.
        """
        if not confirmed_phase or not confirmed_phase.strip():
            raise ValueError(
                "confirmed_phase must be a non-empty string"
            )
        if orientation is None:
            orientation = np.zeros(4, dtype=np.float32)

        eds_vec = eds_dict_to_vector(eds_data)
        resized = resize_pattern(pattern, target_size)
        normalized, _ = normalize_pattern(resized)

        shard = self._current_shard()
        with self._lock, h5py.File(shard, "a") as f:
            idx = len(f.keys())
            grp = f.create_group(_sample_id(idx))

            grp.create_dataset(
                _K_PATTERN_ORIG, data=pattern, compression="gzip",
                compression_opts=4, chunks=True,
            )
            grp.create_dataset(
                _K_PATTERN_NORM, data=normalized, compression="gzip",
                compression_opts=4, chunks=True,
            )
            grp.create_dataset(_K_EDS, data=eds_vec)
            grp.create_dataset(_K_HAS_EDS, data=eds_data is not None)
            grp.create_dataset(
                _K_ORIENTATION,
                data=orientation.astype(np.float32),
            )
            grp.create_dataset(_K_CONFIDENCE, data=np.float32(confidence_score))
            grp.attrs[_K_PHASE] = confirmed_phase
            grp.attrs[_K_SOURCE] = data_source.value
            grp.attrs[_K_TIMESTAMP] = time.time()
            grp.attrs[_K_SOURCE_FILE] = source_file

            # Store detector info as individual attributes
            det_dict = detector_info.to_dict()
            for k, v in det_dict.items():
                if isinstance(v, list):
                    grp.attrs[f"det_{k}"] = v
                else:
                    grp.attrs[f"det_{k}"] = v

    def add_batch(
        self,
        patterns: np.ndarray,
        confirmed_phases: list[str],
        detector_info: DetectorInfo,
        orientations: np.ndarray | None = None,
        confidence_scores: np.ndarray | None = None,
        eds_data: dict[str, np.ndarray] | None = None,
        data_source: DataSource = DataSource.SIMULATION,
        source_file: str = "",
        target_size: int = 128,
    ) -> int:
        """Add a batch of samples.

        Parameters
        ----------
        patterns : np.ndarray
            (N, H, W) array of raw patterns.
        confirmed_phases : list[str]
            Phase name per sample.
        detector_info : DetectorInfo
            Shared detector info for the batch.
        orientations : np.ndarray, optional
            (N, 4) quaternions. Zeros if None.
        confidence_scores : np.ndarray, optional
            (N,) CI values. All 1.0 if None.
        eds_data : dict[str, np.ndarray], optional
            Element -> (N,) arrays. None if no EDS.
        data_source : DataSource
            How these samples were obtained.
        source_file : str
            Provenance.
        target_size : int
            Normalized pattern size.

        Returns
        -------
        int
            Number of samples added.
        """
        if patterns.ndim != 3:
            raise ValueError(
                f"patterns must be 3-D (N, H, W), got {patterns.ndim}-D "
                f"with shape {patterns.shape}"
            )
        n = patterns.shape[0]
        if len(confirmed_phases) != n:
            raise ValueError(
                f"confirmed_phases length ({len(confirmed_phases)}) must "
                f"match patterns count ({n})"
            )
        if orientations is not None and orientations.shape[0] != n:
            raise ValueError(
                f"orientations count ({orientations.shape[0]}) must "
                f"match patterns count ({n})"
            )
        if confidence_scores is not None and confidence_scores.shape[0] != n:
            raise ValueError(
                f"confidence_scores count ({confidence_scores.shape[0]}) "
                f"must match patterns count ({n})"
            )
        if orientations is None:
            orientations = np.zeros((n, 4), dtype=np.float32)
        if confidence_scores is None:
            confidence_scores = np.ones(n, dtype=np.float32)

        count = 0
        for i in range(n):
            sample_eds: dict[str, float] | None = None
            if eds_data is not None:
                sample_eds = {
                    el: float(arr[i]) for el, arr in eds_data.items()
                }

            phase = (
                confirmed_phases[i]
                if i < len(confirmed_phases)
                else confirmed_phases[-1]
            )

            self.add_sample(
                pattern=patterns[i],
                confirmed_phase=phase,
                detector_info=detector_info,
                orientation=orientations[i],
                confidence_score=float(confidence_scores[i]),
                eds_data=sample_eds,
                data_source=data_source,
                source_file=source_file,
                target_size=target_size,
            )
            count += 1
        return count

    def add_from_indexing(
        self,
        patterns: np.ndarray,
        phase_ids: np.ndarray,
        phase_names: list[str],
        orientations: np.ndarray,
        confidence_scores: np.ndarray,
        detector_info: DetectorInfo,
        ci_threshold: float = 0.3,
        eds_data: dict[str, np.ndarray] | None = None,
        source_file: str = "",
        target_size: int = 128,
    ) -> int:
        """Add samples from indexing results, filtering by CI threshold.

        Only samples with ``confidence_scores[i] >= ci_threshold`` are stored.

        Parameters
        ----------
        patterns : np.ndarray
            (N, H, W) patterns.
        phase_ids : np.ndarray
            (N,) integer phase IDs indexing into ``phase_names``.
        phase_names : list[str]
            Lookup table for phase IDs.
        orientations : np.ndarray
            (N, 4) quaternions.
        confidence_scores : np.ndarray
            (N,) CI values.
        detector_info : DetectorInfo
            Detector metadata.
        ci_threshold : float
            Minimum CI for auto-labeling.
        eds_data : dict[str, np.ndarray], optional
            Element -> (N,) arrays. None if no EDS.
        source_file : str
            Provenance.
        target_size : int
            Normalized pattern size.

        Returns
        -------
        int
            Number of samples that passed the CI threshold and were stored.
        """
        if not 0.0 <= ci_threshold <= 1.0:
            raise ValueError(
                f"ci_threshold must be in [0, 1], got {ci_threshold}"
            )
        if phase_ids.max() >= len(phase_names):
            raise ValueError(
                f"phase_ids contains value {phase_ids.max()} but "
                f"phase_names has only {len(phase_names)} entries"
            )
        mask = confidence_scores >= ci_threshold
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return 0

        filtered_phases = [phase_names[phase_ids[i]] for i in indices]

        filtered_eds: dict[str, np.ndarray] | None = None
        if eds_data is not None:
            filtered_eds = {
                el: arr[indices] for el, arr in eds_data.items()
            }

        return self.add_batch(
            patterns=patterns[indices],
            confirmed_phases=filtered_phases,
            detector_info=detector_info,
            orientations=orientations[indices],
            confidence_scores=confidence_scores[indices],
            eds_data=filtered_eds,
            data_source=DataSource.AUTO_INDEXED,
            source_file=source_file,
            target_size=target_size,
        )

    def add_denoising_pair(
        self,
        experimental_pattern: np.ndarray,
        simulated_pattern: np.ndarray,
        confirmed_phase: str,
        detector_info: DetectorInfo,
        orientation: np.ndarray | None = None,
        confidence_score: float = 1.0,
        eds_data: dict[str, float] | None = None,
        data_source: DataSource = DataSource.AUTO_INDEXED,
        source_file: str = "",
        target_size: int = 128,
    ) -> None:
        """Add a denoising training pair (experimental → simulated).

        The experimental pattern is stored as the input and the simulated
        (clean) pattern as the target.  Both are resized and normalized.

        Parameters
        ----------
        experimental_pattern : np.ndarray
            Noisy experimental pattern (2-D).
        simulated_pattern : np.ndarray
            Clean simulated reference pattern (2-D).
        confirmed_phase : str
            Phase label for this pair.
        detector_info : DetectorInfo
            Detector metadata.
        orientation, confidence_score, eds_data, data_source, source_file,
        target_size :
            Same semantics as :meth:`add_sample`.
        """
        if not confirmed_phase or not confirmed_phase.strip():
            raise ValueError(
                "confirmed_phase must be a non-empty string"
            )
        if orientation is None:
            orientation = np.zeros(4, dtype=np.float32)

        eds_vec = eds_dict_to_vector(eds_data)
        resized_exp = resize_pattern(experimental_pattern, target_size)
        normalized_exp, _ = normalize_pattern(resized_exp)
        resized_sim = resize_pattern(simulated_pattern, target_size)
        normalized_sim, _ = normalize_pattern(resized_sim)

        shard = self._current_shard()
        with self._lock, h5py.File(shard, "a") as f:
            idx = len(f.keys())
            grp = f.create_group(_sample_id(idx))

            grp.create_dataset(
                _K_PATTERN_ORIG, data=experimental_pattern,
                compression="gzip", compression_opts=4, chunks=True,
            )
            grp.create_dataset(
                _K_PATTERN_NORM, data=normalized_exp,
                compression="gzip", compression_opts=4, chunks=True,
            )
            grp.create_dataset(
                _K_PATTERN_TARGET, data=normalized_sim,
                compression="gzip", compression_opts=4, chunks=True,
            )
            grp.create_dataset(_K_EDS, data=eds_vec)
            grp.create_dataset(_K_HAS_EDS, data=eds_data is not None)
            grp.create_dataset(
                _K_ORIENTATION,
                data=orientation.astype(np.float32),
            )
            grp.create_dataset(
                _K_CONFIDENCE, data=np.float32(confidence_score)
            )
            grp.attrs[_K_PHASE] = confirmed_phase
            grp.attrs[_K_SOURCE] = data_source.value
            grp.attrs[_K_TIMESTAMP] = time.time()
            grp.attrs[_K_SOURCE_FILE] = source_file
            grp.attrs[_K_PAIR_TYPE] = "denoising"

            det_dict = detector_info.to_dict()
            for k, v in det_dict.items():
                grp.attrs[f"det_{k}"] = v

    def count_denoising_pairs(self) -> int:
        """Return the number of denoising pairs in the store."""
        count = 0
        for shard in self._shard_files():
            with self._lock, h5py.File(shard, "r") as f:
                for key in f.keys():
                    if f[key].attrs.get(_K_PAIR_TYPE, "") == "denoising":
                        count += 1
        return count

    # -- Public API: read data -----------------------------------------------

    def __len__(self) -> int:
        """Total number of samples across all shards."""
        total = 0
        for shard in self._shard_files():
            with self._lock, h5py.File(shard, "r") as f:
                total += len(f.keys())
        return total

    def _read_sample_from_group(
        self, grp: h5py.Group
    ) -> dict[str, Any]:
        """Extract a sample dict from an HDF5 group."""
        det_dict = {
            "manufacturer": str(grp.attrs["det_manufacturer"]),
            "pc": list(grp.attrs["det_pc"]),
            "pc_convention": str(grp.attrs["det_pc_convention"]),
            "kv": float(grp.attrs["det_kv"]),
            "working_distance": float(grp.attrs["det_working_distance"]),
            "sample_tilt": float(grp.attrs["det_sample_tilt"]),
        }
        # Schema v2 fields (backward-compatible)
        pattern_target = None
        if _K_PATTERN_TARGET in grp:
            pattern_target = grp[_K_PATTERN_TARGET][()]
        pair_type = str(grp.attrs.get(_K_PAIR_TYPE, "classification"))

        return {
            "pattern_normalized": grp[_K_PATTERN_NORM][()],
            "eds_atomic_pct": grp[_K_EDS][()],
            "has_eds": bool(grp[_K_HAS_EDS][()]),
            "confirmed_phase": str(grp.attrs[_K_PHASE]),
            "orientation_quaternion": grp[_K_ORIENTATION][()],
            "confidence_score": float(grp[_K_CONFIDENCE][()]),
            "data_source": str(grp.attrs[_K_SOURCE]),
            "detector_info": DetectorInfo.from_dict(det_dict),
            "pattern_target": pattern_target,
            "pair_type": pair_type,
        }

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        """Iterate over all samples, yielding dicts.

        Yields
        ------
        dict
            Keys: ``pattern_normalized``, ``eds_atomic_pct``, ``has_eds``,
            ``confirmed_phase``, ``orientation_quaternion``,
            ``confidence_score``, ``data_source``, ``detector_info``.
        """
        for shard in self._shard_files():
            with self._lock, h5py.File(shard, "r") as f:
                for key in sorted(f.keys()):
                    yield self._read_sample_from_group(f[key])

    def get_sample(self, global_index: int) -> dict[str, Any]:
        """Read a single sample by global index.

        Parameters
        ----------
        global_index : int
            Zero-based index across all shards.

        Returns
        -------
        dict
            Same keys as ``iter_samples`` yields.

        Raises
        ------
        IndexError
            If *global_index* is out of range.
        """
        offset = 0
        for shard in self._shard_files():
            with self._lock, h5py.File(shard, "r") as f:
                n = len(f.keys())
                if global_index < offset + n:
                    local_idx = global_index - offset
                    key = _sample_id(local_idx)
                    return self._read_sample_from_group(f[key])
                offset += n
        raise IndexError(
            f"Sample index {global_index} out of range "
            f"(store has {offset} samples)"
        )

    def get_metadata(self, global_index: int) -> dict[str, Any]:
        """Read lightweight metadata for a sample (no pattern arrays).

        This is much faster than :meth:`get_sample` because it only reads
        HDF5 attributes and small scalar datasets, skipping the pattern
        and EDS arrays entirely.

        Parameters
        ----------
        global_index : int
            Zero-based index across all shards.

        Returns
        -------
        dict
            Keys: ``confirmed_phase``, ``has_eds``, ``confidence_score``,
            ``data_source``.

        Raises
        ------
        IndexError
            If *global_index* is out of range.
        """
        offset = 0
        for shard in self._shard_files():
            with self._lock, h5py.File(shard, "r") as f:
                n = len(f.keys())
                if global_index < offset + n:
                    local_idx = global_index - offset
                    grp = f[_sample_id(local_idx)]
                    return {
                        "confirmed_phase": str(grp.attrs[_K_PHASE]),
                        "has_eds": bool(grp[_K_HAS_EDS][()]),
                        "confidence_score": float(grp[_K_CONFIDENCE][()]),
                        "data_source": str(grp.attrs[_K_SOURCE]),
                    }
                offset += n
        raise IndexError(
            f"Sample index {global_index} out of range "
            f"(store has {offset} samples)"
        )

    def iter_metadata(self) -> Iterator[dict[str, Any]]:
        """Iterate over lightweight metadata for all samples.

        Yields only attributes and small scalar datasets, avoiding
        expensive reads of pattern and EDS arrays.

        Yields
        ------
        dict
            Keys: ``confirmed_phase``, ``has_eds``, ``confidence_score``,
            ``data_source``.
        """
        for shard in self._shard_files():
            with self._lock, h5py.File(shard, "r") as f:
                for key in sorted(f.keys()):
                    grp = f[key]
                    yield {
                        "confirmed_phase": str(grp.attrs[_K_PHASE]),
                        "has_eds": bool(grp[_K_HAS_EDS][()]),
                        "confidence_score": float(grp[_K_CONFIDENCE][()]),
                        "data_source": str(grp.attrs[_K_SOURCE]),
                    }

    def iter_metadata_v2(self) -> Iterator[dict[str, Any]]:
        """Like iter_metadata but includes schema v2 fields (pair_type)."""
        for shard in self._shard_files():
            with self._lock, h5py.File(shard, "r") as f:
                for key in sorted(f.keys()):
                    grp = f[key]
                    yield {
                        "confirmed_phase": str(grp.attrs[_K_PHASE]),
                        "has_eds": bool(grp[_K_HAS_EDS][()]),
                        "confidence_score": float(grp[_K_CONFIDENCE][()]),
                        "data_source": str(grp.attrs[_K_SOURCE]),
                        "pair_type": str(
                            grp.attrs.get(_K_PAIR_TYPE, "classification")
                        ),
                    }

    # -- Public API: statistics -----------------------------------------------

    def get_dataset_stats(self) -> dict[str, Any]:
        """Compute summary statistics over the store.

        Returns
        -------
        dict
            Keys: ``total_samples``, ``samples_per_phase``,
            ``samples_with_eds``, ``samples_without_eds``,
            ``samples_per_source``, ``samples_per_convention``,
            ``mean_confidence``.
        """
        phase_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        convention_counts: dict[str, int] = {}
        with_eds = 0
        without_eds = 0
        total = 0
        ci_sum = 0.0

        for sample in self.iter_samples():
            total += 1
            phase = sample["confirmed_phase"]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

            src = sample["data_source"]
            source_counts[src] = source_counts.get(src, 0) + 1

            conv = sample["detector_info"].pc_convention.value
            convention_counts[conv] = convention_counts.get(conv, 0) + 1

            if sample["has_eds"]:
                with_eds += 1
            else:
                without_eds += 1

            ci_sum += sample["confidence_score"]

        return {
            "total_samples": total,
            "samples_per_phase": phase_counts,
            "samples_with_eds": with_eds,
            "samples_without_eds": without_eds,
            "samples_per_source": source_counts,
            "samples_per_convention": convention_counts,
            "mean_confidence": ci_sum / total if total > 0 else 0.0,
        }

    def get_phase_names(self) -> list[str]:
        """Return sorted list of unique phase names in the store."""
        phases: set[str] = set()
        for shard in self._shard_files():
            with self._lock, h5py.File(shard, "r") as f:
                for key in f.keys():
                    phases.add(str(f[key].attrs[_K_PHASE]))
        return sorted(phases)
