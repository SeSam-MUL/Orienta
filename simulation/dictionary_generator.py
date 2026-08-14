"""
Dictionary Generator - Creates simulated pattern dictionaries from master patterns.

Workflow:
    Master Pattern H5 (square Lambert projection — required by get_patterns())
        -> sample crystal orientations (orix)
        -> project onto detector (kikuchipy get_patterns)
        -> Dictionary of simulated patterns matching experimental geometry

The generated dictionary can be cached to disk with a JSON metadata sidecar
so it can be re-used for future indexing runs with the same detector setup.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class DictionaryMetadata:
    """All parameters that uniquely define a generated dictionary."""

    master_path: str                    # Source master pattern path
    material: str                       # e.g. "Ni"
    phase_name: str                     # e.g. "Nickel"
    space_group_number: int = 0
    point_group: str = ""               # e.g. "m-3m"
    energy_kv: float = 20.0
    detector_shape: Tuple[int, int] = (60, 60)
    pc: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    sample_tilt: float = 70.0
    resolution_deg: float = 5.0
    n_orientations: int = 0
    convention: str = "oxford"
    created_at: str = ""
    dictionary_path: str = ""

    def to_json(self) -> str:
        import numpy as np

        d = asdict(self)
        # Convert tuples to lists for JSON
        d["detector_shape"] = list(d["detector_shape"])
        d["pc"] = list(d["pc"])

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        return json.dumps(d, indent=2, cls=_NumpyEncoder)

    @classmethod
    def from_json(cls, text: str) -> "DictionaryMetadata":
        d = json.loads(text)
        d["detector_shape"] = tuple(d["detector_shape"])
        d["pc"] = tuple(d["pc"])
        return cls(**d)


def build_dictionary_filename(
    material: str,
    energy_kv: float,
    detector_shape: Tuple[int, int],
    resolution_deg: float,
    pc: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> str:
    """Generate a deterministic filename for a dictionary.

    Includes PC so that dictionaries for different projection centres
    are stored side-by-side instead of overwriting each other.

    Example: ``Ni_dict_20kV_60x60_pc507_262_558_5.0deg``
    """
    h, w = detector_shape
    mat = material.replace(" ", "_") or "unknown"
    pc_tag = _pc_tag(pc)
    return f"{mat}_dict_{int(energy_kv)}kV_{h}x{w}_{pc_tag}_{resolution_deg:.1f}deg"


def _pc_tag(pc: Tuple[float, float, float]) -> str:
    """Encode a PC as 3-digit integers (value × 1000) for compact filenames."""
    return f"pc{int(pc[0]*1000)}_{int(pc[1]*1000)}_{int(pc[2]*1000)}"


def dictionary_library_paths(
    library_dir,
    master_path: str,
    energy_kv: float,
    detector_shape: Tuple[int, int],
    pc: Tuple[float, float, float],
    resolution_deg: float,
) -> Tuple[Path, Path]:
    """Canonical (h5, json) location for a generated dictionary.

    This is the ONE place that defines where a dictionary lives, shared by the
    CPU writer (:func:`save_dictionary`) and the GPU route
    (``POST /api/dictionary-gpu/generate`` with ``save_to_library``). Both must
    agree, because the Indexing page finds dictionaries by scanning
    ``Dictionary_Library`` and attributes each file to a phase card by matching
    its FILENAME against the CIF library — so both the folder and the name are
    load-bearing, not cosmetic:

      ``<library_dir>/<Mat>/<master_stem>_dict_<E>kV_<HxW>_pc<x_y_z>_<res>deg.h5``

    e.g. ``Dictionary_Library/Si/Si_master_E20kV_npx500_dict_20kV_128x156_pc547_465_609_2.0deg.h5``

    The leading token of the stem ("Si") is what links the file to its phase;
    the ``_dict_`` separator is what marks it as a dictionary rather than a
    master; the PC tag keeps dictionaries for different projection centres
    side-by-side instead of overwriting each other.

    Parameters
    ----------
    library_dir : str or Path
        Root of the dictionary library (``Database/Dictionary_Library``).
    master_path : str
        Source master pattern. Its stem drives both the subfolder and the name.

    Returns
    -------
    tuple of Path
        ``(h5_path, json_path)``. Neither is created; the caller writes them.
    """
    master_stem = Path(master_path).stem if master_path else "unknown"
    short_mat = master_stem.split("_")[0]  # "Al", "Ni", "Si", …
    h, w = detector_shape
    base_name = (
        f"{master_stem}_dict_{int(energy_kv)}kV"
        f"_{h}x{w}_{_pc_tag(pc)}_{resolution_deg:.1f}deg"
    )
    mat_dir = Path(library_dir) / short_mat
    return mat_dir / f"{base_name}.h5", mat_dir / f"{base_name}.json"


# ---------------------------------------------------------------------------
# Orientation estimation
# ---------------------------------------------------------------------------

# Approximate number of orientations for common Laue groups at various resolutions.
# These are rough estimates; the actual count comes from orix at runtime.
_SYMMETRY_ORDER = {
    "m-3m": 48, "m-3": 24, "-3m": 12, "6/mmm": 24, "6/m": 12,
    "4/mmm": 16, "4/m": 8, "mmm": 8, "2/m": 4, "-1": 2, "1": 1,
}


def estimate_orientations(point_group: str, resolution_deg: float) -> int:
    """Rough estimate of the number of orientations for a symmetry + resolution."""
    order = _SYMMETRY_ORDER.get(point_group, 1)
    # Total orientations on SO(3) ~ (360/res)^3 / (pi^2 * 2)
    # Divided by symmetry order
    n_total = (360 / resolution_deg) ** 3 / (np.pi ** 2 * 2)
    return max(1, int(n_total / order))


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_dictionary(
    master_path: str,
    detector,
    energy: float = 0.0,
    resolution: float = 5.0,
    progress_callback: Optional[Callable[[str], None]] = None,
    chunk_size: Optional[int] = None,
    chunk_progress: Optional[Callable[[int, int], None]] = None,
):
    """Generate a dictionary of simulated patterns from a master pattern.

    Parameters
    ----------
    master_path : str
        Path to EMsoft master pattern H5 file.
    detector : kikuchipy.detectors.EBSDDetector
        Detector geometry matching experimental data.
    energy : float
        Acceleration voltage in kV.  If 0, use highest energy from master.
    resolution : float
        Orientation sampling resolution in degrees.
    progress_callback : callable, optional
        ``fn(message: str)`` for progress updates.

    Returns
    -------
    tuple of (dictionary, metadata)
        dictionary : kikuchipy.signals.EBSD
        metadata : DictionaryMetadata
    """
    import kikuchipy as kp
    from orix.sampling import get_sample_fundamental

    _emit = progress_callback or (lambda m: None)

    # --- Load master pattern ---
    # Lambert projection is required by get_patterns().
    # hemisphere="both" is needed for non-centrosymmetric phases (e.g. Fe compounds).
    # Do NOT pass signal_type — it prevents projection/hemisphere kwargs reaching the reader.
    _emit(f"Loading master pattern: {Path(master_path).name}")
    try:
        mp = kp.load(master_path, projection="lambert", hemisphere="both")
    except Exception as e:
        logger.debug("Loading both hemispheres failed, falling back: %s", e)
        mp = kp.load(master_path, projection="lambert")

    # Guard: give a clear error if projection is still wrong
    if getattr(mp, "projection", "lambert") == "stereographic":
        raise ValueError(
            "Master pattern loaded in stereographic projection, but "
            "get_patterns() requires Lambert projection. "
            "Try updating kikuchipy: pip install -U kikuchipy"
        )
    logger.info(f"Master pattern loaded: {mp.data.shape}, phase={mp.phase}, projection={getattr(mp, 'projection', 'unknown')}")

    # Extract phase info
    phase = mp.phase
    phase_name = str(phase.name) if phase.name else Path(master_path).stem
    point_group_name = str(phase.point_group.name) if phase.point_group else "1"
    sg_number = int(phase.space_group.number) if phase.space_group else 0

    # Determine energy
    if energy <= 0:
        # Use highest available energy from master pattern
        try:
            energies = mp.axes_manager["energy"].axis
            energy = float(energies[-1])
            _emit(f"Using highest energy from master: {energy} kV")
        except Exception as e:
            energy = 20.0
            logger.debug("Could not determine energy from master: %s", e)
            _emit(f"Could not determine energy, defaulting to {energy} kV")

    # --- Sample orientations ---
    _emit(f"Sampling orientations: {point_group_name} @ {resolution}\u00b0")

    rotations = get_sample_fundamental(
        resolution=resolution,
        point_group=phase.point_group,
    )
    n_ori = rotations.size
    _emit(f"Sampled {n_ori} orientations for {point_group_name}")
    logger.info(f"Orientation sampling: {n_ori} rotations for {point_group_name} @ {resolution}deg")

    # --- Simulate dictionary ---
    det_shape = detector.shape
    _emit(f"Simulating {n_ori} patterns ({det_shape[0]}x{det_shape[1]})...")

    if chunk_size and chunk_size > 0 and n_ori > chunk_size:
        # Chunked so the caller can report progress and a patterns/second rate.
        # One get_patterns() call over the whole grid is a single blocking
        # operation with no sub-progress at all, which made the CPU backend
        # look frozen and gave nothing to count.
        import kikuchipy as _kp
        from orix.crystal_map import CrystalMap, PhaseList

        buf = None
        done = 0
        for start in range(0, n_ori, chunk_size):
            part = mp.get_patterns(
                rotations=rotations[start:start + chunk_size],
                detector=detector,
                energy=energy,
                dtype_out=np.float32,
                compute=True,
            )
            arr = np.asarray(part.data, dtype=np.float32)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if buf is None:
                buf = np.empty((n_ori, *arr.shape[1:]), dtype=np.float32)
            buf[start:start + arr.shape[0]] = arr
            done += arr.shape[0]
            if chunk_progress:
                chunk_progress(done, n_ori)
        dictionary = _kp.signals.EBSD(buf)
        # get_patterns attaches this itself; rebuild it so a chunked run is
        # indistinguishable from an unchunked one downstream.
        dictionary.xmap = CrystalMap(
            rotations=rotations, phase_list=PhaseList(phase)
        )
    else:
        dictionary = mp.get_patterns(
            rotations=rotations,
            detector=detector,
            energy=energy,
            dtype_out=np.float32,
            compute=True,
        )
    logger.info(f"Dictionary generated: {dictionary.data.shape}")
    _emit(f"Dictionary ready: {dictionary.data.shape}")

    # --- Build metadata ---
    pc = tuple(float(v) for v in detector.pc.flatten()[:3])
    convention = getattr(detector, "convention", "unknown")
    material = _extract_material(phase_name, master_path)

    metadata = DictionaryMetadata(
        master_path=str(master_path),
        material=material,
        phase_name=phase_name,
        space_group_number=sg_number,
        point_group=point_group_name,
        energy_kv=energy,
        detector_shape=tuple(det_shape),
        pc=pc,
        sample_tilt=float(detector.tilt),
        resolution_deg=resolution,
        n_orientations=n_ori,
        convention=str(convention),
        created_at=datetime.now().isoformat(),
    )

    return dictionary, metadata


def _extract_material(phase_name: str, master_path: str) -> str:
    """Best-effort material name from phase or filename."""
    if phase_name and phase_name.lower() not in ("", "unknown", "phase"):
        # Take first word / element symbol
        parts = phase_name.replace("-", " ").split()
        return parts[0] if parts else Path(master_path).stem.split("_")[0]
    return Path(master_path).stem.split("_")[0]


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_dictionary(dictionary, metadata: DictionaryMetadata, output_dir) -> Path:
    """Save a generated dictionary as H5 + JSON sidecar.

    Parameters
    ----------
    dictionary : kikuchipy.signals.EBSD
        The simulated dictionary signal.
    metadata : DictionaryMetadata
        Parameters describing this dictionary.
    output_dir : str or Path
        Directory to save into (e.g. Database/Dictionary_Library/).

    Returns
    -------
    Path
        Path to the saved H5 file.
    """
    # Folder + name come from the shared convention so this writer and the GPU
    # route land in exactly the same place (see dictionary_library_paths).
    h5_path, json_path = dictionary_library_paths(
        output_dir,
        master_path=metadata.master_path or metadata.material,
        energy_kv=metadata.energy_kv,
        detector_shape=metadata.detector_shape,
        pc=metadata.pc,
        resolution_deg=metadata.resolution_deg,
    )
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dictionary signal
    dictionary.save(str(h5_path), overwrite=True)
    logger.info(f"Dictionary H5 saved: {h5_path}")

    # Save metadata sidecar
    metadata.dictionary_path = str(h5_path)
    json_path.write_text(metadata.to_json(), encoding="utf-8")
    logger.info(f"Dictionary metadata saved: {json_path}")

    return h5_path


def load_dictionary_metadata(json_path) -> DictionaryMetadata:
    """Load dictionary metadata from a JSON sidecar file."""
    text = Path(json_path).read_text(encoding="utf-8")
    return DictionaryMetadata.from_json(text)


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------

def find_compatible_dictionary(
    detector_shape: Tuple[int, int],
    phase_name: str = "",
    energy_kv: float = 0.0,
    db_root=None,
) -> Optional[DictionaryMetadata]:
    """Find a cached dictionary matching the given parameters.

    Parameters
    ----------
    detector_shape : tuple of (int, int)
        Required pattern shape (must match exactly).
    phase_name : str
        Material / phase name hint for prioritization.
    energy_kv : float
        Acceleration voltage hint (tolerance: +/- 2 kV).
    db_root : Path, optional
        Database root.  Defaults to ``get_local_database_path()``.

    Returns
    -------
    DictionaryMetadata or None
        Best matching cached dictionary, or None.
    """
    if db_root is None:
        from path_utils import get_local_database_path, DATABASE_SUBFOLDERS
        db_root = get_local_database_path() / DATABASE_SUBFOLDERS["dictionary_library"]
    else:
        db_root = Path(db_root)

    if not db_root.is_dir():
        return None

    candidates: List[Tuple[int, DictionaryMetadata]] = []

    for json_file in db_root.rglob("*_dict_*.json"):
        try:
            meta = load_dictionary_metadata(json_file)
        except Exception as e:
            logger.debug("Skipping %s: %s", json_file.name, e)
            continue

        # Exact shape match required
        if tuple(meta.detector_shape) != tuple(detector_shape):
            continue

        # Check that the H5 file still exists
        if meta.dictionary_path and not Path(meta.dictionary_path).is_file():
            continue

        # Score: higher is better
        score = 0
        if phase_name and phase_name.lower() in meta.phase_name.lower():
            score += 10
        if phase_name and phase_name.lower() in meta.material.lower():
            score += 5
        if energy_kv > 0 and abs(meta.energy_kv - energy_kv) <= 2:
            score += 3

        candidates.append((score, meta))

    if not candidates:
        return None

    # Return best match
    candidates.sort(key=lambda x: (-x[0], x[1].resolution_deg))
    return candidates[0][1]


def discover_all_dictionaries(db_root=None) -> List[DictionaryMetadata]:
    """List all cached dictionaries with valid JSON sidecars."""
    if db_root is None:
        from path_utils import get_local_database_path, DATABASE_SUBFOLDERS
        db_root = get_local_database_path() / DATABASE_SUBFOLDERS["dictionary_library"]
    else:
        db_root = Path(db_root)

    if not db_root.is_dir():
        return []

    results = []
    for json_file in db_root.rglob("*_dict_*.json"):
        try:
            meta = load_dictionary_metadata(json_file)
            if meta.dictionary_path and Path(meta.dictionary_path).is_file():
                results.append(meta)
        except Exception as e:
            logger.debug("Skipping %s: %s", json_file.name, e)
            continue

    return results
