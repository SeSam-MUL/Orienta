"""High-level generate_dictionary_gpu() — orchestrates loader + orientations + projector + writer."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from .detector import detector_pixel_directions
from .master_loader import load_master_pattern
from .metadata import GPUDictionaryMetadata
from .normalize import normalize_patterns
from .projector import project_patterns
from .runtime import available_vram_bytes, estimate_batch_size, has_cuda
from .storage import ChunkedDictionaryWriter

logger = logging.getLogger(__name__)

# Project root used to anchor the default `tasks/` output directory regardless
# of process CWD. Project convention: no hardcoded/cwd-relative paths.
#   backend/dictionary_gpu/pipeline.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = PROJECT_ROOT / "tasks"


@dataclass
class GenerationResult:
    output_path: Path
    metadata: GPUDictionaryMetadata


def generate_dictionary_gpu(
    master_path: str,
    detector_shape: Tuple[int, int],
    pc: Tuple[float, float, float],
    sample_tilt: float = 70.0,
    energy_kv: Optional[float] = None,
    resolution_deg: float = 5.0,
    output_path: Optional[str] = None,
    normalize: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> GenerationResult:
    """Generate a dictionary on GPU and save it to disk."""
    from orix.sampling import get_sample_fundamental
    from orix.crystal_map import Phase

    _emit = progress_callback or (lambda f, m: None)

    device = "cuda" if has_cuda() else "cpu"
    logger.info("dictionary-gpu: device=%s", device)
    _emit(0.0, f"Loading master pattern on {device}")

    loaded = load_master_pattern(master_path, energy_kv=energy_kv, device=device)

    # Phase / point-group lookup uses kikuchipy CPU briefly (no projection).
    import kikuchipy as kp
    mp = kp.load(master_path, projection="lambert", hemisphere="both")
    phase: Phase = mp.phase
    point_group = phase.point_group
    phase_name = str(phase.name) if phase.name else Path(master_path).stem

    _emit(0.05, f"Sampling orientations @ {resolution_deg} deg")
    rotations = get_sample_fundamental(resolution=resolution_deg, point_group=point_group)
    n_total = rotations.size
    logger.info("dictionary-gpu: %d orientations to project", n_total)

    H, W = detector_shape
    pixel_dirs = detector_pixel_directions(
        detector_shape,
        pc,
        tilt_deg=sample_tilt,
        device=device,
    )

    out_path = (
        Path(output_path)
        if output_path
        else TASKS_DIR / f"dict_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = ChunkedDictionaryWriter(str(out_path), n_total=n_total, detector_shape=detector_shape)

    quat_np = rotations.data.astype(np.float32)            # (N, 4)
    avail = available_vram_bytes() if device == "cuda" else 4_000_000_000
    batch_size = estimate_batch_size(detector_shape, avail)
    logger.info("dictionary-gpu: batch_size=%d (avail=%d MB)", batch_size, avail // (1 << 20))

    cursor = 0
    while cursor < n_total:
        end = min(cursor + batch_size, n_total)
        quats = torch.from_numpy(quat_np[cursor:end]).to(device)
        patterns = project_patterns(quats, pixel_dirs, loaded.hemispheres, detector_shape)
        if normalize:
            patterns = normalize_patterns(patterns)
        writer.append(patterns)
        cursor = end
        _emit(cursor / n_total, f"{cursor}/{n_total} patterns")

    writer.finalize()
    _emit(1.0, "Saved")

    meta = GPUDictionaryMetadata(
        master_path=master_path,
        material=phase_name.split("_")[0] if phase_name else "unknown",
        phase_name=phase_name,
        space_group_number=loaded.space_group,
        energy_kv=loaded.energy_kv,
        detector_shape=detector_shape,
        pc=pc,
        sample_tilt=sample_tilt,
        resolution_deg=resolution_deg,
        n_orientations=n_total,
        created_at=datetime.now().isoformat(),
        dictionary_path=str(out_path),
    )
    json_path = out_path.with_suffix(".json")
    json_path.write_text(meta.to_json(), encoding="utf-8")

    return GenerationResult(output_path=out_path, metadata=meta)
