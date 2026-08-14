"""High-level generate_dictionary_gpu() — orchestrates loader + orientations + projector + writer."""
from __future__ import annotations

import logging
import time
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

# Orientations per get_patterns() call on the CPU path. Small enough that the
# rate counter updates a few times a second on a coarse grid, large enough
# that per-call overhead stays negligible.
CPU_CHUNK_SIZE = 256


@dataclass
class GenerationResult:
    output_path: Path
    metadata: GPUDictionaryMetadata


def _rate_message(done: int, total: int, started: float) -> str:
    """"1024/6579 patterns · 210 pat/s" — the live throughput counter.

    Averaged over the whole run rather than the last chunk: the first chunk
    carries the one-off kernel/JIT warm-up and an instantaneous figure would
    swing wildly because of it.
    """
    elapsed = max(time.perf_counter() - started, 1e-9)
    return f"{done}/{total} patterns · {done / elapsed:,.0f} pat/s"


def _throughput(n: int, started: float):
    """(elapsed seconds, patterns per second) for a finished run."""
    elapsed = max(time.perf_counter() - started, 1e-9)
    return elapsed, n / elapsed


def _summary_message(n: int, elapsed: float, rate: float, device: str) -> str:
    return f"Saved {n:,} patterns in {elapsed:,.1f} s · {rate:,.0f} pat/s ({device})"


def generate_dictionary_gpu(
    master_path: str,
    detector_shape: Tuple[int, int],
    pc: Tuple[float, float, float],
    sample_tilt: float = 70.0,
    energy_kv: Optional[float] = None,
    resolution_deg: float = 5.0,
    output_path: Optional[str] = None,
    normalize: bool = False,
    detector_tilt_deg: float = 0.0,
    azimuthal_deg: float = 0.0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> GenerationResult:
    """Generate a dictionary on GPU and save it to disk.

    ``detector_tilt_deg`` / ``azimuthal_deg`` are the camera geometry from the
    loaded dataset's ``EBSDDetector``. They used to be left at 0 while the real
    detector was tilted, which silently produced a dictionary for a detector
    that does not exist — measured on the user's 3.44 deg setup, the correct and
    the generated pattern for the SAME orientation correlate at NCC 0.017.
    """
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
        detector_tilt_deg=detector_tilt_deg,
        azimuthal_deg=azimuthal_deg,
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
    started = time.perf_counter()
    while cursor < n_total:
        end = min(cursor + batch_size, n_total)
        quats = torch.from_numpy(quat_np[cursor:end]).to(device)
        patterns = project_patterns(quats, pixel_dirs, loaded.hemispheres, detector_shape)
        if normalize:
            patterns = normalize_patterns(patterns)
        writer.append(patterns)
        cursor = end
        _emit(cursor / n_total, _rate_message(cursor, n_total, started))

    # The rotations MUST travel with the patterns: indexing maps a best-match
    # index back to an orientation through them.
    writer.finalize(rotations=rotations, phase=phase)
    elapsed, rate = _throughput(n_total, started)
    logger.info("dictionary-gpu: %d patterns in %.1f s (%.0f pat/s, %s)",
                n_total, elapsed, rate, device)

    meta = GPUDictionaryMetadata(
        master_path=master_path,
        material=phase_name.split("_")[0] if phase_name else "unknown",
        phase_name=phase_name,
        space_group_number=loaded.space_group,
        energy_kv=loaded.energy_kv,
        detector_shape=detector_shape,
        pc=pc,
        sample_tilt=sample_tilt,
        detector_tilt=detector_tilt_deg,
        azimuthal=azimuthal_deg,
        resolution_deg=resolution_deg,
        n_orientations=n_total,
        created_at=datetime.now().isoformat(),
        dictionary_path=str(out_path),
        elapsed_s=elapsed,
        patterns_per_second=rate,
        device=device,
    )
    json_path = out_path.with_suffix(".json")
    json_path.write_text(meta.to_json(), encoding="utf-8")
    _emit(1.0, _summary_message(n_total, elapsed, rate, device))

    return GenerationResult(output_path=out_path, metadata=meta)


def generate_dictionary_cpu(
    master_path: str,
    detector_shape: Tuple[int, int],
    pc: Tuple[float, float, float],
    sample_tilt: float = 70.0,
    energy_kv: Optional[float] = None,
    resolution_deg: float = 5.0,
    output_path: Optional[str] = None,
    normalize: bool = False,
    detector_tilt_deg: float = 0.0,
    azimuthal_deg: float = 0.0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> GenerationResult:
    """Same contract as :func:`generate_dictionary_gpu`, on the CPU.

    Delegates the projection to kikuchipy's ``EBSDMasterPattern.get_patterns``
    (via ``simulation.dictionary_generator.generate_dictionary``) — the same
    routine that produced every dictionary already in the library, so a CPU and
    a GPU dictionary are interchangeable downstream.

    Runs ``get_patterns`` in chunks so it reports the same live
    patterns/second counter as the GPU path; one unchunked call is a single
    blocking operation with nothing to count. It still materialises the whole
    dictionary in RAM (n x H x W x 4 bytes).
    """
    from kikuchipy.detectors import EBSDDetector
    from simulation.dictionary_generator import generate_dictionary

    _emit = progress_callback or (lambda f, m: None)
    _emit(0.0, "Loading master pattern on cpu")

    detector = EBSDDetector(
        shape=tuple(int(v) for v in detector_shape),
        pc=tuple(float(v) for v in pc),
        sample_tilt=float(sample_tilt),
        tilt=float(detector_tilt_deg),
        azimuthal=float(azimuthal_deg),
        convention="bruker",
    )

    stage = {"frac": 0.05}

    def _relay(msg: str) -> None:
        # Stage messages from the loader / sampler, before any pattern exists.
        if msg.startswith("Sampling"):
            stage["frac"] = 0.10
        elif msg.startswith("Sampled"):
            stage["frac"] = 0.15
        elif msg.startswith("Simulating"):
            stage["frac"] = 0.20
        _emit(stage["frac"], msg)

    # Projection runs from 0.20 to 0.90 of the bar; loading and sampling own
    # the head, saving the tail.
    started = time.perf_counter()
    projection_started = {"t": None}

    def _on_chunk(done: int, total: int) -> None:
        if projection_started["t"] is None:
            projection_started["t"] = started
        frac = 0.20 + 0.70 * (done / max(total, 1))
        _emit(frac, _rate_message(done, total, projection_started["t"]))

    dictionary, meta_cpu = generate_dictionary(
        master_path=master_path,
        detector=detector,
        energy=float(energy_kv) if energy_kv else 0.0,
        resolution=resolution_deg,
        progress_callback=_relay,
        chunk_size=CPU_CHUNK_SIZE,
        chunk_progress=_on_chunk,
    )

    if normalize:
        import numpy as _np
        data = _np.asarray(dictionary.data, dtype=_np.float32)
        flat = data.reshape(data.shape[0], -1)
        mean = flat.mean(axis=1, keepdims=True)
        std = flat.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        dictionary.data = ((flat - mean) / std).reshape(data.shape)

    out_path = (
        Path(output_path)
        if output_path
        else TASKS_DIR / f"dict_cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    elapsed, rate = _throughput(int(meta_cpu.n_orientations), started)
    logger.info("dictionary-cpu: %d patterns in %.1f s (%.0f pat/s)",
                meta_cpu.n_orientations, elapsed, rate)

    _emit(0.95, f"Saving {out_path.name}")
    dictionary.save(str(out_path), overwrite=True)

    # Same sidecar shape as the GPU path so discovery reads both identically.
    meta = GPUDictionaryMetadata(
        master_path=master_path,
        material=meta_cpu.material,
        phase_name=meta_cpu.phase_name,
        space_group_number=meta_cpu.space_group_number,
        energy_kv=meta_cpu.energy_kv,
        detector_shape=tuple(detector_shape),
        pc=tuple(pc),
        sample_tilt=sample_tilt,
        detector_tilt=detector_tilt_deg,
        azimuthal=azimuthal_deg,
        resolution_deg=resolution_deg,
        n_orientations=meta_cpu.n_orientations,
        created_at=datetime.now().isoformat(),
        dictionary_path=str(out_path),
        backend="cpu",
        elapsed_s=elapsed,
        patterns_per_second=rate,
        device="cpu",
    )
    out_path.with_suffix(".json").write_text(meta.to_json(), encoding="utf-8")
    _emit(1.0, _summary_message(int(meta_cpu.n_orientations), elapsed, rate, "cpu"))

    return GenerationResult(output_path=out_path, metadata=meta)
