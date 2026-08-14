"""Metadata describing a GPU-generated dictionary.

Field shape mirrors simulation.dictionary_generator.DictionaryMetadata so
downstream consumers (Database Browser, indexing) treat both the same.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Tuple


@dataclass
class GPUDictionaryMetadata:
    master_path: str
    material: str
    phase_name: str
    space_group_number: int = 0
    energy_kv: float = 20.0
    detector_shape: Tuple[int, int] = (60, 60)
    pc: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    sample_tilt: float = 70.0
    # Detector (camera) tilt and azimuthal, in degrees. Recorded because a
    # dictionary is only valid for the geometry it was simulated at — on a
    # 3.44 deg detector a dictionary built at 0 deg correlates at NCC ~0.02
    # against the correct pattern for the SAME orientation.
    detector_tilt: float = 0.0
    azimuthal: float = 0.0
    resolution_deg: float = 5.0
    n_orientations: int = 0
    created_at: str = ""
    dictionary_path: str = ""
    backend: str = "gpu"

    def to_json(self) -> str:
        d = asdict(self)
        d["detector_shape"] = list(d["detector_shape"])
        d["pc"] = list(d["pc"])
        return json.dumps(d, indent=2)
