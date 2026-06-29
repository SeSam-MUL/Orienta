"""Convert IndexResult into orix CrystalMap and write .ang/.ctf files."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .indexer import IndexResult


def to_crystal_map(
    result: IndexResult,
    n_rows: int,
    n_cols: int,
    step_x: float,
    step_y: float,
    point_group: str,
    phase_name: str = "Aluminum",
):
    """Build an orix CrystalMap from an IndexResult.

    Layout matches the Classic-WSL output structure: rotations stored as
    Bunge ZYZ Euler angles, score in ``prop['ci']``, 1-indexed phase ids.
    """
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    if result.euler_xyz.shape[0] != n_rows * n_cols:
        raise ValueError(
            f"euler_xyz has {result.euler_xyz.shape[0]} rows but n_rows*n_cols "
            f"= {n_rows*n_cols}"
        )

    phase = Phase(name=phase_name, point_group=point_group)
    phases = PhaseList(phases=[phase], ids=[1])

    eulers = result.euler_xyz.cpu().numpy()
    rotations = Rotation.from_euler(eulers)

    rows = np.repeat(np.arange(n_rows), n_cols)
    cols = np.tile(np.arange(n_cols), n_rows)
    x = cols * step_x
    y = rows * step_y

    phase_id_arr = result.phase_id.cpu().numpy().astype(np.int32)

    xmap = CrystalMap(
        rotations=rotations,
        phase_id=phase_id_arr,
        x=x, y=y,
        phase_list=phases,
        prop={"ci": result.score.cpu().numpy().astype(np.float32)},
    )
    return xmap


def write_ang(xmap, path: str) -> None:
    """Write CrystalMap to a TSL .ang file via orix.io.save."""
    from orix.io import save as orix_save
    orix_save(path, xmap)
