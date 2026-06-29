"""Build an orix CrystalMap from GPU dictionary-indexing top-k results.

Mirrors the back-mapping pattern from indexing_controller.py:725-760
(the existing kikuchipy-side dictionary path) so the output layout is
indistinguishable to downstream code (Phase Map, EBSD Analysis module,
exporters).
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch

from orix.crystal_map import CrystalMap


def build_crystal_map(
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    rotations,                # orix.quaternion.Rotation
    phase_list,               # orix.crystal_map.PhaseList
    selection_mask: np.ndarray,
    original_shape: Tuple[int, int],
) -> CrystalMap:
    """Construct a CrystalMap from top-k GPU results, back-mapped to grid coords.

    See module docstring.
    """
    if top_indices.ndim != 2 or top_scores.ndim != 2:
        raise ValueError(
            f"top_indices/top_scores must be 2D (n_selected, k), got "
            f"{tuple(top_indices.shape)} / {tuple(top_scores.shape)}"
        )
    if top_indices.shape != top_scores.shape:
        raise ValueError(
            f"shape mismatch: top_indices {tuple(top_indices.shape)} "
            f"vs top_scores {tuple(top_scores.shape)}"
        )
    n_rows, n_cols = original_shape
    if selection_mask.shape != (n_rows, n_cols):
        raise ValueError(
            f"selection_mask shape {selection_mask.shape} != original_shape {original_shape}"
        )

    # selection_mask order MUST match np.argwhere(selection_mask) — that's the
    # convention the indexer uses when flattening selected pixels into the
    # (n_selected, ...) tensor.
    selected_rc = np.argwhere(selection_mask)  # (n_selected, 2) of (row, col)
    n_selected = selected_rc.shape[0]
    if top_indices.shape[0] != n_selected:
        raise ValueError(
            f"top_indices has {top_indices.shape[0]} rows but the mask selects "
            f"{n_selected} pixels"
        )

    indices_np = top_indices.detach().cpu().numpy()
    scores_np = top_scores.detach().cpu().float().numpy().astype(np.float32)

    # Best-match rotation per pixel = rotations[top_indices[:, 0]]
    best_idx = indices_np[:, 0]
    best_rotations = rotations[best_idx]

    xs = selected_rc[:, 1].astype(float)
    ys = selected_rc[:, 0].astype(float)

    # phase_id: 1 everywhere — matches the 1-indexed PhaseList convention
    # the rest of the project uses (spherical-GPU CrystalMaps, and the
    # sht_paths_by_phase metadata in _attach_indexing_metadata which keys
    # with i+1). 0-indexed phase_ids break the Pattern Match dialog's
    # SHT-path lookup.
    phase_id = np.ones(n_selected, dtype=int)

    xmap = CrystalMap(
        rotations=best_rotations,
        phase_id=phase_id,
        x=xs,
        y=ys,
        phase_list=phase_list,
        prop={"scores": scores_np, "simulation_indices": indices_np.astype(int)},
        scan_unit="px",
    )
    return xmap
