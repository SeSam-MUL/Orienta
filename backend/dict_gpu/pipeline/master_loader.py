"""Load a kikuchipy master pattern or a legacy dict-cache .h5.

Legacy dict caches are produced by simulation/dictionary_generator.py.
We support them so users with existing caches can still get GPU-accelerated
matching (the indexer's matching-only branch).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from backend.dict_gpu.exceptions import MasterPatternError


@dataclass
class MasterPayload:
    """Tagged-union dataclass returned by the loader.

    `kind == "master"`:
        - `master` is a kikuchipy.EBSDMasterPattern
        - `dict_patterns` and `dict_rotations` are None
    `kind == "dict_cache"`:
        - `master` is None
        - `dict_patterns` is a (n_rot, h, w) numpy float32 array
        - `dict_rotations` is an orix.quaternion.Rotation, len == n_rot

    `phase` is the orix.crystal_map.Phase carrying point_group and
    name. Populated for `master` from `master.phase`, and for
    `dict_cache` from `sig.xmap.phases_in_data` when kikuchipy can
    load the file (most dictionary files written by kikuchipy carry
    this metadata). May be None for raw h5 dict caches that lack
    phase metadata — the caller must then fail loud about missing
    point_group rather than rendering misleading grey IPF.
    """
    kind: Literal["master", "dict_cache"]
    master: object | None
    dict_patterns: np.ndarray | None
    dict_rotations: object | None
    phase: object | None = None


def _try_load_as_master(path: Path):
    """Try loading via kikuchipy. Returns the master or None on failure.

    Discrimination is done via the master-pattern-specific `.projection`
    attribute (square Lambert / stereographic). Plain EBSD signals don't
    have this. We keep the check defensive: if kikuchipy returns anything
    other than an `EBSDMasterPattern`-shaped object, we bail out so the
    legacy-cache branch can take over.
    """
    try:
        import kikuchipy as kp
    except ImportError:
        return None
    try:
        mp = kp.load(str(path))
    except Exception:
        return None
    # Discriminate: an EBSDMasterPattern has these attributes.
    # `.projection` is master-only (plain EBSD signals don't have it).
    if (
        hasattr(mp, "phase")
        and hasattr(mp, "data")
        and hasattr(mp, "projection")
    ):
        return mp
    return None


def _try_load_as_dict_cache(path: Path) -> tuple[np.ndarray, object] | None:
    """Try loading as a legacy dict cache. Returns (patterns, rotations) or None."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            patterns_key = next(
                (k for k in ("patterns", "dictionary", "data") if k in keys),
                None,
            )
            rotations_key = next(
                (k for k in ("rotations", "quats", "quaternions") if k in keys),
                None,
            )
            if patterns_key is None or rotations_key is None:
                return None
            patterns = np.asarray(f[patterns_key][...], dtype=np.float32)
            quats = np.asarray(f[rotations_key][...], dtype=np.float64)
    except Exception:
        return None

    if patterns.ndim != 3:
        return None
    if quats.ndim != 2 or quats.shape[1] != 4:
        return None
    if patterns.shape[0] != quats.shape[0]:
        return None

    try:
        from orix.quaternion import Rotation
    except ImportError:
        return None
    rotations = Rotation(quats)
    return patterns, rotations


def _try_load_dict_via_kikuchipy(path: Path):
    """Try loading the file as a kikuchipy EBSD signal (a dictionary written
    by kikuchipy carries proper xmap + phase metadata in its CrystalMap
    group). Returns ``(patterns, rotations, phase)`` or ``None``.

    This complements ``_try_load_as_dict_cache`` which uses raw h5py and
    misses the phase info. Preferred over the raw path because IPF
    rendering needs ``phase.point_group`` to colour orientations — without
    it the downstream xmap renders as flat grey.
    """
    try:
        import kikuchipy as kp
    except ImportError:
        return None
    try:
        sig = kp.load(str(path))
    except Exception:
        return None
    # Must be a dictionary-style EBSD signal: has xmap + data, NOT a
    # master pattern (those have .projection and route to _try_load_as_master).
    if not hasattr(sig, "data") or hasattr(sig, "projection"):
        return None
    xmap = getattr(sig, "xmap", None)
    if xmap is None or getattr(xmap, "rotations", None) is None:
        return None

    patterns = np.asarray(sig.data, dtype=np.float32)
    if patterns.ndim == 4 and patterns.shape[0] == 1:
        patterns = patterns.squeeze(0)
    if patterns.ndim != 3:
        return None

    rotations = xmap.rotations
    if rotations.size != patterns.shape[0]:
        return None

    # Phase extraction: dictionary xmaps typically have exactly one phase.
    # PhaseList has no len() — check its .ids array instead. phases_in_data
    # filters to phases actually present (skips the not_indexed sentinel
    # that orix adds at id=-1).
    phase = None
    try:
        phases = xmap.phases_in_data
        ids = list(phases.ids)
        if len(ids) >= 1:
            phase = phases[ids[0]]
    except Exception:
        # Fallback: try xmap.phases (unfiltered) — kikuchipy dictionary
        # files sometimes have phase metadata in `phases` without the
        # `is_in_data` mask being set per pixel.
        try:
            ids = list(xmap.phases.ids)
            if len(ids) >= 1:
                phase = xmap.phases[ids[0]]
        except Exception:
            phase = None

    return patterns, rotations, phase


def load_master_or_dict(path: str | Path) -> MasterPayload:
    """Load a master pattern or legacy dict cache.

    Discrimination order:
        1. Try as kikuchipy EBSDMasterPattern (must have `.projection`).
        2. Try as a kikuchipy-written dictionary EBSD signal (carries
           phase metadata in its xmap — needed for IPF colouring).
        3. Try as a raw legacy dict-cache HDF5 with /patterns +
           /rotations datasets (no phase metadata, IPF will fail loud).
        4. If none match, raise MasterPatternError.

    Raises MasterPatternError if path doesn't exist, is unreadable, or
    matches none of the supported shapes.
    """
    p = Path(path)
    if not p.exists():
        raise MasterPatternError(f"Master pattern file not found: {p}")
    if not p.is_file():
        raise MasterPatternError(f"Path is not a file: {p}")

    mp = _try_load_as_master(p)
    if mp is not None:
        # Master pattern's phase carries point_group directly from the CIF.
        master_phase = getattr(mp, "phase", None)
        return MasterPayload(
            kind="master", master=mp,
            dict_patterns=None, dict_rotations=None,
            phase=master_phase,
        )

    kp_dict = _try_load_dict_via_kikuchipy(p)
    if kp_dict is not None:
        patterns, rotations, phase = kp_dict
        return MasterPayload(
            kind="dict_cache",
            master=None,
            dict_patterns=patterns,
            dict_rotations=rotations,
            phase=phase,
        )

    cache = _try_load_as_dict_cache(p)
    if cache is not None:
        patterns, rotations = cache
        return MasterPayload(
            kind="dict_cache",
            master=None,
            dict_patterns=patterns,
            dict_rotations=rotations,
            phase=None,  # raw .h5 with no phase metadata
        )

    raise MasterPatternError(
        f"File {p} is neither a kikuchipy master pattern nor a recognised "
        f"legacy dict-cache (.h5 with /patterns + /rotations datasets)."
    )
