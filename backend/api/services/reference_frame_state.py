"""Per-loaded-file coordinate-system (reference-frame) state.

Holds a FrameSpec per canonical source_file path. A FrameSpec resolves to:
  * R_user  — an orix Rotation applied (right-multiply, same convention as
              orientation_frame.to_vendor_export_frame) to orientations for
              IPF + pole-figure display, and opt-in to export.
  * plot    — pole-figure plotting convention (display-only).

NEVER mutates xmap.rotations — this is a render-time setting only.
See docs/superpowers/specs/2026-06-22-pole-figure-coordinate-system-design.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import numpy as np
from orix.quaternion import Rotation

logger = logging.getLogger(__name__)

# Preset sample-frame rotations (right-multiply onto native orientations).
# identity = native frame. The others are the empirically-pinned reconciliations
# documented in orientation_frame.py; expose them as one-click presets.
PRESETS: dict[str, Rotation] = {
    "identity": Rotation.identity(),
    # MTEX default import == raw vendor Euler == our native frame → identity.
    "mtex_default": Rotation.identity(),
    # "Match Aztec/Oxford stored Euler" == native * Rz(+90deg).
    "aztec_oxford": Rotation.from_axes_angles([0, 0, 1], np.deg2rad(90.0)),
    # EDAX/TSL/kikuchipy share our native frame → identity.
    "edax": Rotation.identity(),
}

_VALID_PROJECTION = {"equal_area", "stereographic"}
_VALID_HEMISPHERE = {"upper", "lower"}
_VALID_XDIR = {"east", "north"}
_VALID_MODE = {"preset", "axis_angle", "euler"}

DEFAULT_FRAME_SPEC: dict = {
    "rotation": {
        "mode": "preset",
        "preset": "identity",
        "axis": [0.0, 0.0, 1.0],
        "angle_deg": 0.0,
        "euler_deg": [0.0, 0.0, 0.0],
    },
    "plot": {
        "x_direction": "east",
        "z_into_plane": False,
        "hemisphere": "upper",
        "projection": "equal_area",
    },
    "apply_to_export": False,
}

_frame_registry: dict[str, dict] = {}


def canonical_key(path: str | None) -> str:
    if not path:
        return ""
    return os.path.normcase(os.path.abspath(str(path)))


def normalise_spec(spec: dict | None) -> dict:
    """Return a complete, validated FrameSpec (missing keys filled from default)."""
    spec = spec or {}
    out = json.loads(json.dumps(DEFAULT_FRAME_SPEC))  # deep copy
    rot = spec.get("rotation", {}) or {}
    plot = spec.get("plot", {}) or {}
    out["rotation"].update({k: v for k, v in rot.items() if k in out["rotation"]})
    out["plot"].update({k: v for k, v in plot.items() if k in out["plot"]})
    out["apply_to_export"] = bool(spec.get("apply_to_export",
                                            DEFAULT_FRAME_SPEC["apply_to_export"]))
    # Validate enums (fail loud).
    if out["rotation"]["mode"] not in _VALID_MODE:
        raise ValueError(f"rotation.mode must be one of {_VALID_MODE}")
    if out["rotation"]["mode"] == "preset" and out["rotation"]["preset"] not in PRESETS:
        raise ValueError(f"rotation.preset must be one of {sorted(PRESETS)}")
    if out["plot"]["projection"] not in _VALID_PROJECTION:
        raise ValueError(f"plot.projection must be one of {_VALID_PROJECTION}")
    if out["plot"]["hemisphere"] not in _VALID_HEMISPHERE:
        raise ValueError(f"plot.hemisphere must be one of {_VALID_HEMISPHERE}")
    if out["plot"]["x_direction"] not in _VALID_XDIR:
        raise ValueError(f"plot.x_direction must be one of {_VALID_XDIR}")
    return out


def resolve_r_user(spec: dict) -> Rotation:
    """Resolve the rotation block to a single orix Rotation."""
    rot = spec["rotation"]
    mode = rot["mode"]
    if mode == "preset":
        return PRESETS[rot["preset"]]
    if mode == "axis_angle":
        axis = np.asarray(rot["axis"], dtype=float)
        if axis.shape != (3,) or not np.any(axis):
            raise ValueError("rotation.axis must be a non-zero 3-vector")
        return Rotation.from_axes_angles(axis, np.deg2rad(float(rot["angle_deg"])))
    if mode == "euler":
        eul = np.asarray(rot["euler_deg"], dtype=float)
        if eul.shape != (3,):
            raise ValueError("rotation.euler_deg must be 3 angles")
        return Rotation.from_euler(np.deg2rad(eul))
    raise ValueError(f"unknown rotation mode {mode!r}")


def frame_signature(spec: dict) -> str:
    """Short stable hash of the resolved spec, for cache keys."""
    payload = json.dumps(normalise_spec(spec), sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def get_frame(source_file: str | None) -> dict:
    return _frame_registry.get(canonical_key(source_file),
                               json.loads(json.dumps(DEFAULT_FRAME_SPEC)))


def set_frame(source_file: str, spec: dict) -> dict:
    norm = normalise_spec(spec)
    _frame_registry[canonical_key(source_file)] = norm
    return norm


def get_active_source_file() -> str | None:
    """Resolve the active file: active indexing result's source, else open EBSD file."""
    try:
        from backend.api.routes.indexing import get_last_indexing_result
        res = get_last_indexing_result()
        if res is not None:
            sf = (res.metadata or {}).get("source_file")
            if sf:
                return sf
    except Exception:  # noqa: BLE001 — diagnostic fallback, never fatal
        logger.debug("get_active_source_file: no active result", exc_info=True)
    try:
        from backend.api.routes import ebsd_viewer
        return getattr(ebsd_viewer, "_ebsd_file_path", None)
    except Exception:  # noqa: BLE001
        return None
