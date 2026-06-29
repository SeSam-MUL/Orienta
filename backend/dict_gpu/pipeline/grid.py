"""Sample the fundamental zone of a point group at a given angular step.

Wraps orix.sampling.get_sample_fundamental — that's the canonical path.
We expose a thin wrapper so the indexer can change angular_step at runtime
without each caller re-implementing the orix call.
"""
from __future__ import annotations
from typing import Union

from orix.sampling import get_sample_fundamental
from orix.quaternion import Rotation
from orix.quaternion import symmetry as _orix_sym


def sample_orientations(
    point_group: Union[str, object],
    angular_step_deg: float,
) -> Rotation:
    """Sample the fundamental zone at the requested angular step.

    Accepts either a Schoenflies string (e.g. "m-3m") or an orix symmetry
    object (e.g. `mp.phase.point_group`).
    """
    if angular_step_deg <= 0:
        raise ValueError(f"angular_step_deg must be > 0, got {angular_step_deg}")

    # Resolve string → orix symmetry. orix accepts a `point_group=` kwarg
    # that takes either form, but we normalise here so the call site is uniform.
    if isinstance(point_group, str):
        sym = getattr(_orix_sym, _schoenflies_to_orix_attr(point_group), None)
        if sym is None:
            # orix's get_sample_fundamental can also take a string directly via
            # its `point_group=` kwarg in newer versions. Try that.
            return get_sample_fundamental(
                resolution=angular_step_deg, point_group=point_group
            )
    else:
        sym = point_group

    return get_sample_fundamental(resolution=angular_step_deg, point_group=sym)


# Map common Schoenflies strings to orix's symmetry attribute names.
_SCHOENFLIES_MAP = {
    "m-3m": "Oh",
    "m3m": "Oh",
    "-3m": "D3d",
    "6/mmm": "D6h",
    "4/mmm": "D4h",
    "mmm": "D2h",
    "2/m": "C2h",
    "-1": "Ci",
    "1": "C1",
}


def _schoenflies_to_orix_attr(s: str) -> str:
    return _SCHOENFLIES_MAP.get(s, s)  # fall through to original if unknown
