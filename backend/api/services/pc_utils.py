"""Pattern Center deviation utilities for dictionary indexing."""
from __future__ import annotations

import math
from typing import Optional, Sequence


def pc_deviation_percent(
    current_pc: Optional[Sequence[float]],
    dict_pc: Optional[Sequence[float]],
) -> Optional[float]:
    """Calculate PC deviation as percentage of current PC magnitude.

    Parameters
    ----------
    current_pc : list of 3 floats or None
        Current pattern center [xpc, ypc, zpc].
    dict_pc : list of 3 floats or None
        Dictionary's pattern center [xpc, ypc, zpc].

    Returns
    -------
    float or None
        Deviation in percent, or None if either PC is missing.
    """
    if not current_pc or not dict_pc:
        return None
    if len(current_pc) < 3 or len(dict_pc) < 3:
        return None

    delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_pc, dict_pc)))
    magnitude = math.sqrt(sum(v ** 2 for v in current_pc))
    if magnitude == 0:
        return 100.0
    return round((delta / magnitude) * 100.0, 1)


def pc_match_color(deviation_percent: Optional[float]) -> str:
    """Return color category for PC deviation.

    Returns 'good' (<5%), 'ok' (5-15%), 'bad' (>15%), 'unknown' (None).
    """
    if deviation_percent is None:
        return "unknown"
    if deviation_percent < 5:
        return "good"
    if deviation_percent <= 15:
        return "ok"
    return "bad"
