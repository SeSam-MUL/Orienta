"""The PC (and detector tilts) a result was INDEXED at, recorded with the result.

An export used to read the calibration store when it was written, so a PC changed
after the run (typed, refined, a grid map) was stated as the PC of the run: the same
orientations exported twice carried two different pattern centres. The run takes a
record when it starts and keeps it in ``result.metadata["pc_at_run"]``; the exports and
the citation steps read that. A result without the record (indexed before it existed,
or imported from a file that does not state one) does not get a PC from the store: its
exports say "not recorded".
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: ``result.metadata`` key.
KEY = "pc_at_run"


def pc_run_record(dataset_name, store=None, detector=None, typed_pc=None) -> Optional[dict]:
    """What a run starting NOW on ``dataset_name`` indexes at.

    ``pc`` is the PC (Bruker; the mean of a per-pixel map, flagged ``per_pixel`` with the
    map's shape), ``None`` when the file carries none (``pc_source == "missing"``);
    ``pc_source`` is the store's source ("header", "manual", "refined", ...); the tilts
    are the store's. ``typed_pc`` is a PC given with the request itself (it is the PC
    of this run, ``pc_source == "typed_for_run"``). ``detector`` is the fallback when the
    dataset is not in the store. ``None`` when nothing is known.

    ``pc_origin_kind`` is where that PC came from (``pc_in_use.pc_in_use_of``: "file", "find", "refine", "typed",
    "grid", "calibration", "none"; "typed" for ``typed_pc``; ``None`` when the PC is not the store's) and
    ``pc_quality`` the stored quality of the search or calibration it is (``pc_search.quality_of``), else ``None``.
    """
    if store is None:
        from backend.api.services.calibration_store import calibration_store as store
    entry = store.get_entry(dataset_name) if dataset_name else None
    if entry is not None:
        rec = {
            "pc": None if entry.pc_source == "missing"
            else [float(v) for v in np.asarray(entry.pc_single, dtype=float).ravel()[:3]],
            "pc_source": entry.pc_source,
            "per_pixel": entry.pc_map is not None,
            "map_shape": ([int(v) for v in np.asarray(entry.pc_map).shape[:-1]]
                          if entry.pc_map is not None else None),
            "sample_tilt": float(entry.sample_tilt),
            "tilt": float(entry.tilt),
            "azimuthal": float(entry.azimuthal),
        }
    elif detector is not None:
        arr = np.asarray(detector.pc, dtype=float).reshape(-1, 3)
        rec = {
            "pc": [float(v) for v in arr.mean(axis=0)],
            "pc_source": "detector",
            "per_pixel": arr.shape[0] > 1,
            "map_shape": None,
            "sample_tilt": float(getattr(detector, "sample_tilt", 70.0)),
            "tilt": float(getattr(detector, "tilt", 0.0)),
            "azimuthal": float(getattr(detector, "azimuthal", 0.0)),
        }
    else:
        return None
    if typed_pc is not None and len(typed_pc) == 3:
        rec.update(pc=[float(v) for v in typed_pc], pc_source="typed_for_run",
                   per_pixel=False, map_shape=None)
        rec["calibration"] = None
        rec["pc_origin_kind"], rec["pc_quality"] = "typed", None
    else:
        # the named calibration (and version) whose PC this run is indexed at, if it still is that PC:
        # a result later shows "computed with an older PC" when the calibration has moved on
        rec["calibration"] = _calibration_ref(dataset_name, store)
        # where that PC came from and, when it is a search result or a calibration, how good it was
        rec["pc_origin_kind"], rec["pc_quality"] = _origin_of_pc(dataset_name, store, entry is not None)
    return rec


def _origin_of_pc(dataset_name, store, in_store: bool):
    """``(kind, quality)`` of the dataset's PC (``pc_in_use``); ``(None, None)`` for a PC that is not the store's
    (a detector the run was handed) or when it cannot be told: the record is still true without them."""
    if not in_store:
        return None, None
    try:
        from backend.api.services import pc_in_use
        return (pc_in_use.pc_in_use_of(dataset_name, store)["kind"],
                pc_in_use.pc_quality_of(dataset_name, store))
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("could not look up where the PC of %r came from", dataset_name,
                                            exc_info=True)
        return None, None


def _calibration_ref(dataset_name, store):
    """``{"id", "name", "version"}`` of the calibration the dataset's PC is, else ``None``."""
    if not dataset_name:
        return None
    try:
        from backend.api.services import pc_calibration_apply
        return pc_calibration_apply.current_ref(dataset_name, store)
    except Exception:  # noqa: BLE001 - the record is still true without the reference
        import logging
        logging.getLogger(__name__).warning("could not look up the calibration of %r", dataset_name,
                                            exc_info=True)
        return None


def record_of(result) -> Optional[dict]:
    """The record a result carries, or ``None`` ("PC at index time not recorded")."""
    md = getattr(result, "metadata", None) or {}
    rec = md.get(KEY)
    return rec if isinstance(rec, dict) else None
