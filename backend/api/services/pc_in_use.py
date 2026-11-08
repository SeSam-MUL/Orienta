"""The PC a dataset is using now, and where it came from: the header of every PC page.

``pc_in_use_of(dataset)`` answers ``{"pc", "per_pixel", "kind", "calibration", "convention",
"convention_known"}``. It reads; nothing here writes a PC.

``kind`` says where the PC in use came from:

* ``"none"``         the file carries no PC and nobody set one (``pc_source`` "missing"); ``pc`` is ``None``
* ``"file"``         the file's own header (also a copy of a dataset that holds the file's PC)
* ``"find"``         a "Find PC" search (``pc_refinement["mode"] == "find"``)
* ``"refine"``       a "Global PC Refine" (mode "refine"; a record written before ``mode`` existed is a refine)
* ``"typed"``        numbers a person typed
* ``"grid"``         a per-pixel map fitted by the grid calibration (``pc_source`` "refined_map")
* ``"calibration"``  a named calibration's PC; ``calibration`` is ``{"id", "name", "version"}`` while the PC still
                     is that version's (``pc_calibration_apply.current_ref``), else ``None``

A dataset whose PC was passed on from another (``pc_source`` "inherited" or "propagated") has the kind of what
was passed on: the record of a find or a refine travels with the PC, a typed PC is "typed", a copy of a dataset
takes the kind of its parent.

``convention`` is the one the open FILE's vendor uses for its numbers ("oxford", "tsl", "bruker"); Orienta itself
keeps every PC in the Bruker convention. ``convention_known`` is False when the format does not say (then
"bruker" is the fallback, not a statement).

``pc_quality_of`` is the stored quality of the search or the calibration the PC in use is, if it is one.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

#: ``pc_refinement["mode"]`` values that name a kind of their own.
_SEARCH_KINDS = ("find", "refine")
PC_ATOL = 1e-9


def _store(store=None):
    if store is not None:
        return store
    from backend.api.services.calibration_store import calibration_store
    return calibration_store


def _mean_pc(entry) -> list:
    arr = entry.pc_map if entry.pc_map is not None else entry.pc_single
    return [float(v) for v in np.asarray(arr, dtype=float).reshape(-1, 3).mean(axis=0)]


def refinement_kind(refinement: Optional[dict]) -> str:
    """"find" or "refine" for a refinement record; a record without ``mode`` (written before it existed) is a refine."""
    mode = (refinement or {}).get("mode")
    return mode if mode in _SEARCH_KINDS else "refine"


def _kind_of_entry(entry, st, seen=()) -> str:
    """The kind of the PC an entry holds, without the calibration check (that needs the dataset's name)."""
    source = entry.pc_source
    if source == "missing":
        return "none"
    if source == "header":
        return "file"
    if source == "manual":
        return "typed"
    if source == "refined":
        return refinement_kind(entry.pc_refinement)
    if source == "refined_map":
        return "grid"
    if source == "calibration":
        return "calibration"
    if source == "propagated":
        # a refinement record travels with a PC that was passed on; without one a person typed it
        return refinement_kind(entry.pc_refinement) if entry.pc_refinement else "typed"
    if source == "inherited":
        if entry.pc_refinement:
            return refinement_kind(entry.pc_refinement)
        parent = st.get_entry(entry.parent_name) if entry.parent_name and entry.parent_name not in seen else None
        if parent is not None and parent.pc_source != "missing":
            return _kind_of_entry(parent, st, (*seen, entry.dataset_name))
        return "file"
    return "file"


def pc_in_use_of(dataset: Optional[str], store=None) -> dict:
    """See the module docstring. A dataset the store does not know has kind "none"."""
    st = _store(store)
    entry = st.get_entry(dataset) if dataset else None
    if entry is None:
        return {"pc": None, "per_pixel": False, "kind": "none", "calibration": None,
                "convention": None, "convention_known": False}
    convention = getattr(entry, "convention", None)
    out = {"pc": None, "per_pixel": False, "kind": "none", "calibration": None,
           "convention": convention or "bruker", "convention_known": bool(convention)
           and bool(getattr(entry, "convention_known", False))}
    if entry.pc_source == "missing":
        return out
    out["pc"] = _mean_pc(entry)
    out["per_pixel"] = entry.pc_map is not None
    ref = None
    try:
        from backend.api.services import pc_calibration_apply
        ref = pc_calibration_apply.current_ref(dataset, st)
    except Exception:  # noqa: BLE001 - information; the kind below is still true
        logger.warning("could not tell whether %r runs on a calibration", dataset, exc_info=True)
    if ref is not None:
        out["kind"], out["calibration"] = "calibration", ref
    else:
        out["kind"] = _kind_of_entry(entry, st)
    return out


def pc_quality_of(dataset: Optional[str], store=None) -> Optional[dict]:
    """``quality_of`` of the search that made the dataset's PC (while the PC still is that search's result), else the
    quality stored with the calibration version it runs on; ``None`` when it is neither (a typed or file PC)."""
    if not dataset:
        return None
    st = _store(store)
    entry = st.get_entry(dataset)
    if entry is None or entry.pc_source == "missing":
        return None
    try:
        from backend.api.services import pc_calibration_apply as use
        from backend.api.services import pc_calibrations as cal
        from backend.api.services.pc_search import quality_of
        pc = np.asarray(_mean_pc(entry), dtype=float)
        search = use.search_of(dataset, st)
        if search and np.allclose(search["pc"], pc, rtol=0, atol=PC_ATOL):
            return quality_of(search)
        ref = use.current_ref(dataset, st)
        if ref is None:
            return None
        return cal.quality_of_version(cal.get(ref["id"]), ref["version"])
    except Exception:  # noqa: BLE001
        logger.warning("could not look up the quality of the PC of %r", dataset, exc_info=True)
        return None
