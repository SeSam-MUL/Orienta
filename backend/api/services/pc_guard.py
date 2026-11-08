"""Refuse to index a file whose pattern centre nobody supplied.

A file that carries no PC (an EDAX UP1/UP2 without its ``.osc``) is registered in
the calibration store as ``pc_source == "missing"``; its detector then holds
kikuchipy's placeholder (0.5, 0.5, 0.5). Indexing at that PC gives confident,
wrong orientations. Every route that starts an indexing run asks here first.
"""
from __future__ import annotations

from typing import Optional

#: The code the interface translates on (``X-Orienta-Code``), under ``errors.`` of
#: the namespace of the page that started the run.
PC_MISSING_CODE = "pc_missing"


class DetectorMismatchError(RuntimeError):
    """The calibration entry used for a file describes another detector than the file's
    patterns (another pattern shape). ``code`` is what the interface translates on."""

    code = "detector_shape_mismatch"


class PcMissingError(RuntimeError):
    """A file's PC is the placeholder: raised where an HTTP answer is not possible
    (a batch job). ``code`` is what the interface translates on."""

    code = PC_MISSING_CODE


def pc_missing_message(dataset_name: Optional[str] = None) -> str:
    """The English sentence; the interface shows its own translation of the code."""
    which = f"'{dataset_name}'" if dataset_name else "This dataset"
    return (
        f"{which} carries no pattern centre (PC): kikuchipy's placeholder "
        "(0.5, 0.5, 0.5) would be used and the orientations would be wrong. "
        "Calibrate the PC in PC Refinement, or provide the .osc / metadata file "
        "that holds it next to the scan, then start again."
    )


def is_pc_missing(dataset_name, store=None, signal=None, file_key=None) -> bool:
    """True when ``dataset_name``'s PC is the placeholder nobody supplied.

    The calibration store decides when it has an entry. A dataset with NO entry
    (a load path that never registered it) is decided from the signal's own record
    (``Signal.pc_source == "default"``, set by the loader): the guard fails CLOSED -
    an unregistered file that carries no PC is missing, not fine. ``signal`` is the
    loaded signal when the caller has it; otherwise the viewer's signal of that name.
    """
    if store is None:
        from backend.api.services.calibration_store import calibration_store as store
    entry = store.get_entry(dataset_name) if dataset_name else None
    if entry is not None and file_key and getattr(entry, "file_key", None) != file_key:
        # The entry under this NAME belongs to another file of the same stem, or to no
        # file we know (registered without a path): it says nothing about this one
        # (decided from its signal if there is one, else not here).
        entry = None
    if entry is not None:
        return entry.pc_source == "missing"
    if signal is None and dataset_name:
        try:
            from backend.api.routes import ebsd_viewer
            # the viewer's signal of that NAME is the loaded file's: only evidence about
            # this file when it is the one asked about
            loaded = getattr(ebsd_viewer, "_ebsd_file_path", None)
            if not file_key or (loaded and ebsd_viewer._canonical_path(loaded) == file_key):
                signal = ebsd_viewer._raw_signals.get(dataset_name)
        except Exception:  # noqa: BLE001 - nothing loaded: nothing to decide from
            signal = None
    if signal is None:
        return False
    from backend.api.services.calibration_store import _loader_pc_source
    return _loader_pc_source(signal) == "default"


def require_pc(dataset_name, store=None, signal=None, file_key=None) -> None:
    """Raise a coded 400 (``pc_missing``) when the dataset has no PC."""
    if is_pc_missing(dataset_name, store, signal, file_key):
        from backend.api.problem import problem
        raise problem(400, PC_MISSING_CODE, pc_missing_message(dataset_name))
