"""Which variant (scan group x pattern source) each loaded dataset holds.

Same lifecycle as the registry of ``scan_geometry``: set on a file load, copied on a deep copy
and on a crop (a crop of a dataset holds the same variant), dropped with the dataset, cleared on
a file load, carried through the per-file stash. ``safe_loader.load_ebsd_safe`` hands the variant
it loaded to the route on the signal (``signal.metadata.Orienta.load_variant``); the route calls
:func:`register_from_signal`, which reads it once and takes it off the signal, because a copy left
on the signal would outlive the dataset it describes.

A dataset without an entry was loaded without a choice: the file's default variant.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

from backend.api.services.load_choices import LoadChoiceError, LoadVariant

logger = logging.getLogger(__name__)

MIRROR_KEY = "Orienta.load_variant"

_lock = threading.RLock()
_variants: Dict[str, LoadVariant] = {}


def register(name: str, variant: LoadVariant) -> None:
    with _lock:
        _variants[name] = variant
    logger.info("load variant for %r: measurement %s, %s patterns", name, variant.measurement,
                variant.pattern_source)


def variant_for(name: str) -> Optional[LoadVariant]:
    """The variant dataset ``name`` was loaded as, or None when no choice was recorded."""
    with _lock:
        return _variants.get(name)


def copy_to(src: str, dst: str) -> Optional[LoadVariant]:
    """A deep copy or a crop holds what its parent holds."""
    with _lock:
        variant = _variants.get(src)
        if variant is None:
            return None
        _variants[dst] = variant
        return variant


def drop(name: str) -> None:
    with _lock:
        _variants.pop(name, None)


def clear_all() -> None:
    with _lock:
        _variants.clear()


def snapshot(names) -> Dict[str, LoadVariant]:
    """The variants of ``names``, for the per-file stash. Frozen objects: no copy needed."""
    with _lock:
        return {n: _variants[n] for n in names if n in _variants}


def restore(snap: Dict[str, LoadVariant], skip: Optional[str] = None) -> None:
    """Put stashed variants back. A name that has one already keeps it (the freshly loaded raw
    dataset, or a live dataset), and ``skip`` is never touched."""
    with _lock:
        for name, variant in (snap or {}).items():
            if name != skip and name not in _variants:
                _variants[name] = variant


def drop_signal_mirror(signal) -> None:
    """Take the loader's copy of the variant off ``signal``. It is a transport from the loader to
    the registry; left on the signal it is carried into deep copies and crops, where it would
    describe a dataset that may since have been replaced. A stand-in signal has nothing to drop."""
    try:
        del signal.metadata.Orienta.load_variant
    except Exception:  # noqa: BLE001 - no mirror, or a stand-in signal
        pass


def peek_mirror(signal) -> Optional[LoadVariant]:
    """The variant the loader says it loaded, read off ``signal`` without taking it off. None when
    the loader recorded none (loaded without a choice, or a format that has one variant)."""
    try:
        mirror = signal.metadata.get_item(MIRROR_KEY, None)
        mirror = mirror.as_dictionary() if hasattr(mirror, "as_dictionary") else mirror
    except Exception:  # noqa: BLE001 - a stand-in signal without metadata
        return None
    if not isinstance(mirror, dict):
        return None
    try:
        return LoadVariant.from_dict(mirror)
    except LoadChoiceError:
        logger.warning("could not read the load variant on a signal: %r", mirror)
        return None


def register_from_signal(name: str, signal) -> Optional[LoadVariant]:
    """Register the variant a freshly loaded ``signal`` carries under ``name`` and take the
    loader's copy off the signal. A signal without one was loaded without a choice: any entry
    ``name`` had from an earlier load is dropped and None is returned."""
    variant = peek_mirror(signal)
    drop_signal_mirror(signal)
    if variant is None:
        drop(name)
        return None
    register(name, variant)
    return variant


def apex_of_stamp(stamp) -> Optional[dict]:
    """The metadata pairing a stamp names (``{"h5_path", "measurement_id"}``), or None.

    A pattern file that names no measurement is paired with one by the user, and which measurement it
    was paired with is part of "which variant of this file" (``ebsd_viewer._variant_key``): the stamp
    of a result carries it next to the measurement x pattern source."""
    value = stamp.get("apex") if isinstance(stamp, dict) else None
    if isinstance(value, dict) and value.get("measurement_id"):
        return value
    return None


def _is_stamp(value) -> bool:
    """A stamp names a pattern source, or a metadata pairing, or both."""
    return isinstance(value, dict) and bool(value.get("pattern_source") or apex_of_stamp(value))


def stamp_of_result(result) -> Optional[dict]:
    """The variant the patterns of ``result`` were, as stamped when it ran: ``indexed_variant`` (what
    an imported file said about its original scan) before ``load_variant`` (the variant of
    ``source_file``). None = the default variant, also for a result stored before the stamp existed.
    The one rule for "which variant does this result belong to"."""
    meta = getattr(result, "metadata", None)
    if not isinstance(meta, dict):
        return None
    for key in ("indexed_variant", "load_variant"):
        value = meta.get(key)
        if _is_stamp(value):
            return value
    return None


def variant_of_result(result) -> Optional[LoadVariant]:
    """:func:`stamp_of_result` as a ``LoadVariant`` (None = the default variant)."""
    from backend.api.services.load_choices import variant_from_stamp
    return variant_from_stamp(stamp_of_result(result))


def file_stamp_of_result(result) -> Optional[dict]:
    """The variant of the result's ``source_file`` that has to be loaded for the result to be the one on
    screen: ``load_variant`` alone (None = the default variant, also when the key is missing).

    The other question of the pair, next to :func:`stamp_of_result` (what the patterns of the result
    were). For a result that ran on its file they are the same stamp. For an imported result they are
    not: its ``source_file`` is the export, ``load_variant`` says how THAT file is loaded, and
    ``indexed_variant`` says what its original scan was; activating the result loads the export, so it
    must compare and load by the first (the second would deactivate it on every load of its own file,
    and ask a light export for a scan group it does not have). Activation and re-activation both ask
    here."""
    meta = getattr(result, "metadata", None)
    value = meta.get("load_variant") if isinstance(meta, dict) else None
    return value if _is_stamp(value) else None


def apex_of_result(result) -> Optional[dict]:
    """The metadata pairing the file of ``result`` was loaded with when it ran (None: the file names
    its measurement itself, or has none). Read from ``load_variant`` like :func:`file_stamp_of_result`:
    it is the pairing of ``source_file``, the file a reader opens."""
    return apex_of_stamp(file_stamp_of_result(result))


def same_file(a, b) -> bool:
    """Whether two paths name the same file (case and symlinks resolved)."""
    import os
    try:
        return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))
    except (OSError, ValueError):
        return str(a) == str(b)


def variant_for_source(result, path) -> Optional[LoadVariant]:
    """The variant a reader takes when it opens ``path`` for ``result`` (None = the file's default).

    THE rule for every reader of a result's data from a file. A stamp describes the SCAN the patterns
    came from. The file a reader opens is that scan for a result that ran on it, and the stamp applies.
    An imported result is different: its ``source_file`` is the export it came from, and its stamp
    (``indexed_variant``) is what that export says about the original scan. A light export holds no scan
    group, so asking it for measurement "2" was asking for a group that is not there, and the stored
    quality map of the result was lost; a full export is a copy of the scan and does hold the group, so
    there the stamp still applies. Any other path (the viewer's file, an unrelated file) is not the
    export and the stamp stays: a reader of a file that lacks the measurement refuses, as it must.
    A file that cannot be opened keeps the stamp too, so that the reader reports the real failure.
    """
    variant = variant_of_result(result)
    if variant is None or variant.measurement is None:
        return variant
    meta = getattr(result, "metadata", None)
    if not (isinstance(meta, dict) and meta.get("imported_from_h5")):
        return variant
    source = meta.get("source_file")
    if not path or not source or not same_file(path, source):
        return variant
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            holds = variant.measurement in f
    except Exception:  # noqa: BLE001 - unreadable: let the reader meet the same file
        return variant
    return variant if holds else None
