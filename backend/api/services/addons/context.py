"""Everything an add-on is given, and deliberately nothing more.

Two rules decide the contents.

ONLY PLAIN DATA. Arrays, paths, primitives, dicts of those, and one callable —
no CrystalMap, no kikuchipy signal, no open h5py file. That keeps a context of
data picklable, which is the mechanical proof that this can move into a
separate process later. It matters beyond tidiness: a separate process is also
the only boundary at which an add-on could carry a licence of its own.

UNITS IN NAMES. The existing analysis layer stores GOS in radians beside KAM in
degrees, with nothing in the type to say which is which. A facade other people
build against does not get to inherit that. It also does not get to inherit
fields nothing computes: there is no gos_rad and no kam_deg here, because
nothing produces either for a stored IndexingResult and a field that is always
None is a promise this cannot keep.

The facade is deliberately NOT EBSDDataset: that class assumes one phase
(``phase`` returns the first) and square pixels. Publishing it would freeze two
known-wrong simplifications into an interface strangers depend on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ContextError(Exception):
    """A stored result cannot be presented to an add-on, and why."""


def _noop_report(message: str, fraction: Optional[float] = None) -> None:
    """Default progress sink, so an add-on never has to check for one."""


class _ReadOnlyDict(dict):
    """A mapping that refuses the ordinary accident, and still pickles.

    ``_frozen`` guards every ARRAY the context carries and the two DICTS were
    left plain, so an add-on could write into the phase names of a stored
    result and nothing would say so. The standard is ``_frozen``'s, stated the
    same way: a guard against the accident, NOT a boundary -- ``dict.__setitem__``
    reaches past it, and an add-on runs in this process with the user's
    permissions anyway.

    NOT ``types.MappingProxyType``, and the reason is measured rather than
    stylistic: a mappingproxy cannot be pickled at all, and picklability is the
    property this module exists to keep -- the mechanical proof that the whole
    thing can move into a separate process later. ``__reduce__`` rebuilds
    through the constructor, which goes through ``dict.__init__`` rather than
    the ``update`` blocked below.
    """

    def __reduce__(self):
        return (type(self), (dict(self),))

    def _refuse(self, *_args, **_kwargs):
        raise TypeError(
            "an add-on's context is read-only. Copy it first -- dict(context."
            "phase_names) -- and change the copy; what you were handed belongs "
            "to a stored result.")

    __setitem__ = _refuse
    __delitem__ = _refuse
    __ior__ = _refuse
    clear = _refuse
    pop = _refuse
    popitem = _refuse
    setdefault = _refuse
    update = _refuse


def _frozen(array):
    """A read-only VIEW of the caller's array.

    This is a guard against the accident — an add-on that writes into what it
    was handed gets a ValueError instead of corrupting a stored result. It is
    NOT a boundary: ``view.base`` is the caller's array and stays writable,
    and nothing here stops a determined add-on, which runs in this process
    with the user's permissions anyway. The spec says the add-on model is not
    a sandbox; the real boundary, if one is ever wanted, is a separate process.
    """
    if array is None:
        return None
    view = np.asarray(array).view()
    view.setflags(write=False)
    return view


@dataclass(frozen=True)
class AddonContext:
    shape: Tuple[int, int]
    step_size_um: Optional[float]
    phase_ids: Optional[np.ndarray]
    #: phase id -> name. A MAPPING, not a tuple: orix keeps ``not_indexed``
    #: at id -1 and ids need not be contiguous, so a positional sequence
    #: indexed by ``phase_ids`` mislabels every pixel of a real result.
    phase_names: Dict[int, str]
    euler_rad: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    quality: Optional[np.ndarray]
    quality_source: Optional[str]
    eds_at_pct: Dict[str, np.ndarray] = field(default_factory=dict)
    source_file: Optional[Path] = None
    #: Progress sink, ``report(message, fraction=None)``.
    #:
    #: THE ONE FIELD THAT MAY NOT SURVIVE A PROCESS BOUNDARY. Every other
    #: field is plain data and always pickles. This one pickles exactly as the
    #: callback does: a module-level function goes by reference and arrives
    #: unchanged, while a local — a lambda or a closure made where the work
    #: started, which is the usual case — cannot be pickled at all and takes
    #: the whole context down with it (``AttributeError: Can't pickle local
    #: object`` for a local, ``PicklingError`` for a module-scope lambda).
    #:
    #: Nothing here substitutes anything. The alternative — a ``__setstate__``
    #: that quietly restores this no-op — would drop the caller's progress
    #: reporting at exactly the moment the work moved somewhere that needs it,
    #: and say nothing. An exception is information; a restored no-op is a lie.
    report: Callable[..., None] = _noop_report


def _grid(values, shape, field):
    """Reshape a per-pixel array onto the scan grid, or None if it cannot be.

    The per-pixel WIDTH is decided from the element count, never from ``ndim``:
    orix returns Euler angles as ``(N, 3)``, whose ndim already equals the
    grid's, so an ndim-based test yields an empty tail and the reshape of 3N
    elements into ``(rows, cols)`` raises. That failure was swallowed and
    ``euler_rad`` was None for every real result.

    ``field`` is the CONTEXT field being built, and it is required rather than
    optional. Every array reaching here is computed inside Orienta and should
    fit the grid it was computed on, so a mismatch is a bug somewhere else in
    the product — it is logged at WARNING, not DEBUG, because DEBUG is off in
    every real deployment. An anonymous warning ("value array (17,) does not
    fit the (4, 5) grid") names neither the bug nor the field that went
    missing, so the caller says which field it is asking for.
    """
    if values is None:
        return None
    arr = np.asarray(values)
    grid = tuple(int(d) for d in shape)
    if len(grid) != 2:
        return None
    n = grid[0] * grid[1]
    if n <= 0 or arr.size == 0 or arr.size % n != 0:
        logger.warning(
            "%s: array of shape %s does not fit the %s scan grid; the add-on "
            "context will report this field as absent", field, arr.shape, grid)
        return None
    per_pixel = arr.size // n
    # The pixel axis must lead, or a reshape silently transposes the data.
    # Accepted layouts: flat, (N, ...) and already-gridded (rows, cols, ...).
    leading_is_pixels = (
        arr.ndim == 1
        or int(arr.shape[0]) == n
        or tuple(int(d) for d in arr.shape[:2]) == grid
    )
    if not leading_is_pixels:
        logger.warning(
            "%s: array of shape %s is not laid out pixel-first for the %s "
            "scan grid; the add-on context will report this field as absent",
            field, arr.shape, grid)
        return None
    tail = () if per_pixel == 1 else (per_pixel,)
    try:
        return arr.reshape(grid + tail)
    except ValueError:
        logger.warning(
            "%s: array of shape %s does not fit the %s scan grid; the add-on "
            "context will report this field as absent", field, arr.shape, grid)
        return None


def _phase_names(xmap) -> Dict[int, str]:
    """``{phase_id: name}`` from an orix PhaseList.

    ``PhaseList`` iterates as ``(id, Phase)`` TUPLES — verified against orix
    0.14.1 — so ``p.name`` raises on every element. routes/indexing.py and
    routes/analysis.py both already unpack it this way; so does this.
    """
    names: Dict[int, str] = {}
    phases = getattr(xmap, "phases", None)
    if phases is None:
        return names
    try:
        for entry in list(phases):
            phase_id, phase = (
                entry if isinstance(entry, tuple) and len(entry) == 2
                else (getattr(entry, "id", None), entry))
            if phase_id is None:
                continue
            names[int(phase_id)] = str(
                getattr(phase, "name", "") or f"Phase {int(phase_id)}")
    except Exception:
        logger.warning("phase_names: could not read the phase list off the "
                       "xmap; the add-on context will report it as empty",
                       exc_info=True)
    return names


def build_context(result, *, quality=None, quality_source=None,
                  eds_at_pct=None, source_file=None, report=None,
                  step_size_um=None) -> AddonContext:
    """Build the read-only view an add-on receives from a stored result.

    ``step_size_um`` wins over ``metadata``. Nothing in product code writes
    ``metadata["step_size_um"]`` — the key is an h5 ATTRIBUTE on the way out,
    and ``_resolve_step_size_um`` in routes/indexing.py exists because the
    metadata is usually empty. A caller that resolved it must be able to say
    so, or this field is None on every real result.
    """
    metadata = getattr(result, "metadata", None) or {}

    raw_shape = getattr(result, "original_shape", None)
    shape = tuple(int(d) for d in raw_shape) if raw_shape else ()
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ContextError(
            f"this result has no usable scan shape ({raw_shape!r}), so no "
            "add-on output could be matched to a grid. Re-run the indexing "
            "that produced it, or load it from a file that records one."
        )

    xmap = getattr(result, "xmap", None)
    phase_ids = (_grid(getattr(xmap, "phase_id", None), shape, "phase_ids")
                 if xmap is not None else None)
    euler = None
    if xmap is not None:
        rotations = getattr(xmap, "rotations", None)
        to_euler = getattr(rotations, "to_euler", None)
        if callable(to_euler):
            try:
                euler = _grid(to_euler(), shape, "euler_rad")
            except Exception:
                logger.warning("euler_rad: could not read the Euler angles "
                               "off the xmap; the add-on context will report "
                               "it as absent", exc_info=True)

    eds: Dict[str, np.ndarray] = {}
    for element, values in (eds_at_pct or {}).items():
        grid = _grid(values, shape, f"eds_at_pct[{element!r}]")
        if grid is None:
            # Not a None inside a dict the contract types as arrays: an
            # add-on that indexes it would crash on a scan nobody warned it
            # about. Refuse, and name the element and both shapes.
            raise ContextError(
                f"EDS map for {element!r} has shape "
                f"{np.asarray(values).shape} and cannot be laid on the "
                f"{shape} scan grid"
            )
        eds[str(element)] = _frozen(grid)

    src = source_file if source_file is not None else metadata.get("source_file")
    step = step_size_um if step_size_um is not None else metadata.get("step_size_um")

    return AddonContext(
        shape=shape,
        step_size_um=float(step) if step is not None else None,
        phase_ids=_frozen(phase_ids),
        phase_names=_ReadOnlyDict(
            _phase_names(xmap) if xmap is not None else {}),
        euler_rad=_frozen(euler),
        confidence=_frozen(_grid(
            getattr(result, "confidence_scores", None), shape, "confidence")),
        quality=_frozen(_grid(quality, shape, "quality")),
        quality_source=quality_source,
        eds_at_pct=_ReadOnlyDict(eds),
        source_file=Path(src) if src else None,
        report=report or _noop_report,
    )
