"""What the Indexing and PC Refinement pages need to show and change the
reflector families of a Hough phase.

Both pages ask the same questions of the same per-phase registry
(``hough_reflectors``): the family table, change the choice, check one
hand-typed family, what the choice costs.  The pages differ only in how they
name the phase - a CIF path on the Indexing page, the loaded phase's name on the
PC Refinement page - so each route resolves its phase and calls these.

A problem is a ``ReflectorError`` that carries the stable ``code`` and
``params`` of ``hough_reflectors.SpecError`` (the page words it in the user's
language); routes turn it into a 400 ``{code, message, params}``.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import hough_reflectors as hr

_PHASES = OrderedDict()
_PHASES_LOCK = threading.Lock()
_PHASES_MAX = 8


class ReflectorError(ValueError):
    """Anything the user can act on; ``as_detail`` is the 400 body."""

    def __init__(self, code, message, status=400, **params):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.params = params

    def as_detail(self):
        return {"code": self.code, "message": self.message, "params": self.params}


def _from_spec_error(exc):
    return ReflectorError(exc.code, str(exc), **exc.params)


def phase_for_path(cif_path):
    """The Hough phase of a CIF (cached by path, mtime and size).

    The first read of a CIF in a two-origin space group takes a few seconds, and
    the page asks again on every tick, so the phase is kept.  Nothing here
    changes it (``hough_reflectors`` works on copies).
    """
    path = Path(str(cif_path or ""))
    if not str(cif_path or "").strip():
        raise ReflectorError("no_cif", "A CIF path is required.")
    if not path.is_file():
        raise ReflectorError("not_found", f"CIF not found: {path}", status=404,
                             path=str(path))
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    with _PHASES_LOCK:
        if key in _PHASES:
            _PHASES.move_to_end(key)
            return _PHASES[key]
    try:
        phase = hr.phase_from_cif(path)
    except Exception as exc:  # noqa: BLE001 - diffpy raises unrelated types
        raise ReflectorError("unreadable", f"{path.name} could not be read: {exc}",
                             path=str(path), reason=str(exc)[:300]) from exc
    with _PHASES_LOCK:
        _PHASES[key] = phase
        while len(_PHASES) > _PHASES_MAX:
            _PHASES.popitem(last=False)
    return phase


def table(phase, key):
    """The family table under the choice stored for ``key``.

    A stored choice that no longer fits the crystal is reported in
    ``spec_error`` (with the default table), so the page can show it and offer
    to reset it - it is never applied.
    """
    stored = hr.get_spec(key)
    try:
        out = hr.describe(phase, stored)
    except hr.SpecError as exc:
        out = hr.describe(phase, None)
        out["spec"] = stored
        out["mode"] = stored["mode"] if stored else "default"
        out["spec_error"] = {"code": exc.code, "message": str(exc),
                             "params": exc.params}
    out["registry_version"] = hr.registry_version()
    return out


def change(phase, key, spec):
    """Store a new choice (``None`` = back to the default) and return the table.

    ``{"mode": "top_n", "n": N}`` is expanded here to the N strongest families
    the indexer would keep.  Everything is checked BEFORE it is stored: a bad
    choice leaves the old one in place.
    """
    try:
        if spec is None:
            hr.set_spec(key, None)
            return table(phase, key)
        if isinstance(spec, dict) and spec.get("mode") == "top_n":
            rule = (hr.get_spec(key) or {}).get("rule") or None
            new = hr.expand_top_n(phase, spec.get("n"), rule)
        else:
            new = hr.spec_with_fingerprint(phase, spec)
        if new is not None:
            shown = hr.describe(phase, new)  # raises for a family that cannot be used
            if shown["n_effective"] < hr.MIN_FAMILIES:
                raise hr.SpecError(
                    "too_few",
                    f"Hough indexing needs at least {hr.MIN_FAMILIES} distinct "
                    f"reflector families; this selection keeps {shown['n_effective']}.",
                    minimum=hr.MIN_FAMILIES, kept=shown["n_effective"])
            _probe_library(phase, new)
        hr.set_spec(key, new)
    except hr.SpecError as exc:
        raise _from_spec_error(exc) from exc
    return table(phase, key)


def _library_rows(phase, rows):
    """Rows the band-triplet library of this reflector list needs; 0 when it is
    smaller than the probe cap (negligible). Raises whatever PyEBSDIndex raises
    for a list it cannot build.

    Builds the BandIndexer of the phase only - the same call kikuchipy's
    ``get_indexer`` makes first - and not a whole EBSDIndexer, which on a machine
    with OpenCL also opens a GPU context for its Radon plan: one per call, and
    this is called on every click.
    """
    import re

    import ebsd_utils as eu
    from kikuchipy.indexing._hough_indexing import _get_pyebsdindex_phaselist
    from orix.crystal_map import PhaseList

    p = phase.deepcopy()
    fixed = eu._standard_setting_phase_list(PhaseList(p))
    try:
        with eu._capped_triplet_library(eu._PREDICT_PROBE_ROWS):
            _get_pyebsdindex_phaselist(fixed, [rows])
    except MemoryError as exc:
        m = re.search(r"need ([0-9,]+) rows", str(exc))
        if m:
            return int(m.group(1).replace(",", ""))
        raise
    return 0


def _probe_library(phase, spec):
    """Build the band-triplet library of a selection once, under the probe cap.

    PyEBSDIndex fails on some lists of families with an array error deep in its
    library builder (Mg: {0002} with {2-1-11}). Better to hear that when the
    selection is made than when a run starts. Too big for the probe is not a
    failure here - the cost line says how big.
    """
    rows = hr.spec_rows(phase.deepcopy(), spec)[0].tolist()
    try:
        _library_rows(phase, rows)
    except MemoryError:
        return
    except Exception as exc:  # noqa: BLE001 - PyEBSDIndex raises unrelated types
        raise ReflectorError(
            "library_failed",
            "PyEBSDIndex cannot build its band-triplet library from this "
            f"selection ({type(exc).__name__}: {exc}). Add or remove a family.",
            reason=f"{type(exc).__name__}: {exc}"[:300]) from exc


def check_family(phase, key, hkl, spec=None):
    """Properties of one hand-typed family, or why it cannot be added.

    ``spec`` is the choice the page is working on (so a family it already
    holds is a duplicate); the stored one when the page sends none.
    """
    try:
        current = spec if spec is not None else hr.get_spec(key)
        return hr.validate_family(phase, hkl, current)
    except hr.SpecError as exc:
        raise _from_spec_error(exc) from exc


def cost(phase, key, n_bands=12):
    """What the library of the stored choice would cost (nothing is built).

    The sizes come from the REFUSED request of PyEBSDIndex's table allocation
    (see ``ebsd_utils.predict_triplet_library``); the library is sized from the
    poles and the lattice alone, so no detector is involved. ``n_bands`` does not
    change it either; it is accepted so both pages can pass what they show.
    """
    import ebsd_utils as eu
    from orix.crystal_map import PhaseList

    p = phase.deepcopy()
    refl = hr.prepare_reflectors(PhaseList(p))
    rows = refl.hkl.tolist()
    budget = eu._triplet_library_budget_bytes()
    needed = _library_rows(phase, rows)
    nbytes = needed * eu._TRIPLET_ROW_BYTES
    return {
        "bytes": int(nbytes),
        "rows": int(needed),
        "fits": bool(nbytes <= budget),
        "budget_bytes": int(budget),
        "n_rows": int(len(rows)),
    }
