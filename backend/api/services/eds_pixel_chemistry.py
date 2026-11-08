"""Per-pixel EDS atomic-% for the active dataset (shared service).

Extracted verbatim from crystal_hint._get_pixel_at_pct so both the Crystal
Hint route and the Single-Pixel Phase Test route can use it. Returns None on
any failure so chemistry weighting fails soft to pattern-only ranking.

Failing soft is right here -- a missing chemistry must not break indexing --
but it used to be *silent*, and this is the path that steers phase selection.
So every path that gives up now says why: :func:`get_pixel_at_pct_with_reason`
hands the caller a :class:`ChemistryUnavailable`, and the reason is logged once
per distinct cause (once, not per pixel: the prior is evaluated for every pixel
of a map, and a systematic cause would otherwise write one line per pixel).
:func:`get_pixel_at_pct` keeps its old signature exactly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChemistryUnavailable:
    """Why there is no chemistry for this pixel.

    ``code`` is stable and machine-readable; ``detail`` is for a human.

    ``expected`` separates the two kinds of nothing. A closed file, a scan
    without EDS or a pixel outside the grid are ordinary (``expected=True``)
    and mean "no chemistry here". A refused window or a grid mismatch is a
    defect in the data or the setup (``expected=False``): the prior is off and
    somebody should know.
    """

    code: str
    detail: str
    expected: bool = True


#: Causes already logged, so a per-pixel loop does not flood the log.
_warned: set[str] = set()


def reset_warning_dedupe() -> None:
    """Forget which causes have been logged (call when a new file is opened).

    Wired into ``ebsd_viewer._load_ebsd_blocking``. "Once per cause" has to
    mean once per DATASET, or the next file's problem is swallowed by the last
    file's warning.
    """
    _warned.clear()


def warn_once(key: str, message: str, *args) -> bool:
    """Log ``message`` the first time this ``key`` comes up for this dataset.

    Shared with anything else in the EDS path that fires per pixel or per
    request and would otherwise flood the log with one identical line per
    call -- the beam-voltage fallback in the routes uses it too, so both are
    re-armed by the same :func:`reset_warning_dedupe` on file load.

    Returns True if it actually logged.
    """
    if key in _warned:
        return False
    _warned.add(key)
    logger.warning(message, *args)
    return True


def _give_up(code: str, detail: str, expected: bool = True,
             key: Optional[str] = None) -> ChemistryUnavailable:
    """The reason, logged once per ``key`` (default: the detail) if unexpected."""
    reason = ChemistryUnavailable(code, detail, expected)
    if not expected:
        if code == "spectrum_fit_running":
            warn_once(
                key or detail,
                "EDS chemistry is not available yet (%s): %s -- until then "
                "phase ranking uses pattern evidence only. (logged once)",
                code, detail)
        else:
            warn_once(
                key or detail,
                "EDS chemistry prior is OFF for this dataset (%s): %s "
                "-- phase ranking falls back to pattern evidence only. "
                "(further pixels with this cause are not logged again)",
                code, detail)
    return reason


def spectrum_fit_refusal(ext=None) -> Optional[ChemistryUnavailable]:
    """Why the active dataset has no EDS chemistry YET (or at all), else None.

    A file with a spectrum cube gets its EDS numbers from the spectrum fit
    (``eds_spectrum_service``); until that is done every map read raises
    ``SpectrumFitPending``, after a failure ``SpectrumFitFailed``. A caller
    that evaluates chemistry for many pixels asks THIS once, before the first
    pixel, and decides for the whole run: switched off pixel by pixel, an
    indexing run started right after a load would have the prior off for its
    first pixels and on for the rest.

    ``ext``: the extractor to ask; None asks the active one. Anything that is
    not a fit question (no file, a crop that does not fit) is not answered
    here -- None, and the caller reports it as it always did.
    """
    from backend.api.services.eds_spectrum_service import (
        SpectrumFitFailed, SpectrumFitPending,
    )
    if ext is None:
        if not _is_open():
            return None
        try:
            ext = _get_extractor()
        except Exception:
            return None
    check = getattr(ext, "require_fit", None)
    if not callable(check):
        return None
    try:
        check()
    except SpectrumFitPending as exc:
        return ChemistryUnavailable(
            "spectrum_fit_running",
            f"the spectra of this map are still being evaluated "
            f"({int(round(100 * exc.progress))} %); EDS chemistry is "
            f"available when that is done",
            expected=False)
    except SpectrumFitFailed as exc:
        return ChemistryUnavailable(
            "spectrum_fit_failed",
            f"the spectrum fit of this file failed ({exc.error}); EDS "
            f"chemistry is not available for it",
            expected=False)
    return None


def _is_open() -> bool:
    from backend.api.services.h5_session import is_open
    return is_open()


def _get_extractor():
    # The pixel index this module is handed is a flat index over the ACTIVE
    # dataset's grid (it is cross-checked against the EBSD nav shape below),
    # so the extractor has to be the active dataset's view. Off the crop path
    # get_active_extractor() is the raw extractor, so nothing changes there.
    from backend.api.services.h5_session import get_active_extractor
    return get_active_extractor()


def _get_active_signal():
    from backend.api.routes import ebsd_viewer as _ev
    return _ev._get_active_signal()


def chemistry_source() -> Optional[str]:
    """The EDS numbers this module's chemistry is made of.

    The prior's (``eds_indexing_prior.PRIOR_EDS_SOURCE``): the single-pixel
    phase test and the crystal hint rank candidates with the same
    ``chemistry_fit`` the prior weighs with, and that scorer got worse on the
    spectrum numbers (alpha picks 89.37 -> 74.55 % on a reference scan not included in the repository).
    ``"window"``: AZtec's raw
    window maps with the window calibration, no spectrum fit needed;
    ``None``: the open file's own source.
    """
    from backend.api.services.eds_indexing_prior import PRIOR_EDS_SOURCE
    return PRIOR_EDS_SOURCE


def chemistry_waits_for_fit() -> bool:
    """Does a pixel's chemistry wait for the spectrum fit? Not on the window
    numbers -- then no spectrum-fit reason may be reported for a null."""
    return chemistry_source() != "window"


def _window_counts(read, elements, pixel_index: int) -> Iterator[tuple]:
    """``(raw window name, one-pixel counts)`` pairs for build_counts_by_element.

    Do NOT build the ``{element: counts}`` dict here in a loop. It is keyed by
    element symbol, so two windows of one element (``Cu Kalpha`` and
    ``Cu Lalpha``) collapse into one entry that keeps the FIRST key object with
    the LAST value -- a ``Cu K`` key holding the Cu L counts, priced at
    k = 4.13 instead of 1.10. Measured 2026-09-12: a silent 3.75x, and here
    that lands straight in the phase-selection prior.
    """
    for el_name in elements:
        data = read(el_name)
        if data is None or pixel_index >= len(data):
            continue
        yield el_name, np.array([float(data[pixel_index])])


def get_pixel_at_pct_with_reason(
    pixel_index: int,
) -> Tuple[Optional[dict[str, float]], Optional[ChemistryUnavailable]]:
    """Per-pixel EDS atomic-%, plus why it is missing when it is.

    Exactly one of the two is ``None``.
    """
    from eds_utils import (UnknownLineError, build_counts_by_element,
                           counts_to_weight_pct, weight_pct_to_atomic_pct)
    try:
        if not _is_open():
            return None, _give_up("no_file", "no file is open")
        ext = _get_extractor()
        from backend.api.services.eds_spectrum_service import (
            quant_source_of, window_maps)
        if chemistry_waits_for_fit():
            # Before any per-pixel read: the same answer for every pixel
            # while the spectrum fit runs, logged once (keyed on the code --
            # the detail carries the progress and changes on every call).
            refusal = spectrum_fit_refusal(ext)
            if refusal is not None:
                return None, _give_up(refusal.code, refusal.detail,
                                      refusal.expected, key=refusal.code)
            elements, read = ext.get_available_elements(), ext.get_element_map
            source = quant_source_of(ext)
        else:
            # The window numbers (chemistry_source): AZtec's raw windows,
            # cut to the crop, priced with the window curve. No fit needed.
            elements, read = window_maps(ext)
            source = "window"
        if not elements:
            return None, _give_up("no_eds", "this scan carries no EDS windows")
        n_rows, n_cols = ext.get_grid_dimensions()
        if pixel_index < 0 or pixel_index >= n_rows * n_cols:
            return None, _give_up(
                "out_of_range",
                f"pixel {pixel_index} is outside the {n_rows}x{n_cols} EDS grid")
        try:
            _sig = _get_active_signal()
            if _sig is not None:
                _nav = _sig.axes_manager.navigation_shape
                _ebsd_total = int(np.prod([int(x) for x in _nav])) if _nav else 0
                if _ebsd_total and _ebsd_total != n_rows * n_cols:
                    return None, _give_up(
                        "grid_mismatch",
                        f"EDS grid ({n_rows * n_cols} px) does not match the "
                        f"EBSD navigation grid ({_ebsd_total} px), so the "
                        "pixels do not line up",
                        expected=False)
        except Exception:
            pass  # can't cross-check → proceed (fail-soft)

        # The source tells the quantification which response curve to use:
        # net line intensities (spectrum fit) or AZtec's window integrals.
        counts = build_counts_by_element(
            _window_counts(read, elements, pixel_index), source=source)
        if not counts:
            return None, _give_up(
                "no_counts", "no EDS window has data at this pixel")
        at = weight_pct_to_atomic_pct(counts_to_weight_pct(counts))
        # A window the quantification could not price is left out of the
        # composition and the rest renormalise without it. For a DISPLAY that
        # is the right trade -- one stray label must not cost the whole scan
        # its numbers. For the PRIOR it is not, and the difference is not a
        # matter of taste:
        #
        #   chemistry_score.score_phase_ratio reads an element that is not in
        #   the measured composition as `p_of.get(el, zero)` -- a measured
        #   ZERO. Its missing-major veto then fires on every phase whose major
        #   element is the one we dropped (`measured < _ABSENT`), so a hole in
        #   the composition does not merely weaken the ranking, it actively
        #   rules out exactly the phases that contain the missing element.
        #   "We could not measure this" and "this is not here" are opposite
        #   statements and the scorer cannot tell them apart.
        #
        # So the prior goes OFF for this dataset, as it did before the
        # per-window skip, and says why. Teaching the scorer the difference is
        # the better fix and belongs with the scorer.
        excluded = list((getattr(at, "provenance", None) or {}).get(
            "excluded_windows", []))
        if excluded:
            names = ", ".join(str(e.get("element")) for e in excluded)
            return None, _give_up(
                "excluded_window",
                f"window(s) {names} cannot be quantified, so the composition "
                f"would have a hole the phase scorer reads as a measured zero "
                f"({excluded[0].get('reason')})",
                expected=False)
        return {el: float(v[0]) for el, v in at.items()}, None
    except UnknownLineError as exc:
        # NOT ONE window in this scan could be priced -- a single unusable
        # window is skipped inside the quantification (see above). Systematic,
        # since it is a property of the file rather than of this pixel, so the
        # prior is off for the whole map, and that must not pass as "this
        # pixel happens to have no chemistry". The message names the element.
        return None, _give_up("unusable_window", str(exc), expected=False)
    except Exception as exc:
        return None, _give_up(
            "error", f"{type(exc).__name__}: {exc}", expected=False)


def get_pixel_at_pct(pixel_index: int) -> Optional[dict[str, float]]:
    """Per-pixel EDS atomic-% for the active dataset, or None.

    Unchanged contract. Use :func:`get_pixel_at_pct_with_reason` when the
    caller wants to tell "no chemistry here" from "the prior is switched off".
    """
    return get_pixel_at_pct_with_reason(pixel_index)[0]
