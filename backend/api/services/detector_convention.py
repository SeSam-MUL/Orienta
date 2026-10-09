"""One meaning of the detector azimuthal angle on every path.

kikuchipy 0.12.0 (pyxem/kikuchipy#792) reversed what ``EBSDDetector.azimuthal``
means: a detector at ``+w`` in kikuchipy 0.11 produces the same direction
cosines as a detector at ``-w`` in kikuchipy 0.12.1 and later (measured, the two
agree to 2e-6 degrees). The 0.11 meaning is EMsoft's, and it is the one
Orienta's own GPU projection (``backend/dict_gpu/_pcadi/_projection`` and
``backend/dictionary_gpu/detector.py``) copies.

Orienta therefore keeps the EMsoft / kikuchipy 0.11 meaning everywhere. Every
``EBSDDetector`` inside Orienta, and every value a user or a file gives, is read
that way. The one place the convention matters is the moment a detector is
handed to kikuchipy for projection or simulation, because kikuchipy then reads
the angle with *its own* installed meaning. That hand-over goes through
:func:`for_kikuchipy_projection`, so the sign is decided here and nowhere else.

Which sign matches the vendors' headers (for instance EDAX
``Camera Azimuthal Angle``) is not decided here; see :func:`azimuthal_notice`.
"""
from __future__ import annotations

import copy
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Attribute set on the copy that :func:`for_kikuchipy_projection` returns on
#: kikuchipy >= 0.12.1. A converted detector run through the helper again would
#: flip back to Orienta's sign and project with the wrong angle, silently.
CONVERTED_MARK = "_orienta_kikuchipy_sign"

_warn_lock = threading.Lock()
_warned = False


class UnsupportedKikuchipyVersion(RuntimeError):
    """The installed kikuchipy projects detector geometry in a way Orienta
    does not support (0.12.0 only)."""


class DetectorAlreadyConverted(ValueError):
    """``for_kikuchipy_projection`` received a detector it had converted."""


def _warn_once(angle) -> None:
    """Log the interpretation notice the first time a non-zero angle is projected.

    Once per process, from here, so every caller (the indexing routes, the
    PC-refinement overlay, the pattern-match simulation, dictionary generation)
    is covered without a per-pattern message. The run logs keep their own,
    per-run lines.
    """
    global _warned
    if _warned:
        return
    msg = azimuthal_angle_notice(angle)
    if msg is None:
        return
    with _warn_lock:
        if _warned:
            return
        _warned = True
    logger.warning("%s", msg)


def _installed_release() -> tuple:
    """``(major, minor, micro)`` of the installed kikuchipy.

    Parsed from ``kikuchipy.__version__`` so that ``0.12.1.dev3`` and
    ``0.13.1+local`` count as 0.12.1 and 0.13.1. A separate function so tests
    can stand in another version without importing another kikuchipy.
    """
    import kikuchipy
    from packaging.version import Version

    release = Version(kikuchipy.__version__).release
    return tuple(int(v) for v in (tuple(release) + (0, 0, 0))[:3])


def for_kikuchipy_projection(detector):
    """Return ``detector`` expressed in the installed kikuchipy's convention.

    ``detector`` is read in Orienta's convention (EMsoft, kikuchipy 0.11).

    * kikuchipy < 0.12: kikuchipy's meaning is Orienta's, the detector is
      returned as it is.
    * kikuchipy >= 0.12.1: kikuchipy's meaning is the opposite, so a deep copy
      with ``azimuthal = -azimuthal`` is returned. Nothing else differs between
      the two geometries (tilt, sample tilt, projection centre, shape); the
      caller's detector is never modified.
    * kikuchipy 0.12.0: refused. 0.12.0 also projects with an un-inverted
      sample-to-detector rotation (fixed in 0.12.1, pyxem/kikuchipy#797), which
      no sign change repairs.

    Use this only where the detector goes into a kikuchipy projection or
    simulation (``get_patterns``, ``KikuchiPatternSimulator.on_detector``,
    ``refine_orientation``). Do not use it before Orienta's own projection, and
    do not use it twice on the same detector: a detector converted on
    kikuchipy >= 0.12.1 carries a mark and :class:`DetectorAlreadyConverted` is
    raised if it comes back. (Below 0.12 nothing is converted and nothing is
    marked, so the helper is idempotent there.) The first non-zero angle passed
    through logs the interpretation notice once per process, on every version.
    """
    release = _installed_release()
    if release < (0, 12, 0):
        _warn_once(getattr(detector, "azimuthal", 0.0))
        return detector
    if release == (0, 12, 0):
        raise UnsupportedKikuchipyVersion(
            "kikuchipy 0.12.0 is not supported: its detector projection "
            "geometry was corrected in 0.12.1 (pyxem/kikuchipy#797). Install "
            "the kikuchipy version listed in requirements.txt."
        )
    if getattr(detector, CONVERTED_MARK, False):
        raise DetectorAlreadyConverted(
            "This detector was already converted by for_kikuchipy_projection; "
            "converting it again would flip the azimuthal angle back. Pass the "
            "detector in Orienta's convention."
        )
    if getattr(detector, "twist", 0.0):
        # Orienta has no twist angle; one can only arrive on a detector built
        # by kikuchipy itself. It has no counterpart in the EMsoft convention
        # either, so it is carried over unchanged and not reinterpreted.
        logger.warning("Detector twist of %s deg is passed on to kikuchipy as it is.",
                       detector.twist)
    out = detector.deepcopy() if hasattr(detector, "deepcopy") else copy.deepcopy(detector)
    angle = float(detector.azimuthal)
    _warn_once(angle)
    # ``-0.0`` would end up in saved h5 headers; zero stays a plain zero.
    out.azimuthal = -angle if angle else 0.0
    setattr(out, CONVERTED_MARK, True)
    return out


def azimuthal_angle_notice(angle_deg) -> Optional[str]:
    """One line telling the user what a non-zero azimuthal angle means here.

    ``None`` for an angle of zero, where there is nothing to interpret.
    """
    try:
        value = float(angle_deg or 0.0)
    except (TypeError, ValueError):
        return None
    if value == 0.0:
        return None
    return (
        f"Detector azimuthal angle is {value:g} deg. Orienta interprets it with "
        "the EMsoft sign convention; this has not been verified against the "
        "vendor's definition of the angle. Hough and spherical indexing do not "
        "use the angle."
    )


def azimuthal_notice(detector) -> Optional[str]:
    """:func:`azimuthal_angle_notice` for a detector."""
    return azimuthal_angle_notice(getattr(detector, "azimuthal", 0.0))


def report_azimuthal(detector, progress: Optional[Callable[[str], None]] = None) -> None:
    """Say once, through the run's own log channel, that an angle is in use."""
    msg = azimuthal_notice(detector)
    if msg is None:
        return
    logger.warning("%s", msg)
    if progress is not None:
        progress(f"WARNING: {msg}")
