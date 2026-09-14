"""Measure and apply the sub-pixel offset between the EDS and EBSD maps.

EDS and EBSD are separate measurements with different source volumes; their
maps are not pixel-aligned. On the reference scan the offset is dy = -0.40 px.
Left uncorrected it systematically mislabels the outermost pixel ring of every
particle on one side.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


_MIN_STD = 1e-12
#: Smallest trimmed interior window we will correlate over, per axis. A 1x1
#: interior lets a single pixel decide the offset; below that the slice is
#: empty and the mean is nan.
_MIN_INTERIOR = 3
#: Smallest correlation curvature the parabolic vertex fit will accept. Below
#: this the peak has no shape to fit and no sub-pixel offset can be resolved.
_MIN_CURVATURE = 1e-12

#: Shifts smaller than this are treated as no shift at all, per axis.
#:
#: Not cosmetic. ``ndi.shift`` at order 1 touches four neighbours at ANY
#: non-zero offset, so it propagates a non-finite value into all four whatever
#: the magnitude: measured on a 16x16 map with a 6-px NaN block, an offset of
#: (0.003, 0.003) px already takes the non-finite count from 6 to 12, exactly as
#: (0.4, 0.4) does. A caller that excludes unmeasured pixels therefore pays the
#: full spreading cost for an offset that moves nothing -- on a synthetic
#: 301x402 map, ~16000 pixels of grain interior for a measured offset of
#: (0.003, 0.006) px.
#:
#: 0.01 px is below the accuracy ``measure_offset`` itself demonstrates: at the
#: reference scan's magnitude its worst error over five blob seeds was 0.042 px,
#: so a shift under 0.01 px is not distinguishable from zero by the function
#: that produced it. It is also ~40x below the reference scan's real -0.40 px
#: offset, which is unaffected.
_NEGLIGIBLE_SHIFT_PX = 0.01


def _validated(a: np.ndarray, name: str) -> np.ndarray:
    """Return ``a`` as float64, or raise if it cannot carry a registration signal.

    A non-finite or flat map has nothing to align on: every candidate shift
    would tie and the search would hand back its first candidate as though it
    were a measurement. This is user-reachable — the EDS map of an element that
    is absent from the sample is exactly flat — so it must fail loudly.
    """
    a = np.asarray(a, dtype=np.float64)
    bad = int(np.count_nonzero(~np.isfinite(a)))
    if bad:
        raise ValueError(
            f"{name} map has {bad} non-finite value(s) (NaN/inf); registration "
            "needs finite data")
    s = float(a.std())
    if s <= _MIN_STD:
        raise ValueError(
            f"{name} map is constant (std={s:.3g}); it carries no features to "
            "register on")
    return a


def _standardise(a: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance. Assumes ``a`` already passed ``_validated``."""
    return (a - a.mean()) / a.std()


def measure_offset(signal_map: np.ndarray, reference_map: np.ndarray,
                   max_shift: float = 2.0) -> tuple[float, float]:
    """Sub-pixel shift (dy, dx) that aligns ``signal_map`` onto ``reference_map``.

    Coarse integer search followed by a parabolic vertex fit per axis. The
    border is trimmed by ``ceil(max_shift) + 1`` so a wrapped edge cannot fake
    a correlation.

    **The reliable measuring range is about ``ceil(max_shift) - 0.5`` px, not
    ``max_shift``.** The vertex fit needs a sample either side of the coarse
    optimum, so an optimum on the edge of the searched grid is refused rather
    than extrapolated — and a true shift within ~0.5 px of that edge can already
    tie the optimum onto it. Measured at ``max_shift=2.0``: +1.4 px resolves to
    +1.398, while +1.5 px is refused. Give ``max_shift`` a pixel of headroom
    over the largest offset you expect.

    A refusal at the grid edge is *not* proof that the shift is large: an axis
    the data cannot constrain at all (identical rows, say) also ties the
    optimum onto the edge, with a true shift of zero. The message states both
    possibilities rather than picking one.

    Raises ``ValueError`` rather than returning a plausible-looking number when
    the shapes differ, when a map is too small to leave a usable interior
    window, when either map is non-finite or flat, when the coarse optimum
    lands on the edge of the searched grid, or when the correlation there is
    too flat for the sub-pixel fit to resolve.
    """
    signal_map = np.asarray(signal_map, dtype=np.float64)
    reference_map = np.asarray(reference_map, dtype=np.float64)
    if signal_map.shape != reference_map.shape:
        raise ValueError(
            f"shape mismatch: signal {signal_map.shape} vs "
            f"reference {reference_map.shape}")

    reach = int(np.ceil(max_shift))
    pad = reach + 1
    min_size = 2 * pad + _MIN_INTERIOR
    if min(signal_map.shape) < min_size:
        raise ValueError(
            f"map too small for max_shift={max_shift}: shape "
            f"{signal_map.shape} but each axis needs >= {min_size} px so the "
            f"interior left after trimming {pad} px per side is at least "
            f"{_MIN_INTERIOR}x{_MIN_INTERIOR}")

    A = _standardise(_validated(signal_map, "signal"))
    B = _standardise(_validated(reference_map, "reference"))
    sl = (slice(pad, -pad), slice(pad, -pad))

    def corr(dy: float, dx: float) -> float:
        a = ndi.shift(A, (dy, dx), order=3, mode="nearest")
        return float((a[sl] * B[sl]).mean())

    rng = range(-reach, reach + 1)
    best = max(((corr(dy, dx), dy, dx) for dy in rng for dx in rng),
               key=lambda t: t[0])
    _, by, bx = best

    # The vertex fit needs a sample either side of the optimum. On the edge of
    # the grid one side was never evaluated, the triple need not be concave,
    # and the fit extrapolates arbitrarily far (a true +5.0 px came back as
    # +10.31 px). Refuse instead. Note this does NOT establish that the true
    # shift is larger than max_shift: a true +1.5 px ties the optimum onto the
    # edge at max_shift=2.0. All we know is that it is near or past the edge.
    if abs(by) == reach or abs(bx) == reach:
        raise ValueError(
            f"cannot resolve this shift with max_shift={max_shift}: the coarse "
            f"optimum landed on the edge of the searched grid at (dy={by}, "
            f"dx={bx}), so the sub-pixel fit would have to extrapolate past the "
            f"points actually evaluated. Either the true shift is near or past "
            f"that edge, or that axis is not constrained by the data at all. "
            f"Re-run with a larger max_shift (the reliable range is about "
            f"{reach - 0.5} px); if it still refuses, the axis carries no "
            f"registrable feature.")

    def vertex(centre: float, axis: int) -> float:
        step = 1.0
        if axis == 0:
            m, c, p = corr(by - step, bx), corr(by, bx), corr(by + step, bx)
        else:
            m, c, p = corr(by, bx - step), corr(by, bx), corr(by, bx + step)
        denom = m - 2.0 * c + p
        if abs(denom) < _MIN_CURVATURE:
            # Returning ``centre`` here would silently hand back the coarse
            # argmax as though it were a sub-pixel measurement. The caller has
            # no way to tell the difference, so refuse.
            raise ValueError(
                f"correlation is flat along axis {axis} at the coarse optimum "
                f"(dy={by}, dx={bx}): curvature {denom:.3g} is below "
                f"{_MIN_CURVATURE:g}, so the sub-pixel fit has no peak to "
                "solve for. The maps share no feature that localises this axis")
        return centre + 0.5 * (m - p) / denom * step

    return float(vertex(by, 0)), float(vertex(bx, 1))


def sample_shifted(map_2d: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Resample ``map_2d`` by (dy, dx) px. Edge values are extended, not wrapped.

    An offset smaller than ``_NEGLIGIBLE_SHIFT_PX`` on BOTH axes returns a copy
    rather than resampling. See that constant: the resample is not free even
    when it moves nothing, because it spreads every non-finite value into four
    at any non-zero offset, and 0.01 px is below this module's own demonstrated
    accuracy.
    """
    if abs(dy) < _NEGLIGIBLE_SHIFT_PX and abs(dx) < _NEGLIGIBLE_SHIFT_PX:
        return np.asarray(map_2d, dtype=np.float64).copy()
    return ndi.shift(np.asarray(map_2d, dtype=np.float64), (dy, dx),
                     order=1, mode="nearest")
