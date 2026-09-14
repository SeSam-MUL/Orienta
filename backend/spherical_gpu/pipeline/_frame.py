# -*- coding: utf-8 -*-
"""The one constant that separates the cc-volume decode from the renderer.

Measured 2026-09-09/10 (``tasks/fidelity/03_plan.md``): the SO(3) cc-volume to
Bunge-ZXZ decode inside :class:`~backend.spherical_gpu.pipeline.indexer.Tier1Indexer`
labels every peak with an orientation that sits a constant **crystal-frame
(left) 180 deg rotation about <1 -1 0>** away from the orientation Orienta's own
forward renderer -- and EMSphInx -- mean.

That operator is an element of ``O`` and of ``D4``, i.e. a *symmetry* of every
phase the indexer was ever validated against (Al, Ni, Al7Cu2Fe -- all
``z_rot == 4``), so it was invisible for four months.  It surfaces as "Orienta
returns the 90 degree m-3 coset partner of what EMSphInx returns from the same
master" on alpha-Al(Fe,Mn)Si, and as a flat 180 deg error on ``-1`` / ``-3``
masters (Fe3Al2Si3, gamma-Al3FeSi).  Closed-loop render-NCC per master, shipped
vs fixed: ``-1`` 0.033 -> 0.909, ``2/m`` 0.161 -> 0.739, ``mmm`` 0.053 -> 0.871,
``-3`` 0.017 -> 0.832, ``-6m2`` 0.097 -> 0.840, ``m-3`` 0.277 -> 0.897, and
``4/mmm`` 0.818 / ``m-3m`` 0.778 unchanged (the operator lies in those groups).

Scope
-----
Everything INSIDE the pipeline -- the cc volume, the Wigner-d chain,
``newton_refine``, ``cc_at_rotation`` -- stays in the *volume* frame.  This
correction is applied exactly where an orientation LEAVES the pipeline
(the four decodes) and inverted where one ENTERS it (the ANALYTIC_NEWTON seed).

The correction is deliberately a named constant in its own module rather than
folded into the decode formula: the fix aligns the indexer *to the renderer*,
and the absolute convention is anchored only on cubic phases where ``C2y`` is
invisible.  If the renderer's Driscoll-Healy polar axis is ever corrected
(``tasks/fidelity/02_harmonics.md`` part_f), this constant becomes ``Rz(+90)``
alone -- one line, one place.
"""
from __future__ import annotations

import math

__all__ = [
    "CRYSTAL_FRAME_FIX_QUAT",
    "apply_frame_fix_zxz",
    "invert_frame_fix_zxz",
    "decode_cells_to_zxz",
]

_SQRT_HALF = 2.0 ** -0.5

#: ``C2<1 -1 0>`` as (w, x, y, z).  Left-multiplied onto the decoded rotation.
CRYSTAL_FRAME_FIX_QUAT = (0.0, _SQRT_HALF, -_SQRT_HALF, 0.0)

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi
_THREE_HALF_PI = 1.5 * math.pi


def apply_frame_fix_zxz(phi1, Phi, phi2):
    """Left-multiply ``CRYSTAL_FRAME_FIX_QUAT`` onto a Bunge ZXZ triple.

    Works elementwise on scalars, numpy arrays and torch tensors alike (only
    ``+``, ``-`` and ``%`` are used).  ``tasks/fidelity/exp_decide_form.py``
    checks this substitution against ``Rotation.from_axes_angles([1,-1,0], 180)``
    over 4000 random triples with ``Phi`` spanning the full ``[0, 2pi)`` the
    decode can produce: max deviation 0.0 deg.

    Returns
    -------
    (phi1, Phi, phi2) : same container type as the inputs, radians.
    """
    return (
        (phi1 + _PI) % _TWO_PI,
        (_PI - Phi) % _TWO_PI,
        (_THREE_HALF_PI - phi2) % _TWO_PI,
    )


#: ``C2<1 -1 0>`` is an involution, so the inverse map is the same map.
invert_frame_fix_zxz = apply_frame_fix_zxz


def decode_cells_to_zxz(a, b, c, bandwidth):
    """cc-volume cell -> Bunge ZXZ (radians), the ONE copy of the decode.

    ``a`` / ``b`` / ``c`` are the (possibly fractional) cc-volume cell indices
    along the alpha / beta / gamma axes; scalars, numpy arrays and torch
    tensors all work (only ``*``, ``-`` and ``%`` are used).  Returns the
    triple ``(phi1, Phi, phi2)`` already carried into the renderer /
    EMSphInx crystal frame by :func:`apply_frame_fix_zxz`.

    The half turn is written in RADIANS, and that is the point
    -----------------------------------------------------------
    The alpha and gamma axes carry a half-turn offset.  Until 2026-09-10 all
    four copies of this decode expressed it as an integer number of BINS --
    ``off = size // 2`` in the indexer, ``off = L - 1`` in the refiner and the
    resolver -- but the axis length ``size = 2L - 1`` is ODD, so half a turn is
    ``size / 2`` bins (87.5 at L = 88) and the integer floor was short by
    exactly HALF A BIN on both axes: alpha came out 0.5 * 360/size deg too
    large, gamma the same amount too small.  Beta has no offset in the decode
    and showed no error, which is how the defect was localised.

    Measured (``tasks/fidelity/exp_r2_subbin.py``, closed loop against the
    forward renderer): the ground-truth cell sat at ``argmax - 0.51`` bins on
    alpha and ``- 0.49`` on gamma and at ``-0.03`` on beta.  Sweeping the
    offset as a continuous parameter (``exp_r2_offset_sweep.py`` /
    ``exp_r2_offset_bw.py``) puts the minimum at +0.5 BINS on every master
    (m-3, m-3m, mmm, 4/mmm, -3) and at every bandwidth (L = 68 / 88 / 128) --
    a fixed-angle error would have moved with L; a half-bin one does not.
    Closed-loop median disorientation to the rendered truth, before -> after:
    1.60 -> 0.18 deg (m-3, L=88), 1.55 -> 0.26 (m-3m), 1.60 -> 0.29 (mmm),
    1.60 -> 0.21 (4/mmm), 1.23 -> 0.19 (-3).

    This is also why the decode now lives in one place: the constant was
    quantised independently in two different spellings in four files.
    """
    size = 2 * int(bandwidth) - 1
    scale = _TWO_PI / size
    # Half a turn, in radians -- never as a bin count (see above).
    alpha = a * scale - _PI
    gamma = _PI - c * scale
    Phi = b * scale
    # ZYZ -> ZXZ (the two phi1 terms are the convention and cancel; kept
    # written out because they are calibration, not arithmetic).
    phi1 = (alpha + _HALF_PI) % _TWO_PI
    phi2 = (gamma - _HALF_PI) % _TWO_PI
    phi1 = (phi1 + 3.0 * _HALF_PI) % _TWO_PI
    return apply_frame_fix_zxz(phi1, Phi, phi2)
