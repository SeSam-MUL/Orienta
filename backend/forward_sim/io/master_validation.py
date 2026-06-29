"""Detect the disc-masked-Lambert corner bug in master patterns.

Background
----------
Before commit ``15af8f6`` (2026-06-22) the forward-sim master builder masked the
Roşca-Lambert SQUARE (``EMData/EBSDmaster/mLPNH``) with an *inscribed-disc* rule
``x²+y² ≤ 1``, leaving the four square corners (~21.5 % of pixels at large ``npx``,
more at small ``npx``) at exactly zero.  EMsoft's masters fill the WHOLE square —
the corner ``(1,1)`` maps to the valid equatorial direction ``(0.707, 0.707, 0)``.

Such a stale master:

* renders as a teal disc on zeroed (dark) corners in the Database Browser preview;
* produces black-hole patterns in GPU **Dictionary** indexing, which projects
  ``mLPNH`` directly — orientations whose detector projection samples the corner
  directions get zero intensity → wrong NCC → mis-indexing.

These helpers let the loaders / preview / scan FAIL LOUD on such artifacts instead
of silently corrupting indexing.  The signature is unambiguous: essentially ALL
pixels outside the inscribed disc are exactly zero while the disc interior carries
signal.  A correct full square has nonzero corners, so it is never flagged; an
all-zero/empty master is a *different* failure and is deliberately not flagged here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "lambert_disc_mask_fraction",
    "is_lambert_disc_masked",
    "h5_master_is_disc_masked",
]

# Default dataset holding the Lambert (square) north-hemisphere master.
_MLPNH = "EMData/EBSDmaster/mLPNH"


def _as_square_2d(arr) -> Optional[np.ndarray]:
    """Return a 2-D square float array, or ``None`` if the input can't be judged."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 5:
        return None
    return a


def lambert_disc_mask_fraction(mlpnh_2d) -> float:
    """Fraction of OUT-OF-DISC pixels that are exactly zero (0.0 .. 1.0).

    The inscribed disc uses EMsoft's own normalized coordinates: with
    ``npx = (M-1)/2`` and ``x = (col-npx)/npx``, ``y = (row-npx)/npx``, the disc is
    ``x²+y² ≤ 1`` and "out-of-disc" is the four corners.  ``≈1.0`` means the corners
    were zeroed (the bug); ``≈0.0`` means a correct full square.  Returns ``0.0`` for
    inputs that can't be judged (non-square / too small / no out-of-disc pixels).
    """
    a = _as_square_2d(mlpnh_2d)
    if a is None:
        return 0.0
    m = a.shape[0]
    npx = (m - 1) / 2.0
    idx = np.arange(m, dtype=np.float64)
    rr, cc = np.meshgrid(idx, idx, indexing="ij")
    x = (cc - npx) / npx
    y = (rr - npx) / npx
    outside = (x * x + y * y) > 1.0 + 1e-9
    n_out = int(outside.sum())
    if n_out == 0:
        return 0.0
    zero_out = int(np.count_nonzero(a[outside] == 0.0))
    return zero_out / n_out


def is_lambert_disc_masked(
    mlpnh_2d,
    *,
    min_outside_zero_frac: float = 0.95,
) -> bool:
    """True iff ``mlpnh_2d`` is a Lambert square with disc-masked (zeroed) corners.

    Criteria (all must hold):

    * the array is a 2-D square big enough to judge;
    * essentially ALL out-of-disc pixels are exactly zero
      (``lambert_disc_mask_fraction ≥ min_outside_zero_frac``); and
    * the disc interior carries signal (so an all-zero/empty master — a *different*
      problem — is not mislabeled as disc-masked).
    """
    a = _as_square_2d(mlpnh_2d)
    if a is None:
        return False
    if lambert_disc_mask_fraction(a) < min_outside_zero_frac:
        return False
    # Require signal in the disc interior (else it's an empty master, not this bug).
    m = a.shape[0]
    c = m // 2
    k = max(2, m // 20)
    if not np.any(a[c - k:c + k, c - k:c + k] != 0.0):
        return False
    return True


def h5_master_is_disc_masked(h5_path, dataset: str = _MLPNH) -> bool:
    """Open a master ``.h5`` and test its ``mLPNH`` for the disc-mask bug.

    Reads only a single 2-D energy slice (a few MB), not the whole 4-D array.
    Returns ``False`` for a missing/unreadable file or absent dataset (existence and
    I/O errors are the caller's concern — this answers "is it disc-masked").
    """
    import h5py  # local import: keep module import cheap

    try:
        with h5py.File(str(h5_path), "r") as f:
            if dataset not in f:
                return False
            ds = f[dataset]
            # EMsoft stores (numset, nE, M, M); kikuchipy-saved drops to (nE, M, M);
            # take the last energy of the first atom-set as a representative slice.
            if ds.ndim == 4:
                sl = ds[ds.shape[0] - 1, ds.shape[1] - 1]
            elif ds.ndim == 3:
                sl = ds[ds.shape[0] - 1]
            elif ds.ndim == 2:
                sl = ds[...]
            else:
                return False
            return is_lambert_disc_masked(np.asarray(sl, dtype=np.float32))
    except (OSError, KeyError, ValueError):
        return False
