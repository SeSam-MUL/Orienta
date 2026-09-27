"""An add-on whose colour bound is NaN -- the measured escape, on the wire.

``np.float32("nan")`` and not ``float("nan")`` on purpose: the guard that
refused the Python float waved the numpy one through for three rounds. The
point of this fixture is what happens NEXT if that guard is lost. Starlette's
JSONResponse encodes with ``allow_nan=False``, so a NaN that reaches
``_serialise`` does not spoil one field -- it takes the whole response down as
an unattributed error, which is the one thing the add-on contract forbids.

The map itself is finite. An all-NaN map is legal (an unindexed pixel is
genuinely absent) and never travels as JSON; only the BOUND is the problem.
"""
import numpy as np

from backend.api.services.addons.outputs import MapOutput


def analyse(context):
    return [
        MapOutput(key="m", label="Map with a NaN colour bound",
                  values=np.zeros(context.shape, dtype=np.float64),
                  vmin=np.float32("nan"), vmax=1.0),
    ]
