"""The ONE evaluation of PyEBSDIndex's pattern-centre cost that Orienta uses.

PyEBSDIndex finds a pattern centre (PC) by minimising a cost: the mean misfit, over a stack of
calibration patterns, between the bands detected in each pattern and the bands a crystal of the
loaded phase(s) would show at that PC (``pyebsdindex.pcopt._optfunction``). Orienta's "Find PC"
and "Global PC Refine" both minimise that ONE joint cost over all calibration patterns; the
per-pattern values are only kept to show how far the patterns disagree.

Why this module exists and why it wraps a private function:

* There is no public function that returns the cost at a given PC. ``optimize`` and
  ``optimize_pso`` return only the minimiser (and, for the swarm, the cost at it), so the
  "how sharp is the minimum" question of the result note (a ring around the result) needs the
  cost at points they never report.
* The private function changed between the two PyEBSDIndex versions Orienta supports. In 0.3.9.1
  it returns an array of shape (1,); in 0.3.10.1 it returns a 0-d array, and it APPLIES THE
  GNOMONIC CORRECTION TO ITS ``banddat`` ARGUMENT IN PLACE, so the cost it returns for one PC
  depends on how often it was called before (measured on nine rendered patterns at one fixed PC,
  three calls without a copy: 0.12314, 0.12300, 0.12916; with a copy: 0.12314 three times). The
  swarm and the Nelder-Mead of the public functions evaluate it on one shared ``banddat``.
  Here every evaluation gets its own copy, and the result is always a Python float.
* ``tests/test_pc_cost.py`` is the contract: it runs in both environments and fails the moment
  a PyEBSDIndex release changes the shape, the repeatability or the minimum of this cost.

Nothing else in Orienta may import ``pcopt._optfunction``; ``test_only_this_module_touches_the_private_cost``
reads the source tree and fails if anything does.
"""
from __future__ import annotations

import numpy as np


def _private_cost():
    """PyEBSDIndex's cost function. The single import of the private name."""
    from pyebsdindex import pcopt
    return pcopt._optfunction


class JointPcCost:
    """The cost of a PC for a fixed stack of calibration patterns.

    ``indexer`` is the PyEBSDIndex indexer the application built for the detector and the loaded
    phases (``ebsd_utils.create_indexer``); with several phases the cost uses, per pattern, the
    phase that matches best, exactly as indexing does. ``patterns`` is ``(n, rows, cols)`` (or one
    ``(rows, cols)`` pattern). Bands are detected ONCE, at construction.

    Calling the object with one PC (shape ``(3,)``) returns a float; with several (shape
    ``(m, 3)``) an array of ``m`` floats. The PC is in the convention of ``indexer.vendor``
    (Bruker for an indexer kikuchipy built).
    """

    def __init__(self, indexer, patterns=None, *, banddat=None):
        self.indexer = indexer
        if banddat is None:
            stack = np.ascontiguousarray(np.asarray(patterns, dtype=np.float32))
            if stack.ndim == 2:
                stack = stack[None]
            if stack.ndim != 3 or stack.shape[0] < 1:
                raise ValueError(f"calibration patterns must be (n, rows, cols), got {stack.shape}")
            banddat = indexer.bandDetectPlan.find_bands(stack)
        self._banddat = banddat
        #: How many PCs were evaluated, for progress and for the tests.
        self.evaluations = 0

    @property
    def n_patterns(self) -> int:
        return int(self._banddat.shape[0])

    def subset(self, index: int) -> "JointPcCost":
        """The same cost for the single calibration pattern ``index`` (bands not detected again)."""
        i = int(index)
        if not 0 <= i < self.n_patterns:
            raise IndexError(f"pattern {i} is not among the {self.n_patterns} calibration patterns")
        return JointPcCost(self.indexer, banddat=self._banddat[i:i + 1].copy())

    def __call__(self, pc):
        arr = np.asarray(pc, dtype=np.float64)
        rows = np.atleast_2d(arr)
        if rows.ndim != 2 or rows.shape[1] != 3:
            raise ValueError(f"a pattern centre is three numbers, got shape {arr.shape}")
        cost = _private_cost()
        out = np.empty(rows.shape[0], dtype=np.float64)
        for k in range(rows.shape[0]):
            value = cost(rows[k], indexer=self.indexer, banddat=self._banddat.copy())
            out[k] = float(np.atleast_1d(value)[0])
        self.evaluations += int(rows.shape[0])
        return float(out[0]) if arr.ndim == 1 else out
