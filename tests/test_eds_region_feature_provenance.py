"""The boundary tools must work on the data the boundaries were drawn on.

`split_region` and `snap_region_edges` re-fit inside a region using the
feature matrix stored at classification time. If element weights shaped the
grouping but not that stored matrix, the tools would adjust a boundary using
different numbers than the ones that put it there — the "state in two places,
only one maintained" failure this codebase keeps producing.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.routes.eds import _region_feature_matrix
from backend.api.services.eds_clustering import _feature_matrix, _smooth_maps

N_ROWS = N_COLS = 8


def _maps():
    n = N_ROWS * N_COLS
    return {"Al": np.full(n, 90.0), "Fe": np.linspace(1.0, 9.0, n),
            "Si": np.full(n, 5.0)}


def test_without_weights_it_is_the_plain_feature_matrix():
    at = _maps()
    got = _region_feature_matrix(at, N_ROWS, N_COLS, 0)
    _els, want = _feature_matrix(_smooth_maps(at, N_ROWS, N_COLS, 0),
                                 N_ROWS * N_COLS)
    assert np.array_equal(got, want)


def test_weights_reach_the_stored_matrix():
    at = _maps()
    plain = _region_feature_matrix(at, N_ROWS, N_COLS, 0)
    weighted = _region_feature_matrix(at, N_ROWS, N_COLS, 0, {"Fe": 5.0})
    assert not np.array_equal(plain, weighted), (
        "a split would otherwise re-fit on composition the grouping never saw")


def test_it_is_the_same_matrix_the_clustering_builds():
    """Pinned against the clustering's own helper rather than a hand-written
    expectation, so the two cannot drift apart silently."""
    at = _maps()
    w = {"Fe": 3.0}
    got = _region_feature_matrix(at, N_ROWS, N_COLS, 5, w)
    _els, want = _feature_matrix(_smooth_maps(at, N_ROWS, N_COLS, 5),
                                 N_ROWS * N_COLS, w)
    assert np.array_equal(got, want)


def test_the_smoothing_is_applied_before_the_weights():
    """Order matters: weights scale the renormalised fractions, and
    renormalising smoothed maps is not the same as smoothing renormalised
    ones. Both sides go through the same helper, so this pins the order."""
    at = _maps()
    got = _region_feature_matrix(at, N_ROWS, N_COLS, 3, {"Fe": 2.0})
    smoothed = _smooth_maps(at, N_ROWS, N_COLS, 3)
    _els, want = _feature_matrix(smoothed, N_ROWS * N_COLS, {"Fe": 2.0})
    assert np.array_equal(got, want)
