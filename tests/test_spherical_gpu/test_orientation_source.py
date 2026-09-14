"""The "Orientation from Hough" badge must say what the map actually kept.

Background (2026-09-10)
-----------------------
``indexing_controller.spherical_gpu_index_patterns`` set
``metadata["orientation_source"] = "hough"`` the moment the pseudo-symmetry
resolver substituted ANYTHING, and never revisited it.  Three things have made
that a lie since:

* the resolver only touches ``z_rot == 2`` phases -- on a multi-phase map every
  other phase kept its spherical orientation all along;
* pixels where Hough failed (``n_fallback``) are handed back to the sphere
  inside ``resolution.resolve_map``;
* since the render arbitration (``5bbc2df5`` / ``e605d0f8``) and the decode fix
  (``a4b7710c``) the arbiter renders Hough against the raw spherical answer on
  every disputed pixel and keeps whichever fits -- on crop1 that was **all** of
  them (676 of 676 across five grains).

So the app showed "Orientierung aus Hough" over a map whose orientations were
the sphere's.  ``classify_orientation_source`` counts what survived instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.spherical_gpu.pipeline.variant_unification import (
    classify_orientation_source,
)


RAW = np.array([
    [0.10, 0.90, 1.30],
    [2.00, 0.40, 5.10],
    [1.10, 1.90, 0.30],
    [4.00, 2.40, 3.10],
])


def test_nothing_was_substituted_is_spherical():
    src, n_h, n_s = classify_orientation_source(RAW.copy(), RAW)
    assert (src, n_h, n_s) == ("spherical", 0, 4)


def test_everything_was_substituted_is_hough():
    final = RAW + 0.3
    src, n_h, n_s = classify_orientation_source(final, RAW)
    assert (src, n_h, n_s) == ("hough", 4, 0)


def test_a_partly_substituted_map_is_mixed_and_counted():
    """The real shape of a run: one low-symmetry phase resolved, the rest --
    other phases, Hough failures, arbitration restorations -- kept spherical."""
    final = RAW.copy()
    final[1] += 0.3
    src, n_h, n_s = classify_orientation_source(final, RAW)
    assert src == "mixed"
    assert (n_h, n_s) == (1, 3)


def test_the_arbitration_giving_every_disputed_pixel_back_is_not_hough():
    """The crop1 case that made the badge wrong: the resolver substituted, then
    the render arbitration restored the raw spherical orientation on every
    pixel it had touched.  Nothing of Hough's survives, so nothing may be
    reported as Hough's."""
    substituted = RAW + 0.3          # what the resolver produced
    restored = RAW.copy()            # what the arbitration handed back
    assert classify_orientation_source(substituted, RAW)[0] == "hough"
    assert classify_orientation_source(restored, RAW)[0] == "spherical"


def test_a_quaternion_round_trip_still_counts_as_spherical():
    """Hough-failure pixels come back through ``Rotation.from_euler`` and out
    again, so their Euler numbers are NOT bit-identical to the raw ones.  A
    plain ``==`` would have mislabelled exactly the pixels the resolver itself
    calls "kept spherical"."""
    from orix.quaternion import Rotation
    q = np.asarray(Rotation.from_euler(RAW).data).reshape(-1, 4)
    # Round trip + a deliberate sign flip: -q is the same rotation.
    round_tripped = np.asarray(Rotation(-q).to_euler(), dtype=np.float64)
    assert not np.array_equal(round_tripped, RAW), (
        "fixture is void: the round trip has to perturb the numbers")
    src, n_h, n_s = classify_orientation_source(round_tripped, RAW)
    assert (src, n_h, n_s) == ("spherical", 0, 4)


def test_an_empty_or_mismatched_map_does_not_invent_an_answer():
    assert classify_orientation_source(RAW[:0], RAW[:0]) == ("spherical", 0, 0)
    with pytest.raises(ValueError):
        classify_orientation_source(RAW[:2], RAW)
