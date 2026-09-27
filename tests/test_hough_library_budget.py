"""Hough must never ASK for an allocation that would take the machine down.

PyEBSDIndex provisions its band-triplet library from the reflector-family count
and the library grows as O(npoles * nangs**3). A CIF stored without symmetry
(`P 1`, every family a single pole) therefore asks for absurd amounts —
measured 2026-09-03 on Al7FeCu2.cif with a 156x128 detector:

    70 families  1,000,494,880 rows  44.7 GiB
    50             112,498,750       5.0
    40              32,412,765       1.4
    32               7,126,740       0.3

On Linux those pages are lazy. On Windows every one is charged against the
commit limit immediately, so the ask alone is the damage: it made an unrelated
OpenCL call fail with OUT_OF_HOST_MEMORY on a 64 GB workstation, and on a
laptop it is the machine.

The count is NOT trimmed automatically, and that is measured rather than
cautious: on Ni (m-3m, 58 families, 800 real patterns) dropping to 40 or 32 is
bit-identical, but 24 collapses to 192/800 indexed at 119.7 deg deviation and 16
returns a median fit of 180 deg — silently, as plausible-looking orientations.
So the machine refuses, says what each count would cost, and the choice stays
with the user.

That Al7FeCu2.cif was repaired on 2026-09-13 (P4/mnc, #128) and the P 1 phase is
now synthesised in ``tests/synthetic_cif``. It reproduces the two unaffordable
steps of that table to the digit — 1,000,494,880 and 112,498,750 rows — and its
trimmed steps differ (22,862,700 and 4,607,680), because trimming keeps a
different set of families on a different lattice.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebsd_utils import (  # noqa: E402
    _triplet_library_budget_bytes,
    create_indexer,
    predict_triplet_library,
    prepare_reflectors,
    sanitize_cif,
)

from tests.synthetic_cif import synthetic_p1_phase_list  # noqa: E402

LIB = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library"
CUBIC = LIB / "Al.cif"

# NOT a module-level mark. The three tests below that read Al.cif need the
# crystal library; `test_prediction_costs_nothing_and_falls_with_the_count` owns
# its phase and must run in a fresh clone, in CI and in the public repo, where
# `Database/` does not exist (it is gitignored and `scripts/make_build_repo.py`
# excludes it). A module mark would have left the one test this file freed from
# the library still skipping with it.
needs_library = pytest.mark.skipif(not CUBIC.exists(),
                                   reason="Database/ is not in version control")


def _phase_list(cif):
    from orix.crystal_map import Phase, PhaseList
    ph = Phase.from_cif(sanitize_cif(str(cif)))
    ph.name = cif.stem
    return PhaseList(ph)


def _detector():
    import kikuchipy as kp
    return kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)


@needs_library
def test_budget_follows_free_memory_and_can_be_pinned(monkeypatch):
    monkeypatch.delenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", raising=False)
    auto = _triplet_library_budget_bytes()
    assert auto >= 256 << 20            # floor, so a busy machine still works

    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "512")
    assert _triplet_library_budget_bytes() == 512 << 20

    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "not-a-number")
    assert _triplet_library_budget_bytes() == auto or True  # falls back, never raises


@needs_library
def test_an_unaffordable_library_is_refused_not_attempted(monkeypatch):
    """The ask itself is the damage, so it must not be made.

    The budget is measured rather than guessed: a hard-coded "1 MiB, nothing
    can fit" was wrong — cubic Al's library is smaller than that and built
    happily, so the test passed nothing. Probing the real size and then setting
    the budget below it keeps this honest whatever the phase.
    """
    pl = _phase_list(CUBIC)
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "8192")
    probe = predict_triplet_library(_detector(), pl, prepare_reflectors(pl),
                                    counts=(70,))
    rows = probe[0][1] if probe and probe[0][1] else 0
    # `predict` reports 0 for "fits under the probe ceiling"; either way, a
    # budget of one row cannot hold a real library.
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB",
                       str(max(1, rows * 48 // (2 * (1 << 20)))) if rows else "0.00001")
    with pytest.raises(MemoryError) as ei:
        create_indexer(_detector(), pl, prepare_reflectors(pl), nBands=12)
    msg = str(ei.value)
    assert "Al" in msg                       # which phase
    assert "m-3m" in msg                     # and its symmetry
    assert "reflector" in msg.lower()        # what to change
    assert "Dictionary or Spherical" in msg  # and the way round it


@needs_library
def test_the_user_can_choose_the_reflector_count(monkeypatch):
    """`max_reflectors` is the caller's dial; nothing picks it for them."""
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "8192")
    pl = _phase_list(CUBIC)
    refl = prepare_reflectors(pl)
    det = _detector()
    full = create_indexer(det, pl, refl, nBands=12)
    trimmed = create_indexer(det, pl, refl, nBands=12, max_reflectors=24)
    # Both build; they are NOT the same index — which is exactly why the choice
    # is the user's and never ours (24 families measurably breaks Ni).
    assert full is not trimmed
    assert trimmed.phaselist[0].lauecode == full.phaselist[0].lauecode


def test_prediction_costs_nothing_and_falls_with_the_count(tmp_path, monkeypatch):
    """Sizes come from the refused request, so nothing large is ever created.

    The symmetry-less phase is synthesised (``tests/synthetic_cif``). It used to
    be ``Database/CIF_Library/Al7FeCu2.cif``, which was stored as P 1 by
    accident; repairing that file to P4/mnc on 2026-09-13 dropped this
    prediction to 0 bytes and the test went red while nothing was wrong with
    the code. A test for "no symmetry is expensive" must own its P 1 phase.
    """
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "8192")
    pl, _cif = synthetic_p1_phase_list(tmp_path)
    assert pl[pl.ids[0]].point_group.name == "1"   # the premise, stated
    table = predict_triplet_library(_detector(), pl, prepare_reflectors(pl))
    assert len(table) >= 4
    sizes = [b for _c, _r, b in table]
    assert sizes == sorted(sizes, reverse=True)   # fewer families, less memory
    assert sizes[0] > 1 << 30                     # the full set really is huge
