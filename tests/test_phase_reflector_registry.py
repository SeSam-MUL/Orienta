"""A reflector limit belongs to the PHASE, so every Hough build must honour it.

The same CIF is turned into a Hough indexer in nine places: the main run, the
pseudo-symmetry resolver that follows a spherical run, the phase check,
single-pixel pattern match, the quick test and three PC-refinement paths. Only
the first of those is reached by a parameter on the indexing request.

That mattered on 2026-09-03: a phase stored as `P 1` provisions a 44.7 GiB
band-triplet library, which Orienta now refuses. A user who lowered the count
so the run worked would have found "Check phases" and the post-spherical
resolver failing anyway, because those build their own indexer and never saw
the request. Threading an argument through eight more call chains is eight
chances to miss one, and a miss is silent — so the limit is registered against
the phase and read inside `create_indexer` itself.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils as eu  # noqa: E402

LIB = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library"
NO_SYMMETRY = LIB / "Al7FeCu2.cif"


@pytest.fixture(autouse=True)
def clean_registry():
    eu.clear_phase_reflector_limits()
    yield
    eu.clear_phase_reflector_limits()


def test_a_limit_set_by_path_is_found_by_phase_name():
    """The join that makes this work at all.

    Callers register a CIF PATH; `create_indexer` only ever sees `phase.name`,
    which the CIF-based builders set to the file's stem. If those two did not
    meet, every builder would silently fall back to "no limit".
    """
    eu.set_phase_reflector_limit(r"C:\lib\Al7FeCu2.cif", 32)
    assert eu.get_phase_reflector_limit("Al7FeCu2") == 32
    assert eu.get_phase_reflector_limit("al7fecu2") == 32          # case
    assert eu.get_phase_reflector_limit("D:/elsewhere/Al7FeCu2.cif") == 32  # other dir
    assert eu.get_phase_reflector_limit("Al") is None              # not everything


def test_none_clears_rather_than_pinning():
    eu.set_phase_reflector_limit("Al7FeCu2", 32)
    eu.set_phase_reflector_limit("Al7FeCu2", None)
    assert eu.get_phase_reflector_limit("Al7FeCu2") is None
    assert eu.phase_reflector_limits() == {}


@pytest.mark.skipif(not NO_SYMMETRY.exists(), reason="Database/ is not in version control")
def test_the_post_spherical_resolver_call_shape_honours_the_registry(monkeypatch):
    """Exactly the call `resolution.py` makes after a spherical run.

    It passes no `max_reflectors` and cannot: it is handed a CIF path and a
    detector, nothing else. Before the registry this refused; after it, the same
    unchanged call builds.
    """
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "2048")
    import kikuchipy as kp
    from orix.crystal_map import Phase, PhaseList

    phase = Phase.from_cif(eu.sanitize_cif(str(NO_SYMMETRY)))
    phase.name = NO_SYMMETRY.stem                 # what resolution.py does
    pl = PhaseList(phase)
    refl = eu.prepare_reflectors(pl)
    det = kp.detectors.EBSDDetector(shape=(156, 128), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)

    with pytest.raises(MemoryError):
        eu.create_indexer(det, pl, refl, nBands=12)

    eu.set_phase_reflector_limit(str(NO_SYMMETRY), 32)
    assert eu.create_indexer(det, pl, refl, nBands=12) is not None


@pytest.mark.skipif(not NO_SYMMETRY.exists(), reason="Database/ is not in version control")
def test_an_explicit_argument_still_wins(monkeypatch):
    """The run's own choice must override the registry, not be overridden by it.

    Otherwise a stale registry entry would quietly decide a run whose request
    said something else.
    """
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "2048")
    import kikuchipy as kp
    from orix.crystal_map import Phase, PhaseList

    phase = Phase.from_cif(eu.sanitize_cif(str(NO_SYMMETRY)))
    phase.name = NO_SYMMETRY.stem
    pl = PhaseList(phase)
    refl = eu.prepare_reflectors(pl)
    det = kp.detectors.EBSDDetector(shape=(156, 128), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)

    # Registry says 32 (affordable); the caller insists on all 70 (not) — the
    # caller's word stands and the refusal is honest.
    eu.set_phase_reflector_limit(str(NO_SYMMETRY), 32)
    with pytest.raises(MemoryError):
        eu.create_indexer(det, pl, refl, nBands=12, max_reflectors=[70])
