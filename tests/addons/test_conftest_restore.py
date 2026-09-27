"""What the package conftest's own repair does, and to whom.

These measure ``tests/addons/conftest.py`` itself. It is the file that keeps
three tests in ``test_discovery.py`` green regardless of collection order, so
when it is wrong nothing points at it — every symptom lands somewhere else.
"""
import sys
from importlib.machinery import PathFinder

import pytest

from tests.addons.conftest import (
    _require_pristine_find_distributions,
    restore_entry_point_discovery,
)


def test_the_conftest_under_test_is_the_one_pytest_loaded():
    """Guard for the import above, not ceremony.

    Everything below measures module-level state of the conftest. If
    ``tests.addons.conftest`` were a SECOND copy of the module rather than
    the instance pytest loaded, every assertion here would be about a file
    that never runs, and all of them would still pass.
    """
    import tests.addons.conftest as imported

    loaded = [m for name, m in sys.modules.items()
              if name.endswith("addons.conftest") and m is not None]
    assert loaded, sorted(n for n in sys.modules if "conftest" in n)
    assert all(m is imported for m in loaded), loaded


def test_the_restore_takes_away_only_the_backport_finder():
    """The offender goes; everyone else's finder stays.

    A wholesale ``sys.meta_path[:] = pristine`` also removes every finder a
    library registered after the snapshot, and this restore now runs before
    AND after all 86 tests of this package. Measured outside pytest, on the
    real thing rather than these stand-ins:

        pristine = list(sys.meta_path); import six
        sys.meta_path[:] = list(pristine)
        import six.moves.queue
        -> ModuleNotFoundError: No module named 'six.moves'

    six's ``_SixMetaPathImporter`` and toolz's ``TlzLoader`` are installed by
    the very chain that installs the backport finder, so they are exactly
    what a wholesale restore would take.
    """
    class _SomeOtherLibrarysFinder:
        pass

    class MetadataPathFinder:           # named as the backport's class is
        pass

    keep = _SomeOtherLibrarysFinder()
    drop = MetadataPathFinder()
    sys.meta_path.append(keep)
    sys.meta_path.append(drop)
    try:
        restore_entry_point_discovery()
        assert keep in sys.meta_path, "the restore evicted another library's finder"
        assert drop not in sys.meta_path, "the backport finder was left behind"
    finally:
        sys.meta_path[:] = [m for m in sys.meta_path if m is not keep and m is not drop]


def test_the_restore_puts_find_distributions_back():
    """The other half: this one was deleted, so it is assigned back whole."""
    saved = PathFinder.find_distributions
    del PathFinder.find_distributions
    assert not hasattr(PathFinder, "find_distributions")
    try:
        restore_entry_point_discovery()
        assert PathFinder.find_distributions is saved
    finally:
        PathFinder.find_distributions = saved


def test_a_finder_that_was_already_pristine_is_not_disarmed():
    """Name-matching must not confiscate something that was there first.

    An installation that legitimately ships a MetadataPathFinder before this
    package is imported would otherwise lose it 172 times per session.
    """
    from tests.addons import conftest

    class MetadataPathFinder:
        pass

    native = MetadataPathFinder()
    conftest._PRISTINE_META_PATH.append(native)
    sys.meta_path.append(native)
    try:
        restore_entry_point_discovery()
        assert native in sys.meta_path
    finally:
        conftest._PRISTINE_META_PATH.remove(native)
        sys.meta_path[:] = [m for m in sys.meta_path if m is not native]


def test_a_missing_pristine_snapshot_is_refused_loudly_not_ignored():
    """A guard that cannot do its job has to say so.

    ``getattr(..., None)`` stops the SAVE from raising, which is why the
    snapshot moved to module scope — but a None then makes the restore a
    silent no-op, and the three entry-point tests go red naming nothing
    (measured: 3 failed, 83 passed). The message must name the cause and the
    way out, because by the time it fires nothing else will.
    """
    with pytest.raises(RuntimeError) as exc:
        _require_pristine_find_distributions(None)

    message = str(exc.value)
    assert "find_distributions" in message
    assert "tests/addons" in message            # the way out
    assert "importlib_metadata" in message      # the cause


def test_a_live_snapshot_passes_through_unchanged():
    """The other half, so the check cannot be satisfied by always raising."""
    live = PathFinder.find_distributions
    assert _require_pristine_find_distributions(live) is live
