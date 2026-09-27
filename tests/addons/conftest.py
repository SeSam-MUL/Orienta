"""Isolation for every test in this package.

Two module-global registries and two user-directory locations are involved,
and all four bite if left alone.

* ``citations.steps.register_step`` mutates ``STEP_REGISTRY`` IN PLACE.
  ``tests/citations/test_steps.py`` pins ``len(STEP_REGISTRY) == 9`` exactly,
  and ``tests/addons/`` sorts before ``tests/citations/`` — so without this
  fixture a full run leaves ``addon.*`` keys behind and the citations suite
  fails. Restoring must ALSO be in place (clear + update): other modules hold
  a reference to the same dict object, so rebinding the module attribute
  would restore nothing for them. That asymmetry is not theoretical — it is
  why an earlier draft's idempotence test passed while measuring a dict its
  own monkeypatch had disconnected.
* the same is true of the add-on library entries registered in Task 6.
* the trust store and the user add-on directory otherwise live under
  ``Path.home()``, where the first test run leaves a file that makes the
  second run fail.
* building a real ``orix`` ``CrystalMap`` breaks entry-point discovery for
  the rest of the process. That restore lives here, and not in the one file
  that happens to build one today, because the damage is process-global: any
  later test module that constructs a ``CrystalMap`` and sorts before
  ``test_discovery.py`` brings the failure straight back. Its own docstring
  has the measurement.
"""
from __future__ import annotations

import sys
from importlib.machinery import PathFinder

import pytest

#: Entry-point discovery exactly as this package found it, captured at import
#: rather than per test. A per-test snapshot cannot be trusted: pytest imports
#: every test module before it runs any of them, so a module collected later
#: can already have done the damage ``_restore_entry_point_discovery`` exists
#: to undo — the snapshot would then record the broken state, faithfully put
#: it back, and the save itself would raise AttributeError on an attribute
#: that is already gone. Measured, in a run of
#: ``pytest tests/addons/ tests/citations/``: the import that does it at
#: collection time is ``from backend.api.main import app`` at module scope in
#: ``tests/citations/test_citations_api.py`` (NOT an orix import in those
#: tests — every orix import under tests/citations is function-local, and
#: ``backend.api.routes.indexing``, which tests/citations/conftest.py imports,
#: leaves ``find_distributions`` intact). This package's own conftest is
#: imported before any test module, so this is the last honest reading of the
#: pristine state available.
_PRISTINE_META_PATH = list(sys.meta_path)
_PRISTINE_FIND_DISTRIBUTIONS = getattr(PathFinder, "find_distributions", None)

#: The name of the ``sys.meta_path`` entry the backport installs. Matched by
#: NAME because the class is the backport's own and importing it here to
#: compare by identity would cause the very damage this file repairs.
_BACKPORT_FINDER = "MetadataPathFinder"


def _require_pristine_find_distributions(found):
    """Refuse to pretend, when the snapshot this package needs is gone.

    ``getattr(..., None)`` keeps the SAVE from raising, which is the whole
    reason the snapshot moved up here — but a None then makes the restore a
    no-op, and a guard that cannot do its job while looking like it can is
    the failure mode this repository's honesty rule exists against. Measured
    consequence of the silent form: 3 failed, 83 passed — the three
    entry-point tests in test_discovery.py, red and naming nothing.

    A tripwire, not a live bug: pytest loads an explicit argument's conftest
    BEFORE importing any test module, so even
    ``pytest tests/citations tests/addons`` still snapshots the live
    function. It fires only if that ever stops being true.
    """
    if found is None:
        raise RuntimeError(
            "tests/addons/conftest.py was imported after something had "
            "already deleted importlib.machinery.PathFinder."
            "find_distributions (importing the importlib_metadata backport "
            "does this; see _restore_entry_point_discovery). The pristine "
            "function this package restores between its tests can no longer "
            "be read, so the restore would silently protect nothing and the "
            "entry-point tests in test_discovery.py would fail naming no "
            "cause. Run the add-on tests with `tests/addons` as an explicit "
            "pytest argument, or move the earlier import out of collection."
        )
    return found


_PRISTINE_FIND_DISTRIBUTIONS = _require_pristine_find_distributions(
    _PRISTINE_FIND_DISTRIBUTIONS)


def restore_entry_point_discovery() -> None:
    """Put entry-point discovery back, taking away only what is ours.

    Two asymmetric halves, and the asymmetry is the point.

    ``find_distributions`` was DELETED, so it is assigned back outright.

    ``sys.meta_path`` only had an entry APPENDED, so only that entry is
    dropped. Reinstating ``_PRISTINE_META_PATH`` wholesale would evict every
    finder any library registered after the snapshot — measured, outside
    pytest:

        pristine = list(sys.meta_path); import six
        sys.meta_path[:] = list(pristine)
        import six.moves.queue
        -> ModuleNotFoundError: No module named 'six.moves'

    The same chain that installs the backport finder also installs six's
    ``_SixMetaPathImporter`` and toolz's ``TlzLoader``, and this runs before
    AND after all 86 tests of this package, so a wholesale restore would
    leave the rest of the session unable to import those lazily. Anything
    already in the pristine list is kept even if it matches the name, so an
    installation that legitimately ships one is not disarmed.
    """
    sys.meta_path[:] = [m for m in sys.meta_path
                        if m in _PRISTINE_META_PATH
                        or type(m).__name__ != _BACKPORT_FINDER]
    PathFinder.find_distributions = _PRISTINE_FIND_DISTRIBUTIONS


@pytest.fixture(autouse=True)
def _isolate_addon_registries():
    from backend.api.services.citations import steps as steps_mod

    saved_steps = dict(steps_mod.STEP_REGISTRY)
    render_mod = None
    saved_entries = {}
    try:
        from backend.api.services.citations import render as _render
        saved_entries = dict(_render.ADDON_LIBRARY_ENTRIES)
        render_mod = _render
    except (ImportError, AttributeError):        # before Task 6 exists
        pass
    yield
    steps_mod.STEP_REGISTRY.clear()
    steps_mod.STEP_REGISTRY.update(saved_steps)
    if render_mod is not None:
        render_mod.ADDON_LIBRARY_ENTRIES.clear()
        render_mod.ADDON_LIBRARY_ENTRIES.update(saved_entries)


@pytest.fixture(autouse=True)
def _isolate_addon_jobs():
    """Start every test with the job registry it found, and leave it that way.

    Restored IN PLACE for the same reason as the registries above: the routes
    hold a reference to the same dict object, so rebinding ``jobs._JOBS``
    would restore nothing for them. A job left behind by one test would
    otherwise be pruned, polled or counted by the next.
    """
    from backend.api.services.addons import jobs as jobs_mod

    saved = dict(jobs_mod._JOBS)
    yield
    jobs_mod._JOBS.clear()
    jobs_mod._JOBS.update(saved)


@pytest.fixture(autouse=True)
def _isolate_addon_user_state(tmp_path, monkeypatch):
    """Never read or write the developer's own ~/.orienta.

    Without this the first run of tests/addons/test_addons_api.py leaves a
    trust file behind and the SECOND run fails on
    ``test_a_new_addon_is_listed_as_not_yet_known``; and a developer with the
    example add-on installed as its README instructs fails two discovery
    tests that assume the user folder is empty.
    """
    user_dir = tmp_path / "addons-home"
    user_dir.mkdir()
    monkeypatch.setenv("ORIENTA_ADDON_USER_DIR", str(user_dir))
    monkeypatch.setenv("ORIENTA_ADDON_TRUST_FILE", str(tmp_path / "trust.json"))


@pytest.fixture(autouse=True)
def _restore_entry_point_discovery():
    """Undo a side effect of building a REAL orix CrystalMap, not orix's job.

    Measured, in two parts, because restoring only the first part turned an
    eager crash into a silent empty result — the second symptom of the same
    cause, not a fix:

    1. The first construction of a real ``CrystalMap``/``PhaseList`` in this
       process pulls in a dependency that appends a second, stricter
       ``importlib_metadata.MetadataPathFinder`` to ``sys.meta_path`` (the
       import is cached, so this fires exactly once per process).
    2. That backport, on the same import, also DELETES the stdlib
       ``importlib.machinery.PathFinder.find_distributions`` classmethod it
       considers itself authoritative over — measured directly:
       ``PathFinder.find_distributions`` raises ``AttributeError`` afterwards.

    Together they mean every later call to ``importlib.metadata.entry_points``
    in this process is answered by the backport's stricter parser instead of
    stdlib's. That parser raises eagerly on the deliberately malformed
    ``entry_points.txt`` fixtures in ``tests/addons/test_discovery.py`` if
    that module runs afterwards in the same session — turning three
    unrelated, passing tests red purely from file-collection order (and,
    with only sys.meta_path restored, silently empty instead — the stdlib
    resolver was still missing its method). Both are restored, and the
    restore does not depend on which the backport touches first.

    It lives in the package conftest, not beside the one module that builds a
    CrystalMap today, because the damage is process-global: any test module
    that builds one and sorts before ``test_discovery.py`` brings the failure
    straight back, and Task 9's example add-on will build one.
    """
    restore_entry_point_discovery()     # before, too: see _PRISTINE_* above
    yield
    restore_entry_point_discovery()
