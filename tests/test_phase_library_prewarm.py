"""The phase library index is warmed at startup, and only when there is a library.

f7 measured the cost in the user loop: `GET /api/phase-library/index` answers in
**16.4 s cold** (74 s the very first time on a machine) and 0.11 s warm. So the
person who opens the page first waits a quarter of a minute for a payload everyone
after them gets instantly — and the existing prewarm did not cover it, because
`get_index()` is most of the work but not all: the document also scans every `.h5`
under `EBSD_H5_Cache` for a master and reads all 36 CIFs.

Two properties, and the second is the one that could go wrong quietly: on a clone
with no `Database/` the prewarm must not go looking, or every developer without the
crystal library pays a startup delay to warm nothing.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_prewarm(monkeypatch, cif_dir_exists: bool):
    """Run the real prewarm with both expensive calls counted, not executed."""
    import backend.api.main as main
    from backend.api.services import crystal_hint_local_library as chl
    from backend.api.services import phase_library as pl

    calls: list[str] = []

    monkeypatch.setattr(chl, "get_index", lambda: calls.append("get_index") or {})
    monkeypatch.setattr(pl, "build_index_document",
                        lambda: calls.append("build_index_document") or
                        {"phases": []})
    # `safe_loader` is the other half of the prewarm and is not under test here.
    import safe_loader
    monkeypatch.setattr(safe_loader, "_kp", lambda: None)

    root = PROJECT_ROOT if cif_dir_exists else Path(
        str(PROJECT_ROOT) + "_definitely_not_here")
    monkeypatch.setattr(chl, "PROJECT_ROOT", root)

    asyncio.run(main._prewarm_kikuchipy_imports())
    return calls


def test_the_prewarm_builds_the_index_document_once(monkeypatch):
    """Once: warming twice would double an 8 s CIF parse for nothing."""
    if not (PROJECT_ROOT / "Database" / "CIF_Library").is_dir():
        pytest.skip("crystal library not in this checkout")
    calls = _run_prewarm(monkeypatch, cif_dir_exists=True)
    assert calls.count("get_index") == 1, calls
    assert calls.count("build_index_document") == 1, calls


def test_without_a_crystal_library_the_index_is_not_warmed(monkeypatch):
    """A clone must not spend its startup discovering there is nothing to warm.

    `get_index()` is still called -- it is cheap and correct on an empty library,
    and the Crystal Hint prewarm predates this. `build_index_document` is not.
    """
    calls = _run_prewarm(monkeypatch, cif_dir_exists=False)
    assert "build_index_document" not in calls, calls


def test_a_failing_prewarm_is_logged_and_not_raised(monkeypatch, caplog):
    """Startup must survive a library it cannot read.

    The endpoint answers 500 with a reason when asked; a prewarm that threw would
    take the whole backend down for a file permission.
    """
    import backend.api.main as main
    from backend.api.services import crystal_hint_local_library as chl
    from backend.api.services import phase_library as pl

    def _boom():
        raise OSError("library on a disconnected share")

    monkeypatch.setattr(chl, "get_index", lambda: {})
    monkeypatch.setattr(chl, "PROJECT_ROOT", PROJECT_ROOT)
    monkeypatch.setattr(pl, "build_index_document", _boom)
    import safe_loader
    monkeypatch.setattr(safe_loader, "_kp", lambda: None)

    if not (PROJECT_ROOT / "Database" / "CIF_Library").is_dir():
        pytest.skip("crystal library not in this checkout")
    asyncio.run(main._prewarm_kikuchipy_imports())      # must not raise
    assert any("phase library index" in r.getMessage().lower()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_the_prewarm_is_fire_and_forget_in_the_lifespan():
    """It must not be awaited in the startup path.

    Awaiting it would move the 16 s from the first page view to every backend
    start, which is worse: the window comes up and nothing answers.
    """
    import inspect

    import backend.api.main as main
    src = inspect.getsource(main)
    assert "asyncio.create_task(_prewarm_kikuchipy_imports())" in src
    assert "await _prewarm_kikuchipy_imports()" not in src
