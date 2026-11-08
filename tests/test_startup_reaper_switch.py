"""The startup reaper can be switched off, and the test suite switches it off.

The reaper kills every EMsoft process it finds in WSL ("at startup nothing is
legitimately running"). On a development machine that is false: a test that
entered the app lifespan ended a simulation another session was running.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run_reaper(monkeypatch):
    import simulation.simulation_controller as sc
    from backend.api import main as backend_main

    calls = []
    monkeypatch.setattr(sc, "reap_orphaned_emsoft_processes", lambda: calls.append(1) or [])
    asyncio.run(backend_main._reap_orphaned_emsoft())
    return calls


def test_the_suite_runs_with_the_reaper_off():
    assert os.environ.get("ORIENTA_NO_STARTUP_REAPER") == "1"


def test_the_switch_keeps_the_reaper_from_running(monkeypatch):
    monkeypatch.setenv("ORIENTA_NO_STARTUP_REAPER", "1")
    assert _run_reaper(monkeypatch) == []


def test_without_the_switch_the_reaper_runs(monkeypatch):
    monkeypatch.delenv("ORIENTA_NO_STARTUP_REAPER", raising=False)
    assert _run_reaper(monkeypatch) == [1]
