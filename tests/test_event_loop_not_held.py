"""Routes the page calls at start-up must not hold the event loop.

Measured on 2026-11-08 (a real page load against a fresh backend, backend/loop_watch.py):
``GET /api/indexing/files/<method>`` parsed every CIF of the library with pymatgen on the
event loop (33.8 s) and ``GET /api/simulation/system-status`` waited for ``wsl echo ok``
there (21.7 s). Every other request waited with them, the health poll timed out, and the
page showed "Backend: Disconnected" and then "Connection to the backend lost" about 35 s
after it had connected, and "Connected" again some 25 s later: the backend that "crashes
shortly after the start and comes back". The settings answered after 60 s.

Each test runs the slow part for real time (a stand-in that sleeps) and asks for
``/api/health`` while it runs: a route that holds the loop makes the health answer wait.
"""
import asyncio
import time

import httpx
import pytest

from backend.api import main as api_main

SLOW_S = 1.5
HEALTH_LIMIT_S = 0.7


async def _health_latency_while(path: str) -> tuple:
    transport = httpx.ASGITransport(app=api_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        slow = asyncio.create_task(client.get(path, timeout=30))
        # The health request is MEANT to leave 0.2 s later, while the slow route works. The
        # latency counts from that moment: a held loop delays this very sleep, so a clock
        # started after it would see a fast answer from a loop that was held all along.
        meant = time.monotonic() + 0.2
        await asyncio.sleep(0.2)
        health = await client.get("/api/health", timeout=30)
        latency = time.monotonic() - meant
        slow_response = await slow
    return latency, health.status_code, slow_response


def _sleeping(result):
    def stand_in(*args, **kwargs):
        time.sleep(SLOW_S)
        return result
    return stand_in


def test_phase_file_discovery_does_not_hold_the_loop(monkeypatch):
    import indexing_controller
    monkeypatch.setattr(indexing_controller, "discover_files_for_method",
                        _sleeping({"files": [], "groups": []}))
    latency, status, slow = asyncio.run(_health_latency_while("/api/indexing/files/hough"))
    assert slow.status_code == 200 and slow.json() == {"files": [], "groups": []}
    assert status == 200
    assert latency < HEALTH_LIMIT_S, f"/api/health waited {latency:.2f} s behind the file discovery"


@pytest.fixture
def fresh_system_status(monkeypatch):
    from backend.api.routes import simulation
    monkeypatch.setitem(simulation._system_status_cache, "value", None)
    monkeypatch.setitem(simulation._system_status_cache, "expires_at", 0)
    calls = []

    class _Status:
        wsl_installed = True

    def check():
        calls.append(time.monotonic())
        time.sleep(SLOW_S)
        return _Status()

    import simulation.system_check as system_check
    monkeypatch.setattr(system_check, "check_system_status", check)
    monkeypatch.setattr(simulation, "forward_sim_capabilities",
                        lambda: {"gpu_available": False, "numba_available": True,
                                 "emsoft_available": False})
    return calls


def test_system_status_does_not_hold_the_loop(fresh_system_status):
    latency, status, slow = asyncio.run(_health_latency_while("/api/simulation/system-status"))
    assert slow.status_code == 200 and slow.json()["wsl_installed"] is True
    assert status == 200
    assert latency < HEALTH_LIMIT_S, f"/api/health waited {latency:.2f} s behind the system check"


def test_concurrent_system_status_requests_run_the_check_once(fresh_system_status):
    """The page asks three times at start-up (Dashboard, Settings, Simulation); each one
    started its own WSL probe while the cache was empty."""
    async def three():
        transport = httpx.ASGITransport(app=api_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
            return await asyncio.gather(*(client.get("/api/simulation/system-status", timeout=30)
                                          for _ in range(3)))
    answers = asyncio.run(three())
    assert [a.status_code for a in answers] == [200, 200, 200]
    assert len(fresh_system_status) == 1


def test_forward_sim_capabilities_do_not_hold_the_loop(monkeypatch):
    from backend.api.routes import simulation
    monkeypatch.setattr(simulation, "forward_sim_capabilities",
                        _sleeping({"gpu_available": False, "numba_available": True,
                                   "emsoft_available": False}))
    latency, status, slow = asyncio.run(_health_latency_while("/api/simulation/forward-sim/capabilities"))
    assert slow.status_code == 200 and slow.json()["ours_hardware_path"] == "CPU (numba)"
    assert latency < HEALTH_LIMIT_S, f"/api/health waited {latency:.2f} s behind the capability probe"


def test_the_server_folder_check_does_not_hold_the_loop(monkeypatch, tmp_path):
    """Settings checks the database folder on the server (a network share) by counting every
    file below each standard folder; on the loop that held every request for 5.1 s here."""
    from pathlib import Path
    from backend.api.routes import settings as settings_routes
    for sub in settings_routes.SERVER_SUBDIRS.values():
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    real_rglob = Path.rglob

    def slow_rglob(self, pattern):
        time.sleep(SLOW_S / len(settings_routes.SERVER_SUBDIRS))
        return real_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", slow_rglob)

    async def run():
        transport = httpx.ASGITransport(app=api_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
            slow = asyncio.create_task(client.post(
                "/api/settings/server/test", json={"database_root": str(tmp_path)}, timeout=30))
            meant = time.monotonic() + 0.2
            await asyncio.sleep(0.2)
            health = await client.get("/api/health", timeout=30)
            return time.monotonic() - meant, health.status_code, await slow
    latency, status, slow = asyncio.run(run())
    assert slow.status_code == 200 and slow.json()["reachable"] is True
    assert latency < HEALTH_LIMIT_S, f"/api/health waited {latency:.2f} s behind the server folder check"
