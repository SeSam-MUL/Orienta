"""The local backend must not act on requests a web page can forge.

Threat model (tasks/install-security-audit-2026-09-09.md, H1-H3): the backend
binds 127.0.0.1 and has no authentication, so it trusts every caller as the
user. A browser tab on any website can still reach it — a POST with only query
parameters is a CORS "simple request" that is sent without preflight, a
WebSocket has no CORS at all, and DNS rebinding makes a foreign page
same-origin. Browsers send an ``Origin`` header on every non-GET request and on
every WebSocket upgrade, and a ``Host`` header on everything; local processes
(curl, Electron's http.request, test clients) send no Origin. So:

* a mutating request or WebSocket with a NON-local Origin is refused (403),
* a request whose Host is not a local name is refused (400),
* everything without an Origin keeps working exactly as before.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route
from starlette.websockets import WebSocketDisconnect

from backend.api import security


@pytest.fixture(scope="module")
def app():
    from backend.api.main import app as _app
    return _app


@pytest.fixture(scope="module", autouse=True)
def _no_real_side_effects():
    """The route walk below sends a real request to every mutating route. With
    the guard in place none reaches a handler — but the test that reports a
    regression must not also cause it: without the guard, /api/install/wsl-install
    (all query params have defaults) would start a real WSL install. Neuter
    the handful of routes that can touch the machine."""
    from backend.api.routes import install
    mp = pytest.MonkeyPatch()
    # Stub the whole sync worker (not subprocess.Popen — subprocess.run uses
    # it internally and the app's version probe would break at import).
    mp.setattr(install, "_install_wsl_sync",
               lambda *a, **k: {"success": False, "message": "neutered in tests"})
    mp.setattr(install, "_run_elevated", lambda *a, **k: False)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


FOREIGN = "https://attacker.example"
SAFE_MUTATION = "/api/indexing/results/deactivate"  # idempotent, no data needed


def _mutating_routes(app):
    """Every (method, path) the app exposes that is not a read."""
    out = []
    for r in app.routes:
        if isinstance(r, Route):
            for m in sorted(r.methods or ()):
                if m not in ("GET", "HEAD", "OPTIONS"):
                    out.append((m, r.path))
    return out


def _concrete(path: str) -> str:
    # The guard runs before routing, so any value works for path parameters.
    import re
    return re.sub(r"\{[^}]+\}", "x", path)


# --------------------------------------------------------------------------
# HTTP: Origin
# --------------------------------------------------------------------------

def test_every_mutating_route_refuses_a_foreign_origin(app, client):
    routes = _mutating_routes(app)
    assert len(routes) > 40, "route table looks wrong — the guard would protect nothing"
    leaks = []
    for method, path in routes:
        r = client.request(method, _concrete(path), headers={"Origin": FOREIGN})
        if r.status_code != 403:
            leaks.append((method, path, r.status_code))
    assert not leaks, f"reachable from a foreign web page: {leaks}"


def test_the_wsl_install_endpoint_is_refused_before_it_validates_anything(client):
    # The audit's proof case: a query-only POST that any page can send.
    r = client.post("/api/install/wsl-install?repair=true&broken_distro=Ubuntu-22.04",
                    headers={"Origin": FOREIGN})
    assert r.status_code == 403
    assert r.json()["detail"].startswith("Refused")


@pytest.mark.parametrize("origin", [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://[::1]:8000",
])
def test_local_origins_pass(client, origin):
    r = client.post(SAFE_MUTATION, headers={"Origin": origin})
    assert r.status_code != 403


def test_no_origin_passes(client):
    """curl, scripts, Electron's shutdown request: no Origin header at all."""
    r = client.post(SAFE_MUTATION)
    assert r.status_code != 403


@pytest.mark.parametrize("origin", [
    "null",                              # redirect chains, sandboxed iframes
    "http://127.0.0.1.attacker.example", # prefix trick
    "http://localhost.attacker.example",
    "http://attacker.example",
    "file://",
])
def test_lookalike_origins_are_refused(client, origin):
    r = client.post(SAFE_MUTATION, headers={"Origin": origin})
    assert r.status_code == 403


def test_reads_are_not_gated_on_origin(client):
    """GETs never change state; CORS already stops a foreign page from
    reading the answer. Blocking them would only break <img>/<a> uses."""
    r = client.get("/api/health", headers={"Origin": FOREIGN})
    assert r.status_code == 200


def test_preflight_still_answered(client):
    """CORS preflights are OPTIONS and carry the foreign Origin by design;
    the CORS middleware, not the guard, decides them."""
    r = client.options(SAFE_MUTATION, headers={
        "Origin": FOREIGN,
        "Access-Control-Request-Method": "POST",
    })
    assert r.status_code != 403


# --------------------------------------------------------------------------
# HTTP: Host (DNS rebinding)
# --------------------------------------------------------------------------

def test_foreign_host_header_is_refused(client):
    r = client.get("/api/health", headers={"Host": "attacker.example"})
    assert r.status_code == 400


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "localhost", "[::1]:8000"])
def test_local_hosts_pass(client, host):
    r = client.get("/api/health", headers={"Host": host})
    assert r.status_code == 200


def test_allowed_hosts_can_be_extended_for_lan_use(client, monkeypatch):
    """`uvicorn --host 0.0.0.0` for a colleague on the LAN is documented; the
    operator then lists the name they connect with."""
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver,lab-pc")
    security.reset_cache()
    try:
        assert client.get("/api/health", headers={"Host": "lab-pc:8000"}).status_code == 200
        assert client.post(SAFE_MUTATION, headers={"Origin": "http://lab-pc:8000"}).status_code != 403
        assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 400
    finally:
        monkeypatch.undo()
        security.reset_cache()


def test_allowed_hosts_extends_and_never_replaces_the_local_names(monkeypatch):
    """The README says "list that machine's name". A user who lists ONLY the LAN
    name must not lock the desktop app (which loads 127.0.0.1) out of its own
    backend — the review caught exactly that: the first version replaced the
    defaults, and the only test re-listed them in the env value."""
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "lab-pc")
    security.reset_cache()
    try:
        hosts = security.allowed_hosts()
        assert "lab-pc" in hosts
        for name in security.DEFAULT_ALLOWED_HOSTS:
            assert name in hosts, f"{name} dropped by ORIENTA_ALLOWED_HOSTS"
        assert security.host_is_local("127.0.0.1:8000", hosts)
        assert security.host_is_local("localhost", hosts)
        assert security.lan_mode() is True
    finally:
        monkeypatch.undo()
        security.reset_cache()


def test_without_lan_opt_in_only_loopback_clients_are_served(monkeypatch):
    """`--host 0.0.0.0` without ORIENTA_ALLOWED_HOSTS: a LAN client can send
    `Host: 127.0.0.1` and no Origin, which the header checks cannot tell from
    curl on this machine. The client address can."""
    import asyncio

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = security.LocalOriginGuard(inner)

    async def status_for(client, env):
        if env is None:
            monkeypatch.delenv("ORIENTA_ALLOWED_HOSTS", raising=False)
        else:
            monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", env)
        security.reset_cache()
        sent = []
        scope = {"type": "http", "method": "GET", "path": "/api/health",
                 "headers": [(b"host", b"127.0.0.1:8000")], "client": client}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(msg):
            sent.append(msg)

        await guard(scope, receive, send)
        return sent[0]["status"]

    try:
        assert asyncio.run(status_for(("127.0.0.1", 40000), None)) == 200
        assert asyncio.run(status_for(("::1", 40000), None)) == 200
        assert asyncio.run(status_for(None, None)) == 200
        assert asyncio.run(status_for(("10.0.0.5", 40000), None)) == 403
        assert asyncio.run(status_for(("10.0.0.5", 40000), "lab-pc")) == 200
    finally:
        monkeypatch.undo()
        security.reset_cache()


@pytest.mark.parametrize("value", [
    "127.0.0.1:8000 evil.example",   # folded duplicate headers
    "127.0.0.1:8000, evil.example",
    "localhost;evil",
])
def test_joined_or_junk_netlocs_are_not_local(value):
    hosts = frozenset(security.DEFAULT_ALLOWED_HOSTS)
    assert security.host_is_local(value, hosts) is False
    assert security.origin_is_local("http://" + value, hosts) is False


# --------------------------------------------------------------------------
# WebSocket: Origin
# --------------------------------------------------------------------------

def test_websocket_with_foreign_origin_is_closed_before_accept(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"Origin": FOREIGN}):
            pass


def test_install_websocket_with_foreign_origin_is_closed(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/install/ws/install-emsoft",
                                      headers={"Origin": FOREIGN}):
            pass


def test_websocket_with_local_origin_works(client):
    with client.websocket_connect("/ws", headers={"Origin": "http://127.0.0.1:8000"}) as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_websocket_without_origin_works(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


# --------------------------------------------------------------------------
# The decision function on its own
# --------------------------------------------------------------------------

@pytest.mark.parametrize("origin,ok", [
    ("http://127.0.0.1:8000", True),
    ("http://localhost", True),
    ("https://localhost:5173", True),
    ("http://[::1]:8000", True),
    ("http://127.0.0.1:8000/", False),   # an Origin never has a path
    ("127.0.0.1:8000", False),           # no scheme
    ("ftp://127.0.0.1", False),
    ("http://127.0.0.1@attacker.example", False),
    ("", False),
    (None, True),                        # absent = not a browser
])
def test_origin_is_local(origin, ok):
    assert security.origin_is_local(origin, {"127.0.0.1", "localhost", "[::1]"}) is ok
