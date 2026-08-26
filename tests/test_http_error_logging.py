"""The HTTPException handler must log every error the user is about to see.

~543 places raise HTTPException; each returns a message to the client and,
before this handler, left no server-side trace at all. Those messages are
exactly what users screenshot.
"""

import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException


@pytest.fixture()
def client():
    """A minimal app carrying the same handler main.py installs."""
    from backend.api.main import _log_http_exception

    app = FastAPI()
    app.add_exception_handler(StarletteHTTPException, _log_http_exception)

    @app.get("/api/indexing/start")
    async def start():
        raise HTTPException(status_code=400, detail="phase_list required")

    @app.get("/api/phasemap/layer")
    async def layer():
        raise HTTPException(status_code=500, detail="renderer exploded")

    @app.get("/api/health")
    async def health():
        raise HTTPException(status_code=503, detail="not ready")

    @app.get("/api/ok")
    async def ok():
        return {"fine": True}

    return TestClient(app, raise_server_exceptions=False)


def test_client_response_is_unchanged(client, caplog):
    """Behaviour for existing callers must be byte-identical."""
    r = client.get("/api/indexing/start")
    assert r.status_code == 400
    assert r.json() == {"detail": "phase_list required"}


def test_4xx_is_logged_as_warning_with_detail(client, caplog):
    with caplog.at_level(logging.WARNING, logger="backend.api.main"):
        client.get("/api/indexing/start")
    records = [r for r in caplog.records if r.name == "backend.api.main"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    msg = records[0].getMessage()
    assert "400" in msg
    assert "/api/indexing/start" in msg
    assert "phase_list required" in msg
    assert "GET" in msg


def test_5xx_is_logged_as_error(client, caplog):
    with caplog.at_level(logging.WARNING, logger="backend.api.main"):
        client.get("/api/phasemap/layer")
    records = [r for r in caplog.records if r.name == "backend.api.main"]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert "renderer exploded" in records[0].getMessage()


def test_polling_paths_stay_quiet(client, caplog):
    """A busy backend must not fill the log with health-check failures."""
    with caplog.at_level(logging.WARNING, logger="backend.api.main"):
        r = client.get("/api/health")
    assert r.status_code == 503  # response still correct
    assert [rec for rec in caplog.records if rec.name == "backend.api.main"] == []


def test_successful_requests_log_nothing(client, caplog):
    with caplog.at_level(logging.WARNING, logger="backend.api.main"):
        assert client.get("/api/ok").status_code == 200
    assert [rec for rec in caplog.records if rec.name == "backend.api.main"] == []


def test_404_on_unknown_route_is_logged(client, caplog):
    """Starlette raises its own HTTPException for unmatched routes."""
    with caplog.at_level(logging.WARNING, logger="backend.api.main"):
        r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "backend.api.main"]
    assert any("404" in m and "/api/does-not-exist" in m for m in msgs)
