"""Tests for /api/system version, frontend-error, and diagnostics export."""

import io
import json
import logging
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import system as system_routes


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(system_routes.router, prefix="/api/system")
    return TestClient(app)


def test_version_endpoint(client):
    r = client.get("/api/system/version")
    assert r.status_code == 200
    body = r.json()
    assert body["app"] == "Orienta"
    assert body["version"]
    assert "commit" in body and "branch" in body


def test_update_check_reports_availability(client, monkeypatch):
    from backend.api.services import updater

    monkeypatch.setattr(
        updater, "check_for_update",
        lambda force=False: {"available": True, "latest": "v0.3.0",
                             "current": "v0.2.0", "install_kind": "git",
                             "notes": "## v0.3.0", "reason": ""},
    )
    body = client.get("/api/system/update/check").json()
    assert body["available"] is True
    assert body["latest"] == "v0.3.0"


def test_update_check_never_propagates_an_error(client, monkeypatch):
    """An offline machine must not make the app show an error on startup."""
    from backend.api.services import updater

    def boom(force=False):
        raise RuntimeError("network down")

    monkeypatch.setattr(updater, "check_for_update", boom)
    r = client.get("/api/system/update/check")
    assert r.status_code == 200
    assert r.json()["available"] is False
    assert r.json()["reason"] == "check_failed"


def test_update_start_rejects_a_non_release_tag(client):
    r = client.post("/api/system/update/start", json={"tag": "main"})
    assert r.status_code == 400
    assert "release tag" in r.json()["detail"]


def test_update_start_rejects_a_zip_install(client, monkeypatch):
    from backend.api.services import updater

    monkeypatch.setattr(updater, "install_kind", lambda: "zip")
    r = client.post("/api/system/update/start", json={"tag": "v0.3.0"})
    assert r.status_code == 400
    assert "cannot update itself" in r.json()["detail"]


def test_update_start_rejects_a_concurrent_run(client, monkeypatch):
    from backend.api.services import updater

    monkeypatch.setattr(updater, "install_kind", lambda: "git")
    monkeypatch.setattr(updater, "start_update", lambda tag: False)
    r = client.post("/api/system/update/start", json={"tag": "v0.3.0"})
    assert r.status_code == 409


def test_update_progress_is_readable(client):
    r = client.get("/api/system/update/progress")
    assert r.status_code == 200
    assert "state" in r.json()


def test_frontend_error_is_logged(client, caplog):
    with caplog.at_level(logging.ERROR, logger="frontend"):
        r = client.post(
            "/api/system/frontend-error",
            json={
                "kind": "react-boundary",
                "message": "Cannot read properties of undefined",
                "stack": "TypeError: Cannot read properties...\n  at PhaseMapPage",
                "componentStack": "in PhaseMapPage\n  in ErrorBoundary",
                "page": "#phasemap",
                "userAgent": "TestAgent/1.0",
            },
        )
    assert r.status_code == 200
    assert r.json() == {"logged": True}
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "react-boundary" in text
    assert "Cannot read properties of undefined" in text
    assert "PhaseMapPage" in text
    assert "#phasemap" in text


def test_frontend_error_logs_the_breadcrumb_trail(client, caplog):
    """The trail is what turns a one-line report into a diagnosable one."""
    with caplog.at_level(logging.ERROR, logger="frontend"):
        client.post(
            "/api/system/frontend-error",
            json={
                "kind": "ui-error",
                "message": "Dictionary not in memory",
                "where": "IndexingPage",
                "breadcrumbs": (
                    "  -12.0s [nav] page → indexing\n"
                    "  -3.0s [http-error] POST /api/indexing/start → 500"
                ),
            },
        )
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "shown in: IndexingPage" in text
    assert "what happened before:" in text
    assert "page → indexing" in text
    assert "POST /api/indexing/start → 500" in text


def test_frontend_error_truncates_huge_payload(client, caplog):
    with caplog.at_level(logging.ERROR, logger="frontend"):
        r = client.post(
            "/api/system/frontend-error",
            json={"kind": "x" * 500, "message": "M" * 100000, "stack": "S" * 100000},
        )
    assert r.status_code == 200
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    # clipped, not the raw 100k
    assert len(text) < 30000


def test_frontend_error_tolerates_missing_fields(client):
    r = client.post("/api/system/frontend-error", json={})
    assert r.status_code == 200
    assert r.json() == {"logged": True}


def test_diagnostics_export_returns_zip_with_info(client, monkeypatch):
    r = client.get("/api/system/diagnostics/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert "orienta-diagnostics-" in r.headers["content-disposition"]

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "info.json" in names

    info = json.loads(zf.read("info.json"))
    assert info["version"]["app"] == "Orienta"
    assert info["python"]
    assert "packages" in info and "numpy" in info["packages"]
    assert "gpu" in info
    assert "loaded_files" in info and "active_file" in info


def test_diagnostics_export_includes_log_files(client, monkeypatch, tmp_path):
    log = tmp_path / "orienta.log"
    log.write_text("2026-08-26 INFO backend: something happened\n", encoding="utf-8")
    monkeypatch.setattr(
        system_routes.file_log, "get_log_paths", lambda: [log]
    )
    r = client.get("/api/system/diagnostics/export")
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert "logs/orienta.log" in zf.namelist()
    assert b"something happened" in zf.read("logs/orienta.log")


def test_export_with_context_writes_report_txt(client):
    """The user's own words plus the trail — the half no log can reconstruct."""
    r = client.post(
        "/api/system/diagnostics/export",
        json={
            "description": "Indexing stopped at 40% and the phase map stayed empty",
            "page": "#indexing",
            "breadcrumbs": "  -6.0s [nav] page → indexing\n  -0.4s [http-error] POST /start → 500",
        },
    )
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert "report.txt" in zf.namelist()
    text = zf.read("report.txt").decode("utf-8")
    assert "Indexing stopped at 40%" in text
    assert "#indexing" in text
    assert "page → indexing" in text
    assert "POST /start → 500" in text


def test_export_without_context_still_works(client):
    """GET stays valid: logs without a description are better than nothing."""
    r = client.get("/api/system/diagnostics/export")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert "info.json" in zf.namelist()
    assert "report.txt" not in zf.namelist()


def test_export_with_empty_description_is_marked_as_such(client):
    r = client.post("/api/system/diagnostics/export", json={"description": ""})
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    text = zf.read("report.txt").decode("utf-8")
    assert "(no description given)" in text


def test_export_description_also_reaches_the_log(client, caplog):
    with caplog.at_level(logging.ERROR, logger="frontend"):
        client.post(
            "/api/system/diagnostics/export",
            json={"description": "colours look wrong on the phase map"},
        )
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "problem-report" in text
    assert "colours look wrong" in text


def test_diagnostics_export_survives_broken_gpu_probe(client, monkeypatch):
    def boom():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(system_routes, "detect_gpu", boom)
    r = client.get("/api/system/diagnostics/export")
    assert r.status_code == 200
    info = json.loads(zipfile.ZipFile(io.BytesIO(r.content)).read("info.json"))
    assert info["gpu"]["available"] is False
