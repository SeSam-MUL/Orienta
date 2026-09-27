"""The error code travels in a header, and the prose does not move.

The frontend reads `err.response.data.detail` in 159 places and shows it. A
`{code, message}` dict in `detail` would have turned every one of them into
"[object Object]" on the converted routes, so the code goes in a header
instead. These tests pin both halves of that: the body is unchanged, and the
header is there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.problem import CODE_HEADER, problem  # noqa: E402


def test_the_prose_is_still_the_whole_detail():
    """An existing reader sees exactly the string it saw before."""
    exc = problem(400, "noResultOrDataset", "No indexing result available.")
    assert exc.detail == "No indexing result available."
    assert isinstance(exc.detail, str), "a dict here breaks 159 frontend readers"
    assert exc.status_code == 400


def test_the_code_travels_in_the_header():
    exc = problem(400, "noResultOrDataset", "No indexing result available.")
    assert exc.headers[CODE_HEADER] == "noResultOrDataset"


def test_the_header_is_exposed_to_the_dev_server():
    """Without expose_headers the header is invisible to JS on port 5173.

    The packaged app is same-origin and would work either way; development
    would not, and that asymmetry is exactly the kind that ships.
    """
    main = (Path(__file__).resolve().parents[1] / "backend/api/main.py").read_text(encoding="utf-8")
    assert "expose_headers" in main, "the code header would be unreadable in dev"
    assert CODE_HEADER in main


@pytest.mark.parametrize("route", [
    "/api/phasemap/render",
    "/api/phasemap/layer?kind=ipf-z",
    "/api/phasemap/ipf-key",
    "/api/phasemap/phase-legend",
    "/api/phasemap/phase-stats",
    "/api/phasemap/phase-adjacency",
])
def test_every_converted_route_answers_with_the_code(route, monkeypatch):
    """All six, through the app, with nothing loaded.

    This replaces a test that scanned the source for `problem(` near the
    sentence. That one could be fooled by a comment saying "problem()" above a
    reverted `raise HTTPException(...)`, and it read one file. Asking the
    routes cannot be fooled by anything you write in a comment.
    """
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    from fastapi.testclient import TestClient

    from backend.api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get(route)

    assert response.status_code == 400, f"{route} answered {response.status_code}"
    assert response.headers.get(CODE_HEADER.lower()) == "noResultOrDataset", (
        f"{route} lost the code somewhere between the route and the client"
    )
    assert "No indexing result" in response.json()["detail"], (
        f"{route} changed the prose every existing reader shows"
    )


@pytest.mark.parametrize("code", ["noResultOrDataset"])
def test_the_codes_have_a_translation(code):
    import json

    root = Path(__file__).resolve().parents[1]
    locale = json.loads(
        (root / "frontend/src/locales/en/phasemap.json").read_text(encoding="utf-8")
    )
    assert code in locale.get("errors", {}), f"{code} would show the English fallback"


def test_a_real_request_carries_both(monkeypatch):
    """The whole chain, through the app, with nothing loaded.

    Unit tests on the helper prove the shape; this proves the header survives
    the middleware stack — the origin guard, the timing middleware and the
    HTTPException handler all sit between the route and the client.
    """
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    from fastapi.testclient import TestClient

    from backend.api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/phasemap/phase-legend")

    assert response.status_code == 400
    assert response.headers.get(CODE_HEADER.lower()) == "noResultOrDataset"
    # and the body is what every existing reader still expects
    assert response.json()["detail"].startswith("No indexing result")
