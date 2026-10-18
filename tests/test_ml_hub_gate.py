"""The ML Hub is internal: every /api/ml route except /status is 404 unless
ORIENTA_EXPERIMENTAL_ML=1 was set when the request arrived.

/status always answers 200 (a 404 would be logged by the browser as a console
error on every start): ``{"enabled": false}`` and nothing else when the switch
is off, the module details plus ``"enabled": true`` when it is on. The frontend
shows the ML Hub tile and sidebar entry only on ``enabled === true``, so a route
that answered while the switch was off would put the page back in front of
users. The variable is read per request, which is what lets these tests flip it
without re-importing the app.
"""
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import ml_hub

client = TestClient(app)


def _ml_routes():
    """(method, path) for every route the ML router registers."""
    out = []
    for route in ml_hub.router.routes:
        for method in route.methods - {"HEAD", "OPTIONS"}:
            out.append((method, "/api/ml" + route.path.replace("{task_id}", "x")))
    return out


def test_router_has_routes_to_gate():
    # Anchor: the parametrised test below is vacuous if the list is empty.
    assert len(_ml_routes()) >= 6


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes"])
def test_status_is_200_enabled_false_unless_switch_is_exactly_one(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ORIENTA_EXPERIMENTAL_ML", raising=False)
    else:
        monkeypatch.setenv("ORIENTA_EXPERIMENTAL_ML", value)
    resp = client.get("/api/ml/status")
    assert resp.status_code == 200
    # Nothing but the flag: no module details leak while the switch is off.
    assert resp.json() == {"enabled": False}


def test_every_other_ml_route_is_404_without_switch(monkeypatch):
    monkeypatch.delenv("ORIENTA_EXPERIMENTAL_ML", raising=False)
    gated = [(m, p) for m, p in _ml_routes() if p != "/api/ml/status"]
    assert len(gated) >= 5  # anchor: not vacuous
    for method, path in gated:
        # An empty JSON body would be a 422 on the POST routes if the gate
        # did not run first; 404 proves the gate is ahead of validation.
        resp = client.request(method, path, json={})
        assert resp.status_code == 404, (method, path, resp.status_code)


def test_post_route_is_404_without_switch(monkeypatch):
    monkeypatch.delenv("ORIENTA_EXPERIMENTAL_ML", raising=False)
    assert client.post("/api/ml/store/clear").status_code == 404


def test_status_is_200_with_switch(monkeypatch):
    monkeypatch.setenv("ORIENTA_EXPERIMENTAL_ML", "1")
    resp = client.get("/api/ml/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert "available" in body


def test_switch_is_read_per_request(monkeypatch):
    monkeypatch.setenv("ORIENTA_EXPERIMENTAL_ML", "1")
    assert client.get("/api/ml/status").json()["enabled"] is True
    assert client.get("/api/ml/models").status_code == 200
    monkeypatch.delenv("ORIENTA_EXPERIMENTAL_ML")
    assert client.get("/api/ml/status").json() == {"enabled": False}
    assert client.get("/api/ml/models").status_code == 404
