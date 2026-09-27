"""A loaded result without a usable crystal map is a 400 with its code, not a 500.

Every phase-map route guarded "nothing is loaded" and none of them guarded
"something is loaded that has no crystal map". A result object without ``.xmap``
therefore reached ``result.xmap`` and the route answered

    500  {"detail": "'types.SimpleNamespace' object has no attribute 'xmap'"}

Three things are wrong with that. The status is a server fault when the cause is
missing data; the raw exception text lands in ``detail``, which the frontend
shows verbatim in 159 places; and a 500 carries no ``CODE_HEADER``, so the UI
cannot translate it and falls back to English prose.

Measured before the fix on /phase-legend, /render, /phase-stats, /ipf-key,
/phase-adjacency and /layer: 500 on all of them.
"""
from __future__ import annotations

import types

import pytest

ROUTES = [
    "/api/phasemap/phase-legend",
    "/api/phasemap/render",
    "/api/phasemap/phase-stats",
    "/api/phasemap/ipf-key",
    "/api/phasemap/phase-adjacency",
    "/api/phasemap/layer?kind=ipf-z",
    "/api/phasemap/layer?kind=phase",
]

# The POST routes on the same page. These were missed in the first pass, which
# left the hover tooltip and the drag-ROI throwing 500 while the legend beside
# them politely asked for data -- a worse impression than before the fix. They
# take a body, so they need their own parametrisation.
POST_ROUTES = [
    ("/api/phasemap/probe", {"row": 0, "col": 0}),
    ("/api/phasemap/region-stats",
     {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1}),
    ("/api/phasemap/apply-cleanup", {"file_stem": "", "target": "result"}),
]


@pytest.fixture
def client_with_unusable_result(monkeypatch):
    """An active result that carries no crystal map.

    This is not a contrived shape: a light import whose /Indexing group is
    present but unreadable, or any future result type that fails to build its
    xmap, arrives here. The registry is restored by contents, not replaced —
    other modules hold a reference to the same dict.
    """
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("ORIENTA_NO_FILE_LOG", "1")
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.api.routes import indexing as imod

    saved, saved_active = dict(imod._result_registry), imod._active_result_id
    imod._result_registry["no-xmap"] = types.SimpleNamespace(
        metadata={}, original_shape=(3, 4), selection_mask=None,
    )
    imod._active_result_id = "no-xmap"
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        imod._result_registry.clear()
        imod._result_registry.update(saved)
        imod._active_result_id = saved_active


@pytest.mark.parametrize("route", ROUTES)
def test_route_answers_the_coded_400(route, client_with_unusable_result):
    from backend.api.problem import CODE_HEADER

    response = client_with_unusable_result.get(route)

    assert response.status_code == 400, f"{route} answered {response.status_code}"
    assert response.headers.get(CODE_HEADER.lower()) == "noResultOrDataset", (
        f"{route} gave no problem code, so the UI cannot translate it"
    )


@pytest.mark.parametrize("route,body", POST_ROUTES)
def test_the_post_routes_answer_the_coded_400_too(route, body, client_with_unusable_result):
    """Measured before this pass: probe and region-stats answered 500 with a
    generic "Internal Server Error" (they have no except-wrapper, so not even
    the AttributeError text), and apply-cleanup would have too."""
    from backend.api.problem import CODE_HEADER

    response = client_with_unusable_result.post(route, json=body)

    assert response.status_code == 400, f"{route} answered {response.status_code}"
    assert response.headers.get(CODE_HEADER.lower()) == "noResultOrDataset", (
        f"{route} gave no problem code, so the UI cannot translate it"
    )


def test_no_phasemap_route_still_dereferences_xmap_unguarded():
    """The class, not the instances.

    The first pass fixed six routes and claimed the class was closed; three
    were left. This reads the module and fails on a bare `<something>.xmap`
    outside the guarded helpers, so the next one cannot slip through green.
    """
    import re
    from pathlib import Path

    import backend.api.routes.phase_map as pm

    src = Path(pm.__file__).read_text(encoding="utf-8")
    offenders = []
    for n, line in enumerate(src.splitlines(), 1):
        if not re.search(r"\b(result|active|_last_result)\.xmap\b", line):
            continue
        # Allowed: behind an explicit `is not None` / getattr in the same
        # expression, or inside the documented fail-soft helper.
        if "getattr" in line or "is not None" in line:
            continue
        window = "\n".join(src.splitlines()[max(0, n - 12):n])
        if "_require_xmap(" in window or 'xmap", None) is not None' in window:
            continue
        offenders.append(f"{n}: {line.strip()}")

    assert offenders == [], (
        "unguarded .xmap dereference — these answer 500 for a loaded result "
        "that has no crystal map:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("route", ROUTES)
def test_the_raw_exception_text_does_not_reach_the_user(route, client_with_unusable_result):
    """`detail` is shown verbatim by the frontend — it must not carry a traceback."""
    response = client_with_unusable_result.get(route)
    detail = response.json().get("detail", "")

    assert "AttributeError" not in detail
    assert "SimpleNamespace" not in detail
    assert "has no attribute" not in detail
    assert detail.startswith("No indexing result"), detail


def test_a_usable_result_is_unaffected(monkeypatch):
    """The guard must not turn a working request into a 400.

    Uses the real CrystalMap the other phase-map tests build, so this fails if
    `_require_xmap` ever starts rejecting something valid.
    """
    pytest.importorskip("orix")
    import numpy as np
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("ORIENTA_NO_FILE_LOG", "1")
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.api.routes import indexing as imod

    n_rows, n_cols = 2, 3
    n = n_rows * n_cols
    x, y = np.meshgrid(np.arange(n_cols, dtype=float), np.arange(n_rows, dtype=float))
    xmap = CrystalMap(
        rotations=Rotation.identity((n,)),
        phase_id=np.ones(n, dtype=int),
        x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList(Phase(name="Al", space_group=225)),
        prop={"ci": np.full(n, 0.5)},
    )
    result = types.SimpleNamespace(
        xmap=xmap, original_shape=(n_rows, n_cols),
        selection_mask=np.ones((n_rows, n_cols), dtype=bool), metadata={},
    )

    saved, saved_active = dict(imod._result_registry), imod._active_result_id
    imod._result_registry["usable"] = result
    imod._active_result_id = "usable"
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/phasemap/phase-legend")
        assert response.status_code == 200, response.text
    finally:
        imod._result_registry.clear()
        imod._result_registry.update(saved)
        imod._active_result_id = saved_active
