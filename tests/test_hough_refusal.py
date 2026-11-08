"""A refused Hough library reaches the user as what it is, on both pages.

The band-triplet library of a reflector list that does not fit the memory budget is
refused before it is asked for (``create_indexer``). That used to surface as the
generic "Out of memory - the computer ran out of RAM ..." (wrong: nothing ran out,
and the advice, a smaller region or a coarser dictionary, does not help) on the
Indexing page and as an English 500 on the PC refinement page. Now it is a coded
problem (``hough_library_too_big``) with the phase, the rows, the predicted library,
the budget and, when there is one, the count of the one-click answer; the pages word
it and offer the button where the failure is shown.
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from tests.test_hough_reflector_routes import cifs, clean_registry, client  # noqa: E402,F401

MiB = 1 << 20
BUDGET = 5 * MiB          # hcp Mg: 70 rows 33.6 MiB, 50 rows 2.9 MiB, 64/60/56 rows 11 MiB


@pytest.fixture
def small_budget(monkeypatch):
    monkeypatch.setattr(eu, "_triplet_library_budget_bytes", lambda: BUDGET)


def _run(monkeypatch, cif_paths):
    """The real route and its background task, on the 3x3 nickel test signal."""
    import kikuchipy as kp
    import torch

    from backend.api.routes import ebsd_viewer
    from backend.api.routes import indexing as route

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    signal = kp.data.nickel_ebsd_small()
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: signal)
    req = route.IndexingStartRequest(method="hough", cif_paths=[str(p) for p in cif_paths])
    # asyncio.run waits for the default executor, i.e. for the background task
    task_id = asyncio.run(route.start_indexing(req))["task_id"]
    return route._indexing_tasks.pop(task_id)


def _expected_fit_rows(phase):
    """Largest ladder step whose library fits, priced one step at a time."""
    for k in eu._REFLECTOR_LADDER:
        hkl, _ = hr._rule_rows(phase, dict(hr.DEFAULT_RULE, max_rows=k))
        if eu._library_rows_for(phase, hkl.tolist(), probe_rows=1000) * eu._TRIPLET_ROW_BYTES \
                <= BUDGET:
            return k
    return None


def test_the_indexing_route_passes_the_refusal_on_as_a_coded_problem(
        monkeypatch, small_budget, cifs):
    phase = hr.phase_from_cif(cifs["Mg"])
    task = _run(monkeypatch, [cifs["Mg"]])
    assert task["status"] == "failed" and task["message"] == "Hough library too big"
    problem = task["problem"]
    assert problem["code"] == "hough_library_too_big"
    params = problem["params"]
    assert params["budget_bytes"] == BUDGET
    assert params["min_rows"] == 40 and params["min_families"] == 4
    (entry,) = params["phases"]
    assert entry["name"] == "Mg" and entry["rows"] == 70
    assert entry["predicted_bytes"] > BUDGET
    assert entry["fit_rows"] == _expected_fit_rows(phase) == 50
    assert entry["own_selection"] is False
    # not the generic out-of-RAM text: nothing ran out, and that advice does not help
    err = task["error"]
    assert "ran out of RAM" not in err and "smaller region" not in err
    assert "Reflector families" in err and "largest list that fits" in err


def test_a_phase_with_a_selection_of_its_own_is_not_offered_the_one_click(
        monkeypatch, small_budget, cifs):
    phase = hr.phase_from_cif(cifs["Mg"])
    default = [f["hkl"] for f in hr.describe(phase)["families"] if f["selected"]]
    hr.set_spec("Mg", hr.spec_with_fingerprint(phase, {"mode": "custom", "families": default}))
    task = _run(monkeypatch, [cifs["Mg"]])
    (entry,) = task["problem"]["params"]["phases"]
    assert entry["own_selection"] is True and entry["fit_rows"] is None
    assert "largest list that fits" not in task["error"]
    assert "selection of its own" in task["error"]


def test_a_multi_phase_run_carries_the_problem_with_every_phase(
        monkeypatch, small_budget, cifs):
    task = _run(monkeypatch, [cifs["Al"], cifs["Mg"]])
    assert task["problem"]["code"] == "hough_library_too_big"
    names = {p["name"]: p for p in task["problem"]["params"]["phases"]}
    assert set(names) == {"Al", "Mg"}
    assert names["Mg"]["fit_rows"] == 50 and names["Al"]["fit_rows"] is None   # Al fits


def test_a_lost_phase_of_a_multi_phase_run_keeps_the_coded_problem():
    """_run_phases_or_raise: the per-phase loop turns exceptions into strings; the
    refusal's coded problem must survive on MultiPhaseRunFailed."""
    from backend.api.routes import indexing as route
    from types import SimpleNamespace

    refusal = hr.HoughLibraryTooBig("too big", {"phases": [{"name": "B", "rows": 70}],
                                                "budget_bytes": 1})

    def run_one(pc):
        if pc.name == "B":
            raise refusal
        return SimpleNamespace(phase_name=pc.name)

    pcs = [SimpleNamespace(name="A"), SimpleNamespace(name="B")]
    with pytest.raises(route.MultiPhaseRunFailed) as e:
        route._run_phases_or_raise(pcs, run_one, lambda *a, **k: None, "Hough")
    assert e.value.problem["code"] == "hough_library_too_big"
    assert e.value.problem["params"]["phases"] == [{"name": "B", "rows": 70}]


def test_the_pc_refinement_route_answers_with_the_coded_problem(
        monkeypatch, small_budget, cifs, client):
    import kikuchipy as kp
    import torch

    from backend.api.routes import ebsd_viewer

    c, _pcr = client
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    signal = kp.data.nickel_ebsd_small()
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: signal)
    assert c.post("/api/pc/phase/add", json={"cif_path": cifs["Mg"]}).status_code == 200
    r = c.post("/api/pc/index-pattern", json={"row": 0, "col": 0})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "hough_library_too_big"
    (entry,) = detail["params"]["phases"]
    assert entry["name"] == "Mg" and entry["fit_rows"] == 50
