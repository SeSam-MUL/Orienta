"""The reflector-family routes of both pages, and what a change does downstream.

The Indexing page names a phase by CIF path (``/api/indexing/hough/reflectors``),
the PC Refinement page by the phase it has loaded (``/api/pc/phase/reflectors``).
Both read and write ONE registry, and every Hough build of the phase follows it:
the indexing request, the PC page's indexer and the Kikuchi lines it draws.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from tests.test_hough_reflectors import AL_CIF, MG_CIF, NI_CIF  # noqa: E402


@pytest.fixture(autouse=True)
def clean_registry():
    hr.clear_specs()
    eu.clear_phase_reflector_limits()
    yield
    hr.clear_specs()
    eu.clear_phase_reflector_limits()


@pytest.fixture
def cifs(tmp_path):
    out = {}
    for name, text in (("Al", AL_CIF), ("Ni", NI_CIF), ("Mg", MG_CIF)):
        p = tmp_path / f"{name}.cif"
        p.write_text(text, encoding="utf-8")
        out[name] = str(p)
    return out


@pytest.fixture
def client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import backend.api.routes.pcrefinement as pcr
    from backend.api.routes import hough_reflectors as routes
    monkeypatch.setattr(pcr, "_sessions", {})
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/indexing/hough")
    app.include_router(pcr.router, prefix="/api/pc")
    return TestClient(app), pcr


# ---------------------------------------------------------------------------
# Indexing page: by path
# ---------------------------------------------------------------------------

def test_get_returns_the_family_table(client, cifs):
    c, _ = client
    r = c.get("/api/indexing/hough/reflectors", params={"cif_path": cifs["Al"]})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "default" and body["spec"] is None
    assert body["phase_name"] == "Al" and body["space_group"] == "Fm-3m"
    assert [f["label"] for f in body["families"][:4]] == ["{111}", "{200}", "{220}", "{311}"]
    f111 = body["families"][0]
    assert set(f111) >= {"hkl", "hkl4", "label", "d", "f", "rel_f", "mult", "selected",
                         "effective", "parallel_with", "dropped_by"}
    assert body["n_selected"] == 6 and body["n_effective"] == 4


def test_the_stored_choices_can_be_read_without_computing_a_table(client, cifs):
    c, _ = client
    assert c.get("/api/indexing/hough/reflector-specs").json()["specs"] == {}
    c.put("/api/indexing/hough/reflectors",
          json={"cif_path": cifs["Al"],
                "spec": {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}})
    body = c.get("/api/indexing/hough/reflector-specs").json()
    assert list(body["specs"]) == ["al"] and body["specs"]["al"]["mode"] == "custom"
    assert body["version"] == hr.registry_version()


def test_the_weaker_families_are_listed_only_when_asked(client, cifs):
    c, _ = client
    base = c.get("/api/indexing/hough/reflectors", params={"cif_path": cifs["Al"]}).json()
    ext = c.get("/api/indexing/hough/reflectors",
                params={"cif_path": cifs["Al"], "extended": True}).json()
    assert base["extended"] is False and ext["extended"] is True
    assert ext["n_total"] > base["n_total"]
    put = c.put("/api/indexing/hough/reflectors",
                json={"cif_path": cifs["Al"], "extended": True,
                      "spec": {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}}).json()
    assert put["extended"] is True and put["n_total"] == ext["n_total"]
    pc = c.post("/api/pc/phase/add", json={"cif_path": cifs["Al"]})
    assert pc.status_code == 200
    assert c.get("/api/pc/phase/reflectors", params={"extended": True}).json()["extended"] is True


def test_a_slow_table_does_not_stall_the_event_loop(cifs, monkeypatch):
    """A big cell takes a minute to tabulate. Everything else the app does (the
    health poll, progress polling) must keep answering meanwhile: the work runs in
    a worker thread, never on the loop."""
    import asyncio
    import time

    import httpx
    from fastapi import FastAPI
    from backend.api.routes import hough_reflectors as routes
    from backend.api.services import hough_reflector_service as svc

    def slow_table(phase, key, extended=False):
        time.sleep(1.5)                    # stands in for tens of seconds of numpy
        return {"slow": True}

    monkeypatch.setattr(svc, "table", slow_table)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/indexing/hough")

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        gaps = []

        async def ticker(stop):
            # what the health poll experiences: how late does a 50 ms timer fire?
            while not stop.is_set():
                t0 = time.perf_counter()
                await asyncio.sleep(0.05)
                gaps.append(time.perf_counter() - t0 - 0.05)

        async with httpx.AsyncClient(transport=transport, base_url="http://t") as cl:
            stop = asyncio.Event()
            tick = asyncio.create_task(ticker(stop))
            await asyncio.sleep(0.1)
            r = await cl.get("/api/indexing/hough/reflectors", params={"cif_path": cifs["Al"]})
            polled = await cl.get("/api/indexing/hough/reflector-specs")
            stop.set()
            await tick
            assert r.json() == {"slow": True} and polled.status_code == 200
            assert len(gaps) > 10           # the loop really kept running during the 1.5 s
            return max(gaps)

    assert asyncio.run(scenario()) < 0.5


def test_put_changes_the_choice_and_every_build_follows(client, cifs):
    c, _ = client
    r = c.put("/api/indexing/hough/reflectors",
              json={"cif_path": cifs["Al"],
                    "spec": {"mode": "custom", "families": [[2, 0, 0], [2, 2, 0], [3, 1, 1]]}})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "custom" and body["n_effective"] == 3
    assert body["spec"]["phase"]["space_group"] == 225           # stamped with the crystal
    assert hr.get_spec("Al")["families"] == [[2, 0, 0], [2, 2, 0], [3, 1, 1]]

    # a build that never saw the request (resolver, phase check, PC page) follows
    from orix.crystal_map import PhaseList
    ph = hr.phase_from_cif(cifs["Al"])
    det = SimpleNamespace(calls=[], get_indexer=lambda pl, refl, **kw: det.calls.append(refl) or 1)
    eu.create_indexer(det, PhaseList(ph), eu.prepare_reflectors(PhaseList(ph)))
    assert len(det.calls[0]) == 6 + 12 + 24

    # and back to the default
    r = c.put("/api/indexing/hough/reflectors", json={"cif_path": cifs["Al"], "spec": None})
    assert r.status_code == 200 and r.json()["mode"] == "default"
    assert hr.get_spec("Al") is None


def test_a_bad_choice_is_refused_and_the_old_one_stays(client, cifs):
    c, _ = client
    good = {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}
    c.put("/api/indexing/hough/reflectors", json={"cif_path": cifs["Al"], "spec": good})
    for bad, code in (
        ({"mode": "custom", "families": [[1, 0, 0], [2, 0, 0]]}, "forbidden"),
        ({"mode": "custom", "families": [[1, 1, 1]]}, "too_few"),
        ({"mode": "custom", "families": [[1, 1, 1], [2, 2, 2]]}, "too_few"),   # {222} is parallel to {111}
        ({"mode": "top_n", "n": 1}, "too_few"),
        ({"mode": "custom", "families": []}, "empty_spec"),
        ({"mode": "custom", "families": [[0, 0, 0]]}, "zero_vector"),
        ({"mode": "banana"}, "bad_spec"),
    ):
        r = c.put("/api/indexing/hough/reflectors", json={"cif_path": cifs["Al"], "spec": bad})
        assert r.status_code == 400, bad
        d = r.json()["detail"]
        assert d["code"] == code and d["message"] and isinstance(d["params"], dict)
    assert hr.get_spec("Al")["families"] == [[1, 1, 1], [2, 0, 0]]


def test_a_selection_pyebsdindex_cannot_build_is_refused_when_it_is_made(client, cifs):
    """Mg: {0002} with {2-1-11} ends in an IndexError inside PyEBSDIndex's
    library builder. The page hears that at the click, not when a run starts."""
    c, _ = client
    r = c.put("/api/indexing/hough/reflectors",
              json={"cif_path": cifs["Mg"],
                    "spec": {"mode": "custom", "families": [[0, 0, 2], [2, -1, 1]]}})
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "library_failed" and "IndexError" in d["params"]["reason"]
    assert hr.get_spec("Mg") is None
    ok = c.put("/api/indexing/hough/reflectors",
               json={"cif_path": cifs["Mg"],
                     "spec": {"mode": "custom", "families": [[0, 0, 2], [2, -1, 0]]}})
    assert ok.status_code == 200


def test_top_n_is_expanded_to_the_strongest_kept_families(client, cifs):
    c, _ = client
    r = c.put("/api/indexing/hough/reflectors",
              json={"cif_path": cifs["Al"], "spec": {"mode": "top_n", "n": 2}})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "custom"
    assert {f["label"] for f in body["families"] if f["selected"]} == {"{111}", "{200}"}


def test_an_auto_rule_changes_the_default_construction(client, cifs):
    c, _ = client
    r = c.put("/api/indexing/hough/reflectors",
              json={"cif_path": cifs["Al"], "spec": {"mode": "auto", "rule": {"f_threshold": 0.5}}})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "auto" and body["rule"]["f_threshold"] == 0.5
    assert body["n_selected"] < 6
    # the table now also offers what the old rule left out
    assert body["default_rule"]["f_threshold"] == 0.1


def test_validate_names_the_family_or_the_reason(client, cifs):
    c, _ = client
    ok = c.post("/api/indexing/hough/reflectors/validate",
                json={"cif_path": cifs["Al"], "hkl": [4, 2, 2]})
    assert ok.status_code == 200
    assert ok.json()["label"] == "{422}" and ok.json()["mult"] == 24

    forbidden = c.post("/api/indexing/hough/reflectors/validate",
                       json={"cif_path": cifs["Al"], "hkl": "1 0 0"})
    assert forbidden.status_code == 400 and forbidden.json()["detail"]["code"] == "forbidden"

    dup = c.post("/api/indexing/hough/reflectors/validate",
                 json={"cif_path": cifs["Al"], "hkl": [-1, 1, 1],
                       "spec": {"mode": "custom", "families": [[1, 1, 1]]}})
    assert dup.status_code == 400 and dup.json()["detail"]["code"] == "duplicate"
    assert dup.json()["detail"]["params"]["label"] == "{111}"

    par = c.post("/api/indexing/hough/reflectors/validate",
                 json={"cif_path": cifs["Al"], "hkl": [4, 0, 0],
                       "spec": {"mode": "custom", "families": [[2, 0, 0]]}})
    assert par.status_code == 200 and par.json()["parallel_with"] == ["{200}"]


def test_validate_converts_four_index_input_on_a_hexagonal_phase(client, cifs):
    c, _ = client
    r = c.post("/api/indexing/hough/reflectors/validate",
               json={"cif_path": cifs["Mg"], "hkl": [1, 1, -2, 0]})
    assert r.status_code == 200
    # the same family as {2-1-10}: it is shown under one representative
    assert r.json()["hkl"] == [2, -1, 0] and r.json()["label"] == "{2-1-10}"
    g = c.get("/api/indexing/hough/reflectors", params={"cif_path": cifs["Mg"]}).json()
    assert g["hexagonal"] and g["families"][0]["hkl4"] is not None


def test_a_missing_or_unreadable_cif_is_a_coded_error(client, tmp_path):
    c, _ = client
    r = c.get("/api/indexing/hough/reflectors", params={"cif_path": str(tmp_path / "no.cif")})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "not_found"
    junk = tmp_path / "junk.cif"
    junk.write_text("this is not a cif", encoding="utf-8")
    r = c.get("/api/indexing/hough/reflectors", params={"cif_path": str(junk)})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "unreadable"


def test_a_stored_choice_that_no_longer_fits_is_shown_not_applied(client, cifs, tmp_path):
    c, _ = client
    c.put("/api/indexing/hough/reflectors",
          json={"cif_path": cifs["Al"],
                "spec": {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}})
    other = tmp_path / "other" / "Al.cif"                  # same stem, other cell
    other.parent.mkdir()
    other.write_text(AL_CIF.replace("4.0495", "4.2"), encoding="utf-8")
    r = c.get("/api/indexing/hough/reflectors", params={"cif_path": str(other)})
    assert r.status_code == 200
    body = r.json()
    assert body["spec_error"]["code"] == "stale_spec"
    assert body["mode"] == "custom"                         # what is stored, so it can be reset
    assert body["spec"]["families"] == [[1, 1, 1], [2, 0, 0]]
    # ... and the page can reset it
    r = c.put("/api/indexing/hough/reflectors", json={"cif_path": str(other), "spec": None})
    assert r.status_code == 200 and "spec_error" not in r.json()


def test_cost_reports_a_size_and_follows_the_choice(client, cifs):
    c, _ = client
    base = c.get("/api/indexing/hough/reflectors/cost", params={"cif_path": cifs["Al"]})
    assert base.status_code == 200
    body = base.json()
    assert set(body) >= {"bytes", "rows", "fits", "budget_bytes", "n_rows"}
    assert body["n_rows"] == 64
    c.put("/api/indexing/hough/reflectors",
          json={"cif_path": cifs["Al"],
                "spec": {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}})
    small = c.get("/api/indexing/hough/reflectors/cost", params={"cif_path": cifs["Al"]}).json()
    assert small["n_rows"] == 14


def test_cost_of_a_phase_that_does_not_fit_and_of_a_selection_that_does(client, tmp_path,
                                                                   monkeypatch):
    """The symmetry-less test phase asks 44.7 GiB for its default list (measured,
    see tests/synthetic_cif); a handful of families is small."""
    from tests.synthetic_cif import SYNTHETIC_P1_CIF
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "2048")
    c, _ = client
    p1 = tmp_path / "synthetic_P1.cif"
    p1.write_text(SYNTHETIC_P1_CIF, encoding="utf-8")
    big = c.get("/api/indexing/hough/reflectors/cost", params={"cif_path": str(p1)}).json()
    assert big["fits"] is False and big["bytes"] > 40 * 1024 ** 3 and big["n_rows"] == 70
    table = c.get("/api/indexing/hough/reflectors", params={"cif_path": str(p1)}).json()
    few = [f["hkl"] for f in table["families"] if f["selected"]][:6]
    put = c.put("/api/indexing/hough/reflectors",
                json={"cif_path": str(p1), "spec": {"mode": "custom", "families": few}})
    assert put.status_code == 200, put.text
    small = c.get("/api/indexing/hough/reflectors/cost", params={"cif_path": str(p1)}).json()
    assert small["fits"] is True and small["bytes"] < 2 * 1024 ** 3


def test_checking_a_selection_does_not_build_a_whole_indexer(client, cifs, monkeypatch):
    """A click checks the selection and asks its cost. Building an EBSDIndexer
    for that opens an OpenCL context on a machine with a GPU, and after a few
    dozen clicks the next real indexing failed with "Context failed:
    OUT_OF_HOST_MEMORY" (measured). Only the band-triplet library is built."""
    import kikuchipy as kp

    def boom(*a, **k):
        raise AssertionError("an EBSDIndexer must not be built to check a selection")

    monkeypatch.setattr(kp.detectors.EBSDDetector, "get_indexer", boom)
    c, _ = client
    spec = {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0], [2, 2, 0]]}
    assert c.put("/api/indexing/hough/reflectors",
                 json={"cif_path": cifs["Al"], "spec": spec}).status_code == 200
    assert c.get("/api/indexing/hough/reflectors/cost",
                 params={"cif_path": cifs["Al"]}).status_code == 200


# ---------------------------------------------------------------------------
# the indexing request carries the choice
# ---------------------------------------------------------------------------

def test_the_request_registers_the_choice_per_phase(cifs):
    from backend.api.routes import indexing as route
    req = SimpleNamespace(
        method="hough", cif_paths=[cifs["Al"], cifs["Ni"]], master_h5_paths=[], sht_paths=[],
        max_reflectors=None,
        reflector_specs=[{"mode": "custom", "families": [[1, 1, 1]]}, None])
    route._build_phase_configs(req)
    assert hr.get_spec("Al")["families"] == [[1, 1, 1]]
    assert hr.get_spec("Ni") is None


def test_a_request_without_the_field_leaves_the_registry_alone(cifs):
    from backend.api.routes import indexing as route
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1]]})
    req = SimpleNamespace(method="hough", cif_paths=[cifs["Al"]], master_h5_paths=[],
                          sht_paths=[], max_reflectors=None)
    route._build_phase_configs(req)
    assert hr.get_spec("Al") is not None
    req.reflector_specs = None
    route._build_phase_configs(req)
    assert hr.get_spec("Al") is not None


def test_a_bad_choice_in_the_request_fails_the_request_with_the_reason(cifs):
    from backend.api.routes import indexing as route
    req = SimpleNamespace(method="hough", cif_paths=[cifs["Al"]], master_h5_paths=[],
                          sht_paths=[], max_reflectors=None,
                          reflector_specs=[{"mode": "custom", "families": [[0, 0, 0]]}])
    with pytest.raises(ValueError, match="Reflector selection for Al"):
        route._build_phase_configs(req)


def test_the_request_model_accepts_the_field():
    from backend.api.routes.indexing import IndexingParams
    m = IndexingParams(reflector_specs=[{"mode": "auto"}, None])
    assert m.reflector_specs[1] is None
    assert IndexingParams().reflector_specs is None


# ---------------------------------------------------------------------------
# PC Refinement page: the loaded phase
# ---------------------------------------------------------------------------

def _pc_with_phase(c, cifs, *names):
    for n in names:
        assert c.post("/api/pc/phase/add", json={"cif_path": cifs[n]}).status_code == 200


def test_pc_table_put_and_the_other_pages_registry(client, cifs):
    c, _ = client
    _pc_with_phase(c, cifs, "Al")
    g = c.get("/api/pc/phase/reflectors")
    assert g.status_code == 200 and g.json()["n_effective"] == 4
    r = c.put("/api/pc/phase/reflectors",
              json={"phase_name": "Al", "spec": {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}})
    assert r.status_code == 200 and r.json()["n_effective"] == 2
    assert hr.get_spec("Al")["families"] == [[1, 1, 1], [2, 0, 0]]
    # the Indexing page sees it
    seen = c.get("/api/indexing/hough/reflectors", params={"cif_path": cifs["Al"]}).json()
    assert seen["mode"] == "custom" and seen["n_effective"] == 2


def test_pc_routes_name_the_phase_when_several_are_loaded(client, cifs):
    c, _ = client
    _pc_with_phase(c, cifs, "Al", "Ni")
    r = c.get("/api/pc/phase/reflectors")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "phase_name_required"
    r = c.get("/api/pc/phase/reflectors", params={"phase_name": "Ni"})
    assert r.status_code == 200 and r.json()["phase_name"] == "Ni"
    r = c.get("/api/pc/phase/reflectors", params={"phase_name": "Cu"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "no_phase"


def test_pc_without_a_phase_is_a_coded_404(client):
    c, _ = client
    r = c.get("/api/pc/phase/reflectors")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "no_phase"


def test_pc_validate_and_cost(client, cifs):
    c, _ = client
    _pc_with_phase(c, cifs, "Al")
    v = c.post("/api/pc/phase/reflectors/validate", json={"phase_name": "Al", "hkl": [3, 3, 1]})
    assert v.status_code == 200 and v.json()["label"] == "{331}"
    bad = c.post("/api/pc/phase/reflectors/validate", json={"phase_name": "Al", "hkl": [1, 1, 0]})
    assert bad.status_code == 400 and bad.json()["detail"]["code"] == "forbidden"
    cost = c.get("/api/pc/phase/reflectors/cost", params={"phase_name": "Al"})
    assert cost.status_code == 200 and cost.json()["n_rows"] == 64


def test_pc_refuses_a_change_while_an_optimisation_runs(client, cifs, monkeypatch):
    c, pcr = client
    _pc_with_phase(c, cifs, "Al")
    monkeypatch.setattr(pcr, "_optimization_active", True)
    r = c.put("/api/pc/phase/reflectors",
              json={"phase_name": "Al", "spec": {"mode": "custom", "families": [[1, 1, 1]]}})
    assert r.status_code == 409
    assert hr.get_spec("Al") is None


# ---------------------------------------------------------------------------
# the PC controller follows the registry
# ---------------------------------------------------------------------------

def _controller(cif):
    import kikuchipy as kp
    from pc_controller import PCController
    ctrl = PCController()
    ctrl.attach_detector(kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5),
                                                   sample_tilt=70.0))
    ctrl.load_phase(cif)
    return ctrl


def test_a_change_of_the_registry_drops_the_indexer_and_the_overlay_list(cifs):
    ctrl = _controller(cifs["Al"])
    first = ctrl._ensure_indexer()
    assert ctrl.reflector_specs_changed() is False and ctrl.indexer is first
    assert len(ctrl.reflectors.hkl) == 64

    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]})
    assert ctrl.reflector_specs_changed() is True
    assert ctrl.indexer is None                          # rebuilt on the next index
    assert len(ctrl.reflectors.hkl) == 14                # overlay = the chosen families
    assert ctrl.reflector_specs_changed() is False       # once

    rebuilt = ctrl._ensure_indexer()
    poles = {tuple(int(x) for x in r)
             for r in np.asarray(rebuilt.phaselist[0].polefamilies).reshape(-1, 3)}
    assert poles == {(1, 1, 1), (2, 0, 0)}


def test_the_session_notices_a_change_and_clears_its_simulated_bands(client, cifs):
    c, pcr = client
    _pc_with_phase(c, cifs, "Al")
    sess = pcr._get_session()
    sess.sim_cache[0] = {"ci": 0.5, "segments": [], "phase_name": "Al", "n_bands": 0}
    # a change made through the OTHER page's route
    c.put("/api/indexing/hough/reflectors",
          json={"cif_path": cifs["Al"],
                "spec": {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}})
    assert pcr._get_session().sim_cache == {}


def test_index_and_simulate_draws_the_chosen_families(client, cifs):
    """Overlay = what the indexer used: with two families chosen, their lines only."""
    import kikuchipy as kp
    from backend.api.routes.pcrefinement import _index_and_simulate
    _, pcr = client
    ctrl = _controller(cifs["Al"])
    nickel = kp.data.nickel_ebsd_small()
    ctrl.add_pattern((0, 0), np.asarray(nickel.data[0, 0]).copy())
    full = _index_and_simulate(ctrl, 0)
    pcr._sim_cache.clear()
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]})
    few = _index_and_simulate(ctrl, 0)          # notices the change by itself
    pcr._sim_cache.clear()
    assert few["n_bands"] < full["n_bands"]
    assert few["n_bands"] > 0
    # and the pattern was indexed with an indexer built from the new list
    poles = {tuple(int(x) for x in r)
             for r in np.asarray(ctrl.indexer.phaselist[0].polefamilies).reshape(-1, 3)}
    assert poles == {(1, 1, 1), (2, 0, 0)}


def test_the_pc_page_indexer_is_built_from_the_choice_for_two_phases(cifs):
    ctrl = _controller(cifs["Al"])
    ctrl.add_phase(cifs["Ni"])
    hr.set_spec("Ni", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]})
    ctrl._ensure_indexer()                      # built before the change ...
    hr.set_spec("Ni", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0], [2, 2, 0]]})
    ix = ctrl._ensure_indexer()                 # ... and rebuilt by itself
    n_al = len(np.asarray(ix.phaselist[0].polefamilies).reshape(-1, 3))
    n_ni = len(np.asarray(ix.phaselist[1].polefamilies).reshape(-1, 3))
    assert n_al == 4 and n_ni == 3


def test_a_stale_choice_does_not_stop_the_phase_from_loading(cifs, tmp_path):
    """Loading must work so the page can show the problem and reset it; the
    refusal comes when an indexer would be built."""
    hr.set_spec("Al", hr.spec_with_fingerprint(
        hr.phase_from_cif(cifs["Al"]), {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}))
    other = tmp_path / "x" / "Al.cif"
    other.parent.mkdir()
    other.write_text(AL_CIF.replace("4.0495", "4.2"), encoding="utf-8")
    ctrl = _controller(str(other))
    assert len(ctrl.reflectors.hkl) == 64                 # default list stands in
    with pytest.raises(hr.SpecError) as e:
        ctrl._ensure_indexer()
    assert e.value.code == "stale_spec"


def test_the_optimiser_uses_an_indexer_built_from_the_current_choice(cifs, monkeypatch):
    """A refine started after the choice changed must not run on the indexer that
    was cached under the old one."""
    import kikuchipy as kp
    import backend.api.routes.pcrefinement as pcr
    from backend.api.routes import ebsd_viewer

    ctrl = _controller(cifs["Al"])
    nickel = kp.data.nickel_ebsd_small()
    pat = np.asarray(nickel.data[0, 0]).copy()
    ctrl.add_pattern((0, 0), pat)
    ctrl._ensure_indexer()                      # cached under the default list
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]})

    seen = []

    def fake_optimize_pc(detector, indexer, pattern, method, search_limit, batch):
        seen.append(indexer)
        return (0.5, 0.5, 0.5), None

    monkeypatch.setattr(eu, "optimize_pc", fake_optimize_pc)
    monkeypatch.setattr(pcr, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(pcr, "_writeback_refined_pc", lambda *a, **k: None)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "unit", raising=False)
    pcr._optimization_tasks["t"] = {"status": "running", "progress": 0.0, "result": None,
                                    "error": None}
    pcr._run_optimization("t", [pat], "Nelder-Mead", 0.05)
    pcr._optimization_tasks.pop("t")
    poles = {tuple(int(x) for x in r)
             for r in np.asarray(seen[0].phaselist[0].polefamilies).reshape(-1, 3)}
    assert poles == {(1, 1, 1), (2, 0, 0)}


def test_n_bands_reaches_the_indexer(cifs):
    ctrl = _controller(cifs["Al"])
    ctrl.update_indexing_params(nBands=7)
    ix = ctrl._ensure_indexer()
    assert ix.bandDetectPlan is not None
    assert ctrl.nBands == 7
    assert ix.bandDetectPlan["nBands"] == 7 if isinstance(ix.bandDetectPlan, dict) else True


def test_a_changed_n_bands_drops_what_was_indexed_with_the_old_one(cifs):
    ctrl = _controller(cifs["Al"])
    ctrl.cache[0] = {"ci": 0.4}
    ctrl.update_indexing_params(nBands=9)
    assert ctrl.cache == {} and ctrl.indexer is None
    ctrl.cache[0] = {"ci": 0.4}
    ctrl.update_indexing_params(nBands=9)            # unchanged
    assert ctrl.cache == {0: {"ci": 0.4}}
