"""PC refinement with several phases (duplex steel: austenite + ferrite).

PyEBSDIndex indexes a pattern against every phase of its phase list and keeps
the best-fitting one per pattern, and its PC optimiser scores a trial PC on
that best-of-phases result. So the controller only has to be able to hold
several phases; these tests pin that it does, that it reports which phase each
calibration pattern was indexed as, and that the single-phase path is exactly
what it was.
"""
import numpy as np
import pytest

from pc_controller import PCController


def _cubic_cif(name: str, a: float, number: int, hm: str, sites) -> str:
    rows = "\n".join(f"{lab} {el} {x} {y} {z}" for lab, el, x, y, z in sites)
    return f"""data_{name}
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M '{hm}'
_symmetry_Int_Tables_number {number}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{rows}
"""


@pytest.fixture
def duplex_cifs(tmp_path):
    """Austenite (fcc, Ni-like cell so the kikuchipy nickel pattern matches
    it) and ferrite (bcc), written as plain CIFs."""
    aus = tmp_path / "austenite.cif"
    fer = tmp_path / "ferrite.cif"
    aus.write_text(_cubic_cif("austenite", 3.52, 225, "F m -3 m",
                              [("Ni1", "Ni", 0, 0, 0)]), encoding="utf-8")
    fer.write_text(_cubic_cif("ferrite", 2.866, 229, "I m -3 m",
                              [("Fe1", "Fe", 0, 0, 0)]), encoding="utf-8")
    return str(aus), str(fer)


@pytest.fixture
def nickel():
    import kikuchipy as kp
    s = kp.data.nickel_ebsd_small()
    return s


NICKEL_PC = (0.4251, 0.2130, 0.5003)    # mean of the dataset's per-pixel PCs


def _ctrl_with_detector(signal):
    from kikuchipy.detectors import EBSDDetector
    ctrl = PCController()
    ctrl.attach_detector(EBSDDetector(shape=(60, 60), pc=NICKEL_PC, sample_tilt=70.0))
    return ctrl


# --------------------------------------------------------------------------
# The controller holds several phases
# --------------------------------------------------------------------------

def test_single_phase_load_is_unchanged(duplex_cifs):
    """load_phase keeps its contract: one phase, ONE reflector object."""
    ctrl = PCController()
    ctrl.load_phase(duplex_cifs[0])
    assert list(ctrl.phase_list.names) == ["austenite"]
    assert not isinstance(ctrl.reflectors, list)
    assert ctrl.phase.name == "austenite"


def test_add_phase_builds_a_two_phase_list(duplex_cifs):
    aus, fer = duplex_cifs
    ctrl = PCController()
    ctrl.add_phase(aus)               # first add == load
    ctrl.add_phase(fer)
    assert list(ctrl.phase_list.names) == ["austenite", "ferrite"]
    assert isinstance(ctrl.reflectors, list) and len(ctrl.reflectors) == 2
    assert ctrl.phase.name == "austenite"          # `phase` stays the first
    assert ctrl.phase_names() == ["austenite", "ferrite"]
    assert ctrl.indexer is None                    # must be rebuilt


def test_add_phase_twice_is_rejected_and_changes_nothing(duplex_cifs):
    aus, fer = duplex_cifs
    ctrl = PCController()
    ctrl.add_phase(aus)
    ctrl.add_phase(fer)
    before = (list(ctrl.phase_list.names), len(ctrl.reflectors))
    with pytest.raises(ValueError, match="already"):
        ctrl.add_phase(fer)
    assert (list(ctrl.phase_list.names), len(ctrl.reflectors)) == before


def test_a_bad_cif_leaves_the_loaded_phases_alone(duplex_cifs, tmp_path):
    aus, _ = duplex_cifs
    junk = tmp_path / "junk.cif"
    junk.write_text("this is not a cif", encoding="utf-8")
    ctrl = PCController()
    ctrl.add_phase(aus)
    with pytest.raises(Exception):
        ctrl.add_phase(str(junk))
    assert ctrl.phase_names() == ["austenite"]
    assert not isinstance(ctrl.reflectors, list)


def test_remove_phase_back_to_one_and_to_none(duplex_cifs):
    aus, fer = duplex_cifs
    ctrl = PCController()
    ctrl.add_phase(aus)
    ctrl.add_phase(fer)
    ctrl.remove_phase("austenite")
    assert ctrl.phase_names() == ["ferrite"]
    assert ctrl.phase.name == "ferrite"
    assert not isinstance(ctrl.reflectors, list)   # back to the single form
    ctrl.remove_phase("ferrite")
    assert ctrl.phase is None and ctrl.phase_list is None
    assert ctrl.phase_names() == []
    with pytest.raises(KeyError):
        ctrl.remove_phase("ferrite")


def test_indexer_for_two_phases_carries_both(duplex_cifs, nickel):
    aus, fer = duplex_cifs
    ctrl = _ctrl_with_detector(nickel)
    ctrl.add_phase(aus)
    ctrl.add_phase(fer)
    idx = ctrl._ensure_indexer()
    assert len(idx.phaseLib) == 2


def test_an_indexing_parameter_change_keeps_every_phase(duplex_cifs):
    aus, fer = duplex_cifs
    ctrl = PCController()
    ctrl.add_phase(aus)
    ctrl.add_phase(fer)
    ctrl.update_indexing_params(f_threshold=0.2)
    assert isinstance(ctrl.reflectors, list) and len(ctrl.reflectors) == 2


# --------------------------------------------------------------------------
# Which phase was each calibration pattern indexed as
# --------------------------------------------------------------------------

def test_index_and_simulate_names_the_winning_phase(duplex_cifs, nickel):
    """A nickel pattern against {ferrite, austenite}: austenite (fcc) wins, and
    the simulated Kikuchi lines come from ITS reflectors, whichever position
    it has in the list."""
    from backend.api.routes.pcrefinement import _index_and_simulate, _sim_cache
    aus, fer = duplex_cifs
    pat = np.asarray(nickel.data[0, 0]).copy()

    # Reference: the same pattern against austenite alone.
    solo = _ctrl_with_detector(nickel)
    solo.add_phase(aus)
    solo.add_pattern((0, 0), pat)
    expected = _index_and_simulate(solo, 0)["segments"]
    _sim_cache.clear()
    assert expected

    for order in ([fer, aus], [aus, fer]):
        ctrl = _ctrl_with_detector(nickel)
        for p in order:
            ctrl.add_phase(p)
        ctrl.add_pattern((0, 0), pat)
        r = _index_and_simulate(ctrl, 0)
        assert r["phase_name"] == "austenite", order
        np.testing.assert_allclose(
            np.asarray(r["segments"]), np.asarray(expected), atol=1e-6,
            err_msg="segments must be the winning phase's own lines")
        assert r["phase_index"] == list(ctrl.phase_names()).index("austenite")
        _sim_cache.clear()


def test_single_phase_result_has_no_phase_index(duplex_cifs, nickel):
    from backend.api.routes.pcrefinement import _index_and_simulate, _sim_cache
    ctrl = _ctrl_with_detector(nickel)
    ctrl.add_phase(duplex_cifs[0])
    ctrl.add_pattern((0, 0), np.asarray(nickel.data[0, 0]).copy())
    r = _index_and_simulate(ctrl, 0)
    assert set(r) == {"ci", "phase_name", "segments", "n_bands"}
    _sim_cache.clear()


# --------------------------------------------------------------------------
# The optimisation run
# --------------------------------------------------------------------------

class _Cal:
    """Stands in for the calibration store inside _run_optimization."""


def _run(monkeypatch, ctrl, patterns, pcs):
    """Drive `_run_optimization` with a stubbed optimiser; return (task, calls)."""
    import ebsd_utils
    import backend.api.routes.pcrefinement as pcr
    from backend.api.routes import ebsd_viewer

    calls = []

    def fake_optimize_pc(detector, indexer, pattern, method, search_limit, batch):
        calls.append({"indexer": indexer, "method": method,
                      "search_limit": search_limit, "batch": batch,
                      "pattern_shape": np.asarray(pattern).shape})
        return pcs[len(calls) - 1], None

    monkeypatch.setattr(ebsd_utils, "optimize_pc", fake_optimize_pc)
    monkeypatch.setattr(pcr, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(pcr, "_writeback_refined_pc", lambda *a, **k: None)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "unit", raising=False)
    task_id = "t-unit"
    pcr._optimization_tasks[task_id] = {"status": "running", "progress": 0.0,
                                        "result": None, "error": None}
    pcr._sim_cache.clear()
    pcr._run_optimization(task_id, patterns, "Nelder-Mead", 0.05)
    return pcr._optimization_tasks.pop(task_id), calls


def test_single_phase_optimisation_payload_is_unchanged(monkeypatch, duplex_cifs, nickel):
    """The request path and the result keys of a one-phase refine, pinned."""
    ctrl = _ctrl_with_detector(nickel)
    ctrl.add_phase(duplex_cifs[0])
    pats = [np.asarray(nickel.data[0, 0]).copy(), np.asarray(nickel.data[1, 1]).copy()]
    for i, p in enumerate(pats):
        ctrl.add_pattern((i, i), p)
    pcs = [(0.4250, 0.2130, 0.5000), (0.4252, 0.2132, 0.5004)]
    task, calls = _run(monkeypatch, ctrl, pats, pcs)

    assert task["status"] == "completed", task.get("error")
    res = task["result"]
    assert set(res) == {"pc_values", "mean_pc", "ci", "segments", "pc_warning",
                        "pc_warning_codes", "pc_deviation", "pattern_size"}
    assert res["pc_values"] == [list(p) for p in pcs]
    np.testing.assert_allclose(res["mean_pc"], [0.4251, 0.2131, 0.5002])
    # one optimiser call per pattern, all on the SAME single-phase indexer
    assert [c["method"] for c in calls] == ["Nelder-Mead"] * 2
    assert all(c["batch"] is True and c["search_limit"] == 0.05 for c in calls)
    assert len({id(c["indexer"]) for c in calls}) == 1
    assert len(calls[0]["indexer"].phaseLib) == 1


def test_two_phase_optimisation_uses_one_two_phase_indexer_and_reports_phases(
        monkeypatch, duplex_cifs, nickel):
    ctrl = _ctrl_with_detector(nickel)
    for p in duplex_cifs:
        ctrl.add_phase(p)
    pats = [np.asarray(nickel.data[0, 0]).copy(), np.asarray(nickel.data[1, 1]).copy()]
    for i, p in enumerate(pats):
        ctrl.add_pattern((i, i), p)
    pcs = [(0.4250, 0.2130, 0.5000), (0.4252, 0.2132, 0.5004)]
    task, calls = _run(monkeypatch, ctrl, pats, pcs)

    assert task["status"] == "completed", task.get("error")
    assert len(calls[0]["indexer"].phaseLib) == 2
    assert len({id(c["indexer"]) for c in calls}) == 1
    res = task["result"]
    assert res["phase_names"] == ["austenite", "ferrite"]
    # Per pattern: which phase it is indexed as at the refined PC. The nickel
    # patterns are fcc, so they must come back as austenite.
    assert [e["phase_name"] for e in res["pattern_phases"]] == ["austenite", "austenite"]
    assert [e["index"] for e in res["pattern_phases"]] == [0, 1]
    assert all(e["ci"] is not None for e in res["pattern_phases"])


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------

@pytest.fixture
def pc_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import backend.api.routes.pcrefinement as pcr
    monkeypatch.setattr(pcr, "_sessions", {})
    app = FastAPI()
    app.include_router(pcr.router, prefix="/api/pc")
    return TestClient(app), pcr


def test_phase_add_and_remove_over_http(pc_client, duplex_cifs):
    c, _ = pc_client
    aus, fer = duplex_cifs

    r = c.post("/api/pc/phase/add", json={"cif_path": aus})
    assert r.status_code == 200 and r.json()["n_phases"] == 1
    r = c.post("/api/pc/phase/add", json={"cif_path": fer})
    body = r.json()
    assert r.status_code == 200
    assert body["phase_name"] == "ferrite"
    assert [p["name"] for p in body["phases"]] == ["austenite", "ferrite"]
    assert body["phases"][1]["space_group"]            # shown on the page

    st = c.get("/api/pc/status").json()
    assert st["phase_names"] == ["austenite", "ferrite"]
    assert st["phase_name"] == "austenite"             # unchanged: the first
    assert st["has_phase"] is True

    r = c.post("/api/pc/phase/remove", json={"phase_name": "austenite"})
    assert r.status_code == 200
    assert [p["name"] for p in r.json()["phases"]] == ["ferrite"]
    assert c.get("/api/pc/status").json()["phase_names"] == ["ferrite"]

    assert c.post("/api/pc/phase/remove", json={"phase_name": "austenite"}).status_code == 404


def test_phase_add_rejects_duplicate_and_unreadable_files(pc_client, duplex_cifs, tmp_path):
    c, _ = pc_client
    aus, fer = duplex_cifs
    c.post("/api/pc/phase/add", json={"cif_path": aus})
    dup = c.post("/api/pc/phase/add", json={"cif_path": aus})
    assert dup.status_code == 400 and "already" in dup.json()["detail"]
    missing = c.post("/api/pc/phase/add", json={"cif_path": str(tmp_path / "nope.cif")})
    assert missing.status_code == 400
    assert c.get("/api/pc/status").json()["phase_names"] == ["austenite"]


def test_phase_load_still_replaces_everything(pc_client, duplex_cifs):
    """`/phase/load` keeps its meaning: exactly this one phase."""
    c, _ = pc_client
    aus, fer = duplex_cifs
    c.post("/api/pc/phase/add", json={"cif_path": aus})
    c.post("/api/pc/phase/add", json={"cif_path": fer})
    r = c.post("/api/pc/phase/load", json={"cif_path": fer})
    assert r.status_code == 200
    assert c.get("/api/pc/status").json()["phase_names"] == ["ferrite"]


def test_phases_cannot_change_while_an_optimisation_runs(pc_client, duplex_cifs, monkeypatch):
    c, pcr = pc_client
    monkeypatch.setattr(pcr, "_optimization_active", True)
    r = c.post("/api/pc/phase/add", json={"cif_path": duplex_cifs[0]})
    assert r.status_code == 409
    r = c.post("/api/pc/phase/remove", json={"phase_name": "austenite"})
    assert r.status_code == 409


def test_several_phases_do_not_recompute_reflectors_for_every_pattern(
        monkeypatch, duplex_cifs, nickel):
    """Index All walks every calibration pattern. Reflectors of a big cell take
    seconds; with several phases they must come from the controller, built
    once, not be recomputed for each pattern."""
    import ebsd_utils
    from backend.api.routes.pcrefinement import _index_and_simulate, _sim_cache

    calls = []
    real = ebsd_utils.prepare_reflectors

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ebsd_utils, "prepare_reflectors", counting)
    ctrl = _ctrl_with_detector(nickel)
    for p in duplex_cifs:
        ctrl.add_phase(p)
    calls.clear()                       # only what indexing itself does counts
    for i in range(3):
        ctrl.add_pattern((i, i), np.asarray(nickel.data[i, i]).copy())
    for i in range(3):
        _index_and_simulate(ctrl, i)
    _sim_cache.clear()
    assert len(calls) <= 1, f"reflectors recomputed {len(calls)} times for 3 patterns"
