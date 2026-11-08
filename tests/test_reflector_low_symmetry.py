"""No one-click shorter list for monoclinic and triclinic phases.

Measured on synthetic patterns with known orientations, a shorter default list
loses orientations of the low-symmetry classes 2/m and -1 (50 instead of 70 rows:
-5.8 and -17.6 percentage points within 5 degrees for two monoclinic phases, -13.4
against the 64-row list for a triclinic one) while orthorhombic and cubic phases stay
inside the limit set before the measurement. So for a phase of Laue class 2/m or -1
whose default list does not fit, the refusal stays but the one-click answer is not
offered; an orthorhombic phase keeps it.

The three cells are the synthetic crystals of ``test_phase_cell`` (one C2/m, one
P-1, one Cmcm). Their library costs, measured once with the real PyEBSDIndex library
builder (MiB, rows -> library): C2/m 70 -> 2500, 50 -> 112, 44 -> 58; P-1 70 ->
45799, 44 -> 2069, 40 -> 1047; Cmcm 70 -> 713, 56 -> 284, 50 -> 74. With budgets of
100 (C2/m, Cmcm) and 1200 MiB (P-1) the ladder alone WOULD give 44, 40 and 50 rows;
the tests state that, so a refusal that comes from the budget cannot pass for one
that comes from the rule.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from backend.api.services import hough_reflector_service as svc  # noqa: E402
from tests.test_hough_reflector_routes import clean_registry, client  # noqa: E402,F401
from tests.test_hough_refusal import _run  # noqa: E402
from tests.test_phase_cell import _from_cif  # noqa: E402

MiB = 1 << 20

#: cell name, Laue class written out by hand, budget in MiB, what the ladder alone would give
CELLS = {
    "monoclinic-C2/m": ("2/m", 100, 44),
    "triclinic-P-1": ("-1", 1200, 40),
    "orthorhombic-Cmcm": ("mmm", 100, 50),
}
LOW = ["monoclinic-C2/m", "triclinic-P-1"]


@pytest.fixture
def cell(tmp_path, monkeypatch):
    def make(name):
        laue, budget_mib, _ladder = CELLS[name]
        monkeypatch.setattr(eu, "_triplet_library_budget_bytes", lambda: budget_mib * MiB)
        # the cost line probes at 48 MiB; these libraries are small, so probe finer
        monkeypatch.setattr(eu, "_PREDICT_PROBE_ROWS", 10_000)
        phase, path = _from_cif(tmp_path, name)
        return phase, str(path)
    return make


def test_the_laue_class_is_read_from_the_point_group(cell):
    for name, (laue, _b, _l) in CELLS.items():
        phase, _ = cell(name)
        assert phase.point_group.laue.name == laue
        assert hr.has_no_one_click(phase) is (laue in ("2/m", "-1"))


@pytest.mark.parametrize("name", LOW)
def test_no_shorter_list_is_offered_where_the_ladder_alone_would_have_one(cell, monkeypatch, name):
    phase, _ = cell(name)
    found = hr.largest_list_that_fits(phase)
    assert found["fits"] is False and found["rows"] is None and found["families"] is None
    assert found["reason"] == "low_symmetry"
    # the budget is not what refuses: without the rule the ladder gives a count
    monkeypatch.setattr(hr, "has_no_one_click", lambda p: False)
    assert hr.largest_list_that_fits(phase)["rows"] == CELLS[name][2]


def test_an_orthorhombic_phase_keeps_the_one_click(cell):
    phase, _ = cell("orthorhombic-Cmcm")
    found = hr.largest_list_that_fits(phase)
    assert found["fits"] is False and found["rows"] == 50 and found["reason"] is None


def test_a_monoclinic_phase_that_fits_in_full_is_not_refused(cell, monkeypatch):
    phase, _ = cell("monoclinic-C2/m")
    monkeypatch.setattr(eu, "_triplet_library_budget_bytes", lambda: 1 << 40)
    assert hr.largest_list_that_fits(phase)["fits"] is True


@pytest.mark.parametrize("name", LOW)
def test_the_one_click_service_refuses_with_its_own_code_and_stores_nothing(cell, name):
    phase, _ = cell(name)
    with pytest.raises(svc.ReflectorError) as e:
        svc.fit(phase, phase.name)
    assert e.value.code == "low_symmetry"
    text = str(e.value)
    for part in ("monoclinic or triclinic", "lose orientations", "Spherical or Dictionary",
                 "fewer phases", "free memory", "by hand"):
        assert part in text
    assert hr.get_spec(phase.name) is None


def test_the_one_click_service_still_works_for_orthorhombic(cell):
    phase, _ = cell("orthorhombic-Cmcm")
    out = svc.fit(phase, "orthorhombic-Cmcm")
    assert out["fit"]["rows"] == 50
    assert hr.get_spec("orthorhombic-Cmcm")["rule"]["max_rows"] == 50


@pytest.mark.parametrize("name, rows", [("monoclinic-C2/m", None), ("triclinic-P-1", None),
                                        ("orthorhombic-Cmcm", 50)])
def test_the_cost_line_carries_the_reason_instead_of_a_count(cell, name, rows):
    """The card shows a button for a count and the text of the reason without one."""
    phase, _ = cell(name)
    over = svc.cost(phase, phase.name)
    assert over["fits"] is False
    assert over["fit"]["rows"] == rows
    assert over["fit"]["reason"] == (None if rows else "low_symmetry")


@pytest.mark.parametrize("name", LOW)
def test_the_route_of_both_pages_answers_with_the_coded_refusal(cell, client, name):
    phase, path = cell(name)
    c, _pcr = client
    r = c.post("/api/indexing/hough/reflectors/fit", json={"cif_path": path})
    assert r.status_code >= 400
    assert r.json()["detail"]["code"] == "low_symmetry"
    assert c.post("/api/pc/phase/add", json={"cif_path": path}).status_code == 200
    r = c.post("/api/pc/phase/reflectors/fit", json={"phase_name": phase.name})
    assert r.status_code >= 400
    assert r.json()["detail"]["code"] == "low_symmetry"


@pytest.mark.parametrize("name", LOW)
def test_the_indexing_refusal_stays_coded_without_a_count_and_says_what_else_to_do(
        monkeypatch, cell, name):
    _phase, path = cell(name)
    task = _run(monkeypatch, [path])
    assert task["status"] == "failed" and task["problem"]["code"] == "hough_library_too_big"
    (entry,) = task["problem"]["params"]["phases"]
    assert entry["fit_rows"] is None and entry["fit_reason"] == "low_symmetry"
    assert entry["own_selection"] is False
    err = task["error"]
    assert "largest list that fits" not in err
    for part in ("monoclinic or triclinic", "lose orientations", "Spherical or Dictionary",
                 "fewer phases", "free memory", "by hand"):
        assert part in err


def test_the_indexing_refusal_of_an_orthorhombic_phase_offers_the_count(monkeypatch, cell):
    _phase, path = cell("orthorhombic-Cmcm")
    task = _run(monkeypatch, [path])
    (entry,) = task["problem"]["params"]["phases"]
    assert entry["fit_rows"] == 50 and entry["fit_reason"] is None
    assert "largest list that fits" in task["error"]
    assert "monoclinic or triclinic" not in task["error"]


def test_a_multi_phase_run_marks_only_the_low_symmetry_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(eu, "_triplet_library_budget_bytes", lambda: 100 * MiB)
    monkeypatch.setattr(eu, "_PREDICT_PROBE_ROWS", 10_000)
    _m, mono = _from_cif(tmp_path, "monoclinic-C2/m")
    _o, orth = _from_cif(tmp_path, "orthorhombic-Cmcm")
    task = _run(monkeypatch, [mono, orth])
    by_name = {p["name"]: p for p in task["problem"]["params"]["phases"]}
    assert by_name["monoclinic-C2_m"]["fit_reason"] == "low_symmetry"
    assert by_name["monoclinic-C2_m"]["fit_rows"] is None
    assert by_name["orthorhombic-Cmcm"]["fit_rows"] == 50


def test_the_hand_path_stays_open_and_builds(monkeypatch, cell):
    """A shorter list made by hand in the reflector panel is not refused by the rule:
    the 12 strongest families of the monoclinic cell (the families of its 44-row list)
    fit the budget and the run is no longer refused."""
    name = "monoclinic-C2/m"
    phase, path = cell(name)
    out = svc.change(phase, phase.name, {"mode": "top_n", "n": 12})
    assert out["mode"] == "custom"
    assert svc.cost(phase, phase.name)["fits"] is True
    task = _run(monkeypatch, [path])
    assert task["status"] != "failed", task.get("error")
    assert "problem" not in task or task["problem"] is None


#: space group number -> (point group, Laue class), from the International Tables, by hand.
#: Point group m (6 to 9) is the case a test on the NAME of the Laue class misses: orix has
#: no name for it. P 1 (space group 1) is the symmetry-less CIF the refusal advice names.
BY_SPACE_GROUP = {1: ("1", "-1"), 2: ("-1", "-1"), 3: ("2", "2/m"), 5: ("2", "2/m"),
                  6: ("m", "2/m"), 8: ("m", "2/m"), 9: ("m", "2/m"), 14: ("2/m", "2/m"),
                  15: ("2/m", "2/m"), 16: ("222", "mmm"), 38: ("mm2", "mmm"), 62: ("mmm", "mmm"),
                  75: ("4", "4/m"), 143: ("3", "-3"), 194: ("6/mmm", "6/mmm"), 225: ("m-3m", "m-3m")}


@pytest.mark.parametrize("number", list(BY_SPACE_GROUP))
def test_every_monoclinic_and_triclinic_point_group_is_low_symmetry(number):
    from orix.crystal_map import Phase
    point_group, laue = BY_SPACE_GROUP[number]
    phase = Phase(space_group=number)
    assert phase.point_group.name == point_group
    assert hr.has_no_one_click(phase) is (laue in ("2/m", "-1"))


def test_a_phase_without_a_point_group_is_not_low_symmetry():
    from orix.crystal_map import Phase
    assert hr.has_no_one_click(Phase(name="x")) is False
