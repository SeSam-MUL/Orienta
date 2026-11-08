"""The one-click answer to a refusal: the largest default list that fits.

The default list is the same on every machine; only the count of rows that fits
depends on the machine's memory budget, and it is stored as an ordinary ``auto``
spec (``max_rows``) per phase, like any choice of the user's.

hcp Mg (MG_CIF) has library costs that are steps in the number of families:
70 rows 33.6 MiB, 64/60/56 rows 11.0 MiB, 50/44 rows 2.9 MiB, 40/32 rows 0.6 MiB (the ladder stops at 40).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from backend.api.services import hough_reflector_service as svc  # noqa: E402
from tests.test_hough_reflector_routes import (  # noqa: E402,F401
    cifs, clean_registry, client)
from tests.test_hough_reflectors import MG_CIF, _phase  # noqa: E402

MiB = 1 << 20


@pytest.fixture
def mg(tmp_path):
    return _phase(tmp_path, MG_CIF, "Mg")[0]


def _budget(monkeypatch, nbytes):
    monkeypatch.setattr(eu, "_triplet_library_budget_bytes", lambda: int(nbytes))
    # the cost line probes at 48 MiB; these libraries are smaller, so probe finer
    monkeypatch.setattr(eu, "_PREDICT_PROBE_ROWS", 10_000)


@pytest.mark.parametrize("budget_mib, rows", [(20, 64), (5, 50), (1, 40)])
def test_the_count_is_the_largest_ladder_step_that_fits(mg, monkeypatch, budget_mib, rows):
    _budget(monkeypatch, budget_mib * MiB)
    found = hr.largest_list_that_fits(mg)
    assert found["fits"] is False and found["rows"] == rows
    assert found["budget_bytes"] == budget_mib * MiB
    assert found["full_bytes"] > budget_mib * MiB
    assert found["predicted_bytes"] is None or found["predicted_bytes"] <= budget_mib * MiB
    assert found["families"] >= eu.MIN_FIT_FAMILIES


def test_a_list_that_already_fits_needs_nothing(mg, monkeypatch):
    _budget(monkeypatch, 1 << 40)
    found = hr.largest_list_that_fits(mg)
    assert found["fits"] is True and found["rows"] == 70


def test_below_the_floor_nothing_is_offered(mg, monkeypatch):
    """0.6 MiB is the smallest list on the ladder (40 rows); under that, no one-click."""
    _budget(monkeypatch, int(0.4 * MiB))
    found = hr.largest_list_that_fits(mg)
    assert found["rows"] is None and found["reason"] == "nothing_fits"


def test_too_few_families_are_not_offered(mg, monkeypatch):
    _budget(monkeypatch, 5 * MiB)
    monkeypatch.setattr(hr, "effective_families",
                        lambda phase, rows: (np.zeros((eu.MIN_FIT_FAMILIES - 1, 3), int), None))
    found = hr.largest_list_that_fits(mg)
    assert found["rows"] is None and found["reason"] == "too_few_families"


def test_the_list_is_the_same_on_every_machine_only_the_count_differs(mg, monkeypatch):
    lists = {}
    for budget in (20, 5, 1):
        _budget(monkeypatch, budget * MiB)
        n = hr.largest_list_that_fits(mg)["rows"]
        lists[n] = np.asarray(hr._rule_rows(mg, dict(hr.DEFAULT_RULE, max_rows=n))[0])
    full = np.asarray(hr._rule_rows(mg, hr.DEFAULT_RULE)[0])
    f = np.abs(np.asarray(eu._filtered_reflectors(mg, 1.0, 0.1).structure_factor)).ravel()
    assert sorted(lists) == [40, 50, 64]
    for n, rows in lists.items():
        assert len(rows) == n
        # every shorter list is the start of the full list's strongest rows: nested
        assert {tuple(r) for r in rows.round().astype(int)} <= {
            tuple(r) for r in full.round().astype(int)}
    assert {tuple(r) for r in lists[40].round().astype(int)} <= \
        {tuple(r) for r in lists[64].round().astype(int)}
    assert f.size >= 70


def test_fit_registers_an_ordinary_auto_spec_and_records_the_count(mg, monkeypatch):
    _budget(monkeypatch, 5 * MiB)
    out = svc.fit(mg, "Mg")
    spec = hr.get_spec("Mg")
    assert spec == {"mode": "auto", "rule": {"min_d": 1.0, "f_threshold": 0.1, "max_rows": 50}}
    assert out["mode"] == "auto" and out["fit"]["rows"] == 50
    # the indexer's list is exactly that many rows
    from orix.crystal_map import PhaseList
    assert len(hr.prepare_reflectors(PhaseList(phases=[mg])).hkl) == 50


def test_fit_refuses_when_nothing_fits_when_it_fits_and_over_an_own_selection(mg, monkeypatch):
    _budget(monkeypatch, int(0.4 * MiB))
    with pytest.raises(svc.ReflectorError) as e:
        svc.fit(mg, "Mg")
    assert e.value.code == "no_fit" and "Dictionary" in str(e.value)
    _budget(monkeypatch, 1 << 40)
    with pytest.raises(svc.ReflectorError) as e:
        svc.fit(mg, "Mg")
    assert e.value.code == "already_fits"
    _budget(monkeypatch, 5 * MiB)
    svc.change(mg, "Mg", {"mode": "custom", "families": [[0, 0, 2], [1, 0, 0], [1, 0, 1]]})
    with pytest.raises(svc.ReflectorError) as e:
        svc.fit(mg, "Mg")
    assert e.value.code == "own_selection"
    assert hr.get_spec("Mg")["mode"] == "custom"           # not replaced


def test_cost_carries_the_offer_only_when_there_is_one(mg, monkeypatch):
    _budget(monkeypatch, 5 * MiB)
    over = svc.cost(mg, "Mg")
    assert over["fits"] is False and over["fit"]["rows"] == 50
    _budget(monkeypatch, 1 << 40)
    assert "fit" not in svc.cost(mg, "Mg")
    _budget(monkeypatch, 5 * MiB)
    svc.change(mg, "Mg", {"mode": "custom", "families": [[0, 0, 2], [1, 0, 0], [1, 0, 1]]})
    monkeypatch.setattr(eu, "_PREDICT_PROBE_ROWS", 100)
    _budget(monkeypatch, 2000)
    monkeypatch.setattr(eu, "_PREDICT_PROBE_ROWS", 100)
    over_custom = svc.cost(mg, "Mg")
    assert over_custom["fits"] is False                    # it does not fit ...
    assert "fit" not in over_custom                        # ... and a hand selection is never replaced


def test_the_routes_of_both_pages(client, cifs, monkeypatch):
    c, _ = client
    _budget(monkeypatch, 5 * MiB)
    r = c.post("/api/indexing/hough/reflectors/fit", json={"cif_path": cifs["Mg"]})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "auto" and body["fit"]["rows"] == 50
    assert body["rule"]["max_rows"] == 50
    stored = c.get("/api/indexing/hough/reflector-specs").json()["specs"]
    assert stored["mg"]["rule"]["max_rows"] == 50
    # again: already stored -> the cost no longer over budget, the offer is gone
    cost = c.get("/api/indexing/hough/reflectors/cost", params={"cif_path": cifs["Mg"]}).json()
    assert cost["fits"] is True and "fit" not in cost
    # PC refinement page: the loaded phase, the same registry
    c.put("/api/indexing/hough/reflectors", json={"cif_path": cifs["Mg"], "spec": None})
    assert c.post("/api/pc/phase/add", json={"cif_path": cifs["Mg"]}).status_code == 200
    r2 = c.post("/api/pc/phase/reflectors/fit", json={"phase_name": "Mg"})
    assert r2.status_code == 200 and r2.json()["fit"]["rows"] == 50
    assert hr.get_spec("Mg")["rule"]["max_rows"] == 50
    # nothing fits -> a coded 400
    c.put("/api/indexing/hough/reflectors", json={"cif_path": cifs["Mg"], "spec": None})
    _budget(monkeypatch, int(0.4 * MiB))
    bad = c.post("/api/indexing/hough/reflectors/fit", json={"cif_path": cifs["Mg"]})
    assert bad.status_code == 400 and bad.json()["detail"]["code"] == "no_fit"


def test_no_offer_when_the_default_list_fits_but_a_stored_rule_does_not(mg, monkeypatch):
    """A stored rule that is too big while the default list fits: the answer is to reset
    the rule, and a button ending in "already fits" would be dead."""
    _budget(monkeypatch, 60 * MiB)
    svc.change(mg, "Mg", {"mode": "auto", "rule": {"min_d": 0.7, "max_rows": 200}})
    cost = svc.cost(mg, "Mg")
    assert cost["fits"] is False                 # the rule needs about 7.7 GiB
    assert "fit" not in cost                      # the default list (33.6 MiB) fits
    with pytest.raises(svc.ReflectorError) as e:
        svc.fit(mg, "Mg")
    assert e.value.code == "already_fits"


# --------------------------------------------------------------------------
# the phase a run builds: costs must be those of the indexer, on a monoclinic cell
# --------------------------------------------------------------------------

def test_cost_is_what_the_indexer_sees_on_a_monoclinic_cell(monkeypatch):
    """Al13Fe4 (C2/m). hexagonal Mg cannot tell: there the plain orix Phase and the
    FramePhase a run builds agree. For a monoclinic cell they do not (the two-fold axis
    along Z or along b): 7.15 GiB against 1.88 GiB for 70 rows. The service must cost
    what ``create_indexer`` builds, i.e. what ``predict_triplet_library`` reports."""
    import kikuchipy as kp
    from orix.crystal_map import PhaseList
    from tests.data_deps import CIF_LIBRARY, need

    path = need(CIF_LIBRARY / "Al13Fe4.cif")
    phase = svc.phase_for_path(path)
    monkeypatch.setattr(eu, "_triplet_library_budget_bytes", lambda: 2 << 30)
    det = kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)
    plist = PhaseList(phases=[phase])
    refl = hr.prepare_reflectors(plist)
    counts = (70,) + eu._REFLECTOR_LADDER
    seen = {c: b for c, _r, b in eu.predict_triplet_library(det, plist, refl, counts=counts)}
    assert seen[70] > 4 << 30                      # the FramePhase number; a plain Phase gives 1.9 GiB

    cost = svc.cost(phase, "Al13Fe4")
    assert cost["bytes"] == seen[70] and cost["fits"] is False
    # a monoclinic phase is offered no shorter list at all (see test_reflector_low_symmetry)
    assert cost["fit"]["rows"] is None and cost["fit"]["reason"] == "low_symmetry"
    # what the ladder would count there, as the indexer sees it, is what the rule withholds
    monkeypatch.setattr(hr, "has_no_one_click", lambda p: False)
    expected = max(k for k in eu._REFLECTOR_LADDER if seen[k] <= 2 << 30)
    found = svc.cost(phase, "Al13Fe4")["fit"]
    assert found["rows"] == expected
    assert found["families"] >= eu.MIN_FIT_FAMILIES


def test_the_cached_hough_indexer_follows_the_reflector_choice(cifs, monkeypatch):
    """The indexer behind pattern match and the phase check is cached per phase and
    geometry. After a one-click (or any change of the reflector choice or the row
    limit) the next call must build a new one, not return the old indexer."""
    from backend.api.routes import indexing as route

    builds = []
    monkeypatch.setattr(route, "_HOUGH_INDEXER_CACHE", {})
    monkeypatch.setattr(eu, "create_indexer", lambda *a, **kw: builds.append(1) or object())
    det = {"pat_height": 60, "pat_width": 60, "sample_tilt": 70.0, "tilt": 0.0,
           "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5, "binning": 1}
    route._hough_indexer_for(cifs["Mg"], det)
    route._hough_indexer_for(cifs["Mg"], det)
    assert len(builds) == 1                                 # the second call is the cache
    hr.set_spec("Mg", {"mode": "auto", "rule": {"max_rows": 50}})
    route._hough_indexer_for(cifs["Mg"], det)
    assert len(builds) == 2                                 # the choice changed: rebuilt
    route._hough_indexer_for(cifs["Mg"], det)
    assert len(builds) == 2
    eu.set_phase_reflector_limit("Mg", 40)                  # a row limit does as well
    route._hough_indexer_for(cifs["Mg"], det)
    assert len(builds) == 3
