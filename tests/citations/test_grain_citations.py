"""Citations for the grain reconstruction of the Analysis page: library entries, the step, its sentence, the endpoint.

The expected texts are typed here from the published references and from the settings the test sends; none is
produced by the code under test. One endpoint test compares the edge-grain count of the sentence with the
grain-size statistics of the same grains, the numbers the page shows next to it.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from backend.api.main import app
from backend.api.services.citations.grains import grain_reconstruction_step
from backend.api.services.citations.render import load_library, render_methods
from backend.api.services.citations.steps import get_step
from tests.synthetic_grain_map import make_map, to_xmap

client = TestClient(app)


# --------------------------------------------------------------------------------------------------
# library and step
# --------------------------------------------------------------------------------------------------
def test_the_library_holds_both_bachmann_papers_with_their_published_metadata():
    lib = load_library()
    a = lib["bachmann2011"]
    assert a["DOI"] == "10.1016/j.ultramic.2011.08.002"
    assert a["container-title"] == "Ultramicroscopy"
    assert (a["volume"], a["issue"], a["page"]) == ("111", "12", "1720-1733")
    assert a["issued"]["date-parts"][0][0] == 2011
    assert a["title"].startswith("Grain detection from 2d and 3d EBSD data")
    b = lib["bachmann2010"]
    assert b["DOI"] == "10.4028/www.scientific.net/SSP.160.63"
    assert b["container-title"] == "Solid State Phenomena"
    assert (b["volume"], b["page"]) == ("160", "63-68")
    assert b["issued"]["date-parts"][0][0] == 2010
    assert b["title"].startswith("Texture Analysis with MTEX")
    for entry in (a, b):
        assert [p["family"] for p in entry["author"]] == ["Bachmann", "Hielscher", "Schaeben"]
        assert entry["type"] == "article-journal"


def test_the_step_cites_orienta_both_papers_and_orix():
    step = get_step("analysis.grain_reconstruction")
    assert step is not None
    assert step.citation_ids == ("orienta", "bachmann2011", "bachmann2010", "orix")


# --------------------------------------------------------------------------------------------------
# the methods sentence names the real settings
# --------------------------------------------------------------------------------------------------
def _sentence(recon, **kw):
    return render_methods([grain_reconstruction_step(recon, **kw)])


def test_the_sentence_names_every_setting():
    gs = SimpleNamespace(n_grains_total=340, n_edge_grains=12, edge_safeguard_applied=False,
                         stats_phase=None, stats_phase_name=None)
    text = _sentence({"gb_threshold_deg": 7.5, "min_pixels": 11, "alpha": 2.4, "fill": True},
                     edge_rule="matlab_script", ecd_basis="cell_area", grain_size=gs)
    assert "(Bachmann et al., 2010, 2011)" in text
    assert "disorientation was below 7.5 deg" in text
    assert "Grains of fewer than 11 measurements were removed" in text
    assert "closing radius alpha (here 2.4, in pixel steps)" in text
    assert "filled in from the measurements of that grain" in text                  # fill on
    assert "the grains of all phases" in text
    assert "outermost two pixel rows or columns of the measured area were left out of them (12 of 340 grains)" in text
    assert "computed from the area of the grain's measurement cells" in text
    assert "not recorded" not in text


def test_the_other_choices_change_the_sentence():
    text = _sentence({"gb_threshold_deg": 4.0, "min_pixels": 3, "alpha": 0.0, "fill": False},
                     edge_rule="keep_all", ecd_basis="measured_pixels", stats_phase=2)
    assert "Not-indexed areas were not closed (alpha 0, below one pixel step)" in text
    assert "closing radius" not in text
    assert "shown with the orientation" not in text                                  # fill off
    assert "no grain was left out of them for touching the map border" in text
    assert "number of measured pixels of the grain times the square of the step size" in text
    assert "the grains of phase 2" in text


def test_alpha_zero_with_fill_on_does_not_speak_of_covered_pixels():
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 5, "alpha": 0.0, "fill": True},
                     edge_rule="keep_all", ecd_basis="cell_area")
    assert "Not-indexed areas were not closed (alpha 0, below one pixel step)" in text
    assert "covered pixels" not in text
    assert ("in the maps, pixels inside a grain without a measurement of their own were shown with an "
            "orientation filled in from the measurements of that grain") in text


def test_an_alpha_below_one_pixel_closes_nothing():
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 5, "alpha": 0.5, "fill": False},
                     edge_rule="keep_all", ecd_basis="cell_area")
    assert "Not-indexed areas were not closed (alpha 0.5, below one pixel step)" in text
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 5, "alpha": 1.0, "fill": False},
                     edge_rule="keep_all", ecd_basis="cell_area")
    assert "closing radius alpha (here 1, in pixel steps) were covered" in text


def test_a_minimum_of_one_pixel_is_no_second_pass():
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 1, "alpha": 3.1, "fill": True},
                     edge_rule="keep_all", ecd_basis="cell_area")
    assert "No minimum grain size was applied." in text
    assert "segmented again" not in text and "fewer than 1 measurements" not in text
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 6, "alpha": 3.1, "fill": True},
                     edge_rule="keep_all", ecd_basis="cell_area")
    assert ("Grains of fewer than 6 measurements were removed: their measurements were treated as not indexed "
            "and the map was segmented again, and grains still smaller after that were dropped.") in text


def test_the_script_rule_of_one_phase_names_that_phase_s_area():
    gs = SimpleNamespace(n_grains_total=40, n_edge_grains=9, edge_safeguard_applied=False,
                         stats_phase=1, stats_phase_name="Al")
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 5, "alpha": 3.1, "fill": True},
                     edge_rule="matlab_script", ecd_basis="cell_area", grain_size=gs)
    assert ("outermost two pixel rows or columns of the measured area of that phase were left out of them "
            "(9 of 40 grains)") in text
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 5, "alpha": 3.1, "fill": True},
                     edge_rule="matlab_script", ecd_basis="cell_area", stats_phase=None)
    assert "of the measured area were left out of them" in text


def test_no_grain_to_count_is_said_so():
    gs = SimpleNamespace(n_grains_total=0, n_edge_grains=0, edge_safeguard_applied=False,
                         stats_phase=None, stats_phase_name=None)
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 5, "alpha": 3.1, "fill": True},
                     edge_rule="outer_row", ecd_basis="cell_area", grain_size=gs)
    assert "but there were no grains to count" in text and "(0 of 0 grains)" not in text


@pytest.mark.parametrize("rule, words", [
    ("mtex", "grains touching the map border, or a not-indexed area open to the border,"),
    ("outer_row", "grains with a pixel in the first or last row or column of the map"),
])
def test_each_edge_rule_has_its_own_words(rule, words):
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 5, "alpha": 3.1, "fill": True},
                     edge_rule=rule, ecd_basis="cell_area")
    assert words in text


def test_the_safeguard_is_stated_when_it_kept_every_grain():
    gs = SimpleNamespace(n_grains_total=60, n_edge_grains=30, edge_safeguard_applied=True,
                         stats_phase=1, stats_phase_name="Al")
    text = _sentence({"gb_threshold_deg": 5.0, "min_pixels": 5, "alpha": 3.1, "fill": True},
                     edge_rule="mtex", ecd_basis="cell_area", grain_size=gs)
    assert "would have been left out of them, but all 60 grains were kept" in text
    assert "fewer than 50 grains, or less than 15 % of the grains, would have remained" in text
    assert "the grains of the phase Al" in text


def test_a_setting_that_was_not_recorded_is_named_not_guessed():
    text = _sentence({}, edge_rule="matlab_script", ecd_basis="cell_area")
    assert "[threshold_deg not recorded]" in text
    # without a minimum or an alpha the method's clauses stay and name the missing numbers
    assert "Grains of fewer than [min_pixels not recorded] measurements" in text
    assert "closing radius alpha (here [alpha not recorded], in pixel steps)" in text
    assert "culls" not in text and "closes_gaps" not in text            # no internal slot name


# --------------------------------------------------------------------------------------------------
# GET /api/citations/analysis
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def ar_with_grains():
    import backend.api.routes.analysis as ar
    m = make_map(shape=(40, 50), n_seeds=14)
    ds = EBSDDataset(to_xmap(m), step_size=m["step"])
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=6.5, min_pixels=4, alpha=1.5, fill=False)
    ds.grain_recon_params = {"gb_threshold_deg": 6.5, "min_pixels": 4, "alpha": 1.5, "min_intercept_um": None,
                             "fill": False}
    saved = ar._dataset
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    ar._grain_stats_settings.update(edge_rule="outer_row", ecd_basis="measured_pixels")
    try:
        yield ar
    finally:
        ar._dataset = saved
        ar._reset_grain_stats_settings()


def test_the_endpoint_is_404_without_grains():
    import backend.api.routes.analysis as ar
    saved = ar._dataset
    ar._dataset = None
    try:
        r = client.get("/api/citations/analysis")
    finally:
        ar._dataset = saved
    assert r.status_code == 404
    assert "Reconstruct the grains first" in r.json()["detail"]


def test_the_endpoint_answers_like_the_result_endpoint(ar_with_grains):
    r = client.get("/api/citations/analysis")
    assert r.status_code == 200
    body = r.json()
    # the keys of /api/citations/result/{id}, so the panel reads both the same way
    assert set(body) == {"bibtex", "methods", "plain", "steps", "undeclared", "warnings", "app_version"}
    ids = [ln.split("{", 1)[1].split(",", 1)[0] for ln in body["bibtex"].splitlines() if ln.startswith("@")]
    assert ids == ["orienta", "kikuchipy", "orix", "bachmann2011", "bachmann2010"]
    assert "@article{bachmann2011," in body["bibtex"]
    assert "doi = {10.1016/j.ultramic.2011.08.002}" in body["bibtex"]
    assert "https://doi.org/10.4028/www.scientific.net/SSP.160.63" in body["plain"]
    assert body["undeclared"] == [] and body["warnings"] == []
    assert [s["key"] for s in body["steps"]] == ["analysis.grain_reconstruction"]


def test_the_endpoint_sentence_follows_the_settings_the_page_used(ar_with_grains):
    ar = ar_with_grains
    text = client.get("/api/citations/analysis").json()["methods"]
    assert "below 6.5 deg" in text and "fewer than 4 measurements" in text and "(here 1.5, in pixel steps)" in text
    assert "shown with the orientation" not in text
    assert "first or last row or column of the map" in text
    assert "number of measured pixels" in text
    # the edge-grain count comes from the grain-size statistics of these very grains
    from analysis.grain_analysis import analyze_grain_size
    gs = analyze_grain_size(ar._dataset, ar._dataset.grains, edge_rule="outer_row", ecd_basis="measured_pixels")
    assert not gs.edge_safeguard_applied and 0 < gs.n_edge_grains < gs.n_grains_total     # the fixture is such a map
    assert f"({int(gs.n_edge_grains)} of {int(gs.n_grains_total)} grains)" in text

    # another statistics setting: the next answer says so
    ar._grain_stats_settings.update(edge_rule="keep_all", ecd_basis="cell_area")
    text = client.get("/api/citations/analysis").json()["methods"]
    assert "no grain was left out of them" in text and "measurement cells" in text


def test_the_endpoint_answers_for_a_reconstruction_without_grains(ar_with_grains):
    ar = ar_with_grains
    ds = ar._dataset
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=6.5, min_pixels=10 ** 6, alpha=1.5, fill=False)
    assert ds.grains.n_grains == 0
    r = client.get("/api/citations/analysis")
    assert r.status_code == 200
    assert "below 6.5 deg" in r.json()["methods"]


def test_the_result_endpoint_is_unchanged_by_the_shared_builder(register_result):
    rid = register_result([{"key": "indexing.hough", "params": {"orienta_version": "9.9.9"}}])
    body = client.get(f"/api/citations/result/{rid}").json()
    ids = [ln.split("{", 1)[1].split(",", 1)[0] for ln in body["bibtex"].splitlines() if ln.startswith("@")]
    assert ids == ["orienta", "kikuchipy", "orix", "pyebsdindex"]
    assert "bachmann" not in body["bibtex"]
    assert "called from Orienta 9.9.9." in body["methods"]
