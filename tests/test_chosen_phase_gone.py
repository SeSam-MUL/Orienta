"""A phase the RX tab or the Texture tab chose can lose every pixel (the quality filter unindexes them): the routes
that use the remembered choice answer 400 with a sentence the page can show, not 500."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from tests.synthetic_grain_map import make_map, to_xmap


def _cubic_two_phase():
    m = make_map(shape=(40, 50), n_seeds=14)
    m["point_groups"] = np.array(["m-3m", "m-3m"])           # both phases can carry a texture
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.where(np.asarray(m["phase_id"]).reshape(-1) == 1, 0.0, 120.0)
    ds = EBSDDataset(xmap, step_size=0.5)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    return ds


@pytest.fixture
def chosen_then_gone():
    """Phase 1 is chosen on both tabs, then the quality filter unindexes every pixel of it."""
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    saved = ar._dataset
    ar._dataset = ds = _cubic_two_phase()
    ar._reset_grain_stats_settings()
    client = TestClient(app)
    try:
        assert client.post("/api/analysis/rx", json={"phase": 1, "rx_gos_threshold": 20.0}).status_code == 200
        assert client.post("/api/analysis/texture", json={"phase": 1}).status_code == 200
        assert client.get("/api/analysis/map/rx").status_code == 200
        assert client.get("/api/analysis/map/texture_component").status_code == 200
        ds.apply_quality_filter(bc_min=1.0)
        assert not (np.asarray(ds.xmap.phase_id) == 1).any()
        ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)   # the filter drops them
        yield client, ar
    finally:
        ar._dataset = saved
        ar._reset_grain_stats_settings()


def _is_the_sentence(response, kind, name="Al7FeCu2"):
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == f"the chosen {kind} phase {name} is no longer in this map; choose a phase"


def test_the_rx_map_and_a_new_classification_answer_400(chosen_then_gone):
    client, _ = chosen_then_gone
    _is_the_sentence(client.get("/api/analysis/map/rx"), "RX")
    _is_the_sentence(client.post("/api/analysis/rx", json={"rx_gos_threshold": 3.0}), "RX")     # phase: the stored one
    _is_the_sentence(client.post("/api/analysis/rx", json={"phase": 1}), "RX")


def test_the_texture_run_and_its_component_map_answer_400(chosen_then_gone):
    client, _ = chosen_then_gone
    _is_the_sentence(client.post("/api/analysis/texture", json={}), "texture")
    _is_the_sentence(client.post("/api/analysis/texture", json={"phase": 1}), "texture")
    _is_the_sentence(client.get("/api/analysis/map/texture_component"), "texture")


def test_both_exports_answer_400_before_any_work(chosen_then_gone, tmp_path):
    client, _ = chosen_then_gone
    _is_the_sentence(client.post("/api/analysis/export/excel",
                                 json={"output_path": str(tmp_path / "w.xlsx"), "dataset_name": "s"}), "RX")
    _is_the_sentence(client.get("/api/analysis/export/excel-download", params={"name": "s"}), "RX")
    assert not (tmp_path / "w.xlsx").exists()


def test_the_texture_phase_alone_gone_is_named_for_the_export(chosen_then_gone):
    client, _ = chosen_then_gone
    assert client.post("/api/analysis/rx", json={"phase": "most_measured"}).status_code == 200   # RX is fine again
    _is_the_sentence(client.get("/api/analysis/export/excel-download", params={"name": "s"}), "texture")


def test_the_default_choice_and_a_phase_that_is_still_there_are_untouched(chosen_then_gone):
    client, _ = chosen_then_gone
    r = client.post("/api/analysis/rx", json={"phase": "most_measured", "rx_gos_threshold": 20.0})
    assert r.status_code == 200 and r.json()["rx_phase"]["id"] == 0
    assert client.post("/api/analysis/texture", json={"phase": 0}).status_code == 200
    assert client.get("/api/analysis/map/rx").status_code == 200
    assert client.get("/api/analysis/map/texture_component").status_code == 200
    assert client.get("/api/analysis/export/excel-download", params={"name": "s"}).status_code == 200
    # an id the map never had keeps its own sentence, which lists the ids
    r = client.post("/api/analysis/rx", json={"phase": 9})
    assert r.status_code == 400 and "phase ids are" in r.json()["detail"]
