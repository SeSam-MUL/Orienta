"""Texture: phase choice in the grain split, locking, and the component layer.

- the split into RX and substructure grains uses the texture phase (export and batch), pinned by pixel counts;
- ``/texture`` has its own lock and does not hold up the map requests;
- the texture-choice reset on a load happens under the grain-state lock;
- the "Texture Components" map layer draws the components (it used to draw zeros);
- scalar maps of grain quantities leave the pixels without a grain out of their range.
"""
import base64
import io
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("orix")

import analysis.texture_analysis as ta                                           # noqa: E402
from analysis.ebsd_dataset import EBSDDataset                                    # noqa: E402
from analysis.texture_components import get_texture_preset                       # noqa: E402
from tests.test_texture_phase_choice import _grid_xmap, _random_quats            # noqa: E402

# 28 degrees from every FCC_Rolling component (found by search over random orientations): "no component"
FAR_FROM_ALL = [-0.07189701520447721, -0.6852135615895414, -0.5049332994823804, -0.5199572648711621]


@pytest.fixture
def api():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    saved = ar._dataset, dict(ar._texture_settings), dict(ar._grain_stats_settings)
    ar._dataset = None
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._texture_settings.clear()
        ar._texture_settings.update(saved[1])
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[2])


# ------------------------------------------------------------------------------------------------------------------
# the split follows the texture phase
# ------------------------------------------------------------------------------------------------------------------
def test_the_export_split_counts_the_remembered_phase(api, monkeypatch):
    """SmallVSLargeGrain is the split of the remembered texture phase: its pixel counts add up to that phase's pixels."""
    from analysis.grain_analysis import reconstruct_grains
    client, ar = api
    ds = EBSDDataset(_grid_xmap(8, 12, [0, 0, 1, 1, 1, 1], seed=21), step_size=1.0)        # Si 32 px, Al 64 px
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
    ar._dataset = ds
    counts = []
    for body in (None, {"phase": 0}, {"phase": "most_measured"}):
        if body is not None:
            assert client.post("/api/analysis/texture", json=body).status_code == 200
        # the split needs one phase: the RX phase is the texture phase
        assert client.post("/api/analysis/rx", json={"phase": body["phase"] if body else "most_measured"}).status_code == 200
        warnings = []
        results = ar._collect_export_results(warnings)
        assert "texture_by_rx" in results, warnings
        split = results["texture_by_rx"]
        counts.append(split["n_rx_pixels"] + split["n_substructure_pixels"])
    assert counts == [64, 32, 64]


def test_the_batch_split_counts_the_explicit_phase(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    from tests.batch_load_stub import patch_batch_load
    xmap = _grid_xmap(8, 12, [0, 0, 1, 1, 1, 1], seed=31)                                   # Si 32 px, Al 64 px
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    totals = {}
    for choice in ("most_measured", 0, 1):
        results = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"),
                                    texture_phase=choice, rx_phase=choice).process_file(tmp_path / "s.h5")
        split = results["texture_by_rx"]
        totals[choice] = split["n_rx_pixels"] + split["n_substructure_pixels"]
    assert totals == {"most_measured": 64, 0: 32, 1: 64}


# ------------------------------------------------------------------------------------------------------------------
# /texture has a lock of its own
# ------------------------------------------------------------------------------------------------------------------
def test_a_map_request_is_answered_while_a_texture_run_is_going(api, monkeypatch):
    from tests.synthetic_grain_map import make_map, to_xmap
    import analysis.texture_analysis as module
    client, ar = api
    m = make_map(shape=(30, 40), n_seeds=10)
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.random.default_rng(1).uniform(20, 200, xmap.size)
    ar._dataset = EBSDDataset(xmap, step_size=m["step"])
    started, release = threading.Event(), threading.Event()
    real = module.analyze_texture

    def slow(*a, **k):
        started.set()
        assert release.wait(30)
        return real(*a, **k)

    monkeypatch.setattr(module, "analyze_texture", slow)
    texture = {}
    runner = threading.Thread(target=lambda: texture.update(r=client.post("/api/analysis/texture")))
    runner.start()
    try:
        assert started.wait(30)
        assert ar._texture_lock.locked()
        answered = {}
        asker = threading.Thread(target=lambda: answered.update(r=client.get("/api/analysis/map/bc")))
        asker.start()
        asker.join(20)
        assert not asker.is_alive(), "the map request waited for the texture run"
        assert answered["r"].status_code == 200
    finally:
        release.set()
        runner.join(60)
    assert texture["r"].status_code == 200
    assert not ar._texture_lock.locked()


def test_two_texture_runs_do_not_overlap(api, monkeypatch):
    from tests.synthetic_grain_map import make_map, to_xmap
    import analysis.texture_analysis as module
    client, ar = api
    m = make_map(shape=(30, 40), n_seeds=10)
    ar._dataset = EBSDDataset(to_xmap(m), step_size=m["step"])
    real = module.analyze_texture
    inside, overlap = [0], []
    guard = threading.Lock()

    def counting(*a, **k):
        with guard:
            inside[0] += 1
            overlap.append(inside[0])
        try:
            return real(*a, **k)
        finally:
            with guard:
                inside[0] -= 1

    monkeypatch.setattr(module, "analyze_texture", counting)
    threads = [threading.Thread(target=lambda: client.post("/api/analysis/texture")) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert len(overlap) == 3 and max(overlap) == 1


# ------------------------------------------------------------------------------------------------------------------
# the reset of the texture choice happens under the grain-state lock
# ------------------------------------------------------------------------------------------------------------------
def test_the_choice_is_reset_under_the_lock_on_a_file_switch_and_on_a_load(api, monkeypatch, tmp_path):
    from tests.test_analysis_page_sync import _write_ctf
    client, ar = api
    seen = []
    real = ar._reset_grain_stats_settings
    monkeypatch.setattr(ar, "_reset_grain_stats_settings", lambda: (seen.append(ar._grain_state_lock.locked()), real())[1])
    ar.restore_dataset_for_file("nothing-stashed")
    _write_ctf(tmp_path / "s.ctf")
    ar._texture_settings["phase"] = 3
    assert client.post("/api/analysis/load", json={"xmap_path": str(tmp_path / "s.ctf")}).status_code == 200
    assert seen == [True, True]
    assert ar._texture_settings["phase"] == "most_measured"


# ------------------------------------------------------------------------------------------------------------------
# the Texture Components layer
# ------------------------------------------------------------------------------------------------------------------
def _png_pixels(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _component_world(ar):
    """8 x 12 pixels in 4 x 4 blocks. Al (id 1): Cube, Goss, far from all, Goss; Si (id 0): Cube; last block not indexed."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    xmap = _grid_xmap(8, 12, [1, 1, 1, 1, 0, 1], seed=5)
    probe = EBSDDataset(xmap, step_size=1.0)
    comps = get_texture_preset("FCC_Rolling")
    goss = ta.ideal_orientation(comps["Goss"], probe.phase)
    block_q = [[1.0, 0, 0, 0], goss, FAR_FROM_ALL, goss, [1.0, 0, 0, 0], [1.0, 0, 0, 0]]
    q = np.zeros((8, 12, 4))
    ph = np.asarray(xmap.phase_id).reshape(8, 12).copy()
    for b, bq in enumerate(block_q):
        r0, c0 = (b // 3) * 4, (b % 3) * 4
        q[r0:r0 + 4, c0:c0 + 4] = bq
    ph[4:, 8:] = -1                                                   # block 5: not indexed
    xmap = CrystalMap(rotations=Rotation(q.reshape(-1, 4)), phase_id=ph.reshape(-1), x=xmap.x, y=xmap.y,
                      phase_list=PhaseList(names=["Si", "Al"], point_groups=["m-3m", "m-3m"], ids=[0, 1]),
                      prop={"bc": np.full(96, 150.0)})
    ar._dataset = EBSDDataset(xmap, step_size=1.0)


def test_the_component_layer_draws_each_component_in_its_legend_colour(api):
    client, ar = api
    _component_world(ar)
    r = client.get("/api/analysis/map/texture_component")
    assert r.status_code == 200, r.text
    body = r.json()
    legend = {e["name"]: e["color"] for e in body["legend"]}
    names = list(get_texture_preset("FCC_Rolling"))
    assert list(legend) == names and len(set(legend.values())) == len(names)       # one colour each, all different
    rgb = lambda hex_: tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))             # noqa: E731
    img = _png_pixels(body["image"])
    assert img.shape == (8, 12, 3)
    assert tuple(img[1, 1]) == rgb(legend["Cube"])
    assert tuple(img[1, 5]) == rgb(legend["Goss"])
    assert tuple(img[5, 1]) == rgb(legend["Goss"])
    assert rgb(legend["Cube"]) != rgb(legend["Goss"])
    assert rgb(body["no_component_color"]) == (128, 128, 128)
    assert rgb(body["outside_color"]) == (0, 0, 0)
    assert tuple(img[1, 9]) == (128, 128, 128)                          # Al, but 28 degrees from every component
    assert tuple(img[5, 5]) == (0, 0, 0)                                # Si: not the texture phase, though it is Cube
    assert tuple(img[5, 9]) == (0, 0, 0)                                # not indexed
    assert body["texture_phase"] == {"id": 1, "name": "Al"}
    assert "value_range" not in body


def test_the_component_layer_follows_the_remembered_phase(api):
    client, ar = api
    _component_world(ar)
    assert client.post("/api/analysis/texture", json={"phase": 0}).status_code == 200
    body = client.get("/api/analysis/map/texture_component").json()
    legend = {e["name"]: e["color"] for e in body["legend"]}
    img = _png_pixels(body["image"])
    cube = tuple(int(legend["Cube"][i:i + 2], 16) for i in (1, 3, 5))
    assert body["texture_phase"]["id"] == 0
    assert tuple(img[5, 5]) == cube                                   # Si is the texture phase now: its Cube block is drawn
    assert tuple(img[1, 1]) == tuple(int(body["outside_color"][i:i + 2], 16) for i in (1, 3, 5))


def test_the_component_layer_refuses_a_map_without_a_cubic_phase_with_a_400(api):
    from orix.crystal_map import CrystalMap, PhaseList
    client, ar = api
    base = _grid_xmap(4, 4, [0], seed=2)
    xmap = CrystalMap(rotations=base.rotations, phase_id=np.zeros(16, int), x=base.x, y=base.y,
                      phase_list=PhaseList(names=["Mg2"], point_groups=["6/mmm"]))
    ar._dataset = EBSDDataset(xmap, step_size=1.0)
    r = client.get("/api/analysis/map/texture_component")
    assert r.status_code == 400 and "Choose a cubic phase" in r.json()["detail"]


# ------------------------------------------------------------------------------------------------------------------
# pixels without a grain are not part of the range of a grain map
# ------------------------------------------------------------------------------------------------------------------
def test_grain_maps_leave_the_pixels_without_a_grain_out_of_the_range(api, monkeypatch):
    from tests.synthetic_grain_map import make_map, to_xmap
    import backend.api.routes.analysis as mod
    client, ar = api
    m = make_map(shape=(60, 80), n_seeds=40)
    m["phase_id"][5:9, 20:26] = -1                                    # not indexed, and no closing (alpha 0): no grain there
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.random.default_rng(1).uniform(20, 200, xmap.size)
    ar._dataset = EBSDDataset(xmap, step_size=m["step"])
    r = client.post("/api/analysis/grains/reconstruct", json={"misorientation_threshold": 5.0, "min_grain_size": 1,
                                                                "alpha": 0.0, "fill": False})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed"
    seen = {}
    real = mod.colormap_array_to_base64
    monkeypatch.setattr(mod, "colormap_array_to_base64", lambda data, cmap: (seen.update(data=np.array(data)), real(data, cmap))[1])
    for kind in ("ecd", "gos"):
        body = client.get(f"/api/analysis/map/{kind}").json()
        data = seen["data"]
        assert np.isnan(data[5:9, 20:26]).all()                      # the grain-less pixels
        assert np.isfinite(data[~np.isnan(data)]).all() and np.isfinite(data).sum() > 0
        assert body["value_range"]["min"] == float(np.nanmin(data))
        assert body["value_range"]["max"] == float(np.nanmax(data))
    ecd = client.get("/api/analysis/map/ecd").json()
    assert ecd["value_range"]["min"] > 0                              # the smallest ECD, not the 0 of "no grain"
