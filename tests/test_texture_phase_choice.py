"""Which phase the texture is computed on, and the index and entropy of the kernel density.

The owner's frame script computes the texture on one phase (``ph = 'aluminium'``, the matrix, or ``'indexed'``).
Orienta computes it on the phase with the most indexed pixels (default) or on a phase id the caller names, and says
which. The components are cubic, so a phase that is not cubic is refused with the cubic phases of the map named.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("orix")

import analysis.texture_analysis as ta                                           # noqa: E402
from analysis.texture_components import get_texture_preset                       # noqa: E402

EXACT_J, EXACT_S = 182.14142464, -4.746045013       # one orientation, 5 degree kernel, m-3m (tests/test_texture_odf.py)


def _random_quats(n, seed):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _dataset(phases, unindexed=2):
    """``phases``: list of (phase id, name, point group, quaternions). Phase ids need not be 0, 1, ... in order."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from analysis.ebsd_dataset import EBSDDataset
    quats = [np.asarray(q, float).reshape(-1, 4) for _, _, _, q in phases]
    quats.append(np.tile([1.0, 0, 0, 0], (unindexed, 1)))
    ids = np.concatenate([np.full(len(q), pid) for (pid, _, _, _), q in zip(phases, quats)]
                         + [np.full(unindexed, -1)])
    q = np.concatenate(quats)
    n = len(q)
    plist = PhaseList(names=[p[1] for p in phases], point_groups=[p[2] for p in phases],
                      space_groups=None, ids=[p[0] for p in phases])
    xmap = CrystalMap(rotations=Rotation(q), phase_id=ids, x=np.arange(n, dtype=float), y=np.zeros(n),
                      phase_list=plist, prop={"bc": np.full(n, 150.0)})
    return EBSDDataset(xmap, step_size=1.0)


def _comps():
    fcc = get_texture_preset("FCC_Rolling")
    return {"Goss": fcc["Goss"], "Cube": fcc["Cube"], "Brass": fcc["Brass"]}


@pytest.fixture(scope="module")
def world():
    """Particles = id 0 (6 pixels, all Cube), matrix 'Al' = id 1 (14 pixels: 3 Goss, 5 Cube, 6 random), a hexagonal
    phase = id 2 (4 pixels). The matrix has the HIGHER id and the most pixels."""
    probe = _dataset([(0, "Si", "m-3m", [[1.0, 0, 0, 0]])])
    goss = ta.ideal_orientation(_comps()["Goss"], probe.phase)
    matrix = np.vstack([np.tile(goss, (3, 1)), np.tile([1.0, 0, 0, 0], (5, 1)), _random_quats(6, 5)])
    particles = np.tile([1.0, 0, 0, 0], (6, 1))
    hexa = _random_quats(4, 6)
    ds = _dataset([(0, "Si", "m-3m", particles), (1, "Al", "m-3m", matrix), (2, "Mg2", "6/mmm", hexa)])
    alone = _dataset([(0, "Al", "m-3m", matrix)])
    return ds, alone, matrix, particles


def test_the_default_is_the_phase_with_the_most_indexed_pixels_not_the_lowest_id(world):
    ds, _, matrix, _ = world
    assert ta.resolve_texture_phase(ds) == 1
    assert ta.resolve_texture_phase(ds, "most_measured") == 1
    assert ta.texture_phase_info(ds) == {"id": 1, "name": "Al"}


def test_the_numbers_of_the_default_are_those_of_the_matrix_pixels_alone(world):
    ds, alone, _, _ = world
    got = ta.analyze_texture(ds, _comps())
    want = ta.analyze_texture(alone, _comps())
    assert (got.phase_id, got.phase_name) == (1, "Al")
    assert (want.phase_id, want.phase_name) == (0, "Al")
    assert got.component_fractions == want.component_fractions
    assert got.component_fractions["Goss"] == pytest.approx(3 / 14 * 100)
    for name in ("plane_111_fraction", "plane_100_fraction", "plane_110_fraction",
                 "plane_111_family_fraction", "plane_100_family_fraction", "plane_110_family_fraction"):
        assert getattr(got, name) == getattr(want, name), name
    assert (got.texture_index, got.entropy) == (want.texture_index, want.entropy)


def test_an_explicit_id_picks_that_phase(world):
    ds, _, _, particles = world
    res = ta.analyze_texture(ds, _comps(), texture_phase=0)
    assert (res.phase_id, res.phase_name) == (0, "Si")
    assert res.component_fractions == {"Goss": 0.0, "Cube": 100.0, "Brass": 0.0}
    alone = _dataset([(0, "Si", "m-3m", particles)])
    from analysis.texture_odf import texture_index_and_entropy
    assert (res.texture_index, res.entropy) == texture_index_and_entropy(
        alone, np.asarray(alone.xmap.phase_id) == 0, halfwidth_deg=5.0)


def test_the_map_the_split_and_the_angles_follow_the_texture_phase(world):
    ds, alone, _, _ = world
    cmap, _ = ta.calculate_texture_component_map(ds, _comps(), 10.0)
    flat = cmap.reshape(-1)
    ids = np.asarray(ds.xmap.phase_id)
    assert (flat[ids != 1] == 0).all()                              # the particles and the rest get no component
    want, _ = ta.calculate_texture_component_map(alone, _comps(), 10.0)
    assert np.array_equal(flat[ids == 1], want.reshape(-1)[np.asarray(alone.xmap.phase_id) == 0])

    class Grains:
        grain_ids = np.where(np.arange(len(ids)) < 12, 1, 2)
    got = ta.calculate_texture_split_by_rx(ds, Grains(), [1], _comps(), 10.0)
    assert got["n_rx_pixels"] + got["n_substructure_pixels"] == 14  # the matrix pixels only
    other = ta.calculate_texture_split_by_rx(ds, Grains(), [1], _comps(), 10.0, texture_phase=0)
    assert other["n_rx_pixels"] + other["n_substructure_pixels"] == 6


def test_the_angles_of_the_two_phases_are_not_mixed_up_by_the_cache(world):
    ds, _, _, _ = world
    a = ta.get_component_angles(ds, _comps(), 1)
    b = ta.get_component_angles(ds, _comps(), 0)
    c = ta.get_component_angles(ds, _comps())
    assert (a.n_pixels, b.n_pixels, c.n_pixels) == (14, 6, 14)


def test_a_tie_goes_to_the_lower_id():
    ds = _dataset([(0, "A", "m-3m", _random_quats(5, 1)), (1, "B", "m-3m", _random_quats(5, 2))])
    assert ta.resolve_texture_phase(ds) == 0


def test_a_phase_that_is_not_cubic_is_refused_and_the_cubic_ones_are_named(world):
    ds, _, _, _ = world
    with pytest.raises(ValueError) as e:
        ta.analyze_texture(ds, _comps(), texture_phase=2)
    msg = str(e.value)
    assert "not cubic" in msg and "6/mmm" in msg and "0 (Si)" in msg and "1 (Al)" in msg
    with pytest.raises(ValueError, match="not cubic"):
        ta.resolve_texture_phase(ds, 2)
    with pytest.raises(ValueError, match="not cubic"):
        ta.texture_phase_info(ds, 2)
    # the building blocks (angles, fractions, planes) are generic; only the top of the chain requires a cubic phase
    assert ta.calculate_surface_plane_fraction(ds, [0, 0, 1], texture_phase=2) >= 0.0


def test_the_cubic_laue_classes_are_the_cubic_point_groups():
    for pg in ("23", "m-3", "432", "-43m", "m-3m"):
        ds = _dataset([(0, "X", pg, _random_quats(4, 1))])
        assert ta.resolve_texture_phase(ds) == 0, pg
    for pg in ("6/mmm", "mmm", "4/mmm", "-3m", "2/m", "1"):
        ds = _dataset([(0, "X", pg, _random_quats(4, 1))])
        with pytest.raises(ValueError, match="no cubic phase"):
            ta.resolve_texture_phase(ds)


def test_a_most_measured_phase_that_is_not_cubic_says_to_choose_a_cubic_phase():
    ds = _dataset([(0, "Si", "m-3m", _random_quats(3, 1)), (1, "Mg2", "6/mmm", _random_quats(9, 2))])
    with pytest.raises(ValueError, match="Choose a cubic phase") as e:
        ta.analyze_texture(ds, _comps())
    assert "0 (Si)" in str(e.value) and "1 (Mg2)" in str(e.value)
    assert ta.analyze_texture(ds, _comps(), texture_phase=0).phase_id == 0


@pytest.mark.parametrize("bad", [7, -1, 2.0, "Al", True, None, [0]])
def test_an_unknown_choice_raises_and_lists_the_phase_ids(world, bad):
    ds, _, _, _ = world
    with pytest.raises(ValueError) as e:
        ta.resolve_texture_phase(ds, bad)
    assert "[0, 1, 2]" in str(e.value)


# ------------------------------------------------------------------------------------------------------------------
# index and entropy of the 5 degree kernel density
# ------------------------------------------------------------------------------------------------------------------
def test_analyze_texture_gives_the_exact_single_orientation_values():
    """One orientation (all pixels identical) has the module's exact index and entropy: 182.1414 and -4.74605."""
    ds = _dataset([(0, "Al", "m-3m", np.tile(_random_quats(1, 11), (400, 1)))])
    res = ta.analyze_texture(ds, _comps())
    assert res.texture_index == pytest.approx(EXACT_J, abs=1e-3)
    assert res.entropy == pytest.approx(EXACT_S, abs=5e-3)


def test_a_random_texture_is_near_uniform():
    ds = _dataset([(0, "Al", "m-3m", _random_quats(4000, 12))])
    res = ta.analyze_texture(ds, _comps())
    assert 1.0 <= res.texture_index < 1.2
    assert -0.2 < res.entropy <= 0.0


def test_index_and_entropy_use_the_same_pixels_as_the_component_fractions(world):
    ds, _, _, _ = world
    from analysis.texture_odf import texture_index_and_entropy
    res = ta.analyze_texture(ds, _comps())
    mask = np.asarray(ds.xmap.phase_id) == 1                 # the matrix, written out here
    assert int(mask.sum()) == 14
    assert (res.texture_index, res.entropy) == texture_index_and_entropy(ds, mask, halfwidth_deg=5.0)


def test_the_component_based_approximation_is_gone():
    assert not hasattr(ta, "_compute_texture_index") and not hasattr(ta, "_compute_texture_entropy")


def test_a_phase_without_finite_orientations_has_no_index_and_entropy():
    q = np.full((5, 4), np.nan)
    ds = _dataset([(0, "Al", "m-3m", q)])
    res = ta.analyze_texture(ds, _comps())
    assert res.texture_index is None and res.entropy is None
    assert res.component_fractions == {"Goss": 0.0, "Cube": 0.0, "Brass": 0.0}


# ------------------------------------------------------------------------------------------------------------------
# the workbook
# ------------------------------------------------------------------------------------------------------------------
def _sheet1(tmp_path, results):
    from analysis.excel_exporter import ExcelExporter
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ex.export_dataset("ds", {"basic_info": {}, **results})
    return ex.wb[ex.wb.sheetnames[0]]


def test_sheet1_rows_25_26_carry_the_kernel_density_numbers_and_row_67_the_phase(tmp_path, world):
    from analysis.excel_exporter import TEXTURE_ENTROPY_LABEL, TEXTURE_INDEX_LABEL
    ds, _, _, _ = world
    res = ta.analyze_texture(ds, _comps())
    ws = _sheet1(tmp_path, {"texture": res})
    assert ws.cell(row=25, column=1).value == TEXTURE_INDEX_LABEL and "norm(odf)^2" in TEXTURE_INDEX_LABEL
    assert ws.cell(row=26, column=1).value == TEXTURE_ENTROPY_LABEL and "5° kernel density" in TEXTURE_ENTROPY_LABEL
    assert ws.cell(row=25, column=2).value == res.texture_index
    assert ws.cell(row=26, column=2).value == res.entropy
    assert (ws.cell(row=67, column=1).value, ws.cell(row=67, column=2).value) == ("texture_phase", "Al")


def test_sheet1_writes_an_entropy_of_zero_and_leaves_a_missing_value_empty(tmp_path):
    from types import SimpleNamespace
    zero = SimpleNamespace(texture_index=1.0, entropy=0.0, component_fractions={}, plane_111_fraction=0.0,
                           plane_100_fraction=0.0, plane_110_fraction=0.0, phase_name="Al")
    ws = _sheet1(tmp_path, {"texture": zero})
    assert ws.cell(row=26, column=2).value == 0.0            # a real 0, written as 0
    none = SimpleNamespace(texture_index=None, entropy=None, component_fractions={}, plane_111_fraction=0.0,
                           plane_100_fraction=0.0, plane_110_fraction=0.0, phase_name=None)
    ws = _sheet1(tmp_path, {"texture": none})
    assert ws.cell(row=25, column=2).value is None and ws.cell(row=26, column=2).value is None
    assert ws.cell(row=67, column=2).value is None


# ------------------------------------------------------------------------------------------------------------------
# the route
# ------------------------------------------------------------------------------------------------------------------
def _fcc():
    from analysis.texture_components import TEXTURE_PRESETS
    return TEXTURE_PRESETS["FCC_Rolling"]


@pytest.fixture
def client(world):
    from fastapi.testclient import TestClient
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    ds, _, _, _ = world
    saved = ar._dataset, dict(ar._texture_settings)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._texture_settings.clear()
        ar._texture_settings.update(saved[1])


def test_the_route_defaults_to_the_most_measured_phase_and_names_it(client, world):
    c, ar = client
    r = c.post("/api/analysis/texture")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["texture_phase"] == {"id": 1, "name": "Al"}
    want = ta.analyze_texture(world[1], _fcc())
    assert body["texture_index"] == want.texture_index and body["texture_entropy"] == want.entropy
    assert set(body["surface_planes"]) == set(body["surface_planes_family"]) == {"{111}", "{100}", "{110}"}
    assert body["surface_planes"]["{111}"] == want.plane_111_fraction
    assert body["surface_planes_family"]["{100}"] == want.plane_100_family_fraction


def test_the_route_remembers_the_choice_reports_it_in_status_and_forgets_it_on_load(client):
    c, ar = client
    assert c.get("/api/analysis/status").json()["texture_phase"] == "most_measured"
    r = c.post("/api/analysis/texture", json={"phase": 0})
    assert r.json()["texture_phase"] == {"id": 0, "name": "Si"}
    assert c.get("/api/analysis/status").json()["texture_phase"] == 0
    assert c.post("/api/analysis/texture").json()["texture_phase"]["id"] == 0          # left out = the last choice
    assert c.post("/api/analysis/texture", json={"phase": None}).json()["texture_phase"]["id"] == 0
    assert c.post("/api/analysis/texture", json={"phase": "most_measured"}).json()["texture_phase"]["id"] == 1
    assert c.get("/api/analysis/status").json()["texture_phase"] == "most_measured"
    c.post("/api/analysis/texture", json={"phase": 0})
    assert ar._texture_settings["phase"] == 0
    ar.restore_dataset_for_file("nothing-stashed-for-this-file")        # a file switch replaces the map and resets
    assert ar._texture_settings["phase"] == "most_measured"


@pytest.mark.parametrize("bad", ["Al", 1.5, True, [1], {"id": 1}])
def test_the_route_refuses_other_types_with_422(client, bad):
    c, ar = client
    assert c.post("/api/analysis/texture", json={"phase": bad}).status_code == 422
    assert ar._texture_settings["phase"] == "most_measured"


def test_the_route_answers_400_for_an_unknown_or_a_non_cubic_phase_and_keeps_the_last_choice(client):
    c, ar = client
    c.post("/api/analysis/texture", json={"phase": 0})
    r = c.post("/api/analysis/texture", json={"phase": 9})
    assert r.status_code == 400 and "[0, 1, 2]" in r.json()["detail"]
    r = c.post("/api/analysis/texture", json={"phase": 2})
    assert r.status_code == 400 and "not cubic" in r.json()["detail"] and "1 (Al)" in r.json()["detail"]
    assert ar._texture_settings["phase"] == 0


def test_the_route_says_choose_a_cubic_phase_when_the_default_is_not_cubic(client):
    c, ar = client
    ar._dataset = _dataset([(0, "Si", "m-3m", _random_quats(3, 1)), (1, "Mg2", "6/mmm", _random_quats(9, 2))])
    r = c.post("/api/analysis/texture")
    assert r.status_code == 400 and "Choose a cubic phase" in r.json()["detail"]


def _grid_xmap(ny, nx, phase_of_block, seed):
    """A ny x nx map of 4 x 4 pixel blocks; every block has its own orientation and the phase phase_of_block[i]."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    n_bx = nx // 4
    block_q = _random_quats((ny // 4) * n_bx, seed)
    q = np.zeros((ny, nx, 4))
    ph = np.zeros((ny, nx), int)
    for b in range(len(block_q)):
        r0, c0 = (b // n_bx) * 4, (b % n_bx) * 4
        q[r0:r0 + 4, c0:c0 + 4] = block_q[b]
        ph[r0:r0 + 4, c0:c0 + 4] = phase_of_block[b]
    return CrystalMap(rotations=Rotation(q.reshape(-1, 4)), phase_id=ph.reshape(-1),
                      x=np.tile(np.arange(nx, dtype=float), ny), y=np.repeat(np.arange(ny, dtype=float), nx),
                      phase_list=PhaseList(names=["Si", "Al"], point_groups=["m-3m", "m-3m"], ids=[0, 1]),
                      prop={"bc": np.random.default_rng(seed).uniform(20, 200, nx * ny)})


def test_the_export_follows_the_remembered_phase_and_names_it_in_row_67(tmp_path):
    from fastapi.testclient import TestClient
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    from analysis.excel_exporter import ExcelExporter
    from analysis.ebsd_dataset import EBSDDataset
    from analysis.grain_analysis import reconstruct_grains
    ds = EBSDDataset(_grid_xmap(8, 12, [0, 0, 1, 1, 1, 1], seed=21), step_size=1.0)      # Al (id 1) is the larger
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
    saved = ar._dataset, dict(ar._texture_settings)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        names = []
        for body in (None, {"phase": 0}, {"phase": "most_measured"}):
            if body is not None:
                assert TestClient(app).post("/api/analysis/texture", json=body).status_code == 200
            warnings = []
            results = ar._collect_export_results(warnings)
            assert "texture" in results, warnings
            ex = ExcelExporter(str(tmp_path / "d.xlsx"))
            ex.export_dataset("ds", results)
            ws = ex.wb[ex.wb.sheetnames[0]]
            names.append((results["texture"].phase_id, ws.cell(row=67, column=2).value))
        assert names == [(1, "Al"), (0, "Si"), (1, "Al")]
    finally:
        ar._dataset = saved[0]
        ar._texture_settings.clear()
        ar._texture_settings.update(saved[1])


# ------------------------------------------------------------------------------------------------------------------
# the batch
# ------------------------------------------------------------------------------------------------------------------
def test_the_batch_resolves_the_phase_per_file_and_writes_it_per_column(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    from analysis.excel_exporter import ExcelExporter
    from tests.batch_load_stub import patch_batch_load
    maps = {"a": _grid_xmap(8, 12, [0, 0, 1, 1, 1, 1], seed=31),          # Al (1) has the most pixels
            "b": _grid_xmap(8, 12, [0, 0, 0, 0, 1, 1], seed=32)}          # Si (0) has the most pixels
    patch_batch_load(monkeypatch, bp, lambda path: maps[Path(path).stem])
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"))
    assert proc.texture_phase == "most_measured"
    ex = ExcelExporter(str(tmp_path / "batch.xlsx"))
    got = []
    for stem in ("a", "b"):
        results = proc.process_file(tmp_path / f"{stem}.h5")
        got.append((results["texture"].phase_id, results["texture"].phase_name))
        split = results["texture_by_rx"]                              # the split counts the same phase's pixels
        assert split["n_rx_pixels"] + split["n_substructure_pixels"] == 64
        ex.export_dataset(stem, results)
    assert got == [(1, "Al"), (0, "Si")]
    ws = ex.wb[ex.wb.sheetnames[0]]
    assert [ws.cell(row=67, column=c).value for c in (2, 3)] == ["Al", "Si"]
    # an explicit id is the same phase in every file
    fixed = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), texture_phase=0)
    assert [fixed.process_file(tmp_path / f"{s}.h5")["texture"].phase_id for s in ("a", "b")] == [0, 0]


@pytest.mark.parametrize("bad", ["Al", True, 1.5, None, "bogus"])
def test_the_batch_refuses_a_texture_phase_that_is_neither_most_measured_nor_an_id(tmp_path, bad):
    from analysis.batch_processor import BatchProcessor
    with pytest.raises(ValueError, match="texture_phase"):
        BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), texture_phase=bad)


def test_the_batch_route_passes_the_texture_phase_only_when_given(client, monkeypatch):
    c, _ = client
    import analysis.batch_processor as bp
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder", lambda folder, out, progress_callback=None, **kw:
                        seen.append(kw) or bp.BatchResult(processed=["a.h5"], n_files=1))
    for body in ({}, {"texture_phase": "most_measured"}, {"texture_phase": 2}):
        r = c.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o", **body})
        assert r.status_code == 200
        c.get(f"/api/analysis/task/{r.json()['task_id']}")
    assert seen == [{}, {"texture_phase": "most_measured"}, {"texture_phase": 2}]
    assert c.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o",
                                               "texture_phase": "Al"}).status_code == 422


# ------------------------------------------------------------------------------------------------------------------
# real data (slow)
# ------------------------------------------------------------------------------------------------------------------
SAMPLE_B = "EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina"


@pytest.mark.slow
def test_sampleB_through_analyze_texture_equals_the_module_on_the_same_mask():
    from analysis.ebsd_dataset import EBSDDataset
    from analysis.texture_components import TEXTURE_PRESETS
    from analysis.texture_odf import texture_index_and_entropy
    from analysis.xmap_loader import load_xmap_file
    from tests.data_deps import TEST_DATA_DIR, first_missing
    path = Path(os.environ.get("ORIENTA_TEST_DATA") or TEST_DATA_DIR) / SAMPLE_B
    if first_missing(path) is not None:
        pytest.skip(f"needs {path} (set ORIENTA_TEST_DATA to a Test_data directory that has it)")
    xmap, step, _meta = load_xmap_file(str(path))
    ds = EBSDDataset(xmap, step_size=step)
    res = ta.analyze_texture(ds, TEXTURE_PRESETS["FCC_Rolling"])
    mask = np.asarray(ds.xmap.phase_id) == res.phase_id
    assert (res.texture_index, res.entropy) == texture_index_and_entropy(ds, mask, halfwidth_deg=5.0)
    if int(mask.sum()) != 8593:
        pytest.skip(f"the texture phase holds {int(mask.sum())} pixels, not all 8593 indexed ones; "
                    "the MTEX value (25.573309) is for all of them")
    assert res.texture_index == pytest.approx(25.573309, rel=1e-3)
