"""alpha and fill reach the analysis from every entry point; KAM and IPF maps use the grains."""
import base64
import io

import numpy as np
import pytest
from tests.batch_load_stub import patch_batch_load
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset
from tests.synthetic_grain_map import make_map, scratch_mask, to_xmap


def _is_unindexed_grey(pixels):
    """compute_ipf_colors paints phase_id -1 in #44475a (68, 71, 90)."""
    return np.all(np.abs(pixels.astype(int) - np.array([68, 71, 90])) <= 1, axis=1)


@pytest.fixture
def client_with_map():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    m = make_map()
    band = scratch_mask(m["phase_id"].shape, (3, 0), (56, 79), 3.0)
    m["phase_id"][band] = -1
    saved = ar._dataset
    ar._dataset = EBSDDataset(to_xmap(m), step_size=m["step"])
    try:
        yield TestClient(app), ar, band
    finally:
        ar._dataset = saved


def test_request_model_has_alpha_and_fill():
    from backend.api.routes.analysis import GrainReconstructionRequest
    fields = GrainReconstructionRequest.model_fields
    assert fields["alpha"].default == 3.1 and fields["fill"].default is True


def test_negative_alpha_is_refused(client_with_map):
    client, _, _ = client_with_map
    r = client.post("/api/analysis/grains/reconstruct", json={"alpha": -1})
    assert r.status_code == 422


def test_route_passes_alpha_and_fill(client_with_map, monkeypatch):
    client, ar, _ = client_with_map
    seen = {}
    import analysis.grain_analysis as ga
    real = ga.reconstruct_grains

    def spy(dataset, **kw):
        seen.update(kw)
        return real(dataset, **kw)

    monkeypatch.setattr(ga, "reconstruct_grains", spy)
    r = client.post("/api/analysis/grains/reconstruct",
                    json={"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 2.0, "fill": False})
    assert r.status_code == 200
    assert seen["alpha"] == 2.0 and seen["fill"] is False
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed"
    assert task["result"]["alpha"] == 2.0 and task["result"]["fill"] is False
    assert task["result"]["n_filled"] == 0


def test_route_reports_filled_pixels(client_with_map):
    client, ar, _ = client_with_map
    r = client.post("/api/analysis/grains/reconstruct",
                    json={"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 3.1, "fill": True})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed"
    assert task["result"]["n_filled"] == int(ar._dataset.grains.filled_mask.sum()) > 0


def test_ipf_map_shows_filled_pixels(client_with_map):
    client, ar, band = client_with_map
    from analysis.grain_analysis import reconstruct_grains
    from PIL import Image
    ar._dataset.grains = reconstruct_grains(ar._dataset, min_pixels=1, fill=True)
    r = client.get("/api/analysis/map/ipf_z")
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(r.json()["image"]))))
    filled = ar._dataset.grains.filled_mask
    assert filled.any()
    assert _is_unindexed_grey(rgb[filled]).mean() < 0.05   # filled pixels are coloured, not grey


def test_ipf_map_without_fill_leaves_closed_pixels_grey(client_with_map):
    client, ar, band = client_with_map
    from analysis.grain_analysis import reconstruct_grains
    from PIL import Image
    ar._dataset.grains = reconstruct_grains(ar._dataset, min_pixels=1, fill=False)
    assert ar._dataset.grains.filled_mask is None
    r = client.get("/api/analysis/map/ipf_z")
    assert r.status_code == 200
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(r.json()["image"]))))
    assert _is_unindexed_grey(rgb[band]).mean() > 0.9      # unindexed pixels stay grey


def test_kam_map_passes_the_grains(client_with_map, monkeypatch):
    client, ar, _ = client_with_map
    from analysis.grain_analysis import reconstruct_grains
    import analysis.deformation_analysis as da
    ar._dataset.grains = reconstruct_grains(ar._dataset, min_pixels=1, fill=False)
    seen = {}
    real = da.calculate_kam

    def spy(dataset, **kw):
        seen.update(kw)
        return real(dataset, **kw)

    monkeypatch.setattr(da, "calculate_kam", spy)
    r = client.get("/api/analysis/map/kam")
    assert r.status_code == 200
    assert seen["grains"] is ar._dataset.grains


def test_batch_passes_alpha(tmp_path, monkeypatch):
    """grain_alpha reaches reconstruct_grains and is recorded in basic_info."""
    from analysis import batch_processor as bp
    import analysis.grain_analysis as ga

    xmap = to_xmap(make_map(shape=(30, 40), n_seeds=10))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)

    seen = {}
    real = ga.reconstruct_grains

    def spy(dataset, **kw):
        seen.update(kw)
        return real(dataset, **kw)

    monkeypatch.setattr(ga, "reconstruct_grains", spy)

    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), grain_alpha=1.5)
    assert proc.grain_alpha == 1.5
    results = proc.process_file(tmp_path / "sample.h5")
    assert results is not None
    assert seen["alpha"] == 1.5
    assert results["basic_info"]["grain_alpha"] == 1.5
    assert bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx")).grain_alpha == 3.1


def test_excel_sheet1_row48_is_grain_alpha(tmp_path):
    from analysis.excel_exporter import ExcelExporter, PLANE_LABELS
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ws = ex.wb[ex.wb.sheetnames[0]]
    assert ws.cell(row=48, column=1).value == "grain_alpha"
    assert ws.cell(row=47, column=1).value == PLANE_LABELS[2]


def test_excel_export_dataset_writes_grain_alpha(tmp_path):
    from analysis.excel_exporter import ExcelExporter
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ex.export_dataset("ds", {"basic_info": {"grain_alpha": 1.5}})
    ws = ex.wb[ex.wb.sheetnames[0]]
    assert ws.cell(row=48, column=2).value == 1.5
