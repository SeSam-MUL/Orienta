"""GOS and gKAM histograms weigh each grain by its AREA (cells, closed holes included), as MTEX's grain histogram.

Measured with MTEX: two 20 x 20 grains, one with a
2 x 2 closed not-indexed hole, give bars of 50 / 50 (areas 400 / 400), not 49.75 / 50.25 (measurements 396 / 400).
"""
import numpy as np
import pytest
from openpyxl import load_workbook

from analysis.ebsd_dataset import EBSDDataset
from analysis.excel_exporter import _area_percent, export_to_excel
from analysis.grain_analysis import reconstruct_grains
from analysis.rx_analysis import analyze_recrystallization, calculate_area_weighted_histogram
from tests.synthetic_grain_map import _axis_angle, _qmul, random_quaternions, to_xmap


def _probe():
    """Two 20 x 20 grains side by side, a 2 x 2 hole in the first (alpha 3.1 closes it). The second grain
    turns by 1 degree per column, so the two grains have different GOS and gKAM."""
    ny, nx = 20, 40
    labels = np.zeros((ny, nx), int)
    labels[:, :20] = 1
    labels[:, 20:] = 2
    labels[9:11, 9:11] = 0
    base = random_quaternions(np.random.default_rng(5), 3)
    q = np.empty((ny, nx, 4))
    q[:] = np.array([1.0, 0.0, 0.0, 0.0])
    q[labels == 1] = base[1]
    cols = np.broadcast_to(np.arange(nx), (ny, nx))
    turn = _axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(1.0 * (cols - 20)))
    q2 = _qmul(turn, np.broadcast_to(base[2], (ny, nx, 4)))
    q[labels == 2] = q2[labels == 2]
    m = {"quats": q, "phase_id": np.where(labels > 0, 0, -1).astype(np.int64),
         "phase_names": np.array(["Al", "Al7FeCu2"]), "point_groups": np.array(["m-3m", "4/mmm"]), "step": 1.0}
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.where(labels.ravel() > 0, 150.0, 0.0)
    ds = EBSDDataset(xmap, step_size=1.0)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=3.1, fill=False)
    return ds


def _rx(ds):
    from analysis.deformation_analysis import calculate_kam
    return analyze_recrystallization(ds, ds.grains, calculate_kam(ds, grains=ds.grains))


def test_the_probe_has_the_geometry_of_the_measurement():
    g = _probe().grains
    assert list(g.area) == [400.0, 400.0] and list(g.grain_size) == [396, 400]       # cells vs measurements


def test_the_gos_histogram_weighs_by_cell_area_and_is_50_50():
    rx = _rx(_probe())
    counts = rx.gos_histogram[0]
    assert sorted(counts[counts > 0]) == [400.0, 400.0]                              # areas, not 396 / 400
    assert rx.total_grain_area == 800.0
    pct = _area_percent(rx, counts, "GOS")
    assert sorted(pct[pct > 0]) == pytest.approx([50.0, 50.0])
    assert not hasattr(rx, "total_grain_pixels")


def test_the_gkam_histogram_weighs_by_cell_area_and_is_50_50():
    rx = _rx(_probe())
    pct = _area_percent(rx, rx.gkam_histogram[0], "gKAM")
    assert sorted(pct[pct > 0]) == pytest.approx([50.0, 50.0])


def test_the_workbook_sheets_show_50_50(tmp_path):
    ds = _probe()
    out = tmp_path / "w.xlsx"
    export_to_excel(str(out), "probe", {"rx": _rx(ds)})
    wb = load_workbook(out)
    for sheet, last in (("GOS hist, area weighted", 62), ("gKAM - area weighted", 27)):
        col = [wb[sheet].cell(row=r, column=2).value or 0.0 for r in range(3, last + 1)]
        assert sorted(v for v in col if v) == pytest.approx([50.0, 50.0]), sheet


def test_a_grain_outside_the_bins_stays_in_the_denominator():
    # areas 400 / 100 / 100: the 100 area grain with GOS 20 deg is out of the 0-15 deg range, not renormalised
    counts, _ = calculate_area_weighted_histogram(np.array([1.0, 2.0, 20.0]), np.array([400.0, 100.0, 100.0]),
                                                  np.arange(0, 15.25, 0.25))
    assert counts.sum() == 500.0

    class R:
        total_grain_area = 600.0
    assert _area_percent(R, counts, "GOS").sum() == pytest.approx(500.0 / 6.0)
