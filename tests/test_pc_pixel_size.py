"""The pixel size reaches the PC controller's detector.

kikuchipy turns a beam step into a PC shift with ``px_size * binning``
(``extrapolate_pc``). The PC page's detector was built from the calibration store,
which keeps no pixel size, so ``px_size`` stayed at kikuchipy's default of 1 um: an
extrapolated PC drifted by 0.56 of the detector width over a 45 um map instead of
0.0015. "Apply pixel size" posted ``{binning, detector_width_mm}`` to a route whose
model had no such fields, and ``/detector/set`` stored the pixel size under an
attribute kikuchipy never reads.

Expected numbers are arithmetic: width = columns * binning * px_size.
"""
import asyncio

import numpy as np
import pytest
from kikuchipy.detectors import EBSDDetector

from pc_controller import PCController

SHAPE = (60, 80)                                    # rows, cols
POINTS = [(0, 0), (0, 9), (9, 0), (9, 9), (4, 4)]
REFINED = np.array([[0.50, 0.40, 0.60]] * 5)


@pytest.fixture
def ctrl(monkeypatch):
    from backend.api.routes import pcrefinement as R
    c = PCController()
    c.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    monkeypatch.setattr(R, "_get_controller", lambda: c)
    return c


def _span(ctrl):
    det = ctrl.extrapolate_pc_from_points(POINTS, REFINED, 10, 10, step_sizes=(5.0, 5.0))
    pc = np.asarray(det.pc).reshape(-1, 3)
    return pc.max(axis=0) - pc.min(axis=0)


def test_apply_pixel_size_sets_the_detectors_pixel_size(ctrl):
    from backend.api.routes import pcrefinement as R
    res = asyncio.run(R.update_params(R.UpdateParamsRequest(binning=2, detector_width_mm=30.0)))
    # 30 mm across 80 columns of 2 x 2 binned pixels: 30000 / (80 * 2) um per unbinned pixel
    assert ctrl.detector.px_size == pytest.approx(187.5)
    assert ctrl.detector.binning == 2
    assert res["px_size_um"] == pytest.approx(187.5)


def test_the_pixel_size_decides_how_far_a_beam_step_moves_the_pc(ctrl):
    from backend.api.routes import pcrefinement as R
    before = _span(ctrl)                    # px_size 1, binning 1: width 80 um
    asyncio.run(R.update_params(R.UpdateParamsRequest(binning=2, detector_width_mm=30.0)))
    after = _span(ctrl)                     # width 30 mm: 375 times wider
    # the same beam steps move the PC 375 times less (width ratio 30000 / 80)
    np.testing.assert_allclose(before / after, 375.0, rtol=1e-6)
    assert after.max() < 0.01               # 45 um of a 30 mm detector


def test_either_value_alone_changes_nothing(ctrl):
    from backend.api.routes import pcrefinement as R
    asyncio.run(R.update_params(R.UpdateParamsRequest(binning=4)))
    asyncio.run(R.update_params(R.UpdateParamsRequest(detector_width_mm=30.0)))
    assert ctrl.detector.px_size == 1.0 and ctrl.detector.binning == 1


def test_a_nonsense_pixel_size_is_refused(ctrl):
    from fastapi import HTTPException
    from backend.api.routes import pcrefinement as R
    with pytest.raises(HTTPException):
        asyncio.run(R.update_params(R.UpdateParamsRequest(binning=2, detector_width_mm=0.0)))
    assert ctrl.detector.px_size == 1.0


def test_detector_set_stores_the_pixel_size_where_kikuchipy_reads_it(ctrl):
    from backend.api.routes import pcrefinement as R
    asyncio.run(R.set_detector(R.SetDetectorRequest(
        shape=list(SHAPE), pc=[0.5, 0.4, 0.6], binning=8, pixel_size=70.0)))
    # 70 um is the pitch of a stored (binned) pixel: 8.75 um unbinned
    assert ctrl.detector.px_size == pytest.approx(8.75)
    assert ctrl.detector.binning == 8


def test_the_controller_starts_with_the_pixel_size_of_the_file(monkeypatch):
    """The store has no pixel size; the file's detector does."""
    from types import SimpleNamespace

    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from backend.api.services.calibration_store import CalibrationStore

    file_det = EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0,
                            px_size=59.2, binning=8)
    store = CalibrationStore()
    store.register("scan", SimpleNamespace(detector=file_det))
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "scan")
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: SimpleNamespace(detector=file_det))

    c = PCController()
    R._auto_attach_detector(c)
    assert c.detector.px_size == pytest.approx(59.2)
    assert c.detector.binning == 8


def test_a_detector_of_another_shape_is_not_given_the_files_pixel_size(monkeypatch):
    from types import SimpleNamespace

    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from backend.api.services.calibration_store import CalibrationStore

    store = CalibrationStore()
    store.register("scan", SimpleNamespace(
        detector=EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0)))
    other = EBSDDetector(shape=(30, 30), pc=(0.5, 0.4, 0.6), px_size=59.2, binning=8)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "scan")
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: SimpleNamespace(detector=other))
    c = PCController()
    R._auto_attach_detector(c)
    assert c.detector.px_size == 1.0 and c.detector.binning == 1


# --- extrapolate without a pixel size is refused, not run at 1 um ------------------

def _grid(ctrl, monkeypatch, mode, signal_ok=True):
    from fastapi import BackgroundTasks
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    ctrl.load_phase(_ni(ctrl))
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal",
                        lambda *a, **k: type("S", (), {"data": np.zeros((4, 4) + SHAPE, np.uint8)})())
    monkeypatch.setattr(R, "_optimization_active", False)
    bt = BackgroundTasks()
    return asyncio.run(R.calibrate_grid(R.GridCalibrationRequest(grid_step=3, mode=mode), bt)), bt


def _ni(ctrl):
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "Ni.cif")
    open(p, "w", encoding="utf-8").write("\n".join([
        "data_Ni", "_cell_length_a 3.52", "_cell_length_b 3.52", "_cell_length_c 3.52",
        "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
        "_symmetry_space_group_name_H-M 'F m -3 m'", "_symmetry_Int_Tables_number 225",
        "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
        "_atom_site_fract_y", "_atom_site_fract_z", "Ni1 Ni 0 0 0", ""]))
    return p


def test_extrapolate_is_refused_while_the_pixel_size_is_the_default(ctrl, monkeypatch):
    from fastapi import HTTPException
    assert ctrl.detector.px_size == 1.0                # kikuchipy's default: nobody set a real one
    with pytest.raises(HTTPException) as exc:
        _grid(ctrl, monkeypatch, "extrapolate")
    assert exc.value.status_code == 400
    assert exc.value.headers["X-Orienta-Code"] == "pixel_size_unknown"
    assert "Apply Pixel Size" in exc.value.detail


def test_extrapolate_runs_once_a_pixel_size_is_set(ctrl, monkeypatch):
    from backend.api.routes import pcrefinement as R
    asyncio.run(R.update_params(R.UpdateParamsRequest(binning=2, detector_width_mm=30.0)))
    out, bt = _grid(ctrl, monkeypatch, "extrapolate")
    assert out["status"] == "running"
    R._optimization_active = False


def test_the_plane_fit_does_not_need_a_pixel_size(ctrl, monkeypatch):
    """Only the extrapolation turns a beam step into a PC shift."""
    from backend.api.routes import pcrefinement as R
    out, bt = _grid(ctrl, monkeypatch, "plane")
    assert out["status"] == "running"
    R._optimization_active = False
