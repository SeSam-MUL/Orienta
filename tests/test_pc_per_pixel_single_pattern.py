"""Single-pattern paths of PC Refinement with a per-pixel PC detector.

After a grid calibration ("Pixel-wise PC correction") the controller's detector
holds one PC per scan pixel, shape (rows, cols, 3). PyEBSDIndex takes ONE PC for
ONE pattern, so "Index Pattern", the overlay, "Index All", "Refine PC" and a
second grid run all have to pick the PC of the pattern's own pixel. They used to
hand the whole map to PyEBSDIndex and fail with
``indexer.PC must be an array of shape (3,) or (1, 3), but was (N, 3)``.

The per-pixel PCs here are those of kikuchipy's small nickel dataset
(3 x 3 patterns, 60 x 60 px). Their numbers are written out below as the
dataset's own header values (rounded to 4 decimals as kikuchipy prints them), so
the expectations do not come from the code under test.
"""
import logging

import numpy as np
import pytest

from pc_controller import PCController, pc_is_per_pixel

# (row, col) -> the dataset's PC at that pixel (Bruker), read off the dataset
PIXEL_PCS = [
    [(0.4215, 0.2150, 0.5020), (0.4241, 0.2101, 0.5010), (0.4264, 0.2115, 0.5004)],
    [(0.4209, 0.2165, 0.5008), (0.4273, 0.2145, 0.5000), (0.4308, 0.2137, 0.5012)],
    [(0.4219, 0.2120, 0.4997), (0.4309, 0.2144, 0.5007), (0.4225, 0.2126, 0.5005)],
]
PC_ATOL = 6e-5   # the header values above are rounded to 4 decimals


def _ni_cif(tmp_path):
    cif = tmp_path / "Ni.cif"
    cif.write_text(
        "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding="utf-8")
    return str(cif)


@pytest.fixture(scope="module")
def nickel():
    import kikuchipy as kp
    return kp.data.nickel_ebsd_small()


def _controller(nickel, tmp_path, positions):
    """A controller holding the nickel phase, the per-pixel detector of the
    dataset and calibration patterns at ``positions``."""
    ctrl = PCController()
    ctrl.load_phase(_ni_cif(tmp_path))
    ctrl.apply_pixelwise_pc(nickel.detector.deepcopy())
    for (r, c) in positions:
        ctrl.add_pattern((r, c), np.asarray(nickel.data[r, c]).copy())
    return ctrl


def _indexer_pc(ctrl):
    return np.asarray(ctrl.indexer.PC, dtype=float).ravel()


def test_the_fixture_is_a_per_pixel_detector(nickel):
    """Guard the premise: the dataset detector carries 9 PCs and its pixel (1, 2)
    is the value written above."""
    assert pc_is_per_pixel(nickel.detector)
    np.testing.assert_allclose(np.asarray(nickel.detector.pc)[1, 2], PIXEL_PCS[1][2],
                               atol=PC_ATOL)


def test_index_pattern_uses_the_pc_of_its_own_pixel(nickel, tmp_path):
    ctrl = _controller(nickel, tmp_path, [(1, 2), (2, 0)])
    ci, _, band_data = ctrl.index_pattern(0)          # used to raise ValueError
    assert ci > 0 and band_data is not None
    np.testing.assert_allclose(_indexer_pc(ctrl), PIXEL_PCS[1][2], atol=PC_ATOL)
    # another pixel: the indexer is rebuilt for ITS PC, not reused at the first one
    ctrl.index_pattern(1)
    np.testing.assert_allclose(_indexer_pc(ctrl), PIXEL_PCS[2][0], atol=PC_ATOL)
    # the controller's own detector is still the per-pixel one
    assert pc_is_per_pixel(ctrl.detector)


def test_route_index_and_simulate_draws_lines_for_the_pixels_own_pc(nickel, tmp_path):
    """The overlay segments of a pattern indexed with a PC map are those of a
    plain single-PC detector set to that pixel's PC (the long-standing path)."""
    from kikuchipy.detectors import EBSDDetector
    from backend.api.routes import pcrefinement as R

    ctrl = _controller(nickel, tmp_path, [(1, 2), (2, 0)])
    for idx, (r, c) in enumerate([(1, 2), (2, 0)]):
        got = R._index_and_simulate(ctrl, idx)
        # the result says which PC the pattern was indexed at: its own pixel's
        np.testing.assert_allclose(got["pc"], PIXEL_PCS[r][c], atol=PC_ATOL)

        ref = PCController()
        ref.load_phase(_ni_cif(tmp_path))
        ref.attach_detector(EBSDDetector(shape=(60, 60), pc=PIXEL_PCS[r][c],
                                         sample_tilt=70.0))
        ref.add_pattern((r, c), np.asarray(nickel.data[r, c]).copy())
        want = R._index_and_simulate(ref, 0)

        assert got["n_bands"] > 0
        assert got["n_bands"] == want["n_bands"]
        # the header PCs above are rounded to 4 decimals: 0.05 px of line position
        np.testing.assert_allclose(np.asarray(got["segments"]),
                                   np.asarray(want["segments"]), atol=0.05)

        # sensitivity: lines drawn for the MAP MEAN differ (another band count or >0.1 px)
        mean_pc = np.asarray(PIXEL_PCS, dtype=float).reshape(-1, 3).mean(axis=0)
        wrong = PCController()
        wrong.load_phase(_ni_cif(tmp_path))
        wrong.attach_detector(EBSDDetector(shape=(60, 60), pc=mean_pc, sample_tilt=70.0))
        wrong.add_pattern((r, c), np.asarray(nickel.data[r, c]).copy())
        off = R._index_and_simulate(wrong, 0)
        a, b = np.asarray(off["segments"]), np.asarray(want["segments"])
        assert a.shape != b.shape or np.abs(a - b).max() > 0.1


def test_phases_of_patterns_follow_the_pixels(nickel, tmp_path):
    ctrl = _controller(nickel, tmp_path, [])
    pats = [np.asarray(nickel.data[1, 2]), np.asarray(nickel.data[2, 0])]
    out = ctrl.phases_of_patterns(pats, coords=[(1, 2), (2, 0)])
    assert [o["phase_name"] for o in out] == ["Ni", "Ni"]
    np.testing.assert_allclose(_indexer_pc(ctrl), PIXEL_PCS[2][0], atol=PC_ATOL)


# --- the PC of a pattern whose position is not known ------------------------

def test_unknown_position_uses_the_map_mean_and_says_so(nickel, tmp_path, caplog):
    ctrl = _controller(nickel, tmp_path, [])
    mean = np.asarray(PIXEL_PCS, dtype=float).reshape(-1, 3).mean(axis=0)
    with caplog.at_level(logging.WARNING, logger="pc_controller"):
        pc = ctrl.pc_at(None)
    np.testing.assert_allclose(pc, mean, atol=PC_ATOL)
    assert "position unknown" in caplog.text


def test_position_outside_the_map_uses_the_map_mean_and_says_so(nickel, tmp_path, caplog):
    ctrl = _controller(nickel, tmp_path, [])
    mean = np.asarray(PIXEL_PCS, dtype=float).reshape(-1, 3).mean(axis=0)
    with caplog.at_level(logging.WARNING, logger="pc_controller"):
        pc = ctrl.pc_at((7, 1))
    np.testing.assert_allclose(pc, mean, atol=PC_ATOL)
    assert "outside" in caplog.text


def _warnings_about_position(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "outside" in r.getMessage()]


def test_a_pattern_off_the_map_is_reported_once_not_twice(nickel, tmp_path, caplog):
    """Looking the pattern's detector up and then building the indexer for it used
    to place the position twice, so the same warning came out twice."""
    ctrl = _controller(nickel, tmp_path, [])
    pat = np.asarray(nickel.data[1, 2])
    ctrl.add_pattern((7, 1), pat.copy())                     # a position off the 3 x 3 map
    with caplog.at_level(logging.WARNING, logger="pc_controller"):
        ctrl.index_pattern(0)
    assert len(_warnings_about_position(caplog)) == 1
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pc_controller"):
        ctrl.phases_of_patterns([pat], coords=[(7, 1)])
    assert len(_warnings_about_position(caplog)) == 1


def test_detector_at_is_a_single_pc_copy_and_leaves_the_map_alone(nickel, tmp_path):
    ctrl = _controller(nickel, tmp_path, [])
    det = ctrl.detector_at((1, 2))
    assert not pc_is_per_pixel(det)
    np.testing.assert_allclose(np.asarray(det.pc).ravel(), PIXEL_PCS[1][2], atol=PC_ATOL)
    # everything else of the geometry is carried over
    assert tuple(det.shape) == (60, 60)
    assert det.sample_tilt == ctrl.detector.sample_tilt
    assert det.tilt == ctrl.detector.tilt
    assert det.binning == ctrl.detector.binning
    assert det.px_size == ctrl.detector.px_size
    # and the per-pixel detector is untouched
    assert np.asarray(ctrl.detector.pc).shape == (3, 3, 3)


def test_a_single_pc_detector_is_returned_as_it_is():
    from kikuchipy.detectors import EBSDDetector
    ctrl = PCController()
    ctrl.detector = EBSDDetector(shape=(60, 60), pc=(0.5, 0.4, 0.6), sample_tilt=70.0)
    assert ctrl.detector_at((3, 3)) is ctrl.detector
    assert ctrl.detector_at(None) is ctrl.detector


# --- the optimiser starts at each pattern's own pixel -----------------------

def test_global_refine_starts_at_the_mean_of_its_patterns_pixels_and_ends_single(
        nickel, tmp_path, monkeypatch):
    from backend.api.routes import pcrefinement as R
    from tests.pc_engine_stub import stub_search

    ctrl = _controller(nickel, tmp_path, [(1, 2), (2, 0)])
    calls = stub_search(monkeypatch, [(0.4301, 0.2141, 0.5009)] * 2)
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    task = "t-global"
    R._optimization_tasks[task] = {"status": "running", "progress": 0.0,
                                   "result": None, "error": None}
    R._optimization_active = True
    pats = [p for _, p in ctrl.patterns]
    R._run_optimization(task, pats, "PSO", 0.05, [(1, 2), (2, 0)])

    state = R._optimization_tasks.pop(task)
    assert state["status"] == "completed", state.get("error")
    # ONE search over both patterns; it starts at the PC the detector has at THEIR pixels (the mean
    # of the two), not at the corner pixel of the map and not at the mean of the whole map
    assert len(calls) == 1
    expected = (np.array(PIXEL_PCS[1][2]) + np.array(PIXEL_PCS[2][0])) / 2
    np.testing.assert_allclose(calls[0]["start"], expected, atol=PC_ATOL)
    whole_map = np.asarray(nickel.detector.pc).reshape(-1, 3).mean(axis=0)
    assert np.abs(np.asarray(calls[0]["start"]) - whole_map).max() > 5 * PC_ATOL
    # one global PC came out: the controller's detector is now the single one
    assert not pc_is_per_pixel(ctrl.detector)
    np.testing.assert_allclose(np.asarray(ctrl.detector.pc).ravel(),
                               (0.4301, 0.2141, 0.5009), atol=1e-9)
    results = state["result"]["pattern_results"]
    assert len(results) == 2                                # re-indexed, no crash
    # every pattern carries ITS OWN lines, so the page can draw the selected one
    # (the run's top-level `segments` are the first pattern's only)
    assert results[0]["segments"] and results[1]["segments"]
    assert results[0]["segments"] != results[1]["segments"]
    assert results[0]["segments"] == state["result"]["segments"]
    again = R._index_and_simulate(ctrl, 1)["segments"]
    np.testing.assert_allclose(np.asarray(results[1]["segments"]), np.asarray(again), atol=1e-9)


def test_a_second_grid_run_starts_each_point_at_its_own_pc(nickel, tmp_path, monkeypatch):
    import pc_controller

    ctrl = _controller(nickel, tmp_path, [])
    starts = []

    def fake_optimize(detector, indexer, pattern, method="PSO", search_limit=0.05,
                      batch=True):
        starts.append(np.asarray(detector.pc, dtype=float).ravel())
        return (0.43, 0.21, 0.50), None

    monkeypatch.setattr(pc_controller, "optimize_pc", fake_optimize)
    out = ctrl.refine_pc_at_points(nickel, [(0, 0), (1, 2), (2, 0)])
    assert out.shape == (3, 3)
    np.testing.assert_allclose(starts[0], PIXEL_PCS[0][0], atol=PC_ATOL)
    np.testing.assert_allclose(starts[1], PIXEL_PCS[1][2], atol=PC_ATOL)
    np.testing.assert_allclose(starts[2], PIXEL_PCS[2][0], atol=PC_ATOL)


def test_refine_global_pc_starts_each_pattern_at_its_pixel(nickel, tmp_path, monkeypatch):
    import pc_controller

    ctrl = _controller(nickel, tmp_path, [(1, 2), (2, 0)])
    starts = []

    def fake_optimize(detector, indexer, pattern, method="PSO", search_limit=0.05, batch=True):
        starts.append(np.asarray(detector.pc, dtype=float).ravel())
        return (0.4301, 0.2141, 0.5009), None

    monkeypatch.setattr(pc_controller, "optimize_pc", fake_optimize)
    pc = ctrl.refine_global_pc()
    np.testing.assert_allclose(starts[0], PIXEL_PCS[1][2], atol=PC_ATOL)
    np.testing.assert_allclose(starts[1], PIXEL_PCS[2][0], atol=PC_ATOL)
    np.testing.assert_allclose(pc, (0.4301, 0.2141, 0.5009), atol=1e-9)
    assert not pc_is_per_pixel(ctrl.detector)                 # one global PC came out


def test_the_phase_of_each_pattern_after_a_refine_is_looked_up_at_its_pixel(nickel, tmp_path, monkeypatch):
    """With several phases and optimised patterns that are not the calibration list,
    the per-pattern phase and CI are indexed at the PC of each pixel, not the map mean."""
    import ebsd_utils
    from backend.api.routes import pcrefinement as R

    ctrl = PCController()
    ctrl.apply_pixelwise_pc(nickel.detector.deepcopy())
    base = open(_ni_cif(tmp_path), encoding="utf-8").read()
    for name, a in (("aus", "3.52"), ("fer", "2.87")):
        cif = tmp_path / f"{name}.cif"
        cif.write_text(base.replace("data_Ni", f"data_{name}").replace("3.52", a), encoding="utf-8")
        ctrl.add_phase(str(cif))
    asked = {}

    def fake_phases(patterns, coords=None):
        asked["coords"] = coords
        return [{"phase_name": None, "phase_index": None, "ci": None, "phase_fits": None}
                for _ in patterns]

    monkeypatch.setattr(ctrl, "phases_of_patterns", fake_phases)
    from tests.pc_engine_stub import stub_search
    stub_search(monkeypatch, [(0.43, 0.21, 0.5)] * 2)
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    task = "t-phases"
    R._optimization_tasks[task] = {"status": "running", "progress": 0.0, "result": None, "error": None}
    R._optimization_active = True
    pats = [np.asarray(nickel.data[1, 2]), np.asarray(nickel.data[2, 0])]
    R._run_optimization(task, pats, "PSO", 0.05, [(1, 2), (2, 0)])
    state = R._optimization_tasks.pop(task)
    assert state["status"] == "completed", state.get("error")
    assert asked["coords"] == [(1, 2), (2, 0)]
