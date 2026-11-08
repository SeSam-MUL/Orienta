"""What a PC written from the PC Refinement page does to a per-pixel PC map.

A grid calibration leaves a per-pixel map ("refined_map") in the controller AND in
the calibration store. A single PC written afterwards - typed into the page, or
the result of a Global PC Refine - used to

* collapse the controller's map to one value without a word (typed PC), and
* leave the store's map untouched, so the page showed the new PC while indexing
  kept reading the old map (Global refine, "refined_map outranks everything").

Now a typed PC SHIFTS the map (its mean becomes the PC, its shape is kept), a
Global refine REPLACES a grid-calibrated map of the active dataset by its one PC,
a map from the file is shifted as before, a parent copy's genuine map is spared,
and the answer says which of these happened (``pc_map_action``).

Every expected number is arithmetic on a hand-set plane.
"""
import asyncio

import numpy as np
import pytest
from kikuchipy.detectors import EBSDDetector

from backend.api.services.calibration_store import CalibrationStore
from pc_controller import PCController, pc_is_per_pixel

NR, NC = 6, 8
# PC(row, col) = (0.50 + 1e-3 col, 0.40 + 2e-3 row, 0.60)
PLANE_SLOPE = (1e-3, 2e-3)


def _plane():
    rows, cols = np.mgrid[0:NR, 0:NC]
    return np.stack([0.50 + PLANE_SLOPE[0] * cols, 0.40 + PLANE_SLOPE[1] * rows,
                     0.60 + 0 * rows], axis=-1)


def _plane_mean():
    return [0.50 + PLANE_SLOPE[0] * (NC - 1) / 2, 0.40 + PLANE_SLOPE[1] * (NR - 1) / 2, 0.60]


class _Det:
    def __init__(self, pc):
        self.shape = (60, 60)
        self.pc = np.asarray(pc)
        self.sample_tilt, self.tilt, self.azimuthal = 70.0, 0.0, 0.0


class _Sig:
    def __init__(self, pc):
        self.detector = _Det(pc)


def _store_with(name, source):
    """A store holding the plane as a header map or as a grid-calibrated one."""
    store = CalibrationStore()
    store.register(name, _Sig(_plane()))
    if source == "refined_map":
        store.update_pc_map(name, _plane())
    return store


# --- the store ----------------------------------------------------------------

NEW_PC = np.array([0.5321, 0.3987, 0.7123])


def test_refined_map_can_be_shifted_and_keeps_its_shape():
    store = _store_with("A", "refined_map")
    assert store.update_pc_detail("A", NEW_PC, source="manual", on_refined_map="shift") == "shifted"
    entry = store.get_entry("A")
    np.testing.assert_allclose(entry.pc_map.reshape(-1, 3).mean(axis=0), NEW_PC, atol=1e-12)
    # every pixel keeps its offset from the mean: the plane's own offsets
    np.testing.assert_allclose(entry.pc_map - NEW_PC, _plane() - np.asarray(_plane_mean()),
                               atol=1e-12)
    assert entry.pc_source == "refined_map"          # still a grid-fitted map
    served = np.asarray(store.get_detector("A").pc)
    np.testing.assert_allclose(served.reshape(-1, 3).mean(axis=0), NEW_PC, atol=1e-12)


def test_refined_map_can_be_replaced_by_one_pc():
    store = _store_with("A", "refined_map")
    assert store.update_pc_detail("A", NEW_PC, source="refined", on_refined_map="replace") == "replaced"
    entry = store.get_entry("A")
    assert entry.pc_map is None and entry.pc_source == "refined"
    np.testing.assert_allclose(np.asarray(store.get_detector("A").pc).ravel(), NEW_PC, atol=1e-12)


def test_refined_map_is_kept_by_default_as_before():
    store = _store_with("A", "refined_map")
    assert store.update_pc_detail("A", NEW_PC) == "kept"
    entry = store.get_entry("A")
    np.testing.assert_array_equal(entry.pc_map, _plane())
    assert entry.pc_source == "refined_map"


def test_actions_of_the_other_entries():
    header = _store_with("H", "header")
    assert header.update_pc_detail("H", NEW_PC, on_refined_map="replace") == "shifted"   # header map: shift
    store = CalibrationStore()
    store.register("S", _Sig([0.5, 0.4, 0.6]))
    assert store.update_pc_detail("S", NEW_PC) == "single"
    assert store.update_pc_detail("ghost", NEW_PC) is None


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        _store_with("A", "refined_map").update_pc_detail("A", NEW_PC, on_refined_map="drop")


def test_writeback_passes_the_policy_on_to_the_parent_and_says_what_it_did():
    """A parent that kept its old map would show the new PC while its indexing,
    export and phase tests read the old map."""
    from backend.api.routes.pcrefinement import _writeback_refined_pc
    store = _store_with("raw", "refined_map")
    store.register_derived("raw_avg", "raw")
    store.update_pc_map("raw_avg", _plane())           # the active copy has its own grid map
    done = {}
    act = _writeback_refined_pc("raw_avg", NEW_PC, store=store, on_refined_map="replace", report=done)
    assert act == "replaced"
    assert done == {"raw_avg": "replaced", "raw": "replaced"}
    for name in ("raw_avg", "raw"):
        assert store.get_entry(name).pc_map is None
        np.testing.assert_allclose(np.asarray(store.get_detector(name).pc).ravel(), NEW_PC, atol=1e-12)


def test_writeback_shifts_the_parents_grid_map_for_a_typed_pc():
    from backend.api.routes.pcrefinement import _writeback_refined_pc
    store = _store_with("raw", "refined_map")
    store.register_derived("raw_avg", "raw")
    done = {}
    _writeback_refined_pc("raw_avg", NEW_PC, store=store, source="manual",
                          on_refined_map="shift", report=done)
    assert done["raw"] == "shifted"
    raw = store.get_entry("raw")
    np.testing.assert_allclose(raw.pc_map.reshape(-1, 3).mean(axis=0), NEW_PC, atol=1e-12)
    np.testing.assert_allclose(raw.pc_map - NEW_PC, _plane() - np.asarray(_plane_mean()), atol=1e-12)


def test_the_default_still_spares_a_genuine_map_of_the_parent():
    from backend.api.routes.pcrefinement import _writeback_refined_pc
    store = _store_with("raw", "refined_map")
    store.register_derived("raw_avg", "raw")
    _writeback_refined_pc("raw_avg", NEW_PC, store=store)
    np.testing.assert_array_equal(store.get_entry("raw").pc_map, _plane())


# --- the controller -----------------------------------------------------------

def _controller_with_plane():
    c = PCController()
    c.apply_pixelwise_pc(EBSDDetector(shape=(60, 60), pc=_plane(), sample_tilt=70.0))
    return c


def test_controller_shift_keeps_the_map():
    c = _controller_with_plane()
    assert c.shift_pc_map(NEW_PC) == "shifted"
    assert pc_is_per_pixel(c.detector)
    moved = np.asarray(c.detector.pc)
    assert moved.shape == (NR, NC, 3)
    np.testing.assert_allclose(moved.reshape(-1, 3).mean(axis=0), NEW_PC, atol=1e-12)
    np.testing.assert_allclose(moved - NEW_PC, _plane() - np.asarray(_plane_mean()), atol=1e-12)


def test_controller_global_pc_replaces_the_map():
    c = _controller_with_plane()
    assert c.set_global_pc(NEW_PC) == "replaced"
    assert not pc_is_per_pixel(c.detector)
    np.testing.assert_allclose(np.asarray(c.detector.pc).ravel(), NEW_PC, atol=1e-12)


def test_controller_shift_of_a_single_pc_just_sets_it():
    c = PCController()
    c.attach_detector(EBSDDetector(shape=(60, 60), pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    assert c.shift_pc_map(NEW_PC) == "single"
    np.testing.assert_allclose(np.asarray(c.detector.pc).ravel(), NEW_PC, atol=1e-12)


# --- the routes ---------------------------------------------------------------

@pytest.fixture
def page(monkeypatch):
    """The PC page's routes on a controller + store that both hold the plane."""
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    c = _controller_with_plane()
    store = _store_with("ds", "refined_map")
    monkeypatch.setattr(R, "_get_controller", lambda: c)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    return R, c, store


def test_typed_pc_shifts_both_maps_and_says_so(page):
    R, c, store = page
    res = asyncio.run(R.update_pc(R.UpdatePCRequest(pcx=NEW_PC[0], pcy=NEW_PC[1], pcz=NEW_PC[2])))
    assert res["pc_map_action"] == "shifted"
    assert pc_is_per_pixel(c.detector)                       # nothing collapsed
    np.testing.assert_allclose(np.asarray(c.detector.pc).reshape(-1, 3).mean(axis=0), NEW_PC, atol=1e-12)
    entry = store.get_entry("ds")
    assert entry.pc_map is not None
    np.testing.assert_allclose(entry.pc_map.reshape(-1, 3).mean(axis=0), NEW_PC, atol=1e-12)


def _run_global_refine(R, c, monkeypatch, mean_pc):
    from tests.pc_engine_stub import stub_search
    c.load_phase(_ni_cif_path())
    stub_search(monkeypatch, [tuple(mean_pc)] * 2)
    task = "t-policy"
    R._optimization_tasks[task] = {"status": "running", "progress": 0.0, "result": None, "error": None}
    R._optimization_active = True
    pats = [np.zeros((60, 60), np.uint8), np.ones((60, 60), np.uint8)]
    R._run_optimization(task, pats, "PSO", 0.05, [(0, 0), (5, 7)])
    state = R._optimization_tasks.pop(task)
    assert state["status"] == "completed", state.get("error")
    return state["result"]


_CIF = None


def _ni_cif_path():
    global _CIF
    if _CIF is None:
        import tempfile, os
        d = tempfile.mkdtemp()
        _CIF = os.path.join(d, "Ni.cif")
        with open(_CIF, "w", encoding="utf-8") as f:
            f.write("data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
                    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                    "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
                    "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
                    "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n")
    return _CIF


def test_global_refine_after_a_grid_calibration_replaces_the_map_everywhere(page, monkeypatch):
    R, c, store = page
    result = _run_global_refine(R, c, monkeypatch, NEW_PC)
    assert result["pc_map_action"] == "replaced"
    # the page's detector AND what indexing reads agree on the one refined PC
    assert not pc_is_per_pixel(c.detector)
    assert store.get_entry("ds").pc_map is None
    np.testing.assert_allclose(np.asarray(store.get_detector("ds").pc).ravel(), NEW_PC, atol=1e-12)


def test_global_refine_of_a_file_map_shifts_it(monkeypatch):
    """An Oxford-style header map keeps its structure: the page works on its mean
    (single-PC controller), the store's map is shifted to it."""
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    c = PCController()
    c.attach_detector(EBSDDetector(shape=(60, 60), pc=_plane_mean(), sample_tilt=70.0))
    store = _store_with("ds", "header")
    monkeypatch.setattr(R, "_get_controller", lambda: c)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    result = _run_global_refine(R, c, monkeypatch, NEW_PC)
    assert result["pc_map_action"] == "shifted"
    moved = store.get_entry("ds").pc_map
    np.testing.assert_allclose(moved.reshape(-1, 3).mean(axis=0), NEW_PC, atol=1e-12)
    np.testing.assert_allclose(moved - NEW_PC, _plane() - np.asarray(_plane_mean()), atol=1e-12)


def test_a_refine_on_a_copy_replaces_the_roots_grid_map_and_reports_every_dataset(monkeypatch):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    c = PCController()
    c.attach_detector(EBSDDetector(shape=(60, 60), pc=_plane_mean(), sample_tilt=70.0))
    store = _store_with("raw", "refined_map")
    store.register_derived("copy", "raw")
    monkeypatch.setattr(R, "_get_controller", lambda: c)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "copy")
    result = _run_global_refine(R, c, monkeypatch, NEW_PC)
    assert result["pc_map_actions"]["raw"] == "replaced"
    assert result["pc_map_action"] == "replaced"
    assert store.get_entry("raw").pc_map is None
    np.testing.assert_allclose(np.asarray(store.get_detector("raw").pc).ravel(), NEW_PC, atol=1e-12)


def test_a_typed_pc_on_a_copy_shifts_the_roots_grid_map(page):
    R, c, store = page
    store.register_derived("copy", "ds")
    import backend.api.routes.ebsd_viewer as ev
    ev._active_dataset = "copy"
    try:
        res = asyncio.run(R.update_pc(R.UpdatePCRequest(pcx=NEW_PC[0], pcy=NEW_PC[1], pcz=NEW_PC[2])))
    finally:
        ev._active_dataset = "ds"
    assert res["pc_map_actions"]["ds"] == "shifted"
    np.testing.assert_allclose(store.get_entry("ds").pc_map.reshape(-1, 3).mean(axis=0), NEW_PC, atol=1e-12)
