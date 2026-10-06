"""PyEBSDIndex is credited for a pattern centre that came from PC refinement.

Orienta's PC refinement runs PyEBSDIndex's Hough indexing and its PC optimiser
(`ebsd_utils.optimize_pc`). A result indexed at a pattern centre that came out
of it therefore has to credit PyEBSDIndex, even if the indexing method itself
was dictionary or spherical. A result whose PC was typed by the user, read from
the file, or inherited from a dataset that never had it refined must not.

The origin is a fact of the calibration store entry (`pc_refinement`), written
by the refinement routes and cleared by every other way of changing the PC.
"""
import numpy as np
import pytest

from backend.api.routes import indexing as indexing_routes
from backend.api.routes.pcrefinement import _writeback_refined_pc
from backend.api.services.calibration_store import CalibrationStore, calibration_store
from backend.api.services.citations.provenance import get_steps
from backend.api.services.citations.render import render_methods
from backend.api.services.citations.steps import STEP_REGISTRY

KEY = "calibration.pc_refinement"
REFINED = {"n_patterns": 5, "method": "PSO"}


class _Det:
    shape = (60, 60)
    sample_tilt = 70.0
    tilt = 0.0
    azimuthal = 0.0

    def __init__(self, pc=(0.5, 0.3, 0.8)):
        self.pc = np.array(pc)


class _Sig:
    def __init__(self, pc=(0.5, 0.3, 0.8)):
        self.detector = _Det(pc)


class _Result:
    def __init__(self):
        self.metadata = {}
        self.xmap = None


def _keys(result):
    return [s["key"] for s in get_steps(result)]


# ----------------------------------------------------------------- the registry

def test_the_step_is_declared_and_cites_pyebsdindex():
    step = STEP_REGISTRY[KEY]
    assert "pyebsdindex" in step.citation_ids


def test_the_sentence_names_the_software_the_count_and_the_optimiser():
    out = render_methods([{"key": KEY, "params": REFINED}])
    assert "PyEBSDIndex" in out
    assert "Rowenhorst et al., 2024" in out
    assert "5 calibration patterns" in out
    assert "(PSO)" in out
    nm = render_methods([{"key": KEY, "params": {"n_patterns": 3, "method": "Nelder-Mead"}}])
    assert "(Nelder-Mead)" in nm


# ------------------------------------------------------------- the store entry

def test_refinement_sets_the_origin_and_any_other_change_clears_it():
    store = CalibrationStore()
    store.register("a", _Sig())
    assert store.get_entry("a").pc_refinement is None          # the file's PC

    store.update_pc("a", [0.51, 0.31, 0.81], source="refined", refinement=REFINED)
    assert store.get_entry("a").pc_refinement == REFINED

    store.update_pc("a", [0.52, 0.32, 0.82], source="manual")  # typed afterwards
    assert store.get_entry("a").pc_refinement is None


def test_a_per_pixel_map_that_is_not_a_refinement_clears_it():
    store = CalibrationStore()
    store.register("a", _Sig())
    store.update_pc("a", [0.51, 0.31, 0.81], source="refined", refinement=REFINED)
    store.update_pc_map("a", np.full((4, 5, 3), 0.5))          # e.g. imported map
    assert store.get_entry("a").pc_refinement is None
    store.update_pc_map("a", np.full((4, 5, 3), 0.5), refinement={"n_patterns": 12, "method": "Nelder-Mead"})
    assert store.get_entry("a").pc_refinement == {"n_patterns": 12, "method": "Nelder-Mead"}


def test_a_copy_of_a_refined_dataset_has_the_same_origin():
    store = CalibrationStore()
    store.register("raw", _Sig())
    store.update_pc("raw", [0.51, 0.31, 0.81], source="refined", refinement=REFINED)
    store.register_derived("raw_avg", "raw")
    assert store.get_entry("raw_avg").pc_refinement == REFINED


def test_a_new_registration_of_the_same_name_starts_clean():
    store = CalibrationStore()
    store.register("a", _Sig())
    store.update_pc("a", [0.51, 0.31, 0.81], source="refined", refinement=REFINED)
    store.register("a", _Sig())                                 # file loaded again
    assert store.get_entry("a").pc_refinement is None


def test_the_writeback_gives_the_parent_the_origin_and_a_manual_pc_removes_it():
    store = CalibrationStore()
    store.register("raw", _Sig())
    store.register_derived("raw_avg", "raw")
    _writeback_refined_pc("raw_avg", [0.52, 0.32, 0.82], store=store, refinement=REFINED)
    assert store.get_entry("raw_avg").pc_refinement == REFINED
    assert store.get_entry("raw").pc_refinement == REFINED
    _writeback_refined_pc("raw_avg", [0.53, 0.33, 0.83], store=store, source="manual")
    assert store.get_entry("raw_avg").pc_refinement is None
    assert store.get_entry("raw").pc_refinement is None


def test_the_routes_describe_a_refinement_by_count_and_the_optimiser_the_page_sent():
    from backend.api.routes.pcrefinement import _refinement_record

    assert _refinement_record(5, "PSO") == {"n_patterns": 5, "method": "PSO"}
    assert _refinement_record(3, "Nelder-Mead") == {"n_patterns": 3, "method": "Nelder-Mead"}
    assert _refinement_record(7, "pso")["method"] == "PSO"


# ------------------------------------------------------- what a run records

@pytest.fixture
def active(monkeypatch):
    """The module-level store with one dataset, active, removed afterwards."""
    from backend.api.routes import ebsd_viewer

    name = "pc_refinement_step_test"
    calibration_store.register(name, _Sig())
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", name)
    yield name
    calibration_store.remove(name)


@pytest.mark.parametrize("method", ["hough", "dictionary", "spherical"])
def test_a_run_at_a_refined_pc_records_the_step(active, method):
    calibration_store.update_pc(active, [0.51, 0.31, 0.81], source="refined",
                                refinement=REFINED)
    result = _Result()
    indexing_routes._attach_indexing_metadata(result, method)
    steps = [s for s in get_steps(result) if s["key"] == KEY]
    assert steps == [{"key": KEY, "params": REFINED}]


def test_recording_twice_does_not_repeat_the_step(active):
    calibration_store.update_pc(active, [0.51, 0.31, 0.81], source="refined",
                                refinement=REFINED)
    result = _Result()
    indexing_routes._attach_indexing_metadata(result, "hough")
    indexing_routes._attach_indexing_metadata(result, "hough")
    assert _keys(result).count(KEY) == 1


def test_a_run_at_the_file_pc_records_nothing(active):
    result = _Result()
    indexing_routes._attach_indexing_metadata(result, "hough")
    assert KEY not in _keys(result)


def test_a_run_at_a_typed_pc_records_nothing(active):
    calibration_store.update_pc(active, [0.51, 0.31, 0.81], source="refined",
                                refinement=REFINED)
    calibration_store.update_pc(active, [0.55, 0.35, 0.85], source="manual")
    result = _Result()
    indexing_routes._attach_indexing_metadata(result, "hough")
    assert KEY not in _keys(result)


def test_a_run_after_loading_another_file_records_nothing(active, monkeypatch):
    from backend.api.routes import ebsd_viewer

    calibration_store.update_pc(active, [0.51, 0.31, 0.81], source="refined",
                                refinement=REFINED)
    other = active + "_other"
    calibration_store.register(other, _Sig())
    try:
        monkeypatch.setattr(ebsd_viewer, "_active_dataset", other)
        result = _Result()
        indexing_routes._attach_indexing_metadata(result, "hough")
        assert KEY not in _keys(result)
    finally:
        calibration_store.remove(other)
