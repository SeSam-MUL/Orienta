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
def dataset():
    """A dataset in the module-level store, removed afterwards."""
    name = "pc_refinement_step_test"
    calibration_store.register(name, _Sig())
    yield name
    calibration_store.remove(name)


def _run(dataset_name, method="hough", *, switch_to=None):
    """What a run does: look at the PC's origin when it STARTS, record at the end.

    ``switch_to`` makes the active dataset change in between (the user loads or
    switches a file while the run is in progress).
    """
    from backend.api.routes import ebsd_viewer

    saved = ebsd_viewer._active_dataset
    ebsd_viewer._active_dataset = dataset_name
    try:
        started_with = indexing_routes._pc_refinement_snapshot(ebsd_viewer._active_dataset)
        if switch_to is not None:
            ebsd_viewer._active_dataset = switch_to
        result = _Result()
        indexing_routes._attach_indexing_metadata(result, method, pc_refinement=started_with)
        return result
    finally:
        ebsd_viewer._active_dataset = saved


@pytest.mark.parametrize("method", ["hough", "dictionary", "spherical"])
def test_a_run_at_a_refined_pc_records_the_step(dataset, method):
    calibration_store.update_pc(dataset, [0.51, 0.31, 0.81], source="refined",
                                refinement=REFINED)
    steps = [s for s in get_steps(_run(dataset, method)) if s["key"] == KEY]
    assert steps == [{"key": KEY, "params": REFINED}]


def test_recording_twice_does_not_repeat_the_step(dataset):
    calibration_store.update_pc(dataset, [0.51, 0.31, 0.81], source="refined",
                                refinement=REFINED)
    result = _Result()
    snap = indexing_routes._pc_refinement_snapshot(dataset)
    indexing_routes._attach_indexing_metadata(result, "hough", pc_refinement=snap)
    indexing_routes._attach_indexing_metadata(result, "hough", pc_refinement=snap)
    assert _keys(result).count(KEY) == 1


def test_a_run_at_the_file_pc_records_nothing(dataset):
    assert KEY not in _keys(_run(dataset))


def test_a_run_at_a_typed_pc_records_nothing(dataset):
    calibration_store.update_pc(dataset, [0.51, 0.31, 0.81], source="refined",
                                refinement=REFINED)
    calibration_store.update_pc(dataset, [0.55, 0.35, 0.85], source="manual")
    assert KEY not in _keys(_run(dataset))


def test_a_run_after_loading_another_file_records_nothing(dataset):
    calibration_store.update_pc(dataset, [0.51, 0.31, 0.81], source="refined",
                                refinement=REFINED)
    other = dataset + "_other"
    calibration_store.register(other, _Sig())
    try:
        assert KEY not in _keys(_run(other))
    finally:
        calibration_store.remove(other)


def test_switching_the_active_dataset_during_a_run_does_not_change_what_it_records(dataset):
    """The PC a run used is the one of the dataset it STARTED on."""
    other = dataset + "_other"
    calibration_store.register(other, _Sig())
    try:
        # started on a refined dataset, active dataset is a plain one at the end
        calibration_store.update_pc(dataset, [0.51, 0.31, 0.81], source="refined",
                                    refinement=REFINED)
        assert KEY in _keys(_run(dataset, switch_to=other))
        # started on a plain dataset, active dataset is a refined one at the end
        calibration_store.update_pc(other, [0.52, 0.32, 0.82], source="refined",
                                    refinement=REFINED)
        calibration_store.update_pc(dataset, [0.5, 0.3, 0.8], source="manual")
        assert KEY not in _keys(_run(dataset, switch_to=other))
    finally:
        calibration_store.remove(other)


def test_the_snapshot_is_a_copy(dataset):
    calibration_store.update_pc(dataset, [0.51, 0.31, 0.81], source="refined",
                                refinement=dict(REFINED))
    snap = indexing_routes._pc_refinement_snapshot(dataset)
    snap["n_patterns"] = 999
    assert calibration_store.get_entry(dataset).pc_refinement == REFINED


def test_an_unknown_dataset_has_no_snapshot():
    assert indexing_routes._pc_refinement_snapshot("no_such_dataset") is None
    assert indexing_routes._pc_refinement_snapshot(None) is None


def test_both_run_routes_take_the_snapshot_at_the_start_and_hand_it_on():
    """The two indexing routes cannot be run here; this pins that each takes the
    snapshot where it takes the detector and passes it to the metadata call."""
    import re
    from pathlib import Path

    src = Path(indexing_routes.__file__).read_text(encoding="utf-8")
    assert len(re.findall(r"_pc_refinement_snapshot\(", src)) >= 3      # def + two routes
    assert len(re.findall(r"_attach_indexing_metadata\(.{0,900}?pc_refinement=", src, re.S)) >= 2
    # a batch run that applies an explicit PC is not at a refined PC
    assert re.search(r"detector\.pc = np\.array\(\[pc_to_apply\]\)\s+_run_pc_refinement = None", src)


# ------------------------------------------- the routes that move a refined PC

@pytest.fixture
def api():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes import batch_v2, calibration

    app = FastAPI()
    app.include_router(calibration.router, prefix="/api/calibration")
    app.include_router(batch_v2.router, prefix="/api/batch-v2")
    names = ["pcr_route_raw", "pcr_route_copy", "pcr_route_other"]
    yield TestClient(app), names
    for n in names:
        calibration_store.remove(n)


def test_propagating_to_the_parent_carries_the_origin(api):
    client, (raw, copy, _) = api
    calibration_store.register(raw, _Sig())
    calibration_store.register_derived(copy, raw)
    calibration_store.update_pc(copy, [0.52, 0.32, 0.82], source="refined",
                                refinement=REFINED)
    assert calibration_store.get_entry(raw).pc_refinement is None
    assert client.post(f"/api/calibration/{copy}/propagate-to-parent").status_code == 200
    assert calibration_store.get_entry(raw).pc_refinement == REFINED


def test_propagating_a_typed_pc_does_not_claim_a_refinement(api):
    client, (raw, copy, _) = api
    calibration_store.register(raw, _Sig())
    calibration_store.register_derived(copy, raw)
    calibration_store.update_pc(copy, [0.52, 0.32, 0.82], source="manual")
    client.post(f"/api/calibration/{copy}/propagate-to-parent")
    assert calibration_store.get_entry(raw).pc_refinement is None


def test_copying_a_pc_to_other_files_carries_the_origin_only_if_it_has_one(api):
    client, (raw, _, other) = api
    calibration_store.register(raw, _Sig())
    calibration_store.register(other, _Sig())
    body = {"source_dataset": raw, "target_datasets": [other]}
    client.post("/api/batch-v2/copy-pc", json=body)
    assert calibration_store.get_entry(other).pc_refinement is None
    calibration_store.update_pc(raw, [0.52, 0.32, 0.82], source="refined",
                                refinement=REFINED)
    client.post("/api/batch-v2/copy-pc", json=body)
    assert calibration_store.get_entry(other).pc_refinement == REFINED


def test_the_refine_routes_hand_their_origin_to_the_store():
    """`_run_optimization` and the grid calibration cannot be run here (they need
    a detector and Hough indexing of real patterns); this pins the two calls that
    carry the origin to the store, which every test above depends on."""
    import re
    from pathlib import Path
    from backend.api.routes import pcrefinement

    src = Path(pcrefinement.__file__).read_text(encoding="utf-8")
    assert re.search(r"_writeback_refined_pc\(\s*_active_dataset,\s*mean_pc,\s*"
                     r"refinement=_refinement_record\(", src)
    assert re.search(r"update_pc_map\(\s*_grid_ds,.{0,120}refinement=_refinement_record\(",
                     src, re.S)
