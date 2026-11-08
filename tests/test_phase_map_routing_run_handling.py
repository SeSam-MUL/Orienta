"""Phase-map routing, run handling: Stop, honest provenance, compute mode, identity,
files nobody may lose silently, and a routed result that offers what a
competitive one offers.

Built on the helpers of ``test_phase_map_routing_methods.py`` (the real route and
the real ``run_per_phase_indexing`` with the single-phase indexer stubbed).
Expectations are written by hand from the behaviour the feature is meant to have.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import indexing_controller  # noqa: E402
from indexing_controller import (  # noqa: E402
    ComparisonConfig,
    IndexingCancelled,
    IndexingMethod,
    PhaseConfig,
    PixelSelectionMode,
    run_per_phase_indexing,
)
from tests.test_phase_map_routing_methods import (  # noqa: E402
    NAV,
    Recorder,
    _fake_kp_load,
    _grid,
    _sht_request,
    _start_routed,
    _stub_pmr,
)

SPH = IndexingMethod.SPHERICAL
DIC = IndexingMethod.DICTIONARY
HOU = IndexingMethod.HOUGH
TWO_SHT = dict(entries=["PhaseA.cif", "PhaseB.cif"],
               links={"PhaseA.sht": "PhaseA.cif", "PhaseB.sht": "PhaseB.cif"})


def _log(task) -> str:
    return "\n".join(task["pending_log"])


# --- 1. Stop --------------------------------------------------------------------

def _stop_all_tasks():
    from backend.api.routes import indexing as route
    for task in route._indexing_tasks.values():
        task["status"] = "failed"        # what POST /stop/{task_id} does


def test_stop_during_the_first_phase_ends_the_run_cancelled_without_a_second_phase(
        monkeypatch, tmp_path):
    rec = Recorder(SPH, on_call=lambda k: _stop_all_tasks() if k == 0 else None)
    task = _start_routed(monkeypatch, tmp_path, method="spherical", grid=_grid([3, 3]),
                         req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB"]),
                         single_phase=rec, **TWO_SHT)
    assert len(rec.calls) == 1                       # phase B never ran
    assert task["status"] == "failed" and task["message"] == "Cancelled by user"
    assert task["error"] is None                     # a cancel, not a failure
    assert task["stored"] == []                      # nothing half-finished is kept


def test_stop_is_not_counted_as_a_skipped_phase():
    # the controller alone: a cancel raised by a phase propagates, it is not
    # swallowed into "skipped" like a failed phase
    signal = SimpleNamespace(data=SimpleNamespace(shape=(*NAV, 8, 8)))
    pcs = [PhaseConfig(name=f"P{i}", sht_path=f"P{i}.sht") for i in range(2)]
    grid = _grid([3, 3])
    calls = []

    def runner(pc, mask):
        calls.append(pc.name)
        raise IndexingCancelled("Indexing cancelled by user")

    with pytest.raises(IndexingCancelled):
        run_per_phase_indexing(
            signal=signal, detector=None, method=SPH,
            config=ComparisonConfig(phases=pcs, methods=[SPH]),
            phase_configs=pcs, masks_per_phase=[grid == 0, grid == 1],
            consensus_map=grid.astype(np.int32), run_phase=runner)
    assert calls == ["P0"]


def test_cancel_check_is_polled_before_every_phase_of_the_controller(monkeypatch):
    signal = SimpleNamespace(data=SimpleNamespace(shape=(*NAV, 8, 8)))
    pcs = [PhaseConfig(name=f"P{i}", sht_path=f"P{i}.sht") for i in range(2)]
    grid = _grid([3, 3])
    ran = []

    def plain(**kw):
        ran.append(kw["phase_config"].name)
        return _stub_pmr(kw["phase_config"].name, kw["selection_mask"], SPH)

    monkeypatch.setattr(indexing_controller, "run_single_phase_method", plain)
    asked = []

    def cancel():
        asked.append(len(ran))
        return len(ran) >= 1                          # Stop pressed during phase 1

    with pytest.raises(IndexingCancelled):
        run_per_phase_indexing(
            signal=signal, detector=None, method=SPH,
            config=ComparisonConfig(phases=pcs, methods=[SPH]),
            phase_configs=pcs, masks_per_phase=[grid == 0, grid == 1],
            consensus_map=grid.astype(np.int32), cancel_check=cancel)
    assert ran == ["P0"] and asked == [0, 1]


def test_the_cancel_check_and_progress_reach_every_per_phase_run(monkeypatch, tmp_path):
    rec = Recorder(SPH)
    task = _start_routed(monkeypatch, tmp_path, method="spherical", grid=_grid([3, 3]),
                         req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB"]),
                         single_phase=rec, **TWO_SHT)
    assert task["status"] == "completed", task["error"]
    for call in rec.calls:
        assert callable(call["cancel_check"]) and call["cancel_check"]() is False
        assert callable(call["progress_callback"])
    log = _log(task)
    assert "indexer of PhaseA.cif says hello" in log        # pat/s lines reach the user
    assert "indexer of PhaseB.cif says hello" in log
    assert max(rec.bar) < 0.95      # an indexer's own 0.99 is not the run's progress


def test_the_dictionary_runner_carries_the_cancel_check_and_progress_too(
        monkeypatch, tmp_path):
    _fake_kp_load(monkeypatch)
    rec = Recorder(DIC, on_call=lambda k: _stop_all_tasks() if k == 0 else None)
    task = _start_routed(
        monkeypatch, tmp_path, method="dictionary", grid=_grid([3, 3]),
        entries=["PhaseA.cif", "PhaseB.cif"],
        req_kwargs=dict(master_h5_paths=[str(tmp_path / "PhaseA_master.h5"),
                                         str(tmp_path / "PhaseB_master.h5")]),
        links={"PhaseA_master.h5": "PhaseA.cif", "PhaseB_master.h5": "PhaseB.cif"},
        single_phase=rec)
    assert len(rec.calls) == 1
    assert callable(rec.calls[0]["cancel_check"])
    assert task["message"] == "Cancelled by user" and task["stored"] == []
    assert max(rec.bar) < 0.95


def test_progress_does_not_go_backwards_in_a_routed_run(monkeypatch, tmp_path):
    from backend.api.routes import indexing as route

    seen = []
    rec = Recorder(SPH)

    def on_call(k):                 # the bar as each phase starts
        seen.append(next(reversed(route._indexing_tasks.values()))["progress"])
    rec.on_call = on_call
    task = _start_routed(monkeypatch, tmp_path, method="spherical", grid=_grid([3, 3]),
                         req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB"]),
                         single_phase=rec, **TWO_SHT)
    assert task["status"] == "completed", task["error"]
    assert seen == sorted(seen) and seen[0] >= 0.1            # 0.1 -> 0.5, never back


# --- 2. no false EDS-prior citation ---------------------------------------------

def test_a_routed_run_neither_cites_the_eds_prior_nor_runs_the_particle_rescue(
        monkeypatch, tmp_path):
    from backend.api.routes import indexing as route
    from backend.api.services.citations.provenance import get_steps

    def rescue_must_not_run(*a, **k):
        raise AssertionError("the particle rescue ran in a phase-map routed run")

    monkeypatch.setattr(route, "_apply_particle_rescue", rescue_must_not_run)
    shts = [str(tmp_path / "PhaseA.sht"), str(tmp_path / "PhaseB.sht")]
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical", grid=_grid([3, 3]),
        req_kwargs=dict(sht_paths=shts, backend="spherical_gpu",
                        eds_phase_strengths={shts[0]: 0.5, shts[1]: 0.5}),
        single_phase=Recorder(SPH), **TWO_SHT)
    assert task["status"] == "completed", task["error"]
    (result,) = task["stored"]
    assert "eds.chemistry_prior" not in [s["key"] for s in get_steps(result)]
    assert "not applied in a phase-map routed run" in _log(task)


def test_without_an_eds_prior_a_routed_run_says_nothing_about_it(monkeypatch, tmp_path):
    task = _start_routed(monkeypatch, tmp_path, method="spherical", grid=_grid([3, 3]),
                         req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB"]),
                         single_phase=Recorder(SPH), **TWO_SHT)
    assert "not applied in a phase-map routed run" not in _log(task)


# --- 3. compute mode and the calibrated detector reach the Dictionary run --------

class _Detector:
    """The store-resolved (calibrated) detector."""


def _use_detector(monkeypatch):
    from backend.api.routes import indexing as route

    det = _Detector()
    monkeypatch.setattr(route.calibration_store, "get_detector", lambda ds: det)
    return det


def test_a_routed_dictionary_run_gets_the_users_compute_mode_and_detector(
        monkeypatch, tmp_path):
    _fake_kp_load(monkeypatch)
    det = _use_detector(monkeypatch)
    rec = Recorder(DIC)
    task = _start_routed(
        monkeypatch, tmp_path, method="dictionary", grid=_grid([3, 3]),
        entries=["PhaseA.cif", "PhaseB.cif"],
        req_kwargs=dict(master_h5_paths=[str(tmp_path / "PhaseA_master.h5"),
                                         str(tmp_path / "PhaseB_master.h5")],
                        compute_mode="cpu"),
        links={"PhaseA_master.h5": "PhaseA.cif", "PhaseB_master.h5": "PhaseB.cif"},
        single_phase=rec)
    assert task["status"] == "completed", task["error"]
    assert [c["compute_mode"] for c in rec.calls] == ["cpu", "cpu"]
    assert all(c["detector"] is det for c in rec.calls)


def test_a_competitive_multi_phase_dictionary_run_gets_them_too(monkeypatch, tmp_path):
    _fake_kp_load(monkeypatch)
    det = _use_detector(monkeypatch)
    rec = Recorder(DIC, fail_after_record=True)       # enough to see what each phase got
    _start_routed(
        monkeypatch, tmp_path, method="dictionary", grid=_grid([3, 3]),
        entries=["PhaseA.cif", "PhaseB.cif"], routing=False,
        req_kwargs=dict(master_h5_paths=[str(tmp_path / "PhaseA_master.h5"),
                                         str(tmp_path / "PhaseB_master.h5")],
                        compute_mode="gpu"),
        links={}, single_phase=rec)
    assert [c["compute_mode"] for c in rec.calls] == ["gpu", "gpu"]
    assert all(c["detector"] is det for c in rec.calls)


def test_run_single_phase_method_hands_both_to_the_dictionary_indexer(monkeypatch):
    seen = {}

    def fake_dictionary_index(signal, dictionary, config, selection_mask=None,
                              progress_callback=None, cancel_check=None, detector=None):
        seen.update(compute_mode=config.compute_mode, detector=detector,
                    cancel_check=cancel_check, progress_callback=progress_callback)
        return _stub_pmr("x", selection_mask, DIC).indexing_result

    monkeypatch.setattr(indexing_controller, "dictionary_index_patterns",
                        fake_dictionary_index)
    signal = SimpleNamespace(data=SimpleNamespace(shape=(*NAV, 8, 8)))
    pc = PhaseConfig(name="P", master_h5_path="P_master.h5", dictionary=object(),
                     phase_list=object())
    det, stop, say = _Detector(), (lambda: False), (lambda *a: None)
    mask = _grid([3, 3]) == 0
    indexing_controller.run_single_phase_method(
        signal=signal, detector=det, phase_config=pc, method=DIC,
        config=ComparisonConfig(phases=[pc], methods=[DIC], compute_mode="cpu"),
        selection_mask=mask, progress_callback=say, cancel_check=stop)
    assert seen == {"compute_mode": "cpu", "detector": det,
                    "cancel_check": stop, "progress_callback": say}


def _spy_indexer(monkeypatch, name, method):
    """Replace one of the indexers ``run_single_phase_method`` calls by a spy that
    records the keyword arguments it was handed (positional ones are named by
    their parameter order, so a spy cannot hide a dropped argument)."""
    seen = {}

    def spy(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return _stub_pmr("x", _grid([3, 3]) == 0, method).indexing_result

    monkeypatch.setattr(indexing_controller, name, spy)
    return seen


def _run_single(method, backend="emsphinx"):
    stop, say = (lambda: False), (lambda *a: None)
    pc = PhaseConfig(name="P", cif_path="P.cif", sht_path="P.sht",
                     master_h5_path="P_master.h5", phase_list=object(),
                     dictionary=object())
    signal = SimpleNamespace(data=SimpleNamespace(shape=(*NAV, 8, 8)))
    indexing_controller.run_single_phase_method(
        signal=signal, detector=_Detector(), phase_config=pc, method=method,
        config=ComparisonConfig(phases=[pc], methods=[method], backend=backend),
        selection_mask=_grid([3, 3]) == 0, h5_path="scan.h5",
        detector_params={"d": 1}, progress_callback=say, cancel_check=stop)
    return stop, say


@pytest.mark.parametrize("name, method, backend, takes_cancel", [
    ("hough_index_patterns", HOU, "emsphinx", True),
    ("spherical_gpu_index_patterns", SPH, "spherical_gpu", True),
    ("spherical_index_patterns", SPH, "emsphinx", False),      # EMSphInx CPU: no cancel hook
])
def test_each_real_indexer_call_gets_the_progress_callback_and_the_cancel_check(
        monkeypatch, name, method, backend, takes_cancel):
    seen = _spy_indexer(monkeypatch, name, method)
    stop, say = _run_single(method, backend)
    assert seen["kwargs"]["progress_callback"] is say
    if takes_cancel:
        assert seen["kwargs"]["cancel_check"] is stop
    else:
        assert "cancel_check" not in seen["kwargs"]


# --- 4. one file per phase, and nothing selected disappears silently -------------

def test_files_the_run_does_not_use_are_named_with_the_reason(monkeypatch, tmp_path):
    rec = Recorder(SPH)
    names = ["PhaseA", "PhaseA2", "PhaseB", "Stranger", "Orphan", "PhaseZ"]
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif", "PhaseB.cif", "PhaseZ.cif"],
        grid=_grid([3, 3]),                                   # PhaseZ has no pixels
        req_kwargs=_sht_request(tmp_path, names),
        links={"PhaseA.sht": "PhaseA.cif", "PhaseA2.sht": "PhaseA.cif",
               "PhaseB.sht": "PhaseB.cif", "Stranger.sht": "Elsewhere.cif",
               "PhaseZ.sht": "PhaseZ.cif"},                   # Orphan.sht: linked to nothing
        single_phase=rec)
    assert task["status"] == "completed", task["error"]
    assert [c["sht_path"] for c in rec.calls] == [
        str(tmp_path / "PhaseA.sht"), str(tmp_path / "PhaseB.sht")]   # first file wins
    (result,) = task["stored"]
    unused = {u["file"]: u["reason"] for u in result.metadata["phase_map_unused_files"]}
    assert sorted(unused) == ["Orphan.sht", "PhaseA2.sht", "PhaseZ.sht", "Stranger.sht"]
    assert "PhaseA.sht" in unused["PhaseA2.sht"] and "one file per phase" in unused["PhaseA2.sht"]
    assert "no CIF" in unused["Orphan.sht"]
    assert "Elsewhere.cif is not a phase of the phase map" in unused["Stranger.sht"]
    assert "no pixels" in unused["PhaseZ.sht"]
    log = _log(task)
    for name in unused:
        assert name in log
    assert "WARNING" in log


def test_route_unused_files_reasons():
    from backend.api.routes import indexing as route

    links = [("/d/A.sht", "A.cif"), ("/d/A2.sht", "A.cif"), ("/d/B.sht", None),
             ("/d/C.sht", "C.cif"), ("/d/D.sht", "D.cif")]
    unused = route.route_unused_files(links, {"A.cif", "C.cif"}, {"A.cif"}, "SHT")
    assert [(u["file"], u["reason"]) for u in unused] == [
        ("A2.sht", "A.sht is already the SHT file of phase A.cif, one file per phase"),
        ("B.sht", "it is linked to no CIF of the library"),
        ("C.sht", "its phase C.cif has no pixels in the phase map"),
        ("D.sht", "its CIF D.cif is not a phase of the phase map"),
    ]
    assert route.describe_unused_files([], "spherical") == ""
    assert "A2.sht" in route.describe_unused_files(unused, "spherical")


# --- the middle phase missing: ids, names and the SHT map follow the kept phases --

def test_a_missing_middle_phase_leaves_ids_names_and_the_sht_map_consistent(
        monkeypatch, tmp_path):
    rec = Recorder(SPH)
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif", "PhaseB.cif", "PhaseC.cif"], grid=_grid([2, 2, 2]),
        req_kwargs=_sht_request(tmp_path, ["PhaseC", "PhaseA"]),     # B has no SHT
        links={"PhaseA.sht": "PhaseA.cif", "PhaseC.sht": "PhaseC.cif"},
        single_phase=rec)
    assert task["status"] == "completed", task["error"]
    (result,) = task["stored"]
    assert result.xmap.phase_id.tolist() == [0, 0, -1, -1, 1, 1]
    names = {int(p): n for p, n in zip(result.xmap.phases.ids, result.xmap.phases.names)
             if p >= 0}
    assert names == {0: "PhaseA", 1: "PhaseC"}
    assert {k: Path(v).name for k, v in result.metadata["sht_paths_by_phase"].items()} == {
        0: "PhaseA.sht", 1: "PhaseC.sht"}
    assert [m["phase"] for m in result.metadata["phase_map_unrouted"]] == ["PhaseB.cif"]


def test_the_names_and_the_sht_map_are_read_from_the_files_that_ran(monkeypatch, tmp_path):
    from backend.api.routes import indexing as route

    spy = {}
    real = route._attach_indexing_metadata

    def attach(result, method_name, **kw):
        spy.update(kw)
        return real(result, method_name, **kw)

    monkeypatch.setattr(route, "_attach_indexing_metadata", attach)
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([3, 3]),
        req_kwargs=_sht_request(tmp_path, ["PhaseB", "PhaseA", "Spare"]),
        links={"PhaseA.sht": "PhaseA.cif", "PhaseB.sht": "PhaseB.cif"},
        single_phase=Recorder(SPH))
    assert task["status"] == "completed", task["error"]
    ran = [str(tmp_path / "PhaseA.sht"), str(tmp_path / "PhaseB.sht")]
    assert spy["name_source_paths"] == ran
    assert spy["sht_paths"] == ran


# --- missing phases for the other methods; the file kind in the error ------------

def test_dictionary_phase_without_a_file_stays_unindexed_and_is_named(
        monkeypatch, tmp_path):
    _fake_kp_load(monkeypatch)
    task = _start_routed(
        monkeypatch, tmp_path, method="dictionary",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([2, 4]),
        req_kwargs=dict(master_h5_paths=[str(tmp_path / "PhaseB_master.h5")]),
        links={"PhaseB_master.h5": "PhaseB.cif"}, single_phase=Recorder(DIC))
    assert task["status"] == "completed", task["error"]
    (result,) = task["stored"]
    assert result.xmap.phase_id.tolist() == [-1, -1, 0, 0, 0, 0]
    (missing,) = result.metadata["phase_map_unrouted"]
    assert missing["phase"] == "PhaseA.cif" and missing["pixels"] == 2
    assert "dictionary (master .h5)" in missing["reason"]


def test_hough_phase_without_a_file_stays_unindexed_and_is_named(monkeypatch, tmp_path):
    from orix.crystal_map import Phase

    import ebsd_utils

    monkeypatch.setattr(ebsd_utils, "hough_phase_from_cif",
                        lambda path: Phase(name=Path(path).stem))
    task = _start_routed(
        monkeypatch, tmp_path, method="hough",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([2, 4]),
        req_kwargs=dict(cif_paths=[str(tmp_path / "PhaseA.cif")]),
        links={}, single_phase=Recorder(HOU))
    assert task["status"] == "completed", task["error"]
    (result,) = task["stored"]
    assert result.xmap.phase_id.tolist() == [0, 0, -1, -1, -1, -1]
    (missing,) = result.metadata["phase_map_unrouted"]
    assert missing["phase"] == "PhaseB.cif" and "CIF file" in missing["reason"]


def test_a_hough_cif_that_cannot_be_read_becomes_unrouted_and_is_named(
        monkeypatch, tmp_path):
    from orix.crystal_map import Phase

    import ebsd_utils

    def build(path):
        if Path(path).name == "PhaseB.cif":
            raise ValueError("no atoms in this file")
        return Phase(name=Path(path).stem)

    monkeypatch.setattr(ebsd_utils, "hough_phase_from_cif", build)
    task = _start_routed(
        monkeypatch, tmp_path, method="hough",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([3, 3]),
        req_kwargs=dict(cif_paths=[str(tmp_path / "PhaseA.cif"),
                                   str(tmp_path / "PhaseB.cif")]),
        links={}, single_phase=Recorder(HOU))
    assert task["status"] == "completed", task["error"]
    (result,) = task["stored"]
    (lost,) = result.metadata["phase_map_unrouted"]
    assert lost["phase"] == "PhaseB.cif" and "no atoms in this file" in lost["reason"]
    assert result.xmap.phase_id.tolist() == [0, 0, 0, -1, -1, -1]


def test_when_no_phase_has_a_file_the_error_names_method_and_file_kind(
        monkeypatch, tmp_path):
    spherical = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif"], grid=_grid([6]),
        req_kwargs=_sht_request(tmp_path, ["Other"]), links={"Other.sht": "X.cif"},
        single_phase=Recorder(SPH))
    assert "for spherical indexing" in spherical["error"]
    assert "has a SHT file among the request's spherical files" in spherical["error"]
    assert "Select the SHT file of each phase-map phase" in spherical["error"]

    dictionary = _start_routed(
        monkeypatch, tmp_path, method="dictionary",
        entries=["PhaseA.cif"], grid=_grid([6]),
        req_kwargs=dict(master_h5_paths=[str(tmp_path / "Other_master.h5")]),
        links={}, single_phase=Recorder(DIC))
    assert "for dictionary indexing" in dictionary["error"]
    assert "dictionary (master .h5) file among the request's dictionary files" \
        in dictionary["error"]


# --- the mask, the window fields, the pattern-file path --------------------------

def test_an_empty_phase_is_skipped_not_run_and_its_file_is_reported(monkeypatch, tmp_path):
    rec = Recorder(SPH)
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif", "PhaseB.cif", "PhaseC.cif"], grid=_grid([2, 0, 2]),
        req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB", "PhaseC"]),
        links={"PhaseA.sht": "PhaseA.cif", "PhaseB.sht": "PhaseB.cif",
               "PhaseC.sht": "PhaseC.cif"}, single_phase=rec)
    assert task["status"] == "completed", task["error"]
    assert [c["name"] for c in rec.calls] == ["PhaseA.cif", "PhaseC.cif"]
    (result,) = task["stored"]
    assert result.metadata["n_phases_skipped"] == 0
    assert result.xmap.phase_id.tolist() == [0, 0, 1, 1, -1, -1]
    assert [u["file"] for u in result.metadata["phase_map_unused_files"]] == ["PhaseB.sht"]


def test_the_region_fields_and_the_pattern_file_reach_the_per_phase_run(
        monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer

    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", "scan.h5oina")
    seen = []
    rec = Recorder(SPH, on_call=lambda k: None)
    original = rec.__call__

    def spy(signal, detector, phase_config, method, config, selection_mask, **kw):
        seen.append((config.row_start, config.row_end, config.col_start, config.col_end,
                     kw.get("h5_path"), config.selection_mode))
        return original(signal, detector, phase_config, method, config,
                        selection_mask, **kw)

    task = _start_routed(
        monkeypatch, tmp_path, method="spherical", grid=_grid([3, 3]),
        req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB"], row_start=1, row_end=2,
                                col_start=0, col_end=2),
        single_phase=spy, **TWO_SHT)
    assert task["status"] == "completed", task["error"]
    assert seen == [(1, 2, 0, 2, "scan.h5oina", PixelSelectionMode.MASK)] * 2


def test_the_crops_lasso_mask_is_anded_into_every_routed_mask(monkeypatch, tmp_path):
    from backend.api.routes import indexing as route

    keep = np.array([[True, False, True], [True, True, False]])
    monkeypatch.setattr(route, "_active_crop_window",
                        lambda: SimpleNamespace(nav_mask=keep, row0=0, col0=0, original_shape=keep.shape))
    rec = Recorder(SPH)
    task = _start_routed(monkeypatch, tmp_path, method="spherical", grid=_grid([3, 3]),
                         req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB"]),
                         single_phase=rec, **TWO_SHT)
    assert task["status"] == "completed", task["error"]
    a, b = (c["mask"] for c in rec.calls)
    assert a.tolist() == [[True, False, True], [False, False, False]]
    assert b.tolist() == [[False, False, False], [True, True, False]]


def test_routed_hough_registers_each_phases_reflector_limit(monkeypatch, tmp_path):
    from orix.crystal_map import Phase

    import ebsd_utils

    monkeypatch.setattr(ebsd_utils, "hough_phase_from_cif",
                        lambda path: Phase(name=Path(path).stem))
    registered = {}
    monkeypatch.setattr(ebsd_utils, "set_phase_reflector_limit",
                        lambda path, limit: registered.__setitem__(Path(path).name, limit))
    seen = {}

    def capture(**kw):
        seen.update(kw)
        raise RuntimeError("stop after routing")

    monkeypatch.setattr(indexing_controller, "run_per_phase_indexing", capture)
    # the request lists B first, with B's limit first
    _start_routed(
        monkeypatch, tmp_path, method="hough",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([3, 3]),
        req_kwargs=dict(cif_paths=[str(tmp_path / "PhaseB.cif"),
                                   str(tmp_path / "PhaseA.cif")], max_reflectors=[9, 5]),
        links={})
    assert registered == {"PhaseA.cif": 5, "PhaseB.cif": 9}
    assert [(c.name, c.max_reflectors) for c in seen["phase_configs"]] == [
        ("PhaseA.cif", 5), ("PhaseB.cif", 9)]


def test_a_dictionary_that_fails_to_load_still_hands_its_gpu_memory_back(
        monkeypatch, tmp_path):
    import kikuchipy as kp

    from backend.api.routes import indexing as route

    def load(path, *a, **k):
        raise OSError("dictionary file is damaged")

    monkeypatch.setattr(kp, "load", load)
    released = []
    monkeypatch.setattr(route, "_release_phase_gpu_memory",
                        lambda exc=None: released.append(exc))
    task = _start_routed(
        monkeypatch, tmp_path, method="dictionary",
        entries=["PhaseA.cif"], grid=_grid([6]),
        req_kwargs=dict(master_h5_paths=[str(tmp_path / "PhaseA_master.h5")]),
        links={"PhaseA_master.h5": "PhaseA.cif"}, single_phase=Recorder(DIC))
    assert task["status"] == "failed"
    assert len(released) == 1 and isinstance(released[0], OSError)


# --- 8. a routed result offers what a competitive one offers ---------------------

def _routed_dictionary_result(monkeypatch, tmp_path, entries, grid, names):
    _fake_kp_load(monkeypatch)
    dict_files = {}
    for n in names:
        f = tmp_path / f"{n}_dict.h5"
        f.write_bytes(b"x")
        dict_files[f"{n}.cif"] = str(f)
    rec = Recorder(DIC, metadata_for=lambda pc: {
        "dictionary": object(), "dict_path": dict_files[pc.name]})
    task = _start_routed(
        monkeypatch, tmp_path, method="dictionary", entries=entries, grid=grid,
        req_kwargs=dict(master_h5_paths=[str(tmp_path / f"{n}_master.h5") for n in names]),
        links={f"{n}_master.h5": f"{n}.cif" for n in names}, single_phase=rec)
    assert task["status"] == "completed", task["error"]
    (result,) = task["stored"]
    return result, dict_files


def test_a_routed_result_carries_the_confidence_index_on_the_full_grid(
        monkeypatch, tmp_path):
    result, _ = _routed_dictionary_result(
        monkeypatch, tmp_path, ["PhaseA.cif", "PhaseB.cif", "PhaseC.cif"],
        _grid([2, 2, 2]), ["PhaseA", "PhaseB"])      # C has no file: its pixels stay NaN
    expected = [0.25, 0.25, 0.75, 0.75, np.nan, np.nan]
    np.testing.assert_allclose(np.asarray(result.confidence_scores), expected)
    np.testing.assert_allclose(np.asarray(result.xmap.prop["ci"]), expected)

    # and the phase-map page's CI layer reads it
    from backend.api.routes.phase_map import _extract_ci_array
    ci = _extract_ci_array(kind="ci", xmap=result.xmap, result=result, dataset=None,
                           n_rows=NAV[0], n_cols=NAV[1], mask=None)
    np.testing.assert_allclose(ci.ravel(), expected)


def test_a_routed_result_has_the_per_phase_data_and_summary_of_a_competitive_one(
        monkeypatch, tmp_path):
    result, _ = _routed_dictionary_result(
        monkeypatch, tmp_path, ["PhaseA.cif", "PhaseB.cif", "PhaseC.cif"],
        _grid([2, 3, 1]), ["PhaseA", "PhaseB", "PhaseC"])
    md = result.metadata
    final = [str(result.xmap.phases[i].name) for i in range(3)]
    assert md["multi_phase"] is True
    # keyed by the phases' final names, in the competitive shape
    assert sorted(md["per_phase_data"]) == sorted(final)
    a = md["per_phase_data"][final[0]]
    assert a["ci"].shape == NAV and a["euler"].shape == (*NAV, 3)
    np.testing.assert_allclose(a["ci"].ravel(), [0.25, 0.25] + [np.nan] * 4, equal_nan=True)
    assert np.isnan(a["euler"][1, 2]).all() and not np.isnan(a["euler"][0, 0]).any()
    assert a["ci_mean"] == pytest.approx(0.9)
    # the summary the Indexing page shows: biggest phase first, fractions of the run
    stats = md["per_phase_stats"]
    assert [(s["name"], s["pixels"], s["fraction"]) for s in stats] == [
        (final[1], 3, 50.0), (final[0], 2, 33.3), (final[2], 1, 16.7)]


def test_the_per_phase_ci_heatmap_reads_a_routed_result(monkeypatch, tmp_path):
    from backend.api.routes.phase_map import _render_ci_figure

    result, _ = _routed_dictionary_result(
        monkeypatch, tmp_path, ["PhaseA.cif", "PhaseB.cif"], _grid([2, 4]),
        ["PhaseA", "PhaseB"])
    name = str(result.xmap.phases[0].name)

    def render(direction):
        fig, _meta = _render_ci_figure(
            direction, result, None, NAV[0], NAV[1], None, 1.0, False, "lower left",
            10, 12, "k", "w", 0.5, False, None, "viridis", 50)
        return fig

    assert render(f"ci_{name}") is not None
    assert render("uncertainty") is not None          # needs two phases' CI maps


def test_twin_display_names_keep_the_first_phase_and_say_so(monkeypatch, caplog):
    from backend.api.routes import indexing as route

    class Phases:
        def __getitem__(self, i):
            return SimpleNamespace(name="Mg17Al12")       # both phases end up with this name

    result = SimpleNamespace(
        xmap=SimpleNamespace(phases=Phases()),
        metadata={"per_phase_match_sources": {"T1.cif": "one", "T2.cif": "two"},
                  "per_phase_stats": [{"name": "T2.cif"}, {"name": "T1.cif"}]})
    pcs = [PhaseConfig(name="T1.cif"), PhaseConfig(name="T2.cif")]
    route._rename_routed_per_phase_keys(result, pcs)
    assert result.metadata["per_phase_match_sources"] == {"Mg17Al12": "one", "T2.cif": "two"}
    assert [s["name"] for s in result.metadata["per_phase_stats"]] == ["T2.cif", "Mg17Al12"]


@pytest.mark.parametrize("shape_name, confidence, expected", [
    # Hough: (n_phases + 1, n_pix), the LAST row the best across phases
    ("hough", [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], [0.9, 0.8, 0.7]),
    # Dictionary top-N: (n_pix, keep_n), column 0 the best match
    ("top-n", [[0.5, 0.1], [0.6, 0.2], [0.7, 0.3]], [0.5, 0.6, 0.7]),
    ("flat", [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]),
])
def test_a_phases_confidence_is_read_in_the_shape_its_indexer_writes(
        shape_name, confidence, expected):
    mask = _grid([3, 3]) == 0
    pmr = _stub_pmr("PhaseA.cif", mask, HOU)
    pmr.indexing_result.confidence_scores = np.asarray(confidence)
    merged = SimpleNamespace(confidence_scores=None, metadata={},
                             xmap=SimpleNamespace(prop={}))
    indexing_controller._carry_routed_fields(merged, [pmr], NAV)
    np.testing.assert_allclose(merged.confidence_scores,
                               expected + [np.nan] * 3, equal_nan=True)


def test_a_routed_dictionary_result_finds_its_pattern_source_by_the_final_names(
        monkeypatch, tmp_path):
    result, dict_files = _routed_dictionary_result(
        monkeypatch, tmp_path, ["PhaseA.cif", "PhaseB.cif"], _grid([2, 4]),
        ["PhaseA", "PhaseB"])
    sources = result.metadata["per_phase_match_sources"]
    final_names = [str(result.xmap.phases[i].name) for i in (0, 1)]
    assert sorted(sources) == sorted(final_names)                 # keyed like the xmap
    a, b = (sources[n] for n in final_names)
    assert a["dict_path"] == dict_files["PhaseA.cif"]
    assert b["dict_path"] == dict_files["PhaseB.cif"]
    # simulation indices lifted onto the grid: phase A's 100, 101 at pixels 0, 1;
    # phase B's 200 .. 203 at pixels 2 .. 5; -1 where the phase did not run
    assert a["simulation_indices"][:, 0].tolist() == [100, 101, -1, -1, -1, -1]
    assert b["simulation_indices"][:, 0].tolist() == [-1, -1, 200, 201, 202, 203]


def test_the_pattern_match_dialog_reads_a_routed_result(monkeypatch, tmp_path):
    h5py = pytest.importorskip("h5py")
    from tools.pattern_comparison import get_best_match_pattern

    result, dict_files = _routed_dictionary_result(
        monkeypatch, tmp_path, ["PhaseA.cif", "PhaseB.cif"], _grid([2, 4]),
        ["PhaseA", "PhaseB"])
    for key, offset in (("PhaseA.cif", 100), ("PhaseB.cif", 200)):
        with h5py.File(dict_files[key], "w") as f:        # a tiny dictionary file
            f["patterns"] = (offset + np.arange(400).reshape(400, 1, 1)
                             * np.ones((1, 2, 2))).astype(np.float32)
    # pixel (row 0, col 2) is phase B's first pixel -> its dictionary row 200
    pattern = get_best_match_pattern(result, 0, 2)
    assert pattern is not None and pattern.shape == (2, 2)
    assert float(pattern[0, 0]) == 200 + 200
    # pixel (0, 0) is phase A's first pixel -> its dictionary row 100
    assert float(get_best_match_pattern(result, 0, 0)[0, 0]) == 100 + 100


def test_a_routed_spherical_result_carries_its_orientation_source(monkeypatch, tmp_path):
    def metadata(pc):
        return ({"orientation_source": "hough", "orientation_source_reason": "pseudosym"}
                if pc.name == "PhaseB.cif" else {"orientation_source": "spherical"})

    task = _start_routed(monkeypatch, tmp_path, method="spherical", grid=_grid([3, 3]),
                         req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB"]),
                         single_phase=Recorder(SPH, metadata_for=metadata), **TWO_SHT)
    (result,) = task["stored"]
    assert result.metadata["orientation_source"] == "mixed"
    assert "PhaseB.cif: pseudosym" in result.metadata["orientation_source_reason"]
