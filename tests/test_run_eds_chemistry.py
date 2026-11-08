"""The EDS chemistry that steers a run is the chemistry of the dataset the run started on.

A run takes minutes to hours and the viewer can be switched to another measurement of the same file
while it goes on. The prior (Spherical GPU, Dictionary, Hough) and the particle rescue used to ask the
loaded file for the chemistry when they needed it: for Dictionary and Hough that is after the long
per-phase work, for the rescue after the run. A reload of measurement 2 meanwhile gave them measurement
2's chemistry, and equal grids pass the grid guard. Now the chemistry is read once, when the run starts,
and handed down.

The measurements are told apart by their chemistry alone: measurement 1 is pure Al everywhere,
measurement 2 pure Si everywhere (hand-written), and the fake reader of the loaded file answers by the
variant the registry holds for the run's dataset at the moment it is asked.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backend.api.services import variant_registry
from tests.test_run_variant_wiring import drive  # noqa: F401  (fixture)
from tests.test_variant_rereads import RAW, RAW2, two_sources  # noqa: F401  (two_sources: fixture)

N = 9                                      # the 3x3 Ni sample of the run fixture


class _Stop(Exception):
    pass


@pytest.fixture
def chemistry(monkeypatch):
    """The reader of the loaded file, rigged: pure Al on measurement 1, pure Si on measurement 2.
    ``asked`` lists the measurement of the dataset 'ds' at each read, in order."""
    from backend.api.services import eds_indexing_prior as prior
    asked = []

    def fake_loaded_file():
        variant = variant_registry.variant_for("ds")
        measurement = None if variant is None else variant.measurement
        asked.append(measurement)
        pure = {"Al": 100.0, "Si": 0.0} if measurement != "2" else {"Al": 0.0, "Si": 100.0}
        return ({el: np.full(N, v) for el, v in pure.items()}, 3, 3, "multi.h5oina")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", fake_loaded_file)
    return asked


def _cifs(tmp_path):
    from tests.test_result_keeps_its_run_pc import _ni_cif
    first = _ni_cif(tmp_path)
    second = tmp_path / "Ni_second.cif"
    second.write_text(open(first, encoding="utf-8").read(), encoding="utf-8")
    return first, str(second)


def _hough_body(tmp_path):
    a, b = _cifs(tmp_path)
    return {"method": "hough", "cif_paths": [a, b], "eds_phase_strengths": {a: 1.0, b: 1.0}}


def _dictionary_body():
    return {"method": "dictionary", "master_h5_paths": ["a.h5", "b.h5"],
            "eds_phase_strengths": {"a.h5": 1.0, "b.h5": 1.0}}


def _per_phase_indexer(*args, **kwargs):
    """The per-phase indexer of the run; the viewer is switched to measurement 2 while it works."""
    variant_registry.register("ds", RAW2)
    return SimpleNamespace(mean_score=1.0, indexing_result=SimpleNamespace(
        confidence_scores=np.ones(N, np.float32), metadata={}))


@pytest.fixture
def merge_weights(monkeypatch):
    """The merge, stubbed to record the weights it is given and to stop the run there."""
    import indexing_controller
    from backend.api.routes import indexing as I
    monkeypatch.setattr(I, "_phase_formulas_for_paths", lambda paths: ["Al", "Si"][:len(paths)])
    monkeypatch.setattr(I, "_load_phase_dictionary", lambda pc, progress: False)
    monkeypatch.setattr(I, "_detach_dictionary_from_result", lambda result: None)
    got = {}

    def merge(comparison, phase_weights=None, **kwargs):
        got["weights"] = phase_weights
        raise _Stop("weights captured")

    monkeypatch.setattr(indexing_controller, "compute_comparison_maps", merge)
    return got


def _run_to_the_merge(drive, merge_weights, variant, body):
    task = drive(variant, _per_phase_indexer, body, entry="run_single_phase_method")
    assert task["status"] == "failed" and "weights captured" in task["error"], task
    return merge_weights["weights"]


# ------------------------------------------------------------------------------------ the prior

def test_the_hough_prior_of_a_run_on_measurement_1_is_measurement_1s_chemistry(
        drive, merge_weights, tmp_path, chemistry):
    weights = _run_to_the_merge(drive, merge_weights, RAW, _hough_body(tmp_path))
    # phase 0 is Al, phase 1 is Si; measurement 1 is pure Al: Al keeps its weight, Si is vetoed
    assert weights.shape == (2, N)
    assert weights[0].min() > 0.9 and weights[1].max() < 0.1
    assert chemistry == ["1"]                                  # asked once, for the run's measurement


def test_the_hough_prior_of_a_run_on_measurement_2_is_measurement_2s_chemistry(
        drive, merge_weights, tmp_path, chemistry):
    """The converse, so that the test above is not a default of the fake: a run that STARTS on 2 is Si."""
    weights = _run_to_the_merge(drive, merge_weights, RAW2, _hough_body(tmp_path))
    assert weights[1].min() > 0.9 and weights[0].max() < 0.1
    assert chemistry == ["2"]


def test_the_dictionary_prior_of_a_run_on_measurement_1_is_measurement_1s_chemistry(
        drive, merge_weights, chemistry):
    weights = _run_to_the_merge(drive, merge_weights, RAW, _dictionary_body())
    assert weights.shape == (2, N)
    assert weights[0].min() > 0.9 and weights[1].max() < 0.1
    assert chemistry == ["1"]


def test_the_spherical_gpu_prior_is_measurement_1s_chemistry_and_read_once(drive, monkeypatch, chemistry):
    from backend.api.routes import indexing as I
    monkeypatch.setattr(I, "_phase_formulas_for_paths", lambda paths: ["Al", "Si"])
    got = {}

    def indexer(*args, **kwargs):
        got["weights"] = kwargs["phase_weights"]
        raise _Stop("stop")

    task = drive(RAW, indexer, {"method": "spherical", "backend": "spherical_gpu",
                                "sht_paths": ["a.sht", "b.sht"],
                                "eds_phase_strengths": {"a.sht": 1.0, "b.sht": 1.0}})
    assert task["status"] == "failed"
    assert got["weights"][0].min() > 0.9 and got["weights"][1].max() < 0.1
    assert chemistry == ["1"]                                  # once, at the start: not again for the weights


def test_the_loaded_file_is_asked_once_and_before_the_indexer(drive, merge_weights, tmp_path, chemistry):
    """The read is at the start of the run: the first indexer call finds it done, and nothing asks again."""
    seen_at_indexer = []

    def indexer(*args, **kwargs):
        seen_at_indexer.append(len(chemistry))
        return _per_phase_indexer()

    task = drive(RAW, indexer, _hough_body(tmp_path), entry="run_single_phase_method")
    assert task["status"] == "failed"
    assert seen_at_indexer and set(seen_at_indexer) == {1}
    assert len(chemistry) == 1


def test_a_run_without_the_prior_never_reads_the_loaded_file(drive, monkeypatch, tmp_path, chemistry):
    a, b = _cifs(tmp_path)

    def indexer(*args, **kwargs):
        raise _Stop("stop")

    task = drive(RAW, indexer, {"method": "hough", "cif_paths": [a, b],
                                "eds_phase_strengths": {a: 0.0, b: 0.0}}, entry="hough_index_patterns")
    assert task["status"] == "failed"
    assert chemistry == []


def test_a_chemistry_that_cannot_be_read_at_the_start_is_no_prior_and_is_not_asked_again(
        drive, merge_weights, monkeypatch, tmp_path):
    """Unreadable at the start: no weights, as for any chemistry that fails to read. The loaded file is
    not tried a second time later in the run (it could be another measurement's by then)."""
    from backend.api.services import eds_indexing_prior as prior
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("no EDS")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", broken)
    assert _run_to_the_merge(drive, merge_weights, RAW, _hough_body(tmp_path)) is None
    assert len(calls) == 1


def test_a_chemistry_read_while_the_viewer_is_switched_is_not_used(drive, merge_weights, monkeypatch, tmp_path):
    """The read itself takes a moment. If the viewer was switched to another measurement while it ran,
    what came back may be that measurement's: the snapshot is declared unreadable (no prior), not used."""
    from backend.api.services import eds_indexing_prior as prior

    def read_during_a_switch():
        variant_registry.register("ds", RAW2)                    # the switch happens during the read
        return ({"Al": np.full(N, 100.0), "Si": np.zeros(N)}, 3, 3, "multi.h5oina")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", read_during_a_switch)
    assert _run_to_the_merge(drive, merge_weights, RAW, _hough_body(tmp_path)) is None


# ------------------------------------------------------------------------------------- the rescue

def test_the_rescue_of_a_run_uses_the_chemistry_it_started_with(monkeypatch):
    """The particle rescue after the run: the snapshot says there is a Si particle (measurement 1), the
    loaded file now says there is none (measurement 2). It rescues the particle."""
    from backend.api.services import eds_indexing_prior as prior
    from tests.test_eds_particle_rescue import _fake_run
    real_measured = prior.measured_atpct_per_pixel
    si_particle = np.full((20, 20), 1.6)
    si_particle[8:11, 8:12] = 30.0
    route, result, req, blob = _fake_run(monkeypatch, si_particle, {"Al.sht": 0.75, "Si.sht": 0.75})
    monkeypatch.setattr(prior, "measured_atpct_per_pixel", real_measured)    # (_fake_run stubs it)
    state = {"si": si_particle}

    def loaded_file():
        si = state["si"]
        return ({"Al": (100.0 - si).reshape(-1), "Si": si.reshape(-1)}, 20, 20, "multi.h5oina")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", loaded_file)
    snapshot = prior.read_run_chemistry((20, 20))                # measurement 1, when the run started
    state["si"] = np.full((20, 20), 1.6)                         # the viewer is switched: no particle
    report = route._apply_particle_rescue(result, req, result.selection_mask, chemistry=snapshot)
    assert report is not None and report["n_to_particle"] == int(blob.sum())


def test_without_a_snapshot_the_rescue_reads_the_loaded_file_now(monkeypatch):
    """The other half, so the test above proves the snapshot and not the fake: asked of the loaded file
    after the switch, the same rescue finds no particle."""
    from backend.api.services import eds_indexing_prior as prior
    from tests.test_eds_particle_rescue import _fake_run
    real_measured = prior.measured_atpct_per_pixel
    si_particle = np.full((20, 20), 1.6)
    si_particle[8:11, 8:12] = 30.0
    route, result, req, blob = _fake_run(monkeypatch, si_particle, {"Al.sht": 0.75, "Si.sht": 0.75})
    monkeypatch.setattr(prior, "measured_atpct_per_pixel", real_measured)
    flat = np.full((20, 20), 1.6)
    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file",
                        lambda: ({"Al": (100.0 - flat).reshape(-1), "Si": flat.reshape(-1)}, 20, 20, "f"))
    report = route._apply_particle_rescue(result, req, result.selection_mask)
    assert report is None or report.get("n_to_particle", 0) == 0


def test_the_run_hands_its_snapshot_to_the_rescue(drive, monkeypatch, tmp_path, chemistry):
    """The call in the run: the rescue gets the snapshot of the run's start (measurement 1, pure Al,
    so no Si anywhere), although the viewer was switched to measurement 2 by the time it runs."""
    import indexing_controller
    from backend.api.routes import indexing as I
    monkeypatch.setattr(I, "_phase_formulas_for_paths", lambda paths: ["Al", "Si"])
    seen = {}

    def merge(comparison, phase_weights=None, **kwargs):
        return comparison

    def rescue(result, req, selection_mask, progress=None, scan_geometry=None, chemistry=None):
        seen["chemistry"] = chemistry
        raise _Stop("rescue reached")

    monkeypatch.setattr(indexing_controller, "compute_comparison_maps", merge)
    monkeypatch.setattr(indexing_controller, "build_consensus_xmap", lambda comparison: object())
    monkeypatch.setattr(I, "_inject_phase_names", lambda result, req: None)
    monkeypatch.setattr(I, "_attach_indexing_metadata", lambda *a, **k: None)
    monkeypatch.setattr(I, "_apply_particle_rescue", rescue)
    monkeypatch.setattr(I, "_eds_prior_is_on", lambda req: False)
    # the merge helpers that read the per-phase results are given plain objects
    import hough_reflectors
    from backend.api.services.citations import provenance
    monkeypatch.setattr(provenance, "merge_provenance", lambda *a, **k: None)
    monkeypatch.setattr(hough_reflectors, "merge_build_records", lambda *a, **k: None)
    task = drive(RAW, _per_phase_indexer, _hough_body(tmp_path), entry="run_single_phase_method")
    assert task["status"] == "failed" and "rescue reached" in task["error"], task
    snapshot = seen["chemistry"]
    assert snapshot is not None and snapshot.nav_shape == (3, 3)
    assert snapshot.at_maps["Si"].max() == 0.0 and snapshot.at_maps["Al"].min() == 100.0
    assert chemistry == ["1"]


# ------------------------------------------------------------- what the result says about the prior

def _provenance_keys_at_the_rescue(drive, monkeypatch, tmp_path):
    """Run Hough with the prior configured to the point where the rescue is called, and return the keys
    of the citation steps the result holds by then (the prior's step is recorded just before)."""
    import indexing_controller
    from backend.api.routes import indexing as I
    from backend.api.services.citations import provenance
    monkeypatch.setattr(I, "_phase_formulas_for_paths", lambda paths: ["Al", "Si"])
    seen = {}

    def merge(comparison, phase_weights=None, **kwargs):
        return comparison

    def rescue(result, req, selection_mask, progress=None, scan_geometry=None, chemistry=None):
        seen["keys"] = [s["key"] for s in provenance.get_steps(result)]
        raise _Stop("rescue reached")

    monkeypatch.setattr(indexing_controller, "compute_comparison_maps", merge)
    monkeypatch.setattr(indexing_controller, "build_consensus_xmap", lambda comparison: object())
    monkeypatch.setattr(I, "_inject_phase_names", lambda result, req: None)
    monkeypatch.setattr(I, "_attach_indexing_metadata", lambda *a, **k: None)
    monkeypatch.setattr(I, "_apply_particle_rescue", rescue)
    import hough_reflectors
    monkeypatch.setattr(provenance, "merge_provenance", lambda *a, **k: None)
    monkeypatch.setattr(hough_reflectors, "merge_build_records", lambda *a, **k: None)
    task = drive(RAW, _per_phase_indexer, _hough_body(tmp_path), entry="run_single_phase_method")
    assert task["status"] == "failed" and "rescue reached" in task["error"], task
    return seen["keys"]


def test_a_run_that_read_its_chemistry_records_the_prior(drive, monkeypatch, tmp_path, chemistry):
    assert "eds.chemistry_prior" in _provenance_keys_at_the_rescue(drive, monkeypatch, tmp_path)


def test_a_run_whose_snapshot_was_dropped_for_a_switch_records_no_prior(drive, monkeypatch, tmp_path):
    """The prior did not run (the chemistry read during a switch is declared unreadable), so the methods
    text must not say it did."""
    from backend.api.services import eds_indexing_prior as prior

    def read_during_a_switch():
        variant_registry.register("ds", RAW2)                    # the switch happens during the read
        return ({"Al": np.full(N, 100.0), "Si": np.zeros(N)}, 3, 3, "multi.h5oina")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", read_during_a_switch)
    assert "eds.chemistry_prior" not in _provenance_keys_at_the_rescue(drive, monkeypatch, tmp_path)


def test_a_run_whose_chemistry_could_not_be_read_records_no_prior(drive, monkeypatch, tmp_path):
    from backend.api.services import eds_indexing_prior as prior

    def broken():
        raise RuntimeError("no EDS")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", broken)
    assert "eds.chemistry_prior" not in _provenance_keys_at_the_rescue(drive, monkeypatch, tmp_path)


def test_a_chemistry_that_fails_to_read_says_why_in_the_log(monkeypatch, caplog):
    import logging
    from backend.api.services import eds_indexing_prior as prior

    def broken():
        raise RuntimeError("the EDS window cannot be priced")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", broken)
    with caplog.at_level(logging.WARNING, logger=prior.logger.name):
        snapshot = prior.read_run_chemistry((3, 3))
    assert snapshot.at_maps is None and snapshot.nav_shape == (3, 3)
    assert any("the EDS window cannot be priced" in r.getMessage() or
               "the EDS window cannot be priced" in str(r.exc_info) for r in caplog.records), caplog.text


def _prior_lines(task):
    return [m for m in task.get("pending_log", []) if "EDS chemistry prior is off" in m]


def test_a_snapshot_dropped_for_a_switch_is_told_in_the_runs_progress(drive, merge_weights, monkeypatch, tmp_path):
    """The user sees the run window, not the log: a finished run without the prior says why."""
    from backend.api.services import eds_indexing_prior as prior

    def read_during_a_switch():
        variant_registry.register("ds", RAW2)
        return ({"Al": np.full(N, 100.0), "Si": np.zeros(N)}, 3, 3, "multi.h5oina")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", read_during_a_switch)
    task = drive(RAW, _per_phase_indexer, _hough_body(tmp_path), entry="run_single_phase_method")
    lines = _prior_lines(task)
    assert len(lines) == 1 and "changed while" in lines[0], task.get("pending_log")


def test_a_chemistry_that_could_not_be_read_is_told_in_the_runs_progress(drive, merge_weights, monkeypatch, tmp_path):
    from backend.api.services import eds_indexing_prior as prior

    def broken():
        raise RuntimeError("the EDS window cannot be priced")

    monkeypatch.setattr(prior, "_build_at_pct_maps_for_loaded_file", broken)
    task = drive(RAW, _per_phase_indexer, _hough_body(tmp_path), entry="run_single_phase_method")
    lines = _prior_lines(task)
    assert len(lines) == 1 and "the EDS window cannot be priced" in lines[0], task.get("pending_log")


def test_a_run_that_read_its_chemistry_does_not_say_the_prior_is_off(drive, merge_weights, chemistry, tmp_path):
    task = drive(RAW, _per_phase_indexer, _hough_body(tmp_path), entry="run_single_phase_method")
    assert _prior_lines(task) == []
