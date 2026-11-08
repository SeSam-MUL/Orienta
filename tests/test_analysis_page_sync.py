"""/status carries the grain-statistics settings and the reconstruction parameters, the batch takes the
reconstruction parameters and reports processed, skipped and failed files, the export reports failed steps."""
import json
import urllib.parse
from types import SimpleNamespace

import numpy as np
import pytest
from tests.batch_load_stub import patch_batch_load
from fastapi.testclient import TestClient
from fastapi import HTTPException

from analysis.ebsd_dataset import EBSDDataset
from tests.synthetic_grain_map import make_map, to_xmap
from tests.test_grain_stats_phase import _tile_map

DEFAULTS = {"edge_rule": "matlab_script", "ecd_basis": "cell_area", "stats_phase": None}


@pytest.fixture
def client_map():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    m = make_map(shape=(60, 80), n_seeds=40)
    saved = ar._dataset, dict(ar._grain_stats_settings)
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)       # native band contrast
    ar._dataset = EBSDDataset(xmap, step_size=m["step"])
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


@pytest.fixture
def client_two_phase():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    ds, _, _, _ = _tile_map(12, 12, 8)
    ds.xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, ds.xmap.size)    # native band contrast
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


def _reconstruct(client, **extra):
    body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 0.0, "fill": False, **extra}
    r = client.post("/api/analysis/grains/reconstruct", json=body)
    assert r.status_code == 200, r.text
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    return task["result"]


# ---------------------------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------------------------
def test_status_without_a_dataset_is_just_not_loaded():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    saved = ar._dataset
    ar._dataset = None
    try:
        assert TestClient(app).get("/api/analysis/status").json() == {"loaded": False}
    finally:
        ar._dataset = saved


def test_status_before_a_reconstruction_has_the_defaults_and_no_parameters(client_map):
    client, _ = client_map
    s = client.get("/api/analysis/status").json()
    assert (s["edge_rule"], s["ecd_basis"], s["stats_phase"]) == ("matlab_script", "cell_area", None)
    assert s["has_grains"] is False and s["recon_params"] is None


def test_status_carries_the_settings_and_parameters_of_the_last_reconstruction(client_two_phase):
    client, _ = client_two_phase
    _reconstruct(client, misorientation_threshold=6.5, min_grain_size=2, min_intercept_um=2.5, alpha=1.5,
                 fill=True, edge_rule="mtex", ecd_basis="measured_pixels", stats_phase=1)
    s = client.get("/api/analysis/status").json()
    assert (s["edge_rule"], s["ecd_basis"], s["stats_phase"]) == ("mtex", "measured_pixels", 1)
    assert s["has_grains"] is True
    assert s["recon_params"] == {"threshold": 6.5, "min_pixels": 2, "min_intercept": 2.5, "alpha": 1.5, "fill": True}


def test_status_follows_a_grain_size_call_and_keeps_most_measured(client_two_phase):
    client, _ = client_two_phase
    _reconstruct(client)
    r = client.post("/api/analysis/grain-size", json={"edge_rule": "outer_row", "stats_phase": "most_measured"})
    assert r.status_code == 200, r.text
    s = client.get("/api/analysis/status").json()
    assert (s["edge_rule"], s["stats_phase"]) == ("outer_row", "most_measured")
    assert s["recon_params"]["threshold"] == 5.0           # the grains are still the ones of the reconstruction


def test_status_has_no_parameters_for_grains_that_were_not_reconstructed_here(client_two_phase):
    client, ar = client_two_phase
    _reconstruct(client)
    ar._dataset.grain_recon_params = None                  # e.g. a quality filter dropped them
    s = client.get("/api/analysis/status").json()
    assert s["recon_params"] is None


def test_status_drops_the_parameters_of_vanished_grains(client_two_phase):
    client, ar = client_two_phase
    _reconstruct(client)
    ar._dataset.grains = None                              # parameters left behind without grains
    assert client.get("/api/analysis/status").json()["recon_params"] is None


def test_status_reports_the_stored_phase_choice_without_grains(client_two_phase):
    client, ar = client_two_phase
    ar._grain_stats_settings.update(stats_phase="most_measured")
    assert client.get("/api/analysis/status").json()["stats_phase"] == "most_measured"
    assert ar._edge_fields()["stats_phase"] == "most_measured"
    ar._grain_stats_settings.update(stats_phase=1)
    assert ar._edge_fields()["stats_phase"] == 1
    ar._grain_stats_settings.update(stats_phase=None)
    assert ar._edge_fields()["stats_phase"] is None


# ---------------------------------------------------------------------------------------------
# the dataset a route captured is the dataset it checks and exports
# ---------------------------------------------------------------------------------------------
def test_reset_replaces_the_values_without_emptying_the_dict_first(monkeypatch):
    import backend.api.routes.analysis as ar

    class Watched(dict):
        def clear(self):
            raise AssertionError("the settings dict must never be empty for a reader")

    watched = Watched(edge_rule="mtex", ecd_basis="measured_pixels", stats_phase=1)
    monkeypatch.setattr(ar, "_grain_stats_settings", watched)
    ar._reset_grain_stats_settings()
    assert dict(watched) == DEFAULTS


def test_the_phase_check_uses_the_dataset_it_is_given(client_two_phase):
    client, ar = client_two_phase
    one_phase, *_ = _tile_map(2, 4, 0)                    # phase 0 only
    assert [o["id"] for o in ar._phase_options(one_phase)] == [0]
    ar._check_stats_phase(1, ar._dataset)                  # a phase of the two-phase map
    with pytest.raises(HTTPException) as e:
        ar._check_stats_phase(1, one_phase)                # not a phase of the other map, whatever is loaded
    assert e.value.status_code == 400
    ar._check_stats_phase("most_measured", one_phase)
    ar._check_stats_phase(None, one_phase)


def test_the_grain_size_route_checks_the_phase_against_the_captured_dataset(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _reconstruct(client)
    first = ar._dataset
    seen = []
    real = ar._check_stats_phase
    monkeypatch.setattr(ar, "_check_stats_phase", lambda phase, dataset=None: seen.append(dataset) or real(phase, dataset))
    assert client.post("/api/analysis/grain-size", json={"stats_phase": 1}).status_code == 200
    assert seen == [first]


def _stub_slow_steps(monkeypatch):
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no texture here")))


def test_the_export_collects_from_the_dataset_it_started_with(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client, misorientation_threshold=6.5, edge_rule="outer_row", stats_phase=1)
    first = ar._dataset
    import analysis.grain_analysis as ga
    import analysis.deformation_analysis as da
    real = ga.analyze_grain_size
    deform_args = []

    def swapping(ds, grains, **kw):
        out = real(ds, grains, **kw)
        ar._dataset = None                                 # another map is loaded while the export runs
        ar._reset_grain_stats_settings()
        return out

    monkeypatch.setattr(ga, "analyze_grain_size", swapping)
    monkeypatch.setattr(da, "analyze_deformation", lambda ds, grains, **kw: deform_args.append((ds, grains)) or "d")
    warnings = []
    results = ar._collect_export_results(warnings)
    assert results["basic_info"]["gb_threshold_deg"] == 6.5
    assert deform_args and deform_args[0][0] is first and deform_args[0][1] is first.grains
    assert results["deformation"] == "d"


def test_a_failing_export_step_is_reported_not_only_logged(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    import analysis.grain_analysis as ga
    monkeypatch.setattr(ga, "analyze_grain_size", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "grain_size" not in results
    assert any(w.startswith("grain_size") and "boom" in w for w in warnings), warnings
    assert any(w.startswith("texture") for w in warnings), warnings
    assert ar._collect_export_results() is not None       # the argument is optional


def test_the_export_routes_answer_with_the_warnings(client_two_phase, monkeypatch, tmp_path):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    import analysis.grain_analysis as ga
    import analysis.excel_exporter as ee
    monkeypatch.setattr(ga, "analyze_grain_size", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(ee, "export_to_excel", lambda path, name, results: None)
    r = client.post("/api/analysis/export/excel", json={"output_path": str(tmp_path / "o.xlsx"), "dataset_name": "d"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    assert task["result"]["path"] == str(tmp_path / "o.xlsx")
    assert any(w.startswith("grain_size") and "boom" in w for w in task["result"]["warnings"])
    d = client.get("/api/analysis/export/excel-download?name=d")
    assert d.status_code == 200
    from_header = json.loads(urllib.parse.unquote(d.headers["x-orienta-warnings"]))
    assert any(w.startswith("grain_size") and "boom" in w for w in from_header)
    assert from_header == [w for w in from_header if isinstance(w, str)]


def test_an_export_without_grains_reports_only_that(client_map, monkeypatch, tmp_path):
    client, ar = client_map
    import analysis.excel_exporter as ee
    monkeypatch.setattr(ee, "export_to_excel", lambda path, name, results: None)
    r = client.post("/api/analysis/export/excel", json={"output_path": str(tmp_path / "o.xlsx"), "dataset_name": "d"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed"
    assert [w.split(":")[0] for w in task["result"]["warnings"]] == ["grains"]       # nothing else failed


# ---------------------------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------------------------
def _batch_result(**kw):
    from analysis.batch_processor import BatchResult
    return BatchResult(**{"processed": [], "failed": [], "skipped": [], "n_files": 0, **kw})


def test_batch_route_passes_the_reconstruction_parameters_only_when_given(client_map, monkeypatch):
    client, _ = client_map
    import analysis.batch_processor as bp
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder",
                        lambda folder, out, progress_callback=None, **kw: seen.append(kw) or _batch_result(processed=["a.h5"], n_files=1))
    for body in ({"folder_path": "f", "output_path": "o"},
                 {"folder_path": "f", "output_path": "o", "misorientation_threshold": 5.0, "min_grain_size": 3,
                  "alpha": 0.0, "fill": False},
                 {"folder_path": "f", "output_path": "o", "min_intercept_um": 2.5}):
        r = client.post("/api/analysis/batch", json=body)
        assert r.status_code == 200
        task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
        assert task["status"] == "completed", task
    assert seen[0] == {}
    assert seen[1] == {"gb_threshold_deg": 5.0, "min_grain_pixels": 3, "grain_alpha": 0.0, "grain_fill": False}
    assert seen[2] == {"min_intercept_um": 2.5}          # recorded in Sheet 1 row 11, as the Grain tab has it


def test_batch_route_rejects_nonsense_parameters(client_map):
    client, _ = client_map
    base = {"folder_path": "f", "output_path": "o"}
    for bad in ({"misorientation_threshold": 0}, {"min_grain_size": 0}, {"alpha": -1}, {"alpha": 99},
                {"min_intercept_um": -1}):
        assert client.post("/api/analysis/batch", json={**base, **bad}).status_code == 422, bad


def test_the_processor_reconstructs_with_the_given_parameters(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    import analysis.grain_analysis as ga
    monkeypatch.setattr(bp, "analyze_texture", lambda *a, **k: None)
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    real = ga.reconstruct_grains
    seen = []
    monkeypatch.setattr(ga, "reconstruct_grains", lambda ds, **kw: seen.append(kw) or real(ds, **kw))
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), gb_threshold_deg=7.0,
                             min_grain_pixels=2, grain_alpha=1.5, grain_fill=False)
    proc.process_file(tmp_path / "s.h5")
    assert seen == [{"angle_threshold_deg": 7.0, "min_pixels": 2, "alpha": 1.5, "fill": False}]
    default = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"))
    assert default.grain_fill is True


def _fake_texture(dataset=None, *a, **k):
    # the texture step (about 90 s of orix symmetry set-up) is not under test; the exporter needs these fields.
    # The phase is the one the real step would take, so the RX / texture split sees the phases of the map.
    from analysis.texture_analysis import _phase_id_of, _phase_label
    pid = _phase_id_of(dataset, k.get("texture_phase", "most_measured"))
    return SimpleNamespace(texture_index=1.0, entropy=1.0, component_fractions={}, plane_111_fraction=0.1,
                           plane_100_fraction=0.1, plane_110_fraction=0.1, phase_id=pid,
                           phase_name=_phase_label(dataset, pid))


def _mixed_folder(tmp_path, monkeypatch):
    """bad.h5 cannot be read, nograins.h5 gives no grains, ok.h5 is processed."""
    from analysis import batch_processor as bp
    import analysis.grain_analysis as ga
    for name in ("bad.h5", "nograins.h5", "ok.h5"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)

    def load(path):
        if str(path).endswith("bad.h5"):
            raise OSError("cannot read the file")
        return xmap

    patch_batch_load(monkeypatch, bp, load)
    real = ga.reconstruct_grains
    calls = []

    def reconstruct(ds, **kw):
        calls.append(1)
        return SimpleNamespace(n_grains=0) if len(calls) == 1 else real(ds, **kw)

    monkeypatch.setattr(ga, "reconstruct_grains", reconstruct)
    return bp


def test_batch_process_folder_returns_what_was_processed_failed_and_skipped(tmp_path, monkeypatch):
    bp = _mixed_folder(tmp_path, monkeypatch)
    res = bp.batch_process_folder(str(tmp_path), str(tmp_path / "doc.xlsx"))
    assert res.n_files == 3 and res.n_processed == 1
    assert [p.endswith("ok.h5") for p in res.processed] == [True]
    assert len(res.failed) == 1 and res.failed[0]["file"].endswith("bad.h5")
    assert "cannot read the file" in res.failed[0]["reason"]
    assert len(res.skipped) == 1 and res.skipped[0]["file"].endswith("nograins.h5")
    assert "no grains" in res.skipped[0]["reason"]
    assert (tmp_path / "doc.xlsx").exists()
    d = res.to_dict()
    assert d["n_files"] == 3 and d["n_processed"] == 1 and d["failed"] == res.failed and d["skipped"] == res.skipped


def test_batch_route_says_what_happened_to_each_file(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    _mixed_folder(tmp_path, monkeypatch)
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(tmp_path / "doc.xlsx")})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    assert task["progress"] == 1.0
    res = task["result"]
    assert res["n_files"] == 3 and res["n_processed"] == 1
    assert res["failed"][0]["file"].endswith("bad.h5") and "cannot read" in res["failed"][0]["reason"]
    assert res["skipped"][0]["file"].endswith("nograins.h5")
    assert "1 of 3" in task["message"] and "1 skipped" in task["message"] and "1 failed" in task["message"]


def test_a_batch_where_every_file_fails_is_not_announced_as_done(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    from analysis import batch_processor as bp

    def load(path):
        raise OSError("disk gone")

    for name in ("a.h5", "b.h5"):
        (tmp_path / name).write_bytes(b"")
    patch_batch_load(monkeypatch, bp, load)
    out = tmp_path / "doc.xlsx"
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(out)})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed", task
    assert "None of the 2 files" in task["error"] and "disk gone" in task["error"]
    assert task["result"]["n_processed"] == 0 and len(task["result"]["failed"]) == 2
    assert not out.exists()                                # no empty workbook that reads like a result


def test_a_batch_where_every_file_is_skipped_is_not_announced_as_done(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    from analysis import batch_processor as bp
    import analysis.grain_analysis as ga
    (tmp_path / "a.h5").write_bytes(b"")
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    monkeypatch.setattr(ga, "reconstruct_grains", lambda ds, **kw: SimpleNamespace(n_grains=0))
    out = tmp_path / "doc.xlsx"
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(out)})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed", task
    assert "no grains" in task["error"]
    assert task["result"]["skipped"][0]["file"].endswith("a.h5")
    assert not out.exists()


def test_an_empty_folder_is_a_visible_failure(client_map, tmp_path):
    client, _ = client_map
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(tmp_path / "doc.xlsx")})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed", task
    assert "No EBSD files" in task["error"]
    assert task["result"]["n_files"] == 0


# ---------------------------------------------------------------------------------------------
# grain state under the lock, one read of the dataset
# ---------------------------------------------------------------------------------------------
class _RecordingLock:
    """Stands in for ``_grain_state_lock``: says whether it is held and how often it was taken."""

    def __init__(self):
        self.held = False
        self.entered = 0

    def __enter__(self):
        assert not self.held, "the lock is not re-entrant"
        self.held = True
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.held = False
        return False


def _recording_dataset(lock, reads):
    """A dataset that notes, with every read of ``grains`` / ``grain_recon_params``, whether the lock is held."""
    class Recording(EBSDDataset):
        @property
        def grains(self):
            reads.append(("grains", lock.held))
            return self._grains

        @grains.setter
        def grains(self, value):
            self._grains = value

        @property
        def grain_recon_params(self):
            reads.append(("grain_recon_params", lock.held))
            return self._recon

        @grain_recon_params.setter
        def grain_recon_params(self, value):
            self._recon = value

    return Recording


def _first_reads(reads):
    first = {}
    for name, held in reads:
        first.setdefault(name, held)
    return first


def test_status_reads_the_grain_state_under_the_lock(monkeypatch):
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    lock, reads = _RecordingLock(), []
    m = make_map(shape=(30, 40), n_seeds=10)
    ds = _recording_dataset(lock, reads)(to_xmap(m), step_size=m["step"])
    ds.grains = SimpleNamespace(n_grains=3)
    ds.grain_recon_params = {"gb_threshold_deg": 5.0, "min_pixels": 3, "alpha": 3.1, "min_intercept_um": 1.5, "fill": True}
    saved = ar._dataset
    monkeypatch.setattr(ar, "_grain_state_lock", lock)
    ar._dataset = ds
    try:
        reads.clear()
        s = TestClient(app).get("/api/analysis/status").json()
    finally:
        ar._dataset = saved
    assert s["has_grains"] is True and s["recon_params"]["threshold"] == 5.0
    assert lock.entered == 1 and lock.held is False
    assert _first_reads(reads) == {"grains": True, "grain_recon_params": True}
    assert all(held for _, held in reads)          # nothing read outside it


def test_the_export_reads_the_grain_state_under_the_lock(monkeypatch):
    import backend.api.routes.analysis as ar
    _stub_slow_steps(monkeypatch)
    lock, reads = _RecordingLock(), []
    ds, g, *_ = _tile_map(6, 6, 4)
    rec = _recording_dataset(lock, reads)(ds.xmap, step_size=ds.step_size)
    rec.grains = g
    rec.grain_recon_params = {"gb_threshold_deg": 5.0, "min_pixels": 1, "alpha": 0.0, "min_intercept_um": 1.5}
    saved = ar._dataset
    monkeypatch.setattr(ar, "_grain_state_lock", lock)
    ar._dataset = rec
    try:
        reads.clear()
        results = ar._collect_export_results([])
    finally:
        ar._dataset = saved
    assert results["basic_info"]["gb_threshold_deg"] == 5.0
    assert lock.entered == 1 and lock.held is False
    assert _first_reads(reads) == {"grains": True, "grain_recon_params": True}


def test_reconstruct_checks_the_phase_against_the_dataset_it_captured(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    first = ar._dataset
    seen = []
    real = ar._check_stats_phase
    monkeypatch.setattr(ar, "_check_stats_phase", lambda phase, dataset=None: seen.append(dataset) or real(phase, dataset))
    _reconstruct(client, stats_phase=1)
    assert seen == [first]


@pytest.mark.parametrize("trigger", ["xmap", "step_size"])
def test_status_reads_the_dataset_once(client_two_phase, trigger):
    """The phases, the step and the grain state come from one object even when another map is loaded meanwhile,
    whether that happens before the phase loop (first read of ``xmap``) or after it (``step_size``)."""
    import asyncio
    client, ar = client_two_phase
    first = ar._dataset
    other, *_ = _tile_map(3, 4, 0)                   # one phase only

    class Probe(type(first)):
        def _swap(self, name):
            if name == trigger and not self.__dict__.get("_swapped"):
                self.__dict__["_swapped"] = True
                ar._dataset = other                  # another map is loaded while /status is being answered
            return self.__dict__["_" + name]

        @property
        def xmap(self):
            return self._swap("xmap")

        @xmap.setter
        def xmap(self, value):
            self.__dict__["_xmap"] = value

        @property
        def step_size(self):
            return self._swap("step_size")

        @step_size.setter
        def step_size(self, value):
            self.__dict__["_step_size"] = value

    probe = Probe(first.xmap, step_size=first.step_size)
    ar._dataset = probe
    s = asyncio.run(ar.analysis_status())
    assert ar._dataset is other                      # the swap did happen during the call
    assert len(s["phases"]) == 2                     # the probe's phases, not the other map's one
    assert [o["id"] for o in s["phase_options"]] == [0, 1]
    assert s["shape"] == list(probe.shape)


def test_batch_parameters_are_written_to_sheet_1(client_map, monkeypatch, tmp_path):
    from openpyxl import load_workbook
    client, _ = client_map
    _mixed_folder(tmp_path, monkeypatch)
    out = tmp_path / "doc.xlsx"
    r = client.post("/api/analysis/batch", json={
        "folder_path": str(tmp_path), "output_path": str(out), "misorientation_threshold": 7.0,
        "min_grain_size": 2, "alpha": 1.5, "fill": False})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    ws = load_workbook(str(out))["Sheet1"]
    labels = {ws.cell(row=row, column=1).value: row for row in range(2, 70) if ws.cell(row=row, column=1).value}
    col = 2
    assert ws.cell(row=labels["gb_threshold_deg"], column=col).value == 7.0
    assert labels["gb_threshold_deg"] == 9 and labels["min_pixels"] == 10
    assert ws.cell(row=10, column=col).value == 2
    assert ws.cell(row=48, column=col).value == 1.5


def test_non_ascii_warnings_survive_the_header(client_two_phase, monkeypatch, tmp_path):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    import analysis.grain_analysis as ga
    import analysis.excel_exporter as ee
    monkeypatch.setattr(ga, "analyze_grain_size",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Gr\u00f6\u00dfe \u00b5m; a; b\nline 2")))
    monkeypatch.setattr(ee, "export_to_excel", lambda path, name, results: None)
    d = client.get("/api/analysis/export/excel-download?name=d")
    assert d.status_code == 200
    raw = d.headers["x-orienta-warnings"]
    assert raw.isascii() and "\n" not in raw
    warnings = json.loads(urllib.parse.unquote(raw))
    assert "grain_size: RuntimeError: Gr\u00f6\u00dfe \u00b5m; a; b\nline 2" in warnings   # one entry, ";" and newline intact
    r = client.post("/api/analysis/export/excel", json={"output_path": str(tmp_path / "o.xlsx"), "dataset_name": "d"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert "grain_size: RuntimeError: Gr\u00f6\u00dfe \u00b5m; a; b\nline 2" in task["result"]["warnings"]
    assert isinstance(task["result"]["warnings"], list)


def test_a_long_warning_reason_is_cut_in_the_header_and_the_result(client_two_phase, monkeypatch, tmp_path):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    import analysis.grain_analysis as ga
    import analysis.excel_exporter as ee
    monkeypatch.setattr(ga, "analyze_grain_size", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x" * 5000)))
    monkeypatch.setattr(ee, "export_to_excel", lambda path, name, results: None)
    d = client.get("/api/analysis/export/excel-download?name=d")
    from_header = json.loads(urllib.parse.unquote(d.headers["x-orienta-warnings"]))
    r = client.post("/api/analysis/export/excel", json={"output_path": str(tmp_path / "o.xlsx"), "dataset_name": "d"})
    from_task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()["result"]["warnings"]
    for warnings in (from_header, from_task):
        grain = [w for w in warnings if w.startswith("grain_size: ")][0]
        assert len(grain) - len("grain_size: ") == 300 and grain.endswith("...")
    assert len(d.headers["x-orienta-warnings"]) < 8000


def test_the_warning_header_is_exposed_to_a_browser_page(client_map):
    client, _ = client_map
    r = client.get("/api/analysis/export/excel-download?name=d", headers={"Origin": "http://localhost:5173"})
    exposed = r.headers.get("access-control-expose-headers", "")
    assert "X-Orienta-Warnings" in exposed and "X-Orienta-Code" in exposed


def test_too_few_bc_points_for_the_fit_is_reported(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    import analysis.bc_analysis as ba
    monkeypatch.setattr(ba, "fit_bc_gmm", lambda ds: None)
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "bc_gmm" not in results
    assert "bc_gmm: too few points for the fit" in warnings


def test_the_rx_block_carries_the_low_kam_fractions_of_the_rx_result(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    results = ar._collect_export_results([])
    rx, kam = results["rx"], results["deformation"].kam
    finite = np.isfinite(kam)
    want = float(np.sum((kam < 0.55) & finite) / np.sum(finite) * 100.0)
    assert rx.low_kam_fraction == pytest.approx(want)
    assert rx.low_kam_fraction > 0.0                     # not the placeholder zero
    assert rx.low_gkam_fraction > 0.0


def test_the_rx_block_says_so_when_it_cannot_be_made(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    import analysis.deformation_analysis as da
    monkeypatch.setattr(da, "analyze_deformation", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no kam")))
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "rx" not in results and "deformation" not in results
    assert any(w.startswith("rx: ") for w in warnings) and any(w.startswith("deformation: RuntimeError: no kam") for w in warnings)


# ---------------------------------------------------------------------------------------------
# export without native band contrast, warnings with a class name
# ---------------------------------------------------------------------------------------------
def test_the_export_leaves_out_bc_gmm_and_rx_without_native_band_contrast(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    ds = ar._dataset
    ds.xmap.prop["pq"] = np.random.default_rng(1).uniform(0, 1, ds.xmap.size)       # a surrogate only
    del ds.xmap.prop["bc"]
    assert ds.has_native_bc is False
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "bc_gmm" not in results and "bc_histogram" not in results and "rx" not in results
    assert "bc_gmm: no band contrast in this map" in warnings
    assert "rx: no band contrast in this map" in warnings
    assert "grain_size" in results and "deformation" in results         # the steps that do not need it still run


def test_the_export_with_native_band_contrast_still_has_both(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "bc_histogram" in results and "rx" in results
    assert not any(w.startswith(("rx:", "bc_gmm:")) for w in warnings if "native" in w)


def test_a_reconstruction_without_grains_gives_no_rx_block(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client, min_grain_size=10 ** 6)
    assert ar._dataset.grains.n_grains == 0
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "rx" not in results
    assert any(w.startswith("rx: the reconstruction found no grains") for w in warnings), warnings


def test_a_warning_names_the_exception_class_even_for_an_empty_message(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    import analysis.grain_analysis as ga
    import analysis.deformation_analysis as da
    monkeypatch.setattr(ga, "analyze_grain_size", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad value")))
    monkeypatch.setattr(da, "analyze_deformation", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    warnings = []
    ar._collect_export_results(warnings)
    assert "grain_size: ValueError: bad value" in warnings
    assert "deformation: AssertionError: " in warnings


def test_the_download_route_runs_in_the_threadpool():
    import inspect
    import backend.api.routes.analysis as ar
    assert not inspect.iscoroutinefunction(ar.export_excel_download)


def test_a_failed_task_has_its_error_before_its_status(client_map, monkeypatch, tmp_path):
    client, ar = client_map

    class Recording(dict):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.writes = []

        def __setitem__(self, key, value):
            self.writes.append((key, value))
            super().__setitem__(key, value)

    made = []

    def track(task_id, initial):
        task = Recording(initial)
        made.append(task)
        ar._analysis_tasks[task_id] = task

    monkeypatch.setattr(ar, "_track_task", track)
    import analysis.batch_processor as bp
    monkeypatch.setattr(bp, "batch_process_folder", lambda *a, **k: (_ for _ in ()).throw(OSError("folder gone")))
    import analysis.excel_exporter as ee
    monkeypatch.setattr(ee, "export_to_excel", lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone")))
    import analysis.grain_analysis as ga
    monkeypatch.setattr(ga, "reconstruct_grains", lambda *a, **k: (_ for _ in ()).throw(ValueError("no recon")))
    client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o"})
    client.post("/api/analysis/export/excel", json={"output_path": str(tmp_path / "o.xlsx"), "dataset_name": "d"})
    client.post("/api/analysis/grains/reconstruct", json={"misorientation_threshold": 5.0, "min_grain_size": 1})
    assert len(made) == 3
    for task in made:
        keys = [k for k, v in task.writes if not (k == "status" and v != "failed")]
        assert task["status"] == "failed" and task["error"]
        assert keys.index("error") < keys.index("status"), task.writes


# --- batch ------------------------------------------------------------------------------------------
def _xmap_with(quality):
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop[quality] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    return xmap


def test_a_batch_file_without_native_band_contrast_skips_bc_gmm_and_rx_and_says_so(client_map, monkeypatch, tmp_path):
    from openpyxl import load_workbook
    client, _ = client_map
    from analysis import batch_processor as bp
    for name in ("a.h5", "b.h5"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: _xmap_with("pq" if str(path).endswith("a.h5") else "bc"))
    out = tmp_path / "doc.xlsx"
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(out)})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    res = task["result"]
    assert res["n_processed"] == 2
    assert sorted((w["file"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1], w["reason"]) for w in res["warnings"]) == [
        ("a.h5", "bc_gmm: no band contrast in this map"),
        ("a.h5", "rx: no band contrast in this map")]
    assert "left out" in task["message"] and "a.h5: bc_gmm, rx" in task["message"]
    assert "1 file(s) with analysis steps left out" in task["message"]       # files, not steps
    ws = load_workbook(str(out))["Sheet1"]
    assert ws.cell(row=41, column=2).value is None         # a.h5: no rx fraction written
    assert ws.cell(row=41, column=3).value is not None     # b.h5: it has native band contrast


def test_a_batch_without_any_step_left_out_has_an_empty_warning_list(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    _mixed_folder(tmp_path, monkeypatch)
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(tmp_path / "doc.xlsx")})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["result"]["warnings"] == []                # the file that failed or was skipped adds none


def test_a_file_that_cannot_be_exported_is_listed_as_failed_and_the_batch_goes_on(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    from analysis.excel_exporter import ExcelExporter
    for name in ("a.h5", "b.h5"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: _xmap_with("bc"))
    real = ExcelExporter.export_dataset

    def export(self, name, results):
        if name == "a":
            self.dataset_count += 1                      # the real export takes its column first, then writes
            self.wb["Sheet1"].cell(row=2, column=self.dataset_count + 1, value="half")
            raise PermissionError("cannot write")
        return real(self, name, results)

    monkeypatch.setattr(ExcelExporter, "export_dataset", export)
    out = tmp_path / "doc.xlsx"
    res = bp.batch_process_folder(str(tmp_path), str(out))
    assert [p.endswith("b.h5") for p in res.processed] == [True]
    assert len(res.failed) == 1 and res.failed[0]["file"].endswith("a.h5")
    assert res.failed[0]["reason"].startswith("export: PermissionError") and "cannot write" in res.failed[0]["reason"]
    assert out.exists()
    from openpyxl import load_workbook
    ws = load_workbook(str(out))["Sheet1"]
    assert "export failed" in ws.cell(row=1, column=2).value          # the half-written column is marked
    assert ws.cell(row=1, column=3).value == "b"                      # and b.h5 has its own


# ---------------------------------------------------------------------------------------------
# band contrast of a CTF, one wording
# ---------------------------------------------------------------------------------------------
def _write_ctf(path, ny=12, nx=15, eulers=((10.0, 20.0, 30.0), (80.0, 40.0, 5.0))):
    """A small Oxford-style Channel Text File: two orientation blocks and a Band Contrast column.

    orix cannot write .ctf, so the text is written here and read back through the orix reader under test.
    """
    header = [
        "Channel Text File", "Prj\ttest", "Author\t", "JobMode\tGrid", f"XCells\t{nx}", f"YCells\t{ny}",
        "XStep\t1.0", "YStep\t1.0", "AcqE1\t0", "AcqE2\t0", "AcqE3\t0", "Euler angles refer to Sample Coordinate system (CS0)!",
        "Mag\t1000", "Coverage\t100", "Device\t0", "KV\t20", "TiltAngle\t70", "TiltAxis\t0",
        "Phases\t1", "4.05;4.05;4.05\t90;90;90\tAluminium\t11\t225\t0\t0\t0",
        "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS",
    ]
    rng = np.random.default_rng(3)
    rows, bcs = [], []
    for y in range(ny):
        for x in range(nx):
            e = eulers[0] if x < nx // 2 else eulers[1]
            bcs.append(float(rng.uniform(20, 200)))
            rows.append("\t".join(str(v) for v in (1, x * 1.0, y * 1.0, 8, 0, *e, 0.5, bcs[-1], 100)))
    Path(path).write_text("\n".join(header + rows) + "\n")
    return np.array(bcs)                                  # the Band Contrast column as written


from pathlib import Path


def test_a_ctf_band_contrast_column_counts_as_band_contrast(tmp_path):
    import orix.io
    _write_ctf(tmp_path / "t.ctf")
    xmap = orix.io.load(str(tmp_path / "t.ctf"))
    assert "BC" in xmap.prop and "bc" not in xmap.prop          # orix keeps the Oxford column as "BC"
    ds = EBSDDataset(xmap, step_size=1.0)
    assert ds.has_native_bc is True
    np.testing.assert_allclose(ds.bc, xmap.prop["BC"])           # the real column, not the constant 128


def test_the_bc_key_is_found_whatever_its_capitalisation():
    for key in ("bc", "BC", "Bc"):
        xmap = to_xmap(make_map(shape=(10, 12), n_seeds=4))
        xmap.prop[key] = np.arange(xmap.size, dtype=float)
        ds = EBSDDataset(xmap, step_size=1.0)
        assert ds.has_native_bc is True and ds.bc[5] == 5.0


def test_a_batch_over_a_real_ctf_runs_the_band_contrast_steps(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    _write_ctf(tmp_path / "with_bc.ctf")
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), min_grain_pixels=1, grain_alpha=0.0)
    with_bc = proc.process_file(tmp_path / "with_bc.ctf")
    assert with_bc is not None and "rx" in with_bc and "bc_gmm" in with_bc and "bc_histogram" in with_bc
    # no analysis step is left out, and the .ctf header's XStep 1.0 is a measured step, so there is no step note
    assert [w["reason"] for w in proc.warnings] == []



def test_the_three_bc_places_use_one_wording(client_two_phase, monkeypatch):
    from analysis.ebsd_dataset import NO_BAND_CONTRAST
    client, ar = client_two_phase
    ds = ar._dataset
    del ds.xmap.prop["bc"]
    for url, body in (("/api/analysis/bc_gmm", None), ("/api/analysis/quality-filter", {})):
        r = client.post(url, json=body) if body is not None else client.post(url)
        assert r.status_code == 400 and NO_BAND_CONTRAST in r.json()["detail"], (url, r.text)
    _stub_slow_steps(monkeypatch)
    _reconstruct(client)
    warnings = []
    ar._collect_export_results(warnings)
    assert f"bc_gmm: {NO_BAND_CONTRAST}" in warnings and f"rx: {NO_BAND_CONTRAST}" in warnings


# --- the workbook --------------------------------------------------------------------------------
def test_a_failing_save_keeps_the_result_and_moves_the_files_out_of_processed(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    from analysis.excel_exporter import ExcelExporter
    for name in ("a.h5", "b.h5"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: _xmap_with("bc"))
    closed = []
    monkeypatch.setattr(ExcelExporter, "save", lambda self: (_ for _ in ()).throw(PermissionError("open in Excel")))
    real_close = ExcelExporter.close
    monkeypatch.setattr(ExcelExporter, "close", lambda self: closed.append(1) or real_close(self))
    res = bp.batch_process_folder(str(tmp_path), str(tmp_path / "doc.xlsx"))
    assert closed == [1]
    assert res.processed == [] and res.n_processed == 0 and res.workbook_path is None
    assert "doc.xlsx could not be written" in res.save_error and "PermissionError" in res.save_error
    assert "nothing of this run was saved" in res.save_error
    assert [f["file"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for f in res.failed] == ["a.h5", "b.h5"]
    assert all("workbook could not be written" in f["reason"] for f in res.failed)
    assert res.to_dict()["save_error"] == res.save_error and res.warnings == []


def test_the_route_ends_failed_with_the_result_when_the_save_fails(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    from analysis import batch_processor as bp
    from analysis.excel_exporter import ExcelExporter
    (tmp_path / "a.h5").write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: _xmap_with("bc"))
    monkeypatch.setattr(ExcelExporter, "save", lambda self: (_ for _ in ()).throw(PermissionError("open in Excel")))
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(tmp_path / "doc.xlsx")})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed", task
    assert "doc.xlsx could not be written" in task["error"] and "open in Excel" in task["error"]
    assert task["result"]["n_processed"] == 0 and task["result"]["save_error"] == task["error"]
    assert len(task["result"]["failed"]) == 1


def test_the_pre_flight_fails_before_any_work_when_the_workbook_cannot_be_written(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    from analysis import batch_processor as bp
    (tmp_path / "a.h5").write_bytes(b"")
    loaded = []
    patch_batch_load(monkeypatch, bp, lambda path: loaded.append(path) or _xmap_with("bc"))
    monkeypatch.setattr(bp.tempfile, "TemporaryFile", lambda **kw: (_ for _ in ()).throw(PermissionError("read-only folder")))
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(tmp_path / "doc.xlsx")})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed", task
    assert "cannot be written" in task["error"] and "nothing was processed" in task["error"]
    assert loaded == []                                      # not a single file was opened
    assert task["result"]["save_error"] == task["error"]
    assert not (tmp_path / "doc.xlsx").exists()              # the reserved name went back with the failure


def test_a_second_run_never_touches_the_workbook_of_the_first(tmp_path, monkeypatch):
    from openpyxl import load_workbook
    from analysis import batch_processor as bp
    (tmp_path / "a.h5").write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: _xmap_with("bc"))
    out = tmp_path / "doc.xlsx"
    first = bp.batch_process_folder(str(tmp_path), str(out))
    assert first.workbook_path == str(out)
    before = out.read_bytes()
    second = bp.batch_process_folder(str(tmp_path), str(out))
    third = bp.batch_process_folder(str(tmp_path), str(out))
    assert second.workbook_path == str(tmp_path / "doc_2.xlsx") and third.workbook_path == str(tmp_path / "doc_3.xlsx")
    assert out.read_bytes() == before                         # nothing overwritten
    assert load_workbook(second.workbook_path)["Sheet1"]["B1"].value == "a"


def test_the_batch_route_reports_the_real_workbook_path(client_map, monkeypatch, tmp_path):
    client, _ = client_map
    _mixed_folder(tmp_path, monkeypatch)
    out = tmp_path / "doc.xlsx"
    out.write_bytes(b"earlier run")
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(out)})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    assert task["result"]["workbook_path"] == str(tmp_path / "doc_2.xlsx")
    assert str(tmp_path / "doc_2.xlsx") in task["message"] and "already existed" in task["message"]
    assert out.read_bytes() == b"earlier run"


def test_the_single_map_export_follows_the_same_rule(client_map, monkeypatch, tmp_path):
    from openpyxl import load_workbook
    client, _ = client_map
    out = tmp_path / "o.xlsx"
    body = {"output_path": str(out), "dataset_name": "d"}
    first = client.get(f"/api/analysis/task/{client.post('/api/analysis/export/excel', json=body).json()['task_id']}").json()
    assert first["status"] == "completed", first
    assert first["result"]["path"] == str(out) and first["result"]["requested_path"] == str(out)
    before = out.read_bytes()
    second = client.get(f"/api/analysis/task/{client.post('/api/analysis/export/excel', json=body).json()['task_id']}").json()
    assert second["result"]["path"] == str(tmp_path / "o_2.xlsx") and second["result"]["requested_path"] == str(out)
    assert str(tmp_path / "o_2.xlsx") in second["message"]
    assert out.read_bytes() == before and load_workbook(second["result"]["path"])["Sheet1"]["B1"].value == "d"


def test_the_export_without_grains_says_so_and_still_runs_the_band_contrast_steps(client_map, monkeypatch):
    client, ar = client_map
    warnings = []
    results = ar._collect_export_results(warnings)
    assert warnings and warnings[0].startswith("grains: not reconstructed")
    assert "bc_histogram" in results and "bc_gmm" in results        # they need no grains
    assert "grain_size" not in results and "rx" not in results


# --- batch summary, progress and column headers ---------------------------------------------------
def test_the_summary_counts_files_and_cuts_long_reasons():
    import backend.api.routes.analysis as ar
    from analysis.batch_processor import BatchResult
    warnings = [{"file": f"/x/{n}.ang", "reason": r} for n in "ab" for r in ("bc_gmm: x", "rx: x")]
    out = BatchResult(processed=["/x/a.ang", "/x/b.ang"], n_files=2, warnings=warnings,
                      failed=[{"file": "/x/c.ang", "reason": "OSError: " + "y" * 2000}])
    text = ar._batch_summary_text(out)
    assert "2 file(s) with analysis steps left out (a.ang: bc_gmm, rx; b.ang: bc_gmm, rx)" in text
    assert "step(s)" not in text.replace("analysis steps", "")
    assert "y" * 300 not in text and "..." in text and len(text) < 700


def test_the_batch_progress_moves_and_names_the_file(client_map, monkeypatch, tmp_path):
    client, ar = client_map
    from analysis import batch_processor as bp
    seen = []
    real = ar._analysis_tasks
    for name in ("a.h5", "b.h5", "c.h5"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)

    def load(path):
        for key, task in real.items():
            if key not in before and task.get("status") == "running":
                seen.append((task.get("progress"), task.get("current_file")))
        return _xmap_with("bc")

    patch_batch_load(monkeypatch, bp, load)
    before = set(real)
    r = client.post("/api/analysis/batch", json={"folder_path": str(tmp_path), "output_path": str(tmp_path / "doc.xlsx")})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed" and task["progress"] == 1.0
    assert [round(p[0], 2) for p in seen] == [0.0, 0.33, 0.67]
    assert [p[1].rsplit(": ", 1)[-1] for p in seen] == ["a.h5", "b.h5", "c.h5"]


def test_files_with_the_same_stem_get_distinct_column_headers(tmp_path, monkeypatch):
    from openpyxl import load_workbook
    from analysis import batch_processor as bp
    for name in ("scan.ang", "scan.ctf", "other.ang"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: _xmap_with("bc"))
    res = bp.batch_process_folder(str(tmp_path), str(tmp_path / "doc.xlsx"))
    ws = load_workbook(res.workbook_path)["Sheet1"]
    assert [ws.cell(row=1, column=c).value for c in (2, 3, 4)] == ["other", "scan.ang", "scan.ctf"]


def test_a_second_process_all_does_not_carry_the_lists_of_the_first(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    (tmp_path / "bad.h5").write_bytes(b"")
    (tmp_path / "ok.h5").write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: (_ for _ in ()).throw(OSError("x")) if str(path).endswith("bad.h5")
                        else _xmap_with("pq"))
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"))
    proc.process_all()
    proc.process_all()
    res = proc.result()
    assert len(res.failed) == 1 and len(res.warnings) == 2 and res.n_processed == 1
    assert proc.failed_files == [str(tmp_path / "bad.h5")]


def test_a_failed_export_is_marked_on_every_sheet_that_got_the_name(tmp_path, monkeypatch):
    from openpyxl import load_workbook
    from analysis import batch_processor as bp
    from analysis.excel_exporter import ExcelExporter
    (tmp_path / "a.h5").write_bytes(b"")
    (tmp_path / "b.h5").write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: _xmap_with("bc"))
    real = ExcelExporter.export_dataset

    def export(self, name, results):
        if name == "a":
            # fails after the sheets up to the Grain Boundaries one were written
            real(self, name, results)
            raise PermissionError("late failure")
        return real(self, name, results)

    monkeypatch.setattr(ExcelExporter, "export_dataset", export)
    res = bp.batch_process_folder(str(tmp_path), str(tmp_path / "doc.xlsx"))
    wb = load_workbook(res.workbook_path)
    marked = "a (export failed, column incomplete)"
    assert wb["Sheet1"].cell(row=1, column=2).value == marked
    for sheet, row in (("BC hist", 1), ("BandContrast Group", 1), ("Grain Size hist", 1), ("KAM hist", 3),
                       ("GOS hist, area weighted", 1), ("gKAM - area weighted", 1), ("Grain Boundaries", 1)):
        assert wb[sheet].cell(row=row, column=2).value == marked, sheet
    assert wb["BC fit (3G)"].cell(row=2, column=1).value == marked
    assert wb["Sheet1"].cell(row=1, column=3).value == "b" and wb["KAM hist"].cell(row=3, column=3).value == "b"


# ---------------------------------------------------------------------------------------------
# every band contrast reader finds the CTF column
# ---------------------------------------------------------------------------------------------
def _ctf_dataset(tmp_path):
    """EBSDDataset of a real .ctf, read by the page's loader; also the Band Contrast column as written."""
    from analysis.xmap_loader import load_xmap_file
    bcs = _write_ctf(tmp_path / "t.ctf")
    xmap, step, _ = load_xmap_file(str(tmp_path / "t.ctf"))
    return EBSDDataset(xmap, step_size=step), bcs


def test_the_quality_filter_applies_the_band_contrast_threshold_of_a_ctf(tmp_path):
    ds, bcs = _ctf_dataset(tmp_path)
    assert ds.has_native_bc is True
    want = int(np.sum(bcs < 100.0))
    assert 0 < want < bcs.size
    stats = ds.apply_quality_filter(bc_min=100.0, bands_min=0)
    assert stats["filtered_by_bc"] == want and stats["filtered_total"] == want
    assert stats["indexed_after"] == bcs.size - want


def test_every_band_contrast_reader_finds_the_ctf_column(tmp_path):
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    ds, bcs = _ctf_dataset(tmp_path)
    np.testing.assert_allclose(ds.quality, bcs / bcs.max())            # EBSDDataset.quality, not the all-ones fallback
    saved = ar._dataset
    ar._dataset = ds
    seen = {}
    real = ar.colormap_array_to_base64
    ar.colormap_array_to_base64 = lambda data, cmap: seen.update(data=np.array(data)) or real(data, cmap)
    try:
        client = TestClient(app)
        stats = client.get("/api/analysis/quality-stats").json()
        assert stats["bc"]["min"] == pytest.approx(bcs.min()) and stats["bc"]["max"] == pytest.approx(bcs.max())
        r = client.get("/api/analysis/map/bc")
        assert r.status_code == 200, r.text
        np.testing.assert_allclose(seen["data"].ravel(), bcs)            # not the zero fallback
        r = client.post("/api/analysis/quality-filter", json={"bc_min": 100.0, "bands_min": 0})
        assert r.status_code == 200, r.text
        assert r.json()["filtered_by_bc"] == int(np.sum(bcs < 100.0)) > 0
    finally:
        ar.colormap_array_to_base64 = real
        ar._dataset = saved


def test_a_ctf_dataset_gets_its_high_bc_fraction_from_the_ctf_column(tmp_path):
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    ds, bcs = _ctf_dataset(tmp_path)
    saved, saved_settings = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = ds
    try:
        client = TestClient(app)
        r = client.post("/api/analysis/grains/reconstruct", json={"misorientation_threshold": 5.0, "min_grain_size": 1,
                                                                    "alpha": 0.0, "fill": False})
        client.get(f"/api/analysis/task/{r.json()['task_id']}")
        r = client.get("/api/analysis/map/rx")
        assert r.status_code == 200, r.text
        want = float(np.sum(bcs > 0.7 * bcs.max()) / bcs.size * 100)
        assert r.json()["high_bc_fraction"] == pytest.approx(want) and want > 0
    finally:
        ar._dataset = saved
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved_settings)


@pytest.mark.parametrize("order", [("bc", "BC"), ("BC", "bc")])
def test_the_band_contrast_key_does_not_depend_on_the_property_order(order):
    xmap = to_xmap(make_map(shape=(10, 12), n_seeds=4))
    values = {"bc": np.full(xmap.size, 1.0), "BC": np.full(xmap.size, 2.0)}
    for key in order:
        xmap.prop[key] = values[key]
    ds = EBSDDataset(xmap, step_size=1.0)
    assert ds._band_contrast_key() == "bc" and ds.band_contrast_raw().mean() == 1.0 and ds.bc.mean() == 1.0


@pytest.mark.parametrize("order", [("Bc", "BC"), ("BC", "Bc")])
def test_without_an_exact_bc_the_other_spellings_are_chosen_in_sorted_order(order):
    xmap = to_xmap(make_map(shape=(10, 12), n_seeds=4))
    for key in order:
        xmap.prop[key] = np.full(xmap.size, 3.0)
    assert EBSDDataset(xmap, step_size=1.0)._band_contrast_key() == "BC"


def test_a_property_that_only_contains_bc_is_not_band_contrast():
    xmap = to_xmap(make_map(shape=(10, 12), n_seeds=4))
    for key in ("bc_x", "BCS", "abc"):
        xmap.prop[key] = np.arange(xmap.size, dtype=float)
    assert EBSDDataset(xmap, step_size=1.0).has_native_bc is False


def test_current_file_is_cleared_when_the_batch_ends(client_map, monkeypatch, tmp_path):
    client, ar = client_map
    import analysis.batch_processor as bp

    def finishing(folder, out, progress_callback=None, **kw):
        progress_callback("Processing file 1/1: a.h5", 1, 1)
        return _batch_result(processed=["a.h5"], n_files=1, workbook_path=str(tmp_path / "o.xlsx"))

    def failing(folder, out, progress_callback=None, **kw):
        progress_callback("Processing file 1/1: a.h5", 1, 1)
        raise OSError("gone")

    outcomes = []
    for fn in (finishing, failing):
        monkeypatch.setattr(bp, "batch_process_folder", fn)
        r = client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": str(tmp_path / "o.xlsx")})
        task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
        outcomes.append((task["status"], task["current_file"], task.get("progress")))
    assert outcomes == [("completed", None, 1.0), ("failed", None, 1.0)]


# ---------------------------------------------------------------------------------------------
# a map without band contrast, and the name of the loaded file
# ---------------------------------------------------------------------------------------------
NO_BC_DETAIL = "This file has no band contrast; choose another layer."


def test_the_band_contrast_map_of_a_file_without_band_contrast_is_refused_not_drawn_black(client_two_phase):
    client, ar = client_two_phase
    del ar._dataset.xmap.prop["bc"]
    assert ar._dataset.band_contrast_raw() is None
    r = client.get("/api/analysis/map/bc")
    assert r.status_code == 400 and r.json()["detail"] == NO_BC_DETAIL
    # the other layers of the same map still draw
    assert client.get("/api/analysis/map/ipf").status_code == 200


def test_the_band_contrast_map_of_a_file_with_band_contrast_is_unchanged(client_two_phase):
    client, ar = client_two_phase
    seen = {}
    real = ar.colormap_array_to_base64
    ar.colormap_array_to_base64 = lambda data, cmap: seen.update(data=np.array(data)) or real(data, cmap)
    try:
        r = client.get("/api/analysis/map/bc")
    finally:
        ar.colormap_array_to_base64 = real
    assert r.status_code == 200, r.text
    np.testing.assert_allclose(seen["data"].ravel(), ar._dataset.xmap.prop["bc"])


def test_status_says_whether_the_map_has_band_contrast(client_two_phase):
    client, ar = client_two_phase
    assert client.get("/api/analysis/status").json()["has_native_bc"] is True
    del ar._dataset.xmap.prop["bc"]
    assert client.get("/api/analysis/status").json()["has_native_bc"] is False


def test_status_and_load_name_the_file_without_its_folder(tmp_path):
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    folder = tmp_path / "private folder name"
    folder.mkdir()
    _write_ctf(folder / "Sample 7050 AB1.ctf")
    saved, saved_settings = ar._dataset, dict(ar._grain_stats_settings)
    try:
        client = TestClient(app)
        r = client.post("/api/analysis/load", json={"xmap_path": str(folder / "Sample 7050 AB1.ctf")})
        assert r.status_code == 200, r.text
        assert r.json()["source_name"] == "Sample 7050 AB1.ctf" and r.json()["has_native_bc"] is True
        assert r.json()["step_size"] == pytest.approx(1.0)                   # XStep of the file
        s = client.get("/api/analysis/status").json()
        assert s["source_name"] == "Sample 7050 AB1.ctf"
        # a file name only: never the folder, in either answer
        assert "private folder name" not in json.dumps(r.json()) and "private folder name" not in json.dumps(s)
    finally:
        ar._dataset = saved
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved_settings)


def test_a_dataset_without_a_known_file_has_no_name(client_two_phase):
    client, ar = client_two_phase
    ar._dataset.source = "indexing_result"
    assert client.get("/api/analysis/status").json()["source_name"] is None


def test_a_file_name_is_cut_from_either_kind_of_path():
    from backend.api.routes.analysis import _file_name_only
    assert _file_name_only(r"C:\Users\me\data\scan 1.h5oina") == "scan 1.h5oina"
    assert _file_name_only("/home/me/data/scan.ctf") == "scan.ctf"
    assert _file_name_only("scan.ang") == "scan.ang"
    assert _file_name_only("") is None and _file_name_only(None) is None and _file_name_only("C:\\data\\") == "data"


def test_the_ecd_map_follows_the_basis_of_the_last_grain_size_call(client_two_phase):
    """The map is drawn on the remembered basis: a /grain-size call with the other basis changes the next map.
    Grains whose measured pixels differ from their cells (closed holes) are made by hand: the map is scaled to
    its own range, so a change of all grains by one factor would not show."""
    import base64
    client, ar = client_two_phase
    _reconstruct(client)
    grains = ar._dataset.grains
    size = np.asarray(grains.grain_size, dtype=float).copy()
    size[0] = max(1.0, size[0] * 0.2)                       # one grain with most of its cells closed
    grains.grain_size = size

    def ecd_map():
        r = client.get("/api/analysis/map/ecd")
        assert r.status_code == 200, r.text
        return base64.b64decode(r.json()["image"])

    cell = ecd_map()
    assert client.post("/api/analysis/grain-size", json={"ecd_basis": "measured_pixels"}).status_code == 200
    pixels = ecd_map()
    assert pixels != cell
    assert client.post("/api/analysis/grain-size", json={"ecd_basis": "cell_area"}).status_code == 200
    assert ecd_map() == cell


def test_the_min_intercept_of_the_batch_may_be_zero(client_map, monkeypatch):
    client, _ = client_map
    import analysis.batch_processor as bp
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder",
                        lambda folder, out, progress_callback=None, **kw: seen.append(kw) or _batch_result(processed=["a.h5"], n_files=1))
    r = client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o", "min_intercept_um": 0})
    assert r.status_code == 200, r.text
    client.get(f"/api/analysis/task/{r.json()['task_id']}")
    assert seen == [{"min_intercept_um": 0.0}]
    assert client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o", "min_intercept_um": -1}).status_code == 422


def test_the_files_where_the_edge_rule_kept_all_grains_reach_the_task_answer(client_map, monkeypatch):
    client, _ = client_map
    import analysis.batch_processor as bp
    monkeypatch.setattr(
        bp, "batch_process_folder",
        lambda folder, out, progress_callback=None, **kw: _batch_result(
            processed=["a.h5", "b.h5"], n_files=2, edge_safeguard_files=["a.h5"]))
    r = client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    assert task["result"]["edge_safeguard_files"] == ["a.h5"]


def test_a_map_loaded_from_an_indexing_result_is_named_after_its_source_file(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    import backend.api.routes.indexing as ix
    xmap = ar._dataset.xmap
    for meta, want in (({"source_file": r"C:\private folder\data\scan 7.h5oina"}, "scan 7.h5oina"),
                       ({"source_file": "/home/me/private/scan 8.ctf"}, "scan 8.ctf"),
                       ({}, None)):
        monkeypatch.setattr(ix, "get_last_indexing_result", lambda m=meta: SimpleNamespace(xmap=xmap, metadata=m))
        r = client.post("/api/analysis/load", json={"xmap_path": "__from_indexing__"})
        assert r.status_code == 200, r.text
        assert r.json()["source_name"] == want
        assert client.get("/api/analysis/status").json()["source_name"] == want
        assert "private" not in json.dumps(r.json())


def test_status_carries_the_step(client_two_phase):
    client, ar = client_two_phase
    assert client.get("/api/analysis/status").json()["step_size"] == pytest.approx(float(ar._dataset.step_size))
