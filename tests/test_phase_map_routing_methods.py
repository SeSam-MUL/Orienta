"""Per-phase routing from the EDS phase map works for every method.

The routing branch of ``POST /api/indexing/start`` used to match the phase-map
entries (CIF filenames) against ``req.cif_paths`` only, so only Hough could take
part: a Spherical request carries SHT files and a Dictionary request master /
dictionary ``.h5`` files, and the run ended on "none of the phase-map CIFs are
present". These tests drive the real route and the real
``run_per_phase_indexing`` with the one thing that costs time stubbed -- the
indexing of a single phase -- and read what each phase was handed.

The expectations are written by hand from what the user asked for ("each phase
only on its own pixels, with its own file and the backend I chose"), not
computed with the code under test.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import indexing_controller  # noqa: E402
from indexing_controller import (  # noqa: E402
    ComparisonConfig,
    IndexingMethod,
    IndexingResult,
    PhaseConfig,
    PhaseMethodResult,
    PixelSelectionMode,
    run_per_phase_indexing,
)

NAV = (2, 3)  # rows, cols of the stub scan


# --- test doubles --------------------------------------------------------------

def _stub_pmr(name: str, mask: np.ndarray, method: IndexingMethod,
              metadata: dict | None = None, ci: float | None = None,
              sim_start: int | None = None) -> PhaseMethodResult:
    """A PhaseMethodResult with a real one-phase CrystalMap over ``mask``'s pixels,
    so the real ``build_consensus_xmap`` can merge it.

    ``ci``: every pixel of the phase gets this confidence index. ``sim_start``:
    the xmap carries ``simulation_indices`` ``sim_start, sim_start + 1, ...``
    (a Dictionary result)."""
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    n_pix = int(mask.sum())
    rows, cols = np.nonzero(mask)
    xmap = CrystalMap(
        rotations=Rotation.identity(n_pix),
        phase_id=np.zeros(n_pix, dtype=int),
        x=cols.astype(float), y=rows.astype(float),
        phase_list=PhaseList(Phase(name=name, point_group="m-3m")),
        prop=({"simulation_indices": (sim_start + np.arange(n_pix)).reshape(-1, 1)}
              if sim_start is not None else {}))
    result = IndexingResult(xmap=xmap, selection_mask=mask,
                            original_shape=mask.shape, method=method,
                            metadata=metadata,
                            confidence_scores=(None if ci is None
                                               else np.full(n_pix, ci, dtype=np.float32)))
    return PhaseMethodResult(phase_name=name, method=method,
                             indexing_result=result, mean_score=0.9)


class Recorder:
    """Stands in for ``run_single_phase_method`` and notes what each phase got."""

    def __init__(self, method: IndexingMethod, metadata_for=None, on_call=None,
                 fail_after_record=False):
        self.method = method
        self.bar: list[float] = []   # the task's progress bar right after each indexer message
        self.fail_after_record = fail_after_record
        self.on_call = on_call      # on_call(index of this call), before it records
        self.calls: list[dict] = []
        self.results: list[PhaseMethodResult] = []
        self._metadata_for = metadata_for

    def __call__(self, signal, detector, phase_config, method, config,
                 selection_mask, h5_path="", detector_params=None,
                 progress_callback=None, cancel_check=None):
        if self.on_call is not None:
            self.on_call(len(self.calls))
        self.calls.append({
            "detector": detector,
            "compute_mode": config.compute_mode,
            "progress_callback": progress_callback,
            "cancel_check": cancel_check,
            "name": phase_config.name,
            "sht_path": phase_config.sht_path,
            "master_h5_path": phase_config.master_h5_path,
            "cif_path": phase_config.cif_path,
            "dictionary": phase_config.dictionary,
            # Every phase of the run, as the run sees it, at THIS moment: how
            # many dictionaries are resident while this phase is indexed.
            "held_dictionaries": [p.name for p in config.phases
                                  if p.dictionary is not None],
            "mask": np.array(selection_mask, copy=True),
            "backend": config.backend,
            "circmask": config.circmask,
            "gausbckg": config.gausbckg,
            "method": method,
            "h5_path": h5_path,
            "detector_params": detector_params,
        })
        md = self._metadata_for(phase_config) if self._metadata_for else None
        k = len(self.calls) - 1            # this phase's number: distinct values per phase
        pmr = _stub_pmr(phase_config.name, selection_mask, method, md,
                        ci=0.25 + 0.5 * k, sim_start=100 * (k + 1))
        self.results.append(pmr)
        if progress_callback is not None:
            # what an indexer does: a message with ITS phase's fraction
            progress_callback(f"indexer of {phase_config.name} says hello", 0.99)
            from backend.api.routes import indexing as route
            self.bar.extend(t["progress"] for t in route._indexing_tasks.values())
        if self.fail_after_record:
            raise RuntimeError("stop after recording")
        return pmr


def _grid(pixels_per_phase: list[int]) -> np.ndarray:
    """A 2x3 phase_grid: the first phase's pixels, then the next phase's ..."""
    flat = np.full(NAV[0] * NAV[1], -1, dtype=int)
    pos = 0
    for idx, n in enumerate(pixels_per_phase):
        flat[pos:pos + n] = idx
        pos += n
    return flat.reshape(NAV)


def _start_routed(monkeypatch, tmp_path, *, method, entries, grid, req_kwargs,
                  links, single_phase=None, routing=True):
    """``POST /api/indexing/start`` with phase-map routing; returns the task.

    ``entries``: CIF filenames of the phase map, in its order. ``links``:
    ``{basename of a non-CIF request file: basename of the CIF it is linked to}``
    (the stub of ``phase_metadata.find_linked_cif``). The task state comes back
    with ``stored`` (what ``_store_result`` received) and ``progress_log``.
    """
    import torch
    from kikuchipy.signals import EBSD

    import phase_metadata
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import indexing as route
    from backend.api.services import phase_map_store

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    signal = EBSD(np.zeros((*NAV, 8, 8), dtype=np.float32))
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: signal)
    state = SimpleNamespace(
        phase_entries=[SimpleNamespace(cif_filename=n) for n in entries],
        phase_grid=grid)
    monkeypatch.setattr(phase_map_store, "get_phase_map_store",
                        lambda: SimpleNamespace(get_state=lambda: state))

    def find_linked_cif(file_path, cif_library_dir):
        cif = links.get(Path(file_path).name)
        return Path(cif_library_dir) / cif if cif else None

    monkeypatch.setattr(phase_metadata, "strict_linked_cif", find_linked_cif)
    monkeypatch.setattr(route, "build_spherical_det_params",
                        lambda *a, **k: {"det": "params"})
    if single_phase is not None:
        monkeypatch.setattr(indexing_controller, "run_single_phase_method",
                            single_phase)
    stored = []
    monkeypatch.setattr(route, "_store_result",
                        lambda result, *a, **k: stored.append(result))

    req = route.IndexingStartRequest(method=method, use_phase_map_routing=routing,
                                     **req_kwargs)
    task_id = asyncio.run(route.start_indexing(req))["task_id"]
    task = route._indexing_tasks.pop(task_id)
    task["stored"] = stored
    return task


def _sht_request(tmp_path, names, **extra):
    paths = [str(tmp_path / f"{n}.sht") for n in names]
    return dict(sht_paths=paths, backend="spherical_gpu", circmask=77,
                gausbckg=False, **extra)


# --- the rule: which file carries a phase-map phase -----------------------------

def _library(tmp_path) -> Path:
    lib = tmp_path / "CIF_Library"
    lib.mkdir()
    for name in ("Al", "Si"):
        (lib / f"{name}.cif").write_text("data_x\n", encoding="utf-8")
    return lib


def _join(route, req, lib):
    """The join the routed run makes: links, then one file per CIF."""
    return route.first_file_per_cif(route.phase_map_file_links(req, lib))


def test_rule_links_sht_and_master_files_to_phase_map_cifs_by_the_library(tmp_path):
    from backend.api.routes import indexing as route

    lib = _library(tmp_path)
    req = SimpleNamespace(method="spherical", cif_paths=[], master_h5_paths=[],
                          sht_paths=[str(tmp_path / "Si.sht"),
                                     str(tmp_path / "Al.sht"),
                                     str(tmp_path / "Unknown.sht")])
    assert _join(route, req, lib) == {
        "Si.cif": str(tmp_path / "Si.sht"),
        "Al.cif": str(tmp_path / "Al.sht"),
    }

    req = SimpleNamespace(method="dictionary", cif_paths=[], sht_paths=[],
                          master_h5_paths=[str(tmp_path / "Al_master_E20kV_npx500.h5")])
    assert _join(route, req, lib) == {
        "Al.cif": str(tmp_path / "Al_master_E20kV_npx500.h5")}


def test_rule_for_hough_is_the_cif_basename_and_ignores_the_other_lists(tmp_path):
    from backend.api.routes import indexing as route

    req = SimpleNamespace(method="hough", cif_paths=[str(tmp_path / "Al.cif")],
                          sht_paths=[str(tmp_path / "Si.sht")], master_h5_paths=[])
    assert _join(route, req, tmp_path) == {
        "Al.cif": str(tmp_path / "Al.cif")}


def test_rule_takes_the_first_file_when_two_are_linked_to_one_cif(tmp_path):
    from backend.api.routes import indexing as route

    lib = _library(tmp_path)
    first, second = str(tmp_path / "Al.sht"), str(tmp_path / "Al (Al) [cF4] {20kV}.sht")
    # both are linked to Al.cif: the first by its stem, the second by the "(Al)"
    # in its name; whichever the request lists first is the one the run uses
    for order in ([first, second], [second, first]):
        req = SimpleNamespace(method="spherical", cif_paths=[], master_h5_paths=[],
                              sht_paths=order)
        assert _join(route, req, lib) == {"Al.cif": order[0]}


def test_the_discovery_records_the_cif_each_file_is_linked_to(tmp_path):
    from indexing_controller import enrich_phase_entries

    cif_dir = tmp_path / "CIF_Library"
    cif_dir.mkdir()
    (cif_dir / "Al.cif").write_text(
        "data_Al\n_cell_length_a 4.05\n_cell_length_b 4.05\n_cell_length_c 4.05\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_Int_Tables_number 225\n_symmetry_space_group_name_H-M 'F m -3 m'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
        "Al Al 0 0 0 1\n", encoding="utf-8")
    results = [
        {"path": cif_dir / "Al.cif", "filename": "Al.cif"},
        {"path": tmp_path / "Al.sht", "filename": "Al.sht"},
        {"path": tmp_path / "Nothing.sht", "filename": "Nothing.sht"},
    ]
    enrich_phase_entries(results, tmp_path)
    assert [r["linked_cif"] for r in results] == ["Al.cif", "Al.cif", None]


# --- Spherical ---------------------------------------------------------------

def test_spherical_routing_runs_each_phase_with_its_own_sht_mask_and_backend(
        monkeypatch, tmp_path):
    rec = Recorder(IndexingMethod.SPHERICAL)
    # the request lists the phases in the OPPOSITE order of the phase map
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([2, 4]),
        req_kwargs=_sht_request(tmp_path, ["PhaseB", "PhaseA"]),
        links={"PhaseA.sht": "PhaseA.cif", "PhaseB.sht": "PhaseB.cif"},
        single_phase=rec)

    assert task["status"] == "completed", task["error"]
    by_name = {c["name"]: c for c in rec.calls}
    assert sorted(by_name) == ["PhaseA.cif", "PhaseB.cif"]          # each phase once
    assert len(rec.calls) == 2
    a, b = by_name["PhaseA.cif"], by_name["PhaseB.cif"]
    assert a["sht_path"] == str(tmp_path / "PhaseA.sht")
    assert b["sht_path"] == str(tmp_path / "PhaseB.sht")
    assert a["mask"].ravel().tolist() == [True, True, False, False, False, False]
    assert b["mask"].ravel().tolist() == [False, False, True, True, True, True]
    for c in (a, b):
        assert c["backend"] == "spherical_gpu"                       # the chosen backend
        assert c["circmask"] == 77 and c["gausbckg"] is False        # and the other knobs
        assert c["method"] == IndexingMethod.SPHERICAL
        assert c["detector_params"] == {"det": "params"}


def test_spherical_routing_names_and_masters_follow_the_phase_map_order(
        monkeypatch, tmp_path):
    """Ids count the phase map's order, not the request's file order."""
    rec = Recorder(IndexingMethod.SPHERICAL)
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([3, 3]),
        req_kwargs=_sht_request(tmp_path, ["PhaseB", "PhaseA"]),   # reversed
        links={"PhaseA.sht": "PhaseA.cif", "PhaseB.sht": "PhaseB.cif"},
        single_phase=rec)
    assert task["status"] == "completed", task["error"]
    (result,) = task["stored"]
    names = {int(pid): name for pid, name in
             zip(result.xmap.phases.ids, result.xmap.phases.names) if pid >= 0}
    assert names == {0: "PhaseA", 1: "PhaseB"}
    sht_by_id = result.metadata["sht_paths_by_phase"]
    assert {k: Path(v).name for k, v in sht_by_id.items()} == {
        0: "PhaseA.sht", 1: "PhaseB.sht"}


def test_a_phase_without_a_file_keeps_its_pixels_unindexed_and_is_named(
        monkeypatch, tmp_path):
    rec = Recorder(IndexingMethod.SPHERICAL)
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif", "PhaseB.cif", "PhaseC.cif"], grid=_grid([2, 2, 2]),
        req_kwargs=_sht_request(tmp_path, ["PhaseA", "PhaseB"]),
        links={"PhaseA.sht": "PhaseA.cif", "PhaseB.sht": "PhaseB.cif"},
        single_phase=rec)

    assert task["status"] == "completed", task["error"]
    assert sorted(c["name"] for c in rec.calls) == ["PhaseA.cif", "PhaseB.cif"]
    (result,) = task["stored"]
    # PhaseC's two pixels (the last two) are unclassified, not given to a neighbour
    assert result.xmap.phase_id.tolist() == [0, 0, 1, 1, -1, -1]
    (missing,) = result.metadata["phase_map_unrouted"]
    assert missing["phase"] == "PhaseC.cif" and missing["pixels"] == 2
    assert "SHT" in missing["reason"]
    log = "\n".join(task["pending_log"])
    assert "PhaseC.cif" in log and "UNCLASSIFIED" in log


def test_no_phase_with_a_file_for_the_method_fails_naming_method_and_file_kind(
        monkeypatch, tmp_path):
    task = _start_routed(
        monkeypatch, tmp_path, method="spherical",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([3, 3]),
        req_kwargs=_sht_request(tmp_path, ["Other"]),     # linked to no phase-map CIF
        links={"Other.sht": "Elsewhere.cif"},
        single_phase=Recorder(IndexingMethod.SPHERICAL))
    assert task["status"] == "failed"
    err = task["error"]
    assert "spherical" in err and "SHT" in err
    assert "PhaseA.cif" in err and "PhaseB.cif" in err
    assert "none of the phase-map CIFs" not in err


# --- Dictionary --------------------------------------------------------------

def _fake_kp_load(monkeypatch):
    """``kikuchipy.load`` -> an object standing for one phase's dictionary."""
    import kikuchipy as kp

    loaded: list[str] = []

    def load(path, *a, **k):
        loaded.append(Path(path).name)
        return SimpleNamespace(xmap=SimpleNamespace(phases=f"phases of {Path(path).name}"),
                               path=path)

    monkeypatch.setattr(kp, "load", load)
    return loaded


def test_dictionary_routing_holds_one_dictionary_at_a_time_each_with_its_own_mask(
        monkeypatch, tmp_path):
    from backend.api.routes import indexing as route

    loaded = _fake_kp_load(monkeypatch)
    dict_file = tmp_path / "result_dict.h5"
    dict_file.write_bytes(b"x")                       # a dictionary on disk
    rec = Recorder(
        IndexingMethod.DICTIONARY,
        metadata_for=lambda pc: {"dictionary": object(), "dict_path": str(dict_file)})
    released = []
    monkeypatch.setattr(route, "_release_phase_gpu_memory",
                        lambda exc=None: released.append(exc))

    task = _start_routed(
        monkeypatch, tmp_path, method="dictionary",
        entries=["PhaseA.cif", "PhaseB.cif", "PhaseC.cif"], grid=_grid([1, 2, 3]),
        req_kwargs=dict(master_h5_paths=[str(tmp_path / f"{n}_master.h5")
                                         for n in ("PhaseC", "PhaseA", "PhaseB")],
                        backend="emsphinx"),
        links={"PhaseA_master.h5": "PhaseA.cif", "PhaseB_master.h5": "PhaseB.cif",
               "PhaseC_master.h5": "PhaseC.cif"},
        single_phase=rec)

    assert task["status"] == "completed", task["error"]
    assert loaded == ["PhaseA_master.h5", "PhaseB_master.h5", "PhaseC_master.h5"]
    for call, own in zip(rec.calls, ("PhaseA.cif", "PhaseB.cif", "PhaseC.cif")):
        assert call["name"] == own
        assert call["dictionary"] is not None                       # loaded for this pass
        assert call["held_dictionaries"] == [own]                   # and only this one
    assert [c["mask"].sum() for c in rec.calls] == [1, 2, 3]
    assert rec.calls[0]["mask"].ravel().tolist() == [True, False, False, False, False, False]
    assert rec.calls[2]["mask"].ravel().tolist() == [False, False, False, True, True, True]
    # released afterwards, and detached from the result that is kept
    assert all(r.indexing_result.metadata["dictionary"] is None for r in rec.results)
    assert len(released) == 3
    assert rec.calls[0]["master_h5_path"].endswith("PhaseA_master.h5")


def test_dictionary_routing_releases_a_dictionary_when_its_phase_fails(
        monkeypatch, tmp_path):
    from backend.api.routes import indexing as route

    _fake_kp_load(monkeypatch)
    monkeypatch.setattr(route, "_release_phase_gpu_memory", lambda exc=None: None)
    seen_held = []

    def single(signal, detector, phase_config, method, config, selection_mask,
               h5_path="", detector_params=None, progress_callback=None,
               cancel_check=None):
        seen_held.append([p.name for p in config.phases if p.dictionary is not None])
        if phase_config.name == "PhaseA.cif":
            raise RuntimeError("out of memory")
        return _stub_pmr(phase_config.name, selection_mask, method,
                         {"dictionary": None})

    task = _start_routed(
        monkeypatch, tmp_path, method="dictionary",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([3, 3]),
        req_kwargs=dict(master_h5_paths=[str(tmp_path / "PhaseA_master.h5"),
                                         str(tmp_path / "PhaseB_master.h5")]),
        links={"PhaseA_master.h5": "PhaseA.cif", "PhaseB_master.h5": "PhaseB.cif"},
        single_phase=single)
    assert task["status"] == "completed", task["error"]
    assert seen_held == [["PhaseA.cif"], ["PhaseB.cif"]]     # A's was let go after it failed
    (result,) = task["stored"]
    assert "PhaseA.cif" in "\n".join(task["pending_log"])      # the lost phase is named
    assert result.metadata["n_phases_skipped"] == 1


# --- Hough: unchanged ---------------------------------------------------------

def test_hough_routing_still_builds_one_cif_phase_per_entry_and_the_plain_call(
        monkeypatch, tmp_path):
    from orix.crystal_map import Phase

    import ebsd_utils

    monkeypatch.setattr(ebsd_utils, "hough_phase_from_cif",
                        lambda path: Phase(name=Path(path).stem))
    seen = {}

    def capture(**kw):
        seen.update(kw)
        raise RuntimeError("stop after routing")

    monkeypatch.setattr(indexing_controller, "run_per_phase_indexing", capture)
    task = _start_routed(
        monkeypatch, tmp_path, method="hough",
        entries=["PhaseA.cif", "PhaseB.cif"], grid=_grid([2, 4]),
        req_kwargs=dict(cif_paths=[str(tmp_path / "PhaseB.cif"),
                                   str(tmp_path / "PhaseA.cif")], backend="emsphinx"),
        links={})
    assert "stop after routing" in task["error"]
    cfgs = seen["phase_configs"]
    assert [(c.name, c.cif_path) for c in cfgs] == [
        ("PhaseA.cif", str(tmp_path / "PhaseA.cif")),
        ("PhaseB.cif", str(tmp_path / "PhaseB.cif"))]
    assert all(c.sht_path == "" and c.master_h5_path == "" for c in cfgs)
    assert [[p.name for _, p in c.phase_list] for c in cfgs] == [["PhaseA"], ["PhaseB"]]
    assert seen["run_phase"] is None                          # the plain per-phase call
    assert seen["method"] == IndexingMethod.HOUGH
    assert [m.sum() for m in seen["masks_per_phase"]] == [2, 4]


# --- the controller: an optional per-phase runner ------------------------------

def _controller_args(n_phases=2):
    signal = MagicMock()
    signal.data.shape = (*NAV, 8, 8)
    pcs = [PhaseConfig(name=f"P{i}", sht_path=f"P{i}.sht") for i in range(n_phases)]
    grid = _grid([3, 3][:n_phases])
    masks = [grid == i for i in range(n_phases)]
    config = ComparisonConfig(phases=pcs, methods=[IndexingMethod.SPHERICAL],
                              selection_mode=PixelSelectionMode.MASK)
    return dict(signal=signal, detector=MagicMock(), method=IndexingMethod.SPHERICAL,
                config=config, phase_configs=pcs, masks_per_phase=masks,
                consensus_map=grid.astype(np.int32)), pcs, masks


def test_a_per_phase_runner_replaces_the_plain_call(monkeypatch):
    kwargs, pcs, masks = _controller_args()
    monkeypatch.setattr(indexing_controller, "run_single_phase_method",
                        lambda **k: pytest.fail("the plain call must not run"))
    got = []

    def runner(pc, mask):
        got.append((pc.name, int(mask.sum())))
        return _stub_pmr(pc.name, mask, IndexingMethod.SPHERICAL)

    result = run_per_phase_indexing(**kwargs, run_phase=runner)
    assert got == [("P0", 3), ("P1", 3)]
    assert result.metadata["n_phases_run"] == 2


def test_a_failing_per_phase_runner_skips_only_that_phase(monkeypatch):
    kwargs, pcs, masks = _controller_args()

    def runner(pc, mask):
        if pc.name == "P0":
            raise RuntimeError("boom")
        return _stub_pmr(pc.name, mask, IndexingMethod.SPHERICAL)

    result = run_per_phase_indexing(**kwargs, run_phase=runner)
    assert result.metadata["n_phases_skipped"] == 1
    assert result.metadata["skipped"] == [{"phase_index": 0, "reason": "boom"}]


def test_without_a_runner_the_plain_call_is_made(monkeypatch):
    kwargs, pcs, masks = _controller_args()
    calls = []

    def plain(**k):
        calls.append(k["phase_config"].name)
        return _stub_pmr(k["phase_config"].name, k["selection_mask"],
                         IndexingMethod.SPHERICAL)

    monkeypatch.setattr(indexing_controller, "run_single_phase_method", plain)
    run_per_phase_indexing(**kwargs)
    assert calls == ["P0", "P1"]
