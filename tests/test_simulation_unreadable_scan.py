"""'Simulate All Missing' no longer says "all done" for a phase whose only master
the library refuses.

`scan_missing_materials` counted a present master it could
not judge as usable, so a phase whose only master was written by EMsoft's GPU
program (which kikuchipy cannot load, and which the phase library therefore does
not offer) was not listed -- the job warning said "simulate again", the scan said
nothing was missing. Such a phase is now reported separately from "missing":
the file exists, and simulating again with the same program would write the same
file, so it is not queued -- but it is not "done" either.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import simulation.simulation_controller as sc  # noqa: E402
from backend.api.routes import simulation as sim_route  # noqa: E402


def _write_master(path, *, gpu):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("EMData/EBSDmaster/mLPNH", data=[[1.0]])
        f.create_dataset(
            "EMheader/EBSDmaster/ProgramName",
            data=[("EMEBSDmasterOpenCL.f90" if gpu else "EMEBSDmaster.f90").encode()],
            dtype=h5py.special_dtype(vlen=bytes))
        grp = "EBSDMasterOpenCLNameList" if gpu else "EBSDMasterNameList"
        f.create_dataset(f"NMLparameters/{grp}/npx", data=[500])


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project root: the scan locates it from the controller module."""
    (tmp_path / "simulation").mkdir()
    monkeypatch.setattr(sc, "__file__", str(tmp_path / "simulation" / "simulation_controller.py"))
    xtal = tmp_path / "Database" / "XTAL_Library"
    xtal.mkdir(parents=True)
    (xtal / "Mg.xtal").write_bytes(b"x")
    # the Monte-Carlo file, so only the master decides
    mc = tmp_path / "Database" / "EBSD_H5_Cache" / "Mg" / "Mg_E20kV_sig70_n501_o0.h5"
    mc.parent.mkdir(parents=True)
    mc.write_bytes(b"x")
    return tmp_path


def _scan(output_type="master_only"):
    ctrl = sc.SimulationController.__new__(sc.SimulationController)
    unreadable: list = []
    missing = ctrl.scan_missing_materials(output_type, sc.SimulationParameters(ekev=20.0),
                                          unreadable_out=unreadable)
    return missing, unreadable


def test_a_phase_whose_only_master_is_refused_is_reported_not_silent(project):
    _write_master(project / "Database" / "EBSD_H5_Cache" / "Mg" / "Mg_master_E20kV_npx500.h5", gpu=True)
    missing, unreadable = _scan()
    assert missing == []                       # it is not "missing" ...
    assert unreadable == [{                    # ... and it is not silent
        "xtal_path": str(project / "Database" / "XTAL_Library" / "Mg.xtal"),
        "stem": "Mg", "code": "masterUnreadableByKikuchipy",
        "file": "Mg_master_E20kV_npx500.h5", "program": "EMEBSDmasterOpenCL.f90"}]


def test_a_readable_master_is_done(project):
    _write_master(project / "Database" / "EBSD_H5_Cache" / "Mg" / "Mg_master_E20kV_npx500.h5", gpu=False)
    assert _scan() == ([], [])


def test_no_master_at_all_is_still_missing(project):
    missing, unreadable = _scan()
    assert [m["missing"] for m in missing] == [["master"]] and unreadable == []


def test_a_readable_master_beside_a_refused_one_counts_as_done(project):
    cache = project / "Database" / "EBSD_H5_Cache" / "Mg"
    _write_master(cache / "Mg_master_E20kV_npx500.h5", gpu=False)
    _write_master(cache / "Mg_master_E20kV_npx300.h5", gpu=True)
    assert _scan() == ([], [])


def test_the_scan_without_an_unreadable_list_keeps_its_old_signature_and_answer(project):
    _write_master(project / "Database" / "EBSD_H5_Cache" / "Mg" / "Mg_master_E20kV_npx500.h5", gpu=True)
    ctrl = sc.SimulationController.__new__(sc.SimulationController)
    assert ctrl.scan_missing_materials("master_only", sc.SimulationParameters(ekev=20.0)) == []


def test_the_route_carries_the_unreadable_list_next_to_missing(monkeypatch):
    entry = {"xtal_path": "X/Mg.xtal", "stem": "Mg", "code": "masterUnreadableByKikuchipy",
             "file": "Mg_master.h5", "program": "EMEBSDmasterOpenCL.f90"}
    fake = MagicMock()

    def _scan_stub(output_type, params, unreadable_out=None):
        unreadable_out.append(entry)
        return []

    fake.scan_missing_materials.side_effect = _scan_stub
    monkeypatch.setattr(sim_route, "_get_controller", lambda: fake)
    app = FastAPI()
    app.include_router(sim_route.router, prefix="/api/simulation")
    data = TestClient(app).get("/api/simulation/scan-missing").json()
    assert data["missing"] == [] and data["count"] == 0
    assert data["unreadable"] == [entry]
