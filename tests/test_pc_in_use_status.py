"""The PC in use in /api/pc/status, the mode of a search record, the convention of the open file, and the
origin of the PC in a result's record."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from kikuchipy.detectors import EBSDDetector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402
from backend.api.services import pc_in_use  # noqa: E402
from backend.api.services.calibration_store import CalibrationStore  # noqa: E402
from tests.test_pc_calibrations import _loaded_as  # noqa: E402

SHAPE = (60, 60)
FOUND = (0.4301, 0.2141, 0.5009)


@pytest.fixture(autouse=True)
def clean(tmp_path):
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    use.forget_all()
    yield
    use.forget_all()
    cal.set_calibrations_dir_for_test(None)


# ------------------------------------------------------------------ the convention of the file, from its format

def _with(filename, manufacturer=None):
    sig = _loaded_as(filename)
    if manufacturer:
        sig.original_metadata.set_item("manufacturer", manufacturer)
    return sig


@pytest.mark.parametrize("sig,expected", [
    (_with("C:/data/map.ebsp"), ("oxford", True)),                       # Oxford binary patterns
    (_with("scan 1", "oxford instruments"), ("oxford", True)),           # .h5oina as kikuchipy records it: no extension
    (_with("scan.up1"), ("tsl", True)),
    (_with("SCAN.UP2"), ("tsl", True)),
    (_with("HiGainNi", "edax"), ("tsl", True)),                          # EDAX h5ebsd: the maker is the format key
    (_with("hex", "EDAX"), ("tsl", True)),
    (_with("scan", "bruker nano"), ("bruker", True)),
    (_with("scan", "kikuchipy"), ("bruker", True)),                      # kikuchipy's own h5ebsd holds Bruker numbers
    (_with("Pattern.dat"), ("bruker", False)),                           # NORDIF: not said by the file
    (_with("scan.ctf"), ("bruker", False)),                              # a format no rule knows
])
def test_the_convention_comes_from_the_format_the_reader_recorded(sig, expected):
    assert cal.convention_of(sig) == expected


def test_a_signal_that_records_no_name_is_told_by_the_extension_of_its_file():
    """The unified-loader fallback records neither a file name nor a maker: the format key is no help, so the
    extension of the path the store was given says it."""
    sig = _loaded_as("")
    sig.metadata.set_item("General.original_filename", "")
    assert cal.convention_of(sig) == ("bruker", False)
    assert cal.convention_of(sig, "D:/x/Scan 3.h5oina") == ("oxford", True)
    assert cal.convention_of(sig, "D:/x/map.ANG") == ("tsl", True)
    assert cal.convention_of(sig, "D:/x/scan.h5") == ("bruker", False)


def test_the_format_key_wins_over_the_name_of_the_file():
    """An EDAX h5 called like an Oxford file is still EDAX: the maker the reader recorded knows better."""
    assert cal.convention_of(_with("scan", "edax"), "D:/x/scan.h5oina") == ("tsl", True)


def test_a_stand_in_signal_without_metadata_has_no_convention_but_does_not_raise():
    sig = SimpleNamespace(detector=EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0), metadata=None)
    assert cal.convention_of(sig) == ("bruker", False)


def _register(store, name, sig, file_key=None):
    sig.detector = EBSDDetector(shape=(128, 156), pc=(0.5, 0.4, 0.6), sample_tilt=70.0)
    return store.register(name, sig, file_key=file_key)


def test_the_store_keeps_the_convention_of_the_file_and_a_copy_takes_it():
    store = CalibrationStore()
    _register(store, "ox", _with("scan 1", "oxford instruments"))
    store.register_derived("ox_copy", "ox")
    from backend.api.services.crop_window import CropWindow
    store.register_cropped("ox_crop", "ox", CropWindow("scan.h5oina", 0, 0, 2, 2, (4, 4)))
    for name in ("ox", "ox_copy", "ox_crop"):
        got = pc_in_use.pc_in_use_of(name, store)
        assert (got["convention"], got["convention_known"]) == ("oxford", True), name
    _register(store, "unknown", _with("Pattern.dat"))
    got = pc_in_use.pc_in_use_of("unknown", store)
    assert (got["convention"], got["convention_known"]) == ("bruker", False)


def test_an_entry_registered_by_older_code_reads_as_bruker_and_not_known():
    store = CalibrationStore()
    _register(store, "ds", _with("scan.up1"))
    store.get_entry("ds").convention = None
    got = pc_in_use.pc_in_use_of("ds", store)
    assert (got["convention"], got["convention_known"]) == ("bruker", False)


# ------------------------------------------------------------------ the mode of a search record

def test_a_record_carries_the_mode_only_when_it_has_one():
    from backend.api.routes import pcrefinement as R
    assert R._refinement_record(9, "PSO", mode="find") == {"n_patterns": 9, "method": "PSO", "mode": "find"}
    assert R._refinement_record(5, "Nelder-Mead", mode="refine") == {
        "n_patterns": 5, "method": "Nelder-Mead", "mode": "refine"}
    assert R._refinement_record(5, "Nelder-Mead") == {"n_patterns": 5, "method": "Nelder-Mead"}     # the grid
    assert R._refinement_record(5, "PSO", mode="guess") == {"n_patterns": 5, "method": "PSO"}


# ------------------------------------------------------------------ /api/pc/status

class _Sig:
    def __init__(self, tag):
        self.detector = EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0)
        self.metadata = type("M", (), {"get_item": staticmethod(lambda k, d=None: tag if k == "Signal.pc_source"
                                                                 else d)})()


@pytest.fixture
def client(monkeypatch):
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from pc_controller import PCController
    store = CalibrationStore()
    ctrl = PCController()
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    return TestClient(app), store


def test_status_says_where_the_pc_in_use_came_from(client):
    c, store = client
    store.register("ds", _Sig(None))
    got = c.get("/api/pc/status").json()["pc_in_use"]
    assert got["kind"] == "file" and got["pc"] == pytest.approx([0.5, 0.4, 0.6]) and got["per_pixel"] is False
    assert got["calibration"] is None and got["convention"] == "bruker" and got["convention_known"] is False
    store.update_pc("ds", FOUND, source="refined", refinement={"n_patterns": 9, "method": "PSO", "mode": "find"})
    assert c.get("/api/pc/status").json()["pc_in_use"]["kind"] == "find"
    store.update_pc("ds", FOUND, source="manual")
    assert c.get("/api/pc/status").json()["pc_in_use"]["kind"] == "typed"


def test_status_for_a_file_without_a_pc_has_kind_none(client):
    c, store = client
    store.register("ds", _Sig("default"))
    got = c.get("/api/pc/status").json()["pc_in_use"]
    assert got["kind"] == "none" and got["pc"] is None


def test_status_names_the_calibration_the_pc_runs_on(client, tmp_path):
    c, store = client
    path = tmp_path / "scan.h5oina"
    path.write_bytes(b"scan" * 50)
    store.register("ds", _Sig(None), file_key=str(path))
    made = cal.create(name="REM 2026-10-14", pc=FOUND, features=cal.features_of_signal(_Sig(None)),
                      origin={"method": "find"})
    use.apply(made["id"], ["ds"], accept_mismatch=["ds"], store=store)
    got = c.get("/api/pc/status").json()["pc_in_use"]
    assert got["kind"] == "calibration"
    assert got["calibration"] == {"id": made["id"], "name": "REM 2026-10-14", "version": 1}


def test_status_without_an_active_dataset_is_none(client, monkeypatch):
    from backend.api.routes import ebsd_viewer
    c, store = client
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", None)
    got = c.get("/api/pc/status").json()["pc_in_use"]
    assert got["kind"] == "none" and got["pc"] is None and got["convention"] is None


def test_status_of_an_oxford_file_with_a_map_gives_the_mean_and_the_convention(client):
    c, store = client
    sig = _with("scan 1", "oxford instruments")
    _register(store, "ds", sig)
    pc_map = np.stack([np.full((2, 3), v) for v in (0.4, 0.3, 0.8)], axis=-1)
    store.update_pc_map("ds", pc_map, source="header")
    got = c.get("/api/pc/status").json()["pc_in_use"]
    assert got["per_pixel"] is True and got["kind"] == "file" and got["pc"] == pytest.approx([0.4, 0.3, 0.8])
    assert got["convention"] == "oxford" and got["convention_known"] is True


# ------------------------------------------------------------------ the record of a result

def test_the_run_record_says_where_the_pc_came_from_and_how_good_it_was():
    from backend.api.services.pc_record import pc_run_record
    store = CalibrationStore()
    store.register("ds", _Sig(None))
    rec = pc_run_record("ds", store)
    assert rec["pc_origin_kind"] == "file" and rec["pc_quality"] is None
    store.update_pc("ds", FOUND, source="refined", refinement={"n_patterns": 9, "method": "PSO", "mode": "find"})
    use.note_search("ds", {"pc": list(FOUND), "mode": "find", "n_patterns": 9, "on_edge": ["x"],
                           "confidence": {"contrast": 0.8, "typical_cost": 8.0, "reasons": []}})
    rec = pc_run_record("ds", store)
    assert rec["pc_origin_kind"] == "find"
    assert rec["pc_quality"] == {"contrast": 0.8, "warning": "edge", "axes": ["x"], "n_patterns": 9}
    # typed numbers given with the run itself are typed and carry no quality
    typed = pc_run_record("ds", store, typed_pc=[0.45, 0.25, 0.55])
    assert typed["pc_origin_kind"] == "typed" and typed["pc_quality"] is None
    # a PC typed afterwards is no search result
    store.update_pc("ds", (0.45, 0.25, 0.55), source="manual")
    rec = pc_run_record("ds", store)
    assert rec["pc_origin_kind"] == "typed" and rec["pc_quality"] is None


def test_the_run_record_of_a_detector_that_is_not_in_the_store_names_no_origin():
    from backend.api.services.pc_record import pc_run_record
    det = EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0)
    rec = pc_run_record("not_in_store", CalibrationStore(), detector=det)
    assert rec["pc_origin_kind"] is None and rec["pc_quality"] is None


# ----- real files: the convention each vendor's format gives (ORIENTA_TEST_DATA = the Test_data folder)

def _real(name):
    base = Path(os.environ.get("ORIENTA_TEST_DATA") or Path(__file__).resolve().parents[1] / "Test_data")
    path = base / name
    if not path.exists():
        pytest.skip(f"{name} is not in {base} (set ORIENTA_TEST_DATA to the folder that holds the test data)")
    return path


@pytest.mark.parametrize("name,expected", [
    ("EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina", ("oxford", True)),
    ("HiGainNi.h5", ("tsl", True)),
    ("new test/Scan5.up1", ("tsl", True)),
])
def test_a_real_file_of_each_vendor_gives_its_convention(name, expected):
    from safe_loader import load_ebsd_safe
    path = _real(name)
    store = CalibrationStore()
    store.register("ds", load_ebsd_safe(str(path), verbose=False), file_key=str(path))
    got = pc_in_use.pc_in_use_of("ds", store)
    assert (got["convention"], got["convention_known"]) == expected
