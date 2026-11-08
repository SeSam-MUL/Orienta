"""Named pattern-centre calibrations: storage, versions, fingerprints, the "same settings" rule."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibrations as cal  # noqa: E402

FEATURES = {"pattern_shape": [128, 156], "binning": 1, "camera_tilt": 4.32, "sample_tilt": 70.0,
            "working_distance_mm": 15.0, "detector_insertion_mm": 165.65,
            "hints": {"date": "2026-10-14", "microscope": "SEM A"}}
ORIGIN = {"method": "find", "dataset": "scan_a", "n_patterns": 9}


@pytest.fixture(autouse=True)
def folder(tmp_path):
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    yield tmp_path / "Calibrations"
    cal.set_calibrations_dir_for_test(None)


def _make(name="REM 2026-10-14, WD 15 mm", pc=(0.5026, 0.3272, 0.8458), **kw):
    return cal.create(name=name, pc=pc, features=FEATURES, origin=ORIGIN, **kw)


# ------------------------------------------------------------------ the file on disk

def test_a_calibration_is_one_lf_json_file_without_any_path(folder):
    made = _make(author="S. Tester")
    path = folder / f"{made['id']}.json"
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    text = raw.decode("utf-8")
    doc = json.loads(text)
    assert doc["schema"] == 1 and doc["versions"][0]["pc"] == [0.5026, 0.3272, 0.8458]
    assert doc["author"] == "S. Tester" and doc["features"]["pattern_shape"] == [128, 156]
    # no absolute path, drive letter, user folder or host name anywhere in the file
    for needle in (":\\", ":/", "/Users/", "/home/", "\\\\"):
        assert needle not in text, needle


def test_author_is_never_filled_in_from_the_machine():
    assert _make(author="")["author"] == "unknown"


def test_the_id_is_minted_once_and_survives_a_rename(folder):
    made = _make()
    renamed = cal.rename(made["id"], "REM, new name")
    assert renamed["id"] == made["id"] and renamed["name"] == "REM, new name"
    assert cal.get(made["id"])["name"] == "REM, new name"
    assert [p.name for p in folder.glob("*.json")] == [f"{made['id']}.json"]


def test_names_are_unique_ignoring_case_and_spacing():
    _make("Same Name")
    with pytest.raises(cal.CalibrationError) as exc:
        _make("  same   name ")
    assert exc.value.code == "nameTaken"
    with pytest.raises(cal.CalibrationError) as exc:
        _make("   ")
    assert exc.value.code == "needsName"
    other = _make("Another")
    with pytest.raises(cal.CalibrationError) as exc:
        cal.rename(other["id"], "SAME NAME")
    assert exc.value.code == "nameTaken"


def test_a_calibration_without_the_pattern_shape_or_a_finite_pc_is_refused():
    with pytest.raises(cal.CalibrationError) as exc:
        cal.create(name="x", pc=(0.5, 0.5, 0.5), features={"binning": 1}, origin=ORIGIN)
    assert exc.value.code == "needsFeatures"
    with pytest.raises(cal.CalibrationError) as exc:
        cal.create(name="y", pc=(0.5, float("nan"), 0.5), features=FEATURES, origin=ORIGIN)
    assert exc.value.code == "badPc"


@pytest.mark.parametrize("shape", [[float("nan"), 60], ["a", 60], [float("inf"), 60], [None, 60], [60, float("-inf")]])
def test_a_pattern_shape_that_is_no_pair_of_whole_numbers_is_needs_features_not_a_raw_error(shape):
    with pytest.raises(cal.CalibrationError) as exc:
        cal.create(name="x", pc=(0.5, 0.5, 0.5), features={**FEATURES, "pattern_shape": shape}, origin=ORIGIN)
    assert exc.value.code == "needsFeatures"


# ------------------------------------------------------------------ versions

def test_a_new_version_keeps_the_old_one_readable():
    made = _make()
    cal.add_version(made["id"], pc=(0.51, 0.33, 0.85), origin={"method": "refine"})
    stored = cal.get(made["id"])
    assert [v["version"] for v in stored["versions"]] == [1, 2]
    assert cal.latest(stored)["pc"] == [0.51, 0.33, 0.85]
    assert cal.version_pc(stored, 1) == [0.5026, 0.3272, 0.8458]
    with pytest.raises(cal.CalibrationError) as exc:
        cal.version_pc(stored, 3)
    assert exc.value.code == "unknownVersion"
    assert cal.summary(stored)["version"] == 2 and cal.summary(stored)["n_versions"] == 2


# ------------------------------------------------------------------ the "same settings" rule (hand cases)

@pytest.mark.parametrize("change,key", [
    ({"pattern_shape": [60, 60]}, "pattern_shape"),
    ({"binning": 2}, "binning"),
    ({"camera_tilt": 4.5}, "camera_tilt"),               # 0.18 deg apart, tolerance 0.1
    ({"sample_tilt": 69.8}, "sample_tilt"),              # 0.2 apart
    ({"working_distance_mm": 15.2}, "working_distance_mm"),   # 0.2 mm apart, tolerance 0.1
])
def test_a_setting_beyond_its_tolerance_is_a_difference(change, key):
    out = cal.compare_features(FEATURES, {**FEATURES, **change})
    assert out["match"] is False
    assert [d["key"] for d in out["differences"]] == [key]
    assert out["differences"][0]["calibration"] == FEATURES[key] and out["differences"][0]["file"] == change[key]


@pytest.mark.parametrize("change", [
    {"camera_tilt": 4.40},             # 0.08 apart
    {"sample_tilt": 70.05},
    {"working_distance_mm": 15.09},    # 0.09 mm apart
    {"hints": {"date": "2027-01-01", "microscope": "SEM B"}},     # date and device are only hints
])
def test_a_setting_within_its_tolerance_is_not_a_difference(change):
    assert cal.compare_features(FEATURES, {**FEATURES, **change})["match"] is True


def test_a_setting_one_side_does_not_state_is_unknown_not_different():
    out = cal.compare_features(FEATURES, {**FEATURES, "working_distance_mm": None})
    assert out["match"] is True and out["unknown"] == ["working_distance_mm"]
    out = cal.compare_features({**FEATURES, "working_distance_mm": None}, FEATURES)
    assert out["match"] is True and out["unknown"] == ["working_distance_mm"]
    out = cal.compare_features(FEATURES, {**FEATURES, "detector_insertion_mm": None})
    assert out["match"] is True and out["unknown"] == ["detector_insertion_mm"]


# ------------------------------------------------------------------ files are recognised without their path

def test_a_fingerprint_is_name_size_and_a_header_id_and_holds_no_folder(tmp_path):
    a = tmp_path / "deep" / "folder"
    a.mkdir(parents=True)
    f = a / "scan1.h5oina"
    f.write_bytes(b"A" * 100)
    fp = cal.fingerprint_of(f)
    assert set(fp) == {"name", "size", "header_id"} and fp["name"] == "scan1.h5oina" and fp["size"] == 100
    assert "deep" not in json.dumps(fp)
    g = tmp_path / "copy.h5oina"
    g.write_bytes(b"A" * 100)
    assert cal.fingerprint_of(g)["header_id"] == fp["header_id"]            # the id is of the content


def test_the_header_id_looks_at_the_first_64_kib_only(tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * cal.HEADER_BYTES + b"tail one")
    g = tmp_path / "big2.bin"
    g.write_bytes(b"x" * cal.HEADER_BYTES + b"tail two")
    assert cal.fingerprint_of(f)["header_id"] == cal.fingerprint_of(g)["header_id"]
    h = tmp_path / "big3.bin"
    h.write_bytes(b"y" + b"x" * (cal.HEADER_BYTES - 1) + b"tail one")
    assert cal.fingerprint_of(f)["header_id"] != cal.fingerprint_of(h)["header_id"]


def test_a_missing_file_has_no_fingerprint(tmp_path):
    with pytest.raises(cal.CalibrationError) as exc:
        cal.fingerprint_of(tmp_path / "nope.h5")
    assert exc.value.code == "fileUnreadable"


FP = {"name": "scan1.h5oina", "size": 100, "header_id": "aa"}


def test_a_file_is_a_member_once_and_a_second_file_is_another_member():
    made = _make()
    cal.set_member(made["id"], FP, 1)
    cal.set_member(made["id"], FP, 2)                       # the same file again: its version moves
    cal.set_member(made["id"], {**FP, "name": "scan2.h5oina", "header_id": "bb"}, 1)
    members = cal.get(made["id"])["members"]
    assert [(m["name"], m["version"]) for m in members] == [("scan1.h5oina", 2), ("scan2.h5oina", 1)]


# ------------------------------------------------------------------ suggesting never writes

def test_suggest_lists_members_before_feature_matches_and_leaves_out_the_rest(folder):
    mine = _make("A: member and match")
    cal.set_member(mine["id"], FP, 1)
    other = _make("B: match only")
    far = cal.create(name="C: other detector", pc=(0.5, 0.4, 0.6),
                     features={**FEATURES, "pattern_shape": [60, 60]}, origin=ORIGIN)
    before = {p.name: p.read_bytes() for p in folder.glob("*.json")}
    got = cal.suggest(FEATURES, FP)
    assert [s["name"] for s in got] == ["A: member and match", "B: match only"]
    assert got[0]["via_member"] is True and got[0]["member_version"] == 1
    assert got[1]["via_member"] is False and got[1]["match"] is True
    assert far["id"] not in [s["id"] for s in got] and other["id"] in [s["id"] for s in got]
    assert {p.name: p.read_bytes() for p in folder.glob("*.json")} == before       # nothing was written


def test_a_member_whose_settings_differ_is_still_offered_with_the_differences():
    mine = _make()
    cal.set_member(mine["id"], FP, 1)
    got = cal.suggest({**FEATURES, "working_distance_mm": 20.0}, FP)
    assert len(got) == 1 and got[0]["via_member"] is True and got[0]["match"] is False
    assert got[0]["differences"][0]["key"] == "working_distance_mm"
    assert cal.suggest({**FEATURES, "working_distance_mm": 20.0}, None) == []


def test_a_member_is_listed_before_a_non_member_even_when_it_matches_worse():
    """Order is members first, then matches. The names make the folder order put the plain match first and
    both are made within one second, so only the sort by membership can put the member on top."""
    plain = cal.create(name="A: plain match", pc=(0.5, 0.4, 0.6), origin=ORIGIN,
                       features={**FEATURES, "working_distance_mm": 20.0})
    mine = _make("Z: member with a different working distance")           # WD 15, the file has 20
    cal.set_member(mine["id"], FP, 1)
    got = cal.suggest({**FEATURES, "working_distance_mm": 20.0}, FP)
    assert [s["id"] for s in got] == [mine["id"], plain["id"]]
    assert got[0]["match"] is False and got[1]["match"] is True


def test_hints_travel_with_the_proposal():
    made = _make()
    got = cal.suggest({**FEATURES, "hints": {"date": "2026-11-02"}}, None)
    assert got[0]["hints"] == {"calibration": FEATURES["hints"], "file": {"date": "2026-11-02"}}
    assert made["id"] == got[0]["id"]


# ------------------------------------------------------------------ the folder

def test_a_file_this_build_cannot_use_is_reported_and_never_rewritten(folder):
    _make()
    newer = folder / "from-the-future-abc123.json"
    newer.write_text(json.dumps({"schema": 99, "id": "from-the-future-abc123", "name": "x",
                                 "versions": [], "features": {}, "members": []}), encoding="utf-8")
    broken = folder / "broken-def456.json"
    broken.write_text("{ not json", encoding="utf-8")
    before = (newer.read_bytes(), broken.read_bytes())
    found, problems = cal.load_all()
    assert [c["name"] for c in found] == ["REM 2026-10-14, WD 15 mm"]
    assert {p["file"]: p["code"] for p in problems} == {
        "from-the-future-abc123.json": "newerSchema", "broken-def456.json": "unreadable"}
    _make("a second one")                                  # a write beside them
    assert (newer.read_bytes(), broken.read_bytes()) == before


def test_parallel_writes_to_one_calibration_lose_nothing():
    """Twelve files become members at once: a read-modify-write without the folder lock keeps the
    last writer's list only (the lost update that cost phase collections nine of twelve)."""
    made = _make()
    ready = threading.Barrier(12)

    def one(i):
        ready.wait()
        cal.set_member(made["id"], {"name": f"f{i}.h5", "size": i, "header_id": f"id{i}"}, 1)

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(one, range(12)))
    assert len(cal.get(made["id"])["members"]) == 12


def test_every_writer_takes_the_folder_lock():
    for name in ("create", "add_version", "set_member", "rename", "delete"):
        assert getattr(getattr(cal, name), "_serialised", False), name


def test_delete_removes_the_file_and_says_whether_there_was_one(folder):
    made = _make()
    assert cal.delete(made["id"]) is True and cal.get(made["id"]) is None
    assert cal.delete(made["id"]) is False
    with pytest.raises(cal.CalibrationError) as exc:
        cal.get("../../etc/passwd")
    assert exc.value.code == "unknownCalibration"


def test_the_default_folder_is_beside_the_library_and_an_environment_variable_moves_it(monkeypatch, tmp_path):
    cal.set_calibrations_dir_for_test(None)
    monkeypatch.delenv(cal.ENV_DIR, raising=False)
    from path_utils import get_local_database_path
    assert cal.calibrations_dir() == get_local_database_path() / "Calibrations"
    monkeypatch.setenv(cal.ENV_DIR, str(tmp_path / "elsewhere"))
    assert cal.calibrations_dir() == tmp_path / "elsewhere"


# ------------------------------------------------------------------ the settings of a loaded signal

def _signal(**metadata):
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=(128, 156), pc=(0.5, 0.33, 0.85), sample_tilt=70.0, tilt=4.32, binning=2)
    sig = kp.signals.EBSD(np.zeros((2, 2, 128, 156), np.uint8), detector=det)
    sig.metadata.set_item("General.original_filename", "scan.h5oina")     # a file reader records its file
    for path, value in metadata.items():
        sig.metadata.set_item(path.replace("__", "."), value)
    return sig


def test_the_settings_of_a_signal_come_from_its_detector_and_header():
    sig = _signal(Acquisition_instrument__SEM__working_distance=16.04,
                  Acquisition_instrument__SEM__beam_energy=20.0)
    f = cal.features_of_signal(sig)
    assert f["pattern_shape"] == [128, 156] and f["binning"] == 2
    assert f["camera_tilt"] == pytest.approx(4.32) and f["sample_tilt"] == pytest.approx(70.0)
    assert f["working_distance_mm"] == pytest.approx(16.04)
    assert f["hints"]["beam_energy_kv"] == pytest.approx(20.0)


def test_a_header_without_a_working_distance_gives_none():
    assert cal.features_of_signal(_signal())["working_distance_mm"] is None


# ------------------------------------------------------------------ a setting the format does not carry is unknown

def _loaded_as(filename, *, detector=None, pc_source=None):
    """A signal as a loader of that file format leaves it: the file name in the metadata (as kikuchipy's
    readers record it, sometimes with the folder) and, unless the test hands one over, kikuchipy's DEFAULT
    detector (camera tilt 0, sample tilt 70, binning 1): what a reader gives for a file that said nothing."""
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    det = detector or EBSDDetector(shape=(128, 156), pc=(0.5, 0.5, 0.5))
    sig = kp.signals.EBSD(np.zeros((2, 2, 128, 156), np.uint8), detector=det)
    sig.metadata.set_item("General.original_filename", filename)
    if pc_source:
        sig.metadata.set_item("Signal.pc_source", pc_source)
    return sig


SETTINGS = ["binning", "camera_tilt", "sample_tilt", "working_distance_mm", "detector_insertion_mm"]


def test_an_ebsp_style_signal_has_unknown_tilts_and_unknown_is_not_a_difference():
    """An Oxford .ebsp carries no detector geometry, so kikuchipy's detector holds 0 / 70 / 1 and none of it was
    measured. The settings are None; against a calibration made on a file that stated them they are unknown:
    not a difference (the defaults would differ from 4.32 degrees) and not a silent match either."""
    f = cal.features_of_signal(_loaded_as("C:/data/map.ebsp"))
    assert f["pattern_shape"] == [128, 156]
    assert [f[k] for k in SETTINGS] == [None] * len(SETTINGS)
    out = cal.compare_features(FEATURES, f)
    assert out["differences"] == [] and sorted(out["unknown"]) == sorted(SETTINGS)
    assert out["checked"] == ["pattern_shape"] and out["unverified"] is True


@pytest.mark.parametrize("pc_source", ["default", "osc"])
@pytest.mark.parametrize("name", ["scan.up1", "SCAN.UP2"])
def test_an_up_file_states_no_tilt_and_no_binning_with_or_without_an_osc_beside_it(name, pc_source):
    """safe_loader._load_edax_up builds a new detector from the .osc's PC and keeps no tilt: the tilts the
    detector holds afterwards are kikuchipy's defaults, whatever the sidecar contained."""
    f = cal.features_of_signal(_loaded_as(name, pc_source=pc_source))
    assert [f[k] for k in ("binning", "camera_tilt", "sample_tilt")] == [None, None, None]


def test_a_nordif_signal_states_its_tilts_but_not_its_binning():
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=(128, 156), pc=(0.5, 0.5, 0.5), sample_tilt=70.0, tilt=3.0)
    f = cal.features_of_signal(_loaded_as("Pattern.dat", detector=det))
    assert f["sample_tilt"] == pytest.approx(70.0) and f["camera_tilt"] == pytest.approx(3.0)
    assert f["binning"] is None


@pytest.mark.parametrize("manufacturer", ["edax", "EDAX", "bruker nano", "Bruker Nano"])
@pytest.mark.parametrize("name", ["HiGainNi", "scan.h5", "scan.h5ebsd"])
def test_an_edax_or_bruker_h5_states_its_tilts_but_not_its_binning(name, manufacturer):
    """kikuchipy's EDAX and Bruker h5ebsd readers never read a binning (the detector keeps its default 1). They
    record the file name WITHOUT its extension, so the manufacturer in the original metadata is what says it."""
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=(128, 156), pc=(0.5, 0.4, 0.8), sample_tilt=70.0, tilt=4.32)
    sig = _loaded_as(name, detector=det)
    sig.original_metadata.set_item("manufacturer", manufacturer)
    f = cal.features_of_signal(sig)
    assert f["binning"] is None
    assert f["camera_tilt"] == pytest.approx(4.32) and f["sample_tilt"] == pytest.approx(70.0)
    assert cal.settings_stated(sig) == {"camera_tilt", "sample_tilt", "detector_insertion_mm"}
    out = cal.compare_features(FEATURES, f)
    # this signal has no working distance and no insertion either
    assert out["unknown"] == ["binning", "working_distance_mm", "detector_insertion_mm"]
    assert out["checked"] == ["pattern_shape", "camera_tilt", "sample_tilt"]
    assert out["match"] is True and out["differences"] == []


@pytest.mark.parametrize("manufacturer", ["oxford instruments", "kikuchipy"])
def test_other_h5_formats_keep_their_binning(manufacturer):
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=(128, 156), pc=(0.5, 0.4, 0.8), sample_tilt=70.0, tilt=4.32, binning=2)
    sig = _loaded_as("scan", detector=det)
    sig.original_metadata.set_item("manufacturer", manufacturer)
    assert cal.features_of_signal(sig)["binning"] == 2


@pytest.mark.parametrize("name", ["Pattern.dat", "pattern.dat", "scan_07.dat", "C:\\data\\Run 3\\NORDIF.DAT", "/d/x/p.dat"])
def test_every_dat_file_is_read_as_nordif_so_its_binning_is_not_stated(name):
    """The only reader that takes a .dat is kikuchipy's NORDIF one, whatever the file is called."""
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=(128, 156), pc=(0.5, 0.5, 0.5), sample_tilt=70.0, tilt=3.0)
    f = cal.features_of_signal(_loaded_as(name, detector=det))
    assert f["binning"] is None and f["camera_tilt"] == pytest.approx(3.0)


def test_a_signal_that_names_no_manufacturer_and_no_known_file_keeps_all_settings():
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=(128, 156), pc=(0.5, 0.4, 0.8), sample_tilt=70.0, tilt=4.32, binning=2)
    sig = _loaded_as("scan.h5oina", detector=det)
    assert cal.settings_stated(sig) == {"binning", "camera_tilt", "sample_tilt", "detector_insertion_mm"}


def test_a_signal_from_the_unified_loader_fallback_states_neither_binning_nor_tilts():
    """``ebsd_adapter`` builds the signal from a ``DetectorData`` whose binning is never read from the file (the
    dataclass default 1) and whose tilts fall back to defaults (0 and 70): it records neither file name nor
    manufacturer. Nothing it holds is a measured setting, whatever the header said (the Oxford reference scan: binning 8 and
    camera tilt 4.32 in the file, 1 and 0 in this signal)."""
    from ebsd_adapter import ebsd_data_to_kikuchipy_signal
    from unified_loader import DetectorData, EBSDData, NavigationData
    det = DetectorData(0.5, 0.5, 0.5, np.zeros((4, 3)), False, 70.0, 0.0, 0.0, 0.0, 16, 12)
    nav = NavigationData(2, 2, 1.0, 1.0, np.zeros(4), np.zeros(4))
    sig = ebsd_data_to_kikuchipy_signal(
        EBSDData(np.zeros((4, 12, 16), np.uint8), "raw", det, nav, "Oxford", "scan.h5"))
    assert cal.settings_stated(sig) == set()
    f = cal.features_of_signal(sig)
    assert f["pattern_shape"] == [12, 16]
    assert f["binning"] is None and f["camera_tilt"] is None and f["sample_tilt"] is None
    assert cal.compare_features(FEATURES, f)["unverified"] is True


def test_a_format_that_carries_the_settings_keeps_them():
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=(128, 156), pc=(0.5, 0.4, 0.8), sample_tilt=70.0, tilt=4.32, binning=2)
    f = cal.features_of_signal(_loaded_as("scan.h5oina", detector=det))
    assert f["binning"] == 2 and f["camera_tilt"] == pytest.approx(4.32) and f["sample_tilt"] == pytest.approx(70.0)


def test_the_comparison_says_what_it_could_check():
    out = cal.compare_features(FEATURES, FEATURES)
    assert out["checked"] == ["pattern_shape"] + SETTINGS and out["unverified"] is False     # all 6
    out = cal.compare_features(FEATURES, {**FEATURES, "binning": None, "camera_tilt": None, "sample_tilt": None,
                                          "detector_insertion_mm": None})
    assert out["checked"] == ["pattern_shape", "working_distance_mm"] and out["unverified"] is False   # one is still compared
    out = cal.compare_features(FEATURES, {**FEATURES, **{k: None for k in SETTINGS}})
    assert out["checked"] == ["pattern_shape"] and out["unverified"] is True
    assert cal.stated_settings({**FEATURES, "camera_tilt": None}) == [
        "binning", "sample_tilt", "working_distance_mm", "detector_insertion_mm"]


def test_a_calibration_that_could_only_be_compared_by_shape_is_offered_to_its_members_only():
    """A calibration saved from a file that stated none of its settings fits any file of the same size, which says
    nothing: it is not suggested on the shape alone; a file that was a member of it is still offered it, marked."""
    made = cal.create(name="from an ebsp", pc=(0.5, 0.4, 0.7), origin=ORIGIN,
                      features={"pattern_shape": [128, 156], **{k: None for k in SETTINGS}})
    assert cal.suggest(FEATURES, None) == []
    cal.set_member(made["id"], FP, 1)
    got = cal.suggest(FEATURES, FP)
    assert len(got) == 1 and got[0]["via_member"] is True and got[0]["unverified"] is True


# ------------------------------------------------------------------ the tolerance edge, with real floating-point values

@pytest.mark.parametrize("key,base,other", [
    ("camera_tilt", 4.32, 4.42), ("camera_tilt", 4.42, 4.32),             # as floats: 0.09999999999999964 apart
    ("sample_tilt", 70.0, 70.1), ("sample_tilt", 70.1, 70.0),             # 0.09999999999999432
    ("working_distance_mm", 15.0, 15.1), ("working_distance_mm", 15.1, 15.0),
    ("camera_tilt", 4.3, 4.4), ("camera_tilt", 4.4, 4.3),                 # 0.10000000000000053: a hair ABOVE 0.1
    ("sample_tilt", 70.1, 70.2), ("sample_tilt", 70.2, 70.1),             # 0.10000000000000853
    ("working_distance_mm", 16.04, 16.14), ("working_distance_mm", 16.14, 16.04),     # 0.10000000000000142
])
def test_a_measurement_exactly_one_tolerance_apart_is_the_same(key, base, other):
    assert abs(base - other) > 0 and abs(abs(base - other) - 0.1) < 1e-9          # "0.1 apart" as the machine has it
    out = cal.compare_features({**FEATURES, key: base}, {**FEATURES, key: other})
    assert out["match"] is True and out["differences"] == [] and key in out["checked"]


@pytest.mark.parametrize("key,base,other", [
    ("camera_tilt", 4.32, 4.43), ("sample_tilt", 70.0, 70.11), ("sample_tilt", 70.11, 70.0),
    ("working_distance_mm", 15.0, 15.11), ("camera_tilt", 4.32, 4.4200001),
])
def test_a_measurement_beyond_one_tolerance_differs(key, base, other):
    out = cal.compare_features({**FEATURES, key: base}, {**FEATURES, key: other})
    assert out["match"] is False and [d["key"] for d in out["differences"]] == [key]
    assert out["differences"][0]["tolerance"] == 0.1


# ------------------------------------------------------------------ what makes two files the same file

@pytest.mark.parametrize("change", [{"name": "scan2.h5oina"}, {"size": 101}, {"header_id": "ab"}])
def test_files_differing_in_one_part_of_the_fingerprint_are_not_the_same_file(change):
    assert cal.same_file(FP, dict(FP)) is True
    assert cal.same_file(FP, {**FP, **change}) is False
    assert cal.same_file({**FP, **change}, FP) is False


def test_a_file_that_differs_in_one_part_is_another_member_and_not_a_suggestion_by_membership():
    made = _make()
    cal.set_member(made["id"], FP, 1)
    other = {**FP, "header_id": "ab"}
    cal.set_member(made["id"], other, 1)
    assert len(cal.get(made["id"])["members"]) == 2
    got = cal.suggest({**FEATURES, "working_distance_mm": 20.0}, {**FP, "size": 101})
    assert got == []                                       # neither a member nor a match


# ------------------------------------------------------------------ the atomic write

def test_a_write_that_fails_halfway_leaves_the_old_file_and_no_temp_file(folder):
    made = _make()
    path = folder / f"{made['id']}.json"
    before = path.read_bytes()
    with pytest.raises(UnicodeEncodeError):
        cal._write_atomic(path, "new content \ud800")          # a lone surrogate cannot be encoded
    assert path.read_bytes() == before
    assert [p.name for p in folder.iterdir()] == [path.name]


def test_a_temp_file_left_behind_by_a_crash_is_not_a_calibration(folder):
    folder.mkdir(parents=True)
    stray = folder / ".calibration_abc123.json.tmp"
    stray.write_text('{"half": ', encoding="utf-8")
    made = _make()
    found, problems = cal.load_all()
    assert [c["id"] for c in found] == [made["id"]] and problems == []
    assert stray.read_text(encoding="utf-8") == '{"half": '


@pytest.mark.parametrize("content,code", [
    ("{ not json", "unreadable"),
    (json.dumps({"schema": 99, "id": "x-abc123", "name": "x", "versions": [], "features": {}, "members": []}), "newerSchema"),
])
def test_adding_a_version_or_renaming_on_a_file_that_cannot_be_used_changes_nothing(folder, content, code):
    other = _make("Other")
    path = _put(folder, "odd-abc123.json", content)
    before = path.read_bytes()
    with pytest.raises(cal.CalibrationError) as exc:
        cal.add_version("odd-abc123", pc=(0.5, 0.5, 0.5))
    assert exc.value.code == code
    with pytest.raises(cal.CalibrationError) as exc:
        cal.rename("odd-abc123", "Fresh name")
    assert exc.value.code == code
    with pytest.raises(cal.CalibrationError):
        cal.set_member("odd-abc123", FP, 1)
    assert path.read_bytes() == before
    assert [p.name for p in sorted(folder.iterdir())] == sorted(["odd-abc123.json", f"{other['id']}.json"])


@pytest.mark.parametrize("content,code", [
    ("{ not json", "unreadable"),
    (json.dumps({"schema": 99, "id": "odd-abc123", "name": "x", "versions": [], "features": {}, "members": []}), "newerSchema"),
])
def test_deleting_a_file_that_cannot_be_used_is_refused_and_leaves_it_untouched(folder, content, code):
    other = _make("Other")
    path = _put(folder, "odd-abc123.json", content)
    before = path.read_bytes()
    with pytest.raises(cal.CalibrationError) as exc:
        cal.delete("odd-abc123")
    assert exc.value.code == code
    assert path.read_bytes() == before
    assert sorted(p.name for p in folder.iterdir()) == sorted(["odd-abc123.json", f"{other['id']}.json"])


def test_add_version_or_rename_of_a_calibration_that_does_not_exist_says_so(folder):
    for call in (lambda: cal.add_version("nobody-abc123", pc=(0.5, 0.5, 0.5)), lambda: cal.rename("nobody-abc123", "x")):
        with pytest.raises(cal.CalibrationError) as exc:
            call()
        assert exc.value.code == "unknownCalibration"
    assert not folder.exists()


def test_reading_never_creates_the_folder(folder):
    assert not folder.exists()
    assert cal.load_all() == ([], [])
    assert cal.get("nobody-abc123") is None
    assert cal.suggest(FEATURES, FP) == []
    assert cal.delete("nobody-abc123") is False
    assert not folder.exists()
    _make()
    assert folder.is_dir()


# ------------------------------------------------------------------ ids

@pytest.mark.parametrize("name,prefix", [
    ("CON", "con-"), ("Müller & Söhne", "m-ller-s-hne-"), ("ß", "ss-"), ("äöü", "pc-"), ("a" * 200, "a" * 24 + "-"),
    ("  Aux  ", "aux-"), ("Ni / 2026-10-14: WD 15", "ni-2026-10-14-wd-15-"),
])
def test_an_id_is_a_safe_file_name_whatever_the_name_is(folder, name, prefix):
    made = cal.create(name=name, pc=(0.5, 0.4, 0.7), features=FEATURES, origin=ORIGIN)
    assert made["id"].startswith(prefix) and len(made["id"]) <= 31
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{2,60}", made["id"]) and (folder / f"{made['id']}.json").is_file()
    assert cal.get(made["id"])["name"] == " ".join(name.split())          # the name keeps its letters, only the id is a slug
    assert cal.load_all()[1] == []


def test_an_id_that_is_taken_is_not_minted_again(monkeypatch, folder):
    from types import SimpleNamespace
    folder.mkdir(parents=True)
    (folder / "probe-aaaaaa.json").write_text("{}", encoding="utf-8")           # taken, even though unusable
    hexes = iter(["aaaaaa", "aaaaaa", "bbbbbb"])
    monkeypatch.setattr(cal.uuid, "uuid4", lambda: SimpleNamespace(hex=next(hexes) + "0" * 26))
    made = cal.create(name="Probe", pc=(0.5, 0.4, 0.7), features=FEATURES, origin=ORIGIN)
    assert made["id"] == "probe-bbbbbb"


# ------------------------------------------------------------------ a file that states nothing

def test_a_file_that_states_none_of_its_settings_is_offered_only_to_its_members():
    made = _make()
    nothing = {"pattern_shape": [128, 156], **{k: None for k in SETTINGS}, "hints": {}}
    assert cal.suggest(nothing, None) == [] and cal.suggest(nothing, FP) == []
    cal.set_member(made["id"], FP, 1)
    got = cal.suggest(nothing, FP)
    assert len(got) == 1 and got[0]["via_member"] is True and got[0]["unverified"] is True and got[0]["match"] is False
    assert got[0]["unknown"] == SETTINGS and got[0]["differences"] == []


# ------------------------------------------------------------------ a setting that is not a finite number is not stated

NOT_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", NOT_FINITE)
@pytest.mark.parametrize("key", ["binning", "camera_tilt", "sample_tilt", "working_distance_mm", "detector_insertion_mm"])
@pytest.mark.parametrize("side", ["calibration", "file"])
def test_a_setting_that_is_not_finite_is_unknown_in_the_comparison_never_equal(key, bad, side):
    """abs(nan) > tolerance is False, so a nan used to count as 'within tolerance' and as checked."""
    mine, theirs = (({**FEATURES, key: bad}, FEATURES) if side == "calibration" else (FEATURES, {**FEATURES, key: bad}))
    out = cal.compare_features(mine, theirs)
    assert out["unknown"] == [key] and key not in out["checked"]
    assert out["differences"] == [] and out["match"] is True
    assert out["checked"] == ["pattern_shape"] + [k for k in SETTINGS if k != key]


def test_a_signal_whose_detector_holds_a_nan_states_nothing_for_it():
    from types import SimpleNamespace
    det = SimpleNamespace(shape=(128, 156), binning=float("nan"), tilt=float("nan"), sample_tilt=float("inf"))
    f = cal.features_of_signal(SimpleNamespace(detector=det, metadata=None))
    assert f["pattern_shape"] == [128, 156]
    assert [f[k] for k in ("binning", "camera_tilt", "sample_tilt")] == [None, None, None]


def test_a_calibration_never_stores_a_setting_that_is_not_finite(folder):
    made = cal.create(name="nan", pc=(0.5, 0.4, 0.7), origin=ORIGIN,
                      features={**FEATURES, "camera_tilt": float("nan"), "working_distance_mm": float("inf"),
                                "binning": float("-inf")})
    assert (made["features"]["camera_tilt"], made["features"]["working_distance_mm"],
            made["features"]["binning"]) == (None, None, None)
    assert made["features"]["sample_tilt"] == 70.0
    text = (folder / f"{made['id']}.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text


# ------------------------------------------------------------------ a path never lands in the file

@pytest.mark.parametrize("dataset", [r"C:\Users\someone\scans\scan_a.h5", "/home/someone/scans/scan_a.h5",
                                     r"D:/data\mixed/scan_a.h5", "scan_a.h5"])
def test_an_origin_dataset_is_reduced_to_the_file_name(folder, dataset):
    made = cal.create(name="path", pc=(0.5, 0.4, 0.7), features=FEATURES, origin={**ORIGIN, "dataset": dataset})
    cal.add_version(made["id"], pc=(0.51, 0.4, 0.7), origin={"method": "refine", "dataset": dataset})
    stored = cal.get(made["id"])
    assert [v["origin"]["dataset"] for v in stored["versions"]] == ["scan_a.h5", "scan_a.h5"]
    text = (folder / f"{made['id']}.json").read_text(encoding="utf-8")
    for needle in (":\\", ":/", "/home/", "someone", "\\\\", "D:"):
        assert needle not in text, needle


@pytest.mark.parametrize("name", [r"C:\Users\someone\scans\f.h5", "/home/someone/scans/f.h5", r"D:/data\mixed/f.h5"])
def test_a_member_name_is_reduced_to_the_file_name_and_still_the_same_file(folder, name):
    made = cal.create(name="path", pc=(0.5, 0.4, 0.7), features=FEATURES, origin=ORIGIN,
                      member={"name": name, "size": 7, "header_id": "hh"})
    assert made["members"][0]["name"] == "f.h5"
    cal.set_member(made["id"], {"name": "f.h5", "size": 7, "header_id": "hh"}, 1)       # the same file, no path
    cal.set_member(made["id"], {"name": name, "size": 7, "header_id": "hh"}, 1)         # and with its path again
    assert [m["name"] for m in cal.get(made["id"])["members"]] == ["f.h5"]
    assert cal.suggest(FEATURES, {"name": name, "size": 7, "header_id": "hh"})[0]["via_member"] is True
    text = (folder / f"{made['id']}.json").read_text(encoding="utf-8")
    for needle in (":\\", ":/", "/home/", "someone", "D:"):
        assert needle not in text, needle


# ------------------------------------------------------------------ a file that is being read can still be replaced

def _fail_first(monkeypatch, target, name, times, exc):
    """Replace ``target.name`` by a stand-in that raises ``exc`` for its first ``times`` calls (on the
    calibration files only) and then does the real thing; returns the call counter."""
    real = getattr(target, name)
    calls = {"n": 0}

    def stand_in(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= times:
            raise exc
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, stand_in)
    return calls


def _no_pause(monkeypatch):
    monkeypatch.setattr(cal, "_RETRY_PAUSE_S", 0)


def test_a_write_that_meets_a_reader_is_tried_again(monkeypatch, folder):
    """On Windows os.replace fails with PermissionError while anything has the target open: a request that
    reads the calibration must not make the write of a member fail. Injected here, so it holds on every platform."""
    made = _make()
    _no_pause(monkeypatch)
    calls = _fail_first(monkeypatch, os, "replace", 3, PermissionError(13, "in use"))
    cal.set_member(made["id"], FP, 1)
    assert calls["n"] == 4
    assert [m["name"] for m in cal.get(made["id"])["members"]] == ["scan1.h5oina"]
    assert [p.name for p in folder.iterdir()] == [f"{made['id']}.json"]            # no temp file left


def test_a_write_that_never_gets_the_file_gives_up_and_leaves_the_old_one(monkeypatch, folder):
    made = _make()
    _no_pause(monkeypatch)
    before = (folder / f"{made['id']}.json").read_bytes()
    calls = _fail_first(monkeypatch, os, "replace", 10 ** 9, PermissionError(13, "in use"))
    pauses = []
    monkeypatch.setattr(cal.time, "sleep", pauses.append)
    monkeypatch.setattr(cal, "_RETRY_PAUSE_S", 0.25)
    with pytest.raises(PermissionError):
        cal.set_member(made["id"], FP, 1)
    assert calls["n"] == cal._RETRIES
    assert pauses == [0.25] * (cal._RETRIES - 1)             # a pause between tries, none after the last
    assert (folder / f"{made['id']}.json").read_bytes() == before
    assert [p.name for p in folder.iterdir()] == [f"{made['id']}.json"]


def test_a_write_error_that_is_not_a_busy_file_is_not_retried(monkeypatch, folder):
    made = _make()
    _no_pause(monkeypatch)
    before = (folder / f"{made['id']}.json").read_bytes()
    calls = _fail_first(monkeypatch, os, "replace", 10 ** 9, OSError(28, "no space left"))
    with pytest.raises(OSError):
        cal.add_version(made["id"], pc=(0.5, 0.5, 0.5))
    assert calls["n"] == 1
    assert (folder / f"{made['id']}.json").read_bytes() == before
    assert [p.name for p in folder.iterdir()] == [f"{made['id']}.json"]


def test_a_read_that_meets_a_writer_is_tried_again_and_is_never_reported_broken(monkeypatch):
    made = _make()
    _no_pause(monkeypatch)
    real = Path.read_text
    calls = {"n": 0}

    def busy(self, *args, **kwargs):
        if self.suffix == ".json":
            calls["n"] += 1
            if calls["n"] <= 3:
                raise PermissionError(13, "in use")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", busy)
    assert cal.get(made["id"])["id"] == made["id"]
    calls["n"] = 0
    found, problems = cal.load_all()
    assert [c["id"] for c in found] == [made["id"]] and problems == []
    assert calls["n"] == 4


def test_a_file_deleted_between_the_listing_and_the_read_is_not_a_problem(monkeypatch, folder):
    keep, gone = _make("Keep"), _make("Gone")
    real = cal._read_file

    def vanishing(path):
        if path.stem == gone["id"]:
            raise FileNotFoundError(path.name)
        return real(path)

    monkeypatch.setattr(cal, "_read_file", vanishing)
    found, problems = cal.load_all()
    assert [c["id"] for c in found] == [keep["id"]] and problems == []


def test_a_file_that_stays_busy_is_reported_unreadable_not_raised(monkeypatch):
    made = _make()
    _no_pause(monkeypatch)
    real = Path.read_text

    def busy(self, *args, **kwargs):
        if self.suffix == ".json":
            raise PermissionError(13, "in use")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", busy)
    found, problems = cal.load_all()
    assert found == [] and [(p["file"], p["code"]) for p in problems] == [(f"{made['id']}.json", "unreadable")]
    with pytest.raises(cal.CalibrationError) as exc:
        cal.get(made["id"])
    assert exc.value.code == "unreadable"


def test_a_read_waits_for_a_write_in_progress():
    """In one process a read takes the folder lock too, so it never meets the replace of the same file."""
    made = _make()
    done = threading.Event()

    def read():
        cal.get(made["id"])
        done.set()

    with cal._WRITE_LOCK:
        t = threading.Thread(target=read)
        t.start()
        assert not done.wait(0.3)
    t.join(10)
    assert done.is_set()


# ------------------------------------------------------------------ a file with the right keys and wrong content

def _doc_of(made, **change):
    doc = json.loads(json.dumps(made))
    doc.update(change)
    return doc


def _put(folder, name, doc):
    path = folder / name
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return path


GOOD_VERSION = {"version": 1, "created": "2026-10-14T10:00:00+00:00", "pc": [0.5, 0.4, 0.8], "origin": {}}
GOOD_MEMBER = {"name": "a.h5", "size": 3, "header_id": "aa", "version": 1, "applied": "2026-10-14T10:00:00+00:00"}


@pytest.mark.parametrize("change", [
    {"versions": [1]},
    {"versions": "abc"},
    {"versions": [{**GOOD_VERSION, "version": "x"}]},
    {"versions": [{**GOOD_VERSION, "version": None}]},
    {"features": []},
    {"features": {"pattern_shape": [128, 156], "camera_tilt": "steep"}},
    {"features": {"pattern_shape": [128, 156], "detector_insertion_mm": "far"}},
    {"features": {"pattern_shape": "wide"}},
    {"features": {}},
    {"features": {"pattern_shape": None}},
    {"features": {"pattern_shape": [128, None]}},
    {"features": {"pattern_shape": [128, float("nan")]}},
    {"features": {"pattern_shape": [128, 156, 3]}},
    {"members": {}},
    {"members": [1]},
    {"members": [{**GOOD_MEMBER, "size": "big"}]},
    {"members": [{**GOOD_MEMBER, "size": float("nan")}]},
    {"members": [{**GOOD_MEMBER, "version": None}]},
    {"members": [{**GOOD_MEMBER, "version": float("nan")}]},
    {"members": [{**GOOD_MEMBER, "header_id": 5}]},
    {"members": [{**GOOD_MEMBER, "name": None}]},
    {"name": 5},
    {"created": 5},
    {"schema": "one"},
])
def test_a_file_with_the_right_keys_and_wrong_types_is_reported_not_raised_and_left_alone(folder, change):
    good = _make("The good one")
    bad = _put(folder, "typed-abc123.json", _doc_of(good, id="typed-abc123", **change))
    before = bad.read_bytes()
    found, problems = cal.load_all()
    assert [c["id"] for c in found] == [good["id"]]
    assert [(p["file"], p["code"]) for p in problems] == [("typed-abc123.json", "unreadable")]
    with pytest.raises(cal.CalibrationError) as exc:
        cal.get("typed-abc123")
    assert exc.value.code == "unreadable"
    assert cal.suggest(FEATURES, FP)                       # the list is not lost
    _make("a write beside it")
    assert bad.read_bytes() == before


@pytest.mark.parametrize("versions", [
    [{**GOOD_VERSION, "pc": "abc"}], [{**GOOD_VERSION, "pc": [0.5, 0.4]}], [{**GOOD_VERSION, "pc": [0.5, None, 1]}],
    [], [{**GOOD_VERSION, "version": 2}]])
def test_versions_without_a_pc_or_not_numbered_are_not_a_calibration(folder, versions):
    good = _make("The good one")
    _put(folder, "typed-abc123.json", _doc_of(good, id="typed-abc123", versions=versions))
    found, problems = cal.load_all()
    assert [c["id"] for c in found] == [good["id"]]
    assert [(p["file"], p["code"]) for p in problems] == [("typed-abc123.json", "notACalibration")]
    with pytest.raises(cal.CalibrationError) as exc:
        cal.get("typed-abc123")
    assert exc.value.code == "notACalibration"


def test_a_broken_file_raises_its_coded_error_from_get_and_is_never_overwritten(folder):
    for name, content in (("broken-def456.json", "{ not json"), ("binary-abc123.json", b"\xff\xfe\x00\x01")):
        path = folder / name
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    for cal_id in ("broken-def456", "binary-abc123"):
        with pytest.raises(cal.CalibrationError) as exc:
            cal.get(cal_id)
        assert exc.value.code == "unreadable"
    found, problems = cal.load_all()
    assert found == [] and {p["code"] for p in problems} == {"unreadable"} and len(problems) == 2


def test_a_copy_made_in_the_explorer_is_a_problem_not_a_second_calibration(folder):
    made = _make()
    original = folder / f"{made['id']}.json"
    copy = folder / f"{made['id']} - Kopie.json"
    copy.write_bytes(original.read_bytes())
    found, problems = cal.load_all()
    assert [c["id"] for c in found] == [made["id"]]
    assert [(p["file"], p["code"]) for p in problems] == [(copy.name, "idMismatch")]
    cal.set_member(made["id"], FP, 1)
    assert json.loads(copy.read_text(encoding="utf-8"))["members"] == []          # the copy is untouched
    assert len(cal.get(made["id"])["members"]) == 1
    assert cal.suggest(FEATURES, None)[0]["id"] == made["id"] and len(cal.suggest(FEATURES, None)) == 1


def test_a_file_whose_id_is_not_its_name_is_not_served_by_that_name(folder):
    made = _make()
    _put(folder, "renamed-by-hand.json", _doc_of(made))
    with pytest.raises(cal.CalibrationError) as exc:
        cal.get("renamed-by-hand")
    assert exc.value.code == "idMismatch"
    with pytest.raises(cal.CalibrationError):
        cal.set_member("renamed-by-hand", FP, 1)
    assert json.loads((folder / "renamed-by-hand.json").read_text(encoding="utf-8"))["members"] == []


def test_readers_against_a_writer_see_no_error_no_false_unreadable_and_every_member():
    made = _make()
    stop = threading.Event()
    seen = {"errors": [], "problems": [], "reads": 0}

    def reader(k):
        while not stop.is_set():
            try:
                if k % 2:
                    assert cal.get(made["id"]) is not None
                else:
                    found, problems = cal.load_all()
                    seen["problems"] += problems
                    assert [c["id"] for c in found] == [made["id"]]
                seen["reads"] += 1
            except BaseException as exc:  # noqa: BLE001 - the test counts everything
                seen["errors"].append(repr(exc))

    threads = [threading.Thread(target=reader, args=(k,)) for k in range(3)]
    for t in threads:
        t.start()
    try:
        for i in range(400):
            cal.set_member(made["id"], {"name": f"f{i}.h5", "size": i, "header_id": f"id{i}"}, 1)
    finally:
        stop.set()
        for t in threads:
            t.join(30)
    assert seen["errors"] == [] and seen["problems"] == [] and seen["reads"] > 0
    assert len(cal.get(made["id"])["members"]) == 400


# ------------------------------------------------------------------ edges of the stored values

def test_names_of_a_file_that_differ_only_in_case_are_the_same_file_other_names_are_not():
    assert cal.same_file(FP, {**FP, "name": "SCAN1.H5OINA"}) is True
    assert cal.same_file({**FP, "name": "Scan1.H5oina"}, FP) is True
    assert cal.same_file(FP, {**FP, "name": "scan2.h5oina"}) is False       # another name, same size and header id
    assert cal.same_file(FP, {**FP, "name": "SCAN2.H5OINA"}) is False


def test_a_member_stored_in_one_case_is_replaced_not_duplicated_by_the_same_file_in_another():
    made = _make()
    cal.set_member(made["id"], FP, 1)
    cal.set_member(made["id"], {**FP, "name": "SCAN1.H5OINA"}, 1)
    assert [m["name"] for m in cal.get(made["id"])["members"]] == ["SCAN1.H5OINA"]


@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400), 10 ** 309])
def test_a_whole_number_beyond_a_float_is_no_setting_and_no_pc_component(value):
    assert cal._stated(value) is None
    assert cal._finite(value) is False
    assert cal._valid_pc([value, 0.4, 0.8]) is False
    assert cal._stated(10 ** 300) == 10 ** 300                                # still a float, so still a setting


@pytest.mark.parametrize("name,code", [("n" * 201, "nameTooLong"), ("\ud800x", "badName")])
def test_create_and_rename_refuse_an_unwritable_or_overlong_name_before_writing(folder, name, code):
    with pytest.raises(cal.CalibrationError) as exc:
        _make(name)
    assert exc.value.code == code
    assert not folder.exists() or list(folder.glob("*")) == []
    made = _make()
    before = (folder / f"{made['id']}.json").read_bytes()
    with pytest.raises(cal.CalibrationError) as exc:
        cal.rename(made["id"], name)
    assert exc.value.code == code
    assert (folder / f"{made['id']}.json").read_bytes() == before


def test_an_overlong_name_is_counted_after_spacing_is_normalised():
    ok = " ".join(["x"] * 100)                                               # 199 characters with its spaces
    assert _make(ok)["name"] == ok
    spaced = "y" * 100 + " " * 50 + "z" * 100                               # 201 characters, 201 after normalising
    assert len(" ".join(spaced.split())) == 201
    with pytest.raises(cal.CalibrationError) as exc:
        _make(spaced)
    assert exc.value.code == "nameTooLong"
    assert _make("p" * 100 + " " * 80 + "q" * 99)["name"] == "p" * 100 + " " + "q" * 99   # 280 typed, 200 kept
