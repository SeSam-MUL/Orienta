"""The detector insertion in the "same settings" rule, and the microscope, which is no setting."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibrations as cal  # noqa: E402
from tests.test_pc_calibrations import FEATURES, ORIGIN, SETTINGS, _loaded_as, _make  # noqa: E402


@pytest.fixture(autouse=True)
def folder(tmp_path):
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    yield tmp_path / "Calibrations"
    cal.set_calibrations_dir_for_test(None)


def _oxford_like(insertion=165.65, **metadata):
    """An Oxford .h5oina as kikuchipy leaves it: the header in the original metadata, no file extension in the name."""
    sig = _loaded_as("scan 1")
    sig.original_metadata.set_item("manufacturer", "oxford instruments")
    if insertion is not None:
        sig.original_metadata.set_item("Detector Insertion Distance", insertion)
    for path, value in metadata.items():
        sig.metadata.set_item(path.replace("__", "."), value)
    return sig


def test_the_detector_insertion_is_read_from_the_header_when_the_file_states_it():
    assert cal.features_of_signal(_oxford_like(165.65))["detector_insertion_mm"] == pytest.approx(165.65)
    assert cal.features_of_signal(_oxford_like(None))["detector_insertion_mm"] is None


@pytest.mark.parametrize("name", ["map.ebsp", "scan.up1", "Pattern.dat"])
def test_a_format_that_keeps_no_header_never_states_an_insertion(name):
    sig = _loaded_as(name)
    sig.original_metadata.set_item("Detector Insertion Distance", 165.65)
    f = cal.features_of_signal(sig)
    assert f["detector_insertion_mm"] is None


def test_the_microscope_a_file_names_is_a_hint_and_no_setting():
    f = cal.features_of_signal(_oxford_like(Acquisition_instrument__SEM__microscope="SEM A"))
    assert f["hints"]["microscope"] == "SEM A"
    assert "microscope_detector" not in f
    assert set(f) - {"hints"} == set(cal.FEATURE_KEYS) and "microscope_detector" not in cal.SETTING_KEYS


@pytest.mark.parametrize("other,same", [(165.65, True), (165.74, True), (165.56, True), (165.80, False),
                                        (165.40, False)])
def test_the_insertion_is_compared_within_the_working_distance_tolerance(other, same):
    out = cal.compare_features(FEATURES, {**FEATURES, "detector_insertion_mm": other})
    assert out["match"] is same
    assert [d["key"] for d in out["differences"]] == ([] if same else ["detector_insertion_mm"])
    assert "detector_insertion_mm" in out["checked"]
    if not same:
        assert out["differences"][0]["tolerance"] == cal.TOL_WD_MM


@pytest.mark.parametrize("mine_has,file_has", [(True, False), (False, True), (False, False)])
def test_an_insertion_one_side_does_not_state_is_unknown_never_a_match_or_a_difference(mine_has, file_has):
    mine = FEATURES if mine_has else {**FEATURES, "detector_insertion_mm": None}
    theirs = FEATURES if file_has else {**FEATURES, "detector_insertion_mm": None}
    out = cal.compare_features(mine, theirs)
    assert out["unknown"] == ["detector_insertion_mm"] and out["differences"] == [] and out["match"] is True
    assert out["checked"] == ["pattern_shape", "binning", "camera_tilt", "sample_tilt", "working_distance_mm"]
    assert out["unverified"] is False


def test_the_insertion_is_stored_with_a_calibration_and_read_back_and_a_microscope_is_not_kept():
    made = cal.create(name="insertion", pc=(0.5, 0.4, 0.7), origin=ORIGIN,
                      features={**FEATURES, "microscope_detector": "SEM A / Detector B"})
    got = cal.get(made["id"])["features"]
    assert got["detector_insertion_mm"] == 165.65 and "microscope_detector" not in got
    assert cal.summary(cal.get(made["id"]))["features"]["detector_insertion_mm"] == 165.65


@pytest.mark.parametrize("value", ["SEM A / Detector B", 5, None])
def test_a_calibration_file_that_holds_a_microscope_key_loads_keeps_it_and_never_compares_it(folder, value):
    made = cal.create(name="old microscope key", pc=(0.5, 0.4, 0.7), origin=ORIGIN, features=FEATURES)
    path = folder / f"{made['id']}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["features"]["microscope_detector"] = value
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = cal.get(made["id"])
    assert loaded["features"]["microscope_detector"] == value                  # left as it is in the file
    # neither side's microscope text makes a difference, an unknown or a checked row
    for other in (FEATURES, {**FEATURES, "microscope_detector": "Another SEM"}):
        out = cal.compare_features(loaded["features"], other)
        assert out["match"] is True and out["differences"] == [] and out["unknown"] == []
        assert out["checked"] == ["pattern_shape"] + SETTINGS
    got = cal.suggest({**FEATURES, "microscope_detector": "Another SEM"}, None)
    assert [s["id"] for s in got] == [made["id"]] and got[0]["match"] is True
    assert got[0]["checked"] == ["pattern_shape"] + SETTINGS


def test_a_calibration_file_written_before_the_insertion_existed_still_loads_and_reads_it_as_none(folder):
    old = {k: v for k, v in FEATURES.items() if k != "detector_insertion_mm"}
    made = cal.create(name="old key set", pc=(0.5, 0.4, 0.7), origin=ORIGIN, features=FEATURES)
    path = folder / f"{made['id']}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["features"] = old
    assert doc["schema"] == 1                                    # no schema bump
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = cal.get(made["id"])
    assert "detector_insertion_mm" not in loaded["features"]
    out = cal.compare_features(loaded["features"], FEATURES)
    assert out["match"] is True and out["unknown"] == ["detector_insertion_mm"]
    assert out["checked"] == ["pattern_shape", "binning", "camera_tilt", "sample_tilt", "working_distance_mm"]
    got = cal.suggest(FEATURES, None)
    assert [s["id"] for s in got] == [made["id"]] and got[0]["match"] is True


def test_a_suggestion_says_what_was_checked_including_the_pattern_shape():
    _make()
    got = cal.suggest(FEATURES, None)
    assert got[0]["checked"] == ["pattern_shape"] + SETTINGS      # "all 6 settings same"
    assert got[0]["match"] is True and got[0]["unknown"] == []


# ----- real files: what the headers state (ORIENTA_TEST_DATA = the Test_data folder)

def _real(name):
    base = Path(os.environ.get("ORIENTA_TEST_DATA") or Path(__file__).resolve().parents[1] / "Test_data")
    path = base / name
    if not path.exists():
        pytest.skip(f"{name} is not in {base} (set ORIENTA_TEST_DATA to the folder that holds the test data)")
    return path


def test_a_real_oxford_file_states_its_insertion():
    import h5py
    from safe_loader import load_ebsd_safe
    path = _real("EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina")
    with h5py.File(path, "r") as handle:                    # the expected value: the file's own header, read directly
        stated = float(handle["1/EBSD/Header/Detector Insertion Distance"][0])
    sig = load_ebsd_safe(str(path), verbose=False)
    f = cal.features_of_signal(sig)
    assert stated == pytest.approx(165.6486, abs=1e-3)
    assert f["detector_insertion_mm"] == pytest.approx(stated, abs=1e-6)


def test_a_real_edax_file_states_no_insertion():
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(str(_real("HiGainNi.h5")), verbose=False)
    f = cal.features_of_signal(sig)
    assert f["detector_insertion_mm"] is None
    assert f["camera_tilt"] == pytest.approx(10.0)            # the settings it does state are still read


# ----- the date of an Oxford file: the reader leaves it in the original metadata only

def test_an_oxford_file_states_its_date_in_the_header_and_only_the_day_is_kept():
    sig = _oxford_like()
    sig.original_metadata.set_item("Acquisition Date", "2026-01-19T09:04:03")
    assert cal.features_of_signal(sig)["hints"]["date"] == "2026-01-19"
    made = cal.create(name="dated", pc=(0.5, 0.4, 0.7), origin=ORIGIN, features=cal.features_of_signal(sig))
    offer = cal.suggest(cal.features_of_signal(sig), None)
    assert [s["id"] for s in offer] == [made["id"]] and offer[0]["file_date"] == "2026-01-19"


def test_the_date_a_reader_put_in_the_metadata_wins_over_the_header():
    sig = _oxford_like(General__date="2025-12-24")
    sig.original_metadata.set_item("Acquisition Date", "2026-01-19T09:04:03")
    assert cal.features_of_signal(sig)["hints"]["date"] == "2025-12-24"


@pytest.mark.parametrize("stated", [None, "", "yesterday", "19.01.2026 09:04", 20260119,
                                    "2026-13-45T00:00", "2026-00-10T08:00:00", "2026-02-30T10:00:00",
                                    "2025-02-29T10:00:00", "2026-04-31 08:00"])
def test_an_oxford_header_without_a_readable_date_gives_none(stated):
    sig = _oxford_like()
    if stated is not None:
        sig.original_metadata.set_item("Acquisition Date", stated)
    assert "date" not in cal.features_of_signal(sig)["hints"]


def test_the_29th_of_february_of_a_leap_year_is_a_date():
    sig = _oxford_like()
    sig.original_metadata.set_item("Acquisition Date", "2024-02-29T08:00:00")
    assert cal.features_of_signal(sig)["hints"]["date"] == "2024-02-29"


def test_only_an_oxford_header_is_read_for_the_date():
    sig = _loaded_as("scan 1")
    sig.original_metadata.set_item("manufacturer", "edax")
    sig.original_metadata.set_item("Acquisition Date", "2026-01-19T09:04:03")
    assert "date" not in cal.features_of_signal(sig)["hints"]


def test_a_real_oxford_file_has_its_acquisition_day_and_a_real_edax_file_none():
    import h5py
    from safe_loader import load_ebsd_safe
    path = _real("EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina")
    with h5py.File(path, "r") as handle:             # the expected value: the file's own header, read directly
        stated = handle["1/EBSD/Header/Acquisition Date"][0].decode()
    assert stated[:10].count("-") == 2 and len(stated) > 10
    got = cal.features_of_signal(load_ebsd_safe(str(path), verbose=False))
    assert got["hints"]["date"] == stated[:10]
    edax = cal.features_of_signal(load_ebsd_safe(str(_real("HiGainNi.h5")), verbose=False))
    assert "date" not in edax["hints"]
