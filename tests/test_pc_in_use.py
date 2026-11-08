"""Where the PC in use came from (``pc_in_use_of``) and how good it is (``pc_quality_of``)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kikuchipy.detectors import EBSDDetector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402
from backend.api.services import pc_in_use  # noqa: E402
from backend.api.services.calibration_store import CalibrationStore  # noqa: E402

SHAPE = (60, 60)
HEADER_PC = (0.5, 0.4, 0.6)
FOUND = (0.4301, 0.2141, 0.5009)
FIND = {"n_patterns": 9, "method": "PSO", "mode": "find"}
REFINE = {"n_patterns": 9, "method": "Nelder-Mead", "mode": "refine"}


@pytest.fixture(autouse=True)
def clean(tmp_path):
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    use.forget_all()
    yield
    use.forget_all()
    cal.set_calibrations_dir_for_test(None)


def _signal(pc=HEADER_PC, source=None):
    return SimpleNamespace(
        detector=EBSDDetector(shape=SHAPE, pc=pc, sample_tilt=70.0, tilt=4.0),
        metadata=SimpleNamespace(get_item=lambda key, default=None: (source if key == "Signal.pc_source"
                                                                       and source else default)))


@pytest.fixture
def store(tmp_path):
    st = CalibrationStore()
    path = tmp_path / "scan.h5oina"
    path.write_bytes(b"scan" * 50)
    st.register("scan", _signal(), file_key=str(path))
    return st


def _where(store, name="scan"):
    return pc_in_use.pc_in_use_of(name, store)


# ------------------------------------------------------------------ the kind, from the store entry

def test_a_dataset_the_store_does_not_know_has_no_pc_in_use(store):
    got = pc_in_use.pc_in_use_of("nope", store)
    assert (got["pc"], got["kind"], got["calibration"], got["per_pixel"]) == (None, "none", None, False)
    assert pc_in_use.pc_in_use_of(None, store)["kind"] == "none"


def test_a_file_that_carries_no_pc_is_none(tmp_path):
    st = CalibrationStore()
    st.register("nopc", _signal(source="default"))
    got = pc_in_use.pc_in_use_of("nopc", st)
    assert got["kind"] == "none" and got["pc"] is None and got["calibration"] is None


def test_the_header_pc_is_the_file(store):
    got = _where(store)
    assert got["kind"] == "file" and got["pc"] == pytest.approx(HEADER_PC) and got["per_pixel"] is False
    assert got["calibration"] is None


@pytest.mark.parametrize("record,kind", [
    (FIND, "find"), (REFINE, "refine"),
    ({"n_patterns": 9, "method": "PSO"}, "refine"),                    # written before `mode` existed
    ({"n_patterns": 9, "method": "PSO", "mode": "something else"}, "refine"),
])
def test_a_searched_pc_is_told_apart_by_the_mode_of_its_record(store, record, kind):
    store.update_pc("scan", FOUND, source="refined", refinement=record)
    got = _where(store)
    assert got["kind"] == kind and got["pc"] == pytest.approx(FOUND)


def test_a_typed_pc_is_typed(store):
    store.update_pc("scan", FOUND, source="manual")
    assert _where(store)["kind"] == "typed"


def test_a_per_pixel_map_of_the_file_is_the_file_and_the_pc_is_its_mean(store):
    pc_map = np.stack([np.full((2, 3), v) for v in (0.4, 0.3, 0.8)], axis=-1)       # (2, 3, 3) constant planes
    pc_map[0, 0] = (0.5, 0.3, 0.8)                                                  # mean x = (0.5 + 5 * 0.4) / 6
    store.update_pc_map("scan", pc_map, source="header")
    got = _where(store)
    assert got["kind"] == "file" and got["per_pixel"] is True
    assert got["pc"] == pytest.approx([(0.5 + 5 * 0.4) / 6, 0.3, 0.8])


def test_a_grid_fitted_map_is_grid(store):
    pc_map = np.full((2, 3, 3), 0.5)
    store.update_pc_map("scan", pc_map, source="refined_map")
    got = _where(store)
    assert got["kind"] == "grid" and got["per_pixel"] is True


# ------------------------------------------------------------------ a PC passed on to other datasets

def test_a_find_on_a_copy_is_a_find_on_the_parent_too(store):
    from backend.api.routes import pcrefinement
    store.register_derived("scan_copy", "scan")
    pcrefinement._writeback_refined_pc("scan_copy", FOUND, store=store, refinement=dict(FIND))
    assert _where(store, "scan_copy")["kind"] == "find" and _where(store, "scan")["kind"] == "find"
    assert _where(store, "scan")["pc"] == pytest.approx(FOUND)


def test_a_typed_pc_on_a_copy_is_typed_on_the_parent_too(store):
    from backend.api.routes import pcrefinement
    store.register_derived("scan_copy", "scan")
    pcrefinement._writeback_refined_pc("scan_copy", FOUND, store=store, source="manual")
    assert store.get_entry("scan").pc_source == "propagated"
    assert _where(store, "scan")["kind"] == "typed" and _where(store, "scan_copy")["kind"] == "typed"


def test_a_copy_takes_the_kind_of_its_parent(store):
    store.register_derived("from_header", "scan")
    assert _where(store, "from_header")["kind"] == "file"
    store.update_pc("scan", FOUND, source="manual")
    store.register_derived("from_typed", "scan")
    assert store.get_entry("from_typed").pc_source == "inherited"
    assert _where(store, "from_typed")["kind"] == "typed"
    store.update_pc("scan", FOUND, source="refined", refinement=dict(FIND))
    store.register_derived("from_find", "scan")
    assert _where(store, "from_find")["kind"] == "find"
    store.register_derived("copy_of_copy", "from_find")
    assert _where(store, "copy_of_copy")["kind"] == "find"


def test_a_copy_of_a_file_without_a_pc_is_none(tmp_path):
    st = CalibrationStore()
    st.register("nopc", _signal(source="default"))
    st.register_derived("nopc_copy", "nopc")
    assert pc_in_use.pc_in_use_of("nopc_copy", st)["kind"] == "none"


# ------------------------------------------------------------------ a calibration

def _calibration(store, name="REM 2026-10-14"):
    return cal.create(name=name, pc=FOUND, features=cal.features_of_signal(_signal()),
                      origin={"method": "find", "refinement": dict(FIND)})


def test_a_dataset_on_a_calibration_says_which_and_which_version(store):
    made = _calibration(store)
    use.apply(made["id"], ["scan"], accept_mismatch=["scan"], store=store)
    got = _where(store)
    assert got["kind"] == "calibration"
    assert got["calibration"] == {"id": made["id"], "name": "REM 2026-10-14", "version": 1}
    # a copy of the file runs on it too
    store.register_derived("scan_copy", "scan")
    assert _where(store, "scan_copy")["kind"] == "calibration"


def test_a_pc_typed_after_the_calibration_is_typed_and_names_no_calibration(store):
    made = _calibration(store)
    use.apply(made["id"], ["scan"], accept_mismatch=["scan"], store=store)
    store.update_pc("scan", (0.45, 0.25, 0.55), source="manual")
    got = _where(store)
    assert got["kind"] == "typed" and got["calibration"] is None


def test_a_calibration_that_was_deleted_leaves_a_calibration_pc_without_a_name(store):
    made = _calibration(store)
    use.apply(made["id"], ["scan"], accept_mismatch=["scan"], store=store)
    cal.delete(made["id"])
    got = _where(store)
    assert got["kind"] == "calibration" and got["calibration"] is None and got["pc"] == pytest.approx(FOUND)


# ------------------------------------------------------------------ the quality of the PC in use

SEARCH = {"pc": list(FOUND), "mode": "find", "n_patterns": 9, "on_edge": [],
          "confidence": {"contrast": 0.8, "typical_cost": 8.0, "reasons": []}}
GOOD = {"contrast": 0.8, "warning": None, "axes": [], "n_patterns": 9}


def test_the_quality_of_the_search_that_made_the_pc(store):
    store.update_pc("scan", FOUND, source="refined", refinement=dict(FIND))
    assert pc_in_use.pc_quality_of("scan", store) is None                 # no search was noted
    use.note_search("scan", SEARCH, store)
    assert pc_in_use.pc_quality_of("scan", store) == GOOD


def test_a_search_that_did_not_make_the_pc_in_use_gives_no_quality(store):
    use.note_search("scan", SEARCH, store)
    store.update_pc("scan", (0.45, 0.25, 0.55), source="manual")           # typed afterwards
    assert pc_in_use.pc_quality_of("scan", store) is None
    assert pc_in_use.pc_quality_of("nope", store) is None


def test_the_quality_stored_with_the_calibration_version_in_use(store):
    quality = {"contrast": 0.7, "warning": "edge", "axes": ["z"], "n_patterns": 5}
    made = cal.create(name="With quality", pc=FOUND, features=cal.features_of_signal(_signal()),
                      origin={"method": "find", "quality": quality})
    use.apply(made["id"], ["scan"], accept_mismatch=["scan"], store=store)
    assert pc_in_use.pc_quality_of("scan", store) == quality


def test_a_search_noted_on_a_copy_is_the_search_of_the_raw_file_and_of_every_copy(store):
    from backend.api.routes import pcrefinement
    store.register_derived("scan_copy", "scan")
    store.register_derived("copy_of_copy", "scan_copy")
    pcrefinement._writeback_refined_pc("copy_of_copy", FOUND, store=store, refinement=dict(FIND))
    use.note_search("copy_of_copy", SEARCH, store)
    for name in ("scan", "scan_copy", "copy_of_copy"):
        assert pc_in_use.pc_quality_of(name, store) == GOOD, name
        assert use.search_of(name, store)["pc"] == list(FOUND), name
