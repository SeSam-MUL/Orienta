"""What the APEX measurement dialog needs from the backend:

* per measurement: can it be opened, and if not, why (``loadable`` / ``problem``);
* per pattern file: its pattern count and size, how it was paired, too few or too many patterns;
* the pattern files of the folder that name no measurement, and the same listing for another folder;
* when a measurement was recorded.

Every expectation is written out by hand; the files are the synthetic ones of ``apex_helpers``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edax_apex as apex  # noqa: E402
from backend.api.problem import CODE_HEADER  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_A_BYTES, GUID_B, GUID_B_BYTES, make_apex_h5, make_up, measurement_spec, wipe_backend,
)

#: A GUID that no measurement of these files has; all sixteen bytes alike, so the byte order of the
#: header cannot matter.
GUID_X = "11111111-1111-1111-1111-111111111111"
GUID_X_BYTES = bytes.fromhex("11" * 16)

# 2001-02-03T04:05:06 as .NET ticks, worked out by hand: 2001-02-03 is day 730518 after 0001-01-01
# (2000-01-01 is day 730119, the year 2000 has 366 days, then 31 + 2 more), 730518 * 86400 +
# 4 * 3600 + 5 * 60 + 6 = 63116769906 s, times 10^7.
TICKS_2001_02_03_040506 = 631167699060000000


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


def _a(**over):
    return measurement_spec(id=GUID_A, area="Area 1", n_rows=3, n_cols=5, **over)


def _b(**over):
    return measurement_spec(id=GUID_B, area="Area 2", grid="SqrGrid", n_rows=2, n_cols=4, step_x=1.0, **over)


def _inspect(client, path):
    r = client.get("/api/ebsd/inspect", params={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _by_id(out):
    return {m["id"]: m for m in out["measurements"]}


# ------------------------------------------------------------------- 1. loadable / problem

def test_a_measurement_with_a_fitting_pattern_file_is_loadable(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    a = _by_id(_inspect(client, h5))[GUID_A]
    assert (a["loadable"], a["problem"], a["measurement_problem"]) == (True, None, None)


def test_a_measurement_without_a_pattern_file_says_so_but_is_not_broken(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    a = _by_id(_inspect(client, h5))[GUID_A]
    assert (a["loadable"], a["problem"], a["measurement_problem"]) == (False, "apexNoPatternFile", None)
    assert a["pattern_files"] == []


@pytest.mark.parametrize("over, code, flag", [
    (dict(grid="TriGrid"), "apexUnknownGrid", 0),
    (dict(step_y=0.5), "apexUnexpectedLattice", 1),               # a "HexGrid" whose rows are not sqrt(3)/2 apart
    (dict(n_table=14), "apexGridMismatch", 1),                    # a table with one record too few for 3 x 5
])
def test_a_measurement_the_load_would_refuse_is_not_loadable_whatever_pattern_file_it_has(
        client, tmp_path, over, code, flag):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(**over)])
    make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES, hex_flag=flag)
    a = _by_id(_inspect(client, h5))[GUID_A]
    assert (a["loadable"], a["problem"], a["measurement_problem"]) == (False, code, code)
    assert [f["matches_grid"] for f in a["pattern_files"]] == [True]     # the file is fine; the measurement is not


def test_a_square_measurement_with_a_step_of_zero_is_not_loadable(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_b()])
    with h5py.File(h5, "r+") as f:
        del f["TOP/Sample/Area 2/OIM Map 1/Sample/Step X"]
        f["TOP/Sample/Area 2/OIM Map 1/Sample"].create_dataset("Step X", data=np.array([0.0], dtype=np.float32))
    b = _by_id(_inspect(client, h5))[GUID_B]
    assert (b["loadable"], b["problem"]) == (False, "apexInvalidGrid")


def test_the_table_that_does_not_fit_the_grid_is_flagged_consistent_false(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(n_table=14)])
    assert _by_id(_inspect(client, h5))[GUID_A]["consistent"] is False


def test_a_pattern_file_whose_header_contradicts_the_grid_type_is_not_usable(client, tmp_path):
    """The load refuses a v3+ header whose hexagonal flag disagrees with the measurement; so must the
    listing, or a ticked row fails at Open."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])                      # HexGrid
    make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES, hex_flag=0, version=4)
    a = _by_id(_inspect(client, h5))[GUID_A]
    f = a["pattern_files"][0]
    assert (f["matches_grid"], f["problem"], f["count"]) == (False, "apexGridMismatch", "fits")
    assert (a["loadable"], a["problem"], a["measurement_problem"]) == (False, "apexGridMismatch", None)


def test_a_version_1_header_carries_no_hexagonal_flag_and_is_not_judged_on_one(client, tmp_path):
    """A version 1 header has no grid fields at all (the helper writes zeros there); the same bytes
    in a version 4 header are a flag saying "square", which a hexagonal measurement refuses."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])                       # HexGrid
    v1 = make_up(tmp_path / "v1.up1", n=15, guid_bytes=None, version=1, hex_flag=0, nx=5, ny=3)
    v4 = make_up(tmp_path / "v4.up1", n=15, guid_bytes=None, version=4, hex_flag=0, nx=5, ny=3)
    d1 = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, v1)).json()
    d4 = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, v4)).json()
    assert (d1["matches_grid"], d1["problem"]) == (True, None)
    assert (d4["matches_grid"], d4["problem"]) == (False, "apexGridMismatch")
    # neither names a measurement, so both are listed as unmatched; both have the right count, but
    # only the v1 file is offered for the measurement: Open would refuse the v4 file's flag
    out = _inspect(client, h5)
    assert [(u["name"], u["fits"]) for u in out["unmatched_pattern_files"]] == [
        ("v1.up1", [GUID_A]), ("v4.up1", [])]


def test_a_measurement_is_loadable_when_one_of_its_files_is_usable(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "a_short.up1", n=14, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    make_up(tmp_path / "b_good.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    a = _by_id(_inspect(client, h5))[GUID_A]
    assert [(f["name"], f["matches_grid"]) for f in a["pattern_files"]] == [
        ("a_short.up1", False), ("b_good.up1", True)]
    assert (a["loadable"], a["problem"]) == (True, None)


def test_when_no_file_is_usable_the_problem_is_the_first_files(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "a_short.up1", n=14, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    make_up(tmp_path / "b_flag.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES, hex_flag=0)
    a = _by_id(_inspect(client, h5))[GUID_A]
    assert (a["loadable"], a["problem"]) == (False, "apexPatternCountMismatch")


# ------------------------------------------------- 1b. one broken group does not fail the listing

def _break_camera(h5, area):
    with h5py.File(h5, "r+") as f:
        del f[f"TOP/Sample/{area}/OIM Map 1/Camera"]


def test_one_unreadable_measurement_is_listed_with_its_problem_and_the_others_are_not_lost(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    make_up(tmp_path / "b.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=GUID_B_BYTES)
    _break_camera(h5, "Area 1")
    out = _inspect(client, h5)
    a, b = out["measurements"]
    assert (a["area"], a["name"], a["id"]) == ("Area 1", "OIM Map 1", GUID_A)
    assert (a["loadable"], a["problem"], a["measurement_problem"]) == (False, "apexUnreadable", "apexUnreadable")
    assert a["pattern_files"] == [] and a["grid_type"] is None and a["phases"] == []
    assert (b["id"], b["loadable"], b["problem"]) == (GUID_B, True, None)
    assert set(a) == set(b)                                       # the same keys on every entry


def test_a_group_whose_id_is_no_guid_is_listed_without_an_id(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    with h5py.File(h5, "r+") as f:
        g = f["TOP/Sample/Area 1/OIM Map 1"]
        del g.attrs["Id"]
        g.attrs.create("Id", np.array(["not-a-guid"], dtype=object), dtype=h5py.string_dtype())
    a, b = _inspect(client, h5)["measurements"]
    assert (a["id"], a["loadable"], a["problem"]) == (None, False, "apexBadId")
    assert b["id"] == GUID_B


def test_the_good_measurement_next_to_a_broken_one_can_still_be_loaded(client, tmp_path):
    """The listing says it is fine; opening it must agree (the strict reader refused the whole file)."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    up = make_up(tmp_path / "b.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=GUID_B_BYTES)
    _break_camera(h5, "Area 1")
    r = client.post("/api/ebsd/load", json={"path": str(up),
                                            "apex": {"h5_path": str(h5), "measurement_id": GUID_B}})
    assert r.status_code == 200, r.text
    assert r.json()["apex"]["measurement_id"] == GUID_B
    assert apex.read_measurement(h5, GUID_B).id == GUID_B
    with pytest.raises(apex.ApexError) as ei:                      # the broken one itself is refused by its code
        apex.read_measurement(h5, GUID_A)
    assert ei.value.code == "apexUnreadable"
    with pytest.raises(apex.ApexError) as ei2:                     # the strict listing stays strict
        apex.list_measurements(h5)
    assert ei2.value.code == "apexUnreadable"


# ----------------------------------------------------------- 2. what a pattern file shows

def test_a_listed_pattern_file_carries_its_count_size_and_pairing(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES, sx=8, sy=6)
    f = _by_id(_inspect(client, h5))[GUID_A]["pattern_files"][0]
    assert f["n_patterns"] == 15
    assert f["pattern_shape"] == [6, 8]                            # [rows, cols] = [sy, sx]
    assert (f["pairing"], f["count"], f["matches_grid"], f["problem"]) == ("guid", "fits", True, None)
    assert f["name"] == "p.up1"


@pytest.mark.parametrize("n, relation", [(14, "too_few"), (15, "fits"), (16, "too_many"), (30, "too_many")])
def test_the_count_says_too_few_or_too_many(client, tmp_path, n, relation):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "p.up1", n=n, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    f = _by_id(_inspect(client, h5))[GUID_A]["pattern_files"][0]
    assert (f["n_patterns"], f["count"]) == (n, relation)
    assert f["matches_grid"] is (relation == "fits")
    assert f["problem"] == (None if relation == "fits" else "apexPatternCountMismatch")


def test_a_file_that_names_another_measurement_stays_out_of_this_ones_list(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    make_up(tmp_path / "for_b.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=GUID_B_BYTES)
    by = _by_id(_inspect(client, h5))
    assert by[GUID_A]["pattern_files"] == []
    assert [f["name"] for f in by[GUID_B]["pattern_files"]] == ["for_b.up1"]


def test_a_file_that_cannot_be_read_as_a_pattern_file_is_listed_with_its_code(client, tmp_path):
    """Its header names the measurement (the GUID is readable) but the body-side check fails: here a
    version 2 header, which kikuchipy refuses."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES, version=2)
    f = _by_id(_inspect(client, h5))[GUID_A]["pattern_files"][0]
    assert (f["problem"], f["matches_grid"], f["n_patterns"], f["pattern_shape"], f["count"]) == (
        "apexPatternFileUnreadable", False, None, None, None)


# ---------------------------------------------- 3. pattern files that name no measurement

def test_pattern_files_without_a_guid_are_listed_with_the_measurements_their_count_fits(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    make_up(tmp_path / "no_guid_v3.up1", n=15, nx=5, ny=3, guid_bytes=None, version=3, sx=8, sy=6)
    make_up(tmp_path / "nil_guid.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=bytes(16), sx=4, sy=4)
    make_up(tmp_path / "fits_nothing.up1", n=9, nx=3, ny=3, guid_bytes=None, version=3)
    out = _inspect(client, h5)
    assert [(u["name"], u["n_patterns"], u["pattern_shape"], u["fits"], u["problem"])
            for u in out["unmatched_pattern_files"]] == [
        ("fits_nothing.up1", 9, [8, 8], [], None),
        ("nil_guid.up1", 8, [4, 4], [GUID_B], None),
        ("no_guid_v3.up1", 15, [6, 8], [GUID_A], None)]
    assert all(m["pattern_files"] == [] for m in out["measurements"])        # nothing is paired from them


def test_a_file_naming_a_measurement_of_another_session_is_in_no_list(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "other_session.up1", n=15, nx=5, ny=3, guid_bytes=GUID_X_BYTES)
    out = _inspect(client, h5)
    assert out["unmatched_pattern_files"] == []
    assert out["measurements"][0]["pattern_files"] == []


def test_an_unreadable_file_without_a_guid_is_listed_with_its_code_and_fits_nothing(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    (tmp_path / "stub.up1").write_bytes(b"\x01\x00\x00\x00")             # 4 bytes: no header
    (tmp_path / "notes.txt").write_text("not a pattern file")
    out = _inspect(client, h5)
    assert out["unmatched_pattern_files"] == [{
        "path": str(tmp_path / "stub.up1"), "name": "stub.up1", "n_patterns": None,
        "pattern_shape": None, "problem": "apexPatternFileUnreadable", "fits": [],
        "names_measurement": None, "names_group": None}]


def test_a_second_pattern_file_for_a_measurement_is_listed_not_picked(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "one.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    make_up(tmp_path / "two.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    a = _by_id(_inspect(client, h5))[GUID_A]
    assert [f["name"] for f in a["pattern_files"]] == ["one.up1", "two.up1"]
    assert a["loadable"] is True


# ----------------------------------------------------------- 4. another folder

def _pattern_files(client, h5, folder):
    return client.get("/api/ebsd/apex/pattern-files", params={"h5_path": str(h5), "folder": str(folder)})


def test_another_folder_is_listed_like_the_metadata_files_own(client, tmp_path):
    meta = tmp_path / "meta"
    scans = tmp_path / "scans"
    meta.mkdir()
    scans.mkdir()
    h5 = make_apex_h5(meta / "s.h5", [_a(), _b()])
    make_up(scans / "a.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    make_up(scans / "loose.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=None, version=3)
    assert all(m["pattern_files"] == [] for m in _inspect(client, h5)["measurements"])    # the metadata folder has none
    r = _pattern_files(client, h5, scans)
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["h5_path"], out["folder"]) == (str(h5), str(scans))
    by = _by_id(out)
    assert [f["name"] for f in by[GUID_A]["pattern_files"]] == ["a.up1"]
    assert (by[GUID_A]["loadable"], by[GUID_B]["loadable"], by[GUID_B]["problem"]) == (
        True, False, "apexNoPatternFile")
    assert [(u["name"], u["fits"]) for u in out["unmatched_pattern_files"]] == [("loose.up1", [GUID_B])]
    assert set(by[GUID_A]) == set(_by_id(_inspect(client, h5))[GUID_A])          # the same entry as /inspect's


def test_a_folder_that_cannot_be_listed_is_a_coded_refusal_not_an_empty_list(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    for folder in (tmp_path / "missing_folder", h5):                       # does not exist; is a file
        r = _pattern_files(client, h5, folder)
        assert r.status_code == 400, r.text
        assert r.headers[CODE_HEADER.lower()] == "apexFolderUnreadable"
        assert str(tmp_path) not in r.json()["detail"]                    # the folder's name, never its path


def test_the_other_folder_route_refuses_a_nul_and_a_file_that_is_no_metadata_file(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    for params in ({"h5_path": str(h5) + "\x00", "folder": str(tmp_path)},
                   {"h5_path": str(h5), "folder": str(tmp_path) + "\x00"}):
        r = client.get("/api/ebsd/apex/pattern-files", params=params)
        assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexBadPath"
    plain = tmp_path / "plain.h5"
    with h5py.File(plain, "w") as f:
        f.create_group("1/EBSD/Data")
    r = _pattern_files(client, plain, tmp_path)
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexNoMeasurement"   # was apexUnreadable


# ------------------------- the other-folder route says what /inspect says about the metadata file

def test_the_other_folder_route_gives_the_codes_inspect_gives_for_the_same_file(client, tmp_path):
    missing = tmp_path / "missing.h5"
    garbage = tmp_path / "garbage.h5"
    garbage.write_bytes(b"this is not an HDF5 file" * 20)
    # /inspect: 400 apexFileNotFound for the missing file; 200 kind "unreadable" (apexFileUnreadable)
    # for the garbage one. The other-folder route needs a metadata file, so the second is a coded 400.
    i = client.get("/api/ebsd/inspect", params={"path": str(missing)})
    p = _pattern_files(client, missing, tmp_path)
    assert (i.status_code, i.headers[CODE_HEADER.lower()]) == (p.status_code, p.headers[CODE_HEADER.lower()]) == (
        400, "apexFileNotFound")
    assert _inspect(client, garbage)["problem"] == "apexFileUnreadable"
    g = _pattern_files(client, garbage, tmp_path)
    assert g.status_code == 400 and g.headers[CODE_HEADER.lower()] == "apexFileUnreadable"


def test_a_folder_named_like_an_h5_file_is_a_correct_coded_answer_not_does_not_exist(client, tmp_path):
    folder = tmp_path / "x.h5"
    folder.mkdir()
    for r in (client.get("/api/ebsd/inspect", params={"path": str(folder)}),
              _pattern_files(client, folder, tmp_path)):
        assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexFileUnreadable"
        assert "folder" in r.json()["detail"] and "does not exist" not in r.json()["detail"]
        assert str(tmp_path) not in r.json()["detail"]


def test_a_path_the_os_will_not_let_us_look_at_is_a_coded_400_not_a_500(client, tmp_path, monkeypatch):
    locked = make_apex_h5(tmp_path / "locked.h5", [_a()])
    real_is_dir, real_is_file = Path.is_dir, Path.is_file

    def deny(real):
        def check(self, *args, **kwargs):
            if self.name == "locked.h5":
                raise PermissionError(13, "Permission denied")
            return real(self, *args, **kwargs)
        return check

    with monkeypatch.context() as patched:
        patched.setattr(Path, "is_dir", deny(real_is_dir))
        patched.setattr(Path, "is_file", deny(real_is_file))
        for r in (client.get("/api/ebsd/inspect", params={"path": str(locked)}),
                  _pattern_files(client, locked, tmp_path)):
            assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexFileUnreadable"
            assert "PermissionError" in r.json()["detail"] and str(tmp_path) not in r.json()["detail"]
    assert _inspect(client, locked)["kind"] == "apex"


def test_inspect_says_so_when_its_own_folder_cannot_be_listed(client, tmp_path, monkeypatch):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])

    def refuse(folder):
        raise apex.ApexError("apexFolderUnreadable", "x cannot be listed (PermissionError)")

    with monkeypatch.context() as patched:
        patched.setattr(apex, "pattern_files_in", refuse)
        out = _inspect(client, h5)
    assert out["folder_problem"] == "apexFolderUnreadable"
    assert out["unmatched_pattern_files"] == []
    a = out["measurements"][0]
    # the measurement is still listed, and it does not claim that no pattern file exists: nobody looked
    assert (a["loadable"], a["problem"], a["measurement_problem"]) == (False, "apexFolderUnreadable", None)
    assert _inspect(client, h5)["folder_problem"] is None


# ------------------------------------------------- the chosen file, as data (describe + check)

def _body(h5, up, mid=GUID_A):
    return {"h5_path": str(h5), "measurement_id": mid, "pattern_path": str(up)}


def test_a_chosen_file_with_too_many_patterns_is_described_not_refused(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "x.up1", n=16, nx=5, ny=3, guid_bytes=GUID_A_BYTES, sx=8, sy=6)
    r = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, up))
    assert r.status_code == 200, r.text
    d = r.json()
    assert (d["n_patterns"], d["count"], d["problem"], d["matches_grid"], d["pairing"],
            d["pattern_shape"]) == (16, "too_many", "apexPatternCountMismatch", False, "guid", [6, 8])
    # the check-pairing route keeps refusing it with the code
    bad = client.post("/api/ebsd/apex/check-pairing", json=_body(h5, up))
    assert bad.status_code == 400 and bad.headers[CODE_HEADER.lower()] == "apexPatternCountMismatch"


def test_a_chosen_file_of_another_measurement_is_described_with_that_code(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    up = make_up(tmp_path / "x.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=GUID_B_BYTES)
    d = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, up, GUID_A)).json()
    assert (d["problem"], d["pairing"], d["matches_grid"]) == ("apexGuidMismatch", "guid", False)


def test_a_chosen_file_without_a_guid_is_paired_by_count(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "x.up1", n=15, nx=5, ny=3, guid_bytes=None, version=3)
    d = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, up)).json()
    assert (d["pairing"], d["count"], d["problem"], d["matches_grid"]) == ("count", "fits", None, True)
    ok = client.post("/api/ebsd/apex/check-pairing", json=_body(h5, up)).json()
    assert ok["ok"] is True and ok["pairing"] == "count"
    assert ok["pattern_shape"] == [8, 8] and ok["n_patterns"] == 15


def test_a_file_that_is_no_pattern_file_is_described_without_being_read(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"secret text " * 20)
    d = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, notes)).json()
    assert d["problem"] == "apexPatternFileUnreadable"
    assert (d["pairing"], d["count"], d["n_patterns"], d["matches_grid"]) == (None, None, None, False)
    assert "secret" not in str(d)


def test_the_describe_route_refuses_a_nul_and_an_unknown_measurement(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "x.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    r = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, str(up) + "\x00"))
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexBadPath"
    r = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, up, GUID_X))
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexMeasurementNotFound"


def test_check_pairing_refuses_a_header_that_contradicts_the_grid_type(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])                           # HexGrid
    up = make_up(tmp_path / "x.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES, hex_flag=0)
    r = client.post("/api/ebsd/apex/check-pairing", json=_body(h5, up))
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexGridMismatch"
    # ... which is what the load says to the same file
    load = client.post("/api/ebsd/load", json={
        "path": str(up), "apex": {"h5_path": str(h5), "measurement_id": GUID_A}})
    assert load.status_code == 400 and load.headers[CODE_HEADER.lower()] == "apexGridMismatch"


# ---------------------------------------------------------------------- 5. when it was measured

def test_ticks_become_iso_text_without_a_zone():
    assert apex.dotnet_ticks_to_iso(np.array([TICKS_2001_02_03_040506], dtype=np.int64)) == "2001-02-03T04:05:06"
    assert apex.dotnet_ticks_to_iso(TICKS_2001_02_03_040506 + 9_999_999) == "2001-02-03T04:05:06"   # under a second is dropped


@pytest.mark.parametrize("value", [
    np.array([float(TICKS_2001_02_03_040506)]),           # a float is not a tick count, even a well-formed one
    np.array(["2001-02-03"], dtype=object),               # text is not
    np.array([0], dtype=np.int64),                        # year 1
    np.array([2 ** 62], dtype=np.int64),                  # beyond year 9999
    np.array([], dtype=np.int64),
    None,
])
def test_anything_that_is_not_a_plausible_tick_count_gives_nothing(value):
    assert apex.dotnet_ticks_to_iso(value) is None


def test_the_end_of_the_scan_is_reported_with_what_it_is_and_that_its_zone_is_unknown(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [
        _a(scan_end=np.array([TICKS_2001_02_03_040506], dtype=np.int64)), _b()])
    a, b = _inspect(client, h5)["measurements"]
    assert (a["measured_at"], a["measured_at_kind"], a["measured_at_timezone"]) == (
        "2001-02-03T04:05:06", "scan_end", "unknown")
    assert (b["measured_at"], b["measured_at_kind"], b["measured_at_timezone"]) == (None, None, None)


def test_an_attribute_of_the_wrong_type_is_no_time_stamp_and_does_not_break_the_listing(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(scan_end="yesterday")])
    a = _inspect(client, h5)["measurements"][0]
    assert a["measured_at"] is None and a["loadable"] is False and a["problem"] == "apexNoPatternFile"


# =============================================================== unreadable and plain HDF5 files

# ---------------------------------------------- an APEX file that will not open is not "plain"

def test_a_file_with_an_h5_name_that_is_not_hdf5_is_unreadable_not_plain(client, tmp_path, caplog):
    import logging
    bad = tmp_path / "session.h5"
    bad.write_bytes(b"this is not an HDF5 file" * 20)
    with caplog.at_level(logging.WARNING):
        out = _inspect(client, bad)
    assert out["kind"] == "unreadable" and out["problem"] == "apexFileUnreadable"
    assert out["h5_path"] == str(bad)
    assert "session.h5" in out["problem_detail"] and str(tmp_path) not in out["problem_detail"]
    assert any("session.h5" in r.getMessage() for r in caplog.records)         # the reason is logged


def test_an_hdf5_file_that_cannot_be_opened_is_unreadable_and_the_reason_is_logged(
        client, tmp_path, monkeypatch, caplog):
    """What a session APEX still holds looks like to h5py: the signature is there, opening fails."""
    import logging
    held = make_apex_h5(tmp_path / "held.h5", [_a()])

    def refuse(name, mode="r", *args, **kwargs):
        raise OSError("unable to lock file, errno = 33 (simulated: held by the writing program)")

    with monkeypatch.context() as patched:
        patched.setattr(h5py, "File", refuse)
        with caplog.at_level(logging.WARNING):
            out = _inspect(client, held)
    assert out["kind"] == "unreadable" and out["problem"] == "apexFileUnreadable"
    assert "held.h5" in out["problem_detail"] and "OSError" in out["problem_detail"]
    assert str(tmp_path) not in out["problem_detail"]
    logged = [r for r in caplog.records if r.exc_info and "held.h5" in r.getMessage()]
    assert logged, "the OS/HDF5 reason must be in the log with its traceback"
    assert _inspect(client, held)["kind"] == "apex"                               # and it opens again afterwards


def test_a_plain_hdf5_file_is_still_plain_and_a_missing_h5_is_a_coded_400(client, tmp_path):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_group("1/EBSD/Data")
    assert _inspect(client, other) == {"kind": "plain"}
    for name in ("missing.h5", "missing.hdf5"):
        r = client.get("/api/ebsd/inspect", params={"path": str(tmp_path / name)})
        assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexFileNotFound"
        assert str(tmp_path) not in r.json()["detail"]


# ------------------------------------------------------ which measurement a file names

GROUP_A = "TOP/Sample/Area 1/OIM Map 1"
GROUP_B = "TOP/Sample/Area 2/OIM Map 1"


def test_a_listed_file_says_which_measurement_it_names(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    make_up(tmp_path / "a.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    f = _by_id(_inspect(client, h5))[GUID_A]["pattern_files"][0]
    assert (f["names_measurement"], f["names_group"]) == (GUID_A, GROUP_A)


def test_a_chosen_file_of_another_measurement_names_that_ones_group(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    up = make_up(tmp_path / "x.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=GUID_B_BYTES)
    d = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, up, GUID_A)).json()
    assert d["problem"] == "apexGuidMismatch"
    assert (d["names_measurement"], d["names_group"]) == (GUID_B, GROUP_B)        # "belongs to OIM Map 1 of Area 2"


def test_a_file_naming_a_measurement_of_another_session_has_a_guid_but_no_group(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "x.up1", n=15, nx=5, ny=3, guid_bytes=GUID_X_BYTES)
    d = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, up)).json()
    assert (d["names_measurement"], d["names_group"]) == (GUID_X, None)


def test_a_file_without_a_guid_and_a_file_that_is_no_pattern_file_name_nothing(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    nameless = make_up(tmp_path / "n.up1", n=15, nx=5, ny=3, guid_bytes=None, version=3)
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"x" * 100)
    for path in (nameless, notes):
        d = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, path)).json()
        assert (d["names_measurement"], d["names_group"]) == (None, None), path
    out = _inspect(client, h5)
    assert [(u["names_measurement"], u["names_group"]) for u in out["unmatched_pattern_files"]] == [(None, None)]


def test_the_other_folder_listing_carries_the_names_too(client, tmp_path):
    meta, scans = tmp_path / "meta", tmp_path / "scans"
    meta.mkdir()
    scans.mkdir()
    h5 = make_apex_h5(meta / "s.h5", [_a(), _b()])
    make_up(scans / "b.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=GUID_B_BYTES)
    out = client.get("/api/ebsd/apex/pattern-files", params={"h5_path": str(h5), "folder": str(scans)}).json()
    f = _by_id(out)[GUID_B]["pattern_files"][0]
    assert (f["names_measurement"], f["names_group"]) == (GUID_B, GROUP_B)


# ---------------------- "fits" must not offer a file that Open will refuse (hex and square, same count)

def test_fits_offers_only_the_measurement_whose_grid_type_the_header_agrees_with(client, tmp_path):
    """A hexagonal 3 x 5 and a square 3 x 5 have the same point count; a v3+ header's flag decides."""
    h5 = make_apex_h5(tmp_path / "s.h5", [
        _a(), measurement_spec(id=GUID_B, area="Area 2", grid="SqrGrid", n_rows=3, n_cols=5, step_x=1.0)])
    make_up(tmp_path / "1_hex.up1", n=15, nx=5, ny=3, guid_bytes=None, version=3, hex_flag=1)
    make_up(tmp_path / "2_sqr.up1", n=15, nx=5, ny=3, guid_bytes=None, version=3, hex_flag=0)
    make_up(tmp_path / "3_v1.up1", n=15, guid_bytes=None, version=1, hex_flag=0)          # no flag in a v1 header
    out = _inspect(client, h5)
    assert [(u["name"], u["fits"]) for u in out["unmatched_pattern_files"]] == [
        ("1_hex.up1", [GUID_A]), ("2_sqr.up1", [GUID_B]), ("3_v1.up1", [GUID_A, GUID_B])]


def test_fits_leaves_out_a_measurement_that_cannot_be_opened_whatever_file_it_gets(client, tmp_path):
    broken = measurement_spec(id=GUID_X, area="Area 3", grid="SqrGrid", n_rows=3, n_cols=5, step_x=1.0,
                              n_table=14)                       # a table one record short
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), broken])
    make_up(tmp_path / "p.up1", n=15, guid_bytes=None, version=1)
    out = _inspect(client, h5)
    assert [(u["name"], u["fits"]) for u in out["unmatched_pattern_files"]] == [("p.up1", [GUID_A])]
    assert _by_id(out)[GUID_X]["measurement_problem"] == "apexGridMismatch"


# ---------------------------------------------------------------- one list of the APEX codes

def _apex_codes_in(source: str, skip_assignment_to: str = "") -> dict:
    """``{code: occurrences}`` of the string constants of a Python source that are exactly an APEX
    code (``apex`` + capital + word characters). Read from the syntax tree: a single-quoted literal
    counts, a word inside a docstring or a longer sentence does not, and the value assigned to
    ``skip_assignment_to`` (the list itself) is not a use."""
    import ast
    import re
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    skipped = set()
    if skip_assignment_to:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == skip_assignment_to for t in node.targets):
                skipped.update(id(n) for n in ast.walk(node))
    found: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings \
                and id(node) not in skipped and re.fullmatch(r"apex[A-Z]\w*", node.value):
            found[node.value] = found.get(node.value, 0) + 1
    return found


def test_the_code_scanner_reads_syntax_not_text():
    source = (
        '"""Module docstring that names "apexInDocstring" and apexAlsoInDocstring."""\n'
        'CODES = ("apexListed",)\n'
        'def f():\n'
        '    """apexInFunctionDocstring"""\n'
        "    raise E('apexSingleQuoted', 'a sentence about apexNotWholeString')\n"
        'x = "apexListed"\n')
    assert _apex_codes_in(source) == {"apexListed": 2, "apexSingleQuoted": 1}
    assert _apex_codes_in(source, skip_assignment_to="CODES") == {"apexListed": 1, "apexSingleQuoted": 1}


def test_every_apex_code_the_source_uses_is_in_the_one_list_and_no_code_in_it_is_dead():
    """Every ``.py`` under ``backend/`` plus the two loaders at the top of the tree; only files that
    mention ``apex`` inside a quote are parsed, which keeps this fast."""
    root = Path(apex.__file__).resolve().parent
    paths = [root / "edax_apex.py", root / "safe_loader.py", *sorted((root / "backend").rglob("*.py"))]
    seen: dict = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if '"apex' not in text and "'apex" not in text:
            continue
        skip = "APEX_CODES" if path.name == "edax_apex.py" else ""
        for code, n in _apex_codes_in(text, skip).items():
            seen[code] = seen.get(code, 0) + n
    assert set(seen) <= set(apex.APEX_CODES), sorted(set(seen) - set(apex.APEX_CODES))
    assert set(apex.APEX_CODES) <= set(seen), sorted(set(apex.APEX_CODES) - set(seen))      # none is dead
    assert len(set(apex.APEX_CODES)) == len(apex.APEX_CODES)


def test_the_codes_are_written_down_by_hand():
    assert set(apex.APEX_CODES) == {
        "apexBadId", "apexBadPath", "apexFileNotFound", "apexFileUnreadable", "apexFolderUnreadable",
        "apexGridMismatch", "apexGuidMismatch", "apexInvalidGrid", "apexMeasurementNotFound",
        "apexNeedsMeasurement", "apexNoMeasurement", "apexNoPatternFile", "apexNoPreview",
        "apexPairingLost", "apexPatternCountMismatch", "apexPatternFileUnreadable", "apexUnexpectedLattice",
        "apexUnknownGrid", "apexUnreadable"}


# ----------------------------------------------------------------------- an empty folder

@pytest.mark.parametrize("folder", ["", "   ", "\t"])
def test_an_empty_folder_is_refused_not_read_as_the_working_directory(client, tmp_path, folder):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    r = client.get("/api/ebsd/apex/pattern-files", params={"h5_path": str(h5), "folder": folder})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexFolderUnreadable"


# ========================================= header version, pattern width and whole patterns

def test_a_version_3_header_has_no_guid_even_when_the_first_pattern_is_not_zero(client, tmp_path):
    """A v3 header ends at byte 42; the next 16 bytes are the first pattern. With a pattern of zeros they
    look like a nil GUID (which names nothing anyway), so only a non-zero first pattern shows whether the
    reader checks where the header ends."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "v3.up1", n=15, nx=5, ny=3, guid_bytes=None, version=3, first=5)
    assert apex.up_guid(up) is None
    out = _inspect(client, h5)
    assert [(u["name"], u["fits"]) for u in out["unmatched_pattern_files"]] == [("v3.up1", [GUID_A])]


def test_a_up2_file_is_counted_with_two_bytes_per_pixel(client, tmp_path):
    """15 patterns of 4 x 6 pixels, two bytes each: the body is 15 * 48 = 720 bytes."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    named = make_up(tmp_path / "named.up2", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES, sx=6, sy=4, dtype_bytes=2)
    make_up(tmp_path / "nameless.up2", n=15, nx=5, ny=3, guid_bytes=None, version=3, sx=6, sy=4, dtype_bytes=2)
    assert named.stat().st_size == 58 + 720
    out = _inspect(client, h5)
    f = _by_id(out)[GUID_A]["pattern_files"][0]
    assert (f["name"], f["n_patterns"], f["pattern_shape"], f["count"], f["matches_grid"]) == (
        "named.up2", 15, [4, 6], "fits", True)
    u = out["unmatched_pattern_files"][0]
    assert (u["name"], u["n_patterns"], u["pattern_shape"], u["fits"]) == ("nameless.up2", 15, [4, 6], [GUID_A])


def test_only_whole_patterns_are_counted(client, tmp_path):
    """14 whole 8 x 8 patterns (64 bytes each) and 30 bytes of a 15th: that is 14, not 15."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "cut.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    data = up.read_bytes()
    assert len(data) == 58 + 15 * 64
    up.write_bytes(data[:58 + 14 * 64 + 30])
    f = _by_id(_inspect(client, h5))[GUID_A]["pattern_files"][0]
    assert (f["n_patterns"], f["count"], f["problem"]) == (14, "too_few", "apexPatternCountMismatch")


# 1999-12-31T23:59:59, 2000-01-01T00:00:00, 2100-12-31T23:59:59 and 2101-01-01T00:00:00 as .NET ticks,
# worked out by hand: 2000-01-01 is day 730119 after 0001-01-01 (730119 * 86400 s = 63082281600 s);
# 2000..2100 are 101 years with 25 leap days (2000, 2004 ... 2096; 2100 is not one), so 2101-01-01 is
# day 730119 + 101 * 365 + 25 = 767009 (767009 * 86400 s = 66269577600 s); a tick is 1e-7 s.
TICKS_1999_END = 630822815990000000
TICKS_2000_START = 630822816000000000
TICKS_2100_END = 662695775990000000
TICKS_2101_START = 662695776000000000


@pytest.mark.parametrize("ticks, iso", [
    (TICKS_1999_END, None),
    (TICKS_2000_START, "2000-01-01T00:00:00"),
    (TICKS_2100_END, "2100-12-31T23:59:59"),
    (TICKS_2101_START, None),
])
def test_the_year_bounds_of_a_time_stamp_are_2000_to_2100(ticks, iso):
    assert apex.dotnet_ticks_to_iso(np.array([ticks], dtype=np.int64)) == iso


# ==================================================================== robustness

def test_a_damaged_group_logs_its_traceback_and_carries_an_english_detail(client, tmp_path, caplog):
    import logging
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(), _b()])
    _break_camera(h5, "Area 1")
    with caplog.at_level(logging.WARNING, logger="edax_apex"):
        a, b = _inspect(client, h5)["measurements"]
    assert a["problem"] == "apexUnreadable"
    assert "s.h5" in a["problem_detail"] and "Area 1" in a["problem_detail"] and "OIM Map 1" in a["problem_detail"]
    assert str(tmp_path) not in a["problem_detail"]
    assert b["problem_detail"] is None                                  # a group that was read has none
    logged = [r for r in caplog.records if r.exc_info and "Area 1" in r.getMessage()]
    assert len(logged) == 1, "the non-ApexError case is logged once, with its traceback"


def test_a_group_whose_id_is_no_guid_carries_a_detail_too(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    with h5py.File(h5, "r+") as f:
        g = f["TOP/Sample/Area 1/OIM Map 1"]
        del g.attrs["Id"]
        g.attrs.create("Id", np.array(["not-a-guid"], dtype=object), dtype=h5py.string_dtype())
    a = _inspect(client, h5)["measurements"][0]
    assert a["problem"] == "apexBadId" and "not-a-guid" in a["problem_detail"]


def test_a_file_name_that_is_not_utf8_does_not_turn_the_listing_into_a_500(client, tmp_path, monkeypatch):
    """On Linux a name that is not valid UTF-8 arrives with a lone surrogate. The directory listing is
    patched to return one; the whole answer must still be JSON."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    bad = str(tmp_path / "bad\udcff.up1")
    with monkeypatch.context() as patched:
        patched.setattr(apex, "pattern_files_in", lambda folder: [(bad, None)])
        out = _inspect(client, h5)
    (u,) = out["unmatched_pattern_files"]
    assert u["name"] == "bad?.up1" and "\udcff" not in u["path"]
    assert u["problem"] == "apexPatternFileUnreadable"                   # the file is not there to read


def test_a_refusal_that_names_a_non_utf8_file_is_still_a_coded_400(client, tmp_path):
    """The English sentence of a refusal carries the file's name; it must not break the JSON answer."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    import json
    # httpx cannot encode a lone surrogate itself; json.dumps writes it as the escape \udcff, which a
    # server-side json parser turns back into the lone surrogate
    body = json.dumps(_body(h5, str(tmp_path / "bad\udcff.up1")))
    r = client.post("/api/ebsd/apex/check-pairing", content=body,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexPatternFileUnreadable"
    assert "bad?.up1" in r.json()["detail"]


def test_opening_a_non_utf8_pattern_file_that_is_refused_is_a_coded_400_not_a_500(client, tmp_path):
    """The step the dialog leads to: /load refuses a broken pattern file whose name is not UTF-8."""
    import json
    bad = tmp_path / "bad\udcff.up1"
    try:
        bad.write_bytes(b"\x00" * 10)                          # no header: refused by name
    except (OSError, UnicodeError):
        pytest.skip("this file system cannot hold a name with a lone surrogate")
    r = client.post("/api/ebsd/load", content=json.dumps({"path": str(bad)}),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexPatternFileUnreadable"
    assert "bad?.up1" in r.json()["detail"]


def test_a_path_the_os_cannot_open_has_no_guid_instead_of_an_error():
    """``open`` raises ValueError (not OSError) for a NUL in the path; a listing must not break on it."""
    assert apex.up_guid("scan\x00.up1") is None


def test_a_file_that_vanishes_between_the_first_and_the_second_look_at_its_header_fits_nothing(
        client, tmp_path, monkeypatch):
    """A GUID-less file is described, then its header is read again to decide where it fits; if the
    file is gone by then the entry says so and offers it to nobody."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=None, version=3)
    real = apex.read_up_header
    calls = []

    def second_look_fails(path):
        calls.append(path)
        if len(calls) >= 2:
            raise apex.ApexError("apexPatternFileUnreadable", "p.up1 cannot be read as an EDAX pattern file")
        return real(path)

    with monkeypatch.context() as patched:
        patched.setattr(apex, "read_up_header", second_look_fails)
        out = _inspect(client, h5)
    (u,) = out["unmatched_pattern_files"]
    assert (u["problem"], u["fits"]) == ("apexPatternFileUnreadable", [])
    assert u["n_patterns"] == 15                 # what the first look saw is kept


def test_the_load_answers_a_coded_refusal_when_the_pattern_file_vanishes_before_its_size_is_taken(
        client, tmp_path):
    """``load_edax_up_with_apex`` measures the file once more before it maps it. Called with the header
    and measurement of a real file but a path that is gone, it must raise the coded error."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    header = apex.read_up_header(up)
    m = apex.read_measurement(h5, GUID_A)
    gone = tmp_path / "gone.up1"
    with pytest.raises(apex.ApexError) as ei:
        apex.load_edax_up_with_apex(str(gone), header, m, "guid", use_lazy=False, verbose=False)
    assert ei.value.code == "apexPatternFileUnreadable" and "gone.up1" in str(ei.value)
    assert str(tmp_path) not in str(ei.value)


def test_the_load_route_carries_that_code_to_the_caller(client, tmp_path, monkeypatch):
    """Through /api/ebsd/load: the pairing and the header are as for a real file, the file is gone by
    the time the loader measures it (its size is taken last)."""
    import edax_up1
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    header = apex.read_up_header(up)
    m = apex.read_measurement(h5, GUID_A)
    gone = tmp_path / "gone.up1"
    with monkeypatch.context() as patched:
        patched.setattr(apex, "measurement_for_pattern_file", lambda path, explicit=None: (m, "guid"))
        patched.setattr(edax_up1, "read_up1_header", lambda path: header)
        r = client.post("/api/ebsd/load", json={"path": str(gone)})
    assert r.status_code == 400, r.text
    assert r.headers[CODE_HEADER.lower()] == "apexPatternFileUnreadable"
    assert "gone.up1" in r.json()["detail"] and str(tmp_path) not in r.json()["detail"]


def test_a_pattern_file_that_vanishes_between_two_looks_is_a_coded_answer(client, tmp_path, monkeypatch):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a()])
    up = make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    header = apex.read_up_header(up)
    up.unlink()                                  # its header was read; by the time it is measured it is gone
    with pytest.raises(apex.ApexError) as ei:
        apex.pattern_count(up, header)
    assert ei.value.code == "apexPatternFileUnreadable"
    with monkeypatch.context() as patched:
        patched.setattr(apex, "read_up_header", lambda path: header)
        d = client.post("/api/ebsd/apex/describe-pattern-file", json=_body(h5, up))
        c = client.post("/api/ebsd/apex/check-pairing", json=_body(h5, up))
    assert d.status_code == 200 and d.json()["problem"] == "apexPatternFileUnreadable"
    assert c.status_code == 400 and c.headers[CODE_HEADER.lower()] == "apexPatternFileUnreadable"


def test_find_pattern_files_is_gone_the_listing_is_the_one_way_to_look_into_a_folder():
    assert not hasattr(apex, "find_pattern_files")


# ------------------------------------------------------------------- pattern size of a measurement

def test_a_measurement_reports_the_pattern_size_its_metadata_records_as_rows_and_columns(client, tmp_path):
    """Width 156, height 128 in the file: [rows, cols] = [128, 156], the order of a pattern file's own
    ``pattern_shape`` (so the two can be compared as they are)."""
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(pattern_size=(156, 128)), _b()])
    out = _by_id(_inspect(client, h5))
    assert out[GUID_A]["pattern_shape"] == [128, 156]
    assert out[GUID_B]["pattern_shape"] is None                   # a file that does not record it


def test_the_pattern_size_is_the_same_key_on_a_measurement_that_could_not_be_read(client, tmp_path):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(pattern_size=(156, 128)), _b()])
    _break_camera(h5, "Area 1")
    a, b = _inspect(client, h5)["measurements"]
    assert a["pattern_shape"] is None and "pattern_shape" in b


@pytest.mark.parametrize("width, height", [(0, 128), (156, -1)])
def test_a_pattern_size_that_is_not_positive_is_not_reported(client, tmp_path, width, height):
    h5 = make_apex_h5(tmp_path / "s.h5", [_a(pattern_size=(width, height))])
    assert _by_id(_inspect(client, h5))[GUID_A]["pattern_shape"] is None
