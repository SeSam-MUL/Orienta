"""``GET /api/ebsd/inspect`` for every HDF5 EBSD file, and ``GET /api/ebsd/inspect/preview``.

Expected values are written out by hand from the fixtures below (a 2 x 3 scan of 4 x 5 pixel
patterns; Oxford unprocessed patterns are ``arange * 7`` as int16), never computed by the code under
test.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.problem import CODE_HEADER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "frontend/src/components/EBSDViewer/__fixtures__/inspectChoices.json"
NY, NX, SY, SX = 2, 3, 4, 5
N = NY * NX


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


def _code(r):
    return r.headers[CODE_HEADER.lower()]


def _text(f, name, value):
    f.create_dataset(name, data=np.bytes_(value))


def _oxford_group(f, name, *, unprocessed=False, backgrounds=False):
    hdr = f.create_group(f"{name}/EBSD/Header")
    data = f.create_group(f"{name}/EBSD/Data")
    hdr.create_dataset("X Cells", data=NX)
    hdr.create_dataset("Y Cells", data=NY)
    data.create_dataset("Processed Patterns", data=np.arange(N * SY * SX, dtype=np.uint8).reshape(N, SY, SX))
    if backgrounds:
        hdr.create_dataset("Processed Static Background", data=np.ones((SY, SX), np.uint8))
    if unprocessed:
        data.create_dataset("Unprocessed Patterns",
                            data=(np.arange(N * SY * SX, dtype=np.int16) * 7).reshape(N, SY, SX))
        if backgrounds:
            hdr.create_dataset("Unprocessed Static Background", data=np.ones((SY, SX), np.int16))


@pytest.fixture
def oxford_two(tmp_path):
    """Measurement 1 with both sources and both backgrounds, measurement 2 processed only: 3 variants."""
    path = tmp_path / "two.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _oxford_group(f, "1", unprocessed=True, backgrounds=True)
        _oxford_group(f, "2")
    return path


@pytest.fixture
def oxford_one(tmp_path):
    path = tmp_path / "one.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _oxford_group(f, "1")
    return path


@pytest.fixture
def bruker_two(tmp_path):
    path = tmp_path / "bruker.h5"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Bruker Nano")
        for name in ("Scan 10", "Scan 2"):
            hdr = f.create_group(f"{name}/EBSD/Header")
            data = f.create_group(f"{name}/EBSD/Data")
            hdr.create_dataset("NCOLS", data=NX)
            hdr.create_dataset("NROWS", data=NY)
            data.create_dataset("RawPatterns", data=np.zeros((N, SY, SX), np.uint8))
    return path


@pytest.fixture
def kikuchipy_one(tmp_path):
    """A 4-D pattern dataset (rows, columns, y, x): pattern (r, c) is filled with ``10 * r + c``."""
    path = tmp_path / "kp.h5"
    data = np.zeros((NY, NX, SY, SX), np.uint8)
    for r in range(NY):
        for c in range(NX):
            data[r, c] = 10 * r + c + np.arange(SY * SX, dtype=np.uint8).reshape(SY, SX)
    with h5py.File(path, "w") as f:
        _text(f, "manufacturer", "kikuchipy")
        hdr = f.create_group("Scan 1/EBSD/Header")
        grp = f.create_group("Scan 1/EBSD/Data")
        hdr.create_dataset("n_columns", data=NX)
        hdr.create_dataset("n_rows", data=NY)
        grp.create_dataset("patterns", data=data)
    return path, data


def _inspect(client, path):
    r = client.get("/api/ebsd/inspect", params={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------------------------- inspect


def test_two_source_oxford_is_choices(client, oxford_two):
    body = _inspect(client, oxford_two)
    assert body["kind"] == "choices"
    assert body["path"] == str(oxford_two)
    assert body["format"] == "oxford"
    assert body["n_variants"] == 3
    assert body["default_variant"] == {"measurement": "1", "pattern_source": "processed"}
    assert "default" not in body and "suggested" not in body
    assert [m["name"] for m in body["measurements"]] == ["1", "2"]
    first, second = body["measurements"]
    assert (first["n_rows"], first["n_cols"]) == (2, 3)
    assert [s["key"] for s in first["sources"]] == ["processed", "unprocessed"]
    assert [s["dtype"] for s in first["sources"]] == ["uint8", "int16"]
    assert [s["has_background"] for s in first["sources"]] == [True, True]
    assert all(s["pattern_shape"] == [4, 5] and s["n_patterns"] == 6 and s["loadable"] for s in first["sources"])
    assert [s["key"] for s in second["sources"]] == ["processed"]
    assert second["sources"][0]["has_background"] is False


def test_choices_carry_no_other_path_than_the_one_asked_about(client, oxford_two):
    """The measurement and source entries name HDF5 datasets, never a folder of this machine."""
    text = json.dumps(_inspect(client, oxford_two)["measurements"])
    assert oxford_two.parent.name not in text and oxford_two.name not in text


def test_other_hdf5_formats_with_several_scans_are_choices_too(client, bruker_two):
    body = _inspect(client, bruker_two)
    assert body["kind"] == "choices" and body["format"] == "bruker"
    assert [m["name"] for m in body["measurements"]] == ["Scan 2", "Scan 10"]        # natural order
    assert body["default_variant"] == {"measurement": "Scan 2", "pattern_source": "single"}
    assert body["n_variants"] == 2


def test_single_variant_is_plain(client, oxford_one, kikuchipy_one):
    assert _inspect(client, oxford_one) == {"kind": "plain"}
    assert _inspect(client, kikuchipy_one[0]) == {"kind": "plain"}


def test_h5oina_and_h5ebsd_names_are_asked_about(client, tmp_path):
    for name in ("missing.h5oina", "missing.h5ebsd"):
        r = client.get("/api/ebsd/inspect", params={"path": str(tmp_path / name)})
        assert r.status_code == 400 and _code(r) == "apexFileNotFound", name
        bad = tmp_path / ("bad_" + name)
        bad.write_bytes(b"not hdf5")
        body = _inspect(client, bad)
        assert body["kind"] == "unreadable" and body["problem"] == "apexFileUnreadable", name


def test_other_names_are_not_asked_about(client, tmp_path):
    for name in ("a.up1", "Pattern.dat", "Setting.txt", "a.ebsp", "a.ang"):
        assert _inspect(client, tmp_path / name) == {"kind": "plain"}


def test_apex_unchanged(client, tmp_path):
    from tests.apex_helpers import GUID_A, make_apex_h5, measurement_spec
    path = make_apex_h5(tmp_path / "session.h5", [measurement_spec(id=GUID_A, area="Area 1")])
    body = _inspect(client, path)
    assert body["kind"] == "apex" and body["h5_path"] == str(path)
    assert "format" not in body and len(body["measurements"]) == 1


def _break_with_dangling_link(path):
    with h5py.File(path, "a") as f:
        del f["2/EBSD/Data/Processed Patterns"]
        f["2/EBSD/Data/Processed Patterns"] = h5py.SoftLink("/nowhere")          # KeyError when read


def _break_with_scalar_header(path):
    with h5py.File(path, "a") as f:
        del f["2/EBSD/Header"]
        f["2/EBSD/Header"] = np.float64(1.0)                                    # TypeError when read


@pytest.mark.parametrize("damage", [_break_with_dangling_link, _break_with_scalar_header])
def test_a_damaged_group_is_a_coded_400_never_a_500(client, oxford_two, damage):
    damage(oxford_two)
    r = client.get("/api/ebsd/inspect", params={"path": str(oxford_two)})
    assert r.status_code == 400, r.text
    assert _code(r) == "loadInspectFailed"
    assert oxford_two.parent.name not in r.text


def test_the_fixture_matches_the_backend(client, oxford_two):
    """``inspectChoices.json`` is the answer for ``oxford_two``; the vitest routing tests read it."""
    body = _inspect(client, oxford_two)
    body["path"] = "<path>"
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == body, (
        "the /inspect answer changed: regenerate inspectChoices.json from the oxford_two fixture "
        "(path replaced by '<path>')")


# ------------------------------------------------------------------------------------ to_dict


def test_file_choices_to_dict_carries_no_path(oxford_two):
    from backend.api.services import load_choices as lc
    d = lc.inspect_file(str(oxford_two)).to_dict()
    assert "path" not in d
    assert oxford_two.parent.name not in json.dumps(d) and oxford_two.name not in json.dumps(d)
    assert d["format"] == "oxford" and d["n_variants"] == 3


# -------------------------------------------------------------------------------------- preview


def _preview(client, path, measurement, source):
    return client.get("/api/ebsd/inspect/preview",
                      params={"path": str(path), "measurement": measurement, "source": source})


def _png(r) -> np.ndarray:
    from PIL import Image
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    return np.array(Image.open(io.BytesIO(r.content)))


def test_preview_of_an_int16_source_is_stretched_by_the_range_of_that_pattern(client, oxford_two):
    # pattern 3 of 6 is the middle one: its values are 7 * (60 + k), k = 0..19, so 420..553.
    # (v - 420) * 255 / 133 = k * 255 / 19: 0 for k = 0, 255 for k = 19, 13 for k = 1, 121 for k = 9.
    img = _png(_preview(client, oxford_two, "1", "unprocessed"))
    assert img.dtype == np.uint8 and img.shape == (SY, SX)
    flat = img.reshape(-1)
    assert flat[0] == 0 and flat[19] == 255 and flat[1] == 13 and flat[9] == 121 and flat[10] == 134
    assert [int(v) for v in flat] == [int(round(255 * k / 19)) for k in range(20)]


def test_preview_of_a_uint8_source_is_the_middle_pattern_unchanged(client, oxford_two):
    # processed patterns are arange(120) as uint8: pattern 3 holds 60..79
    img = _png(_preview(client, oxford_two, "1", "processed"))
    assert img.reshape(-1).tolist() == list(range(60, 80))


def test_preview_middle_of_a_four_dimensional_dataset_is_flat_index_n_over_two(client, kikuchipy_one):
    path, data = kikuchipy_one
    # 6 patterns, flat index 3 is (row 1, column 0): filled with 10 + k, k = 0..19
    img = _png(_preview(client, path, "Scan 1", "single"))
    assert img.reshape(-1).tolist() == [10 + k for k in range(20)]


def test_preview_reads_only_that_one_pattern(client, oxford_two, monkeypatch):
    reads = []
    original = h5py.Dataset.__getitem__

    def spy(self, args, new_dtype=None):
        reads.append((self.name, args))
        return original(self, args, new_dtype)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    _png(_preview(client, oxford_two, "1", "unprocessed"))
    patterns = [(name, key) for name, key in reads if name.endswith("Patterns")]
    assert patterns == [("/1/EBSD/Data/Unprocessed Patterns", 3)]


def test_preview_loads_nothing_into_the_viewer(client, oxford_two):
    from backend.api.routes import ebsd_viewer
    before = (ebsd_viewer._ebsd_file_path, ebsd_viewer._ebsd_signal, list(ebsd_viewer._loaded_files))
    _png(_preview(client, oxford_two, "1", "processed"))
    assert (ebsd_viewer._ebsd_file_path, ebsd_viewer._ebsd_signal, list(ebsd_viewer._loaded_files)) == before


def test_preview_unknown_measurement_or_source_is_a_coded_400(client, oxford_two):
    for measurement, source in (("9", "processed"), ("2", "unprocessed"), ("1", "bogus"), ("1", "single")):
        r = _preview(client, oxford_two, measurement, source)
        assert r.status_code == 400 and _code(r) == "loadVariantUnknown", (measurement, source, r.text)


def test_preview_of_a_file_without_choices_is_a_coded_400(client, tmp_path):
    from tests.apex_helpers import GUID_A, make_apex_h5, measurement_spec
    apex_path = make_apex_h5(tmp_path / "session.h5", [measurement_spec(id=GUID_A, area="Area 1")])
    r = _preview(client, apex_path, "1", "single")
    assert r.status_code == 400 and _code(r) == "loadNoPatterns"


def test_preview_checks_the_path_as_inspect_does(client, tmp_path):
    r = _preview(client, tmp_path / "missing.h5oina", "1", "processed")
    assert r.status_code == 400 and _code(r) == "apexFileNotFound"
    folder = tmp_path / "folder.h5"
    folder.mkdir()
    r = _preview(client, folder, "1", "processed")
    assert r.status_code == 400 and _code(r) == "apexFileUnreadable"
    bad = tmp_path / "bad.h5oina"
    bad.write_bytes(b"not hdf5")
    r = _preview(client, bad, "1", "processed")
    assert r.status_code == 400 and _code(r) == "apexFileUnreadable"
    r = _preview(client, "a\x00b.h5", "1", "processed")
    assert r.status_code == 400 and _code(r) == "apexBadPath"


def test_preview_answers_alike_for_every_name_inspect_does_not_ask_about(client, tmp_path, oxford_two):
    hdf5_under_another_name = tmp_path / "scan.dat"
    hdf5_under_another_name.write_bytes(Path(oxford_two).read_bytes())
    note = tmp_path / "note.txt"
    note.write_text("text", encoding="utf-8")
    for path in (hdf5_under_another_name, note, tmp_path / "missing.xyz"):
        r = _preview(client, path, "1", "processed")
        assert r.status_code == 400 and _code(r) == "loadNoPatterns"


@pytest.mark.parametrize("damage", [_break_with_dangling_link, _break_with_scalar_header])
def test_preview_of_a_damaged_group_is_a_coded_400(client, oxford_two, damage):
    damage(oxford_two)
    r = _preview(client, oxford_two, "1", "processed")
    assert r.status_code == 400 and _code(r) == "loadInspectFailed", r.text


def test_preview_of_the_real_sampleb_file(client):
    found = sorted((ROOT / "Test_data").glob("EBSD_SampleB_extrusion_withPattern*.h5oina"))
    if not found:
        pytest.skip("EBSD_SampleB real file is not available here")
    with h5py.File(found[0], "r") as f:
        processed = f["1/EBSD/Data/Processed Patterns"]
        middle = processed[processed.shape[0] // 2]
        raw = f["1/EBSD/Data/Unprocessed Patterns"][processed.shape[0] // 2]
    assert raw.dtype == np.int16 and raw.max() > raw.min()
    # the 8 bit source is shown as stored, the int16 one stretched over its own range
    assert np.array_equal(_png(_preview(client, found[0], "1", "processed")), middle)
    img = _png(_preview(client, found[0], "1", "unprocessed"))
    assert img.shape == raw.shape and img.dtype == np.uint8 and img.min() == 0 and img.max() == 255
    # monotone in the stored value: the darkest raw pixel is the darkest shown, the brightest the brightest
    assert img.flat[int(np.argmin(raw))] == 0 and img.flat[int(np.argmax(raw))] == 255
