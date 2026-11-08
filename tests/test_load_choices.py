"""Which measurements and pattern sources an EBSD file offers, read from HDF5 metadata only."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import load_choices as lc  # noqa: E402
from backend.api.services.load_choices import LoadVariant  # noqa: E402

NY, NX, SY, SX = 2, 3, 4, 5


def _text(f, name, value):
    f.create_dataset(name, data=np.bytes_(value))


def _oxford_group(f, name, *, processed=True, unprocessed=False, backgrounds=False, patterns=True):
    hdr = f.create_group(f"{name}/EBSD/Header")
    data = f.create_group(f"{name}/EBSD/Data")
    hdr.create_dataset("X Cells", data=NX)
    hdr.create_dataset("Y Cells", data=NY)
    hdr.create_dataset("Pattern Height", data=SY)
    hdr.create_dataset("Pattern Width", data=SX)
    if not patterns:
        return
    n = NY * NX
    if processed:
        data.create_dataset("Processed Patterns", data=np.arange(n * SY * SX, dtype=np.uint8).reshape(n, SY, SX))
        if backgrounds:
            hdr.create_dataset("Processed Static Background", data=np.ones((SY, SX), np.uint8))
    if unprocessed:
        data.create_dataset("Unprocessed Patterns",
                            data=(np.arange(n * SY * SX, dtype=np.int16) * 7).reshape(n, SY, SX))
        if backgrounds:
            hdr.create_dataset("Unprocessed Static Background", data=np.ones((SY, SX), np.int16))


@pytest.fixture
def oxford_two(tmp_path):
    path = tmp_path / "two.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _text(f, "Format Version", "5.0")
        _oxford_group(f, "1", unprocessed=True, backgrounds=True)
        _oxford_group(f, "2")
    return str(path)


@pytest.fixture
def oxford_only_unprocessed(tmp_path):
    path = tmp_path / "unproc.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _oxford_group(f, "1", processed=False, unprocessed=True)
    return str(path)


@pytest.fixture
def oxford_with_empty_group(tmp_path):
    path = tmp_path / "empty.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _oxford_group(f, "1")
        _oxford_group(f, "2", patterns=False)
    return str(path)


@pytest.fixture
def oxford_all_empty(tmp_path):
    path = tmp_path / "allempty.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _oxford_group(f, "1", patterns=False)
    return str(path)


@pytest.fixture
def bruker_two(tmp_path):
    path = tmp_path / "bruker.h5"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Bruker Nano")
        _text(f, "Version", "1.0")
        for name in ("Scan 10", "Scan 2"):
            hdr = f.create_group(f"{name}/EBSD/Header")
            data = f.create_group(f"{name}/EBSD/Data")
            hdr.create_dataset("NCOLS", data=NX)
            hdr.create_dataset("NROWS", data=NY)
            hdr.create_dataset("StaticBackground", data=np.ones((SY, SX), np.uint8))
            data.create_dataset("RawPatterns", data=np.zeros((NY * NX, SY, SX), np.uint8))
    return str(path)


@pytest.fixture
def edax_one(tmp_path):
    path = tmp_path / "edax.h5"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "EDAX")
        _text(f, "Version", "OIM 8")
        hdr = f.create_group("Scan 1/EBSD/Header")
        data = f.create_group("Scan 1/EBSD/Data")
        hdr.create_dataset("nColumns", data=NX)
        hdr.create_dataset("nRows", data=NY)
        data.create_dataset("Pattern", data=np.zeros((NY * NX, SY, SX), np.uint8))
    return str(path)


@pytest.fixture
def kikuchipy_one(tmp_path):
    path = tmp_path / "kp.h5"
    with h5py.File(path, "w") as f:
        _text(f, "manufacturer", "kikuchipy")
        _text(f, "version", "0.13")
        hdr = f.create_group("Scan 1/EBSD/Header")
        data = f.create_group("Scan 1/EBSD/Data")
        hdr.create_dataset("n_columns", data=NX)
        hdr.create_dataset("n_rows", data=NY)
        data.create_dataset("patterns", data=np.zeros((NY, NX, SY, SX), np.uint8))
    return str(path)


def test_oxford_two_groups_lists_three_variants(oxford_two):
    ch = lc.inspect_file(oxford_two)
    assert ch.format == "oxford"
    assert ch.n_variants() == 3
    first, second = ch.measurements
    assert (first.name, second.name) == ("1", "2")
    assert [s.key for s in first.sources] == ["processed", "unprocessed"]
    assert [s.key for s in second.sources] == ["processed"]
    proc, unproc = first.sources
    assert proc.dtype == "uint8" and unproc.dtype == "int16"
    assert proc.dataset == "1/EBSD/Data/Processed Patterns"
    assert unproc.dataset == "1/EBSD/Data/Unprocessed Patterns"
    assert unproc.n_patterns == NY * NX and unproc.pattern_shape == (SY, SX)
    assert (first.n_rows, first.n_cols) == (NY, NX)
    assert proc.has_background and unproc.has_background
    assert second.sources[0].has_background is False
    assert all(s.loadable and s.problem is None for m in ch.measurements for s in m.sources)


def test_background_follows_its_own_source(tmp_path):
    path = tmp_path / "bg.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _oxford_group(f, "1", unprocessed=True, backgrounds=True)
        del f["1/EBSD/Header/Unprocessed Static Background"]
    ch = lc.inspect_file(str(path))
    proc, unproc = ch.measurements[0].sources
    assert proc.has_background is True
    assert unproc.has_background is False


def test_default_variant_is_first_group_processed(oxford_two):
    assert lc.inspect_file(oxford_two).default_variant() == LoadVariant("1", "processed")


def test_only_unprocessed_defaults_to_it(oxford_only_unprocessed):
    ch = lc.inspect_file(oxford_only_unprocessed)
    assert ch.n_variants() == 1
    assert [s.key for s in ch.measurements[0].sources] == ["unprocessed"]
    assert ch.default_variant() == LoadVariant("1", "unprocessed")


def test_group_without_patterns_is_listed_not_loadable(oxford_with_empty_group):
    ch = lc.inspect_file(oxford_with_empty_group)
    empty = ch.measurements[1]
    assert empty.name == "2"
    assert len(empty.sources) == 1
    assert empty.sources[0].loadable is False
    assert empty.sources[0].problem == "loadNoPatterns"


def test_n_variants_counts_loadable_only(oxford_with_empty_group):
    ch = lc.inspect_file(oxford_with_empty_group)
    assert len(ch.measurements) == 2
    assert ch.n_variants() == 1
    assert ch.default_variant() == LoadVariant("1", "processed")


def test_default_skips_a_leading_group_without_patterns(tmp_path):
    path = tmp_path / "lead.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _oxford_group(f, "1", patterns=False)
        _oxford_group(f, "2")
    ch = lc.inspect_file(str(path))
    assert ch.default_variant() == LoadVariant("2", "processed")


def test_no_loadable_variant_raises_coded_error(oxford_all_empty):
    ch = lc.inspect_file(oxford_all_empty)
    assert ch.n_variants() == 0
    with pytest.raises(ValueError) as err:
        ch.default_variant()
    assert err.value.code == "loadNoPatterns"


def test_bruker_two_groups_single_source(bruker_two):
    ch = lc.inspect_file(bruker_two)
    assert ch.format == "bruker"
    assert [m.name for m in ch.measurements] == ["Scan 2", "Scan 10"]
    for m in ch.measurements:
        assert [s.key for s in m.sources] == ["single"]
        assert m.sources[0].dataset.endswith("RawPatterns")
        assert m.sources[0].has_background is True
        assert (m.n_rows, m.n_cols) == (NY, NX)
    assert ch.n_variants() == 2
    assert ch.default_variant() == LoadVariant("Scan 2", "single")


def test_single_variant_file_reports_one(edax_one):
    ch = lc.inspect_file(edax_one)
    assert ch.format == "edax_h5ebsd"
    assert ch.n_variants() == 1
    assert ch.measurements[0].sources[0].dataset == "Scan 1/EBSD/Data/Pattern"
    assert (ch.measurements[0].n_rows, ch.measurements[0].n_cols) == (NY, NX)
    assert ch.measurements[0].sources[0].has_background is False


def test_kikuchipy_h5ebsd_four_dimensional_patterns(kikuchipy_one):
    ch = lc.inspect_file(kikuchipy_one)
    assert ch.format == "kikuchipy_h5ebsd"
    src = ch.measurements[0].sources[0]
    assert src.key == "single" and src.dataset == "Scan 1/EBSD/Data/patterns"
    assert src.n_patterns == NY * NX and src.pattern_shape == (SY, SX)


def test_formats_with_one_variant_by_construction_return_none(tmp_path):
    for name in ("a.up1", "a.up2", "Pattern.dat", "Setting.txt", "a.ebsp"):
        p = tmp_path / name
        p.write_bytes(b"x")
        assert lc.inspect_file(str(p)) is None


def test_apex_and_unknown_hdf5_return_none(tmp_path):
    from tests.apex_helpers import make_apex_h5, measurement_spec
    apex_path = make_apex_h5(tmp_path / "session.h5", [measurement_spec()])
    assert lc.inspect_file(str(apex_path)) is None
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_dataset("x", data=1)
    assert lc.inspect_file(str(other)) is None
    assert lc.inspect_file(str(tmp_path / "missing.h5oina")) is None


def test_oxford_layout_without_manufacturer_is_found_by_structure(tmp_path):
    path = tmp_path / "nomfg.h5oina"
    with h5py.File(path, "w") as f:
        _oxford_group(f, "1", unprocessed=True)
    ch = lc.inspect_file(str(path))
    assert ch.format == "oxford" and ch.n_variants() == 2


def test_pattern_dataset_path_follows_variant(oxford_two, bruker_two):
    assert lc.pattern_dataset_path(oxford_two, LoadVariant("1", "unprocessed")).endswith(
        "1/EBSD/Data/Unprocessed Patterns")
    assert lc.pattern_dataset_path(oxford_two, LoadVariant("2", "processed")).endswith(
        "2/EBSD/Data/Processed Patterns")
    assert lc.pattern_dataset_path(oxford_two, None).endswith("1/EBSD/Data/Processed Patterns")
    assert lc.pattern_dataset_path(bruker_two, LoadVariant("Scan 10", "single")).endswith(
        "Scan 10/EBSD/Data/RawPatterns")
    assert lc.pattern_dataset_path(bruker_two, None).endswith("Scan 2/EBSD/Data/RawPatterns")


def test_pattern_dataset_path_none_variant_takes_an_existing_dataset(oxford_only_unprocessed):
    path = lc.pattern_dataset_path(oxford_only_unprocessed, None)
    assert path.endswith("1/EBSD/Data/Unprocessed Patterns")
    with h5py.File(oxford_only_unprocessed, "r") as f:
        assert path in f


def test_pattern_dataset_path_refuses_what_the_file_does_not_hold(oxford_two):
    for bad in (LoadVariant("2", "unprocessed"), LoadVariant("9", "processed")):
        with pytest.raises(ValueError) as err:
            lc.pattern_dataset_path(oxford_two, bad)
        assert err.value.code == "loadVariantUnknown"


def test_reads_no_pattern_array(oxford_two, bruker_two, edax_one, kikuchipy_one, monkeypatch):
    real = h5py.Dataset.__getitem__

    def guarded(self, args):
        if self.ndim >= 3 or "Patterns" in self.name or self.name.endswith(("/Pattern", "/patterns")):
            raise AssertionError(f"pattern array read: {self.name}")
        return real(self, args)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    for path in (oxford_two, bruker_two, edax_one, kikuchipy_one):
        assert lc.inspect_file(path) is not None
        lc.pattern_dataset_path(path, None)


def test_load_variant_round_trip_and_is_default(oxford_two):
    ch = lc.inspect_file(oxford_two)
    v = LoadVariant("1", "unprocessed")
    assert v.to_dict() == {"measurement": "1", "pattern_source": "unprocessed"}
    assert LoadVariant.from_dict(v.to_dict()) == v
    assert LoadVariant("1", "processed").is_default(ch) is True
    assert v.is_default(ch) is False
    assert LoadVariant("2", "processed").is_default(ch) is False
    assert LoadVariant(None, "processed").is_default(ch) is True
    with pytest.raises(ValueError):
        LoadVariant.from_dict({"measurement": "1", "pattern_source": "bogus"})


def test_file_choices_to_dict_is_json_ready(oxford_two):
    import json
    d = lc.inspect_file(oxford_two).to_dict()
    json.dumps(d)
    assert d["format"] == "oxford" and d["n_variants"] == 3
    assert d["default"] == {"measurement": "1", "pattern_source": "processed"}
    src = d["measurements"][0]["sources"][1]
    assert src["key"] == "unprocessed" and src["dtype"] == "int16" and src["pattern_shape"] == [SY, SX]
    assert src["loadable"] is True and src["problem"] is None


def test_detect_format_names_each_layout(tmp_path, oxford_two, bruker_two, edax_one, kikuchipy_one):
    assert lc.detect_format(oxford_two) == "oxford"
    assert lc.detect_format(bruker_two) == "bruker"
    assert lc.detect_format(edax_one) == "edax_h5ebsd"
    assert lc.detect_format(kikuchipy_one) == "kikuchipy_h5ebsd"
    for name, expected in (("a.up1", "edax_up"), ("a.UP2", "edax_up"), ("a.ebsp", "oxford_binary")):
        p = tmp_path / name
        p.write_bytes(b"x")
        assert lc.detect_format(str(p)) == expected
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_dataset("x", data=1)
    assert lc.detect_format(str(other)) == "other"
    plain = tmp_path / "notes.csv"
    plain.write_text("a,b")
    assert lc.detect_format(str(plain)) == "other"
    try:
        from tests.apex_helpers import make_apex_h5, measurement_spec
        import edax_apex  # noqa: F401
    except ImportError:
        pytest.skip("edax_apex not available")
    apex_path = make_apex_h5(tmp_path / "session.h5", [measurement_spec()])
    assert lc.detect_format(str(apex_path)) == "apex"


def test_to_uint8_int16_is_monotone_not_wrapped():
    a = np.array([[600, 1200, 3000], [4500, 6000, 6500]], dtype=np.int16)
    out = lc.to_uint8(a)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
    flat_in, flat_out = a.ravel(), out.ravel()
    order = np.argsort(flat_in)
    assert (np.diff(flat_out[order].astype(int)) >= 0).all()
    assert out[0, 0] == 0 and out[1, 2] == 255
    assert out[0, 1] == round((1200 - 600) / (6500 - 600) * 255)


def test_to_uint8_passes_uint8():
    a = np.array([[0, 7], [200, 255]], dtype=np.uint8)
    out = lc.to_uint8(a)
    assert out.dtype == np.uint8
    assert np.array_equal(out, a)


def test_to_uint8_constant_input_gives_zeros_and_float_is_rescaled():
    assert not lc.to_uint8(np.full((2, 2), 900, np.int16)).any()
    f = lc.to_uint8(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    assert f.dtype == np.uint8 and list(f) == [0, 128, 255]


def _sampleb():
    for parent in Path(__file__).resolve().parents:
        for hit in sorted((parent / "Test_data").glob("EBSD_SampleB_extrusion_withPattern*.h5oina")):
            if hit.stat().st_size > 1_000_000:
                return hit
    return None


def test_sampleb_has_both_sources():
    path = _sampleb()
    if path is None:
        pytest.skip("EBSD_SampleB real file is not available here")
    ch = lc.inspect_file(str(path))
    assert ch.format == "oxford"
    assert ch.n_variants() == 2
    proc, unproc = ch.measurements[0].sources
    assert (proc.key, proc.dtype) == ("processed", "uint8")
    assert (unproc.key, unproc.dtype) == ("unprocessed", "int16")
    assert proc.n_patterns == unproc.n_patterns > 0
    assert ch.default_variant() == LoadVariant(ch.measurements[0].name, "processed")


# ---------------------------------------------------------------- NORDIF is told by content, not by extension

_NORDIF_SETTING = "[NORDIF]\nSoftware version\t3.1\n[Microscope]\nVoltage\t20\n[Area]\nResolution\t10x10\n"


def test_a_text_file_is_nordif_only_when_it_is_a_nordif_setting_file(tmp_path):
    setting = tmp_path / "Setting.txt"
    setting.write_text(_NORDIF_SETTING, encoding="latin-1")
    assert lc.detect_format(str(setting)) == "nordif"
    crlf = tmp_path / "other_setting.txt"                                 # CRLF and a UTF-8 byte order mark
    crlf.write_bytes(b"\xef\xbb\xbf" + _NORDIF_SETTING.replace("\n", "\r\n").encode("latin-1"))
    assert lc.detect_format(str(crlf)) == "nordif"
    for name, text in (("notes.txt", "meeting on friday\n[Microscope room booked]\n"),
                       ("empty.txt", ""), ("table.txt", "1\t2\t3\n4\t5\t6\n"),
                       ("half.txt", "[Microscope]\nVoltage\t20\n")):    # no [Area]: not what the reader needs
        p = tmp_path / name
        p.write_text(text, encoding="latin-1")
        assert lc.detect_format(str(p)) == "other", name
    assert lc.detect_format(str(tmp_path / "missing.txt")) == "other"


def test_a_dat_file_is_nordif_only_beside_a_nordif_setting_file(tmp_path):
    lone = tmp_path / "lone" / "Pattern.dat"
    lone.parent.mkdir()
    lone.write_bytes(b"\x00" * 16)
    assert lc.detect_format(str(lone)) == "other"
    beside = tmp_path / "scan" / "Pattern.dat"
    beside.parent.mkdir()
    beside.write_bytes(b"\x00" * 16)
    (beside.parent / "Setting.txt").write_text(_NORDIF_SETTING, encoding="latin-1")
    assert lc.detect_format(str(beside)) == "nordif"
    wrong = tmp_path / "wrong" / "Pattern.dat"
    wrong.parent.mkdir()
    wrong.write_bytes(b"\x00" * 16)
    (wrong.parent / "Setting.txt").write_text("not a setting file\n", encoding="latin-1")
    (wrong.parent / "notes.txt").write_text(_NORDIF_SETTING, encoding="latin-1")     # not named Setting.txt
    assert lc.detect_format(str(wrong)) == "other"
