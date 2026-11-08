"""EDAX APEX metadata files: listing, GUID pairing and previews."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edax_apex as apex  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_A_BYTES, GUID_B, GUID_B_BYTES, make_apex_h5, make_up, measurement_spec,
)


@pytest.fixture
def folder(tmp_path):
    return tmp_path


def _two(folder):
    return make_apex_h5(folder / "session.h5", [
        measurement_spec(id=GUID_A, area="Area 1", name="OIM Map 1"),
        measurement_spec(id=GUID_B, area="Area 2", name="OIM Map 1", grid="SqrGrid",
                         n_rows=2, n_cols=4, step_x=1.0, pc=(0.55, 0.45, 0.7)),
    ])


def test_it_recognises_an_apex_file_and_nothing_else(folder, tmp_path):
    h5 = _two(folder)
    assert apex.is_apex_file(h5)
    import h5py
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_group("1/EBSD/Data")
    assert not apex.is_apex_file(other)
    assert not apex.is_apex_file(tmp_path / "missing.h5")
    assert not apex.is_apex_file(folder / "x.up1")


def test_it_lists_every_measurement_with_what_the_dialog_shows(folder):
    ms = apex.list_measurements(_two(folder))
    assert [(m.area, m.name) for m in ms] == [("Area 1", "OIM Map 1"), ("Area 2", "OIM Map 1")]
    a, b = ms
    assert a.id == GUID_A and a.grid_type == "HexGrid" and a.is_hex
    assert (a.n_rows, a.n_cols, a.expected_patterns) == (3, 5, 15)
    assert a.step_x == pytest.approx(0.5) and a.step_y == pytest.approx(0.5 * 0.8660254, rel=1e-6)
    assert a.pc_tsl == pytest.approx((0.5, 0.4, 0.6), abs=1e-6)
    assert (a.sample_tilt, a.elevation, a.azimuth, a.beam_kv) == (70.0, 5.0, 0.0, 20.0)
    assert [(p.index, p.name, p.symmetry) for p in a.phases] == [(1, "Phase A", 62), (2, "Phase B", 43)]
    assert a.has_sem and a.has_iq and a.n_points == 15
    assert b.grid_type == "SqrGrid" and not b.is_hex and b.pc_tsl[0] == pytest.approx(0.55, abs=1e-6)


def test_areas_and_maps_are_listed_in_natural_order_not_text_order(tmp_path):
    ids = [f"00000000-0000-0000-0000-00000000000{i}" for i in range(1, 5)]
    h5 = make_apex_h5(tmp_path / "s.h5", [
        measurement_spec(id=ids[0], area="Area 10", name="OIM Map 1"),
        measurement_spec(id=ids[1], area="Area 2", name="OIM Map 10"),
        measurement_spec(id=ids[2], area="Area 2", name="OIM Map 2"),
        measurement_spec(id=ids[3], area="Area 1", name="OIM Map 1"),
    ])
    assert [(m.area, m.name) for m in apex.list_measurements(h5)] == [
        ("Area 1", "OIM Map 1"), ("Area 2", "OIM Map 2"), ("Area 2", "OIM Map 10"),
        ("Area 10", "OIM Map 1")]


def test_the_hex_flag_attribute_is_not_what_decides(folder):
    # the helper writes HexGrid=0 on every group, as real files do; Sample/Grid Type decides
    assert apex.list_measurements(_two(folder))[0].is_hex


def test_the_guid_is_read_from_byte_42_in_field_order_of_a_little_endian_guid(folder):
    up = make_up(folder / "any name.up1", guid_bytes=GUID_A_BYTES)
    assert apex.up_guid(up) == GUID_A          # GUID_A_BYTES was written out by hand
    assert apex.up_guid(make_up(folder / "b.up1", guid_bytes=GUID_B_BYTES)) == GUID_B


def test_a_header_that_ends_before_byte_58_has_no_guid(folder):
    assert apex.up_guid(make_up(folder / "v3.up1", guid_bytes=None, version=3)) is None


def test_pairing_ignores_the_file_name(folder):
    h5 = _two(folder)
    # The file that belongs to Area 1 is called like Area 2 and vice versa.
    make_up(folder / "Sample_Area 2_OIM Map 1.up1", guid_bytes=GUID_A_BYTES)
    make_up(folder / "Sample_Area 1_OIM Map 1.up1", guid_bytes=GUID_B_BYTES, hex_flag=0,
            n=8, nx=4, ny=2)
    a, b = apex.list_measurements(h5)
    listed, unmatched = apex.scan_pattern_files(folder, [a, b])
    assert [f["name"] for f in listed[a.id]] == ["Sample_Area 2_OIM Map 1.up1"]
    assert [f["name"] for f in listed[b.id]] == ["Sample_Area 1_OIM Map 1.up1"]
    assert unmatched == []


def test_pairing_looks_in_the_folder_it_is_given_only(folder, tmp_path):
    h5 = _two(folder)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    make_up(elsewhere / "p.up1", guid_bytes=GUID_A_BYTES)
    m = apex.list_measurements(h5)[0]
    here, _ = apex.scan_pattern_files(folder, [m])              # the metadata file's own folder
    there, _ = apex.scan_pattern_files(elsewhere, [m])
    assert here[m.id] == []
    assert [f["name"] for f in there[m.id]] == ["p.up1"]


def test_the_sidecar_of_a_pattern_file_is_found_from_its_guid(folder):
    _two(folder)
    up = make_up(folder / "p.up1", guid_bytes=GUID_B_BYTES, hex_flag=0, n=8, nx=4, ny=2)
    m = apex.find_apex_sidecar(up)
    assert m is not None and m.id == GUID_B and m.area == "Area 2"
    assert apex.find_apex_sidecar(make_up(folder / "v3.up1", guid_bytes=None, version=3)) is None


def test_check_pairing_accepts_the_right_file_and_refuses_the_wrong_one(folder):
    h5 = _two(folder)
    a, b = apex.list_measurements(h5)
    up = make_up(folder / "p.up1", guid_bytes=GUID_A_BYTES, n=15)
    assert apex.check_pairing(a, up) == "guid"
    with pytest.raises(apex.ApexError) as ei:
        apex.check_pairing(b, up)
    assert ei.value.code == "apexGuidMismatch"


def test_check_pairing_refuses_a_file_with_the_wrong_number_of_patterns(folder):
    a = apex.list_measurements(_two(folder))[0]
    up = make_up(folder / "short.up1", guid_bytes=GUID_A_BYTES, n=14)
    with pytest.raises(apex.ApexError) as ei:
        apex.check_pairing(a, up)
    assert ei.value.code == "apexPatternCountMismatch"


def test_a_file_without_a_guid_is_paired_by_count(folder):
    a = apex.list_measurements(_two(folder))[0]
    up = make_up(folder / "v3.up1", guid_bytes=None, version=3, n=15)
    assert apex.check_pairing(a, up) == "count"


def test_a_missing_measurement_is_a_coded_refusal(folder):
    with pytest.raises(apex.ApexError) as ei:
        apex.read_measurement(_two(folder), "11111111-1111-1111-1111-111111111111")
    assert ei.value.code == "apexMeasurementNotFound"
    with pytest.raises(apex.ApexError) as ei:
        apex.read_measurement(_two(folder), "not a guid")
    assert ei.value.code == "apexBadId"


def test_the_point_table_is_reshaped_in_index_order(folder):
    a = apex.list_measurements(_two(folder))[0]
    iq = apex.point_map(a, "IQ")
    assert iq.shape == (3, 5)
    assert iq[2, 3] == 10.0 * (2 * 5 + 3)       # point (row 2, col 3) is record 13


def test_previews_are_small_grey_pngs(folder):
    from PIL import Image
    a = apex.list_measurements(_two(folder))[0]
    for kind in ("sem", "iq"):
        img = Image.open(io.BytesIO(apex.preview_png(a, kind)))
        assert img.mode == "L" and max(img.size) <= 320
    assert Image.open(io.BytesIO(apex.preview_png(a, "iq"))).size == (5, 3)
    with pytest.raises(apex.ApexError) as ei:
        apex.preview_png(a, "nope")
    assert ei.value.code == "apexNoPreview"


def _png_array(data):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)))


def test_a_large_overview_image_is_thinned_and_keeps_its_grey_values(tmp_path):
    # Two constant halves, 10 rows x 800 columns: columns 0..399 are 50, columns 400..799 are 200.
    # Half the pixels are 50 and half 200, so the 2nd percentile is 50 and the 98th is 200:
    # 50 scales to 0 and 200 to 1, i.e. 0 and 255 in 8 bit.
    # The longer side is 800, 800 / 320 = 2.5, so every ceil(2.5) = 3rd pixel is kept:
    # columns 0, 3, ..., 798 = ceil(800 / 3) = 267 columns, rows 0, 3, 6, 9 = 4 rows.
    # The kept columns below 400 are 0, 3, ..., 399 = 134 columns (399 = 3 * 133), the other
    # 267 - 134 = 133 columns are 200.
    sem = np.zeros((10, 800), dtype=np.uint8)
    sem[:, :400] = 50
    sem[:, 400:] = 200
    h5 = make_apex_h5(tmp_path / "big.h5", [measurement_spec(sem_data=sem)])
    out = _png_array(apex.preview_png(apex.list_measurements(h5)[0], "sem"))
    assert out.shape == (4, 267)                       # (rows, columns)
    assert (out[:, :134] == 0).all() and (out[:, 134:] == 255).all()


def test_a_wide_point_grid_is_thinned_and_the_iq_ramp_is_scaled_by_the_percentiles(tmp_path):
    # 2 rows x 700 columns, IQ of point i is 10 * i (i = 0..1399).
    # Percentiles (linear interpolation): 2 % at index 0.02 * 1399 = 27.98 -> 279.8,
    # 98 % at index 0.98 * 1399 = 1371.02 -> 13710.2; span 13430.4.
    # Longer side 700, 700 / 320 = 2.19 -> every 3rd: 234 columns (0, 3, ..., 699), 1 row (row 0).
    # Output column j is source column 3 * j, i.e. IQ 30 * j:
    #   j = 0:   IQ 0 is below 279.8 -> 0
    #   j = 117: IQ 3510 -> (3510 - 279.8) / 13430.4 * 255 = 0.24052 * 255 = 61.3 -> 61
    #   j = 233: IQ 6990 -> (6990 - 279.8) / 13430.4 * 255 = 0.49963 * 255 = 127.4 -> 127
    h5 = make_apex_h5(tmp_path / "wide.h5", [measurement_spec(n_rows=2, n_cols=700)])
    out = _png_array(apex.preview_png(apex.list_measurements(h5)[0], "iq"))
    assert out.shape == (1, 234)
    assert (out[0, 0], out[0, 117], out[0, 233]) == (0, 61, 127)


def test_point_map_refuses_a_table_that_does_not_fit_the_grid(tmp_path):
    # a 3 x 5 grid needs 15 records; the table holds 14
    h5 = make_apex_h5(tmp_path / "short_table.h5", [measurement_spec(n_table=14)])
    a = apex.list_measurements(h5)[0]
    assert a.n_points == 14
    with pytest.raises(apex.ApexError) as ei:
        apex.point_map(a, "IQ")
    assert ei.value.code == "apexPatternCountMismatch"


def test_a_file_without_any_oim_map_is_a_coded_refusal(tmp_path):
    import h5py
    p = tmp_path / "empty.h5"
    with h5py.File(p, "w") as f:
        f.create_group("TOP/Sample/Area 1/Something Else")
    with pytest.raises(apex.ApexError) as ei:
        apex.list_measurements(p)
    assert ei.value.code == "apexNoMeasurement"


def test_a_file_that_cannot_be_read_is_a_coded_refusal(tmp_path):
    import h5py
    garbage = tmp_path / "garbage.h5"
    garbage.write_bytes(b"this is not an HDF5 file" * 10)
    with pytest.raises(apex.ApexError) as ei:
        apex.list_measurements(garbage)
    assert ei.value.code == "apexUnreadable"
    # a measurement group that lacks the camera group the reader needs
    h5 = make_apex_h5(tmp_path / "broken.h5", [measurement_spec()])
    with h5py.File(h5, "r+") as f:
        del f["TOP/Sample/Area 1/OIM Map 1/Camera"]
    with pytest.raises(apex.ApexError) as ei:
        apex.list_measurements(h5)
    assert ei.value.code == "apexUnreadable"


def test_a_measurement_without_an_overview_image_has_no_sem_preview(tmp_path):
    h5 = make_apex_h5(tmp_path / "nosem.h5", [measurement_spec(sem=False)])
    a = apex.list_measurements(h5)[0]
    assert not a.has_sem and apex.sem_image(a) is None
    with pytest.raises(apex.ApexError) as ei:
        apex.preview_png(a, "sem")
    assert ei.value.code == "apexNoPreview"


def test_a_broken_apex_file_next_to_a_pattern_file_does_not_stop_the_sidecar_search(folder, caplog):
    import logging
    import h5py
    _two(folder)
    # "aborted_session.h5" sorts before "session.h5", so the search meets it first. It is an APEX
    # file (it has a measurement) whose measurement lacks the camera group the reader needs.
    broken = make_apex_h5(folder / "aborted_session.h5", [
        measurement_spec(id="22222222-2222-2222-2222-222222222222")])
    with h5py.File(broken, "r+") as f:
        del f["TOP/Sample/Area 1/OIM Map 1/Camera"]
    up = make_up(folder / "p.up1", guid_bytes=GUID_B_BYTES, hex_flag=0, n=8, nx=4, ny=2)
    with caplog.at_level(logging.WARNING, logger="edax_apex"):
        m = apex.find_apex_sidecar(up)
    assert m is not None and m.id == GUID_B
    assert any("aborted_session.h5" in r.getMessage() and "apexUnreadable" in r.getMessage()
               for r in caplog.records)
    # the explicit path stays strict
    with pytest.raises(apex.ApexError) as ei:
        apex.list_measurements(broken)
    assert ei.value.code == "apexUnreadable"


def test_a_nil_guid_is_no_guid_and_the_file_is_paired_by_count(folder):
    a = apex.list_measurements(_two(folder))[0]
    up = make_up(folder / "nil.up1", guid_bytes=bytes(16), n=15)      # 16 zero bytes at 42..58
    assert apex.up_guid(up) is None
    assert apex.check_pairing(a, up) == "count"


def test_an_hdf5_file_that_will_not_open_is_reported_not_silently_skipped(folder, caplog, monkeypatch):
    import logging
    import h5py
    h5 = _two(folder)

    def locked(*args, **kwargs):
        raise OSError("unable to lock file")

    monkeypatch.setattr(h5py, "File", locked)
    with caplog.at_level(logging.WARNING, logger="edax_apex"):
        assert apex.is_apex_file(h5) is False
    assert any("session.h5" in r.getMessage() and "OSError" in r.getMessage() for r in caplog.records)
