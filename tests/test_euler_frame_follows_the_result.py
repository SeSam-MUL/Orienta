"""The Euler frame of an export follows the scan the result was computed on, not the file in the viewer.

The vendor of a result is stamped when its run starts (``metadata["source_vendor"]``) and written to
``/Indexing@source_vendor``, which an import restores. Without a stamp the vendor is read from the
result's own ``source_file``, and from nothing else: an Oxford file that is open while an EDAX result is
exported neither rotates the Euler angles of the ``.h5`` and ``.ctf`` nor labels them Oxford, and
none of its header values is stated in the files.

Hand values: every orientation in these results is the identity (Euler 0, 0, 0). Written in the
Oxford/Aztec frame, which is the native one turned by 90 degrees about the sample normal, phi1 + phi2
is a quarter turn (the documented relation of the files: phi1(.h5) = phi1(.ang) - 90 degrees, so
270 degrees mod 360 for phi1 + phi2 with Phi = 0); in the EDAX frame it is 0.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from backend.api.main import app
from backend.api.routes import indexing as indexing_mod
from backend.api.services.scan_geometry import ScanGeometry
from indexing_controller import IndexingMethod, IndexingResult

ROWS, COLS = 3, 4
N = ROWS * COLS
OXFORD_SCANNING_ROTATION_RAD = np.deg2rad(37.0)       # 37 degrees, by hand


@pytest.fixture(autouse=True)
def restore_indexing_registry():
    saved = dict(indexing_mod._result_registry)
    saved_active = indexing_mod._active_result_id
    try:
        yield
    finally:
        indexing_mod._result_registry.clear()
        indexing_mod._result_registry.update(saved)
        indexing_mod._active_result_id = saved_active


def _write_oxford(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"Oxford Instruments"]))
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("Scanning Rotation Angle", data=np.float32(OXFORD_SCANNING_ROTATION_RAD))
        hdr.create_dataset("Tilt Angle", data=np.float32(np.deg2rad(70.0)))
        hdr.create_dataset("Beam Voltage", data=np.float32(20.0))
        f.create_group("1/EBSD/Data")
    return str(path)


def _write_edax(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"EDAX"]))
        f.create_group("Scan1/EBSD/Header")
        f.create_group("Scan1/EBSD/Data")
    return str(path)


@pytest.fixture
def files(tmp_path):
    return {"oxford": _write_oxford(tmp_path / "oxford.h5oina"), "edax": _write_edax(tmp_path / "edax.h5")}


def _result(source_file, *, hex_scan, vendor_stamp=None):
    from diffpy.structure import Lattice, Structure
    phase = Phase(name="Al", point_group="m-3m",
                  structure=Structure(lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)))
    ys, xs = np.mgrid[0:ROWS, 0:COLS]
    xmap = CrystalMap(rotations=Rotation.from_euler(np.zeros((N, 3))),
                      phase_id=np.ones(N, dtype=int), x=xs.ravel().astype(float),
                      y=ys.ravel().astype(float), phase_list=PhaseList(phases=[phase], ids=[1]))
    md = {"step_size_um": 2.0, "source_file": source_file}
    if hex_scan:
        md["scan_geometry"] = ScanGeometry.hex(
            ROWS, COLS, 2.0, odd_row_shift=0.5, shift_confirmed=True).to_dict()
    if vendor_stamp is not None:
        md["source_vendor"] = vendor_stamp
    return IndexingResult(
        xmap=xmap, selection_mask=np.ones(N, dtype=bool), original_shape=(ROWS, COLS),
        method=IndexingMethod.SPHERICAL, confidence_scores=np.full(N, 0.8, np.float32), metadata=md)


def _open_file(monkeypatch, path, result):
    from backend.api.routes import ebsd_viewer
    monkeypatch.setitem(indexing_mod._result_registry, "frame_test", result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", "frame_test")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", path, raising=False)


def _export(fmt, out):
    resp = TestClient(app).post("/api/indexing/export", json={"format": fmt, "filename": str(out)})
    assert resp.status_code == 200, resp.text
    return Path(out)


def _comments(path, key):
    return [ln for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.startswith(f"# {key}")]


def _phi_sum_deg(h5_path):
    with h5py.File(h5_path, "r") as f:
        eul = np.degrees(f["Indexing/euler_angles"][()]).reshape(-1, 3)
    return np.mod(eul[:, 0] + eul[:, 2], 360.0), eul[:, 1]


def _attr(path, name):
    with h5py.File(path, "r") as f:
        v = f["Indexing"].attrs[name]
    return v.decode() if isinstance(v, bytes) else str(v)


# ------------------------------------------------------------ an Oxford file open, an EDAX result

@pytest.mark.parametrize("hex_scan", [True, False])
def test_light_h5_of_an_unstamped_result_does_not_take_the_open_files_vendor(
        monkeypatch, tmp_path, files, hex_scan):
    """The result ran on the EDAX file and states no vendor; an Oxford file is open. The vendor is
    unknown (not Oxford): the Euler angles stay native and the label says so."""
    _open_file(monkeypatch, files["oxford"], _result(files["edax"], hex_scan=hex_scan))
    out = _export("h5_light", tmp_path / "light.h5")
    assert _attr(out, "source_vendor") == "unknown"
    assert "native" in _attr(out, "orientation_reference_frame")
    phi, big_phi = _phi_sum_deg(out)
    np.testing.assert_allclose(phi, 0.0, atol=1e-4)                  # not a quarter turn
    np.testing.assert_allclose(big_phi, 0.0, atol=1e-4)


@pytest.mark.parametrize("hex_scan", [True, False])
def test_ang_of_an_unstamped_result_does_not_state_the_open_files_vendor_or_header(
        monkeypatch, tmp_path, files, hex_scan):
    _open_file(monkeypatch, files["oxford"], _result(files["edax"], hex_scan=hex_scan))
    out = _export("ang", tmp_path / "r.ang")
    frame = " ".join(_comments(out, "ORIENTA_EULER_FRAME"))
    assert "source system was not recorded" in frame
    assert "The scan came from an Oxford" not in frame and "Subtract 90 degrees" not in frame
    assert not _comments(out, "ORIENTA_MTEX")                      # that line is for an Oxford source
    assert not _comments(out, "ORIENTA_ACQUISITION")                 # nothing of the open file's header
    assert "scanning_rotation_angle" not in Path(out).read_text(encoding="utf-8")


def _acquisition_keys(path):
    with h5py.File(path, "r") as f:
        return sorted(f["Acquisition"].attrs) if "Acquisition" in f else []


def test_light_h5_states_nothing_of_the_open_files_header_or_location(monkeypatch, tmp_path, files):
    """The open Oxford file's header (scanning rotation 37 degrees, 20 kV) and its path do not go
    into the file of an EDAX result: /Acquisition holds none of its values, /SourceReference names
    no source."""
    _open_file(monkeypatch, files["oxford"], _result(files["edax"], hex_scan=True))
    out = _export("h5_light", tmp_path / "light.h5")
    assert "scanning_rotation_angle_deg" not in _acquisition_keys(out)
    assert "beam_voltage_kv" not in _acquisition_keys(out)
    with h5py.File(out, "r") as f:
        assert "source_file_path" not in f["SourceReference"].attrs


def test_light_h5_of_the_open_file_keeps_its_header_and_location(monkeypatch, tmp_path, files):
    """The same result when its own source IS the open file: unchanged (hand: 37 degrees, 20 kV)."""
    _open_file(monkeypatch, files["oxford"], _result(files["oxford"], hex_scan=False))
    out = _export("h5_light", tmp_path / "light.h5")
    with h5py.File(out, "r") as f:
        acq = f["Acquisition"].attrs
        assert float(acq["scanning_rotation_angle_deg"]) == pytest.approx(37.0, abs=1e-4)
        assert float(acq["beam_voltage_kv"]) == pytest.approx(20.0)
        assert Path(f["SourceReference"].attrs["source_file_path"]).name == "oxford.h5oina"


def test_rich_h5_states_nothing_of_the_open_files_header(monkeypatch, tmp_path, files):
    """A square result of the EDAX file while an Oxford file is open: the rich file is a copy of the
    open file (as it always was for a square result) but its /Acquisition and its vendor are not the
    open file's."""
    _open_file(monkeypatch, files["oxford"], _result(files["edax"], hex_scan=False))
    out = _export("h5", tmp_path / "rich.h5")
    with h5py.File(out, "r") as f:
        written = dict(f["Acquisition"].attrs) if "Acquisition" in f else {}
    assert "scanning_rotation_angle_deg" not in written
    assert _attr(out, "source_vendor") == "unknown"


def test_a_stamped_edax_result_keeps_its_vendor_whatever_file_is_open(monkeypatch, tmp_path, files):
    _open_file(monkeypatch, files["oxford"], _result(files["edax"], hex_scan=True, vendor_stamp="edax"))
    light = _export("h5_light", tmp_path / "light.h5")
    ang = _export("ang", tmp_path / "r.ang")
    ctf = _export("ctf", tmp_path / "r.ctf")
    assert _attr(light, "source_vendor") == "edax"
    np.testing.assert_allclose(_phi_sum_deg(light)[0], 0.0, atol=1e-4)
    frame = " ".join(_comments(ang, "ORIENTA_EULER_FRAME"))
    assert "EDAX system, whose own frame this is" in frame and "Oxford" not in frame
    # the .ctf holds the same angles as the .h5: native, 0 + 0 + 0
    rows = [ln.split("\t") for ln in Path(ctf).read_text(encoding="utf-8").splitlines()
            if re.match(r"^\d+\t", ln) and len(ln.split("\t")) >= 11]
    assert rows and all(abs(float(r[5])) < 1e-3 and abs(float(r[7])) < 1e-3 for r in rows)
    prj = next(ln for ln in Path(ctf).read_text(encoding="utf-8").splitlines() if ln.startswith("Prj\t"))
    assert "EDAX TSL frame of the source scan" in prj


def test_a_stamped_oxford_result_is_rotated_in_all_three_files(monkeypatch, tmp_path, files):
    """The same result stamped Oxford (the file open is EDAX): .h5 and .ctf hold the quarter turn,
    the .ang the native angles, and each says so."""
    _open_file(monkeypatch, files["edax"], _result(files["oxford"], hex_scan=True, vendor_stamp="oxford"))
    light = _export("h5_light", tmp_path / "light.h5")
    ang = _export("ang", tmp_path / "r.ang")
    ctf = _export("ctf", tmp_path / "r.ctf")
    assert _attr(light, "source_vendor") == "oxford"
    assert _phi_sum_deg(light)[0] == pytest.approx(270.0, abs=1e-3)       # -90 degrees, see above
    assert "Oxford Instruments / Bruker system" in " ".join(_comments(ang, "ORIENTA_EULER_FRAME"))
    rows = [ln.split("\t") for ln in Path(ctf).read_text(encoding="utf-8").splitlines()
            if re.match(r"^\d+\t", ln) and len(ln.split("\t")) >= 11]
    assert all((float(r[5]) + float(r[7])) % 360.0 == pytest.approx(270.0, abs=1e-2) for r in rows)
    assert "Oxford/Aztec frame of the source scan" in next(
        ln for ln in Path(ctf).read_text(encoding="utf-8").splitlines() if ln.startswith("Prj\t"))


# ----------------------------------------- the same file open: nothing changes for a square result

@pytest.mark.parametrize("fmt", ["h5_light", "h5"])
def test_a_square_result_of_the_open_oxford_file_is_exported_as_before(
        monkeypatch, tmp_path, files, fmt):
    """No stamp, the result's source_file is the file that is open: the vendor is read from it."""
    _open_file(monkeypatch, files["oxford"], _result(files["oxford"], hex_scan=False))
    out = _export(fmt, tmp_path / f"{fmt}.h5")
    assert _attr(out, "source_vendor") == "oxford"
    assert "vendor_stored" in _attr(out, "orientation_reference_frame")
    assert _phi_sum_deg(out)[0] == pytest.approx(270.0, abs=1e-3)


def test_the_ang_of_a_square_result_of_the_open_oxford_file_is_exported_as_before(
        monkeypatch, tmp_path, files):
    _open_file(monkeypatch, files["oxford"], _result(files["oxford"], hex_scan=False))
    out = _export("ang", tmp_path / "r.ang")
    frame = " ".join(_comments(out, "ORIENTA_EULER_FRAME"))
    assert "Subtract 90 degrees from phi1" in frame
    mtex = " ".join(_comments(out, "ORIENTA_MTEX"))
    assert "Scanning Rotation Angle of 37 degrees" in mtex            # the header of the file, by hand
    assert "scanning_rotation_angle_deg=37.000" in " ".join(_comments(out, "ORIENTA_ACQUISITION"))


# --------------------------------------------------------------- a .up1/.up2 source is EDAX

@pytest.mark.parametrize("suffix", [".up1", ".up2"])
def test_an_up1_or_up2_source_is_edax_and_the_angles_stay_native(monkeypatch, tmp_path, suffix):
    """No stamp, the open file is the result's own source and not HDF5: APEX pairs are EDAX files.
    Hand: the offset of the EDAX frame is the identity, so only the label and the text differ."""
    up = tmp_path / f"scan{suffix}"
    up.write_bytes(b"\x01\x00\x00\x00" + b"\x00" * 64)
    _open_file(monkeypatch, str(up), _result(str(up), hex_scan=True))
    light = _export("h5_light", tmp_path / "light.h5")
    ang = _export("ang", tmp_path / "r.ang")
    assert _attr(light, "source_vendor") == "edax"
    np.testing.assert_allclose(_phi_sum_deg(light)[0], 0.0, atol=1e-4)
    assert "EDAX system, whose own frame this is" in " ".join(_comments(ang, "ORIENTA_EULER_FRAME"))


@pytest.mark.parametrize("name, vendor", [("a.up1", "edax"), ("a.UP2", "edax"), ("a.dat", "")])
def test_the_batch_writers_read_an_up_file_as_edax(tmp_path, name, vendor):
    from backend.api.services.result_exporter import _read_vendor_from_h5
    path = tmp_path / name
    path.write_bytes(b"\x01\x00\x00\x00" + b"\x00" * 16)
    assert _read_vendor_from_h5(str(path)) == vendor


def test_the_batch_writers_read_the_manufacturer_of_an_hdf5_file(files):
    from backend.api.services.result_exporter import _read_vendor_from_h5
    assert _read_vendor_from_h5(files["oxford"]) == "oxford"
    assert _read_vendor_from_h5(files["edax"]) == "edax"


def test_the_edax_frame_is_the_identity():
    from backend.api.services.orientation_frame import export_frame_offset
    assert float(export_frame_offset("edax").angle.max()) == 0.0


# -------------------------------------------------------------- the vendor is stamped at run start

@pytest.mark.parametrize("name, writer, vendor", [
    ("scan.h5oina", _write_oxford, "oxford"),
    ("scan.h5", _write_edax, "edax"),
])
def test_the_run_start_stamp_holds_the_vendor_of_the_file_that_is_open(monkeypatch, tmp_path, name, writer, vendor):
    from backend.api.routes import ebsd_viewer
    path = writer(tmp_path / name)
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", path, raising=False)
    stamp = indexing_mod._scan_stamp_at_run_start("ds", None)
    assert stamp["source_vendor"] == vendor


@pytest.mark.parametrize("suffix, vendor", [(".up1", "edax"), (".up2", "edax"), (".xyz", "")])
def test_the_run_start_stamp_of_a_file_that_is_not_hdf5(monkeypatch, tmp_path, suffix, vendor):
    from backend.api.routes import ebsd_viewer
    path = tmp_path / f"scan{suffix}"
    path.write_bytes(b"not hdf5")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", str(path), raising=False)
    assert indexing_mod._scan_stamp_at_run_start("ds", None)["source_vendor"] == vendor


def test_the_stamp_survives_the_store_and_an_open_file_of_another_vendor(monkeypatch, tmp_path, files):
    """Stamped at the start with the EDAX file open, stored later with an Oxford file open."""
    from backend.api.routes import ebsd_viewer
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", files["edax"], raising=False)
    stamp = indexing_mod._scan_stamp_at_run_start("ds", None)
    result = _result(files["edax"], hex_scan=True)
    del result.metadata["source_file"]
    indexing_mod._seed_scan_stamp(result, stamp)
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", files["oxford"], raising=False)
    indexing_mod._store_result(result, "spherical")
    assert result.metadata["source_vendor"] == "edax"
    assert result.metadata["source_file"] == files["oxford"]          # the store tags the file open then


# -------------------------------------------------------------------------- the import

@pytest.mark.parametrize("written, restored", [("edax", "edax"), ("oxford", "oxford"),
                                               ("bruker", "bruker"), ("unknown", None)])
def test_an_import_restores_the_vendor_the_file_states(tmp_path, written, restored):
    path = tmp_path / "result.h5"
    with h5py.File(path, "w") as f:
        f.create_group("Indexing").attrs["source_vendor"] = written
    assert indexing_mod._read_source_vendor(str(path)) == restored


@pytest.mark.parametrize("vendor, phi_sum", [("oxford", 270.0), ("edax", 0.0)])
def test_an_exported_result_that_is_imported_keeps_its_vendor_and_exports_the_same_angles(
        monkeypatch, tmp_path, files, vendor, phi_sum):
    """Export (stamped), import, export again with a file of the other vendor open: the vendor came
    back from the file, so the second file holds the same angles as the first."""
    _open_file(monkeypatch, files["edax" if vendor == "oxford" else "oxford"],
               _result(files["edax"], hex_scan=False, vendor_stamp=vendor))
    first = _export("h5_light", tmp_path / "first.h5")
    assert _phi_sum_deg(first)[0] == pytest.approx(phi_sum, abs=1e-3)
    resp = TestClient(app).post("/api/indexing/import-h5", json={"path": str(first)})
    assert resp.status_code == 200, resp.text
    imported = indexing_mod._result_registry[indexing_mod._active_result_id]
    assert imported.metadata["source_vendor"] == vendor
    from backend.api.routes import ebsd_viewer
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path",
                        files["oxford" if vendor == "edax" else "edax"], raising=False)
    second = _export("h5_light", tmp_path / "second.h5")
    assert _attr(second, "source_vendor") == vendor
    assert _phi_sum_deg(second)[0] == pytest.approx(phi_sum, abs=1e-3)


def test_an_import_of_a_file_without_the_attribute_states_no_vendor(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.create_group("Indexing")
    assert indexing_mod._read_source_vendor(str(path)) is None


# ------------------------------------------------- a refined result is the same scan as its parent

def _refine(parent):
    """Run the real orchestrator on ``parent`` (renderer and pattern reader stubbed, as in
    ``test_refinement_service``) and return the refined result it registered."""
    from unittest.mock import MagicMock, patch
    import torch
    from backend.api.services import refinement as rf

    parent.metadata.update(
        indexing_method="spherical", sht_paths_by_phase={1: "/fake.sht"},
        detector_geometry={"pat_height": 8, "pat_width": 8, "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.6,
                           "vendor": "Bruker", "pixel_size": 70.0, "sample_tilt": 70.0, "binning": 1})
    renderer = MagicMock()
    renderer.render_batch.return_value = torch.ones(N, 8, 8, dtype=torch.float32)
    renderer.render.return_value = torch.ones(8, 8, dtype=torch.float32)
    with patch.object(rf, "load_or_get_phase", return_value=MagicMock()), \
            patch.object(rf, "get_renderer", return_value=renderer), \
            patch.object(rf, "_get_experimental_pattern", return_value=np.ones((8, 8), np.float32)):
        rf.compute_full_refinement(parent, smoothness_lambda=0.01)
    return indexing_mod._result_registry[parent.metadata["refinement"]["refined_result_id"]]


@pytest.mark.parametrize("hex_scan", [True, False])
@pytest.mark.parametrize("vendor, phi_sum, frame_text", [
    # Hand values from the module docstring: Oxford frame = native turned by 90 degrees, so
    # phi1 + phi2 = 270 (mod 360) for the identity orientation; the EDAX frame is the identity.
    ("oxford", 270.0, "Oxford Instruments / Bruker system"),
    ("edax", 0.0, "EDAX system, whose own frame this is"),
])
def test_a_refined_result_exports_the_same_angles_and_frame_text_as_its_parent_with_another_file_open(
        monkeypatch, tmp_path, files, hex_scan, vendor, phi_sum, frame_text):
    """The parent is stamped with the vendor of the file it ran on; a file of the other vendor is
    open. The refined result must carry the stamp, or its export falls back to 'unknown' and a 90
    degree split opens between parent and child."""
    other = files["edax"] if vendor == "oxford" else files["oxford"]
    parent = _result(files["oxford" if vendor == "oxford" else "edax"], hex_scan=hex_scan,
                     vendor_stamp=vendor)
    _open_file(monkeypatch, other, parent)
    child = _refine(parent)
    assert child is not parent and child.metadata["is_refined"] is True
    assert child.metadata["source_vendor"] == vendor
    seen = {}
    for who, result in (("parent", parent), ("child", child)):
        _open_file(monkeypatch, other, result)
        light = _export("h5_light", tmp_path / f"{who}.h5")
        ang = _export("ang", tmp_path / f"{who}.ang")
        assert _attr(light, "source_vendor") == vendor
        assert _phi_sum_deg(light)[0] == pytest.approx(phi_sum, abs=1e-3)
        frame = " ".join(_comments(ang, "ORIENTA_EULER_FRAME"))
        assert frame_text in frame
        seen[who] = frame
    assert seen["child"] == seen["parent"]


def test_a_refined_result_of_a_parent_without_a_vendor_stamp_stays_without_one(monkeypatch, files):
    """An old result states no vendor: its child must not invent one (the vendor is then read from
    the same ``source_file`` for both)."""
    parent = _result(files["edax"], hex_scan=False)
    _open_file(monkeypatch, files["edax"], parent)
    assert "source_vendor" not in _refine(parent).metadata
