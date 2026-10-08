"""The facts a batch ``.ang`` / ``.ctf`` has to carry.

Same contract as ``tests/test_ang_export.py`` for the interactive export, here
for the checkpoint-based exporter: real lattice constants (never the 1 1 1 90 90
90 placeholder), the Euler frame, the scan origin, the acquisition geometry of
the source scan and the statement about the EDS.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from backend.api.services import ang_export
from backend.api.services.checkpoint_writer import CheckpointWriter
from backend.api.services.result_exporter import export_all, export_ang_ctf

AL_CIF = """data_Al
_cell_length_a 4.0495
_cell_length_b 4.0495
_cell_length_c 4.0495
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_space_group_IT_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Al1 Al 0 0 0 1
"""


def _make(tmp_path, *, lattice=None, phase_file="Al.cif", vendor_oxford=True):
    source = tmp_path / "Scan.h5oina"
    with h5py.File(source, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"Oxford Instruments"]))
        h = f.create_group("1/EBSD/Header")
        h.create_dataset("Beam Voltage", data=np.float32(20.0))
        h.create_dataset("Tilt Angle", data=np.float32(np.deg2rad(70.0)))
        h.create_dataset("Scanning Rotation Angle", data=np.float32(np.pi))
        h.create_dataset("Working Distance", data=np.float32(16.5))
        f.create_group("1/EBSD/Data")
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="facts", method="spherical",
                     step_size_um=0.4)
    meta = {"phase_file": str(phase_file), "ci_mean": 0.7, "duration_sec": 1.0,
            "space_group": 225, "point_group": "m-3m"}
    if lattice is not None:
        meta["lattice_constants"] = lattice
    rng = np.random.default_rng(1)
    cw.write_phase_result(
        "Al", np.full((3, 4), 0.7, np.float32),
        rng.uniform(0, 3, (3, 4, 3)).astype(np.float32), meta)
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


def _lattice_line(text):
    return [ln for ln in text.splitlines() if ln.startswith("# LatticeConstants")]


def test_recorded_lattice_reaches_ang_and_ctf(tmp_path):
    source, cw = _make(tmp_path, lattice=[4.0495, 4.0495, 4.0495, 90, 90, 90],
                       phase_file="gone.sht")
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"),
                              source_h5_path=str(source))
    (line,) = _lattice_line(Path(ang).read_text(encoding="utf-8"))
    assert [float(v) for v in line.split()[2:]] == pytest.approx(
        [4.0495, 4.0495, 4.0495, 90, 90, 90], abs=1e-3)
    ctf_text = Path(ctf).read_text(encoding="utf-8")
    phase_row = ctf_text.splitlines()[next(
        i for i, ln in enumerate(ctf_text.splitlines()) if ln.startswith("Phases")) + 1]
    assert phase_row.startswith("4.050;4.050;4.050\t90.0;90.0;90.0\tAl")


def test_lattice_is_read_from_the_phase_file_when_not_recorded(tmp_path):
    cif = tmp_path / "Al.cif"
    cif.write_text(AL_CIF)
    source, cw = _make(tmp_path, phase_file=cif)
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"),
                            source_h5_path=str(source))
    (line,) = _lattice_line(Path(ang).read_text(encoding="utf-8"))
    assert float(line.split()[2]) == pytest.approx(4.0495, abs=1e-3)


def test_an_unknown_lattice_is_refused_in_ang_and_ctf(tmp_path):
    source, cw = _make(tmp_path, phase_file="not_there.cif")
    with pytest.raises(ang_export.LatticeUnknown, match="Al"):
        export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"),
                       source_h5_path=str(source))
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o2"),
                     formats=["ang", "ctf"])
    assert res["ang"] is None and res["ctf"] is None
    assert "lattice" in res["_errors"]["ang_ctf"].lower()


def test_headers_state_frame_origin_geometry_and_assignment(tmp_path):
    source, cw = _make(tmp_path, lattice=[4.0495] * 3 + [90.0] * 3)
    scan = {"scan_row_offset": 43, "scan_col_offset": 194, "scan_shape": [226, 301]}
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"),
                              source_h5_path=str(source), scan_provenance=scan)
    a = Path(ang).read_text(encoding="utf-8")
    c = Path(ctf).read_text(encoding="utf-8")
    prj = next(ln for ln in c.splitlines() if ln.startswith("Prj\t"))
    assert "Subtract 90 degrees" in a
    assert "Oxford/Aztec frame" in prj and "Euler1(.ang) = Euler1(this file) + 90" in prj
    for text in (a, prj):
        assert "EDAX TSL" in text
        assert "row 43, column 194" in text and "226 x 301" in text
        assert "x 77.600 um" in text and "y 17.200 um" in text    # 194*0.4, 43*0.4
        assert "scanning_rotation_angle_deg=180.000" in text
        assert "sample_tilt_deg=70.000" in text
        assert "EDS chemistry prior: no" in text
    assert "phi1(.h5)" in a
    assert "\t" not in prj.split("\t", 1)[1], "the Prj text must stay one tab-free field"
    wd = [ln for ln in a.splitlines() if ln.startswith("# WorkingDistance")]
    assert float(wd[0].split()[-1]) == pytest.approx(16.5)


def test_without_scan_provenance_the_origin_is_not_claimed(tmp_path):
    source, cw = _make(tmp_path, lattice=[4.0495] * 3 + [90.0] * 3)
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"),
                            source_h5_path=str(source))
    text = Path(ang).read_text(encoding="utf-8")
    assert "not recorded by this export" in text
    assert "not a crop" not in text


def test_export_all_hands_scan_and_detector_shape_on(tmp_path):
    source, cw = _make(tmp_path, lattice=[4.0495] * 3 + [90.0] * 3)
    res = export_all(
        str(source), cw.checkpoint_path, str(tmp_path / "o"),
        formats=["ang"], pc=[0.5, 0.4, 0.6], detector_shape=(60, 80),
        scan_provenance={"scan_row_offset": 5, "scan_col_offset": 7,
                         "scan_shape": [90, 120]})
    text = Path(res["ang"]).read_text(encoding="utf-8")
    assert "row 5, column 7" in text
    assert float(re.search(r"^# y-star\s+(\S+)", text, re.M).group(1)) == \
        pytest.approx(1 - 0.4, abs=1e-6)


def test_the_declared_unit_decides_not_the_size():
    """kikuchipy keeps master-pattern structures in nm, diffpy/orix CIFs are in
    angstrom. The unit comes from where the structure came from; a cell of
    2.5 nm is 25 angstrom, not 2.5."""
    nm = ang_export.to_angstrom
    assert nm((0.4049, 0.4049, 0.4049, 90, 90, 90), "nm")[0] == pytest.approx(4.049)
    assert nm((2.5, 3.0, 2.2, 90, 90, 90), "nm")[:3] == pytest.approx((25.0, 30.0, 22.0))
    assert nm((4.049, 4.049, 4.049, 90, 90, 90), "angstrom")[0] == pytest.approx(4.049)
    assert nm((15.488, 8.087, 12.477, 90, 107.7, 90), "angstrom")[1] == pytest.approx(8.087)
    # a size no crystal has, for the unit it was declared in: refuse
    with pytest.raises(ang_export.LatticeUnknown, match="unit"):
        nm((0.4049, 0.4049, 0.4049, 90, 90, 90), "angstrom")


def test_pc_conversion_is_the_kikuchipy_tsl_convention():
    # wide detector (60 rows x 80 columns)
    assert ang_export.pc_tsl_from_bruker([0.4, 0.2, 0.6], (60, 80)) == \
        pytest.approx((0.4, 0.8, 0.6))
    # tall detector: z is rescaled by nrows / min(nrows, ncols)
    assert ang_export.pc_tsl_from_bruker([0.4, 0.2, 0.6], (80, 60)) == \
        pytest.approx((0.4, 0.8, 0.6 * 80 / 60))
    assert ang_export.pc_tsl_from_bruker([0.4, 0.2, 0.6], None) is None


@pytest.mark.parametrize("fmt", ["h5_light", "h5_rich"])
def test_batch_h5_exports_carry_the_same_statements(tmp_path, fmt):
    source, cw = _make(tmp_path, lattice=[4.0495] * 3 + [90.0] * 3)
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=[fmt])
    path = res[fmt]
    assert path, res.get("_errors")
    with h5py.File(path, "r") as f:
        idx = f["Indexing"]
        assert str(idx.attrs["format_version"]) == "1.4"
        assert int(idx.attrs["eds_chemistry_prior"]) == 0
        assert "EDS chemistry prior: no" in str(idx.attrs["phase_assignment"])
        assert "X||a" in str(idx.attrs["crystal_reference_frame"])
        ph = idx["Phases/1"].attrs
        assert int(ph["space_group"]) == 225
        np.testing.assert_allclose(ph["lattice_constants"],
                                   [4.0495] * 3 + [90.0] * 3, atol=1e-3)
        assert str(ph["lattice_length_unit"]) == "angstrom"
        acq = f["Acquisition"].attrs
        assert float(acq["Scanning Rotation Angle"]) == pytest.approx(np.pi, abs=1e-5)
        assert int(acq["applied_to_euler_angles"]) == 0


def _ctf_rows(path):
    rows = [ln.split("\t") for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln and ln[0].isdigit() and ln.count("\t") >= 10]
    return np.array(rows, dtype=float)


@pytest.mark.parametrize("fmt_vendor, shifted", [(b"Oxford Instruments", True),
                                                 (b"EDAX", False)])
def test_ctf_holds_the_same_euler_angles_as_the_light_h5(tmp_path, fmt_vendor, shifted):
    """A .ctf is an Oxford format: its angles are the ones the .h5 holds (the
    source system's frame), not the TSL-frame ones of the .ang."""
    source, cw = _make(tmp_path, lattice=[4.0495] * 3 + [90.0] * 3)
    with h5py.File(source, "r+") as f:
        del f["Manufacturer"]
        f.create_dataset("Manufacturer", data=np.array([fmt_vendor]))
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["h5_light", "ang", "ctf"])
    with h5py.File(res["h5_light"], "r") as f:
        e_h5 = np.degrees(np.asarray(f["Indexing/Assignment/euler_angles"])).reshape(-1, 3)
    e_ctf = _ctf_rows(res["ctf"])[:, 5:8]
    e_ang = np.degrees(load_ang_rows(res["ang"])[:, :3])
    d = (e_ctf - e_h5 + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(d, 0.0, atol=2e-2)
    # and relative to the TSL-frame .ang: phi1 - 90 for Oxford, identical for EDAX
    d2 = (e_ctf - e_ang + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(d2[:, 0], -90.0 if shifted else 0.0, atol=2e-2)
    np.testing.assert_allclose(d2[:, 1:], 0.0, atol=2e-2)
    prj = next(ln for ln in Path(res["ctf"]).read_text(encoding="utf-8").splitlines()
               if ln.startswith("Prj\t"))
    if shifted:
        assert "Oxford/Aztec frame" in prj and "same Euler angles as the .h5" in prj
    assert "TSL" in prj


def load_ang_rows(path):
    rows = [ln.split() for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    return np.array(rows, dtype=float)
