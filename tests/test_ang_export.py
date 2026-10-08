"""The ``.ang`` written by ``POST /api/indexing/export``.

What these pin, each against a fault seen on a real Oxford scan exported from
a spherical (SHT) run with the EDS chemistry prior on:

* the header said ``XSTEP 1.0`` and the x/y columns counted pixels, because the
  consensus / spherical CrystalMaps carry pixel coordinates and the exporter
  handed that map to orix unchanged -- every length came out wrong by the step
  size and every area by its square;
* every phase of such a map got ``LatticeConstants 1.000 1.000 1.000 90 90 90``,
  orix's placeholder for a phase without a structure;
* the Euler frame was not stated anywhere, so a reader could not tell that the
  ``.ang`` is in the EDAX TSL sample frame (the frame orix and kikuchipy use)
  while the ``.h5`` of the same result is in the Oxford/Aztec frame;
* the acquisition geometry of the source scan (Scanning Rotation Angle, tilt)
  and the crop offset did not reach the file;
* nothing said that the phase assignment had used the EDS chemistry prior;
* exporting over an existing file raised ``EOF when reading a line`` (orix asks
  on stdin before overwriting).
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.io import load as orix_load
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh

from backend.api.main import app
from backend.api.routes import indexing as indexing_mod
from indexing_controller import IndexingMethod, IndexingResult

N_ROWS, N_COLS = 4, 3
STEP_UM = 0.5

# Lattices the stubbed .sht reader hands back, in the units the real reader
# returns (nm and degrees).
_SHT_LATTICE_NM = {
    "al.sht": (0.4049, 0.4049, 0.4049, 90.0, 90.0, 90.0),
    "fe4al13.sht": (1.5488, 0.8087, 1.2477, 90.0, 107.669, 90.0),
}


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


def _euler_rad():
    n = N_ROWS * N_COLS
    return np.deg2rad(np.array(
        [[10.0 + i * 11.0, 20.0 + i, 30.0 + 2 * i] for i in range(n)]))


def _make_result(*, steps=None, scan_offset=None, vendor="oxford",
                 sht_paths=True, structured=False):
    """A spherical-style result: PIXEL coordinates, 1-based phase ids, phases
    without a structure -- the shape the real spherical-GPU map has."""
    n = N_ROWS * N_COLS
    rot = Rotation.from_euler(_euler_rad())
    if structured:
        from diffpy.structure import Lattice, Structure
        s = Structure(lattice=Lattice(4.049, 4.049, 4.049, 90, 90, 90))
        al = Phase(name="Al", point_group="m-3m", structure=s)
    else:
        al = Phase(name="Al", point_group="m-3m")
    fe = Phase(name="Fe4Al13", point_group="2/m")
    phases = PhaseList(phases=[al, fe], ids=[1, 2])
    pid = np.where(np.arange(n) % 3 == 0, 2, 1)
    xs = np.tile(np.arange(N_COLS, dtype=float), N_ROWS)      # pixels!
    ys = np.repeat(np.arange(N_ROWS, dtype=float), N_COLS)    # pixels!
    xmap = CrystalMap(rotations=rot, phase_id=pid, x=xs, y=ys,
                      phase_list=phases)
    md = {"step_size_um": STEP_UM, "source_vendor": vendor}
    if sht_paths:
        md["sht_paths_by_phase"] = {1: "/lib/Al/al.sht",
                                    2: "/lib/Fe4Al13/fe4al13.sht"}
    if steps is not None:
        md["provenance"] = {"schema": 1, "steps": steps}
    if scan_offset is not None:
        md.update({"scan_row_offset": scan_offset[0],
                   "scan_col_offset": scan_offset[1],
                   "scan_shape": [90, 120]})
    return IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool),
        original_shape=(N_ROWS, N_COLS), method=IndexingMethod.SPHERICAL,
        confidence_scores=np.linspace(0.3, 0.9, n).astype(np.float32),
        metadata=md)


def _activate(monkeypatch, result, source_path=None):
    rid = "test_ang_export"
    monkeypatch.setitem(indexing_mod._result_registry, rid, result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", rid)
    monkeypatch.setattr("backend.api.routes.ebsd_viewer._ebsd_file_path",
                        source_path, raising=False)


@pytest.fixture(autouse=True)
def stub_sht_reader(monkeypatch):
    """The real reader needs a .sht; the lattices above stand in for it."""
    from backend.api.services import ang_export

    def fake(path):
        key = Path(str(path)).name.lower()
        a, b, c, al, be, ga = _SHT_LATTICE_NM[key]
        return (a * 10.0, b * 10.0, c * 10.0, al, be, ga)

    monkeypatch.setattr(ang_export, "lattice_from_sht", fake, raising=False)
    yield


def _export(client, fmt, out):
    r = client.post("/api/indexing/export",
                    json={"format": fmt, "filename": str(out)})
    return r


def _header(path):
    return [ln.rstrip("\n") for ln in open(path, encoding="utf-8")
            if ln.startswith("#")]


def _data(path):
    rows = [ln.split() for ln in open(path, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    return np.array(rows, dtype=float)


def _kv(header, key):
    """All ``# KEY: value`` header lines named ``key``."""
    pat = re.compile(r"^#\s*" + re.escape(key) + r"\s*:\s*(.*)$")
    return [m.group(1) for ln in header if (m := pat.match(ln))]


# ---------------------------------------------------------------------------
# step size and coordinates
# ---------------------------------------------------------------------------

def test_step_and_coordinates_are_micrometres(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result())
    out = tmp_path / "r.ang"
    r = _export(TestClient(app), "ang", out)
    assert r.status_code == 200, r.text

    hdr = _header(out)
    assert float(_kv(hdr, "XSTEP")[0]) == pytest.approx(STEP_UM)
    assert float(_kv(hdr, "YSTEP")[0]) == pytest.approx(STEP_UM)
    assert int(_kv(hdr, "NCOLS_ODD")[0]) == N_COLS
    assert int(_kv(hdr, "NROWS")[0]) == N_ROWS

    d = _data(out)
    cols = np.tile(np.arange(N_COLS), N_ROWS)
    rows = np.repeat(np.arange(N_ROWS), N_COLS)
    np.testing.assert_allclose(d[:, 3], cols * STEP_UM, atol=1e-6)
    np.testing.assert_allclose(d[:, 4], rows * STEP_UM, atol=1e-6)


def test_exporting_over_an_existing_file_replaces_it(monkeypatch, tmp_path):
    """orix asks on stdin before overwriting: a server has no stdin (EOFError ->
    HTTP 500), and under a captured stdin it silently keeps the OLD file."""
    first = _make_result()
    _activate(monkeypatch, first)
    out = tmp_path / "r.ang"
    client = TestClient(app)
    assert _export(client, "ang", out).status_code == 200
    assert float(_kv(_header(out), "XSTEP")[0]) == pytest.approx(STEP_UM)

    second = _make_result()
    second.metadata["step_size_um"] = 0.25
    _activate(monkeypatch, second)
    again = _export(client, "ang", out)
    assert again.status_code == 200, again.text
    assert float(_kv(_header(out), "XSTEP")[0]) == pytest.approx(0.25),         "the old file was kept"


def test_crop_offset_is_documented_and_coordinates_stay_relative(
        monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result(scan_offset=(30, 20)))
    out = tmp_path / "crop.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200

    hdr = _header(out)
    d = _data(out)
    # coordinates are relative to the exported grid ...
    assert d[0, 3] == 0.0 and d[0, 4] == 0.0
    # ... and the line says where that grid sits in the original scan
    line = " ".join(_kv(hdr, "ORIENTA_SCAN_ORIGIN"))
    assert "row 30" in line and "column 20" in line
    # 20 columns * 0.5 um and 30 rows * 0.5 um
    assert "x 10.000" in line and "y 15.000" in line
    assert "90 x 120" in line


def test_a_full_scan_states_that_it_is_not_a_crop(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result())
    out = tmp_path / "full.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    line = " ".join(_kv(_header(out), "ORIENTA_SCAN_ORIGIN"))
    assert "row 0" in line and "column 0" in line


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def _phase_blocks(header):
    blocks, cur = {}, None
    for ln in header:
        m = re.match(r"^#\s*Phase\s+(\d+)\s*$", ln)
        if m:
            cur = int(m.group(1))
            blocks[cur] = {}
        elif cur is not None:
            m = re.match(r"^#\s*(MaterialName|Symmetry|LatticeConstants)\s+(.*)$", ln)
            if m:
                blocks[cur][m.group(1)] = m.group(2).split()
    return blocks


def test_lattice_constants_come_from_the_phase_source(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result())
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    blocks = _phase_blocks(_header(out))
    by_name = {b["MaterialName"][0]: b for b in blocks.values()}
    assert [float(v) for v in by_name["Al"]["LatticeConstants"]] == \
        pytest.approx([4.049, 4.049, 4.049, 90, 90, 90])
    assert [float(v) for v in by_name["Fe4Al13"]["LatticeConstants"]] == \
        pytest.approx([15.488, 8.087, 12.477, 90, 107.669, 90])


def test_phase_numbers_in_the_data_match_the_header(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result())
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    blocks = _phase_blocks(_header(out))
    name_of = {k: b["MaterialName"][0] for k, b in blocks.items()}
    d = _data(out)
    n = N_ROWS * N_COLS
    expect = np.where(np.arange(n) % 3 == 0, "Fe4Al13", "Al")
    got = np.array([name_of[int(p)] for p in d[:, 7]])
    assert (got == expect).all()


def test_a_structured_phase_keeps_its_own_lattice(monkeypatch, tmp_path):
    """Hough results carry the CIF structure; that must not be overwritten."""
    _activate(monkeypatch, _make_result(structured=True, sht_paths=False))
    out = tmp_path / "r.ang"
    r = _export(TestClient(app), "ang", out)
    # Fe4Al13 has no structure and no source: that is the refusal below.
    assert r.status_code == 400, r.text
    assert "Fe4Al13" in r.json()["detail"]


def test_unknown_lattice_is_refused_not_written_as_one(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result(sht_paths=False))
    out = tmp_path / "r.ang"
    r = _export(TestClient(app), "ang", out)
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "lattice" in detail.lower()
    assert "Al" in detail
    assert not out.exists() or "1.000 1.000 1.000" not in out.read_text()


# ---------------------------------------------------------------------------
# Euler frame
# ---------------------------------------------------------------------------

def test_orix_reads_back_the_same_orientations_at_the_same_positions(
        monkeypatch, tmp_path):
    result = _make_result()
    _activate(monkeypatch, result)
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200

    xmap = orix_load(str(out))
    order = np.lexsort((np.asarray(xmap.x), np.asarray(xmap.y)))
    got = xmap.rotations[order]
    want = result.xmap.rotations                      # row-major, native frame
    ang = np.degrees(Orientation(got, Oh).angle_with(Orientation(want, Oh)))
    assert float(np.max(ang)) < 1e-3
    # positions in micrometres, in the same order
    np.testing.assert_allclose(np.asarray(xmap.x)[order],
                               np.tile(np.arange(N_COLS), N_ROWS) * STEP_UM,
                               atol=1e-6)


def test_frame_is_stated_in_plain_words_for_an_oxford_source(
        monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result(vendor="oxford"))
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    text = " ".join(_kv(_header(out), "ORIENTA_EULER_FRAME"))
    assert "EDAX TSL" in text
    assert "orix" in text and "kikuchipy" in text
    assert "Oxford" in text and "90" in text and "phi1" in text


def test_ang_and_light_h5_differ_by_exactly_the_stated_phi1_offset(
        monkeypatch, tmp_path):
    """The relation the header states is the relation the files have."""
    _activate(monkeypatch, _make_result(vendor="oxford"))
    client = TestClient(app)
    ang, h5 = tmp_path / "r.ang", tmp_path / "r_light.h5"
    assert _export(client, "ang", ang).status_code == 200
    assert _export(client, "h5_light", h5).status_code == 200
    with h5py.File(h5, "r") as f:
        e_h5 = np.asarray(f["Indexing/euler_angles"]).reshape(-1, 3)
    e_ang = _data(ang)[:, :3]
    d = np.degrees(e_h5 - e_ang)
    d = (d + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(d[:, 0], -90.0, atol=1e-3)
    np.testing.assert_allclose(d[:, 1:], 0.0, atol=1e-3)


def test_an_edax_source_states_that_both_files_agree(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result(vendor="edax"))
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    text = " ".join(_kv(_header(out), "ORIENTA_EULER_FRAME"))
    assert "EDAX TSL" in text
    assert "same" in text.lower()


# ---------------------------------------------------------------------------
# acquisition geometry
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_h5oina(tmp_path):
    p = tmp_path / "scan.h5oina"
    with h5py.File(p, "w") as f:
        h = f.create_group("1/EBSD/Header")
        h.create_dataset("Scanning Rotation Angle", data=np.float32(np.pi))
        h.create_dataset("Tilt Angle", data=np.float32(np.deg2rad(70.0)))
        h.create_dataset("Specimen Orientation Euler",
                         data=np.array([[0.0, -np.pi / 2, 0.0]], np.float32))
        h.create_dataset("Detector Orientation Euler",
                         data=np.array([[0.006, 1.6462399, 6.279933]],
                                       np.float32))
        h.create_dataset("Working Distance", data=np.float32(16.0))
        h.create_dataset("Beam Voltage", data=np.float32(20.0))
    return str(p)


def test_scanning_rotation_reaches_the_ang_header(
        monkeypatch, tmp_path, fake_h5oina):
    _activate(monkeypatch, _make_result(), source_path=fake_h5oina)
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    line = " ".join(_kv(_header(out), "ORIENTA_ACQUISITION"))
    assert "scanning_rotation_angle_deg=180.000" in line
    assert "sample_tilt_deg=70.000" in line
    assert "specimen_orientation_euler_deg=0.000,-90.000,0.000" in line
    # the EDAX field with a real meaning is filled in, not left at 0
    wd = [ln for ln in _header(out) if ln.startswith("# WorkingDistance")]
    assert wd and float(wd[0].split()[-1]) == pytest.approx(16.0)


def test_scanning_rotation_reaches_the_light_h5(
        monkeypatch, tmp_path, fake_h5oina):
    _activate(monkeypatch, _make_result(), source_path=fake_h5oina)
    out = tmp_path / "r_light.h5"
    assert _export(TestClient(app), "h5_light", out).status_code == 200
    with h5py.File(out, "r") as f:
        acq = f["Acquisition"]
        assert float(acq.attrs["scanning_rotation_angle_deg"]) == pytest.approx(180.0)
        assert float(acq.attrs["sample_tilt_deg"]) == pytest.approx(70.0)
        np.testing.assert_allclose(
            acq.attrs["specimen_orientation_euler_deg"], [0.0, -90.0, 0.0],
            atol=1e-4)


def test_a_source_without_geometry_does_not_break_the_export(
        monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result(), source_path=None)
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    assert _kv(_header(out), "ORIENTA_ACQUISITION") == [] or \
        "scanning_rotation" not in " ".join(_kv(_header(out), "ORIENTA_ACQUISITION"))


# ---------------------------------------------------------------------------
# provenance of the phase assignment
# ---------------------------------------------------------------------------

_PRIOR_STEPS = [
    {"key": "indexing.spherical_gpu", "params": {"bandwidth": 63}},
    {"key": "eds.chemistry_prior",
     "params": {"strength_by_phase": {"Al": 0.75, "Fe4Al13": 0.75},
                "n_adjusted": 118}},
]


def test_the_ang_states_that_the_eds_prior_was_used(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result(steps=_PRIOR_STEPS))
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    line = " ".join(_kv(_header(out), "ORIENTA_PHASE_ASSIGNMENT"))
    assert "EDS chemistry prior: yes" in line
    assert "not independent" in line
    assert "Al 0.75" in line and "118" in line


def test_the_ang_states_that_no_prior_was_used(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result(
        steps=[{"key": "indexing.spherical_gpu", "params": {}}]))
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    line = " ".join(_kv(_header(out), "ORIENTA_PHASE_ASSIGNMENT"))
    assert "EDS chemistry prior: no" in line
    assert "not independent" not in line


@pytest.mark.parametrize("fmt", ["h5", "h5_light"])
def test_both_h5_exports_state_the_prior(monkeypatch, tmp_path, fmt,
                                          fake_h5oina):
    # the rich export copies the source file, so the fake source must be a
    # real HDF5 file (it is)
    _activate(monkeypatch, _make_result(steps=_PRIOR_STEPS),
              source_path=fake_h5oina)
    out = tmp_path / f"r_{fmt}.h5"
    r = _export(TestClient(app), fmt, out)
    assert r.status_code == 200, r.text
    with h5py.File(out, "r") as f:
        assert int(f["Indexing"].attrs["eds_chemistry_prior"]) == 1
        assert "not independent" in str(f["Indexing"].attrs["phase_assignment"])
        assert float(f["Acquisition"].attrs["scanning_rotation_angle_deg"]) == \
            pytest.approx(180.0)


def test_h5_without_a_prior_says_zero(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result(
        steps=[{"key": "indexing.spherical_gpu", "params": {}}]))
    out = tmp_path / "r_light.h5"
    assert _export(TestClient(app), "h5_light", out).status_code == 200
    with h5py.File(out, "r") as f:
        assert int(f["Indexing"].attrs["eds_chemistry_prior"]) == 0


# ---------------------------------------------------------------------------
# the stub is only a stand-in: the real reader and the CIF agree
# ---------------------------------------------------------------------------

_LIB = Path(__file__).resolve().parents[1] / "Database"


@pytest.mark.skipif(
    not (_LIB / "EBSD_SHT_Database" / "Al13Fe4").is_dir(),
    reason="phase library not present")
def test_real_sht_lattice_agrees_with_the_cif():
    import importlib
    ang_export = importlib.import_module("backend.api.services.ang_export")
    importlib.reload(ang_export)           # drop the stub installed above
    sht = next((_LIB / "EBSD_SHT_Database" / "Al13Fe4").glob("*.sht"))
    cif = _LIB / "CIF_Library" / "Al13Fe4.cif"
    from_sht = ang_export.lattice_from_sht(sht)
    from_cif = ang_export.lattice_from_cif(cif)
    assert from_sht == pytest.approx(from_cif, rel=2e-3)
    assert from_sht[0] > 10.0              # angstrom, not nanometre


# ---------------------------------------------------------------------------
# TSL symmetry codes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("point_group, code", [
    ("1", "1"), ("-1", "1"),
    ("2/m", "2"), ("112", "2"),
    ("222", "22"), ("mmm", "22"),
    ("4", "4"), ("4/m", "4"), ("422", "42"), ("4/mmm", "42"),
    ("3", "3"), ("-3", "3"), ("32", "32"), ("-3m", "32"),
    ("6", "6"), ("6/m", "6"), ("622", "62"), ("6/mmm", "62"),
    ("23", "23"), ("m-3", "23"),
    ("432", "43"), ("-43m", "43"), ("m-3m", "43"),
])
def test_tsl_symmetry_code_is_the_digits_of_the_laue_rotation_group(
        point_group, code):
    from orix.quaternion.symmetry import _groups
    from backend.api.services import ang_export
    pg = next(g for g in _groups if g.name == point_group)
    assert ang_export.tsl_symmetry_code(pg) == code


def test_a_point_group_without_a_tsl_code_is_refused():
    from backend.api.services import ang_export
    with pytest.raises(ValueError, match="symmetry"):
        ang_export.tsl_symmetry_code(None)


def test_the_ang_writes_tsl_codes_not_group_names(monkeypatch, tmp_path):
    _activate(monkeypatch, _make_result())
    out = tmp_path / "r.ang"
    assert _export(TestClient(app), "ang", out).status_code == 200
    blocks = _phase_blocks(_header(out))
    codes = {b["MaterialName"][0]: b["Symmetry"][0] for b in blocks.values()}
    assert codes == {"Al": "43", "Fe4Al13": "2"}
    # and orix, which reads these codes back, recovers the Laue group
    xmap = orix_load(str(out))
    laue = {p.name: p.point_group.laue.name for _, p in xmap.phases if p.name}
    assert laue == {"Al": "m-3m", "Fe4Al13": "2/m"}


def test_a_phase_without_symmetry_stops_the_ang_export(monkeypatch, tmp_path):
    result = _make_result()
    result.xmap.phases[2].point_group = None
    _activate(monkeypatch, result)
    r = _export(TestClient(app), "ang", tmp_path / "r.ang")
    assert r.status_code == 400
    assert "symmetry" in r.json()["detail"].lower()
    assert "Fe4Al13" in r.json()["detail"]


# ---------------------------------------------------------------------------
# the phase table of the .h5 files
# ---------------------------------------------------------------------------

_SG = {"al.sht": 225, "fe4al13.sht": 12}


@pytest.fixture(autouse=True)
def stub_sht_space_group(monkeypatch):
    from backend.api.services import ang_export
    monkeypatch.setattr(
        ang_export, "space_group_from_sht",
        lambda path: _SG[Path(str(path)).name.lower()], raising=False)


@pytest.mark.parametrize("fmt", ["h5_light", "h5"])
def test_phase_table_carries_space_group_lattice_and_frame(
        monkeypatch, tmp_path, fmt, fake_h5oina):
    _activate(monkeypatch, _make_result(), source_path=fake_h5oina)
    out = tmp_path / f"r_{fmt}.h5"
    assert _export(TestClient(app), fmt, out).status_code == 200
    with h5py.File(out, "r") as f:
        ph = {str(g.attrs["name"]): g.attrs for g in f["Indexing/Phases"].values()}
        al, fe = ph["Al"], ph["Fe4Al13"]
        assert int(al["space_group"]) == 225
        assert str(al["space_group_symbol"]).replace(" ", "") == "Fm-3m"
        assert int(fe["space_group"]) == 12
        assert str(fe["space_group_symbol"]).replace(" ", "") == "C2/m"
        np.testing.assert_allclose(al["lattice_constants"],
                                   [4.049, 4.049, 4.049, 90, 90, 90], atol=1e-3)
        np.testing.assert_allclose(fe["lattice_constants"],
                                   [15.488, 8.087, 12.477, 90, 107.669, 90], atol=1e-3)
        assert str(fe["lattice_length_unit"]) == "angstrom"
        frame = str(fe["crystal_reference_frame"])
        assert "X||a" in frame and "Z||c*" in frame
        idx_frame = str(f["Indexing"].attrs["crystal_reference_frame"])
        assert "X||a" in idx_frame and "Z||c*" in idx_frame
        assert "X||a*" in idx_frame      # names what the other convention is


def test_the_h5_frame_statement_is_the_same_for_every_route(monkeypatch, tmp_path):
    seen = set()
    for method in (IndexingMethod.SPHERICAL, IndexingMethod.HOUGH,
                   IndexingMethod.DICTIONARY):
        result = _make_result()
        result.method = method
        _activate(monkeypatch, result)
        out = tmp_path / f"{method.value}.h5"
        assert _export(TestClient(app), "h5_light", out).status_code == 200
        with h5py.File(out, "r") as f:
            seen.add(str(f["Indexing"].attrs["crystal_reference_frame"]))
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# the source header, verbatim, and not applied
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["h5_light", "h5"])
def test_source_header_values_are_copied_under_their_own_names(
        monkeypatch, tmp_path, fmt, fake_h5oina):
    _activate(monkeypatch, _make_result(), source_path=fake_h5oina)
    out = tmp_path / f"r_{fmt}.h5"
    assert _export(TestClient(app), fmt, out).status_code == 200
    with h5py.File(out, "r") as f:
        a = f["Acquisition"].attrs
        assert float(a["Scanning Rotation Angle"]) == pytest.approx(np.pi, abs=1e-5)
        assert float(a["Tilt Angle"]) == pytest.approx(np.deg2rad(70.0), abs=1e-5)
        np.testing.assert_allclose(np.ravel(a["Specimen Orientation Euler"]),
                                   [0.0, -np.pi / 2, 0.0], atol=1e-5)
        assert str(a["header_values_unit"]) == "radians"
        assert int(a["applied_to_euler_angles"]) == 0
        note = str(a["applied_to_euler_angles_note"])
        assert "none" in note.lower() and "euler_angles" in note
        assert "EulerCorrection" in note


# ---------------------------------------------------------------------------
# format version and its documentation
# ---------------------------------------------------------------------------

_DOC = Path(__file__).resolve().parents[1] / "docs" / "light_h5_format.md"


def test_writer_and_documentation_agree_on_the_format_version(
        monkeypatch, tmp_path):
    from backend.api.services.result_exporter import FORMAT_VERSION
    _activate(monkeypatch, _make_result())
    out = tmp_path / "r_light.h5"
    assert _export(TestClient(app), "h5_light", out).status_code == 200
    with h5py.File(out, "r") as f:
        assert str(f["Indexing"].attrs["format_version"]) == FORMAT_VERSION
        assert str(f["Documentation"].attrs["format_version"]) == FORMAT_VERSION
    doc = _DOC.read_text(encoding="utf-8")
    assert f'| `format_version` | str | currently `"{FORMAT_VERSION}"`' in doc
    assert f'@format_version        str    "{FORMAT_VERSION}"' in doc
    assert re.search(rf"^\| {re.escape(FORMAT_VERSION)} \|", doc, re.M), \
        "the version history has no row for the current version"


def test_a_current_light_file_imports_again(monkeypatch, tmp_path, fake_h5oina):
    _activate(monkeypatch, _make_result(), source_path=fake_h5oina)
    out = tmp_path / "r_light.h5"
    client = TestClient(app)
    assert _export(client, "h5_light", out).status_code == 200
    r = client.post("/api/indexing/import-h5", json={"path": str(out)})
    assert r.status_code == 200, r.text


def test_the_documentation_describes_the_scan_provenance_attributes():
    doc = _DOC.read_text(encoding="utf-8")
    # each attribute has its own row in the /Indexing attribute table ...
    for name in ("scan_row_offset", "scan_col_offset", "scan_shape"):
        assert f"| `{name}` |" in doc, name
    # ... and the section says which axis, which origin, which unit
    section = doc[doc.index("### Position in the original scan"):]
    section = re.split(r"^### ", section[10:], maxsplit=1, flags=re.M)[0]
    section = " ".join(section.split())
    for needle in ("0-based", "first array axis", "second axis", "relative to this",
                   "x_original", "not micrometres"):
        assert needle in section, needle
