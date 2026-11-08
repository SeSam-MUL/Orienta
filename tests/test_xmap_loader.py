"""One crystal-map loader for the Analysis page and the batch, Aztec .h5oina included.

Real data: the Oxford reference scan (an Aztec-indexed .h5oina, aluminium). It is not in the repository; set ORIENTA_TEST_DATA
to a Test_data directory that has it (the tests skip when the file is absent).
"""
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from analysis.ebsd_dataset import EBSDDataset
from analysis.xmap_loader import load_xmap_file
from tests.synthetic_grain_map import make_map, to_xmap
from tests.test_analysis_page_sync import _fake_texture, _write_ctf

SAMPLE_B = "EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina"


def _test_data_dir():
    from tests.data_deps import TEST_DATA_DIR
    return Path(os.environ.get("ORIENTA_TEST_DATA", TEST_DATA_DIR))


def _sample_b():
    path = _test_data_dir() / SAMPLE_B
    if not path.exists():
        pytest.skip(f"needs {path} - measurement data is not in the repository (set ORIENTA_TEST_DATA to a "
                    "Test_data directory that has it)")
    return str(path)


# ---------------------------------------------------------------------------------------------
# synthetic files
# ---------------------------------------------------------------------------------------------
def _str_ds(group, name, value):
    group.create_dataset(name, data=np.array([value.encode()], dtype=object), dtype=h5py.string_dtype())


def write_h5oina(path, euler, phase, ny, nx, step=0.5, step_y=None, phases=None, bc=None, slice_name="1"):
    """An Aztec-style indexed .h5oina: arrays under /1/EBSD/Data, scalars as shape-(1,) arrays in the header.

    phases: {header key: (name, space group, (a, b, c) in angstrom)}; the angles are 90 degrees (radians).
    """
    phases = phases if phases is not None else {1: ("Aluminium", 225, (4.05, 4.05, 4.05))}
    with h5py.File(path, "w") as f:
        f.create_dataset("Format Version", data=np.array([b"5.0"]))
        d = f.create_group(f"{slice_name}/EBSD/Data")
        h = f.create_group(f"{slice_name}/EBSD/Header")
        d["Euler"] = np.asarray(euler, np.float32).reshape(-1, 3)
        d["Phase"] = np.asarray(phase, np.uint8).reshape(-1)
        d["X"] = (np.tile(np.arange(nx), ny) * step).astype(np.float32)
        d["Y"] = (np.repeat(np.arange(ny), nx) * (step if step_y is None else step_y)).astype(np.float32)
        if bc is not None:
            d["Band Contrast"] = np.asarray(bc, np.uint8).reshape(-1)
        h["X Step"] = np.array([step], np.float32)
        h["Y Step"] = np.array([step if step_y is None else step_y], np.float32)
        h["X Cells"] = np.array([nx], np.int32)
        h["Y Cells"] = np.array([ny], np.int32)
        for key, (name, sg, (a, b, c)) in phases.items():
            g = h.create_group(f"Phases/{key}")
            _str_ds(g, "Phase Name", name)
            g["Space Group"] = np.array([sg], np.int32)
            g["Laue Group"] = np.array([11], np.int32)
            g["Lattice Dimensions"] = np.array([[a, b, c]], np.float32)
            g["Lattice Angles"] = np.array([[np.pi / 2] * 3], np.float32)


def write_orienta_h5(path, euler, phase, ny, nx, step=0.5, names=("Al", "Al7FeCu2"), sgs=(225, 139)):
    """Orienta's own result .h5 (rich multi-phase layout): phase_id 0 = not indexed, 1..N = phase."""
    with h5py.File(path, "w") as f:
        idx = f.create_group("Indexing")
        idx.attrs["step_size_um"] = step
        a = idx.create_group("Assignment")
        a["phase_id"] = (np.asarray(phase).reshape(ny, nx) + 1).astype(np.uint8)
        a["euler_angles"] = np.asarray(euler, np.float32).reshape(ny, nx, 3)
        a["confidence_index"] = np.full((ny, nx), 0.5, np.float32)
        for i, (nm, sg) in enumerate(zip(names, sgs), start=1):
            g = idx.create_group(f"Phases/{i}")
            g.attrs["name"] = nm
            g.attrs["space_group"] = sg


def _blocks(ny=12, nx=15):
    """Two orientation blocks (left / right half) as Bunge Euler in radians, phase 0 everywhere."""
    e = np.empty((ny, nx, 3))
    e[:, : nx // 2] = np.deg2rad([10.0, 20.0, 30.0])
    e[:, nx // 2:] = np.deg2rad([80.0, 40.0, 5.0])
    return e.reshape(-1, 3), np.zeros(ny * nx, int)


# ---------------------------------------------------------------------------------------------
# Aztec .h5oina, synthetic
# ---------------------------------------------------------------------------------------------
def test_h5oina_values_come_from_the_arrays_and_the_header(tmp_path):
    euler, phase = _blocks()
    phase[:5] = 0                     # Aztec 0 = not indexed
    phase[5:] = 1
    bc = np.arange(phase.size) % 256
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, step=0.75, bc=bc)
    xmap, step, meta = load_xmap_file(str(tmp_path / "s.h5oina"))
    assert xmap.shape == (12, 15) and step == pytest.approx(0.75) and xmap.dx == pytest.approx(0.75)
    assert meta["format"] == "aztec_h5oina" and meta["step_from_file"] is True
    pid = np.asarray(xmap.phase_id)
    assert (pid[:5] == -1).all() and (pid[5:] == 0).all()                     # Aztec n -> orix n - 1, 0 -> -1
    from orix.quaternion import Rotation
    want = Rotation.from_euler(np.asarray(euler, np.float32).astype(np.float64))
    assert np.rad2deg(xmap.rotations.angle_with(want)).max() < 1e-4            # Euler as stored, no frame change
    np.testing.assert_array_equal(xmap.prop["bc"], bc.astype(np.float32))
    assert EBSDDataset(xmap, step_size=step).has_native_bc is True


def test_h5oina_phases_carry_name_and_the_point_group_of_their_own_space_group(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    phase[100:] = 2
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, phases={
        1: ("Aluminium", 225, (4.05, 4.05, 4.05)), 2: ("Mg2Si-like", 62, (7.0, 4.0, 5.0))})
    xmap, _, meta = load_xmap_file(str(tmp_path / "s.h5oina"))
    got = {int(i): (p.name, p.point_group.name, p.space_group.number) for i, p in xmap.phases if i >= 0}
    assert got == {0: ("Aluminium", "m-3m", 225), 1: ("Mg2Si-like", "mmm", 62)}
    assert meta["phases"] == ["Aluminium", "Mg2Si-like"]
    # the pixels point at the right phase: the first pixel is Aluminium, the last one the second phase
    assert xmap.phases[int(xmap.phase_id[0])].name == "Aluminium"
    assert xmap.phases[int(xmap.phase_id[-1])].name == "Mg2Si-like"
    assert xmap.phases[0].structure.lattice.a == pytest.approx(4.05)           # the cell of the header, angstrom


def test_h5oina_pixels_follow_their_header_phase_when_an_earlier_phase_has_no_pixels(tmp_path):
    euler, phase = _blocks()
    phase[:] = 2                                       # only the second header phase is in the data
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, phases={
        1: ("Aluminium", 225, (4.05, 4.05, 4.05)), 2: ("Mg2Si-like", 62, (7.0, 4.0, 5.0))})
    xmap, _, _ = load_xmap_file(str(tmp_path / "s.h5oina"))
    ph = xmap.phases[int(xmap.phase_id[0])]
    assert ph.name == "Mg2Si-like" and ph.point_group.name == "mmm"


def test_h5oina_monoclinic_phase_gets_its_frame_symmetry(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, phases={1: ("Mono", 14, (5.0, 6.0, 7.0))})
    ds = EBSDDataset(load_xmap_file(str(tmp_path / "s.h5oina"))[0], step_size=0.5)
    from crystal_symmetry import unique_axis
    assert unique_axis(ds.xmap.phases[0].point_group) == "b"


def test_h5oina_phase_without_a_usable_symmetry_fails_loud_when_pixels_use_it(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, phases={1: ("Mystery", 0, (4.0, 4.0, 4.0))})
    with pytest.raises(ValueError, match="Mystery.*space group"):
        load_xmap_file(str(tmp_path / "s.h5oina"))


def test_h5oina_header_phase_nobody_uses_may_lack_a_symmetry(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, phases={
        1: ("Aluminium", 225, (4.05, 4.05, 4.05)), 2: ("Placeholder", 0, (0.0, 0.0, 0.0))})
    xmap, _, meta = load_xmap_file(str(tmp_path / "s.h5oina"))
    assert meta["phases"] == ["Aluminium"] and meta["skipped_phases"] == ["Placeholder"]


def test_h5oina_pixels_of_a_phase_the_header_does_not_list_fail_loud(tmp_path):
    euler, phase = _blocks()
    phase[:] = 3
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15)
    with pytest.raises(ValueError, match="phase 3"):
        load_xmap_file(str(tmp_path / "s.h5oina"))


def test_h5oina_with_different_x_and_y_steps_is_refused(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, step=0.5, step_y=0.25)
    with pytest.raises(ValueError, match="X Step.*Y Step"):
        load_xmap_file(str(tmp_path / "s.h5oina"))


def test_h5oina_without_band_contrast_has_no_bc(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15)
    xmap, _, _ = load_xmap_file(str(tmp_path / "s.h5oina"))
    assert "bc" not in xmap.prop and EBSDDataset(xmap, step_size=0.5).has_native_bc is False


def test_h5oina_reads_the_first_slice_and_says_so(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15)
    with h5py.File(tmp_path / "s.h5oina", "a") as f:
        f.copy("1", "2")
    _, _, meta = load_xmap_file(str(tmp_path / "s.h5oina"))
    assert meta["slice"] == "1" and meta["n_slices"] == 2


def test_h5oina_without_slice_one_fails_loud(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, slice_name="2")
    with pytest.raises(ValueError, match="slice"):
        load_xmap_file(str(tmp_path / "s.h5oina"))


def test_the_frame_the_loader_assumes_is_what_the_frame_service_says():
    """Aztec Euler as stored is Orienta's canonical (CS1) frame: the Oxford adapter is the identity, so the loader
    applies nothing. If that ever changes, this loader must change with it."""
    from backend.api.services.orientation_frame import apply_reference_frame_correction
    from orix.quaternion import Rotation
    rot = Rotation.from_euler(np.deg2rad([[10.0, 20.0, 30.0], [80.0, 40.0, 5.0]]))
    out = apply_reference_frame_correction(rot, "oxford", {}, "vendor")
    assert np.rad2deg(out.angle_with(rot)).max() < 1e-9


def test_the_loader_module_does_not_import_the_backend_at_import_time():
    import ast
    src = (Path(__file__).parents[1] / "analysis" / "xmap_loader.py").read_text(encoding="utf-8")
    top_level = [n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for n in top_level:
        names = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) else [])
        assert not any(str(x).split(".")[0] == "backend" for x in names)


# ---------------------------------------------------------------------------------------------
# Aztec .h5oina, real data
# ---------------------------------------------------------------------------------------------
def test_sample_b_through_the_loader():
    path = _sample_b()
    xmap, step, meta = load_xmap_file(path)
    assert xmap.shape == (90, 120) and step == pytest.approx(0.5)
    assert [p.name for _, p in xmap.phases if _ >= 0] == ["Aluminium"]
    assert xmap.phases[0].point_group.name == "m-3m"
    assert "bc" in xmap.prop and xmap.prop["bc"].size == 10800
    with h5py.File(path, "r") as f:
        raw = np.asarray(f["1/EBSD/Data/Euler"], np.float64)
        raw_phase = np.asarray(f["1/EBSD/Data/Phase"]).astype(int)
        raw_bc = np.asarray(f["1/EBSD/Data/Band Contrast"])
    from orix.quaternion import Rotation
    assert np.rad2deg(xmap.rotations.angle_with(Rotation.from_euler(raw)).max()) < 1e-4
    np.testing.assert_array_equal(np.asarray(xmap.phase_id), raw_phase - 1)
    np.testing.assert_array_equal(xmap.prop["bc"], raw_bc.astype(np.float32))
    assert meta["step_from_file"] is True


def test_sample_b_grains_equal_the_independent_reading():
    """The independent reader of tests/raw_h5oina.py reads the same arrays and takes every phase as m-3m, which is
    what the Oxford reference scan's one aluminium phase is (space group 225); the grain analysis must not see a
    difference."""
    path = _sample_b()
    from analysis.grain_analysis import reconstruct_grains
    from tests.raw_h5oina import xmap_from_h5oina
    ref_xmap, ref_step = xmap_from_h5oina(path)
    xmap, step, _ = load_xmap_file(path)
    assert step == ref_step
    assert xmap.phases[0].point_group.name == ref_xmap.phases[0].point_group.name == "m-3m"
    a = reconstruct_grains(EBSDDataset(xmap, step_size=step), angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=True)
    b = reconstruct_grains(EBSDDataset(ref_xmap, step_size=ref_step), angle_threshold_deg=5.0, min_pixels=3,
                           alpha=3.1, fill=True)
    assert a.n_grains == b.n_grains and a.n_grains > 0
    np.testing.assert_array_equal(a.grain_ids, b.grain_ids)


# ---------------------------------------------------------------------------------------------
# the other formats through the same function
# ---------------------------------------------------------------------------------------------
def test_a_ctf_with_band_contrast(tmp_path):
    _write_ctf(tmp_path / "t.ctf")
    xmap, step, meta = load_xmap_file(str(tmp_path / "t.ctf"))
    # the .ctf header states XStep 1.0: a measured step, not the unit-pixel placeholder (no batch warning for it)
    assert step == 1.0 and meta["format"] == "orix" and meta["step_from_file"] is True
    ds = EBSDDataset(xmap, step_size=step)
    assert ds.has_native_bc is True and ds.bc.min() >= 20


def test_orienta_result_h5_through_the_same_function(tmp_path):
    euler, phase = _blocks()
    phase[100:] = 1
    write_orienta_h5(tmp_path / "result_x.h5", euler, phase, 12, 15, step=0.5)
    xmap, step, meta = load_xmap_file(str(tmp_path / "result_x.h5"))
    assert step == pytest.approx(0.5) and xmap.shape == (12, 15) and meta["format"] == "orienta_result_h5"
    assert [p.name for i, p in xmap.phases if i >= 0] == ["Al", "Al7FeCu2"]
    assert xmap.phases[0].point_group.name == "m-3m"


def test_the_route_reader_name_is_the_loader_function():
    import backend.api.routes.analysis as ar
    from analysis.xmap_loader import load_orienta_result_h5
    assert ar._load_kikuchipy_rich_h5 is load_orienta_result_h5          # indexing.py and the tests import this name


def test_a_file_nobody_can_read_says_what_both_readers_said(tmp_path):
    (tmp_path / "junk.h5").write_bytes(b"not an hdf5 file")
    with pytest.raises(ValueError) as e:
        load_xmap_file(str(tmp_path / "junk.h5"))
    assert "orix.io.load failed:" in str(e.value) and "Custom .h5 fallback failed:" in str(e.value)


def test_the_route_loads_an_h5oina_and_keeps_the_header_step(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, step=1.0, bc=np.full(phase.size, 100))
    saved = ar._dataset
    try:
        r = TestClient(app).post("/api/analysis/load", json={"xmap_path": str(tmp_path / "s.h5oina")})
        assert r.status_code == 200, r.text
        assert r.json()["shape"] == [12, 15] and r.json()["phases"][0].startswith("Aluminium")
        # a header step of exactly 1.0 um is a measurement, not "unknown": no fallback to the viewer's file
        # an open viewer file with another step: the fallback, if it were asked, would answer 0.3
        monkeypatch.setattr("backend.api.routes.ebsd_viewer._get_active_signal", lambda: object())
        monkeypatch.setattr("backend.api.routes.ebsd_viewer._extract_step_size", lambda sig: {"x": 0.3, "y": 0.3})
        r = TestClient(app).post("/api/analysis/load", json={"xmap_path": str(tmp_path / "s.h5oina")})
        assert r.status_code == 200, r.text
        assert ar._dataset.step_size == 1.0 and ar._dataset.has_native_bc
    finally:
        ar._dataset = saved


# ---------------------------------------------------------------------------------------------
# the batch over a folder with one of each
# ---------------------------------------------------------------------------------------------
def test_a_batch_over_a_folder_with_one_of_each_format(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    m = make_map(shape=(30, 40), n_seeds=8)
    ny, nx = m["phase_id"].shape
    xmap = to_xmap(m)
    euler = xmap.rotations.to_euler()
    phase = m["phase_id"].reshape(-1)
    folder = tmp_path / "scans"
    folder.mkdir()
    import orix.io
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    orix.io.save(str(folder / "a_orix.ang"), xmap)
    _write_ctf(folder / "b.ctf")
    write_orienta_h5(folder / "c_result.h5", euler, phase, ny, nx)
    write_h5oina(folder / "d_aztec.h5oina", euler, phase + 1, ny, nx, bc=np.random.default_rng(1).integers(20, 200, ny * nx),
                 phases={1: ("Al", 225, (4.05,) * 3), 2: ("Al7FeCu2", 139, (6.3, 6.3, 14.8))})
    proc = bp.BatchProcessor(str(folder), str(tmp_path / "doc.xlsx"), min_grain_pixels=1, grain_alpha=0.0)
    proc.process_all()
    res = proc.result()
    assert res.failed == [], res.failed
    assert sorted(Path(p).name for p in res.processed) == ["a_orix.ang", "b.ctf", "c_result.h5", "d_aztec.h5oina"]
    assert res.save_error is None


def test_the_batch_reads_the_aztec_step_and_the_orienta_step(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    m = make_map(shape=(30, 40), n_seeds=8)
    ny, nx = m["phase_id"].shape
    euler = to_xmap(m).rotations.to_euler()
    phase = m["phase_id"].reshape(-1)
    write_h5oina(tmp_path / "d.h5oina", euler, phase + 1, ny, nx, step=0.25,
                 phases={1: ("Al", 225, (4.05,) * 3), 2: ("Al7FeCu2", 139, (6.3, 6.3, 14.8))})
    write_orienta_h5(tmp_path / "c.h5", euler, phase, ny, nx, step=0.4)
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), min_grain_pixels=1, grain_alpha=0.0)
    assert proc.process_file(tmp_path / "d.h5oina")["basic_info"]["step_micron"] == pytest.approx(0.25)
    assert proc.process_file(tmp_path / "c.h5")["basic_info"]["step_micron"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------------------------
# Odd grids, missing headers, odd array lengths, step digits, layering
# ---------------------------------------------------------------------------------------------
def _valid(tmp_path, name="s.h5oina", **kw):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / name, euler, phase, 12, 15, **kw)
    return str(tmp_path / name)


def _edit(path, fn):
    with h5py.File(path, "a") as f:
        fn(f)


def test_a_column_major_file_is_refused_with_its_name(tmp_path):
    path = _valid(tmp_path)

    def column_major(f):                   # Y runs fastest: the same 180 pixels, written in the other order
        d = f["1/EBSD/Data"]
        x = np.repeat(np.arange(15), 12) * 0.5
        y = np.tile(np.arange(12), 15) * 0.5
        del d["X"], d["Y"]
        d["X"], d["Y"] = x.astype(np.float32), y.astype(np.float32)

    _edit(path, column_major)
    with pytest.raises(ValueError, match="s.h5oina.*fastest"):
        load_xmap_file(path)


def test_a_grid_with_shifted_rows_is_refused_even_with_equal_steps(tmp_path):
    path = _valid(tmp_path)

    def hex_like(f):                       # every second row sits half a step to the right
        d = f["1/EBSD/Data"]
        x = (np.tile(np.arange(15), 12) + np.repeat(np.arange(12) % 2, 15) * 0.5) * 0.5
        del d["X"]
        d["X"] = x.astype(np.float32)

    _edit(path, hex_like)
    with pytest.raises(ValueError, match="s.h5oina.*shifted rows"):
        load_xmap_file(path)


@pytest.mark.parametrize("missing", ["X Step", "Y Step"])
def test_a_missing_step_names_the_file_and_the_field(tmp_path, missing):
    path = _valid(tmp_path)
    _edit(path, lambda f: f["1/EBSD/Header"].__delitem__(missing))
    with pytest.raises(ValueError, match=f"s.h5oina.*{missing}"):
        load_xmap_file(path)


def test_missing_cells_in_the_header_are_refused(tmp_path):
    path = _valid(tmp_path)
    _edit(path, lambda f: f["1/EBSD/Header"].__delitem__("X Cells"))
    with pytest.raises(ValueError, match="s.h5oina.*X Cells"):
        load_xmap_file(path)


def test_cells_that_do_not_match_the_pixels_are_refused(tmp_path):
    path = _valid(tmp_path)

    def wrong_cells(f):
        del f["1/EBSD/Header/X Cells"]
        f["1/EBSD/Header/X Cells"] = np.array([16], np.int32)

    _edit(path, wrong_cells)
    with pytest.raises(ValueError, match="s.h5oina.*16 x 12 cells.*180 pixels"):
        load_xmap_file(path)


def test_arrays_of_different_length_are_refused(tmp_path):
    path = _valid(tmp_path)

    def short_x(f):
        d = f["1/EBSD/Data"]
        x = np.asarray(d["X"])[:-3]
        del d["X"]
        d["X"] = x

    _edit(path, short_x)
    with pytest.raises(ValueError, match="s.h5oina.*same number of pixels"):
        load_xmap_file(path)


@pytest.mark.parametrize("missing", ["Euler", "Phase", "X", "Y"])
def test_a_missing_data_array_is_refused_with_its_name(tmp_path, missing):
    path = _valid(tmp_path)
    _edit(path, lambda f: f["1/EBSD/Data"].__delitem__(missing))
    with pytest.raises(ValueError, match=f"s.h5oina.*{missing} is missing"):
        load_xmap_file(path)


def test_band_contrast_of_the_wrong_length_is_refused_not_stored(tmp_path):
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, bc=np.arange(10))
    with pytest.raises(ValueError, match="s.h5oina.*Band Contrast has 10 values for 180 pixels"):
        load_xmap_file(str(tmp_path / "s.h5oina"))


def test_the_header_step_keeps_the_digits_that_were_meant(tmp_path):
    path = _valid(tmp_path, step=0.15)                     # stored as float32: 0.15000000596046448
    xmap, step, _ = load_xmap_file(path)
    assert step == 0.15
    assert str(step) == "0.15"


def test_a_crop_files_step_is_not_widened_float32():
    crops = sorted(_test_data_dir().glob("*_crop.h5oina"))
    if not crops:
        pytest.skip("needs a *_crop.h5oina in Test_data (set ORIENTA_TEST_DATA)")
    with h5py.File(crops[0], "r") as f:
        raw = float(np.asarray(f["1/EBSD/Header/X Step"])[0])
    assert raw != 0.15 and abs(raw - 0.15) < 1e-6          # the file really holds the float32 neighbour
    _, step, _ = load_xmap_file(str(crops[0]))
    assert step == 0.15


def test_importing_the_loader_does_not_import_the_backend():
    import subprocess
    import sys
    code = ("import sys, analysis.xmap_loader; "
            "bad = sorted(m for m in sys.modules if m == 'backend' or m.startswith('backend.')); "
            "assert not bad, bad")
    root = Path(__file__).parents[1]
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(root)})
    assert r.returncode == 0, r.stderr[-800:]


def test_ground_truth_g1_14_loads_with_its_step_of_exactly_one():
    path = _test_data_dir() / "ground_truth" / "G1_14.h5oina"
    if not path.exists():
        pytest.skip(f"needs {path} (set ORIENTA_TEST_DATA to a Test_data directory that has it)")
    xmap, step, meta = load_xmap_file(str(path))
    assert step == 1.0 and meta["step_from_file"] is True       # a measurement, not "unknown"
    assert xmap.shape == (602, 805)
    assert [p.name for i, p in xmap.phases if i >= 0] == ["Aluminium"]
    assert xmap.phases[0].point_group.name == "m-3m"
    assert meta["has_native_bc"] is True and xmap.prop["bc"].size == 602 * 805


def test_the_load_answer_says_what_the_reader_did_with_the_file(tmp_path, caplog):
    import logging
    from fastapi.testclient import TestClient
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    euler, phase = _blocks()
    phase[:] = 1
    write_h5oina(tmp_path / "s.h5oina", euler, phase, 12, 15, bc=np.full(phase.size, 90), phases={
        1: ("Aluminium", 225, (4.05, 4.05, 4.05)), 2: ("Placeholder", 0, (0.0, 0.0, 0.0))})
    with h5py.File(tmp_path / "s.h5oina", "a") as f:
        f.copy("1", "2")
    saved = ar._dataset
    try:
        with caplog.at_level(logging.INFO, logger="backend.api.routes.analysis"):
            r = TestClient(app).post("/api/analysis/load", json={"xmap_path": str(tmp_path / "s.h5oina")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_slices"] == 2 and body["skipped_phases"] == ["Placeholder"] and body["has_native_bc"] is True
        assert any("n_slices=2" in m and "Placeholder" in m for m in caplog.messages)
    finally:
        ar._dataset = saved
