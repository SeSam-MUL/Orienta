"""Hough on the seven trigonal space groups PyEBSDIndex gets wrong says so in the result.

orix and PyEBSDIndex do not tell the two settings of a trigonal point group apart
(-31m / -3m1, 312 / 321, 31m / 3m1). For the space groups 149, 151, 153, 157, 159,
162, 163 PyEBSDIndex works with the two-fold axes of the other setting, so a Hough
orientation of such a phase can be wrong (for example by 60 degrees about c). The run is
not refused and its numbers are not changed: ``result.metadata["caveats"]`` carries
``{"code": "houghTrigonalSetting", "phases": [...], "text": ...}``, the run log gets
a warning line, ``task_result`` and ``/result/last`` repeat the list, a saved file
holds it next to ``hough_reflectors`` and an import puts it back.

Every expectation is written out here by hand: the seven numbers, the negatives, the
two sentences. Nothing is derived from the code under test. The indexer itself is the
one seam (a stub that returns a map of fixed orientations); the controller function,
the merges, the route, the exporters and the importer are the real ones.
"""
import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("orix")
pytest.importorskip("diffsims")

from diffpy.structure import Atom, Lattice, Structure  # noqa: E402
from orix.crystal_map import CrystalMap, Phase, PhaseList  # noqa: E402
from orix.quaternion import Rotation  # noqa: E402

import hough_caveats as hc  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from indexing_controller import (  # noqa: E402
    ComparisonConfig, IndexingConfig, IndexingMethod, hough_index_patterns,
    run_per_phase_indexing)
from tests.test_batch_run_record import _light, _run, world  # noqa: E402,F401  (world: fixture)
from tests.test_per_phase_indexing import (  # noqa: E402
    _make_stub_pmr, _stub_phase_config, _stub_signal)

CODE = "houghTrigonalSetting"
SEVEN = (149, 151, 153, 157, 159, 162, 163)
NOT_AFFECTED = (150, 152, 154, 155, 156, 158, 160, 161, 164, 165, 166, 167, 194, 225)


def sentence_one(name):
    return (f"Phase {name}: For this space group PyEBSDIndex uses the two-fold axes of the "
            "other trigonal setting, and Hough often cannot tell apart orientations 60 "
            "degrees apart about c. Spherical or Dictionary indexing may help.")


def sentence_many(names):
    return ("Phases " + ", ".join(names) + ": For these space groups PyEBSDIndex uses the "
            "two-fold axes of the other trigonal setting, and Hough often cannot tell "
            "apart orientations 60 degrees apart about c. Spherical or Dictionary "
            "indexing may help.")


def phase(name, sg):
    """A two-atom phase of space group number ``sg`` (hexagonal cell, or cubic for 225)."""
    if sg == 225:
        lat = Lattice(4.05, 4.05, 4.05, 90, 90, 90)
        atoms = [Atom("Al", xyz=(0, 0, 0))]
    else:
        lat = Lattice(4.1, 4.1, 6.3, 90, 90, 120)
        atoms = [Atom("Al", xyz=(.11, .23, .31)), Atom("Fe", xyz=(.42, .07, .65))]
    return Phase(name=name, space_group=sg, structure=Structure(atoms=atoms, lattice=lat))


def plist(*phases):
    return PhaseList(phases=list(phases))


# --------------------------------------------------------------------------
# Which phases
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sg", SEVEN)
def test_each_of_the_seven_space_groups_gets_the_caveat(sg):
    assert hc.hough_caveats(plist(phase("P", sg))) == [
        {"code": CODE, "phases": ["P"], "text": sentence_one("P")}]


@pytest.mark.parametrize("sg", NOT_AFFECTED)
def test_the_other_space_groups_get_no_trigonal_setting_caveat(sg):
    """(The softer ``houghHexagonalVariants`` note some of them carry is held in
    ``test_hough_hexagonal_variants.py``.)"""
    assert [c for c in hc.hough_caveats(plist(phase("P", sg))) if c["code"] == CODE] == []


def test_a_phase_without_a_space_group_gets_nothing():
    p = Phase(name="nosg", structure=Structure(atoms=[Atom("Al", xyz=(0, 0, 0))],
                                               lattice=Lattice(4, 4, 4, 90, 90, 90)))
    assert p.space_group is None
    assert hc.hough_caveats(plist(p)) == []


def test_no_alternative_setting_number_exists_for_the_seven_groups():
    """The precondition of testing the ``% 1000`` rule with a stub below: diffpy has
    284 numbers above 230 (alternative settings, 1003 ... 17015), none of which reduces
    to one of the seven, so a real Phase cannot carry one."""
    from diffpy.structure.spacegroups import SpaceGroupList
    alt = {sg.number for sg in SpaceGroupList if sg.number > 230}
    assert not [n for n in alt if n % 1000 in SEVEN]


def test_an_alternative_setting_number_is_reduced_to_its_standard_group():
    stub = lambda name, n: (0, SimpleNamespace(name=name, space_group=SimpleNamespace(number=n)))  # noqa: E731
    assert hc.trigonal_setting_phases([stub("A", 1162), stub("B", 2149), stub("C", 1166)]) == ["A", "B"]


def test_only_the_affected_phases_are_named_in_list_order_each_once():
    pl = plist(phase("Al", 225), phase("Cr2N", 162), phase("Mg", 194), phase("Z", 149))
    got = hc.hough_caveats(pl)
    assert got == [{"code": CODE, "phases": ["Cr2N", "Z"], "text": sentence_many(["Cr2N", "Z"])}]


def test_the_sentence_names_no_percentage_and_claims_no_cause():
    for text in (hc.trigonal_setting_text(["A", "B"]), hc.trigonal_setting_text(["A"])):
        assert "%" not in text
        assert "therefore" not in text and "pattern match" not in text.lower()
        assert text.startswith(("Phase A:", "Phases A, B:")) and text.endswith("may help.")


# --------------------------------------------------------------------------
# The merge
# --------------------------------------------------------------------------

class R:
    def __init__(self, md):
        self.metadata = md


def entry(*names):
    return {"code": CODE, "phases": list(names), "text": "whatever it said"}


def test_the_merge_unites_the_phases_once_and_words_the_text_for_all_of_them():
    target = R({})
    hr.merge_build_records(target, [R({"caveats": [entry("A")]}), R({"caveats": [entry("B", "A")]}),
                                    R({}), None, R({"caveats": [entry("C")]})])
    assert target.metadata["caveats"] == [
        {"code": CODE, "phases": ["A", "B", "C"], "text": sentence_many(["A", "B", "C"])}]


def test_the_merge_keeps_what_the_target_has_and_adds_to_it():
    target = R({"caveats": [entry("A")]})
    hr.merge_build_records(target, [R({"caveats": [entry("B")]})])
    assert target.metadata["caveats"][0]["phases"] == ["A", "B"]


def test_without_a_caveat_the_merge_adds_no_key():
    target = R({})
    hr.merge_build_records(target, [R({}), R({"caveats": []})])
    assert "caveats" not in target.metadata
    target = SimpleNamespace(metadata=None)            # not a dict: contributes nothing
    hr.merge_build_records(target, [R({"caveats": [entry("A")]})])
    assert target.metadata is None


def test_a_code_the_merge_does_not_know_keeps_its_own_text():
    target = R({})
    hr.merge_build_records(target, [R({"caveats": [
        {"code": "fromTheFuture", "phases": ["A"], "text": "an English sentence"}]}),
        R({"caveats": [{"code": "fromTheFuture", "phases": ["B"], "text": "another"}]})])
    assert target.metadata["caveats"] == [
        {"code": "fromTheFuture", "phases": ["A", "B"], "text": "an English sentence"}]


def test_the_per_phase_merge_of_a_phase_map_run_carries_the_caveats():
    """``run_per_phase_indexing`` builds a fresh result out of the per-phase results;
    only the two affected phases are named, and a phase named twice appears once."""
    sig = _stub_signal()
    cfgs = [_stub_phase_config(n) for n in ("A", "B", "C")]
    masks = []
    for row in (0, 1, 3):
        m = np.zeros((4, 4), dtype=bool)
        m[row, :] = True
        masks.append(m)
    consensus = np.full((4, 4), -1, dtype=np.int32)
    for i, row in enumerate((0, 1, 3)):
        consensus[row, :] = i
    pmrs = {n: _make_stub_pmr(n, 4, 4, m) for n, m in zip("ABC", masks)}
    pmrs["A"].indexing_result.metadata["caveats"] = [entry("A")]
    pmrs["B"].indexing_result.metadata["caveats"] = [entry("B", "A")]    # A again

    def side_effect(*args, **kwargs):
        return pmrs[kwargs["phase_config"].name]

    with patch("indexing_controller.run_single_phase_method", side_effect=side_effect), \
         patch("indexing_controller.build_consensus_xmap", return_value=MagicMock()):
        result = run_per_phase_indexing(
            signal=sig, detector=None, method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=cfgs, methods=[IndexingMethod.HOUGH]),
            phase_configs=cfgs, masks_per_phase=masks, consensus_map=consensus)
    assert result.metadata["caveats"] == [
        {"code": CODE, "phases": ["A", "B"], "text": sentence_many(["A", "B"])}]


def test_every_site_that_merges_the_citation_trail_merges_the_caveats_too():
    """The merge sites (competitive multi-phase, EDS-weighted Hough, phase-map routed)
    call ``merge_provenance`` and ``merge_build_records`` together. A new site that
    calls only the first would drop the caveat; count the calls in the two files."""
    root = Path(__file__).resolve().parents[1]
    counts = {}
    for rel in ("indexing_controller.py", "backend/api/routes/indexing.py"):
        src = (root / rel).read_text(encoding="utf-8")
        counts[rel] = (len(re.findall(r"^\s+merge_provenance\(", src, re.M)),
                       len(re.findall(r"^\s+merge_build_records\(", src, re.M)))
    assert counts == {"indexing_controller.py": (1, 1), "backend/api/routes/indexing.py": (2, 2)}


# --------------------------------------------------------------------------
# The real hough_index_patterns with a stubbed indexer
# --------------------------------------------------------------------------

class _FakeIndexer:
    def index_pats(self, patsin, verbose=0, chunksize=528):
        n = patsin.shape[0]
        return [{"cm": np.full(n, 0.5)}], None, None, None


def _fake_xmap_from_hough(data, phase_list, navigation_shape, step_sizes):
    n = int(np.prod(navigation_shape))
    ncols = navigation_shape[-1] if len(navigation_shape) == 2 else n
    x = np.tile(np.arange(ncols, dtype=float), max(n // ncols, 1))
    y = np.repeat(np.arange(max(n // ncols, 1), dtype=float), ncols)
    eul = np.deg2rad(np.array([[i * 7.0, 20.0, 30.0] for i in range(n)]))
    return CrystalMap(rotations=Rotation.from_euler(eul), phase_id=np.zeros(n, dtype=int),
                      x=x, y=y, phase_list=phase_list)


@pytest.fixture
def stubbed_indexer(monkeypatch):
    import ebsd_utils as eu
    import kikuchipy.indexing._hough_indexing as kh
    hr.clear_specs()
    monkeypatch.setattr(eu, "prepare_reflectors", lambda pl, *a, **k: None)
    monkeypatch.setattr(eu, "create_indexer", lambda *a, **k: _FakeIndexer())
    monkeypatch.setattr(kh, "xmap_from_hough_indexing_data", _fake_xmap_from_hough)
    yield
    hr.clear_specs()


def _signal(n_rows=2, n_cols=3):
    return SimpleNamespace(data=np.zeros((n_rows, n_cols, 8, 8), dtype=np.uint8))


def _index(plist_, mask=None):
    lines = []
    result = hough_index_patterns(
        _signal(), plist_, SimpleNamespace(pc=np.array([[0.5, 0.5, 0.5]])),
        IndexingConfig(method=IndexingMethod.HOUGH), mask,
        progress_callback=lambda msg, pct=None: lines.append(msg))
    return result, lines


def test_a_hough_run_on_an_affected_phase_finishes_and_carries_the_caveat(stubbed_indexer):
    result, lines = _index(plist(phase("Cr2N", 162)))
    assert result.xmap.size == 6                                     # the run ran to the end
    assert result.metadata["caveats"] == [
        {"code": CODE, "phases": ["Cr2N"], "text": sentence_one("Cr2N")}]
    warned = [ln for ln in lines if ln.startswith("⚠")]
    assert warned == ["⚠ " + sentence_one("Cr2N")]


def test_the_caveat_survives_a_partial_selection(stubbed_indexer):
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = mask[1, 2] = True
    result, _ = _index(plist(phase("Cr2N", 162)), mask)
    assert result.xmap.size == 2
    assert result.metadata["caveats"][0]["phases"] == ["Cr2N"]


def test_a_hough_run_outside_the_seven_groups_is_unchanged(stubbed_indexer):
    result, lines = _index(plist(phase("Al", 225)))
    assert "caveats" not in result.metadata
    assert not [ln for ln in lines if ln.startswith("⚠")]
    # band data, the citation trail and the reflector record: what a run has always had
    assert set(result.metadata) == {"band_data", "provenance", "hough_reflectors"}


def test_a_multi_phase_run_names_only_the_affected_phases(stubbed_indexer):
    result, lines = _index(plist(phase("Al", 225), phase("Cr2N", 162), phase("Mg", 194)))
    assert [c["phases"] for c in result.metadata["caveats"]] == [["Cr2N"]]
    assert len([ln for ln in lines if ln.startswith("⚠")]) == 1


def test_a_failing_check_never_costs_the_run(stubbed_indexer, monkeypatch):
    def boom(_pl):
        raise RuntimeError("cannot say")
    monkeypatch.setattr(hc, "hough_caveats", boom)
    result, _ = _index(plist(phase("Cr2N", 162)))
    assert result.xmap.size == 6 and "caveats" not in result.metadata


# --------------------------------------------------------------------------
# Task result and last result (the real route and its background task)
# --------------------------------------------------------------------------

CR2N_CIF = """data_Cr2N
_cell_length_a 4.80000
_cell_length_b 4.80000
_cell_length_c 4.47200
_cell_angle_alpha 90.0000
_cell_angle_beta 90.0000
_cell_angle_gamma 120.0000
_symmetry_space_group_name_H-M 'P -3 1 m'
_symmetry_Int_Tables_number 162
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Cr1 Cr 0.346000 0.000000000000 0.244000 1.0000
N2 N 0.333333333333 0.666666666667 0.500000000000 0.7400
N3 N 0.333333333333 0.666666666667 0.000000000000 0.1200
N4 N 0.000000000000 0.000000000000 0.000000000000 1.0000
"""
NI_CIF = """data_Ni
_cell_length_a 3.52
_cell_length_b 3.52
_cell_length_c 3.52
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni1 Ni 0 0 0
"""


@pytest.fixture
def route(monkeypatch, tmp_path, stubbed_indexer):
    from fastapi.testclient import TestClient
    from kikuchipy.detectors import EBSDDetector
    import kikuchipy as kp
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from backend.api.services.calibration_store import CalibrationStore

    saved, saved_active = dict(I._result_registry), I._active_result_id
    saved_tasks = dict(I._indexing_tasks)
    sig = kp.data.nickel_ebsd_small()
    sig.detector = EBSDDetector(shape=(60, 60), pc=(0.48, 0.52, 0.60), sample_tilt=70.0)
    store = CalibrationStore()
    store.register("ds", sig)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: sig)
    up1 = tmp_path / "scan.up1"
    up1.write_bytes(b"not an hdf5 file")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(up1), raising=False)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="indexing": fn())
    client = TestClient(app)

    def run(cif_text, stem):
        cif = tmp_path / f"{stem}.cif"
        cif.write_text(cif_text, encoding="utf-8")
        resp = client.post("/api/indexing/start", json={"method": "hough", "cif_paths": [str(cif)]})
        assert resp.status_code == 200, resp.text
        task = I._indexing_tasks[resp.json()["task_id"]]
        assert task["status"] == "completed", task
        return task["result"], client.get("/api/indexing/result/last").json()

    yield run
    I._result_registry.clear()
    I._result_registry.update(saved)
    I._active_result_id = saved_active
    I._indexing_tasks.clear()                 # other tests read the progress of every task
    I._indexing_tasks.update(saved_tasks)


def test_task_result_and_last_result_carry_the_caveat(route):
    task_result, last = route(CR2N_CIF, "Cr2N")
    want = [{"code": CODE, "phases": ["Cr2N"], "text": sentence_one("Cr2N")}]
    assert task_result["caveats"] == want
    assert last["caveats"] == want


def test_the_answers_outside_the_seven_groups_have_no_such_key(route):
    task_result, last = route(NI_CIF, "Ni")
    assert "caveats" not in task_result and "caveats" not in last
    assert set(task_result) == {"n_indexed", "original_shape", "method", "has_confidence",
                                "mean_ci", "phases"}


# --------------------------------------------------------------------------
# The saved file, and its way back
# --------------------------------------------------------------------------

CAVEATS = [{"code": CODE, "phases": ["Cr2N", "Z"], "text": sentence_many(["Cr2N", "Z"])}]


@pytest.mark.parametrize("fmt", ["h5", "h5_light"])
def test_the_caveat_is_saved_next_to_the_reflector_record_and_comes_back(monkeypatch, tmp_path, fmt):
    from tests.citations.test_citations_export import (
        _fake_active_result, _patch_interactive_export_seams)
    result = _fake_active_result([{"key": "indexing.hough", "params": {"orienta_version": "x"}}])
    result.metadata["caveats"] = CAVEATS
    result.metadata["hough_reflectors"] = {"Al": {"mode": "default", "n_rows": 5}}
    ix = _patch_interactive_export_seams(monkeypatch, result)
    out = tmp_path / f"cav_{fmt}.h5"
    asyncio.run(ix.export_indexing_result(ix.ExportRequest(
        format=fmt, filename=str(out), include_eds=False, include_detector=False)))
    with h5py.File(out, "r") as f:
        attrs = f["Indexing/Citations"].attrs
        assert json.loads(attrs["caveats"]) == CAVEATS
        assert "hough_reflectors" in attrs                  # next to it, not instead of it
    rid = None
    try:
        rid = asyncio.run(ix.import_h5_result(ix.ImportH5Request(path=str(out))))["result_id"]
        assert ix._result_registry[rid].metadata["caveats"] == CAVEATS
    finally:
        if rid is not None:
            ix._result_registry.pop(rid, None)


def test_a_result_without_caveats_writes_no_attribute(monkeypatch, tmp_path):
    from tests.citations.test_citations_export import (
        _fake_active_result, _patch_interactive_export_seams)
    result = _fake_active_result([{"key": "indexing.hough", "params": {}}])
    ix = _patch_interactive_export_seams(monkeypatch, result)
    out = tmp_path / "none.h5"
    asyncio.run(ix.export_indexing_result(ix.ExportRequest(
        format="h5_light", filename=str(out), include_eds=False, include_detector=False)))
    with h5py.File(out, "r") as f:
        assert "caveats" not in f["Indexing/Citations"].attrs


@pytest.mark.parametrize("bad", [
    "not json {", json.dumps({"code": CODE}), json.dumps(["x"]), json.dumps([{"code": CODE}]),
    json.dumps([{"code": CODE, "phases": "Cr2N", "text": "t"}]),
    json.dumps([{"code": CODE, "phases": [3], "text": "t"}]),
    json.dumps([{"code": 3, "phases": ["A"], "text": "t"}]),
])
def test_a_malformed_record_in_a_file_is_ignored(tmp_path, bad):
    from backend.api.routes import indexing as ix
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.create_group("Indexing/Citations").attrs["caveats"] = bad
    assert ix._read_caveats(p) is None


def test_a_good_record_is_read_and_unknown_keys_are_dropped(tmp_path):
    from backend.api.routes import indexing as ix
    p = tmp_path / "ok.h5"
    with h5py.File(p, "w") as f:
        f.create_group("Indexing/Citations").attrs["caveats"] = json.dumps(
            [dict(CAVEATS[0], extra="x"), {"code": 5}])
    assert ix._read_caveats(p) == CAVEATS
    with h5py.File(p, "w") as f:
        f.create_group("Indexing")
    assert ix._read_caveats(p) is None


# --------------------------------------------------------------------------
# The batch: checkpoint -> run record -> both .h5 writers
# --------------------------------------------------------------------------

def _two_phase_checkpoint(tmp_path):
    from backend.api.services.checkpoint_writer import CheckpointWriter

    source = tmp_path / "Two.h5oina"
    with h5py.File(source, "w") as f:
        f.create_group("1/EBSD/Data")
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("Beam Voltage", data=np.float32(20.0))
        hdr.create_dataset("Tilt Angle", data=np.float32(70.0))
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="rec", method="hough", step_size_um=0.5)
    ci = np.full((3, 4), 0.6, dtype=np.float32)
    ori = np.zeros((3, 4, 3), dtype=np.float32)
    common = {"ci_mean": 0.6, "duration_sec": 1.0, "point_group": "m-3m",
              "lattice_constants": [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]}
    cw.write_phase_result("Cr2N", ci, ori, dict(
        common, phase_file="Cr2N.cif", space_group=162, caveats=[entry("Cr2N")]))
    cw.write_phase_result("Z", ci, ori, dict(
        common, phase_file="Z.cif", space_group=149, caveats=[entry("Z")]))
    cw.write_phase_result("Ni", ci, ori, dict(common, phase_file="Ni.cif", space_group=225))
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


def test_the_checkpoint_merges_the_caveats_of_its_phases(tmp_path):
    from backend.api.services.checkpoint_writer import CheckpointWriter

    source, _cw = _two_phase_checkpoint(tmp_path)
    record = CheckpointWriter(str(source)).read_run_record()      # a fresh reader: a resumed batch
    assert record["caveats"] == [
        {"code": CODE, "phases": ["Cr2N", "Z"], "text": sentence_many(["Cr2N", "Z"])}]


def test_a_checkpoint_without_caveats_gives_the_record_it_always_gave(tmp_path):
    from backend.api.services.checkpoint_writer import CheckpointWriter

    source = tmp_path / "Old.h5oina"
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="rec", method="hough", step_size_um=0.5)
    cw.write_phase_result("Ni", np.zeros((3, 4), np.float32), np.zeros((3, 4, 3), np.float32),
                          {"phase_file": "Ni.cif", "ci_mean": 0.1, "duration_sec": 1.0})
    assert cw.read_run_record() == {"citation_steps": None, "hough_reflectors": None}


@pytest.mark.parametrize("fmt", ["h5_light", "h5_rich"])
def test_export_all_hands_the_caveats_to_each_h5_writer(tmp_path, fmt):
    from backend.api.services.result_exporter import export_all

    source, cw = _two_phase_checkpoint(tmp_path)
    record = cw.read_run_record()
    out = tmp_path / "out"
    out.mkdir()
    results = export_all(source_h5_path=str(source), checkpoint_path=cw.checkpoint_path,
                         output_dir=str(out), formats=[fmt], include_eds=False, **record)
    assert results.get(fmt), results.get("_errors")
    with h5py.File(results[fmt], "r") as f:
        got = json.loads(f["Indexing/Citations"].attrs["caveats"])
    assert got == [{"code": CODE, "phases": ["Cr2N", "Z"], "text": sentence_many(["Cr2N", "Z"])}]


def test_a_batch_job_on_a_trigonal_phase_is_exported_with_the_caveat(
        world, stubbed_indexer, monkeypatch):
    """The real batch route, manager, checkpoint and exporter. The CIF the job reads is
    replaced by a space-group-162 phase and the indexer is the stub; the job names the
    phase after its file ("Ni")."""
    import ebsd_utils as eu

    monkeypatch.setattr(eu, "hough_phase_from_cif", lambda path, *a, **k: phase("x", 162))
    mgr, scan, cif, tmp = world
    batch_id, out = _run(mgr, scan, cif, tmp, ["h5_light"])
    (job,) = mgr.get_jobs(batch_id)
    assert job["status"] == "done", job
    with h5py.File(_light(out), "r") as f:
        got = json.loads(f["Indexing/Citations"].attrs["caveats"])
    assert got == [{"code": CODE, "phases": ["Ni"], "text": sentence_one("Ni")}]


def test_the_legacy_batch_auto_export_carries_the_caveat(monkeypatch, tmp_path):
    """``POST /api/indexing/batch/start`` writes its own small .h5 per dataset. The
    seam is the Hough function (it returns a 3 x 3 map that carries the caveat); the
    route, its loader and its export are real."""
    import indexing_controller
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from backend.api.services.calibration_store import CalibrationStore
    from tests.test_batch_same_stem import NI_CIF as BATCH_CIF, OSC_B, _write_osc, _write_up1
    from tests.test_result_keeps_its_run_pc import _fake_hough

    def hough_with_caveat(*a, **k):
        result = _fake_hough(*a, **k)
        result.metadata["caveats"] = CAVEATS
        return result

    store = CalibrationStore()
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    monkeypatch.setattr(indexing_controller, "hough_index_patterns", hough_with_caveat)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    scan = _write_up1(tmp_path / "Scan9.up1")
    _write_osc(tmp_path / "Scan9.osc", OSC_B)
    cif = tmp_path / "Ni.cif"
    cif.write_text(BATCH_CIF, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    saved = dict(I._batch_state)
    try:
        resp = TestClient(app).post("/api/indexing/batch/start", json={
            "datasets": [{"file_path": scan, "method": "hough", "cif_paths": [str(cif)]}],
            "auto_export": True, "export_dir": str(out)})
        assert resp.status_code == 200, resp.text
        assert I._batch_state["results"][-1]["error"] is None
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved)
    (exported,) = out.glob("*_indexed.h5")
    with h5py.File(exported, "r") as f:
        assert json.loads(f["Indexing/Citations"].attrs["caveats"]) == CAVEATS


def test_a_batch_job_on_a_cubic_phase_writes_no_caveat(world, stubbed_indexer, monkeypatch):
    import ebsd_utils as eu

    monkeypatch.setattr(eu, "hough_phase_from_cif", lambda path, *a, **k: phase("x", 225))
    mgr, scan, cif, tmp = world
    batch_id, out = _run(mgr, scan, cif, tmp, ["h5_light"])
    assert mgr.get_jobs(batch_id)[0]["status"] == "done"
    with h5py.File(_light(out), "r") as f:
        assert "caveats" not in f["Indexing/Citations"].attrs
