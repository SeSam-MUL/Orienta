"""Hough on a trigonal or hexagonal phase with a Laue class below 6/mmm carries a note.

Rendered patterns of such structures indexed with PyEBSDIndex give, in a share of the
patterns that depends on the structure, an orientation that differs from the true one by
a symmetry of the hexagonal lattice that is not a symmetry of the crystal (60 degrees about
c, or 180 degrees about an axis in the basal plane). ``result.metadata["caveats"]`` then
carries one of two codes next to (not instead of) the strongest ``houghTrigonalSetting``:
``houghHexagonalLaue`` (Laue classes -3 and 6/m: a warning, the share is 23 to 68 %) or
``houghHexagonalVariants`` (the other groups below 6/mmm: a soft hint, the progress line
reads ``Hough note: ...`` and carries no warning sign). Numbers, settings and the run
itself are untouched.

Every expectation is written out here by hand: the space-group numbers (positive and
negative), the sentences, the order of the two codes. Nothing is derived from the code under
test. The indexer is the one seam (a stub, as in ``test_hough_trigonal_caveat.py``, whose
helpers this file reuses); the controller function, the merges, the route, the exporters and
the importer are the real ones.
"""
import asyncio
import json
import logging
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

import hough_caveats as hc  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from indexing_controller import (  # noqa: E402
    ComparisonConfig, IndexingMethod, run_per_phase_indexing)
from tests.test_hough_trigonal_caveat import (  # noqa: E402,F401  (fixtures: stubbed_indexer, route)
    CR2N_CIF, NI_CIF, SEVEN, R, _index, phase, plist, route, sentence_many as trigonal_many,
    sentence_one as trigonal_one, stubbed_indexer)
from tests.test_per_phase_indexing import (  # noqa: E402
    _make_stub_pmr, _stub_phase_config, _stub_signal)

CODE = "houghHexagonalVariants"
LAUE = "houghHexagonalLaue"
TRIGONAL = "houghTrigonalSetting"
# Laue classes -3 and 6/m: 143-148 and 168-176, written out
LAUE_SGS = (143, 144, 145, 146, 147, 148, 168, 169, 170, 171, 172, 173, 174, 175, 176)
# 150, 152, 154, 156, 158, 164, 165 and 167 (corundum, haematite), written out
AFFECTED = (150, 152, 154, 156, 158, 164, 165, 167)
# R lattices (155, 160, 161, 166), the 6/mmm groups 177 and up, the seven, and
# groups of other crystal systems
NEGATIVE = (155, 160, 161, 166, 177, 191, 194, 225, 12, 62, 1, 2, 75, 99, 142)
BRUCITE_CIF = """data_Mg(OH)2
_cell_length_a 3.147
_cell_length_b 3.147
_cell_length_c 4.769
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 120.0
_symmetry_space_group_name_H-M 'P -3 m 1'
_symmetry_Int_Tables_number 164
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mg1 Mg 0.0 0.0 0.0
O1 O 0.333333333333 0.666666666667 0.2212
"""


def sentence_one(name):
    return (f"Phase {name}: Hough cannot always tell apart orientations related by a rotation "
            "that is a symmetry of the hexagonal cell but not of this crystal (60 degrees "
            "about c). For important results, Spherical or Dictionary indexing may help.")


def sentence_many(names):
    return ("Phases " + ", ".join(names) + ": Hough cannot always tell apart orientations "
            "related by a rotation that is a symmetry of the hexagonal cell but not of "
            "these crystals (60 degrees about c). For important results, Spherical or "
            "Dictionary indexing may help.")


def laue_one(name):
    return (f"Phase {name}: Hough often returns an orientation related to the true one by a "
            "rotation that is a symmetry of the hexagonal cell but not of this crystal "
            "(depending on the crystal class, 180 degrees about an axis in the basal plane "
            "or 60 degrees about c). Spherical or Dictionary indexing may help.")


def laue_many(names):
    return ("Phases " + ", ".join(names) + ": Hough often returns an orientation related to "
            "the true one by a rotation that is a symmetry of the hexagonal cell but not of "
            "these crystals (depending on the crystal class, 180 degrees about an axis in "
            "the basal plane or 60 degrees about c). Spherical or Dictionary indexing may "
            "help.")


def stub(name, n):
    return (0, SimpleNamespace(name=name, space_group=SimpleNamespace(number=n)))


def codes(caveats):
    return [c["code"] for c in caveats]


# --------------------------------------------------------------------------
# Which phases
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sg", AFFECTED)
def test_each_listed_space_group_gets_only_the_soft_note(sg):
    assert hc.hough_caveats(plist(phase("P", sg))) == [
        {"code": CODE, "phases": ["P"], "text": sentence_one("P")}]


@pytest.mark.parametrize("sg", LAUE_SGS)
def test_each_laue_space_group_gets_only_the_strong_note(sg):
    assert hc.hough_caveats(plist(phase("P", sg))) == [
        {"code": LAUE, "phases": ["P"], "text": laue_one("P")}]


@pytest.mark.parametrize("sg", NEGATIVE)
def test_the_other_space_groups_get_nothing(sg):
    assert hc.hough_caveats(plist(phase("P", sg))) == []


@pytest.mark.parametrize("sg", SEVEN)
def test_the_seven_trigonal_setting_groups_get_only_the_first_code(sg):
    assert codes(hc.hough_caveats(plist(phase("P", sg)))) == [TRIGONAL]


def test_the_lists_are_exactly_the_ones_written_out_here_and_overlap_nothing():
    assert hc.HEXAGONAL_VARIANT_SPACE_GROUPS == frozenset(AFFECTED)
    assert hc.HEXAGONAL_LAUE_SPACE_GROUPS == frozenset(LAUE_SGS)
    assert not hc.HEXAGONAL_VARIANT_SPACE_GROUPS & hc.HEXAGONAL_LAUE_SPACE_GROUPS
    from phase_cell import PER_REFLECTION_SPACE_GROUPS
    for lst in (hc.HEXAGONAL_VARIANT_SPACE_GROUPS, hc.HEXAGONAL_LAUE_SPACE_GROUPS):
        assert not lst & set(SEVEN)
        assert not lst & set(PER_REFLECTION_SPACE_GROUPS)
    # the two lists together are the trigonal and hexagonal groups of Laue class below
    # 6/mmm that carry a note: 143-176 minus the seven, minus the R lattices of -3m1
    # that were measured clean (155, 160, 161, 166)
    both = hc.HEXAGONAL_VARIANT_SPACE_GROUPS | hc.HEXAGONAL_LAUE_SPACE_GROUPS
    assert both == set(range(143, 177)) - set(SEVEN) - {155, 160, 161, 166}


def test_a_space_group_gets_at_most_one_code():
    for sg in range(1, 231):
        found = [bool(f([stub("P", sg)])) for f in
                 (hc.trigonal_setting_phases, hc.hexagonal_laue_phases, hc.hexagonal_variant_phases)]
        assert sum(found) <= 1, sg


def test_a_phase_without_a_space_group_gets_nothing():
    from diffpy.structure import Atom, Lattice, Structure
    from orix.crystal_map import Phase
    p = Phase(name="nosg", structure=Structure(atoms=[Atom("Al", xyz=(0, 0, 0))],
                                               lattice=Lattice(4, 4, 4, 90, 90, 90)))
    assert p.space_group is None
    assert hc.hough_caveats(plist(p)) == []


def test_diffpy_has_exactly_these_alternative_numbers_in_the_trigonal_range():
    """The precondition of the next test: the alternative settings of the groups 143 to
    176 in diffpy are the rhombohedral axes of 146, 148 (the hexagonal-axes pair of the
    list), of 167 (in the list) and of the R groups 155, 160, 161, 166 (outside)."""
    from diffpy.structure.spacegroups import SpaceGroupList
    alt = sorted(sg.number for sg in SpaceGroupList if sg.number > 230 and sg.number % 1000 in range(143, 177))
    assert alt == [1146, 1148, 1155, 1160, 1161, 1166, 1167]
    names = {sg.number: sg.pdb_name for sg in SpaceGroupList if sg.number in (146, 1146, 148, 1148)}
    assert names == {146: "H 3", 1146: "R 3", 148: "H -3", 1148: "R -3"}


def test_146_148_and_167_get_their_note_in_both_axes_and_the_other_r_groups_in_neither():
    # rhombohedral axes of 146, 148 and 167: the same crystal, so the same note
    assert hc.hexagonal_laue_phases([stub("A", 146), stub("B", 1146), stub("C", 148),
                                     stub("D", 1148)]) == ["A", "B", "C", "D"]
    assert hc.hexagonal_variant_phases([stub("E", 167), stub("F", 1167)]) == ["E", "F"]
    # each list takes only its own groups
    assert hc.hexagonal_variant_phases([stub("A", 146), stub("B", 1148)]) == []
    assert hc.hexagonal_laue_phases([stub("E", 167), stub("F", 1167)]) == []
    # the other R groups reduce to numbers that are not in either list, in either axes
    for n in (155, 1155, 160, 1160, 161, 1161, 166, 1166):
        assert hc.hexagonal_variant_phases([stub("R", n)]) == [], n
        assert hc.hexagonal_laue_phases([stub("R", n)]) == [], n
    # an alternative number never reaches the seven's code
    assert hc.trigonal_setting_phases([stub("A", 1146), stub("B", 1148)]) == []


def test_a_real_phase_in_rhombohedral_axes_is_handled_like_the_stub_if_orix_can_build_it():
    from diffpy.structure import Atom, Lattice, Structure
    from orix.crystal_map import Phase
    lat = Lattice(5.0, 5.0, 5.0, 60, 60, 60)
    st = Structure(atoms=[Atom("Al", xyz=(.11, .23, .31))], lattice=lat)
    try:
        p = Phase(name="Rh", space_group=1146, structure=st)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"orix cannot hold space group 1146: {exc!r}")
    assert hc.hough_caveats(plist(p)) == [{"code": LAUE, "phases": ["Rh"], "text": laue_one("Rh")}]


def test_only_the_affected_phases_are_named_in_list_order_each_once_per_code():
    pl = plist(phase("Al", 225), phase("Cr2N", 162), phase("Brucite", 164), phase("Mg", 194),
               phase("Z", 149), phase("Quartz", 154), phase("Berlinite", 147),
               phase("Apatite", 176))
    assert hc.hough_caveats(pl) == [
        {"code": TRIGONAL, "phases": ["Cr2N", "Z"], "text": trigonal_many(["Cr2N", "Z"])},
        {"code": LAUE, "phases": ["Berlinite", "Apatite"], "text": laue_many(["Berlinite", "Apatite"])},
        {"code": CODE, "phases": ["Brucite", "Quartz"], "text": sentence_many(["Brucite", "Quartz"])},
    ]


def test_the_stronger_note_comes_first_whatever_the_phase_order():
    pl = plist(phase("Brucite", 164), phase("Apatite", 176), phase("Cr2N", 162))
    assert codes(hc.hough_caveats(pl)) == [TRIGONAL, LAUE, CODE]


def test_the_sentences_name_no_percentage_and_no_cause():
    for text in (hc.hexagonal_variants_text(["A", "B"]), hc.hexagonal_variants_text(["A"]),
                 hc.hexagonal_laue_text(["A", "B"]), hc.hexagonal_laue_text(["A"])):
        assert "%" not in text
        assert "therefore" not in text and "pattern match" not in text.lower()
        assert text.endswith("may help.")


# --------------------------------------------------------------------------
# The progress line and the log
# --------------------------------------------------------------------------

def test_the_progress_line_of_the_soft_note_has_no_warning_sign():
    soft = {"code": CODE, "phases": ["A"], "text": "T"}
    strong = {"code": TRIGONAL, "phases": ["A"], "text": "T"}
    laue = {"code": LAUE, "phases": ["A"], "text": "T"}
    assert hc.log_and_progress(soft) == (False, "Hough note: T")
    assert hc.log_and_progress(strong) == (True, "⚠ T")
    assert hc.log_and_progress(laue) == (True, "⚠ T")
    assert hc.SOFT_CODES == frozenset({CODE})


# --------------------------------------------------------------------------
# The merge for two codes
# --------------------------------------------------------------------------

def tri(*names):
    return {"code": TRIGONAL, "phases": list(names), "text": "old"}


def hexa(*names):
    return {"code": CODE, "phases": list(names), "text": "old"}


def lau(*names):
    return {"code": LAUE, "phases": list(names), "text": "old"}


def test_the_merge_keeps_two_codes_apart_unites_each_and_orders_them():
    target = R({})
    hr.merge_build_records(target, [
        R({"caveats": [hexa("H1")]}), R({"caveats": [tri("T1")]}),
        R({"caveats": [hexa("H2", "H1"), tri("T2")]}), R({}), None])
    assert target.metadata["caveats"] == [
        {"code": TRIGONAL, "phases": ["T1", "T2"], "text": trigonal_many(["T1", "T2"])},
        {"code": CODE, "phases": ["H1", "H2"], "text": sentence_many(["H1", "H2"])},
    ]


def test_the_merge_orders_three_codes_strongest_first_and_words_each():
    target = R({})
    hr.merge_build_records(target, [
        R({"caveats": [hexa("H1")]}), R({"caveats": [lau("L1")]}), R({"caveats": [tri("T1")]}),
        R({"caveats": [lau("L2", "L1")]})])
    assert target.metadata["caveats"] == [
        {"code": TRIGONAL, "phases": ["T1"], "text": trigonal_one("T1")},
        {"code": LAUE, "phases": ["L1", "L2"], "text": laue_many(["L1", "L2"])},
        {"code": CODE, "phases": ["H1"], "text": sentence_one("H1")},
    ]


def test_the_merge_words_a_single_phase_in_the_singular():
    target = R({"caveats": [hexa("H1")]})
    hr.merge_build_records(target, [R({"caveats": [hexa("H1")]})])
    assert target.metadata["caveats"] == [{"code": CODE, "phases": ["H1"], "text": sentence_one("H1")}]


def test_an_unknown_code_stays_after_the_known_ones_with_its_own_text():
    target = R({})
    hr.merge_build_records(target, [R({"caveats": [
        {"code": "fromTheFuture", "phases": ["A"], "text": "an English sentence"}, hexa("H"), tri("T")]})])
    assert codes(target.metadata["caveats"]) == [TRIGONAL, CODE, "fromTheFuture"]
    assert target.metadata["caveats"][2]["text"] == "an English sentence"
    target = R({})
    hr.merge_build_records(target, [R({"caveats": [
        {"code": "fromTheFuture", "phases": ["A"], "text": "x"}, hexa("H"), lau("L")]})])
    assert codes(target.metadata["caveats"]) == [LAUE, CODE, "fromTheFuture"]


def test_the_per_phase_merge_of_a_phase_map_run_carries_both_codes():
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
    pmrs["A"].indexing_result.metadata["caveats"] = [hexa("A")]
    pmrs["B"].indexing_result.metadata["caveats"] = [tri("B")]
    # C has none

    def side_effect(*args, **kwargs):
        return pmrs[kwargs["phase_config"].name]

    with patch("indexing_controller.run_single_phase_method", side_effect=side_effect), \
         patch("indexing_controller.build_consensus_xmap", return_value=MagicMock()):
        result = run_per_phase_indexing(
            signal=sig, detector=None, method=IndexingMethod.HOUGH,
            config=ComparisonConfig(phases=cfgs, methods=[IndexingMethod.HOUGH]),
            phase_configs=cfgs, masks_per_phase=masks, consensus_map=consensus)
    assert result.metadata["caveats"] == [
        {"code": TRIGONAL, "phases": ["B"], "text": trigonal_one("B")},
        {"code": CODE, "phases": ["A"], "text": sentence_one("A")}]


# --------------------------------------------------------------------------
# The real hough_index_patterns with a stubbed indexer
# --------------------------------------------------------------------------

def _soft_lines(lines):
    return [ln for ln in lines if ln.startswith("Hough note:")]


def _warn_lines(lines):
    return [ln for ln in lines if ln.startswith("⚠")]


def test_a_hough_run_on_a_listed_phase_finishes_with_the_note_and_a_plain_progress_line(
        stubbed_indexer, caplog):
    with caplog.at_level(logging.INFO, logger="indexing_controller"):
        result, lines = _index(plist(phase("Brucite", 164)))
    assert result.xmap.size == 6                                     # the run ran to the end
    assert result.metadata["caveats"] == [
        {"code": CODE, "phases": ["Brucite"], "text": sentence_one("Brucite")}]
    assert _soft_lines(lines) == ["Hough note: " + sentence_one("Brucite")]
    assert _warn_lines(lines) == []
    # logged, but not as a warning
    hits = [r for r in caplog.records if "Hough cannot always tell apart" in r.getMessage()]
    assert [r.levelno for r in hits] == [logging.INFO]


def test_a_run_with_both_kinds_names_each_phase_under_its_own_code(stubbed_indexer):
    result, lines = _index(plist(phase("Al", 225), phase("Cr2N", 162), phase("Brucite", 164),
                                 phase("Mg", 194)))
    assert result.metadata["caveats"] == [
        {"code": TRIGONAL, "phases": ["Cr2N"], "text": trigonal_one("Cr2N")},
        {"code": CODE, "phases": ["Brucite"], "text": sentence_one("Brucite")}]
    assert _warn_lines(lines) == ["⚠ " + trigonal_one("Cr2N")]
    assert _soft_lines(lines) == ["Hough note: " + sentence_one("Brucite")]


def test_a_laue_phase_gets_the_strong_note_and_a_warning_line(stubbed_indexer, caplog):
    with caplog.at_level(logging.INFO, logger="indexing_controller"):
        result, lines = _index(plist(phase("Berlinite", 147)))
    assert result.xmap.size == 6
    assert result.metadata["caveats"] == [
        {"code": LAUE, "phases": ["Berlinite"], "text": laue_one("Berlinite")}]
    assert _warn_lines(lines) == ["⚠ " + laue_one("Berlinite")]
    assert _soft_lines(lines) == []
    hits = [r for r in caplog.records if "Hough often returns an orientation" in r.getMessage()]
    assert [r.levelno for r in hits] == [logging.WARNING]


def test_the_seven_groups_alone_get_no_soft_line(stubbed_indexer):
    result, lines = _index(plist(phase("Cr2N", 162)))
    assert codes(result.metadata["caveats"]) == [TRIGONAL]
    assert _soft_lines(lines) == [] and len(_warn_lines(lines)) == 1


@pytest.mark.parametrize("sg", [225, 194, 191, 177, 155, 160, 161, 166, 12, 62])
def test_a_run_outside_the_listed_groups_is_unchanged(stubbed_indexer, sg):
    result, lines = _index(plist(phase("X", sg)))
    assert "caveats" not in result.metadata
    assert _soft_lines(lines) == [] and _warn_lines(lines) == []
    assert set(result.metadata) == {"band_data", "provenance", "hough_reflectors"}


def test_a_failing_laue_check_never_costs_the_run(stubbed_indexer, monkeypatch):
    def boom(_pl):
        raise RuntimeError("cannot say")
    monkeypatch.setattr(hc, "hexagonal_laue_phases", boom)
    result, lines = _index(plist(phase("Berlinite", 147)))
    assert result.xmap.size == 6 and "caveats" not in result.metadata
    assert _warn_lines(lines) == []


def test_a_failing_check_never_costs_the_run_and_leaves_no_half_note(stubbed_indexer, monkeypatch):
    def boom(_pl):
        raise RuntimeError("cannot say")
    monkeypatch.setattr(hc, "hexagonal_variant_phases", boom)
    result, lines = _index(plist(phase("Brucite", 164)))
    assert result.xmap.size == 6 and "caveats" not in result.metadata
    assert _soft_lines(lines) == []


# --------------------------------------------------------------------------
# Task result and last result (the real route and its background task)
# --------------------------------------------------------------------------

def test_task_result_and_last_result_carry_the_soft_note(route):
    task_result, last = route(BRUCITE_CIF, "Brucite")
    want = [{"code": CODE, "phases": ["Brucite"], "text": sentence_one("Brucite")}]
    assert task_result["caveats"] == want
    assert last["caveats"] == want


BERLINITE_LIKE_CIF = """data_Laue
_cell_length_a 4.9
_cell_length_b 4.9
_cell_length_c 11.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 120.0
_symmetry_space_group_name_H-M 'P -3'
_symmetry_Int_Tables_number 147
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al1 Al 0.33 0.11 0.20
O1 O 0.12 0.41 0.35
"""


def test_task_result_and_last_result_carry_the_strong_note(route):
    task_result, last = route(BERLINITE_LIKE_CIF, "Laue")
    want = [{"code": LAUE, "phases": ["Laue"], "text": laue_one("Laue")}]
    assert task_result["caveats"] == want
    assert last["caveats"] == want


def test_the_trigonal_cif_still_gets_only_its_own_code(route):
    task_result, last = route(CR2N_CIF, "Cr2N")
    assert codes(task_result["caveats"]) == [TRIGONAL] == codes(last["caveats"])


def test_a_cubic_cif_has_no_such_key(route):
    task_result, last = route(NI_CIF, "Ni")
    assert "caveats" not in task_result and "caveats" not in last


# --------------------------------------------------------------------------
# The saved file, and its way back, with both codes
# --------------------------------------------------------------------------

BOTH = [{"code": TRIGONAL, "phases": ["Cr2N"], "text": trigonal_one("Cr2N")},
        {"code": LAUE, "phases": ["Berlinite"], "text": laue_one("Berlinite")},
        {"code": CODE, "phases": ["Brucite", "Quartz"], "text": sentence_many(["Brucite", "Quartz"])}]


@pytest.mark.parametrize("fmt", ["h5", "h5_light"])
def test_both_codes_are_saved_and_come_back_in_order(monkeypatch, tmp_path, fmt):
    from tests.citations.test_citations_export import (
        _fake_active_result, _patch_interactive_export_seams)
    result = _fake_active_result([{"key": "indexing.hough", "params": {"orienta_version": "x"}}])
    result.metadata["caveats"] = BOTH
    ix = _patch_interactive_export_seams(monkeypatch, result)
    out = tmp_path / f"both_{fmt}.h5"
    asyncio.run(ix.export_indexing_result(ix.ExportRequest(
        format=fmt, filename=str(out), include_eds=False, include_detector=False)))
    with h5py.File(out, "r") as f:
        assert json.loads(f["Indexing/Citations"].attrs["caveats"]) == BOTH
    rid = None
    try:
        rid = asyncio.run(ix.import_h5_result(ix.ImportH5Request(path=str(out))))["result_id"]
        assert ix._result_registry[rid].metadata["caveats"] == BOTH
    finally:
        if rid is not None:
            ix._result_registry.pop(rid, None)


def test_the_checkpoint_merges_both_codes_of_its_phases(tmp_path):
    from backend.api.services.checkpoint_writer import CheckpointWriter

    source = tmp_path / "Two.h5oina"
    with h5py.File(source, "w") as f:
        f.create_group("1/EBSD/Data")
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="rec", method="hough", step_size_um=0.5)
    ci = np.full((3, 4), 0.6, dtype=np.float32)
    ori = np.zeros((3, 4, 3), dtype=np.float32)
    common = {"ci_mean": 0.6, "duration_sec": 1.0, "point_group": "m-3m",
              "lattice_constants": [3.52, 3.52, 3.52, 90.0, 90.0, 90.0]}
    cw.write_phase_result("Brucite", ci, ori, dict(
        common, phase_file="B.cif", space_group=164, caveats=[hexa("Brucite")]))
    cw.write_phase_result("Cr2N", ci, ori, dict(
        common, phase_file="C.cif", space_group=162, caveats=[tri("Cr2N")]))
    cw.write_phase_result("Quartz", ci, ori, dict(
        common, phase_file="Q.cif", space_group=154, caveats=[hexa("Quartz")]))
    cw.write_phase_result("Berlinite", ci, ori, dict(
        common, phase_file="L.cif", space_group=147, caveats=[lau("Berlinite")]))
    cw.compute_auto_assignment(confidence_threshold=0.0)
    record = CheckpointWriter(str(source)).read_run_record()      # a fresh reader: a resumed batch
    assert record["caveats"] == BOTH
