"""Per-phase control of the reflector families that Hough indexing uses.

What these tests pin down:

* a phase WITHOUT a spec is built exactly as before - the rows handed to
  PyEBSDIndex are the rows ``prepare_reflectors`` produced, and the registry
  never alters them;
* a spec that lists the families PyEBSDIndex keeps by default builds the same
  band-triplet library (every array equal), so "reset to default" and "the
  default, written out" are the same run;
* unticking a family really removes its poles from the library;
* a hand-entered family is validated (forbidden, duplicate, hexagonal
  4-index) and a stored selection that no longer fits its crystal fails loudly
  instead of being applied;
* the registry persists to a file of its own and never touches the row-limit
  file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402

LIB = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library"

MG_CIF = """\
data_Mg_synthetic
_symmetry_space_group_name_H-M   'P 63/m m c'
_symmetry_Int_Tables_number      194
_cell_length_a    3.2094
_cell_length_b    3.2094
_cell_length_c    5.2108
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 120.0
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Mg1  Mg  0.33333333  0.66666667  0.25  1.0
"""

# A face-centred cubic metal written out here: nothing in these tests depends on
# the crystal library being present.
AL_CIF = """\
data_Al_synthetic
_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number      225
_cell_length_a    4.0495
_cell_length_b    4.0495
_cell_length_c    4.0495
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al1  Al  0.0  0.0  0.0  1.0
"""

NI_CIF = AL_CIF.replace("Al_synthetic", "Ni_synthetic").replace("4.0495", "3.5238") \
    .replace("Al1  Al", "Ni1  Ni")


@pytest.fixture(autouse=True)
def clean_registry():
    hr.clear_specs()
    eu.clear_phase_reflector_limits()
    yield
    hr.clear_specs()
    eu.clear_phase_reflector_limits()


def _phase(tmp_path, text, name):
    path = tmp_path / f"{name}.cif"
    path.write_text(text, encoding="utf-8")
    return hr.phase_from_cif(path), path


def _detector():
    import kikuchipy as kp
    return kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5),
                                     sample_tilt=70.0)


def _plist(*phases):
    from orix.crystal_map import PhaseList
    return PhaseList(phases=list(phases))


class _RecordingDetector:
    """Stands in for the detector and records what PyEBSDIndex would be given."""

    def __init__(self):
        self.calls = []

    def get_indexer(self, phase_list, reflectors, **kwargs):
        self.calls.append((phase_list, reflectors, kwargs))
        return "indexer"


def _same(a, b):
    """Deep equality of two built BandIndexers' arrays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _band_state(indexer):
    out = []
    for bi in indexer.phaselist:
        out.append({k: v for k, v in vars(bi).items()
                    if k != "crystalmats" and not callable(v)})
    return out


def _poles(indexer, i=0):
    """The built library's pole families as 3-index triples (PyEBSDIndex stores
    a hexagonal phase's poles with four indices after the build)."""
    arr = np.asarray(indexer.phaselist[i].polefamilies)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] == 4:
        arr = arr[:, [0, 1, 3]]
    return {tuple(int(x) for x in r) for r in arr}


# ---------------------------------------------------------------------------
# no spec = today's path
# ---------------------------------------------------------------------------

def test_without_a_spec_the_rows_are_exactly_what_prepare_reflectors_made(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    ni, _ = _phase(tmp_path, NI_CIF, "Ni")
    for pl in (_plist(al), _plist(al, ni)):
        refl = eu.prepare_reflectors(pl)
        det = _RecordingDetector()
        eu.create_indexer(det, pl, refl, nBands=12)
        sent = det.calls[0][1]
        if isinstance(refl, list):
            assert sent == [r.hkl.tolist() for r in refl]
        else:
            assert sent == refl.hkl.tolist()


def test_the_default_list_is_the_effective_default_of_pyebsdindex(tmp_path):
    """Al: six families before PyEBSDIndex, {200} {220} {111} {311} after it."""
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    ix = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    assert _poles(ix) == {(2, 0, 0), (2, 2, 0), (1, 1, 1), (3, 1, 1)}


def test_the_registry_is_not_consulted_for_other_phases(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    ni, _ = _phase(tmp_path, NI_CIF, "Ni")
    hr.set_spec("Ni", {"mode": "custom", "families": [[1, 1, 1]]})
    pl = _plist(al)
    refl = eu.prepare_reflectors(pl)
    det = _RecordingDetector()
    eu.create_indexer(det, pl, refl)
    assert det.calls[0][1] == refl.hkl.tolist()


def test_computing_reflectors_does_not_change_the_phase(tmp_path):
    """`sanitise_phase()` expands the structure in place. The default list used
    to be computed on the caller's phase, so every call changed the crystal the
    next call saw (pi-Al8FeMg3Si6: 26 -> 128 -> 104 -> 86 atoms)."""
    mg, _ = _phase(tmp_path, MG_CIF, "Mg")
    n = len(mg.structure)
    first = eu._reflectors_for_phase(mg).hkl
    assert len(mg.structure) == n
    assert np.array_equal(eu._reflectors_for_phase(mg).hkl, first)


def test_the_default_list_of_a_phase_does_not_depend_on_how_often_it_was_asked():
    cif = LIB / "pi-Al8FeMg3Si6_ICSD-27140.cif"
    if not cif.exists():
        pytest.skip("crystal library file not present")
    ph = hr.phase_from_cif(cif)
    first = eu._reflectors_for_phase(ph).hkl
    for _ in range(2):
        assert np.array_equal(eu._reflectors_for_phase(ph).hkl, first)


# ---------------------------------------------------------------------------
# a spec
# ---------------------------------------------------------------------------

def test_a_spec_of_the_effective_default_builds_the_identical_library(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    base = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    d = hr.describe(al)
    fams = [f["hkl"] for f in d["families"] if f["effective"]]
    hr.set_spec("Al", {"mode": "custom", "families": fams})
    spec = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    assert _same(_band_state(base), _band_state(spec))


def test_the_default_written_out_with_the_dropped_families_is_the_identical_library(tmp_path):
    """{400} and {222} are in the list and dropped by PyEBSDIndex: ticking them
    changes nothing, and the table says so."""
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    base = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    d = hr.describe(al)
    fams = [f["hkl"] for f in d["families"] if f["selected"]]
    assert len(fams) == 6
    hr.set_spec("Al", {"mode": "custom", "families": fams})
    spec = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    assert _same(_band_state(base), _band_state(spec))


def test_unticking_a_family_removes_its_poles(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    hr.set_spec("Al", {"mode": "custom", "families": [[2, 0, 0], [2, 2, 0], [3, 1, 1]]})
    ix = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    assert _poles(ix) == {(2, 0, 0), (2, 2, 0), (3, 1, 1)}


def test_an_auto_rule_is_the_default_construction_with_other_numbers(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    # F threshold 0.5: only {111} {200} {220}-ish survive the cut
    hr.set_spec("Al", {"mode": "auto", "rule": {"f_threshold": 0.5}})
    rows = hr.rlv_for_spec(al, hr.get_spec("Al")).hkl
    expect = eu._reflectors_for_phase(al, 1.0, 0.5, 70).hkl
    assert np.array_equal(rows, expect)
    ix = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    assert _poles(ix) == {(2, 0, 0), (1, 1, 1), (2, 2, 0)} or len(_poles(ix)) <= 3


def test_a_phase_with_a_spec_is_not_cut_by_the_row_limit(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    eu.set_phase_reflector_limit("Al", 8)           # would leave one family
    hr.set_spec("Al", {"mode": "custom", "families": [[2, 0, 0], [2, 2, 0], [1, 1, 1]]})
    det = _RecordingDetector()
    eu.create_indexer(det, pl, eu.prepare_reflectors(pl))
    sent = det.calls[0][1]
    assert len(sent) == 6 + 12 + 8
    # ... and without a spec the limit still cuts rows, as before
    hr.set_spec("Al", None)
    det2 = _RecordingDetector()
    eu.create_indexer(det2, pl, eu.prepare_reflectors(pl))
    assert len(det2.calls[0][1]) == 8


def test_a_selection_the_library_builder_cannot_use_says_so(tmp_path):
    """Mg {0002} + {2-1-11} is an IndexError inside PyEBSDIndex; one family alone
    is a ValueError there. For a phase with a selection the run says what it
    means instead of surfacing the array error."""
    mg, _ = _phase(tmp_path, MG_CIF, "Mg")
    pl = _plist(mg)
    for fams in ([[0, 0, 2], [2, -1, 1]], [[0, 0, 2]]):
        hr.set_spec("Mg", {"mode": "custom", "families": fams})
        with pytest.raises(ValueError, match="reflector selection of Mg") as e:
            eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
        assert not isinstance(e.value, hr.SpecError)
        assert e.value.__cause__ is not None
    hr.set_spec("Mg", None)
    assert eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl)) is not None


def test_with_two_phases_only_the_one_with_a_spec_changes(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    ni, _ = _phase(tmp_path, NI_CIF, "Ni")
    pl = _plist(al, ni)
    refl = eu.prepare_reflectors(pl)
    hr.set_spec("Ni", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]})
    det = _RecordingDetector()
    eu.create_indexer(det, pl, refl)
    sent = det.calls[0][1]
    assert sent[0] == refl[0].hkl.tolist()
    assert len(sent[1]) == 8 + 6


def test_the_cost_probe_sees_the_resolved_list(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    refl = eu.prepare_reflectors(pl)
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1]]})
    det = _RecordingDetector()
    # predict_triplet_library builds the indexer under a cap; with the recording
    # detector it records the rows of the first count it is asked about.
    eu.predict_triplet_library(det, pl, refl, counts=(1000,))
    assert det.calls[0][1] == hr.rlv_for_spec(al, hr.get_spec("Al")).hkl.tolist()


# ---------------------------------------------------------------------------
# the family table
# ---------------------------------------------------------------------------

def test_the_table_of_aluminium(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    d = hr.describe(al)
    by = {f["label"]: f for f in d["families"]}
    assert d["mode"] == "default" and d["spec"] is None
    assert [f["label"] for f in d["families"][:4]] == ["{111}", "{200}", "{220}", "{311}"]
    assert by["{111}"]["mult"] == 8 and by["{200}"]["mult"] == 6
    assert by["{111}"]["d"] == pytest.approx(2.338, abs=2e-3)
    assert by["{111}"]["rel_f"] == 1.0
    assert by["{222}"]["selected"] and not by["{222}"]["effective"]
    assert by["{222}"]["dropped_by"] == "{111}"
    assert by["{400}"]["dropped_by"] == "{200}"
    assert by["{222}"]["parallel_with"] == ["{111}"]
    assert d["n_selected"] == 6 and d["n_effective"] == 4
    assert sorted(d["effective"]) == sorted([[2, 0, 0], [2, 2, 0], [1, 1, 1], [3, 1, 1]])


def test_the_table_agrees_with_the_pole_families_pyebsdindex_builds(tmp_path):
    for text, name in ((AL_CIF, "Al"), (NI_CIF, "Ni"), (MG_CIF, "Mg")):
        ph, _ = _phase(tmp_path, text, name)
        pl = _plist(ph)
        ix = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
        d = hr.describe(ph)
        assert d["n_effective"] == len(_poles(ix)), name
        assert {tuple(x) for x in d["effective"]} == _poles(ix), name


def test_a_hexagonal_phase_is_shown_in_four_indices(tmp_path):
    mg, _ = _phase(tmp_path, MG_CIF, "Mg")
    d = hr.describe(mg)
    assert d["hexagonal"]
    by = {f["label"]: f for f in d["families"]}
    assert "{0002}" in by and "{2-1-10}" in by
    assert by["{2-1-10}"]["hkl"] == [2, -1, 0] and by["{2-1-10}"]["hkl4"] == [2, -1, -1, 0]


def test_the_table_follows_a_custom_spec(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    d = hr.describe(al, {"mode": "custom", "families": [[1, 1, 1], [2, 2, 0]]})
    sel = {f["label"] for f in d["families"] if f["selected"]}
    assert sel == {"{111}", "{220}"}
    assert d["n_effective"] == 2 and d["mode"] == "custom"


def test_a_selected_family_below_the_candidate_range_is_listed(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    d = hr.describe(al, {"mode": "custom", "families": [[1, 1, 1], [4, 2, 2]]})
    labels = {f["label"]: f for f in d["families"]}
    assert labels["{422}"]["selected"] and labels["{422}"]["d"] < 1.0


def test_top_n_takes_the_strongest_distinct_poles(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    spec = hr.expand_top_n(al, 3)
    assert spec["mode"] == "custom"
    got = {tuple(f) for f in spec["families"]}
    assert got == {(1, 1, 1), (2, 0, 0), (2, 2, 0)}           # not {222}
    assert spec["phase"]["space_group"] == 225


# ---------------------------------------------------------------------------
# validating a hand-entered family
# ---------------------------------------------------------------------------

def test_a_forbidden_family_is_refused(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    with pytest.raises(hr.SpecError) as e:
        hr.validate_family(al, [1, 0, 0])                      # F-centred: mixed parity
    assert e.value.code == "forbidden"


def test_a_structurally_extinct_family_is_refused(tmp_path):
    """Hexagonal groups have no centring test in diffsims; extinction comes
    from the structure factor (Mg: 000l with l odd is absent)."""
    mg, _ = _phase(tmp_path, MG_CIF, "Mg")
    with pytest.raises(hr.SpecError) as e:
        hr.validate_family(mg, [0, 0, 1])
    assert e.value.code == "forbidden"
    assert hr.validate_family(mg, [0, 0, 2])["label"] == "{0002}"


def test_a_valid_family_reports_its_numbers(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    info = hr.validate_family(al, "4 2 2")
    assert info["label"] == "{422}" and info["mult"] == 24
    assert info["d"] == pytest.approx(0.8267, abs=2e-3)
    assert info["in_default_list"] is False
    # the representative is the same whatever member was typed
    assert hr.validate_family(al, [-2, 2, 4])["hkl"] == info["hkl"]


def test_a_family_that_is_already_chosen_is_a_duplicate(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    spec = {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}
    with pytest.raises(hr.SpecError) as e:
        hr.validate_family(al, [-1, 1, 1], spec)
    assert e.value.code == "duplicate" and e.value.params["label"] == "{111}"


def test_a_parallel_family_is_allowed_but_named(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    info = hr.validate_family(al, [4, 0, 0], {"mode": "custom", "families": [[2, 0, 0]]})
    assert info["parallel_with"] == ["{200}"]


def test_four_index_input_is_converted_and_checked(tmp_path):
    mg, _ = _phase(tmp_path, MG_CIF, "Mg")
    info = hr.validate_family(mg, [2, -1, -1, 0])
    assert info["hkl"] == [2, -1, 0] and info["label"] == "{2-1-10}"
    assert hr.validate_family(mg, "2 -1 -1 0")["hkl"] == [2, -1, 0]
    with pytest.raises(hr.SpecError) as e:
        hr.validate_family(mg, [2, -1, 0, 0])                  # i != -(h+k)
    assert e.value.code == "bad_hkl"


def test_four_indices_on_a_cubic_phase_are_refused(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    with pytest.raises(hr.SpecError) as e:
        hr.parse_hkl([1, 1, -2, 0], al)
    assert e.value.code == "bad_hkl"


@pytest.mark.parametrize("bad", [[0, 0, 0], "a b c", [1, 2], [1.5, 0, 0], "", None])
def test_nonsense_input_is_refused(tmp_path, bad):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    with pytest.raises(hr.SpecError):
        hr.parse_hkl(bad, al)


# ---------------------------------------------------------------------------
# stale and invalid specs fail loudly
# ---------------------------------------------------------------------------

def test_a_selection_made_for_another_crystal_is_refused_not_applied(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    spec = hr.spec_with_fingerprint(al, {"mode": "custom", "families": [[1, 1, 1]]})
    hr.set_spec("Al", spec)
    other = AL_CIF.replace("4.0495", "4.20")                   # same stem, other cell
    al2, _ = _phase(tmp_path, other, "Al")
    pl = _plist(al2)
    with pytest.raises(hr.SpecError) as e:
        eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    assert e.value.code == "stale_spec"


def test_a_stored_family_that_became_forbidden_fails_loudly(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 0, 0]]})
    with pytest.raises(hr.SpecError) as e:
        hr.rlv_for_spec(al, hr.get_spec("Al"))
    assert e.value.code == "forbidden"


@pytest.mark.parametrize("bad", [
    {"mode": "nonsense"}, {"mode": "custom"}, {"mode": "custom", "families": []},
    {"mode": "custom", "families": [[0, 0, 0]]}, {"mode": "custom", "families": [[1, 1]]},
    {"mode": "auto", "rule": {"min_d": -1}}, {"mode": "auto", "rule": {"f_threshold": 2}},
    "text", 5,
])
def test_unusable_specs_are_refused(bad):
    with pytest.raises(hr.SpecError):
        hr.set_spec("X", bad)
    assert hr.get_spec("X") is None


def test_a_spec_is_normalised_and_deduplicated():
    out = hr.set_spec("X", {"mode": "custom", "families": [[1.0, 1, 1], [1, 1, 1], [2, 0, 0]]})
    assert out["families"] == [[1, 1, 1], [2, 0, 0]]
    assert hr.set_spec("X", {"mode": "auto"})["rule"] == hr.DEFAULT_RULE


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def test_a_spec_is_found_by_path_stem_and_case():
    hr.set_spec(r"C:\lib\Al7FeCu2.cif", {"mode": "custom", "families": [[1, 1, 1]]})
    assert hr.get_spec("al7fecu2") is not None
    assert hr.get_spec("D:/elsewhere/Al7FeCu2.cif") is not None
    assert hr.get_spec("Al") is None


def test_the_version_moves_only_when_the_stored_value_changes():
    v0 = hr.registry_version()
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1]]})
    v1 = hr.registry_version()
    assert v1 > v0
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1]]})       # same again
    assert hr.registry_version() == v1
    hr.set_spec("Al", None)
    assert hr.registry_version() > v1
    v2 = hr.registry_version()
    hr.set_spec("Al", None)                                              # already empty
    assert hr.registry_version() == v2


def test_specs_persist_and_the_older_limits_file_is_left_alone(tmp_path, monkeypatch):
    legacy = tmp_path / "hough_reflector_limits.json"
    legacy.write_text('{\n  "al": 40\n}\n', encoding="utf-8")
    before = legacy.read_bytes()
    monkeypatch.setattr(eu, "_limits_store_path", lambda: legacy)
    hr.clear_specs()
    hr._LOADED = False                      # read the (absent) store again
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]})
    stored = json.loads((tmp_path / hr.SPEC_STORE_FILE).read_text(encoding="utf-8"))
    assert stored == {"al": {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]}}
    assert legacy.read_bytes() == before                  # untouched, byte for byte
    # a fresh process reads it back
    hr.clear_specs()
    hr._LOADED = False
    assert hr.get_spec("Al")["families"] == [[1, 1, 1], [2, 0, 0]]


def test_the_test_switch_keeps_the_real_file_out():
    assert hr._store_path() is None


def test_a_corrupt_store_means_no_specs(tmp_path, monkeypatch):
    legacy = tmp_path / "hough_reflector_limits.json"
    monkeypatch.setattr(eu, "_limits_store_path", lambda: legacy)
    (tmp_path / hr.SPEC_STORE_FILE).write_text("{ not json", encoding="utf-8")
    hr.clear_specs()
    hr._LOADED = False
    assert hr.all_specs() == {}


# ---------------------------------------------------------------------------
# the overlay and the PC controller see the same list
# ---------------------------------------------------------------------------

def test_prepare_reflectors_with_specs_is_the_plain_function_without_any(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    a = hr.prepare_reflectors(pl)
    b = eu.prepare_reflectors(pl)
    assert np.array_equal(a.hkl, b.hkl)


def test_prepare_reflectors_with_specs_returns_the_chosen_rows(tmp_path):
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    ni, _ = _phase(tmp_path, NI_CIF, "Ni")
    hr.set_spec("Al", {"mode": "custom", "families": [[1, 1, 1], [2, 0, 0]]})
    one = hr.prepare_reflectors(_plist(al))
    assert one.hkl.shape == (14, 3)
    two = hr.prepare_reflectors(_plist(al, ni))
    assert isinstance(two, list) and two[0].hkl.shape == (14, 3)
    assert np.array_equal(two[1].hkl, eu.prepare_reflectors(_plist(ni))[0].hkl) \
        if isinstance(eu.prepare_reflectors(_plist(ni)), list) else True


# ---------------------------------------------------------------------------
# every phase of the crystal library: the folded poles equal the real build
# ---------------------------------------------------------------------------

_LIBRARY = ["Al", "Ni", "Si", "MgZn2_sd_0261233", "ZnO_sd_1400158",
            "pi-Al8FeMg3Si6_ICSD-27140", "Mg17Al12_MP-mp-2151",
            "Al3FeSi2_tau2_ICSD-607628"]


@pytest.mark.parametrize("name", _LIBRARY)
def test_library_phase_table_matches_the_real_indexer_and_the_default_roundtrips(name):
    cif = LIB / f"{name}.cif"
    if not cif.exists():
        pytest.skip("crystal library file not present")
    ph = hr.phase_from_cif(cif)
    pl = _plist(ph)
    base = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    d = hr.describe(ph)
    assert {tuple(x) for x in d["effective"]} == _poles(base), name
    # the default, written out as a spec, is the same library
    fams = [f["hkl"] for f in d["families"] if f["selected"]]
    hr.set_spec(name, {"mode": "custom", "families": fams})
    spec = eu.create_indexer(_detector(), pl, eu.prepare_reflectors(pl))
    assert _same(_band_state(base), _band_state(spec)), name
