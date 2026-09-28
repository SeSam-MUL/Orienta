"""The phase library's index endpoint: one answer, and it must not contradict itself.

Spec §3.1. The frontend's frozen fixture
(`frontend/src/components/PhaseLibrary/__fixtures__/library.json`, built by
`scripts/build_phase_library_fixture.py`) is the agreed contract. That file lives
on the frontend branch, so these tests do not depend on it being present: the
field set is asserted explicitly here, and the fixture is compared against only
when it exists. A test that silently skips is a test that proves nothing, so the
field-set assertion runs either way.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.data_deps import CIF_LIBRARY, requires

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (PROJECT_ROOT / "frontend" / "src" / "components" / "PhaseLibrary"
           / "__fixtures__" / "library.json")

#: Every field a phase record carries. Written out rather than derived, because
#: this IS the contract: if the endpoint stops sending one, the page loses a
#: column silently, and if it starts sending a new one nobody agreed to it.
EXPECTED_FIELDS = {
    "key", "formula_label", "formula_structure", "prototype", "pearson",
    "space_group_hm", "space_group_it", "crystal_system", "n_atoms",
    "elements_structure", "elements_label", "elements_label_source",
    "elements_disagree", "cell", "cell_setting", "cell_settings_available",
    "cell_setting_conflict", "reference", "reference_rejected", "doi", "url",
    "icsd", "cod", "ids_from_filename", "folder", "files", "capabilities",
    "parse_error", "display_name", "search_terms",
    # from the synonym store (§2.4); the author travels with the name because
    # in a shared library a name without one is a silent edit to someone
    # else's work.
    "name_author", "name_updated",
}

#: The cases the spec's approval requires to be answerable. Named, not counted:
#: a count is accidentally satisfiable (`s phase -> 1` was, by the wrong entry).
REQUIRED_KEYS = {
    "sd_0302719": "canonical duplicate, half 1",
    "sd_1401510": "canonical duplicate, half 2",
    "alpha-AlFeMnSi_ICSD-52623": "alpha pair, cP138",
    "α-(AlMnSi)": "alpha pair, cP138",
    "Al": "two cells in one file",
    "sd_1816951": "AmbiguousCifError",
    "beta-AlFeSi": "label says Si, structure has none",
}


@pytest.fixture(scope="module")
def doc():
    from backend.api.services.phase_library import build_index_document
    return build_index_document()


@pytest.fixture(scope="module")
def by_key(doc):
    return {p["key"]: p for p in doc["phases"]}


# --- shape ----------------------------------------------------------------

@requires(CIF_LIBRARY)
def test_one_answer_carries_both_the_phases_and_the_library_totals(doc):
    """Two requests would let the counts arrive before the list (spec §2.8)."""
    assert set(doc) >= {"schema", "phases", "library_totals"}
    assert doc["phases"], "an empty library here would be a failure, not an answer"
    assert set(doc["library_totals"]) == {"elements", "crystal_system",
                                         "capabilities", "flags"}


@requires(CIF_LIBRARY)
def test_every_record_has_exactly_the_agreed_fields(doc):
    for p in doc["phases"]:
        assert set(p) == EXPECTED_FIELDS, (
            f"{p['key']}: extra={sorted(set(p) - EXPECTED_FIELDS)} "
            f"missing={sorted(EXPECTED_FIELDS - set(p))}")


@requires(CIF_LIBRARY)
def test_the_cases_the_approval_named_are_all_answerable(by_key):
    missing = [k for k in REQUIRED_KEYS if k not in by_key]
    assert missing == [], f"required cases absent: {missing}"


@requires(CIF_LIBRARY)
def test_every_cell_carries_all_six_parameters(by_key):
    """`a b c` without gamma = 120 deg is three numbers, not a cell.

    12 of the 36 phases have an angle off 90 deg, so this is not decoration.
    """
    for key, p in by_key.items():
        assert set(p["cell"]) == {"a", "b", "c", "alpha", "beta", "gamma"}, key
        if p["cell"]["a"] is not None:
            assert p["cell"]["alpha"] is not None, f"{key} has a but no angles"
    assert by_key["Al13Fe4"]["cell"]["beta"] == pytest.approx(107.669, abs=1e-2)
    assert by_key["Al4Fe1.7Si (τ11)"]["cell"]["gamma"] == pytest.approx(120, abs=1e-6)


# --- the property that makes a total trustworthy --------------------------

@requires(CIF_LIBRARY)
def test_every_total_is_counted_from_the_returned_phases(doc):
    """A total is counted FROM the returned list, so it cannot disagree with it.

    These are LIBRARY TOTALS, not the sidebar's counters -- those narrow with
    the active filter and live in the page. The property still matters for what
    these ARE used for: the header "N of 36" and the load state.

    This is the class of bug the rewrite exists to remove: the search said `Cu`
    matched 6 phases while the facet said 4. Any facet computed from a second
    source can drift; one counted from the payload cannot.
    """
    phases, f = doc["phases"], doc["library_totals"]
    for el, n in f["elements"].items():
        assert n == sum(1 for p in phases if el in p["elements_structure"]), el
    for sys_, n in f["crystal_system"].items():
        assert n == sum(1 for p in phases if p["crystal_system"] == sys_), sys_
    for cap, n in f["capabilities"].items():
        assert n == sum(1 for p in phases if p["capabilities"][cap]), cap
    assert f["flags"]["parse_error"] == sum(1 for p in phases if p["parse_error"])
    assert f["flags"]["elements_disagree"] == sum(
        1 for p in phases if p["elements_disagree"])


@requires(CIF_LIBRARY)
def test_the_bands_are_the_measured_nine_with_no_phantoms(doc):
    """93 placements over 9 bands, and no phase without a band.

    The phantom bands (`Al0+` and friends) are what the element-symbol fix
    removed; this is the assertion that they stay gone.
    """
    bands = doc["library_totals"]["elements"]
    assert not [b for b in bands if "+" in b or "-" in b], bands
    assert len(bands) == 9
    assert sum(bands.values()) == 93
    assert bands["Al"] == 28
    assert [p["key"] for p in doc["phases"] if not p["elements_structure"]] == []


# --- capabilities ---------------------------------------------------------

@requires(CIF_LIBRARY)
def test_hough_is_false_for_a_cif_whose_structure_cannot_be_read(by_key):
    """A green tick the backend then refuses is worse than an honest cross.

    `resolve(method="hough")` refuses `sd_1816951`, because the reader Hough
    actually uses (orix/diffpy, not pymatgen) SUCCEEDS on that file and returns
    Mg 80 / Cu 20 at% where the .xtal says Cu 66.7 / Mg 33.3. The capability has
    to agree with the refusal, or the card promises a run that cannot happen.
    """
    p = by_key["sd_1816951"]
    assert p["parse_error"]
    assert p["capabilities"]["hough"] is False
    assert p["capabilities"]["spherical"] is True, (
        "the .sht does not come from that CIF, so Spherical is unaffected")


@requires(CIF_LIBRARY)
def test_the_capability_agrees_with_what_the_collection_resolver_hands_out(by_key):
    """The two must never disagree: same question, two places."""
    from backend.api.services import crystal_hint_local_library as chl
    idx = chl.get_index()
    for key, p in by_key.items():
        e = idx[key]
        refused = bool(e.parse_error)
        assert p["capabilities"]["hough"] == (e.cif_path is not None and not refused), key


@requires(CIF_LIBRARY)
def test_dictionary_capability_counts_masters_by_content(doc, by_key):
    """16, and specifically the one the name rule loses.

    `pi-Al8FeMg3Si6_ICSD-27140` has a real master called
    `..._E20kV_sig70_n501_o0.h5` -- a name indistinguishable from the
    Monte-Carlo outputs beside it. A `*master*.h5` glob answers 15.
    """
    assert doc["library_totals"]["capabilities"]["dictionary"] == 16
    p = by_key["pi-Al8FeMg3Si6_ICSD-27140"]
    assert p["capabilities"]["dictionary"] is True
    assert "master" not in p["files"]["master"]["name"].lower()


@requires(CIF_LIBRARY)
def test_short_keys_find_their_own_master(by_key):
    """`Al` and `Ni` are two characters.

    The shipped matcher rejects tokens under four characters so they cannot
    over-match, and consequently answers "no master" for the two phases whose
    masters are the easiest to find.
    """
    for key in ("Al", "Ni"):
        assert by_key[key]["files"]["master"], f"{key} lost its own master"
        assert by_key[key]["capabilities"]["dictionary"] is True


# --- cell setting ---------------------------------------------------------

@requires(CIF_LIBRARY)
def test_the_named_setting_is_the_standardized_cell(by_key):
    """Al holds 4.049 AND 2.8631; quoting the primitive one is 29 % too small."""
    al = by_key["Al"]
    assert al["cell_setting"] == "standardized_unitcell"
    assert al["cell"]["a"] == pytest.approx(4.049, abs=1e-3)
    assert set(al["cell_settings_available"]) >= {
        "standardized_unitcell", "published_cell", "niggli_reduced_cell"}
    assert al["cell_setting_conflict"] is None


def test_a_setting_that_disagrees_with_the_cell_is_not_claimed():
    """The label describes numbers it did not produce, so it is checked.

    Today pymatgen reports the standardized block for all 9 files that state
    different values -- 0 exceptions. That is a property of the files, not of
    the code, so the endpoint verifies rather than assumes: on a mismatch the
    setting is null with a reason, because a cell labelled "standardized" that
    is actually primitive is off by sqrt(2) AND wears a tag contradicting it.
    """
    from backend.api.services.phase_library import resolve_cell

    class _E:
        key = "synthetic"
        lattice_a_A, lattice_b_A, lattice_c_A = 2.8631, 2.8631, 2.8631
        lattice_alpha_deg = lattice_beta_deg = lattice_gamma_deg = 90.0

    text = ("data_sm_isp_X-standardized_unitcell\n"
            "_cell_length_a 4.049\n_cell_angle_alpha 90\n")
    cell, setting, available, conflict = resolve_cell(_E(), text)
    assert cell["a"] == pytest.approx(2.8631)     # the numbers are untouched
    assert setting is None, "a mismatched setting must not be claimed"
    assert conflict == "setting_mismatch"
    assert available == ["standardized_unitcell"]


def test_a_matching_setting_is_reported():
    from backend.api.services.phase_library import resolve_cell

    class _E:
        key = "synthetic"
        lattice_a_A, lattice_b_A, lattice_c_A = 4.049, 4.049, 4.049
        lattice_alpha_deg = lattice_beta_deg = lattice_gamma_deg = 90.0

    text = ("data_sm_isp_X-standardized_unitcell\n_cell_length_a 4.049\n"
            "\ndata_sm_isp_X-niggli_reduced_cell\n_cell_length_a 2.8631\n")
    _, setting, available, conflict = resolve_cell(_E(), text)
    assert setting == "standardized_unitcell" and conflict is None
    assert available == ["niggli_reduced_cell", "standardized_unitcell"]


# --- citations ------------------------------------------------------------

@pytest.mark.parametrize("body,want_ref,want_reason", [
    ("Tonejc A., Rocak D.: Mechanical and structural properties of Al-Ni "
     "alloys. Acta Metallurgica 19 (1971) 311-316.", True, None),
    ("_publ_author_address", False, "cif_tag"),
    ("https", False, "bare_url"),
    ("https://example.org/a/b", False, "bare_url"),
    ("short", False, "too_short"),
])
def test_a_source_that_is_plainly_not_a_citation_is_rejected_with_a_reason(
        body, want_ref, want_reason):
    """Rather no citation than one that is visibly not a citation.

    But `null` alone cannot tell "this CIF names no source" from "the source was
    unusable", and those are two different sentences on the card -- so the
    reason travels with it.
    """
    from backend.api.services.phase_library import read_reference
    ref, reason = read_reference(
        f"_publ_section_references\n;\n{body}\n;\n")
    assert bool(ref) is want_ref
    assert reason == want_reason


def test_a_citation_spanning_several_lines_is_not_severed():
    """The semicolon block is the field; its first line is not.

    Reading only the first line produced the doubled sentence with a severed
    URL that a user test saw on screen.
    """
    from backend.api.services.phase_library import read_reference
    ref, reason = read_reference(
        "_publ_section_references\n;\n"
        "Stadnik Z. M., Purdie D.: <i>Icosahedral</i> quasicrystal\n"
        "1/1 approximant &#x03b1;-AlCuFeSi. Phys. Rev. B 64 (2001) 214202.\n;\n")
    assert reason is None
    assert "quasicrystal 1/1 approximant" in ref, ref
    assert "α-AlCuFeSi" in ref, "the &#x03b1; entity must be decoded"
    assert "<i>" not in ref


def test_a_cif_without_a_citation_is_silent_not_rejected():
    from backend.api.services.phase_library import read_reference
    assert read_reference("data_x\n_cell_length_a 4.05\n") == (None, None)


# --- the 44 MB array is never read ----------------------------------------

def test_the_master_check_is_a_key_lookup_not_a_read():
    """`mLPNH` is 44 MB of a 210 MB file and the endpoint must never load it.

    Asserted on the source, because the cheap correct call and the ruinous one
    differ by two characters (`in f` vs `f[...]`) and both pass a behavioural
    test.
    """
    import inspect

    from backend.api.services import phase_library
    src = inspect.getsource(phase_library._has_master_array)
    assert "'EMData/EBSDmaster/mLPNH' in f" in src or \
           '"EMData/EBSDmaster/mLPNH" in f' in src
    body = src.split('"""', 2)[-1]
    for forbidden in ("[()]", "np.array(", ".value", "[:]"):
        assert forbidden not in body, f"{forbidden} would read the array"


def test_a_master_is_recognised_by_content_and_a_monte_carlo_file_is_not(tmp_path):
    import h5py

    from backend.api.services.phase_library import _has_master_array, master_files

    good = tmp_path / "phase" / "x_E20kV_sig70_n501_o0.h5"
    good.parent.mkdir()
    with h5py.File(good, "w") as f:
        f.create_dataset("EMData/EBSDmaster/mLPNH", data=[[1.0]])
    mc = tmp_path / "phase2" / "y_master_E20kV_npx500.h5"
    mc.parent.mkdir()
    with h5py.File(mc, "w") as f:
        f.create_dataset("EMData/MCOpenCL/accum_e", data=[[1]])

    assert _has_master_array(good) is True, "named like an MC run, but IS a master"
    assert _has_master_array(mc) is False, "named 'master', but is not one"
    found = master_files(tmp_path)
    assert [q.name for q in found] == [good.name]


# --- against the frozen contract, when it is here -------------------------

def test_it_matches_the_frontend_fixture_field_for_field(by_key):
    """The agreed contract, compared on the real library.

    Skips only when the fixture is not in this checkout (it is built on the
    frontend branch). `test_every_record_has_exactly_the_agreed_fields` covers
    the shape unconditionally, so a skip here never leaves the contract
    unchecked.
    """
    if not FIXTURE.is_file():
        pytest.skip("frontend fixture not in this checkout")
    fx = {p["key"]: p for p in json.loads(
        FIXTURE.read_text(encoding="utf-8"))["phases"]}
    assert set(fx) == set(by_key), (
        f"key sets differ: only-fixture={sorted(set(fx) - set(by_key))} "
        f"only-endpoint={sorted(set(by_key) - set(fx))}")

    # Fields the endpoint adds on purpose, and the one it answers differently.
    ADDED = {"cell_setting_conflict", "reference_rejected"}
    DIFFERS = {"capabilities"}
    # The four fields a PERSON writes. They come from the synonym store in the
    # library folder, so they differ per machine and change on every rename;
    # comparing them by value would make `git` see a test as owning somebody's
    # working data, and would turn a rename into a red test. Their TYPE is
    # still compared below, because the list has to render both states.
    USER_EDITABLE = {"display_name", "search_terms", "name_author",
                     "name_updated"}
    diffs = []
    for key, want in fx.items():
        got = by_key[key]
        for field in set(want) - DIFFERS:
            if field in ADDED:
                continue
            if field in USER_EDITABLE:
                # Present, and of the right shape -- a string or nothing, a
                # list of strings. A row that dropped the field, or answered
                # a bare string where the frontend maps a list, still fails.
                assert field in got, f"{key}: endpoint dropped {field}"
                if field == "search_terms":
                    assert isinstance(got[field], list) and all(
                        isinstance(t, str) for t in got[field]), (
                        f"{key}.search_terms is {got[field]!r}, not a list of "
                        f"strings")
                else:
                    assert got[field] is None or isinstance(got[field], str), (
                        f"{key}.{field} is {got[field]!r}, not a string or null")
                continue
            if got.get(field) != want[field]:
                diffs.append((key, field, want[field], got.get(field)))
    assert diffs == [], "\n".join(
        f"{k}.{f}: fixture={w!r} endpoint={g!r}" for k, f, w, g in diffs[:12])


# ==========================================================================
# Added after the review of ad748968. Eight mutations passed that suite: the
# key set was asserted and no VALUE was, so six of the card's columns could go
# blank while the tests stayed green. Each test below fails for exactly one.
# ==========================================================================

@requires(CIF_LIBRARY)
def test_the_values_the_card_shows_are_asserted_not_just_their_keys(by_key):
    """Concrete values for one well-known phase.

    `test_every_record_has_exactly_the_agreed_fields` checks the key set, which
    a mutation returning null for `pearson`, `prototype`, `n_atoms`,
    `space_group_hm/it`, `folder` and every `files.*.bytes` passes untouched.
    """
    al = by_key["Al"]
    assert al["formula_structure"] == "Al"
    assert al["prototype"] == "Cu"
    assert al["pearson"] == "cF4"
    assert al["space_group_hm"] == "Fm-3m"
    assert al["space_group_it"] == 225
    assert al["crystal_system"] == "cubic"
    assert al["n_atoms"] == 4
    assert al["elements_structure"] == ["Al"]
    assert al["folder"] == "Al"
    assert al["files"]["cif"]["name"] == "Al.cif"
    assert al["files"]["cif"]["bytes"] > 0
    assert al["files"]["cif"]["date"] != "1970-01-01"
    assert al["files"]["cif"]["rel"].startswith("Database/")
    assert al["files"]["master"]["name"] == "Al_master_E20kV_npx500.h5"


@requires(CIF_LIBRARY)
def test_the_label_provenance_is_reported_per_phase(by_key):
    """`elements_label_source` exists so the card can say WHICH claim it shows.

    Nothing asserted it, so a mutation reporting `sm_phase_labels` for all 36
    passed -- the card would word an invented element as a file's statement.
    """
    assert by_key["sd_0302719"]["elements_label_source"] == "sm_phase_labels"
    assert by_key["Al3FeSi2_tau2_ICSD-607628"]["elements_label_source"] == "sht_formula"
    assert by_key["beta-AlFeSi"]["elements_label_source"] == "filename"


@requires(CIF_LIBRARY)
def test_the_two_real_badges_are_there_and_the_phantom_one_is_gone(doc, by_key):
    """`beta-AlFeSi` is in REQUIRED_KEYS for this property; assert the property.

    Only its presence was asserted, so `elements_in_label` returning [] for
    everything passed: every badge went false, the flag to 0, and the facet test
    still agreed (0 == 0).
    """
    assert by_key["beta-AlFeSi"]["elements_disagree"] is True
    assert by_key["beta-AlFeSi"]["elements_label"] == ["Al", "Fe", "Si"]
    assert by_key["beta-AlFeSi"]["elements_structure"] == ["Al", "Fe"]
    assert by_key["sd_1401510"]["elements_disagree"] is True
    z = by_key["Al2Zn_MP-mp-aaacqrcb"]
    assert z["elements_label"] == ["Al", "Zn"], "MP is Materials Project"
    assert z["elements_disagree"] is False
    assert doc["library_totals"]["flags"]["elements_disagree"] == 2


@requires(CIF_LIBRARY)
def test_no_element_set_contains_a_non_element(doc):
    """The regex alone returns `M`, which is not an element."""
    from phase_metadata import KNOWN_ELEMENTS
    for p in doc["phases"]:
        for field in ("elements_structure", "elements_label"):
            bad = [e for e in p[field] if e not in KNOWN_ELEMENTS]
            assert bad == [], f"{p['key']}.{field} contains {bad}"


@requires(CIF_LIBRARY)
def test_the_ids_read_from_the_filename_are_real_ids(doc, by_key):
    """`r"mp-([0-9a-z]+)"` with re.I captured the `mp` of `MP-mp-2199`.

    Four phases reported {"mp": "mp"} as a Materials Project identifier, and the
    field had no coverage, so returning None always passed.
    """
    assert by_key["Fe3Si_MP-mp-2199"]["ids_from_filename"] == {"mp": "2199"}
    assert by_key["alpha-AlFeMnSi_ICSD-52623"]["ids_from_filename"] == {"icsd": "52623"}
    assert by_key["beta-AlFeSi_withSi_COD-2107329"]["ids_from_filename"] == {"cod": "2107329"}
    assert by_key["Al"]["ids_from_filename"] is None
    for p in doc["phases"]:
        for kind, val in (p["ids_from_filename"] or {}).items():
            assert val and val.lower() != kind, f"{p['key']}: {kind}={val!r}"


@requires(CIF_LIBRARY)
def test_each_flag_counts_its_own_predicate(doc):
    """Three of the five flags were unchecked, so one could return another."""
    phases, f = doc["phases"], doc["library_totals"]["flags"]
    assert f["no_reference"] == sum(1 for p in phases if not p["reference"])
    assert f["reference_rejected"] == sum(
        1 for p in phases if p["reference_rejected"])
    assert f["several_cell_settings"] == sum(
        1 for p in phases if len(p["cell_settings_available"]) > 1)
    assert f["no_reference"] != f["reference_rejected"], (
        "on this library these differ, so swapping them must fail")


def test_the_library_totals_count_the_list_they_are_given_not_the_library():
    """The property the earlier test could not see.

    It compared the facets against the returned phases -- two sources that agree
    today -- so a `_library_totals` counting `get_index()` instead passed. Here the list
    is hand-built and unlike the library, so only counting THIS list can match.
    """
    from backend.api.services.phase_library import _library_totals
    phases = [
        {"key": "a", "elements_structure": ["Xe"], "crystal_system": "cubic",
         "capabilities": {"hough": True, "spherical": False, "dictionary": False},
         "elements_disagree": False, "parse_error": None, "reference": "x",
         "reference_rejected": None, "cell_settings_available": []},
        {"key": "b", "elements_structure": ["Xe", "Kr"], "crystal_system": "cubic",
         "capabilities": {"hough": False, "spherical": True, "dictionary": False},
         "elements_disagree": True, "parse_error": "no", "reference": None,
         "reference_rejected": "cif_tag", "cell_settings_available": ["a", "b"]},
    ]
    f = _library_totals(phases)
    assert f["elements"] == {"Xe": 2, "Kr": 1}
    assert f["crystal_system"] == {"cubic": 2}
    assert f["capabilities"] == {"hough": 1, "spherical": 1, "dictionary": 0}
    assert f["flags"] == {"elements_disagree": 1, "parse_error": 1,
                          "no_reference": 1, "reference_rejected": 1,
                          "several_cell_settings": 1}


# --- the cell's numbers come from the index, not from the block -----------

def test_the_cell_numbers_come_from_the_index_not_the_named_block():
    """The central invariant, unpinned on the real library.

    The two agree on all 15 files that declare a setting, so returning the
    BLOCK's numbers passed everything. Here they differ by less than the
    tolerance, so only the index's values can be the answer.
    """
    from backend.api.services.phase_library import resolve_cell

    class _E:
        key = "synthetic"
        lattice_a_A = lattice_b_A = lattice_c_A = 4.0495
        lattice_alpha_deg = lattice_beta_deg = lattice_gamma_deg = 90.0

    text = ("data_x-standardized_unitcell\n_cell_length_a 4.0490\n"
            "_cell_length_b 4.0490\n_cell_length_c 4.0490\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n")
    cell, setting, _, conflict = resolve_cell(_E(), text)
    assert cell["a"] == pytest.approx(4.0495), "the index's number must win"
    assert setting == "standardized_unitcell" and conflict is None


def test_the_guard_checks_every_parameter_not_only_a():
    """`a` is the axis a primitive/conventional pair can SHARE.

    Demonstrated on the first version: an entry carrying the Niggli cell's b, c
    and alpha was labelled `standardized_unitcell` with no conflict -- exactly
    the mislabelling the guard exists to prevent.
    """
    from backend.api.services.phase_library import resolve_cell

    class _E:
        key = "synthetic"
        lattice_a_A, lattice_b_A, lattice_c_A = 5.0, 3.536, 3.536
        lattice_alpha_deg, lattice_beta_deg, lattice_gamma_deg = 60.0, 90.0, 90.0

    text = ("data_x-standardized_unitcell\n_cell_length_a 5.000\n"
            "_cell_length_b 5.000\n_cell_length_c 5.000\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n")
    _, setting, _, conflict = resolve_cell(_E(), text)
    assert setting is None, "b, c and alpha are the other block's"
    assert conflict == "setting_mismatch"


def test_a_large_cell_is_not_punished_by_an_absolute_tolerance():
    """0.01 A is 0.04 % of a 26 A cell and 0.35 % of a 2.8 A one."""
    from backend.api.services.phase_library import resolve_cell

    class _E:
        key = "synthetic"
        lattice_a_A = lattice_b_A = 12.404
        lattice_c_A = 26.2455
        lattice_alpha_deg = lattice_beta_deg = 90.0
        lattice_gamma_deg = 120.0

    text = ("data_x-standardized_unitcell\n_cell_length_a 12.404\n"
            "_cell_length_b 12.404\n_cell_length_c 26.234\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 120\n")
    _, setting, _, conflict = resolve_cell(_E(), text)
    assert setting == "standardized_unitcell", conflict


def test_no_cell_and_an_unreadable_block_are_told_apart():
    """One reason string covered three situations; the card cannot word that."""
    from backend.api.services.phase_library import resolve_cell

    class _NoCell:
        key = "a"
        lattice_a_A = lattice_b_A = lattice_c_A = None
        lattice_alpha_deg = lattice_beta_deg = lattice_gamma_deg = None

    class _HasCell:
        key = "b"
        lattice_a_A = lattice_b_A = lattice_c_A = 4.049
        lattice_alpha_deg = lattice_beta_deg = lattice_gamma_deg = 90.0

    assert resolve_cell(
        _NoCell(), "data_x-standardized_unitcell\n_cell_length_a 4.049\n"
    )[3] == "cell_missing"
    assert resolve_cell(
        _HasCell(), "data_x-standardized_unitcell\n_cell_length_a ?\n"
    )[3] == "block_unreadable"


# --- masters: one algorithm, ranked against the library -------------------

@requires(CIF_LIBRARY)
def test_a_shorter_key_does_not_claim_a_longer_keys_master(by_key):
    """`beta-AlFeSi` is a prefix of `beta-AlFeSi_withSi_COD-2107329`.

    Matching each key independently gave the shorter phase the longer one's
    master -- two DIFFERENT phases, one without a Si site -- and would have
    offered Dictionary indexing against the wrong structure. It moved the agreed
    capability count from 16 to 17, which is how it was caught.
    """
    assert by_key["beta-AlFeSi"]["files"]["master"] is None
    assert by_key["beta-AlFeSi"]["capabilities"]["dictionary"] is False
    other = by_key["beta-AlFeSi_withSi_COD-2107329"]["files"]["master"]
    assert other and other["name"].startswith("beta-AlFeSi_withSi_COD-2107329")


@requires(CIF_LIBRARY)
def test_the_dictionary_capability_agrees_with_the_resolver_by_calling_it(by_key):
    """CALLS the resolver instead of re-deriving its expression.

    The first version re-implemented `cif_path is not None and not parse_error`
    character for character -- a tautology that could not fail. Meanwhile the
    endpoint said Dictionary for 16 phases and resolve() refused 14.
    """
    from backend.api.routes.indexing import _master_h5_for_phase
    from backend.api.services import crystal_hint_local_library as chl
    idx = chl.get_index()
    for key, p in by_key.items():
        got = _master_h5_for_phase(idx[key]) is not None
        assert got == p["capabilities"]["dictionary"], (
            f"{key}: resolver={got} card={p['capabilities']['dictionary']}")


@requires(CIF_LIBRARY)
def test_the_hough_capability_agrees_with_the_resolver_by_calling_it():
    """Same for Hough: the route, not a copy of its condition."""
    import tempfile

    from backend.api.routes.phase_collections import resolve
    from backend.api.services import phase_collections as pc
    from backend.api.services.phase_library import build_index_document

    by = {p["key"]: p for p in build_index_document()["phases"]}
    with tempfile.TemporaryDirectory() as tmp:
        pc.set_collections_dir_for_test(Path(tmp))
        try:
            for key in ("sd_1816951", "Al"):
                name = f"guard-{key}"
                pc.save(pc.PhaseCollection(
                    name=name, members=[pc.PhaseMember(key=key)]))
                handed = bool(resolve(name=name, method="hough")["paths"])
                assert handed == by[key]["capabilities"]["hough"], (
                    f"{key}: resolver handed={handed} "
                    f"card={by[key]['capabilities']['hough']}")
        finally:
            pc.set_collections_dir_for_test(None)


@requires(CIF_LIBRARY)
def test_a_dictionary_in_a_folder_named_otherwise_is_still_found(by_key):
    """`Dictionary_Library/sd/` holds `sd_0302719`'s file.

    Keyed on the folder, the phase that HAS one was told it had none.
    """
    d = by_key["sd_0302719"]["files"]["dictionary"]
    assert d and d["name"].startswith("sd_0302719")


@requires(CIF_LIBRARY)
def test_masters_with_no_library_phase_are_reported_not_hidden(doc):
    """One of the four is the S-phase master."""
    names = sorted(u["name"] for u in doc["unassigned_masters"])
    assert names == sorted([
        "deposit_Al2CuMg_master_E20kV_npx500.h5",
        "deposit_Al7Cu2Fe_master_E20kV_npx500.h5",
        "deposit_Al_master_E20kV_npx500.h5",
        "deposit_alpha-AlMnSi_master_E20kV_npx500.h5",
    ]), names
    for u in doc["unassigned_masters"]:
        assert u["rel"].startswith("Database/") and u["bytes"] > 0


def test_assign_files_gives_each_file_to_the_longest_matching_key(tmp_path):
    from backend.api.services.phase_library import assign_files
    files = [tmp_path / "beta-AlFeSi_withSi_COD-2107329_master_E20kV.h5",
             tmp_path / "Al13Fe4_master_E20kV.h5",
             tmp_path / "Al_master_E20kV.h5",
             tmp_path / "nobody_master_E20kV.h5"]
    got = assign_files(files, ["Al", "Al13Fe4", "beta-AlFeSi",
                               "beta-AlFeSi_withSi_COD-2107329"])
    assert [p.name for p in got["Al"]] == ["Al_master_E20kV.h5"]
    assert [p.name for p in got["Al13Fe4"]] == ["Al13Fe4_master_E20kV.h5"]
    assert "beta-AlFeSi" not in got
    assert len(got["beta-AlFeSi_withSi_COD-2107329"]) == 1


# --- citations, after the DOI decision -----------------------------------

@requires(CIF_LIBRARY)
def test_the_doi_is_read_from_all_three_places_and_stored_bare(doc, by_key):
    """10 of 36, and only ONE is in the references block.

    Reading only that block finds one; the others are in `_publ_section_doi` (8)
    and `_journal_paper_doi` (1). Stored bare, because half the files write
    `https://doi.org/` and keeping it makes the link
    `https://doi.org/https://doi.org/...`.
    """
    assert by_key["beta-AlFeSi"]["doi"] == "10.1107/S0108768193013096"
    assert by_key["α-(AlMnSi)"]["doi"] == "10.1107/S0108270197015989"
    assert by_key["Al3Fe2Si_mp-1190708_symmetrized"]["doi"] == "10.1063/1.4812323"
    assert sum(1 for p in doc["phases"] if p["doi"]) == 10
    for p in doc["phases"]:
        assert "http" not in (p["doi"] or ""), p["key"]
        assert (p["doi"] or "10.").startswith("10.")


@requires(CIF_LIBRARY)
def test_six_phases_stop_saying_no_source(doc):
    """Nine phases had no usable citation; six carry a DOI, so three remain."""
    neither = [p["key"] for p in doc["phases"]
               if not p["reference"] and not p["doi"]]
    assert len(neither) == 3, neither


def test_a_citation_that_begins_with_its_doi_is_kept():
    """Rejecting anything starting with http threw away whole references."""
    from backend.api.services.phase_library import read_doi, read_reference
    text = ("_publ_section_references\n;\n"
            "https://doi.org/10.1107/S0567740872007903 Cooper M. J.: The "
            "structure of the intermetallic phase theta-(Al-Cu). Acta "
            "Crystallogr. B 28 (1972) 2910-2916.\n;\n")
    ref, reason = read_reference(text)
    assert reason is None, reason
    assert ref.startswith("Cooper M. J."), ref
    assert "http" not in ref
    assert read_doi(text) == "10.1107/S0567740872007903"


def test_an_indented_terminator_still_closes_the_block():
    """CIF permits leading whitespace on the closing `;`."""
    from backend.api.services.phase_library import read_reference
    ref, reason = read_reference(
        "_publ_section_references\n;\nSmith J.: A title. Journal 1 (2000) 1.\n ;\n")
    assert reason is None and ref.startswith("Smith J."), (ref, reason)


def test_a_bare_non_doi_url_is_still_no_source():
    from backend.api.services.phase_library import read_reference
    assert read_reference(
        "_publ_section_references\n;\nhttps://example.org/x\n;\n"
    ) == (None, "bare_url")


# ==========================================================================
# b9's two contract decisions (rounding in the endpoint; `url` as its own field
# with a reference that survives only as ids becoming null), with the
# acceptance cases they named.
# ==========================================================================

@requires(CIF_LIBRARY)
def test_the_cell_is_rounded_once_in_the_endpoint(by_key):
    """Six decimals on a length, four on an angle.

    The index carries `4.651200000000001` and `119.99999999999999`. Leaving that
    to the card means every consumer invents its own precision -- and six
    decimals on an Angstrom is far finer than any of these files states, so
    nothing is lost.
    """
    assert by_key["Fe3_Al2_Si3"]["cell"]["a"] == 4.6512
    assert by_key["Al4Fe1.7Si (τ11)"]["cell"]["gamma"] == 120.0
    for key, p in by_key.items():
        for axis, v in p["cell"].items():
            if v is None:
                continue
            places = len(str(v).split(".")[-1]) if "." in str(v) else 0
            limit = 4 if axis in ("alpha", "beta", "gamma") else 6
            assert places <= limit, f"{key}.{axis} = {v}"


def test_rounding_cannot_fool_the_setting_guard():
    """The guard compares RAW values; only the output is rounded.

    Otherwise two numbers that round together would silently pass a check whose
    whole job is to notice they differ.
    """
    from backend.api.services.phase_library import _round_cell, resolve_cell

    class _E:
        key = "synthetic"
        lattice_a_A = lattice_b_A = lattice_c_A = 5.0
        lattice_alpha_deg = lattice_beta_deg = lattice_gamma_deg = 90.0

    text = ("data_x-standardized_unitcell\n_cell_length_a 3.5355\n"
            "_cell_length_b 3.5355\n_cell_length_c 3.5355\n")
    _, setting, _, conflict = resolve_cell(_E(), text)
    assert setting is None and conflict == "setting_mismatch"
    assert _round_cell({"a": 4.651200000000001, "alpha": 119.99999999999999}) == \
        {"a": 4.6512, "alpha": 120.0}


@requires(CIF_LIBRARY)
def test_a_reference_that_survives_only_as_an_id_becomes_null(by_key):
    """b9's acceptance case: `Al3Fe2Si_mp-1190708`.

    Its block is two bare URLs. Stripping only words that BEGIN with "http" left
    `mp-1190708_https://legacy.materialsproject.org/...` -- it starts with `mp` --
    and returned that as the phase's citation. URLs are now removed wherever they
    sit, and what remains must contain a word: `mp-1190708_` has only `mp`.
    """
    p = by_key["Al3Fe2Si_mp-1190708_symmetrized"]
    assert p["reference"] is None
    assert p["reference_rejected"] == "ids_only"
    assert p["url"] == "https://doi.org/10.1063/1.4812323"
    assert p["doi"] == "10.1063/1.4812323"


@requires(CIF_LIBRARY)
def test_the_springer_entries_keep_their_url(doc, by_key):
    """b9's other acceptance case: five SpringerMaterials entries.

    For them the URL is the only way to follow the source, so stripping it out of
    `reference` without keeping it anywhere would throw away the one link the
    card could offer. Their citation text stays as well.
    """
    springer = sorted(p["key"] for p in doc["phases"]
                      if (p["url"] or "").startswith("https://materials.springer.com"))
    assert springer == ["Mg2Zn11_sd_1251136", "MgZn2_sd_0261233", "ZnO_sd_1400158",
                        "sd_1814127", "sd_1816951"], springer
    for key in springer:
        p = by_key[key]
        assert p["reference"] and p["reference"].startswith("Pierre Villars")
        assert "http" not in p["reference"]


@requires(CIF_LIBRARY)
def test_no_reference_anywhere_still_contains_a_url(doc):
    """A URL inside the citation text is what severed the sentence on screen."""
    for p in doc["phases"]:
        assert "http" not in (p["reference"] or ""), p["key"]


def test_a_url_is_stripped_from_the_middle_of_a_citation():
    from backend.api.services.phase_library import read_reference, read_url
    text = ("_publ_section_references\n;\n"
            "Cooper M. J.: The structure of theta-(Al-Cu). "
            "https://doi.org/10.1107/S0567740872007903 "
            "Acta Crystallogr. B 28 (1972) 2910-2916.\n;\n")
    ref, reason = read_reference(text)
    assert reason is None
    assert "http" not in ref
    assert ref.startswith("Cooper M. J.") and ref.endswith("2910-2916.")
    assert read_url(text) == "https://doi.org/10.1107/S0567740872007903"


def test_a_block_of_only_ids_and_punctuation_is_not_a_citation():
    from backend.api.services.phase_library import read_reference
    for body in ("mp-1190708_https://legacy.materialsproject.org/materials/mp-1190708/",
                 "10.1063/1.4812323 ;;; ---",
                 "https://example.org/a  ,  ."):
        ref, reason = read_reference(f"_publ_section_references\n;\n{body}\n;\n")
        assert ref is None, (body, ref)
        assert reason in ("ids_only", "bare_url"), (body, reason)


@requires(CIF_LIBRARY)
def test_the_two_numbers_b9_corrected_in_the_spec(doc, by_key):
    """Hough 35 of 36, and four pre-built dictionaries.

    Hough excludes `sd_1816951`: the reader Hough uses parses that CIF to
    Mg 80 / Cu 20 at%. The fourth dictionary is `sd_0302719`'s, which sits in
    `Dictionary_Library/sd/` -- a folder named after neither the phase nor its
    stem, which is why it is matched on the FILE NAME.
    """
    assert doc["library_totals"]["capabilities"]["hough"] == 35
    assert by_key["sd_1816951"]["capabilities"]["hough"] is False
    with_dict = sorted(p["key"] for p in doc["phases"] if p["files"]["dictionary"])
    assert with_dict == ["Al", "Al7FeCu2", "Ni", "sd_0302719"], with_dict


# ==========================================================================
# §3.2 -- the Steckbrief, and b9's bracket one-liner.
# ==========================================================================

@requires(CIF_LIBRARY)
def test_the_detail_carries_the_row_plus_the_simulation_and_the_deep_link():
    """One phase's card: everything the row has, plus what is too big for 36."""
    from backend.api.services.phase_library import build_phase_detail
    d = build_phase_detail("Al")
    assert d is not None
    assert EXPECTED_FIELDS <= set(d), sorted(EXPECTED_FIELDS - set(d))
    assert {"simulation", "deep_link"} <= set(d)

    master = d["simulation"]["master"]
    assert master["program"] == "EMEBSDmaster.f90"
    assert master["EBSDMasterNameList"]["npx"] == 500
    assert master["MCCLNameList"]["EkeV"] == 20.0
    assert master["EBSDMasterNameList"]["dmin"] > 0

    sht = d["simulation"]["sht"]
    assert sht["parameters"]["npx"] == 500
    assert sht["parameters"]["voltage_kV"] == 20.0
    assert sht["crystallography"]["space_group"] == 225
    # existence measured here, never taken from the sidecar
    assert sht["provenance"]["source_xtal"]["found"] is True

    assert d["deep_link"]["cif"] == {
        "category": "cif_library", "name": "Al.cif",
        "rel": "Database/CIF_Library/Al.cif"}
    assert d["deep_link"]["master"]["category"] == "h5_cache"


@requires(CIF_LIBRARY)
def test_a_phase_without_a_master_has_no_master_simulation_block():
    """The card says what it has, and is silent about what it has not."""
    from backend.api.services.phase_library import build_phase_detail
    d = build_phase_detail("beta-AlFeSi")
    assert d["files"]["master"] is None
    assert "master" not in d["simulation"]
    assert "master" not in d["deep_link"]


@requires(CIF_LIBRARY)
def test_an_unknown_key_is_not_an_empty_card():
    from backend.api.services.phase_library import build_phase_detail
    assert build_phase_detail("no-such-phase") is None


@requires(CIF_LIBRARY)
def test_the_awkward_keys_survive_the_round_trip():
    """Library keys carry spaces, dots, parentheses and Greek.

    The key is a QUERY parameter, not a path segment, for exactly this reason --
    and the collections routes carry a structural test forbidding `{name}`
    segments after one shadowed a literal route.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.routes import phase_library as route
    assert [r.path for r in route.router.routes if "{" in r.path] == []
    app = FastAPI()
    app.include_router(route.router, prefix="/api/phase-library")
    c = TestClient(app)
    for key in ("Al4Fe1.7Si (τ11)", "α-(AlMnSi)", "Mn2(AlSi)5_ICSD-608515",
                "sd_1816951"):
        r = c.get("/api/phase-library/phase", params={"key": key})
        assert r.status_code == 200, (key, r.status_code, r.text[:80])
        assert r.json()["key"] == key
    assert c.get("/api/phase-library/phase",
                 params={"key": "nope"}).status_code == 404


def test_the_detail_route_reads_no_bulk_array():
    """`mLPNH` is 44 MB of a 210 MB file; a card costs kilobytes.

    Asserted on the source, because the cheap call and the ruinous one differ by
    two characters and both pass a behavioural test.
    """
    import inspect

    from backend.api.services import phase_library
    src = inspect.getsource(phase_library.read_master_parameters)
    body = src.split('"""', 2)[-1]
    for forbidden in ("EMData", "mLPNH", "[()]", "[:]", "np.array("):
        assert forbidden not in body, f"{forbidden} would read bulk data"


# --- b9's bracket one-liner ----------------------------------------------

@requires(CIF_LIBRARY)
def test_no_citation_reaches_the_card_with_an_open_bracket(doc):
    """`rest.strip(" ,;:-_/()[]")` ran unconditionally and ate a real `)`.

    `Fe3Si_MP-mp-2199`'s citation legitimately ends
    `... (ICSD: icsd-56281, ..., icsd-157941)`. Four phases were affected, and
    none of them had a URL -- which is what hid it: the strip was written for
    URL leftovers and ran on everything.
    """
    bad = [p["key"] for p in doc["phases"] if p["reference"]
           and (p["reference"].count("(") != p["reference"].count(")")
                or p["reference"].count("[") != p["reference"].count("]"))]
    assert bad == [], bad
    for key in ("Fe3Si_MP-mp-2199", "Mg17Al12_MP-mp-2151", "Mg2Si_MP-mp-1367"):
        ref = {p["key"]: p for p in doc["phases"]}[key]["reference"]
        assert ref.endswith(")"), (key, ref[-40:])


def test_a_balanced_bracket_survives_and_an_orphan_does_not():
    from backend.api.services.phase_library import _trim_junk
    assert _trim_junk("Smith J.: A title (ICSD: 1)") == "Smith J.: A title (ICSD: 1)"
    assert _trim_junk("Smith J.: A title)") == "Smith J.: A title"
    assert _trim_junk("Smith J.: A title,") == "Smith J.: A title"
    assert _trim_junk("Smith J.: A title.") == "Smith J.: A title."


@requires(CIF_LIBRARY)
def test_a_url_inside_brackets_does_not_orphan_the_opener(by_key):
    """A citation writes the link in parentheses.

    A URL class that swallowed the `)` left the `(` mid-sentence, and the card
    read `... mp-aaacqrcb ( A. Jain et al. ...`. Found by checking my own claim
    that three of four were balanced.
    """
    ref = by_key["Al2Zn_MP-mp-aaacqrcb"]["reference"]
    assert "( A." not in ref and " ( " not in ref
    assert ref.count("(") == ref.count(")")
    assert "mp-aaacqrcb. A. Jain" in ref, ref
    assert by_key["Al2Zn_MP-mp-aaacqrcb"]["url"] == \
        "https://materialsproject.org/materials/mp-aaacqrcb", "no trailing )"


@requires(CIF_LIBRARY)
def test_no_reference_has_a_gap_before_its_punctuation(doc):
    """Removing a bracketed URL left `mp-aaacqrcb . A. Jain`."""
    import re as _re
    for p in doc["phases"]:
        ref = p["reference"] or ""
        assert "  " not in ref, p["key"]
        assert not _re.search(r"\s[.,;:]", ref), (p["key"], ref[:80])
