"""Two ways Crystal Hint lost phases whose files were perfectly good.

**Charged element symbols.** A CIF `_atom_site_type_symbol` should be a bare
element, but ICSD exports write `Al0+`. pymatgen keeps the charge in its
composition keys, so `LocalEntry.elements` held `('Al0+','Fe0+','Mn0+','Si0+')`
while `search()` intersects against `capitalize()`d symbols — an empty
intersection every time. Measured on the shipped library before the fix:

    search(['Al','Fe','Mn','Si','Mg'], strict_chemistry=True)   22 candidates
        alpha-AlFeMnSi_ICSD-52623      DROPPED     (chemistry is exactly right)
        Al3FeSi2_tau2_ICSD-607628      DROPPED
        Mn2(AlSi)5_ICSD-608515         DROPPED
        pi-Al8FeMg3Si6_ICSD-27140      DROPPED
        sd_0302719 (same chemistry, plain symbols)   present
    …_chemistry=False: all four scored 0.0

after: 26 candidates, all four at plausibility 1.00. The normaliser already
existed in `ebsd_utils` for the diffsims reflector path (same root cause: a
charged symbol makes diffsims return zero scattering). It now lives in
`phase_metadata` and both consumers share it, rather than a second copy.

**An unreadable structure hid the phase completely.** `search` skipped every
entry with a `parse_error`, and `sd_1816951.cif` (MgCu2) gives pymatgen two
compositions. Measured before the fix:

    search(['Mg','Cu'], strict_chemistry=True)    0 results

MgCu2 is the library's only Mg-Cu phase, so the phase that IS the answer was the
only one missing — while its entry already carried elements from its `.xtal`, an
IT number and a `.sht`. It is now offered, with the reason attached.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from phase_metadata import (
    build_canonical_label, clean_element_symbol, extract_elements, phase_nickname,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY = PROJECT_ROOT / "Database" / "CIF_Library"

# The four library files, written the way ICSD writes them. Real cells, but the
# structures are synthetic: the point is the `0+` on the type symbol.
_CHARGED_CIF = """\
data_synthetic_icsd
_cell_length_a                   12.6430
_cell_length_b                   12.6430
_cell_length_c                   12.6430
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
_symmetry_space_group_name_H-M   'I m -3'
_symmetry_Int_Tables_number      204
_chemical_formula_sum            'Al100 Fe21 Mn5 Si10'
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
  Al1 Al0+ 0.0000 0.0000 0.0000 1.0
  Fe1 Fe0+ 0.2500 0.2500 0.2500 1.0
  Mn1 Mn0+ 0.5000 0.0000 0.0000 1.0
  Si1 Si0+ 0.0000 0.5000 0.2500 1.0
"""

# Two blocks that disagree about the COMPOSITION — the sd_1816951 shape, whose
# real error message is "parses to 2 different compositions (Mg2Cu, Mg4Cu)".
#
# Both blocks must actually PARSE, or pymatgen returns one structure and nothing
# is ambiguous: the first version of this fixture had blocks that failed for
# other reasons, the entry came back with no `parse_error` and a wrong formula
# (`MgCu`, R3m), and the fixture's own guard assertion caught it. So both blocks
# carry explicit symmetry (`x, y, z`, hence composition = the atom list) and the
# SAME cell and H-M symbol, so the block-consistency guard stays quiet and only
# the composition differs. The `_sm_*` side stays readable throughout.
_AMBIGUOUS_CIF = """\
data_sm_global
_sm_phase_labels                 'MgCu2'
_sm_pearson_symbol               'cF24'
_symmetry_space_group_name_H-M   'Fd-3m'
_symmetry_Int_Tables_number      227

data_sm_isp_SD1816951-standardized_unitcell
_cell_length_a                   7.0309
_cell_length_b                   7.0309
_cell_length_c                   7.0309
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
_symmetry_space_group_name_H-M   'Fd-3m'
_symmetry_Int_Tables_number      227
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
  Mg1 Mg 0.1250 0.1250 0.1250 1.0
  Cu1 Cu 0.5000 0.5000 0.5000 1.0
  Cu2 Cu 0.5000 0.2500 0.7500 1.0

data_sm_isp_SD1816951-published_cell
_cell_length_a                   7.0309
_cell_length_b                   7.0309
_cell_length_c                   7.0309
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
_symmetry_space_group_name_H-M   'Fd-3m'
_symmetry_Int_Tables_number      227
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
  Mg1 Mg 0.1250 0.1250 0.1250 1.0
  Mg2 Mg 0.6250 0.6250 0.6250 1.0
  Cu1 Cu 0.5000 0.5000 0.5000 1.0
"""


# --- the normaliser -------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("Al0+", "Al"), ("Fe0+", "Fe"), ("Fe3+", "Fe"), ("O2-", "O"),
    ("Mn0+", "Mn"), ("Si0+", "Si"),
    ("Al", "Al"), ("al", "Al"), ("  Cu  ", "Cu"),
    ("0.884Al + 0.116Si", "Al"),      # mixed site -> dominant element
    ("0.2Fe + 0.8Mn", "Mn"),
])
def test_a_charged_or_shared_symbol_becomes_one_element(raw, want):
    assert clean_element_symbol(raw) == want


def test_it_never_invents_an_element():
    """Unresolvable input comes back unchanged, not as a guess or a blank."""
    assert clean_element_symbol("Xx9+") == "Xx9+"
    assert clean_element_symbol("") == ""
    assert clean_element_symbol(None) == "", (
        "`str(None)` is 'None', which capitalize()s into neon")


def test_it_is_idempotent():
    once = clean_element_symbol("Al0+")
    assert clean_element_symbol(once) == once


def test_it_is_not_a_formula_parser_and_says_so():
    """Pinning the trap, so the next reader of the docstring believes it.

    `_MIX_TERM_RE` expects the coefficient BEFORE the symbol, because a shared
    CIF site is written `0.884Al + 0.116Si`. A formula puts it after, so the
    subscript is credited to the following element and "dominant" inverts.
    """
    assert clean_element_symbol("Al2Cu") == "Cu"      # NOT Al
    assert clean_element_symbol("Mg17Al12") == "Al"   # NOT Mg
    # the function to use instead:
    assert extract_elements("Al2Cu") == ["Al", "Cu"]


def test_a_space_group_is_not_an_element():
    """`extract_elements` keeps a narrower table on purpose.

    Its input is often not a formula: an .sht stem with no linked CIF becomes its
    own `formula`, so `NiAl_Pm-3m` gets parsed. With the full periodic table,
    `Pm` and `Pa` — promethium and protactinium — are read out of the SPACE
    GROUP, and the phase's element group changes with them. A library entry of
    exactly that shape (`Al-Cu-Fe-Si_Pm-3_ICSD-99302`) existed here until June.
    """
    assert extract_elements("NiAl_Pm-3m") == ["Al", "Ni"]
    assert extract_elements("Al2Cu_Pa-3") == ["Al", "Cu"]
    assert extract_elements("SomePhase_Pmma") == []
    # …while the site-symbol cleaner must still know those elements exist
    assert clean_element_symbol("Pm3+") == "Pm"
    assert clean_element_symbol("Tc") == "Tc"


def test_ebsd_utils_shares_the_same_function_rather_than_a_copy():
    """One implementation, two consumers. A second copy is how they diverge."""
    import ebsd_utils
    import phase_metadata
    assert ebsd_utils._clean_element_label is phase_metadata.clean_element_symbol
    assert ebsd_utils._KNOWN_ELEMENTS is phase_metadata.KNOWN_ELEMENTS


# --- what the charge cost, at the reader ---------------------------------

def _index_over(tmp_path, monkeypatch, cifs: dict[str, str]):
    from backend.api.services import crystal_hint_local_library as lib
    cif_dir = tmp_path / "CIF_Library"
    cif_dir.mkdir()
    for name, body in cifs.items():
        (cif_dir / name).write_text(body, encoding="utf-8")
    monkeypatch.setattr(lib, "CIF_DIR", cif_dir)
    monkeypatch.setattr(lib, "XTAL_DIR", tmp_path / "absent_xtal")
    monkeypatch.setattr(lib, "SHT_DIR", tmp_path / "absent_sht")
    monkeypatch.setattr(lib, "_INDEX_CACHE", None, raising=False)
    return lib, lib.rebuild_index()


def test_a_charged_cif_yields_plain_element_symbols(tmp_path, monkeypatch):
    _, idx = _index_over(tmp_path, monkeypatch, {"charged.cif": _CHARGED_CIF})
    assert idx["charged"].elements == ("Al", "Fe", "Mn", "Si"), (
        "pymatgen's keys carry the charge; the index must not")


def test_a_charged_cif_survives_the_strict_chemistry_filter(tmp_path, monkeypatch):
    """The user-visible defect: the phase was dropped, not merely mis-scored."""
    lib, _ = _index_over(tmp_path, monkeypatch, {"charged.cif": _CHARGED_CIF})
    hits = lib.search(["Al", "Fe", "Mn", "Si"], strict_chemistry=True)
    assert [m.entry.key for m in hits] == ["charged"]
    assert hits[0].plausibility == pytest.approx(1.0)


def test_a_charged_cif_scores_on_shared_chemistry(tmp_path, monkeypatch):
    lib, _ = _index_over(tmp_path, monkeypatch, {"charged.cif": _CHARGED_CIF})
    hits = lib.search(["Al", "Fe"], strict_chemistry=False)
    assert hits and hits[0].plausibility > 0.0, (
        "a charged symbol shared nothing with a real symbol, so every "
        "plausibility was 0.0")


def test_the_pymatgen_reader_agrees_with_pymatgens_own_element_list():
    """The index asks pymatgen for the elements instead of cleaning its keys.

    `Element.symbol` cannot carry a charge by construction, so there is no string
    rule left to get wrong. This pins that the index and pymatgen's own oracle
    agree on every readable library CIF — the property the old normalising call
    was trying to achieve indirectly.
    """
    if not LIBRARY.is_dir():
        pytest.skip("crystal library not in this checkout")
    import warnings

    from pymatgen.io.cif import CifParser

    from backend.api.services import crystal_hint_local_library as lib
    from backend.api.services.cif_phase_library import one_structure
    idx = lib.rebuild_index()
    checked = 0
    for key, e in idx.items():
        if e.cif_path is None or e.parse_error:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = one_structure(CifParser(str(e.cif_path)).parse_structures(
                primitive=False), e.cif_path.name)
        assert e.elements == tuple(sorted({el.symbol for el in s.composition.elements})), key
        checked += 1
    assert checked >= 30, f"only {checked} CIFs compared"


def test_the_nickname_needs_clean_symbols(tmp_path, monkeypatch):
    """`_PHASE_NICKNAMES` is keyed on (space group, sorted elements).

    Latent rather than live — no backend caller passes library elements into
    `build_canonical_label` today — but it is the same key mismatch, and the
    fallback prints the charges AT the user, which is worse than losing the
    nickname.
    """
    assert phase_nickname("Im-3", ("Al0+", "Fe0+", "Mn0+", "Si0+")) == ""
    assert phase_nickname("Im-3", ("Al", "Fe", "Mn", "Si")) == "α-Al(Fe,Mn)Si"

    charged = build_canonical_label(
        formula="Mn5.268Al100.89Fe21.24Si10.602", space_group="Im-3",
        pearson="cI138", elements=("Al0+", "Fe0+", "Mn0+", "Si0+"))
    clean = build_canonical_label(
        formula="Mn5.268Al100.89Fe21.24Si10.602", space_group="Im-3",
        pearson="cI138",
        elements=tuple(sorted(clean_element_symbol(s)
                              for s in ("Al0+", "Fe0+", "Mn0+", "Si0+"))))
    assert "0+" in charged and "α-Al(Fe,Mn)Si" not in charged
    assert "0+" not in clean and clean.endswith("α-Al(Fe,Mn)Si")


# --- an unreadable structure is no longer an invisible phase -------------

def test_an_ambiguous_cif_still_becomes_a_candidate(tmp_path, monkeypatch):
    lib, idx = _index_over(tmp_path, monkeypatch,
                           {"ambiguous.cif": _AMBIGUOUS_CIF})
    e = idx["ambiguous"]
    assert e.parse_error, "the fixture must actually be refused by pymatgen"
    # …and the readable side was rescued
    assert e.formula == "MgCu2"
    assert e.space_group == "Fd-3m"
    assert e.space_group_number == 227
    assert e.crystal_system == "cubic"
    assert e.lattice_a_A == pytest.approx(7.0309)
    # NOT elements: the rescue must not derive chemistry from a label (see
    # `test_the_rescue_never_adds_an_element_the_structure_lacks`). With no
    # sibling .xtal there is nothing to supply them, and the entry is still
    # offered on its space group.
    assert e.elements == ()
    assert [m.entry.key for m in lib.search(["Mg", "Cu"],
                                            strict_chemistry=True)] == ["ambiguous"]


def test_the_rescue_never_adds_an_element_the_structure_lacks():
    """Two shipped CIFs label themselves with an element they have no site for.

    `beta-AlFeSi.cif` says `_chemical_formula_sum 'Al4 Fe Si'` and has no Si site;
    `sd_1401510.cif` labels itself `Mn0.5Fe0.5Al5Si0.68` with only Al and Fe/Mn
    sites. Deriving `elements` from the label would CLAIM that Si. Measured cost
    of doing it, with the rescue forced on those two:

        search(['Al','Fe','Mn'], strict)  ['beta-AlFeSi','sd_1401510'] -> []

    i.e. the rescue would make the phases it exists to save vanish for their own
    chemistry. So `_readable_without_structure` returns no `elements` at all.
    """
    from backend.api.services.crystal_hint_local_library import (
        _readable_without_structure,
    )
    if not LIBRARY.is_dir():
        pytest.skip("crystal library not in this checkout")
    for name in ("beta-AlFeSi.cif", "sd_1401510.cif"):
        p = LIBRARY / name
        if not p.is_file():
            continue
        assert "elements" not in _readable_without_structure(p), name


def test_the_rescue_never_puts_the_filename_in_the_formula(tmp_path):
    """`extract_metadata_from_cif` ends its formula chain at the file stem.

    `meta.source` is "cif" either way, so the only way to tell a real formula
    from a filename is to compare with the stem. It matters because
    `crystal_hint_phase_fit.phase_nominal_at_pct` parses this field for the EDS
    multiplier — a filename there is a composition invented out of a name.
    """
    from backend.api.services.crystal_hint_local_library import (
        _readable_without_structure,
    )
    p = tmp_path / "SomePhase_Pm-3m.cif"
    p.write_text("data_x\n_cell_length_a 4.05\n", encoding="utf-8")
    out = _readable_without_structure(p)
    assert "formula" not in out, out


def test_a_half_read_phase_cannot_outrank_a_fully_read_one(monkeypatch):
    """Three of the five ranking multipliers are free passes when input is missing.

    No lattice means the lattice filter and `dspacing_fit` both skip; an unknown
    crystal system makes `symmetry_fit` return 1.0; a formula that parses to no
    element makes `chemistry_fit` return its neutral 1.0. So an entry that knows
    nothing paid nothing, and a review built one that scored 1.0000 and came
    FIRST on an Al-Cu pixel, ahead of Al2Cu (0.6724) and Al (0.6421).

    The entries are injected rather than parsed from fixtures, because the first
    version of this test used CIF fixtures and passed WITHOUT the penalty: the
    half-read entry happened to have no elements, so it lost on plausibility
    (0.5 vs 1.0) for an unrelated reason. It asserted the right thing and proved
    nothing — removing the penalty left it green. Here both entries have the same
    chemistry, and the only difference is that the fully-read one HAS a cell and
    is slightly outside the requested range, so it pays 1/(1+0.036) = 0.966 while
    the half-read one pays nothing at all.
    """
    from backend.api.services import crystal_hint_local_library as lib

    cases = {
        "half_read": lib.LocalEntry(
            key="half_read", parse_error="composition cannot be read",
            elements=("Al", "Cu"), space_group_number=227,
            crystal_system="cubic"),                      # no lattice at all
        "fully_read": lib.LocalEntry(
            key="fully_read", elements=("Al", "Cu"), space_group_number=225,
            crystal_system="cubic", lattice_a_A=4.05),
    }
    monkeypatch.setattr(lib, "_INDEX_CACHE", cases, raising=False)
    monkeypatch.setattr(lib, "get_index", lambda: cases)

    hits = lib.search(["Al", "Cu"], crystal_system="cubic",
                      a_range_A=(4.2, 5.0), strict_chemistry=True)
    by_key = {m.entry.key: m.score for m in hits}
    assert set(by_key) == {"half_read", "fully_read"}, by_key
    assert by_key["fully_read"] > by_key["half_read"], (
        f"half-read {by_key['half_read']:.4f} outranks fully-read "
        f"{by_key['fully_read']:.4f}: a phase whose file cannot be parsed came "
        f"first because it paid none of the multipliers")
    assert [m.entry.key for m in hits][0] == "fully_read"


def test_the_candidate_says_its_structure_was_not_readable(tmp_path, monkeypatch):
    """Offered, but never presented as fully characterised."""
    from backend.api.routes.crystal_hint import _local_match_to_out
    lib, _ = _index_over(tmp_path, monkeypatch,
                         {"ambiguous.cif": _AMBIGUOUS_CIF})
    hits = lib.search(["Mg", "Cu"], strict_chemistry=True)
    out = _local_match_to_out(hits[0], None)
    assert out.structure_unreadable, "the reason must reach the API"
    assert "composition" in out.structure_unreadable.lower()
    # No elements: this fixture has no sibling .xtal, and the rescue does not
    # invent chemistry from a label. The real sd_1816951 gets ('Cu','Mg') from
    # its .xtal — asserted in `test_the_only_mg_cu_phase_is_found`.
    assert out.elements == []


def test_an_entry_with_nothing_to_match_on_is_still_skipped(tmp_path, monkeypatch):
    """The skip was too broad, not wrong. An empty entry stays out."""
    lib, idx = _index_over(tmp_path, monkeypatch,
                           {"junk.cif": "not a cif at all\n"})
    e = idx.get("junk")
    if e is None or not e.parse_error:
        pytest.skip("pymatgen accepted the junk fixture; nothing to assert")
    assert not e.elements and not e.space_group_number
    assert lib.search(["Al"], strict_chemistry=False) == []


def test_either_elements_or_a_space_group_is_enough_to_be_offered(monkeypatch):
    """`and`, not `or`, in the narrowed skip.

    A review mutated it to `or` and NO test noticed, because `sd_1816951` happens
    to have both and every other parse-error entry has neither. The mutation
    re-drops exactly the case the fix is justified by — "it carried elements from
    its .xtal AND an IT number" — so each half has to be enough on its own.
    """
    from backend.api.services import crystal_hint_local_library as lib

    def _entry(**kw):
        return lib.LocalEntry(key=kw.pop("key"), parse_error="unreadable", **kw)

    cases = {
        "els_only": _entry(key="els_only", elements=("Al",)),
        "sg_only": _entry(key="sg_only", space_group_number=225,
                          crystal_system="cubic"),
        "neither": _entry(key="neither"),
    }
    monkeypatch.setattr(lib, "_INDEX_CACHE", cases, raising=False)
    monkeypatch.setattr(lib, "get_index", lambda: cases)
    keys = {m.entry.key for m in lib.search(["Al"], strict_chemistry=False)}
    assert "els_only" in keys, "elements alone must be enough"
    assert "sg_only" in keys, "a space group alone must be enough"
    assert "neither" not in keys, "an entry with nothing to match on stays out"


def test_hough_refuses_a_cif_whose_structure_is_not_uniquely_readable():
    """The caveat has to be ENFORCED where the file is handed over, not displayed.

    Hough computes its reflectors from the CIF, and the path that reads it is
    `orix Phase.from_cif(sanitize_cif(...))` — diffpy, not pymatgen — which does
    NOT refuse `sd_1816951.cif`. Measured: it returns 40 sites, Cu 8 / Mg 32, so
    **Mg 80 / Cu 20 at%**, while the .xtal EMsoft simulated the master from has
    Atomtypes [29, 12] = Cu 16c + Mg 8b = **Cu 66.7 / Mg 33.3**. Handing that
    file to the reflector builder computes structure factors from a composition
    that is not the phase, in silence. `parse_error` is the only thing that knows.
    """
    if not (LIBRARY / "sd_1816951.cif").is_file():
        pytest.skip("sd_1816951.cif is not on this machine")
    from backend.api.routes.phase_collections import resolve
    from backend.api.services import crystal_hint_local_library as lib
    from backend.api.services import phase_collections as pc

    lib.rebuild_index()
    name = "guard-structure-unreadable"
    # The collections directory is redirected: a previous live test left three
    # stray JSONs in the user's real `Database/Collections/`.
    with tempfile.TemporaryDirectory() as tmp:
        pc.set_collections_dir_for_test(Path(tmp))
        try:
            pc.save(pc.PhaseCollection(
                name=name, members=[pc.PhaseMember(key="sd_1816951")]))
            out = resolve(name=name, method="hough")
            assert out["paths"] == [], "the unreadable CIF was handed to Hough"
            assert [m["reason"] for m in out["missing"]] == \
                ["structure_unreadable"]
            # …and Spherical is unaffected: the .sht does not come from that CIF
            sph = resolve(name=name, method="spherical")
            assert len(sph["paths"]) == 1, sph
        finally:
            pc.set_collections_dir_for_test(None)


# --- and the real library, because a synthetic fixture is not the library --

def test_the_shipped_library_has_no_charged_symbols_left():
    if not LIBRARY.is_dir():
        pytest.skip("crystal library not in this checkout")
    from backend.api.services import crystal_hint_local_library as lib
    idx = lib.rebuild_index()
    assert len(idx) >= 30, f"only {len(idx)} entries — not the real library"
    charged = {k: e.elements for k, e in idx.items()
               if any("+" in s or "-" in s for s in e.elements)}
    assert charged == {}, f"charged symbols survived for: {charged}"
    empty = [k for k, e in idx.items() if not e.elements]
    assert empty == [], f"entries left with no elements at all: {empty}"


def test_the_four_icsd_phases_are_offered_again():
    """The measurement in this module's docstring, as an assertion."""
    if not LIBRARY.is_dir():
        pytest.skip("crystal library not in this checkout")
    from backend.api.services import crystal_hint_local_library as lib
    lib.rebuild_index()
    keys = {m.entry.key for m in
            lib.search(["Al", "Fe", "Mn", "Si", "Mg"], strict_chemistry=True)}
    for k in ("alpha-AlFeMnSi_ICSD-52623", "Al3FeSi2_tau2_ICSD-607628",
              "Mn2(AlSi)5_ICSD-608515", "pi-Al8FeMg3Si6_ICSD-27140"):
        assert k in keys, f"{k} is still dropped by the chemistry filter"


def test_the_only_mg_cu_phase_is_found():
    if not LIBRARY.is_dir():
        pytest.skip("crystal library not in this checkout")
    from backend.api.services import crystal_hint_local_library as lib
    lib.rebuild_index()
    hits = lib.search(["Mg", "Cu"], strict_chemistry=True)
    assert [m.entry.key for m in hits] == ["sd_1816951"], (
        "this search returned zero results before the fix")
