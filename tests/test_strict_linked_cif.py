"""``phase_metadata.strict_linked_cif`` -- the one identity rule behind per-phase
routing and the Indexing page's hand-off.

Tested with the NAMES the real library uses, in a temporary CIF library, so the
tests mean something in a checkout without ``Database/``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import phase_metadata  # noqa: E402
from phase_metadata import strict_linked_cif  # noqa: E402

CIFS = ["Al.cif", "Si.cif", "alpha-AlFeMnSi_ICSD-52623.cif", "sd_0302719.cif",
        "alpha-Fe.cif", "Ni.cif", "alpha-(AlFeSi)_x.cif"]
SUFFIX = "_master_E20kV_npx500_dict_20kV_128x156_pc503_326_846_2.0deg.h5"


@pytest.fixture()
def library(tmp_path):
    lib = tmp_path / "CIF_Library"
    lib.mkdir()
    for name in CIFS:
        (lib / name).write_text("data_x\n", encoding="utf-8")
    # alpha-Fe is a real, parseable CIF: its formula is what the loose rule matches
    (lib / "alpha-Fe.cif").write_text(
        "data_alphaFe\n_chemical_formula_sum 'Fe'\n"
        "_cell_length_a 2.87\n_cell_length_b 2.87\n_cell_length_c 2.87\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_Int_Tables_number 229\n_symmetry_space_group_name_H-M 'I m -3 m'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
        "Fe Fe 0 0 0 1\n", encoding="utf-8")
    return lib


def link(library, filename):
    got = strict_linked_cif(Path("/somewhere") / filename, library)
    return None if got is None else got.name


@pytest.mark.parametrize("filename, cif", [
    ("Al (Al) [cF4] {20kV}.sht", "Al.cif"),
    ("Al100.89Fe21.24Mn5.31Si10.62 (alpha-AlFeMnSi_ICSD-52623) [cP138] {20kV}.sht",
     "alpha-AlFeMnSi_ICSD-52623.cif"),
    (f"Al{SUFFIX}", "Al.cif"),
    (f"sd_0302719{SUFFIX}", "sd_0302719.cif"),
    ("Ni_master_E20kV_npx500.h5", "Ni.cif"),
    ("Si.sht", "Si.cif"),                      # same stem
    ("Si.h5", "Si.cif"),
])
def test_names_of_the_real_library_link_to_their_cif(library, filename, cif):
    assert link(library, filename) == cif


def test_a_renamed_sht_is_not_linked_to_another_polymorph_by_formula(library):
    # gamma-Fe renamed away: the formula (Fe) matches alpha-Fe.cif, the name does not.
    # The trap exists -- the display rule falls into it -- and the strict one does not.
    renamed = Path("/somewhere") / "Fe (gamma-Fe_RENAMED) [cF4] {20kV}.sht"
    assert phase_metadata.find_linked_cif(renamed, library).name == "alpha-Fe.cif"
    assert link(library, renamed.name) is None
    # and a stem with no CIF of that name stays unlinked as well
    assert link(library, "Fe3Al (Fe3Al) [cF16] {20kV}.sht") is None


def test_a_cif_name_with_parentheses_of_its_own_still_links(library):
    # the source-CIF name itself contains "(AlFeSi)": only the whole name finds it
    assert link(library, "Fe23Al81Si15 (alpha-(AlFeSi)_x) [cI128] {20kV}.sht") \
        == "alpha-(AlFeSi)_x.cif"


def test_the_cif_is_found_among_several_parenthesised_groups(library):
    # the greedy whole name ("Bar) (Al") finds nothing; the group "(Al)" does
    assert link(library, "Foo (Bar) (Al) [cF4] {20kV}.sht") == "Al.cif"


def test_a_dictionary_without_a_parameter_suffix_is_not_guessed(library):
    assert link(library, "Al_dict_20kV_128x156_2.0deg.h5") is None


def test_parentheses_naming_two_different_cifs_are_ambiguous(library):
    assert link(library, "Al (Si) (Al) [cF4] {20kV}.sht") is None


def test_the_cif_comes_back_with_the_spelling_on_disk(library):
    assert link(library, "al (AL) [cF4] {20kV}.sht") == "Al.cif"
    assert link(library, "ni_master_e20kv_npx500.h5") == "Ni.cif"


def test_a_cif_is_its_own_cif(library, tmp_path):
    elsewhere = tmp_path / "mine" / "Strange.cif"
    assert strict_linked_cif(elsewhere, library) == elsewhere


def test_a_missing_library_links_nothing(tmp_path):
    assert strict_linked_cif(tmp_path / "Al.sht", tmp_path / "nope") is None


def test_other_extensions_link_by_their_own_stem_only(library):
    assert link(library, "Al.xtal") == "Al.cif"
    assert link(library, "Al (Al) [cF4] {20kV}.xtal") is None     # no (…) rule for them


# --- the routing join and the discovery use the SAME rule ------------------------

NAMES = ["Al (Al) [cF4] {20kV}.sht", "Fe (gamma-Fe_RENAMED) [cF4] {20kV}.sht",
         f"Al{SUFFIX}", "Al_dict_20kV_2.0deg.h5", "Si.sht"]


def test_the_routing_join_and_the_discoverys_linked_cif_agree(library, tmp_path):
    from backend.api.routes import indexing as route
    from indexing_controller import enrich_phase_entries

    # routing side
    req = SimpleNamespace(method="spherical", cif_paths=[], master_h5_paths=[],
                          sht_paths=[str(tmp_path / n) for n in NAMES])
    routed = {Path(f).name: cif for f, cif in route.phase_map_file_links(req, library)}

    # discovery side: the same files as records, library under <db_root>/CIF_Library
    db_root = library.parent
    entries = [{"path": tmp_path / n, "filename": n} for n in NAMES]
    enrich_phase_entries(entries, db_root)
    discovered = {e["filename"]: e["linked_cif"] for e in entries}

    assert routed == discovered
    assert discovered["Al (Al) [cF4] {20kV}.sht"] == "Al.cif"
    assert discovered["Fe (gamma-Fe_RENAMED) [cF4] {20kV}.sht"] is None


def test_both_sides_go_through_the_strict_function_and_not_find_linked_cif(
        library, tmp_path, monkeypatch):
    from backend.api.routes import indexing as route
    from indexing_controller import enrich_phase_entries

    calls = []

    def strict(path, cif_dir):
        calls.append(Path(path).name)
        return Path(cif_dir) / "Al.cif"

    def loose(*a, **k):
        raise AssertionError("find_linked_cif (formula search) decided a join")

    monkeypatch.setattr(phase_metadata, "strict_linked_cif", strict)
    monkeypatch.setattr(phase_metadata, "find_linked_cif", loose)
    req = SimpleNamespace(method="spherical", cif_paths=[], master_h5_paths=[],
                          sht_paths=[str(tmp_path / "X.sht")])
    assert route.phase_map_file_links(req, library) == [(str(tmp_path / "X.sht"), "Al.cif")]
    entries = [{"path": tmp_path / "Y.sht", "filename": "Y.sht"}]
    try:
        enrich_phase_entries(entries, library.parent)
    except AssertionError:
        pass            # the metadata part may use find_linked_cif; the join must not
    assert entries[0]["linked_cif"] == "Al.cif"
    assert calls == ["X.sht", "Y.sht"]
