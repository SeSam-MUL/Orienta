"""The spreadsheet builder reads the citation the way the converter does, and a
missing reference no longer makes a file "unfit for .xtal" there."""
import importlib.util
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "crystal-structures-for-ebsd-main" / "calculationxtal" / "cif_database_builder.py"

AL = """\
data_Al
{extra}_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number   225
_cell_length_a   4.05
_cell_length_b   4.05
_cell_length_c   4.05
_cell_angle_alpha   90
_cell_angle_beta   90
_cell_angle_gamma   90
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al1  Al  0  0  0  1.0
"""


@pytest.fixture(scope="module")
def builder():
    sys.path.insert(0, str(ROOT))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = importlib.util.spec_from_file_location("cif_database_builder_under_test", BUILDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _row(builder, tmp_path, extra=""):
    path = tmp_path / "Al.cif"
    path.write_text(AL.format(extra=extra), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return builder.parse_cif_file(path)


def test_no_reference_is_fit_but_warned(builder, tmp_path):
    row = _row(builder, tmp_path)
    assert row["Fit for .xtal"] == "Yes"
    assert row["DOI"] == "none found" and row["Reference Text"] == "none found"
    assert "No reference or DOI found." in row["Warnings"]


def test_an_unquoted_doi_and_a_doi_link_are_read_like_the_converter_reads_them(builder, tmp_path):
    assert _row(builder, tmp_path, "_publ_section_doi 10.1000/bare\n")["DOI"] == "10.1000/bare"
    assert _row(builder, tmp_path, "_citation_journal_doi 'https://doi.org/10.1000/link'\n")["DOI"] == "10.1000/link"


def test_a_doi_column_of_a_citation_loop_is_read_by_column(builder, tmp_path):
    # `primary` (the id column) is not the DOI; the DOI column's first row is.
    extra = "loop_\n_citation_id\n_citation_year\n_citation_doi\nprimary 1999 10.1000/loopvalue\n"
    assert _row(builder, tmp_path, extra)["DOI"] == "10.1000/loopvalue"
    extra = "loop_\n_citation_id\n_citation_doi\nprimary n/a\n"
    assert _row(builder, tmp_path, extra)["DOI"] == "none found"


def test_a_reference_text_is_kept(builder, tmp_path):
    row = _row(builder, tmp_path, "_publ_section_references\n;\nDoe, J. (2001). J. Test 1, 2.\n;\n")
    assert row["DOI"] == "none found"
    assert row["Reference Text"] == "Doe, J. (2001). J. Test 1, 2."
    assert "No reference" not in row["Warnings"]


def test_no_atoms_is_still_unfit(builder, tmp_path):
    path = tmp_path / "Empty.cif"
    path.write_text(
        "data_E\n_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "_cell_length_a 4\n_cell_length_b 4\n_cell_length_c 4\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = builder.parse_cif_file(path)
    assert row is None or row["Fit for .xtal"] == "No"


def test_journal_fields_become_the_reference_text_of_the_spreadsheet_row(builder, tmp_path):
    extra = (
        "loop_\n_publ_author_name\n'Panday P.K.'\n'Schubert K.'\n"
        "_journal_name_full 'Journal of the Less-Common Metals'\n"
        "_journal_page_first 395\n_journal_page_last 401\n_journal_volume 18\n_journal_year 1969\n"
    )
    row = _row(builder, tmp_path, extra)
    assert row["DOI"] == "none found"
    assert row["Reference Text"] == "Panday P.K., Schubert K.: Journal of the Less-Common Metals 18 (1969) 395-401"
    assert "No reference" not in row["Warnings"]
