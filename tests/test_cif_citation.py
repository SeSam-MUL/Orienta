"""CIF citation: one reader, an honest answer when there is none.

An external user could not convert a CIF without DOI or reference to .xtal
("No reference or DOI found") and had to type one into the Crystal Database. The
gate was pure policy, and behind it three code paths read the citation with
three different regular expressions, so a CIF could show a DOI in the list and
still convert as "none found".

The expectations below are literal CIF snippets and literal strings: nothing is
computed with the code under test.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services.cif_citation import (  # noqa: E402
    MAX_CITATION_CHARS, MAX_DOI_CHARS, CitationError, apply_citation, normalise_doi, read_citation,
)


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, doi",
    [
        ("data_x\n_publ_section_doi '10.1107/S0021889811047258'\n", "10.1107/S0021889811047258"),
        ('data_x\n_publ_section_doi "10.1000/abc"\n', "10.1000/abc"),
        # The converter used to demand quotes; the info endpoint never did.
        ("data_x\n_publ_section_doi 10.1000/unquoted\n", "10.1000/unquoted"),
        ("data_x\n_citation_doi 10.1002/b\n", "10.1002/b"),
        ("data_x\n_journal_paper_doi 10.1003/c\n", "10.1003/c"),
        # Only the converter knew this tag.
        ("data_x\n_citation_journal_doi '10.1004/d'\n", "10.1004/d"),
        ("data_x\n_citation_journal_doi 10.1004/d\n", "10.1004/d"),
        # Value on the next line, indented tag, CRLF.
        ("data_x\n  _publ_section_doi\n    10.1000/next\n", "10.1000/next"),
        ("data_x\r\n_publ_section_doi '10.1000/crlf'\r\n", "10.1000/crlf"),
        # A DOI given as a link or with a prefix is the bare DOI.
        ("data_x\n_publ_section_doi 'https://doi.org/10.1000/abc'\n", "10.1000/abc"),
        ("data_x\n_publ_section_doi http://dx.doi.org/10.1000/abc\n", "10.1000/abc"),
        ("data_x\n_publ_section_doi 'doi:10.1000/abc'\n", "10.1000/abc"),
        # A DOI as a text field.
        ("data_x\n_publ_section_doi\n;\n10.1000/field\n;\n", "10.1000/field"),
        # CIF's 'unknown' and 'not applicable' are not a DOI.
        ("data_x\n_publ_section_doi ?\n", ""),
        ("data_x\n_publ_section_doi .\n", ""),
        ("data_x\n_cell_length_a 4.05\n", ""),
    ],
)
def test_doi_reader(text, doi):
    assert read_citation(text).doi == doi


@pytest.mark.parametrize(
    "text",
    [
        # A loop with the column but no row has no value.
        "data_x\nloop_\n_citation_doi\n_citation_year\n",
        # The row is read by column: the DOI column holds no DOI here.
        "data_x\nloop_\n_citation_id\n_citation_doi\nprimary n/a\n",
        # A tag followed by loop_, by another tag, or by a text field that is no DOI.
        "data_x\n_publ_section_doi\nloop_\n_atom_site_label\nAl1\n",
        "data_x\n_publ_section_doi\n_cell_length_a 4.05\n",
        "data_x\n_publ_section_doi\n;\nnot a doi at all\n;\n",
        "data_x\n_publ_section_doi 'n/a'\n",
        "data_x\n_publ_section_doi 10.5\n",
    ],
)
def test_a_value_that_is_not_a_doi_is_never_invented(text):
    assert read_citation(text).doi == ""


@pytest.mark.parametrize(
    "text",
    [
        "data_x\n_x_publ_section_doi 10.1000/prefixed\n",              # another tag that ends in ours
        "data_x\n# _publ_section_doi 10.1000/commented\n",             # a comment
        "data_x\n_publ_section_abstract\n;\nsee _publ_section_doi 10.1000/inside\n_publ_section_doi 10.1000/inside2\n;\n",
        "data_x\n_note 'x _publ_section_doi 10.1000/mid-line'\n",     # not at line start
    ],
)
def test_tags_in_comments_text_fields_and_longer_names_are_ignored(text):
    assert read_citation(text).doi == ""


def test_first_doi_wins_and_duplicates_are_dropped():
    text = "data_x\n_publ_section_doi 10.1001/a\n_citation_doi https://doi.org/10.1001/a\n_citation_doi 10.1002/b\n"
    assert read_citation(text).dois == ["10.1001/a", "10.1002/b"]


def test_reference_text_field_is_flattened_and_not_cut_at_a_semicolon():
    text = (
        "data_x\n_publ_section_references\n;\n"
        "Barlock, J. G. &amp; Mondolfo, L. F. (1975). <i>Z. Metallkd.</i>\n"
        "66; 605-611.\n;\n"
    )
    c = read_citation(text)
    assert c.reference == "Barlock, J. G. & Mondolfo, L. F. (1975). Z. Metallkd. 66; 605-611."
    assert c.doi == ""
    assert c.primary == c.reference


@pytest.mark.parametrize(
    "gap",
    ["", "\n", "   \n", "# a comment line\n", "\n# comment\n\n"],
)
def test_reference_is_found_across_blank_and_comment_lines(gap):
    # Valid CIF: only the ';' has to sit in column 1.
    text = f"data_x\n_publ_section_references\n{gap};\nSome Author (2001) Journal 3, 4.\n;\n"
    assert read_citation(text).reference == "Some Author (2001) Journal 3, 4."


def test_one_line_reference_is_read():
    assert read_citation("data_x\n_publ_section_references 'Doe 2020'\n").reference == "Doe 2020"


def test_doi_is_preferred_over_reference_as_the_primary_value():
    text = "data_x\n_publ_section_doi 10.1009/z\n_publ_section_references\n;\nSome text\n;\n"
    c = read_citation(text)
    assert c.primary == "10.1009/z"
    assert c.reference == "Some text"


def test_nothing_found():
    c = read_citation("data_x\n_cell_length_a 4.05\n")
    assert not c.found and c.primary == ""


@pytest.mark.parametrize(
    "value, bare",
    [
        ("10.1000/abc", "10.1000/abc"),
        (" https://doi.org/10.1000/abc ", "10.1000/abc"),
        ("HTTP://DX.DOI.ORG/10.1000/abc", "10.1000/abc"),
        ("doi: 10.1000/abc", "10.1000/abc"),
        ("10.123456789/x", "10.123456789/x"),
        ("10.12/abc", ""), ("10.1234567890/abc", ""), ("10.1000/", ""), ("10.1000", ""),
        ("abc", ""), ("", ""), (None, ""),
    ],
)
def test_normalise_doi(value, bare):
    assert normalise_doi(value) == bare


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------

def test_apply_citation_round_trips_and_replaces_instead_of_duplicating():
    text = "data_x\n_cell_length_a 4.05\n"
    text, changed = apply_citation(text, doi="10.1001/a", reference="First text")
    assert changed
    text, _ = apply_citation(text, doi="10.1002/b", reference="Second text")
    c = read_citation(text)
    assert c.doi == "10.1002/b" and c.reference == "Second text"
    assert text.count("_publ_section_doi") == 1
    assert text.count("_publ_section_references") == 1


def test_a_doi_link_is_stored_as_the_bare_doi():
    text, _ = apply_citation("data_x\n", doi="https://doi.org/10.1000/abc")
    assert "'10.1000/abc'" in text and "doi.org" not in text


def test_a_value_that_is_not_a_doi_is_refused_not_written():
    with pytest.raises(ValueError):
        apply_citation("data_x\n", doi="not a doi")


def test_apply_citation_replaces_the_doi_it_would_read_under_its_own_tag():
    # Only _citation_doi exists. The old writer appended a second DOI under
    # _publ_section_doi and the reader kept returning the old one first.
    text = "data_x\n_citation_doi 10.1000/old\n_cell_length_a 4.05\n"
    new, _ = apply_citation(text, doi="10.1000/new")
    assert read_citation(new).dois == ["10.1000/new"]
    assert new.count("_publ_section_doi") == 0
    assert new.count("_citation_doi") == 1
    assert "_cell_length_a 4.05" in new


def test_apply_citation_replaces_a_text_field_doi_without_breaking_the_cif():
    text = "data_x\n_publ_section_doi\n;\n10.1000/old\n;\n_cell_length_a 4.05\n"
    new, _ = apply_citation(text, doi="10.1000/new")
    assert read_citation(new).dois == ["10.1000/new"]
    assert ";" not in new                      # no orphaned text-field delimiter
    assert new.count("_publ_section_doi") == 1


def test_apply_citation_replaces_a_placeholder_doi():
    new, _ = apply_citation("data_x\n_publ_section_doi ?\n", doi="10.1000/new")
    assert read_citation(new).dois == ["10.1000/new"]
    assert new.count("_publ_section_doi") == 1


def test_apply_citation_replaces_a_loop_cell_in_place():
    text = "data_x\nloop_\n_citation_id\n_citation_doi\nprimary 10.1000/loopvalue\n"
    new, _ = apply_citation(text, doi="10.1000/new")
    assert new == "data_x\nloop_\n_citation_id\n_citation_doi\nprimary '10.1000/new'\n"
    assert read_citation(new).dois == ["10.1000/new"]


@pytest.mark.parametrize("gap", ["\n", "# comment\n", "   \n\n"])
def test_apply_citation_replaces_a_reference_across_blank_and_comment_lines(gap):
    # The old writer did not find this field and appended a second block.
    text = f"data_x\n_publ_section_references\n{gap};\nOld text\n;\n"
    new, _ = apply_citation(text, reference="New text")
    assert read_citation(new).reference == "New text"
    assert new.count("_publ_section_references") == 1
    assert "Old text" not in new


def test_apply_citation_survives_regex_metacharacters_and_line_breaks():
    # A backslash group in a replacement string used to raise "bad escape".
    text, _ = apply_citation("data_x\n_publ_section_references\n;\nold\n;\n", reference="J. Phys. \\1 (2020)")
    assert read_citation(text).reference == "J. Phys. \\1 (2020)"
    # A line starting with ';' would end a CIF text field early.
    text, _ = apply_citation("data_x\n", reference="line one\n; line two\nline three")
    assert read_citation(text).reference == "line one ; line two line three"


def test_apply_citation_with_empty_arguments_changes_nothing():
    text, changed = apply_citation("data_x\n", doi=" ", reference="")
    assert not changed and text == "data_x\n"


# ---------------------------------------------------------------------------
# DOI as a column of a loop (ICSD / COD style)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, doi",
    [
        # DOI last column, first of two rows: the 1999 paper, not the 2005 one.
        ("data_x\nloop_\n_citation_id\n_citation_year\n_citation_doi\n"
         "primary 1999 10.1000/first\nsecondary 2005 10.1000/second\n", "10.1000/first"),
        # DOI first column.
        ("data_x\nloop_\n_citation_doi\n_citation_year\n10.1000/first 1999\n10.1000/second 2005\n", "10.1000/first"),
        # Quoted cells with spaces and a ';' text field before the DOI column.
        ("data_x\nloop_\n_citation_author\n_citation_title\n_citation_doi\n"
         "'Smith, J. and Doe, A.'\n;\nA long title; with a semicolon\n;\n10.1000/after\n", "10.1000/after"),
        # The loop ends at the next tag; the data after it is not read as a row.
        ("data_x\nloop_\n_citation_id\n_citation_doi\n_cell_length_a 4\n", ""),
        # Windows line ends, comment lines inside the header.
        ("data_x\r\nloop_\r\n_citation_id\r\n# the DOI follows\r\n_citation_doi\r\nprimary 10.1000/crlf\r\n", "10.1000/crlf"),
    ],
)
def test_the_first_row_of_a_doi_column_is_read(text, doi):
    assert read_citation(text).doi == doi


def test_a_scalar_after_a_loop_is_still_read():
    text = "data_x\nloop_\n_atom_site_label\nAl1\n_publ_section_doi 10.1000/scalar\n"
    assert read_citation(text).dois == ["10.1000/scalar"]


def test_reading_thousands_of_tags_takes_linear_time():
    import time

    block = "loop_\n_citation_id\n_citation_doi\nprimary 10.1000/x\n_publ_section_doi 10.1000/y\n"
    text = "data_x\n" + block * 6000                        # ~ 12000 DOI tags, 400 KB
    t0 = time.perf_counter()
    citation = read_citation(text)
    assert time.perf_counter() - t0 < 1.0                  # the quadratic scan took several seconds
    assert citation.dois == ["10.1000/x", "10.1000/y"]


def test_a_reference_that_starts_with_a_semicolon_keeps_the_text_field_closed():
    text, _ = apply_citation("data_x\n_cell_length_a 4\n", reference="; Smith (2020) J. Test 1, 2")
    assert read_citation(text).reference == "; Smith (2020) J. Test 1, 2"
    assert text.count("\n;") == 2                          # exactly the opening and the closing delimiter
    assert "_cell_length_a 4" in text


def test_the_text_limit_is_far_below_the_sht_header_field():
    # doiLen of an SHT header is a signed 16-bit integer: 32767 bytes. Even at
    # four bytes per character the limit stays below it.
    assert MAX_CITATION_CHARS == 2000
    assert MAX_CITATION_CHARS * 4 < 32767


def test_a_reference_at_the_limit_is_accepted_and_one_over_is_refused():
    ok, _ = apply_citation("data_x\n", reference="x" * MAX_CITATION_CHARS)
    assert len(read_citation(ok).reference) == MAX_CITATION_CHARS
    with pytest.raises(CitationError) as err:
        apply_citation("data_x\n", reference="x" * (MAX_CITATION_CHARS + 1))
    assert err.value.code == "too_long"


def test_the_limit_counts_the_text_as_it_is_written_not_its_line_breaks():
    # 1500 words separated by a line break each are 1500 + 1499 characters once
    # the breaks are collapsed to single spaces.
    words = "\n".join(["ab"] * 1000)                      # 3000 raw characters, 2999 collapsed
    with pytest.raises(CitationError):
        apply_citation("data_x\n", reference=words)


def test_an_invalid_doi_carries_its_code():
    with pytest.raises(CitationError) as err:
        apply_citation("data_x\n", doi="n/a")
    assert err.value.code == "invalid_doi"


# ---------------------------------------------------------------------------
# Published structures that carry their bibliography as fields (raw COD / ICSD)
# ---------------------------------------------------------------------------

COD_FIELDS = (
    "loop_\n_publ_author_name\n'Panday P.K.'\n'Schubert K.'\n"
    "_journal_name_full 'Journal of the Less-Common Metals'\n"
    "_journal_page_first 395\n_journal_page_last 401\n_journal_volume 18\n_journal_year 1969\n"
)


def test_a_cod_style_file_has_a_reference_composed_from_its_journal_fields():
    c = read_citation("data_x\n" + COD_FIELDS)
    assert c.reference == "Panday P.K., Schubert K.: Journal of the Less-Common Metals 18 (1969) 395-401"
    assert c.composed is True and c.found is True and c.doi == ""
    assert c.primary == c.reference                         # what a .xtal would carry


@pytest.mark.parametrize(
    "fields, expected",
    [
        # scalar author, abbreviated journal name, single page
        ("_publ_author_name 'Smith J.'\n_journal_name_abbrev 'J. Appl. Cryst.'\n_journal_volume 44\n"
         "_journal_year 2011\n_journal_page_first 1272\n", "Smith J.: J. Appl. Cryst. 44 (2011) 1272"),
        # no authors
        ("_journal_name_full 'Acta Cryst. B'\n_journal_year 1995\n", "Acta Cryst. B (1995)"),
        # the year alone is a journal field
        ("_journal_year 1999\n", "(1999)"),
        # more than six authors
        ("loop_\n_publ_author_name\nA\nB\nC\nD\nE\nF\nG\n_journal_year 2001\n", "A, B, C, D, E, F et al.: (2001)"),
        # placeholders are not fields
        ("_journal_name_full ?\n_journal_year .\n", ""),
        # authors alone are not a citation
        ("loop_\n_publ_author_name\n'Smith J.'\n", ""),
        ("_cell_length_a 4.05\n", ""),
    ],
)
def test_journal_fields_to_a_reference(fields, expected):
    assert read_citation("data_x\n" + fields).reference == expected


def test_the_files_own_reference_text_wins_over_the_journal_fields():
    text = "data_x\n" + COD_FIELDS + "_publ_section_references\n;\nOwn text (2001).\n;\n"
    c = read_citation(text)
    assert c.reference == "Own text (2001)." and c.composed is False


def test_journal_tags_inside_text_fields_and_comments_are_not_fields():
    text = "data_x\n# _journal_year 1999\n_publ_section_abstract\n;\n_journal_year 2000\n;\n"
    assert read_citation(text).reference == ""


def test_text_read_from_a_file_is_capped_like_typed_text():
    long_text = "word " * 1000                                  # 5000 characters
    c = read_citation(f"data_x\n_publ_section_references\n;\n{long_text}\n;\n")
    assert len(c.reference) == MAX_CITATION_CHARS
    journal = read_citation("data_x\n_journal_name_full '" + "J" * 3000 + "'\n")
    assert len(journal.reference) == MAX_CITATION_CHARS


def test_a_doi_over_the_limit_is_neither_read_nor_written():
    long_doi = "10.1000/" + "x" * MAX_DOI_CHARS
    assert read_citation(f"data_x\n_publ_section_doi '{long_doi}'\n").doi == ""
    assert normalise_doi("10.1000/" + "x" * (MAX_DOI_CHARS - 8)) != ""      # exactly at the limit
    with pytest.raises(CitationError) as err:
        apply_citation("data_x\n", doi=long_doi)
    assert err.value.code == "too_long"


# ---------------------------------------------------------------------------
# TeX / IUCr escapes in names and titles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (r"\<Cern\'y, Radovan", "Černý, Radovan"),                   # the caron on the C, acute on the y
        (r"Universit\'e de Gen\`eve", "Université de Genève"),
        (r"M\"uller", "Müller"),
        (r"Fran\c{c}ois", "François"),
        (r"{\'e}cole", "école"),
        (r"\v{S}ebesta and \v Z\'ak", "Šebesta and Žák"),
        (r"Stra\ss e, \o{}re, \AA ngstr\"om", "Straße, øre, Ångström"),
        ("plain text, 100 % unchanged", "plain text, 100 % unchanged"),
        (r"Sm\'{\i}th and Ren\'{e}", "Smíth and René"),             # dotless i under an accent
        (r"M\"{\i}ller", "Mïller"),
        (r"a \< b", r"a \< b"),                                              # a space ends the accent: not one
        (r"\vector and \c{c}", "\\vector and ç"),    # \v needs a space or a brace
    ],
)
def test_tex_escapes_become_the_letters_they_stand_for(raw, expected):
    from backend.api.services.cif_citation import _clean_text

    assert _clean_text(raw) == expected


def test_a_tag_is_still_stripped_and_an_html_entity_still_unescaped():
    from backend.api.services.cif_citation import _clean_text

    assert _clean_text("<i>Acta</i> Cryst. M&uuml;ller \\'e") == "Acta Cryst. Müller é"


def test_the_composed_citation_of_a_file_with_tex_escapes_is_plain_text():
    # The shape of Fe3Al2Si4.cif: author names and the journal in TeX-escaped text.
    cif = (
        "data_x\n"
        "loop_\n_publ_author_name\n'\\<Cern\\'y, Radovan'\n"
        "_journal_name_full 'Acta Cryst. C'\n_journal_volume 60\n_journal_year 2004\n_journal_page_first 12\n"
    )
    ref = read_citation(cif).reference
    assert "\\" not in ref
    assert ref.startswith("Černý")


def test_a_caron_escape_is_not_taken_for_the_start_of_a_tag():
    from backend.api.services.cif_citation import _clean_text

    # Read as a tag, "<Cern\'y <i>" would be deleted up to the first ">".
    assert _clean_text(r"\<Cern\'y, Radovan <i>et al.</i>") == "\u010cern\u00fd, Radovan et al."
