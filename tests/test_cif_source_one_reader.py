"""One reader for "where does this CIF come from".

The phase library card, the CIF info panel and the
CIF -> .xtal converter each read a CIF's citation with their own regexes, and
gave different answers for the same file. Measured on the 49 library CIFs
before the fix, the info panel disagreed with the card on 25:

* it cut the citation at the first ``;`` -- "Romming, C." for
  "Romming, C.; Hansen, V.; Gjonnes, J. Crystal structure of beta-Al4.5FeSi. ...";
* it left HTML tags in ("<i>The constitution and structure of ...</i>");
* it missed a DOI that only the references text states (the Zenodo files);
* the converter accepted a DOI tag only when quoted.

All expectations are hand-written from the CIF texts below.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("pymatgen")
pytest.importorskip("orix")

from backend.api.routes import database as db_route  # noqa: E402

_CELL = """_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
_cell_length_a 4.05
_cell_length_b 4.05
_cell_length_c 4.05
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al1 Al 0 0 0
"""

_CITATION = ("Romming, C.; Hansen, V.; Gjonnes, J. Crystal structure of "
             "beta-Al4.5FeSi. Acta Crystallographica B50 (1994) 307-314.")

#: Three authors separated by ';' -- the old reader stopped at the first one.
CIF_SEMICOLONS = f"data_a\n{_CELL}_publ_section_references\n;\n{_CITATION}\n;\n"

#: SpringerMaterials writes its citation with markup.
CIF_MARKUP = (f"data_a\n{_CELL}_publ_section_references\n;\nYim W.M., Paff R.J.: "
              "<i>Thermal expansion of AlN, sapphire, and silicon</i>. Journal of "
              "Applied Physics <b>45</b> (1974) 1456-1457.\n;\n")

#: The only place this file states its DOI is the citation text.
CIF_DOI_IN_TEXT = (f"data_a\n{_CELL}_publ_section_references\n;\nAnes H W et al., "
                   "Dynamical simulation of EBSD master pattern of Cr2N, Zenodo, "
                   "doi:10.5281/zenodo.20376534, CC-BY 4.0\n;\n")

CIF_TAG_UNQUOTED = f"data_a\n_publ_section_doi 10.1107/S0108768193013096\n{_CELL}"
CIF_TAG_QUOTED = f"data_a\n_publ_section_doi 'https://doi.org/10.1063/1.4812323'\n{_CELL}"

CIF_TAG_AND_CITATION = (f"data_a\n_publ_section_doi '10.1107/S0108768193013096'\n{_CELL}"
                        f"_publ_section_references\n;\n{_CITATION}\n;\n")


@pytest.fixture
def client(monkeypatch, tmp_path):
    (tmp_path / "Database" / "CIF_Library").mkdir(parents=True)
    (tmp_path / "Database" / "XTAL_Library").mkdir(parents=True)
    calc = tmp_path / "crystal-structures-for-ebsd-main" / "calculationxtal"
    calc.mkdir(parents=True)
    shutil.copy(Path(__file__).resolve().parents[1] / "crystal-structures-for-ebsd-main"
                / "calculationxtal" / "DWF.xlsx", calc / "DWF.xlsx")
    monkeypatch.setattr(db_route, "_project_root", lambda: tmp_path)
    app = FastAPI()
    app.include_router(db_route.router, prefix="/api/database")
    return TestClient(app), tmp_path


def _info(client, name, text):
    c, root = client
    (root / "Database" / "CIF_Library" / name).write_text(text, encoding="utf-8")
    r = c.get(f"/api/database/cif/{name}/info")
    assert r.status_code == 200, r.text
    return r.json()


def _use_reference(client, name, text):
    c, root = client
    (root / "Database" / "CIF_Library" / name).write_text(text, encoding="utf-8")
    r = c.post("/api/database/convert-cif-to-xtal", json={"cif_path": name})
    assert r.status_code == 200, r.text
    with h5py.File(root / "Database" / "XTAL_Library" / name.replace(".cif", ".xtal"), "r") as f:
        raw = f["CrystalData/useReference"][()][0]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


# --- the info panel ------------------------------------------------------------

def test_a_citation_is_not_cut_at_its_first_semicolon(client):
    assert _info(client, "a.cif", CIF_SEMICOLONS)["reference"] == _CITATION


def test_markup_is_not_shown_in_the_citation(client):
    ref = _info(client, "a.cif", CIF_MARKUP)["reference"]
    assert ref == ("Yim W.M., Paff R.J.: Thermal expansion of AlN, sapphire, and "
                   "silicon. Journal of Applied Physics 45 (1974) 1456-1457.")


def test_a_doi_stated_only_in_the_citation_text_is_found(client):
    info = _info(client, "a.cif", CIF_DOI_IN_TEXT)
    assert info["doi"] == "10.5281/zenodo.20376534"


def test_the_doi_is_the_bare_doi_whether_quoted_or_a_link(client):
    assert _info(client, "a.cif", CIF_TAG_UNQUOTED)["doi"] == "10.1107/S0108768193013096"
    assert _info(client, "b.cif", CIF_TAG_QUOTED)["doi"] == "10.1063/1.4812323"


def test_the_batch_endpoint_answers_like_the_single_one(client):
    c, root = client
    for name, text in (("a.cif", CIF_SEMICOLONS), ("b.cif", CIF_MARKUP),
                       ("c.cif", CIF_DOI_IN_TEXT), ("d.cif", CIF_TAG_AND_CITATION)):
        (root / "Database" / "CIF_Library" / name).write_text(text, encoding="utf-8")
    r = c.post("/api/database/cif/batch-info",
               json={"filenames": ["a.cif", "b.cif", "c.cif", "d.cif"]})
    assert r.status_code == 200, r.text
    for res in r.json()["results"]:
        single = c.get(f"/api/database/cif/{res['filename']}/info").json()
        assert (res["doi"], res["reference"]) == (single["doi"], single["reference"]), res["filename"]
    assert r.json()["results"][0]["reference"] == _CITATION


# --- the converter ----------------------------------------------------------------

def test_the_xtal_carries_the_whole_citation_when_the_cif_has_no_doi_tag(client):
    assert _use_reference(client, "a.cif", CIF_SEMICOLONS) == _CITATION


def test_an_explicit_doi_tag_still_wins_in_the_xtal(client):
    assert _use_reference(client, "d.cif", CIF_TAG_AND_CITATION) == "10.1107/S0108768193013096"


def test_an_unquoted_doi_tag_is_found_by_the_converter(client):
    assert _use_reference(client, "e.cif", CIF_TAG_UNQUOTED) == "10.1107/S0108768193013096"


def test_the_xtal_says_what_the_info_panel_says(client):
    """The .xtal's reference and the panel's reference are one answer."""
    info = _info(client, "f.cif", CIF_MARKUP)
    assert _use_reference(client, "f.cif", CIF_MARKUP) == info["reference"]


# --- the panel is editable: what the user types comes back ----------------------------
# The panel got the CARD's filtered citation, so "Smith 2020"
# (too short for a card) or a bare link (moved to `url`) read back as "click to add"
# although the text was in the file, and saving an edited row could drop a link.

_LINK = "https://example.org/papers/42"
_LONG_WITH_LINK = ("Doe J., Roe R.: A long enough title about aluminium alloys. "
                   "Acta Crystallographica 1 (2020) 1-2. " + "https://doi.org/10.1107/S0108768193013096")


def _patch(client, name, **body):
    c, _root = client
    r = c.patch(f"/api/database/cif/{name}/meta", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _file(client, name):
    c, root = client
    return (root / "Database" / "CIF_Library" / name).read_text(encoding="utf-8")


def test_a_short_note_comes_back_as_typed(client):
    from backend.api.services.phase_library import read_reference
    _info(client, "n.cif", f"data_a\n{_CELL}")
    _patch(client, "n.cif", reference="Smith 2020")
    assert _info(client, "n.cif", _file(client, "n.cif"))["reference"] == "Smith 2020"
    # ... while the card still refuses it, from the same cleaning step.
    assert read_reference(_file(client, "n.cif")) == (None, "too_short")


def test_a_bare_link_comes_back_as_the_reference_and_as_the_link(client):
    from backend.api.services.phase_library import read_reference
    _info(client, "l.cif", f"data_a\n{_CELL}")
    _patch(client, "l.cif", reference=_LINK)
    info = _info(client, "l.cif", _file(client, "l.cif"))
    assert info["reference"] == _LINK and info["url"] == _LINK
    assert read_reference(_file(client, "l.cif")) == (None, "bare_url")


def test_a_reference_with_a_link_keeps_the_link_in_the_panel_and_loses_it_on_the_card(client):
    from backend.api.services.phase_library import read_reference
    _info(client, "w.cif", f"data_a\n{_CELL}")
    _patch(client, "w.cif", reference=_LONG_WITH_LINK)
    info = _info(client, "w.cif", _file(client, "w.cif"))
    assert info["reference"] == _LONG_WITH_LINK
    assert info["doi"] == "10.1107/S0108768193013096"
    card, why = read_reference(_file(client, "w.cif"))
    assert why is None and "https://" not in card and card.startswith("Doe J., Roe R.")


@pytest.mark.parametrize("typed", ["Smith 2020", _LINK, _LONG_WITH_LINK])
def test_saving_what_the_panel_shows_changes_nothing(client, typed):
    """The frontend sends back the row it displays; that must be a no-op."""
    _info(client, "s.cif", f"data_a\n{_CELL}")
    _patch(client, "s.cif", reference=typed)
    once = _file(client, "s.cif")
    shown = _info(client, "s.cif", once)["reference"]
    _patch(client, "s.cif", reference=shown)
    assert _file(client, "s.cif") == once
    assert _info(client, "s.cif", once)["reference"] == typed


def test_replacing_a_citation_that_contains_semicolons_leaves_no_remnant(client):
    """The writer's `;.*?;` ended at the first ';' inside the old citation."""
    _info(client, "r.cif", CIF_SEMICOLONS)
    new = "Lee, K.; Park, J. Another study of aluminium alloys. J. Alloys 5 (2001) 9-10."
    _patch(client, "r.cif", reference=new)
    text = _file(client, "r.cif")
    assert "Gjonnes" not in text and "Hansen" not in text
    assert _info(client, "r.cif", text)["reference"] == new
    assert text.count("_publ_section_references") == 1


def test_a_backslash_in_the_typed_text_survives(client):
    _info(client, "b.cif", CIF_SEMICOLONS)
    typed = r"Müller, H.: The \alpha phase and C:\data notes. J. 1 (1999) 1-9."
    _patch(client, "b.cif", reference=typed, doi=r"10.1000/a\1")
    info = _info(client, "b.cif", _file(client, "b.cif"))
    assert info["reference"] == typed


# --- DOI: the writer updates the tag the reader reads ----------------------------------

_TWO_DOI_TAGS = (f"data_a\n_journal_paper_doi '10.1107/S0108270197015989'\n"
                 f"_publ_section_doi '10.9999/old'\n{_CELL}")


def test_a_doi_edit_on_a_file_with_two_doi_tags_is_not_ignored(client):
    """alpha-(AlMnSi).cif has both; the reader prefers `_journal_paper_doi`, the
    writer used to write only `_publ_section_doi`."""
    assert _info(client, "d.cif", _TWO_DOI_TAGS)["doi"] == "10.1107/S0108270197015989"
    _patch(client, "d.cif", doi="10.1234/new")
    text = _file(client, "d.cif")
    assert _info(client, "d.cif", text)["doi"] == "10.1234/new"
    assert "10.9999/old" not in text and "10.1107/S0108270197015989" not in text


def test_a_doi_edit_is_written_when_the_file_has_no_doi_tag(client):
    _info(client, "e.cif", f"data_a\n{_CELL}")
    _patch(client, "e.cif", doi="10.1234/fresh")
    assert _info(client, "e.cif", _file(client, "e.cif"))["doi"] == "10.1234/fresh"


# --- the database builder writes the same answer into crystal_database.xlsx -------------
# cif_database_builder.parse_cif_file still carried the old
# regexes; on 32 of the 49 library CIFs its DOI / "Reference Text" differed from
# the card and the panel.

def _builder():
    pytest.importorskip("spglib")
    pytest.importorskip("tqdm")
    pytest.importorskip("h5py")
    import importlib.util
    d = Path(__file__).resolve().parents[1] / "crystal-structures-for-ebsd-main" / "calculationxtal"
    spec = importlib.util.spec_from_file_location("cif_database_builder_under_test",
                                                  d / "cif_database_builder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name, text", [
    ("sem.cif", CIF_SEMICOLONS), ("markup.cif", CIF_MARKUP),
    ("intext.cif", CIF_DOI_IN_TEXT), ("both.cif", CIF_TAG_AND_CITATION),
    ("two.cif", _TWO_DOI_TAGS), ("none.cif", f"data_a\n{_CELL}"),
])
def test_the_builder_row_says_what_the_panel_says(client, name, text):
    cdb = _builder()
    info = _info(client, name, text)
    _c, root = client
    row = cdb.parse_cif_file(root / "Database" / "CIF_Library" / name)
    assert row is not None
    assert row["DOI"] == (info["doi"] or "none found")
    assert row["Reference Text"] == (info["reference"] or info["url"] or "none found")


def test_the_builder_keeps_the_whole_citation_and_the_doi_together(client):
    """Old: DOI present -> the Reference Text column was "none found"; and the
    citation was cut at its first ';'."""
    cdb = _builder()
    _info(client, "x.cif", CIF_TAG_AND_CITATION)
    _c, root = client
    row = cdb.parse_cif_file(root / "Database" / "CIF_Library" / "x.cif")
    assert row["DOI"] == "10.1107/S0108768193013096"
    assert row["Reference Text"] == _CITATION


def test_the_builder_does_not_skip_every_file_when_the_shared_reader_is_missing(client, monkeypatch):
    cdb = _builder()
    _info(client, "y.cif", CIF_SEMICOLONS)
    _c, root = client

    def _gone(raw):
        raise ImportError("backend.api.services.phase_library is not here")

    monkeypatch.setattr(cdb, "_cif_source", _gone)
    with pytest.raises(ImportError):
        cdb.parse_cif_file(root / "Database" / "CIF_Library" / "y.cif")


def test_the_real_alpha_almnsi_file_round_trips_a_doi_edit(client):
    """The same round trip on the library file itself (skipped without it)."""
    src = (Path(__file__).resolve().parents[1] / "Database" / "CIF_Library" / "α-(AlMnSi).cif")
    if not src.is_file():
        pytest.skip("library CIF not on this machine")
    _info(client, "alpha.cif", src.read_text(encoding="utf-8", errors="ignore"))
    _patch(client, "alpha.cif", doi="10.1234/edited")
    assert _info(client, "alpha.cif", _file(client, "alpha.cif"))["doi"] == "10.1234/edited"


# --- the one reader stands on cif_citation: what it adds reaches every consumer ---------

_COD_FIELDS = (f"data_a\n_publ_author_name 'Smith, J.'\n_journal_name_full 'Acta Cryst.'\n"
               f"_journal_year 1999\n_journal_volume 12\n_journal_page_first 34\n{_CELL}")

_TEX_CITATION = r"Cern\'y, R. Studies of Universit\'e alloys. J. Test 1 (2000) 1-9."
_TEX_PLAIN = "Cerný, R. Studies of Université alloys. J. Test 1 (2000) 1-9."


def test_a_file_with_only_journal_fields_is_cited_in_the_panel_and_in_the_xtal(client):
    info = _info(client, "cod.cif", _COD_FIELDS)
    assert info["reference"] == "Smith, J.: Acta Cryst. 12 (1999) 34"
    assert info["reference_missing"] is False
    assert _use_reference(client, "cod2.cif", _COD_FIELDS) == info["reference"]


def test_a_doi_column_of_a_loop_reaches_panel_and_card(client):
    from backend.api.services.phase_library import read_doi
    text = f"data_a\nloop_\n_citation_id\n_citation_doi\n1 10.1234/looped\n{_CELL}"
    assert _info(client, "loop.cif", text)["doi"] == read_doi(text) == "10.1234/looped"


def test_panel_and_card_answer_alike_when_the_later_doi_tag_differs(client):
    from backend.api.services.phase_library import read_doi
    text = (f"data_a\n_publ_section_doi '10.1234/first'\n"
            f"_journal_paper_doi '10.1234/second'\n{_CELL}")
    assert _info(client, "two.cif", text)["doi"] == read_doi(text) == "10.1234/first"


def test_panel_and_card_clean_a_citation_with_tex_accents_alike(client):
    from backend.api.services.phase_library import read_reference
    text = f"data_a\n{_CELL}_publ_section_references\n;\n{_TEX_CITATION}\n;\n"
    assert _info(client, "tex.cif", text)["reference"] == _TEX_PLAIN
    assert read_reference(text) == (_TEX_PLAIN, None)


def test_a_cif_tag_where_the_citation_should_be_is_no_citation(client):
    from backend.api.services.phase_library import read_reference
    text = f"data_a\n{_CELL}_publ_section_references\n;\n_publ_author_address\n;\n"
    info = _info(client, "tag.cif", text)
    assert info["reference"] == "" and info["reference_missing"] is True
    assert read_reference(text) == (None, "cif_tag")


def test_a_closing_semicolon_that_is_indented_still_ends_the_citation(client):
    text = f"data_a\n{_CELL}_publ_section_references\n;\n{_CITATION}\n  ;\n"
    assert _info(client, "indent.cif", text)["reference"] == _CITATION


def test_a_doi_edit_with_no_doi_in_it_is_refused_with_its_code(client):
    c, _root = client
    _info(client, "bad.cif", CIF_SEMICOLONS)
    r = c.patch("/api/database/cif/bad.cif/meta", json={"doi": "not-a-doi"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "invalid_doi"
    assert _file(client, "bad.cif") == CIF_SEMICOLONS


def test_an_edit_replaces_a_citation_with_an_indented_closing_semicolon(client):
    new = "Lee, K. Another study of aluminium alloys. J. Alloys 5 (2001) 9-10."
    _info(client, "i2.cif", f"data_a\n{_CELL}_publ_section_references\n;\n{_CITATION}\n  ;\n")
    _patch(client, "i2.cif", reference=new)
    text = _file(client, "i2.cif")
    assert text.count("_publ_section_references") == 1 and "Gjonnes" not in text
    assert _info(client, "i2.cif", text)["reference"] == new


def test_the_card_shows_the_composed_citation_the_panel_shows(client):
    from backend.api.services.phase_library import read_reference
    assert read_reference(_COD_FIELDS) == ("Smith, J.: Acta Cryst. 12 (1999) 34", None)
    assert _info(client, "cod3.cif", _COD_FIELDS)["reference"] == read_reference(_COD_FIELDS)[0]
