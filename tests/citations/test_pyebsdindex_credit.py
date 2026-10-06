"""PyEBSDIndex is credited wherever the Hough indexing it performs is described.

The Hough/Radon band detection and band-triplet indexing run inside
PyEBSDIndex, so a Hough result, the shipped documents that list the software
stack, and the machine-readable citation metadata must all name it and give its
reference (Rowenhorst, Callahan & Ånes, J. Appl. Cryst. 57, 3-19, 2024).
"""
from pathlib import Path

import yaml

from backend.api.services.citations.render import (
    load_library,
    render_bibtex,
    render_methods,
    render_plain,
)
from backend.api.services.citations.steps import STEP_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
DOI = "10.1107/S1600576723010221"
TITLE = ("Fast Radon transforms for high-precision EBSD orientation "
         "determination using PyEBSDIndex")


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_hough_methods_sentence_names_the_software_and_its_reference():
    out = render_methods([
        {"key": "indexing.hough", "params": {"orienta_version": "0.4.7"}},
    ])
    assert "PyEBSDIndex" in out
    assert "Rowenhorst et al., 2024" in out
    # The detection step is attributed too, not only "indexing".
    assert "Radon" in out
    assert "Orienta 0.4.7" in out


def test_hough_step_cites_pyebsdindex_next_to_kikuchipy():
    ids = STEP_REGISTRY["indexing.hough"].citation_ids
    assert "pyebsdindex" in ids
    assert "kikuchipy" in ids


def test_non_hough_steps_do_not_claim_pyebsdindex():
    """Credit follows what ran: a dictionary or spherical run did not use it."""
    for key in ("indexing.dictionary", "indexing.spherical",
                "indexing.spherical_emsphinx"):
        assert "pyebsdindex" not in STEP_REGISTRY[key].citation_ids


def test_pseudosymmetry_step_credits_the_hough_anchor():
    """The spherical runs that resolve pseudo-symmetry take their anchor
    orientation from PyEBSDIndex, so that step cites it and says so."""
    step = STEP_REGISTRY["pseudosym.resolver"]
    assert "pyebsdindex" in step.citation_ids
    assert "orix" in step.citation_ids
    out = render_methods([{"key": "pseudosym.resolver", "params": {}}])
    assert "Hough indexing with PyEBSDIndex (Rowenhorst et al., 2024)" in out
    assert "anchor" in out


def test_library_entry_is_complete():
    entry = load_library()["pyebsdindex"]
    assert entry["DOI"] == DOI
    assert entry["title"] == TITLE
    assert entry["container-title"] == "Journal of Applied Crystallography"
    assert (entry["volume"], str(entry["issue"]), entry["page"]) == (
        "57", "1", "3-19")
    assert [a["family"] for a in entry["author"]] == [
        "Rowenhorst", "Callahan", "Ånes"]


def test_rendered_bibliography_carries_volume_pages_and_doi():
    entry = load_library()["pyebsdindex"]
    bib = render_bibtex([entry])
    assert "volume = {57}" in bib
    assert "pages = {3-19}" in bib
    assert DOI in bib
    plain = render_plain([entry])
    assert "Journal of Applied Crystallography 57, 3-19" in plain
    assert DOI in plain


def test_citation_cff_lists_the_reference():
    cff = yaml.safe_load(_text("CITATION.cff"))
    refs = cff.get("references") or []
    hits = [r for r in refs if r.get("doi") == DOI]
    assert len(hits) == 1, "CITATION.cff must reference PyEBSDIndex exactly once"
    ref = hits[0]
    assert ref["title"] == TITLE
    assert ref["journal"] == "Journal of Applied Crystallography"
    assert str(ref["volume"]) == "57" and str(ref["issue"]) == "1"
    assert str(ref["start"]) == "3" and str(ref["end"]) == "19"
    assert [a["family-names"] for a in ref["authors"]] == [
        "Rowenhorst", "Callahan", "Ånes"]


def test_readme_cites_pyebsdindex_with_its_reference():
    readme = _text("README.md")
    assert DOI in readme
    assert "PyEBSDIndex" in readme
    # the "Built on ..." paragraph and the architecture diagram both name it
    built_on = next(p for p in readme.split("\n\n") if p.startswith("Built on"))
    assert "PyEBSDIndex" in built_on
    assert "kikuchipy, orix, diffsims, PyEBSDIndex" in readme


def test_notice_gives_the_reference_where_it_attributes_hough():
    notice = _text("NOTICE.md")
    assert DOI in notice
    assert "Rowenhorst" in notice


def test_user_docs_that_list_the_stack_name_pyebsdindex():
    for rel in ("docs/README.md", "docs/user-guide/README.md",
                "docs/ARCHITECTURE.md"):
        head = _text(rel)[:900]
        assert "PyEBSDIndex" in head, rel


def test_architecture_doc_does_not_credit_kikuchipy_with_hough():
    arch = _text("docs/ARCHITECTURE.md")
    assert "Hough/Dictionary via kikuchipy" not in arch
