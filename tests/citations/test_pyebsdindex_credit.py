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


def test_pseudosymmetry_step_does_not_claim_pyebsdindex():
    """Render-based arbitration runs no Hough (the manual /pseudosym/unify
    route records this step too), so it must not cite PyEBSDIndex."""
    step = STEP_REGISTRY["pseudosym.resolver"]
    assert "pyebsdindex" not in step.citation_ids
    out = render_methods([{"key": "pseudosym.resolver", "params": {}}])
    assert "PyEBSDIndex" not in out


def test_hough_anchor_step_cites_pyebsdindex_and_says_what_it_did():
    step = STEP_REGISTRY["indexing.hough_anchor"]
    assert step.citation_ids == ("pyebsdindex",)
    out = render_methods([{"key": "indexing.hough_anchor", "params": {}}])
    assert "anchor orientations" in out
    assert "Hough indexing with PyEBSDIndex (Rowenhorst et al., 2024)" in out


class _Result:
    metadata = None


def _keys(result):
    from backend.api.services.citations.provenance import get_steps
    return [s["key"] for s in get_steps(result)]


def test_anchor_step_is_recorded_when_the_hough_anchor_ran():
    import numpy as np
    from indexing_controller import _record_hough_anchor_if_ran

    r = _Result()
    _record_hough_anchor_if_ran(r, np.zeros((3, 3)))
    assert _keys(r) == ["indexing.hough_anchor"]


def test_anchor_step_is_absent_when_no_phase_was_resolved():
    from indexing_controller import _record_hough_anchor_if_ran

    r = _Result()
    _record_hough_anchor_if_ran(r, None)
    assert _keys(r) == []


def _call_site_text(src: str) -> str:
    start = src.index("def spherical_gpu_index_patterns")
    return src[start:src.index("\ndef ", start + 1)]


def test_spherical_run_hands_the_resolver_output_to_the_recorder():
    """The call site passes what resolve_eulers_multiphase produced, not a
    flag that is set merely because the resolver was enabled."""
    src = (ROOT / "indexing_controller.py").read_text(encoding="utf-8")
    body = _call_site_text(src)
    assert "_record_hough_anchor_if_ran(indexing_result, _resolved_eulers)" in body
    assert "_resolved_eulers = _res_eul" in body


def test_only_the_spherical_run_records_the_anchor_step():
    """The manual /pseudosym/unify route (and every other recorder) runs no
    Hough indexer, so none of them may record the anchor step."""
    names = ("indexing_controller.py", "backend/api/routes/indexing.py")
    sources = {n: (ROOT / n).read_text(encoding="utf-8") for n in names}
    # the helper in indexing_controller passes the key on a separate line
    assert "indexing.hough_anchor" in sources["indexing_controller.py"]
    assert "indexing.hough_anchor" not in sources["backend/api/routes/indexing.py"]


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
