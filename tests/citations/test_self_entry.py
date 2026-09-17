from pathlib import Path

import pytest

from backend.api.services.citations.self_entry import (
    orienta_entry_from_cff,
    stated_dois,
)

ROOT = Path(__file__).resolve().parents[2]

CFF = """cff-version: 1.2.0
title: "Orienta"
type: software
authors:
  - given-names: "Sebastian"
    family-names: "Samberger"
    orcid: "https://orcid.org/0009-0000-2559-0172"
  - given-names: "Stefan"
    family-names: "Pogatscher"
version: "0.1.0"
date-released: "2026-06-29"
license: "GPL-3.0-or-later"
url: "https://doi.org/10.5281/zenodo.99999999"
doi: "10.5281/zenodo.99999999"
"""


def test_entry_is_csl_shaped(tmp_path):
    f = tmp_path / "CITATION.cff"
    f.write_text(CFF, encoding="utf-8")
    e = orienta_entry_from_cff(f)
    assert e["id"] == "orienta"
    assert e["type"] == "software"
    assert e["title"] == "Orienta"
    assert e["DOI"] == "10.5281/zenodo.99999999"
    assert e["issued"]["date-parts"] == [[2026]]


def test_authors_are_mapped_to_csl_names(tmp_path):
    f = tmp_path / "CITATION.cff"
    f.write_text(CFF, encoding="utf-8")
    e = orienta_entry_from_cff(f)
    assert e["author"][0] == {"family": "Samberger", "given": "Sebastian"}
    assert len(e["author"]) == 2


def test_version_is_not_taken_from_the_cff(tmp_path):
    """The CFF says 0.1.0 while the app ships 0.3.x — app_version owns this."""
    f = tmp_path / "CITATION.cff"
    f.write_text(CFF, encoding="utf-8")
    assert "version" not in orienta_entry_from_cff(f)


def test_missing_file_raises_rather_than_inventing(tmp_path):
    with pytest.raises(FileNotFoundError):
        orienta_entry_from_cff(tmp_path / "nope.cff")


def test_stated_dois_finds_all_of_them():
    assert stated_dois("see 10.5281/zenodo.123 and 10.5281/zenodo.456") == {
        "10.5281/zenodo.123", "10.5281/zenodo.456"}
    assert stated_dois("no doi here") == set()


def test_real_cff_parses_and_carries_a_doi():
    e = orienta_entry_from_cff()
    assert e["id"] == "orienta"
    assert e["DOI"].startswith("10.5281/zenodo.")
    assert e["author"], "CITATION.cff lists no authors"


def test_readme_and_citation_cff_state_the_same_doi():
    """Drift guard. Passes today in BOTH trees; stops them diverging later.

    This is the check that caught a controller error: a DOI read from the
    public repo (beta) compared against the local CITATION.cff (dev).
    """
    readme = stated_dois((ROOT / "README.md").read_text(encoding="utf-8"))
    cff = stated_dois((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert readme, "README.md states no Zenodo DOI"
    assert cff, "CITATION.cff states no Zenodo DOI"
    assert readme & cff, (
        f"README cites {sorted(readme)} and CITATION.cff cites {sorted(cff)}; "
        "they must share a DOI"
    )


def test_the_drift_guard_actually_bites():
    """A guard that has never failed is not known to work."""
    readme = stated_dois("cite 10.5281/zenodo.111")
    cff = stated_dois('doi: "10.5281/zenodo.222"')
    assert not (readme & cff)
