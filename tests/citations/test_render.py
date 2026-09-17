import json
from pathlib import Path

import pytest

from backend.api.services.citations.render import (
    load_library,
    render_bibtex,
    render_plain,
)

FIXTURE = Path(__file__).parent / "fixtures" / "library_fixture.json"


@pytest.fixture
def entries():
    lib = load_library(FIXTURE)
    return [lib["fixture-article"], lib["fixture-software"]]


def test_load_library_keys_by_id():
    lib = load_library(FIXTURE)
    assert set(lib) == {"fixture-software", "fixture-article"}
    assert lib["fixture-article"]["volume"] == "207"


def test_bibtex_article_shape(entries):
    out = render_bibtex([entries[0]])
    assert out.startswith("@article{fixture-article,")
    assert "author = {Roe, Richard and Poe, Paula}" in out
    assert "year = {2019}" in out
    assert "journal = {Journal of Fixtures}" in out
    assert "doi = {10.0000/fixture.article}" in out
    assert out.rstrip().endswith("}")


def test_bibtex_software_uses_misc_with_version(entries):
    out = render_bibtex([entries[1]])
    assert out.startswith("@misc{fixture-software,")
    assert "note = {Version 1.2.3}" in out


def test_bibtex_escapes_braces_and_backslashes():
    entry = {"id": "x", "type": "article-journal",
             "title": "A {weird} title with \ in it",
             "author": [{"family": "Doe", "given": "J."}],
             "issued": {"date-parts": [[2021]]}}
    out = render_bibtex([entry])
    assert "\{weird\}" in out
    assert "\textbackslash{}" in out


def test_plain_is_one_line_per_entry(entries):
    out = render_plain(entries)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "Roe, R., Poe, P. (2019)" in lines[0]
    assert "https://doi.org/10.0000/fixture.article" in lines[0]


def test_plain_survives_missing_optional_fields():
    entry = {"id": "bare", "type": "software", "title": "Bare"}
    out = render_plain([entry])
    assert "Bare" in out
    assert "None" not in out
    assert "n.d." in out


def test_renderers_are_deterministic(entries):
    assert render_bibtex(entries) == render_bibtex(entries)
    assert render_plain(entries) == render_plain(entries)


def test_default_library_merges_orienta_fixture_does_not():
    default_lib = load_library()
    assert "orienta" in default_lib

    fixture_lib = load_library(FIXTURE)
    assert "orienta" not in fixture_lib
