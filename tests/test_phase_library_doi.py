"""A DOI is a source, and this library keeps most of its sources that way.

The first survey of the library reported "DOI 0 of 36" and moved on. It was
an artefact of the reader: the generator looked at `_publ_section_references`
and at nothing else, so a CIF whose provenance is a bare DOI came out as a
phase with no source at all.

Measured on the real 36: TEN carry one -- eight in `_publ_section_doi`, one
in `_journal_paper_doi`, and one sitting alone inside the reference block --
and for SIX of them it is the only source there is. Reading the field takes
"no source" from nine phases down to three.

`reference()` rejecting a bare `http…` block is right; a URL is not a
citation. Throwing the content away with it was not.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'build_phase_library_fixture.py'
FIXTURE = (ROOT / 'frontend' / 'src' / 'components' / 'PhaseLibrary'
           / '__fixtures__' / 'library.json')


@pytest.fixture(scope='module')
def gen():
    spec = importlib.util.spec_from_file_location('_pldoi', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def phases():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))['phases']


# How the ten are actually written in the files, including the half that
# carry the resolver in front of the identifier.
WRITTEN = [
    ("_publ_section_doi 'https://doi.org/10.1063/1.4812323'", '10.1063/1.4812323'),
    ('_publ_section_doi 10.1107/S0108768193013096', '10.1107/S0108768193013096'),
    ("_journal_paper_doi '10.1107/S0108270197015989'", '10.1107/S0108270197015989'),
    ('_journal_paper_DOI 10.17188/1192439', '10.17188/1192439'),
    ('_publ_section_references\n;\nhttps://doi.org/10.1063/1.4812323\nmp-1\n;\n',
     '10.1063/1.4812323'),
]


@pytest.mark.parametrize('text,expected', WRITTEN)
def test_the_spellings_this_library_uses(gen, text, expected):
    assert gen.doi(text) == expected


def test_the_resolver_is_not_part_of_the_identifier(gen):
    """A DOI is the identifier; the card builds its own link. Storing
    `https://doi.org/10...` would produce `https://doi.org/https://doi.org/…`
    the first time someone concatenates."""
    got = gen.doi("_publ_section_doi 'https://doi.org/10.1063/1.4812323'")
    assert not got.startswith('http')
    assert got.startswith('10.')


def test_a_field_that_holds_no_doi_yields_none(gen):
    assert gen.doi('_publ_section_doi ?') is None
    assert gen.doi('_publ_section_doi .') is None
    assert gen.doi('a file with no identifier in it at all') is None
    assert gen.doi('10.x/not-a-registrant') is None


def test_trailing_sentence_punctuation_is_not_part_of_it(gen):
    assert gen.doi('_publ_section_references\n;\nSee 10.1107/S010827.\n;\n') \
        == '10.1107/S010827'


def test_ten_of_thirty_six_carry_one(phases):
    with_doi = [p['key'] for p in phases if p.get('doi')]
    assert len(with_doi) == 10, with_doi


def test_the_six_for_which_it_is_the_only_source(phases):
    only = sorted(p['key'] for p in phases if p.get('doi') and not p.get('reference'))
    assert only == [
        'Al3Fe2Si_mp-1190708_symmetrized',
        'Al6Fe_mp-570001_symmetrized',
        'Fe3Al2Si4',
        'Fe3_Al2_Si3',
        'MnAl6_mp-173_symmetrized',
        'beta-AlFeSi',
    ], only


def test_beta_alfesi_is_the_acceptance_case(phases):
    """Named in the hand-over from c1: it must show a DOI and never "no
    source". It is also the label/structure case from §2.6.2, so it is the
    phase most likely to be looked at closely."""
    beta = next(p for p in phases if p['key'] == 'beta-AlFeSi')
    assert beta['doi'] == '10.1107/S0108768193013096'
    assert beta['reference'] is None      # which is why the DOI matters here


def test_only_three_phases_have_no_source_at_all(phases):
    """Was nine before the field was read."""
    none = sorted(p['key'] for p in phases
                  if not p.get('reference') and not p.get('doi'))
    assert none == ['Al2Cu_mp-985806_conventional_standard',
                    'Al7FeCu2',
                    'Mg17Al12_mp-2151_conventional_standard'], none


def test_every_stored_doi_is_bare(phases):
    for p in phases:
        if p.get('doi'):
            assert p['doi'].startswith('10.'), (p['key'], p['doi'])
            assert ' ' not in p['doi'], (p['key'], p['doi'])
