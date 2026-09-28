"""A file stem is not a formula.

The phase library derives two element sets per phase: what the STRUCTURE
contains, and what the LABEL claims. The label side falls back to the file
stem when a phase has neither an `_sm_phase_labels` entry nor an .sht -- and
reading a stem as a formula invents elements:

    Al2Zn_MP-mp-aaacqrcb          gave  Al, M, P, Zn
    T-phase_Mg32(AlZn)49_Bergman  gives Be, out of an author's surname

Each phantom is a phantom element facet, and it marks a phase as "label and
structure disagree" when they do not. Measured on the real library: the
first one was the only phantom that reached the fixture, and it made three
disagreements out of the two that are real.

This pins the rule, because the backend endpoint (spec §3) has to derive the
same two sets, and the phantoms are invisible in the output -- M and P are
perfectly good elements.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'build_phase_library_fixture.py'


@pytest.fixture(scope='module')
def gen():
    """Import the generator without running it (it reads Database/)."""
    spec = importlib.util.spec_from_file_location('_plfix', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Every stem in the library that carries something other than chemistry,
# with the elements it must yield -- and only those.
STEMS = [
    ('Al2Zn_MP-mp-aaacqrcb', ['Al', 'Zn']),            # the Materials Project tag
    ('Al3Fe2Si_mp-1190708_symmetrized', ['Al', 'Fe', 'Si']),
    ('alpha-AlFeMnSi_ICSD-52623', ['Al', 'Fe', 'Mn', 'Si']),
    ('MgZn2_sd_0261233', ['Mg', 'Zn']),
    ('beta-AlFeSi', ['Al', 'Fe', 'Si']),               # named Si, structure has none
    ('Al4Fe1.7Si (τ11)', ['Al', 'Fe', 'Si']),
    ('Al13Fe4', ['Al', 'Fe']),
    ('Al7FeCu2', ['Al', 'Cu', 'Fe']),
]


@pytest.mark.parametrize('stem,expected', STEMS)
def test_a_stem_yields_its_chemistry_and_nothing_else(gen, stem, expected):
    assert gen.elements_from_formula(stem, from_filename=True) == expected


def test_the_two_phantoms_by_name(gen):
    """Named, because both were found by reading output rather than code."""
    got = gen.elements_from_formula('Al2Zn_MP-mp-aaacqrcb', from_filename=True)
    assert 'M' not in got and 'P' not in got
    got = gen.elements_from_formula('T-phase_Mg32(AlZn)49_Bergman',
                                    from_filename=True)
    assert 'Be' not in got, 'beryllium out of the surname Bergman'


def test_a_real_formula_is_still_read_whole(gen):
    """The stem rule must not reach strings that ARE formulae. An
    `_sm_phase_labels` value or an .sht formula is parsed as written."""
    assert gen.elements_from_formula('Fe23Al81Si15') == ['Al', 'Fe', 'Si']
    assert gen.elements_from_formula('Mg32(Al Zn)49') == ['Al', 'Mg', 'Zn']
    # An underscore in a real label must NOT truncate it.
    assert gen.elements_from_formula('Al_Fe_Si') == ['Al', 'Fe', 'Si']


def test_T_is_a_phase_name_not_tellurium(gen):
    assert gen.elements_from_formula('T-phase') == []
