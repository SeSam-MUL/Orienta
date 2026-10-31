"""One table of element symbols for the whole app.

After the "Z12 instead of Mg" fix: ``phase_metadata`` had its
own 55-entry table with gaps (no He, Ne, Ar, Br, Kr, Rb, Sr, Tc, I, Xe, Cs, Ba,
Pr-Lu, Hg, Tl, Bi, Po ...) and it names the SHT file on the Orienta Engine path
("Z83SeZ56Z53"); ``simulation/file_sync_manager.py`` stopped at Cf (98); the
database route's 3D-viewer table at Cm (96). All of them now read
``element_symbols.py``.

Expectations are the periodic table, written by hand.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import element_symbols as es

ROOT = Path(__file__).resolve().parents[1]

#: Z -> symbol, hand-written for the elements that were missing or that mark an end.
HAND = {1: "H", 2: "He", 7: "N", 10: "Ne", 12: "Mg", 18: "Ar", 34: "Se", 35: "Br",
        36: "Kr", 37: "Rb", 38: "Sr", 43: "Tc", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba",
        59: "Pr", 71: "Lu", 80: "Hg", 81: "Tl", 83: "Bi", 84: "Po", 96: "Cm", 98: "Cf",
        99: "Es", 100: "Fm", 118: "Og"}


def test_the_table_is_the_periodic_table():
    assert len(es.ATOMIC_SYMBOLS) == 118
    assert len(set(es.ATOMIC_SYMBOLS)) == 118
    for z, sym in HAND.items():
        assert es.SYMBOL_OF[z] == sym
        assert es.symbol_of(z) == sym


def test_a_number_that_is_not_an_element_keeps_the_visible_placeholder():
    assert es.symbol_of(119) == "Z119"
    assert es.symbol_of(0, default="?") == "?"


def _xtal(path, ztypes, counts=None):
    with h5py.File(path, "w") as f:
        g = f.create_group("CrystalData")
        g.create_dataset("Atomtypes", data=np.array(ztypes, dtype=np.int32))
        g.create_dataset("Natomtypes",
                         data=np.array(counts or [1] * len(ztypes), dtype=np.int32))
    return path


@pytest.mark.parametrize("z", [34, 53, 56, 83])
def test_the_sht_name_formula_of_the_engine_path_has_no_placeholder(tmp_path, z):
    from phase_metadata import resolve_sht_name_fields
    xtal = _xtal(tmp_path / "X.xtal", [z])
    formula, _pearson = resolve_sht_name_fields("X", xtal_path=xtal)
    assert formula == es.SYMBOL_OF[z] == HAND[z]


def test_a_four_element_phase_is_named_by_symbols(tmp_path):
    from phase_metadata import resolve_sht_name_fields
    xtal = _xtal(tmp_path / "Q.xtal", [34, 83, 56, 53])
    assert resolve_sht_name_fields("Q", xtal_path=xtal)[0] == "SeBiBaI"


def test_the_phase_library_formula_counts_with_symbols(tmp_path):
    from phase_metadata import extract_metadata_from_h5
    xtal = _xtal(tmp_path / "BiSe.xtal", [83, 34], counts=[2, 3])
    assert extract_metadata_from_h5(xtal).formula == "Bi2Se3"


def test_the_orienta_engine_sht_writer_names_every_element():
    from backend.forward_sim.io.sht_writer import _formula_from_atoms
    atoms = [SimpleNamespace(Z=z) for z in (83, 34, 56, 53)]
    # mp2sht lists the distinct elements in ascending Z: Se 34, I 53, Ba 56, Bi 83
    assert _formula_from_atoms(SimpleNamespace(atoms=atoms)) == "SeIBaBi"


def test_the_material_folder_lookup_reaches_past_californium():
    pytest.importorskip("PyQt5")
    from simulation import file_sync_manager as fsm
    assert fsm.PERIODIC_TABLE[99] == "Es" and fsm.PERIODIC_TABLE[118] == "Og"
    assert fsm.PERIODIC_TABLE[34] == "Se"


# --- the guard: no module defines its own table --------------------------------------

#: Files that legitimately hold a LIST of element symbols that is not a Z -> symbol
#: table: ``Ai_Ml`` indexes a 92-element feature vector, the shared module is the
#: table itself, and the two legacy scripts are not used by the app.
#: ``phase_metadata`` is NOT here: it is the module that had the short table, so a
#: whole-file pass would let the next one in. Its one deliberately narrow set
#: (formula parsing; the docstring of ``extract_elements`` says why) is allowed by
#: name below.
_ALLOWED = {
    "element_symbols.py",
    "Ai_Ml/ebsd_ai/config.py",
    "crystal-structures-for-ebsd-main/_Phyton_Automization/cluster_automization/emsphinx_automation_cluster.py",
    "crystal-structures-for-ebsd-main/_Phyton_Automization/windwos_to_WSL/script_no server mode.py",
}


def _tracked_app_sources():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    skip = ("tests/", "tasks/", "research/", "docs/", "scripts/")
    return [f for f in out if f and not f.startswith(skip)]


#: ``(file, enclosing function)`` pairs whose literal is allowed: the one narrow
#: formula-parsing set in phase_metadata.
_ALLOWED_IN_FUNCTION = {("phase_metadata.py", "extract_elements")}


def _table_literals(source: str, rel: str):
    """Line numbers of every list/tuple/set/dict literal in ``source`` holding 20 or
    more element symbols, except the allowed ones."""
    symbols = set(es.ATOMIC_SYMBOLS)
    found = []

    def walk(node, func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            vals = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        elif isinstance(node, ast.Dict):
            vals = [v.value for v in node.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        else:
            vals = []
        # exempt only a SET literal in the named function: a Z -> symbol dict or a
        # list there would still be a table
        exempt = (rel, func) in _ALLOWED_IN_FUNCTION and isinstance(node, ast.Set)
        if sum(v in symbols for v in vals) >= 20 and not exempt:
            found.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            walk(child, func)

    walk(ast.parse(source), None)
    return found


def test_no_module_defines_its_own_symbol_table():
    offenders = []
    for rel in _tracked_app_sources():
        if rel in _ALLOWED:
            continue
        try:
            source = (ROOT / rel).read_text(encoding="utf-8")
            lines = _table_literals(source, rel)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        offenders += [f"{rel}:{n}" for n in lines]
    assert not offenders, ("define their own element table instead of importing "
                           "element_symbols: " + ", ".join(offenders))


def test_a_new_table_in_phase_metadata_is_caught():
    """The module that had the bug is not exempt: the real file passes, the real
    file plus one more Z -> symbol dict at module level, or a second literal in the
    allowed function, does not."""
    source = (ROOT / "phase_metadata.py").read_text(encoding="utf-8")
    assert _table_literals(source, "phase_metadata.py") == []
    short = "_NEW = {" + ", ".join(f'{z}: "{s}"' for z, s in list(es.SYMBOL_OF.items())[:30]) + "}\n"
    assert _table_literals(source + "\n" + short, "phase_metadata.py")
    narrow = ("\n\ndef extract_elements(formula):\n    valid = {"
              + ", ".join(f'"{s}"' for s in es.ATOMIC_SYMBOLS[:30]) + "}\n    return valid\n")
    assert _table_literals(narrow, "phase_metadata.py") == []    # the named narrow set
    assert _table_literals(narrow, "other.py")                    # ... and only there
    as_list = narrow.replace("{", "[").replace("}", "]")
    assert _table_literals(as_list, "phase_metadata.py")          # a list there is a table
    as_dict = ("\n\ndef extract_elements(formula):\n    return {"
               + ", ".join(f'{z}: "{s}"' for z, s in list(es.SYMBOL_OF.items())[:30]) + "}\n")
    assert _table_literals(as_dict, "phase_metadata.py")          # so is a dict
