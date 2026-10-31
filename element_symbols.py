"""The one table of element symbols by atomic number (Z = 1 ... 118).

Every place that turns a Z from a ``.xtal`` into a symbol -- the SHT file name
the EMsoft automation and the Orienta Engine write, the phase-library formula,
the material folder, the 3D viewer's tooltip -- reads this table. They used to
carry their own, and three of them were short: the SHT name of a selenide came
out as ``Z34``, of a bismuth phase as ``Z83``, and Mg as ``Z12`` (see
``tests/test_element_symbols.py``, which also fails if a module defines another
copy).

Plain data, no imports, so the standalone EMsoft automation script can load it
too.
"""
from __future__ import annotations

#: Index ``z - 1`` is the symbol of atomic number ``z``.
ATOMIC_SYMBOLS: tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

#: ``{Z: symbol}`` for Z = 1 ... 118.
SYMBOL_OF: dict[int, str] = {z: s for z, s in enumerate(ATOMIC_SYMBOLS, start=1)}

assert len(ATOMIC_SYMBOLS) == 118 and len(set(ATOMIC_SYMBOLS)) == 118


def symbol_of(z, default: str | None = None) -> str:
    """Symbol for atomic number ``z`` (anything ``int()`` accepts).

    An unknown Z returns ``default`` when given, else ``"Z<n>"`` -- the visible
    placeholder the file names have always used, now reachable only for a number
    that is not an element.
    """
    n = int(z)
    sym = SYMBOL_OF.get(n)
    if sym is not None:
        return sym
    return default if default is not None else f"Z{n}"
