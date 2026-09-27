"""Build a structure from a CIF in the origin choice the CIF was written in.

WHY THIS EXISTS
---------------
``origin_choice.py`` fixed the reading end: a ``.xtal`` that records its origin
choice is expanded correctly. This is the writing end, and it is the one that
matters, because **pymatgen ignores the origin-choice marker entirely**.

Measured here, on a silicon cell with the 8a coordinate (1/8,1/8,1/8):

    _symmetry_space_group_name_H-M    atoms   density
    'Fd-3m'                             16      4.658
    'Fd-3m O2'                          16      4.658
    'Fd-3m :2'                          16      4.658
    'Fd-3m S'                           16      4.658
    'Fd-3m Z'                           16      4.658

All five are origin choice 2 as far as a crystallographer is concerned, and all
five come back expanded with origin-choice-1 operators, onto the 16-fold 16c
site. Silicon at twice its density.

Two consequences, both of which this module acts on:

1. **Stamping the setting into the .xtal cannot repair it.** The coordinates
   pymatgen hands the writer are already a wrong orbit; recording "choice 2"
   next to them only makes the reader shift a wrong answer. The origin choice
   has to be honoured *before* the expansion, which is what
   :func:`structure_from_cif` does.
2. **What the writer then produces is origin choice 1**, always: pymatgen's
   operator set is choice 1 for all 24 two-origin groups (measured against
   spglib). So ``SpaceGroupSetting = 1`` is the truthful field to write, and
   deriving it from the H-M symbol would be a lie about the coordinates.

HOW THE CHOICE IS DECIDED
-------------------------
Not from the symbol — the survey of our own library found that the two files
that are wrong (``Si.cif``, ``sd_1816951.cif``) both say plain ``'Fd-3m'`` with
no marker at all. It is decided from what the block *states about its own
atoms*, in this order, and every source that has an opinion must agree:

* **explicit symmetry operators** — matched against spglib's two settings.
  Definitive when present, and the reason ``Al3Fe2Si_mp-1190708`` needs no
  correction.
* **Wyckoff symbols** — ``8a`` means the orbit has eight atoms in it. Expand
  the site both ways and keep the one that delivers the stated multiplicity.
  This is what catches ``Si.cif`` (8a declared, 16 delivered) and
  ``sd_1816951.cif`` (Mg 8a -> 16, Cu 16d -> 8: the multiplicities *swap*, so
  the atom count stays 24 while the composition inverts to Mg2Cu).
* **the chemical formula sum** — the whole cell's composition, same test.
* **the H-M marker** (``:1``/``:2``, ``O1``/``O2``, trailing ``S``/``Z``),
  last, because it is a statement of intent that nothing checks.

If nothing has an opinion, or two sources contradict each other, this raises.
It does not guess: a guess here is a silently wrong master pattern.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .origin_choice import has_two_origin_choices, to_choice_1


class CifOriginError(ValueError):
    """The origin choice of a CIF could not be established, or is contradicted."""


@dataclass
class BlockVerdict:
    """What one ``data_`` block of a CIF says about itself."""

    key: str
    space_group: int | None = None
    choice: int | None = None
    evidence: list[str] = field(default_factory=list)
    problem: str | None = None

    @property
    def usable(self) -> bool:
        return self.problem is None and self.space_group is not None


@dataclass
class CifOriginReport:
    """Where the structure came from, in a form fit to put in provenance."""

    path: str
    block_key: str
    space_group: int
    cif_origin_choice: int
    shifted: bool
    evidence: list[str]
    other_blocks: list[BlockVerdict] = field(default_factory=list)

    def summary(self) -> str:
        how = "shifted to origin choice 1" if self.shifted else "used as written"
        return (f"{Path(self.path).name} block {self.block_key!r}: space group "
                f"{self.space_group}, CIF origin choice {self.cif_origin_choice}, "
                f"{how} ({'; '.join(self.evidence)})")


# ---------------------------------------------------------------------------
# reading the raw block
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _first(data: dict, *keys: str):
    for k in keys:
        if data.get(k) not in (None, "", "?"):
            return data[k]
    return None


#: What a CIF writes when a value is unknown or inapplicable.
_CIF_NULLS = {"?", ".", ""}


def _number(value: Any, *, field: str = "") -> float | None:
    """CIF numbers carry an uncertainty in brackets: ``7.0309(4)``.

    ``None`` means the file **declined** to give a value (``?`` or ``.``).
    Anything else that will not parse raises, because the alternative is to drop
    an atom and keep going: a MgCu2 cell whose copper coordinate is written as
    the fraction ``1/2`` came back as Mg8 at 0.929 g/cm3 instead of Mg8 Cu16 at
    5.787, with no warning anywhere. Silently losing an atom also silently loses
    the evidence that decides the origin choice.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text in _CIF_NULLS:
        return None
    try:
        return float(text.split("(")[0])
    except ValueError:
        raise CifOriginError(
            f"cannot read {field or 'a numeric field'} from {text!r}. CIF numbers "
            "are decimal (optionally with an uncertainty in brackets); fractions "
            "like '1/2' and comma decimals are not, and guessing at one would "
            "drop the atom it belongs to."
        ) from None


def _space_group_number(data: dict) -> int | None:
    raw = _first(data, "_symmetry_Int_Tables_number", "_space_group_IT_number")
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def _space_group_number_from_structure(path: Path, key: str) -> int | None:
    """Several library CIFs state only the operators, not the group number.

    Those are the files the origin question cannot touch anyway — but they have
    to get past the analysis rather than be refused, so the number is taken from
    the structure itself, exactly as the converter does a few lines later.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        return int(SpacegroupAnalyzer(_parse_block(path, key), symprec=1e-5).get_space_group_number())
    except Exception:
        return None


def _hm_symbol(data: dict) -> str:
    return str(_first(data, "_symmetry_space_group_name_H-M",
                      "_space_group_name_H-M_alt") or "")


def _symop_strings(data: dict) -> list[str]:
    return [str(s) for s in _as_list(_first(data, "_symmetry_equiv_pos_as_xyz",
                                            "_space_group_symop_operation_xyz"))]


def _sites(data: dict) -> list[dict]:
    """The atom_site loop as a list of dicts, tolerant of single-row loops."""
    xs = _as_list(data.get("_atom_site_fract_x"))
    if not xs:
        return []
    ys = _as_list(data.get("_atom_site_fract_y"))
    zs = _as_list(data.get("_atom_site_fract_z"))
    labels = _as_list(data.get("_atom_site_label")) or [""] * len(xs)
    symbols = _as_list(data.get("_atom_site_type_symbol")) or [""] * len(xs)
    wyckoff = _as_list(_first(data, "_atom_site_Wyckoff_symbol",
                              "_atom_site_Wyckoff_label")) or [""] * len(xs)
    occ = _as_list(data.get("_atom_site_occupancy")) or ["1"] * len(xs)
    out = []
    for i, x in enumerate(xs):
        label = str(labels[i] if i < len(labels) else "") or f"row {i + 1}"
        xyz = [_number(x, field=f"_atom_site_fract_x of {label}"),
               _number(ys[i] if i < len(ys) else None, field=f"_atom_site_fract_y of {label}"),
               _number(zs[i] if i < len(zs) else None, field=f"_atom_site_fract_z of {label}")]
        if any(v is None for v in xyz):
            # The site itself says it has no coordinate; that one is skippable.
            continue
        sym = str(symbols[i] if i < len(symbols) else "") or str(labels[i] if i < len(labels) else "")
        out.append({
            "label": str(labels[i] if i < len(labels) else ""),
            "symbol": _element_of(sym),
            "xyz": np.asarray(xyz, dtype=np.float64),
            "wyckoff": str(wyckoff[i] if i < len(wyckoff) else "").lstrip("."),
            "occupancy": _number(occ[i] if i < len(occ) else None,
                                 field=f"_atom_site_occupancy of {label}") or 1.0,
        })
    return out


_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def _element_of(token: str) -> str:
    """``Si2`` -> ``Si``, ``Fe3+`` -> ``Fe``, ``Mg1`` -> ``Mg``."""
    m = _ELEMENT_RE.match(str(token).strip())
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# the four sources of evidence
# ---------------------------------------------------------------------------

def _pymatgen_orbit_size(space_group: int, xyz) -> int:
    from pymatgen.symmetry.groups import SpaceGroup

    sg = SpaceGroup.from_int_number(int(space_group))
    return len(sg.get_orbit(np.asarray(xyz, dtype=np.float64)))


def choice_from_symops(symops: list[str], space_group: int) -> int | None:
    """Match the CIF's own operators against spglib's two settings."""
    if not symops:
        return None
    try:
        import spglib
        from pymatgen.core.operations import SymmOp
    except Exception:  # pragma: no cover - spglib is a hard dependency elsewhere
        return None

    def opset(rotations, translations):
        return {(tuple(np.asarray(R, int).ravel()),
                 tuple(np.round(np.asarray(t, float) % 1.0, 4)))
                for R, t in zip(rotations, translations)}

    try:
        ops = [SymmOp.from_xyz_str(s) for s in symops]
    except Exception:
        return None
    mine = opset([o.rotation_matrix for o in ops], [o.translation_vector for o in ops])
    for hall in range(1, 531):
        t = spglib.get_spacegroup_type(hall)
        if t["number"] != int(space_group) or t.get("choice") not in ("1", "2"):
            continue
        d = spglib.get_symmetry_from_database(hall)
        if opset(d["rotations"], d["translations"]) == mine:
            return int(t["choice"])
    return None


def choice_from_wyckoff(sites: list[dict], space_group: int) -> int | None:
    """``8a`` states the orbit size; expand both ways and see which delivers it."""
    declared = []
    for s in sites:
        m = re.match(r"^(\d+)", s["wyckoff"])
        if m:
            declared.append((s["xyz"], int(m.group(1))))
    if not declared:
        return None
    as_written = all(_pymatgen_orbit_size(space_group, xyz) == n for xyz, n in declared)
    shifted = all(_pymatgen_orbit_size(space_group, to_choice_1(xyz, space_group, 2)) == n
                  for xyz, n in declared)
    if as_written and not shifted:
        return 1
    if shifted and not as_written:
        return 2
    return None


def _formula_counts(text: str) -> dict[str, float]:
    """``'Al48 Fe32 Si16'`` -> ``{'Al': 48, 'Fe': 32, 'Si': 16}``."""
    counts: dict[str, float] = {}
    for token in re.findall(r"([A-Z][a-z]?)\s*([0-9.]*)", str(text)):
        el, n = token
        if not el:
            continue
        counts[el] = counts.get(el, 0.0) + (float(n) if n else 1.0)
    return counts


def _expanded_counts(sites: list[dict], space_group: int, shift: bool) -> dict[str, float]:
    counts: dict[str, float] = {}
    for s in sites:
        xyz = to_choice_1(s["xyz"], space_group, 2) if shift else s["xyz"]
        n = _pymatgen_orbit_size(space_group, xyz)
        if s["symbol"]:
            counts[s["symbol"]] = counts.get(s["symbol"], 0.0) + n * s["occupancy"]
    return counts


def choice_from_formula(data: dict, sites: list[dict], space_group: int) -> int | None:
    """The declared cell contents, against both expansions."""
    declared = _first(data, "_chemical_formula_sum")
    if not declared:
        return None
    want = _formula_counts(declared)
    if not want:
        return None

    def agrees(counts):
        if set(counts) != set(want):
            return False
        return all(abs(counts[k] - want[k]) <= 0.05 * max(1.0, want[k]) for k in want)

    as_written = agrees(_expanded_counts(sites, space_group, shift=False))
    shifted = agrees(_expanded_counts(sites, space_group, shift=True))
    if as_written and not shifted:
        return 1
    if shifted and not as_written:
        return 2
    return None


#: Only forms that can ONLY mean an origin choice: ``:1``/``:2`` (the CIF and
#: spglib convention) and ``O1``/``O2`` (SpringerMaterials). Deliberately NOT a
#: bare trailing digit — this library contains ``'P 1'``, ``'C 1 2/c 1'`` and
#: ``'A 1 2/a 1'``, and a looser pattern read all three as "origin choice 1".
_MARKER_RE = re.compile(r"(?::\s*|\s+[Oo])([12])\s*$")


def choice_from_symbol(hm: str) -> int | None:
    """``Fd-3m :2``, ``Fd-3m O1``. Weakest evidence, so it is last.

    A trailing ``S``/``Z`` is **not** read. It is sometimes used for the two
    origins, but the International Tables do not define it, the two readings in
    circulation are opposites ("S" = symmetry centre = choice 2 against "Z" =
    *zweite* = choice 2), and no CIF in this library uses it — so the rule could
    never be checked against a real file. An undecidable block already has a
    designed answer here, and it is to refuse; a rule nobody can verify is
    worse than that.
    """
    m = _MARKER_RE.search(str(hm).strip())
    return int(m.group(1)) if m else None


def decide_origin_choice(data: dict, sites: list[dict], space_group: int) -> tuple[int, list[str]]:
    """The origin choice of a block, with every source that had an opinion.

    Raises :class:`CifOriginError` when two sources disagree, or when nothing
    can tell — never falls back on a default, because the default is what put
    silicon on the wrong Wyckoff site for months.
    """
    votes: list[tuple[str, int]] = []
    ops = choice_from_symops(_symop_strings(data), space_group)
    if ops is not None:
        votes.append(("explicit symmetry operators", ops))
    wyck = choice_from_wyckoff(sites, space_group)
    if wyck is not None:
        votes.append(("Wyckoff multiplicity", wyck))
    formula = choice_from_formula(data, sites, space_group)
    if formula is not None:
        votes.append(("_chemical_formula_sum", formula))
    symbol = choice_from_symbol(_hm_symbol(data))
    if symbol is not None:
        votes.append(("the H-M symbol", symbol))

    if not votes:
        raise CifOriginError(
            f"space group {space_group} is published in two origin choices and this "
            "block says nothing that decides which one it is written in: no symmetry "
            "operators, no Wyckoff symbols, no formula sum, no marker on the H-M "
            "symbol. Refusing to guess — the wrong choice puts the atoms on a "
            "different Wyckoff site and doubles or inverts the cell."
        )
    distinct = {c for _, c in votes}
    if len(distinct) > 1:
        detail = ", ".join(f"{src} says {c}" for src, c in votes)
        raise CifOriginError(
            f"the block contradicts itself about its origin choice ({detail}). "
            "One of these statements is wrong; the file has to be corrected by hand."
        )
    choice = votes[0][1]
    return choice, [f"{src} says origin choice {c}" for src, c in votes]


# ---------------------------------------------------------------------------
# building the structure
# ---------------------------------------------------------------------------

def _lattice(data: dict):
    from pymatgen.core import Lattice

    vals = [_number(data.get(k)) for k in (
        "_cell_length_a", "_cell_length_b", "_cell_length_c",
        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")]
    if any(v is None for v in vals):
        return None
    return Lattice.from_parameters(*vals)


def _orbit_positions(space_group: int, xyz) -> set[tuple]:
    from pymatgen.symmetry.groups import SpaceGroup

    sg = SpaceGroup.from_int_number(int(space_group))
    return {tuple(np.round(np.asarray(p, dtype=np.float64) % 1.0, 4))
            for p in sg.get_orbit(np.asarray(xyz, dtype=np.float64))}


def _require_asymmetric_unit(sites: list[dict], space_group: int) -> None:
    """Refuse a site loop that is already the whole cell.

    :func:`_build_shifted` expands every listed site, so a CIF that lists more
    than the asymmetric unit would be expanded a second time. Measured on a
    complete 8-atom silicon cell with correct choice-2 operators: 64 atoms and
    18.634 g/cm3, reported as a normal result. pymatgen's own parser refuses
    such a file ("occupancies sum to > 1"), so this path must not accept it
    either -- and it is the density that goes into the Monte Carlo.

    Two rows at the *same* coordinate are a mixed site (Al/Si sharing a
    position) and are fine; two rows at *different* coordinates of one orbit are
    the same atom listed twice.
    """
    seen: list[tuple[set[tuple], dict]] = []
    for site in sites:
        here = tuple(np.round(site["xyz"] % 1.0, 4))
        orbit = _orbit_positions(space_group, site["xyz"])
        for other_orbit, other in seen:
            if tuple(np.round(other["xyz"] % 1.0, 4)) == here:
                continue                      # same site, another species
            if here in other_orbit:
                raise CifOriginError(
                    f"atom sites {other['label'] or '?'} and {site['label'] or '?'} are "
                    "symmetry images of each other, so this loop is not an asymmetric "
                    "unit -- it lists a cell that has already been expanded. Expanding "
                    "it again would multiply the atom count and the density. Use the "
                    "block that gives the asymmetric unit, or a P1 cell with the "
                    "operators written out."
                )
        seen.append((orbit, site))


def _build_shifted(data: dict, sites: list[dict], space_group: int):
    """Expand the asymmetric unit ourselves, after moving it to origin choice 1."""
    from pymatgen.core import Structure

    lattice = _lattice(data)
    if lattice is None:
        raise CifOriginError("the block has no complete set of cell parameters.")
    _require_asymmetric_unit(sites, space_group)
    species: list[Any] = []
    coords = []
    for s in sites:
        if not s["symbol"]:
            raise CifOriginError(f"atom site {s['label']!r} has no element symbol.")
        occ = float(s["occupancy"])
        species.append(s["symbol"] if occ >= 1.0 else {s["symbol"]: occ})
        coords.append(to_choice_1(s["xyz"], space_group, 2))
    return Structure.from_spacegroup(int(space_group), lattice, species, coords)


def _parse_block(path: Path, key: str):
    """pymatgen's own structure for one block, so its disorder handling stands."""
    from pymatgen.io.cif import CifFile, CifParser

    cf = CifFile.from_file(str(path))
    block = cf.data[key]
    parser = CifParser.from_str(str(block))
    structures = parser.parse_structures()
    if not structures:
        raise CifOriginError(f"pymatgen parsed no structure from block {key!r}.")
    return structures[0]


def analyse_cif(path: str | Path) -> list[BlockVerdict]:
    """One verdict per ``data_`` block that has atoms, in file order."""
    from pymatgen.io.cif import CifFile

    path = Path(path)
    cf = CifFile.from_file(str(path))
    verdicts: list[BlockVerdict] = []
    for key, block in cf.data.items():
        data = block.data
        sites = _sites(data)
        if not sites:
            continue
        number = _space_group_number(data) or _space_group_number_from_structure(path, key)
        v = BlockVerdict(key=key, space_group=number)
        if v.space_group is None:
            v.problem = "no space-group number, and none derivable from the structure"
        elif not has_two_origin_choices(v.space_group):
            v.choice = 1
            v.evidence = [f"space group {v.space_group} has only one origin"]
        else:
            try:
                v.choice, v.evidence = decide_origin_choice(data, sites, v.space_group)
            except CifOriginError as exc:
                v.problem = str(exc)
        verdicts.append(v)
    return verdicts


def structure_from_cif(path: str | Path):
    """Return ``(structure, report)`` with the origin choice honoured.

    The structure that comes back is always in origin choice 1, whatever the
    file was written in — which is the setting the rest of the pipeline
    (diffpy, and therefore ``.xtal`` with ``SpaceGroupSetting = 1``) expands in.
    """
    from pymatgen.io.cif import CifFile

    path = Path(path)
    cf = CifFile.from_file(str(path))
    verdicts = analyse_cif(path)
    if not verdicts:
        raise CifOriginError(f"{path.name}: no data block with atom sites.")

    problems = []
    for v in verdicts:
        if not v.usable:
            problems.append(f"block {v.key!r}: {v.problem}")
            continue
        data = cf.data[v.key].data
        sites = _sites(data)
        if v.choice == 2:
            structure = _build_shifted(data, sites, v.space_group)
            shifted = True
        else:
            structure = _parse_block(path, v.key)
            shifted = False
        report = CifOriginReport(
            path=str(path), block_key=v.key, space_group=v.space_group,
            cif_origin_choice=v.choice, shifted=shifted, evidence=list(v.evidence),
            other_blocks=[o for o in verdicts if o.key != v.key],
        )
        return structure, report

    raise CifOriginError(f"{path.name}: no usable data block. " + " | ".join(problems))
