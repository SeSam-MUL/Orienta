"""Hidden symmetry in CIF files that are stored as space group P 1.

A CIF written without symmetry (a Materials Project / DFT download that went
through ``CifWriter(structure)``, or a relaxed structure with 0.003 A of noise
on every atom) is space group P 1 as far as every reader is concerned, although
its atoms may sit on a high-symmetry lattice. The cost is not cosmetic:
PyEBSDIndex builds its band-triplet library from the number of reflector
families, a P 1 phase has no symmetry-equivalent poles to merge, so a noisy
P 1 cell with about 70 reflector families asks for about 1.0e9 triplet rows
(44.7 GiB; reproduced on a synthetic cell, see tests/synthetic_cif.py) where
the symmetrised copy of the same crystal needs under 48 MiB.

This module detects that case and writes a symmetrised COPY. It never touches
the original, and it refuses where a symmetry guess would be a guess about the
sample rather than about the file (partial occupancies).

The tolerance is 0.01 A, the value Orienta's own Materials Project download
already uses (``crystal_hint_mp``), together with an ANGLE tolerance of 0.5
degrees. pymatgen's default angle tolerance is 5 degrees, which on its own
turns a body-centred tetragonal cell with c/a = 1.002 to 1.05 into cubic Im-3m
and a two-atom monoclinic cell with beta = 90.4 degrees into orthorhombic Cmmm,
whatever the length tolerance says; at 0.5 degrees both stay what they are,
and the intended cases (0.003 A of noise) are not affected.

A length tolerance of 0.01 A is not a guarantee, though. With realistic noise
(0.003 to 0.015 A) the analyser can answer with a SUBGROUP of the true group at
that one tolerance. So the group is looked at along a ladder of tolerances
(0.01, 0.02, 0.05, 0.1 A) and a copy is offered only where the answer does not
change along it; otherwise the ladder is shown and the user chooses.
"""

from __future__ import annotations

import itertools
import logging
import re
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from backend.api.services.cif_citation import apply_citation, read_citation

logger = logging.getLogger(__name__)

SYMPREC = 0.01

#: Angle tolerance (degrees) of every symmetry analysis in this module and in the
#: Materials Project download. pymatgen's own default is 5.
ANGLE_TOLERANCE = 0.5


def spacegroup_analyzer(structure, symprec: float = SYMPREC):
    """``SpacegroupAnalyzer`` with the angle tolerance stated, not inherited."""

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    return SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=ANGLE_TOLERANCE)

# Why no symmetrised copy is offered. The strings are codes, not prose: the
# interface translates them.
#: The tolerances (Angstrom) at which the group is looked at before a copy is offered.
LADDER = (0.01, 0.02, 0.05, 0.1)

NOT_P1 = "not_p1"
AMBIGUOUS_TOLERANCE = "ambiguous_tolerance"
INVALID_TOLERANCE = "invalid_tolerance"
DISORDERED = "disordered"
UNREADABLE = "unreadable"
NO_SYMMETRY = "no_symmetry"
CHANGED_STRUCTURE = "changed_structure"
NOT_WRITABLE = "not_writable"
SAME_CONTENT_EXISTS = "same_content_exists"
CONTRADICTORY_HEADER = "contradictory_header"


def _failure(exc: BaseException) -> str:
    """``ExceptionType: message`` of a failed analysis, short and without a path
    (these come from spglib / pymatgen computations, not from file access)."""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}".rstrip(": ")


class SymmetriseError(ValueError):
    """The copy could not be written. ``code`` is one of the reason codes."""

    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.extra = extra        # e.g. {"existing": "<file name>"}


@dataclass
class P1Verdict:
    applicable: bool = False
    reason: str = ""
    n_atoms: int = 0
    space_group_number: Optional[int] = None
    space_group_symbol: str = ""
    crystal_system: str = ""
    symprec: float = SYMPREC
    #: ``[{"symprec", "number", "symbol"}, ...]``: the group at each tolerance of the ladder.
    ladder: list = field(default_factory=list)
    #: True when the tolerance was chosen by the user instead of the group being the
    #: same along the whole ladder.
    chosen: bool = False
    #: For ``contradictory_header``: ``{"number": 1, "symbol": "F m -3 m"}``.
    header: dict = field(default_factory=dict)
    #: For ``unreadable``: what failed, as ``ExceptionType: message``. Never silent.
    detail: str = ""
    # Held for the copy, not part of the answer.
    _structure: object = field(default=None, repr=False)
    _text: str = field(default="", repr=False)       # the CIF text that was read back
    _copy_atoms: int = field(default=0, repr=False)  # atoms in the copy's cell
    _back: object = field(default=None, repr=False)  # the copy, read back
    _cell: dict = field(default_factory=dict, repr=False)  # what happened to the cell

    def as_dict(self) -> dict:
        out = {
            "applicable": self.applicable,
            "reason": self.reason,
            "n_atoms": self.n_atoms,
            "symprec": self.symprec,
        }
        if self.ladder:
            out["ladder"] = list(self.ladder)
            out["ambiguous"] = self.reason == AMBIGUOUS_TOLERANCE
        if self.header:
            out["header"] = dict(self.header)
        if self.detail:
            out["detail"] = self.detail
        if self.applicable:
            out["space_group_number"] = self.space_group_number
            out["space_group_symbol"] = self.space_group_symbol
            out["crystal_system"] = self.crystal_system
        return out


def _rotations():
    """The 24 proper signed axis permutations, the identity first."""
    import numpy as np

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, j in enumerate(perm):
                m[i, j] = signs[i]
            if round(float(np.linalg.det(m))) == 1:
                out.append(m)
    out.sort(key=lambda m: 0 if np.allclose(m, np.eye(3)) else 1)
    return out


def _cell_params(lattice):
    import numpy as np

    return np.array(list(lattice.abc) + list(lattice.angles))


def _matching_axes(source_lattice, other_lattice):
    """Every signed axis permutation ``m`` with ``m @ other`` having the cell shape
    of ``source`` (lengths within 1 %, angles within 1 degree), identity first, so
    a cell whose axes are interchangeable keeps its own order where that is right.
    Within 1 % is not "the same axes": a=5.10, b=5.09 matches both ways round, and
    which one is meant is decided by the atoms (:func:`_atoms_map_onto`)."""
    import numpy as np
    from pymatgen.core import Lattice

    want = _cell_params(source_lattice)
    for m in _rotations():
        have = _cell_params(Lattice(m @ other_lattice.matrix))
        if np.all(np.abs(have[:3] - want[:3]) / want[:3] < 0.01) and np.all(np.abs(have[3:] - want[3:]) < 1.0):
            yield m


def _match_axes(source_lattice, other_lattice):
    """The first of :func:`_matching_axes`, or None."""
    return next(_matching_axes(source_lattice, other_lattice), None)


#: How far (in Angstrom) a source atom may be from the copy's atom it maps onto: a
#: few times the symmetry tolerance for the noise of a relaxed structure, far below
#: the distance two swapped axes move an atom (0.1 in fractional units of a 5 A
#: axis is 0.5 A).
MAP_TOLERANCE_A = 0.08


def _atoms_map_onto(source, other, symprec: float = SYMPREC, tolerance: float = MAP_TOLERANCE_A) -> bool:
    """Do the source's atoms sit where the copy's atoms sit, in the SAME axes?

    True if some operation of ``other``'s space group, followed by a common shift
    of origin, carries every source atom onto a distinct atom of the same
    element within ``tolerance``. The structure matcher cannot answer this: it
    accepts any relabelling of the axes, so a copy whose a and b are swapped is
    "the same crystal" to it while every orientation computed against the copy
    refers to different axes than the file the user started from."""
    import numpy as np
    from collections import Counter


    species = [str(sp) for sp in source.species]
    other_species = [str(sp) for sp in other.species]
    if len(source) != len(other) or Counter(species) != Counter(other_species):
        return False
    positions = {sp: np.array([c for c, s in zip(other.frac_coords, other_species) if s == sp]) for sp in set(species)}
    matrix = other.lattice.matrix
    moved_all = np.array(source.frac_coords)
    for operation in spacegroup_analyzer(other, symprec).get_symmetry_operations(cartesian=False):
        moved = operation.operate_multi(moved_all)
        for candidate in positions[species[0]]:
            shift = candidate - moved[0]
            used = {sp: set() for sp in positions}
            for x, sp in zip(moved, species):
                delta = (positions[sp] - (x + shift) + 0.5) % 1.0 - 0.5
                distance = np.linalg.norm(delta @ matrix, axis=1)
                nearest = int(distance.argmin())
                if distance[nearest] > tolerance or nearest in used[sp]:
                    break
                used[sp].add(nearest)
            else:
                return True
    return False


def _axes_text(m) -> str:
    """``a->b, b->c, c->a``: which axis of the copy each axis of the source became."""
    import numpy as np

    letters = "abc"
    return ", ".join(f"{letters[i]}->{letters[int(np.argmax(np.abs(m[i])))]}" for i in range(3))


def _in_source_axes(structure, standard):
    """``standard`` (the conventional cell) written in the source's own axes, or
    None when the source cell is not a conventional cell of the group.

    A conventional cell has the same number of atoms and the shape of the
    standard one, only with its axes labelled differently (a=6.74, b=4.52,
    c=5.09 against the standard 4.52, 5.09, 6.74). Relabelling would make the
    orientations of a later indexing refer to different crystal axes than the
    file the user started from, so when the source cell qualifies the symmetry
    is written in it. Where several labellings give the same cell shape (a=5.10,
    b=5.09), the one whose atoms land on the source's atoms is taken; if none
    does, None: the standard setting is the only honest description. Likewise
    for a primitive cell, a supercell, another cell shape.
    """
    from pymatgen.core import Lattice, Structure

    if len(structure) != len(standard):
        return None
    for m in _matching_axes(structure.lattice, standard.lattice):
        lattice = Lattice(m @ standard.lattice.matrix)
        candidate = Structure(lattice, standard.species, lattice.get_fractional_coords(standard.cart_coords))
        if _atoms_map_onto(structure, candidate):
            return candidate
    return None


def _as_the_pipeline_reads_it(text: str):
    """``(atoms_in_phase, structure)`` of CIF text as the Hough pipeline reads it.

    pymatgen is not the reader that matters for indexing: ``hough_phase_from_cif``
    goes through diffpy and orix, and diffpy decides which operators to expand
    with. For a setting it does not recognise it falls back to the STANDARD
    operators of the group number in the file, whatever operators the file lists.
    A hexagonal cell with gamma = 60 degrees (a common DFT convention) in
    P6_3/mmc came back from pymatgen with 4 atoms and from the pipeline with 14.
    ``structure`` is diffpy's own expansion, converted, so that positions can
    be compared too; ``atoms_in_phase`` is the number the indexing sees."""
    from diffpy.structure.parsers import p_cif
    from pymatgen.core import Lattice, Structure

    from ebsd_utils import hough_phase_from_cif

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "candidate.cif"
        path.write_text(text, encoding="utf-8", newline="\n")
        in_phase = len(hough_phase_from_cif(str(path)).structure)
        parsed = p_cif.P_cif().parseFile(str(path))
    lat = parsed.lattice
    structure = Structure(
        Lattice.from_parameters(lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma),
        [str(a.element) for a in parsed],
        [[float(x) for x in a.xyz] for a in parsed],
    )
    return in_phase, structure


def _pipeline_accepts(structure, candidate: str, symprec: float = SYMPREC) -> bool:
    """Does the Hough pipeline read ``candidate`` as the source's crystal: the same
    number of atoms in the phase it builds, and diffpy's own expansion carrying
    the source's atoms onto its atoms in the same axes?"""
    in_phase, as_read = _as_the_pipeline_reads_it(candidate)
    return in_phase == len(structure) and _atoms_map_onto(structure, as_read, symprec)


def _write_in_cell(cell, standard_setting: bool, symprec: float) -> str:
    """CIF text with the symmetry operations of ``cell`` listed in ``cell``'s own
    axes (``CifWriter`` always rewrites the cell in the standard setting).
    The Hermann-Mauguin symbol is only written when the axes are the standard
    ones; with other axes it would name a setting the operations do not have."""

    sga = spacegroup_analyzer(cell, symprec)
    operations = sga.get_symmetry_operations(cartesian=False)
    classes = sga.get_symmetrized_structure().equivalent_sites
    a, b, c = cell.lattice.abc
    alpha, beta, gamma = cell.lattice.angles
    lines = [f"data_{cell.composition.reduced_formula}"]
    if standard_setting:
        lines.append(f"_symmetry_space_group_name_H-M   '{sga.get_space_group_symbol()}'")
    lines += [
        f"_cell_length_a   {a:.8f}", f"_cell_length_b   {b:.8f}", f"_cell_length_c   {c:.8f}",
        f"_cell_angle_alpha   {alpha:.8f}", f"_cell_angle_beta   {beta:.8f}", f"_cell_angle_gamma   {gamma:.8f}",
        f"_symmetry_Int_Tables_number   {sga.get_space_group_number()}",
        f"_cell_volume   {cell.volume:.8f}",
        "loop_", " _symmetry_equiv_pos_site_id", " _symmetry_equiv_pos_as_xyz",
    ]
    lines += [f"  {i}  '{op.as_xyz_str()}'" for i, op in enumerate(operations, 1)]
    lines += [
        "loop_", " _atom_site_type_symbol", " _atom_site_label", " _atom_site_symmetry_multiplicity",
        " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy",
    ]
    for i, sites in enumerate(classes):
        first = sites[0]
        x, y, z = first.frac_coords
        lines.append(f"  {first.specie}  {first.specie}{i}  {len(sites)}  {x:.8f}  {y:.8f}  {z:.8f}  1")
    return "\n".join(lines) + "\n"


_HM_RE = re.compile(r"^_symmetry_space_group_name_H-M[ \t]+(?:'([^'\r\n]+)'|(\S+))", re.MULTILINE)


def _written_symbol(text: str) -> str:
    m = _HM_RE.search(text)
    return (m.group(1) or m.group(2)).strip() if m else ""


def _read_back(text: str):
    """``(structure, space_group_number, symbol)`` of CIF text, read the way the
    pipeline reads a CIF file (origin choice honoured)."""
    from backend.forward_sim.crystal.cif_origin import structure_from_cif

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "readback.cif"
        path.write_text(text, encoding="utf-8", newline="\n")
        structure, report = structure_from_cif(path)
    return structure, int(report.space_group), _written_symbol(text)


def _detect(structure, symprec: float = SYMPREC) -> P1Verdict:
    """Run the analyser on an already-expanded structure (declared as P 1).

    The group reported is the one of the CIF that would be WRITTEN, read back
    the way the pipeline reads it. ``CifWriter`` analyses the structure again,
    and on a borderline cell its answer differs from the analyser's (a noisy fcc
    cell was "Cmcm" to the analyser and Fm-3m in the file; a noisy Pnnn cell
    "P2" and Pnnn). Reporting the analyser's number would name a group the copy
    does not have.
    """
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.groups import SpaceGroup

    structure = _plain(structure)
    v = P1Verdict(n_atoms=len(structure), symprec=symprec, _structure=structure)
    if len(structure) == 0:
        v.reason = UNREADABLE
        return v
    if not structure.is_ordered:
        v.reason = DISORDERED
        return v
    try:
        sga = spacegroup_analyzer(structure, symprec)
        if int(sga.get_space_group_number()) <= 1:
            v.reason = NO_SYMMETRY
            return v
        standard = sga.get_conventional_standard_structure()
        text = back = number = symbol = None
        kept = False
        in_source = _in_source_axes(structure, standard)
        if in_source is not None:
            # The symmetry goes into the source's own axes. Accepted only if
            # the file read back is the same group by an analysis of its own.
            refined = sga.get_refined_structure()
            standard_setting = _match_axes(in_source.lattice, refined.lattice) is not None and bool(
                (_cell_params(in_source.lattice) - _cell_params(refined.lattice)).__abs__().max() < 1e-6)
            try:
                candidate = _write_in_cell(in_source, standard_setting, symprec)
                c_back, c_number, c_symbol = _read_back(candidate)
                accepted = (
                    len(c_back) == len(structure)
                    and c_number == int(spacegroup_analyzer(c_back, symprec).get_space_group_number())
                    and _atoms_map_onto(structure, c_back, symprec)
                    # What the indexing will read, not only what pymatgen reads.
                    and _pipeline_accepts(structure, candidate, symprec)
                )
            except Exception:
                accepted = False
            if accepted:
                text, back, number, symbol, kept = candidate, c_back, c_number, c_symbol, True
                if not symbol:
                    symbol = str(spacegroup_analyzer(in_source, symprec).get_space_group_symbol())
        if text is None:
            text = str(CifWriter(standard, symprec=symprec, angle_tolerance=ANGLE_TOLERANCE))
            back, number, symbol = _read_back(text)
            # The standard path is held to the same standard as the one that keeps
            # the source's axes: the file is what the indexing will read, so the
            # Hough pipeline's reader (diffpy/orix) must see the crystal that pymatgen
            # read back, not whatever its own expansion makes of the operators.
            if number > 1 and not _pipeline_accepts(back, text, symprec):
                v.reason = CHANGED_STRUCTURE
                return v
        system = SpaceGroup.from_int_number(number).crystal_system
    except Exception as exc:  # spglib refuses some degenerate cells
        logger.warning("symmetry analysis failed at symprec %g", symprec, exc_info=True)
        v.reason = UNREADABLE
        v.detail = _failure(exc)
        return v
    if number <= 1:
        v.reason = NO_SYMMETRY
        return v
    if not _same_crystal(structure, back):
        v.reason = CHANGED_STRUCTURE
        return v
    v.applicable = True
    v.space_group_number = number
    v.space_group_symbol = symbol
    v.crystal_system = str(system)
    v._text = text
    v._copy_atoms = len(back)
    v._back = back
    axes = "" if kept else _axes_of_conversion(structure, back)
    v._cell = {
        "kept_source_axes": kept,
        "atoms_before": len(structure),
        "atoms_after": len(back),
        "axes": axes,
    }
    return v


def _axes_of_conversion(structure, copy) -> str:
    """How the axes were relabelled when the cell was standardised, as
    ``a->b, b->c, c->a``, or '' when the cell shape changed so that no
    relabelling describes it (a primitive cell, a supercell) or when the axes
    kept their order: an identity mapping is not a relabelling."""
    import numpy as np

    m = _match_axes(structure.lattice, copy.lattice)
    if m is None or np.allclose(m, np.eye(3)):
        return ""
    return _axes_text(m)


def _plain(structure):
    """The structure with elements instead of oxidation-state species (``Mg2+`` ->
    ``Mg``). A P 1 file with charged type symbols is read back as ``Mg4 O4`` and
    compared with ``Mg2+4 O2-4`` otherwise, which never matches: every candidate in
    the source's axes was refused for such a file. The copy carries elements."""
    plain = structure.copy()
    plain.remove_oxidation_states()
    return plain


def _cell_sentence(cell: dict) -> str:
    if cell["kept_source_axes"]:
        return "Cell and axes as in the source file."
    text = (f"Cell converted to the conventional standard setting; atoms per cell "
            f"{cell['atoms_before']} -> {cell['atoms_after']}")
    if cell["axes"]:
        text += f"; source axes relabelled {cell['axes']} (source -> copy)"
    return text + "."


_DECLARED_NUMBER_RE = re.compile(
    r"^[ \t]*_(?:symmetry_Int_Tables_number|space_group_IT_number)[ \t]+(\d+)", re.MULTILINE | re.IGNORECASE
)
_DECLARED_NAME_RE = re.compile(
    r"^[ \t]*_(?:symmetry_space_group_name_H-M|space_group_name_H-M_alt)[ \t]+"
    r"(?:'([^'\r\n]*)'|\"([^\"\r\n]*)\"|(\S+))",
    re.MULTILINE | re.IGNORECASE,
)


_DECLARED_HALL_RE = re.compile(
    r"^[ \t]*_(?:symmetry_space_group_name_Hall|space_group_name_Hall)[ \t]+"
    r"(?:'([^'\r\n]*)'|\"([^\"\r\n]*)\"|(\S+))",
    re.MULTILINE | re.IGNORECASE,
)


def _normal_symbol(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def header_contradiction(text: str) -> Optional[dict]:
    """``{"number": 1, "symbol": "F m -3 m"}`` when the header's space group number
    and its symbol disagree about P 1, else None.

    The symbol is the Hermann-Mauguin one or the Hall one (``-F 4 2 3 (Hall)``).
    Two cases: the number says 1 and a symbol names another group (the operators,
    a lone ``x, y, z``, follow the number), or the number names another group and
    a symbol says P 1 (the Hough pipeline reads such a file as P 1 with the atoms
    that are listed). Such a header is not P 1 and not the group it names: taking
    it as P 1 would offer a one-atom Pm-3m copy of what the symbol calls an fcc
    cell. Nothing is offered; the header has to be corrected first. A header that
    gives no number, or several, is not judged here."""
    numbers = {int(n) for n in _DECLARED_NUMBER_RE.findall(text)}
    if len(numbers) != 1:
        return None
    number = next(iter(numbers))
    symbols = [" ".join(next(g for g in m if g).split()) for m in _DECLARED_NAME_RE.findall(text)]
    symbols += [" ".join(next(g for g in m if g).split()) + " (Hall)" for m in _DECLARED_HALL_RE.findall(text)]
    for symbol in symbols:
        says_p1 = _normal_symbol(symbol.removesuffix(" (Hall)")) == "p1"
        if says_p1 != (number == 1):
            return {"number": number, "symbol": symbol}
    return None


def may_be_p1(cif_path: str | Path) -> bool:
    """Does the file's own header say P 1 (or nothing at all)? A text check, a
    few milliseconds, to decide whether the expensive analysis is worth running.

    ``analyse_p1`` reads the structure with the origin-choice machinery, which
    took 4.5 s for Si.cif and 8.3 s for sd_1816951.cif: doing that for every
    CIF that is imported would make an import of ordinary files a wait for
    nothing. A file that declares a space group number other than 1, or a
    Hermann-Mauguin name other than ``P 1``, is not P 1. A file that declares
    nothing is read by pymatgen as P 1, so it counts as a candidate. Not every
    reader agrees (the Hough pipeline's diffpy expands whatever operators the
    file lists, and a file with a number but no operators is expanded with the
    standard ones), which is why this is only the pre-check: ``analyse_p1`` asks
    the origin-aware reader and decides."""
    try:
        text = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if header_contradiction(text):
        return True       # not P 1, but analyse_p1 is the one that says why
    numbers = {int(n) for n in _DECLARED_NUMBER_RE.findall(text)}
    if numbers:
        return numbers == {1}
    names = {re.sub(r"\s+", "", next(g for g in m if g)).lower() for m in _DECLARED_NAME_RE.findall(text)}
    if names:
        return names == {"p1"}
    return True


def symmetry_ladder(structure) -> list[dict]:
    """The space group of ``structure`` at each tolerance of :data:`LADDER`, as
    ``[{"symprec": 0.01, "number": 8, "symbol": "Cm"}, ...]``. A tolerance at
    which the analysis failed has ``number`` None and an ``error`` text.

    One tolerance is not a measurement. A relaxed or noisy structure answers with
    a SUBGROUP of its true group at one tolerance and with the true group at
    another (fcc Al with +-0.008 A of noise: Cm at 0.01 A, Fm-3m at 0.05 A), and
    which one is right is not something the analyser can say."""
    rungs = []
    for tolerance in LADDER:
        try:
            analyser = spacegroup_analyzer(structure, tolerance)
            rungs.append({
                "symprec": tolerance,
                "number": int(analyser.get_space_group_number()),
                "symbol": str(analyser.get_space_group_symbol()),
            })
        except Exception as exc:  # spglib refuses some degenerate cells at some tolerances
            # The rung stays in the ladder as a failure: skipping it would let the
            # remaining rungs agree ("stable") on a question that was not asked.
            logger.warning("symmetry analysis failed at symprec %g", tolerance, exc_info=True)
            rungs.append({"symprec": tolerance, "number": None, "symbol": "", "error": _failure(exc)})
    return rungs


def ladder_text(ladder: list[dict]) -> str:
    """``0.01 A: Cm (8); 0.02 A: Cm (8); 0.05 A: Fm-3m (225); ...`` for messages."""
    return "; ".join(
        f"{r['symprec']:g} A: " + (f"{r['symbol']} ({r['number']})" if r.get("number") is not None else "not determined")
        for r in ladder
    )


def hough_hint(verdict: P1Verdict, name: Optional[str] = None) -> str:
    """The sentence the Hough memory errors add for a P 1 phase whose atoms have
    more symmetry, or '' when the verdict has none to report. It states the ladder
    where the group depends on the tolerance, never a single guess."""
    subject = f"{name} is stored as space group P 1" if name else "This CIF is stored as space group P 1"
    if verdict.applicable:
        return (f"{subject}, but its atom positions have the symmetry {verdict.space_group_symbol} "
                f"({verdict.space_group_number}) at every tolerance from 0.01 to 0.1 A. Open the CIF in the "
                f"Crystal Database and use \"Save symmetrised copy\", then index with the copy.")
    if verdict.reason == AMBIGUOUS_TOLERANCE:
        return (f"{subject}, and the symmetry of its atom positions depends on the tolerance "
                f"({ladder_text(verdict.ladder)}). Open the CIF in the Crystal Database, use "
                f"\"Save symmetrised copy\" and choose a tolerance, then index with the copy.")
    return ""


def _ladder_outcome(ladder: list[dict]) -> str:
    """``stable`` (one group above P 1 at every tolerance), ``none`` (P 1 at every
    tolerance), ``ambiguous`` (the answer changes along the ladder), ``incomplete``
    (the analysis failed at some tolerance: the others cannot make it stable) or
    ``unreadable`` (it failed at every tolerance)."""
    numbers = {r["number"] for r in ladder if r.get("number") is not None}
    if not numbers:
        return "unreadable"
    if any(r.get("number") is None for r in ladder):
        return "incomplete"
    if len(numbers) > 1:
        return "ambiguous"
    return "none" if numbers == {1} else "stable"


def analyse_p1(cif_path: str | Path, symprec: Optional[float] = None) -> P1Verdict:
    """Is ``cif_path`` a P 1 file whose atoms have more symmetry?

    ``symprec=None``: the group is looked at along :data:`LADDER`. If it is the same
    at every tolerance the verdict is that group (and a copy is offered); P 1 at
    every tolerance is no symmetry; if the answer changes the verdict is
    ``ambiguous_tolerance``, with the ladder, and no copy is offered until a
    tolerance is chosen. ``symprec`` set to one of the ladder's tolerances is
    that choice."""
    try:
        header = header_contradiction(Path(cif_path).read_text(encoding="utf-8", errors="ignore"))
    except OSError as exc:
        return P1Verdict(reason=UNREADABLE, detail=f"{type(exc).__name__}: the file could not be read")
    if header:
        return P1Verdict(reason=CONTRADICTORY_HEADER, header=header)
    try:
        from backend.forward_sim.crystal.cif_origin import structure_from_cif

        structure, report = structure_from_cif(cif_path)
    except Exception as exc:
        logger.warning("structure of %s could not be read for the symmetry check", Path(cif_path).name, exc_info=True)
        return P1Verdict(reason=UNREADABLE, detail=_failure(exc))
    if int(report.space_group) != 1:
        return P1Verdict(reason=NOT_P1, n_atoms=len(structure))
    structure = _plain(structure)
    if len(structure) == 0:
        return P1Verdict(reason=UNREADABLE, detail="the file lists no atoms")
    if not structure.is_ordered:
        return P1Verdict(reason=DISORDERED, n_atoms=len(structure))
    if symprec is not None and symprec not in LADDER:
        return P1Verdict(reason=INVALID_TOLERANCE, n_atoms=len(structure))
    ladder = symmetry_ladder(structure)
    if symprec is not None:
        verdict = _detect(structure, symprec)
        verdict.ladder, verdict.chosen = ladder, True
        return verdict
    outcome = _ladder_outcome(ladder)
    if outcome == "unreadable":
        return P1Verdict(reason=UNREADABLE, n_atoms=len(structure), ladder=ladder, detail=ladder[0]["error"])
    if outcome in ("ambiguous", "incomplete"):
        return P1Verdict(reason=AMBIGUOUS_TOLERANCE, n_atoms=len(structure), ladder=ladder)
    verdict = _detect(structure, LADDER[0])
    verdict.ladder = ladder
    return verdict


def copy_base_name(stem: str) -> str:
    """The name a copy is built on: the source's stem without a repair marker.

    A copy of ``Al7FeCu2.P1-backup-2026-09-13.cif`` named
    ``...-backup-..._sym.cif`` would still look like a repair's leftover to
    ``is_backup_file`` and be hidden from the phase library and the Hough
    picker, which is where the copy is meant to appear. Everything from the
    first marker on is dropped (``Al7FeCu2.P1``); a name that is nothing but a
    marker becomes ``copy``.
    """
    from backend.api.services.cif_phase_library import _BACKUP_MARKERS

    low = stem.lower()
    cuts = [i for i in (low.find(m) for m in _BACKUP_MARKERS) if i >= 0]
    base = stem[:min(cuts)] if cuts else stem
    return base.rstrip("._- ") or "copy"


def _sibling_candidates(src: Path, base: str) -> list[Path]:
    """The other CIFs in the folder whose name starts with the copy's first token
    (``Al7FeCu2`` for ``Al7FeCu2.P1``), skipping repair markers; a cheap listing."""
    from backend.api.services.cif_phase_library import is_backup_file

    token = re.split(r"[._\- ]", base, maxsplit=1)[0].lower()
    if not token:
        return []
    candidates = []
    for path in sorted(src.parent.iterdir(), key=lambda q: q.name.lower()):
        if path.suffix.lower() != ".cif" or path == src or is_backup_file(path):
            continue
        stem = path.stem.lower()
        if stem == token or (stem.startswith(token) and not stem[len(token)].isalnum()):
            candidates.append(path)
    return candidates


def _same_content_sibling(src: Path, base: str, back, skip=frozenset()) -> Optional[str]:
    """Name of a file in the folder that already holds this crystal, or None.

    Candidates are the other CIFs whose name starts with the same first token
    as the copy's (``Al7FeCu2`` for ``Al7FeCu2.P1``), skipping repair markers
    and anything that is itself P 1. Each is read and compared with the same
    matcher that guards the copy, so a hand-repaired sibling in another setting
    (the library has them in groups 128, 225 and 217) is recognised, and
    a copy saved a moment ago is too.
    """
    from backend.forward_sim.crystal.cif_origin import structure_from_cif

    candidates = [p for p in _sibling_candidates(src, base) if p.name not in skip]
    # Every candidate is looked at: a cap would miss the real twin whenever a
    # family of similarly named files sorts in front of it. A file whose header
    # says P 1 cannot be "already there" and is skipped without being read.
    for path in candidates:
        if may_be_p1(path):
            continue
        try:
            other, report = structure_from_cif(path)
        except Exception:
            continue
        if int(report.space_group) > 1 and _same_crystal(back, other):
            return path.name
    return None


_SAVE_LOCK = threading.Lock()


def _create_new_file(src: Path, base: str, text: str) -> Path:
    """Write ``text`` to the first free ``<base>_sym.cif``, ``<base>_sym2.cif``...

    Each name is claimed with an exclusive create (``open(..., "x")``), so two
    requests that choose a name at the same moment cannot both get it: the
    loser sees ``FileExistsError`` and takes the next one. Checking
    ``exists()`` first and writing afterwards let parallel requests overwrite
    each other's ``_sym3``.
    """
    n = 1
    while True:
        name = f"{base}_sym.cif" if n == 1 else f"{base}_sym{n}.cif"
        dest = src.with_name(name)
        try:
            handle = open(dest, "x", encoding="utf-8", newline="\n")
        except FileExistsError:
            n += 1
            continue
        try:
            with handle:
                handle.write(text)
        except BaseException:
            # The name was claimed and the write failed (disk full, a quota): do
            # not leave an empty file behind. It would be taken for a copy, and a
            # retry would go on to _sym2.
            try:
                dest.unlink()
            except OSError:
                logger.warning("could not remove the incomplete copy %s", dest.name)
            raise
        return dest


def _same_crystal(original, refined) -> bool:
    """The symmetrised structure must still be the crystal that was read.

    ``get_conventional_standard_structure`` averages positions and lattice onto
    the detected group; with a tolerance of 0.01 A that is a small move. This
    asks the structure matcher whether the two are the same crystal at
    tolerances an order of magnitude tighter than its defaults, so a silent
    change of composition or cell is refused instead of written.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(
        ltol=0.01, stol=0.02, angle_tol=0.5,
        primitive_cell=True, scale=False, attempt_supercell=False,
    )
    return bool(matcher.fit(original, refined))


def _tolerance_sentence(verdict: P1Verdict) -> str:
    if verdict.chosen:
        return (f"Tolerance {verdict.symprec:g} Angstrom chosen by the user from the ladder "
                f"{ladder_text(verdict.ladder)}")
    return (f"The same group at every tolerance of "
            f"{', '.join(f'{t:g}' for t in LADDER)} Angstrom (symprec {verdict.symprec:g} Angstrom used)")


def save_symmetrised_copy(src_path: str | Path, today: Optional[date] = None,
                          symprec: Optional[float] = None) -> dict:
    """Write ``<stem>_sym.cif`` next to ``src_path`` and describe it.

    Never overwrites: an existing name gets a numeric suffix. The new file
    carries its provenance (source name, method, tolerance, group, date) as
    comments and in ``_audit_creation_method`` and keeps the original's DOI and
    reference text. The group named in all of them, and returned, is the one
    read back from the text that is written.
    """
    src = Path(src_path)
    verdict = analyse_p1(src, symprec)
    if not verdict.applicable:
        message = _REFUSALS.get(verdict.reason, "Not possible.")
        if verdict.reason == AMBIGUOUS_TOLERANCE:
            message = f"{message} {ladder_text(verdict.ladder)}."
        extra = {"ladder": verdict.ladder} if verdict.ladder else {}
        if verdict.detail:
            message = f"{message} ({verdict.detail})"
            extra["detail"] = verdict.detail
        raise SymmetriseError(verdict.reason or UNREADABLE, message, **extra)

    today = today or date.today()
    method = (
        f"Symmetrised by Orienta from P 1 file {src.name}; space group "
        f"{verdict.space_group_symbol} ({verdict.space_group_number}). "
        f"{_tolerance_sentence(verdict)}. {_cell_sentence(verdict._cell)}"
    )
    out: list[str] = [
        "# Symmetrised copy written by Orienta. The source file was not changed.",
        f"# Source file: {src.name} (space group P 1)",
        f"# Space group of this copy: {verdict.space_group_symbol} ({verdict.space_group_number}), "
        f"symprec {verdict.symprec:g} Angstrom",
        f"# Tolerance: {_tolerance_sentence(verdict)}",
        f"# Cell: {_cell_sentence(verdict._cell)}",
        f"# Date: {today.isoformat()}",
    ]
    inserted = False
    for line in verdict._text.splitlines():
        if line.startswith("# generated using pymatgen"):
            continue
        out.append(line)
        if line.startswith("data_") and not inserted:
            out.append(f"_audit_creation_date   {today.isoformat()}")
            # A text field, not a quoted string: a file name may hold quotes,
            # and a quote followed by a space ends a CIF string early.
            out.extend(["_audit_creation_method", ";", method, ";"])
            inserted = True
    new_text = "\n".join(out) + "\n"

    citation = read_citation(src.read_text(encoding="utf-8", errors="ignore"))
    new_text, _ = apply_citation(new_text, doi=citation.doi, reference=citation.reference)

    base = copy_base_name(src.stem)
    try:
        # The scan reads every candidate (seconds each) and only reads: it runs
        # outside the lock, so a save in another folder is not queued behind it.
        examined = frozenset(p.name for p in _sibling_candidates(src, base))
        existing = _same_content_sibling(src, base, verdict._back)
        if existing:
            raise SymmetriseError(
                SAME_CONTENT_EXISTS,
                f"The same crystal is already in this folder as {existing}; nothing was written.",
                existing=existing,
            )
        # One create at a time. Inside the lock only the files that appeared since
        # the scan are looked at (a parallel request's copy, typically): the
        # check and the create must not interleave with it.
        with _SAVE_LOCK:
            existing = _same_content_sibling(src, base, verdict._back, skip=examined)
            if existing:
                raise SymmetriseError(
                    SAME_CONTENT_EXISTS,
                    f"The same crystal is already in this folder as {existing}; nothing was written.",
                    existing=existing,
                )
            dest = _create_new_file(src, base, new_text)
    except OSError:
        # The OS message carries the full local path (and with it a user name);
        # it goes to the log, not into the answer.
        logger.warning("symmetrised copy of %s could not be written", src.name, exc_info=True)
        raise SymmetriseError(NOT_WRITABLE, _REFUSALS[NOT_WRITABLE]) from None
    return {
        "name": dest.name,
        "path": str(dest),
        "source": src.name,
        "space_group_number": verdict.space_group_number,
        "space_group_symbol": verdict.space_group_symbol,
        "symprec": verdict.symprec,
        "tolerance_chosen": verdict.chosen,
        "ladder": list(verdict.ladder),
        "n_atoms": verdict._copy_atoms,
        "cell": dict(verdict._cell),
    }


_REFUSALS = {
    NOT_P1: "The file is not stored as space group P 1.",
    DISORDERED: "Partially occupied sites: symmetry detection would be a guess about the sample.",
    UNREADABLE: "The file could not be read as a crystal structure.",
    NOT_WRITABLE: "The folder of this CIF file is not writable, so no copy was saved.",
    SAME_CONTENT_EXISTS: "The same crystal is already in this folder; nothing was written.",
    NO_SYMMETRY: "No symmetry above P 1 was found at any tolerance from 0.01 to 0.1 Angstrom.",
    AMBIGUOUS_TOLERANCE: "The group of the atom positions depends on the tolerance; choose one:",
    INVALID_TOLERANCE: "The tolerance must be one of 0.01, 0.02, 0.05 or 0.1 Angstrom.",
    CONTRADICTORY_HEADER: (
        "The header contradicts itself (the space group number and the space group symbol "
        "disagree about P 1); no copy was written. Correct the header first."
    ),
    CHANGED_STRUCTURE: (
        "The symmetrised structure no longer matches the file within tolerance; "
        "nothing was written."
    ),
}
