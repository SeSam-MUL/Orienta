"""Local CIF / XTAL / SHT library indexer for the Crystal Hint feature.

Scans `Database/CIF_Library/`, `Database/XTAL_Library/`, and
`Database/EBSD_SHT_Database/` at import time, builds a unified searchable
index of all available phases. The index is queried by symmetry, lattice
range, and chemistry to suggest candidate phases for unindexed patterns.

See docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md Section 4.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# Composition display matches the Crystal Database (reduced_formula + subscripts).
# _PHASE_NICKNAMES is kept imported for the legacy _prettify_formula + its test.
from display_names import display_stem
from phase_metadata import (
    format_formula_subscripts, extract_metadata_from_cif, _PHASE_NICKNAMES,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CIF_DIR = PROJECT_ROOT / "Database" / "CIF_Library"
XTAL_DIR = PROJECT_ROOT / "Database" / "XTAL_Library"
SHT_DIR = PROJECT_ROOT / "Database" / "EBSD_SHT_Database"

def _is_backup(path: Path) -> bool:
    """True for a repair's leftover copy — it is not a phase.

    On 2026-09-13 a session symmetrised three CIFs and kept the P1 originals
    as ``<name>.P1-backup-<date>.cif``; the index offered all three in the
    phase list, so a user could pick the un-symmetrised copy of a phase they
    already had — the very file whose P1 symmetry makes a Hough reflector
    library explode (2026-08-03).

    ONE rule, defined in :mod:`cif_phase_library`, so the next backup naming
    convention is added in one place. Imported inside the function because
    this module is the light one of the two.
    """
    from backend.api.services.cif_phase_library import is_backup_file
    return is_backup_file(path)


# DOMAIN: Al-alloy EBSD only. Revisit if expanding the library to Fe/Ti/Ni
# systems — (space_group, sorted_elements) is NOT a unique fingerprint in
# general crystallography (multiple phases can share that pair across
# different chemistries). For our 22-CIF Al-alloy library it is unique;
# new families need to be checked.
#
# _PHASE_NICKNAMES is imported from phase_metadata (the single source) above and
# re-exported here for the existing collision test.
#
# Its two readers are `_prettify_formula` below and
# `phase_metadata.build_canonical_label` -- and NEITHER is reached from this
# module: `build_index` sets the display fields with `format_formula_subscripts`
# directly, and `build_canonical_label` has no backend caller at all. So the
# nicknames are, today, a table nothing displays. That is why the synonym store
# (`phase_synonyms`) SEEDS from this table rather than replacing it: whoever
# eventually wires a label builder should find the names still here.


def _prettify_formula(raw: str, space_group: str, elements: tuple[str, ...]) -> str:
    """Make CIF formulas displayable.

    - Plain integer formulas like `Al2Cu`, `Mg2Si` stay unchanged.
    - Decimal-heavy formulas (`Mn4.512Al127.296Fe19.488Si16.704`) get
      replaced with a literature nickname if we know one for the
      `(space_group, elements)` tuple, else `(Al,Fe,Mn,Si)` — element
      tuple in sorted order so the user can identify the chemistry
      without reading 30+ characters.
    """
    if not raw:
        return ""
    # If the raw formula has decimal points in any element subscript, it's
    # a partial-occupancy CIF — substitute.
    has_decimal = any(c == "." for c in raw)
    if not has_decimal:
        # Check it's also reasonably short (< 30 chars) — some integer
        # formulas like Al162Fe46Si30 also benefit from a shorter form
        # but aren't strictly wrong, so keep them.
        return raw
    sg_norm = (space_group or "").replace(" ", "")
    elements_sorted = tuple(sorted(elements))
    nickname = _PHASE_NICKNAMES.get((sg_norm, elements_sorted))
    if nickname:
        return nickname
    # Generic fallback: just the element tuple
    return "(" + ",".join(elements_sorted) + ")"


# Map from space group number to canonical crystal system.
# (Standard crystallographic conventions.)
def _system_from_sg_number(sg: int) -> str:
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    if 195 <= sg <= 230:
        return "cubic"
    return "unknown"


def _system_from_sg_name(name: str) -> str:
    """Heuristic mapping from H-M space group symbol to crystal system."""
    if not name:
        return "unknown"
    n = name.strip().lower().replace(" ", "")
    # Cubic markers
    if any(t in n for t in ("fm-3m", "im-3m", "pm-3m", "fd-3m", "ia-3d", "im-3", "pm-3", "ia-3", "p-43m")):
        return "cubic"
    # Hexagonal markers
    if any(t in n for t in ("p6_3/mmc", "p6/mmm", "p-6m2", "p-62m", "p6_3", "p6_5", "p6_1", "p6mm")) or n.startswith("p6") or n.startswith("p-6"):
        return "hexagonal"
    # Trigonal markers (R-3, P-3 etc.)
    if any(t in n for t in ("r-3", "p-3", "r3", "p3")):
        return "trigonal"
    # Tetragonal markers
    if "4" in n and not any(t in n for t in ("4_1", "4_2", "4_3")):
        # very rough — could be tetragonal i4/mmm, p4/mnc, etc.
        if any(t in n for t in ("i4", "p4", "i-4", "p-4")):
            return "tetragonal"
    # Orthorhombic markers
    if any(t in n for t in ("cmcm", "pnma", "pbca", "ibam", "a2/a")):
        return "orthorhombic"
    # Monoclinic markers
    if any(t in n for t in ("c2/m", "p2_1/c", "p2/c", "c12/m1", "p21/c")):
        return "monoclinic"
    return "unknown"


@dataclass
class LocalEntry:
    """Unified library entry for a phase in our local DB."""

    key: str                           # canonical filename stem (matches SHT name)
    cif_path: Optional[Path] = None
    xtal_path: Optional[Path] = None
    sht_path: Optional[Path] = None

    formula: str = ""
    # BOTH of these are the pymatgen `reduced_formula` rendered with subscripts
    # (`format_formula_subscripts`), set in `build_index`.
    #
    # They are NOT `phase_metadata.build_canonical_label`, whatever this comment
    # said until 2026-09-27. That function exists and produces
    # `<formula-or-(elements)> — <pearson> (<H-M>)[ · <nickname>]`, but nothing in
    # the backend calls it: measured, zero call sites outside its own tests. The
    # visible consequence is that `alpha-AlFeMnSi_ICSD-52623` carries
    # `Mn₅.₂₆₈Al₁₀₀.₈₉Fe₂₁.₂₄Si₁₀.₆₀₂` here while its literature nickname
    # `α-Al(Fe,Mn)Si` sits in the nickname table and appears nowhere. A reader who
    # believed the old comment would look for the bug in the label builder.
    display_formula: str = ""
    display_label: str = ""
    elements: tuple[str, ...] = ()
    space_group: str = ""
    space_group_number: Optional[int] = None
    crystal_system: str = "unknown"
    lattice_a_A: Optional[float] = None
    lattice_b_A: Optional[float] = None
    lattice_c_A: Optional[float] = None
    # Angles too, because the phase library's card promises "all lattice
    # parameters" and 12 of the 36 phases have one that is not 90 deg.
    lattice_alpha_deg: Optional[float] = None
    lattice_beta_deg: Optional[float] = None
    lattice_gamma_deg: Optional[float] = None
    n_atoms: Optional[int] = None
    parse_error: Optional[str] = None


@dataclass
class LocalMatch:
    entry: LocalEntry
    plausibility: float = 1.0       # 1.0 if all elements in sample, lower if partial
    lattice_distance: float = 0.0   # 0.0 if a in range, >0 if outside
    system_match: bool = True
    score: float = 0.0              # final ranking score (high = better)
    # Per-phase fit scores (0..1, 1.0 = perfect) computed from the
    # observed pattern's symmetry NCC + d-spacings. When None, no
    # pattern data was provided to the search → these scores aren't
    # applied as multipliers and the UI hides their chips.
    symmetry_fit: Optional[float] = None
    dspacing_fit: Optional[float] = None
    # 0..1 match between the pixel's measured EDS At% and this phase's
    # nominal composition. None when EDS weighting is off / unavailable.
    chemistry_fit: Optional[float] = None

    @property
    def has_sht(self) -> bool:
        return self.entry.sht_path is not None


def _parse_cif(cif_path: Path) -> dict:
    """Parse a CIF file with pymatgen. Returns dict with fields, or {'error': msg}."""
    try:
        from pymatgen.io.cif import CifParser

        # CifParser API changed in newer pymatgen — handle both shapes.
        parser = CifParser(str(cif_path), occupancy_tolerance=10.0)
        try:
            structures = parser.parse_structures(primitive=False)
        except (AttributeError, TypeError):
            # legacy fallback
            structures = parser.get_structures(primitive=False)
        # One guard for every place that reads a CIF; see
        # cif_phase_library.one_structure. Taking structures[0] lets the ORDER
        # of a CIF's data blocks decide a phase's composition, and on
        # sd_1816951.cif (MgCu2) the two answers are Mg4Cu and Mg2Cu, neither
        # of them the compound. Crystal Hint would have suggested it with an
        # inverted Mg:Cu.
        from backend.api.services.cif_phase_library import (
            AmbiguousCifError, one_structure,
        )
        try:
            s = one_structure(structures, cif_path.name)
        except AmbiguousCifError as exc:
            logger.warning("%s", exc)
            return {"error": str(exc)}
        try:
            sg_info = s.get_space_group_info()
            sg_symbol, sg_number = sg_info
        except Exception:
            sg_symbol, sg_number = ("", None)
        return {
            "formula": s.composition.reduced_formula,
            # `.composition.elements`, NOT `.as_dict().keys()`: the keys keep the
            # oxidation state, so an ICSD CIF writing `_atom_site_type_symbol` as
            # `Al0+` gave `('Al0+','Fe0+','Mn0+','Si0+')`. Every chemistry
            # intersection against real symbols was then empty and
            # `search(strict_chemistry)` DROPPED the phase -- measured on four
            # library files including alpha-AlFeMnSi_ICSD-52623, whose chemistry
            # is exactly right.
            #
            # This asks pymatgen for the elements instead of cleaning its keys
            # with our own normaliser. A review pointed it out and it is plainly
            # better: `Element.symbol` cannot carry a charge by construction, so
            # there is no string rule to get wrong, and nothing here can reach
            # the mixed-site branch of `clean_element_symbol` (which reads a
            # FORMULA backwards -- see its docstring).
            "elements": tuple(sorted({el.symbol for el in s.composition.elements})),
            "space_group": str(sg_symbol),
            "space_group_number": sg_number,
            "a_A": s.lattice.a,
            "b_A": s.lattice.b,
            "c_A": s.lattice.c,
            # The angles come from the SAME parsed structure as `n_atoms` and
            # the composition. Taking them from a second reader would let one
            # phase card describe two different data blocks without anyone
            # noticing -- and `a b c` without gamma = 120 deg is three numbers,
            # not a cell. 12 of the 36 library phases have an angle off 90 deg
            # (eight hexagonal, Al13Fe4 monoclinic 107.669, both beta-AlFeSi
            # 90.42, Fe3_Al2_Si3 triclinic).
            "alpha_deg": s.lattice.alpha,
            "beta_deg": s.lattice.beta,
            "gamma_deg": s.lattice.gamma,
            "n_atoms": len(s),
        }
    except Exception as exc:
        logger.warning("CIF parse failed for %s: %s", cif_path, exc)
        return {"error": str(exc)}


def _parse_xtal(xtal_path: Path) -> dict:
    """Parse a .xtal HDF5 file (EMsoft format). Lattice in nm in EMsoft."""
    try:
        with h5py.File(xtal_path, "r") as f:
            cd = f["CrystalData"]
            sg_number = int(cd["SpaceGroupNumber"][()][0]) if "SpaceGroupNumber" in cd else None
            lp = np.asarray(cd["LatticeParameters"][()])
            # EMsoft stores lengths in NM and angles in DEGREES.
            # lp = [a, b, c, alpha, beta, gamma]
            a_A = float(lp[0]) * 10.0
            b_A = float(lp[1]) * 10.0
            c_A = float(lp[2]) * 10.0
            atomtypes = []
            if "Atomtypes" in cd:
                atomtypes = list(map(int, cd["Atomtypes"][()]))
            # Convert atomic numbers to symbols
            from pymatgen.core.periodic_table import Element

            elements_set = set()
            for z in atomtypes:
                try:
                    elements_set.add(Element.from_Z(z).symbol)
                except Exception:
                    pass
            return {
                # `Element.from_Z(z).symbol`, so these are canonical by
                # construction -- atomic numbers cannot carry a charge. Running a
                # normaliser over them would be decorative: a review showed the
                # call could be deleted without any test noticing, because no
                # input can tell the two apart.
                "elements": tuple(sorted(elements_set)),
                "space_group_number": sg_number,
                "a_A": a_A,
                "b_A": b_A,
                "c_A": c_A,
                "n_atoms": len(atomtypes),
            }
    except Exception as exc:
        logger.warning("XTAL parse failed for %s: %s", xtal_path, exc)
        return {"error": str(exc)}


def _readable_without_structure(cif: Path) -> dict:
    """The fields a CIF still yields when pymatgen refuses its structure.

    `phase_metadata.extract_metadata_from_cif` is a regex reader over the
    `_sm_*`/symmetry/cell tags and does not go through pymatgen, so it answers
    for files the structure path rejects. Returns kwargs for `LocalEntry`; empty
    on any failure, because this is a best-effort rescue and must never be the
    reason an entry is missing altogether.
    """
    try:
        meta = extract_metadata_from_cif(cif)
    except Exception:
        logger.debug("metadata rescue failed for %s", cif.name, exc_info=True)
        return {}
    out: dict = {}
    # NO `elements` HERE, deliberately. The label and the atom table disagree on
    # two shipped files -- `beta-AlFeSi.cif` says `Al4 Fe Si` and has no Si site,
    # `sd_1401510.cif` labels itself `Mn0.5Fe0.5Al5Si0.68` with only Al and
    # Fe/Mn sites -- so deriving chemistry from the label would CLAIM Si the file
    # does not contain. A review measured the cost of doing it: forcing the
    # rescue on those two, `search(['Al','Fe','Mn'], strict)` went from
    # `['beta-AlFeSi','sd_1401510']` to `[]` -- the rescue would make the phases
    # it exists to save disappear for their own chemistry. Elements come from the
    # `.xtal` pass (atomic numbers, and the one file that needs this has one), or
    # not at all.
    #
    # `formula` is still taken, for DISPLAY, but never the filename:
    # `extract_metadata_from_cif` ends its chain at `display_stem(cif_path)`, and
    # `meta.source` is "cif" either way, so the only way to tell is to compare.
    # It matters because `crystal_hint_phase_fit.phase_nominal_at_pct` parses
    # this field for the EDS multiplier -- a filename there is a composition
    # invented out of a name.
    if meta.formula and meta.formula != display_stem(cif):
        display = format_formula_subscripts(meta.formula)
        out.update(formula=meta.formula, display_formula=display,
                   display_label=display)
    if meta.space_group:
        out["space_group"] = meta.space_group
    if meta.space_group_number:
        out["space_group_number"] = meta.space_group_number
        out["crystal_system"] = _system_from_sg_number(meta.space_group_number)
    for src, dst in (("lattice_a", "lattice_a_A"), ("lattice_b", "lattice_b_A"),
                     ("lattice_c", "lattice_c_A"),
                     ("lattice_alpha", "lattice_alpha_deg"),
                     ("lattice_beta", "lattice_beta_deg"),
                     ("lattice_gamma", "lattice_gamma_deg")):
        v = getattr(meta, src, None)
        if v:
            out[dst] = v
    return out


def build_index() -> dict[str, LocalEntry]:
    """Scan the Database/ directory tree and build the unified index."""
    entries: dict[str, LocalEntry] = {}

    # First pass: walk CIF files
    if CIF_DIR.exists():
        for cif in sorted(CIF_DIR.glob("*.cif")):
            if _is_backup(cif):
                continue
            key = cif.stem
            info = _parse_cif(cif)
            if "error" in info:
                # The STRUCTURE path refused (e.g. AmbiguousCifError: the blocks
                # give two compositions). That does not mean the file says
                # nothing -- the `_sm_*`/symmetry reader is independent of
                # pymatgen and usually reads fine. Keep what IS readable, and
                # keep `parse_error` so the entry can say what is missing.
                # Measured on sd_1816951 (MgCu2): formula, space group, IT number
                # and lattice all come back; only the composition is ambiguous.
                entries[key] = LocalEntry(
                    key=key, cif_path=cif, parse_error=info["error"],
                    **_readable_without_structure(cif),
                )
                continue
            sg_name = info["space_group"]
            sg_num = info["space_group_number"]
            system = (
                _system_from_sg_number(sg_num) if sg_num
                else _system_from_sg_name(sg_name)
            )
            # Composition shown exactly like the Crystal Database: the pymatgen
            # reduced_formula (info["formula"]) rendered with subscripts.
            display = format_formula_subscripts(info["formula"])
            entries[key] = LocalEntry(
                key=key,
                cif_path=cif,
                formula=info["formula"],
                display_formula=display,
                display_label=display,
                elements=info["elements"],
                space_group=sg_name,
                space_group_number=sg_num,
                crystal_system=system,
                lattice_a_A=info["a_A"],
                lattice_b_A=info["b_A"],
                lattice_c_A=info["c_A"],
                lattice_alpha_deg=info.get("alpha_deg"),
                lattice_beta_deg=info.get("beta_deg"),
                lattice_gamma_deg=info.get("gamma_deg"),
                n_atoms=info["n_atoms"],
            )

    # Second pass: walk XTAL files. Match by stem, augment if CIF missing
    if XTAL_DIR.exists():
        for xtal in sorted(XTAL_DIR.glob("*.xtal")):
            if _is_backup(xtal):
                continue
            key = xtal.stem
            info = _parse_xtal(xtal)
            existing = entries.get(key)
            if "error" in info:
                if existing is None:
                    entries[key] = LocalEntry(key=key, xtal_path=xtal, parse_error=info["error"])
                else:
                    existing.xtal_path = xtal
                continue
            sg_num = info["space_group_number"]
            system = _system_from_sg_number(sg_num) if sg_num else "unknown"
            if existing is None:
                entries[key] = LocalEntry(
                    key=key,
                    xtal_path=xtal,
                    elements=info["elements"],
                    space_group_number=sg_num,
                    crystal_system=system,
                    lattice_a_A=info["a_A"],
                    lattice_b_A=info["b_A"],
                    lattice_c_A=info["c_A"],
                    lattice_alpha_deg=info.get("alpha_deg"),
                    lattice_beta_deg=info.get("beta_deg"),
                    lattice_gamma_deg=info.get("gamma_deg"),
                    n_atoms=info["n_atoms"],
                )
            else:
                existing.xtal_path = xtal
                # Fill in missing fields from XTAL
                if not existing.elements:
                    existing.elements = info["elements"]
                if existing.space_group_number is None:
                    existing.space_group_number = sg_num
                    existing.crystal_system = system

    # Third pass: walk SHT files and attach to matching entries.
    # SHT filename convention: "Formula (CIF_stem) [Pearson] {voltage}.sht"
    # The CIF_stem appears as substring; iterate entries longest-first to win
    # against any incidental short-name collision.
    if SHT_DIR.exists():
        sorted_keys = sorted(entries.keys(), key=lambda k: -len(k))
        for sht in sorted(SHT_DIR.rglob("*.sht")):
            if sht.stem == "test" or _is_backup(sht):
                continue
            sht_stem = sht.stem
            matched = None
            for k in sorted_keys:
                if k in sht_stem:
                    matched = k
                    break
            if matched is None:
                entries[sht_stem] = LocalEntry(key=sht_stem, sht_path=sht)
            else:
                entries[matched].sht_path = sht

    return entries


_INDEX_CACHE: Optional[dict[str, LocalEntry]] = None
_INDEX_SIG: Optional[tuple] = None


def _library_signature() -> tuple:
    """A cheap (count, newest-mtime) fingerprint of the CIF/XTAL/SHT libraries.

    Changes whenever a file is added, removed, or modified — e.g. after a
    re-simulation writes new .sht / .xtal. Used to auto-invalidate the cached
    index so the phase pickers + phase test never serve stale masters without a
    backend restart.
    """
    n = 0
    newest = 0
    for d in (CIF_DIR, XTAL_DIR, SHT_DIR):
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            try:
                if p.is_file():
                    n += 1
                    m = p.stat().st_mtime_ns
                    if m > newest:
                        newest = m
            except OSError:
                pass
    return (n, newest)


def get_index() -> dict[str, LocalEntry]:
    """Cached accessor; auto-rebuilds when the library files change on disk."""
    global _INDEX_CACHE, _INDEX_SIG
    sig = _library_signature()
    if _INDEX_CACHE is None or sig != _INDEX_SIG:
        _INDEX_CACHE = build_index()
        _INDEX_SIG = sig
    return _INDEX_CACHE


def rebuild_index() -> dict[str, LocalEntry]:
    """Force re-scan of the Database/ directory."""
    global _INDEX_CACHE
    _INDEX_CACHE = build_index()
    return _INDEX_CACHE


def search(
    elements: list[str],
    crystal_system: Optional[str] = None,
    a_range_A: Optional[tuple[float, float]] = None,
    strict_chemistry: bool = True,
    compatible_systems: Optional[list[str]] = None,
    n_fold_scores: Optional[dict[int, float]] = None,
    d_observed_A: Optional[list[float]] = None,
    d_observed_trustworthy: bool = False,
    pixel_at_pct: Optional[dict[str, float]] = None,
    eds_weighting: str = "off",
    eds_filter_threshold: float = 0.35,
    stats: Optional[dict] = None,
) -> list[LocalMatch]:
    """Return ranked list of LocalMatch.

    Args:
        elements: Sample chemistry. Phase elements must be a subset of these
            (if strict_chemistry=True), else partial overlap is allowed.
        crystal_system: Primary filter. Phases with this system are
            system_match=True. Used when caller has a single hint.
        compatible_systems: Optional broader list of systems that should
            ALL count as system_match=True. When the symmetry detector
            sees multiple folds (e.g. 3-fold AND 4-fold both ≥ LOW), the
            pattern is consistent with multiple systems — tetragonal
            phases shouldn't get penalised because 3-fold ranks slightly
            higher than 4-fold by chance. If both arguments are given,
            the union of {crystal_system} ∪ compatible_systems is used.
        a_range_A: (min_a, max_a) in Angstroms. Lattice param 'a' must fall
            in this range. If None, no lattice filter is applied.
        strict_chemistry: If True, phase elements must be ⊆ sample elements.
            If False, plausibility is set to fraction-of-shared-elements.

    Returns sorted by score (highest first).
    """
    sample_set = {e.strip().capitalize() for e in elements}
    # When caller passes no chemistry hints, treat as "skip chemistry filter"
    # rather than dropping every phase. Otherwise strict_chemistry + empty
    # sample_set would return zero candidates silently.
    no_chemistry = len(sample_set) == 0
    matches: list[LocalMatch] = []

    # Pre-compute the set of "matching" systems for fast lookup.
    matching_systems: Optional[set[str]] = None
    if crystal_system is not None or compatible_systems:
        matching_systems = set(compatible_systems or [])
        if crystal_system is not None:
            matching_systems.add(crystal_system)

    for entry in get_index().values():
        # A `parse_error` used to drop the entry outright. Measured cost of that:
        # `search(['Mg','Cu'], strict_chemistry=True)` returned ZERO results,
        # because MgCu2 (sd_1816951) is the library's only Mg-Cu phase and its
        # CIF gives pymatgen two compositions. The phase that IS the answer was
        # the only one missing. It carries elements (from its .xtal), an IT
        # number and a .sht -- enough to be a candidate. So skip only an entry
        # with nothing to match ON, and let the rest through carrying the reason.
        if entry.parse_error and not entry.elements and not entry.space_group_number:
            continue

        # Crystal system filter
        system_match = True
        if matching_systems is not None:
            if entry.crystal_system not in matching_systems:
                system_match = False

        # Chemistry filter / plausibility scoring
        phase_set = set(entry.elements)
        if no_chemistry:
            plausibility = 0.7  # no chemistry filter — score down a notch
        elif not phase_set:
            plausibility = 0.5  # unknown chemistry; don't fully exclude
        else:
            shared = phase_set & sample_set
            if strict_chemistry:
                if phase_set.issubset(sample_set):
                    plausibility = 1.0
                else:
                    continue  # strict: drop unrelated phases entirely
            else:
                plausibility = len(shared) / max(len(phase_set), 1)

        # Lattice filter
        lattice_distance = 0.0
        if a_range_A is not None and entry.lattice_a_A is not None:
            lo, hi = a_range_A
            a = entry.lattice_a_A
            if a < lo:
                lattice_distance = (lo - a) / lo
            elif a > hi:
                lattice_distance = (a - hi) / hi

        # Per-phase fit scores (Option A + Option B). Multiplied into the
        # score so phases whose symmetry signature + d-spacings actually
        # match the pattern rank above those that just pass categorical
        # filters. Without these, every preset-expected cubic phase
        # tied at score 1.50.
        sym_fit: Optional[float] = None
        d_fit: Optional[float] = None
        sym_multiplier = 1.0
        d_multiplier = 1.0
        if n_fold_scores:
            from backend.api.services.crystal_hint_phase_fit import symmetry_fit as _sf
            sym_fit = _sf(n_fold_scores, entry.crystal_system)
            sym_multiplier = sym_fit
        # Only apply the d-spacing fit when the observed d-spacings are
        # trustworthy. On real 118×118 EDAX patterns the Hough band
        # detector is dominated by the square detector's ±45° frame
        # diagonals (artifact, identical for every pixel) and the lattice
        # estimator reports confidence="none" — feeding those bogus
        # d-spacings into dspacing_fit just injects per-candidate noise
        # into the ranking. The caller passes d_observed_trustworthy=True
        # only when lattice confidence is at least "low".
        if d_observed_A and entry.lattice_a_A and d_observed_trustworthy:
            from backend.api.services.crystal_hint_phase_fit import dspacing_fit as _df
            d_fit = _df(
                d_observed_A,
                lattice_a_A=entry.lattice_a_A,
                crystal_system=entry.crystal_system,
                lattice_c_A=entry.lattice_c_A,
                space_group=entry.space_group or "",
            )
            d_multiplier = d_fit

        # EDS chemistry weighting (Option C). The pixel's measured At% vs
        # this phase's nominal composition — the discriminator pattern
        # symmetry/d-spacing cannot provide (e.g. Al vs Al2Cu vs
        # α-Al(Fe,Mn)Si all tie on a cubic pattern). 'soft' damps the score
        # by chemistry match (never to zero); 'filter' drops chemically
        # inconsistent phases entirely; 'off' is bit-identical to before.
        chem_fit: Optional[float] = None
        chem_multiplier = 1.0
        if eds_weighting in ("soft", "filter") and pixel_at_pct:
            from backend.api.services.crystal_hint_phase_fit import (
                chemistry_fit as _cf, phase_nominal_at_pct as _at,
            )
            chem_fit = _cf(pixel_at_pct, _at(entry.formula or ""))
            if eds_weighting == "filter":
                if chem_fit < eds_filter_threshold:
                    # Hard-drop chemically inconsistent phase. Count it so
                    # the caller can warn the user (fail loud, not silent).
                    if stats is not None:
                        stats["eds_filtered"] = stats.get("eds_filtered", 0) + 1
                    continue
            else:  # soft: ×0.3..×1.0, never zero
                chem_multiplier = 0.3 + 0.7 * chem_fit

        # A half-read phase must never OUTRANK a fully-read one. Three of the
        # five multipliers above are free passes when their input is missing --
        # no lattice means the lattice filter and `dspacing_fit` both skip
        # (`1.0` twice), `crystal_system == "unknown"` makes `symmetry_fit`
        # return 1.0, and a formula that parses to no element makes
        # `chemistry_fit` return its neutral 1.0 and become un-droppable under
        # `eds_weighting="filter"`. So an entry that knows nothing pays nothing,
        # while every real competitor pays something. A review built the case:
        # a parse-error entry with no cell and no label scored 1.0000 on an Al-Cu
        # pixel and came FIRST, ahead of Al2Cu (0.6724) and Al (0.6421).
        #
        # The penalty is a cap, not a guess at a fit: it says "ranked below
        # anything fully read", which is the only honest claim available for a
        # phase whose own file could not be parsed. It still beats absence --
        # being offered last with a reason is the point of offering it at all.
        unreadable_penalty = 0.5 if entry.parse_error else 1.0
        score = (
            plausibility
            * (1.0 if system_match else 0.5)
            * (1.0 / (1.0 + lattice_distance))
            * sym_multiplier
            * d_multiplier
            * chem_multiplier
            * unreadable_penalty
        )

        matches.append(
            LocalMatch(
                entry=entry,
                plausibility=plausibility,
                lattice_distance=lattice_distance,
                system_match=system_match,
                score=score,
                symmetry_fit=sym_fit,
                dspacing_fit=d_fit,
                chemistry_fit=chem_fit,
            )
        )

    matches.sort(key=lambda m: -m.score)
    return matches


def all_entries() -> list[LocalEntry]:
    """Return the full index (debug / UI helper)."""
    return list(get_index().values())
