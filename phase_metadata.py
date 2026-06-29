"""Phase metadata extraction from CIF, SHT, H5, and XTAL files.

Provides chemical formula, space group, and Pearson symbol for phase
selection display in the indexing GUI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass
class PhaseMetadata:
    """Metadata extracted from a crystallographic phase file."""
    formula: str = ""
    space_group: str = ""
    pearson: str = ""
    phase_name: str = ""
    prototype: str = ""
    source: str = ""       # "cif" | "sht_filename" | "h5_crystaldata" | "xtal" | "stem"
    cif_path: Optional[Path] = None
    display_label: str = ""
    # Crystallographic fields used by the degeneracy detector.
    lattice_a: Optional[float] = None
    lattice_b: Optional[float] = None
    lattice_c: Optional[float] = None
    lattice_alpha: Optional[float] = None
    lattice_beta: Optional[float] = None
    lattice_gamma: Optional[float] = None
    space_group_number: Optional[int] = None
    laue_class: str = ""
    centering: str = ""

# ---------------------------------------------------------------------------
# Element extraction & grouping
# ---------------------------------------------------------------------------

_ELEMENT_RE = re.compile(r'([A-Z][a-z]?)')


def extract_elements(formula: str) -> list[str]:
    """Extract unique element symbols from a chemical formula, sorted alphabetically."""
    if not formula:
        return []
    elements = sorted(set(_ELEMENT_RE.findall(formula)))
    valid = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
        "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
        "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
        "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
        "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
        "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Pb", "Bi",
        "Th", "U",
    }
    return [e for e in elements if e in valid]


def compute_element_group(elements: list[str]) -> str:
    """Compute a display group name from a list of element symbols."""
    if not elements:
        return "Sonstiges"
    if len(elements) == 1:
        return "Reine Elemente"
    return "-".join(sorted(elements))


# ---------------------------------------------------------------------------
# CIF parsing
# ---------------------------------------------------------------------------

def _parse_cif_field(text: str, tag: str) -> str:
    """Extract first non-empty value for a CIF tag.

    Handles tab-delimited (SpringerMaterials), space-delimited (Materials
    Project), and single-quoted (IUCr) formats.  Returns first non-empty
    match across multi-data-block files.
    """
    pattern = rf"^{re.escape(tag)}\s+(.+)$"
    for match in re.finditer(pattern, text, re.MULTILINE):
        raw = match.group(1).strip().strip("'").strip()
        if raw and raw != "?":
            return raw
    return ""


# A few well-known phase nicknames keyed by (normalized space group, sorted
# elements). They surface a readable prototype tag (\u03b1-Al(Fe,Mn)Si) for
# decimal-heavy disordered CIF formulas. DOMAIN: Al-alloy EBSD library \u2014 the
# (space_group, elements) pair is a unique fingerprint there, not in general
# crystallography. This is the SINGLE home for these (crystal_hint imports it).
_PHASE_NICKNAMES: dict[tuple[str, tuple[str, ...]], str] = {
    ("Im-3", ("Al", "Fe", "Mn", "Si")): "\u03b1-Al(Fe,Mn)Si",
    ("Im-3", ("Al", "Fe", "Mn")): "\u03b1-Al(Fe,Mn)",
    ("C2/c", ("Al", "Fe", "Si")): "\u03b2-AlFeSi",
    ("C2/m", ("Al", "Fe")): "Al13Fe4-like",
    ("Pm-3", ("Al", "Mn", "Si")): "Al-Mn-Si (cubic)",
}


def phase_nickname(space_group: str, elements) -> str:
    """Literature nickname (\u03b1/\u03b2/\u03c0\u2026) for a (space group, elements) pair, or ""."""
    sg_norm = (space_group or "").replace(" ", "")
    return _PHASE_NICKNAMES.get((sg_norm, tuple(sorted(elements))), "")


def _formula_display(formula: str, elements=()) -> str:
    """A readable formula: keep clean integer formulas as-is; for decimal-heavy
    or over-long formulas use a sorted element list ``(Al,Fe,Mn,Si)``."""
    if formula and "." not in formula and len(formula) <= 24:
        return formula
    els = tuple(elements) or tuple(extract_elements(formula))
    if els:
        return "(" + ",".join(els) + ")"
    return formula or ""


def build_canonical_label(
    *, formula: str = "", space_group: str = "", pearson: str = "",
    elements=(), prototype: str = "",
) -> str:
    """The ONE phase-identity display string used everywhere (Indexing,
    Phase-Tester, Crystal-Hint, file lists).

    Format \u2014 chemistry leads, structure next, prototype/nickname as a tag:
    ``<formula-or-(elements)> \u2014 <pearson> (<H-M>)[ \u00b7 <prototype/nickname>]``.
    """
    fdisp = _formula_display(formula, elements)
    if not fdisp:
        return ""
    parts = [fdisp]
    if pearson and space_group:
        parts.append(f"\u2014 {pearson} ({space_group})")
    elif space_group:
        parts.append(f"\u2014 ({space_group})")
    elif pearson:
        parts.append(f"\u2014 {pearson}")
    els = tuple(elements) or tuple(extract_elements(formula))
    tag = prototype or phase_nickname(space_group, els)
    if tag and tag != fdisp and tag != formula:
        parts.append(f"\u00b7 {tag}")
    return " ".join(parts)


_SUBSCRIPT_MAP = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")


def format_formula_subscripts(formula: str) -> str:
    """Render the digits of a chemical formula as Unicode subscripts
    (``Al13Fe4`` -> ``Al\u2081\u2083Fe\u2084``, ``Mn4.512Al127`` -> ``Mn\u2084.\u2085\u2081\u2082Al\u2081\u2082\u2087``), matching
    the Crystal Database's composition display. Same rule as the DB builder's
    ``_format_formula_with_subscripts``."""
    if not formula:
        return ""
    return re.sub(r"(\d)", lambda m: m.group(1).translate(_SUBSCRIPT_MAP), formula)


def _build_display_label(meta: PhaseMetadata) -> str:
    """Phase identity for the UI = the chemical COMPOSITION (formula). The crystal
    system / space group are shown SEPARATELY by the UI \u2014 not mashed in. Returned
    plain (no subscripts) here because ``meta.formula`` may be a stem/ID fallback
    (e.g. ``sd_0302719``) where digits are NOT stoichiometry; the picker paths
    (composition_for_cif / crystal_hint) subscript the real reduced_formula."""
    if not meta.formula:
        return meta.cif_path.stem if meta.cif_path else ""
    return meta.formula


def extract_metadata_from_cif(cif_path: Path) -> PhaseMetadata:
    """Extract metadata from a CIF file."""
    text = cif_path.read_text(encoding="utf-8", errors="replace")

    # Formula fallback chain
    formula = _parse_cif_field(text, "_sm_phase_labels")
    if not formula:
        raw = _parse_cif_field(text, "_chemical_formula_structural")
        formula = raw.replace("_", "") if raw else ""
    if not formula:
        raw = _parse_cif_field(text, "_chemical_formula_sum")
        formula = raw.replace(" ", "") if raw else ""
    if not formula:
        raw = _parse_cif_field(text, "_chemical_formula_iupac")
        formula = raw.replace(" ", "") if raw else ""
    if not formula:
        formula = cif_path.stem

    # Space group with fallback
    space_group = _parse_cif_field(text, "_symmetry_space_group_name_H-M")
    if not space_group:
        space_group = _parse_cif_field(text, "_space_group_name_H-M_alt")

    pearson = _parse_cif_field(text, "_sm_pearson_symbol")
    prototype = _parse_cif_field(text, "_sm_phase_prototype")

    phase_name = ""
    if prototype and prototype != formula:
        phase_name = prototype

    # Lattice parameters (degeneracy detector). CIF tags carry optional
    # standard-uncertainty suffixes like "12.643(2)" — _strip_cif_su drops them.
    lattice_a = _strip_cif_su(_parse_cif_field(text, "_cell_length_a"))
    lattice_b = _strip_cif_su(_parse_cif_field(text, "_cell_length_b"))
    lattice_c = _strip_cif_su(_parse_cif_field(text, "_cell_length_c"))
    lattice_alpha = _strip_cif_su(_parse_cif_field(text, "_cell_angle_alpha"))
    lattice_beta = _strip_cif_su(_parse_cif_field(text, "_cell_angle_beta"))
    lattice_gamma = _strip_cif_su(_parse_cif_field(text, "_cell_angle_gamma"))

    sg_number_raw = (_parse_cif_field(text, "_symmetry_Int_Tables_number")
                     or _parse_cif_field(text, "_space_group_IT_number"))
    space_group_number = None
    if sg_number_raw:
        try:
            space_group_number = int(float(sg_number_raw))
        except ValueError:
            space_group_number = None

    meta = PhaseMetadata(
        formula=formula,
        space_group=space_group,
        pearson=pearson,
        phase_name=phase_name,
        prototype=prototype,
        source="cif",
        cif_path=cif_path,
        lattice_a=lattice_a,
        lattice_b=lattice_b,
        lattice_c=lattice_c,
        lattice_alpha=lattice_alpha,
        lattice_beta=lattice_beta,
        lattice_gamma=lattice_gamma,
        space_group_number=space_group_number,
        laue_class=_laue_class_from_sg_number(space_group_number),
        centering=_centering_from_hm(space_group),
    )
    meta.display_label = _build_display_label(meta)
    return meta


# ---------------------------------------------------------------------------
# SHT filename parsing
# ---------------------------------------------------------------------------

_SHT_REGEX = re.compile(
    r'^(?P<formula>[^(\[{]+?)\s*'
    r'(?:\((?P<phase>.+)\)\s*)?'
    r'(?:\[(?P<pearson>[^\]]+)\]\s*)?'
    r'(?:\{(?P<kv>\d+)kV(?:\s+(?P<tilt>\d+)deg)?\})?'
    r'\s*$'
)


def extract_metadata_from_sht_filename(sht_path: Path) -> PhaseMetadata:
    """Extract metadata from SHT filename convention."""
    stem = sht_path.stem
    m = _SHT_REGEX.match(stem)
    if m:
        formula = m.group("formula").strip()
        phase_name = (m.group("phase") or "").strip()
        pearson = (m.group("pearson") or "").strip()
    else:
        formula = stem
        phase_name = ""
        pearson = ""

    meta = PhaseMetadata(
        formula=formula,
        pearson=pearson,
        phase_name=phase_name,
        source="sht_filename",
    )
    meta.display_label = _build_display_label(meta)
    return meta


# ---------------------------------------------------------------------------
# H5 / XTAL CrystalData parsing
# ---------------------------------------------------------------------------

_SPACE_GROUP_SYMBOLS = {
    1: "P1", 2: "P-1", 3: "P2", 4: "P21", 5: "C2",
    10: "P2/m", 11: "P21/m", 12: "C2/m", 13: "P2/c", 14: "P21/c",
    15: "C2/c", 47: "Pmmm", 51: "Pmma", 55: "Pbam", 58: "Pnnm",
    62: "Pnma", 63: "Cmcm", 64: "Cmce", 65: "Cmmm", 69: "Fmmm",
    71: "Immm", 74: "Imma", 123: "P4/mmm", 129: "P4/nmm",
    136: "P42/mnm", 139: "I4/mmm", 141: "I41/amd",
    148: "R-3", 160: "R3m", 166: "R-3m", 167: "R-3c",
    176: "P63/m", 186: "P63mc", 191: "P6/mmm", 194: "P63/mmc",
    197: "I23", 200: "Pm-3", 204: "Im-3", 216: "F-43m",
    217: "I-43m", 220: "I-43d", 221: "Pm-3m", 225: "Fm-3m",
    227: "Fd-3m", 229: "Im-3m", 230: "Ia-3d",
}

_SG_TO_CRYSTAL_SYSTEM: dict[int, str] = {}
for _sg in range(1, 3):
    _SG_TO_CRYSTAL_SYSTEM[_sg] = "triclinic"
for _sg in range(3, 16):
    _SG_TO_CRYSTAL_SYSTEM[_sg] = "monoclinic"
for _sg in range(16, 75):
    _SG_TO_CRYSTAL_SYSTEM[_sg] = "orthorhombic"
for _sg in range(75, 143):
    _SG_TO_CRYSTAL_SYSTEM[_sg] = "tetragonal"
for _sg in range(143, 168):
    _SG_TO_CRYSTAL_SYSTEM[_sg] = "trigonal"
for _sg in range(168, 195):
    _SG_TO_CRYSTAL_SYSTEM[_sg] = "hexagonal"
for _sg in range(195, 231):
    _SG_TO_CRYSTAL_SYSTEM[_sg] = "cubic"


def get_crystal_system(space_group: str) -> str:
    """Derive crystal system from space group name or number.

    Accepts formats like "Fm-3m (225)" (with number in parens) or a bare
    Hermann-Mauguin symbol like "Fm-3m".
    """
    import re as _re
    m = _re.search(r'\((\d+)\)', space_group)
    if m:
        return _SG_TO_CRYSTAL_SYSTEM.get(int(m.group(1)), "")
    for sgn, name in _SPACE_GROUP_SYMBOLS.items():
        if name == space_group or space_group.startswith(name):
            return _SG_TO_CRYSTAL_SYSTEM.get(sgn, "")
    return ""


# ---------------------------------------------------------------------------
# Degeneracy-detection helpers
# ---------------------------------------------------------------------------

# 11 Laue classes by space-group-number range.
_LAUE_RANGES = [
    (1, 2, "-1"), (3, 15, "2/m"), (16, 74, "mmm"),
    (75, 88, "4/m"), (89, 142, "4/mmm"), (143, 148, "-3"),
    (149, 167, "-3m"), (168, 176, "6/m"), (177, 194, "6/mmm"),
    (195, 206, "m-3"), (207, 230, "m-3m"),
]


def _strip_cif_su(raw: str) -> Optional[float]:
    """Parse a CIF numeric value, dropping a standard-uncertainty suffix.

    ``"12.643(2)"`` -> ``12.643``. Returns ``None`` for empty / "?" / unparseable.
    """
    if not raw:
        return None
    cleaned = re.sub(r"\(\d+\)", "", raw).strip()
    if not cleaned or cleaned == "?":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _laue_class_from_sg_number(sg_number: Optional[int]) -> str:
    """Map an IT space-group number (1-230) to its Laue class. ``""`` if unknown."""
    if not sg_number:
        return ""
    for lo, hi, name in _LAUE_RANGES:
        if lo <= sg_number <= hi:
            return name
    return ""


def _centering_from_hm(space_group: str) -> str:
    """Bravais centering letter (P/A/B/C/I/F/R) from a Hermann-Mauguin symbol."""
    if not space_group:
        return ""
    first = space_group.strip()[:1].upper()
    return first if first in ("P", "A", "B", "C", "I", "F", "R") else ""


_ATOM_SYMBOLS = {
    1: "H", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
    29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh",
    46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb",
    52: "Te", 57: "La", 58: "Ce", 72: "Hf", 73: "Ta", 74: "W",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 82: "Pb",
}


def _xtal_element_formula(xtal_path: Path) -> str:
    """Distinct element symbols from a .xtal's Atomtypes (filename-safe, honest).

    Reads ONLY CrystalData/Atomtypes (the per-type Z list) and joins the unique
    element symbols in first-appearance order. Does NOT use Natomtypes (that is
    the number of types, not per-type counts). Returns "" on any failure.
    """
    try:
        import h5py  # noqa: PLC0415
        with h5py.File(str(xtal_path), "r") as f:
            zs = list(f["/CrystalData/Atomtypes"][()])
    except Exception:
        return ""
    out, seen = [], set()
    for z in zs:
        sym = _ATOM_SYMBOLS.get(int(z), f"Z{int(z)}")
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return "".join(out)


def _is_filename_safe_formula(s: str) -> bool:
    """True if s (spaces already stripped) is a clean formula token for a filename."""
    return bool(s) and re.fullmatch(r"[A-Za-z0-9.+-]+", s) is not None


def extract_metadata_from_h5(h5_path: Path) -> PhaseMetadata:
    """Extract metadata from HDF5 Master Pattern or XTAL file."""
    import h5py

    try:
        with h5py.File(str(h5_path), "r") as f:
            if "CrystalData" not in f:
                return PhaseMetadata(
                    formula=h5_path.stem, source="stem",
                    display_label=h5_path.stem,
                )
            cd = f["CrystalData"]

            # Build formula from atom types and counts
            formula = ""
            if "Atomtypes" in cd and "Natomtypes" in cd:
                atoms = cd["Atomtypes"][()]
                counts = cd["Natomtypes"][()]
                parts = []
                for z, n in zip(atoms, counts):
                    sym = _ATOM_SYMBOLS.get(int(z), f"Z{z}")
                    parts.append(f"{sym}{int(n)}" if int(n) > 1 else sym)
                formula = "".join(parts)

            # Space group
            space_group = ""
            sgn = None
            if "SpaceGroupNumber" in cd:
                sgn_raw = cd["SpaceGroupNumber"][()]
                sgn = int(sgn_raw[0]) if hasattr(sgn_raw, '__len__') and len(sgn_raw) > 0 else int(sgn_raw)
                space_group = _SPACE_GROUP_SYMBOLS.get(sgn, f"SG{sgn}")

            # Lattice parameters — EMsoft CrystalData stores them as a
            # 6-element LatticeParameters dataset [a, b, c, alpha, beta, gamma]
            # (lengths in nm, angles in degrees).
            lat = [None] * 6
            if "LatticeParameters" in cd:
                try:
                    raw = list(cd["LatticeParameters"][()])
                    for i in range(min(6, len(raw))):
                        lat[i] = float(raw[i])
                except (TypeError, ValueError):
                    lat = [None] * 6

            meta = PhaseMetadata(
                formula=formula or h5_path.stem,
                space_group=space_group,
                source="h5_crystaldata",
                lattice_a=lat[0], lattice_b=lat[1], lattice_c=lat[2],
                lattice_alpha=lat[3], lattice_beta=lat[4], lattice_gamma=lat[5],
                space_group_number=sgn,
                laue_class=_laue_class_from_sg_number(sgn),
                centering=_centering_from_hm(space_group),
            )
            meta.display_label = _build_display_label(meta)
            return meta

    except Exception:
        return PhaseMetadata(
            formula=h5_path.stem, source="stem",
            display_label=h5_path.stem,
        )


def extract_metadata_from_xtal(xtal_path: Path) -> PhaseMetadata:
    """Extract metadata from XTAL file (same HDF5 structure as Master H5)."""
    meta = extract_metadata_from_h5(xtal_path)
    if meta.source == "h5_crystaldata":
        meta.source = "xtal"
    return meta


# ---------------------------------------------------------------------------
# File linking: SHT/H5 → CIF
# ---------------------------------------------------------------------------

def _formulas_match(formula_a: str, formula_b: str) -> bool:
    """Check if two formula strings refer to the same composition.

    Uses exact match only. Substring matching was removed because it caused
    short formulas like 'Al' to match everything containing 'Al'.
    """
    a = formula_a.strip().lower()
    b = formula_b.strip().lower()
    if not a or not b:
        return False
    return a == b


def find_linked_cif(
    file_path: Path, cif_library_dir: Path
) -> Optional[Path]:
    """Find the CIF file linked to an SHT/H5/XTAL file."""
    stem = file_path.stem
    ext = file_path.suffix.lower()

    # Strategy 1: Direct stem match
    candidate = cif_library_dir / f"{stem}.cif"
    if candidate.exists():
        return candidate

    # Strategy 2: For SHT files, try the phase_name (parenthesized source CIF name)
    if ext == ".sht":
        sht_meta = extract_metadata_from_sht_filename(file_path)
        # The phase_name is the original CIF stem, e.g., "Al7FeCu2" from "Al3FeCu (Al7FeCu2)"
        if sht_meta.phase_name:
            candidate = cif_library_dir / f"{sht_meta.phase_name}.cif"
            if candidate.exists():
                return candidate
        search_formula = sht_meta.formula
        search_phase_name = sht_meta.phase_name
    elif ext in (".h5", ".hdf5"):
        # For H5 master files, extract the material name from filename
        # e.g., "Al6Fe_mp-570001_symmetrized_master_E20kV_npx500.h5" -> "Al6Fe_mp-570001_symmetrized"
        # Try progressively shorter stems
        h5_stem = stem
        for suffix in ("_master_E20kV_npx500", "_E20kV_sig70_n501_o0", "_master"):
            if suffix.lower() in h5_stem.lower():
                idx = h5_stem.lower().index(suffix.lower())
                h5_stem = h5_stem[:idx]
                break
        candidate = cif_library_dir / f"{h5_stem}.cif"
        if candidate.exists():
            return candidate
        search_formula = h5_stem
        search_phase_name = ""
    else:
        return None

    # Strategy 3: Search CIF library by formula exact match
    try:
        for cif_file in sorted(cif_library_dir.glob("*.cif")):
            cif_meta = extract_metadata_from_cif(cif_file)
            if search_formula and _formulas_match(search_formula, cif_meta.formula):
                return cif_file
            if search_phase_name and _formulas_match(search_phase_name, cif_meta.formula):
                return cif_file
            # Also check if the CIF stem matches the phase_name
            if search_phase_name and search_phase_name.lower() == cif_file.stem.lower():
                return cif_file
    except (OSError, PermissionError):
        pass

    return None


# ---------------------------------------------------------------------------
# Main entry point: get_phase_metadata
# ---------------------------------------------------------------------------

def get_phase_metadata(
    file_path: Path, cif_library_dir: Path = None
) -> PhaseMetadata:
    """Get the richest metadata available for a phase file.

    Fallback chain: CIF > SHT filename > H5 CrystalData > XTAL > stem
    """
    ext = file_path.suffix.lower()

    # CIF: parse directly
    if ext == ".cif":
        return extract_metadata_from_cif(file_path)

    # Try to find linked CIF (richest metadata). extract_metadata_from_cif
    # already sets source="cif" and cif_path, so the parsed object is
    # returned directly — this also propagates the lattice/Laue/centering
    # fields the degeneracy detector needs.
    if cif_library_dir:
        linked_cif = find_linked_cif(file_path, cif_library_dir)
        if linked_cif:
            return extract_metadata_from_cif(linked_cif)

    # Method-specific fallbacks
    if ext == ".sht":
        return extract_metadata_from_sht_filename(file_path)
    elif ext in (".h5", ".hdf5"):
        return extract_metadata_from_h5(file_path)
    elif ext == ".xtal":
        return extract_metadata_from_xtal(file_path)

    # Ultimate fallback
    return PhaseMetadata(
        formula=file_path.stem, source="stem",
        display_label=file_path.stem,
    )


# ---------------------------------------------------------------------------
# Pearson symbol derivation (algorithmic)
# ---------------------------------------------------------------------------
# EMsoft's DatabaseConventions.md accepts a Pearson symbol as the [structure
# symbol]. We DERIVE it from the space group + cell so it is ALWAYS available,
# never relying on the CIF carrying _sm_pearson_symbol. EMsoft itself derives
# neither Pearson nor Strukturbericht (the symbol lives only in the filename),
# so deriving it is strictly better than leaving the bracket empty.

# Crystal-FAMILY letter by space-group number. NOTE: the hexagonal family spans
# BOTH the trigonal (143-167) and hexagonal (168-194) crystal systems -> 'h'
# (do NOT use the 7-crystal-system split here).
_PEARSON_FAMILY_RANGES = (
    (1, 2, "a"),      # triclinic (anorthic)
    (3, 15, "m"),     # monoclinic
    (16, 74, "o"),    # orthorhombic
    (75, 142, "t"),   # tetragonal
    (143, 194, "h"),  # trigonal + hexagonal = hexagonal family
    (195, 230, "c"),  # cubic
)

_BRAVAIS_CENTERINGS = frozenset("PABCIFRS")


def derive_pearson(
    sg_number: int,
    hm_symbol: str,
    conventional_natoms: int,
    *,
    strict_S: bool = False,
) -> str:
    """Build the Pearson symbol ``<family><centering><Z>`` (e.g. ``cF4``, ``oC28``).

    Fully determined by the space group + cell, so it never needs the CIF to
    carry ``_sm_pearson_symbol``.

    Args:
        sg_number: IT space-group number (1-230).
        hm_symbol: Hermann-Mauguin symbol; its first letter is the Bravais
            centering (P/A/B/C/I/F/R).
        conventional_natoms: atom count in the CONVENTIONAL unit cell (Z). For
            rhombohedral (R) lattices this is the hexagonal-setting count.
        strict_S: when True, collapse single-face centerings A/B/C -> ``S`` (the
            strict IUCr convention, e.g. ``oS28``). Default keeps the literal
            A/B/C letter (``oC28``), matching AFLOW and common literature.

    Raises:
        ValueError: on an out-of-range space group, a first H-M letter that is
            not a Bravais centering, or a non-positive atom count.
    """
    fam = ""
    for lo, hi, letter in _PEARSON_FAMILY_RANGES:
        if lo <= sg_number <= hi:
            fam = letter
            break
    if not fam:
        raise ValueError(f"space-group number out of range 1-230: {sg_number}")

    cen = (hm_symbol or "").strip()[:1].upper()
    if cen not in _BRAVAIS_CENTERINGS:
        raise ValueError(
            f"H-M symbol does not start with a Bravais centering: {hm_symbol!r}"
        )
    if strict_S and cen in ("A", "B", "C"):
        cen = "S"

    if int(conventional_natoms) <= 0:
        raise ValueError(
            f"conventional_natoms must be positive: {conventional_natoms}"
        )

    return f"{fam}{cen}{int(conventional_natoms)}"


def _derive_pearson_from_cif(cif_path: Path) -> str:
    """Best-effort Pearson symbol for a CIF via pymatgen (sg + conventional Z).

    Returns ``""`` on any failure (pymatgen missing, unparseable or heavily
    disordered CIF), so the SHT-name ``[structure symbol]`` bracket is simply
    omitted rather than wrong. pymatgen is already a converter dependency; the
    import is lazy so it never weighs on ``import phase_metadata``.
    """
    try:
        from pymatgen.core import Structure  # noqa: PLC0415
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: PLC0415

        structure = Structure.from_file(str(cif_path), primitive=False)
        sga = SpacegroupAnalyzer(structure)
        sg_number = int(sga.get_space_group_number())
        hm = sga.get_space_group_symbol()
        conventional_natoms = len(sga.get_conventional_standard_structure())
        return derive_pearson(sg_number, hm, conventional_natoms)
    except Exception:
        return ""


@lru_cache(maxsize=1024)
def _derive_pearson_cached(cif_path_str: str, mtime: float) -> str:
    return _derive_pearson_from_cif(Path(cif_path_str))


def derive_pearson_for_cif(cif_path) -> str:
    """Cached Pearson-from-CIF (keyed by path + mtime). ``""`` on any failure.

    Cheap to call per-phase at discover time after the first hit — the heavy
    pymatgen parse runs once per (file, mtime) and is reused.
    """
    try:
        p = Path(cif_path)
        return _derive_pearson_cached(str(p), p.stat().st_mtime)
    except Exception:
        return ""


@lru_cache(maxsize=1024)
def _composition_cached(cif_path_str: str, mtime: float) -> str:
    try:
        from pymatgen.core import Structure  # noqa: PLC0415
        s = Structure.from_file(cif_path_str, primitive=False)
        return format_formula_subscripts(s.composition.reduced_formula)
    except Exception:
        return ""


def composition_for_cif(cif_path) -> str:
    """Cached Crystal-Database-style composition for a CIF: the pymatgen
    ``reduced_formula`` rendered with subscripts (e.g. ``Al₁₃Fe₄``), IDENTICAL to
    what the Crystal Database table shows. ``""`` on any failure. Cached by
    path + mtime so it is cheap per-phase at discover time."""
    try:
        p = Path(cif_path)
        return _composition_cached(str(p), p.stat().st_mtime)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# SHT filename construction (EMsoft DatabaseConventions.md)
# ---------------------------------------------------------------------------

def resolve_sht_name_fields(
    stem: str,
    *,
    cif_library_dir: Optional[Path] = None,
    xtal_path: Optional[Path] = None,
) -> tuple[str, str]:
    """Return ``(formula, pearson)`` for an SHT filename, READ from files and
    filename-safe. Formula: clean CIF formula → distinct .xtal elements → stem.
    Pearson: CIF ``_sm_pearson_symbol`` → DERIVED from the CIF (space group +
    conventional-cell Z) → "" (brackets omitted). The curated CIF tag wins when
    present; otherwise the symbol is computed so it is almost always available.
    Never invented from thin air — derivation needs a parseable CIF with a cell.
    """
    pearson = ""
    cif = (Path(cif_library_dir) / f"{stem}.cif") if cif_library_dir is not None else None
    if cif is not None and cif.exists():
        meta = extract_metadata_from_cif(cif)
        pearson = meta.pearson or _derive_pearson_from_cif(cif)
        cand = (meta.formula or "").replace(" ", "")
        if _is_filename_safe_formula(cand):
            return cand, pearson
        # The _chemical_formula_sum is often cleaner/more complete than the
        # structural label — try it before dropping to bare elements.
        text = cif.read_text(encoding="utf-8", errors="replace")
        sum_cand = _parse_cif_field(text, "_chemical_formula_sum").replace(" ", "")
        if _is_filename_safe_formula(sum_cand):
            return sum_cand, pearson

    # CIF absent or its formula was not filename-safe → distinct .xtal elements.
    if xtal_path is not None and Path(xtal_path).exists():
        el = _xtal_element_formula(Path(xtal_path))
        if el:
            return el, pearson

    return stem, pearson


def build_sht_filename(
    stem: str,
    formula: str,
    pearson: str,
    voltage_label: str,
    tilt_deg: Optional[float] = 70.0,
) -> str:
    """Assemble ``formula (stem) [pearson] {<kV>kV[ <tilt>deg]}.sht``.

    ``[pearson]`` is omitted when ``pearson`` is empty; the tilt comment is
    added only when ``tilt_deg`` differs from 70°. ``stem`` stays in the
    parentheses so the scan-missing ``f"({stem})" in name`` check still matches.
    """
    sym = f" [{pearson}]" if pearson else ""
    if tilt_deg is None or abs(float(tilt_deg) - 70.0) < 0.5:
        tilt = ""
    else:
        tilt = f" {int(round(float(tilt_deg)))}deg"
    return f"{formula} ({stem}){sym} {{{voltage_label}kV{tilt}}}.sht"
