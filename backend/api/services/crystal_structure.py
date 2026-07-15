"""Parse a .cif or .xtal into a full-unit-cell pymatgen Structure + a JSON
payload for the Database Browser's 3D crystal-structure viewer.

CIF  -> pymatgen CifParser (symmetry already expanded, partial occ preserved).
XTAL -> EMsoft asymmetric unit (read_crystal_structure) expanded via
        Structure.from_spacegroup, then merge_sites(mode="sum") to collapse the
        per-species partial-occupancy duplication EMsoft stores in AtomData.

See docs/superpowers/specs/2026-07-14-cif-xtal-crystal-structure-viewer-design.md
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# CPK / Jmol element colours (hex). Covers the metals/metalloids in this
# project's Al-alloy library + common companions; grey fallback for the rest.
_CPK_COLORS = {
    "H": "#ffffff", "C": "#909090", "N": "#3050f8", "O": "#ff0d0d",
    "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6", "Si": "#f0c8a0",
    "P": "#ff8000", "S": "#ffff30", "Cl": "#1ff01f", "K": "#8f40d4",
    "Ca": "#3dff00", "Ti": "#bfc2c7", "V": "#a6a6ab", "Cr": "#8a99c7",
    "Mn": "#9c7ac7", "Fe": "#e06633", "Co": "#f090a0", "Ni": "#50d050",
    "Cu": "#c88033", "Zn": "#7d80b0", "Ga": "#c28f8f", "Ge": "#668f8f",
    "Zr": "#94e0e0", "Nb": "#73c2c9", "Mo": "#54b5b5", "Ag": "#c0c0c0",
    "Sn": "#668080", "Au": "#ffd123", "Pb": "#575961", "Bi": "#9e4fb5",
}
_COLOR_FALLBACK = "#b0b0b0"

# Cordero (2008) single-bond covalent radii (Angstrom), common subset; 1.25 fallback.
_COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "K": 2.03,
    "Ca": 1.76, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Ag": 1.45, "Sn": 1.39, "Au": 1.36,
    "Pb": 1.46, "Bi": 1.48,
}
_RADIUS_FALLBACK = 1.25


def cpk_color(element: str) -> str:
    """CPK/Jmol hex colour for an element symbol; grey fallback if unknown."""
    return _CPK_COLORS.get(str(element).capitalize(), _COLOR_FALLBACK)


def covalent_radius(element: str) -> float:
    """Covalent radius (Angstrom) for an element symbol; 1.25 fallback."""
    return _COVALENT_RADII.get(str(element).capitalize(), _RADIUS_FALLBACK)


def load_structure(path):
    """Read a .cif or .xtal into a full-unit-cell pymatgen ``Structure``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file cannot be parsed into a structure.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"structure file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".cif":
        return _load_cif(p)
    if suffix == ".xtal":
        return _load_xtal(p)
    raise ValueError(
        f"unsupported structure file type: {p.suffix!r} (need .cif/.xtal)"
    )


def _load_cif(p: Path):
    from pymatgen.io.cif import CifParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = CifParser(str(p))
        get = (
            parser.parse_structures
            if hasattr(parser, "parse_structures")
            else parser.get_structures
        )
        structs = get(primitive=False)
    if not structs:
        raise ValueError(f"no structure could be parsed from CIF {p.name!r}")
    return structs[0]


def _load_xtal(p: Path):
    """Full-cell Structure for the public loader (space group re-derived later)."""
    return _load_xtal_with_sg(p)[0]


def _load_xtal_with_sg(p: Path):
    """Return ``(full-cell Structure, EMsoft space-group number)`` for a .xtal.

    Raises ValueError if the asymmetric unit cannot be expanded/merged.
    """
    import sys

    from pymatgen.core import Element, Lattice, Structure

    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from backend.forward_sim.crystal.xtal_io import read_crystal_structure

    cs = read_crystal_structure(str(p))
    a, b, c, alpha, beta, gamma = cs.lattice  # a,b,c in nm
    lattice = Lattice.from_parameters(a * 10, b * 10, c * 10, alpha, beta, gamma)
    # Round occupancy to shed float32 storage noise: EMsoft stores occ as float32,
    # so e.g. float32(0.8)+float32(0.2) sums to 1.0000000149 after merge_sites and
    # pymatgen rejects the site ("occupancies sum to more than 1"). round(_, 6)
    # restores the exact 1.0 (fixes sd_1401510.xtal / sd_1802610.xtal). from_Z
    # covers the whole periodic table and raises on a genuinely invalid Z.
    species = [
        {Element.from_Z(int(at.Z)).symbol: round(float(at.occ), 6)}
        for at in cs.atoms
    ]
    coords = [list(at.xyz) for at in cs.atoms]
    try:
        s = Structure.from_spacegroup(int(cs.space_group), lattice, species, coords)
        # Collapse EMsoft's per-species partial-occupancy duplicates that now sit
        # on top of each other (sum the occupancies at each shared site).
        s.merge_sites(tol=0.01, mode="sum")
    except Exception as exc:
        raise ValueError(
            f"could not expand XTAL by space group {cs.space_group}: {exc}"
        ) from exc
    return s, int(cs.space_group)


# Bond iff dist < (r_cov_i + r_cov_j) * _BOND_TOL. 1.3 (not the ~1.15 used for
# covalent solids) because covalent radii undercount *metallic* bond lengths:
# Al FCC nearest-neighbour is 2.86 A but (r_cov + r_cov) = 2.42 A, so a tighter
# tolerance leaves metals with zero bonds. 1.3 catches the first coordination
# shell (Al-Al 2.86 < 3.15) while still excluding the second shell (Al 4.05).
_BOND_TOL = 1.3
_MAX_BONDS = 20000  # guard against hairballs in dense metallic cells
_MAX_POLYHEDRA = 400  # guard against a wall of translucent hulls in dense cells


def structure_payload(path) -> dict:
    """Full-cell structure as a JSON-safe dict for the 3D viewer.

    Raises FileNotFoundError (missing) / ValueError (unparseable) — fail loud.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"structure file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".cif":
        struct, known_sg = _load_cif(p), None
    elif suffix == ".xtal":
        # Keep the authoritative EMsoft space-group number so we never have to
        # fabricate one if spglib fails to re-derive it from the geometry.
        struct, known_sg = _load_xtal_with_sg(p)
    else:
        raise ValueError(
            f"unsupported structure file type: {p.suffix!r} (need .cif/.xtal)"
        )
    return _structure_to_payload(struct, source=suffix.lstrip("."), known_sg=known_sg)


def _dominant_symbol(site) -> str:
    """Element symbol of the highest-occupancy species on a (mixed) site."""
    return max(site.species.items(), key=lambda kv: kv[1])[0].symbol


def _structure_to_payload(struct, source: str, known_sg=None) -> dict:
    from pymatgen.core.periodic_table import Element

    lat = struct.lattice
    atoms = []
    for i, site in enumerate(struct):
        species = [
            {"el": sp.symbol, "occ": round(float(occ), 4)}
            for sp, occ in sorted(site.species.items(), key=lambda kv: -kv[1])
        ]
        dom = _dominant_symbol(site)
        atoms.append({
            "label": f"{dom}{i + 1}",
            "element": dom,
            "Z": int(Element(dom).Z),
            "cart": [round(float(v), 4) for v in site.coords],
            "frac": [round(float(v) % 1.0, 5) for v in site.frac_coords],
            "occ": round(float(sum(site.species.values())), 4),
            "species": species,
            "color": cpk_color(dom),
            "radius": round(covalent_radius(dom), 3),
        })

    cell_vectors = [[round(float(v), 4) for v in row] for row in lat.matrix]

    sg_symbol, sg_number = _space_group(struct, known_sg)
    crystal_system = _crystal_system_name(int(sg_number))

    bonds = _compute_bonds(struct)
    polyhedra = _compute_polyhedra(struct)

    return {
        "source": source,
        "lattice": {
            "a": round(float(lat.a), 5), "b": round(float(lat.b), 5),
            "c": round(float(lat.c), 5), "alpha": round(float(lat.alpha), 4),
            "beta": round(float(lat.beta), 4), "gamma": round(float(lat.gamma), 4),
        },
        "cell_vectors": cell_vectors,
        "space_group": {
            "number": int(sg_number), "symbol": str(sg_symbol),
            "crystal_system": crystal_system,
        },
        "atoms": atoms,
        "bonds": bonds,
        "polyhedra": polyhedra,
        "meta": {
            "formula": struct.composition.reduced_formula,
            "n_atoms": len(atoms),
            "disordered": not struct.is_ordered,
            "bonds_truncated": len(bonds) >= _MAX_BONDS,
        },
    }


def _space_group(struct, known_sg=None):
    """Return ``(symbol, number)`` — never fabricate one on failure (fail loud).

    Primary source is spglib via ``get_space_group_info()`` (geometry-derived,
    gives symbol + number together). If that fails and the authoritative number
    is known (XTAL path), fall back to it and look the symbol up. If neither is
    available (CIF whose symmetry can't be derived), raise rather than silently
    claim ``P1`` — a bogus "no symmetry" label would mislabel a real crystal.
    """
    try:
        sym, num = struct.get_space_group_info()
        return str(sym), int(num)
    except Exception as exc:
        if known_sg is None:
            raise ValueError(f"could not determine space group: {exc}") from exc
        num = int(known_sg)
        try:
            from pymatgen.symmetry.groups import SpaceGroup
            sym = SpaceGroup.from_int_number(num).symbol
        except Exception:
            sym = f"#{num}"
        return sym, num


def _crystal_system_name(sg_number: int) -> str:
    table = [
        (2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
        (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
        (230, "cubic"),
    ]
    for hi, name in table:
        if sg_number <= hi:
            return name
    return "unknown"


def _compute_bonds(struct):
    """Distance-cutoff connectivity as explicit Cartesian segments.

    Uses covalent-radii sums (x _BOND_TOL). Includes bonds to periodic images
    (endpoint computed from the neighbour's true fractional coords + offset), so
    atoms at cell faces are not left dangling. Metals have no covalent bonds —
    this is a VESTA-style connectivity aid, not chemistry.
    """
    n = len(struct)
    if n == 0:
        return []
    radii = np.array([covalent_radius(_dominant_symbol(s)) for s in struct])
    r_max = float(radii.max() * 2.0 * _BOND_TOL)
    centers, points, offsets, dists = struct.get_neighbor_list(
        r=r_max, exclude_self=True
    )

    frac = struct.frac_coords
    segs = []
    seen = set()
    for ci, pj, off, d in zip(centers, points, offsets, dists):
        cutoff = (radii[ci] + radii[pj]) * _BOND_TOL
        if d > cutoff:
            continue
        off_t = (int(off[0]), int(off[1]), int(off[2]))
        neg = tuple(-x for x in off_t)
        # De-duplicate the two directions of the same bond. For distinct atoms
        # order by index; for a self-image bond (ci == pj) order by offset sign,
        # so (+off) and (-off) collapse to one segment instead of two mirrors.
        if ci == pj:
            key = (ci, pj, min(off_t, neg))
        else:
            key = (min(ci, pj), max(ci, pj), off_t if ci < pj else neg)
        if key in seen:
            continue
        seen.add(key)
        p1 = struct.lattice.get_cartesian_coords(frac[ci])
        p2 = struct.lattice.get_cartesian_coords(frac[pj] + off)
        segs.append([
            [round(float(v), 4) for v in p1],
            [round(float(v), 4) for v in p2],
        ])
        if len(segs) >= _MAX_BONDS:
            # Never truncate silently — surface the cap so a hairball is visible
            # in the logs rather than looking like a complete bond set.
            logger.warning(
                "crystal_structure: bond list hit the %d cap for a %d-atom cell; "
                "the rendered connectivity is truncated.",
                _MAX_BONDS, n,
            )
            break
    return segs


def _compute_polyhedra(struct):
    """Coordination polyhedra around minority (non-framework) sites.

    The *framework* element is the single most abundant one (Al in an Al alloy);
    every other element is a polyhedron centre. For each centre atom we gather its
    framework neighbours within the bond cutoff and, if there are >= 4, emit their
    Cartesian positions as hull vertices — the frontend builds the convex hull and
    draws a translucent solid (the VESTA / Materials-Project coordination look).

    Returns ``[{element, color, center:[x,y,z], vertices:[[x,y,z], ...]}]``,
    capped at ``_MAX_POLYHEDRA``. Empty for single-element cells.
    """
    from collections import Counter, defaultdict

    n = len(struct)
    if n == 0:
        return []
    doms = [_dominant_symbol(s) for s in struct]
    counts = Counter(doms)
    if len(counts) < 2:
        return []  # single element: no coordination polyhedra
    framework = counts.most_common(1)[0][0]

    radii = np.array([covalent_radius(d) for d in doms])
    r_max = float(radii.max() * 2.0 * _BOND_TOL)
    centers, points, offsets, dists = struct.get_neighbor_list(
        r=r_max, exclude_self=True
    )
    frac = struct.frac_coords

    verts_by_center = defaultdict(list)
    for ci, pj, off, d in zip(centers, points, offsets, dists):
        if doms[ci] == framework:      # only build hulls around non-framework centres
            continue
        if doms[pj] != framework:      # vertices are the coordinating framework atoms
            continue
        if d > (radii[ci] + radii[pj]) * _BOND_TOL:
            continue
        pos = struct.lattice.get_cartesian_coords(frac[pj] + off)
        verts_by_center[int(ci)].append([round(float(v), 4) for v in pos])

    polys = []
    for ci, verts in verts_by_center.items():
        if len(verts) < 4:             # a convex hull needs >= 4 points
            continue
        dom = doms[ci]
        polys.append({
            "element": dom,
            "color": cpk_color(dom),
            "center": [round(float(v), 4) for v in struct[ci].coords],
            "vertices": verts,
        })
        if len(polys) >= _MAX_POLYHEDRA:
            logger.warning(
                "crystal_structure: polyhedra hit the %d cap for a %d-atom cell.",
                _MAX_POLYHEDRA, n,
            )
            break
    return polys
