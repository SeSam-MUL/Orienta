"""Check every ``.xtal`` in the library against the CIF it was made from.

A ``.xtal`` holds an asymmetric unit; what the simulation actually uses is the
orbit of that unit under the space group, and between the two sits every way a
cell can come out wrong — the origin choice, a disorder fallback, a
multi-block CIF whose first block is not the published one. None of that is
visible in the file, and all of it is visible in two numbers: how many atoms
the cell has and what it weighs.

So this expands the ``.xtal`` exactly as the forward simulation does, builds
the CIF's structure with :mod:`backend.forward_sim.crystal.cif_origin`, and
prints them side by side.

    python -m backend.forward_sim.crystal.xtal_audit

It reports, it does not repair: a mismatch means the ``.xtal`` has to be
regenerated, and every master built from it is stale.

**What "ok" does and does not mean.** Both sides of the comparison read the CIF
through :mod:`backend.forward_sim.crystal.cif_origin`, so this is a net for
``.xtal`` files written by the *old* converter, which did not honour the origin
choice. A ``.xtal`` that a future run of the *new* converter gets wrong would
match the new reader's own answer and pass. "35 files, 2 differ" therefore reads
on the other 33 as "consistent with today's reader", not as a clean bill of
health against the literature.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

_AVOGADRO = 6.02214076e23


@dataclass
class AuditRow:
    name: str
    status: str                       # "ok" | "differs" | "no cif" | "error"
    xtal_atoms: int | None = None
    cif_atoms: int | None = None
    xtal_rho: float | None = None
    cif_rho: float | None = None
    xtal_formula: str = ""
    cif_formula: str = ""
    setting: int | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _counts_from_orbit(orbit) -> dict[int, float]:
    counts: dict[int, float] = {}
    for entry in orbit:
        Z = int(entry[0])
        occ = float(entry[1]) if len(entry) > 1 else 1.0
        counts[Z] = counts.get(Z, 0.0) + occ
    return counts


def _formula(counts: dict[int, float]) -> str:
    from diffpy.structure.spacegroups import GetSpaceGroup  # noqa: F401  (keeps the dep explicit)
    from pymatgen.core.periodic_table import Element

    parts = []
    for Z in sorted(counts):
        n = counts[Z]
        parts.append(f"{Element.from_Z(Z).symbol}{n:g}")
    return " ".join(parts)


def _cif_counts(structure) -> dict[int, float]:
    counts: dict[int, float] = {}
    for site in structure:
        for species, occ in site.species.items():
            Z = int(species.Z)
            counts[Z] = counts.get(Z, 0.0) + float(occ)
    return counts


def audit_one(xtal_path: Path, cif_path: Path | None) -> AuditRow:
    from backend.forward_sim.crystal import xtal_io
    from backend.forward_sim.crystal.structure_matrix import _expand_symmetry_orbit

    row = AuditRow(name=xtal_path.name, status="error")
    try:
        structure = xtal_io.read_crystal_structure(str(xtal_path))
        row.setting = int(getattr(structure, "space_group_setting", 1) or 1)
        orbit = _expand_symmetry_orbit(structure)
        counts = _counts_from_orbit(orbit)
        row.xtal_atoms = int(round(sum(counts.values())))
        row.xtal_formula = _formula(counts)
        from backend.forward_sim.mc.composition import mc_composition_from_structure
        row.xtal_rho = float(mc_composition_from_structure(structure).rho)
    except Exception as exc:
        row.note = f"reading the .xtal failed: {type(exc).__name__}: {exc}"
        return row

    if cif_path is None:
        row.status = "no cif"
        row.note = "no CIF of the same name in the library"
        return row

    try:
        from backend.forward_sim.crystal.cif_origin import structure_from_cif

        cif_structure, report = structure_from_cif(cif_path)
        cc = _cif_counts(cif_structure)
        row.cif_atoms = int(round(sum(cc.values())))
        row.cif_formula = _formula(cc)
        row.cif_rho = float(cif_structure.composition.weight
                            / (_AVOGADRO * cif_structure.volume * 1e-24))
        if report.shifted:
            row.note = f"CIF is origin choice 2 ({report.block_key})"
    except Exception as exc:
        row.status = "error"
        row.note = f"reading the CIF failed: {type(exc).__name__}: {exc}"
        return row

    same_atoms = row.xtal_atoms == row.cif_atoms
    same_rho = abs(row.xtal_rho - row.cif_rho) <= 0.01 * max(row.cif_rho, 1e-9)
    same_formula = row.xtal_formula == row.cif_formula
    row.status = "ok" if (same_atoms and same_rho and same_formula) else "differs"
    return row


def _find_cif(stem: str, cif_dir: Path) -> Path | None:
    lowered = stem.lower()
    for candidate in cif_dir.rglob("*.cif"):
        if ".P1-backup-" in candidate.name:
            continue
        if candidate.stem.lower() == lowered:
            return candidate
    return None


def audit_library(xtal_dir: Path, cif_dir: Path) -> list[AuditRow]:
    rows = []
    for xtal in sorted(xtal_dir.rglob("*.xtal")):
        rows.append(audit_one(xtal, _find_cif(xtal.stem, cif_dir)))
    return rows


def _default_database() -> Path:
    return Path(__file__).resolve().parents[3] / "Database"


def main(argv: list[str] | None = None) -> int:
    db = _default_database()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xtal-dir", type=Path, default=db / "XTAL_Library")
    ap.add_argument("--cif-dir", type=Path, default=db / "CIF_Library")
    ap.add_argument("--quiet", action="store_true", help="only print what differs")
    args = ap.parse_args(argv)

    # Two library filenames carry Greek letters (tau, alpha). Windows' console
    # is cp1252 by default, and the table died on the sixth row with
    # UnicodeEncodeError -- on the very command this module documents.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):       # pragma: no cover - not a tty
        pass

    warnings.filterwarnings("ignore")
    # `warn_if_setting_unverified` fires per .xtal read, through logging rather
    # than warnings; three of its paragraphs used to land above a table whose
    # job is to say what is wrong -- including one for a file the table passes.
    logging.getLogger("backend.forward_sim.crystal.xtal_io").setLevel(logging.ERROR)

    rows = audit_library(args.xtal_dir, args.cif_dir)
    width = max((len(r.name) for r in rows), default=10)
    for r in rows:
        if args.quiet and r.ok:
            continue
        head = f"{r.name:{width}}  {r.status:8}"
        if r.status in ("ok", "differs"):
            print(f"{head} xtal {r.xtal_atoms:4} atoms rho {r.xtal_rho:6.3f} | "
                  f"cif {r.cif_atoms:4} atoms rho {r.cif_rho:6.3f}  {r.note}")
            if r.status == "differs":
                print(f"{'':{width}}           xtal: {r.xtal_formula}")
                print(f"{'':{width}}           cif : {r.cif_formula}")
        else:
            print(f"{head} {r.note}")
    bad = [r for r in rows if r.status == "differs"]
    print(f"\n{len(rows)} files, {len(bad)} differ from their CIF, "
          f"{sum(1 for r in rows if r.status == 'no cif')} without a CIF, "
          f"{sum(1 for r in rows if r.status == 'error')} unreadable")
    for r in bad:
        print(f"  regenerate: {r.name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
