"""CIF downloader + validator for the Crystal Hint feature.

Downloads CIF text from a COD (Crystallography Open Database) entry,
validates it with pymatgen, writes it into the local
`Database/CIF_Library/`. Returns metadata + a list of warnings so the
frontend can show a preview before the user commits to an expensive
SHT generation run.

Full SHT generation is NOT done here — that requires a working
WSL+EMsoft toolchain and is a long-running async operation; the user
should run it explicitly from the existing Simulation page.

See docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CIF_LIB_DIR = PROJECT_ROOT / "Database" / "CIF_Library"
DOWNLOAD_TIMEOUT_S = 10.0


@dataclass
class CifMetadata:
    """Parsed metadata from a CIF file."""

    formula: str = ""
    space_group: str = ""
    space_group_number: Optional[int] = None
    crystal_system: str = "unknown"
    a_A: Optional[float] = None
    b_A: Optional[float] = None
    c_A: Optional[float] = None
    alpha_deg: Optional[float] = None
    beta_deg: Optional[float] = None
    gamma_deg: Optional[float] = None
    elements: list[str] = field(default_factory=list)
    n_atoms: int = 0


@dataclass
class CifDownloadResult:
    success: bool
    source: str
    structure_id: str
    local_path: Optional[Path] = None
    metadata: Optional[CifMetadata] = None
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


_FORBIDDEN_NAME = re.compile(r"[^A-Za-z0-9._+\-()\s]")

# Windows reserved device names (case-insensitive, with optional extension).
# Files matching these will fail to create on Windows; prefix with 'phase_'
# to make them safe.
_WIN_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _safe_filename(stem: str, source: str, structure_id: str) -> str:
    """Construct a filesystem-safe CIF filename.

    Convention: `{formula}_{source}-{id}.cif`. Stripped of any unusual
    characters so we never end up with traversal or shell-special filenames.
    Also avoids Windows reserved device names (CON, PRN, AUX, NUL, COM1-9,
    LPT1-9) which cannot be created as files on Windows.
    """
    base = f"{stem.strip()}_{source}-{structure_id}".strip("_")
    base = _FORBIDDEN_NAME.sub("_", base)
    base = re.sub(r"_+", "_", base)
    # Guard against Windows reserved device names. Check the stem (without
    # extension) case-insensitively.
    stem_lower = base.split(".")[0].lower()
    if stem_lower in _WIN_RESERVED:
        base = f"phase_{base}"
    base = base[:120]
    return base + ".cif"


def parse_cif_text(cif_text: str) -> tuple[Optional[CifMetadata], list[str]]:
    """Parse CIF text with pymatgen. Returns (metadata, warnings).

    Returns (None, [warnings_with_error]) on parse failure.
    """
    warnings: list[str] = []
    try:
        from pymatgen.io.cif import CifParser
    except ImportError:
        return None, ["pymatgen not installed — cannot parse CIF"]

    try:
        # CifParser API changed across versions
        from io import StringIO
        try:
            parser = CifParser.from_string(cif_text, occupancy_tolerance=10.0)
        except AttributeError:
            # newer API: CifParser(stream)
            parser = CifParser(StringIO(cif_text), occupancy_tolerance=10.0)
        try:
            structures = parser.parse_structures(primitive=False)
        except (AttributeError, TypeError):
            structures = parser.get_structures(primitive=False)
        if not structures:
            return None, ["pymatgen returned no structures from CIF"]
        s = structures[0]
        try:
            sg_symbol, sg_number = s.get_space_group_info()
        except Exception as exc:
            warnings.append(f"Space group lookup failed: {exc}")
            sg_symbol, sg_number = ("", None)
        meta = CifMetadata(
            formula=s.composition.reduced_formula,
            space_group=str(sg_symbol) if sg_symbol else "",
            space_group_number=int(sg_number) if sg_number else None,
            crystal_system=_system_from_sg(int(sg_number)) if sg_number else "unknown",
            a_A=float(s.lattice.a),
            b_A=float(s.lattice.b),
            c_A=float(s.lattice.c),
            alpha_deg=float(s.lattice.alpha),
            beta_deg=float(s.lattice.beta),
            gamma_deg=float(s.lattice.gamma),
            elements=sorted(set(str(sp.symbol) for sp in s.species)),
            n_atoms=len(s),
        )
        return meta, warnings
    except Exception as exc:
        return None, [f"CIF parse failed: {exc}"]


def _system_from_sg(sg: int) -> str:
    if 1 <= sg <= 2: return "triclinic"
    if 3 <= sg <= 15: return "monoclinic"
    if 16 <= sg <= 74: return "orthorhombic"
    if 75 <= sg <= 142: return "tetragonal"
    if 143 <= sg <= 167: return "trigonal"
    if 168 <= sg <= 194: return "hexagonal"
    if 195 <= sg <= 230: return "cubic"
    return "unknown"


def _validation_warnings(meta: CifMetadata) -> list[str]:
    """Sanity-check the parsed metadata and emit user-facing warnings."""
    out: list[str] = []
    if meta.n_atoms > 200:
        out.append(
            f"Large unit cell ({meta.n_atoms} atoms) — SHT generation will be "
            "SLOW (~1-2 hours), and the EBSD master pattern may be very dense."
        )
    if max(meta.a_A or 0, meta.b_A or 0, meta.c_A or 0) > 30.0:
        out.append(
            f"Lattice parameter > 30 Å — verify the CIF uses the conventional "
            "unit cell, not a supercell."
        )
    if not meta.space_group:
        out.append(
            "Space group missing — SHT generation may fail without explicit symmetry."
        )
    if not meta.elements:
        out.append("No elements detected — CIF may be malformed.")
    return out


def inject_cif_reference(cif_text: str, doi: str = "",
                         reference: str = "") -> str:
    """Return ``cif_text`` with DOI / reference tags added or updated.

    Uses the same CIF tags the Crystal Database reads and writes
    (`_publ_section_doi`, `_publ_section_references`) so injected values
    show up in the View Database / Crystal Parameters panels. Existing tags
    are replaced; missing ones are appended. No-op for empty inputs.

    MP-rendered CIFs (pure pymatgen structures) carry no bibliography, so
    this is how the source reference fetched from MP's provenance endpoint
    gets persisted into the saved file.
    """
    content = cif_text
    doi = (doi or "").strip()
    reference = (reference or "").strip()
    if doi:
        if re.search(r"_publ_section_doi\s+", content):
            content = re.sub(
                r"(_publ_section_doi\s+)['\"]?[^'\"\n]+['\"]?",
                rf"\g<1>'{doi}'", content)
        else:
            content = content.rstrip() + f"\n_publ_section_doi      '{doi}'\n"
    if reference:
        # Strip any stray semicolons so the multiline text block can't be
        # closed early by the content itself.
        ref_clean = reference.replace(";", ",")
        if re.search(r"_publ_section_references\s*;", content):
            content = re.sub(
                r"_publ_section_references\s*;.*?;",
                f"_publ_section_references\n;\n{ref_clean}\n;",
                content, flags=re.DOTALL)
        else:
            content = content.rstrip() + \
                f"\n_publ_section_references\n;\n{ref_clean}\n;\n"
    return content


def download_cif_from_cod(
    cod_id: str,
    save: bool = True,
    overwrite: bool = False,
) -> CifDownloadResult:
    """Fetch a CIF from the Crystallography Open Database.

    Args:
        cod_id: COD numeric ID (e.g. "1010976").
        save: write to Database/CIF_Library/ after validation.
        overwrite: replace existing file with the same name.

    Returns CifDownloadResult. Never raises — network / parse failures
    are captured in the result.
    """
    cod_id = str(cod_id).strip()
    if not cod_id.isdigit():
        return CifDownloadResult(
            success=False,
            source="COD",
            structure_id=cod_id,
            error="COD ID must be numeric",
        )

    url = f"https://www.crystallography.net/cod/{cod_id}.cif"

    try:
        import requests
    except ImportError:
        return CifDownloadResult(
            success=False, source="COD", structure_id=cod_id,
            error="`requests` not installed",
        )

    try:
        r = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT_S,
            headers={"User-Agent": "Orienta/CrystalHint"},
        )
        r.raise_for_status()
    except Exception as exc:
        return CifDownloadResult(
            success=False, source="COD", structure_id=cod_id,
            error=f"Download failed: {exc}",
        )

    cif_text = r.text
    # Sanity check: CIFs have a `data_<name>` block header. COD prefixes
    # a long license/source comment block (~700+ chars), so we scan the
    # first 4 KB instead of the first 500 chars.
    if not cif_text or "data_" not in cif_text[:4096]:
        return CifDownloadResult(
            success=False, source="COD", structure_id=cod_id,
            error="Downloaded content doesn't look like a CIF",
        )

    meta, parse_warnings = parse_cif_text(cif_text)
    if meta is None:
        return CifDownloadResult(
            success=False, source="COD", structure_id=cod_id,
            error="; ".join(parse_warnings) or "Unknown parse failure",
            warnings=parse_warnings,
        )

    val_warnings = _validation_warnings(meta)
    all_warnings = list(parse_warnings) + list(val_warnings)

    local_path = None
    if save:
        CIF_LIB_DIR.mkdir(parents=True, exist_ok=True)
        fname = _safe_filename(meta.formula or "phase", "COD", cod_id)
        local_path = CIF_LIB_DIR / fname
        if local_path.exists() and not overwrite:
            all_warnings.append(
                f"File {fname} already exists in CIF library — keeping the "
                "existing copy. Pass overwrite=true to replace."
            )
        else:
            local_path.write_text(cif_text, encoding="utf-8")
            all_warnings.append(f"Saved to Database/CIF_Library/{fname}")

    return CifDownloadResult(
        success=True,
        source="COD",
        structure_id=cod_id,
        local_path=local_path,
        metadata=meta,
        warnings=all_warnings,
    )


def download_cif_from_mp(
    mp_id: str,
    save: bool = True,
    overwrite: bool = False,
) -> CifDownloadResult:
    """Fetch a CIF from the Materials Project for a given mp-XXXX id.

    Goes through `crystal_hint_mp.fetch_cif_text` which uses the user's
    configured MP API key + pymatgen to render the CIF text. Same
    validate-and-save pipeline as the COD path.

    Args:
        mp_id: Materials Project ID, e.g. "mp-1367" or "mp-985806".
        save: write to Database/CIF_Library/.
        overwrite: replace existing file with the same name.
    """
    mp_id = str(mp_id).strip()
    if not mp_id.startswith("mp-"):
        return CifDownloadResult(
            success=False, source="MP", structure_id=mp_id,
            error="MP ID must start with 'mp-' (e.g. mp-1367)",
        )

    # Lazy import: keeps the COD path free of pymatgen/MP dependencies.
    try:
        from backend.api.services.crystal_hint_mp import fetch_cif_text
    except ImportError as exc:
        return CifDownloadResult(
            success=False, source="MP", structure_id=mp_id,
            error=f"MP module unavailable: {exc}",
        )

    cif_text = fetch_cif_text(mp_id)
    if not cif_text:
        return CifDownloadResult(
            success=False, source="MP", structure_id=mp_id,
            error=(
                "MP fetch returned no CIF — check that the API key is set "
                "in user config and that the MP ID exists."
            ),
        )

    # MP CIFs are bare pymatgen structures with no bibliography. Pull the
    # source reference + DOI from MP's provenance endpoint and embed them
    # so the Crystal Database shows DOI / Reference instead of "click to
    # add". Never fatal — a fetch failure just leaves the CIF reference-less.
    try:
        from backend.api.services.crystal_hint_mp import fetch_mp_reference
        _ref = fetch_mp_reference(mp_id)
        if _ref.get("doi") or _ref.get("reference"):
            cif_text = inject_cif_reference(
                cif_text, doi=_ref.get("doi", ""),
                reference=_ref.get("reference", ""))
    except Exception as exc:
        logger.warning("MP reference embed failed for %s: %s", mp_id, exc)

    meta, parse_warnings = parse_cif_text(cif_text)
    if meta is None:
        return CifDownloadResult(
            success=False, source="MP", structure_id=mp_id,
            error="; ".join(parse_warnings) or "Unknown parse failure",
            warnings=parse_warnings,
        )

    val_warnings = _validation_warnings(meta)
    all_warnings = list(parse_warnings) + list(val_warnings)

    local_path = None
    if save:
        CIF_LIB_DIR.mkdir(parents=True, exist_ok=True)
        fname = _safe_filename(meta.formula or "phase", "MP", mp_id)
        local_path = CIF_LIB_DIR / fname
        if local_path.exists() and not overwrite:
            all_warnings.append(
                f"File {fname} already exists in CIF library — keeping the "
                "existing copy. Pass overwrite=true to replace."
            )
        else:
            local_path.write_text(cif_text, encoding="utf-8")
            all_warnings.append(f"Saved to Database/CIF_Library/{fname}")

    return CifDownloadResult(
        success=True,
        source="MP",
        structure_id=mp_id,
        local_path=local_path,
        metadata=meta,
        warnings=all_warnings,
    )
