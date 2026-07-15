"""
Database Browser API Routes

Manages the local crystal structure database:
- Browse CIF, SHT, H5, XTAL files by scanning local directories
- Build crystal_database.xlsx by parsing all CIF files (pymatgen)
- Read built database entries as JSON
- Add CIF/XTAL files to the database
- Parse CIF file details via orix/diffpy
- LRU cache status
"""

import hashlib
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import re
import sys as _sys

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Ensure project root is on sys.path so path_utils is importable
_project_root_str = str(Path(__file__).resolve().parents[3])
if _project_root_str not in _sys.path:
    _sys.path.insert(0, _project_root_str)

from path_utils import sanitize_filename  # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter()


def _file_hash(path: Path, chunk_size: int = 8192) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Datasets inside .xtal HDF5 that carry the actual scientific data.
# CreationDate, CreationTime, Creator, ProgramName are metadata that
# differ between conversions even when the crystal data is identical.
_XTAL_SCIENCE_KEYS = {
    "CrystalSystem", "SpaceGroupNumber", "SpaceGroupSetting",
    "LatticeParameters", "Natomtypes", "Atomtypes", "AtomData",
}


def _xtal_science_hash(path: Path) -> str:
    """SHA-256 hash of only the scientific datasets inside a .xtal HDF5 file.

    Ignores metadata like CreationDate/Time/Creator so that two files
    produced from the same CIF at different times compare as equal.
    Rounds floats to 4 decimal places to tolerate minor precision
    differences between library versions (pymatgen, spglib, h5py).
    Falls back to full-file hash if HDF5 reading fails.
    """
    try:
        import h5py
        import numpy as np
        h = hashlib.sha256()
        with h5py.File(str(path), "r") as f:
            cd = f.get("CrystalData")
            if cd is None:
                return _file_hash(path)
            for key in sorted(_XTAL_SCIENCE_KEYS):
                ds = cd.get(key)
                if ds is None:
                    continue
                h.update(key.encode("utf-8"))
                val = np.asarray(ds[()])
                if np.issubdtype(val.dtype, np.floating):
                    # Round floats so minor precision diffs don't cause conflicts
                    val = np.round(val, 4)
                h.update(val.tobytes())
        return h.hexdigest()
    except Exception:
        return _file_hash(path)


def _build_xtal_atomdata(rep_sites, dwf_map):
    """Build EMsoft .xtal ``(Atomtypes, AtomData, warnings)`` from asymmetric-unit sites.

    A DISORDERED (mixed) site emits ONE atom PER co-occupying species — same
    coordinates, each with its own occupancy + Debye-Waller factor — so minority
    elements that share a site (e.g. Mn/Si on an Al position) are PRESERVED.
    The previous logic collapsed a mixed site to its first-listed species and
    handed it the summed occupancy, silently dropping every other element
    (verified: alpha-AlFeMnSi lost Mn+Si, Al3FeSi2 lost Si). EMsoft supports
    fractional site occupancy for solid solutions, so this is the correct
    representation.

    Args:
        rep_sites: list of representative ``PeriodicSite`` objects (one per
            asymmetric-unit orbit).
        dwf_map: ``{element_symbol: B_nm2}`` Debye-Waller lookup (default 0.005).

    Returns:
        ``(Z_arr int32 (N,), atomdata float32 (5,N)=[x,y,z,occ,B], warnings list)``.
    """
    import numpy as np  # noqa: PLC0415  (numpy is imported lazily in this module)

    z_list, col_coords, occ_list, dwf_list = [], [], [], []
    for site in rep_sites:
        fc = site.frac_coords
        if site.is_ordered:
            el = site.specie
            z_list.append(int(el.number))
            col_coords.append(fc)
            occ_list.append(1.0)
            dwf_list.append(dwf_map.get(el.symbol, 0.005))
        else:
            for el, occ in site.species.items():
                z_list.append(int(el.number))
                col_coords.append(fc)
                occ_list.append(float(occ))
                dwf_list.append(dwf_map.get(el.symbol, 0.005))

    Z_arr = np.array(z_list, dtype=np.int32)
    coords = np.array(col_coords).T  # 3×N
    atomdata = np.vstack([coords, occ_list, dwf_list]).astype(np.float32)

    warnings_out = []
    if occ_list:
        min_occ = min(occ_list)
        if min_occ < 0.5:
            warnings_out.append(
                f"Severe site disorder (min occupancy {min_occ:.3f}). The master "
                "simulation may be very slow or unstable for such structures."
            )
        elif min_occ < 0.98:
            warnings_out.append(
                f"Partial site disorder (min occupancy {min_occ:.3f})."
            )
    return Z_arr, atomdata, warnings_out


# Database directory layout relative to project root.
# Folder names MUST match path_utils.DATABASE_SUBFOLDERS exactly (incl. casing):
# on case-sensitive filesystems (Linux/macOS) a casing mismatch silently splits
# the library into two folders. "CIF_Library" is the canonical on-disk name.
_DB_SUBDIRS = {
    "cif":  "Database/CIF_Library",
    "xtal": "Database/XTAL_Library",
    "sht":  "Database/EBSD_SHT_Database",
    "h5":   "Database/EBSD_H5_Cache",
}


def _project_root() -> Path:
    """Return the project root directory (two levels above this file's package)."""
    # This file is at  <root>/backend/api/routes/database.py
    return Path(__file__).resolve().parents[3]


def _db_dir(category: str) -> Path:
    """Return the absolute Path for a given category subfolder."""
    subdir = _DB_SUBDIRS.get(category)
    if subdir is None:
        raise ValueError(f"Unknown category: {category}")
    return _project_root() / subdir


def _scan_directory(dirpath: Path, extensions: set) -> list:
    """Recursively scan a directory for files matching given extensions."""
    entries = []
    if not dirpath.is_dir():
        return entries
    try:
        for item in sorted(dirpath.rglob("*")):
            if item.is_file() and item.suffix.lower() in extensions:
                try:
                    stat = item.stat()
                    entries.append({
                        "name": item.name,
                        "path": str(item),
                        "size": stat.st_size,
                        # Empty when the file sits directly in the category dir
                        # (no material subfolder). The UI shows "—" for empty;
                        # a literal "-" here used to leak into delete/cascade and
                        # made them look in a nonexistent "<dir>/-/" subfolder.
                        "material": item.parent.name if item.parent != dirpath else "",
                    })
                except OSError:
                    continue
    except (OSError, PermissionError) as exc:
        logger.warning("Error scanning %s: %s", dirpath, exc)
    return entries


class BrowseRequest(BaseModel):
    category: str = "all"  # "cif", "sht", "h5", "xtal", "all"
    material_filter: str = ""


class DownloadRequest(BaseModel):
    file_path: str
    target_category: str = "cif"


class AddCifRequest(BaseModel):
    file_path: str          # Source path of the CIF file to import
    material_name: str = "" # Optional subfolder / material name


class AddXtalRequest(BaseModel):
    file_path: str
    material_name: str = ""


class DwfUpdate(BaseModel):
    element: str
    dwb_300k: float
    reference: str = ""  # literature source (provenance); upsert adds new elements


class UpdateDwfRequest(BaseModel):
    updates: list[DwfUpdate]


class DeleteFileEntry(BaseModel):
    path: str
    category: str  # "cif", "xtal", "h5", "master", "sht", "dictionary"
    material: str = ""  # for resolving subfolder (e.g. "Fe" → EBSD_H5_Cache/Fe/)


class DeleteFilesRequest(BaseModel):
    files: list[DeleteFileEntry]
    # New: 3-way scope. Falls back from legacy delete_from_server if delete_from absent.
    delete_from: str = ""           # "local", "server", "everywhere"
    delete_from_server: bool = False  # Legacy compat for Crystal DB


# --- Cascade resolve models ---------------------------------------------------

class CascadeFileEntry(BaseModel):
    name: str
    category: str        # "h5", "master", "sht", "dictionary"
    material: str = ""


class CascadeFileInfo(BaseModel):
    name: str
    category: str
    material: str
    location: str        # "local", "server", "both"
    size: int = 0
    reason: str = ""     # e.g. "Master pattern from Fe_E20kV.h5"
    required: bool = False  # True = cannot be unchecked (dict from master)


class CascadeRequest(BaseModel):
    files: list[CascadeFileEntry]


class CascadeResponse(BaseModel):
    selected: list[CascadeFileInfo]
    associated: list[CascadeFileInfo]
    has_local: bool
    has_server: bool


class BatchInfoRequest(BaseModel):
    filenames: list[str]


class SyncResult(BaseModel):
    uploaded: list[str] = []
    downloaded: list[str] = []
    conflicts: list[dict] = []
    errors: list[str] = []
    up_to_date: list[str] = []


class ResolveConflictRequest(BaseModel):
    filename: str
    file_type: str  # "cif" or "xtal"
    action: str     # "keep_local", "keep_server", "keep_both", "skip"


class ResolveConflictsRequest(BaseModel):
    resolutions: list[ResolveConflictRequest]


def _get_server_db_root() -> Optional[Path]:
    """Return the server database root if server mode is enabled + reachable.

    Reads the per-machine user config — the SAME store the Settings UI writes
    via ``user_config_manager`` (``%APPDATA%/Kikuchipy/config.json`` on Windows,
    ``~/.config/Kikuchipy/config.json`` on Linux, …).

    Previously this read ``<project_root>/data/server_config.json`` directly.
    But that legacy file is migrated into the per-machine config and DELETED on
    first load (see user_config_manager), so after migration it never existed
    again — and every Database Browser sync/browse silently reported
    "Server mode not configured or not connected" even while the Settings page
    showed the server as connected.
    """
    try:
        from backend.api.services import user_config_manager as uc
        server = uc.get_section("server")
        if not server.get("enabled") or server.get("offline_mode"):
            return None
        root = server.get("database_root", "")
        if root:
            p = Path(root)
            if p.is_dir():
                return p
    except Exception as exc:
        logger.debug("Could not read server config: %s", exc)
    return None


# Map local DB subdir names to server equivalents
_SERVER_SUBDIRS = {
    "cif":  "EBSD_CIF_Library",
    "xtal": "EBSD_XTAL_Library",
    "sht":  "EBSD_SHT_Database",
    "h5":   "EBSD_H5_Cache",
}


def _classify_h5(filename: str) -> str:
    """Classify an .h5 file into 'dictionary', 'master', or 'h5' (Monte-Carlo).

    Dictionary files (``*_dict_*.h5``, derived from a master) are checked FIRST,
    because their names often ALSO contain '_master' (e.g.
    ``Al_master_E20kV_npx500_dict_20kV_…``). Without this they were mislabelled
    'master' or — when no '_master' token was present — 'h5' (MC), which leaked
    them into the MC h5 tab and into bulk upload.
    """
    low = filename.lower()
    if "_dict_" in low:
        return "dictionary"
    if "_master" in low or "master_" in low:
        return "master"
    return "h5"


def _find_associated_files(
    selected_names: set,
    entries: list[CascadeFileEntry],
    server_root: Optional[Path],
) -> list[CascadeFileInfo]:
    """Discover files associated with the selected set via cascade logic.

    For each selected file:
    - MC .h5   → look for matching master .h5 (optional) + .sht (optional)
                 then transitively: master → dictionaries (required)
    - Master .h5 → look for matching *_dict_*.h5 + .json sidecar (required)

    Returns deduplicated CascadeFileInfo entries not already in selected_names.
    """
    def _loc(local_path: Optional[Path], server_path: Optional[Path]) -> tuple[str, int]:
        """Return (location, size) for a discovered file."""
        has_local = local_path is not None and local_path.is_file()
        has_server = server_path is not None and server_path.is_file()
        if has_local and has_server:
            return "both", local_path.stat().st_size
        if has_local:
            return "local", local_path.stat().st_size
        if has_server:
            return "server", server_path.stat().st_size
        return "local", 0

    def _add(result: dict, info: CascadeFileInfo):
        """Add to result dict only if not already seen (deduplicate by name+category)."""
        key = (info.name, info.category)
        if key not in result:
            result[key] = info

    result: dict = {}

    # Local directory roots
    proj = _project_root()
    local_h5_root = proj / "Database" / "EBSD_H5_Cache"
    local_sht_root = proj / "Database" / "EBSD_SHT_Database"
    local_dict_root = proj / "Database" / "Dictionary_Library"

    def _extract_base_stem(mc_name: str) -> str:
        """Extract base material stem from MC filename using anchored regex.

        E.g. 'Fe_E20kV.h5' → 'Fe', 'Al_ECAP_E15kV.h5' → 'Al_ECAP'
        """
        mc_stem = Path(mc_name).stem
        m = re.match(r'^(.+?)_E\d+kV', mc_stem)
        return m.group(1) if m else mc_stem

    def _find_master_in_dir(base_stem: str, search_dir: Path) -> Optional[Path]:
        """Find a master .h5 matching base_stem in a directory.

        Matches files like {base_stem}_master*.h5 — the stem must appear
        at the START of the filename to avoid false positives (e.g. "Al"
        should not match "Al6Fe_master...").
        """
        if not search_dir.is_dir():
            return None
        safe_base = sanitize_filename(base_stem).lower()
        for c in search_dir.iterdir():
            if not (c.is_file() and c.suffix == ".h5" and "_master" in c.name.lower()):
                continue
            c_lower = c.stem.lower()
            # Stem must start with base_stem (exact prefix match)
            if c_lower.startswith(safe_base + "_master") or c_lower.startswith(base_stem.lower() + "_master"):
                return c
        return None

    def _find_sht_for_stem(base_stem: str, material: str) -> tuple[Optional[Path], Optional[Path]]:
        safe_base = sanitize_filename(base_stem)
        search_dir = local_sht_root / material if material else local_sht_root
        local_sht = None
        if search_dir.is_dir():
            for c in search_dir.rglob("*.sht"):
                cs = sanitize_filename(c.stem)
                if cs.lower() == safe_base.lower() or base_stem.lower() in c.stem.lower():
                    local_sht = c
                    break
        server_sht = None
        if server_root:
            srv_dir = server_root / "EBSD_SHT_Database" / material if material else server_root / "EBSD_SHT_Database"
            if srv_dir.is_dir():
                for c in srv_dir.rglob("*.sht"):
                    cs = sanitize_filename(c.stem)
                    if cs.lower() == safe_base.lower() or base_stem.lower() in c.stem.lower():
                        server_sht = c
                        break
        return local_sht, server_sht

    def _find_dicts_for_master(master_name: str, material: str) -> list[CascadeFileInfo]:
        """Find dictionary .h5 and .json sidecars matching a master pattern."""
        master_stem = Path(master_name).stem  # e.g. "Fe_master_E20kV"
        safe_master = sanitize_filename(master_stem)
        found = []

        local_dir = local_dict_root / material if material else local_dict_root
        server_dir = (
            (server_root / "EBSD_Dictionary_Library" / material) if (server_root and material)
            else (server_root / "EBSD_Dictionary_Library" if server_root else None)
        )

        # Gather all candidate names from both sides
        local_candidates: dict[str, Path] = {}
        if local_dir.is_dir():
            for c in local_dir.rglob("*"):
                if c.is_file() and c.suffix.lower() in {".h5", ".json"}:
                    safe = sanitize_filename(c.stem)
                    if "_dict_" in c.name.lower() and (
                        safe_master.lower() in safe.lower()
                        or master_stem.lower() in c.stem.lower()
                    ):
                        local_candidates[c.name] = c

        server_candidates: dict[str, Path] = {}
        if server_dir and server_dir.is_dir():
            for c in server_dir.rglob("*"):
                if c.is_file() and c.suffix.lower() in {".h5", ".json"}:
                    safe = sanitize_filename(c.stem)
                    if "_dict_" in c.name.lower() and (
                        safe_master.lower() in safe.lower()
                        or master_stem.lower() in c.stem.lower()
                    ):
                        server_candidates[c.name] = c

        all_names = set(local_candidates) | set(server_candidates)
        for fname in all_names:
            lp = local_candidates.get(fname)
            sp = server_candidates.get(fname)
            loc, sz = _loc(lp, sp)
            cat = "dictionary"
            found.append(CascadeFileInfo(
                name=fname,
                category=cat,
                material=material,
                location=loc,
                size=sz,
                reason=f"Dictionary from {master_name}",
                required=True,
            ))
        return found

    for entry in entries:
        material = entry.material
        name = entry.name

        if entry.category in ("h5",):
            # MC .h5 → find master (optional) + sht (optional), then master → dicts (required)
            base_stem = _extract_base_stem(name)

            # SHT
            local_sht, server_sht = _find_sht_for_stem(base_stem, material)
            if local_sht or server_sht:
                fname = local_sht.name if local_sht else server_sht.name
                if fname not in selected_names:
                    loc, sz = _loc(local_sht, server_sht)
                    _add(result, CascadeFileInfo(
                        name=fname, category="sht", material=material,
                        location=loc, size=sz,
                        reason=f"SHT for {name}", required=False,
                    ))

            # Master .h5
            local_h5_dir = local_h5_root / material if material else local_h5_root
            server_h5_dir = (server_root / "EBSD_H5_Cache" / material) if server_root and material else (server_root / "EBSD_H5_Cache" if server_root else None)
            local_master = _find_master_in_dir(base_stem, local_h5_dir)
            server_master = _find_master_in_dir(base_stem, server_h5_dir) if server_h5_dir else None
            if local_master or server_master:
                mname = local_master.name if local_master else server_master.name
                if mname not in selected_names:
                    loc, sz = _loc(local_master, server_master)
                    _add(result, CascadeFileInfo(
                        name=mname, category="master", material=material,
                        location=loc, size=sz,
                        reason=f"Master pattern from {name}", required=False,
                    ))
                # Transitively: master → dictionaries
                for dict_info in _find_dicts_for_master(mname, material):
                    if dict_info.name not in selected_names:
                        _add(result, dict_info)

        elif entry.category == "master":
            # Master .h5 → find dictionaries (required)
            for dict_info in _find_dicts_for_master(name, material):
                if dict_info.name not in selected_names:
                    _add(result, dict_info)

    return list(result.values())


@router.get("/browse")
async def browse(category: str = "all", material: str = ""):
    """Browse the crystal structure database by scanning local AND server directories."""
    _category_extensions = {
        "cif":  {".cif"},
        "xtal": {".xtal"},
        "sht":  {".sht"},
        "h5":   {".h5"},
    }

    categories_to_scan = (
        list(_category_extensions.keys()) if category == "all" else [category]
    )

    # Merge local + server entries by name to detect "both" locations
    # Key: (category, filename) → entry dict
    merged: dict = {}

    server_root = _get_server_db_root()

    for cat in categories_to_scan:
        if cat not in _category_extensions:
            continue

        # --- Scan LOCAL ---
        try:
            dirpath = _db_dir(cat)
        except ValueError:
            dirpath = None

        if dirpath:
            raw = _scan_directory(dirpath, _category_extensions[cat])
            for item in raw:
                if material and material.lower() not in item["name"].lower() and \
                        material.lower() not in item["material"].lower():
                    continue
                file_cat = _classify_h5(item["name"]) if cat == "h5" else cat
                key = (file_cat, item["name"])
                merged[key] = {
                    "name": item["name"],
                    "path": item["path"],
                    "category": file_cat,
                    "material": item["material"],
                    "size": item["size"],
                    "location": "local",
                }

        # --- Scan SERVER ---
        if server_root:
            server_subdir = server_root / _SERVER_SUBDIRS.get(cat, "")
            if server_subdir.is_dir():
                raw_server = _scan_directory(server_subdir, _category_extensions[cat])
                for item in raw_server:
                    if material and material.lower() not in item["name"].lower() and \
                            material.lower() not in item["material"].lower():
                        continue
                    file_cat = _classify_h5(item["name"]) if cat == "h5" else cat
                    key = (file_cat, item["name"])
                    if key in merged:
                        # File exists both locally and on server
                        merged[key]["location"] = "both"
                    else:
                        merged[key] = {
                            "name": item["name"],
                            "path": item["path"],
                            "category": file_cat,
                            "material": item["material"],
                            "size": item["size"],
                            "location": "server",
                        }

    # --- Scan the Dictionary_Library (its own folder, not covered above) ---
    # Dictionary .h5 files that physically live in Dictionary_Library are tagged
    # 'dictionary' so they get their own tab and stay OUT of the MC h5 / Master
    # tabs (and out of the default bulk upload). Dict files that ended up in
    # EBSD_H5_Cache are already reclassified by _classify_h5 above.
    if category in ("all", "dictionary"):
        local_dict = _project_root() / "Database" / "Dictionary_Library"
        if local_dict.is_dir():
            for item in _scan_directory(local_dict, {".h5"}):
                if material and material.lower() not in item["name"].lower() and \
                        material.lower() not in item["material"].lower():
                    continue
                key = ("dictionary", item["name"])
                merged[key] = {
                    "name": item["name"],
                    "path": item["path"],
                    "category": "dictionary",
                    "material": item["material"],
                    "size": item["size"],
                    "location": "local",
                }
        if server_root:
            server_dict = server_root / "EBSD_Dictionary_Library"
            if server_dict.is_dir():
                for item in _scan_directory(server_dict, {".h5"}):
                    if material and material.lower() not in item["name"].lower() and \
                            material.lower() not in item["material"].lower():
                        continue
                    key = ("dictionary", item["name"])
                    if key in merged:
                        merged[key]["location"] = "both"
                    else:
                        merged[key] = {
                            "name": item["name"],
                            "path": item["path"],
                            "category": "dictionary",
                            "material": item["material"],
                            "size": item["size"],
                            "location": "server",
                        }

    result = list(merged.values())

    # Cross-reference CIF entries with XTAL library for has_xtal flag
    xtal_dir = _db_dir("xtal")
    xtal_stems = set()
    if xtal_dir.is_dir():
        for f in xtal_dir.rglob("*.xtal"):
            xtal_stems.add(f.stem.lower())
    for entry in result:
        if entry.get("category") == "cif":
            cif_stem = Path(entry["name"]).stem.lower()
            entry["has_xtal"] = cif_stem in xtal_stems

    # Compute per-category counts
    counts = {}
    for entry in result:
        c = entry.get("category", "unknown")
        counts[c] = counts.get(c, 0) + 1

    return {"entries": result, "category_counts": counts}


@router.post("/add-cif")
async def add_cif(req: AddCifRequest):
    """Copy a CIF file into the local CIF library, with duplicate detection."""
    src = Path(req.file_path)
    if not src.is_file():
        raise HTTPException(status_code=400, detail=f"Source file not found: {req.file_path}")
    if src.suffix.lower() != ".cif":
        raise HTTPException(status_code=400, detail="File must have a .cif extension")

    try:
        # --- Duplicate detection ---
        existing_matches = list(_db_dir("cif").rglob(src.name))
        if existing_matches:
            src_hash = _file_hash(src)
            for existing in existing_matches:
                if _file_hash(existing) == src_hash:
                    return {
                        "success": True,
                        "name": src.name,
                        "path": str(existing),
                        "category": "cif",
                        "duplicate": True,
                        "message": "File already exists in database (identical content)",
                    }
            # Same name, different content
            return {
                "success": True,
                "name": src.name,
                "path": str(existing_matches[0]),
                "category": "cif",
                "duplicate": True,
                "different_content": True,
                "message": "File with same name exists but content differs",
            }

        # No duplicate — proceed with copy
        target_dir = _db_dir("cif")
        if req.material_name.strip():
            target_dir = target_dir / req.material_name.strip()
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / src.name
        shutil.copy2(str(src), str(dest))
        return {
            "success": True,
            "name": src.name,
            "path": str(dest),
            "category": "cif",
            "duplicate": False,
        }
    except Exception as e:
        logger.exception("Failed to add CIF file")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cif/{filename}/info")
async def cif_info(filename: str):
    """Parse and return details for a CIF file in the library."""
    import re as _re
    import html as _html

    # Search in the CIF library directory
    cif_dir = _db_dir("cif")
    matches = list(cif_dir.rglob(filename))
    if not matches:
        raise HTTPException(status_code=404, detail=f"CIF file '{filename}' not found in library")

    cif_path = matches[0]
    info: dict = {
        "filename": filename,
        "path": str(cif_path),
        "size": cif_path.stat().st_size,
    }

    # --- Extract DOI and reference from raw CIF text ---
    # Important: read BOTH independently. An earlier version fell into an
    # if/else ladder that set reference="" whenever a DOI was also present,
    # which made the "Reference" field in the GUI appear to be deleted every
    # time the user saved a DOI — the text was still in the CIF on disk, it
    # just wasn't reported back.
    try:
        raw_text = cif_path.read_text(encoding="utf-8", errors="ignore")
        dois = _re.findall(
            r"(?:_publ_section_doi|_citation_doi|_journal_paper_doi)\s+['\"]?([^'\"\s]+)['\"]?",
            raw_text,
        )
        info["doi"] = dois[0].strip() if dois else ""
        ref_matches = _re.findall(
            r"_publ_section_references\s*;\s*(.*?)\s*;", raw_text, _re.DOTALL
        )
        info["reference"] = (
            _re.sub(r"\s+", " ", _html.unescape(ref_matches[0])).strip()
            if ref_matches
            else ""
        )
    except Exception:
        info["doi"] = ""
        info["reference"] = ""

    # Try to parse with orix
    is_ordered = True
    try:
        from orix.crystal_map import Phase
        from ebsd_utils import sanitize_cif
        phase = Phase.from_cif(sanitize_cif(str(cif_path)))
        # Override name if sanitize_cif created a temp file (temp name != original)
        if phase.name != cif_path.stem:
            phase.name = cif_path.stem
        info["phase_name"] = str(phase.name)
        info["space_group"] = str(phase.space_group) if hasattr(phase, "space_group") else ""
        if hasattr(phase, "space_group") and phase.space_group is not None:
            sg = phase.space_group
            info["space_group_number"] = int(sg.number) if hasattr(sg, "number") else None
            info["crystal_system"] = str(sg.crystal_system) if hasattr(sg, "crystal_system") else None
        if hasattr(phase, "structure") and phase.structure is not None:
            struct = phase.structure
            is_ordered = struct.is_ordered if hasattr(struct, "is_ordered") else True
            lat = struct.lattice
            info["lattice"] = {
                "a": float(lat.a),
                "b": float(lat.b),
                "c": float(lat.c),
                "alpha": float(lat.alpha),
                "beta": float(lat.beta),
                "gamma": float(lat.gamma),
            }
            # Extract atom sites for DWF table
            if len(struct) > 0:
                atoms = []
                for site in struct:
                    atoms.append({
                        "element": str(site.element),
                        "x": round(float(site.x), 5),
                        "y": round(float(site.y), 5),
                        "z": round(float(site.z), 5),
                        "occ": round(float(site.occupancy), 4) if hasattr(site, "occupancy") else 1.0,
                        "dwf": round(float(site.Bisoequiv), 4) if hasattr(site, "Bisoequiv") else 0.5,
                    })
                info["atoms"] = atoms
                info["n_atoms"] = len(atoms)
    except Exception as exc:
        info["parse_warning"] = str(exc)

    # --- Compute fit_for_xtal (same criteria as cif_database_builder) ---
    fit_warnings = []
    fit = True
    if not info.get("doi") and not info.get("reference"):
        fit = False
        fit_warnings.append("No reference or DOI found.")
    if not info.get("n_atoms"):
        fit = False
        fit_warnings.append("No atomic sites.")
    sg_num = info.get("space_group_number")
    if sg_num is None or sg_num < 1:
        fit = False
        fit_warnings.append("No space group.")
    if not is_ordered:
        fit_warnings.append("Disordered structure.")
    info["fit_for_xtal"] = fit
    info["fit_warnings"] = fit_warnings

    return info


class UpdateCifMetaRequest(BaseModel):
    doi: str = ""
    reference: str = ""


@router.patch("/cif/{filename}/meta")
async def update_cif_meta(filename: str, req: UpdateCifMetaRequest):
    """Write DOI and/or reference text into a CIF file."""
    import re as _re

    cif_dir = _db_dir("cif")
    matches = list(cif_dir.rglob(filename))
    if not matches:
        raise HTTPException(status_code=404, detail=f"CIF file '{filename}' not found")

    cif_path = matches[0]
    try:
        content = cif_path.read_text(encoding="utf-8", errors="ignore")
        updated = False

        # Update or append DOI
        if req.doi.strip():
            doi_val = req.doi.strip()
            if _re.search(r"_publ_section_doi\s+", content):
                content = _re.sub(
                    r"(_publ_section_doi\s+)['\"]?[^'\"\n]+['\"]?",
                    rf"\g<1>'{doi_val}'",
                    content,
                )
            else:
                content += f"\n_publ_section_doi      '{doi_val}'\n"
            updated = True

        # Update or append reference text
        if req.reference.strip():
            ref_val = req.reference.strip()
            if _re.search(r"_publ_section_references\s*;", content):
                content = _re.sub(
                    r"_publ_section_references\s*;.*?;",
                    f"_publ_section_references\n;\n{ref_val}\n;",
                    content,
                    flags=_re.DOTALL,
                )
            else:
                content += f"\n_publ_section_references\n;\n{ref_val}\n;\n"
            updated = True

        if updated:
            cif_path.write_text(content, encoding="utf-8")

            # Also update crystal_database.xlsx so View Database shows current values
            xlsx_path = _project_root() / "Database" / "crystal_database.xlsx"
            if xlsx_path.is_file():
                try:
                    import pandas as pd
                    df = pd.read_excel(str(xlsx_path)).fillna("")
                    # Find row by CIF filename
                    name_col = None
                    for col in df.columns:
                        if col.lower() in ("cif file name", "cif_file_name", "filename"):
                            name_col = col
                            break
                    if name_col:
                        mask = df[name_col].astype(str) == filename
                        if mask.any():
                            if req.doi.strip() and "DOI" in df.columns:
                                df.loc[mask, "DOI"] = req.doi.strip()
                            if req.reference.strip() and "Reference Text" in df.columns:
                                df.loc[mask, "Reference Text"] = req.reference.strip()
                            df.to_excel(str(xlsx_path), index=False)
                            logger.info(f"Updated database entry for {filename}")
                except Exception as exc:
                    logger.warning(f"Failed to update crystal_database.xlsx: {exc}")

        return {"updated": updated, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/xtal/{stem}/info")
async def xtal_info(stem: str):
    """Read and return the contents of a .xtal HDF5 file."""
    xtal_dir = _db_dir("xtal")
    # stem may or may not have .xtal extension
    search_name = stem if stem.lower().endswith(".xtal") else f"{stem}.xtal"
    matches = list(xtal_dir.rglob(search_name))
    if not matches:
        raise HTTPException(status_code=404, detail=f"XTAL file '{search_name}' not found")

    xtal_path = matches[0]
    try:
        import h5py
        import numpy as np

        info: dict = {
            "filename": xtal_path.name,
            "path": str(xtal_path),
            "size": xtal_path.stat().st_size,
        }

        with h5py.File(str(xtal_path), "r") as f:
            cd = f.get("CrystalData")
            if cd is None:
                info["error"] = "No CrystalData group found"
                return info

            # Scalar datasets
            for key in ["CrystalSystem", "SpaceGroupNumber", "SpaceGroupSetting", "Natomtypes"]:
                ds = cd.get(key)
                if ds is not None:
                    info[key] = int(np.asarray(ds[()]).flat[0])

            # Lattice parameters (6 floats: a, b, c in nm, alpha, beta, gamma in degrees)
            lp = cd.get("LatticeParameters")
            if lp is not None:
                vals = np.asarray(lp[()]).flatten()
                if len(vals) == 6:
                    info["lattice"] = {
                        "a": round(float(vals[0]), 5),
                        "b": round(float(vals[1]), 5),
                        "c": round(float(vals[2]), 5),
                        "alpha": round(float(vals[3]), 4),
                        "beta": round(float(vals[4]), 4),
                        "gamma": round(float(vals[5]), 4),
                    }
                    # Also provide in Angstrom for display
                    info["lattice_angstrom"] = {
                        "a": round(float(vals[0]) * 10, 4),
                        "b": round(float(vals[1]) * 10, 4),
                        "c": round(float(vals[2]) * 10, 4),
                    }

            # Atom types (Z numbers)
            at = cd.get("Atomtypes")
            atomtypes = list(int(z) for z in np.asarray(at[()]).flatten()) if at is not None else []
            info["Atomtypes"] = atomtypes

            # AtomData (5×N: x, y, z, occ, dwf)
            ad = cd.get("AtomData")
            if ad is not None:
                data = np.asarray(ad[()], dtype=np.float64)
                # May be (5, N) or (N, 5)
                if data.shape[0] == 5:
                    coords = data[:3].T  # (N, 3)
                    occ = data[3]
                    dwf = data[4]
                elif data.shape[1] == 5:
                    coords = data[:, :3]
                    occ = data[:, 3]
                    dwf = data[:, 4]
                else:
                    coords, occ, dwf = np.array([]), np.array([]), np.array([])

                # Map Z to element symbol (inline table, no extra dependency)
                _Z_TO_SYM = {
                    1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',
                    11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',
                    19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',
                    27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',34:'Se',
                    35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',42:'Mo',
                    43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',
                    51:'Sb',52:'Te',53:'I',54:'Xe',55:'Cs',56:'Ba',57:'La',58:'Ce',
                    59:'Pr',60:'Nd',61:'Pm',62:'Sm',63:'Eu',64:'Gd',65:'Tb',66:'Dy',
                    67:'Ho',68:'Er',69:'Tm',70:'Yb',71:'Lu',72:'Hf',73:'Ta',74:'W',
                    75:'Re',76:'Os',77:'Ir',78:'Pt',79:'Au',80:'Hg',81:'Tl',82:'Pb',
                    83:'Bi',84:'Po',85:'At',86:'Rn',87:'Fr',88:'Ra',89:'Ac',90:'Th',
                    91:'Pa',92:'U',93:'Np',94:'Pu',95:'Am',96:'Cm',
                }
                atoms = []
                for i in range(len(coords)):
                    z = atomtypes[i] if i < len(atomtypes) else 0
                    sym = _Z_TO_SYM.get(z, f"Z={z}")
                    atoms.append({
                        "element": sym,
                        "Z": z,
                        "x": round(float(coords[i][0]), 5),
                        "y": round(float(coords[i][1]), 5),
                        "z": round(float(coords[i][2]), 5),
                        "occ": round(float(occ[i]), 4),
                        "dwf": round(float(dwf[i]), 6),
                    })
                info["atoms"] = atoms

            # Metadata strings
            for key in ["Creator", "ProgramName", "CreationDate", "CreationTime", "useReference"]:
                ds = cd.get(key)
                if ds is not None:
                    val = ds[()]
                    if isinstance(val, np.ndarray):
                        val = val.flat[0]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    info[key] = str(val)

        # Crystal system name
        csys_names = {1: "Cubic", 2: "Tetragonal", 3: "Orthorhombic",
                      4: "Hexagonal", 5: "Trigonal", 6: "Monoclinic", 7: "Triclinic"}
        info["crystal_system_name"] = csys_names.get(info.get("CrystalSystem"), "Unknown")

        return info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-xtal")
async def add_xtal(req: AddXtalRequest):
    """Copy an XTAL file into the local XTAL library."""
    src = Path(req.file_path)
    if not src.is_file():
        raise HTTPException(status_code=400, detail=f"Source file not found: {req.file_path}")
    if src.suffix.lower() != ".xtal":
        raise HTTPException(status_code=400, detail="File must have a .xtal extension")

    try:
        target_dir = _db_dir("xtal")
        if req.material_name.strip():
            target_dir = target_dir / req.material_name.strip()
        target_dir.mkdir(parents=True, exist_ok=True)

        dest = target_dir / src.name
        shutil.copy2(str(src), str(dest))
        return {
            "success": True,
            "name": src.name,
            "path": str(dest),
            "category": "xtal",
        }
    except Exception as e:
        logger.exception("Failed to add XTAL file")
        raise HTTPException(status_code=500, detail=str(e))


class ConvertCifRequest(BaseModel):
    cif_path: str
    output_dir: str = ""  # If empty, uses default XTAL library
    dwf_overrides: dict = {}  # Element symbol -> DWF value overrides


@router.post("/convert-cif-to-xtal")
async def convert_cif_to_xtal(req: ConvertCifRequest):
    """Convert a CIF file to EMsoft .xtal format.

    Logic ported 1:1 from Xtal_Generator_GUI.py (V3.8) to ensure
    identical .xtal output — including disordered-structure handling,
    full reference extraction, and DWF.xlsx lookup.
    """
    cif_path = Path(req.cif_path)
    if not cif_path.is_file():
        cif_dir = _db_dir("cif")
        matches = list(cif_dir.rglob(req.cif_path))
        if matches:
            cif_path = matches[0]
        else:
            raise HTTPException(status_code=400, detail=f"CIF file not found: {req.cif_path}")

    # --- Pre-flight fit check: block conversion for unfit CIF files ---
    try:
        pre_info = await cif_info(cif_path.name)
        if pre_info.get("fit_for_xtal") is False:
            warnings = pre_info.get("fit_warnings", [])
            raise HTTPException(
                status_code=422,
                detail=f"CIF not fit for .xtal conversion: {'; '.join(warnings)}",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # If pre-check fails, allow conversion to proceed (fail later with better error)

    try:
        import re
        import html as html_mod
        import h5py
        import numpy as np
        import pandas as pd
        import spglib
        from datetime import date, datetime

        from pymatgen.core.structure import Structure
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        # --- Constants (identical to PyQt5 Xtal_Generator_GUI.py) ---
        CREATOR_TAG = b"EMsoftXtalGenerator_py_v3"
        PROGRAM_NAME_TAG = b"AutoXtalGen_v3"
        CRYSTAL_SYSTEMS = {
            'cubic': 1, 'tetragonal': 2, 'orthorhombic': 3,
            'hexagonal': 4, 'trigonal': 5, 'monoclinic': 6, 'triclinic': 7,
        }

        # --- Parse CIF (matching PyQt5 line 362-366) ---
        parser = CifParser(str(cif_path))
        structures = parser.parse_structures()
        if not structures:
            raise ValueError("Pymatgen could not parse any structure from the CIF file.")
        structure: Structure = structures[0]

        # --- Disordered structure handling (matching PyQt5 lines 368-400) ---
        if not structure.is_ordered:
            try:
                temp_struct = structure.get_primitive_structure(tolerance=0.25)
                ordered_structure_for_spglib = temp_struct.get_sorted_structure()
                if not ordered_structure_for_spglib.is_ordered:
                    possible_orderings = temp_struct.get_orderings()
                    if possible_orderings:
                        ordered_structure_for_spglib = possible_orderings[0]
                    else:
                        raise ValueError("Could not derive an ordered structure using get_orderings().")
            except Exception:
                # Fallback: pick highest-occupancy species per site
                species, coords_fallback = [], []
                for site in structure:
                    if site.is_ordered:
                        species.append(site.specie)
                    else:
                        species.append(max(site.species, key=site.species.get))
                    coords_fallback.append(site.frac_coords)
                ordered_structure_for_spglib = Structure(structure.lattice, species, coords_fallback)
        else:
            ordered_structure_for_spglib = structure

        # --- Build spglib input (matching PyQt5 lines 389-400) ---
        if not hasattr(ordered_structure_for_spglib, 'atomic_numbers'):
            if ordered_structure_for_spglib.is_ordered:
                raise AttributeError(
                    f"Ordered structure for spglib is missing 'atomic_numbers'. "
                    f"Structure: {ordered_structure_for_spglib.formula}"
                )
            else:
                atomic_numbers_list = []
                for site_idx, site in enumerate(ordered_structure_for_spglib):
                    if site.is_ordered:
                        atomic_numbers_list.append(site.specie.number)
                    elif site.species:
                        atomic_numbers_list.append(site.species.elements[0].number)
                    else:
                        raise ValueError(f"Site {site_idx} in {cif_path.name} has no species information.")
                sym_data_input = (
                    ordered_structure_for_spglib.lattice.matrix,
                    ordered_structure_for_spglib.frac_coords,
                    atomic_numbers_list,
                )
        else:
            sym_data_input = (
                ordered_structure_for_spglib.lattice.matrix,
                ordered_structure_for_spglib.frac_coords,
                ordered_structure_for_spglib.atomic_numbers,
            )

        sym_data = spglib.get_symmetry_dataset(sym_data_input, symprec=1e-5)

        # --- Reference extraction (matching PyQt5 lines 404-419) ---
        raw_text = cif_path.read_text(encoding="utf8", errors="ignore")
        found_refs = []
        doi_patterns = [
            r"_publ_section_doi\s+['\"]([^'\"]+)['\"]",
            r"_citation_doi\s+['\"]([^'\"]+)['\"]",
            r"_citation_journal_doi\s+['\"]([^'\"]+)['\"]",
        ]
        for pat in doi_patterns:
            for m_doi in re.findall(pat, raw_text):
                if m_doi.strip() not in found_refs:
                    found_refs.append(m_doi.strip())

        if not found_refs:
            matches_ref = re.findall(
                r"_publ_section_references\s*;\s*(.*?)\s*;", raw_text, flags=re.DOTALL
            )
            for m_ref in matches_ref:
                cite_text = m_ref.strip()
                plain = re.sub(r"<[^>]+>", "", cite_text)       # Strip HTML tags
                plain = html_mod.unescape(plain)                 # Decode HTML entities
                plain = re.sub(r'\s+', ' ', plain).strip()       # Normalize whitespace
                if plain and (plain not in found_refs):
                    found_refs.append(plain)

        if not found_refs:
            found_refs.append("none found")

        # --- DWF lookup (matching PyQt5 line 421) ---
        dwf_xlsx = _project_root() / "crystal-structures-for-ebsd-main" / "calculationxtal" / "DWF.xlsx"
        if not dwf_xlsx.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"DWF.xlsx not found at {dwf_xlsx}. Cannot convert without Debye-Waller factors.",
            )
        dwf_map = {
            str(r["Element"]).split()[0]: float(r["DWB 300 K"])
            for _, r in pd.read_excel(str(dwf_xlsx)).iterrows()
        }
        # Apply optional overrides from request
        if req.dwf_overrides:
            dwf_map.update(req.dwf_overrides)

        # --- Crystallographic data (matching PyQt5 lines 422-436) ---
        a, b, c = np.array(structure.lattice.abc) / 10.0  # Angstrom → nm
        alpha, beta, gamma = structure.lattice.angles
        lp_vals = (a, b, c, alpha, beta, gamma)

        sga = SpacegroupAnalyzer(structure, symprec=1e-5)
        csys = CRYSTAL_SYSTEMS.get(sga.get_crystal_system().lower(), 7)
        spg_number = sga.get_space_group_number()

        origin_shift = sym_data.get('origin_shift', np.zeros(3))
        spg_setting = 2 if np.linalg.norm(origin_shift) > 1e-6 else 1

        reps = sga.get_symmetrized_structure().equivalent_sites
        # Build AtomData preserving ALL co-occupying species on mixed sites
        # (one atom per species) — see _build_xtal_atomdata.
        Z_arr, atomdata, disorder_warnings = _build_xtal_atomdata(
            [r[0] for r in reps], dwf_map
        )
        N = int(Z_arr.shape[0])

        ref_str = found_refs[0]
        ref_bytes = ref_str.encode("utf-8", errors="ignore")

        # --- Write .xtal HDF5 (matching PyQt5 lines 541-569) ---
        # SECURITY: contain output_dir inside the XTAL library. Previously any
        # path (absolute or ../..) was accepted, allowing an arbitrary HDF5
        # write anywhere on disk. Require a relative path with no '..' that
        # resolves inside _db_dir("xtal").
        xtal_base = _db_dir("xtal").resolve()
        if req.output_dir:
            rel = Path(req.output_dir)
            if rel.is_absolute() or ".." in rel.parts:
                raise HTTPException(
                    status_code=400,
                    detail="output_dir must be a relative path inside the XTAL library "
                           "(no absolute paths or '..').",
                )
            out_dir = (xtal_base / rel).resolve()
            if out_dir != xtal_base and xtal_base not in out_dir.parents:
                raise HTTPException(status_code=400, detail="output_dir escapes the XTAL library.")
        else:
            out_dir = xtal_base
        out_dir.mkdir(parents=True, exist_ok=True)
        out_xtal = out_dir / cif_path.with_suffix('.xtal').name

        with h5py.File(str(out_xtal), "w") as f:
            cd = f.create_group("CrystalData")
            cd.create_dataset("CrystalSystem", data=[csys])
            cd.create_dataset("SpaceGroupNumber", data=[spg_number])
            cd.create_dataset("SpaceGroupSetting", data=[spg_setting])
            cd.create_dataset("LatticeParameters", data=lp_vals)
            cd.create_dataset("Natomtypes", data=[N])
            cd.create_dataset("Atomtypes", data=Z_arr)
            cd.create_dataset("AtomData", data=atomdata)
            cd.create_dataset(
                "useReference",
                data=[ref_bytes],
                dtype=h5py.special_dtype(vlen=bytes),
            )
            for name, val in [
                ("CreationDate", date.today().isoformat().encode("utf-8")),
                ("CreationTime", datetime.now().strftime("%H:%M:%S").encode("utf-8")),
                ("Creator", CREATOR_TAG),
                ("ProgramName", PROGRAM_NAME_TAG),
            ]:
                cd.create_dataset(
                    name, data=[val],
                    dtype=h5py.special_dtype(vlen=bytes),
                )

        return {
            "success": True,
            "name": out_xtal.name,
            "path": str(out_xtal),
            "crystal_system": csys,
            "space_group": spg_number,
            "n_atoms": N,
            "lattice": {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma},
            "reference": ref_str,
            "warnings": disorder_warnings,
        }
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency for CIF conversion: {e}")
    except Exception as e:
        logger.exception("Failed to convert CIF to XTAL")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/status")
async def cache_status():
    """Get local database file stats (SHT, H5, XTAL, CIF)."""
    try:
        total_bytes = 0
        total_files = 0
        for cat, exts in [("sht", {".sht"}), ("h5", {".h5"}), ("xtal", {".xtal"}), ("cif", {".cif"})]:
            try:
                entries = _scan_directory(_db_dir(cat), exts)
                total_files += len(entries)
                total_bytes += sum(e.get("size", 0) for e in entries)
            except Exception:
                pass
        return {
            "size_bytes": total_bytes,
            "max_bytes": 20 * 1024 * 1024 * 1024,  # 20 GB nominal limit
            "n_files": total_files,
        }
    except Exception:
        return {"size_bytes": 0, "max_bytes": 0, "n_files": 0}


@router.get("/categories")
async def list_categories():
    """List available database categories with file counts."""
    categories = [
        {"id": "cif",  "name": "CIF Files",        "description": "Crystal structure files for Hough indexing"},
        {"id": "sht",  "name": "SHT Files",         "description": "Spherical harmonic transform files for EMSphinx"},
        {"id": "h5",   "name": "H5 Master Files",   "description": "Master pattern files for Dictionary indexing"},
        {"id": "xtal", "name": "XTAL Files",        "description": "EMsoft crystal structure files"},
    ]

    ext_map = {"cif": {".cif"}, "sht": {".sht"}, "h5": {".h5"}, "xtal": {".xtal"}}
    for cat in categories:
        try:
            dirpath = _db_dir(cat["id"])
            entries = _scan_directory(dirpath, ext_map[cat["id"]])
            cat["file_count"] = len(entries)
            cat["directory"] = str(dirpath)
        except Exception:
            cat["file_count"] = 0
            cat["directory"] = ""

    return {"categories": categories}


# --- Build Database (background task with progress tracking) ---

_build_tasks: dict = {}  # task_id -> {status, progress, total, message, error}


def _run_build_database(task_id: str, cif_folder: Path, output_path: Path, skip_online: bool):
    """Background worker that runs cif_database_builder.build_database()."""
    try:
        # Add project root to sys.path so the builder can be imported
        import sys
        project_root = str(_project_root())
        builder_dir = str(_project_root() / "crystal-structures-for-ebsd-main" / "calculationxtal")
        for p in [project_root, builder_dir]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from cif_database_builder import build_database

        def progress_cb(current, total, message):
            _build_tasks[task_id].update({
                "progress": current,
                "total": total,
                "message": message,
            })

        _build_tasks[task_id]["status"] = "running"
        _build_tasks[task_id]["message"] = "Starting CIF parsing..."

        df = build_database(
            cif_folder=str(cif_folder),
            output_path=str(output_path),
            skip_online=skip_online,
            progress_callback=progress_cb,
        )

        if df is not None:
            _build_tasks[task_id].update({
                "status": "completed",
                "message": f"Database built: {len(df)} entries",
                "progress": _build_tasks[task_id].get("total", 0),
                "entry_count": len(df),
            })
        else:
            _build_tasks[task_id].update({
                "status": "failed",
                "message": "build_database returned None — check CIF folder path",
            })
    except Exception as e:
        logger.exception("Build database failed")
        _build_tasks[task_id].update({
            "status": "failed",
            "message": str(e),
            "error": str(e),
        })


class BuildDatabaseRequest(BaseModel):
    skip_online: bool = True


@router.post("/build")
async def build_database_endpoint(req: BuildDatabaseRequest, background_tasks: BackgroundTasks):
    """Start building crystal_database.xlsx from all CIF files (background task)."""
    cif_folder = _db_dir("cif")
    if not cif_folder.is_dir():
        raise HTTPException(status_code=400, detail=f"CIF library not found: {cif_folder}")

    output_path = _project_root() / "Database" / "crystal_database.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_id = str(uuid.uuid4())
    _build_tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "total": 0,
        "message": "Queued...",
    }

    # Run in a real thread (not asyncio) since build_database is CPU-bound
    thread = threading.Thread(
        target=_run_build_database,
        args=(task_id, cif_folder, output_path, req.skip_online),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "status": "queued"}


@router.get("/build/status/{task_id}")
async def build_status(task_id: str):
    """Poll build database task progress."""
    task = _build_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Unknown task ID")
    return task


# --- Read built database entries ---

@router.get("/entries")
async def database_entries():
    """Read crystal_database.xlsx and return all rows as JSON.

    Returns the rich crystal metadata (13 columns) built by /build.
    Falls back to empty list if the database hasn't been built yet.
    """
    xlsx_path = _project_root() / "Database" / "crystal_database.xlsx"
    if not xlsx_path.is_file():
        return {"entries": [], "source": "none", "message": "No database built yet — use Build Database first"}

    try:
        import pandas as pd
        df = pd.read_excel(str(xlsx_path)).fillna("")
        entries = df.to_dict(orient="records")
        return {
            "entries": entries,
            "source": "crystal_database.xlsx",
            "count": len(entries),
        }
    except Exception as e:
        logger.exception("Failed to read crystal_database.xlsx")
        raise HTTPException(status_code=500, detail=str(e))


# --- Update database entries ---

class EntryUpdate(BaseModel):
    row: int
    column: str
    value: str


class UpdateEntriesRequest(BaseModel):
    updates: list[EntryUpdate]


@router.patch("/entries/update")
async def update_entries(req: UpdateEntriesRequest):
    """Apply cell-level edits to crystal_database.xlsx and write back.

    Accepts a list of {row, column, value} patches and applies them in order.
    Row index is 0-based (matches the DataFrame index, not the Excel row number).
    """
    xlsx_path = _project_root() / "Database" / "crystal_database.xlsx"
    if not xlsx_path.is_file():
        raise HTTPException(status_code=404, detail="crystal_database.xlsx not found — build the database first")

    try:
        import pandas as pd
        df = pd.read_excel(str(xlsx_path))

        errors = []
        applied = 0
        for patch in req.updates:
            if patch.column not in df.columns:
                errors.append(f"Unknown column: '{patch.column}'")
                continue
            if patch.row < 0 or patch.row >= len(df):
                errors.append(f"Row {patch.row} out of range (0–{len(df) - 1})")
                continue
            df.at[patch.row, patch.column] = patch.value
            applied += 1

        if applied > 0:
            df.to_excel(str(xlsx_path), index=False)

        result = {"success": True, "applied": applied}
        if errors:
            result["warnings"] = errors
        return result

    except Exception as e:
        logger.exception("Failed to update crystal_database.xlsx")
        raise HTTPException(status_code=500, detail=str(e))


# --- Master pattern / SHT thumbnail preview ---

@router.get("/preview/{filename:path}")
async def preview_file(filename: str):
    """Return a base64-encoded PNG thumbnail of a master pattern or SHT HDF5 file.

    Searches in EBSD_H5_Cache and EBSD_SHT_Database.  Returns:
        {"image": "<base64 PNG or null>", "metadata": {...}, "filename": str}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    import base64
    import numpy as np
    import h5py

    # Resolve the file on disk
    search_dirs = [
        _project_root() / "Database" / "EBSD_H5_Cache",
        _project_root() / "Database" / "EBSD_SHT_Database",
    ]
    file_path: Optional[Path] = None
    for base_dir in search_dirs:
        candidate = base_dir / filename
        if candidate.is_file():
            file_path = candidate
            break
        # Also try a recursive search in case the caller omitted subdirectory
        if base_dir.is_dir():
            matches = list(base_dir.rglob(Path(filename).name))
            if matches:
                file_path = matches[0]
                break

    if file_path is None:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    metadata: dict = {"filename": filename}
    image_b64: Optional[str] = None

    try:
        with h5py.File(str(file_path), "r") as f:
            # Extract metadata
            try:
                metadata["space_group"] = int(
                    np.array(f["CrystalData/SpaceGroupNumber"]).flat[0]
                )
            except Exception:
                pass
            try:
                metadata["npx"] = int(
                    np.array(f["NMLparameters/EBSDMasterNameList/npx"]).flat[0]
                )
            except Exception:
                pass
            try:
                metadata["energy_kev"] = float(
                    np.array(f["NMLparameters/MCCLNameList/EkeV"]).flat[0]
                )
            except Exception:
                pass

            # Find master pattern data
            data_paths = [
                "EMData/EBSDmaster/mLPNH",
                "EMData/EBSDmaster/masterSPNH",
                "EMData/MCOpenCL/accum_e",
            ]
            arr = None
            used_dpath = None
            for dpath in data_paths:
                if dpath in f:
                    arr = np.array(f[dpath], dtype=np.float32)
                    used_dpath = dpath
                    break

            if arr is not None:
                # If 3-D, take the middle energy slice
                if arr.ndim == 3:
                    arr = arr[arr.shape[0] // 2]
                elif arr.ndim > 3:
                    arr = arr[arr.shape[0] // 2, arr.shape[1] // 2]

                # Flag the disc-masked-Lambert corner bug (stale pre-2026-06-22
                # forward-sim masters). Only the SQUARE Lambert (mLPNH) is checked —
                # masterSPNH is a legitimate stereographic disc.
                if used_dpath is not None and used_dpath.endswith("mLPNH"):
                    from backend.forward_sim.io.master_validation import (
                        is_lambert_disc_masked,
                    )
                    if is_lambert_disc_masked(arr):
                        metadata["corrupt"] = True
                        metadata["corrupt_reason"] = (
                            "Disc-masked Lambert square (corners zeroed) — stale "
                            "artifact from before the 2026-06-22 corner fix. "
                            "Delete and regenerate this master pattern."
                        )

                # Normalize to 0-255
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max > arr_min:
                    arr = (arr - arr_min) / (arr_max - arr_min)
                else:
                    arr = np.zeros_like(arr)

                fig, ax = plt.subplots(figsize=(3, 3), dpi=72)
                fig.patch.set_facecolor("#282a36")
                ax.imshow(arr, cmap="viridis", origin="upper")
                ax.axis("off")
                if metadata.get("corrupt"):
                    ax.text(
                        0.5, 0.5, "⚠ CORRUPT\nREGENERATE",
                        transform=ax.transAxes, ha="center", va="center",
                        color="#ff5555", fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="#000000aa",
                                  edgecolor="#ff5555"),
                    )
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight",
                            facecolor=fig.get_facecolor(), pad_inches=0.05)
                plt.close(fig)
                buf.seek(0)
                image_b64 = base64.b64encode(buf.read()).decode("utf-8")

    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("preview_file: could not render thumbnail for %s: %s", filename, exc)

    return {"image": image_b64, "metadata": metadata, "filename": filename}


# --- SHT master-pattern SPHERE (interactive 3D viewer in the Database Browser) ---

@router.get("/sphere/{filename:path}")
async def sphere_file(filename: str, max_bandwidth: int = 128, target_size: int = 128):
    """Return the master pattern of a ``.sht`` reconstructed onto a sphere grid.

    Used by the Database Browser's interactive (rotatable) master-pattern sphere
    viewer.  Reuses the SAME inverse-SHT the indexer/forward path runs (cached), so
    no new physics.  The reconstruction is CPU/GPU-bound, so it runs in a worker
    thread to keep the event loop responsive.

    Returns ``{filename, grid, phi[], theta[], bandwidth, file_bandwidth, meta}``.
    404 if the file is absent; 422 if it exists but cannot be reconstructed.
    """
    import asyncio  # noqa: PLC0415
    from backend.api.services.sht_sphere import sphere_from_sht, SphereError  # noqa: PLC0415

    # Clamp params defensively (bandwidth must be sane; target a reasonable surface).
    bw = max(16, min(384, int(max_bandwidth)))
    tgt = max(32, min(256, int(target_size)))
    try:
        return await asyncio.to_thread(
            sphere_from_sht, filename, max_bandwidth=bw, target_size=tgt
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SphereError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # fail loud with an actionable message, never a blank
        logger.exception("sphere_file: failed for %s", filename)
        raise HTTPException(status_code=422, detail=f"sphere reconstruction failed: {e}")


@router.get("/structure/{filename:path}")
async def structure_file(filename: str):
    """Return the full-unit-cell crystal structure of a local .cif/.xtal.

    Used by the Database Browser's interactive 3D crystal-structure viewer.
    Returns ``{source, lattice, cell_vectors, space_group, atoms[], bonds[],
    polyhedra[], meta}``. 404 if the file is absent; 422 if it cannot be parsed.
    """
    import asyncio  # noqa: PLC0415
    import glob as _glob  # noqa: PLC0415

    from backend.api.services.crystal_structure import structure_payload  # noqa: PLC0415

    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix == ".cif":
        base = _db_dir("cif")
    elif suffix == ".xtal":
        base = _db_dir("xtal")
    else:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported type: {suffix!r} (need .cif/.xtal)",
        )

    # Escape glob metacharacters so a crafted name (e.g. "*.cif") is matched
    # literally, not as a wildcard against arbitrary library files.
    matches = list(base.rglob(_glob.escape(name)))
    if not matches:
        raise HTTPException(status_code=404, detail=f"structure file not found: {name}")
    try:
        return await asyncio.to_thread(structure_payload, str(matches[0]))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # fail loud with an actionable message
        logger.exception("structure_file: failed for %s", filename)
        raise HTTPException(status_code=422, detail=f"structure parse failed: {e}")


@router.get("/sht/{filename:path}/info")
async def sht_info(filename: str):
    """Provenance + simulation parameters for a local .sht (File Info panel)."""
    import asyncio  # noqa: PLC0415
    from backend.api.services.sht_provenance import build_sht_info  # noqa: PLC0415
    sht_dir = _db_dir("sht")
    matches = [p for p in sht_dir.rglob("*.sht") if p.name == Path(filename).name]
    if not matches:
        raise HTTPException(status_code=404, detail=f"SHT not found: {filename}")
    return await asyncio.to_thread(
        build_sht_info, matches[0],
        xtal_dir=_db_dir("xtal"), cif_dir=_db_dir("cif"),
    )


def _read_dwf_df(dwf_path):
    """Read DWF.xlsx and guarantee a (string-typed, NaN-free) Reference column."""
    import pandas as pd
    df = pd.read_excel(str(dwf_path))
    if "Reference" not in df.columns:
        df["Reference"] = ""
    df["Reference"] = df["Reference"].fillna("").astype(str)
    return df


@router.get("/dwf")
async def get_dwf():
    """Read Debye-Waller factor reference table (DWF.xlsx).

    Always includes a ``Reference`` column (provenance); older files without
    it get an empty column so the UI can show / fill the source.
    """
    dwf_path = _project_root() / "crystal-structures-for-ebsd-main" / "calculationxtal" / "DWF.xlsx"
    if not dwf_path.is_file():
        raise HTTPException(status_code=404, detail="DWF.xlsx not found")
    try:
        df = _read_dwf_df(dwf_path)
        entries = df.to_dict(orient="records")
        return {"entries": entries, "path": str(dwf_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/dwf")
async def update_dwf(req: UpdateDwfRequest):
    """Upsert Debye-Waller factors in DWF.xlsx.

    Existing elements are updated; unknown elements are ADDED (this is what
    lets the user add e.g. Cu, which was missing and silently fell back to the
    generic 0.005 during CIF→.xtal conversion). Each entry carries an optional
    literature reference. An empty reference on an update does not wipe an
    existing one.
    """
    dwf_path = _project_root() / "crystal-structures-for-ebsd-main" / "calculationxtal" / "DWF.xlsx"
    if not dwf_path.is_file():
        raise HTTPException(status_code=404, detail="DWF.xlsx not found")
    try:
        import pandas as pd
        df = _read_dwf_df(dwf_path)
        updated, added = [], []
        for upd in req.updates:
            elem = (upd.element or "").strip()
            if not elem:
                continue
            # Match by first token so "Fe" matches the stored "Fe (BCC)".
            mask = df["Element"].astype(str).apply(
                lambda v: v.split()[0] if v.split() else ""
            ) == elem
            if mask.any():
                df.loc[mask, "DWB 300 K"] = upd.dwb_300k
                if upd.reference:
                    df.loc[mask, "Reference"] = upd.reference
                updated.append(elem)
            else:
                df = pd.concat(
                    [df, pd.DataFrame([{
                        "Element": elem,
                        "DWB 300 K": upd.dwb_300k,
                        "Reference": upd.reference,
                    }])],
                    ignore_index=True,
                )
                added.append(elem)
        if updated or added:
            df.to_excel(str(dwf_path), index=False)
        return {"updated": updated, "added": added}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files")
async def delete_files(req: DeleteFilesRequest):
    """Delete database files from local, server, or both.

    Supports categories: cif, xtal, h5, master, sht, dictionary.
    Uses 'material' field to resolve the correct subfolder.
    Backwards-compatible: accepts legacy delete_from_server bool.
    """
    project_root = _project_root()
    local_root = project_root / "Database"
    server_root = _get_server_db_root()

    # Resolve delete scope (new field takes priority over legacy bool)
    if req.delete_from:
        scope = req.delete_from
    elif req.delete_from_server:
        scope = "everywhere"
    else:
        scope = "local"

    delete_local = scope in ("local", "everywhere")
    delete_server = scope in ("server", "everywhere")

    # Category → local subfolder name (canonical casing — see _DB_SUBDIRS note;
    # "CIF_Library" must match path_utils.DATABASE_SUBFOLDERS for Linux/macOS).
    cat_local_map = {
        "cif": "CIF_Library", "xtal": "XTAL_Library",
        "h5": "EBSD_H5_Cache", "master": "EBSD_H5_Cache",
        "sht": "EBSD_SHT_Database", "dictionary": "Dictionary_Library",
    }
    # Category → server subfolder name
    cat_server_map = {
        "cif": "EBSD_CIF_Library", "xtal": "EBSD_XTAL_Library",
        "h5": "EBSD_H5_Cache", "master": "EBSD_H5_Cache",
        "sht": "EBSD_SHT_Database", "dictionary": "EBSD_Dictionary_Library",
    }

    deleted = []
    server_deleted = []
    errors = []

    for entry in req.files:
        cat = entry.category
        # Treat display placeholders ("-", en/em dash) and blanks as "no material
        # subfolder" — older listings sent "-" for root-level files, which made
        # the lookups search a nonexistent "<dir>/-/" and fail for every file.
        material = (entry.material or "").strip()
        if material in ("-", "–", "—"):
            material = ""
        filename = Path(entry.path).name

        # --- Delete from LOCAL ---
        if delete_local:
            local_subdir = cat_local_map.get(cat)
            if not local_subdir:
                errors.append({"path": entry.path, "error": f"Unsupported category: {cat}"})
                continue

            local_dir = local_root / local_subdir
            if material:
                local_dir = local_dir / material

            if local_dir.is_dir():
                # Use manual walk instead of rglob to avoid glob pattern
                # interpretation of special chars like [] {} () in filenames
                matches = [
                    p for p in local_dir.rglob("*")
                    if p.is_file() and p.name == filename
                ]
                if matches:
                    try:
                        os.remove(str(matches[0]))
                        deleted.append(entry.path)
                    except Exception as e:
                        errors.append({"path": entry.path, "error": f"Local delete failed: {e}"})
                elif scope == "local":
                    errors.append({"path": entry.path, "error": "File not found locally"})
            elif scope == "local":
                errors.append({"path": entry.path, "error": "Local directory not found"})

        # --- Delete from SERVER ---
        if delete_server and server_root:
            server_subdir = cat_server_map.get(cat)
            if server_subdir:
                server_dir = server_root / server_subdir
                if material:
                    server_dir = server_dir / material

                if server_dir.is_dir():
                    # Manual walk to avoid glob pattern issues with special chars
                    matches = [
                        p for p in server_dir.rglob("*")
                        if p.is_file() and p.name == filename
                    ]
                    for m in matches:
                        try:
                            os.remove(str(m))
                            server_deleted.append(str(m))
                        except Exception as e:
                            errors.append({"path": str(m), "error": f"Server delete failed: {e}"})
                elif scope == "server":
                    errors.append({"path": entry.path, "error": "Server directory not found"})

    # --- Remove deleted CIF entries from crystal_database.xlsx ---
    deleted_cif_stems = set()
    for entry in req.files:
        if entry.category == "cif" and entry.path in deleted:
            deleted_cif_stems.add(Path(entry.path).stem.lower())

    if deleted_cif_stems:
        xlsx_path = project_root / "Database" / "crystal_database.xlsx"
        if xlsx_path.is_file():
            try:
                import pandas as pd
                df = pd.read_excel(str(xlsx_path))
                # Find the filename column — builder uses "CIF File Name"
                name_col = df.columns[0]
                for col in df.columns:
                    if col.lower() in ("cif file name", "cif_file_name", "filename", "file", "cif_file", "name"):
                        name_col = col
                        break
                before = len(df)
                df = df[~df[name_col].astype(str).str.replace(".cif", "", case=False).str.lower().isin(deleted_cif_stems)]
                removed = before - len(df)
                if removed > 0:
                    df.to_excel(str(xlsx_path), index=False)
                    logger.info(f"Removed {removed} entries from crystal_database.xlsx")
            except Exception as exc:
                logger.warning(f"Failed to clean crystal_database.xlsx: {exc}")

    result = {"deleted": deleted, "errors": errors}
    if server_deleted:
        result["server_deleted"] = server_deleted
    return result


@router.post("/cif/batch-info")
async def cif_batch_info(req: BatchInfoRequest):
    """Parse details for multiple CIF files in one request."""
    cif_dir = _db_dir("cif")
    xtal_dir = _db_dir("xtal")

    # Build set of available xtal stems for fast lookup
    xtal_stems = set()
    if xtal_dir.is_dir():
        for f in xtal_dir.rglob("*.xtal"):
            xtal_stems.add(f.stem.lower())

    results = []
    for filename in req.filenames:
        info: dict = {"filename": filename}
        try:
            matches = list(cif_dir.rglob(filename))
            if not matches:
                info["error"] = f"CIF file '{filename}' not found in library"
                info["has_xtal"] = False
                results.append(info)
                continue

            cif_path = matches[0]
            info["path"] = str(cif_path)
            info["size"] = cif_path.stat().st_size
            info["has_xtal"] = Path(filename).stem.lower() in xtal_stems

            # Parse with orix
            try:
                from orix.crystal_map import Phase
                from ebsd_utils import sanitize_cif
                phase = Phase.from_cif(sanitize_cif(str(cif_path)))
                if phase.name != cif_path.stem:
                    phase.name = cif_path.stem
                info["phase_name"] = str(phase.name)
                info["space_group"] = str(phase.space_group) if hasattr(phase, "space_group") else ""
                if hasattr(phase, "space_group") and phase.space_group is not None:
                    sg = phase.space_group
                    info["space_group_number"] = int(sg.number) if hasattr(sg, "number") else None
                    info["crystal_system"] = str(sg.crystal_system) if hasattr(sg, "crystal_system") else None
                if hasattr(phase, "structure") and phase.structure is not None:
                    struct = phase.structure
                    lat = struct.lattice
                    info["lattice"] = {
                        "a": float(lat.a),
                        "b": float(lat.b),
                        "c": float(lat.c),
                        "alpha": float(lat.alpha),
                        "beta": float(lat.beta),
                        "gamma": float(lat.gamma),
                    }
                    if len(struct) > 0:
                        atoms = []
                        for site in struct:
                            atoms.append({
                                "element": str(site.element),
                                "x": round(float(site.x), 5),
                                "y": round(float(site.y), 5),
                                "z": round(float(site.z), 5),
                                "occ": round(float(site.occupancy), 4) if hasattr(site, "occupancy") else 1.0,
                                "dwf": round(float(site.Bisoequiv), 4) if hasattr(site, "Bisoequiv") else 0.5,
                            })
                        info["atoms"] = atoms
                        info["n_atoms"] = len(atoms)
            except Exception as exc:
                info["parse_warning"] = str(exc)

            # Also surface DOI + reference here so the cached parsedInfo
            # in the frontend carries them across refreshCifList() calls
            # (e.g. after a Sync). Previously only the single-file
            # cif_info endpoint returned them, so the list cache was
            # always missing these fields and the user perceived their
            # freshly-typed DOI as "gone" right after a sync.
            try:
                import html as _html
                raw_text = cif_path.read_text(encoding="utf-8", errors="ignore")
                dois = re.findall(
                    r"(?:_publ_section_doi|_citation_doi|_journal_paper_doi)\s+['\"]?([^'\"\s]+)['\"]?",
                    raw_text,
                )
                info["doi"] = dois[0].strip() if dois else ""
                ref_matches = re.findall(
                    r"_publ_section_references\s*;\s*(.*?)\s*;", raw_text, re.DOTALL
                )
                info["reference"] = (
                    re.sub(r"\s+", " ", _html.unescape(ref_matches[0])).strip()
                    if ref_matches else ""
                )
            except Exception:
                info["doi"] = ""
                info["reference"] = ""

        except Exception as exc:
            info["error"] = str(exc)
            info["has_xtal"] = False

        results.append(info)

    return {"results": results}


# ---------------------------------------------------------------------------
# Sync endpoints
# ---------------------------------------------------------------------------

@router.get("/sync/status")
async def sync_status():
    """Check if server sync is available."""
    server_root = _get_server_db_root()
    if server_root is None:
        return {"available": False, "reason": "Server not configured or offline"}

    dirs_ok = {}
    for cat, subdir_name in _SERVER_SUBDIRS.items():
        d = server_root / subdir_name
        dirs_ok[cat] = d.is_dir()

    return {
        "available": True,
        "server_root": str(server_root),
        "directories": dirs_ok,
    }


@router.post("/sync")
async def sync_database():
    """Bidirectional sync between local and server crystal databases."""
    server_root = _get_server_db_root()
    if server_root is None:
        raise HTTPException(
            status_code=400,
            detail="Server mode not configured or not connected",
        )

    result = SyncResult()

    for cat, exts in [("cif", {".cif"}), ("xtal", {".xtal"})]:
        local_dir = _db_dir(cat)
        server_subdir = server_root / _SERVER_SUBDIRS.get(cat, "")

        if not local_dir.is_dir():
            local_dir.mkdir(parents=True, exist_ok=True)
        if not server_subdir.is_dir():
            result.errors.append(f"Server directory not found: {server_subdir}")
            continue

        local_files: dict[str, Path] = {}
        for f in local_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                local_files[f.name] = f

        server_files: dict[str, Path] = {}
        for f in server_subdir.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                server_files[f.name] = f

        local_names = set(local_files)
        server_names = set(server_files)

        # Upload: local only → server
        for name in local_names - server_names:
            try:
                dest = server_subdir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(local_files[name]), str(dest))
                result.uploaded.append(name)
            except Exception as e:
                result.errors.append(f"Upload {name}: {e}")

        # Download: server only → local
        for name in server_names - local_names:
            try:
                dest = local_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(server_files[name]), str(dest))
                result.downloaded.append(name)
            except Exception as e:
                result.errors.append(f"Download {name}: {e}")

        # Both sides: check for conflicts by hash
        for name in local_names & server_names:
            try:
                hash_fn = _xtal_science_hash if name.lower().endswith(".xtal") else _file_hash
                local_hash = hash_fn(local_files[name])
                server_hash = hash_fn(server_files[name])
                if local_hash == server_hash:
                    result.up_to_date.append(name)
                else:
                    local_stat = local_files[name].stat()
                    server_stat = server_files[name].stat()
                    result.conflicts.append({
                        "filename": name,
                        "file_type": cat,
                        "local_path": str(local_files[name]),
                        "server_path": str(server_files[name]),
                        "local_mtime": datetime.fromtimestamp(local_stat.st_mtime).isoformat(),
                        "server_mtime": datetime.fromtimestamp(server_stat.st_mtime).isoformat(),
                        "local_size": local_stat.st_size,
                        "server_size": server_stat.st_size,
                    })
            except Exception as e:
                result.errors.append(f"Compare {name}: {e}")

    return result.model_dump()


@router.post("/sync/resolve")
async def resolve_conflicts(req: ResolveConflictsRequest):
    """Resolve sync conflicts."""
    server_root = _get_server_db_root()
    if server_root is None:
        raise HTTPException(status_code=400, detail="Server not configured")

    resolved = []
    errors = []

    for res in req.resolutions:
        try:
            local_dir = _db_dir(res.file_type)
            server_dir = server_root / _SERVER_SUBDIRS.get(res.file_type, "")
            local_path = local_dir / res.filename
            server_path = server_dir / res.filename

            # Recursive search if not found at root
            if not local_path.is_file():
                matches = list(local_dir.rglob(res.filename))
                local_path = matches[0] if matches else local_path
            if not server_path.is_file():
                matches = list(server_dir.rglob(res.filename))
                server_path = matches[0] if matches else server_path

            if res.action == "keep_local":
                if local_path.is_file():
                    server_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(local_path), str(server_path))
                    resolved.append(res.filename)
                else:
                    errors.append(f"Local file not found: {res.filename}")

            elif res.action == "keep_server":
                if server_path.is_file():
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(server_path), str(local_path))
                    resolved.append(res.filename)
                else:
                    errors.append(f"Server file not found: {res.filename}")

            elif res.action == "keep_both":
                if server_path.is_file() and local_path.is_file():
                    stem = server_path.stem
                    suffix = server_path.suffix
                    ts = datetime.fromtimestamp(server_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
                    renamed = server_path.parent / f"{stem}_server_{ts}{suffix}"
                    shutil.copy2(str(server_path), str(renamed))
                    shutil.copy2(str(local_path), str(server_path))
                    resolved.append(res.filename)
                else:
                    errors.append(f"Both files needed for keep_both: {res.filename}")

            elif res.action == "skip":
                resolved.append(res.filename)

        except Exception as e:
            errors.append(f"Resolve {res.filename}: {e}")

    return {"resolved": resolved, "errors": errors}


# ---------------------------------------------------------------------------
# Selective upload / download (local <-> server, ALL categories)
# ---------------------------------------------------------------------------
#
# "Sync All" (/sync) only moves CIF/XTAL. The large simulation outputs
# (SHT / MC h5 / Master h5) had no transfer path at all, and the per-row
# "download" button misfired into Sync All. These endpoints copy an explicit
# selection in either direction, for every category, with hash dedup and an
# overwrite flag for differing files.

# Category -> on-disk subfolder. Mirrors delete_files' maps; "master" shares
# the EBSD_H5_Cache dir with "h5" (both are master/MC .h5 files).
_CAT_LOCAL_SUBDIR = {
    "cif": "CIF_Library", "xtal": "XTAL_Library",
    "h5": "EBSD_H5_Cache", "master": "EBSD_H5_Cache",
    "sht": "EBSD_SHT_Database", "dictionary": "Dictionary_Library",
}
_CAT_SERVER_SUBDIR = {
    "cif": "EBSD_CIF_Library", "xtal": "EBSD_XTAL_Library",
    "h5": "EBSD_H5_Cache", "master": "EBSD_H5_Cache",
    "sht": "EBSD_SHT_Database", "dictionary": "EBSD_Dictionary_Library",
}


class TransferFileEntry(BaseModel):
    name: str
    category: str        # cif, xtal, sht, h5, master, dictionary
    material: str = ""   # subfolder (e.g. "Fe"); "" = directly in category dir


class TransferRequest(BaseModel):
    files: list[TransferFileEntry]
    overwrite: bool = False


def _find_file_by_name(directory: Path, filename: str) -> Optional[Path]:
    """First file named exactly `filename` anywhere under `directory`.

    Uses a manual walk (not rglob(filename)) so special chars like [] {} () in
    filenames aren't interpreted as glob patterns. Returns None if not found.
    """
    if not directory.is_dir():
        return None
    for p in directory.rglob("*"):
        if p.is_file() and p.name == filename:
            return p
    return None


def _transfer_files(req: TransferRequest, *, to_server: bool) -> dict:
    """Copy selected files local <-> server.

    Per file:
      dst missing               -> copy        (success)
      dst identical (by hash)   -> up_to_date  (no-op)
      dst differs + !overwrite  -> conflict    (left intact)
      dst differs + overwrite   -> copy        (success, overwrites)

    Returns a dict with the direction-appropriate success key
    ('uploaded' for local->server, 'downloaded' for server->local) plus
    up_to_date / conflicts / errors.
    """
    server_root = _get_server_db_root()
    if server_root is None:
        raise HTTPException(
            status_code=400,
            detail="Server mode not configured or not connected",
        )
    local_root = _project_root() / "Database"

    success_key = "uploaded" if to_server else "downloaded"
    success: list[str] = []
    up_to_date: list[str] = []
    conflicts: list[str] = []
    errors: list[dict] = []

    for entry in req.files:
        cat = entry.category
        # Treat display placeholders ("-", en/em dash) and blanks as "no
        # material subfolder" (older listings sent "-" for root-level files).
        material = (entry.material or "").strip()
        if material in ("-", "–", "—"):
            material = ""
        filename = Path(entry.name).name

        local_subdir = _CAT_LOCAL_SUBDIR.get(cat)
        server_subdir = _CAT_SERVER_SUBDIR.get(cat)
        if not local_subdir or not server_subdir:
            errors.append({"name": filename, "error": f"Unsupported category: {cat}"})
            continue

        local_dir = local_root / local_subdir
        server_dir = server_root / server_subdir
        if material:
            local_dir = local_dir / material
            server_dir = server_dir / material

        if to_server:
            src = _find_file_by_name(local_dir, filename)
            dst = server_dir / filename
        else:
            src = _find_file_by_name(server_dir, filename)
            dst = local_dir / filename

        if src is None:
            errors.append({
                "name": filename,
                "error": "File not found locally" if to_server else "File not found on server",
            })
            continue

        try:
            if dst.is_file():
                hash_fn = _xtal_science_hash if filename.lower().endswith(".xtal") else _file_hash
                if hash_fn(src) == hash_fn(dst):
                    up_to_date.append(filename)
                    continue
                if not req.overwrite:
                    conflicts.append(filename)
                    continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            success.append(filename)
        except Exception as e:
            errors.append({"name": filename, "error": str(e)})

    return {
        success_key: success,
        "up_to_date": up_to_date,
        "conflicts": conflicts,
        "errors": errors,
    }


@router.post("/upload")
async def upload_files(req: TransferRequest):
    """Upload selected local files to the server (local -> server).

    Covers every category (cif, xtal, sht, h5, master, dictionary) so large
    simulation outputs (.sht / master .h5) can be pushed to the shared server.
    Differing server copies are left intact unless overwrite=True (the UI
    confirms before sending overwrite).
    """
    return _transfer_files(req, to_server=True)


@router.post("/download")
async def download_files(req: TransferRequest):
    """Download selected server files into the local cache (server -> local)."""
    return _transfer_files(req, to_server=False)


# ---------------------------------------------------------------------------
# Cascade resolve endpoint
# ---------------------------------------------------------------------------

@router.post("/resolve-cascade", response_model=CascadeResponse)
async def resolve_cascade(req: CascadeRequest):
    """Resolve a set of selected database files to include all associated files.

    For each selected file, discovers associated files via cascade logic:
    - MC .h5    → optional master .h5 + .sht, then master → required dictionaries
    - Master .h5 → required dictionary .h5 + .json sidecars

    Returns the selected files (with location/size info) plus all associated files.
    """
    server_root = _get_server_db_root()

    cat_local_map = {
        "h5": "EBSD_H5_Cache",
        "master": "EBSD_H5_Cache",
        "sht": "EBSD_SHT_Database",
        "dictionary": "Dictionary_Library",
    }
    cat_server_map = {
        "h5": "EBSD_H5_Cache",
        "master": "EBSD_H5_Cache",
        "sht": "EBSD_SHT_Database",
        "dictionary": "EBSD_Dictionary_Library",
    }

    selected_names: set = {e.name for e in req.files}

    # Build selected list with location + size info
    selected: list[CascadeFileInfo] = []
    for entry in req.files:
        local_subdir = cat_local_map.get(entry.category, "EBSD_H5_Cache")
        local_dir = _project_root() / "Database" / local_subdir
        if entry.material:
            local_dir = local_dir / entry.material
        local_path = None
        if local_dir.is_dir():
            candidates = list(local_dir.rglob(entry.name))
            if candidates:
                local_path = candidates[0]

        server_path = None
        if server_root:
            srv_subdir = cat_server_map.get(entry.category, "EBSD_H5_Cache")
            srv_dir = server_root / srv_subdir
            if entry.material:
                srv_dir = srv_dir / entry.material
            if srv_dir.is_dir():
                srv_candidates = list(srv_dir.rglob(entry.name))
                if srv_candidates:
                    server_path = srv_candidates[0]

        has_local = local_path is not None and local_path.is_file()
        has_server = server_path is not None and server_path.is_file()
        if has_local and has_server:
            location = "both"
            size = local_path.stat().st_size
        elif has_local:
            location = "local"
            size = local_path.stat().st_size
        elif has_server:
            location = "server"
            size = server_path.stat().st_size
        else:
            location = "local"
            size = 0

        selected.append(CascadeFileInfo(
            name=entry.name,
            category=entry.category,
            material=entry.material,
            location=location,
            size=size,
        ))

    # Find associated files
    try:
        associated = _find_associated_files(selected_names, req.files, server_root)
    except Exception as exc:
        logger.warning("Cascade resolution failed: %s", exc)
        associated = []

    # Compute has_local / has_server across all files
    all_files = selected + associated
    has_local = any(f.location in ("local", "both") for f in all_files)
    has_server = any(f.location in ("server", "both") for f in all_files)

    return CascadeResponse(
        selected=selected,
        associated=associated,
        has_local=has_local,
        has_server=has_server,
    )
