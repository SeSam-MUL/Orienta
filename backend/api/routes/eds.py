"""
EDS (Energy Dispersive Spectroscopy) API Routes

Wraps eds_utils for:
- EDS data extraction from H5OINA
- Counts → Wt.% → At.% conversion
- Phase suggestion based on chemistry
- Pixel-level and region-level analysis
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.services.image_utils import colormap_array_to_base64, element_color_overlay_to_base64
# EDS is per-dataset data: when the active dataset is a crop, every map here
# must be the cropped map. h5_session.get_active_extractor() is the raw
# extractor when nothing is cropped, so this is a no-op off the crop path.
from backend.api.services.h5_session import get_active_extractor as get_extractor, is_open
from backend.api.services.cif_phase_library import (
    auto_classify_pixels,
    candidates_for,
    load_cif_phase_library,
    suggest_phases_from_cif_library,
)
from backend.api.services.chemistry_score import (
    background_levels, infer_matrix_element,
)
from backend.api.services.eds_clustering import cluster_and_match
from backend.api.services.eds_wand import (
    flood_from, global_growth_curve, selection_stats, wand_field,
)
from backend.api.services.phase_map_store import (
    get_phase_map_store,
    palette_hex_for_state,
    render_phase_map_to_base64,
    render_structure_map_to_base64,
    structure_color_hex,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# The user's phase-colour choices, keyed on phase name. The frontend already
# persists these for the EBSD phase map; the EDS map honours the same ones so
# a phase looks the same on both pages. Held per process rather than threaded
# through every endpoint: the colours are a display preference, not part of
# the classification, and every response that carries an image or a legend
# needs them.
_COLOR_OVERRIDES: Dict[str, str] = {}


def _project_root() -> Path:
    """Project root — ``backend/api/routes/eds.py`` is 3 levels deep."""
    return Path(__file__).resolve().parents[3]


def _crystal_db_path() -> Path:
    return _project_root() / "Database" / "crystal_database.xlsx"


class QuantifyRequest(BaseModel):
    row: int
    col: int
    display_mode: str = "counts"  # "counts", "wt_pct", "at_pct"


class ProbeRequest(BaseModel):
    row: int
    col: int
    display_mode: str = "at_pct"  # one of counts / wt_pct / at_pct


class RegionQuantifyRequest(BaseModel):
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    display_mode: str = "at_pct"


class PhaseSuggestionRequest(BaseModel):
    row: int
    col: int


@router.get("/elements")
async def get_elements():
    """Get available EDS elements from the open file."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")
    ext = get_extractor()
    elements = ext.get_available_elements()
    return {"elements": elements}


def _resolve_element_name(element: str, ext) -> str:
    """Resolve short element names (e.g. 'Al') to full H5OINA names (e.g. 'Al Kα1')."""
    available = ext.get_available_elements()
    if element in available:
        return element
    # Try matching by prefix (short name)
    for full_name in available:
        if full_name.startswith(element + " ") or full_name.split()[0] == element:
            return full_name
    return element  # Return as-is, let the caller handle 404


@router.get("/map/{element}")
async def get_eds_map(element: str, mode: str = "counts", cmap: str = "hot", color: str = ""):
    """Get EDS element map with optional quantification."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")

    ext = get_extractor()
    resolved = _resolve_element_name(element, ext)
    raw_data = ext.get_element_map_2d(resolved)
    if raw_data is None:
        raise HTTPException(status_code=404, detail=f"No data for element: {element}")

    if mode == "counts":
        data = raw_data
    else:
        # Convert using eds_utils
        try:
            from eds_utils import parse_element_name
            el = parse_element_name(element)

            # Get all element maps for quantification
            all_elements = ext.get_available_elements()
            counts_dict = {}
            for el_name in all_elements:
                el_map = ext.get_element_map(el_name)
                if el_map is not None:
                    pure_el = parse_element_name(el_name)
                    counts_dict[pure_el] = el_map.astype(np.float64)

            if mode == "wt_pct":
                from eds_utils import counts_to_weight_pct
                wt_maps = counts_to_weight_pct(counts_dict)
                data = wt_maps.get(el, raw_data).reshape(raw_data.shape)
            elif mode == "at_pct":
                from eds_utils import counts_to_weight_pct, weight_pct_to_atomic_pct
                wt_maps = counts_to_weight_pct(counts_dict)
                at_maps = weight_pct_to_atomic_pct(wt_maps)
                data = at_maps.get(el, raw_data).reshape(raw_data.shape)
            else:
                data = raw_data
        except ImportError:
            data = raw_data

    # Use single-color overlay when a hex color is provided, otherwise matplotlib cmap
    if color:
        image_b64 = element_color_overlay_to_base64(data, color)
    else:
        image_b64 = colormap_array_to_base64(data, cmap)

    return {
        "image": image_b64,
        "element": element,
        "mode": mode,
        "min_val": float(np.nanmin(data)),
        "max_val": float(np.nanmax(data)),
        "shape": list(data.shape),
    }


@router.post("/quantify/pixel")
async def quantify_pixel(req: QuantifyRequest):
    """Get quantified EDS data for a single pixel."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")

    ext = get_extractor()
    elements = ext.get_available_elements()
    n_rows, n_cols = ext.get_grid_dimensions()
    # Validate the pixel is inside the scan — otherwise index arithmetic
    # silently lands on the wrong (or no) pixel and returns bogus data.
    if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"pixel ({req.row},{req.col}) outside scan {n_rows}x{n_cols}",
        )
    index = req.row * n_cols + req.col

    try:
        from eds_utils import parse_element_name, counts_to_weight_pct, weight_pct_to_atomic_pct
    except ImportError as e:
        # eds_utils is a project-internal module — an ImportError is a
        # packaging/path bug, not a missing optional dependency. Returning
        # counts-only (silently dropping wt%/at%) would hide it, so fail loud
        # to match the chemistry-mask / at%-map endpoints in this same file.
        logger.exception("eds_utils import failed in quantify_pixel")
        raise HTTPException(status_code=500, detail=f"eds_utils not available: {e}")

    try:
        counts_dict = {}
        for el_name in elements:
            data = ext.get_element_map(el_name)
            if data is not None and index < len(data):
                pure_el = parse_element_name(el_name)
                counts_dict[pure_el] = np.array([float(data[index])])

        # Single pixel quantification
        wt_pct = counts_to_weight_pct(counts_dict)
        at_pct = weight_pct_to_atomic_pct(wt_pct)

        result = {}
        for el in counts_dict:
            result[el] = {
                "counts": float(counts_dict[el][0]),
                "wt_pct": float(wt_pct[el][0]) if el in wt_pct else 0.0,
                "at_pct": float(at_pct[el][0]) if el in at_pct else 0.0,
            }

        return {
            "row": req.row,
            "col": req.col,
            "data": result,
            "display_mode": req.display_mode,
        }
    except Exception as e:
        logger.exception("Failed to quantify pixel (%d, %d)", req.row, req.col)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/region-quantify")
async def region_quantify(req: RegionQuantifyRequest):
    """Get average EDS composition for a rectangular region of pixels."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")

    ext = get_extractor()
    elements = ext.get_available_elements()
    n_rows, n_cols = ext.get_grid_dimensions()

    # Clamp bounds to valid range
    r0 = max(0, req.row_start)
    r1 = min(n_rows - 1, req.row_end)
    c0 = max(0, req.col_start)
    c1 = min(n_cols - 1, req.col_end)

    if r0 > r1 or c0 > c1:
        raise HTTPException(status_code=400, detail="Invalid region: row_start/col_start must be <= row_end/col_end")

    # Collect linear indices for all pixels in the region
    row_indices = np.arange(r0, r1 + 1)
    col_indices = np.arange(c0, c1 + 1)
    linear_indices = (row_indices[:, None] * n_cols + col_indices[None, :]).ravel()

    try:
        from eds_utils import parse_element_name, counts_to_weight_pct, weight_pct_to_atomic_pct

        # Build counts arrays for the region pixels
        counts_dict = {}
        for el_name in elements:
            data = ext.get_element_map(el_name)
            if data is not None:
                pure_el = parse_element_name(el_name)
                region_counts = data[linear_indices].astype(np.float64)
                counts_dict[pure_el] = region_counts

        if not counts_dict:
            return {"row_start": r0, "row_end": r1, "col_start": c0, "col_end": c1,
                    "n_pixels": len(linear_indices), "data": {}}

        wt_pct = counts_to_weight_pct(counts_dict)
        at_pct = weight_pct_to_atomic_pct(wt_pct)

        result = {}
        for el in counts_dict:
            c_arr = counts_dict[el]
            w_arr = wt_pct.get(el, np.zeros_like(c_arr))
            a_arr = at_pct.get(el, np.zeros_like(c_arr))
            result[el] = {
                "counts":  {"mean": float(np.mean(c_arr)),  "std": float(np.std(c_arr)),  "min": float(np.min(c_arr)),  "max": float(np.max(c_arr))},
                "wt_pct":  {"mean": float(np.mean(w_arr)),  "std": float(np.std(w_arr)),  "min": float(np.min(w_arr)),  "max": float(np.max(w_arr))},
                "at_pct":  {"mean": float(np.mean(a_arr)),  "std": float(np.std(a_arr)),  "min": float(np.min(a_arr)),  "max": float(np.max(a_arr))},
            }

        return {
            "row_start": r0, "row_end": r1,
            "col_start": c0, "col_end": c1,
            "n_pixels": int(len(linear_indices)),
            "data": result,
        }
    except ImportError:
        # Fallback: counts only
        raw = {}
        for el_name in elements:
            data = ext.get_element_map(el_name)
            if data is not None:
                region_counts = data[linear_indices].astype(np.float64)
                raw[el_name] = {
                    "counts": {"mean": float(np.mean(region_counts)), "std": float(np.std(region_counts)),
                               "min": float(np.min(region_counts)), "max": float(np.max(region_counts))},
                }
        return {"row_start": r0, "row_end": r1, "col_start": c0, "col_end": c1,
                "n_pixels": int(len(linear_indices)), "data": raw}


def _map_phase_at(row: int, col: int) -> Optional[dict]:
    """What the CURRENT phase map says about this pixel, and how it decided.

    The suggestion list ranks THIS ONE PIXEL. The map, in its default
    cluster mode, assigns by the mean of the pixel's whole composition
    group. Both are internally correct and they disagree on roughly half
    the pixels of SampleB (measured: 53 % in cluster mode, 1 % in pixel
    mode) — because they answer different questions.

    Nothing on screen used to say so, so the panel appeared to contradict
    the map. It returns the map's verdict alongside the per-pixel ranking
    so the UI can lead with the map and show the single pixel as the
    diagnostic detail it is. The map's answer is the more reliable of the
    two: a single pixel's composition carries several at% of error, which
    is the whole reason the grouping exists.
    """
    store = get_phase_map_store()
    state = store.get_state()
    if state is None:
        return None
    if not (0 <= row < state.n_rows and 0 <= col < state.n_cols):
        return None
    idx = int(state.phase_grid[row, col])
    locked = bool(state.locked_mask[row, col]) if state.locked_mask is not None else False
    if idx < 0:
        return {"phase_index": -1, "cif_filename": None, "formula": None,
                "unclassified": True, "hand_set": locked}
    entry = state.phase_entries[idx]
    return {
        "phase_index": idx,
        "cif_filename": entry.cif_filename,
        "formula": entry.formula,
        "unclassified": False,
        "hand_set": locked,
    }


@router.post("/suggest-phases")
async def suggest_phases(req: PhaseSuggestionRequest):
    """Suggest crystal phases based on EDS chemistry at a pixel.

    Prefers the user's CIF library (``Database/crystal_database.xlsx``) so
    that suggestions are restricted to phases the user has actually
    curated and can index against. Falls back to the hardcoded
    ``DEFAULT_PHASE_LIBRARY`` only when the CIF library is missing /
    unbuildable, so phase suggestion never hard-fails just because the
    user hasn't built their database yet.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")

    try:
        from eds_utils import (
            parse_element_name, counts_to_weight_pct,
            weight_pct_to_atomic_pct, suggest_phases as _suggest_simple,
        )

        ext = get_extractor()
        elements = ext.get_available_elements()
        n_rows, n_cols = ext.get_grid_dimensions()
        # Bounds-check the pixel (siblings do) — an out-of-range index would
        # silently yield empty chemistry and a meaningless phase suggestion.
        if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
            raise HTTPException(
                status_code=400,
                detail=f"pixel ({req.row},{req.col}) outside scan {n_rows}x{n_cols}",
            )
        index = req.row * n_cols + req.col

        counts_dict = {}
        for el_name in elements:
            data = ext.get_element_map(el_name)
            if data is not None and index < len(data):
                pure_el = parse_element_name(el_name)
                counts_dict[pure_el] = np.array([float(data[index])])

        wt = counts_to_weight_pct(counts_dict)
        at = weight_pct_to_atomic_pct(wt)
        at_scalar = {el: float(v[0]) for el, v in at.items()}

        # 1) Try the user's curated CIF library first.
        cif_library = load_cif_phase_library(_crystal_db_path())
        if cif_library:
            # The enrichment gate asks whether an element is enriched over
            # THIS MAP's background, so a single-pixel caller has to supply
            # the map-level statistics. Without them the median of one value
            # is that value and every phase is vetoed.
            try:
                whole_map, _mr, _mc, _fp = _build_at_pct_maps_for_loaded_file()
                matrix_element = infer_matrix_element(whole_map)
                background = background_levels(whole_map)
            except Exception:
                logger.exception("could not derive map-level EDS statistics")
                matrix_element, background = None, None
            cif_hits = suggest_phases_from_cif_library(
                at_scalar, cif_library,
                matrix_element=matrix_element, background=background,
            )
            if cif_hits:
                # Adapter shape: keep `name` for backward-compat callers,
                # add the rich CIF fields so the new EDS UI can show them.
                suggestions = [
                    {
                        "name": h["cif_filename"],
                        "cif_filename": h["cif_filename"],
                        "formula": h["formula"],
                        "space_group": h["space_group"],
                        "space_group_number": h["space_group_number"],
                        "crystal_system": h["crystal_system"],
                        "score": h["score"],
                        "expected": h["expected"],
                        "elements": h["elements"],
                    }
                    for h in cif_hits
                ]
                return {
                    "suggestions": suggestions,
                    "atomic_pct": at_scalar,
                    "library_source": "cif",
                    "library_size": len(cif_library),
                    "map_phase": _map_phase_at(req.row, req.col),
                }

        # 2) Fall back to the hardcoded library so the feature isn't
        # dead-on-arrival for users who haven't built their CIF DB.
        from eds_utils import DEFAULT_PHASE_LIBRARY
        raw_suggestions = _suggest_simple(at_scalar, DEFAULT_PHASE_LIBRARY)
        suggestions = [
            {"name": s.phase_name, "score": s.score, "expected": s.expected_composition}
            for s in raw_suggestions
        ]
        return {
            "suggestions": suggestions,
            "atomic_pct": at_scalar,
            "library_source": "default",
            "library_size": len(DEFAULT_PHASE_LIBRARY),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Phase suggestion failed at (%d, %d)", req.row, req.col)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cif-phases")
async def cif_phases():
    """Return the CIF-derived phase library (filename, formula, space group, composition).

    Used by the EDS frontend to:
    - Show the user how many curated phases are available for matching.
    - Populate manual-assign dropdowns once the phase-map builder lands.

    Returns ``{"phases": [], "source": "none", "message": ...}`` when the
    Excel database hasn't been built yet — the frontend treats that as
    "use the hardcoded library" rather than an error.
    """
    db_path = _crystal_db_path()
    if not db_path.is_file():
        return {
            "phases": [],
            "source": "none",
            "message": "Database/crystal_database.xlsx not found — build the CIF database first",
        }
    library = load_cif_phase_library(db_path)
    if not library:
        return {
            "phases": [],
            "source": "none",
            "message": "CIF database is empty or unparseable",
        }
    phases = [
        {
            "key": entry.key,
            "cif_filename": entry.cif_filename,
            "formula": entry.formula,
            "space_group": entry.space_group,
            "space_group_number": entry.space_group_number,
            "crystal_system": entry.crystal_system,
            "composition": entry.composition,
            "elements": entry.elements,
        }
        for entry in library.values()
    ]
    return {
        "phases": phases,
        "source": "crystal_database.xlsx",
        "count": len(phases),
    }


class ChemMaskFilter(BaseModel):
    element: str
    operator: str  # ">", "<", ">=", "<=", "between"
    min_val: float = 0.0
    max_val: float = 100.0
    unit: str = "at_pct"  # per-filter unit: "counts", "wt_pct", "at_pct"


class ChemMaskRequest(BaseModel):
    filters: List[ChemMaskFilter]
    combine: str = "and"  # "and" or "or"
    margin_px: int = 0    # positive = dilate, negative = erode


@router.post("/chemistry-mask")
async def generate_chemistry_mask(req: ChemMaskRequest):
    """Generate a boolean pixel mask based on EDS chemistry filters.

    Returns a flat boolean array (n_pixels) where True means the pixel
    passes all (AND) or any (OR) of the specified chemistry filters.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")
    if not req.filters:
        raise HTTPException(status_code=400, detail="At least one filter is required")

    ext = get_extractor()
    elements = ext.get_available_elements()
    n_rows, n_cols = ext.get_grid_dimensions()
    n_pixels = n_rows * n_cols

    try:
        from eds_utils import parse_element_name, counts_to_weight_pct, weight_pct_to_atomic_pct

        # Build element data maps
        counts_dict = {}
        for el_name in elements:
            data = ext.get_element_map(el_name)
            if data is not None:
                pure_el = parse_element_name(el_name)
                counts_dict[pure_el] = data.astype(np.float64)

        # Pre-compute unit maps to avoid redundant conversions
        wt_maps_cache = None
        at_maps_cache = None

        def get_unit_maps(unit):
            nonlocal wt_maps_cache, at_maps_cache
            if unit == "wt_pct":
                if wt_maps_cache is None:
                    wt_maps_cache = counts_to_weight_pct(counts_dict)
                return wt_maps_cache
            elif unit == "at_pct":
                if wt_maps_cache is None:
                    wt_maps_cache = counts_to_weight_pct(counts_dict)
                if at_maps_cache is None:
                    at_maps_cache = weight_pct_to_atomic_pct(wt_maps_cache)
                return at_maps_cache
            else:
                return counts_dict

        # Apply each filter with per-filter unit and collect per-filter stats
        filter_masks = []
        filter_stats = []
        for f in req.filters:
            el = parse_element_name(f.element) if f.element else f.element
            unit_maps = get_unit_maps(f.unit)

            if el not in unit_maps:
                filter_masks.append(np.zeros(n_pixels, dtype=bool))
                filter_stats.append({
                    "element": f.element, "unit": f.unit,
                    "operator": f.operator, "n_pass": 0, "pct_pass": 0.0,
                })
                continue

            vals = unit_maps[el]
            if f.operator == ">":
                mask = vals > f.min_val
            elif f.operator == ">=":
                mask = vals >= f.min_val
            elif f.operator == "<":
                mask = vals < f.min_val
            elif f.operator == "<=":
                mask = vals <= f.min_val
            elif f.operator == "between":
                mask = (vals >= f.min_val) & (vals <= f.max_val)
            else:
                mask = vals > f.min_val
            filter_masks.append(mask.astype(bool))
            n_pass = int(np.sum(mask))
            filter_stats.append({
                "element": f.element, "unit": f.unit,
                "operator": f.operator,
                "n_pass": n_pass,
                "pct_pass": round(n_pass / n_pixels * 100, 1) if n_pixels > 0 else 0.0,
            })

        # Combine masks
        if req.combine == "or":
            combined = np.zeros(n_pixels, dtype=bool)
            for m in filter_masks:
                combined |= m
        else:
            combined = np.ones(n_pixels, dtype=bool)
            for m in filter_masks:
                combined &= m

        # Apply morphological margin (dilation/erosion)
        if req.margin_px != 0:
            from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
            struct = generate_binary_structure(2, 1)  # 4-connected
            mask_2d_morph = combined.reshape(n_rows, n_cols)
            if req.margin_px > 0:
                mask_2d_morph = binary_dilation(mask_2d_morph, structure=struct, iterations=req.margin_px)
            else:
                mask_2d_morph = binary_erosion(mask_2d_morph, structure=struct, iterations=abs(req.margin_px))
            combined = mask_2d_morph.flatten().astype(bool)

        n_selected = int(np.sum(combined))
        pct = (n_selected / n_pixels * 100) if n_pixels > 0 else 0

        mask_2d = combined.reshape(n_rows, n_cols)
        mask_img = colormap_array_to_base64(mask_2d.astype(np.float32), "gray")

        return {
            "mask": combined.tolist(),
            "n_selected": n_selected,
            "n_total": n_pixels,
            "percentage": round(pct, 1),
            "preview_image": mask_img,
            "shape": [n_rows, n_cols],
            "filter_stats": filter_stats,
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="eds_utils not available for chemistry mask generation")
    except Exception as e:
        logger.exception("Failed to generate chemistry mask")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/display-modes")
async def display_modes():
    """List available EDS display modes."""
    return {
        "modes": [
            {"id": "counts", "label": "Raw Counts"},
            {"id": "wt_pct", "label": "Weight %"},
            {"id": "at_pct", "label": "Atomic %"},
        ]
    }


# =============================================================================
# Phase-map builder (M3 + M4)
# =============================================================================


class AutoClassifyRequest(BaseModel):
    """Tunables for auto-classify.

    ``mode`` picks how pixels are grouped before a library phase is chosen:

    - ``"cluster"`` (default): cluster the composition, then match each
      cluster's eroded-interior mean. Per-pixel composition on this data
      carries several at% of systematic error, so matching a single pixel
      against nominal stoichiometries separated by ~1 at% is not reliable.
    - ``"pixel"``: match every pixel independently. Kept for comparison.
    """
    tolerance: float = 15.0   # retained for API compatibility; unused by the scorer
    min_score: float = 0.3    # below this -> unclassified
    mode: str = "cluster"
    # None -> chosen by the spatial coherence of the resulting phase map
    n_clusters: Optional[int] = None
    # Box width in pixels for smoothing the composition before clustering.
    # None = the module default. 0 reproduces the pre-2026-08-24 behaviour.
    scale: Optional[int] = None
    # User-authored rules. See backend/api/services/phase_rules.py — they
    # decide which phases may COMPETE for a region, not how well they score.
    rules: Optional[dict] = None
    phase_keys: Optional[List[str]] = None  # None -> every library phase
    # Carry hand-assigned pixels across the re-classify. Default True: the
    # old behaviour silently destroyed them, which is the bug, not the
    # feature. Send False for a deliberate "start over".
    keep_manual_edits: bool = True


def _structure_feature_matrix(at_maps, n_rows: int, n_cols: int,
                              scale: Optional[int]):
    """The exact feature matrix the clustering used, for later re-splitting.

    Built through the clustering module rather than re-derived here, because a
    second implementation that drifts would split a structure on different
    data than the one that created it.
    """
    from backend.api.services.eds_clustering import (
        DEFAULT_SCALE, _feature_matrix, _smooth_maps,
    )

    sc = DEFAULT_SCALE if scale is None else int(scale)
    smoothed = _smooth_maps(at_maps, n_rows, n_cols, sc)
    _els, X = _feature_matrix(smoothed, n_rows * n_cols)
    return X if X.size else None


def _build_at_pct_maps_for_loaded_file() -> tuple[dict, int, int, str]:
    """Compute At.% maps for all elements in the currently loaded file.

    Returns ``({element_symbol: 1D array}, n_rows, n_cols, file_path_for_tagging)``.
    Raises HTTPException 400 if no file is open or eds_utils is unavailable —
    same failure surface as quantify-pixel so the frontend can show the
    same error UX.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")

    try:
        from eds_utils import (
            parse_element_name, counts_to_weight_pct, weight_pct_to_atomic_pct,
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="eds_utils not available — At.% conversion impossible")

    ext = get_extractor()
    elements = ext.get_available_elements()
    n_rows, n_cols = ext.get_grid_dimensions()

    counts_dict: dict = {}
    for el_name in elements:
        data = ext.get_element_map(el_name)
        if data is not None:
            pure_el = parse_element_name(el_name)
            counts_dict[pure_el] = data.astype(np.float64)

    if not counts_dict:
        raise HTTPException(status_code=400, detail="No EDS element data in this file")

    wt_maps = counts_to_weight_pct(counts_dict)
    at_maps = weight_pct_to_atomic_pct(wt_maps)

    # Authoritative file-path tag for the store: the h5_session knows
    # exactly which file is open. Using ``ext.file_path`` is fragile
    # because not every extractor implementation exposes that attribute,
    # and a missing attribute would silently disable phase-map
    # persistence (the autosave needs a path).
    try:
        from backend.api.services.h5_session import get_current_path
        file_path = get_current_path()
    except Exception:
        file_path = None

    return at_maps, int(n_rows), int(n_cols), str(file_path) if file_path else None


def _state_to_response(include_image: bool = True) -> dict:
    """Serialise the current PhaseMapStore state for the frontend."""
    store = get_phase_map_store()
    state = store.get_state()
    if state is None:
        return {"loaded": False}
    palette = palette_hex_for_state(state, _COLOR_OVERRIDES)
    summary = store.phase_summary()
    # Inject the matching colour into each summary entry so the legend
    # and the rendered preview agree without the frontend having to
    # know the palette algorithm.
    for entry in summary:
        idx = entry.get("phase_index", -1)
        entry["color"] = palette[idx] if 0 <= idx < len(palette) else "#3c3c3c"
    response = {
        "loaded": True,
        "n_rows": state.n_rows,
        "n_cols": state.n_cols,
        "tolerance": state.tolerance,
        "min_score": state.min_score,
        "palette": palette,
        "summary": summary,
        "n_phases": len(state.phase_entries),
        "n_classified": int((state.phase_grid >= 0).sum()),
        "n_unclassified": int((state.phase_grid < 0).sum()),
    }
    # EVERY candidate, not only the ones that won pixels. `phase_summary`
    # deliberately skips empty phases (the legend does not need them), but
    # the legend is also the only phase PICKER — so a phase the classifier
    # placed nowhere could not be hand-assigned, which is exactly the case
    # a manual correction exists for ("this particle is beta-AlFeSi, the
    # classifier missed it").
    counts = np.bincount((state.phase_grid + 1).ravel(),
                         minlength=len(state.phase_entries) + 1)
    response["all_phases"] = [
        {
            "phase_index": i,
            "cif_filename": e.cif_filename,
            "formula": e.formula,
            "space_group": e.space_group,
            "crystal_system": e.crystal_system,
            "n_pixels": int(counts[i + 1]),
            "color": palette[i] if i < len(palette) else "#3c3c3c",
            # The nominal composition, so the structure picker can rank
            # candidates against a structure's measured mean without a second
            # round trip per structure.
            "composition": {el: round(float(v), 2)
                            for el, v in (e.composition or {}).items()},
        }
        for i, e in enumerate(state.phase_entries)
    ]
    # Which pixels the user set by hand. Their epistemic status differs from
    # a measured assignment and downstream (indexing) consumes both.
    response["n_locked"] = (int(state.locked_mask.sum())
                            if state.locked_mask is not None else 0)
    # What an undo would take back, so the button can name it instead of
    # asking the user to remember.
    response["undo_label"] = store.undo_label
    response["structures"] = _structures_payload(state)
    if include_image:
        response["image"] = render_phase_map_to_base64(state, _COLOR_OVERRIDES)
        if state.structure_grid is not None:
            response["structure_image"] = render_structure_map_to_base64(state)
    return response


def _structures_payload(state) -> list:
    """One entry per structure: how big, what it is made of, what it is called.

    The composition is the honest content of a structure - it is what grouped
    those pixels in the first place - so the legend can show it without the
    frontend recomputing anything. Ranked CIF candidates come with it, because
    naming a structure is the one action the legend exists for.
    """
    if state.structure_grid is None:
        return []
    import numpy as _np

    n_s = len(state.structure_phase)
    if n_s == 0:
        return []
    flat = state.structure_grid.ravel()
    counts = _np.bincount(flat[flat >= 0], minlength=n_s)
    total = int(state.n_rows * state.n_cols) or 1

    at_maps = None
    try:
        at_maps, n_rows, n_cols, _fp = _build_at_pct_maps_for_loaded_file()
        if n_rows != state.n_rows or n_cols != state.n_cols:
            at_maps = None      # a different file is open; do not mix them
    except Exception:
        at_maps = None

    out = []
    for sid in range(n_s):
        m = flat == sid
        mean = {}
        if at_maps is not None and m.any():
            # Only the elements the grouping actually used. C and O are
            # excluded from the clustering (`_CHEM_IGNORE`), so listing them
            # in a structure's description would imply they helped decide it.
            from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE
            for el, arr in at_maps.items():
                if el in _CHEM_IGNORE:
                    continue
                v = float(_np.asarray(arr, dtype=float).ravel()[m].mean())
                if v >= 0.5:
                    mean[el] = round(v, 2)
        phase_idx = int(state.structure_phase[sid])
        entry = (state.phase_entries[phase_idx]
                 if 0 <= phase_idx < len(state.phase_entries) else None)
        out.append({
            "structure_id": sid,
            "n_pixels": int(counts[sid]) if sid < len(counts) else 0,
            "percentage": round((int(counts[sid]) if sid < len(counts) else 0)
                                / total * 100, 2),
            "mean_at_pct": dict(sorted(mean.items(), key=lambda kv: -kv[1])),
            "phase_index": phase_idx,
            "cif_filename": entry.cif_filename if entry else None,
            "formula": entry.formula if entry else None,
            "color": structure_color_hex(sid),
        })
    return out


@router.post("/auto-classify")
async def auto_classify(req: AutoClassifyRequest):
    """Classify every pixel against the curated CIF library.

    Stores the result in the per-process :class:`PhaseMapStore` and
    returns a rendered PNG plus per-phase summary so the UI can show
    the map immediately. Re-running this endpoint replaces the
    previous classification (any manual edits from M4 are lost — by
    design, the re-classify is the "I want to start over" path).
    """
    cif_library = load_cif_phase_library(_crystal_db_path())
    if not cif_library:
        raise HTTPException(
            status_code=400,
            detail="No CIF library available — build Database/crystal_database.xlsx first",
        )

    # Restrict to the phases the user ticked. Only a strict subset filters —
    # an empty or full selection means "use everything", so a stale UI can
    # never silently narrow the run.
    if req.phase_keys:
        wanted = set(req.phase_keys)
        subset = {k: v for k, v in cif_library.items() if k in wanted}
        if not subset:
            raise HTTPException(
                status_code=400,
                detail=(
                    "None of the selected phases exist in the CIF library. "
                    "Reload the phase list and try again."
                ),
            )
        cif_library = subset

    mode = (req.mode or "cluster").lower()
    if mode not in ("cluster", "pixel"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode {req.mode!r} — expected 'cluster' or 'pixel'.",
        )

    # Synchronous HDF5 reads plus, in cluster mode, one KMeans fit per
    # candidate k. Measured 3.5 s at the user's real scan size (53 868 px);
    # on the event loop that stalls /health, the WebSocket pumps and every
    # other request for the duration.
    return await asyncio.to_thread(
        _auto_classify_blocking, req, cif_library, mode,
    )


def _auto_classify_blocking(req, cif_library, mode: str) -> dict:
    """The CPU-bound body of :func:`auto_classify`, run off the event loop."""
    at_maps, n_rows, n_cols, file_path = _build_at_pct_maps_for_loaded_file()

    clusters_payload: List[dict] = []
    k_used = 0

    from backend.api.services.phase_rules import rule_set_from_dict
    rule_set = rule_set_from_dict(req.rules)
    # Participation is filtered HERE and only here. The candidate list is built
    # in two places (this route and `auto_classify_pixels`); narrowing it in
    # one would make the two modes number phases differently, and that number
    # is what gets persisted and handed to indexing.
    if rule_set is not None and rule_set.phase_keys is not None:
        keep = set(rule_set.phase_keys)
        cif_library = {k: v for k, v in cif_library.items() if k in keep}             if isinstance(cif_library, dict) else cif_library

    if mode == "pixel":
        phase_grid, score_grid, candidates, ambiguous = auto_classify_pixels(
            at_pct_per_element=at_maps,
            n_rows=n_rows,
            n_cols=n_cols,
            cif_library=cif_library,
            tolerance=req.tolerance,
            min_score=req.min_score,
            rule_set=rule_set,
        )
    else:
        # Same helper auto_classify_pixels uses — the index into this list is
        # the phase id persisted in the sidecar and handed to indexing, so
        # the two modes must never build it differently.
        candidates = candidates_for(cif_library, at_maps.keys())
        phase_grid, cluster_grid, matches, k_used = cluster_and_match(
            at_pct_per_element=at_maps,
            n_rows=n_rows,
            n_cols=n_cols,
            candidates=candidates,
            k=req.n_clusters,
            min_score=req.min_score,
            scale=req.scale,
            rule_set=rule_set,
        )
        score_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        ambiguous = np.zeros((n_rows, n_cols), dtype=bool)
        for m in matches:
            sel = (cluster_grid == m.cluster_id)
            score_grid[sel] = m.score
            if m.ambiguous:
                ambiguous[sel] = True
            clusters_payload.append({
                "cluster_id": m.cluster_id,
                "n_pixels": m.n_pixels,
                "percentage": round(m.n_pixels / (n_rows * n_cols) * 100, 2),
                # Only the elements that actually carry signal — a full
                # dump of every trace element makes this unreadable.
                "mean_at_pct": {el: round(v, 2)
                                for el, v in sorted(m.mean_at_pct.items(),
                                                    key=lambda kv: -kv[1])
                                if v >= 0.5},
                "cif_filename": (candidates[m.phase_index].cif_filename
                                 if m.phase_index >= 0 else None),
                "formula": (candidates[m.phase_index].formula
                            if m.phase_index >= 0 else None),
                "phase_index": m.phase_index,
                "score": round(m.score, 4),
                "ambiguous": m.ambiguous,
                "runners_up": [
                    {"cif_filename": candidates[i].cif_filename,
                     "score": round(s, 4)}
                    for i, s in m.runners_up
                ],
            })

    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=(
                "No CIF phase has elements that are a subset of the measured EDS elements — "
                "auto-classify cannot match any phase. Check that the CIF database covers "
                "the elements present in this scan."
            ),
        )

    # The structure layer: which pixels belong together by composition alone,
    # before anything is named. Only cluster mode produces it - per-pixel
    # matching has no notion of a group.
    structure_grid = None
    structure_phase = None
    structure_features = None
    if mode != "pixel":
        structure_grid = np.asarray(cluster_grid).reshape(n_rows, n_cols)
        n_struct = int(structure_grid.max()) + 1 if structure_grid.size else 0
        structure_phase = [-1] * n_struct
        for m in matches:
            if 0 <= m.cluster_id < n_struct:
                structure_phase[m.cluster_id] = int(m.phase_index)
        structure_features = _structure_feature_matrix(at_maps, n_rows, n_cols,
                                                       req.scale)

    store = get_phase_map_store()
    store.set_classification(
        phase_grid=phase_grid,
        score_grid=score_grid,
        phase_entries=candidates,
        tolerance=req.tolerance,
        min_score=req.min_score,
        file_path=file_path,
        preserve_locked=bool(req.keep_manual_edits),
        structure_grid=structure_grid,
        structure_phase=structure_phase,
        structure_features=structure_features,
    )
    response = _state_to_response(include_image=True)
    response["mode"] = mode
    response["k_used"] = k_used
    response["clusters"] = clusters_payload
    response["n_ambiguous"] = int(np.asarray(ambiguous).sum())
    return response


@router.get("/phase-map")
async def get_phase_map(include_image: bool = True):
    """Return the current phase map (image + legend + summary)."""
    return _state_to_response(include_image=include_image)


@router.delete("/phase-map")
async def clear_phase_map():
    """Drop the stored phase map. Used when starting from scratch."""
    get_phase_map_store().clear()
    return {"loaded": False, "cleared": True}


def _phase_grid_mismatch(state):
    """``(stored_shape, active_shape)`` when the stored phase map is NOT on the
    grid of the active dataset, else ``None``.

    The map lives on whatever grid it was classified on, while every caller
    below indexes it with coordinates belonging to the ACTIVE dataset — and
    those became crop-local the moment that dataset is a crop. When the two
    grids disagree, an in-range ``(row, col)`` names a DIFFERENT pixel, so the
    read and both writes have to notice. One helper for all three so they
    cannot drift apart; what they DO about it differs (see
    ``_refuse_write_on_grid_mismatch``).

    With no file open there is no active grid to compare against, and nothing
    can be cropped either: ``None``, which leaves every caller exactly where it
    stood before there was such a thing as a crop. Anything else that goes
    wrong PROPAGATES — ``get_active_extractor`` is deliberately fail-loud when
    the crop window does not fit the open file, and swallowing that here would
    hand the caller a grid nobody can name.
    """
    stored = tuple(int(v) for v in np.shape(state.phase_grid)[:2])
    if not is_open():
        return None
    active = tuple(int(v) for v in get_extractor().get_grid_dimensions())
    return None if stored == active else (stored, active)


def _refuse_write_on_grid_mismatch(state) -> None:
    """Abort a phase-map write whose coordinates belong to another grid.

    The read side (``_probe_phase_at_safe``) degrades to "no value" on exactly
    this condition — its own ``except`` also turns a fail-loud grid lookup into
    "no phase". A write may not: a wrong read is a wrong pixel on screen for as
    long as the tooltip is open, a wrong write is persistent state painted over
    a region of the scan the user never looked at and cannot see is wrong.
    """
    try:
        mismatch = _phase_grid_mismatch(state)
    except Exception as exc:
        logger.warning("refusing a phase-map write: the grid of the active "
                       "dataset could not be determined", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=(f"The grid of the active dataset could not be determined "
                    f"({exc}) — refusing to edit the phase map until it can."),
        )
    if mismatch is None:
        return
    stored, active = mismatch
    logger.warning(
        "refusing a phase-map write: the stored map is %s but the active "
        "dataset is %s", stored, active,
    )
    raise HTTPException(
        status_code=400,
        detail=(
            f"The stored phase map is {stored[0]}x{stored[1]} but the active "
            f"dataset is {active[0]}x{active[1]} — the coordinates of this "
            f"edit belong to a different grid. Re-run auto-classify on the "
            f"dataset you are looking at before painting it."
        ),
    )


class AssignRegionRequest(BaseModel):
    """Paint a rectangular region with a phase id (-1 = unclassified)."""
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    phase_index: int  # -1 -> mark unclassified


class AssignPolygonRequest(BaseModel):
    """Paint a closed polygon with a phase id.

    ``vertices`` are ``[col, row]`` pairs in pixel coordinates — same
    convention as the SVG overlay in the EDS page (x = column, y = row).
    Order doesn't matter (the polygon is auto-closed) but must form a
    simple non-self-intersecting outline; matplotlib's path-contains
    test handles concave outlines fine but cannot fix figure-of-eight
    self-crossings (the inside/outside is undefined).
    """
    vertices: List[List[float]]
    phase_index: int  # -1 -> mark unclassified


@router.post("/phase-map/assign-polygon")
async def assign_polygon(req: AssignPolygonRequest):
    """Manually overwrite a polygon of pixels with a chosen phase id.

    Used by the polygon-paint mode in the EDS page when a rectangle
    is too coarse for what the user wants to mark. The polygon is
    rasterised on the backend (matplotlib's ``Path.contains_points``)
    so the frontend doesn't need to ship the resulting mask back —
    only the vertex list, which is always small.
    """
    if len(req.vertices) < 3:
        raise HTTPException(
            status_code=400,
            detail="Polygon needs at least 3 vertices",
        )
    store = get_phase_map_store()
    state = store.get_state()
    if state is None:
        raise HTTPException(status_code=400, detail="No phase map loaded — run auto-classify first")
    _refuse_write_on_grid_mismatch(state)

    try:
        from matplotlib.path import Path as _MplPath
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="matplotlib not available — polygon assign cannot rasterise the mask",
        )

    # Build a (row, col) grid of pixel centres and ask matplotlib which
    # ones lie inside the polygon. Note: contains_points expects
    # ``(x, y)`` pairs and our request gives ``(col, row)`` which is
    # already in (x, y) order — no transpose needed.
    rows = np.arange(state.n_rows)
    cols = np.arange(state.n_cols)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    points = np.column_stack([cc.ravel() + 0.5, rr.ravel() + 0.5])

    polygon = _MplPath(np.array(req.vertices, dtype=float))
    inside = polygon.contains_points(points).reshape(state.n_rows, state.n_cols)

    try:
        n_overwritten = store.assign_mask(
            mask=inside,
            target_phase_index=int(req.phase_index),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response = _state_to_response(include_image=True)
    response["n_overwritten"] = n_overwritten
    return response


@router.post("/phase-map/assign-region")
async def assign_region(req: AssignRegionRequest):
    """Manually overwrite a rectangle of pixels with a chosen phase id."""
    store = get_phase_map_store()
    state = store.get_state()
    if state is None:
        raise HTTPException(status_code=400, detail="No phase map loaded — run auto-classify first")
    _refuse_write_on_grid_mismatch(state)
    try:
        n_overwritten = store.assign_region(
            row_start=req.row_start,
            row_end=req.row_end,
            col_start=req.col_start,
            col_end=req.col_end,
            target_phase_index=req.phase_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    response = _state_to_response(include_image=True)
    response["n_overwritten"] = n_overwritten
    return response


@router.get("/phase-map/indexing-config")
async def phase_map_indexing_config():
    """Build a hand-off payload the Indexing page can consume directly.

    Returns three things derived from the current phase map:

    - ``cif_filenames``: unique CIF filenames that appear in the map
      (so the Indexing UI can pre-select them from its discovered
      phase files — the backend doesn't know the full path yet, the
      frontend resolves filename → discovered file path).
    - ``combined_mask``: flat list of 0/1, length ``n_rows * n_cols``,
      with 1 wherever any phase was assigned. The Indexing page wires
      this into selection_mode='mask' so unclassified pixels are
      skipped.
    - ``per_phase_masks``: ``{cif_filename: [0/1...]}`` for the future
      per-phase routing mode, where each phase is tested only on its
      own classified pixels. Frontend ignores this for now but the
      shape is stable so the UI can adopt it later without a backend
      change.

    Returns ``loaded: false`` (200, not 4xx) when nothing is
    classified — that's a "use the page normally" signal, not an
    error.
    """
    store = get_phase_map_store()
    state = store.get_state()
    if state is None:
        return {"loaded": False}

    entries, masks_2d = store.get_phase_masks()

    # Combined mask: union of every classified pixel across every phase.
    if masks_2d:
        combined_2d = np.zeros((state.n_rows, state.n_cols), dtype=bool)
        for m in masks_2d.values():
            combined_2d |= m
    else:
        combined_2d = np.zeros((state.n_rows, state.n_cols), dtype=bool)

    cif_filenames: List[str] = []
    seen_filenames: set = set()
    per_phase_masks: Dict[str, List[int]] = {}
    for idx, entry in enumerate(entries):
        m = masks_2d.get(idx)
        if m is None:
            continue
        if entry.cif_filename in seen_filenames:
            # Two distinct phase entries that point at the same CIF —
            # union their masks under the same filename so the Indexing
            # page sees one entry.
            existing = np.array(per_phase_masks[entry.cif_filename], dtype=bool)
            existing |= m.flatten()
            per_phase_masks[entry.cif_filename] = existing.astype(int).tolist()
        else:
            seen_filenames.add(entry.cif_filename)
            cif_filenames.append(entry.cif_filename)
            per_phase_masks[entry.cif_filename] = m.flatten().astype(int).tolist()

    n_classified = int(combined_2d.sum())
    n_total = state.n_rows * state.n_cols

    return {
        "loaded": True,
        "n_rows": state.n_rows,
        "n_cols": state.n_cols,
        "cif_filenames": cif_filenames,
        "combined_mask": combined_2d.flatten().astype(int).tolist(),
        "per_phase_masks": per_phase_masks,
        "n_classified": n_classified,
        "n_total": n_total,
        "percentage": round(n_classified / n_total * 100, 1) if n_total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Hover-tooltip probe (T10) — multi-layer pixel inspection in one roundtrip.
# ---------------------------------------------------------------------------

def _read_native_bc_safe() -> Optional[np.ndarray]:
    """Read native Band Contrast as a 2-D array (rows × cols), or None if absent.

    Test-mockable wrapper around virtual_images._read_native_band_contrast() so
    the probe handler can stay a one-liner. Returns None on any failure (file
    not open, BC group missing, IO error) — the tooltip just shows ``BC: --``.
    """
    try:
        from backend.api.routes.virtual_images import _read_native_band_contrast
        return _read_native_band_contrast()
    except Exception:
        return None


def _bc_grid_for(ext) -> "Optional[np.ndarray]":
    """Native Band Contrast on the ACTIVE dataset's grid, or None.

    The probe and the linescan index this grid with coordinates that belong to
    ``ext`` — crop-local when the active dataset is a crop.
    ``_read_native_bc_safe`` reads the raw HDF5 file and always returns the
    FULL scan, so under a crop the same (row, col) lands on a different pixel
    than the element maps do. A half-cropped response is worse than either
    alternative, so on the crop path read BC through the proxy, which cuts it
    to the same window.

    Off the crop path this is literally ``_read_native_bc_safe()`` — same
    reader, same values, still monkeypatchable by the existing tests.
    """
    from backend.api.services.cropped_extractor import CroppedExtractor

    if not isinstance(ext, CroppedExtractor):
        return _read_native_bc_safe()
    try:
        bc = ext.get_band_contrast_map()
    except Exception:
        # The extractor's reader is fail-loud on a header/data mismatch. A
        # hover tooltip must not 500, but it must not show the full scan's
        # value either — report "no BC" and say why in the log.
        logger.warning("cropped Band Contrast read failed — reporting no BC",
                       exc_info=True)
        return None
    return None if bc is None else np.asarray(bc)


def _probe_phase_at_safe(row: int, col: int) -> Optional[dict]:
    """Return ``{ 'id': int, 'name': str }`` for the phase at (row, col), or None.

    Test-mockable wrapper around the phase_map_store. Phase map may not exist
    (no auto-classify run yet); that returns None and the tooltip omits the
    Phase row. The store uses ``-1`` for unclassified pixels.
    """
    try:
        store = get_phase_map_store()
        state = store.get_state()
        if state is None:
            return None
        grid = state.phase_grid  # rows × cols int32; -1 = unclassified
        # The stored map is on whatever grid it was classified on. If that is
        # not the grid the caller is indexing — a full-scan map probed
        # with crop-local coordinates, say — an in-range (row, col) would
        # return a DIFFERENT pixel's phase. A READ may degrade to "no value";
        # the two writes share this check and refuse outright instead.
        mismatch = _phase_grid_mismatch(state)
        if mismatch is not None:
            logger.warning(
                "phase map is %s but the active dataset is %s — omitting the "
                "phase from the probe rather than reading the wrong pixel",
                mismatch[0], mismatch[1],
            )
            return None
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return None
        phase_idx = int(grid[row, col])
        if phase_idx < 0:
            return None
        entries = state.phase_entries
        if 0 <= phase_idx < len(entries):
            entry = entries[phase_idx]
            # CifPhaseEntry has formula + cif_filename — prefer the formula
            # (e.g. "Al7Cu2Fe") as the human-facing name, fall back to the
            # filename so we never return an empty string.
            name = (
                getattr(entry, "formula", None)
                or getattr(entry, "cif_filename", None)
                or f"Phase {phase_idx}"
            )
            return {"id": phase_idx, "name": str(name)}
        return {"id": phase_idx, "name": f"Phase {phase_idx}"}
    except Exception:
        # The explicit mismatch above logs; this catch did not, and it is the
        # branch that swallows a RAISED mismatch — `_phase_grid_mismatch` asks
        # `get_active_extractor`, which is deliberately fail-loud when the crop
        # window does not fit the open file. Degrading a read to "no value" is
        # allowed; doing it in silence is the one thing the spec forbids, and
        # this was the only place left doing it.
        logger.warning("could not read the phase at (%s, %s) — omitting it "
                       "from the probe", row, col, exc_info=True)
        return None


@router.post("/probe")
async def probe(req: ProbeRequest):
    """Return all-layer values at a pixel in ONE roundtrip.

    Used by the EDS-page hover tooltip (T12). Currently covers:

    - EDS elements (counts / wt% / at%)
    - BC (native band contrast value)
    - Phase (if a phase map was generated)

    V-BSE and electron-image values are intentionally omitted to keep the
    response time inside the 100 ms tooltip budget.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")

    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()
    if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
        raise HTTPException(
            status_code=422,
            detail=f"pixel ({req.row}, {req.col}) outside scan {n_rows}x{n_cols}",
        )

    # --- Element values (reuse quantify_pixel's math) ---
    elements_out: Dict[str, dict] = {}
    try:
        from eds_utils import parse_element_name, counts_to_weight_pct, weight_pct_to_atomic_pct
        index = req.row * n_cols + req.col
        counts_dict: Dict[str, np.ndarray] = {}
        for el_name in ext.get_available_elements():
            data = ext.get_element_map(el_name)
            if data is not None and index < len(data):
                pure_el = parse_element_name(el_name)
                counts_dict[pure_el] = np.array([float(data[index])])
        wt_pct = counts_to_weight_pct(counts_dict) if counts_dict else {}
        at_pct = weight_pct_to_atomic_pct(wt_pct) if wt_pct else {}
        for el, c in counts_dict.items():
            elements_out[el] = {
                "counts": float(c[0]),
                "wt_pct": float(wt_pct.get(el, [0.0])[0]) if el in wt_pct else 0.0,
                "at_pct": float(at_pct.get(el, [0.0])[0]) if el in at_pct else 0.0,
            }
    except ImportError:
        # eds_utils is a project-internal module — failing to import it is a
        # real packaging bug. Return counts only (the probe is multi-layer and
        # must not 500 the other layers), but DON'T fabricate wt%/at%=0.0
        # (that reads as a real measurement of zero). Omit them entirely.
        logger.exception("eds_utils import failed in EDS probe — returning counts only")
        index = req.row * n_cols + req.col
        for el_name in ext.get_available_elements():
            data = ext.get_element_map(el_name)
            if data is not None and index < len(data):
                elements_out[el_name] = {
                    "counts": float(data[index]),
                    "wt_pct": None,
                    "at_pct": None,
                }
    except Exception:
        logger.exception("EDS probe element extraction failed at (%d, %d)", req.row, req.col)
        # Don't fail the whole probe — return empty elements + the other layers.
        elements_out = {}

    # --- BC (native) ---
    bc_value: Optional[float] = None
    bc_grid = _bc_grid_for(ext)
    # Only ever "native" or None here — a hover tooltip must stay cheap, so we
    # never compute BC on the probe path. Computed BC surfaces via the BC layer
    # (the /band-contrast endpoint), not the probe.
    bc_source: Optional[str] = "native" if bc_grid is not None else None
    if bc_grid is not None and bc_grid.ndim == 2:
        if 0 <= req.row < bc_grid.shape[0] and 0 <= req.col < bc_grid.shape[1]:
            bc_value = float(bc_grid[req.row, req.col])

    # --- Phase ---
    phase_info = _probe_phase_at_safe(req.row, req.col)

    return {
        "row": req.row,
        "col": req.col,
        "elements": elements_out,
        "bc": bc_value,
        "bc_source": bc_source,
        "phase": phase_info,
        "display_mode": req.display_mode,
    }


# ---------------------------------------------------------------------------
# Linescan (T16) — per-layer profile arrays along a line.
# ---------------------------------------------------------------------------


class LinescanRequest(BaseModel):
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    n_samples: int = 128
    layers: List[str]
    display_mode: str = "at_pct"


@router.post("/linescan")
async def linescan(req: LinescanRequest):
    """Sample values for each requested layer along a line from (start) → (end).

    Returns ``{ samples: int, series: { layer_id: [values...] } }``. Values for
    an unavailable / unknown layer are filled with None so the frontend Profile-
    Plot can render a "no data" gap without an extra roundtrip. EDS element
    layers use the ``eds-<element>`` id convention from the layer-stack design
    (matches what /api/eds/layers returns); BC uses the literal id ``bc``.
    V-BSE and electron-image layers are out of scope for T16 (would need extra
    plumbing into the EBSD signal), so any id outside ``bc`` / ``eds-*`` falls
    through to the unknown-layer branch and returns a None-array.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No file open")

    # Clamp sample count to keep payload reasonable.
    n = max(2, min(512, int(req.n_samples)))

    # Short-circuit on empty layers — no need to touch the extractor.
    if not req.layers:
        return {
            "samples": n,
            "series": {},
            "display_mode": req.display_mode,
        }

    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()
    rs = np.linspace(req.start_row, req.end_row, n)
    cs = np.linspace(req.start_col, req.end_col, n)

    # scipy is preferred for proper bilinear sampling; fall back to nearest
    # neighbour if scipy.ndimage is unavailable in the env.
    try:
        from scipy.ndimage import map_coordinates

        def sample(arr2d):
            return map_coordinates(
                arr2d, np.vstack([rs, cs]), order=1, mode="nearest"
            ).tolist()
    except Exception:
        def sample(arr2d):
            ri = np.clip(np.round(rs).astype(int), 0, arr2d.shape[0] - 1)
            ci = np.clip(np.round(cs).astype(int), 0, arr2d.shape[1] - 1)
            return arr2d[ri, ci].tolist()

    series: Dict[str, list] = {}

    # Pre-load BC once if requested (any 'bc' layer).
    bc_grid = None
    if "bc" in req.layers:
        bc_grid = _bc_grid_for(ext)

    # Pre-quantify the full EDS grids for the requested elements + display_mode.
    # Cheaper than per-pixel quantify for n samples — single matrix conversion
    # then reshape-and-sample per layer.
    element_grids: Dict[str, np.ndarray] = {}
    requested_elements = [l for l in req.layers if l.startswith("eds-")]
    if requested_elements:
        try:
            from eds_utils import (
                parse_element_name,
                counts_to_weight_pct,
                weight_pct_to_atomic_pct,
            )

            elements = ext.get_available_elements()
            counts_dict: Dict[str, np.ndarray] = {}
            for el_name in elements:
                data = ext.get_element_map(el_name)
                if data is not None:
                    pure_el = parse_element_name(el_name)
                    counts_dict[pure_el] = np.asarray(data, dtype=np.float32)
            if req.display_mode == "counts":
                full = counts_dict
            elif req.display_mode == "wt_pct":
                full = counts_to_weight_pct(counts_dict)
            else:  # at_pct (default)
                wt = counts_to_weight_pct(counts_dict)
                full = weight_pct_to_atomic_pct(wt)
            # Reshape each 1D map to 2D for sampling.
            for lid in requested_elements:
                el_label = lid[len("eds-"):]
                key = parse_element_name(el_label)
                if key in full:
                    grid_2d = np.asarray(full[key], dtype=np.float32).reshape(
                        n_rows, n_cols
                    )
                    element_grids[lid] = grid_2d
        except Exception:
            # eds_utils unavailable or conversion failed — leave element_grids
            # empty so requested element series come back as None-arrays.
            pass

    for lid in req.layers:
        if lid == "bc":
            if bc_grid is not None and bc_grid.ndim == 2:
                series[lid] = sample(bc_grid)
            else:
                series[lid] = [None] * n
        elif lid.startswith("eds-"):
            if lid in element_grids:
                series[lid] = sample(element_grids[lid])
            else:
                series[lid] = [None] * n
        else:
            # Unknown layer kind — out of scope for T16 (V-BSE, electron,
            # phase, etc.). Return None-array so the Profile-Plot can show
            # "no data" without an error.
            series[lid] = [None] * n

    return {
        "samples": n,
        "series": series,
        "display_mode": req.display_mode,
    }


# =============================================================================
# Seeded selection ("magic wand")
# =============================================================================


class WandFieldRequest(BaseModel):
    """Seed a selection at one pixel."""
    row: int
    col: int
    smooth: int = 5     # composition smoothing window; see eds_wand's docstring


class WandAssignRequest(BaseModel):
    """Commit a previewed selection.

    ``mask_b64`` is the exact selection the user saw, packed with
    ``np.packbits``. Re-deriving it here from (row, col, threshold) would be
    smaller, but it would also mean the committed region is whatever THIS
    code computes rather than what the preview showed — and for a manual
    correction, what-you-saw-is-what-you-get matters more than the bytes.
    """
    mask_b64: str
    phase_index: int    # -1 marks the region unclassified


@router.post("/phase-map/wand-field")
async def wand_field_endpoint(req: WandFieldRequest):
    """Distance-from-seed field for a live selection preview.

    Returned once per click; the client then thresholds and flood-fills it
    locally on every slider tick (measured ~2.8 ms for 485 k px), so the
    preview needs no round trip and no debouncing.
    """
    at_maps, n_rows, n_cols, _fp = _build_at_pct_maps_for_loaded_file()
    if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"pixel ({req.row},{req.col}) outside scan {n_rows}x{n_cols}",
        )
    try:
        field, scale, growth, seed_comp = await asyncio.to_thread(
            wand_field, at_maps, n_rows, n_cols, req.row, req.col,
            max(1, int(req.smooth)),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not growth or growth[-1]["n_pixels"] == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "This pixel has no EDS measurement, so nothing can be "
                "selected from it."
            ),
        )

    import base64
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "seed": {"row": req.row, "col": req.col},
        "seed_at_pct": {el: round(v, 2) for el, v in seed_comp.items() if v >= 0.05},
        # uint8 field, row-major. Real distance in at% = value * scale.
        "field_b64": base64.b64encode(field.tobytes()).decode("ascii"),
        "scale": scale,
        "growth": growth,
        # "every pixel like this one", ignoring connectivity — a phase is
        # rarely one blob, and the connected fill needs a pass per particle.
        "growth_all": global_growth_curve(field),
    }


@router.post("/phase-map/wand-stats")
async def wand_stats_endpoint(req: WandAssignRequest):
    """Composition and enrichment of a previewed selection, before committing."""
    at_maps, n_rows, n_cols, _fp = _build_at_pct_maps_for_loaded_file()
    mask = _unpack_mask(req.mask_b64, n_rows, n_cols)
    bg = background_levels(at_maps)
    return selection_stats(at_maps, mask, background=bg)


@router.post("/phase-map/wand-assign")
async def wand_assign_endpoint(req: WandAssignRequest):
    """Commit the selection to the phase map."""
    store = get_phase_map_store()
    state = store.get_state()
    if state is None:
        raise HTTPException(
            status_code=400, detail="No phase map loaded — run auto-classify first")
    # Same guard the rectangle and polygon writes use: a write whose
    # coordinates belong to another grid paints a region of the scan the
    # user never looked at.
    _refuse_write_on_grid_mismatch(state)
    mask = _unpack_mask(req.mask_b64, state.n_rows, state.n_cols)
    try:
        n = store.assign_mask(mask, int(req.phase_index))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    response = _state_to_response(include_image=True)
    response["n_assigned"] = n
    return response


def _unpack_mask(mask_b64: str, n_rows: int, n_cols: int) -> np.ndarray:
    """Bit-packed selection -> (n_rows, n_cols) bool.

    Fails loud on a size mismatch: a silently truncated or padded mask would
    paint a region of the scan the user never selected.
    """
    import base64
    try:
        raw = base64.b64decode(mask_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="mask_b64 is not valid base64")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    n_px = n_rows * n_cols
    if bits.size < n_px:
        raise HTTPException(
            status_code=400,
            detail=(f"selection covers {bits.size} pixels, the map has {n_px} — "
                    f"reload the map and try again"),
        )
    return bits[:n_px].astype(bool).reshape(n_rows, n_cols)


class ReplacePhaseRequest(BaseModel):
    """Repoint every pixel of one phase at another, map-wide."""
    from_phase_index: int
    to_phase_index: int


@router.post("/phase-map/replace-phase")
async def replace_phase_endpoint(req: ReplacePhaseRequest):
    """The correction a seeded selection cannot make.

    The recorded case is "sd_0302719 won 55 % of my map and it should be
    Al" — lassoing 55 % of a map by hand is not a workflow.
    """
    store = get_phase_map_store()
    state = store.get_state()
    if state is None:
        raise HTTPException(
            status_code=400, detail="No phase map loaded — run auto-classify first")
    _refuse_write_on_grid_mismatch(state)
    try:
        n = store.replace_phase(int(req.from_phase_index), int(req.to_phase_index))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    response = _state_to_response(include_image=True)
    response["n_replaced"] = n
    return response


@router.post("/phase-map/undo")
async def undo_endpoint():
    """Take back the last phase-map change. One level, and it redoes."""
    store = get_phase_map_store()
    if not store.undo():
        raise HTTPException(status_code=400, detail="Nothing to undo.")
    return _state_to_response(include_image=True)


class PhaseColorsRequest(BaseModel):
    """Phase-name -> "#rrggbb". An empty dict clears every override."""
    overrides: Dict[str, str]


@router.post("/phase-map/colors")
async def set_phase_colors(req: PhaseColorsRequest):
    """Pin colours to phase names for the EDS map.

    Same names and the same defaults as the EBSD phase map, so a phase keeps
    its colour across both pages. A malformed value is ignored rather than
    rejected — a bad colour should not cost the user their classification.
    """
    global _COLOR_OVERRIDES
    _COLOR_OVERRIDES = {str(k): str(v) for k, v in (req.overrides or {}).items()}
    state = get_phase_map_store().get_state()
    if state is None:
        return {"loaded": False, "n_overrides": len(_COLOR_OVERRIDES)}
    response = _state_to_response(include_image=True)
    response["n_overrides"] = len(_COLOR_OVERRIDES)
    return response


# --- Structures (2026-08-24) ------------------------------------------------
#
# A structure is a group of pixels that belong together by composition alone,
# before anything is named. Naming it is one click; the four tools below are
# for the cases where the grouping itself is wrong.


class AssignStructureRequest(BaseModel):
    structure_id: int
    phase_index: int          # -1 clears the name


class MergeStructuresRequest(BaseModel):
    keep_id: int
    drop_id: int


class SplitStructureRequest(BaseModel):
    structure_id: int
    n_parts: int = 2


class GrowStructureRequest(BaseModel):
    structure_id: int
    n_pixels: int             # negative shrinks


class SnapEdgesRequest(BaseModel):
    strength: float = 1.0     # erosion radius in px: how wide a band is re-decided


def _structure_op(fn, *args):
    """Run one structure operation and return the refreshed map.

    Every one of them shares the same failure surface, so they share the same
    translation of it: a missing map or a stale composition is a 400 the
    frontend can show, not a 500.
    """
    try:
        n = fn(*args)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    response = _state_to_response(include_image=True)
    response["n_changed"] = int(n)
    return response


@router.post("/phase-map/structure/assign")
async def assign_structure_endpoint(req: AssignStructureRequest):
    """Name a structure: every pixel of it becomes that phase."""
    store = get_phase_map_store()
    return _structure_op(store.assign_structure, req.structure_id, req.phase_index)


@router.post("/phase-map/structure/merge")
async def merge_structures_endpoint(req: MergeStructuresRequest):
    """Fold one structure into another.

    The most-used boundary tool: over-segmentation is the expected error,
    because a small particle reads as a dilution gradient rather than a
    plateau and gets cut into concentric rings.
    """
    store = get_phase_map_store()
    return _structure_op(store.merge_structures, req.keep_id, req.drop_id)


@router.post("/phase-map/structure/split")
async def split_structure_endpoint(req: SplitStructureRequest):
    """Re-cluster one structure's own pixels, leaving the rest of the map alone."""
    store = get_phase_map_store()
    return _structure_op(store.split_structure, req.structure_id, req.n_parts)


@router.post("/phase-map/structure/grow")
async def grow_structure_endpoint(req: GrowStructureRequest):
    """Move one structure's boundary out (positive) or in (negative)."""
    store = get_phase_map_store()
    return _structure_op(store.grow_structure, req.structure_id, req.n_pixels)


@router.post("/phase-map/structure/snap")
async def snap_edges_endpoint(req: SnapEdgesRequest):
    """Let every boundary relax onto the nearest strong chemistry edge."""
    store = get_phase_map_store()
    return _structure_op(store.snap_structure_edges, req.strength)


class StructureAtRequest(BaseModel):
    row: int
    col: int


def _structure_detail(state, sid: int) -> dict:
    """Everything worth knowing about one structure.

    Computed on demand rather than for every structure on every response: the
    composition spread and the neighbour scan are O(pixels x elements) each,
    which is nothing for one structure and real work for twenty on a 485k-px
    map.
    """
    from scipy import ndimage as _ndi

    from backend.api.services.chemistry_score import background_levels
    from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE
    from backend.api.services.eds_wand import selection_stats

    grid = state.structure_grid
    mask2d = grid == sid
    n_px = int(mask2d.sum())
    detail = {
        "structure_id": sid,
        "n_pixels": n_px,
        "percentage": round(n_px / max(1, state.n_rows * state.n_cols) * 100, 2),
        "color": structure_color_hex(sid),
        "phase_index": int(state.structure_phase[sid]),
        # The phase NAME, not only its index. A rule is keyed on the
        # entry key rather than a position, so anything seeding a rule
        # from this structure needs the name; without it the "rule from
        # this structure" path has nothing to key on.
        "cif_filename": (
            state.phase_entries[state.structure_phase[sid]].cif_filename
            if 0 <= state.structure_phase[sid] < len(state.phase_entries)
            else None),
        "formula": (
            state.phase_entries[state.structure_phase[sid]].formula
            if 0 <= state.structure_phase[sid] < len(state.phase_entries)
            else None),
        "elements": [],
        "pieces": [],
        "neighbours": [],
        "candidates": [],
    }
    if n_px == 0:
        return detail

    # --- connected pieces: three rings of one particle look like three
    # structures, and this is where that becomes visible.
    labelled, n_pieces = _ndi.label(mask2d)
    sizes = sorted((int(v) for v in np.bincount(labelled.ravel())[1:]), reverse=True)
    detail["n_pieces"] = int(n_pieces)
    detail["pieces"] = sizes[:12]

    try:
        at_maps, n_rows, n_cols, _fp = _build_at_pct_maps_for_loaded_file()
    except Exception:
        at_maps = None
    if at_maps is None or n_rows != state.n_rows or n_cols != state.n_cols:
        return detail

    flat = mask2d.ravel()
    # Mean and enrichment come from the wand's own reporter, not a second
    # implementation. Enrichment needs the composition renormalised over the
    # scored elements before dividing by the background - doing that by hand
    # here produced "Al 49.68x" on a map whose background IS aluminium.
    stats = selection_stats(at_maps, mask2d, background=background_levels(at_maps))
    means = {el: v for el, v in stats["mean_at_pct"].items()
             if el not in _CHEM_IGNORE}
    els = []
    for el, mean in means.items():
        vals = np.asarray(at_maps[el], dtype=float).ravel()[flat]
        els.append({
            "element": el,
            "at_pct": round(float(mean), 2),
            # Spread inside the structure: one that is not homogeneous is
            # either two things or a gradient, and both are worth seeing.
            "spread": round(float(vals.std()), 2),
            "enrichment": stats["enrichment"].get(el),
        })
    els.sort(key=lambda d: -d["at_pct"])
    detail["elements"] = els

    # --- neighbours: who this structure touches, and how far away it is
    # chemically. The decision basis for merging.
    struct = _ndi.generate_binary_structure(2, 1)
    rim = _ndi.binary_dilation(mask2d, structure=struct) & ~mask2d
    touching = sorted({int(v) for v in grid[rim] if v >= 0 and v != sid})
    for other in touching:
        om = (grid == other).ravel()
        if not om.any():
            continue
        gap = 0.0
        for el, mean in means.items():
            ov = float(np.asarray(at_maps[el], dtype=float).ravel()[om].mean())
            gap = max(gap, abs(mean - ov))
        detail["neighbours"].append({
            "structure_id": other,
            "color": structure_color_hex(other),
            "shared_edge_px": int((grid[rim] == other).sum()),
            # Largest single-element difference, in at% - the same measure the
            # cluster count is chosen by, so "2 at% apart" means the same
            # thing here as it does there.
            "gap_at_pct": round(gap, 2),
        })
    detail["neighbours"].sort(key=lambda d: d["gap_at_pct"])

    # --- candidates, with the distance spelled out. ONE ranking definition,
    # here, so the inspector and any other view cannot disagree.
    for i, e in enumerate(state.phase_entries):
        comp = e.composition or {}
        keys = set(means) | set(comp)
        if not keys:
            continue
        gap = sum(abs(means.get(k, 0.0) - float(comp.get(k, 0.0))) for k in keys)
        detail["candidates"].append({
            "phase_index": i,
            "cif_filename": e.cif_filename,
            "formula": e.formula,
            "gap_at_pct": round(gap / len(keys), 2),
        })
    detail["candidates"].sort(key=lambda d: d["gap_at_pct"])
    return detail


@router.get("/phase-map/structure/{structure_id}")
async def get_structure_detail(structure_id: int):
    """Full description of one structure, for the inspector."""
    state = get_phase_map_store().get_state()
    if state is None or state.structure_grid is None:
        raise HTTPException(status_code=400, detail="No structures on this map")
    if not (0 <= structure_id < len(state.structure_phase)):
        raise HTTPException(
            status_code=400,
            detail=f"Structure {structure_id} out of range "
                   f"(0..{len(state.structure_phase) - 1})")
    return await asyncio.to_thread(_structure_detail, state, int(structure_id))


@router.post("/phase-map/structure-at")
async def structure_at(req: StructureAtRequest):
    """Which structure is under this pixel, described in full.

    One round trip for the whole click: the map has no structure ids on the
    client, and fetching the id and then its detail would be two.
    """
    state = get_phase_map_store().get_state()
    if state is None or state.structure_grid is None:
        raise HTTPException(status_code=400, detail="No structures on this map")
    if not (0 <= req.row < state.n_rows and 0 <= req.col < state.n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"pixel ({req.row},{req.col}) outside scan "
                   f"{state.n_rows}x{state.n_cols}")
    sid = int(state.structure_grid[req.row, req.col])
    if sid < 0:
        # No data here - say so rather than returning structure 0.
        return {"structure_id": None}
    return await asyncio.to_thread(_structure_detail, state, sid)
