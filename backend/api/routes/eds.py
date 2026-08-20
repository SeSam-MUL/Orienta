"""
EDS (Energy Dispersive Spectroscopy) API Routes

Wraps eds_utils for:
- EDS data extraction from H5OINA
- Counts → Wt.% → At.% conversion
- Phase suggestion based on chemistry
- Pixel-level and region-level analysis
"""

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
    load_cif_phase_library,
    suggest_phases_from_cif_library,
)
from backend.api.services.phase_map_store import (
    get_phase_map_store,
    palette_hex_for_state,
    render_phase_map_to_base64,
)

logger = logging.getLogger(__name__)
router = APIRouter()


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
            cif_hits = suggest_phases_from_cif_library(at_scalar, cif_library)
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
    """Tunables for auto-classify; defaults match the suggest-phase scoring."""
    tolerance: float = 15.0   # per-element At.% deviation that scores 0
    min_score: float = 0.3    # below this -> unclassified


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
    palette = palette_hex_for_state(state)
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
    if include_image:
        response["image"] = render_phase_map_to_base64(state)
    return response


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

    at_maps, n_rows, n_cols, file_path = _build_at_pct_maps_for_loaded_file()

    phase_grid, score_grid, candidates = auto_classify_pixels(
        at_pct_per_element=at_maps,
        n_rows=n_rows,
        n_cols=n_cols,
        cif_library=cif_library,
        tolerance=req.tolerance,
        min_score=req.min_score,
    )
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=(
                "No CIF phase has elements that are a subset of the measured EDS elements — "
                "auto-classify cannot match any phase. Check that the CIF database covers "
                "the elements present in this scan."
            ),
        )

    store = get_phase_map_store()
    store.set_classification(
        phase_grid=phase_grid,
        score_grid=score_grid,
        phase_entries=candidates,
        tolerance=req.tolerance,
        min_score=req.min_score,
        file_path=file_path,
    )
    return _state_to_response(include_image=True)


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
