"""
HDF5 Viewer API Routes

Wraps H5OINADataExtractor to provide:
- File open/close
- Pattern retrieval (single + batch)
- EDS element maps
- Electron images
- Grid navigation
"""

import asyncio
import logging
from typing import Literal, Optional

import h5py
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from backend.api.services.h5_session import open_file, get_extractor, close_file, is_open, get_current_path, get_cached_pattern, get_h5_file
from backend.api.services.image_utils import (
    array_to_base64_raw,
    array_to_png_bytes,
    colormap_array_to_base64,
    element_color_overlay_to_base64,
    phase_map_to_base64,  # NEW
)

logger = logging.getLogger(__name__)
router = APIRouter()

# X-ray emission line preference order for ambiguous EDS short symbols.
# Kα1 is the strongest line for most light-to-medium elements; Lα1 for heavier.
_PREFERRED_EDS_LINES = ("Kα1", "Kα", "Lα1", "Lα")


def _resolve_element(short_or_full: str, available: list[str], line: str | None = None) -> str:
    """Resolve 'Al' -> 'Al Kα1' deterministically.

    - If full name in available: return as-is
    - If short symbol + line given: match exactly (line is the suffix after the symbol)
    - If short only: prefer Kα1, then Kα, then Lα1, then Lα, then first lexicographic
    """
    if short_or_full in available:
        return short_or_full
    candidates = [n for n in available if n.split()[0] == short_or_full]
    if not candidates:
        raise ValueError(f"Element '{short_or_full}' not found. Available: {available}")
    if line:
        exact = [n for n in candidates if n.endswith(line)]
        if not exact:
            raise ValueError(
                f"Element '{short_or_full}' has no line '{line}'. Available: {candidates}"
            )
        return exact[0]
    for preferred in _PREFERRED_EDS_LINES:
        for n in candidates:
            if n.endswith(preferred):
                return n
    return sorted(candidates)[0]


class FileOpenRequest(BaseModel):
    path: str


class NavigateRequest(BaseModel):
    row: int
    col: int


# --- File Management ---

@router.post("/open")
async def h5_open(req: FileOpenRequest):
    """Open an HDF5 file and return metadata."""
    try:
        extractor, fmt = open_file(req.path)
        features = extractor.detect_available_features()
        # Ensure grid_shape and pattern_shape are JSON-serializable lists
        gs = features.get('grid_shape', (0, 0))
        ps = features.get('pattern_shape', (0, 0))
        features['grid_shape'] = list(gs) if hasattr(gs, '__iter__') else [gs, gs]
        features['pattern_shape'] = list(ps) if hasattr(ps, '__iter__') else [ps, ps]
        features['pattern_count'] = int(features.get('pattern_count', 0))
        # Convert element/image lists to plain strings for JSON
        features['eds_elements'] = [str(e) for e in features.get('eds_elements', [])]
        features['electron_images'] = [str(e) for e in features.get('electron_images', [])]
        return {
            "success": True,
            "format_type": fmt,
            "file_path": req.path,
            **features,
        }
    except Exception as e:
        logger.exception("Failed to open file: %s", req.path)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/close")
async def h5_close():
    """No-op close called by the HDF5 Viewer modal.

    The h5_session is shared with EDS and other routes, so closing the modal
    must NOT close the underlying file handle.  Local UI state is reset on the
    frontend side; the backend file stays open until a new file is loaded or
    /force-close is called explicitly.
    """
    return {"success": True}


@router.post("/force-close")
async def h5_force_close():
    """Actually close the HDF5 session (call when unloading from EBSD Viewer)."""
    close_file()
    return {"success": True}


@router.get("/status")
async def h5_status():
    """Check if a file is open and return current info."""
    if not is_open():
        return {"is_open": False}
    ext = get_extractor()
    try:
        grid_shape = list(ext.get_grid_dimensions())
        pattern_count = ext.get_pattern_count()
    except ValueError as e:
        # Corrupt header (commit a58d10f raises ValueError on n_cols<1).
        # Don't 500 — return is_open=True with the diagnostic so the
        # frontend can show "file open but grid invalid" instead of
        # crashing the status panel.
        return {
            "is_open": True,
            "file_path": get_current_path(),
            "grid_shape": [0, 0],
            "pattern_count": 0,
            "error": str(e),
        }
    return {
        "is_open": True,
        "file_path": get_current_path(),
        "grid_shape": grid_shape,
        "pattern_count": pattern_count,
    }


# --- Pattern Access ---

@router.get("/pattern/{index}")
async def get_pattern(index: int, pattern_type: str = "processed"):
    """Get a single pattern by flat index as Base64 PNG (cached)."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    pattern = get_cached_pattern(index, pattern_type)
    if pattern is None:
        raise HTTPException(status_code=404, detail=f"Pattern {index} not found")

    row, col = ext.index_to_position(index)
    return {
        "image": array_to_base64_raw(pattern),
        "row": row,
        "col": col,
        "index": index,
        "shape": list(pattern.shape),
    }


@router.get("/pattern/{index}/binary")
async def get_pattern_binary(index: int):
    """Return a single pattern as raw PNG bytes (no JSON wrapping).

    ~33% smaller than the base64 endpoint plus skips the frontend's JSON parse.
    Use for hot navigation paths (arrow-key sweeps, neighbor prefetch).
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    pat = get_cached_pattern(index)
    if pat is None:
        raise HTTPException(status_code=404, detail=f"Pattern {index} not found")
    return Response(content=array_to_png_bytes(pat), media_type="image/png")


@router.get("/pattern/pos/{row}/{col}")
async def get_pattern_by_pos(row: int, col: int, pattern_type: str = "processed"):
    """Get a single pattern by grid position (cached)."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()
    # Bounds-check BEFORE computing index — position_to_index doesn't validate,
    # so a negative row would wrap into a valid negative flat index and load
    # the wrong pattern silently. A 400 with the actual limits is clearer.
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Position ({row},{col}) out of bounds for {n_rows}x{n_cols} grid",
        )
    index = ext.position_to_index(row, col)
    pattern = get_cached_pattern(index, pattern_type)
    if pattern is None:
        raise HTTPException(status_code=404, detail=f"Pattern at ({row},{col}) not found")

    return {
        "image": array_to_base64_raw(pattern),
        "row": row,
        "col": col,
        "index": index,
        "shape": list(pattern.shape),
    }


@router.post("/navigate")
async def navigate(req: NavigateRequest):
    """Navigate to a position - returns pattern + available EDS + electron data."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()
    if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Position ({req.row},{req.col}) out of bounds for {n_rows}x{n_cols} grid",
        )
    index = ext.position_to_index(req.row, req.col)
    pattern = get_cached_pattern(index)

    result = {
        "row": req.row,
        "col": req.col,
        "index": index,
        "pattern": None,
        "eds_available": False,
        "electron_available": False,
    }

    if pattern is not None:
        result["pattern"] = array_to_base64_raw(pattern)

    elements = ext.get_available_elements()
    if elements:
        result["eds_available"] = True
        result["eds_elements"] = elements

    e_images = ext.get_available_electron_images()
    if e_images:
        result["electron_available"] = True
        result["electron_images"] = e_images

    return result


# --- Scoped reads: the file as it is, or the dataset the user is on ---

ReadScope = Literal["file", "dataset"]

# Kept as an alias: the electron endpoints introduced the parameter and the
# name still reads correctly at their call sites.
ElectronScope = ReadScope


def _scoped_extractor(scope: ReadScope):
    """The extractor a grid-shaped read should come from.

    ``scope="file"`` (the default) shows the data as the FILE holds it — that
    is what the H5 cockpit, this module's main tenant, wants. ``scope="dataset"``
    shows it as the ACTIVE DATASET sees it, so a cropped dataset gets the
    matching cut-out; that is what the EDS and Phase Map layer stacks want.

    The two cannot be reconciled into one choice: the same endpoint serves both
    kinds of consumer, so the caller has to say which view it means — and the
    parameter is a ``Literal`` so a typo is a 422 rather than a silent fall
    back to the file view.
    """
    from backend.api.services.h5_session import get_active_extractor
    return get_active_extractor() if scope == "dataset" else get_extractor()


# Historical name; the electron endpoints call it.
_electron_extractor = _scoped_extractor


# --- EDS ---

@router.get("/eds/elements")
async def get_eds_elements(scope: ReadScope = "file"):
    """Get list of available EDS elements. See ``_scoped_extractor``.

    The names themselves do not depend on the crop; the parameter exists so a
    caller can use one scope for the list and the maps it then fetches — and
    so a caller that means "the dataset" gets the crop-window guard rather
    than a full-scan answer that merely happens to be right today.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = _scoped_extractor(scope)
    return {"elements": ext.get_available_elements()}


@router.get("/eds/map/{element}")
async def get_eds_map(
    element: str,
    cmap: str = "hot",
    line: str | None = None,
    color: str = "",
    scope: ReadScope = "file",
):
    """Get an EDS element map as Base64 PNG.

    `element` can be a full name ("Al Kα1") or a short symbol ("Al").
    `line` optional override (e.g. "Kβ1") when the short symbol has multiple lines.
    `color` (hex like "#8be9fd") switches to a single-color RGBA overlay
    using the user's chosen element colour from the global colour
    store; when empty the legacy ``cmap`` heatmap is rendered instead.
    `scope` — see ``_scoped_extractor``. The default "file" keeps the H5
    cockpit's view; the Phase Map layer stack asks for "dataset" so its
    element map is cut to the same window as the phase map it sits under.
    Without that the two are composited on different grids and every feature
    on one of them is displaced.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = _scoped_extractor(scope)
    available = ext.get_available_elements()
    try:
        resolved = _resolve_element(element, available, line)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    data = ext.get_element_map_2d(resolved)
    if data is None:
        raise HTTPException(status_code=404, detail=f"EDS map for {resolved} not found")
    image_b64 = (
        element_color_overlay_to_base64(data, color)
        if color
        else colormap_array_to_base64(data, cmap)
    )
    return {
        "image": image_b64,
        "element": resolved,
        "min_val": float(np.nanmin(data)),
        "max_val": float(np.nanmax(data)),
        "shape": list(data.shape),
    }


@router.get("/eds/pixel/{row}/{col}")
async def get_eds_pixel(row: int, col: int):
    """Get EDS data for a specific pixel (counts for all elements)."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    elements = ext.get_available_elements()
    n_rows, n_cols = ext.get_grid_dimensions()
    # Bounds-check BEFORE computing flat index. A negative row would pass
    # the `index < len(data)` guard (since negative < positive is True) but
    # then numpy would silently return data from the wrong end of the array.
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Position ({row},{col}) out of bounds for {n_rows}x{n_cols} grid",
        )
    index = row * n_cols + col

    pixel_data = {}
    for el in elements:
        data = ext.get_element_map(el)
        if data is not None and index < len(data):
            pixel_data[el] = float(data[index])

    return {
        "row": row,
        "col": col,
        "counts": pixel_data,
        "elements": elements,
    }


# --- Electron Images ---

def _crop_status_of(ext, key: str) -> dict:
    """Whether ``key``'s last read could follow the crop, and why not.

    A raw extractor has no such notion — nothing was cut, nothing failed to be
    cut — so it answers "cropped" the same way an uncropped read does.
    """
    reader = getattr(ext, "last_crop_status", None)
    if reader is None:
        return {"cropped": True, "exact": True, "reason": None}
    return reader(key)


@router.get("/electron/list")
async def get_electron_list(scope: ElectronScope = "file"):
    """Get list of available electron images. See ``_electron_extractor``.

    The names themselves do not depend on the crop; the parameter exists so a
    caller can use one scope for the list and the images it then fetches.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = _electron_extractor(scope)
    return {"images": ext.get_available_electron_images()}


@router.get("/electron/{image_name:path}")
async def get_electron_image(image_name: str, scope: ElectronScope = "file"):
    """Get an electron image as Base64 PNG. See ``_electron_extractor``.

    ``crop`` is ``{"cropped", "exact", "reason"}``. Under ``scope="dataset"``
    a file that cannot place the two acquisition areas against each other
    yields ``cropped: false`` and the FULL image — never a guessed cut-out,
    which would misplace every feature on it. A cut-out clamped at the edge of
    the image keeps ``cropped: true`` but reports ``exact: false``: it is real
    data, but no longer the window's shape.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = _electron_extractor(scope)
    data = ext.get_electron_image(image_name)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Electron image '{image_name}' not found")

    return {
        "image": array_to_base64_raw(data),
        "name": image_name,
        "shape": list(data.shape),
        "crop": _crop_status_of(ext, image_name),
    }


# --- HDF5 Tree Structure ---

@router.get("/tree")
async def get_tree(max_depth: int = 5):
    """Get the HDF5 file tree structure for the tree browser."""
    try:
        h5_file = get_h5_file()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def _build_tree(group, depth=0):
        if depth >= max_depth:
            return {"type": "group", "children_count": len(group)}
        children = []
        for key in sorted(group.keys()):
            item = group[key]
            if isinstance(item, h5py.Group):
                children.append({
                    "name": key,
                    "type": "group",
                    "path": item.name,
                    "children": _build_tree(item, depth + 1) if depth < max_depth - 1 else [],
                    "children_count": len(item),
                })
            elif isinstance(item, h5py.Dataset):
                children.append({
                    "name": key,
                    "type": "dataset",
                    "path": item.name,
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                    "size": int(item.size),
                })
        return children

    tree = _build_tree(h5_file)
    return {"tree": tree, "filename": get_current_path()}


@router.get("/tree/node")
async def get_tree_node(path: str = "/"):
    """Get direct children of a specific HDF5 group path (lazy-load)."""
    try:
        h5_file = get_h5_file()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if path not in h5_file:
        raise HTTPException(status_code=404, detail=f"Path '{path}' not found")

    item = h5_file[path]
    if not isinstance(item, h5py.Group):
        raise HTTPException(status_code=400, detail=f"Path '{path}' is not a group")

    children = []
    for key in sorted(item.keys()):
        child = item[key]
        if isinstance(child, h5py.Group):
            children.append({
                "name": key,
                "type": "group",
                "path": child.name,
                "children": [],
                "children_count": len(child),
            })
        elif isinstance(child, h5py.Dataset):
            children.append({
                "name": key,
                "type": "dataset",
                "path": child.name,
                "shape": list(child.shape),
                "dtype": str(child.dtype),
                "size": int(child.size),
            })
    return {"children": children, "path": path}


@router.get("/attributes")
async def get_attributes(path: str = "/"):
    """Get HDF5 attributes for a given path."""
    try:
        h5_file = get_h5_file()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if path not in h5_file:
        raise HTTPException(status_code=404, detail=f"Path '{path}' not found")

    item = h5_file[path]
    attrs = {}
    for key in item.attrs:
        val = item.attrs[key]
        # Convert numpy types to Python types for JSON. Preserve the actual
        # type (int/float/bool/list) when possible so the frontend can
        # display "True" as a boolean, [1, 2, 3] as a list, etc., instead
        # of every value becoming a stringified copy. Long strings still
        # get truncated.
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='replace')
        elif hasattr(val, 'tolist'):  # numpy array → list (handles 0-d too)
            val = val.tolist()
        elif hasattr(val, 'item'):    # numpy scalar → python scalar
            val = val.item()
        # Truncate only string values to keep the response small
        if isinstance(val, str) and len(val) > 200:
            val = val[:200] + "..."
        attrs[key] = val

    info = {"path": path, "attributes": attrs}
    if isinstance(item, h5py.Dataset):
        info["shape"] = list(item.shape)
        info["dtype"] = str(item.dtype)
        info["size"] = int(item.size)
    elif isinstance(item, h5py.Group):
        info["children"] = list(item.keys())

    return info


# --- Minimap ---

@router.get("/minimap")
async def get_minimap():
    """Get a minimap overview.

    Priority:
    1. Band Contrast (single dataset read from /N/EBSD/Data/Band Contrast)
    2. First electron image
    3. Downsampled mean of sampled patterns (capped at 256 samples)
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()

    # Priority 1: Band Contrast — always present in H5OINA, single dataset read.
    bc = ext.get_band_contrast_map()
    if bc is not None:
        return {
            "image": array_to_base64_raw(bc),
            "source": "band_contrast",
            "shape": [n_rows, n_cols],
        }

    # Priority 2: First electron image
    electron_images = ext.get_available_electron_images()
    if electron_images:
        data = ext.get_electron_image(electron_images[0])
        if data is not None:
            return {
                "image": array_to_base64_raw(data),
                "source": electron_images[0],
                "shape": list(data.shape),
            }

    # Priority 3 (fallback): sampled mean intensity, capped at 256 samples
    # for responsiveness on files without BC and without electron images
    # (synthetic data, older files).
    total = n_rows * n_cols
    target_samples = min(total, 256)
    step = max(1, int(np.ceil((total / target_samples) ** 0.5)))
    minimap = np.zeros((n_rows, n_cols), dtype=np.float32)
    for r in range(0, n_rows, step):
        for c in range(0, n_cols, step):
            idx = r * n_cols + c
            pat = ext.get_pattern_at_index(idx)
            if pat is not None:
                val = float(np.mean(pat))
                minimap[r:min(r + step, n_rows), c:min(c + step, n_cols)] = val

    return {
        "image": array_to_base64_raw(minimap),
        "source": f"sampled_mean_{step}x{step}",
        "shape": [n_rows, n_cols],
    }


# --- Scans (numpy-side, fast) ---

# Defect classification thresholds. Calibrated for uint8 (0..255) patterns —
# the actual dtype produced by H5OINADataExtractor.get_pattern_at_index() for
# Oxford H5OINA files (verified on the SampleB test data: dtype=uint8,
# min=1, max=255). These mirror the previous frontend heuristic in
# classifyPattern() so behavior is preserved across the move.
_BEAM_OFF_MEAN = 5.0
_SATURATED_P99 = 250.0
_SATURATED_FRAC = 0.05
_LOW_SIGNAL_STD = 5.0
_HOT_PIXEL_K_STD = 10.0


def _classify_pattern(pat: np.ndarray) -> str:
    """Label a pattern as ok / beam_off / saturated / low_signal / hot_pixels.

    Mirrors the heuristic the frontend used to apply per-pattern. Order matters:
    beam_off → saturated → low_signal → hot_pixels → ok.

    Thresholds assume uint8 (0..255) input — see module-level constants.
    Raises ValueError on non-uint8 input rather than silently misclassifying.
    """
    if pat.dtype != np.uint8:
        raise ValueError(
            f"_classify_pattern thresholds are calibrated for uint8 only, "
            f"got dtype {pat.dtype}. Update thresholds before extending dtype support."
        )
    flat = pat.astype(np.float32).ravel()
    mean = float(flat.mean())
    if mean < _BEAM_OFF_MEAN:
        return "beam_off"
    # Both percentiles needed below — single batched call avoids a second sort.
    p99, p999 = np.percentile(flat, [99, 99.9])
    p99 = float(p99)
    p999 = float(p999)
    if p99 > _SATURATED_P99 and (flat > _SATURATED_P99).sum() / flat.size > _SATURATED_FRAC:
        return "saturated"
    std = float(flat.std())
    if std < _LOW_SIGNAL_STD:
        return "low_signal"
    if p999 > mean + _HOT_PIXEL_K_STD * std:
        return "hot_pixels"
    return "ok"


def _sample_indices(total: int, samples: int) -> list[int]:
    """Evenly-spaced sample indices, clamped so we always return at least 1 and at most `total`."""
    if total < 1:
        raise ValueError("Pattern count must be >= 1")
    samples = max(1, min(samples, total))
    step = max(1, total // samples)
    return list(range(0, total, step))[:samples]


def _scan_quality_sync(ext, indices: list[int]) -> list[dict]:
    """Synchronous part of /scan/quality — runs in a thread so concurrent
    requests (e.g. minimap + scan) don't stall every other endpoint."""
    scores = []
    for idx in indices:
        pat = ext.get_pattern_at_index(idx)
        if pat is None:
            continue
        scores.append({"index": int(idx), "score": float(pat.astype(np.float32).std())})
    return scores


def _scan_defects_sync(ext, indices: list[int]) -> tuple[dict, int]:
    """Synchronous part of /scan/defects — runs in a thread."""
    counts = {"ok": 0, "low_signal": 0, "saturated": 0, "hot_pixels": 0, "beam_off": 0}
    sampled = 0
    for idx in indices:
        pat = ext.get_pattern_at_index(idx)
        if pat is None:
            continue
        counts[_classify_pattern(pat)] += 1
        sampled += 1
    return counts, sampled


@router.get("/scan/quality")
async def scan_quality(samples: int = 50, bins: int = 20):
    """Quality scan: sample N patterns, score each by std-dev intensity, return best/worst + histogram."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()
    total = n_rows * n_cols
    indices = _sample_indices(total, samples)

    scores = await asyncio.to_thread(_scan_quality_sync, ext, indices)

    if not scores:
        # Fail loud — caller asked for samples but we couldn't load any. The empty 200
        # would silently hide a real problem (corrupt file, empty grid, etc.).
        raise HTTPException(status_code=500, detail="Could not sample any patterns")

    scores_sorted = sorted(scores, key=lambda s: s["score"])
    score_values = [s["score"] for s in scores]
    hist, edges = np.histogram(score_values, bins=bins)
    return {
        "best": scores_sorted[-1],
        "worst": scores_sorted[0],
        "sampled": len(scores),
        "histogram": {"counts": hist.tolist(), "edges": edges.tolist()},
    }


@router.get("/scan/defects")
async def scan_defects(samples: int = 60):
    """Defect scan: sample N patterns, classify each, return per-category counts."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()
    total = n_rows * n_cols
    indices = _sample_indices(total, samples)

    counts, sampled = await asyncio.to_thread(_scan_defects_sync, ext, indices)

    if sampled == 0:
        raise HTTPException(status_code=500, detail="Could not sample any patterns")

    return {"counts": counts, "sampled": sampled}


_SCALAR_MAP_ALLOWED_PREFIXES = ("/1/EBSD/Data/", "/1/EDS/Data/")


@router.get("/scalar-map")
async def get_scalar_map(path: str, cmap: str = "gray"):
    """Read any 1D per-pixel dataset, reshape to grid, return colormapped PNG."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    if not any(path.startswith(p) for p in _SCALAR_MAP_ALLOWED_PREFIXES):
        raise HTTPException(
            status_code=403,
            detail=f"Path '{path}' not in allowlist {_SCALAR_MAP_ALLOWED_PREFIXES}",
        )
    ext = get_extractor()
    try:
        arr = ext.get_scalar_map(path)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if arr is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{path}' not found")
    return {
        "image": colormap_array_to_base64(arr.astype(np.float32), cmap),
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "source": path,
    }


@router.get("/phase-map")
async def get_phase_map():
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    phase_ids = ext.get_phase_map()
    if phase_ids is None:
        raise HTTPException(status_code=404, detail="No phase map (/1/EBSD/Data/Phase) in file")
    phases = ext.get_phases_metadata()
    return {
        "image": phase_map_to_base64(phase_ids, phases),
        "shape": list(phase_ids.shape),
        "phases": phases,
    }


@router.get("/ipf-map")
async def get_ipf_map(direction: str = "Z"):
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    if direction not in ("X", "Y", "Z"):
        raise HTTPException(status_code=400, detail=f"direction must be X|Y|Z, got {direction!r}")
    ext = get_extractor()
    try:
        rgb = ext.compute_ipf_map(direction)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if rgb is None:
        raise HTTPException(status_code=404, detail="No Euler/Phase data — cannot compute IPF")
    from backend.api.services.image_utils import rgb_array_to_base64
    return {
        "image": rgb_array_to_base64(rgb),
        "shape": list(rgb.shape[:2]),
        "direction": direction,
    }


@router.get("/eds/spectrum/{row}/{col}")
async def get_eds_spectrum_pixel(row: int, col: int):
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise HTTPException(status_code=400, detail=f"Position ({row},{col}) out of bounds for {n_rows}x{n_cols} grid")
    index = row * n_cols + col
    try:
        spec = ext.get_eds_spectrum(index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if spec is None:
        raise HTTPException(status_code=404, detail="No EDS spectrum data in this file")
    return spec


@router.get("/eds/header")
async def get_eds_header():
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    h = get_extractor().get_eds_header()
    if h is None:
        raise HTTPException(status_code=404, detail="No EDS header in this file")
    return h


@router.get("/ebsd/header")
async def get_ebsd_header():
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    h = get_extractor().get_ebsd_header()
    if h is None:
        raise HTTPException(status_code=404, detail="No EBSD header in this file")
    return h


@router.get("/background")
async def get_background(type: str = "processed"):
    """Return the static background pattern (processed or unprocessed) as base64 PNG."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    if type not in ("processed", "unprocessed"):
        raise HTTPException(
            status_code=400,
            detail=f"type must be 'processed' or 'unprocessed', got {type!r}",
        )
    arr = get_extractor().get_static_background(type)
    if arr is None:
        raise HTTPException(status_code=404, detail=f"No {type} static background in this file")
    return {
        "image": array_to_base64_raw(arr),
        "shape": list(arr.shape),
        "type": type,
    }


@router.get("/aztec/pixel/{row}/{col}")
async def get_aztec_pixel_endpoint(row: int, col: int):
    """Return the consolidated Aztec per-pixel record (phase, Euler, MAD, BC,
    BS, bands, quality, error, PC, EDS timing) for the given grid position."""
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    ext = get_extractor()
    n_rows, n_cols = ext.get_grid_dimensions()
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Position ({row},{col}) out of bounds for {n_rows}x{n_cols} grid",
        )
    index = row * n_cols + col
    try:
        return ext.get_aztec_pixel(index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/layers")
async def get_layers():
    """Catalog of every map layer present in the open file, grouped by category.

    Frontend uses this to render the cockpit's layer selector without having
    to probe each endpoint individually.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")
    return get_extractor().enumerate_layers()
