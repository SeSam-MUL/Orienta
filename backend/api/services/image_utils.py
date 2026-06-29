"""Utility functions for converting numpy arrays to Base64 images."""

import io
import base64
import numpy as np


def array_to_base64_png(arr: np.ndarray, cmap: str = "gray", normalize: bool = True) -> str:
    """
    Convert a 2D numpy array to a Base64-encoded PNG string.

    Uses matplotlib for rendering to support colormaps.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Materialise dask-backed arrays (lazy-loaded EBSD signals produce them
    # from signal.data[row, col]); PIL/matplotlib only accept numpy. Cheap
    # no-op on plain numpy arrays.
    arr = np.asarray(arr)

    h, w = arr.shape[:2]
    aspect = w / max(h, 1)
    fig_w = min(max(3, 5 * aspect), 10)
    fig_h = fig_w / max(aspect, 0.1)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=150)

    if normalize and arr.dtype != np.uint8:
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    else:
        vmin, vmax = None, None

    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to uint8 [0, 255] via min-max scaling.

    Constant arrays (max == min) become all-zeros — same behavior as the
    historical inline scaling in `array_to_base64_raw`.
    Pre-normalized uint8 input is returned unchanged.
    """
    if arr.dtype == np.uint8:
        return arr
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max > arr_min:
        return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def array_to_base64_raw(arr: np.ndarray) -> str:
    """
    Convert a 2D grayscale array to Base64 PNG without matplotlib (faster).

    Uses PIL/Pillow for minimal overhead.
    """
    from PIL import Image

    # Materialise dask-backed arrays (lazy-loaded EBSD signals produce them
    # from signal.data[row, col]); PIL.Image.fromarray needs the numpy
    # __array_interface__ which dask.array.Array does not implement.
    arr = np.asarray(arr)
    arr = _normalize_to_u8(arr)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def apply_display_filter(pattern: np.ndarray, filter_name: str) -> np.ndarray:
    """Apply a display-time filter to a pattern array.

    These are visualization filters — they don't modify the stored data.
    """
    if not filter_name or filter_name == "None":
        return pattern

    if filter_name == "Dynamic BG":
        from kikuchipy.pattern import remove_dynamic_background
        return remove_dynamic_background(
            pattern.astype("float32"),
            operation="subtract",
            filter_domain="frequency",
        )

    from scipy import ndimage

    p = pattern.astype(np.float64)

    if filter_name == "Sobel":
        sx = ndimage.sobel(p, axis=0)
        sy = ndimage.sobel(p, axis=1)
        return np.hypot(sx, sy)

    if filter_name == "Canny":
        try:
            from skimage.feature import canny as sk_canny
            return sk_canny(p, sigma=1.0).astype(np.float64) * 255
        except ImportError:
            sx = ndimage.sobel(p, axis=0)
            sy = ndimage.sobel(p, axis=1)
            return np.hypot(sx, sy)

    if filter_name == "Difference":
        blurred = ndimage.gaussian_filter(p, sigma=3)
        return p - blurred

    if filter_name == "FFT Highpass":
        from numpy.fft import fft2, ifft2, fftshift, ifftshift
        F = fftshift(fft2(p))
        rows, cols = p.shape
        crow, ccol = rows // 2, cols // 2
        r = max(min(rows, cols) // 12, 5)
        Y, X = np.ogrid[:rows, :cols]
        mask = ((X - ccol) ** 2 + (Y - crow) ** 2 >= r ** 2).astype(np.float64)
        return np.abs(ifft2(ifftshift(F * mask)))

    if filter_name == "CLAHE":
        try:
            from skimage.exposure import equalize_adapthist
            p_norm = p / max(p.max(), 1e-8)
            return equalize_adapthist(p_norm, clip_limit=0.03) * 255
        except ImportError:
            return p

    if filter_name == "Sharpen":
        blurred = ndimage.gaussian_filter(p, sigma=1)
        return np.clip(p + (p - blurred) * 2, 0, None)

    if filter_name == "Denoise":
        return ndimage.median_filter(p, size=3).astype(np.float64)

    return pattern


def colormap_array_to_base64(arr: np.ndarray, cmap_name: str = "viridis") -> str:
    """Convert 2D array to a colormapped Base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image

    # Normalize
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max > arr_min:
        normalized = (arr - arr_min) / (arr_max - arr_min)
    else:
        normalized = np.zeros_like(arr, dtype=np.float64)

    try:
        colormap = matplotlib.colormaps[cmap_name]
    except (AttributeError, KeyError):
        colormap = cm.get_cmap(cmap_name)
    colored = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(colored)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def element_color_overlay_to_base64(arr: np.ndarray, hex_color: str) -> str:
    """Convert 2D array to an RGBA PNG where intensity maps to alpha of the given color.

    Low values → transparent, high values → solid element color.
    This produces clean single-color overlays instead of heatmaps.
    """
    from PIL import Image

    # Normalize to 0..1
    arr_min, arr_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    if arr_max > arr_min:
        normalized = (arr.astype(np.float64) - arr_min) / (arr_max - arr_min)
    else:
        normalized = np.zeros_like(arr, dtype=np.float64)

    # Parse hex color
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Build RGBA image: constant color, intensity → alpha
    h, w = arr.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = r
    rgba[:, :, 1] = g
    rgba[:, :, 2] = b
    rgba[:, :, 3] = (normalized * 255).astype(np.uint8)

    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def phase_map_to_base64(phase_ids: np.ndarray, phases: list[dict]) -> str:
    """Render phase-ID array as RGB PNG using each phase's stored color.

    phase_ids: (H, W) integer array; 0 = unindexed/black.
    phases: list of {id, color: [r,g,b]} dicts.
    """
    h, w = phase_ids.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    color_lut = {int(p["id"]): np.array(p["color"], dtype=np.uint8) for p in phases}
    for pid, color in color_lut.items():
        rgb[phase_ids == pid] = color
    from PIL import Image
    import io, base64
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def rgb_array_to_base64(arr: np.ndarray) -> str:
    """Encode (H, W, 3) uint8 array as base64 PNG."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) array, got {arr.shape}")
    from PIL import Image
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a 2D numpy array as raw PNG bytes (no base64).

    Mirrors `array_to_base64_raw` (same min-max normalization via
    `_normalize_to_u8`) but skips the b64encode step. Use for
    `Response(content=..., media_type="image/png")` routes — saves ~33%
    bandwidth plus the JSON parse the frontend would otherwise pay.
    """
    from PIL import Image

    if arr.ndim == 2:
        arr_u8 = _normalize_to_u8(arr)
        img = Image.fromarray(arr_u8)
    else:
        img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
