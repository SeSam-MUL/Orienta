"""
Phase Map API Routes

Wraps phase_map_generator for:
- Phase map rendering with configurable scalebar
- IPF (Inverse Pole Figure) color computation
- Phase map export (image files — PNG/SVG/PDF — matching the live view)
- Post-processing cleanup (CI/uncertainty/cluster filters + modal + fill)

Design note:
    /render and /export share `_render_to_figure(...)`. /render encodes it as
    base64 PNG for the browser; /export saves the figure to disk at the
    requested format and DPI. Before, /export hand-rolled its own IPF-only
    render — switching display mode in the UI had no effect on the exported
    image. Now every knob on the panel (direction, cleanup, scalebar, legend,
    confidence overlay, title, DPI, etc.) maps 1:1 between preview and export.
"""

import asyncio
import io
import base64
import logging
import threading
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# matplotlib's pyplot keeps a global figure registry that is NOT thread-safe.
# The render helpers (_render_to_figure etc.) use plt.figure/plt.subplots, so
# we serialise ALL matplotlib work under this lock and run it via
# asyncio.to_thread: the event loop stays free (other requests proceed) while
# concurrent render requests take turns instead of corrupting pyplot state.
_mpl_lock = threading.Lock()

from backend.api.services.image_utils import array_to_base64_png  # noqa: F401  (re-export kept for callers)
from backend.api.services import state_version

# Re-exported at module level so tests can patch them via
# `backend.api.routes.phase_map.get_last_indexing_result`. Deferred imports
# inside individual functions keep working for other callers; this module-
# level binding is consulted by _compute_layer_rgba and friends.
from backend.api.routes.indexing import get_last_indexing_result  # noqa: F401
from backend.api.routes.analysis import get_analysis_dataset  # noqa: F401

logger = logging.getLogger(__name__)
router = APIRouter()


def _active_r_user():
    """Resolve R_user for the active file, or None on any failure (fail-soft for render).

    IPF colours are recomputed per request (this module has NO backend
    layer/IPF response cache keyed by result_id/kind), so the render always
    reflects the current frame. The frontend ImageBitmap cache (a later task)
    is therefore the invalidation point — it must include the frame signature
    (reference_frame_state.frame_signature) in its key so a frame change forces
    a re-fetch.
    """
    try:
        from backend.api.services import reference_frame_state as rfs
        spec = rfs.get_frame(rfs.get_active_source_file())
        return rfs.resolve_r_user(spec)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Dracula palette — matches frontend DRACULA_PHASE_COLORS + phase_map_gui.py
# ---------------------------------------------------------------------------
PHASE_COLORS = [
    [0.373, 0.898, 0.482],   # #5ff77a — green
    [0.510, 0.667, 1.000],   # #82aaff — blue
    [0.969, 0.549, 0.424],   # #f78c6c — orange
    [0.784, 0.576, 0.918],   # #c892ea — purple
    [1.000, 0.796, 0.420],   # #ffcb6b — yellow
    [0.537, 0.867, 0.976],   # #89ddff — cyan
    [1.000, 0.325, 0.439],   # #ff5370 — red
    [0.765, 0.910, 0.553],   # #c3e88d — lime
]
UNINDEXED_COLOR = [0.267, 0.278, 0.353]  # Dracula #44475a — neutral gray

# Phase names that orix / various loaders emit for "not indexed" sentinel
# entries. These should be rendered in UNINDEXED_COLOR, not as a real phase.
_UNINDEXED_NAMES = {
    "", "not_indexed", "not indexed", "notindexed",
    "unindexed", "nothing", "none", "null",
}


def _name_hash(name: str) -> int:
    """JavaScript-compatible string hash ``((h << 5) - h + c) | 0``.

    Shared basis for both ``stable_color_idx`` (8-slot palette) and the
    name-stable hue (``_hsv_phase_color_by_name``) so the same phase name always
    maps to the same colour, and a JS consumer using the identical hash agrees.
    """
    h = 0
    for c in (name or ""):
        h = ((h << 5) - h + ord(c)) & 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
    return abs(h)


def stable_color_idx(name: str) -> int:
    """Deterministic color slot for a phase name.

    Uses the JS-compatible ``_name_hash`` so the identical function in the
    React legend lands on the same palette slot — Al stays the same colour,
    regardless of which file supplies it and regardless of that file's internal
    phase_id numbering.
    """
    return _name_hash(name) % len(PHASE_COLORS)


def _is_unindexed_phase(name) -> bool:
    if name is None:
        return True
    return str(name).strip().lower() in _UNINDEXED_NAMES


def _group_phases_by_symmetry(xmap, only_phase_id: int | None = None):
    """Group real phases by point-group symmetry.

    Returns an ordered ``{point_group_name: {"symmetry": obj, "phases": [names...]}}``
    mapping. Two phases with the same point group share one IPF key — no need
    to repeat a cubic m-3m triangle three times just because the sample has
    three different cubic phases. Phases without a point_group are skipped so
    a triclinic fallback doesn't sneak an IPF key in for data we can't
    meaningfully colour anyway.

    ``only_phase_id`` restricts the grouping to a single phase (used by the
    per-phase IPF view so the colour key matches the filtered map).
    """
    groups: dict = {}
    try:
        iterator = list(xmap.phases)
    except Exception:
        return groups
    for entry in iterator:
        if isinstance(entry, tuple) and len(entry) == 2:
            pid, phase_obj = entry
            try:
                if int(pid) < 0:
                    continue
                if only_phase_id is not None and int(pid) != int(only_phase_id):
                    continue
            except Exception:
                pass
        else:
            phase_obj = entry
            try:
                pid = int(getattr(entry, "id", 0))
                if pid < 0:
                    continue
                if only_phase_id is not None and pid != int(only_phase_id):
                    continue
            except Exception:
                pass
        name = getattr(phase_obj, "name", "") or ""
        if _is_unindexed_phase(name):
            continue
        pg = getattr(phase_obj, "point_group", None)
        if pg is None:
            continue
        # Collapse to the Laue class (adds inversion centre) so the IPF key
        # is the compact TSL/MTEX-style triangle users expect. Without this
        # a cubic 432 proper group renders a "half-full" wedge instead of
        # the familiar [001]-[101]-[111] triangle, because the fundamental
        # zone is twice as large when inversion is not implied.
        laue = getattr(pg, "laue", pg) or pg
        laue_name = getattr(laue, "name", None) or str(laue)
        bucket = groups.setdefault(laue_name, {"symmetry": laue, "phases": []})
        if name not in bucket["phases"]:
            bucket["phases"].append(name)
    return groups


def _draw_ipf_color_keys(fig, gs_cell, xmap, direction: str,
                         color_overrides: dict | None = None,
                         orientation: str = "vertical",
                         only_phase_id: int | None = None):
    """Render one IPF stereographic triangle per unique Laue class into the
    given gridspec cell. ``direction`` is 'X', 'Y' or 'Z' — the reference
    vector used when reducing orientations for colour. Skips silently if
    orix can't build the projection for a phase.

    Layout per cell:
        colored bar(s)      ← phase colour(s) from the Phase-Map legend
        **Phase name(s)**   ← big, bold, so users know which triangle is which
        (Laue class)        ← provides the orix symmetry label
        [IPF triangle]

    When two phases share a Laue class (e.g. two cubic m-3m phases) the bar
    is split into coloured segments so the legend square from each phase is
    still recognisable.

    ``orientation`` picks how groups are stacked within the cell:
      * ``"vertical"`` — one row per group (used for right-column placement)
      * ``"horizontal"`` — one column per group (used for row-below placement
        under wide landscape maps)
    """
    from orix.plot import IPFColorKeyTSL  # noqa: F401 — registers projection
    from orix.vector import Vector3d

    dir_map = {
        "X": Vector3d.xvector(),
        "Y": Vector3d.yvector(),
        "Z": Vector3d.zvector(),
    }
    ref_vec = dir_map.get(direction.upper(), Vector3d.zvector())

    groups = _group_phases_by_symmetry(xmap, only_phase_id=only_phase_id)
    if not groups:
        return

    n = len(groups)
    # Big hspace / wspace keeps one group's triangle vertex labels ([111],
    # [010], …) from colliding with the next group's header.
    if orientation == "horizontal":
        sub_gs = gs_cell.subgridspec(1, n, wspace=0.4)
    else:
        sub_gs = gs_cell.subgridspec(n, 1, hspace=1.0)
    for i, (pg_name, info) in enumerate(groups.items()):
        try:
            cell = sub_gs[0, i] if orientation == "horizontal" else sub_gs[i, 0]
            # Inner layout: colour bar (tiny) + triangle. The phase name sits
            # as the axis title with enough pad to clear orix's [hkl] corner
            # labels at the top of the triangle.
            inner = cell.subgridspec(2, 1, height_ratios=[1, 16], hspace=0.12)

            # Colour bar — one segment per phase, matches the Phase-Map legend
            # square. Crucial for the user: "which triangle belongs to which
            # phase" is answered by matching the bar colour to the legend.
            ax_bar = fig.add_subplot(inner[0, 0])
            phases = info["phases"]
            for j, phase_name in enumerate(phases):
                rgb = _resolve_phase_color(phase_name, color_overrides)
                ax_bar.axvspan(j / len(phases), (j + 1) / len(phases),
                               facecolor=rgb, edgecolor='none')
            ax_bar.set_xlim(0, 1)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])
            for sp in ax_bar.spines.values():
                sp.set_edgecolor('#333333')
                sp.set_linewidth(0.5)

            # Triangle axis
            ax_key = fig.add_subplot(
                inner[1, 0],
                projection="ipf",
                symmetry=info["symmetry"],
                direction=ref_vec,
            )
            ax_key.plot_ipf_color_key(show_title=False)

            # Single-line label: "Al (m-3m)" or "Al, MgCu2 (m-3m)".
            # pad=14 clears the [111]/[010] vertex labels orix draws at
            # the triangle's top corners. Colour is dark so it stays
            # readable against the white axis — the previous white-on-white
            # made the titles invisible.
            phases_str = ", ".join(phases)
            ax_key.set_title(
                f"{phases_str}  ({pg_name})",
                fontsize=9,
                fontweight="bold",
                color="#1a1a2e",
                pad=26,
            )
        except Exception as e:
            logger.debug("IPF key render failed for %s: %s", pg_name, e)


def _compute_effective_phase_ids_2d(
    xmap, n_rows: int, n_cols: int, mask,
    last_result,
    *,
    ci_threshold: float,
    uncertainty_threshold: float,
    min_cluster_size: int,
    fill_unindexed: bool,
    modal_filter_size: int,
):
    """Compute cleanup state for every display mode.

    Returns ``(effective_pid_2d, ipf_valid_2d, had_cleanup)``.

    * ``effective_pid_2d`` — (n_rows, n_cols) int16 with -1 wherever a
      pixel has been rejected by the live filters or was already
      unindexed. This drives Phase-Map colouring.
    * ``ipf_valid_2d`` — (n_rows, n_cols) bool: True means "trust the
      pixel's original rotation". False means either the pixel is
      rejected OR its phase was relabelled (filled / modal-voted) and
      therefore its rotation is no longer a trustworthy match for the
      new phase — we refuse to synthesise orientations, so those pixels
      render as unindexed grey in IPF / CI views too.
    * ``had_cleanup`` — True if a persisted override or any live filter
      is active, so callers know whether masking is needed at all.

    When the xmap shape can't be derived, returns ``(None, None, False)``
    so callers can skip the cleanup mask entirely.
    """
    import numpy as np

    # 1) Start from the raw xmap.phase_id so we can track what the
    #    user's original measurement said at each pixel. This original
    #    grid is the ground truth for "can we trust the rotation here?".
    try:
        n_indexed = xmap.size
        raw_ids = np.asarray(xmap.phase_id)
    except Exception:
        return None, None, False

    if raw_ids.size == n_rows * n_cols:
        original_2d = raw_ids.reshape(n_rows, n_cols).astype(np.int16).copy()
    else:
        original_2d = np.full((n_rows, n_cols), -1, dtype=np.int16)
        if mask is not None and not mask.all():
            original_2d[mask] = raw_ids[:n_indexed]
        else:
            original_2d.flat[:raw_ids.size] = raw_ids

    # 2) Start effective with original, layer persisted and live changes on top.
    applied_2d = None
    if last_result is not None:
        applied_2d = last_result.metadata.get("cleaned_phase_id")
    if applied_2d is not None and np.asarray(applied_2d).shape == (n_rows, n_cols):
        effective_2d = np.asarray(applied_2d).astype(np.int16).copy()
    else:
        effective_2d = original_2d.copy()

    any_live = (ci_threshold > 0 or min_cluster_size > 0
                or fill_unindexed or modal_filter_size)

    if any_live:
        try:
            ci_prop = np.array(xmap.prop.get("ci", np.zeros(xmap.size, dtype=np.float32)))
            if ci_prop.size == n_rows * n_cols:
                ci_2d = ci_prop.reshape(n_rows, n_cols).astype(np.float32)
            else:
                ci_2d = np.zeros((n_rows, n_cols), dtype=np.float32)
                if mask is not None and not mask.all():
                    ci_2d[mask] = ci_prop[:int(mask.sum())]
                else:
                    ci_2d.flat[:ci_prop.size] = ci_prop
            dummy_unc = np.zeros_like(ci_2d)
            from backend.api.services.checkpoint_writer import CheckpointWriter as _CW
            effective_2d = _CW.clean_phase_assignment(
                effective_2d, ci_2d, dummy_unc,
                ci_threshold=ci_threshold,
                uncertainty_threshold=0.0,
                min_cluster_size=min_cluster_size,
                fill_unindexed=fill_unindexed,
                modal_filter_size=modal_filter_size,
            )
        except Exception as e:
            logger.warning("Live cleanup failed, falling back to raw: %s", e)

    had_cleanup = (applied_2d is not None) or any_live

    # 3) ipf_valid = pixel has a valid rotation _and_ we didn't relabel its
    #    phase. "Original == effective AND effective != -1" covers both:
    #    - rejected pixels (effective == -1) → invalid
    #    - filled pixels   (original -1, effective 1) → invalid (no real rot)
    #    - modal-voted     (original A, effective B)  → invalid (rot mismatch)
    ipf_valid_2d = (original_2d == effective_2d) & (effective_2d != -1)

    return effective_2d, ipf_valid_2d, had_cleanup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_step_size_from_signal(fallback: float) -> float:
    """Pull the real microns-per-pixel step size from the active EBSD signal.

    Falls back to the user-supplied value (defaults to 1.0 µm/px) when no
    signal is loaded or metadata doesn't expose it.
    """
    try:
        from backend.api.routes.ebsd_viewer import _extract_step_size, _get_active_signal
        sig = _get_active_signal()
        if sig is not None:
            extracted = _extract_step_size(sig)
            if isinstance(extracted, dict):
                sx = extracted.get("x", 0)
                if sx and sx > 0:
                    return float(sx)
            elif extracted and extracted > 0:
                return float(extracted)
    except Exception:
        pass
    return float(fallback)


def _apply_scalebar(ax, step_size, *, enabled, location, length, font_size,
                    bar_color, box_color, box_alpha):
    """Attach a matplotlib-scalebar artist. Silently no-ops when disabled or
    the library isn't installed."""
    if not enabled or step_size <= 0:
        return
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        sb = ScaleBar(
            step_size, 'um',
            location=location,
            length_fraction=length,
            font_properties={'size': font_size},
            color=bar_color,
            box_color=box_color,
            box_alpha=box_alpha,
        )
        ax.add_artist(sb)
    except Exception as e:
        logger.debug("scalebar skipped: %s", e)


# ---------------------------------------------------------------------------
# Core render — produces a matplotlib Figure. Caller decides output format.
# ---------------------------------------------------------------------------
def _hex_to_rgb01(hex_str):
    """Parse '#rrggbb' → (r, g, b) with each channel in 0..1.
    Returns None on malformed input so callers can fall back to the hash palette.
    """
    try:
        s = str(hex_str).lstrip('#')
        if len(s) != 6:
            return None
        return (int(s[0:2], 16) / 255.0,
                int(s[2:4], 16) / 255.0,
                int(s[4:6], 16) / 255.0)
    except (ValueError, TypeError):
        return None


def _resolve_phase_color(phase_name: str, color_overrides: dict | None):
    """Override wins over deterministic hash. Returns a (r, g, b) tuple in 0..1.
    ``phase_name`` matching is case-sensitive so "Al" and "AL" stay distinct —
    the frontend always sends the exact name the legend shows.

    Legacy fallback: used only when no phase-id context is available. Prefer
    :func:`build_phase_color_map` which is collision-free for any phase count.
    """
    if color_overrides and phase_name in color_overrides:
        rgb = _hex_to_rgb01(color_overrides[phase_name])
        if rgb is not None:
            return rgb
    return PHASE_COLORS[stable_color_idx(phase_name or "")]


def _hsv_phase_color(index: int, n_phases: int):
    """Maximally-distinct RGB for phase ``index`` of ``n_phases`` total.

    Spaces hues evenly around the colour wheel — ``hue = index / n_phases`` at
    fixed saturation/value — so no two phases ever share a colour regardless of
    N. The classic 8-slot Dracula palette wrapped around with ~19 phases and
    distinct phases collided; HSV spacing eliminates that by construction.

    Returns a (r, g, b) tuple in 0..1.
    """
    import colorsys
    if n_phases <= 0:
        n_phases = 1
    hue = (index % n_phases) / float(n_phases)
    return colorsys.hsv_to_rgb(hue, 0.65, 0.95)


def _hsv_phase_color_by_name(name: str):
    """NAME-STABLE, maximally-distinct RGB (same HSV family as
    :func:`_hsv_phase_color`, but hue derived from the phase NAME instead of its
    position).

    A phase keeps the same colour in every result regardless of which other
    phases are present or that result's internal phase-id numbering — fixing
    "the colours change when I switch results". Continuous hue (3600 steps) so
    distinct names stay distinct (no 8-slot palette wrap-around collisions).
    Phases sharing a name share a colour (they are the same phase).

    Returns a (r, g, b) tuple in 0..1.
    """
    import colorsys
    hue = (_name_hash(name) % 3600) / 3600.0
    return colorsys.hsv_to_rgb(hue, 0.65, 0.95)


def _real_phase_ids(xmap) -> list:
    """Sorted list of real (indexed, non-sentinel) phase ids in ``xmap``.

    Unindexed sentinels (id < 0, or phases named "not_indexed" etc.) are
    excluded — they always render in :data:`UNINDEXED_COLOR`, never a hue.
    Sorting makes the colour assignment deterministic: a given result always
    colours its lowest-id phase the same, its next phase the same, and so on.
    """
    import numpy as np
    ids = set()
    try:
        for pid in np.unique(np.asarray(xmap.phase_id)):
            pid = int(pid)
            if pid < 0:
                continue
            try:
                name = xmap.phases[pid].name
            except (KeyError, IndexError, TypeError, AttributeError):
                name = None
            if _is_unindexed_phase(name):
                continue
            ids.add(pid)
    except Exception:
        pass
    # Also pick up phases declared on the PhaseList that have no pixels yet —
    # keeps the legend / colour assignment stable even for empty phases.
    try:
        for entry in list(xmap.phases):
            if isinstance(entry, tuple) and len(entry) == 2:
                pid, phase_obj = entry
            else:
                pid, phase_obj = getattr(entry, "id", None), entry
            if pid is None:
                continue
            pid = int(pid)
            if pid < 0:
                continue
            if _is_unindexed_phase(getattr(phase_obj, "name", None)):
                continue
            ids.add(pid)
    except Exception:
        pass
    return sorted(ids)


def build_phase_color_map(xmap, color_overrides: dict | None = None) -> dict:
    """Return ``{phase_id: (r, g, b)}`` — a stable, distinct colour per phase.

    Colours are assigned from the phase NAME (continuous HSV hue), so:
      * A phase keeps its colour across DIFFERENT results / phase-id orderings
        — "Al" is the same colour everywhere (fixes "the colours change when I
        switch results"). This is the single source of truth for both the
        rendered map and the /phase-stats legend, so they always agree.
      * Distinct names get distinct hues (continuous 3600-step hue, not the
        8-slot palette, so no wrap-around collisions even with 14+ phases).
      * A ``color_overrides`` entry (keyed by phase NAME, '#rrggbb') still
        wins, preserving the user's manual colour picks.

    The FIRST phase with a given name keeps the clean name-stable hue; any
    further phases sharing that name (degenerate same-named candidates that
    spherical indexing keeps apart) get a distinct, still-deterministic hue so
    they never collide. Unnamed phases fall back to a per-id hue.
    """
    pids = _real_phase_ids(xmap)
    out: dict = {}
    name_occurrence: dict = {}
    for pid in pids:
        try:
            name = xmap.phases[pid].name
        except (KeyError, IndexError, TypeError, AttributeError):
            name = None
        rgb = None
        if color_overrides and name in color_overrides:
            rgb = _hex_to_rgb01(color_overrides[name])
        if rgb is None:
            key = name or f"__pid_{pid}"
            occ = name_occurrence.get(key, 0)
            name_occurrence[key] = occ + 1
            rgb = _hsv_phase_color_by_name(key if occ == 0 else f"{key}#{occ}")
        out[pid] = tuple(float(c) for c in rgb)
    return out


def _render_to_figure(
    *,
    direction: str,
    show_legend: bool,
    title: str,
    step_x: float,
    step_y: float,
    scalebar_enabled: bool,
    scalebar_location: str,
    scalebar_length: float,
    scalebar_font_size: int,
    scalebar_bar_color: str,
    scalebar_box_color: str,
    scalebar_box_alpha: float,
    confidence_overlay: bool,
    confidence_alpha: float,
    confidence_cmap: str,
    ci_threshold: float,
    uncertainty_threshold: float,
    min_cluster_size: int,
    fill_unindexed: bool,
    modal_filter_size: int,
    color_overrides: dict | None = None,
    show_ipf_keys: bool = True,
    bc_modulate: bool = False,
    dpi: int = 150,
):
    """Render the active indexing result / analysis dataset to a matplotlib
    Figure. Returns (fig, metadata_dict).

    Raises HTTPException on missing data or bad phase name.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from backend.api.routes.indexing import get_last_indexing_result
    from backend.api.routes.analysis import get_analysis_dataset

    _last_result = get_last_indexing_result()
    _analysis_dataset = get_analysis_dataset()

    xmap = None
    n_rows = n_cols = 0
    mask = None

    if _last_result is not None:
        xmap = _last_result.xmap
        n_rows, n_cols = _last_result.original_shape
        mask = _last_result.selection_mask
    elif _analysis_dataset is not None and hasattr(_analysis_dataset, 'xmap'):
        xmap = _analysis_dataset.xmap
        n_rows, n_cols = _analysis_dataset.shape
        mask = None
    else:
        raise HTTPException(
            status_code=400,
            detail="No indexing result or analysis dataset available. Load data first."
        )

    step_size = _get_step_size_from_signal(step_x)

    # Compute the effective cleanup state ONCE — every display mode honours
    # it, so rejected pixels drop out of Phase, IPF *and* CI views the same
    # way. Avoids the footgun where filtering only changed the Phase Map
    # while IPF kept showing the raw (wrong) pixel colours.
    effective_pid_2d, ipf_valid_2d, had_cleanup = _compute_effective_phase_ids_2d(
        xmap, n_rows, n_cols, mask, _last_result,
        ci_threshold=ci_threshold,
        uncertainty_threshold=uncertainty_threshold,
        min_cluster_size=min_cluster_size,
        fill_unindexed=fill_unindexed,
        modal_filter_size=modal_filter_size,
    )

    # ------------------------------------------------------------------
    # CI / uncertainty / per-phase CI maps — different render path
    # ------------------------------------------------------------------
    if direction.startswith("ci") or direction == "uncertainty":
        return _render_ci_figure(
            direction, _last_result, _analysis_dataset,
            n_rows, n_cols, mask, step_size,
            scalebar_enabled, scalebar_location, scalebar_length,
            scalebar_font_size, scalebar_bar_color, scalebar_box_color,
            scalebar_box_alpha, show_legend, title,
            confidence_cmap, dpi,
            ipf_valid_2d=ipf_valid_2d, had_cleanup=had_cleanup,
        )

    # ------------------------------------------------------------------
    # Band Contrast greyscale map (sourced from xmap.prop['bc'], which
    # was carried into the light h5 from the source h5oina)
    # ------------------------------------------------------------------
    if direction == "bc":
        return _render_bc_figure(
            xmap, n_rows, n_cols, mask, step_size,
            scalebar_enabled, scalebar_location, scalebar_length,
            scalebar_font_size, scalebar_bar_color, scalebar_box_color,
            scalebar_box_alpha, show_legend, title, dpi,
            ipf_valid_2d=ipf_valid_2d, had_cleanup=had_cleanup,
        )

    # ------------------------------------------------------------------
    # Phase map / IPF render
    # ------------------------------------------------------------------
    from tools.phase_map_generator import compute_ipf_colors

    if direction == "phase":
        phase_ids_full = effective_pid_2d
        if phase_ids_full is None:
            # No cleanup, no xmap shape match — fall back to raw ids.
            phase_ids_full = np.full((n_rows, n_cols), -1, dtype=np.int16)

        # Build full-grid color array. Colour is keyed on phase ID via a
        # collision-free HSV-spaced palette so every distinct phase gets its
        # own hue even with ~19 phases (the old 8-slot hash wrapped around).
        # Phases named "not_indexed" collapse into the unindexed bucket.
        phase_colors = build_phase_color_map(xmap, color_overrides)
        color_grid = np.empty((n_rows, n_cols, 3), dtype=np.float32)
        color_grid[...] = UNINDEXED_COLOR
        for pid in np.unique(phase_ids_full):
            if pid < 0:
                continue
            color = phase_colors.get(int(pid))
            if color is None:
                # Unindexed sentinel — leave as UNINDEXED_COLOR, no legend patch.
                continue
            color_grid[phase_ids_full == pid] = color
        rendered_colors = color_grid  # already 3D (rows, cols, 3)
    else:
        # IPF mode — needs valid rotations on the xmap. Loaded .h5/.ang files
        # that were saved without orientations (e.g. rich H5 exports from
        # /api/indexing/export) return all-zero IPF colors, producing a blank
        # grey image with no explanation. Detect + raise instead.
        rot = getattr(xmap, "rotations", None)
        rot_data = getattr(rot, "data", None) if rot is not None else None
        has_rotations = (
            rot is not None and rot_data is not None
            and np.asarray(rot_data).size > 0
            and not np.allclose(np.asarray(rot_data), 0)
        )
        if not has_rotations:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This dataset has no orientation data — IPF maps require "
                    "rotations from an indexing run. Switch to Phase Map / CI "
                    "Heatmap, or load a result that includes orientations."
                ),
            )
        rendered_colors = compute_ipf_colors(xmap, direction, r_user=_active_r_user())

        # Apply the cleanup mask to IPF: rejected, filled and modal-voted
        # pixels all show as grey here because their orientation is either
        # missing or no longer trustworthy after a relabel. The helper has
        # already encoded this decision in ipf_valid_2d.
        if had_cleanup and ipf_valid_2d is not None:
            try:
                ipf_2d = np.asarray(rendered_colors).reshape(-1, 3)
                if ipf_2d.shape[0] == n_rows * n_cols:
                    ipf_3d = ipf_2d.reshape(n_rows, n_cols, 3).astype(np.float32).copy()
                    ipf_3d[~ipf_valid_2d] = UNINDEXED_COLOR
                    rendered_colors = ipf_3d
                elif mask is not None and not mask.all() and ipf_2d.shape[0] == int(mask.sum()):
                    # Colors cover only the masked (indexed) pixels —
                    # expand to full grid before applying the cleanup mask.
                    ipf_3d = np.empty((n_rows, n_cols, 3), dtype=np.float32)
                    ipf_3d[...] = UNINDEXED_COLOR
                    ipf_3d[mask] = ipf_2d
                    ipf_3d[~ipf_valid_2d] = UNINDEXED_COLOR
                    rendered_colors = ipf_3d
            except Exception as e:
                logger.debug("IPF cleanup mask skipped: %s", e)

    color_map = np.zeros((n_rows, n_cols, 3))
    colors_flat = np.asarray(rendered_colors).reshape(-1, 3)
    n_color_pixels = colors_flat.shape[0]

    if mask is not None and not mask.all():
        n_masked = int(mask.sum())
        if n_color_pixels == n_masked:
            color_map[mask] = colors_flat
        elif n_color_pixels == n_rows * n_cols:
            color_map = colors_flat.reshape(n_rows, n_cols, 3)
        else:
            logger.warning(
                "Color count mismatch: %d colors vs %d masked vs %d total",
                n_color_pixels, n_masked, n_rows * n_cols,
            )
            if n_color_pixels == n_rows * n_cols:
                color_map = colors_flat.reshape(n_rows, n_cols, 3)
    else:
        color_map = colors_flat.reshape(n_rows, n_cols, 3)

    # Aztec-Crystal-style BC modulation: multiply RGB by per-pixel BC
    # brightness so grain boundaries naturally darken without needing a
    # separate boundary detection. Applied after the cleanup-grey because
    # we want unindexed pixels to stay grey, not become black.
    if bc_modulate:
        bc_factor = _bc_brightness_factor(xmap, n_rows, n_cols, mask)
        if bc_factor is not None:
            color_map = (color_map * bc_factor[..., None]).clip(0.0, 1.0)
        else:
            logger.info("bc_modulate requested but no BC available — rendering unmodulated")

    # Figure layout — adaptive based on map aspect ratio:
    #   * Wide landscape map (aspect > 1.3) → keys placed in a ROW BELOW
    #     the map.  Wide maps waste the space right of them and squeeze a
    #     vertical key column into a skinny strip; horizontal layout uses
    #     the native aspect instead.
    #   * Square / portrait map (aspect ≤ 1.3) → keys placed in a COLUMN
    #     to the RIGHT as before.
    #   * Non-IPF modes → no keys, single axis.
    want_ipf_keys = (
        show_ipf_keys
        and direction.upper() in ("X", "Y", "Z")
        and bool(_group_phases_by_symmetry(xmap))
    )
    if want_ipf_keys:
        groups = _group_phases_by_symmetry(xmap)
        n_keys = len(groups)
        map_aspect = n_cols / max(n_rows, 1)  # width / height
        keys_below = map_aspect > 1.3

        if keys_below:
            # Width-matched: map on top at native aspect, keys in a single
            # row below. Figure height = map height + 1 key row.
            # Each key cell ≈ 1.8 in wide for legibility.
            fig_w = 10.0
            # Map subaxis height follows the data aspect; add ~2 in for
            # the keys row (triangle + bar + title).
            map_h = fig_w / map_aspect
            # Cap really-tall narrow maps so the figure stays reasonable.
            map_h = min(max(map_h, 2.5), 6.0)
            keys_h = 2.2
            fig = plt.figure(figsize=(fig_w, map_h + keys_h), dpi=dpi)
            outer = fig.add_gridspec(
                2, 1,
                height_ratios=[map_h, keys_h],
                hspace=0.25,
            )
            ax = fig.add_subplot(outer[0, 0])
            _draw_ipf_color_keys(fig, outer[1, 0], xmap, direction, color_overrides,
                                  orientation="horizontal")
        else:
            # Original right-column layout for square / portrait maps.
            key_col_w = max(1.6, min(2.6, 0.9 + 0.4 * n_keys))
            fig = plt.figure(figsize=(8 + key_col_w, 6), dpi=dpi)
            outer = fig.add_gridspec(1, 2, width_ratios=[8, key_col_w], wspace=0.15)
            ax = fig.add_subplot(outer[0, 0])
            _draw_ipf_color_keys(fig, outer[0, 1], xmap, direction, color_overrides,
                                  orientation="vertical")
    else:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.imshow(color_map)

    # Confidence overlay (only for phase / IPF modes; CI heatmap path handles
    # its own colorbar and doesn't use this overlay).
    if confidence_overlay and hasattr(xmap, 'scores'):
        try:
            scores = xmap.scores[:, 0] if xmap.scores.ndim > 1 else xmap.scores
            score_map = np.zeros((n_rows, n_cols))
            if mask is not None and not mask.all():
                score_map[mask] = scores
            else:
                score_map = np.asarray(scores).reshape(n_rows, n_cols)
            ax.imshow(score_map, cmap=confidence_cmap, alpha=confidence_alpha,
                      vmin=0, vmax=1)
        except Exception:
            pass

    ax.axis('off')
    display_title = title if title else ("Phase Map" if direction == "phase" else f"IPF-{direction}")
    if show_legend:
        ax.set_title(display_title)

    if direction == "phase" and show_legend:
        import matplotlib.patches as mpatches
        phase_colors = build_phase_color_map(xmap, color_overrides)
        patches = []
        for pid in sorted(phase_colors.keys()):
            try:
                phase_name = xmap.phases[pid].name
            except (KeyError, IndexError):
                phase_name = f"Phase {pid}"
            color = phase_colors[pid]
            patches.append(mpatches.Patch(color=color, label=phase_name or f"Phase {pid}"))
        if patches:
            ax.legend(
                handles=patches,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=min(len(patches), 4),
                fontsize=9,
                framealpha=0.8,
                edgecolor='#444444',
                facecolor='#1a1a2e',
                labelcolor='white',
            )
            fig.subplots_adjust(bottom=0.12)

    _apply_scalebar(
        ax, step_size,
        enabled=scalebar_enabled, location=scalebar_location,
        length=scalebar_length, font_size=scalebar_font_size,
        bar_color=scalebar_bar_color, box_color=scalebar_box_color,
        box_alpha=scalebar_box_alpha,
    )

    meta = {
        "direction": direction,
        "shape": [n_rows, n_cols],
        "step_size": step_size,
    }
    return fig, meta


def _extract_ci_array(
    *, kind: str, xmap, result, dataset, n_rows: int, n_cols: int, mask
) -> "np.ndarray":
    """Pull a (n_rows, n_cols) CI/uncertainty/per-phase-CI array.

    Returns NaN where data is unavailable. Used by both _render_ci_figure
    and _compute_layer_rgba.
    """
    import numpy as np

    if kind == "uncertainty":
        per_phase = (
            (result.metadata.get("per_phase_data") if result is not None else None)
            or (getattr(dataset, "per_phase_data", None) if dataset is not None else None)
            or {}
        )
        if len(per_phase) < 2:
            arr = np.full((n_rows, n_cols), np.nan)
        else:
            reshaped = []
            for name in list(per_phase.keys()):
                arr_pp = np.asarray(per_phase[name])
                if arr_pp.size != n_rows * n_cols:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Per-phase data for {name!r} has size {arr_pp.size}, "
                               f"expected {n_rows * n_cols}.",
                    )
                reshaped.append(arr_pp.reshape(n_rows, n_cols))
            stacked = np.stack(reshaped, axis=0)
            sorted_desc = np.sort(stacked, axis=0)[::-1]
            arr = sorted_desc[0] - sorted_desc[1]
        return arr

    if kind.startswith("ci_"):
        phase_name = kind[3:]
        per_phase = (
            (result.metadata.get("per_phase_data") if result is not None else None)
            or (getattr(dataset, "per_phase_data", None) if dataset is not None else None)
            or {}
        )
        if phase_name not in per_phase:
            raise HTTPException(
                status_code=400,
                detail=f"No per-phase CI for {phase_name!r}.",
            )
        per_phase_arr = np.asarray(per_phase[phase_name])
        if per_phase_arr.size != n_rows * n_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Per-phase CI for {phase_name!r} has size {per_phase_arr.size}, "
                       f"expected {n_rows * n_cols}.",
            )
        return per_phase_arr.reshape(n_rows, n_cols).astype(float)

    # Bare "ci" — best-phase confidence index. Try multiple sources in
    # order: result.confidence_scores (in-memory indexing result),
    # xmap.scores (some kikuchipy versions), xmap.prop['ci'] (Hough
    # output via orix load), xmap.prop['scores'] (alt name).
    # Only accept a source that is a NON-EMPTY NUMERIC array. A missing source
    # (None), an empty array, or a non-numeric placeholder (e.g. a MagicMock
    # attribute in tests, or an object-dtype array) must fall through to the
    # next source rather than be mistaken for real scores.
    def _numeric_scores(obj):
        if obj is None:
            return None
        sc = np.asarray(obj)
        if sc.size == 0 or not np.issubdtype(sc.dtype, np.number):
            return None
        return sc

    scores = None
    if result is not None:
        sc = _numeric_scores(getattr(result, "confidence_scores", None))
        if sc is not None:
            scores = sc[-1].ravel() if sc.ndim >= 2 else sc.ravel()
    if scores is None and hasattr(xmap, "scores"):
        scores = _numeric_scores(getattr(xmap, "scores", None))
    if scores is None and hasattr(xmap, "prop"):
        try:
            for key in ("ci", "scores"):
                if key in xmap.prop:
                    sc = _numeric_scores(xmap.prop[key])
                    if sc is not None:
                        scores = sc
                        break
        except TypeError:
            pass  # xmap.prop isn't a real mapping (e.g. a MagicMock)
    if scores is None:
        raise HTTPException(
            status_code=400,
            detail="No CI scores available on the indexing result. "
                   "Switch to per-phase CI (ci_<phase>) or load a result that includes scores.",
        )
    scores = np.asarray(scores)
    if scores.ndim > 1:
        scores = scores[:, 0]
    if scores.size == n_rows * n_cols:
        return scores.reshape(n_rows, n_cols).astype(float)
    if mask is not None and scores.size == int(mask.sum()):
        out = np.full((n_rows, n_cols), np.nan)
        out[mask] = scores
        return out
    raise HTTPException(
        status_code=400,
        detail=f"CI scores have unexpected size: got {scores.size}, "
               f"expected {n_rows * n_cols} or {int(mask.sum()) if mask is not None else 'matching mask'}.",
    )


# ---------------------------------------------------------------------------
# Forward NCC Diagnostic Layers (Task 11)
# ---------------------------------------------------------------------------
_DIAG_KIND_TO_KEY = {
    "forward_ncc":       "ncc_map",
    "local_anomaly":     "local_anomaly_map",
    "pc_sensitivity":    "pc_sensitivity_map",
    "pattern_residual":  "pattern_residual_map",
}
_DIAG_KIND_TO_CMAP = {
    "forward_ncc":       "viridis",
    "local_anomaly":     "RdBu_r",
    "pc_sensitivity":    "magma",
    "pattern_residual":  "inferno",
}

# ---------------------------------------------------------------------------
# Joint R+PC Refinement Layers (Phase B, Task 10)
# ---------------------------------------------------------------------------
_REF_KIND_TO_KEY = {
    "refined_ncc":         "refined_ncc_map",
    "convergence_status":  "convergence_status_map",
    "orientation_delta":   "orientation_delta_map",
    "pc_delta_x":          "pc_delta_x_map",
    "pc_delta_y":          "pc_delta_y_map",
    "pc_delta_l":          "pc_delta_l_map",
}
_REF_KIND_TO_CMAP = {
    "refined_ncc":         "viridis",
    "convergence_status":  "Set1",      # categorical
    "orientation_delta":   "inferno",
    "pc_delta_x":          "RdBu_r",
    "pc_delta_y":          "RdBu_r",
    "pc_delta_l":          "RdBu_r",
}


def _ref_rgba(kind: str, n_rows: int, n_cols: int) -> "np.ndarray":
    """Render one of the six refinement layers."""
    import numpy as np
    import matplotlib
    result = get_last_indexing_result()
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "no active indexing result"})
    block = (result.metadata or {}).get("refinement")
    if block is None:
        raise HTTPException(status_code=404, detail={"error": "refinement not yet computed"})
    arr = block[_REF_KIND_TO_KEY[kind]]
    if arr.shape != (n_rows, n_cols):
        raise HTTPException(status_code=400,
                            detail={"error": f"refinement shape {arr.shape} != ({n_rows}, {n_cols})"})

    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    elif kind == "refined_ncc":
        vmin, vmax = -1.0, 1.0
    elif kind == "convergence_status":
        vmin, vmax = 0.0, 2.0
    elif kind in ("pc_delta_x", "pc_delta_y", "pc_delta_l"):
        vmax = float(np.percentile(np.abs(valid), 99))
        if vmax == 0:
            vmax = 1e-6
        vmin = -vmax
    else:  # orientation_delta
        vmin = 0.0
        vmax = float(np.percentile(valid, 95))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    cmap = matplotlib.colormaps[_REF_KIND_TO_CMAP[kind]]
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isnan(arr), 0, 255).astype(np.uint8)
    return rgba


def _diag_rgba(kind: str, n_rows: int, n_cols: int) -> "np.ndarray":
    """Render one of the four forward-diagnostic layers as transparent RGBA."""
    import numpy as np
    import matplotlib
    result = get_last_indexing_result()
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "no active indexing result"})
    diag = (result.metadata or {}).get("forward_diagnostics")
    if diag is None:
        raise HTTPException(status_code=404, detail={"error": "forward diagnostics not yet computed"})
    arr = diag[_DIAG_KIND_TO_KEY[kind]]
    if arr.shape != (n_rows, n_cols):
        raise HTTPException(status_code=400, detail={"error": f"diagnostic shape {arr.shape} != ({n_rows}, {n_cols})"})

    # Color range
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    elif kind == "local_anomaly":
        vmax = float(np.percentile(np.abs(valid), 99))
        vmin = -vmax
        if vmax == 0:
            vmax = 1e-6
            vmin = -1e-6
    elif kind == "forward_ncc":
        vmin, vmax = -1.0, 1.0
    else:
        vmin = 0.0
        vmax = float(np.percentile(valid, 95))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    cmap = matplotlib.colormaps[_DIAG_KIND_TO_CMAP[kind]]
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (cmap(norm) * 255).astype(np.uint8)   # (H, W, 4)
    # Make NaN pixels transparent
    rgba[..., 3] = np.where(np.isnan(arr), 0, 255).astype(np.uint8)
    return rgba


def _compute_layer_rgba(
    *,
    kind: str,
    ci_threshold: float = 0.0,
    uncertainty_threshold: float = 0.0,
    min_cluster_size: int = 0,
    fill_unindexed: bool = False,
    modal_filter_size: int = 0,
    color_overrides: dict | None = None,
    band_min: float = 0.0,
    band_max: float = 1.0,
    out_color: str = "ff3333",
    out_alpha: int = 153,
    grain_stabilized: bool = False,
    phase_filter: int | None = None,
) -> "np.ndarray":
    """Render a single layer as a raw (H, W, 4) uint8 RGBA array.

    Background (unindexed / NaN) pixels have alpha=0 so the frontend can
    composite without halo artefacts. No scalebar, legend, or IPF key.
    Native pixel resolution.

    Supported kinds: phase, ipf-x, ipf-y, ipf-z, bc, ci, ci_<phase>, uncertainty,
    plus forward diagnostics: forward_ncc, local_anomaly, pc_sensitivity,
    pattern_residual.

    Raises HTTPException(400) on unknown kind or missing data.
    Raises HTTPException(404) on missing forward diagnostics.
    """
    import numpy as np

    # Forward NCC diagnostic dispatch — handled by a separate helper that
    # reads result.metadata["forward_diagnostics"] and applies its own colormap.
    if kind in _DIAG_KIND_TO_KEY:
        result = get_last_indexing_result()
        if result is None:
            raise HTTPException(status_code=404, detail={"error": "no active indexing result"})
        n_rows, n_cols = result.original_shape
        return _diag_rgba(kind, n_rows, n_cols)

    # Joint R+PC refinement dispatch (Phase B) — reads result.metadata["refinement"].
    if kind in _REF_KIND_TO_KEY:
        result = get_last_indexing_result()
        if result is None:
            raise HTTPException(status_code=404, detail={"error": "no active indexing result"})
        n_rows, n_cols = result.original_shape
        return _ref_rgba(kind, n_rows, n_cols)

    # Render-verified phase check (Stage A) — per-grain margin of the stored
    # phase vs the best alternative (stored − best alt). Positive/green =
    # stored phase wins, negative/red = another phase renders clearly better,
    # transparent = healthy grain (never questioned) or unindexed. Data comes
    # from POST /api/indexing/phase-check.
    if kind == "phase_margin":
        result = get_last_indexing_result()
        if result is None:
            raise HTTPException(status_code=404, detail={"error": "no active indexing result"})
        pc = (getattr(result, "metadata", None) or {}).get("phase_check")
        if not pc or pc.get("margin_map") is None:
            raise HTTPException(status_code=404,
                                detail={"error": "phase check not yet computed"})
        import matplotlib
        n_rows, n_cols = result.original_shape
        arr = np.asarray(pc["margin_map"], dtype=np.float64).reshape(n_rows, n_cols)
        finite = arr[np.isfinite(arr)]
        vmax = max(0.15, float(np.percentile(np.abs(finite), 99))) if finite.size else 0.15
        cmap = matplotlib.colormaps["RdYlGn"]
        norm = np.clip((arr + vmax) / (2 * vmax), 0.0, 1.0)
        rgba = (cmap(norm) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(np.isfinite(arr), 255, 0).astype(np.uint8)
        return rgba

    valid_kinds = {"phase", "ipf-x", "ipf-y", "ipf-z", "bc", "ci", "uncertainty", "ci-threshold"}
    if kind not in valid_kinds and not kind.startswith("ci_"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown layer kind: {kind!r}. Valid: {sorted(valid_kinds)} or 'ci_<phase>'.",
        )

    result = get_last_indexing_result()
    dataset = get_analysis_dataset()
    if result is None and dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No indexing result or analysis dataset available. Load data first.",
        )

    if result is not None:
        xmap = result.xmap
        n_rows, n_cols = result.original_shape
        mask = result.selection_mask
    else:
        xmap = dataset.xmap
        n_rows, n_cols = dataset.shape
        mask = None

    effective_pid_2d, ipf_valid_2d, had_cleanup = _compute_effective_phase_ids_2d(
        xmap, n_rows, n_cols, mask, result,
        ci_threshold=ci_threshold,
        uncertainty_threshold=uncertainty_threshold,
        min_cluster_size=min_cluster_size,
        fill_unindexed=fill_unindexed,
        modal_filter_size=modal_filter_size,
    )

    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.uint8)
    alpha = np.zeros((n_rows, n_cols), dtype=bool)

    if kind == "phase":
        # Collision-free HSV palette keyed on phase id (see build_phase_color_map).
        phase_colors = build_phase_color_map(xmap, color_overrides)
        rgb = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
        for pid in np.unique(effective_pid_2d):
            if pid < 0:
                continue
            color = phase_colors.get(int(pid))
            if color is None:
                continue
            sel = (effective_pid_2d == pid)
            rgb[sel] = color
            alpha[sel] = True
        rgba[..., :3] = (rgb * 255.0).clip(0, 255).astype(np.uint8)

    elif kind.startswith("ipf-"):
        from tools.phase_map_generator import compute_ipf_colors
        direction_map = {"ipf-x": "X", "ipf-y": "Y", "ipf-z": "Z"}
        direction = direction_map[kind]
        rot = getattr(xmap, "rotations", None)
        rot_data = getattr(rot, "data", None) if rot is not None else None
        has_rot = (
            rot is not None and rot_data is not None
            and np.asarray(rot_data).size > 0
            and not np.allclose(np.asarray(rot_data), 0)
        )
        if not has_rot:
            raise HTTPException(
                status_code=400,
                detail="No orientation data — IPF layers require indexing rotations.",
            )
        try:
            if grain_stabilized:
                # Grain-consistent IPF (v2): each pixel keeps its OWN direction,
                # but every pixel of a grain is routed through the same side of
                # the colour-key discontinuity, so gradients survive and only the
                # low-symmetry speckle is removed (see
                # compute_ipf_colors_grain_consistent). Returns the full grid.
                from tools.phase_map_generator import (
                    compute_ipf_colors_grain_consistent,
                )
                ipf_3d = compute_ipf_colors_grain_consistent(
                    xmap, direction, r_user=_active_r_user(),
                    n_rows=n_rows, n_cols=n_cols, mask=mask,
                )
            else:
                ipf_raw = compute_ipf_colors(
                    xmap, direction, r_user=_active_r_user())
                ipf = np.asarray(ipf_raw).reshape(-1, 3)
                if ipf.shape[0] == n_rows * n_cols:
                    ipf_3d = ipf.reshape(n_rows, n_cols, 3)
                elif mask is not None and ipf.shape[0] == int(mask.sum()):
                    ipf_3d = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
                    ipf_3d[mask] = ipf
                else:
                    raise HTTPException(
                        status_code=500, detail="IPF shape mismatch.")
        except ValueError as e:
            # Phase has no usable symmetry — compute_ipf_colors raises so we can
            # surface a useful error instead of silently rendering grey.
            raise HTTPException(status_code=400, detail=str(e))
        rgba[..., :3] = (np.asarray(ipf_3d) * 255.0).clip(0, 255).astype(np.uint8)
        alpha[:] = (effective_pid_2d >= 0)
        if ipf_valid_2d is not None:
            alpha &= ipf_valid_2d
        if phase_filter is not None:
            # Per-phase IPF view (community standard): only the selected
            # phase keeps its IPF colours; every other phase goes transparent
            # so it can't be confused with same-RGB directions of a different
            # colour key. Colour math is untouched — this is an alpha mask,
            # so it composes with BOTH the standard and grain-stabilised path.
            alpha &= (effective_pid_2d == int(phase_filter))

    elif kind == "bc":
        # Primary source: xmap.prop['bc'] propagated through the indexing
        # pipeline. Spherical / dictionary results often don't carry BC,
        # so we fall back to the native H5OINA dataset of the currently
        # loaded EBSD file (same source the EDS Analysis page reads from).
        bc_factor = _bc_brightness_factor(xmap, n_rows, n_cols, mask)
        bc_2d = None
        if bc_factor is None:
            native_bc = None
            # Prefer the result's OWN source file (read with a private handle —
            # no shared-session mutation), so BC works even when a different
            # file is currently open in the viewer/EDS page. This is the
            # cross-file BC fix: viewing result A's map while file B is open no
            # longer 400s just because B has no/mismatched BC.
            try:
                src = (result.metadata.get("source_file")
                       if result is not None and isinstance(getattr(result, "metadata", None), dict)
                       else None)
            except Exception:
                src = None
            if src:
                try:
                    from backend.api.routes.virtual_images import _read_native_band_contrast_from
                    native_bc = _read_native_band_contrast_from(src)
                except Exception:
                    native_bc = None
            # Fall back to whatever file is currently open (legacy behaviour).
            if native_bc is None:
                try:
                    from backend.api.routes.virtual_images import _read_native_band_contrast
                    native_bc = _read_native_band_contrast()
                except Exception:
                    native_bc = None
            if native_bc is not None and native_bc.shape == (n_rows, n_cols):
                # Normalise to [0, 1] for the grey ramp.
                lo = float(np.nanmin(native_bc))
                hi = float(np.nanmax(native_bc))
                if hi > lo:
                    bc_2d = ((native_bc - lo) / (hi - lo)).astype(np.float32)
                else:
                    bc_2d = np.zeros_like(native_bc, dtype=np.float32)
        else:
            bc_2d = bc_factor

        if bc_2d is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No Band Contrast available: indexing result doesn't carry "
                    "BC and no source H5OINA file is currently open. Load the "
                    "source file in the EBSD Viewer to enable BC overlay."
                ),
            )

        grey = (bc_2d * 255.0).clip(0, 255).astype(np.uint8)
        rgba[..., 0] = grey
        rgba[..., 1] = grey
        rgba[..., 2] = grey
        # When BC comes from the source file (not the xmap), unindexed
        # pixels still have valid BC — show them all instead of clipping
        # to effective_pid_2d. When BC comes from the xmap, only indexed
        # pixels are valid.
        if bc_factor is None:
            alpha[:] = True
        else:
            alpha[:] = (effective_pid_2d >= 0)

    elif kind == "ci" or kind.startswith("ci_") or kind == "uncertainty":
        ci_arr_2d = _extract_ci_array(
            kind=kind, xmap=xmap, result=result, dataset=dataset,
            n_rows=n_rows, n_cols=n_cols, mask=mask,
        )
        import matplotlib
        cmap = matplotlib.colormaps["RdYlGn"]
        valid = np.isfinite(ci_arr_2d)
        if valid.any():
            v = ci_arr_2d.copy()
            vmin = float(np.nanmin(v))
            vmax = float(np.nanmax(v))
            if vmax > vmin:
                v = (v - vmin) / (vmax - vmin)
            else:
                v = np.zeros_like(v)
            v[~valid] = 0
            rgba_float = cmap(v)
            rgba[..., :3] = (rgba_float[..., :3] * 255.0).astype(np.uint8)
        alpha[:] = valid

    elif kind == "ci-threshold":
        # Misindex-diagnose helper (2026-05-26). Read CI per pixel; paint
        # pixels INSIDE the [band_min, band_max] window in ``out_color``
        # at ``out_alpha``. Pixels outside the band (= "OK" confidence)
        # stay transparent so the user's underlying map (Phase, IPF, ...)
        # shines through. Default band [0.0, 0.3] flags low-CI suspects.
        # The slider on the frontend is fixed to [0, 1] so the band is
        # interpretable across indexing methods.
        ci_arr_2d = _extract_ci_array(
            kind="ci", xmap=xmap, result=result, dataset=dataset,
            n_rows=n_rows, n_cols=n_cols, mask=mask,
        )
        valid = np.isfinite(ci_arr_2d)
        try:
            r_hex = max(0, min(255, int(out_color[0:2], 16)))
            g_hex = max(0, min(255, int(out_color[2:4], 16)))
            b_hex = max(0, min(255, int(out_color[4:6], 16)))
        except (ValueError, IndexError):
            r_hex, g_hex, b_hex = 255, 51, 51  # fallback red
        a_clamp = max(0, min(255, int(out_alpha)))
        inside = (
            valid
            & (ci_arr_2d >= float(band_min))
            & (ci_arr_2d <= float(band_max))
        )
        rgba[inside, 0] = r_hex
        rgba[inside, 1] = g_hex
        rgba[inside, 2] = b_hex
        alpha[:] = inside
        rgba[..., 3] = np.where(alpha, a_clamp, 0).astype(np.uint8)
        return rgba  # short-circuit — alpha already assigned with custom value

    rgba[..., 3] = np.where(alpha, 255, 0).astype(np.uint8)
    return rgba


def _render_ci_figure(
    direction, last_result, analysis_dataset,
    n_rows, n_cols, mask, step_size,
    scalebar_enabled, scalebar_location, scalebar_length,
    scalebar_font_size, scalebar_bar_color, scalebar_box_color,
    scalebar_box_alpha, show_legend, title,
    cmap_name, dpi,
    *, ipf_valid_2d=None, had_cleanup: bool = False,
):
    """Render a CI heatmap — returns (fig, meta)."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = np.zeros((n_rows, n_cols), dtype=np.float32)
    map_title = title or "CI Heatmap"
    vmin, vmax = 0.0, 1.0

    # Per-phase CI source — prefer the in-memory indexing result, fall
    # back to whatever the analysis_dataset was told to remember when the
    # user opened a .h5 via "Add file…" (analysis.load copies it off the
    # custom reader onto dataset.per_phase_data).
    per_phase_data = {}
    if last_result is not None:
        per_phase_data = last_result.metadata.get('per_phase_data', {}) or {}
    if not per_phase_data and analysis_dataset is not None:
        ppd = getattr(analysis_dataset, 'per_phase_data', None) or {}
        per_phase_data = ppd

    if direction == "ci":
        map_title = title or "Confidence Index (Best Phase)"
        if last_result is not None and last_result.confidence_scores is not None:
            scores_flat = np.asarray(last_result.confidence_scores).ravel()
            n_pixels = n_rows * n_cols
            if mask is not None and not mask.all():
                n_sel = int(mask.sum())
                if scores_flat.size >= n_sel:
                    data[mask] = scores_flat[:n_sel]
                else:
                    padded = np.full(n_sel, np.nan, dtype=np.float32)
                    padded[:scores_flat.size] = scores_flat
                    data[mask] = padded
            else:
                if scores_flat.size == n_pixels:
                    data = scores_flat.reshape(n_rows, n_cols).astype(np.float32)
                elif scores_flat.size > n_pixels and scores_flat.size % n_pixels == 0:
                    data = scores_flat[:n_pixels].reshape(n_rows, n_cols).astype(np.float32)
        elif analysis_dataset is not None and hasattr(analysis_dataset, 'xmap'):
            try:
                data = np.array(analysis_dataset.xmap.prop.get('ci', np.zeros(n_rows * n_cols))).reshape(n_rows, n_cols)
            except Exception:
                pass

    elif direction.startswith("ci_"):
        phase_name = direction[3:]
        map_title = title or f"CI — {phase_name}"
        if phase_name in per_phase_data and per_phase_data[phase_name].get('ci') is not None:
            data = per_phase_data[phase_name]['ci']
        else:
            raise HTTPException(
                status_code=400,
                detail=f"No per-phase CI data for '{phase_name}'. "
                       f"Available phases: {list(per_phase_data.keys())}"
            )

    elif direction == "uncertainty":
        map_title = title or "Phase Assignment Uncertainty"
        if len(per_phase_data) >= 2:
            ci_maps = []
            for name in per_phase_data:
                ci = per_phase_data[name].get('ci')
                if ci is not None:
                    ci_maps.append(ci)
            if len(ci_maps) >= 2:
                ci_stack = np.stack(ci_maps, axis=0)
                best_ci = np.max(ci_stack, axis=0)
                ci_masked = ci_stack.copy()
                best_idx = np.argmax(ci_stack, axis=0)
                rows_idx, cols_idx = np.indices(best_idx.shape)
                ci_masked[best_idx, rows_idx, cols_idx] = -np.inf
                second_ci = np.max(ci_masked, axis=0)
                second_ci = np.where(np.isinf(second_ci), 0.0, second_ci)
                data = (best_ci - second_ci).astype(np.float32)
                vmax = float(np.nanmax(data)) if np.nanmax(data) > 0 else 1.0
        else:
            raise HTTPException(
                status_code=400,
                detail="Uncertainty requires multi-phase indexing (>= 2 phases)"
            )

    # Apply cleanup mask — rejected / relabelled pixels become NaN so the
    # colormap's "bad" colour renders them as the same unindexed grey the
    # Phase / IPF views use. Consistent handling across display modes.
    if had_cleanup and ipf_valid_2d is not None and ipf_valid_2d.shape == data.shape:
        data = data.astype(np.float32)
        data[~ipf_valid_2d] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    cmap = plt.get_cmap(cmap_name or 'RdYlGn').copy()
    cmap.set_bad(UNINDEXED_COLOR)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    if show_legend:
        ax.set_title(map_title, color='white', fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white', labelsize=8)
    cbar.set_label(
        'Uncertainty (low = ambiguous)' if direction == "uncertainty" else 'CI',
        color='white', fontsize=9,
    )
    fig.patch.set_facecolor('#1a1b26')
    ax.set_facecolor('#1a1b26')

    _apply_scalebar(
        ax, step_size,
        enabled=scalebar_enabled, location=scalebar_location,
        length=scalebar_length, font_size=scalebar_font_size,
        bar_color=scalebar_bar_color, box_color=scalebar_box_color,
        box_alpha=scalebar_box_alpha,
    )

    meta = {
        "direction": direction,
        "shape": [n_rows, n_cols],
        "step_size": step_size,
        "min_val": float(np.nanmin(data)),
        "max_val": float(np.nanmax(data)),
        "mean_val": float(np.nanmean(data)),
    }
    return fig, meta


def _bc_brightness_factor(xmap, n_rows: int, n_cols: int, mask) -> "np.ndarray | None":
    """Compute a (n_rows, n_cols) 0..1 brightness factor from xmap.prop['bc'].

    Returns None when BC is missing, has the wrong size, or the array is all
    zero. Uses the 98th percentile to set 'fully bright' so a few saturated
    pixels don't crush the whole map. This mirrors what Aztec Crystal does
    when it overlays IPF onto BC.
    """
    import numpy as np
    bc_raw = xmap.prop.get("bc") if hasattr(xmap, "prop") else None
    if bc_raw is None:
        return None
    bc_flat = np.asarray(bc_raw).ravel().astype(np.float32)
    if bc_flat.size == 0:
        return None
    n_total = n_rows * n_cols
    bc_2d = np.zeros((n_rows, n_cols), dtype=np.float32)
    if bc_flat.size == n_total:
        bc_2d = bc_flat.reshape(n_rows, n_cols)
    elif mask is not None and not mask.all() and bc_flat.size == int(mask.sum()):
        bc_2d[mask] = bc_flat
    else:
        return None
    p98 = float(np.percentile(bc_2d[bc_2d > 0], 98)) if np.any(bc_2d > 0) else 0.0
    if p98 <= 0:
        return None
    return np.clip(bc_2d / p98, 0.0, 1.0)


def _render_bc_figure(
    xmap, n_rows, n_cols, mask, step_size,
    scalebar_enabled, scalebar_location, scalebar_length,
    scalebar_font_size, scalebar_bar_color, scalebar_box_color,
    scalebar_box_alpha, show_legend, title, dpi,
    *, ipf_valid_2d=None, had_cleanup: bool = False,
):
    """Render the Band Contrast quality map as a greyscale image.

    BC is the standard EBSD pattern-quality metric: bright = clean
    diffraction, dark = grain boundary or low-quality pixel. Pulls the
    per-pixel values from xmap.prop['bc'] (carried in via the light-h5
    quality-field copy) and renders them with the 'gray' colormap.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bc_raw = xmap.prop.get("bc") if hasattr(xmap, "prop") else None
    if bc_raw is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No Band Contrast data on this dataset. The light h5 was "
                "exported before BC carry-over was added — re-export from "
                "the checkpoint or open the original h5oina."
            ),
        )

    bc_flat = np.asarray(bc_raw).ravel().astype(np.float32)
    n_total = n_rows * n_cols
    data = np.zeros((n_rows, n_cols), dtype=np.float32)
    if bc_flat.size == n_total:
        data = bc_flat.reshape(n_rows, n_cols)
    elif mask is not None and not mask.all() and bc_flat.size == int(mask.sum()):
        data[mask] = bc_flat
    else:
        raise HTTPException(
            status_code=400,
            detail=f"BC size {bc_flat.size} doesn't fit grid {n_rows}x{n_cols}",
        )

    # Honour the cleanup mask — rejected pixels show as the unindexed grey
    # so BC-map and Phase-map line up visually.
    if had_cleanup and ipf_valid_2d is not None and ipf_valid_2d.shape == data.shape:
        data = data.astype(np.float32)
        data[~ipf_valid_2d] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad(UNINDEXED_COLOR)
    # Use 2nd / 98th percentile auto-stretch so the dynamic range matches
    # what the user sees in Aztec — fixed 0..255 wastes contrast when no
    # pixel reaches the extremes.
    finite = data[np.isfinite(data)]
    if finite.size > 0:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max() or 1.0)
    else:
        vmin, vmax = 0.0, 255.0
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    if show_legend:
        ax.set_title(title or "Band Contrast", color='white', fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white', labelsize=8)
    cbar.set_label('BC (bright = better pattern)', color='white', fontsize=9)
    fig.patch.set_facecolor('#1a1b26')
    ax.set_facecolor('#1a1b26')

    _apply_scalebar(
        ax, step_size,
        enabled=scalebar_enabled, location=scalebar_location,
        length=scalebar_length, font_size=scalebar_font_size,
        bar_color=scalebar_bar_color, box_color=scalebar_box_color,
        box_alpha=scalebar_box_alpha,
    )

    meta = {
        "direction": "bc",
        "shape": [n_rows, n_cols],
        "step_size": step_size,
        "min_val": float(np.nanmin(data)),
        "max_val": float(np.nanmax(data)),
        "mean_val": float(np.nanmean(data)),
    }
    return fig, meta


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ExportRequest(BaseModel):
    output_path: str
    format: str = "png"  # "png", "svg", "pdf"
    dpi: int = 300
    # Visual params — must match /render so the file equals what the user sees
    direction: str = "phase"
    title: str = "Phase Map"
    show_legend: bool = True
    step_x: float = 1.0
    step_y: float = 1.0
    scalebar_enabled: bool = True
    scalebar_location: str = "lower right"
    scalebar_length: float = 0.25
    scalebar_font_size: int = 10
    scalebar_bar_color: str = "#ffffff"
    scalebar_box_color: str = "#000000"
    scalebar_box_alpha: float = 0.6
    confidence_overlay: bool = False
    confidence_alpha: float = 0.5
    confidence_cmap: str = "RdYlGn"
    ci_threshold: float = 0.0
    uncertainty_threshold: float = 0.0
    min_cluster_size: int = 0
    fill_unindexed: bool = False
    modal_filter_size: int = 0
    bc_modulate: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/render")
async def render_phase_map(
    direction: str = "Z",
    scalebar_enabled: bool = True,
    scalebar_location: str = "lower right",
    scalebar_length: float = 0.25,
    scalebar_font_size: int = 10,
    scalebar_bar_color: str = "#ffffff",
    scalebar_box_color: str = "#000000",
    scalebar_box_alpha: float = 0.6,
    show_legend: bool = True,
    title: str = "Phase Map",
    step_x: float = 1.0,
    step_y: float = 1.0,
    confidence_overlay: bool = False,
    confidence_alpha: float = 0.5,
    confidence_cmap: str = "RdYlGn",
    # Live cleanup filters — applied to the rendered phase map only.
    # These do NOT modify any file on disk; use /apply-cleanup for that.
    ci_threshold: float = 0.0,
    uncertainty_threshold: float = 0.0,
    min_cluster_size: int = 0,
    fill_unindexed: bool = False,
    modal_filter_size: int = 0,
    # JSON mapping {"phaseName": "#rrggbb"} — empty string / malformed JSON
    # falls back to the deterministic hash palette so a bad override never
    # breaks rendering.
    color_overrides: str = "",
    # Render one IPF stereographic triangle per unique point-group on the
    # right side of the figure. Phase-Map and CI modes ignore this flag.
    show_ipf_keys: bool = True,
    # Aztec-Crystal-style modulation: multiply the rendered RGB by per-pixel
    # BC brightness. Only meaningful for phase / IPF directions. No-op when
    # the dataset has no BC (older light h5 without quality-field copy).
    bc_modulate: bool = False,
):
    """Render the current phase map and return it as a base64 PNG."""
    try:
        import json as _json
        overrides_dict = None
        if color_overrides:
            try:
                parsed = _json.loads(color_overrides)
                if isinstance(parsed, dict):
                    overrides_dict = {str(k): str(v) for k, v in parsed.items()}
            except (ValueError, TypeError):
                logger.debug("Ignoring malformed color_overrides: %r", color_overrides)

        def _work():
            import matplotlib.pyplot as plt
            with _mpl_lock:
                fig, meta = _render_to_figure(
                    direction=direction,
                    show_legend=show_legend,
                    title=title,
                    step_x=step_x, step_y=step_y,
                    scalebar_enabled=scalebar_enabled,
                    scalebar_location=scalebar_location,
                    scalebar_length=scalebar_length,
                    scalebar_font_size=scalebar_font_size,
                    scalebar_bar_color=scalebar_bar_color,
                    scalebar_box_color=scalebar_box_color,
                    scalebar_box_alpha=scalebar_box_alpha,
                    confidence_overlay=confidence_overlay,
                    confidence_alpha=confidence_alpha,
                    confidence_cmap=confidence_cmap,
                    ci_threshold=ci_threshold,
                    uncertainty_threshold=uncertainty_threshold,
                    min_cluster_size=min_cluster_size,
                    fill_unindexed=fill_unindexed,
                    modal_filter_size=modal_filter_size,
                    color_overrides=overrides_dict,
                    show_ipf_keys=show_ipf_keys,
                    bc_modulate=bc_modulate,
                    dpi=150,
                )
                buf = io.BytesIO()
                # Dark face color matters for CI heatmap; harmless for phase/IPF.
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1,
                            dpi=150, facecolor=fig.get_facecolor())
                plt.close(fig)
                buf.seek(0)
                return {"image": base64.b64encode(buf.read()).decode('utf-8'), **meta}

        return await asyncio.to_thread(_work)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to render phase map")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layer")
async def get_layer(
    kind: str,
    ci_threshold: float = 0.0,
    uncertainty_threshold: float = 0.0,
    min_cluster_size: int = 0,
    fill_unindexed: bool = False,
    modal_filter_size: int = 0,
    color_overrides: str = "",
    band_min: float = 0.0,
    band_max: float = 1.0,
    out_color: str = "ff3333",
    out_alpha: int = 153,
    grain_stabilized: bool = False,
    phase_filter: int = -1,
):
    """Return a single layer as base64 RGBA PNG with transparent background.

    For frontend compositing. Unlike /render this returns native pixel
    resolution with NO scalebar / legend / IPF key — just the raw map.
    """
    try:
        import io as _io
        import json as _json
        from PIL import Image

        overrides_dict = None
        if color_overrides:
            try:
                parsed = _json.loads(color_overrides)
                if isinstance(parsed, dict):
                    overrides_dict = {str(k): str(v) for k, v in parsed.items()}
            except (ValueError, TypeError):
                pass

        def _work():
            # Some layer kinds (ci/bc) build a matplotlib figure internally,
            # so hold the matplotlib lock for the whole compute.
            with _mpl_lock:
                rgba = _compute_layer_rgba(
                    kind=kind,
                    ci_threshold=ci_threshold,
                    uncertainty_threshold=uncertainty_threshold,
                    min_cluster_size=min_cluster_size,
                    fill_unindexed=fill_unindexed,
                    modal_filter_size=modal_filter_size,
                    color_overrides=overrides_dict,
                    band_min=band_min,
                    band_max=band_max,
                    out_color=out_color,
                    out_alpha=out_alpha,
                    grain_stabilized=grain_stabilized,
                    phase_filter=None if phase_filter < 0 else int(phase_filter),
                )
                img = Image.fromarray(rgba)  # mode inferred from uint8 HxWx4 → RGBA
                buf = _io.BytesIO()
                img.save(buf, format="PNG", optimize=False)  # speed > size
                buf.seek(0)
                return {
                    "image": base64.b64encode(buf.read()).decode("utf-8"),
                    "kind": kind,
                    "shape": [int(rgba.shape[0]), int(rgba.shape[1])],
                }

        return await asyncio.to_thread(_work)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to render layer %s", kind)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ipf-key")
async def get_ipf_key(direction: str = "Z", phase_filter: int = -1):
    """Render only the IPF colour key(s) on a transparent background.

    One stereographic triangle per Laue class in the active xmap. Used by
    the LayeredCanvas as a floating overlay when an IPF layer is shown.
    ``phase_filter`` (a phase id, -1 = all) restricts the key to a single
    phase — used when the IPF layer itself is phase-filtered so the key on
    screen always matches the colours on the map.

    Returns a base64 PNG. 400 if no result/dataset, or no orientation data.
    """
    try:
        import io as _io
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        direction = direction.upper()
        if direction not in ("X", "Y", "Z"):
            raise HTTPException(status_code=400, detail=f"direction must be X/Y/Z, got {direction!r}")

        result = get_last_indexing_result()
        dataset = get_analysis_dataset()
        if result is None and dataset is None:
            raise HTTPException(status_code=400, detail="No indexing result or analysis dataset available.")

        xmap = result.xmap if result is not None else dataset.xmap

        rot = getattr(xmap, "rotations", None)
        rot_data = getattr(rot, "data", None) if rot is not None else None
        has_rot = (
            rot is not None and rot_data is not None
            and np.asarray(rot_data).size > 0
            and not np.allclose(np.asarray(rot_data), 0)
        )
        if not has_rot:
            raise HTTPException(status_code=400, detail="No orientation data — IPF keys require indexing rotations.")

        only_pid = None if phase_filter < 0 else int(phase_filter)
        groups = _group_phases_by_symmetry(xmap, only_phase_id=only_pid)
        if not groups:
            raise HTTPException(status_code=400, detail="No phases with usable symmetry information.")

        n_keys = len(groups)

        def _work():
            with _mpl_lock:
                # Compact one-row layout — wider per key for legible vertex labels
                fig_w = max(2.2, 2.0 * n_keys)
                fig_h = 2.4
                fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
                # Transparent figure / axes background — frontend composites on
                # top of the map canvas.
                fig.patch.set_alpha(0.0)
                gs = fig.add_gridspec(1, 1)
                _draw_ipf_color_keys(fig, gs[0, 0], xmap, direction,
                                      color_overrides=None,
                                      orientation="horizontal",
                                      only_phase_id=only_pid)
                buf = _io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05,
                            dpi=120, transparent=True)
                plt.close(fig)
                buf.seek(0)
                return {
                    "image": base64.b64encode(buf.read()).decode('utf-8'),
                    "direction": direction,
                    "n_keys": n_keys,
                }

        return await asyncio.to_thread(_work)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to render IPF key")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-maps")
async def get_available_maps():
    """Return list of available map types including per-phase CI maps.

    Source of truth is **either** the active indexing result **or** the loaded
    analysis dataset — never both. The same render-endpoint precedence applies
    so the dropdown matches what the map actually shows:

    * If an indexing result is active (gallery entry with result_id), the
      dropdown reflects its per-phase data.
    * Otherwise fall back to the analysis dataset (a file-loaded gallery
      entry).  .ang files carry no per-phase data so only the 5 base modes
      are offered — previously the caller's per-phase-data from an earlier
      .h5 would bleed through because we OR-merged both sources.
    """
    from backend.api.routes.indexing import get_last_indexing_result
    from backend.api.routes.analysis import get_analysis_dataset
    result = get_last_indexing_result()
    dataset = get_analysis_dataset()

    maps = [
        {"id": "phase", "label": "Phase Map", "group": "Phase"},
        {"id": "Z", "label": "IPF-Z [001]", "group": "IPF"},
        {"id": "X", "label": "IPF-X [100]", "group": "IPF"},
        {"id": "Y", "label": "IPF-Y [010]", "group": "IPF"},
        {"id": "ci", "label": "CI Heatmap (Best)", "group": "Confidence"},
    ]

    # BC entry only when the active dataset actually carries it.
    # Older light files without quality-field copy don't have it, and
    # showing the menu item would let the user select a mode that errors
    # out instead of just hiding gracefully.
    has_bc = False
    src = result.xmap if result is not None else (
        getattr(dataset, "xmap", None) if dataset is not None else None
    )
    if src is not None and hasattr(src, "prop") and "bc" in src.prop:
        has_bc = True
    if has_bc:
        maps.append({"id": "bc", "label": "Band Contrast", "group": "Quality"})

    # Pick exactly one source (same precedence as _render_to_figure)
    if result is not None:
        per_phase = result.metadata.get('per_phase_data', {}) or {}
    elif dataset is not None:
        per_phase = getattr(dataset, 'per_phase_data', None) or {}
    else:
        per_phase = {}

    if per_phase:
        for name in per_phase:
            maps.append({
                "id": f"ci_{name}",
                "label": f"CI — {name}",
                "group": "Confidence",
            })
        if len(per_phase) >= 2:
            maps.append({
                "id": "uncertainty",
                "label": "Uncertainty (low = ambiguous)",
                "group": "Confidence",
            })

    return {"maps": maps}


class ApplyCleanupRequest(BaseModel):
    file_stem: Optional[str] = None
    search_dir: Optional[str] = None
    ci_threshold: float = 0.0
    uncertainty_threshold: float = 0.0
    min_cluster_size: int = 0
    fill_unindexed: bool = False
    modal_filter_size: int = 0
    # Target: "checkpoint" writes into <stem>_multiphase.h5 (requires stem);
    # "result" mutates the in-memory indexing xmap so subsequent exports see
    # the cleaned assignment; "auto" tries checkpoint first, falls back to
    # result if no checkpoint exists.
    target: str = "auto"


@router.post("/apply-cleanup")
async def apply_cleanup(req: ApplyCleanupRequest):
    """Persist a cleanup pass.

    Two targets:
      - ``checkpoint``: writes into ``<stem>_multiphase.h5`` (Batch workflow).
      - ``result``: mutates the in-memory indexing result so future /render
        calls, .ang / .h5 exports and the Analysis handoff all see the cleaned
        phase assignment. No files touched.
      - ``auto`` (default): checkpoint when a stem + matching file exist;
        result otherwise.
    """
    from pathlib import Path
    import os
    import numpy as np

    cp_path = None
    if req.target in ("auto", "checkpoint") and req.file_stem:
        candidates = []
        if req.search_dir:
            candidates.append(os.path.join(req.search_dir, f"{req.file_stem}_multiphase.h5"))
        candidates.append(os.path.join("Test_data/batch_test", f"{req.file_stem}_multiphase.h5"))
        candidates.append(os.path.join(".", f"{req.file_stem}_multiphase.h5"))
        cp_path = next((p for p in candidates if os.path.isfile(p)), None)

    # ------------------------------------------------------------------
    # Checkpoint-based cleanup (Batch Dashboard flow)
    # ------------------------------------------------------------------
    if cp_path is not None and req.target != "result":
        source_proxy = str(Path(cp_path).with_name(f"{req.file_stem}.h5oina"))
        from backend.api.services.checkpoint_writer import CheckpointWriter
        cw = CheckpointWriter(source_proxy)
        cw.checkpoint_path = cp_path

        stats = cw.apply_cleanup(
            ci_threshold=req.ci_threshold,
            uncertainty_threshold=req.uncertainty_threshold,
            min_cluster_size=req.min_cluster_size,
        )
        # The checkpoint path only accepts the original three filters for now;
        # the new fill_unindexed / modal_filter_size run in a second pass
        # directly on the cleaned_phase_id dataset.
        if req.fill_unindexed or req.modal_filter_size:
            import h5py
            with h5py.File(cp_path, "a") as f:
                aa = f.get("auto_assignment")
                if aa is not None and "cleaned_phase_id" in aa:
                    from backend.api.services.checkpoint_writer import CheckpointWriter as _CW
                    cleaned = np.array(aa["cleaned_phase_id"])
                    ci = np.array(aa["best_ci"])
                    unc = np.array(aa["uncertainty"])
                    polished = _CW.clean_phase_assignment(
                        cleaned, ci, unc,
                        ci_threshold=0, uncertainty_threshold=0,
                        min_cluster_size=0,
                        fill_unindexed=req.fill_unindexed,
                        modal_filter_size=req.modal_filter_size,
                    )
                    del aa["cleaned_phase_id"]
                    aa.create_dataset(
                        "cleaned_phase_id", data=polished, dtype=np.int16,
                        compression="gzip",
                    )
                    aa.attrs["cleanup_fill_unindexed"] = bool(req.fill_unindexed)
                    aa.attrs["cleanup_modal_filter_size"] = int(req.modal_filter_size)
                    stats["after_unindexed"] = int((polished < 0).sum())
        return {
            "checkpoint": cp_path,
            "target": "checkpoint",
            "filters": req.model_dump(),
            **stats,
        }

    # ------------------------------------------------------------------
    # In-memory result cleanup (single-file workflow) — mutate xmap.phase_id
    # ------------------------------------------------------------------
    from backend.api.routes.indexing import get_last_indexing_result
    active = get_last_indexing_result()
    if active is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No checkpoint found for '{req.file_stem}' and no in-memory "
                "result to clean. Run indexing first, or open a batch result."
            ),
        )

    xmap = active.xmap
    n_rows, n_cols = active.original_shape
    n_total = n_rows * n_cols

    phase_ids = np.asarray(xmap.phase_id).copy()
    ci_prop = np.array(xmap.prop.get("ci", np.zeros(phase_ids.size, dtype=np.float32)))
    # Uncertainty: best-vs-second over per-phase data when available
    per_phase = active.metadata.get("per_phase_data", {})
    if req.uncertainty_threshold > 0 and len(per_phase) >= 2:
        ci_stack = np.stack(
            [per_phase[n]["ci"] for n in per_phase if per_phase[n].get("ci") is not None],
            axis=0,
        )
        best_ci = np.max(ci_stack, axis=0)
        masked = ci_stack.copy()
        best_idx = np.argmax(ci_stack, axis=0)
        rows_idx, cols_idx = np.indices(best_idx.shape)
        masked[best_idx, rows_idx, cols_idx] = -np.inf
        second_ci = np.max(masked, axis=0)
        second_ci = np.where(np.isinf(second_ci), 0.0, second_ci)
        unc = (best_ci - second_ci).astype(np.float32)
    else:
        unc = np.zeros(n_total, dtype=np.float32)
        if unc.size >= n_total:
            unc = unc[:n_total].reshape(n_rows, n_cols)
        else:
            unc = np.zeros((n_rows, n_cols), dtype=np.float32)

    # Reshape phase_ids + ci for 2D cleanup
    if phase_ids.size != n_total:
        # Selection-mask-based indexing result — expand into full grid first
        mask = active.selection_mask
        full_pid = np.full(n_total, -1, dtype=np.int16)
        full_ci = np.zeros(n_total, dtype=np.float32)
        if mask is not None and not mask.all():
            full_pid[mask.ravel()] = phase_ids
            full_ci[mask.ravel()] = ci_prop[:phase_ids.size] if ci_prop.size >= phase_ids.size else ci_prop
        pid_2d = full_pid.reshape(n_rows, n_cols)
        ci_2d = full_ci.reshape(n_rows, n_cols)
    else:
        pid_2d = phase_ids.reshape(n_rows, n_cols).astype(np.int16)
        ci_2d = ci_prop.reshape(n_rows, n_cols).astype(np.float32) if ci_prop.size == n_total else np.zeros((n_rows, n_cols), dtype=np.float32)

    if unc.shape != pid_2d.shape:
        unc = np.zeros_like(ci_2d)

    from backend.api.services.checkpoint_writer import CheckpointWriter as _CW
    cleaned = _CW.clean_phase_assignment(
        pid_2d, ci_2d, unc,
        ci_threshold=req.ci_threshold,
        uncertainty_threshold=req.uncertainty_threshold,
        min_cluster_size=req.min_cluster_size,
        fill_unindexed=req.fill_unindexed,
        modal_filter_size=req.modal_filter_size,
    )

    before = int((pid_2d >= 0).sum())
    after_unindexed = int((cleaned < 0).sum())
    moved = int(((pid_2d != cleaned) & (cleaned >= 0)).sum())

    # Stash the cleaned phase_id map + the exact filters applied onto the
    # result's metadata. Render & downstream (Analysis handoff) can read
    # metadata["cleaned_phase_id"] for a cleaned view without mutating orix
    # internals (which would require rebuilding the CrystalMap).
    active.metadata["cleaned_phase_id"] = cleaned
    active.metadata["applied_cleanup"] = {
        "ci_threshold": req.ci_threshold,
        "uncertainty_threshold": req.uncertainty_threshold,
        "min_cluster_size": req.min_cluster_size,
        "fill_unindexed": req.fill_unindexed,
        "modal_filter_size": req.modal_filter_size,
    }

    # Best-effort: if the xmap exposes a mutable phase_id (orix >= 0.11
    # sometimes does), write the cleaned values back so .ang export picks it
    # up. Catch all failures — orix internals vary by version.
    try:
        if cleaned.size == xmap.phase_id.size:
            xmap.phase_id = cleaned.ravel()
    except Exception as e:
        logger.debug("xmap.phase_id mutation refused, metadata override in use: %s", e)

    # Cleanup changed how the active result renders → notify polling clients.
    state_version.bump()

    return {
        "target": "result",
        "filters": req.model_dump(),
        "before": before,
        "after_unindexed": after_unindexed,
        "moved": moved,
    }


@router.post("/export")
async def export_phase_map(req: ExportRequest):
    """Export the phase map to file, matching the current live view.

    Accepts every render parameter so the exported PNG/SVG/PDF equals what
    the user sees in the preview — including current display mode, cleanup,
    scalebar, confidence overlay and title.
    """
    fmt = (req.format or "png").lower().lstrip(".")
    if fmt not in {"png", "svg", "pdf"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    try:
        from pathlib import Path
        out_path = Path(req.output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.suffix or out_path.suffix.lstrip(".").lower() != fmt:
            out_path = out_path.with_suffix(f".{fmt}")

        def _work():
            import matplotlib.pyplot as plt
            with _mpl_lock:
                fig, meta = _render_to_figure(
                    direction=req.direction,
                    show_legend=req.show_legend,
                    title=req.title,
                    step_x=req.step_x, step_y=req.step_y,
                    scalebar_enabled=req.scalebar_enabled,
                    scalebar_location=req.scalebar_location,
                    scalebar_length=req.scalebar_length,
                    scalebar_font_size=req.scalebar_font_size,
                    scalebar_bar_color=req.scalebar_bar_color,
                    scalebar_box_color=req.scalebar_box_color,
                    scalebar_box_alpha=req.scalebar_box_alpha,
                    confidence_overlay=req.confidence_overlay,
                    confidence_alpha=req.confidence_alpha,
                    confidence_cmap=req.confidence_cmap,
                    ci_threshold=req.ci_threshold,
                    uncertainty_threshold=req.uncertainty_threshold,
                    min_cluster_size=req.min_cluster_size,
                    fill_unindexed=req.fill_unindexed,
                    modal_filter_size=req.modal_filter_size,
                    bc_modulate=req.bc_modulate,
                    dpi=req.dpi,
                )
                fig.savefig(
                    str(out_path),
                    format=fmt,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    dpi=req.dpi,
                    facecolor=fig.get_facecolor(),
                )
                plt.close(fig)
                return {"success": True, "path": str(out_path), **meta}

        return await asyncio.to_thread(_work)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to export phase map")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phase-legend")
async def get_phase_legend(color_overrides: str = ""):
    """Return the phase id -> name -> colour mapping for the active result.

    The phase-map LAYER (``/layer?kind=phase``) deliberately bakes in no
    legend — layers are transparent-bg composites. This endpoint exposes the
    SAME colour assignment (:func:`build_phase_color_map`) so the frontend can
    render a separate legend panel that always matches the displayed map.

    Response::

        {
          "phases": [
            {"phase_id": 0, "phase_name": "Al", "color_hex": "#rrggbb"},
            ...
          ],
          "unindexed": {"phase_name": "Not indexed", "color_hex": "#44475a"}
        }

    Raises HTTPException(400) when no indexing result / analysis dataset.
    """
    try:
        import json as _json

        overrides_dict = None
        if color_overrides:
            try:
                parsed = _json.loads(color_overrides)
                if isinstance(parsed, dict):
                    overrides_dict = {str(k): str(v) for k, v in parsed.items()}
            except (ValueError, TypeError):
                logger.debug("Ignoring malformed color_overrides: %r", color_overrides)

        result = get_last_indexing_result()
        dataset = get_analysis_dataset()
        if result is None and dataset is None:
            raise HTTPException(
                status_code=400,
                detail="No indexing result or analysis dataset available. Load data first.",
            )
        xmap = result.xmap if result is not None else dataset.xmap

        color_map = build_phase_color_map(xmap, overrides_dict)

        def _to_hex(rgb):
            r, g, b = (int(round(c * 255)) for c in rgb)
            return f"#{r:02x}{g:02x}{b:02x}"

        phases = []
        for pid in sorted(color_map.keys()):
            try:
                name = xmap.phases[pid].name
            except (KeyError, IndexError, TypeError, AttributeError):
                name = None
            phases.append({
                "phase_id": int(pid),
                "phase_name": str(name) if name else f"Phase {pid}",
                "color_hex": _to_hex(color_map[pid]),
            })
        return {
            "phases": phases,
            "unindexed": {
                "phase_name": "Not indexed",
                "color_hex": _to_hex(UNINDEXED_COLOR),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to build phase legend")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phase-stats")
async def get_phase_stats(color_overrides: str = "", include_empty: bool = False):
    """Return per-phase statistics for the active indexing result.

    By default counts only phases that are actually PRESENT in the indexed
    xmap (pixel_count >= 1). Pass ``include_empty=true`` to ALSO include
    phases declared in the PhaseList that ended up with zero pixels (with
    pixel_count=0, area_pct=0, CI=None) — useful when the user wants to
    audit "which of my input phases didn't win anywhere".

    Returns counts, area %, and CI percentiles per phase, plus a
    ``suspect`` flag that is True when a phase has significant area
    coverage but low confidence — a heuristic that flags candidate
    misindex regions for the user to investigate.

    Response::

        {
          "phases": [
            {"phase_id": 1, "name": "Al", "color_hex": "#80ff66",
             "pixel_count": 12453, "area_pct": 49.4,
             "median_ci": 0.71, "p25_ci": 0.62, "p75_ci": 0.78,
             "suspect": false},
            ...
          ],
          "total_indexed": 25230,
          "unindexed_count": 0
        }

    Suspect heuristic: ``median_ci < 0.5 AND area_pct > 5%``.

    Used by the Phase Map page's Phase Legend panel — replaces the
    older ``/phase-legend`` which listed every PhaseList entry
    regardless of whether it was actually indexed.
    """
    try:
        import json as _json
        import numpy as np

        overrides_dict = None
        if color_overrides:
            try:
                parsed = _json.loads(color_overrides)
                if isinstance(parsed, dict):
                    overrides_dict = {str(k): str(v) for k, v in parsed.items()}
            except (ValueError, TypeError):
                logger.debug("Ignoring malformed color_overrides: %r", color_overrides)

        result = get_last_indexing_result()
        dataset = get_analysis_dataset()
        if result is None and dataset is None:
            raise HTTPException(
                status_code=400,
                detail="No indexing result or analysis dataset available. Load data first.",
            )
        xmap = result.xmap if result is not None else dataset.xmap

        # Use the SAME colour assignment as the rendered map.
        color_map = build_phase_color_map(xmap, overrides_dict)

        def _to_hex(rgb):
            r, g, b = (int(round(c * 255)) for c in rgb)
            return f"#{r:02x}{g:02x}{b:02x}"

        phase_id_arr = np.asarray(xmap.phase_id).ravel()
        # Pull CI from the result's confidence_scores (consensus row for
        # multi-phase runs) or from xmap.prop['ci'] if present.
        ci_arr = None
        if result is not None and getattr(result, "confidence_scores", None) is not None:
            ci_raw = np.asarray(result.confidence_scores)
            if ci_raw.ndim >= 2:
                ci_arr = ci_raw[-1].ravel()
            else:
                ci_arr = ci_raw.ravel()
        if ci_arr is None:
            for key in ("ci", "scores"):
                if hasattr(xmap, "prop") and key in xmap.prop:
                    ci_arr = np.asarray(xmap.prop[key]).ravel()
                    break

        # Count indexed pixels (phase_id >= 0). xmap may carry sentinel
        # "not_indexed" entries at -1; everything non-negative counts.
        indexed_mask = phase_id_arr >= 0
        total_indexed = int(indexed_mask.sum())
        unindexed_count = int((~indexed_mask).sum())

        phases_out = []
        for pid in sorted(color_map.keys()):
            mask = phase_id_arr == pid
            count = int(mask.sum())
            if count <= 0 and not include_empty:
                continue  # default: filter out empty phases (the whole point)
            area_pct = (count / max(total_indexed, 1)) * 100.0
            median_ci = p25_ci = p75_ci = None
            if ci_arr is not None and ci_arr.shape[0] == phase_id_arr.shape[0]:
                ci_phase = ci_arr[mask]
                ci_phase = ci_phase[np.isfinite(ci_phase)]
                if ci_phase.size > 0:
                    median_ci = float(np.median(ci_phase))
                    p25_ci = float(np.percentile(ci_phase, 25))
                    p75_ci = float(np.percentile(ci_phase, 75))

            try:
                name = xmap.phases[pid].name
            except (KeyError, IndexError, TypeError, AttributeError):
                name = None

            suspect = (
                median_ci is not None
                and median_ci < 0.5
                and area_pct > 5.0
            )

            phases_out.append({
                "phase_id": int(pid),
                "name": str(name) if name else f"Phase {pid}",
                "color_hex": _to_hex(color_map[pid]),
                "pixel_count": count,
                "area_pct": round(area_pct, 2),
                "median_ci": (round(median_ci, 4) if median_ci is not None else None),
                "p25_ci": (round(p25_ci, 4) if p25_ci is not None else None),
                "p75_ci": (round(p75_ci, 4) if p75_ci is not None else None),
                "suspect": bool(suspect),
            })

        # Sort dominant phases first
        phases_out.sort(key=lambda x: -x["area_pct"])

        return {
            "phases": phases_out,
            "total_indexed": total_indexed,
            "unindexed_count": unindexed_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to compute phase stats")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phase-adjacency")
async def get_phase_adjacency():
    """Return a phase-pair adjacency (confusion) matrix for the active xmap.

    For every 4-neighbour pixel pair where both ends are indexed AND the
    phases differ, increment ``adjacency[a][b]`` and ``adjacency[b][a]``.
    The resulting matrix reveals which phase pairs share borders most
    often — high counts on a chemically-similar pair (e.g. Al4FeSi ↔
    Al3Fe2Si) suggest cubic-lattice degeneracy + likely misindex, not a
    real phase interface.

    Frontend (``PhaseAdjacencyPanel``) renders this as a heatmap with
    hover-tooltips. Diagonal is always 0 (same-phase neighbours don't
    count as a border).

    Response::

        {
          "phases": [{"id": 1, "name": "Al"}, ...],
          "adjacency": [[0, 234, 12], [234, 0, 567], [12, 567, 0]],
          "border_total_per_phase": [...],
          "total_borders": 1626
        }

    ``border_total_per_phase[i]`` is the sum of row i = how many
    inter-phase borders this phase participates in. The frontend uses
    it to normalise per-row tooltips ("X% of Al4FeSi borders").
    """
    try:
        import numpy as np

        result = get_last_indexing_result()
        dataset = get_analysis_dataset()
        if result is None and dataset is None:
            raise HTTPException(
                status_code=400,
                detail="No indexing result or analysis dataset available. Load data first.",
            )
        xmap = result.xmap if result is not None else dataset.xmap

        # Phase grid (n_rows, n_cols). Sentinel -1 marks not-indexed.
        # For SPARSE / region-cropped results, ``xmap.phase_id`` only
        # contains the indexed pixels — we scatter them back into the
        # full (n_rows, n_cols) grid via ``result.selection_mask`` so
        # adjacency math works on any indexing run, not only full-grid.
        # (Bug 2026-05-26: a 1534-pixel region result on a 174×145 grid
        # was raising "cannot reshape array of size 1534 into shape
        # (174, 145)" before this fix.)
        n_rows, n_cols = result.original_shape if result is not None else (
            int(getattr(xmap, "shape", (1, 1))[0]),
            int(getattr(xmap, "shape", (1, 1))[1]),
        )
        phase_id_flat_xmap = np.asarray(xmap.phase_id).ravel()
        full_size = int(n_rows) * int(n_cols)
        if phase_id_flat_xmap.size == full_size:
            phase_id_arr = phase_id_flat_xmap.reshape(n_rows, n_cols)
        else:
            phase_id_arr = np.full((n_rows, n_cols), -1, dtype=np.int64)
            sel = None
            try:
                sel = np.asarray(result.selection_mask, dtype=bool).reshape(n_rows, n_cols)
            except Exception:
                sel = None
            if sel is not None and sel.sum() == phase_id_flat_xmap.size:
                phase_id_arr[sel] = phase_id_flat_xmap
            else:
                # Last resort: fill the FIRST N pixels in row-major order so
                # we don't 500. Adjacency count will be approximate but the
                # endpoint stays alive.
                n_fit = min(phase_id_flat_xmap.size, full_size)
                phase_id_arr.flat[:n_fit] = phase_id_flat_xmap[:n_fit]

        # Phase metadata in stable order (sorted by id), filtered to the
        # phases that actually appear in the grid.
        pids_present = sorted(set(int(p) for p in np.unique(phase_id_arr) if int(p) >= 0))
        phases_out = []
        for pid in pids_present:
            try:
                name = xmap.phases[pid].name
            except Exception:
                name = f"Phase {pid}"
            phases_out.append({"id": int(pid), "name": str(name)})

        pid_to_idx = {pid: i for i, pid in enumerate(pids_present)}
        n = len(pids_present)
        adj = np.zeros((n, n), dtype=np.int64)

        if n >= 2:
            # 4-neighbour pairs: horizontal (left, right) and vertical (up, down).
            # Both ends must be indexed (phase_id >= 0). Phase != phase to count.
            left  = phase_id_arr[:, :-1]
            right = phase_id_arr[:,  1:]
            up    = phase_id_arr[:-1, :]
            down  = phase_id_arr[ 1:, :]

            def _accumulate(a, b):
                mask = (a >= 0) & (b >= 0) & (a != b)
                a_f = a[mask]
                b_f = b[mask]
                # Map to compact indices [0, n)
                # NB: not every pid in pids_present is guaranteed by np.unique
                # to be a key here, but pids_present came FROM np.unique so
                # every pid is a key in pid_to_idx.
                # Use np.vectorize-style mapping via searchsorted on sorted pids.
                pid_arr = np.asarray(pids_present, dtype=np.int64)
                ai = np.searchsorted(pid_arr, a_f)
                bi = np.searchsorted(pid_arr, b_f)
                # Increment symmetric entries via add.at
                np.add.at(adj, (ai, bi), 1)
                np.add.at(adj, (bi, ai), 1)

            _accumulate(left, right)
            _accumulate(up, down)

        border_total_per_phase = adj.sum(axis=1).astype(int).tolist()
        total_borders = int(adj.sum() // 2)  # each border counted twice (i,j)+(j,i)

        return {
            "phases": phases_out,
            "adjacency": adj.astype(int).tolist(),
            "border_total_per_phase": border_total_per_phase,
            "total_borders": total_borders,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to compute phase adjacency")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-directions")
async def available_directions():
    """List available IPF directions."""
    return {
        "directions": [
            {"id": "X", "label": "IPF-X (RD)"},
            {"id": "Y", "label": "IPF-Y (TD)"},
            {"id": "Z", "label": "IPF-Z (ND)"},
        ]
    }


# ---------------------------------------------------------------------------
# Tier-2 features (PhaseMap port) — probe, region-stats, linescan
# ---------------------------------------------------------------------------
#
# These three endpoints expose the same scalar layers as the Phase Map render
# pipeline (BC, CI, per-phase CI, uncertainty, KAM, GOS) but as raw numbers, so
# the React frontend can show hover tooltips, region statistics, and line-scan
# profiles without round-tripping a PNG. They all share a private helper that
# pulls a (n_rows, n_cols) array for a known scalar layer; that helper never
# raises — it returns None on any failure so the JSON response stays valid.
#


def _phase_probe_from_xmap(xmap, row: int, col: int) -> Optional[dict]:
    """Return ``{id, name}`` for the phase at (row, col) in the active xmap, or None.

    Mirrors the behaviour of ``backend.api.routes.eds._probe_phase_at_safe`` but
    reads the indexing result's xmap instead of the phase_map_store. ``id < 0``
    encodes "unindexed" and returns None. Wrapped in try/except so a malformed
    xmap can never crash a probe call.
    """
    try:
        result = get_last_indexing_result()
        if result is None:
            return None
        n_rows, n_cols = result.original_shape
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            return None
        # Build the (n_rows, n_cols) phase-id grid the same way
        # _compute_effective_phase_ids_2d does.
        import numpy as np
        raw_ids = np.asarray(xmap.phase_id)
        if raw_ids.size == n_rows * n_cols:
            pid_2d = raw_ids.reshape(n_rows, n_cols).astype(np.int32)
        else:
            pid_2d = np.full((n_rows, n_cols), -1, dtype=np.int32)
            mask = result.selection_mask
            if mask is not None and not mask.all() and raw_ids.size == int(mask.sum()):
                pid_2d[mask] = raw_ids
            else:
                pid_2d.flat[:raw_ids.size] = raw_ids
        pid = int(pid_2d[row, col])
        if pid < 0:
            return None
        try:
            phase_name = xmap.phases[pid].name
        except (KeyError, IndexError, TypeError, AttributeError):
            phase_name = f"Phase {pid}"
        if _is_unindexed_phase(phase_name):
            return None
        return {"id": pid, "name": str(phase_name)}
    except Exception:
        return None


def _get_scalar_array_2d(name: str) -> "Optional['np.ndarray']":
    """Return a (n_rows, n_cols) numpy array for a known scalar layer, or None.

    Supported names: ``ci``, ``bc``, ``kam``, ``gos``, ``uncertainty``, and
    ``ci_<phase_id>`` (where ``<phase_id>`` is either the integer id or the
    phase name — both encodings are accepted, matching what _extract_ci_array
    already does for the per-phase CI layer).

    Returns ``None`` on any failure: no active indexing result, missing array,
    wrong shape, or unknown name. Callers (probe, region-stats, linescan) all
    handle ``None`` gracefully, so this helper must never raise.

    Implementation note: KAM and GOS aren't stored on the indexing result —
    they're computed by the Analysis module from the analysis dataset. When
    the analysis dataset has populated them on its xmap.prop, we read them
    from there; otherwise return None and let the frontend show "no data".
    """
    import numpy as np
    try:
        result = get_last_indexing_result()
        dataset = get_analysis_dataset()
        if result is None and dataset is None:
            return None

        if result is not None:
            xmap = result.xmap
            n_rows, n_cols = result.original_shape
            mask = result.selection_mask
        else:
            xmap = dataset.xmap
            n_rows, n_cols = dataset.shape
            mask = None

        def _reshape_to_grid(flat) -> "Optional[np.ndarray]":
            """Place a 1-D xmap-property array into a (n_rows, n_cols) grid."""
            flat = np.asarray(flat).astype(float).ravel()
            if flat.size == 0:
                return None
            if flat.size == n_rows * n_cols:
                return flat.reshape(n_rows, n_cols)
            out = np.full((n_rows, n_cols), np.nan, dtype=float)
            if mask is not None and not mask.all() and flat.size == int(mask.sum()):
                out[mask] = flat
                return out
            # Last-ditch: pad/truncate so we still return *something* the
            # frontend can render with NaN for missing pixels.
            if flat.size < n_rows * n_cols:
                out.flat[: flat.size] = flat
                return out
            return None

        if name == "ci":
            return _extract_ci_array(
                kind="ci", xmap=xmap, result=result, dataset=dataset,
                n_rows=n_rows, n_cols=n_cols, mask=mask,
            )

        if name == "uncertainty":
            try:
                return _extract_ci_array(
                    kind="uncertainty", xmap=xmap, result=result, dataset=dataset,
                    n_rows=n_rows, n_cols=n_cols, mask=mask,
                )
            except HTTPException:
                # Single-phase indexing → no uncertainty layer; treat as missing.
                return None

        if name.startswith("ci_"):
            # _extract_ci_array accepts the phase NAME after "ci_"; for
            # numeric ids translate via xmap.phases.
            suffix = name[3:]
            try:
                return _extract_ci_array(
                    kind=name, xmap=xmap, result=result, dataset=dataset,
                    n_rows=n_rows, n_cols=n_cols, mask=mask,
                )
            except HTTPException:
                # The suffix may be a phase ID like "ci_1". Translate to name.
                try:
                    phase_name = xmap.phases[int(suffix)].name
                    return _extract_ci_array(
                        kind=f"ci_{phase_name}", xmap=xmap, result=result,
                        dataset=dataset, n_rows=n_rows, n_cols=n_cols, mask=mask,
                    )
                except (ValueError, HTTPException, KeyError, IndexError,
                        TypeError, AttributeError):
                    return None

        if name == "bc":
            bc_factor = _bc_brightness_factor(xmap, n_rows, n_cols, mask)
            if bc_factor is not None:
                return bc_factor.astype(float)
            # Fall back to the raw H5OINA band-contrast grid the EBSD viewer
            # exposes — same source `_compute_layer_rgba` uses for BC.
            try:
                from backend.api.routes.virtual_images import _read_native_band_contrast
                native_bc = _read_native_band_contrast()
            except Exception:
                native_bc = None
            if native_bc is not None and native_bc.shape == (n_rows, n_cols):
                return native_bc.astype(float)
            return None

        if name in ("kam", "gos"):
            # Per-pixel KAM / GOS live on the analysis dataset's xmap.prop
            # (populated when the user runs the analysis pipeline). The bare
            # indexing result doesn't carry them, so we return None and the
            # frontend renders "no data".
            for source in (dataset, result):
                if source is None:
                    continue
                src_xmap = getattr(source, "xmap", None) if source is dataset else getattr(source, "xmap", None)
                if src_xmap is None or not hasattr(src_xmap, "prop"):
                    continue
                try:
                    arr = src_xmap.prop.get(name)
                except Exception:
                    arr = None
                if arr is None:
                    continue
                grid = _reshape_to_grid(arr)
                if grid is not None:
                    return grid
            return None

        return None
    except Exception:
        logger.debug("_get_scalar_array_2d(%r) failed", name, exc_info=True)
        return None


# --- Probe -----------------------------------------------------------------


class PhaseMapProbeRequest(BaseModel):
    row: int
    col: int


@router.post("/probe")
async def phasemap_probe(req: PhaseMapProbeRequest):
    """Return phase + all known scalar values at one pixel.

    Response shape::

        {
          "row": int, "col": int,
          "phase": { "id": int, "name": str } | null,
          "scalars": {
            "ci":   float | null,
            "bc":   float | null,
            "kam":  float | null,
            "gos":  float | null,
            "uncertainty": float | null,
            "ci_<phase_id>": float | null    # only when per-phase CI exists
          }
        }

    Returns 400 when no indexing result is loaded. Returns 422 when the pixel
    is outside the scan grid. Every scalar field that can't be computed (no
    array, NaN value, etc.) is returned as ``null`` so the tooltip can render
    a "-" placeholder without an extra roundtrip.
    """
    import numpy as np

    result = get_last_indexing_result()
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available.")

    n_rows, n_cols = result.original_shape
    if not (0 <= req.row < n_rows and 0 <= req.col < n_cols):
        raise HTTPException(
            status_code=422,
            detail=f"pixel ({req.row}, {req.col}) outside scan {n_rows}x{n_cols}",
        )

    # Phase --------------------------------------------------------------
    phase_info = _phase_probe_from_xmap(result.xmap, req.row, req.col)

    # Scalars ------------------------------------------------------------
    scalar_names: List[str] = ["ci", "bc", "kam", "gos", "uncertainty"]

    # Add per-phase CI keys when the indexing run produced them — preserves
    # the API contract that ci_<phase_id> appears only when meaningful.
    per_phase: Dict[str, object] = {}
    try:
        per_phase = (result.metadata or {}).get("per_phase_data", {}) or {}
    except Exception:
        per_phase = {}
    for phase_name in per_phase.keys():
        scalar_names.append(f"ci_{phase_name}")

    scalars: Dict[str, Optional[float]] = {}
    for name in scalar_names:
        try:
            arr = _get_scalar_array_2d(name)
            if arr is None:
                scalars[name] = None
                continue
            value = float(arr[req.row, req.col])
            if not np.isfinite(value):
                scalars[name] = None
            else:
                scalars[name] = value
        except Exception:
            scalars[name] = None

    return {
        "row": req.row,
        "col": req.col,
        "phase": phase_info,
        "scalars": scalars,
    }


# --- Region Statistics ------------------------------------------------------


class PhaseMapRegionStatsRequest(BaseModel):
    row_start: int
    row_end: int
    col_start: int
    col_end: int


@router.post("/region-stats")
async def phasemap_region_stats(req: PhaseMapRegionStatsRequest):
    """Summarise phase fractions + scalar mean/std over a rectangular ROI.

    Response::

        {
          "row_start": int, "row_end": int, "col_start": int, "col_end": int,
          "n_pixels": int,
          "phases": [
            { "id": int, "name": str, "count": int, "fraction": float }
          ],   # sorted by fraction DESC; id=-1 → "unindexed"
          "scalars": {
            "ci":  { "mean": float, "std": float } | null,
            ...
          }
        }

    Reversed-drag rectangles (row_end < row_start, etc.) are normalised. A
    scalar with all-NaN pixels in the rectangle returns ``null``.
    """
    import numpy as np

    result = get_last_indexing_result()
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available.")

    n_rows, n_cols = result.original_shape

    # Normalise rectangle — allow drag from any corner.
    r0 = max(0, min(int(req.row_start), int(req.row_end)))
    r1 = min(n_rows - 1, max(int(req.row_start), int(req.row_end)))
    c0 = max(0, min(int(req.col_start), int(req.col_end)))
    c1 = min(n_cols - 1, max(int(req.col_start), int(req.col_end)))
    if r0 > r1 or c0 > c1:
        raise HTTPException(
            status_code=422,
            detail=f"rectangle ({req.row_start}, {req.row_end}, {req.col_start}, {req.col_end}) "
                   f"is empty after clipping to scan {n_rows}x{n_cols}",
        )

    # Phase histogram -----------------------------------------------------
    xmap = result.xmap
    raw_ids = np.asarray(xmap.phase_id)
    if raw_ids.size == n_rows * n_cols:
        pid_2d = raw_ids.reshape(n_rows, n_cols).astype(np.int32)
    else:
        pid_2d = np.full((n_rows, n_cols), -1, dtype=np.int32)
        mask = result.selection_mask
        if mask is not None and not mask.all() and raw_ids.size == int(mask.sum()):
            pid_2d[mask] = raw_ids
        else:
            pid_2d.flat[:raw_ids.size] = raw_ids

    sub = pid_2d[r0:r1 + 1, c0:c1 + 1]
    n_pixels = int(sub.size)

    ids, counts = np.unique(sub, return_counts=True)
    phases_list: List[Dict[str, object]] = []
    for pid, cnt in zip(ids.tolist(), counts.tolist()):
        pid = int(pid)
        if pid < 0:
            name = "unindexed"
        else:
            try:
                name = str(xmap.phases[pid].name)
            except (KeyError, IndexError, TypeError, AttributeError):
                name = f"Phase {pid}"
            if _is_unindexed_phase(name):
                name = "unindexed"
        phases_list.append({
            "id": pid,
            "name": name,
            "count": int(cnt),
            "fraction": float(cnt) / float(n_pixels) if n_pixels else 0.0,
        })
    phases_list.sort(key=lambda e: e["fraction"], reverse=True)

    # Scalar mean / std ---------------------------------------------------
    scalar_names: List[str] = ["ci", "bc", "kam", "gos", "uncertainty"]
    try:
        per_phase = (result.metadata or {}).get("per_phase_data", {}) or {}
    except Exception:
        per_phase = {}
    for phase_name in per_phase.keys():
        scalar_names.append(f"ci_{phase_name}")

    scalars: Dict[str, Optional[Dict[str, float]]] = {}
    for name in scalar_names:
        try:
            arr = _get_scalar_array_2d(name)
            if arr is None:
                scalars[name] = None
                continue
            sub_arr = arr[r0:r1 + 1, c0:c1 + 1]
            valid = np.isfinite(sub_arr)
            if not valid.any():
                scalars[name] = None
                continue
            mean = float(np.nanmean(sub_arr))
            std = float(np.nanstd(sub_arr))
            scalars[name] = {"mean": mean, "std": std}
        except Exception:
            scalars[name] = None

    return {
        "row_start": r0,
        "row_end": r1,
        "col_start": c0,
        "col_end": c1,
        "n_pixels": n_pixels,
        "phases": phases_list,
        "scalars": scalars,
    }


# --- Linescan ---------------------------------------------------------------


class PhaseMapLinescanRequest(BaseModel):
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    n_samples: int = 128
    layers: List[str]


@router.post("/linescan")
async def phasemap_linescan(req: PhaseMapLinescanRequest):
    """Sample requested scalar layers along a line from (start) → (end).

    Response::

        { "samples": int, "series": { layer_id: [number | null, ...] } }

    Scalar layers allowed: ``bc``, ``ci``, ``kam``, ``gos``, ``uncertainty``,
    and any ``ci_<phase_id>``. Disallowed layers (``phase``, ``ipf-x/y/z``,
    unknown ids) return a ``[None, ..., None]`` array so the Profile-Plot can
    render a "no data" gap without an extra round-trip.

    Sample count is clamped to ``[2, 512]`` to keep the response small. Uses
    bilinear ``scipy.ndimage.map_coordinates`` when available; falls back to
    nearest-neighbour otherwise.
    """
    import numpy as np

    result = get_last_indexing_result()
    if result is None:
        raise HTTPException(status_code=400, detail="No indexing result available.")

    n = max(2, min(512, int(req.n_samples)))

    if not req.layers:
        return {"samples": n, "series": {}}

    rs = np.linspace(req.start_row, req.end_row, n)
    cs = np.linspace(req.start_col, req.end_col, n)

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

    # Layers we know how to sample. Anything outside this set (including
    # phase, ipf-x/y/z, eds-*, vbse, electron-*) falls through to the null
    # branch so callers can still render the line.
    allowed_scalar_set = {"bc", "ci", "kam", "gos", "uncertainty"}

    series: Dict[str, List[Optional[float]]] = {}
    for lid in req.layers:
        if lid in allowed_scalar_set or lid.startswith("ci_"):
            arr = _get_scalar_array_2d(lid)
            if arr is None:
                series[lid] = [None] * n
                continue
            try:
                values = sample(arr.astype(float))
                # Replace non-finite values (NaN/inf from map_coordinates near
                # the edge of an all-NaN region) with None for clean JSON.
                series[lid] = [
                    (float(v) if np.isfinite(v) else None) for v in values
                ]
            except Exception:
                series[lid] = [None] * n
        else:
            series[lid] = [None] * n

    return {"samples": n, "series": series}
