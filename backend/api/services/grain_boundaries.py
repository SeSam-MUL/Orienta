"""Grain boundaries as a drawable overlay.

Two jobs, deliberately separated:

  ``boundary_angles``  — the physics. Misorientation between each pair of
                         neighbouring indexed pixels, in degrees, under the
                         crystal symmetry. Expensive, so it is cached per
                         indexing result: the angles do not depend on how the
                         user wants them drawn.

  ``render_boundaries`` — the drawing. Turns those angles into an RGBA overlay
                         given the user's bands (angle range, colour, width).
                         Cheap, and re-run on every settings change.

Boundaries are drawn on the EDGES between pixels, not on the pixels
themselves. A line down the middle of a pixel row would sit half a step off
the actual interface, which at 0.2 µm steps is a visible lie about where a
grain ends. The overlay is therefore rendered supersampled (SS pixels per scan
step) so an edge has a place to live and a width to grow into.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Scan-step subdivisions in the overlay. 4 gives quarter-step line widths,
# which is enough for print, and keeps a 500x500 map at 2000x2000 = 16 MB RGBA.
SS = 4

# Pairs whose misorientation is below this are not boundaries by any reading —
# it is measurement noise. Keeping them out shrinks the drawing work.
MIN_ANGLE_DEG = 0.5


def _phase_symmetries(xmap) -> dict[int, Any]:
    """Point group per phase id, for the symmetry-reduced misorientation."""
    out: dict[int, Any] = {}
    phases = getattr(xmap, "phases", None)
    if phases is None:
        return out
    for pid in getattr(phases, "ids", []):
        try:
            phase = phases[pid]
            pg = getattr(phase, "point_group", None)
            if pg is not None:
                out[int(pid)] = pg
        except Exception:  # noqa: BLE001 — a phase without a point group is skipped
            continue
    return out


def boundary_angles(xmap, n_rows: int, n_cols: int, mask=None) -> dict[str, np.ndarray]:
    """Misorientation angle (degrees) across every horizontal and vertical edge.

    Returns ``{"h": (n_rows, n_cols-1), "v": (n_rows-1, n_cols)}``. NaN marks a
    pair that cannot be compared: an unindexed pixel, or two DIFFERENT phases —
    a misorientation between different crystals is not a grain boundary angle,
    and reporting one would be meaningless. Those edges are still interfaces,
    which the caller may draw separately; here they are simply not angles.
    """
    from orix.quaternion import Misorientation, Orientation

    rot = getattr(xmap, "rotations", None)
    if rot is None or np.asarray(getattr(rot, "data", [])).size == 0:
        raise ValueError("The result carries no orientations, so it has no grain boundaries.")

    n_px = n_rows * n_cols
    quats = np.asarray(rot.data, dtype=np.float64).reshape(-1, 4)

    # Results indexed on a sub-region carry only the selected pixels; put them
    # back where they belong before looking at neighbours.
    full = np.full((n_px, 4), np.nan, dtype=np.float64)
    pid_flat = np.full(n_px, -1, dtype=np.int64)
    phase_id = np.asarray(getattr(xmap, "phase_id", np.zeros(len(quats))), dtype=np.int64).ravel()
    if mask is not None:
        idx = np.flatnonzero(np.asarray(mask).ravel())
        take = min(len(idx), len(quats))
        full[idx[:take]] = quats[:take]
        pid_flat[idx[:take]] = phase_id[:take] if len(phase_id) >= take else -1
    else:
        take = min(n_px, len(quats))
        full[:take] = quats[:take]
        pid_flat[:take] = phase_id[:take] if len(phase_id) >= take else -1

    q2d = full.reshape(n_rows, n_cols, 4)
    pid2d = pid_flat.reshape(n_rows, n_cols)
    indexed = np.isfinite(q2d).all(axis=-1) & (pid2d >= 0)

    syms = _phase_symmetries(xmap)

    def _edge_angles(qa, qb, pa, pb, ok):
        """Degrees for the selected pairs; NaN elsewhere."""
        out = np.full(ok.shape, np.nan, dtype=np.float32)
        if not ok.any():
            return out
        # Same-phase only — see the docstring.
        same = ok & (pa == pb)
        for pid in np.unique(pa[same]) if same.any() else []:
            sel = same & (pa == int(pid))
            if not sel.any():
                continue
            sym = syms.get(int(pid))
            try:
                o_a = Orientation(qa[sel], symmetry=sym) if sym is not None else Orientation(qa[sel])
                o_b = Orientation(qb[sel], symmetry=sym) if sym is not None else Orientation(qb[sel])
                mori = Misorientation(o_b * (~o_a), symmetry=(sym, sym) if sym is not None else None)
                if sym is not None:
                    mori = mori.map_into_symmetry_reduced_zone()
                ang = np.rad2deg(np.atleast_1d(np.asarray(mori.angle)).ravel())
                out[sel] = ang.astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                # Fail loud in the log, transparent on the map: an edge whose
                # angle could not be computed must not be drawn as if it had one.
                logger.warning("Misorientation failed for phase %s: %s", pid, exc)
        return out

    ok_h = indexed[:, :-1] & indexed[:, 1:]
    ang_h = _edge_angles(q2d[:, :-1], q2d[:, 1:], pid2d[:, :-1], pid2d[:, 1:], ok_h)

    ok_v = indexed[:-1, :] & indexed[1:, :]
    ang_v = _edge_angles(q2d[:-1, :], q2d[1:, :], pid2d[:-1, :], pid2d[1:, :], ok_v)

    return {"h": ang_h, "v": ang_v}


# The angles depend only on the data, so they are computed once per result and
# reused while the user tries colours, widths and cut points.
_ANGLE_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def cached_boundary_angles(xmap, n_rows: int, n_cols: int, mask=None, result=None):
    key = (
        getattr(result, "result_id", None) or id(xmap),
        int(n_rows), int(n_cols),
        int(np.count_nonzero(mask)) if mask is not None else -1,
    )
    hit = _ANGLE_CACHE.get(key)
    if hit is not None:
        return hit
    angles = boundary_angles(xmap, n_rows, n_cols, mask)
    # One result at a time is all anybody looks at; a bigger cache would just
    # hold megabytes of a map nobody is on any more.
    _ANGLE_CACHE.clear()
    _ANGLE_CACHE[key] = angles
    return angles


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    s = str(value or "").lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return (255, 255, 255)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (255, 255, 255)


def render_boundaries(angles: dict[str, np.ndarray], n_rows: int, n_cols: int,
                      bands: list[dict]) -> np.ndarray:
    """Draw the bands onto a transparent RGBA overlay, supersampled by `SS`.

    ``bands`` is drawn in the given order, so a later band paints over an
    earlier one where they overlap. Each band is
    ``{"min": deg, "max": deg|None, "color": "#rrggbb", "width": px, "on": bool}``
    with ``width`` in overlay pixels (1 = a quarter scan step at SS=4).
    """
    h = np.asarray(angles["h"], dtype=np.float32)
    v = np.asarray(angles["v"], dtype=np.float32)
    out = np.zeros((n_rows * SS, n_cols * SS, 4), dtype=np.uint8)

    for band in bands or []:
        if not band.get("on", True):
            continue
        lo = float(band.get("min", 0.0))
        raw_max = band.get("max", None)
        hi = float("inf") if raw_max in (None, "", "inf") else float(raw_max)
        if not (hi > lo):
            continue
        width = max(1, min(SS * 2, int(band.get("width", 2) or 2)))
        r, g, b = _hex_to_rgb(band.get("color", "#ffffff"))

        # `lo` inclusive, `hi` exclusive: the bands the user sets are adjacent
        # (…<5, 5–15, ≥15) and an angle must land in exactly one of them.
        sel_h = np.isfinite(h) & (h >= lo) & (h < hi)
        sel_v = np.isfinite(v) & (v >= lo) & (v < hi)
        if not (sel_h.any() or sel_v.any()):
            continue

        half = width // 2
        # A vertical line on the interface between (r, c) and (r, c+1):
        # that interface sits at overlay column (c+1)*SS.
        for rr, cc in zip(*np.nonzero(sel_h)):
            x = (cc + 1) * SS
            y0, y1 = rr * SS, (rr + 1) * SS
            x0 = max(0, x - half - (width % 2))
            x1 = min(out.shape[1], x + half + 1)
            out[y0:y1, x0:x1] = (r, g, b, 255)
        # …and a horizontal line between (r, c) and (r+1, c) at row (r+1)*SS.
        for rr, cc in zip(*np.nonzero(sel_v)):
            y = (rr + 1) * SS
            x0, x1 = cc * SS, (cc + 1) * SS
            y0 = max(0, y - half - (width % 2))
            y1 = min(out.shape[0], y + half + 1)
            out[y0:y1, x0:x1] = (r, g, b, 255)

    return out


def parse_bands(raw: str) -> list[dict]:
    """Read the band list from the query string; empty/broken → sane defaults."""
    import json

    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return [b for b in parsed if isinstance(b, dict)]
        except (ValueError, TypeError):
            logger.warning("Ignoring unreadable grain-boundary bands: %r", raw[:120])
    return default_bands()


def default_bands() -> list[dict]:
    """Sub-boundaries < 5°, low-angle 5–15°, high-angle ≥ 15°.

    The 15° cut is the textbook low/high-angle divide (Read–Shockley breaks
    down around there); 5° is where sub-grain walls are usually separated from
    genuine low-angle boundaries. Both are editable — conventions differ per
    material and per group.
    """
    return [
        {"id": "sub",  "min": MIN_ANGLE_DEG, "max": 5.0,  "color": "#8be9fd", "width": 1, "on": True},
        {"id": "lagb", "min": 5.0,           "max": 15.0, "color": "#f1fa8c", "width": 2, "on": True},
        {"id": "hagb", "min": 15.0,          "max": None, "color": "#ffffff", "width": 3, "on": True},
    ]
