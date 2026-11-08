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
import zlib
from typing import Any

import numpy as np

from crystal_symmetry import phase_point_group

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
            # The group with the phase's real unique axis: a b-unique
            # monoclinic phase reduced by orix's z-axis 2/m would report two
            # orientations related by its true two-fold axis as 180 deg apart.
            pg = phase_point_group(phases[pid])
            if pg is not None:
                out[int(pid)] = pg
        except Exception:  # noqa: BLE001 — a phase without a point group is skipped
            continue
    return out


def boundary_angles(xmap, n_rows: int, n_cols: int, mask=None, topology=None) -> dict[str, np.ndarray]:
    """Misorientation angle (degrees) across every horizontal and vertical edge.

    Returns ``{"h": (n_rows, n_cols-1), "v": (n_rows-1, n_cols)}``. NaN marks a
    pair that cannot be compared: an unindexed pixel, or two DIFFERENT phases —
    a misorientation between different crystals is not a grain boundary angle,
    and reporting one would be meaningless. Those edges are still interfaces,
    which the caller may draw separately; here they are simply not angles.

    With a hexagonal ``topology`` it returns ``{"pairs": (a, b), "angle": (n_pairs,)}``: the angle
    across every pair of neighbouring points (flat indices ``a < b``), six neighbours per point.
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

    if topology is not None:
        a, b = topology.pairs()
        flat_q, flat_pid, flat_ok = q2d.reshape(-1, 4), pid2d.ravel(), indexed.ravel()
        angle = _edge_angles(flat_q[a], flat_q[b], flat_pid[a], flat_pid[b], flat_ok[a] & flat_ok[b])
        return {"pairs": (a, b), "angle": angle}

    ok_h = indexed[:, :-1] & indexed[:, 1:]
    ang_h = _edge_angles(q2d[:, :-1], q2d[:, 1:], pid2d[:, :-1], pid2d[:, 1:], ok_h)

    ok_v = indexed[:-1, :] & indexed[1:, :]
    ang_v = _edge_angles(q2d[:-1, :], q2d[1:, :], pid2d[:-1, :], pid2d[1:, :], ok_v)

    return {"h": ang_h, "v": ang_v}


# The angles depend only on the data, so they are computed once per result and
# reused while the user tries colours, widths and cut points.
_ANGLE_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def _crc(array) -> tuple:
    """(shape, dtype, CRC-32 of the bytes) of an array: a name for its content."""
    a = np.ascontiguousarray(array)
    return (a.shape, a.dtype.str, zlib.crc32(a.reshape(-1)))


def _content_key(xmap, mask) -> tuple:
    """Everything ``boundary_angles`` reads from the map, as a key: the orientations, the phase of
    every point, the point group of every phase and the selection mask.

    The map is not a name for its content: unify, apply-to-grain, undo, reassign and clean-up change
    one ``xmap`` in place, and a map that is built anew can get the memory address of one that was
    freed. A key made of ``id(xmap)`` or of a result id then hands back the angles of the map as it
    was. A checksum of the bytes cannot. It takes about 40 ms for a map of 1000 x 1200 points.

    The public ``xmap.rotations`` builds a new ``Rotation`` of the points in the data on every call
    (130 ms at that size), so the key reads the stored rotations and the ``is_in_data`` flags, which
    together hold all that it returns. A map that does not keep them that way (a stand-in) is read
    through ``rotations``."""
    stored = getattr(xmap, "_rotations", None)
    in_data = getattr(xmap, "is_in_data", None)
    quats = getattr(stored, "data", None)
    if isinstance(quats, np.ndarray) and isinstance(in_data, np.ndarray):
        rotation_key = (_crc(quats), _crc(in_data))
    else:
        rotation_key = (_crc(np.asarray(getattr(getattr(xmap, "rotations", None), "data", np.empty((0, 4))))),)
    pids = np.asarray(getattr(xmap, "phase_id", np.empty(0)))
    # The name alone is not the group: the three axis variants of 2/m share one name and have different
    # operators, so the operators (their quaternions and improper flags) are part of the key.
    groups = tuple(sorted(
        (pid, getattr(pg, "name", repr(pg)),
         _crc(np.asarray(getattr(pg, "data", np.empty((0, 4))))),
         _crc(np.asarray(getattr(pg, "improper", np.empty(0)), dtype=bool)))
        for pid, pg in _phase_symmetries(xmap).items()))
    return (rotation_key, _crc(pids), groups, None if mask is None else _crc(np.asarray(mask, dtype=bool)))


def cached_boundary_angles(xmap, n_rows: int, n_cols: int, mask=None, result=None, topology=None):
    key = (
        getattr(result, "result_id", None),
        int(n_rows), int(n_cols),
        _content_key(xmap, mask),
        None if topology is None else (topology.kind, topology.odd_row_shift),
    )
    hit = _ANGLE_CACHE.get(key)
    if hit is not None:
        return hit
    angles = boundary_angles(xmap, n_rows, n_cols, mask, topology=topology)
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


def render_boundaries_hex(angles: dict, geometry, spec, bands: list[dict]) -> np.ndarray:
    """The bands of a HEXAGONAL scan, drawn on its display raster (``hex_raster``): each boundary is a line along
    the edge two hexagons share, i.e. perpendicular to the line between their centres, through its middle, as long
    as a hexagon side (the circumradius, ``2 * step_y / 3``). ``angles`` is what ``boundary_angles`` returns for a
    hexagonal topology; ``bands`` and their order mean what they mean in :func:`render_boundaries`. ``width`` is in
    raster pixels, measured as ``SS`` is above: the line is as thick as the same ``width`` is on the square overlay,
    ``(width + 1) / 4`` scan steps (``width + 1`` overlay pixels of a quarter step), rounded to whole raster pixels
    with a half rounded up (``width`` 2 is 3, 2 and 1 pixels at raster factor 4, 2 and 1). Nothing is drawn outside
    the hexagons of the scan.
    """
    from PIL import Image, ImageDraw
    from backend.api.services import hex_raster

    a, b = angles["pairs"]
    angle = np.asarray(angles["angle"], dtype=np.float32)
    x, y = geometry.positions()
    px, py = x.ravel(), y.ravel()
    dx, dy = px[b] - px[a], py[b] - py[a]
    length = np.hypot(dx, dy)
    tx, ty = -dy / length, dx / length                          # unit tangent of the shared edge
    half = (2.0 * float(geometry.step_y) / 3.0) / 2.0           # half a hexagon side, um
    mx, my = (px[a] + px[b]) / 2.0, (py[a] + py[b]) / 2.0
    # um -> raster pixel, as a continuous coordinate in which pixel j spans [j, j + 1)
    to_u = lambda v: (v - spec.x0) / spec.step_um               # noqa: E731
    to_v = lambda v: (v - spec.y0) / spec.step_um               # noqa: E731
    u1, v1 = to_u(mx - tx * half), to_v(my - ty * half)
    u2, v2 = to_u(mx + tx * half), to_v(my + ty * half)
    # PIL cuts the end points of a line to whole pixels and, for a wide line, spreads the width about them
    # unevenly (a line of even width reaches one pixel further to one side, and which side depends on the
    # direction it is drawn in). Whole end points, the same direction for every edge (from the left end to the
    # right one, a vertical edge from its top), and an offset of a quarter pixel keep the middle of the line
    # within about half a pixel of the true edge for every direction and width. The quarter also breaks a tie
    # downwards: an end point at a whole number of pixels (t integer) would floor to t, but with the quarter
    # taken off it floors to t - 1, and so does one a hair above or below the whole number, which rounding
    # noise in the positions would otherwise send either way; measured against the lattice
    # (tests/test_hex_grain_boundaries.py), a plain cut put it up to 1.5 pixels to the upper left.
    swap = (u2 < u1) | ((u2 == u1) & (v2 < v1))
    u1, u2 = np.where(swap, u2, u1), np.where(swap, u1, u2)
    v1, v2 = np.where(swap, v2, v1), np.where(swap, v1, v2)
    u1, v1, u2, v2 = (np.floor(t - 0.25) for t in (u1, v1, u2, v2))

    image = Image.new("RGBA", (spec.width, spec.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for band in bands or []:
        if not band.get("on", True):
            continue
        lo = float(band.get("min", 0.0))
        raw_max = band.get("max", None)
        hi = float("inf") if raw_max in (None, "", "inf") else float(raw_max)
        if not (hi > lo):
            continue
        width = max(1, min(SS * 2, int(band.get("width", 2) or 2)))
        # the thickness the square overlay gives this width, in scan steps, in raster pixels (a half rounds up)
        width = max(1, int(np.floor((width + 1) * spec.factor / SS + 0.5)))
        r, g, bl = _hex_to_rgb(band.get("color", "#ffffff"))
        sel = np.flatnonzero(np.isfinite(angle) & (angle >= lo) & (angle < hi))
        for k in sel:
            draw.line([(float(u1[k]), float(v1[k])), (float(u2[k]), float(v2[k]))], fill=(r, g, bl, 255), width=width)
    out = np.array(image, dtype=np.uint8)
    # a line at the edge of the scan reaches outside the hexagons: keep what the cell index covers (the raster
    # the other layers are drawn on), so the overlay is transparent wherever they are
    out[hex_raster.cell_index(geometry, spec) < 0] = 0
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
