"""Independent oracle for the display raster of a hexagonal scan, and the writer of the shared fixture.

Written from the lattice alone (numpy and math): it imports nothing from the application, so a test that
holds the application against it compares two derivations and not the code with itself. The fixture it
writes is read by the Python tests and by the frontend's (hexGeometry.test.js), which is what keeps the two
implementations of the raster rule on the same numbers.

``python -m tests.hex_raster_oracle --write`` regenerates tests/fixtures/hex_raster_cases.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hex_raster_cases.json"
MAX_PIXELS = 16_000_000

#: name, rows, columns, step_x (um), odd_row_shift (in steps), factor of the raster. The last case has a
#: step that is exact in binary and factor 1, so pixel centres meet the lattice exactly: ties between two
#: columns (dx == a/2) and pixels on the border of a hexagon, which no case of factor 4 reaches.
GEOMETRIES = (
    ("a_3x3_right", 3, 3, 2.0, 0.5, 4),
    ("b_4x5_left", 4, 5, 0.5, -0.5, 4),
    ("c_one_row", 1, 4, 1.0, 0.5, 4),
    ("d_2x2", 2, 2, 0.4, 0.5, 4),
    ("e_oim_step", 5, 3, 0.15, 0.5, 4),
    ("f_7x6_left", 7, 6, 1.3, -0.5, 4),
    ("g_exact_ties", 3, 3, 2.0, -0.5, 1),
)
#: rows, columns, step_x: scans whose factor is decided by the pixel cap alone (no cell index is built)
FACTOR_SCANS = ((100, 100, 0.5), (1000, 1000, 0.4), (1200, 1000, 0.5), (3000, 3000, 0.5))


def geometry_dict(n_rows, n_cols, step_x, shift):
    """The dict the API reports for a hexagonal scan (rows sqrt(3)/2 steps apart)."""
    return {"kind": "hex", "n_rows": n_rows, "n_cols": n_cols, "step_x": step_x,
            "step_y": step_x * (math.sqrt(3.0) / 2.0), "odd_row_shift": shift}


def positions(g):
    """x, y (um) of every point: odd rows shifted by odd_row_shift steps, rows step_y apart."""
    r = np.arange(g["n_rows"], dtype=np.float64)[:, None]
    c = np.arange(g["n_cols"], dtype=np.float64)[None, :]
    x = c * g["step_x"] + np.where(r % 2 == 1, g["odd_row_shift"] * g["step_x"], 0.0)
    return x, np.broadcast_to(r * g["step_y"], x.shape)


def spec(g, factor):
    """Raster of square pixels step_x/factor covering every hexagon (circumradius 2/3 step_y)."""
    a, b, s = g["step_x"], g["step_y"], g["odd_row_shift"]
    offsets = [0.0, s * a] if g["n_rows"] >= 2 else [0.0]
    x_lo = min(offsets) - a / 2
    x_hi = (g["n_cols"] - 1) * a + max(offsets) + a / 2
    rho = 2 * b / 3
    y_lo, y_hi = -rho, (g["n_rows"] - 1) * b + rho
    d = a / factor
    return {"factor": factor, "step_um": d, "x0": x_lo, "y0": y_lo,
            "width": math.ceil((x_hi - x_lo) / d - 1e-9), "height": math.ceil((y_hi - y_lo) / d - 1e-9)}


def factor_for(g):
    for f in (4, 2, 1):
        s = spec(g, f)
        if s["width"] * s["height"] <= MAX_PIXELS:
            return f
    return 1


def brute_cell_index(g, sp):
    """(height, width) flat index of the nearest point if the pixel centre lies inside its hexagon,
    else -1. Plain argmin over every point; an exact tie goes to the lowest flat index. Exact ties need
    positions that are exact in binary; with a step carrying rounding noise (1.7, say) a pixel on the border
    of two cells can fall to either one, in the last bit, and this function and the application may differ
    there (and this one can lose such a pixel altogether: covered_mask is what holds the application to
    "no pixel of the scan is left out"). The raster is for display and does not depend on which one."""
    px, py = positions(g)
    px, py = px.ravel(), py.ravel()
    a, rho = g["step_x"], 2 * g["step_y"] / 3
    xs = sp["x0"] + (np.arange(sp["width"]) + 0.5) * sp["step_um"]
    ys = sp["y0"] + (np.arange(sp["height"]) + 0.5) * sp["step_um"]
    x, y = np.meshgrid(xs, ys)
    best = np.full(x.shape, np.inf)
    owner = np.full(x.shape, -1, dtype=np.int64)
    for k in range(px.size):
        d2 = (x - px[k]) ** 2 + (y - py[k]) ** 2
        closer = d2 < best
        best = np.where(closer, d2, best)
        owner = np.where(closer, k, owner)
    dx, dy = np.abs(x - px[owner]), np.abs(y - py[owner])
    return np.where((dx <= a / 2) & (dy <= rho * (1 - dx / a)), owner, -1)


#: name, rows, columns, step_x (um), odd_row_shift, factor 1: steps with rounding noise, whose raster is
#: held to the union of the hexagons (covered_mask) and not to a cell per pixel: a pixel on the border between
#: two cells can fall to either one there, but never to none.
COVERAGE = (
    ("h_noise_1p7_right", 6, 6, 1.7, 0.5, 1),
    ("i_noise_1p7_left", 6, 6, 1.7, -0.5, 1),
    ("j_noise_0p7_right", 9, 7, 0.7, 0.5, 1),
    ("k_noise_3p3_left", 8, 7, 3.3, -0.5, 1),
    ("l_noise_0p0017_right", 6, 6, 0.0017, 0.5, 1),
    ("m_noise_1700_left", 6, 6, 1700.0, -0.5, 1),
)
#: how far beyond its border (as a fraction of a step) a hexagon still covers a pixel centre: far above the
#: rounding noise of the positions, far below the distance of a pixel centre that is not on a border. The
#: fixture carries it as "edge_tolerance": the Python and the JavaScript implementation are each held to this
#: value (EDGE_TOLERANCE in hex_raster.py and in hexGeometry.js), and it is a fraction of the step, not of um.
COVER_TOLERANCE = 1e-9
#: step_x (um) of the one-row scans whose borders are probed: exact in binary, small and large (1.7 mm in um),
#: the last two carrying rounding noise. A tolerance in um instead of in steps is wrong in the last two.
BORDER_STEPS = (2.0, 0.0017, 1700.0)
#: how far beyond the border of a hexagon (fraction of a step; negative: inside it) a pixel centre lies, and
#: whether the hexagon still owns it. 1e-12 and 5e-10 lie within COVER_TOLERANCE, 2e-9 and 1e-8 beyond it.
BORDER_OFFSETS = ((-1e-8, True), (0.0, True), (1e-12, True), (5e-10, True), (2e-9, False), (1e-8, False),
                  (1e-6, False))


def covered_mask(g, sp):
    """(height, width) bool: the pixel centre lies in at least one closed hexagon of the scan. The union of
    the hexagons, found without asking which point is nearest, so it holds with rounding noise in the step
    where brute_cell_index (nearest first) may lose a pixel on a border."""
    px, py = positions(g)
    a, rho = g["step_x"], 2 * g["step_y"] / 3
    tol = COVER_TOLERANCE * a
    xs = sp["x0"] + (np.arange(sp["width"]) + 0.5) * sp["step_um"]
    ys = sp["y0"] + (np.arange(sp["height"]) + 0.5) * sp["step_um"]
    covered = np.zeros((sp["height"], sp["width"]), dtype=bool)
    for r in range(g["n_rows"]):
        for c in range(g["n_cols"]):
            j0, j1 = np.searchsorted(xs, px[r, c] - a), np.searchsorted(xs, px[r, c] + a)
            i0, i1 = np.searchsorted(ys, py[r, c] - rho - tol), np.searchsorted(ys, py[r, c] + rho + tol)
            dx = np.abs(xs[j0:j1][None, :] - px[r, c])
            dy = np.abs(ys[i0:i1][:, None] - py[r, c])
            covered[i0:i1, j0:j1] |= (dx <= a / 2 + tol) & (dy <= rho * (1 - dx / a) + tol)
    return covered


def build_coverage():
    out = []
    for name, rows, cols, step_x, shift, factor in COVERAGE:
        g = geometry_dict(rows, cols, step_x, shift)
        sp = spec(g, factor)
        mask = covered_mask(g, sp)
        out.append({"name": name, "geometry": g,
                    "spec": {k: sp[k] for k in ("factor", "height", "width", "step_um", "x0", "y0")},
                    "covered": ["".join("#" if v else "." for v in row) for row in mask]})
    return out


def build_borders():
    """Pixel centres next to the border of cell (0, 0) of a row of two points, for each of BORDER_STEPS: on the
    vertical border at dx = a/2 and on the slanted one at dx = a/4 (where it is 3/4 of rho high), offset
    outwards by a fraction of the step. A raster of one pixel of a (step_um) whose centre is (x, y)."""
    out = []
    for a in BORDER_STEPS:
        g = geometry_dict(1, 2, a, 0.5)
        rho = 2 * g["step_y"] / 3
        for side, (x0, y0) in (("vertical", (-a / 2, 0.0)), ("slanted", (a / 4, 0.75 * rho))):
            for frac, inside in BORDER_OFFSETS:
                x, y = (x0 - frac * a, y0) if side == "vertical" else (x0, y0 + frac * a)
                out.append({"name": f"{side}_a{a:g}_{frac:g}", "geometry": g, "x": x, "y": y, "inside": inside})
    return out


def build_cases():
    cases = []
    for name, rows, cols, step_x, shift, factor in GEOMETRIES:
        g = geometry_dict(rows, cols, step_x, shift)
        sp = spec(g, factor)
        idx = brute_cell_index(g, sp)
        h, w = sp["height"], sp["width"]
        flat = [(i, j) for i in range(h) for j in range(w)]
        # a factor-1 raster is small and every pixel of it matters: all of them; else every 7th
        picked = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)] + (flat if factor == 1 else flat[::7])
        pixels = [{"i": i, "j": j,
                   "cell": None if idx[i, j] < 0 else [int(idx[i, j]) // cols, int(idx[i, j]) % cols]}
                  for i, j in dict.fromkeys(picked)]
        px, py = positions(g)
        cells = [{"row": r, "col": c,
                  "pixel": [math.floor((py[r, c] - sp["y0"]) / sp["step_um"]),
                            math.floor((px[r, c] - sp["x0"]) / sp["step_um"])]}
                 for r in range(rows) for c in range(cols)]
        cases.append({"name": name, "geometry": g,
                      "spec": {k: sp[k] for k in ("factor", "height", "width", "step_um", "x0", "y0")},
                      "n_inside": int((idx >= 0).sum()), "pixels": pixels, "cells": cells})
    factors = [{"n_rows": r, "n_cols": c, "step_x": a, "factor": factor_for(geometry_dict(r, c, a, 0.5))}
               for r, c, a in FACTOR_SCANS]
    return {"version": 1, "edge_tolerance": COVER_TOLERANCE, "geometries": cases, "factors": factors,
            "coverage": build_coverage(), "borders": build_borders()}


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("pass --write to overwrite " + str(FIXTURE))
    # bytes, not write_text: write_text emits CRLF on Windows so the fixture keeps LF line endings on Windows
    FIXTURE.write_bytes((json.dumps(build_cases(), indent=1) + "\n").encode("utf-8"))
    print("wrote", FIXTURE)
