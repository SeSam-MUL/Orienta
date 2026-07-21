"""EDAX TSL UP1/UP2 + OSC sidecar geometry reader.

EDAX exports raw EBSD patterns as uncompressed ``.up1`` (8-bit) / ``.up2``
(16-bit) files. kikuchipy reads the *patterns* natively and lazily, but:

* A **version-1** UP1/UP2 header carries no map grid, so kikuchipy returns a
  flat 1-D navigation (all patterns in a single row). The real grid lives in
  the companion ``.osc`` file.
* Neither the v1 nor (in practice) the v3 UP1 header carries the real scan
  **step size** — the v3 header field is a ``1.0`` placeholder. The true µm
  step also lives in the ``.osc``.

This module reads exactly the two things we need for correct loading — the
**grid** (so v1 files reshape to 2-D) and the **step size** (so the map scale
bar is right) — from the ``.osc`` sidecar. It deliberately does NOT decode the
OSC's embedded OIM Hough orientations: we index the patterns ourselves.

All byte offsets below are verified against real EDAX files; see
``tasks/up1-osc-format-facts.md`` for the full, cross-checked format notes.
The UP1/UP2 layout mirrors kikuchipy's authoritative reader
(``kikuchipy/io/plugins/edax_binary/_api.py``).
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The 8-byte marker that precedes the OSC point-data block. The block is NOT
# 4-byte aligned to the start of the file, so it must be located by this magic
# and read at absolute byte offsets from it (never via a 0-based float view).
_OSC_MAGIC = bytes([0xB9, 0x0B, 0xEF, 0xFF, 0x02, 0x00, 0x00, 0x00])

# Fixed byte offset of the pattern centre (xstar, ystar, zstar) as 3x float32
# in the .osc header, in the fixed-layout region before the variable phase
# table. Verified constant across the sample files.
_OSC_PC_OFFSET = 1860

_UP_DTYPE_NBYTES = {"up1": 1, "up2": 2}


@dataclass
class Up1Header:
    version: int
    sx: int  # pattern width
    sy: int  # pattern height
    pattern_offset: int
    n_patterns: int
    nx: Optional[int]  # map columns (v3 only, else None)
    ny: Optional[int]  # map rows (v3 only, else None)
    is_hex: bool


@dataclass
class OscMetadata:
    ncols: Optional[int]
    nrows: Optional[int]
    npoints: Optional[int]
    xstep: Optional[float]  # µm, column (x) step
    ystep: Optional[float]  # µm, row (y) step
    pc: Optional[tuple]     # (xstar, ystar, zstar) EDAX/TSL pattern centre, or None


@dataclass
class Up1Geometry:
    """Resolved geometry to hand to ``kikuchipy.load``."""

    version: int
    n_patterns: int
    # (n map rows, n map cols) to pass as kikuchipy ``nav_shape``; None means
    # "let kikuchipy use the file header" (v3) or "grid unknown" (v1 w/o .osc).
    nav_shape: Optional[tuple[int, int]]
    # (dy, dx) = (row step, column step) in µm from the .osc, or None.
    step_yx: Optional[tuple[float, float]]
    # (xstar, ystar, zstar) EDAX/TSL pattern centre from the .osc, or None.
    # None means indexing must fall back to the (wrong) kikuchipy default PC —
    # callers should warn the user loudly in that case.
    pc_edax: Optional[tuple]
    osc_path: Optional[str]


def _suffix_key(path: str) -> str:
    ext = Path(path).suffix[1:].lower()
    if ext not in _UP_DTYPE_NBYTES:
        raise ValueError(f"Not an EDAX UP1/UP2 file: {path}")
    return ext


def is_edax_up_file(path: str) -> bool:
    """True if *path* has a ``.up1`` / ``.up2`` extension."""
    return Path(path).suffix[1:].lower() in _UP_DTYPE_NBYTES


def read_up1_header(path: str) -> Up1Header:
    """Parse the UP1/UP2 binary header (little-endian).

    Layout (verified, mirrors kikuchipy's reader):
      @0  uint32 version
      @4  uint32 sx (pattern width)
      @8  uint32 sy (pattern height)
      @12 uint32 pattern_offset
      v1: 16-byte header, no grid.
      v3+: @16 skip 1 byte, @17 uint32 nx, @21 uint32 ny, @25 uint8 is_hex,
           @26 float64 dx, @34 float64 dy, patterns @42.
    """
    nbytes = _UP_DTYPE_NBYTES[_suffix_key(path)]
    file_size = Path(path).stat().st_size
    with open(path, "rb") as f:
        head = f.read(42)
    if len(head) < 16:
        raise ValueError(f"UP file too small to contain a header: {path}")

    version, sx, sy, pattern_offset = struct.unpack_from("<4I", head, 0)

    if sx == 0 or sy == 0:
        raise ValueError(f"UP file reports zero pattern size ({sx}x{sy}): {path}")

    if version == 2:
        # kikuchipy: "Only files with version 1 or >= 3, not 2, can be read"
        raise ValueError(f"Unsupported UP file version 2: {path}")

    per_pattern = sx * sy * nbytes
    if version == 1:
        n_patterns = (file_size - pattern_offset) // per_pattern
        return Up1Header(version, sx, sy, pattern_offset, int(n_patterns),
                         nx=None, ny=None, is_hex=False)

    # version >= 3
    if len(head) < 42:
        raise ValueError(f"UP v{version} header truncated: {path}")
    nx, ny = struct.unpack_from("<2I", head, 17)
    is_hex = bool(head[25])
    if is_hex:
        n_patterns = (file_size - pattern_offset) // per_pattern
        return Up1Header(version, sx, sy, pattern_offset, int(n_patterns),
                         nx=None, ny=None, is_hex=True)
    return Up1Header(version, sx, sy, pattern_offset, int(nx * ny),
                     nx=int(nx), ny=int(ny), is_hex=False)


def read_osc_metadata(osc_path: str) -> OscMetadata:
    """Read grid + step size from an EDAX ``.osc`` file.

    Grid lives at fixed offsets @16/@20/@24 (ncols-1, nrows-1, npoints).
    Step size is the first two float32 after the point-data magic marker.
    Every field is independently validated; anything that fails its sanity
    check comes back as ``None`` rather than a guessed value.
    """
    ncols = nrows = npoints = None
    xstep = ystep = None
    pc = None
    try:
        raw = Path(osc_path).read_bytes()
    except OSError:
        logger.warning("Could not read .osc sidecar %s", osc_path, exc_info=True)
        return OscMetadata(None, None, None, None, None, None)

    # --- grid (fixed offsets) ---
    if len(raw) >= 28:
        o16, o20, o24 = struct.unpack_from("<3I", raw, 16)
        nc, nr, npt = o16 + 1, o20 + 1, o24
        # Trust only when the three fields are mutually consistent.
        if npt > 0 and nc > 0 and nr > 0 and nc * nr == npt:
            ncols, nrows, npoints = nc, nr, npt
        else:
            logger.warning(
                ".osc grid fields inconsistent (ncols-1=%d, nrows-1=%d, "
                "npoints=%d) in %s — ignoring grid", o16, o20, o24, osc_path)

    # --- step size (magic-anchored) ---
    k = raw.find(_OSC_MAGIC)
    if k >= 0 and k + 24 <= len(raw):
        xs, ys = struct.unpack_from("<2f", raw, k + 16)
        if _plausible_step(xs) and _plausible_step(ys):
            xstep, ystep = float(xs), float(ys)
        else:
            logger.warning(
                ".osc step values implausible (xstep=%r, ystep=%r) in %s — "
                "ignoring step", xs, ys, osc_path)

    # --- pattern centre (xstar, ystar, zstar), EDAX/TSL convention ---
    # Three consecutive float32 at a fixed header offset. Verified across real
    # files (Scan5/Scan57: 0.550/0.503/0.702; map: 0.576/0.649/0.644) and
    # confirmed correct: a joint orientation+PC refinement converges straight
    # back to these values, and using them ~2.5x's the match NCC vs the
    # kikuchipy default (0.5,0.5,0.5). See tasks/up1-osc-format-facts.md.
    if len(raw) >= _OSC_PC_OFFSET + 12:
        vals = struct.unpack_from("<3f", raw, _OSC_PC_OFFSET)
        if all(_plausible_pc(v) for v in vals) and not _all_equal(vals):
            pc = tuple(float(v) for v in vals)
        else:
            logger.warning(
                ".osc PC values implausible %s in %s — ignoring PC", vals, osc_path)

    return OscMetadata(ncols, nrows, npoints, xstep, ystep, pc)


def _plausible_step(v: float) -> bool:
    return (v == v) and (v not in (float("inf"), float("-inf"))) and (1e-4 <= v <= 1e4)


def _plausible_pc(v: float) -> bool:
    # A pattern-centre fraction sits well inside (0, 1.5); a stored-but-unset
    # slot is usually 0.0 or garbage far outside this band.
    return (v == v) and (v not in (float("inf"), float("-inf"))) and (0.1 <= v <= 1.5)


def _all_equal(vals) -> bool:
    return len(set(round(float(v), 6) for v in vals)) <= 1


def _find_osc_sidecar(up1_path: str) -> Optional[str]:
    """Return the companion ``.osc`` path (same stem, same folder), or None."""
    p = Path(up1_path)
    # Case-insensitive match so ``Scan5.OSC`` is found next to ``Scan5.up1``.
    for cand in p.parent.glob(f"{p.stem}.*"):
        if cand.suffix.lower() == ".osc":
            return str(cand)
    return None


def resolve_up1_geometry(up1_path: str) -> Up1Geometry:
    """Resolve nav_shape + step for a UP1/UP2 file, using its ``.osc`` sidecar.

    * v1 (flat header): nav_shape MUST come from the .osc grid; falls back to a
      perfect-square guess only if the pattern count is a square and no .osc is
      present. Otherwise nav_shape is None and the caller should fail loudly.
    * v3: kikuchipy already knows the grid, so nav_shape stays None (do not
      override the authoritative header). Step still comes from the .osc.
    * Step: taken from the .osc whenever available (the UP header step is a
      placeholder), as (dy, dx) = (ystep, xstep).
    """
    header = read_up1_header(up1_path)
    osc_path = _find_osc_sidecar(up1_path)
    osc = (read_osc_metadata(osc_path) if osc_path
           else OscMetadata(None, None, None, None, None, None))

    step_yx = None
    if osc.ystep is not None and osc.xstep is not None:
        step_yx = (osc.ystep, osc.xstep)

    nav_shape: Optional[tuple[int, int]] = None
    if header.version == 1 and not header.is_hex:
        if osc.nrows is not None and osc.ncols is not None:
            if osc.npoints == header.n_patterns:
                nav_shape = (osc.nrows, osc.ncols)
            else:
                logger.warning(
                    ".osc npoints (%s) != UP pattern count (%s) for %s — "
                    "not reshaping from .osc", osc.npoints, header.n_patterns,
                    up1_path)
        if nav_shape is None:
            # Last resort: a perfectly square scan.
            root = int(round(header.n_patterns ** 0.5))
            if root * root == header.n_patterns and root > 1:
                nav_shape = (root, root)
                logger.info(
                    "No usable .osc grid for %s; assuming square %dx%d map",
                    up1_path, root, root)

    return Up1Geometry(
        version=header.version,
        n_patterns=header.n_patterns,
        nav_shape=nav_shape,
        step_yx=step_yx,
        pc_edax=osc.pc,
        osc_path=osc_path,
    )
