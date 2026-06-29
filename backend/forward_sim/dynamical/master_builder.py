"""SP1 — assemble the dynamical EBSD master pattern on the Lambert grid (Task 11).

This is the SP1 capstone: it loops over the northern-hemisphere directions of the
modified-Lambert square grid and, for each direction ``k``, evaluates the
master-pattern value

    I(k) = Re{ Σ_{g,h strong}  Lgh(k)[g,h] · Sgh[g,h] } / N_atoms

(design §2 step 5, §4.6; full algorithm spec in the Task-11 brief).  The master is
a **pure crystal property** — it carries NO detector / tilted-sample geometry.  In
particular it is **not** weighted by the per-direction Monte-Carlo back-scatter
yield ``accum_e(k, E)``: ``accum_e`` encodes the 70°-tilted-sample MC geometry
(strongly upper/lower asymmetric, ~15× on Ni 20 kV), which belongs to the *forward
projection onto a detector*, not to the master.  EMsoft itself normalises ``mLPNH``
by a **scalar** electron count, not a per-pixel ``accum_e``; its ``mLPNH`` is
upper/lower symmetric to ~0.999.  (Depth absorption — the ``accum_z`` histogram via
``lambdaE`` — *is* part of the master and is kept.)  The pieces it composes are the
validated SP0/SP1 primitives:

* :func:`backend.forward_sim.crystal.structure_matrix.compute_Ug_table` — the
  complex potential Fourier coefficients ``U_g`` (CACHED over unique ``|g|``: a
  single ``Ug_lookup`` closure is reused for every direction, see below).
* :func:`backend.forward_sim.crystal.structure_matrix.reflection_list` /
  :func:`backend.forward_sim.crystal.structure_matrix.bethe_partition` — the
  geometric reflections and the per-direction strong set.
* :func:`backend.forward_sim.dynamical.scattering_matrix.build_A` — the dynamical
  matrix ``A`` for a batch of directions.
* :func:`backend.forward_sim.dynamical.depth_integral.depth_integrated_Lgh` —
  the MC-depth-weighted ``Lgh`` (EMsoft ``CalcLgh``).
* :func:`backend.forward_sim.dynamical.structure_weight.compute_Sgh` — the
  direction-independent ``Sgh`` (EMsoft ``CalcSgh``).

Grid / convention parity (reuse, do NOT rebuild)
------------------------------------------------
The output grid is **pixel-aligned with EMsoft ``mLPNH``**.  EMsoft stores the
modified-Lambert square as a ``(2·npx+1, 2·npx+1)`` array; pixel ``(row, col)``
(0-based, centre at ``npx``) corresponds to the Lambert square coordinate

    xy = ((col − npx)/npx,  (row − npx)/npx)   ∈ [−1, 1]²

— exactly the unit-square convention consumed by
:func:`backend.dictionary_gpu.lambert.lambert_to_direction` /
``direction_to_lambert`` (which are themselves kikuchipy/EMsoft-validated, used by
the dictionary-GPU sampler with ``grid_sample(align_corners=True)``: ``xy[0]``
indexes the last axis = columns, ``xy[1]`` the second-to-last = rows).  We invert
each pixel's ``xy`` with :func:`lambert_to_direction` to get the **northern-
hemisphere** Cartesian unit direction, evaluate the master value there, and store
it back at ``[row, col]`` — so a direction recovered from a pixel re-projects to
the very same pixel (the convention-parity test pins this).

Reciprocity convention
----------------------
The sampled direction ``k`` is taken as the **incident-beam direction** (the
electron entering the crystal), with **no sign flip** — per the reciprocity
principle for back-scattered emission (Task-11 research §1: "There is no sign flip
applied to the k-vector itself").  The master-pattern value at pixel ``k̂`` is the
emitted intensity for electrons that, by reciprocity, enter along ``k̂``.

Cartesian-direction → Miller wavevector
---------------------------------------
The dynamical primitives (``excitation_error`` / ``build_A`` / ``depth_integral``)
expect the beam direction as a **Miller/reciprocal-coordinate** wavevector ``k``
with ``|k| = 1/λ`` under the reciprocal metric.  For a **cubic** crystal the
Cartesian axes are aligned with the crystal axes, so a Cartesian unit direction
``(dx, dy, dz)`` maps to Miller ``k = (dx, dy, dz)·(a/λ)`` (since
``|k|² = (1/a²)|k_Miller|² = 1/λ²`` ⟹ ``|k_Miller| = a/λ``).  **Non-cubic crystals
are fully supported**: the Cartesian direction is mapped to a Miller wavevector via
the general direct-structure-matrix transform (see :func:`_cartesian_to_miller`), so all
seven crystal systems build correctly (the SP0+SP1 NCC gate validates the cubic
Ni/Al path; SP6 added the non-cubic dsm path).

Hemisphere / symmetry scope
---------------------------
:func:`build_master` evaluates ONE hemisphere per call (``hemisphere="north"`` or
``"south"``).  A ``hemisphere="south"`` call on a **centrosymmetric** cell short-
circuits to the exact ``mLPNH.flip([0, 1])`` (the both-axis Lambert flip = inversion
on the grid) rather than a fresh eigensolve.  The GPU runner builds only NH for a
**cubic** cell and lets the writer copy-mirror it (the writer's mLPSH = mLPNH identity
copy is correct only when NH is flip-invariant — true for cubic, but NOT for
low-symmetry centro cells like monoclinic, where mLPNH ≠ mLPNH.flip).  For **every
non-cubic** cell (centro or not) the runner builds BOTH hemispheres directly via two
``build_master`` calls and hands both to the writer.  The modified-Lambert (Roşca-
Lambert) **square is solved in FULL** — every pixel, corners included — because the
square corners map to valid equatorial directions (az 45/135/225/315°, z=0) that
EMsoft also fills (``kvectors.f90::Calckvectors`` iterates the whole square, no disc
skip).  CORNER FIX 2026-06-22: this previously masked out the corners with an
inscribed-circle rule, leaving ~21.5% of the grid at 0 → four black equatorial
diamonds in the SHT reconstruction.  (The HEX path keeps its own ``InsideHexGrid``
mask — those hexagon corners genuinely fall outside the hex grid.)
"""
from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..crystal.structure_matrix import (
    bethe_partition,
    bethe_partition_batched,
    build_ug_lut,
    compute_Ug_table,
    excitation_error,
    reflection_list,
)
from ..crystal.xtal_io import CrystalStructure
from ..mc.emsoft_mc_input import MCData
from ..runtime import get_device
from .depth_integral import depth_integrated_Lgh, depth_integrated_Lgh_propagate
from .scattering_matrix import (
    build_A,
    build_A_batched,
    build_A_batched_gpu,
    build_diff_lut,
)
from .structure_weight import (
    compute_Sgh,
    compute_Sgh_for_build_master,
    build_sgh_box,
    _gather_sgh_block,
)

# Reuse the kikuchipy/EMsoft-validated Lambert square ↔ direction mapping.
from backend.dictionary_gpu.lambert import lambert_to_direction

# Default Bethe strong/weak cutoffs (EMsoft code defaults; the file's stored
# ``BetheParameters`` are tighter — pass them explicitly to match an oracle).
_DEFAULT_BETHE = (4.0, 8.0, 50.0, 1.0)


def relativistic_wavelength_nm(voltage_kV: float) -> float:
    """Relativistic electron wavelength ``λ`` in nm (EMsoft ``CalcWaveLength``).

    ``λ = 1226.426 / sqrt(V·(1 + 0.97845e-6·V))`` pm with ``V`` in volts, i.e. the
    relativistic de Broglie wavelength ``λ = h / sqrt(2 m_e e V (1 + eV/(2 m_e c²)))``.
    At 20 kV this gives 8.589 pm (= 0.008589 nm, ``1/λ = 116.4 nm⁻¹``) — matching
    the lessons.md verification.

    Args:
        voltage_kV: accelerating voltage in kV.

    Returns:
        Wavelength in nm.

    Raises:
        ValueError: if ``voltage_kV`` is not strictly positive.
    """
    if voltage_kV <= 0.0:
        raise ValueError(f"voltage_kV must be > 0, got {voltage_kV!r}")
    v = voltage_kV * 1.0e3  # volts
    lam_pm = 1226.426 / math.sqrt(v * (1.0 + 0.97845e-6 * v))
    return lam_pm * 1.0e-3  # pm → nm


def _is_cubic(structure: CrystalStructure) -> bool:
    """True if the lattice is cubic (a=b=c, all angles 90°)."""
    a, b, c, al, be, ga = structure.lattice
    return (
        abs(a - b) < 1e-9
        and abs(b - c) < 1e-9
        and abs(al - 90.0) < 1e-6
        and abs(be - 90.0) < 1e-6
        and abs(ga - 90.0) < 1e-6
    )


def _is_hexagonal(structure: CrystalStructure) -> bool:
    """True for the EMsoft ``usehex`` cells (hexagonal/trigonal, γ=120°).

    EMsoft switches to the sheared-60° hexagonal Lambert sampling whenever
    ``cell%xtal_system`` is 4 (hexagonal) or 5 (trigonal)
    (``SEM/EMEBSDmaster.f90:401-402``); the ``CrystalSystem`` field of the
    ``.xtal``/master HDF5 carries that same code
    (:pyattr:`backend.forward_sim.crystal.xtal_io.CrystalStructure.crystal_system`).
    Cubic (1) / tetragonal (2) / orthorhombic (3) / monoclinic (6) / triclinic (7)
    stay on the orthogonal **square** Lambert grid (the validated 0.998 path), so
    this returns False for them and the hex two-stage scheme is never triggered.
    """
    return int(structure.crystal_system) in (4, 5)


def _grid_directions(npx: int, device: torch.device):
    """Northern-hemisphere Cartesian directions for every Lambert grid pixel.

    Returns ``(directions, inside)`` where:

    * ``directions`` is ``(P, 3)`` float64 unit directions (``z ≥ 0``), one per
      grid pixel in row-major (``[row, col]``) order, ``P = (2·npx+1)²``;
    * ``inside`` is ``(P,)`` bool — **all True** for the square Roşca-Lambert grid:
      every point of ``[−1,1]²`` maps to a valid northern-hemisphere direction, so
      the WHOLE square is covered (the corners included).

    The pixel→Lambert mapping is EMsoft's: ``xy = ((col−npx)/npx, (row−npx)/npx)``,
    consumed by :func:`lambert_to_direction` (unit-square ``[−1,1]²``).

    CORNER FIX (2026-06-22).  This used to mask out the four square corners with an
    *inscribed-circle* rule ``x²+y² ≤ 1`` (dropping ~21.5 % of the grid), on the
    false premise that "the modified-Lambert projection does not cover the corners /
    EMsoft corners are undefined".  That is wrong: the Roşca-Lambert (modified-
    Lambert) projection is area-preserving and maps the ENTIRE square to the
    hemisphere — a corner ``(1,1)`` maps to the equatorial direction
    ``(0.707, 0.707, 0)`` (az=45°, ``z=0``), which is perfectly valid.  EMsoft's own
    ``kvectors.f90::Calckvectors`` RoscaLambert loop iterates the full square
    (``i=-npx..npx, j=-npx..npx`` / its point-group wedge) and calls ``AddkVector``
    for **every** pixel — no disc skip.  Leaving the corners at 0 produced four black
    equatorial diamonds in our reconstructed masters that EMsoft's masters do NOT
    have (corner directions map to the equator at az=45°/135°/225°/315°).  We now
    solve the full square, matching EMsoft.  (The HEX path keeps its own
    ``InsideHexGrid`` mask — those hexagon corners genuinely fall outside the hex
    grid; see :func:`_grid_directions_hex`.)
    """
    m = 2 * npx + 1
    idx = torch.arange(m, dtype=torch.float64, device=device)
    # rows (i) index the second-to-last axis → Lambert y; cols (j) → Lambert x.
    rr, cc = torch.meshgrid(idx, idx, indexing="ij")  # (m, m)
    x = (cc - npx) / npx
    y = (rr - npx) / npx
    xy = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)  # (P, 2)
    directions, _hemi = lambert_to_direction(xy)              # (P, 3), z ≥ 0
    # The Roşca-Lambert square covers the WHOLE square — every pixel is a valid
    # hemisphere direction, so they are all "inside" (matches EMsoft's full-square
    # RoscaLambert loop). No inscribed-disc mask.
    inside = torch.ones(xy.shape[0], dtype=torch.bool, device=device)
    return directions, inside


# --- EMsoft hexagonal-Lambert parameters (constants.f90, LambertParametersType) ---
# These reproduce EMsoft's ``LPs%…`` constants to double precision exactly.
_LP_PI = math.pi
_LP_sPio2 = math.sqrt(_LP_PI / 2.0)          # sqrt(pi/2)
_LP_srt = math.sqrt(3.0) / 2.0               # sqrt(3)/2
_LP_isrt = 1.0 / math.sqrt(3.0)              # 1/sqrt(3)
_LP_rtt = math.sqrt(3.0)                     # sqrt(3)
_LP_prea = 3.0 ** 0.25 / math.sqrt(2.0 * _LP_PI)   # 3^(1/4)/sqrt(2pi)
_LP_preb = 3.0 ** 0.25 * math.sqrt(2.0 / _LP_PI)   # 3^(1/4)*sqrt(2/pi)
_LP_prec = _LP_PI / (2.0 * math.sqrt(3.0))         # pi/(2 sqrt(3))
_LP_pred = 2.0 * _LP_PI / 3.0                       # 2pi/3
_LP_pree = 3.0 ** (-0.25)                           # 3^(-1/4)
_LP_pref = math.sqrt(6.0 / _LP_PI)                  # sqrt(6/pi)
_LP_preg = 2.0 * math.sqrt(_LP_PI) / (3.0 ** 0.75)  # 2 sqrt(pi)/3^(3/4)


def _get_sextant(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """EMsoft ``GetSextantDouble`` (vectorised) → int sextant 0..5 per (x, y)."""
    xx = (x * _LP_isrt).abs()
    pos = x >= 0.0
    le = y.abs() <= xx
    gt = y > xx
    # Seed with an invalid sextant (-1): the six torch.where clauses below
    # exhaustively partition (x, y) so this is overwritten for every point — but a
    # future partition gap then surfaces as a loud -1 rather than silent garbage.
    res = torch.full(x.shape, -1, dtype=torch.int64, device=x.device)
    res = torch.where(pos & le, torch.zeros_like(res), res)
    res = torch.where(pos & ~le & gt, torch.ones_like(res), res)
    res = torch.where(pos & ~le & ~gt, torch.full_like(res, 5), res)
    res = torch.where(~pos & le, torch.full_like(res, 3), res)
    res = torch.where(~pos & ~le & gt, torch.full_like(res, 2), res)
    res = torch.where(~pos & ~le & ~gt, torch.full_like(res, 4), res)
    return res


def _hex_to_direction(xy_hex: torch.Tensor):
    """EMsoft ``LambertSquareToSphere``'s hex sibling ``Lambert2DHexForwardDouble``.

    Maps hex-grid Lambert coordinates ``xy_hex`` (our (x, y) convention, the same
    one :func:`_grid_directions` uses for the square grid) to northern-hemisphere
    Cartesian unit directions.  Returns ``(directions (N,3), ok (N,) bool)`` where
    ``ok`` is False for points the projection places outside the hexagon
    (``ierr==1`` in the Fortran).
    """
    x = xy_hex[:, 0].to(torch.float64)
    y = xy_hex[:, 1].to(torch.float64)
    xc = (x - 0.5 * y) * _LP_preg
    yc = (y * _LP_srt) * _LP_preg
    n = x.shape[0]
    res = torch.zeros(n, 3, dtype=torch.float64, device=xy_hex.device)
    ok = torch.ones(n, dtype=torch.bool, device=xy_hex.device)
    origin = (xc.abs() == 0.0) & (yc.abs() == 0.0)
    res[origin] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=xy_hex.device)
    work = ~origin
    # flip coordinates: XY2 = (yc, xc)
    X = yc
    Y = xc
    ks = _get_sextant(X, Y)
    XX = torch.zeros_like(X)
    YY = torch.zeros_like(X)
    one = torch.ones_like(X)
    # safeX / safexp / safexp2 below only avoid a 0/0 in the divisions; the guarded
    # denominator is exactly 0 only on a sextant edge that _get_sextant routes to a
    # different branch for in-hexagon points, so the substituted 1 is never used in
    # the selected (torch.where) result — result-neutral, present purely for safety.
    m03 = (ks == 0) | (ks == 3)
    safeX = torch.where(X.abs() > 0, X, one)
    q = Y * _LP_prec / safeX
    XX = torch.where(m03, _LP_preb * X * torch.cos(q), XX)
    YY = torch.where(m03, _LP_preb * X * torch.sin(q), YY)
    m14 = (ks == 1) | (ks == 4)
    xp = X + _LP_rtt * Y
    safexp = torch.where(xp.abs() > 0, xp, one)
    yp = X * _LP_pred / safexp
    XX = torch.where(m14, _LP_prea * xp * torch.sin(yp), XX)
    YY = torch.where(m14, _LP_prea * xp * torch.cos(yp), YY)
    m25 = (ks == 2) | (ks == 5)
    xp2 = X - _LP_rtt * Y
    safexp2 = torch.where(xp2.abs() > 0, xp2, one)
    yp2 = X * _LP_pred / safexp2
    XX = torch.where(m25, _LP_prea * xp2 * torch.sin(yp2), XX)
    YY = torch.where(m25, -_LP_prea * xp2 * torch.cos(yp2), YY)
    q2 = XX.pow(2) + YY.pow(2)
    outside = q2 > 4.0
    ok = torch.where(work & outside, torch.zeros_like(ok), ok)
    good = work & ~outside
    sq = torch.sqrt((4.0 - q2).clamp(min=0.0))
    r0 = 0.5 * XX * sq      # res(1) before x/y flip-back
    r1 = 0.5 * YY * sq      # res(2) before x/y flip-back
    r2 = 1.0 - 0.5 * q2
    # flip the x and y coordinates back: res = (r1, r0, r2)
    res[:, 0] = torch.where(good, r1, res[:, 0])
    res[:, 1] = torch.where(good, r0, res[:, 1])
    res[:, 2] = torch.where(good, r2, res[:, 2])
    return res, ok


def _sphere_to_hex(dc: torch.Tensor):
    """EMsoft ``LambertSphereToHex`` (``Lambert2DHexInverseDouble``), vectorised.

    Maps unit sphere directions ``dc`` (our (x, y, z) convention) to hex-grid
    Lambert coordinates in the **same (x, y) convention** :func:`_hex_to_direction`
    consumes, so the two compose to the identity on the hex grid.  Returns
    ``(xy_hex (N,2), ok (N,) bool)``.
    """
    # eps2 mirrors EMsoft Lambert.f90:686 — a tiny offset that breaks the seam
    # degeneracy at the sextant boundaries (MDG 05/09/15 note); kept bit-faithful.
    eps2 = 1.0e-4
    x = dc[:, 0].to(torch.float64)
    y = dc[:, 1].to(torch.float64)
    z = dc[:, 2].to(torch.float64)
    n = x.shape[0]
    res = torch.zeros(n, 2, dtype=torch.float64, device=dc.device)
    # ok stays True for every point: EMsoft's Lambert2DHexInverseDouble sets ierr=1
    # only on the on-sphere check |1-Σxyz²|>1e-12, which never trips here because the
    # sole caller (_hex_to_square_resample) feeds renormalised unit vectors from
    # lambert_to_direction.  Out-of-grid rejection is delegated to that caller's
    # `in_bounds` test, so this map never needs to reject on its own.
    ok = torch.ones(n, dtype=torch.bool, device=dc.device)
    pole = (z.abs() - 1.0).abs() < 1e-15   # the ±z pole maps to the hex origin
    work = ~pole
    # flip x,y components; take |z|
    X2 = y
    Y2 = x
    Z2 = z.abs()
    q = torch.sqrt(2.0 / (1.0 + Z2))
    XX = q * X2 + eps2
    YY = q * Y2 + eps2
    ks = _get_sextant(XX, YY)
    sgnX = torch.sqrt(XX.pow(2) + YY.pow(2))
    sgnX = torch.where(XX < 0.0, -sgnX, sgnX)
    xxx = torch.zeros_like(XX)
    yyy = torch.zeros_like(XX)
    one = torch.ones_like(XX)
    m03 = (ks == 0) | (ks == 3)
    qa = _LP_pree * sgnX
    safeXX = torch.where(XX.abs() > 0, XX, one)
    yyy03 = torch.where(
        XX == 0.0,
        torch.full_like(XX, _LP_pref * _LP_PI * 0.5),
        qa * _LP_pref * torch.atan(YY / safeXX),
    )
    xxx = torch.where(m03, qa * _LP_sPio2, xxx)
    yyy = torch.where(m03, yyy03, yyy)
    m14 = (ks == 1) | (ks == 4)
    qb = _LP_prea * sgnX
    den14 = XX + _LP_rtt * YY
    safe14 = torch.where(den14.abs() > 0, den14, one)
    qq14 = torch.atan((YY - _LP_rtt * XX) / safe14)
    xxx = torch.where(m14, qb * _LP_rtt * (_LP_PI / 6.0 - qq14), xxx)
    yyy = torch.where(m14, qb * (0.5 * _LP_PI + qq14), yyy)
    m25 = (ks == 2) | (ks == 5)
    qc = _LP_prea * sgnX
    den25 = XX - _LP_rtt * YY
    safe25 = torch.where(den25.abs() > 0, den25, one)
    qq25 = torch.atan((YY + _LP_rtt * XX) / safe25)
    xxx = torch.where(m25, qc * _LP_rtt * (_LP_PI / 6.0 + qq25), xxx)
    yyy = torch.where(m25, qc * (-0.5 * _LP_PI + qq25), yyy)
    # flip the coordinates back: res = (yyy, xxx)
    rx = yyy
    ry = xxx
    # transform to the hexagonal grid
    hx = (rx + ry * _LP_isrt) / _LP_preg
    hy = (ry * 2.0 * _LP_isrt) / _LP_preg
    res[:, 0] = torch.where(work, hx, torch.zeros_like(hx))
    res[:, 1] = torch.where(work, hy, torch.zeros_like(hy))
    return res, ok


def _inside_hex_grid(xy_hex: torch.Tensor) -> torch.Tensor:
    """EMsoft ``InsideHexGrid`` (vectorised) → bool mask of in-hexagon points."""
    ax = (xy_hex[:, 0] - 0.5 * xy_hex[:, 1]).abs()
    ay = (xy_hex[:, 1] * _LP_srt).abs()
    res = ~((ax > 1.0) | (ay > _LP_srt))
    res = res & ~((ax + ay * _LP_isrt) > 1.0)
    return res


def _grid_directions_hex(npx: int, device: torch.device):
    """Hex-grid (sheared-60°) directions for every Lambert grid pixel.

    The hexagonal sibling of :func:`_grid_directions`: array position ``[row, col]``
    samples the direction whose **hex-Lambert** coordinate is
    ``xy_hex = ((col−npx)/npx, (row−npx)/npx)`` (our (x, y) convention), projected
    to the sphere by :func:`_hex_to_direction` (EMsoft ``LambertHexToSphere``).
    ``inside`` is the intersection of the hexagon-interior test (EMsoft
    ``InsideHexGrid``) and the projection's validity flag.  This is EMsoft's Stage-1
    grid for ``usehex`` cells, which is later resampled onto the square grid by
    :func:`_hex_to_square_resample`.
    """
    m = 2 * npx + 1
    idx = torch.arange(m, dtype=torch.float64, device=device)
    rr, cc = torch.meshgrid(idx, idx, indexing="ij")  # (m, m)
    x = (cc - npx) / npx
    y = (rr - npx) / npx
    xy = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)  # (P, 2)
    inside_hex = _inside_hex_grid(xy)
    directions, ok = _hex_to_direction(xy)
    # renormalise (FP drift, matching EMsoft's res/dsqrt(sum(res*res)) and the
    # square path's renormalisation in lambert_to_direction).
    norm = directions.norm(dim=-1, keepdim=True).clamp(
        min=torch.finfo(directions.dtype).eps
    )
    directions = directions / norm
    inside = inside_hex & ok
    return directions, inside


def _hex_to_square_resample(grid_hex: torch.Tensor, npx: int) -> torch.Tensor:
    """Resample a hex-sampled master grid onto the stored square Lambert grid.

    Mirrors ``SEM/EMEBSDmaster.f90:923-963`` exactly: for each square pixel
    ``[row, col]`` (Lambert ``xy_sq = ((col−npx)/npx, (row−npx)/npx)``, our
    convention) project to the sphere with the **square** map
    (:func:`lambert_to_direction`, == EMsoft ``LambertSquareToSphere``), find where
    that direction lands on the hex grid with :func:`_sphere_to_hex`
    (== ``LambertSphereToHex``), and 4-tap bilinearly blend the four neighbouring
    hex-grid samples.  EMsoft's edge clamp (``if nixp>npx: nixp=nix``) and the
    ``ierr`` validity gate are reproduced; out-of-bounds / invalid pixels stay 0
    (EMsoft leaves them at their original value, which for our freshly-zeroed
    output is 0 — the inscribed-disc NCC mask excludes them anyway).
    """
    m = 2 * npx + 1
    device = grid_hex.device
    idx = torch.arange(m, dtype=torch.float64, device=device)
    rr, cc = torch.meshgrid(idx, idx, indexing="ij")
    xs = (cc.reshape(-1) - npx) / npx
    ys = (rr.reshape(-1) - npx) / npx
    xy_sq = torch.stack([xs, ys], dim=-1)
    dc, _hemi = lambert_to_direction(xy_sq)            # square map → sphere
    uv, ok = _sphere_to_hex(dc)                        # sphere → hex Lambert coord
    # continuous array coordinates in our convention (col=x, row=y):
    cf = uv[:, 0] * npx + npx                          # column (float)
    rf = uv[:, 1] * npx + npx                          # row (float)
    nix = torch.floor(cf).to(torch.int64)
    niy = torch.floor(rf).to(torch.int64)
    nixp = nix + 1
    niyp = niy + 1
    # EMsoft edge clamp (top index is 2*npx == m-1 in array space).
    nixp = torch.where(nixp > (m - 1), nix, nixp)
    niyp = torch.where(niyp > (m - 1), niy, niyp)
    in_bounds = (nix >= 0) & (niy >= 0) & (nix <= m - 1) & (niy <= m - 1)
    valid = ok & in_bounds
    dx = cf - nix.to(torch.float64)
    dy = rf - niy.to(torch.float64)
    dxm = 1.0 - dx
    dym = 1.0 - dy
    g = grid_hex.to(torch.float64)
    cnix = nix.clamp(0, m - 1)
    cniy = niy.clamp(0, m - 1)
    cnixp = nixp.clamp(0, m - 1)
    cniyp = niyp.clamp(0, m - 1)
    val = (
        g[cniy, cnix] * dxm * dym
        + g[cniy, cnixp] * dx * dym
        + g[cniyp, cnix] * dxm * dy
        + g[cniyp, cnixp] * dx * dy
    )
    out = torch.zeros(m * m, dtype=torch.float64, device=device)
    out[valid] = val[valid]
    return out.reshape(m, m)


def _cartesian_to_miller_cubic(
    directions: torch.Tensor, a_nm: float, wavelength_nm: float
) -> torch.Tensor:
    """Cubic Cartesian unit directions → Miller wavevectors with ``|k| = 1/λ``.

    ``k_Miller = (dx, dy, dz) · (a/λ)``.  Valid only for a cubic cell (Cartesian
    axes aligned with crystal axes); see the module docstring.
    """
    return directions * (a_nm / wavelength_nm)


def _cartesian_to_miller(
    directions: torch.Tensor, structure: CrystalStructure, wavelength_nm: float
) -> torch.Tensor:
    """General Cartesian unit directions → Miller wavevectors with ``|k| = 1/λ``.

    Implements EMsoft's k-vector recipe (``kvectors.f90::AddkVector``): a Cartesian
    unit direction ``k̂`` is mapped to reciprocal-fractional (Miller) coordinates via
    ``TransSpace('c'→'r') = dsmᵀ · k̂`` and scaled to ``1/λ``:

        ``k_Miller = (dsmᵀ @ k̂ᵀ)ᵀ / λ``

    where ``dsm`` is the crystal's direct structure matrix
    (:pyattr:`backend.forward_sim.crystal.xtal_io.CrystalStructure.structure_matrix`).
    Because ``rsmᵀ·rsm = g*`` and ``rsm = (dsm⁻¹)ᵀ``, the resulting ``k`` has
    ``|k|² = kᵀ g* k = |k̂|²/λ² = 1/λ²`` for **every** crystal system (verified on
    the hexagonal cell).

    Reduces **bit-identically** to :func:`_cartesian_to_miller_cubic` for a cubic
    cell: there ``dsm = a·I`` so ``dsmᵀ·k̂/λ = (a/λ)·k̂``.  The cubic build path uses
    the scalar function directly (preserving the validated Ni/Al NCC); this matrix
    form is the route for non-cubic cells.

    Args:
        directions: ``(P, 3)`` Cartesian unit directions (any real dtype).
        structure: the crystal (supplies the direct structure matrix ``dsm``).
        wavelength_nm: electron wavelength ``λ`` in nm.

    Returns:
        ``(P, 3)`` Miller wavevectors, same dtype/device as ``directions``.
    """
    dsm = torch.as_tensor(
        structure.structure_matrix, dtype=directions.dtype, device=directions.device
    )  # (3, 3)
    # k_Miller = (dsm^T @ dir^T)^T / λ  ==  dir @ dsm / λ.
    return (directions @ dsm) / wavelength_nm


def build_master(
    structure: CrystalStructure,
    mc: MCData,
    *,
    npx: int = 8,
    energy_idx: int,
    dmin: float = 0.05,
    device: torch.device | str | None = None,
    uniform_lambda: bool = False,
    bethe_params=None,
    batch_size: int = 256,
    absflg: int = 1,
    use_batched: bool = True,
    use_lever3: bool = True,
    lever3_chunk: int = 4_000_000,
    n_workers: int | None = None,
    complex64_eig: bool = True,
    use_propagation: bool | None = None,
    propagation_s1_dtype: torch.dtype = torch.complex128,
    eig_use_flip_guard: bool = True,
    build_a_gpu: bool | None = None,
    dir_chunk: int | str | None = 32_768,
    lut_dtype: torch.dtype = torch.complex128,
    weak_beam_dmin_floor: float = 0.0,
    weak_pert_tau: float = 0.1,
    weak_chunk: int = 64,
    n_cap: int | None = None,
    hemisphere: str = "north",
    use_symmetry: bool = True,
    progress_cb=None,
) -> torch.Tensor:
    """Build the dynamical EBSD master pattern on a hemisphere's Lambert grid.

    For every pixel ``(row, col)`` of the ``(2·npx+1, 2·npx+1)`` modified-Lambert
    square (EMsoft ``mLPNH`` convention), invert the pixel to its northern-
    hemisphere incident-beam direction ``k`` and evaluate

        I(k) = Re{ Σ_{g,h strong}  Lgh(k)[g,h] · Sgh[g,h] } / N_atoms

    where the strong-reflection set is the per-direction Bethe partition of the
    ``dmin``-bounded reflection list, ``Lgh`` is the MC-depth-weighted dynamical
    intensity (EMsoft ``CalcLgh``), and ``Sgh`` the direction-independent structure
    weighting (EMsoft ``CalcSgh``).  ``N_atoms`` is the number of atoms in the
    conventional cell (per-atom normalisation, design §8).

    The master is a **pure crystal property** and is deliberately **not** multiplied
    by the per-direction ``accum_e(k, E)`` back-scatter yield: ``accum_e`` carries
    the tilted-sample MC geometry (upper/lower asymmetric ~15× on Ni 20 kV) that
    belongs to the detector forward-projection, not the master.  NCC is scale/offset
    invariant, so dropping this per-pixel factor needs no replacement normalisation.
    (The depth-absorption weighting from the ``accum_z`` histogram — ``lambdaE`` — is
    a crystal/depth property and *is* retained.)

    Performance / caching.  ``compute_Ug_table`` is invoked through a single
    ``Ug_lookup`` closure shared by every direction, and ``build_A`` already caches
    it over the **unique** difference vectors ``g−h`` (so ``U_g`` is *not*
    recomputed per direction).  ``Sgh`` is direction-independent → computed once
    over the global reflection list and indexed per direction.  The wavelength is
    derived once from the chosen energy bin.

    Args:
        structure: the crystal (lattice + asymmetric-unit atoms + space group).
            **Any crystal system** (SP6): a cubic cell uses the bit-identical scalar
            Cartesian→Miller map (:func:`_cartesian_to_miller_cubic`, preserving the
            validated Ni/Al NCC); a non-cubic cell routes through the general direct
            structure-matrix map (:func:`_cartesian_to_miller`).
        mc: EMsoft Monte-Carlo data — supplies ``λ(z)`` (depth) and ``accum_e``
            (per-direction energy weight).
        npx: Lambert half-grid size; the grid is ``(2·npx+1)²``.  Defaults to a
            **small** value (8 → 17×17) for dev speed; pass 500 to match the oracle.
        energy_idx: index into the MC energy axis selecting the depth profile and
            the ``accum_e`` weight grid; also fixes the wavelength via
            ``mc.EkeVs[energy_idx]``.
        dmin: resolution limit in nm for the reflection list (design default 0.05).
        device: compute device; defaults to :func:`backend.forward_sim.runtime.get_device`.
        uniform_lambda: if True, replace the MC ``λ(z)`` with a flat (uniform)
            depth profile — the geometry-only sanity mode (design §2 ``uniform``;
            G1.2 gate).  Implemented by overriding ``mc.lambda_z`` for this call.
        bethe_params: optional ``[c1, c2, c3, sgdbdiff]`` Bethe cutoffs; defaults to
            the EMsoft code defaults ``(4, 8, 50, 1)``.  Pass the oracle's stored
            ``BetheParameters`` (via ``read_bethe_parameters``) to match it exactly.
        batch_size: number of directions per ``build_A`` / ``depth_integral`` batch
            (memory/throughput knob; does not affect the result).
        absflg: EMsoft absorption flag for the complex potential ``U_g`` (passed to
            :func:`compute_Ug_table` → :func:`wk_scattering_factor`).  ``1`` (default)
            = phonon-only absorption (no behaviour change vs the pre-FCORE master);
            ``3`` = phonon + core-loss (the EMsoft EBSD production setting).  The
            flag affects both the off-diagonal ``U_{g−h}`` coupling and the normal-
            absorption diagonal ``Upz = Im(U_0)``, since both come from the same
            cached ``Ug_lookup``.
        use_batched: if True (default) use the SP4 vectorised Bethe path —
            precompute the direction-independent ``|U_{g−h}|`` row-maxima ONCE
            (:func:`build_ug_lut`) and classify strong/weak for the whole grid in
            batched ``(B, M)`` tensors (:func:`bethe_partition_batched`),
            eliminating the per-direction numpy ``O(M²)`` rebuild that dominated
            the wall time (~73 %).  This is **bit-faithful** to the serial path:
            it only changes HOW the strong/weak masks are computed, not the
            physics.  Set False to run the original per-direction serial Bethe
            loop (the correctness oracle); the result is identical.
        use_lever3: if True (default, only active when ``use_batched`` is also
            True) use the SP4 **lever-3** batched dynamical solve — group the
            inside directions by their Bethe **strong-beam count** ``n`` (ragged,
            typically 10–30) and, for each group, batched-assemble ``A`` as a
            ``(B', n, n)`` tensor (:func:`build_A_batched`), run a SINGLE batched
            ``torch.linalg.eig`` + depth integral (:func:`depth_integrated_Lgh`),
            and scatter the master values back to the grid by direction index.
            This eliminates the per-direction Python loop over ``build_A`` /
            ``eig`` / ``depth_integral`` (the ~43 % of wall time remaining after
            lever 1+2).  The eigendecomposition runs **on CPU** even when
            ``device='cuda'``: ``torch.linalg.eig`` has a known CUDA
            sequential-sync defect (PyTorch #107291) that makes the batched GPU
            eig ~20× SLOWER than the batched CPU LAPACK call for these small
            (``n``≈10–37) matrices — measured on the RTX 4070, see
            ``tasks/forward_sim/lessons.md`` §17.  **Bit-faithful** to the serial
            path: the final ``Lgh`` is a FULL bilinear sum over ALL eigenpairs,
            invariant to the eigenvector ordering/phase the batched ``eig`` may
            pick differently — so the master matches the serial oracle to FP
            precision.  Set False to use the lever-1+2 per-direction loop.
        lever3_chunk: VRAM/throughput knob for lever 3 — a group of ``B'``
            directions sharing strong count ``n`` is processed in chunks sized so
            ``chunk_B'·n·n ≤ lever3_chunk`` (bounds the ``complex128`` working set
            of the batched ``A`` / eigenvectors).  Does not affect the result.
        n_workers: SP4 **iter-3** parallelism knob — number of worker threads the
            lever-3 driver uses to run the per-``(n``-group, chunk``)`` assemble +
            ``eig`` + ``Lgh`` work concurrently.  The chunks are independent (each
            writes a disjoint set of grid pixels), and ``torch.linalg.eig`` on CPU
            LAPACK **releases the GIL**, so a ``ThreadPoolExecutor`` over chunks
            with intra-op threads pinned to 1 scales near-linearly with cores
            (measured ~9× at 12 workers, ~11× at 24 on this 24-thread box; see
            ``tasks/forward_sim/lessons.md`` §18).  **Critically, batched eig is
            run at intra-op = 1**: torch's default 12 intra-op threads make the
            batched LAPACK eig pathologically slow (severe contention — the same
            wall got ~50–100× WORSE), so the parallelism MUST be across the batch,
            not inside LAPACK.  ``None`` (default) → ``min(os.cpu_count(), 24)``.
            ``1`` runs the chunks serially (the lever-3 v1 behaviour).  Does not
            affect the result — the master is identical regardless of worker count
            (the scatter targets are disjoint pixel sets).
        complex64_eig: SP4 **iter-3** precision knob — when True (default), cast the
            dynamical matrix ``A`` to ``complex64`` for the ``eig``/``inv`` step
            only (the eigenpairs are promoted back to ``complex128`` immediately, so
            the depth integral and the ``Lgh·Sgh`` bilinear sum still accumulate in
            float64).  Single-precision eig is ~1.2–1.4× faster.  Gated on the SP4
            correctness test: at npx=15 Ni+Al the complex64 master holds NCC ≥
            0.9999 and max-abs-rel-err < 1e-3 vs the complex128 serial oracle, so it
            is **on by default**; set False to force the full-double eig.
        use_propagation: SP4 **iter-4** GPU-native depth integral — replace the
            eig-based ``depth_integrated_Lgh`` (``torch.linalg.eig``, CUDA-hostile
            #107291, deliberately pinned to CPU) with the **scattering-matrix
            propagation** :func:`depth_integrated_Lgh_propagate` (one batched
            ``torch.matrix_exp`` + ``izz`` batched matvecs + outer-product
            accumulation — **all on the GPU, no eig**).  When ``None`` (default), it
            is turned **ON automatically for a CUDA device** (the whole point:
            finally use the GPU) and OFF on CPU (where the CPU-parallel eig of iter-3
            is already fast and bit-exact).  The propagation evaluates the **same**
            depth integral as the eig path with EMsoft's overflow guard off —
            bit-identical to the no-flip eig integral (~1e-14), NCC ≥ 0.9999 vs the
            production flip-eig master, and **equally faithful to the EMsoft oracle**
            (NCC vs EMsoft identical to 2e-5 — see lessons §19).  The eig path
            (lever 3) stays the correctness oracle/fallback.  Explicit
            ``True``/``False`` overrides the device default.
        propagation_s1_dtype: precision for the single ``S1 = expm(…)`` step of the
            propagation path (``complex128`` default; ``complex64`` is the perf
            knob — the ``izz`` propagation matvecs + accumulation always run in
            complex128 regardless, so depth drift is bounded).
        eig_use_flip_guard: applies only to the **eig** path (``use_propagation``
            False) — EMsoft's ``CalcLgh`` overflow guard (default ``True``,
            production-faithful).  Set ``False`` to compute the *un-guarded* eig
            integral, which is the **exact** integral the scattering-matrix
            propagation evaluates → the two then match to ~1e-14 (the SP4-iter4
            parity oracle).  Does not affect the propagation path.
        build_a_gpu: SP4 **iter-5** — assemble the dynamical matrix ``A`` with the
            **GPU dense-difference-LUT** path (:func:`build_A_batched_gpu`) instead
            of the CPU ``np.unique``+dict ``build_A_batched``.  iter-4 profiling
            showed ``build_A_batched`` is the wall bottleneck (~70 %) and **92 % of
            its own time is ``np.unique``'s ``argsort``** over the ``(B',n,n,3)``
            difference grids.  iter-5 precomputes a tiny dense complex LUT
            (:func:`build_diff_lut`, ≤0.6 MB even at dmin=0.05) once per master and
            gathers ``U_{g−h}`` from it with pure device index ops — the whole
            off-diagonal / weak-beam / diagonal assembly then runs as batched CUDA
            tensor ops (measured **181× faster** than the CPU ``build_A_batched`` on
            the RTX 4070's largest npx=50 group, and bit-faithful to ~5e-13).  Only
            wired into the **propagation** path (``use_propagation`` True), since the
            eig path is pinned to CPU anyway (#107291).  ``None`` (default) → ON when
            ``use_propagation`` is on (i.e. CUDA); the eig/CPU path always uses the
            CPU ``build_A_batched``.  Explicit ``True``/``False`` overrides.
        dir_chunk: SP4 **iter-5 OOM fix** — max directions per chunk in the batched
            Bethe partition (:func:`bethe_partition_batched`).  Its ``(B',M,3)``
            excitation-error working set is ``B'·M·3`` float64; at npx=500/dmin=0.05
            the un-chunked ``(B,M,3)`` would be ~24.9 GiB → OOM.  Default 32 768
            bounds it (~1.1 GiB).  Does not affect the result.
        lut_dtype: complex dtype of the GPU difference-LUT (``complex128`` default;
            ``complex64`` halves the gather/assembly memory traffic — ``A`` is then
            assembled in single precision and the propagation promotes ``S1`` back
            to double, so depth accumulation stays float64).
        weak_beam_dmin_floor: **SUPERSEDED by the EMsoft c3 reflection-list
            pre-filter (iter-10) — default now ``0.0`` (off).**  dmin (nm) below
            which the Bethe **weak-beam** perturbation correction is disabled by
            collapsing the weak window ``c2 → c1`` (the empty-weak / "bare-strong"
            path).  The iter-8 measurement that motivated this floor (Ni [001]
            dmin=0.05 classifying ~68 weak beams that over-correct ~30–40 %) was a
            **symptom** of the real bug: our :func:`reflection_list` kept the full
            geometric ``|g| ≤ 1/dmin`` sphere, whereas EMsoft's
            ``Initialize_ReflectionList`` only lists reflections with self ratio
            ``λ⁻¹·|s_g|/|U_g| ≤ c3`` (=50).  That gate — now applied in
            :func:`bethe_partition` / :func:`bethe_partition_batched` — removes the
            spurious far-from-Ewald reflections that contaminated BOTH the strong
            and weak sets (Ni [001] dmin=0.05: strong 61→25, weak 68→24), so the
            surviving weak beams are EMsoft-faithful and the perturbation is back in
            regime.  With the c3 pre-filter active, emptying the (now-correct) weak
            set would make us *less* faithful than EMsoft (which keeps it), so the
            default is ``0.0`` (never collapse).  The parameter is retained for
            reproducing the pre-iter-10 floor experiment: set ``> dmin`` to still
            collapse ``c2 → c1`` at fine dmin (no longer recommended).  At the coarse
            ``dmin=0.10`` SP1-validation setting the floor does not fire (0.10 ≥ any
            sensible floor), so it cannot regress those gates.
        weak_pert_tau: **Bethe-perturbation-validity guard (iter-12) — SUPERSEDES
            ``weak_beam_dmin_floor``; default ``0.1`` (ON).**  Passed straight to
            :func:`build_A` / :func:`build_A_batched` / :func:`build_A_batched_gpu`.
            Our weak-beam ``weaksum``/``weaksgsum`` math already matches EMsoft
            ``GetDynMat`` mode-``'D'`` **exactly** (same raw signed ``s_w``
            denominator, same ``λ/2`` prefactor — verified vs the develop source),
            so the iter-8 full-res over-correction is NOT a units/factor bug.  It is
            that a dense (fine-``dmin``) reflection list admits many weak beams with
            small ``|s_w|`` whose *collective* correction leaves the perturbation
            regime (per-pair correction ≫ the direct coupling), over-correcting the
            off-diagonal couplings and lowering the full-res NCC.  ``τ`` drops, per
            pair, any single weak-beam term whose magnitude exceeds ``τ`` × the
            direct coupling it perturbs (off-diagonal: ``|U_{g−h}|``; diagonal:
            ``|2·s_g/λ|``), keeping the EMsoft-faithful in-regime terms intact (and
            keeping terms with a zero direct coupling — genuine Umweganregung).
            This is finer than the all-or-nothing ``c2→c1`` floor: it removes only
            the out-of-regime contributions, not the whole weak set.  At the coarse
            ``dmin=0.10`` SP1-validation setting Ni [001] has **0 weak beams**, so
            the guard is a strict **no-op** there (the bare-strong matrix is
            unchanged) → it cannot regress the matched-settings master.  Set ``0.0``
            to apply the raw EMsoft GetDynMat weaksum over the full weak set
            (pre-iter-12 behaviour, the over-correcting path at fine dmin).
        weak_chunk: **iter-13 OOM fix — chunk size for the weak-beam guard
            reduction; default ``64``.**  Passed to :func:`build_A` /
            :func:`build_A_batched` / :func:`build_A_batched_gpu`.  The iter-12
            guard keeps the per-(b,ir,ic,w) weak-term tensor explicit so it can
            drop out-of-regime contributions before the ``Σ_w`` reduction, but
            that ``(B', n, n, W)`` tensor scales with the weak-beam count ``W``,
            which at full resolution (npx=500/dmin=0.05, ``n``~72 strong,
            ``W``~hundreds weak) reaches ~41 GiB → CUDA OOM on the 12 GB RTX 4070.
            ``weak_chunk`` reduces the weak axis ``W`` in chunks of this many
            beams, applying the guard mask per chunk and accumulating the masked
            ``weaksum`` / ``weaksgsum`` incrementally, so the full
            ``(B', n, n, W)`` tensor is **never** materialised (per-chunk working
            set ≈ ``B'·n²·weak_chunk`` complex128 — bounded).  Default ``64`` keeps
            the per-chunk working set well within VRAM at full res; raise it
            (e.g. 256) when ``W`` is small / memory is ample for fewer reduction
            steps, or set ``0`` to reduce all ``W`` at once (the pre-iter-13 path
            that OOMs at full res).  **Bit-faithful** to the unchunked guard (same
            ``τ``; a single chunk ``weak_chunk ≥ W`` is the unchunked path
            exactly, smaller chunks match to FP precision).  At the coarse
            ``dmin=0.10`` SP1 validation Ni has 0 weak beams → no-op (the chunk
            loop never runs), so it cannot regress the matched-settings master.
        n_cap: **adaptive Bethe strong-beam cap (iter-15) — opt-in, default
            ``None`` = OFF (bit-identical to the un-capped master).**  Forwarded to
            the batched Bethe partition (:func:`bethe_partition_batched`).  On large
            cells at the production ``dmin`` the per-direction strong count ``n`` is
            uniformly high (T-phase 162-atom @ dmin=0.10: ``n`` ≈ 132 with no cheap
            tail), and the dense solve scales ``~n³`` (``matrix_exp``/``eig``),
            dominating wall AND VRAM.  ``n_cap`` (an int ``≥ 1``) caps the strong set
            per direction: after the standard ``c1`` partition, the strong beams
            CLOSEST to the strong/weak boundary (largest Bethe ratio ``m``) are
            DEMOTED into the weak (perturbative) set until the count is ``n_cap``
            (the transmitted beam is always kept strong).  Those marginal beams are
            exactly the ones Bethe perturbation handles accurately, so the master NCC
            is ~unchanged while ``n³`` drops by ``~(n/n_cap)³``.  ``None`` (default)
            performs NO demotion — the master is bit-identical to the un-capped path.
            Threaded into **both** the batched partition
            (:func:`bethe_partition_batched`, ``use_batched=True``, the default) AND
            the serial oracle partition (:func:`bethe_partition`,
            ``use_batched=False``) via the SHARED :func:`_apply_n_cap` demotion, so a
            capped serial build is bit-faithful to a capped batched build (the
            autopilot's batched==serial correctness gate).  ``ValueError`` if ``< 1``.
        hemisphere: ``"north"`` (default) evaluates the **northern**-hemisphere
            Lambert grid directions (``z ≥ 0``) — the EMsoft ``mLPNH`` grid and the
            sole behaviour of every existing (cubic) caller, so the cubic path is
            unchanged.  ``"south"`` returns the EMsoft ``mLPSH`` grid.  For a
            **centrosymmetric** crystal (the cubic Ni/Al gate, and every Laue-class
            cell) the master obeys ``I(k) == I(−k)``, so ``mLPSH`` is the **bit-
            identical inversion** of ``mLPNH`` (``mLPNH[::-1, ::-1]``): the southern
            request DERIVES the grid from a single northern build + an in-plane flip
            (deterministic — an independent southern eigensolve would differ from NH
            by ~1e-6 FP-ordering noise, SP6-iter1's NH-vs-SH NCC 0.9995).  For a
            **non-centrosymmetric** crystal (NH ≠ SH) the southern request runs a
            genuine build with each grid direction's z-component negated **before**
            the Cartesian→Miller map (a true independent ``mLPSH``), at ~2× cost.

            **σ_h non-centrosymmetric hex/trig caveat (SHFIX, 2026-06-18).**  For
            non-centrosymmetric hexagonal/trigonal point groups that *contain* a
            horizontal mirror σ_h (``-6m2``/``-6``/``6/m`` — e.g. π-AlFeSi, SG189),
            the crystal is physically NH = SH, so the z-negated solve above collapses
            **bit-identically** to NH (``our_SH ≡ our_NH``) — which exactly matches
            EMsoft's σ_h symmetry scatter (``EMEBSDmaster.f90:893`` writes the same
            ``svals`` to the SH pixel; the oracle's own ``mLPSH − mLPNH`` is **0.0**
            across the whole interior).  The only residual vs the EMsoft ``mLPSH``
            oracle is a **≤0.005-NCC artifact on the single outermost inscribed-disc
            ring** (≈452/7845 px at npx=50): EMsoft derives that ring by an
            integer-rounded per-direction scatter (``nint(npx·LambertSphereToHex)``,
            ``Lambert.f90:2328-2331``) plus a hex→square bilinear that lands the σ_h
            partner on a *neighbouring* pixel, whereas our vectorised z-negation
            reproduces the NH ring instead.  Measured pi: ``our_SH↔oracle_SH`` 0.993
            full-disc → **0.998 (= NH parity) when the outer 1-px ring is dropped**.
            This is below the SP-gate (≥0.95) by a wide margin and is invisible in any
            downstream EBSD pattern projection (the disc rim maps to the extreme
            detector periphery).  Closing it would require re-implementing EMsoft's
            integer-rounded per-direction scatter at full hex-grid resolution — a
            large, fragile rewrite with real regression risk to the validated NH
            masters (commit 8d537d8) and the centrosymmetric ``SH == NH`` guarantee —
            for ≤0.005 NCC on the disc rim, so it is **left as-is by design**.
            (Polar non-centro groups without σ_h, e.g. ZnS wurtzite ``6mm`` SG186,
            keep their genuine NH ≠ SH: ``our_SH↔oracle_SH`` 0.997, ``our NH↔SH``
            0.927 ≈ oracle 0.925.)
        use_symmetry: **SP-PERF lever 1 — point-group symmetry-orbit reduction;
            default ``True``.**  The master value ``I(d)`` is invariant under every
            point-group operation ``R`` (``A`` at ``R·d`` is a permutation of ``A``
            at ``d`` → identical eigenvalues → identical ``I``), exactly the symmetry
            EMsoft uses when it solves only the irreducible Lambert fundamental zone
            and scatters via ``Apply3DPGSymmetry``.  When ``True`` the per-direction
            eigensolve runs only on ONE representative pixel per point-group orbit
            (:func:`backend.forward_sim.dynamical.symmetry_orbit.compute_orbit_reduction`)
            and the computed value is scattered to every pixel in the orbit — cutting
            the expensive solve count by ~8× (cubic/tetragonal), ~8× (hexagonal), ~2×
            (monoclinic), 1× (triclinic, no win).  This is correctness-EXACT up to the
            ``nint`` pixel-quantisation EMsoft itself incurs (self-test gate: the
            symmetry master matches the full-disc master at NCC > 0.9999, bit-faithful
            in the interior; the oracle NCC vs EMsoft is preserved).  Set ``False`` to
            solve **every** inside-disc direction (the full-disc path) — the
            correctness oracle and fallback; both paths produce a master pixel-aligned
            with EMsoft ``mLPNH``/``mLPSH``.  No-op for triclinic (orbit size 1).
        progress_cb: optional callable ``progress_cb(directions_done,
            directions_total)`` invoked after each direction chunk completes
            (monotonically increasing ``directions_done``, ending at
            ``directions_total`` = number of inside-disc directions).  Purely a
            UI hook — it is a strict **no-op** when ``None`` (default) and changes
            **zero** numerics: the master array is bit-identical with or without
            it (it is called only to report progress, never read).

    Returns:
        ``(2·npx+1, 2·npx+1)`` ``float32`` tensor on ``device`` — the requested
        hemisphere's master pattern, pixel-aligned with EMsoft
        ``mLPNH[0, energy_idx]`` (``hemisphere="north"``) or
        ``mLPSH[0, energy_idx]`` (``hemisphere="south"``).  The FULL square is
        populated (corners included) — they map to valid equatorial directions
        EMsoft also fills (corner fix 2026-06-22).

    Raises:
        ValueError: if ``npx`` < 1, ``energy_idx`` is out of range, ``hemisphere``
            is not ``"north"``/``"south"``, ``n_cap`` < 1, or the lattice is
            degenerate (``sin γ`` ≈ 0).
    """
    if npx < 1:
        raise ValueError(f"npx must be >= 1, got {npx}")
    if not 0 <= energy_idx < mc.n_energy:
        raise ValueError(
            f"energy_idx {energy_idx} out of range [0, {mc.n_energy})"
        )
    if hemisphere not in ("north", "south"):
        raise ValueError(
            f"hemisphere must be 'north' or 'south', got {hemisphere!r}"
        )
    if n_cap is not None and int(n_cap) < 1:
        raise ValueError(f"n_cap must be >= 1 or None, got {n_cap!r}")

    # --- Centrosymmetric southern hemisphere = inversion of the northern grid ----
    # SP6-iter2 Gap 2.  A centrosymmetric crystal obeys I(k) == I(-k); on the
    # Lambert grid the southern pixel SH(x, y, −z) therefore equals NH(−x, −y, z),
    # which is the northern grid flipped along BOTH axes (the pixel for (−x, −y, z)
    # is (2·npx−row, 2·npx−col)).  Deriving SH from the already-computed NH this way
    # is exact and DETERMINISTIC — an *independent* southern eigensolve would differ
    # from NH by ~1e-6 FP-ordering noise (SP6-iter1's NH-vs-SH NCC 0.9995), whereas
    # the flip is bit-identical.  The northern path is completely unchanged (the
    # recursion re-enters with hemisphere="north"), so every existing cubic caller
    # is untouched.  Non-centrosymmetric crystals fall through to a genuine southern
    # build (NH ≠ SH).
    if hemisphere == "south" and structure.is_centrosymmetric:
        nh = build_master(
            structure, mc,
            npx=npx, energy_idx=energy_idx, dmin=dmin, device=device,
            uniform_lambda=uniform_lambda, bethe_params=bethe_params,
            batch_size=batch_size, absflg=absflg, use_batched=use_batched,
            use_lever3=use_lever3, lever3_chunk=lever3_chunk, n_workers=n_workers,
            complex64_eig=complex64_eig, use_propagation=use_propagation,
            propagation_s1_dtype=propagation_s1_dtype,
            eig_use_flip_guard=eig_use_flip_guard, build_a_gpu=build_a_gpu,
            dir_chunk=dir_chunk, lut_dtype=lut_dtype,
            weak_beam_dmin_floor=weak_beam_dmin_floor, weak_pert_tau=weak_pert_tau,
            weak_chunk=weak_chunk, n_cap=n_cap, hemisphere="north",
            use_symmetry=use_symmetry,
            progress_cb=progress_cb,
        )
        return nh.flip([0, 1]).contiguous()

    dev = torch.device(device) if device is not None else get_device()
    a_nm = float(structure.lattice[0])

    # SP4 iter-4: GPU-native scattering-matrix propagation depth integral.  On a
    # CUDA device this is ON by default (the whole point — the eig path is pinned
    # to CPU by #107291, so the GPU sat idle); on CPU it is OFF (iter-3's
    # CPU-parallel eig is already fast + bit-exact).  Explicit flag overrides.
    if use_propagation is None:
        use_propagation = dev.type == "cuda"

    # SP4 iter-5: GPU dense-LUT build_A — assemble A with batched device tensor
    # ops gathered from a precomputed difference-LUT, replacing the CPU
    # np.unique+dict build_A_batched (the iter-4 wall bottleneck).  Only used on
    # the propagation path (the eig path runs on CPU by necessity, #107291); ON by
    # default whenever propagation is on (a CUDA device).
    if build_a_gpu is None:
        build_a_gpu = bool(use_propagation) and dev.type == "cuda"

    # Wavelength from the selected energy bin (relativistic, EMsoft CalcWaveLength).
    voltage_kV = float(mc.EkeVs[energy_idx].item())
    wavelength_nm = relativistic_wavelength_nm(voltage_kV)

    gstar = structure.reciprocal_metric  # (3, 3) reciprocal metric, nm⁻²

    bp = _DEFAULT_BETHE if bethe_params is None else bethe_params

    # --- Disable the out-of-regime Bethe weak-beam correction at fine dmin ------
    # At a dense reflection list (fine ``dmin``) many reflections fall in the Bethe
    # weak band ``c1 < m ≤ c2`` and their collective weaksum/weaksgsum correction
    # leaves the perturbation regime (correction ~30–40 % of the direct coupling,
    # not the ≲5 % the perturbation theory assumes), systematically over-correcting
    # the off-diagonal couplings and *lowering* the full-res master NCC vs the
    # EMsoft oracle.  Collapse the weak window ``c2 → c1`` → the weak set is empty
    # (``m > c1 & m ≤ c1`` is never satisfied) → the bare-strong path.  Keyed on the
    # physical regime indicator ``dmin`` (NOT a phase/value hardcode); at the coarse
    # ``dmin`` of the SP1 matched-settings validation the weak set is already empty,
    # so this is a no-op there.  See the ``weak_beam_dmin_floor`` docstring.
    bp = list(bp)
    if (
        weak_beam_dmin_floor > 0.0
        and float(dmin) < float(weak_beam_dmin_floor)
        and len(bp) >= 2
    ):
        bp[1] = bp[0]  # c2 = c1  → empty weak window (bare-strong path)

    # --- Optionally swap in a uniform depth profile (geometry-only sanity) ------
    mc_used = mc
    if uniform_lambda:
        mc_used = _UniformLambdaMC(mc)

    # --- Global reflection list + cached U_g lookup -----------------------------
    reflections = reflection_list(structure, dmin)            # (M, 3) int, no g=0
    # Prepend the transmitted beam g=0 as index 0 (required by Bethe + Lgh).
    zero = torch.zeros(1, 3, dtype=reflections.dtype)
    all_refl = torch.cat([zero, reflections], dim=0)          # (M+1, 3)

    # --- Adaptive dir_chunk (large-cell OOM guard, iter-16) -------------------
    # The per-chunk (B, M) Bethe tensors (3 float64 arrays: k+g, 2k+g, sg) are
    # each B×M×3×8 bytes.  For a large cell (M≈97k at dmin=0.05) a chunk of
    # 32 768 directions would need 32768×97569×3×8 ≈ 77 GB — an OOM crash.
    # When dir_chunk is None or 'auto', compute a VRAM/RAM-safe value from M:
    # target ≈ 10 GB → B = 10e9 / (M × 3 × 8) directions per chunk.  The iter-16
    # harness used 448 for the T-phase (M≈97k), which is within this budget.
    # Small cells (M≤11000) keep the existing default (32 768) unchanged.
    _M = all_refl.shape[0]
    if dir_chunk is None or dir_chunk == "auto":
        _bytes_per_dir = _M * 3 * 8  # float64, 3 arrays: k+g, 2k+g, sg
        _target_bytes = 10 * 1024**3   # 10 GB budget
        _auto = max(1, int(_target_bytes / _bytes_per_dir))
        dir_chunk = min(_auto, 32_768)  # don't exceed the small-cell default
    dir_chunk = int(dir_chunk)

    # --- Memoised U_g lookup (CACHED across ALL directions) ---------------------
    # The per-direction Bethe partition and build_A both query U_{g−h} over the
    # difference vectors of the strong set.  The SAME difference vectors recur
    # across directions, and compute_Ug_table internally calls the slow scipy
    # phonon-absorption integral (`_fphon`) per unique |g| — so recomputing it per
    # direction is the dominant cost.  We memoise by the full (h,k,l) tuple (U_g
    # depends on g through both |g| AND the phase exp(−i2π g·r), so |g| alone is an
    # insufficient key), computing each U_g exactly once for the whole grid.  This
    # is the explicit caching the Task-11 brief mandates ("CACHE compute_Ug over
    # unique |g| … do NOT recompute per direction").
    _ug_cache: dict[tuple[int, int, int], complex] = {}
    # The lever-3 driver may call ``Ug_lookup`` from multiple worker threads
    # (build_A_batched queries new difference vectors per chunk).  CPython dict
    # *reads* are GIL-safe, but a concurrent miss-compute-and-store would race;
    # guard only the miss path with a lock (uncontended after the LUT warm-up).
    _ug_lock = threading.Lock()

    def Ug_lookup(hkl_diffs: torch.Tensor) -> torch.Tensor:
        keys = np.asarray(hkl_diffs.detach().cpu(), dtype=np.int64).reshape(-1, 3)
        miss_mask = np.array(
            [(int(h), int(k), int(l)) not in _ug_cache for (h, k, l) in keys]
        )
        if miss_mask.any():
            with _ug_lock:
                # Re-check under the lock (another thread may have filled these).
                miss_keys = keys[miss_mask]
                uniq_miss = np.unique(miss_keys, axis=0)
                still_miss = np.array(
                    [
                        (int(h), int(k), int(l)) not in _ug_cache
                        for (h, k, l) in uniq_miss
                    ]
                )
                if still_miss.any():
                    todo = uniq_miss[still_miss]
                    # Compute only the not-yet-seen difference vectors (a single
                    # batched compute_Ug_table over the unique misses).
                    u_uniq = compute_Ug_table(
                        structure,
                        torch.from_numpy(np.ascontiguousarray(todo)),
                        voltage_kV,
                        absflg=absflg,
                    )
                    u_uniq_np = u_uniq.detach().cpu().numpy()
                    for row, key in enumerate(todo):
                        _ug_cache[(int(key[0]), int(key[1]), int(key[2]))] = (
                            complex(u_uniq_np[row])
                        )
        out = torch.tensor(
            [_ug_cache[(int(h), int(k), int(l))] for (h, k, l) in keys],
            dtype=torch.complex128,
        )
        return out

    # --- Direction-independent Sgh over the FULL reflection list ----------------
    # Sgh is indexed [ir, ic] by reflection; per direction we slice to the strong
    # subset.  Computing it once over all reflections avoids recomputing the
    # symmetry-orbit phase sums for every direction.
    # For large cells (M > _SGH_DENSE_M_THRESHOLD) the dense (M,M) matrix would
    # exceed ~1.9 GB; compute_Sgh_for_build_master returns a lazy proxy instead
    # that computes only the per-direction strong sub-block on demand (iter-16).
    #
    # Lever 2b-2: on the GPU box-gather path (lever-3 + propagation + GPU build_A)
    # the dense (M,M) ``compute_Sgh`` is the SETUP wall (3.28 s on tau2 — a Python
    # ×n_atoms loop over a full (M,M) einsum each).  Since ``Sgh[ir,ic]`` depends
    # ONLY on ``d = g_ic − g_ir`` it is built ONCE as a tiny ``(D,D,D)`` difference
    # box inside ``_lever3_eval`` (:func:`build_sgh_box`) and gathered per chunk —
    # so we SKIP the dense build entirely here (bit-identical, proven Δ=0.0).  The
    # serial fallback + the eig path still consume the dense ``sgh_full``.
    _sgh_box_path = bool(
        use_batched and use_lever3 and build_a_gpu and use_propagation
    )
    if _sgh_box_path:
        sgh_full = None  # built lazily in _lever3_eval as a difference box
    else:
        sgh_full = compute_Sgh_for_build_master(  # (M+1, M+1) or lazy proxy
            structure, all_refl, voltage_kV, dev
        )

    # Per-atom normalisation: number of atoms in the conventional cell.
    from ..crystal.structure_matrix import _expand_symmetry_orbit

    n_atoms = float(len(_expand_symmetry_orbit(structure)))

    # --- Normal absorption Upz + lambdaE depth-absorption factor ----------------
    # GetDynMat 'D' places Upz = rlp%Upmod = Im(U_0) on every diagonal imaginary
    # part (the normal-absorption coefficient).  EMEBSDmaster.f90 additionally
    # re-weights the MC depth histogram by exp(2π·(iz−1)·depthstep / xgp), with the
    # absorption length xgp = 1/(λ·Upmod) (diffraction.f90::CalcUcg).  MCData.lambda_z
    # stays the pure histogram; this absorption factor is applied HERE and fed to
    # CalcLgh (depth_integrated_Lgh) as ``depth_weight``.
    u0 = Ug_lookup(torch.zeros(1, 3, dtype=torch.int64))[0]
    upz = abs(float(u0.imag))  # rlp%Upmod (nm⁻²)

    depth_weight = None
    if not uniform_lambda:
        lam_hist = mc_used.lambda_z(energy_idx).to(dev).to(torch.float64)  # (izz,)
        if upz > 0.0:
            nabsl = 1.0 / (wavelength_nm * upz)  # xgp absorption length (nm)
            iz0 = torch.arange(
                lam_hist.shape[0], dtype=torch.float64, device=dev
            )  # (iz − 1): 0-based depth index, matching EMsoft's (iz-1)
            depth_weight = lam_hist * torch.exp(
                (2.0 * math.pi) * iz0 * float(mc_used.depth_step) / nabsl
            )
        else:
            depth_weight = lam_hist

    # --- Lambert grid directions (a hemisphere) ---------------------------------
    # ``_grid_directions`` always yields the northern-hemisphere (z ≥ 0) Cartesian
    # directions, one per Lambert pixel.  For ``hemisphere="south"`` we negate the
    # z-component → the same pixels evaluated on the lower hemisphere (a true
    # EMsoft mLPSH grid, used for non-centrosymmetric crystals where NH ≠ SH).
    #
    # SP6 hexagonal fix (HEXROOT): EMsoft solves hex/trigonal (``usehex``) cells on
    # a sheared-60° HEXAGONAL Lambert grid, then bilinearly resamples that grid onto
    # the stored SQUARE grid (SEM/EMEBSDmaster.f90:921-963).  Our square-grid direct
    # solve omits this, costing ~0.76 NCC on γ=120° cells while leaving cubic/
    # tetragonal (γ=90°, square path) at 0.998.  For a hex cell we mirror EMsoft:
    # Stage 1 builds directions on the hex grid (``_grid_directions_hex``) and runs
    # the *unchanged* dynamical solve there; Stage 2 (after the solve fills ``out``)
    # resamples to the square grid (``_hex_to_square_resample``).  Non-hex cells take
    # the bit-identical square path → zero regression.
    hex_cell = _is_hexagonal(structure)
    if hex_cell:
        directions_cart, inside = _grid_directions_hex(npx, dev)  # (P, 3), (P,)
    else:
        directions_cart, inside = _grid_directions(npx, dev)      # (P, 3), (P,)
    if hemisphere == "south":
        # SHFIX caveat (see the ``hemisphere`` docstring): for σ_h non-centro hex/trig
        # groups this z-negation collapses bit-identically to NH (correct — EMsoft's
        # σ_h scatter does the same in the interior); the only residual vs the oracle
        # is a ≤0.005-NCC artifact on the outermost inscribed-disc ring (EMsoft's
        # integer-rounded per-pixel scatter), left as-is by design.
        directions_cart = directions_cart.clone()
        directions_cart[:, 2] = -directions_cart[:, 2]
    m = 2 * npx + 1
    out = torch.zeros(m * m, dtype=torch.float64, device=dev)

    inside_idx = torch.nonzero(inside, as_tuple=False).reshape(-1)
    if inside_idx.numel() == 0:
        # Degenerate grid (no inside pixels): the all-zero ``out`` is unchanged by
        # the hex Stage-2 resample (a bilinear blend of zeros is zero), so this
        # short-circuit is resample-invariant and intentionally skips the gate.
        return out.reshape(m, m).to(torch.float32)

    # --- SP-PERF lever 1: point-group symmetry-orbit reduction ------------------
    # I(d) is invariant under every point-group operation R (A at R·d is a
    # permutation of A at d → identical eigenvalues → identical I), so we solve
    # ONE representative pixel per orbit and scatter the value to all orbit members
    # (EMsoft's Apply3DPGSymmetry strategy).  ``inside_idx`` is restricted to the
    # representatives for the entire solve; ``_orbit_scatter`` (src→dst pixel pairs)
    # copies each representative's value to its orbit AFTER the solve fills ``out``,
    # BEFORE the hex Stage-2 resample.  No-op for triclinic (every orbit size 1).
    _orbit_scatter = None
    if use_symmetry:
        from .symmetry_orbit import compute_orbit_reduction

        rep_idx, _scatter_src, _scatter_dst = compute_orbit_reduction(
            structure, directions_cart, inside, npx,
            hemisphere=hemisphere,
            grid="hex" if hex_cell else "square",
        )
        if rep_idx.numel() > 0 and rep_idx.numel() < inside_idx.numel():
            inside_idx = rep_idx.to(inside_idx.device)
            _orbit_scatter = (
                _scatter_src.to(out.device),
                _scatter_dst.to(out.device),
            )

    # Miller wavevectors for the inside pixels.  A cubic cell uses the bit-identical
    # scalar map (preserves the validated Ni/Al NCC); a non-cubic cell routes
    # through the general direct-structure-matrix map (dsmᵀ·k̂/λ).
    inside_dirs = directions_cart[inside_idx]
    if _is_cubic(structure):
        k_miller_all = _cartesian_to_miller_cubic(
            inside_dirs, a_nm, wavelength_nm
        )
    else:
        k_miller_all = _cartesian_to_miller(
            inside_dirs, structure, wavelength_nm
        )

    # --- SP4 lever 1+2: precompute the direction-independent Bethe row-maxima ----
    # and classify strong/weak for the ENTIRE inside grid in batched (B, M)
    # tensors, hoisting the per-direction numpy O(M²) |U|-grid rebuild (profiled
    # at ~73 % of wall time) out of the loop.  Bit-faithful to the serial path:
    # only HOW the masks are computed changes (see build_ug_lut /
    # bethe_partition_batched).  ``strong_masks_all[i]`` / ``weak_masks_all[i]``
    # are the (M,) masks for inside direction ``i`` (same order as inside_idx).
    # Lever 2b-1/2b-3: build the U difference box ONCE on the GPU box path and
    # share it across build_ug_lut (drop np.unique(M²)) AND _lever3_eval's build_A
    # (which otherwise rebuilds the same box).  The box is tiny ((D,D,D) complex,
    # ≤0.6 MB even at dmin=0.05) and lives on ``dev``.
    shared_diff_lut = None
    shared_diff_lut_H = 0
    if _sgh_box_path:
        shared_diff_lut, shared_diff_lut_H = build_diff_lut(
            all_refl, Ug_lookup, dev, dtype=lut_dtype
        )

    strong_masks_all = None
    weak_masks_all = None
    if use_batched:
        # On the GPU box path, gather the Bethe row-maxima from the shared box
        # (no np.unique(M²)); otherwise the original M-threshold path.
        if shared_diff_lut is not None and lut_dtype == torch.complex128:
            max_u_row, all_zero_row, u_self_row = build_ug_lut(
                all_refl, Ug_lookup,
                diff_lut=shared_diff_lut, diff_lut_H=shared_diff_lut_H,
            )
        else:
            max_u_row, all_zero_row, u_self_row = build_ug_lut(all_refl, Ug_lookup)
        strong_masks_all, weak_masks_all = bethe_partition_batched(
            all_refl,
            k_miller_all,
            max_u_row,
            all_zero_row,
            bp,
            reciprocal_metric=gstar,
            wavelength_nm=wavelength_nm,
            dir_chunk=dir_chunk,
            u_self=u_self_row,
            n_cap=n_cap,
        )                                                     # (B, M) each, CPU

    # --- SP4 lever 3: batched dynamical solve grouped by strong-beam count ------
    # Replace the per-direction Python loop over build_A + eig + depth_integral
    # with a per-n-group batched solve (assemble (B', n, n) A, ONE batched eig on
    # CPU, batched depth integral, scatter back).  Bit-faithful to the serial path
    # (the master is a full-bilinear eigenpair sum, invariant to eig ordering).
    if use_batched and use_lever3:
        _lever3_eval(
            out=out,
            inside_idx=inside_idx,
            k_miller_all=k_miller_all,
            strong_masks_all=strong_masks_all,
            weak_masks_all=weak_masks_all,
            all_refl=all_refl,
            Ug_lookup=Ug_lookup,
            sgh_full=sgh_full,
            structure=structure,
            voltage_kV=voltage_kV,
            sgh_box_path=_sgh_box_path,
            shared_diff_lut=shared_diff_lut,
            shared_diff_lut_H=shared_diff_lut_H,
            gstar=gstar,
            wavelength_nm=wavelength_nm,
            upz=upz,
            n_atoms=n_atoms,
            mc_used=mc_used,
            energy_idx=energy_idx,
            depth_weight=depth_weight,
            dev=dev,
            chunk_cells=int(lever3_chunk),
            n_workers=n_workers,
            complex64_eig=complex64_eig,
            use_propagation=use_propagation,
            propagation_s1_dtype=propagation_s1_dtype,
            eig_use_flip_guard=eig_use_flip_guard,
            build_a_gpu=bool(build_a_gpu),
            lut_dtype=lut_dtype,
            weak_pert_tau=weak_pert_tau,
            weak_chunk=weak_chunk,
            progress_cb=progress_cb,
        )
        if _orbit_scatter is not None:
            src, dst = _orbit_scatter
            out[dst] = out[src]
        grid = out.reshape(m, m)
        if hex_cell:
            grid = _hex_to_square_resample(grid, npx)  # SP6 hex Stage 2
        return grid.to(torch.float32)

    # --- Batched master evaluation over inside directions -----------------------
    n_inside = inside_idx.shape[0]
    if progress_cb is not None:
        progress_cb(0, int(n_inside))
    for start in range(0, n_inside, batch_size):
        stop = min(start + batch_size, n_inside)
        batch_global = inside_idx[start:stop]
        k_batch = k_miller_all[start:stop]                    # (b, 3)

        vals = torch.empty(stop - start, dtype=torch.float64, device=dev)
        for bi in range(stop - start):
            k = k_batch[bi]
            if use_batched:
                # Precomputed batched masks (bit-identical to bethe_partition).
                strong_mask = strong_masks_all[start + bi]
                weak_mask = weak_masks_all[start + bi]
            else:
                # Serial oracle path — thread n_cap through too (iter-15b: was a
                # SILENT no-op here, a fail-loud violation; now a capped serial
                # build is bit-faithful to a capped batched build, the autopilot's
                # batched==serial correctness gate).
                strong_mask, weak_mask = bethe_partition(
                    all_refl,
                    k,
                    Ug_lookup,
                    bp,
                    reciprocal_metric=gstar,
                    wavelength_nm=wavelength_nm,
                    n_cap=n_cap,
                )
            strong_refl = all_refl[strong_mask]               # (n, 3), g=0 first
            weak_refl = all_refl[weak_mask]                   # (W, 3) Bethe weak set
            # Index into the precomputed full Sgh for this strong subset.
            # strong_mask comes from bethe_partition on CPU, but sgh_full lives on
            # `dev` (possibly CUDA); index_select requires sidx on the same device.
            sidx = torch.nonzero(strong_mask, as_tuple=False).reshape(-1)
            sidx = sidx.to(sgh_full.device)
            sgh = sgh_full.index_select(0, sidx).index_select(1, sidx)  # (n, n)

            # Bethe weak-beam excitation errors s_w (nm⁻¹) for this direction; the
            # GetDynMat 'D' weaksum/weaksgsum corrections need 1/s_w.
            if weak_refl.shape[0] > 0:
                weak_sg = excitation_error(weak_refl, k, gstar)
                weak_arg = weak_refl
            else:
                weak_sg = None
                weak_arg = None

            A = build_A(
                k,
                strong_refl,
                Ug_lookup,
                reciprocal_metric=gstar,
                wavelength_nm=wavelength_nm,
                Upz=upz,
                weak_refl=weak_arg,
                weak_sg=weak_sg,
                weak_pert_tau=weak_pert_tau,
                weak_chunk=weak_chunk,
            )                                                 # (1, n, n)
            # build_A assembles A on CPU (its U-lookup + excitation-error math run
            # through numpy); move it onto `dev` so the eigen-integral and the
            # downstream Lgh·Sgh contraction (sgh lives on `dev`) all share one
            # device — otherwise the CUDA path raises a two-devices RuntimeError.
            A = A.to(dev)
            Lgh = depth_integrated_Lgh(
                A,
                mc_used,
                energy_idx,
                wavelength_nm=wavelength_nm,
                depth_weight=depth_weight,
            )[0]                                              # (n, n)
            # I(k) = Re{ Σ_{g,h} Lgh[g,h]·Sgh[g,h] } / N_atoms  (Hadamard sum).
            val = torch.sum(Lgh * sgh.to(Lgh.dtype)).real / n_atoms
            vals[bi] = val

        # The master is a pure crystal property — store I(k) with NO per-pixel
        # accum_e weight (see module/build_master docstrings: accum_e carries the
        # tilted-sample detector geometry, not the master).
        out[batch_global] = vals
        if progress_cb is not None:
            progress_cb(int(stop), int(n_inside))

    if _orbit_scatter is not None:
        src, dst = _orbit_scatter
        out[dst] = out[src]
    grid = out.reshape(m, m)
    if hex_cell:
        grid = _hex_to_square_resample(grid, npx)  # SP6 hex Stage 2
    return grid.to(torch.float32)


def _lever3_eval(
    *,
    out: torch.Tensor,
    inside_idx: torch.Tensor,
    k_miller_all: torch.Tensor,
    strong_masks_all: torch.Tensor,
    weak_masks_all: torch.Tensor,
    all_refl: torch.Tensor,
    Ug_lookup,
    sgh_full: torch.Tensor,
    structure=None,
    voltage_kV: float = 0.0,
    sgh_box_path: bool = False,
    shared_diff_lut: torch.Tensor | None = None,
    shared_diff_lut_H: int = 0,
    gstar,
    wavelength_nm: float,
    upz: float,
    n_atoms: float,
    mc_used,
    energy_idx: int,
    depth_weight,
    dev: torch.device,
    chunk_cells: int,
    n_workers: int | None = None,
    complex64_eig: bool = True,
    use_propagation: bool = False,
    propagation_s1_dtype: torch.dtype = torch.complex128,
    eig_use_flip_guard: bool = True,
    build_a_gpu: bool = False,
    lut_dtype: torch.dtype = torch.complex128,
    weak_pert_tau: float = 0.1,
    weak_chunk: int = 64,
    progress_cb=None,
) -> None:
    """SP4 lever 3 — batched dynamical solve grouped by strong-beam count ``n``,
    parallelised across CPU cores (SP4 **iter-3**), or via GPU scattering-matrix
    propagation (SP4 **iter-4**, ``use_propagation=True``).

    Replaces ``build_master``'s per-direction ``build_A`` → ``eig`` →
    ``depth_integral`` loop.  The inside directions are partitioned by their Bethe
    strong-beam count ``n``; each ``n``-group is chunked so ``B'·n·n`` ≤
    ``chunk_cells`` (bounds the working set).  Each chunk is an **independent**
    unit of work — it assembles its own ``(B', n, n)`` ``A`` (:func:`build_A_batched`),
    runs a batched ``torch.linalg.eig`` + depth integral
    (:func:`depth_integrated_Lgh`), forms ``I(k) = Re{Σ Lgh·Sgh}/N_atoms``, and
    writes to a **disjoint** set of grid pixels.

    **iter-3 parallelism.** The chunks are dispatched to a
    :class:`~concurrent.futures.ThreadPoolExecutor` of ``n_workers`` threads
    (default ``min(os.cpu_count(), 24)``).  ``torch.linalg.eig`` on CPU LAPACK
    releases the GIL, so this scales near-linearly with cores.  Inside the parallel
    region torch's **intra-op thread count is pinned to 1** (restored afterward):
    the default 12 intra-op threads make the *batched* LAPACK eig pathologically
    slow (severe contention — measured ~50–100× worse wall), so the parallelism
    must live across the batch, not inside LAPACK.  ``n_workers=1`` runs serially
    (the lever-3 v1 path).

    The batched ``eig`` runs on **CPU** even when ``dev`` is CUDA: ``torch.linalg.eig``
    has a known CUDA sequential-sync defect (PyTorch #107291) making the GPU batched
    eig ~20× slower than CPU LAPACK for ``n``≈10–37 (measured RTX 4070; lessons §17).
    The Sgh contraction is therefore also done on CPU; only the resulting scalar
    master values are written to ``out`` (on ``dev``).

    Bit-faithful: ``Lgh`` is a full bilinear sum over all eigenpairs, invariant to
    the eigenvector ordering/phase a batched eig may differ on; the per-chunk
    scatter targets are disjoint, so the result is independent of ``n_workers`` and
    of the order chunks complete.  ``complex64_eig`` casts ``A`` to ``complex64``
    for the eig/inv only (eigenpairs promoted back to ``complex128``; the depth
    integral accumulates in float64), gated on the SP4 correctness test.

    ``weak_chunk`` (iter-13 OOM fix) is forwarded to the per-chunk ``build_A_*``:
    it bounds the weak-beam guard's ``(B', n, n, W)`` per-pair term tensor by
    reducing the weak axis in chunks (the full tensor reaches ~41 GiB at
    npx=500/dmin=0.05 → CUDA OOM otherwise).  Bit-faithful (same ``τ``, masked
    incremental accumulation).
    """
    n_inside = int(inside_idx.shape[0])
    if n_inside == 0:
        return

    # iter-13 OOM fix — cap the weak-beam chunk by VRAM.  The per-weak-chunk
    # ``off_terms`` tensor is ``B'·n²·weak_chunk`` complex128 with ``B'·n² ≈
    # chunk_cells`` (the lever-3 direction batch is sized so ``B'·n² ≤
    # chunk_cells``), so its element count is ≈ ``chunk_cells·weak_chunk`` and the
    # working set (the tensor + a few same-shape temporaries from the guard
    # mask: ``abs``/``term_mag``/``keep``/``where``) is several× that.  At the
    # default ``chunk_cells=4e6`` an un-capped ``weak_chunk=64`` would size
    # ``off_terms`` at ~256 M cells (~4 GiB raw, ~25 GiB with temporaries) → OOM
    # on a 12 GB GPU.  Cap the EFFECTIVE chunk so ``chunk_cells·weak_chunk_eff``
    # stays under a cell budget (~32 M cells → off_terms ≤ ~0.5 GiB raw, peak
    # ≤ ~2–4 GiB with temporaries — measured 2.3 GiB on the RTX 4070 at
    # npx=200/dmin=0.05).  Bit-faithful: a smaller chunk only changes the Σ_w
    # reduction grouping (≤1e-14), and ``weak_chunk_eff ≥ W`` is still the
    # unchunked path exactly.  Only applied on CUDA (CPU has host RAM headroom and
    # no per-allocation VRAM ceiling); the explicit ``weak_chunk`` is the upper
    # bound the user requested.
    weak_chunk_eff = int(weak_chunk)
    if dev.type == "cuda" and weak_chunk_eff > 0:
        _WEAK_CELL_BUDGET = 32_000_000
        cap = max(1, _WEAK_CELL_BUDGET // max(1, int(chunk_cells)))
        weak_chunk_eff = min(weak_chunk_eff, cap)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 24)
    n_workers = max(1, int(n_workers))
    eig_dtype = torch.complex64 if complex64_eig else torch.complex128

    # Strong-beam count per inside direction (mask sum); group identical counts.
    strong_masks_cpu = strong_masks_all.detach().cpu()
    weak_masks_cpu = weak_masks_all.detach().cpu()
    ns = strong_masks_cpu.sum(dim=1).to(torch.int64).numpy()  # (n_inside,)
    all_refl_cpu = all_refl.detach().cpu()
    k_cpu = k_miller_all.detach().cpu()
    inside_idx_cpu = inside_idx.detach().cpu()
    # On the Sgh box-gather path (lever 2b-2) ``sgh_full`` is None — the dense
    # (M, M) matrix is never built; the per-chunk strong Sgh sub-block is gathered
    # from a tiny difference box (built below).  Otherwise materialise it on CPU as
    # before (the eig/serial path indexes ``sgh_cpu``; the propagation path stages
    # ``sgh_dev``).
    sgh_cpu = (
        None if sgh_box_path else sgh_full.detach().cpu().to(torch.complex128)
    )
    depth_w_cpu = None if depth_weight is None else depth_weight.detach().cpu()

    # SP4 iter-4: GPU scattering-matrix propagation depth integral.  The eig path
    # is pinned to CPU (#107291), so it cannot use the GPU; the propagation
    # (matrix_exp + matvecs) runs ON ``dev``.  Pre-stage the depth weight + Sgh
    # there once.  Chunks still go through the ThreadPoolExecutor — the per-chunk
    # GPU ops serialise harmlessly on the one CUDA stream while the (now-dominant)
    # CPU ``build_A_batched`` assembly overlaps across cores.
    sgh_dev = (
        (sgh_cpu.to(dev) if (use_propagation and not sgh_box_path) else None)
    )
    depth_w_dev = (
        None if depth_weight is None else depth_weight.detach().to(dev)
    ) if use_propagation else None

    # Lever 2b-2: Sgh difference box (built ONCE; tiny (D,D,D)).  On the box path
    # the per-chunk strong Sgh sub-block is GATHERED from this box on ``dev`` —
    # never materialising any (M, M) Sgh and replacing the per-chunk index_select
    # ×2/dir + torch.stack.  Bit-identical to compute_Sgh (proven Δ=0.0).
    sgh_box = None
    sgh_box_H = 0
    if sgh_box_path and structure is not None:
        sgh_box, sgh_box_H = build_sgh_box(
            structure, all_refl, voltage_kV, dev, dtype=torch.complex128
        )

    # SP4 iter-5: GPU dense-difference-LUT for build_A.  Built ONCE here (tiny,
    # ≤0.6 MB) and shared read-only across all chunks — the per-chunk assembly
    # then gathers ``U_{g−h}`` from it with pure device index ops instead of the
    # CPU ``np.unique``+dict path (the iter-4 wall bottleneck; profiled at 92 % of
    # ``build_A_batched``'s own time on ``np.unique``'s ``argsort``).  Only on the
    # propagation path (the eig path is CPU-pinned by #107291).
    diff_lut = None
    diff_lut_H = 0
    if build_a_gpu and use_propagation:
        if shared_diff_lut is not None:
            # Lever 2b-3: reuse the box already built in build_master (shared with
            # build_ug_lut) instead of rebuilding it here — same box, identical U.
            diff_lut, diff_lut_H = shared_diff_lut, shared_diff_lut_H
        else:
            diff_lut, diff_lut_H = build_diff_lut(
                all_refl, Ug_lookup, dev, dtype=lut_dtype
            )

    # --- Enumerate the independent chunk work-items (cheap; serial) -------------
    # Each item is (sub, g_strong_idx_sub, g_weak_idx_sub, w_max) — the index
    # lists for one chunk.  The heavy U-lookup / eig / Lgh happens in the worker.
    chunks: list[tuple[np.ndarray, list, list, int]] = []
    uniq_n = np.unique(ns)
    for n in uniq_n:
        n = int(n)
        grp = np.nonzero(ns == n)[0]                          # local inside indices
        g_strong_idx = [
            torch.nonzero(strong_masks_cpu[gi], as_tuple=False).reshape(-1)
            for gi in grp
        ]
        g_weak_idx = [
            torch.nonzero(weak_masks_cpu[gi], as_tuple=False).reshape(-1)
            for gi in grp
        ]
        w_max = max((wi.shape[0] for wi in g_weak_idx), default=0)
        per_dir = max(n * n, 1)
        chunk = max(1, chunk_cells // per_dir)
        for c0 in range(0, len(grp), chunk):
            c1 = min(c0 + chunk, len(grp))
            chunks.append(
                (
                    grp[c0:c1],
                    g_strong_idx[c0:c1],
                    g_weak_idx[c0:c1],
                    w_max,
                )
            )

    def _process(item):
        """Assemble + eig + Lgh + Sgh for one chunk → (global_idx, vals). Thread-safe."""
        sub, g_strong, g_weak, w_max = item
        bsz = len(sub)

        strong_hkl = torch.stack(
            [all_refl_cpu[g_strong[j]] for j in range(bsz)], dim=0
        )                                                      # (B', n, 3)
        k_grp = k_cpu[torch.from_numpy(sub.astype(np.int64))]  # (B', 3)

        # Build the per-direction zero-padded weak set (shared by both A paths).
        weak_hkl = weak_mask = weak_sg = None
        if w_max > 0:
            weak_hkl = torch.zeros(bsz, w_max, 3, dtype=all_refl_cpu.dtype)
            weak_mask = torch.zeros(bsz, w_max, dtype=torch.bool)
            weak_sg = torch.zeros(bsz, w_max, dtype=torch.float64)
            for j in range(bsz):
                wi = g_weak[j]
                wn = int(wi.shape[0])
                if wn == 0:
                    continue
                wref = all_refl_cpu[wi]
                weak_hkl[j, :wn] = wref
                weak_mask[j, :wn] = True
                weak_sg[j, :wn] = excitation_error(
                    wref, k_grp[j], gstar
                ).to(torch.float64)

        if build_a_gpu and use_propagation and diff_lut is not None:
            # --- SP4 iter-5: assemble A on the GPU via the dense difference-LUT --
            # Pure device tensor ops gathered from the precomputed LUT (no
            # np.unique/argsort, no CPU dict); A is born on ``dev``.
            A = build_A_batched_gpu(
                k_grp.to(dev),
                strong_hkl.to(dev),
                diff_lut,
                diff_lut_H,
                reciprocal_metric=gstar,
                wavelength_nm=wavelength_nm,
                Upz=upz,
                weak_hkl=None if weak_hkl is None else weak_hkl.to(dev),
                weak_mask=None if weak_mask is None else weak_mask.to(dev),
                weak_sg=None if weak_sg is None else weak_sg.to(dev),
                weak_pert_tau=weak_pert_tau,
                weak_chunk=weak_chunk_eff,
            ).to(torch.complex128)                             # (B', n, n) on dev
        else:
            A = build_A_batched(
                k_grp,
                strong_hkl,
                Ug_lookup,
                reciprocal_metric=gstar,
                wavelength_nm=wavelength_nm,
                Upz=upz,
                weak_hkl=weak_hkl,
                weak_mask=weak_mask,
                weak_sg=weak_sg,
                weak_pert_tau=weak_pert_tau,
                weak_chunk=weak_chunk_eff,
            )                                                  # (B', n, n) CPU

        if use_propagation:
            # --- SP4 iter-4: GPU scattering-matrix propagation (no eig) ---------
            # Run the whole depth integral (one batched matrix_exp + izz batched
            # matvecs + outer-product accumulation) on the GPU.  ``A`` is already
            # on ``dev`` (GPU build_A) or moved there (CPU build_A).  Sgh is
            # pre-staged on ``dev``.
            A_dev = A if A.device == dev else A.to(dev)
            Lgh = depth_integrated_Lgh_propagate(
                A_dev,
                mc_used,
                energy_idx,
                wavelength_nm=wavelength_nm,
                depth_weight=depth_w_dev,
                s1_dtype=propagation_s1_dtype,
            )                                                  # (B', n, n) on dev
            if sgh_box is not None:
                # Lever 2b-2: gather the strong Sgh sub-block directly from the
                # difference box (pure device index op, no (M,M), no per-dir
                # index_select/stack).  ``strong_hkl`` is the (B', n, 3) per-
                # direction strong reflection list; bit-identical to compute_Sgh.
                sgh_grp = _gather_sgh_block(
                    sgh_box, sgh_box_H, strong_hkl.to(dev)
                )                                              # (B', n, n) on dev
            else:
                sidx_dev = [g_strong[j].to(dev) for j in range(bsz)]
                sgh_grp = torch.stack(
                    [
                        sgh_dev.index_select(0, sidx_dev[j]).index_select(1, sidx_dev[j])
                        for j in range(bsz)
                    ],
                    dim=0,
                )                                              # (B', n, n) on dev
            vals = (Lgh * sgh_grp).sum(dim=(-2, -1)).real / n_atoms  # (B',) on dev
            vals = vals.to("cpu")
        else:
            # Batched depth-integrated Lgh on CPU (eig is CUDA-hostile — #107291;
            # eig_dtype=complex64 halves the eig cost, eigenpairs promoted to c128).
            Lgh = depth_integrated_Lgh(
                A,
                mc_used,
                energy_idx,
                wavelength_nm=wavelength_nm,
                depth_weight=depth_w_cpu,
                eig_dtype=eig_dtype,
                use_flip_guard=eig_use_flip_guard,
            )                                                  # (B', n, n) CPU

            sgh_grp = torch.stack(
                [
                    sgh_cpu.index_select(0, g_strong[j]).index_select(1, g_strong[j])
                    for j in range(bsz)
                ],
                dim=0,
            )                                                  # (B', n, n)

            # I(k) = Re{ Σ_{g,h} Lgh·Sgh } / N_atoms  (Hadamard sum per direction).
            vals = (Lgh * sgh_grp).sum(dim=(-2, -1)).real / n_atoms  # (B',)
        global_idx = inside_idx_cpu[torch.from_numpy(sub.astype(np.int64))]
        return global_idx, vals

    def _scatter(global_idx, vals):
        out[global_idx.to(out.device)] = vals.to(out.device, dtype=out.dtype)

    # Progress accounting (UI-only; does not touch numerics).  ``done`` is
    # accumulated in the MAIN thread as chunks are scattered, so it is
    # monotonic and ends at ``n_inside`` regardless of completion order.
    _done = 0
    if progress_cb is not None:
        progress_cb(0, int(n_inside))

    def _report(n_scattered: int) -> None:
        nonlocal _done
        if progress_cb is None:
            return
        _done += int(n_scattered)
        progress_cb(min(_done, int(n_inside)), int(n_inside))

    # --- Run the chunks --------------------------------------------------------
    # Serial when there's nothing to parallelise.
    if n_workers == 1 or len(chunks) <= 1:
        for item in chunks:
            gi, vals = _process(item)
            _scatter(gi, vals)
            _report(int(gi.shape[0]))
        return

    # Parallel across cores.  Pin torch intra-op threads to 1 — both the batched
    # LAPACK eig (iter-3) AND the numpy U-lookup inside ``build_A_batched`` contend
    # badly with intra-op threads, so the parallelism must live across the chunk
    # work-items.  In **propagation** mode this overlaps the (now-dominant ~70 %)
    # CPU ``build_A_batched`` assembly across cores while the per-chunk GPU
    # ``matrix_exp`` + matvecs (tiny, ~8 %) serialise harmlessly on the one CUDA
    # stream — turning the GPU depth integral's near-zero cost into a real wall
    # win once the CPU assembly is no longer serial.  Eig releases the GIL; numpy
    # ``np.unique`` / matmul largely do too — both scale across threads here.
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for gi, vals in ex.map(_process, chunks):
                # Scatter in the main thread (disjoint targets; order-independent).
                _scatter(gi, vals)
                _report(int(gi.shape[0]))
    finally:
        torch.set_num_threads(prev_threads)


class _UniformLambdaMC:
    """Wrap an :class:`MCData` so ``lambda_z`` returns a flat depth profile.

    Used for the ``uniform_lambda=True`` geometry-sanity mode: every depth bin gets
    equal weight ``1/izz`` (sums to 1), isolating band *geometry* from the MC depth
    weighting.  ``energy_weight`` and all other attributes delegate to the wrapped
    ``MCData`` unchanged.
    """

    def __init__(self, mc: MCData) -> None:
        self._mc = mc

    def __getattr__(self, name):  # delegate everything else (n_energy, depth_step…)
        return getattr(self._mc, name)

    def lambda_z(self, energy_idx: int) -> torch.Tensor:
        ref = self._mc.lambda_z(energy_idx)  # validates range + gives izz/device/dtype
        izz = int(ref.shape[0])
        return torch.full_like(ref, 1.0 / izz)
