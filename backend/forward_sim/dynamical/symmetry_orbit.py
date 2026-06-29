"""SP-PERF lever 1 — point-group symmetry-orbit reduction for the master build.

The dynamical master value ``I(d)`` at a Lambert-grid direction ``d`` is a **pure
crystal property** that is invariant under every operation ``R`` of the crystal's
point group: for any point-group rotation ``R``, the dynamical matrix ``A`` at the
rotated direction ``R·d`` is a *permutation* of the matrix at ``d`` (the same set of
reflections with relabelled indices), so it has identical eigenvalues and the
master value comes out identical — ``I(R·d) == I(d)`` exactly (this is precisely the
symmetry EMsoft exploits with its ``Apply3DPGSymmetry`` scatter after solving only
the irreducible Lambert fundamental zone).

This module exploits that to cut the *direction count* the expensive per-direction
eigensolve actually runs on:

1. take the crystal's point-group **rotation** operations (the unique ``R`` parts of
   the space group's ``symop_list``, the same machinery as
   :func:`backend.forward_sim.crystal.structure_matrix._expand_symmetry_orbit`),
   expressed in the **Cartesian** direction frame as
   ``R_cart = dsm · R_frac · dsm⁻¹`` (orthogonal — verified to ~1e-16);
2. for every inside-disc grid pixel direction ``d`` (the SAME pixel↔direction map
   the grid uses, square or hex), apply every ``R_cart`` and map each image back to
   its grid pixel with ``nint`` rounding (mirroring EMsoft's integer scatter);
3. union pixels that share an orbit (a vectorised union-find), pick ONE
   representative per orbit, and emit the representative→orbit scatter map.

``build_master`` then restricts the dynamical solve to the representatives and
scatters each representative's computed value to all pixels in its orbit.  Because
``I`` is constant on an orbit, this is correctness-EXACT up to the ``nint``
pixel-quantisation that EMsoft itself incurs (the self-test gate pins the resulting
master to NCC > 0.9999 vs the full-disc solve — agreement is ~1e-6 relative, i.e.
FP reduction-order jitter from solving the representative instead of the pixel, NOT
strictly bit-identical).

Hemisphere handling.  The reduction is done **per hemisphere** on the hemisphere's
own grid directions (``z ≥ 0`` for north, the z-negated set for south), using the
point-group operations whose image **stays in that hemisphere** (``z`` keeps sign).
Both proper and improper operations are used: an improper operation that maps
``z ≥ 0 → z ≥ 0`` (e.g. a vertical mirror or a horizontal mirror σ_h) is a genuine
symmetry of the master *within* the hemisphere and contributes real reduction; the
ones that flip the hemisphere (inversion, σ_h on the wrong side) relate NH↔SH and
are simply not unioned here (NH and SH are built as separate ``build_master`` calls,
so cross-hemisphere identities need not be realised inside one call).
"""
from __future__ import annotations

import numpy as np
import torch

from ..crystal.xtal_io import CrystalStructure
from backend.dictionary_gpu.lambert import direction_to_lambert


def cartesian_point_group_ops(structure: CrystalStructure) -> np.ndarray:
    """Unique Cartesian point-group operation matrices ``R_cart`` for the crystal.

    The space group's symmetry operations carry a rotation/reflection part ``R``
    (and a translation ``t``, irrelevant to a *direction*).  ``R`` acts on
    **fractional** coordinates; a Cartesian direct vector ``x = dsm·r`` therefore
    transforms as ``x' = dsm·R·r = (dsm·R·dsm⁻¹)·x``, so the Cartesian operation is

        ``R_cart = dsm · R_frac · dsm⁻¹``

    with ``dsm`` the direct structure matrix (columns ``a, b, c`` in the IUCr
    Cartesian setting).  ``R_cart`` is orthogonal for every crystal system
    (``R_cart·R_cartᵀ == I`` to ~1e-16) because ``R_frac`` is an isometry of the
    lattice.  The unique ``R`` parts are deduplicated (the full ``symop_list``
    repeats each ``R`` once per Bravais centring translation).

    Args:
        structure: the crystal (supplies the space group + direct structure matrix).

    Returns:
        ``(G, 3, 3)`` float64 array of unique Cartesian point-group operations,
        including the identity.  Both proper (``det = +1``) and improper
        (``det = −1``) operations are returned; the caller decides which to apply.
    """
    from diffpy.structure.spacegroups import GetSpaceGroup

    sg = GetSpaceGroup(int(structure.space_group))
    dsm = np.asarray(structure.structure_matrix, dtype=np.float64)
    dsm_inv = np.linalg.inv(dsm)

    uniq: list[np.ndarray] = []
    for op in sg.symop_list:
        r = np.asarray(op.R, dtype=np.float64)
        if not any(np.allclose(r, u, atol=1e-9) for u in uniq):
            uniq.append(r)

    ops = np.stack([dsm @ r @ dsm_inv for r in uniq], axis=0)  # (G, 3, 3)
    return ops


class _UnionFind:
    """Minimal union-find with path-halving + union-by-min-root.

    Union-by-min-root keeps the representative of each orbit **deterministic**
    (always the lowest pixel index in the orbit), so the representative set — and
    therefore which pixels are solved — is reproducible run to run.
    """

    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Lower index becomes the root → deterministic representatives.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


def compute_orbit_reduction(
    structure: CrystalStructure,
    directions_cart: torch.Tensor,
    inside: torch.Tensor,
    npx: int,
    *,
    hemisphere: str = "north",
    grid: str = "square",
):
    """Group inside-disc grid pixels into point-group orbits (lever 1).

    For the hemisphere's grid directions ``directions_cart`` (one per Lambert
    pixel, in the SAME row-major ``[row, col]`` order :func:`_grid_directions` /
    :func:`_grid_directions_hex` produce) and the inside-disc mask ``inside``,
    apply every in-hemisphere point-group operation to each inside direction, map
    each image to its grid pixel (``nint`` rounding of the Lambert square
    coordinate, EMsoft-style), and union pixels sharing an orbit.  Returns the
    representative pixel indices and a scatter map that copies each
    representative's value to all pixels in its orbit.

    The pixel↔direction convention is exactly the grid's: a pixel ``(row, col)``
    has Lambert coordinate ``xy = ((col−npx)/npx, (row−npx)/npx)``, and the inverse
    (direction → pixel) is ``col = nint(x·npx + npx)``, ``row = nint(y·npx + npx)``
    with ``xy`` the grid's Lambert projection of the direction — the **square** map
    ``direction_to_lambert`` for ``grid="square"`` (the validated cubic/tetragonal/
    orthorhombic/monoclinic/triclinic path) or the **hex** map
    :func:`_sphere_to_hex` for ``grid="hex"`` (the γ=120° usehex path).  Using the
    matching map is correctness-critical for hex: the 6-/3-fold rotations map hex
    grid points to hex grid points **exactly** under the hex map (100 % exact
    landings) but only ~4 % under the square map, so a square-mapped hex reduction
    would capture almost no orbits.  Both maps collapse ``|z|`` (they project both
    hemispheres onto the same square), so an operation whose image leaves the build
    hemisphere (``z`` sign flips) is **excluded** — it would alias onto the
    wrong-hemisphere pixel; such NH↔SH identities are handled by the separate SH
    ``build_master`` call, not within one hemisphere's reduction.

    Args:
        structure: the crystal (supplies the point-group operations).
        directions_cart: ``(P, 3)`` Cartesian unit directions, one per grid pixel
            (``P = (2·npx+1)²``), in row-major order.  For ``hemisphere="south"``
            this is the z-negated grid (matching ``build_master``).
        inside: ``(P,)`` bool — inside-inscribed-disc mask for the grid.
        npx: Lambert half-grid size (grid is ``(2·npx+1)²``).
        hemisphere: ``"north"`` (``z ≥ 0`` build) or ``"south"`` (``z ≤ 0``); fixes
            the in-hemisphere sign test for the operation images.
        grid: ``"square"`` (default — the orthogonal-axis Lambert grid, all systems
            except usehex) or ``"hex"`` (the sheared-60° hexagonal grid for
            hexagonal/trigonal usehex cells); selects the direction→pixel Lambert
            map so the orbit images land exactly on grid pixels.

    Returns:
        ``(rep_idx, scatter_src, scatter_dst)`` where:

        * ``rep_idx`` is a 1-D int64 tensor of the **representative** pixel indices
          (a subset of the inside-disc pixels) the dynamical solve runs on;
        * ``scatter_src`` / ``scatter_dst`` are 1-D int64 tensors of equal length
          such that ``out[scatter_dst] = out[scatter_src]`` (after the
          representatives are filled) copies every representative value to its
          orbit members.  ``scatter_src`` entries are always representative pixels;
          ``scatter_dst`` covers every non-representative inside pixel exactly once.

        All tensors are on the CPU (indexing is host-side; the caller moves them to
        the compute device as needed).

    Raises:
        ValueError: if ``hemisphere`` is not ``"north"``/``"south"`` or ``grid`` is
            not ``"square"``/``"hex"``.
    """
    if hemisphere not in ("north", "south"):
        raise ValueError(
            f"hemisphere must be 'north' or 'south', got {hemisphere!r}"
        )
    if grid not in ("square", "hex"):
        raise ValueError(f"grid must be 'square' or 'hex', got {grid!r}")

    device = torch.device("cpu")
    dirs = directions_cart.detach().to(device, torch.float64)
    inside_np = inside.detach().cpu().numpy().astype(bool)
    P = dirs.shape[0]
    m = 2 * npx + 1
    zsign = 1.0 if hemisphere == "north" else -1.0

    inside_idx_np = np.nonzero(inside_np)[0]
    N = inside_idx_np.shape[0]
    if N == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty.clone(), empty.clone()

    ops = cartesian_point_group_ops(structure)                # (G, 3, 3) float64
    ops_t = torch.from_numpy(ops)                             # CPU float64
    G = ops_t.shape[0]

    d_in = dirs[torch.from_numpy(inside_idx_np)]              # (N, 3)

    # Apply every op: rot[g, n] = R_cart[g] · d_in[n]  → (G, N, 3).
    rot = torch.einsum("gij,nj->gni", ops_t, d_in)           # (G, N, 3)
    rot_flat = rot.reshape(-1, 3)
    # Renormalise (orthogonal R keeps |d|=1 to ~1e-16, but renormalise for the
    # Lambert map's |z|≤1 clamp to be exact at the pole).
    rot_flat = rot_flat / rot_flat.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).eps
    )

    # In-hemisphere test: keep only images whose z keeps the build-hemisphere sign
    # (direction_to_lambert collapses |z|, so a flipped-hemisphere image would
    # alias onto the wrong pixel).  A tiny tolerance keeps equator images (z≈0).
    z = rot_flat[:, 2]
    in_hemi = (z * zsign) >= -1e-9

    # Direction → Lambert coordinate → pixel (nint rounding, EMsoft scatter).  Use
    # the grid's OWN Lambert map: the square map for orthogonal-axis grids, the hex
    # map for usehex cells (the 6-/3-fold ops only land exactly under the hex map).
    if grid == "hex":
        from .master_builder import _sphere_to_hex

        xy, _ok = _sphere_to_hex(rot_flat)                   # (G·N, 2) hex-Lambert
    else:
        xy = direction_to_lambert(rot_flat)                  # (G·N, 2) in [-1,1]²
    col = torch.round(xy[:, 0] * npx + npx).to(torch.int64)
    row = torch.round(xy[:, 1] * npx + npx).to(torch.int64)
    valid = (
        in_hemi
        & (col >= 0) & (col < m) & (row >= 0) & (row < m)
    )
    pix = (row * m + col)                                     # (G·N,) target pixel

    # --- EXACTNESS gate (correctness-critical) ---------------------------------
    # ``nint`` rounding lands the orbit image on the NEAREST grid pixel — but for an
    # operation that is NOT a signed axis permutation (the cubic 3-fold body-diagonal
    # rotations, the hex 6-/3-fold rotations) the rotated direction ``R·d`` is NOT a
    # grid point, so the nearest pixel's OWN direction differs from ``R·d`` by up to
    # ~1.4° — unioning them would equate two genuinely-different master values and
    # corrupt the result (measured: cubic NCC 0.66, hex 0.20 without this gate).
    # We therefore union ONLY when ``R·d`` matches the target pixel's ACTUAL stored
    # grid direction to within a tight tolerance, so ``I`` is truly constant on the
    # unioned set (``f(R·d) == f(d_target)`` exactly by symmetry).  Axis-aligned ops
    # (every tetragonal/orthorhombic/monoclinic op, and the 16 grid-aligned cubic
    # ops) land at 0.000° → kept; the misaligned ops are skipped (those orbits are
    # simply not reduced — bit-faithful is non-negotiable).  ``|·|`` on the dot
    # handles the ``direction_to_lambert`` |z|-collapse at the equator.
    pix_clamped = pix.clamp(0, P - 1)
    tgt_dir = dirs[pix_clamped]                              # (G·N, 3) stored grid dir
    tgt_dir = tgt_dir / tgt_dir.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).eps
    )
    cosang = (rot_flat * tgt_dir).sum(dim=-1).abs()         # (G·N,) |R·d · d_target|
    # Threshold ``1 − 1e-10`` (≈ 0.0008°): the exact axis-permutation matches sit at
    # cos ≥ 1 − 1e-12 (FP noise on the orthogonal ops) and the exact equator-antipode
    # matches (centrosymmetric inversion) at cos = 1.0 — both KEPT; the misaligned
    # 3-/6-fold images sit at cos ≈ 1 − 8e-7 (≈ 0.07°) — REJECTED.  A looser 1e-6
    # gate would admit those 0.07° cubic/hex misalignments and re-introduce a small
    # NCC error, so the tight gate is required for the bit-faithful interior.
    exact = cosang >= (1.0 - 1e-10)
    valid = valid & exact

    pix = pix.reshape(G, N)
    valid = valid.reshape(G, N)

    # Only union to targets that are themselves inside-disc grid pixels (the orbit
    # lives entirely on the inside-disc set the solve covers).
    pix_np = pix.numpy()
    valid_np = valid.numpy()
    inside_target = inside_np[np.clip(pix_np, 0, P - 1)]     # (G, N) bool
    valid_np = valid_np & inside_target

    uf = _UnionFind(P)
    self_pix = inside_idx_np                                 # (N,) the source pixels
    # Build the union edges: for each (g, n) valid pair, edge self_pix[n] ↔ pix.
    # Vectorise the edge extraction, then DROP self-edges (identity op + any op that
    # fixes a pixel) and DEDUPLICATE canonical (min, max) pairs before the Python
    # union loop — at npx=500 the raw (G·N) edge set is ~37 M (Ni: 48 ops × 785 k);
    # self/dup removal collapses it to the genuine orbit edges (~N), keeping the
    # one-time reduction setup negligible vs the build.  ``np.unique`` on the packed
    # ``a·P + b`` key is the canonical dedup (a ≤ b enforced first).
    src_col = np.broadcast_to(self_pix[None, :], (G, N))     # (G, N)
    edges_a = src_col[valid_np]
    edges_b = pix_np[valid_np]
    lo = np.minimum(edges_a, edges_b)
    hi = np.maximum(edges_a, edges_b)
    nontrivial = lo != hi                                    # drop self-edges
    lo, hi = lo[nontrivial], hi[nontrivial]
    if lo.size:
        key = lo.astype(np.int64) * np.int64(P) + hi.astype(np.int64)
        uniq_key = np.unique(key)                            # canonical unique edges
        u_lo = (uniq_key // P).astype(np.int64)
        u_hi = (uniq_key % P).astype(np.int64)
        for a, b in zip(u_lo.tolist(), u_hi.tolist()):
            uf.union(a, b)

    # Resolve roots for every inside pixel.
    roots = np.array([uf.find(int(p)) for p in self_pix], dtype=np.int64)  # (N,)
    # Representative = the (deterministic, min-index) root pixel of each orbit.
    rep_pixels = np.unique(roots)                            # sorted unique roots

    # The roots are by construction inside-disc pixels (a root is the min index of
    # an orbit, and every orbit contains its source inside pixel — the source is a
    # union endpoint, so the min root is ≤ that inside index and reachable only via
    # inside-target unions ⇒ itself inside).  Assert to fail loud if ever violated.
    if not inside_np[rep_pixels].all():
        bad = rep_pixels[~inside_np[rep_pixels]]
        raise RuntimeError(
            f"orbit representative(s) fell outside the inscribed disc: {bad[:8]}"
        )

    rep_idx = torch.from_numpy(np.ascontiguousarray(rep_pixels))

    # Scatter map: every non-representative inside pixel copies from its root.
    is_rep = roots == self_pix                               # source pixel IS its root
    dst = self_pix[~is_rep]                                  # non-rep inside pixels
    src = roots[~is_rep]                                     # their representative root
    scatter_src = torch.from_numpy(np.ascontiguousarray(src))
    scatter_dst = torch.from_numpy(np.ascontiguousarray(dst))

    return rep_idx, scatter_src, scatter_dst
