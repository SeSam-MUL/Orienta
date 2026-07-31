"""Tier-1 spherical indexing: single-pass cross-correlation, no refinement.

Per-pattern flow:

1. Preprocess (circmask + gausbckg + nregions).
2. For each GL square-Legendre grid cell inside the detector window, compute
   its sphere normal via the EMSphInx square-Legendre ``normals()`` formula and
   forward-project it to detector coordinates.
3. Compute the SHT coefficients by GL-quadrature-weighted direct projection:
     c[m,l] = sum_p f_p * w_p * conj(Y_l^m(n_p))
   where w_p = gl_weight[ar-1] / (8*ar) is the per-cell GL quadrature weight
   and n_p is the unit sphere normal for GL cell p.
4. Subtract mean (zero m=0 row) to suppress the orientation-independent peak.
5. Cross-correlate via ``rs2cc_`` → 3-D Euler-angle correlation volume.
6. Find peak → Bunge ZYZ Euler angles → Bunge ZXZ Euler angles.

Why GL square-Legendre grid SHT
--------------------------------
The detector window covers only a ~29° polar cap (n2 ≈ 0.875–1.0).  For
Driscoll-Healy (DH) grids the quadrature weight per cell is proportional to
sin(θ), which is tiny near the pole — polar-cap signal is severely
underweighted.  Gauss-Legendre (GL) quadrature has larger weights near the
poles, making it well-conditioned for the polar-cap geometry.

The square Legendre grid used here exactly matches EMSphInx's ``squareSHT``
normals formula (``include/util/square_sht.hpp:normals()``):
  - Grid dim = bw + 3 if bw even, bw + 2 if bw odd  (= bw + 3 for bw=68 → 71)
  - half = dim // 2 (= 35)
  - GL nodes = positive-cosine half of leggauss(dim-2) sorted high→low
  - Ring ar (1..half): 8*ar azimuthal cells, all at colatitude arccos(GL_node[ar-1])
  - For cell (i,j) with ri=i-half, rj=j-half, ar=max(|ri|,|rj|):
      sX = ri/ar,  sY = rj/ar
      if |ri| <= |rj|:  qq = π/4 * sX * sY;  x = sY*sin(qq);  y = sY*cos(qq)
      else:             qq = π/4 * sY * sX;  x = sX*cos(qq);  y = sX*sin(qq)
      n[0,1] = sin_lat * [x,y] / hypot(x,y)
      n[2]   = GL_node[ar-1]

References
----------
- EMSphInx ``include/util/square_sht.hpp:normals()``
- EMSphInx ``include/modality/ebsd/detector.hpp``
- ebsdtorch ``_math/sht_cc.py`` for the cross-correlation API contract.

Decode formula (identity at (a=0, b=0, c=0)):
  dim -2 (b):  Φ     = b · scale
  dim -3 (a):  α     = a · scale
  dim -1 (c):  γ     = c · scale
  ZYZ → ZXZ:  phi1  = α + π/2 + 3π/2,  phi2 = γ − π/2
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .._math import (
    RSHT,
)
from .._math.sht import dltWeightsDH
from .._math._wigner_logspace import wigner_d_eq_half_pi
from .._math.sht_cc import rs2cc_, rs2cc_fp64, rs2cc_sparse, rs2cc_vectorized
from .._math.fused_kernels import fused_mul_max as _fused_mul_max_cupy
from .._math.fused_kernels import fused_max_only as _fused_max_only_cupy
from .._math.fused_kernels import is_cupy_kernel_available as _cupy_kernel_ok
from .detector import DetectorGeometry
from .preprocessing import gausbckg, circmask, nregions
from .sht_io import SHTMasterFile
from ._shared_tables import SharedSphericalTables


# Default indexing bandwidth — matches EMSphInx 'bw=68' default
DEFAULT_INDEX_BANDWIDTH = 68


def _build_normalized_alf(L: int, cos_theta_np: np.ndarray) -> np.ndarray:
    """Build the normalized associated-Legendre table alp[m, l, p].

    ``alp[m, l, p] = K_l^m * P_l^m(cos θ_p)`` for all valid (m, l) pairs
    (l >= m) and all p cells; entries with l < m are zero.

    This is a vectorized rewrite of the original triple Python loop in
    :meth:`Tier1Indexer._build_direct_sht_map`.  It vectorizes over the cell
    index ``p`` (formerly ``j``): ``pmm``, ``pm_prev``, ``pm_curr``,
    ``pm_next``, ``x`` and ``sinphi`` are now ``(N,)`` float64 arrays instead
    of scalars.  The per-element arithmetic, evaluation order, and float32
    cast-on-assign are kept BIT-IDENTICAL to the scalar version, so the table
    is numerically unchanged (verified by ``tests/test_alf_vectorized.py``).

    Depends only on ``L`` (bandwidth) and ``cos_theta`` (detector geometry),
    NOT on the phase / SHT master file.

    Parameters
    ----------
    L : int
        Bandwidth.
    cos_theta_np : (N,) array
        ``cos θ_p`` per inside GL cell (i.e. ``n2_in``).

    Returns
    -------
    alp_np : (L, L, N) float32
    """
    # ---- log-factorial cumulative sum (same array as the scalar code) ----
    log_fac = np.zeros(2 * L + 2, dtype=np.float64)
    for i in range(1, len(log_fac)):
        log_fac[i] = log_fac[i - 1] + math.log(i)

    # ---- K coefficients K[l, m], same formula as the scalar K_lm() -------
    # m == 0  -> sqrt((2l+1)/(4π))
    # else    -> exp(0.5*log((2l+1)/(4π)) + 0.5*(log_fac[l-m] - log_fac[l+m]))
    K = np.zeros((L, L), dtype=np.float64)
    for l in range(L):
        for m in range(l + 1):
            if m == 0:
                K[l, m] = math.sqrt((2 * l + 1) / (4.0 * math.pi))
            else:
                K[l, m] = math.exp(
                    0.5 * math.log((2 * l + 1) / (4.0 * math.pi))
                    + 0.5 * (log_fac[l - m] - log_fac[l + m])
                )

    x = np.asarray(cos_theta_np, dtype=np.float64).reshape(-1)   # (N,)
    sinphi = np.sqrt(np.maximum(1.0 - x * x, 0.0))               # (N,)
    N = x.shape[0]

    alp_np = np.zeros((L, L, N), dtype=np.float32)
    for m in range(L):
        # pmm via the SAME iterative product as the scalar (exact factor order)
        pmm = np.ones(N, dtype=np.float64)
        for k in range(1, m + 1):
            pmm *= sinphi * (2 * k - 1)
        alp_np[m, m, :] = (K[m, m] * pmm).astype(np.float32)
        pm_prev = pmm
        pm_curr = x * (2 * m + 1) * pmm
        if m + 1 < L:
            alp_np[m, m + 1, :] = (K[m + 1, m] * pm_curr).astype(np.float32)
        for l in range(m + 2, L):
            pm_next = (
                (2 * l - 1) * x * pm_curr - (l + m - 1) * pm_prev
            ) / (l - m)
            alp_np[m, l, :] = (K[l, m] * pm_next).astype(np.float32)
            pm_prev = pm_curr
            pm_curr = pm_next
    return alp_np


# perf Q0.4: cupy RawKernel that fuses (cc * rDen) + flat-max into ONE
# CUDA kernel. Eliminates the 21 MB nc_vol DRAM round-trip at L=88.
# Falls back to torch eager (mul + flat.max) if cupy is missing.
# Standalone bench at (B=32, 175^3): torch eager 5.96 ms -> cupy fused 4.73 ms = -21%.
# Q0.2 (torch.compile) was REJECTED for adding ~91 ms/batch overhead on Windows.
_CUPY_FUSION_AVAILABLE: Optional[bool] = None


def _fused_max_only_dispatch(
    cc_real_f32: torch.Tensor, rDen: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lever A: skip nc_vol materialisation. Returns just (max_vals, max_idx).

    Pairs with refiner.gather_n27_from_cc_rden in _decode_peak: instead of
    reading 27 elements from a 21.4 MB-per-pattern materialised volume, we
    re-read 27 cc + 27 rDen on the fly. Avoids the 654 MB allocation +
    write per batch.

    Bench (RTX 4070, B=32, L=88): kernel time 5.06 ms -> 1.99 ms (2.55x).
    Pipeline saving: ~3 ms / batch wall time.
    """
    global _CUPY_FUSION_AVAILABLE
    if _CUPY_FUSION_AVAILABLE is None:
        _CUPY_FUSION_AVAILABLE = _cupy_kernel_ok()
    if _CUPY_FUSION_AVAILABLE:
        return _fused_max_only_cupy(cc_real_f32, rDen)
    # Eager fallback: still materialises nc_vol since torch can't fuse
    # mul+max without it. Returns just the (max_vals, max_idx) tuple.
    nc_vol = cc_real_f32 * rDen
    B = nc_vol.shape[0]
    flat = nc_vol.reshape(B, -1)
    m = flat.max(dim=1)
    return m.values, m.indices


def _fused_mul_max_dispatch(
    cc_real_f32: torch.Tensor, rDen: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy dispatch — still returns (nc_vol, max_vals, max_idx).

    Kept for backward-compatibility with callers that explicitly need
    the materialised nc_vol (none in current codebase after Lever A).
    """
    global _CUPY_FUSION_AVAILABLE
    if _CUPY_FUSION_AVAILABLE is None:
        _CUPY_FUSION_AVAILABLE = _cupy_kernel_ok()
    if _CUPY_FUSION_AVAILABLE:
        return _fused_mul_max_cupy(cc_real_f32, rDen)
    # Fallback: torch eager — same math, two kernels.
    nc_vol = cc_real_f32 * rDen
    B = nc_vol.shape[0]
    flat = nc_vol.reshape(B, -1)
    m = flat.max(dim=1)
    return nc_vol, m.values, m.indices


@dataclass
class IndexResult:
    """Indexing output for a batch of patterns."""
    euler_xyz: torch.Tensor       # (N, 3) float32 Bunge ZYZ in radians, on CPU
    score: torch.Tensor           # (N,) float32 correlation peak, on CPU
    phase_id: torch.Tensor        # (N,) int8 (1-indexed phase), on CPU
    runtime_seconds: float
    n_adjusted: int = 0           # pixels whose winner changed vs the unweighted argmax (EDS prior)


class Tier1Indexer:
    """Tier-1: single-phase, single-PC, no refinement, FP32 main path."""

    def __init__(
        self,
        geom: DetectorGeometry,
        master: SHTMasterFile,
        device: torch.device,
        bandwidth: int = DEFAULT_INDEX_BANDWIDTH,
        flip_y: bool = None,
        cc_fp64: bool = False,
        sample_tilt_deg: float | None = None,
        # iter-Phase1: EMSphInx preprocessing flags. Defaults match the
        # values that were previously hardcoded inside ``_index_batch``,
        # so existing callers see identical behaviour.
        circmask: int = -1,         # -1 = off, 0 = inscribed-circle, >0 = explicit radius (px)
        gausbckg: bool = True,
        nregions: int = 10,
        normed: bool = True,
        refine: bool = True,        # Tier-2 sub-bin refinement on the cc volume
        # perf: shared L+geom-only precomputations to skip the
        # ~7.5s/phase redundant build cost in multi-phase mode. When None
        # (solo path), all tables are built locally as before. When given,
        # caller (SphericalGPUBackend) is responsible for ensuring geom
        # compatibility — Tier1Indexer only verifies bandwidth + sample_tilt.
        shared_tables: Optional[SharedSphericalTables] = None,
    ):
        if bandwidth <= 1:
            raise ValueError(f"bandwidth must be >= 2, got {bandwidth}")
        if bandwidth > master.bandwidth:
            raise ValueError(
                f"Indexing bandwidth {bandwidth} exceeds master storage "
                f"bandwidth {master.bandwidth}"
            )
        self.geom = geom
        self.master = master
        self.device = device
        self.bandwidth = int(bandwidth)
        # Auto-detect flip_y from vendor if not explicitly specified.
        # Oxford patterns are stored top-down (image origin at top-left), while
        # EMSphInx detector.hpp uses a bottom-up y-axis.  The flip is applied
        # AFTER fractional coordinate computation in _build_direct_sht_map,
        # matching EMSphInx interpolatePixel line 360: if(flp) Y = 1 - Y.
        # iter-15G: HiGainNi/EDAX validation showed both Oxford H5OINA and
        # EDAX H5 store patterns bottom-up in the camera frame, so both
        # need flip_y=True to align with our top-down internal convention.
        # Previous code had flip_y=False for EDAX which caused 0 percent
        # match on the HiGainNi dataset (44.84 deg median), now 97.5 percent
        # match within 5 deg.
        if flip_y is None:
            # Bruker added 2026-05-09: kikuchipy stores PC in Bruker
            # convention internally, so the FastAPI route passes
            # vendor='Bruker' even for Oxford-source files. Without
            # flip_y=True the projection is mirrored along Y and indexing
            # gives random orientations (median 34 deg neighbor diso vs
            # 3.13 deg with flip_y=True on the 7050 Al-7050 dataset).
            flip_y = (geom.vendor in ("Oxford", "EDAX", "Bruker"))
        self.flip_y = flip_y
        self.cc_fp64 = bool(cc_fp64)
        # EMSphInx-equivalent preprocessing config (used in _index_batch).
        self.circmask = int(circmask)
        self.gausbckg_on = bool(gausbckg)
        self.nregions_n = int(nregions)
        self.normed = bool(normed)
        self.refine = bool(refine)

        # Compute alpha_tilt = angle between sample surface and detector plane.
        # Used for the GL-frame → sample-frame rotation (quNp in EMSphInx).
        # alpha = 90° - sample_tilt + detector_tilt  (standard EBSD geometry)
        # sample_tilt source priority:
        #   1. explicit sample_tilt_deg arg (caller override)
        #   2. geom.sample_tilt_deg (from detector_params["sample_tilt"];
        #      the EXPERIMENTAL tilt set during calibration)
        #   3. master.primary_tilt_deg (MC simulation metadata; last-resort
        #      fallback for callers that don't propagate the calibration —
        #      e.g. unit tests that build a minimal det_params dict)
        #   4. 70.0 (legacy default)
        # Historical bug (2026-05-22): falling back to master.primary_tilt_deg
        # when the route already had the experimental sample_tilt in
        # det_params silently used the MC tilt (70°) instead of the
        # calibration's tilt (75.7° on LoGainNi), producing a fixed
        # ~5.7° rotation offset on every recovered orientation. See
        # tasks/_kikuchipy_oracle_roundtrip.py and the matching regression
        # in tests/test_spherical_gpu/.
        if sample_tilt_deg is not None:
            sigma_deg = float(sample_tilt_deg)
        elif getattr(geom, "sample_tilt_deg", None) is not None:
            sigma_deg = float(geom.sample_tilt_deg)
        elif getattr(master, "primary_tilt_deg", None) is not None:
            sigma_deg = float(master.primary_tilt_deg)
        else:
            sigma_deg = 70.0
        self.sample_tilt_deg = sigma_deg
        self._alpha_tilt = math.radians(sigma_deg - float(geom.tilt_deg))
        # Wait — EMSphInx uses (90 - sTlt + dTlt). Reverting to that formula.
        self._alpha_tilt = math.radians(90.0 - sigma_deg + float(geom.tilt_deg))

        # Precompute per-instance state used in every batch.
        # perf: split into (L+geom)-only and master-dependent stages.
        # Stage 1 (L+geom-only): bind from shared_tables OR build locally.
        # Stage 2 (master-dependent): always build locally — depends on this
        # phase's master file.
        L = self.bandwidth

        if shared_tables is not None:
            # Stage 1 SHARED: skip ~7.5s/phase of redundant build cost.
            shared_tables.assert_compatible(
                bandwidth=L, device=device,
                sample_tilt_deg=self.sample_tilt_deg,
                geom_signature=(),  # caller verifies geom compatibility
            )
            self._rsht = shared_tables.rsht
            self._wigner_table = shared_tables.wigner_table
            self._d_lmk_full = shared_tables.d_lmk_full
            self._sign_factors_L = shared_tables.sign_factors_L
            self._dp_inside_flat = shared_tables.dp_inside_flat
            self._dp_alp = shared_tables.dp_alp
            self._dp_phase = shared_tables.dp_phase
            self._dp_omega = shared_tables.dp_omega
            self._dp_grid_xy = shared_tables.dp_grid_xy
            self._gl_ring_cell_flat = shared_tables.gl_ring_cell_flat
            self._gl_ring_alp = shared_tables.gl_ring_alp
            self._gl_ring_wy = shared_tables.gl_ring_wy
            self._gl_dim = shared_tables.gl_dim
            self._gl_ring_cell_flat_dev = shared_tables.gl_ring_cell_flat_dev
            self._gl_ring_wy_dev = shared_tables.gl_ring_wy_dev
            self._gl_south_ring_gxy = shared_tables.gl_south_ring_gxy
            self._gl_south_ring_inside = shared_tables.gl_south_ring_inside
            self._n_south_inside_per_ring = shared_tables.n_south_inside_per_ring
            self._south_gxy_inside_dev = shared_tables.south_gxy_inside_dev
            self._south_inside_idx_dev = shared_tables.south_inside_idx_dev
            self._dp_south_grid_xy = shared_tables.dp_south_grid_xy
            self._dp_south_omega = shared_tables.dp_south_omega
            self._dp_sht_basis = shared_tables.dp_sht_basis
            self._rescale_factor = shared_tables.rescale_factor
            self._bp_grid_xy = shared_tables.bp_grid_xy
            self._bp_north_mask = shared_tables.bp_north_mask
            self._bp_north_flat = shared_tables.bp_north_flat
            self._bp_south_flat = shared_tables.bp_south_flat
            self._bp_antipodal_flat = shared_tables.bp_antipodal_flat
            self._bp_dh_weights = shared_tables.bp_dh_weights
            self._bp_complete_ring_mask = shared_tables.bp_complete_ring_mask
        else:
            # Stage 1 SOLO: build all (L+geom)-only precomputations locally.
            self._rsht = RSHT(L, device=device, precision="single")
            # Wigner-d table at β=π/2 — used by rs2cc_
            self._wigner_table = wigner_d_eq_half_pi(
                L + 2, dtype=torch.float32, device=device,
            )
            # iter-15F: precompute the FULL (L, 2L-1, 2L-1) wigner-d half-pi
            # table for the einsum-fused rs2cc_fast_ kernel. Avoids the per-l
            # Python loop in rs2cc_, which was the dominant runtime bottleneck
            # (~400 kernel launches per batch -> launch overhead bound).
            self._d_lmk_full = self._build_full_d_lmk_table(L, device)
            # iter-15H: precomputed (-1)^m sign factors for the SHT m-loop
            # vectorization inside _direct_sht_coefs.
            self._sign_factors_L = (-1.0) ** torch.arange(
                L, dtype=torch.float32, device=device,
            )

            # Build GL square-Legendre SHT precomputation tables.
            # For each GL grid cell inside the detector window:
            #   - sphere normal n_p from EMSphInx normals() formula
            #   - normalized ALFs K_l^m * P_l^m(cos θ_p)
            #   - azimuthal phase e^{-iφ_p}
            #   - GL quadrature weight w_p = gl_weight[ar-1] / (8*ar)
            #   - grid_sample coords (X, Y) to sample pattern at this GL cell's position
            (
                self._dp_inside_flat,   # (N_inside,) int64 — flat GL-grid indices
                self._dp_alp,           # (L, L, N_inside) float32 — K_l^m * P_l^m(cos θ_p)
                self._dp_phase,         # (N_inside,) complex64 — e^{-iφ_p}
                self._dp_omega,         # (N_inside,) float32 — GL quadrature weight
                self._dp_grid_xy,       # (N_inside, 2) float32 — grid_sample coords [-1,1]
            ) = self._build_direct_sht_map()

            # Build ring-by-ring SHT tables (EMSphInx DiscreteSHT::analyze equivalent).
            (
                self._gl_ring_cell_flat,  # list[Tensor(8*ar,int64)] for ar=1..half
                self._gl_ring_alp,        # list[Tensor(L,L,float32)] — alp[m,l] per ring
                self._gl_ring_wy,         # Tensor(half,float32) — EMSphInx per-cell ring wt
                self._gl_dim,             # int — GL grid side length
            ) = self._build_gl_ring_tables()

            # iter-15L: precompute every per-ring static tensor on device so the
            # hot SHT loop has zero CPU↔GPU transfers and zero ``.item()`` syncs.
            self._gl_ring_cell_flat_dev = [
                t.to(device) for t in self._gl_ring_cell_flat
            ]
            self._gl_ring_wy_dev = self._gl_ring_wy.to(device)              # (half,)

            # Build south-hemisphere GL ring detector projection tables.
            (
                self._gl_south_ring_gxy,     # list[Tensor(8*ar,2,float32)] — grid_sample coords
                self._gl_south_ring_inside,  # list[Tensor(8*ar,bool)] — inside detector mask
            ) = self._build_south_hemi_gl_ring_map()

            # Per-ring south-inside helpers: precompute Python int (count) +
            # device-side gxy slice and index tensor restricted to inside cells.
            self._n_south_inside_per_ring: list[int] = []
            self._south_gxy_inside_dev: list[torch.Tensor] = []
            self._south_inside_idx_dev: list[torch.Tensor] = []
            for ar_idx, s_ins in enumerate(self._gl_south_ring_inside):
                n_si = int(s_ins.sum().item())
                self._n_south_inside_per_ring.append(n_si)
                if n_si > 0:
                    inside_idx = s_ins.nonzero(as_tuple=True)[0].to(device)
                    gxy_inside = self._gl_south_ring_gxy[ar_idx][s_ins].to(device).contiguous()
                    self._south_inside_idx_dev.append(inside_idx)
                    self._south_gxy_inside_dev.append(gxy_inside)
                else:
                    empty_idx = torch.empty(0, dtype=torch.long, device=device)
                    empty_gxy = torch.empty(0, 2, dtype=torch.float32, device=device)
                    self._south_inside_idx_dev.append(empty_idx)
                    self._south_gxy_inside_dev.append(empty_gxy)

            # FLAT south tensors for normalization. EMSphInx detector.hpp:533-554
            # pushes BOTH north detector projections AND z-flipped (south) detector
            # projections to the iPts list, then unproject() (lines 596-605) computes
            # mu/std over the COMBINED iVal vector. We replicate that by gathering
            # the per-ring south_inside cells into a single flat tensor that pairs
            # with `_dp_grid_xy/_dp_omega` for the normalization step.
            (
                self._dp_south_grid_xy,    # (N_south_inside, 2) float32
                self._dp_south_omega,      # (N_south_inside,) float32
            ) = self._build_flat_south_tensors()

            # Precompute direct-quadrature SHT basis for fast per-pattern SHT.
            # basis[m*L+l, p] = alp[m,l,p] * omega[p] * e^{-im*phi_p}
            self._dp_sht_basis = self._build_sht_basis()

            # Precompute EMSphInx-style pattern rescale factor.
            # scaleFactor(dim) = sqrt(solidAngle(501) * sqrPix / detPix)
            # total scale = sqrt(2 * solidAngle * sqrPix / detPix).
            dim_gl = self._gl_dim
            sq_pix = dim_gl * dim_gl * 2 - (dim_gl - 1) * 4
            det_pix = int(self.geom.pat_h) * int(self.geom.pat_w)
            omega_total = float(self._dp_omega.sum().item())
            solid_angle_frac = omega_total / (4.0 * math.pi)
            self._rescale_factor = math.sqrt(2.0 * solid_angle_frac * sq_pix / det_pix)

            # Build the DH backprojection map — used by _precompute_norm_volume_dh,
            # _build_window_cc, and legacy code that calls _backproject_and_fsht.
            # Must be built BEFORE norm volume precomputation.
            (
                self._bp_grid_xy,          # (n_north, 2) float32  grid_sample coords [-1,1]
                self._bp_north_mask,       # (n_north,) bool — True = inside detector
                self._bp_north_flat,       # (n_north,) int64 — flat idx in (2L,2L) grid
                self._bp_south_flat,       # (n_south,) int64 — flat idx for southern cells
                self._bp_antipodal_flat,   # (n_south,) int64 — flat idx of their N mirror
                self._bp_dh_weights,       # (2L,) float32 — DH quadrature weights per phi ring
                self._bp_complete_ring_mask,  # (n_north,) bool — True if cell's phi ring is fully inside
            ) = self._build_backprojection_map()

        # ---- Stage 2: master-dependent state — always built locally. ----

        # Master coefs in the RSHT (1, L, L) layout [m, l], m in [0, L).
        # Store a version with m=0 row zeroed — used in CC to suppress the
        # orientation-independent (Phi-only) peak from even-l m=0 master coefs.
        # m=0 MUST be zeroed on BOTH sides of rs2cc_ or it corrupts the peak.
        self._master_coefs = self._convert_master_coefs(L)            # (1, L, L) c64
        self._master_coefs_m0z = self._master_coefs.clone()
        self._master_coefs_m0z[:, 0, :] = 0.0

        # iter-15I: Detect master-coefs sparsity along the l axis and prepare
        # active-l + ifftshift-folded tensors for ``rs2cc_sparse``. For Ni
        # m-3m at L=68 only 34 of 68 l-values are nonzero (even-only by
        # inversion symmetry). The contraction in the spectrum einsum is over
        # l, so dropping zero slices halves the dominant cost.
        self._l_active, self._d_mk_pre, self._d_kn_pre = (
            self._build_sparse_rs2cc_tables(L, self._master_coefs, device)
        )
        self._La = int(self._l_active.shape[0])
        # Pre-restrict master coefs along the l axis (last dim).
        self._master_coefs_a = self._master_coefs.index_select(-1, self._l_active).contiguous()
        self._master_coefs_m0z_a = self._master_coefs_m0z.index_select(-1, self._l_active).contiguous()
        # Toggle: only use sparse path if it's actually sparse (>= 25% drop).
        self._use_sparse_cc = self._La <= int(0.75 * L)

        # iter-15K: pre-restrict the per-ring associated-Legendre tables to the
        # active l indices. Depends on BOTH _gl_ring_alp (shared or solo) AND
        # _l_active (always master-dependent), so it must run after Stage 2's
        # _l_active build.
        self._gl_ring_alp_a = [
            alp[..., self._l_active.cpu()].to(device).contiguous()
            for alp in self._gl_ring_alp
        ]

        # EMSphInx NormalizedCorrelator denominator — precomputed from master+window.
        # rDen_vol[R] = 1/sqrt(CC(window,master²)(R) - CC(window,master)(R)²/s2m)
        # Requires _bp_north_flat/_bp_north_mask (Stage 1) AND master coefs (Stage 2).
        self._rDen_vol, self._nc_s2m = self._precompute_nc_denom()  # (1,2L-1,2L-1,2L-1), float
        # perf A1: rDen is shape (1, 2L-1, 2L-1, 2L-1) float32 — at L=88
        # that's ~21 MB. Move once at init so the per-batch hot path is one
        # tensor read instead of a 21 MB CPU->GPU transfer.
        self._rDen_vol = self._rDen_vol.to(self.device)

    # --- Public API ----------------------------------------------------------

    def index_h5oina(
        self,
        h5oina_path: str,
        batch_size: int = 32,
        progress_callback: Optional[callable] = None,
        cancel_check: Optional[callable] = None,
    ) -> IndexResult:
        """Index every pattern in the H5OINA dataset.

        ``cancel_check``: optional zero-arg callable returning True when the
        caller (typically the Spherical-GPU backend wrapper) wants the run
        cancelled. Polled at the start of each batch; raising
        ``CancelledIndexingError`` exits cleanly from the loop.
        """
        t0 = time.perf_counter()
        n_total, dset, dset_path, h5_handle = self._open_pattern_dataset(h5oina_path)
        try:
            all_eulers = torch.empty(n_total, 3, dtype=torch.float32)
            all_scores = torch.empty(n_total, dtype=torch.float32)

            for start in range(0, n_total, batch_size):
                if cancel_check is not None and cancel_check():
                    from indexing_controller import CancelledIndexingError
                    raise CancelledIndexingError(
                        "Spherical-GPU indexing cancelled by user"
                    )
                end = min(start + batch_size, n_total)
                pat_np = np.asarray(dset[start:end], dtype=np.float32)
                pat = torch.from_numpy(pat_np).to(self.device)
                eulers, scores = self._index_batch(pat)
                all_eulers[start:end] = eulers.cpu()
                all_scores[start:end] = scores.cpu()
                if progress_callback:
                    progress_callback(
                        f"Indexed {end}/{n_total} patterns",
                        end / n_total,
                    )
        finally:
            h5_handle.close()

        runtime = time.perf_counter() - t0
        return IndexResult(
            euler_xyz=all_eulers,
            score=all_scores,
            phase_id=torch.ones(n_total, dtype=torch.int8),
            runtime_seconds=runtime,
        )

    def index_h5oina_to_xmap(
        self,
        h5oina_path: str,
        batch_size: int = 32,
        progress_callback: Optional[callable] = None,
    ):
        """Index + return an orix CrystalMap."""
        from .output import to_crystal_map

        result = self.index_h5oina(
            h5oina_path,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
        with h5py.File(h5oina_path, "r") as f:
            n_rows = int(f["1/EBSD/Header/Y Cells"][0])
            n_cols = int(f["1/EBSD/Header/X Cells"][0])
            step_x = (
                float(f["1/EBSD/Header/X Step"][0])
                if "1/EBSD/Header/X Step" in f else 1.0
            )
            step_y = (
                float(f["1/EBSD/Header/Y Step"][0])
                if "1/EBSD/Header/Y Step" in f else 1.0
            )
        return to_crystal_map(
            result, n_rows=n_rows, n_cols=n_cols,
            step_x=step_x, step_y=step_y,
            point_group=self.master.point_group,
            phase_name=self.master.formula,
        )

    # --- Internals -----------------------------------------------------------

    def _run_preprocessing(self, patterns: torch.Tensor) -> torch.Tensor:
        """Apply EMSphInx-style preprocessing in EMSphInx order.

        Honours the indexer's configured ``circmask`` / ``gausbckg`` /
        ``nregions`` flags. EMSphInx applies these in the order:
        circmask → gausbckg → nregions (adaptive histogram equalization).

        ``circmask < 0`` skips the mask entirely; ``0`` uses the inscribed
        circle (matches EMSphInx's default when circmask is on without an
        explicit radius); positive values are treated as an explicit radius
        in pixels — at present only inscribed-circle masking is implemented,
        explicit radius is approximated by the same inscribed mask (good
        enough for square detectors which is the common case).
        """
        out = patterns
        if self.circmask >= 0:
            out = circmask(out)
        if self.gausbckg_on:
            out = gausbckg(out)
        if self.nregions_n > 0:
            out = nregions(out, n=self.nregions_n)
        return out

    def _index_batch_with_volume(
        self, patterns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as _index_batch but ALSO returns the nc_vol and pattern_coefs.

        Used by Tier-2 refinement which needs:
          - top-K from nc_vol (instead of just argmax)
          - pattern_coefs for direct cc-score evaluation at refined orientation

        Returns: (eulers, scores, nc_vol, pattern_coefs)
        """
        prep = self._run_preprocessing(patterns)
        # iter-15I/K: when the master has l-axis sparsity (m-3m, etc.) compute
        # the SHT only at active l (B, L, La) and pass straight to rs2cc_sparse.
        # Otherwise compute full (B, L, L) and use rs2cc_vectorized.
        if self.cc_fp64:
            pattern_coefs = self._direct_sht_coefs(prep)
            cc = rs2cc_fp64(
                self.bandwidth,
                pattern_coefs,
                self._master_coefs,
                self._wigner_table,
            )
        elif self._use_sparse_cc:
            # Compute the full-l SHT so the documented (B, L, L) pattern_coefs
            # is returned for Tier-2; slice the active-l subset for rs2cc_sparse.
            # (mirrors _index_batch, which pays the same cost for the same reason)
            pattern_coefs = self._direct_sht_coefs(prep)  # (B, L, L)
            pattern_coefs_a = pattern_coefs.index_select(
                -1, self._l_active
            ).contiguous()  # (B, L, La)
            cc = rs2cc_sparse(
                self.bandwidth,
                self._La,
                pattern_coefs_a,
                self._master_coefs_a,
                self._d_mk_pre,
                self._d_kn_pre,
            )
        else:
            pattern_coefs = self._direct_sht_coefs(prep)
            cc = rs2cc_vectorized(
                self.bandwidth,
                pattern_coefs,
                self._master_coefs,
                self._d_lmk_full,
            )
        # A1: rDen lives on device; no per-batch transfer.
        # perf Q0.4: cupy RawKernel fuses mul+max+argmax into one CUDA kernel.
        nc_vol, max_vals, max_idx = _fused_mul_max_dispatch(
            cc.real.float(), self._rDen_vol,
        )
        # Decode argmax for compatibility with the existing API.
        eulers, scores = self._decode_peak(nc_vol, max_vals, max_idx)
        return eulers, scores, nc_vol, pattern_coefs

    def _decode_peak(
        self,
        nc_vol: torch.Tensor,
        max_values: Optional[torch.Tensor] = None,
        max_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode argmax of nc_vol into (eulers ZXZ rad, scores).

        When ``self.refine`` is True (default — the EMSphInx ``refine`` flag),
        applies a 3x3x3 triquadratic Newton step around the argmax for sub-bin
        precision (~0.05 deg vs the ~2.66 deg cc bin spacing at L=68). Note
        this is NOT the same as EMSphInx's continuous-SHT ``refineNewton`` —
        that is a separate, more expensive step (a few ms per pattern) that
        operates on the full sphere coefficients rather than the cc bins. We
        document this difference so callers know the precision floor.

        When ``self.refine`` is False, returns the integer-bin argmax cast
        to float, equivalent to EMSphInx ``refine=False``.

        perf Q0.4: callers may pass precomputed (max_values, max_indices)
        from the cupy fused mul+max kernel to skip the local max pass entirely.
        """
        from .refiner import triquadratic_subbin
        B = nc_vol.shape[0]
        size = 2 * self.bandwidth - 1
        # perf A5: fuse argmax+max into one pass via torch.max(dim=1)
        # which returns (values, indices) together. Old code did argmax
        # (one full pass over 5.4M-element nc_vol per pattern at L=88)
        # then a separate .max().values for the score (another full pass).
        # Combined max is ~half the bandwidth on the decode stage.
        if max_values is None or max_indices is None:
            flat = nc_vol.reshape(B, -1)
            max_result = flat.max(dim=1)
            flat_idx = max_result.indices
            _scores_max = max_result.values.to(torch.float32)
        else:
            flat_idx = max_indices
            _scores_max = max_values.to(torch.float32)
        a_idx = (flat_idx // (size * size)).long()
        rem   = flat_idx % (size * size)
        b_idx = (rem // size).long()
        c_idx = (rem % size).long()

        if self.refine:
            # Sub-bin refinement via triquadratic Newton step.
            a_sub, b_sub, c_sub = triquadratic_subbin(nc_vol, a_idx, b_idx, c_idx)
        else:
            a_sub = a_idx.to(torch.float64)
            b_sub = b_idx.to(torch.float64)
            c_sub = c_idx.to(torch.float64)

        two_pi = 2.0 * math.pi
        scale  = two_pi / size
        a_f = a_sub.to(torch.float64)
        b_f = b_sub.to(torch.float64)
        c_f = c_sub.to(torch.float64)
        half_pi = math.pi / 2.0
        off = float(size // 2)
        size_f = float(size)
        alpha  = ((a_f - off + size_f).fmod(size_f) + size_f).fmod(size_f) * scale
        gamma_ = ((off - c_f + size_f).fmod(size_f) + size_f).fmod(size_f) * scale
        Phi    = b_f * scale
        phi1   = (alpha + half_pi) % two_pi
        phi2   = (gamma_ - half_pi) % two_pi
        phi1   = (phi1 + 3.0 * half_pi) % two_pi
        eulers = torch.stack([phi1, Phi, phi2], dim=1).to(torch.float32)
        # A5: scores already computed by the fused max above; reuse instead
        # of doing a second full pass over nc_vol.
        return eulers, _scores_max

    def _decode_peak_skip_vol(
        self,
        cc_real: torch.Tensor,            # (B, S, S, S) contiguous real cc
        max_values: torch.Tensor,         # (B,)
        max_indices: torch.Tensor,        # (B,) int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode peak without a materialised nc_vol.

        Lever A counterpart to ``_decode_peak``. Pairs with the
        ``fused_max_only`` cupy kernel: instead of indexing into a 21.4 MB
        per-pattern materialised volume, we re-read 27 cc values and
        27 rDen values from the original tensors and multiply on the fly.
        Saves the volume's DRAM round-trip and 654 MB allocation per batch.

        For ``self.refine=True``, sub-bin refinement uses the gathered
        ``n_27`` (B, 27) tensor via the new ``triquadratic_subbin_n27``
        API. The result is bit-identical to ``_decode_peak`` because the
        gather indices and the (cc * rDen) values are mathematically the
        same as what the legacy path computed by indexing into nc_vol.
        """
        from .refiner import gather_n27_from_cc_rden, triquadratic_subbin_n27
        size = 2 * self.bandwidth - 1
        flat_idx = max_indices
        _scores_max = max_values.to(torch.float32)
        a_idx = (flat_idx // (size * size)).long()
        rem   = flat_idx % (size * size)
        b_idx = (rem // size).long()
        c_idx = (rem % size).long()

        if self.refine:
            n_27 = gather_n27_from_cc_rden(cc_real, self._rDen_vol, a_idx, b_idx, c_idx)
            a_sub, b_sub, c_sub = triquadratic_subbin_n27(n_27, a_idx, b_idx, c_idx)
        else:
            a_sub = a_idx.to(torch.float64)
            b_sub = b_idx.to(torch.float64)
            c_sub = c_idx.to(torch.float64)

        two_pi = 2.0 * math.pi
        scale  = two_pi / size
        a_f = a_sub.to(torch.float64)
        b_f = b_sub.to(torch.float64)
        c_f = c_sub.to(torch.float64)
        half_pi = math.pi / 2.0
        off = float(size // 2)
        size_f = float(size)
        alpha  = ((a_f - off + size_f).fmod(size_f) + size_f).fmod(size_f) * scale
        gamma_ = ((off - c_f + size_f).fmod(size_f) + size_f).fmod(size_f) * scale
        Phi    = b_f * scale
        phi1   = (alpha + half_pi) % two_pi
        phi2   = (gamma_ - half_pi) % two_pi
        phi1   = (phi1 + 3.0 * half_pi) % two_pi
        eulers = torch.stack([phi1, Phi, phi2], dim=1).to(torch.float32)
        return eulers, _scores_max

    def _index_batch(
        self, patterns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index a batch of patterns. Returns (eulers, scores) on device.

        Implements EMSphInx GL square-Legendre SHT + SO(3) cross-correlation:

        1. Preprocess: gausbckg + nregions (matches EMSphInx ImageProcessor).
        2. GL square-Legendre ring SHT: south cells that project to the detector
           are added to north hemisphere ring positions (south hemisphere of SHT
           is zero, matching EMSphInx BackProjector::unproject convention).
        3. Zero out m=0 row of pattern coefs to suppress the orientation-
           independent DC peak (matched by zeroing m=0 on master side too).
        4. Cross-correlate pattern vs master: CC(pattern_m0z, master_m0z)(R).
        5. Peak → Euler angles via the ZYZ→ZXZ decode formula.
        """
        device = self.device

        # Stage 1: preprocessing — match EMSphInx ImageProcessor chain:
        #   gausbckg (Gaussian background subtraction) then nregions=10
        #   (adaptive histogram equalization by region).
        prep = self._run_preprocessing(patterns)

        # Stage 2: GL square-Legendre ring SHT (EMSphInx convention).
        # South cells that project to detector are combined with north cells
        # in the north hemisphere ring buffer; SHT south hemisphere = 0.
        pattern_coefs = self._direct_sht_coefs(prep)                   # (B, L, L) c64

        # Stage 3 EXPERIMENT 2026-05-03: don't zero m=0 (EMSphInx idx.hpp
        # doesn't zero m=0 — the orientation-independent peak is supposed to
        # be suppressed by NormalizedCorrelator's denominator, not by
        # zeroing). Test if this fixes the bimodal disorientation.
        # iter-15I/K: rs2cc_sparse exploits master l-axis sparsity + folds
        # ifftshift into d_mk/d_kn. The full-l SHT is still needed here
        # because this code path returns pattern_coefs (B, L, L) for Tier-2.
        if self.cc_fp64:
            cc = rs2cc_fp64(
                self.bandwidth,
                pattern_coefs,
                self._master_coefs,
                self._wigner_table,
            )
        elif self._use_sparse_cc:
            f_a = pattern_coefs.index_select(-1, self._l_active).contiguous()
            cc = rs2cc_sparse(
                self.bandwidth,
                self._La,
                f_a,
                self._master_coefs_a,
                self._d_mk_pre,
                self._d_kn_pre,
            )
        else:
            cc = rs2cc_vectorized(
                self.bandwidth,
                pattern_coefs,
                self._master_coefs,
                self._d_lmk_full,
            )                                                          # (B, 2L-1, 2L-1, 2L-1)

        # Stage 4b: apply NC normalization — multiply CC by rDen_vol.
        # NC(R) = CC(pattern, master)(R) * rDen(R)
        # where rDen(R) = 1/sqrt(CC(window,master²)(R) - CC(window,master)(R)²/s2m).
        # Lever A (2026-05-11): skip nc_vol materialisation. The fused_max_only
        # cupy kernel streams (cc * rDen) through registers and only writes
        # the per-pattern argmax. Sub-bin refinement gathers the 27 needed
        # nc values on the fly from cc and rDen. Saves the 21.4 MB/pattern
        # DRAM round-trip and the 654 MB allocation at B=32, L=88.
        cc_real = cc.real.float().contiguous()
        max_vals, max_idx = _fused_max_only_dispatch(cc_real, self._rDen_vol)

        # Stage 5: find peak → Euler angles. _decode_peak_skip_vol uses
        # gather_n27_from_cc_rden when refine=True instead of indexing nc_vol.
        return self._decode_peak_skip_vol(cc_real, max_vals, max_idx)

    def _index_prepped_batch(
        self, prep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index a batch of ALREADY-preprocessed patterns.

        Same as ``_index_batch`` but skips the ``_run_preprocessing`` step
        — the caller has already produced the preprocessed tensor. Used by
        ``SphericalGPUBackend._index_with_phases_interleaved`` to share
        the preprocessing + H5 read across phases that have identical
        preprocessing flags. ~2-3x speedup for multi-phase L=88 since
        the gausbckg/circmask/nregions chain at L=88 isn't free.
        """
        device = self.device
        # Stage 2 onward — identical to _index_batch but starts from prep.
        if self.cc_fp64:
            pattern_coefs = self._direct_sht_coefs(prep)
            cc = rs2cc_fp64(
                self.bandwidth, pattern_coefs, self._master_coefs,
                self._wigner_table,
            )
        elif self._use_sparse_cc:
            pattern_coefs_a = self._direct_sht_coefs_active(prep)
            cc = rs2cc_sparse(
                self.bandwidth, self._La, pattern_coefs_a,
                self._master_coefs_a, self._d_mk_pre, self._d_kn_pre,
            )
        else:
            pattern_coefs = self._direct_sht_coefs(prep)
            cc = rs2cc_vectorized(
                self.bandwidth, pattern_coefs, self._master_coefs,
                self._d_lmk_full,
            )
        # Lever A: skip nc_vol materialisation (see _index_batch comment).
        cc_real = cc.real.float().contiguous()
        max_vals, max_idx = _fused_max_only_dispatch(cc_real, self._rDen_vol)
        return self._decode_peak_skip_vol(cc_real, max_vals, max_idx)

    def _direct_sht_coefs(self, patterns: torch.Tensor) -> torch.Tensor:
        """Compute SHT via EMSphInx BackProjector::unproject + DiscreteSHT::analyze.

        Implements the EMSphInx EBSD back-projection and SHT chain:

        1. Sample normalized pattern at GL inside-cells (north hemisphere only
           for the pre-scattered gl_grid).
        2. For each ring ar = 1..half:
             a. Build combined ring buffer = north cell values + south cell values
                at the same ring positions (south cells add to north positions,
                matching EMSphInx ``sph[idx] = val`` for both north and south cells).
             b. rfft of combined buffer → G_m for m = 0..min(L-1, 4·ar).
             c. Accumulate (south hemisphere of SHT is always zero):
                  coefs[b, m, l] += (-1)^m · G[b,m] · wy[ar] · alp_ar[m, l]
                for ALL l ≥ m (no even/odd parity split).
           where wy[ar] is the per-cell EMSphInx ring weight (from solving the
           Chebyshev linear system in ``_compute_ring_weights``).

        Parameters
        ----------
        patterns : (B, H, W) float32

        Returns
        -------
        coefs : (B, L, L) complex64
        """
        B, H, W = patterns.shape
        L = self.bandwidth
        device = self.device
        dim = self._gl_dim
        n_gl = dim * dim

        omega = self._dp_omega.to(device)          # (N_inside,) float32
        gxy   = self._dp_grid_xy.to(device)        # (N_inside, 2) float32

        # FIX 2026-05-03: include south-hemisphere detector projections in the
        # mu/std calculation. EMSphInx detector.hpp:533-554 pushes BOTH north
        # and z-flipped (south) detector projections into the iPts list, and
        # unproject() (lines 596-605) computes mean/stdev over the COMBINED
        # iVal vector weighted by the COMBINED omega. Our previous version
        # normalized using north-only stats, which under-counted ~half the
        # detector signal whenever south projections also hit (typical for
        # EBSD geometry). Discrepancy was a candidate root cause for the
        # bimodal disorientation distribution we observed.
        s_gxy   = self._dp_south_grid_xy.to(device)   # (N_south_inside, 2) f32
        s_omega = self._dp_south_omega.to(device)     # (N_south_inside,)   f32

        omega_sum = (omega.sum() + s_omega.sum()).clamp_min(1e-12)

        # --- Stage 1: sample at GL inside-cells via grid_sample ---------------
        pat_4d  = patterns.unsqueeze(1).float()           # (B, 1, H, W)

        # EMSphInx BackProjector::unproject rescales (downsamples) the pattern
        # before interpolating at GL positions.  The scale = scaleFactor(dim) * sqrt(2)
        # ≈ 0.21 for a 156×128 detector at bw=68, reducing each axis by ~5× so
        # that one GL sample averages ~20 original pixels — a ~4.5× SNR improvement.
        # The grid_sample coords (grid_xy) are in [-1,1] normalised space and
        # remain valid at any resolution, so we can downsample pat_4d first.
        # FIX 2026-05-07 (iter-14 root cause #2): use bicubic-AA downsampling
        # instead of adaptive_avg_pool2d. EMSphInx Image::Rescaler uses DCT2D
        # for resizing — bicubic with anti-aliasing is the closest torch-native
        # equivalent (DumpTopK comparison: r=0.78 bicubic vs r=0.79 DCT vs
        # r=0.74 avg_pool, on common cells of pixel-0 sphere image).
        scale = self._rescale_factor
        w_out = max(1, round(W * scale))
        h_out = max(1, round(H * scale))
        if w_out < W or h_out < H:
            pat_4d = F.interpolate(
                pat_4d, (h_out, w_out),
                mode="bicubic", align_corners=False, antialias=True,
            )

        # FIX 2026-05-07 (iter-14 root cause #3): align_corners=True matches
        # EMSphInx bilinearCoeff()'s `x *= w-1; y *= h-1` indexing (image.hpp
        # line 528). Default align_corners=False maps X_frac=0 to half-pixel
        # OUTSIDE the leftmost pixel, which mis-locates every sample by ~0.5
        # pixels. After this fix the sphere-image correlation jumps from
        # r=0.74 to r=0.90 on common cells.
        grid_4d = gxy.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        sampled = F.grid_sample(
            pat_4d, grid_4d,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(1).squeeze(1)                           # (B, N_inside)

        # Sample south projections too (for normalization stats).
        if s_gxy.shape[0] > 0:
            grid_s_4d = s_gxy.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
            sampled_s = F.grid_sample(
                pat_4d, grid_s_4d,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            ).squeeze(1).squeeze(1)                        # (B, N_south_inside)
        else:
            sampled_s = torch.empty(B, 0, dtype=sampled.dtype, device=device)

        # --- Stage 2: normalize over BOTH hemispheres' contributions ---------
        # mu = sum(iVal * omega) / sum(omega), iVal = north + south concatenated
        n_term = (sampled   * omega.unsqueeze(0)  ).sum(dim=1, keepdim=True)
        s_term = (sampled_s * s_omega.unsqueeze(0)).sum(dim=1, keepdim=True)
        mu  = (n_term + s_term) / omega_sum

        res_n = sampled   - mu
        res_s = sampled_s - mu
        v_n = (res_n ** 2 * omega.unsqueeze(0)  ).sum(dim=1, keepdim=True)
        v_s = (res_s ** 2 * s_omega.unsqueeze(0)).sum(dim=1, keepdim=True)
        var = (v_n + v_s) / omega_sum
        std = var.sqrt().clamp_min(1e-6)
        # EMSphInx ``normed`` flag: when True (default), divide by std to
        # produce a zero-mean, unit-variance pattern before SHT — improves
        # robustness against per-pattern intensity scale variation. When
        # False, only mean-subtract; the cc score then weighs absolute
        # pattern intensity which can hurt cross-pattern comparability.
        f_norm = (res_n / std) if self.normed else res_n   # (B, N_inside)
        # f_norm_s computed lazily — south sampling for SHT happens per-ring
        # below; mu/std is what we needed and downstream code uses (mu, std)
        # variables which now reflect the combined statistics.

        # --- Stage 3: scatter north hemisphere values into full GL sphere grid ----
        inside_flat = self._dp_inside_flat.to(device)    # (N_inside,) int64
        gl_grid = torch.zeros(B, n_gl, dtype=torch.float32, device=device)
        gl_grid.scatter_(1, inside_flat.unsqueeze(0).expand(B, -1), f_norm)

        # --- Stage 4: ring-by-ring rfft + Legendre accumulation ---------------
        # Implements EMSphInx BackProjector::unproject + DiscreteSHT::analyze().
        #
        # EMSphInx convention (detector.hpp lines 533-556):
        #   Both north AND south hemisphere cells that project to the detector
        #   write their values to sph[idx] where idx is the NORTH hemisphere
        #   flat grid index (i.e. south cells overwrite/add to the same positions
        #   as north cells in the north hemisphere array).  The SHT then receives
        #   north_hemi = sph[0..dim*dim), south_hemi = sph[dim*dim..2*dim*dim) = 0.
        #
        # With south_hemi = 0:
        #   gmyS = (G_N + 0) * wy * sign * 0.5 = G_N * factor
        #   gmyA = (G_N - 0) * wy * sign * 0.5 = G_N * factor
        #   → both even and odd l+m rows get the same contribution G_N * factor
        #   → coefs[m, l] += G_N[m] * wy * sign * 0.5 * alp[m, l]  for ALL l ≥ m
        #
        # The north hemisphere ring buffer for ring ar therefore contains the SUM
        # of north cell values AND south cell values at the same azimuthal positions.
        coefs = torch.zeros(B, L, L, dtype=torch.complex64, device=device)
        ring_wy    = self._gl_ring_wy             # (half,) float32, on CPU
        ring_flat  = self._gl_ring_cell_flat       # list of Tensor(8*ar,)
        ring_alp   = self._gl_ring_alp             # list of Tensor(L, L)
        south_gxy  = self._gl_south_ring_gxy       # list of Tensor(8*ar, 2)
        south_ins  = self._gl_south_ring_inside    # list of Tensor(8*ar,) bool

        for ar_idx in range(len(ring_flat)):
            ar = ar_idx + 1
            n_pts = 8 * ar

            # --- North hemisphere values (from pre-scattered gl_grid) ---
            flat_dev = ring_flat[ar_idx].to(device)       # (8*ar,) int64
            combined_vals = gl_grid[:, flat_dev].clone()   # (B, 8*ar) float32

            # --- Add south hemisphere cells to the same north ring positions ---
            # EMSphInx: south cells write to sph[idx] (north hemisphere array),
            # so their values are ADDED to (or replace) north values at those idx.
            # Here we ADD them because each north position is hit by at most one
            # of north or south — the detector polar cap is far from the equator.
            s_ins = south_ins[ar_idx].to(device)           # (8*ar,) bool
            n_south_inside = int(s_ins.sum().item())
            if n_south_inside > 0:
                s_gxy_dev  = south_gxy[ar_idx].to(device)  # (8*ar, 2) float32
                inside_idx = s_ins.nonzero(as_tuple=True)[0]  # (n_south_inside,)
                gxy_inside = s_gxy_dev[inside_idx]          # (n_south_inside, 2)
                grid_s = gxy_inside.unsqueeze(0).unsqueeze(0).expand(
                    B, 1, -1, -1
                )                                           # (B, 1, n_s, 2)
                raw_s = F.grid_sample(
                    pat_4d, grid_s,
                    mode="bilinear", padding_mode="zeros", align_corners=True,
                ).squeeze(1).squeeze(1)                     # (B, n_south_inside)
                # Normalize with same mean/std as north
                raw_s = (raw_s - mu) / std                  # (B, n_south_inside)
                # scatter_add_: south values ADD onto north values at same indices
                combined_vals.scatter_add_(
                    1,
                    inside_idx.unsqueeze(0).expand(B, -1),
                    raw_s,
                )

            # --- Single FFT of combined (north + south) ring values ---
            # South hemisphere of SHT is zero (EMSphInx convention).
            G = torch.fft.rfft(combined_vals, n=n_pts, dim=1)  # (B, 4*ar+1) c64

            mLim    = min(L, 4 * ar + 1)
            wy      = float(ring_wy[ar_idx].item())
            alp_dev = ring_alp[ar_idx].to(device)              # (L, L) float32

            # iter-15H: vectorize the inner m-loop. Per-m factor is
            # wy * (-1)^m * 0.5; we apply it as an elementwise multiply on
            # the rfft result, then a single (B, L) -> (B, L, L) outer
            # product against alp[m, l] (which already has zeros for l<m).
            G_padded = torch.zeros(B, L, dtype=G.dtype, device=device)
            G_padded[:, :mLim] = G[:, :mLim]
            sign_factors = (
                self._sign_factors_L
                if hasattr(self, "_sign_factors_L")
                else (-1.0) ** torch.arange(L, dtype=torch.float32, device=device)
            )
            G_weighted = G_padded * (sign_factors * (wy * 0.5)).to(G.dtype)
            # alp_dev shape (L, L): alp[m, l] is zero for l < m, so the
            # outer product naturally masks invalid (m, l) entries.
            coefs += G_weighted.unsqueeze(2) * alp_dev.unsqueeze(0).to(G.dtype)

        return coefs                                       # (B, L, L) c64

    def _direct_sht_coefs_active(self, patterns: torch.Tensor) -> torch.Tensor:
        """SHT producing only the active l columns (B, L, La).

        Identical to ``_direct_sht_coefs`` except the per-ring (m, l) outer-
        product is restricted to l_active, halving the multiply-add work in
        the ring accumulation for symmetry-driven sparse phases. Output
        feeds straight into ``rs2cc_sparse`` without an index_select.
        """
        B, H, W = patterns.shape
        L = self.bandwidth
        La = self._La
        device = self.device
        dim = self._gl_dim
        n_gl = dim * dim

        omega = self._dp_omega.to(device)
        gxy   = self._dp_grid_xy.to(device)
        s_gxy   = self._dp_south_grid_xy.to(device)
        s_omega = self._dp_south_omega.to(device)
        omega_sum = (omega.sum() + s_omega.sum()).clamp_min(1e-12)

        pat_4d  = patterns.unsqueeze(1).float()
        scale = self._rescale_factor
        w_out = max(1, round(W * scale))
        h_out = max(1, round(H * scale))
        if w_out < W or h_out < H:
            pat_4d = F.interpolate(
                pat_4d, (h_out, w_out),
                mode="bicubic", align_corners=False, antialias=True,
            )

        grid_4d = gxy.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        sampled = F.grid_sample(
            pat_4d, grid_4d,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(1).squeeze(1)

        if s_gxy.shape[0] > 0:
            grid_s_4d = s_gxy.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
            sampled_s = F.grid_sample(
                pat_4d, grid_s_4d,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            ).squeeze(1).squeeze(1)
        else:
            sampled_s = torch.empty(B, 0, dtype=sampled.dtype, device=device)

        n_term = (sampled   * omega.unsqueeze(0)  ).sum(dim=1, keepdim=True)
        s_term = (sampled_s * s_omega.unsqueeze(0)).sum(dim=1, keepdim=True)
        mu  = (n_term + s_term) / omega_sum

        res_n = sampled   - mu
        res_s = sampled_s - mu
        v_n = (res_n ** 2 * omega.unsqueeze(0)  ).sum(dim=1, keepdim=True)
        v_s = (res_s ** 2 * s_omega.unsqueeze(0)).sum(dim=1, keepdim=True)
        var = (v_n + v_s) / omega_sum
        std = var.sqrt().clamp_min(1e-6)
        f_norm = (res_n / std) if self.normed else res_n

        inside_flat = self._dp_inside_flat.to(device)
        gl_grid = torch.zeros(B, n_gl, dtype=torch.float32, device=device)
        gl_grid.scatter_(1, inside_flat.unsqueeze(0).expand(B, -1), f_norm)

        coefs = torch.zeros(B, L, La, dtype=torch.complex64, device=device)
        ring_flat_dev    = self._gl_ring_cell_flat_dev          # list of device tensors
        ring_wy_dev      = self._gl_ring_wy_dev                 # (half,) on device
        ring_alp_a       = self._gl_ring_alp_a                  # list of (L, La) on device
        south_inside_idx = self._south_inside_idx_dev           # list of (n_si,) on device
        south_gxy_inside = self._south_gxy_inside_dev           # list of (n_si, 2) on device
        n_si_list        = self._n_south_inside_per_ring        # list of Python ints
        sign_factors     = self._sign_factors_L
        n_rings = len(ring_flat_dev)

        for ar_idx in range(n_rings):
            ar = ar_idx + 1
            n_pts = 8 * ar

            combined_vals = gl_grid[:, ring_flat_dev[ar_idx]].clone()

            n_si = n_si_list[ar_idx]                            # Python int — no sync
            if n_si > 0:
                gxy_inside = south_gxy_inside[ar_idx]
                inside_idx = south_inside_idx[ar_idx]
                grid_s = gxy_inside.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
                raw_s = F.grid_sample(
                    pat_4d, grid_s,
                    mode="bilinear", padding_mode="zeros", align_corners=True,
                ).squeeze(1).squeeze(1)
                raw_s = (raw_s - mu) / std
                combined_vals.scatter_add_(
                    1, inside_idx.unsqueeze(0).expand(B, -1), raw_s,
                )

            G = torch.fft.rfft(combined_vals, n=n_pts, dim=1)
            mLim    = min(L, 4 * ar + 1)
            alp_dev = ring_alp_a[ar_idx]                       # (L, La) c64-promotable

            G_padded = torch.zeros(B, L, dtype=G.dtype, device=device)
            G_padded[:, :mLim] = G[:, :mLim]
            # 0-d device tensor for wy*0.5 — no .item() sync.
            wy_half = ring_wy_dev[ar_idx] * 0.5
            G_weighted = G_padded * (sign_factors * wy_half).to(G.dtype)
            coefs += G_weighted.unsqueeze(2) * alp_dev.unsqueeze(0).to(G.dtype)

        return coefs                                       # (B, L, La) c64

    def _build_direct_sht_map(self):
        """Precompute GL square-Legendre grid cells and SHT basis.

        Uses EMSphInx's square Legendre grid (``normals()`` from
        ``include/util/square_sht.hpp``) to generate sample points on the
        sphere.  Each cell is forward-projected to detector fractional
        coordinates (X, Y) so that at inference time we can sample the
        pattern value at that GL grid point via grid_sample.

        The GL quadrature weight for cell p in ring ar is:
            w_p = gl_weight[ar-1] / (8 * ar)   (uniform in azimuth per ring)

        Returns
        -------
        inside_flat : (N_inside,) int64
            Flat indices into the GL grid (dim*dim) of cells whose
            forward-projected detector coords fall inside the window.
        alp         : (L, L, N_inside) float32 — K_l^m * P_l^m(cos θ_p)
        phase       : (N_inside,) complex64 — e^{-iφ_p}
        omega       : (N_inside,) float32 — GL quadrature weight per cell
        grid_xy_gl  : (N_inside, 2) float32 — grid_sample coords for each
                      inside GL cell [in (-1,1)^2 detector space]
        """
        from numpy.polynomial.legendre import leggauss

        L = self.bandwidth
        H, W = self.geom.pat_h, self.geom.pat_w

        # ---- GL grid dimensions (matching EMSphInx) -----------------------
        # dim must be odd; EMSphInx uses dim = bw + 3 if bw even, bw + 2 if odd
        dim = L + 3 if (L % 2 == 0) else L + 2
        if dim % 2 == 0:
            dim += 1   # enforce odd
        half = dim // 2   # ring index range: 1..half

        # ---- GL quadrature nodes and weights (north hemisphere only) -------
        # EMSphInx calls roots(dim-2, cosLats) which fills the half=dim//2
        # largest (positive) roots of the (dim-2)-th Legendre polynomial.
        # numpy's leggauss gives all dim-2 roots; sort by decreasing cos and
        # take the first 'half' (= the non-negative ones).
        n_gl = dim - 2
        cos_nodes_all, gl_weights_all = leggauss(n_gl)
        sort_idx = np.argsort(cos_nodes_all)[::-1].copy()
        cos_nodes_all = cos_nodes_all[sort_idx]
        gl_weights_all = gl_weights_all[sort_idx]
        # North hemisphere nodes: first 'half' entries (cos >= 0 approximately)
        cos_nodes = cos_nodes_all[:half]        # (half,) in [~0, ~1]
        gl_weights = gl_weights_all[:half]      # (half,) GL integration weights

        # ---- Build square GL grid normals (EMSphInx normals() formula) -----
        # EMSphInx uses C-style j=row, i=col with ri=i-half (col offset) and
        # rj=j-half (row offset). With meshgrid(indexing="ij"), I_g varies along
        # dim0 (row) and J_g along dim1 (col). We therefore assign:
        #   ri = J_g - half   (col offset  -- matches C++ ri)
        #   rj = I_g - half   (row offset  -- matches C++ rj)
        #   sX = ri/ar        (col normalised -- matches C++ sX)
        #   sY = rj/ar        (row normalised -- matches C++ sY)
        #   Case A: ai=|col| <= aj=|row|  (matches C++ 'if(ai <= aj)')
        i_idx = torch.arange(dim, dtype=torch.float32)
        j_idx = torch.arange(dim, dtype=torch.float32)
        I_g, J_g = torch.meshgrid(i_idx, j_idx, indexing="ij")   # (dim, dim)
        # ri = col offset, rj = row offset  (matches EMSphInx legendre::normals)
        ri = J_g - half   # col offset
        rj = I_g - half   # row offset
        ai = ri.abs()     # abs col
        aj = rj.abs()     # abs row
        ar = torch.maximum(ai, aj)   # ring index (0 = pole, 1..half = rings)

        # Normalized coordinates: sX = col/ar, sY = row/ar (clamp ar=0 → 1)
        ar_safe = ar.clamp_min(1.0)
        sX = ri / ar_safe   # col normalised
        sY = rj / ar_safe   # row normalised

        kPi_4 = math.pi / 4.0

        # Case A: |col| <= |row|  (ai <= aj)
        qq_a = kPi_4 * sX * sY
        hx_a = sY * torch.sin(qq_a)
        hy_a = sY * torch.cos(qq_a)

        # Case B: |col| > |row|   (ai > aj)
        qq_b = kPi_4 * sY * sX    # same as qq_a (product is symmetric)
        hx_b = sX * torch.cos(qq_b)
        hy_b = sX * torch.sin(qq_b)

        case_a = (ai <= aj) & (ar > 0)
        case_b = (ai > aj)
        hx = torch.where(case_a, hx_a, torch.where(case_b, hx_b, torch.zeros_like(hx_a)))
        hy = torch.where(case_a, hy_a, torch.where(case_b, hy_b, torch.zeros_like(hy_a)))
        h_mag = torch.hypot(hx, hy).clamp_min(1e-15)

        # cos-latitude per cell from GL lookup table (indexed by ar-1)
        cos_lat = torch.zeros(dim, dim, dtype=torch.float32)
        for ar_val in range(1, half + 1):
            mask = (ar == ar_val)
            cos_lat[mask] = float(cos_nodes[ar_val - 1])
        cos_lat[half, half] = 1.0   # pole
        sin_lat = (1.0 - cos_lat ** 2).clamp_min(0.0).sqrt()

        # Sphere normals: n2 = cos_lat; n0,n1 = sin_lat * (hx,hy)/h_mag
        n_n0 = sin_lat * hx / h_mag
        n_n1 = sin_lat * hy / h_mag
        n_n2 = cos_lat
        n_n0[half, half] = 0.0
        n_n1[half, half] = 0.0
        # n_n2[half, half] already 1.0

        # ---- Geometry parameters (for forward projection) ------------------
        alpha_tilt = self._alpha_tilt
        sA = math.sin(alpha_tilt)
        cA = math.cos(alpha_tilt)

        if self.geom.cx is not None:
            cX = float(self.geom.cx.item())
            cY = float(self.geom.cy.item())
        else:
            cX = -float(self.geom.xpc.item())
            cY = float(self.geom.ypc.item())
        Lscint = float(self.geom.L.item())
        delta  = self.geom.pixel_size

        # ---- Forward-project GL normals → detector fractional coords ----------
        # EMSphInx detector.hpp interpolatePixel (no R_y rotation):
        #   denom = n0*sA + n2*cA
        #   d = sDst / denom
        #   x = n1 * d
        #   y = (sA*n2 - cA*n0) * d
        #   X = (cX + x/pX) / w + 0.5
        #   Y = (cY + y/pY) / h + 0.5
        #   [then flip_y applied AFTER bounds check for Oxford: Y = 1 - Y]
        #
        # GL north pole (0,0,1) → denom = cA, d = Lscint/cA,
        # x=0, y = (sA - 0)/cA * Lscint/cA... gives the on-axis PC point.
        # The quNp quaternion in EMSphInx is the identity (northPoleQuat).
        # NO R_y rotation is applied to the GL normals before projection.
        denom = n_n0 * sA + n_n2 * cA           # (dim, dim)
        valid = denom > 1e-9
        denom_s = denom.clamp_min(1e-9)
        d_proj  = Lscint / denom_s
        x_um = n_n1 * d_proj
        y_um = (sA * n_n2 - cA * n_n0) * d_proj

        X_frac = (cX + x_um / delta) / W + 0.5
        Y_frac = (cY + y_um / delta) / H + 0.5
        # Oxford patterns are stored top-down (image convention) while the
        # detector.hpp projection uses a bottom-up y-axis. Flip Y after
        # computing fractional coords to convert to image-top-down space.
        if self.flip_y:
            Y_frac = 1.0 - Y_frac

        # Rectangular window check
        in_rect = (
            valid
            & (X_frac >= 0.0) & (X_frac <= 1.0)
            & (Y_frac >= 0.0) & (Y_frac <= 1.0)
        )                                         # (dim, dim) bool

        # FIX 2026-05-07 (iter-14 root-cause): EMSphInx idx.hpp:230 sets
        # `geom.maskPattern(nml.circRad == 0)` — the circular mask is ONLY
        # enabled when circRad == 0. For circRad == -1 (no mask, the oracle's
        # default) the inscribed-circle check is NOT applied. Our pipeline
        # was applying it unconditionally, dropping ~30% of valid detector
        # cells at the polar edges. The DumpTopK bin-by-bin diff showed
        # we hit 378 cells; EMSphInx hits 539 — and the missing 161 are
        # exactly this discarded outer ring.
        if getattr(self, "circmask", -1) == 0:
            X_pix = X_frac * W - 0.5
            Y_pix = Y_frac * H - 0.5
            cx_pix = (W - 1) * 0.5
            cy_pix = (H - 1) * 0.5
            radius_pix = min(H, W) / 2.0
            dist2 = (X_pix - cx_pix) ** 2 + (Y_pix - cy_pix) ** 2
            in_circle = dist2 <= radius_pix ** 2
            inside = in_rect & in_circle              # (dim, dim) bool
        else:
            inside = in_rect

        # ---- Flat indices of inside GL cells --------------------------------
        all_flat_gl = torch.arange(dim * dim, dtype=torch.int64)
        inside_flat = all_flat_gl[inside.reshape(-1)]    # (N_inside,)
        N_inside = inside_flat.shape[0]

        # ---- grid_sample coords for inside cells ---------------------------
        gx = (2.0 * X_frac - 1.0).reshape(-1)[inside_flat]  # (N_inside,) in [-1,1]
        gy = (2.0 * Y_frac - 1.0).reshape(-1)[inside_flat]
        grid_xy_gl = torch.stack([gx, gy], dim=-1).float()  # (N_inside, 2)

        # ---- Sphere normals for inside cells --------------------------------
        n0_in = n_n0.reshape(-1)[inside_flat].numpy()
        n1_in = n_n1.reshape(-1)[inside_flat].numpy()
        n2_in = n_n2.reshape(-1)[inside_flat].numpy()

        # ---- GL quadrature weights per cell --------------------------------
        # Ring ar has 8*ar cells; weight = gl_weight[ar-1] / (8*ar).
        # Pole cell (ar=0) gets zero weight (excluded from SHT sum).
        ar_flat = ar.reshape(-1)[inside_flat].numpy().astype(int)
        omega_np = np.zeros(N_inside, dtype=np.float32)
        two_pi = 2.0 * math.pi
        for idx_p in range(N_inside):
            ar_p = ar_flat[idx_p]
            if ar_p >= 1 and ar_p <= half:
                # GL weight integrates over dcos(θ) ∈ [0,1] for north hemisphere.
                # Azimuthal bin width = 2π / (8*ar).
                # Combined cell solid-angle weight = gl_weight[ar-1] * 2π / (8*ar).
                omega_np[idx_p] = float(gl_weights[ar_p - 1]) * two_pi / (8.0 * ar_p)

        # ---- Phase e^{-iφ_p} -----------------------------------------------
        phi_p = np.arctan2(n1_in, n0_in)
        phase_np = (np.cos(phi_p) - 1j * np.sin(phi_p)).astype(np.complex64)

        # ---- Normalized ALFs: alp[m, l, p] = K_l^m * P_l^m(cos θ_p) ------
        # Vectorized over the cell index (was a ~N_inside × L × L scalar triple
        # loop, ~43 s at bw=128). The per-element arithmetic / order / float32
        # cast-on-assign are kept bit-identical — see _build_normalized_alf and
        # tests/test_alf_vectorized.py.
        alp_np = _build_normalized_alf(L, n2_in)    # (L, L, N_inside) float32

        alp_t     = torch.from_numpy(alp_np)        # (L, L, N_inside) float32
        phase_t   = torch.from_numpy(phase_np)      # (N_inside,) complex64
        omega_t   = torch.from_numpy(omega_np)      # (N_inside,) float32

        # inside_flat is in terms of the GL grid (dim*dim), not detector pixels.
        # We store grid_xy_gl separately so _direct_sht_coefs can do grid_sample.
        # For API compatibility the first return value is inside_flat (GL-grid flat).
        return inside_flat, alp_t, phase_t, omega_t, grid_xy_gl

    def _build_sht_basis(self) -> torch.Tensor:
        """Precompute the direct-quadrature SHT basis matrix.

        Computes ``basis[m*L+l, p] = alp[m,l,p] * omega[p] * e^{-im*phi_p}``
        for all valid (m, l) pairs (l >= m) and all N_inside detector-inside GL
        cells p.  Entries for l < m are zero.

        Uses the precomputed ``_dp_alp``, ``_dp_phase``, and ``_dp_omega``
        tensors populated by ``_build_direct_sht_map``.

        Returns
        -------
        basis : (L*L, N_inside) complex64 torch.Tensor (on CPU)
        """
        L = self.bandwidth
        alp   = self._dp_alp    # (L, L, N_inside) float32
        phase = self._dp_phase  # (N_inside,) complex64  — e^{-iφ_p}
        omega = self._dp_omega  # (N_inside,) float32

        N_inside = phase.shape[0]

        # Weighted ALFs: w_alp[m, l, p] = alp[m, l, p] * omega[p]
        # Broadcast omega over (L, L) dims
        w_alp = alp * omega.unsqueeze(0).unsqueeze(0)  # (L, L, N_inside) float32

        # Phase powers: phase_pow[m, p] = (e^{-iφ_p})^m = e^{-im*φ_p}
        # Compute iteratively to avoid complex pow inaccuracies
        phase_pow = torch.zeros(L, N_inside, dtype=torch.complex64)
        phase_pow[0, :] = 1.0
        for m in range(1, L):
            phase_pow[m, :] = phase_pow[m - 1, :] * phase  # (N_inside,)

        # basis[m, l, p] = w_alp[m, l, p] * phase_pow[m, p]
        # Broadcast phase_pow over l-dim
        basis_3d = w_alp.to(torch.complex64) * phase_pow.unsqueeze(1)  # (L, L, N_inside)

        # Flatten (m, l) → single axis
        basis = basis_3d.reshape(L * L, N_inside)  # (L*L, N_inside) c64

        return basis  # stored on CPU; moved to device in _direct_sht_coefs

    # ------------------------------------------------------------------
    # EMSphInx ring-by-ring SHT tables
    # ------------------------------------------------------------------

    def _compute_ring_weights(self, dim: int, cos_nodes: np.ndarray) -> np.ndarray:
        """Compute per-cell EMSphInx GL ring weights via ``computeWeightsSkip(skp=0)``.

        Solves the Chebyshev linear system
            A[j,i] = cos(2j · arccos(cos_nodes[i+1])), b[j] = δ_{j,0} − 1/(4j²-1)
        to obtain integration weights ``wgt_raw``, then scales them by the cell
        solid-angle factor ``w0 / (8·ar)`` to give per-cell weights.

        Parameters
        ----------
        dim : int
            GL grid side length (odd).
        cos_nodes : (half,) float64
            Cosines of GL ring colatitudes for rings 1..half, sorted high→low.

        Returns
        -------
        wgt : (half+1,) float64
            wgt[0] = 0 (pole skipped), wgt[ar] = per-cell weight for ring ar.
        """
        half = dim // 2
        nMat = (dim + 1) // 2 - 1  # = half for odd dim

        # x[i] = cos(2 * arccos(cos_nodes[i])) for i = 0..nMat-1
        x = 2.0 * cos_nodes[:nMat] ** 2 - 1.0

        # Chebyshev matrix: A[j, i] = T_j(x[i])
        A = np.zeros((nMat, nMat), dtype=np.float64)
        A[0, :] = 1.0
        if nMat > 1:
            A[1, :] = x
        for j in range(2, nMat):
            A[j, :] = 2.0 * x * A[j - 1, :] - A[j - 2, :]

        # RHS: b[0] = 1, b[j] = -1/(4j²-1) for j>=1
        b = np.zeros(nMat, dtype=np.float64)
        b[0] = 1.0
        for j in range(1, nMat):
            b[j] = -1.0 / (4.0 * j * j - 1.0)

        wgt_raw = np.linalg.solve(A, b)  # (nMat,)

        # Solid-angle scaling factor: EMSphInx uses wn = pi2 * 4 / gridPoints = 8*pi / gridPoints
        # (square_sht.hpp line 1051: const Real wn = emsphinx::Constants<Real>::pi2 * 4 / gridPoints)
        grid_points = dim * dim * 2 - (dim - 1) * 4
        wn = 8.0 * math.pi / grid_points
        w0 = wn * ((dim - 2) * dim + 2)

        # Per-cell weights: ring ar gets wgt_raw[ar-1] * w0 / (8*ar)
        # This matches EMSphInx line 1062:
        #   wgt[i] *= w0 / (8 * i + offset);  // offset=0 for odd dim
        # The weight is per-CELL (not per-ring), so the unnormalized rfft output
        # (sum of N=8*ar values) combined with the per-cell weight gives the
        # correct quadrature: sum_k f[k] * w_cell = sum_k f[k] * (wgt_raw * w0 / N).
        # Ring 0 (pole) is skipped (weight = 0).
        wgt = np.zeros(half + 1, dtype=np.float64)
        for ar in range(1, nMat + 1):
            wgt[ar] = wgt_raw[ar - 1] * w0 / (8.0 * ar)

        return wgt  # (half+1,)

    def _build_gl_ring_tables(self):
        """Precompute per-ring data for ring-by-ring SHT (EMSphInx DiscreteSHT).

        For each GL ring ar = 1..half:
          * ring_cell_flat[ar-1]: (8·ar,) int64 — flat GL grid indices in phi order
            (phi = 0, 2π/8ar, 4π/8ar, …).  Azimuthal FFT of these values gives
            G_m coefficients for m = 0..4·ar.
          * ring_alp[ar-1]: (L, L) float32 — K_l^m P_l^m(cos θ_ar) at ring
            colatitude.  ``alp[m, l]`` is the ALF for order m, degree l.
        ring_wy: (half,) float32 — per-cell EMSphInx ring weights for ar=1..half.
        gl_dim: int — GL grid side length.

        Returns
        -------
        ring_cell_flat : list[Tensor(8*ar, int64)]
        ring_alp       : list[Tensor(L, L, float32)]
        ring_wy        : Tensor(half, float32)
        gl_dim         : int
        """
        from numpy.polynomial.legendre import leggauss

        L = self.bandwidth
        dim = L + 3 if (L % 2 == 0) else L + 2
        if dim % 2 == 0:
            dim += 1
        half = dim // 2

        # GL nodes (same as _build_direct_sht_map)
        n_gl = dim - 2
        cos_all, _ = leggauss(n_gl)
        cos_nodes = np.sort(cos_all)[::-1][:half].copy()  # (half,) high→low

        # --- Ring weights via computeWeightsSkip ----------------------------
        wgt = self._compute_ring_weights(dim, cos_nodes)  # (half+1,)
        ring_wy_np = wgt[1:].astype(np.float32)           # (half,) ar=1..half

        # --- Build full GL grid normals (same as _build_direct_sht_map) -----
        i_idx = torch.arange(dim, dtype=torch.float32)
        j_idx = torch.arange(dim, dtype=torch.float32)
        I_g, J_g = torch.meshgrid(i_idx, j_idx, indexing="ij")
        ri = J_g - half
        rj = I_g - half
        ai, aj = ri.abs(), rj.abs()
        ar_grid = torch.maximum(ai, aj)
        ar_safe = ar_grid.clamp_min(1.0)
        sX = ri / ar_safe
        sY = rj / ar_safe
        kPi_4 = math.pi / 4.0
        qq_a = kPi_4 * sX * sY
        case_a = (ai <= aj) & (ar_grid > 0)
        case_b = (ai > aj)
        hx_a = sY * torch.sin(qq_a)
        hy_a = sY * torch.cos(qq_a)
        qq_b = kPi_4 * sY * sX
        hx_b = sX * torch.cos(qq_b)
        hy_b = sX * torch.sin(qq_b)
        hx = torch.where(case_a, hx_a, torch.where(case_b, hx_b, torch.zeros_like(hx_a)))
        hy = torch.where(case_a, hy_a, torch.where(case_b, hy_b, torch.zeros_like(hy_a)))
        h_mag = torch.hypot(hx, hy).clamp_min(1e-15)

        cos_lat = torch.zeros(dim, dim, dtype=torch.float64)
        for ar_val in range(1, half + 1):
            mask = (ar_grid == ar_val)
            cos_lat[mask] = float(cos_nodes[ar_val - 1])
        cos_lat[half, half] = 1.0
        sin_lat = (1.0 - cos_lat ** 2).clamp_min(0.0).sqrt()

        n_n0 = (sin_lat * hx / h_mag).numpy()
        n_n1 = (sin_lat * hy / h_mag).numpy()
        n_n0[half, half] = 0.0
        n_n1[half, half] = 0.0
        n_n0_flat = n_n0.ravel()
        n_n1_flat = n_n1.ravel()
        ar_flat   = ar_grid.numpy().astype(int).ravel()

        # --- ALF normalisation constants ------------------------------------
        log_fac = np.zeros(2 * L + 2, dtype=np.float64)
        for i in range(1, len(log_fac)):
            log_fac[i] = log_fac[i - 1] + math.log(i)

        def K_lm(l: int, m: int) -> float:
            if m == 0:
                return math.sqrt((2 * l + 1) / (4.0 * math.pi))
            return math.exp(
                0.5 * math.log((2 * l + 1) / (4.0 * math.pi))
                + 0.5 * (log_fac[l - m] - log_fac[l + m])
            )

        # --- Build per-ring tables -------------------------------------------
        ring_cell_flat_list: list = []
        ring_alp_list: list = []

        for ar in range(1, half + 1):
            n_pts = 8 * ar
            cell_mask = (ar_flat == ar)
            ring_cells = np.where(cell_mask)[0]  # flat indices, arbitrary order
            assert len(ring_cells) == n_pts, (
                f"Ring {ar}: expected {n_pts} cells, got {len(ring_cells)}"
            )

            # Sort cells by azimuthal angle phi = atan2(n1, n0) in [0, 2π)
            phi = np.arctan2(n_n1_flat[ring_cells], n_n0_flat[ring_cells])
            phi = phi % (2.0 * math.pi)
            ring_cells = ring_cells[np.argsort(phi)]

            # Legendre values at this ring's colatitude
            cos_theta = float(cos_nodes[ar - 1])
            x     = cos_theta
            sphi  = math.sqrt(max(1.0 - x * x, 0.0))
            alp   = np.zeros((L, L), dtype=np.float32)
            for m in range(L):
                pmm = 1.0
                for k in range(1, m + 1):
                    pmm *= sphi * (2 * k - 1)
                alp[m, m] = K_lm(m, m) * pmm
                pm_prev = pmm
                pm_curr = x * (2 * m + 1) * pmm
                if m + 1 < L:
                    alp[m, m + 1] = K_lm(m + 1, m) * pm_curr
                for l in range(m + 2, L):
                    pm_next = (
                        (2 * l - 1) * x * pm_curr - (l + m - 1) * pm_prev
                    ) / (l - m)
                    alp[m, l] = K_lm(l, m) * pm_next
                    pm_prev = pm_curr
                    pm_curr = pm_next

            ring_cell_flat_list.append(torch.from_numpy(ring_cells).long())
            ring_alp_list.append(torch.from_numpy(alp))

        ring_wy_t = torch.from_numpy(ring_wy_np)  # (half,) float32

        return ring_cell_flat_list, ring_alp_list, ring_wy_t, dim

    def _build_south_hemi_gl_ring_map(self):
        """Precompute south-hemisphere GL ring detector projections.

        For each ring ar = 1..half, the south-hemisphere cells have the SAME
        azimuthal positions as the north-hemisphere cells (same n0, n1) but
        negated n2: n_south = (n0, n1, -cos_lat).  This method computes the
        detector-fractional-coordinate pairs (gx, gy) for each south-hemisphere
        cell, and an inside-detector boolean mask, both sorted by azimuthal angle
        to match the north-hemisphere ordering in _gl_ring_cell_flat.

        Returns
        -------
        south_gxy   : list[Tensor(8*ar, 2, float32)]  — grid_sample coords per ring
        south_inside : list[Tensor(8*ar, bool)]        — True = inside detector window
        """
        from numpy.polynomial.legendre import leggauss

        L = self.bandwidth
        H, W = self.geom.pat_h, self.geom.pat_w
        dim = self._gl_dim
        half = dim // 2

        # GL nodes — same as _build_direct_sht_map / _build_gl_ring_tables
        n_gl_nodes = dim - 2
        cos_all, _ = leggauss(n_gl_nodes)
        cos_nodes = np.sort(cos_all)[::-1][:half].copy()  # (half,) high→low

        # Geometry parameters — same as _build_direct_sht_map
        alpha_tilt = self._alpha_tilt
        sA = math.sin(alpha_tilt)
        cA = math.cos(alpha_tilt)

        if self.geom.cx is not None:
            cX = float(self.geom.cx.item())
            cY = float(self.geom.cy.item())
        else:
            cX = -float(self.geom.xpc.item())
            cY = float(self.geom.ypc.item())
        Lscint = float(self.geom.L.item())
        delta  = self.geom.pixel_size

        # North GL grid azimuthal geometry (hx, hy, n0, n1 per ring)
        # These are shared between north and south hemispheres.
        i_idx = torch.arange(dim, dtype=torch.float32)
        j_idx = torch.arange(dim, dtype=torch.float32)
        I_g, J_g = torch.meshgrid(i_idx, j_idx, indexing="ij")
        ri = J_g - half
        rj = I_g - half
        ai, aj = ri.abs(), rj.abs()
        ar_grid = torch.maximum(ai, aj)
        ar_safe = ar_grid.clamp_min(1.0)
        sX = ri / ar_safe
        sY = rj / ar_safe
        kPi_4 = math.pi / 4.0
        qq_a = kPi_4 * sX * sY
        case_a = (ai <= aj) & (ar_grid > 0)
        case_b = (ai > aj)
        hx_a = sY * torch.sin(qq_a)
        hy_a = sY * torch.cos(qq_a)
        qq_b = kPi_4 * sY * sX
        hx_b = sX * torch.cos(qq_b)
        hy_b = sX * torch.sin(qq_b)
        hx = torch.where(case_a, hx_a, torch.where(case_b, hx_b, torch.zeros_like(hx_a)))
        hy = torch.where(case_a, hy_a, torch.where(case_b, hy_b, torch.zeros_like(hy_a)))
        h_mag = torch.hypot(hx, hy).clamp_min(1e-15)

        # cos_lat for north hemisphere (per cell)
        cos_lat_north = torch.zeros(dim, dim, dtype=torch.float64)
        for ar_val in range(1, half + 1):
            mask = (ar_grid == ar_val)
            cos_lat_north[mask] = float(cos_nodes[ar_val - 1])
        cos_lat_north[half, half] = 1.0
        sin_lat = (1.0 - cos_lat_north ** 2).clamp_min(0.0).sqrt()

        # North n0, n1 (shared with south)
        n_n0 = (sin_lat * hx / h_mag)
        n_n1 = (sin_lat * hy / h_mag)
        n_n0[half, half] = 0.0
        n_n1[half, half] = 0.0
        n_n0_flat = n_n0.numpy().ravel()
        n_n1_flat = n_n1.numpy().ravel()
        ar_flat   = ar_grid.numpy().astype(int).ravel()

        south_gxy_list: list = []
        south_inside_list: list = []

        for ar in range(1, half + 1):
            n_pts = 8 * ar
            cos_lat_ar = float(cos_nodes[ar - 1])  # > 0

            # South hemisphere at this ring: same n0, n1 but n2 = -cos_lat_ar
            cell_mask = (ar_flat == ar)
            ring_cells = np.where(cell_mask)[0]  # flat indices in north grid
            assert len(ring_cells) == n_pts

            # Sort by azimuthal angle (same as north ring ordering)
            phi = np.arctan2(n_n1_flat[ring_cells], n_n0_flat[ring_cells])
            phi = phi % (2.0 * math.pi)
            sort_idx = np.argsort(phi)
            ring_cells = ring_cells[sort_idx]

            n0_ring = n_n0_flat[ring_cells].astype(np.float64)  # (n_pts,)
            n1_ring = n_n1_flat[ring_cells].astype(np.float64)  # (n_pts,)
            n2_south = -cos_lat_ar  # scalar

            # Forward-project south normal to detector
            denom = n0_ring * sA + n2_south * cA           # (n_pts,)
            valid = denom > 1e-9
            denom_s = np.where(valid, denom, 1.0)           # avoid division by zero
            d_proj  = Lscint / denom_s
            x_um = n1_ring * d_proj
            y_um = (sA * n2_south - cA * n0_ring) * d_proj

            X_frac = (cX + x_um / delta) / W + 0.5
            Y_frac = (cY + y_um / delta) / H + 0.5
            if self.flip_y:
                Y_frac = 1.0 - Y_frac

            in_rect = (
                valid
                & (X_frac >= 0.0) & (X_frac <= 1.0)
                & (Y_frac >= 0.0) & (Y_frac <= 1.0)
            )

            # FIX 2026-05-07 (iter-14+2): same circmask convention as north
            # — only enable inscribed-circle check when circRad == 0
            # (EMSphInx idx.hpp:230). The default circmask=-1 disables.
            if getattr(self, "circmask", -1) == 0:
                X_pix = X_frac * W - 0.5
                Y_pix = Y_frac * H - 0.5
                cx_pix = (W - 1) * 0.5
                cy_pix = (H - 1) * 0.5
                radius_pix = min(H, W) / 2.0
                dist2 = (X_pix - cx_pix) ** 2 + (Y_pix - cy_pix) ** 2
                in_circle = dist2 <= radius_pix ** 2
                inside = in_rect & in_circle
            else:
                inside = in_rect

            # Grid-sample coords: [-1, 1]
            gx = (2.0 * X_frac - 1.0).astype(np.float32)
            gy = (2.0 * Y_frac - 1.0).astype(np.float32)
            gxy = np.stack([gx, gy], axis=-1)  # (n_pts, 2)

            south_gxy_list.append(torch.from_numpy(gxy))          # (8*ar, 2) float32
            south_inside_list.append(torch.from_numpy(inside))    # (8*ar,) bool

        return south_gxy_list, south_inside_list

    def _build_flat_south_tensors(self):
        """Concatenate per-ring south-hemi data into flat (N_south_inside,...) tensors.

        Each south cell is a grid point (row, col) on the GL Lambert square whose
        z-flipped (south) normal projects onto the detector. Per ring ar, the
        omega weight is the same as the north hemisphere cells in that ring:
            omega_ar = gl_weights[ar - 1] * 2π / (8 * ar)

        Returns
        -------
        south_grid_xy : (N_south_inside, 2) float32 — grid_sample coords in [-1, 1]
        south_omega   : (N_south_inside,) float32 — GL solid-angle quadrature weight
        """
        from numpy.polynomial.legendre import leggauss

        dim = self._gl_dim
        half = dim // 2

        # GL weights — same as in _build_direct_sht_map.
        n_gl = dim - 2
        cos_all, w_all = leggauss(n_gl)
        sort_idx = np.argsort(cos_all)[::-1].copy()
        gl_weights = w_all[sort_idx][:half]            # (half,) float64
        two_pi = 2.0 * math.pi

        gxy_chunks = []
        omega_chunks = []
        for ar_idx, (gxy, inside) in enumerate(
            zip(self._gl_south_ring_gxy, self._gl_south_ring_inside)
        ):
            ar = ar_idx + 1
            in_mask = inside  # (8*ar,) bool
            n_in = int(in_mask.sum().item())
            if n_in == 0:
                continue
            gxy_chunks.append(gxy[in_mask])             # (n_in, 2) float32
            w_per_cell = float(gl_weights[ar_idx]) * two_pi / (8.0 * ar)
            omega_chunks.append(
                torch.full((n_in,), w_per_cell, dtype=torch.float32)
            )

        if not gxy_chunks:
            # No south projections hit the detector — return empty tensors.
            return (
                torch.empty(0, 2, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )

        return (
            torch.cat(gxy_chunks, dim=0),
            torch.cat(omega_chunks, dim=0),
        )

    def _backproject_and_fsht(self, patterns: torch.Tensor) -> torch.Tensor:
        """DH-grid back-projection → RSHT path.

        Places the normalized detector pattern onto the DH sphere grid via
        bilinear grid_sample, applies Friedel symmetry (real-signal antipodal
        fill), solid-angle-weighted normalization (mean-subtraction + unit-std),
        then runs RSHT.fsht to obtain coefs in the same 4π-orthonormal
        convention as the master SHT.

        We do NOT filter to complete rings: the partial-ring cells still
        contribute partially correct data. The RSHT's DLT quadrature handles
        the non-uniform ring coverage; the resulting coefs are a windowed
        SHT of the detector pattern over the partial sphere, which is the
        correct input to rs2cc_ against the full-sphere master coefs.
        """
        B, H, W = patterns.shape
        L = self.bandwidth
        dim = 2 * L
        device = self.device
        n_cells = dim * dim

        grid_xy    = self._bp_grid_xy.to(device)
        north_mask = self._bp_north_mask.to(device)
        north_flat = self._bp_north_flat.to(device)

        pat_4d = patterns.unsqueeze(1).float()
        grid_4d = grid_xy.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)
        sampled = F.grid_sample(
            pat_4d, grid_4d,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(1).squeeze(-1)

        sampled = sampled * north_mask.unsqueeze(0).float()

        sphere_flat = torch.zeros(B, n_cells, dtype=torch.float32, device=device)
        nf_exp = north_flat.unsqueeze(0).expand(B, -1)
        sphere_flat.scatter_(1, nf_exp, sampled)

        south_flat     = self._bp_south_flat.to(device)
        antipodal_flat = self._bp_antipodal_flat.to(device)

        north_inside_flag = torch.zeros(n_cells, dtype=torch.bool, device=device)
        north_inside_idx  = north_flat[north_mask]
        north_inside_flag.scatter_(0, north_inside_idx, True)
        south_active      = north_inside_flag[antipodal_flat]

        active_south_flat = south_flat[south_active]
        active_antipodal  = antipodal_flat[south_active]

        south_values = sphere_flat[:, active_antipodal]
        sf_exp = active_south_flat.unsqueeze(0).expand(B, -1)
        sphere_flat.scatter_(1, sf_exp, south_values)

        # Solid-angle-weighted normalization (EMSphInx unproject convention):
        # weight each cell by its DH quadrature weight, restricted to the
        # detector window.
        dh_weights = self._bp_dh_weights.to(device)
        window_mask = torch.zeros(n_cells, dtype=torch.float32, device=device)
        window_mask.scatter_(0, north_inside_idx, 1.0)
        window_mask.scatter_(0, active_south_flat, 1.0)

        j_idx    = torch.arange(n_cells, dtype=torch.int64, device=device) % dim
        dh_w_cell = dh_weights[j_idx]
        w      = dh_w_cell * window_mask
        w_sum  = w.sum().clamp_min(1e-12)

        mu   = (sphere_flat * w.unsqueeze(0)).sum(dim=1) / w_sum
        res  = sphere_flat - mu.unsqueeze(1)
        var  = (res ** 2 * w.unsqueeze(0)).sum(dim=1) / w_sum
        std  = var.sqrt().clamp_min(1e-6)

        sphere_norm = res / std.unsqueeze(1)

        # Apply the window mask so that outside-window cells are zero,
        # preserving the correct partial-sphere SHT semantics.
        sphere_sht = sphere_norm * window_mask.unsqueeze(0)

        sphere_dh = sphere_sht.reshape(B, dim, dim)
        coefs = self._rsht.fsht(sphere_dh)

        return coefs

    def _build_window_cc(self) -> torch.Tensor:
        """Precompute the SO(3) auto-correlation of the detector window function."""
        L = self.bandwidth
        device = self.device
        dim = 2 * L
        n_cells = dim * dim

        north_flat    = self._bp_north_flat.to(device)
        north_mask    = self._bp_north_mask.to(device)

        w_flat = torch.zeros(1, n_cells, dtype=torch.float32, device=device)
        inside_north = north_flat[north_mask]
        w_flat.scatter_(1, inside_north.unsqueeze(0), 1.0)

        w_dh = w_flat.reshape(1, dim, dim)
        w_coefs = self._rsht.fsht(w_dh)

        cc_ww = rs2cc_(L, w_coefs, w_coefs, self._wigner_table)
        return cc_ww

    def _build_backprojection_map(self):
        """Precompute forward-projection lookup tables for DH → detector mapping.

        Returns 7 tensors (same as before) for the legacy DH path.
        """
        L = self.bandwidth
        H, W = self.geom.pat_h, self.geom.pat_w
        dim = 2 * L
        device = self.device

        sigma_deg = float(getattr(self, "sample_tilt_deg", 70.0))
        theta_c_deg = self.geom.tilt_deg
        alpha_tilt = math.radians(90.0 - sigma_deg + theta_c_deg)
        sA = math.sin(alpha_tilt)
        cA = math.cos(alpha_tilt)

        if self.geom.cx is not None:
            cX = float(self.geom.cx.item())
            cY = float(self.geom.cy.item())
        else:
            cX = -float(self.geom.xpc.item())
            cY = float(self.geom.ypc.item())
        Lscint = float(self.geom.L.item())
        delta = self.geom.pixel_size

        i_idx = torch.arange(dim, dtype=torch.float32)
        j_idx = torch.arange(dim, dtype=torch.float32)
        theta_dh = 2.0 * math.pi * i_idx / dim
        phi_dh   = math.pi * (2.0 * j_idx + 1.0) / (4.0 * L)

        theta_g, phi_g = torch.meshgrid(theta_dh, phi_dh, indexing="ij")

        sp = torch.sin(phi_g)
        n0 = sp * torch.cos(theta_g)
        n1 = sp * torch.sin(theta_g)
        n2 = torch.cos(phi_g)

        all_flat = torch.arange(dim * dim, dtype=torch.int64)
        j_flat   = all_flat % dim

        north_mask_all = (j_flat < L)
        south_mask_all = ~north_mask_all

        north_flat = all_flat[north_mask_all]
        south_flat = all_flat[south_mask_all]

        n0_n = n0.reshape(-1)[north_flat]
        n1_n = n1.reshape(-1)[north_flat]
        n2_n = n2.reshape(-1)[north_flat]

        # EMSphInx interpolatePixel formula applied to DH normals directly.
        # The DH normals are in the GL frame (pole at z-axis). The projection
        # formula implicitly encodes the sample-tilt geometry via sA and cA.
        denom = n0_n * sA + n2_n * cA
        valid = denom > 1e-9

        denom_safe = denom.clamp_min(1e-9)
        d    = Lscint / denom_safe
        x_um = n1_n * d
        y_um = (sA * n2_n - cA * n0_n) * d

        X_frac = (cX + x_um / delta) / W + 0.5
        Y_frac = (cY + y_um / delta) / H + 0.5
        if self.flip_y:
            Y_frac = 1.0 - Y_frac

        inside = (
            valid
            & (X_frac >= 0.0) & (X_frac <= 1.0)
            & (Y_frac >= 0.0) & (Y_frac <= 1.0)
        )

        grid_x = 2.0 * X_frac - 1.0
        grid_y = 2.0 * Y_frac - 1.0
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        i_south = south_flat // dim
        j_south = south_flat % dim
        i_anti  = (i_south + L) % dim   # true antipodal: theta -> theta + pi
        j_anti  = dim - 1 - j_south     # true antipodal: phi -> pi - phi
        antipodal_flat = i_anti * dim + j_anti

        dh_weights = dltWeightsDH(L, device=torch.device("cpu"), dtype=torch.float32)

        j_north = north_flat % dim
        inside_count = torch.zeros(L, dtype=torch.int64)
        inside_count.scatter_add_(0, j_north, inside.long())
        complete_ring = (inside_count == dim)
        complete_ring_mask = complete_ring[j_north]

        return (
            grid_xy,
            inside,
            north_flat,
            south_flat,
            antipodal_flat,
            dh_weights,
            complete_ring_mask,
        )

    # Keep the old direct-SHT methods for backward compatibility
    def _direct_sht(self, patterns: torch.Tensor) -> torch.Tensor:
        """Alias of ``_backproject_and_fsht`` for backward compatibility."""
        return self._backproject_and_fsht(patterns)

    def _sht_coefs_unnorm(self, f_inside: np.ndarray) -> torch.Tensor:
        """Ring-FFT SHT of values at inside GL cells, WITHOUT normalization.

        Parameters
        ----------
        f_inside : (N_inside,) float64 ndarray
            Function values at inside GL cells (same ordering as _dp_inside_flat).

        Returns
        -------
        coefs : (1, L, L) complex64 torch.Tensor
        """
        L = self.bandwidth
        device = self.device
        dim = self._gl_dim
        n_gl = dim * dim

        inside_flat = self._dp_inside_flat  # (N_inside,) int64
        ring_wy = self._gl_ring_wy           # (half,) float32
        ring_flat = self._gl_ring_cell_flat  # list of Tensor(8*ar,)
        ring_alp = self._gl_ring_alp         # list of Tensor(L, L)

        # Scatter into GL grid
        f_t = torch.from_numpy(f_inside.astype(np.float32)).to(device)
        gl_grid = torch.zeros(n_gl, dtype=torch.float32, device=device)
        gl_grid.scatter_(0, inside_flat.to(device), f_t)

        # Ring FFT accumulation
        coefs = torch.zeros(L, L, dtype=torch.complex64, device=device)
        for ar_idx in range(len(ring_flat)):
            ar = ar_idx + 1
            n_pts = 8 * ar
            flat_dev = ring_flat[ar_idx].to(device)
            ring_vals = gl_grid[flat_dev].unsqueeze(0)          # (1, 8*ar)
            G = torch.fft.rfft(ring_vals, n=n_pts, dim=1)       # (1, 4*ar+1) c64
            mLim = min(L, 4 * ar + 1)
            wy = float(ring_wy[ar_idx].item())
            alp_dev = ring_alp[ar_idx].to(device)               # (L, L) float32
            for m in range(mLim):
                sign = 1.0 if m % 2 == 0 else -1.0
                Gm = G[0, m] * (wy * sign)                      # scalar c64
                coefs[m, m:] += Gm * alp_dev[m, m:]

        return coefs.unsqueeze(0)                                # (1, L, L)

    def _synthesize_master_at_inside_cells(self) -> np.ndarray:
        """Evaluate the master pattern at inside-cell GL normals via inverse SHT.

        Returns
        -------
        master_vals : (N_inside,) float64 ndarray
        """
        L = self.bandwidth
        mc_np = self.master.coefs_ml[:L, :L].cpu().numpy().astype(np.complex128)
        alp = self._dp_alp.numpy()       # (L, L, N_inside) float32
        phase = self._dp_phase.numpy()   # (N_inside,) complex64
        N_inside = phase.shape[0]

        phi_p = np.angle(np.conj(phase)).astype(np.float64)
        master_vals = np.zeros(N_inside, dtype=np.float64)
        for m in range(L):
            cos_m = np.cos(m * phi_p)
            sin_m = np.sin(m * phi_p)
            for l in range(m, L):
                c = mc_np[m, l]
                if abs(c) < 1e-15:
                    continue
                alp_ml = alp[m, l, :].astype(np.float64)
                if m == 0:
                    master_vals += c.real * alp_ml
                else:
                    master_vals += 2.0 * (c.real * alp_ml * cos_m
                                          - c.imag * alp_ml * sin_m)
        return master_vals

    def _sht_coefs_fullsphere(self, f_allcells: np.ndarray) -> torch.Tensor:
        """Full-sphere ring-FFT SHT matching EMSphInx DiscreteSHT::analyze with south=0.

        EMSphInx analyze() uses both north and south hemispheres.  When south=0
        (as for all window/master² functions that are zero on the south hemisphere),
        the formula reduces to multiplying each ring's FFT by ``0.5 * wy * alp`` for
        ALL degrees (both symmetric and antisymmetric).  This 0.5 factor is the key
        difference from our partial-sphere ``_sht_coefs_unnorm`` which uses the full
        north-only contribution (no 0.5 factor).

        Parameters
        ----------
        f_allcells : (dim*dim,) float64 ndarray
            Function values at ALL north-hemisphere GL cells in flat order.
            South hemisphere is implicitly zero.

        Returns
        -------
        coefs : (1, L, L) complex64 torch.Tensor
        """
        L = self.bandwidth
        device = self.device
        dim = self._gl_dim
        n_gl = dim * dim

        ring_wy = self._gl_ring_wy           # (half,) float32
        ring_flat = self._gl_ring_cell_flat  # list of Tensor(8*ar,)
        ring_alp = self._gl_ring_alp         # list of Tensor(L, L)

        f_t = torch.from_numpy(f_allcells.astype(np.float32)).to(device)
        gl_grid = torch.zeros(n_gl, dtype=torch.float32, device=device)
        # Put values at all cells (f_allcells is indexed 0..n_gl-1)
        gl_grid[:] = f_t

        coefs = torch.zeros(L, L, dtype=torch.complex64, device=device)
        for ar_idx in range(len(ring_flat)):
            ar = ar_idx + 1
            n_pts = 8 * ar
            flat_dev = ring_flat[ar_idx].to(device)
            ring_vals = gl_grid[flat_dev].unsqueeze(0)          # (1, 8*ar)
            G = torch.fft.rfft(ring_vals, n=n_pts, dim=1)       # (1, 4*ar+1) c64
            mLim = min(L, 4 * ar + 1)
            # EMSphInx uses 0.5 factor when south=0 (symmetric+antisymmetric both give 0.5*north)
            wy = float(ring_wy[ar_idx].item()) * 0.5
            alp_dev = ring_alp[ar_idx].to(device)               # (L, L) float32
            for m in range(mLim):
                sign = 1.0 if m % 2 == 0 else -1.0
                Gm = G[0, m] * (wy * sign)                      # scalar c64
                coefs[m, m:] += Gm * alp_dev[m, m:]

        return coefs.unsqueeze(0)                                # (1, L, L)

    def _synthesize_master_on_fullgrid(self) -> np.ndarray:
        """Evaluate the master pattern at ALL north-hemisphere GL cells via inverse SHT.

        Unlike ``_synthesize_master_at_inside_cells``, this synthesizes at every
        cell in the ``dim×dim`` GL grid, not just the detector-inside cells.

        Returns
        -------
        master_allcells : (dim*dim,) float64 ndarray, row-major order
        """
        from numpy.polynomial.legendre import leggauss

        L = self.bandwidth
        mc_np = self.master.coefs_ml[:L, :L].cpu().numpy().astype(np.complex128)
        dim = self._gl_dim
        half = dim // 2

        # Rebuild GL grid geometry (same as _build_direct_sht_map / _build_gl_ring_tables)
        n_gl_nodes = dim - 2
        cos_all, _ = leggauss(n_gl_nodes)
        cos_nodes = np.sort(cos_all)[::-1][:half].copy()  # (half,) high→low

        # Log-factorial table for K_lm
        log_fac = np.zeros(2 * L + 2, dtype=np.float64)
        for i in range(1, len(log_fac)):
            log_fac[i] = log_fac[i - 1] + math.log(i)

        def K_lm(l: int, m: int) -> float:
            if m == 0:
                return math.sqrt((2 * l + 1) / (4.0 * math.pi))
            return math.exp(
                0.5 * math.log((2 * l + 1) / (4.0 * math.pi))
                + 0.5 * (log_fac[l - m] - log_fac[l + m])
            )

        i_idx = np.arange(dim, dtype=np.float64)
        j_idx = np.arange(dim, dtype=np.float64)
        I_g, J_g = np.meshgrid(i_idx, j_idx, indexing="ij")
        ri = J_g - half
        rj = I_g - half
        ai = np.abs(ri)
        aj = np.abs(rj)
        ar_grid = np.maximum(ai, aj).astype(int)  # (dim, dim)

        kPi_4 = math.pi / 4.0
        ar_safe = np.maximum(ar_grid, 1).astype(np.float64)
        sX = ri / ar_safe
        sY = rj / ar_safe

        # Case A: ai <= aj
        qq_a = kPi_4 * sX * sY
        hx_a = sY * np.sin(qq_a)
        hy_a = sY * np.cos(qq_a)
        # Case B: ai > aj
        qq_b = kPi_4 * sY * sX
        hx_b = sX * np.cos(qq_b)
        hy_b = sX * np.sin(qq_b)

        case_a = (ai <= aj) & (ar_grid > 0)
        case_b = (ai > aj)
        hx = np.where(case_a, hx_a, np.where(case_b, hx_b, np.zeros_like(hx_a)))
        hy = np.where(case_a, hy_a, np.where(case_b, hy_b, np.zeros_like(hy_a)))
        h_mag = np.hypot(hx, hy).clip(min=1e-15)

        cos_lat = np.zeros((dim, dim), dtype=np.float64)
        for ar_val in range(1, half + 1):
            cos_lat[ar_grid == ar_val] = float(cos_nodes[ar_val - 1])
        cos_lat[half, half] = 1.0  # pole
        sin_lat = np.sqrt(np.clip(1.0 - cos_lat ** 2, 0.0, None))

        n_n0 = sin_lat * hx / h_mag
        n_n1 = sin_lat * hy / h_mag
        n_n0[half, half] = 0.0
        n_n1[half, half] = 0.0
        cos_theta_flat = cos_lat.ravel()
        n0_flat = n_n0.ravel()
        n1_flat = n_n1.ravel()

        phi_flat = np.arctan2(n1_flat, n0_flat)  # (dim*dim,)
        master_allcells = np.zeros(dim * dim, dtype=np.float64)

        for m in range(L):
            cos_m = np.cos(m * phi_flat)
            sin_m = np.sin(m * phi_flat)
            # Compute alp[m, l] at all colatitudes
            x_arr = cos_theta_flat
            sin_arr = np.sqrt(np.clip(1.0 - x_arr ** 2, 0.0, None))
            pmm_arr = np.ones(dim * dim, dtype=np.float64)
            for k in range(1, m + 1):
                pmm_arr *= sin_arr * (2 * k - 1)
            alp_mm = K_lm(m, m) * pmm_arr
            pm_prev_arr = pmm_arr.copy()
            pm_curr_arr = x_arr * (2 * m + 1) * pmm_arr

            c = mc_np[m, m]
            if abs(c) >= 1e-15:
                if m == 0:
                    master_allcells += c.real * alp_mm
                else:
                    master_allcells += 2.0 * (c.real * alp_mm * cos_m - c.imag * alp_mm * sin_m)

            if m + 1 < L:
                alp_next = K_lm(m + 1, m) * pm_curr_arr
                c = mc_np[m, m + 1]
                if abs(c) >= 1e-15:
                    if m == 0:
                        master_allcells += c.real * alp_next
                    else:
                        master_allcells += 2.0 * (c.real * alp_next * cos_m - c.imag * alp_next * sin_m)

            for l in range(m + 2, L):
                pm_next_arr = ((2 * l - 1) * x_arr * pm_curr_arr - (l + m - 1) * pm_prev_arr) / (l - m)
                alp_lm = K_lm(l, m) * pm_next_arr
                c = mc_np[m, l]
                if abs(c) >= 1e-15:
                    if m == 0:
                        master_allcells += c.real * alp_lm
                    else:
                        master_allcells += 2.0 * (c.real * alp_lm * cos_m - c.imag * alp_lm * sin_m)
                pm_prev_arr = pm_curr_arr
                pm_curr_arr = pm_next_arr

        return master_allcells  # (dim*dim,) float64

    def _sht_coefs_invsym(self, f_allcells: np.ndarray) -> torch.Tensor:
        """Ring-FFT SHT for an inversion-symmetric function (f_south = f_north).

        For a function satisfying f(-n) = f(n) (inversion symmetry), both
        hemispheres carry identical values.  The EMSphInx full-sphere analysis
        formula then gives:

            Even l+m:  coef[m,l] = G_N[m] * wy * sign     (north doubled by south)
            Odd  l+m:  coef[m,l] = 0                       (north and south cancel)

        This is the correct convention for computing SHT coefficients of M² when
        the master pattern is inversion-symmetric (i.e., M(-n) = M(n), cmp_flg=7).
        It differs from ``_sht_coefs_fullsphere`` in two ways:
          - Factor is 1.0 instead of 0.5 for even l+m (south contributes equally)
          - Odd l+m coefficients are explicitly zero (they would cancel anyway)

        Parameters
        ----------
        f_allcells : (dim*dim,) float64 ndarray
            North-hemisphere function values at all GL cells in flat order.

        Returns
        -------
        coefs : (1, L, L) complex64 torch.Tensor
            Even l+m are populated; odd l+m are zero.
        """
        L = self.bandwidth
        device = self.device
        dim = self._gl_dim
        n_gl = dim * dim

        ring_wy   = self._gl_ring_wy            # (half,) float32
        ring_flat = self._gl_ring_cell_flat     # list of Tensor(8*ar,)
        ring_alp  = self._gl_ring_alp           # list of Tensor(L, L)

        f_t = torch.from_numpy(f_allcells.astype(np.float32)).to(device)
        gl_grid = torch.zeros(n_gl, dtype=torch.float32, device=device)
        gl_grid[:] = f_t

        coefs = torch.zeros(L, L, dtype=torch.complex64, device=device)
        for ar_idx in range(len(ring_flat)):
            ar = ar_idx + 1
            n_pts = 8 * ar
            flat_dev = ring_flat[ar_idx].to(device)
            ring_vals = gl_grid[flat_dev].unsqueeze(0)          # (1, 8*ar)
            G = torch.fft.rfft(ring_vals, n=n_pts, dim=1)       # (1, 4*ar+1) c64
            mLim = min(L, 4 * ar + 1)
            # For inversion-symmetric f: south=north → double the north contribution.
            # Factor = wy * 1.0 (not 0.5) for even l+m; odd l+m get zero.
            wy = float(ring_wy[ar_idx].item())
            alp_dev = ring_alp[ar_idx].to(device)               # (L, L) float32
            for m in range(mLim):
                sign = 1.0 if m % 2 == 0 else -1.0
                Gm = G[0, m] * (wy * sign)                      # (no 0.5 factor)
                alp_m = alp_dev[m, m:]                          # (L-m,)
                # Even l+m: l-m = 0, 2, 4, ...  (l+m parity: (l-m) parity = (l+m) parity)
                coefs[m, m::2] += Gm * alp_m[0::2]
                # Odd l+m: zero (inversion symmetry cancels these)

        return coefs.unsqueeze(0)                                # (1, L, L)

    def _precompute_norm_volume_gl_invsym(self) -> torch.Tensor:
        """Precompute norm_vol using GL convention with inversion-symmetric master².

        Computes ``norm_vol[R] = CC(c_win, c_master_sq)[R]`` where both SHT
        coefficients are in the GL ring-FFT convention matching ``_direct_sht_coefs``
        (the convention used for pattern coefficients and for master file coefs).

        The window function has south=0, so ``c_win`` uses ``_sht_coefs_fullsphere``
        (0.5 factor, both even and odd l+m populated).

        The master² is inversion-symmetric (M²(-n) = M²(n) since M(-n) = M(n)
        for cmp_flg=7 masters), so ``c_m2`` uses ``_sht_coefs_invsym``:
        even l+m gets full north+south contribution (factor 1.0), odd l+m = 0.

        Returns
        -------
        norm_vol : (1, 2L-1, 2L-1, 2L-1) float32 tensor (on CPU)
        """
        L = self.bandwidth
        dim = self._gl_dim
        n_gl = dim * dim
        inside_flat = self._dp_inside_flat.numpy()   # (N_inside,) int64

        # 1. Window SHT: 1 at inside cells, 0 elsewhere; south=0 → full-sphere path
        win_allcells = np.zeros(n_gl, dtype=np.float64)
        win_allcells[inside_flat] = 1.0
        c_win = self._sht_coefs_fullsphere(win_allcells)         # (1, L, L) c64

        # 2. Master² SHT: synthesize M on full north grid, square, invsym path
        master_allcells = self._synthesize_master_on_fullgrid()  # (dim*dim,) f64
        master_sq_allcells = master_allcells ** 2                # (dim*dim,) f64
        c_m2 = self._sht_coefs_invsym(master_sq_allcells)       # (1, L, L) c64

        # 3. norm_vol(R) = CC(c_win, c_m2)(R)
        # Zero m=0 to match the m=0-zeroed convention used in _index_batch for
        # pattern_coefs and _master_coefs_m0z.  Without this, the m=0 constant
        # term dominates norm_vol, making sqrt(norm_vol) nearly orientation-
        # independent and introducing a Phi-dependent bias that moves CC peaks
        # away from the oracle orientation.
        c_win[:, 0, :] = 0.0   # zero m=0 row — match pattern/master cc convention
        c_m2[:, 0, :]  = 0.0   # zero m=0 row — match pattern/master cc convention

        device = self.device
        c_win = c_win.to(device)
        c_m2  = c_m2.to(device)
        norm_vol = rs2cc_(
            L,
            c_win,                                               # (1, L, L) window
            c_m2,                                                # (1, L, L) master²
            self._wigner_table,
        )                                                        # (1, 2L-1, 2L-1, 2L-1)

        norm_vol_real = norm_vol.real.clamp_min(0.0)
        return norm_vol_real.cpu()

    def _precompute_norm_volume(self) -> torch.Tensor:
        """Precompute the SO(3) normalization volume for EMSphInx normed=TRUE CC.

        Computes ``norm_vol[R] = CC(c_win, c_master_sq)[R]``, the energy of the
        master pattern visible through the detector window at orientation R:
            norm_vol(R) = integral_sphere window(n) * master²(R^{-1}n) dn

        Dividing the unnormalized CC by ``sqrt(norm_vol)`` gives the normalized
        spherical cross-correlation (cosine similarity), which is robust to partial-
        sphere data (matches EMSphInx ``normed = .TRUE.`` namelist parameter).

        Key: both ``c_master_sq`` and ``c_win`` must be full-sphere SHT coefficients
        in the same basis.  We use ``_sht_coefs_fullsphere`` which applies the
        correct 0.5 factor for south-hemisphere=zero and synthesizes on ALL north-
        hemisphere GL cells (not just inside-detector cells).  The 0.5 factor
        produces coefficients that differ by a global constant from the master
        convention, but this constant cancels in the normalized argmax.

        Returns
        -------
        norm_vol : (1, 2L-1, 2L-1, 2L-1) float32 tensor (on CPU)
        """
        L = self.bandwidth
        dim = self._gl_dim
        n_gl = dim * dim
        inside_flat = self._dp_inside_flat.numpy()  # (N_inside,) int64

        # 1. Window function SHT: 1.0 at inside cells, 0 at all other cells.
        #    Use full-sphere convention (0.5 factor for south=0).
        win_allcells = np.zeros(n_gl, dtype=np.float64)
        win_allcells[inside_flat] = 1.0
        c_win = self._sht_coefs_fullsphere(win_allcells)            # (1, L, L) c64

        # 2. Master^2 SHT: synthesize master at ALL north-hemisphere GL cells, square.
        #    Use full-sphere convention.
        master_allcells = self._synthesize_master_on_fullgrid()     # (dim*dim,) f64
        master_sq_allcells = master_allcells ** 2                   # (dim*dim,) f64
        c_master_sq = self._sht_coefs_fullsphere(master_sq_allcells) # (1, L, L) c64

        # 3. norm_vol(R) = CC(window, master²)(R)
        #      = integral window(n) * master²(R^{-1}*n) dn
        # Argument order: f = window (first arg), g = master² (second arg).
        # rs2cc_(L, f, g)(R) = integral f(n) * conj(g(R^{-1}n)) dn.
        # This peaks at R=identity where the window maximally overlaps master².
        # Move to same device as master coefs for the computation.
        device = self.device
        c_win = c_win.to(device)
        c_master_sq = c_master_sq.to(device)
        norm_vol = rs2cc_(
            L,
            c_win,                                                   # (1, L, L) window
            c_master_sq,                                            # (1, L, L) master²
            self._wigner_table,
        )                                                           # (1, 2L-1, 2L-1, 2L-1)

        return norm_vol.cpu()                                       # store on CPU

    def _precompute_norm_volume_dh(self) -> torch.Tensor:
        """Precompute the SO(3) normalization volume using the DH RSHT path.

        Computes ``norm_vol[R] = CC(c_win, c_master_sq)[R]``, the energy of
        the squared master pattern visible through the detector window at R:

            norm_vol(R) = integral_sphere window(n) * master²(R^{-1}n) dn

        Both ``c_win`` and ``c_master_sq`` are computed via ``RSHT.fsht`` on the
        DH grid — the same convention used for the master file coefs — so that
        ``rs2cc_`` gives the correct orientation for the normalization peak.

        Returns
        -------
        norm_vol : (1, 2L-1, 2L-1, 2L-1) float32 tensor (on CPU)
        """
        L = self.bandwidth
        dim = 2 * L          # DH grid side length
        n_cells = dim * dim
        device = self.device

        # ---- 1. Window function on DH grid -----------------------------------
        # Reuse the precomputed backprojection mask (north hemisphere only).
        # South hemisphere is 0 → window is a partial-sphere (north-cap) function.
        north_flat = self._bp_north_flat.to(device)   # (n_north,) int64
        north_mask = self._bp_north_mask.to(device)   # (n_north,) bool
        inside_north = north_flat[north_mask]          # flat DH indices inside detector

        win_flat = torch.zeros(1, n_cells, dtype=torch.float32, device=device)
        win_flat.scatter_(1, inside_north.unsqueeze(0), 1.0)
        win_dh = win_flat.reshape(1, dim, dim)         # (1, 2L, 2L)

        c_win = self._rsht.fsht(win_dh)                # (1, L, L) complex64

        # ---- 2. Master² on DH grid -------------------------------------------
        # Synthesize master on full DH grid by applying the inverse SHT to the
        # stored master coefs (already in RSHT convention).
        c_master_1 = self._master_coefs.to(device)     # (1, L, L) complex64
        master_dh = self._rsht.isht(c_master_1)        # (1, 2L, 2L) float32

        master_sq_dh = master_dh ** 2                  # (1, 2L, 2L) float32
        c_master_sq = self._rsht.fsht(master_sq_dh)    # (1, L, L) complex64

        # ---- 3. norm_vol(R) = CC(window, master²)(R) -------------------------
        # Do NOT zero m=0 here.  The window function and master² are both
        # non-negative, so their CC should be non-negative everywhere and peak
        # near the identity.  Zeroing m=0 converts them to signed functions
        # whose CC can be negative — clamp_min(0) then turns those to zeros,
        # which causes catastrophic amplification when dividing by sqrt(nv).
        norm_vol = rs2cc_(
            L,
            c_win,                                     # (1, L, L) window (full, no m=0 zero)
            c_master_sq,                               # (1, L, L) master² (full)
            self._wigner_table,
        )                                              # (1, 2L-1, 2L-1, 2L-1) complex

        # The norm_vol is a real-valued function (both inputs are real → imaginary
        # parts of the CC volume should be negligible). Take real part and clamp
        # to ensure non-negative denominator.
        norm_vol_real = norm_vol.real.clamp_min(0.0)
        return norm_vol_real.cpu()                     # store on CPU

    def _precompute_nc_denom(self) -> Tuple[torch.Tensor, float]:
        """Precompute the EMSphInx NormalizedCorrelator denominator volume.

        Implements the precomputation from NormalizedCorrelator::Constants::Constants
        (sht_xcorr.hpp) adapted to our rs2cc_ argument order convention.

        The NC formula is:
            NC(R) = CC(pattern, master)(R) * rDen(R)
        where:
            rDen(R) = 1/sqrt(CC(window, master²)(R) - CC(window, master)(R)² / s2m)
            s2m = c_window[0,0].real * sqrt(4π)  = total solid angle of window

        The indexing convention: our numerator rs2cc_(L, pattern, master) peaks at
        R_crystal^{-1} (the inverse of the physical crystal rotation). The denominator
        must be evaluated at the SAME grid index.  Using CC(window, master²)(R) and
        CC(window, master)(R) — with window as the first argument — achieves this
        because CC(window, f)(R_grid) = ∫window(n) * f(R_phys^{-1} * n) dn, which
        evaluates the master energy/cross-term at R_phys = R_crystal^{-1} when
        R_grid is the peak index.

        Returns
        -------
        rDen_vol : (1, 2L-1, 2L-1, 2L-1) float32 tensor (on CPU)
        s2m      : float — solid angle of detector window in steradians [0, 4π]
        """
        L = self.bandwidth
        dim = 2 * L          # DH grid side length
        n_cells = dim * dim
        device = self.device

        # ---- 1. Window function SHT (DH grid, north-only) -------------------
        # Binary mask: 1 at DH cells projecting inside the detector, 0 elsewhere.
        # South hemisphere is 0 (detector only faces north).
        north_flat = self._bp_north_flat.to(device)    # (n_north,) int64
        north_mask = self._bp_north_mask.to(device)    # (n_north,) bool
        inside_north = north_flat[north_mask]           # DH flat indices inside detector

        win_flat = torch.zeros(1, n_cells, dtype=torch.float32, device=device)
        win_flat.scatter_(1, inside_north.unsqueeze(0), 1.0)
        win_dh = win_flat.reshape(1, dim, dim)          # (1, 2L, 2L)
        c_win = self._rsht.fsht(win_dh)                 # (1, L, L) complex64

        # s2m = integral of window function = total solid angle of the window.
        # c_win[0, 0, 0] = (1/sqrt(4π)) * integral(window) = omega_window / sqrt(4π)
        # so s2m = c_win[0,0,0].real * sqrt(4π) = omega_window
        s2m = float(c_win[0, 0, 0].real.item()) * math.sqrt(4.0 * math.pi)

        # ---- 2. Master² SHT (DH grid) ----------------------------------------
        # Synthesise master on the DH grid, square, then forward SHT.
        c_master_full = self._master_coefs.to(device)  # (1, L, L) complex64
        master_dh = self._rsht.isht(c_master_full)     # (1, 2L, 2L) float32
        master_sq_dh = master_dh ** 2                  # (1, 2L, 2L) float32
        c_master_sq = self._rsht.fsht(master_sq_dh)    # (1, L, L) complex64

        # ---- 3. CC(window, master²)(R) and CC(window, master)(R) -------------
        # Both use window as first argument so the denominator is indexed at the
        # same grid point as our numerator CC(pattern, master)(R).
        cc_wm2 = rs2cc_(
            L, c_win, c_master_sq, self._wigner_table,
        ).real                                          # (1, 2L-1, 2L-1, 2L-1) float32

        cc_wm = rs2cc_(
            L, c_win, c_master_full, self._wigner_table,
        ).real                                          # (1, 2L-1, 2L-1, 2L-1) float32

        # ---- 4. Denominator: denom² = CC(window,master²) - CC(window,master)²/s2m
        # This is the variance of the master visible through the window at R.
        # Negative values (numerical artefacts) are clamped to a small ε before
        # taking the sqrt, so rDen is always finite and positive.
        s2m_safe = max(float(s2m), 1e-12)
        denom_sq = cc_wm2 - cc_wm ** 2 / s2m_safe      # (1, 2L-1, 2L-1, 2L-1) float32
        eps = 1e-8
        rDen_vol = (1.0 / denom_sq.clamp_min(eps).sqrt()).float()  # (1, 2L-1, 2L-1, 2L-1)

        return rDen_vol.cpu(), s2m

    def _convert_master_coefs(self, L: int) -> torch.Tensor:
        """Convert SHT-file ``[m, l]`` → RSHT ``(1, L, L)`` complex64."""
        bw = self.master.bandwidth
        if L > bw:
            raise ValueError(f"L={L} > master.bandwidth={bw}")

        m_dense = self.master.coefs_ml.to(self.device)
        m_trunc = m_dense[:L, :L].to(torch.complex64)

        return m_trunc.unsqueeze(0)

    def _build_full_d_lmk_table(self, L: int, device: torch.device) -> torch.Tensor:
        """Expand the compressed wigner table into a dense (L, 2L-1, 2L-1) tensor.

        rs2cc_fast_ requires d_lmk[l, m+L-1, k+L-1] = d^l_{m, k}(pi/2) for
        all (l, m, k) with -(L-1) <= m, k <= (L-1) AND l >= max(|m|, |k|).
        Outside the valid (l, m, k) triangle the Wigner-d is identically zero.

        We MUST zero those entries because read_mn_wigner_d_half_pi_table
        returns garbage for l < max(|m|, |k|) (it indexes into the storage
        region of a higher l).
        """
        from .._math._wigner_logspace import read_mn_wigner_d_half_pi_table
        sl = 2 * L - 1
        d_lmk = torch.zeros(L, sl, sl, dtype=torch.float32, device=device)
        for l in range(L):
            # Only fill the valid (m, k) range for this l: |m|, |k| <= l
            idx_l = torch.arange(-l, l + 1, dtype=torch.int32, device=device)
            m_grid, k_grid = torch.meshgrid(idx_l, idx_l, indexing="ij")
            block = read_mn_wigner_d_half_pi_table(
                self._wigner_table, m_coords=m_grid, n_coords=k_grid, l=l,
            ).float()
            d_lmk[l, L - 1 - l : L + l, L - 1 - l : L + l] = block
        return d_lmk

    def _build_sparse_rs2cc_tables(
        self,
        L: int,
        master_coefs: torch.Tensor,
        device: torch.device,
        zero_tol: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect master-coefs sparsity and build pre-shifted Wigner tables.

        Returns
        -------
        l_active : (La,) LongTensor
            Indices of l where master_coefs[..., l] has any nonzero entry.
            For Ni m-3m at L=68: 34 values (even-l only).
        d_mk_pre : (La, L, 2L-1) complex64
            ``d_full[l_active, L-1:, :]`` after ``ifftshift`` along k axis
            (last). Folds the post-einsum ``ifftshift(spectrum, dim=-2)``.
        d_kn_pre : (La, 2L-1, 2L-1) complex64
            ``d_full[l_active]`` after ``ifftshift`` along (k, n). Folds
            both shifts that ``rs2cc_vectorized`` does after the einsum.

        Sparsity is detected from the master coefs themselves rather than
        hard-coded by point group, so this works correctly for arbitrary
        master files (incl. low-symmetry phases that simply have La == L).
        """
        # Per-l max |coef|. Master shape is (1, L, L) [b, m, l].
        per_l_max = master_coefs.abs().amax(dim=(0, 1))                 # (L,)
        l_active = torch.nonzero(per_l_max > zero_tol, as_tuple=False).flatten().long()
        if l_active.numel() == 0:
            # Master is all zero — keep all l to avoid empty tensors.
            l_active = torch.arange(L, dtype=torch.long, device=device)

        # Slice the dense Wigner-d table at active l rows.
        d_mk_lact = self._d_lmk_full[l_active, L - 1 :, :]              # (La, L, 2L-1)
        d_kn_lact = self._d_lmk_full[l_active]                          # (La, 2L-1, 2L-1)
        # Fold ifftshift on the FFT axes into the d tables.
        d_mk_pre = torch.fft.ifftshift(d_mk_lact, dim=-1)
        d_kn_pre = torch.fft.ifftshift(d_kn_lact, dim=(-2, -1))
        d_mk_pre = d_mk_pre.to(torch.complex64).contiguous()
        d_kn_pre = d_kn_pre.to(torch.complex64).contiguous()
        return l_active.to(device), d_mk_pre, d_kn_pre

    def _open_pattern_dataset(
        self, h5oina_path: str,
    ) -> Tuple[int, h5py.Dataset, str, h5py.File]:
        """Open H5OINA / EDAX H5, locate the patterns dataset."""
        f = h5py.File(h5oina_path, "r")
        # H5OINA (Oxford) candidates first
        candidates = [
            "1/EBSD/Data/Processed Patterns",
            "1/EBSD/Data/Patterns",
            "1/EBSD/Data/Raw Patterns",
        ]
        # EDAX H5 stores patterns as <ScanName>/EBSD/Data/Pattern. Find them
        # by walking the top-level groups and probing standard sub-paths.
        for top_name in list(f.keys()):
            top = f[top_name]
            if not isinstance(top, h5py.Group):
                continue
            for sub in ("EBSD/Data/Pattern", "EBSD/Data/Patterns"):
                full = f"{top_name}/{sub}"
                if full in f and isinstance(f[full], h5py.Dataset):
                    candidates.append(full)
        for ds_path in candidates:
            if ds_path in f and isinstance(f[ds_path], h5py.Dataset):
                dset = f[ds_path]
                if dset.ndim == 3:
                    return int(dset.shape[0]), dset, ds_path, f
        f.close()
        raise FileNotFoundError(
            f"No EBSD pattern dataset in {h5oina_path} (tried: {candidates})"
        )
