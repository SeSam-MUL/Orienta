"""Forward EBSD pattern rendering from SHT spherical-harmonic coefficients.

Skip the Master-H5 step entirely: load `.sht` -> inverse SHT -> DH grid ->
sample at rotated detector pixel directions -> 2D simulated pattern.

See `docs/superpowers/specs/2026-05-10-sht-forward-pattern-simulation-design.md`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .._math.sht_inverse import inverse_sht_to_dh_grid
from .sht_io import read_sht_master

logger = logging.getLogger(__name__)


@dataclass
class LambertGrid:
    """Cached master-pattern signal for one phase.

    Despite the name (kept for spec consistency), the underlying
    representation is a Driscoll-Healy `(2L, 2L)` real grid:
        theta_ti = pi * (2*ti) / (2*L),     ti in [0, 2L)
        phi_pj   = pi * (2*pj + 1) / (4*L), pj in [0, 2L)
    where ``L = bandwidth`` is the *truncated* bandlimit used for
    reconstruction (the file's bandwidth is in ``file_bandwidth``).
    """
    grid: torch.Tensor          # (2*bw, 2*bw) float32 on the renderer's device
    bandwidth: int              # truncated bandwidth used for the grid
    file_bandwidth: int         # original SHT file bandwidth
    sht_path: str
    point_group: str
    voltage_kv: float


class PatternRenderer:
    """Phase-cached forward pattern renderer.

    Phase A implements the single-pattern path; ``render_batch`` is a
    NotImplementedError stub for Phase B.
    """

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def load_phase(
        self,
        sht_path: str,
        max_bandwidth: Optional[int] = None,
    ) -> LambertGrid:
        """Load an SHT file and reconstruct the master pattern grid.

        Parameters
        ----------
        sht_path : str
        max_bandwidth : Optional[int]
            Override the default bandwidth truncation. ``None`` uses the
            default from ``sht_inverse.DEFAULT_MAX_BANDWIDTH`` (128).
            Higher values give sharper / less blurry patterns at the cost
            of GPU memory: bw=256 needs ~1.7 GB, bw=384 needs ~5.7 GB.

        Caching is the responsibility of the caller (service layer).
        Raises ``SHTReadError`` if the file is invalid.
        """
        master = read_sht_master(sht_path, device=self.device)
        kwargs = {}
        if max_bandwidth is not None:
            kwargs["max_bandwidth"] = int(max_bandwidth)
        grid = inverse_sht_to_dh_grid(
            master.coefs_ml,
            bandwidth=master.bandwidth,
            device=self.device,
            **kwargs,
        )
        # The grid shape (2L, 2L) defines the truncated bandwidth used
        truncated_bw = grid.shape[0] // 2
        logger.info(
            "PatternRenderer: loaded %s (file bw=%d, used bw=%d, pg=%s, %.2f kV)",
            sht_path, master.bandwidth, truncated_bw,
            master.point_group, master.voltage_kv,
        )
        return LambertGrid(
            grid=grid,
            bandwidth=truncated_bw,
            file_bandwidth=master.bandwidth,
            sht_path=sht_path,
            point_group=master.point_group,
            voltage_kv=master.voltage_kv,
        )

    def render(
        self,
        grid: LambertGrid,
        orientation_quat: torch.Tensor,
        pc_emsoft: Tuple[float, float, float],
        detector_shape: Tuple[int, int],
        pixel_size_um: float,
        tilt_deg: float = 70.0,
        det_tilt_deg: float = 0.0,
    ) -> torch.Tensor:
        """Render one simulated EBSD pattern.

        Geometry mirrors `backend/spherical_gpu/pipeline/indexer.py
        ::_build_backprojection_map` (the validated SP-GPU indexer
        formula), but inverted to go detector -> sphere instead of
        sphere -> detector. Three fixes vs. the original implementation
        (validated 2026-05-11 against a CI=0.73 Al pixel from the
        7050 dataset, NCC=+0.54):

        1. Combined tilt: alpha = pi/2 - sample_tilt + det_tilt.
           The previous code used `tilt_deg` directly as the rotation
           angle, which is wrong for a 70 deg sample-tilt setup.

        2. Image y orientation: row 0 is the BOTTOM of the pattern when
           the detector is read with rfft/EMSphInx convention. We flip
           the row index internally so the caller can keep "image with
           row 0 at top" semantics.

        3. Grid axis order: in `sht.py::grid_DriscollHealy`, axis 0 is
           the variable named "theta" which is the AZIMUTHAL angle, and
           axis 1 is "phi" which is the POLAR angle (project convention,
           swapped from physics). Sample directions therefore map as:
             azimuth in [0, 2pi) -> axis 0 = height index in [0, 2L)
             polar   in (0, pi)  -> axis 1 = width  index in [-0.5, 2L-0.5)

        4. Orientation application: apply q directly (not q^-1).
           orix's ``Rotation.from_euler(..., direction='lab2crystal')``
           returns a rotation whose ``q*v*q^*`` action takes a lab/sample
           vector to crystal coords. That is exactly what we want here.

        Parameters
        ----------
        grid : LambertGrid
            From ``load_phase``.
        orientation_quat : torch.Tensor
            Shape (4,), dtype float64, ordering (w, x, y, z). Must be
            the orix ``lab2crystal`` quaternion (the default of
            ``Rotation.from_euler``). Identity is ``[1, 0, 0, 0]``.
        pc_emsoft : tuple
            EMsoft pattern center ``(xpc, ypc, L_um)``.
        detector_shape : tuple
            ``(H, W)`` in pixels.
        pixel_size_um : float
            Detector pixel size in micrometres.
        tilt_deg : float
            SAMPLE tilt angle in degrees (typically 70 for EBSD).
        det_tilt_deg : float
            Detector tilt angle in degrees (typically 0; some setups
            have small positive values).

        Returns
        -------
        torch.Tensor
            Shape ``(H, W)``, float32, on CPU.

        Raises
        ------
        ValueError
            On invalid detector_shape or quaternion shape.
        """
        H, W = int(detector_shape[0]), int(detector_shape[1])
        if H <= 0 or W <= 0:
            raise ValueError(f"invalid detector_shape: {detector_shape}")

        if grid.grid.device != self.device:
            grid_t = grid.grid.to(self.device)
        else:
            grid_t = grid.grid

        device = self.device
        delta = float(pixel_size_um)
        xpc, ypc, L_um = (float(v) for v in pc_emsoft)

        # Indexer-formula combined tilt (validated 2026-05-11):
        #     alpha = pi/2 - sigma_sample + theta_detector
        alpha = math.pi / 2.0 - math.radians(float(tilt_deg)) + math.radians(float(det_tilt_deg))
        sA = math.sin(alpha)
        cA = math.cos(alpha)

        # EMSphInx PC convention (indexer line 1857-1862):
        #     cX = -xpc, cY = ypc
        cX = -xpc
        cY = ypc

        # Pixel grid -> fractional (X_frac, Y_frac) in [0, 1].
        # Image row 0 is rendered at the BOTTOM of the pattern
        # (flip vs. naive top-down) so the caller can treat row 0 as
        # the top of the image; we invert internally to match the EMSphInx
        # backprojection formula.
        i_idx = torch.arange(H, dtype=torch.float64, device=device)
        j_idx = torch.arange(W, dtype=torch.float64, device=device)
        ii, jj = torch.meshgrid(i_idx, j_idx, indexing="ij")
        ii_eff = (H - 1) - ii

        X_frac = (jj + 0.5) / W
        Y_frac = (ii_eff + 0.5) / H

        # Invert the indexer projection:
        #   X_frac = (cX + x_um/delta)/W + 0.5  =>  x_um = ((X_frac-0.5)*W - cX)*delta
        x_um = ((X_frac - 0.5) * W - cX) * delta
        y_um = ((Y_frac - 0.5) * H - cY) * delta

        # 3D position of the pixel in SAMPLE frame, using the detector
        # basis derived from the indexer formula:
        #   det_x_axis_in_sample = (0, 1, 0)
        #   det_y_axis_in_sample = (-cA, 0, sA)
        #   det_normal_in_sample = (sA, 0, cA)
        #   P = L*normal + x_um*det_x_axis + y_um*det_y_axis
        Px = L_um * sA - y_um * cA
        Py = x_um.clone()
        Pz = L_um * cA + y_um * sA

        norms = torch.sqrt(Px * Px + Py * Py + Pz * Pz).clamp_min(1e-30)
        nx_s = Px / norms
        ny_s = Py / norms
        nz_s = Pz / norms
        sample_unit = torch.stack([nx_s, ny_s, nz_s], dim=-1)        # (H, W, 3)

        # Apply orientation: q (lab2crystal) * v * q^* maps lab -> crystal.
        # No inversion of q.
        q = orientation_quat.to(device=device, dtype=torch.float64)
        if q.shape != (4,):
            raise ValueError(f"orientation_quat must be (4,), got {tuple(q.shape)}")
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        v = sample_unit
        qxyz = torch.stack([qx, qy, qz])
        cross1 = torch.linalg.cross(qxyz.expand_as(v), v, dim=-1)
        t = 2.0 * cross1
        cross2 = torch.linalg.cross(qxyz.expand_as(v), t, dim=-1)
        crystal = v + qw * t + cross2                                # (H, W, 3)

        cx_, cy_, cz_ = crystal[..., 0], crystal[..., 1], crystal[..., 2]
        polar   = torch.atan2(torch.sqrt(cx_ * cx_ + cy_ * cy_), cz_)  # [0, pi]
        azimuth = torch.atan2(cy_, cx_) % (2.0 * math.pi)              # [0, 2pi)

        # DH grid (axis 0 = azimuth, axis 1 = polar per project sht.py):
        #   azimuth_dh = pi*2*i/(2L)    -> i = azimuth * L / pi    (range [0, 2L))
        #   polar_dh   = pi*(2*j+1)/(4L) -> j = (polar*4L/pi - 1)/2  (range [-0.5, 2L-0.5))
        bw = grid.bandwidth
        n = 2 * bw
        i_az  = (azimuth * bw / math.pi) % n
        j_pol = (polar * 4.0 * bw / math.pi - 1.0) * 0.5

        # grid_sample expects ``(..., 2)`` last dim = (x_norm, y_norm) where
        # y is height (axis 0) and x is width (axis 1).
        y_norm = (i_az  / max(n - 1, 1)) * 2.0 - 1.0
        y_norm = y_norm.clamp(-1.0, 1.0)
        x_norm = (j_pol / max(n - 1, 1)) * 2.0 - 1.0
        x_norm = x_norm.clamp(-1.0, 1.0)

        sample_grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        signal = grid_t.unsqueeze(0).unsqueeze(0).to(torch.float32)         # (1, 1, n, n)
        out = torch.nn.functional.grid_sample(
            signal,
            sample_grid.to(torch.float32),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )                                                                   # (1, 1, H, W)
        return out.squeeze(0).squeeze(0).contiguous().cpu()

    def render_batch(
        self,
        grid: LambertGrid,
        orientation_quats: torch.Tensor,
        pc_emsoft: Tuple[float, float, float],
        detector_shape: Tuple[int, int],
        pixel_size_um: float,
        tilt_deg: float = 70.0,
        det_tilt_deg: float = 0.0,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """Render N simulated patterns at N orientations.

        All patterns share the same PC + detector geometry. Quaternions
        are processed in chunks to keep GPU memory bounded.

        Uses the same conventions as ``render()`` (validated FEAT-SHT-FWD-A
        fix on 2026-05-11):
          - alpha = pi/2 - sample_tilt + det_tilt
          - cX = -xpc, cY = ypc
          - image y is internally flipped (row 0 = top in caller view)
          - apply q directly (no inversion; q is lab2crystal)
          - DH grid axis 0 = azimuth, axis 1 = polar

        Parameters
        ----------
        grid : LambertGrid
        orientation_quats : torch.Tensor
            Shape ``(N, 4)``, dtype float64. Each row is a unit
            quaternion (w, x, y, z).
        pc_emsoft : tuple
            EMsoft pattern center ``(xpc, ypc, L_um)``. Constant across
            all N orientations (single-PC assumption).
        detector_shape : (H, W)
        pixel_size_um : float
        tilt_deg : float
            SAMPLE tilt in degrees.
        det_tilt_deg : float
            Detector tilt in degrees.
        chunk_size : int
            Number of patterns per GPU launch. Lower if you hit OOM.

        Returns
        -------
        torch.Tensor
            Shape ``(N, H, W)``, dtype float32, on CPU.
        """
        H, W = int(detector_shape[0]), int(detector_shape[1])
        if H <= 0 or W <= 0:
            raise ValueError(f"invalid detector_shape: {detector_shape}")
        if orientation_quats.ndim != 2 or orientation_quats.shape[1] != 4:
            raise ValueError(
                f"orientation_quats must have shape (N, 4); got {tuple(orientation_quats.shape)}"
            )

        N = int(orientation_quats.shape[0])
        device = self.device
        delta = float(pixel_size_um)
        xpc, ypc, L_um = (float(v) for v in pc_emsoft)

        alpha = math.pi / 2.0 - math.radians(float(tilt_deg)) + math.radians(float(det_tilt_deg))
        sA = math.sin(alpha)
        cA = math.cos(alpha)
        cX = -xpc
        cY = ypc

        # Precompute the per-pixel sample-frame direction once (constant
        # across all N orientations because PC + detector + tilt are fixed).
        i_idx = torch.arange(H, dtype=torch.float64, device=device)
        j_idx = torch.arange(W, dtype=torch.float64, device=device)
        ii, jj = torch.meshgrid(i_idx, j_idx, indexing="ij")
        ii_eff = (H - 1) - ii

        X_frac = (jj + 0.5) / W
        Y_frac = (ii_eff + 0.5) / H
        x_um = ((X_frac - 0.5) * W - cX) * delta
        y_um = ((Y_frac - 0.5) * H - cY) * delta

        Px = L_um * sA - y_um * cA
        Py = x_um.clone()
        Pz = L_um * cA + y_um * sA
        norms = torch.sqrt(Px * Px + Py * Py + Pz * Pz).clamp_min(1e-30)
        sample_unit = torch.stack([Px / norms, Py / norms, Pz / norms], dim=-1)  # (H, W, 3)
        sample_unit_flat = sample_unit.reshape(-1, 3)                            # (H*W, 3)

        # Grid signal expanded for batch grid_sample (1, 1, n, n)
        if grid.grid.device != device:
            grid_t = grid.grid.to(device)
        else:
            grid_t = grid.grid
        signal = grid_t.unsqueeze(0).unsqueeze(0).to(torch.float32)              # (1, 1, n, n)
        bw = grid.bandwidth
        n_grid = 2 * bw
        n_pixels = H * W

        # Output buffer
        out_cpu = torch.empty((N, H, W), dtype=torch.float32)

        quats_all = orientation_quats.to(device=device, dtype=torch.float64)

        # Process in chunks
        for start in range(0, N, int(chunk_size)):
            end = min(start + int(chunk_size), N)
            B = end - start
            q_chunk = quats_all[start:end]                                       # (B, 4)

            # Rotate sample directions by each quaternion in the chunk.
            # Hamilton "v' = v + 2*qw*(q.xyz x v) + 2*(q.xyz x (q.xyz x v))"
            # applied per-quat in the batch dimension.
            qw = q_chunk[:, 0:1]                                                 # (B, 1)
            qxyz = q_chunk[:, 1:4]                                               # (B, 3)

            # v: (1, P, 3); qxyz: (B, 1, 3)
            v = sample_unit_flat.unsqueeze(0)                                    # (1, P, 3)
            qxyz_b = qxyz.unsqueeze(1)                                           # (B, 1, 3)
            cross1 = torch.linalg.cross(qxyz_b.expand(B, n_pixels, 3),
                                        v.expand(B, n_pixels, 3), dim=-1)
            t = 2.0 * cross1                                                     # (B, P, 3)
            cross2 = torch.linalg.cross(qxyz_b.expand(B, n_pixels, 3), t, dim=-1)
            crystal = v.expand(B, n_pixels, 3) + qw.unsqueeze(-1) * t + cross2   # (B, P, 3)

            cx_ = crystal[..., 0]; cy_ = crystal[..., 1]; cz_ = crystal[..., 2]
            polar = torch.atan2(torch.sqrt(cx_ * cx_ + cy_ * cy_), cz_)          # (B, P)
            azimuth = torch.atan2(cy_, cx_) % (2.0 * math.pi)

            i_az = (azimuth * bw / math.pi) % n_grid                             # (B, P)
            j_pol = (polar * 4.0 * bw / math.pi - 1.0) * 0.5

            y_norm = (i_az / max(n_grid - 1, 1)) * 2.0 - 1.0
            y_norm = y_norm.clamp(-1.0, 1.0)
            x_norm = (j_pol / max(n_grid - 1, 1)) * 2.0 - 1.0
            x_norm = x_norm.clamp(-1.0, 1.0)

            # grid_sample wants shape (B, H_out, W_out, 2). Reshape (B, P) -> (B, H, W).
            y_norm = y_norm.reshape(B, H, W)
            x_norm = x_norm.reshape(B, H, W)
            sample_grid = torch.stack([x_norm, y_norm], dim=-1).to(torch.float32)  # (B, H, W, 2)

            signal_b = signal.expand(B, 1, n_grid, n_grid)
            chunk_out = torch.nn.functional.grid_sample(
                signal_b, sample_grid,
                mode="bilinear", padding_mode="border", align_corners=True,
            )                                                                    # (B, 1, H, W)
            out_cpu[start:end] = chunk_out.squeeze(1).contiguous().cpu()

        return out_cpu
